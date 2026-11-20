"""Startup checks and local backups for the established Windows install."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
IDENTITY = ROOT / ".data/runtime/data-identity.json"
VOLUMES = tuple(f"devflow-ai_devflow_{name}_data" for name in ("postgres", "etcd", "minio", "milvus"))


def docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise RuntimeError(f"Docker command failed ({' '.join(args[:2])}). Check Docker Desktop and .data/logs.")
    return result.stdout.strip()


def settings():
    os.chdir(BACKEND)
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from app.core.config import settings as configured
    return configured


def configuration() -> dict:
    from sqlalchemy.engine import make_url
    cfg = settings()
    url = make_url(cfg.database_url)
    if url.get_backend_name() != "postgresql" or not cfg.milvus_enabled:
        raise RuntimeError("Full project requires PostgreSQL and Milvus; demo configuration was detected.")
    if cfg.embedding_provider not in {"qwen_local", "local"}:
        raise RuntimeError("This GPU launcher requires the configured local embedding provider.")
    return {
        "database": {"driver": url.drivername, "host": url.host, "port": url.port or 5432, "name": url.database, "user": url.username},
        "milvus": {"uri": cfg.milvus_uri, "collection": cfg.milvus_collection},
        "embedding": {"provider": cfg.embedding_provider, "model": cfg.embedding_model, "dimensions": cfg.embedding_dimensions},
        "uploads": str(Path(cfg.upload_dir).resolve()),
    }


def volumes() -> dict:
    rows = json.loads(docker("volume", "inspect", *VOLUMES))
    return {row["Name"]: row["CreatedAt"] for row in rows}


def database_state() -> dict:
    from sqlalchemy import create_engine, text
    engine = create_engine(settings().database_url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as connection:
            cluster = str(connection.execute(text("SELECT system_identifier FROM pg_control_system()")).scalar_one())
            counts = {table: connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
                      for table in ("repositories", "documents", "conversations", "chat_messages")}
        return {"cluster_id": cluster, "counts": counts}
    finally:
        engine.dispose()


def load_identity() -> dict:
    if not IDENTITY.is_file():
        raise RuntimeError("Data identity file is missing. Restore .data/runtime/data-identity.json from backup; do not initialize a new database.")
    return json.loads(IDENTITY.read_text(encoding="utf-8-sig"))


def verify_storage() -> dict:
    expected = load_identity()
    if docker("context", "show") != expected["docker_context"]:
        raise RuntimeError("Docker context changed. Select the original Docker Desktop context before starting.")
    if volumes() != expected["volumes"]:
        raise RuntimeError("Original Docker data volumes are missing or were replaced. Startup stopped to avoid opening an empty database.")
    if configuration() != expected["configuration"]:
        raise RuntimeError("Database, upload path or embedding contract changed. Check .env against the saved configuration before starting.")
    return expected


def verify_data() -> dict:
    expected = verify_storage()
    actual = database_state()
    if actual["cluster_id"] != expected["database_cluster_id"]:
        raise RuntimeError("PostgreSQL cluster identity changed. Refusing to use a different database.")
    return actual


def check_gpu() -> dict:
    import torch
    cfg = settings()
    if cfg.local_embedding_device != "cuda":
        raise RuntimeError("Set LOCAL_EMBEDDING_DEVICE=cuda in .env so GPU failures cannot silently use CPU.")
    if not cfg.local_embedding_preload:
        raise RuntimeError("Set LOCAL_EMBEDDING_PRELOAD=true in .env to warm the model before reporting ready.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch in backend/.venv; see the usage guide.")
    probe = torch.ones((8, 8), device="cuda")
    if float((probe @ probe).sum()) != 512.0:
        raise RuntimeError("CUDA computation check failed.")
    torch.cuda.synchronize()
    return {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def backup(reason: str) -> dict:
    state = verify_data()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder = ROOT / ".data/backups" / f"{stamp}_{reason}"
    folder.mkdir(parents=True, exist_ok=False)
    database = load_identity()["configuration"]["database"]
    partial = folder / "database.dump.partial"
    with partial.open("xb") as output:
        result = subprocess.run(
            ["docker", "exec", "devflow-ai-postgres", "pg_dump", "-U", database["user"], "-d", database["name"], "-Fc"],
            stdout=output, stderr=subprocess.PIPE, timeout=120,
        )
    if result.returncode or partial.stat().st_size < 100:
        raise RuntimeError(f"Database backup failed; incomplete output retained in {folder}.")
    # Do not report a completed backup or stop PostgreSQL until its dump can be read.
    with partial.open("rb") as input_file:
        check = subprocess.run(
            ["docker", "exec", "-i", "devflow-ai-postgres", "pg_restore", "--list"],
            stdin=input_file, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60,
        )
    if check.returncode:
        raise RuntimeError(f"Database backup integrity check failed; PostgreSQL will remain running. See {folder}.")
    partial.rename(folder / "database.dump")
    uploads = Path(load_identity()["configuration"]["uploads"])
    if not uploads.is_dir():
        raise RuntimeError(f"Upload directory is missing: {uploads}. Refusing to mark the backup complete.")
    count = 0
    with ZipFile(folder / "uploads.zip", "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(uploads.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(uploads))
                count += 1
    with ZipFile(folder / "uploads.zip", "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Upload backup integrity check failed; PostgreSQL will remain running. See {folder}.")
    shutil.copy2(ROOT / ".env", folder / "project.env")
    shutil.copy2(IDENTITY, folder / IDENTITY.name)
    summary = {"created_at": datetime.now().astimezone().isoformat(), "reason": reason, **state, "upload_files": count,
               "database_dump_verified": True, "upload_archive_verified": True,
               "note": "PostgreSQL, uploads and config backup. Milvus stays in its named volumes; vectors can be rebuilt from documents."}
    (folder / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"backup": str(folder), "upload_files": count, "counts": state["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record-identity", "preflight", "verify-data", "backup", "status"))
    parser.add_argument("--reason", default="manual", choices=("start", "stop", "manual"))
    args = parser.parse_args()
    if args.command == "record-identity":
        # Explicit and non-overwriting: normal startup can never create a new identity.
        state = database_state()
        if not state["counts"]["repositories"]:
            raise RuntimeError("No existing repositories found. Refusing to register an empty installation.")
        payload = {"docker_context": docker("context", "show"), "volumes": volumes(), "configuration": configuration(),
                   "database_cluster_id": state["cluster_id"], "recorded_at": datetime.now().astimezone().isoformat()}
        IDENTITY.parent.mkdir(parents=True, exist_ok=True)
        with IDENTITY.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
        result = {"identity_recorded": True, **state}
    elif args.command == "preflight":
        verify_storage()
        result = {"storage": "matched", **check_gpu()}
    elif args.command == "verify-data":
        result = verify_data()
    elif args.command == "backup":
        result = backup(args.reason)
    else:
        result = {"data": verify_data(), "embedding": check_gpu()}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
