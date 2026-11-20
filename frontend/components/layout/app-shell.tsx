"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "工作台" },
  { href: "/knowledge", label: "项目知识" },
  { href: "/rag", label: "RAG 工作室" },
  { href: "/repos", label: "仓库接入" },
  { href: "/action-drafts", label: "操作草稿" },
  { href: "/workflow-runs", label: "运行记录" },
  { href: "/workspaces", label: "团队周报" },
  { href: "/code-graph", label: "代码图谱" },
  { href: "/evals", label: "质量评测" }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isStudio = pathname === "/rag" || pathname.startsWith("/knowledge/");
  return (
    <>
      <header className="site-header">
        <Link href="/" className="site-brand" aria-label="DevFlow AI 工作台">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>DevFlow <span className="text-accent">AI</span></span>
        </Link>
        <nav aria-label="主导航">
          {navItems.map(({ href, label }) => {
            const active = pathname === href || (href !== "/" && pathname.startsWith(href + "/"));
            return <Link key={href} href={href} aria-current={active ? "page" : undefined}>{label}</Link>;
          })}
        </nav>
      </header>
      <main className={pathname === "/" ? "workspace-page" : isStudio ? "studio-page" : "content-page"}>{children}</main>
    </>
  );
}
