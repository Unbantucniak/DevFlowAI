export type PullRequestStatus = {
  state: string;
  merged_at?: string | null;
  mergeable?: boolean | null;
  author?: string | null;
};

export function classifyPr(pr: PullRequestStatus) {
  // Terminal states win even if older mergeability information says otherwise.
  if (pr.merged_at) return "done";
  if (pr.state === "closed") return "closed";
  // A code change mentioning "conflict" is not evidence of a merge conflict.
  if (pr.mergeable === false) return "conflict";
  if (!pr.author) return "needsOwner";
  return "reviewing";
}
