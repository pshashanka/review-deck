// Mirrors agent/src/review/state.py (ReviewState) and agent/src/review/models.py (Finding).

export type Severity = "high" | "medium" | "low";
export type FindingCategory = "security" | "tests" | "correctness";
export type FindingStatus =
  | "open"
  | "approved"
  | "rejected"
  | "applied"
  | "dismissed";

export type Finding = {
  id: string;
  category: FindingCategory;
  severity: Severity;
  title: string;
  explanation: string;
  file: string;
  line: number;
  suggestion?: string | null;
  status: FindingStatus;
  comment_url?: string;
  error?: string;
};

export type PullRequestInfo = {
  owner: string;
  repo: string;
  number: number;
  title: string;
  author: string;
  url: string;
  head_sha?: string;
  base_ref?: string;
  head_ref?: string;
  truncated?: boolean;
};

export type NodeStatusValue = "running" | "done" | "error" | "timeout";

export type NodeUsage = {
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  cost_usd: number | null;
  files?: number;
};

export type ReviewStats = {
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  cost_usd: number | null;
  skipped_files: string[];
  partial?: boolean;
  files_total: number;
};

export type ReviewState = {
  pr: PullRequestInfo | null;
  // Present in the graph state; large (annotated diffs), rarely needed by the UI.
  files?: { filename: string; status: string }[];
  findings: Finding[];
  summary: string | null;
  node_status: Record<string, NodeStatusValue>;
  node_models?: Record<string, string>;
  node_usage?: Record<string, NodeUsage>;
  review_stats?: ReviewStats | null;
  review_errors: string[];
  error: string | null;
};
