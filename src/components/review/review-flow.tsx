"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./review-flow.css";
import { useAgent, UseAgentUpdate } from "@copilotkit/react-core/v2";
import { useTheme } from "@/hooks/use-theme";
import type { ReviewState, Severity } from "@/lib/types";

type NodeStatus =
  | "idle"
  | "running"
  | "done"
  | "error"
  | "interrupted"
  | "timeout";
type FlowNodeData = {
  label: string;
  status: NodeStatus;
  minor?: boolean;
  /** Elapsed (running) or final (done) duration, e.g. "12s". */
  time?: string;
  slow?: boolean;
  model?: string;
  modelTitle?: string;
};
type FlowNode = Node<FlowNodeData, "step">;

const LABELS: Record<string, string> = {
  input: "Input",
  fetch_pr: "Fetch PR",
  lint_review: "Lint",
  security_review: "Security",
  tests_review: "Tests",
  regression_review: "Regression",
  approval_gate: "Approval gate",
  summarize: "Summarize",
  render_findings: "Render findings",
  chat: "Chat",
  handle_action: "Handle action",
  fetch_error: "Fetch error",
};

const POSITIONS: Record<string, { x: number; y: number }> = {
  input: { x: 170, y: 0 },
  chat: { x: 380, y: 0 },
  handle_action: { x: 540, y: 60 },
  fetch_pr: { x: 170, y: 90 },
  fetch_error: { x: 380, y: 130 },
  lint_review: { x: -70, y: 190 },
  security_review: { x: 90, y: 190 },
  tests_review: { x: 250, y: 190 },
  regression_review: { x: 410, y: 190 },
  approval_gate: { x: 170, y: 290 },
  summarize: { x: 170, y: 380 },
  render_findings: { x: 170, y: 470 },
};

const EDGE_PAIRS: [string, string][] = [
  ["input", "chat"],
  ["input", "handle_action"],
  ["input", "fetch_pr"],
  ["handle_action", "render_findings"],
  ["fetch_pr", "fetch_error"],
  ["fetch_pr", "lint_review"],
  ["fetch_pr", "security_review"],
  ["fetch_pr", "tests_review"],
  ["fetch_pr", "regression_review"],
  ["lint_review", "approval_gate"],
  ["security_review", "approval_gate"],
  ["tests_review", "approval_gate"],
  ["regression_review", "approval_gate"],
  ["approval_gate", "summarize"],
  ["summarize", "render_findings"],
];

// A running node past this many seconds gets a "taking longer than usual" hint.
const SLOW_THRESHOLD_S = 30;

const LLM_NODES = new Set([
  "lint_review",
  "security_review",
  "tests_review",
  "regression_review",
  "summarize",
  "chat",
]);

const REVIEWER_NAMES: Record<string, string> = {
  lint_review: "Lint reviewer",
  security_review: "Security reviewer",
  tests_review: "Tests reviewer",
  regression_review: "Regression reviewer",
};

const shortModel = (spec: string) => {
  const i = spec.indexOf(":");
  return i >= 0 ? spec.slice(i + 1) : spec;
};

const MINOR = new Set(["chat", "handle_action", "fetch_error"]);

// render_tools is the ToolNode that emits the A2UI surface; show as render_findings.
const normalizeStep = (name: string) =>
  name === "render_tools" ? "render_findings" : name;

function StepNode({ data }: NodeProps<FlowNode>) {
  return (
    <div
      className="review-node"
      data-status={data.status}
      data-minor={data.minor ? "true" : undefined}
    >
      <Handle type="target" position={Position.Top} className="review-handle" />
      {data.label}
      {data.model && (
        <div className="review-node-model" title={data.modelTitle}>
          {data.model}
        </div>
      )}
      {data.status === "timeout" && (
        <div className="review-node-note review-node-timeout-note">
          &#9201; timed out
        </div>
      )}
      {data.status === "running" && data.time && (
        <div className="review-node-note">{data.time}</div>
      )}
      {data.status === "running" && data.slow && (
        <div className="review-node-note review-node-slow">
          taking longer than usual&hellip;
        </div>
      )}
      {data.status === "done" && data.time && (
        <div className="review-node-note">{data.time}</div>
      )}
      <Handle
        type="source"
        position={Position.Bottom}
        className="review-handle"
      />
    </div>
  );
}

const nodeTypes = { step: StepNode };

const SEVERITY_STYLE: Record<Severity, string> = {
  high: "#ef4444",
  medium: "#f59e0b",
  low: "#3b82f6",
};

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

export function ReviewFlow() {
  const { agent } = useAgent({
    updates: [UseAgentUpdate.OnStateChanged, UseAgentUpdate.OnRunStatusChanged],
  });

  const { theme } = useTheme();
  const [liveRunning, setLiveRunning] = useState<Set<string>>(new Set());
  const [liveDone, setLiveDone] = useState<Set<string>>(new Set());
  const [interrupted, setInterrupted] = useState(false);
  // Timing is derived from node_status transitions observed in this session
  // (STEP events fire in bursts for parallel nodes, so they are never used
  // to start/clear timers). startRef: earliest time seen "running";
  // durations: seconds from that start to the latest terminal sighting.
  const startRef = useRef<Map<string, number>>(new Map());
  const [durations, setDurations] = useState<Record<string, number>>({});
  const [now, setNow] = useState(() => Date.now());
  // node_status object from before the current run started; its statuses are
  // stale and ignored until a new snapshot (new object identity) arrives.
  const [baseline, setBaseline] = useState<object | null>(null);

  const clearTiming = () => {
    startRef.current.clear();
    setDurations({});
  };

  // Clear live state when switching threads.
  useEffect(() => {
    setLiveRunning(new Set());
    setLiveDone(new Set());
    setInterrupted(false);
    setBaseline(null);
    clearTiming();
  }, [agent.threadId]);

  useEffect(() => {
    const { unsubscribe } = agent.subscribe({
      onRunStartedEvent: () => {
        setLiveRunning(new Set());
        setLiveDone(new Set());
        setInterrupted(false);
        clearTiming();
        const ns = isRecord(agent.state) ? agent.state.node_status : null;
        setBaseline(isRecord(ns) ? ns : null);
      },
      // Secondary hint only; never touches timers.
      onStepStartedEvent: ({ event }) => {
        const s = normalizeStep(event.stepName);
        setLiveRunning((p) => new Set(p).add(s));
      },
      onStepFinishedEvent: ({ event }) => {
        const s = normalizeStep(event.stepName);
        setLiveRunning((p) => {
          const n = new Set(p);
          n.delete(s);
          return n;
        });
        setLiveDone((p) => new Set(p).add(s));
      },
      onCustomEvent: ({ event }) => {
        if (event.name === "on_interrupt") setInterrupted(true);
      },
      onRunFinishedEvent: (params) => {
        if (params.outcome === "interrupt") setInterrupted(true);
        setLiveRunning(new Set());
      },
      onRunErrorEvent: () => setLiveRunning(new Set()),
    });
    return () => unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent]);

  const state: Partial<ReviewState> = isRecord(agent.state) ? agent.state : {};
  const rawNodeStatus = isRecord(state.node_status) ? state.node_status : null;
  const nodeStatus: Record<string, string> =
    rawNodeStatus && rawNodeStatus !== baseline ? rawNodeStatus : {};
  const nodeModels: Record<string, string> = isRecord(state.node_models)
    ? (state.node_models as Record<string, string>)
    : {};
  const pr = isRecord(state.pr) ? state.pr : null;
  const findings = Array.isArray(state.findings) ? state.findings : [];
  const errorText =
    typeof state.error === "string" && state.error ? state.error : null;
  const isRunning = agent.isRunning;

  const statusOf = (id: string): NodeStatus => {
    if (id === "input") return "idle";
    const s = nodeStatus[id];
    if (s === "error") return "error";
    if (s === "timeout") return "timeout";
    if (id === "approval_gate" && interrupted && !isRunning) return "interrupted";
    // node_status is the source of truth; a persisted "running" with no
    // active run is stale (paused/reloaded). STEP events only fill gaps.
    if (s === "running") return isRunning ? "running" : "idle";
    if (s === "done") return "done";
    if (s === undefined && isRunning) {
      if (liveRunning.has(id)) return "running";
      if (liveDone.has(id)) return "done";
    }
    return "idle";
  };

  const statuses: Record<string, NodeStatus> = {};
  for (const id of Object.keys(LABELS)) statuses[id] = statusOf(id);
  const statusKey = Object.keys(LABELS)
    .map((id) => statuses[id])
    .join(",");

  // Record start/finish times on node_status transitions.
  const nsKey = Object.keys(LABELS)
    .map((id) => `${id}=${nodeStatus[id] ?? ""}`)
    .join(",");
  useEffect(() => {
    if (!isRunning) return;
    const t = Date.now();
    const finished: Record<string, number> = {};
    for (const id of Object.keys(LABELS)) {
      const s = nodeStatus[id];
      if (s === "running") {
        if (!startRef.current.has(id)) startRef.current.set(id, t);
      } else if (s === "done" || s === "timeout" || s === "error") {
        const start = startRef.current.get(id);
        if (start !== undefined) {
          finished[id] = Math.max(0, Math.round((t - start) / 1000));
        }
      }
    }
    setNow(t);
    if (Object.keys(finished).length)
      setDurations((p) => ({ ...p, ...finished }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nsKey, isRunning]);

  // Single 1s interval while the run is active and any node is running.
  const anyRunning = isRunning && statusKey.split(",").includes("running");
  useEffect(() => {
    if (!anyRunning) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [anyRunning]);

  // Per-node display info (timer text, slow flag, model label).
  const info: Record<string, Partial<FlowNodeData>> = {};
  const elapsedOf: Record<string, number> = {};
  for (const id of Object.keys(LABELS)) {
    const d: Partial<FlowNodeData> = {};
    const spec = nodeModels[id];
    if (LLM_NODES.has(id) && typeof spec === "string" && spec) {
      d.model = shortModel(spec);
      d.modelTitle = spec;
    }
    if (statuses[id] === "running") {
      const start = startRef.current.get(id);
      if (start !== undefined) {
        const secs = Math.max(0, Math.floor((now - start) / 1000));
        elapsedOf[id] = secs;
        d.time = `${secs}s`;
        d.slow = secs >= SLOW_THRESHOLD_S;
      }
    } else if (statuses[id] === "done" && durations[id] !== undefined) {
      d.time = `${durations[id]}s`;
    }
    info[id] = d;
  }
  const infoKey = Object.keys(LABELS)
    .map((id) => {
      const d = info[id];
      return `${d.time ?? ""}|${d.slow ? 1 : 0}|${d.modelTitle ?? ""}`;
    })
    .join(",");

  const nodes: FlowNode[] = useMemo(
    () =>
      Object.keys(LABELS).map((id) => ({
        id,
        type: "step" as const,
        position: POSITIONS[id],
        data: {
          label: LABELS[id],
          status: statuses[id],
          minor: MINOR.has(id),
          ...info[id],
        },
        draggable: false,
        selectable: false,
        connectable: false,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [statusKey, infoKey],
  );

  const edges: Edge[] = useMemo(() => {
    const st = statuses;
    return EDGE_PAIRS.map(([source, target]) => {
      const active = st[target] === "running";
      return {
        id: `${source}-${target}`,
        source,
        target,
        type: "smoothstep",
        animated: active,
        className: active ? "review-edge-running" : undefined,
        selectable: false,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusKey]);

  const counts: Record<Severity, number> = { high: 0, medium: 0, low: 0 };
  for (const f of findings) {
    if (!isRecord(f) || f.status === "rejected") continue;
    if (f.severity in counts) counts[f.severity]++;
  }

  const runningLabels = Object.keys(LABELS)
    .filter((id) => statuses[id] === "running")
    .map((id) => LABELS[id]);

  const slowId = Object.keys(elapsedOf).find(
    (id) => elapsedOf[id] >= SLOW_THRESHOLD_S,
  );
  const timedOut = Object.keys(LABELS)
    .filter((id) => statuses[id] === "timeout")
    .map((id) => REVIEWER_NAMES[id] ?? LABELS[id]);

  let statusLine = "Idle";
  if (errorText) statusLine = `Error: ${errorText}`;
  else if (Object.values(statuses).includes("error")) statusLine = "Error";
  else if (interrupted && !isRunning) statusLine = "Waiting for approval";
  else if (isRunning && slowId)
    statusLine = `${REVIEWER_NAMES[slowId] ?? LABELS[slowId]} is taking longer than usual (${elapsedOf[slowId]}s)…`;
  else if (isRunning)
    statusLine = runningLabels.length
      ? `Running: ${runningLabels.join(", ")}`
      : "Running";
  else if (nodeStatus.render_findings === "done") statusLine = timedOut.length
      ? `Done — ${timedOut.join(", ")} timed out`
      : "Done";

  const prUrl = pr?.url ?? "";
  const safeUrl = /^https?:\/\//i.test(prUrl) ? prUrl : undefined;
  const prTitle = pr?.title || "Untitled PR";

  return (
    <div className="h-full flex flex-col bg-[var(--background)] text-[var(--foreground)]">
      <div className="flex-1 min-h-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          proOptions={{ hideAttribution: true }}
          colorMode={theme}
        >
          <Background />
        </ReactFlow>
      </div>
      <div className="shrink-0 border-t border-[var(--border)] p-4 space-y-3 text-sm">
        {pr && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] text-[var(--card-foreground)] p-3">
            {safeUrl ? (
              <a
                href={safeUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="font-semibold hover:underline"
              >
                {prTitle}
              </a>
            ) : (
              <span className="font-semibold">{prTitle}</span>
            )}
            <div className="text-xs text-[var(--muted-foreground)] mt-1">
              {pr.owner}/{pr.repo}#{pr.number}
              {pr.author && <> &middot; {pr.author}</>}
            </div>
          </div>
        )}
        <div className="flex gap-2 flex-wrap">
          {(Object.keys(counts) as Severity[]).map((sev) => (
            <span
              key={sev}
              className="rounded-full border px-2.5 py-0.5 text-xs font-medium"
              style={{ borderColor: SEVERITY_STYLE[sev], color: SEVERITY_STYLE[sev] }}
            >
              {sev}: {counts[sev]}
            </span>
          ))}
        </div>
        <div className="text-xs text-[var(--muted-foreground)]">{statusLine}</div>
      </div>
    </div>
  );
}
