/**
 * A2UI Catalog — React Renderers
 *
 * Each renderer maps a component name from definitions.ts to a React
 * implementation. Props are type-checked against the Zod schemas.
 *
 * To add a component: define its schema in definitions.ts, then add a
 * renderer here. See README.md "Adding a custom component" for details.
 *
 * The assembled catalog is registered in layout.tsx via
 * <CopilotKit a2ui={{ catalog: demonstrationCatalog }}>.
 */
"use client";

import React, { useEffect, useState } from "react";
import { createCatalog } from "@copilotkit/a2ui-renderer";
import type { CatalogRenderers } from "@copilotkit/a2ui-renderer";
import { demonstrationCatalogDefinitions } from "./definitions";
import type { DemonstrationCatalogDefinitions } from "./definitions";

// ─── Theme-aware colors ─────────────────────────────────────────────

const c = {
  card: "var(--card)",
  cardFg: "var(--card-foreground)",
  border: "var(--border)",
  muted: "var(--muted-foreground)",
  divider: "color-mix(in srgb, var(--border) 50%, var(--card))",
  shadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  btnBg: "color-mix(in srgb, var(--muted) 40%, var(--card))",
  btnDoneBg: "color-mix(in srgb, #22c55e 10%, var(--card))",
};

function ActionButton({
  label,
  doneLabel,
  action,
  children: child,
}: {
  label: string;
  doneLabel: string;
  action: any;
  children?: React.ReactNode;
}) {
  const [done, setDone] = useState(false);
  return (
    <button
      disabled={done}
      style={{
        width: "100%",
        padding: "10px 16px",
        borderRadius: "10px",
        border: done ? "1px solid #bbf7d0" : `1px solid ${c.border}`,
        background: done ? c.btnDoneBg : c.btnBg,
        color: done ? "#059669" : c.cardFg,
        fontSize: "0.85rem",
        fontWeight: 500,
        cursor: done ? "default" : "pointer",
        transition: "all 0.2s ease",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "6px",
      }}
      onClick={() => {
        if (!done) {
          action?.();
          setDone(true);
        }
      }}
    >
      {done && (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#059669"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      )}
      {done ? doneLabel : (child ?? label)}
    </button>
  );
}

// ─── Review components ──────────────────────────────────────────────

const SEVERITY_STYLE: Record<
  string,
  { bg: string; color: string; stripe: string }
> = {
  high: { bg: "#fee2e2", color: "#991b1b", stripe: "#ef4444" },
  medium: { bg: "#fef3c7", color: "#92400e", stripe: "#f59e0b" },
  low: { bg: "#dbeafe", color: "#1e40af", stripe: "#3b82f6" },
};

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  open: { bg: "#dbeafe", color: "#1e40af" },
  approved: { bg: "#fef3c7", color: "#92400e" },
  applied: { bg: "#dcfce7", color: "#166534" },
  dismissed: { bg: "var(--muted)", color: "var(--muted-foreground)" },
};

const pillStyle = (bg: string, color: string): React.CSSProperties => ({
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: "9999px",
  fontSize: "0.7rem",
  fontWeight: 500,
  background: bg,
  color,
});

const MONO =
  "var(--font-mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace)";

// Long enough to cover a slow GitHub call so it cannot be double-submitted
// (handle_action is idempotent as the backstop).
const PENDING_TIMEOUT_MS = 20000;

function ReviewButton({
  label,
  action,
  disabled,
  primary,
  resetKey,
}: {
  label: string;
  action: any;
  disabled: boolean;
  primary?: boolean;
  /** Pending state applies only to the key it was set for (finding id + status). */
  resetKey: string;
}) {
  const [pendingFor, setPendingFor] = useState<string | null>(null);
  // Pending guard against double-clicks; clears when the finding, its status
  // or its error changes, so a failed action or index-reused card is not stuck.
  const pending = pendingFor === resetKey;
  useEffect(() => {
    if (pendingFor === null) return;
    const t = setTimeout(() => setPendingFor(null), PENDING_TIMEOUT_MS);
    return () => clearTimeout(t);
  }, [pendingFor]);
  const off = disabled || pending;
  return (
    <button
      disabled={off}
      onClick={() => {
        if (!off) {
          action?.();
          setPendingFor(resetKey);
        }
      }}
      style={{
        padding: "6px 14px",
        borderRadius: "8px",
        border: `1px solid ${c.border}`,
        background: primary && !off ? "var(--primary)" : c.btnBg,
        color: primary && !off ? "var(--primary-foreground)" : c.cardFg,
        fontSize: "0.8rem",
        fontWeight: 500,
        cursor: off ? "default" : "pointer",
        opacity: off ? 0.5 : 1,
        transition: "all 0.2s ease",
      }}
    >
      {label}
    </button>
  );
}

/** Shared finding visuals: severity stripe, pills, title, file:line, explanation, optional fix. */
function FindingBody({
  props,
  statusLabel,
  statusStyle,
  suggestion,
  children,
}: {
  props: Record<string, any>;
  statusLabel: string;
  statusStyle: { bg: string; color: string };
  suggestion?: string;
  children?: React.ReactNode;
}) {
  const sev = SEVERITY_STYLE[props.severity] ?? SEVERITY_STYLE.low;
  return (
    <div
      style={{
        border: `1px solid ${c.border}`,
        borderLeft: `4px solid ${sev.stripe}`,
        borderRadius: "12px",
        padding: "14px 16px",
        background: c.card,
        color: c.cardFg,
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        boxShadow: c.shadow,
        width: "100%",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "6px",
        }}
      >
        <span style={pillStyle(sev.bg, sev.color)}>
          {String(props.severity ?? "").toUpperCase()}
        </span>
        <span style={pillStyle("var(--muted)", c.cardFg)}>{props.category}</span>
        <span style={{ flex: 1 }} />
        <span style={pillStyle(statusStyle.bg, statusStyle.color)}>
          {statusLabel}
        </span>
      </div>
      <div style={{ fontSize: "0.95rem", fontWeight: 600 }}>{props.title}</div>
      <div style={{ fontFamily: MONO, fontSize: "0.75rem", color: c.muted }}>
        {props.file}:{props.line}
      </div>
      <div style={{ fontSize: "0.85rem", lineHeight: 1.5 }}>
        {props.explanation}
      </div>
      {suggestion && (
        <details>
          <summary
            style={{ cursor: "pointer", fontSize: "0.8rem", color: c.muted }}
          >
            Suggested fix
          </summary>
          <pre
            style={{
              margin: "6px 0 0",
              padding: "10px 12px",
              borderRadius: "8px",
              background: "var(--muted)",
              fontFamily: MONO,
              fontSize: "0.75rem",
              overflowX: "auto",
              whiteSpace: "pre",
            }}
          >
            {suggestion}
          </pre>
        </details>
      )}
      {children}
    </div>
  );
}

const DECISION_STYLE: Record<string, { bg: string; color: string }> = {
  pending: { bg: "#dbeafe", color: "#1e40af" },
  approved: { bg: "#dcfce7", color: "#166534" },
  rejected: { bg: "#fee2e2", color: "#991b1b" },
};

// ─── Renderers (type-checked against schema definitions) ────────────

const demonstrationCatalogRenderers: CatalogRenderers<DemonstrationCatalogDefinitions> =
  {
    Title: ({ props }) => {
      const Tag = (
        props.level === "h1" ? "h1" : props.level === "h3" ? "h3" : "h2"
      ) as keyof React.JSX.IntrinsicElements;
      const sizes: Record<string, string> = {
        h1: "1.75rem",
        h2: "1.25rem",
        h3: "1rem",
      };
      return (
        <Tag
          style={{
            margin: 0,
            fontWeight: 600,
            fontSize: sizes[props.level ?? "h2"],
            color: c.cardFg,
            letterSpacing: "-0.01em",
          }}
        >
          {props.text}
        </Tag>
      );
    },

    // Text: removed — use the basic catalog's Text (supports DynamicStringSchema
    // for path bindings in fixed-schema templates).

    Row: ({ props, children }) => {
      const justifyMap: Record<string, string> = {
        start: "flex-start",
        center: "center",
        end: "flex-end",
        spaceBetween: "space-between",
      };
      const items = Array.isArray(props.children) ? props.children : [];
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            gap: `${props.gap ?? 16}px`,
            alignItems: props.align ?? "stretch",
            justifyContent:
              justifyMap[props.justify ?? "start"] ?? "flex-start",
            flexWrap: "wrap",
            width: "100%",
          }}
        >
          {items.map((item: any, i: number) => {
            if (typeof item === "string")
              return (
                <div
                  key={`${item}-${i}`}
                  style={{ flex: "1 1 0", minWidth: 0 }}
                >
                  {children(item)}
                </div>
              );
            if (item && typeof item === "object" && "id" in item)
              return (
                <div
                  key={`${item.id}-${i}`}
                  style={{ flex: "1 1 0", minWidth: 0 }}
                >
                  {(children as any)(item.id, item.basePath)}
                </div>
              );
            return null;
          })}
        </div>
      );
    },

    Column: ({ props, children }) => {
      const items = Array.isArray(props.children) ? props.children : [];
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: `${props.gap ?? 12}px`,
            width: "100%",
          }}
        >
          {items.map((item: any, i: number) => {
            if (typeof item === "string")
              return (
                <React.Fragment key={`${item}-${i}`}>
                  {children(item)}
                </React.Fragment>
              );
            if (item && typeof item === "object" && "id" in item)
              return (
                <React.Fragment key={`${item.id}-${i}`}>
                  {(children as any)(item.id, item.basePath)}
                </React.Fragment>
              );
            return null;
          })}
        </div>
      );
    },

    Badge: ({ props }) => {
      const variants: Record<string, { bg: string; color: string }> = {
        success: { bg: "#dcfce7", color: "#166534" },
        warning: { bg: "#fef3c7", color: "#92400e" },
        error: { bg: "#fee2e2", color: "#991b1b" },
        info: { bg: "#dbeafe", color: "#1e40af" },
        neutral: { bg: "var(--muted)", color: c.cardFg },
      };
      const v = variants[props.variant ?? "neutral"] ?? variants.neutral;
      return (
        <span
          style={{
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: "9999px",
            fontSize: "0.7rem",
            fontWeight: 500,
            background: v.bg,
            color: v.color,
          }}
        >
          {props.text}
        </span>
      );
    },

    Button: ({ props, children }) => {
      return (
        <ActionButton label="Click" doneLabel="Done" action={props.action}>
          {props.child ? children(props.child) : null}
        </ActionButton>
      );
    },

    Note: ({ props: rawProps }) => {
      const text = (rawProps as Record<string, any>).text;
      return (
        <div
          style={{
            fontSize: "0.85rem",
            color: c.muted,
            whiteSpace: "pre-wrap",
          }}
        >
          {typeof text === "string" ? text : ""}
        </div>
      );
    },

    FindingCard: ({ props: rawProps }) => {
      // The binder resolves path bindings to plain values at runtime.
      const props = rawProps as Record<string, any>;
      const st = STATUS_STYLE[props.status] ?? STATUS_STYLE.open;
      const actionable = props.status === "open" || props.status === "approved";
      const resetKey = `${props.findingId}:${props.status}:${props.error ?? ""}`;
      const commentUrl =
        typeof props.commentUrl === "string" && /^https?:\/\//i.test(props.commentUrl)
          ? props.commentUrl
          : "";
      const error = typeof props.error === "string" ? props.error : "";
      const suggestion =
        typeof props.suggestion === "string" ? props.suggestion : "";

      return (
        <FindingBody
          props={props}
          statusLabel={String(props.status)}
          statusStyle={st}
          suggestion={suggestion}
        >
          {error && (
            <div role="alert" style={{ fontSize: "0.8rem", color: "#991b1b" }}>
              {error}
            </div>
          )}
          {props.status === "applied" && commentUrl && (
            <a
              href={commentUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: "0.8rem", color: "var(--primary)" }}
            >
              View comment
            </a>
          )}
          <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
            <ReviewButton
              label={suggestion ? "Apply fix" : "Post comment"}
              action={props.applyAction}
              disabled={!actionable}
              primary
              resetKey={resetKey}
            />
            <ReviewButton
              label="Dismiss"
              action={props.dismissAction}
              disabled={!actionable}
              resetKey={resetKey}
            />
          </div>
        </FindingBody>
      );
    },

    ApprovalItem: ({ props: rawProps }) => {
      const props = rawProps as Record<string, any>;
      const decision = String(props.decision ?? "pending");
      const st = DECISION_STYLE[decision] ?? DECISION_STYLE.pending;
      // Decisions are changeable until submit, so no pending-click guard (resetKey
      // includes the decision so the guard never sticks).
      const key = `${props.findingId}:${decision}`;
      return (
        <FindingBody props={props} statusLabel={decision} statusStyle={st}>
          <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
            <ReviewButton
              label="Approve"
              action={props.approveAction}
              disabled={decision === "approved"}
              primary={decision !== "approved"}
              resetKey={key}
            />
            <ReviewButton
              label="Reject"
              action={props.rejectAction}
              disabled={decision === "rejected"}
              resetKey={key}
            />
          </div>
        </FindingBody>
      );
    },

    ApprovalHeader: ({ props: rawProps }) => {
      const props = rawProps as Record<string, any>;
      return (
        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          <div style={{ fontSize: "1rem", fontWeight: 600 }}>{props.message}</div>
          <div style={{ fontSize: "0.8rem", color: c.muted }}>
            {props.title} ·{" "}
            <span style={{ fontFamily: MONO }}>{props.repoRef}</span>
          </div>
          <div style={{ fontSize: "0.8rem", color: "#92400e" }}>
            Submit your decisions to continue — chat is paused until then.
          </div>
        </div>
      );
    },

    ApprovalFooter: ({ props }) => (
      <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
        <ReviewButton
          label="Approve all"
          action={props.approveAllAction}
          disabled={false}
          resetKey="approve_all"
        />
        <ReviewButton
          label="Submit decisions"
          action={props.submitAction}
          disabled={false}
          primary
          resetKey="submit"
        />
      </div>
    ),

    ReviewSummary: ({ props: rawProps }) => {
      const props = rawProps as Record<string, any>;
      const summary = typeof props.summary === "string" ? props.summary : "";
      const url = typeof props.url === "string" ? props.url : "";
      const safeUrl = /^https?:\/\//.test(url) ? url : undefined;
      return (
        <div
          style={{
            border: `1px solid ${c.border}`,
            borderRadius: "12px",
            padding: "16px",
            background: c.card,
            color: c.cardFg,
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            boxShadow: c.shadow,
            width: "100%",
          }}
        >
          <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
            {safeUrl ? (
              <a
                href={safeUrl}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "inherit", textDecoration: "underline" }}
              >
                {props.title}
              </a>
            ) : (
              props.title
            )}
          </div>
          <div style={{ fontSize: "0.8rem", color: c.muted }}>
            <span style={{ fontFamily: MONO }}>{props.repoRef}</span>
            {props.author ? ` · by ${props.author}` : ""}
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <span style={pillStyle("#fee2e2", "#991b1b")}>{props.high} high</span>
            <span style={pillStyle("#fef3c7", "#92400e")}>
              {props.medium} medium
            </span>
            <span style={pillStyle("#dbeafe", "#1e40af")}>{props.low} low</span>
          </div>
          {summary && (
            <div
              style={{
                fontSize: "0.85rem",
                lineHeight: 1.5,
                whiteSpace: "pre-wrap",
              }}
            >
              {summary}
            </div>
          )}
          {props.truncated === true && (
            <div style={{ fontSize: "0.75rem", color: "#92400e" }}>
              Note: this PR was too large, so the diff was truncated and the
              review may be incomplete.
            </div>
          )}
        </div>
      );
    },
  };

// ─── Assembled Catalog ───────────────────────────────────────────────

export const demonstrationCatalog = createCatalog(
  demonstrationCatalogDefinitions,
  demonstrationCatalogRenderers,
  {
    catalogId: "copilotkit://app-dashboard-catalog",
  },
);
