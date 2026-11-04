"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { A2UIProvider, A2UIRenderer, useA2UI } from "@copilotkit/a2ui-renderer";
import type { A2UIClientEventMessage } from "@copilotkit/a2ui-renderer";
import { useInterrupt } from "@copilotkit/react-core/v2";
import { demonstrationCatalog } from "@/app/declarative-generative-ui/renderers";

type Decision = "approve" | "reject";
type A2UIOp = Record<string, unknown>;
type ApprovalPayload = {
  type: "high_severity_review";
  message?: string;
  findings: { id: string }[];
  a2ui_operations: A2UIOp[];
};

/** The AG-UI `on_interrupt` value arrives as a JSON string; accept an object too. */
function parsePayload(value: unknown): Partial<ApprovalPayload> | null {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return null;
    }
  }
  return value && typeof value === "object"
    ? (value as Partial<ApprovalPayload>)
    : null;
}

function surfaceIdOf(ops: A2UIOp[]): string | null {
  for (const op of ops) {
    const cs = op.createSurface as { surfaceId?: string } | undefined;
    if (cs?.surfaceId) return cs.surfaceId;
  }
  return null;
}

/** Renders the approval surface; exposes `processMessages` to the parent via `apiRef`. */
function ApprovalSurface({
  ops,
  surfaceId,
  onceKey,
  apiRef,
}: {
  ops: A2UIOp[];
  surfaceId: string;
  onceKey: string;
  apiRef: React.MutableRefObject<ReturnType<typeof useA2UI> | null>;
}) {
  const api = useA2UI();
  apiRef.current = api;
  const processed = useRef<string | null>(null);
  useEffect(() => {
    if (processed.current === onceKey) return;
    processed.current = onceKey;
    api.processMessages(ops);
  }, [api, ops, onceKey]);
  return <A2UIRenderer surfaceId={surfaceId} />;
}

function ApprovalForm({
  payload,
  resolve,
}: {
  payload: ApprovalPayload;
  resolve: (p: unknown) => Promise<unknown> | void;
}) {
  const ids = payload.findings.map((f) => f.id);
  const surfaceId = surfaceIdOf(payload.a2ui_operations) ?? "pr-approval";
  const onceKey = `${surfaceId}:${ids.join(",")}`;
  const decisions = useRef<Record<string, Decision>>({});
  const submitted = useRef(false);
  const apiRef = useRef<ReturnType<typeof useA2UI> | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const setDecision = useCallback(
    (id: string, d: Decision) => {
      const index = ids.indexOf(id);
      if (index < 0) return;
      decisions.current[id] = d;
      apiRef.current?.processMessages([
        {
          version: "v0.9",
          updateDataModel: {
            surfaceId,
            path: `/findings/${index}/decision`,
            value: d === "approve" ? "approved" : "rejected",
          },
        },
      ]);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onceKey],
  );

  const submit = useCallback(
    (all?: boolean) => {
      if (submitted.current) return;
      submitted.current = true;
      setSubmitting(true);
      const out: Record<string, Decision> = {};
      for (const id of ids) out[id] = all ? "approve" : (decisions.current[id] ?? "approve");
      Promise.resolve(resolve({ decisions: out })).catch(() => {
        // Resume failed: let the user try again.
        submitted.current = false;
        setSubmitting(false);
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onceKey, resolve],
  );

  const onAction = useCallback(
    (msg: A2UIClientEventMessage) => {
      const { name, context, surfaceId: sid } = msg.userAction ?? {};
      if (sid && sid !== surfaceId) return;
      const findingId =
        context && typeof context.findingId === "string" ? context.findingId : null;
      if (submitted.current) return;
      if ((name === "approve" || name === "reject") && findingId) {
        setDecision(findingId, name);
      } else if (name === "approve_all") {
        submit(true);
      } else if (name === "submit") {
        submit();
      }
    },
    [surfaceId, setDecision, submit],
  );

  return (
    <div style={{ padding: "8px 0", width: "100%" }}>
      <A2UIProvider catalog={demonstrationCatalog} onAction={onAction}>
        <ApprovalSurface
          ops={payload.a2ui_operations}
          surfaceId={surfaceId}
          onceKey={onceKey}
          apiRef={apiRef}
        />
      </A2UIProvider>
      {submitting && (
        <div style={{ fontSize: "0.8rem", opacity: 0.7, marginTop: 6 }}>
          Submitting decisions…
        </div>
      )}
    </div>
  );
}

/** Registers the high-severity approval interrupt UI (renders inside the chat). */
export function ApprovalInterrupt() {
  useInterrupt({
    agentId: "default",
    enabled: (event) => parsePayload(event.value)?.type === "high_severity_review",
    render: ({ event, resolve }) => {
      const raw = parsePayload(event.value) ?? {};
      const payload: ApprovalPayload = {
        type: "high_severity_review",
        message: raw.message,
        findings: raw.findings ?? [],
        a2ui_operations: raw.a2ui_operations ?? [],
      };
      const key = `${surfaceIdOf(payload.a2ui_operations)}:${payload.findings.map((f) => f.id).join(",")}`;
      return <ApprovalForm key={key} payload={payload} resolve={resolve} />;
    },
  });
  return null;
}
