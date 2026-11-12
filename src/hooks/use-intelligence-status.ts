import { useEffect, useState } from "react";

export type IntelligenceStatus = "disabled" | "checking" | "ok" | "unreachable";

interface StatusResponse {
  enabled: boolean;
  reachable: boolean | null;
  url: string | null;
}

/**
 * Checks (once, on mount) whether CopilotKit Intelligence is usable. "disabled"
 * when the build has no key (no fetch needed); "unreachable" only when a
 * self-hosted URL did not respond. A failure of the status route itself is
 * treated as "ok" so we never hide the drawer because of our own error.
 */
export function useIntelligenceStatus(): {
  status: IntelligenceStatus;
  url: string | null;
} {
  const enabled = process.env.NEXT_PUBLIC_COPILOTKIT_THREADS_ENABLED === "true";
  const [state, setState] = useState<{
    status: IntelligenceStatus;
    url: string | null;
  }>({ status: enabled ? "checking" : "disabled", url: null });

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    fetch("/api/intelligence-status", {
      cache: "no-store",
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.json() as Promise<StatusResponse>;
      })
      .then((data) =>
        setState({
          status: data.reachable === false ? "unreachable" : "ok",
          url: data.url,
        }),
      )
      .catch((err) => {
        if (controller.signal.aborted) return;
        console.warn("intelligence-status check failed", err);
        setState({ status: "ok", url: null });
      });
    return () => controller.abort();
  }, [enabled]);

  return state;
}
