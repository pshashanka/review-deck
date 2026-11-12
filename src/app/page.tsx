"use client";

import { Suspense, useCallback, useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ExampleLayout } from "@/components/example-layout";
import { ApprovalInterrupt, ReviewFlow } from "@/components/review";
import {
  useGenerativeUIExamples,
  useExampleSuggestions,
  useIntelligenceStatus,
} from "@/hooks";

import {
  CopilotChat,
  CopilotChatConfigurationProvider,
  CopilotThreadsDrawer,
} from "@copilotkit/react-core/v2";

import styles from "./page.module.css";

function HomeContent() {
  useGenerativeUIExamples();
  useExampleSuggestions();
  const { status: intelligence, url: intelligenceUrl } = useIntelligenceStatus();
  // Keep the drawer while "checking" so the normal case has no layout shift.
  const hasDrawer = intelligence === "checking" || intelligence === "ok";

  // The active thread lives in the URL (?thread=<id>) so a reload reconnects to the
  // same thread (and re-surfaces a pending approval interrupt). If it is missing we
  // mint one and replace (not push) it into the URL.
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const threadId = params.get("thread");

  const setThread = useCallback(
    (id: string, mode: "push" | "replace" = "push") => {
      const next = new URLSearchParams(window.location.search);
      next.set("thread", id);
      const url = `${pathname}?${next.toString()}`;
      if (mode === "push") router.push(url);
      else router.replace(url);
    },
    [pathname, router],
  );

  useEffect(() => {
    if (!threadId) setThread(crypto.randomUUID(), "replace");
  }, [threadId, setThread]);

  if (!threadId) return null;

  return (
    /*
      The provider is CONTROLLED: `threadId` comes from the URL (?thread=) so a
      reload reconnects to the same thread. A controlled provider ignores the
      drawer's own setActiveThreadId/startNewThread, so the drawer's
      onThreadSelect / onNewThread and our "New review" button write the URL
      instead, which changes the controlled id. The chat and the canvas read the
      same active thread from the provider (the canvas's `useAgent()` falls back
      to it), so they stay on the same per-thread agent clone.
    */
    <CopilotChatConfigurationProvider agentId="default" threadId={threadId}>
      <ApprovalInterrupt />
      <div
        className={`${styles.layout} ${hasDrawer ? "" : styles.layoutNoThreads}`}
      >
        {/* SDK threads drawer (replaces the hand-rolled fork). License-gated: the locked view's Upgrade CTA opens the Intelligence docs by default. */}
        {hasDrawer && (
          <CopilotThreadsDrawer
            agentId="default"
            onThreadSelect={(id) => setThread(id)}
            onNewThread={() => setThread(crypto.randomUUID())}
          />
        )}
        <div
          className={`${styles.mainPanel} ${hasDrawer ? "" : styles.mainPanelFullWidth}`}
        >
          {intelligence === "unreachable" && (
            <div
              role="alert"
              className="border-b border-[var(--border)] bg-[var(--muted)] px-4 py-2 text-sm"
            >
              CopilotKit Intelligence is unreachable at {intelligenceUrl}. Threads
              and chat won&apos;t work until it&apos;s running (for the local
              stack, start Docker).
            </div>
          )}
          <ExampleLayout
            hasThreadsDrawer={hasDrawer}
            chatContent={
              <CopilotChat
                attachments={{ enabled: true }}
                input={{ disclaimer: () => null, className: "pb-6" }}
              />
            }
            headerActions={
              <button
                type="button"
                onClick={() => setThread(crypto.randomUUID())}
                className="rounded-md border border-[var(--border)] px-3 py-1 text-sm font-medium hover:bg-[var(--muted)]"
              >
                New review
              </button>
            }
            defaultMode="app"
            appContent={<ReviewFlow />}
          />
        </div>
      </div>
    </CopilotChatConfigurationProvider>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={null}>
      <HomeContent />
    </Suspense>
  );
}
