"use client";

import { Suspense, useCallback, useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ExampleLayout } from "@/components/example-layout";
import { ApprovalInterrupt, ReviewFlow } from "@/components/review";
import { useGenerativeUIExamples, useExampleSuggestions } from "@/hooks";

import {
  CopilotChat,
  CopilotChatConfigurationProvider,
  CopilotThreadsDrawer,
} from "@copilotkit/react-core/v2";

import styles from "./page.module.css";

const threadsDrawerEnabled =
  process.env.NEXT_PUBLIC_COPILOTKIT_THREADS_ENABLED === "true";

function HomeContent() {
  useGenerativeUIExamples();
  useExampleSuggestions();

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
        className={
          threadsDrawerEnabled
            ? styles.layout
            : `${styles.layout} ${styles.layoutNoThreads}`
        }
      >
        {threadsDrawerEnabled ? (
          <CopilotThreadsDrawer
            agentId="default"
            onThreadSelect={(id) => setThread(id)}
            onNewThread={() => setThread(crypto.randomUUID())}
          />
        ) : null}
        <div
          className={
            threadsDrawerEnabled
              ? styles.mainPanel
              : `${styles.mainPanel} ${styles.mainPanelFullWidth}`
          }
        >
          <ExampleLayout
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
