export const dynamic = "force-dynamic";

/**
 * Reports whether CopilotKit Intelligence (threads) is configured and, for a
 * self-hosted INTELLIGENCE_API_URL, reachable. Never returns secrets.
 */
export async function GET() {
  const enabled = !!process.env.CPK_INTELLIGENCE_API_KEY;
  const url = enabled ? process.env.INTELLIGENCE_API_URL || null : null;

  let reachable: boolean | null = null;
  if (url) {
    try {
      // Any HTTP response means the service is up.
      await fetch(url, { signal: AbortSignal.timeout(2000), cache: "no-store" });
      reachable = true;
    } catch {
      reachable = false;
    }
  }

  return Response.json({ enabled, reachable, url });
}
