import { useDefaultRenderTool } from "@copilotkit/react-core/v2";
import { ToolReasoning } from "@/components/tool-rendering";

// Tool calls that are plumbing, not something to show as a reasoning card.
const HIDDEN_TOOLS = new Set([
  "render_review", // emits the A2UI review surface, which renders itself
  "log_a2ui_event", // A2UI button clicks (Apply fix / Dismiss) forwarded to the agent
  "render_a2ui", // rendered by A2UI streaming
  "generate_a2ui", // legacy A2UI tool
]);

/** Default renderer for backend tool calls; hides the internal ones above. */
export const useGenerativeUIExamples = () => {
  useDefaultRenderTool({
    render: ({ name, status, parameters }) => {
      if (HIDDEN_TOOLS.has(name)) return <></>;
      return <ToolReasoning name={name} status={status} args={parameters} />;
    },
  });
};
