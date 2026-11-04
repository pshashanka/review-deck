/**
 * Suggestion pills shown in the chat UI for starting a PR review.
 */
import { useConfigureSuggestions } from "@copilotkit/react-core/v2";

export const useExampleSuggestions = () => {
  useConfigureSuggestions({
    suggestions: [
      {
        title: "Review a PR",
        message: "Review https://github.com/<owner>/<repo>/pull/<n>",
      },
      {
        title: "Example: psf/requests#6710",
        message: "Review https://github.com/psf/requests/pull/6710",
      },
      {
        title: "Example: pallets/flask#6096",
        message: "Review https://github.com/pallets/flask/pull/6096",
      },
    ],
    available: "always",
  });
};
