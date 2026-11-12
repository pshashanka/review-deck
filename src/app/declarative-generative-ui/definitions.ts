/**
 * Demonstration Catalog — Component Definitions
 *
 * Platform-agnostic definitions: component names, props (Zod), descriptions.
 * This is the contract between the app and the AI agent. Agents receive these
 * definitions as context so they know what components are available.
 *
 * Renderers (React, React Native, etc.) import these definitions and provide
 * platform-specific implementations, type-checked against the Zod schemas.
 */

import { z } from "zod";

/**
 * Dynamic string: accepts either a literal string or a data-model path binding
 * like `{ path: "airline" }`. The GenericBinder resolves path bindings to the
 * actual value at render time.
 */
const DynString = z.union([z.string(), z.object({ path: z.string() })]);

const DynNumber = z.union([z.number(), z.object({ path: z.string() })]);
const DynBool = z.union([z.boolean(), z.object({ path: z.string() })]);

/** Action prop: the renderer receives a zero-arg function (or undefined). */
const ActionProp = z
  .union([
    z.object({
      event: z.object({
        name: z.string(),
        context: z.record(z.any()).optional(),
      }),
    }),
    z.null(),
  ])
  .optional();

export const demonstrationCatalogDefinitions = {
  Title: {
    description: "A heading. Use for section titles and page headers.",
    props: z.object({
      text: z.string(),
      level: z.string().optional(),
    }),
  },

  // Text: removed — the basic catalog's Text uses DynamicStringSchema
  // which supports path bindings (e.g. { path: "flights[*].airline" }).
  // Overriding it with z.string() breaks fixed-schema data binding.

  Row: {
    description: "Horizontal layout container.",
    props: z.object({
      gap: z.number().optional(),
      align: z.string().optional(),
      justify: z.string().optional(),
      // Union with { componentId, path } so GenericBinder treats this as
      // STRUCTURAL and resolves template children from the data model.
      children: z.union([
        z.array(z.string()),
        z.object({ componentId: z.string(), path: z.string() }),
      ]),
    }),
  },

  Column: {
    description: "Vertical layout container.",
    props: z.object({
      gap: z.number().optional(),
      align: z.string().optional(),
      // Same union as Row — required for template children support.
      children: z.union([
        z.array(z.string()),
        z.object({ componentId: z.string(), path: z.string() }),
      ]),
    }),
  },





  Badge: {
    description:
      "A small status badge/tag. Use for labels, statuses, categories.",
    props: z.object({
      text: z.string(),
      variant: z
        .enum(["success", "warning", "error", "info", "neutral"])
        .optional(),
    }),
  },


  Button: {
    description:
      "An interactive button with an action event. Use 'child' with a Text component ID for the label. 'action' is dispatched on click.",
    props: z.object({
      child: z
        .string()
        .describe(
          "The ID of the child component (e.g. a Text component for the label).",
        ),
      variant: z.enum(["primary", "secondary", "ghost"]).optional(),
      // Union with { event } so GenericBinder resolves this as ACTION → callable () => void.
      action: z
        .union([
          z.object({
            event: z.object({
              name: z.string(),
              context: z.record(z.any()).optional(),
            }),
          }),
          z.null(),
        ])
        .optional(),
    }),
  },


  Note: {
    description:
      "A short muted note (line breaks preserved). Accepts a literal or a data-model path binding.",
    props: z.object({
      text: DynString,
    }),
  },

  FindingCard: {
    description:
      "A code-review finding card: severity, category, title, explanation, file:line, optional suggested fix, status, and Apply fix / Dismiss buttons. Use as a template item over a findings list.",
    props: z.object({
      findingId: DynString.describe("Unique finding id; resets button state."),
      severity: DynString.describe("high | medium | low"),
      category: DynString.describe("security | tests | correctness"),
      title: DynString,
      explanation: DynString,
      file: DynString,
      line: DynNumber,
      suggestion: DynString.optional().describe(
        "Suggested replacement code; empty hides Apply fix.",
      ),
      status: DynString.describe(
        "open | approved | applied | dismissed; buttons only act on open/approved.",
      ),
      commentUrl: DynString.optional().describe(
        "URL of the posted GitHub comment (empty until applied).",
      ),
      error: DynString.optional().describe(
        "Last apply error, human-readable (empty when none).",
      ),
      alreadyRaised: DynBool.optional().describe(
        "True when an existing PR comment already covers this finding; shows a badge and relabels the post button 'Post anyway'.",
      ),
      priorLabel: DynString.optional().describe(
        "Badge text, e.g. 'Already raised by cursor[bot]'.",
      ),
      priorUrl: DynString.optional().describe(
        "URL of the existing comment the badge links to.",
      ),
      applyAction: ActionProp,
      dismissAction: ActionProp,
    }),
  },

  ApprovalItem: {
    description:
      "A high-severity finding awaiting a human decision: same visuals as FindingCard plus a decision pill (pending | approved | rejected) and Approve / Reject buttons. Use as a template item.",
    props: z.object({
      findingId: DynString,
      severity: DynString.describe("high | medium | low"),
      category: DynString.describe("security | tests | correctness"),
      title: DynString,
      explanation: DynString,
      file: DynString,
      line: DynNumber,
      decision: DynString.describe("pending | approved | rejected"),
      approveAction: ActionProp,
      rejectAction: ActionProp,
    }),
  },

  ApprovalHeader: {
    description: "Header of the approval form: message plus PR title and ref.",
    props: z.object({
      title: DynString,
      repoRef: DynString,
      message: DynString,
    }),
  },

  ApprovalFooter: {
    description:
      "Footer of the approval form with 'Approve all' and 'Submit decisions' buttons.",
    props: z.object({
      approveAllAction: ActionProp,
      submitAction: ActionProp,
    }),
  },

  ReviewSummary: {
    description:
      "Header for a PR review: linked title, owner/repo#number, author, severity counts, summary text, and a truncation note.",
    props: z.object({
      title: DynString,
      url: DynString,
      repoRef: DynString.describe("e.g. owner/repo#123"),
      author: DynString,
      high: DynNumber,
      medium: DynNumber,
      low: DynNumber,
      summary: DynString.optional(),
      truncated: DynBool.optional(),
    }),
  },
};

/** Type helper for renderers */
export type DemonstrationCatalogDefinitions =
  typeof demonstrationCatalogDefinitions;
