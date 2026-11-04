#!/usr/bin/env node
// Warns about missing LLM vendor API keys and missing CopilotKit Intelligence
// configuration before `npm run dev`, then lets the dev server start anyway
// (chat/generations and Intelligence features fail until the values are set).
// Written by `copilotkit init`; safe to delete if you manage env validation
// yourself.
import { existsSync, readFileSync } from 'node:fs';

/** Vendor API keys this scaffold needs before chat and generations will work. */
const REQUIRED_ENV_KEYS = [
  {
    "key": "OPENAI_API_KEY",
    "note": "Required by the agent runtime.",
    "url": "https://platform.openai.com/api-keys",
    "example": "sk-..."
  }
];

/**
 * Hosted CopilotKit Intelligence env var NAMES this scaffold was generated with
 * (no values — those live only in .env). Empty for a non-hosted build. A
 * missing/blank value means the generated .env was overwritten.
 */
const REQUIRED_INTELLIGENCE_KEYS = [];

const envFileExists = existsSync('.env');
const envFileContent = envFileExists ? readFileSync('.env', 'utf8') : '';

// Shared detection predicate — authored once in vendor-key-predicate.ts and
// spliced here verbatim so the CLI pre-check (ENT-677) and this gate cannot
// drift. Defines readDotenvValue / isPlaceholderValue / resolveVendorKeyValue /
// findUnsatisfiedVendorKeys.

function readDotenvValue(envFileContent, key) {
  // The key is matched as text, never as a pattern (PE-520).
  const literal = key.replace(/[.*+?^$(){}|[\]\\]/g, '\\$&');
  const re = new RegExp('^\\s*(?:export\\s+)?' + literal + '=(.*)$', 'gm');
  let match;
  let last = null;
  while ((match = re.exec(envFileContent)) !== null) {
    last = match;
  }
  if (!last) {
    return '';
  }
  const raw = last[1].trim();
  if (
    raw.length >= 2 &&
    ((raw.startsWith('"') && raw.endsWith('"')) ||
      (raw.startsWith("'") && raw.endsWith("'")))
  ) {
    return raw.slice(1, -1).trim();
  }
  if (raw.startsWith('#')) {
    return '';
  }
  return raw.replace(/\s+#.*$/, '').trim();
}

function isPlaceholderValue(value) {
  return /^<.*>$/.test(value) || /^your[-_]/i.test(value) || /\.\.\.$/.test(value);
}

function resolveVendorKeyValue(processEnv, envFileContent, key) {
  const fromProcess = ((processEnv && processEnv[key]) || '').trim();
  return fromProcess !== '' ? fromProcess : readDotenvValue(envFileContent, key);
}

function findUnsatisfiedVendorKeys(processEnv, envFileContent, requirements) {
  return requirements
    .map((requirement) => ({
      requirement,
      value: resolveVendorKeyValue(processEnv, envFileContent, requirement.key),
    }))
    .filter(({ value }) => value === '' || isPlaceholderValue(value));
}


const failedKeys = findUnsatisfiedVendorKeys(process.env, envFileContent, REQUIRED_ENV_KEYS)
  .map(({ requirement, value }) => ({ entry: requirement, value }));

if (failedKeys.length > 0) {
  for (const { entry, value } of failedKeys) {
    if (value === '') {
      console.warn(
        `⚠ Missing API key: ${entry.key} — chat and generations will not work until you set it.`,
      );
    } else {
      console.warn(
        `⚠ Placeholder value for API key: ${entry.key} — chat and generations will not work until you replace it with your real key.`,
      );
    }
    console.warn(`  ${entry.note}`);
    console.warn(`  Get a key:  ${entry.url}`);
    console.warn(`  Add to .env:  ${entry.key}=${entry.example}`);
    console.warn('');
  }
  if (!envFileExists) {
    console.warn(
      'No .env file found in this directory; create one containing the line(s) above.',
    );
  }
  console.warn(
    'Starting the dev server anyway — set the key(s) above and restart to enable chat.',
  );
}

// Hosted Intelligence config check (ENT-949). Reuses the shared predicate above:
// a key is unsatisfied when missing/blank in both process.env and .env — the
// symptom of an overwritten generated .env. Names the missing keys only; the
// values belong solely in .env, so the fix is to restore it or re-run init.
const intelligenceFailures = findUnsatisfiedVendorKeys(
  process.env,
  envFileContent,
  REQUIRED_INTELLIGENCE_KEYS,
).map(({ requirement }) => requirement);

if (intelligenceFailures.length > 0) {
  console.warn(
    '⚠ CopilotKit Intelligence is not configured — your .env may have been overwritten.',
  );
  for (const entry of intelligenceFailures) {
    console.warn(`  Missing:  ${entry.key}`);
  }
  console.warn(
    'Restore the value(s) in .env, or re-run `copilotkit init` to reconfigure.',
  );
  console.warn(
    'Starting the dev server anyway — Intelligence features will not work until these are set.',
  );
}

// GITHUB_TOKEN is optional: without it, public PRs still review (rate-limited)
// but private repos fail and "Apply fix" cannot post suggestion comments.
if (resolveVendorKeyValue(process.env, envFileContent, 'GITHUB_TOKEN') === '') {
  console.warn(
    '⚠ GITHUB_TOKEN is not set — private repos and "Apply fix" (posting suggestions) will not work, and public PR fetches are rate-limited.',
  );
  console.warn(
    '  Fine-grained token with Pull requests read/write + Contents read; add to .env:  GITHUB_TOKEN=github_pat_...',
  );
}

// Warn when a configured model (COPILOTKIT_AGENT_MODEL or any per-step *_MODEL) uses a
// provider whose API key is missing.
const MODEL_VARS = [
  'COPILOTKIT_AGENT_MODEL', 'REVIEWER_MODEL', 'LINT_MODEL', 'SECURITY_MODEL',
  'TESTS_MODEL', 'REGRESSION_MODEL', 'SUMMARY_MODEL', 'CHAT_MODEL',
];
const PROVIDER_KEYS = {
  anthropic: 'ANTHROPIC_API_KEY',
  google: 'GOOGLE_API_KEY', gemini: 'GOOGLE_API_KEY', 'google-gemini': 'GOOGLE_API_KEY',
  xai: 'XAI_API_KEY', grok: 'XAI_API_KEY',
  openai: 'OPENAI_API_KEY',
};
const missingProviderKeys = new Map();
for (const name of MODEL_VARS) {
  const spec = resolveVendorKeyValue(process.env, envFileContent, name);
  const provider = spec.split(/[:/]/)[0].toLowerCase();
  const keyName = PROVIDER_KEYS[provider];
  if (keyName && resolveVendorKeyValue(process.env, envFileContent, keyName) === '') {
    missingProviderKeys.set(keyName, [...(missingProviderKeys.get(keyName) ?? []), name]);
  }
}
for (const [keyName, vars] of missingProviderKeys) {
  if (keyName === 'OPENAI_API_KEY') continue; // already covered by the required-key check above
  console.warn(`⚠ ${vars.join(', ')} use a model whose provider key ${keyName} is not set — those steps will fail until you set it.`);
}

process.exit(0);
