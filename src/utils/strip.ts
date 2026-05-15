/**
 * Recursively delete fields from an object tree. Used to scrub Anthropic-only
 * fields (cache_control, reasoning) before forwarding to OpenAI-compatible
 * upstreams that 400 on unknown keys.
 */
export function stripFields(obj: unknown, fields: ReadonlySet<string>): void {
  if (!obj || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const item of obj) stripFields(item, fields);
    return;
  }
  const o = obj as Record<string, unknown>;
  for (const key of Object.keys(o)) {
    if (fields.has(key)) {
      delete o[key];
    } else {
      stripFields(o[key], fields);
    }
  }
}

// Always rejected by OpenAI-shape upstreams (both Chat Completions and Responses).
const ALWAYS_STRIP = new Set(["cache_control"]);

// `reasoning` is emitted by AnthropicTransformer from `request.thinking` and
// consumed by OpenAIResponsesTransformer.transformRequestIn — strip only on
// the Chat Completions path, where vanilla upstreams 400 on unknown keys.
const CHAT_COMPLETIONS_REJECT = new Set(["reasoning"]);

export function scrubAnthropicOnlyFields(body: Record<string, unknown>): void {
  stripFields(body, ALWAYS_STRIP);
}

export function scrubChatCompletionsIncompatibleFields(
  body: Record<string, unknown>,
): void {
  stripFields(body, CHAT_COMPLETIONS_REJECT);
  renameMaxTokensForReasoningModels(body);
}

/**
 * OpenAI reasoning-tier models (gpt-5 family, o-series) reject the legacy
 * `max_tokens` parameter and require `max_completion_tokens`. Traditional
 * Chat Completions models (gpt-4o etc.) accept both names, while other
 * OpenAI-compatible upstreams (Kimi, DeepSeek, OpenRouter aliases, Claude
 * via Bedrock, etc.) often only know the legacy `max_tokens` name and 400
 * on unknown keys. We therefore only rename when the request model name
 * looks like an OpenAI reasoning model.
 *
 * The pattern matches:
 *   - `gpt-5`, `gpt-5-mini`, `gpt-5.5`, `gpt-5-codex`, ...  (gpt-5 family)
 *   - `o1`, `o3`, `o4-mini`, `o1-preview`, ...               (o-series)
 *
 * It deliberately does NOT match unrelated names like `o100k_base` (a
 * tokenizer ID) or `gpt-4o` (which accepts both field names anyway).
 *
 * Only the top-level field is touched; nested usage in tool definitions
 * etc. is left intact.
 */
const REASONING_MODEL_RE = /^(gpt-5|o\d)([-._]|$)/i;

function renameMaxTokensForReasoningModels(
  body: Record<string, unknown>,
): void {
  if (!body || typeof body !== "object") return;
  const model = body.model;
  if (typeof model !== "string" || !REASONING_MODEL_RE.test(model)) return;
  if (
    Object.prototype.hasOwnProperty.call(body, "max_tokens") &&
    !Object.prototype.hasOwnProperty.call(body, "max_completion_tokens")
  ) {
    body.max_completion_tokens = body.max_tokens;
    delete body.max_tokens;
  }
}
