// Offline slice of the effort-history contract (revision 9). Pure logic, no network,
// no imports from opencodex. Nothing here is wired into production requests.
import { createHash } from "node:crypto";

export const MAX_ENTRIES = 64;
export const MAX_TOTAL_BYTES = 512 * 1024; // section 12, hard bound
// Proposed cap on the output buffered for one pending slot (owner to confirm). Entries store hashes only,
// so the cap applies to the buffer that must exist until message_stop, not to the stored record.
export const MAX_BUFFER_BYTES = 128 * 1024;
// Change points are re-sent on every later turn, so their count is capped (owner to confirm).
export const MAX_CHANGE_POINTS = 64;
export const ENTRY_TTL_MS = 60 * 60 * 1000; // 1 hour from last use (section 12)
export const PENDING_TTL_MS = 10 * 60 * 1000; // proposed: a pending slot never outlives this
export const SCOPE_PARTS = ["cohort", "provider", "destination", "adapter", "model", "credential"];

// Canonical JSON: object keys sorted at every level, so equal content hashes equally.
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((k) => [k, canonical(value[k])]));
  }
  return value;
}
const stableJson = (value) => JSON.stringify(canonical(value));
// The one projection used for commit and lookup (section 6). Key order is canonical; nothing else is dropped.
export const project = (items) => canonical(items);
export const hash = (value) => createHash("sha256").update(stableJson(value)).digest("hex");
const bytesOf = (value) => Buffer.byteLength(stableJson(value));

// Section 6: minimal maps to low, so the two are not a change point.
// Section 6: `minimal` maps to `low` for adaptive output, as in anthropic.ts adaptiveEffort (L612).
export const mapEffort = (effort) => (effort === "minimal" ? "low" : effort);

// Reasoning budget by effort (anthropic.ts reasoningBudget, L599).
const BUDGET = { minimal: 1024, low: 4096, medium: 8192, high: 16384, xhigh: 24576, max: 32000 };
// Effective output limit for this request (section 6). An explicit client cap is used as sent.
// Otherwise the adaptive derivation: max(omitted, min(40192, max(8192, budget + 8192))).
export function effectiveLimit(req) {
  if (req.explicitCap !== undefined) return req.explicitCap;
  // The budget is derived from the mapped effort (anthropic.ts L1110–1113: wantBudget uses adaptiveEffort).
  const budget = BUDGET[mapEffort(req.effort)] ?? 8192;
  return Math.max(req.omittedMaxTokens ?? 8192, Math.min(40192, Math.max(8192, budget + 8192)));
}

const off = (reason, extra = {}) => ({ action: "off", reason, ...extra });

// Documented 400 text for a model without per-message effort (Anthropic effort docs, read 2026-10-25):
// "output_config.effort requires a model that supports per-turn effort; this model does not".
// Anything else in a 400 sets nothing.
export const BETA_FRAGMENT = "requires a model that supports per-turn effort";
// Capability is per model and destination, not per chat or credential (section 3a).
const capabilityKey = (s) => hash({ model: s.model, destination: s.destination });

export function createStore() {
  return { entries: new Map(), pending: new Map(), capability: new Map(), committedInputs: new Map(), nextVersion: 1 };
}

// Entry key: scope + committed prefix (section 5). The frame is a field, not part of the key,
// so a frame change finds the entry by prefix and rule 6 can fire.
const entryKey = (scopeKey, prefixHash) => hash([scopeKey, prefixHash]);
const newKey = (scopeKey, inputHash) => hash(["new", scopeKey, inputHash]);

// Rule 5 signal from the x-codex-turn-metadata header value. Only an explicit
// history_initialization "New" counts. Missing, malformed, ColdResume and others are not new.
export function lineageFromMetadata(headerValue) {
  if (typeof headerValue !== "string") return false;
  let parsed;
  try {
    parsed = JSON.parse(headerValue);
  } catch {
    return false;
  }
  return parsed !== null && typeof parsed === "object" && parsed.history_initialization === "New";
}

const isUser = (item) => item.kind === "user";
const isTool = (item) => item.kind === "tool_result";
const isDeveloper = (item) => item.kind === "developer";
const hasConfigUpdate = (items) => items.some((item) => item.kind === "configuration_update");

// Off, and the parent entry is deleted before the request is sent (sections 7 and 9).
function offSend(store, parentKey, reason) {
  store.entries.delete(parentKey);
  store.pending.delete(parentKey);
  return off(reason, { deleted: true });
}

// Drop entries past their TTL and pending slots past theirs.
function sweep(store, now) {
  for (const [key, entry] of store.entries) {
    if (now - entry.lastUsed > ENTRY_TTL_MS) store.entries.delete(key);
  }
  for (const [key, slot] of store.pending) {
    if (now - slot.createdAt > PENDING_TTL_MS) store.pending.delete(key);
  }
}

// Ordered algorithm, section 7. The first rule that decides wins.
export function decide(store, req, opts = {}) {
  const now = opts.now ?? Date.now();
  sweep(store, now);

  // Rule 0 (global gate): nothing runs until activation is confirmed (section 13, item 1).
  if (opts.enabled !== true) return off("switch-off");
  if (opts.activated !== true) return off("not-activated");
  if (req.transport === "websocket" && opts.websocketHeaderRead !== true) return off("websocket-before-activation");

  // Fable 5.1 is in scope only once its wire ID is checked against the Mythos rule; until then off.
  if (/fable/i.test(req.scope?.model ?? "")) return off("fable-wire-id-unchecked");
  // Rule 2: every scope part is required.
  const s = req.scope;
  if (!s || SCOPE_PARTS.some((k) => !s[k])) return off("missing-scope");
  // Chat identity: two chats can share a first message, so each chat needs its own lineage value.
  // Without one, nothing is optimized (off), and entries and replay records are per chat.
  if (!req.chatLineage) return off("no-chat-lineage");
  const scopeKey = hash([s, req.chatLineage]);
  const parentHash = hash(req.parent);
  const parentKey = entryKey(scopeKey, parentHash);
  const inputHash = hash([...req.parent, ...req.suffix]);

  // Rule 2a: no cache breakpoint. Rule 3: the proxy stripped reasoning.
  if (!req.cacheControl) return offSend(store, parentKey, "no-cache-control");
  if (req.stripped) return offSend(store, parentKey, "proxy-stripped");

  // Rule 3a: an unsupported capability record.
  if (store.capability.get(capabilityKey(s)) === "unsupported") return offSend(store, parentKey, "capability-unsupported");

  // Rule 3b runs before rule 4: a thinking change on a matching parent.
  const parent = store.entries.get(parentKey);
  if (parent && req.thinking !== parent.thinking) return offSend(store, parentKey, "thinking-change");

  // Rule 4: replay. The request is a turn already committed (same input hash). Sent as today,
  // and the matched entry plus the parent are deleted before the send. Nothing is written.
  for (const [key, entry] of store.entries) {
    if (entry.scopeKey === scopeKey && entry.inputHash === inputHash) {
      store.entries.delete(key);
      store.entries.delete(parentKey);
      return off("replay", { deleted: true });
    }
  }
  // The turn's entry may already be consumed by a later turn; the committed-input record still refuses it.
  if (store.committedInputs.has(hash([scopeKey, inputHash]))) {
    store.entries.delete(parentKey);
    return off("replay", { deleted: true });
  }
  // A placeholder with the same input hash is a duplicate first request: off, and it is deleted.
  const placeholder = newKey(scopeKey, inputHash);
  if (store.pending.has(placeholder)) {
    store.pending.delete(placeholder);
    return off("placeholder-exists", { deleted: true });
  }

  if (hasConfigUpdate(req.suffix)) return offSend(store, parentKey, "configuration-update-in-suffix");

  // A second request for a parent that is already pending is a fork or retry (section 8).
  if (store.pending.has(parentKey)) {
    store.pending.delete(parentKey);
    return off("in-flight-collision", { deleted: true });
  }

  const suffix = req.suffix;
  const userCount = suffix.filter(isUser).length;
  const toolOnly = suffix.length > 0 && suffix.every(isTool);
  // Shape (a): one user message, optionally after tool results. Developer items may appear.
  const firstUser = suffix.findIndex(isUser);
  const shapeA = userCount === 1
    && suffix.every((item) => isUser(item) || isTool(item) || isDeveloper(item))
    && suffix.slice(firstUser + 1).every((item) => !isTool(item));

  if (!parent) {
    // Rule 7: a compaction summary is user-role; it never starts a history.
    if (req.compactionSummary === true) return off("compaction-summary");
    // Activation condition 2: nothing starts a history until the compaction marker is identified.
    if (opts.compactionMarkerIdentified !== true) return off("compaction-marker-unidentified");
    // Rules 5 and 7: no entry. A new history needs the positive new-chat signal, one user
    // message, and no tool or assistant items. A restart, eviction, or compaction is not new.
    const newChat = lineageFromMetadata(req.turnMetadata) && req.parent.length === 0
      && userCount === 1 && suffix.every((item) => isUser(item) || isDeveloper(item));
    if (!newChat) return off("no-history");
    if (store.entries.size + store.pending.size >= MAX_ENTRIES) {
      const oldest = leastRecentlyUsed(store);
      if (oldest === undefined) return off("bounds-refused");
      store.entries.delete(oldest);
    }
    const version = store.nextVersion++;
    const token = `${placeholder}:${version}`;
    store.pending.set(placeholder, {
      kind: "new", key: placeholder, token, version, createdAt: now, inputHash, scopeKey, req, attemptGeneration: req.generation,
    });
    return { action: "send", kind: "new", pendingToken: token };
  }

  // Rule 6: frame and generation.
  if (parent.frame !== req.frame) return offSend(store, parentKey, "frame-mismatch");
  if (parent.generation !== req.generation) return offSend(store, parentKey, "generation-mismatch");

  // Section 6: any output limit different from the anchor's is off.
  if (effectiveLimit(req) !== parent.limit) return offSend(store, parentKey, "limit-differs");

  if (userCount > 1) return offSend(store, parentKey, "more-than-one-user-message");

  const effort = mapEffort(req.effort);
  if (toolOnly) {
    if (effort !== parent.effortInEffect) return offSend(store, parentKey, "tool-only-effort-change");
    return admitMatched(store, { scopeKey, parentKey, req, parent, now, inputHash, changePoint: false, shape: "tool-only" });
  }
  if (!shapeA) return offSend(store, parentKey, "unclassified-suffix");

  const changePoint = effort !== parent.effortInEffect;
  if (changePoint && parent.changePoints.length >= MAX_CHANGE_POINTS) return offSend(store, parentKey, "change-points-full");
  // Mythos: any effort change is off (section 6), because its budget depends on effort.
  if (changePoint && /mythos/i.test(s.model)) return offSend(store, parentKey, "mythos-effort-change");
  return admitMatched(store, { scopeKey, parentKey, req, parent, now, inputHash, changePoint, shape: "a" });
}

// Matched: the parent entry moves to a pending slot until its commit (section 8).
function admitMatched(store, plan) {
  store.entries.delete(plan.parentKey);
  const version = store.nextVersion++;
  const token = `${plan.parentKey}:${version}`;
  store.pending.set(plan.parentKey, {
    kind: "matched", key: plan.parentKey, token, version, createdAt: plan.now, inputHash: plan.inputHash,
    attemptGeneration: plan.req.generation,
    scopeKey: plan.scopeKey, entry: plan.parent, req: plan.req, changePoint: plan.changePoint,
  });
  return {
    action: "send",
    kind: "matched",
    pendingToken: token,
    changePoint: plan.changePoint,
    shape: plan.shape,
    // What the adapter inserts for the change point: an effort-only system message at this position.
    // The wire form (role, JSON shape) is not yet verified (contract section 13, open item).
    insert: plan.changePoint ? { at: changePointAt(plan.req), effort: mapEffort(plan.req.effort) } : null,
    // Every change point on this chat, in prefix order, so the adapter re-sends all of them each turn.
    changePoints: [
      ...plan.parent.changePoints.map((c) => ({ at: c.at, effort: c.effort })),
      ...(plan.changePoint ? [{ at: changePointAt(plan.req), effort: mapEffort(plan.req.effort) }] : []),
    ],
  };
}

// The pending slot is the compare-and-set check: a commit is written only if its slot still exists.
// A slot that was deleted or replaced by another request is never brought back.
function totalBytes(store) {
  let total = 0;
  for (const entry of store.entries.values()) total += entry.bytes;
  // A matched parent in flight still holds its bytes until its commit or restore.
  // Streaming output buffered for a pending slot counts too (section 12).
  for (const slot of store.pending.values()) {
    if (slot.entry) total += slot.entry.bytes;
    total += slot.buffered ?? 0;
  }
  return total;
}

// Resolve a token to its slot without removing it. A token from an earlier admission does not match.
function peekSlot(store, token) {
  const sep = token.lastIndexOf(":");
  if (sep < 0) return null;
  const key = token.slice(0, sep);
  const slot = store.pending.get(key);
  if (!slot || slot.token !== token) return null;
  return slot;
}

function takeSlot(store, token) {
  const slot = peekSlot(store, token);
  if (slot) store.pending.delete(slot.key);
  return slot;
}

// Eviction order: the entry with the oldest last use (section 12), not the first inserted.
function leastRecentlyUsed(store) {
  let best;
  for (const [key, entry] of store.entries) {
    if (best === undefined || entry.lastUsed < store.entries.get(best).lastUsed) best = key;
  }
  return best;
}

// Streaming: buffered output for a pending slot. If it would push the store past 512 KB, the slot
// is deleted and nothing commits; the response itself is not interrupted (section 12).
export function buffer(store, token, bytes) {
  const slot = peekSlot(store, token);
  if (!slot) return { action: "dropped", reason: "pending-missing" };
  if (totalBytes(store) + bytes > MAX_TOTAL_BYTES) {
    store.pending.delete(slot.key);
    return { action: "deleted", reason: "streaming-over-store-cap" };
  }
  // The per-slot cap is enforced while streaming, not only at commit (section 9, step 5).
  if ((slot.buffered ?? 0) + bytes > MAX_BUFFER_BYTES) {
    store.pending.delete(slot.key);
    return { action: "deleted", reason: "over-buffer-cap" };
  }
  slot.buffered = (slot.buffered ?? 0) + bytes;
  return { action: "buffering" };
}

function evictUntilFits(store, bytes) {
  // Decide first, evict second: if the pending bytes alone do not leave room, nothing is evicted.
  let evictable = 0;
  for (const entry of store.entries.values()) evictable += entry.bytes;
  const pendingBytes = totalBytes(store) - evictable;
  if (pendingBytes + bytes > MAX_TOTAL_BYTES) return false;
  while (totalBytes(store) + bytes > MAX_TOTAL_BYTES) {
    const oldest = leastRecentlyUsed(store);
    store.entries.delete(oldest);
  }
  return true;
}

function recordCommittedInput(store, slot) {
  store.committedInputs.set(hash([slot.scopeKey, slot.inputHash]), true);
  while (store.committedInputs.size > MAX_ENTRIES * 4) {
    store.committedInputs.delete(store.committedInputs.keys().next().value);
  }
}

// Outcomes after a send (section 9), in the contract's order.
export function commit(store, token, result, now = Date.now()) {
  const slot = takeSlot(store, token);
  if (!slot) return { action: "dropped", reason: "pending-missing" };
  if (now - slot.createdAt > PENDING_TTL_MS) return { action: "dropped", reason: "pending-expired" };
  const { req } = slot;

  if (result.failedBeforeSend) {
    // Default: delete. Restore only the matched parent, and only when the owner has enabled it
    // (section 13.4). A failed placeholder is always deleted.
    if (slot.kind === "matched" && result.restoreOnFailure === true) {
      if (!evictUntilFits(store, slot.entry.bytes)) return { action: "dropped", reason: "over-store-cap" };
      store.entries.set(slot.key, slot.entry);
      return { action: "restored" };
    }
    return { action: "deleted", reason: slot.kind === "matched" ? "failed-before-send" : "placeholder-failed" };
  }
  if (!result.messageStop) return { action: "deleted", reason: "uncertain-outcome" };

  // 2. Cache check. A missing cache read counts as zero.
  const cacheRead = result.cacheRead ?? 0;
  if (slot.kind === "matched" && cacheRead < slot.entry.P) {
    return { action: "deleted", reason: "cache-below-P" };
  }
  if (!(result.P > 0)) {
    if (slot.kind === "new") recordCommittedInput(store, slot);
    return { action: "not-committed", reason: "P-zero" };
  }
  // 3. Generation check: the generation at commit must equal the attempt's.
  // Section 9, step 4: the value at commit must equal both the lookup value and the attempt-start value.
  if (result.generation !== req.generation || result.generation !== slot.attemptGeneration) {
    return { action: "deleted", reason: "generation-changed" };
  }

  // 5. Size checks (after the slot check). The buffered output cap, then the store total.
  // The stored record is hashes and pinned values only, so its size is small (section 10).
  const items = [...req.parent, ...req.suffix, ...result.outputItems];
  // One measure for both checks: the larger of the bytes streamed (buffer) and the canonical output size.
  if (Math.max(slot.buffered ?? 0, bytesOf(result.outputItems)) > MAX_BUFFER_BYTES) return { action: "deleted", reason: "over-buffer-cap" };

  const prior = slot.kind === "matched" ? slot.entry : null;
  const effort = mapEffort(req.effort);
  const changePoints = prior
    ? [...prior.changePoints, ...(slot.changePoint ? [{ at: changePointAt(req), effort }] : [])]
    : [];
  const effortInEffect = prior && !slot.changePoint ? prior.effortInEffect : effort;

  const prefixHash = hash(items);
  const record = {
    scopeKey: slot.scopeKey,
    frame: req.frame,
    generation: result.generation,
    prefixHash,
    inputHash: slot.inputHash,
    effortInEffect,
    thinking: req.thinking,
    limit: effectiveLimit(req),
    changePoints,
    P: result.P,
    version: store.nextVersion++,
    lastUsed: now,
  };
  const bytes = bytesOf(record);
  if (!evictUntilFits(store, bytes)) return { action: "dropped", reason: "over-store-cap" };

  const key = entryKey(slot.scopeKey, prefixHash);
  // Section 5(a): the change point sits before the user message, after any leading tool results.
  // Remember the committed input so a later retry of this turn is refused even after its entry is consumed.
  // Bounded: the oldest record goes first.
  recordCommittedInput(store, slot);
  store.entries.set(key, { ...record, bytes });
  return { action: "committed", key };
}

// A 400 from upstream (section 11). A 400 that names the beta or effort message sets the
// capability record to unsupported and is not resent. Other errors delete the pending slot.
export function rejectUpstream(store, token, { status, errorText = "" }) {
  const slot = takeSlot(store, token);
  if (!slot) return { action: "dropped", reason: "pending-missing" };
  if (status === 400 && errorText.includes(BETA_FRAGMENT)) {
    store.capability.set(capabilityKey(slot.req.scope), "unsupported");
    return { action: "off", reason: "capability-unsupported", deleted: true };
  }
  // Any other error (other 400, 5xx, transport failure): error to the caller, no resend, no capability record.
  return { action: "error", reason: status === 400 ? "other-400" : "upstream-error", deleted: true };
}

// The client disconnected or the request was abandoned after the send: nothing is committed.
// `sent` says whether any bytes reached the upstream. If not, a matched parent is restored (as in
// the failed-before-send rule); if so, the parent is deleted.
export function abandon(store, token, { sent = true, restoreOnFailure = false } = {}) {
  const slot = takeSlot(store, token);
  if (!slot) return { action: "dropped", reason: "pending-missing" };
  if (!sent && restoreOnFailure && slot.kind === "matched") {
    if (!evictUntilFits(store, slot.entry.bytes)) return { action: "dropped", reason: "over-store-cap" };
    store.entries.set(slot.key, slot.entry);
    return { action: "restored" };
  }
  return { action: "deleted", reason: "abandoned" };
}

// Section 5(a): the change point goes after the leading tool results, before the user message.
// Returned as an index into the request input array (the parent items come first).
function changePointAt(req) {
  const firstUser = req.suffix.findIndex(isUser);
  return req.parent.length + (firstUser < 0 ? 0 : firstUser);
}
