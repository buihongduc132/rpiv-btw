/**
 * btw-context-budget.ts — Context window budget module for /btw.
 *
 * Provides token estimation, cut-point finding, and budget-aware message building.
 * Zero btw-specific dependencies — designed for reuse by any pi extension.
 *
 * Port of OMP compaction algorithm (can1357/oh-my-pi compaction.ts)
 * with chars/4 heuristic instead of native BPE.
 */

import type { Message, UserMessage } from "@mariozechner/pi-ai";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Conservative safety margin — only use 80% of declared context window */
const SAFETY_MARGIN = 0.80;

/** Chars per token heuristic (conservative: overestimates tokens) */
const CHARS_PER_TOKEN = 4;

/** Default response tokens to reserve */
const DEFAULT_RESERVE_RESPONSE_TOKENS = 4096;

/** Default percentage of budget to keep as recent (verbatim) */
const DEFAULT_KEEP_RECENT_PERCENT = 0.20;

/** Default percentage of budget for /btw history */
const DEFAULT_HISTORY_PERCENT = 0.10;

/** Fallback context window when model doesn't report one */
const FALLBACK_CONTEXT_WINDOW = 200_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BudgetOptions {
	/** All messages from the conversation branch */
	branchMessages: Message[];
	/** /btw history Q&A pairs (interleaved user/assistant) */
	historyMessages: Message[];
	/** The new /btw question */
	userMessage: UserMessage;
	/** Model's context window in tokens */
	contextWindow: number;
	/** System prompt text (for token budget reservation) */
	systemPromptText: string;
	/** Max response tokens to reserve (default: 4096) */
	reserveResponseTokens?: number;
	/** Percentage of budget to keep as recent verbatim (default: 0.20) */
	keepRecentPercent?: number;
	/** Fallback if contextWindow is 0 or undefined */
	fallbackContextWindow?: number;
}

export interface BudgetResult {
	/** Final messages array to send to completeSimple */
	messages: Message[];
	/** Whether compaction was applied */
	compacted: boolean;
	/** Stats for debugging/logging */
	stats: {
		totalInputTokens: number;
		branchTokens: number;
		historyTokens: number;
		keptTokens: number;
		budgetTokens: number;
		summaryTokens: number;
	};
}

// ---------------------------------------------------------------------------
// Token estimation — chars/4 heuristic, handles all content types
// ---------------------------------------------------------------------------

/**
 * Estimate token count for a message using chars/4 heuristic.
 * Conservative: overestimates tokens to stay within budget.
 */
export function estimateTokens(message: Message): number {
	const fragments: string[] = [];

	if (typeof message.content === "string") {
		fragments.push(message.content);
	} else if (Array.isArray(message.content)) {
		for (const part of message.content) {
			if (part.type === "text") {
				fragments.push(part.text ?? "");
			} else if (part.type === "toolCall") {
				fragments.push(part.name + JSON.stringify(part.arguments ?? {}));
			} else if (part.type === "toolResult") {
				// toolResult content can be array of text/image parts
				if (typeof part.content === "string") {
					fragments.push(part.content);
				} else if (Array.isArray(part.content)) {
					for (const c of part.content) {
						if (c?.type === "text") fragments.push(c.text ?? "");
						// Images: fixed overhead estimate, not actual base64 size
						else if (c?.type === "image") fragments.push(`[image:${c.mimeType ?? "unknown"}]`);
					}
				}
			} else if (part.type === "thinking") {
				fragments.push(part.thinking ?? "");
			} else if (part.type === "image") {
				// Images: fixed overhead, don't count base64
				fragments.push(`[image:${part.mimeType ?? "unknown"}]`);
			}
		}
	}

	const totalChars = fragments.join("").length;
	return Math.ceil(totalChars / CHARS_PER_TOKEN);
}

/**
 * Sum token estimates for an array of messages.
 */
export function estimateAll(messages: Message[]): number {
	return messages.reduce((sum, m) => sum + estimateTokens(m), 0);
}

// ---------------------------------------------------------------------------
// Cut point finding — OMP-style backward walk
// ---------------------------------------------------------------------------

/**
 * Find the cut point in messages that keeps approximately `keepTokens` worth
 * of recent messages. Walks backwards from newest, accumulating tokens.
 *
 * Only cuts at `user` message boundaries (never splits toolCall↔toolResult pairs).
 *
 * Returns the index of the first message to KEEP. Everything before this index
 * is "old" (to be summarized). Everything from this index onward is "recent" (kept).
 *
 * Returns 0 if the entire conversation fits within budget (no cut needed).
 */
export function findCutPoint(messages: Message[], keepTokens: number): number {
	if (messages.length === 0) return 0;

	// Collect valid cut points: indices of user messages
	const userIndices: number[] = [];
	for (let i = 0; i < messages.length; i++) {
		if (messages[i].role === "user") {
			userIndices.push(i);
		}
	}

	if (userIndices.length === 0) {
		// No user messages — can't cut, keep everything or cut at 0
		return 0;
	}

	// Walk backwards accumulating tokens
	let accumulated = 0;
	for (let i = messages.length - 1; i >= 0; i--) {
		accumulated += estimateTokens(messages[i]);

		if (accumulated >= keepTokens) {
			// Find the closest user message at or after this index
			// (ensure we cut at a user boundary, not mid-tool-result-chain)
			for (let u = userIndices.length - 1; u >= 0; u--) {
				if (userIndices[u] >= i) {
					return userIndices[u];
				}
			}
			// All user indices are before this point — keep everything
			return 0;
		}
	}

	// Everything fits — no cut needed
	return 0;
}

// ---------------------------------------------------------------------------
// Summary message creation — lightweight VCC-lite
// ---------------------------------------------------------------------------

/**
 * Create a summary user message from old (discarded) messages.
 * Extracts text content, truncates to budget.
 */
function createSummaryMessage(oldMessages: Message[], budgetTokens: number): UserMessage | null {
	if (oldMessages.length === 0) return null;

	const parts: string[] = [];
	const budgetChars = budgetTokens * CHARS_PER_TOKEN;

	for (const msg of oldMessages) {
		let text = "";
		if (typeof msg.content === "string") {
			text = msg.content;
		} else if (Array.isArray(msg.content)) {
			text = msg.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");
		}
		if (text.trim()) {
			parts.push(`[${msg.role}]: ${text.slice(0, 500)}`);
		}
	}

	const summary = parts.join("\n\n").slice(0, budgetChars);

	if (!summary.trim()) return null;

	return {
		role: "user",
		content: [
			{
				type: "text",
				text: `[Earlier conversation summary (${oldMessages.length} messages omitted for context budget)]:\n\n${summary}`,
			},
		],
		timestamp: Date.now(),
	};
}

// ---------------------------------------------------------------------------
// Budget-aware message building
// ---------------------------------------------------------------------------

/**
 * Trim history messages from the front to fit within a token budget.
 * Removes oldest Q&A pairs first.
 */
function trimHistoryFromFront(historyMessages: Message[], budgetTokens: number): Message[] {
	if (historyMessages.length === 0) return [];

	const totalTokens = estimateAll(historyMessages);
	if (totalTokens <= budgetTokens) return historyMessages;

	// Remove pairs from the front (user+assistant = 2 messages per pair)
	// History is [user, assistant, user, assistant, ...]
	let kept = [...historyMessages];
	while (kept.length >= 2 && estimateAll(kept) > budgetTokens) {
		kept = kept.slice(2); // Remove oldest pair
	}

	return kept;
}

/**
 * Build a budgeted message array for sending to completeSimple.
 *
 * Algorithm:
 * 1. Calculate effective budget = contextWindow * SAFETY_MARGIN - reserveResponse - systemPrompt
 * 2. Trim history to fit within historyBudget
 * 3. If branch fits in remaining budget, return as-is
 * 4. Otherwise: OMP-style cut → summarize old, keep recent
 * 5. Assemble: [summary?, ...recentBranch, ...history, userMessage]
 */
export function buildBudgetedMessages(opts: BudgetOptions): BudgetResult {
	const contextWindow = opts.contextWindow || opts.fallbackContextWindow || FALLBACK_CONTEXT_WINDOW;

	if (contextWindow <= 0) {
		return {
			messages: [opts.userMessage],
			compacted: false,
			stats: {
				totalInputTokens: estimateTokens(opts.userMessage),
				branchTokens: 0,
				historyTokens: 0,
				keptTokens: estimateTokens(opts.userMessage),
				budgetTokens: 0,
				summaryTokens: 0,
			},
		};
	}

	const reserveResponse = opts.reserveResponseTokens ?? DEFAULT_RESERVE_RESPONSE_TOKENS;
	const keepRecentPercent = opts.keepRecentPercent ?? DEFAULT_KEEP_RECENT_PERCENT;
	const systemPromptTokens = estimateTokens({
		role: "system",
		content: [{ type: "text", text: opts.systemPromptText }],
	} as Message);

	const effectiveBudget = Math.floor(contextWindow * SAFETY_MARGIN) - reserveResponse - systemPromptTokens;

	if (effectiveBudget <= 0) {
		// Degraded: budget is negative or zero after deductions
		return {
			messages: [opts.userMessage],
			compacted: false,
			stats: {
				totalInputTokens: estimateTokens(opts.userMessage),
				branchTokens: 0,
				historyTokens: 0,
				keptTokens: estimateTokens(opts.userMessage),
				budgetTokens: 0,
				summaryTokens: 0,
			},
		};
	}

	// Step 1: Calculate budgets
	const historyBudget = Math.floor(effectiveBudget * DEFAULT_HISTORY_PERCENT);
	const userMsgTokens = estimateTokens(opts.userMessage);

	// Step 2: Trim history
	const trimmedHistory = trimHistoryFromFront(opts.historyMessages, historyBudget);
	const historyTokens = estimateAll(trimmedHistory);

	// Step 3: Calculate branch budget
	const branchBudget = effectiveBudget - historyTokens - userMsgTokens;

	if (branchBudget <= 0) {
		// History alone fills the budget — drop all history, try again
		const retryBranchBudget = effectiveBudget - userMsgTokens;
		if (retryBranchBudget <= 0) {
			return {
				messages: [opts.userMessage],
				compacted: true,
				stats: {
					totalInputTokens: userMsgTokens,
					branchTokens: estimateAll(opts.branchMessages),
					historyTokens: 0,
					keptTokens: userMsgTokens,
					budgetTokens: effectiveBudget,
					summaryTokens: 0,
				},
			};
		}

		// Try fitting branch without history
		const cutIndex = findCutPoint(opts.branchMessages, Math.floor(retryBranchBudget * keepRecentPercent));
		if (cutIndex === 0) {
			// Everything fits without history
			return {
				messages: [...opts.branchMessages, opts.userMessage],
				compacted: false,
				stats: {
					totalInputTokens: estimateAll(opts.branchMessages) + userMsgTokens,
					branchTokens: estimateAll(opts.branchMessages),
					historyTokens: 0,
					keptTokens: estimateAll(opts.branchMessages) + userMsgTokens,
					budgetTokens: effectiveBudget,
					summaryTokens: 0,
				},
			};
		}

		// Cut branch, summarize old portion
		const oldMessages = opts.branchMessages.slice(0, cutIndex);
		const recentMessages = opts.branchMessages.slice(cutIndex);
		const summaryBudget = retryBranchBudget - estimateAll(recentMessages);
		const summaryMsg = summaryBudget > 100 ? createSummaryMessage(oldMessages, Math.floor(summaryBudget * 0.3)) : null;

		const finalMessages: Message[] = [];
		if (summaryMsg) finalMessages.push(summaryMsg);
		finalMessages.push(...recentMessages, opts.userMessage);

		return {
			messages: finalMessages,
			compacted: true,
			stats: {
				totalInputTokens: estimateAll(finalMessages),
				branchTokens: estimateAll(opts.branchMessages),
				historyTokens: 0,
				keptTokens: estimateAll(recentMessages) + (summaryMsg ? estimateTokens(summaryMsg) : 0) + userMsgTokens,
				budgetTokens: effectiveBudget,
				summaryTokens: summaryMsg ? estimateTokens(summaryMsg) : 0,
			},
		};
	}

	// Step 4: Check if branch fits
	const branchTokens = estimateAll(opts.branchMessages);

	if (branchTokens <= branchBudget) {
		// Everything fits — no compaction needed
		return {
			messages: [...opts.branchMessages, ...trimmedHistory, opts.userMessage],
			compacted: false,
			stats: {
				totalInputTokens: branchTokens + historyTokens + userMsgTokens,
				branchTokens,
				historyTokens,
				keptTokens: branchTokens + historyTokens + userMsgTokens,
				budgetTokens: effectiveBudget,
				summaryTokens: 0,
			},
		};
	}

	// Step 5: OMP-style cut
	const keepRecentTokens = Math.floor(branchBudget * keepRecentPercent);
	const cutIndex = findCutPoint(opts.branchMessages, keepRecentTokens);

	if (cutIndex === 0) {
		// No valid cut point found — keep everything (will overflow, but at least won't crash)
		return {
			messages: [...opts.branchMessages, ...trimmedHistory, opts.userMessage],
			compacted: false,
			stats: {
				totalInputTokens: branchTokens + historyTokens + userMsgTokens,
				branchTokens,
				historyTokens,
				keptTokens: branchTokens + historyTokens + userMsgTokens,
				budgetTokens: effectiveBudget,
				summaryTokens: 0,
			},
		};
	}

	// Step 6: Summarize old portion
	const oldMessages = opts.branchMessages.slice(0, cutIndex);
	const recentMessages = opts.branchMessages.slice(cutIndex);
	const recentTokens = estimateAll(recentMessages);
	const summaryBudget = branchBudget - recentTokens;
	const summaryMsg = summaryBudget > 100 ? createSummaryMessage(oldMessages, Math.floor(summaryBudget * 0.3)) : null;

	// Step 7: Assemble final
	const finalMessages: Message[] = [];
	if (summaryMsg) finalMessages.push(summaryMsg);
	finalMessages.push(...recentMessages, ...trimmedHistory, opts.userMessage);

	return {
		messages: finalMessages,
		compacted: true,
		stats: {
			totalInputTokens: estimateAll(finalMessages),
			branchTokens,
			historyTokens,
			keptTokens: recentTokens + historyTokens + (summaryMsg ? estimateTokens(summaryMsg) : 0) + userMsgTokens,
			budgetTokens: effectiveBudget,
			summaryTokens: summaryMsg ? estimateTokens(summaryMsg) : 0,
		},
	};
}
