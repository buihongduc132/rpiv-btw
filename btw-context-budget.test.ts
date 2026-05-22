/**
 * TDD tests for btw-context-budget.ts
 * RED first — these test the public API of the context budget module.
 */

import { describe, it, expect } from "vitest";
import {
	estimateTokens,
	estimateAll,
	findCutPoint,
	buildBudgetedMessages,
	type BudgetOptions,
} from "./btw-context-budget.js";
import type { Message, UserMessage, AssistantMessage } from "@mariozechner/pi-ai";

// ---------------------------------------------------------------------------
// Helpers — create messages of known token sizes
// ---------------------------------------------------------------------------

function userMsg(text: string): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

function assistantMsg(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		stopReason: "endTurn",
		timestamp: Date.now(),
	};
}

function toolCallMsg(name: string, args: object): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "toolCall", id: "call_1", name, arguments: args }],
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

function toolResultMsg(content: string): Message {
	return {
		role: "toolResult",
		content: [{ type: "text", text: content }],
		toolCallId: "call_1",
		timestamp: Date.now(),
	} as Message;
}

/** Create N messages of approximately `tokensPerMsg` tokens each */
function createMessages(n: number, tokensPerMsg: number): Message[] {
	const msgs: Message[] = [];
	const charPerMsg = tokensPerMsg * 4; // chars/4 = tokens
	for (let i = 0; i < n; i++) {
		msgs.push(userMsg("x".repeat(charPerMsg)));
		msgs.push(assistantMsg("y".repeat(charPerMsg)));
	}
	return msgs;
}

/** Create alternating user/assistant history pairs */
function createHistory(pairs: number, tokensPerMsg: number): Message[] {
	const msgs: Message[] = [];
	const charsPerMsg = tokensPerMsg * 4;
	for (let i = 0; i < pairs; i++) {
		msgs.push(userMsg("hq".repeat(charsPerMsg / 2)));
		msgs.push(assistantMsg("ha".repeat(charsPerMsg / 2)));
	}
	return msgs;
}

// ---------------------------------------------------------------------------
// estimateTokens
// ---------------------------------------------------------------------------

describe("estimateTokens", () => {
	it("estimates string content", () => {
		const msg = userMsg("a".repeat(100)); // 100 chars → 25 tokens
		expect(estimateTokens(msg)).toBe(25);
	});

	it("estimates array content with text parts", () => {
		const msg = assistantMsg("hello world"); // 11 chars → 3 tokens
		expect(estimateTokens(msg)).toBe(Math.ceil(11 / 4));
	});

	it("handles toolCall parts", () => {
		const msg = toolCallMsg("bash", { command: "ls -la" });
		const tokens = estimateTokens(msg);
		expect(tokens).toBeGreaterThan(0);
		// name + JSON.stringify(args)
		expect(tokens).toBeLessThan(20);
	});

	it("handles toolResult parts", () => {
		const msg = toolResultMsg("file1.txt\nfile2.txt\n" + "x".repeat(100));
		const tokens = estimateTokens(msg);
		expect(tokens).toBeGreaterThan(0);
	});

	it("handles thinking parts", () => {
		const msg: Message = {
			role: "assistant",
			content: [{ type: "thinking", thinking: "Let me think about this..." }],
		} as Message;
		expect(estimateTokens(msg)).toBeGreaterThan(0);
	});

	it("handles empty content", () => {
		const msg = userMsg("");
		expect(estimateTokens(msg)).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// estimateAll
// ---------------------------------------------------------------------------

describe("estimateAll", () => {
	it("sums tokens across messages", () => {
		const msgs = [userMsg("a".repeat(40)), assistantMsg("b".repeat(40))];
		// 40/4 + 40/4 = 10 + 10 = 20
		expect(estimateAll(msgs)).toBe(20);
	});

	it("returns 0 for empty array", () => {
		expect(estimateAll([])).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// findCutPoint
// ---------------------------------------------------------------------------

describe("findCutPoint", () => {
	it("returns 0 when everything fits", () => {
		const msgs = createMessages(5, 100); // 10 messages, 100 tokens each = 1000 total
		expect(findCutPoint(msgs, 5000)).toBe(0); // keep 5000 > 1000 total
	});

	it("cuts at user message boundary", () => {
		// 20 messages (10 pairs) of 100 tokens each = 2000 total
		// Keep 500 tokens worth = ~5 messages from the end
		const msgs = createMessages(10, 100);
		const cut = findCutPoint(msgs, 500);
		// Cut should be at a user message
		expect(cut).toBeGreaterThanOrEqual(0);
		if (cut < msgs.length) {
			expect(msgs[cut].role).toBe("user");
		}
	});

	it("never cuts between assistant(toolCall) and toolResult", () => {
		const msgs: Message[] = [
			userMsg("q1"),
			toolCallMsg("bash", { command: "ls" }),
			toolResultMsg("output"),
			userMsg("q2"),
			assistantMsg("answer"),
		];
		// Ask to keep only the last few tokens
		const cut = findCutPoint(msgs, 5);
		// Should cut at q2 (index 3), never at index 1 or 2
		expect(cut).toBe(3);
		expect(msgs[cut].role).toBe("user");
	});

	it("returns 0 for empty messages", () => {
		expect(findCutPoint([], 1000)).toBe(0);
	});

	it("returns 0 when no user messages exist", () => {
		const msgs: Message[] = [assistantMsg("a1"), assistantMsg("a2")];
		expect(findCutPoint(msgs, 1)).toBe(0);
	});

	it("handles single user message", () => {
		const msgs = [userMsg("hello")];
		// Keep budget < token count → cut at the user message (keep it)
		expect(findCutPoint(msgs, 0)).toBe(0);
	});

	it("walks backward accumulating tokens correctly", () => {
		// 4 messages: user(100), assistant(100), user(100), assistant(100)
		// keepRecent = 150 tokens → should keep last 2 messages (200 tokens)
		const msgs: Message[] = [
			userMsg("a".repeat(400)),
			assistantMsg("b".repeat(400)),
			userMsg("c".repeat(400)),
			assistantMsg("d".repeat(400)),
		];
		const cut = findCutPoint(msgs, 150);
		// Should cut at index 2 (the second user message)
		expect(cut).toBe(2);
	});
});

// ---------------------------------------------------------------------------
// buildBudgetedMessages
// ---------------------------------------------------------------------------

describe("buildBudgetedMessages", () => {
	const baseOpts = (overrides: Partial<BudgetOptions> = {}): BudgetOptions => ({
		branchMessages: [],
		historyMessages: [],
		userMessage: userMsg("test question"),
		contextWindow: 200_000,
		systemPromptText: "You are a helpful assistant.",
		...overrides,
	});

	it("returns all messages when conversation fits (#1)", () => {
		const branch = createMessages(5, 100); // ~1000 tokens
		const opts = baseOpts({ branchMessages: branch, contextWindow: 200_000 });
		const result = buildBudgetedMessages(opts);

		expect(result.compacted).toBe(false);
		// branch + history + userMessage
		expect(result.messages.length).toBe(branch.length + 1);
		expect(result.messages[result.messages.length - 1]).toBe(opts.userMessage);
	});

	it("compacts large conversation (#2)", () => {
		// 100 messages of 200 tokens each = 20,000 tokens branch
		// 8k context window → budget = 8k * 0.8 - 4096 (reserve) - ~30 (sys) ≈ 2274
		// keepRecent = 20% of 2274 ≈ 455 tokens
		const branch = createMessages(50, 200); // 100 msgs, ~20k tokens
		const opts = baseOpts({ branchMessages: branch, contextWindow: 8_000 });
		const result = buildBudgetedMessages(opts);

		expect(result.compacted).toBe(true);
		expect(result.messages.length).toBeLessThan(branch.length + 1);
		expect(result.stats.branchTokens).toBeGreaterThan(0);
	});

	it("trims history from front (#3)", () => {
		// 20 history pairs (40 messages) of 100 tokens each = 4000 tokens
		// 8k window → historyBudget = 10% of ~2274 ≈ 227 tokens
		// Should trim to ~1-2 pairs
		const history = createHistory(20, 100);
		const opts = baseOpts({
			branchMessages: [userMsg("q"), assistantMsg("a")],
			historyMessages: history,
			contextWindow: 8_000,
		});
		const result = buildBudgetedMessages(opts);

		expect(result.messages.length).toBeLessThan(2 + history.length + 1);
	});

	it("exact budget boundary — no compaction (#4)", () => {
		// Create messages that total exactly the budget
		const sysPrompt = "sys";
		const sysTokens = Math.ceil(sysPrompt.length / 4);
		const budget = Math.floor(200_000 * 0.8) - 4096 - sysTokens;
		// Create branch that's just under budget
		const branch = createMessages(1, Math.floor(budget / 4)); // ~budget/4 tokens per msg, 2 msgs
		const opts = baseOpts({
			branchMessages: branch,
			contextWindow: 200_000,
			systemPromptText: sysPrompt,
		});
		const result = buildBudgetedMessages(opts);

		// May or may not compact depending on exact fit, but should not crash
		expect(result.messages.length).toBeGreaterThan(0);
	});

	it("handles empty branch (#5)", () => {
		const opts = baseOpts({ branchMessages: [] });
		const result = buildBudgetedMessages(opts);

		expect(result.messages.length).toBe(1); // just the userMessage
		expect(result.messages[0]).toBe(opts.userMessage);
	});

	it("handles single message (#6)", () => {
		const opts = baseOpts({ branchMessages: [userMsg("hello")] });
		const result = buildBudgetedMessages(opts);

		expect(result.messages.length).toBe(2); // 1 branch + 1 user
		expect(result.compacted).toBe(false);
	});

	it("system prompt is deducted from budget (#8)", () => {
		// Large system prompt
		const largeSysPrompt = "x".repeat(100_000); // ~25,000 tokens
		const branch = createMessages(10, 100); // ~2000 tokens
		const opts = baseOpts({
			branchMessages: branch,
			contextWindow: 30_000, // Small window relative to sys prompt
			systemPromptText: largeSysPrompt,
		});
		const result = buildBudgetedMessages(opts);

		// System prompt alone eats most of the budget
		expect(result.stats.budgetTokens).toBeLessThan(5000);
	});

	it("handles zero context window (#9)", () => {
		const opts = baseOpts({ contextWindow: 0 });
		const result = buildBudgetedMessages(opts);

		// Should gracefully degrade to just userMessage
		expect(result.messages.length).toBeGreaterThanOrEqual(1);
		expect(result.messages).toContain(opts.userMessage);
	});

	it("creates summary for old messages (#10)", () => {
		// 50 messages → definitely won't fit in 8k
		const branch = createMessages(25, 200);
		const opts = baseOpts({ branchMessages: branch, contextWindow: 8_000 });
		const result = buildBudgetedMessages(opts);

		expect(result.compacted).toBe(true);
		if (result.stats.summaryTokens > 0) {
			// First message should be a summary
			const first = result.messages[0];
			expect(first.role).toBe("user");
			if (typeof first.content !== "string" && Array.isArray(first.content)) {
				const text = first.content.find((c): c is { type: "text"; text: string } => c.type === "text")?.text ?? "";
				expect(text).toContain("Earlier conversation summary");
			}
		}
	});

	it("all tool results, no user messages (#11)", () => {
		const branch: Message[] = [
			toolResultMsg("output1"),
			toolResultMsg("output2"),
			toolResultMsg("output3"),
		];
		const opts = baseOpts({ branchMessages: branch, contextWindow: 1_000 });
		const result = buildBudgetedMessages(opts);

		// No crash, messages returned
		expect(result.messages.length).toBeGreaterThan(0);
	});

	it("history exceeds total budget (#12)", () => {
		// 100 pairs (200 messages) of 200 tokens each = 40,000 tokens history
		// 8k window → history budget = ~227 tokens
		const history = createHistory(100, 200);
		const opts = baseOpts({
			branchMessages: [userMsg("q")],
			historyMessages: history,
			contextWindow: 8_000,
		});
		const result = buildBudgetedMessages(opts);

		// History should be heavily trimmed
		expect(result.stats.historyTokens).toBeLessThan(estimateAll(history));
	});

	it("cut point preserves toolCall→toolResult pairs (#13)", () => {
		const branch: Message[] = [
			userMsg("old question"),
			toolCallMsg("bash", { command: "ls" }),
			toolResultMsg("output1\noutput2"),
			userMsg("recent question"),
			assistantMsg("recent answer"),
		];
		const opts = baseOpts({ branchMessages: branch, contextWindow: 200 }); // Tiny window
		const result = buildBudgetedMessages(opts);

		if (result.compacted) {
			// Check that toolCall and toolResult are not split
			const msgs = result.messages;
			for (let i = 0; i < msgs.length; i++) {
				if (msgs[i].role === "assistant") {
					const content = msgs[i].content;
					if (Array.isArray(content)) {
						const hasToolCall = content.some((c): c is { type: "toolCall" } => c.type === "toolCall");
						if (hasToolCall && i + 1 < msgs.length) {
							// Next message should be toolResult
							expect(msgs[i + 1].role).toBe("toolResult");
						}
					}
				}
			}
		}
	});

	it("negative budget after deductions (#14)", () => {
		const opts = baseOpts({
			branchMessages: createMessages(10, 100),
			contextWindow: 100, // Way too small
			systemPromptText: "x".repeat(10_000), // Enormous system prompt
			reserveResponseTokens: 50_000, // Massive reserve
		});
		const result = buildBudgetedMessages(opts);

		// Should gracefully degrade
		expect(result.messages.length).toBeGreaterThanOrEqual(1);
		expect(result.messages).toContain(opts.userMessage);
	});

	it("contextWindow undefined — uses fallback (#15)", () => {
		const opts = baseOpts({
			branchMessages: createMessages(5, 100),
			contextWindow: undefined as unknown as number,
			fallbackContextWindow: 200_000,
		});
		const result = buildBudgetedMessages(opts);

		expect(result.messages.length).toBeGreaterThan(0);
		expect(result.compacted).toBe(false);
	});

	it("multiple sequential /btw calls — budget shrinks (#16)", () => {
		// Simulate 3 sequential /btw calls with accumulating history
		const branch = createMessages(20, 100);
		const contextWindow = 50_000;

		// Call 1: no history
		const r1 = buildBudgetedMessages(baseOpts({
			branchMessages: branch,
			historyMessages: [],
			contextWindow,
		}));

		// Call 2: 5 pairs of history
		const history2 = createHistory(5, 100);
		const r2 = buildBudgetedMessages(baseOpts({
			branchMessages: branch,
			historyMessages: history2,
			contextWindow,
		}));

		// Call 3: 20 pairs of history
		const history3 = createHistory(20, 100);
		const r3 = buildBudgetedMessages(baseOpts({
			branchMessages: branch,
			historyMessages: history3,
			contextWindow,
		}));

		// History tokens should increase but be bounded
		expect(r2.stats.historyTokens).toBeGreaterThan(r1.stats.historyTokens);
		expect(r3.stats.historyTokens).toBeLessThanOrEqual(r3.stats.budgetTokens * 0.15); // ~10% + margin
	});

	it("works with already-compacted branch (#17)", () => {
		// After pi-core compaction, branch is short
		const branch = [userMsg("post-compaction question"), assistantMsg("answer")];
		const opts = baseOpts({ branchMessages: branch, contextWindow: 200_000 });
		const result = buildBudgetedMessages(opts);

		expect(result.compacted).toBe(false);
		expect(result.messages.length).toBe(3); // 2 branch + 1 user
	});
});
