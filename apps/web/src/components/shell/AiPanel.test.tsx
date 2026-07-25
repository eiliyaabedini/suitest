import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AiPanel } from "@/components/shell/AiPanel";
import { useCapabilities, type Capabilities } from "@/stores/use-capabilities";

const streamChatMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/chat-client", () => ({
  streamChat: streamChatMock,
}));

const ZERO_CAPS: Capabilities = {
  tier: "ZERO",
  llm: { provider: "none", model: null, base_url: null, is_test_provider: false },
  embeddings: { enabled: false, backend: "none", model: null, dim: null },
  features: {
    manual_tcm: true,
    deterministic_runner: true,
    deterministic_generator_openapi: true,
    deterministic_generator_recorder: true,
    deterministic_generator_crawler: true,
    ai_generation: false,
    ai_execution_agentic: false,
    ai_diagnose: false,
    ai_conversation: false,
    semantic_search: false,
    fts_search: true,
    auto_defect_filing_ai: false,
    auto_defect_filing_rule: true,
  },
  autonomy: { available: ["manual"], default: "manual" },
  mcpProviders: [],
  version: "1.0.0",
};

const CLOUD_ASSIST_CAPS: Capabilities = {
  tier: "CLOUD",
  llm: {
    provider: "anthropic",
    model: "claude-sonnet-4-5",
    base_url: null,
    is_test_provider: false,
  },
  embeddings: { enabled: true, backend: "openai", model: "text-embedding-3-small", dim: 1536 },
  features: {
    manual_tcm: true,
    deterministic_runner: true,
    deterministic_generator_openapi: true,
    deterministic_generator_recorder: true,
    deterministic_generator_crawler: true,
    ai_generation: true,
    ai_execution_agentic: true,
    ai_diagnose: true,
    ai_conversation: true,
    semantic_search: true,
    fts_search: true,
    auto_defect_filing_ai: true,
    auto_defect_filing_rule: true,
  },
  autonomy: { available: ["manual", "assist", "semi_auto", "auto"], default: "assist" },
  mcpProviders: [],
  version: "1.0.0",
};

function setCaps(caps: Capabilities): void {
  act(() => {
    useCapabilities.setState({ capabilities: caps, loading: false, error: null });
  });
}

describe("<AiPanel>", () => {
  beforeEach(() => {
    streamChatMock.mockReset();
    streamChatMock.mockResolvedValue(undefined);
    act(() => {
      useCapabilities.setState({ capabilities: null, loading: true, error: null });
    });
  });
  afterEach(() => {
    act(() => {
      useCapabilities.setState({ capabilities: null, loading: true, error: null });
    });
  });

  it("renders nothing in ZERO tier (ai_conversation disabled)", () => {
    setCaps(ZERO_CAPS);
    const { container } = render(<AiPanel />);
    expect(container.textContent).toBe("");
    expect(screen.queryByTestId("ai-panel")).toBeNull();
  });

  it("renders the panel in CLOUD tier with provider + model + autonomy subtitle", () => {
    setCaps(CLOUD_ASSIST_CAPS);
    render(<AiPanel />);
    expect(screen.getByTestId("ai-panel")).toBeInTheDocument();
    expect(screen.getByText("Suitest Agent")).toBeInTheDocument();
    expect(screen.getByTestId("ai-panel-subtitle")).toHaveTextContent(
      "anthropic:claude-sonnet-4-5 · assist",
    );
  });

  it("renders the empty-thread agent greeting", () => {
    setCaps(CLOUD_ASSIST_CAPS);
    render(<AiPanel />);
    expect(screen.getByTestId("ai-panel-thread")).toHaveTextContent(/I stream answers live/i);
  });

  it("renders an enabled composer (send gated until input typed)", () => {
    setCaps(CLOUD_ASSIST_CAPS);
    render(<AiPanel />);
    const input = screen.getByTestId("ai-panel-composer-input");
    expect(input).not.toBeDisabled();
    expect(input).toHaveAttribute("placeholder", "Ask the agent…");
    // Send is disabled until the user types something.
    expect(screen.getByTestId("ai-panel-send")).toBeDisabled();
  });

  it("stops an in-flight stream by aborting its transport", async () => {
    let signal: AbortSignal | undefined;
    streamChatMock.mockImplementation(
      (_messages: unknown, _handlers: unknown, nextSignal?: AbortSignal) =>
        new Promise<void>((resolve) => {
          signal = nextSignal;
          nextSignal?.addEventListener("abort", () => resolve(), { once: true });
        }),
    );
    setCaps(CLOUD_ASSIST_CAPS);
    render(<AiPanel />);
    const user = userEvent.setup();

    await user.type(screen.getByTestId("ai-panel-composer-input"), "Check the last run");
    await user.click(screen.getByTestId("ai-panel-send"));
    await user.click(await screen.findByRole("button", { name: "Stop generating" }));

    expect(signal?.aborted).toBe(true);
  });

  it("aborts an in-flight stream when the panel unmounts", async () => {
    let signal: AbortSignal | undefined;
    streamChatMock.mockImplementation(
      (_messages: unknown, _handlers: unknown, nextSignal?: AbortSignal) =>
        new Promise<void>((resolve) => {
          signal = nextSignal;
          nextSignal?.addEventListener("abort", () => resolve(), { once: true });
        }),
    );
    setCaps(CLOUD_ASSIST_CAPS);
    const { unmount } = render(<AiPanel />);
    const user = userEvent.setup();

    await user.type(screen.getByTestId("ai-panel-composer-input"), "Check the last run");
    await user.click(screen.getByTestId("ai-panel-send"));
    await screen.findByRole("button", { name: "Stop generating" });
    unmount();

    expect(signal?.aborted).toBe(true);
  });

  it("renders nothing while capabilities are still loading", () => {
    // beforeEach already sets capabilities=null; do nothing.
    const { container } = render(<AiPanel />);
    expect(container.textContent).toBe("");
  });
});
