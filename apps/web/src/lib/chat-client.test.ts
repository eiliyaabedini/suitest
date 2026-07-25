import { afterEach, describe, expect, it, vi } from "vitest";

import { streamChat } from "@/lib/chat-client";

describe("streamChat", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("cancels the response body when an oversized SSE frame is received", async () => {
    let cancelled = false;
    const oversizedDelta = "x".repeat(1024 * 1024 + 1);
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(`event: token\ndata: {"delta":"${oversizedDelta}"}`),
        );
      },
      cancel() {
        cancelled = true;
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(body, {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        }),
      ),
    );

    await expect(streamChat([{ role: "user", content: "hello" }], {})).rejects.toThrow(
      /size limit/i,
    );
    expect(cancelled).toBe(true);
  });
});
