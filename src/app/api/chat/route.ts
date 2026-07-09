
/**
 * CHAT MODES CONVENTION DOCUMENTATION:
 * 
 * 1. EXPLORE MODE (default):
 *    - Gemini responds with plain prose text (max 2 lines).
 *    - It appends a tag `[Search: <query>]` at the end to trigger the product carousel.
 *    - Example: "Here are some awesome jackets. [Search: jackets]"
 *    - If a complex query is sent, it can append `[SuggestMode: deep_research]`.
 * 
 * 2. DEEP RESEARCH MODE:
 *    - Gemini returns ONLY a raw JSON payload (no markdown code blocks, no other text).
 *    - Clarifying Question payload format:
 *      {
 *        "type": "clarifying_question",
 *        "acknowledgement": "A short comment on their answer...",
 *        "question": "The question string",
 *        "options": ["Option A", "Option B", "Option C"],
 *        "allow_skip": true,
 *        "allow_custom": true
 *      }
 *    - Final Results payload format:
 *      {
 *        "type": "results",
 *        "acknowledgement": "Hero products found...",
 *        "primary_query": "Urban Classic Black Denim Jacket",
 *        "backup_queries": ["Minimalist White Cotton Tee", "Vintage Wash Relaxed Jeans"]
 *      }
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";
import { env } from "@/lib/env";
import { getFallbackChatResponse } from "@/lib/fallbackResponses";
import { enforceChatAccess } from "@/lib/chatAccess";
import fs from "fs";
import path from "path";
import os from "os";
import { createClient } from "@/lib/supabase/server";
import { determineIntent } from "@/lib/agents/router";
import { searchForProducts, type SearchedProduct } from "@/lib/agents/search";
import { runWriter } from "@/lib/agents/writer";
import { executeRerankedSearch } from "@/lib/providers/test-serper";
import type { RerankedContext } from "@/lib/retrieval/index";
import { executeGenerativeOrchestration } from "@/lib/guardrails/apiOrchestrator";
import { checkSemanticCache, storeInSemanticCache } from "@/lib/caching/semanticCache";
import { getNextGroqKey } from "@/lib/agents/keyManager";

const genAI = new GoogleGenerativeAI(env.GEMINI_API_KEYS[0]);

export const maxDuration = 60;

export async function POST(req: NextRequest) {
  let userMessage = "";
  let userHistory: ChatHistoryMessage[] = [];
  let mode = "explore";

  try {
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      return NextResponse.json(
        { error: "Invalid request: JSON body is malformed or empty." },
        { status: 400 }
      );
    }

    // Guardrail: Return empty chunked stream silently for empty array pings to stop frontend infinite retry locks
    if (!isChatRequestBody(body) || !body.message || !body.message.trim()) {
      const emptyStream = new ReadableStream({
        start(controller) { controller.close(); }
      });
      return new Response(emptyStream, {
        headers: { "Content-Type": "text/plain; charset=utf-8", "Transfer-Encoding": "chunked" }
      });
    }

    // Validate history if provided
    if (body.history !== undefined && !Array.isArray(body.history)) {
      return NextResponse.json(
        { error: "Invalid request: 'history' must be an array if provided." },
        { status: 400 }
      );
    }

    let { chatId = "default", history = [], message, mode: bodyMode = "explore", requirements = {}, isRegenerate = false } = body;
    mode = bodyMode;
    userMessage = message;

    // We respect the mode passed from the frontend (bodyMode) because the user explicitly toggled it.
    // Do NOT override it with the router agent, otherwise Deep Research mode gets hijacked into Explore mode.

    // Backend JSON file logic (Msg Quality Improvement & History Tracking like Gemini Ecosystem)
    const baseDir = process.env.NODE_ENV === "production" ? os.tmpdir() : process.cwd();
    const chatDataDir = path.join(baseDir, "chat_data");
    try {
      if (!fs.existsSync(chatDataDir)) {
        fs.mkdirSync(chatDataDir);
      }
    } catch (fsErr) {
      console.warn("Failed to create chatDataDir, continuing without file persistence:", fsErr);
    }
    const contextFilePath = path.join(chatDataDir, `${chatId}.json`);

    let backendContext: {
      chatId: string;
      messageCount: number;
      lastActive: string;
      history: ChatHistoryMessage[];
      context: Record<string, any>;
      memory: {
        userLanguagePreference?: string;
        generalProfileSummary?: string;
        preferredCategories?: string[];
      };
    } = {
      chatId,
      messageCount: history.length,
      lastActive: new Date().toISOString(),
      history: history || [],
      context: requirements || {},
      memory: {}
    };

    if (fs.existsSync(contextFilePath)) {
      try {
        const fileContent = fs.readFileSync(contextFilePath, "utf8");
        const parsed = JSON.parse(fileContent);
        backendContext = {
          chatId: parsed.chatId || chatId,
          messageCount: parsed.messageCount ?? history.length,
          lastActive: parsed.lastActive || new Date().toISOString(),
          history: isRegenerate ? (history || []) : (parsed.history || history || []),
          context: { ...parsed.context, ...requirements },
          memory: parsed.memory || {}
        };
      } catch (e) {
        console.warn("Failed to parse backend JSON file:", e);
      }
    }

    // Append current user message to backend history if not present
    const lastMsg = backendContext.history[backendContext.history.length - 1];
    if (!lastMsg || lastMsg.content !== userMessage || lastMsg.role !== "user") {
      backendContext.history.push({ role: "user", content: userMessage });
    }

    // Use full history from the JSON file to feed the agent
    userHistory = backendContext.history;
    backendContext.messageCount = userHistory.length;
    backendContext.lastActive = new Date().toISOString();

    // Dynamically detect language from the user's prompt to store in memory
    const userPromptLower = userMessage.toLowerCase();
    const hindiKeywords = ["kya", "hai", "chahiye", "batao", "krde", "thik", "mujhe", "dekha", "kaise", "nahi", "kyon", "bhai", "samjhao"];
    if (hindiKeywords.some(word => userPromptLower.includes(word))) {
      backendContext.memory.userLanguagePreference = "Hinglish/Hindi";
    } else if (userPromptLower.trim().split(/\s+/).length >= 2) {
      backendContext.memory.userLanguagePreference = "English";
    }

    // Update memory summary based on categories or items searched
    if (requirements.category) {
      if (!backendContext.memory.preferredCategories) {
        backendContext.memory.preferredCategories = [];
      }
      if (!backendContext.memory.preferredCategories.includes(String(requirements.category))) {
        backendContext.memory.preferredCategories.push(String(requirements.category));
      }
    }

    // Save current state before LLM call
    try {
      fs.writeFileSync(contextFilePath, JSON.stringify(backendContext, null, 2));
    } catch (e) {
      console.warn("Could not save context file before LLM:", e);
    }

    const access = await enforceChatAccess(req);
    if (access.response) {
      return access.response;
    }
    // access.isGuest indicates whether this is an unauthenticated guest request.

    // Map frontend conversation history ({ role: "assistant"|"user", content })
    // to the structure expected by the Gemini SDK ({ role: "model"|"user", parts: [{ text }] })
    const formattedHistory = history.map((msg) => {
      const role = msg.role === "assistant" ? "model" : "user";
      return {
        role,
        parts: [{ text: msg.content || "" }]
      };
    });

    if (mode === "buy_explanation") {
      try {
        const product = JSON.parse(userMessage);
        const explanationPrompt = `The user selected the product "${product.name}" priced at ${product.price}. Their stated need was their previous conversation context. Write a short, genuine 2-3 sentence explanation of why this specific product is a good fit for their stated need, referencing whatever real specs/features are available from the product title/description. Be honest — if the product isn't a perfect fit, say so briefly rather than oversell it. Output your response as a valid JSON object in this format: { "ui_type": "text_response", "text": "Your explanation here" }`;

        const text = await executeGenerativeOrchestration({
          systemInstruction: explanationPrompt,
          formattedHistory,
          effectiveUserMessage: "Provide explanation",
          groqApiKey: getNextGroqKey(),
          historyForGroq: history.map((m) => ({ role: m.role ?? "user", content: m.content ?? "" }))
        });

        // The orchestrator already parses and validates the output using Zod, ensuring structural determinism
        const parsedData = JSON.parse(text);
        if (parsedData && parsedData.ui_type === "text_response") {
          return NextResponse.json({ text: [text], products: null });
        } else {
          throw new Error("Invalid format from Generative Orchestrator");
        }
      } catch (e) {
        console.error("Failed to generate buy explanation:", e);
        let product: any = { name: "this product" };
        try { product = JSON.parse(userMessage); } catch { }
        const fallback = JSON.stringify({
          ui_type: "text_response",
          text: `Great choice! ${product.name} is a solid pick within your budget.`
        });
        return NextResponse.json({ text: [fallback], products: null });
      }
    }

    // ── Step 1: Router determines mode ──────────────────
    const originalMode = mode;
    // Removed automatic routing to 'deep_research'.
    // The user explicitly selects the mode, and if not selected, it defaults to 'explore'.

    // ── Step 2: Cache check ───────────────────────────────────────────────────
    // Skip cache for regenerate requests and buy_explanation (handled above).
    // Also skip cache temporarily to flush out broken cached search_intent JSONs.
    if (!isRegenerate && false) {
      const cacheResult = await checkSemanticCache(userMessage, chatId);
      const payload = cacheResult.payload;
      if (cacheResult.type === "HARD_HIT" && payload) {
        return NextResponse.json({
          text: payload!.responseTexts,
          products: payload!.products,
          cached: true,
        });
      }
    }

    // ── Step 3: Search & Deep Reranking (only in deep_research) ───────────────
    // In explore mode the writer handles inline search via search_intent.
    // In deep_research mode we only search when the user has already supplied
    // both use-case and budget (i.e. requirements is non-empty).

    // ── Step 3: Search & Deep Reranking ───────────────
    let searchResults: SearchedProduct[] = [];
    let rerankedContext: RerankedContext | null = null;
    const requirementsReady =
      mode === "deep_research" &&
      requirements &&
      typeof requirements === "object" &&
      Object.keys(requirements).length >= 2;

    let shoppingSearchFailed = false;

    if (requirementsReady || mode === "explore") {
      const query = mode === "deep_research"
        ? [
          requirements?.category,
          requirements?.use_case,
          requirements?.budget ? `under ${requirements.budget}` : "",
        ].filter(Boolean).join(" ")
        : userMessage;

      try {
        searchResults = await searchForProducts(query, 6);
      } catch (searchErr) {
        shoppingSearchFailed = true;
        console.warn("[route] Shopping API search failed:", searchErr);
      }

      if (mode === "deep_research") {
        try {
          // Execute Deep Web Scraping & Reranking Pipeline
          rerankedContext = await executeRerankedSearch(query);
          if (rerankedContext && rerankedContext.error) {
            console.warn("[route] Scraper/Organic search failed (partial fallback active):", rerankedContext.error);
          }
        } catch (searchErr) {
          console.warn("[route] Pre-search failed unexpectedly:", searchErr);
        }
      }
      // If Shopping API completely failed or returned 0 results, we cannot provide product options.
      // We will allow it to continue to the writer agent, which can handle empty products gracefully
      // (either by triggering a second-pass search in explore mode, or outputting a graceful fallback in deep research).
      if (shoppingSearchFailed) {
        console.warn("[route] Search failed, but proceeding to writer agent for graceful fallback.");
      }
    }

    // ── Step 4: Writer produces the final structured response (STREAMING) ─────────
    const { runStreamingWriter } = await import("@/lib/agents/writer");
    const writerStream = runStreamingWriter({
      mode: mode === "deep_research" ? "deep_research" : "explore",
      userMessage,
      history: userHistory.map((m) => ({ role: m.role ?? "user", content: m.content ?? "" })),
      products: searchResults,
      sessionContext: {
        ...backendContext.context,
        ...requirements,
        chatMemory: backendContext.memory,
        reranked_context: rerankedContext
          ? {
            primary: rerankedContext.primary.map(c => c.text),
            secondary: rerankedContext.secondary.map(c => c.text)
          }
          : undefined
      },
      isRegenerate,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        let fullResponse = "";

        // Send initial metadata (products array and/or mode override for frontend)
        const initialMetadata: any = { type: 'metadata' };
        if (searchResults && searchResults.length > 0) {
          initialMetadata.products = searchResults;
        }
        if (mode !== originalMode) {
          initialMetadata.mode = mode;
        }
        if (initialMetadata.products || initialMetadata.mode) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(initialMetadata)}\n\n`));
        }

        try {
          let dataTagFound = false;
          let jsonBuffer = "";

          for await (const chunk of writerStream) {
            const chunkText = chunk || "";
            fullResponse += chunkText;
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'chunk', text: chunkText })}\n\n`));
          }

          // Once the stream finishes, save the full response to context
          let assistantCleanText = "";
          let confirmedCategory: string | null = null;
          let fullJsonResponse = fullResponse;
          const jsonBlockMatch = fullResponse.match(/```(?:json)?\s*([\s\S]*?)```/i);
          if (jsonBlockMatch && jsonBlockMatch[1]) {
            fullJsonResponse = jsonBlockMatch[1].trim();
          } else {
            const firstBrace = fullResponse.indexOf('{');
            const firstBracket = fullResponse.indexOf('[');
            let startIndex = -1;
            if (firstBrace !== -1 && firstBracket !== -1) startIndex = Math.min(firstBrace, firstBracket);
            else if (firstBrace !== -1) startIndex = firstBrace;
            else if (firstBracket !== -1) startIndex = firstBracket;
            
            if (startIndex !== -1) {
              fullJsonResponse = fullResponse.substring(startIndex).replace(/```$/, "").trim();
            } else {
              fullJsonResponse = fullResponse.replace(/^```(?:json)?\s*/i, "").replace(/```$/, "").trim();
            }
          }
          try {
            const parsedRes = JSON.parse(fullJsonResponse);
            if (parsedRes.fingerprint && parsedRes.fingerprint.language) {
              backendContext.memory.userLanguagePreference = parsedRes.fingerprint.language;
            }
            if (typeof parsedRes.confirmed_category === "string") {
              confirmedCategory = parsedRes.confirmed_category;
            }
            assistantCleanText = String(parsedRes.headline || parsedRes.text || parsedRes.summary || "");
          } catch (err) {
            const match = fullResponse.match(/"(?:text|headline|summary)"\s*:\s*"((?:[^"\\]|\\.)*)"/);
            if (match && match[1]) {
              assistantCleanText = match[1].replace(/\\n/g, "\n").replace(/\\"/g, '"');
            } else {
              assistantCleanText = fullResponse.replace(/[{}"[\]]/g, "").replace(/\b(?:ui_type|text|thought|fingerprint|language|tone|verbosity)\b/g, "").trim();
            }
          }

          if (!assistantCleanText.trim()) {
            assistantCleanText = "Hello! How can I assist you?";
          }

          // Save the full JSON response in history so the frontend can parse UI elements (carousel, deep_dive, etc.)
          backendContext.history.push({ role: "assistant", content: fullJsonResponse });
          backendContext.messageCount = backendContext.history.length;
          backendContext.lastActive = new Date().toISOString();

          // Execute preference extraction pipeline
          const preferenceRegex = /\[TRACK_PREFERENCE:\s*(.*?)\]/;
          const match = fullResponse.match(preferenceRegex);
          if (match && match[1]) {
            if (!backendContext.memory.preferredCategories) backendContext.memory.preferredCategories = [];
            if (!backendContext.memory.preferredCategories.includes(match[1].trim())) {
              backendContext.memory.preferredCategories.push(match[1].trim());
            }
          }

          // Parse out the filled jsonBuffer to attach to metadata tracking database hooks
          if (jsonBuffer.trim()) {
            try {
              const parsedProducts = JSON.parse(jsonBuffer.replace(/^```(?:json)?\s*/i, "").replace(/```$/, "").trim());
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'metadata', products: parsedProducts })}\n\n`));
            } catch (err) {
              console.warn("Failed to parse split-boundary JSON buffer:", err);
            }
          }

          try {
            fs.writeFileSync(contextFilePath, JSON.stringify(backendContext, null, 2));
          } catch (e) {
            console.warn("Could not save context file after LLM:", e);
          }

          if (confirmedCategory) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'metadata', confirmed_category: confirmedCategory })}\n\n`));
          }

          // Also update Redis Semantic Cache asynchronously
          storeInSemanticCache(userMessage, {
            responseTexts: [fullResponse],
            products: searchResults.length > 0 ? searchResults : null,
          }, chatId);

        } catch (streamErr: any) {
          console.warn("Switching to Groq streaming fallback orchestration channel...", streamErr.message || streamErr);
          try {
            const activeGroqKey = getNextGroqKey();
            if (!activeGroqKey) throw new Error("Groq credentials pool unavailable.");

            const { EXPLORE_SYSTEM_PROMPT, DEEP_RESEARCH_SYSTEM_PROMPT } = await import('@/lib/agents/writer');
            const systemInstruction = mode === "deep_research" ? DEEP_RESEARCH_SYSTEM_PROMPT : EXPLORE_SYSTEM_PROMPT;

            const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
              method: "POST",
              headers: { "Authorization": `Bearer ${activeGroqKey}`, "Content-Type": "application/json" },
              body: JSON.stringify({
                model: "llama-3.3-70b-versatile",
                stream: true,
                messages: [
                  { role: "system", content: systemInstruction },
                  ...userHistory.map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content || "" })),
                  { role: "user", content: userMessage }
                ]
              })
            });

            if (!groqResponse.ok) throw new Error(`Groq stream connection error: ${groqResponse.status}`);

            const reader = groqResponse.body?.getReader();
            const decoder = new TextDecoder();
            if (!reader) throw new Error("Failed to extract active reader stream descriptor.");

            let fallbackDataTagFound = false;
            let fallbackJsonBuffer = "";

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunkStr = decoder.decode(value, { stream: true });
              const lines = chunkStr.split("\n").filter(line => line.trim() !== "");

              for (const line of lines) {
                if (line.includes("data: [DONE]")) break;
                if (line.startsWith("data: ")) {
                  try {
                    const parsed = JSON.parse(line.slice(6));
                    const textDelta = parsed.choices[0]?.delta?.content || "";
                    fullResponse += textDelta;
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'chunk', text: textDelta })}\n\n`));
                  } catch (e) { }
                }
              }
            }

            if (fallbackJsonBuffer.trim()) {
              try {
                const parsedProducts = JSON.parse(fallbackJsonBuffer.replace(/^```(?:json)?\s*/i, "").replace(/```$/, "").trim());
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'metadata', products: parsedProducts })}\n\n`));
              } catch (err) { }
            }
          } catch (fallbackError: any) {
            console.error("Critical Execution Fault across both pipelines:", fallbackError);
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'chunk', text: `\\n[Stream Interrupted: ${fallbackError.message}]` })}\n\n`));
          }
        } finally {
          controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
      }
    });

  } catch (error: unknown) {
    logGeminiFailure(error);
    const errMessage = error instanceof Error ? error.stack : String(error);

    return NextResponse.json({
      text: [JSON.stringify({ ui_type: "text_response", text: "SYSTEM DIAGNOSTIC: " + errMessage })],
      fallback: true,
      products: null
    });
  }
}

interface ChatHistoryMessage {
  role?: string;
  content?: string;
}

interface ChatRequestBody {
  message: string;
  chatId?: string;
  history?: ChatHistoryMessage[];
  mode?: "deep_research" | "deep-research" | "explore" | "buy_explanation";
  requirements?: Record<string, unknown>;
  isRegenerate?: boolean;
}

function isChatRequestBody(body: unknown): body is ChatRequestBody {
  if (!body || typeof body !== "object") {
    return false;
  }

  return "message" in body && typeof body.message === "string";
}

function sanitizeError(error: unknown): unknown {
  if (!error) return error;

  try {
    // If it's a standard Error object, we might want to sanitize its message or stack
    let errorStr = "";
    if (error instanceof Error) {
      errorStr = error.stack || error.message;
    } else if (typeof error === "object") {
      errorStr = JSON.stringify(error);
    } else {
      return String(error);
    }
    return JSON.stringify(error, Object.getOwnPropertyNames(error));
  } catch (e) {
    return "[Un-stringifiable Error Object]";
  }
}

function logGeminiFailure(error: unknown) {
  const sanitized = sanitizeError(error);

  if (isRateLimitError(error)) {
    if (process.env.NODE_ENV === "development") {
      console.error("Gemini API rate limit hit. Serving fallback response.", sanitized);
    } else {
      console.error("Gemini API rate limit hit. Serving fallback response.");
    }
    return;
  }

  if (process.env.NODE_ENV === "development") {
    console.error("Gemini API failed. Serving fallback response.", sanitized);
  } else {
    console.error("Gemini API failed. Serving fallback response.");
  }
}

function isRateLimitError(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }

  const maybeError = error as { status?: unknown; message?: unknown };
  const message = typeof maybeError.message === "string" ? maybeError.message.toLowerCase() : "";

  return (
    maybeError.status === 429 ||
    message.includes("429") ||
    message.includes("too many requests") ||
    message.includes("quota exceeded") ||
    message.includes("rate limit")
  );
}
