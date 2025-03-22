import { streamText } from "ai"
import { openai } from "@ai-sdk/openai"
import { NextResponse } from "next/server"

export async function POST(req: Request) {
  try {
    const { messages } = await req.json()

    // Create a prompt from the messages
    const lastMessage = messages[messages.length - 1].content

    // Stream the response from OpenAI
    const { textStream } = await streamText({
      model: openai("gpt-4o"),
      prompt: lastMessage,
      system: "You are a helpful, friendly multilingual voice assistant. Provide concise and helpful responses.",
    })

    // Return the stream
    return new Response(textStream)
  } catch (error) {
    console.error("Error in chat API:", error)
    return NextResponse.json({ error: "Failed to process chat request" }, { status: 500 })
  }
}

