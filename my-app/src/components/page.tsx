"use client"

import type React from "react"

import { useState, useRef } from "react"
import { useChat } from "ai/react"
import { Mic, MicOff, Send, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"

export default function VoiceAssistant() {
  const [isListening, setIsListening] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: "/api/chat",
    onFinish: (message) => {
      // Speak the response when chat completes
      if (message.role === "assistant") {
        speakText(message.content)
      }
    },
  })

  // Start voice recognition
  const startListening = () => {
    if (!("webkitSpeechRecognition" in window)) {
      alert("Speech recognition is not supported in your browser")
      return
    }

    setIsListening(true)
    setIsProcessing(true)

    const recognition = new (window as any).webkitSpeechRecognition()
    recognition.lang = "en-US"
    recognition.continuous = false
    recognition.interimResults = false

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript
      handleInputChange({ target: { value: transcript } } as any)
      setIsProcessing(false)
    }

    recognition.onerror = () => {
      setIsListening(false)
      setIsProcessing(false)
    }

    recognition.onend = () => {
      setIsListening(false)
      // Auto-submit after recognition completes
      if (input) {
        const fakeEvent = { preventDefault: () => {} } as React.FormEvent<HTMLFormElement>
        handleSubmit(fakeEvent)
      }
    }

    recognition.start()
  }

  // Speak text using the backend TTS endpoint
  const speakText = async (text: string) => {
    try {
      setIsSpeaking(true)
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) throw new Error("TTS request failed")

      const audioBlob = await response.blob()
      const audioUrl = URL.createObjectURL(audioBlob)

      if (audioRef.current) {
        audioRef.current.src = audioUrl
        audioRef.current.play()
      }
    } catch (error) {
      console.error("Error with text-to-speech:", error)
    } finally {
      setIsSpeaking(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gradient-to-b from-purple-950/40 via-purple-800/20 to-purple-950/40">
      <Card className="w-full max-w-2xl p-4 shadow-lg border-purple-500/20 bg-background/80 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-foreground">Multilingual Voice Assistant</h1>
          <div className="flex space-x-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className={`h-2 w-2 rounded-full ${
                  isListening || isLoading || isSpeaking
                    ? i % 2 === 0
                      ? "bg-purple-500 animate-pulse"
                      : "bg-blue-500 animate-pulse delay-100"
                    : "bg-gray-300"
                }`}
              />
            ))}
          </div>
        </div>

        <div className="h-80 overflow-y-auto mb-4 p-4 rounded-md bg-background/50 border border-border">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <p>Start a conversation with your voice assistant</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`mb-3 p-3 rounded-lg ${
                  message.role === "user"
                    ? "bg-purple-500/10 border border-purple-500/20 ml-8"
                    : "bg-blue-500/10 border border-blue-500/20 mr-8"
                }`}
              >
                <p className="text-sm font-semibold mb-1">{message.role === "user" ? "You" : "Assistant"}</p>
                <p className="text-foreground">{message.content}</p>
              </div>
            ))
          )}
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col space-y-2">
          <Textarea
            value={input}
            onChange={handleInputChange}
            placeholder="Type your message or click the mic to speak..."
            className="resize-none"
            rows={3}
          />

          <div className="flex justify-between">
            <Button
              type="button"
              variant={isListening ? "destructive" : "secondary"}
              size="icon"
              onClick={startListening}
              disabled={isListening || isLoading || isSpeaking}
            >
              {isProcessing ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : isListening ? (
                <MicOff className="h-5 w-5" />
              ) : (
                <Mic className="h-5 w-5" />
              )}
            </Button>

            <Button
              type="submit"
              disabled={!input || isLoading || isListening || isSpeaking}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <>
                  <Send className="h-5 w-5 mr-2" />
                  Send
                </>
              )}
            </Button>
          </div>
        </form>
      </Card>

      <audio ref={audioRef} className="hidden" />
    </div>
  )
}