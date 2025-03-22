"use client"

import { useState, useEffect } from "react"
import { Check, ChevronsUpDown, Globe } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"

type Language = {
  code: string
  name: string
}

interface LanguageSelectorProps {
  onLanguageChange: (language: string) => void
}

export function LanguageSelector({ onLanguageChange }: LanguageSelectorProps) {
  const [open, setOpen] = useState(false)
  const [languages, setLanguages] = useState<Language[]>([])
  const [selectedLanguage, setSelectedLanguage] = useState<Language>({ code: "en", name: "English" })

  useEffect(() => {
    // In a real app, fetch from your Flask backend
    // For demo purposes, we'll use hardcoded values
    setLanguages([
      { code: "en", name: "English" },
      { code: "es", name: "Spanish" },
      { code: "fr", name: "French" },
      { code: "de", name: "German" },
      { code: "it", name: "Italian" },
      { code: "pt", name: "Portuguese" },
      { code: "ru", name: "Russian" },
      { code: "zh", name: "Chinese" },
      { code: "ja", name: "Japanese" },
      { code: "ko", name: "Korean" },
      { code: "ar", name: "Arabic" },
      { code: "hi", name: "Hindi" },
    ])
  }, [])

  const handleSelect = (language: Language) => {
    setSelectedLanguage(language)
    setOpen(false)
    onLanguageChange(language.code)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} className="w-full justify-between">
          <div className="flex items-center">
            <Globe className="mr-2 h-4 w-4" />
            {selectedLanguage.name}
          </div>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search language..." />
          <CommandList>
            <CommandEmpty>No language found.</CommandEmpty>
            <CommandGroup>
              {languages.map((language) => (
                <CommandItem key={language.code} value={language.code} onSelect={() => handleSelect(language)}>
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selectedLanguage.code === language.code ? "opacity-100" : "opacity-0",
                    )}
                  />
                  {language.name}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

