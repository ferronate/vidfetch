"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Info } from "lucide-react";

interface ResultsInfoButtonProps {
  queryObject: string | null;
}

export function ResultsInfoButton({ queryObject }: ResultsInfoButtonProps) {
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button type="button" variant="ghost" size="icon" className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground" aria-label="What do these results mean?">
          <Info className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-72 text-sm" align="start">
        <p className="font-medium text-foreground mb-1">About these results</p>
        <p className="text-muted-foreground">
          Videos are ordered by how similar they are to your query (color and content). &quot;Best match&quot; is the closest; &quot;Similar&quot; and &quot;Related&quot; are further matches.
        </p>
        {queryObject && (
          <p className="mt-2 text-muted-foreground">
            The <span className="text-emerald-600 dark:text-emerald-400 font-medium">green bar</span> on a video shows when &quot;{queryObject}&quot; appears in that clip — click a segment to jump to that time.
          </p>
        )}
      </PopoverContent>
    </Popover>
  );
}
