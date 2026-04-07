"use client";

import * as React from "react";
import { Search, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  filterOpen: boolean;
  onFilterOpenChange: (open: boolean) => void;
  filterPanel: React.ReactNode;
  loading: boolean;
}

export function SearchBar({
  value,
  onChange,
  onSearch,
  filterOpen,
  onFilterOpenChange,
  filterPanel,
  loading,
}: SearchBarProps) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
      <div className="relative flex flex-1 items-center">
        <Search className="absolute left-3 h-4 w-4 shrink-0 text-muted-foreground pointer-events-none" />
        {loading ? (
          <Skeleton className="h-9 w-full rounded-md" />
        ) : (
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSearch()}
            placeholder="Search for an object in videos (e.g. person, dog, car)"
            className="h-9 w-full rounded-md border border-input bg-background py-1 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        )}
      </div>
      <div className="flex gap-2">
        <Popover open={filterOpen} onOpenChange={onFilterOpenChange}>
          <Tooltip>
            <TooltipTrigger asChild>
              <PopoverTrigger asChild>
                <Button type="button" variant="outline" size="sm">
                  Filter
                </Button>
              </PopoverTrigger>
            </TooltipTrigger>
            <TooltipContent>Open filters</TooltipContent>
          </Tooltip>
          <PopoverContent className="glass-surface w-80" align="end" sideOffset={8}>
            {filterPanel}
          </PopoverContent>
        </Popover>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button onClick={onSearch} disabled={loading} size="sm">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              <span className="ml-2">Fetch</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>{loading ? "Fetching…" : "Fetch videos"}</TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
