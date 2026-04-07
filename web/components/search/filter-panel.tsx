"use client";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const COLOR_OPTIONS = [
  { value: "any", label: "Any" },
  { value: "warm", label: "Warm" },
  { value: "cool", label: "Cool" },
  { value: "bright", label: "Bright" },
  { value: "dark", label: "Dark" },
] as const;

interface FilterPanelProps {
  colorFilter: string;
  onColorChange: (value: string) => void;
  objectTypes: string[];
  selectedTypes: string[];
  onToggleType: (type: string) => void;
  onReset: () => void;
  onApply: () => void;
  onCancel: () => void;
  videos: { id: string; name: string }[];
  colorRefVideoId: string;
  onColorRefChange: (id: string) => void;
}

export function FilterPanel({
  colorFilter,
  onColorChange,
  objectTypes,
  selectedTypes,
  onToggleType,
  onReset,
  onApply,
  onCancel,
  videos,
  colorRefVideoId,
  onColorRefChange,
}: FilterPanelProps) {
  return (
    <div className="space-y-4">
      <div>
        <p className="mb-2 text-sm font-medium text-foreground">Filter by color</p>
        <Select value={colorFilter} onValueChange={onColorChange}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {COLOR_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
            <SelectItem value="same">Same as video</SelectItem>
          </SelectContent>
        </Select>
        {colorFilter === "same" && (
          <Select value={colorRefVideoId} onValueChange={onColorRefChange}>
            <SelectTrigger className="mt-2 w-full">
              <SelectValue placeholder="Select video" />
            </SelectTrigger>
            <SelectContent>
              {videos.map((v) => (
                <SelectItem key={v.id} value={v.id}>
                  {v.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-foreground">
          Filter by type
          <Badge variant="secondary" className="ml-2">{selectedTypes.length}</Badge>
        </p>
        <p className="mb-2 text-xs text-muted-foreground">Show videos containing:</p>
        <div className="max-h-44 space-y-1.5 overflow-y-auto rounded-lg border border-border bg-muted/35 p-2.5">
          {objectTypes.length === 0 ? (
            <p className="text-xs text-muted-foreground">Run build_object_index to enable.</p>
          ) : (
            objectTypes.map((obj) => (
              <label
                key={obj}
                className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-sm hover:bg-background/70"
              >
                <input
                  type="checkbox"
                  checked={selectedTypes.includes(obj)}
                  onChange={() => onToggleType(obj)}
                  className="h-4 w-4 rounded border-input accent-blue-600"
                />
                {obj}
              </label>
            ))
          )}
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={onReset}
              className="text-sm text-muted-foreground underline-offset-4 hover:underline"
            >
              Reset all filters
            </button>
          </TooltipTrigger>
          <TooltipContent>Reset filters to defaults</TooltipContent>
        </Tooltip>
        <div className="flex gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button type="button" variant="outline" size="sm" onClick={onCancel}>
                Cancel
              </Button>
            </TooltipTrigger>
            <TooltipContent>Close without applying</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button type="button" size="sm" onClick={onApply}>
                Apply Filters
              </Button>
            </TooltipTrigger>
            <TooltipContent>Apply filters and search</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}
