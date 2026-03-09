"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, Loader2, Film, SlidersHorizontal, LayoutGrid, Info, Play, Pause } from "lucide-react";

const API_BASE = "http://localhost:8000";

const COLOR_OPTIONS = [
  { value: "any", label: "Any" },
  { value: "warm", label: "Warm" },
  { value: "cool", label: "Cool" },
  { value: "bright", label: "Bright" },
  { value: "dark", label: "Dark" },
] as const;

type VideoEntry = { id: string; name: string };
type QueryResult = {
  id: string;
  name: string;
  distance: number;
  object_segments?: { start: number; end: number }[];
};

function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function CustomProgressBar({
  videoRef,
  duration,
  currentTime,
  isPlaying,
  segments,
  queryObject,
  onSeek,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  segments: { start: number; end: number }[];
  queryObject: string | null;
  onSeek: (t: number) => void;
}) {
  const barRef = useRef<HTMLDivElement>(null);
  const dur = duration > 0 ? duration : 1;
  const playedPercent = (currentTime / dur) * 100;
  const hasSegments = queryObject && segments.length > 0;

  const handleBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const bar = barRef.current;
    const v = videoRef.current;
    if (!bar || !v || duration <= 0) return;
    const rect = bar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = (x / rect.width) * duration;
    v.currentTime = Math.max(0, Math.min(t, duration));
    onSeek(t);
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0 text-white hover:bg-white/20"
          onClick={() => {
            const v = videoRef.current;
            if (!v) return;
            if (v.paused) v.play().catch(() => {});
            else v.pause();
          }}
          aria-label="Play / Pause"
        >
          {!isPlaying ? (
            <Play className="h-4 w-4 fill-current" />
          ) : (
            <Pause className="h-4 w-4 fill-current" />
          )}
        </Button>
        <span className="min-w-[4rem] text-xs text-white/90 tabular-nums">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <div
          ref={barRef}
          className="relative h-2 flex-1 min-w-0 cursor-pointer overflow-hidden rounded-full bg-neutral-600 shadow-inner"
          onClick={handleBarClick}
          role="slider"
          aria-label="Seek"
          aria-valuenow={currentTime}
          aria-valuemin={0}
          aria-valuemax={duration}
        >
          {/* Played portion */}
          <div
            className="absolute inset-y-0 left-0 rounded-l-full bg-neutral-500 transition-[width] duration-75"
            style={{ width: `${playedPercent}%` }}
          />
          {/* Green segments (object in video) */}
          {hasSegments &&
            segments.map((seg, i) => {
              const left = (seg.start / dur) * 100;
              const width = Math.max(2, ((seg.end - seg.start) / dur) * 100);
              return (
                <button
                  key={i}
                  type="button"
                  className="absolute inset-y-0 rounded-full bg-emerald-500 transition hover:bg-emerald-400 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white/50"
                  style={{ left: `${left}%`, width: `${width}%` }}
                  title={`Jump to ${seg.start.toFixed(1)}s`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSeek(seg.start);
                    videoRef.current && (videoRef.current.currentTime = seg.start);
                    videoRef.current?.play().catch(() => {});
                  }}
                  aria-label={`Object at ${seg.start.toFixed(1)}s`}
                />
              );
            })}
          {/* Playhead */}
          <div
            className="absolute top-1/2 z-10 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-md ring-2 ring-black/30 pointer-events-none"
            style={{ left: `${Math.min(playedPercent, 99)}%` }}
          />
        </div>
      </div>
      {hasSegments && (
        <span className="rounded bg-white/20 px-1.5 py-0.5 text-xs font-medium text-white/95 w-fit">
          {queryObject}
        </span>
      )}
    </div>
  );
}

function ResultCard({
  result,
  index,
  queryObject,
}: {
  result: QueryResult;
  index: number;
  queryObject: string | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const segments = result.object_segments ?? [];
  const hasObjectBar = queryObject && segments.length > 0;
  const barDuration = duration > 0 ? duration : Math.max(0, ...segments.flatMap((s) => [s.start, s.end]), 1);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTimeUpdate = () => setCurrentTime(v.currentTime);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onLoadedMetadata = () => {
      setDuration(v.duration);
      setCurrentTime(v.currentTime);
    };
    v.addEventListener("timeupdate", onTimeUpdate);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("loadedmetadata", onLoadedMetadata);
    return () => {
      v.removeEventListener("timeupdate", onTimeUpdate);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("loadedmetadata", onLoadedMetadata);
    };
  }, []);

  return (
    <Card className="overflow-hidden border-border bg-card transition-shadow hover:shadow-md">
      <div className="relative aspect-video w-full bg-muted group">
        <video
          ref={videoRef}
          src={`${API_BASE}/api/video/${encodeURIComponent(result.id)}`}
          className="h-full w-full object-cover"
          preload="metadata"
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
          onClick={() => videoRef.current?.paused && videoRef.current?.play().catch(() => {})}
        />
        <div className="absolute bottom-0 left-0 right-0 px-2.5 pb-1.5 pt-4 bg-gradient-to-t from-black/85 to-transparent opacity-100 transition-opacity group-hover:opacity-100">
          <CustomProgressBar
            videoRef={videoRef}
            duration={barDuration}
            currentTime={currentTime}
            isPlaying={isPlaying}
            segments={segments}
            queryObject={queryObject}
            onSeek={(t) => {
              const v = videoRef.current;
              if (v) {
                v.currentTime = t;
                v.play().catch(() => {});
              }
            }}
          />
        </div>
      </div>
      <CardHeader className="p-3 pb-2 pt-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Film className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate font-medium">{result.name}</span>
        </CardTitle>
        <CardDescription className="text-xs">
          {index === 0 ? "Best match" : `Result ${index + 1}`}
          {" · "}
          {result.distance < 0.2 ? "Very similar" : result.distance < 0.4 ? "Similar" : "Related"}
        </CardDescription>
      </CardHeader>
    </Card>
  );
}

function ResultsInfoButton({ queryObject }: { queryObject: string | null }) {
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

const DEFAULT_COLOR = "any";
const DEFAULT_TYPES: string[] = [];

export function VideoSearch() {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [objectTypes, setObjectTypes] = useState<string[]>([]);
  const [filterColor, setFilterColor] = useState(DEFAULT_COLOR);
  const [filterTypes, setFilterTypes] = useState<string[]>(DEFAULT_TYPES);
  const [colorRefVideoId, setColorRefVideoId] = useState<string>("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [results, setResults] = useState<QueryResult[] | null>(null);
  const [queryObject, setQueryObject] = useState<string | null>(null);
  const [timeMs, setTimeMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/videos`)
      .then((res) => (res.ok ? res.json() : { videos: [] }))
      .then((data) => {
        setVideos(data.videos || []);
        if (data.videos?.length && !colorRefVideoId) setColorRefVideoId(data.videos[0].id);
      })
      .catch(() => setVideos([]));
  }, [colorRefVideoId]);

  useEffect(() => {
    fetch(`${API_BASE}/api/objects`)
      .then((res) => (res.ok ? res.json() : { objects: [] }))
      .then((data) => setObjectTypes(data.objects || []))
      .catch(() => setObjectTypes([]));
  }, []);

  const runQuery = () => {
    setError(null);
    setResults(null);
    setTimeMs(null);
    setLoading(true);
    const form = new FormData();
    form.set("color_filter", filterColor === "same" && colorRefVideoId ? colorRefVideoId : filterColor);
    form.set("k", "5");
    const typesToSend = searchInput.trim()
      ? searchInput.split(",").map((s) => s.trim()).filter(Boolean)
      : filterTypes;
    typesToSend.forEach((t) => form.append("object_types", t));

    fetch(`${API_BASE}/api/query`, { method: "POST", body: form })
      .then((res) => {
        if (!res.ok) return res.json().then((err) => { throw new Error(err.detail || "Query failed"); });
        return res.json();
      })
      .then((data) => {
        setResults(data.results || []);
        setQueryObject(data.query_object ?? null);
        setTimeMs(data.time_ms ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Something went wrong"))
      .finally(() => setLoading(false));
  };

  const applyFilters = () => {
    setFilterOpen(false);
    runQuery();
  };

  const resetFilters = () => {
    setFilterColor(DEFAULT_COLOR);
    setFilterTypes(DEFAULT_TYPES);
  };

  const toggleType = (type: string) => {
    setFilterTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  return (
    <div className="space-y-6">
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Fetch videos
          </CardTitle>
          <CardDescription>
            Object search is the default. Open the filter to choose object type(s) and color.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="relative flex flex-1 items-center">
              <Search className="absolute left-3 h-4 w-4 shrink-0 text-muted-foreground pointer-events-none" />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runQuery()}
                placeholder="Search for an object in videos (e.g. person, dog, car)"
                className="h-10 w-full rounded-md border border-input bg-background py-2 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div className="flex gap-2">
              <Popover open={filterOpen} onOpenChange={setFilterOpen}>
                <PopoverTrigger asChild>
                  <Button type="button" variant="outline" className="gap-2">
                    <SlidersHorizontal className="h-4 w-4" />
                    Filter
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-80" align="end">
                  <div className="space-y-4">
                    <div>
                      <p className="mb-2 text-sm font-medium text-foreground">Filter by color</p>
                      <Select
                        value={filterColor}
                        onValueChange={setFilterColor}
                      >
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
                      {filterColor === "same" && (
                        <Select
                          value={colorRefVideoId}
                          onValueChange={setColorRefVideoId}
                          key="colorRef"
                        >
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
                      <p className="mb-2 text-sm font-medium text-foreground">Filter by type</p>
                      <p className="mb-2 text-xs text-muted-foreground">
                        Show videos containing:
                      </p>
                      <div className="max-h-40 space-y-2 overflow-y-auto rounded border border-border p-2">
                        {objectTypes.length === 0 ? (
                          <p className="text-xs text-muted-foreground">Run build_object_index to enable.</p>
                        ) : (
                          objectTypes.map((obj) => (
                            <label
                              key={obj}
                              className="flex cursor-pointer items-center gap-2 text-sm"
                            >
                              <input
                                type="checkbox"
                                checked={filterTypes.includes(obj)}
                                onChange={() => toggleType(obj)}
                                className="h-4 w-4 rounded border-input"
                              />
                              {obj}
                            </label>
                          ))
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
                      <button
                        type="button"
                        onClick={resetFilters}
                        className="text-sm text-muted-foreground underline-offset-4 hover:underline"
                      >
                        Reset all filters
                      </button>
                      <div className="flex gap-2">
                        <Button type="button" variant="outline" size="sm" onClick={() => setFilterOpen(false)}>
                          Cancel
                        </Button>
                        <Button type="button" size="sm" onClick={applyFilters}>
                          Apply Filters
                        </Button>
                      </div>
                    </div>
                  </div>
                </PopoverContent>
              </Popover>
              <Button onClick={runQuery} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                <span className="ml-2">Fetch videos</span>
              </Button>
            </div>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {results && results.length === 0 && timeMs != null && (
        <div className="rounded-lg border border-border bg-muted/50 p-6 text-center">
          <p className="font-medium text-foreground">No videos found</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {queryObject || filterTypes.length
              ? `No videos match the selected type(s) or color. Try changing the filter.`
              : "Select at least one object type in the filter, then click Fetch videos."}
          </p>
        </div>
      )}

      {results && results.length > 0 && (
        <div>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <h3 className="text-lg font-semibold text-foreground">Results</h3>
            <ResultsInfoButton queryObject={queryObject} />
            {timeMs != null && (
              <span className="text-sm text-muted-foreground">
                Found in {(timeMs / 1000).toFixed(1)}s
              </span>
            )}
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {results.map((r, i) => (
              <ResultCard key={r.id} result={r} index={i} queryObject={queryObject} />
            ))}
          </div>
        </div>
      )}

      {videos.length > 0 && (
        <div>
          <div className="mb-3 flex items-center gap-2">
            <h3 className="text-lg font-semibold text-foreground">
              <LayoutGrid className="mr-1.5 inline-block h-4 w-4 align-middle text-muted-foreground" />
              Gallery
            </h3>
            <span className="hidden text-sm text-muted-foreground sm:inline">Indexed videos · search above to find by object or color</span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {videos.map((v) => (
              <div
                key={v.id}
                className="overflow-hidden rounded-lg border border-border bg-muted shadow-sm transition-shadow hover:shadow"
              >
                <div className="aspect-video w-full">
                  <video
                    src={`${API_BASE}/api/video/${encodeURIComponent(v.id)}`}
                    className="h-full w-full object-cover"
                    controls
                    preload="metadata"
                    muted
                    playsInline
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
