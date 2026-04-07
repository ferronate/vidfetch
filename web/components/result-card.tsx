"use client";

import { useRef, useState, useEffect } from "react";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Film, Play, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { API_BASE } from "@/lib/api";
import type { QueryResult, TimeSegment } from "@/lib/types";
import { formatSeconds, getDistanceLabel } from "@/lib/utils";

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
  segments: TimeSegment[];
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
          {formatSeconds(currentTime)} / {formatSeconds(duration)}
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
          <div
            className="absolute inset-y-0 left-0 rounded-l-full bg-neutral-500 transition-[width] duration-75"
            style={{ width: `${playedPercent}%` }}
          />
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
                  onClick={(e) => {
                    e.stopPropagation();
                    onSeek(seg.start);
                    if (videoRef.current) {
                      videoRef.current.currentTime = seg.start;
                      videoRef.current.play().catch(() => {});
                    }
                  }}
                  aria-label={`Object at ${seg.start.toFixed(1)}s`}
                />
              );
            })}
          <div
            className="absolute top-1/2 z-10 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-md ring-2 ring-black/30 pointer-events-none"
            style={{ left: `${Math.min(playedPercent, 99)}%` }}
          />
        </div>
      </div>
      {hasSegments && (
        <Badge variant="secondary" className="px-1.5 py-0.5 text-xs">
          {queryObject}
        </Badge>
      )}
    </div>
  );
}

interface ResultCardProps {
  result: QueryResult;
  index: number;
  queryObject: string | null;
}

export function ResultCard({ result, index, queryObject }: ResultCardProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const segments = result.object_segments ?? [];
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
    <Card className="glass-surface overflow-hidden rounded-xl transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg">
      <div className="relative aspect-video w-full bg-muted group">
        <video
          ref={videoRef}
          src={`${API_BASE}/api/video/${encodeURIComponent(result.id)}`}
          className="h-full w-full object-cover"
          preload="metadata"
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
          onClick={() => videoRef.current?.paused && videoRef.current?.play().catch(() => {})}
        />
        {duration <= 0 && (
          <div className="absolute inset-0">
            <Skeleton className="h-full w-full rounded-t-md" />
          </div>
        )}
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
        <CardTitle className="flex items-center gap-2 text-sm leading-tight">
          <Film className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate font-medium">{result.name}</span>
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground">
          {index === 0 ? "Best match" : `Result ${index + 1}`}
          {" · "}
          {getDistanceLabel(result.distance)}
        </CardDescription>
      </CardHeader>
    </Card>
  );
}