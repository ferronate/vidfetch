"use client";

import { useState } from "react";
import { LayoutGrid, Wand2 } from "lucide-react";
import { CorrectionReview } from "@/components/correction-review";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { DetectionJob, VideoEntry } from "@/lib/types";

interface VideoGalleryProps {
  videos: VideoEntry[];
  apiBase: string;
  selectedVideoId: string;
  onSelectVideo: (videoId: string) => void;
  detectionJobs: Record<string, DetectionJob>;
}

export function VideoGallery({
  videos,
  apiBase,
  selectedVideoId,
  onSelectVideo,
  detectionJobs,
}: VideoGalleryProps) {
  const [reviewVideo, setReviewVideo] = useState<{ id: string; name: string } | null>(null);

  if (videos.length === 0) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
          <LayoutGrid className="inline-block h-4 w-4 align-middle text-muted-foreground" />
          Gallery
          <Badge variant="secondary">{videos.length}</Badge>
        </h3>
        <span className="hidden text-sm text-muted-foreground sm:inline">
          Check one video to detect that clip, or use the all-videos action.
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {videos.map((v) => {
          const job = detectionJobs[v.id];
          const isSelected = selectedVideoId === v.id;
          const isActive = Boolean(job && job.status !== "done" && job.status !== "failed");
          const isFinished = job?.status === "done";
          const progressValue = job?.status === "failed" ? 100 : isFinished ? 100 : isActive ? 65 : 0;
          const statusLabel = job
            ? job.status === "queued"
              ? "Queued"
              : job.status === "running"
                ? "Running"
                : job.status === "done"
                  ? "Complete"
                  : "Failed"
            : "";

          return (
            <div
              key={v.id}
              className={cn(
                "glass-surface group relative overflow-hidden rounded-xl transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg",
                isSelected && "ring-2 ring-primary ring-offset-2 ring-offset-background"
              )}
            >
              <div className="relative aspect-video w-full bg-muted">
                <video
                  src={`${apiBase}/api/video/${encodeURIComponent(v.id)}`}
                  className="h-full w-full object-cover"
                  controls
                  preload="metadata"
                  muted
                  playsInline
                />

                {isActive && (
                  <div className="absolute inset-0 flex flex-col justify-end bg-black/55 p-3 text-white">
                    <div className="flex items-center justify-between gap-2 text-[11px] uppercase tracking-wide text-white/85">
                      <span>{statusLabel}</span>
                      <span>{isFinished ? "Done" : "Working"}</span>
                    </div>
                    <Progress value={progressValue} className="mt-2 h-2 bg-white/20" />
                  </div>
                )}

                <label className="absolute left-2 top-2 inline-flex items-center gap-2 rounded-full bg-black/55 px-2 py-1 text-xs text-white backdrop-blur">
                  <Checkbox
                    checked={isSelected}
                    onCheckedChange={(checked) => onSelectVideo(checked ? v.id : "")}
                    aria-label={`Select ${v.name}`}
                  />
                  <span className="max-w-[8rem] truncate">Select</span>
                </label>

                <div className="absolute right-2 top-2 flex gap-2 opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100">
                  {isFinished && (
                    <Badge variant="secondary" className="bg-black/55 text-white backdrop-blur">
                      Done
                    </Badge>
                  )}
                  {isSelected && (
                    <Badge variant="secondary" className="bg-primary/90 text-primary-foreground backdrop-blur">
                      Selected
                    </Badge>
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between gap-2 bg-gradient-to-t from-black/70 via-black/35 to-transparent p-2">
                <span className="truncate text-xs text-white">{v.name}</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      className="h-7 gap-1 bg-white/20 text-white hover:bg-white/30"
                      onClick={() => setReviewVideo(v)}
                    >
                      <Wand2 data-icon="inline-start" />
                      Review
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Review & correct detections</TooltipContent>
                </Tooltip>
              </div>
            </div>
          );
        })}
      </div>

      {/* Correction Review Dialog */}
      <Dialog open={!!reviewVideo} onOpenChange={(open) => { if (!open) setReviewVideo(null); }}>
        {reviewVideo && (
          <DialogContent className="mx-4 max-h-[90vh] w-[min(96vw,1200px)] sm:max-w-[1200px] overflow-y-auto rounded-xl border border-border bg-background shadow-xl">
            <div className="flex items-center justify-between border-b p-4">
              <div>
                <DialogTitle className="text-lg font-semibold">Review Detections</DialogTitle>
                <DialogDescription className="text-sm text-muted-foreground">
                  {reviewVideo.name}
                </DialogDescription>
              </div>
            </div>
            <div className="grid gap-4 p-4 lg:grid-cols-[minmax(320px,0.95fr)_minmax(0,1.05fr)]">
              <div className="flex flex-col gap-3">
                <div className="overflow-hidden rounded-xl border border-border bg-black">
                  <video
                    src={`${apiBase}/api/video/${encodeURIComponent(reviewVideo.id)}`}
                    className="aspect-video w-full object-cover"
                    controls
                    preload="metadata"
                    playsInline
                  />
                </div>
                <div className="rounded-xl border border-border bg-muted p-3 text-sm text-muted-foreground">
                  Use the timeline and labels on the right to inspect what the model saw, then correct
                  anything that is mislabeled or missing.
                </div>
              </div>
              <div className="min-w-0 rounded-xl border border-border bg-card p-3">
                <CorrectionReview videoId={reviewVideo.id} />
              </div>
            </div>
          </DialogContent>
        )}
      </Dialog>
    </div>
  );
}