"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { SearchBar, FilterPanel, ResultsInfoButton } from "@/components/search";
import { VideoGallery } from "@/components/video-gallery";
import { ResultCard } from "@/components/result-card";
import { API_BASE } from "@/lib/api";
import { useVideoSearch } from "@/hooks/useVideoSearch";

export function VideoSearch() {
  const {
    videos,
    objectTypes,
    filterColor,
    filterTypes,
    colorRefVideoId,
    filterOpen,
    searchInput,
    results,
    queryObject,
    timeMs,
    loading,
    error,
    selectedVideoId,
    detectionJobs,
    detectionBusy,
    detectionSummary,
    setFilterColor,
    setColorRefVideoId,
    setFilterOpen,
    setSearchInput,
    setSelectedVideoId,
    runQuery,
    applyFilters,
    resetFilters,
    toggleType,
    detectSelectedVideo,
    detectAllVideos,
  } = useVideoSearch();

  const detectionProgress = detectionSummary && detectionSummary.total > 0
    ? Math.round((detectionSummary.done / detectionSummary.total) * 100)
    : 0;

  const selectedVideo = videos.find((video) => video.id === selectedVideoId);

  return (
    <div className="flex flex-col gap-7">
      <Card className="glass-surface rounded-2xl p-4 transition-shadow duration-200 hover:shadow-lg">
        <CardHeader className="p-0">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Search className="h-5 w-5" />
            Fetch videos
          </CardTitle>
          <CardDescription className="text-sm text-muted-foreground">
            Object search is the default. Open the filter to choose object type(s) and color.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 p-0 mt-3">
          <SearchBar
            value={searchInput}
            onChange={setSearchInput}
            onSearch={runQuery}
            filterOpen={filterOpen}
            onFilterOpenChange={setFilterOpen}
            filterPanel={
              <FilterPanel
                colorFilter={filterColor}
                onColorChange={setFilterColor}
                objectTypes={objectTypes}
                selectedTypes={filterTypes}
                onToggleType={toggleType}
                onReset={resetFilters}
                onApply={applyFilters}
                onCancel={() => setFilterOpen(false)}
                videos={videos}
                colorRefVideoId={colorRefVideoId}
                onColorRefChange={setColorRefVideoId}
              />
            }
            loading={loading}
          />
            {error && <p className="text-sm text-destructive">{error}</p>}

            {loading && !results && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="rounded overflow-hidden border border-border bg-card p-2">
                    <Skeleton className="h-40 w-full rounded-md" />
                    <Skeleton className="h-4 mt-2 w-1/2" />
                  </div>
                ))}
              </div>
            )}
        </CardContent>
      </Card>

      <Card className="glass-surface rounded-2xl p-4 transition-shadow duration-200 hover:shadow-lg">
        <CardHeader className="p-0">
          <CardTitle className="text-lg">Detect objects</CardTitle>
          <CardDescription className="text-xs text-muted-foreground">
            Detect one or all videos.
          </CardDescription>
        </CardHeader>
        <CardContent className="mt-2 flex flex-col gap-2 p-0">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              size="sm"
              onClick={() => { void detectSelectedVideo(); }}
              disabled={!selectedVideoId || detectionBusy}
            >
              Detect selected
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => { void detectAllVideos(); }}
              disabled={videos.length === 0 || detectionBusy}
            >
              Detect all
            </Button>
            {selectedVideo && (
              <Badge variant="secondary" className="max-w-full truncate">
                {selectedVideo.name}
              </Badge>
            )}
          </div>

          {detectionSummary && (
            <div className="flex flex-col gap-1 rounded-lg border border-border bg-muted/30 p-2">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="text-muted-foreground truncate">
                  {detectionSummary.mode === "all"
                    ? "Running all"
                    : "Running selected"}
                </span>
                <span className="tabular-nums text-muted-foreground">
                  {detectionSummary.done}/{detectionSummary.total}
                </span>
              </div>
              <Progress value={detectionProgress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {results && results.length === 0 && timeMs != null && (
        <div className="glass-surface rounded-xl p-6 text-center">
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
            <h3 className="text-lg font-semibold text-foreground">
              Results
              <Badge variant="secondary" className="ml-2">{results.length}</Badge>
            </h3>
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

      <VideoGallery
        videos={videos}
        apiBase={API_BASE}
        selectedVideoId={selectedVideoId}
        onSelectVideo={setSelectedVideoId}
        detectionJobs={detectionJobs}
      />
    </div>
  );
}