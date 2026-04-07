"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import type { DetectionJob, QueryResult, VideoEntry } from "@/lib/types";

type DetectionRunSummary = {
  mode: "selected" | "all";
  total: number;
  done: number;
  reused: number;
} | null;

export function useVideoSearch() {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [objectTypes, setObjectTypes] = useState<string[]>([]);
  const [filterColor, setFilterColor] = useState("any");
  const [filterTypes, setFilterTypes] = useState<string[]>([]);
  const [colorRefVideoId, setColorRefVideoId] = useState<string>("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [results, setResults] = useState<QueryResult[] | null>(null);
  const [queryObject, setQueryObject] = useState<string | null>(null);
  const [timeMs, setTimeMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [detectionJobs, setDetectionJobs] = useState<Record<string, DetectionJob>>({});
  const [detectionBusy, setDetectionBusy] = useState(false);
  const [detectionSummary, setDetectionSummary] = useState<DetectionRunSummary>(null);

  const activeDetectionJobIds = useRef<Set<string>>(new Set());
  const detectionRunTotal = useRef(0);
  const detectionRunReused = useRef(0);

  const loadVideos = useCallback(async () => {
    const fetchedVideos = await apiClient.getVideos();
    setVideos(fetchedVideos);
    return fetchedVideos;
  }, []);

  const loadObjects = useCallback(async () => {
    const objects = await apiClient.getObjects();
    setObjectTypes(objects);
    return objects;
  }, []);

  const refreshCatalog = useCallback(async () => {
    await Promise.all([loadVideos(), loadObjects()]);
  }, [loadObjects, loadVideos]);

  useEffect(() => {
    let cancelled = false;

    loadVideos()
      .then((fetchedVideos) => {
        if (cancelled) return;
        setColorRefVideoId((prev) => prev || fetchedVideos[0]?.id || "");
      })
      .catch(() => {
        if (!cancelled) setVideos([]);
      });

    return () => {
      cancelled = true;
    };
  }, [loadVideos]);

  useEffect(() => {
    let cancelled = false;

    loadObjects()
      .then((objects) => {
        if (!cancelled) setObjectTypes(objects);
      })
      .catch(() => {
        if (!cancelled) setObjectTypes([]);
      });

    return () => {
      cancelled = true;
    };
  }, [loadObjects]);

  useEffect(() => {
    if (!detectionBusy) {
      return;
    }

    let cancelled = false;

    const syncDetectionJobs = async () => {
      const jobs = await apiClient.getDetectionJobs();
      if (cancelled) {
        return;
      }

      const trackedJobs = jobs.filter((job) => activeDetectionJobIds.current.has(job.id));
      if (trackedJobs.length === 0) {
        return;
      }

      setDetectionJobs((prev) => {
        const next = { ...prev };
        for (const job of trackedJobs) {
          next[job.video_id] = job;
        }
        return next;
      });

      const completedCount = trackedJobs.filter((job) => job.status === "done").length;
      setDetectionSummary({
        mode: detectionRunTotal.current > 1 ? "all" : "selected",
        total: detectionRunTotal.current,
        done: detectionRunReused.current + completedCount,
        reused: detectionRunReused.current,
      });

      const hasPending = trackedJobs.some(
        (job) => job.status !== "done" && job.status !== "failed"
      );

      if (!hasPending) {
        activeDetectionJobIds.current.clear();
        setDetectionBusy(false);
        await refreshCatalog();
        if (!cancelled) {
          toast.success(
            trackedJobs.length === 1 ? "Detection completed" : "All detections completed"
          );
        }
      }
    };

    void syncDetectionJobs();
    const intervalId = window.setInterval(() => {
      void syncDetectionJobs();
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [detectionBusy, refreshCatalog]);

  const runQuery = useCallback(() => {
    setError(null);
    setResults(null);
    setTimeMs(null);
    setLoading(true);

    apiClient
      .queryVideos({
        colorFilter: filterColor,
        colorRefVideoId,
        searchInput,
        filterTypes,
        k: 5,
      })
      .then((data) => {
        setResults(data.results);
        setQueryObject(data.queryObject);
        setTimeMs(data.timeMs);
      })
      .catch((e) => {
        const msg = e instanceof Error ? e.message : "Something went wrong";
        setError(msg);
        try {
          toast.error(msg);
        } catch {}
      })
      .finally(() => setLoading(false));
  }, [colorRefVideoId, filterColor, filterTypes, searchInput]);

  const applyFilters = useCallback(() => {
    setFilterOpen(false);
    runQuery();
  }, [runQuery]);

  const resetFilters = useCallback(() => {
    setFilterColor("any");
    setFilterTypes([]);
  }, []);

  const toggleType = useCallback((type: string) => {
    setFilterTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  }, []);

  const submitDetectionJob = useCallback(async (videoId: string) => {
    const response = await apiClient.runDetection(videoId);
    const jobId = response.job_id;
    if (response.reused || !jobId) {
      setDetectionJobs((prev) => ({
        ...prev,
        [videoId]: {
          id: `reused-${videoId}`,
          video_id: videoId,
          status: "done",
        },
      }));
      return { queued: false, reused: true };
    }

    activeDetectionJobIds.current.add(jobId);
    setDetectionJobs((prev) => ({
      ...prev,
      [videoId]: {
        id: jobId,
        video_id: videoId,
        status: "queued",
      },
    }));
    return { queued: true, reused: false, jobId };
  }, []);

  const detectSelectedVideo = useCallback(async () => {
    if (detectionBusy || !selectedVideoId) {
      return false;
    }

    detectionRunTotal.current = 1;
    detectionRunReused.current = 0;
    setDetectionBusy(true);
    setDetectionSummary({ mode: "selected", total: 1, done: 0, reused: 0 });

    try {
      const result = await submitDetectionJob(selectedVideoId);
      if (result.reused) {
        detectionRunReused.current = 1;
        setDetectionBusy(false);
        setDetectionSummary({ mode: "selected", total: 1, done: 1, reused: 1 });
        await refreshCatalog();
        toast.success("Using existing detections from index");
      }
      return true;
    } catch (err) {
      setDetectionBusy(false);
      setDetectionSummary(null);
      const msg = err instanceof Error ? err.message : "Failed to start detection";
      toast.error(msg);
      return false;
    }
  }, [detectionBusy, selectedVideoId, submitDetectionJob]);

  const detectAllVideos = useCallback(async () => {
    if (detectionBusy || videos.length === 0) {
      return false;
    }

    detectionRunTotal.current = videos.length;
    detectionRunReused.current = 0;
    setDetectionBusy(true);
    setDetectionSummary({ mode: "all", total: videos.length, done: 0, reused: 0 });

    try {
      const results = await Promise.allSettled(
        videos.map((video) => submitDetectionJob(video.id))
      );
      const fulfilled = results.filter(
        (result): result is PromiseFulfilledResult<{ queued: boolean; reused: boolean; jobId?: string }> =>
          result.status === "fulfilled"
      );
      const startedCount = fulfilled.filter((result) => result.value.queued).length;
      const reusedCount = fulfilled.filter((result) => result.value.reused).length;
      detectionRunReused.current = reusedCount;
      setDetectionSummary({
        mode: "all",
        total: videos.length,
        done: reusedCount,
        reused: reusedCount,
      });

      if (startedCount === 0 && reusedCount > 0) {
        setDetectionBusy(false);
        await refreshCatalog();
        toast.success(`Using existing detections for ${reusedCount}/${videos.length} videos`);
        return true;
      }

      if (startedCount === 0 && reusedCount === 0) {
        throw new Error("No detection jobs could be started");
      }

      const failedCount = results.length - startedCount;
      if (failedCount > 0) {
        toast.warning(`Started ${startedCount}/${results.length} detection jobs`);
      }

      return true;
    } catch (err) {
      if (activeDetectionJobIds.current.size === 0) {
        setDetectionBusy(false);
        setDetectionSummary(null);
      }
      const msg = err instanceof Error ? err.message : "Failed to start detection";
      toast.error(msg);
      return false;
    }
  }, [detectionBusy, submitDetectionJob, videos]);

  return {
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
  };
}
