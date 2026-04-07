"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import type { Correction, CorrectionRule, Detection, VideoDetections } from "@/lib/types";

export function useCorrectionData(videoId: string) {
  const [detections, setDetections] = useState<VideoDetections | null>(null);
  const [corrections, setCorrections] = useState<Correction[]>([]);
  const [rules, setRules] = useState<CorrectionRule[]>([]);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [detectionsData, correctionsData, rulesData] = await Promise.all([
        apiClient.getVideoDetections(videoId),
        apiClient.getCorrections(videoId, 100),
        apiClient.getRules(false),
      ]);

      setDetections(detectionsData);
      setCorrections(correctionsData);
      setRules(rulesData);
    } catch (err) {
      console.error("Failed to load correction data:", err);
      toast.error("Failed to load correction data");
    } finally {
      setLoading(false);
    }
  }, [videoId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const addCorrection = useCallback(
    async (
      frameIndex: number,
      detection: Detection,
      correctedClass: string,
      action: string = "relabel"
    ) => {
      try {
        const ok = await apiClient.addCorrection({
          videoId,
          frameNumber: frameIndex,
          timestamp: detections?.timeline[frameIndex]?.t || 0,
          originalClass: detection.class,
          correctedClass,
          originalConfidence: detection.confidence,
          action,
          bbox: detection.bbox,
        });

        if (ok) {
          toast.success(`Corrected: ${detection.class} -> ${correctedClass}`);
          await loadData();
          return true;
        }

        toast.error("Failed to save correction");
        return false;
      } catch (err) {
        console.error("Failed to add correction:", err);
        toast.error("Failed to save correction");
        return false;
      }
    },
    [detections, loadData, videoId]
  );

  const addRule = useCallback(
    async (pattern: string, target: string, confidence?: string) => {
      if (!pattern || !target) return false;
      try {
        const ok = await apiClient.addRule({ pattern, target, confidence });

        if (ok) {
          await loadData();
          return true;
        }

        return false;
      } catch (err) {
        console.error("Failed to add rule:", err);
        return false;
      }
    },
    [loadData]
  );

  const toggleRule = useCallback(
    async (ruleId: number, currentEnabled: boolean) => {
      try {
        await apiClient.toggleRule(ruleId, currentEnabled);
        await loadData();
      } catch (err) {
        console.error("Failed to toggle rule:", err);
      }
    },
    [loadData]
  );

  const deleteRule = useCallback(
    async (ruleId: number) => {
      try {
        await apiClient.deleteRule(ruleId);
        await loadData();
      } catch (err) {
        console.error("Failed to delete rule:", err);
      }
    },
    [loadData]
  );

  const deleteCorrection = useCallback(
    async (correctionId: number) => {
      try {
        await apiClient.deleteCorrection(correctionId);
        await loadData();
      } catch (err) {
        console.error("Failed to delete correction:", err);
      }
    },
    [loadData]
  );

  const generateRules = useCallback(async () => {
    try {
      await apiClient.generateRules(3);
      await loadData();
    } catch (err) {
      console.error("Failed to generate rules:", err);
    }
  }, [loadData]);

  return {
    detections,
    corrections,
    rules,
    loading,
    loadData,
    addCorrection,
    addRule,
    toggleRule,
    deleteRule,
    deleteCorrection,
    generateRules,
  };
}
