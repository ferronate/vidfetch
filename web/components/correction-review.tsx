"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useCorrectionData } from "@/hooks/useCorrectionData";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface CorrectionReviewProps {
  videoId: string;
}

type TimeRange = { start: number; end: number };

function formatTimeRange(range: TimeRange) {
  return `${range.start.toFixed(1)}s-${range.end.toFixed(1)}s`;
}

function buildPerClassRanges(
  timeline: { t: number; objects: { class: string }[] }[],
  maxGapSeconds = 1.5
): Map<string, TimeRange[]> {
  const classTimes = new Map<string, number[]>();

  for (const entry of timeline) {
    const uniqueClassesInFrame = new Set(entry.objects.map((obj) => obj.class));
    for (const className of uniqueClassesInFrame) {
      const times = classTimes.get(className) ?? [];
      times.push(entry.t);
      classTimes.set(className, times);
    }
  }

  const ranges = new Map<string, TimeRange[]>();
  for (const [className, times] of classTimes.entries()) {
    const sortedTimes = [...times].sort((a, b) => a - b);
    if (sortedTimes.length === 0) continue;

    const segments: TimeRange[] = [];
    let segmentStart = sortedTimes[0];
    let segmentEnd = sortedTimes[0];

    for (let i = 1; i < sortedTimes.length; i += 1) {
      const nextTime = sortedTimes[i];
      if (nextTime - segmentEnd <= maxGapSeconds) {
        segmentEnd = nextTime;
      } else {
        segments.push({ start: segmentStart, end: segmentEnd });
        segmentStart = nextTime;
        segmentEnd = nextTime;
      }
    }

    segments.push({ start: segmentStart, end: segmentEnd });
    ranges.set(className, segments);
  }

  return ranges;
}

export function CorrectionReview({ videoId }: CorrectionReviewProps) {
  const {
    detections,
    corrections,
    rules,
    loading,
    addCorrection,
    addRule,
    toggleRule,
    deleteRule,
    deleteCorrection,
    generateRules,
  } = useCorrectionData(videoId);

  const [activeTab, setActiveTab] = useState<"review" | "corrections" | "rules">("review");
  const [editingDetection, setEditingDetection] = useState<{
    frameIndex: number;
    detectionIndex: number;
    newClass: string;
  } | null>(null);
  const [newRule, setNewRule] = useState({ pattern: "", target: "", confidence: "" });
  const [showAddRule, setShowAddRule] = useState(false);

  const onAddRule = async () => {
    const ok = await addRule(newRule.pattern, newRule.target, newRule.confidence);
    if (ok) {
      setNewRule({ pattern: "", target: "", confidence: "" });
      setShowAddRule(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <p className="text-muted-foreground">Loading correction data...</p>
      </div>
    );
  }

  if (!detections || detections.timeline.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8">
        <p className="text-muted-foreground">No detections found for this video.</p>
        <p className="text-sm text-muted-foreground mt-2">
          Run object detection first to enable corrections.
        </p>
      </div>
    );
  }

  const timeline = detections.timeline;
  const allClasses = [...new Set(detections.classes)].sort();
  const classRanges = buildPerClassRanges(timeline);

  return (
    <div className="space-y-4 min-w-0">
      {/* Tabs */}
      <div className="flex flex-wrap gap-2 border-b pb-1">
        {(["review", "corrections", "rules"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`rounded-md px-3 py-2 text-sm font-medium capitalize transition-colors ${
              activeTab === tab
                ? "bg-primary/10 text-foreground"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            }`}
          >
            {tab}
            {tab === "corrections" && corrections.length > 0 && (
              <span className="ml-1 rounded-full bg-primary/10 px-1.5 py-0.5 text-xs">
                {corrections.length}
              </span>
            )}
            {tab === "rules" && rules.length > 0 && (
              <span className="ml-1 rounded-full bg-primary/10 px-1.5 py-0.5 text-xs">
                {rules.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Review Tab */}
      {activeTab === "review" && (
        <div className="space-y-4">
          <div className="rounded-xl border border-border bg-muted/30 p-3">
            <p className="text-sm font-medium text-foreground">Object time ranges</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Per-class ranges where each object appears in this video.
            </p>
            <div className="mt-2 space-y-1.5">
              {allClasses.map((className) => {
                const ranges = classRanges.get(className) ?? [];
                const displayRanges = ranges.slice(0, 4);
                const hiddenCount = ranges.length - displayRanges.length;
                return (
                  <div key={className} className="text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">{className}:</span>{" "}
                    {displayRanges.length > 0
                      ? displayRanges.map(formatTimeRange).join(", ")
                      : "not found"}
                    {hiddenCount > 0 ? ` +${hiddenCount} more` : ""}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Detections list */}
          <div className="space-y-2">
            {timeline.map((entry, frameIndex) => (
              <Card key={`frame-${frameIndex}-${entry.t}`} className="relative">
                <CardHeader className="pb-2 pt-3">
                  <CardTitle className="text-sm">At {entry.t.toFixed(1)}s</CardTitle>
                  <CardDescription className="text-xs">
                    {entry.objects.length} detection(s)
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 pb-3 pt-0">
                  {entry.objects.map((det, detectionIndex) => (
                    <div
                      key={`det-${frameIndex}-${detectionIndex}-${det.class}`}
                      className="flex flex-col gap-3 rounded-lg border border-border p-3 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <span className="rounded bg-primary/10 px-2 py-1 text-xs font-mono">
                          #{detectionIndex + 1}
                        </span>
                        {editingDetection?.frameIndex === frameIndex &&
                        editingDetection?.detectionIndex === detectionIndex ? (
                          <div className="flex items-center gap-2">
                            <Select
                              value={editingDetection.newClass}
                              onValueChange={(v) =>
                                setEditingDetection({ ...editingDetection, newClass: v })
                              }
                            >
                              <SelectTrigger className="w-40">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {allClasses.map((cls) => (
                                  <SelectItem key={cls} value={cls}>
                                    {cls}
                                  </SelectItem>
                                ))}
                                <SelectItem value="__custom__">Custom...</SelectItem>
                              </SelectContent>
                            </Select>
                            <Button
                              size="sm"
                              onClick={() => {
                                if (
                                  editingDetection.newClass &&
                                  editingDetection.newClass !== det.class
                                ) {
                                  addCorrection(frameIndex, det, editingDetection.newClass);
                                }
                                setEditingDetection(null);
                              }}
                            >
                              Save
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setEditingDetection(null)}
                            >
                              Cancel
                            </Button>
                          </div>
                        ) : (
                          <div className="min-w-0">
                            <span className="font-medium break-words">{det.class}</span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              {(det.confidence * 100).toFixed(1)}%
                            </span>
                            {det.corrected && (
                              <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-800 dark:bg-amber-900 dark:text-amber-200">
                                corrected
                              </span>
                            )}
                          </div>
                        )}
                      </div>

                      <div className="flex shrink-0 gap-1">
                        {!editingDetection && (
                          <>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() =>
                                setEditingDetection({
                                  frameIndex,
                                  detectionIndex,
                                  newClass: det.class,
                                })
                              }
                            >
                              Relabel
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => addCorrection(frameIndex, det, "__DELETE__", "delete")}
                            >
                              Delete
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Corrections Tab */}
      {activeTab === "corrections" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">
              {corrections.length} correction(s) for this video
            </h3>
            <Button size="sm" variant="outline" onClick={generateRules}>
              Auto-generate rules
            </Button>
          </div>

          {corrections.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No corrections yet. Use the Review tab to correct detections.
            </p>
          ) : (
            <div className="space-y-2">
              {corrections.map((c) => (
                <Card key={c.id}>
                  <CardContent className="flex items-center justify-between p-3">
                    <div>
                      <span className="font-medium">{c.original_class}</span>
                      <span className="mx-2 text-muted-foreground">→</span>
                      <span className="font-medium">{c.corrected_class}</span>
                      <span className="ml-2 text-xs text-muted-foreground">
                        at {c.timestamp.toFixed(1)}s ({c.action})
                      </span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => deleteCorrection(c.id)}
                    >
                      Delete
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Rules Tab */}
      {activeTab === "rules" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">{rules.length} rule(s)</h3>
            <Button size="sm" onClick={() => setShowAddRule(!showAddRule)}>
              {showAddRule ? "Cancel" : "Add Rule"}
            </Button>
          </div>

          {showAddRule && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">New Rule</CardTitle>
                <CardDescription>
                  Automatically relabel detections matching a pattern
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">If class is</label>
                    <input
                      type="text"
                      value={newRule.pattern}
                      onChange={(e) =>
                        setNewRule({ ...newRule, pattern: e.target.value })
                      }
                      placeholder="e.g., dog"
                      className="mt-1 w-full rounded border px-2 py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Relabel as</label>
                    <input
                      type="text"
                      value={newRule.target}
                      onChange={(e) =>
                        setNewRule({ ...newRule, target: e.target.value })
                      }
                      placeholder="e.g., cat"
                      className="mt-1 w-full rounded border px-2 py-1 text-sm"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">
                    Only when confidence below (optional)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={newRule.confidence}
                    onChange={(e) =>
                      setNewRule({ ...newRule, confidence: e.target.value })
                    }
                    placeholder="e.g., 0.5"
                    className="mt-1 w-32 rounded border px-2 py-1 text-sm"
                  />
                </div>
                <Button size="sm" onClick={onAddRule}>
                  Add Rule
                </Button>
              </CardContent>
            </Card>
          )}

          {rules.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No rules yet. Add rules or auto-generate from corrections.
            </p>
          ) : (
            <div className="space-y-2">
              {rules.map((rule) => (
                <Card key={rule.id}>
                  <CardContent className="flex items-center justify-between p-3">
                    <div>
                      <span className="font-medium">{rule.pattern_class}</span>
                      <span className="mx-2 text-muted-foreground">→</span>
                      <span className="font-medium">{rule.target_class}</span>
                      {rule.confidence_threshold && (
                        <span className="ml-2 text-xs text-muted-foreground">
                          (conf &lt; {rule.confidence_threshold})
                        </span>
                      )}
                      <span className="ml-2 text-xs text-muted-foreground">
                        Used {rule.usage_count}x
                      </span>
                    </div>
                    <div className="flex gap-1">
                      <Button
                        size="sm"
                        variant={rule.enabled ? "outline" : "secondary"}
                        onClick={() => toggleRule(rule.id, rule.enabled)}
                      >
                        {rule.enabled ? "ON" : "OFF"}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => deleteRule(rule.id)}
                      >
                        Delete
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
