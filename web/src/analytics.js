/** Null-safe display aggregates; coverage and risk are not interchangeable. */
export function finiteValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function mean(items, field) {
  const values = items.map((item) => finiteValue(item[field])).filter((value) => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function meanRisk(items) {
  return mean(
    items.map((item) => {
      const risks = [item.academic_risk_probability, item.placement_risk_probability]
        .map(finiteValue)
        .filter((value) => value !== null);
      return { risk: risks.length ? Math.max(...risks) : null };
    }),
    "risk",
  );
}

export function scoreDistribution(items) {
  const labels = { "<40": 0, "40-49": 0, "50-59": 0, "60-69": 0, "70-79": 0, "80+": 0 };
  items.forEach((item) => {
    const score = finiteValue(item.success_score);
    if (score === null) return;
    const band = score < 40 ? "<40" : score < 50 ? "40-49" : score < 60 ? "50-59" : score < 70 ? "60-69" : score < 80 ? "70-79" : "80+";
    labels[band] += 1;
  });
  return labels;
}

export function riskCategory(value, threshold) {
  const risk = finiteValue(value);
  const cutoff = finiteValue(threshold);
  if (risk === null || cutoff === null) return "unvalidated";
  return risk >= cutoff ? "high" : "low";
}
