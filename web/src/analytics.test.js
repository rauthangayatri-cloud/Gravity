/** Pure data-display regressions; no browser or generated output required. */
import assert from "node:assert/strict";
import test from "node:test";
import { finiteValue, mean, meanRisk, riskCategory, scoreDistribution } from "./analytics.js";

test("unavailable values are not zero scores", () => {
  const rows = [{ success_score: null }, {}, { success_score: "" }, { success_score: 0 }, { success_score: 80 }];
  assert.equal(mean(rows, "success_score"), 40);
  assert.equal(scoreDistribution(rows)["<40"], 1);
  assert.equal(scoreDistribution(rows)["80+"], 1);
  assert.equal(mean([{}], "success_score"), null);
  assert.equal(finiteValue(Infinity), null);
});

test("risk aggregation ignores missing data and uses persisted thresholds", () => {
  assert.equal(meanRisk([{ academic_risk_probability: null }, { placement_risk_probability: 0.4 }]), 0.4);
  assert.equal(riskCategory(0.4, 0.3), "high");
  assert.equal(riskCategory(0.4, 0.5), "low");
  assert.equal(riskCategory(0.8, null), "unvalidated");
});
