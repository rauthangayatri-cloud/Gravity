import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";
import { mean, meanRisk, riskCategory, scoreDistribution } from "./analytics.js";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");
const get = (path, options = {}) =>
  fetch(`${API}${path}`, options).then((r) => {
    if (!r.ok) throw new Error("Dashboard data is unavailable");
    return r.json();
  });
const label = (value) =>
  String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
const percent = (value) =>
  value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : `${Math.round(Number(value) * 100)}%`;

function useRoute() {
  const [route, setRoute] = useState(() => window.location.pathname);
  useEffect(() => {
    const change = () => setRoute(window.location.pathname);
    window.addEventListener("popstate", change);
    return () => window.removeEventListener("popstate", change);
  }, []);
  const navigate = (path) => {
    window.history.pushState({}, "", path);
    setRoute(path);
  };
  return { route, navigate };
}

function App() {
  const { route, navigate } = useRoute();
  const [mode, setMode] = useState("admin");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [facultyDepartment, setFacultyDepartment] = useState("");
  const departmentOptions = (data?.summary?.department_comparison || [])
    .map((row) => row.department)
    .filter(Boolean)
    .sort();
  const departmentOptionsKey = departmentOptions.join("|");
  const activeDatasetKey = data?.active?.dataset_id || data?.active?.mode || "loading";

  useEffect(() => {
    const availableDepartments = departmentOptionsKey ? departmentOptionsKey.split("|") : [];
    if (availableDepartments.length && !availableDepartments.includes(facultyDepartment)) {
      setFacultyDepartment(availableDepartments[0]);
    }
  }, [departmentOptionsKey, facultyDepartment]);

  useEffect(() => {
    let current = true;
    const controller = new AbortController();
    setData(null);
    setError("");
    const department =
      mode === "faculty" && facultyDepartment
        ? `?department=${encodeURIComponent(facultyDepartment)}&page_size=100`
        : "?page_size=100";
    const requestOptions = { signal: controller.signal };
    Promise.all([
      get("/summary", requestOptions),
      get("/worklist?top_n=25", requestOptions),
      get(`/students${department}`, requestOptions),
      get("/data/active", requestOptions),
    ])
      .then(([summary, worklist, students, active]) => {
        if (current) setData({ summary, worklist, students: students.items || [], active });
      })
      .catch((reason) => {
        if (current && reason.name !== "AbortError") setError(reason.message);
      });
    return () => {
      current = false;
      controller.abort();
    };
  }, [mode, refreshKey, facultyDepartment]);
  const nav = (path) => navigate(path);
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">SC</span>
          <div>
            <strong>Smart Campus</strong>
            <small>Student success operations</small>
          </div>
        </div>
        <nav>
          <button className={route === "/" ? "active" : ""} onClick={() => nav("/")}>
            Overview
          </button>
          <button className={route === "/worklist" ? "active" : ""} onClick={() => nav("/worklist")}>
            Worklist
          </button>
          <button className={route === "/segments" ? "active" : ""} onClick={() => nav("/segments")}>
            Segments
          </button>
          <button className={route === "/admin" ? "active" : ""} onClick={() => nav("/admin")}>
            Admin
          </button>
        </nav>
        <div className="top-actions">
          <span className="synthetic">Synthetic local demo · no authentication or faculty isolation</span>
          <div className="view-toggle">
            <button className={mode === "faculty" ? "active" : ""} onClick={() => setMode("faculty")}>
              Faculty
            </button>
            <button className={mode === "admin" ? "active" : ""} onClick={() => setMode("admin")}>
              Admin
            </button>
          </div>
          {mode === "faculty" && departmentOptions.length > 0 && (
            <label className="faculty-department">
              <span>Department</span>
              <select value={facultyDepartment} onChange={(event) => setFacultyDepartment(event.target.value)}>
                {departmentOptions.map((department) => (
                  <option key={department} value={department}>
                    {department}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </header>
      <main className="page-shell">
        {data && <DatasetControls active={data.active} onChanged={() => setRefreshKey((value) => value + 1)} />}
        {error && (
          <EmptyState title="Data unavailable" message={`${error}. Build outputs and start the API, then refresh.`} />
        )}
        {!error && !data && (
          <div className="loading">
            <span className="spinner" />
            Loading precomputed campus data…
          </div>
        )}
        {!error && data && route === "/" && (
          <Overview data={data} mode={mode} facultyDepartment={facultyDepartment} />
        )}
        {!error && data && route === "/worklist" && (
          <Worklist
            rows={data.worklist}
            mode={mode}
            facultyDepartment={facultyDepartment}
            onStudent={(id) => nav(`/student/${id}`)}
          />
        )}
        {!error && data && route === "/segments" && (
          <SegmentsPage datasetKey={activeDatasetKey} onStudent={(id) => nav(`/student/${id}`)} />
        )}
        {!error && data && route === "/admin" && <AdminPage datasetKey={activeDatasetKey} />}
        {!error && data && route.startsWith("/student/") && (
          <StudentPage
            id={route.split("/").pop()}
            datasetKey={activeDatasetKey}
            onBack={() => nav("/worklist")}
          />
        )}
      </main>
    </div>
  );
}

const STUDENT_FIELDS = [
  ["cgpa", "CGPA", 0, 10, 0.1],
  ["active_backlogs", "Active backlogs", 0, 20, 1],
  ["attendance_pct", "Attendance %", 0, 100, 1],
  ["lms_logins", "LMS logins", 0, 130, 1],
  ["assignment_completion_pct", "Assignment completion %", 0, 100, 1],
  ["coding_score", "Coding score", 0, 100, 1],
  ["aptitude_score", "Aptitude score", 0, 100, 1],
  ["mock_interview_score", "Mock interview score", 0, 100, 1],
  ["placement_readiness", "Placement readiness", 0, 100, 1],
  ["engagement_score", "Engagement score", 0, 100, 1],
  ["communication_score", "Communication score", 0, 100, 1],
  ["hackathons", "Hackathons", 0, 50, 1],
  ["certifications", "Certifications", 0, 50, 1],
  ["feedback_score", "Feedback score", 0, 100, 1],
];
const EMPTY_STUDENT = Object.fromEntries(STUDENT_FIELDS.map(([key]) => [key, ""]));
const DEPARTMENTS = ["Business", "Computer Science", "Digital Media", "Mechanical Engineering"];

function DatasetControls({ active, onChanged }) {
  const [panel, setPanel] = useState("");
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({ student_id: "", department: "Business", ...EMPTY_STUDENT });
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (url, options) => {
    setBusy(true);
    setStatus("");
    try {
      const response = await fetch(`${API}${url}`, options);
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload.detail || {};
        const errors = Array.isArray(detail.errors) ? detail.errors.join(" ") : detail.message || payload.detail;
        throw new Error(errors || "Request could not be completed.");
      }
      setStatus(payload.message || "Demo data restored.");
      setPanel("");
      onChanged();
    } catch (reason) {
      setStatus(reason.message);
    } finally {
      setBusy(false);
    }
  };
  const upload = (event) => {
    event.preventDefault();
    if (!file) {
      setStatus("Choose a CSV file first.");
      return;
    }
    const body = new FormData();
    body.append("file", file);
    submit("/data/upload", { method: "POST", body });
  };
  const addStudent = (event) => {
    event.preventDefault();
    const record = { student_id: form.student_id, department: form.department };
    STUDENT_FIELDS.forEach(([key]) => {
      if (form[key] !== "") record[key] = Number(form[key]);
    });
    submit("/data/students", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    });
  };
  return (
    <section className="dataset-controls card">
      <div>
        <p className="eyebrow">ACTIVE DATASET</p>
        <strong>{active?.mode === "user" ? active.label : "Demo data"}</strong>
        <span>{active?.student_count ?? 0} students</span>
      </div>
      <div className="dataset-buttons">
        <button className="secondary-button" onClick={() => setPanel(panel === "upload" ? "" : "upload")}>
          Upload student data
        </button>
        <button className="secondary-button" onClick={() => setPanel(panel === "add" ? "" : "add")}>
          Add student
        </button>
        <button
          className="text-button"
          disabled={busy || active?.mode !== "user"}
          onClick={() => submit("/data/reset", { method: "POST" })}
        >
          Reset to Demo
        </button>
      </div>
      {panel === "upload" && (
        <form className="runtime-form" onSubmit={upload}>
          <label>
            CSV file
            <input type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          </label>
          <p className="muted">
            Required: student_id, department, cgpa, active_backlogs, attendance_pct, lms_logins,
            assignment_completion_pct, coding_score, aptitude_score, mock_interview_score.
          </p>
          <button className="primary-button" disabled={busy} type="submit">
            {busy ? "Processing…" : "Process CSV"}
          </button>
        </form>
      )}
      {panel === "add" && (
        <form className="runtime-form" onSubmit={addStudent}>
          <div className="runtime-form-grid">
            <label>
              Student ID
              <input
                required
                value={form.student_id}
                onChange={(event) => setForm({ ...form, student_id: event.target.value })}
              />
            </label>
            <label>
              Department
              <select
                value={form.department}
                onChange={(event) => setForm({ ...form, department: event.target.value })}
              >
                {DEPARTMENTS.map((department) => (
                  <option key={department}>{department}</option>
                ))}
              </select>
            </label>
            {STUDENT_FIELDS.map(([key, title, min, max, step]) => (
              <label key={key}>
                {title}
                <input
                  required={STUDENT_FIELDS.slice(0, 8).some(([required]) => required === key)}
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={form[key]}
                  onChange={(event) => setForm({ ...form, [key]: event.target.value })}
                />
              </label>
            ))}
          </div>
          <button className="primary-button" disabled={busy} type="submit">
            {busy ? "Processing…" : "Add and analyse student"}
          </button>
        </form>
      )}
      {status && (
        <p className={status.includes("failed") || status.includes("must") ? "form-error" : "form-success"}>{status}</p>
      )}
    </section>
  );
}

function Overview({ data, mode, facultyDepartment }) {
  const faculty = mode === "faculty";
  const students = data.students;
  const localTiers = countBy(students, "score_tier");
  const localDistribution = scoreDistribution(students);
  const kpis = faculty
    ? {
        student_count: students.length,
        mean_success_score: mean(students, "success_score"),
        mean_risk: meanRisk(students),
        high_priority_count: students.filter((item) => Number(item.priority_score) >= 60).length,
      }
    : data.summary.kpis;
  const tierCounts = faculty ? localTiers : data.summary.tier_counts;
  const distribution = faculty ? localDistribution : data.summary.score_distribution;
  const departmentRows = faculty
    ? (data.summary.department_comparison || []).filter((row) => row.department === facultyDepartment)
    : data.summary.department_comparison || [];
  const bars = Object.entries(distribution || {}).map(([band, students]) => ({ band, students }));
  return (
    <>
      <section className="intro">
        <div>
          <p className="eyebrow">{faculty ? `FACULTY VIEW · ${facultyDepartment.toUpperCase()}` : "ADMIN OVERVIEW"}</p>
          <h1>Support students before they fall behind.</h1>
          <p>Scores, risks, and priorities are precomputed from synthetic data.</p>
        </div>
        <div className="asof">
          Current snapshot
          <br />
          <strong>Offline refresh</strong>
        </div>
      </section>
      <section className="kpi-grid">
        <Kpi label="Students" value={kpis.student_count} />
        <Kpi label="Mean Success Score" value={formatScore(kpis.mean_success_score)} />
        <Kpi label="Mean risk" value={percent(kpis.mean_risk)} tone="warm" />
        <Kpi label="High priority" value={kpis.high_priority_count} tone="alert" />
      </section>
      <section className="overview-grid">
        <article className="card chart-card">
          <CardHeading title="Success Score distribution" detail="Students by score band" />
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={bars}>
              <CartesianGrid vertical={false} stroke="#e7edf5" />
              <XAxis dataKey="band" tickLine={false} axisLine={false} />
              <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ fill: "#f2f6fb" }} />
              <Bar dataKey="students" radius={[5, 5, 0, 0]} fill="#4767d9" />
            </BarChart>
          </ResponsiveContainer>
        </article>
        <article className="card">
          <CardHeading title="Tier counts" detail="Current Success Score bands" />
          <div className="tier-list">
            {["Green", "Amber", "Red", "Insufficient data"].map((tier) => (
              <div className="tier-row" key={tier}>
                <span className={`tier-dot ${tier.toLowerCase().replaceAll(" ", "-")}`} />
                <span>{tier}</span>
                <strong>{tierCounts?.[tier] || 0}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>
      <section className="card comparison">
        <CardHeading
          title="Department comparison"
          detail={faculty ? `${facultyDepartment} only` : "Scores, risk, and priority by department"}
        />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Department</th>
                <th>Mean score</th>
                <th>Academic risk</th>
                <th>Placement risk</th>
                <th>Mean priority</th>
              </tr>
            </thead>
            <tbody>
              {departmentRows.length ? (
                departmentRows.map((row) => (
                  <tr key={row.department}>
                    <td>{row.department}</td>
                    <td>{formatScore(row.success_score)}</td>
                    <td>{percent(row.academic_risk_probability)}</td>
                    <td>{percent(row.placement_risk_probability)}</td>
                    <td>{formatScore(row.priority_score)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="empty-cell">
                    No department comparison is available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function Worklist({ rows, mode, facultyDepartment, onStudent }) {
  const [filters, setFilters] = useState({
    department: mode === "faculty" ? facultyDepartment : "",
    tier: "",
    segment: "",
  });
  const [sort, setSort] = useState({ key: "priority_score", direction: "desc" });
  useEffect(
    () => setFilters((current) => ({ ...current, department: mode === "faculty" ? facultyDepartment : "" })),
    [mode, facultyDepartment],
  );
  const options = (field) => [...new Set(rows.map((row) => row[field]).filter(Boolean))].sort();
  const filtered = rows
    .filter(
      (row) =>
        (!filters.department || row.department === filters.department) &&
        (!filters.tier || row.score_tier === filters.tier) &&
        (!filters.segment || row.segment === filters.segment),
    )
    .sort((a, b) => {
      const first = a[sort.key] ?? -Infinity;
      const second = b[sort.key] ?? -Infinity;
      return (first > second ? 1 : first < second ? -1 : 0) * (sort.direction === "asc" ? 1 : -1);
    });
  const toggleSort = (key) =>
    setSort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }));
  return (
    <>
      <section className="intro worklist-intro">
        <div>
          <p className="eyebrow">INTERVENTION QUEUE</p>
          <h1>Priority worklist</h1>
          <p>Top 25 students, ordered by precomputed priority.</p>
        </div>
        <span className="count-badge">{filtered.length} students</span>
      </section>
      <section className="card">
        <div className="filters">
          <Filter
            label="Department"
            value={filters.department}
            options={options("department")}
            onChange={(value) => setFilters({ ...filters, department: value })}
          />
          <Filter
            label="Tier"
            value={filters.tier}
            options={options("score_tier")}
            onChange={(value) => setFilters({ ...filters, tier: value })}
          />
          <Filter
            label="Segment"
            value={filters.segment}
            options={options("segment")}
            onChange={(value) => setFilters({ ...filters, segment: value })}
          />
        </div>
        {filtered.length ? (
          <div className="table-wrap">
            <table className="worklist-table">
              <thead>
                <tr>
                  <SortHeader label="Student" onClick={() => toggleSort("student_id")} />
                  <th>Trend</th>
                  <SortHeader label="Priority" onClick={() => toggleSort("priority_score")} />
                  <SortHeader label="Success Score" onClick={() => toggleSort("success_score")} />
                  <th>Risk</th>
                  <th>Tier</th>
                  <th>Segment</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((row) => (
                  <tr
                    key={row.student_id}
                    tabIndex="0"
                    onClick={() => onStudent(row.student_id)}
                    onKeyDown={(event) => event.key === "Enter" && onStudent(row.student_id)}
                  >
                    <td>
                      <strong>{row.student_id}</strong>
                      <small>{row.department}</small>
                    </td>
                    <td>
                      <Trend trend={row.trend} />
                    </td>
                    <td>
                      <strong>{formatScore(row.priority_score)}</strong>
                    </td>
                    <td>
                      <div className="risk-badges">
                        <RiskBadge label="A" value={row.academic_risk_probability} threshold={row.academic_risk_threshold} />
                        <RiskBadge label="P" value={row.placement_risk_probability} threshold={row.placement_risk_threshold} />
                      </div>
                    </td>
                    <td>
                      <span className={`tier-chip ${String(row.score_tier || "unknown").toLowerCase()}`}>
                        {row.score_tier || "—"}
                      </span>
                    </td>
                    <td>{row.segment || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState title="No matching students" message="Try clearing a filter or switching to Admin view." />
        )}
      </section>
    </>
  );
}

function SegmentsPage({ datasetKey, onStudent }) {
  const [segments, setSegments] = useState(null);
  const [members, setMembers] = useState({});
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const requestOptions = { signal: controller.signal };
    setSegments(null);
    setMembers({});
    setError("");
    get("/segments", requestOptions)
      .then(async (profiles) => {
        const grouped = await Promise.all(
          profiles.map(async (profile) => [profile.segment, await getAllStudents(profile.segment, requestOptions)]),
        );
        if (active) {
          setSegments(profiles);
          setMembers(Object.fromEntries(grouped));
        }
      })
      .catch((reason) => active && reason.name !== "AbortError" && setError(reason.message));
    return () => {
      active = false;
      controller.abort();
    };
  }, [datasetKey]);
  if (error) return <EmptyState title="Segments unavailable" message={error} />;
  if (!segments)
    return (
      <div className="loading">
        <span className="spinner" />
        Loading segment profiles…
      </div>
    );
  return (
    <section>
      <section className="intro">
        <div>
          <p className="eyebrow">STUDENT PROFILES</p>
          <h1>Segments with a practical next step.</h1>
          <p>Offline KMeans profiles are descriptive groupings, not diagnoses.</p>
        </div>
      </section>
      <div className="segment-stack">
        {segments.map((profile) => {
          const students = members[profile.segment] || [];
          const fields = Object.entries(profile).filter(
            ([key, value]) =>
              !["segment", "segment_id", "size", "chosen_k"].includes(key) && Number.isFinite(Number(value)),
          );
          return (
            <article className="card segment-card" key={profile.segment}>
              <div className="segment-card-head">
                <div>
                  <p className="eyebrow">SEGMENT {profile.segment_id}</p>
                  <h2>{profile.segment}</h2>
                  <p>
                    {profile.size} students · silhouette-selected k={profile.chosen_k}
                  </p>
                </div>
                <span className="count-badge">{students.length} loaded</span>
              </div>
              <div className="segment-content">
                <div>
                  <h3>Profile</h3>
                  <dl className="profile-grid">
                    {fields.map(([key, value]) => (
                      <div key={key}>
                        <dt>{label(key)}</dt>
                        <dd>{formatScore(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
                <div>
                  <h3>Assigned playbook</h3>
                  <PlaybookForSegment profile={profile} />
                </div>
              </div>
              <details className="segment-students">
                <summary>Students in this segment ({students.length})</summary>
                <div>
                  {students.length ? (
                    students.map((student) => (
                      <button key={student.student_id} onClick={() => onStudent(student.student_id)}>
                        {student.student_id}
                      </button>
                    ))
                  ) : (
                    <span>None available.</span>
                  )}
                </div>
              </details>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function PlaybookForSegment({ profile }) {
  const name = String(profile.segment || "").toLowerCase();
  const action = name.includes("attendance")
    ? ["Attendance recovery plan", "Faculty", "7d"]
    : name.includes("placement")
      ? ["Career readiness clinic", "Placement cell", "14d"]
      : name.includes("strugglers")
        ? ["Academic mentor check-in", "Faculty", "7d"]
        : ["Continue monitoring", "Mentor", "30d"];
  return (
    <p className="playbook-copy">
      <strong>{action[0]}</strong>
      <br />
      <span>
        {action[1]} · {action[2]}
      </span>
    </p>
  );
}

function AdminPage({ datasetKey }) {
  const [tab, setTab] = useState("weights");
  const [weights, setWeights] = useState({
    academic: 35,
    attendance: 20,
    lms: 15,
    placement_readiness: 15,
    engagement_skills: 10,
    feedback: 5,
  });
  const [scenario, setScenario] = useState(null);
  const [adminData, setAdminData] = useState({});
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const requestOptions = { signal: controller.signal };
    setAdminData({});
    setError("");
    Promise.all([get("/quality", requestOptions), get("/metrics", requestOptions)])
      .then(([quality, metrics]) => active && setAdminData({ quality, metrics }))
      .catch((reason) => active && reason.name !== "AbortError" && setError(reason.message));
    return () => {
      active = false;
      controller.abort();
    };
  }, [datasetKey]);
  useEffect(() => {
    let active = true;
    fetch(`${API}/weights`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weights }),
    })
      .then((response) => {
        if (!response.ok) throw new Error("Weight scenario unavailable");
        return response.json();
      })
      .then((data) => active && setScenario(data))
      .catch((reason) => active && setError(reason.message));
    return () => {
      active = false;
    };
  }, [weights, datasetKey]);
  const distribution = Object.entries(scenario?.score_distribution || {}).map(([band, students]) => ({
    band,
    students,
  }));
  return (
    <section>
      <section className="intro">
        <div>
          <p className="eyebrow">ADMIN CONTROLS</p>
          <h1>Test score weights and audit outputs.</h1>
          <p className="synthetic-copy">
            All performance, fairness, and quality content is computed on synthetic data.
          </p>
        </div>
      </section>
      <div className="admin-tabs">
        <button className={tab === "weights" ? "active" : ""} onClick={() => setTab("weights")}>
          Score weights
        </button>
        <button className={tab === "quality" ? "active" : ""} onClick={() => setTab("quality")}>
          Data quality
        </button>
        <button className={tab === "metrics" ? "active" : ""} onClick={() => setTab("metrics")}>
          Model metrics
        </button>
      </div>
      {error && <p className="form-error">{error}</p>}
      {tab === "weights" && (
        <section className="admin-grid">
          <article className="card">
            <CardHeading
              title="Success Score weights"
              detail="Scenario only; does not change the saved configuration"
            />
            <div className="weight-controls">
              {Object.entries(weights).map(([key, value]) => (
                <label key={key}>
                  <span>
                    {label(key)} <strong>{value}</strong>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="50"
                    value={value}
                    onChange={(event) => setWeights({ ...weights, [key]: Number(event.target.value) })}
                  />
                </label>
              ))}
            </div>
          </article>
          <article className="card">
            <CardHeading title="Scenario score distribution" detail="Recomputed from exported component values" />
            {scenario ? (
              <>
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={distribution}>
                    <CartesianGrid vertical={false} stroke="#e7edf5" />
                    <XAxis dataKey="band" tickLine={false} axisLine={false} />
                    <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                    <Tooltip />
                    <Bar dataKey="students" fill="#4767d9" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="rank-stats">
                  <span>
                    Rank Spearman <strong>{scenario.rank_change?.spearman_correlation ?? "—"}</strong>
                  </span>
                  <span>
                    Mean rank movement <strong>{scenario.rank_change?.mean_absolute_rank_change ?? "—"}</strong>
                  </span>
                  <span>
                    Max movement <strong>{scenario.rank_change?.max_absolute_rank_change ?? "—"}</strong>
                  </span>
                </div>
              </>
            ) : (
              <div className="loading">Refreshing scenario…</div>
            )}
          </article>
        </section>
      )}
      {tab === "quality" && <QualityPanel quality={adminData.quality} />}
      {tab === "metrics" && <MetricsPanel metrics={adminData.metrics} />}
    </section>
  );
}

function QualityPanel({ quality }) {
  if (!quality) return <div className="loading">Loading data quality…</div>;
  if (quality.available === false)
    return (
      <EmptyState
        title="Cleaning log unavailable"
        message={quality.message || "No precomputed cleaning log is available."}
      />
    );
  const rows = Array.isArray(quality)
    ? quality
    : Object.entries(quality).map(([key, value]) => ({
        key,
        value: typeof value === "object" ? JSON.stringify(value) : value,
      }));
  return (
    <section className="card quality-card">
      <CardHeading title="Data quality log" detail="Precomputed cleaning-stage results" />
      <table>
        <thead>
          <tr>
            <th>Check</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key || row.check}>
              <td>{row.key || row.check}</td>
              <td>{row.value || row.result}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function MetricsPanel({ metrics }) {
  if (!metrics) return <div className="loading">Loading synthetic model metrics…</div>;
  const targets = metrics.targets || {};
  const selectedMetrics = (models) => models[models.recommended_model] || models.xgboost || {};
  return (
    <section className="metrics-stack">
      <p className="synthetic-notice">
        Synthetic data only — metrics are demonstration outputs, not real-world performance claims.
      </p>
      <p className="muted">{metrics.data_note} {metrics.evaluation_scope}</p>
      {Object.entries(targets).map(([target, models]) => (
        <article className="card metric-card" key={target}>
          <CardHeading
            title={`${label(target)} model metrics`}
            detail={`Selected: ${label(models.recommended_model || "legacy")}. Unseen synthetic students, not temporal validation.`}
          />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Decision threshold</th>
                  <th>Confusion matrix</th>
                </tr>
              </thead>
              <tbody>
                {["baseline_rule", "logistic_regression", "xgboost"]
                  .filter((name) => models[name])
                  .map((name) => (
                    <tr key={name}>
                      <td>{label(name)}</td>
                      <td>{models[name].roc_auc ?? "—"}</td>
                      <td>{models[name].pr_auc ?? "—"}</td>
                      <td>{models[name].precision ?? "—"}</td>
                      <td>{models[name].recall ?? "—"}</td>
                      <td>{models[name].threshold ?? "Legacy: 0.5"}</td>
                      <td>
                        {Array.isArray(models[name].confusion_matrix)
                          ? models[name].confusion_matrix.map((row) => `[${row.join(", ")}]`).join(" ")
                          : "—"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          {models.xgboost && (
            <div className="metric-foot">
              <span>
                Selected model Recall@10%: <strong>{selectedMetrics(models).recall_at_top_10pct ?? "—"}</strong>
              </span>
              <span>
                Development threshold (80% recall objective, not guarantee): <strong>{selectedMetrics(models).threshold_for_approximately_80pct_recall ?? "—"}</strong>
              </span>
              <span>
                Evaluation precision at threshold: <strong>{selectedMetrics(models).precision_at_that_threshold ?? "—"}</strong>
              </span>
            </div>
          )}
          <FairnessTable rows={selectedMetrics(models).fairness_by_department || []} />
        </article>
      ))}
    </section>
  );
}

function FairnessTable({ rows }) {
  return (
    <div className="fairness">
      <h3>Department error analysis — not a fairness guarantee</h3>
      {rows.length ? (
        <table>
          <thead>
            <tr>
              <th>Department</th>
              <th>Students</th>
              <th>Positive / negative labels</th>
              <th>Recall</th>
              <th>False-positive rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.department}>
                <td>{row.department}</td>
                <td>{row.students}</td>
                <td>{row.positives ?? "—"} / {row.negatives ?? "—"}</td>
                <td>{row.recall ?? "—"} {row.recall_95ci && `(${row.recall_95ci.join("–")}, 95% CI)`}</td>
                <td>{row.false_positive_rate ?? "—"} {row.false_positive_rate_95ci && `(${row.false_positive_rate_95ci.join("–")}, 95% CI)`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">No precomputed fairness rows are available.</p>
      )}
    </div>
  );
}

async function getAllStudents(segment, options = {}) {
  const first = await get(`/students?segment=${encodeURIComponent(segment)}&page_size=100&page=1`, options);
  const pages = Math.ceil(first.total / 100);
  const remaining = await Promise.all(
    Array.from({ length: Math.max(0, pages - 1) }, (_, index) =>
      get(`/students?segment=${encodeURIComponent(segment)}&page_size=100&page=${index + 2}`, options),
    ),
  );
  return [...(first.items || []), ...remaining.flatMap((page) => page.items || [])];
}

function StudentPage({ id, datasetKey, onBack }) {
  const [student, setStudent] = useState(null);
  const [whatIf, setWhatIf] = useState(null);
  const [error, setError] = useState("");
  const [intervention, setIntervention] = useState("idle");
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const requestOptions = { signal: controller.signal };
    setStudent(null);
    setWhatIf(null);
    setError("");
    setIntervention("idle");
    Promise.all([get(`/students/${id}`, requestOptions), get(`/students/${id}/what_if`, requestOptions)])
      .then(([profile, sensitivity]) => {
        if (active) {
          setStudent(profile);
          setWhatIf(sensitivity);
        }
      })
      .catch((reason) => {
        if (active && reason.name !== "AbortError") setError(reason.message);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [id, datasetKey]);
  const startIntervention = async () => {
    const action = student.actions?.[0] || {};
    setIntervention("saving");
    try {
      const response = await fetch(`${API}/interventions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_id: id,
          type: action.recommended_action || "Student support follow-up",
          owner: action.owner || "Faculty",
          start_date: new Date().toISOString().slice(0, 10),
          before_snapshot: student.trajectory?.snapshot_month || "latest pre-intervention snapshot",
        }),
      });
      if (!response.ok) throw new Error("Unable to start intervention");
      setIntervention("saved");
    } catch (reason) {
      setIntervention(reason.message);
    }
  };
  if (error) return <EmptyState title="Student data unavailable" message={error} />;
  if (!student)
    return (
      <div className="loading">
        <span className="spinner" />
        Loading student profile…
      </div>
  );
  const score = student.score || {};
  const hasObservedTrajectory = Number(student.trajectory?.trajectory_points) >= 2;
  const trajectory = hasObservedTrajectory ? chartTrajectory(student.trajectory || {}) : [];
  const components = Object.entries(score.components || {}).filter(([key]) => key.endsWith("_component"));
  const drivers = (student.shap_drivers || []).map((driver) => ({
    ...driver,
    displayFeature: `${label(driver.target)} · ${label(driver.model || "legacy_xgboost")} · ${label(driver.feature.replace("_current", ""))}`,
    signed: driver.direction === "increases risk" ? Number(driver.magnitude) : -Number(driver.magnitude),
  }));
  const followups = (student.simulated_followups || []).map((point) => ({
    ...point,
    risk_percent: Number(point.risk) * 100,
  }));
  return (
    <section className="student-page">
      <button className="back-link" onClick={onBack}>
        ← Back to worklist
      </button>
      <header className="student-hero">
        <div>
          <p className="eyebrow">STUDENT SUPPORT VIEW</p>
          <h1>{id}</h1>
          <p>{student.segment || "Unsegmented"}</p>
        </div>
        <div className="student-score">
          <span>Success Score</span>
          <strong>
            {score.status === "Insufficient data" ? "Insufficient data" : formatScore(score.success_score)}
          </strong>
          <small>
            {score.status === "Insufficient data"
              ? "Component-weight coverage below 50%"
              : `${percent(score.confidence)} component-weight coverage · ${score.tier || "—"} tier`}
          </small>
        </div>
      </header>
      <section className="student-metrics">
        <MetricCard label="Academic risk" value={percent(student.risks?.academic_risk_probability)} />
        <MetricCard label="Placement risk" value={percent(student.risks?.placement_risk_probability)} />
        <MetricCard label={hasObservedTrajectory ? "Heuristic projected risk index" : "Current max-risk index (no forecast)"} value={percent(student.risks?.projected_risk)} />
        <MetricCard label="Risk trend" value={<Trend trend={student.trajectory?.trend} />} />
      </section>
      {student.risks?.runtime_model_note && <p className="synthetic-notice">{student.risks.runtime_model_note}</p>}
      <section className="student-grid">
        <article className="card trajectory-card">
          <CardHeading
            title="Risk trajectory"
            detail={
              hasObservedTrajectory
                ? "Heuristic fit to snapshot risk indices; residual-spread band is not a confidence interval"
                : "One uploaded snapshot is available; no trajectory is shown."
            }
          />
          {hasObservedTrajectory ? (
            <ResponsiveContainer width="100%" height={255}>
              <AreaChart data={trajectory}>
                <CartesianGrid vertical={false} stroke="#e7edf5" />
                <XAxis dataKey="month" tickLine={false} axisLine={false} />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(value) => `${Math.round(value * 100)}%`}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip formatter={(value) => percent(value)} />
                <Area type="monotone" dataKey="upper" stroke="none" fill="#dfe7ff" fillOpacity={0.9} />
                <Area type="monotone" dataKey="lower" stroke="none" fill="#fff" fillOpacity={1} />
                <Line type="monotone" dataKey="risk" stroke="#4767d9" strokeWidth={3} dot={{ r: 3 }} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="single-snapshot-state">
              <strong>{percent(student.risks?.risk)}</strong>
              <span>Current combined risk from the uploaded snapshot</span>
            </div>
          )}
        </article>
        <article className="card">
          <CardHeading title="Recommended action" detail="Rule-based from top SHAP drivers" />
          {student.actions?.length ? (
            <>
              {student.actions.map((action) => (
                <div className="action-item" key={`${action.recommended_action}-${action.owner}`}>
                  <strong>{action.recommended_action}</strong>
                  <span>
                    {action.owner} · {action.timeline}
                  </span>
                </div>
              ))}
              <button
                className="primary-button intervention-button"
                disabled={intervention === "saving" || intervention === "saved"}
                onClick={startIntervention}
              >
                {intervention === "saving"
                  ? "Starting…"
                  : intervention === "saved"
                    ? "Intervention started"
                    : "Start intervention"}
              </button>
              {intervention !== "idle" && intervention !== "saving" && intervention !== "saved" && (
                <p className="form-error">{intervention}</p>
              )}
            </>
          ) : (
            <p className="muted">No action rule matched the currently exported drivers.</p>
          )}
          <div className="student-summary">
            <p className="eyebrow">SUMMARY</p>
            <p>{student.summary_text || "No precomputed summary is available."}</p>
          </div>
        </article>
      </section>
      {followups.length > 0 && (
        <section className="card simulation-card">
          <CardHeading
            title="Score change after intervention (simulated)"
            detail="SIMULATED illustrative follow-up snapshots at 30 and 60 days; not observed outcomes or causal estimates."
          />
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={followups}>
              <CartesianGrid vertical={false} stroke="#e7edf5" />
              <XAxis dataKey="timepoint" tickLine={false} axisLine={false} />
              <YAxis yAxisId="score" domain={[0, 100]} tickLine={false} axisLine={false} />
              <YAxis
                yAxisId="risk"
                orientation="right"
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip />
              <Legend />
              <Line
                yAxisId="score"
                type="monotone"
                dataKey="success_score"
                name="Success Score"
                stroke="#4767d9"
                strokeWidth={3}
                dot={{ r: 4 }}
              />
              <Line
                yAxisId="risk"
                type="monotone"
                dataKey="risk_percent"
                name="Risk"
                stroke="#d35b6a"
                strokeWidth={3}
                strokeDasharray="5 3"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}
      <section className="card what-if-card">
        <CardHeading
          title="What-if sensitivity"
          detail={whatIf?.data_note || "Model-based sensitivity on synthetic data, not a causal estimate."}
        />
        {whatIf?.items?.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Target</th>
                  <th>Change</th>
                  <th>Before</th>
                  <th>After</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {whatIf.items.map((row) => (
                  <tr key={`${row.target}-${row.action}`}>
                    <td>{label(row.target)}</td>
                    <td>{row.action}</td>
                    <td>{percent(row.risk_before)}</td>
                    <td>{percent(row.risk_after)}</td>
                    <td className={Number(row.delta) < 0 ? "risk-down" : "risk-up"}>{percent(row.delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No precomputed sensitivity rows are available for this student.</p>
        )}
      </section>
      <section className="student-grid">
        <article className="card">
          <CardHeading title="Indicator strengths" detail="Current score components; no historical lines are fabricated." />
          <div className="sparks">
            {components.length ? (
              components.map(([key, value]) => (
                <ComponentStrength key={key} label={label(key.replace("_component", ""))} value={Number(value)} />
              ))
            ) : (
              <p className="muted">No component values are available.</p>
            )}
          </div>
        </article>
        <article className="card shap-card">
          <CardHeading title="Top risk drivers" detail="Target-labelled base-model log-odds contributions; not calibrated probability changes or causal effects" />
          {drivers.length ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={drivers} layout="vertical" margin={{ left: 18 }}>
                <CartesianGrid horizontal={false} stroke="#e7edf5" />
                <XAxis type="number" tickLine={false} axisLine={false} />
                <YAxis
                  type="category"
                  dataKey="displayFeature"
                  width={210}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => label(value.replace("_current", ""))}
                />
                <Tooltip formatter={(value) => Math.abs(Number(value)).toFixed(3)} />
                <Bar dataKey="signed" radius={[0, 4, 4, 0]} fill="#d35b6a" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">No precomputed SHAP drivers are available.</p>
          )}
        </article>
      </section>
    </section>
  );
}

function MetricCard({ label: title, value }) {
  return (
    <article className="student-metric">
      <span>{title}</span>
      <strong>{value}</strong>
    </article>
  );
}
function ComponentStrength({ label: title, value }) {
  const boundedValue = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  return (
    <div className="component-strength-row">
      <span>{title}</span>
      <div className="component-strength-track" aria-label={`${title} strength ${percent(boundedValue)}`}>
        <span style={{ width: `${boundedValue * 100}%` }} />
      </div>
      <strong>{percent(boundedValue)}</strong>
    </div>
  );
}
function chartTrajectory(trajectory) {
  const slope = Number(trajectory.risk_slope) || 0;
  const projected = Number(trajectory.projected_risk) || 0;
  const low = Number(trajectory.risk_band_low);
  const high = Number(trajectory.risk_band_high);
  return [0, 1, 2, 3]
    .map((index) => {
      const risk = Math.max(0, Math.min(1, projected - slope * (4 - index)));
      return {
        month: `M${index + 2}`,
        risk,
        lower: Math.max(0, risk - (Number.isFinite(low) ? Math.abs(projected - low) : 0.04)),
        upper: Math.min(1, risk + (Number.isFinite(high) ? Math.abs(high - projected) : 0.04)),
      };
    })
    .concat([
      {
        month: "Next",
        risk: projected,
        lower: Number.isFinite(low) ? low : Math.max(0, projected - 0.04),
        upper: Number.isFinite(high) ? high : Math.min(1, projected + 0.04),
      },
    ]);
}
function CardHeading({ title, detail }) {
  return (
    <div className="card-heading">
      <div>
        <h2>{title}</h2>
        <p>{detail}</p>
      </div>
    </div>
  );
}
function Kpi({ label: title, value, tone = "" }) {
  return (
    <article className={`kpi ${tone}`}>
      <span>{title}</span>
      <strong>{value ?? "—"}</strong>
    </article>
  );
}
function Filter({ label: title, value, options, onChange }) {
  return (
    <label>
      {title}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All</option>
        {options.map((option) => (
          <option value={option} key={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
function SortHeader({ label: title, onClick }) {
  return (
    <th>
      <button className="sort-button" onClick={onClick}>
        {title} ↕
      </button>
    </th>
  );
}
function Trend({ trend }) {
  const icon = trend === "declining" ? "↓" : trend === "improving" ? "↑" : "→";
  return (
    <span className={`trend ${trend || "stable"}`}>
      {icon} {label(trend || "stable")}
    </span>
  );
}
function RiskBadge({ label: short, value, threshold }) {
  return (
    <span className={`risk-badge ${riskCategory(value, threshold)}`} title={threshold == null ? "Legacy: no validated runtime threshold" : `Development-selected threshold: ${percent(threshold)}`}>
      {short} {percent(value)}
    </span>
  );
}
function EmptyState({ title, message }) {
  return (
    <section className="empty-state">
      <h2>{title}</h2>
      <p>{message}</p>
    </section>
  );
}
function countBy(items, field) {
  return items.reduce(
    (result, item) => ({ ...result, [item[field] || "Unknown"]: (result[item[field] || "Unknown"] || 0) + 1 }),
    {},
  );
}
function formatScore(value) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(1);
}

createRoot(document.getElementById("root")).render(<App />);
