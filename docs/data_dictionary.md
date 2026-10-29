# Raw synthetic data dictionary

## Current source schema

`student_master.csv` is the identifier crosswalk. The cleaner resolves each
source to `student_id`, then aligns released semester and event records to
monthly snapshots without copying later values into prior months. There are
nine source tables, plus the crosswalk.

All records and labels are synthetic. `student_id`, `cohort`, `department`, and `snapshot_month` join the seven monthly source tables. There are 600 `historical` students with known future labels and 800 `current` students whose labels are held separately for evaluation. Hidden generation traits—ability, effort, consistency, career interest, and department effect—are never exported.

| File | Measures | Notes |
| --- | --- | --- |
| `student_master.csv` | `student_id`, `roll_no`, `email`, `placement_id`, `department`, `year`, `section` | Clean crosswalk used only to resolve source identifiers. |
| `academic.csv` | `roll_no`, `semester`, `release_date`, `cgpa`, `average_mark`, `active_backlogs` | Semester grain; records are carried forward only after `release_date`. |
| `attendance.csv` | `attendance_pct`, `classes_attended`, `classes_scheduled` | `attendance_pct` deliberately includes strings such as `78%`. |
| `lms.csv` | `email`, `lms_logins`, `assignment_completion_pct`, `learning_minutes` | Monthly LMS activity; joined through the crosswalk. |
| `engagement.csv` | `engagement_score`, `event_participation`, `mentoring_sessions` | Participation and engagement snapshots. |
| `placement.csv` | `placement_id`, `assessment_date`, `placement_readiness`, `mock_interview_score`, `aptitude_score` | Assessment-event grain; values are carried forward only after each event. |
| `skills.csv` | `coding_score`, `communication_score`, `hackathons`, `certifications` | Skills and cumulative activity snapshots. |
| `feedback.csv` | `feedback_score`, `support_rating`, `response_count` | Feedback/satisfaction snapshots. |
| `subject_marks.csv` | `student_id`, `semester`, `release_date`, `subject`, `internal_marks`, `exam_marks` | Derives optional `weakest_subject` and `subject_marks_std`. |
| `subject_attendance.csv` | `student_id`, `semester`, `release_date`, `subject`, `attendance_pct` | Derives optional `subject_attendance_min`. |
| `outcomes_historical.csv` | `next_semester_cgpa_drop`, `new_backlog`, `placement_result` | Known future labels for the historical cohort only. |
| `outcomes_current_holdout.csv` | `next_semester_cgpa_drop`, `new_backlog`, `placement_result` | Generated labels held separately for current-cohort evaluation. |

## Intentional raw-data issues

The nine source files are altered only after clean ground truth is generated:
about 5% duplicate rows, varied missing metric values, source-system gaps,
score outliers, inconsistent department names, and percentage strings. Three
percent of LMS email rows have case, whitespace, or domain-typo issues. The
cleaning log records matched, unmatched, and ambiguous crosswalk joins per
source; only normalisable case and whitespace variants match.
