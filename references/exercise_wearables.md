# Domain profile: Exercise and wearable data (smartwatch, fitness tracker, sport)

Data from watches, bands, phones, bike computers, gym equipment and training logs: workouts, sensor time series, and daily health summaries.

> Synthetic data only. Physiology constants are rough rules of thumb for plausibility, not coaching or medical guidance.

## Entities
| Table | Grain | Key columns |
|---|---|---|
| `users` | one per person | user_id, age, sex, height_cm, weight_kg, resting_hr, max_hr, fitness_level (low/mid/high), primary_sport, device_model |
| `workouts` | one per session | workout_id, user_id, sport (run/ride/walk/swim/gym/hiit/...), start_ts, end_ts, duration_s, distance_m, avg_hr, max_hr, calories, elevation_gain_m, avg_cadence |
| `samples` | one per sensor reading | workout_id, ts, hr, speed_mps, cadence, power_w, altitude_m, lat, lon |
| `daily` | one per user per day | user_id, date, steps, active_minutes, resting_hr, hrv_rmssd, sleep_minutes, sleep_deep_min, sleep_rem_min, spo2_avg, readiness |
| `sleep` (optional) | one per night | user_id, sleep_start, sleep_end, stages (light/deep/REM/awake minutes) |

Keep sensor `samples` at a coarse rate for demos (5-60 s). A 1 Hz hour is 3,600 rows per workout and grows fast.

## Realism features
- **Heart rate follows effort with lag.** Model HR as a first-order response to a target intensity: `hr[t+1] = hr[t] + (target[t] - hr[t]) * k + noise`, k about 0.05-0.2 per sample step depending on rate. Warm-up ramp, steady state, intervals, cool-down recovery.
- **Cardiac drift:** in long steady efforts HR rises a few bpm per 10 minutes at constant pace, more in heat.
- **Max HR and zones:** approximate `max_hr` as 208 - 0.7*age (plus person-level offset, SD about 8 bpm); zones as fractions of max HR (about 50-60, 60-70, 70-80, 80-90, 90-100 percent). Resting HR roughly 40-80, lower for fitter users.
- **Pace, grade and HR are coupled:** uphill lowers speed and raises HR; descents do the opposite; fitter users run faster at the same HR.
- **Speed by sport:** walking about 1-2 m/s, running 2-6 m/s, cycling 4-15 m/s with higher peaks downhill; cadence walking 90-120 spm, running 150-190 spm, cycling 60-100 rpm; swimming uses laps and strokes rather than GPS.
- **Routes:** GPS as a smoothed random walk or looped path with consistent heading changes; altitude from a smooth noise profile; lat/lon within a fake region.
- **Training structure:** weekly pattern (hard day, easy days, rest day), progressive overload with periodic recovery weeks, tapering before an event, long breaks for injury or illness. Duration lognormal (most sessions 30-60 min, a few long), distance proportional to duration times speed.
- **Daily activity:** steps lognormal (median about 6-8k, weekday vs weekend differences, higher on workout days); active minutes correlated with steps and workouts.
- **Sleep:** duration about 7 h with SD about 1 h, later and longer on weekends, stage mix roughly light 50-60%, deep 13-23%, REM 20-25% (the exact split varies by device); sleep disrupted after very hard training or late sessions.
- **HRV (rMSSD):** lognormal per person, declining with age, dips after hard training and illness, recovers over days; resting HR moves inversely with HRV.
- **Calories:** MET x weight_kg x hours (running about 8-12 MET depending on pace; walking about 3-4), or HR-based estimates; deliberately noisy.
- **Individual differences:** per-user baselines (resting HR, pace at a given HR, preferred sport, schedule) with day-to-day variation around them.
- **Seasonality:** more outdoor volume in warm months, holiday dips, New Year spikes.
- **Sensor artefacts (flag with `is_anomaly` or a quality column):**
  - HR dropouts (zeros or gaps), spikes, and cadence lock (HR equals cadence)
  - GPS jitter, signal loss in tunnels or buildings, teleport jumps
  - device not worn (flat HR, no steps, missing nights), charging gaps
  - duplicate uploads from sync, wrong timezone or DST shifts
  - irregular sampling intervals

## Sanity rules
- `start_ts < end_ts`; `duration_s = end_ts - start_ts` (minus paused time if modelled); timestamps strictly increasing within a workout; no overlapping workouts for one user.
- HR within 30-220 (excluding labelled artefacts); `avg_hr <= max_hr`; session `max_hr` at most the user's `max_hr` plus a small margin.
- `distance_m` is approximately the integral of speed over time; `avg speed = distance / moving time`; consecutive GPS points imply a speed consistent with `speed_mps`.
- Speeds plausible for the sport; no running at cycling speeds; altitude changes consistent with `elevation_gain_m`.
- `steps` consistent with distance (step length about 0.7-0.8 m walking, longer running) for tracked walks and runs.
- Sleep stage minutes sum to total sleep time; sleep_end after sleep_start; at most one main sleep per night.
- Latitude in -90..90 and longitude in -180..180; no points over water for land sports unless modelled.
- `workouts` aggregates (avg_hr, max_hr, distance, duration) recomputed from `samples`, not sampled separately.
- Calories within a plausible range for duration, weight and intensity.

## Typical volumes
- 50-1,000 users, 3-5 workouts per user per week for 3-12 months, one `daily` row per user per day.
- Demo default: 50 users x 90 days, 20-40 s sampling, about 200k sample rows.

## Choosing a backend and evaluation notes
- **Sequences need a generator, not a row-wise GAN.** Use the rules backend or an LLM-written generator for workout time series (state-space or first-order HR response, smoothed paths). SDV has a sequential synthesizer (PAR) for sequence data, but it is not wired into this skill yet.
- **Tabular parts** (`users`, `daily`, `workouts` summaries) suit CTGAN or TVAE when real data exists; use `primary_keys`, `foreign_keys` (`workouts.user_id` -> `users.user_id`, `samples.workout_id` -> `workouts.workout_id`) and `derived` for aggregates.
- **Evaluation:** the default evaluator treats rows as independent. For time series also compare per-user distributions, autocorrelation, HR-vs-pace relationship, weekly volume patterns, and sequence similarity (for example DTW) on a time-based split. Those checks are not implemented in `evaluation/` yet.
- **Privacy:** GPS tracks and routines reveal home, work and habits. Use fake routes in a made-up region; do not copy or lightly perturb real tracks; treat any model trained on real tracks as privacy-sensitive. Few-shot to a hosted LLM should use aggregates, never raw tracks.

## Pitfalls
- Independent noise per second (HR that jumps randomly) instead of smooth dynamics.
- HR unrelated to speed, grade or fitness; identical users with identical schedules.
- Aggregates that do not match the underlying samples.
- Perfectly regular sampling and no missing data or sync artefacts.
- Ignoring timezones, DST and daily boundaries for `daily` rollups.
- Presenting physiological constants as individual-level medical advice.
