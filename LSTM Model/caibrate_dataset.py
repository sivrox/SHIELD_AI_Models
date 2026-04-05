#DATASET CALIBRATOR

import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score

# === configuration ===
n_patients = 30
n_hospitals = 3
duration_hours = 48
sample_interval = 5 # seconds
seed = 42
output_file = 'LSTM Model/shield_training_dataset.csv'

# File paths - adjust these to match your folder structure
uci_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/uci_data')
mitbih_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/mit_bih_data')

# Activity states
sleep, rest, active, exercise = 0, 1, 2, 3

# ============================================================
# step 1: Load UCI Heart Disease Dataset
# Extracts: age, resting bp, max hr, disease prevalence by age
# ============================================================

def load_uci_data():
    #Load all 4 uci center files and extract clinical distributions
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    files = {
        'Cleveland': os.path.join(uci_dir, 'processed.cleveland.data'),
        'Hungarian': os.path.join(uci_dir, 'processed.hungarian.data'),
        'Switzerland': os.path.join(uci_dir, 'processed.switzerland.data'),
        'va': os.path.join(uci_dir, 'processed.va.data'),
    }

    all_data = []
    for name, path in files.items():
        df = pd.read_csv(path, names=columns, na_values='?')
        all_data.append(df)
        print(f"  Loaded {name}: {len(df)} patients")
    uci = pd.concat(all_data, ignore_index=True)
    uci['has_disease'] = (uci['num'] > 0).astype(int)
    print(f"  Total: {len(uci)} patients")

    # Extract distributions
    healthy = uci[uci['has_disease'] == 0]
    disease = uci[uci['has_disease'] == 1]
    dist = {
        'bp_healthy_mean': healthy['trestbps'].dropna().mean(),
        'bp_healthy_std': healthy['trestbps'].dropna().std(),
        'bp_disease_mean': disease['trestbps'].dropna().mean(),
        'bp_disease_std': disease['trestbps'].dropna().std(),
        'max_hr_healthy': healthy['thalach'].dropna().mean(),
        'max_hr_disease': disease['thalach'].dropna().mean(),
        'age_mean': uci['age'].mean(),
        'age_std': uci['age'].std(),
    }
    print(f"\nExtracted from uci:")
    print(f"bp (healthy): {dist['bp_healthy_mean']:.1f} ± {dist['bp_healthy_std']:.1f} mmHg")
    print(f"bp (disease): {dist['bp_disease_mean']:.1f} ± {dist['bp_disease_std']:.1f} mmHg")
    print(f"max hr (healthy): {dist['max_hr_healthy']:.1f} bpm")
    print(f"max hr (disease): {dist['max_hr_disease']:.1f} bpm")
    return dist


# ============================================================
# step 2: Load mit-bih Arrhythmia Database
# Reads ecg files, detects r-peaks, computes real hr and hrv
# Requires: pip install wfdb
# ============================================================

def load_mitbih_data():
    """
    Process mit-bih ecg recordings to extract real hr and hrv distributions.
    Each record has .dat (signal), .hea (header), .atr (annotations) files.
    We use the beat annotations to find r-peak positions, then compute:
    - hr from r-r intervals (60 / interval_seconds)
    - hrv as rmssd (root mean square of successive r-r differences)
    """
    try:
        import wfdb
    except ImportError:
        print("  warning: wfdb not installed. Run: pip install wfdb")
        print("  Using published mit-bih reference values instead.")
        return get_mitbih_fallback_values()

    # Standard mit-bih record numbers
    records = ['100','101','102','103','104','105','106','107','108','109',
               '111','112','113','114','115','116','117','118','119',
               '121','122','123','124','200','201','202','203','205',
               '207','208','209','210','211','212','213','214','215',
               '217','219','220','221','222','223','228','230','231','232','233','234']

    all_hr, all_hrv = [], []
    records_processed = 0

    for rec_id in records:
        rec_path = os.path.join(mitbih_dir, rec_id)
        if not os.path.exists(rec_path + '.dat'):
            continue
        try:
            # Read beat annotations (r-peak positions)
            ann = wfdb.rdann(rec_path, 'atr')
            # Filter to only beat annotations (not comments or rhythm changes)
            beat_types = ['n','l','r','b','a','a','j','s','v','r','f','e','j','n','e','/','f','q','?']
            beat_mask = [s in beat_types for s in ann.symbol]
            beat_samples = ann.sample[beat_mask]

            if len(beat_samples) < 10:
                continue

            # Read header to get sampling frequency
            header = wfdb.rdheader(rec_path)
            fs = header.fs # typically 360 Hz for mit-bih

            # Compute r-r intervals in seconds
            rr_intervals = np.diff(beat_samples) / fs

            # Filter out unrealistic intervals (< 0.3s or > 2.0s)
            rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
            if len(rr_valid) < 5:
                continue

            # Heart rate from r-r intervals
            hr_values = 60.0 / rr_valid
            all_hr.extend(hr_values.tolist())

            # hrv as rmssd (root mean square of successive differences)
            # This is the standard short-term hrv metric
            rr_diff = np.diff(rr_valid)
            rmssd = np.sqrt(np.mean(rr_diff ** 2)) * 1000 # convert to milliseconds
            all_hrv.append(rmssd)

            records_processed += 1
        except Exception as e:
            continue # skip corrupted records

    if records_processed == 0:
        print("  warning: No mit-bih records found. Using published values.")
        return get_mitbih_fallback_values()

    all_hr = np.array(all_hr)
    all_hrv = np.array(all_hrv)

    dist = {
        'hr_mean': float(np.mean(all_hr)),
        'hr_std': float(np.std(all_hr)),
        'hr_min': float(np.percentile(all_hr, 5)),
        'hr_max': float(np.percentile(all_hr, 95)),
        'hrv_mean': float(np.mean(all_hrv)),
        'hrv_std': float(np.std(all_hrv)),
        'hrv_min': float(np.percentile(all_hrv, 10)),
        'hrv_max': float(np.percentile(all_hrv, 90)),
        'source': 'extracted',
    }

    print(f"Processed {records_processed} mit-bih records")
    print(f"\nExtracted from mit-bih ecg:")
    print(f"hr: {dist['hr_mean']:.1f} ± {dist['hr_std']:.1f} bpm (range: {dist['hr_min']:.0f}-{dist['hr_max']:.0f})")
    print(f"hrv (rmssd): {dist['hrv_mean']:.1f} ± {dist['hrv_std']:.1f} ms (range: {dist['hrv_min']:.0f}-{dist['hrv_max']:.0f})")
    return dist


def get_mitbih_fallback_values():
    """Published mit-bih statistics (Moody & Mark 2001, Shaffer & Ginsberg 2017)."""
    return {
        'hr_mean': 75.0, 'hr_std': 18.0,
        'hr_min': 50.0, 'hr_max': 110.0,
        'hrv_mean': 42.0, 'hrv_std': 15.0,
        'hrv_min': 18.0, 'hrv_max': 70.0,
        'source': 'published',
    }


# ============================================================
# step 3: Patient Profile Definition
# Each patient's baseline vitals are derived from the real
# distributions extracted in Steps 1 and 2
# ============================================================

@dataclass
class PatientProfile:
    patient_id: int
    age: int
    fitness: float # 0.0 (sedentary) to 1.0 (athletic)
    has_hypertension: bool
    has_obesity: bool
    has_sleep_disorder: bool
    hospital_id: int

    resting_hr: float = 0.0
    resting_hrv: float = 0.0
    baseline_bp_s: float = 0.0
    baseline_bp_d: float = 0.0
    avg_sleep: float = 7.0

    def compute_baselines(self, uci_dist, mitbih_dist):
        """Set physiological baselines using real clinical distributions."""
        age_norm = (self.age - 20) / 57
        unfit = 1.0 - self.fitness

        # Resting hr from mit-bih distribution
        base_hr = mitbih_dist['hr_mean'] # real mean from ecg data
        self.resting_hr = base_hr - 15 + age_norm * 12 + unfit * 18
        if self.has_obesity:
            self.resting_hr += 7
        self.resting_hr = np.clip(self.resting_hr, mitbih_dist['hr_min'], mitbih_dist['hr_max'])

        # hrv from mit-bih distribution
        base_hrv = mitbih_dist['hrv_mean'] # real rmssd from ecg data
        self.resting_hrv = base_hrv + 20 - age_norm * 28 - unfit * 15
        if self.has_hypertension:
            self.resting_hrv -= 10
        self.resting_hrv = np.clip(self.resting_hrv, mitbih_dist['hrv_min'], 95)

        # Systolic bp from uci distribution
        base_bp = uci_dist['bp_healthy_mean'] # real mean from 920 patients
        self.baseline_bp_s = base_bp - 12 + age_norm * 20 + unfit * 8
        if self.has_hypertension:
            self.baseline_bp_s += 18

        # Diastolic bp (clinical pulse pressure: 30-50 mmHg)
        pulse_pressure = 40 + age_norm * 8
        self.baseline_bp_d = self.baseline_bp_s - pulse_pressure
        if self.has_hypertension:
            self.baseline_bp_d += 10

        # Sleep hours (uae average ~6.5h)
        self.avg_sleep = 7.0 - unfit * 1.2
        if self.has_sleep_disorder:
            self.avg_sleep -= 1.5
        self.avg_sleep = np.clip(self.avg_sleep, 3.5, 9.0)


# ============================================================
# step 4: Patient Cohort Generator
# ============================================================

def generate_cohort(uci_dist, mitbih_dist):
    """Generate uae-focused cohort using real disease prevalence from uci."""
    rng = np.random.RandomState(seed)
    patients = []

    for i in range(n_patients):
        # Age: 60% in 25-50 (uae cvd focus), 20% young, 20% older
        roll = rng.random()
        if roll < 0.20: age = rng.randint(18, 26)
        elif roll < 0.80: age = rng.randint(25, 51)
        else: age = rng.randint(50, 76)

        # Disease risk from uci's real prevalence by age group
        if age < 30: risk = 0.05
        elif age < 40: risk = 0.36
        elif age < 50: risk = 0.42
        elif age < 60: risk = 0.58
        else: risk = 0.74

        at_risk = rng.random() < risk
        if at_risk:
            fitness = rng.uniform(0.1, 0.45)
            hyp = rng.random() < 0.45
            obesity = rng.random() < 0.40
            sleep_dis = rng.random() < 0.30
        else:
            fitness = rng.uniform(0.5, 0.9)
            hyp = rng.random() < 0.08
            obesity = rng.random() < 0.10
            sleep_dis = rng.random() < 0.08

        p = PatientProfile(
            patient_id=i, age=age, fitness=fitness,
            has_hypertension=hyp, has_obesity=obesity,
            has_sleep_disorder=sleep_dis, hospital_id=i % n_hospitals)
        p.compute_baselines(uci_dist, mitbih_dist)
        patients.append(p)

    return patients


# ============================================================
# step 5: Vital Sign Simulation
# ============================================================

def make_activity_schedule(rng):
    """Create a realistic 48h activity cycle: sleep/rest/active/exercise."""
    n = (duration_hours * 3600) // sample_interval
    schedule = np.full(n, rest, dtype=np.int32)
    sph = 3600 // sample_interval

    for day in range(duration_hours // 24 + 1):
        off = day * 24 * sph
        sleep_start = int((22.5 + rng.uniform(-0.5, 0.5)) * sph)
        wake_time = int((6.5 + rng.uniform(-0.5, 0.5)) * sph)

        if day == 0:
            schedule[off:min(off + wake_time, n)] = sleep
        sl, wk = off + sleep_start, off + 24 * sph + wake_time
        if sl < n:
            schedule[sl:min(wk, n)] = sleep

        for start_h, end_h in [(8, 12), (13, 17)]:
            s = off + int((start_h + rng.uniform(-0.3, 0.3)) * sph)
            e = off + int((end_h + rng.uniform(-0.3, 0.3)) * sph)
            schedule[max(0, s):min(e, n)] = active

        if rng.random() < 0.60:
            ex_h = rng.choice([7.0, 17.0, 18.0])
            s = off + int(ex_h * sph)
            e = off + int((ex_h + rng.uniform(0.5, 1.0)) * sph)
            schedule[max(0, s):min(e, n)] = exercise

    return schedule[:n]


def simulate_vitals(patient, uci_dist):
    """Generate 48h of continuous vital signs for one patient."""
    rng = np.random.RandomState(patient.patient_id * 7 + 13)
    n = (duration_hours * 3600) // sample_interval
    p = patient
    activity = make_activity_schedule(rng)

    # Target vitals per activity state
    hr_tgt = {
        sleep: p.resting_hr - 7, rest: p.resting_hr,
        active: p.resting_hr + 22 + p.fitness * 10,
        exercise: p.resting_hr + 50 + p.fitness * 20}
    hrv_tgt = {
        sleep: p.resting_hrv + 8, rest: p.resting_hrv,
        active: p.resting_hrv - 12, exercise: p.resting_hrv - 22}
    spo2_tgt = {
        sleep: 96.5 - (0.5 if p.has_obesity else 0), rest: 97.5,
        active: 96.5, exercise: 95.0 - (1.5 if p.fitness < 0.3 else 0)}
    bps_tgt = {
        sleep: p.baseline_bp_s - 8, rest: p.baseline_bp_s,
        active: p.baseline_bp_s + 10, exercise: p.baseline_bp_s + 22}
    bpd_tgt = {
        sleep: p.baseline_bp_d - 5, rest: p.baseline_bp_d,
        active: p.baseline_bp_d + 5, exercise: p.baseline_bp_d + 8}

    # Smoothing: alpha=0.02 means ~4 min to reach a new target
    alpha = 0.02
    noise = {'hr': 1.8, 'hrv': 1.2, 'spo2': 0.25, 'bps': 1.2, 'bpd': 0.8}

    hr, hrv, spo2 = np.zeros(n), np.zeros(n), np.zeros(n)
    bp_s, bp_d = np.zeros(n), np.zeros(n)
    hr[0] = hr_tgt[activity[0]]
    hrv[0] = hrv_tgt[activity[0]]
    spo2[0] = spo2_tgt[activity[0]]
    bp_s[0] = bps_tgt[activity[0]]
    bp_d[0] = bpd_tgt[activity[0]]

    # Stress events: random cardiovascular episodes
    risk = (1 - p.fitness) * 0.35 + (p.age / 80) * 0.3
    if p.has_hypertension: risk += 0.15
    if p.has_obesity: risk += 0.10
    if p.has_sleep_disorder: risk += 0.05
    n_events = rng.poisson(lam=2 + risk * 7)

    event_mask = np.zeros(n)
    for _ in range(n_events):
        start = rng.randint(0, n)
        dur = rng.randint(60, 420)
        sev = rng.uniform(0.25, 1.0)
        end = min(start + dur, n)
        ramp = np.ones(end - start)
        rl = min(30, len(ramp) // 3)
        if rl > 0:
            ramp[:rl] = np.linspace(0, 1, rl)
            ramp[-rl:] = np.linspace(1, 0, rl)
        event_mask[start:end] = np.maximum(event_mask[start:end], sev * ramp)

    # Main simulation: smooth transitions + noise + stress perturbations
    for t in range(1, n):
        act, ev = activity[t], event_mask[t]
        hr[t] = hr[t-1] + alpha * (hr_tgt[act] + ev*40 - hr[t-1]) + rng.normal(0, noise['hr'])
        hrv[t] = hrv[t-1] + alpha * (max(8, hrv_tgt[act] - ev*22) - hrv[t-1]) + rng.normal(0, noise['hrv'])
        spo2[t] = spo2[t-1] + alpha * (spo2_tgt[act] - ev*4.5 - spo2[t-1]) + rng.normal(0, noise['spo2'])
        bp_s[t] = bp_s[t-1] + alpha * (bps_tgt[act] + ev*28 - bp_s[t-1]) + rng.normal(0, noise['bps'])
        bp_d[t] = bp_d[t-1] + alpha * (bpd_tgt[act] + ev*14 - bp_d[t-1]) + rng.normal(0, noise['bpd'])

    # Clip to valid physiological ranges
    hr = np.clip(np.round(hr, 1), 38, 210)
    hrv = np.clip(np.round(hrv, 1), 5, 120)
    spo2 = np.clip(np.round(spo2, 1), 82, 100)
    bp_s = np.clip(np.round(bp_s, 1), 80, 210)
    bp_d = np.clip(np.round(bp_d, 1), 45, 135)
    bp_d = np.minimum(bp_d, bp_s - 20)

    # Sleep hours (slight daily variation around patient average)
    sleep_arr = np.full(n, p.avg_sleep)
    day_len = 24 * 3600 // sample_interval
    for d in range(duration_hours // 24 + 1):
        s, e = d * day_len, min((d+1) * day_len, n)
        sleep_arr[s:e] = p.avg_sleep + rng.uniform(-0.5, 0.5)
    sleep_arr = np.round(np.clip(sleep_arr, 3.0, 10.0), 1)

    # Map to paper schema: 0=Sleep/Rest, 1=Active, 2=Exercise
    activity_out = np.where(activity <= 1, 0, np.where(activity == 2, 1, 2))
    timestamps = pd.date_range('2025-01-01', periods=n, freq=f'{sample_interval}s')

    return pd.DataFrame({
        'patient_id': p.patient_id, 'hospital_id': p.hospital_id,
        'timestamp': timestamps, 'age': p.age,
        'hr': hr, 'hrv': hrv, 'spo2': spo2,
        'bp_s': bp_s, 'bp_d': bp_d,
        'activity': activity_out, 'sleep': sleep_arr})


# ============================================================
# step 6: Stress Score Computation
# Multi-feature, activity-aware, with interaction terms
# ============================================================

def compute_stress_score(df):
    """Compute cardiovascular stress score (0.0 to 1.0) for every row."""
    hr, hrv, spo2 = df['hr'].values, df['hrv'].values, df['spo2'].values
    bp_s, bp_d = df['bp_s'].values, df['bp_d'].values
    age, activity, sleep = df['age'].values, df['activity'].values, df['sleep'].values

    # c1: hr deviation (is hr higher than expected for this activity?)
    exp_hr = np.where(activity == 0, 68 + (age-30)*0.2,
             np.where(activity == 1, 88 + (age-30)*0.15, 128 + (age-30)*0.1))
    hr_std = np.where(activity == 0, 12, np.where(activity == 1, 15, 22))
    c1 = 1 / (1 + np.exp(-1.5 * ((hr - exp_hr)/hr_std - 0.5)))

    # c2: hrv deficit (is hrv lower than expected?)
    exp_hrv = np.maximum(12, np.where(activity == 0, 50-(age-30)*0.3,
              np.where(activity == 1, 38-(age-30)*0.2, 24-(age-30)*0.12)))
    c2 = np.clip(1 / (1 + np.exp(-3 * ((exp_hrv - hrv)/exp_hrv - 0.1))), 0, 1)

    # c3: SpO2 risk (below 94% is clinically concerning)
    c3 = 1 / (1 + np.exp(2.5 * (spo2 - 94)))

    # c4: Blood pressure risk (uci healthy mean ~130 systolic)
    c4 = np.clip((np.maximum(0, (bp_s-132)/20) + np.maximum(0, (bp_d-85)/15)) / 2, 0, 1)

    # c5: Sleep deficit (<6h = chronic cardiovascular risk factor)
    c5 = np.clip((6.5 - sleep) / 3.5, 0, 1)

    # c6: Age risk (fitted to uci disease prevalence curve)
    c6 = 1 / (1 + np.exp(-0.08 * (age - 48))) * 0.5

    # Weighted sum (no feature exceeds 25% influence)
    raw = 0.22*c1 + 0.22*c2 + 0.18*c3 + 0.16*c4 + 0.12*c5 + 0.10*c6

    # Interaction terms: combined risk factors amplify the score
    interact = c1*c2*0.08 + c3*c1*0.06 + c4*c2*0.04

    # Final sigmoid to produce 0-1 output
    stress = 1 / (1 + np.exp(-8 * (raw + interact - 0.38)))
    noise = np.random.RandomState(42).normal(0, 0.008, len(df))
    return np.round(np.clip(stress + noise, 0.0, 1.0), 4)


# ============================================================
# step 7: Validation
# ============================================================

def validate(df, uci_dist, mitbih_dist):
    """Compare generated data against real clinical distributions."""
    rest = df[df['activity'] == 0]
    ex = df[df['activity'] == 2]

    print(f"\n{'='*60}")
    print("validation: Simulated vs Real Clinical Data")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Simulated':<22} {'Real Source'}")
    print(f"  {'-'*58}")
    print(f"  {'Resting bp (sys)':<25} {rest.bp_s.mean():.1f} ± {rest.bp_s.std():.1f} mmHg    uci: {uci_dist['bp_healthy_mean']:.0f} ± {uci_dist['bp_healthy_std']:.0f}")
    print(f"  {'Resting hr':<25} {rest.hr.mean():.1f} ± {rest.hr.std():.1f} bpm      mit-bih: {mitbih_dist['hr_mean']:.0f} ± {mitbih_dist['hr_std']:.0f}")
    print(f"  {'Resting hrv (rmssd)':<25} {rest.hrv.mean():.1f} ± {rest.hrv.std():.1f} ms      mit-bih: {mitbih_dist['hrv_mean']:.0f} ± {mitbih_dist['hrv_std']:.0f}")
    print(f"  {'Exercise hr':<25} {ex.hr.mean():.1f} ± {ex.hr.std():.1f} bpm     uci max: {uci_dist['max_hr_healthy']:.0f}")
    src = "extracted from ecg" if mitbih_dist['source'] == 'extracted' else "published stats"
    print(f"\n  mit-bih source: {src}")

    # Stress distribution
    print(f"\n  Stress Score Distribution:")
    for label, lo, hi in [("Low (<35%)", 0, 0.35), ("Moderate (35-70%)", 0.35, 0.70),
                           ("High (70-90%)", 0.70, 0.90), ("Emergency (>90%)", 0.90, 1.01)]:
        pct = ((df.stress_score >= lo) & (df.stress_score < hi)).mean()
        print(f"    {label:<20}: {pct:.1%}")

    # Single-feature auroc check
    print(f"\n  Single-Feature auroc:")
    y_bin = (df['stress_score'] > 0.5).astype(int)
    for col in ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d', 'age', 'activity', 'sleep']:
        auc = max(roc_auc_score(y_bin, df[col]), 1 - roc_auc_score(y_bin, df[col]))
        print(f"    {col:<12}: {auc:.4f}")

    # fl partition summary
    print(f"\n  fl Partitions:")
    for h in sorted(df['hospital_id'].unique()):
        hdf = df[df['hospital_id'] == h]
        print(f"    Hospital {h}: {hdf.patient_id.nunique()} patients, "
              f"{len(hdf):,} samples, stress={hdf.stress_score.mean():.3f}")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("S.H.I.E.L.D Dataset Calibrator")
    print("=" * 60)

    # Load real clinical data
    print("\n[1/6] Loading UCI Heart Disease Dataset...")
    uci_dist = load_uci_data()

    print("\n[2/6] Loading MIT-BIH Arrhythmia Database...")
    mitbih_dist = load_mitbih_data()

    # Generate patients
    print(f"\n[3/6] Generating {n_patients} patient profiles...")
    patients = generate_cohort(uci_dist, mitbih_dist)
    ages = [p.age for p in patients]
    print(f"  Ages: {min(ages)}-{max(ages)} (mean {np.mean(ages):.0f})")
    print(f"  Hypertensive: {sum(1 for p in patients if p.has_hypertension)}/{n_patients}")

    # Simulate vital signs
    print(f"\n[4/6] Simulating {duration_hours}h vital streams...")
    all_dfs = []
    for i, p in enumerate(patients):
        if (i+1) % 10 == 0 or i == 0:
            print(f"  Patient {i+1}/{n_patients} (age={p.age}, fitness={p.fitness:.2f})")
        all_dfs.append(simulate_vitals(p, uci_dist))
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Compute stress scores
    print("\n[5/6] Computing stress scores...")
    full_df['stress_score'] = compute_stress_score(full_df)

    # Validate
    print("\n[6/6] Validating...")
    validate(full_df, uci_dist, mitbih_dist)

    # Save
    full_df.to_csv(output_file, index=False)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Saved: {output_file} ({full_df.shape[0]:,} rows, {size_mb:.1f} mb)")
    print(f"{'='*60}")