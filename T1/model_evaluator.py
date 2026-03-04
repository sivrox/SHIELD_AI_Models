import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score, brier_score_loss,
    log_loss, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PLOT STYLE — clinical dark theme
# =============================================================================
plt.rcParams.update({
    'figure.facecolor':  '#0D1117',
    'axes.facecolor':    '#161B22',
    'axes.edgecolor':    '#30363D',
    'axes.labelcolor':   '#C9D1D9',
    'axes.titlecolor':   '#E6EDF3',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.color':       '#8B949E',
    'ytick.color':       '#8B949E',
    'text.color':        '#C9D1D9',
    'grid.color':        '#21262D',
    'grid.linewidth':    0.8,
    'legend.facecolor':  '#161B22',
    'legend.edgecolor':  '#30363D',
    'font.family':       'monospace',
})

ACCENT   = '#58A6FF'   # blue
RISK     = '#F85149'   # red
SAFE     = '#3FB950'   # green
WARN     = '#D29922'   # amber
PURPLE   = '#BC8CFF'
TEAL     = '#39D3BB'
FG       = '#E6EDF3'
GRID     = '#21262D'

FEATURES = ['HR', 'BP_sys', 'HRV', 'BP_dia', 'SpO2', 'Activity', 'Age', 'Sleep']
WINDOW   = 60

# =============================================================================
# STEP 1 — TIME-SERIES DATA SIMULATOR
# Generates physiologically plausible vital-sign sequences with known labels.
# =============================================================================
def simulate_vitals(n_samples: int = 3000, window: int = WINDOW, seed: int = 42):
    """
    Produce realistic 60-timestep vital-sign windows with binary stress labels.

    Healthy class  (label 0): normal resting ranges, low variance
    Stress  class  (label 1): elevated HR/BP, suppressed HRV/SpO2, high variance
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    sequences, labels = [], []

    def smooth(sig, w=5):
        """Simple moving-average to simulate physiological autocorrelation."""
        kernel = np.ones(w) / w
        return np.convolve(sig, kernel, mode='same')

    for label in [0, 1]:
        count = half if label == 0 else (n_samples - half)
        for _ in range(count):
            t = np.linspace(0, 1, window)

            if label == 0:   # ── HEALTHY ──────────────────────────────────────
                hr      = smooth(rng.normal(68,  5,  window) + 4 * np.sin(2*np.pi*t))
                bp_s    = smooth(rng.normal(115, 6,  window))
                hrv     = smooth(rng.normal(55,  8,  window))
                bp_d    = smooth(rng.normal(75,  5,  window))
                spo2    = smooth(np.clip(rng.normal(98, 0.4, window), 94, 100))
                activity= rng.choice([0, 1], window, p=[0.7, 0.3]).astype(float)
                age     = np.full(window, rng.integers(20, 45))
                sleep   = smooth(rng.normal(7.2, 0.5, window))

            else:            # ── STRESS / RISK ─────────────────────────────────
                # Sudden HR spike mid-window
                spike = np.zeros(window)
                onset = rng.integers(10, 40)
                spike[onset:] = rng.uniform(15, 30)

                hr      = smooth(rng.normal(90,  10, window) + spike + 8*np.sin(4*np.pi*t))
                bp_s    = smooth(rng.normal(140, 12, window) + spike * 0.6)
                hrv     = smooth(rng.normal(28,  10, window) - spike * 0.3)
                bp_d    = smooth(rng.normal(92,  10, window) + spike * 0.4)
                spo2    = smooth(np.clip(rng.normal(95, 1.2, window) - spike*0.05, 88, 100))
                activity= rng.choice([1, 2], window, p=[0.4, 0.6]).astype(float)
                age     = np.full(window, rng.integers(45, 80))
                sleep   = smooth(rng.normal(5.0, 1.0, window))

            seq = np.stack([hr, bp_s, hrv, bp_d, spo2, activity, age, sleep], axis=-1)
            sequences.append(seq)
            labels.append(label)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.float32)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# =============================================================================
# STEP 2 — NORMALISE
# =============================================================================
def normalise(X_tr, X_te):
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    sig = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_tr - mu) / sig, (X_te - mu) / sig, mu, sig


# =============================================================================
# STEP 3 — LOAD OR BUILD MODEL
# =============================================================================
def load_or_build_model(X_train, y_train, X_val, y_val):
    model_path = 'shield_v3.h5'

    if os.path.exists(model_path):
        print(f"✅ Loaded existing model from '{model_path}'")
        model = tf.keras.models.load_model(model_path)
        history = None
    else:
        print("⚙️  No saved model found — training a fresh LSTM for demo...")
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(WINDOW, len(FEATURES))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1,  activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        ]
        history = model.fit(
            X_train, y_train, epochs=40, batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks, verbose=1
        )
        model.save(model_path)
        print(f"💾 Model saved to '{model_path}'")

    return model, history


# =============================================================================
# STEP 4 — COMPUTE ALL METRICS
# =============================================================================
def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity  = tp / (tp + fn + 1e-8)          # Recall / True Positive Rate
    specificity  = tn / (tn + fp + 1e-8)           # True Negative Rate
    ppv          = tp / (tp + fp + 1e-8)           # Precision
    npv          = tn / (tn + fn + 1e-8)           # Negative Predictive Value
    f1           = 2*ppv*sensitivity / (ppv + sensitivity + 1e-8)
    f2           = 5*ppv*sensitivity / (4*ppv + sensitivity + 1e-8)  # recall-weighted
    mcc          = matthews_corrcoef(y_true, y_pred)
    kappa        = cohen_kappa_score(y_true, y_pred)
    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    brier        = brier_score_loss(y_true, y_probs)
    logloss      = log_loss(y_true, y_probs)

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
    roc_auc  = auc(fpr_arr, tpr_arr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs)
    pr_auc   = average_precision_score(y_true, y_probs)

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'sensitivity': sensitivity,  'specificity': specificity,
        'ppv': ppv,                  'npv': npv,
        'f1': f1,                    'f2': f2,
        'mcc': mcc,                  'kappa': kappa,
        'bal_acc': bal_acc,          'brier': brier,
        'logloss': logloss,          'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr_arr, 'tpr': tpr_arr,
        'prec': prec_arr, 'rec': rec_arr,
        'y_pred': y_pred,
    }


# =============================================================================
# STEP 5 — FIND OPTIMAL THRESHOLD (max F1)
# =============================================================================
def optimal_threshold(y_true, y_probs):
    prec, rec, thresh = precision_recall_curve(y_true, y_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s[:-1])
    return float(thresh[best_idx]), float(f1s[best_idx])


# =============================================================================
# STEP 6 — VISUALISE SIMULATED SEQUENCES
# =============================================================================
def plot_sample_sequences(X, y, n=3):
    fig, axes = plt.subplots(n, len(FEATURES), figsize=(22, 3.5*n))
    fig.suptitle('Simulated Vital-Sign Sequences  ·  Healthy vs Stress',
                 fontsize=15, color=FG, y=1.01)

    stress_idx  = np.where(y == 1)[0][:n]
    healthy_idx = np.where(y == 0)[0][:n]
    pairs = list(zip(healthy_idx, stress_idx))

    for row, (h_i, s_i) in enumerate(pairs):
        for col, feat in enumerate(FEATURES):
            ax = axes[row, col]
            ax.plot(X[h_i, :, col], color=SAFE,   lw=1.4, label='Healthy', alpha=0.85)
            ax.plot(X[s_i, :, col], color=RISK,   lw=1.4, label='Stress',  alpha=0.85)
            ax.set_title(feat if row == 0 else '', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f'Pair {row+1}', fontsize=9)
            if row == 0 and col == len(FEATURES)-1:
                ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig('plot_1_sample_sequences.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_1_sample_sequences.png")


# =============================================================================
# STEP 7 — TRAINING HISTORY (if available)
# =============================================================================
def plot_training_history(history):
    if history is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History', fontsize=15, color=FG)

    epochs = range(1, len(history.history['loss']) + 1)

    ax1.plot(epochs, history.history['loss'],     color=RISK,   lw=2,  label='Train Loss')
    ax1.plot(epochs, history.history['val_loss'], color=ACCENT, lw=2,  label='Val Loss',  linestyle='--')
    ax1.set_title('Loss  (Binary Cross-Entropy)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history['accuracy'],     color=SAFE,   lw=2, label='Train Acc')
    ax2.plot(epochs, history.history['val_accuracy'], color=PURPLE, lw=2, label='Val Acc', linestyle='--')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_2_training_history.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_2_training_history.png")


# =============================================================================
# STEP 8 — PROBABILITY DISTRIBUTION
# =============================================================================
def plot_probability_distribution(y_true, y_probs, threshold):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('Prediction Probability Distribution', fontsize=15, color=FG)

    ax.hist(y_probs[y_true == 0], bins=60, color=SAFE,  alpha=0.6, label='Healthy (true 0)', density=True)
    ax.hist(y_probs[y_true == 1], bins=60, color=RISK,  alpha=0.6, label='Stress  (true 1)', density=True)
    ax.axvline(threshold, color=WARN, lw=2, linestyle='--', label=f'Threshold = {threshold:.2f}')
    ax.axvline(0.5,       color=FG,   lw=1, linestyle=':',  label='Default 0.5', alpha=0.5)

    ax.set_xlabel('Predicted Probability of Stress')
    ax.set_ylabel('Density')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Annotation boxes
    overlap = np.sum((y_probs[y_true==0] > 0.4) & (y_probs[y_true==0] < 0.6))
    ax.text(0.5, ax.get_ylim()[1]*0.85,
            f'Ambiguous zone\n({overlap} healthy samples\nnear boundary)',
            ha='center', fontsize=9, color=WARN,
            bbox=dict(boxstyle='round', fc='#161B22', ec=WARN, alpha=0.8))

    plt.tight_layout()
    plt.savefig('plot_3_probability_distribution.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_3_probability_distribution.png")


# =============================================================================
# STEP 9 — CONFUSION MATRIX (annotated)
# =============================================================================
def plot_confusion_matrix(m):
    cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
    total = cm.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrix Analysis', fontsize=15, color=FG)

    # Raw counts
    cmap = LinearSegmentedColormap.from_list('risk', ['#161B22', RISK])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[0],
                xticklabels=['Pred Healthy', 'Pred Stress'],
                yticklabels=['True Healthy', 'True Stress'],
                linewidths=1, linecolor='#0D1117', cbar=False,
                annot_kws={'size': 18, 'weight': 'bold', 'color': FG})
    axes[0].set_title('Raw Counts')

    # Percentage normalised
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap=cmap, ax=axes[1],
                xticklabels=['Pred Healthy', 'Pred Stress'],
                yticklabels=['True Healthy', 'True Stress'],
                linewidths=1, linecolor='#0D1117', cbar=False,
                annot_kws={'size': 18, 'weight': 'bold', 'color': FG})
    axes[1].set_title('Row-Normalised  (%)')

    # Clinical annotation
    labels_raw = [
        ['True Negatives\n(Correctly safe)', 'False Positives\n(Unnecessary alert)'],
        ['False Negatives\n(Missed risk!)',   'True Positives\n(Caught risk)'],
    ]
    for i in range(2):
        for j in range(2):
            axes[0].text(j+0.5, i+0.75, labels_raw[i][j],
                         ha='center', va='center', fontsize=7.5,
                         color='#8B949E', style='italic')

    plt.tight_layout()
    plt.savefig('plot_4_confusion_matrix.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_4_confusion_matrix.png")


# =============================================================================
# STEP 10 — ROC + PR CURVES
# =============================================================================
def plot_roc_and_pr(m):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Discrimination & Precision-Recall Analysis', fontsize=15, color=FG)

    # ROC
    ax1.plot(m['fpr'], m['tpr'], color=ACCENT, lw=2.5,
             label=f'LSTM  AUC = {m["roc_auc"]:.3f}')
    ax1.fill_between(m['fpr'], m['tpr'], alpha=0.12, color=ACCENT)
    ax1.plot([0,1],[0,1], '--', color='#555', lw=1.5, label='Random (AUC = 0.50)')
    ax1.scatter([1 - m['specificity']], [m['sensitivity']], color=WARN,
                s=120, zorder=5, label=f'Operating point\n(thresh={0.5:.2f})')
    ax1.set_xlabel('False Positive Rate  (1 – Specificity)')
    ax1.set_ylabel('True Positive Rate  (Sensitivity)')
    ax1.set_title('ROC Curve')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Shade clinical zones
    ax1.axvspan(0, 0.1, alpha=0.05, color=SAFE, label='Low FPR zone')

    # PR Curve
    ax2.plot(m['rec'], m['prec'], color=PURPLE, lw=2.5,
             label=f'PR AUC = {m["pr_auc"]:.3f}')
    ax2.fill_between(m['rec'], m['prec'], alpha=0.12, color=PURPLE)
    baseline = (m['TP'] + m['FN']) / (m['TP'] + m['TN'] + m['FP'] + m['FN'])
    ax2.axhline(baseline, linestyle='--', color='#555', lw=1.5,
                label=f'Random baseline ({baseline:.2f})')
    ax2.scatter([m['sensitivity']], [m['ppv']], color=WARN, s=120, zorder=5,
                label='Operating point')
    ax2.set_xlabel('Recall  (Sensitivity)')
    ax2.set_ylabel('Precision  (PPV)')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_5_roc_pr_curves.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_5_roc_pr_curves.png")


# =============================================================================
# STEP 11 — THRESHOLD SWEEP
# =============================================================================
def plot_threshold_sweep(y_true, y_probs):
    thresholds = np.linspace(0.01, 0.99, 200)
    sens_list, spec_list, ppv_list, f1_list, f2_list = [], [], [], [], []

    for t in thresholds:
        m = compute_metrics(y_true, y_probs, threshold=t)
        sens_list.append(m['sensitivity'])
        spec_list.append(m['specificity'])
        ppv_list.append(m['ppv'])
        f1_list.append(m['f1'])
        f2_list.append(m['f2'])

    best_t, best_f1 = optimal_threshold(y_true, y_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Threshold Sensitivity Analysis', fontsize=15, color=FG)

    ax1.plot(thresholds, sens_list, color=SAFE,   lw=2, label='Sensitivity (Recall)')
    ax1.plot(thresholds, spec_list, color=ACCENT, lw=2, label='Specificity')
    ax1.plot(thresholds, ppv_list,  color=PURPLE, lw=2, label='Precision (PPV)')
    ax1.axvline(best_t, color=WARN, lw=2, linestyle='--',
                label=f'Optimal  = {best_t:.2f}')
    ax1.axvline(0.5, color=FG, lw=1, linestyle=':', alpha=0.5, label='Default 0.5')
    ax1.set_xlabel('Decision Threshold'); ax1.set_ylabel('Score')
    ax1.set_title('Sensitivity / Specificity / Precision')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(thresholds, f1_list, color=TEAL,   lw=2, label='F1  (balanced)')
    ax2.plot(thresholds, f2_list, color=RISK,   lw=2, label='F2  (recall-weighted)')
    ax2.axvline(best_t, color=WARN, lw=2, linestyle='--',
                label=f'Best F1 @ {best_t:.2f}  =  {best_f1:.3f}')
    ax2.set_xlabel('Decision Threshold'); ax2.set_ylabel('F-Score')
    ax2.set_title('F1 & F2 vs Threshold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_6_threshold_sweep.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_6_threshold_sweep.png")


# =============================================================================
# STEP 12 — CALIBRATION CURVE
# =============================================================================
def plot_calibration(y_true, y_probs):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle('Probability Calibration', fontsize=15, color=FG)

    frac_pos, mean_pred = calibration_curve(y_true, y_probs, n_bins=12)
    brier = brier_score_loss(y_true, y_probs)

    ax.plot([0,1],[0,1], '--', color='#555', lw=1.5, label='Perfectly calibrated')
    ax.plot(mean_pred, frac_pos, 'o-', color=ACCENT, lw=2.5, ms=7,
            label=f'LSTM  (Brier = {brier:.3f})')
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.15, color=RISK, label='Calibration gap')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    plt.tight_layout()
    plt.savefig('plot_7_calibration.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_7_calibration.png")


# =============================================================================
# STEP 13 — FEATURE IMPORTANCE (mean absolute gradient)
# =============================================================================
def plot_feature_importance(model, X_test):
    print("  Computing gradient-based feature importance...")
    X_tensor = tf.constant(X_test[:200], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor)

    grads = tape.gradient(preds, X_tensor)          # (N, timesteps, features)
    importance = np.abs(grads.numpy()).mean(axis=(0, 1))   # (features,)
    importance /= importance.sum()

    order = np.argsort(importance)
    colors = [SAFE if importance[i] < np.median(importance) else RISK for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Feature Importance  (Mean |Gradient| w.r.t. Output)', fontsize=15, color=FG)

    bars = ax.barh([FEATURES[i] for i in order], importance[order],
                   color=colors, edgecolor='#0D1117', height=0.6)
    for bar, val in zip(bars, importance[order]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, color=FG)

    ax.set_xlabel('Relative Importance')
    ax.grid(True, axis='x', alpha=0.3)

    red_patch  = mpatches.Patch(color=RISK, label='High importance')
    blue_patch = mpatches.Patch(color=SAFE, label='Lower importance')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig('plot_8_feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_8_feature_importance.png")


# =============================================================================
# STEP 14 — TEMPORAL ATTENTION: per-timestep mean gradient
# =============================================================================
def plot_temporal_attention(model, X_test, y_test):
    print("  Computing temporal attention maps...")
    for label, color, name in [(0, SAFE, 'Healthy'), (1, RISK, 'Stress')]:
        idx = np.where(y_test == label)[0][:50]
        X_sub = tf.constant(X_test[idx], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_sub)
            preds = model(X_sub)

        grads = tape.gradient(preds, X_sub).numpy()
        attention = np.abs(grads).mean(axis=(0, 2))  # (timesteps,)
        attention /= attention.max()

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3,1]})
        fig.suptitle(f'Temporal Attention Map  —  {name} Class', fontsize=15, color=FG)

        # Feature lines
        mean_seq = X_test[idx].mean(axis=0)   # (timesteps, features)
        for fi, feat in enumerate(FEATURES):
            normed = (mean_seq[:, fi] - mean_seq[:, fi].min())
            denom = normed.max() + 1e-8
            normed /= denom
            axes[0].plot(normed, lw=1.2, alpha=0.65, label=feat)

        axes[0].set_ylabel('Normalised Value')
        axes[0].set_title('Mean Feature Trajectories')
        axes[0].legend(fontsize=8, ncol=4, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Attention heatmap
        axes[1].fill_between(range(WINDOW), attention, color=color, alpha=0.6)
        axes[1].plot(attention, color=color, lw=1.5)
        axes[1].set_xlabel('Timestep (seconds)')
        axes[1].set_ylabel('Attention')
        axes[1].set_title('Model Attention (where in the window it focuses)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'plot_9_temporal_attention_{name.lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#0D1117')
        plt.close()
        print(f"  ✅ {fname}")


# =============================================================================
# STEP 15 — SUMMARY DASHBOARD
# =============================================================================
def plot_summary_dashboard(m, best_t, best_f1):
    fig = plt.figure(figsize=(16, 9), facecolor='#0D1117')
    fig.suptitle('SHIELD LSTM  ·  Clinical Performance Dashboard',
                 fontsize=18, color=FG, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

    # ── KPI tiles ─────────────────────────────────────────────────────────────
    kpis = [
        ('ROC AUC',       f'{m["roc_auc"]:.3f}',    ACCENT),
        ('Sensitivity',   f'{m["sensitivity"]:.3f}', SAFE),
        ('Specificity',   f'{m["specificity"]:.3f}', TEAL),
        ('Precision',     f'{m["ppv"]:.3f}',         PURPLE),
        ('F1 Score',      f'{m["f1"]:.3f}',          WARN),
        ('MCC',           f'{m["mcc"]:.3f}',         ACCENT),
        ('Brier Score',   f'{m["brier"]:.3f}',       RISK),
        ('Cohen κ',       f'{m["kappa"]:.3f}',       SAFE),
    ]
    tile_positions = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
    for (kpi_name, kpi_val, kpi_color), (row, col) in zip(kpis, tile_positions):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#161B22')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('off')
        ax.text(0.5, 0.65, kpi_val, ha='center', va='center',
                fontsize=22, fontweight='bold', color=kpi_color)
        ax.text(0.5, 0.25, kpi_name, ha='center', va='center',
                fontsize=10, color='#8B949E')
        for spine in ['top','bottom','left','right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(kpi_color)
            ax.spines[spine].set_linewidth(1.5)

    # ── ROC mini ──────────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[2, 0:2])
    ax_roc.plot(m['fpr'], m['tpr'], color=ACCENT, lw=2,
                label=f'AUC = {m["roc_auc"]:.3f}')
    ax_roc.fill_between(m['fpr'], m['tpr'], alpha=0.12, color=ACCENT)
    ax_roc.plot([0,1],[0,1],'--', color='#555', lw=1)
    ax_roc.set_title('ROC Curve'); ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.legend(fontsize=9); ax_roc.grid(True, alpha=0.3)

    # ── PR mini ───────────────────────────────────────────────────────────────
    ax_pr = fig.add_subplot(gs[2, 2:4])
    ax_pr.plot(m['rec'], m['prec'], color=PURPLE, lw=2,
               label=f'PR AUC = {m["pr_auc"]:.3f}')
    ax_pr.fill_between(m['rec'], m['prec'], alpha=0.12, color=PURPLE)
    ax_pr.set_title('Precision-Recall'); ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
    ax_pr.legend(fontsize=9); ax_pr.grid(True, alpha=0.3)

    plt.savefig('plot_10_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print("  ✅ plot_10_dashboard.png")


# =============================================================================
# STEP 16 — PRINT FULL TEXT REPORT
# =============================================================================
def print_report(m, best_t, best_f1, n_test):
    sep = '─' * 52
    print(f'\n{"═"*52}')
    print(f'  SHIELD LSTM  ·  FULL EVALUATION REPORT')
    print(f'{"═"*52}')
    print(f'  Test samples  : {n_test}')
    print(f'  Threshold used: 0.50  (default)')
    print(f'  Optimal thresh: {best_t:.3f}  (max F1 = {best_f1:.3f})')
    print(sep)
    print(f'  CONFUSION MATRIX')
    print(f'    TP  {m["TP"]:>6}   FP  {m["FP"]:>6}')
    print(f'    FN  {m["FN"]:>6}   TN  {m["TN"]:>6}')
    print(sep)
    print(f'  DISCRIMINATION')
    print(f'    ROC AUC          : {m["roc_auc"]:.4f}')
    print(f'    PR  AUC          : {m["pr_auc"]:.4f}')
    print(sep)
    print(f'  CLINICAL METRICS')
    print(f'    Sensitivity      : {m["sensitivity"]:.4f}  (True Positive Rate)')
    print(f'    Specificity      : {m["specificity"]:.4f}  (True Negative Rate)')
    print(f'    Precision (PPV)  : {m["ppv"]:.4f}')
    print(f'    Neg. Pred. Value : {m["npv"]:.4f}')
    print(f'    Balanced Accuracy: {m["bal_acc"]:.4f}')
    print(sep)
    print(f'  F-SCORES')
    print(f'    F1               : {m["f1"]:.4f}')
    print(f'    F2 (recall-wt.)  : {m["f2"]:.4f}')
    print(sep)
    print(f'  RELIABILITY')
    print(f'    Matthews MCC     : {m["mcc"]:.4f}   (1=perfect, 0=random)')
    print(f'    Cohen Kappa      : {m["kappa"]:.4f}')
    print(f'    Brier Score      : {m["brier"]:.4f}   (lower=better)')
    print(f'    Log Loss         : {m["logloss"]:.4f}   (lower=better)')
    print(f'{"═"*52}\n')
    print(classification_report(
        (m['TP'] + m['FN']) * [1] + (m['TN'] + m['FP']) * [0],   # dummy — use y directly
        (m['TP']) * [1] + (m['FP']) * [1] + (m['FN']) * [0] + (m['TN']) * [0],
        target_names=['Healthy', 'Stress']
    ))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print('\n' + '='*52)
    print('  SHIELD LSTM — Evaluation Pipeline')
    print('='*52 + '\n')

    # ── 1. Data ───────────────────────────────────────────────────────────────
    if os.path.exists('X_windows.npy') and os.path.exists('y_labels.npy'):
        print("📥 Loading real data from X_windows.npy / y_labels.npy...")
        X = np.load('X_windows.npy')
        y = np.load('y_labels.npy')
    else:
        print("🔬 Simulating physiological time-series data...")
        X, y = simulate_vitals(n_samples=3000, window=WINDOW)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,  random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    X_train, X_val  = normalise(X_train, X_val)[:2]
    X_train_n, X_test_n, _, _ = normalise(X_train, X_test)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 2. Sample sequences ───────────────────────────────────────────────────
    print("\n📈 Generating plots...")
    plot_sample_sequences(X, y)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model, history = load_or_build_model(X_train, y_train, X_val, y_val)
    plot_training_history(history)

    # ── 4. Predictions ────────────────────────────────────────────────────────
    print("\n🔮 Running predictions on test set...")
    y_probs = model.predict(X_test_n, verbose=0).ravel()
    best_t, best_f1 = optimal_threshold(y_test, y_probs)
    m = compute_metrics(y_test, y_probs, threshold=0.5)

    # ── 5. All plots ──────────────────────────────────────────────────────────
    plot_probability_distribution(y_test, y_probs, best_t)
    plot_confusion_matrix(m)
    plot_roc_and_pr(m)
    plot_threshold_sweep(y_test, y_probs)
    plot_calibration(y_test, y_probs)
    plot_feature_importance(model, X_test_n)
    plot_temporal_attention(model, X_test_n, y_test)
    plot_summary_dashboard(m, best_t, best_f1)

    # ── 6. Text report ────────────────────────────────────────────────────────
    print_report(m, best_t, best_f1, len(y_test))

    print("\n✅ All done. Output files:")
    for i in range(1, 11):
        prefix = f'plot_{i}_'
        matches = [f for f in os.listdir('.') if f.startswith(prefix)]
        for f in matches:
            print(f"   {f}")