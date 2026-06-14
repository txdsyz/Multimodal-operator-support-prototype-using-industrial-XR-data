'''
Set DATA_DIR to the folder that holds the prepared data.
Expected structure (produced by merge.py and dinov2_rgb.py):
  DATA_DIR/
    train/merge.csv               #motion features, which is outputted from the Merge_data.py, for all training recordings
    train/dinov2_feats.csv        #DINOv2 RGB features for training, which is outputted from thedinov2_tgb.py
    test/test.csv                 #motion features, test recording, which is outputted from the Merge_data.py, for test recordings
    test/dinov2_feats_test.csv    #DINOv2 RGB features for test, which is outputted from thedinov2_tgb.py
    test/AR_labels.csv            #AR labels of the test recording (for Stall indicator in Part B)
Edit this one line to your own path; everything else is derived from it.
'''

#%%
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data" 

TRAIN_CSV    = os.path.join(DATA_DIR, "train", "merge.csv")
TRAIN_DINOV2 = os.path.join(DATA_DIR, "train", "dinov2_feats.csv")
TEST_CSV     = os.path.join(DATA_DIR, "test", "test.csv")
TEST_DINOV2  = os.path.join(DATA_DIR, "test", "dinov2_feats_test.csv")
AR_PATH      = os.path.join(DATA_DIR, "test", "AR_labels.csv")

W=30
CORR_THRESH=0.9
THUMB_TIP,INDEX_TIP=5,10
HAS_TEST_LABELS=True

PART_NAMES=["front_chassis","front_chassis_pin","rear_chassis","short_rear_chassis",
            "front_rear_chassis_pin","rear_rear_chassis_pin","front_bracket",
            "front_bracket_screw","front_wheel_assy","rear_wheel_assy"]

#Feature engineer (Gaze + Hand + Pose)
def hand_xy(d,side):
    xs=d[[f"{side}_j{j}_x" for j in range(26)]].values.astype(float)
    ys=d[[f"{side}_j{j}_y" for j in range(26)]].values.astype(float); return np.stack([xs,ys],-1)

def hand_features(d, side, present):
    J = hand_xy(d, side)
    J[~present] = np.nan
    c = np.nanmean(J, axis=1)  #the center point of palm
    # The moving speed of palm
    cspeed = np.r_[np.nan, np.linalg.norm(np.diff(c, axis=0), axis=1)]
    # The moving speed of each hand joints
    energy = np.r_[np.nan, np.nanmean(np.linalg.norm(np.diff(J, axis=0), axis=2), axis=1)]
    # How wide the palm is open (The average distance between each joints to palm)
    spread = np.nanmean(np.linalg.norm(J - c[:, None, :], axis=2), axis=1)
    # From the tip of the thumb to the tip of the index finger (the action of taking parts)
    pinch  = np.linalg.norm(J[:, THUMB_TIP] - J[:, INDEX_TIP], axis=1)
    return pd.DataFrame({f"{side}_cx": c[:, 0], f"{side}_cy": c[:, 1], f"{side}_cspeed": cspeed,
                         f"{side}_energy": energy, f"{side}_spread": spread, f"{side}_pinch": pinch},
                        index=d.index)
def features_one(d):
    d = d.sort_values("frame").reset_index(drop=True)

    # The hand modal sensor
    # present hands
    Lp = d["left_present"].astype(bool).values
    Rp = d["right_present"].astype(bool).values
    sl, sr = hand_features(d, "L", Lp), hand_features(d, "R", Rp)
    # Identify the domain hand
    dom = "R" if (np.nanmean(sr["R_energy"]) >= np.nanmean(sl["L_energy"])) else "L"
    non = "L" if dom == "R" else "R"
    f = pd.DataFrame(index=d.index)
    roll = lambda s: s.rolling(W, min_periods=2)
    for tag, side, sc in [("dom", dom, sl if dom == "L" else sr),("non", non, sl if non == "L" else sr)]:
        pres = d[f"{'left' if side == 'L' else 'right'}_present"]
        f[f"{tag}_present_ratio"] = pres.rolling(W, min_periods=1).mean()
        f[f"{tag}_cspeed"] = roll(sc[f"{side}_cspeed"]).mean()
        f[f"{tag}_energy"] = roll(sc[f"{side}_energy"]).mean()
        f[f"{tag}_spread"] = roll(sc[f"{side}_spread"]).mean()
        # The crude position variation
        f[f"{tag}_pos_std"] = roll(sc[f"{side}_cx"]).std() + roll(sc[f"{side}_cy"]).std()
        f[f"{tag}_pinch_min"]  = roll(sc[f"{side}_pinch"]).min()
        f[f"{tag}_pinch_mean"] = roll(sc[f"{side}_pinch"]).mean()
        f[f"{tag}_pinch_std"]  = roll(sc[f"{side}_pinch"]).std()
    both = Lp & Rp
    hd = pd.Series(np.where(both, np.linalg.norm(
        sl[["L_cx", "L_cy"]].values - sr[["R_cx", "R_cy"]].values, axis=1), np.nan), index=d.index)
    f["hands_dist_mean"] = roll(hd).mean()
    f["hands_dist_std"] = roll(hd).std()
    #The proportion of the two hands in the image
    f["both_present_ratio"] = pd.Series(both.astype(float), index=d.index).rolling(W, min_periods=1).mean()

    # The Pose(head) modal sensor
    for c in ["fwd_x", "fwd_y", "fwd_z"]: f[f"{c}_mean"] = roll(d[c]).mean()
    f["fwd_var"] = roll(d["fwd_x"]).std() + roll(d["fwd_y"]).std() + roll(d["fwd_z"]).std()
    hp = d[["pos_x", "pos_y", "pos_z"]].values
    f["head_pos_speed"] = pd.Series(np.r_[np.nan, np.linalg.norm(np.diff(hp, axis=0), axis=1)],
                                    index=d.index).rolling(W, min_periods=2).mean()
    f["head_pos_std"] = roll(d["pos_x"]).std() + roll(d["pos_y"]).std() + roll(d["pos_z"]).std()

    # The gaze modal sensor
    f["gaze_mean_x"] = roll(d["gaze_x"]).mean(); f["gaze_mean_y"] = roll(d["gaze_y"]).mean()
    f["gaze_std_x"] = roll(d["gaze_x"]).std();  f["gaze_std_y"] = roll(d["gaze_y"]).std()
    f["gaze_speed"] = pd.Series(np.r_[np.nan, np.linalg.norm(
        np.diff(d[["gaze_x", "gaze_y"]].values, axis=0), axis=1)],
        index=d.index).rolling(W, min_periods=2).mean()

    f["dominant_is_right"] = int(dom == "R")
    f.insert(0, "recording_id", d["recording_id"].values)
    f.insert(1, "frame", d["frame"].values)
    for c in [c for c in d.columns if c.startswith("done_")]: f[c] = d[c].values
    return f

# HMM
def hmm_decode(prob_lvl, n=11):
    T=prob_lvl.shape[0]; logE=np.log(prob_lvl+1e-9)
    logT=np.full((n,n),-1e9)
    for s in range(n):
        for ns in range(s,min(s+3,n)): logT[s,ns]=np.log({s:0.8,s+1:0.15}.get(ns,0.05)+1e-9)
    dp=np.full((T,n),-1e9); bk=np.zeros((T,n),int); dp[0]=logE[0]
    for t in range(1,T):
        for ns in range(n):
            cand=dp[t-1]+logT[:,ns]; bk[t,ns]=cand.argmax(); dp[t,ns]=cand[bk[t,ns]]+logE[t,ns]
    path=np.zeros(T,int); path[-1]=dp[-1].argmax()
    for t in range(T-2,-1,-1): path[t]=bk[t+1,path[t+1]]
    return path

# HMM: Turn a per-frame progress count into an HMM emission distribution:
# emission probability [0.6, 0.2, 0.2], transition probability[0.8,0.15,0.0.05]
# n = 11 (11prats)
def to_emission(raw_prog, n=11):
    T=len(raw_prog); prob=np.zeros((T,n))
    for i,p in enumerate(raw_prog):
        p=int(p); prob[i,p]=0.6
        if p>0:    prob[i,p-1]+=0.2
        if p<n-1:  prob[i,p+1]+=0.2
    return prob/prob.sum(1,keepdims=True)

# Predict one recording with a given model set + feature set:
# per part(0/1), raw progress(sum), HMM-smoothed progress, and truth (if available).
def predict_with(sub, models, feats, targets):
    pred=np.zeros((len(sub),len(targets)),int)
    for k,t in enumerate(targets):
        m,const=models[t]
        pred[:,k]=const if m is None else (m.predict_proba(sub[feats])[:,1]>=0.5).astype(int)
    raw_prog=pred.sum(1)
    hmm=hmm_decode(to_emission(raw_prog))
    return pred, raw_prog, hmm

#  train features(Gaze + Hand + Pose) + RGB
tr_raw=pd.read_csv(TRAIN_CSV)
tr_feat=pd.concat([features_one(d) for _,d in tr_raw.groupby("recording_id")],ignore_index=True)
motion=[c for c in tr_feat.columns if c not in ["recording_id","frame"] and not c.startswith("done_")]
targets=[c for c in tr_feat.columns if c.startswith("done_") and c!="done_base"]
#feartures from RGB
tr_dino=pd.read_csv(TRAIN_DINOV2)
rgb_cols=[c for c in tr_dino.columns if c.startswith("img_")]
# merge+rgb
TR=tr_feat.merge(tr_dino,on=["recording_id","frame"],how="inner")

#%%

'''
Hyperparameter sensitivity analysis (leave-one-recording-out):
It is not to find optimal hyperparameters, but to check that the model is
insensitive to them, so the LightGBM defaults sit in a stable region.
With only four operators, tuning would overfit; a robustness check is the
honest thing to report instead.
Note: No leaking. correlation dedup and model fitting are redone inside each fold
on the training recordings only; the held-out recording is only evaluated.
'''

#LightGBM
def build_lgb(**overrides):
    params = dict(verbose=-1, random_state=27, n_jobs=1, num_leaves=31,
                  learning_rate=0.05, n_estimators=200, min_child_samples=30)
    params.update(overrides)
    return LGBMClassifier(**params)

def dedup_features(df, feats, thresh=CORR_THRESH):
    corr = df[feats].fillna(0).corr().abs().values
    keep = np.ones(len(feats), bool)
    for i in range(len(feats)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(feats)):
            if keep[j] and corr[i, j] > thresh:
                keep[j] = False
    return [feats[i] for i in range(len(feats)) if keep[i]]

def fit_models(df, feats):
    m = {}
    for t in targets:
        y = df[t].values
        m[t] = (None, y[0]) if y.min() == y.max() else (build_lgb().fit(df[feats], y), None)
    return m

def logo_cv_detailed(hp):
    recs = TR["recording_id"].unique().tolist()
    all_feats = motion + rgb_cols
    maes, macros = [], []
    perpart = {t: [] for t in targets}        # per fold per part F1
    for held in recs:
        tr = TR[TR.recording_id != held]
        va = TR[TR.recording_id == held].sort_values("frame").reset_index(drop=True)
        feats = dedup_features(tr, all_feats)              # fit on training fold only
        fold_models = {}
        for t in targets:
            y = tr[t].values
            fold_models[t] = (None, y[0]) if y.min() == y.max() else (build_lgb(**hp).fit(tr[feats], y), None)
        pred, raw, hmm = predict_with(va, fold_models, feats, targets)
        Y = va[targets].values
        maes.append(np.mean(np.abs(Y.sum(1) - hmm)))
        fold_f1 = [f1_score(Y[:, k], pred[:, k], zero_division=0) for k in range(len(targets))]
        macros.append(np.mean(fold_f1))
        for k, t in enumerate(targets):
            perpart[t].append(fold_f1[k])
    perpart_mean = {t: float(np.mean(perpart[t])) for t in targets}
    return (float(np.mean(maes)), float(np.std(maes)),
            float(np.mean(macros)), float(np.std(macros)), perpart_mean)

# default config = the config of the final model and the test evaluation
default_hp = {"num_leaves": 31, "n_estimators": 200, "min_child_samples": 30}
cv_mae_m, cv_mae_s, cv_macro_m, cv_macro_s, cv_perpart = logo_cv_detailed(default_hp)

# One-dimensional sweep: vary one hyperparameter, keep the others at default
sweep = {
    "num_leaves":        [15, 31, 63],
    "n_estimators":      [150, 200, 300],
    "min_child_samples": [20, 30, 50],
}
sens_rows = []
for param, values in sweep.items():
    for v in values:
        hp = dict(default_hp); hp[param] = v
        mae_m, mae_s, f1_m, f1_s, _ = logo_cv_detailed(hp)
        sens_rows.append({
            "hyperparameter": param,
            "value": v,
            "is_default": (v == default_hp[param]),
            "MAE": f"{mae_m:.2f}+/-{mae_s:.2f}",
            "Macro-F1": f"{f1_m:.2f}+/-{f1_s:.2f}",
        })
sens_df = pd.DataFrame(sens_rows)
disp = sens_df.copy()
disp["value"] = disp.apply(lambda r: f"{r['value']}*" if r["is_default"] else f"{r['value']}", axis=1)
print(disp[["hyperparameter", "value", "MAE", "Macro-F1"]].to_string(index=False))
sens_df.to_csv("sensitivity_results.csv", index=False)
print("\nsaved sensitivity_results.csv")

# Correlation, fit on Train only
FEATS = dedup_features(TR, motion + rgb_cols)
print(f"\nSelected features: {len(FEATS)} "
      f"(motion {len([f for f in FEATS if f in motion])} + RGB {len([f for f in FEATS if f in rgb_cols])})")

# heatmap(only motion, there are too much in RGB and image dimension is meaningless)
# threshold 0.9
corr_m = TR[motion].fillna(0).corr().abs()
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(corr_m.values, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(motion))); ax.set_yticks(range(len(motion)))
ax.set_xticklabels(motion, rotation=90, fontsize=6)
ax.set_yticklabels(motion, fontsize=6)
fig.colorbar(im, label="|Pearson correlation|", fraction=0.046, pad=0.04)
ax.set_title(f"Motion-feature correlation, threshold={CORR_THRESH})", fontweight="bold")
fig.tight_layout(); fig.savefig("featsel_motion_corr.png", dpi=130); plt.close(fig)
print("saved featsel_motion_corr.png")

motion_kept = [f for f in FEATS if f in motion]
print(f"Dedup: motion {len(motion)} -> {len(motion_kept)} kept "
      f"({len(motion)-len(motion_kept)} removed); RGB kept {len([f for f in FEATS if f in rgb_cols])}; "
      f"total selected {len(FEATS)}.")

# save the selected feature table
feat_table = pd.DataFrame({
    "feature": FEATS,
    "modality": ["motion" if f in motion else "RGB(DINOv2)" for f in FEATS],
})
feat_table.to_csv("selected_features.csv", index=False)
print(f"saved selected_features.csv ({len(FEATS)} rows)")

# final model on ALL train
models = fit_models(TR, FEATS)

# Feature importance (interpretability of each part classifiers)
imp = pd.Series(0.0, index=FEATS)
for t in targets:
    m, const = models[t]
    if m is not None:
        imp += pd.Series(m.feature_importances_, index=FEATS)

motion_in_feats = [f for f in FEATS if f in motion] #motion
rgb_in_feats    = [f for f in FEATS if f in rgb_cols] #RGB

# named motion features individually + RGB collapsed into one entry
agg = imp[motion_in_feats].copy()
agg["RGB total (384 dims)"] = imp[rgb_in_feats].sum()
agg = agg.sort_values(ascending=True)

rgb_share = imp[rgb_in_feats].sum() / imp.sum() * 100
print(f"\nRGB contributes {rgb_share:.1f}% of total feature importance "
      f"(summed over the ten classifiers); motion contributes {100 - rgb_share:.1f}%.")
print("Top motion features:")
for f, v in imp[motion_in_feats].sort_values(ascending=False).head(8).items():
    print(f"  {f:<22} {v:.0f}")

# Bar chart
fig, ax = plt.subplots(figsize=(8, max(5, 0.32 * len(agg))))
colors = ["#d62728" if i == "RGB total (384 dims)" else "#1f77b4" for i in agg.index]
ax.barh(range(len(agg)), agg.values, color=colors)
ax.set_yticks(range(len(agg)))
ax.set_yticklabels(agg.index, fontsize=8)
ax.set_xlabel("Summed importance over the ten part classifiers")
ax.set_title("Feature importance: named motion features vs RGB total", fontweight="bold")
fig.tight_layout()
fig.savefig("featimp_motion_vs_rgb.png", dpi=130)
plt.close(fig)
print("saved featimp_motion_vs_rgb.png")



# Test
te_raw=pd.read_csv(TEST_CSV)
te_feat=pd.concat([features_one(d) for _,d in te_raw.groupby("recording_id")],ignore_index=True)
te_dino=pd.read_csv(TEST_DINOV2)
TE=te_feat.merge(te_dino,on=["recording_id","frame"],how="inner")
print(f"\nTEST: {TE.shape[0]} frame")

for c in FEATS:
    if c not in TE.columns: TE[c]=0.0

test_recs=TE.recording_id.unique().tolist()

# pull one recording's rows
def get_sub(rec):
    return TE[TE.recording_id==rec].sort_values("frame").reset_index(drop=True)


print("\n TEST Result：")
results={}
all_mae=[]; all_mae_hmm=[]; all_macro=[]; per_part={t:[] for t in targets}
for rec in test_recs:
    sub=get_sub(rec)
    pred, raw_prog, hmm = predict_with(sub, models, FEATS, targets)
    Y = sub[targets].values if HAS_TEST_LABELS else None
    results[rec]=(sub, pred, raw_prog, hmm, Y)
    if HAS_TEST_LABELS:
        true=Y.sum(1)
        mae=np.mean(np.abs(true-raw_prog)); mae_h=np.mean(np.abs(true-hmm))
        macro=np.mean([f1_score(Y[:,k],pred[:,k],zero_division=0) for k in range(len(targets))])
        for k,t in enumerate(targets): per_part[t].append(f1_score(Y[:,k],pred[:,k],zero_division=0))
        all_mae.append(mae); all_mae_hmm.append(mae_h); all_macro.append(macro)
        print(f"{rec}: MAE raw={mae:.2f} HMM={mae_h:.2f} | Macro-F1={macro:.2f}")
    else:
        out=pd.DataFrame({"recording_id":rec,"frame":sub["frame"],"pred_progress_hmm":hmm})
        for k,t in enumerate(targets): out[t.replace("done_","pred_")]=pred[:,k]
        out.to_csv(f"test_pred_{rec}.csv",index=False); print(f"{rec}: test_pred_{rec}.csv")

if HAS_TEST_LABELS:
    print(f"\n TEST Result Summary")
    print(f"MAE raw = {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}")
    print(f"MAE HMM = {np.mean(all_mae_hmm):.2f} ± {np.std(all_mae_hmm):.2f}")
    print(f"Macro-F1    = {np.mean(all_macro):.2f} ± {np.std(all_macro):.2f}")
    print("F1 for each parts:")
    for t in targets:
        print(f"  {t:<28} {np.mean(per_part[t]):.2f}")

    # CV (leave-one-operator, honest train estimate) vs TEST
    test_mae   = float(np.mean(all_mae_hmm))
    test_macro = float(np.mean(all_macro))
    test_perpart = {t: float(np.mean(per_part[t])) for t in targets}
    print("\n CV (leave-one-operator) for Sensitive Analysis ")
    print(f"  Progress MAE (HMM):  CV = {cv_mae_m:.2f} +/- {cv_mae_s:.2f}   |   TEST = {test_mae:.2f}")
    print(f"  Macro-F1:            CV = {cv_macro_m:.2f} +/- {cv_macro_s:.2f}   |   TEST = {test_macro:.2f}")
    print(f"\n  {'part':<28}{'CV F1':>8}{'TEST F1':>10}")
    for t in targets:
        print(f"  {t.replace('done_',''):<28}{cv_perpart[t]:>8.2f}{test_perpart[t]:>10.2f}")


# Figures / text
def get_transitions(p): return [i for i in range(1,len(p)) if p[i]!=p[i-1]]
def state_to_text(bits):
    on=[PART_NAMES[i] for i in range(len(bits)) if bits[i]==1]
    return "+".join(on) if on else "(none)"

# GT vs Prediction
ncol=len(test_recs); fig,axes=plt.subplots(1,ncol,figsize=(7*ncol,5),squeeze=False)
for ax,rec in zip(axes[0],test_recs):
    sub,pred,raw_prog,hmm,Y=results[rec]
    if HAS_TEST_LABELS:
        true=Y.sum(1); ax.plot(true,label="true",lw=3,color="#1f77b4")
        for tpt in get_transitions(true): ax.axvline(tpt,color="#1f77b4",alpha=0.08)
    ax.plot(raw_prog,label="raw pred",lw=0.6,alpha=0.3,color="#999")
    ax.plot(hmm,label="HMM pred",lw=1.8,color="#d62728")
    ax.set_title(f"TEST: {rec}",fontweight="bold"); ax.set_xlabel("frame"); ax.set_ylabel("# parts"); ax.legend(fontsize=9)
fig.suptitle("TEST set: true vs predicted progress (HMM)",fontweight="bold")
fig.tight_layout(); fig.savefig("test_progress_compare.png",dpi=120)
print("\nsave in test_progress_compare.png")

# Confusion matrix
if HAS_TEST_LABELS:
    for rec in test_recs:
        sub,pred,raw_prog,hmm,Y=results[rec]
        nparts=len(targets)
        # 10 parts 2x2 confusion matrices
        fig,axes2=plt.subplots(2,5,figsize=(18,7))
        total_cm=np.zeros((2,2),int); f1s=[]
        for k in range(nparts):
            cm=confusion_matrix(Y[:,k],pred[:,k],labels=[0,1]); total_cm+=cm
            f1=f1_score(Y[:,k],pred[:,k],zero_division=0); f1s.append(f1)
            ax=axes2[k//5,k%5]; ax.imshow(cm,cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,cm[i,j],ha="center",va="center",
                            color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=11)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["pred 0","pred 1"],fontsize=7); ax.set_yticklabels(["true 0","true 1"],fontsize=7)
            ax.set_title(f"{PART_NAMES[k]}\nF1={f1:.2f}",fontsize=8)
        fig.suptitle(f"Per-part confusion matrices (rows=true, cols=pred) - {rec}",fontweight="bold")
        fig.tight_layout(); fig.savefig(f"test_confusion_perpart_{rec}.png",dpi=110); plt.close(fig)
        print(f"Save test_confusion_perpart_{rec}.png")


# Test Summary
for rec in test_recs:
    sub,pred,raw_prog,hmm,Y=results[rec]
    print(f"\n Test summary")
    print("[1] TASK PROGRESS (Number of completed parts):")
    if HAS_TEST_LABELS:
        true=Y.sum(1)
        print(f"    Ground Truth: {true[0]} -> {true[-1]};  Prediction(HMM): {hmm[0]} -> {hmm[-1]}")
    else:
        print(f"     Prediction(HMM): {hmm[0]} -> {hmm[-1]}")
    print("[2] WORKFLOW STATE (Current state of the keyframe):")
    for frac in [0.25,0.5,0.75,1.0]:
        idx=min(int(frac*(len(pred)-1)),len(pred)-1)
        if HAS_TEST_LABELS:
            print(f"    Frame{idx}: Ground Truth=[{state_to_text(Y[idx])}]")
            print(f"           Prediction=[{state_to_text(pred[idx])}]")
        else:
            print(f"    Frame{idx}: Prediction=[{state_to_text(pred[idx])}]")
    print("[3] TRANSITIONS:")
    pt=get_transitions(hmm)
    print(f"    Prediction(HMM)State transition frame: {pt[:12]}{'...' if len(pt)>12 else ''}")
    if HAS_TEST_LABELS:
        tt=get_transitions(Y.sum(1))
        print(f"    Frame for state transition in Ground Truth:     {tt[:12]}{'...' if len(tt)>12 else ''}")
