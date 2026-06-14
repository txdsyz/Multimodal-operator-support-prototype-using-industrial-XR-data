#%%
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_CSV    = r"E:\uu\others\phd apply\interview\XR reality- UU\final\Data for technical task\Data for technical task\train\PartA\merge.csv"
TRAIN_DINOV2 = r"E:\uu\others\phd apply\interview\XR reality- UU\final\Data for technical task\Data for technical task\train\PartA\dinov2_feats.csv"
TEST_CSV     = r"E:\uu\others\phd apply\interview\XR reality- UU\final\Data for technical task\Data for technical task\test\PartA_test\test.csv"
TEST_DINOV2  = r"E:\uu\others\phd apply\interview\XR reality- UU\final\Data for technical task\Data for technical task\test\PartA_test\dinov2_feats_test.csv"
# AR_labels aligned to the SAME test recording (27), so events and score share one timeline
AR_PATH      = r"E:\uu\others\phd apply\interview\XR reality- UU\final\Data for technical task\Data for technical task\test\PartB_test\27_assy_0_1\AR_labels.csv"

W=30; CORR_THRESH=0.9; THUMB_TIP,INDEX_TIP=5,10
HAS_TEST_LABELS=True

PART_NAMES=["front_chassis","front_chassis_pin","rear_chassis","short_rear_chassis",
            "front_rear_chassis_pin","rear_rear_chassis_pin","front_bracket",
            "front_bracket_screw","front_wheel_assy","rear_wheel_assy"]

# scoring parameters
DUR_FACTOR  = 1.75   # current step > 1.75 x own mean step duration = abnormally prolonged
P_GAZE      = 85     # personal gaze-distance percentile, above = gaze drift
P_ENERGY    = 85     # personal hand-energy percentile, above = high-effort struggle
WARM_STEPS  = 2      # first 2 steps: don't judge prolonged (rhythm not learned yet)
WARM_FRAMES = 200    # first 20s: don't trigger gaze/energy (personal sample too small)
LAMBDA_FAST = 0.016  # gaze drift and struggle, 5s to reach threshold
LAMBDA_SLOW = 0.004  # quiet but prolonged, slow accumulation
DECAY       = -0.05  #  normal pace


#  Part A (mainly the features extraction)
def hand_xy(d,side):
    xs=d[[f"{side}_j{j}_x" for j in range(26)]].values.astype(float)
    ys=d[[f"{side}_j{j}_y" for j in range(26)]].values.astype(float); return np.stack([xs,ys],-1)

def hand_features(d,side,present):
    J=hand_xy(d,side); J[~present]=np.nan
    c=np.nanmean(J,axis=1)
    cspeed=np.r_[np.nan,np.linalg.norm(np.diff(c,axis=0),axis=1)]
    energy=np.r_[np.nan,np.nanmean(np.linalg.norm(np.diff(J,axis=0),axis=2),axis=1)]
    spread=np.nanmean(np.linalg.norm(J-c[:,None,:],axis=2),axis=1)
    pinch=np.linalg.norm(J[:,THUMB_TIP]-J[:,INDEX_TIP],axis=1)
    return pd.DataFrame({f"{side}_cx":c[:,0],f"{side}_cy":c[:,1],f"{side}_cspeed":cspeed,
                         f"{side}_energy":energy,f"{side}_spread":spread,f"{side}_pinch":pinch},index=d.index)

def features_one(d):
    d=d.sort_values("frame").reset_index(drop=True)
    Lp=d["left_present"].astype(bool).values
    Rp=d["right_present"].astype(bool).values
    sl,sr=hand_features(d,"L",Lp),hand_features(d,"R",Rp)
    dom="R" if (np.nanmean(sr["R_energy"])>=np.nanmean(sl["L_energy"])) else "L"; non="L" if dom=="R" else "R"
    f=pd.DataFrame(index=d.index); roll=lambda s:s.rolling(W,min_periods=2)
    for tag,side,sc in [("dom",dom,sl if dom=="L" else sr),("non",non,sl if non=="L" else sr)]:
        pres=d[f"{'left' if side=='L' else 'right'}_present"]
        f[f"{tag}_present_ratio"]=pres.rolling(W,min_periods=1).mean()
        f[f"{tag}_cspeed"]=roll(sc[f"{side}_cspeed"]).mean()
        f[f"{tag}_energy"]=roll(sc[f"{side}_energy"]).mean()
        f[f"{tag}_spread"]=roll(sc[f"{side}_spread"]).mean()
        f[f"{tag}_pos_std"]=roll(sc[f"{side}_cx"]).std()+roll(sc[f"{side}_cy"]).std()
        f[f"{tag}_pinch_min"]=roll(sc[f"{side}_pinch"]).min()
        f[f"{tag}_pinch_mean"]=roll(sc[f"{side}_pinch"]).mean()
        f[f"{tag}_pinch_std"]=roll(sc[f"{side}_pinch"]).std()
    both=Lp&Rp
    hd=pd.Series(np.where(both,np.linalg.norm(sl[["L_cx","L_cy"]].values-sr[["R_cx","R_cy"]].values,axis=1),np.nan),index=d.index)
    f["hands_dist_mean"]=roll(hd).mean()
    f["hands_dist_std"]=roll(hd).std()
    f["both_present_ratio"]=pd.Series(both.astype(float),index=d.index).rolling(W,min_periods=1).mean()
    for c in ["fwd_x","fwd_y","fwd_z"]: f[f"{c}_mean"]=roll(d[c]).mean()
    f["fwd_var"]=roll(d["fwd_x"]).std()+roll(d["fwd_y"]).std()+roll(d["fwd_z"]).std()
    hp=d[["pos_x","pos_y","pos_z"]].values
    f["head_pos_speed"]=pd.Series(np.r_[np.nan,np.linalg.norm(np.diff(hp,axis=0),axis=1)],index=d.index).rolling(W,min_periods=2).mean()
    f["head_pos_std"]=roll(d["pos_x"]).std()+roll(d["pos_y"]).std()+roll(d["pos_z"]).std()
    f["gaze_mean_x"]=roll(d["gaze_x"]).mean()
    f["gaze_mean_y"]=roll(d["gaze_y"]).mean()
    f["gaze_std_x"]=roll(d["gaze_x"]).std()
    f["gaze_std_y"]=roll(d["gaze_y"]).std()
    f["gaze_speed"]=pd.Series(np.r_[np.nan,np.linalg.norm(np.diff(d[["gaze_x","gaze_y"]].values,axis=0),axis=1)],index=d.index).rolling(W,min_periods=2).mean()
    f["dominant_is_right"]=int(dom=="R")
    f.insert(0,"recording_id",d["recording_id"].values); f.insert(1,"frame",d["frame"].values)
    for c in [c for c in d.columns if c.startswith("done_")]: f[c]=d[c].values
    return f

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

def dedup_features(df, feats, thresh=CORR_THRESH):
    corr=df[feats].fillna(0).corr().abs().values
    keep=np.ones(len(feats),bool)
    for i in range(len(feats)):
        if not keep[i]: continue
        for j in range(i+1,len(feats)):
            if keep[j] and corr[i,j]>thresh: keep[j]=False
    return [feats[i] for i in range(len(feats)) if keep[i]]

def build_lgb():
    return LGBMClassifier(verbose=-1, random_state=27, n_jobs=1, num_leaves=31,
                          learning_rate=0.05, n_estimators=200, min_child_samples=30)

# Train final models (same as Part A )
tr_raw=pd.read_csv(TRAIN_CSV)
tr_feat=pd.concat([features_one(d) for _,d in tr_raw.groupby("recording_id")],ignore_index=True)
motion=[c for c in tr_feat.columns if c not in ["recording_id","frame"] and not c.startswith("done_")]
targets=[c for c in tr_feat.columns if c.startswith("done_") and c!="done_base"]
tr_dino=pd.read_csv(TRAIN_DINOV2)
rgb_cols=[c for c in tr_dino.columns if c.startswith("img_")]
TR=tr_feat.merge(tr_dino,on=["recording_id","frame"],how="inner")

FEATS=dedup_features(TR, motion+rgb_cols)   # correlation dedup, train only
models={}
for t in targets:
    y=TR[t].values
    models[t]=(None,y[0]) if y.min()==y.max() else (build_lgb().fit(TR[FEATS],y),None)

# Build test features, raw test table
te_raw=pd.read_csv(TEST_CSV)                  # raw merged test table
te_feat=pd.concat([features_one(d) for _,d in te_raw.groupby("recording_id")],ignore_index=True)
te_dino=pd.read_csv(TEST_DINOV2)
TE=te_feat.merge(te_dino,on=["recording_id","frame"],how="inner")
for c in FEATS:
    if c not in TE.columns: TE[c]=0.0
test_recs=TE.recording_id.unique().tolist()
print(f"TEST: {TE.shape[0]} frames, recordings {test_recs}")

def predict_rec(rec):
    sub=TE[TE.recording_id==rec].sort_values("frame").reset_index(drop=True)
    pred=np.zeros((len(sub),len(targets)),int)
    for k,t in enumerate(targets):
        m,const=models[t]
        pred[:,k]=const if m is None else (m.predict_proba(sub[FEATS])[:,1]>=0.5).astype(int)
    raw_prog=pred.sum(1)
    prob_lvl=np.zeros((len(sub),11))
    for i,p in enumerate(raw_prog):
        p=int(p); prob_lvl[i,p]=0.6
        if p>0: prob_lvl[i,p-1]+=0.2
        if p<10: prob_lvl[i,p+1]+=0.2
    hmm=hmm_decode(prob_lvl/prob_lvl.sum(1,keepdims=True))
    Y=sub[targets].values if HAS_TEST_LABELS else None
    return sub,pred,raw_prog,hmm,Y


# Part B specific: scoring, validation signals
def gaze_hand_min_dist(raw):
    #Distance from gaze to the nearer hand
    g=raw[["gaze_x","gaze_y"]].values
    L=raw[["L_j0_x","L_j0_y"]].values
    R=raw[["R_j0_x","R_j0_y"]].values
    d=np.fmin(np.linalg.norm(g-L,axis=1), np.linalg.norm(g-R,axis=1))
    gaze_valid=~((g[:,0]==0)&(g[:,1]==0))                                # gaze captured this frame?
    # smooth over W frames using valid frames only
    d_valid_only=pd.Series(np.where(gaze_valid & np.isfinite(d), d, np.nan))
    d_smooth=d_valid_only.rolling(W, min_periods=5).mean().values
    d_smooth=np.nan_to_num(d_smooth, nan=9999.0)
    return d_smooth, gaze_valid

def compute_score_online(hmm, Dt, Et, gaze_valid):
    #gaze drift， hand energy only climb rate once prolonged.
    T=len(hmm); score=np.zeros(T)
    Et=np.nan_to_num(Et,nan=0.0)
    completed=[]                                 # durations (frames) of completed steps
    last=0
    for t in range(1,T):
        if hmm[t]>hmm[t-1]:                      # a step completed - record its duration
            completed.append(t-last); last=t
        since=t-last                             # frames spent on the current step

        # personal duration baseline
        over=(len(completed)>=WARM_STEPS) and (since>DUR_FACTOR*np.mean(completed))

        # personal gaze,energy thresholds
        if t>=WARM_FRAMES and gaze_valid[:t].sum()>20:
            gthr=np.percentile(Dt[:t][gaze_valid[:t]], P_GAZE)   # only frames with gaze
            ethr=np.percentile(Et[:t], P_ENERGY)
            if not gaze_valid[t]:
                gaze_off=False
            else:
                gaze_off=(Dt[t]>gthr)
            busy=Et[t]>ethr
        else:
            gaze_off=busy=False

        if not over:        rate=DECAY            # normal pace
        elif gaze_off:      rate=LAMBDA_FAST      # prolonged + gaze drift = stuck
        elif busy:          rate=LAMBDA_FAST*0.5  # prolonged + high effort = stuck
        else:               rate=LAMBDA_SLOW      # prolonged + quiet = norm
        score[t]=max(0.0,min(1.0,score[t-1]+rate))
    return score

def extract_correction_events(ar, gap=40):
    # take,put,pull - wrong-pick and removal events.
    a=ar.copy()
    a["s"]=a["start_img"].astype(str).str.replace(".jpg","",regex=False).astype(int)
    a["e"]=a["end_img"].astype(str).str.replace(".jpg","",regex=False).astype(int)
    a=a.sort_values("s").reset_index(drop=True)
    takes=a[a["action_name"].str.startswith("take_")]
    ev=[]
    for _,p in a[a["action_name"].str.startswith("put_")].iterrows():
        part=p["action_name"].split("_",1)[1]
        cand=takes[(takes["action_name"]=="take_"+part)&(takes["s"]<=p["s"])]
        if len(cand) and (p["s"]-cand["s"].max())<=gap:
            ev.append((int(p["s"]),"wrong_pick",part))
    for _,p in a[a["action_name"].str.startswith("pull_")].iterrows():
        ev.append((int(p["s"]),"removal",p["action_name"].split("_",1)[1]))
    return sorted(ev)

def load_ar():
    if not os.path.exists(AR_PATH): return None
    return pd.read_csv(AR_PATH, header=None,
                       names=["recording_id","action_id","action_name","start_img","end_img"])


# Plot
ar=load_ar()
for rec in test_recs:
    sub_te =TE[TE.recording_id==rec].sort_values("frame").reset_index(drop=True)
    sub_raw=te_raw[te_raw.recording_id==rec].sort_values("frame").reset_index(drop=True)
    sub_raw=sub_raw[sub_raw["frame"].isin(sub_te["frame"].values)].reset_index(drop=True)
    _,_,_,hmm,Y=predict_rec(rec)

    Dt,gaze_valid=gaze_hand_min_dist(sub_raw)
    Et=sub_te["dom_energy"].fillna(0).values
    score=compute_score_online(hmm, Dt, Et, gaze_valid)
    T=len(score)

    events=extract_correction_events(ar) if ar is not None else []

    # figure
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,6.5),sharex=True)
    if HAS_TEST_LABELS:
        ax1.plot(Y.sum(1),label="True progress",lw=3,color="#1f77b4")
    ax1.plot(hmm,label="Predicted (HMM)",lw=2,color="#d62728")
    ax1.set_ylabel("# parts"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
    ax1.set_title(f"Stall - {rec}",fontweight="bold")

    ax2.plot(score,label="Stall score",lw=2,color="#ff7f0e")
    ax2.axhline(0.8,color="red",ls="--",alpha=0.7,label="Threshold 0.8")
    first=True
    wp=[f for f,t,_ in events if t=="wrong_pick"]
    rm=[f for f,t,_ in events if t=="removal"]
    if wp: ax2.scatter(wp,[0.04]*len(wp),marker="v",s=70,color="#ff7f0e",edgecolor="k",zorder=5,label="Wrong-pick")
    if rm: ax2.scatter(rm,[0.04]*len(rm),marker="X",s=80,color="red",edgecolor="k",zorder=5,label="Removal (pull)")
    for f,_,_ in events: ax2.axvline(f,color="k",alpha=0.12,lw=1)
    ax2.set_ylim(-0.05,1.05); ax2.set_ylabel("score")
    ax2.set_xlabel(f"frame (10 FPS, {T/10:.0f}s)")
    ax2.legend(loc="upper left",fontsize=8,ncol=2); ax2.grid(alpha=0.3)
    plt.tight_layout()
    out=f"Stall_{rec}.png"; fig.savefig(out,dpi=120); plt.close(fig)
    print(f"\n[{rec}] saved figure: {out}")