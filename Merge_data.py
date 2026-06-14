#%%
"""
Merge multimodal sensor data (multi-recording) : Organize multiple recordings into a single summary table containing
the `recording_id`.
Sample directory structure: Each dataset corresponds to a folder containing "gaze/hands/pose/PSR_labels.csv".
   recordings/
     rec01/ gaze.csv hands.csv pose.csv PSR_labels.csv
     rec02/ ...
     rec03/ ...
     rec04/ ...
"""
#%%
"""
Classify the status of the 11 parts into two states, 0 and 1,
representing “not installed” and “installed”, respectively. 
This way, the status of each frame is represented by 11 binary digits (0s and 1s). 
Since there is already a “base” by default, the digital representation of each frame's status begins with a 1.

Each state is represented by this 11-bit code. You can cross-reference it with the official state numbers; 
if there is a match, use the official state designation; if not, use “unnamed.”
"""

#%%
import pandas as pd, numpy as np, os, glob

ROOT = "data" #The root directory where your data is stored locally

PART_IDX = {
    "base":0, "front chassis":1, "front chassis pin":2, "rear chassis":3,
    "short rear chassis":4, "front rear chassis pin":5, "rear rear chassis pin":6,
    "front bracket":7, "front bracket screw":8, "front wheel assy":9, "rear wheel assy":10,
}
PART_NAMES = [None]*11
for k, v in PART_IDX.items():
    PART_NAMES[v] = k

# 22 states
STATES = {
 "10000000000":"state 1","10010010000":"state 2","10010100000":"state 3","10010110000":"state 4",
 "11100000000":"state 5","11110010000":"state 6","11110100000":"state 7","11110110000":"state 8",
 "11110111100":"state 9","11110111110":"state 10","11110110001":"state 11","11110111101":"state 12",
 "11110111111":"state 13","11110101111":"state 14","11110011111":"state 15","11110011110":"state 16",
 "11110101110":"state 17","11100001110":"state 18","11101101110":"state 19","11101011110":"state 20",
 "11101111110":"state 21","11101111111":"state 22",
}

def frame_to_idx(name): return int(str(name).split(".")[0])

def step_to_part(step_name):
    # Remove the "Install" of "Install + Parts"
    return step_name.lower().replace("install", "", 1).strip()

def load_one(folder, rec_id):
    # no header
    gaze = pd.read_csv(f"{folder}/gaze.csv", header=None)
    gaze.columns = ["frame", "gaze_x", "gaze_y"]
    pose = pd.read_csv(f"{folder}/pose.csv", header=None)
    pose.columns = ["frame","fwd_x","fwd_y","fwd_z","pos_x","pos_y","pos_z","up_x","up_y","up_z"]
    hands = pd.read_csv(f"{folder}/hands.csv", header=None)
    hcols = [f"{s}_j{j}_{c}" for s in ["L","R"] for j in range(26) for c in ["x","y"]]
    hands.columns = ["frame"] + hcols

    # Align by frame name
    df = gaze.merge(pose, on="frame").merge(hands, on="frame")
    df["idx"] = df["frame"].map(frame_to_idx)
    df = df.sort_values("idx").reset_index(drop=True)

    #Hand track?
    Lc = [c for c in hcols if c.startswith("L_")]
    Rc = [c for c in hcols if c.startswith("R_")]
    df["left_present"]  = (df[Lc].abs().sum(axis=1) > 0).astype(int)
    df["right_present"] = (df[Rc].abs().sum(axis=1) > 0).astype(int)

    # PSR
    psr = pd.read_csv(f"{folder}/PSR_labels.csv", header=None,
                      names=["frame","step_id","step_name"])
    psr["idx"] = psr["frame"].map(frame_to_idx)

    unknown = [n for n in psr["step_name"] if step_to_part(n) not in PART_IDX]
    if unknown:
        print(f"   {rec_id} has unknown step: {set(unknown)}")

    #For each completed frame to complete the list of part bits to be set to 1 for that frame
    bits_at_frame = {}
    for _, r in psr.iterrows():
        part = step_to_part(r["step_name"])
        if part in PART_IDX:
            bits_at_frame.setdefault(r["idx"], []).append(PART_IDX[part])

    # Accumulate frame by frame to produce an 11-bit code (the base is always 1)
    completion_frames = sorted(bits_at_frame.keys())
    codes, state_names = [], []
    part_matrix = np.zeros((len(df), 11), dtype=int)
    code = [0]*11; code[0] = 1
    ptr = 0
    for row_i, fidx in enumerate(df["idx"].values):
        while ptr < len(completion_frames) and completion_frames[ptr] <= fidx:
            for b in bits_at_frame[completion_frames[ptr]]:
                code[b] = 1
            ptr += 1
        s = "".join(map(str, code))
        codes.append(s)
        state_names.append(STATES.get(s, "unnamed"))
        part_matrix[row_i] = code

    df["part_code"]  = codes
    df["state_name"] = state_names
    for b in range(11):
        df[f"done_{PART_NAMES[b].replace(' ','_')}"] = part_matrix[:, b]

    df.insert(0, "recording_id", rec_id)
    return df, psr

# Scan all recording folders
folders = sorted([f for f in glob.glob(f"{ROOT}/*") if os.path.isdir(f)])
all_df, paths = [], {}
for f in folders:
    rec_id = os.path.basename(f)
    d, psr = load_one(f, rec_id)
    all_df.append(d)

    traj = list(dict.fromkeys(d["part_code"]))
    paths[rec_id] = [(c, STATES.get(c, "unnamed")) for c in traj]
    n_named = d["state_name"].ne("unnamed").mean()*100
    print(f"{rec_id}: {d.shape[0]} frame "
          f"Frames matching the official state account for {n_named:.0f}%")

big = pd.concat(all_df, ignore_index=True)


for rec, traj in paths.items():
    print(f"  {rec}:")
    for code, name in traj:
        print(f"      {code}  {name}")

big.to_csv("Merge_recording.csv", index=False)
print(f"\nThe table：{big.shape[0]} frame in Merge_recording.csv")
