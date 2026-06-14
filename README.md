**1.Multimodal Operator-Support Prototype on IndustReal**

A prototype pipeline that estimates assembly state and an operator stall indicator from
HoloLens 2 egocentric data (IndustReal). It has two parts:
(1) Assembly state estimation predicts the per-part installation status at each frame,from which task progress, workflow state, and transitions follow.
(2) Operator stall indicator detects abnormal hesitation at the current step, measured against each operator's own baseline.


A full description of the problem formulation, method, results, and limitations is in the
report [Report.pdf](Report.pdf)
<img width="523" height="286" alt="method" src="https://github.com/user-attachments/assets/43e03bc2-8714-47d1-997f-3a79ff3c74dd" />



**2.How to run?**

The scripts run in order. Steps 1 and 2 prepare the data; steps 3 and 4 are the two parts.


Step 1: Merge sensor data 
Place the recording folders under data/ (each with gaze.csv, hands.csv, pose.csv,
PSR_labels.csv), then run Merge_data.py. This produces the merged motion table.


Step 2: Extract RGB features (Google Colab, GPU)
Upload the merged table and the RGB image frames to Google Drive, then run dinov2_rbg.py
in Colab (Runtime → Change runtime type → GPU). This produces dinov2_feats.csv for train and test, respectively.


Step 3: Assembly state estimation
Run Assembly_state_estimation.py. It trains the part classifiers, applies HMM smoothing, 
and reports the progress MAE and Macro-F1 on the test recording, with the progress curve, 
confusion matrices, and feature importance.


Step 4: Operator stall indicator
Run Indicators.py. It reuses the Assembly state estimation progress and the gaze and hand features to
compute the  stall score and plots it.

Update the file paths at the top of each script to your local paths before running.



**3.Requirements**

The three local scripts need the packages in requirements.txt:

pip install -r requirements.txt

The RGB feature extraction (dinov2_rbg.py) runs on Colab, where torch, torchvision,
and Pillow are already available.


**4.Data**

This work uses a subset of the IndustReal dataset (Schoonbeek et al., 2024). The training set is four operators' recordings(22_assy_0_1, 22_assy_2_3, 25_assy_0_1, 25_assy_2_1); 
the test set is one separate recording (27_assy_0_1) from an operator not seen during training.
