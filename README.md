# Odor Plume Simulator
Realistic outdoor odor plume simulator 


### Algorithm
```
Start
 │
 └─ For Each Time Step `i`:
       │
       ├─ Retrieve `prior_prob` from `heatmap`
       |
       ├─ Calculate whiff frequency - B = 1+[(n(recent_whiffs)/len(recent_history))] * 2
       │
       ├─ Compute `posterior` using - posterior = prior * t_mat * Boost {where t_mat is transition mat}
       │
       ├─ Predict Whiff:
       │    └─ If random number < `posterior`: Whiff Predicted
       │
       ├─ Update `recent_history` with prediction
       │
       ├─ If Whiff Predicted AND Distance ≤ Threshold:
       │    ├─ Find Nearest Whiff Location
       │    ├─ Generate Odor Concentration Values
       │    ├─ Update `odor_concentration_samples`
       │    └─ Advance `i` Intermittency Duration
       │
       └─ Else:
            └─ Increment `i` by 1
 │
 └─ Post-Processing for No-Whiff Regions:
       ├─ Identify Regions with `base_odor_level`
       ├─ Generate No-Whiff Concentration Values
       └─ Smooth with Moving Average
 │
 └─ Update `df_test` with Results
 │     ├─ `predicted_odor`
 │     └─ `whiff_predicted`
 │
End
```