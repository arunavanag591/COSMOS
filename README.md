# Odor Plume Simulator
Realistic outdoor odor plume simulator 


### Algorithm
```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'arial', 'lineWidth': '2px', 'primaryTextColor': '#fff', 'textColor': '#fff' }}}%%
flowchart TD
    subgraph Main["Main Simulation Flow"]
        Init["1. Initialize System
        - Base odor level: 0.6
        - Rows per second: 200
        - History length: 7"] 
        --> SpatialProb["2. Calculate Spatial Probabilities"]
        SpatialProb --> Predict["3. Prediction Loop"]
        Predict --> Output["4. Post-processing
        - Gaussian smoothing (σ=0.8)
        - Rolling window (size=14)"]
    end
    
    subgraph SpatialProbCalc["Spatial Probability Calculation"]
        SP1["Create 2D Histogram
        (50x50 bins)"] --> SP2["Calculate Probability Field
        p_ij = k_ij/n_ij"]
        SP2 --> SP3["Fit Gaussian Plume Model
        A, x0, y0, σx, σy, θ"]
        SP3 --> SP4["Apply Spatial Smoothing
        and Distance Adjustment"]
    end
    
    subgraph PredictionLoop["Prediction Process"]
        P1["Process Segments
        (size=2000)"] --> P2["Update State"]
        P2 --> P3{"In Whiff State?
        Check Spatial Prior π_i"}
        P3 -->|Yes| W1["Generate Whiff
        Duration from empirical dist."]
        P3 -->|No| B1["Generate Background
        μ_blank ≈ 0.6"]
        W1 --> StateUpdate["Update System State"]
        B1 --> StateUpdate
        StateUpdate --> P2
    end
    
    subgraph WhiffGeneration["Whiff Generation"]
        WG1["Calculate AR2 Parameters
        φ1=0.98, φ2=-0.02"] --> 
        WG2["Update in Z-space
        logit transform"] -->
        WG3["Apply Inverse Logit
        C ∈ [0,10]"] -->
        WG4["Add Distance-Dependent Noise
        scale ∝ exp(-d/50)"]
    end
    
    subgraph StateUpdate["State Management"]
        SU1["Update Histories
        - Recent concentrations
        - Whiff states"] --> 
        SU2["Calculate Transition Probabilities
        Based on spatial prior"] -->
        SU3["Apply Markov Chain
        P(transition) matrix"] -->
        SU4["Update Concentrations
        And sample intermittency"]
    end
    
    Main -.->|"Spatial priors π_i"| SpatialProbCalc
    PredictionLoop -.->|"Whiff parameters"| WhiffGeneration
    PredictionLoop -.->|"State variables"| StateUpdate
    
    classDef main fill:#2d3436,stroke:#dfe6e9,stroke-width:2px
    classDef process fill:#2c3e50,stroke:#ecf0f1,stroke-width:1px
    classDef decision fill:#34495e,stroke:#ecf0f1,stroke-width:1px
    
    class Main,SpatialProbCalc,PredictionLoop,WhiffGeneration,StateUpdate main
    class P3 decision
    class Init,SpatialProb,Predict,Output,SP1,SP2,SP3,SP4,P1,P2,W1,B1,StateUpdate,WG1,WG2,WG3,WG4,SU1,SU2,SU3,SU4 process
