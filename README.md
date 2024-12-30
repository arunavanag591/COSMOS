# Odor Plume Simulator
Realistic outdoor odor plume simulator 


### Algorithm
```mermaid
flowchart TD
    %% Main Simulation Flow
    subgraph Main["Main Simulation Flow"]
        Init[1. Initialize System] --> SpatialProb[2. Calculate Spatial Probabilities]
        SpatialProb --> Predict[3. Prediction Loop]
        Predict --> Output[4. Post-processing & Output]
    end

    %% Spatial Probability Calculation
    subgraph SpatialProbCalc["Spatial Probability Calculation"]
        SP1[Create 2D Histogram] --> SP2[Calculate Probability Field]
        SP2 --> SP3[Fit Gaussian Plume Model]
        SP3 --> SP4[Apply Spatial Smoothing]
    end

    %% Prediction Process
    subgraph PredictionLoop["Prediction Process"]
        P1[Process Segments] --> P2[Update State]
        P2 --> P3{In Whiff State?}
        P3 -->|Yes| W1[Generate Whiff]
        P3 -->|No| B1[Generate Background]
        W1 --> StateUpdate[Update System State]
        B1 --> StateUpdate
        StateUpdate --> P2
    end

    %% Whiff Generation
    subgraph WhiffGeneration["Whiff Generation"]
        WG1[Calculate AR2 Parameters] --> WG2[Update in Z-Space]
        WG2 --> WG3[Apply Inverse Logit]
        WG3 --> WG4[Add Distance-Dependent Noise]
    end

    %% State Management
    subgraph StateUpdate["State Management"]
        SU1[Update Histories] --> SU2[Calculate Transition Probabilities]
        SU2 --> SU3[Apply Markov Chain]
        SU3 --> SU4[Update Concentrations]
    end

    %% Linking subgraphs`
    Main -.- SpatialProbCalc
    PredictionLoop -.- WhiffGeneration
    PredictionLoop -.- StateUpdate

    
    Main -.- SpatialProbCalc
    PredictionLoop -.- WhiffGeneration
    PredictionLoop -.- StateUpdate
    
    classDef main fill:#2d3436,stroke:#dfe6e9,stroke-width:2px
    classDef process fill:#2c3e50,stroke:#ecf0f1
    classDef decision fill:#34495e,stroke:#ecf0f1
    
    class Main,SpatialProbCalc,PredictionLoop,WhiffGeneration,StateUpdate main
    class P3 decision
    class Init,SpatialProb,Predict,Output,SP1,SP2,SP3,SP4,P1,P2,W1,B1,StateUpdate,WG1,WG2,WG3,WG4,SU1,SU2,SU3,SU4 process
