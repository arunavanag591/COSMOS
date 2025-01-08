# Odor Plume Simulator
Realistic outdoor odor plume simulator 


### Algorithm
```mermaid
  flowchart TD

    %% -----------------------------
    %% 1) INPUT DATA
    %% -----------------------------
    subgraph InD["Input Data"]
        A1[Raw Odor Measurements]
        A2[Wind Velocity]
        A3[Spatial Coordinates]
    end

    %% -----------------------------
    %% 2) SPATIAL PROBABILITY MODEL
    %% -----------------------------
    subgraph SPM["Spatial Probability Model"]
        B1[Empirical Bin Probabilities]
        B2[Gaussian Plume Model]
        B3[Parameter Optimization]
        B4[Final Spatial Probability Field]

        B1 --> B2
        B2 --> B3
        B3 --> B4
    end

    %% -----------------------------
    %% 3) TEMPORAL FRAMEWORK
    %% -----------------------------
    subgraph TempF["Temporal Framework"]
        C1[Logistic Transform]
        C2[State Management]
        C3[Whiff Memory]
    end

    %% -----------------------------
    %% 4) STATE TRANSITIONS
    %% -----------------------------
    subgraph ST["State Transitions"]
        D1[Prior Probability]
        D2[Markov Chain]
        D3[Recent History]
        D4[Whiff State Decision]
    end

    %% -----------------------------
    %% 5) CONCENTRATION EVOLUTION
    %% -----------------------------
    subgraph CE["Concentration Evolution"]
        E1[AR-2 in z-space]
        E2[Target Concentration]
        E3[Inverse Transform]
        E4[Final Concentration]
    end

    %% -----------------------------
    %% FINAL OUTPUT
    %% -----------------------------
    F1[Predicted Odor Field]

    %% -----------------------------
    %% MAIN CONNECTIONS
    %% -----------------------------
    %% Final Probability Field -> used as "Prior Probability"
    SPM --> D1       

    %% (e.g. part of config or an additional prior adjustment)
    TempF --> D1     

    %% State Management updates Recent History
    C2 --> D3        

    %% Whiff Memory also updates Recent History
    C3 --> D3        

    %% Prior Probability -> Markov Chain
    D1 --> D2        

    %% Markov Chain -> Whiff Decision
    D2 --> D4        

    %% Combined with recent events -> Whiff Decision
    D3 --> D4        

    %% Logistic Transform used inside AR-2
    C1 --> E1        

    %% Whiff Decision toggles AR-2 update
    D4 --> E1        

    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> F1

    %% -----------------------------
    %% STYLE / COLOR CODING (optional)
    %% -----------------------------
    classDef input fill:#f3f4f6,stroke:#1c1c1c,color:#1c1c1c,stroke-width:1px
    classDef spatial fill:#e8f5e9,stroke:#388e3c,color:#1c1c1c,stroke-width:1px
    classDef temporal fill:#e3f2fd,stroke:#1976d2,color:#1c1c1c,stroke-width:1px
    classDef state fill:#ede7f6,stroke:#673ab7,color:#1c1c1c,stroke-width:1px
    classDef concentration fill:#fff3e0,stroke:#ff6f00,color:#1c1c1c,stroke-width:1px
    classDef output fill:#f5f5f5,stroke:#aaaaaa,color:#1c1c1c,stroke-width:1px

    class A1,A2,A3 input
    class B1,B2,B3,B4 spatial
    class C1,C2,C3 temporal
    class D1,D2,D3,D4 state
    class E1,E2,E3,E4 concentration
    class F1 output


    %% -----------------------------
    %% STYLE / COLOR CODING (optional)
    %% -----------------------------
    classDef input fill:#f3f4f6,stroke:#1c1c1c,color:#1c1c1c,stroke-width:1px
    classDef spatial fill:#e8f5e9,stroke:#388e3c,color:#1c1c1c,stroke-width:1px
    classDef temporal fill:#e3f2fd,stroke:#1976d2,color:#1c1c1c,stroke-width:1px
    classDef state fill:#ede7f6,stroke:#673ab7,color:#1c1c1c,stroke-width:1px
    classDef concentration fill:#fff3e0,stroke:#ff6f00,color:#1c1c1c,stroke-width:1px
    classDef output fill:#f5f5f5,stroke:#aaaaaa,color:#1c1c1c,stroke-width:1px

    class A1,A2,A3 input
    class B1,B2,B3,B4 spatial
    class C1,C2,C3 temporal
    class D1,D2,D3,D4 state
    class E1,E2,E3,E4 concentration
    class F1 output


