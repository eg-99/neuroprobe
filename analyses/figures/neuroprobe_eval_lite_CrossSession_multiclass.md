# Performance Comparison - CrossSession (Neuroprobe MultiClass)

Performance comparison across tasks (mean ± SEM). **Best performing model** for each task is shown in bold.

| Model | Overall | Sentence Onset | Speech | Volume |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.554 ± 0.002 | 0.728 ± 0.021 | 0.611 ± 0.014 | 0.530 ± 0.003 |
| Linear (spectrogram) | 0.593 ± 0.003 | 0.861 ± 0.016 | 0.849 ± 0.020 | **0.616 ± 0.015** |
| Linear (Laplacian+spectrogram) | **0.611 ± 0.003** | **0.904 ± 0.012** | **0.889 ± 0.018** | 0.611 ± 0.012 |
| BrainBERT (untrained, frozen) | 0.552 ± 0.003 | 0.722 ± 0.032 | 0.609 ± 0.016 | 0.526 ± 0.005 |
| BrainBERT (frozen) | 0.557 ± 0.003 | 0.738 ± 0.032 | 0.631 ± 0.018 | 0.535 ± 0.006 |
| PopulationTransformer | 0.562 ± 0.005 | 0.760 ± 0.034 | 0.711 ± 0.032 | 0.561 ± 0.014 |

| Model | Delta Volume | Voice Pitch | Word Position | Inter-word Gap |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.596 ± 0.005 | 0.515 ± 0.003 | 0.642 ± 0.016 | 0.537 ± 0.006 |
| Linear (spectrogram) | 0.604 ± 0.014 | 0.530 ± 0.005 | 0.632 ± 0.022 | 0.532 ± 0.010 |
| Linear (Laplacian+spectrogram) | 0.630 ± 0.014 | **0.539 ± 0.008** | **0.681 ± 0.019** | **0.551 ± 0.008** |
| BrainBERT (untrained, frozen) | 0.585 ± 0.011 | 0.503 ± 0.002 | 0.631 ± 0.022 | 0.525 ± 0.010 |
| BrainBERT (frozen) | 0.585 ± 0.011 | 0.507 ± 0.003 | 0.634 ± 0.022 | 0.529 ± 0.009 |
| PopulationTransformer | **0.632 ± 0.025** | 0.517 ± 0.008 | 0.572 ± 0.026 | 0.526 ± 0.012 |

| Model | GPT-2 Surprisal | Head Word Position | Part of Speech | Word Length |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.529 ± 0.006 | 0.537 ± 0.005 | 0.530 ± 0.005 | 0.536 ± 0.008 |
| Linear (spectrogram) | 0.535 ± 0.006 | 0.557 ± 0.011 | 0.531 ± 0.007 | 0.537 ± 0.009 |
| Linear (Laplacian+spectrogram) | 0.547 ± 0.007 | **0.580 ± 0.009** | **0.549 ± 0.007** | **0.571 ± 0.007** |
| BrainBERT (untrained, frozen) | 0.534 ± 0.008 | 0.579 ± 0.015 | 0.515 ± 0.005 | 0.541 ± 0.010 |
| BrainBERT (frozen) | 0.534 ± 0.008 | 0.577 ± 0.015 | 0.517 ± 0.006 | 0.540 ± 0.009 |
| PopulationTransformer | **0.558 ± 0.014** | 0.523 ± 0.006 | 0.505 ± 0.006 | 0.526 ± 0.006 |

| Model | Global Optical Flow | Local Optical Flow | Frame Brightness | Number of Faces |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.510 ± 0.002 | 0.506 ± 0.002 | 0.504 ± 0.004 | 0.503 ± 0.002 |
| Linear (spectrogram) | 0.538 ± 0.006 | 0.534 ± 0.008 | **0.523 ± 0.007** | 0.512 ± 0.003 |
| Linear (Laplacian+spectrogram) | **0.549 ± 0.006** | **0.539 ± 0.006** | 0.514 ± 0.008 | **0.516 ± 0.006** |
| BrainBERT (untrained, frozen) | 0.509 ± 0.004 | 0.508 ± 0.003 | 0.499 ± 0.002 | 0.497 ± 0.003 |
| BrainBERT (frozen) | 0.510 ± 0.003 | 0.512 ± 0.003 | 0.506 ± 0.006 | 0.499 ± 0.003 |
| PopulationTransformer | 0.518 ± 0.010 | 0.516 ± 0.012 | 0.509 ± 0.013 | 0.497 ± 0.008 |