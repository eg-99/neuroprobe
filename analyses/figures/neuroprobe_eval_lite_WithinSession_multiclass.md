# Performance Comparison - WithinSession (Neuroprobe MultiClass)

Performance comparison across tasks (mean ± SEM). **Best performing model** for each task is shown in bold.

| Model | Overall | Sentence Onset | Speech | Volume |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.575 ± 0.003 | 0.795 ± 0.021 | 0.656 ± 0.022 | 0.539 ± 0.008 |
| Linear (spectrogram) | 0.593 ± 0.004 | 0.851 ± 0.025 | 0.825 ± 0.028 | **0.615 ± 0.022** |
| Linear (Laplacian+spectrogram) | **0.617 ± 0.003** | **0.891 ± 0.018** | **0.883 ± 0.018** | 0.612 ± 0.019 |
| BrainBERT (untrained, frozen) | 0.560 ± 0.003 | 0.752 ± 0.029 | 0.598 ± 0.021 | 0.529 ± 0.006 |
| BrainBERT (frozen) | 0.560 ± 0.003 | 0.753 ± 0.029 | 0.606 ± 0.022 | 0.530 ± 0.006 |
| PopulationTransformer | 0.546 ± 0.005 | 0.725 ± 0.043 | 0.687 ± 0.034 | 0.561 ± 0.016 |

| Model | Delta Volume | Voice Pitch | Word Position | Inter-word Gap |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.623 ± 0.011 | 0.520 ± 0.003 | 0.696 ± 0.016 | 0.546 ± 0.007 |
| Linear (spectrogram) | 0.609 ± 0.017 | 0.532 ± 0.005 | 0.643 ± 0.023 | 0.542 ± 0.012 |
| Linear (Laplacian+spectrogram) | **0.640 ± 0.017** | **0.541 ± 0.007** | **0.712 ± 0.022** | **0.558 ± 0.008** |
| BrainBERT (untrained, frozen) | 0.599 ± 0.012 | 0.516 ± 0.005 | 0.640 ± 0.021 | 0.547 ± 0.010 |
| BrainBERT (frozen) | 0.599 ± 0.013 | 0.513 ± 0.005 | 0.641 ± 0.022 | 0.543 ± 0.012 |
| PopulationTransformer | 0.602 ± 0.027 | 0.508 ± 0.011 | 0.542 ± 0.020 | 0.517 ± 0.014 |

| Model | GPT-2 Surprisal | Head Word Position | Part of Speech | Word Length |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.539 ± 0.006 | 0.570 ± 0.008 | 0.537 ± 0.005 | 0.562 ± 0.007 |
| Linear (spectrogram) | 0.526 ± 0.009 | 0.565 ± 0.012 | 0.528 ± 0.008 | 0.538 ± 0.010 |
| Linear (Laplacian+spectrogram) | **0.555 ± 0.008** | **0.602 ± 0.012** | **0.551 ± 0.008** | **0.569 ± 0.010** |
| BrainBERT (untrained, frozen) | 0.536 ± 0.007 | 0.581 ± 0.011 | 0.524 ± 0.004 | 0.547 ± 0.010 |
| BrainBERT (frozen) | 0.537 ± 0.008 | 0.583 ± 0.012 | 0.522 ± 0.004 | 0.549 ± 0.011 |
| PopulationTransformer | 0.516 ± 0.011 | 0.511 ± 0.004 | 0.499 ± 0.008 | 0.507 ± 0.009 |

| Model | Global Optical Flow | Local Optical Flow | Frame Brightness | Number of Faces |
|-------|---------|---------|---------|---------|
| Linear (voltage) | 0.526 ± 0.006 | 0.521 ± 0.003 | 0.500 ± 0.008 | 0.498 ± 0.005 |
| Linear (spectrogram) | 0.552 ± 0.008 | 0.539 ± 0.012 | **0.518 ± 0.010** | 0.507 ± 0.004 |
| Linear (Laplacian+spectrogram) | **0.564 ± 0.009** | **0.552 ± 0.010** | 0.513 ± 0.013 | **0.514 ± 0.007** |
| BrainBERT (untrained, frozen) | 0.514 ± 0.005 | 0.514 ± 0.004 | 0.503 ± 0.003 | 0.505 ± 0.004 |
| BrainBERT (frozen) | 0.511 ± 0.005 | 0.513 ± 0.006 | 0.501 ± 0.005 | 0.503 ± 0.004 |
| PopulationTransformer | 0.519 ± 0.010 | 0.516 ± 0.010 | 0.482 ± 0.016 | 0.498 ± 0.010 |