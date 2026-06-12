# Deep Learning Approaches for Music Source Separation

**COMP6252: Deep Learning Technologies**

**Ritesh Kumar Behera**  
University of Southampton  
Student ID: 37614002  
Email: rkb1u25@soton.ac.uk

## Abstract

Music Source Separation (MSS) is recognised as one of the most difficult tasks. Separation of single-instrument components (and/or vocal) from a combined audio signal is very difficult. The project undertakes the task of isolating source musical components from multi-stem musical tracks, using Deep Neural Network techniques. Specifically, two Neural Network models are implemented: Band-Split Recurrent Neural Network (BS-RNN) and Temporal Frequency and Shifted Window Attention-based ResUNet (TFSWA-ResUNet), and these two models are then tested and compared on a subset of the MoisesDB database. The performance of both models is assessed over the extraction of the vocal stem, and the strengths and limitations, along with suitability for real-world multi-source separation tasks, are analysed. A comparative study is done based on the results achieved by the two models. Experimental results on MoisesDB database demonstrate that BS-RNN outperforms TFSWA-ResUNet by 0.238 dB in mean SDR of 4.482 dB vs 4.244 dB.

## 1. Introduction

The music source separation task is a complex process that involves separating a mixed audio signal, such as a song, into its constituent sound sources. The different types of sound sources for a typical musical track can be vocals, drums, bass, and some other melodic instruments. The task of music source separation is challenging due to the overlapping and interdependent nature of these elements. Despite that, music source separation has immense importance in the audio processing world. It opens the door to a huge number of applications, such as:

- Music remixing
- Automatic transcription
- Music education
- Content-based audio retrieval

Researchers in the Deep Neural Network Technology domain have also had a profound interest in solving such a complicated process. There are several research studies about implementing different Neural Network-based Audio processing techniques to carry out meaningful source separation of musical tracks. Despite significant progress, this task still possesses a certain number of open challenges:

- The separation of similar frequencies ranged audio sources such as vocals and lead guitar
- Most of the existing models use Western datasets, ignoring a large database of musical instruments that might not be of Western origins

### 1.1. Problem Statement

The main objective of this experimental project is to investigate and evaluate methods for Music source separation (MSS), addressing the challenge of isolating individual audio sources from a mixed recording. The project employs two main Deep Learning Neural Network pipelines, taking into account the audio processing steps within the models. For proper and impartial comparison of the models, the evaluation is done on the separation task of the vocal stem of the mixed audio signals only.

The experimental setup employed in this project is documented within the GitLab repository referenced in this report.

### 1.2. Dataset

This project makes use of an open-source dataset **MoisesDB** that is specifically prepared for the task of Music Source Separation. MoisesDB is made up of:

- 240 songs
- 47 different artists
- 12 musical genres
- Isolated audio for each individual instrument source in a song (rather than just a handful of mixed stems per track)

MoisesDB organizes its audio into 11 top-level stem categories:

- Bass
- Bowed strings
- Drums
- Guitar
- Other
- Other keys
- Other plucked
- Percussion
- Piano
- Vocals
- Wind

Each song has a different number of stems (between 3 and 10) depending on which instruments are used. Vocals, drums, and bass appear in nearly every song, while stems such as wind are rare.

To make the dataset easy to adopt, the authors provide a dedicated Python library that handles downloading, processing, and interacting with MoisesDB. This is accompanied by detailed documentation and an in-depth analysis of what the dataset contains, making it easier for researchers who want to start using it in their own work.

### 1.3. Evaluation Metrics

The experimental setup for this report makes use of the numerical value of Signal-to-Distortion Ratio (SDR) as the primary objective evaluation metric for assessing the performance of both audio source separation models.

#### Signal-to-Distortion Ratio (SDR)

Given an estimated source signal (s_i), SDR measures the ratio of the signal energy of true source (s_target) to the total distortion energy, which is decomposed into three distinct error components:

- Interference from other sources (e_interf)
- Background noise (e_noise)
- Processing artefacts introduced by the separation algorithm (e_artif)

The resulting score is expressed in decibels (dB), where higher values indicate closer resemblance to the clean target source and therefore superior separation quality.

**Mathematical Formula:**

```
SDR = 10 log₁₀ (||s_target||²) / (||e_interf + e_noise + e_artif||²)
```

SDR is particularly favoured over simpler metrics because its multi-component distortion formulation provides a better evaluation, penalising models that suppress noise at the cost of introducing artefacts and/or residual interference, thereby ensuring a balanced comparison between architectures.

## 2. Model Architecture and Experimental Setup

### 2.1. TFSWA-ResUNet Model

#### 2.1.1 Innovation

Traditional Convolutional Neural Networks (CNNs) are excellent at analyzing local audio details but struggle to understand the "big picture" or long-range patterns of a song. In contrast, modern Transformers are incredible at understanding the global context of a song, but they require massive, expensive supercomputers to run.

The authors built a hybrid model that:

1. Uses a lightweight CNN (ResUNet) to handle the heavy lifting of shrinking and expanding the local audio data
2. Applies the TFSWA attention module at the very bottom of the network, where the data is fully compressed, to analyze the global context

The proposed model successfully mitigates the performance-efficiency trade-off, yielding the superior separation fidelity typical of massive Transformer architectures without sacrificing the computational speed and lightweight parameterization of standard CNNs.

#### 2.1.2 Architecture

The TFSWA-ResUNet model is implemented using a ResUNet architecture, which features a symmetric U-shaped encoder-decoder structure connected by skip connections. The process consists of:

**Input Processing:**
- Stereo mixture waveforms transformed into spectrogram magnitudes through Short-Time Fourier Transform (STFT)
- Spectrograms split into sub-bands and stacked to form an eight channel input feature map

**Encoder:**
- Five blocks, each containing:
  - Conv block with four residual convolutional modules (RCM)
  - Down-sampling layer utilizing average pooling
  - Each RCM composed of:
    - Two 3×3 convolutional layers
    - GELU activation
    - Batch normalization
    - 1×1 convolution providing shortcut connection

**Decoder:**
- Mirror-images encoder structure
- Uses bilinear interpolation for up-sampling
- Concatenates features from encoder to recover target source through Inverse STFT (ISTFT)

**Bottleneck:**
- Four Temporal Frequency and Shifted Window Attention (TFSWA) modules designed to capture both global and local correlations within the music spectrogram
- Incorporates:
  - Time Sequence Attention (TSA) block
  - Frequency Sequence Attention (FSA) block
  - Both utilize multi-head self attention to model long-range dependencies along temporal and frequency axes respectively
  - Residual branch featuring a Swin transformer with shifted window mechanism for computing self attention within local non-overlapping windows

This hybrid approach allows the deep learning model to achieve high separation performance with a relatively small number of parameters.

#### 2.1.3 Experimental Setup

The training process begins by randomly splitting the entire Moises dataset tracks into training, validation and test sets in a ratio of 80:10:10.

**Data Augmentation Strategy:**
- Each song converted into a stereo input
- Songs segmented into 3-second clips
- Two 3-second segments drawn from the same source are randomly combined to produce a new training segment
- This approach generates a large number of single-source combinations while preserving source identity and thereby creating a more diverse and generalised dataset for training

**Spectrogram Transformation:**
- Short-Time Fourier Transform (STFT) with window size of 2048 samples and hop size of 441 samples

**Training Configuration:**
- Batch size: 16 samples per epoch
- Loss function: L1 loss computed in the waveform domain
- Optimiser: Adam
- Initial learning rate: 1e⁻³
- Hyperparameters finetuned with varying learning rate decay, providing a smooth annealing schedule throughout training

**Decoder and Output:**
- Output features from the last layer are flattened and merged along the channel dimension
- Merged spectrogram magnitudes combined with phase mixture to recover target waveform by applying Inverse STFT (ISTFT)
- Final waveforms used to evaluate SDR value

### 2.2. Band-Split RNN Model

#### 2.2.1 Innovation

Conventional spectrogram separators apply a single, uniform backbone across the entire frequency axis, implicitly assuming every bin carries the same kind of information. In music, however, each source occupies a distinct, target-specific frequency support:

- Bass: below ~500 Hz
- Vocals: 0.2–4 kHz
- Drum cymbals: above 8 kHz

Band-Split RNN (BSRNN) exploits this anisotropy by:

1. Splitting the spectrogram into non-uniform, target-specific sub-bands
2. Embedding each sub-band into a shared latent space
3. Alternating two recurrent passes—one along time, one along bands—to capture temporal dynamics and inter-band dependencies

This approach avoids the quadratic cost of full self-attention, resulting in a compact, fully recurrent model that is target-aware by construction and reaches state-of-the-art SDR with far fewer parameters than spectrogram U-Nets or end-to-end waveform Transformers.

#### 2.2.2 Architecture

The model follows the band-split → dual-path RNN → mask-estimation pipeline, with internal STFT/iSTFT (nfft=2048, hop=512, Hann window).

**Frequency Partitioning:**
- 1025 frequency bins partitioned into K = 41 non-uniform sub-bands
- Target-specific scheme (V7 layout for vocals/other; low-end-biased schemes for bass/drums)
- Each complex sub-band slice is:
  - Reshaped
  - LayerNorm-ed
  - Linearly projected to shared latent dimension N = 128
  - Produces tensor z ∈ ℝ^(B×N×K×T)

**Backbone:**
- Stacks L = 12 dual-path blocks
- Each containing two residual BLSTM units:
  - **Sequence RNN** (shared across bands, applied along time axis T)
  - **Band RNN** (shared across frames, applied along band axis K)
  - Each unit applies:
    - LayerNorm → BLSTM(hidden=2N) → Linear(4N → N) with residual skip

This captures long-range temporal dynamics and inter-band correlations at a cost linear in L rather than quadratic in K·T.

**Mask Estimation and Output:**
- Per-band MLP: Linear(N → 4N) → Tanh → Linear(4N → 4C_wk) → GLU
- Produces bounded complex time–frequency mask of width w_k
- Sub-band masks concatenated along frequency
- Multiplied with mixture spectrogram
- Inverted by iSTFT to recover estimated waveform

#### 2.2.3 Experimental Setup

A separate BSRNN is trained per target on MoisesDB v0.1, split at the song level (80/10/10% train/val/test).

**Data Preparation:**
- Instrument stems remapped to four MUSDB-style targets: {vocals, bass, drums, other}
- Training samples: 3 s segments drawn by a source-activity detector that flags 6 s windows as salient when at least 50% of their 0.6 s sub-chunks exceed the 15th-percentile dB threshold
- Mixture and target sliced from the same song at the same offset to preserve the additive relationship

**Data Augmentation:**
- Applied jointly to both signals:
  - Random gain in [−10, +10] dB
  - Stereo channel swap (p=0.5)
  - Polarity inversion (p=0.5)
  - Peak normalisation to [−1, 1]

**Optimisation:**
- AdamW (lr = 10⁻³, weight-decay 10⁻²)
- Cosine annealing to lr/100
- Gradient clipping at ℓ2 norm 5.0
- 30 epochs × 800 segments
- Batch size: 4
- Hardware: Single CUDA GPU
- Early stopping: After 15 stagnant epochs

**Inference:**
- Overlap-add with 3 s windows
- 1.5 s hop
- Bartlett synthesis window identical to validation pipeline

## 3. Results and Discussion

### Results Summary

| Model | Mean SDR |
|-------|----------|
| TFSWA-ResUNet | 4.244 dB |
| Band-Split RNN | 4.482 dB |

**Table 1.** Mean SDR values calculated over 16 test dataset songs

The Band-Split RNN model achieves a mean SDR of 4.482 dB, which outperforms the recorded mean SDR value of 4.244 dB for the TFSWA-ResUNet model. The margin between the two model performances is approximately 0.238 dB, indicating that while both the deep learning model architectures demonstrate comparable separation capability, the Band-Split RNN holds a modest but measurable advantage.

### Analysis of Band-Split RNN's Superior Performance

The possible explanation for the Band-Split RNN model's slightly superior performance can be due to its **frequency-domain decomposition strategy**, which processes different frequency bands independently before recombining them. This allows the model to better capture the spectral characteristics of individual sources.

In contrast, the TFSWA-ResUNet model, despite leveraging residual connections and time-frequency attention mechanisms, appears slightly less effective at minimising the combined distortion, resulting in a slightly lower SDR value.

### Training Dynamics

#### TFSWA-ResUNet Training Dynamics

Both models are evaluated over respective training parameters for 30 epochs on the vocal stem. For the TFSWA-ResUNet model, the training demonstrates a **stable behaviour**:

- Training and validation loss curves descend smoothly from outset
- Finally converge steadily after 30 epochs
- Validation loss consistently lower than training loss throughout the training process
- Suggests effective generalisation and minimal overfitting
- Narrow and stable gap between curves indicates efficient learning without significant variance across epochs

#### Band-Split RNN Training Dynamics

In contrast, the Band-Split RNN model exhibits **considerably more volatile training dynamics**:

- Both training and validation curves start at substantially higher values
- Fluctuate considerably throughout the training process
- Several prominent spikes observed in the validation loss curve at regular intervals
- Suggests an unsteady training process
- While overall trend remains downward and converges eventually, fluctuations suggest the model is more sensitive to batch-level variations

### SDR Validation Performance Comparison

Finally, Figure 6 shows the evaluation of the SDR values on the validation dataset in a regular interval of epochs:

- **TFSWA-ResUNet:** Exhibited higher peak SDR values, reaching nearly 5 dB in some early epochs and also around epoch 25, with considerable fluctuations
- **BS-RNN:** Demonstrates a steadier and more consistent upward SDR trajectory, ultimately converging to comparable final values near 3.5 dB

This suggests there exists a **trade-off between peak performance and training stability** across the two architectures.

## 4. Conclusion

The experimental results demonstrate that both the neural network models show considerable capability of performing meaningful audio source separation of musical tracks. Although the Band-Split RNN model achieved a marginally superior SDR value of 4.482 dB compared to 4.244 dB for TFSWA-ResUNet, the TFSWA-ResUNet model exhibited a rather stable and smooth training dynamics suggesting a more reliable model architecture.

Nevertheless, the relatively narrow performance gap suggests that both models are competitive for this task. Future work could explore hybrid architectures that integrate:

- The attention-based spatial modelling of TFSWA-ResUNet
- The band-splitting strategy of Band-Split RNN

This potential hybrid approach could yield further SDR improvements.

## References

[1] E. Cano, D. Fitzgerald, A. Liutkus, M. D. Plumbley, and F. R. Stoter, "Musical Source Separation: An Introduction," *IEEE Signal Processing Magazine*, vol. 36, no. 1, pp. 31–40, January 2019. [Online]. Available: https://ieeexplore.ieee.org/document/8588410

[2] "Music-Source-Separation-Git." [Online]. Available: https://github.com/RiteshBeheraUoS/Music-Source-Separation

[3] I. Pereira, F. Araujo, F. Korzeniowski, and R. Vogl, "MoisesDB: A dataset for source separation beyond 4-stems," *Proceedings of the International Society for Music Information Retrieval Conference*, vol. 2023, pp. 619–626, July 2023. [Online]. Available: https://arxiv.org/pdf/2307.15913

[4] "GitHub - moises-ai/moises-db: Moises Source Separation Public Dataset · GitHub." [Online]. Available: https://github.com/moises-ai/moises-db

[5] Z. Yao, Y. Su, H. Yang, Y. Zhang, and X. Wu, "TFSWA-ResUNet: music source separation with time–frequency sequence and shifted window attention-based ResUNet," *Eurasip Journal on Advances in Signal Processing*, vol. 2025, no. 1, December 2025.

[6] X. Song, Q. Kong, X. Du, and Y. Wang, "CatNet: music source separation system with mix-audio augmentation," February 2021. [Online]. Available: http://arxiv.org/abs/2102.09966

[7] Y. Luo and J. Yu, "Music Source Separation With Band-Split RNN," *IEEE/ACM Transactions on Audio Speech and Language Processing*, vol. 31, pp. 1893–1901, 2023.
