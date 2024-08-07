<img src="https://raw.githubusercontent.com/OpenGVLab/LORIS/main/imgs/logo.png" align="center"> 

# LORIS

This is the official implementation of "Long-Term Rhythmic Video Soundtracker", ICML2023. 

Jiashuo Yu, Yaohui Wang, Xinyuan Chen, Xiao Sun, and Yu Qiao.  

OpenGVLab, Shanghai Artificial Intelligence Laboratory  

[Arxiv](https://arxiv.org/abs/2305.01319) | [Project Page](https://justinyuu.github.io/LORIS/)  

## Introduction  

We present Long-Term Rhythmic Video Soundtracker (LORIS), a novel framework to synthesize long-term conditional waveforms in sync with visual cues. Our framework consists of a latent conditional diffusion probabilistic model to perform waveform synthesis. Furthermore, a series of context-aware conditioning encoders are proposed to take temporal information into consideration for a long-term generation. We also extend our model's applicability from dances to multiple sports scenarios such as floor exercise and figure skating. To perform comprehensive evaluations, we establish a benchmark for rhythmic video soundtracks including the pre-processed dataset, improved evaluation metrics, and robust generative baselines.    

![intro](/imgs/pipeline.png)  

## How to Start  

`pip install -r requirements.txt`

## Training  

`bash scripts/loris_{subset}_s{length}.sh`  

## Inference  

`bash scripts/infer_{subset}_s{length}.sh`  

## Dataset  

Dataset is available in [huggingface](https://huggingface.co/datasets/awojustin/LORIS).  

    from datasets import load_dataset
    dataset = load_dataset("OpenGVLab/LORIS")

## Model Zoo  

We provide the pre-trained checkpoints and backbone audio diffusion model as follow:  

[Audio-diffusion-pytorch-v0.0.43](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/loris/audio_1136_720k.pt),  
[Dance 25 seconds](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/loris/loris_dance25_epoch100.pt),  
[Figure Skating 25 seconds](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/loris/loris_fs25_loss_min.pt),  
[Floor Exercise 25 seconds](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/loris/loris_fe25_epoch200.pt),  
[Floor Exercise 50 seconds](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/loris/loris_fe50_epoch250.pt)  

It should be noted that these checkpoints must only be used for research purposes.

## Citation  

    @inproceedings{Yu2023Long,
    title={Long-Term Rhythmic Video Soundtracker},
    author={Yu, Jiashuo and Wang, Yaohui and Chen, Xinyuan and Sun, Xiao and Qiao, Yu },
    booktitle={International Conference on Machine Learning (ICML)},
    year={2023}
    }

## Acknowledgement  

We would like to thank the authors of previous related projects for generously sharing their code and insights: [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch), [CDCD](https://github.com/L-YeZhu/CDCD), [D2M-GAN](https://github.com/L-YeZhu/D2M-GAN), [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), and [JukeBox](https://github.com/openai/jukebox).
