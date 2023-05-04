# LORIS

This is the official implementation of ["Long-Term Rhythmic Video Soundtracker"](https://arxiv.org/abs/2305.01319), ICML2023. 

Jiashuo Yu, Yaohui Wang, Xinyuan Chen, Xiao Sun, and Yu Qiao.  

<img src="https://raw.githubusercontent.com/OpenGVLab/LORIS/main/imgs/loris.png" width="224" height="224"> 

## Introduction  

We consider the problem of generating musical soundtracks in sync with rhythmic visual cues. Most existing works rely on pre-defined music representations, leading to the incompetence of generative flexibility and complexity. Other methods directly generating video-conditioned waveforms suffer from limited scenarios, short lengths, and unstable generation quality. To this end, we present Long-Term Rhythmic Video Soundtracker (LORIS), a novel framework to synthesize long-term conditional waveforms. Specifically, our framework consists of a latent conditional diffusion probabilistic model to perform waveform synthesis. Furthermore, a series of context-aware conditioning encoders are proposed to take temporal information into consideration for a long-term generation. Notably, we extend our model's applicability from dances to multiple sports scenarios such as floor exercise and figure skating. To perform comprehensive evaluations, we establish a benchmark for rhythmic video soundtracks including the pre-processed dataset, improved evaluation metrics, and robust generative baselines. Extensive experiments show that our model generates long-term soundtracks with state-of-the-art musical quality and rhythmic correspondence.  

![intro](/imgs/pipeline.png)  

## How to Start  

`pip install -r requirements.txt`

## Training  

## Inference  

## Dataset  

Download link for the dataset will be available soon.  

## Citation  

    @inproceedings{Yu2023Long,
    title={Long-Term Rhythmic Video Soundtracker},
    author={Yu, Jiashuo and Wang, Yaohui and Chen, Xinyuan and Sun, Xiao and Qiao, Yu },
    booktitle={International Conference on Machine Learning (ICML)},
    year={2023}
    }

## Acknowledgement  

We would like to thank the authors of previous related projects for generously sharing their code and insights: [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch), [CDCD](https://github.com/L-YeZhu/CDCD), [D2M-GAN](https://github.com/L-YeZhu/D2M-GAN), [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), and [JukeBox](https://github.com/openai/jukebox).
