
# Music Style Transfer Using Pre-Trained Autoencoders
## Final project for Computers and Music

Convolutional Neural Networks (CNNs) extract features whose representations can encode both the style and the content information of an image. This particular property has made them the perfect candidates to transfer the style of one image onto another one. The quality of the resulting pictures motivates us to adopt the audio equivalent of Image CNNs to perform music style transfer. Most existing methods operate on spectral audio representation instead of raw waveform. Unlike current methods, we propose using auto-encoders trained in a self-supervised manner to transfer the timbre of one instrument to an audio sample played by another instrument. To that effect, we first experiment in the image domain and then apply the methods whose output were promising onto the audio domain. Results show that well-established methods for image style transfer are not very promising for audio style transfer.


This repo is organized in the following way: 
```
.
├── README.md
├── final_report.pdf
├── data 
│   ├── exp1
│   ├── exp2
│   └── exp3
└──notebooks
└──apps



```

### Instructions to understand the project
1. Read the report final_report.pdf
2. Read and see the notebooks (in order)
3. Replicate our analysis. We have added the raw files in the data folder per experiments.

The results are available [here](https://drive.google.com/drive/folders/1QVr9ckQe0ZWqHEcA6_eZ9GeX2BGiMuj8?usp=sharing)
