<div align="center">

<h1>Extension of MotionDiffuse for SMPLX features</h1>
<h3>02456 Deep Learning, DTU Compute, Fall 2023</h3>
<div>
    <a>Elle Mcfarlane</a>&emsp;
    <a>Alejandro Cirugeda</a>&emsp;
    <a>Jonathan Mikler</a>&emsp;
    <a>Menachem Franko</a>&emsp;
</div>
<div>
    equal contribution&emsp;
<h4 align="center">
  <a href="https://arxiv.org/abs/2208.15001" target='_blank'>[Paper]</a> •
  <a href="https://github.com/ellemcfarlane/MotionDiffuse" target='_blank'>[Github]</a>
</h4>
</div>

<div style="display: flex; justify-content: center;">
    <img src="text2motion/other/denoising.png" alt="De-noising process" style="width: 50%; margin: 0 auto;">
    <img src="text2motion/other/happy_guy.png" alt="Happy guy" style="width: 40%; margin: 0 auto;">
</div>
    
</div>

## Summary
Conditioning human motion on natural language (text-to-motion) is critical for many graphics-based applications, including training neural nets for motion-based tasks like detecting changes in posture for medical applications. Recently diffusion models have become popular for text-to-motion generation, but many are trained on human pose representations that lack face and hand details and thus fall short on prompts that involve emotion or detailed object interaction. To fill this gap, we re-trained the text-to-motion model MotionDiffuse on a new dataset Motion-X, which uses SMPL-X poses to include facial expressions and fully articulated hands.

## Installation
Go to [text2motion/DTU_readme.md](text2motion/dtu_README.md) for installation instructions

## Demo
to experience the demo of the model, check the notebook [text2motion/demo.ipynb](text2motion/demo.ipynb). The notebook will guide you through the process of generating a motion from a text prompt. the same code is also available as a python script [text2motion/demo.py](text2motion/demo.py).\
**Note:** To visualize the output, the `make gen` command must be run from the `text2motion` directory.

## Acknowledgements
The group would like to thank the authors of the original paper for their work and for making their code available. Also, a deep thank you to Frederik Warbug, for his support and technical guidance and to the DTU HPC team for their support with the HPC cluster.