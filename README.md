# Training code for Loss-to-loss prediction

## Description

Code to train models that was used for the experiments on loss-to-loss prediction in the paper [Loss-to-loss prediction: scaling laws for all datasets](https://arxiv.org/abs/2411.12925).

For data analysis and results see https://github.com/KempnerInstitute/loss-to-loss-notebooks.

For model checkpoints see https://huggingface.co/KempnerInstituteAI/loss-to-loss.

## Installation

Build a conda environment and install the required packages:

```bash
conda create -n loss-to-loss python=3.10
conda activate loss-to-loss
pip install -e .[all]
```

## Usage

Training models requires preparing all of the tokenized data and then running the following command to launch the training sweep on a slurm cluster:

```bash
sbatch launch_sweep_scale.sh configs/base.yaml configs/sweeps/scale.yaml
```

Note: you will need to update paths to the data in `olmo/registry.py` and define the array job as well as paths for logs and checkpoints in `launch_sweep_scale.sh`.

## Citation

If you use this code in your research, please cite the following papers:

```bibtex
@article{brandfonbrener2024loss,
      title={Loss-to-Loss Prediction: Scaling Laws for All Datasets}, 
      author={Brandfonbrener, David and Anand, Nikhil and Vyas, Nikhil and Malach, Eran and Kakade, Sham},
      journal={arXiv preprint arXiv:2411.12925},
      year={2024}
}
```

```bibtex
@article{OLMo,
  title={OLMo: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```
