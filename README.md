# PAMalyzer - Audio anaylsis software based on AviaNZ

# Description

PAMalyzer enables you to:

- Show and navigate through spectrograms of audio files
- Annotate audio files
- Use the [BirdNET](https://github.com/birdnet-team/birdnet) classification models on your data
- Review the results of the classification process

# Installation

Depending on your operating system we provide different solutions to install and run PAMalyzer:

## Windows

Windows binaries are available under [realeases](https://github.com/FloMee/PAMalyzer/releases).
To install from source, follow the Linux instructions.

## Linux/MacOS

No binaries are available. The following procedure was succesfully testet with Python 3.11
On Ubuntu, install from source as follows:

1. Ensure Python, pip and git are available on your system. these can be installed by running the following from the command line:
   > sudo apt install python3-pip git
2. Clone the repository by running:
   > git clone https://github.com/FloMee/PAMalyzer.git
3. Install the required packages by running:
   > pip3 install -r requirements.txt --user
4. Build the Cython extensions by running:
   > cd ext; python3 setup.py build_ext -i; cd ..
5. Done! Launch the software with:
   > python3 PAMalyzer.py

# Manual

We provide an in-depth manual in the 'Help' menu of the software or under [Docs/PAMalyzerManual.pdf](https://github.com/FloMee/PAMalyzer/blob/master/Docs/PAMalyzerManual.pdf).

# Acknowledgements

PAMalyzer is a fork of [AviaNZ](https://github.com/smarsland/AviaNZ) and is based on PyQtGraph and PyQt, and uses Librosa and Scikit-learn amongst others.

The development of this software is supported by the European Union through the European Social Fund Plus (ESF+) within the framework of the ESF PLUS programme “Cooperative State Innovation Doctorates” and the State of Saxony.

# Citation

If you use this software, please credit us in any papers that you write. An appropriate reference is:

```
@article{Marsland19,
  title = "AviaNZ: A future-proofed program for annotation and recognition of animal sounds in long-time field recordings",
  author = "{Marsland}, Stephen and {Priyadarshani}, Nirosha and {Juodakis}, Julius and {Castro}, Isabel",
  journal = "Methods in Ecology and Evolution",
  volume = 10,
  number = 8,
  pages = "1189--1195",
  year = 2019
}
```
