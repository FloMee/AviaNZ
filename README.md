
# AviaNZ: open-source software for manual and automatic analysis of bio-acoustic recordings

**This version of AviaNZ was developed within the scope of a master's thesis at the [Unviversity of Applied Sciences Dresden](https://www.htw-dresden.de)
and introduces the two BirdNET classifiers [BirdNET-Lite](https://github.com/kahst/BirdNET-Lite) and [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) into the [original Version of AviaNZ](https://github.com/smarsland/AviaNZ).**


This software enables you to:
* classify recordings with BirdNET
* review and listen to wav files from acoustic field recorders, 
* segment and annotate the recordings, 
* train filters to recognise calls from particular species, 
* use filters that others have provided to batch process many files
* review annotations
* produce output in spreadsheet form, or as files ready for further statistical analyses

For more information about the project, see http://www.avianz.net

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

# Installation

## Windows
Windows binaries are available under [realeases](https://github.com/FloMee/AviaNZ/releases).
To install from source, follow the Linux instructions.
<!-- 
## macOS
An installer script is available at http://www.avianz.net.
To install from source, follow the Linux instructions. -->

## Linux

No binaries are available. The following procedure was succesfully testet with Python 3.9.16.
On Ubuntu, install from source as follows:

1. Ensure Python, pip and git are available on your system. these can be installed by running the following from the command line:  
>sudo apt install python3-pip git
2. Clone the repository by running:
>git clone https://github.com/FloMee/AviaNZ.git
3. Install the required packages by running:
>pip3 install -r requirements.txt --user
4. Build the Cython extensions by running:
>cd ext; python3 setup.py build_ext -i; cd ..  
5. Done! Launch the software with:
>python3 AviaNZ.py

# Acknowledgements

AviaNZ is based on PyQtGraph and PyQt, and uses Librosa and Scikit-learn amongst others.

Development of this software was supported by the RSNZ Marsden Fund, and the NZ Department of Conservation.
