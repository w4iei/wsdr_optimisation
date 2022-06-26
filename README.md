# wsdr_optimisation
##Resources to download:
###SiSEC18-MUS-30-WAV.zip: 
https://zenodo.org/record/1256003#.YrhhgC8Rpf1
These are the submitted estimates used in SiSEC2018
###Submitted scores from SiSEC2018:
https://github.com/sigsep/sigsep-mus-2018
###MUSDB18
https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems
These should go into the `wsdr_optimization` project directory with the following names: "SiSEC18-MUS-30-WAV", "sigsep-mus-2018" "musdb18"


## Requirements
See requirements.txt for required.
This code leverages NUSSL, but it is important that your virtual environment not have access to the released version of NUSSL or MUSEVAL, as this uses a hacked version (included) that allows for adjusting WSDR weights.
Same goes for museval