This file contains the improved application that was created after the Be Curious 2022 event. Please make sure that you download the FeatureExtract.py file and have the following files installed:
- sounddevice
- soundfile
- numpy
- pandas
- scipy
- time
- matplotlib
- sklearn
- parselmouth
- tkinter
- pickle
- pysinewave
- os
- random

The files from the Saarbruecken voice dataset were downloaded in six age categories under 14, 15-20, 21-30, 31-40, 41-50, over 50. All of the files were healthy patients except for the under 14 category in which all of the recordings in the database were downloaded due to the small size of the category. Each age category was downloaded into a sperate folder given the name of the category. The files can be downloaded here: http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4

The csv files with the acoustic features, ages, genders, and predictions must be downloaded from this repository. Ensure that the paths in these csv files is updated to the place where you store them. The models must also be downloaded. The file paths on lines 42, 52, 58, 64, 208, and 214 must also be updated.
