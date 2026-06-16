import os
from jcamp import jcamp_read
import json
import pandas as pd
from data_preproccesing_lib import *
import numpy as np

files = [r"data-preprocessing-pipeline\IR_data-lab\Ethyleneglycol11.dx",
         r"data-preprocessing-pipeline\IR_data-lab\PET measurement final 1.dx",
         r"data-preprocessing-pipeline\IR_data-lab\TPA measurement1.dx"]


jcampfile = open(files[0], "r")
raw_data = jcamp_read(jcampfile)
jcampfile.close()
x = np.linspace(raw_data["firstx"],raw_data["lastx"],raw_data['y'].size)
plot_spectrum("Spectrum of EG", x, raw_data["y"])

jcampfile = open(files[1], "r")
raw_data = jcamp_read(jcampfile)
jcampfile.close()
x = np.linspace(raw_data["firstx"],raw_data["lastx"],raw_data['y'].size)
plot_spectrum("Spectrum of PET", x, raw_data["y"])

jcampfile = open(files[2], "r")
raw_data = jcamp_read(jcampfile)
jcampfile.close()
x = np.linspace(raw_data["firstx"],raw_data["lastx"],raw_data['y'].size)
plot_spectrum("Spectrum of TPA", x, raw_data["y"])