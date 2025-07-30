import numpy as np
import os
from LSTM_TCN_model import LSTM_TCN_fcn_1room

myModel = LSTM_TCN_fcn_1room()
suffix = "savedModel"
realData = "two_room_artificial_hl_1_0_exchange_0_0001.csv"
generatedCO2 = "genEx_co2_r1_TR_artificial_NV_hl_1_0_exchange_0_0001.csv"
generatedOcc = "genEx_occ_r1_TR_artificial_NV_hl_1_0_exchange_0_0001.csv"
accTestMainDataGen=myModel.run(suffix, "data/"+realData,"data/"+generatedCO2,"data/"+generatedOcc)