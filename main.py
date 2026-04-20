from sympy import factor

from functions import *
import math

FOLDER_PATH_0 = "./23_12_2025_vernier/test_3/"
DEVICE_SET_0 = {
    "output_1": {
        "trigger": "TEK00101.csv",
        "channel": "TEK00100.csv"
    },
}

FOLDER_PATH_1 = "./19_12_2025_vernier/"
DEVICE_SET_1 = {
    "voltage_1": {
        "trigger": "TEK00021.csv",
        "channel": "TEK00020.csv"
    },
    "voltage_2": {
        "trigger": "TEK00023.csv",
        "channel": "TEK00022.csv"
    },

}

FOLDER_PATH_2 = "./3_12_2025_vernier/test_2/"
FOLDER_PATH_3 = "./3_12_2025_vernier/test_3/"
FOLDER_PATH_4 = "./3_12_2025_vernier/test_4/"
FOLDER_PATH_5 = "./20_3_2026_vernier/"

SWEEP_RANGE = 100 # from 1260nm to 1360nm
SWEEP_RATE = 2 # nm /2
NUM_ROWS = 2*10**6
TIME_AMOUNT = 64
DATA_LENGTH = int(NUM_ROWS / TIME_AMOUNT * SWEEP_RANGE / SWEEP_RATE)

DEVICE_SET_2 = {
    # test 2
    "single_bus": {
        "channel": "TEK00102.csv",
        "trigger": "TEK00103.csv"
    },
    "output_1": {
        "channel": "TEK00104.csv",
        "trigger": "TEK00105.csv"
    },
    "output_2": {
        "channel": "TEK00106.csv",
        "trigger": "TEK00107.csv"
    },
    "output_3": {
        "channel": "TEK00109.csv",
        "trigger": "TEK00110.csv"
    },
    "output_4": {
        "channel": "TEK00111.csv",
        "trigger": "TEK00112.csv"
    },
    "output_5": {
        "channel": "TEK00113.csv",
        "trigger": "TEK00114.csv"
    },
}

DEVICE_SET_3 = {
    # test 3
    # output 3
    "single_bus": {
        "channel": "TEK00102.csv",
        "trigger": "TEK00103.csv"
    },
    "voltage_0": {
        "channel": "TEK00128.csv",
        "trigger": "TEK00129.csv"
    },
    "voltage_1": {
        "channel": "TEK00130.csv",
        "trigger": "TEK00131.csv"
    },
    "voltage_2": {
        "channel": "TEK00132.csv",
        "trigger": "TEK00133.csv"
    },
    "voltage_3": {
        "channel": "TEK00134.csv",
        "trigger": "TEK00135.csv"
    },
    "voltage_4": {
        "channel": "TEK00124.csv",
        "trigger": "TEK00125.csv"
    },
    "voltage_5": {
        "channel": "TEK00126.csv",
        "trigger": "TEK00127.csv"
    }
}

DEVICE_SET_4 = {
    "single_bus": {
        "channel": "TEK00102.csv",
        "trigger": "TEK00103.csv"
    },
    "voltage_0": {
        "channel": "TEK00136.csv",
        "trigger": "TEK00137.csv"
    },
    "voltage_1": {
        "channel": "TEK00138.csv",
        "trigger": "TEK00139.csv"
    },
    "voltage_2": {
        "channel": "TEK00140.csv",
        "trigger": "TEK00141.csv"
    },
    "voltage_3": {
        "channel": "TEK00142.csv",
        "trigger": "TEK00143.csv"
    },
    "voltage_4": {
        "channel": "TEK00144.csv",
        "trigger": "TEK00145.csv"
    },
    "voltage_5": {
        "channel": "TEK00146.csv",
        "trigger": "TEK00147.csv"
    },
    "voltage_6": {
        "channel": "TEK00148.csv",
        "trigger": "TEK00149.csv"
    },
}

DEVICE_SET_5 = {
    "single_bus": {
        "channel": "TEK00102.csv",
        "trigger": "TEK00103.csv"
    },
    "voltage_0": {
        "trigger": "TEK00019.csv",
        "channel": "TEK00018.csv"
    },
    "voltage_2": {
        "trigger": "TEK00017.csv",
        "channel": "TEK00016.csv"
    }
}
# folder path : ./20_3_2026_vernier/
DEVICE_SET_6 = {
    "single_bus": {
        "channel": "TEK00102.csv",
        "trigger": "TEK00103.csv"
    },
    "voltage_0": {
        "trigger": "TEK00020.csv",
        "channel": "TEK00021.csv"
    },
    "voltage_1_5": {
        "trigger": "TEK00025.csv",
        "channel": "TEK00024.csv"
    },
    "voltage_2": {
        "trigger": "TEK00027.csv",
        "channel": "TEK00026.csv"
    },
    "voltage_3": {
        "trigger": "TEK00023.csv",
        "channel": "TEK00022.csv"
    },
    "voltage_3_b": {
        "trigger": "TEK00029.csv",
        "channel": "TEK00028.csv"
    }
}

# folder path : ./20_3_2026_vernier/
DEVICE_SET_7 = {
    "single_bus": {
        "channel": "TEK00102.csv",
        "trigger": "TEK00103.csv"
    },
    "voltage_0": {
        "trigger": "TEK00031.csv",
        "channel": "TEK00030.csv"
    },
    "voltage_1": {
        "trigger": "TEK00033.csv",
        "channel": "TEK00032.csv"
    },
    "voltage_1_5": {
        "trigger": "TEK00035.csv",
        "channel": "TEK00034.csv"
    },
    "voltage_1_7": {
        "trigger": "TEK00047.csv",
        "channel": "TEK00046.csv"
    },
    "voltage_1_8": {
        "trigger": "TEK00045.csv",
        "channel": "TEK00044.csv"
    },
    "voltage_1_9": {
        "trigger": "TEK00043.csv",
        "channel": "TEK00042.csv"
    },
    "voltage_2": {
        "trigger": "TEK00037.csv",
        "channel": "TEK00036.csv"
    },
    "voltage_2_5": {
        "trigger": "TEK00039.csv",
        "channel": "TEK00038.csv"
    },
    "voltage_3": {
        "trigger": "TEK00041.csv",
        "channel": "TEK00040.csv"
    },
}

FOLDER_PATH_6 = "./27_3_2026_vernier/"
DEVICE_SET_8 = {
    # row 4 V10
    # voltage unit: 0.1V
    # output scale: 200mV * 4
    # recording outout 1, through port
    "voltage_05": {
        "trigger": "TEK00049.csv",
        "channel": "TEK00048.csv"
    },
    # fwhm mode: 0.0907
    # fsr mode: 0.08756
    # Q factor mode: 13683.65868
    # Q factor average: 14473.44625
    # Q factor std: 3573.17202
    "voltage_06": {
        "trigger": "TEK00051.csv",
        "channel": "TEK00050.csv"
    },
    # fwhm mode: 0.08357
    # fsr mode: 0.08643
    # Q factor mode: 15690.55405
    # Q factor average: 14526.1814
    # Q factor std: 3457.75566
    "voltage_07": {
        "trigger": "TEK00053.csv",
        "channel": "TEK00052.csv"
    },
    # fwhm mode: 0.07304
    # fsr mode: 0.0854
    # Q factor mode: 16198.48706
    # Q factor average: 14477.22511
    # Q factor std: 3575.50313
    "voltage_08": { #retest
        "trigger": "TEK00071.csv",
        "channel": "TEK00070.csv"
    },
    # fwhm mode: 0.08109
    # fsr mode: 0.08785
    # Q factor mode: 15131.77077
    # Q factor average: 14303.44377
    # Q factor std: 3238.3053
    "voltage_09": { #retest
        "trigger": "TEK00073.csv",
        "channel": "TEK00072.csv"
    },
    # fwhm mode: 0.08007
    # fsr mode: 0.08773
    # Q factor mode: 13677.30069
    # Q factor average: 14322.82197
    # Q factor std: 3465.6421
    "voltage_10": {
        "trigger": "TEK00059.csv",
        "channel": "TEK00058.csv"
    },
    # fwhm mode: 0.08559
    # fsr mode: 0.09058
    # Q factor mode: 14693.17604
    # Q factor average: 14244.25979
    # Q factor std: 3346.65403
    "voltage_11": {
        "trigger": "TEK00061.csv",
        "channel": "TEK00060.csv"
    },
    # fwhm mode: 0.07807
    # fsr mode: 0.08664
    # Q factor mode: 14263.24846
    # Q factor average: 14500.06402
    # Q factor std: 3446.0433
    "voltage_12": { #retest
        "trigger": "TEK00075.csv",
        "channel": "TEK00074.csv"
    },
    # fwhm mode: 0.09731
    # fsr mode: 0.08804
    # Q factor mode: 13908.35187
    # Q factor average: 14213.9647
    # Q factor std: 3370.1399
    "voltage_13": { #retest
        "trigger": "TEK00077.csv",
        "channel": "TEK00076.csv"
    },
    # fwhm mode: 0.07888
    # fsr mode: 0.08749
    # Q factor mode: 15535.50044
    # Q factor average: 14423.06491
    # Q factor std: 3531.29794
    "voltage_14": {
        "trigger": "TEK00067.csv",
        "channel": "TEK00066.csv"
    },
    # fwhm mode: 0.07966
    # fsr mode: 0.08643
    # Q factor mode: 12367.51057
    # Q factor average: 14262.34758
    # Q factor std: 3435.44962
    "voltage_15": {
        "trigger": "TEK00069.csv",
        "channel": "TEK00068.csv"
    },
    # fwhm mode: 0.08028
    # fsr mode: 0.08743
    # Q factor mode: 13967.31993
    # Q factor average: 14290.76553
    # Q factor std: 3193.88138
}

DEVICE_SET_9 = {
    # output of the overall port of vernier device
    "voltage_05": {
        "trigger": "TEK00079.csv",
        "channel": "TEK00078.csv"
    },
    "voltage_06": {
        "trigger": "TEK00081.csv",
        "channel": "TEK00080.csv"
    },
    "voltage_07": {
        "trigger": "TEK00083.csv",
        "channel": "TEK00082.csv"
    },
    "voltage_08": {
        "trigger": "TEK00085.csv",
        "channel": "TEK00084.csv"
    },
    "voltage_09": {
        "trigger": "TEK00087.csv",
        "channel": "TEK00086.csv"
    },
    "voltage_10": {
        "trigger": "TEK00089.csv",
        "channel": "TEK00088.csv"
    },
    "voltage_11": {
        "trigger": "TEK00091.csv",
        "channel": "TEK00090.csv"
    },
    "voltage_12": {
        "trigger": "TEK00093.csv",
        "channel": "TEK00092.csv"
    },
    "voltage_13": {
        "trigger": "TEK00095.csv",
        "channel": "TEK00094.csv"
    },
    "voltage_14": {
        "trigger": "TEK00097.csv",
        "channel": "TEK00096.csv"
    },
    "voltage_15": {
        "trigger": "TEK00099.csv",
        "channel": "TEK00098.csv"
    }
}

FOLDER_PATH_7 = "./30_3_2026_vernier/"
DEVICE_SET_10 = {
    # output of through port of the first microring
    "voltage_06": {
        "trigger": "TEK00101.csv",
        "channel": "TEK00100.csv"
    },
    "voltage_15": {
        "trigger": "TEK00103.csv",
        "channel": "TEK00102.csv"
    },
}

FOLDER_PATH_8 = "./31_3_2026_vernier/test_V9_2nd_ring/"
DEVICE_SET_11 = {
    # through port of the 2nd single microring of the V9. not V1, a typo
    # temperature: 21.3-21.4 degree
    # 1360nm with 100mV * 6
    "voltage_05": {
        "trigger": "TEK00116.csv",
        "channel": "TEK00115.csv"
    },
    "voltage_10": {
        "trigger": "TEK00112.csv",
        "channel": "TEK00111.csv"
    },
    "voltage_15": {
        "trigger": "TEK00114.csv",
        "channel": "TEK00113.csv"
    },
    "voltage_20": {
        "trigger": "TEK00109.csv",
        "channel": "TEK00108.csv"
    },
    "voltage_25": {
        "trigger": "TEK00107.csv",
        "channel": "TEK00106.csv"
    },
    "voltage_30": {
        "trigger": "TEK00105.csv",
        "channel": "TEK00104.csv"
    },
}

FOLDER_PATH_9 = "./31_3_2026_vernier/test_V9_1st_ring_drop/"
DEVICE_SET_12 = {
    "voltage_10V": {
        "trigger": "TEK00049.csv",
        "channel": "TEK00048.csv"
    },
    "voltage_15V": {
        "trigger": "TEK00051.csv",
        "channel": "TEK00050.csv"
    },
    "voltage_20V": {
        "trigger": "TEK00053.csv",
        "channel": "TEK00052.csv"
    },
    "voltage_25V": {
        "trigger": "TEK00055.csv",
        "channel": "TEK00054.csv"
    },
    "voltage_30V": {
        "trigger": "TEK00057.csv",
        "channel": "TEK00056.csv"
    },
}

FOLDER_PATH_13 = "./10_04_2026_vernier/"
DEVICE_SET_13 = {
    "voltage_0V": {
        "trigger": "trigger_2.csv",
        "channel": "channel_2.csv"
    },
    "voltage_31V": {
        "trigger": "trigger_1.csv",
        "channel": "channel_1.csv"
    },
}

FOLDER_PATH_14 = "./10_04_2026_vernier/"
DEVICE_SET_14 = {
    "through": {
        "trigger": "TEK00001.csv",
        "channel": "TEK00000.csv"
    },
    "drop": {
        "trigger": "TEK00002.csv",
        "channel": "TEK00003.csv"
    },
}

FOLDER_PATH_15 = "./vernier_4_14_2026/"
DEVICE_SET_15 = {
    "single_bus": {
        "trigger": "TEK00004.csv",
        "channel": "TEK00005.csv"
    },
    "drop": {
        "trigger": "TEK00006.csv",
        "channel": "TEK00007.csv"
    },
    "V5_1st_single_ring_through": {
        "trigger": "TEK00008.csv",
        "channel": "TEK00009.csv"
    },
    "V5_1st_single_ring_drop": {
        "trigger": "TEK00010.csv",
        "channel": "TEK00011.csv"
    },
    "V5_2nd_single_ring_through": {
        "trigger": "TEK00012.csv",
        "channel": "TEK00013.csv"
    },
    "V5_2nd_single_ring_drop": {
        "trigger": "TEK00014.csv",
        "channel": "TEK00015.csv"
    },
    "V5_vernier_through": {
        "trigger": "TEK00016.csv",
        "channel": "TEK00017.csv"
    },
    "V5_vernier_1st_drop": {
        "trigger": "TEK00018.csv",
        "channel": "TEK00019.csv"
    },
    "V5_vernier_2nd_drop": {
        "trigger": "TEK00020.csv",
        "channel": "TEK00021.csv"
    },
    "V6_1st_single_ring_through": {
        "trigger": "TEK00022.csv",
        "channel": "TEK00023.csv"
    },
    "V6_1st_single_ring_drop": {
        "trigger": "TEK00024.csv",
        "channel": "TEK00025.csv"
    },
    "V6_2nd_single_ring_through": {
        "trigger": "TEK00026.csv",
        "channel": "TEK00027.csv"
    },
    "V6_2nd_single_ring_drop": {
        "trigger": "TEK00028.csv",
        "channel": "TEK00029.csv"
    },
    "V8_1st_single_ring_through": {
        "trigger": "TEK00030.csv",
        "channel": "TEK00031.csv"
    },
    "V8_1st_single_ring_through_1V":{
        "trigger": "TEK00032.csv",
        "channel": "TEK00033.csv"
    },
    "V8_1st_single_ring_through_1point5V": {
        "trigger": "TEK00034.csv",
        "channel": "TEK00035.csv"
    },
    "V8_1st_single_ring_through_2V": {
        "trigger": "TEK00036.csv",
        "channel": "TEK00037.csv"
    },
    "V8_1st_single_ring_through_2point5V": {
        "trigger": "TEK00038.csv",
        "channel": "TEK00039.csv"
    },
    "V8_1st_single_ring_through_3V": {
        "trigger": "TEK00040.csv",
        "channel": "TEK00041.csv"
    },
    "V8_vernier_through": {
        "trigger": "TEK00042.csv",
        "channel": "TEK00043.csv"
    },
    "V8_vernier_through_1point5V": {
        "trigger": "TEK00044.csv",
        "channel": "TEK00045.csv"
    },
    "V8_vernier_through_2V": {
        "trigger": "TEK00046.csv",
        "channel": "TEK00047.csv"
    },
    "V8_vernier_through_2point5V": {
        "trigger": "TEK00048.csv",
        "channel": "TEK00049.csv"
    },
    "V8_vernier_final_drop_0V": {
        "trigger": "TEK00050.csv",
        "channel": "TEK00051.csv"
    },
    "V8_vernier_final_drop_1point5V": {
        "trigger": "TEK00052.csv",
        "channel": "TEK00053.csv"
    },
    "V8_vernier_final_drop_2V": {
        "trigger": "TEK00054.csv",
        "channel": "TEK00055.csv"
    },
    "V8_vernier_final_drop_2point5V" :{
        "trigger": "TEK00056.csv",
        "channel": "TEK00057.csv"
    },

}

def case_1():
    START_INDEX = 1260
    END_INDEX = 1360
    subset = ["single_bus", "output_5"]
    LENGTH = 27000 * 2 * math.pi
    data_tensor = package_data(DEVICE_SET_2, DATA_LENGTH, FOLDER_PATH_2, subset=subset)
    print("Data tensor shape:", data_tensor.shape)
    basis = moving_average(data_tensor[0], window_size=10000)
    print("basis shape:", basis.shape)
    normalized_data_tensor = normalize_data(data_tensor, basis, dB=False)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    multi_plot(downsampled_data_tensor[1:], start_index=START_INDEX, end_index=END_INDEX)

    TESTING_ID = 1
    REVERSE = False
    testing_array = downsampled_data_tensor[TESTING_ID]
    if REVERSE:
        testing_array = np.max(testing_array) - testing_array
    fwhm, fsr, q_factor_mode, q_factor, q_factor_std = calculate_fwhm_whole(testing_array,peak_number=0, distance=3000, start_index=START_INDEX, end_index=END_INDEX, length=LENGTH)
    print("fwhm mode:", round(fwhm, 5))
    print("fsr mode:", round(fsr,5))
    print("Q factor mode:", round(q_factor_mode, 5))
    print("Q factor average:", round(q_factor, 5))
    print("Q factor std:", round(q_factor_std, 5))

# display, with single bus normalization
def case_2(subset, data_set, folder_path):
    START_INDEX = 1260
    END_INDEX = 1360
    data_tensor = package_data(data_set, DATA_LENGTH, folder_path, subset=subset)
    print("Data tensor shape:", data_tensor.shape)
    basis = moving_average(data_tensor[0], window_size=1000)
    print("basis shape:", basis.shape)
    normalized_data_tensor = normalize_data(data_tensor, basis, dB=True)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    multi_plot(downsampled_data_tensor[1:], start_index=START_INDEX, end_index=END_INDEX, legend=subset[1:])

# display, with self normalization, for all voltage levels
def case_3(subset, data_set, folder_path):
    START_INDEX = 1260
    END_INDEX = 1360
    data_tensor = package_data(data_set, DATA_LENGTH, folder_path, subset=subset)
    print("Data tensor shape:", data_tensor.shape)
    normalized_data_tensor = normalize_each_date(data_tensor, window_size=None)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)
    print("Downsampled data tensor shape:", downsampled_data_tensor.shape)
    multi_plot(downsampled_data_tensor, start_index=START_INDEX, end_index=END_INDEX, legend=subset)

# find the FWHM, FSR, and Q factor for testing array
def case_4(subset, data_set, folder_path, port = "drop"):
    START_INDEX = 1260
    END_INDEX = 1360
    data_tensor = package_data(data_set, DATA_LENGTH, folder_path, subset=subset)
    print("Data tensor shape:", data_tensor.shape)
    normalized_data_tensor = normalize_each_date(data_tensor)
    print("Normalized data tensor shape:", normalized_data_tensor.shape)
    downsampled_data_tensor = downsample_data(normalized_data_tensor, points=10)

    TESTING_ID = 0
    REVERSE = True
    testing_array = downsampled_data_tensor[TESTING_ID]
    LENGTH = 26000 * 2 * math.pi
    if REVERSE:
        testing_array = np.max(testing_array) - testing_array
    find_peak_over_spectrum(testing_array,peak_number=0, distance=3000, start_index=START_INDEX, end_index=END_INDEX, length=LENGTH, port=port)

if __name__ == "__main__":
    subset = ["V8_1st_single_ring_through"]
    #case_4(subset, DEVICE_SET_15, FOLDER_PATH_15, port="through")
    case_2(subset = ["single_bus", "V5_1st_single_ring_through"], data_set=DEVICE_SET_15, folder_path=FOLDER_PATH_15)
