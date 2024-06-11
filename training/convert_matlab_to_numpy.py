import spectrum_painting_data as sp_data

sp_data.convert_matlab_to_numpy(matlab_dir="data/matlab-test",
                                numpy_dir="data/numpy-test",
                                classes=["Z", "B", "W", "BW", "ZB", "ZW", "ZBW"],
                                snr_list=[0, 5, 10, 20, 30])
