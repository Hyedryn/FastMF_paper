import os
import sys
import numpy as np
import fastmf.generation as gen
from tqdm import tqdm
from dotenv import load_dotenv

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Wrong number of arguments"
    dotenv_path = sys.argv[1]
    load_dotenv(dotenv_path)
    
    base_path = os.getenv('BASE_PATH')
    scheme_file = os.getenv('SCHEME_FILE')
    bvals_file = os.getenv('BVALS_FILE')
    dic_file = os.getenv('DIC_FILE')
    task_name = os.getenv("TASK_NAME")
    base_seed = int(os.getenv("BASE_SEED"))
    session_id = os.getenv("SESSION_ID")
    
    num_samples = 100000
    run_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
               "20", "21", "22"]
    seed = base_seed+9999
    print("Synthetizer seed: ", seed)

    print("Initializing formatter")
    formatterStandard = gen.DataFormatter(base_path, task_name=task_name, session_id=session_id, reference_dic_path=dic_file,
                                          run_ids=run_ids, type_="standard", set_size=[2000000, 100000, 100000])

    bar = tqdm(total=5, desc="Formatting data")
    print("Formatting NNLS target")
    formatterStandard.genNNLSTarget(min_max_scaling=True, orientation_estimate="CSD")
    bar.update()
    #formatterStandard.genNNLSTarget(min_max_scaling=True, orientation_estimate="MSMTCSD")
    #bar.update()
    formatterStandard.genNNLSTarget(min_max_scaling=True, orientation_estimate="GROUNDTRUTH")
    bar.update()

    print("Formatting NNLS input")
    formatterStandard.genNNLSInput(normalization="SumToOne", orientation_estimate="CSD")
    bar.update()
    #formatterStandard.genNNLSInput(normalization="SumToOne", orientation_estimate="MSMTCSD")
    #bar.update()
    #formatterStandard.genNNLSInput(normalization="SumToOne", orientation_estimate="GROUNDTRUTH")
    #bar.update()

    print("Formatting spherical harmonic target")
    formatterStandard.genSphericalHarmonicTarget(min_max_scaling=True)
    bar.update()
    print("Formatting spherical harmonic input")
    formatterStandard.genSphericalHarmonicInput()
    bar.update()
