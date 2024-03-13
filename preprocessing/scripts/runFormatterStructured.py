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
    
    run_ids = ["0"]
    seed = base_seed+9999
    print("Synthetizer seed: ", seed)

    print("Initializing formatter")
    formatterStructured = gen.DataFormatter(base_path, task_name=task_name, session_id=session_id, reference_dic_path=dic_file,
                                          run_ids=run_ids, type_="structured", set_size=[0, 0, 75000])

    bar = tqdm(total=5, desc="Formatting data")
    print("Formatting NNLS target")
    formatterStructured.genNNLSTarget(min_max_scaling=True, orientation_estimate="CSD")
    bar.update()
    #formatterStructured.genNNLSTarget(min_max_scaling=True, orientation_estimate="MSMTCSD")
    #bar.update()
    formatterStructured.genNNLSTarget(min_max_scaling=True, orientation_estimate="GROUNDTRUTH")
    bar.update()

    print("Formatting NNLS input")
    formatterStructured.genNNLSInput(normalization="SumToOne", orientation_estimate="CSD")
    bar.update()
    #formatterStructured.genNNLSInput(normalization="SumToOne", orientation_estimate="MSMTCSD")
    #bar.update()
    #formatterStructured.genNNLSInput(normalization="SumToOne", orientation_estimate="GROUNDTRUTH")
    #bar.update()

    print("Formatting spherical harmonic target")
    formatterStructured.genSphericalHarmonicTarget(min_max_scaling=True)
    bar.update()
    print("Formatting spherical harmonic input")
    formatterStructured.genSphericalHarmonicInput()
    bar.update()

