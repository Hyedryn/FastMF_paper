import os
import sys
from dotenv import load_dotenv
#%%
import fastmf.generation as gen
from fastmf.reports import evaluator


if __name__ == "__main__":
    assert len(sys.argv) == 4, "Wrong number of arguments"
    dotenv_path = sys.argv[1]
    load_dotenv(dotenv_path)
    
    base_path = os.getenv('BASE_PATH')
    scheme_file = os.getenv('SCHEME_FILE')
    bvals_file = os.getenv('BVALS_FILE')
    dic_file = os.getenv('DIC_FILE')
    task_name = os.getenv("TASK_NAME")
    base_seed = int(os.getenv("BASE_SEED"))
    
    assert sys.argv[3] in ["GROUNDTRUTH", "MSMTCSD", "CSD"], "Orientation estimate can either be GROUNDTRUTH, MSMTCSD or CSD"
    
    run_id = sys.argv[2]
    orientation_estimate = sys.argv[3]
    seed = base_seed+3000+int(run_id)
    print("Seed: ", seed)
    
    
    #Synthetizer Path
    synthetizer_file = os.path.join(base_path,"synthetizer","type-standard","raw",f"type-standard_task-{task_name}_run-{run_id}_raw.pickle")
    
    # MF Generation
    genStandard = gen.Generator(synthetizer_file, base_path, orientation_estimate_sh_max_order = 12,  orientation_estimate = orientation_estimate, recompute_S0mean = False, compute_vf = False, compute_swap = False)
    genStandard.computeNNLSWeights(sparse_storage=True)
    
    