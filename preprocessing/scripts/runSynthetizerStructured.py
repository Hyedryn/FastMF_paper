import os
import sys
from dotenv import load_dotenv
#%%
import fastmf.generation as gen
from fastmf.reports import evaluator


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
    
    seed = base_seed + 111
    print("Synthetizer seed: ",seed)
    
    synth_HCP_FixRadDist = gen.Synthetizer(scheme_file, bvals_file, dic_file, task_name=task_name, include_csf = False)
    
    synthStructured = synth_HCP_FixRadDist.generateStructuredSet(nu1_values=[0.5, 0.6, 0.7, 0.8, 0.9], nucsf_values=[], include_csf = False, SNR_values=[30, 50, 100], repetition=5000, run_id=0, crossangle_min=30, random_seed=seed)
    
    synthStructured.save(base_path)

