import os
import sys
from dotenv import load_dotenv

#%%
import fastmf.generation as gen
from fastmf.reports import evaluator


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Wrong number of arguments"
    dotenv_path = sys.argv[1]
    load_dotenv(dotenv_path)
    
    base_path = os.getenv('BASE_PATH')
    scheme_file = os.getenv('SCHEME_FILE')
    bvals_file = os.getenv('BVALS_FILE')
    dic_file = os.getenv('DIC_FILE')
    task_name = os.getenv("TASK_NAME")
    base_seed = int(os.getenv("BASE_SEED"))
    
    run_id= sys.argv[2]
    num_samples = 100000
    seed = base_seed + 1000 + int(run_id)
    print("Synthetizer seed: ",seed)
    
    synth_FixRadDist = gen.Synthetizer(scheme_file, bvals_file, dic_file, task_name=task_name, include_csf = False)
    
    synthStandard = synth_FixRadDist.generateStandardSet(num_samples, run_id=run_id, SNR_min=20, SNR_max=100, 
                            SNR_dist='uniform', nu_min=0.05, nu_csf_max = 0, crossangle_min=30, nu1_dominant=True, random_seed=seed)
    synthStandard.save(base_path)

