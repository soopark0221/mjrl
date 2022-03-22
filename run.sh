'''
# ens swag 
CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-models_swag_ens.pickle --mdl swag_ens --param_dict_fname param_dict_ens
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-example_swag_ens2 --mdl swag_ens --pess 1.2
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag_ens2/plot.png --data hopper-medium-v0-example_swag_ens2/logs/log.pickle
## swag pess test
CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-models_swag.pickle --mdl swag

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag1 --mdl swag --pess 1.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag1/plot.png --data hopper-medium-v0-example_swag1/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag2 --mdl swag --pess 1.5

python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag2/plot.png --data hopper-medium-v0-example_swag2/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag3 --mdl swag --pess 2.0

python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag3/plot.png --data hopper-medium-v0-example_swag3/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag4 --mdl swag --pess 4.0

python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag4/plot.png --data hopper-medium-v0-example_swag4/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag5 --mdl swag --pess 1.0

python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag5/plot.png --data hopper-medium-v0-example_swag5/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag6 --mdl swag --pess 1.2
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag6/plot.png --data hopper-medium-v0-example_swag6/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag7 --mdl swag --pess 1.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag7/plot.png --data hopper-medium-v0-example_swag7/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag8 --mdl swag --pess 0.9
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag8/plot.png --data hopper-medium-v0-example_swag8/logs/log.pickle

##ens
CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-models.pickle --mdl ensemble
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example1 --mdl ensemble 
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example1/plot.png --data hopper-medium-v0-example1/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example2 --mdl ensemble --pess 2.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example2/plot.png --data hopper-medium-v0-example2/logs/log.pickle
# swag epoch test
# epoch 500, swag start 300, k 20

'''
# with no pess
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag1 --mdl swag
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag1/plot.png --data hopper-medium-v0-example_swag1/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-models_swag_ens.pickle --mdl swag_ens --param_dict_fname param_dict_ens > log2.txt
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-example_swag_ens1 --mdl swag_ens --param_dict_fname param_dict_ens 
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag_ens1/plot.png --data hopper-medium-v0-example_swag_ens1/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-models.pickle --mdl ensemble > log3.txt
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example1 --mdl ensemble
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example1/plot.png --data hopper-medium-v0-example1/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_multiswag.txt --output hopper-medium-v0-example_multiswag1 --mdl multiswag --param_dict_fname param_dict_multiswag
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_multiswag1/plot.png --data hopper-medium-v0-example_multiswag1/logs/log.pickle

# interval swag
# no diag swag
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag2 --mdl swag
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag2/plot.png --data hopper-medium-v0-example_swag2/logs/log.pickle
