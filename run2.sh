'''
#ens
CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-models.pickle --mdl ensemble > log2.txt
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example1 --mdl ensemble --pess 2.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example1/plot.png --data hopper-medium-v0-example1/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example2 --mdl ensemble --pess 2.1
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example2/plot.png --data hopper-medium-v0-example2/logs/log.pickle

#swag
CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-models_swag.pickle --mdl swag > log2.txt
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag1 --mdl swag --pess 2.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag1/plot.png --data hopper-medium-v0-example_swag1/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag2 --mdl swag --pess 2.1
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag2/plot.png --data hopper-medium-v0-example_swag2/logs/log.pickle

#swag_ens
CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-models_swag_ens.pickle --mdl swag_ens > log3.txt
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-example_swag_ens1 --mdl swag_ens --pess 2.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag_ens1/plot.png --data hopper-medium-v0-example_swag_ens1/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-example_swag_ens2 --mdl swag_ens --pess 2.1
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag_ens2/plot.png --data hopper-medium-v0-example_swag_ens2/logs/log.pickle


CUDA_VISIBLE_DEVICES=0 python projects/morel/learn_model.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-models.pickle --mdl ensemble > log1.txt
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag_ens.txt --output hopper-medium-v0-example_swag_ens3 --mdl swag_ens --pess 4.95 #max error 1.241
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag_ens3/plot.png --data hopper-medium-v0-example_swag_ens3/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example3 --mdl ensemble --pess 1.95 # max error 0.5749
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example3/plot.png --data hopper-medium-v0-example3/logs/log.pickle
'''

#swag original
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag3 --mdl swag --pess 3.5 # max error 1.5668
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag3/plot.png --data hopper-medium-v0-example_swag3/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag4 --mdl swag --pess 4.0 # max error 1.3709
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag4/plot.png --data hopper-medium-v0-example_swag4/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag5 --mdl swag --pess 5.0 # 1.096795
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag5/plot.png --data hopper-medium-v0-example_swag5/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag6 --mdl swag --pess 6.5 # 0.843689
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag6/plot.png --data hopper-medium-v0-example_swag6/logs/log.pickle

CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium_swag.txt --output hopper-medium-v0-example_swag6 --mdl swag --pess 6.8 # 0.5549
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example_swag7/plot.png --data hopper-medium-v0-example_swag7/logs/log.pickle


CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example4 --mdl ensemble --pess 1.97
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example4/plot.png --data hopper-medium-v0-example4/logs/log.pickle
CUDA_VISIBLE_DEVICES=0 python projects/morel/run_morel.py --config projects/morel/configs/d4rl_hopper_medium.txt --output hopper-medium-v0-example5 --mdl ensemble --pess 2.0
python mjrl/utils/plot_from_logs.py --output hopper-medium-v0-example5/plot.png --data hopper-medium-v0-example5/logs/log.pickle