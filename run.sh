# python train.py -s ego_exo_data/all_saves/gopro/georgiatech_covid_03_4 -m output/ego_exo/all_saves/gopro/georgiatech_covid_03_4 --eval
# python train.py -s ego_exo_data/all_saves/camera-rgb/georgiatech_covid_03_4 -m output/ego_exo/all_saves/camera-rgb/georgiatech_covid_03_4 --eval

python train.py -s ego_exo_data/all_saves/gopro/iiith_cooking_58_2 -m output/ego_exo/all_saves/gopro/iiith_cooking_58_2 --eval
python train.py -s ego_exo_data/all_saves/camera-rgb/iiith_cooking_58_2 -m output/ego_exo/all_saves/camera-rgb/iiith_cooking_58_2 --eval

python train.py -s ego_exo_data/all_saves/gopro/unc_basketball_03-31-23_01_17 -m output/ego_exo/all_saves/gopro/unc_basketball_03-31-23_01_17 --eval
python train.py -s ego_exo_data/all_saves/camera-rgb/unc_basketball_03-31-23_01_17 -m output/ego_exo/all_saves/camera-rgb/unc_basketball_03-31-23_01_17 --eval

# python render.py --mode render -m output/ego_exo/gopro/georgiatech_covid_03_4
# python render.py --mode render -m output/ego_exo/camera-rgb/georgiatech_covid_03_4

# python render.py --mode render -m output/ego_exo/gopro/iiith_cooking_58_2
# python render.py --mode render -m output/ego_exo/camera-rgb/iiith_cooking_58_2

# python render.py --mode render -m output/ego_exo/gopro/unc_basketball_03-31-23_01_17
# python render.py --mode render -m output/ego_exo/camera-rgb/unc_basketball_03-31-23_01_17