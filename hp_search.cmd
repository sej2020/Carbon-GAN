@REM lr 1e-3
@REM python -m src.actions.train_GAN simple --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v1
@REM python -m src.actions.train_GAN simple --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name AUS-coal-v1
@REM @REM lr 5e-4
@REM python -m src.actions.train_GAN simple --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 5e-4 --lr_D 5e-4 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v2
@REM python -m src.actions.train_GAN simple --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 5e-4 --lr_D 5e-4 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name AUS-coal-v2
@REM @REM lr 1e-4
@REM python -m src.actions.train_GAN simple --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-4 --lr_D 1e-4 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v3
@REM python -m src.actions.train_GAN simple --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-4 --lr_D 1e-4 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name AUS-coal-v3
@REM noisy input = 0.5
@REM python -m src.actions.train_GAN simple --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --noisy_input --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v4
@REM python -m src.actions.train_GAN simple --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --noisy_input --disable_tqdm --n_jobs 7 --run_name AUS-coal-v4
@REM @REM one-sided label smoothing
@REM python -m src.actions.train_GAN simple --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --label_smoothing --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v5
@REM python -m src.actions.train_GAN simple --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --label_smoothing --disable_tqdm --n_jobs 7 --run_name AUS-coal-v5

@REM if either did not degrade performance, we could try increasing noisy input coefficient or do two sided aggressive label smoothing

@REM @REM ni, ls, or none with dropout D = [0.2, 0.6]
@REM python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v6
@REM python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name AUS-coal-v6
@REM @REM ni, ls, or none with dropout D = [0.2, 0.75]
@REM python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.75 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v7
@REM python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.75 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name AUS-coal-v7
@REM @REM ni, ls, or none with dropout D = [0.2, 0.9]
@REM python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.9 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v8
@REM python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.9 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --disable_tqdm --n_jobs 7 --run_name AUS-coal-v8

@REM best combination thus far + supervised loss and eta = 0.1
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.1 --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v9
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.1 --disable_tqdm --n_jobs 7 --run_name AUS-coal-v9
@REM eta = 
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.2 --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v10
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.2 --disable_tqdm --n_jobs 7 --run_name AUS-coal-v10
@REM eta = 
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.4 --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v11
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.4 --disable_tqdm --n_jobs 7 --run_name AUS-coal-v11
@REM eta = 
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region CISO --elec_source nat_gas --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.8 --disable_tqdm --n_jobs 7 --run_name CISO-nat_gas-v12
python -m src.actions.train_GAN simple --dropout_D_in 0.2 --dropout_D_hid 0.6 --region AUS_QLD --elec_source coal --n_epochs 1000 --batch_size 1024 --lr_Gs 1e-3 --lr_D 1e-3 --lr_scheduler adaptive --sup_loss --eta 0.8 --disable_tqdm --n_jobs 7 --run_name AUS-coal-v12