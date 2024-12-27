/data/es833/dataforfrank/uv_tables # contain the uv_tables for use by frank

python -m frank.fit -uv /data/es833/dataforfrank/uv_tables/DR_Tau_selfcal_cont_bin60s_1chan_nopnt_nofl.npz -p /home/es833/Proj42/params/DR_Tau_parameters.json

python -m frank.fit -uv /data/es833/modeldata/model_8hrs_a-3Mth-dust3.txt -p /home/es833/Proj42/params/model_parameters.json
# have experimented with different frank hyperparams but they make barely any difference 
