# terminal commands
ssh -Y es833@calx079.ast.cam.ac.uk
cd /data/es833 
conda activate 42env

cd /data/es833/dataforfrank/uv_tables #are the uv_tables for use by frank

e.g. /data/es833/dataforfrank/uv_tables/DR_Tau_selfcal_cont_bin60s_1chan_nopnt_nofl.npz

xdg-open # to open file 


python -m frank.fit -uv /data/es833/dataforfrank/uv_tables/DR_Tau_selfcal_cont_bin60s_1chan_nopnt_nofl.npz -p /home/es833/Proj42/params/DR_Tau_parameters.json

python -m frank.fit -uv /data/es833/modeldata/model_8hrs_a-3Mth-dust3.txt -p /home/es833/Proj42/params/model_parameters.json
# have experimented with different frank hyperparams but they make barely any difference 

time #before script for runtime

### to copy a file to/from calx079, do in terminal in laptop

# from
scp -r es833@calx079.ast.cam.ac.uk:/home/es833/Proj42/chi2_plots C:\Users\edwar\OneDrive\Downloads\Project_42

# to 
scp -r C:\Users\edwar\OneDrive\Downloads\spirals1 es833@calx079.ast.cam.ac.uk:/data/es833/spirals2

# [-r for transferring folder]


#### seems to be a problem using nomachine to extract files from gz folder so have had to use 7-zip on windows laptop and then scp