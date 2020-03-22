from importlib import reload
#import complexnn
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciio
import recon_complex_utils as rcu
import pandas

import keras
from keras.models import load_model
#from keras.layers import Dense, Dropout, GaussianDropout
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
#from keras.optimizers import SGD
from keras.models import model_from_json
#from sklearn.cross_validation import train_test_split
from sklearn import metrics
import random
import time
import os

plotdir='/Users/xiao/Desktop/Complexnn/deep_complex_networks/Output'
datadir='/Users/xiao/Desktop/Complexnn/deep_complex_networks/Dataset'
# parameters
psanadir = '/reg/data/ana04/amo/amox23616/scratch/generateSASE'
modeldir = '/Users/xiao/Desktop/Complexnn/deep_complex_networks/Model'
if os.getcwd() == psanadir:
    model_path = psanadir
    data_path='/reg/d/psdm/AMO/amox23616/scratch/simulations/FEL_1D_Code'
    filename='sase_bk_120k.mat'
else:
    model_path = modeldir
    #data_path='/Users/dratner/Dropbox/ResearchTopics/GhostImaging/Data/sase_sim'
    data_path=datadir
    save_path=plotdir
    #filename='sase_bk_1k.mat'



#-----------------------------
# load variables
#-----------------------------

#power_train_file = 'power_train.csv'
#pspec_train_file = 'pspec_train.csv'
#power_GT_train_file = 'groundtruth_power_train.csv'
#phase_GT_train_file = 'groundtruth_phase_train.csv'

power_train_file = 'Inputs/power_train.csv'
pspec_train_file = 'Inputs/pspec_train.csv'
power_GT_train_file = 'Labels/groundtruth_power_train.csv'
phase_GT_train_file = 'Labels/groundtruth_phase_train.csv'
XZpred_power_train_file = 'Predictions/power_prediction_train.csv'
XZpred_phase_train_file = 'Predictions/phase_prediction_train.csv'

power_test_file = 'Inputs/power_test.csv'
pspec_test_file = 'Inputs/pspec_test.csv'
power_GT_test_file = 'Labels/groundtruth_power_test.csv'
phase_GT_test_file = 'Labels/groundtruth_phase_test.csv'
XZpred_power_test_file = 'Predictions/power_prediction_test.csv'
XZpred_phase_test_file = 'Predictions/phase_prediction_test.csv'


spectral_coord_file =  'Coordinates/spectral_coordinates.csv'
time_coord_file =  'Coordinates/time_coordinates.csv'
#time_for_pred_file =  'time_for_prediction.csv'


#power_feats_train = pandas.read_csv(os.path.join(datadir,power_train_file)).values
#pspec_feats_train = pandas.read_csv(os.path.join(datadir,pspec_train_file)).values
#power_GT_train = pandas.read_csv(os.path.join(datadir,power_GT_train_file)).values
#phase_GT_train = pandas.read_csv(os.path.join(datadir,phase_GT_train_file)).values

power_feats_train = pandas.read_csv(os.path.join(datadir,power_train_file)).values
pspec_feats_train = pandas.read_csv(os.path.join(datadir,pspec_train_file)).values
power_GT_train = pandas.read_csv(os.path.join(datadir,power_GT_train_file)).values
phase_GT_train = pandas.read_csv(os.path.join(datadir,phase_GT_train_file)).values
XZpred_power_train = pandas.read_csv(os.path.join(datadir,XZpred_power_train_file)).values
XZpred_phase_train = pandas.read_csv(os.path.join(datadir,XZpred_phase_train_file)).values

power_feats_test = pandas.read_csv(os.path.join(datadir,power_test_file)).values
pspec_feats_test = pandas.read_csv(os.path.join(datadir,pspec_test_file)).values
power_GT_test = pandas.read_csv(os.path.join(datadir,power_GT_test_file)).values
phase_GT_test = pandas.read_csv(os.path.join(datadir,phase_GT_test_file)).values
XZpred_power_test = pandas.read_csv(os.path.join(datadir,XZpred_power_test_file)).values
XZpred_phase_test = pandas.read_csv(os.path.join(datadir,XZpred_phase_test_file)).values


freq_feat = pandas.read_csv(os.path.join(datadir,spectral_coord_file)).values
t_feat = pandas.read_csv(os.path.join(datadir,time_coord_file)).values
#t_pred_orig = pandas.read_csv(os.path.join(datadir,time_for_pred_file)).values

## fix time shape
#t_pred = t_pred_orig[:,0];
#dt=t_pred[1]-t_pred[0]
#t_pred=t_pred-t_pred[0]
#t_pred=np.append(t_pred,t_pred[-1]+dt)

# Flag to either use complex network and field labels (use_complex=1) or real network and phase labels (use_phase=1).  If neither, uses real network and field labels.
use_complex=0
use_phase=0

if use_complex:
    savefile_prefix = os.path.join(model_path,'field_complex2')    # for saving models as you run
    reload_prefix = os.path.join(model_path,'field_complex2') # prefix for reloading models/weights, e.g. "temp" for "temp_model.json"
elif use_phase:
    savefile_prefix = os.path.join(model_path,'temp_phase')    # for saving models as you run
    reload_prefix = os.path.join(model_path,'temp_phase') # prefix for reloading models/weights, e.g. "temp" for "temp_model.json"
else:
    savefile_prefix = os.path.join(model_path,'field1')    # for saving models as you run
    reload_prefix = os.path.join(model_path,'field1') # prefix for reloading models/weights, e.g. "temp" for "temp_model.json"


# training parameters
epochs=2000
epoch_per_chunk=10
batch_size=5000
lr=3e-4
dropout=.025

reload_model = False
reload_weights = True
retrain = False
ntrain=-1       # reduce number of training examples


# Remove phase information as check
#pspec_feats_train=np.zeros(pspec_feats_train.shape)
#pspec_feats_test=np.zeros(pspec_feats_test.shape)
#phase_GT_train=np.zeros(phase_GT_train.shape)
#phase_GT_test=np.zeros(phase_GT_test.shape)

# Load and split data
print('Loading data...')
x_train, y_train = rcu.prep_data(power_feats_train,pspec_feats_train,
                        power_GT_train,phase_GT_train,use_phase=use_phase,ntrain=ntrain)
x_test, y_test = rcu.prep_data(power_feats_test,pspec_feats_test,
                        power_GT_test,phase_GT_test,use_phase=use_phase)
                        
XZpred_train = rcu.prep_pred(XZpred_power_train,XZpred_phase_train)
XZpred_test = rcu.prep_pred(XZpred_power_test,XZpred_phase_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)



#-------------------------------------
# model parameters
num_feat=x_train.shape[1]
num_label=y_train.shape[1]
num_neur=int(256*2)
if not use_complex:
    num_neur=num_neur*2
num_t = np.int(num_label/2)


if reload_model:
    print("reloading model... ")
    
    modelfile = reload_prefix + '_model.json'
    weightsfile = reload_prefix + '_weights.h5'
    model = model_from_json(open(modelfile).read())
    model.load_weights(weightsfile)

else:
    if use_complex:
        model = rcu.recon_model_complexnn(num_feat=num_feat,num_label=num_label,
                                num_neur=num_neur,f_drop=dropout)
    else:
        model = rcu.recon_model(num_feat=num_feat,num_label=num_label,
                                num_neur=num_neur,f_drop=dropout)
    if reload_weights:
        print("reloading model weights.. ")
        weightsfile = reload_prefix + '_weights.h5'
        model.load_weights(weightsfile)

#print("\tDone.")


# Optimizers and metrics
#Ada=keras.optimizers.Adagrad(lr=lr, epsilon=1e-12, decay=0.000) #0.002 for relu #5,1e-10,0 originally
Adam=keras.optimizers.Adam(lr=lr)
#RMS=keras.optimizers.RMSprop(lr=2, rho=0.9, epsilon=1e-10, decay=0.15)
model.compile(loss='mean_squared_error', optimizer=Adam,metrics=['mse'])
#model.compile(loss='mean_absolute_error', optimizer=Adam,metrics=['mae'])

#estimator = KerasRegressor(build_fn=gu.model, epochs=500, batch_size=75, verbose=1)
#history=estimator.fit(x_train,y_train)
n_chunk=np.int(epochs/epoch_per_chunk)
t0=time.time()
test_pred_mse=np.zeros(n_chunk)
train_pred_mse=np.zeros(n_chunk)
epoch_count=np.zeros(n_chunk)
if retrain:
    print('Training model...')
    for e in range(n_chunk):
        model.fit(x_train, y_train, epochs=epoch_per_chunk, batch_size=batch_size, verbose=0)
        test_pred = model.predict(x_test)
        train_pred = model.predict(x_train)
        test_pred_mse[e]=metrics.mean_squared_error(test_pred,y_test)
        train_pred_mse[e]=metrics.mean_squared_error(train_pred,y_train)
        epoch_count[e]=(e+1)*epoch_per_chunk
        t_e=time.time()-t0
        print('%d/%d epochs in %d seconds, train error: %0.4f, test error: %0.4f' % (epoch_count[e],epochs,t_e,100*train_pred_mse[e],100*test_pred_mse[e]))
        rcu.save_model(model,savefile_prefix, verbose=0, overwrite='yes')


#-----------------------------
# post processing
#-----------------------------

noise_level=0.0
noise=1+noise_level*np.random.randn(x_test.shape[0],x_test.shape[1])
pred_test = model.predict(x_test*noise)
#pred_train = model.predict(x_train*noise)

plt.figure()
j=1; plt.plot(x_test[j,:]); plt.plot(x_test[j,:]*noise[j])

if not use_phase:
    pred_plot = rcu.Comp2Phase(pred_test,removeLowPowPhase=1)
    y_plot = rcu.Comp2Phase(y_test,removeLowPowPhase=1)
else:
    pred_plot=pred_test
    y_plot=y_test

meanerr_power,meanerr_phase,medianerr_power,medianerr_phase=rcu.calc_err(y_plot,pred_plot)
print('New prediction errors (mean/median): power: %0.3f, %0.3f phase: %0.3f, %0.3f' %
            (meanerr_power,medianerr_power,meanerr_phase,medianerr_phase))

XZmeanerr_power,XZmeanerr_phase,XZmedianerr_power,XZmedianerr_phase=rcu.calc_err(y_plot,XZpred_test)
print('Old Prediction errors (mean/median): power: %0.3f, %0.3f phase: %0.3f, %0.3f' %
            (XZmeanerr_power,XZmedianerr_power,XZmeanerr_phase,XZmedianerr_phase))


for j in np.arange(20,30):
    predj=pred_plot[j,:];
    #predj=XZpred_test[j,:];
    yj=y_plot[j,:]
    rcu.plot_pred(predj,yj)
    plt.pause(0.5)


