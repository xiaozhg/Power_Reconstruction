from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciio
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianDropout, BatchNormalization
#import complexnn

def prep_data(power,pspec,power_GT,phase_GT,use_phase=0,ntrain=-1):

    x_train=np.concatenate((power,pspec),axis=1)
    phase_GT_max = DefZeroPhase(phase_GT,power_GT)
    power_norm=np.max(power_GT,axis=1)
    power_GT = power_GT/power_norm[:,np.newaxis]*np.pi
    power = power/power_norm[:,np.newaxis]*np.pi
    phase_GT_clean = RemoveLowPowPhase(phase_GT_max,power_GT)

    if use_phase:
        y_train=np.concatenate((power_GT,phase_GT_clean),axis=1)
    else:
        ytemp = np.concatenate((power_GT,phase_GT_clean),axis=1)
        y_train = Phase2Comp(ytemp,method='separate')

    if ntrain>0:
        train_ind=np.random.choice(x_train.shape[0],ntrain,replace=False)
        x_train=x_train[train_ind,:]
        y_train=y_train[train_ind,:]

    return x_train, y_train

def prep_pred(power_pred,phase_pred):

    power_norm=np.max(power_pred,axis=1)
    power_pred_norm = power_pred/power_norm[:,np.newaxis]*np.pi
    phase_pred_clean = RemoveLowPowPhase(phase_pred,power_pred)
    pred = np.concatenate((power_pred_norm,phase_pred_clean),axis=1)

    return(pred)


def RemoveLowPowPhase(phase,power,min_frac=0.01):

    nex=phase.shape[0]
    clean_phase=phase.copy()
    for j in range(nex):
        minpow = min_frac*np.max(power[j,:])
        pzone = np.where(power[j,:]>minpow)[0]
        pstart=pzone[0]; pend=pzone[-1]
        clean_phase[j,:pstart]=phase[j,pstart]
        clean_phase[j,pend:]=phase[j,pend]

    return(clean_phase)


def DefZeroPhase(phase,power):

    n=power.shape[0]
    clean_phase = phase.copy()
    for j in range(n):
        offset = phase[j,power[j,:]==np.max(power[j,:])][0]
        clean_phase[j,:] = phase[j,:]-offset
        
    return(clean_phase)

def Comp2Phase(y,removeLowPowPhase=0):

    nex=y.shape[0]; npts=np.int(y.shape[1]/2)
    
    c=np.zeros([nex,npts]).astype('complex')
    c[:,:] = y[:,:npts] + 1j*y[:,npts:]
    
    c_power = np.abs(c)**2
    c_phase = np.angle(c)
    
    c_phase = RemoveLowPowPhase(c_phase,c_power)   #here we should also DefZeroPhase(c_phase, c_power)?
    
    y2 = np.concatenate((c_power,c_phase),axis=1)
    
    return(y2)
    
def Phase2Comp(y,method='separate'):

    nex=y.shape[0]; npts=np.int(y.shape[1]/2)
    
    y_amp=np.sqrt(y[:,:npts]); y_phase=y[:,npts:]
    y_c=y_amp*np.exp(1j*y_phase)
    
    if method is 'separate':
        y2=np.concatenate((np.real(y_c),np.imag(y_c)),axis=1)
    else:
        y2=np.zeros([nex,2*npts])
        for j in range(npts):
            y2[:,2*j]=np.real(y_c[:,j])
            y2[:,2*j+1]=np.imag(y_c[:,j])
    
    return(y2)

def power_to_spec(A,phi,t):

    f = A*np.exp(1j*phi)
    spec = np.fft.fft(p)

def calc_err(A,B):

    npts = np.int(A.shape[1]/2)
    err_power = np.sum((A[:,:npts]-B[:,:npts])**2,axis=1)
    #err_phase = calc_phase_err(A[:,npts:],B[:,npts:],power=A[:,:npts])
    err_phase = calc_sine_err(A,B)
        
    return np.mean(err_power),np.mean(err_phase), np.median(err_power), np.median(err_phase)


def calc_phase_err(A,B,power=None,phase_wrap=True,power_scale=True):
    
    diff = A-B  #cos?
    if phase_wrap:
        while np.max(diff) > np.pi:
            diff[diff>np.pi] = diff[diff>np.pi] - 2*np.pi
        while np.min(diff) < -np.pi:
            diff[diff<-np.pi] = diff[diff<-np.pi] + 2*np.pi

    if power_scale:
        err = np.sum((diff*power)**2,axis=1)
    else:
        rr = np.sum(diff**2,axis=1)


    return err

def calc_sine_err(A,B):

    npts = np.int(A.shape[1]/2)
    power=A[:,:npts];
    Aphase=A[:,npts:]; Bphase=B[:,npts:]
    sine_err = np.sum(np.sin(Aphase-Bphase)**2*power) #use this one for consistency
    return sine_err


def remove_phase_offset(phase,power):

    [m,n]=phase.shape
    new_phase=np.zeros(phase.shape)
    for j in range(m):
        k = np.where(power[j,:]==np.max(power[j,:]))[0][0]
        new_phase[j,:] = phase[j,:] - phase[j,k]

    return(new_phase)


def eval_cut(err,err_pred, thresh=80, err_percent=80):

    cut_thresh = np.percentile(err_pred,thresh)
    new_err = err[err_pred<cut_thresh]
    new_err_pred = err_pred[err_pred<cut_thresh]
    max_err = np.max(new_err)/np.max(err)
    med_err = np.median(new_err)/np.median(err)
    err_percent_err = np.percentile(new_err,err_percent)/np.percentile(err,err_percent)

    print('Max error: %0.2f, Median error: %0.2f, %d percentile error: %0.2f'
          % (max_err,med_err,err_percent,err_percent_err))


def plot_pred(pred,y,savename=None,mytitle=None,linewidth=2):
    
    #npts=np.int(y.shape[1]/2)
    npts=y.shape[0]
    t=np.arange(0,npts)


    plt.figure()
    plt.plot(t,y,'-',t,pred,'--',linewidth=linewidth);
    plt.legend(('ground truth','prediction'),loc='best')
    if mytitle is not None:
        plt.title(mytitle)
    if savename is not None:
        plt.savefig(savename)
    plt.show()


def save_model(model,savefile_prefix, verbose=0, overwrite='yes'):

    modelfile = savefile_prefix + '_model.json'
    weightsfile = savefile_prefix + '_weights.h5'

    if overwrite == 'ask':
        y_or_n=raw_input('overwrite file "' + modelfile + '" (y or n)?')
    else:
        y_or_n='y'

    if y_or_n=='y':
        json_string = model.to_json()
        with open(modelfile, 'w') as o:
            o.write(json_string)
        
        model.save_weights(weightsfile)

        if verbose:
            print('Model saved in file: %s'%modelfile)
            print('Weights saved in file: %s'%weightsfile)

    return modelfile, weightsfile

def recon_model(num_feat,num_label=None,num_neur=256/4,f_drop=0.2,f_drop2=0.2):
    # create model
    if num_label is None:
        num_label=num_feat

    model = Sequential()
    #model.add(GaussianDropout(0.02,input_shape=(num_feat,)))
    #model.add(Dense(850, input_dim=900, kernel_initializer='normal', activation='linear'))
    model.add(Dense(num_neur*1, input_dim=num_feat, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(f_drop))
    model.add(Dense(num_neur*1, activation='relu'))
    model.add(Dropout(f_drop))
    model.add(Dense(num_neur*1, activation='relu'))
    model.add(Dropout(f_drop))
    model.add(Dense(num_neur*1, activation='relu'))
    model.add(Dropout(f_drop))
    #model.add(Dense(num_neur*1, activation='relu'))
    #model.add(Dropout(f_drop))
    model.add(Dense(num_label, activation='linear'))
            #sgd = SGD(lr=l, decay=1e-8, momentum=0.9, nesterov=True)
            
    return model



def recon_model_complexnn(num_feat,num_label=None,num_neur=256,f_drop=0.2,f_drop2=0.2):
    # create model
    if num_label is None:
        num_label=num_feat

    num_label=np.int(num_label/2)
    
    model = Sequential()
    #model.add(GaussianDropout(0.02,input_shape=(num_feat,)))
    #model.add(Dense(850, input_dim=900, kernel_initializer='normal', activation='linear'))
    model.add(Dense(num_neur, input_dim=num_feat, kernel_initializer='uniform', activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dropout(f_drop))
    model.add(complexnn.dense.ComplexDense(np.int(num_neur), activation='relu'))   # the elements are all real?
#    model.add(BatchNormalization())
    model.add(Dropout(f_drop))
    model.add(complexnn.dense.ComplexDense(np.int(num_neur), activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dropout(f_drop))
    model.add(complexnn.dense.ComplexDense(np.int(num_neur), activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dropout(f_drop))
    
    # just for complex3
#    model.add(complexnn.dense.ComplexDense(num_neur, activation='relu'))
#    model.add(Dropout(f_drop))
    
    #model.add(Dense(num_neur*1, activation='relu'))
    #model.add(Dropout(f_drop))
    model.add(complexnn.dense.ComplexDense(np.int(num_label), activation='linear'))
            #sgd = SGD(lr=l, decay=1e-8, momentum=0.9, nesterov=True)
            
    return model


