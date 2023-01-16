import numpy as np
import obspy as opy
import scipy.signal

import fnmatch
import os
import datetime
import matplotlib.pyplot as plt
from numpy import linalg as LA
from obspy.core import UTCDateTime
from collections import defaultdict
from obspy.core.stream import Stream
from obspy.signal.invsim import corn_freq_2_paz, paz_to_freq_resp
from obspy.signal.filter import envelope
from scipy.signal import find_peaks
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from scipy import signal
import scipy.fft as ft
from scipy.optimize import minimize
import arviz as az
import pymc3 as pm
import theano
import theano.tensor as thnts

from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Read data


# ----- default parameters ------ #
# remove response

 
tm_extend = 1*60*60 # hrs
paz = {'gain':1.0,
       'poles':[-22.211059 + 22.217768j,
                -22.211059 - 22.217768j],
       'sensitivity': 76.7,
       'zeros': [0j,0j]}

sensitivity = 76.7            
paz_str = corn_freq_2_paz(0.01, damp=0.707)  
paz_str['sensitivity'] = sensitivity
tmz_correction = 8*3600

default_par = {  
    'rm_timeextend'     :   tm_extend,
    'pas'               :   paz,
    'pas_str'           :   paz_str,
    'time_extend'       :   tm_extend,
    'tmz_correction'    :   tmz_correction  
}


def getdatalist(path, station, flag, starttime, endtime, cmp):
    """
    Get data list
    """
    if not path.endswith('/'):
        path = path + '/'
        
    
    data_list = []
    srckey = '%s-%s-*-%s.npz' %(station, flag, cmp)
    for file in sorted(os.listdir(path)):
        if fnmatch.fnmatch(file, srckey):   
            data_list    += [path+file]

    index = []
    for ii in range(0, len(data_list)):
        time_bg  = data_list[ii].split('-')[2]
        time_bg  = time_bg.replace(':','.')
        [year,month,days,hours,minute,second]= time_bg.split('.')
       
        date  = UTCDateTime('%s,%s,%s,%s,%s,%s'%(year,month,days, hours, minute, second))       
        if  starttime<=date and date<endtime:
            index += [ii]
                
    data_list = [data_list[ii] for ii in index]
    return data_list

def gettracelist(path, station, flag, starttime, endtime, cmp):
    """
    Get data list
    """
    if not path.endswith('/'):
        path = path + '/'
        
    
    data_list = []
    srckey = '%s-%s-*-%s' %(station, flag, cmp)
    for file in sorted(os.listdir(path)):
        if fnmatch.fnmatch(file, srckey):   
            data_list    += [path+file]

    index = []
    for ii in range(0, len(data_list)):
        time_bg  = data_list[ii].split('-')[2]
        time_bg  = time_bg.replace(':','.')
        [year,month,days,hours,minute,second]= time_bg.split('.')
       
        date  = UTCDateTime('%s,%s,%s,%s,%s,%s'%(year,month,days, hours, minute, second))       
        if  starttime<=date and date<endtime:
            index += [ii]
                
    data_list = [data_list[ii] for ii in index]
    return data_list




class DataProcessing:
    
    def __init__(self, project_par=None, processing_par=None, default_par=default_par):
        self.par  = {**project_par, **processing_par, **default_par}
        self.data = Stream()
        self.data_list = []
        self.station_list = []
        self.station = ''
        self.component = ''
        self.current_index = 0
        self.title = ''

        # Judge compliance
        for keys in ['data_input_path', 'data_output_path', 'fig_output_path'] :
            if not self.par[keys].endswith('/'):
                self.par[keys] = self.par[keys] + '/'
                
        for keys in ['data_output_path', 'fig_output_path'] :
            if not os.path.exists(self.par[keys]):
                print('creat %s' %self.par[keys])
                os.makedirs(self.par[keys])


    def getdatalist(self):
        """
        Get data list for read file list
        output: data_list
                station_list
        """
        data_list = []
        station_list = []
        
        ifall = True
        if 'proj_time_bgn' in self.par.keys() and 'proj_time_end' in self.par.keys():
            if self.par['proj_time_bgn'] < self.par['proj_time_end']:
                ifall = False

        stations = self.par['stations']
        for station in stations.keys():
            station_values = stations[station]
            for ii in range(0, len(station_values)):
                for cmp in self.par['components']:
                    srckey = '*%s*%s.sac' %(station_values[ii], cmp)
                    for file in os.listdir(self.par['data_input_path']):
                        if fnmatch.fnmatch(file, srckey):
                            data_list    += [file]
                            station_list += [station]

        if ifall == False:
            index = []
            for ii in range(0, len(data_list)):
                year  = data_list[ii].split('.')[4]
                month = data_list[ii].split('.')[5]
                days  = data_list[ii].split('.')[6]
                hours = data_list[ii].split('.')[7]        
                date  = UTCDateTime('%s,%s,%s,%s,0,0'%(year,month,days, hours))
                
                if  self.par['proj_time_bgn']<=date and date<=self.par['proj_time_end']:
                    index += [ii]

            data_list    = [ data_list[ii]    for ii in index]
            station_list = [ station_list[ii] for ii in index]
                
            
        # add path
        for ii in range(0, len(data_list)):
            data_list[ii] = self.par['data_input_path'] + data_list[ii]
        
        self.data_list = data_list
        self.station_list = station_list
        
        if self.par['verbosity']:
            print('>>> Create input data_list and station_list!')
            for ii in range(0,len(data_list)):
                print(data_list[ii])
        return self.data_list, self.station_list
    
    
    def readdata(self, data_path):
        """
        Read SAC data
        """
        self.data = Stream() 
        self.data = opy.read(data_path, debug_headers=True)
        self.current_index = self.data_list.index(data_path)
        
        
        if self.par['verbosity']:
            print(" >>> Read data from %s" %data_path)
            print(self.data)
            
    def preprocessing(self):
        """
        Preprocessing: sampling the file 
        """
        
        # set station
        self.data[0].stats.network = self.station_list[self.current_index]
        
        # set component
        self.component = self.data[0].stats.channel
        
        # correct time zone
        # local time zone   
        if self.par['local_timezone'] == True:
            self.data[0].stats.starttime = self.data[0].stats.starttime+self.par['tmz_correction']
            if self.par['verbosity']:
                print(" Pre-processing:  correct time-zone ")
                print(self.data)
  
        # set unified title
        self.title = '%s-%s-%s'%(self.station_list[self.current_index], 
                                 self.data[0].stats.starttime.strftime("%Y.%m.%d.%H:%M:%S"),
                                 self.data[0].stats.endtime.strftime("%Y.%m.%d.%H:%M:%S"))
        self.station = self.station_list[self.current_index]
        
        # downsampling from sampling_rate_raw to sampling_rate_down 
        self.data[0].stats.delta = round(self.data[0].stats.delta, 3)
        sampling_rate_raw  = self.data[0].stats.delta
        sampling_rate_down = self.par['downsampling_rate']
        factor = int(sampling_rate_down/sampling_rate_raw)

        if self.data[0].stats.delta != sampling_rate_down:
            # SmartSolo Anti-alias filter
            # 206.5 Hz @ 2ms (82.6% of Nyquist)
            self.data[0].filter("lowpass", corners=30, freq=206.5)  
            self.data[0].decimate(factor, no_filter=True, strict_length=False)
            
            if self.par['verbosity']:
                print(" Pre-processing:  downsampling ")
                print(self.data)
        """
        Remove single frequency
        
        """
        if self.par['remove_singlef'] == True:
            

            # Create/view notch filter
            samp_freq = self.par['sampleRate' ]  # Sample frequency (Hz)
            notch_freq = self.par['notch_freq'] # Frequency to be removed from signal (Hz)
            quality_factor = self.par['qality_factor']  # Quality factor ()
            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
            freq, h = signal.freqz(b_notch, a_notch, fs = samp_freq)
            y_notched = signal.filtfilt(b_notch, a_notch, self.data[0])
            
            self.data[0].data= y_notched
            print('Notch filter is applied.')
                
    def gettitle(self, flag, starttime, endtime, cmp):
        """
        Generate unified title
        """
        title = '%s-%s-%s-%s-%s'%(self.station_list[self.current_index], 
                                  flag, starttime.strftime("%Y.%m.%d.%H:%M:%S"),
                                  endtime.strftime("%Y.%m.%d.%H:%M:%S"), cmp)
        return title
    
                        
    def remove_response(self):
        """
        Remove receiver response
        
        """
        if self.par['remove_response'] == True:           
            self.data[0].simulate(paz_remove=self.par['paz'], paz_simulate=self.par['paz_str']) 
            print('Remove response ...')
    
    
        
            
    
    def high_pass(self):
        """
        High pass filter
        """
        self.data[0].filter("highpass", freq=self.par['freq_highpass'])
        if self.par['verbosity']:
            print(" High pass filter @%f Hz " %self.par['freq_highpass'])
            print(self.data)
            
    def plot_waveform(self, plot_color='blue'):
        """
        Plot wave form
        """
        fig_title = '%s-waveform.png' %self.title
        fig_path = self.par['fig_output_path'] + fig_title
        self.data[0].plot(color=plot_color, outfile = fig_path)
        
    def psd(self):
        """
        Output PSD
        """
        dt = self.data[0].stats.delta
        fs = 1/dt
        
        starttime = self.data[0].stats.starttime
        endtime   = self.data[0].stats.endtime
        win_len   = self.par['PSD_winlen']
        
        nwindow   = int((endtime-starttime)/win_len)
        
        for iid in range(0, nwindow): #nframe
            win_bgn = starttime+iid*win_len
            win_end = win_bgn+win_len
            st_slice = self.data.slice(win_bgn, win_end)
            pxx, fre = plt.psd(st_slice[0].data, Fs=fs)

            psd_title = self.gettitle(flag='PSD', starttime=win_bgn, endtime=win_end, cmp=self.component)
            path = self.par['data_output_path'] + psd_title
            self.writedata(data=pxx, ax1=fre, time=win_bgn.strftime("%Y.%m.%d.%H:%M:%S"), file_path=path, txt_hrd=psd_title)

        if self.par['verbosity']:
            print(" Generate PSD ")
            

        
    def iso_traffic(self):
        
        
        
        win_len       = self.par['winlen_events']
        sel_lfre      = self.par['sel_lfre']
        sel_hfre      = self.par['sel_hfre']
        prominence    = self.par['prominence']
        distance      = self.par['distance']
        alp           = self.par['alp']
        t_h           = self.par['t_h']
        c0            = self.par['c0']
        d_ind         = self.par['d_ind']
        f_l           = self.par['f_l']
        f_h           = self.par['f_h']
        mu            = self.par['mu']
        std_h         = self.par['std_h']
        delta_t1      = self.par['delta_t1']
        delta_t2      = self.par['delta_t2']
        nfft          = self.par['nfft']
        nperseg       = self.par['nperseg']
        noverlap      = self.par['noverlap']
        pk_pct        = self.par['pk_pct']
        
        dt = self.data[0].stats.delta
        fs = 1/(dt*1.0)
    
        starttime     = self.data[0].stats.starttime+delta_t1
        endtime       = starttime+delta_t2
        
        loc_peaks=[]
        
        nframe = int((endtime- starttime)/win_len)
        for iid in range(0,nframe):
            win_bgn = starttime+iid*win_len
            win_end = win_bgn+win_len
            st_slice = self.data.slice(win_bgn, win_end)        
            ax_f, ax_t, Sxx = scipy.signal.spectrogram(st_slice[0].data, fs,mode='magnitude', scaling='spectrum',
                                                       nfft=nfft,nperseg=nperseg,noverlap=noverlap)
            
#             fig, ax = plt.subplots(figsize=(16,8))
#             im=ax.pcolormesh(ax_t, ax_f, Sxx,cmap='jet',vmin=0,vmax=0.05)
#             ax.set_ylabel('Frequency (Hz)', fontsize=16)

#             ax.set_xlabel('Time (s)', fontsize=16)
#             cax = fig.add_axes([0.27, 0.02, 0.5, 0.03])
#             cbar=fig.colorbar(im, ax=ax,cax=cax,orientation='horizontal')
#             cbar.ax.tick_params(labelsize=12)
#             cbar.set_label('Magnitude Spectrum',fontsize=12)
            
            
        
            fre_mask = (sel_lfre<ax_f) & (ax_f<sel_hfre)

            sel_ax_f = ax_f[fre_mask]
            sel_sxx  = Sxx[fre_mask, :]
            sel_sxx_scale = (sel_sxx.T*(100/ax_f[fre_mask])).T
#             sel_sxx_scale = sel_sxx
            sel_sxx_sum = np.sum(sel_sxx_scale, axis=0)

            # find peaks
            sel_sxx_sum = sel_sxx_sum[~np.isnan(sel_sxx_sum)]
            peaks, _ = find_peaks(sel_sxx_sum, prominence=prominence, distance=distance)

        peak_title = self.gettitle(flag='peaks', starttime=win_bgn, endtime=win_end, cmp=self.component)
        path = self.par['data_output_path'] + peak_title
        self.writedata(data=len(peaks), ax1=np.array([]), 
                       time=win_bgn.strftime("%Y-%m-%d"), file_path=path, txt_hrd=peak_title)

    
    

    def writedata(self, data=np.array([]), ax1=np.array([]), time='', file_path='./', txt_hrd=''):
        """
        Write data as npy
        """
        np.savez(file_path, data=data, ax1=ax1, time=time, txt_hrd=txt_hrd)
        
        
    def cleardata(self):
        """
        Clear data
        """
        self.data.clear()









