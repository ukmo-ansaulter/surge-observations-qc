# visualize monthly netCDF residual and tide data

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]

d = nc.Dataset(fname)

stations = nc.chartostring(d.variables['station_name'][:])
times = nc.num2date(d.variables['time'][:], units=d.variables['time'].units)

for i, station in enumerate(stations):

    best = d.variables['zos_residual_observed'][i,:]
    ch1  = d.variables['zos_residual_channel1'][i,:]
    ch2  = d.variables['zos_residual_channel2'][i,:]
    mod  = d.variables['zos_residual_model'][i,:]
    flagch1  = d.variables['zos_residual_channel1_status_flag'][i,:]
    flagch2  = d.variables['zos_residual_channel2_status_flag'][i,:]
    ch1 = np.ma.masked_where(flagch1==0, ch1)    
    ch2 = np.ma.masked_where(flagch2==0, ch2)    

    edst = d.variables['zos_tide_edserplo'][i,:]
    tskt = d.variables['zos_tide_task'][i,:]
    modt = d.variables['zos_tide_model'][i,:]
    meanedst = np.mean(edst)
    meantskt = np.mean(tskt)
    meanmodt = np.mean(modt)
    edst = edst - meanedst
    tskt = tskt - meantskt
    modt = modt - meanmodt

    fig = plt.figure(figsize=(8,9))

    ax1 = plt.subplot(3,1,1)
    ax1.plot(times,best,label='Best')
    ax1.plot(times,ch1,label='Channel1')
    ax1.plot(times,ch2,label='Channel2')
    ax1.plot(times,mod,label='Model')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Residual (m)')
    ax1.legend()

    ax2 = plt.subplot(3,1,2)
    ax2.plot(times,edst,label='EDSERPLO, %.2fm' %meanedst)
    ax2.plot(times,tskt,label='TASK, %.2fm' %meantskt)
    ax2.plot(times,modt,label='Model, %.2fm' %meanmodt)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Tide wrt Series Mean (m)')
    ax2.legend()

    ax3 = plt.subplot(3,1,3)
    ax3.plot(times,tskt-edst,label='TASK')
    ax3.plot(times,modt-edst,label='Model')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Tide wrt EDSERPLO (m)')
    ax3.legend()

    fig.suptitle(station)

    plt.show()

