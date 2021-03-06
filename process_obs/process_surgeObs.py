#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas import DataFrame
from pandas import to_datetime
import sys
import netCDF4 as nc
from collections import OrderedDict

####
# set dictionaries

def setPorts():
    """Set port metadata in ordered dictionary"""

    ports = OrderedDict()
    ports['EA-Avonmouth'] = {'region':'Bristol Channel','shortname':'PTB', 'locname':'AVON', 'latlon':[51.511,-2.712], 'CDtoODN':-6.50}
    ports['EA-Bournemouth'] = {'region':'South Coast','shortname':'BOU', 'locname':'BMTH', 'latlon':[50.714,-1.875], 'CDtoODN':-1.40}
    ports['EA-Cromer'] = {'region':'East Coast','shortname':'CRO', 'locname':'CROM', 'latlon':[52.934,1.302], 'CDtoODN':-2.75}
    ports['EA-Dover'] = {'region':'East Coast','shortname':'DOV', 'locname':'DVER', 'latlon':[51.114,1.323], 'CDtoODN':-3.67}
    ports['EA-Harwich'] = {'region':'East Coast','shortname':'HAR', 'locname':'HARW', 'latlon':[51.948,1.292], 'CDtoODN':-1.95}
    ports['EA-Heysham'] = {'region':'Northwest Coast','shortname':'HEY', 'locname':'HEYS', 'latlon':[54.032,-2.920], 'CDtoODN':-4.90}
    ports['EA-HinkleyPoint'] = {'region':'Bristol Channel','shortname':'HIN', 'locname':'HINK', 'latlon':[51.211,-3.131], 'CDtoODN':-5.90}
    ports['EA-Ilfracombe'] = {'region':'Bristol Channel','shortname':'ILF', 'locname':'ILFR', 'latlon':[51.211,-4.112], 'CDtoODN':-4.80}
    ports['EA-Immingham'] = {'region':'East Coast','shortname':'IMM', 'locname':'IMMI', 'latlon':[53.630,-0.188], 'CDtoODN':-3.90}
    ports['EA-Liverpool'] = {'region':'Northwest Coast','shortname':'LIV', 'locname':'LVPL', 'latlon':[53.450,-3.018], 'CDtoODN':-4.93}
    ports['EA-Lowestoft'] = {'region':'East Coast','shortname':'LOW', 'locname':'LOFT', 'latlon':[52.473,1.750], 'CDtoODN':-1.50}
    ports['EA-Newhaven'] = {'region':'South Coast','shortname':'NHA', 'locname':'NWHN', 'latlon':[50.782,0.057], 'CDtoODN':-3.52}
    ports['EA-Newlyn'] = {'region':'South Coast','shortname':'NEW', 'locname':'NLYN', 'latlon':[50.103,-5.543], 'CDtoODN':-3.05}
    ports['EA-NorthShields'] = {'region':'East Coast','shortname':'NSH', 'locname':'NSHL', 'latlon':[55.007,-1.440], 'CDtoODN':-2.60}
    ports['EA-Plymouth'] = {'region':'South Coast','shortname':'DEV', 'locname':'PLYM', 'latlon':[50.367,-4.185], 'CDtoODN':-3.22}
    ports['EA-Portsmouth'] = {'region':'South Coast','shortname':'PTM', 'locname':'PMTH', 'latlon':[50.802,-1.111], 'CDtoODN':-2.73}
    ports['EA-Sheerness'] = {'region':'East Coast','shortname':'SHE', 'locname':'SHNS', 'latlon':[51.446,0.743], 'CDtoODN':-2.90}
    ports['EA-StMarys'] = {'region':'South Coast','shortname':'STM', 'locname':'MARY', 'latlon':[49.918,-6.317], 'CDtoODN':-2.91}
    ports['EA-Weymouth'] = {'region':'South Coast','shortname':'WEY', 'locname':'PLND', 'latlon':[50.609,-2.448], 'CDtoODN':-0.93}
    ports['EA-Whitby'] = {'region':'East Coast','shortname':'WHI', 'locname':'WTBY', 'latlon':[54.490,-0.614], 'CDtoODN':-3.00}
    ports['EA-Workington'] = {'region':'Northwest Coast','shortname':'WOR', 'locname':'WORK', 'latlon':[54.651,-3.567], 'CDtoODN':-4.20}
    ports['NRW-Barmouth'] = {'region':'Wales West Coast','shortname':'BAR', 'locname':'BARM', 'latlon':[52.727,-4.065], 'CDtoODN':-2.44}
    ports['NRW-Fishguard'] = {'region':'Wales West Coast','shortname':'FIS', 'locname':'FISH', 'latlon':[52.013,-4.984], 'CDtoODN':-2.44}
    ports['NRW-Holyhead'] = {'region':'Wales West Coast','shortname':'HOL', 'locname':'HOLY', 'latlon':[53.314,-4.620], 'CDtoODN':-3.05}
    ports['NRW-Llandudno'] = {'region':'Wales West Coast','shortname':'LLA', 'locname':'LDNO', 'latlon':[53.308,-3.842], 'CDtoODN':-3.85}
    ports['NRW-MilfordHaven'] = {'region':'Wales West Coast','shortname':'MHA', 'locname':'MILF', 'latlon':[51.707,-5.051], 'CDtoODN':-3.71}
    ports['NRW-Mumbles'] = {'region':'Bristol Channel','shortname':'MUM', 'locname':'MUMB', 'latlon':[51.573,-3.993], 'CDtoODN':-5.00}
    ports['NRW-Newport'] = {'region':'Bristol Channel','shortname':'NPO', 'locname':'NEWP', 'latlon':[51.550,-2.987], 'CDtoODN':-5.81}
    ports['RA-Bangor'] = {'region':'Northern Ireland','shortname':'BAN', 'locname':'BANG', 'latlon':[54.665,-5.670], 'CDtoODN':-2.01}
    ports['RA-Portrush'] = {'region':'Northern Ireland','shortname':'PRU', 'locname':'RUSH', 'latlon':[55.207,-6.657], 'CDtoODN':-1.24}
    ports['SEPA-Aberdeen'] = {'region':'Scottish East Coast','shortname':'ABE', 'locname':'ABDN', 'latlon':[57.144,-2.080], 'CDtoODN':-2.25}
    ports['SEPA-Kinlochbervie'] = {'region':'Scottish West Coast','shortname':'KIN', 'locname':'KBER', 'latlon':[58.457,-5.050], 'CDtoODN':-2.50}
    ports['SEPA-Leith'] = {'region':'Scottish East Coast','shortname':'LEI', 'locname':'LETH', 'latlon':[55.990,-3.182], 'CDtoODN':-2.90}
    ports['SEPA-Lerwick'] = {'region':'Scottish East Coast','shortname':'LER', 'locname':'LERK', 'latlon':[60.154,-1.140], 'CDtoODN':-1.22}
    ports['SEPA-Millport'] = {'region':'Scottish West Coast','shortname':'MIL', 'locname':'MILL', 'latlon':[55.750,-4.906], 'CDtoODN':-1.62}
    ports['SEPA-Portpatrick'] = {'region':'Scottish West Coast','shortname':'POR', 'locname':'PORP', 'latlon':[54.843,-5.120], 'CDtoODN':-1.24}
    ports['SEPA-Stornoway'] = {'region':'Scottish West Coast','shortname':'STO', 'locname':'STWY', 'latlon':[58.207,-6.389], 'CDtoODN':-2.71}
    ports['SEPA-Tobermory'] = {'region':'Scottish West Coast','shortname':'TOB', 'locname':'TOBY', 'latlon':[56.623,-6.064], 'CDtoODN':-2.39}
    ports['SEPA-Ullapool'] = {'region':'Scottish West Coast','shortname':'ULL', 'locname':'UPOL', 'latlon':[57.895,-5.158], 'CDtoODN':-2.75}
    ports['SEPA-Wick'] = {'region':'Scottish East Coast','shortname':'WIC', 'locname':'WICK', 'latlon':[58.441,-3.086], 'CDtoODN':-1.71}
    #ports['TG-PortErin'] = {'region':'Northwest Coast','shortname':'IOM', 'locname':'NONE', 'latlon':[54.085,-4.768], 'CDtoODN':-2.75}
    #ports['TG-StHellier'] = {'region':'South Coast','shortname':'JER', 'locname':'NONE', 'latlon':[49.176,-2.115], 'CDtoODN':-5.88}

    return ports


def setObsDict():
    return OrderedDict()


def setModelDict():
    return OrderedDict()


def nextPort(x, ports):
    """Increments port called from ordered dictionary"""

    portlist = [port for port in ports]

    return x+1, portlist[x+1]


def setMonths():
    """Set month names and indexes"""

    months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
              'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

    return months


####
# read and store observations

def readNOCobs(filein, tdelta=900, maskthresh=5.0):
    """Read 15 minute surge residual data from an NOC .mdl text file"""

    print('[INFO] Reading data from %s' %filein)
    with open(filein,'r') as inp:
        rddata = inp.readlines()
        inp.close()

    dtstart = np.int(rddata[0].split(' ')[1].split(':')[0])

    res = np.ravel(np.genfromtxt(filein, skip_header=2, defaultfmt='%.2f'))
    res = np.ma.masked_invalid(res)
    if maskthresh is not None:
        res = np.ma.masked_where(np.abs(res) > maskthresh, res)

    yyyy = np.int(dtstart / 10000)
    mm = np.int(np.mod(dtstart,10000) / 100)
    dd = np.int(np.mod(np.mod(dtstart,10000),100))
    datestart = dt.datetime(yyyy,mm,dd,0,0,0)
    times = np.array([datestart + dt.timedelta(seconds=i) for i in range(0,tdelta*len(res),tdelta)])

    return times, res


def genMissing(yyyy, mm, tdelta=900):
    """Create a times-series of 15 minute surge residual missing data"""

    datestart = dt.datetime(yyyy,mm,1,0,0,0)
    mmnext = mm + 1
    yyyynext = yyyy
    if mmnext > 12:
        mmnext = 1
        yyyynext = yyyy + 1
    imax = np.int((dt.datetime(yyyynext,mmnext,1,0,0,0) - datestart).total_seconds())
    lenres = np.int(imax / tdelta)
    times = np.array([datestart + dt.timedelta(seconds=i) for i in range(0,imax,tdelta)])
    res = np.ones(lenres) * 99.99
    res = np.ma.masked_where(np.abs(res) > 5.0, res)

    return times, res


def loadPortObsNOC(portshort, year, month, stdmax=1.0, datadir='.'):
    """Load surge residual observations from the NOC .mdl files for best, channel1 and channel2 data"""

    #dictionary matches numeric month value with mmm key
    months = setMonths()

    # load data from best and both channels
    try:
        times1, res1 = readNOCobs(datadir+'/'+portshort+year+month+'ch1.mdl')
    except:
        print('[WARN] Unable to read %s, returning missing data array' 
              %(datadir+'/'+portshort+year+month+'ch1.mdl'))
        times1, res1 = genMissing(np.int(year), months[month])
    try:
        times2, res2 = readNOCobs(datadir+'/'+portshort+year+month+'ch2.mdl')
    except:
        print('[WARN] Unable to read %s, returning missing data array' 
              %(datadir+'/'+portshort+year+month+'ch1.mdl'))
        times2, res2 = genMissing(np.int(year), months[month])
    try:
        times, res = readNOCobs(datadir+'/'+portshort+year+month+'.mdl')
    except:
        print('[WARN] Unable to read %s, returning missing data array' 
              %(datadir+'/'+portshort+year+month+'ch1.mdl'))
        times, res = genMissing(np.int(year), months[month])

    try:
        std = np.ma.min([2.0*np.nanstd(res),stdmax])
    except:
        # use this to deal with arrays where all data is missing
        std = stdmax
    print('[INO] %s best data std: %.2f' %(portshort,std))

    channels = np.ma.ones([len(res1),3]) * 99.99
    flags    = np.ones([len(res1),3],dtype=int)
    dres = np.abs(res[0:-1] - res[1:])
    if std == stdmax:
        # mask everything - the data is really bad
        channels[:,0] = np.ma.masked_where(channels[:,0] > 0.0, channels[:,0])
        channels[:,1] = np.ma.masked_invalid(res1)
        channels[:,2] = np.ma.masked_invalid(res2)
        flags[:,:] = 0
    else:
        mres = np.ma.masked_invalid(res)
        mres[:-1] = np.ma.masked_where(dres > std, mres[:-1])
        channels[:,0] = np.ma.masked_array(mres)
        #channels[:,1] = np.ma.masked_where(np.abs(res1-mres) > 0.1, res1)
        #channels[:,2] = np.ma.masked_where(np.abs(res2-mres) > 0.1, res2)
        channels[:,1] = np.ma.masked_invalid(res1)
        channels[:,2] = np.ma.masked_invalid(res2)
        flags[channels[:,0] == channels[:,1],0] = 1
        flags[channels[:,0] == channels[:,2],0] = 2
        flags[channels[:,0].mask,0] = 0
        flags[np.abs(channels[:,0]-channels[:,1])>0.1,1] = 0
        flags[channels[:,1].mask,1] = 0
        flags[:,2] = flags[:,2] * 2
        flags[np.abs(channels[:,0]-channels[:,2])>0.1,2] = 0
        flags[channels[:,2].mask,2] = 0

    return times, channels, flags


def loadPortTideNOC(portshort, year, month, cd2odn=None, datadir='.'):
    """Load harmonic tide predictions from the NOC .mdl files"""

    #dictionary matches numeric month value with mmm key
    months = setMonths()

    # load data from tide file
    try:
        times, tide = readNOCobs(datadir+'/'+portshort+year+month+'tide.mdl',
                                 maskthresh=None)
        if cd2odn is not None:
            tide = tide + cd2odn
    except:
        print('[WARN] Unable to read %s, returning missing data array' 
              %(datadir+'/'+portshort+year+month+'tide.mdl'))
        times, tide = genMissing(np.int(year), months[month])

    return times, tide


def loadPortTideUKCFF(portshort, year, month, datadir='.'):
    """Load harmonic tide predictions from the NOC .mdl files"""

    #dictionary matches numeric month value with mmm key
    months = setMonths()

    # load data from tide file
    fname = datadir + '/%s' %year + '/%s.dat' %portshort
    try:
        with open(fname,'r') as inp:
            rddata = inp.readlines()
            inp.close()
        times = []
        tide  = []
        for i in range(1,len(rddata)):
            mm = np.int(rddata[i][4:6])
            if mm == months[month]:
                tide.append(np.float(rddata[i].split(' ')[-1].strip()))
                yyyy = np.int(rddata[i][0:4])
                dd   = np.int(rddata[i][6:8])
                hh   = np.int(rddata[i][8:10])
                mn   = np.int(rddata[i][10:12])
                times.append(dt.datetime(yyyy,mm,dd,hh,mn))
        tide  = np.array(tide)
        times = np.array(times,dtype=object)
    except:
        print('[WARN] Unable to read %s, returning missing data array' 
              %(fname))
        times, tide = genMissing(np.int(year), months[month])

    return times, tide


def appendObsDict(obsdict, portdict, port, times, dataframe, tide=None, tide_ukcff=None):
    """Add port observations data to the output dictionary as numpy masked arrays"""

    #convert dataframe data into numpy masked arrays and aggregate
    residuals = np.ma.empty([len(dataframe['times']),3])
    residuals[:,0] = np.ma.masked_invalid(dataframe['best'])
    residuals[:,1] = np.ma.masked_invalid(dataframe['ch1'])
    residuals[:,2] = np.ma.masked_invalid(dataframe['ch2'])

    flags      = np.empty([len(dataframe['times']),3])
    flags[:,0] = dataframe['bestflag']
    flags[:,1] = dataframe['ch1flag']
    flags[:,2] = dataframe['ch2flag']

    # add to the obs dictionary
    obsdict[port] = {'latlon':portdict[port]['latlon'],'times':times, 'residuals':residuals, 'flags':flags}
    if tide is not None:
        obsdict[port]['tide'] = tide
    if tide_ukcff is not None:
        obsdict[port]['tide_ukcff'] = tide_ukcff

    return obsdict


####
# observations dataframe create and update

def createDataFrame(times, channels, flags):
    """Import port observation data into a pandas DataFrame"""

    print('[INFO] Creating pandas DataFrame for observations')
    df = DataFrame( {'times':times,
                     'best':channels[:,0],
                     'bestflag':flags[:,0],
                     'ch1':channels[:,1],
                     'ch1flag':flags[:,1],
                     'ch2':channels[:,2],
                     'ch2flag':flags[:,2] } )
    return df


def setIndex(series, firstindex, lastindex=None):
    """Tests and check indices of array slice
        firstindex is the location of the first value we want to update
        lastindex is the location of the last value we want to update
        (this routine adds +1 to lastindex for python referencing)"""

    serieslength = len(series)
    if firstindex < 0:
        print('[WARNING] Setting first index to zero')
        firstindex = 0
    if lastindex is None:
        lastindex = firstindex + 1
    else:
        lastindex = lastindex + 1
    if lastindex > serieslength:
        lastindex = serieslength    

    return [firstindex, lastindex]


def maskBest(df, firstindex, lastindex=None, maskchannel=False):
    """Masks best data and sets best/channel flags to 0 for given array slice"""

    i = setIndex(df['best'], firstindex, lastindex=lastindex)
    # find the channel being used and set flags to zero
    chi = df['bestflag'][i[0]]
    df = maskChannel(df, chi, firstindex, lastindex=lastindex, mask=maskchannel)
    # now mask best data and set flags to zero
    df['best'][i[0]:i[1]] = None
    df['bestflag'][i[0]:i[1]] = 0

    return df


def swapBest(df, swapchi, firstindex, lastindex=None, maskchannel=False):
    """Swaps channel data being used by best and updates best/channel flags for given array slice"""

    i = setIndex(df['best'], firstindex, lastindex=lastindex)
    # find the channel being used now and set flags to zero
    if swapchi == 1:
        channel = 'ch1'
        chflag  = 'ch1flag'
        chi = 2
    elif swapchi == 2:
        channel = 'ch2'
        chflag  = 'ch2flag'
        chi = 1
    df = maskChannel(df, chi, firstindex, lastindex=lastindex, mask=maskchannel)
    # now swap in data from the new channel set flags to zero
    df['best'][i[0]:i[1]] = df[channel][i[0]:i[1]]
    df['bestflag'][i[0]:i[1]] = swapchi
    df[chflag][i[0]:i[1]] = swapchi

    return df


def maskChannel(df, chi, firstindex, lastindex=None, mask=False):
    """Sets channel flags to 0 for given array slice, optionally mask data"""

    i = setIndex(df['best'], firstindex, lastindex=lastindex)
    if chi == 1:
        channel = 'ch1'
        chflag  = 'ch1flag'
    elif chi == 2:
        channel = 'ch2'
        chflag  = 'ch2flag'
    df[chflag][i[0]:i[1]] = 0
    if mask:
        df[channel][i[0]:i[1]] = None

    return df


####
# read and store model data

def readModelMonitoring(cycle, portshort, datadir='.', verbose=False):
    """Read port model data from a Met Office surge ports monitoring file"""

    fname = datadir + '/' + 'srguk_port_timeseries_' + \
            cycle.strftime('%Y%m%d%H') + '.nc'
    if verbose:
        print('[INFO] Reading data from %s' %fname)
    try:
        d = nc.Dataset(fname)
        times = nc.num2date(d.variables['time'][:24], d.variables['time'].units)
        locs  = d.variables['location'][:]
        loci  = np.where(locs == portshort)[0]
        residual = d.variables['residual_ht'][:24,loci]
        tide     = d.variables['tide_ht'][:24,loci]
        d.close()
    except:
        print('[WARNING] Unable to read data from file %s - generating missing data series' %fname)
        # generate a 6-hour series with data every 15 minutes (24 values)
        times = np.array([cycle + dt.timedelta(seconds=900) for i in range(0,24)])
        residual = np.ones(24) * 99.99
        tide     = np.ones(24) * 99.99
        residual = np.ma.masked_greater(residual,1.0)
        tide     = np.ma.masked_greater(tide,1.0)

    return times, residual, tide


def loadPortModelMonitoring(portshort, year, month, datadir='.'):
    """Load surge residual model data from the Met Office monitoring archive"""

    print('[INFO] Loading model port residuals data from Met Office monitoring system')
    months = setMonths()

    cycle = dt.datetime(np.int(year),months[month],1,0,0)
    nextmonth = months[month] + 1
    if nextmonth == 13: nextmonth = 1
    first = True
    while cycle.month < nextmonth:
        timestmp, residualtmp, tidetmp = readModelMonitoring(cycle, portshort, datadir=datadir)
        if first:
            times = timestmp
            residual = residualtmp
            tide = tidetmp
            first = False
        else:
            times = np.concatenate((times,timestmp))
            residual = np.concatenate((residual,residualtmp))
            tide = np.concatenate((tide,tidetmp))
        cycle = cycle + dt.timedelta(hours=6)

    moddict = {'port':portshort, 'times':times, 'residual':residual, 'tide':tide}

    return times, residual, tide


def appendModelDict(modeldict, portdict, port, times, residual, tide):
    """Add model observations data to the output dictionary"""

    # add to the dictionary
    modeldict[port] = {'latlon':portdict[port]['latlon'], 'times':times, 'residuals':residual, 'tides':tide}

    return modeldict


####
# plotting

def plotPort(portshort, year, month, times, channels, model=None, tide=None,
             plotCh1=True, plotCh2=True, stdmax=1.0):
    """Plot port observations (model, astros if set) time-series"""

    print('[INFO] Plotting port residuals data')
    months = setMonths()

    dres = np.abs(channels[0:-1,0] - channels[1:,0])
    if channels[:,0].mask.all():
        std = stdmax
    else:
        std = np.ma.min([2.0*np.nanstd(channels[:,0]),stdmax])
    if std == stdmax:
        resflag = np.ma.masked_where(dres > 10.0 * std, channels[:-1,0])
    else:
        resflag = np.ma.masked_where(dres < std, channels[:-1,0])

    plt.figure(figsize=(9,6), facecolor='white')
    plt.plot(times, channels[:,0], marker='+', label='Best')
    if plotCh1:
        plt.plot(times, channels[:,1], label='Ch1')
    if plotCh2:
        plt.plot(times, channels[:,2], label='Ch2')
    plt.scatter(times[:-1],resflag,color='red',marker='o',s=80,zorder=5,label='Flagged')
    if model is not None:
        plt.plot(times, model, label='Model')
    if tide is not None:
        plt.plot(times, tide, label='Tide')
    plt.legend()
    plt.grid()
    plt.title('Surge residuals for port: '+portshort)
    plt.show()


####
# write out functions

def createSurgeObsnc(portdata, model=None, outdir='.'):
    """Write the QCd data (model, astros if set) out to a CF compliant netCDF file"""

    # set the ports list
    ports = np.array([i for i in portdata.keys()], dtype=object)

    # check for tide data in portdata dictionary
    tide = False
    tide_ukcff = False
    if 'tide' in portdata[ports[0]].keys():
        tide = True
    if 'tide_ukcff' in portdata[ports[0]].keys():
        tide_ukcff = True

    # generate arrays from the portdata dictionary
    lats   = np.array([portdata[i]['latlon'][0] for i in portdata.keys()], dtype=float)
    lons   = np.array([portdata[i]['latlon'][1] for i in portdata.keys()], dtype=float)
    vtimes = portdata[ports[0]]['times'][:]
    best = np.ma.empty([len(ports),len(vtimes)])
    ch1  = np.ma.empty([len(ports),len(vtimes)])
    ch2  = np.ma.empty([len(ports),len(vtimes)])
    if model is not None:
        modres  = np.ma.empty([len(ports),len(vtimes)])
        modtide = np.ma.empty([len(ports),len(vtimes)])
    if tide:
        edstide = np.ma.empty([len(ports),len(vtimes)])
    if tide_ukcff:
        tsktide = np.ma.empty([len(ports),len(vtimes)])
    bestflag = np.empty([len(ports),len(vtimes)])
    ch1flag  = np.empty([len(ports),len(vtimes)])
    ch2flag  = np.empty([len(ports),len(vtimes)])
    for lp,port in enumerate(ports):
        lenrd = len(portdata[port]['residuals'][:,0])
        best[lp,:lenrd] = portdata[port]['residuals'][:,0]
        ch1[lp,:lenrd]  = portdata[port]['residuals'][:,1]
        ch2[lp,:lenrd]  = portdata[port]['residuals'][:,2]
        bestflag[lp,:lenrd] = portdata[port]['flags'][:,0]
        ch1flag[lp,:lenrd]  = portdata[port]['flags'][:,1]
        ch2flag[lp,:lenrd]  = portdata[port]['flags'][:,2]
        if model is not None:
            modres[lp,:lenrd]  = np.ravel(model[port]['residuals'][:])
            modtide[lp,:lenrd] = np.ravel(model[port]['tides'][:])
        if tide:
            edstide[lp,:lenrd]  = portdata[port]['tide'][:]
        if tide_ukcff:
            tsktide[lp,:lenrd]  = portdata[port]['tide_ukcff'][:]
    # set channel2 flags to 0,1 status
    ch2flag[ch2flag == 2] = 1

    # open netCDF file for writing
    monthstamp = vtimes[0].strftime('%Y%m')
    if tide:
        outfile = outdir + '/tide_surgeobs_classa_qc_' + monthstamp + '.nc'
    else:
        outfile = outdir + '/surgeobs_classa_qc_' + monthstamp + '.nc'
    print('Writing data to '+outfile)
    outp = nc.Dataset(outfile,'w',format='NETCDF4')

    # create a locations and time dimension
    time      = outp.createDimension('time',len(vtimes))
    station   = outp.createDimension('station',len(ports))
    nchar     = outp.createDimension('nchar', size = 50)

    # populate the time array
    times = outp.createVariable('time','f8',('time',))
    times.standard_name = 'time'
    times.long_name = 'time'
    times.units = 'seconds since 1970-01-01 00:00:00'
    times.calendar = 'gregorian'
    times.axis = 'T'
    times[:] = nc.date2num( vtimes, times.units, times.calendar )

    # populate the location based arrays
    sitesnc = outp.createVariable('station_name', 'S1', dimensions=('station','nchar'))
    sitesnc.standard_name = 'platform_name'
    sitesnc.long_name='tide gauge name'
    sitesnc.cf_role = 'timeseries_id'
    sitesnc[:] = nc.stringtochar(np.asarray(ports, dtype='S50'))

    lonsnc = outp.createVariable('longitude','f4',dimensions=('station'))
    lonsnc.standard_name = 'longitude'
    lonsnc.long_name = 'longitude'
    lonsnc.units = 'degree_east'
    lonsnc.valid_min = -180.0
    lonsnc.valid_max = 180.0
    lonsnc[:] = lons

    latsnc = outp.createVariable('latitude','f4',dimensions=('station'))
    latsnc.standard_name = 'latitude'
    latsnc.long_name = 'latitude'
    latsnc.units = 'degree_north'
    latsnc.valid_min = -90.0
    latsnc.valid_max = 90.0
    latsnc[:] = lats

    # populate the data arrays
    bestnc = outp.createVariable('zos_residual_observed','f4',('station','time',),fill_value=-32768)
    bestnc.standard_name = 'non_tidal_elevation_of_sea_surface_height'
    bestnc.long_name = 'Storm surge residual from quality controlled observations'
    bestnc.units = 'm'
    bestnc.ancillary_variables = 'zos_residual_observed_flag'
    bestnc.cell_methods = 'time: point'
    bestnc.coordinates = 'latitude longitude station_name'
    bestnc.grid_mapping = 'crs'
    myrange = np.array([-10.0, 10.0], dtype='f4')
    bestnc.valid_min = myrange[0]
    bestnc.valid_max = myrange[1]
    bestnc[:,:] = best

    bestflagnc = outp.createVariable('zos_residual_observed_status_flag','i4',('station','time',),fill_value=-32768)
    bestflagnc.standard_name = 'status_flag'
    bestflagnc.long_name = 'Storm surge residual quality control flag'
    bestflagnc.units = '1'
    bestflagnc.coordinates = 'latitude longitude station_name'
    bestflagnc.grid_mapping = 'crs'
    myrange = np.array([0, 2], dtype='i4')
    bestflagnc.valid_range = myrange
    bestflagnc.flag_values = np.arange(myrange[0],myrange[1]+1,dtype='i4')
    bestflagnc.flag_meanings = 'rejected_data data_from_channel_1 data_from_channel_2'
    bestflagnc[:,:] = bestflag

    ch1nc = outp.createVariable('zos_residual_channel1','f4',('station','time',),fill_value=-32768)
    ch1nc.standard_name = 'non_tidal_elevation_of_sea_surface_height'
    ch1nc.long_name = 'Storm surge residual from channel 1'
    ch1nc.units = 'm'
    bestnc.ancillary_variables = 'zos_residual_channel1_flag'
    ch1nc.cell_methods = 'time: point'
    ch1nc.coordinates = 'latitude longitude station_name'
    ch1nc.grid_mapping = 'crs'
    myrange = np.array([-10.0, 10.0], dtype='f4')
    ch1nc.valid_min = myrange[0]
    ch1nc.valid_max = myrange[1]
    ch1nc[:,:] = ch1

    ch1flagnc = outp.createVariable('zos_residual_channel1_status_flag','i4',('station','time',),fill_value=-32768)
    ch1flagnc.standard_name = 'status_flag'
    ch1flagnc.long_name = 'Storm surge channel 1 quality control flag'
    ch1flagnc.units = '1'
    ch1flagnc.coordinates = 'latitude longitude station_name'
    ch1flagnc.grid_mapping = 'crs'
    myrange = np.array([0, 1], dtype='i4')
    ch1flagnc.valid_range = myrange
    ch1flagnc.flag_values = np.arange(myrange[0],myrange[1]+1,dtype='i4')
    ch1flagnc.flag_meanings = 'rejected_data data_within_0.1m_of_best'
    ch1flagnc[:,:] = ch1flag

    ch2nc = outp.createVariable('zos_residual_channel2','f4',('station','time',),fill_value=-32768)
    ch2nc.standard_name = 'non_tidal_elevation_of_sea_surface_height'
    ch2nc.long_name = 'Storm surge residual from channel 2'
    ch2nc.units = 'm'
    bestnc.ancillary_variables = 'zos_residual_channel2_flag'
    ch2nc.cell_methods = 'time: point'
    ch2nc.coordinates = 'latitude longitude station_name'
    ch2nc.grid_mapping = 'crs'
    myrange = np.array([-10.0, 10.0], dtype='f4')
    ch2nc.valid_min = myrange[0]
    ch2nc.valid_max = myrange[1]
    ch2nc[:,:] = ch2

    ch2flagnc = outp.createVariable('zos_residual_channel2_status_flag','i4',('station','time',),fill_value=-32768)
    ch2flagnc.standard_name = 'status_flag'
    ch2flagnc.long_name = 'Storm surge channel 2 quality control flag'
    ch2flagnc.units = '1'
    ch2flagnc.coordinates = 'latitude longitude station_name'
    ch2flagnc.grid_mapping = 'crs'
    myrange = np.array([0, 1], dtype='i4')
    ch2flagnc.valid_range = myrange
    ch2flagnc.flag_values = np.arange(myrange[0],myrange[1]+1,dtype='i4')
    ch2flagnc.flag_meanings = 'rejected_data data_within_0.1m_of_best'
    ch2flagnc[:,:] = ch2flag

    if model is not None:
        modnc = outp.createVariable('zos_residual_model','f4',('station','time',),fill_value=-32768)
        modnc.standard_name = 'non_tidal_elevation_of_sea_surface_height'
        modnc.long_name = 'Storm surge residual from Met Office operational surge model'
        modnc.units = 'm'
        modnc.cell_methods = 'time: point'
        modnc.coordinates = 'latitude longitude station_name'
        modnc.grid_mapping = 'crs'
        myrange = np.array([-10.0, 10.0], dtype='f4')
        modnc.valid_min = myrange[0]
        modnc.valid_max = myrange[1]
        modnc[:,:] = modres

        modnc = outp.createVariable('zos_tide_model','f4',('station','time',),fill_value=-32768)
        modnc.standard_name = 'tidal_sea_surface_height_above_mean_sea_level'
        modnc.long_name = 'Tide wrt MSL from Met Office operational surge model'
        modnc.units = 'm'
        modnc.cell_methods = 'time: point'
        modnc.coordinates = 'latitude longitude station_name'
        modnc.grid_mapping = 'crs'
        modnc.comment = 'mean sea level refers to model at rest, tide harmonics supplied at boundaries'
        myrange = np.array([-20.0, 20.0], dtype='f4')
        modnc.valid_min = myrange[0]
        modnc.valid_max = myrange[1]
        modnc[:,:] = modtide

    if tide:
        edsnc = outp.createVariable('zos_tide_edserplo','f4',('station','time',),fill_value=-32768)
        edsnc.standard_name = 'tidal_sea_surface_height_above_mean_sea_level'
        edsnc.long_name = 'Tide wrt ODN (harmonic prediction from EDSERPLO software)'
        edsnc.units = 'm'
        edsnc.cell_methods = 'time: point'
        edsnc.coordinates = 'latitude longitude station_name'
        edsnc.grid_mapping = 'crs'
        edsnc.comment = 'mean sea level refers to Ordnance Datum Newlyn'
        myrange = np.array([-20.0, 20.0], dtype='f4')
        edsnc.valid_min = myrange[0]
        edsnc.valid_max = myrange[1]
        edsnc[:,:] = edstide

    if tide_ukcff:
        tsknc = outp.createVariable('zos_tide_task','f4',('station','time',),fill_value=-32768)
        tsknc.standard_name = 'tidal_sea_surface_height_above_mean_sea_level'
        tsknc.long_name = 'Tide wrt ODN (harmonic prediction from TASK software)'
        tsknc.units = 'm'
        tsknc.cell_methods = 'time: point'
        tsknc.coordinates = 'latitude longitude station_name'
        tsknc.grid_mapping = 'crs'
        tsknc.comment = 'mean sea level refers to Ordnance Datum Newlyn'
        myrange = np.array([-20.0, 20.0], dtype='f4')
        tsknc.valid_min = myrange[0]
        tsknc.valid_max = myrange[1]
        tsknc[:,:] = tsktide

    crs = outp.createVariable('crs', 'i4')
    crs.grid_mapping_name = "latitude_longitude";
    crs.longitude_of_prime_meridian = 0.0 ;
    crs.semi_major_axis = 6378137.0 ;
    crs.inverse_flattening = 298.257223563 ;

    # add a little bit of metadata
    outp.Conventions = "CF-1.7" ;
    outp.institution = "National Oceanography Centre / Met Office" ;
    outp.contact = "enquiries@metoffice.gov.uk" ;
    outp.references = "http://ntslf.org" ;
    title_str = '15 Minute Quality Controlled Class-A Tide Gauge Surge Residual Data from NOC/Met Office'
    outp.title = title_str ;
    outp.source = "NOC Tide Gauge Observations Quality Control Procedure" ;
    outp.comment = "Observed residuals in this file are reduced from measured water levels using the EDSERPLO tide series"
    outp.featureType="timeSeries"
    outp.history = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ') + ": File Created" ;

    outp.close()

    return
