# run the monthly obs processing wihout user intervention on QC

# import standard libraries needed for the analysis
import numpy as np
import sys
# local library for the surge observations processing
import process_surgeObs as pso

# set up for the analysis

# directories for input and output data
obsdir = '/home/h01/frxs/surge_obs'
moddir = '/data/users/frwave/surge_port_files/uk_det'
tiddir = '/home/h01/frxs/surge_obs/tides'
ukfdir = '/project/ofrd/surge/port_harmonics'
#outdir = '/home/h01/frxs/surge_obs'
outdir = '.'

# year and month to be processed
#year = '2020'
#month = 'oct'
year = sys.argv[1]
month = sys.argv[2]

# set dictionaries and initial port index value
portdict = pso.setPorts()
datafornc = pso.setObsDict()
modelfornc = pso.setModelDict()
index = -1

# sets whether we are including model data or not
modelread = True
# sets whether we are including tide data or not
tideread = True

# load data looping on port
for port in portdict.keys():

    # load the port observations and model data
    times, channels, flags = pso.loadPortObsNOC(portdict[port]['shortname'], year, month, stdmax=0.8, 
                                                datadir=obsdir)
    if modelread:
        modeltimes,modelresidual,modeltide = pso.loadPortModelMonitoring(portdict[port]['locname'],
                                                                         year, month, 
                                                                         datadir=moddir)
    if tideread:
        timestide,tide = pso.loadPortTideNOC(portdict[port]['shortname'], year, month,
                                             cd2odn=portdict[port]['CDtoODN'], datadir=tiddir)
        timestide_ukcff, tide_ukcff = pso.loadPortTideUKCFF(portdict[port]['locname'], year, month,
                                                            datadir=ukfdir)

    # set up the pandas dataframe for the obs
    print('[INFO] Port: %s' %port)
    df = pso.createDataFrame(times,channels,flags)

    # add data to dictionaries for write out to netCDF file
    if tideread:
        datafornc = pso.appendObsDict(datafornc, portdict, port, times, df, tide=tide, tide_ukcff=tide_ukcff)
    else:
        datafornc = pso.appendObsDict(datafornc, portdict, port, times, df)
    if modelread:
        modelfornc = pso.appendModelDict(modelfornc, portdict, port, times, modelresidual, modeltide)

    ## check what ports we have done so far
    #print('[INFO] %d Ports processed so far' %len(datafornc.keys()))
    #print(datafornc.keys())


# generate netCDF file
if not modelread: modelfornc = None
pso.createSurgeObsnc(datafornc, model=modelfornc, outdir=outdir)
