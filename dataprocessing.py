import math
import os
import sys

from digital_rf.digital_rf_hdf5 import DigitalRFReader
import numpy as np

# from thorosmo import *
import pandas as pd
from digital_rf.digital_metadata import DigitalMetadataReader
from drf_process import *


def test_metadata():
    print(os.path.dirname(os.path.abspath(os.curdir)))
    f = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'srtlabs/srt-azel-test/ch1/metadata')
    meta = DigitalMetadataReader(f, accept_empty=False)
    df = meta.read_dataframe(meta.get_bounds()[0], meta.get_bounds()[1])
    print(df.columns)
    #print(df.head())
    df.to_csv('test_metadata.csv')
    print(len(df))
    print(df.iloc[0],df.iloc[-1])
#test_metadata()

class SRTData(object):
    """
    Convert raw SRT data/metadata to pandas DF suitable for experimental analysis
    """
    def __init__(self, datadir, params, starttime=None, endtime=None,ranges=None):
        """
        Parameters
        ----------
        datadir: string
            path to channel directory containing raw data, drf_properties, and metadata
        params: list of strings
            list of parameters that user wants to consolidate into dataframe
        starttime: float, optional
            Start time, seconds since epoch. If None, start at beginning of data.
        endtime: float, optional
            End time, seconds since epoch. If None, end at end of data.
        ranges: list of tuples, optional
            (starttime,endtime) ranges to concatenate into 1 dataset
        """
        self.datadir = os.path.normpath(datadir)
        self.params = params
        self.starttime = starttime
        self.endtime = endtime
        self.meta = DigitalMetadataReader(os.path.join(self.datadir, "metadata"))
        self.drf = DigitalRFReader(os.path.dirname(os.path.abspath(self.datadir)))
        self.ranges = ranges
        self.ustart,self.ustop,self.rxfs = self.calctime()

    def calctime(self):
        input_rf = [i for i in os.listdir(self.datadir) if i != 'metadata' and \
                    i!= '.DS_Store' and i != 'drf_properties.h5'][0]
        channel = os.path.split(self.datadir)[1]
        rxfs = self.drf.get_properties(channel)['samples_per_second']
        if self.starttime is None or self.endtime is None:
            bounds = []
            bounds.append(self.drf.get_bounds(channel))
            print('bounds',self.drf.get_bounds(channel))
            print('bounds diff',self.drf.get_bounds(channel)[1]-self.drf.get_bounds(channel)[0])
            bounds.append(self.meta.get_bounds())
            print('metabounds:',self.meta.get_bounds())
            print('metabounds diff',self.meta.get_bounds()[1]-self.meta.get_bounds()[0])
            bounds = np.asarray(bounds)
            ss = np.max(bounds[:, 0])
            se = np.min(bounds[:, 1])
        if self.starttime is None:
            s0 = ss
        else:
            s0 = int(np.uint64(self.starttime * rxfs))

        if self.endtime is None:
            s1 = se
        else:
            s1 = int(np.uint64(self.endtime * rxfs))
        return s0, s1, rxfs

    def load_data(self):
        """Use DigitalRFReader to load in data"""
        # default values
        #print(input_files)
        sfreq = 0.0
        cfreq = None
        plot_type = None
        channel = self.datadir.split('/')[-1]
        # print(channel)
        subchan = 0  # sub channel to plot
        atime = 0
        #start_sample = self.start
        #stop_sample = self.end
        if self.starttime is None and self.endtime is None:
            start_sample, stop_sample, freq = self.calctime()
        else:
            start_sample = self.starttime
            stop_sample = self.endtime
        modulus = None
        integration = 1
        zscale = (0, 0)
        bins = 256
        log_scale = False
        detrend = False
        show_plots = True
        plot_file = ""

        msl_code_length = 0
        msl_baud_length = 0
        chans = self.drf.get_channels()
        # print(chans)
        if channel == "":
            chidx = 0
        else:
            chidx = chans.index(channel)
        ustart, ustop = self.drf.get_bounds(chans[chidx])
        # print(ustart, ustop)
        # print("loading metadata")
        drf_properties = self.drf.get_properties(chans[chidx])
        sfreq_ld = drf_properties["samples_per_second"]
        sfreq = float(sfreq_ld)
        toffset = start_sample
        # print(toffset)
        if atime == 0:
            atime = ustart
        else:
            atime = int(np.uint64(atime * sfreq_ld))
        sstart = toffset
        dlen = stop_sample - start_sample
        # print(sstart, dlen)
        if cfreq is None:
            print(
                "Center frequency metadata does not exist for given"
                " start sample."
            )
            cfreq = 0.0
        if self.ranges is None:
            d = self.drf.read_vector(sstart, dlen, chans[chidx], subchan)
        else:
            d = []
            for r in self.ranges:
                print(r)
                for x in self.drf.read_vector(r[0],r[1]-r[0],chans[chidx],subchan):
                    d.append(x)
        print(len(d))
        # print("d", d[0:10])
        if msl_code_length > 0:
            d = apply_msl_filter(d, msl_code_length, msl_baud_length)
        print('sfreq',sfreq)
        print('DATA READ DONE')
        return d,sfreq

    def load_metadata(self):
        if self.ranges is None:
            df = self.meta.read_dataframe(self.ustart,self.ustop)
        else:
            dfs = []
            for r in self.ranges:
                d = self.meta.read_dataframe(r[0],r[1])
                dfs.append(d)
            df = pd.concat(dfs)
        print(len(df))
        return df

    def get_alt(self,d,sfreq,times):
        """Return Series of altitude coords for each data entry"""
        df = self.load_metadata()
        df['motor_el'].iloc[0] = 45.0 # first line in metadata is blank
        #df.to_csv('test-metadata.csv')
        #df = pd.read_csv('test-metadata.csv')
        unique_times = np.unique(times.astype(int))
        alt = []
        for t in unique_times:
            print(t)
            try:
                #print(df.loc[int(t)]['motor_el'])
                alt.append(df.loc[int(t)]['motor_el'])
            except:
                print('Nothing at time ' + str(t) + ' found')
                alt.append(-1)
        print(len(alt))
        return np.array(alt)

    def get_az(self):
        """Return Series of azimuth coords for each data entry"""
        df = self.load_metadata()
        df['motor_az'].iloc[0] = 118.0 #first line in metadata is blank
        #df.to_csv('test-metadata.csv')
        #df = pd.read_csv('test-metadata.csv')
        unique_times = np.unique(times.astype(int))
        az = []
        for t in unique_times:
            print(t)
            try:
                #print(df.loc[int(t)]['motor_el'])
                az.append(df.loc[int(t)]['motor_az'])
            except:
                print('Nothing at time ' + str(t) + ' found')
                az.append(-1)
        print(len(az))
        return np.array(az)


    def get_times(self):
        """Get the time for each data entry in the specified series"""
        if self.ranges is None:
            start_sample, stop_sample, freq = self.calctime()
            sfreq = float(freq)
            toffset = start_sample
            dlen = stop_sample-start_sample
            t_axis = np.arange(0, dlen) / freq + toffset
        else:
            ustart, ustop, freq= self.calctime()
            t_axis = []
            sfreq = float(freq)
            for r in self.ranges:
                print(r)
                toffset = r[0]
                dlen = r[1]-r[0]
                for x in np.arange(0,dlen) / freq + toffset:
                    t_axis.append(x)
        return np.array(t_axis)

    def get_power(self,d,sfreq,times):
        print('calculating power')
        """Use drf_plot code to return power measurements for each data entry"""
        pdata = np.array(drf_process(d,sfreq,'power',self.ustart,self.ustop))
        print('initial pdata calculated')
        unique_times = np.unique(times.astype(int))
        times = times.tolist()
        print('length of unique times:',len(unique_times))
        print(unique_times)
        pdata_avg = []
        for i in range(len(unique_times)-1):
            print(i,times.index(unique_times[i]),times.index(unique_times[i+1]))
            pdata_avg.append(np.mean(pdata[times.index(unique_times[i]):times.index(unique_times[i+1])]))
        pdata_avg.append(np.mean(pdata[times.index(unique_times[-1]):]))
        print('length of pdata_avg:',len(pdata_avg))
        return pdata_avg

    def get_psd(self,d,sfreq,times):
        print('calculating psd')
        """Use drf_plot code to return power spectral density measurements for each data entry"""
        pdata = np.array(drf_process(d,sfreq,'spectrum',self.ustart,self.ustop))
        print('initial pdata calculated')
        unique_times = np.unique(times.astype(int))
        times = times.tolist()
        print('length of unique times:',len(unique_times))
        print(unique_times)
        freq_avg = []
        psd_avg = []
        for i in range(len(unique_times)-1):
            print(i,times.index(unique_times[i]),times.index(unique_times[i+1]))
            freq_avg.append(np.mean(pdata[0,times.index(unique_times[i]):times.index(unique_times[i+1])]))
            psd_avg.append(np.mean(pdata[1,times.index(unique_times[i]):times.index(unique_times[i+1])]))
        freq_avg.append(np.mean(pdata[0,times.index(unique_times[-1]):]))
        psd_avg.append(np.mean(pdata[1,times.index(unique_times[-1]):]))
        print('length of pdata_avg:',len(freq_avg))
        return [freq_avg,psd_avg]

    def process(self):
        data = pd.DataFrame()
        d,sfreq = self.load_data()
        times= self.get_times()
        param_dict = {
            "alt": self.get_alt,
            "az": self.get_az,
            "power": self.get_power,
            "time":np.unique(times),
            "power spectral density":self.get_psd,
        }
        for param in self.params:
            if param == "time":
                #print(param, " shape ", param_dict[param](d,sfreq,times).shape)
                data[param] = param_dict[param]
            elif param == 'power spectral density':
                calc = self.get_psd(d,sfreq,times)
                data['frequency'] = calc[0]
                data['power spectral density'] = calc[0]
            else:
                data[param] = param_dict[param](d,sfreq,times)
        return data

def find_boundset(metadf,outfile):
    """
    figure out which data ranges are readable
    """
    #metadf = pd.read_csv('test_metadata.csv')
    datasets = []
    ranges = []
    begin = 0
    out = open(outfile,'w')
    for i in range(780,len(metadf)):
        if (i-begin) >500:
            data = SRTData(datadir=os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'srtlabs/srt-azel-test/ch1'),
                                params=["power", "alt"],starttime = metadf['Unnamed: 0'].iloc[begin],
                                endtime = metadf['Unnamed: 0'].iloc[i-1]).load_data()
            datasets.append(data)
            ranges.append((metadf['Unnamed: 0'].iloc[begin], metadf['Unnamed: 0'].iloc[i-1]))
            out.write(str(metadf['Unnamed: 0'].iloc[begin]) + ' ' + str(metadf['Unnamed: 0'].iloc[i-1]) + '\n')
            print('setting begin to ',i)
            begin = i
            print(data)
        elif (i-begin) > 5:
            try:
                print('BEGIN',begin,'i',i)
                data = SRTData(datadir=os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'srtlabs/srt-azel-test/ch1'),
                                    params=["power", "alt"],starttime = metadf['Unnamed: 0'].iloc[begin],
                                    endtime = metadf['Unnamed: 0'].iloc[i]).load_data()

            except:
                print('FAILED, trying shift back')
                try:
                    data = SRTData(datadir=os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'srtlabs/srt-azel-test/ch1'),
                                        params=["power", "alt"],starttime = metadf['Unnamed: 0'].iloc[begin],
                                        endtime = metadf['Unnamed: 0'].iloc[i-1]).load_data()
                    datasets.append(data)
                    ranges.append((metadf['Unnamed: 0'].iloc[begin], metadf['Unnamed: 0'].iloc[i-1]))
                    out.write(str(metadf['Unnamed: 0'].iloc[begin]) + ' ' + str(metadf['Unnamed: 0'].iloc[i-1]) + '\n')
                    print('setting begin to ',i)
                    begin = i
                    print(data)
                except:
                    print('FAILED, moving to next set')
                    begin = i

        else:
            print(i-begin)
    print('RANGES',ranges)
    print('LENGTH',len(datasets))

def example():
    f = open('testing.txt','r')
    r = [line.replace('\n','') for line in f]
    ranges = []
    for pair in r[:1]:
        ranges.append((int(pair.split(' ')[0]),int(pair.split(' ')[1])))
    print(ranges)
    #srtdata = SRTData(datadir=os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'srtlabs/srt-azel-test/ch1'),
                        #params=["alt","az"],ranges = ranges).process()
    #srtdata.to_csv('srtdata-test-altaz.csv')
    srtdata2 = SRTData(datadir=os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'srtlabs/srt-azel-test/ch1'),
                        params=["alt","power"],ranges = ranges).process()
    print(len(srtdata2))
    print(srtdata2.iloc[400:450])
    #srtdata2.to_csv('srtdata-test-time.csv')
    #print(len(srtdata))
    print(len(srtdata2))
#example()
#find_boundset()
