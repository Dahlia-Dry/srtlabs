from digital_rf.digital_metadata import *
import digital_rf
#from thorosmo import *
import pandas as pd
import os
import sys
import numpy as np
import numpy as np
import math

def test_metadata():
    meta = DigitalMetadataReader('/Users/dahliadry/Projects/HaystackSummer2020/srtlabs/ch1/metadata/',accept_empty=False)
    df = meta.read_dataframe(meta.get_bounds()[0],meta.get_bounds()[1])
    print(df['ALT'])
#test_metadata()

class SRTData(object):
    """
    Convert raw SRT data/metadata to pandas DF suitable for experimental analysis
    """
    def __init__(self,datadir, params, start, end):
        """
        Parameters
        ----------
        datadir: string
            path to directory containing raw data, drf_properties, and metadata
        params: list of strings
            list of parameters that user wants to consolidate into dataframe
        start: int
            start data index
        end: int
            end data index
        """
        self.datadir = datadir
        if datadir[-1] == '/':
            self.datadir = datadir[:-1]
        self.params = params
        meta = DigitalMetadataReader(os.path.join(self.datadir,'metadata'))
        self.metastart = meta.get_bounds()[0]
        self.metaend = meta.get_bounds()[1]
        self.start = start
        self.end = end

    def load_data(self):
        """Use DigitalRFReader to load in data"""
        # default values
        input_files = [os.path.split(self.datadir)[0]]
        sfreq = 0.0
        cfreq = None
        plot_type = None
        channel = os.path.split(self.datadir)[1]
        #print(channel)
        subchan = 0  # sub channel to plot
        atime = 0
        start_sample = self.start
        stop_sample = self.end
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
        for f in input_files:
            #print(("file %s" % f))
            try:
                #print("loading data")
                drf = digital_rf.DigitalRFReader(f)
                chans = drf.get_channels()
                #print(chans)
                if channel == "":
                    chidx = 0
                else:
                    chidx = chans.index(channel)
                ustart, ustop = drf.get_bounds(chans[chidx])
                #print(ustart, ustop)
                #print("loading metadata")
                drf_properties = drf.get_properties(chans[chidx])
                sfreq_ld = drf_properties["samples_per_second"]
                sfreq = float(sfreq_ld)
                toffset = start_sample
                #print(toffset)
                if atime == 0:
                    atime = ustart
                else:
                    atime = int(np.uint64(atime * sfreq_ld))
                sstart = atime + int(toffset)
                dlen = stop_sample - start_sample
                #print(sstart, dlen)
                if cfreq is None:
                    # read center frequency from metadata
                    metadata_samples = drf.read_metadata(
                        start_sample=sstart,
                        end_sample=sstart + dlen,
                        channel_name=chans[chidx],
                    )
                    # use center frequency of start of data, even if it changes
                    for metadata in metadata_samples.values():
                        try:
                            cfreq = metadata["center_frequencies"].ravel()[subchan]
                        except KeyError:
                            continue
                        else:
                            break
                    if cfreq is None:
                        print(
                            "Center frequency metadata does not exist for given"
                            " start sample."
                        )
                        cfreq = 0.0
                d = drf.read_vector(sstart, dlen, chans[chidx], subchan)
                #print(d.shape)
                #print("d", d[0:10])
                if len(d) < (stop_sample - start_sample):
                    print(
                        "Probable end of file, the data size is less than expected value."
                    )
                    sys.exit()
                if msl_code_length > 0:
                    d = apply_msl_filter(d, msl_code_length, msl_baud_length)
            except:
                print(("problem loading file %s" % f))
                #traceback.print_exc(file=sys.stdout)
                sys.exit()
        return d

    def get_alt(self):
        """Return Series of altitude coords for each data entry"""
        meta = DigitalMetadataReader(os.path.join(self.datadir,'metadata'))
        df = meta.read_dataframe(meta.get_bounds()[0],meta.get_bounds()[1])
        #print(df)
        return df['ALT']

    def get_az(self):
        """Return Series of azimuth coords for each data entry"""
        meta = DigitalMetadataReader(os.path.join(self.datadir,'metadata'))
        df = meta.read_dataframe(meta.get_bounds()[0],meta.get_bounds()[1])
        return df['AZ']

    def get_times(self):
        """Get the time for each data entry in the specified series"""
        drf = digital_rf.DigitalRFReader(os.path.split(self.datadir)[0])
        channel = os.path.split(self.datadir)[1]
        chans = drf.get_channels()
        if channel == "":
            chidx = 0
        else:
            chidx = chans.index(channel)
        drf_properties = drf.get_properties(chans[chidx])
        sfreq_ld = drf_properties["samples_per_second"]
        sfreq = float(sfreq_ld)
        toffset = self.start
        dlen = self.end - self.start
        t_axis = np.arange(0, dlen) / sfreq + toffset
        return t_axis

    def get_power(self):
        """Use drf_plot code to return power measurements for each data entry"""
        d = self.load_data()
        pdata = (d * np.conjugate(d)).real
        return pdata

    def get_voltage(self):
        """Use drf_plot code to return voltage measurements for each data entry"""
        d = self.load_data()
        voltage = d.real
        return voltage

    def get_phase(self):
        d = self.load_data()
        phase = np.angle(d) / np.pi
        return phase

    def get_r(self):
        d = self.load_data()
        r = np.asarray([math.hypot(x.real,x.imag) for x in d])
        return r

    def process(self):
        data = pd.DataFrame()
        param_dict = {'alt':self.get_alt,'az':self.get_az,'power':self.get_power,
                        'voltage':self.get_voltage,'time':self.get_times,
                        'r':self.get_r,'phase':self.get_phase}
        for param in self.params:
            print(param,' shape ',param_dict[param]().shape)
            data[param] = param_dict[param]()
        return data

def example():
    srtdata = SRTData(datadir = '/Users/dahliadry/Projects/HaystackSummer2020/srtlabs/ch1/',
                        params=['time','power', 'alt']).process()
#example()
