''' command line example:
    python mdaio.py 'path-to-kwd/filename.kwd' 'path-to-mda-output/filename.mda' 'comma-separated-channel-map' whether-to-subtract-median-reference-one-or-zero dead_chans startend


    python Dropbox/python/mlpy/mdaio.py '/home/m/data/oe/maze/149497139278/2017-05-16_17-49-52/experiment1_114.raw.kwd' '/home/m/ssd/res/oe/9278.mda' '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64' 0
'''

import numpy as np
from pdb import *
import struct,tqdm,copy,sys,h5py,glob,os

class MdaHeader:
    def __init__(self, dt0, dims0):
        uses64bitdims=(max(dims0)>2e9)            
        self.uses64bitdims=uses64bitdims
        self.dt_code=_dt_code_from_dt(dt0)
        self.dt=dt0
        self.num_bytes_per_entry=get_num_bytes_per_entry_from_dt(dt0)
        self.num_dims=len(dims0)
        self.dimprod=np.prod(dims0)
        self.dims=dims0
        if uses64bitdims:
            self.header_size=3*4+self.num_dims*8
        else:
            self.header_size=(3+self.num_dims)*4

class DiskReadMda:
    def __init__(self,path,header=None):
        self._path=path
        if header:
            self._header=header
            self._header.header_size=0
        else:
            self._header=_read_header(self._path)            
    def dims(self):
        return self._header.dims
    def N1(self):
         return int(self._header.dims[0])
    def N2(self):
        return int(self._header.dims[1])
    def N3(self):
        return int(self._header.dims[2])
    def dt(self):
        return self._header.dt
    def readChunk(self,i1=-1,i2=-1,i3=-1,N1=1,N2=1,N3=1):
        if (i2<0):
            return self._read_chunk_1d(i1,N1)
        elif (i3<0):
            if N1 != self._header.dims[0]:
                print ("Unable to support N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            X=self._read_chunk_1d(i1+N1*i2,N1*N2)
            return np.reshape(X,(N1,N2),order='F')
        else:
            if N1 != self._header.dims[0]:
                print ("Unable to support N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            if N2 != self._header.dims[1]:
                print ("Unable to support N2 {} != {}".format(N2,self._header.dims[1]))
                return None
            X=self._read_chunk_1d(i1+N1*i2+N1*N2*i3,N1*N2*N3)
            return np.reshape(X,(N1,N2,N3),order='F')
    def _read_chunk_1d(self,i,N):
        f=open(self._path,"rb")
        try:
            f.seek(self._header.header_size+self._header.num_bytes_per_entry*i)
            ret=np.fromfile(f,dtype=self._header.dt,count=N)
            f.close()
            return ret
        except Exception as e: # catch *all* exceptions
            print (e)
            f.close()
            return None

class DiskWriteMda:
    def __init__(self,path,dims,dt='float64'):
        self._path=path
        self._header=MdaHeader(dt,dims)
        _write_header(path, self._header)
    def N1(self):
        return self._header.dims[0]
    def N2(self):
        return self._header.dims[1]
    def N3(self):
        return self._header.dims[2]
    def writeChunk(self,X,i1=-1,i2=-1,i3=-1):
        if (len(X.shape)>=2):
            N1=X.shape[0]
        else:
            N1=1
        if (len(X.shape)>=2):
            N2=X.shape[1]
        else:
            N2=1
        if (len(X.shape)>=3):
            N3=X.shape[2]
        else:
            N3=1
        if (i2<0):
            return self._write_chunk_1d(X,i1)
        elif (i3<0):
            if N1 != self._header.dims[0]:
                print ("Unable to support DiskWriteMda N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            return self._write_chunk_1d(X.ravel(order='F'),i1+N1*i2)
        else:
            if N1 != self._header.dims[0]:
                print ("Unable to support DiskWriteMda N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            if N2 != self._header.dims[1]:
                print ("Unable to support DiskWriteMda N2 {} != {}".format(N2,self._header.dims[1]))
                return None
            return self._write_chunk_1d(X.ravel(order='F'),i1+N1*i2+N1*N2*i3)
    def _write_chunk_1d(self,X,i):
        N=X.size
        f=open(self._path,"ab")
        try:
            f.seek(self._header.header_size+self._header.num_bytes_per_entry*i)
            X.astype(self._header.dt).tofile(f)
            f.close()
            return True
        except Exception as e: # catch *all* exceptions
            print (e)
            f.close()
            return False

def _dt_from_dt_code(dt_code):
    if dt_code == -2:
        dt='uint8'
    elif dt_code == -3:
        dt='float32'
    elif dt_code == -4:
        dt='int16'
    elif dt_code == -5:
        dt='int32'
    elif dt_code == -6:
        dt='uint16'
    elif dt_code == -7:
        dt='float64'
    elif dt_code == -8:
        dt='uint32'
    else:
        dt=None
    return dt

def _dt_code_from_dt(dt):
    if dt == 'uint8':
        return -2
    if dt == 'float32':
        return -3
    if dt == 'int16':
        return -4
    if dt == 'int32':
        return -5
    if dt == 'uint16':
        return -6
    if dt == 'float64':
        return -7
    if dt == 'uint32':
        return -8
    return None

def get_num_bytes_per_entry_from_dt(dt):
    if dt == 'uint8':
        return 1
    if dt == 'float32':
        return 4
    if dt == 'int16':
        return 2
    if dt == 'int32':
        return 4
    if dt == 'uint16':
        return 2
    if dt == 'float64':
        return 8
    if dt == 'uint32':
        return 4
    return None

def _read_header(path):
    f=open(path,"rb")
    try:
        dt_code=_read_int32(f)
        num_bytes_per_entry=_read_int32(f)
        num_dims=_read_int32(f)
        uses64bitdims=False
        if (num_dims<0):
            uses64bitdims=True
            num_dims=-num_dims
        if (num_dims<2) or (num_dims>6):
            print ("Invalid number of dimensions: {}".format(num_dims))
            return None
        dims=[]
        dimprod=1
        if uses64bitdims:
            for j in range(0,num_dims):
                tmp0=_read_int64(f)
                dimprod=dimprod*tmp0
                dims.append(tmp0)
        else:
            for j in range(0,num_dims):
                tmp0=_read_int32(f)
                dimprod=dimprod*tmp0
                dims.append(tmp0)
        dt=_dt_from_dt_code(dt_code)
        if dt is None:
            print ("Invalid data type code: {}".format(dt_code))
            return None
        H=MdaHeader(dt,dims)
        if (uses64bitdims):
            H.uses64bitdims=True
            H.header_size=3*4+H.num_dims*8
        f.close()
        return H
    except Exception as e: # catch *all* exceptions
        print (e)
        f.close()
        return None

def _write_header(path,H):
    f=open(path,"wb")
    try:
        _write_int32(f,H.dt_code)
        _write_int32(f,H.num_bytes_per_entry)
        if H.uses64bitdims:
            _write_int32(f,-H.num_dims)
            for j in range(0,H.num_dims):
                _write_int64(f,H.dims[j])
        else:
            _write_int32(f,H.num_dims)
            for j in range(0,H.num_dims):
                _write_int32(f,H.dims[j])
        f.close()
        return True
    except Exception as e: # catch *all* exceptions
        print (e)
        f.close()
        return False

def readmda(path):
    H=_read_header(path)
    if (H is None):
        print ("Problem reading header of: {}".format(path))
        return None
    ret=np.array([])
    f=open(path,"rb")
    try:
        f.seek(H.header_size)
        #This is how I do the column-major order
        ret=np.fromfile(f,dtype=H.dt,count=H.dimprod)
        ret=np.reshape(ret,H.dims,order='F')
        f.close()
        return ret
    except Exception as e: # catch *all* exceptions
        print (e)
        f.close()
        return None

def writemda32(X,fname):
    return _writemda(X,fname,'float32')

def writemda64(X,fname):
    return _writemda(X,fname,'float64')

def writemda8(X,fname):
    return _writemda(X,fname,'uint8')

def writemda32i(X,fname):
    return _writemda(X,fname,'int32')

def writemda32ui(X,fname):
    return _writemda(X,fname,'uint32')    

def writemda16i(X,fname):
    return _writemda(X,fname,'int16')    

def writemda16ui(X,fname):
    return _writemda(X,fname,'uint16')    
def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:])) 
def load_kwd_without_loading(filename, dataset=0):
    # Try to fill in any wildcards in the filename:
    fname = glob.glob(filename)
    assert len(fname)>0, 'Did not find this file: ' +filename
    filename=fname
    f = h5py.File(filename[0], "r")
    assert f.attrs["kwik_version"] == 2
    data = {}
    recording = f["recordings"][str(dataset)]
    data["info"] = dict(recording.attrs)
    data["app_attrs"] = dict(recording["application_data"].attrs)
    data["data"] = recording["data"]
    print('Size of KWD data is '+str(data['data']))
    return data
def loadContinuous(filepath, dtype=float, verbose=True, 
    start_record=None, stop_record=None, ignore_last_record=True):
    """Load continuous data from a single channel in the file `filepath`.
    
    This is intended to be mostly compatible with the previous version.
    The differences are:
    - Ability to specify start and stop records
    - Converts numeric data in the header from string to numeric data types
    - Does not rely on a predefined maximum data size
    - Does not necessarily drop the last record, which is usually incomplete
    - Uses the block length that is specified in the header, instead of
        hardcoding it.
    - Returns timestamps and recordNumbers as int instead of float
    - Tests the record metadata (N and record marker) for internal consistency
    The OpenEphys file format breaks the data stream into "records", 
    typically of length 1024 samples. There is only one timestamp per record.
    Args:
        filepath : string, path to file to load
        dtype : float or np.int16
            If float, then the data will be multiplied by bitVolts to convert
            to microvolts. This increases the memory required by 4 times.
        verbose : whether to print debugging messages
        start_record, stop_record : indices that control how much data
            is read and returned. Pythonic indexing is used,
            so `stop_record` is not inclusive. If `start` is None, reading
            begins at the beginning; if `stop` is None, reading continues
            until the end.
        ignore_last_record : The last record in the file is almost always
            incomplete (padded with zeros). By default it is ignored, for
            compatibility with the old version of this function.
    Returns: dict, with following keys
        data : array of samples of data
        header : the header info, as returned by readHeader
        timestamps : the timestamps of each record of data that was read
        recordingNumber : the recording number of each record of data that
            was read. The length is the same as `timestamps`.
    """
    if dtype not in [float, np.int16]:
        raise ValueError("Invalid data type. Must be float or np.int16")

    if verbose:
        print "Loading continuous data from " + filepath

    """Here is the OpenEphys file format:
    'each record contains one 64-bit timestamp, one 16-bit sample 
    count (N), 1 uint16 recordingNumber, N 16-bit samples, and 
    one 10-byte record marker (0 1 2 3 4 5 6 7 8 255)'
    Thus each record has size 2*N + 22 bytes.
    """
    # This is what the record marker should look like
    spec_record_marker = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

    # Lists for data that's read
    timestamps = []
    recordingNumbers = []
    samples = []
    samples_read = 0
    records_read = 0
    
    # Open the file
    with file(filepath, 'rb') as f:
        # Read header info, file length, and number of records
        header = readHeader(f)
        record_length_bytes = 2 * header['blockLength'] + 22
        fileLength = os.fstat(f.fileno()).st_size
        n_records = get_number_of_records(filepath)
        
        # Use this to set start and stop records if not specified
        if start_record is None:
            start_record = 0
        if stop_record is None:
            stop_record = n_records
        
        # We'll stop reading after this many records are read
        n_records_to_read = stop_record - start_record
        
        # Seek to the start location, relative to the current position
        # right after the header.
        f.seek(record_length_bytes * start_record, 1)
        
        # Keep reading till the file is finished
        while f.tell() < fileLength and records_read < n_records_to_read:
            # Skip the last record if requested, which usually contains
            # incomplete data
            if ignore_last_record and f.tell() == (
                fileLength - record_length_bytes):
                break
            
            # Read the timestamp for this record
            # litte-endian 64-bit signed integer
            timestamps.append(np.fromfile(f, np.dtype('<i8'), 1))
        
            # Read the number of samples in this record
            # little-endian 16-bit unsigned integer
            N = np.fromfile(f, np.dtype('<u2'), 1).item() 
            if N != header['blockLength']:
                raise IOError('Found corrupted record in block ' )
            
            # Read and store the recording numbers
            # big-endian 16-bit unsigned integer
            recordingNumbers.append(np.fromfile(f, np.dtype('>u2'), 1))
            
            # Read the data
            # big-endian 16-bit signed integer
            data = np.fromfile(f, np.dtype('>i2'), N)
            if len(data) != N:
                raise IOError("could not load the right number of samples")
            
            # Optionally convert dtype
            if dtype == float: 
                data = data * header['bitVolts']
                        
            # Store the data
            samples.append(data)

            # Extract and test the record marker
            record_marker = np.fromfile(f, np.dtype('<u1'), 10)
            if np.any(record_marker != spec_record_marker):
                raise IOError("corrupted record marker at record %d" %
                    records_read)
            
            # Update the count
            samples_read += len(samples)            
            records_read += 1

    # Concatenate results, or empty arrays if no data read (which happens
    # if start_sample is after the end of the data stream)
    res = {'header': header}
    if samples_read > 0:
        res['timestamps'] = np.concatenate(timestamps)
        res['data'] = np.concatenate(samples)
        res['recordingNumber'] = np.concatenate(recordingNumbers)
    else:
        res['timestamps'] = np.array([], dtype=np.int)
        res['data'] = np.array([], dtype=dtype)
        res['recordingNumber'] = np.array([], dtype=np.int)
    return res
    
def get_number_of_records(filepath):
    # Open the file
    with file(filepath, 'rb') as f:
        # Read header info
        header = readHeader(f)
        
        # Get file length
        fileLength = os.fstat(f.fileno()).st_size
        
        # Determine the number of records
        record_length_bytes = 2 * header['blockLength'] + 22
        n_records = int((fileLength - 1024) / record_length_bytes)
        if (n_records * record_length_bytes + 1024) != fileLength:
            raise IOError("file does not divide evenly into full records")
    
    return n_records
def readHeader(f):
    """Read header information from the first 1024 bytes of an OpenEphys file.
    
    Args:
        f: An open file handle to an OpenEphys file
    
    Returns: dict with the following keys.
        - bitVolts : float, scaling factor, microvolts per bit
        - blockLength : int, e.g. 1024, length of each record (see 
            loadContinuous)
        - bufferSize : int, e.g. 1024
        - channel : the channel, eg "'CH1'"
        - channelType : eg "'Continuous'"
        - date_created : eg "'15-Jun-2016 21212'" (What are these numbers?)
        - description : description of the file format
        - format : "'Open Ephys Data Format'"
        - header_bytes : int, e.g. 1024
        - sampleRate : float, e.g. 30000.
        - version: eg '0.4'
        Note that every value is a string, even numeric data like bitVolts.
        Some strings have extra, redundant single apostrophes.
    """
    header = {}
    
    # Read the data as a string
    # Remove newlines and redundant "header." prefixes
    # The result should be a series of "key = value" strings, separated
    # by semicolons.
    header_string = f.read(1024).replace('\n','').replace('header.','')
    
    # Parse each key = value string separately
    for pair in header_string.split(';'):
        if '=' in pair:
            key, value = pair.split(' = ')
            key = key.strip()
            value = value.strip()
            
            # Convert some values to numeric
            if key in ['bitVolts', 'sampleRate']:
                header[key] = float(value)
            elif key in ['blockLength', 'bufferSize', 'header_bytes']:
                header[key] = int(value)
            else:
                # Keep as string
                header[key] = value

    return header
def kwd2mda(fname_kwd,fname_mda,channels=[],do_median_ref=False,startends=None,
            sampling_rate=30000,nshanks  = 4,dead_chans=[]):
    dt='int16'
    num_bytes_per_entry=get_num_bytes_per_entry_from_dt(dt)
    dt_code=_dt_code_from_dt(dt)
    is_kwd=fname_kwd[-3:]=='kwd'
    
    if is_kwd:
        data=load_kwd_without_loading(fname_kwd)
        print('Found data with shape '+str(data['data'].shape))
        # here we can restrict the size of the data for processing to just a  part (chunk)
        X=data['data'][int(1e7):data['data'].shape[0],:]
        print('Found open ephys data with shape '+str(X.shape))    
        if len(channels)<1:
            channels = range(X.shape[1]) #TODO maybe as input for reshuffling
        try:
            import mmy.ss #TODO maybe add limitations of when to start and stop
        except Exception:
            print('Starting at zero and ending at the end')
        if (startends is None):        startends=[0.]
        if len(startends)<1:           startends=[0.]
        if len(startends)<2:
            startends.append(float(X.shape[0])/sampling_rate)
        
        startends[0]=(np.floor(np.array(startends[0])*sampling_rate)).astype(int) # convert to samples
        startends[1]=(np.ceil(np.array(startends[1])*sampling_rate)).astype(int) # convert to samples
        print('startends='+str(startends))
        ndim=X.ndim
        if np.mod(X.shape[1],2)==0: # eg 64 channels, no acceleromter data
              nephys_chans=X.shape[1]
        else: nephys_chans=X.shape[1]-3
        duration = X.shape[0] 
    else:
        #        try:
        #            glob.glob(fname_kwd+'/1*H11.continuous')[0]
        #        except Exception:
        ##            import os
        ##            print(os.listdir(fname_kwd))
        assert len(glob.glob(fname_kwd+'/1*H11.continuous'))>0,'Cant find continuous file in '+fname_kwd
        nrecords_continous=get_number_of_records(glob.glob(fname_kwd+'/1*H11.continuous')[0])
        print ('Found .continuous data with number of records = '+str(nrecords_continous))
        ndim=2
        if len(glob.glob(fname_kwd+'/1*H11.continuous')[0])>40:
            print('Found 64 channels')
            nephys_chans=64
        else:
            print('Found 32 channels')
            nephys_chans=32
        duration =  loadContinuous(glob.glob(fname_kwd+'/1*H11.continuous')[0])['data'].shape[0]
    if dt_code is None:
        print ("Unexpected X type: {}".format(dt))
        return False
    f=open(fname_mda,'wb');f.close()    
    with open(fname_mda, "wb") as f:
        print 'writing dt_code '+str(dt_code)
        _write_int32(f,dt_code)
        print 'writing num_bytes_per_entry= '+ str(num_bytes_per_entry)
        _write_int32(f,num_bytes_per_entry)
        print 'writing X.ndim' +str(ndim)
        _write_int32(f,ndim)
        
        print 'writing nephys_chans = '+str(nephys_chans)
        _write_int32(f,nephys_chans) #minus 3 bc accelerometer
        print 'writing duration = '+str(duration)
        _write_int32(f,duration)
        #%% Write chunk by chunk 
        if is_kwd:
            # number of X points to write at a time, prevents excess memory usage
            # Should be as small as possible for speed
            BUFFERSIZE =int(140)#20000#X.shape[1]/10
            
            n_rows= int(duration/BUFFERSIZE )
            
            for i in tqdm.trange(int(duration / n_rows) + 1,desc='Converting .kwd to .mda'):
                index = i * n_rows
                if strictly_increasing(channels):
                    buf= X[index:index + n_rows, channels]
                
                else: # I must want to reshuffle channel layout
                    buf = X[index:index + n_rows,:]
                    buf= buf[:,channels]
                if do_median_ref:
                    buf=median_reference(buf,nshanks,dead_chans=dead_chans)
                # set to zero all data before hs_plugin:
                if (index)<startends[0]: # if start of chunk is less than hs_plugin
                    if (index + n_rows)>(startends[0]): # chunk ends after plugin
                        print('Zeroing up to headstage plugin time')
                        #buf=np.delete(buf,range(startends[0]-index),axis=0)
                        buf[:startends[0]-index]=0
                    else: # if buffer completely before hs_plugin, just set it to zero
                        print('Zeroing whole buffer since we are before headstage plugin time')
                        buf=np.zeros(buf.shape,dtype='int16')#np.zeros([0,buf.shape[1]])#
                # set to zero all data after hs_unplug:
                if (index)>startends[1]: 
                    #import pdb;pdb.set_trace()
                    if (index + n_rows)<(startends[1]): #if we are partly before hs_unplug 
                        print('Emptying from headstage plugin time to end')
                        buf[startends[1]-index:]=0#np.zeros([0,buf.shape[1]])#0
                    else : # if buffer completely before hs_plugin, just set it to zero
                        set_trace()
                        print('Zeroing whole buffer since we are after unplug')
                        buf=np.zeros(buf.shape,dtype='int16')#np.zeros([0,buf.shape[1]])#
                    
                buf=buf.T # go from KWD to MDA convention. Must be chunked already to do this!
                
                #This is how I do column-major order
                A=np.reshape(buf,buf.size,order='F').astype(dt)
                
                A.tofile(f)
        else:
            #%%
            
            for this_rec in tqdm.trange( get_number_of_records(glob.glob(fname_kwd+'/1*H11.continuous')[0])):
                buf=[]
                for ich in np.arange(nephys_chans)+1:
                    if ich>9:
                        chstr=str(ich)
                    else:
                        chstr='0'+str(ich)
                    this_file=glob.glob(fname_kwd+'/1*H'+chstr+'.continuous')
                    assert len(this_file)>0,'did not find channel '+ chstr + ' in ' + fname_kwd
                    this_file = this_file[0]
                    buf.append(loadContinuous(this_file,start_record=this_rec, stop_record=this_rec+1,verbose=False)['data'])                    
                buf=np.vstack(buf)
                if do_median_ref & (buf.shape[0]>10):
                    buf=median_reference(buf,nshanks,dead_chans=dead_chans)
                A=np.reshape(buf,buf.size,order='F').astype(dt)
                A.tofile(f)
                #%%
    return True
def median_reference(buf,nshanks,dead_chans=[],do_median_ref=True,ref_buffer=False):
#    if ref_channel:
#        
##        ref_buf= np.reshape(data[index:index + n_rows, ref_channel], (-1,1))
#        buf= buf- ref_buf
    if do_median_ref & (buf.ndim>1):
        Bufmed=copy.copy(buf).astype(float)
        
        if len(dead_chans)>0:
            print ('setting these dead_chans to nan: '+ str(dead_chans))
            Bufmed[:,dead_chans]=np.nan#np.delete(bufmed,np.hstack((dead_chans,range(bufmed.shape[1]-3,bufmed.shape[1]))),axis=1)
        if np.mod(buf.shape[1],2)==0: # eg 64 channels, no acceleromter data
              nephys_chans=buf.shape[1]
        else: nephys_chans=buf.shape[1]-3
        nchans_per_shank=int(nephys_chans/nshanks)
        shankstarts=range(0,nephys_chans,int(nephys_chans/nshanks))
#        nchans_per_shank=int((buf.shape[1]-3)/nshanks)
#        shankstarts=range(0,(buf.shape[1]-3),int(buf.shape[1]/nshanks))
            
        for i in range(nshanks):
            shankchans=range(shankstarts[i],shankstarts[i]+nchans_per_shank)
            bufmed=Bufmed[:,shankchans]                
            ref_buf= np.nanmedian(bufmed,axis=1).astype(int) 
            for ich in shankchans:
                buf[:,ich]=buf[:,ich]-ref_buf
            
        # set dead chans to zero again, bc they will now contain local reference
        if len(dead_chans)>0:
            for dead_chan in dead_chans:
                print ('setting these dead_chans to zero: '+ str(dead_chan))
                if buf.ndim>1:
                    buf[:,dead_chan] = np.zeros(buf.shape[0])
                else:
                    buf[dead_chan]=0
    return buf
def _writemda(X,fname_mda,dt):
    dt_code=0
    num_bytes_per_entry=get_num_bytes_per_entry_from_dt(dt)
    dt_code=_dt_code_from_dt(dt)
    if dt_code is None:
        print ("Unexpected data type: {}".format(dt))
        return False

    f=open(fname_mda,'wb')
    try:
        _write_int32(f,dt_code)
        _write_int32(f,num_bytes_per_entry)
        _write_int32(f,X.ndim)
        for j in range(0,X.ndim):
            _write_int32(f,X.shape[j])
        #This is how I do column-major order
        A=np.reshape(X,X.size,order='F').astype(dt)
        A.tofile(f)
    except Exception as e: # catch *all* exceptions
        print (e)
    finally:
        f.close()
        return True

def _read_int32(f):
    return struct.unpack('<i',f.read(4))[0]
    
def _read_int64(f):
    return struct.unpack('<q',f.read(8))[0]

def _write_int32(f,val):
    f.write(struct.pack('<i',val))
    
def _write_int64(f,val):
    f.write(struct.pack('<q',val))

def mdaio_test(do_generate_ordered_array=False,N=int(1e7)):
    M=4
    
    if do_generate_ordered_array:
        X=np.ndarray((M,N))
        for n in tqdm.trange(0,N,desc='generating array...'):
            for m in range(0,M):
                X[m,n]=n*10+m
    else:
        X=np.random.random_integers(1e5,size=(M,N))
    #    writemda32(X,'tmp1.mda')
    print('Writing array...')
    writemda32(X,'tmp1.mda')
    print('Reading array...')
    Y=readmda('tmp1.mda')
    print (Y)
    try:
        print ('diff is '+str(np.absolute(X-Y).max())+' (should be zero)')
    except Exception:pass
    Z=DiskReadMda('tmp1.mda')
    print (Z.readChunk(i1=0,i2=4,N1=M,N2=N-4))

    A=DiskWriteMda('tmpA.mda',(M,N))
    A.writeChunk(Y,i1=0,i2=0)
    B=readmda('tmpA.mda')
    print (B.shape)
    print (B)

def kwd2mda_test(do_generate_ordered_array=False,N=int(1e7)):
    M=4
    if N>1e4:
        print('Cant generate ordered array with N greater than 1e4! generating random instead')
        do_generate_ordered_array=False
    if do_generate_ordered_array:
        X=np.ndarray((M,N))
        for n in tqdm.trange(0,N,desc='generating array...'):
            for m in range(0,M):
                X[m,n]=n*10+m
    else:
        X=np.random.random_integers(1e4,size=(M,N))
    #    writemda32(X,'tmp1.mda')
    print('Writing array...')
    kwd2mda(X.T,'tmp1.mda')
    print('Reading array...')
    Y=readmda('tmp1.mda')
    print (Y)
    try:
        print ('diff is '+str(np.absolute(X-Y).max())+' (should be zero)')
    except Exception:pass
    Z=DiskReadMda('tmp1.mda')
    print (Z.readChunk(i1=0,i2=4,N1=M,N2=N-4))

    A=DiskWriteMda('tmpA.mda',(M,N))
    A.writeChunk(Y,i1=0,i2=0)
    B=readmda('tmpA.mda')
    print (B.shape)
    print (B)    
def main(argv):#kwd_filename,mda_fname,channels=[],do_median_ref=False,startends=None,
           # sampling_rate=30000,nshanks  = 4):
    ''' command line example:
        python Dropbox/python/mlpy/mdaio.py '/home/m/ssd/data/oe/maze/149497139278/2017-05-16_17-49-52/experiment1_114.raw.kwd' '/home/m/ssd/res/oe/9278.mda' '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64' 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    '''
    print argv
    if len(argv)>4: dead_chans = np.array(argv[4].split(',')).astype(int);print ('dead channels requested are '+str(dead_chans))
    else:           dead_chans = []
    if len(argv)>5: startends  = np.array(argv[5].split(',')).astype(int);print ('start and end requested at '+str(startends))
    else:           startends  = []
    kwd2mda(fname_kwd=argv[0],
            fname_mda=argv[1],
            channels=np.array(argv[2].split(',')).astype(int),
            do_median_ref=bool(int(argv[3])),
            startends=startends,
            sampling_rate=30000,nshanks=4 ,
            dead_chans=dead_chans)
    
if __name__ == "__main__":
    print 'sd'
    main(sys.argv[1:])
    
#mdaio_test()
