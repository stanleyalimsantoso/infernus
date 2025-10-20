import numpy as np
import pycbc.catalog
import os
from gwpy.timeseries import TimeSeries as GWPYTimeSeries
from pycbc.types import TimeSeries
import h5py

from GWSamplegen.noise_utils import get_data_from_OzStar
# def get_data_from_OzStar(gps_start, duration, ifo, verbose = False):
# 	if gps_start != int(gps_start):
# 		print("NOTE: you have specified a non-integer GPS time to fetch. Make sure this is what you want!")
# 	if gps_start >= 1249850209 and gps_start + duration <= 1249850209 + 4096:
# 		print("we're looking for GW190814 data")
# 		fp = "/fred/oz016/alistair/GWSamplegen/noise/GW190814/GW190814_{}_4096.hdf5".format(ifo)
# 		f = h5py.File(fp, 'r')
# 		print("loaded")
# 		data = f['strain']['Strain'][()]
# 		start_idx = int((gps_start-1249850209)*4096)

# 		data = data[start_idx:start_idx+duration*4096]
# 		data = TimeSeries(data, delta_t = 1/4096, epoch = gps_start)
# 		data = data.resample(1/2048)
# 		return data
# 	elif gps_start >= 1180920447 and gps_start + duration <= 1180920447 + 4096:
# 		print("We're looking for GW170608 data")
# 		fp = "/fred/oz016/alistair/GWSamplegen/noise/GW170608/GW170608_{}_4096.hdf5".format(ifo)
# 		f = h5py.File(fp, 'r')
# 		print("loaded")
# 		data = f['strain']['Strain'][()]
# 		start_idx = int((gps_start-1180920447)*4096)

# 		data = data[start_idx:start_idx+duration*4096]
# 		data = TimeSeries(data, delta_t = 1/4096, epoch = gps_start)
# 		data = data.resample(1/2048)
# 		return data

# 	root = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata"
# 	#ObsRuns = ["O1", "O2", "O3a", "O3b"]
# 	#by looking in reverse order we avoid prematurely selecting the wrong chunk
# 	ObsRuns = ["O3b", "O3a", "O2", "O1"]
# 	for run in ObsRuns:
# 		if verbose:
# 			print("Checking run", run)
# 		chunks = np.sort(np.array(os.listdir(os.path.join(root, run, "strain.4k", "hdf.v1", ifo)), dtype = int))
# 		chunkstart = np.where((chunks <= int(gps_start)))[0]
# 		chunkend = np.where((chunks >= int(gps_start+duration)))[0]
# 		if verbose:
# 			print("chunks:", chunks)
# 		if len(chunkstart) == 0:
# 			continue
# 		else:
# 			chunkstart = chunkstart[-1]
# 			if len(chunkend) == 0:
# 				chunkend = chunkstart
# 				print("In last chunk of run")
# 			else:
# 				chunkend = chunkend[0] -1
# 			if verbose:
# 				print("Found chunk", chunkstart, chunkend)
# 			#print(chunkstart)
# 			#print(chunkend)
# 			break

# 	if chunkend is None:
# 		print("Bad GPS time! not found in any chunk")
# 		return None

# 	#print(chunkstart, chunkend)

# 	segments_start = np.sort(np.array(os.listdir(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkstart])))))
# 	segments_end = np.sort(np.array(os.listdir(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkend])))))
						
# 	segments_split_start = np.array([seg.split("-") for seg in segments_start])
# 	segments_split_end = np.array([seg.split("-") for seg in segments_end])

# 	if verbose:
# 		print("segments start", segments_split_start[:,2])
# 	seg_idx = np.where(segments_split_start[:,2].astype('int') <= gps_start)[0][-1]
# 	seg_idx_end = np.where(segments_split_end[:,2].astype('int') + 4096 >= gps_start+duration)[0][0]


# 	if seg_idx != seg_idx_end:
# 		#print("Complex segment/chunk loading")
# 		simple = False
# 	else:
# 		#print("Simple segment/chunk loading")
# 		simple = True

# 	if simple:
# 		dat = GWPYTimeSeries.read(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkstart]),segments_start[seg_idx]), 
# 					format="hdf5.gwosc", start = gps_start, end = gps_start+duration)
# 		#if there are NaNs, replace them with zeros and print a warning
# 		if np.any(np.isnan(dat)):
# 			print("WARNING: Found NaNs in the data!")
# 			print("GPS time:", gps_start)
# 			print("Ifo:", ifo)
# 			#dat[np.where(np.isnan(dat.data))] = 0.0
# 			#dat.data = np.nan_to_num(dat.data)

# 		dat = dat.to_pycbc()
# 		dat = dat.resample(1/2048)

# 	else:
# 		dat_start = GWPYTimeSeries.read(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkstart]),segments_start[seg_idx]),
# 				format="hdf5.gwosc", start = gps_start)
# 		dat_end = GWPYTimeSeries.read(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkend]),segments_end[seg_idx_end]),
# 				format="hdf5.gwosc", end = gps_start+duration)	
# 		dat_start = dat_start.to_pycbc()
# 		dat_end = dat_end.to_pycbc()
# 		#resample
# 		dat_start = dat_start.resample(1/2048)
# 		dat_end = dat_end.resample(1/2048)
# 		#pad the end of the first segment 
# 		dat_start.append_zeros(len(dat_end))
# 		#concatenate
# 		dat_start[-len(dat_end):] = dat_end
# 		dat = dat_start
# 		#dat = np.concatenate([dat_start, dat_end])
# 	if np.any(np.isnan(dat.data)):
# 		print("WARNING: Found NaNs in the data!")
# 		print("GPS time:", gps_start)
# 		print("Ifo:", ifo)	
# 		print("Returning None for now.")
# 		return None

# 	return dat




# def get_GWTC_events(m1_lower = 0.0, m2_lower = 0.0, m1_upper=1000, m2_upper=1000, exclude_marginal = True):
# 	check_ifos = ['H1', 'L1', 'V1']
# 	#Note that the masses should be  in detector frame (i.e. template bank boundaries)
# 	event_shortlist = {}
# 	found_events = []
# 	#From the PyCBC source code
# 	catalogs = {'GWTC-1-confident': 'LVC',
#              'GWTC-1-marginal': 'LVC',
#              'Initial_LIGO_Virgo': 'LVC',
#              'O1_O2-Preliminary': 'LVC',
#              'O3_Discovery_Papers': 'LVC',
#              'GWTC-2': 'LVC',
#              'GWTC-2.1-confident': 'LVC',
#              'GWTC-2.1-marginal': 'LVC',
#              'GWTC-3-confident': 'LVC',
#              'GWTC-3-marginal': 'LVC'}
# 	for catalog in catalogs.keys():
# 		x = pycbc.catalog.Catalog(source=catalog)
# 		if exclude_marginal:
# 			if 'marginal' in catalog:
# 				continue
# 		#print(catalog)
# 		for key in x.data.keys():
# 			if x.data[key]['mass_2_source']:
# 				#if key[:8] not in found_events:

# 				m1 = x.data[key]['mass_1_source'] * (1+x.data[key]['redshift'])
# 				m2 = x.data[key]['mass_2_source'] * (1+x.data[key]['redshift'])
# 				if m1 < m1_upper and m1 > m1_lower and m2 < m2_upper and m2 > m2_lower:
# 						print(key, "in catalog", catalog)

# 						add = True
# 						for other_event in event_shortlist.keys():
# 							if np.abs(event_shortlist[other_event]['gps'] - x.data[key]['GPS']) < 1:
# 								print("Found duplicate event of", key, ":", other_event)
# 								if key[-1] > other_event[-1]:
# 									print("Removing", other_event)
# 									del event_shortlist[other_event]
# 									break
# 								else:
# 									add = False
						
# 						if add:

# 							found_events.append(key[:-3])
# 							print("m1:",m1)
# 							print("m2:",m2)

# 							event_shortlist[key] = {}
# 							event_shortlist[key]['name'] = key
# 							event_shortlist[key]['m1'] = m1
# 							event_shortlist[key]['m2'] = m2
# 							event_shortlist[key]['gps'] = x.data[key]['GPS']
# 							event_shortlist[key]['catalog'] = catalog
# 							event_shortlist[key]['snr'] = x.data[key]['network_matched_filter_snr']
# 							event_ifos = []
# 							for ifo in check_ifos:
# 								if ifo in [i['detector'] for i in x.data[key]['strain']]:
# 									event_ifos.append(ifo)
# 							event_shortlist[key]['ifos'] = event_ifos
# 							print(" ")
# 	return event_shortlist







# def get_real_events(ifos = ['H1', 'L1'], m1_lower = 0.0, m2_lower = 0.0, m1_upper=1000, m2_upper=1000, exclude_marginal = True, padding = 1024):
# 	#returns a list of GPS times that match the given criteria.
# 	#padding is the amount of time before and after the event to load
# 	events = get_GWTC_events(m1_lower = m1_lower, m2_lower = m2_lower, m1_upper=m1_upper, m2_upper=m2_upper, exclude_marginal = exclude_marginal)

# 	good_events = {}
# 	for event in events.keys():
# 		good = True
# 		for ifo in ifos:
# 			if ifo not in events[event]['ifos']:
# 				print("Skipping", event, "as it does not have data in", ifo)
# 				good = False
# 				break
# 		if good:
# 			good_events[event] = events[event]
	
# 	#next, try to load the data
# 	events = good_events
# 	good_events = {}
# 	for event in events.keys():
# 		print("Loading", event)
# 		good = True
# 		for ifo in ifos:
# 			print("Loading", ifo)
# 			data = get_data_from_OzStar(events[event]['gps']-padding/2, padding, ifo)
# 			if data is None:
# 				print("Failed to load", event, "in", ifo)
# 				good = False
# 				break
# 			else:
# 				print("Loaded", event, "in", ifo)
# 				#events[event][ifo] = data
# 				#append 
# 			if "GW190814" in event and ifo == "H1":
# 				print("")
# 				print("NOTE: NEED TO HANDLE GW190814 (The H1 data is not available publicly)")
				
		
# 		if good:
# 			good_events[event] = events[event]
	
# 	return good_events



from infernus.real_utils import get_real_events


if __name__ == "__main__":
	import argparse
	import json

	parser = argparse.ArgumentParser(description='Load real events')

	parser.add_argument('--configfile', type=str, default = None)

	cfg = parser.parse_args()
	cfg_file = cfg.configfile

	#load the config file
	with open(cfg_file) as f:
		config = json.load(f)
		print(config)
		#need to extract the template limits

		# m1_lower = config['template_mass1_min']
		# m2_lower = config['template_mass2_min']
		# m1_upper = config['template_mass1_max']
		# m2_upper = config['template_mass2_max']
		m1_lower = 1
		m2_lower = 1
		m1_upper = 1000
		m2_upper = 1000
		print("For now, requiring H1 and L1 data")
		events = get_real_events(ifos = ['H1', 'L1'], m1_lower = m1_lower, m2_lower = m2_lower, m1_upper = m1_upper, m2_upper = m2_upper, \
						   exclude_marginal = False, padding = config['duration'], verbose= True)
		print(events)

		#create a directory for each event

		for event in events.keys():
			print("Creating directory for", event)
			os.makedirs(os.path.join(config['save_dir'], event), exist_ok = True)

			# with open(os.path.join(config['save_dir'], "inj.json"), 'r') as f:
			# 	real = json.load(f)
			# 	real['jobname'] = real['jobname'][:-3] + event
			# 	real['injfile'] = "real"


		print(len(events.keys()))

		#make a copy of the inj config file in the save_dir directory

		#import shutil
		#shutil.copy(os.path.join(config['save_dir'], "inj.json"), os.path.join(config['save_dir'], "real_events.json"))
		#open and modify the real_events.json file (only with the old style of workflow)
		#check if inj.json exists first
		if os.path.exists(os.path.join(config['save_dir'], "inj.json")):
			#print("old style of workflow")
			with open(os.path.join(config['save_dir'], "inj.json"), 'r') as f:
				real = json.load(f)
				real['jobname'] = real['jobname'][:-3] + "real"
				real['injfile'] = "real" 

				#save the modified file
				json.dump(real, open(os.path.join(config['save_dir'], "real_events.json"), "w"), indent = 4)