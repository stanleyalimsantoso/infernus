"""utilities for noise handling in Infernus."""

import numpy as np


def noise_generator(valid_times, paths, file_list, duration, sample_rate):
	file_idx = 0
	
	for i in range(len(valid_times)):
		#print(i)
		if i == 0:
			noise = np.load(file_list[file_idx], mmap_mode='r')
		
		if valid_times[i] + duration > int(paths[file_idx][1]) + int(paths[file_idx][2]):
			file_idx += 1
			print("loading new file")
			noise = np.load(file_list[file_idx], mmap_mode='r')
			
		if int(paths[file_idx][1]) <= valid_times[i]:
			#print("start time good")
			if int(paths[file_idx][1]) + int(paths[file_idx][2]) >= valid_times[i] + duration:
				start_idx = int((valid_times[i] - int(paths[file_idx][1])) * sample_rate)
				end_idx = int(start_idx + duration * sample_rate)
				#print(start_idx, end_idx)
				yield np.copy(noise[:,start_idx:end_idx])

def noise_fetcher(index, valid_times, paths, file_list, duration, sample_rate):
	file_idx = 0
	
	for i in range(len(paths)):

		if valid_times[index] >= int(paths[i][1]) and valid_times[index] + duration <= int(paths[i][1]) + int(paths[i][2]):
			noise = np.load(file_list[i], mmap_mode='r')
			file_idx = i
			break
		else:
			pass
			#print("loading new file")
	
	
	start_idx = int((valid_times[index] - int(paths[file_idx][1])) * sample_rate)
	end_idx = int(start_idx + duration * sample_rate)
	return np.copy(noise[:,start_idx:end_idx])