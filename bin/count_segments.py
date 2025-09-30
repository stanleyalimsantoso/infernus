# a small script to determine how many valid noise segments a job will have
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--jsonfile', type=str, default=None)
argsfile = parser.parse_args().jsonfile
args = json.load(open(argsfile, "r"))
noise_dir = args["noise_dir"]
duration = args["duration"]

print(noise_dir)

if isinstance(noise_dir, list):	
	print("noise_dir is a list, using new function")
	from GWSamplegen.noise_utils import get_valid_noise_times_from_segments
	valid_times = get_valid_noise_times_from_segments(noise_dir, duration, 900, blacklisting = False)

else:
	from GWSamplegen.noise_utils import get_valid_noise_times
	valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, 900, blacklisting = False)

print(len(valid_times))