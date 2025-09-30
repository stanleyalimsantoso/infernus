from GWSamplegen.noise_utils import get_valid_noise_times, get_valid_noise_times_from_segments
import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--jsonfile', type=str, default=None)
argsfile = parser.parse_args().jsonfile

args = json.load(open(argsfile, "r"))
noise_dir = args["noise_dir"]
maxnoisesegs = args["max_noise_segments"]
duration = args["duration"]
savedir = args["save_dir"]
injfile = args["injfile"]

if injfile == "None":
	fail_tolerant = True
	print("Background run, fail tolerant mode is on")
else:
	fail_tolerant = False
	print("Not a background run, fail tolerant mode is off")

print(noise_dir)
delta_f = 1/duration

if isinstance(noise_dir, list):
	print("new noise loading")
	valid_times = get_valid_noise_times_from_segments(noise_dir, duration, 900, blacklisting = False)
else:
	valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, 900, blacklisting = False)

print(len(valid_times), "files to merge")


missing_segments = []
for i in range(1, len(valid_times)):
	#check if ALL files are present
	if os.path.exists(os.path.join(savedir, "timeslides_{}.npy".format(i))) == False:
		print("Missing segment", i)
		if fail_tolerant == False:
			missing_segments.append(i)

if len(missing_segments) > 0:

	print("Missing segments", missing_segments)
	print("To rerun, use the following command:")

	missing_ranges = []
	start = missing_segments[0]
	end = start
	for i in range(1, len(missing_segments)):
		if missing_segments[i] == end + 1:
			end = missing_segments[i]
		else:
			if start == end:
				missing_ranges.append(str(start))
			else:
				missing_ranges.append(f"{start}-{end}")
			start = missing_segments[i]
			end = start
	if start == end:
		missing_ranges.append(str(start))
	else:
		missing_ranges.append(f"{start}-{end+1}")

	missing='"'+','.join(missing_ranges)+'"'
	print("bash /fred/oz016/alistair/infernus/dev/streamline_recovery.sh {} {}".format(argsfile, missing))
	print("missing files in bash format:", missing)
	print("Exiting")
	exit(1)

timeslides = np.load(os.path.join(savedir, "timeslides_0.npy"))

for i in range(1, len(valid_times)):
	try:
		ts_load = np.load(os.path.join(savedir, "timeslides_{}.npy".format(i)))
		timeslides = np.concatenate((timeslides, ts_load), axis = 1)
		#remove the file
		os.remove(os.path.join(savedir, "timeslides_{}.npy".format(i)))
	except:
		print("Failed to merge segment",i)

np.save(os.path.join(savedir, "timeslides.npy"), timeslides)
print("Merged all segments")
os.remove(os.path.join(savedir, "timeslides_0.npy"))