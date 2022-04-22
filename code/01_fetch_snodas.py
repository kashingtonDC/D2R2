import os
import time
import gzip
import shutil
import ftplib
import datetime
import subprocess 

from tqdm import tqdm

# Helpers 

def login_to_ftp(ftproot,data_dir):
	'''
	Use ftplib to login to the NSIDC ftp server, return ftp object cd'd into data dir 
	'''
	f = ftplib.FTP(ftproot)
	f.login()
	f.cwd(data_dir)
	return f

def download_snodat(yeardir, ftp, writedir):
	'''
	Given list of the directories containing data for each year, navigate to each dir, download files, return list of files downloaded 
	'''
	ftp.cwd(yeardir)
	months = [x for x in ftp.nlst() if "." not in x]

	print("Processing SNODAS for {}".format(yeardir))

	allfiles = []
	for month in months[:]:
		monthdir = os.path.join(yeardir,month)
		ftp.cwd(monthdir)
		mfiles = [x for x in ftp.nlst() if x.endswith("tar")]
		for mf in tqdm(mfiles[:]): 
			outfn = os.path.join(writedir,mf)
			if not os.path.exists(outfn): # If file already exists, skip
				with open(outfn, 'wb') as fp:
					ftp.retrbinary('RETR {}'.format(mf), fp.write)
		allfiles.append(mfiles)

	# unnest the lists
	flatfiles = [item for sublist in allfiles for item in sublist]

	print("Wrote SNODAS files for {}".format(yeardir))
	return  [os.path.join(writedir,x) for x in flatfiles]

def extract_snofiles(filelist, writedir):
	'''
	For the files that have been downloaded, (0) untar file, (1) extract desired variables, (2) convert to gtiff, (2) save, and write to write_dir
	'''

	for file in filelist:
		subprocess.Popen(['tar', '-xvf', file, "-C", writedir],stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

	print("extracted tar files in {}".format(writedir))
	return [os.path.join(writedir,x) for x in os.listdir(writedir) if x.endswith(".gz")]


def process_tarfile(tarfile, writedir, snovars = ['1044']):
	'''
	For the files that have been downloaded, 
	(0) untar file, 
	(1) extract desired variables : 
	1025: Precipitation 
	1034: Snow water equivalent X 
	1036: Snow depth
	1038: Snow pack average temperature
	1039: Blowing snow sublimation 
	1044: Snow melt
	1050: Snow pack sublimation
	
	return lists of the txtfiles (to convert to hdr) and the datfiles (to convert to gtiff)
	'''
	# Extract date from og tar file for string matching
	date = os.path.splitext(os.path.split(tarfile)[1])[0].replace("SNODAS_","")

	# Untar the file we want
	cmd = '''tar -xvf {} -C {}'''.format(tarfile,writedir)
	print(cmd)
	os.system(cmd)
	# subprocess.Popen(['tar', '-xvf', tarfile, "-C", writedir],stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
	# Find the untarred gz files
	gz_files = [os.path.join(writedir,x) for x in os.listdir(writedir) if date in x if x.endswith(".gz")]
	# Get the variable strings from each file
	varstrs = [x[x.find("ssmv")+5:x.find("ssmv")+9] for x in gz_files]

	# Compare to list of vars we want, extract if we want it 
	for varstr,file in zip(varstrs,gz_files):
		outfn = os.path.splitext(file)[0]
		if varstr in snovars:
			with gzip.open(file, 'r') as f_in, open(outfn, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
		else:
			continue

	datfiles = [os.path.join(writedir,x) for x in os.listdir(writedir) if date in x if x.endswith(".dat")]
	txtfiles = [os.path.join(writedir,x) for x in os.listdir(writedir) if date in x if x.endswith(".txt")]
	gz_files = [os.path.join(writedir,x) for x in os.listdir(writedir) if date in x if x.endswith(".gz")]

	return [datfiles,txtfiles, gz_files]

def txt2hdr(txtfiles, writedir):
	dates = [x[x.find("TS")+2:x.find("TS")+10] for x in txtfiles]
	ymd = [datetime.datetime.strptime(x, '%Y%m%d') for x in dates]
	hdrfiles = []
	# Account for the absurd datum change in 2013... 
	for date,file in zip(ymd, txtfiles):
		if date < datetime.datetime(2013, 10, 1):
			hdrfile = os.path.join(writedir,"../pre_10_2013.hdr")
		if date >= datetime.datetime(2013, 10, 1):
			hdrfile = os.path.join(writedir,"../post_10_2013.hdr")
		
		# Spec dest file
		snofn = os.path.split((os.path.splitext(file)[0]))[1] + ".hdr"
		snowpath = os.path.join(writedir,snofn)
		hdrfiles.append(snowpath)
		shutil.copy(hdrfile,snowpath)

	return hdrfiles

def dat2tif(datfiles, writedir):

	prod_lookup = dict({
	"1025": "PREC",
	"1034": "SNWE",
	"1044": "SMLT",
	})
	
	outfnsv1 = {}

	for file in datfiles:
		date= file[file.find("TS")+2:file.find("TS")+10]
		for k,v in prod_lookup.items():
			if k in file:
				outfnsv1[file] = date+v+".tif"

	outfnsvf = {}
	for k,v in outfnsv1.items():
		if "PREC" in v:
			if "L01" in k:
				outfnsvf[k] = os.path.join(writedir,os.path.splitext(v)[0]+"LQD.tif")

			if "L00" in k:
				outfnsvf[k] = os.path.join(writedir,os.path.splitext(v)[0]+"SOL.tif")
		else:
			outfnsvf[k]= os.path.join(writedir,v)

	outfiles = []
	for infile, outfile in outfnsvf.items():
		if not os.path.exists(outfile): # Dont write if already there 
			cmd = '''gdal_translate -of GTiff -a_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' -a_nodata -9999 -a_ullr -124.73333333 52.87500000 -66.94166667 24.95000000 {} {}'''.format(infile,outfile)
			os.system(cmd)
		else:
			print("{} already exists.... Moving to next file".format(outfile))	
		
		outfiles.append(outfile)

	return outfiles

def clip_tifs(tifs,dst_dir, shapefile = "../shape/cvws.shp"):
	if not os.path.exists(dst_dir):
		os.mkdir(dst_dir)

	for tif in tifs:
		outfn = os.path.join(dst_dir,os.path.split(tif)[1])
		if not os.path.exists(outfn): # Dont write if already there 
			cmd = '''gdalwarp -cutline {} -crop_to_cutline {} {}'''.format(shapefile,tif, outfn)
			print(cmd)
			os.system(cmd)

	return 

def main():

	# Setup write dirs and hdr files 
	if not os.path.exists("../data"):
		os.mkdir("../data")
	if not os.path.exists("../data/SNODAS"):
		os.mkdir("../data/SNODAS")
	if not os.path.exists("../data/SNODAS/pre_10_2013.hdr"):
		shutil.copyfile("pre_10_2013.hdr","../data/SNODAS/pre_10_2013.hdr")
	if not os.path.exists("../data/SNODAS/post_10_2013.hdr"):
		shutil.copyfile("pre_10_2013.hdr","../data/SNODAS/post_10_2013.hdr")

	# Set some directories and global vars
	ftproot = 'sidads.colorado.edu'
	data_dir = '/DATASETS/NOAA/G02158/masked/'
	tar_dir = os.path.join("../data/SNODAS",'SNODAS_tar')
	gz_dir = os.path.join("../data/SNODAS",'SNODAS_tar_gz')
	tif_dir = os.path.join("../data/SNODAS",'SNODAS_tifs')
	out_dir = os.path.join("../data/SNODAS",'SNODAS_processed')
	for fdir in [tar_dir,gz_dir,tif_dir,out_dir]:
		if not os.path.exists(fdir):
			os.mkdir(fdir)

	# Main routine 
	ftp = login_to_ftp(ftproot,data_dir)
	dirlist = ftp.nlst()
	yeardirs = [os.path.join(data_dir,x) for x in dirlist if "." not in x]

	for y in yeardirs[:]:
		ftp = login_to_ftp(ftproot,data_dir)
		tarfiles = download_snodat(y, ftp, writedir = tar_dir)
		
		# Make sure files are sorted by date 
		tarfiles.sort()

		# Quit the connection so it does not become stale
		ftp.close()

		for tarfile in tarfiles[:]:
			# Quick check to see if the outfn already was processed. If so, skip. 
			proctif_fn = os.path.splitext(os.path.split(tarfile)[1])[0].replace("SNODAS_","")
			procfiles = [x for x in os.listdir(out_dir) if proctif_fn in x]
			# If the files are not processed, execute functions above 
			if len(procfiles) == 0:
				print("=====" * 10)
				print("Processing {}".format(tarfile))
				print("=====" * 10)
				datfiles,txtfiles,gz_files = process_tarfile(tarfile, writedir = gz_dir)
				hdrfiles = txt2hdr(txtfiles,writedir = gz_dir)
				tiffiles = dat2tif(datfiles, writedir = tif_dir)
				clipped = clip_tifs(tiffiles, dst_dir = out_dir)
				# clean up

				print(len(gz_files), len(datfiles), len(txtfiles), len(hdrfiles))
				for gzf in gz_files:
					os.remove(gzf)
				for a,b,c,d in zip(datfiles,txtfiles,hdrfiles,tiffiles):
					os.remove(a)
					os.remove(b)
					os.remove(c)
					# os.remove(d)

			# Cleanup everything in gz dir just in case
			gz_leftovers = [os.path.join(gz_dir,x) for x in os.listdir(gz_dir)]
			for gzl in gz_leftovers:
				os.remove(gzl)

			# # Cleanup the solid precip files (we dont need them)
			solP_files = [os.path.join(tif_dir,x) for x in os.listdir(tif_dir) if "SOL" in x]
			for file in solP_files:
				os.remove(file)

			# solP_files = [os.path.join(out_dir,x) for x in os.listdir(out_dir) if "SOL" in x]
			# for file in solP_files:
			# 	os.remove(file)

if __name__ == '__main__':
	main()