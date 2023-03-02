
# import statements for BirdNET-Lite

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tensorflow import lite as tflite

import argparse
import operator
import librosa
import numpy as np
import math
import time
import glob
import concurrent.futures
import copy
import sys
import json
##

from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QSlider, QGridLayout, QGridLayout, QLabel, QComboBox, QHBoxLayout, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QCheckBox, QFileDialog

import AviaNZ_manual

class BirdNETDialog(QDialog):

    def __init__(self, parent=None):
        super(BirdNETDialog, self).__init__(parent)
        self.parent = parent

        self.setWindowTitle("Classify Recordings with BirdNET")
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

        self.form = QGridLayout()
        self.form.setSpacing(25)

        lbltitle = QLabel("To analyze the audiofiles of the current directory, set the parameters for BirdNET-Lite or BirdNET-Analyzer. Be aware that the process might take several hours, depending on the number of audiorecordings, calculation power and number of threads")
        lbltitle.setWordWrap(True)
        
        # BirdNET-Lite/BirdNET-Analyzer options
        self.lite = QRadioButton("BirdNET-Lite")
        self.analyzer = QRadioButton("BirdNET-Analyzer")

        self.lat = QLineEdit()
        self.lat.setText("0.0")
        self.lat.setValidator(QDoubleValidator(-90, 90, 2))        

        self.lon = QLineEdit()
        self.lon.setText("0.0")
        self.lon.setValidator(QDoubleValidator(-180, 180, 2))

        self.week = QLineEdit()
        self.week.setText("0")
        self.week.setValidator(QIntValidator(1, 48))

        self.overlap = QLineEdit()
        self.overlap.setText("0.0")
        self.overlap.setValidator(QDoubleValidator(0, 2.9, 1))

        self.sensitivity = QLineEdit()
        self.sensitivity.setText("1.0")
        self.sensitivity.setValidator(QDoubleValidator(0.5, 1.5, 2))

        self.min_conf = QLineEdit()
        self.min_conf.setText("0.05")
        self.min_conf.setValidator(QDoubleValidator(0.01, 0.99, 2))

        self.slist = QLineEdit()
        self.slist.setReadOnly(True)
        
        self.btn_slist = QPushButton("Choose file")
        self.btn_slist.clicked.connect(self.chooseSpeciesList)

        self.threads = QLineEdit()
        self.threads.setText("1")
        self.threads.setValidator(QIntValidator(0, 99))

        # Lite specific options

        self.mea = QCheckBox("Calculate moving exponential average?")
        self.datetime_format = QLineEdit()

        # Analyzer specific options

        self.batchsize = QLineEdit()
        self.batchsize.setValidator(QIntValidator(1, 99))

        self.locale = QComboBox()
        # TODO: get list of possible languages from labels_directory?
        self.locale.addItems(['af', 'ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'hu', 'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'th', 'tr', 'uk', 'zh'])

        self.sf_thresh = QLineEdit()
        self.sf_thresh.setValidator(QDoubleValidator(0.01, 0.99, 2))
        
        # Button to start analysis
        self.btnAnalyze = QPushButton('Analyze')
        self.btnAnalyze.clicked.connect(self.analyze)

        # labels for QLineEdit analyze options
        self.lat_label = QLabel("Latitude")
        self.lon_label = QLabel("Longitude")
        self.week_label = QLabel("Week")
        self.overlap_label = QLabel("Overlap")
        self.sensitivity_label = QLabel("Sensitivity")
        self.min_conf_label = QLabel("Minimum Confidence")
        self.threads_label = QLabel("Number of Threads")

        # parameter layout
        param_grid = QGridLayout()

        param_grid.addWidget(self.lite, 0, 0)
        param_grid.addWidget(self.analyzer, 1, 0)

        param_grid.addWidget(self.lat_label, 2, 0)
        param_grid.addWidget(self.lat, 2, 1)

        param_grid.addWidget(self.lon_label, 3, 0)
        param_grid.addWidget(self.lon, 3, 1)

        param_grid.addWidget(self.week_label, 4, 0)
        param_grid.addWidget(self.week, 4, 1)

        param_grid.addWidget(self.overlap_label, 5, 0)
        param_grid.addWidget(self.overlap, 5, 1)

        param_grid.addWidget(self.sensitivity_label, 6, 0)
        param_grid.addWidget(self.sensitivity, 6, 1)

        param_grid.addWidget(self.min_conf_label, 7, 0)
        param_grid.addWidget(self.min_conf, 7, 1)

        param_grid.addWidget(self.slist, 8, 0)
        param_grid.addWidget(self.btn_slist, 8, 1)

        param_grid.addWidget(self.threads_label, 9, 0)
        param_grid.addWidget(self.threads, 9, 1)

        param_grid.addWidget(self.mea, 10, 1)
        
        param_grid.addWidget(QLabel("Datetime format"), 11, 0)
        param_grid.addWidget(self.datetime_format, 11, 1)

        param_grid.addWidget(QLabel("Batchsize"), 12, 0)
        param_grid.addWidget(self.batchsize, 12, 1)

        param_grid.addWidget(QLabel("Language of the labels"), 13, 0)
        param_grid.addWidget(self.locale, 13, 1)

        param_grid.addWidget(QLabel("Threshold for location filter"), 14, 0)
        param_grid.addWidget(self.sf_thresh, 14, 1)

        param_grid.addWidget(self.btnAnalyze, 15, 1)        

        # overall Layout
        layout = QVBoxLayout()        
        layout.addWidget(lbltitle)
        layout.addLayout(self.form)
        layout.addLayout(param_grid)
        layout.setSpacing(25)
        self.setLayout(layout)

        # default: BirdNET-Lite
        self.lite.setChecked(True)

    def chooseSpeciesList(self):
        species_list = QFileDialog.getOpenFileName(self, 'Choose filter species list')
        self.slist.setText(os.path.basename(species_list[0]))
    
    def analyze(self):
        if self.lite.isChecked():
            if not self.parent.BirdNETLite:
                self.parent.BirdNETLite = BirdNETLite(self.parent)
            self.parent.BirdNETLite.set_parameters(self.lat.text(), self.lon.text(), self.week.text(), self.overlap.text(), self.sensitivity.text(), self.min_conf.text(), self.slist.text(), self.threads.text(), self.mea.isChecked(), self.datetime_format.text())
            self.parent.BirdNETLite.main()
        elif self.analyzer.isChecked():
            if not AviaNZ_manual.BirdNETAnalyzer:
                AviaNZ_manual.BirdNETAnalyzer = BirdNETAnalyzer()
            AviaNZ_manual.BirdNETAnalyzer.analyze()
        else: 
            # TODO: Raise error, choose BirdNET-Lite/BirdNET-Analyzer
            pass

        
class BirdNETLite():
    def __init__(self, AvianNZmanual):
        self.AvianNZ = AvianNZmanual
        self.loadModel()

    def set_parameters(self, lat, lon, week, overlap, sensitivity, min_conf, slist, threads, mea, datetime_format):
        self.lat = float(lat)
        self.lon = float(lon)
        self.week = int(week)
        self.overlap = float(overlap)
        self.sensitivity = max(0.5, min(1.0 - (float(sensitivity) - 1.0), 1.5))
        self.min_conf = float(min_conf)
        self.slist = loadCustomSpeciesList(slist) if slist else []
        self.threads = int(threads)
        self.mea = mea
        self.datetime_format = datetime_format

    def loadModel(self):

        print('Loading BirdNET-Lite model...', end=' ')

        mdlpath = os.path.join('model', 'BirdNET_6K_GLOBAL_MODEL.tflite')
        # Load TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=mdlpath)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get input tensor index
        input_layer_index = input_details[0]['index']
        mdata_input_index = input_details[1]['index']
        output_layer_index = output_details[0]['index']

        # Load labels
        classes = []
        lblpath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'model', 'labels.txt')

        with open(lblpath, 'r') as lfile:
            for line in lfile.readlines():
                classes.append(line.replace('\n', ''))

        self.model = [input_layer_index, mdata_input_index, output_layer_index, classes, interpreter]

        print('DONE!')

    def loadCustomSpeciesList(self, path):

        slist = []
        if os.path.isfile(path):
            with open(path, 'r') as csfile:
                for line in csfile.readlines():
                    slist.append(line.replace('\r', '').replace('\n', ''))
        else:
            raise Exception('Custom species list file or file path does not exist!')
        
        return slist

    def splitSignal(self, sig, rate, seconds=3.0, minlen=1.5):

        # Split signal with overlap
        sig_splits = []
        for i in range(0, len(sig), int((seconds - self.overlap) * rate)):
            split = sig[i:i + int(seconds * rate)]

            # End of signal?
            if len(split) < int(minlen * rate):
                break

            # Signal chunk too short? Fill with zeros.
            if len(split) < int(rate * seconds):
                temp = np.zeros((int(rate * seconds)))
                temp[:len(split)] = split
                split = temp

            sig_splits.append(split)

        return sig_splits
    
    def readAudioData(self, path, sample_rate=48000):

        print('READING AUDIO DATA FROM FILE {}...'.format(os.path.split(path)[1]), end=' ', flush=True)

        # Open file with librosa (uses ffmpeg or libav)
        sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')

        # Split audio into 3-second chunks
        chunks = self.splitSignal(sig, rate)

        print('DONE! READ {} CHUNKS.'.format(len(chunks)))

        return chunks

    def convertMetadata(self, filename):

        if self.datetime_format:
            day = time.strptime(os.path.split(filename)[1], self.datetime_format)[7]
            week = math.cos(math.radians(day/365*360)) + 1

        else:
            # Convert week to cosine
            if self.week >= 1 and self.week <= 48:
                week = math.cos(math.radians(self.week * 7.5)) + 1
            else:
                week = -1

        # Add binary mask
        mask = np.ones((3,))
        if self.lat == -1 or self.lon == -1:
            mask = np.zeros((3,))
        if week == -1:
            mask[2] = 0.0

        return (np.concatenate([np.array([self.lat, self.lon, week]), mask]))

    def custom_sigmoid(self, x):
        
        return 1 / (1.0 + np.exp(-self.sensitivity * x))

    def predict(self, sample):
        interpreter = self.model[4]
        # Make a prediction
        interpreter.set_tensor(self.model[0], np.array(sample[0], dtype='float32'))
        interpreter.set_tensor(self.model[1], np.array(sample[1], dtype='float32'))
        interpreter.invoke()
        prediction = interpreter.get_tensor(self.model[2])[0]

        # Apply custom sigmoid

        p_sigmoid = self.custom_sigmoid(prediction)

        # Get label and scores for pooled predictions
        p_labels = dict(zip(self.model[3], p_sigmoid))

        # Sort by score
        p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

        # Remove species that are on blacklist
        for i in range(min(10, len(p_sorted))):
            if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
                p_sorted[i] = (p_sorted[i][0], 0.0)

        # Only return first the top ten results
        return (p_sorted[:10], p_sigmoid)

    def analyzeAudioData(self, chunks, file):

        # different format for standard and post-processing approach
        detections = {}
        detections_mea = np.zeros(shape=(len(chunks), len(self.model[3])))

        start = time.time()
        print('ANALYZING AUDIO FROM {} ...'.format(os.path.basename(file)), end=' ', flush=True)

        # Convert and prepare metadata
        mdata = self.convertMetadata(file)
        mdata = np.expand_dims(mdata, 0)

        # Parse every chunk
        timestamps = []
        pred_start = 0.0

        i = 0
        for c in chunks:

            # Prepare as input signal
            sig = np.expand_dims(c, 0)

            # Make prediction
            p1, p2 = self.predict([sig, mdata])

            # Save result and timestamp
            detections_mea[i] = p2
            timestamps.append(pred_start)

            pred_end = pred_start + 3.0
            detections[file + ',' + str(pred_start) + ',' + str(pred_end)] = p1
            pred_start = pred_end - self.overlap
            i += 1

        print('DONE! TIME {:.1f} SECONDS'.format(time.time() - start))
        return (detections_mea.transpose(), timestamps, detections)

    def convert_mea_output(self, mea_output, file, timestamps):
        detections = {}
        cntr = 0
        for start in timestamps:
            key = "{},{},{}".format(file, start, start + 3)
            detections[key] = [(d.split("_")[1], mea_output[d][cntr]) for d in mea_output]
            cntr += 1
        
        return detections

    def writeAvianzOutput(self, detections, file, white_list):
        # TODO: get Duration from file
        # TODO: adapt for mea detections
        output = [{"Operator": "FloMee", "Reviewer": "FloMee", "Duration": 60}]
        for d in detections:
            seg = [float(d.split(",")[1]), float(d.split(",")[2]), 0.0, 0.0]
            labels = []
            for entry in detections[d]:   
                if entry[1] >= self.min_conf and (entry[0] in white_list or len(white_list) == 0):
                    labels.append({"species": entry[0].split("_")[1], "certainty": float(entry[1])*100, "filter": "BirdNET-Lite", "calltype": "non-specified"})

            if len(labels) > 0:
                seg.append(labels)
                output.append(seg)

        if len(output) > 1:
            with open(file + ".data", "w") as rfile:
                json.dump(output, rfile)

    def movingExpAverage(self, timetable, n=3):
        """calculate moving exponential average"""

        weights = np.exp(np.linspace(-1., 0., n))
        weights /= weights.sum()
        i = 0
        for row in timetable:
            # a = np.convolve(row, weights, mode='full')[:len(row)]
            a = np.convolve(row, weights, mode='full')[n-1:]
            # a[:n-1] = row[:n-1]
            # a[:n-1] = a[n-1]
            timetable[i] = a
            i += 1
        return timetable

    def whiteListing(self, timetable, white_list, model):
        detections = {}
        i = 0
        for j in timetable:
            if (self.model[3][i] in white_list) or (len(white_list) == 0 and self.model[3][i] not in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']):
                detections[self.model[3][i]] = j
            i += 1
        return detections
    
    def analyze(self, filelist, white_list):

        for file in filelist:
            # Read audio data
            audioData = self.readAudioData(file)

            # Process audio data and get detections
            pp_det, timestamps, def_det = self.analyzeAudioData(audioData, file)

            if self.mea is False:
                self.writeAvianzOutput(def_det, file, white_list)
            
            elif self.mea is True:
            # apply moving exponential average to pp_det and write results to tempfile
                mea_det = self.whiteListing(movingExpAverage(pp_det), white_list, model)
                mea_det_convert = self.convert_mea_output(mea_det, file, timestamps)
                self.writeAvianzOutput(mea_det_convert, file, white_list)
        
        self.AvianNZ.loadFile(name = self.AvianNZ.filename)

    def main(self):
        # Load custom species list

        # # check variables range
        # print(args.lat)
        # if not -90 <= args.lat <= 90:
        #     raise Exception('Argument --lat not in [-90, 90]')
        # if not -180 <= args.lon <= 180:
        #     raise Exception('Argument --lon not in [-180, 180]')
        # if not 0 <= args.overlap <= 2.9:
        #     raise Exception('Argument --overlap not in [0.0, 2.9]')
        # if not -1 <= args.week <= 48 or args.week == 0:
        #     raise Exception('Argument --week not in [1, 48]')
        # if not 0.5 <= args.sensitivity <= 1.5:
        #     raise Exception('Argument --sensitivity not in [0.5, 1.5]')
        # if not 0.01 <= args.min_conf <= 0.99:
        #     raise Exception('Argument --min_conf not in [0.01, 0.99]')

        # create list of filenames
        filelist = [file.absoluteFilePath() for file in self.AvianNZ.listFiles.listOfFiles if file.isFile()]
        print(filelist)

        # if not filelist:
        #     raise Exception("No file(s) to analyze. Check input file path and try again!")

        # create list of lists of filenames to pass to different threads
        step = -(-len(filelist)//self.threads)
        file_threads = [filelist[i:i + step] for i in range(0, len(filelist), step)]

        # run analyze on different threads
        # with concurrent.futures.ThreadPoolExecutor() as executer:
        #     futures = [executer.submit(self.analyze, flist, copy.deepcopy(self.slist)) for flist in file_threads]
        self.analyze(filelist, self.slist)
        
class BirdNETAnalyzer():

    def __init__(self, parent=None):
        pass



if __name__ == '__main__':

    main()

    # Example calls
    # python3 analyze.py --i '/absolute/path/to/your/soundfile/directory/*.wav' --lat 35.4244 --lon -120.7463 --week 18
    # python3 analyze.py --i 'relative/path/to/your/soundfile/directory/*.mp3' --lat 47.6766 --lon -122.294 --week 11 --overlap 1.5 --min_conf 0.25 --sensitivity 1.25 --slist 'example/custom_species_list.txt'
