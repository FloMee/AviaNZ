
# import statements for BirdNET-Lite

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tensorflow import lite as tflite

import operator
import librosa
import numpy as np
import math
import time
import glob
import concurrent.futures
from multiprocessing import Pool 
import copy
import sys
import json
##
import traceback

from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QSlider, QGridLayout, QGridLayout, QLabel, QComboBox, QHBoxLayout, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QCheckBox, QFileDialog, QMessageBox

import AviaNZ_manual

class BirdNETDialog(QDialog):

    def __init__(self, parent=None):
        super(BirdNETDialog, self).__init__(parent)
        self.parent = parent

        self.setWindowTitle("Classify Recordings with BirdNET")
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

        lbltitle = QLabel("To analyze the audiofiles of the current directory, set the parameters for BirdNET-Lite or BirdNET-Analyzer. Be aware that the process might take several hours, depending on the number of audiorecordings, calculation power and number of threads")
        lbltitle.setWordWrap(True)
        
        # BirdNET-Lite/BirdNET-Analyzer options
        self.lite = QRadioButton("BirdNET-Lite")
        self.analyzer = QRadioButton("BirdNET-Analyzer")
        self.lite.clicked.connect(self.updateDialog)
        self.analyzer.clicked.connect(self.updateDialog)

        self.lat = QLineEdit()
        self.lat.setValidator(QDoubleValidator(-90, 90, 2))        

        self.lon = QLineEdit()
        self.lon.setValidator(QDoubleValidator(-180, 180, 2))

        self.week = QLineEdit()
        self.week.setValidator(QIntValidator(1, 48))

        self.overlap = QLineEdit()
        self.overlap.setText("0.0")
        self.overlap.setValidator(QDoubleValidator(0, 2.9, 1))

        self.sensitivity = QLineEdit()
        self.sensitivity.setText("1.0")
        self.sensitivity.setValidator(QDoubleValidator(0.5, 1.5, 2))

        self.min_conf = QLineEdit()
        self.min_conf.setText("0.1")
        self.min_conf.setValidator(QDoubleValidator(0.01, 0.99, 2))

        self.slist = QLineEdit()
        self.slist.setReadOnly(True)
        
        self.btn_slist = QPushButton("Choose file")
        self.btn_slist.clicked.connect(self.chooseSpeciesList)

        self.threads = QLineEdit()
        self.threads.setText("1")
        self.threads.setValidator(QIntValidator(0, os.cpu_count()))

        # Lite specific options

        self.mea = QCheckBox("Calculate moving exponential average?")
        self.datetime_format = QLineEdit()

        # Analyzer specific options

        self.batchsize = QLineEdit()
        self.batchsize.setText("1")
        self.batchsize.setValidator(QIntValidator(1, 99))

        self.locale = QComboBox()
        # TODO: get list of possible languages from labels_directory?
        self.locale.addItems(['af', 'ar', 'cs', 'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'th', 'tr', 'uk', 'zh'])
        self.locale.setCurrentIndex(5)

        self.sf_thresh = QLineEdit()
        self.sf_thresh.setText("0.03")
        self.sf_thresh.setValidator(QDoubleValidator(0.01, 0.99, 2))
        
        # Button to start analysis
        self.btnAnalyze = QPushButton('Analyze')
        self.btnAnalyze.clicked.connect(self.onClickanalyze)

        # labels for QLineEdit analyze options
        self.lat_label = QLabel("Latitude")
        self.lon_label = QLabel("Longitude")
        self.week_label = QLabel("Week")
        self.overlap_label = QLabel("Overlap")
        self.sensitivity_label = QLabel("Sensitivity")
        self.min_conf_label = QLabel("Minimum Confidence")
        self.threads_label = QLabel("Number of Threads")
        self.datetime_format_label = QLabel("Datetime format")
        self.batchsize_label = QLabel("Batchsize")
        self.sf_thresh_label = QLabel("Threshold for location filter")

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
        
        param_grid.addWidget(self.datetime_format_label, 11, 0)
        param_grid.addWidget(self.datetime_format, 11, 1)

        param_grid.addWidget(QLabel("Language of the labels"), 12, 0)
        param_grid.addWidget(self.locale, 12, 1)

        param_grid.addWidget(self.batchsize_label, 13, 0)
        param_grid.addWidget(self.batchsize, 13, 1)

        param_grid.addWidget(self.sf_thresh_label, 14, 0)
        param_grid.addWidget(self.sf_thresh, 14, 1)

        param_grid.addWidget(self.btnAnalyze, 15, 1)        

        # overall Layout
        layout = QVBoxLayout()        
        layout.addWidget(lbltitle)
        layout.addLayout(param_grid)
        layout.setSpacing(25)
        self.setLayout(layout)

        # default: BirdNET-Lite
        self.lite.setChecked(True)
        self.updateDialog()


    def chooseSpeciesList(self):
        species_list = QFileDialog.getOpenFileName(self, 'Choose filter species list')
        self.slist.setText(os.path.basename(species_list[0]))
    
    def validateInputParameters(self):
        correct = True
        for param in [self.lat, self.lon, self.week]:
            if not param.hasAcceptableInput() and param.text() != "":
                correct = False
                param.setStyleSheet("background-color: red")            
            else:
                param.setStyleSheet("background-color: white")
        for param in [self.overlap, self.sensitivity, self.min_conf, self.threads, self.sf_thresh, self.batchsize]:
            if not param.hasAcceptableInput():
                correct = False
                param.setStyleSheet("background-color: red")            
            else:
                param.setStyleSheet("background-color: white")
        return correct
        
    def onClickanalyze(self):
        
        if self.validateInputParameters():
            if not self.parent.BirdNET:
                self.parent.BirdNET = BirdNET(self.parent)
            self.parent.BirdNET.set_parameters(
                self.lite.isChecked(),
                self.lat.text(), 
                self.lon.text(), 
                self.week.text(), 
                self.overlap.text(), 
                self.sensitivity.text(), 
                self.min_conf.text(), 
                self.slist.text(), 
                self.threads.text(), 
                self.mea.isChecked(), 
                self.datetime_format.text(),
                self.locale.currentText(),
                self.batchsize.text(),
                self.sf_thresh.text())
            self.parent.BirdNET.main()
        else: 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Parameters!")
            msg.setInformativeText('You did not input correct values for parameters, please check the red parameters again!')
            msg.setWindowTitle("Warning")
            msg.exec_()

    def updateDialog(self):
        print("Update Dialog")
        if self.lite.isChecked():
            self.mea.setVisible(True)
            self.datetime_format.setVisible(True)
            self.datetime_format_label.setVisible(True)
            self.batchsize.setVisible(False)
            self.batchsize_label.setVisible(False)
            self.sf_thresh.setVisible(False)
            self.sf_thresh_label.setVisible(False)
        else:
            self.mea.setVisible(False)
            self.mea.setChecked(False)
            self.datetime_format.setVisible(False)
            self.datetime_format_label.setVisible(False)
            self.batchsize.setVisible(True)
            self.batchsize_label.setVisible(True)
            self.sf_thresh.setVisible(True)
            self.sf_thresh_label.setVisible(True)

class BirdNET():
    def __init__(self, AviaNZmanual):
        self.AviaNZ = AviaNZmanual
        self.m_interpreter = None

    def set_parameters(self, lite, lat, lon, week, overlap, sensitivity, min_conf, slist, threads, mea, datetime_format, locale, batchsize, sf_thresh):
        self.lite = lite
        self.lat = float(lat) if lat else -1
        self.lon = float(lon) if lon else -1
        self.week = int(week) if week else -1
        self.overlap = float(overlap)
        self.sensitivity = max(0.5, min(1.0 - (float(sensitivity) - 1.0), 1.5))
        self.min_conf = float(min_conf)
        self.sf_tresh = float(sf_thresh)
        self.locale = locale
        # TODO: check if self.labels works or if deepcopy is needed
        self.labels = self.loadLabels()
        self.slist = self.getSpeciesList(slist)
        self.threads = int(threads)
        self.mea = mea
        self.datetime_format = datetime_format
        self.batchsize = int(batchsize)

    def loadModel(self):
        try:
            print('Loading BirdNET model...', end=' ')

            if self.lite:
                mdlpath = os.path.join('models', 'Lite', 'BirdNET_6K_GLOBAL_MODEL.tflite')
            else:
                mdlpath = os.path.join('models', 'Analyzer', 'BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite')
            print(mdlpath)
            # Load TFLite model and allocate tensors.
            interpreter = tflite.Interpreter(model_path=mdlpath)
            interpreter.allocate_tensors()
            print("loaded")
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Get input tensor index
            input_layer_index = input_details[0]['index']
            if self.lite:
                mdata_input_index = input_details[1]['index']
            else:
                mdata_input_index = None
            output_layer_index = output_details[0]['index']        

            # TODO: check if self.labels works or if deepcopy is needed
            model = [input_layer_index, mdata_input_index, output_layer_index, copy.deepcopy(self.labels), interpreter]

            print('DONE!')
        except Exception() as e:
            print(traceback.format_exc())
        return model
    
    def loadMetaModel(self):        
        
        # Load TFLite model and allocate tensors.
        self.m_interpreter = tflite.Interpreter(model_path=os.path.join('models', 'Analyzer', 'BirdNET_GLOBAL_3K_V2.2_MData_Model_FP16.tflite')) #TODO: , num_threads=self.tflite_threads)
        self.m_interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = self.m_interpreter.get_input_details()
        output_details = self.m_interpreter.get_output_details()

        # Get input tensor index
        self.m_input_layer_index = input_details[0]['index']
        self.m_output_layer_index = output_details[0]['index']

    def loadLabels(self):
        # Load labels
        if self.lite:
            lblpath = os.path.join('labels', 'Lite', 'labels_{}.txt'.format(self.locale))
        else:
            lblpath = os.path.join('labels', 'Analyzer', 'BirdNET_GLOBAL_3K_V2.2_Labels_{}.txt'.format(self.locale))
        
        with open(lblpath, 'r') as lfile:
            classes = [line[:-1] for line in lfile]
        
        return classes

    def getSpeciesList(self, path):
        if self.lite:
            slist = self.loadCustomSpeciesList(path)            
        elif self.lat == -1 and self.lon == -1:
            slist = self.loadCustomSpeciesList(path)
        else:
            slist = self.predictSpeciesList()
        return slist
    
    def loadCustomSpeciesList(self, path):
        slist = []
        if path:
            if os.path.isfile(path):
                with open(path, 'r') as csfile:
                    for line in csfile.readlines():
                        slist.append(line.replace('\r', '').replace('\n', ''))
            else:
                raise Exception('Custom species list file or file path does not exist!')
        return slist

    def predictSpeciesList(self):
        l_filter = self.explore()
        # cfg.SPECIES_LIST_FILE = None
        slist = []
        for s in l_filter:
            if s[0] >= self.sf_tresh:
                slist.append(s[1])
        
        return slist

    def explore(self):

        # Make filter prediction
        l_filter = self.predictFilter()

        # Apply threshold
        l_filter = np.where(l_filter >= self.sf_tresh, l_filter, 0)

        # Zip with labels
        l_filter = list(zip(l_filter, self.labels))

        # Sort by filter value
        l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)

        return l_filter
    
    def predictFilter(self):

        # Does interpreter exist?
        if self.m_interpreter == None:
            self.loadMetaModel()

        # Prepare mdata as sample
        sample = np.expand_dims(np.array([self.lat, self.lon, self.week], dtype='float32'), 0)

        # Run inference
        self.m_interpreter.set_tensor(self.m_input_layer_index, sample)
        self.m_interpreter.invoke()

        return self.m_interpreter.get_tensor(self.m_output_layer_index)[0]

    def splitSignal(self, sig, rate, seconds=3.0, minlen=1.5):

        # Split signal with overlap
        sig_splits = []
        for i in range(0, len(sig), int((seconds - self.overlap) * rate)):
            split = sig[i:i + int(seconds * rate)]

            # End of signal?
            if len(split) < int(minlen * rate):
                break

            # Signal chunk too short? Fill with zeros (Lite) or noise (Analyzer).
            if len(split) < int(rate * seconds):
                if self.lite:
                    temp = np.zeros((int(rate * seconds)))
                    temp[:len(split)] = split
                    split = temp
                else:       
                    split = np.hstack((split, self.noise(split, (int(rate * seconds) - len(split)), 0.5)))

            sig_splits.append(split)

        return sig_splits
    
    def noise(self, sig, shape, amount=None):
        random_seed = 42
        random = np.random.RandomState(random_seed)

        # Random noise intensity
        if amount == None:
            amount = random.uniform(0.1, 0.5)

        # Create Gaussian noise
        try:
            noise = random.normal(min(sig) * amount, max(sig) * amount, shape)
        except:
            noise = np.zeros(shape)

        return noise.astype('float32')
    
    def readAudioData(self, path, sample_rate=48000):

        print('READING AUDIO DATA FROM FILE {}...'.format(os.path.split(path)[1]), end=' ', flush=True)

        # Open file with librosa (uses ffmpeg or libav)
        # sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')
        
        # TODO: Does the following makes sense? Taken from BirdNET-Analyzer
        try:
            sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')
        except:
            sig, rate = [], sample_rate

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
    
    def flat_sigmoid(self, x, sensitivity=-1):

        return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))

    def predict(self, samples, model):
        
        interpreter = model[4]
        input_layer_index = model[0]
        mdata_input_index = model[1]
        output_layer_index = model[2]
        # labels = model[3]

        if self.lite:
            # Make a prediction
            interpreter.set_tensor(input_layer_index, np.array(samples[0], dtype='float32'))
            interpreter.set_tensor(mdata_input_index, np.array(samples[1], dtype='float32'))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_layer_index)[0]

            # Apply custom sigmoid

            p_sigmoid = self.custom_sigmoid(prediction)

            # # Get label and scores for pooled predictions
            # p_labels = dict(zip(labels, p_sigmoid))

            # # Sort by score
            # p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

            # # Remove species that are on blacklist
            # for i in range(min(10, len(p_sorted))):
            #     if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
            #         p_sorted[i] = (p_sorted[i][0], 0.0)

            # # Only return first the top ten results
            # return (p_sorted[:10], p_sigmoid)
        
        else:

            # Prepare sample and pass through model
            data = np.array(samples, dtype='float32')

            # Reshape input tensor
            interpreter.resize_tensor_input(input_layer_index, [len(data), *data[0].shape])
            interpreter.allocate_tensors()

            # Make a prediction (Audio only for now)
            interpreter.set_tensor(input_layer_index, np.array(data, dtype='float32'))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_layer_index)

            # return prediction
        
            #TODO: Need for Protobuf model?
            # else:

            #     # Make a prediction (Audio only for now)
            #     prediction = PBMODEL.predict(sample)

            #     return prediction

            # TODO: Need for option of activations?
            # Logits or sigmoid activations?
            # if cfg.APPLY_SIGMOID:
            p_sigmoid = self.flat_sigmoid(np.array(prediction), sensitivity=-self.sensitivity)

        return p_sigmoid

    def analyzeAudioData(self, chunks, file, model):

        # different format for standard and post-processing (mea) approach
        detections = {}
        detections_mea = np.zeros(shape=(len(chunks), len(model[3])))

        start = time.time()
        print('ANALYZING AUDIO FROM {} ...'.format(os.path.basename(file)), end=' ', flush=True)
    
        # Parse every chunk
        timestamps = []
        pred_start = 0.0
        sig_length = 3.0

        labels = model[3]

        i = 0
        if self.lite:
            # Convert and prepare metadata
            mdata = self.convertMetadata(file)
            mdata = np.expand_dims(mdata, 0)
            for c in chunks:

                # Prepare as input signal
                sig = np.expand_dims(c, 0)

                # Make prediction
                # p1, p2 = self.predict([sig, mdata], model)

                p_sigmoid = self.predict([sig, mdata], model)

                # Get label and scores for pooled predictions
                p_labels = dict(zip(labels, p_sigmoid))

                # Sort by score
                p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

                # Remove species that are on blacklist
                for i in range(min(10, len(p_sorted))):
                    if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
                        p_sorted[i] = (p_sorted[i][0], 0.0)

                # Save result and timestamp
                detections_mea[i] = p_sigmoid
                timestamps.append(pred_start)

                pred_end = pred_start + sig_length
                detections[file + ',' + str(pred_start) + ',' + str(pred_end)] = p_sorted[:10]
                pred_start = pred_end - self.overlap
                i += 1

            print('DONE! TIME {:.1f} SECONDS'.format(time.time() - start))
            return (detections_mea.transpose(), timestamps, detections)

        else:
            timestamps_return = []
            start, end = 0, sig_length 
            samples = []
            for c in range(len(chunks)):

                # Add to batch
                samples.append(chunks[c])
                timestamps.append([start, end])

                # Advance start and end
                start += sig_length - self.overlap
                end = start + sig_length

                # Check if batch is full or last chunk        
                if len(samples) < self.batchsize and c < len(chunks) - 1:
                    continue

                # Predict
                p = self.predict(samples, model)

                # Add to results
                for i in range(len(samples)):

                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    # Get prediction
                    pred = p[i]
                    
                    # Add to moving exponential average 
                    detections_mea[i] = pred
                    timestamps_return.append(s_start)
                    
                    # Assign scores to labels
                    p_labels = dict(zip(labels, pred))

                    # Sort by score
                    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

                    # Store top 5 results and advance indicies
                    detections[file + ',' + str(s_start) + ',' + str(s_end)] = p_sorted

                # Clear batch
                samples = []
                timestamps = []
            
            return (detections_mea.transpose(), timestamps_return, detections)
#    except:
#         # Print traceback
#         print(traceback.format_exc(), flush=True)

#         # Write error log
#         msg = 'Error: Cannot analyze audio file {}.\n{}'.format(fpath, traceback.format_exc())
#         print(msg, flush=True)
#         writeErrorLog(msg)
#         return False     


    def convert_mea_output(self, mea_output, file, timestamps):
        detections = {}
        cntr = 0
        for start in timestamps:
            key = "{},{},{}".format(file, start, start + 3)
            detections[key] = [(d, mea_output[d][cntr]) for d in mea_output]
            cntr += 1
        
        return detections

    def writeAvianzOutput(self, detections, file, white_list):
        # TODO: get Duration from file
        output = [{"Operator": self.AviaNZ.operator, "Reviewer": self.AviaNZ.reviewer, "Duration": 60}]
        for d in detections:
            seg = [float(d.split(",")[1]), float(d.split(",")[2]), 0.0, 0.0]
            labels = []
            for entry in detections[d]:   
                if entry[1] >= self.min_conf and (entry[0] in white_list or len(white_list) == 0):
                    labels.append({"species": entry[0].split("_")[1], "certainty": float(entry[1])*100, "filter": "BirdNET-Lite" if self.lite else "BirdNET-Analyzer", "calltype": "non-specified"})

            if len(labels) > 0:
                seg.append(labels)
                output.append(seg)

        if len(output) > 1:
            with open(file + ".data", "w") as rfile:
                json.dump(output, rfile)

                #     # Save as selection table
#     try:

#         # We have to check if output path is a file or directory
#         if not cfg.OUTPUT_PATH.rsplit('.', 1)[-1].lower() in ['txt', 'csv']:

#             rpath = fpath.replace(cfg.INPUT_PATH, '')
#             rpath = rpath[1:] if rpath[0] in ['/', '\\'] else rpath

#             # Make target directory if it doesn't exist
#             rdir = os.path.join(cfg.OUTPUT_PATH, os.path.dirname(rpath))
#             if not os.path.exists(rdir):
#                 os.makedirs(rdir, exist_ok=True)

#             if cfg.RESULT_TYPE == 'table':
#                 rtype = '.BirdNET.selection.table.txt' 
#             elif cfg.RESULT_TYPE == 'audacity':
#                 rtype = '.BirdNET.results.txt'
#             else:
#                 rtype = '.BirdNET.results.csv'
#             saveResultFile(results, os.path.join(cfg.OUTPUT_PATH, rpath.rsplit('.', 1)[0] + rtype), fpath)
#         else:
#             saveResultFile(results, cfg.OUTPUT_PATH, fpath)        
#     except:

#         # Print traceback
#         print(traceback.format_exc(), flush=True)

#         # Write error log
#         msg = 'Error: Cannot save result for {}.\n{}'.format(fpath, traceback.format_exc())
#         print(msg, flush=True)
#         writeErrorLog(msg)
#         return False

#     delta_time = (datetime.datetime.now() - start_time).total_seconds()
#     print('Finished {} in {:.2f} seconds'.format(fpath, delta_time), flush=True)

#     return True

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
            if (model[3][i] in white_list) or (len(white_list) == 0 and model[3][i] not in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']):
                detections[model[3][i]] = j
            i += 1
        return detections
    
    def analyze(self, filelist, white_list):

    #     # Get file path and restore cfg
    #     fpath = item[0]
    #     cfg.setConfig(item[1])    

        try:
            model = self.loadModel()            
            for file in filelist:
                # Read audio data
                audioData = self.readAudioData(file)

            #     # If no chunks, show error and skip
            #     if len(audioData) == 0:
            #         msg = 'Error: Cannot open audio file {}'.format(fpath)
            #         print(msg, flush=True)
            #         writeErrorLog(msg)
            #         return False

                # Process audio data and get detections
                pp_det, timestamps, def_det = self.analyzeAudioData(audioData, file, model)

                if self.mea is False:
                    self.writeAvianzOutput(def_det, file, white_list)
                
                elif self.mea is True:
                # apply moving exponential average to pp_det and write results to tempfile
                    mea_det = self.whiteListing(self.movingExpAverage(pp_det), white_list, model)
                    mea_det_convert = self.convert_mea_output(mea_det, file, timestamps)
                    self.writeAvianzOutput(mea_det_convert, file, white_list)
        except:
            print(traceback.format_exc())
   
           
    def main(self):

        try:
            # create list of filenames
            filelist = [file.absoluteFilePath() for file in self.AviaNZ.listFiles.listOfFiles if file.isFile()]

            # create list of lists of filenames to pass to different threads
            step = -(-len(filelist)//self.threads)
            file_threads = [filelist[i:i + step] for i in range(0, len(filelist), step)]

            # run analyze on different threads
            with concurrent.futures.ThreadPoolExecutor() as executer:
                futures = [executer.submit(self.analyze, flist, copy.deepcopy(self.slist)) for flist in file_threads]

            # self.analyze(filelist, self.slist)
            # TODO: Bug: loadFile loads old annotation file if exists! 
            self.AviaNZ.loadFile(name = self.AviaNZ.filename)
            self.AviaNZ.fillFileList(self.AviaNZ.SoundFileDir, os.path.basename(self.AviaNZ.filename))
        except:
            print(traceback.format_exc())
        #     # Freeze support for excecutable
        #     freeze_support()


        # TODO: Error log needed?
        # ERROR_LOG_FILE = 'error_log.txt'

        # TODO: introduce Codes
        #     # Load eBird codes, labels
        #     CODES_FILE = 'eBird_taxonomy_codes_2021E.json'
        #     cfg.CODES = loadCodes()

        # TODO: include following option
        #     # Set number of threads
        #     if os.path.isdir(cfg.INPUT_PATH):
        #         cfg.CPU_THREADS = max(1, int(args.threads))
        #         cfg.TFLITE_THREADS = 1
        #     else:
        #         cfg.CPU_THREADS = 1
        #         cfg.TFLITE_THREADS = max(1, int(args.threads))

        #     # Add config items to each file list entry.
        #     # We have to do this for Windows which does not
        #     # support fork() and thus each process has to
        #     # have its own config. USE LINUX!
        #     flist = []
        #     for f in cfg.FILE_LIST:
        #         flist.append((f, cfg.getConfig()))

        #     # Analyze files   
        #     if cfg.CPU_THREADS < 2:
        #         for entry in flist:
        #             analyzeFile(entry)
        #     else:
        #         with Pool(cfg.CPU_THREADS) as p:
        #             p.map(analyzeFile, flist)

# class BirdNETAnalyzer():

#     def __init__(self, parent=None):
#         pass

# from multiprocessing import Pool, freeze_support

# def clearErrorLog():

#     if os.path.isfile(cfg.ERROR_LOG_FILE):
#         os.remove(cfg.ERROR_LOG_FILE)

# def writeErrorLog(msg):

#     with open(cfg.ERROR_LOG_FILE, 'a') as elog:
#         elog.write(msg + '\n')

# def loadCodes():

#     with open(cfg.CODES_FILE, 'r') as cfile:
#         codes = json.load(cfile)

#     return codes

# def saveResultFile(r, path, afile_path):

#     # Make folder if it doesn't exist
#     if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
#         os.makedirs(os.path.dirname(path))

#     # Selection table
#     out_string = ''

#     if cfg.RESULT_TYPE == 'table':

#         # Raven selection header
#         header = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies Code\tCommon Name\tConfidence\n'
#         selection_id = 0

#         # Write header
#         out_string += header
        
#         # Extract valid predictions for every timestamp
#         for timestamp in getSortedTimestamps(r):
#             rstring = ''
#             start, end = timestamp.split('-')
#             for c in r[timestamp]:
#                 if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
#                     selection_id += 1
#                     label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
#                     rstring += '{}\tSpectrogram 1\t1\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}\n'.format(
#                         selection_id, 
#                         start, 
#                         end, 
#                         150, 
#                         12000, 
#                         cfg.CODES[c[0]], 
#                         label.split('_')[1], 
#                         c[1])

#             # Write result string to file
#             if len(rstring) > 0:
#                 out_string += rstring

#     elif cfg.RESULT_TYPE == 'audacity':

#         # Audacity timeline labels
#         for timestamp in getSortedTimestamps(r):
#             rstring = ''
#             for c in r[timestamp]:
#                 if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
#                     label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
#                     rstring += '{}\t{}\t{:.4f}\n'.format(
#                         timestamp.replace('-', '\t'), 
#                         label.replace('_', ', '), 
#                         c[1])

#             # Write result string to file
#             if len(rstring) > 0:
#                 out_string += rstring

#     elif cfg.RESULT_TYPE == 'r':

#         # Output format for R
#         header = 'filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity,min_conf,species_list,model'
#         out_string += header

#         for timestamp in getSortedTimestamps(r):
#             rstring = ''
#             start, end = timestamp.split('-')
#             for c in r[timestamp]:
#                 if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):                    
#                     label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
#                     rstring += '\n{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}'.format(
#                         afile_path,
#                         start,
#                         end,
#                         label.split('_')[0],
#                         label.split('_')[1],
#                         c[1],
#                         cfg.LATITUDE,
#                         cfg.LONGITUDE,
#                         cfg.WEEK,
#                         cfg.SIG_OVERLAP,
#                         (1.0 - cfg.SIGMOID_SENSITIVITY) + 1.0,
#                         cfg.MIN_CONFIDENCE,
#                         cfg.SPECIES_LIST_FILE,
#                         os.path.basename(cfg.MODEL_PATH)
#                     )
#             # Write result string to file
#             if len(rstring) > 0:
#                 out_string += rstring

#     else:

#         # CSV output file
#         header = 'Start (s),End (s),Scientific name,Common name,Confidence\n'

#         # Write header
#         out_string += header

#         for timestamp in getSortedTimestamps(r):
#             rstring = ''
#             for c in r[timestamp]:                
#                 start, end = timestamp.split('-')
#                 if c[1] > cfg.MIN_CONFIDENCE and c[0] in cfg.CODES and (c[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
#                     label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
#                     rstring += '{},{},{},{},{:.4f}\n'.format(
#                         start,
#                         end,
#                         label.split('_')[0],
#                         label.split('_')[1],
#                         c[1])

#             # Write result string to file
#             if len(rstring) > 0:
#                 out_string += rstring

#     # Save as file
#     with open(path, 'w') as rfile:
#         rfile.write(out_string)

# def getSortedTimestamps(results):
#     return sorted(results, key=lambda t: float(t.split('-')[0]))





