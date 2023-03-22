
# import statements for BirdNET-Lite

import os
from tensorflow import lite as tflite
import operator
import librosa
import numpy as np
import math
import time
import glob
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import copy
import sys
import json
import traceback
from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QSlider, QGridLayout, QGridLayout, QLabel, QComboBox, QHBoxLayout, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QCheckBox, QFileDialog, QMessageBox

import AviaNZ_manual

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class BirdNETDialog(QDialog):

    def __init__(self, parent=None):
        super(BirdNETDialog, self).__init__(parent)
        self.parent = parent

        self.slist_path = ""
        self.setWindowTitle("Classify Recordings with BirdNET")
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

        lbltitle = QLabel("To analyze the audiofiles of the current directory, set the parameters for BirdNET-Lite or BirdNET-Analyzer. Be aware that the process might take several hours, depending on the number of audiorecordings, calculation power and number of threads")
        lbltitle.setWordWrap(True)

        # BirdNET-Lite/BirdNET-Analyzer options
        self.lite = QRadioButton("BirdNET-Lite…")
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
        species_list = QFileDialog.getOpenFileName(self, 'Choose filter species list', filter='Text (*.txt)')
        self.slist.setText(os.path.basename(species_list[0]))
        self.slist_path = species_list[0]

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
                                            self.slist_path,
                                            self.threads.text(),
                                            self.mea.isChecked(),
                                            self.datetime_format.text(),
                                            self.locale.currentText(),
                                            self.batchsize.text(),
                                            self.sf_thresh.text())
            self.parent.BirdNET.main()
            self.parent.loadFile(name=self.parent.filename)
            self.parent.fillFileList(
              self.parent.SoundFileDir,
              os.path.basename(self.parent.filename)
              )
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

MODEL = None
M_INTERPRETER = None
SLIST = None
M_INPUT_LAYER_INDEX = None
M_OUTPUT_LAYER_INDEX = None

class BirdNET():
    def __init__(self, AviaNZmanual):
        # self.AviaNZ = AviaNZmanual
        self.filelist = [file.absoluteFilePath() for file in AviaNZmanual.listFiles.listOfFiles if file.isFile()]

        self.operator = AviaNZmanual.operator
        self.reviewer = AviaNZmanual.reviewer
        self.m_interpreter = None
        self.model = None       

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
        print(slist)
        # self.slist = self.getSpeciesList(slist)
        self.slist = None
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
        global M_INTERPRETER
        global M_INPUT_LAYER_INDEX
        global M_OUTPUT_LAYER_INDEX
        print("load MetaModel", flush = True)
        # Load TFLite model and allocate tensors.
        M_INTERPRETER = tflite.Interpreter(model_path=os.path.join('models', 'Analyzer', 'BirdNET_GLOBAL_3K_V2.2_MData_Model_FP16.tflite'))
        M_INTERPRETER.allocate_tensors()

        # Get input and output tensors.
        input_details = M_INTERPRETER.get_input_details()
        output_details = M_INTERPRETER.get_output_details()

        # Get input tensor index
        M_INPUT_LAYER_INDEX = input_details[0]['index']
        M_OUTPUT_LAYER_INDEX = output_details[0]['index']

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
        global M_INTERPRETER
        global M_INPUT_LAYER_INDEX
        global M_OUTPUT_LAYER_INDEX
        # Does interpreter exist?
        if M_INTERPRETER is None:
            self.loadMetaModel()

        # Prepare mdata as sample
        sample = np.expand_dims(np.array([self.lat, self.lon, self.week], dtype='float32'), 0)

        # Run inference
        M_INTERPRETER.set_tensor(M_INPUT_LAYER_INDEX, sample)
        M_INTERPRETER.invoke()

        return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER_INDEX)[0]

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
        if amount is None:
            amount = random.uniform(0.1, 0.5)

        # Create Gaussian noise
        try:
            noise = random.normal(min(sig) * amount, max(sig) * amount, shape)
        except:
            noise = np.zeros(shape)

        return noise.astype('float32')

    def readAudioData(self, path, sample_rate=48000):

        print('READING AUDIO DATA FROM FILE {}...'.format(os.path.split(path)[1]), end=' ', flush=True)

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

    def predict(self, samples):
        global MODEL
        interpreter = MODEL[4]
        input_layer_index = MODEL[0]
        mdata_input_index = MODEL[1]
        output_layer_index = MODEL[2]
        # labels = model[3]

        if self.lite:
            # Make a prediction
            interpreter.set_tensor(input_layer_index, np.array(samples[0], dtype='float32'))
            interpreter.set_tensor(mdata_input_index, np.array(samples[1], dtype='float32'))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_layer_index)[0]

            # Apply custom sigmoid

            p_sigmoid = self.custom_sigmoid(prediction)

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

            p_sigmoid = self.flat_sigmoid(np.array(prediction), sensitivity=-self.sensitivity)

        return p_sigmoid

    def analyzeAudioData(self, chunks, file):
        global MODEL

        # different format for standard and post-processing (mea) approach
        detections = {}
        detections_mea = np.zeros(shape=(len(chunks), len(MODEL[3])))

        start = time.time()
        print('ANALYZING AUDIO FROM {} ...'.format(os.path.basename(file)), end=' ', flush=True)

        # Parse every chunk
        timestamps = []
        pred_start = 0.0
        sig_length = 3.0

        labels = MODEL[3]

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

                p_sigmoid = self.predict([sig, mdata])

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
                p = self.predict(samples)

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
                    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

                    # Store top 5 results and advance indicies
                    detections[file + ',' + str(s_start) + ',' + str(s_end)] = p_sorted

                # Clear batch
                samples = []
                timestamps = []

            return (detections_mea.transpose(), timestamps_return, detections)

    def convert_mea_output(self, mea_output, file, timestamps):
        detections = {}
        cntr = 0
        for start in timestamps:
            key = "{},{},{}".format(file, start, start + 3)
            detections[key] = [(d, mea_output[d][cntr]) for d in mea_output]
            cntr += 1

        return detections

    def writeAvianzOutput(self, detections, file, white_list, append=True):
        # TODO: get Duration from file
        rfilepath = file + ".data"
        if append and os.path.exists(rfilepath):
            with open(rfilepath, "r") as infile:
                output = json.load(infile)
            
            for d in detections:
                start = float(d.split(",")[1])
                end = float(d.split(",")[2])
                seg_index = -1
                for e in output[1:]:
                    if e[0] == start and e[1] == end:
                        seg_index = output.index(e)
                        break
                
                if seg_index >= 0:
                    seg = output[seg_index]
                    labels = output[seg_index][4]
                    for entry in detections[d]:
                        if entry[1] >= self.min_conf and (entry[0] in white_list or len(white_list) == 0):
                            output[seg_index][4].append({"species": entry[0].split("_")[1], "certainty": float(entry[1])*100, "filter": "BirdNET-Lite" if self.lite else "BirdNET-Analyzer", "calltype": "non-specified"})
                        
                else:
                    seg = [float(d.split(",")[1]), float(d.split(",")[2]), 0.0, 0.0]
                    labels = []
                    for entry in detections[d]:
                        if entry[1] >= self.min_conf and (entry[0] in white_list or len(white_list) == 0):
                            labels.append({"species": entry[0].split("_")[1], "certainty": float(entry[1])*100, "filter": "BirdNET-Lite" if self.lite else "BirdNET-Analyzer", "calltype": "non-specified"})

                    if len(labels) > 0:
                        seg.append(labels)
                        output.append(seg)
                    
        else:
            output = [{"Operator": self.operator, "Reviewer": self.reviewer, "Duration": 60}]
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
            with open(rfilepath, "w") as rfile:
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

    def whiteListing(self, timetable, white_list):
        global MODEL
        detections = {}
        i = 0
        for j in timetable:
            if (MODEL[3][i] in white_list) or (len(white_list) == 0 and MODEL[3][i] not in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']):
                detections[MODEL[3][i]] = j
            i += 1
        return detections

    def analyze(self, file):
        global MODEL
        global SLIST
        try:
            white_list = SLIST
        
            # Read audio data
            audioData = self.readAudioData(file)

            #     # If no chunks, show error and skip
            #     if len(audioData) == 0:
            #         msg = 'Error: Cannot open audio file {}'.format(fpath)
            #         print(msg, flush=True)
            #         writeErrorLog(msg)
            #         return False

            # Process audio data and get detections
            pp_det, timestamps, def_det = self.analyzeAudioData(audioData, file)

            if self.mea is False:
                self.writeAvianzOutput(def_det, file, white_list)

            elif self.mea is True:
                # apply moving exponential average to pp_det and write results to tempfile
                mea_det = self.whiteListing(self.movingExpAverage(pp_det), white_list)
                mea_det_convert = self.convert_mea_output(mea_det, file, timestamps)
                self.writeAvianzOutput(mea_det_convert, file, white_list)
        except:
            print(traceback.format_exc())

    def initProcess(self):
        global MODEL
        global SLIST
        MODEL = self.loadModel()
        SLIST = self.getSpeciesList(self.slist)

    def main(self):

        try:
            # create list of filenames
            # filelist = [file.absoluteFilePath() for file in self.AviaNZ.listFiles.listOfFiles if file.isFile()]

            # create list of lists of filenames to pass to different threads
            # step = -(-len(filelist)//self.threads)
            # file_threads = [filelist[i:i + step] for i in range(0, len(filelist), step)]

            # run analyze on different threads
            # with concurrent.futures.ThreadPoolExecutor() as executer:
            #     futures = [executer.submit(self.analyze, flist, copy.deepcopy(self.slist)) for flist in file_threads]
            # with Pool(self.threads) as p:
            #     p.map(self.analyze, self.filelist)
            with ProcessPoolExecutor(self.threads, initializer=self.initProcess) as executer:
                executer.map(self.analyze, self.filelist)

            # self.analyze(filelist, self.slist)
            

        except:
            print(traceback.format_exc())
