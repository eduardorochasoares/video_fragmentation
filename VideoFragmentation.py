import sys

from document_similarity.DocSim import DocSim

from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize, pos_tag, ne_chunk, tokenize
import nltk
from nltk import RegexpParser
from nltk import Tree
from nltk.corpus import stopwords

import pandas as pd
from video_fragmentation_Galanopoulos_et_al.TextWindow import TextWindow
import numpy as np
import matplotlib.pyplot as plt
import evaluate_method
from nltk.tag import StanfordPOSTagger
import re
import os
import glob
nltk.internals.config_java(options='-xmx2G')

class VideoFragmentation:

    def __init__(self, sizeof_text_windows, k_cuts, docSim, st, video_path):

        self.docSim = docSim
        self.video_path = video_path
        self.sizeof_text_windows = sizeof_text_windows
        self.stopwords = set(stopwords.words('english'))
        self.text_windows = []
        self.signal = []
        self.k_cuts = k_cuts
        self.n_chunks = len(glob.glob(self.video_path+"/chunks/chunk*"))
        self.threshold = 0
        self.st = st
        java_path = "C:/Program Files/Java/jdk1.8.0_201/bin/java.exe"
        os.environ['JAVAHOME'] = java_path
        nltk.internals.config_java(options='-Xmx3096m')



    def _tag_whole_text(self, text):
        return self.st.tag(word_tokenize(text))
    def __extract_noun_phrases__(self, tags):

        NP = r"""
                          NBAR:
                              {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                          NP:
                              {<NBAR>}
                              {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                      """
        chunker = RegexpParser(NP)

        if tags:

            df = pd.DataFrame({'text': [tags]})

            np = df['text'].apply(lambda sent: self.__get_continuous_chunks__(sent, self.st, chunker.parse))[0]
            np = [token.lower() for token in np if token.lower() not in self.stopwords]

            return ' '.join(np)

        else:
            return ''

        '''extract pause duration before being voiced of every audio chunk'''

    def extractTimeStamps(self):
        file_path = self.video_path + "seg.txt"
        file = open(file_path, 'r')
        f = file.read()
        times = []
        timesEnd = []
        pause_list = []
        l = re.findall("\+\(\d*\.\d*\)", f)
        for i in l:
            i = i.replace("+", "")
            i = i.replace("(", "")
            i = i.replace(")", "")
            times.append(float(i))

        return times

    def __get_continuous_chunks__(self, tags, my_tagger, chunk_func=ne_chunk):

        #  chunked = chunk_func(my_tagger.tag(word_tokenize(text)))
        chunked = chunk_func(tags)
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if type(subtree) == Tree:
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk


    def __prepare_text_for_real_videolectures(self):
        times_words = []
        words = []
        for i_chunk in range(self.n_chunks):
            init_times = self.extractTimeStamps()
            text_file = self.video_path + "transcript/transcript" + str(i_chunk) + ".txt"
            with open(text_file, 'r') as f:
                tokens = f.read().split('\n')
                for token in tokens:
                    words.append(token.lower())
                    times_words.append(init_times[i_chunk])

        return words, times_words
    def __prepare_texts_ant_init_times__(self, transcript_path):
        with open(transcript_path, 'r') as tf:
            print(transcript_path)
            lines = tf.readlines()

        transcript = ''
        aux_transcript = ''
        words = []
        init_times = []
        i = 0
        while i < len(lines):
            i += 1
            h_m_s = lines[i].split(',')

            aux = h_m_s[0].split(":")
            seconds = int(aux[0]) * 3600 + int(aux[1]) * 60 + int(aux[2])

            i += 1

            aux_transcript = lines[i].strip()

            aux = aux_transcript.split(" ")

            for w in aux:
                init_times.append(seconds)

            words += aux

            i += 1
            i += 1

        return words, init_times

    def build_text_windows(self, transcript_path):
        self.video_path = transcript_path

        words, init_times = self.__prepare_text_for_real_videolectures()
        if len(words) < self.sizeof_text_windows:
            self.sizeof_text_windows = len(words)
        tags = self._tag_whole_text(" ".join(words))
        window_start_time = []
        word_windows = []
        window_id = 0
        avg_cues = []
        for i in range(0, len(words), int(self.sizeof_text_windows / 6)):
            window_start_time.append(init_times[i])

            if i + self.sizeof_text_windows <= len(words):
                tw = TextWindow(self.__extract_noun_phrases__(tags[i:i + self.sizeof_text_windows]),
                                init_times[i], window_id)

                avg_cues.append(len(tw.cues.split(" ")))
                word_windows.append(tw)
                window_id += 1

        self.text_windows = word_windows

    def build__similarity_signal(self):
        x = np.arange(0, len(self.text_windows) - 1, 1)
        for w in range(0, len(self.text_windows) - 1):
            self.signal.append(self.docSim.calculate_similarity(
                self.text_windows[w].cues, self.text_windows[w + 1].cues)[0]['score'])

        self.signal.append(0)

    def calculate_depths(self):

        self.text_windows[0].depth = 0
        self.text_windows[-1].depth = 0
        local_minima = []

        for i in range(1, len(self.text_windows) - 1):
            # if self.signal[i - 1] > self.signal[i] and self.signal[i + 1] > self.signal[i]:
            peak1 = 0
            peak2 = 0


            self.text_windows[i].depth = (self.signal[i - 1] - self.signal[i]) + \
                                         (self.signal[i + 1] - self.signal[i])


    def find_nearest_peak(self, i, side):

        if side == 'left':
            factor = -1
        else:
            factor = 1

        j = i + factor

        while self.signal[i] > self.signal[j]:
            j += factor
            if j > len(self.signal) or j < 0:
                return -1

        return j



    def get_k_best_points(self):

        # bounds = filter(lambda x: x.depth > self.threshold, self.text_windows)
        sorted_text_windows = sorted(self.text_windows, key=lambda x: x.depth, reverse=True)
        selected = sorted_text_windows[0:self.k_cuts]
        bounds = sorted([x.id for x in selected])

        # print(bounds)

        return bounds


def load_word2vec_model():
    stopwords_path = "document_similarity/data/stopwords_en.txt"
    with open(stopwords_path, 'r') as fh:
        stopwords = fh.read().split(",")

    model_path = 'document_similarity/data/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=1000000)
    return DocSim(model, stopwords=stopwords)



'''if __name__ == '__main__':
    p_mean = []
    r_mean = []
    f_mean = []
    docSim = load_word2vec_model()
    st = StanfordPOSTagger('pos_tagger_models/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger',
                          'pos_tagger_models/stanford-postagger-2018-10-16/stanford-postagger.jar')

    for index in range(1, 301):
        video_frag = VideoFragmentation(sizeof_text_windows=840, k_cuts=40, docSim=docSim, st=st)
    
        video_frag.build_text_windows("dataset/ALV_srt/" + str(
            index).zfill(4) + ".srt")
        video_frag.build__similarity_signal()
        video_frag.calculate_depths()
        boundaries = video_frag.get_k_best_points()
        p, r, f = evaluate_method.evaluate_str_only(boundaries,
                                                    'dataset/ALV_srt_GT/' + str(
                                                        index).zfill(4) + ".txt", video_frag.text_windows)
        p_mean.append(p)
        r_mean.append(r)
        f_mean.append(f)

    print("Precision: " + str(np.mean(p_mean)))
    print("Recall: " + str(np.mean(r_mean)))
    print("F-measure: " + str(np.mean(f_mean)))'''

if __name__ == '__main__':
    p_mean = []
    r_mean = []
    f_mean = []
    docSim = load_word2vec_model()
    st = StanfordPOSTagger('pos_tagger_models/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger',
                          'pos_tagger_models/stanford-postagger-2018-10-16/stanford-postagger.jar', java_options='-Xmx6096m')

    root_database = r'C:\Users\dudur\OneDrive\Área de Trabalho\videolecture_net'
    dirlist = [item for item in os.listdir(root_database) if os.path.isdir(os.path.join(root_database, item))]
    for d in dirlist:
        video_frag = VideoFragmentation(sizeof_text_windows=840, k_cuts=25, docSim=docSim, st=st,
                                        video_path=root_database + "/" + d + "/")
        video_frag.build_text_windows(root_database + '/' + d + '/')
        video_frag.build__similarity_signal()
        video_frag.calculate_depths()
        boundaries = video_frag.get_k_best_points()
        p, r, f = evaluate_method.evaluate_aux(video_frag.video_path, boundaries, "ground_truth_segmentation/gt_" + video_frag.video_path.split("/")[-2] + ".json", video_frag.text_windows)

        p_mean.append(p)
        r_mean.append(r)
        f_mean.append(f)

    with open('result_videofragmentation_k25.txt', 'w') as outfile:
       outfile.write('Precision: ' + str(np.mean(p_mean)) + "\n")
       outfile.write('Precision Std: ' + str(np.std(p_mean)) + "\n")
       outfile.write('Recall: ' + str(np.mean(r_mean)) + "\n")
       outfile.write('Recall Std: ' + str(np.std(r_mean)) + "\n")
       outfile.write('F1: ' + str(np.mean(f_mean)) + "\n")
       outfile.write('F1 Std: ' + str(np.std(f_mean)) + "\n")






