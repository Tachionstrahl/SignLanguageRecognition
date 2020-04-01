# -*- coding: utf-8 -*- 
import os
import sys
import argparse

def main(input_data_path,output_data_path):
	# Kommando für: Building mediapipe
	comp='bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu'
    # Kommando für: MediaPipe ausführen um die Videodateien in Textfiles zu konvertieren
	cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt'
    #Liste aller Verzeichnisse im Input-Ordner
	listfile=os.listdir(input_data_path)
	for file in listfile:
        #Nur Verzeichnisse werden weiterverarbeitet
		if not(os.path.isdir(input_data_path+file)): #ignore .DS_Store - Mac-spezfisches Problem...
			continue
		word=file+'/' # z.B. 'Computer/'
		fullfilename=os.listdir(input_data_path+word) #Ganzer Pfad zu einem Wort-Ordner
        # 하위디렉토리의 모든 비디오들의 이름을 저장 = Speichern Sie die Namen aller Videos im Unterverzeichnis
		# Es werden die Ordner mit _ angelegt, falls noch nicht vorhanden. z.B. '_Computer'
		if not(os.path.isdir(output_data_path+"_"+word)):
			os.mkdir(output_data_path+"_"+word)
		if not(os.path.isdir(output_data_path+word)):
			os.mkdir(output_data_path+word)
		os.system(comp) #MediaPipe Build ausführen
		
		outputfilelist=os.listdir(output_data_path+'_'+word)
		# Für jede 
		for mp4list in fullfilename:
			if ".DS_Store" in mp4list:
				continue         
			inputfilen='   --input_video_path='+input_data_path+word+mp4list
			outputfilen='   --output_video_path='+output_data_path+'_'+word+mp4list
			cmdret=cmd+inputfilen+outputfilen
			os.system(cmdret)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='operating Mediapipe')
	parser.add_argument("--input_data_path",help=" ")
	parser.add_argument("--output_data_path",help=" ")
	args=parser.parse_args()
	input_data_path=args.input_data_path
	output_data_path=args.output_data_path
    #print(input_data_path)
	main(input_data_path,output_data_path)
