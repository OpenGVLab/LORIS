import glob, os, sys
import soundfile as sf
import librosa
from librosa.core import load
import numpy as np
import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", default='./OUTPUT/loris_fe_s50/samples_min_20_50')
	args = parser.parse_args()
	return args


def beat_detect(x, sr=22050):
	onsets = librosa.onset.onset_detect(x, sr=sr, wait=1, delta=0.2, pre_avg=3, post_avg=3, pre_max=3, post_max=3, units='time')
	n = np.ceil(len(x) / sr)
	beats = [0] * int(n)
	for time in onsets:
		beats[int(np.trunc(time))] = 1
	return beats

def beat_scores(gt, syn):
	gt = gt[:len(syn)]
	assert len(gt) == len(syn)
	total_beats = sum(gt)
	cover_beats = sum(syn)
	hit_beats = 0
	for i in range(len(gt)):
		if gt[i] == 1 and gt[i] == syn[i]:
			hit_beats += 1
	print(f"Total Beats: {total_beats}, Cover Beats: {cover_beats}, Hit Beats: {hit_beats}")
	hit_rate = hit_beats/total_beats if total_beats else 0 
	cover_rate = hit_beats/cover_beats if cover_beats else 0 
	return cover_rate, hit_rate 


if __name__ == '__main__':
	args = parse_args()
	path = args.path
	ref_music = sorted(glob.glob(path+'/gt*'))
	syn_music = sorted(glob.glob(path+'/generated*'))
	total_score_cover = 0
	total_score_hit = 0
	hit_ls = []
	cover_ls = []
	print(f"ref_music num: {len(ref_music)}, syn_music num: {len(syn_music)}")
	for i, c in enumerate(ref_music):
		ref, _ = load(ref_music[i])
		syn, _ = load(syn_music[i])
		gt_beats = beat_detect(ref)
		syn_beats = beat_detect(syn)
		score_cover, score_hit = beat_scores(gt_beats, syn_beats)
		total_score_cover += score_cover
		total_score_hit += score_hit
		cover_ls.append(score_cover)
		hit_ls.append(score_hit)
		print("Inference:", score_cover, score_hit)
	cover_score, hit_score = total_score_cover/len(ref_music), total_score_hit/len(ref_music)
	cover_std, hit_std = np.std(cover_ls, ddof=1), np.std(hit_ls, ddof=1)
	f1_score = 2 * cover_score * hit_score / (cover_score + hit_score)
	print(f"Cover Score: {cover_score}, Cover Std: {cover_std}, Hit Score: {hit_score}, Hit Std: {hit_std}, F1 Score: {f1_score}")
