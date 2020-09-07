import pandas as pd
import numpy as np
import random
import math
import Gui

def init_song_perf(M,z_s,n_bins):
    weight_vector = [1/(len(z_s)+n_bins)]*100
    z_s.reset_index()
    preferred_set = np.asarray(z_s)
    for song in preferred_set:
        weight_vector += (1/(len(z_s)+n_bins) * song[2:])
    return weight_vector
def planning_Tree_search(M,R_s,k):
    M_star = M.copy()
    M_star = create_rewards(M_star,R_s)
    M_star.drop(M.tail(len(M_star)-k).index, inplace=True)
    return M_star
def dj_mc_framework(M,q,z_s,z_t,nbins_s,n_bins_t,B):
    songs_weight_vector = init_song_perf(M,z_s,nbins_s)
    k = 0
    songs_dict = dict()
    v_list = list()
    M = planning_Tree_search(M,songs_weight_vector,q)
    for index,row in M.iterrows():
        k = k + 1
        song = row['title']
        print("{} is the next song in list - press 1 for like and 0 for dislike".format(str(song).strip('[]')))
        v = Gui.gui(str(song).strip('[]'))
        v = int(v)
        songs_dict['song{}'.format(k)] = v
        v_list.append(int(v))
        v_avg = sum(v_list)/len(v_list)
    return songs_dict
def create_rewards(M,R_s):
    R = list()
    M_set = np.asarray(M)
    ######################################### calcuating rewards ###########################
    for song in M_set:
        R.append(np.dot(song[2:],R_s))
    M['Reward'] = R_s
    ######################################### calcuating M_star according to 50 prcent #####
    M = M.sort_values(by=['Reward'], ascending=False)
    return M