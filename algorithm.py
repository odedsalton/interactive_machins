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


def init_transition_perf(M,z_s,n_bins,R_s):
    weight_vector = [1/(len(z_s)+n_bins)]*100
    # print(len(weight_vector))
    lamda = 24
    ####################### calculate M star #########################
    M_star = calculate_m_star(M,R_s,len(M))
    M_star = M_star.drop(['Reward'], axis=1)
    ####################### calculate lamda mediods ###################
    data = prepering_data_mediods(M_star)
    representatives, clusters = lamda_medioids(data,lamda)
    ####################### extracting H and calculate the vector #####
    H = list(representatives.keys())
    matrix = calculate_transition_matrix(H,M_star)
    song = np.asarray([matrix[0, :]])
    for i in range(9):
        song1 = np.asarray([matrix[i+1,:]])
        weight_vector += [1 / (len(z_s) + n_bins)]*(song1*song)
        song = song1
    return weight_vector

def model_update(M,song,k,v_k,v_avg,r_s,r_t,weights_s,weights_t,prev_song):
    if v_avg == 0:
        v_inc = 0
    elif v_k == 0:
        v_inc = 0
    else:
        x = v_k/v_avg
        v_inc = math.log(x)
    ############################# update  ##################################################
    song_features = extract_song_variables(M,song)
    prev_song_features = extract_song_variables(M,prev_song)
    W_s = float(r_s/(r_s + r_t))
    W_t = float(r_t/(r_s + r_t))
    weights_s = (k/k+1)*weights_s + (1/(k+1)) * song_features * v_inc * W_s
    weights_t = (k/k+1)*weights_t + (1/(k+1)) * (prev_song_features*song_features) * v_inc *W_t
    ############################# normalization ##################################################
    weights_s = weights_s / np.linalg.norm(weights_s)
    weights_t = weights_t / np.linalg.norm(weights_t)
    return weights_s, weights_t
def planning_Tree_search(M,q,R_s,R_t,B):
    M = M.drop(['Reward'], axis=1)
    M_star = calculate_m_star(M,R_s,B)
    traj = M_star.sample(n=10)
    transitions_matrix = np.matrix(R_t).transpose()
    Best_traj = []
    Highest_payoff = -100000
    H = list(M_star['title'])
    num_iter = 100
    song_reward = 0
    transition_reward = 0
    for j in range(num_iter):
        # print("iter",j)
        traj = M_star.sample(n=20)
        traj = np.matrix(traj)
        song_total_reward = 0
        total_transition_reward = 0
        for i in range(q-1):
            song_total_reward += traj[i, 103]
            transition = np.dot(traj[i, 3:103].transpose(), traj[i+1, 3:103])
            song_transition_reward = np.dot(transition.transpose(), transitions_matrix)
            total_transition_reward += sum(song_transition_reward/100)
        exp_traj = (song_total_reward + total_transition_reward)
        if exp_traj > Highest_payoff:
            Highest_payoff = exp_traj
            song_reward = traj[0, 103]
            transition_reward = total_transition_reward
            Best_traj = traj[:, 2]
    return Best_traj[0], song_reward, transition_reward
def dj_mc_framework(M,q,z_s,z_t,nbins_s,n_bins_t,B):
    songs_weight_vector = init_song_perf(M,z_s,nbins_s)
    songs_transition_weight_matrix = init_transition_perf(M,z_t,n_bins_t,songs_weight_vector)
    k = 0
    songs_dict = dict()
    v_list = list()
    prev_song = []
    while k < q:
        k += 1
        song,song_reward,transtion_reward = planning_Tree_search(M,q-k,songs_weight_vector,songs_transition_weight_matrix,B)
        print("{} is the next song in list - press 1 for like and 0 for dislike".format(str(song).strip('[]')))
        v = Gui.gui(str(song).strip('[]'))
        v = int(v)
        songs_dict['song{}'.format(k)] = v
        v_list.append(int(v))
        v_avg = sum(v_list)/len(v_list)
        songs_weight_vector,songs_transition_weight_matrix = \
            model_update(M,song,k,v,v_avg,song_reward,transtion_reward,songs_weight_vector,songs_transition_weight_matrix,prev_song)
        prev_song = song
    return songs_dict
def prepering_data_mediods(M_star):
    new_df = pd.DataFrame()
    list1 = list([1,2,3,4,5,6,7,8,9,10])
    list2 = list(['title','bpm','nrgy','dnce','dB','live','val','dur','acous','spch','pop'])
    data_dict = dict()
    for element in list2:
        temp = pd.DataFrame(columns=[element])
        new_df = pd.concat([temp, new_df], axis=1)
    for index,row in M_star.iterrows():
        new_df = new_df.append(pd.Series(0, index=new_df.columns), ignore_index=True)
        for element in list2:
            if element == 'title':
                new_df[element][index] = row['title']
                continue
            for number in list1:
                if row[element+str(number)] == 1:
                    new_df[element][index] = number
                    break
                else:
                    continue
    list2 = list(['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop'])
    for index,row in new_df.iterrows():
        data_dict[row['title']] = list()
        for element in list2:
            data_dict[row['title']].append(row[element])
    return data_dict


def lamda_medioids(data,lamda):
    t = 0
    representatives = dict()
    clusters = dict()
    while t < 5:
        t += 1
        representatives,clusters = lamda_subroutine(data,lamda,representatives,clusters)

    return representatives, clusters


def lamda_subroutine(data,lamda,representitives,cluster):
    representitive = 0
    representitives1 = dict()
    counter = 1
    for key,value in data.items():
        dist = 1000000
        for name,rep in representitives.items():
            if key == name:
                continue
            if calc_distance(value,rep) < dist:
                representitive = name
                # cluster[representitive] = list()
                dist = calc_distance(value,rep)
        if dist <= lamda:
            cluster[representitive].append(key)
        else:
            representitive = key
            cluster[representitive] = list()
            cluster[representitive].append(key)
            representitives[representitive] = value

    return representitives, cluster


def calc_distance(val1,val2):
    dist = 0
    for i in range(len(val1)):
        dist += abs(val1[i] - val2[i])
    return dist


def adding_class(M_star,clusters):
    M_star['cluster'] = 0
    counter = 0
    for key,value in clusters.items():
        counter += 1
        for val in value:
            for index,row in M_star.iterrows():
                if val == row['title']:
                    M_star['cluster'][index] = counter
    return M_star


def find_location(matrix,song):
    names = matrix[:,2].copy()
    for i in range(len(names)):
        if song == names[i]:
            return i
    return 0


def calculate_transition_matrix(H,M):
    matrics = np.matrix(M)
    names = matrics[:2].copy()
    song = random.sample(H, 1)
    number = find_location(matrics, song)
    tmp = matrics[number, 3:]
    transition_matrics = tmp.copy()
    for i in range(9):
        song = random.sample(H,1)
        number = find_location(matrics,song)
        tmp = matrics[number,3:]
        transition_matrics = np.vstack((transition_matrics, tmp))
    return transition_matrics


def extract_song_variables(M,song):
    matrics = np.matrix(M)
    names = matrics[:2].copy()
    number = find_location(matrics, song)
    value = matrics[number, 3:103]
    return np.asarray(value).reshape(-1)


def calculate_m_star(M,songs_weights,B):
    R = list()
    M_set = np.asarray(M)
    ######################################### calcuating rewards ###########################
    for song in M_set:
        R.append(np.dot(song[2:],songs_weights))
    M['Reward'] = songs_weights
    ######################################### calcuating M_star according to 50 prcent #####
    M = M.sort_values(by=['Reward'], ascending=False)
    M = M.reset_index()
    M_star = M
    M = M.drop(['Reward'], axis=1)
    M.drop(M.tail(B).index, inplace=True)
    return M_star
