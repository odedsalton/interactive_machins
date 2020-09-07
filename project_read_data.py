import pandas as pd
import preprocessing_functions as pf
import algorithm as al
import greedy_algorithm as gr
def main():
    # data_frame_songs = c.read_csv('song_hash.txt')
    songs_data = pd.read_csv('songs_features.csv')
    songs = ['song10','song9','song8','song7','song6','song5','song4','song3','song2','song1']
    df_greedy = pf.creating_user_dataframe(songs)
    df_our = pf.creating_user_dataframe(songs)
    for i in range(20):
        M = songs_data.sample(n=100)
        M.reset_index()
        z = M.sample(n=10)
        n_bins = 10
        songs_dict2 = gr.dj_mc_framework(M,10,z,z,10,10,50)
        print("songs dict", songs_dict2)
        df_greedy = pf.appending_series(df_greedy,songs_dict2,i)
        songs_dict = al.dj_mc_framework(M,10,z,z,10,10,50)
        df_our = pf.appending_series(df_our,songs_dict,i)
        print("songs dict", songs_dict)
    df_greedy.to_csv('greedy_data.csv')
    df_our.to_csv('DJMC_data.csv')


    return

if __name__ == '__main__':
    main()