import numpy as np
import pandas as pd
import pickle

# loc_match 불러와서 이름순으로 정렬
with open('loc_match.pkl', 'rb') as f:
    loc_match = pickle.load(f)

for key in loc_match:
    pm_loc = key
    aws_loc = loc_match[key]
    pm_df_train = pd.read_csv('TRAIN/{}.csv'.format(pm_loc))
    aws_df_train = pd.read_csv('TRAIN_AWS/{}.csv'.format(aws_loc))
    pm_df_test = pd.read_csv('TEST_INPUT/{}.csv'.format(pm_loc))
    aws_df_test = pd.read_csv('TEST_AWS/{}.csv'.format(aws_loc))
        
    # date 컬럼 생성
    date_train = list()
    for index, row in pm_df_train.iterrows():
        date_train.append('000' + str(row[0]) + row[1][:2] + row[1][3:5] + row[1][6:8])
    date_train = np.array(date_train)
    
    date_test = list()
    for index, row in pm_df_test.iterrows():
        date_test.append('000' + str(row[0]) + row[1][:2] + row[1][3:5] + row[1][6:8])
    date_test = np.array(date_test)
    
    
    # 하나의 DataFrame으로 합치기
    df_train = pd.concat([pm_df_train, aws_df_train[aws_df_train.columns[3:]]], axis=1)
    df_test = pd.concat([pm_df_test, aws_df_test[aws_df_test.columns[3:]]], axis=1)
    
    # date 컬럼 추가
    df_train.insert(loc=0, column='date', value = date_train)
    df_test.insert(loc=0, column='date', value = date_test)
    
    # column 이름 간단하게 변경
    df_train.rename(columns={'기온(°C)':'기온', '풍향(deg)':'풍향', '풍속(m/s)':'풍속', '강수량(mm)':'강수량', '습도(%)':'습도'}, inplace=True)
    df_test.rename(columns={'기온(°C)':'기온', '풍향(deg)':'풍향', '풍속(m/s)':'풍속', '강수량(mm)':'강수량', '습도(%)':'습도'}, inplace=True)
    
    # 전의 값으로 결측치 처리
    # df_train.fillna(method='ffill', inplace=True)
    # print('{}: {}'.format(pm_loc, df_train.isna().sum()))
    # print('-------------------------------------------------------')
    # df_train.fillna(method='bfill', inplace=True)
    # print('{}: {}'.format(pm_loc, df_train.isna().sum()))
    
    # 강수량은 0으로, 나머지는 평균으로 결측치 처리
    cols = ['기온', '풍향', '풍속', '습도', 'PM2.5']
    for col in cols:
        mean = df_train[col].mean(skipna=True)
        df_train[col].replace(np.nan, mean, inplace=True)
    df_train['강수량'].replace(np.nan, 0, inplace=True)
        
    # 필요없는 column 제거
    df_train.drop(columns=['연도', '일시', '측정소'], inplace=True)
    df_test.drop(columns=['연도', '일시', '측정소'], inplace=True)
    
    # Model의 Input형태로 column 순서 변경
    df_train = df_train[['date', '기온', '풍향', '풍속', '강수량', '습도', 'PM2.5']]
    df_test = df_test[['date', '기온', '풍향', '풍속', '강수량', '습도', 'PM2.5']]
    
    
    cols = ['date', '기온', '풍향', '풍속', '강수량', '습도', 'PM2.5']
        
    # df_test[cols].to_csv('NLinear_LSTM/data/{}test.csv'.format(pm_loc))
    
    df_test[cols].to_csv('INFORMER2020/data/{}test.csv'.format(pm_loc))
    
    # 첫 test_input은 처음 2틀간의 test_input + train
    df_input = pd.concat([df_train, df_test[:48]], axis=0, ignore_index=True)
    
    # df_train[cols].to_csv('NLinear_LSTM/data/{}train.csv'.format(pm_loc))
    # df_input[cols].to_csv('NLinear_LSTM/data/{}input.csv'.format(pm_loc))
    
    df_train[cols].to_csv('Informer2020/data/{}train.csv'.format(pm_loc))
    df_input[cols].to_csv('Informer2020/data/{}input.csv'.format(pm_loc))
    