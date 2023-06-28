import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

from utils.tools import dotdict
from exp.exp_informer import Exp_Informer

with open('loc_match.pkl', 'rb') as f:
    loc_match = pickle.load(f)

submit = list()
    
for pm_loc in loc_match:
    df_len = pd.read_csv('data/{}test.csv'.format(pm_loc))
    DAYS = len(df_len)
    
    args = dotdict()

    args.model = 'informer'

    args.data = 'custom'
    args.root_path = './data/'
    args.data_path = '{}train.csv'.format(pm_loc)
    args.features = 'M'
    args.target = 'PM2.5'
    args.freq = 'h'
    args.checkpoints = './checkpoints/'

    args.seq_len = 24*2
    args.label_len = 24*2
    args.pred_len = 24*3

    args.enc_in = 6
    args.dec_in = 6
    args.c_out = 6
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.factor = 5
    args.padding = 0
    args.distil = True
    args.dropout = 0.05
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = True
    args.do_predict = True
    args.mix = True
    args.cols = ['기온', '풍향', '풍속', '강수량', '습도', 'PM2.5']
    args.num_workers = 0
    args.itr = 2
    args.train_epochs = 6
    args.batch_size = 32
    args.patience = 3
    args.learning_rate = 0.0001
    args.des = 'test'
    args.loss = 'mae'
    args.lradj = 'type1'
    args.use_amp = False
    args.inverse = True

    args.gpu = 0
    args.use_gpu = True 
    args.use_multi_gpu = False

    Exp = Exp_Informer

    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}'.format(args.model, args.data, args.features, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                    args.embed, args.distil, args.mix, args.des)

    exp = Exp(args)

    print('>>>>>>> start training : {}>>>>>>>'.format(setting))
    exp.train(setting)

    with tqdm(total = DAYS/24-2) as pbar:
        n = 5
        while n <= DAYS/24-2:
            args.data_path = '{}input.csv'.format(pm_loc)

            Exp = Exp_Informer
            setting = '{}_{}_ft{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}'.format(args.model, args.data, args.features, 
                            args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                            args.embed, args.distil, args.mix, args.des)

            exp = Exp(args)

            print('>>>>>>> start predicting : {}>>>>>>>'.format(setting))
            preds = exp.predict(setting, load=True)
            preds = preds.reshape(72, 6).transpose()
            submit.append(preds[-1])
            
            df_train = pd.read_csv('data/{}train.csv'.format(pm_loc))
            df_test = pd.read_csv('data/{}test.csv'.format(pm_loc))
            
            df_test['기온'][24*(n-3):24*n] = preds[0]
            df_test['풍향'][24*(n-3):24*n] = preds[1]
            df_test['풍속'][24*(n-3):24*n] = preds[2]
            df_test['강수량'][24*(n-3):24*n] = preds[3]
            df_test['습도'][24*(n-3):24*n] = preds[4]
            df_test['PM2.5'][24*(n-3):24*n] = preds[5]
            df_test.to_csv('data/공주test.csv')

            df_input = pd.concat([df_train, df_test[:24*n]], axis=0)
            df_input.to_csv('data/공주input.csv')
            
            n += 5
            
        print('{}의 미세먼지 농도 예측 완료'.format(pm_loc))
        print('현재 submit list의 길이: {}'.format(len(submit)))
        