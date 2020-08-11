import chainer
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import numpy as np
import datetime

import temp_network as tpn

import csv

def predict_weather(year, month):
    tempture_data = []
    with open('data.csv', 'r', encoding='sjis') as file:
        reader = csv.reader(file)
        for i in range(5):
            header = next(reader)
        for row in reader:
            t = float(row[1])
            # -10〜40度の範囲を0〜1にする
            tempture_data.append((t + 10) / 50)

    # 直近120ヶ月分を取得
    tempture = tempture_data[len(tempture_data)-120:len(tempture_data)]
    print(len(tempture))
    # RNN用のデータにする
    temp_dataset = tpn.get_sourceset(tempture, 120)
    # ニューラルネットワークのモデルを作成
    temp_net = tpn.Tempture_NN()
    chainer.serializers.load_npz( 'tempture_model.npz', temp_net )
    # 一番最近のデータを取得してバッチサイズ=1分の配列にする
    in_data = np.array([temp_dataset[len(temp_dataset)-1][0]], dtype=np.float32)
    # 120ヶ月分のデータを入力する
    result = temp_net(in_data)
    # 119月分の気温が返されるので、最後を取得
    last_data = result[118]
    # 次の12ヶ月分の予想を行う
    # 予測気温を入れる配列
    all_predict_temp = []
    df_year = int(year) - 2020
    if df_year == 0:
        count_num = 12
    else:
        count_num = int(df_year * 2 * 12)
    for i in range(count_num):
        # 結果を取得して表示する
        data = last_data.data[0]
        temp = (data*50.0 - 10.0)
        # バッチサイズ=1,1ヶ月分の配列にする
        in_data = np.array([[[data]]], dtype=np.float32)
        # 次の日のデータを入力する
        result = temp_net(in_data, reset=False)
        last_data = result[0]
        # 予測値を詰める
        # print(count_num)
        all_predict_temp.append(temp)

    # 一年間分取得する
    predict_temp = all_predict_temp[len(all_predict_temp)-12:len(all_predict_temp)]
    # 1月から7月分までの配列
    jun_jul_temp = predict_temp[5:]
    # 8月から12月までの配列
    aug_sep_temp = predict_temp[:5]
    predict_date = {}
    # 2つの配列を1つの辞書型に詰めなおす
    for i in range(len(jun_jul_temp)):
        key = str(i + 1)
        predict_date[key] = jun_jul_temp[i]
    for i in range(len(aug_sep_temp)):
        key = str(i + 8)
        predict_date[key] = aug_sep_temp[i]

    # print(predict_date)
    return predict_date
