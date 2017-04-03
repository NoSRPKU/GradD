# -*- coding: utf-8 -*-

import csv
import time
import datetime
import numpy as np

def load_data(file_path="", n=8, stdd = 1,
              train_sample_beg='2009-01-01 00:00:00;',
              test_sample_beg='2013-01-01 00:00:00',
              verify_time_beg='2014-01-01 00:00:00',
              verify_time_end='2015-01-01 00:00:00'):

    if not file_path:
        return []

    def get_data(t_b, t_e, smp, num, avg, std, to_std):
        inp = []
        outp = []
        outp_t = []
        time_beg = t_b
        time_end = time_beg + datetime.timedelta(minutes=num * 15)
        while time_end < t_e:
            cur_val = []
            t = time_beg
            while t < time_end:
                val_t = [((v - a) * to_std / s) for v, a, s in zip(smp.get(t), avg, std)]
                cur_val += val_t
                t += datetime.timedelta(minutes=15)
            inp.append(cur_val)
            outp.append(smp.get(time_end)[::3])
            outp_t.append(time_end)
            time_beg += datetime.timedelta(minutes=15)
            time_end += datetime.timedelta(minutes=15)
        return inp, outp, outp_t

    sample = {}
    test_beg = datetime.datetime.strptime(test_sample_beg, "%Y-%m-%d %H:%M:%S")
    verify_beg = datetime.datetime.strptime(verify_time_beg, "%Y-%m-%d %H:%M:%S")
    with open(file_path, 'rb') as f:
        data = csv.reader(f)
        count = 0
        bad_count = 0
        journey_time_list = []
        speed_list = []
        flow_list = []
        for val in data:
            count += 1
            dt = datetime.datetime.strptime(val[2], "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=int(val[3]) * 15)
            journey_time = float(val[4])
            journey_time_list.append(journey_time)
            speed = float(val[5])
            speed_list.append(speed)
            quality = int(val[6])
            length = float(val[7])
            flow = float(val[8])
            flow_list.append(flow)
            if quality not in (1, 3):
                print dt, journey_time, speed, journey_time * speed - length * 3600, quality, flow
                bad_count += 1
            sample[dt] = [journey_time, speed, quality, flow]
        print count, bad_count
        average = np.average(sample.values(), axis=0)
        average[2] = 2
        standard = np.std(sample.values(), axis=0)
        standard[2] = stdd
        print average
        print standard

        train_beg = datetime.datetime.strptime(train_sample_beg, "%Y-%m-%d %H:%M:%S")
        verify_end = datetime.datetime.strptime(verify_time_end, "%Y-%m-%d %H:%M:%S")

        train_inp, train_outp, train_outp_t = get_data(train_beg, test_beg, sample, n, average, standard, stdd)
        test_inp, test_outp, test_outp_t = get_data(test_beg, verify_beg, sample, n, average, standard, stdd)
        verify_inp, verify_outp, verify_out_t = get_data(verify_beg, verify_end, sample, n, average, standard, stdd)
        return train_inp, train_outp, train_outp_t, test_inp, test_outp, test_outp_t, verify_inp, verify_outp, verify_out_t
if __name__ == "__main__":
    val = load_data('/home/nosr/Documents/out/LM782.csv', stdd=2)
    print val[-3][-10]