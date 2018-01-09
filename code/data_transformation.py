# -*- coding:utf-8 -*-

import numpy as np
import json, random


def transform(data_file, out_file):
    '''
    由于每一秒测量两个值, 取这两者的平均值, 作为特征
    :param data_file:
    :param out_file:
    :return:
    '''

    writer = open(out_file, 'w+')
    # 写入header
    max_column = 2402
    header = 'id'
    for i in range(0, max_column / 2):
        header += ',y%s' % i
    writer.write(header + '\n')

    with open(data_file, 'r') as f:
        line = f.readline()
        while True:
            line = f.readline()
            print line
            if not line:
                break
            values = line.split(',')
            writer.write(values[0])
            for i in np.arange(1, len(values), 2):
                mean = (float(values[i]) + float(values[i + 1])) / 2
                writer.write(',' + str(mean))
            writer.write('\n')

        writer.close()
        print 'transformation done.'


def data_to_mean():
    '''
    将每秒测量的2次取平均值做为特征值
    :return:
    '''
    data_file = '../data/data.csv.filter'
    gzip_file = '../data/data_gzip.csv.filter'
    transform(data_file, data_file + '.mean')
    transform(gzip_file, gzip_file + '.mean')


def transform_data(data_file, out_file):
    '''
    将全部的haifu.csv转化为特征
    :return:
    :param data_file:
    :param out_file:
    :return:
    '''
    writer = open(out_file, 'w+')
    filter_zero_writer = open(out_file + '.filter', 'w+')
    exception_id_writer = open(out_file + '.exception', 'w+')

    # 写入header
    max_column = 2402
    header = 'id'
    for i in range(0, max_column):
        header += ',y%s' % i
    writer.write(header + '\n')
    filter_zero_writer.write(header + '\n')

    # 总记录条数
    count = 0
    with open(data_file, 'r') as f:

        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace("\\", "").replace('\n', '')
            comma_index = line.find(',')
            id = line[0: comma_index]
            values = line[comma_index + 1: len(line)]

            try:
                # print 'importing %s' % id
                data = json.loads(values)
                # 每一行的值
                values = str(id)
                filter_zero_values = str(id)
                sum = 0
                cnt = 0
                for i in range(0, max_column):
                    values += ',' + str(data[i]['y'])
                    # 将为0的值用平均值替代
                    sum += int(data[i]['y'])
                    if int(data[i]['y']) != 0:
                        cnt += 1
                writer.write(values + '\n')

                # 将为0的过滤掉
                mean = 1.0 * sum / cnt
                print 'mean = %d' % mean
                for i in range(0, max_column):
                    filter_zero_values += ','
                    filter_zero_values += str(mean) if int(data[i]['y'] == 0) else str(data[i]['y'])

                filter_zero_writer.write(filter_zero_values + '\n')
                count += 1
            except Exception, e:
                # pass
                print 'caught exception with id: %s' % id
                exception_id_writer.write(id + '\n')

    writer.close()
    filter_zero_writer.close()
    exception_id_writer.close()
    print 'import done. total rows: %d' % count


def transform_all_data():
    '''
    20170321: 转化haifu.csv
    :return:
    '''
    data_file = '../data/haifu.csv'
    out_file = '../data/data_all.csv'
    transform_data(data_file, out_file)


def file_len(file):
    '''
    计算文件的行数
    :param file:
    :return:
    '''
    count = 0
    thefile = open(file, 'rb')
    while True:
        buffer = thefile.read(8192 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def copyfile(srcfile, dstfile, linenum):
    """
        get linenum different lines out from srcfile at random
        and write them into dstfile
        """
    result = []
    ret = False
    try:
        srcfd = open(srcfile, 'r')
    except IOError:
        print 'srcfile doesnot exist!'
        return ret
    try:
        dstfd = open(dstfile, 'w')
    except IOError:
        print 'dstfile doesnot exist!'
        return ret
    # 写入header
    max_column = 2402
    header = 'id'
    for i in range(0, max_column / 2):
        header += ',y%s' % i
    dstfd.write(header + '\n')

    # 生成随机行
    srclines = srcfd.readlines()
    srclen = len(srclines)
    while len(set(result)) < int(linenum):
        s = random.randint(0, srclen - 1)
        result.append(srclines[s])
    for content in set(result):
        dstfd.write(content)
    srcfd.close()
    dstfd.close()
    ret = True
    return ret


def filter_zero(data_file, out_file):
    '''
    对于0值的点, 使用左右的平均值做为测量值
    :param data_file:
    :param out_file:
    :return:
    '''
    writer = open(out_file, 'w+')
    zero_writer = open(out_file + '.zero_count', 'w+')

    with open(data_file, 'r') as f:
        line = f.readline()
        writer.write(line)

        total_row = 0
        while True:
            line = f.readline()
            # print line
            if not line:
                break
            values = line.split(',')
            zero_count = 0
            for i in np.arange(1, len(values) - 1):
                if int(values[i]) == 0:
                    zero_count += 1
            zero_writer.write('%s,%d\n' % (values[0], zero_count))
            # print 'id: %s, zero count: %d' % (values[0], zero_count)

            # 0值缺失严重的, 忽略
            if zero_count >= 1000:
                continue

            total_row += 1
            # 找出0的个数小于300的
            writer.write(values[0])
            max_len = len(values)
            for i in np.arange(1, max_len):
                if int(values[i]) == 0:
                    # 用相邻的均值替代0值
                    left_index = i
                    right_index = i
                    while left_index >= 0 and int(values[left_index]) == 0:
                        left_index -= 1
                    while right_index < max_len - 1 and int(values[right_index]) == 0:
                        right_index += 1

                    val = (int(values[left_index]) + int(values[right_index])) / 2
                else:
                    val = values[i]
                writer.write(',' + str(val))
            writer.write('\n')

        writer.close()
        print 'transformation done. total_row: %d' % total_row


def generate_subtrain():
    '''
    生成subtrain
    :return:
    '''
    data_file = '../data/data_gzip.csv.filter.mean'
    subtrain_data = '../data/data_gzip.csv.filter.mean.subtrain'
    copyfile(data_file, subtrain_data, 1000)


def main():
    # filter_zero('../data/data_gzip.csv', '../data/data_gzip.csv.neighbor_mean')
    data_file = '../data/data_gzip.csv.neighbor_mean'
    transform(data_file, data_file + '.mean')

if __name__ == '__main__':
    main()
