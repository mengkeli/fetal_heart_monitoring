# -*- coding:utf-8 -*-
#import the raw data from 'adviceData_haifu.csv' and 'adviceinfo_haifu'
#to file 'data.csv' and 'info.csv'
import sys, MySQLdb, MySQLdb.cursors, json, func

reload(sys)
sys.setdefaultencoding('utf-8')


def export_data(cursor, filename):
    '''
    导出 adviceData_haifu 表的数据
    :param cursor:
    :param filename:
    :return:
    '''

    # 遍历并且导出文件
    file = open(filename, 'w+')
    filter_zero_file = open(filename + '.filter', 'w+')

    # 写入header
    max_column = 2402
    header = 'id'
    for i in range(0, max_column):
        header += ',y%s' % i
    file.write(header + '\n')
    filter_zero_file.write(header + '\n')

    cursor.execute("""select `id`, `data` from `adviceData_haifu`""")
    ret = cursor.fetchall()
    count = 0
    for item in ret:
        id, data = item
        print 'importing row: %s' % id
        values = str(id)
        filter_zero_values = str(id)

        try:
            if data != '':
                data = json.loads(data)
                sum = 0
                cnt = 0
                for i in range(0, max_column):
                    values += ',' + str(data[i]['y'])
                    # 将为0的值用平均值替代
                    sum += int(data[i]['y'])
                    if int(data[i]['y']) != 0:
                        cnt += 1
                file.write(values + '\n')

                # 计算0值数据的比例
                zero_fields = 1.0 * cnt / max_column
                print 'zero_fields = %d%' % zero_fields*100
                if zero_fields < 0.5:
                    for i in range(0, max_column):
                        filter_zero_values += ',' + str(data[i]['y'])
                    filter_zero_file.write(filter_zero_values + '\n')
                    count += 1
        except Exception, e:
            pass

    file.close()
    print 'import done. zero_fields < 50%  rows: %d' % count


def export_info(cursor, filename):
    '''
    导出 adviceinfo_haifu 表的数据
    :param cursor:
    :param filename:
    :return:
    '''
    # 遍历并且导出文件
    file = open(filename, 'w+')

    # 写入header
    # header = 'id,userid,super_id,service_id,index_num,jianceshichang,hospitalid,doctorid,status,flag,jixian,jixianbianyi,taidongjiasutime,taidongjiasufudu,taidongcishu,bianyizhouqi,jiasu,jiasuzhenfu,jiasutimes,jiansu,jiansuzhenfu,jiansuzhouqi,jiansutimes,wanjiansu,wanjiansuzhenfu,wanjiansuzhouqi,wanjiansutimes,is_doubt,is_nst,is_zhengxuan,nst_result'
    # header = 'id,jianceshichang,jixian,jixianbianyi,taidongjiasutime,taidongjiasufudu,taidongcishu,bianyizhouqi,jiasu,jiasuzhenfu,jiasutimes,jiansu,jiansuzhenfu,jiansuzhouqi,jiansutimes,wanjiansu,wanjiansuzhenfu,wanjiansuzhouqi,wanjiansutimes,nst_result'
    header = 'id,nst_result'
    file.write(header + '\n')
    cursor.execute("""select %s from `adviceinfo_haifu`""" % header)
    ret = cursor.fetchall()

    for item in ret:
        id, nst_result = item
        values = str(id) + ',' + str(nst_result)
        for k, v in enumerate(item):
            # jiansu 这个字段有"多"或者"多次"
            if str(v).find('多') != -1:
                v = 10
            # 去除其他的非数字字段, 用空来代替
            elif str(v).find('/') != -1 or str(v).find('随诊') != -1 or str(v).find('人') != -1 \
                    or str(v).find('.') != -1 or str(v).find('·') != -1:
                v = ''
            values += str(v) + ','
        values = values[0: len(values) - 1]
        file.write(values + '\n')
    file.close()
    print 'import done. total rows: %d' % len(ret)


def export_gzip_data(cursor, filename):
    '''
    从gzip压缩字段里面提取数据
    :param cursor:
    :param filename:
    :return:
    '''
    # 遍历并且导出文件
    file = open(filename, 'w+')
    filter_zero_file = open(filename + '.filter', 'w+')

    # 写入header, id,y0,y1,y2...y2401
    max_column = 2402
    header = 'id'
    for i in range(0, max_column):
        header += ',y%s' % i
    file.write(header + '\n')
    filter_zero_file.write(header + '\n')

    cursor.execute("select count(1) from `adviceData_haifu`")
    rows = cursor.fetchone()
    cursor.execute("select `id`, `gzip_data` from `adviceData_haifu`")
    count = 0
    for i in range(int(rows[0])):
        id, gzip_data = cursor.fetchone()
        data = func.gzip_uncompress(gzip_data)

        print 'importing row: %s' % id
        values = str(id)
        filter_zero_values = str(id)

        try:
            if data != '':
                data = json.loads(data)
                sum = 0
                cnt = 0
                for i in range(0, max_column):
                    values += ',' + str(data[i]['y'])
                    # 将为0的值用平均值替代
                    sum += int(data[i]['y'])
                    if int(data[i]['y']) != 0:
                        cnt += 1
                file.write(values + '\n')

                # 将为0的过滤掉
                mean = 1.0 * sum / cnt
                print 'mean = %d' % mean
                for i in range(0, max_column):
                    filter_zero_values += ','
                    filter_zero_values += str(mean) if int(data[i]['y'] == 0) else str(data[i]['y'])

                filter_zero_file.write(filter_zero_values + '\n')
                count += 1
        except Exception, e:
            pass

    file.close()
    print 'import done. total rows: %d' % count


def main():
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='1q2w3e4r5t',
        db='heart_rate'
    )

    cursor = conn.cursor()
    # export_data(cursor=cursor, filename='../data/data.csv')
    # export_info(cursor=cursor, filename='../data/info.csv')
    export_gzip_data(cursor=cursor, filename='../data/data_gzip.csv')


if __name__ == '__main__':
    main()
