import requests
import json
import pandas as pd

# db 연결
import psycopg2
conn_string = "host='localhost' dbname = 'postgres' user = 'postgres' password = '1029'"

# conn = psycopg2.connect(conn_string)
# cur = conn.cursor()
#
# cur.execute("CREATE TABLE accommodation (place_confirmid VARCHAR(30) PRIMARY KEY, place_name VARCHAR(100), place_x INTEGER, place_y INTEGER, place_address VARCHAR(100), place_tel VARCHAR(20), place_rating_avg INTEGER);")
# conn.commit()
#
# cur.execute("CREATE TABLE accommodation_review (place_confirmid VARCHAR(20), review_id VARCHAR(20), review_date DATE, review_point INTEGER, review_contents VARCHAR);")
# conn.commit()
#
# cur.close()
# conn.close()

addr_xls = pd.read_excel('강원도/강원도_강릉시_도로명.xls', header=1)

exitOuterLoop = False

for addr in addr_xls['도로명']:
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    for i in range(1, 35):
        response = requests.get('https://search.map.daum.net/mapsearch/map.daum', headers={'Referer': 'ht4tps://map.kakao.com/',},
                                params=(
                                    ('callback', 'jQuery181006563952053544142_1612336187435'),
                                    ('q', '강원도 강릉시 ' + addr + ' 숙소'),
                                    ('page', i),),
                                )

        response_text = response.text
        response_text = response_text.replace('jQuery181006563952053544142_1612336187435 (', '')
        response_text = response_text.replace(')', '')
        # print(response_text)

        json_tmp = json.loads(response_text)
        place_list = json_tmp['place']


        for place in place_list:
            try:
                cur.execute("INSERT into accommodation(place_confirmid, place_name, place_x, place_y, place_address, place_tel, place_rating_avg) values (%s, %s, %s, %s, %s, %s, %s);", (
                        place['confirmid'], place['name'], place['x'], place['y'],
                        place['address'], place['tel'], place['rating_average']))
                conn.commit()
                print(place['confirmid'])
            except Exception as ex:
                print(ex)
                exitOuterLoop = True
                cur.close()
                conn.close()
                break

            i = 0
            while True:
                i = i + 1
                try:
                    url = 'https://place.map.kakao.com/commentlist/v/' + place['confirmid'] + '/' + str(i)
                    response = requests.get(url, headers={'Referer': 'https://place.map.kakao.com/' + place['confirmid'],})
                    json_tmp = json.loads(response.text)
                    review_list = json_tmp['comment']['list']

                    for review in review_list:
                        review_point = review['point']
                        review_id = review['commentid']
                        review_date = review['date']
                        try:
                            if review['contents'] == "":
                                if review['point'] == 0:
                                    review_contents = ''
                                elif review['point'] == 1:
                                    review_contents = '별로'
                                elif review['point'] == 2:
                                    review_contents = '조금 아쉬워요'
                                elif review['point'] == 3:
                                    review_contents = '보통이에요'
                                elif review['point'] == 4:
                                    review_contents = '좋아요'
                                else:
                                    review_contents = '최고예요'
                            else:
                                review_contents = review['contents'].replace('&quot;', '').replace('\n', '')
                        except:
                            if review['point'] == 0:
                                review_contents = ''
                            elif review['point'] == 1:
                                review_contents = '별로'
                            elif review['point'] == 2:
                                review_contents = '조금 아쉬워요'
                            elif review['point'] == 3:
                                review_contents = '보통이에요'
                            elif review['point'] == 4:
                                review_contents = '좋아요'
                            else:
                                review_contents = '최고예요'

                        cur.execute("INSERT into accommodation_review(place_confirmid, review_id, review_date, review_point, review_contents) values (%s, %s, %s, %s, %s);",(
                                place['confirmid'], review_id, review_date, review_point, review_contents))
                        conn.commit()
                except:
                    # print("end_review")
                    break
        if exitOuterLoop == True:
            break
    print(addr, 'end')

print("end")
cur.close()
conn.close()
