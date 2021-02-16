import psycopg2
import re

# db 연결
conn_string = "host='172.30.1.29' dbname = 'siadb' user = 'sia' password = '1029'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

cur.execute("SELECT review_id, preprocessed_review FROM accommodation_review;")

symbols = []
for i in cur.fetchall():
    review = re.sub('[()|\,\'\"]', '', str(i)) # remove tuple
    review = re.sub('[ㄱ-ㅣ가-힣]+', '', review) # remove Korean
    review = re.sub('[a-z]', '', review) # remove English
    review = re.sub(' ', '', review)
    if review.isdigit() == False:
        if review != '':
            symbols.append(review)
conn.commit()

for i in symbols:
    print(i)

cur.close()
conn.close()