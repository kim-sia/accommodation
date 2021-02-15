import psycopg2
import kss
import re

# db 연결
conn_string = "host='localhost' dbname = 'postgres' user = 'postgres' password = '1029'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

# cur.execute("ALTER TABLE accommodation_review ADD preprocessed_review varchar;")
# conn.commit()

# Basic Preprocessing
review_tokenized_contents = []

cur.execute("SELECT place_confirmid, review_id, review_contents FROM accommodation_review;")

for i in cur.fetchall():
    tuple_tmp = (i[0], i[1])
    i = i[2].strip()
    for content in kss.split_sentences(i):
        # review_tokenized_contents.append(content.strip())
        review_tokenized_contents.append(list(tuple_tmp + tuple([content.strip()])))
conn.commit()


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_punc(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()

cleaned_corpus = []
for content in review_tokenized_contents:
    tuple_tmp = (content[0], content[1])
    cleaned_corpus.append(list(tuple_tmp + tuple([clean_punc(content[2], punct, punct_mapping)])))
# print(cleaned_corpus)

def clean_text(texts):
    corpus = []
    # for i in range(0, len(texts)):
    # review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation

    review = re.sub(r'[@%\\*=()/~#&\+á?^♡★♥☕🍵🍰☎🔥🍺🍽●\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\[\]]', '',str(texts)) #remove punctuation
    # review = re.sub(r'\d+','', str(texts[i]))# remove number
    review = re.sub(r'\d+', '', review)  # remove number
    review = review.lower() #lower case
    review = re.sub(r'\s+', ' ', review) #remove extra space
    review = re.sub(r'<[^>]+>','',review) #remove Html tags
    review = re.sub(r'[<>]', '', review)
    review = re.sub(r'\s+', ' ', review) #remove spaces
    review = re.sub(r"^\s+", '', review) #remove space from start
    review = re.sub(r'\s+$', '', review) #remove space from the end
    corpus.append(review)
    return corpus

# Spell check

from pykospacing import spacing

from hanspell import spell_checker

# sent = "대체 왜 않돼는지 설명을 해바"
# spelled_sent = spell_checker.check(sent)
# checked_sent = spelled_sent.checked
#
# print(checked_sent)

from soynlp.normalizer import *

# print(repeat_normalize('와하하하하하하하하하핫', num_repeats=2))

lownword_map = {}
lownword_data = open('content/confused_loanwords.txt', 'r', encoding='utf-8')

lines = lownword_data.readlines()

for line in lines:
    line = line.strip()
    miss_spell = line.split('\t')[0]
    ori_word = line.split('\t')[1]
    lownword_map[miss_spell] = ori_word

def spell_check_text(texts):
    corpus = []
    for sent in texts:
        try:
            spaced_text = spacing(sent)
            spelled_sent = spell_checker.check(sent)
            checked_sent = spelled_sent.checked
            normalized_sent = repeat_normalize(checked_sent)
            for lownword in lownword_map:
                normalized_sent = normalized_sent.replace(lownword, lownword_map[lownword])
            corpus.append(normalized_sent)
        except Exception as ex:
            print(ex)
    return corpus

for content in cleaned_corpus:
    basic_preprocessed_corpus = clean_text(content[2])
    spell_preprocessed_corpus = spell_check_text(basic_preprocessed_corpus)
    cur.execute("SELECT count(*) FROM accommodation_review WHERE place_confirmid = %s AND review_id = %s AND preprocessed_review IS NULL;",(content[0], content[1]))
    is_null = cur.fetchone()[0]
    if is_null:
        cur.execute("UPDATE accommodation_review SET preprocessed_review = %s WHERE place_confirmid = %s AND review_id = %s;",(''.join(spell_preprocessed_corpus), content[0], content[1]))
    else:
        cur.execute("SELECT concat(preprocessed_review, ' ' , %s) FROM accommodation_review WHERE place_confirmid = %s AND review_id = %s;",(''.join(spell_preprocessed_corpus), content[0], content[1]))
        tmp = cur.fetchone()[0]
        cur.execute("UPDATE accommodation_review SET preprocessed_review = %s WHERE place_confirmid = %s AND review_id = %s;",(tmp, content[0], content[1]))
    conn.commit()

cur.close()
conn.close()