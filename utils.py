import pandas as pd

from underthesea import word_tokenize
import re

regex_train = r"train_[0-9]*[\s\S]*?\"\n[0|1]"
regex_test = r"test_[0-9]*[\s\S]*?\"\n"

with open("train.txt", mode='r', encoding='UTF-8') as f:
  lines = f.read()

  # Find trainning data with regex pattern
  train = re.findall(regex_train, lines)
  # Split to ids, labels, comments
  train_ids = [t.split("\n")[0] for t in train]
  train_labels = [t.split("\n")[-1] for t in train]
  train_comments = ["\n".join(t.split("\n")[1:-1]) for t in train]
  train_comments = [t[1:-1] for t in train_comments]
  train_comments = [t.strip().replace('\n', '').replace('\xa0', ' ').lower() for t in train_comments]
  f.close()
  
def preprocess(train_comments):
  #bỏ dấu câu và thay từ viết tắt
  new_train_comments = []
  punctuations = '''-,."!+?'''
  # punctuations = ['-',',','.','\"','!', '+', '?']
  for comment in train_comments:
    string = ''
    comment = replaceAcronym(comment)
    for c in comment:
      if c not in punctuations:
        string = string + c
      else:
        string = string + ' '
    new_train_comments.append(string)

  # loại bỏ stopword và tách từ
  stop_word = []
  with open("vietnamese_stopwords.txt",encoding="utf-8") as f :
    text = f.read()
    for word in text.splitlines() :
        stop_word.append(word.strip().replace(' ','_'))
    f.close()

  sentences = []
  for comment in new_train_comments:
    text = [word.strip().replace(' ', '_') for word in word_tokenize(comment)]
    sent = []
    for word in text:
      if (word not in stop_word):
        sent.append(word)
    sentences.append(" ".join(sent))

  return sentences

def replaceAcronym(s):
  mapping = {
    "ship": "vận chuyển",
    "shop": "cửa hàng",
    "sp": "sản phẩm",
    "m": " mình",
    "mik": "mình",
    "k": "không",
    "kh": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", # facebook
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn", 
    "dc": "được",
    "ok": "tốt",
    "dt": "điện thoại",
    "h": "giờ",
    "hsd": "hạn sử dụng",
    "trc": "trước",
    "oki": "tốt",
    "ad": "cửa hàng",
    "ko": "không"
  } 
  mapping = dict((re.escape(k), v) for k, v in mapping.items()) 
  string = re.compile(r'\b%s\b' % r'\b|\b'.join(mapping.keys()))
  s = string.sub(lambda m: mapping[re.escape(m.group(0))], s)
  return s

new_train_comments = preprocess(train_comments)

assert len(train_ids) == len(train_labels) == len(new_train_comments)

train_df = pd.DataFrame(
  {
      "id": train_ids,
      "comment": new_train_comments,
      "label": train_labels,
  }
)

# # Save
train_df.to_csv("train_data.csv", index=False)