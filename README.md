# 2nd-Author-Journal-KCI
# 연구 제목 : Developing data-driven QFD: A systematic approach to employing text information using product manuals
* 저자 : 박가문날비(1st), 김신호(2nd), 금영정(corresponding author)
* 역할 : 코드 구현·수정, 이미지 → 텍스트 추출, 키워드 분석, 감정 분석

## 연구 배경
* QFD(Quality Function Deployment)는 새로운 제품이나 서비스를 디자인하는 데 중요한 방법으로, 고객 요구사항(Customer Requirements, CRs)과 기능적 요구사항(Functional Requirements, FRs) 간의 관계를 측정함
* 이전 연구들에서는 QFD를 제품 디자인에 활용한 많은 작업들이 있었으며 QFD에 대한 데이터 기반 접근이 유망한 연구 분야 분야로 각광 받고 있음
* 그러나 대부분 데이터 기반 QFD 연구의 경우 CRs에만 초점을 두어 데이터 기반 접근을 적용하는데 한정되어 있음
* 따라서 FR 구성과 관계 측정을 고려한 데이터 기반 QFD 개발에 더 효과적인 방법을 제안해야함

## 연구 동기
* 체계적인 접근을 통해 데이터 기반 FR 구성 및 측정을 적용하여 통합된 데이터 기반 QFD 프레임워크 개발을 목표로함
* 데이터 마이닝 및 텍스트 마이닝 기법을 활용하여 적절한 FR을 추출하고 CR과 FR 간의 관계를 정량적으로 측정함

## 프레임워크

![image](https://github.com/shinho123/23.11.10-2nd-Korean-Society-of-Industrial-Engineers/assets/105840783/d5e6988e-d686-4a60-b0c2-200ff9fa9199)

## Technical role

### _WAN282X1GB Amazon review data keyword frequency_
  * 아마존 홈페이지에 세탁기(모델명 : WAN282X1GB) 리뷰 데이터를 전처리 후 키워드 분석 수행
 
```python

def Crawling_Text_Data_Keyword_Frequecy(list1):
    
    list1_cp = copy.deepcopy(list1) # 데이터 복사본 생성
    
    count = 0
    
    for idx1, st1 in enumerate(list1_cp['lemmatization']):
        for idx2, st2 in enumerate(st1):
            if len(st2) <= 2:
                count += 1
            else:
                continue
                
                
    dictionary = corpora.Dictionary(list1_cp['lemmatization'])
    corpus = [dictionary.doc2bow(text) for text in list1_cp['lemmatization']]
    
    df_lemma = []

    for idx1, st1 in enumerate(list1_cp['lemmatization']):
        for idx2, st2 in enumerate(st1):
            df_lemma.append(st2)
            
    wordcount2 = {}

    for word in df_lemma:

        wordcount2[word] = wordcount2.get(word, 0) + 1
        keys = sorted(wordcount2.keys())
        
    dict_count2 = {}

    for word in keys:
        dict_count2[word] = [wordcount2[word]]
        
    df_count2 = pd.DataFrame(dict_count2)
    
    df_count2 = df_count2.T
    df_count2.rename(columns = {0 : 'Frequency'}, inplace = True)
    df_count2.sort_values('Frequency', ascending = False)
    review_text_keyword_frequency = df_count2.sort_values('Frequency', ascending = False)
    review_text_keyword_frequency['Portion'] = round(100 * (review_text_keyword_frequency / review_text_keyword_frequency.sum()), 2)
    
    return review_text_keyword_frequency
```

### _WAN282X1GB Manual Keyword Frequency_
  * 세탁기(모델명 : WAN282X1GB) 메뉴얼 텍스트 데이터를 전처리 후 키워드 분석 수행

```python
def Manual_Keyword_Frequency(list2):
    
    # Data copy
    sentence = copy.deepcopy(list2)
    
    # Preprocessing
    for idx, st in enumerate(sentence):
        sentence[idx] = re.sub(r"[^a-zA-Z0-9]", " ", sentence[idx])
        sentence[idx] = re.sub(r"[0-9]", " ", sentence[idx])
        sentence[idx] = re.sub(' +', ' ', sentence[idx])
        sentence[idx] = nltk.word_tokenize(sentence[idx])
    
    sentence_1d = sum(sentence, [])
    stops = stopwords.words('english') # 라이브러리에서 제공하는 불용어 
    stops_plus = ['tion', 'ance', 'tem', 'con', 'machine', 'product', 'bosch', 'ines', 
                  'page', 'Start','bosch', 'www', 'zips', 'tions', 'wmzpw', 'wmz'] # 사용자 지정 불용어 설정

    for i in range(len(stops_plus)): # 라이브러리에서 제공하는 불용어에 사용자 지정 불용어를 추가함
        stops.append(stops_plus[i])
        
    clean_sentence_1d = [word for word in sentence_1d if (len(word) > 2) & (not(word in stops))]
    
    # 품사 태깅
    tokens_postag = nltk.pos_tag(clean_sentence_1d)
    
    # Noun Extraction
    NN_words = []
    for word, pos in tokens_postag:
        
        if ('NN' in pos):
            NN_words.append(word)
        else:
            continue
    
    # Lemmatization Extraction
    lemma_collection = []
    lemmatizer = WordNetLemmatizer()

    for word in NN_words:
        
        lemma_collection.append(lemmatizer.lemmatize(word))
    
    lemma_collection_pre = [word for word in sentence_1d if (len(word) > 2) & (not(word in stops))]
    
    word_count = {}
    dict_count = {}
    
    for word in lemma_collection_pre:
        
        word_count[word] = word_count.get(word, 0) + 1
        keys = sorted(word_count.keys())
        
    for word in keys:
        
        dict_count[word] = [word_count[word]]
        
    df_count = pd.DataFrame(dict_count)
    df_count = df_count.T
    df_count.rename(columns = {0 : 'Frequency'}, inplace = True)
    key_word_frequency = df_count.sort_values('Frequency', ascending = False)
    key_word_frequency['Portion'] = round(100 * (key_word_frequency / key_word_frequency.sum()),2)
    
    return key_word_frequency

```

### _WAN282X1GB Manual Keyword Search_
  * 세탁기(모델명 : WAN282X1GB) 메뉴얼 키워드 검색 함수

```python
def Manual_Keyword_Search(list2):
    
    # Data copy
    sentence = copy.deepcopy(list2)
    
    # Preprocessing
    for idx, st in enumerate(sentence):
        sentence[idx] = re.sub(r"[^a-zA-Z0-9]", " ", sentence[idx])
        sentence[idx] = re.sub(r"[0-9]", " ", sentence[idx])
        sentence[idx] = re.sub(' +', ' ', sentence[idx])
        sentence[idx] = nltk.word_tokenize(sentence[idx])
    
    sentence_1d = sum(sentence, [])
    stops = stopwords.words('english') # 라이브러리에서 제공하는 불용어 
    stops_plus = ['tion', 'ance', 'tem', 'con', 'machine', 'product', 'bosch', 'ines', 
                  'page', 'Start','bosch', 'www', 'zips', 'tions', 'wmzpw', 'wmz'] # 사용자 지정 불용어 설정
    
    for i in range(len(stops_plus)): # 라이브러리에서 제공하는 불용어에 사용자 지정 불용어를 추가함
        stops.append(stops_plus[i])
    
    # 품사 태깅
    tokens_postag = nltk.pos_tag(sentence_1d)
    
    # Noun Extraction
    NN_words = []
    for word, pos in tokens_postag:
        
        if ('NN' in pos):
            NN_words.append(word)
        else:
            continue
    
    # Lemmatization Extraction
    lemma_collection = []
    lemmatizer = WordNetLemmatizer()

    for word in NN_words:
        
        lemma_collection.append(lemmatizer.lemmatize(word))
    
    lemma_collection_lw = [word.lower() for word in sentence_1d]
    
    lemma_collection_pre = [word for word in lemma_collection_lw if (len(word) > 2) & (not word in stops)]
    
    word_coll = []
    
    for idx1, st1 in enumerate(df['lemmatization']):
        for idx2, st2 in enumerate(st1):
            if st2 in lemma_collection_pre:
                word_coll.append(st2)
            else:
                continue
            
    # lemma_collection_pre = sorted(list(set(lemma_collection_pre))) 
    
    return(list(set(word_coll)))
``` 
