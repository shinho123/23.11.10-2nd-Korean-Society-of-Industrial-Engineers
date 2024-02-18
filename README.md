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

### _Crawling Data Preprocessing_
  * 크롤링 데이터 전처리 함수
```python
def clean_text(texts): 
    corpus = []
    
    for i in tqdm(range(0, len(texts))):
        
        body = texts[i]
        
        body = re.sub('[^a-zA-Z]', ' ', body) # 특수문자 제거 
        body = body.lower().split() # 대문자를 소문자로 변경, 문장을 단어 단위로 구분
        
        df['clean_text'][i] = body
        
        stops = stopwords.words('english')
        stops.append('machine')
        stops.append('product')
        stops.append('bosch')
        
        no_stops = [word for word in body if not word in stops] # 불용어 제거
        df['stopwords_after'][i] = no_stops
        
        tokens_pos = nltk.pos_tag(df['stopwords_after'][i]) # pos tagging (품사 태깅)
        df['pos_tag'][i] = tokens_pos
        
        NN_words = [] # 명사만 추출
        for word, pos in tokens_pos:
            if 'NN' in pos:
                NN_words.append(word)
                df['NN'][i] = NN_words
                
        wlem = nltk.WordNetLemmatizer() # Lemmatization(원형(lemma) 찾기) # nltk에서 제공되는 WordNetLemmatizer을 이용
        lemmatized_words = []
        
        for word in NN_words:
            new_word = wlem.lemmatize(word)
            lemmatized_words.append(new_word)
            df['lemmatization'][i] = lemmatized_words
        
        corpus.append(no_stops) 
        
    return corpus
```

### _Contents_List_Return_
  * 메뉴얼 이미지 → 목차 이미지 텍스트로 추출 → 리스트로 반환 함수
```python
def contents_exteaction(list1):
    
    pattern = re.compile("n\d{1,2}\s{1}[A-Z][a-z\t\n\r\f\v\s]+[a-z\t\n\r\f\v\s]+\W[a-z\t\n\r\f\v\s]*")
    contents = pattern.findall(str(document))[:-6]

    for idx, st in enumerate(contents):
        contents[idx] = re.sub("n[0-9]{1,2}[\s]", "", contents[idx])
        contents[idx] = contents[idx].replace("\\n", " ").replace(".", "").strip()
        
    return contents, len(contents)
```

### _Contents_List_Return_Search_
  * 추출된 목차리스트 중 리뷰 검색이 가능 키워드 리스트 반환 함수
```python
def contents_review_search(document):
    
    stops = stopwords.words('english') # 라이브러리에서 제공하는 불용어 
    stops_plus = ['tion', 'ance', 'tem', 'con', 'machine', 'product', 'bosch', 'ines', 
                  'page', 'Start','bosch', 'www', 'zips', 'tions', 'wmzpw', 'wmz'] # 사용자 지정 불용어 설정
    
    pattern = re.compile("n\d{1,2}\s{1}[A-Z][a-z\t\n\r\f\v\s]+[a-z\t\n\r\f\v\s]+\W[a-z\t\n\r\f\v\s]*")
    contents = pattern.findall(str(document))[:-6]

    for idx, st in enumerate(contents):
        contents[idx] = re.sub("n[0-9]{1,2}[\s]", "", contents[idx])
        contents[idx] = contents[idx].replace("\\n", " ").replace(".", "").strip()
    
    contents_word = []

    for word in contents:
        contents_word.append(word.lower().split(' '))

    contents_word = sum(contents_word, [])

    contents_word = list(set(contents_word))

    lem = nltk.WordNetLemmatizer()
    contents_word_lemma = [lem.lemmatize(word) for word in contents_word]

    contents_word_re = [word for word in contents_word if (not word in stops)]


    possible_keyword = []

    for idx1, st1 in enumerate(df['lemmatization']):
        for idx2, st2 in enumerate(st1):
            if st2 in contents_word_re:
                possible_keyword.append(st2)
            else:
                continue

    return sorted(list(set(possible_keyword)))
```


    
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

### _Crawling data score vis_
  * 세탁기(모델명 : WAN282X1GB) 메뉴얼에 존재하는 키워드가 리뷰 데이터 키워드에 포함시 평점 시각화(Pie chart)함수
```python
def Crwaling_data_score_vis(keyword):
    search_keyword = Manual_Keyword_Search(document)
    date, title, review, star = [], [], [], []
    
    for idx, st in enumerate(df['lemmatization']):

        if keyword in st:
            date.append(df['Date'][idx])
            title.append(df['Title'][idx])
            review.append(df['Body'][idx])
            star.append(df['Star'][idx])

        else:
            continue
            
    review_collection = pd.DataFrame({'date' : date, 'title' : title, 'review' : review, 'star' : star})
    collection = review_collection.sort_values('date')
    collection_score = collection['star'].value_counts().to_frame('score')
    collection_score['portion'] = round(collection_score['score'] / collection_score['score'].sum(), 2)
    wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}
    plt.pie(collection_score['portion'], labels = collection_score.index, autopct='%.1f%%', startangle=260, counterclock=False, wedgeprops=wedgeprops)

    return plt.show()
```

### _Review All Search_
  * 메뉴얼 키워드가 리뷰 데이터 키워드에 포함시 제목, 내용, 날짜, 평점 조회
```python
def Review_All_Search(Keyword):
    
    title, sentence, date, star = [], [], [], []

    for idx1, st1 in enumerate(df['lemmatization']):
        if Keyword in st1:
            title.append(df['Title'][idx1])
            sentence.append(df['Body'][idx1])
            date.append(df['Date'][idx1])
            star.append(df['Star'][idx1])
        else:
            continue
    
    return title, sentence, date, star, len(date)
```

### _Image Extraction Text <-> Review_
  * 메뉴얼 이미지에서 추출된 텍스트와 리뷰 데이터가 존재하는지 확인하는 함수
```python
def Image_compare_review():
    
    results = []
    path = os.getcwd() # get a current file path
    file_list = os.listdir(path) # pull the file list from 'path' folder
    file_list_py = [file for file in file_list if file.endswith('.png') or file.endswith('jpeg')]
    
    for path in file_list_py:
        image = Image.open(path)
        result = pytesseract.image_to_string(image, lang='eng') # bosch don need kor
        arr = result.split('\n') 
        result = '\n'.join(arr)
        results.append(result)
        
    for idx, st in enumerate(results):
        results[idx] = re.sub(r"[^a-zA-Z0-9]", " ", results[idx])
        results[idx] = re.sub(r"[0-9]", " ", results[idx])
        results[idx] = re.sub(' +', ' ', results[idx])
        results[idx] = nltk.word_tokenize(results[idx])    
    
    key_word_results = []

    for word in results:
        if len(word) != 0:
            for idx, st in enumerate(word):
                key_word_results.append(st.lower())
        else:
            continue
    
    key_word_results = list(set(key_word_results))
    
    key_word_results = [word for word in key_word_results if len(word) > 2]
    
    count = 0

    for idx1, st1 in enumerate(df['lemmatization']):
        for idx2, st2 in enumerate(st1):
            if st2 in key_word_results:
                count += 1
            else:
                continue
        
    return key_word_results, count
``` 
