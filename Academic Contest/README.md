신뢰성 프로젝트 경진 대회
<보증데이터 텍스트 마이닝을 이용한 자동차 원인부품 예측>

- 연구 목적
1) 텍스트 마이닝 기법을 통한 보증 데이터를 분석해 고장 관련 특징 및 정보를 추출
2) 고장 원인 부품 예측 방법론을 제시하고 예측 성능 비교

- 연구데이터: 2012~2020년 생산된 차량의 보증 데이터를 이용

- Text Mining
1) Word2Vec
-> 카테고리 별 단어 간 유사도를 기반으로 원인부품 Class 별 Word2Vec 시각화를 수행

2) CountVectorizer
-> 특이사항 카테고리를 이용해 WordCloud를 생성하고 카테고리 별 단어의 빈도수를 파악함

- 분석 결과
1) 분석 알고리즘은 Random Forest, Multi-modal DNN, Support Vector Machine을 사용하였음
2) CountVectorizer 기반 Multi-modal DNN을 수행한 결과가 Accuray, F1-score 모두 각각 0.7892, 0.7785로 가장 높았음

- 결론
1) 특이사항에 Text Mining 기법을 적용해 특징을 도출하였음 => 고장 관련 정보 추출 및 원인 부품을 식별 가능
2) 방법론에 따른 예측 성능 결과를 비교 분석함 => CountVectorizer 기반 Multi-modal DNN 모델이 성능이 가장 높음

-시사점
1) 실제 필드에서의 보증 데이터를 분석하여 고장의 근본 원인을 이해하고 신뢰성을 추정함
2) 자동차 보증 정책을 수립하는데 도움을 줄 수 있으며, 보증 비용 절감 및 제품 품질 향상 효과를 기대할 수 있음
