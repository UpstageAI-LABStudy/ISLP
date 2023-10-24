# 통계 학습 개요
통계적 학습은 데이터를 이해하기 위한 광범위한 도구세트 를 의미.  
이 도구들은 크게 말하면 supervised or unsupervised로 분류가능.  
Supervised statistical learning: 하나 이상의 압력을 기반으로 출력을 예측하거나 추정하기 위한 통계 모델을 구축하는 작업이 포함된다.  
이러한 문제는 비즈니스,의학,천문학 및 공공 정책과 같은 다양한 분야에서 발생한다.  
Unsupervised statistical learning: 입력 변수는 있지만 지도 출력이 없으며, 그럼에도 불구하고 이러한 데이터로부터 관계와 구조를 학습할 수 있다.  
이 책 내의 Wage 데이터 세트를 활용한 프로그램 에서는 대서양 지역의 일련의 남성들의 급여와 관련된 여러 가지 요인을 조사한다.  
특히, 나이와 교육, 그리고 달력 연도가 급여에 미치는 요인을 조사한다.  Figure1.1의 왼쪽 패널은 데이터 세트의 각 개인의 대한 나이 대 급여를 나타낸다. 그림에서 나이가 증가함에 따라 급여도 증가하는 증거가 있지만, 약60세 이후로 다시 감소하는 경향이 있다. 주어진 나이에 대한 평균 급여를 추정하는 파란색 선은 이 경향을 더 명확하게 보여준다.  나이가 주어지면 이 곡선을 사용하여 해당 사람의 급여를 예측할 수 있다. 그러나 Figure1.1에서도 이 평균값과 관련된 상당한 변동성이 있음을 분명히 볼 수 있으므로 나이만으로는 특정 남성의 급여를 정확하게 예측하는 데는 부족함이 있다. 
## 임금 데이터 ![Image](./1.png) 
그림 1.1은 미국 중부 대서양 지역의 남성을 대상으로 한 소득 조사 정보를 포함한 Wage 데이터를 보여줌. 일반적으로 나이가 증가함에 따라 급여가 증가하지만 약 60세에서 급여가 감소하기 시작한다. 중앙 패널에서는 연도에 따른 급여를 보여준다. 오른쪽 패널에서는 연도에 따른 급여를 보여준다. 1은 가장 낮은 수준~5는 가장 높은 수준을 나타내고, 평균적으로 교육 수준이 높아질수록 급여도 증가하는 경향이 있다.
데이터에는 각 직원의 교육 수준 및 급여 획득 연도에 대한 정보도 있다. 그림1.1의 중앙 및 오른쪽 패널은 연도 및 교육에 따른 급여를 나타내며, 이 두 요인이 급여와 관련이 있음을 보여준다. 급여의 상승은 대략 선형적인 패턴을 따른다. 그러나 이 급여 상승은 데이터의 변동성에 비해 매우 미미하다. 일반적으로 교육 수준이 높은 사람들이 급여가 더 높다. 따라서 특정 남성의 급여를 가장 정확하게 예측하려면 그의 나이, 교육 및 연도를 모두 고려해야 한다. 3장에서는 선형 회귀를 다루고, 가능한 급여와 나이 사이의 비선형 관계를 고려하여 급여를 예측해야 한다. 7장에서는 문제 해결을 위한 접근 방식 중 하나를 다룬다. Wage 데이터 집합은 출력 값을 예측하는 작업을 수행하는 것을 다룬다. 4장에서는 5년간의 Standard & Poor’s 주식 지수의 주식 시장 데이터 집합을 검토한다. ![Image](./2.png) Figure1.2에는 Smarket 데이터에서 얻은 시장이 상승이든 하락이든 두 그룹의 이전 날 주식 지수 백분율 변화에 대한 그림이 나타난다. 4장에서는 이러한 데이터를 다양한 통계 학습 방법을 사용하여 탐구한다.  주식시장 데이터에서는 특정 날의 주식 지수가 오를지 내릴지 예측하는 분류문제가 있으며, 시장이 상승할지 하락할지 예측하는 모델을 구축하는 것이 목표이다.  
## 유전자 발현 데이터
![Image](./3.png) 분류 문제가 아닌 클러스터링 문제 > 12장에서 설명, 차원축소 사용.  

![Image](./4.png) 유전자발현 데이터 세트의 표현, 14가지 종류의 암을 각각 다른 색상의 기호로 표시.
## A Brief History of Statistical Learning
통계학습이라는 용어는 상대적으로 최근에 등장했지만, 이 분야의 기반 개념들은 오래전부터 개발되었다. 19세기 초에 최소 제곱법의 방법이 개발되었으며, 선형 회귀로 알려진 것의 초기 형태를 구현하였다. 양적인 값의 예측에 사용. 질적인 값 예측을 위한 선형 판별 분석은 1936년제안,  1940년대 로지스틱 회귀 제시, 1970년대 일반화 선형 모델이라는 용어 개발, 통계 학습 방법 클래스를 설명하기 위해 사용, 1970년대 말까지 데이터의 학습에 사용되는 여러 기술이 등장했지만, 그 시기에는 컴퓨터 리소스 부족으로 선형 방법에 국한되었다. 1980년대에는 컴퓨팅 기술이 충분히 개선되어 비선형 방법을 사용하는 것이 계산상 불가능하지 않았다. 1980년대에 분류 및 회귀 트리가 개발되었고, 그 뒤로 일반화된 가법 모델이 등장하였음. 통계학습은 통계학의 새로운 하위 분야로 등장하여 지도 및 비지도 모델링 및 예측에 중점을 두고 있다.  
이 책은 The Elements of Statistical Learning(ESL)에 대한 참고 자료이다. ESL을 중요한 동반자로 보고 있다.   

표기법    

## Organization of This Book
2.기본 용어와 개념소개  
3.선형 회귀  
4.분류 방법  
5.최적 방법 선택  
6.선형 및 비선형 방법  
7.비선형 방법  
8.비지도 학습  
9.다중 가설 검정  

데이터 세트 목록
![Image](./5.png)  

## 2.Statistical Learning
What Is Statistical Learning?
통계적 학습의 개념을 이해하기 위해 간단한 예제로 시작, 목표는 세 가지 매체 예산을 기반으로 판매를 예측하는 정확한 모델 개발.
광고 예산은 입력변수, 판매는 출력 변수. 입력변수는 X1이 TV예산, X2가 라디오 예산, X3가 신문 예산. 출력변수는 Y로 표시.  
일반적인 수식: Y = f(X) + "e".  
![Image](./6.png)  
변수에 대한 판매량, 파란색 선은 TV, 라디오를 사용하여 매출을 예측하는 데 사용할 수 있는 간단한 모델을 나타냄. 이 공식에서 f는 X가 Y에 대해 제공하는 체계적인 정보를 나타냄.
### 소득 데이터 세트
![Image](./7.png)
빨간점:관측된 값 파란색 곡선:소득과 소득 간의 실제 기본 관계를 나타냄. 검은색 선: 오류  
수년간의 데이터를 사용하여 소득을 예측할 수 있음을 시사한다. 함수f는 일반적으로 알려져 있지 않고 이 상황에서는 추정해야한다.
f는 관찰된 점을 기반으로한다.소득의 f는 알려져 있으며 파란색 곡선으로 표시된다. figure2.3에서는 교육 기간의 함수로 소득을 표시하고 여기서 f는 추정되어야 하는 2차원 표면이다.
### Why Estimate f?
f를 추정하려는 두 가지 주요 이류는 예측과 추론이다.  
예측: 많은 상황에서 X는 알지만 Y는 알기 힘들다.  
따라서 Y값을 얻기 위해 오차 항의 평균이 0인 점을 이용하여(오차항은 random variable로 평균이 0이고 분산이 Var(ε)인 정규분포를 따르게 됨) 예측이 가능하다. Yˆ = fˆ  
fˆ은 f에 대한 추정치, Yˆ은 Y에 대한 결과 예측을 나타냄. 블랙박스 역할. Yˆ의 정확성은 줄일 수 있는 오차(reducible error)와 줄일 수 없는 오차(irreducible error) 두 가지 요소에 의존한다. 일반적으로 ˆf는 f의 완벽한 추정이 아니므로 일부 오차를 도입한다. 이 오차는 줄일 수 있는 오차이다. f를 추정하기 위해 가장 적합한 통계적 학습 기술을 사용하여 fˆ의 정확성을 향상시킬 수 있기 때문. 
### 교육 기간에 따른 소득
![Image](./9.png)
![Image](./8.png)
reducible error: 적절한 통계적 방법을 사용하여 모델의 정확도를 높임으로써 감소 가능.  
irreducible error: e에 대한 부분인데, 이는 설명변수 X를 사용하여 감소시킬 수 없는 부분. e는 관측되지 않은 변수들(unmeasured variable)과 변동성(variation)에 대한 정보를 가지고 있을 수 있음  
추론:Yˆ = fˆ(X), fˆ의 정확한 형태를 알아야 함.  
목적: 설명변수 X와 반응변수 Y의 관계에 대한 이해.  
많은 X들 중 중요한 소수의 설명변수 찾기  
Y와 각X들 간의 관계 유무(인과 관계)  
Y와 X들의 관계가 선형 방정식으로 설명될 수 있는지  
Prediction:예측 변수를 사용하여 반응을 예측하는 정확한 모델을 원할경우  
Inference:어떤 매체가 매출에 기여하는지/증가시키는지 알고자 할 때
Prediction + Inference: 이 집의 가치가 얼마나 과소/과대 평가 되었는가?  
선형 모델: 간단하고 해석하기 쉬움+ 정확한 예측 불가  
복잡한 비선형 모델: 정확한 예측 + 해석하기 어려움  
### How do we estimate f?  
n개의 데이터가 주어진다면 이를 학습 데이터라고 부른다.  
목적은 통계 학습을 학습 데이터에 잘 적용해서 함수 fˆ을 찾는 것이다.  
Parametric Methods(모수 방법), Non-parametric Methods(비모수 방법)
### Parametric Methods(모수 방법)
![Image](./10.png)  
함수 형태를 특정 모양으로 모양에 대해 가정한다.  
![Image](./12.png)  
위와 같은 식이라고 가정하면, 이는 선형 모델이라고 부를 수 있다. 모델을 선택한 후, 훈련 데이터를 통해 모형을 fit하거나 훈련시키고 파라미터 집합을 추정한다.  
![Image](./13.png)  
장점:f를 추정하는 문제를 단순화하며, 임의의 함수 f를 적합 시키는 것보다 파라미터를 추정하는 것이 일반적으로 쉽다.  
단점:선택한 모델이 알려지지 않은 f의 실제 형태와 보통은 맞지 않다. 복잡한 모델을 선택하면 데이터에 대한 과적합이 뒤따를 수 있다.  
### Non-parametric Methods(비모수 방법)  
![Image](./11.png)  
f함수 형태에 대해 명시적인 가정을 하지 않고 f를 추정하는 방법  
장점: 추정한 f가 실제 f와 많이 달라질 위험을 방지. 가능한 다양한 형태의 f에 대해 비교적으로 정확한 fit을 보여준다.  
단점: 모델에 대한 추정이 모수에 대한 추정이 되지 않으므로, 아주 많은 수의 관측치가 필요하다.  
### The Trade-off between prediction accuracy and model interpretability(예측 정확성과 모델 해석성 사이의 균형)
![Image](./14.png)  
일반적으로 통계 학습 모델의 복잡도와 해석 가능성은 trade-off 관계에 있다. 그림에서 X축은 모델의 복잡도, Y축은 해석 가능성을 나타낸다. 이때 복잡한 모델이 성능은 더 잘 나올 수 있지만, 추론을 하기가 어려워서, 모델이 어떻게 예측을 잘하였는지 해석하기가 어렵다.  
restrictive models(제한적인 모델):제한적인 모델이 훨씬 해석하기 쉬우므로 덜 복잡한 방법을 사용하는 것이 좋은 선택일 수 있다.  
flexible models(복잡한 모델): 예측에 중점을 두는 경우 복잡한 모델이 더 적합할 수 있다.  
무조건 모델의 복잡도가 높다고 해서 좋은것은 아니다. 과적합 문제가 뒤따를 수 있기때문.  
### Supervised vs. Unsupervised learning  
Supervised learning: X와 Y가 모두 주어져 있어 모두를 가지고 학습,   
ex) linear regression, logistic regression  
Unsupervised learning: X는 주어져 있지만 이에 해당하는 Y가 없어 X만을 가지고 학습  
ex) Clustering  
Regression vs. Classification  
입력 및 출력 변수는 수치형 or 범주형 값으로 나눌 수 있다.  
수치형 값은 연속적인 값.(나이,키,온도,가격)  
범주형 값은 브랜드(A,B,C),병의 유병률(True,False)등의 특징이 숫자로 표현되지 않음.  
Regression: 숫자형  
Classification: 범주형  
### Assessing Model Accuracy(모델 정확도 평가)  
주어진 데이터 집합에 대해 어떤 방법이 최상의 결과를 생성하는지 결정하는 것이 중요한 작업이다.  
Measuring the quality of fit  
MSE(평균 제곱 오차) ![Image](./15.png)  
training data로 MSE를 줄이는 방향으로 모델링. 하지만, 우리의 관심은 test data에 대한 예측 정확도임.  training MSE가 작다고 해서 test MSE도 작다는 보장이 없음. 따라서, 단순히 training MSE가 가장 작은 모델을 선택하는 것보다 test MSE에 대한 비교도 필요함  
### 모델의 복잡성과 MSE의 관계
![Image](./16.png)  
주황선: 모델의 복잡성이 가장 떨어지는 모델.  
녹색:가장 복잡한 모델  
Train MSE와 같은 경우는 모델의 복잡성이 가장 높은 녹색이 가장 좋은 모습을 보인다. 하지만 Test MSE와 극명한 차이를 보임.  
주황색과 녹색의 중간정도의 복잡성을 보여주는 파란색과 같은 경우는 Train MSE는 녹색과 비교했을 때 다소 높지만 Test MSE는 수평 점선에 가장 근접한 MSE의 성능을 나타낸다.  df=5정도가 적절해보임. df=20을 넘어간 초록색 모델은 과적합.
### 비선형적인 예시  
![Image](./18.png) 
훈련 및 테스트 MSE 곡선은 여전히 동일한 일반 패턴을 나타내지만 이제 테스트 MSE가 천천히 증가하기 시작하기 전에 두 곡선 모두에서 급격한 감소가 보임. Test MSE가 최소가 되는 모델에 대응하는 복잡성 수준은 데이터에 따라 다르다. Test MSE를 최소로 하기 위해 교차검증(Cross-Validation)을 사용할 수 있다.  
### The bias-variance trade-off(편향 분산 트레이드오프)  
Test MSE 곡선이 U모양인 이유는, 분산과 편향 두 가지 상충되는 성질 때문이다.  ![Image](./19.png)  
위의 식에서 Var(e)(irreducible error)는 낮출 수 있는 방안이 없다. 따라서, Test MSE의 기댓값을 최소화 하는 것을 목표로 낮은 분산과 낮은 편향을 같이 가져 갈수 있는 통계 학습 방법을 선정해야 한다.  
이때 편향과 분산 ![Image](./20.png)  
편향(Bias): 복잡한 실 세계 문제를 단순한 모델로 근사시킴으로 발생되는 오차.  비선형적인 데이터를 선형 데이터로 근사하더라도 아무리 많은 데이터가 존재해도 정확한 추정치를 생성하는 것은 어렵다. 일반적으로 복잡성이 높을수록 편향이 적다.  
분산(Variance): 데이터 셋의 변동에 따른 f^의 변화의 정도.  
일반적으로 복잡성이 높을수록 분산도 높다.  ![Image](./21.png)
이러한 mse, bias, variance의 관계를 트레이드 오프 관계라고 함  
### The classification setting  
일반적으로 Error rate를 계산한다. ![Image](./22.png)  
I(True) = 1, I(False) = 0으로 정의된다. Error rate는 전체 예측 중 잘못 예측(분류)한 비율을 나타낸다.  
### Bayes classifier  
![Image](./23.png)
이 식을 최대화하는 클래스j를 고르는 방법이다.  
bayes error rate  ![Image](./24.png)  
### KNN(K-최근접 이웃)  
판별하고 싶은 데이터와 인접한 k개수의 데이터를 찾아, 해당 데이터의 라벨이 다수인 범주로 데이터를 분류하는 방식. ![Image](./26.png) ![Image](./25.png)
k값에 따라 성능이 결정된다.  
k = 1인 경우, 과적합 발생 가능성이 매우 높음(편향은 낮지만 분산이 높다).  
k = 100인 경우, 과소적합 발생 가능성 있음(분산은 낮지만 편향이 높다).  ![Image](./27.png)  
적당한 수의 k를 정하는 것이 필요하다.  
