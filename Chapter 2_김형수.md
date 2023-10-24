# Chapter 2. Statistical Learning

# 2.1 **What is statistical learning?**

- 예시) 광고 예산(TV, radio, newspaper)과 매출과의 관계 찾기
- Input variable(*X*, predictor, independent variable, feature, variable) 과 output variable (*Y,* response variable, dependent variable)
- $Y = f(x) + \epsilon$, $X = (X_1, X_2, \dots , X_p)$
- $f$ 는 $X_1, X_2, \dots , X_p$에 대한 함수, $\epsilon$은 random error term
- $f$는 $X$가 $Y$에게 미치는 systemic information를 뜻함

## 2.1.1 Why estimate $f$?

- Prediction (예측)
    - $\hat Y = \hat f (X)$ ← error term의 평균은 0이기 때문
    - $\hat Y$는 $Y$를 예측한 값, $\hat f$는 $f$의 추정치
    - $\hat Y$의 정확도는 두 가지로 구분할 수 있음
        - Reducible error: $\hat f$의 정확도를 올림으로서 줄일 수 있는 에러
        - Irreducible error: $Y = f(X) + \epsilon$은 $\epsilon$에 대한 함수이기도 하나, $\epsilon$은 $X$로 예측할 수 없다. 즉, $f$를 정확하게 추정하더라도, $\epsilon$은 구족적으로 줄일 수 없다.
    - $E(Y - \hat Y)^2 = E[f(X) + \epsilon - \hat f(X)]^2 = [f(X) - \hat f(X)]^2 + Var(\epsilon)$
        - $E(Y - \hat Y)^2$ : 실제 $Y$ 값과 $\hat Y$의 차이의 제곱의 expected value (average)
        - $Var(\epsilon)$ : $\epsilon$의 variance (줄일 수 없음)
        - $[f(X) - \hat f(X)]^2$ : $\hat f$의 정확도를 높임으로서 줄일 수 있음
- Inference (추론)
    - $Y$의 값을 예측하는 것보다는 $Y$와 $X_1, X_2, \dots , X_p$의 관계를 이해하는 것에 초점
    - 어떤 모델링은 예측과 추론 모두를 목적으로 한다.
    - 목적에 따라 $f$를 추론하는 방법이 다를 수 있다.

## 2.1.2 How Do We Estimate $f$?

- 이 책에서는 우리가 $n$개의 데이터를 가지고 있다고 가정 함. 그 관측된 데이터를 사용하여 $f$를 추론하는 방법을 학습시키기 위하여 사용할 것이기 때문에 **training data** 라고 부른다.

### Parametric Methods

- Parametric method는 두 단계의 접근 방법을 가진다
    1. $f$의 형태에 대한 가정을 한다.
        1. 예) $f(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$ ($f$는 linear model)
    2. Model을 결정한 후, training data를 사용하여 모델을 학습시키다.
        1. 예1) 위 식의 parameter들, $\beta_0, \beta_1, \dots , \beta_p$의 값을 추정한다.
        2. 예2)  Least squares는 linear model을 학습시키는 방법 중 하나
    - 장점: $f$를 추정하는 문제를 linear model에서의 $\beta_0, \beta_1, \dots , \beta_p$와 같은  모수를 추정하는 좀 더 쉬운 문제로 변환한다.
    - 단점: 우리가 가장한 $f$의 형태가 진짜 $f$의 형태와 다를 가능성이 높다.
        - 좀 더 유연한(flexible) 모델을 선택함으로서 이를 완화할 수 있다.
        - 반면 유연한 모델은 더 많은 모수를 필요로 하고, 이는 과적합(overfitting)을 유발할 수 있다.

### Non-Parametric Methods

- Non-parametric method는 $f$의 형태에 대한 명확한 가정을 하지 않는다. 대신 데이터와 최대한 가깝지만 너무 구불구불 하지는 않은 $f$의 추정치을 찾는다.
- 장점: Non-parametric method는 $f$의 형태에 대한 가정을 하지 않기 때문에 좀 더 넓은 범위에서 정확한 $f$의 형태를 맞출 수 있다.
- 단점: 정확한 $f$의 추정치을 구하기 위해서는 많은 양의 관측 데이터가 필요하다.
- 과적합 문제는 non-parametric method에서도 발생한다.

## 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

- 모델을 예측하는 방법들은 다른 방법들에 비해서 때때로 유연(flexible)하기도 하고 제한적(restrictive)이기도 하다.
- 제한적인 모델(restrictive model)은 유연한 모델(flexible model)에 비해 제한된 범위에서 $f$의 형태를 추정하기 때문에 정확한 $f$의 형태를 추정하는 점에서 불리함이 있지만, 모델을 해석하고 이해하기가 더 쉽다는 장점이 있다.
    - tradeoff between flexibility and interpretability
- 가장 flexible한 모델이 항상 가장 정확한 예측을 하는 것은 아님. 덜 유연한 모델이 더 정확한 예측을 하기도 하는데, 이는 매우 유연한 모델은 과적합할 가능성이 높기 때문이다.

## 2.1.4 Supervised Versus Unsupervised Learning

- Supervised learning: 각각의 관측된 예측 변수 $x_i, i = 1, \dots, n$에 대하여 해당하는 종속 변수가 존재한다. 예측(prediction) 혹은 추론(inference)를 위해 종속 변수와 예측 변수 사이의 모델을 학습하는 것이 목적
    - 예) Linear regression, logistic regression, GAM, boosting, and support vector machines
- Unsupervised learning: 관측되 데이터는 있지만, 그 데이터들에 해당하는 종속 변수가 존재하지 않는다. 변수들 혹은 관측된 데이터들 사이의 관계를 이해하는 것이 목적.
    - 예) Cluster analysis (Clustering)

## 2.1.5 Regression Versus Classification Problems

- Quantitative variable: 수치형 값을 가지는 변수
    - 예) 나이, 키, 수입, 집값, 주식 가격
- Qualitative variable: ***************************K개중 하나의 등급 혹은 카테고리에 속하는 변수***************************
    - 예) 결혼 여부, 브랜드, 체납 여부,  암 발생 여부
- 종속변수가 quantitative인지 qualitative인지에 따라서 statistical learning method를 정하기도 하지만, 많은 statistical learning method들은 종속변수가 둘 중 어떤 타입이여도 사용될 수 있다.

# 2.2 Assessing Model Accuracy

- There is no free lunch in statistics: 모든 데이터셋에서 다른 모든 statistical learning method들을 능가하는 단 하나의 method는 존재하지 않는다.
- 주어진 데이터에서 가장 좋은 결과를 낼 method를 정하는 것이 중요

## 2.2.1 Measuring the Quality of Fit

- Mean squared error (MSE): $\frac{1}{n}\sum\limits_{i=1}^n(y_i - \hat{f}(x_i))^2$
    - regression에서 가장 자주 사용되는 평가지표
- Training data에서의 정확도 보다는 statistical learning method 학습에 사용되지 않은 test data에서의 정확도에 더 관심이 있음
- Training MSE가 가장 낮은 모델 보다는 test MSE ($Ave(y_0 - \hat{f}(x_0))^2$)가 가장 낮은 모델을 선택 ($(x_0, y_0)$은 method를 학습하는데 사용되지 않은 test 관측값)
- 많은 statistical learning method는 training MSE를 값을 최소화하는 것에 초점이 맞춰져있기 때문에, training MSE가 낮아도 test MSE가 높을 수 있다.
- 모델의 유연성이 높아질수록 training MSE는 감소하나, test MSE는 그렇지 않을 수 있다.
    - Statistical learning method는 training data에서 패턴을 찾으려고 하고, 그 method가 training data에서 찾은 패턴은 우연히 생겼기 때문에 test data에는 존재하지 않기 떄문.
    - Training MSE가 낮지만 test MSE가 높을 때 과적합(overfitting)한다고 한다.
- 과적합을 떠나 대부분의 경우 training MSE는 test MSE에 비해 낮으며, 더 단순한 (덜 유연한) 모델이 더 낮은 test MSE를 보여줄 때 과적합이라고 할 수 있다.

## 2.2.2 The Bias-Variance Trade-Off

- 주어진 $x_0$값에 대한 expected test MSE는 3부분으로 나눌 수 있다.
    - $E(y_0 - \hat{f}(x_0))^2 = Var(\hat{f}(x_0))+[Bias(\hat{f}(x_0))]^2 + Var(\epsilon)$
    - $E(y_0 - \hat{f}(x_0))^2$:  $f$를 계속 해서 추정했을 때 $x_0$에서의 평균 test MSE 값
    - $Var(\hat{f}(x_0)) > 0$ & $[Bias(\hat{f}(x_0))]^2 > 0$
        - ⇒ $E(y_0 - \hat{f}(x_0))^2 > Var(\epsilon)$
- Variance(분산): 다른 training data set을 사용하여 $\hat f$을 추정하면 $\hat f$가 얼마나 달라질지 나타냄
    - 이상적으로는 다른 학습 데이터를 사용한다고 해도 $\hat f$가  크게 달라져서는 안 됨.
    - 만약 method의 variance가 높다면 학습 데이터의 작은 변화가 $\hat f$의 큰 차이로 이어짐
    - 더 유연한 statistical method일 수록 variance가 높음
- Bias(편향): 복잡한 문제를 단순한 모델로 계산함으로서 발생하는 에러(error)
    - 예) 실제 $f$가 비선형일 때, 선형회귀는 $Y$와 $X_1, X_2, \dots , X_p$ 사이에 선형 관계가 존재할 것으로 가정 함. 따라서, 데이터가 아무리 많아도 정확한 추정을 할 수 없고, 높은 편향이 발생함.
    - 더 유연한 모델일수록 적은 편향을 가짐
- 더 유연한 statistical method일수록 분산은 높아지고, 편향은 줄어듬
- Statistical method를 더 유연하게 할 때 초반에는 분산이 늘어나는 것에 비해 편향이 빠르게 줄어들고 결굴 test MSE는 줄어든다. 하지만 어느 순간이 지나면서 편향이 줄어드는 것에 비해 분산이 빠르게 증가하고 test MSE 또한 증가하기 시작한다.

## 2.2.3 The Classification Setting

- Training error rate: $\frac{1}{n}\sum\limits_{i=1}^{n}I(y_i \neq \hat y_i)$
    - Training observations: ${(x_1, y_1), \dots , (x_n, y_n)}$
    - $y_1, \dots , y_n$ qualitative
    - $\hat y_i$ predicted class label
    - Indicator variable:  $I(y_i \neq \hat y_i) = \begin{cases} 1, &\text{if } y_i \neq \hat y_i \\ 0, &\text{if } y_i = \hat y_i \end{cases}$’
- Test error rate: $Ave(I(y_0 \neq \hat y_0))$
    - Test observation: $(x_0, y_0)$

### The Bayes Classifier

- Bayes classifier: 관측된 predictor vector인 $x_0$를$x_0$$\Pr(Y =  j | X = x_0)$가 가장 높은 클래스 j에 배정한다.
- Response 값이 class 1과 class2 두 가지 밖에 없을 경우, $Pr(Y=1|X=x_0) > 0.5$ 라면 class 1을, 아닐 경우 class 2를 배정한다.
- $X = x_0$일 때 error rate: $1 - \max\limits_j Pr(Y=j|X=x_0)$
- Overall Bayes error rate: $1 - E(\max\limits_j Pr(Y=j|X))$
    - Expectation이 가능한 모든 X의 값에 대한 Y의 확률의 평균을 낸다.

### K-Nearest Neighbors

- 실제 데이터의 경우 우리는 주어진 X에 대한 Y의 조건부 확률의 분포를 알 수 없기 때문에 Bayes classifier를 사용하는 것은 불가능 하다. 대신 많은 경우 주어진 X에 대한 Y의 조건부 확률을 추정하고, 추정된 확률을 사용하여 관측된 데이터를 분류한다. K-nearest neighbors (KNN) classifier도 그러한 방식 중 하나.
1. 주어진 양의 정수 K와 test observation인 $x_0$에 대해서,  KNN classifier는 $x_0$과 가장 가까운 K개의 training data를 찾는다. ($\mathcal{N}_0$: $x_0$과 가까운 K개의 training data의 집합)
2. 그 후 class j에 대한 조건부 확률을 구한다.
    - $Pr(Y=j|X=x_0) = \frac{1}{K}\sum\limits_{i \in \mathcal{N}_0}I(y_i = j)$
3. 마지막으로 가장 높은 확률을 가진 class j로 $x_0$를 분류한다.
- regression과 마찬가지로 KNN의 decision boundary가  flexible해질수록 (즉, K가 작아질수록)  training error은 계속해서 줄어들지만, test error 줄어들다 다시 커지는 U자 모양을 보인다.