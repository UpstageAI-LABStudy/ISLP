# Chapter 3. Linear Regression

- 선형회귀(linear regression)은 지도학습(supervised learning) 중 아주 간단한 statistical method이며, 질적 변수를 예측하는데 특히 유용한 도구이다.
- 선형회귀를 통해 답하고자 하는 질문들:
    - 종속 변수와 독립 변수 사이의 관계 유무
    - 종속 변수와 독립 변수 사이의 관계가 얼마나 강한지
    - 어떤 변수가 종속 변수와 연관이 있는지
    - 종속 변수와 독립 변수 사이의 관계가 선형적인지
    - 등..

# 3.1 Simple Linear Regression

- 단순 선형 회귀(Simple linear regression)은 질적 종속 변수 Y를 하나의 독립변수 X로 예측하는 접근법
- Y와 X 사이에 선형적인 관계가 있을 것이라고 가정
- 수학적 표기법: $Y \approx \beta_0 + \beta_1X$
    - $\approx$ : 근접하게 모델되어 있다
    - $\beta_0, \beta_1$ : 우리가 알지 못하는 절편(intercept)과 기울기(slope)를 나타내는 상수. Model coefficients 혹은 model parameters라고 부름.
- $\hat{y} = \hat \beta_0 + \hat \beta_1 x$
    - $\hat y$ : X가 x일 때 예측된 Y 값
    - $\hat \beta_0, \hat \beta_1$ : 학습 데이터를 사용하여 추정한 $\beta_0$와 $\beta_1$의 값
    - $\hat{}$  : 우리가 알지 못하는 모수 혹은 Y의 추정된 값을 나타낼 때 사용

## 3.1.1 Estimating the Coefficients

- Simple linear regression의 목적은 $i = 1, \dots, n$에서 $y_i \approx \hat{\beta}_0 + \hat{\beta}_1 x_i$인 $\hat{\beta}_0$과 $\hat{\beta}_1$을 찾는 것. 즉, n개의 데이터와 최대한 가까운 $\hat{\beta}_0$의 절편과 $\hat{\beta}_1$의 기울기를 가진 선을 찾는 것.
- Residual sum of squares (RSS)
    - 위에서 말한 가까운 정도를 나타내는 방법 중 하나
    - $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i$ (주어진 i번째 X값을 사용하여 구한 Y에 대한 예측값)
    - Residual ($e_i$): $e_i = y_i - \hat{y}_i$ (실제 관측된 i번째 y 값과 linear model을 사용하여 구한 y의 i번째 예측값의 차이)
    - $RSS = e_1^2 + e_2^2 + \cdots + e_n^2$
- Least squares coefficient estimates for simple linear regression
    - $\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar y)}{\sum_{i=1}^n(x_i - \bar x)^2}$, $\hat\beta_0 = \bar y - \hat \beta_1\bar x$
        - $\bar y \equiv \frac{1}{n} \sum_{i=1}^n y_i$, $\bar x \equiv \frac{1}{n} \sum_{i=1}^n x_i$ (sample means)

## 3.1.2 Assessing the Accuracy of the Coefficient Estimates

- $Y = \beta_0 + \beta_1 X + \epsilon$일 때, $\beta_0$는 X=0 일 때 Y의 expected value, $\beta_1$은 X가 one-unit 만큼 상승했을 때 평균적인 Y의 변화. $\epsilon$은 간단한 모델을 사용하면서 놓치는 모든 것을 의미.
- Unbiasedness: $\beta_0$와 $\beta_1$을 추정했을 때 그 추정값을 $\beta_0$와 $\beta_1$에 정확히 일치하지 않을 것이다. 하지만 엄청나게 많은 데이터셋을 통해 두 모수를 추정하고 평균을 내게 되면 정확한 $\beta_0$와 $\beta_1$의 값을 구할 수 있다.
- $\hat \mu$의 standard error ($SE(\hat \mu)$):
    - Random variable Y의 실제 평균값 (population mean)을 $\mu$라고 하고 sample mean을 $\hat\mu$라고 했을 때, $Var(\hat \mu) = SE(\hat \mu)^2 =\frac{\sigma ^2}{n}$.
    - 하나의 추정값 $\hat \mu$이 실제값 $\mu$로부터 얼마나 벗어나 있는지를 나타냄.
    - $\sigma$는 Y의 각각의 $y_i$의 표준편차
    - n이 커질수록 $SE(\hat\mu)$는 줄어듬
- $\hat\beta_0$와 $\hat\beta_1$의 standard error:
    - $SE(\hat\beta_0)^2 = \sigma^2 [\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar x)^2}]$
    - $SE(\hat\beta_1)^2 = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar x)^2}$
        - $\sigma ^2 = Var(\epsilon)$
        - 이 공식들이 유요하기 위해서는 각각의 관측치에 대한 에러들 $\epsilon_i$들이 같은 분산 $\sigma^2$을 가지고 있고, 서로 상관관계가 없다는 가정이 필요함
    - $\sigma^2$의 실제 값은 알 수 없지만, 추정할 수는 있음 (Residual standard error)
        - $RSE = \sqrt{RSS/(n-2)}$
    - Confidence interval (신뢰구간)
        - 95% confidence interval: 반복해서 샘플을 뽑아 각각의 샘플에 대해서 confidence interval을 구했을 때, 95%의 confidence interval들은 우리가 알지 못하는 일제 모수의 값을 포함하고 있다.
        - 선형회귀에서 $\beta_1$의 95% 신뢰구간은 대략 $\hat \beta_1 \pm 2 \cdot SE(\hat\beta_1)$의 형식을 띈다. ($[\hat\beta_1 - 2 \cdot SE(\hat\beta_1), \hat\beta_1 + 2 \cdot SE(\hat\beta_1)]$)
        - $\beta_0$의 경우도 마찬가지 ($\hat\beta_0 \pm 2 \cdot SE(\hat\beta_0)$
    - Hypothesis test (단순선형회귀에서의 가설검정)
        - 귀무가설 $H_0$: X와 Y 사이에는 아무 관계가 없다. ($H_0 : \beta_1 = 0$)
        - 대립가설 $H_a$: X와 Y 사이에 관계가 있다. ($H_a: \beta_1 \neq 0$)
        - 대립가설을 검정하는 것은 $\beta_1$의 추정값인 $\hat\beta_1$이 0에서부터 충분히 떨어져있어서 $\beta_1$이 0이 아니라고 신뢰할 수 있는지 보는 것
        - 0에서부터 얼마나 떨어져있어야 하는지는 $\hat\beta_1$의 정확도, 즉 $SE(\hat\beta_1)$에 달려있다.
        - t-statistic: $t = \frac{\hat\beta_1 - 0}{SE(\hat\beta_1)}$($\hat\beta_1$이 0에서부터 표준편차 몇 개 만큼 떨어져있는지 계산)
        - 만약 X와 Y사이에 관계가 없다면 t-statistic이 자유도가 n-2인 t분포를 따를 것이라고 예상한다.
        - p-value: 절대값이 |t| 보다 크거나 같은 값은 관측할 확율
            - p-value가 작으면 실제 X와 Y 사이에 아무런 관계가 없을 때, 우연히 X와 Y사이에 그런 큰 관계를 관측했을 가능성이 낮다고 판단 ⇒ Predictor와 response가 연관이 있다고 추론할 수 있음
            - ⇒ p-value가 충분히 낮으면 (보통 5% 혹은 1%) 귀무가설을 기각

## 3.1.3 Assessing the Accuracy of the Model

- 선형회귀는 보통 residual standard error (RSE) 혹은 $R^2$를 사용하여 정확도를 측정함.

### Residual Standard Error

- RSE는 $\epsilon$의 표준편차의 추정값. (response 값들이 실제 regression line에서 얼마나 벗어나 있는지의 평균값)
- $RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2}\sum_{i=1}^n (y_i - \hat y_i) ^2}$
- RSE는 모델이 데이터를 얼마나 잘 맞추지 못하는지를 측정
    - 만약 $i = 1, \dots, n$에 대하여 모델을 사용하여 예측한 값 $\hat y_i$이 $y_i$와 가깝다면 RSE가 낮고, 반대의 경우 RSE가 높다.

### $R^2$ Statistic

- RSE는 Y에 대해서만 측정하기 때문에 무엇이 좋은 RSE 값을 구성하고 있는지 (왜 RSE가 낮은 값을 보여주는지) 명확하지 않을 수 있다.
- $R^2$도 모델이 얼마나 데이터를 잘 맞추는지에 대한 측정지표이다.
- $R^2$는 비율을 형태를 띄며, 항상 0에서 1사이의 값을 가진다.
- $R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS}$
    - Total sum of squares: $TSS = \sum(y_i - \bar y)^2$ (Y의 분산의 합)
    - RSS는 회귀모델 이후에도 설명되지 않는 Y의 변산성
    - $TSS -RSS$는 회귀모델로 설명되는 변산성
    - $R^2$는 X로 설명될 수 있는 Y속 변상의 비율
    - $R^2$가 1에 가까우면 종속변수의 변산성 중 큰 비율이 회귀 모델로 설명되어 진다고 할 수 있고, 0에 가까우면 회귀모델이 종속변수의 변상을 별로 설명하지 못한다고 볼 수 있다.
- 선형회귀 모델이 옳은 모델이 아니거나, error의 분산 $\sigma^2$이 큰 경우, 두 가지 모두 해당되는 경우 0과 가까운 $R^2$가 측정될 수 있다.

# 3.2 Multiple Linear Regression

- p개의 예측 변수가 있을 때 multiple linear regression 모델(다중 선형 회귀 모델)은 다음과 같은 형식을 가진다.
    - $Y =  \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$
    - $X_j$는 j번째 예측변수
    - $\beta_j$는 $X_j$와 종속변수간의 연관성을 수량화한 것
        - 다른 예측 변수를 고정시키고 $X_j$를 한 단위만큼 증가시켰을 때 Y값에 나타나는 영향의 평균으로 해석

## 3.2.1 Estimating the Regression Coefficients

- 단순 선형 회귀와 마찬가지로 $\beta_0, \beta_1, \dots, \beta_p$의 값을 알 수 없기 때문에 추정해야 한다.
- $\hat y = \hat \beta_1 x_1 + \hat \beta_2 x_2 + \cdots + \hat\beta_p x_p$
- Multiple least squares regression coefficient estimates:
    - RSS를 최소화하는 $\hat\beta_0, \hat\beta_1, \dots, \hat\beta_p$의 값
    
    $$
    \begin{align} RSS &= \sum_{i=1}^n (y_i - \hat y_i)^2 \\ &= \sum_{i=1}^n (y_i - \hat\beta_0 - \hat\beta_1x_{i1} - \hat\beta_2 x_{i2} - \cdots - \hat\beta_p x_{ip})^2 \end{align}
    $$
    
- 단순선형회귀와 다중선형회귀에서 기울기값이 차이가 날 수 있는데, 단순선형회귀에서 기울기는 단 하나의 독립변수가 한 단위만큼 증가하였을 때 종속변수의 변화의 평균인 반면, 다중선형회귀에서 기울기는 다른 독립변수를 고정시키고 하나의 독립변수가 한 단위만큼 증가하였을 때 종속변수의 변화의 평균이기 때문이다.
- 독립변수간 상관관계가 있을 경우, 독립변수가 하나 이상의 독립변수에 영향을 받아, 단순선형회귀에서 그 독립변수가 종속변수에 직접적으로 영향을 미치지 않더라고, 다른 독립변수로 인해 둘 사이에 연관성이 생길 수 있다.

## 3.2.2 Some Important Questions

### 종속변수와 독립변수 사이에 관계가 존재하는가?

- 가설검정
    - $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$
    - $H_a: \text{최소 한 개의 } \beta_j \text{는 0이 아님}$
    - F-statistic: $F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)}$
- 선형회귀를 위한 가정들이 맞다면 $E\{RSS/(n-p-1)\} = \sigma^2$
- $H_0$이 옳다면 $E\{(TSS-RSS)/p\} = \sigma^2$
    - ⇒ F는 1과 가까운 값을 가진다.
- $H_a$이 옳다면 $E\{(TSS-RSS)/p\} > \sigma^2$
    - ⇒ F는 1보다 큰 값을 가진다.
- F가 얼마나 커야 $H_0$를 기각할 수 있는지는 n과 p의 값에 따라 다르다.
    - n이 큰 경우 F가 1보다 조금만 더 커도 $H_0$을 기각할 수 있는 반면, n이 작으면 $H_0$을 기각하기 위해서 더 큰 F값이 필요하다.
- $H_0$이 참이고 error들($\epsilon_i$)이 정규분포를 따른다면, F-statistic은 F분포를 따른다.  ⇒ p-value를 구할 수 있다. ⇒ p-value에 따라 $H_0$을 기각하거나 하지 않을 수 있다.
- 각각의 독립변수에 대한 p-value를 통해 그 변수가 종속변수와 연관성이 있는지 확인할 수 있음에도 불구하고 F-statistic으로 검정해야하는 이유:
    - 독립변수가 많을 경우 종속변수와 연관성이 없음에도 각각의 p-value가 0.05보다 잦을 가능성이 높아짐
- F-statistic을 활용하는 방식은 독립 변수의 개수 p가 n보다 작은 경우에만 사용할 수 있음
    - p가 n보다 큰 경우에는 least squares 방식으로 모델을 훈련시킬 수 조차 없음.

### 중요한 변수 고르기

- F-statistic을 사용하여 귀무가설을 기각했다면 어떤 변수가 종속변수와 연관이 있는지 찾아야한다. 이것을 variable selection(변수선택)이라고 한다.
- 변수의 개수가 적을 경우 모든 경우 수의 모델을 만들고 Mallow’s $C_p$, Akaike information criterion (AIC), Bayesian information criterion (BIC), adjusted $R^2$와 같은 통계량을 활용하여 가장 좋은 모델을 고를 수 있다.
- 변수의 개수 많을 경우, 만들 수 있는 모델의 경우의 수는 기하급수적으로 커지기 때문에 모든 모델을 시도해보는 것은 실용적이지 않다. 이러한 경우 다음과 같은 방식을 사용할 수 있다.
    - Forward selection: 절편만 있는 모델을 시작으로 추가했을 때 가장 낮은 RSS을 보이는 변수를 하나씩 추가시키고 모델을 다시 학습시킨다. Stopping rule을 만족할 때까지 이 과정을 반복한다.
    - Backward selection: 모든 독립변수를 포함하고 있는 모델을 시작으로 가장 큰 p-value를 가지는 변수를 하나씩 제거한고 다시 학습시키다. Stopping rule을 만족할 때까지  이 과정을 반복한다.
    - Mixed selection: 절편만 있는 모델을 시작으로 추가했을 때 가장 RSS를 보이는 변수를 하나씩 추가하며 학습시킨다. 추가했을 때 어떤 변수의 p-value가 일정 값보다 커진다면 그 변수를 모델에서 제거한다. 모델 안 모든 변수의 p-value가 일정 값보다 낮고, 모델 밖에는 추가했을 때 p-value가 큰 변수들만 남을 때까지 추가하고 제거하는 과정을 반복한다.

### 모델이 얼마나 데이터와 일치하는가?

- 독립변수가 변수가 종속변수와 큰 연관성이 없더라도 모델에 변수를 추가할수록 $R^2$은 항상 증가하게 된다.
- 독립변수를 추가했을 때 $R^2$가 조금밖에 증가하지 않아도 모델에 변수를 추가하는 것은 실제 모델의 성능을 강화시키는 것이 아니기 때문에 테스트 샘플에서 과적합을 유발하여 안 좋은 성능을 보일 수 있다.
- $RSE = \sqrt{\frac{1}{n-p-1}RSS}$
- 모델에 변수를 추가하였을 때 변수의 개수 p가 증가한 것 보다 RSS의 감속 적다면 결과적으로 RSE가 증가할 수 있다.
- 때때로 데이터를 시각화하는 graphical summary를 통해 통계량에서 확인할 수 없는 것들을 발견할 수 있다.

### 예측

- 모델을 학습시킨 후에 그 모델을 사용하여 종속변수 Y를 예측할 때에 3가지 불확실성이 존재한다.
    - $\beta_0, \beta_1, \dots , \beta_p$의 coefficient estimates인 $\hat\beta_0, \hat\beta_1, \dots, \hat\beta_p$로 이루어진 least squares plane $\hat Y = \hat\beta_0 + \hat\beta_1 X_1 + \cdots +\hat\beta_p X_p$는 true population regression plane $f(X) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$의 추정값이다. 우리는 $\hat Y$가 $f(X)$에 얼마나 가까운지 알기 위해 신뢰구간을 구할 수 있다.
    - $f(X)$에 대하여 우리는 linear model일 것이라 가정하기 때문에 model bias라고 하는 reducible error가 존재한다.
    - 우리가 실제 $f(X)$를 안다고 하더라도 random error $\epsilon$으로 인하여 정확한 예측을 할 수 없다 (irreducible error). $Y$가 $\hat Y$로부터 얼마나 다를지는 prediction interval (예측 구간)을 사용하여 답할 수 있다.
        - Prediction interval은 $f(X)$를 추정하여면서 발생한 에러(reducible error)와 각각의 데이터가 $f(X)$에서 얼마나 떨어져 있을지에 대한 불확실성 (irreducible error)를 포함하고 있기 때문에 confidence interval(신뢰구간)보다 항상 범위가 넓다.

# 3.3 Other Considerations in the Regression Model

## 3.3.1 Qualitative Predictors

### Predictors with Only Two Levels

- 만약 질적 변수(qualitative predictor)가 두 가지의 level을 가지고 있을 경우, dummy variable을 만들어 사용할 수 있다.
- 예: $x_i = \begin{cases} 1, &\text{if ith person owns a house } \\ 0, &\text{if ith perosn does not own a house} \end{cases}$
- $y_i =  \beta_0 + \beta_1 x_1 + \epsilon_i = \begin{cases} \beta_0 + \beta_1 + \epsilon_i &\text{if ith person owns a house} \\ \beta_0 + \epsilon_i, &\text{if ith perosn does not} \end{cases}$

### Qualitative Predictors with More than Two Levels

- Dummy variable을 더 추가한다.
    - $x_{i1} = \begin{cases} 1, &\text{if ith person is from the South } \\ 0, &\text{if ith perosn is not from the South} \end{cases}$
    - $x_{i2} = \begin{cases} 1, &\text{if ith person is from the West } \\ 0, &\text{if ith perosn is not from the West} \end{cases}$

$$
\begin{align} y_i & = \beta_0 + \beta_1 x_{i1} + \beta_{2} x_{i2} + \epsilon_i \\ &= \begin{cases} \beta_0 + \beta_1 + \epsilon_i &\text{if ith person is from the South} \\ \beta_0 + \beta_2 + \epsilon_i, &\text{if ith person is from the West} \\ \beta_0 + \epsilon_i, &\text{if ith perosn is from the East} \end{cases} \end{align}
$$

- Dummy variable의 개수는 항상 level의 개수보다 하나 작다.

## 3.3.2 Extensions of the Linear Model

- 선형회귀 모델의 경우 꽤 제한적인 가정을 지켜야한다.
    - Additivity assumpition: 독립 변수  $X_j$와 종속변수 $Y$의 연관성은 다른 독립변수의 영향을 받으면 안 된다.
    - Linearity assumption: 독립변수 $X_j$가 한 단위만큼 바뀌었을 때 $Y$의 변화는 $X_j$의 값에 상관없이 항상 일정해야 한다.

### Removing the Additive Assumption

- 하나의 독립변수가 다른 독립변수와 종속변수의 연관성에 영향을 미칠 때 이를 interaction effect 혹은 synergy effect라고 한다.
- 예:
    
    $$
    \begin{align} Y &= \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \epsilon \\ &= \beta_0 + (\beta_1 + \beta_3 X_2)X_1 + \beta_2 X_2 + \epsilon \\ &= \beta_0 + \tilde \beta_1 X_1 + \beta_2 X_2 + \epsilon \end{align}
    $$
    
    - $\tilde \beta_1 = \beta_1 + \beta_3 X_2$
- $\tilde \beta_1$은 $X_2$에 대한 함수이기 때문에 $X_1$과 $Y$의 연관성은 일정하지 않고, $X_2$의 값에 영향을 받는다.
- 모델에 interaction term을 추가하였을 때 때때로 interaction term의 p-value는 낮지만, 관련된 main effect들의 p-value는 높을 때가 있다. 이때 hierarchical principle에 때라 interaction term을 모델에 유지하기 위해서는 p-value가 충분히 낮지 않더라도 관련된 main effect들을 모델에 포함시켜야 한다.
- 질적 변수 사이의 혹은 질적 변수와 양적 변수 사이의 interaction term도 가능하다.
- 예)
    
    $$
    \begin{align} Y &\approx \beta_0 + \beta_1 \times X + \begin{cases} \beta_2 + \beta_3 \times X, &\text{if student} \\ 0, &\text{if not student} \end{cases} \\ &= \begin{cases} (\beta_0 +\beta_2) + (\beta_1 + \beta_3) \times X, &\text{if student} \\ \beta_0 + \beta_1 \times X, &\text{if not student} \end{cases}\end{align}
    $$
    
    - 학생이냐 아니야에 따라서 모델의 절편과 기울기가 달라짐.

### Non-linear Relationship

- 선형회귀모델은 기본적으로 전형관계를 가정하지만, 관계가 선형이 아닌 경우 변수의 제곱값을 갖는 변수로 갖는 선형모델을 사용할 수 있다.
- 세제곱, 네제곱, 혹은 그 이상의 값을 갖는 변수를 추가할 수도 있지만, 모델의 항이 너무 높아지면 과적합할 수 있다.
- 선형 모델을 비선형관계를 위해 확장하는 방식을 polynomial regression이라고 한다.

## 3.3.3 Potential Problems

- 선형 회귀 모델을 사용할 때 발생할 수 있는 문제들

### 1. Non-linearity of the Data

- 선형 회귀 모델은 독립변수와 종속변수 사이에 선형 관계가 있다고 가정한다. 만약 독립변수와 종속변수의 관계가 선형적이지 않을 경우 모델의 정확도가 매우 낮아질 수 있고, 모델에서 도출한 결론들 또한 정확하지 않을 수 있다.
- 잔차(residuals)와 독립변수 혹은 예측값($\hat y_i$)의 도표인 residual plot에서 눈에 띄는 패턴이 존재하지 않는다면 선형 관계가 존재한다고 볼 수 있다.
- Residual plot에서 팬턴이 발견되어 비선형 관계가 존재한다고 보여진다면 $log X, \sqrt{X}, X^2$와 같은 비선형 변환(non-linear transformation)을 시도해볼 수 있다.

### 2. Correlation of Error Terms

- 선형회귀모델의 중요한 가정 중 하나는 error terms ($\epsilon_1, \epsilon_2, \dots, \epsilon_n$) 사이에 상관 관계가 존재하면 안 된다는 것이다. (uncorrelated)
- 즉 $\epsilon_i$의 부호와 $\epsilon_{i+1}$의 부호가 아무런 관련이 없어어야 한다는 것.
- 만약 error terms 사이에 상관관계가 존재한다면 standard error를 실제 standard error보다 더 작게 추정하여 신뢰구간을 더 작게 만들 것이다. p-value 또한 원래보다 더 작은 값을 보여주어 잘못된 모수를 모델에 포함하게 될 수 있다. ⇒ 모델을 신뢰할 수 없어진다.
- Error 사이에 상관성이 없다는 가정은 선형회귀와 다른 statistical method들에서 특히 중요하다.

### 3. Non-constant Variance of Error Terms

- Error term들의 분산($Var(\epsilon_i) = \sigma^2$) 역시 선형 회귀 모델에서 중요한 가정이다.
- Standard errors, 신뢰구간, 가설검정이 이 가정을 바탕으로 하고 있다.
- Error terms의 분산이 일정하지 않을 경우 (heteroscedastic), Y를 $log Y$나 $\sqrt{Y}$로 변환시키는 방법이 있다.

### 4. Outliers

- 이상치(outlier)는 모델(절편과 기울기)에 영향을 미칠 수 있을 뿐만 아니라 RSE, 신뢰구간, $R^2$에 영향을 미칠 수 있다.
- 이상치는 residual plot을 사용하여 발견할 수 있고, 추정된 estimated standard error로 나눈 각각의 잔차인 studentized residual을 표시한 도표를 사용할 수도 있다. 만약 studentized residual이 3보다 크다면 이상치로 판단할 수 있다.

### 5. High Leverage Points

- 이상치가 주어진 $x_i$에 비해 $y_i$가 이상한 데이터라면 hight leverage point는 이상한(멀리 떨어진)  $x_i$값을 갖는 데이터.
- High leverage point는 regression line을 추정하는데 큰 영향을 미친다.
- 단순선형회귀에서 leverage point를 찾은 것은 단순히 정상적인 범위 밖의 데이터를 찾는 것이기 때문에 단순하다.
- 반면 다중선형회귀의 경우 leverage point가 각각의 독립변수에서는 정상적인 범위 안에 존재하지만 전체적인 독립변수들을 봤을 때 드문 위치에 존재할 수 있기 때문에 leverage point를 찾는 것은 어렵다.
- 따라서 Leverage statistic($h_i$)를 사용하여 high leverage를 가지고 있는 데이터를 검출할 수 있다.
    - $h_i$는 $1/n$과 $1$ 사이의 값을 가지고, 평균 leverage는  항상 $(p+1)/n$의 값을 갖는다.
    - $h_i$가 $(p+1)/n$보다 훨씬 큰 leverage statistic 값을 갖는 데이터를 high leverage point로 의심해 볼 수 있다.

### 6. Collinearity

- Collinearity(공산성)는 두 개 이상의 독립변수가 서로 밀접하게 연관되어 있는 상황을 의미.
- 공산성이 존재할 경우 각각의 독립변수가 종속변수에 미치는 영향을 분리하기 힘들어 회귀 모델에 문제가 된다.
- 공산성은 regression coefficient 추정값의 정확도를 낮춤 ⇒ $\hat\beta_j$의 starndard error가 커짐 ⇒ t-statistic이 감소함 ⇒ 귀무 가설($H_0: \beta_j = 0$)을 기각할 수 있는 가능성이 낮아짐
- 공산성이 존재하는지 알 수 있는 간단한 방법은 correlation matrix를 확인하는 것. 하지만 공산성이 세개 이상의 독립변수 사이에 존재하는 경우(multicollinearity, 다중공산성) correlation matrix으로 확인하기 어려움.
- Variance inflation factor (VIF): 모든 변수를 포함한 모델에서 $\hat\beta_j$의 부산을 $\hat\beta_j$만 있는 모델에서 $\hat\beta_j$의 분산으로 나누 것.
    - VIF는 1 이상의 값을 가지고, 1인 경우 공산성이 전혀 없다는 것을 의미
    - VIF가 5나 10일 초과할 경우 심각할 정도의 공산성이 있다고 판단.
    - $VIF(\hat\beta_j) = \frac{1}{1- R^2_{X_j|X_{-j}}}$
        - $R^2_{X_j|X_{-j}}$는 $X_j$를 제외한 모든 독립변수로 두고 $X_j$를 종속변수로 두고 회귀분석을 하였을 때 $R^2$.
        - $R^2_{X_j|X_{-j}}$이 1과 가깝다면 공산성이 존재하는 것이고, VIF는 큰 값을 가짐.
- 공산성이 존재할 경우
    1. 문제가 되는 변수 중 하나를 모델에서 제거한다. 혹은,
    2. 공산성이 존재한 변수를 하나로 합친다.

# 3.4 The Marketing Plan

# 3.5 Comparison of Linear Regression with K-Nearest Neighbors

- 선형회귀는 $f(X)$의 형태를 선형함수의 형태로 가정하기 때문에 parametric 접근법이다.
- Parametric method는 학습이 쉽고 해석하기 쉽다는 장점이 있지만 $f(X)$에 대하여 잘못 가정할경우 모델의 성능이 낮아진다는 단점이 있음.
- 반면 non-parametric method는 $f(X)$의 형태에 대하여 가정하지 않기 때문에 회귀를 하는데 있어 더 유연한 접근이 가능하다.
- K-nearest neighbors regression (KNN regression) 역시 non-parametric method이다.
- KNN regression은 우선 주어진 데이터 $x_0$와 가장 가까운 K개의 학습데이터를 찾아 평균을 내어 $f(x_0)$을 추정한다.
    - $\hat f(x_0) = \frac{1}{K} \sum_{x_i \in\mathcal{N}_0} y_i$
- Parametric approach가 f의 형태를 근접하게 정했을 때 non-parametric approach보다 더 좋은 성능을 낸다.
- 또, 예측변수가 많은 경우 KNN의 성능이 선형 회귀보다 떨어질 수 있다.
    - 보통의 경우 각각의 독립변수의 데이터가 적을 경우 parametric method가 non-parametric method보다 더 좋은 성능을 낸다. (차원의 저주, curse of dimensionality)
- 차원이 낮을 때에도 해석을 위하여 KNN보다 선형 회귀를 선호할 수도 있다.