## 데이터마이닝 기말과제 MI-FGSM 구현

원문 : https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf

>## 실험환경 및 패키지 버전
### 패키지 버전
- python >= 3.7
- tf >= 2.1
### 실험 환경
- Intel I7-9700
- 16GB RAM
- RTX2070 Super 8GB

> ## Abstract
&nbsp;&nbsp;&nbsp;&nbsp; 대부분의 딥러닝 모델들은 adversarial examples에 취약한 면을 보인다. adversarial example은 현재 Apple Inc.에서 Special Projects Group의 기계 학습 담당 이사로 근무하는 이안 굿펠로우(Ian Goodfellow)에 의해 제안되었다. 이미지를 분류하는 image Classifier의 입력 이미지를 특정 노이즈 데이터를 더해 의도적으로 분류기의 예측을 오분류를 일으킨다. 이러한 노이즈가 더해진 이미지를 adversarial example이라고 한다. 이렇듯 adversarial example이 실세계에 적용되면 인명 피해 및 재산 피해가 발생할 수 있을 것이다. 만약 자율주행 자동차의 예를 들면 자율주행 자동차의 Object detector가 Stop sign(정지 표지판)을 다른 표지판(예: 속도 제한 100)으로 오분류하여 자율주행 자동차를 오동작으로 유도할 수 있다. 이 경우 상황에 따라 경미한 피해로 끝날 수 있지만 큰 사고로 이어진다면 인명 피해까지 발생할 수 있을 것이다. 오늘날 인공지능은 하드웨어 기술이 발달하여 인공지능 관련 알고리즘들이 각광받기 시작하면서 4차 산업혁명이라고 불리고 있다. 이처럼 많은 인공지능 관련 알고리즘들이 실세계에서 여러 분야에 융합되어 우리의 일상생활에 자리를 잡기 시작했다. 사람 4명이서 일주일에 걸려 분석하는 주식 데이터를 15분만에 분석하는 등, 인간보다 더 뛰어난 연산 수행능력을 가진 인공지능은 인간의 편리를 위해서 많은 분야에서 적용되어 가고 있다. 그만큼 사람 대신에 기계가 처리해주는 일들이 많아지면서 기계의 안정성 및 신뢰성이 보장되어야 한다. 따라서 딥러닝에서 이러한 의도적이고, 악의적인 공격을 예방하기 위한 대안은 필수적이다. 악의적인 공격을 예방하기 위한 방어는 먼저 공격의 메커니즘을 이해하는 것이다. 어떻게 공격하는지 알아야 그에 맞는 방어도 할 수 있는 것이다. 따라서 공격과 방어의 경쟁은 선의 경쟁이라고 생각한다. 이번 파이널 과제의 문제에서는 single-step”공격 기법인 FGSM을 이용하여 adversarial example을 생성하였다. single-step 공격의 경우 블랙 박스 공격 메커니즘, Defensive Distillation이나 adversarial example에 대해 훈련된 모델에 대한 공격은 성공률이 다소 낮은 성공률을 보인다. 따라서 이러한 약한 부분을 보완하기 위해 본 논문에서는 강한 adversarial example을  생성하는 기법들의 논문들을 참고하여 구현해보았다.


>## Back Ground

1. Transferability
<br/><br/>![image](https://user-images.githubusercontent.com/48381447/121781874-bf26c000-cbe1-11eb-8c34-b92ea25a2417.png)<br>

&nbsp;&nbsp; Transferability란 그림과 같이 학습된 모델들이 비슷한 Decision Boundary를 가지는 특징을 이용하여 생겨난 현상을 말한다. 신경망 모델들이 서로 다른 구조를 가져도 똑같은 학습 데이터를 학습시켰을 때 비슷한 Decision Boundary를 가진다. 예를 들어서, Resnet, vgg 두 개의 모델에서 만약 Resnet 아키텍처에서 생성된 adversarial example들이 vgg 아키텍처에서도 adversarial example로서 동작이 가능하다. 이렇게 adversarial example이 transfer 가능한 특징을 가지고 있다고해서 이것을 transferability라고 한다. 즉, 어떤 모델에서 생성된 adversarial example은 다른 모델로 전파가 되는 것을 의미한다. 이러한 transferability의 특징을 이용해, 어떤 모델의 하이퍼 파라미터나 모델의 아키텍처를 모르는 상황에서 즉, 블랙 박스 상에서의 공격을 수행한다.

2. FGSM
One-step gradient-based approaches<br/>
<br/>![image](https://user-images.githubusercontent.com/48381447/121782089-e5992b00-cbe2-11eb-82a3-8f746a3b8c8f.png)<br><br>

3. I-FGSM 
Iterative method<br/>
<br/>![image](https://user-images.githubusercontent.com/48381447/121782433-6147a780-cbe4-11eb-8b98-92521387314b.png)<br>


>## MI-FGSM (Momemtum Iterative-Fast Gradient Sign Method)

&nbsp;&nbsp;&nbsp;&nbsp;MI-FGSM은 기존의 “multi-step” 공격 기법인 I-FGSM(Iterative-FGSM)을 개선한 기법이다. I-FGSM은 “single-step”공격 기법인 FGSM 보다 더 강한 공격 성능을 보이지만, transferability가 빈약해 블랙박스 공격에서의 성공률이 현저히 낮은 결과를 보인다. I-FGSM이 빈약한 transferability를 보이는 이유는 기울기를 안정적으로 업데이트하지 못해서이다. 따라서 모멘텀 기법을 적용하여 기울기를 보다 안정적으로 업데이트를 수행하여 빈약한 transferability를 보완한다.
따라서 MI-FGSM 공격 알고리즘은 FGSM 보다 강한 공격 성능을 보이는 I-FGSM에 모멘텀 기법을 적용하여 빈약한 transferability를 개선해 화이트 박스에서의 공격뿐만 아니라 블랙박스에서의 공격 성공률을 높혔다.

<br/>![image](https://user-images.githubusercontent.com/48381447/121782738-0747e180-cbe6-11eb-9843-d345c62dc5a8.png) <br/>

> ## 결과
&nbsp;&nbsp;&nbsp;&nbsp;구현은 원래 저자의 방법과 달리 앙상블 모델을 다르게 구성을 해봤다. 원래 저자의 앙상블 모델들은 Inception V3, Inception V4, Inception Resnet V2, Resnet-152, Inception V3-ens3,ens4, Inception Resnet V2-ens의 7가지 모델들로 구성되어 있고, 본인은 Inception V4 모델 대신 Mobile V2 모델을 적용하였고, 테스트 및 Adversarial example을 생성한 base model 모델도 Mobile V2 모델로 구현하였다. 실험은 Perturbation의 사이즈 은 8, 16으로 설정하였고, 전체 반복 횟수 T는 10으로 설정하여 실험을 수행하였다.
Mobile NET을 베이스모델로 생성한 Adversarial examples들은 화이트박스에서는 100%의 공격 성공률을 보였고, Inception V3 74.67%, Inception v3 ens3 84.67%, Inception v3 adv ens3 - 56.67%, Inception v3 adv ens4 - 54.67%, Inceptin res adv - 48%, Inception res adv ens = 24.67%, Resnet 152 - 80.67%의 공격 성공률을 보였다.

저자의 결과와 달리 블랙 박스에서의 공격 성능이 많이 떨어졌다. 저자가 제안한 방법에 따르면 위의 7 개 모델들로 부터 98% 이상의 공격 성공률을 보였다. 이후 논문에서 발표한 공격 성공률과 근사한 결과를 도출하기 위해 베이스 모델을 모바일 넷이 아닌 다른 모델들로 구성하여 실험을 해볼 것이다.


