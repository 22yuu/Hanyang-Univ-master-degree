## 데이터마이닝 기말과제 MI-FGSM 구현

### MI-FGSM (Momemtum I-FGSM)

I-FGSM은 기존의 “multi-step” 공격 기법인 I-FGSM(Iterative-FGSM)을 개선한 기법이다. I-FGSM은 “single-step”공격 기법인 FGSM 보다 더 강한 공격 성능을 보이지만, transferability가 빈약해 블랙박스 공격에서의 성공률이 현저히 낮은 결과를 보인다. I-FGSM이 빈약한 transferability를 보이는 이유는 기울기를 안정적으로 업데이트하지 못해서이다. 따라서 모멘텀 기법을 적용하여 기울기를 보다 안정적으로 업데이트를 수행하여 빈약한 transferability를 보완한다.
따라서 MI-FGSM 공격 알고리즘은 FGSM 보다 강한 공격 성능을 보이는 I-FGSM에 모멘텀 기법을 적용하여 빈약한 transferability를 개선해 화이트 박스에서의 공격뿐만 아니라 블랙박스에서의 공격 성공률을 높혔다.

![image](https://user-images.githubusercontent.com/48381447/121781874-bf26c000-cbe1-11eb-8c34-b92ea25a2417.png)
