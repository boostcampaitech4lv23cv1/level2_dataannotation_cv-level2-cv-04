![mosaic1](https://user-images.githubusercontent.com/113182482/207420824-8fb78476-e316-4765-b4be-8a1ec7524159.jpg)

4사분면을 같은 크기로, crop 없이 resize 만 하는 naive mosaic 구현입니다

utils/mosaic.ipynb
#FIXME 아래
얘를 들어 ICDAR17_Korean 대 aihub 데이터를 3:1 비율로 쓰고 싶으시면
dataset_0,1,2 에 ICDAR17_Korean을, dataset_3 에 aihub 를 쓰시면 됩니다 (순서 상관 없음)
<img width="718" alt="image" src="https://user-images.githubusercontent.com/113182482/207422668-084bae84-cb21-4eec-a57a-92338da5d288.png">



현재는 SCALE_RANGE 가 (0.5, 0.5) 로 고정이어서 4사분면 나누는 축이 정중앙입니다.
항상 이미지 같은 위치에 경계선이 생기면 학습에 문제가 될 것 같아서 축 위치를 

![demo1](https://user-images.githubusercontent.com/113182482/207421902-48f8937e-1d94-4476-b1cc-399b3971540f.jpg)
처럼 일정하지 않게 하는 기능 수요일에 추가할 예정입니다


.ipynb 파일 끝까지 실행하시면
<img width="293" alt="image" src="https://user-images.githubusercontent.com/113182482/207419493-9afaf5d9-3df4-4a7c-ac1b-6b1f6c95a00c.png">
처럼 images 와 ufo 포맷의 train.json 이 나옵니다


-based on https://github.com/jason9075/opencv-mosaic-data-aug by jason9075