# split annot  

거대한 데이터셋의 세트의 테스트를 위해서  
원하는 개수만큼의 이미지를 sampling 하여 json을 생성하는 util입니다.  
원본 annot는 <code>full_train.json</code>로 이름 변경하여 저장되며  
subsampling된 annot는 <code>train.json</code>로 저장이 됩니다.