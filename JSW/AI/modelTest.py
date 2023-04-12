import torch, timm, cv2, numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from customDataset import CustomDataset, TRANSFORM50, TRANSFORM128, TRANSFORM256
from Capture_Image import CaptureFrame

def sliceImage():
    while(CF.capture()):
        ROI = CF.getROI()
        ## 추출된 관심 영역 확인
        try:
        
            ### 지금은 일반적으로 하나씩 출력
            ## 추후 멀티 프로세싱으로 동시에 처리
            leftState = getState(ROI[0], leftEyeModel)
            rightState = getState(ROI[1], rightEyeModel)
            mouthState = getState(ROI[2], mouthModel)
            print(f'left Eye : {leftState}, right Eye : {rightState}, mouth : {mouthState}')

            CF.Frame = cv2.putText(CF.Frame,f'leftEye : {str(leftState)}',(20,30),1,1.2,(0,0,255),2)
            CF.Frame = cv2.putText(CF.Frame,f'rightEye : {str(rightState)}',(220,30),1,1.2,(0,0,255),2)
            CF.Frame = cv2.putText(CF.Frame,f'mouth : {str(mouthState)}',(420,30),1,1.2,(0,0,255),2)

            cv2.imshow("right Eye", ROI[1])
            cv2.imshow("left Eye", ROI[0])
            cv2.imshow("mouth", ROI[2])
            cv2.imshow("Face Mesh", CF.Frame)

            cv2.moveWindow("right Eye", 500, 200)
            cv2.moveWindow("left Eye", 300, 200)
            cv2.moveWindow("mouth", 400, 400)
            cv2.moveWindow("Face Mesh", 600, 200)

            cv2.waitKey(5)

        except Exception as e:
            print(e)
            cv2.destroyAllWindows()
            continue

def getState(src, Model):
    src = Image.fromarray(np.uint8(src))
    inputs = transforms[testType](src)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = Model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    
    return className[predicted.item()]

if __name__ == "__main__":

    CF = CaptureFrame()

    # 테스트 모델 타입 설정
    testType = '256'

    # 트랜스폼 지정
    transforms = {'50':TRANSFORM50,'128': TRANSFORM128,'256': TRANSFORM256}
    print(transforms[testType])

    className = ['OPEN', 'CLOSE']
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # 모델 생성 및 초기화 모델을 GPU로 이동
    leftEyeModel = timm.create_model('convnext_tiny', num_classes=2, pretrained=True).to(device)
    rightEyeModel = timm.create_model('convnext_tiny', num_classes=2, pretrained=True).to(device)
    mouthModel = timm.create_model('convnext_tiny', num_classes=2, pretrained=True).to(device)
    # 저장된 모델 불러오기
    leftEyeModel.load_state_dict(torch.load(f'leftEye_Model_{testType}.pth', map_location=device))
    rightEyeModel.load_state_dict(torch.load(f'rightEye_Model_{testType}.pth', map_location=device))
    mouthModel.load_state_dict(torch.load(f'mouth_Model_{testType}.pth', map_location=device))

    # 모델 평가 모드 전환
    leftEyeModel.eval()
    rightEyeModel.eval()
    mouthModel.eval()

    sliceImage()