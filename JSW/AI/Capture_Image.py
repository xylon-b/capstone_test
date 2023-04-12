import cv2, mediapipe as mp, keyboard as kb, os, time as t, numpy as np
from DataAugmentation import dataAugmentaion

class CaptureFrame:
    def __init__(self, Type = None):
        if Type != None:
            self.trainPath = r""
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles
    ### 리스트 앞 부터 왼쪽 눈, 오른쪽 눈, 입 특징점 번호
    ROIs = [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]]
    ### 비디오 영상 불러오기
    Cap = cv2.VideoCapture(0)
    ### Frame을 저장할 변수 : ndarray 타입
    Frame : np.ndarray
    ### FaceMesh가 그려질 Frame 변수 : ndarray 타입
    meshFrame : np.ndarray
    ### Frame의 세로, 가로 크기 : int형
    h : int
    w : int
    ### Timer
    T = t.time()
    ### input Key
    key = None
    ### 데이터 저장 위치
    trainPath = './Train DataSet/'
    testPath = './Test Image/'
    ### 데이터 저장 Dict
    frameNames = {0:['leftEye/','rightEye/'], 1:['mouse/'], 2:['mask/']}
    savePaths = {'1':['_L_OE.jpg','_R_OE.jpg'],'2':['_L_CE.jpg','_R_CE.jpg'],'3':['_M_CM.jpg'],'4':['_M_OM.jpg'],'5':['_M_MSK.jpg']}
    ### 반복 수 체크
    cnt = len(os.listdir(f'{trainPath}leftEye/'))//2
    limit = cnt + 150

    ## 데이터 증강 객체
    DA = dataAugmentaion()

    ### Frame 추출하는 함수 : 추출 성공시 True 반환, 실패시 False 반환
    def capture(self):
        ## 비디오 영상에서 Frame 추출 : ret -> 추출 성공 :True, 실패 : False
        ret, self.Frame = self.Cap.read()

        ## 추출 실패시 오류 메시지 출력 후 False 반환
        if not ret:
            print('Nothing Captured')
            return False
        
        ## Frame 변수에 추출된 Frame의 좌우 반전 후 RGB 컬러를 BGR 컬러로 변환
        ## mediapipe가 BGR 이미지의 인식에 강함
        self.Frame = cv2.cvtColor(cv2.flip(self.Frame, 1), cv2.COLOR_RGB2BGR)
        ## Frame의 세로, 가로 값 저장
        self.h, self.w = self.Frame.shape[:2]
        ## 정상적으로 연산이 완료되면 True 반환
        return True
    
    def getROI(self):
        ## Right, Left, Mouse 특징점을 임시로 담을 리스트
        rois = []

        ## mediapipe Face Mesh사용
        with self.mp_face_mesh.FaceMesh(
            min_detection_confidence = 0.5,
            max_num_faces = 1,
            min_tracking_confidence = 0.5
        ) as face_mesh:
            ## Face Mesh 연산 후 반환 값 저장
            Result = face_mesh.process(self.Frame)
            ## BGR 영상을 보기 편하게 다시 RGB 영상로로 변환
            self.Frame = cv2.cvtColor(self.Frame, cv2.COLOR_BGR2RGB)

            ## Face Mesh 연산결과에서 각 landmark만 추출
            landMarks = Result.multi_face_landmarks
            
            ## landmark가 추출되었다면 연산 진행
            ## landmark가 없다면 빈 rois 반환
            if landMarks:
                for landMark in landMarks:
                    ## 추출된 landmark에서 landmark 좌표 정보 추출
                    facePos = landMark.landmark
                    ## BGR 이미지 RGB로 전환
                    self.meshFrame = self.Frame.copy()
                    ## frame에 face mesh 그려주기
                    self.mp_drawing.draw_landmarks(
                        image = self.meshFrame,
                        landmark_list = landMark,
                        ## mesh point 기반 3D 모델 생성
                        connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                        ## mesh에 점 없애기
                        landmark_drawing_spec = None,
                        ## mesh point 선 얇은 스타일로 변경
                        connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    xs = []
                    ys = []
                    ## 인식된 얼굴의 크기 구하기
                    for i in range(len(facePos)):
                        xs.append(facePos[i].x)
                        ys.append(facePos[i].y)
                    fx = int(min(xs)*self.w)
                    fy = int(min(ys)*self.h)
                    f_x = int(max(xs)*self.w)
                    f_y = int(max(ys)*self.h)
                    ## 전체 영상 크기와 인식된 얼굴 크기 비율 구하기
                    ## 전체 이미지 비율과 추출된 특징 영상 크기 비율을 비교하여 실제 추출할 영역 고정 시키기
                    x_ratio = int((f_x - fx)/self.w*100)
                    y_ratio = int((f_y - fy)/self.h*100)

                    ## 정의된 randmark 번호의 좌표 정보를 이용해
                    ## 눈, 입 주변 영역 이미지 추출
                    for roiPos in self.ROIs:
                        ## 초기 관심 영역의 최소, 최대 x, y값 설정
                        # 최소는 image 원본의 w, h 값 최대는 0부터 측정
                        minX, minY, maxX, maxY = self.w, self.h, 0, 0

                        # 촬영된 영상의 mediapipe 좌표에서 각 좌표상 x, y값 추출
                        for key in roiPos:
                            lx = int(facePos[key].x * self.w)
                            ly = int(facePos[key].y * self.h)

                            # 추출된 값들중 최소, 최대 x, y값 찾기
                            if lx < minX:
                                minX = lx
                            if ly < minY:
                                minY = ly
                            if lx > maxX:
                                maxX = lx
                            if ly > maxY:
                                maxY = ly
                        
                        roiFrame = self.Frame[minY-y_ratio:maxY+y_ratio, minX-x_ratio:maxX+x_ratio].copy()

                        h, w = roiFrame.shape[:2]

                        if h != 50 and w != 50:
                            if h < 100 or w < 100:
                                try:
                                    roiFrame = cv2.resize(roiFrame, dsize=(50,50),interpolation=cv2.INTER_CUBIC)
                                except:
                                    continue
                            else : 
                                try:
                                    roiFrame = cv2.resize(roiFrame, dsize=(50,50),interpolation=cv2.INTER_AREA)
                                except:
                                    continue
                        rois.append(roiFrame)
            else : rois = [[],[],[]]
        return rois

    def showFrame(self, l, r, m):
        cv2.imshow('Origin Frame',self.Frame)
        cv2.imshow('Mesh Frame',self.meshFrame)
        cv2.imshow('leftEye',l)
        cv2.imshow('rightEye',r)
        cv2.imshow('Mouse',m)

        cv2.moveWindow('Origin Frame',100,100)
        cv2.moveWindow('Mesh Frame',900,100)
        cv2.moveWindow('leftEye',100,700)
        cv2.moveWindow('rightEye',250,700)
        cv2.moveWindow('Mouse',400,700)
        
        cv2.waitKey(10)
    ## 이미지를 저장할 때 Data Augmentation 적용해서 저장하기
    def save(self, Names, inputKey, Frames):
        savePaths = self.savePaths[inputKey]
        for i, savePath in enumerate(savePaths):
            cv2.imwrite(rf'{self.trainPath}{Names[i]}{self.cnt}{savePath}',Frames[i])
            cv2.imwrite(rf'{self.trainPath}{Names[i]}{self.cnt+1}{savePath}', self.DA.augmentationImage(Frames[i]))
            cv2.imwrite(rf'{self.trainPath}{Names[i]}{self.cnt+2}{savePath}', self.DA.augmentationImage(Frames[i]))
    
    def saveFrame(self, Frames):
        ## 현재 입력된 키가 없을 때만 새로운 키 입력 받음
        if self.key == None:
            self.key = '1' if kb.is_pressed('1') else '2' if kb.is_pressed('2') else '3' if kb.is_pressed('3') else\
                        '4' if kb.is_pressed('4') else None
            if self.key != None:
                print(f'key Click : {self.key}')
        else:
            ## 입력된 key가 있다면 저장 실행
            ## 0.1초마다 저장
            if t.time() - self.T >= 0.1:
                ## 정해진 횟수 만큼 저장
                if self.cnt < self.limit:
                    ## key 인덱싱을 위해서만 사용될 익명 함수
                    CalcKey = lambda x : int(x)**2//(5+int(x))
                    ## 입력 영상 인덱싱 계산
                    CalcFrame = lambda x, y : y[:2] if int(x)//3 == 0 else y[2:]
                    ## ROI 이미지 저장
                    self.save(self.frameNames[CalcKey(self.key)],self.key, CalcFrame(self.key,Frames))
                    self.cnt += 3
                else : 
                    ## 정해진 횟수 만큼 저장 하였다면
                    ## count 및 key 초기화
                    print('end')
                    self.cnt -= 150
                    self.key = None
                ## 시간 갱신
                self.T = t.time()

    def refresh(self):
        if kb.is_pressed('r'):
            self.cnt = len(os.listdir(f'{self.trainPath}leftEye/'))//2
            self.limit = self.cnt + 100
            os.system('cls')

    def Run(self):
        ## 영상 추출에 성공했을 때만 반복
        while(self.capture()):
            try:
                ## 관심 영역 추출하여 분리
                ROI = self.getROI()
                self.showFrame(*ROI)
                self.saveFrame(ROI)
                self.refresh()
            except Exception as e:
                print(f'ERROR : {e}')
                cv2.destroyAllWindows()
                continue

if __name__ == "__main__":
    Model = CaptureFrame()
    Model.Run()