# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:47:01 2019

@author: engus
"""


import io             # 이진 데이터를 통신으로 받기 위해 io 라이브러리 사용
import socket         # 소켓 통신을 위해 socket 라이브러리 import
import struct         # 파이썬에서 아스키 코드 값을 바로 처리하기 위해 struct 모듈을 사용
from PIL import Image # Image를 처리하기 위해 PIL 라이브러리 import
import numpy as np    # 데이터를 배열로 저장하기 위해 numpy 사용
import pandas as pd   # 수집 데이터를 데이터 프레임 형식으로 메모리 저장
import pickle         # pickle로 수집데이터 파일 저장
import time           # 촬영시간 반환
import cv2

# Start a socket listening for connections on 192.168.0.62: 8282

## host & port 셋팅 ##
host = '141.223.140.41'
port = 8282

#### 서버 세팅 ####
server_socket = socket.socket()       # 서버 소켓 객체 생성 ( 통신을 위해선 객체 생성이 필요 )
print('server_Socket created')        # 서버 소켓 생성 문구

server_socket.bind((host, port))      # 서버 소켓에 (host, port) 바인딩
print('server_Socket bind complete')  # 소켓 바인딩 완료 문구

server_socket.listen(5)               # 서버 소켓에 클라이언트 소켓 1개만 listening
print('Socket now listening')         # 클라이언트 1개를 받을 준비 문구

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb') # connection은 서버 소켓 객체
print('[{0}] connection complete'.format(host))


# 이미지 데이터를 메모리에 저장하기 위해 pd_image 생성
pd_image = pd.DataFrame({'image':[]})


try:
    # 입력 사진수를 출력 #
    count = 0
    start_time = time.time()
    Time=[time.time()]
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop

        # clinet에서 받은 stringData 크기
        # struct.unpack()은 little-endian으로 구성 된 unsigned long으로 unpack하겠다는 의미.
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0] # '<L'에서 '<'은 Little-Endian을 'L'은 unsigned long을 의미

        if not image_len:        # 만약 받은 데이터 길이가 0이라면 프로그램 중지
            break

        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO() # image length가 0이 아니라면 image data를 보관하기위한 stream
        image_stream.write(connection.read(image_len)) # connection으로 부터 image data를 읽는다.

        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)   #스트림된 데이터를 되돌린다.(순서대로 받았으므)
        image = Image.open(image_stream)

        # 이미지 생성
        np_image = np.array(image) # 받아온 데이터를 넘파이 배열로 구성

        pd_image = pd_image.append({'image':np_image}, ignore_index=True)



        count += 1
        Time.append(time.time())
        print('Count_image: {}'.format(count), end='')
        print(', Image is %d*%d'%image.size, end='')
        print(', diff_time %.3f'%(Time[-1]-Time[-2]))


        image.verify()
        print('Image is verified')

        # cv2.imshow('Stream', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = time.time() - start_time
    print('connected time: {}s'.format(end_time))
    with open('/home/pirl/posker/images/test_result_jy.pickle', 'wb') as f:
        pickle.dump(pd_image, f)
finally:

    print('end')
    connection.close()
    cv2.destroyAllWindows()

print('processing data...')

import run_getdata