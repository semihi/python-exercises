from pyautogui import *
import pyautogui
import cv2
import numpy as np
import time
import keyboard
import win32api, win32con
import threading
import math

def click(x, y, bombs):
    abort = False
    for bomb in bombs:
        if math.dist([x,y],bomb) < 80:
            # print("abort")
            abort = True
            break
    if not abort and not pause_event.is_set():

        win32api.SetCursorPos((x, y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.006)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        time.sleep(0.001)
    # elif pause_event.is_set():
    #     print("on pause")


print("started")

stop_event = threading.Event()
pause_event = threading.Event()

def checkStop():
    while not stop_event.is_set():
        if keyboard.is_pressed('q'):
            stop_event.set()
        time.sleep(0.05)

def checkGameOver():
    while not stop_event.is_set():

        if pyautogui.pixel(555,1100) > (243,243,243):
            pause_event.set()

        elif pyautogui.pixel(520,1350) > (192,243,86):
            pause_event.set()
            win32api.SetCursorPos((520, 1350))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
            time.sleep(0.006)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            time.sleep(0.001)
            # print("paused")
        else:
            pause_event.clear()
            # print("resumed")
        time.sleep(0.01)


def scan(region):
    while not stop_event.is_set():
        scrRgb = np.array(pyautogui.screenshot(region=region))
        scrHsv = cv2.cvtColor(scrRgb, cv2.COLOR_RGB2HSV)
        # scrBgr = cv2.cvtColor(scrRgb, cv2.COLOR_RGB2BGR)
        maskGreen = cv2.inRange(scrHsv, (29, 85, 215), (47,255,255))
        maskBomb = cv2.inRange(scrRgb, (127,122,119), (190,186,177))
        # cv2.imshow("maskGreen ", maskGreen)
        # cv2.waitKey(1)
        # cv2.imshow("maskBomb ", maskBomb)
        # cv2.waitKey(1)
        
        contours, _ = cv2.findContours(maskGreen, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contoursBomb, _ = cv2.findContours(maskBomb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        bombs = []
        for bomb in contoursBomb:
            if cv2.contourArea(bomb) >= 30:
                M = cv2.moments(bomb)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    bombs.append([5+cX,200+cY+20])
                    # cv2.circle(scrBgr, (cX, cY), 4, (255, 10, 30), -1)

        i = 0
        for contour in contours:
            if cv2.contourArea(contour) >= 100:
                M = cv2.moments(contour)
                
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    click(5+cX, 200+cY+40, bombs)
                    time.sleep(0.002)
                    # cv2.circle(scrBgr, (cX, cY), 4, (10, 10, 255), -1)
                    
                    i += 1
                    if i == 4:
                        break
        # print(time.time())
        time.sleep(0.02)  # 50 - 80 milisaniye arası
        # cv2.imshow('Centers ', scrBgr)
        # cv2.waitKey(1)

region = (5,200,635,1000)

threads = [threading.Thread(target=checkStop), threading.Thread(target=scan, args=(region,)), threading.Thread(target=checkGameOver)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print("quit")
