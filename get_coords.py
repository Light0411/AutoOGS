import pyautogui
import time

print("Starting coordinate finder...")
print("Move your mouse to the desired location and press Ctrl-C to quit.")

try:
    while True:
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print('\nDone.')