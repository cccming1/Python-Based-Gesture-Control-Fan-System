import cv2
import mediapipe as mp
import math
import serial
import time
import sys

print("👉 gesture.py 文件已被 Python 运行")  # 调试用


# ================= 串口相关 =================

SERIAL_PORT = "/dev/tty.usbserial-210"  # 换成你串口测试成功的那个
BAUDRATE = 115200

ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    time.sleep(2)
    print("✅ 串口已打开:", ser.port)
except Exception as e:
    print("⚠ 串口打开失败:", e)
    ser = None


def set_fan(level: int):
    """
    通过串口发送风扇档位:
    0=关, 1=低速, 2=中速, 3=高速
    """
    global ser
    if ser is None or (not ser.is_open):
        print("⚠ 串口未打开，无法发送风扇命令")
        return

    if level < 0 or level > 3:
        return

    cmd = str(level).encode("ascii")
    try:
        ser.write(cmd)
        print(f"➡ 已发送风扇档位命令: {level}")
    except Exception as e:
        print("⚠ 发送串口失败:", e)


# ================= 手势识别相关 =================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pinch_threshold = 0.02
pinch_cooldown_frames = 4

pinch_cooldown = {'Left': 0, 'Right': 0}
pinch_active = {'Left': False, 'Right': False}


def calc_pinch_dist(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dist = math.sqrt(dx * dx + dy * dy)
    return dist


def main():
    print("👉 进入 main() 函数，开始初始化摄像头和 MediaPipe")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        sys.exit(1)
    print("✅ 摄像头已成功打开")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    global pinch_cooldown, pinch_active

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ 无法读取摄像头画面")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            for hand_label in ('Left', 'Right'):
                if pinch_cooldown[hand_label] > 0:
                    pinch_cooldown[hand_label] -= 1
                else:
                    pinch_active[hand_label] = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):
                    hand_label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    dist = calc_pinch_dist(hand_landmarks.landmark)
                    text = f"{hand_label} dist={dist:.3f}"
                    cv2.putText(frame, text,
                                (10, 30 if hand_label == 'Left' else 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)

                    is_pinch = dist < pinch_threshold

                    if is_pinch and not pinch_active[hand_label]:
                        pinch_active[hand_label] = True
                        pinch_cooldown[hand_label] = pinch_cooldown_frames

                        if hand_label == "Right":
                            print("✋ 右手检测到捏合 -> FAN HIGH")
                            set_fan(1)
                        elif hand_label == "Left":
                            print("✋ 左手检测到捏合 -> FAN OFF")
                            set_fan(0)

            cv2.putText(frame, "Press 'q' or ESC to quit",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

            cv2.imshow("Gesture Fan Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("🔚 检测到退出按键，准备退出")
                break

    except KeyboardInterrupt:
        print("⏹ 收到 Ctrl+C，退出程序")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None and ser.is_open:
            ser.close()
            print("🔌 串口已关闭")
        print("✅ main() 正常结束")


if __name__ == "__main__":
    print("👉 __name__ == '__main__' 成立，调用 main()")
    main()
else:
    print("⚠ __name__ 不是 '__main__'，这行只会在被 import 时出现")