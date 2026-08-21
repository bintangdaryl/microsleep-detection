import cv2
import dlib
import threading
import time
import numpy as np
import subprocess
from imutils import face_utils
from scipy.spatial import distance as dist
from scipy.signal import find_peaks, butter, filtfilt
import RPi.GPIO as GPIO 

# [TAMBAHAN CSV & LCD] Import library
import csv
from datetime import datetime
from RPLCD.i2c import CharLCD  # Library untuk LCD 16x2 I2C

# Pastikan letak file driver sensor max30102 sudah sesuai
from sensor.max30102 import MAX30102

# ================= SETUP GPIO RELAY =================
RELAY_PIN = 4  
GPIO.setmode(GPIO.BCM) 
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)

# ================= SETUP LCD 16x2 I2C =================
ALAMAT_I2C = 0x27  # Sesuaikan alamat I2C (biasanya 0x27 atau 0x3f)
lcd_aktif = False
try:
    lcd = CharLCD(i2c_expander='PCF8574', address=ALAMAT_I2C, port=1,
                  cols=16, rows=2, dotsize=8)
    lcd.clear()
    lcd_aktif = True
    print("LCD 16x2 I2C Berhasil Terhubung.")
except Exception as e:
    print(f"Peringatan: LCD tidak terdeteksi ({e}). Program tetap berjalan tanpa LCD.")

last_lcd_update = 0
interval_lcd = 0.3  # Update LCD setiap 0.3 detik agar tidak menurunkan FPS kamera

# ================= VARIABEL GLOBAL BPM =================
current_bpm = 0.0
sensor_status = "Menunggu Sensor..."
sensor_running = True

# ================= FUNCTION =================

def play_alarm():
    subprocess.Popen(
        [
            "cvlc",
            "--intf", "dummy",
            "--play-and-exit",
            "--no-video",
            "bangun.wav"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_LAR(lips):
    A = dist.euclidean(lips[2], lips[6])
    B = dist.euclidean(lips[0], lips[4])
    return A / B

def calc_bpm(data_mentah):
    try:
        nyq = 0.5 * 280
        b, a = butter(2, [1.0/nyq, 3.0/nyq], btype='band')
        data_bersih = filtfilt(b, a, data_mentah)
        puncak, _ = find_peaks(data_bersih, distance=140)
        
        if len(puncak) >= 2:
            jarak_antar_puncak = np.diff(puncak) / 280
            return 60 / np.mean(jarak_antar_puncak)
    except:
        pass
    return 0

def thread_baca_sensor():
    global current_bpm, sensor_status, sensor_running
    
    try:
        sensor = MAX30102()
    except Exception as e:
        sensor_status = "Sensor Error"
        return

    ir_data = []
    bpm_history = []
    IR_THRESHOLD = 70000 

    while sensor_running:
        try:
            red, ir = sensor.read_fifo()
            ir_data.append(ir)

            if len(ir_data) > 1400:
                ir_data.pop(0)

            data_terbaru = ir_data[-10:]
            
            # Logika Jari Lepas
            if not data_terbaru or max(data_terbaru) < IR_THRESHOLD:
                sensor_status = "Jari Lepas"
                current_bpm = 0.0
                bpm_history.clear()
                ir_data.clear() 
                
            # Logika Mulai Berhitung Cepat
            elif len(ir_data) >= 1120:
                bpm = calc_bpm(ir_data)
                if 30 < bpm < 200:
                    bpm_history.append(bpm)
                    
                    if len(bpm_history) > 300:
                        bpm_history.pop(0)
                    
                    current_bpm = sum(bpm_history) / len(bpm_history)
                    
                    if current_bpm <= 60:
                        sensor_status = "Over Relaks"
                    elif current_bpm <= 90:
                        sensor_status = "Normal"
                    else:
                        sensor_status = "Nervous"
                else:
                    sensor_status = "Menghitung..."
                    
            time.sleep(0.01)
            
        except Exception as e:
            time.sleep(0.01)

    if sensor:
        sensor.shutdown()

# ================= SETUP =================

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

lip_index = [60,61,62,63,64,65,66,67]

EAR_THRESH = 0.2
EAR_FRAMES = 5
LAR_THRESH = 0.4
LAR_FRAMES = 5

eye_counter = 0
lip_counter = 0
alarm_on = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

# Start BPM Thread
t_sensor = threading.Thread(target=thread_baca_sensor, daemon=True)
t_sensor.start()

# ================= SETUP CSV LOGGER =================
nama_file_csv = f"log_deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
file_csv = open(nama_file_csv, mode='w', newline='')
csv_writer = csv.writer(file_csv)
# Header dimodifikasi untuk merekam Status LCD ke dalam log
csv_writer.writerow(["Timestamp", "FPS", "EAR", "MAR", "Status_LCD", "BPM", "Relay_Active"])

# ================= VARIABEL KONTROL RELAY =================
waktu_relay_mati = 0.0
status_relay_menyala = False
relay_sudah_trigger = False  # Flag pengunci agar relay hanya menyala 1 kali per kejadian

# ================= LOOP =================

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # ===== MENGHITUNG FPS & WAKTU =====
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    current_ear = 0.0
    current_lar = 0.0
    is_drowsy = False
    relay_active = False

    for face in faces:
        shape = predictor(gray, face)
        points = face_utils.shape_to_np(shape)

        # ===== EYE =====
        leftEye = points[lStart:lEnd]
        rightEye = points[rStart:rEnd]

        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0
        current_ear = EAR

        # ===== LIPS =====
        lips = points[lip_index]
        LAR = calculate_LAR(lips)
        current_lar = LAR

        # ===== DRAW =====
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,0,255), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,0,255), 1)
        cv2.drawContours(frame, [cv2.convexHull(lips)], -1, (0,255,0), 1)

        # ===== LOGIC EYE & YAWN =====
        if EAR < EAR_THRESH:
            eye_counter += 1
        else:
            eye_counter = 0

        if LAR > LAR_THRESH:
            lip_counter += 1
        else:
            lip_counter = 0

        # ===== ALERT KANTUK (AUDIO ALARM) =====
        if eye_counter > EAR_FRAMES or lip_counter > LAR_FRAMES:
            is_drowsy = True
            cv2.putText(frame, "DROWSINESS ALERT!", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if not alarm_on:
                alarm_on = True
                threading.Thread(target=play_alarm,daemon=True).start()
        else:
            alarm_on = False

        # ===== DISPLAY EAR & MAR DI MONITOR =====
        cv2.putText(frame, f"EAR: {EAR:.2f}", (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"MAR: {LAR:.2f}", (300,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    # ================= PENENTUAN STATUS PENGEMUDI (3 KELAS LCD) =================
    status_pengemudi = "NORMAL" # Status default

    # 1. KELAS MICROSLEEP (Syarat Khusus: Hanya jika BPM di bawah 60)
    if 0 < current_bpm < 60:
        status_pengemudi = "MICROSLEEP"
    # 2. KELAS WASPADA (Syarat: Terdeteksi gejala kantuk dari mata/mulut)
    elif current_ear <= 0.20 or current_lar > 0.40:
        status_pengemudi = "WASPADA"
    # 3. KELAS NORMAL (Syarat: Mata/mulut normal & BPM rentang 60 - 90)
    elif current_ear > 0.20 and current_lar <= 0.40 and (60 <= current_bpm <= 90):
        status_pengemudi = "NORMAL"
    else:
        # Menangani kondisi transisi / di luar rentang standar
        status_pengemudi = "NORMAL"
    # ==============================================================================


    # ===== ALERT BPM RENDAH & KONTROL RELAY (ONE-SHOT 2 DETIK) =====
    # Disinkronkan dengan batas MICROSLEEP yaitu di bawah 60 BPM
    if 0 < current_bpm < 60:
        cv2.putText(frame, "BAHAYA: MICROSLEEP (BPM < 60)!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Hanya jalankan jika relay belum pernah di-trigger pada siklus drop ini
        if not relay_sudah_trigger:
            GPIO.output(RELAY_PIN, GPIO.LOW) # Nyalakan Relay (Active Low)
            waktu_relay_mati = new_frame_time + 2.0  # Aktif selama 2 detik
            status_relay_menyala = True
            relay_sudah_trigger = True # Kunci flag (jangan nyala lagi sebelum BPM normal)
            
    elif current_bpm >= 60:
        # Jika BPM sudah kembali normal di atas/sama dengan 60, reset kuncinya
        relay_sudah_trigger = False

    # Timer eksekusi pemadaman relay setelah 2 detik
    if status_relay_menyala:
        if new_frame_time >= waktu_relay_mati:
            GPIO.output(RELAY_PIN, GPIO.HIGH) # Matikan Relay
            status_relay_menyala = False
            relay_active = False
        else:
            relay_active = True
    else:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        relay_active = False


    # ===== UPDATE DISPLAY LCD 16x2 =====
    if lcd_aktif and (new_frame_time - last_lcd_update > interval_lcd):
        try:
            # Baris 1: Status Pengemudi (dipotong & dipadatkan maksimal 16 karakter)
            if status_pengemudi == "MICROSLEEP":
                teks_baris_1 = "** MICROSLEEP **"
            elif status_pengemudi == "WASPADA":
                teks_baris_1 = "STATUS: WASPADA "
            else:
                teks_baris_1 = "STATUS: NORMAL  "
            
            lcd.cursor_pos = (0, 0)
            lcd.write_string(teks_baris_1.ljust(16)[:16])
            
            # Baris 2: Parameter BPM dan EAR secara real-time (maksimal 16 karakter)
            teks_baris_2 = f"BPM:{current_bpm:.1f} E:{current_ear:.2f}"
            lcd.cursor_pos = (1, 0)
            lcd.write_string(teks_baris_2.ljust(16)[:16])
            
            last_lcd_update = new_frame_time
        except Exception:
            pass

    # ===== TAMPILAN DISPLAY DI MONITOR KOMPUTER =====
    cv2.putText(frame, f"FPS: {int(fps)}", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status LCD: {status_pengemudi}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"BPM: {current_bpm:.1f}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Sensor: {sensor_status}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ================= MENULIS LOG KE CSV =================
    waktu_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    csv_writer.writerow([
        waktu_sekarang, 
        round(fps, 2), 
        round(current_ear, 3), 
        round(current_lar, 3), 
        status_pengemudi, 
        round(current_bpm, 2), 
        relay_active
    ])

    cv2.imshow("Microsleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
sensor_running = False 
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup() 
file_csv.close()
if lcd_aktif:
    try:
        lcd.clear()
    except Exception:
        pass