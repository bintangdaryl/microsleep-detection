# Microsleep Detection System

## Deskripsi

Sistem pendeteksi microsleep secara real-time untuk memberikan peringatan dini kepada pengemudi. Sistem menggunakan Raspberry Pi, webcam, Python, OpenCV, dan Shape Predictor 68 Face Landmark untuk menganalisis kondisi wajah dan mata pengemudi.

## Tujuan

Proyek ini bertujuan untuk mengembangkan sistem yang mampu mendeteksi indikasi microsleep pada pengemudi secara real-time sehingga dapat memberikan peringatan dini untuk meningkatkan keselamatan berkendara.

## Teknologi

- Python
- OpenCV
- dlib
- Shape Predictor 68 Face Landmark
- Raspberry Pi 4
- Webcam
- MAX30102
- LCD I2C
- Relay
- Water Pump

## Cara Kerja

1. Webcam menangkap wajah pengemudi secara real-time.
2. Shape Predictor 68 Face Landmark mendeteksi titik-titik pada wajah.
3. Sistem mengambil landmark pada area mata.
4. Nilai Eye Aspect Ratio (EAR) dihitung untuk mengetahui kondisi mata.
5. Sistem mendeteksi indikasi microsleep berdasarkan nilai EAR dan durasi mata tertutup.
6. Sistem memberikan peringatan kepada pengemudi.
7. Sensor MAX30102 digunakan untuk membaca detak jantung.
8. LCD menampilkan status kondisi sistem.

## Fitur

- Deteksi wajah secara real-time
- Deteksi kondisi mata menggunakan EAR
- Deteksi microsleep
- Monitoring detak jantung
- Tampilan status melalui LCD
- Peringatan dini kepada pengemudi

## Hardware

- Raspberry Pi 4
- Webcam
- MAX30102
- LCD I2C
- Relay
- Water Pump
- Buzzer
- Power Supply

## Software

- Raspberry Pi OS
- Python 3
- OpenCV
- dlib
- imutils
- NumPy
- SciPy

## Instalasi

Clone repository:

```bash
git clone https://github.com/USERNAME/microsleep-detection.git
cd microsleep-detection
