import UnicornPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.cross_decomposition import CCA
from multiprocessing import Process, Queue
import time
import pygame
import pygame_gui
import serial
import os

# Design Parameters
MIN_FREQ = 5
MAX_FREQ = 50
DEFAULT_FREQ = 11
TARGET_FREQS = [9, 10, 11]
FREQ_ACTIONS = {9: '1', 10: '2', 11: '3'}  

DETECTION_INTERVAL = 0.2  # second
WINDOW_LENGTH = 5  # second

SERIAL_PORT = "COM3"
BAUD_RATE = 9600
# FREQ_MAPPING no used currently
#FREQ_MAPPING = {
#    18: 9,
#    20: 10,
#    22: 11
#}

def initialize_arduino():
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2) 
    return arduino

def create_blinking_window(queue):
    pygame.init()

    Width = 800
    Height = 600

    Window = pygame.display.set_mode((Width, Height))
    Manager = pygame_gui.UIManager((Width, Height))

    FreqInput = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((Width - 100, 0), (100, 50)), manager=Manager)
    FreqLabel = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (100, 50)), text=f"Freq: {DEFAULT_FREQ:0.0f} Hz", manager=Manager)

    Freq = DEFAULT_FREQ
    Duration = 1.0 / (2 * Freq)

    Running = True
    clock = pygame.time.Clock()

    while Running:
        DeltaTime = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Running = False
                queue.put('done') 
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    Running = False
                    queue.put('done')  
                elif event.key == pygame.K_RETURN:
                    try:
   
                        Freq = float(FreqInput.get_text())
        
                        Freq = min(max(Freq, MIN_FREQ), MAX_FREQ)
                        Duration = 1.0 / (2 * Freq)
                  
                        FreqLabel.set_text(f"Freq: {Freq:0.0f} Hz")
                    except ValueError:
                        pass

            Manager.process_events(event)

        Manager.update(DeltaTime)

        # Set the Window to Red
        Window.fill((255, 0, 0))
        Manager.draw_ui(Window)
        pygame.display.flip()
        time.sleep(Duration)

        # Set the Window to Black
        Window.fill((0, 0, 0))
        Manager.draw_ui(Window)
        pygame.display.flip()
        time.sleep(Duration)

    pygame.quit()

def bandpass_filter(data, lowcut, highcut, fs, order=4):

    notch_freq = 50.0
    quality_factor = 30.0
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
    data_notch_filtered = filtfilt(b_notch, a_notch, data)

 
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b_band, a_band = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b_band, a_band, data_notch_filtered)
    return filtered_data

def cca_analysis(data, fs, freq_range, step):
    cca = CCA(n_components=1)
    scores = []
    t = np.arange(data.shape[0]) / fs
    for freq in np.arange(freq_range[0], freq_range[1], step):
        sin_template = np.sin(2 * np.pi * freq * t)
        cos_template = np.cos(2 * np.pi * freq * t)
        template_matrix = np.vstack((sin_template, cos_template)).T
        cca.fit(data, template_matrix)
        U, V = cca.transform(data, template_matrix)
        score = np.corrcoef(U.T, V.T)[0, 1]
        scores.append(score)
    return np.arange(freq_range[0], freq_range[1], step), scores

def send_to_arduino(arduino, freq):
    try:
        action = FREQ_ACTIONS.get(freq, None)
        if action is not None:
            arduino.write(action.encode())
            print(f"Sent {action} to Arduino for frequency {freq} Hz.")
    except Exception as e:
        print(f"Failed to send data to Arduino: {e}")

def plot_scores(freqs, scores):
    plt.figure()
    plt.plot(freqs, scores)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('CCA Score')
    plt.title('CCA Scores for Different Frequencies')

 
    max_idx = np.argmax(scores)
    max_freq = freqs[max_idx]
    max_score = scores[max_idx]

    plt.axvline(x=max_freq, color='red', linestyle='--', label=f'Peak: {max_freq:.2f} Hz')

 
    plt.annotate(f'{max_freq:.2f} Hz', xy=(max_freq, 0), xytext=(max_freq, max_score * 0.5),
                 arrowprops=dict(facecolor='red', arrowstyle="->"),
                 ha='center', va='bottom', color='red')

    plt.scatter(max_freq, max_score, color='red')
    plt.legend()
    plt.show()

def plot_signal(signal, fs, title):
    t = np.arange(len(signal)) / fs
    plt.figure()
    plt.plot(t, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()

#def is_target_or_multiple(rounded_freq):
    
  #  return FREQ_MAPPING.get(rounded_freq, rounded_freq)

def data_acquisition(deviceID, queue):
    arduino = initialize_arduino()
    TestsignaleEnabled = False
    FrameLength = 1
    DataFile = "data.csv"


    if os.path.exists(DataFile):
        os.remove(DataFile)

    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        print(f"\nTrying to connect to '{deviceID}'.")
        device = UnicornPy.Unicorn(deviceID)
        print(f"Connected to '{deviceID}'.\n")

        # Wait 10 seconds to ensure the connection is stable and display a countdown
        print("Waiting for 10 seconds to stabilize the connection...")
        for i in range(10, 0, -1):
            print(f"{i} seconds remaining...")
            time.sleep(1)

        file = open(DataFile, "wb")
        numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()
        print("Acquisition Configuration:")
        print(f"Sampling Rate: {UnicornPy.SamplingRate} Hz")
        print(f"Frame Length: {FrameLength}")
        print(f"Number Of Acquired Channels: {numberOfAcquiredChannels}")
        print()
        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        buffer = np.zeros((UnicornPy.SamplingRate * WINDOW_LENGTH, numberOfAcquiredChannels))  
        buffer_idx = 0
        detection_cooldown = False
        cooldown_start = time.time()

        try:
            device.StartAcquisition(TestsignaleEnabled)
            print("Data acquisition started.")
            consoleUpdateRate = int((UnicornPy.SamplingRate / FrameLength) / 25.0)
            if consoleUpdateRate == 0:
                consoleUpdateRate = 1

            consecutive_counter = 0
            last_rounded_freq = None
            ceshicishu = 0

            while True:
              
                device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))
                np.savetxt(file, data, delimiter=',', fmt='%.3f', newline='\n')

                if detection_cooldown:
                   
                    if (time.time() - cooldown_start) >= 12:
                        detection_cooldown = False
                        buffer_idx = 0  
                        buffer = np.zeros((UnicornPy.SamplingRate * WINDOW_LENGTH, numberOfAcquiredChannels))  
                    continue  

               
                buffer = np.roll(buffer, -FrameLength, axis=0)
                buffer[-FrameLength:, :] = data

          
                if buffer_idx >= UnicornPy.SamplingRate * DETECTION_INTERVAL:
                  
                    filtered_data = bandpass_filter(buffer[:, 6], 2, 40, UnicornPy.SamplingRate)
                    freqs, scores = cca_analysis(filtered_data.reshape(-1, 1), UnicornPy.SamplingRate, (6, 40), 0.1)

                    max_idx = np.argmax(scores)
                    max_freq = freqs[max_idx]

                    rounded_freq = round(max_freq)
                   # corrected_freq = is_target_or_multiple(rounded_freq)
                    corrected_freq = rounded_freq
                    ceshicishu += 1
                    current_accel = data[0, 9]
                    if current_accel >= 0.9:
                        corrected_freq = 0

                    print(f"Detected frequency: {max_freq:.2f} Hz, Rounded frequency: {corrected_freq} Hz")

                    if corrected_freq in TARGET_FREQS:
                        if corrected_freq == last_rounded_freq:
                            consecutive_counter += 1
                        else:
                            consecutive_counter = 1
                            last_rounded_freq = corrected_freq

                        if consecutive_counter >= 5:
                            print(f"Detected frequency 5 times consecutively: {corrected_freq} Hz, detect time:{ceshicishu}")
                            send_to_arduino(arduino, corrected_freq)

                            detection_cooldown = True
                            cooldown_start = time.time()
                            consecutive_counter = 0

                    else:
                        consecutive_counter = 0

                    buffer_idx = 0

                buffer_idx += FrameLength

                if buffer_idx % consoleUpdateRate == 0:
                    print('.', end='', flush=True)

             
                if not queue.empty() and queue.get() == 'done':
                    break

            device.StopAcquisition()
            print("\nData acquisition stopped.")
        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print(f"An unknown error occurred. {e}")
        finally:
            del receiveBuffer
            file.close()
            del device
            print("Disconnected from Unicorn")

     
        if not os.path.exists(DataFile) or os.path.getsize(DataFile) == 0:
            raise Exception("Data file is empty or does not exist. Please check the acquisition process.")

    except UnicornPy.DeviceException as e:
        print(e)
    except Exception as e:
        print(f"An unknown error occurred. {e}")

    input("\n\nPress ENTER key to exit")

if __name__ == "__main__":
    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        deviceList = UnicornPy.GetAvailableDevices(True)
        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available. Please pair with a Unicorn first.")
        print("Available devices:")
        for i, device in enumerate(deviceList):
            print(f"#{i} {device}")
        print()
        deviceID = int(input("Select device by ID #"))
        if deviceID < 0 or deviceID >= len(deviceList):
            raise IndexError('The selected device ID is not valid.')
        selected_device = deviceList[deviceID]
    except Exception as e:
        print(f"An error occurred: {e}")
        input("\n\nPress ENTER key to exit")
        exit()

    
    queue = Queue()
    queue.put(selected_device)

 
    blink_process = Process(target=create_blinking_window, args=(queue,))
    blink_process.start()


    data_acquisition_process = Process(target=data_acquisition, args=(queue.get(), queue))
    data_acquisition_process.start()

  
    data_acquisition_process.join()
    blink_process.join()
