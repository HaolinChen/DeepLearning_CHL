#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2020 Bitcraze AB
#
#  AI-deck demo
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License along with
#  this program; if not, write to the Free Software Foundation, Inc., 51
#  Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#  Demo for showing streamed JPEG images from the AI-deck example.
#
#  By default this demo connects to the IP of the AI-deck example when in
#  Access point mode.
#
#  The demo works by opening a socket to the AI-deck, downloads a stream of
#  JPEG images and looks for start/end-of-frame for the streamed JPEG images.
#  Once an image has been fully downloaded it's rendered in the UI.
#
#  Note that the demo firmware is continously streaming JPEG files so a single
#  JPEG image is taken from the stream using the JPEG start-of-frame (0xFF 0xD8)
#  and the end-of-frame (0xFF 0xD9).

import argparse
import threading
import time
import socket,os
import cv2
import numpy as np

deck_ip = None
deck_port = None

# Args for setting IP/port of AI-deck. Default settings are for when
# AI-deck is in AP mode.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

imgdata = None
data_buffer = bytearray()
start = time.time()

dir = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
os.mkdir(dir)
index = 0
capture_on = False
take_one_photo = False
font = cv2.FONT_HERSHEY_SIMPLEX

while(1):
    # Reveive image data from the AI-deck

    data_buffer.extend(client_socket.recv(8))
    # Look for start-of-frame and end-of-frame
    start_idx = data_buffer.find(b"\xff\xd8")
    end_idx = data_buffer.find(b"\xff\xd9")

    # At startup we might get an end before we get the first start, if
    # that is the case then throw away the data before start
    if end_idx > -1 and end_idx < start_idx:
        data_buffer = data_buffer[start_idx:]

    # We have a start and an end of the image in the buffer now
    if start_idx > -1 and end_idx > -1 and end_idx > start_idx:
        # Pick out the image to render ...
        imgdata = data_buffer[start_idx:end_idx + 2]
        imgdata = np.array(imgdata)
        # .. and remove it from the buffer
        data_buffer = data_buffer[end_idx + 2 :]
        image = cv2.imdecode(imgdata, cv2.IMREAD_GRAYSCALE)
        image = image[image.shape[0]//2 - 128 : image.shape[0]//2 + 128,image.shape[1]//2 - 160 : image.shape[1]//2 + 160]
        fps = 1 / (time.time() - start)
        start = time.time()
        fpsinfo = "{:.1f} fps / {:.1f} kb".format(fps, len(imgdata)/1000)
        
        if capture_on:
            name = dir + '/' + '%05d'%index+'.jpg'
            cv2.imwrite(name,image,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            print('saving...' + name)
            index+=1
        
        image_show = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        if capture_on:
            cv2.putText(image_show, 'REC...' + fpsinfo, (0, 20), font, 0.6, (0,0,255), 1)
        else:
            cv2.putText(image_show, 'r=record,space=take photo,q=quit ', (0, 20), font, 0.5, (0,255,0), 1)
            if take_one_photo:
                name = dir + '/' + '%05d'%index+'.jpg'
                cv2.imwrite(name,image,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                print('saving...' + name)
                index+=1
                image_show[:,:,:] = [255,255,255]
                cv2.imshow('image_show',image_show)
                cv2.waitKey(50)
                take_one_photo = False

        cv2.imshow('image_show',image_show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            capture_on = not capture_on
            if capture_on:
                print('Action!')
            else:
                print('Cut!')
        
        elif k == ord('q'):
            break

        elif k == ord(' ') and capture_on == False:
            take_one_photo = True


