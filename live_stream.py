#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from libs import AudioLiveStream, AudioVisualization

def main(args):

    audio = AudioLiveStream(
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk=args.chunk,
        input_device=args.input_device,
        output_device=args.output_device)
    audio.start()

    if args.visualize:
        vis = AudioVisualization(audio)
        vis.show()
    else:
        try:
            while audio.stream.is_active():
                pass
        except KeyboardInterrupt:
            audio.stop()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--channels', help='channels', type=int, default=1)
    parser.add_argument('-r', '--sample_rate', help='sample rate', type=int, default=44100)
    parser.add_argument('-i', '--input_device', help='input device #', type=int)
    parser.add_argument('-o', '--output_device', help='output device #', type=int)
    parser.add_argument('--chunk', help='chunk size', type=int, default=1024*4)
    parser.add_argument('-v', '--visualize', action='store_true')


    args = parser.parse_args()
    print(args)

    main(args)
