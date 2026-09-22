# dr14meter: compute the DR14 value of the given audio files
# Copyright (C) 2024  pe7ro
#
# dr14_t.meter: compute the DR14 value of the given audiofiles
# Copyright (C) 2011  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pathlib
import time
import sys
import subprocess
import wave
import numpy
import shutil
import functools


from dr14meter.out_messages import print_msg, dr14_log_info
from dr14meter.dr14_global import get_ffmpeg_cmd


@functools.lru_cache(maxsize=32)
def _probe_channels(file_name):
    """Number of audio channels in file_name, via ffprobe. Cached because a
    cue-sheet split probes the same underlying file once per track."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=channels', '-of', 'csv=p=0', str(file_name)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        )
        return int(out.stdout.decode().strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 2

#from _ftdi1 import NONE

# ffmpeg -i example.m4a -f wav pipe:1 > test.wav

# def check_command(cmd):
#     try:
#         subprocess.run([cmd, '--version'], check=True, stdout=subprocess.DEVNULL)
#         return True
#     except FileNotFoundError:
#         return False


class AudioFileReader:

    def __init__(self):
        self.__ffmpeg_cmd = get_ffmpeg_cmd()

        if sys.platform.startswith('win'):
            c = self.get_cmd()
            if shutil.which(c):
                self.__cmd = self.get_cmd()
            else:
                print_msg(f'Unable to find "{c}" in PATH')
        else:
            self.__cmd = self.get_cmd()

    def get_cmd(self):
        return self.__ffmpeg_cmd

    def get_cmd_options(self, file_name, start=None, end=None):
        opts = ['-y']

        # -ss/-to as *input* options (before -i) seek/trim the source file itself,
        # which is what we need to carve a single track out of a cue-sheet image.
        if start is not None:
            opts += ['-ss', f'{start:.6f}']
        if end is not None:
            opts += ['-to', f'{end:.6f}']

        opts += [
            '-i',
            file_name,
            '-ar', '44100',
            '-acodec', 'pcm_s16le',
            '-f', 's16le',
            'pipe:1',
            '-loglevel', 'quiet',
        ]
        return opts

    def read_audio_file_new(self, file_name, target, start=None, end=None):
        file_name = pathlib.Path(file_name)

        time_a = time.time_ns()

        # headerless raw PCM over a pipe: no temp file ever touches disk, and
        # unlike a WAV container it needs no length header ffmpeg would be
        # unable to backpatch on a non-seekable pipe output
        channels = _probe_channels(file_name)
        full_command = [self.__cmd] + self.get_cmd_options(file_name, start, end)

        proc = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              shell=False, check=True)
        ret_f = self._decode_pcm_s16le(proc.stdout, channels, 44100, target)

        time_a = time.time_ns() - time_a
        dr14_log_info(f"AudioFileReader.read_audio_file_new: Clock: {time_a / 1000_000_000:2.8f}")

        return ret_f

    def _decode_pcm_s16le(self, data, channels, framerate, target):
        sampwidth = 2

        try:
            nframes = len(data) // (channels * sampwidth)
            usable = nframes * channels * sampwidth

            target.channels = channels
            target.Fs = framerate
            target.sample_width = sampwidth

            Y = numpy.frombuffer(data[:usable], dtype='int16').reshape(nframes, channels)
            target.Y = Y / numpy.float32(2 ** 15 + 1)
        except:
            self.__init__()
            print_msg(f"Unexpected error: {sys.exc_info()}")
            print_msg("\n - ERROR ! ")
            return False

        return True

    def read_wav(self, file_name, target, start=None, end=None):
        file_name = pathlib.Path(file_name)

        time_a = time.time_ns()

        try:
            with wave.open(str(file_name), 'rb') as wave_read:

                target.channels = wave_read.getnchannels()
                target.Fs = wave_read.getframerate()
                target.sample_width = wave_read.getsampwidth()

                total_frames = wave_read.getnframes()

                start_frame = 0 if start is None else max(0, int(round(start * target.Fs)))
                end_frame = total_frames if end is None else min(total_frames, int(round(end * target.Fs)))
                start_frame = min(start_frame, total_frames)
                nframes = max(0, end_frame - start_frame)

                if start_frame:
                    wave_read.setpos(start_frame)

                #print_msg( file_name + "!!!!!!!!!!!!: " + str(target.channels) + " " + str(target.sample_width ) + " " + str( target.Fs ) + " " + str( nframes ) )

                X = wave_read.readframes(nframes)
                sample_type = f"int{target.sample_width * 8}"
                target.Y = numpy.frombuffer(X, dtype=sample_type).reshape(nframes, target.channels)

            if sample_type == 'int16':
                convert_16_bit = numpy.float32(2 ** 15 + 1)
                target.Y = target.Y / convert_16_bit
            elif sample_type == 'int32':
                convert_32_bit = numpy.float32(2 ** 31 + 1)
                target.Y = target.Y / convert_32_bit
            else:
                convert_8_bit = numpy.float32(2 ** 8 + 1)
                target.Y = target.Y / convert_8_bit

            #print_msg( "target.Y: " + str(target.Y.dtype) )
        except:
            self.__init__()
            print_msg(f"Unexpected error: {sys.exc_info()}")
            print_msg("\n - ERROR ! ")
            return False

        time_a = time.time_ns() - time_a
        dr14_log_info(f"AudioFileReader.read_wav: Clock: {time_a / 1000_000_000:2.8f}s")

        return True


class WavFileReader(AudioFileReader):

    def read_audio_file_new(self, file_name, target, start=None, end=None):
        return self.read_wav(file_name, target, start, end)

    def get_cmd(self):
        return ""

    def get_cmd_options(self, file_name, start=None, end=None):
        return ""

