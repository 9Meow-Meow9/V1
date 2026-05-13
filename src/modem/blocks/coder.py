from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class HammongEncoder(Block[np.ndarray, np.ndarray]):
    
    def hamming_encode(self, bits4:np.ndarray) -> np.ndarray:
        d1,d2,d3,d4 = bits4

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4

        return  np.array ([p1, p2, d1, p3, d2, d3, d4])

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:

        encoded_bits = []

        for i in range(0, len(bits), 4):
            block = bits[i:i + 4]
            if len(block) < 4:
                block = np.pad(block, (0, 4 - len(block)), constant_values=0)
            encoded_bits.append(self.hamming_encode(block))

        result = np.concatenate(encoded_bits)

        return result
@dataclass
class StringToBits(Block[str, np.ndarray]):

    encoding: str = 'utf-8'


    def process(self, text: str, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:

        data_bytes = text.encode(self.encoding)

        bits = []
        for i in data_bytes:
            bit = format(i,'08b')
            bits.append(bit)  #['01100001', '01100001', '01100001']
        bits_string = ''.join(bits)   #output 011000010110000101100001 
        bits_array = [int(i) for i in bits_string]  


        if ctx is not None:
            ctx['output_bits'] = bits_array
        return np.array(bits_array)


@dataclass
class PacketBuileder(Block[np.ndarray, np.ndarray]):

    sfd: str = "10101011"
    
    preamble: str = '1010101010101010'
    add_crc: bool = True 

    def calculate_checksum(self, data_bits: str) -> str:
        checksum = 0
        for i in range(0,len(data_bits),8):
            byte = int(data_bits[i:i+8], 2)
            checksum ^= byte
        return format(checksum, '08b')




    def process(self,  bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        
        data_bits = ''.join(str(int(i)) for i in bits)
        checksum = self.calculate_checksum(data_bits)

        bits_string = self.preamble + self.sfd + data_bits + checksum
        
        result_arr = np.array([int(i) for i in bits_string])

        if ctx is not None:
            ctx['preamble'] = np.array([int(i) for i in self.preamble])
            ctx['sfd'] = np.array([int(i) for i in self.sfd])
            ctx['result_arr'] = result_arr
            ctx['sum'] = np.array([int(b) for b in checksum])

        return result_arr
