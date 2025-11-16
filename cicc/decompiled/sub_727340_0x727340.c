// Function: sub_727340
// Address: 0x727340
//
_QWORD *sub_727340()
{
  _QWORD *result; // rax
  __m128i si128; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __m128i v4; // xmm4
  __m128i v5; // xmm5
  __m128i v6; // xmm6
  __int64 v7; // rdx
  __int64 v8; // rdx
  __m128i v9; // xmm7

  result = sub_7247C0(224);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v2 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  *(__m128i *)result = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  *((__m128i *)result + 1) = si128;
  v7 = unk_4D03FA0;
  *((__m128i *)result + 2) = v2;
  *((__m128i *)result + 3) = v3;
  *((__m128i *)result + 4) = v4;
  *((__m128i *)result + 5) = v5;
  *((__m128i *)result + 6) = v6;
  if ( v7 )
    result[7] = *(_QWORD *)(v7 + 8);
  *((_WORD *)result + 60) &= 0xF000u;
  result[14] = 0;
  v8 = *(_QWORD *)&dword_4F077C8;
  *((_BYTE *)result + 184) = 0;
  result[21] = 0;
  *(_QWORD *)((char *)result + 140) = v8;
  result[22] = 0;
  v9 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  result[24] = 0;
  result[25] = result;
  result[26] = 0;
  result[27] = 0;
  *((_DWORD *)result + 31) = 0;
  result[16] = 0;
  *((_BYTE *)result + 136) = 0;
  *(__m128i *)((char *)result + 148) = v9;
  return result;
}
