// Function: sub_725D60
// Address: 0x725d60
//
_QWORD *sub_725D60()
{
  _QWORD *result; // rax
  __m128i si128; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __m128i v4; // xmm4
  __m128i v5; // xmm5
  __m128i v6; // xmm6
  __int64 v7; // rdx
  __m128i v8; // xmm7

  result = sub_7247C0(208);
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
  result[14] = 0;
  *((_WORD *)result + 68) = 0;
  *(_QWORD *)((char *)result + 140) &= 0xFFF0000000000000LL;
  result[15] = 0;
  result[16] = 0;
  v8 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  result[19] = 0;
  result[20] = 0;
  result[21] = 0;
  result[22] = 0;
  result[23] = 0;
  *((__m128i *)result + 12) = v8;
  return result;
}
