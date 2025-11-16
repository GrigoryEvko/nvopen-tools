// Function: sub_726270
// Address: 0x726270
//
__m128i *sub_726270()
{
  __m128i *result; // rax
  __m128i si128; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __m128i v4; // xmm4
  __m128i v5; // xmm5
  __m128i v6; // xmm6
  __int64 v7; // rdx
  __int8 v8; // dl

  result = (__m128i *)sub_7246D0(168);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v2 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  *result = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  result[1] = si128;
  v7 = unk_4D03FA0;
  result[2] = v2;
  result[3] = v3;
  result[4] = v4;
  result[5] = v5;
  result[6] = v6;
  if ( v7 )
    result[3].m128i_i64[1] = *(_QWORD *)(v7 + 8);
  v8 = result[8].m128i_i8[0];
  result[7].m128i_i64[0] = 0;
  result[7].m128i_i64[1] = 0;
  result[8].m128i_i64[1] = 0;
  result[9].m128i_i64[0] = 0;
  result[8].m128i_i8[0] = v8 & 0xC0 | 1;
  result[9].m128i_i64[1] = 0;
  result[10].m128i_i64[0] = 0;
  return result;
}
