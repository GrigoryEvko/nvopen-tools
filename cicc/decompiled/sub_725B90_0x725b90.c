// Function: sub_725B90
// Address: 0x725b90
//
__int64 __fastcall sub_725B90(__m128i *a1)
{
  __m128i si128; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __m128i v4; // xmm4
  __m128i v5; // xmm5
  __m128i v6; // xmm6
  __int64 v7; // rax
  __m128i v8; // xmm7

  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v2 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  *a1 = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  a1[1] = si128;
  v7 = unk_4D03FA0;
  a1[2] = v2;
  a1[3] = v3;
  a1[4] = v4;
  a1[5] = v5;
  a1[6] = v6;
  if ( v7 )
    a1[3].m128i_i64[1] = *(_QWORD *)(v7 + 8);
  a1[7].m128i_i64[0] = 0;
  a1[8].m128i_i16[4] = 0;
  a1[7].m128i_i64[1] = 0;
  v8 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  a1[8].m128i_i64[0] = 0;
  a1[11].m128i_i64[1] = 0;
  a1[9].m128i_i64[1] &= 0xFE0000000000uLL;
  a1[13].m128i_i64[0] = 0;
  a1[13].m128i_i64[1] = 0;
  a1[14].m128i_i64[0] = 0;
  a1[14].m128i_i64[1] = 0;
  a1[15].m128i_i64[0] = 0;
  a1[15].m128i_i64[1] = 0;
  a1[8].m128i_i32[3] = 0;
  a1[9].m128i_i64[0] = 0;
  a1[10].m128i_i64[0] = 0;
  a1[10].m128i_i64[1] = 2048;
  a1[11].m128i_i16[0] = 0;
  a1[16].m128i_i64[0] = 0;
  a1[16].m128i_i64[1] = 0;
  a1[12] = v8;
  return 0xFE0000000000LL;
}
