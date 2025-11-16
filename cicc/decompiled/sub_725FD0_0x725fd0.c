// Function: sub_725FD0
// Address: 0x725fd0
//
__m128i *sub_725FD0()
{
  __m128i *v0; // rax
  __m128i si128; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __m128i *v4; // r12
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  int v8; // edx
  bool v9; // zf
  unsigned __int64 v10; // rdx

  v0 = (__m128i *)sub_7247C0(368);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v2 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v4 = v0;
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  *v0 = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v7 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  v0[1] = si128;
  v0[2] = v2;
  v0[3] = v3;
  v0[4] = v5;
  v0[5] = v6;
  v0[6] = v7;
  if ( unk_4D03FA0 )
    v0[3].m128i_i64[1] = *(_QWORD *)(unk_4D03FA0 + 8LL);
  v0[7].m128i_i64[0] = 0;
  v0[10].m128i_i16[6] = 0;
  v0[7].m128i_i64[1] = 0;
  v0[8].m128i_i64[0] = 0;
  v0[8].m128i_i64[1] = 0;
  v0[9].m128i_i32[0] = 0;
  v0[9].m128i_i64[1] = 0;
  v0[10].m128i_i64[0] = 0;
  v0[10].m128i_i32[2] = 0;
  sub_725ED0((__int64)v0, 0);
  v4[12].m128i_i64[1] = 0;
  v4[13].m128i_i32[1] = 0;
  v4[13].m128i_i64[1] = 0;
  v8 = unk_4D04630;
  v4[14].m128i_i16[0] = -1;
  v4[14].m128i_i64[1] = 0;
  v9 = v8 == 0;
  v4[15].m128i_i64[0] = 0;
  v10 = v4[12].m128i_i64[0] & 0xC00001C000000000LL;
  v4[13].m128i_i8[0] &= 0xF8u;
  v4[15].m128i_i64[1] = 0;
  v4[12].m128i_i64[0] = v10 | ((unsigned __int64)!v9 << 52);
  v4[12].m128i_i16[2] = 0;
  v4[16].m128i_i64[0] = 0;
  v4[16].m128i_i64[1] = 0;
  v4[17].m128i_i64[0] = 0;
  v4[17].m128i_i64[1] = 0;
  v4[18].m128i_i64[0] = 0;
  v4[18].m128i_i64[1] = 0;
  v4[19].m128i_i64[0] = 0;
  v4[19].m128i_i64[1] = 0;
  v4[20].m128i_i64[0] = 0;
  v4[20].m128i_i64[1] = 0;
  v4[21].m128i_i64[0] = 0;
  v4[21].m128i_i64[1] = 0;
  v4[22].m128i_i16[0] = 0;
  v4[22].m128i_i64[1] = 0;
  return v4;
}
