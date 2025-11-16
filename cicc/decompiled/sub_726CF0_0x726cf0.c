// Function: sub_726CF0
// Address: 0x726cf0
//
__int64 __fastcall sub_726CF0(__m128i *a1, char a2)
{
  __m128i si128; // xmm1
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __int64 v8; // rax
  int v9; // eax
  __int64 result; // rax

  si128 = _mm_load_si128((const __m128i *)&xmmword_4F079B0);
  v3 = _mm_load_si128((const __m128i *)&xmmword_4F079C0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_4F079D0);
  v5 = _mm_load_si128((const __m128i *)&xmmword_4F079E0);
  *a1 = _mm_load_si128((const __m128i *)&xmmword_4F079A0);
  v6 = _mm_load_si128((const __m128i *)&xmmword_4F079F0);
  v7 = _mm_load_si128((const __m128i *)&xmmword_4F07A00);
  a1[1] = si128;
  v8 = unk_4D03FA0;
  a1[2] = v3;
  a1[3] = v4;
  a1[4] = v5;
  a1[5] = v6;
  a1[6] = v7;
  if ( v8 )
    a1[3].m128i_i64[1] = *(_QWORD *)(v8 + 8);
  v9 = a1[7].m128i_u8[12];
  a1[7].m128i_i64[0] = 0;
  a1[7].m128i_i32[2] = 0;
  a1[8].m128i_i64[0] = 0;
  result = a2 & 1 | v9 & 0xFFFFFFC0;
  a1[7].m128i_i8[12] = result;
  return result;
}
