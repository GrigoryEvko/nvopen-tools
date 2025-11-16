// Function: sub_EA1150
// Address: 0xea1150
//
__m128i *__fastcall sub_EA1150(__m128i *a1, _QWORD *a2, _DWORD *a3, size_t a4)
{
  const __m128i *v5; // rax
  __int64 v6; // rdx
  __m128i v7; // xmm3
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm4
  __int64 v13; // rax

  v5 = (const __m128i *)sub_EA0AA0((__int64)a2, a3, a4);
  v6 = a2[28];
  v7 = _mm_loadu_si128(v5 + 3);
  v8 = a2[26];
  v9 = _mm_loadu_si128(v5);
  v10 = _mm_loadu_si128(v5 + 1);
  a1[5].m128i_i64[1] = a2[27];
  v11 = _mm_loadu_si128(v5 + 2);
  v12 = _mm_loadu_si128(v5 + 4);
  a1[3] = v7;
  v13 = a1[3].m128i_i64[1];
  a1[5].m128i_i64[0] = v8;
  a1[6].m128i_i64[0] = v6;
  a1[6].m128i_i64[1] = v13;
  *a1 = v9;
  a1[1] = v10;
  a1[2] = v11;
  a1[4] = v12;
  return a1;
}
