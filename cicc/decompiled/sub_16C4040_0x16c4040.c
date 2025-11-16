// Function: sub_16C4040
// Address: 0x16c4040
//
__m128i *__fastcall sub_16C4040(__m128i *a1, __int64 a2, __int64 a3, int a4)
{
  const __m128i *v4; // rax
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  _QWORD v9[5]; // [rsp+0h] [rbp-40h] BYREF
  int v10; // [rsp+28h] [rbp-18h]

  v9[2] = 0;
  v9[3] = 0;
  v9[0] = a2;
  v9[1] = a3;
  v9[4] = a3;
  v10 = a4;
  v4 = (const __m128i *)sub_16C3F10((__int64)v9);
  v5 = _mm_loadu_si128(v4);
  v6 = _mm_loadu_si128(v4 + 1);
  v7 = _mm_loadu_si128(v4 + 2);
  *a1 = v5;
  a1[1] = v6;
  a1[2] = v7;
  return a1;
}
