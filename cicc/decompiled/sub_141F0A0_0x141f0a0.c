// Function: sub_141F0A0
// Address: 0x141f0a0
//
__m128i *__fastcall sub_141F0A0(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __m128i v3; // xmm0
  __int64 v4; // rax
  __m128i v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h]

  v6 = 0u;
  v7 = 0;
  sub_14A8180(a2, &v6, 0);
  v2 = *(_QWORD *)(a2 - 24);
  v3 = _mm_loadu_si128(&v6);
  a1->m128i_i64[1] = -1;
  a1->m128i_i64[0] = v2;
  v4 = v7;
  a1[1] = v3;
  a1[2].m128i_i64[0] = v4;
  return a1;
}
