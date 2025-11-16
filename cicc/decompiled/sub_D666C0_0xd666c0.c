// Function: sub_D666C0
// Address: 0xd666c0
//
__m128i *__fastcall sub_D666C0(__m128i *a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __m128i v5; // [rsp+0h] [rbp-30h] BYREF
  __m128i v6[2]; // [rsp+10h] [rbp-20h] BYREF

  sub_B91FC0(v5.m128i_i64, a2);
  v2 = _mm_loadu_si128(&v5);
  v3 = _mm_loadu_si128(v6);
  a1->m128i_i64[0] = *(_QWORD *)(a2 - 32);
  a1->m128i_i64[1] = 0xBFFFFFFFFFFFFFFELL;
  a1[1] = v2;
  a1[2] = v3;
  return a1;
}
