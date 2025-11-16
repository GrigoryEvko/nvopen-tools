// Function: sub_C8B260
// Address: 0xc8b260
//
__m128i *__fastcall sub_C8B260(__m128i *a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v4; // [rsp+0h] [rbp-30h] BYREF
  __int32 v5; // [rsp+10h] [rbp-20h]

  sub_C8B230(a2, (__int64)&v4);
  v2 = _mm_loadu_si128(&v4);
  a1[1].m128i_i32[0] = v5;
  *a1 = v2;
  return a1;
}
