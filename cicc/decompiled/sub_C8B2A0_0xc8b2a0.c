// Function: sub_C8B2A0
// Address: 0xc8b2a0
//
__m128i *__fastcall sub_C8B2A0(__m128i *a1, __m128i *a2)
{
  __int32 v2; // eax
  __m128i v3; // xmm5
  __m128i v4; // xmm6
  __m128i v5; // xmm7
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __int32 v8; // eax
  __m128i v10; // [rsp+0h] [rbp-70h] BYREF
  __m128i v11; // [rsp+10h] [rbp-60h] BYREF
  __m128i v12; // [rsp+20h] [rbp-50h] BYREF
  __m128i v13; // [rsp+30h] [rbp-40h] BYREF
  __m128i v14; // [rsp+40h] [rbp-30h] BYREF
  __int64 v15; // [rsp+50h] [rbp-20h]
  __int32 v16; // [rsp+58h] [rbp-18h]

  v15 = a2[5].m128i_i64[0];
  v2 = a2[5].m128i_i32[2];
  v10 = _mm_loadu_si128(a2);
  v11 = _mm_loadu_si128(a2 + 1);
  v16 = v2;
  v12 = _mm_loadu_si128(a2 + 2);
  v13 = _mm_loadu_si128(a2 + 3);
  v14 = _mm_loadu_si128(a2 + 4);
  sub_C8B260(a1, (__int64)a2);
  v3 = _mm_loadu_si128(&v10);
  v4 = _mm_loadu_si128(&v11);
  v5 = _mm_loadu_si128(&v12);
  v6 = _mm_loadu_si128(&v13);
  v7 = _mm_loadu_si128(&v14);
  a2[5].m128i_i64[0] = v15;
  v8 = v16;
  *a2 = v3;
  a2[1] = v4;
  a2[5].m128i_i32[2] = v8;
  a2[2] = v5;
  a2[3] = v6;
  a2[4] = v7;
  return a1;
}
