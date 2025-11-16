// Function: sub_C832B0
// Address: 0xc832b0
//
__m128i *__fastcall sub_C832B0(__m128i *a1, __int64 a2)
{
  char v2; // dl
  __int32 v3; // eax
  __int64 v4; // rdx
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __int64 v9[4]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v10; // [rsp+20h] [rbp-70h]
  __m128i v11; // [rsp+30h] [rbp-60h] BYREF
  __m128i v12; // [rsp+40h] [rbp-50h] BYREF
  __m128i v13; // [rsp+50h] [rbp-40h] BYREF
  __int64 v14; // [rsp+70h] [rbp-20h]

  v2 = *(_BYTE *)(a2 + 36);
  v9[0] = a2;
  memset(&v13, 0, 32);
  v14 = 0;
  v13.m128i_i32[3] = 0xFFFF;
  v10 = 260;
  v11 = 0;
  v12 = 0;
  v3 = sub_C826E0((__int64)v9, (__int64)&v11, v2);
  if ( v3 )
  {
    a1->m128i_i32[0] = v3;
    a1[3].m128i_i8[0] |= 1u;
    a1->m128i_i64[1] = v4;
  }
  else
  {
    v6 = _mm_loadu_si128(&v11);
    v7 = _mm_loadu_si128(&v12);
    v8 = _mm_loadu_si128(&v13);
    a1[3].m128i_i8[0] &= ~1u;
    *a1 = v6;
    a1[1] = v7;
    a1[2] = v8;
  }
  return a1;
}
