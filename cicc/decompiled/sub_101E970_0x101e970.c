// Function: sub_101E970
// Address: 0x101e970
//
unsigned __int8 *__fastcall sub_101E970(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        const __m128i *a4,
        char a5,
        __int64 a6)
{
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __int64 v10; // rax
  _QWORD v12[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v13[4]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+50h] [rbp-10h]

  if ( a5 )
  {
    v13[0].m128i_i64[0] = a2;
    v13[0].m128i_i64[1] = a3;
    return sub_100F630(a1, (__int64)v13, 1, a4, 1u, a6, 3);
  }
  else
  {
    v6 = _mm_loadu_si128(a4);
    v7 = _mm_loadu_si128(a4 + 1);
    v8 = _mm_loadu_si128(a4 + 2);
    v9 = _mm_loadu_si128(a4 + 3);
    v12[0] = a2;
    v10 = a4[4].m128i_i64[0];
    v12[1] = a3;
    v14 = v10;
    v13[0] = v6;
    BYTE1(v14) = 0;
    v13[1] = v7;
    v13[2] = v8;
    v13[3] = v9;
    return sub_100F630(a1, (__int64)v12, 1, v13, 0, a6, 3);
  }
}
