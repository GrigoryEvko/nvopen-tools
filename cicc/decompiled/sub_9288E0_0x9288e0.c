// Function: sub_9288E0
// Address: 0x9288e0
//
__int64 __fastcall sub_9288E0(__int64 *a1, __int64 a2)
{
  __m128i v3; // xmm1
  __m128i v4; // xmm0
  __m128i v5; // xmm2
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD v11[4]; // [rsp+0h] [rbp-B0h] BYREF
  __m128i v12; // [rsp+20h] [rbp-90h] BYREF
  __m128i v13; // [rsp+30h] [rbp-80h] BYREF
  __m128i v14; // [rsp+40h] [rbp-70h] BYREF
  __int64 v15; // [rsp+50h] [rbp-60h]
  __m128i v16; // [rsp+60h] [rbp-50h]
  __m128i v17; // [rsp+70h] [rbp-40h]
  __m128i v18; // [rsp+80h] [rbp-30h]
  __int64 v19; // [rsp+90h] [rbp-20h]

  sub_926800((__int64)&v12, *a1, a2);
  v3 = _mm_loadu_si128(&v13);
  v4 = _mm_loadu_si128(&v12);
  v5 = _mm_loadu_si128(&v14);
  v19 = v15;
  v6 = *a1;
  v16 = v4;
  v17 = v3;
  v18 = v5;
  sub_9286A0(
    (__int64)v11,
    v6,
    (_DWORD *)(a2 + 36),
    v7,
    v8,
    v9,
    v4.m128i_i64[0],
    v4.m128i_u64[1],
    v3.m128i_u64[0],
    v3.m128i_i64[1],
    v5.m128i_i64[0],
    v5.m128i_i64[1],
    v15);
  return v11[0];
}
