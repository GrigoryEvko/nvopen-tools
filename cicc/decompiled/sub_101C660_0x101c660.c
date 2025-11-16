// Function: sub_101C660
// Address: 0x101c660
//
unsigned __int8 *__fastcall sub_101C660(
        int a1,
        unsigned __int8 *a2,
        __int64 *a3,
        unsigned int a4,
        __m128i *a5,
        unsigned int a6)
{
  __int64 *v7; // r14
  __m128i v8; // xmm0
  __int64 *v9; // r15
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  unsigned __int8 *v13; // rax
  __m128i v14; // xmm4
  __int64 *v15; // r13
  __m128i v16; // xmm5
  __m128i v17; // xmm6
  __m128i v18; // xmm7
  __int64 *v19; // rdx
  unsigned __int8 *result; // rax
  __m128i v24; // [rsp+20h] [rbp-80h] BYREF
  __m128i v25; // [rsp+30h] [rbp-70h]
  __m128i v26; // [rsp+40h] [rbp-60h]
  __m128i v27; // [rsp+50h] [rbp-50h]
  __int64 v28; // [rsp+60h] [rbp-40h]

  v7 = (__int64 *)*((_QWORD *)a2 - 8);
  v8 = _mm_loadu_si128(a5);
  v9 = (__int64 *)*((_QWORD *)a2 - 4);
  v10 = _mm_loadu_si128(a5 + 1);
  v11 = _mm_loadu_si128(a5 + 2);
  v28 = a5[4].m128i_i64[0];
  v12 = _mm_loadu_si128(a5 + 3);
  BYTE1(v28) = 0;
  v24 = v8;
  v25 = v10;
  v26 = v11;
  v27 = v12;
  v13 = sub_101AFF0(a1, v7, a3, &v24, a6);
  if ( !v13 )
    return 0;
  v14 = _mm_loadu_si128(a5);
  v15 = (__int64 *)v13;
  v16 = _mm_loadu_si128(a5 + 1);
  v17 = _mm_loadu_si128(a5 + 2);
  v18 = _mm_loadu_si128(a5 + 3);
  v28 = a5[4].m128i_i64[0];
  v24 = v14;
  BYTE1(v28) = 0;
  v25 = v16;
  v26 = v17;
  v27 = v18;
  v19 = (__int64 *)sub_101AFF0(a1, v9, a3, &v24, a6);
  if ( !v19 )
    return 0;
  if ( v7 != v15 || (result = a2, v9 != v19) )
  {
    if ( a4 > 0x1E )
      return sub_101AFF0(a4, v15, v19, a5, a6);
    if ( ((1LL << a4) & 0x70066000) == 0 )
      return sub_101AFF0(a4, v15, v19, a5, a6);
    if ( v9 != v15 )
      return sub_101AFF0(a4, v15, v19, a5, a6);
    result = a2;
    if ( v7 != v19 )
      return sub_101AFF0(a4, v15, v19, a5, a6);
  }
  return result;
}
