// Function: sub_3715DD0
// Address: 0x3715dd0
//
__int64 __fastcall sub_3715DD0(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__m128i *, __m128i *),
        __int64 a5,
        __int64 a6)
{
  __int64 result; // rax
  __m128i *v8; // r15
  __m128i *v9; // rbx
  __m128i *v10; // r14
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int128 v17; // xmm4
  __int128 v18; // xmm5
  __int64 v19; // [rsp+10h] [rbp-70h]
  __m128i *v20; // [rsp+18h] [rbp-68h]

  result = (__int64)a2->m128i_i64 - a1;
  v20 = a2;
  v19 = a3;
  if ( (__int64)a2->m128i_i64 - a1 <= 640 )
    return result;
  if ( !a3 )
  {
    v10 = a2;
    goto LABEL_13;
  }
  while ( 2 )
  {
    v8 = v20;
    v9 = (__m128i *)(a1 + 40);
    --v19;
    sub_3712030(
      (__m128i *)a1,
      (__m128i *)(a1 + 40),
      (__m128i *)(a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (result >> 3)) >> 1)),
      (__m128i *)((char *)v20 - 40),
      a4);
    while ( 1 )
    {
      v10 = v9;
      if ( a4(v9, (__m128i *)a1) )
        goto LABEL_4;
      do
        v8 = (__m128i *)((char *)v8 - 40);
      while ( a4((__m128i *)a1, v8) );
      if ( v9 >= v8 )
        break;
      v11 = _mm_loadu_si128(v9);
      v12 = _mm_loadu_si128(v9 + 1);
      v13 = v9[2].m128i_i64[0];
      *v9 = _mm_loadu_si128(v8);
      v9[1] = _mm_loadu_si128(v8 + 1);
      v9[2].m128i_i8[0] = v8[2].m128i_i8[0];
      v8[2].m128i_i8[0] = v13;
      *v8 = v11;
      v8[1] = v12;
LABEL_4:
      v9 = (__m128i *)((char *)v9 + 40);
    }
    sub_3715DD0(v9, v20, v19, a4);
    result = (__int64)v9->m128i_i64 - a1;
    if ( (__int64)v9->m128i_i64 - a1 > 640 )
    {
      if ( v19 )
      {
        v20 = v9;
        continue;
      }
LABEL_13:
      sub_3715C60(
        (const __m128i *)a1,
        v10,
        (unsigned __int64)v10,
        (unsigned __int8 (__fastcall *)(__m128i *, unsigned __int64 *))a4,
        a5,
        a6);
      do
      {
        v10 = (__m128i *)((char *)v10 - 40);
        v16 = v10[2].m128i_i64[0];
        v17 = (__int128)_mm_loadu_si128(v10);
        *v10 = _mm_loadu_si128((const __m128i *)a1);
        v18 = (__int128)_mm_loadu_si128(v10 + 1);
        v10[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
        v10[2].m128i_i8[0] = *(_BYTE *)(a1 + 32);
        result = sub_3715A20(
                   a1,
                   0,
                   0xCCCCCCCCCCCCCCCDLL * (((__int64)v10->m128i_i64 - a1) >> 3),
                   (unsigned __int8 (__fastcall *)(__m128i *, unsigned __int64 *))a4,
                   v14,
                   v15,
                   v17,
                   v18,
                   v16);
      }
      while ( (__int64)v10->m128i_i64 - a1 > 40 );
    }
    return result;
  }
}
