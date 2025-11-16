// Function: sub_1B231B0
// Address: 0x1b231b0
//
__int64 __fastcall sub_1B231B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r14
  __int64 i; // r14
  __int64 v11; // r12
  __m128i *v12; // r15
  int v13; // eax
  bool v14; // sf
  __m128i *v15; // rax
  __int64 v16; // r13
  __int64 v17; // r15
  const __m128i *v18; // r14
  int v19; // r8d
  __m128i *v20; // rax
  const __m128i *v22; // rax
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+28h] [rbp-48h]

  v9 = a1;
  v24 = a3 & 1;
  v25 = (a3 - 1) / 2;
  if ( a2 >= v25 )
  {
    v12 = (__m128i *)(a1 + 24 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_16;
    v11 = a2;
    goto LABEL_19;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    v12 = (__m128i *)(a1 + 48 * (i + 1));
    v13 = sub_16AEA10(v12->m128i_i64[0] + 24, *(_QWORD *)(a1 + 24 * (v11 - 1) + 8) + 24LL);
    v14 = v13 < 0;
    if ( v13 < 0 )
      v12 = (__m128i *)(a1 + 24 * (v11 - 1));
    v15 = (__m128i *)(a1 + 24 * i);
    if ( v14 )
      --v11;
    *v15 = _mm_loadu_si128(v12);
    v15[1].m128i_i64[0] = v12[1].m128i_i64[0];
    if ( v11 >= v25 )
      break;
  }
  v9 = a1;
  if ( !v24 )
  {
LABEL_19:
    if ( (a3 - 2) / 2 == v11 )
    {
      v11 = 2 * v11 + 1;
      v22 = (const __m128i *)(v9 + 24 * v11);
      *v12 = _mm_loadu_si128(v22);
      v12[1].m128i_i64[0] = v22[1].m128i_i64[0];
      v12 = (__m128i *)v22;
    }
  }
  v16 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    v17 = v9;
    while ( 1 )
    {
      v18 = (const __m128i *)(v17 + 24 * v16);
      v19 = sub_16AEA10(v18->m128i_i64[0] + 24, a8 + 24);
      v20 = (__m128i *)(v17 + 24 * v11);
      if ( v19 >= 0 )
      {
        v12 = (__m128i *)(v17 + 24 * v11);
        goto LABEL_16;
      }
      v11 = v16;
      *v20 = _mm_loadu_si128(v18);
      v20[1].m128i_i64[0] = v18[1].m128i_i64[0];
      if ( a2 >= v16 )
        break;
      v16 = (v16 - 1) / 2;
    }
    v12 = (__m128i *)(v17 + 24 * v16);
  }
LABEL_16:
  v12->m128i_i64[0] = a7;
  v12->m128i_i64[1] = a8;
  v12[1].m128i_i64[0] = a9;
  return a9;
}
