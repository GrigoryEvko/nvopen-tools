// Function: sub_2D22C30
// Address: 0x2d22c30
//
__m128i *__fastcall sub_2D22C30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7,
        const __m128i a8,
        __int64 a9)
{
  __int64 v9; // r8
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // r9
  __int8 v14; // r10
  __int64 i; // rdx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r13
  __m128i *result; // rax
  __m128i *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rcx
  unsigned __int64 v23; // r11
  const __m128i *v24; // rdx
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rcx
  __m128i v31; // [rsp+0h] [rbp-50h] BYREF
  __m128i v32; // [rsp+10h] [rbp-40h] BYREF
  __int64 v33; // [rsp+20h] [rbp-30h]

  v9 = a2;
  v11 = (a3 - 1) / 2;
  v12 = a3 & 1;
  v13 = a7.m128i_i64[1];
  v14 = a8.m128i_i8[8];
  if ( a2 >= v11 )
  {
    result = (__m128i *)(a1 + 40 * a2);
    if ( (a3 & 1) != 0 )
    {
      v31 = _mm_loadu_si128(&a7);
      v33 = a9;
      v32 = _mm_loadu_si128(&a8);
      goto LABEL_22;
    }
LABEL_24:
    if ( (a3 - 2) / 2 == a2 )
    {
      v27 = a2 + 1;
      v28 = 2 * v27;
      v29 = a1 + 80 * v27;
      *result = _mm_loadu_si128((const __m128i *)(v29 - 40));
      result[1] = _mm_loadu_si128((const __m128i *)(v29 - 24));
      v30 = *(_QWORD *)(v29 - 8);
      a2 = v28 - 1;
      result[2].m128i_i64[0] = v30;
      result = (__m128i *)(a1 + 40 * (v28 - 1));
    }
    goto LABEL_13;
  }
  for ( i = a2; ; i = a2 )
  {
    a2 = 2 * (i + 1);
    v20 = 80 * (i + 1);
    v21 = a1 + v20 - 40;
    result = (__m128i *)(a1 + v20);
    if ( *(_BYTE *)(v21 + 24) )
    {
      v16 = *(_QWORD *)(v21 + 8);
      if ( result[1].m128i_i8[8] )
        goto LABEL_4;
    }
    else
    {
      v16 = qword_4F81350[0];
      if ( result[1].m128i_i8[8] )
      {
LABEL_4:
        v17 = result->m128i_u64[1];
        goto LABEL_5;
      }
    }
    v17 = qword_4F81350[0];
LABEL_5:
    if ( v17 < v16 )
    {
      --a2;
      result = (__m128i *)(a1 + 40 * a2);
    }
    v19 = (__m128i *)(a1 + 40 * i);
    *v19 = _mm_loadu_si128(result);
    v19[1] = _mm_loadu_si128(result + 1);
    v19[2].m128i_i64[0] = result[2].m128i_i64[0];
    if ( a2 >= v11 )
      break;
  }
  if ( !v12 )
    goto LABEL_24;
LABEL_13:
  v33 = a9;
  v31 = _mm_loadu_si128(&a7);
  v32 = _mm_loadu_si128(&a8);
  v22 = (a2 - 1) / 2;
  if ( a2 > v9 )
  {
    while ( 1 )
    {
      v23 = v13;
      v24 = (const __m128i *)(a1 + 40 * v22);
      if ( !v14 )
        v23 = qword_4F81350[0];
      if ( v24[1].m128i_i8[8] )
      {
        result = (__m128i *)(a1 + 40 * a2);
        if ( v24->m128i_i64[1] >= v23 )
          goto LABEL_22;
      }
      else
      {
        result = (__m128i *)(a1 + 40 * a2);
        if ( qword_4F81350[0] >= v23 )
          goto LABEL_22;
      }
      *result = _mm_loadu_si128(v24);
      result[1] = _mm_loadu_si128(v24 + 1);
      result[2].m128i_i64[0] = v24[2].m128i_i64[0];
      a2 = v22;
      if ( v9 >= v22 )
        break;
      v22 = (v22 - 1) / 2;
    }
    result = (__m128i *)(a1 + 40 * v22);
  }
LABEL_22:
  v31.m128i_i64[1] = v13;
  v32.m128i_i8[8] = v14;
  v25 = _mm_loadu_si128(&v31);
  result[2].m128i_i64[0] = v33;
  v26 = _mm_loadu_si128(&v32);
  *result = v25;
  result[1] = v26;
  return result;
}
