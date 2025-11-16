// Function: sub_2D25A70
// Address: 0x2d25a70
//
__m128i *__fastcall sub_2D25A70(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8)
{
  __m128i *result; // rax
  __m128i *v9; // r10
  __int64 v11; // r13
  __m128i *v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rcx
  __int8 v15; // si
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __m128i v18; // xmm7
  __int64 v19; // rdx
  char v20; // di
  unsigned __int64 v21; // rsi
  __m128i *v22; // rax
  __m128i *i; // rbx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  __m128i *v26; // r15
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rax
  __m128i v32; // xmm5
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // xmm4_8
  __m128i *v39; // [rsp-70h] [rbp-70h]

  result = (__m128i *)((char *)a2 - a1);
  if ( (__int64)a2->m128i_i64 - a1 <= 640 )
    return result;
  v9 = a2;
  v11 = a3;
  if ( !a3 )
  {
    v26 = a2;
    goto LABEL_42;
  }
  v39 = (__m128i *)(a1 + 40);
  while ( 2 )
  {
    --v11;
    v12 = (__m128i *)(a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * ((__int64)result >> 3)) >> 1));
    if ( v12[1].m128i_i8[8] )
      v13 = v12->m128i_u64[1];
    else
      v13 = qword_4F81350[0];
    if ( *(_BYTE *)(a1 + 64) )
      v14 = *(_QWORD *)(a1 + 48);
    else
      v14 = qword_4F81350[0];
    v15 = v9[-1].m128i_i8[0];
    if ( v14 >= v13 )
    {
      if ( v15 )
      {
        v29 = v9[-2].m128i_u64[0];
        if ( v14 >= v29 )
        {
LABEL_31:
          if ( v13 < v29 )
            goto LABEL_36;
          v30 = *(_QWORD *)(a1 + 32);
          a8 = _mm_loadu_si128((const __m128i *)a1);
          a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
          *(__m128i *)a1 = _mm_loadu_si128(v12);
          *(__m128i *)(a1 + 16) = _mm_loadu_si128(v12 + 1);
LABEL_33:
          *(_QWORD *)(a1 + 32) = v12[2].m128i_i64[0];
          v12[2].m128i_i64[0] = v30;
          *v12 = a8;
          v12[1] = a7;
          goto LABEL_14;
        }
      }
      else
      {
        v29 = qword_4F81350[0];
        if ( v14 >= qword_4F81350[0] )
          goto LABEL_31;
      }
      a8 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v17 = *(_QWORD *)(a1 + 32);
      v32 = _mm_loadu_si128((const __m128i *)(a1 + 56));
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      *(__m128i *)(a1 + 16) = v32;
      goto LABEL_13;
    }
    if ( v15 )
    {
      v16 = v9[-2].m128i_u64[0];
      if ( v13 >= v16 )
        goto LABEL_11;
      goto LABEL_38;
    }
    v16 = qword_4F81350[0];
    if ( v13 < qword_4F81350[0] )
    {
LABEL_38:
      a8 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v30 = *(_QWORD *)(a1 + 32);
      *(__m128i *)a1 = _mm_loadu_si128(v12);
      *(__m128i *)(a1 + 16) = _mm_loadu_si128(v12 + 1);
      goto LABEL_33;
    }
LABEL_11:
    if ( v14 < v16 )
    {
LABEL_36:
      a8 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v31 = *(_QWORD *)(a1 + 32);
      *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v9 - 40));
      *(__m128i *)(a1 + 16) = _mm_loadu_si128((__m128i *)((char *)v9 - 24));
      *(_QWORD *)(a1 + 32) = v9[-1].m128i_i64[1];
      v9[-1].m128i_i64[1] = v31;
      *(__m128i *)((char *)v9 - 40) = a8;
      *(__m128i *)((char *)v9 - 24) = a7;
      goto LABEL_14;
    }
    v17 = *(_QWORD *)(a1 + 32);
    a8 = _mm_loadu_si128((const __m128i *)a1);
    a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 56));
    *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 40));
    *(__m128i *)(a1 + 16) = v18;
LABEL_13:
    v19 = *(_QWORD *)(a1 + 72);
    *(_QWORD *)(a1 + 72) = v17;
    *(__m128i *)(a1 + 40) = a8;
    *(_QWORD *)(a1 + 32) = v19;
    *(__m128i *)(a1 + 56) = a7;
LABEL_14:
    v20 = *(_BYTE *)(a1 + 24);
    v21 = qword_4F81350[0];
    v22 = v9;
    for ( i = v39; ; i = (__m128i *)((char *)i + 40) )
    {
      v26 = i;
      if ( v20 )
      {
        v24 = *(_QWORD *)(a1 + 8);
        v25 = v21;
        if ( !i[1].m128i_i8[8] )
          goto LABEL_17;
      }
      else
      {
        v24 = v21;
        if ( !i[1].m128i_i8[8] )
          goto LABEL_21;
      }
      v25 = i->m128i_u64[1];
LABEL_17:
      if ( v24 > v25 )
        continue;
      do
      {
LABEL_21:
        v22 = (__m128i *)((char *)v22 - 40);
        v27 = v21;
        if ( v22[1].m128i_i8[8] )
          v27 = v22->m128i_u64[1];
      }
      while ( v27 > v24 );
      if ( i >= v22 )
        break;
      a8 = _mm_loadu_si128(i);
      a7 = _mm_loadu_si128(i + 1);
      v28 = i[2].m128i_i64[0];
      *i = _mm_loadu_si128(v22);
      i[1] = _mm_loadu_si128(v22 + 1);
      i[2].m128i_i64[0] = v22[2].m128i_i64[0];
      *v22 = a8;
      v21 = qword_4F81350[0];
      v22[2].m128i_i64[0] = v28;
      v22[1] = a7;
      v20 = *(_BYTE *)(a1 + 24);
    }
    sub_2D25A70(i, v9, v11);
    result = (__m128i *)((char *)i - a1);
    if ( (__int64)i->m128i_i64 - a1 > 640 )
    {
      if ( v11 )
      {
        v9 = i;
        continue;
      }
LABEL_42:
      v33 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)result >> 3);
      v34 = (v33 - 2) >> 1;
      sub_2D22C30(a1, v34, v33, a4, a5, a6, a7, a8, _mm_loadu_si128((const __m128i *)(a1 + 40 * v34)).m128i_i64[0]);
      do
      {
        --v34;
        sub_2D22C30(a1, v34, v33, v35, v36, v37, a7, a8, _mm_loadu_si128((const __m128i *)(a1 + 40 * v34)).m128i_i64[0]);
      }
      while ( v34 );
      do
      {
        v26 = (__m128i *)((char *)v26 - 40);
        v38 = _mm_loadu_si128(v26).m128i_u64[0];
        *v26 = _mm_loadu_si128((const __m128i *)a1);
        v26[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
        v26[2].m128i_i64[0] = *(_QWORD *)(a1 + 32);
        result = sub_2D22C30(
                   a1,
                   0,
                   0xCCCCCCCCCCCCCCCDLL * (((__int64)v26->m128i_i64 - a1) >> 3),
                   v35,
                   v36,
                   v37,
                   a7,
                   a8,
                   v38);
      }
      while ( (__int64)v26->m128i_i64 - a1 > 40 );
    }
    return result;
  }
}
