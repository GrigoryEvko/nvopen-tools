// Function: sub_1411980
// Address: 0x1411980
//
signed __int64 __fastcall sub_1411980(__m128i *a1, __m128i *a2, __int64 a3)
{
  signed __int64 result; // rax
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  __m128i *v7; // r9
  __m128i *v8; // r12
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdi
  __int64 v12; // r10
  __m128i *v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __m128i *v17; // rsi
  unsigned __int64 v18; // r13
  __m128i *v19; // rdx
  __m128i *v20; // r10
  __m128i *v21; // rax
  __int64 v22; // rax
  __m128i v23; // xmm3
  __int64 v24; // rbx
  __int64 i; // rsi
  __m128i *v26; // r14
  unsigned __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r8
  __m128i v30; // xmm5
  __m128i *v31; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v5 = (unsigned __int64)a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_25;
  v7 = a2;
  v8 = a1 + 1;
  v31 = a1 + 2;
  while ( 2 )
  {
    v9 = a1[1].m128i_u64[0];
    v10 = v7[-1].m128i_u64[0];
    --v6;
    v11 = a1->m128i_i64[0];
    v12 = a1->m128i_i64[1];
    v13 = &a1[(__int64)(v7 - a1 + ((unsigned __int64)((char *)v7 - (char *)a1) >> 63)) >> 1];
    v14 = v13->m128i_i64[0];
    if ( v9 < v13->m128i_i64[0] )
    {
      if ( v14 < v10 )
        goto LABEL_6;
      if ( v9 >= v10 )
      {
        v30 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v11;
        a1[1].m128i_i64[1] = v12;
        *a1 = v30;
        v15 = v7[-1].m128i_u64[0];
        goto LABEL_7;
      }
LABEL_22:
      v15 = a1->m128i_i64[0];
      *a1 = _mm_loadu_si128(v7 - 1);
      v7[-1].m128i_i64[0] = v11;
      v7[-1].m128i_i64[1] = v12;
      v11 = a1[1].m128i_u64[0];
      goto LABEL_7;
    }
    if ( v9 < v10 )
    {
      v23 = _mm_loadu_si128(a1 + 1);
      a1[1].m128i_i64[0] = v11;
      a1[1].m128i_i64[1] = v12;
      *a1 = v23;
      v15 = v7[-1].m128i_u64[0];
      goto LABEL_7;
    }
    if ( v14 < v10 )
      goto LABEL_22;
LABEL_6:
    *a1 = _mm_loadu_si128(v13);
    v13->m128i_i64[0] = v11;
    v13->m128i_i64[1] = v12;
    v11 = a1[1].m128i_u64[0];
    v15 = v7[-1].m128i_u64[0];
LABEL_7:
    v16 = a1->m128i_i64[0];
    v17 = v31;
    v18 = (unsigned __int64)v8;
    v19 = v7;
    while ( 1 )
    {
      v5 = v18;
      if ( v16 > v11 )
        goto LABEL_14;
      v20 = v19 - 1;
      if ( v16 >= v15 )
      {
        --v19;
        if ( v18 >= (unsigned __int64)v20 )
          break;
        goto LABEL_13;
      }
      v21 = v19 - 2;
      do
        v19 = v21--;
      while ( v16 < v21[1].m128i_i64[0] );
      if ( v18 >= (unsigned __int64)v19 )
        break;
LABEL_13:
      v22 = v17[-1].m128i_i64[1];
      v17[-1] = _mm_loadu_si128(v19);
      v19->m128i_i64[1] = v22;
      v15 = v19[-1].m128i_u64[0];
      v19->m128i_i64[0] = v11;
      v16 = a1->m128i_i64[0];
LABEL_14:
      v11 = v17->m128i_i64[0];
      v18 += 16LL;
      ++v17;
    }
    sub_1411980(v18, v7, v6);
    result = v18 - (_QWORD)a1;
    if ( (__int64)(v18 - (_QWORD)a1) > 256 )
    {
      if ( v6 )
      {
        v7 = (__m128i *)v18;
        continue;
      }
LABEL_25:
      v24 = result >> 4;
      for ( i = ((result >> 4) - 2) >> 1; ; --i )
      {
        sub_1411670((__int64)a1, i, v24, a1[i].m128i_u64[0], a1[i].m128i_i64[1]);
        if ( !i )
          break;
      }
      v26 = (__m128i *)(v5 - 16);
      do
      {
        v27 = v26->m128i_i64[0];
        v28 = (char *)v26 - (char *)a1;
        v29 = v26->m128i_i64[1];
        --v26;
        v26[1] = _mm_loadu_si128(a1);
        result = (signed __int64)sub_1411670((__int64)a1, 0, v28 >> 4, v27, v29);
      }
      while ( v28 > 16 );
    }
    return result;
  }
}
