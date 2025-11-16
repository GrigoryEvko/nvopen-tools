// Function: sub_2443BE0
// Address: 0x2443be0
//
signed __int64 __fastcall sub_2443BE0(__m128i *a1, __m128i *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 v4; // r15
  __m128i *v6; // r13
  __m128i *v7; // r8
  __m128i *v8; // r12
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rdi
  __int64 v11; // r9
  unsigned __int64 v12; // rsi
  __m128i *v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __m128i *v17; // rbx
  __m128i *v18; // rdx
  __m128i *v19; // rdi
  __m128i *v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm3
  __int64 v23; // rbx
  __int64 i; // rsi
  __m128i *v25; // r13
  __int64 v26; // rcx
  __int64 v27; // rbx
  unsigned __int64 v28; // r8
  __m128i v29; // xmm5

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = a3;
  v6 = a2;
  if ( !a3 )
    goto LABEL_25;
  v7 = a2;
  v8 = a1 + 1;
  while ( 2 )
  {
    v9 = a1[1].m128i_u64[1];
    v10 = v7[-1].m128i_u64[1];
    --v4;
    v11 = a1->m128i_i64[0];
    v12 = a1->m128i_u64[1];
    v13 = &a1[(__int64)(v7 - a1 + ((unsigned __int64)((char *)v7 - (char *)a1) >> 63)) >> 1];
    v14 = v13->m128i_u64[1];
    if ( v9 > v14 )
    {
      if ( v14 > v10 )
        goto LABEL_6;
      if ( v9 <= v10 )
      {
        v29 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v11;
        a1[1].m128i_i64[1] = v12;
        *a1 = v29;
        v15 = v7[-1].m128i_u64[1];
        goto LABEL_7;
      }
LABEL_22:
      v15 = a1->m128i_u64[1];
      *a1 = _mm_loadu_si128(v7 - 1);
      v7[-1].m128i_i64[0] = v11;
      v7[-1].m128i_i64[1] = v12;
      v12 = a1[1].m128i_u64[1];
      goto LABEL_7;
    }
    if ( v9 > v10 )
    {
      v22 = _mm_loadu_si128(a1 + 1);
      a1[1].m128i_i64[0] = v11;
      a1[1].m128i_i64[1] = v12;
      *a1 = v22;
      v15 = v7[-1].m128i_u64[1];
      goto LABEL_7;
    }
    if ( v14 > v10 )
      goto LABEL_22;
LABEL_6:
    *a1 = _mm_loadu_si128(v13);
    v13->m128i_i64[0] = v11;
    v13->m128i_i64[1] = v12;
    v12 = a1[1].m128i_u64[1];
    v15 = v7[-1].m128i_u64[1];
LABEL_7:
    v16 = a1->m128i_u64[1];
    v17 = v8;
    v18 = v7;
    while ( 1 )
    {
      v6 = v17;
      if ( v16 < v12 )
        goto LABEL_14;
      v19 = v18 - 1;
      if ( v16 <= v15 )
      {
        --v18;
        if ( v17 >= v19 )
          break;
        goto LABEL_13;
      }
      v20 = v18 - 2;
      do
        v18 = v20--;
      while ( v16 > v20[1].m128i_i64[1] );
      if ( v17 >= v18 )
        break;
LABEL_13:
      v21 = v17->m128i_i64[0];
      *v17 = _mm_loadu_si128(v18);
      v18->m128i_i64[0] = v21;
      v15 = v18[-1].m128i_u64[1];
      v18->m128i_i64[1] = v12;
      v16 = a1->m128i_u64[1];
LABEL_14:
      v12 = v17[1].m128i_u64[1];
      ++v17;
    }
    sub_2443BE0(v17, v7, v4);
    result = (char *)v17 - (char *)a1;
    if ( (char *)v17 - (char *)a1 > 256 )
    {
      if ( v4 )
      {
        v7 = v17;
        continue;
      }
LABEL_25:
      v23 = result >> 4;
      for ( i = ((result >> 4) - 2) >> 1; ; --i )
      {
        sub_2443620((__int64)a1, i, v23, a1[i].m128i_i64[0], a1[i].m128i_u64[1]);
        if ( !i )
          break;
      }
      v25 = v6 - 1;
      do
      {
        v26 = v25->m128i_i64[0];
        v27 = (char *)v25 - (char *)a1;
        v28 = v25->m128i_u64[1];
        --v25;
        v25[1] = _mm_loadu_si128(a1);
        result = (signed __int64)sub_2443620((__int64)a1, 0, v27 >> 4, v26, v28);
      }
      while ( v27 > 16 );
    }
    return result;
  }
}
