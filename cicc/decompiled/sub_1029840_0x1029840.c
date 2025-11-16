// Function: sub_1029840
// Address: 0x1029840
//
signed __int64 __fastcall sub_1029840(__m128i *a1, __m128i *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 v4; // r15
  __m128i *v6; // r13
  __m128i *v7; // r12
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  __m128i v10; // xmm0
  __m128i *v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __m128i *v14; // rbx
  __m128i *v15; // rdx
  __m128i *v16; // rax
  __m128i *v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm4
  __int64 v20; // rbx
  __int64 i; // rsi
  __m128i *v22; // r13
  unsigned __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // r8
  __m128i v26; // xmm6

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = a3;
  v6 = a2;
  if ( !a3 )
    goto LABEL_25;
  v7 = a1 + 1;
  while ( 2 )
  {
    v8 = a1[1].m128i_u64[0];
    v9 = a2[-1].m128i_u64[0];
    --v4;
    v10 = _mm_loadu_si128(a1);
    v11 = &a1[(__int64)(a2 - a1 + ((unsigned __int64)((char *)a2 - (char *)a1) >> 63)) >> 1];
    v12 = v11->m128i_i64[0];
    if ( v8 < v11->m128i_i64[0] )
    {
      if ( v12 < v9 )
        goto LABEL_6;
      if ( v8 >= v9 )
      {
        v26 = _mm_loadu_si128(a1 + 1);
        a1[1] = v10;
        *a1 = v26;
        goto LABEL_7;
      }
LABEL_22:
      *a1 = _mm_loadu_si128(a2 - 1);
      a2[-1] = v10;
      goto LABEL_7;
    }
    if ( v8 < v9 )
    {
      v19 = _mm_loadu_si128(a1 + 1);
      a1[1] = v10;
      *a1 = v19;
      goto LABEL_7;
    }
    if ( v12 < v9 )
      goto LABEL_22;
LABEL_6:
    *a1 = _mm_loadu_si128(v11);
    *v11 = v10;
LABEL_7:
    v13 = a1->m128i_i64[0];
    v14 = v7;
    v15 = a2;
    while ( 1 )
    {
      v6 = v14;
      if ( v14->m128i_i64[0] < v13 )
        goto LABEL_14;
      v16 = v15 - 1;
      if ( v15[-1].m128i_i64[0] <= v13 )
      {
        --v15;
        if ( v14 >= v16 )
          break;
        goto LABEL_13;
      }
      v17 = v15 - 2;
      do
        v15 = v17--;
      while ( v17[1].m128i_i64[0] > v13 );
      if ( v14 >= v15 )
        break;
LABEL_13:
      v18 = _mm_loadu_si128(v14);
      *v14 = _mm_loadu_si128(v15);
      *v15 = v18;
      v13 = a1->m128i_i64[0];
LABEL_14:
      ++v14;
    }
    sub_1029840(v14, a2, v4);
    result = (char *)v14 - (char *)a1;
    if ( (char *)v14 - (char *)a1 > 256 )
    {
      if ( v4 )
      {
        a2 = v14;
        continue;
      }
LABEL_25:
      v20 = result >> 4;
      for ( i = ((result >> 4) - 2) >> 1; ; --i )
      {
        sub_1029510((__int64)a1, i, v20, a1[i].m128i_u64[0], a1[i].m128i_i64[1]);
        if ( !i )
          break;
      }
      v22 = v6 - 1;
      do
      {
        v23 = v22->m128i_i64[0];
        v24 = (char *)v22 - (char *)a1;
        v25 = v22->m128i_i64[1];
        --v22;
        v22[1] = _mm_loadu_si128(a1);
        result = (signed __int64)sub_1029510((__int64)a1, 0, v24 >> 4, v23, v25);
      }
      while ( v24 > 16 );
    }
    return result;
  }
}
