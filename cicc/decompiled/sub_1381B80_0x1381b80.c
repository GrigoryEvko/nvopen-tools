// Function: sub_1381B80
// Address: 0x1381b80
//
signed __int64 __fastcall sub_1381B80(__m128i *a1, __m128i *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 v4; // r14
  __m128i *v6; // r11
  __m128i *v7; // rbx
  unsigned __int64 v8; // rcx
  __int64 v9; // r10
  unsigned __int64 v10; // rdi
  __int64 v11; // r12
  unsigned __int64 v12; // rsi
  __int64 v13; // r8
  __m128i *v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // r9
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  __int64 v20; // r10
  __m128i *v21; // rdx
  __m128i *v22; // rax
  __m128i *v23; // r12
  __m128i v24; // xmm3
  __int64 v25; // r14
  __int64 i; // rbx
  __m128i *v27; // r12
  unsigned __int64 v28; // rcx
  __int64 v29; // rbx
  __int64 v30; // r8
  __m128i v31; // xmm5

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = a3;
  if ( !a3 )
  {
    v23 = a2;
    goto LABEL_35;
  }
  v6 = a2;
  v7 = a1 + 2;
  while ( 2 )
  {
    v8 = a1[1].m128i_u64[0];
    v9 = a1[1].m128i_i64[1];
    --v4;
    v10 = v6[-1].m128i_u64[0];
    v11 = v6[-1].m128i_i64[1];
    v12 = a1->m128i_i64[0];
    v13 = a1->m128i_i64[1];
    v14 = &a1[(__int64)(v6 - a1 + ((unsigned __int64)((char *)v6 - (char *)a1) >> 63)) >> 1];
    v15 = v14->m128i_i64[0];
    v16 = v14->m128i_i64[1];
    if ( v8 < v14->m128i_i64[0] || v9 < v16 && v8 == v15 )
    {
      if ( v15 < v10 || v16 < v11 && v15 == v10 )
      {
LABEL_29:
        *a1 = _mm_loadu_si128(v14);
        v14->m128i_i64[0] = v12;
        v14->m128i_i64[1] = v13;
        v12 = a1[1].m128i_u64[0];
        v13 = a1[1].m128i_i64[1];
        v18 = v6[-1].m128i_u64[0];
        v17 = v6[-1].m128i_i64[1];
        goto LABEL_17;
      }
      if ( v8 >= v10 && (v9 >= v11 || v8 != v10) )
      {
        v31 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v12;
        a1[1].m128i_i64[1] = v13;
        *a1 = v31;
        v18 = v6[-1].m128i_u64[0];
        v17 = v6[-1].m128i_i64[1];
        goto LABEL_17;
      }
    }
    else
    {
      if ( v8 < v10 || v9 < v11 && v8 == v10 )
      {
        v24 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v12;
        a1[1].m128i_i64[1] = v13;
        *a1 = v24;
        v18 = v6[-1].m128i_u64[0];
        v17 = v6[-1].m128i_i64[1];
        goto LABEL_17;
      }
      if ( v15 >= v10 && (v16 >= v11 || v15 != v10) )
        goto LABEL_29;
    }
    *a1 = _mm_loadu_si128(v6 - 1);
    v6[-1].m128i_i64[0] = v12;
    v17 = v13;
    v18 = v12;
    v6[-1].m128i_i64[1] = v13;
    v13 = a1[1].m128i_i64[1];
    v12 = a1[1].m128i_u64[0];
LABEL_17:
    v19 = a1->m128i_i64[0];
    v20 = a1->m128i_i64[1];
    v21 = v7;
    v22 = v6;
    while ( 1 )
    {
      v23 = v21 - 1;
      if ( v12 >= v19 && (v13 >= v20 || v12 != v19) )
        break;
LABEL_22:
      v12 = v21->m128i_i64[0];
      v13 = v21->m128i_i64[1];
      ++v21;
    }
    --v22;
    while ( v18 > v19 || v17 > v20 && v18 == v19 )
    {
      --v22;
      v18 = v22->m128i_i64[0];
      v17 = v22->m128i_i64[1];
    }
    if ( v22 > v23 )
    {
      v21[-1] = _mm_loadu_si128(v22);
      v18 = v22[-1].m128i_u64[0];
      v17 = v22[-1].m128i_i64[1];
      v22->m128i_i64[0] = v12;
      v22->m128i_i64[1] = v13;
      v19 = a1->m128i_i64[0];
      v20 = a1->m128i_i64[1];
      goto LABEL_22;
    }
    sub_1381B80(v23, v6, v4);
    result = (char *)v23 - (char *)a1;
    if ( (char *)v23 - (char *)a1 > 256 )
    {
      if ( v4 )
      {
        v6 = v23;
        continue;
      }
LABEL_35:
      v25 = result >> 4;
      for ( i = ((result >> 4) - 2) >> 1; ; --i )
      {
        sub_1381540((__int64)a1, i, v25, a1[i].m128i_u64[0], a1[i].m128i_i64[1]);
        if ( !i )
          break;
      }
      v27 = v23 - 1;
      do
      {
        v28 = v27->m128i_i64[0];
        v29 = (char *)v27 - (char *)a1;
        v30 = v27->m128i_i64[1];
        --v27;
        v27[1] = _mm_loadu_si128(a1);
        result = (signed __int64)sub_1381540((__int64)a1, 0, v29 >> 4, v28, v30);
      }
      while ( v29 > 16 );
    }
    return result;
  }
}
