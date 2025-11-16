// Function: sub_26339F0
// Address: 0x26339f0
//
__int64 __fastcall sub_26339F0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  const void *v4; // r14
  unsigned __int64 v5; // rbx
  __m128i *i; // r15
  size_t v7; // rdx
  int v8; // eax
  unsigned __int64 j; // r13
  size_t v10; // r12
  size_t v11; // rdx
  int v12; // eax
  __int64 v13; // r12
  __int64 k; // rbx
  __m128i *v15; // rbx
  const void *v16; // rcx
  __int64 v17; // r13
  size_t v18; // r8
  __m128i *v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  __m128i *v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  unsigned __int64 v23; // [rsp+30h] [rbp-40h]
  unsigned __int64 v24; // [rsp+38h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v20 = a3;
  v19 = a2;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v21 = a2;
    goto LABEL_29;
  }
  while ( 2 )
  {
    --v20;
    sub_A3B340(
      a1->m128i_i64,
      a1 + 1,
      &a1[(__int64)(((unsigned __int64)((char *)v19 - (char *)a1) >> 63) + v19 - a1) >> 1],
      v19 - 1);
    v24 = (unsigned __int64)v19;
    v4 = (const void *)a1->m128i_i64[0];
    v5 = a1->m128i_u64[1];
    for ( i = a1 + 1; ; ++i )
    {
      v21 = i;
      v7 = i->m128i_u64[1];
      v22 = i->m128i_i64[0];
      if ( v5 <= v7 )
        v7 = v5;
      v23 = i->m128i_u64[1];
      if ( !v7 )
        break;
      v8 = memcmp((const void *)i->m128i_i64[0], v4, v7);
      if ( !v8 )
        break;
      if ( v8 >= 0 )
        goto LABEL_13;
LABEL_6:
      ;
    }
    if ( v5 != v23 && v5 > v23 )
      goto LABEL_6;
LABEL_13:
    for ( j = v24 - 16; ; j -= 16LL )
    {
      v10 = *(_QWORD *)(j + 8);
      v24 = j;
      v11 = v10;
      if ( v5 <= v10 )
        v11 = v5;
      if ( v11 )
      {
        v12 = memcmp(v4, *(const void **)j, v11);
        if ( v12 )
          break;
      }
      if ( v5 == v10 || v5 >= v10 )
      {
        if ( (unsigned __int64)i >= j )
          goto LABEL_23;
LABEL_5:
        *i = _mm_loadu_si128((const __m128i *)j);
        *(_QWORD *)j = v22;
        *(_QWORD *)(j + 8) = v23;
        v5 = a1->m128i_u64[1];
        v4 = (const void *)a1->m128i_i64[0];
        goto LABEL_6;
      }
LABEL_20:
      ;
    }
    if ( v12 < 0 )
      goto LABEL_20;
    if ( (unsigned __int64)i < j )
      goto LABEL_5;
LABEL_23:
    sub_26339F0(i, v19, v20);
    result = (char *)i - (char *)a1;
    if ( (char *)i - (char *)a1 > 256 )
    {
      if ( v20 )
      {
        v19 = i;
        continue;
      }
LABEL_29:
      v13 = result >> 4;
      for ( k = ((result >> 4) - 2) >> 1; ; --k )
      {
        sub_A3B750((__int64)a1, k, v13, (const void *)a1[k].m128i_i64[0], a1[k].m128i_u64[1]);
        if ( !k )
          break;
      }
      v15 = v21 - 1;
      do
      {
        v16 = (const void *)v15->m128i_i64[0];
        v17 = (char *)v15 - (char *)a1;
        v18 = v15->m128i_u64[1];
        --v15;
        v15[1] = _mm_loadu_si128(a1);
        result = (__int64)sub_A3B750((__int64)a1, 0, v17 >> 4, v16, v18);
      }
      while ( v17 > 16 );
    }
    return result;
  }
}
