// Function: sub_A3B340
// Address: 0xa3b340
//
__int64 __fastcall sub_A3B340(__int64 *a1, __m128i *a2, __m128i *a3, __m128i *a4)
{
  unsigned __int64 v6; // r15
  size_t v7; // r14
  const void *v8; // r9
  const void *v9; // r10
  size_t v10; // rdx
  int v11; // eax
  void *v12; // r8
  size_t v13; // rdx
  const void *v14; // rsi
  int v15; // eax
  __int64 v16; // rdx
  __int64 result; // rax
  void *v18; // r8
  const void *v19; // rsi
  size_t v20; // rdx
  int v21; // eax
  size_t v22; // rdx
  int v23; // eax
  __int64 v24; // rdx
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // rdx
  __m128i *v28; // [rsp+0h] [rbp-50h]
  __m128i *v29; // [rsp+0h] [rbp-50h]
  __m128i *v30; // [rsp+8h] [rbp-48h]
  const void *v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  const void *v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  const void *v35; // [rsp+10h] [rbp-40h]
  __m128i *v36; // [rsp+10h] [rbp-40h]
  __m128i *v37; // [rsp+10h] [rbp-40h]
  void *s2; // [rsp+18h] [rbp-38h]
  void *s2a; // [rsp+18h] [rbp-38h]
  void *s2b; // [rsp+18h] [rbp-38h]

  v6 = a2->m128i_u64[1];
  v8 = (const void *)a3->m128i_i64[0];
  v9 = (const void *)a2->m128i_i64[0];
  v10 = a3->m128i_u64[1];
  v7 = v10;
  if ( v6 <= v10 )
    v10 = a2->m128i_u64[1];
  if ( v10 )
  {
    v30 = a4;
    v33 = v8;
    s2 = (void *)a2->m128i_i64[0];
    v11 = memcmp(v9, v8, v10);
    v9 = s2;
    v8 = v33;
    a4 = v30;
    if ( v11 )
    {
      if ( v11 < 0 )
        goto LABEL_7;
LABEL_15:
      v18 = (void *)a4->m128i_i64[1];
      v19 = (const void *)a4->m128i_i64[0];
      v20 = (size_t)v18;
      if ( v6 <= (unsigned __int64)v18 )
        v20 = v6;
      if ( v20
        && (v29 = a4,
            v32 = a4->m128i_i64[1],
            v35 = v8,
            v21 = memcmp(v9, v19, v20),
            v8 = v35,
            v18 = (void *)v32,
            a4 = v29,
            v21) )
      {
        if ( v21 < 0 )
          goto LABEL_41;
      }
      else if ( (void *)v6 != v18 && v6 < (unsigned __int64)v18 )
      {
        goto LABEL_41;
      }
      v25 = (size_t)v18;
      if ( v7 <= (unsigned __int64)v18 )
        v25 = v7;
      if ( !v25 || (v37 = a4, s2b = v18, v26 = memcmp(v8, v19, v25), v18 = s2b, a4 = v37, !v26) )
      {
        if ( (void *)v7 == v18 || v7 >= (unsigned __int64)v18 )
          goto LABEL_13;
        goto LABEL_29;
      }
      if ( v26 < 0 )
        goto LABEL_29;
      goto LABEL_13;
    }
  }
  if ( v6 == v7 || v6 >= v7 )
    goto LABEL_15;
LABEL_7:
  v12 = (void *)a4->m128i_i64[1];
  v13 = v7;
  v14 = (const void *)a4->m128i_i64[0];
  if ( (unsigned __int64)v12 <= v7 )
    v13 = a4->m128i_u64[1];
  if ( v13 )
  {
    v28 = a4;
    v31 = v9;
    v34 = a4->m128i_i64[1];
    v15 = memcmp(v8, v14, v13);
    v12 = (void *)v34;
    v9 = v31;
    a4 = v28;
    if ( v15 )
    {
      if ( v15 >= 0 )
        goto LABEL_23;
LABEL_13:
      v16 = *a1;
      result = a1[1];
      *(__m128i *)a1 = _mm_loadu_si128(a3);
      a3->m128i_i64[1] = result;
      a3->m128i_i64[0] = v16;
      return result;
    }
  }
  if ( v12 != (void *)v7 && (unsigned __int64)v12 > v7 )
    goto LABEL_13;
LABEL_23:
  v22 = v6;
  if ( (unsigned __int64)v12 <= v6 )
    v22 = (size_t)v12;
  if ( v22 )
  {
    v36 = a4;
    s2a = v12;
    v23 = memcmp(v9, v14, v22);
    v12 = s2a;
    a4 = v36;
    if ( v23 )
    {
      if ( v23 >= 0 )
        goto LABEL_41;
LABEL_29:
      v24 = *a1;
      result = a1[1];
      *(__m128i *)a1 = _mm_loadu_si128(a4);
      a4->m128i_i64[0] = v24;
      a4->m128i_i64[1] = result;
      return result;
    }
  }
  if ( v12 != (void *)v6 && (unsigned __int64)v12 > v6 )
    goto LABEL_29;
LABEL_41:
  v27 = *a1;
  result = a1[1];
  *(__m128i *)a1 = _mm_loadu_si128(a2);
  a2->m128i_i64[1] = result;
  a2->m128i_i64[0] = v27;
  return result;
}
