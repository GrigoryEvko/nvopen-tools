// Function: sub_3366040
// Address: 0x3366040
//
void __fastcall sub_3366040(unsigned __int64 *a1, __m128i *a2, const __m128i *a3, const __m128i *a4)
{
  size_t v4; // r8
  unsigned __int64 v5; // r10
  const __m128i *v6; // r15
  __m128i *v8; // r13
  __m128i *v10; // rbx
  __int64 v11; // r9
  __m128i *v12; // rdx
  __m128i *v13; // rsi
  const __m128i *v14; // rax
  size_t v15; // rdx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rax
  bool v18; // cf
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __m128i *v21; // r8
  const __m128i *v22; // rsi
  __m128i *v23; // rdx
  __m128i *v24; // rax
  __m128i *v25; // rdx
  __m128i *v26; // rbx
  const __m128i *v27; // r8
  const __m128i *v28; // rsi
  __m128i *v29; // rax
  unsigned __int64 v30; // rax
  __m128i *v31; // rdx
  const __m128i *v32; // rcx
  __m128i *v33; // rbx
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __m128i *v36; // [rsp-48h] [rbp-48h]
  size_t v37; // [rsp-40h] [rbp-40h]
  unsigned __int64 v38; // [rsp-40h] [rbp-40h]
  unsigned __int64 v39; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v4 = (char *)a4 - (char *)a3;
  v5 = a4 - a3;
  v6 = a3;
  v8 = a2;
  v10 = (__m128i *)a1[1];
  if ( a1[2] - (unsigned __int64)v10 < (char *)a4 - (char *)a3 )
  {
    v16 = *a1;
    v17 = (__int64)((__int64)v10->m128i_i64 - v16) >> 4;
    if ( v5 > 0x7FFFFFFFFFFFFFFLL - v17 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v5 < v17 )
      v5 = (__int64)((__int64)v10->m128i_i64 - v16) >> 4;
    v18 = __CFADD__(v5, v17);
    v19 = v5 + v17;
    if ( v18 )
    {
      v34 = 0x7FFFFFFFFFFFFFF0LL;
    }
    else
    {
      if ( !v19 )
      {
        v20 = 0;
        v21 = 0;
LABEL_18:
        if ( a2 == (__m128i *)v16 )
        {
          v24 = v21;
        }
        else
        {
          v22 = (const __m128i *)v16;
          v23 = v21;
          v24 = (__m128i *)((char *)v8 + (_QWORD)v21 - v16);
          do
          {
            if ( v23 )
              *v23 = _mm_loadu_si128(v22);
            ++v23;
            ++v22;
          }
          while ( v23 != v24 );
        }
        v25 = (__m128i *)((char *)v24 + (char *)a4 - (char *)v6);
        do
        {
          if ( v24 )
            *v24 = _mm_loadu_si128(v6);
          ++v24;
          ++v6;
        }
        while ( v24 != v25 );
        if ( v8 == v10 )
        {
          v26 = v24;
        }
        else
        {
          v26 = (__m128i *)((char *)v25 + (char *)v10 - (char *)v8);
          do
          {
            if ( v25 )
              *v25 = _mm_loadu_si128(v8);
            ++v25;
            ++v8;
          }
          while ( v25 != v26 );
        }
        if ( v16 )
        {
          v36 = v21;
          v38 = v20;
          j_j___libc_free_0(v16);
          v21 = v36;
          v20 = v38;
        }
        *a1 = (unsigned __int64)v21;
        a1[1] = (unsigned __int64)v26;
        a1[2] = v20;
        return;
      }
      if ( v19 > 0x7FFFFFFFFFFFFFFLL )
        v19 = 0x7FFFFFFFFFFFFFFLL;
      v34 = 16 * v19;
    }
    v39 = v34;
    v35 = sub_22077B0(v34);
    v16 = *a1;
    v10 = (__m128i *)a1[1];
    v21 = (__m128i *)v35;
    v20 = v35 + v39;
    goto LABEL_18;
  }
  v11 = (char *)v10 - (char *)a2;
  if ( v4 < (char *)v10 - (char *)a2 )
  {
    v12 = (__m128i *)a1[1];
    v13 = (__m128i *)((char *)v10 - v4);
    v14 = (__m128i *)((char *)v10 - v4);
    do
    {
      if ( v12 )
        *v12 = _mm_loadu_si128(v14);
      ++v12;
      ++v14;
    }
    while ( v12 != (__m128i *)&v10->m128i_i8[v4] );
    a1[1] += v4;
    if ( v8 != v13 )
    {
      v37 = v4;
      memmove(&v8->m128i_i8[v4], v8, (char *)v13 - (char *)v8);
      v4 = v37;
    }
    v15 = v4;
    goto LABEL_11;
  }
  v27 = (const __m128i *)((char *)a3 + v11);
  if ( a4 == (const __m128i *)&a3->m128i_i8[v11] )
  {
    v30 = a1[1];
  }
  else
  {
    v28 = (const __m128i *)((char *)a3 + v11);
    v29 = (__m128i *)a1[1];
    do
    {
      if ( v29 )
        *v29 = _mm_loadu_si128(v28);
      ++v29;
      ++v28;
    }
    while ( v29 != (__m128i *)&v10->m128i_i8[(char *)a4 - (char *)v27] );
    v30 = a1[1];
  }
  v31 = (__m128i *)(v30 + 16 * (v5 - (v11 >> 4)));
  a1[1] = (unsigned __int64)v31;
  if ( v8 != v10 )
  {
    v32 = v8;
    v33 = (__m128i *)((char *)v31 + (char *)v10 - (char *)v8);
    do
    {
      if ( v31 )
        *v31 = _mm_loadu_si128(v32);
      ++v31;
      ++v32;
    }
    while ( v31 != v33 );
    v31 = (__m128i *)a1[1];
  }
  a1[1] = (unsigned __int64)v31->m128i_u64 + v11;
  if ( v6 != v27 )
  {
    v15 = v11;
LABEL_11:
    memmove(v8, v6, v15);
  }
}
