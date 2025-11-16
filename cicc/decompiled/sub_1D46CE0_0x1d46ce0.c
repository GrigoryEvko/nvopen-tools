// Function: sub_1D46CE0
// Address: 0x1d46ce0
//
void __fastcall sub_1D46CE0(const __m128i **a1, __m128i *a2, const __m128i *a3, const __m128i *a4)
{
  size_t v4; // r8
  unsigned __int64 v5; // r10
  const __m128i *v6; // r15
  __m128i *v8; // r13
  const __m128i *v10; // r9
  __m128i *v11; // rbx
  __int64 v12; // r9
  __m128i *v13; // rdx
  __m128i *v14; // rsi
  const __m128i *v15; // rax
  size_t v16; // rdx
  const __m128i *v17; // rdi
  unsigned __int64 v18; // rax
  bool v19; // cf
  unsigned __int64 v20; // rax
  const __m128i *v21; // rcx
  __m128i *v22; // r8
  const __m128i *v23; // rsi
  __m128i *v24; // rdx
  __m128i *v25; // rax
  __m128i *v26; // rdx
  __m128i *v27; // rbx
  const __m128i *v28; // r8
  const __m128i *v29; // rsi
  __m128i *v30; // rax
  const __m128i *v31; // rax
  __m128i *v32; // rdx
  const __m128i *v33; // rcx
  __m128i *v34; // rbx
  __int64 v35; // rcx
  __int64 v36; // rax
  __m128i *v37; // [rsp-48h] [rbp-48h]
  size_t v38; // [rsp-40h] [rbp-40h]
  const __m128i *v39; // [rsp-40h] [rbp-40h]
  __int64 v40; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v4 = (char *)a4 - (char *)a3;
  v5 = a4 - a3;
  v6 = a3;
  v8 = a2;
  v10 = a1[2];
  v11 = (__m128i *)a1[1];
  if ( (char *)v10 - (char *)v11 < (unsigned __int64)((char *)a4 - (char *)a3) )
  {
    v17 = *a1;
    v18 = v11 - v17;
    if ( v5 > 0x7FFFFFFFFFFFFFFLL - v18 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v5 < v18 )
      v5 = v11 - v17;
    v19 = __CFADD__(v5, v18);
    v20 = v5 + v18;
    if ( v19 )
    {
      v35 = 0x7FFFFFFFFFFFFFF0LL;
    }
    else
    {
      if ( !v20 )
      {
        v21 = 0;
        v22 = 0;
LABEL_18:
        if ( a2 == v17 )
        {
          v25 = v22;
        }
        else
        {
          v23 = v17;
          v24 = v22;
          v25 = (__m128i *)((char *)v22 + (char *)v8 - (char *)v17);
          do
          {
            if ( v24 )
              *v24 = _mm_loadu_si128(v23);
            ++v24;
            ++v23;
          }
          while ( v24 != v25 );
        }
        v26 = (__m128i *)((char *)v25 + (char *)a4 - (char *)v6);
        do
        {
          if ( v25 )
            *v25 = _mm_loadu_si128(v6);
          ++v25;
          ++v6;
        }
        while ( v25 != v26 );
        if ( v8 == v11 )
        {
          v27 = v25;
        }
        else
        {
          v27 = (__m128i *)((char *)v26 + (char *)v11 - (char *)v8);
          do
          {
            if ( v26 )
              *v26 = _mm_loadu_si128(v8);
            ++v26;
            ++v8;
          }
          while ( v26 != v27 );
        }
        if ( v17 )
        {
          v37 = v22;
          v39 = v21;
          j_j___libc_free_0(v17, (char *)v10 - (char *)v17);
          v22 = v37;
          v21 = v39;
        }
        *a1 = v22;
        a1[1] = v27;
        a1[2] = v21;
        return;
      }
      if ( v20 > 0x7FFFFFFFFFFFFFFLL )
        v20 = 0x7FFFFFFFFFFFFFFLL;
      v35 = 16 * v20;
    }
    v40 = v35;
    v36 = sub_22077B0(v35);
    v17 = *a1;
    v11 = (__m128i *)a1[1];
    v10 = a1[2];
    v22 = (__m128i *)v36;
    v21 = (const __m128i *)(v36 + v40);
    goto LABEL_18;
  }
  v12 = (char *)v11 - (char *)a2;
  if ( v4 < (char *)v11 - (char *)a2 )
  {
    v13 = (__m128i *)a1[1];
    v14 = (__m128i *)((char *)v11 - v4);
    v15 = (__m128i *)((char *)v11 - v4);
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v15);
      ++v13;
      ++v15;
    }
    while ( v13 != (__m128i *)&v11->m128i_i8[v4] );
    a1[1] = (const __m128i *)((char *)a1[1] + v4);
    if ( v8 != v14 )
    {
      v38 = v4;
      memmove(&v8->m128i_i8[v4], v8, (char *)v14 - (char *)v8);
      v4 = v38;
    }
    v16 = v4;
    goto LABEL_11;
  }
  v28 = (const __m128i *)((char *)a3 + v12);
  if ( a4 == (const __m128i *)&a3->m128i_i8[v12] )
  {
    v31 = a1[1];
  }
  else
  {
    v29 = (const __m128i *)((char *)a3 + v12);
    v30 = (__m128i *)a1[1];
    do
    {
      if ( v30 )
        *v30 = _mm_loadu_si128(v29);
      ++v30;
      ++v29;
    }
    while ( v30 != (__m128i *)&v11->m128i_i8[(char *)a4 - (char *)v28] );
    v31 = a1[1];
  }
  v32 = (__m128i *)&v31[v5 - (v12 >> 4)];
  a1[1] = v32;
  if ( v8 != v11 )
  {
    v33 = v8;
    v34 = (__m128i *)((char *)v32 + (char *)v11 - (char *)v8);
    do
    {
      if ( v32 )
        *v32 = _mm_loadu_si128(v33);
      ++v32;
      ++v33;
    }
    while ( v32 != v34 );
    v32 = (__m128i *)a1[1];
  }
  a1[1] = (__m128i *)((char *)v32 + v12);
  if ( v6 != v28 )
  {
    v16 = v12;
LABEL_11:
    memmove(v8, v6, v16);
  }
}
