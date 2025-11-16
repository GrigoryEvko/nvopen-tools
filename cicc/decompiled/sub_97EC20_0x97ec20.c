// Function: sub_97EC20
// Address: 0x97ec20
//
void __fastcall sub_97EC20(__m128i **a1, __m128i *a2, const __m128i *a3, const __m128i *a4)
{
  size_t v5; // r15
  const __m128i *v6; // r13
  unsigned __int64 v8; // rsi
  __m128i *v10; // r9
  __m128i *v11; // rdi
  __int64 v12; // r8
  __m128i *v13; // rax
  __m128i *v14; // rsi
  const __m128i *v15; // rdx
  size_t v16; // rdx
  __m128i *v17; // r8
  unsigned __int64 v18; // rdx
  bool v19; // cf
  unsigned __int64 v20; // rdx
  __int64 v21; // r15
  __m128i *v22; // rcx
  const __m128i *v23; // rsi
  __m128i *v24; // rax
  __m128i *v25; // rdx
  __m128i *v26; // rax
  __m128i v27; // xmm5
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  __m128i *v30; // rbx
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  const __m128i *v34; // r9
  signed __int64 v35; // rbx
  __m128i *v36; // rdx
  const __m128i *v37; // rcx
  __m128i *v38; // rbx
  __m128i *v39; // rdx
  __m128i *v40; // rax
  const __m128i *v41; // rdx
  __m128i *v42; // rdi
  __int64 v43; // r15
  __int64 v44; // rax
  __m128i *v45; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v5 = (char *)a4 - (char *)a3;
  v6 = a3;
  v8 = ((char *)a4 - (char *)a3) >> 6;
  v10 = a1[2];
  v11 = a1[1];
  if ( (char *)v10 - (char *)v11 < (unsigned __int64)((char *)a4 - (char *)a3) )
  {
    v17 = *a1;
    v18 = ((char *)v11 - (char *)*a1) >> 6;
    if ( v8 > 0x1FFFFFFFFFFFFFFLL - v18 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v8 < v18 )
      v8 = ((char *)v11 - (char *)*a1) >> 6;
    v19 = __CFADD__(v8, v18);
    v20 = v8 + v18;
    if ( v19 )
    {
      v43 = 0x7FFFFFFFFFFFFFC0LL;
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
          v25 = (__m128i *)((char *)v22 + (char *)a2 - (char *)v17);
          do
          {
            if ( v24 )
            {
              *v24 = _mm_loadu_si128(v23);
              v24[1] = _mm_loadu_si128(v23 + 1);
              v24[2] = _mm_loadu_si128(v23 + 2);
              v24[3] = _mm_loadu_si128(v23 + 3);
            }
            v24 += 4;
            v23 += 4;
          }
          while ( v25 != v24 );
        }
        v26 = (__m128i *)((char *)v25 + (char *)a4 - (char *)v6);
        do
        {
          if ( v25 )
          {
            v27 = _mm_loadu_si128(v6 + 1);
            v28 = _mm_loadu_si128(v6 + 2);
            v29 = _mm_loadu_si128(v6 + 3);
            *v25 = _mm_loadu_si128(v6);
            v25[1] = v27;
            v25[2] = v28;
            v25[3] = v29;
          }
          v25 += 4;
          v6 += 4;
        }
        while ( v26 != v25 );
        if ( a2 == v11 )
        {
          v30 = v26;
        }
        else
        {
          v30 = (__m128i *)((char *)v26 + (char *)v11 - (char *)a2);
          do
          {
            if ( v26 )
            {
              v31 = _mm_loadu_si128(a2 + 1);
              v32 = _mm_loadu_si128(a2 + 2);
              v33 = _mm_loadu_si128(a2 + 3);
              *v26 = _mm_loadu_si128(a2);
              v26[1] = v31;
              v26[2] = v32;
              v26[3] = v33;
            }
            v26 += 4;
            a2 += 4;
          }
          while ( v26 != v30 );
        }
        if ( v17 )
        {
          v45 = v22;
          j_j___libc_free_0(v17, (char *)v10 - (char *)v17);
          v22 = v45;
        }
        *a1 = v22;
        a1[1] = v30;
        a1[2] = (__m128i *)v21;
        return;
      }
      if ( v20 > 0x1FFFFFFFFFFFFFFLL )
        v20 = 0x1FFFFFFFFFFFFFFLL;
      v43 = v20 << 6;
    }
    v44 = sub_22077B0(v43);
    v17 = *a1;
    v11 = a1[1];
    v10 = a1[2];
    v22 = (__m128i *)v44;
    v21 = v44 + v43;
    goto LABEL_18;
  }
  v12 = (char *)v11 - (char *)a2;
  if ( v5 < (char *)v11 - (char *)a2 )
  {
    v13 = v11;
    v14 = (__m128i *)((char *)v11 - v5);
    v15 = (__m128i *)((char *)v11 - v5);
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(v15);
        v13[1] = _mm_loadu_si128(v15 + 1);
        v13[2] = _mm_loadu_si128(v15 + 2);
        v13[3] = _mm_loadu_si128(v15 + 3);
      }
      v13 += 4;
      v15 += 4;
    }
    while ( v13 != (__m128i *)&v11->m128i_i8[v5] );
    a1[1] = (__m128i *)((char *)a1[1] + v5);
    if ( a2 != v14 )
      memmove(&a2->m128i_i8[v5], a2, (char *)v14 - (char *)a2);
    v16 = v5;
    goto LABEL_11;
  }
  v34 = (const __m128i *)((char *)a3 + v12);
  if ( a4 == (const __m128i *)&a3->m128i_i8[v12] )
  {
    v39 = v11;
  }
  else
  {
    v35 = (char *)a4 - (char *)v34;
    v36 = v11;
    v37 = v34;
    v38 = (__m128i *)((char *)v11 + v35);
    do
    {
      if ( v36 )
      {
        *v36 = _mm_loadu_si128(v37);
        v36[1] = _mm_loadu_si128(v37 + 1);
        v36[2] = _mm_loadu_si128(v37 + 2);
        v36[3] = _mm_loadu_si128(v37 + 3);
      }
      v36 += 4;
      v37 += 4;
    }
    while ( v38 != v36 );
    v39 = a1[1];
  }
  v40 = &v39[4 * (v8 - (v12 >> 6))];
  a1[1] = v40;
  if ( a2 != v11 )
  {
    v41 = a2;
    v42 = (__m128i *)((char *)v40 + (char *)v11 - (char *)a2);
    do
    {
      if ( v40 )
      {
        *v40 = _mm_loadu_si128(v41);
        v40[1] = _mm_loadu_si128(v41 + 1);
        v40[2] = _mm_loadu_si128(v41 + 2);
        v40[3] = _mm_loadu_si128(v41 + 3);
      }
      v40 += 4;
      v41 += 4;
    }
    while ( v42 != v40 );
    v40 = a1[1];
  }
  a1[1] = (__m128i *)((char *)v40 + v12);
  if ( v6 != v34 )
  {
    v16 = v12;
LABEL_11:
    memmove(a2, v6, v16);
  }
}
