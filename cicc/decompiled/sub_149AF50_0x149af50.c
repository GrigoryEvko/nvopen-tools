// Function: sub_149AF50
// Address: 0x149af50
//
void __fastcall sub_149AF50(const __m128i **a1, __m128i *a2, const __m128i *a3, const __m128i *a4)
{
  size_t v4; // r15
  unsigned __int64 v9; // rcx
  const __m128i *v10; // rsi
  __m128i *v11; // rdi
  __int64 v12; // r8
  __m128i *v13; // rdx
  __m128i *v14; // rsi
  const __m128i *v15; // rax
  size_t v16; // rdx
  const __m128i *v17; // r9
  unsigned __int64 v18; // rax
  bool v19; // cf
  unsigned __int64 v20; // rax
  const __m128i *v21; // r8
  __m128i *v22; // r15
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  unsigned __int64 v25; // r10
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  unsigned __int64 v30; // rbx
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  __m128i v33; // xmm0
  __m128i v34; // xmm1
  const __m128i *v35; // rsi
  __m128i *v36; // rdx
  const __m128i *v37; // rax
  __m128i *v38; // r9
  __m128i *v39; // rax
  const __m128i *v40; // rdx
  __int64 v41; // r8
  __int64 v42; // rax
  const __m128i *v43; // [rsp-40h] [rbp-40h]
  __int64 v44; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v4 = (char *)a4 - (char *)a3;
  v9 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a4 - (char *)a3) >> 3);
  v10 = a1[2];
  v11 = (__m128i *)a1[1];
  if ( (char *)v10 - (char *)v11 < v4 )
  {
    v17 = *a1;
    v18 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v11 - (char *)*a1) >> 3);
    if ( v9 > 0x333333333333333LL - v18 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v9 < v18 )
      v9 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v11 - (char *)*a1) >> 3);
    v19 = __CFADD__(v9, v18);
    v20 = v9 - 0x3333333333333333LL * (((char *)v11 - (char *)*a1) >> 3);
    if ( v19 )
    {
      v41 = 0x7FFFFFFFFFFFFFF8LL;
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
          v25 = (unsigned __int64)v22;
        }
        else
        {
          v23 = v22;
          v24 = v17;
          do
          {
            if ( v23 )
            {
              *v23 = _mm_loadu_si128(v24);
              v23[1] = _mm_loadu_si128(v24 + 1);
              v23[2].m128i_i64[0] = v24[2].m128i_i64[0];
            }
            v24 = (const __m128i *)((char *)v24 + 40);
            v23 = (__m128i *)((char *)v23 + 40);
          }
          while ( a2 != v24 );
          v25 = (unsigned __int64)&v22[2].m128i_u64[((unsigned __int64)((char *)&a2[-3].m128i_u64[1] - (char *)v17) >> 3)
                                                  + 1];
        }
        v26 = (__m128i *)v25;
        v27 = a3;
        do
        {
          if ( v26 )
          {
            v28 = _mm_loadu_si128(v27);
            v29 = _mm_loadu_si128(v27 + 1);
            v26[2].m128i_i64[0] = v27[2].m128i_i64[0];
            *v26 = v28;
            v26[1] = v29;
          }
          v27 = (const __m128i *)((char *)v27 + 40);
          v26 = (__m128i *)((char *)v26 + 40);
        }
        while ( a4 != v27 );
        v30 = v25 + 8 * ((unsigned __int64)((char *)&a4[-3].m128i_u64[1] - (char *)a3) >> 3) + 40;
        if ( a2 != v11 )
        {
          v31 = (__m128i *)v30;
          v32 = a2;
          do
          {
            if ( v31 )
            {
              v33 = _mm_loadu_si128(v32);
              v34 = _mm_loadu_si128(v32 + 1);
              v31[2].m128i_i64[0] = v32[2].m128i_i64[0];
              *v31 = v33;
              v31[1] = v34;
            }
            v32 = (const __m128i *)((char *)v32 + 40);
            v31 = (__m128i *)((char *)v31 + 40);
          }
          while ( v32 != v11 );
          v30 += 8 * ((unsigned __int64)((char *)v32 - (char *)a2 - 40) >> 3) + 40;
        }
        if ( v17 )
        {
          v43 = v21;
          j_j___libc_free_0(v17, (char *)v10 - (char *)v17);
          v21 = v43;
        }
        *a1 = v22;
        a1[1] = (const __m128i *)v30;
        a1[2] = v21;
        return;
      }
      if ( v20 > 0x333333333333333LL )
        v20 = 0x333333333333333LL;
      v41 = 40 * v20;
    }
    v44 = v41;
    v42 = sub_22077B0(v41);
    v17 = *a1;
    v11 = (__m128i *)a1[1];
    v10 = a1[2];
    v22 = (__m128i *)v42;
    v21 = (const __m128i *)(v42 + v44);
    goto LABEL_18;
  }
  v12 = (char *)v11 - (char *)a2;
  if ( v4 < (char *)v11 - (char *)a2 )
  {
    v13 = v11;
    v14 = (__m128i *)((char *)v11 - v4);
    v15 = (__m128i *)((char *)v11 - v4);
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(v15);
        v13[1] = _mm_loadu_si128(v15 + 1);
        v13[2].m128i_i64[0] = v15[2].m128i_i64[0];
      }
      v15 = (const __m128i *)((char *)v15 + 40);
      v13 = (__m128i *)((char *)v13 + 40);
    }
    while ( v11 != v15 );
    a1[1] = (const __m128i *)((char *)a1[1] + v4);
    if ( a2 != v14 )
      memmove(&a2->m128i_i8[v4], a2, (char *)v14 - (char *)a2);
    v16 = v4;
    goto LABEL_11;
  }
  v35 = (const __m128i *)((char *)a3 + v12);
  if ( a4 == (const __m128i *)&a3->m128i_i8[v12] )
  {
    v38 = v11;
  }
  else
  {
    v36 = v11;
    v37 = (const __m128i *)((char *)a3 + v12);
    do
    {
      if ( v36 )
      {
        *v36 = _mm_loadu_si128(v37);
        v36[1] = _mm_loadu_si128(v37 + 1);
        v36[2].m128i_i64[0] = v37[2].m128i_i64[0];
      }
      v37 = (const __m128i *)((char *)v37 + 40);
      v36 = (__m128i *)((char *)v36 + 40);
    }
    while ( a4 != v37 );
    v38 = (__m128i *)a1[1];
  }
  v39 = (__m128i *)((char *)v38 + 40 * (v9 - 0xCCCCCCCCCCCCCCCDLL * (v12 >> 3)));
  a1[1] = v39;
  if ( a2 != v11 )
  {
    v40 = a2;
    do
    {
      if ( v39 )
      {
        *v39 = _mm_loadu_si128(v40);
        v39[1] = _mm_loadu_si128(v40 + 1);
        v39[2].m128i_i64[0] = v40[2].m128i_i64[0];
      }
      v40 = (const __m128i *)((char *)v40 + 40);
      v39 = (__m128i *)((char *)v39 + 40);
    }
    while ( v11 != v40 );
    v39 = (__m128i *)a1[1];
  }
  a1[1] = (__m128i *)((char *)v39 + v12);
  if ( a3 != v35 )
  {
    v16 = (char *)v11 - (char *)a2;
LABEL_11:
    memmove(a2, a3, v16);
  }
}
