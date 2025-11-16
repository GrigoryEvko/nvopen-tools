// Function: sub_2352B10
// Address: 0x2352b10
//
unsigned __int64 *__fastcall sub_2352B10(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int8 *v8; // r12
  __int64 m128i_i64; // r13
  __m128i *v10; // rax
  __int64 v11; // rdx
  __m128i v12; // xmm2
  __int64 v13; // rdx
  __int64 v14; // rdx
  const __m128i *v15; // rbx
  __m128i *i; // r13
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned __int64 v19; // r15
  unsigned __int64 *v20; // rdi
  const __m128i *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __m128i v24; // xmm0
  __int64 v25; // rcx
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  const __m128i *v32; // [rsp+28h] [rbp-38h]

  v32 = (const __m128i *)a1[1];
  v30 = *a1;
  v4 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v32->m128i_i64 - *a1) >> 3);
  if ( v4 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v32->m128i_i64 - *a1) >> 3);
  v6 = __CFADD__(v5, v4);
  v7 = v5 - 0x3333333333333333LL * ((__int64)((__int64)v32->m128i_i64 - *a1) >> 3);
  v8 = &a2->m128i_i8[-v30];
  if ( v6 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v7 )
    {
      v28 = 0;
      m128i_i64 = 40;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0x333333333333333LL )
      v7 = 0x333333333333333LL;
    v27 = 40 * v7;
  }
  v31 = sub_22077B0(v27);
  v28 = v31 + v27;
  m128i_i64 = v31 + 40;
LABEL_7:
  v10 = (__m128i *)&v8[v31];
  if ( &v8[v31] )
  {
    v11 = a3[1].m128i_i64[0];
    v12 = _mm_loadu_si128(a3);
    a3[1].m128i_i64[0] = 0;
    v10[1].m128i_i64[0] = v11;
    v13 = a3[1].m128i_i64[1];
    a3[1].m128i_i64[1] = 0;
    v10[1].m128i_i64[1] = v13;
    v14 = a3[2].m128i_i64[0];
    a3[2].m128i_i64[0] = 0;
    v10[2].m128i_i64[0] = v14;
    *v10 = v12;
  }
  v15 = (const __m128i *)v30;
  if ( a2 != (const __m128i *)v30 )
  {
    for ( i = (__m128i *)v31; !i; i = (__m128i *)v17 )
    {
      v18 = v15[1].m128i_i64[1];
      v19 = v15[1].m128i_u64[0];
      if ( v18 != v19 )
      {
        do
        {
          v20 = (unsigned __int64 *)(v19 + 16);
          v19 += 40LL;
          sub_234A6B0(v20);
        }
        while ( v18 != v19 );
        v19 = v15[1].m128i_u64[0];
      }
      if ( !v19 )
        goto LABEL_12;
      v15 = (const __m128i *)((char *)v15 + 40);
      j_j___libc_free_0(v19);
      v17 = 40;
      if ( v15 == a2 )
      {
LABEL_20:
        m128i_i64 = (__int64)i[5].m128i_i64;
        goto LABEL_21;
      }
LABEL_13:
      ;
    }
    *i = _mm_loadu_si128(v15);
    i[1].m128i_i64[0] = v15[1].m128i_i64[0];
    i[1].m128i_i64[1] = v15[1].m128i_i64[1];
    i[2].m128i_i64[0] = v15[2].m128i_i64[0];
    v15[2].m128i_i64[0] = 0;
    v15[1].m128i_i64[1] = 0;
    v15[1].m128i_i64[0] = 0;
LABEL_12:
    v15 = (const __m128i *)((char *)v15 + 40);
    v17 = (__int64)&i[2].m128i_i64[1];
    if ( v15 == a2 )
      goto LABEL_20;
    goto LABEL_13;
  }
LABEL_21:
  if ( a2 != v32 )
  {
    v21 = a2;
    v22 = m128i_i64;
    do
    {
      v23 = v21[1].m128i_i64[0];
      v24 = _mm_loadu_si128(v21);
      v22 += 40;
      v21 = (const __m128i *)((char *)v21 + 40);
      *(_QWORD *)(v22 - 24) = v23;
      v25 = v21[-1].m128i_i64[0];
      *(__m128i *)(v22 - 40) = v24;
      *(_QWORD *)(v22 - 16) = v25;
      *(_QWORD *)(v22 - 8) = v21[-1].m128i_i64[1];
    }
    while ( v21 != v32 );
    m128i_i64 += 8 * ((unsigned __int64)((char *)v21 - (char *)a2 - 40) >> 3) + 40;
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  *a1 = v31;
  a1[1] = m128i_i64;
  a1[2] = v28;
  return a1;
}
