// Function: sub_2707CD0
// Address: 0x2707cd0
//
unsigned __int64 *__fastcall sub_2707CD0(
        unsigned __int64 *a1,
        const __m128i *a2,
        const __m128i *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  const __m128i *v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // rbx
  __m128i v14; // xmm2
  const __m128i *v15; // rbx
  __m128i *i; // r13
  __m128i *v17; // rax
  __m128i v18; // xmm0
  __int64 m128i_i64; // rsi
  __m128i v20; // xmm1
  __int32 v21; // eax
  __int64 v22; // rsi
  unsigned __int64 j; // r14
  __int64 v24; // rbx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r15
  __int64 v28; // r13
  unsigned __int64 v29; // rdi
  unsigned __int64 v31; // r13
  __int64 v32; // rax
  const __m128i *v33; // [rsp+8h] [rbp-68h]
  unsigned __int64 v35; // [rsp+18h] [rbp-58h]
  unsigned __int64 v36; // [rsp+20h] [rbp-50h]
  unsigned __int64 v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h]
  const __m128i *v39; // [rsp+38h] [rbp-38h]

  v7 = a2;
  v39 = (const __m128i *)a1[1];
  v37 = *a1;
  v8 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v39->m128i_i64 - *a1) >> 3);
  if ( v8 == 0x1C71C71C71C71C7LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v39->m128i_i64 - *a1) >> 3);
  v10 = __CFADD__(v9, v8);
  v11 = v9 - 0x71C71C71C71C71C7LL * ((__int64)((__int64)v39->m128i_i64 - *a1) >> 3);
  v12 = v10;
  if ( v10 )
  {
    v31 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_43:
    v33 = a3;
    v32 = sub_22077B0(v31);
    a3 = v33;
    v36 = v32;
    v35 = v32 + v31;
    v38 = v32 + 72;
    goto LABEL_7;
  }
  if ( v11 )
  {
    if ( v11 > 0x1C71C71C71C71C7LL )
      v11 = 0x1C71C71C71C71C7LL;
    v31 = 72 * v11;
    goto LABEL_43;
  }
  v38 = 72;
  v35 = 0;
  v36 = 0;
LABEL_7:
  v13 = (__m128i *)((char *)a2 + v36 - v37);
  if ( v13 )
  {
    v14 = _mm_loadu_si128(a3);
    v12 = a3[1].m128i_u32[2];
    v13[1].m128i_i64[0] = (__int64)v13[2].m128i_i64;
    v13[1].m128i_i64[1] = 0x100000000LL;
    *v13 = v14;
    if ( (_DWORD)v12 )
      sub_2707540((__int64)v13[1].m128i_i64, (__int64)a3[1].m128i_i64, (__int64)a3, v12, a5, a6);
  }
  v15 = (const __m128i *)v37;
  if ( a2 != (const __m128i *)v37 )
  {
    for ( i = (__m128i *)v36; ; i = v17 )
    {
      if ( i
        && (v18 = _mm_loadu_si128(v15),
            i[1].m128i_i32[2] = 0,
            i[1].m128i_i64[0] = (__int64)i[2].m128i_i64,
            i[1].m128i_i32[3] = 1,
            *i = v18,
            a3 = (const __m128i *)v15[1].m128i_u32[2],
            (_DWORD)a3) )
      {
        m128i_i64 = (__int64)v15[1].m128i_i64;
        v15 = (const __m128i *)((char *)v15 + 72);
        sub_2707240((__int64)i[1].m128i_i64, m128i_i64, (__int64)a3, v12, a5, a6);
        v17 = (__m128i *)((char *)i + 72);
        if ( a2 == v15 )
        {
LABEL_17:
          v38 = (__int64)i[9].m128i_i64;
          break;
        }
      }
      else
      {
        v15 = (const __m128i *)((char *)v15 + 72);
        v17 = (__m128i *)((char *)i + 72);
        if ( a2 == v15 )
          goto LABEL_17;
      }
    }
  }
  if ( a2 != v39 )
  {
    do
    {
      while ( 1 )
      {
        v20 = _mm_loadu_si128(v7);
        *(_DWORD *)(v38 + 24) = 0;
        *(_QWORD *)(v38 + 16) = v38 + 32;
        v21 = v7[1].m128i_i32[2];
        *(_DWORD *)(v38 + 28) = 1;
        *(__m128i *)v38 = v20;
        if ( v21 )
          break;
        v38 += 72;
        v7 = (const __m128i *)((char *)v7 + 72);
        if ( v39 == v7 )
          goto LABEL_23;
      }
      v22 = (__int64)v7[1].m128i_i64;
      v7 = (const __m128i *)((char *)v7 + 72);
      sub_2707240(v38 + 16, v22, (__int64)a3, v38, a5, a6);
      v38 += 72;
    }
    while ( v39 != v7 );
  }
LABEL_23:
  for ( j = v37; (const __m128i *)j != v39; j += 72LL )
  {
    v24 = *(_QWORD *)(j + 16);
    v25 = v24 + 40LL * *(unsigned int *)(j + 24);
    if ( v24 != v25 )
    {
      do
      {
        v25 -= 40LL;
        v26 = *(_QWORD *)(v25 + 16);
        if ( v26 != v25 + 40 )
          _libc_free(v26);
        v27 = *(_QWORD *)v25;
        v28 = *(_QWORD *)v25 + 80LL * *(unsigned int *)(v25 + 8);
        if ( *(_QWORD *)v25 != v28 )
        {
          do
          {
            v28 -= 80;
            v29 = *(_QWORD *)(v28 + 8);
            if ( v29 != v28 + 24 )
              _libc_free(v29);
          }
          while ( v27 != v28 );
          v27 = *(_QWORD *)v25;
        }
        if ( v27 != v25 + 16 )
          _libc_free(v27);
      }
      while ( v24 != v25 );
      v25 = *(_QWORD *)(j + 16);
    }
    if ( v25 != j + 32 )
      _libc_free(v25);
  }
  if ( v37 )
    j_j___libc_free_0(v37);
  *a1 = v36;
  a1[1] = v38;
  a1[2] = v35;
  return a1;
}
