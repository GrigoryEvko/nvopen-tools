// Function: sub_1DA0880
// Address: 0x1da0880
//
const __m128i **__fastcall sub_1DA0880(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r14
  const __m128i *v5; // rbx
  __int64 v6; // rax
  const __m128i *v7; // r9
  bool v8; // zf
  __int64 v9; // rcx
  __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  signed __int64 v13; // r10
  __int64 m128i_i64; // r15
  __m128i *v15; // r10
  __m128i *v16; // r13
  __m128i v17; // xmm2
  __int64 v18; // rsi
  const __m128i *v19; // r13
  __m128i *i; // r15
  __int64 v21; // rsi
  __m128i v22; // xmm1
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  _QWORD *v27; // rdi
  const __m128i *j; // r12
  unsigned __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v32; // r15
  __int64 v33; // rax
  const __m128i *v35; // [rsp+8h] [rbp-58h]
  const __m128i *v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __m128i *v39; // [rsp+20h] [rbp-40h]
  const __m128i *v40; // [rsp+28h] [rbp-38h]

  v3 = a2;
  v5 = a1[1];
  v40 = *a1;
  v6 = ((char *)v5 - (char *)*a1) >> 7;
  if ( v6 == 0xFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = a3;
  v8 = v6 == 0;
  v9 = ((char *)v5 - (char *)*a1) >> 7;
  v10 = 1;
  if ( !v8 )
    v10 = ((char *)v5 - (char *)*a1) >> 7;
  v11 = __CFADD__(v9, v10);
  v12 = v9 + v10;
  v13 = (char *)a2 - (char *)v40;
  if ( v11 )
  {
    v32 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v12 )
    {
      v37 = 0;
      m128i_i64 = 128;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0xFFFFFFFFFFFFFFLL )
      v12 = 0xFFFFFFFFFFFFFFLL;
    v32 = v12 << 7;
  }
  v33 = sub_22077B0(v32);
  v13 = (char *)a2 - (char *)v40;
  v7 = a3;
  v39 = (__m128i *)v33;
  v37 = v33 + v32;
  m128i_i64 = v33 + 128;
LABEL_7:
  v8 = &v39->m128i_i8[v13] == 0;
  v15 = (__m128i *)((char *)v39 + v13);
  v16 = v15;
  if ( !v8 )
  {
    v17 = _mm_loadu_si128(v7);
    v18 = v7[1].m128i_i64[1];
    v15[1].m128i_i64[0] = v7[1].m128i_i64[0];
    v15[1].m128i_i64[1] = v18;
    *v15 = v17;
    if ( v18 )
    {
      v35 = v7;
      sub_1623A60((__int64)&v15[1].m128i_i64[1], v18, 2);
      v7 = v35;
    }
    v36 = v7;
    v16[2].m128i_i64[0] = v7[2].m128i_i64[0];
    sub_16CCCB0(&v16[2].m128i_i64[1], (__int64)v16[5].m128i_i64, (__int64)&v7[2].m128i_i64[1]);
    v16[7].m128i_i32[0] = v36[7].m128i_i32[0];
    v16[7].m128i_i64[1] = v36[7].m128i_i64[1];
  }
  v19 = v40;
  if ( a2 != v40 )
  {
    for ( i = v39; ; i += 8 )
    {
      if ( i )
      {
        *i = _mm_loadu_si128(v19);
        i[1].m128i_i64[0] = v19[1].m128i_i64[0];
        v21 = v19[1].m128i_i64[1];
        i[1].m128i_i64[1] = v21;
        if ( v21 )
          sub_1623A60((__int64)&i[1].m128i_i64[1], v21, 2);
        i[2].m128i_i64[0] = v19[2].m128i_i64[0];
        sub_16CCCB0(&i[2].m128i_i64[1], (__int64)i[5].m128i_i64, (__int64)&v19[2].m128i_i64[1]);
        i[7].m128i_i32[0] = v19[7].m128i_i32[0];
        i[7].m128i_i64[1] = v19[7].m128i_i64[1];
      }
      v19 += 8;
      if ( a2 == v19 )
        break;
    }
    m128i_i64 = (__int64)i[16].m128i_i64;
  }
  if ( a2 != v5 )
  {
    do
    {
      v22 = _mm_loadu_si128(v3);
      v23 = v3[1].m128i_i64[1];
      *(_QWORD *)(m128i_i64 + 16) = v3[1].m128i_i64[0];
      *(_QWORD *)(m128i_i64 + 24) = v23;
      *(__m128i *)m128i_i64 = v22;
      if ( v23 )
        sub_1623A60(m128i_i64 + 24, v23, 2);
      v24 = v3[2].m128i_i64[0];
      v25 = (__int64)&v3[2].m128i_i64[1];
      v26 = m128i_i64 + 80;
      v3 += 8;
      v27 = (_QWORD *)(m128i_i64 + 40);
      m128i_i64 += 128;
      *(_QWORD *)(m128i_i64 - 96) = v24;
      sub_16CCCB0(v27, v26, v25);
      *(_DWORD *)(m128i_i64 - 16) = v3[-1].m128i_i32[0];
      *(_QWORD *)(m128i_i64 - 8) = v3[-1].m128i_i64[1];
    }
    while ( v5 != v3 );
  }
  for ( j = v40; j != v5; j += 8 )
  {
    v29 = j[3].m128i_u64[1];
    if ( v29 != j[3].m128i_i64[0] )
      _libc_free(v29);
    v30 = j[1].m128i_i64[1];
    if ( v30 )
      sub_161E7C0((__int64)&j[1].m128i_i64[1], v30);
  }
  if ( v40 )
    j_j___libc_free_0(v40, (char *)a1[2] - (char *)v40);
  *a1 = v39;
  a1[1] = (const __m128i *)m128i_i64;
  a1[2] = (const __m128i *)v37;
  return a1;
}
