// Function: sub_20EB6D0
// Address: 0x20eb6d0
//
const __m128i **__fastcall sub_20EB6D0(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // rbx
  const __m128i *v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  const __m128i *v8; // r15
  bool v9; // cf
  unsigned __int64 v10; // rax
  signed __int64 v11; // rsi
  __int64 m128i_i64; // r12
  __int8 *v13; // rsi
  _QWORD *v14; // rdi
  __int64 v15; // rsi
  __m128i *v16; // r12
  const __m128i *v17; // rcx
  __m128i v18; // xmm0
  __int64 v19; // rdx
  __int64 v20; // rsi
  _QWORD *v21; // rdi
  const __m128i *i; // r14
  unsigned __int64 v23; // rdi
  __int64 v25; // r12
  __int64 v26; // rax
  const __m128i *v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __m128i *v30; // [rsp+20h] [rbp-40h]
  const __m128i *v31; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = 0xD37A6F4DE9BD37A7LL * (((char *)v3 - (char *)*a1) >> 3);
  if ( v5 == 0xB21642C8590B21LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  v8 = a2;
  if ( v5 )
    v6 = 0xD37A6F4DE9BD37A7LL * (((char *)v3 - (char *)v4) >> 3);
  v9 = __CFADD__(v6, v5);
  v10 = v6 - 0x2C8590B21642C859LL * (((char *)v3 - (char *)v4) >> 3);
  v11 = (char *)a2 - (char *)v4;
  if ( v9 )
  {
    v25 = 0x7FFFFFFFFFFFFFB8LL;
  }
  else
  {
    if ( !v10 )
    {
      v28 = 0;
      m128i_i64 = 184;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0xB21642C8590B21LL )
      v10 = 0xB21642C8590B21LL;
    v25 = 184 * v10;
  }
  v27 = a3;
  v26 = sub_22077B0(v25);
  a3 = v27;
  v30 = (__m128i *)v26;
  v28 = v26 + v25;
  m128i_i64 = v26 + 184;
LABEL_7:
  v13 = &v30->m128i_i8[v11];
  if ( v13 )
  {
    v14 = v13 + 16;
    v15 = (__int64)(v13 + 56);
    *(__m128i *)(v15 - 56) = _mm_loadu_si128(a3);
    sub_16CCEE0(v14, v15, 16, (__int64)a3[1].m128i_i64);
  }
  if ( a2 != v4 )
  {
    v16 = v30;
    v17 = v4;
    while ( 1 )
    {
      if ( v16 )
      {
        v31 = v17;
        *v16 = _mm_loadu_si128(v17);
        sub_16CCCB0((__m128i *)v16[1].m128i_i64, (__int64)&v16[3].m128i_i64[1], (__int64)v17[1].m128i_i64);
        v17 = v31;
      }
      v17 = (const __m128i *)((char *)v17 + 184);
      if ( a2 == v17 )
        break;
      v16 = (__m128i *)((char *)v16 + 184);
    }
    m128i_i64 = (__int64)v16[23].m128i_i64;
  }
  if ( a2 != v3 )
  {
    do
    {
      v18 = _mm_loadu_si128(v8);
      v19 = (__int64)v8[1].m128i_i64;
      v20 = m128i_i64 + 56;
      v8 = (const __m128i *)((char *)v8 + 184);
      v21 = (_QWORD *)(m128i_i64 + 16);
      m128i_i64 += 184;
      *(__m128i *)(m128i_i64 - 184) = v18;
      sub_16CCCB0(v21, v20, v19);
    }
    while ( v3 != v8 );
  }
  for ( i = v4; v3 != i; i = (const __m128i *)((char *)i + 184) )
  {
    v23 = i[2].m128i_u64[0];
    if ( v23 != i[1].m128i_i64[1] )
      _libc_free(v23);
  }
  if ( v4 )
    j_j___libc_free_0(v4, (char *)a1[2] - (char *)v4);
  *a1 = v30;
  a1[1] = (const __m128i *)m128i_i64;
  a1[2] = (const __m128i *)v28;
  return a1;
}
