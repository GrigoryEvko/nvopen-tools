// Function: sub_31C9090
// Address: 0x31c9090
//
unsigned __int64 __fastcall sub_31C9090(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // rsi
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  unsigned __int64 v12; // rbx
  __m128i *v13; // r15
  __m128i *v14; // rax
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  __m128i v17; // xmm6
  __m128i v18; // xmm7
  __m128i *v19; // rdx
  const __m128i *v20; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  const __m128i *v24; // [rsp+0h] [rbp-50h]
  size_t v25; // [rsp+10h] [rbp-40h]
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 3);
  if ( v7 == 0x1C71C71C71C71C7LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x71C71C71C71C71C7LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 3);
  v11 = &a2->m128i_i8[-v6];
  if ( v9 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v26 = 0;
      v12 = 72;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x1C71C71C71C71C7LL )
      v10 = 0x1C71C71C71C71C7LL;
    v22 = 72 * v10;
  }
  v24 = a3;
  v23 = sub_22077B0(v22);
  v11 = &a2->m128i_i8[-v6];
  v13 = (__m128i *)v23;
  a3 = v24;
  v26 = v22 + v23;
  v12 = v23 + 72;
LABEL_7:
  v14 = (__m128i *)&v11[(_QWORD)v13];
  if ( &v11[(_QWORD)v13] )
  {
    v15 = _mm_loadu_si128(a3);
    v16 = _mm_loadu_si128(a3 + 1);
    v17 = _mm_loadu_si128(a3 + 2);
    v18 = _mm_loadu_si128(a3 + 3);
    v14[4].m128i_i64[0] = a3[4].m128i_i64[0];
    *v14 = v15;
    v14[1] = v16;
    v14[2] = v17;
    v14[3] = v18;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v19 = v13;
    v20 = (const __m128i *)v6;
    do
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v20);
        v19[1] = _mm_loadu_si128(v20 + 1);
        v19[2] = _mm_loadu_si128(v20 + 2);
        v19[3] = _mm_loadu_si128(v20 + 3);
        v19[4].m128i_i64[0] = v20[4].m128i_i64[0];
      }
      v20 = (const __m128i *)((char *)v20 + 72);
      v19 = (__m128i *)((char *)v19 + 72);
    }
    while ( a2 != v20 );
    v12 = (unsigned __int64)&v13[9] + 8 * (((unsigned __int64)&a2[-5].m128i_u64[1] - v6) >> 3);
  }
  if ( a2 != v5 )
  {
    v25 = 8 * ((unsigned __int64)((char *)v5 - (char *)a2 - 72) >> 3) + 72;
    memcpy((void *)v12, a2, v25);
    v12 += v25;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  *a1 = (unsigned __int64)v13;
  a1[1] = v12;
  a1[2] = v26;
  return v26;
}
