// Function: sub_1292220
// Address: 0x1292220
//
__int64 __fastcall sub_1292220(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // r15
  const __m128i *v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rdx
  __int64 m128i_i64; // rbx
  __m128i *v14; // rcx
  __m128i *v15; // rax
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  void *v20; // rdi
  size_t v21; // rdx
  __int64 v23; // rbx
  __int64 v24; // rax
  __m128i *v25; // [rsp+10h] [rbp-40h]
  __m128i *v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v5 - *a1);
  if ( v7 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * (a1[1] - *a1);
  v10 = __CFADD__(v8, v7);
  v11 = v8 - 0x5555555555555555LL * (a1[1] - *a1);
  v12 = (char *)((char *)a2 - (char *)v6);
  if ( v10 )
  {
    v23 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v27 = 0;
      m128i_i64 = 48;
      v14 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x2AAAAAAAAAAAAAALL )
      v11 = 0x2AAAAAAAAAAAAAALL;
    v23 = 48 * v11;
  }
  v24 = sub_22077B0(v23);
  v12 = (char *)((char *)a2 - (char *)v6);
  v14 = (__m128i *)v24;
  v27 = v23 + v24;
  m128i_i64 = v24 + 48;
LABEL_7:
  v15 = (__m128i *)&v12[(_QWORD)v14];
  if ( &v12[(_QWORD)v14] )
  {
    v16 = _mm_loadu_si128(a3 + 1);
    v17 = _mm_loadu_si128(a3 + 2);
    *v15 = _mm_loadu_si128(a3);
    v15[1] = v16;
    v15[2] = v17;
  }
  if ( a2 != v6 )
  {
    v18 = v14;
    v19 = v6;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v19);
        v18[1] = _mm_loadu_si128(v19 + 1);
        v18[2] = _mm_loadu_si128(v19 + 2);
      }
      v19 += 3;
      v18 += 3;
    }
    while ( v19 != a2 );
    m128i_i64 = (__int64)v14[3
                           * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&a2[-3] - (char *)v6) >> 4))
                            & 0xFFFFFFFFFFFFFFFLL)
                           + 6].m128i_i64;
  }
  if ( a2 != v5 )
  {
    v20 = (void *)m128i_i64;
    v25 = v14;
    v21 = 16
        * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v5 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
         + 3);
    m128i_i64 += v21;
    memcpy(v20, a2, v21);
    v14 = v25;
  }
  if ( v6 )
  {
    v26 = v14;
    j_j___libc_free_0(v6, (char *)a1[2] - (char *)v6);
    v14 = v26;
  }
  *a1 = v14;
  a1[1] = (const __m128i *)m128i_i64;
  a1[2] = (const __m128i *)v27;
  return v27;
}
