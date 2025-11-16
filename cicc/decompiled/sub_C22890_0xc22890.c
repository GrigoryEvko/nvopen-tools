// Function: sub_C22890
// Address: 0xc22890
//
__int64 __fastcall sub_C22890(const __m128i **a1, const __m128i *a2, const __m128i *a3, __int64 *a4)
{
  const __m128i *v7; // r15
  const __m128i *v8; // r14
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rbx
  __m128i *v16; // rax
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  __int64 v19; // rdx
  __int64 v20; // rdx
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  void *v23; // rdi
  size_t v24; // rdx
  __int64 v26; // rbx
  __int64 v27; // rax
  const __m128i *v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v7 = a1[1];
  v8 = *a1;
  v9 = 0xAAAAAAAAAAAAAAABLL * (v7 - *a1);
  if ( v9 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xAAAAAAAAAAAAAAABLL * (v7 - v8);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x5555555555555555LL * (v7 - v8);
  v13 = (char *)((char *)a2 - (char *)v8);
  v14 = v11;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v12 )
    {
      v31 = 0;
      v15 = 48;
      goto LABEL_7;
    }
    if ( v12 > 0x2AAAAAAAAAAAAAALL )
      v12 = 0x2AAAAAAAAAAAAAALL;
    v26 = 48 * v12;
  }
  v28 = a3;
  v27 = sub_22077B0(v26);
  v13 = (char *)((char *)a2 - (char *)v8);
  a3 = v28;
  v14 = v27;
  v31 = v26 + v27;
  v15 = v27 + 48;
LABEL_7:
  v16 = (__m128i *)&v13[v14];
  if ( &v13[v14] )
  {
    v17 = _mm_loadu_si128(a3);
    v18 = _mm_loadu_si128(a3 + 1);
    v19 = a3[2].m128i_i64[0];
    *v16 = v17;
    v16[2].m128i_i64[0] = v19;
    v20 = *a4;
    v16[1] = v18;
    v16[2].m128i_i64[1] = v20;
  }
  if ( a2 != v8 )
  {
    v21 = (__m128i *)v14;
    v22 = v8;
    do
    {
      if ( v21 )
      {
        *v21 = _mm_loadu_si128(v22);
        v21[1] = _mm_loadu_si128(v22 + 1);
        v21[2] = _mm_loadu_si128(v22 + 2);
      }
      v22 += 3;
      v21 += 3;
    }
    while ( v22 != a2 );
    v15 = v14
        + 16
        * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&a2[-3] - (char *)v8) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
         + 6);
  }
  if ( a2 != v7 )
  {
    v23 = (void *)v15;
    v29 = v14;
    v24 = 16
        * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v7 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
         + 3);
    v15 += v24;
    memcpy(v23, a2, v24);
    v14 = v29;
  }
  if ( v8 )
  {
    v30 = v14;
    j_j___libc_free_0(v8, (char *)a1[2] - (char *)v8);
    v14 = v30;
  }
  a1[1] = (const __m128i *)v15;
  *a1 = (const __m128i *)v14;
  a1[2] = (const __m128i *)v31;
  return v31;
}
