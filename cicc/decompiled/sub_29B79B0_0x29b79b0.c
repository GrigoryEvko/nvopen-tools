// Function: sub_29B79B0
// Address: 0x29b79b0
//
unsigned __int64 __fastcall sub_29B79B0(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r15
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  __int64 v8; // rcx
  bool v9; // zf
  __int64 v10; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int8 *v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // rbx
  __m128i *v17; // rdx
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  void *v20; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+10h] [rbp-40h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h]
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]

  v3 = (const __m128i *)a1[1];
  v4 = *a1;
  v5 = (__int64)((__int64)v3->m128i_i64 - *a1) >> 4;
  if ( v5 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = (__int64)(a1[1] - *a1) >> 4;
  v9 = v5 == 0;
  v10 = 1;
  if ( !v9 )
    v10 = (__int64)(a1[1] - *a1) >> 4;
  v12 = __CFADD__(v8, v10);
  v13 = v8 + v10;
  v14 = &a2->m128i_i8[-v4];
  v15 = v12;
  if ( v12 )
  {
    v22 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v13 )
    {
      v26 = 0;
      v16 = 16;
      goto LABEL_7;
    }
    if ( v13 > 0x7FFFFFFFFFFFFFFLL )
      v13 = 0x7FFFFFFFFFFFFFFLL;
    v22 = 16 * v13;
  }
  v23 = sub_22077B0(v22);
  v14 = &a2->m128i_i8[-v4];
  v15 = v23;
  v26 = v22 + v23;
  v16 = v23 + 16;
LABEL_7:
  v17 = (__m128i *)&v14[v15];
  if ( v17 )
    *v17 = _mm_loadu_si128(a3);
  if ( a2 != (const __m128i *)v4 )
  {
    v18 = (__m128i *)v15;
    v19 = (const __m128i *)v4;
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v19);
      ++v19;
      ++v18;
    }
    while ( v19 != a2 );
    v16 = (__int64)a2[1].m128i_i64 + v15 - v4;
  }
  if ( a2 != v3 )
  {
    v20 = (void *)v16;
    v24 = v15;
    v16 += (char *)v3 - (char *)a2;
    memcpy(v20, a2, (char *)v3 - (char *)a2);
    v15 = v24;
  }
  if ( v4 )
  {
    v25 = v15;
    j_j___libc_free_0(v4);
    v15 = v25;
  }
  *a1 = v15;
  a1[1] = v16;
  a1[2] = v26;
  return v26;
}
