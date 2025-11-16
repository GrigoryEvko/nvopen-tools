// Function: sub_1FD42F0
// Address: 0x1fd42f0
//
__int64 __fastcall sub_1FD42F0(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r15
  const __m128i *v4; // r14
  __int64 v5; // rax
  __int64 v8; // rcx
  bool v9; // zf
  __int64 v10; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  char *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rbx
  __m128i *v17; // rdx
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  void *v20; // rdi
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = v3 - *a1;
  if ( v5 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = a1[1] - *a1;
  v9 = v5 == 0;
  v10 = 1;
  if ( !v9 )
    v10 = a1[1] - *a1;
  v12 = __CFADD__(v8, v10);
  v13 = v8 + v10;
  v14 = (char *)((char *)a2 - (char *)v4);
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
  v14 = (char *)((char *)a2 - (char *)v4);
  v15 = v23;
  v26 = v22 + v23;
  v16 = v23 + 16;
LABEL_7:
  v17 = (__m128i *)&v14[v15];
  if ( v17 )
    *v17 = _mm_loadu_si128(a3);
  if ( a2 != v4 )
  {
    v18 = (__m128i *)v15;
    v19 = v4;
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v19);
      ++v19;
      ++v18;
    }
    while ( v19 != a2 );
    v16 = v15 + (char *)a2 - (char *)v4 + 16;
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
    j_j___libc_free_0(v4, (char *)a1[2] - (char *)v4);
    v15 = v25;
  }
  *a1 = (const __m128i *)v15;
  a1[1] = (const __m128i *)v16;
  a1[2] = (const __m128i *)v26;
  return v26;
}
