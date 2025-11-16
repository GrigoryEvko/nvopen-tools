// Function: sub_18BBFB0
// Address: 0x18bbfb0
//
__int64 __fastcall sub_18BBFB0(const __m128i **a1, const __m128i *a2, const __m128i *a3)
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
  __m128i v18; // xmm3
  __m128i *v19; // rdx
  const __m128i *v20; // rax
  void *v21; // rdi
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = ((char *)v3 - (char *)*a1) >> 5;
  if ( v5 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = ((char *)a1[1] - (char *)*a1) >> 5;
  v9 = v5 == 0;
  v10 = 1;
  if ( !v9 )
    v10 = ((char *)a1[1] - (char *)*a1) >> 5;
  v12 = __CFADD__(v8, v10);
  v13 = v8 + v10;
  v14 = (char *)((char *)a2 - (char *)v4);
  v15 = v12;
  if ( v12 )
  {
    v23 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v13 )
    {
      v27 = 0;
      v16 = 32;
      goto LABEL_7;
    }
    if ( v13 > 0x3FFFFFFFFFFFFFFLL )
      v13 = 0x3FFFFFFFFFFFFFFLL;
    v23 = 32 * v13;
  }
  v24 = sub_22077B0(v23);
  v14 = (char *)((char *)a2 - (char *)v4);
  v15 = v24;
  v27 = v23 + v24;
  v16 = v24 + 32;
LABEL_7:
  v17 = (__m128i *)&v14[v15];
  if ( v17 )
  {
    v18 = _mm_loadu_si128(a3 + 1);
    *v17 = _mm_loadu_si128(a3);
    v17[1] = v18;
  }
  if ( a2 != v4 )
  {
    v19 = (__m128i *)v15;
    v20 = v4;
    do
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v20);
        v19[1] = _mm_loadu_si128(v20 + 1);
      }
      v20 += 2;
      v19 += 2;
    }
    while ( v20 != a2 );
    v16 = v15 + (char *)a2 - (char *)v4 + 32;
  }
  if ( a2 != v3 )
  {
    v21 = (void *)v16;
    v25 = v15;
    v16 += (char *)v3 - (char *)a2;
    memcpy(v21, a2, (char *)v3 - (char *)a2);
    v15 = v25;
  }
  if ( v4 )
  {
    v26 = v15;
    j_j___libc_free_0(v4, (char *)a1[2] - (char *)v4);
    v15 = v26;
  }
  *a1 = (const __m128i *)v15;
  a1[1] = (const __m128i *)v16;
  a1[2] = (const __m128i *)v27;
  return v27;
}
