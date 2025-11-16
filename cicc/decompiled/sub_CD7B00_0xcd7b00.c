// Function: sub_CD7B00
// Address: 0xcd7b00
//
void __fastcall sub_CD7B00(const __m128i **a1, unsigned __int64 a2)
{
  const __m128i *v4; // rdi
  const __m128i *v5; // rsi
  const __m128i *v6; // rdx
  __int64 v7; // rbx
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rdx
  const __m128i *v10; // rax
  unsigned __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  const __m128i *v14; // r8
  __m128i *v15; // r15
  __int8 *v16; // rax
  unsigned __int64 v17; // rcx
  __m128i *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  const __m128i *v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = *a1;
  v6 = *a1;
  v7 = (char *)v4 - (char *)*a1;
  v8 = 0x8E38E38E38E38E39LL * (v7 >> 2);
  if ( a2 <= 0x8E38E38E38E38E39LL * (((char *)a1[2] - (char *)v4) >> 2) )
  {
    v9 = a2;
    v10 = v4;
    do
    {
      if ( v10 )
      {
        v10[2].m128i_i32[0] = 0;
        *v10 = 0;
        v10[1] = 0;
      }
      v10 = (const __m128i *)((char *)v10 + 36);
      --v9;
    }
    while ( v9 );
    a1[1] = (const __m128i *)((char *)v4 + 36 * a2);
    return;
  }
  if ( 0x38E38E38E38E38ELL - v8 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v11 = 0x8E38E38E38E38E39LL * (((char *)v4 - (char *)*a1) >> 2);
  if ( a2 >= v8 )
    v11 = a2;
  v12 = __CFADD__(v8, v11);
  v13 = v8 + v11;
  if ( v12 )
  {
    v19 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v13 )
    {
      v14 = 0;
      v15 = 0;
      goto LABEL_15;
    }
    if ( v13 > 0x38E38E38E38E38ELL )
      v13 = 0x38E38E38E38E38ELL;
    v19 = 36 * v13;
  }
  v22 = v19;
  v20 = sub_22077B0(v19);
  v5 = *a1;
  v4 = a1[1];
  v15 = (__m128i *)v20;
  v6 = *a1;
  v14 = (const __m128i *)(v20 + v22);
LABEL_15:
  v16 = &v15->m128i_i8[v7];
  v17 = a2;
  do
  {
    if ( v16 )
    {
      *((_DWORD *)v16 + 8) = 0;
      *(_OWORD *)v16 = 0;
      *((_OWORD *)v16 + 1) = 0;
    }
    v16 += 36;
    --v17;
  }
  while ( v17 );
  if ( v5 != v4 )
  {
    v18 = v15;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v6);
        v18[1] = _mm_loadu_si128(v6 + 1);
        v18[2].m128i_i32[0] = v6[2].m128i_i32[0];
      }
      v6 = (const __m128i *)((char *)v6 + 36);
      v18 = (__m128i *)((char *)v18 + 36);
    }
    while ( v6 != v4 );
    v4 = v5;
  }
  if ( v4 )
  {
    v21 = v14;
    j_j___libc_free_0(v4, (char *)a1[2] - (char *)v4);
    v14 = v21;
  }
  *a1 = v15;
  a1[2] = v14;
  a1[1] = (__m128i *)((char *)v15 + 36 * v8 + 36 * a2);
}
