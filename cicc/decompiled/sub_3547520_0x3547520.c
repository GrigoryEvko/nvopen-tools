// Function: sub_3547520
// Address: 0x3547520
//
void __fastcall sub_3547520(const __m128i **a1, unsigned __int64 a2)
{
  unsigned __int64 v4; // rdi
  const __m128i *v5; // rsi
  const __m128i *v6; // rdx
  __int64 v7; // rbx
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rdx
  _DWORD *v10; // rax
  __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r15
  __m128i *v15; // r8
  _DWORD *v16; // rax
  unsigned __int64 v17; // rcx
  __m128i *v18; // rax
  __m128i *v19; // rdi
  unsigned __int64 v20; // r15
  __int64 v21; // rax
  __m128i *v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = (unsigned __int64)a1[1];
  v5 = *a1;
  v6 = *a1;
  v7 = v4 - (_QWORD)*a1;
  v8 = v7 >> 4;
  if ( a2 <= (__int64)((__int64)a1[2]->m128i_i64 - v4) >> 4 )
  {
    v9 = a2;
    v10 = (_DWORD *)v4;
    do
    {
      if ( v10 )
      {
        *v10 = 0;
        v10[1] = 0;
        v10[2] = 0;
        v10[3] = 0;
      }
      v10 += 4;
      --v9;
    }
    while ( v9 );
    a1[1] = (const __m128i *)(v4 + 16 * a2);
    return;
  }
  if ( 0x7FFFFFFFFFFFFFFLL - v8 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v11 = (__int64)(v4 - (_QWORD)*a1) >> 4;
  if ( a2 >= v8 )
    v11 = a2;
  v12 = __CFADD__(v8, v11);
  v13 = v8 + v11;
  if ( v12 )
  {
    v20 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v13 )
    {
      v14 = 0;
      v15 = 0;
      goto LABEL_15;
    }
    if ( v13 > 0x7FFFFFFFFFFFFFFLL )
      v13 = 0x7FFFFFFFFFFFFFFLL;
    v20 = 16 * v13;
  }
  v21 = sub_22077B0(v20);
  v5 = *a1;
  v4 = (unsigned __int64)a1[1];
  v15 = (__m128i *)v21;
  v14 = v21 + v20;
  v6 = *a1;
LABEL_15:
  v16 = (__int32 *)((char *)v15->m128i_i32 + v7);
  v17 = a2;
  do
  {
    if ( v16 )
    {
      *v16 = 0;
      v16[1] = 0;
      v16[2] = 0;
      v16[3] = 0;
    }
    v16 += 4;
    --v17;
  }
  while ( v17 );
  if ( v5 != (const __m128i *)v4 )
  {
    v18 = v15;
    v19 = (__m128i *)((char *)v15 + v4 - (_QWORD)v5);
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v6);
      ++v18;
      ++v6;
    }
    while ( v18 != v19 );
    v4 = (unsigned __int64)v5;
  }
  if ( v4 )
  {
    v22 = v15;
    j_j___libc_free_0(v4);
    v15 = v22;
  }
  *a1 = v15;
  a1[2] = (const __m128i *)v14;
  a1[1] = &v15[a2 + v8];
}
