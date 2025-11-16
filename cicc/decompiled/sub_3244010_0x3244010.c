// Function: sub_3244010
// Address: 0x3244010
//
unsigned __int64 __fastcall sub_3244010(unsigned __int64 *a1, const __m128i *a2, int *a3, _BYTE *a4)
{
  const __m128i *v5; // rcx
  unsigned __int64 v6; // r14
  __int64 v7; // rax
  bool v10; // zf
  __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  __int8 *v16; // rdx
  __int64 v17; // rbx
  __m128i *v18; // r15
  char *v19; // rdx
  char v20; // al
  int v21; // esi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  _BYTE *v27; // [rsp+8h] [rbp-58h]
  signed __int64 v28; // [rsp+20h] [rbp-40h]
  const __m128i *v29; // [rsp+20h] [rbp-40h]
  unsigned __int64 v30; // [rsp+28h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = (__int64)((__int64)v5->m128i_i64 - *a1) >> 4;
  if ( v7 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = v7 == 0;
  v12 = (__int64)(a1[1] - *a1) >> 4;
  v13 = 1;
  if ( !v10 )
    v13 = v12;
  v14 = __CFADD__(v12, v13);
  v15 = v12 + v13;
  v16 = &a2->m128i_i8[-v6];
  if ( v14 )
  {
    v25 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v15 )
    {
      v30 = 0;
      v17 = 16;
      v18 = 0;
      goto LABEL_7;
    }
    if ( v15 > 0x7FFFFFFFFFFFFFFLL )
      v15 = 0x7FFFFFFFFFFFFFFLL;
    v25 = 16 * v15;
  }
  v27 = a4;
  v29 = v5;
  v26 = sub_22077B0(v25);
  v5 = v29;
  v16 = &a2->m128i_i8[-v6];
  v18 = (__m128i *)v26;
  a4 = v27;
  v30 = v25 + v26;
  v17 = v26 + 16;
LABEL_7:
  v19 = &v16[(_QWORD)v18];
  if ( v19 )
  {
    v20 = *a4;
    v21 = *a3;
    *((_QWORD *)v19 + 1) = 0;
    *(_DWORD *)v19 = v21;
    v19[4] = v20;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v22 = v18;
    v23 = (const __m128i *)v6;
    do
    {
      if ( v22 )
        *v22 = _mm_loadu_si128(v23);
      ++v23;
      ++v22;
    }
    while ( v23 != a2 );
    v17 = (__int64)a2[1].m128i_i64 + (_QWORD)v18 - v6;
  }
  if ( a2 != v5 )
  {
    v28 = (char *)v5 - (char *)a2;
    memcpy((void *)v17, a2, (char *)v5 - (char *)a2);
    v17 += v28;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  *a1 = (unsigned __int64)v18;
  a1[1] = v17;
  a1[2] = v30;
  return v30;
}
