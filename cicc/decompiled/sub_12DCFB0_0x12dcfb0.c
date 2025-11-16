// Function: sub_12DCFB0
// Address: 0x12dcfb0
//
__int64 __fastcall sub_12DCFB0(const __m128i **a1, const __m128i *a2, __int64 a3, _DWORD *a4)
{
  const __m128i *v5; // rcx
  const __m128i *v6; // r14
  __int64 v7; // rax
  bool v10; // zf
  __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  char *v16; // rdx
  __int64 v17; // rbx
  __m128i *v18; // r15
  char *v19; // rdx
  int v20; // eax
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  _DWORD *v26; // [rsp+8h] [rbp-58h]
  signed __int64 v27; // [rsp+20h] [rbp-40h]
  const __m128i *v28; // [rsp+20h] [rbp-40h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = v5 - *a1;
  if ( v7 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = v7 == 0;
  v12 = a1[1] - *a1;
  v13 = 1;
  if ( !v10 )
    v13 = v12;
  v14 = __CFADD__(v12, v13);
  v15 = v12 + v13;
  v16 = (char *)((char *)a2 - (char *)v6);
  if ( v14 )
  {
    v24 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v15 )
    {
      v29 = 0;
      v17 = 16;
      v18 = 0;
      goto LABEL_7;
    }
    if ( v15 > 0x7FFFFFFFFFFFFFFLL )
      v15 = 0x7FFFFFFFFFFFFFFLL;
    v24 = 16 * v15;
  }
  v26 = a4;
  v28 = v5;
  v25 = sub_22077B0(v24);
  v5 = v28;
  v16 = (char *)((char *)a2 - (char *)v6);
  v18 = (__m128i *)v25;
  a4 = v26;
  v29 = v24 + v25;
  v17 = v25 + 16;
LABEL_7:
  v19 = &v16[(_QWORD)v18];
  if ( v19 )
  {
    v20 = *a4;
    *(_QWORD *)v19 = a3;
    *((_DWORD *)v19 + 2) = v20;
  }
  if ( a2 != v6 )
  {
    v21 = v18;
    v22 = v6;
    do
    {
      if ( v21 )
        *v21 = _mm_loadu_si128(v22);
      ++v22;
      ++v21;
    }
    while ( v22 != a2 );
    v17 = (__int64)v18[1].m128i_i64 + (char *)a2 - (char *)v6;
  }
  if ( a2 != v5 )
  {
    v27 = (char *)v5 - (char *)a2;
    memcpy((void *)v17, a2, (char *)v5 - (char *)a2);
    v17 += v27;
  }
  if ( v6 )
    j_j___libc_free_0(v6, (char *)a1[2] - (char *)v6);
  *a1 = v18;
  a1[1] = (const __m128i *)v17;
  a1[2] = (const __m128i *)v29;
  return v29;
}
