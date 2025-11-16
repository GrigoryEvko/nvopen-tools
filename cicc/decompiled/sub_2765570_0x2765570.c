// Function: sub_2765570
// Address: 0x2765570
//
void __fastcall sub_2765570(unsigned __int64 *a1, const __m128i *a2)
{
  __m128i *v2; // rbx
  unsigned __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  bool v7; // cf
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // r13
  __m128i *v11; // r15
  __m128i *v12; // rdx
  __m128i *v13; // rdx
  const __m128i *v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int8 *v18; // [rsp+8h] [rbp-38h]

  v2 = (__m128i *)a1[1];
  if ( v2 != (__m128i *)a1[2] )
  {
    if ( v2 )
    {
      *v2 = _mm_loadu_si128(a2);
      v2 = (__m128i *)a1[1];
    }
    a1[1] = (unsigned __int64)&v2[1];
    return;
  }
  v3 = *a1;
  v4 = (__int64)v2->m128i_i64 - *a1;
  v5 = v4 >> 4;
  if ( v4 >> 4 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = (__int64)((__int64)v2->m128i_i64 - *a1) >> 4;
  v7 = __CFADD__(v6, v5);
  v8 = v6 + v5;
  if ( v7 )
  {
    v15 = 0x7FFFFFFFFFFFFFF0LL;
LABEL_25:
    v18 = &v2->m128i_i8[-*a1];
    v16 = sub_22077B0(v15);
    v4 = (__int64)v18;
    v11 = (__m128i *)v16;
    v10 = v16 + v15;
    v9 = v16 + 16;
    goto LABEL_11;
  }
  if ( v8 )
  {
    if ( v8 > 0x7FFFFFFFFFFFFFFLL )
      v8 = 0x7FFFFFFFFFFFFFFLL;
    v15 = 16 * v8;
    goto LABEL_25;
  }
  v9 = 16;
  v10 = 0;
  v11 = 0;
LABEL_11:
  v12 = (__m128i *)((char *)v11 + v4);
  if ( v12 )
    *v12 = _mm_loadu_si128(a2);
  if ( v2 != (__m128i *)v3 )
  {
    v13 = v11;
    v14 = (const __m128i *)v3;
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v14);
      ++v14;
      ++v13;
    }
    while ( v2 != v14 );
    v9 = (__int64)v2[1].m128i_i64 + (_QWORD)v11 - v3;
  }
  if ( v3 )
  {
    v17 = v9;
    j_j___libc_free_0(v3);
    v9 = v17;
  }
  *a1 = (unsigned __int64)v11;
  a1[2] = v10;
  a1[1] = v9;
}
