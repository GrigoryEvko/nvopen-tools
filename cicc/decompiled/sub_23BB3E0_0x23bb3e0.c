// Function: sub_23BB3E0
// Address: 0x23bb3e0
//
unsigned __int64 *__fastcall sub_23BB3E0(unsigned __int64 *a1, const __m128i *a2, __int64 a3)
{
  const __m128i *v3; // rbx
  __int64 v4; // rax
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  const __m128i *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int8 *v13; // rdx
  __int64 m128i_i64; // r9
  __int64 *v15; // rdi
  _BYTE *v16; // r10
  __int64 v17; // rdx
  __m128i *v18; // r13
  const __m128i *i; // r15
  const __m128i *v20; // rsi
  unsigned __int64 v21; // rdi
  __m128i *v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  __m128i *v32; // [rsp+28h] [rbp-38h]

  v3 = (const __m128i *)a1[1];
  v31 = *a1;
  v4 = (__int64)((__int64)v3->m128i_i64 - *a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = (__int64)((__int64)v3->m128i_i64 - *a1) >> 5;
  v8 = v4 == 0;
  v9 = 1;
  v10 = a2;
  if ( !v8 )
    v9 = v7;
  v11 = __CFADD__(v7, v9);
  v12 = v7 + v9;
  v13 = &a2->m128i_i8[-v31];
  if ( v11 )
  {
    v25 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v12 )
    {
      v29 = 0;
      m128i_i64 = 32;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x3FFFFFFFFFFFFFFLL )
      v12 = 0x3FFFFFFFFFFFFFFLL;
    v25 = 32 * v12;
  }
  v26 = sub_22077B0(v25);
  v13 = &a2->m128i_i8[-v31];
  v32 = (__m128i *)v26;
  m128i_i64 = v26 + 32;
  v29 = v26 + v25;
LABEL_7:
  v15 = (__int64 *)&v13[(_QWORD)v32];
  if ( &v13[(_QWORD)v32] )
  {
    v16 = *(_BYTE **)a3;
    v17 = *(_QWORD *)(a3 + 8);
    v27 = m128i_i64;
    *v15 = (__int64)(v15 + 2);
    sub_23AEDD0(v15, v16, (__int64)&v16[v17]);
    m128i_i64 = v27;
  }
  if ( a2 != (const __m128i *)v31 )
  {
    v18 = v32;
    for ( i = (const __m128i *)(v31 + 16); ; i += 2 )
    {
      if ( v18 )
      {
        v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
        v20 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v20 == i )
        {
          v18[1] = _mm_loadu_si128(i);
        }
        else
        {
          v18->m128i_i64[0] = (__int64)v20;
          v18[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v18->m128i_i64[1] = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v21 = i[-1].m128i_u64[0];
        if ( (const __m128i *)v21 != i )
          j_j___libc_free_0(v21);
      }
      if ( a2 == &i[1] )
        break;
      v18 += 2;
    }
    m128i_i64 = (__int64)v18[4].m128i_i64;
  }
  if ( a2 != v3 )
  {
    v22 = (__m128i *)m128i_i64;
    do
    {
      v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
      if ( (const __m128i *)v10->m128i_i64[0] == &v10[1] )
      {
        v22[1] = _mm_loadu_si128(v10 + 1);
      }
      else
      {
        v22->m128i_i64[0] = v10->m128i_i64[0];
        v22[1].m128i_i64[0] = v10[1].m128i_i64[0];
      }
      v23 = v10->m128i_i64[1];
      v10 += 2;
      v22 += 2;
      v22[-2].m128i_i64[1] = v23;
    }
    while ( v10 != v3 );
    m128i_i64 += (char *)v3 - (char *)a2;
  }
  if ( v31 )
  {
    v28 = m128i_i64;
    j_j___libc_free_0(v31);
    m128i_i64 = v28;
  }
  *a1 = (unsigned __int64)v32;
  a1[1] = m128i_i64;
  a1[2] = v29;
  return a1;
}
