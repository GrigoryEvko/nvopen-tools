// Function: sub_8F99A0
// Address: 0x8f99a0
//
__m128i **__fastcall sub_8F99A0(__m128i **a1, const __m128i *a2, __m128i *a3)
{
  const __m128i *v3; // r12
  __int64 v4; // rax
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  const __m128i *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  signed __int64 v13; // rdx
  __int64 m128i_i64; // rbx
  __m128i *v15; // rdx
  __m128i *v16; // rdi
  __int64 v17; // rdi
  __m128i *v18; // r15
  const __m128i *i; // rbx
  const __m128i *v20; // rdx
  const __m128i *v21; // rdi
  __m128i *v22; // rax
  __int64 v23; // rdx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // [rsp+10h] [rbp-50h]
  const __m128i *v29; // [rsp+20h] [rbp-40h]
  __m128i *v30; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v29 = *a1;
  v4 = ((char *)v3 - (char *)*a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = ((char *)v3 - (char *)*a1) >> 5;
  v8 = v4 == 0;
  v9 = 1;
  v10 = a2;
  if ( !v8 )
    v9 = v7;
  v11 = __CFADD__(v7, v9);
  v12 = v7 + v9;
  v13 = (char *)a2 - (char *)v29;
  if ( v11 )
  {
    v25 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v12 )
    {
      v27 = 0;
      m128i_i64 = 32;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x3FFFFFFFFFFFFFFLL )
      v12 = 0x3FFFFFFFFFFFFFFLL;
    v25 = 32 * v12;
  }
  v26 = sub_22077B0(v25);
  v13 = (char *)a2 - (char *)v29;
  v30 = (__m128i *)v26;
  v27 = v26 + v25;
  m128i_i64 = v26 + 32;
LABEL_7:
  v15 = (__m128i *)((char *)v30 + v13);
  if ( v15 )
  {
    v16 = (__m128i *)a3->m128i_i64[0];
    v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
    if ( v16 == &a3[1] )
    {
      v15[1] = _mm_loadu_si128(a3 + 1);
    }
    else
    {
      v15->m128i_i64[0] = (__int64)v16;
      v15[1].m128i_i64[0] = a3[1].m128i_i64[0];
    }
    v17 = a3->m128i_i64[1];
    a3->m128i_i64[0] = (__int64)a3[1].m128i_i64;
    a3->m128i_i64[1] = 0;
    v15->m128i_i64[1] = v17;
    a3[1].m128i_i8[0] = 0;
  }
  if ( a2 != v29 )
  {
    v18 = v30;
    for ( i = v29 + 1; ; i += 2 )
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
        v21 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v21 != i )
          j_j___libc_free_0(v21, i->m128i_i64[0] + 1);
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
  if ( v29 )
    j_j___libc_free_0(v29, (char *)a1[2] - (char *)v29);
  *a1 = v30;
  a1[1] = (__m128i *)m128i_i64;
  a1[2] = (__m128i *)v27;
  return a1;
}
