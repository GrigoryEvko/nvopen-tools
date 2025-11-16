// Function: sub_9CBC60
// Address: 0x9cbc60
//
const __m128i **__fastcall sub_9CBC60(const __m128i **a1, const __m128i *a2)
{
  const __m128i *v2; // r12
  const __m128i *v3; // r15
  __int64 v4; // rax
  bool v5; // zf
  __int64 v7; // rsi
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rdx
  __int64 m128i_i64; // rbx
  char *v13; // rax
  __m128i *v14; // r13
  const __m128i *i; // rbx
  const __m128i *v16; // rdx
  const __m128i *v17; // rdi
  __m128i *v18; // rax
  signed __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-48h]
  __m128i *v26; // [rsp+18h] [rbp-38h]

  v2 = a1[1];
  v3 = *a1;
  v4 = ((char *)v2 - (char *)*a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = v4 == 0;
  v7 = ((char *)v2 - (char *)v3) >> 5;
  v8 = 1;
  if ( !v5 )
    v8 = ((char *)v2 - (char *)v3) >> 5;
  v9 = __CFADD__(v7, v8);
  v10 = v7 + v8;
  v11 = (char *)((char *)a2 - (char *)v3);
  if ( v9 )
  {
    v22 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v24 = 0;
      m128i_i64 = 32;
      v26 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v22 = 32 * v10;
  }
  v23 = sub_22077B0(v22);
  v11 = (char *)((char *)a2 - (char *)v3);
  v26 = (__m128i *)v23;
  v24 = v23 + v22;
  m128i_i64 = v23 + 32;
LABEL_7:
  v13 = &v11[(_QWORD)v26];
  if ( &v11[(_QWORD)v26] )
  {
    *((_QWORD *)v13 + 1) = 0;
    *(_QWORD *)v13 = v13 + 16;
    v13[16] = 0;
  }
  if ( a2 != v3 )
  {
    v14 = v26;
    for ( i = v3 + 1; ; i += 2 )
    {
      if ( v14 )
      {
        v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
        v16 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v16 == i )
        {
          v14[1] = _mm_loadu_si128(i);
        }
        else
        {
          v14->m128i_i64[0] = (__int64)v16;
          v14[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v14->m128i_i64[1] = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v17 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v17 != i )
          j_j___libc_free_0(v17, i->m128i_i64[0] + 1);
      }
      if ( a2 == &i[1] )
        break;
      v14 += 2;
    }
    m128i_i64 = (__int64)v14[4].m128i_i64;
  }
  if ( a2 != v2 )
  {
    v18 = (__m128i *)m128i_i64;
    v19 = (char *)v2 - (char *)a2;
    do
    {
      v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
      if ( (const __m128i *)a2->m128i_i64[0] == &a2[1] )
      {
        v18[1] = _mm_loadu_si128(a2 + 1);
      }
      else
      {
        v18->m128i_i64[0] = a2->m128i_i64[0];
        v18[1].m128i_i64[0] = a2[1].m128i_i64[0];
      }
      v20 = a2->m128i_i64[1];
      a2 += 2;
      v18 += 2;
      v18[-2].m128i_i64[1] = v20;
    }
    while ( a2 != v2 );
    m128i_i64 += v19;
  }
  if ( v3 )
    j_j___libc_free_0(v3, (char *)a1[2] - (char *)v3);
  *a1 = v26;
  a1[1] = (const __m128i *)m128i_i64;
  a1[2] = (const __m128i *)v24;
  return a1;
}
