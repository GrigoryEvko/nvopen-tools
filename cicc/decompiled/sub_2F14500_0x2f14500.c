// Function: sub_2F14500
// Address: 0x2f14500
//
unsigned __int64 *__fastcall sub_2F14500(unsigned __int64 *a1, const __m128i *a2, __m128i *a3)
{
  const __m128i *v4; // r15
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rsi
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rsi
  __int64 m128i_i64; // rbx
  __m128i *v12; // rax
  __m128i *v13; // rdi
  __int64 v14; // rdi
  __m128i v15; // xmm4
  __m128i *v16; // r12
  const __m128i *i; // rbx
  __m128i v18; // xmm1
  unsigned __int64 v19; // rdi
  const __m128i *v20; // rdx
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  __int64 v23; // rsi
  __m128i v24; // xmm0
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __m128i *v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+18h] [rbp-48h]
  unsigned __int64 v31; // [rsp+28h] [rbp-38h]

  v4 = (const __m128i *)a1[1];
  v5 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v4->m128i_i64 - *a1) >> 4);
  if ( v6 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v4->m128i_i64 - v5) >> 4);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x5555555555555555LL * ((__int64)((__int64)v4->m128i_i64 - v5) >> 4);
  v10 = &a2->m128i_i8[-v5];
  if ( v8 )
  {
    v26 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v29 = 0;
      m128i_i64 = 48;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x2AAAAAAAAAAAAAALL )
      v9 = 0x2AAAAAAAAAAAAAALL;
    v26 = 48 * v9;
  }
  v28 = a3;
  v27 = sub_22077B0(v26);
  v10 = &a2->m128i_i8[-v5];
  a3 = v28;
  v31 = v27;
  v29 = v27 + v26;
  m128i_i64 = v27 + 48;
LABEL_7:
  v12 = (__m128i *)&v10[v31];
  if ( &v10[v31] )
  {
    v13 = (__m128i *)a3->m128i_i64[0];
    v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
    if ( v13 == &a3[1] )
    {
      v12[1] = _mm_loadu_si128(a3 + 1);
    }
    else
    {
      v12->m128i_i64[0] = (__int64)v13;
      v12[1].m128i_i64[0] = a3[1].m128i_i64[0];
    }
    v14 = a3->m128i_i64[1];
    v15 = _mm_loadu_si128(a3 + 2);
    a3->m128i_i64[0] = (__int64)a3[1].m128i_i64;
    a3->m128i_i64[1] = 0;
    v12->m128i_i64[1] = v14;
    a3[1].m128i_i8[0] = 0;
    v12[2] = v15;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v16 = (__m128i *)v31;
    for ( i = (const __m128i *)(v5 + 16); ; i += 3 )
    {
      if ( v16 )
      {
        v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
        v20 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v20 == i )
        {
          v16[1] = _mm_loadu_si128(i);
        }
        else
        {
          v16->m128i_i64[0] = (__int64)v20;
          v16[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v16->m128i_i64[1] = i[-1].m128i_i64[1];
        v18 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v16[2] = v18;
      }
      v19 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v19 != i )
        j_j___libc_free_0(v19);
      if ( a2 == &i[2] )
        break;
      v16 += 3;
    }
    m128i_i64 = (__int64)v16[6].m128i_i64;
  }
  if ( a2 != v4 )
  {
    v21 = (__m128i *)m128i_i64;
    v22 = a2;
    do
    {
      v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
      if ( (const __m128i *)v22->m128i_i64[0] == &v22[1] )
      {
        v21[1] = _mm_loadu_si128(v22 + 1);
      }
      else
      {
        v21->m128i_i64[0] = v22->m128i_i64[0];
        v21[1].m128i_i64[0] = v22[1].m128i_i64[0];
      }
      v23 = v22->m128i_i64[1];
      v24 = _mm_loadu_si128(v22 + 2);
      v22 += 3;
      v21 += 3;
      v21[-3].m128i_i64[1] = v23;
      v21[-1] = v24;
    }
    while ( v4 != v22 );
    m128i_i64 += 16
               * (3
                * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v4 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  *a1 = v31;
  a1[1] = m128i_i64;
  a1[2] = v29;
  return a1;
}
