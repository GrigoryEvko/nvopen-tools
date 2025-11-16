// Function: sub_C99F30
// Address: 0xc99f30
//
__int64 *__fastcall sub_C99F30(__int64 *a1, const __m128i *a2, __m128i *a3, const __m128i *a4)
{
  const __m128i *v4; // r15
  const __m128i *v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  bool v9; // cf
  unsigned __int64 v10; // rax
  signed __int64 v11; // r9
  __int64 m128i_i64; // rbx
  __m128i *v13; // rax
  __m128i *v14; // r9
  __int64 v15; // r9
  __m128i v16; // xmm4
  __m128i *v17; // r12
  const __m128i *i; // rbx
  __m128i v19; // xmm1
  const __m128i *v20; // rdi
  const __m128i *v21; // rdx
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __int64 v24; // rsi
  __m128i v25; // xmm0
  __int64 v27; // rbx
  __int64 v28; // rax
  const __m128i *v29; // [rsp+0h] [rbp-60h]
  __m128i *v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v4 = (const __m128i *)a1[1];
  v5 = (const __m128i *)*a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v4->m128i_i64 - *a1) >> 4);
  if ( v6 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * (v4 - v5);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x5555555555555555LL * (v4 - v5);
  v11 = (char *)a2 - (char *)v5;
  if ( v9 )
  {
    v27 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v31 = 0;
      m128i_i64 = 48;
      v33 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x2AAAAAAAAAAAAAALL )
      v10 = 0x2AAAAAAAAAAAAAALL;
    v27 = 48 * v10;
  }
  v29 = a4;
  v30 = a3;
  v28 = sub_22077B0(v27);
  v11 = (char *)a2 - (char *)v5;
  a3 = v30;
  v33 = v28;
  a4 = v29;
  v31 = v28 + v27;
  m128i_i64 = v28 + 48;
LABEL_7:
  v13 = (__m128i *)(v33 + v11);
  if ( v33 + v11 )
  {
    v14 = (__m128i *)a3->m128i_i64[0];
    v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
    if ( v14 == &a3[1] )
    {
      v13[1] = _mm_loadu_si128(a3 + 1);
    }
    else
    {
      v13->m128i_i64[0] = (__int64)v14;
      v13[1].m128i_i64[0] = a3[1].m128i_i64[0];
    }
    v15 = a3->m128i_i64[1];
    a3->m128i_i64[0] = (__int64)a3[1].m128i_i64;
    a3->m128i_i64[1] = 0;
    a3[1].m128i_i8[0] = 0;
    v16 = _mm_loadu_si128(a4);
    v13->m128i_i64[1] = v15;
    v13[2] = v16;
  }
  if ( a2 != v5 )
  {
    v17 = (__m128i *)v33;
    for ( i = v5 + 1; ; i += 3 )
    {
      if ( v17 )
      {
        v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
        v21 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v21 == i )
        {
          v17[1] = _mm_loadu_si128(i);
        }
        else
        {
          v17->m128i_i64[0] = (__int64)v21;
          v17[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v17->m128i_i64[1] = i[-1].m128i_i64[1];
        v19 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v17[2] = v19;
      }
      v20 = (const __m128i *)i[-1].m128i_i64[0];
      if ( v20 != i )
        j_j___libc_free_0(v20, i->m128i_i64[0] + 1);
      if ( a2 == &i[2] )
        break;
      v17 += 3;
    }
    m128i_i64 = (__int64)v17[6].m128i_i64;
  }
  if ( a2 != v4 )
  {
    v22 = (__m128i *)m128i_i64;
    v23 = a2;
    do
    {
      v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
      if ( (const __m128i *)v23->m128i_i64[0] == &v23[1] )
      {
        v22[1] = _mm_loadu_si128(v23 + 1);
      }
      else
      {
        v22->m128i_i64[0] = v23->m128i_i64[0];
        v22[1].m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v24 = v23->m128i_i64[1];
      v25 = _mm_loadu_si128(v23 + 2);
      v23 += 3;
      v22 += 3;
      v22[-3].m128i_i64[1] = v24;
      v22[-1] = v25;
    }
    while ( v4 != v23 );
    m128i_i64 += 16
               * (3
                * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v4 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - (_QWORD)v5);
  *a1 = v33;
  a1[1] = m128i_i64;
  a1[2] = v31;
  return a1;
}
