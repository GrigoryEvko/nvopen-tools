// Function: sub_39D5900
// Address: 0x39d5900
//
unsigned __int64 *__fastcall sub_39D5900(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // rbx
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  __int64 m128i_i64; // r8
  bool v13; // zf
  __int64 *v14; // rdx
  __m128i *v15; // r12
  _BYTE *v16; // rsi
  __m128i *v17; // r13
  const __m128i *i; // r12
  __m128i v19; // xmm1
  unsigned __int64 v20; // rdi
  const __m128i *v21; // rsi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __int64 v24; // rsi
  __m128i v25; // xmm0
  unsigned __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  const __m128i *v31; // [rsp+8h] [rbp-58h]
  const __m128i *v32; // [rsp+10h] [rbp-50h]
  unsigned __int64 v33; // [rsp+18h] [rbp-48h]
  unsigned __int64 v35; // [rsp+28h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 4);
  if ( v7 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 4);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 4);
  v11 = &a2->m128i_i8[-v6];
  if ( v9 )
  {
    v27 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v33 = 0;
      m128i_i64 = 48;
      v35 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x2AAAAAAAAAAAAAALL )
      v10 = 0x2AAAAAAAAAAAAAALL;
    v27 = 48 * v10;
  }
  v31 = a3;
  v28 = sub_22077B0(v27);
  v11 = &a2->m128i_i8[-v6];
  a3 = v31;
  v35 = v28;
  m128i_i64 = v28 + 48;
  v33 = v28 + v27;
LABEL_7:
  v13 = &v11[v35] == 0;
  v14 = (__int64 *)&v11[v35];
  v15 = (__m128i *)v14;
  if ( !v13 )
  {
    v16 = (_BYTE *)a3->m128i_i64[0];
    v29 = m128i_i64;
    *v14 = (__int64)(v14 + 2);
    v32 = a3;
    sub_39CF630(v14, v16, (__int64)&v16[a3->m128i_i64[1]]);
    m128i_i64 = v29;
    v15[2] = _mm_loadu_si128(v32 + 2);
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v17 = (__m128i *)v35;
    for ( i = (const __m128i *)(v6 + 16); ; i += 3 )
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
      v20 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v20 != i )
        j_j___libc_free_0(v20);
      if ( a2 == &i[2] )
        break;
      v17 += 3;
    }
    m128i_i64 = (__int64)v17[6].m128i_i64;
  }
  if ( a2 != v5 )
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
    while ( v5 != v23 );
    m128i_i64 += 16
               * (3
                * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v5 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v6 )
  {
    v30 = m128i_i64;
    j_j___libc_free_0(v6);
    m128i_i64 = v30;
  }
  *a1 = v35;
  a1[1] = m128i_i64;
  a1[2] = v33;
  return a1;
}
