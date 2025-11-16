// Function: sub_2F14070
// Address: 0x2f14070
//
unsigned __int64 *__fastcall sub_2F14070(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v4; // rbx
  unsigned __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  __int64 m128i_i64; // rcx
  bool v13; // zf
  __int64 *v14; // rdx
  __m128i *v15; // r12
  _BYTE *v16; // rsi
  __m128i v17; // xmm4
  __m128i *v18; // r13
  const __m128i *i; // r12
  __m128i v20; // xmm1
  unsigned __int64 v21; // rdi
  const __m128i *v22; // rdx
  const __m128i *v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rsi
  __m128i v26; // xmm0
  unsigned __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  const __m128i *v32; // [rsp+8h] [rbp-58h]
  const __m128i *v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int64 v36; // [rsp+28h] [rbp-38h]

  v4 = (const __m128i *)a1[1];
  v5 = *a1;
  v6 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)((__int64)v4->m128i_i64 - *a1) >> 3);
  if ( v6 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)((__int64)v4->m128i_i64 - v5) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 + v6;
  v11 = &a2->m128i_i8[-v5];
  if ( v9 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v34 = 0;
      m128i_i64 = 56;
      v36 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x249249249249249LL )
      v10 = 0x249249249249249LL;
    v28 = 56 * v10;
  }
  v32 = a3;
  v29 = sub_22077B0(v28);
  v11 = &a2->m128i_i8[-v5];
  a3 = v32;
  v36 = v29;
  m128i_i64 = v29 + 56;
  v34 = v29 + v28;
LABEL_7:
  v13 = &v11[v36] == 0;
  v14 = (__int64 *)&v11[v36];
  v15 = (__m128i *)v14;
  if ( !v13 )
  {
    v16 = (_BYTE *)a3->m128i_i64[0];
    v30 = m128i_i64;
    *v14 = (__int64)(v14 + 2);
    v33 = a3;
    sub_2F07250(v14, v16, (__int64)&v16[a3->m128i_i64[1]]);
    m128i_i64 = v30;
    v17 = _mm_loadu_si128(v33 + 2);
    v15[3].m128i_i16[0] = v33[3].m128i_i16[0];
    v15[2] = v17;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v18 = (__m128i *)v36;
    for ( i = (const __m128i *)(v5 + 16); ; i = (const __m128i *)((char *)i + 56) )
    {
      if ( v18 )
      {
        v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
        v22 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v22 == i )
        {
          v18[1] = _mm_loadu_si128(i);
        }
        else
        {
          v18->m128i_i64[0] = (__int64)v22;
          v18[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v18->m128i_i64[1] = i[-1].m128i_i64[1];
        v20 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v18[2] = v20;
        v18[3].m128i_i16[0] = i[2].m128i_i16[0];
      }
      v21 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v21 != i )
        j_j___libc_free_0(v21);
      if ( a2 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
      v18 = (__m128i *)((char *)v18 + 56);
    }
    m128i_i64 = (__int64)v18[7].m128i_i64;
  }
  if ( a2 != v4 )
  {
    v23 = a2;
    v24 = (__m128i *)m128i_i64;
    do
    {
      v24->m128i_i64[0] = (__int64)v24[1].m128i_i64;
      if ( (const __m128i *)v23->m128i_i64[0] == &v23[1] )
      {
        v24[1] = _mm_loadu_si128(v23 + 1);
      }
      else
      {
        v24->m128i_i64[0] = v23->m128i_i64[0];
        v24[1].m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v25 = v23->m128i_i64[1];
      v26 = _mm_loadu_si128(v23 + 2);
      v23 = (const __m128i *)((char *)v23 + 56);
      v24 = (__m128i *)((char *)v24 + 56);
      v24[-3].m128i_i64[0] = v25;
      LOWORD(v25) = v23[-1].m128i_i16[4];
      *(__m128i *)((char *)v24 - 24) = v26;
      v24[-1].m128i_i16[4] = v25;
    }
    while ( v23 != v4 );
    m128i_i64 += 56
               * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)v23 - (char *)a2 - 56) >> 3))
                 & 0x1FFFFFFFFFFFFFFFLL)
                + 1);
  }
  if ( v5 )
  {
    v31 = m128i_i64;
    j_j___libc_free_0(v5);
    m128i_i64 = v31;
  }
  a1[1] = m128i_i64;
  *a1 = v36;
  a1[2] = v34;
  return a1;
}
