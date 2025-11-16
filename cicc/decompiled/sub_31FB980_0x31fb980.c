// Function: sub_31FB980
// Address: 0x31fb980
//
unsigned __int64 *__fastcall sub_31FB980(unsigned __int64 *a1, const __m128i *a2, __m128i *a3, _QWORD *a4)
{
  const __m128i *v4; // r15
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // r9
  __int64 m128i_i64; // rbx
  __m128i *v13; // rax
  __m128i *v14; // r9
  __int64 v15; // r9
  __int64 v16; // rdx
  __m128i *v17; // r12
  const __m128i *i; // rbx
  __int64 v19; // rdx
  unsigned __int64 v20; // rdi
  const __m128i *v21; // rdx
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-60h]
  __m128i *v29; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+18h] [rbp-48h]
  __m128i *v32; // [rsp+28h] [rbp-38h]

  v4 = (const __m128i *)a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v4->m128i_i64 - *a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v4->m128i_i64 - v5) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x3333333333333333LL * ((__int64)((__int64)v4->m128i_i64 - v5) >> 3);
  v11 = &a2->m128i_i8[-v5];
  if ( v9 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v30 = 0;
      m128i_i64 = 40;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x333333333333333LL )
      v10 = 0x333333333333333LL;
    v26 = 40 * v10;
  }
  v28 = a4;
  v29 = a3;
  v27 = sub_22077B0(v26);
  v11 = &a2->m128i_i8[-v5];
  a3 = v29;
  v32 = (__m128i *)v27;
  a4 = v28;
  v30 = v27 + v26;
  m128i_i64 = v27 + 40;
LABEL_7:
  v13 = (__m128i *)&v11[(_QWORD)v32];
  if ( &v11[(_QWORD)v32] )
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
    v16 = *a4;
    v13->m128i_i64[1] = v15;
    v13[2].m128i_i64[0] = v16;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v17 = v32;
    for ( i = (const __m128i *)(v5 + 16); ; i = (const __m128i *)((char *)i + 40) )
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
        v19 = i[1].m128i_i64[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v17[2].m128i_i64[0] = v19;
      }
      v20 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v20 != i )
        j_j___libc_free_0(v20);
      if ( a2 == (const __m128i *)&i[1].m128i_u64[1] )
        break;
      v17 = (__m128i *)((char *)v17 + 40);
    }
    m128i_i64 = (__int64)v17[5].m128i_i64;
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
      v23 = (const __m128i *)((char *)v23 + 40);
      v22 = (__m128i *)((char *)v22 + 40);
      v22[-2].m128i_i64[0] = v24;
      v22[-1].m128i_i64[1] = v23[-1].m128i_i64[1];
    }
    while ( v4 != v23 );
    m128i_i64 += 8 * ((unsigned __int64)((char *)v4 - (char *)a2 - 40) >> 3) + 40;
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  *a1 = (unsigned __int64)v32;
  a1[1] = m128i_i64;
  a1[2] = v30;
  return a1;
}
