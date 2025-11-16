// Function: sub_39D5540
// Address: 0x39d5540
//
unsigned __int64 *__fastcall sub_39D5540(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rdx
  __int64 m128i_i64; // r8
  bool v12; // zf
  __int64 *v13; // rdx
  __m128i *v14; // r13
  _BYTE *v15; // rsi
  _BYTE *v16; // rsi
  __int64 v17; // rdx
  __m128i *v18; // r15
  const __m128i *v19; // r14
  unsigned __int64 v20; // r13
  __m128i v21; // xmm1
  __int64 v22; // rsi
  __m128i v23; // xmm2
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  const __m128i *v26; // rsi
  const __m128i *v27; // rax
  const __m128i *v28; // rcx
  __m128i *v29; // rdx
  __int64 v30; // rsi
  __m128i v31; // xmm3
  __int64 v32; // rsi
  __int64 v33; // rsi
  __m128i v34; // xmm0
  const __m128i *v35; // rsi
  unsigned __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-60h]
  const __m128i *v40; // [rsp+0h] [rbp-60h]
  const __m128i *v41; // [rsp+8h] [rbp-58h]
  __int64 v42; // [rsp+8h] [rbp-58h]
  unsigned __int64 v43; // [rsp+10h] [rbp-50h]
  unsigned __int64 v45; // [rsp+20h] [rbp-40h]
  unsigned __int64 v46; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v45 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v5 - *a1) >> 5);
  if ( v6 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v5 - v45) >> 5);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x5555555555555555LL * ((__int64)(v5 - v45) >> 5);
  v10 = &a2->m128i_i8[-v45];
  if ( v8 )
  {
    v37 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v43 = 0;
      m128i_i64 = 96;
      v46 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x155555555555555LL )
      v9 = 0x155555555555555LL;
    v37 = 96 * v9;
  }
  v40 = a3;
  v38 = sub_22077B0(v37);
  v10 = &a2->m128i_i8[-v45];
  v46 = v38;
  a3 = v40;
  m128i_i64 = v38 + 96;
  v43 = v38 + v37;
LABEL_7:
  v12 = &v10[v46] == 0;
  v13 = (__int64 *)&v10[v46];
  v14 = (__m128i *)v13;
  if ( !v12 )
  {
    v15 = (_BYTE *)a3->m128i_i64[0];
    v39 = m128i_i64;
    *v13 = (__int64)(v13 + 2);
    v41 = a3;
    sub_39CF630(v13, v15, (__int64)&v15[a3->m128i_i64[1]]);
    v14[3].m128i_i64[0] = (__int64)v14[4].m128i_i64;
    v16 = (_BYTE *)v41[3].m128i_i64[0];
    v17 = v41[3].m128i_i64[1];
    v14[2] = _mm_loadu_si128(v41 + 2);
    sub_39CF630(v14[3].m128i_i64, v16, (__int64)&v16[v17]);
    m128i_i64 = v39;
    v14[5] = _mm_loadu_si128(v41 + 5);
  }
  if ( a2 != (const __m128i *)v45 )
  {
    v18 = (__m128i *)v46;
    v19 = (const __m128i *)(v45 + 16);
    v20 = v45 + 64;
    while ( 1 )
    {
      if ( v18 )
      {
        v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
        v26 = (const __m128i *)v19[-1].m128i_i64[0];
        if ( v26 == v19 )
        {
          v18[1] = _mm_loadu_si128(v19);
        }
        else
        {
          v18->m128i_i64[0] = (__int64)v26;
          v18[1].m128i_i64[0] = v19->m128i_i64[0];
        }
        v18->m128i_i64[1] = v19[-1].m128i_i64[1];
        v21 = _mm_loadu_si128(v19 + 1);
        v19[-1].m128i_i64[0] = (__int64)v19;
        v19[-1].m128i_i64[1] = 0;
        v19->m128i_i8[0] = 0;
        v18[3].m128i_i64[0] = (__int64)v18[4].m128i_i64;
        v18[2] = v21;
        v22 = v19[2].m128i_i64[0];
        if ( v22 == v20 )
        {
          v18[4] = _mm_loadu_si128(v19 + 3);
        }
        else
        {
          v18[3].m128i_i64[0] = v22;
          v18[4].m128i_i64[0] = v19[3].m128i_i64[0];
        }
        v18[3].m128i_i64[1] = v19[2].m128i_i64[1];
        v23 = _mm_loadu_si128(v19 + 4);
        v19[2].m128i_i64[0] = v20;
        v19[2].m128i_i64[1] = 0;
        v19[3].m128i_i8[0] = 0;
        v18[5] = v23;
      }
      v24 = v19[2].m128i_u64[0];
      if ( v24 != v20 )
        j_j___libc_free_0(v24);
      v25 = v19[-1].m128i_u64[0];
      if ( (const __m128i *)v25 != v19 )
        j_j___libc_free_0(v25);
      v20 += 96LL;
      if ( a2 == &v19[5] )
        break;
      v19 += 6;
      v18 += 6;
    }
    m128i_i64 = (__int64)v18[12].m128i_i64;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v27 = a2 + 1;
    v28 = a2;
    v29 = (__m128i *)m128i_i64;
    do
    {
      v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
      v35 = (const __m128i *)v27[-1].m128i_i64[0];
      if ( v35 == v27 )
      {
        v29[1] = _mm_loadu_si128(v27);
      }
      else
      {
        v29->m128i_i64[0] = (__int64)v35;
        v29[1].m128i_i64[0] = v27->m128i_i64[0];
      }
      v30 = v27[-1].m128i_i64[1];
      v31 = _mm_loadu_si128(v27 + 1);
      v27->m128i_i8[0] = 0;
      v27[-1].m128i_i64[0] = (__int64)v27;
      v29->m128i_i64[1] = v30;
      v29[3].m128i_i64[0] = (__int64)v29[4].m128i_i64;
      v32 = v27[2].m128i_i64[0];
      v27[-1].m128i_i64[1] = 0;
      v29[2] = v31;
      if ( (const __m128i *)v32 == &v28[4] )
      {
        v29[4] = _mm_loadu_si128(v27 + 3);
      }
      else
      {
        v29[3].m128i_i64[0] = v32;
        v29[4].m128i_i64[0] = v27[3].m128i_i64[0];
      }
      v33 = v27[2].m128i_i64[1];
      v28 += 6;
      v29 += 6;
      v27 += 6;
      v34 = _mm_loadu_si128(v27 - 2);
      v29[-3].m128i_i64[1] = v33;
      v29[-1] = v34;
    }
    while ( v28 != (const __m128i *)v5 );
    m128i_i64 += 32
               * (3
                * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v28 - (char *)a2 - 96) >> 5)) & 0x7FFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v45 )
  {
    v42 = m128i_i64;
    j_j___libc_free_0(v45);
    m128i_i64 = v42;
  }
  *a1 = v46;
  a1[1] = m128i_i64;
  a1[2] = v43;
  return a1;
}
