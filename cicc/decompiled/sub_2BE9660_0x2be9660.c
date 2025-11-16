// Function: sub_2BE9660
// Address: 0x2be9660
//
unsigned __int64 *__fastcall sub_2BE9660(unsigned __int64 *a1, const __m128i *a2, __m128i *a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  bool v5; // zf
  __int64 v7; // rsi
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rsi
  __int64 m128i_i64; // r8
  __m128i *v13; // rax
  __m128i *v14; // rsi
  __int64 v15; // rsi
  __m128i *v16; // rsi
  __int64 v17; // rsi
  __m128i *v18; // r15
  const __m128i *v19; // r12
  unsigned __int64 v20; // r14
  const __m128i *v21; // rsi
  __int64 v22; // rsi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  const __m128i *v25; // rax
  __m128i *v26; // rdx
  __int64 v27; // rcx
  const __m128i *v28; // rcx
  __int64 v29; // rcx
  const __m128i *v30; // rcx
  unsigned __int64 v32; // rcx
  __m128i *v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 v35; // [rsp+10h] [rbp-50h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  const __m128i *v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v3 = *a1;
  v38 = (const __m128i *)a1[1];
  v4 = (__int64)((__int64)v38->m128i_i64 - *a1) >> 6;
  if ( v4 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = v4 == 0;
  v7 = (__int64)((__int64)v38->m128i_i64 - *a1) >> 6;
  v8 = 1;
  if ( !v5 )
    v8 = (__int64)((__int64)v38->m128i_i64 - *a1) >> 6;
  v9 = __CFADD__(v7, v8);
  v10 = v7 + v8;
  v11 = &a2->m128i_i8[-v3];
  if ( v9 )
  {
    v32 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v10 )
    {
      v35 = 0;
      m128i_i64 = 64;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x1FFFFFFFFFFFFFFLL )
      v10 = 0x1FFFFFFFFFFFFFFLL;
    v32 = v10 << 6;
  }
  v33 = a3;
  v36 = v32;
  v11 = &a2->m128i_i8[-v3];
  v39 = sub_22077B0(v32);
  a3 = v33;
  m128i_i64 = v39 + 64;
  v35 = v39 + v36;
LABEL_7:
  v13 = (__m128i *)&v11[v39];
  if ( &v11[v39] )
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
    v13[2].m128i_i64[0] = (__int64)v13[3].m128i_i64;
    v13->m128i_i64[1] = v15;
    v16 = (__m128i *)a3[2].m128i_i64[0];
    a3->m128i_i64[1] = 0;
    a3[1].m128i_i8[0] = 0;
    if ( v16 == &a3[3] )
    {
      v13[3] = _mm_loadu_si128(a3 + 3);
    }
    else
    {
      v13[2].m128i_i64[0] = (__int64)v16;
      v13[3].m128i_i64[0] = a3[3].m128i_i64[0];
    }
    v17 = a3[2].m128i_i64[1];
    a3[2].m128i_i64[0] = (__int64)a3[3].m128i_i64;
    a3[2].m128i_i64[1] = 0;
    v13[2].m128i_i64[1] = v17;
    a3[3].m128i_i8[0] = 0;
  }
  if ( a2 != (const __m128i *)v3 )
  {
    v18 = (__m128i *)v39;
    v19 = (const __m128i *)(v3 + 16);
    v20 = v3 + 48;
    while ( 1 )
    {
      if ( v18 )
      {
        v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
        v21 = (const __m128i *)v19[-1].m128i_i64[0];
        if ( v21 == v19 )
        {
          v18[1] = _mm_loadu_si128(v19);
        }
        else
        {
          v18->m128i_i64[0] = (__int64)v21;
          v18[1].m128i_i64[0] = v19->m128i_i64[0];
        }
        v18->m128i_i64[1] = v19[-1].m128i_i64[1];
        v19[-1].m128i_i64[0] = (__int64)v19;
        v19[-1].m128i_i64[1] = 0;
        v19->m128i_i8[0] = 0;
        v18[2].m128i_i64[0] = (__int64)v18[3].m128i_i64;
        v22 = v19[1].m128i_i64[0];
        if ( v22 == v20 )
        {
          v18[3] = _mm_loadu_si128(v19 + 2);
        }
        else
        {
          v18[2].m128i_i64[0] = v22;
          v18[3].m128i_i64[0] = v19[2].m128i_i64[0];
        }
        v18[2].m128i_i64[1] = v19[1].m128i_i64[1];
        v19[1].m128i_i64[0] = v20;
      }
      else
      {
        v24 = v19[1].m128i_u64[0];
        if ( v24 != v20 )
          j_j___libc_free_0(v24);
      }
      v23 = v19[-1].m128i_u64[0];
      if ( (const __m128i *)v23 != v19 )
        j_j___libc_free_0(v23);
      v20 += 64LL;
      if ( a2 == &v19[3] )
        break;
      v19 += 4;
      v18 += 4;
    }
    m128i_i64 = (__int64)v18[8].m128i_i64;
  }
  if ( a2 != v38 )
  {
    v25 = a2 + 1;
    v26 = (__m128i *)m128i_i64;
    do
    {
      v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
      v30 = (const __m128i *)v25[-1].m128i_i64[0];
      if ( v25 == v30 )
      {
        v26[1] = _mm_loadu_si128(v25);
      }
      else
      {
        v26->m128i_i64[0] = (__int64)v30;
        v26[1].m128i_i64[0] = v25->m128i_i64[0];
      }
      v27 = v25[-1].m128i_i64[1];
      v25[-1].m128i_i64[0] = (__int64)v25;
      v25[-1].m128i_i64[1] = 0;
      v26->m128i_i64[1] = v27;
      v26[2].m128i_i64[0] = (__int64)v26[3].m128i_i64;
      v28 = (const __m128i *)v25[1].m128i_i64[0];
      v25->m128i_i8[0] = 0;
      if ( v28 == &v25[2] )
      {
        v26[3] = _mm_loadu_si128(v25 + 2);
      }
      else
      {
        v26[2].m128i_i64[0] = (__int64)v28;
        v26[3].m128i_i64[0] = v25[2].m128i_i64[0];
      }
      v29 = v25[1].m128i_i64[1];
      v25 += 4;
      v26 += 4;
      v26[-2].m128i_i64[1] = v29;
    }
    while ( v25 != &v38[1] );
    m128i_i64 += (char *)v38 - (char *)a2;
  }
  if ( v3 )
  {
    v34 = m128i_i64;
    j_j___libc_free_0(v3);
    m128i_i64 = v34;
  }
  *a1 = v39;
  a1[1] = m128i_i64;
  a1[2] = v35;
  return a1;
}
