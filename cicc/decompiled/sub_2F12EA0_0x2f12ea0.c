// Function: sub_2F12EA0
// Address: 0x2f12ea0
//
unsigned __int64 *__fastcall sub_2F12EA0(unsigned __int64 *a1, const __m128i *a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdx
  bool v4; // cf
  unsigned __int64 v5; // rax
  __int64 m128i_i64; // rbx
  _QWORD *v7; // r12
  __m128i *v8; // r15
  const __m128i *v9; // r14
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rbx
  __m128i v13; // xmm1
  __int64 v14; // rcx
  __m128i v15; // xmm2
  __int64 v16; // rcx
  __m128i v17; // xmm3
  __int64 v18; // rcx
  __m128i v19; // xmm4
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  const __m128i *v24; // rcx
  const __m128i *v25; // rcx
  const __m128i *v26; // rax
  __int64 v27; // rdi
  __m128i *v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r8
  __m128i v31; // xmm5
  __int64 v32; // r8
  __int64 v33; // r8
  __m128i v34; // xmm6
  __int64 v35; // r8
  __int64 v36; // r8
  __m128i v37; // xmm7
  __int64 v38; // r8
  __int64 v39; // r8
  __m128i v40; // xmm0
  const __m128i *v41; // r8
  unsigned __int64 v43; // rbx
  unsigned __int64 v44; // [rsp+0h] [rbp-60h]
  unsigned __int64 v46; // [rsp+10h] [rbp-50h]
  __int64 v47; // [rsp+18h] [rbp-48h]
  unsigned __int64 v48; // [rsp+20h] [rbp-40h]

  v48 = a1[1];
  v46 = *a1;
  v2 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v48 - *a1) >> 6);
  if ( v2 == 0xAAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v3 = 1;
  if ( v2 )
    v3 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v48 - *a1) >> 6);
  v4 = __CFADD__(v3, v2);
  v5 = v3 - 0x5555555555555555LL * ((__int64)(v48 - *a1) >> 6);
  if ( v4 )
  {
    v43 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v5 )
    {
      v44 = 0;
      m128i_i64 = 192;
      v47 = 0;
      goto LABEL_7;
    }
    if ( v5 > 0xAAAAAAAAAAAAAALL )
      v5 = 0xAAAAAAAAAAAAAALL;
    v43 = 192 * v5;
  }
  v47 = sub_22077B0(v43);
  v44 = v47 + v43;
  m128i_i64 = v47 + 192;
LABEL_7:
  v7 = (__int64 *)((char *)a2->m128i_i64 + v47 - v46);
  if ( v7 )
  {
    memset(v7, 0, 0xC0u);
    *v7 = v7 + 2;
    v7[6] = v7 + 8;
    v7[12] = v7 + 14;
    v7[18] = v7 + 20;
  }
  if ( a2 != (const __m128i *)v46 )
  {
    v8 = (__m128i *)v47;
    v9 = (const __m128i *)(v46 + 16);
    v10 = v46 + 64;
    v11 = v46 + 112;
    v12 = v46 + 160;
    while ( 1 )
    {
      if ( v8 )
      {
        v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
        v24 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v24 == v9 )
        {
          v8[1] = _mm_loadu_si128(v9);
        }
        else
        {
          v8->m128i_i64[0] = (__int64)v24;
          v8[1].m128i_i64[0] = v9->m128i_i64[0];
        }
        v8->m128i_i64[1] = v9[-1].m128i_i64[1];
        v13 = _mm_loadu_si128(v9 + 1);
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        v8[3].m128i_i64[0] = (__int64)v8[4].m128i_i64;
        v8[2] = v13;
        v14 = v9[2].m128i_i64[0];
        if ( v14 == v10 )
        {
          v8[4] = _mm_loadu_si128(v9 + 3);
        }
        else
        {
          v8[3].m128i_i64[0] = v14;
          v8[4].m128i_i64[0] = v9[3].m128i_i64[0];
        }
        v8[3].m128i_i64[1] = v9[2].m128i_i64[1];
        v15 = _mm_loadu_si128(v9 + 4);
        v9[2].m128i_i64[0] = v10;
        v9[2].m128i_i64[1] = 0;
        v9[3].m128i_i8[0] = 0;
        v8[6].m128i_i64[0] = (__int64)v8[7].m128i_i64;
        v8[5] = v15;
        v16 = v9[5].m128i_i64[0];
        if ( v16 == v11 )
        {
          v8[7] = _mm_loadu_si128(v9 + 6);
        }
        else
        {
          v8[6].m128i_i64[0] = v16;
          v8[7].m128i_i64[0] = v9[6].m128i_i64[0];
        }
        v8[6].m128i_i64[1] = v9[5].m128i_i64[1];
        v17 = _mm_loadu_si128(v9 + 7);
        v9[5].m128i_i64[0] = v11;
        v9[5].m128i_i64[1] = 0;
        v9[6].m128i_i8[0] = 0;
        v8[9].m128i_i64[0] = (__int64)v8[10].m128i_i64;
        v8[8] = v17;
        v18 = v9[8].m128i_i64[0];
        if ( v18 == v12 )
        {
          v8[10] = _mm_loadu_si128(v9 + 9);
        }
        else
        {
          v8[9].m128i_i64[0] = v18;
          v8[10].m128i_i64[0] = v9[9].m128i_i64[0];
        }
        v8[9].m128i_i64[1] = v9[8].m128i_i64[1];
        v19 = _mm_loadu_si128(v9 + 10);
        v9[8].m128i_i64[0] = v12;
        v9[8].m128i_i64[1] = 0;
        v9[9].m128i_i8[0] = 0;
        v8[11] = v19;
      }
      v20 = v9[8].m128i_u64[0];
      if ( v20 != v12 )
        j_j___libc_free_0(v20);
      v21 = v9[5].m128i_u64[0];
      if ( v21 != v11 )
        j_j___libc_free_0(v21);
      v22 = v9[2].m128i_u64[0];
      if ( v22 != v10 )
        j_j___libc_free_0(v22);
      v23 = v9[-1].m128i_u64[0];
      if ( (const __m128i *)v23 != v9 )
        j_j___libc_free_0(v23);
      v10 += 192LL;
      v11 += 192LL;
      v12 += 192LL;
      if ( a2 == &v9[11] )
        break;
      v9 += 12;
      v8 += 12;
    }
    m128i_i64 = (__int64)v8[24].m128i_i64;
  }
  v25 = a2;
  if ( a2 != (const __m128i *)v48 )
  {
    v26 = a2 + 1;
    v27 = (__int64)a2[7].m128i_i64;
    v28 = (__m128i *)m128i_i64;
    v29 = (__int64)a2[4].m128i_i64;
    do
    {
      v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
      v41 = (const __m128i *)v26[-1].m128i_i64[0];
      if ( v26 == v41 )
      {
        v28[1] = _mm_loadu_si128(v26);
      }
      else
      {
        v28->m128i_i64[0] = (__int64)v41;
        v28[1].m128i_i64[0] = v26->m128i_i64[0];
      }
      v30 = v26[-1].m128i_i64[1];
      v31 = _mm_loadu_si128(v26 + 1);
      v26[-1].m128i_i64[0] = (__int64)v26;
      v26[-1].m128i_i64[1] = 0;
      v28->m128i_i64[1] = v30;
      v28[3].m128i_i64[0] = (__int64)v28[4].m128i_i64;
      v32 = v26[2].m128i_i64[0];
      v26->m128i_i8[0] = 0;
      v28[2] = v31;
      if ( v32 == v29 )
      {
        v28[4] = _mm_loadu_si128(v26 + 3);
      }
      else
      {
        v28[3].m128i_i64[0] = v32;
        v28[4].m128i_i64[0] = v26[3].m128i_i64[0];
      }
      v33 = v26[2].m128i_i64[1];
      v34 = _mm_loadu_si128(v26 + 4);
      v26[2].m128i_i64[0] = v29;
      v26[2].m128i_i64[1] = 0;
      v28[3].m128i_i64[1] = v33;
      v28[6].m128i_i64[0] = (__int64)v28[7].m128i_i64;
      v35 = v26[5].m128i_i64[0];
      v26[3].m128i_i8[0] = 0;
      v28[5] = v34;
      if ( v35 == v27 )
      {
        v28[7] = _mm_loadu_si128(v26 + 6);
      }
      else
      {
        v28[6].m128i_i64[0] = v35;
        v28[7].m128i_i64[0] = v26[6].m128i_i64[0];
      }
      v36 = v26[5].m128i_i64[1];
      v37 = _mm_loadu_si128(v26 + 7);
      v26[5].m128i_i64[0] = v27;
      v26[5].m128i_i64[1] = 0;
      v28[6].m128i_i64[1] = v36;
      v28[9].m128i_i64[0] = (__int64)v28[10].m128i_i64;
      v38 = v26[8].m128i_i64[0];
      v26[6].m128i_i8[0] = 0;
      v28[8] = v37;
      if ( (const __m128i *)v38 == &v25[10] )
      {
        v28[10] = _mm_loadu_si128(v26 + 9);
      }
      else
      {
        v28[9].m128i_i64[0] = v38;
        v28[10].m128i_i64[0] = v26[9].m128i_i64[0];
      }
      v39 = v26[8].m128i_i64[1];
      v40 = _mm_loadu_si128(v26 + 10);
      v25 += 12;
      v28 += 12;
      v26 += 12;
      v27 += 192;
      v29 += 192;
      v28[-3].m128i_i64[1] = v39;
      v28[-1] = v40;
    }
    while ( v25 != (const __m128i *)v48 );
    m128i_i64 += (3
                * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v25 - (char *)a2 - 192) >> 6))
                 & 0x3FFFFFFFFFFFFFFLL)
                + 3) << 6;
  }
  if ( v46 )
    j_j___libc_free_0(v46);
  a1[1] = m128i_i64;
  *a1 = v47;
  a1[2] = v44;
  return a1;
}
