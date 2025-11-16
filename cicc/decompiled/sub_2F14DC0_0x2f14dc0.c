// Function: sub_2F14DC0
// Address: 0x2f14dc0
//
unsigned __int64 *__fastcall sub_2F14DC0(unsigned __int64 *a1, const __m128i *a2, __m128i *a3)
{
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rsi
  __int64 m128i_i64; // rdx
  __m128i *v12; // rax
  __m128i *v13; // rdi
  __int64 v14; // rdi
  __m128i v15; // xmm4
  __m128i *v16; // rdi
  __int64 v17; // rdi
  __m128i v18; // xmm5
  __m128i *v19; // r15
  const __m128i *v20; // r14
  unsigned __int64 v21; // r13
  __m128i v22; // xmm1
  __int64 v23; // rsi
  __m128i v24; // xmm2
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  const __m128i *v27; // rsi
  const __m128i *v28; // rax
  const __m128i *v29; // rsi
  __m128i *v30; // rcx
  __int64 v31; // rdi
  __m128i v32; // xmm3
  __int64 v33; // rdi
  __int64 v34; // rdi
  __m128i v35; // xmm0
  const __m128i *v36; // rdi
  unsigned __int64 v38; // rdx
  __m128i *v39; // [rsp+0h] [rbp-60h]
  __int64 v40; // [rsp+8h] [rbp-58h]
  unsigned __int64 v41; // [rsp+10h] [rbp-50h]
  unsigned __int64 v42; // [rsp+10h] [rbp-50h]
  unsigned __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v44 = *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 5);
  if ( v5 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 5);
  v8 = __CFADD__(v6, v5);
  v9 = v6 - 0x5555555555555555LL * ((__int64)(v4 - *a1) >> 5);
  v10 = &a2->m128i_i8[-v44];
  if ( v8 )
  {
    v38 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v41 = 0;
      m128i_i64 = 96;
      v45 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x155555555555555LL )
      v9 = 0x155555555555555LL;
    v38 = 96 * v9;
  }
  v39 = a3;
  v42 = v38;
  v45 = sub_22077B0(v38);
  a3 = v39;
  v41 = v45 + v42;
  m128i_i64 = v45 + 96;
LABEL_7:
  v12 = (__m128i *)&v10[v45];
  if ( &v10[v45] )
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
    v12[3].m128i_i64[0] = (__int64)v12[4].m128i_i64;
    v12->m128i_i64[1] = v14;
    v16 = (__m128i *)a3[3].m128i_i64[0];
    a3->m128i_i64[1] = 0;
    a3[1].m128i_i8[0] = 0;
    v12[2] = v15;
    if ( v16 == &a3[4] )
    {
      v12[4] = _mm_loadu_si128(a3 + 4);
    }
    else
    {
      v12[3].m128i_i64[0] = (__int64)v16;
      v12[4].m128i_i64[0] = a3[4].m128i_i64[0];
    }
    v17 = a3[3].m128i_i64[1];
    v18 = _mm_loadu_si128(a3 + 5);
    a3[3].m128i_i64[0] = (__int64)a3[4].m128i_i64;
    a3[3].m128i_i64[1] = 0;
    v12[3].m128i_i64[1] = v17;
    a3[4].m128i_i8[0] = 0;
    v12[5] = v18;
  }
  if ( a2 != (const __m128i *)v44 )
  {
    v19 = (__m128i *)v45;
    v20 = (const __m128i *)(v44 + 16);
    v21 = v44 + 64;
    while ( 1 )
    {
      if ( v19 )
      {
        v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
        v27 = (const __m128i *)v20[-1].m128i_i64[0];
        if ( v27 == v20 )
        {
          v19[1] = _mm_loadu_si128(v20);
        }
        else
        {
          v19->m128i_i64[0] = (__int64)v27;
          v19[1].m128i_i64[0] = v20->m128i_i64[0];
        }
        v19->m128i_i64[1] = v20[-1].m128i_i64[1];
        v22 = _mm_loadu_si128(v20 + 1);
        v20[-1].m128i_i64[0] = (__int64)v20;
        v20[-1].m128i_i64[1] = 0;
        v20->m128i_i8[0] = 0;
        v19[3].m128i_i64[0] = (__int64)v19[4].m128i_i64;
        v19[2] = v22;
        v23 = v20[2].m128i_i64[0];
        if ( v23 == v21 )
        {
          v19[4] = _mm_loadu_si128(v20 + 3);
        }
        else
        {
          v19[3].m128i_i64[0] = v23;
          v19[4].m128i_i64[0] = v20[3].m128i_i64[0];
        }
        v19[3].m128i_i64[1] = v20[2].m128i_i64[1];
        v24 = _mm_loadu_si128(v20 + 4);
        v20[2].m128i_i64[0] = v21;
        v20[2].m128i_i64[1] = 0;
        v20[3].m128i_i8[0] = 0;
        v19[5] = v24;
      }
      v25 = v20[2].m128i_u64[0];
      if ( v25 != v21 )
        j_j___libc_free_0(v25);
      v26 = v20[-1].m128i_u64[0];
      if ( (const __m128i *)v26 != v20 )
        j_j___libc_free_0(v26);
      v21 += 96LL;
      if ( a2 == &v20[5] )
        break;
      v20 += 6;
      v19 += 6;
    }
    m128i_i64 = (__int64)v19[12].m128i_i64;
  }
  if ( a2 != (const __m128i *)v4 )
  {
    v28 = a2 + 1;
    v29 = a2;
    v30 = (__m128i *)m128i_i64;
    do
    {
      v30->m128i_i64[0] = (__int64)v30[1].m128i_i64;
      v36 = (const __m128i *)v28[-1].m128i_i64[0];
      if ( v36 == v28 )
      {
        v30[1] = _mm_loadu_si128(v28);
      }
      else
      {
        v30->m128i_i64[0] = (__int64)v36;
        v30[1].m128i_i64[0] = v28->m128i_i64[0];
      }
      v31 = v28[-1].m128i_i64[1];
      v32 = _mm_loadu_si128(v28 + 1);
      v28->m128i_i8[0] = 0;
      v28[-1].m128i_i64[0] = (__int64)v28;
      v30->m128i_i64[1] = v31;
      v30[3].m128i_i64[0] = (__int64)v30[4].m128i_i64;
      v33 = v28[2].m128i_i64[0];
      v28[-1].m128i_i64[1] = 0;
      v30[2] = v32;
      if ( (const __m128i *)v33 == &v29[4] )
      {
        v30[4] = _mm_loadu_si128(v28 + 3);
      }
      else
      {
        v30[3].m128i_i64[0] = v33;
        v30[4].m128i_i64[0] = v28[3].m128i_i64[0];
      }
      v34 = v28[2].m128i_i64[1];
      v29 += 6;
      v30 += 6;
      v28 += 6;
      v35 = _mm_loadu_si128(v28 - 2);
      v30[-3].m128i_i64[1] = v34;
      v30[-1] = v35;
    }
    while ( v29 != (const __m128i *)v4 );
    m128i_i64 += 32
               * (3
                * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v29 - (char *)a2 - 96) >> 5)) & 0x7FFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v44 )
  {
    v40 = m128i_i64;
    j_j___libc_free_0(v44);
    m128i_i64 = v40;
  }
  *a1 = v45;
  a1[1] = m128i_i64;
  a1[2] = v41;
  return a1;
}
