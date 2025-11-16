// Function: sub_39DBAA0
// Address: 0x39dbaa0
//
void __fastcall sub_39DBAA0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v4; // r14
  const __m128i *v5; // rsi
  __int64 v6; // rbx
  _QWORD *v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r11
  __m128i *v14; // r12
  const __m128i *v15; // rbx
  __int64 m128i_i64; // rax
  __m128i v17; // xmm0
  __int64 v18; // rdx
  __m128i v19; // xmm1
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  const __m128i *v22; // rdx
  unsigned __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // [rsp-58h] [rbp-58h]
  __int64 v26; // [rsp-50h] [rbp-50h]
  unsigned __int64 v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-40h] [rbp-40h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = (const __m128i *)*a1;
  v6 = v4 - *a1;
  v27 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 5);
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[2] - v4) >> 5) >= a2 )
  {
    v7 = (_QWORD *)a1[1];
    v8 = a2;
    do
    {
      if ( v7 )
      {
        memset(v7, 0, 0x60u);
        *v7 = v7 + 2;
        v7[6] = v7 + 8;
      }
      v7 += 12;
      --v8;
    }
    while ( v8 );
    a1[1] = v4 + 96 * a2;
    return;
  }
  if ( 0x155555555555555LL - v27 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v9 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 5);
  if ( v27 < a2 )
    v9 = a2;
  v10 = __CFADD__(v27, v9);
  v11 = v27 + v9;
  if ( v10 )
  {
    v23 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v25 = 0;
      v26 = 0;
      goto LABEL_15;
    }
    if ( v11 > 0x155555555555555LL )
      v11 = 0x155555555555555LL;
    v23 = 96 * v11;
  }
  v24 = sub_22077B0(v23);
  v4 = a1[1];
  v5 = (const __m128i *)*a1;
  v26 = v24;
  v25 = v24 + v23;
LABEL_15:
  v12 = (_QWORD *)(v26 + v6);
  v13 = a2;
  do
  {
    if ( v12 )
    {
      memset(v12, 0, 0x60u);
      *v12 = v12 + 2;
      v12[6] = v12 + 8;
    }
    v12 += 12;
    --v13;
  }
  while ( v13 );
  if ( v5 != (const __m128i *)v4 )
  {
    v14 = (__m128i *)v26;
    v15 = v5 + 1;
    m128i_i64 = (__int64)v5[4].m128i_i64;
    while ( 1 )
    {
      if ( v14 )
      {
        v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
        v22 = (const __m128i *)v15[-1].m128i_i64[0];
        if ( v15 == v22 )
        {
          v14[1] = _mm_loadu_si128(v15);
        }
        else
        {
          v14->m128i_i64[0] = (__int64)v22;
          v14[1].m128i_i64[0] = v15->m128i_i64[0];
        }
        v14->m128i_i64[1] = v15[-1].m128i_i64[1];
        v17 = _mm_loadu_si128(v15 + 1);
        v15[-1].m128i_i64[0] = (__int64)v15;
        v15[-1].m128i_i64[1] = 0;
        v15->m128i_i8[0] = 0;
        v14[3].m128i_i64[0] = (__int64)v14[4].m128i_i64;
        v14[2] = v17;
        v18 = v15[2].m128i_i64[0];
        if ( m128i_i64 == v18 )
        {
          v14[4] = _mm_loadu_si128(v15 + 3);
        }
        else
        {
          v14[3].m128i_i64[0] = v18;
          v14[4].m128i_i64[0] = v15[3].m128i_i64[0];
        }
        v14[3].m128i_i64[1] = v15[2].m128i_i64[1];
        v19 = _mm_loadu_si128(v15 + 4);
        v15[2].m128i_i64[0] = m128i_i64;
        v15[2].m128i_i64[1] = 0;
        v15[3].m128i_i8[0] = 0;
        v14[5] = v19;
      }
      v20 = v15[2].m128i_u64[0];
      if ( m128i_i64 != v20 )
      {
        v28 = m128i_i64;
        j_j___libc_free_0(v20);
        m128i_i64 = v28;
      }
      v21 = v15[-1].m128i_u64[0];
      if ( v15 != (const __m128i *)v21 )
      {
        v29 = m128i_i64;
        j_j___libc_free_0(v21);
        m128i_i64 = v29;
      }
      v14 += 6;
      m128i_i64 += 96;
      if ( (const __m128i *)v4 == &v15[5] )
        break;
      v15 += 6;
    }
    v4 = *a1;
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = v26;
  a1[1] = v26 + 96 * (a2 + v27);
  a1[2] = v25;
}
