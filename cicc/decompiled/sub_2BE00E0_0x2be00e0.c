// Function: sub_2BE00E0
// Address: 0x2be00e0
//
unsigned __int64 __fastcall sub_2BE00E0(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r10
  const __m128i *v4; // r15
  const __m128i *v6; // r13
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // r9
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int8 *v13; // rdx
  __int64 m128i_i64; // r12
  __m128i *v15; // rax
  __m128i v16; // xmm7
  bool v17; // zf
  __m128i v18; // xmm6
  __m128i *v19; // r12
  const __m128i *v20; // rbx
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  unsigned __int64 i; // rbx
  void (__fastcall *v24)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v25; // rdi
  void (__fastcall *v27)(__m128i *, const __m128i *, __int64); // rax
  void (__fastcall *v28)(__int64, const __m128i *, __int64); // rax
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  const __m128i *v34; // [rsp+10h] [rbp-50h]
  unsigned __int64 v35; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v36; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v37; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v39; // [rsp+20h] [rbp-40h]
  __m128i *v40; // [rsp+28h] [rbp-38h]

  v3 = a2;
  v4 = a2;
  v6 = (const __m128i *)a1[1];
  v7 = *a1;
  v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v6->m128i_i64 - *a1) >> 4);
  if ( v8 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  v10 = a1;
  if ( v8 )
    v9 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[1] - *a1) >> 4);
  v11 = __CFADD__(v9, v8);
  v12 = v9 - 0x5555555555555555LL * ((__int64)(a1[1] - *a1) >> 4);
  v13 = &a2->m128i_i8[-v7];
  if ( v11 )
  {
    v29 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_34:
    v30 = sub_22077B0(v29);
    v13 = &a2->m128i_i8[-v7];
    v10 = a1;
    v40 = (__m128i *)v30;
    v3 = a2;
    v35 = v30 + v29;
    m128i_i64 = v30 + 48;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x2AAAAAAAAAAAAAALL )
      v12 = 0x2AAAAAAAAAAAAAALL;
    v29 = 48 * v12;
    goto LABEL_34;
  }
  v35 = 0;
  m128i_i64 = 48;
  v40 = 0;
LABEL_7:
  v15 = (__m128i *)&v13[(_QWORD)v40];
  if ( &v13[(_QWORD)v40] )
  {
    v16 = _mm_loadu_si128(a3 + 1);
    v17 = a3->m128i_i32[0] == 11;
    *v15 = _mm_loadu_si128(a3);
    v18 = _mm_loadu_si128(a3 + 2);
    v15[1] = v16;
    v15[2] = v18;
    if ( v17 )
    {
      v31 = a3[2].m128i_i64[0];
      v32 = v15[2].m128i_i64[1];
      a3[2].m128i_i64[0] = 0;
      v15[2].m128i_i64[0] = v31;
      v33 = a3[2].m128i_i64[1];
      a3[2].m128i_i64[1] = v32;
      v15[2].m128i_i64[1] = v33;
    }
  }
  if ( v3 != (const __m128i *)v7 )
  {
    v19 = v40;
    v20 = (const __m128i *)v7;
    while ( 1 )
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v20);
        v19[1] = _mm_loadu_si128(v20 + 1);
        v19[2] = _mm_loadu_si128(v20 + 2);
        if ( v20->m128i_i32[0] == 11 )
        {
          v19[2].m128i_i64[0] = 0;
          v27 = (void (__fastcall *)(__m128i *, const __m128i *, __int64))v20[2].m128i_i64[0];
          if ( v27 )
          {
            v34 = v3;
            v38 = v10;
            v27(v19 + 1, v20 + 1, 2);
            v3 = v34;
            v10 = v38;
            v19[2].m128i_i64[1] = v20[2].m128i_i64[1];
            v19[2].m128i_i64[0] = v20[2].m128i_i64[0];
          }
        }
      }
      v20 += 3;
      if ( v3 == v20 )
        break;
      v19 += 3;
    }
    m128i_i64 = (__int64)v19[6].m128i_i64;
  }
  if ( v3 != v6 )
  {
    do
    {
      v21 = _mm_loadu_si128(v4 + 1);
      v22 = _mm_loadu_si128(v4 + 2);
      v17 = v4->m128i_i32[0] == 11;
      *(__m128i *)m128i_i64 = _mm_loadu_si128(v4);
      *(__m128i *)(m128i_i64 + 16) = v21;
      *(__m128i *)(m128i_i64 + 32) = v22;
      if ( v17 )
      {
        *(_QWORD *)(m128i_i64 + 32) = 0;
        v28 = (void (__fastcall *)(__int64, const __m128i *, __int64))v4[2].m128i_i64[0];
        if ( v28 )
        {
          v39 = v10;
          v28(m128i_i64 + 16, v4 + 1, 2);
          v10 = v39;
          *(_QWORD *)(m128i_i64 + 40) = v4[2].m128i_i64[1];
          *(_QWORD *)(m128i_i64 + 32) = v4[2].m128i_i64[0];
        }
      }
      v4 += 3;
      m128i_i64 += 48;
    }
    while ( v6 != v4 );
  }
  for ( i = v7; v6 != (const __m128i *)i; v10 = v36 )
  {
    while ( 1 )
    {
      if ( *(_DWORD *)i == 11 )
      {
        v24 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(i + 32);
        if ( v24 )
          break;
      }
      i += 48LL;
      if ( v6 == (const __m128i *)i )
        goto LABEL_26;
    }
    v25 = i + 16;
    i += 48LL;
    v36 = v10;
    v24(v25, v25, 3);
  }
LABEL_26:
  if ( v7 )
  {
    v37 = v10;
    j_j___libc_free_0(v7);
    v10 = v37;
  }
  v10[1] = m128i_i64;
  *v10 = (unsigned __int64)v40;
  v10[2] = v35;
  return v35;
}
