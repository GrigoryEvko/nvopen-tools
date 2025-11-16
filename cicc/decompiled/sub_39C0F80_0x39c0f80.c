// Function: sub_39C0F80
// Address: 0x39c0f80
//
unsigned __int64 __fastcall sub_39C0F80(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3, __int64 a4)
{
  const __m128i *v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  const __m128i *v9; // r8
  const __m128i *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int8 *v13; // r9
  __int64 m128i_i64; // rbx
  __m128i *v15; // rdi
  __m128i v16; // xmm2
  __m128i *v17; // rbx
  const __m128i *v18; // rax
  __m128i *v19; // rsi
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  __int32 v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 i; // r14
  unsigned __int64 v26; // rdi
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  const __m128i *v30; // [rsp+8h] [rbp-58h]
  unsigned __int64 v31; // [rsp+10h] [rbp-50h]
  const __m128i *v32; // [rsp+18h] [rbp-48h]
  const __m128i *v33; // [rsp+20h] [rbp-40h]
  const __m128i *v34; // [rsp+20h] [rbp-40h]
  unsigned __int64 v35; // [rsp+28h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 5);
  if ( v7 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 5);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x5555555555555555LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 5);
  v13 = &a2->m128i_i8[-v6];
  if ( v11 )
  {
    v28 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_31:
    v30 = a3;
    v29 = sub_22077B0(v28);
    v13 = &a2->m128i_i8[-v6];
    v9 = a2;
    v35 = v29;
    a3 = v30;
    v31 = v29 + v28;
    m128i_i64 = v29 + 96;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x155555555555555LL )
      v12 = 0x155555555555555LL;
    v28 = 96 * v12;
    goto LABEL_31;
  }
  v31 = 0;
  m128i_i64 = 96;
  v35 = 0;
LABEL_7:
  v15 = (__m128i *)&v13[v35];
  if ( &v13[v35] )
  {
    v16 = _mm_loadu_si128(a3);
    a4 = a3[1].m128i_u32[2];
    v15[1].m128i_i64[0] = (__int64)v15[2].m128i_i64;
    v15[1].m128i_i64[1] = 0x400000000LL;
    *v15 = v16;
    if ( (_DWORD)a4 )
    {
      v34 = v9;
      sub_39C0A30((__int64)v15[1].m128i_i64, (__int64)a3[1].m128i_i64, (__int64)a3, a4, (int)v9, (int)v13);
      v9 = v34;
    }
  }
  if ( v9 != (const __m128i *)v6 )
  {
    v17 = (__m128i *)v35;
    v18 = (const __m128i *)v6;
    while ( 1 )
    {
      if ( v17
        && (v20 = _mm_loadu_si128(v18),
            v17[1].m128i_i32[2] = 0,
            v17[1].m128i_i64[0] = (__int64)v17[2].m128i_i64,
            v17[1].m128i_i32[3] = 4,
            *v17 = v20,
            a3 = (const __m128i *)v18[1].m128i_u32[2],
            (_DWORD)a3) )
      {
        v32 = v9;
        v33 = v18;
        sub_39C0900((__int64)v17[1].m128i_i64, (__int64)v18[1].m128i_i64, (__int64)a3, a4, (int)v9, (int)v13);
        v9 = v32;
        v19 = v17 + 6;
        v18 = v33 + 6;
        if ( v32 == &v33[6] )
        {
LABEL_17:
          m128i_i64 = (__int64)v17[12].m128i_i64;
          break;
        }
      }
      else
      {
        v18 += 6;
        v19 = v17 + 6;
        if ( v9 == v18 )
          goto LABEL_17;
      }
      v17 = v19;
    }
  }
  if ( v9 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v21 = _mm_loadu_si128(v10);
        *(_DWORD *)(m128i_i64 + 24) = 0;
        *(_QWORD *)(m128i_i64 + 16) = m128i_i64 + 32;
        v22 = v10[1].m128i_i32[2];
        *(_DWORD *)(m128i_i64 + 28) = 4;
        *(__m128i *)m128i_i64 = v21;
        if ( v22 )
          break;
        v10 += 6;
        m128i_i64 += 96;
        if ( v5 == v10 )
          goto LABEL_23;
      }
      v23 = (__int64)v10[1].m128i_i64;
      v24 = m128i_i64 + 16;
      v10 += 6;
      m128i_i64 += 96;
      sub_39C0900(v24, v23, (__int64)a3, a4, (int)v9, (int)v13);
    }
    while ( v5 != v10 );
  }
LABEL_23:
  for ( i = v6; v5 != (const __m128i *)i; i += 96LL )
  {
    v26 = *(_QWORD *)(i + 16);
    if ( v26 != i + 32 )
      _libc_free(v26);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  a1[1] = m128i_i64;
  *a1 = v35;
  a1[2] = v31;
  return v31;
}
