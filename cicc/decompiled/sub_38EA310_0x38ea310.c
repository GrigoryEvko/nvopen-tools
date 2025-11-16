// Function: sub_38EA310
// Address: 0x38ea310
//
__int64 *__fastcall sub_38EA310(__int64 *a1, const __m128i *a2, const __m128i *a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rcx
  bool v6; // cf
  unsigned __int64 v7; // rax
  signed __int64 v8; // rsi
  __int64 m128i_i64; // rcx
  __m128i *v10; // rax
  __int64 v11; // rsi
  __m128i v12; // xmm2
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int16 v15; // dx
  const __m128i *v16; // r15
  __m128i *i; // r14
  __int8 v18; // dl
  __int64 v19; // r12
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rdi
  const __m128i *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  __m128i v25; // xmm0
  __int64 v26; // rsi
  unsigned __int64 v27; // rdi
  unsigned __int64 v29; // rcx
  const __m128i *v30; // [rsp+8h] [rbp-68h]
  unsigned __int64 v31; // [rsp+18h] [rbp-58h]
  unsigned __int64 v32; // [rsp+18h] [rbp-58h]
  const __m128i *v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h]
  const __m128i *v37; // [rsp+38h] [rbp-38h]

  v37 = (const __m128i *)a1[1];
  v34 = (const __m128i *)*a1;
  v3 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v37->m128i_i64 - *a1) >> 4);
  if ( v3 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v4 = 1;
  if ( v3 )
    v4 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v37->m128i_i64 - *a1) >> 4);
  v6 = __CFADD__(v4, v3);
  v7 = v4 - 0x5555555555555555LL * (((__int64)v37->m128i_i64 - *a1) >> 4);
  v8 = (char *)a2 - (char *)v34;
  if ( v6 )
  {
    v29 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v7 )
    {
      v31 = 0;
      m128i_i64 = 48;
      v36 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0x2AAAAAAAAAAAAAALL )
      v7 = 0x2AAAAAAAAAAAAAALL;
    v29 = 48 * v7;
  }
  v30 = a3;
  v32 = v29;
  v36 = sub_22077B0(v29);
  a3 = v30;
  v31 = v36 + v32;
  m128i_i64 = v36 + 48;
LABEL_7:
  v10 = (__m128i *)(v36 + v8);
  if ( v36 + v8 )
  {
    v11 = a3[1].m128i_i64[0];
    v12 = _mm_loadu_si128(a3);
    a3[1].m128i_i64[0] = 0;
    v10[1].m128i_i64[0] = v11;
    v13 = a3[1].m128i_i64[1];
    a3[1].m128i_i64[1] = 0;
    v10[1].m128i_i64[1] = v13;
    v14 = a3[2].m128i_i64[0];
    a3[2].m128i_i64[0] = 0;
    v15 = a3[2].m128i_i16[4];
    v10[2].m128i_i64[0] = v14;
    v10[2].m128i_i16[4] = v15;
    *v10 = v12;
  }
  v16 = v34;
  if ( a2 != v34 )
  {
    for ( i = (__m128i *)v36; ; i += 3 )
    {
      if ( i )
      {
        *i = _mm_loadu_si128(v16);
        i[1].m128i_i64[0] = v16[1].m128i_i64[0];
        i[1].m128i_i64[1] = v16[1].m128i_i64[1];
        i[2].m128i_i64[0] = v16[2].m128i_i64[0];
        v18 = v16[2].m128i_i8[8];
        v16[2].m128i_i64[0] = 0;
        v16[1].m128i_i64[1] = 0;
        v16[1].m128i_i64[0] = 0;
        i[2].m128i_i8[8] = v18;
        i[2].m128i_i8[9] = v16[2].m128i_i8[9];
      }
      v19 = v16[1].m128i_i64[1];
      v20 = v16[1].m128i_u64[0];
      if ( v19 != v20 )
      {
        do
        {
          if ( *(_DWORD *)(v20 + 32) > 0x40u )
          {
            v21 = *(_QWORD *)(v20 + 24);
            if ( v21 )
              j_j___libc_free_0_0(v21);
          }
          v20 += 40LL;
        }
        while ( v19 != v20 );
        v20 = v16[1].m128i_u64[0];
      }
      if ( v20 )
        j_j___libc_free_0(v20);
      v16 += 3;
      if ( v16 == a2 )
        break;
    }
    m128i_i64 = (__int64)i[6].m128i_i64;
  }
  if ( a2 != v37 )
  {
    v22 = a2;
    v23 = m128i_i64;
    do
    {
      v24 = v22[1].m128i_i64[0];
      v25 = _mm_loadu_si128(v22);
      v23 += 48;
      v22 += 3;
      *(_QWORD *)(v23 - 32) = v24;
      v26 = v22[-2].m128i_i64[1];
      *(__m128i *)(v23 - 48) = v25;
      *(_QWORD *)(v23 - 24) = v26;
      *(_QWORD *)(v23 - 16) = v22[-1].m128i_i64[0];
      *(_BYTE *)(v23 - 8) = v22[-1].m128i_i8[8];
      *(_BYTE *)(v23 - 7) = v22[-1].m128i_i8[9];
    }
    while ( v22 != v37 );
    m128i_i64 += 16
               * (3
                * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v22 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v34 )
  {
    v27 = (unsigned __int64)v34;
    v35 = m128i_i64;
    j_j___libc_free_0(v27);
    m128i_i64 = v35;
  }
  *a1 = v36;
  a1[1] = m128i_i64;
  a1[2] = v31;
  return a1;
}
