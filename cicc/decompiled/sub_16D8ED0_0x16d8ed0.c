// Function: sub_16D8ED0
// Address: 0x16d8ed0
//
__int64 *__fastcall sub_16D8ED0(__int64 *a1, char *a2, const __m128i *a3, __int64 a4, __int64 a5)
{
  const __m128i *v5; // r14
  char *v7; // rbx
  char *v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  bool v11; // cf
  unsigned __int64 v12; // rax
  signed __int64 v13; // r8
  __int64 m128i_i64; // r15
  bool v15; // zf
  __m128i *v16; // r8
  __m128i *v17; // r12
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  _BYTE *v20; // rsi
  __int64 v21; // rdx
  const __m128i *v22; // r12
  __m128i *i; // r15
  __m128i v24; // xmm3
  _BYTE *v25; // rsi
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __int64 v28; // rdx
  _BYTE *v29; // rsi
  __int64 *v30; // rdi
  char *j; // r13
  char *v32; // rdi
  char *v33; // rdi
  __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // [rsp+0h] [rbp-70h]
  const __m128i *v38; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+30h] [rbp-40h]
  char *v43; // [rsp+38h] [rbp-38h]

  v5 = (const __m128i *)a2;
  v7 = (char *)a1[1];
  v8 = (char *)*a1;
  v43 = (char *)*a1;
  v9 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v7[-*a1] >> 5);
  if ( v9 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xAAAAAAAAAAAAAAABLL * ((v7 - v8) >> 5);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x5555555555555555LL * ((v7 - v8) >> 5);
  v13 = a2 - v43;
  if ( v11 )
  {
    v35 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v12 )
    {
      v40 = 0;
      m128i_i64 = 96;
      v42 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x155555555555555LL )
      v12 = 0x155555555555555LL;
    v35 = 96 * v12;
  }
  v37 = a4;
  v38 = a3;
  v36 = sub_22077B0(v35);
  v13 = a2 - v43;
  a3 = v38;
  v42 = v36;
  a4 = v37;
  v40 = v36 + v35;
  m128i_i64 = v36 + 96;
LABEL_7:
  v15 = v42 + v13 == 0;
  v16 = (__m128i *)(v42 + v13);
  v17 = v16;
  if ( !v15 )
  {
    v18 = _mm_loadu_si128(a3);
    v19 = _mm_loadu_si128(a3 + 1);
    v20 = *(_BYTE **)a4;
    v16[2].m128i_i64[0] = (__int64)v16[3].m128i_i64;
    v21 = *(_QWORD *)(a4 + 8);
    *v16 = v18;
    v16[1] = v19;
    sub_16D5EB0(v16[2].m128i_i64, v20, (__int64)&v20[v21]);
    v17[4].m128i_i64[0] = (__int64)v17[5].m128i_i64;
    sub_16D5EB0(v17[4].m128i_i64, *(_BYTE **)a5, *(_QWORD *)a5 + *(_QWORD *)(a5 + 8));
  }
  v22 = (const __m128i *)v43;
  if ( a2 != v43 )
  {
    for ( i = (__m128i *)v42; ; i += 6 )
    {
      if ( i )
      {
        *i = _mm_loadu_si128(v22);
        v24 = _mm_loadu_si128(v22 + 1);
        i[2].m128i_i64[0] = (__int64)i[3].m128i_i64;
        i[1] = v24;
        sub_16D5EB0(i[2].m128i_i64, (_BYTE *)v22[2].m128i_i64[0], v22[2].m128i_i64[0] + v22[2].m128i_i64[1]);
        i[4].m128i_i64[0] = (__int64)i[5].m128i_i64;
        sub_16D5EB0(i[4].m128i_i64, (_BYTE *)v22[4].m128i_i64[0], v22[4].m128i_i64[0] + v22[4].m128i_i64[1]);
      }
      v22 += 6;
      if ( a2 == (char *)v22 )
        break;
    }
    m128i_i64 = (__int64)i[12].m128i_i64;
  }
  if ( a2 != v7 )
  {
    do
    {
      v25 = (_BYTE *)v5[2].m128i_i64[0];
      v5 += 6;
      v26 = _mm_loadu_si128(v5 - 6);
      v27 = _mm_loadu_si128(v5 - 5);
      *(_QWORD *)(m128i_i64 + 32) = m128i_i64 + 48;
      v28 = v5[-4].m128i_i64[1];
      *(__m128i *)m128i_i64 = v26;
      *(__m128i *)(m128i_i64 + 16) = v27;
      sub_16D5EB0((__int64 *)(m128i_i64 + 32), v25, (__int64)&v25[v28]);
      v29 = (_BYTE *)v5[-2].m128i_i64[0];
      v30 = (__int64 *)(m128i_i64 + 64);
      *(_QWORD *)(m128i_i64 + 64) = m128i_i64 + 80;
      m128i_i64 += 96;
      sub_16D5EB0(v30, v29, (__int64)&v29[v5[-2].m128i_i64[1]]);
    }
    while ( v7 != (char *)v5 );
  }
  for ( j = v43; v7 != j; j += 96 )
  {
    v32 = (char *)*((_QWORD *)j + 8);
    if ( v32 != j + 80 )
      j_j___libc_free_0(v32, *((_QWORD *)j + 10) + 1LL);
    v33 = (char *)*((_QWORD *)j + 4);
    if ( v33 != j + 48 )
      j_j___libc_free_0(v33, *((_QWORD *)j + 6) + 1LL);
  }
  if ( v43 )
    j_j___libc_free_0(v43, a1[2] - (_QWORD)v43);
  *a1 = v42;
  a1[1] = m128i_i64;
  a1[2] = v40;
  return a1;
}
