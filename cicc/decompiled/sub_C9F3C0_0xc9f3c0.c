// Function: sub_C9F3C0
// Address: 0xc9f3c0
//
__m128i **__fastcall sub_C9F3C0(__m128i **a1, __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5)
{
  __m128i *v5; // r14
  __m128i *v7; // rbx
  __m128i *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  bool v11; // cf
  unsigned __int64 v12; // rax
  signed __int64 v13; // r8
  __int64 m128i_i64; // r15
  bool v15; // zf
  __m128i *v16; // r8
  __m128i *v17; // r12
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  __int64 v20; // rdx
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __m128i *v23; // r12
  __m128i *i; // r15
  __int64 v25; // rdx
  _BYTE *v26; // rsi
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __int64 v29; // rdx
  _BYTE *v30; // rsi
  __int64 *v31; // rdi
  __m128i *j; // r13
  unsigned __int64 *v33; // rdi
  unsigned __int64 *v34; // rdi
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // [rsp+0h] [rbp-70h]
  const __m128i *v39; // [rsp+8h] [rbp-68h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  __m128i *v43; // [rsp+30h] [rbp-40h]
  __m128i *v44; // [rsp+38h] [rbp-38h]

  v5 = a2;
  v7 = a1[1];
  v8 = *a1;
  v44 = *a1;
  v9 = 0x4EC4EC4EC4EC4EC5LL * (((char *)v7 - (char *)*a1) >> 3);
  if ( v9 == 0x13B13B13B13B13BLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0x4EC4EC4EC4EC4EC5LL * (((char *)v7 - (char *)v8) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 + v9;
  v13 = (char *)a2 - (char *)v44;
  if ( v11 )
  {
    v36 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v41 = 0;
      m128i_i64 = 104;
      v43 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x13B13B13B13B13BLL )
      v12 = 0x13B13B13B13B13BLL;
    v36 = 104 * v12;
  }
  v38 = a4;
  v39 = a3;
  v37 = sub_22077B0(v36);
  v13 = (char *)a2 - (char *)v44;
  a3 = v39;
  v43 = (__m128i *)v37;
  a4 = v38;
  v41 = v37 + v36;
  m128i_i64 = v37 + 104;
LABEL_7:
  v15 = &v43->m128i_i8[v13] == 0;
  v16 = (__m128i *)((char *)v43 + v13);
  v17 = v16;
  if ( !v15 )
  {
    v18 = _mm_loadu_si128(a3);
    v19 = _mm_loadu_si128(a3 + 1);
    v20 = a3[2].m128i_i64[0];
    v21 = *(_BYTE **)a4;
    *v16 = v18;
    v16[2].m128i_i64[0] = v20;
    v16[2].m128i_i64[1] = (__int64)&v16[3].m128i_i64[1];
    v22 = *(_QWORD *)(a4 + 8);
    v16[1] = v19;
    sub_C9CAB0(&v16[2].m128i_i64[1], v21, (__int64)&v21[v22]);
    v17[4].m128i_i64[1] = (__int64)&v17[5].m128i_i64[1];
    sub_C9CAB0(&v17[4].m128i_i64[1], *(_BYTE **)a5, *(_QWORD *)a5 + *(_QWORD *)(a5 + 8));
  }
  v23 = v44;
  if ( a2 != v44 )
  {
    for ( i = v43; ; i = (__m128i *)((char *)i + 104) )
    {
      if ( i )
      {
        *i = _mm_loadu_si128(v23);
        i[1] = _mm_loadu_si128(v23 + 1);
        i[2].m128i_i64[0] = v23[2].m128i_i64[0];
        i[2].m128i_i64[1] = (__int64)&i[3].m128i_i64[1];
        sub_C9CAB0(&i[2].m128i_i64[1], (_BYTE *)v23[2].m128i_i64[1], v23[2].m128i_i64[1] + v23[3].m128i_i64[0]);
        i[4].m128i_i64[1] = (__int64)&i[5].m128i_i64[1];
        sub_C9CAB0(&i[4].m128i_i64[1], (_BYTE *)v23[4].m128i_i64[1], v23[4].m128i_i64[1] + v23[5].m128i_i64[0]);
      }
      v23 = (__m128i *)((char *)v23 + 104);
      if ( a2 == v23 )
        break;
    }
    m128i_i64 = (__int64)i[13].m128i_i64;
  }
  if ( a2 != v7 )
  {
    do
    {
      v25 = v5[2].m128i_i64[0];
      v26 = (_BYTE *)v5[2].m128i_i64[1];
      v5 = (__m128i *)((char *)v5 + 104);
      v27 = _mm_loadu_si128((__m128i *)((char *)v5 - 104));
      v28 = _mm_loadu_si128((__m128i *)((char *)v5 - 88));
      *(_QWORD *)(m128i_i64 + 32) = v25;
      *(_QWORD *)(m128i_i64 + 40) = m128i_i64 + 56;
      v29 = v5[-4].m128i_i64[1];
      *(__m128i *)m128i_i64 = v27;
      *(__m128i *)(m128i_i64 + 16) = v28;
      sub_C9CAB0((__int64 *)(m128i_i64 + 40), v26, (__int64)&v26[v29]);
      v30 = (_BYTE *)v5[-2].m128i_i64[0];
      v31 = (__int64 *)(m128i_i64 + 72);
      *(_QWORD *)(m128i_i64 + 72) = m128i_i64 + 88;
      m128i_i64 += 104;
      sub_C9CAB0(v31, v30, (__int64)&v30[v5[-2].m128i_i64[1]]);
    }
    while ( v7 != v5 );
  }
  for ( j = v44; v7 != j; j = (__m128i *)((char *)j + 104) )
  {
    v33 = (unsigned __int64 *)j[4].m128i_i64[1];
    if ( v33 != &j[5].m128i_u64[1] )
      j_j___libc_free_0(v33, j[5].m128i_i64[1] + 1);
    v34 = (unsigned __int64 *)j[2].m128i_i64[1];
    if ( v34 != &j[3].m128i_u64[1] )
      j_j___libc_free_0(v34, j[3].m128i_i64[1] + 1);
  }
  if ( v44 )
    j_j___libc_free_0(v44, (char *)a1[2] - (char *)v44);
  *a1 = v43;
  a1[1] = (__m128i *)m128i_i64;
  a1[2] = (__m128i *)v41;
  return a1;
}
