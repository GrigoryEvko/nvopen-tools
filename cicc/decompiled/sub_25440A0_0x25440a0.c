// Function: sub_25440A0
// Address: 0x25440a0
//
__int64 __fastcall sub_25440A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  const __m128i *v8; // r14
  const __m128i *v10; // r12
  unsigned __int64 *v11; // r9
  __int64 v12; // rsi
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rsi
  size_t v16; // rdx
  unsigned __int64 *v17; // r15
  size_t v18; // rdx
  char *v19; // rsi
  _BYTE *v20; // r12
  size_t v21; // r14
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rax
  const __m128i *v24; // rcx
  unsigned __int64 v25; // r8
  const __m128i *v26; // rdx
  __m128i *v27; // rcx
  const __m128i *v28; // rax
  unsigned __int64 *v29; // [rsp+8h] [rbp-88h]
  int v30; // [rsp+8h] [rbp-88h]
  __int64 *v31; // [rsp+10h] [rbp-80h]
  __int64 v33; // [rsp+28h] [rbp-68h] BYREF
  void *base; // [rsp+30h] [rbp-60h] BYREF
  __int64 v35; // [rsp+38h] [rbp-58h]
  _BYTE *v36; // [rsp+40h] [rbp-50h] BYREF
  size_t v37; // [rsp+48h] [rbp-48h]
  _BYTE v38[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = 1;
  if ( *(_BYTE *)(a1 + 96) )
    return v6;
  v8 = *(const __m128i **)(a1 + 152);
  v31 = (__int64 *)(a1 + 72);
  v10 = &v8[*(unsigned int *)(a1 + 168)];
  if ( !*(_DWORD *)(a1 + 160) || v8 == v10 )
  {
LABEL_4:
    HIDWORD(v35) = 0;
    base = &v36;
LABEL_5:
    LODWORD(v35) = 0;
    v11 = (unsigned __int64 *)&v36;
    v12 = 0;
    goto LABEL_6;
  }
  while ( v8->m128i_i64[0] == -1 || v8->m128i_i64[0] == -2 )
  {
    if ( v10 == ++v8 )
      goto LABEL_4;
  }
  v35 = 0;
  base = &v36;
  if ( v10 == v8 )
    goto LABEL_5;
  v24 = v8;
  v25 = 0;
  while ( 1 )
  {
    v26 = v24 + 1;
    if ( v10 == &v24[1] )
      break;
    while ( 1 )
    {
      v24 = v26;
      if ( v26->m128i_i64[0] < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v26 )
        goto LABEL_29;
    }
    ++v25;
    if ( v10 == v26 )
      goto LABEL_30;
  }
LABEL_29:
  ++v25;
LABEL_30:
  v30 = v25;
  sub_C8D5F0((__int64)&base, &v36, v25, 0x10u, v25, a6);
  v27 = (__m128i *)((char *)base + 16 * (unsigned int)v35);
  do
  {
    if ( v27 )
      *v27 = _mm_loadu_si128(v8);
    v28 = v8 + 1;
    if ( v10 == &v8[1] )
      break;
    while ( 1 )
    {
      v8 = v28;
      if ( v28->m128i_i64[0] < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v28 )
        goto LABEL_36;
    }
    ++v27;
  }
  while ( v10 != v28 );
LABEL_36:
  v11 = (unsigned __int64 *)base;
  LODWORD(v35) = v30 + v35;
  v12 = 2LL * (unsigned int)v35;
  if ( (unsigned int)v35 > 1uLL )
  {
    qsort(base, (v12 * 8) >> 4, 0x10u, (__compar_fn_t)sub_A16990);
    v11 = (unsigned __int64 *)base;
    v12 = 2LL * (unsigned int)v35;
  }
LABEL_6:
  v13 = &v11[v12];
  v37 = 0;
  v36 = v38;
  v38[0] = 0;
  if ( &v11[v12] == v11 )
  {
    v21 = 0;
    v20 = v38;
  }
  else
  {
    v14 = v11;
    v15 = ((v12 * 8) >> 4) - 1;
    do
    {
      v15 += v14[1];
      v14 += 2;
    }
    while ( v13 != v14 );
    v29 = v11;
    sub_2240E30((__int64)&v36, v15);
    v16 = v29[1];
    if ( v16 > 0x3FFFFFFFFFFFFFFFLL - v37 )
      goto LABEL_43;
    sub_2241490((unsigned __int64 *)&v36, (char *)*v29, v16);
    v17 = v29 + 2;
    if ( v13 != v29 + 2 )
    {
      while ( v37 != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)&v36, ",", 1u);
        v18 = v17[1];
        v19 = (char *)*v17;
        if ( v18 > 0x3FFFFFFFFFFFFFFFLL - v37 )
          break;
        v17 += 2;
        sub_2241490((unsigned __int64 *)&v36, v19, v18);
        if ( v13 == v17 )
          goto LABEL_14;
      }
LABEL_43:
      sub_4262D8((__int64)"basic_string::append");
    }
LABEL_14:
    v20 = v36;
    v21 = v37;
  }
  v22 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v22 = *(_QWORD *)(v22 + 24);
  v23 = (_QWORD *)sub_BD5C60(v22);
  v33 = sub_A78730(v23, "llvm.assume", qword_49D3AE8, v20, v21);
  v6 = sub_2516380(a2, v31, (__int64)&v33, 1, 1);
  sub_2240A30((unsigned __int64 *)&v36);
  if ( base != &v36 )
    _libc_free((unsigned __int64)base);
  return v6;
}
