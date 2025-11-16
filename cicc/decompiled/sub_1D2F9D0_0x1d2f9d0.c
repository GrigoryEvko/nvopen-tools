// Function: sub_1D2F9D0
// Address: 0x1d2f9d0
//
__int64 __fastcall sub_1D2F9D0(_QWORD *a1, const char *a2, unsigned __int8 a3, __int64 a4, unsigned __int8 a5)
{
  size_t v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  size_t v8; // r12
  __m128i *v9; // rdx
  __int64 v10; // r13
  __m128i *v11; // r14
  size_t v12; // r15
  int v13; // eax
  __int64 v14; // rcx
  size_t v15; // rdx
  signed __int64 v16; // rax
  signed __int64 v17; // rax
  size_t v18; // rbx
  const void *v19; // r12
  _QWORD *m128i_i64; // r12
  size_t v21; // rbx
  const void *v22; // r13
  size_t v23; // rdx
  int v24; // eax
  signed __int64 v25; // rax
  signed __int64 v26; // rax
  __int64 v27; // r15
  __m128i *v28; // rax
  __m128i *v29; // rbx
  size_t v30; // r14
  unsigned __int8 v31; // r13
  __int64 v32; // rax
  _QWORD *v33; // rdx
  _QWORD *v34; // r15
  _BOOL8 v35; // rdi
  __int64 result; // rax
  __m128i *v37; // rdi
  size_t v38; // rbx
  void *v39; // r8
  const void *v40; // r9
  size_t v41; // rdx
  int v42; // eax
  int v43; // eax
  __int64 v44; // rbx
  __m128i *v45; // rdi
  __int64 v46; // r13
  __int128 v47; // rdi
  __int64 v48; // rax
  unsigned __int8 *v49; // rsi
  signed __int64 v50; // rax
  _QWORD *v55; // [rsp+30h] [rbp-A0h]
  size_t v58; // [rsp+48h] [rbp-88h]
  _QWORD *s1; // [rsp+50h] [rbp-80h]
  void *s1a; // [rsp+50h] [rbp-80h]
  size_t na; // [rsp+58h] [rbp-78h]
  size_t n; // [rsp+58h] [rbp-78h]
  const void *nb; // [rsp+58h] [rbp-78h]
  size_t nc; // [rsp+58h] [rbp-78h]
  size_t nd; // [rsp+58h] [rbp-78h]
  __int64 v66; // [rsp+68h] [rbp-68h] BYREF
  void *s2; // [rsp+70h] [rbp-60h] BYREF
  size_t v68; // [rsp+78h] [rbp-58h]
  __m128i v69; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int8 v70; // [rsp+90h] [rbp-40h]

  s2 = &v69;
  if ( !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v5 = strlen(a2);
  v66 = v5;
  v8 = v5;
  if ( v5 > 0xF )
  {
    s2 = (void *)sub_22409D0(&s2, &v66, 0);
    v37 = (__m128i *)s2;
    v69.m128i_i64[0] = v66;
    goto LABEL_54;
  }
  if ( v5 != 1 )
  {
    if ( !v5 )
    {
      v9 = &v69;
      goto LABEL_5;
    }
    v37 = &v69;
LABEL_54:
    memcpy(v37, a2, v8);
    v5 = v66;
    v9 = (__m128i *)s2;
    goto LABEL_5;
  }
  v69.m128i_i8[0] = *a2;
  v9 = &v69;
LABEL_5:
  v68 = v5;
  v9->m128i_i8[v5] = 0;
  v70 = a5;
  v10 = a1[105];
  v55 = a1 + 104;
  if ( !v10 )
  {
    m128i_i64 = a1 + 104;
    goto LABEL_41;
  }
  v11 = (__m128i *)s2;
  v12 = v68;
  s1 = a1 + 104;
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v10 + 40);
      v15 = v12;
      v19 = *(const void **)(v10 + 32);
      if ( v18 <= v12 )
        v15 = *(_QWORD *)(v10 + 40);
      if ( !v15 )
      {
        v16 = v18 - v12;
        if ( (__int64)(v18 - v12) >= 0x80000000LL )
          goto LABEL_13;
LABEL_9:
        v14 = 0xFFFFFFFF7FFFFFFFLL;
        if ( v16 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v16 < 0 )
          goto LABEL_26;
        if ( !v15 )
          goto LABEL_13;
        goto LABEL_12;
      }
      na = v15;
      v13 = memcmp(*(const void **)(v10 + 32), v11, v15);
      v15 = na;
      if ( v13 )
      {
        if ( v13 < 0 )
          goto LABEL_26;
      }
      else
      {
        v16 = v18 - v12;
        if ( (__int64)(v18 - v12) < 0x80000000LL )
          goto LABEL_9;
      }
LABEL_12:
      LODWORD(v17) = memcmp(v11, v19, v15);
      if ( (_DWORD)v17 )
        break;
LABEL_13:
      v14 = 0x80000000LL;
      v17 = v12 - v18;
      if ( (__int64)(v12 - v18) >= 0x80000000LL )
        goto LABEL_16;
      if ( v17 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        break;
LABEL_17:
      s1 = (_QWORD *)v10;
      v10 = *(_QWORD *)(v10 + 16);
      if ( !v10 )
        goto LABEL_27;
    }
    if ( (int)v17 < 0 )
      goto LABEL_17;
LABEL_16:
    if ( *(_BYTE *)(v10 + 64) >= a5 )
      goto LABEL_17;
LABEL_26:
    v10 = *(_QWORD *)(v10 + 24);
  }
  while ( v10 );
LABEL_27:
  m128i_i64 = s1;
  if ( v55 == s1 )
    goto LABEL_41;
  v21 = s1[5];
  v22 = (const void *)s1[4];
  v23 = v21;
  if ( v12 <= v21 )
    v23 = v12;
  if ( !v23 )
  {
    v25 = v12 - v21;
    if ( (__int64)(v12 - v21) <= 0x7FFFFFFF )
      goto LABEL_33;
    goto LABEL_37;
  }
  n = v23;
  v24 = memcmp(v11, (const void *)s1[4], v23);
  v23 = n;
  if ( v24 )
  {
    if ( v24 < 0 )
      goto LABEL_41;
    LODWORD(v26) = memcmp(v22, v11, n);
    if ( !(_DWORD)v26 )
      goto LABEL_37;
LABEL_39:
    if ( (int)v26 < 0 )
      goto LABEL_49;
LABEL_40:
    if ( *((_BYTE *)s1 + 64) <= a5 )
      goto LABEL_49;
LABEL_41:
    v27 = (__int64)m128i_i64;
    v28 = (__m128i *)sub_22077B0(80);
    v29 = v28 + 3;
    m128i_i64 = v28->m128i_i64;
    v28[2].m128i_i64[0] = (__int64)v28[3].m128i_i64;
    if ( s2 == &v69 )
    {
      v28[3] = _mm_load_si128(&v69);
    }
    else
    {
      v28[2].m128i_i64[0] = (__int64)s2;
      v28[3].m128i_i64[0] = v69.m128i_i64[0];
    }
    v30 = v68;
    v31 = v70;
    v69.m128i_i8[0] = 0;
    v68 = 0;
    v28[2].m128i_i64[1] = v30;
    s2 = &v69;
    v28[4].m128i_i8[0] = v31;
    v28[4].m128i_i64[1] = 0;
    v32 = sub_1D2F470(a1 + 103, v27, (__int64)v28[2].m128i_i64);
    v34 = v33;
    if ( !v33 )
    {
      v45 = (__m128i *)m128i_i64[4];
      if ( v29 != v45 )
      {
        nc = v32;
        j_j___libc_free_0(v45, m128i_i64[6] + 1LL);
        v32 = nc;
      }
      nd = v32;
      j_j___libc_free_0(m128i_i64, 80);
      m128i_i64 = (_QWORD *)nd;
      goto LABEL_48;
    }
    if ( v55 == v33 || v32 )
    {
      v35 = 1;
LABEL_47:
      sub_220F040(v35, m128i_i64, v34, v55);
      ++a1[108];
LABEL_48:
      v11 = (__m128i *)s2;
      goto LABEL_49;
    }
    v39 = (void *)v33[4];
    v40 = (const void *)m128i_i64[4];
    v41 = v33[5];
    v38 = v41;
    if ( v30 <= v41 )
      v41 = v30;
    if ( v41 )
    {
      v58 = v41;
      s1a = v39;
      nb = (const void *)m128i_i64[4];
      v42 = memcmp(nb, v39, v41);
      v40 = nb;
      v39 = s1a;
      v41 = v58;
      if ( v42 )
      {
        v35 = 1;
        if ( v42 < 0 )
          goto LABEL_47;
LABEL_65:
        v43 = memcmp(v39, v40, v41);
        if ( v43 )
          goto LABEL_69;
        goto LABEL_66;
      }
      v50 = v30 - v38;
      if ( (__int64)(v30 - v38) > 0x7FFFFFFF )
        goto LABEL_65;
    }
    else
    {
      v50 = v30 - v38;
      if ( (__int64)(v30 - v38) > 0x7FFFFFFF )
        goto LABEL_66;
    }
    v35 = 1;
    if ( v50 < (__int64)0xFFFFFFFF80000000LL || (int)v50 < 0 )
      goto LABEL_47;
    if ( v41 )
      goto LABEL_65;
LABEL_66:
    v44 = v38 - v30;
    if ( v44 > 0x7FFFFFFF )
      goto LABEL_70;
    if ( v44 < (__int64)0xFFFFFFFF80000000LL )
    {
      v35 = 0;
      goto LABEL_47;
    }
    v43 = v44;
LABEL_69:
    v35 = 0;
    if ( v43 < 0 )
      goto LABEL_47;
LABEL_70:
    v35 = v31 < *((_BYTE *)v34 + 64);
    goto LABEL_47;
  }
  v25 = v12 - v21;
  if ( (__int64)(v12 - v21) > 0x7FFFFFFF )
    goto LABEL_36;
LABEL_33:
  if ( v25 < (__int64)0xFFFFFFFF80000000LL || (int)v25 < 0 )
    goto LABEL_41;
  if ( !v23 )
    goto LABEL_37;
LABEL_36:
  LODWORD(v26) = memcmp(v22, v11, v23);
  if ( (_DWORD)v26 )
    goto LABEL_39;
LABEL_37:
  v26 = v21 - v12;
  if ( (__int64)(v21 - v12) > 0x7FFFFFFF )
    goto LABEL_40;
  if ( v26 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_39;
LABEL_49:
  if ( v11 != &v69 )
    j_j___libc_free_0(v11, v69.m128i_i64[0] + 1);
  result = m128i_i64[9];
  if ( !result )
  {
    v46 = a1[26];
    if ( v46 )
      a1[26] = *(_QWORD *)v46;
    else
      v46 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v47 + 1) = a4;
    *(_QWORD *)&v47 = a3;
    v48 = sub_1D274F0(v47, v23, v14, v6, v7);
    s2 = 0;
    *(_QWORD *)(v46 + 40) = v48;
    *(_QWORD *)v46 = 0;
    *(_QWORD *)(v46 + 8) = 0;
    *(_QWORD *)(v46 + 16) = 0;
    *(_WORD *)(v46 + 24) = 39;
    *(_DWORD *)(v46 + 28) = -1;
    *(_QWORD *)(v46 + 32) = 0;
    *(_QWORD *)(v46 + 48) = 0;
    *(_QWORD *)(v46 + 56) = 0x100000000LL;
    *(_DWORD *)(v46 + 64) = 0;
    v49 = (unsigned __int8 *)s2;
    *(_QWORD *)(v46 + 72) = s2;
    if ( v49 )
      sub_1623210((__int64)&s2, v49, v46 + 72);
    *(_WORD *)(v46 + 80) &= 0xF000u;
    *(_WORD *)(v46 + 26) = 0;
    *(_QWORD *)(v46 + 88) = a2;
    *(_BYTE *)(v46 + 96) = a5;
    m128i_i64[9] = v46;
    sub_1D172A0((__int64)a1, v46);
    return m128i_i64[9];
  }
  return result;
}
