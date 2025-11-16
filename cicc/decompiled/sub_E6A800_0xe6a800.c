// Function: sub_E6A800
// Address: 0xe6a800
//
__m128i *__fastcall sub_E6A800(_QWORD *a1, __m128i *a2)
{
  __m128i *v2; // rax
  __m128i *v3; // rdx
  __m128i *v4; // r15
  size_t v5; // r13
  __m128i v6; // xmm0
  __int32 v7; // eax
  _QWORD *v8; // rbx
  size_t v9; // rdx
  signed __int64 v10; // rax
  unsigned int v11; // eax
  _QWORD *v12; // rdx
  _QWORD *v13; // rsi
  size_t v14; // r14
  const void *v15; // r12
  void *v16; // r9
  void *v17; // r8
  const void *v18; // rdi
  const void *v19; // rsi
  size_t v20; // rdx
  int v21; // eax
  size_t v22; // rcx
  _QWORD *v23; // r10
  const void *v24; // r14
  size_t v25; // r12
  bool v26; // cc
  size_t v27; // rdx
  int v28; // eax
  __int64 v29; // r12
  __int64 v30; // rdi
  int v32; // eax
  __int64 v33; // rax
  void *v34; // rcx
  size_t v35; // r12
  const void *v36; // rdi
  const void *v37; // rsi
  size_t v38; // rdx
  unsigned __int8 v39; // al
  __m128i *v40; // [rsp+0h] [rbp-70h]
  _QWORD *v41; // [rsp+8h] [rbp-68h]
  size_t n; // [rsp+10h] [rbp-60h]
  size_t na; // [rsp+10h] [rbp-60h]
  void *s2; // [rsp+18h] [rbp-58h]
  void *v45; // [rsp+20h] [rbp-50h]
  _QWORD *v46; // [rsp+20h] [rbp-50h]
  void *v47; // [rsp+20h] [rbp-50h]
  unsigned __int32 v49; // [rsp+30h] [rbp-40h]
  _QWORD *v50; // [rsp+30h] [rbp-40h]
  _QWORD *v51; // [rsp+30h] [rbp-40h]
  __m128i *s1; // [rsp+38h] [rbp-38h]
  _QWORD *s1a; // [rsp+38h] [rbp-38h]

  v2 = (__m128i *)sub_22077B0(96);
  v3 = (__m128i *)a2->m128i_i64[0];
  v4 = v2;
  v40 = v2 + 3;
  v2[2].m128i_i64[0] = (__int64)v2[3].m128i_i64;
  if ( v3 == &a2[1] )
  {
    v2[3] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v2[2].m128i_i64[0] = (__int64)v3;
    v2[3].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v5 = a2->m128i_u64[1];
  v6 = _mm_loadu_si128(a2 + 2);
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v7 = a2[3].m128i_i32[0];
  a2->m128i_i64[1] = 0;
  a2[1].m128i_i8[0] = 0;
  v49 = v7;
  v4[5].m128i_i32[0] = v7;
  v4[2].m128i_i64[1] = v5;
  v8 = (_QWORD *)a1[2];
  v4[4] = v6;
  v4[5].m128i_i64[1] = 0;
  v41 = a1 + 1;
  if ( !v8 )
  {
    v8 = a1 + 1;
    if ( v41 == (_QWORD *)a1[3] )
    {
      v23 = a1 + 1;
      v30 = 1;
LABEL_38:
      sub_220F040(v30, v4, v23, v41);
      ++a1[5];
      return v4;
    }
    s1 = (__m128i *)v4[2].m128i_i64[0];
LABEL_53:
    v33 = sub_220EF80(v8);
    v23 = v8;
    v25 = *(_QWORD *)(v33 + 40);
    v24 = *(const void **)(v33 + 32);
    v8 = (_QWORD *)v33;
    v26 = v5 <= v25;
    if ( v5 == v25 )
      goto LABEL_54;
LABEL_28:
    v27 = v25;
    if ( v26 )
      v27 = v5;
    if ( v27 && (v50 = v23, v28 = memcmp(v24, s1, v27), v23 = v50, v28) )
    {
LABEL_56:
      if ( v28 >= 0 )
        goto LABEL_47;
    }
    else
    {
      v29 = v25 - v5;
      if ( v29 > 0x7FFFFFFF || v29 >= (__int64)0xFFFFFFFF80000000LL && (int)v29 >= 0 )
        goto LABEL_47;
    }
    goto LABEL_35;
  }
  s1 = (__m128i *)v4[2].m128i_i64[0];
  while ( 1 )
  {
    v14 = v8[5];
    v15 = (const void *)v8[4];
    if ( v5 != v14 )
    {
      v9 = v8[5];
      if ( v5 <= v14 )
        v9 = v5;
      if ( v9 )
      {
        LODWORD(v10) = memcmp(s1, (const void *)v8[4], v9);
        if ( (_DWORD)v10 )
          goto LABEL_11;
      }
      v10 = v5 - v14;
      if ( (__int64)(v5 - v14) < 0x80000000LL )
      {
        if ( v10 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_25;
LABEL_11:
        v11 = (unsigned int)v10 >> 31;
LABEL_12:
        v12 = (_QWORD *)v8[2];
        v13 = (_QWORD *)v8[3];
        if ( (_BYTE)v11 )
          goto LABEL_14;
        goto LABEL_13;
      }
      goto LABEL_43;
    }
    if ( v5 )
    {
      LODWORD(v10) = memcmp(s1, (const void *)v8[4], v5);
      if ( (_DWORD)v10 )
        goto LABEL_11;
    }
    v16 = (void *)v4[4].m128i_i64[1];
    v17 = (void *)v8[9];
    v18 = (const void *)v4[4].m128i_i64[0];
    v19 = (const void *)v8[8];
    if ( v17 == v16 )
      break;
    v20 = v4[4].m128i_u64[1];
    if ( v17 <= v16 )
      v20 = v8[9];
    if ( v20 )
    {
      s2 = (void *)v4[4].m128i_i64[1];
      v45 = (void *)v8[9];
      v21 = memcmp(v18, v19, v20);
      v17 = v45;
      v16 = s2;
      if ( v21 )
        goto LABEL_50;
    }
    if ( v17 > v16 )
      goto LABEL_25;
LABEL_43:
    v13 = (_QWORD *)v8[3];
LABEL_13:
    v12 = v13;
    LOBYTE(v11) = 0;
LABEL_14:
    if ( !v12 )
      goto LABEL_26;
LABEL_15:
    v8 = v12;
  }
  if ( !v17 || (n = v8[9], !memcmp(v18, v19, n)) )
  {
    LOBYTE(v11) = v49 < *((_DWORD *)v8 + 20);
    goto LABEL_12;
  }
  v21 = memcmp(v18, v19, n);
  if ( !v21 )
    goto LABEL_43;
LABEL_50:
  if ( v21 >= 0 )
  {
    v13 = (_QWORD *)v8[3];
    goto LABEL_13;
  }
LABEL_25:
  v12 = (_QWORD *)v8[2];
  LOBYTE(v11) = 1;
  if ( v12 )
    goto LABEL_15;
LABEL_26:
  v22 = v14;
  v23 = v8;
  v24 = v15;
  v25 = v22;
  if ( (_BYTE)v11 )
  {
    if ( v8 == (_QWORD *)a1[3] )
    {
      v23 = v8;
LABEL_36:
      v30 = 1;
      if ( v41 != v23 )
      {
        s1a = v23;
        v39 = sub_E640F0((__int64)v4[2].m128i_i64, (__int64)(v23 + 4));
        v23 = s1a;
        v30 = v39;
      }
      goto LABEL_38;
    }
    goto LABEL_53;
  }
  v26 = v5 <= v22;
  if ( v5 != v22 )
    goto LABEL_28;
LABEL_54:
  if ( v5 )
  {
    v46 = v23;
    v28 = memcmp(v24, s1, v5);
    v23 = v46;
    if ( v28 )
      goto LABEL_56;
  }
  v34 = (void *)v8[9];
  v35 = v4[4].m128i_u64[1];
  v36 = (const void *)v8[8];
  v37 = (const void *)v4[4].m128i_i64[0];
  if ( (void *)v35 == v34 )
  {
    if ( v35 )
    {
      na = (size_t)v23;
      v32 = memcmp(v36, v37, v4[4].m128i_u64[1]);
      v23 = (_QWORD *)na;
      if ( v32 )
      {
        v28 = memcmp(v36, v37, v35);
        v23 = (_QWORD *)na;
        if ( v28 )
          goto LABEL_56;
        goto LABEL_47;
      }
    }
    if ( v49 <= *((_DWORD *)v8 + 20) )
      goto LABEL_47;
    goto LABEL_35;
  }
  v38 = v8[9];
  if ( v35 <= (unsigned __int64)v34 )
    v38 = v4[4].m128i_u64[1];
  if ( v38 )
  {
    v47 = (void *)v8[9];
    v51 = v23;
    v28 = memcmp(v36, v37, v38);
    v23 = v51;
    v34 = v47;
    if ( v28 )
      goto LABEL_56;
  }
  if ( v35 > (unsigned __int64)v34 )
  {
LABEL_35:
    if ( !v23 )
    {
      v8 = 0;
      goto LABEL_47;
    }
    goto LABEL_36;
  }
LABEL_47:
  if ( v40 != s1 )
    j_j___libc_free_0(s1, v4[3].m128i_i64[0] + 1);
  j_j___libc_free_0(v4, 96);
  return (__m128i *)v8;
}
