// Function: sub_1078750
// Address: 0x1078750
//
__int64 __fastcall sub_1078750(_QWORD *a1, const __m128i *a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  const void *v4; // r14
  size_t v5; // r15
  size_t v6; // r13
  size_t v7; // rdx
  int v8; // eax
  size_t v9; // rbx
  size_t v10; // rdx
  int v11; // eax
  __int64 v12; // rbx
  __m128i v13; // xmm0
  size_t v14; // r14
  void *v15; // r8
  void *v16; // r9
  const void *v17; // r10
  size_t v18; // rdx
  int v19; // eax
  int v20; // eax
  __int64 v22; // rax
  size_t v23; // r10
  __int64 v24; // r8
  size_t v25; // rdx
  int v26; // eax
  bool v27; // al
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  _BOOL8 v31; // rdi
  __int64 v32; // rax
  size_t v33; // r10
  _QWORD *v34; // r8
  size_t v35; // rdx
  int v36; // eax
  __int64 v37; // r8
  __m128i v38; // xmm1
  size_t v39; // r14
  __int64 v40; // rcx
  size_t v41; // rbx
  size_t v42; // rdx
  int v43; // eax
  void *v44; // rcx
  size_t v45; // rbx
  size_t v46; // rdx
  unsigned int v47; // eax
  void *v48; // [rsp+0h] [rbp-60h]
  size_t n; // [rsp+8h] [rbp-58h]
  size_t na; // [rsp+8h] [rbp-58h]
  size_t nb; // [rsp+8h] [rbp-58h]
  void *s1; // [rsp+10h] [rbp-50h]
  void *s1a; // [rsp+10h] [rbp-50h]
  void *s1b; // [rsp+10h] [rbp-50h]
  _QWORD *s1c; // [rsp+10h] [rbp-50h]
  void *s2; // [rsp+18h] [rbp-48h]
  void *s2c; // [rsp+18h] [rbp-48h]
  void *s2a; // [rsp+18h] [rbp-48h]
  void *s2b; // [rsp+18h] [rbp-48h]
  void *s2d; // [rsp+18h] [rbp-48h]
  void *s2e; // [rsp+18h] [rbp-48h]
  _QWORD *v63; // [rsp+28h] [rbp-38h]

  v2 = (__int64)(a1 + 1);
  v3 = a1[2];
  v63 = a1 + 1;
  if ( !v3 )
  {
LABEL_67:
    v2 = sub_22077B0(72);
    *(_QWORD *)(v2 + 48) = 0;
    v37 = v2 + 32;
    *(_QWORD *)(v2 + 56) = 0;
    v38 = _mm_loadu_si128(a2);
    *(_QWORD *)(v2 + 64) = 0;
    *(__m128i *)(v2 + 32) = v38;
    if ( a1[5] )
    {
      v39 = *(_QWORD *)(v2 + 40);
      v40 = a1[4];
      v41 = *(_QWORD *)(v40 + 40);
      v42 = v41;
      if ( v39 <= v41 )
        v42 = *(_QWORD *)(v2 + 40);
      if ( v42
        && (s2d = (void *)a1[4],
            v43 = memcmp(*(const void **)(v40 + 32), *(const void **)(v2 + 32), v42),
            v40 = (__int64)s2d,
            v37 = v2 + 32,
            v43) )
      {
        if ( v43 < 0 )
        {
LABEL_74:
          v12 = v2;
          v27 = 0;
          v2 = v40;
          goto LABEL_66;
        }
      }
      else if ( v39 != v41 && v39 > v41 )
      {
        goto LABEL_74;
      }
    }
    v28 = sub_1077BB0((__int64)a1, v37);
    v30 = v29;
    goto LABEL_50;
  }
  v4 = (const void *)a2->m128i_i64[0];
  v5 = a2->m128i_u64[1];
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + 40);
      v7 = v6;
      if ( v5 <= v6 )
        v7 = v5;
      if ( v7 )
      {
        v8 = memcmp(*(const void **)(v3 + 32), v4, v7);
        if ( v8 )
          break;
      }
      if ( v5 != v6 && v5 > v6 )
      {
        v3 = *(_QWORD *)(v3 + 24);
        goto LABEL_11;
      }
LABEL_3:
      v2 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      if ( !v3 )
        goto LABEL_12;
    }
    if ( v8 >= 0 )
      goto LABEL_3;
    v3 = *(_QWORD *)(v3 + 24);
LABEL_11:
    ;
  }
  while ( v3 );
LABEL_12:
  if ( v63 == (_QWORD *)v2 )
    goto LABEL_67;
  v9 = *(_QWORD *)(v2 + 40);
  v10 = v5;
  if ( v9 <= v5 )
    v10 = *(_QWORD *)(v2 + 40);
  if ( v10 )
  {
    v11 = memcmp(v4, *(const void **)(v2 + 32), v10);
    if ( v11 )
    {
      if ( v11 < 0 )
        goto LABEL_19;
      return v2 + 48;
    }
  }
  if ( v9 == v5 || v9 <= v5 )
    return v2 + 48;
LABEL_19:
  v12 = sub_22077B0(72);
  *(_QWORD *)(v12 + 48) = 0;
  v13 = _mm_loadu_si128(a2);
  *(_QWORD *)(v12 + 56) = 0;
  *(_QWORD *)(v12 + 64) = 0;
  *(__m128i *)(v12 + 32) = v13;
  v14 = *(_QWORD *)(v12 + 40);
  v15 = *(void **)(v2 + 40);
  v16 = *(void **)(v12 + 32);
  v17 = *(const void **)(v2 + 32);
  v18 = v14;
  if ( (unsigned __int64)v15 <= v14 )
    v18 = *(_QWORD *)(v2 + 40);
  if ( !v18 )
  {
    if ( v15 == (void *)v14 )
      goto LABEL_28;
LABEL_24:
    if ( (unsigned __int64)v15 <= v14 )
    {
      if ( v18 )
      {
        s1a = v15;
        s2c = v16;
        v20 = memcmp(v17, v16, v18);
        v16 = s2c;
        v15 = s1a;
        if ( v20 )
          goto LABEL_27;
      }
LABEL_37:
      if ( (unsigned __int64)v15 >= v14 )
        goto LABEL_28;
      goto LABEL_38;
    }
LABEL_56:
    s2b = v16;
    if ( a1[3] != v2 )
    {
      v32 = sub_220EF80(v2);
      v33 = *(_QWORD *)(v32 + 40);
      v34 = (_QWORD *)v32;
      v35 = v33;
      if ( v14 <= v33 )
        v35 = v14;
      if ( v35
        && (nb = *(_QWORD *)(v32 + 40),
            s1c = (_QWORD *)v32,
            v36 = memcmp(*(const void **)(v32 + 32), s2b, v35),
            v34 = s1c,
            v33 = nb,
            v36) )
      {
        if ( v36 >= 0 )
          goto LABEL_49;
      }
      else if ( v14 == v33 || v14 <= v33 )
      {
        goto LABEL_49;
      }
      if ( !v34[3] )
        v2 = (__int64)v34;
      v27 = v34[3] != 0;
      goto LABEL_66;
    }
    goto LABEL_46;
  }
  v48 = *(void **)(v2 + 40);
  n = v18;
  s1 = *(void **)(v2 + 32);
  s2 = *(void **)(v12 + 32);
  v19 = memcmp(s2, s1, v18);
  v16 = s2;
  v17 = s1;
  v18 = n;
  v15 = v48;
  if ( v19 )
  {
    if ( v19 >= 0 )
    {
      v20 = memcmp(s1, s2, n);
      v16 = s2;
      v15 = v48;
      if ( v20 )
        goto LABEL_27;
      if ( v48 == (void *)v14 )
        goto LABEL_28;
      goto LABEL_37;
    }
    goto LABEL_56;
  }
  if ( v48 != (void *)v14 )
    goto LABEL_24;
  v20 = memcmp(s1, s2, n);
  v16 = s2;
  if ( !v20 )
  {
LABEL_28:
    j_j___libc_free_0(v12, 72);
    return v2 + 48;
  }
LABEL_27:
  if ( v20 >= 0 )
    goto LABEL_28;
LABEL_38:
  s2a = v16;
  if ( a1[4] == v2 )
    goto LABEL_88;
  v22 = sub_220EEE0(v2);
  v23 = *(_QWORD *)(v22 + 40);
  v24 = v22;
  v25 = v23;
  if ( v14 <= v23 )
    v25 = v14;
  if ( !v25
    || (na = *(_QWORD *)(v22 + 40),
        s1b = (void *)v22,
        v26 = memcmp(s2a, *(const void **)(v22 + 32), v25),
        v24 = (__int64)s1b,
        v23 = na,
        !v26) )
  {
    if ( v14 == v23 || v14 >= v23 )
      goto LABEL_49;
LABEL_44:
    if ( *(_QWORD *)(v2 + 24) )
    {
      v2 = v24;
LABEL_46:
      v27 = 1;
LABEL_66:
      v30 = v2;
      v2 = v12;
      goto LABEL_52;
    }
LABEL_88:
    v27 = 0;
    goto LABEL_66;
  }
  if ( v26 < 0 )
    goto LABEL_44;
LABEL_49:
  v2 = v12;
  v28 = sub_1077BB0((__int64)a1, v12 + 32);
  v30 = v29;
LABEL_50:
  if ( !v29 )
  {
    v12 = v2;
    v2 = v28;
    goto LABEL_28;
  }
  v27 = v28 != 0;
LABEL_52:
  if ( v63 == (_QWORD *)v30 || v27 )
  {
    v31 = 1;
  }
  else
  {
    v44 = *(void **)(v2 + 40);
    v45 = *(_QWORD *)(v30 + 40);
    v46 = v45;
    if ( (unsigned __int64)v44 <= v45 )
      v46 = *(_QWORD *)(v2 + 40);
    if ( v46
      && (s2e = *(void **)(v2 + 40),
          v47 = memcmp(*(const void **)(v2 + 32), *(const void **)(v30 + 32), v46),
          v44 = s2e,
          v47) )
    {
      v31 = v47 >> 31;
    }
    else
    {
      v31 = (unsigned __int64)v44 < v45;
      if ( v44 == (void *)v45 )
        v31 = 0;
    }
  }
  sub_220F040(v31, v2, v30, v63);
  ++a1[5];
  return v2 + 48;
}
