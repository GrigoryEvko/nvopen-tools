// Function: sub_123EA90
// Address: 0x123ea90
//
__int64 __fastcall sub_123EA90(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4)
{
  __int64 v5; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r14
  __int64 v12; // rax
  size_t v13; // rdx
  const void *v14; // rsi
  unsigned __int64 v15; // rax
  char *v16; // r8
  __int64 v17; // rax
  char *v18; // rcx
  int v19; // eax
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdx
  char v28; // r14
  __int64 v29; // rax
  void *v30; // r8
  __int64 v31; // rax
  unsigned int v32; // ecx
  __int64 v33; // rax
  _DWORD *v34; // rdx
  bool v35; // al
  __m128i *v36; // rsi
  signed __int64 v37; // rax
  int v38; // eax
  int v39; // eax
  void *v40; // rdi
  char *v41; // [rsp+0h] [rbp-B0h]
  char *v42; // [rsp+0h] [rbp-B0h]
  char *v43; // [rsp+0h] [rbp-B0h]
  size_t n; // [rsp+8h] [rbp-A8h]
  size_t na; // [rsp+8h] [rbp-A8h]
  size_t nb; // [rsp+8h] [rbp-A8h]
  size_t nc; // [rsp+8h] [rbp-A8h]
  void *s2a; // [rsp+10h] [rbp-A0h]
  _DWORD *s2; // [rsp+10h] [rbp-A0h]
  char *s2b; // [rsp+10h] [rbp-A0h]
  char *s2c; // [rsp+10h] [rbp-A0h]
  size_t v52; // [rsp+18h] [rbp-98h]
  size_t v53; // [rsp+18h] [rbp-98h]
  size_t v54; // [rsp+18h] [rbp-98h]
  size_t v55; // [rsp+18h] [rbp-98h]
  size_t v56; // [rsp+18h] [rbp-98h]
  size_t v57; // [rsp+18h] [rbp-98h]
  char *v58; // [rsp+20h] [rbp-90h]
  unsigned int v59; // [rsp+20h] [rbp-90h]
  int v60; // [rsp+28h] [rbp-88h]
  char *v61; // [rsp+28h] [rbp-88h]
  unsigned __int8 v62; // [rsp+28h] [rbp-88h]
  void *v63; // [rsp+28h] [rbp-88h]
  char *v64; // [rsp+28h] [rbp-88h]
  void *v65; // [rsp+28h] [rbp-88h]
  int v66; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v67; // [rsp+34h] [rbp-7Ch] BYREF
  unsigned __int64 v68; // [rsp+38h] [rbp-78h] BYREF
  __int64 v69; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v70; // [rsp+48h] [rbp-68h] BYREF
  __m128i v71; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v72[10]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a1 + 176;
  v68 = *(_QWORD *)(v5 + 56);
  *(_DWORD *)(a1 + 240) = sub_1205200(v5);
  LOWORD(v66) = 0;
  v71 = 0u;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_1212200(a1, &v71) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here") )
    return 1;
  if ( (unsigned __int8)sub_1211B70(a1, &v66) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 450, "expected 'aliasee' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    return 1;
  v69 = 0;
  if ( (unsigned __int8)sub_12122D0(a1, &v69, &v67) || (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
    return 1;
  v60 = v66;
  v10 = sub_22077B0(72);
  v11 = (_QWORD *)v10;
  if ( v10 )
  {
    *(_DWORD *)(v10 + 8) = 0;
    v12 = v10 + 56;
    *(_QWORD *)(v12 - 40) = 0;
    *(_DWORD *)(v12 - 44) = v60;
    v11[5] = v12;
    v11[6] = 0;
    v11[7] = 0;
    *v11 = &unk_49D9790;
    v11[8] = 0;
  }
  v13 = v71.m128i_u64[1];
  v14 = (const void *)v71.m128i_i64[0];
  v15 = v69 & 0xFFFFFFFFFFFFFFF8LL;
  v11[3] = v71.m128i_i64[0];
  v11[4] = v13;
  if ( v15 == -8 )
  {
    v29 = *(_QWORD *)(a1 + 1592);
    if ( v29 )
    {
      v30 = (void *)(a1 + 1584);
      do
      {
        if ( *(_DWORD *)(v29 + 32) < v67 )
        {
          v29 = *(_QWORD *)(v29 + 24);
        }
        else
        {
          v30 = (void *)v29;
          v29 = *(_QWORD *)(v29 + 16);
        }
      }
      while ( v29 );
      if ( (void *)(a1 + 1584) != v30 && v67 >= *((_DWORD *)v30 + 8) )
      {
LABEL_47:
        v72[0] = v11;
        v36 = (__m128i *)*((_QWORD *)v30 + 6);
        if ( v36 == *((__m128i **)v30 + 7) )
        {
          sub_1213900((const __m128i **)v30 + 5, v36, v72, &v68);
        }
        else
        {
          if ( v36 )
          {
            v36->m128i_i64[0] = (__int64)v11;
            v36->m128i_i64[1] = v68;
          }
          *((_QWORD *)v30 + 6) += 16LL;
        }
        goto LABEL_34;
      }
    }
    else
    {
      v30 = (void *)(a1 + 1584);
    }
    v55 = (size_t)v30;
    s2 = (_DWORD *)(a1 + 1584);
    v31 = sub_22077B0(64);
    v32 = v67;
    *(_QWORD *)(v31 + 40) = 0;
    *(_DWORD *)(v31 + 32) = v32;
    *(_QWORD *)(v31 + 48) = 0;
    *(_QWORD *)(v31 + 56) = 0;
    v59 = v32;
    v63 = (void *)v31;
    v33 = sub_123CAC0((_QWORD *)(a1 + 1576), v55, (unsigned int *)(v31 + 32));
    if ( v34 )
    {
      v35 = s2 == v34 || v33 != 0;
      if ( !v35 )
        v35 = v59 < v34[8];
      sub_220F040(v35, v63, v34, s2);
      ++*(_QWORD *)(a1 + 1616);
      v30 = v63;
    }
    else
    {
      v40 = v63;
      v65 = (void *)v33;
      j_j___libc_free_0(v40, 64);
      v30 = v65;
    }
    goto LABEL_47;
  }
  v16 = *(char **)(v15 + 24);
  v58 = *(char **)(v15 + 32);
  v17 = (v58 - v16) >> 5;
  if ( v17 > 0 )
  {
    v18 = v16;
    v61 = &v16[32 * v17];
    do
    {
      if ( v13 == *(_QWORD *)(*(_QWORD *)v18 + 32LL) )
      {
        if ( !v13
          || (n = (size_t)v18,
              s2a = (void *)v13,
              v19 = memcmp(*(const void **)(*(_QWORD *)v18 + 24LL), v14, v13),
              v13 = (size_t)s2a,
              v18 = (char *)n,
              !v19) )
        {
          v16 = v18;
          goto LABEL_31;
        }
      }
      v20 = *((_QWORD *)v18 + 1);
      v16 = v18 + 8;
      if ( v13 == *(_QWORD *)(v20 + 32) )
      {
        v52 = (size_t)v18;
        if ( !v13 )
          goto LABEL_31;
        v41 = v18 + 8;
        na = v13;
        v21 = memcmp(*(const void **)(v20 + 24), v14, v13);
        v13 = na;
        v16 = v41;
        v18 = (char *)v52;
        if ( !v21 )
          goto LABEL_31;
      }
      v22 = *((_QWORD *)v18 + 2);
      v16 = v18 + 16;
      if ( v13 == *(_QWORD *)(v22 + 32) )
      {
        v53 = (size_t)v18;
        if ( !v13 )
          goto LABEL_31;
        v42 = v18 + 16;
        nb = v13;
        v23 = memcmp(*(const void **)(v22 + 24), v14, v13);
        v13 = nb;
        v16 = v42;
        v18 = (char *)v53;
        if ( !v23 )
          goto LABEL_31;
      }
      v24 = *((_QWORD *)v18 + 3);
      v16 = v18 + 24;
      if ( v13 == *(_QWORD *)(v24 + 32) )
      {
        v54 = (size_t)v18;
        if ( !v13 )
          goto LABEL_31;
        v43 = v18 + 24;
        nc = v13;
        v25 = memcmp(*(const void **)(v24 + 24), v14, v13);
        v13 = nc;
        v16 = v43;
        v18 = (char *)v54;
        if ( !v25 )
          goto LABEL_31;
      }
      v18 += 32;
    }
    while ( v61 != v18 );
    v16 = v61;
  }
  v37 = v58 - v16;
  if ( v58 - v16 == 16 )
  {
LABEL_60:
    if ( v13 == *(_QWORD *)(*(_QWORD *)v16 + 32LL) )
    {
      if ( !v13 )
        goto LABEL_31;
      s2c = v16;
      v57 = v13;
      v39 = memcmp(*(const void **)(*(_QWORD *)v16 + 24LL), v14, v13);
      v13 = v57;
      v16 = s2c;
      if ( !v39 )
        goto LABEL_31;
    }
    v16 += 8;
    goto LABEL_64;
  }
  if ( v37 == 24 )
  {
    if ( v13 == *(_QWORD *)(*(_QWORD *)v16 + 32LL) )
    {
      if ( !v13 )
        goto LABEL_31;
      s2b = v16;
      v56 = v13;
      v38 = memcmp(*(const void **)(*(_QWORD *)v16 + 24LL), v14, v13);
      v13 = v56;
      v16 = s2b;
      if ( !v38 )
        goto LABEL_31;
    }
    v16 += 8;
    goto LABEL_60;
  }
  if ( v37 != 8 )
    goto LABEL_55;
LABEL_64:
  if ( v13 != *(_QWORD *)(*(_QWORD *)v16 + 32LL) )
    goto LABEL_55;
  if ( v13 )
  {
    v64 = v16;
    if ( memcmp(*(const void **)(*(_QWORD *)v16 + 24LL), v14, v13) )
      goto LABEL_55;
    v16 = v64;
  }
LABEL_31:
  if ( v58 == v16 )
  {
LABEL_55:
    v26 = 0;
    goto LABEL_33;
  }
  v26 = *(_QWORD *)v16;
LABEL_33:
  v27 = v69;
  v11[8] = v26;
  v11[7] = v27;
LABEL_34:
  v70 = v11;
  v28 = v66;
  sub_2241BD0(v72, a2);
  v62 = sub_123DE00(a1, v72, a3, v28 & 0xF, a4, (__int64 *)&v70, v68);
  sub_2240A30(v72);
  sub_9C9050((__int64 *)&v70);
  return v62;
}
