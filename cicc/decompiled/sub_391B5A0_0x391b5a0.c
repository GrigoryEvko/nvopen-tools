// Function: sub_391B5A0
// Address: 0x391b5a0
//
unsigned __int64 __fastcall sub_391B5A0(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v6; // rax
  _QWORD *v7; // rcx
  unsigned __int64 v8; // r12
  __m128i v9; // xmm0
  size_t v10; // r14
  size_t v11; // r9
  _QWORD *v12; // r8
  void *v13; // r10
  void *v14; // r11
  int v15; // eax
  __int64 v16; // rax
  void *v17; // r8
  const void *v18; // rdi
  _QWORD *v19; // r15
  const void *v20; // rsi
  int v21; // eax
  _QWORD *v22; // rax
  size_t v23; // rdx
  __int64 v24; // rax
  _QWORD *v25; // rdx
  bool v26; // al
  unsigned int v27; // edi
  int v29; // eax
  unsigned __int64 v30; // r14
  const void *v31; // rsi
  unsigned __int64 v32; // rbx
  const void *v33; // rdi
  int v34; // eax
  const void *v35; // rsi
  __int64 v36; // rax
  size_t v37; // r9
  void *v38; // r8
  const void *v39; // rsi
  const void *v40; // r11
  void *v41; // r10
  const void *v42; // rdi
  int v43; // eax
  int v44; // eax
  unsigned int v45; // edi
  int v46; // eax
  const void *v47; // rdi
  _QWORD *v48; // [rsp+0h] [rbp-60h]
  _QWORD *v49; // [rsp+0h] [rbp-60h]
  void *v50; // [rsp+0h] [rbp-60h]
  const void *v51; // [rsp+0h] [rbp-60h]
  _QWORD *v52; // [rsp+8h] [rbp-58h]
  _QWORD *v53; // [rsp+8h] [rbp-58h]
  void *v54; // [rsp+8h] [rbp-58h]
  void *v55; // [rsp+8h] [rbp-58h]
  _QWORD *v56; // [rsp+8h] [rbp-58h]
  void *v57; // [rsp+10h] [rbp-50h]
  void *v58; // [rsp+10h] [rbp-50h]
  void *v59; // [rsp+10h] [rbp-50h]
  void *v60; // [rsp+10h] [rbp-50h]
  _QWORD *v61; // [rsp+10h] [rbp-50h]
  void *v62; // [rsp+10h] [rbp-50h]
  void *s1b; // [rsp+18h] [rbp-48h]
  _QWORD *s1; // [rsp+18h] [rbp-48h]
  void *s1c; // [rsp+18h] [rbp-48h]
  void *s1d; // [rsp+18h] [rbp-48h]
  void *s1e; // [rsp+18h] [rbp-48h]
  void *s1f; // [rsp+18h] [rbp-48h]
  _QWORD *s1a; // [rsp+18h] [rbp-48h]
  void *s1g; // [rsp+18h] [rbp-48h]
  void *s1h; // [rsp+18h] [rbp-48h]
  void *s2b; // [rsp+20h] [rbp-40h]
  void *s2; // [rsp+20h] [rbp-40h]
  void *s2c; // [rsp+20h] [rbp-40h]
  void *s2d; // [rsp+20h] [rbp-40h]
  _QWORD *s2e; // [rsp+20h] [rbp-40h]
  void *s2f; // [rsp+20h] [rbp-40h]
  _QWORD *s2g; // [rsp+20h] [rbp-40h]
  void *s2h; // [rsp+20h] [rbp-40h]
  void *s2a; // [rsp+20h] [rbp-40h]
  void *s2i; // [rsp+20h] [rbp-40h]
  _QWORD *s2j; // [rsp+20h] [rbp-40h]
  _QWORD *s2k; // [rsp+20h] [rbp-40h]
  void *s2l; // [rsp+20h] [rbp-40h]
  __int64 v85; // [rsp+28h] [rbp-38h]
  _QWORD *v86; // [rsp+28h] [rbp-38h]
  size_t v87; // [rsp+28h] [rbp-38h]
  size_t v88; // [rsp+28h] [rbp-38h]

  v6 = sub_22077B0(0x48u);
  v7 = a1 + 1;
  v8 = v6;
  v9 = _mm_loadu_si128(*a3);
  *(_QWORD *)(v6 + 48) = 0;
  *(_QWORD *)(v6 + 56) = 0;
  *(_QWORD *)(v6 + 64) = 0;
  v85 = v6 + 32;
  *(__m128i *)(v6 + 32) = v9;
  if ( a1 + 1 == a2 )
  {
    if ( !a1[5] )
      goto LABEL_21;
    v19 = (_QWORD *)a1[4];
    v30 = *(_QWORD *)(v6 + 40);
    v31 = *(const void **)(v6 + 32);
    v32 = v19[5];
    v33 = (const void *)v19[4];
    if ( v32 > v30 )
    {
      if ( !v30 )
        goto LABEL_21;
      v34 = memcmp(v33, v31, *(_QWORD *)(v6 + 40));
      v7 = a1 + 1;
      if ( !v34 )
      {
LABEL_35:
        if ( v32 < v30 )
        {
LABEL_36:
          v26 = 0;
          goto LABEL_23;
        }
        goto LABEL_21;
      }
    }
    else if ( !v32 || (v34 = memcmp(v33, v31, v19[5]), v7 = a1 + 1, !v34) )
    {
      if ( v32 == v30 )
        goto LABEL_21;
      goto LABEL_35;
    }
    if ( v34 < 0 )
    {
      v26 = 0;
LABEL_23:
      if ( v7 == v19 || v26 )
        goto LABEL_25;
      v10 = *(_QWORD *)(v8 + 40);
      v37 = v19[5];
      v40 = (const void *)v19[4];
      v41 = *(void **)(v8 + 32);
      goto LABEL_58;
    }
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(v6 + 40);
  v11 = a2[5];
  v12 = a2;
  v13 = *(void **)(v6 + 32);
  v14 = (void *)a2[4];
  if ( v11 >= v10 )
  {
    if ( !v10 )
      goto LABEL_5;
    v57 = (void *)a2[5];
    s1b = (void *)a2[4];
    s2b = *(void **)(v6 + 32);
    v15 = memcmp(s2b, s1b, *(_QWORD *)(v6 + 40));
    v13 = s2b;
    v14 = s1b;
    v11 = (size_t)v57;
    v7 = a1 + 1;
    v12 = a2;
    if ( !v15 )
    {
LABEL_5:
      if ( v11 == v10 )
      {
        if ( !v11 )
          goto LABEL_30;
LABEL_28:
        v49 = v12;
        v53 = v7;
        v59 = (void *)v11;
        s1d = v13;
        s2f = v14;
        v29 = memcmp(v14, v13, v11);
        v14 = s2f;
        v13 = s1d;
        v11 = (size_t)v59;
        v7 = v53;
        v12 = v49;
        if ( !v29 )
          goto LABEL_29;
LABEL_55:
        if ( v29 >= 0 )
          goto LABEL_30;
        goto LABEL_47;
      }
      if ( v11 > v10 )
        goto LABEL_7;
LABEL_68:
      if ( !v11 )
      {
LABEL_29:
        if ( v11 == v10 )
          goto LABEL_30;
        goto LABEL_46;
      }
      goto LABEL_28;
    }
    goto LABEL_41;
  }
  if ( !v11 )
    goto LABEL_46;
  v52 = a1 + 1;
  v58 = (void *)a2[5];
  s1c = (void *)a2[4];
  s2d = *(void **)(v6 + 32);
  v48 = a2;
  v15 = memcmp(s2d, s1c, (size_t)v58);
  v13 = s2d;
  v14 = s1c;
  v11 = (size_t)v58;
  v7 = a1 + 1;
  v12 = a2;
  if ( v15 )
  {
LABEL_41:
    if ( v15 >= 0 )
    {
      if ( v11 <= v10 )
        goto LABEL_68;
      if ( !v10 )
        goto LABEL_30;
      v48 = v12;
      v23 = v10;
      v52 = v7;
      v58 = (void *)v11;
      goto LABEL_45;
    }
    goto LABEL_7;
  }
  if ( (unsigned __int64)v58 <= v10 )
  {
    v23 = (size_t)v58;
LABEL_45:
    s1f = v13;
    s2h = v14;
    v29 = memcmp(v14, v13, v23);
    v14 = s2h;
    v13 = s1f;
    v11 = (size_t)v58;
    v7 = v52;
    v12 = v48;
    if ( v29 )
      goto LABEL_55;
LABEL_46:
    if ( v11 >= v10 )
      goto LABEL_30;
LABEL_47:
    if ( (_QWORD *)a1[4] == a2 )
    {
      v22 = 0;
      goto LABEL_15;
    }
    v54 = v13;
    v60 = v14;
    s1a = v7;
    s2a = (void *)v11;
    v36 = sub_220EEE0((__int64)a2);
    v37 = (size_t)s2a;
    v7 = s1a;
    v38 = *(void **)(v36 + 40);
    v39 = *(const void **)(v36 + 32);
    v19 = (_QWORD *)v36;
    v40 = v60;
    v41 = v54;
    if ( v10 > (unsigned __int64)v38 )
    {
      if ( !v38 )
        goto LABEL_21;
      v47 = v54;
      v51 = v60;
      v56 = s1a;
      v62 = s2a;
      s1h = *(void **)(v36 + 40);
      s2l = v41;
      v43 = memcmp(v47, v39, (size_t)s1h);
      v41 = s2l;
      v38 = s1h;
      v37 = (size_t)v62;
      v7 = v56;
      v40 = v51;
      if ( !v43 )
        goto LABEL_52;
    }
    else
    {
      if ( !v10 )
        goto LABEL_51;
      v42 = v54;
      v50 = *(void **)(v36 + 40);
      v55 = v60;
      v61 = s1a;
      s1g = s2a;
      s2i = v41;
      v43 = memcmp(v42, v39, v10);
      v41 = s2i;
      v37 = (size_t)s1g;
      v7 = v61;
      v40 = v55;
      v38 = v50;
      if ( !v43 )
      {
LABEL_51:
        if ( (void *)v10 == v38 )
          goto LABEL_21;
LABEL_52:
        if ( v10 >= (unsigned __int64)v38 )
          goto LABEL_21;
        goto LABEL_53;
      }
    }
    if ( v43 >= 0 )
      goto LABEL_21;
LABEL_53:
    if ( a2[3] )
    {
LABEL_25:
      LOBYTE(v27) = 1;
LABEL_26:
      sub_220F040(v27, v8, v19, v7);
      ++a1[5];
      return v8;
    }
    v19 = a2;
LABEL_58:
    if ( v10 > v37 )
    {
      LOBYTE(v27) = 0;
      if ( !v37 )
        goto LABEL_26;
      s2k = v7;
      v88 = v37;
      v46 = memcmp(v41, v40, v37);
      v37 = v88;
      v7 = s2k;
      v45 = v46;
      if ( !v46 )
        goto LABEL_62;
    }
    else if ( !v10 || (s2j = v7, v87 = v37, v44 = memcmp(v41, v40, v10), v37 = v87, v7 = s2j, (v45 = v44) == 0) )
    {
      LOBYTE(v27) = 0;
      if ( v10 == v37 )
        goto LABEL_26;
LABEL_62:
      LOBYTE(v27) = v10 < v37;
      goto LABEL_26;
    }
    v27 = v45 >> 31;
    goto LABEL_26;
  }
LABEL_7:
  s2 = v13;
  if ( (_QWORD *)a1[3] == a2 )
  {
LABEL_14:
    v22 = a2;
LABEL_15:
    v19 = a2;
    v12 = v22;
LABEL_22:
    v26 = v12 != 0;
    goto LABEL_23;
  }
  s1 = v7;
  v16 = sub_220EF80((__int64)a2);
  v7 = s1;
  v17 = *(void **)(v16 + 40);
  v18 = *(const void **)(v16 + 32);
  v19 = (_QWORD *)v16;
  if ( v10 < (unsigned __int64)v17 )
  {
    if ( !v10 )
      goto LABEL_21;
    v35 = s2;
    s1e = *(void **)(v16 + 40);
    s2g = v7;
    v21 = memcmp(v18, v35, v10);
    v7 = s2g;
    v17 = s1e;
    if ( !v21 )
    {
LABEL_12:
      if ( v10 > (unsigned __int64)v17 )
        goto LABEL_13;
      goto LABEL_21;
    }
LABEL_39:
    if ( v21 < 0 )
    {
LABEL_13:
      if ( v19[3] )
        goto LABEL_14;
      goto LABEL_36;
    }
    goto LABEL_21;
  }
  if ( v17 )
  {
    v20 = s2;
    s2c = *(void **)(v16 + 40);
    v21 = memcmp(v18, v20, (size_t)s2c);
    v17 = s2c;
    v7 = s1;
    if ( v21 )
      goto LABEL_39;
  }
  if ( (void *)v10 != v17 )
    goto LABEL_12;
LABEL_21:
  s2e = v7;
  v24 = sub_391AB80((__int64)a1, v85);
  v7 = s2e;
  v12 = (_QWORD *)v24;
  v19 = v25;
  if ( v25 )
    goto LABEL_22;
LABEL_30:
  v86 = v12;
  j_j___libc_free_0(v8);
  return (unsigned __int64)v86;
}
