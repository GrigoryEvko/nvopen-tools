// Function: sub_10648E0
// Address: 0x10648e0
//
_QWORD *__fastcall sub_10648E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // r10
  int v11; // r12d
  __int64 *v12; // r11
  __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rbx
  _QWORD *v16; // r12
  _QWORD *v17; // r14
  int v19; // eax
  __int64 v20; // r8
  __int64 *v21; // rax
  unsigned int i; // edi
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // r12
  __int64 *v25; // rax
  __int64 *j; // rdx
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // r13
  int v30; // r12d
  char v31; // r15
  __int64 v32; // rax
  bool v33; // dl
  char v34; // r14
  char v35; // cl
  unsigned int v36; // r9d
  __int64 v37; // r10
  int v38; // r8d
  _QWORD *v39; // rdx
  unsigned int v40; // r13d
  _QWORD *v41; // rax
  __int64 v42; // rdi
  _QWORD *v43; // r13
  unsigned __int8 v44; // al
  unsigned __int64 v45; // rax
  unsigned int v46; // edx
  unsigned int v47; // eax
  unsigned int v48; // edx
  char v49; // al
  __int64 v50; // rdi
  int v51; // edx
  __int64 v52; // rax
  char v53; // dl
  char v54; // r8
  _QWORD *v55; // rax
  int v56; // eax
  int v57; // eax
  int v58; // edi
  __int64 v59; // rsi
  unsigned int v60; // eax
  int v61; // ebx
  int v62; // eax
  int v63; // edi
  __int64 v64; // rsi
  int v65; // ebx
  unsigned int v66; // eax
  int v67; // esi
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 **v70; // rdi
  __int64 **v71; // r14
  __int64 v72; // rax
  size_t v73; // rdx
  __int64 v74; // r9
  const void *v75; // r8
  const void *v76; // rsi
  char *v77; // rdi
  char v78; // [rsp+Fh] [rbp-B1h]
  void *src; // [rsp+18h] [rbp-A8h]
  void *srca; // [rsp+18h] [rbp-A8h]
  int n; // [rsp+20h] [rbp-A0h]
  int nb; // [rsp+20h] [rbp-A0h]
  char na; // [rsp+20h] [rbp-A0h]
  size_t nc; // [rsp+20h] [rbp-A0h]
  size_t nd; // [rsp+20h] [rbp-A0h]
  _QWORD **v86; // [rsp+28h] [rbp-98h] BYREF
  char *v87; // [rsp+30h] [rbp-90h] BYREF
  size_t v88; // [rsp+38h] [rbp-88h]
  __int64 v89; // [rsp+40h] [rbp-80h]
  _BYTE v90[24]; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v91; // [rsp+60h] [rbp-60h] BYREF
  __int64 v92; // [rsp+68h] [rbp-58h]
  _BYTE v93[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = a1;
  v8 = *(unsigned int *)(a1 + 32);
  v86 = (_QWORD **)a2;
  src = (void *)(a1 + 8);
  if ( !(_DWORD)v8 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_75;
  }
  v9 = *(_QWORD *)(a1 + 16);
  v10 = a2;
  v11 = 1;
  v12 = 0;
  v13 = ((_DWORD)v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v9 + 16 * v13);
  v15 = *v14;
  if ( a2 != *v14 )
  {
    while ( v15 != -4096 )
    {
      if ( v15 == -8192 && !v12 )
        v12 = v14;
      a6 = (unsigned int)(v11 + 1);
      v13 = ((_DWORD)v8 - 1) & (unsigned int)(v11 + v13);
      v14 = (__int64 *)(v9 + 16LL * (unsigned int)v13);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
    v19 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v13 = (unsigned int)(v19 + 1);
    if ( 4 * (int)v13 < (unsigned int)(3 * v8) )
    {
      v9 = (unsigned int)v8 >> 3;
      if ( (int)v8 - *(_DWORD *)(a1 + 28) - (int)v13 > (unsigned int)v9 )
      {
LABEL_15:
        *(_DWORD *)(v6 + 24) = v13;
        if ( *v12 != -4096 )
          --*(_DWORD *)(v6 + 28);
        v12[1] = 0;
        v17 = v12 + 1;
        *v12 = v10;
        a2 = (__int64)v86;
        goto LABEL_18;
      }
      sub_1062330((__int64)src, v8);
      v62 = *(_DWORD *)(a1 + 32);
      if ( v62 )
      {
        v63 = v62 - 1;
        v64 = *(_QWORD *)(v6 + 16);
        v8 = 0;
        v9 = (__int64)v86;
        v13 = (unsigned int)(*(_DWORD *)(v6 + 24) + 1);
        v65 = 1;
        v66 = (v62 - 1) & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
        v12 = (__int64 *)(v64 + 16LL * v66);
        v10 = *v12;
        if ( v86 == (_QWORD **)*v12 )
          goto LABEL_15;
        while ( v10 != -4096 )
        {
          if ( !v8 && v10 == -8192 )
            v8 = (__int64)v12;
          a6 = (unsigned int)(v65 + 1);
          v66 = v63 & (v65 + v66);
          v12 = (__int64 *)(v64 + 16LL * v66);
          v10 = *v12;
          if ( v86 == (_QWORD **)*v12 )
            goto LABEL_15;
          ++v65;
        }
        goto LABEL_79;
      }
      goto LABEL_117;
    }
LABEL_75:
    sub_1062330((__int64)src, 2 * v8);
    v57 = *(_DWORD *)(a1 + 32);
    if ( v57 )
    {
      v9 = (__int64)v86;
      v58 = v57 - 1;
      v59 = *(_QWORD *)(v6 + 16);
      v13 = (unsigned int)(*(_DWORD *)(v6 + 24) + 1);
      v60 = (v57 - 1) & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
      v12 = (__int64 *)(v59 + 16LL * v60);
      v10 = *v12;
      if ( v86 == (_QWORD **)*v12 )
        goto LABEL_15;
      v61 = 1;
      v8 = 0;
      while ( v10 != -4096 )
      {
        if ( !v8 && v10 == -8192 )
          v8 = (__int64)v12;
        a6 = (unsigned int)(v61 + 1);
        v60 = v58 & (v61 + v60);
        v12 = (__int64 *)(v59 + 16LL * v60);
        v10 = *v12;
        if ( v86 == (_QWORD **)*v12 )
          goto LABEL_15;
        ++v61;
      }
LABEL_79:
      v10 = v9;
      if ( v8 )
        v12 = (__int64 *)v8;
      goto LABEL_15;
    }
LABEL_117:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_3:
  v16 = (_QWORD *)v14[1];
  v17 = v14 + 1;
  if ( v16 )
    return v16;
LABEL_18:
  v16 = (_QWORD *)a2;
  if ( *(_BYTE *)(a2 + 8) != 15 || (*(_BYTE *)(a2 + 9) & 4) != 0 )
  {
    v91 = (__int64 *)v93;
    v92 = 0x400000000LL;
    v20 = *(unsigned int *)(a2 + 12);
    if ( !(_DWORD)v20 )
    {
      *v17 = a2;
      goto LABEL_21;
    }
    v78 = 1;
    v24 = (unsigned int)v20;
    goto LABEL_31;
  }
  if ( *(_BYTE *)(a3 + 28) )
  {
    v13 = *(unsigned int *)(a3 + 20);
    v21 = *(__int64 **)(a3 + 8);
    v9 = (__int64)&v21[v13];
    for ( i = *(_DWORD *)(a3 + 20); (__int64 *)v9 != v21; ++v21 )
    {
      v13 = *v21;
      if ( a2 == *v21 )
        goto LABEL_29;
    }
    if ( i < *(_DWORD *)(a3 + 16) )
    {
      *(_DWORD *)(a3 + 20) = i + 1;
      *(_QWORD *)v9 = a2;
      a2 = (__int64)v86;
      ++*(_QWORD *)a3;
      goto LABEL_58;
    }
  }
  sub_C8CC70(a3, a2, v13, v9, v8, a6);
  a2 = (__int64)v86;
  v54 = v53;
  v13 = (__int64)v86;
  if ( !v54 )
  {
LABEL_29:
    v23 = sub_BCC900(*(_QWORD **)v13);
    *v17 = v23;
    return (_QWORD *)v23;
  }
LABEL_58:
  v91 = (__int64 *)v93;
  v92 = 0x400000000LL;
  v24 = *(unsigned int *)(a2 + 12);
  v20 = v24;
  if ( !*(_DWORD *)(a2 + 12) )
  {
    v78 = 0;
    v35 = 0;
    v34 = 0;
    goto LABEL_41;
  }
  v78 = 0;
LABEL_31:
  if ( v24 > 4 )
  {
    nb = v20;
    sub_C8D5F0((__int64)&v91, v93, v24, 8u, v20, a6);
    LODWORD(v20) = nb;
  }
  v25 = &v91[(unsigned int)v92];
  for ( j = &v91[v24]; j != v25; ++v25 )
  {
    if ( v25 )
      *v25 = 0;
  }
  a2 = (__int64)v86;
  LODWORD(v92) = v20;
  n = *((_DWORD *)v86 + 3);
  if ( n )
  {
    v27 = a3;
    v28 = v86[2];
    v29 = v6;
    v30 = 0;
    v31 = 0;
    do
    {
      v32 = sub_10648E0(v29, v28[v30], v27);
      v91[v30] = v32;
      a2 = (__int64)v86;
      v28 = v86[2];
      v33 = v91[v30] != v28[v30];
      ++v30;
      v31 |= v33;
    }
    while ( v30 != n );
    v34 = v31;
    v6 = v29;
    v35 = v78 & (v34 ^ 1);
  }
  else
  {
    v35 = v78;
    v34 = 0;
  }
LABEL_41:
  v36 = *(_DWORD *)(v6 + 32);
  if ( !v36 )
  {
    ++*(_QWORD *)(v6 + 8);
    v87 = 0;
LABEL_95:
    na = v35;
    v67 = 2 * v36;
LABEL_96:
    sub_1062330((__int64)src, v67);
    sub_1061BC0((__int64)src, (__int64 *)&v86, &v87);
    v16 = v86;
    v39 = v87;
    v35 = na;
    a2 = (unsigned int)(*(_DWORD *)(v6 + 24) + 1);
    goto LABEL_71;
  }
  v37 = *(_QWORD *)(v6 + 16);
  v16 = (_QWORD *)a2;
  v38 = 1;
  v39 = 0;
  v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v41 = (_QWORD *)(v37 + 16LL * v40);
  v42 = *v41;
  if ( *v41 == a2 )
  {
LABEL_43:
    v43 = v41 + 1;
    goto LABEL_44;
  }
  while ( v42 != -4096 )
  {
    if ( !v39 && v42 == -8192 )
      v39 = v41;
    v40 = (v36 - 1) & (v38 + v40);
    v41 = (_QWORD *)(v37 + 16LL * v40);
    v42 = *v41;
    if ( a2 == *v41 )
      goto LABEL_43;
    ++v38;
  }
  if ( !v39 )
    v39 = v41;
  v56 = *(_DWORD *)(v6 + 24);
  ++*(_QWORD *)(v6 + 8);
  a2 = (unsigned int)(v56 + 1);
  v87 = (char *)v39;
  if ( 4 * (int)a2 >= 3 * v36 )
    goto LABEL_95;
  if ( v36 - *(_DWORD *)(v6 + 28) - (unsigned int)a2 <= v36 >> 3 )
  {
    na = v35;
    v67 = v36;
    goto LABEL_96;
  }
LABEL_71:
  *(_DWORD *)(v6 + 24) = a2;
  if ( *v39 != -4096 )
    --*(_DWORD *)(v6 + 28);
  *v39 = v16;
  v43 = v39 + 1;
  v16 = v86;
  v39[1] = 0;
LABEL_44:
  if ( v35 )
  {
LABEL_50:
    *v43 = v16;
    goto LABEL_21;
  }
  v44 = *((_BYTE *)v16 + 8);
  if ( v44 == 16 )
  {
    a2 = v16[4];
    v16 = sub_BCD420((__int64 *)*v91, a2);
    goto LABEL_50;
  }
  if ( v44 > 0x10u )
  {
    if ( (unsigned __int8)(v44 - 17) <= 1u )
    {
      v51 = *((_DWORD *)v16 + 8);
      BYTE4(v87) = v44 == 18;
      LODWORD(v87) = v51;
      a2 = (__int64)v87;
      v52 = sub_BCE1B0((__int64 *)*v91, (__int64)v87);
      *v43 = v52;
      v16 = (_QWORD *)v52;
      goto LABEL_21;
    }
LABEL_118:
    BUG();
  }
  if ( v44 == 13 )
  {
    a2 = (__int64)(v91 + 1);
    v45 = sub_BCF480((__int64 *)*v91, v91 + 1, (unsigned int)v92 - 1LL, *((_DWORD *)v16 + 2) >> 8 != 0);
    *v43 = v45;
    v16 = (_QWORD *)v45;
    goto LABEL_21;
  }
  if ( v44 != 15 )
    goto LABEL_118;
  v46 = *((_DWORD *)v16 + 2);
  v47 = v46 >> 9;
  v48 = v46 >> 8;
  v49 = v47 & 1;
  if ( v78 )
  {
    a2 = (__int64)v91;
    v55 = sub_BD0B90((_QWORD *)*v16, v91, (unsigned int)v92, v49);
    *v43 = v55;
    v16 = v55;
    goto LABEL_21;
  }
  v50 = *(_QWORD *)(v6 + 632);
  if ( (v48 & 1) == 0 )
  {
    a2 = (__int64)v16;
    sub_10641A0(v50, (__int64)v16);
    v16 = v86;
    *v43 = v86;
    goto LABEL_21;
  }
  v68 = sub_1061FD0(v50, (__int64)v91, (unsigned int)v92, v49);
  v69 = v68;
  if ( v68 )
  {
    v70 = (__int64 **)v16;
    a2 = (__int64)byte_3F871B3;
    v16 = (_QWORD *)v68;
    sub_BCB4B0(v70, byte_3F871B3, 0);
    *v43 = v69;
    goto LABEL_21;
  }
  if ( !v34 )
  {
    a2 = (__int64)v16;
    sub_1063E40(*(_QWORD *)(v6 + 632), (__int64)v16);
    v16 = v86;
    *v43 = v86;
    goto LABEL_21;
  }
  v71 = (__int64 **)sub_BD0E80(*v86, v91, (unsigned int)v92, byte_3F871B3, 0, (v16[1] & 0x200) != 0);
  if ( v16[3] )
  {
    v72 = sub_BCB490((__int64)v16);
    v89 = 16;
    v88 = 0;
    v75 = (const void *)v72;
    v87 = v90;
    if ( v73 > 0x10 )
    {
      srca = (void *)v72;
      nc = v73;
      sub_C8D290((__int64)&v87, v90, v73, 1u, v72, v74);
      v73 = nc;
      v75 = srca;
      v77 = &v87[v88];
    }
    else
    {
      if ( !v73 )
        goto LABEL_105;
      v77 = v90;
    }
    nd = v73;
    memcpy(v77, v75, v73);
    v73 = nd;
LABEL_105:
    v88 += v73;
    sub_BCB4B0((__int64 **)v16, byte_3F871B3, 0);
    v76 = v87;
    sub_BCB4B0(v71, v87, v88);
    if ( v87 != v90 )
      _libc_free(v87, v76);
  }
  a2 = (__int64)v71;
  v16 = v71;
  sub_1063E40(*(_QWORD *)(v6 + 632), (__int64)v71);
  *v43 = v71;
LABEL_21:
  if ( v91 != (__int64 *)v93 )
    _libc_free(v91, a2);
  return v16;
}
