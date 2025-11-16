// Function: sub_2ED6C00
// Address: 0x2ed6c00
//
__int64 *__fastcall sub_2ED6C00(__int64 a1, unsigned __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned int v8; // edx
  __int64 v9; // rcx
  _QWORD *v10; // r13
  int v12; // r8d
  __int64 v13; // rsi
  unsigned __int64 v14; // rbx
  __int64 v15; // rcx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  __int64 i; // rsi
  __int16 v19; // dx
  unsigned __int64 v20; // rcx
  __int64 j; // rbx
  _QWORD *v22; // rdi
  _BYTE *v23; // rbx
  unsigned __int64 *v24; // r13
  unsigned __int64 *v25; // rsi
  _BYTE *v26; // rdx
  unsigned __int64 *v27; // rbx
  unsigned int v28; // r14d
  unsigned int v29; // r9d
  unsigned __int64 *v30; // rax
  unsigned __int64 v31; // r8
  unsigned __int64 *v32; // r13
  unsigned __int64 v33; // r15
  size_t v34; // r14
  _BYTE *v35; // r8
  char *v36; // rsi
  char *v37; // r14
  unsigned int v38; // r13d
  unsigned int v39; // r9d
  unsigned __int64 *v40; // rax
  unsigned __int64 v41; // r8
  __int64 v42; // rcx
  __int64 v43; // rdx
  unsigned __int64 *v44; // r15
  unsigned __int64 v45; // r14
  __int64 v46; // rax
  char *v47; // rdi
  const void *v48; // rax
  const void *v49; // rsi
  size_t v50; // r15
  int v51; // edx
  int v52; // esi
  _QWORD *v53; // rcx
  unsigned int v54; // eax
  _QWORD *v55; // r13
  char **v56; // r8
  __int64 v57; // rax
  char *v58; // rbx
  char *v59; // rsi
  int v60; // r9d
  int v61; // r9d
  __int64 v62; // r11
  int v63; // edx
  unsigned int v64; // ecx
  unsigned __int64 v65; // r8
  int v66; // edi
  unsigned __int64 *v67; // rcx
  int v68; // edx
  int v69; // r8d
  int v70; // r8d
  int v71; // edi
  unsigned int v72; // ecx
  __int64 v73; // r11
  unsigned __int64 v74; // r9
  int v75; // edi
  unsigned __int64 *v76; // rcx
  int v77; // edx
  int v78; // edx
  int v79; // r9d
  int v80; // r9d
  int v81; // edi
  unsigned __int64 *v82; // rsi
  unsigned int v83; // ecx
  __int64 v84; // r11
  unsigned __int64 v85; // r8
  int v86; // r8d
  int v87; // r8d
  __int64 v88; // r11
  unsigned int v89; // ecx
  unsigned __int64 v90; // r9
  int v91; // edi
  int v92; // edi
  __int64 *v93; // [rsp+20h] [rbp-640h]
  _BYTE *v94; // [rsp+38h] [rbp-628h]
  __int64 v95; // [rsp+40h] [rbp-620h]
  __int64 v96; // [rsp+40h] [rbp-620h]
  _QWORD v98[6]; // [rsp+50h] [rbp-610h] BYREF
  unsigned __int64 *v99; // [rsp+80h] [rbp-5E0h]
  __int16 v100; // [rsp+88h] [rbp-5D8h]
  char v101; // [rsp+8Ah] [rbp-5D6h]
  __int64 v102; // [rsp+90h] [rbp-5D0h]
  unsigned __int64 v103; // [rsp+98h] [rbp-5C8h]
  __int64 v104; // [rsp+A0h] [rbp-5C0h]
  __int64 v105; // [rsp+A8h] [rbp-5B8h]
  _BYTE *v106; // [rsp+B0h] [rbp-5B0h]
  __int64 v107; // [rsp+B8h] [rbp-5A8h]
  _BYTE v108[192]; // [rsp+C0h] [rbp-5A0h] BYREF
  unsigned __int64 v109; // [rsp+180h] [rbp-4E0h]
  int v110; // [rsp+188h] [rbp-4D8h]
  int v111; // [rsp+190h] [rbp-4D0h]
  _BYTE *v112; // [rsp+198h] [rbp-4C8h]
  __int64 v113; // [rsp+1A0h] [rbp-4C0h]
  _BYTE v114[32]; // [rsp+1A8h] [rbp-4B8h] BYREF
  unsigned __int64 v115; // [rsp+1C8h] [rbp-498h]
  int v116; // [rsp+1D0h] [rbp-490h]
  unsigned __int64 v117; // [rsp+1D8h] [rbp-488h]
  __int64 v118; // [rsp+1E0h] [rbp-480h]
  __int64 v119; // [rsp+1E8h] [rbp-478h]
  unsigned __int64 v120[3]; // [rsp+1F0h] [rbp-470h] BYREF
  _BYTE *v121; // [rsp+208h] [rbp-458h]
  __int64 v122; // [rsp+210h] [rbp-450h]
  _BYTE v123[192]; // [rsp+218h] [rbp-448h] BYREF
  _BYTE *v124; // [rsp+2D8h] [rbp-388h]
  __int64 v125; // [rsp+2E0h] [rbp-380h]
  _BYTE v126[192]; // [rsp+2E8h] [rbp-378h] BYREF
  __int64 v127; // [rsp+3A8h] [rbp-2B8h]
  __int64 v128; // [rsp+3B0h] [rbp-2B0h]
  char **v129; // [rsp+3C0h] [rbp-2A0h] BYREF
  __int64 v130; // [rsp+3C8h] [rbp-298h]
  char *v131; // [rsp+3D0h] [rbp-290h] BYREF
  char *v132; // [rsp+3D8h] [rbp-288h]
  _BYTE *v133; // [rsp+490h] [rbp-1D0h]
  __int64 v134; // [rsp+498h] [rbp-1C8h]
  _BYTE v135[192]; // [rsp+4A0h] [rbp-1C0h] BYREF
  _BYTE *v136; // [rsp+560h] [rbp-100h]
  __int64 v137; // [rsp+568h] [rbp-F8h]
  _BYTE v138[240]; // [rsp+570h] [rbp-F0h] BYREF

  v4 = *(unsigned int *)(a1 + 1240);
  v5 = *(_QWORD *)(a1 + 1224);
  if ( !(_DWORD)v4 )
  {
LABEL_9:
    v93 = (__int64 *)(v5 + 32 * v4);
    goto LABEL_10;
  }
  v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v93 = (__int64 *)(v5 + 32LL * v8);
  v9 = *v93;
  if ( a2 != *v93 )
  {
    v12 = 1;
    while ( v9 != -4096 )
    {
      v8 = (v4 - 1) & (v12 + v8);
      v93 = (__int64 *)(v5 + 32LL * v8);
      v9 = *v93;
      if ( a2 == *v93 )
        goto LABEL_3;
      ++v12;
    }
    goto LABEL_9;
  }
LABEL_3:
  if ( a3 && v93 != (__int64 *)(v5 + 32 * v4) )
    return v93 + 1;
LABEL_10:
  v99 = v120;
  v100 = 0;
  v121 = v123;
  v106 = v108;
  v112 = v114;
  v124 = v126;
  memset(v120, 0, sizeof(v120));
  v122 = 0x800000000LL;
  v125 = 0x800000000LL;
  v127 = 0;
  v128 = 0;
  memset(v98, 0, sizeof(v98));
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v107 = 0x800000000LL;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v113 = 0x800000000LL;
  v115 = 0;
  v13 = *(_QWORD *)(a2 + 32);
  v14 = a2 + 48;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  sub_2F796A0((unsigned int)v98, v13, a1 + 88, 0, a2, a2 + 48, 0, 1);
  v95 = *(_QWORD *)(a2 + 56);
  if ( v95 != a2 + 48 )
  {
    do
    {
      v15 = *(_QWORD *)v14;
      v16 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v16 )
        goto LABEL_177;
      v17 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v16 & 4) == 0 && (*(_BYTE *)(v16 + 44) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v16; ; i = *(_QWORD *)v17 )
        {
          v17 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v17 + 44) & 4) == 0 )
            break;
        }
      }
      v19 = *(_WORD *)(v17 + 68);
      if ( (unsigned __int16)(v19 - 14) > 4u && v19 != 24 )
      {
        v42 = *(_QWORD *)(a1 + 24);
        v43 = *(_QWORD *)(a1 + 16);
        v130 = 0x800000000LL;
        v136 = v138;
        v133 = v135;
        v129 = &v131;
        v134 = 0x800000000LL;
        v137 = 0x800000000LL;
        sub_2F75980(&v129, v17, v43, v42, 0, 0);
        sub_2F771D0(v98);
        sub_2F78160(v98, &v129, 0);
        if ( v136 != v138 )
          _libc_free((unsigned __int64)v136);
        if ( v133 != v135 )
          _libc_free((unsigned __int64)v133);
        if ( v129 != &v131 )
          _libc_free((unsigned __int64)v129);
        v15 = *(_QWORD *)v14;
      }
      v20 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v20 )
LABEL_177:
        BUG();
      v14 = v20;
      if ( (*(_QWORD *)v20 & 4) == 0 && (*(_BYTE *)(v20 + 44) & 4) != 0 )
      {
        for ( j = *(_QWORD *)v20; ; j = *(_QWORD *)v14 )
        {
          v14 = j & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v14 + 44) & 4) == 0 )
            break;
        }
      }
    }
    while ( v95 != v14 );
  }
  v22 = v98;
  v96 = a1 + 1216;
  sub_2F75920(v98);
  v23 = *(_BYTE **)(a1 + 1224);
  v24 = (unsigned __int64 *)*(unsigned int *)(a1 + 1240);
  v25 = v24;
  v26 = v23;
  if ( v93 == (__int64 *)&v23[32 * (_QWORD)v24] )
  {
    v44 = v99;
    v129 = (char **)a2;
    v45 = v99[1] - *v99;
    v130 = 0;
    v131 = 0;
    v132 = 0;
    if ( v45 )
    {
      if ( v45 > 0x7FFFFFFFFFFFFFFCLL )
        goto LABEL_173;
      v46 = sub_22077B0(v45);
      v23 = *(_BYTE **)(a1 + 1224);
      LODWORD(v24) = *(_DWORD *)(a1 + 1240);
      v47 = (char *)v46;
    }
    else
    {
      v45 = 0;
      v47 = 0;
    }
    v130 = (__int64)v47;
    v131 = v47;
    v132 = &v47[v45];
    v48 = (const void *)v44[1];
    v49 = (const void *)*v44;
    v50 = (size_t)v48 - *v44;
    if ( v48 != v49 )
      v47 = (char *)memmove(v47, v49, v50);
    v131 = &v47[v50];
    if ( (_DWORD)v24 )
    {
      v51 = (_DWORD)v24 - 1;
      v52 = 1;
      v53 = 0;
      v54 = ((_DWORD)v24 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
      v55 = &v23[32 * v54];
      v56 = (char **)*v55;
      if ( v129 == (char **)*v55 )
      {
LABEL_72:
        if ( v47 )
          j_j___libc_free_0((unsigned __int64)v47);
        goto LABEL_74;
      }
      while ( v56 != (char **)-4096LL )
      {
        if ( !v53 && v56 == (char **)-8192LL )
          v53 = v55;
        v54 = v51 & (v52 + v54);
        v55 = &v23[32 * v54];
        v56 = (char **)*v55;
        if ( v129 == (char **)*v55 )
          goto LABEL_72;
        ++v52;
      }
      if ( v53 )
        v55 = v53;
    }
    else
    {
      v55 = 0;
    }
    v55 = sub_2ED6A60(v96, &v129, v55);
    *v55 = v129;
    v55[1] = v130;
    v55[2] = v131;
    v55[3] = v132;
LABEL_74:
    v10 = v55 + 1;
    goto LABEL_40;
  }
  v27 = v99;
  if ( (_DWORD)v24 )
  {
    v28 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v29 = ((_DWORD)v24 - 1) & v28;
    v30 = (unsigned __int64 *)&v26[32 * v29];
    v31 = *v30;
    if ( a2 == *v30 )
    {
LABEL_29:
      v32 = v30 + 1;
      if ( v99 == v30 + 1 )
        goto LABEL_38;
      goto LABEL_30;
    }
    v66 = 1;
    v67 = 0;
    while ( v31 != -4096 )
    {
      if ( v31 == -8192 && !v67 )
        v67 = v30;
      v29 = ((_DWORD)v24 - 1) & (v66 + v29);
      v30 = (unsigned __int64 *)&v26[32 * v29];
      v31 = *v30;
      if ( a2 == *v30 )
        goto LABEL_29;
      ++v66;
    }
    v68 = *(_DWORD *)(a1 + 1232);
    if ( v67 )
      v30 = v67;
    ++*(_QWORD *)(a1 + 1216);
    v63 = v68 + 1;
    if ( 4 * v63 < (unsigned int)(3 * (_DWORD)v24) )
    {
      if ( (int)v24 - *(_DWORD *)(a1 + 1236) - v63 <= (unsigned int)v24 >> 3 )
      {
        sub_2ED6830(v96, (int)v24);
        v69 = *(_DWORD *)(a1 + 1240);
        if ( !v69 )
          goto LABEL_176;
        v70 = v69 - 1;
        v71 = 1;
        v25 = 0;
        v72 = v70 & v28;
        v73 = *(_QWORD *)(a1 + 1224);
        v63 = *(_DWORD *)(a1 + 1232) + 1;
        v30 = (unsigned __int64 *)(v73 + 32LL * (v70 & v28));
        v74 = *v30;
        if ( a2 != *v30 )
        {
          while ( v74 != -4096 )
          {
            if ( !v25 && v74 == -8192 )
              v25 = v30;
            v72 = v70 & (v71 + v72);
            v30 = (unsigned __int64 *)(v73 + 32LL * v72);
            v74 = *v30;
            if ( a2 == *v30 )
              goto LABEL_94;
            ++v71;
          }
LABEL_107:
          if ( v25 )
            v30 = v25;
          goto LABEL_94;
        }
      }
      goto LABEL_94;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 1216);
  }
  v25 = (unsigned __int64 *)(unsigned int)(2 * (_DWORD)v24);
  sub_2ED6830(v96, (int)v25);
  v60 = *(_DWORD *)(a1 + 1240);
  if ( !v60 )
    goto LABEL_176;
  v61 = v60 - 1;
  v62 = *(_QWORD *)(a1 + 1224);
  v63 = *(_DWORD *)(a1 + 1232) + 1;
  v64 = v61 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v30 = (unsigned __int64 *)(v62 + 32LL * v64);
  v65 = *v30;
  if ( a2 != *v30 )
  {
    v92 = 1;
    v25 = 0;
    while ( v65 != -4096 )
    {
      if ( !v25 && v65 == -8192 )
        v25 = v30;
      v64 = v61 & (v92 + v64);
      v30 = (unsigned __int64 *)(v62 + 32LL * v64);
      v65 = *v30;
      if ( a2 == *v30 )
        goto LABEL_94;
      ++v92;
    }
    goto LABEL_107;
  }
LABEL_94:
  *(_DWORD *)(a1 + 1232) = v63;
  if ( *v30 != -4096 )
    --*(_DWORD *)(a1 + 1236);
  v32 = v30 + 1;
  v30[1] = 0;
  v30[2] = 0;
  *v30 = a2;
  v30[3] = 0;
  if ( v27 == v30 + 1 )
    goto LABEL_36;
LABEL_30:
  v26 = (_BYTE *)v27[1];
  v33 = *v27;
  v22 = (_QWORD *)*v32;
  v34 = (size_t)&v26[-*v27];
  if ( v34 > v32[2] - *v32 )
  {
    if ( !v34 )
    {
      v58 = 0;
      goto LABEL_86;
    }
    if ( v34 <= 0x7FFFFFFFFFFFFFFCLL )
    {
      v94 = (_BYTE *)v27[1];
      v57 = sub_22077B0((unsigned __int64)&v94[-*v27]);
      v26 = v94;
      v58 = (char *)v57;
LABEL_86:
      if ( v26 != (_BYTE *)v33 )
        memcpy(v58, (const void *)v33, v34);
      if ( *v32 )
        j_j___libc_free_0(*v32);
      v37 = &v58[v34];
      *v32 = (unsigned __int64)v58;
      v32[2] = (unsigned __int64)v37;
      goto LABEL_35;
    }
LABEL_173:
    sub_4261EA(v22, v25, v26);
  }
  v35 = (_BYTE *)v32[1];
  v36 = (char *)(v35 - (_BYTE *)v22);
  if ( v34 > v35 - (_BYTE *)v22 )
  {
    if ( v36 )
    {
      memmove(v22, (const void *)*v27, v32[1] - (_QWORD)v22);
      v35 = (_BYTE *)v32[1];
      v22 = (_QWORD *)*v32;
      v26 = (_BYTE *)v27[1];
      v33 = *v27;
      v36 = &v35[-*v32];
    }
    v59 = &v36[v33];
    if ( v26 != v59 )
    {
      memmove(v35, v59, v26 - v59);
      v37 = (char *)(*v32 + v34);
      goto LABEL_35;
    }
  }
  else if ( v26 != (_BYTE *)v33 )
  {
    memmove(v22, (const void *)*v27, v27[1] - *v27);
    v22 = (_QWORD *)*v32;
  }
  v37 = (char *)v22 + v34;
LABEL_35:
  v32[1] = (unsigned __int64)v37;
LABEL_36:
  LODWORD(v25) = *(_DWORD *)(a1 + 1240);
  if ( !(_DWORD)v25 )
  {
    ++*(_QWORD *)(a1 + 1216);
    goto LABEL_126;
  }
  v26 = *(_BYTE **)(a1 + 1224);
LABEL_38:
  v38 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v39 = ((_DWORD)v25 - 1) & v38;
  v40 = (unsigned __int64 *)&v26[32 * v39];
  v41 = *v40;
  if ( a2 != *v40 )
  {
    v75 = 1;
    v76 = 0;
    while ( v41 != -4096 )
    {
      if ( v41 == -8192 && !v76 )
        v76 = v40;
      v39 = ((_DWORD)v25 - 1) & (v75 + v39);
      v40 = (unsigned __int64 *)&v26[32 * v39];
      v41 = *v40;
      if ( a2 == *v40 )
        goto LABEL_39;
      ++v75;
    }
    v77 = *(_DWORD *)(a1 + 1232);
    if ( v76 )
      v40 = v76;
    ++*(_QWORD *)(a1 + 1216);
    v78 = v77 + 1;
    if ( 4 * v78 < (unsigned int)(3 * (_DWORD)v25) )
    {
      if ( (int)v25 - *(_DWORD *)(a1 + 1236) - v78 > (unsigned int)v25 >> 3 )
      {
LABEL_116:
        *(_DWORD *)(a1 + 1232) = v78;
        if ( *v40 != -4096 )
          --*(_DWORD *)(a1 + 1236);
        v40[1] = 0;
        v40[2] = 0;
        *v40 = a2;
        v40[3] = 0;
        goto LABEL_39;
      }
      sub_2ED6830(v96, (int)v25);
      v79 = *(_DWORD *)(a1 + 1240);
      if ( v79 )
      {
        v80 = v79 - 1;
        v81 = 1;
        v82 = 0;
        v83 = v80 & v38;
        v84 = *(_QWORD *)(a1 + 1224);
        v78 = *(_DWORD *)(a1 + 1232) + 1;
        v40 = (unsigned __int64 *)(v84 + 32LL * (v80 & v38));
        v85 = *v40;
        if ( a2 == *v40 )
          goto LABEL_116;
        while ( v85 != -4096 )
        {
          if ( !v82 && v85 == -8192 )
            v82 = v40;
          v83 = v80 & (v81 + v83);
          v40 = (unsigned __int64 *)(v84 + 32LL * v83);
          v85 = *v40;
          if ( a2 == *v40 )
            goto LABEL_116;
          ++v81;
        }
LABEL_122:
        if ( v82 )
          v40 = v82;
        goto LABEL_116;
      }
      goto LABEL_176;
    }
LABEL_126:
    sub_2ED6830(v96, 2 * (_DWORD)v25);
    v86 = *(_DWORD *)(a1 + 1240);
    if ( v86 )
    {
      v87 = v86 - 1;
      v88 = *(_QWORD *)(a1 + 1224);
      v78 = *(_DWORD *)(a1 + 1232) + 1;
      v89 = v87 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v40 = (unsigned __int64 *)(v88 + 32LL * v89);
      v90 = *v40;
      if ( a2 == *v40 )
        goto LABEL_116;
      v91 = 1;
      v82 = 0;
      while ( v90 != -4096 )
      {
        if ( !v82 && v90 == -8192 )
          v82 = v40;
        v89 = v87 & (v91 + v89);
        v40 = (unsigned __int64 *)(v88 + 32LL * v89);
        v90 = *v40;
        if ( a2 == *v40 )
          goto LABEL_116;
        ++v91;
      }
      goto LABEL_122;
    }
LABEL_176:
    ++*(_DWORD *)(a1 + 1232);
    BUG();
  }
LABEL_39:
  v10 = v40 + 1;
LABEL_40:
  if ( v117 )
    j_j___libc_free_0(v117);
  if ( v115 )
    _libc_free(v115);
  if ( v112 != v114 )
    _libc_free((unsigned __int64)v112);
  if ( v109 )
    _libc_free(v109);
  if ( v106 != v108 )
    _libc_free((unsigned __int64)v106);
  if ( v103 )
    j_j___libc_free_0(v103);
  if ( v124 != v126 )
    _libc_free((unsigned __int64)v124);
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  if ( v120[0] )
    j_j___libc_free_0(v120[0]);
  return v10;
}
