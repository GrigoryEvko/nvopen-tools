// Function: sub_1EDD780
// Address: 0x1edd780
//
__int64 __fastcall sub_1EDD780(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  int v4; // r8d
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  __int64 v8; // r13
  int v9; // r8d
  unsigned int v10; // r9d
  __int64 v11; // r13
  char v12; // bl
  int v13; // edx
  int v14; // ecx
  _BYTE *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r13
  _BYTE *v20; // rax
  _BYTE *i; // rdx
  __int64 v22; // rax
  int v23; // ecx
  int v24; // esi
  __int64 v25; // rax
  unsigned __int64 v26; // rbx
  __int64 v27; // rdx
  _BYTE *v28; // rdi
  __int64 v29; // rcx
  unsigned __int64 v30; // r12
  _BYTE *v31; // rax
  _BYTE *j; // rdx
  int v33; // ebx
  __int64 k; // r14
  _BYTE *v35; // rdi
  unsigned int v37; // r9d
  __int64 v38; // rcx
  unsigned int v39; // r9d
  __int64 v40; // rdx
  int v41; // ebx
  __int64 m; // r14
  __int64 v43; // r10
  __int64 v44; // rsi
  _QWORD *v45; // rcx
  _QWORD *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rsi
  _QWORD *v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rcx
  __int64 v53; // rbx
  __int64 v54; // r12
  __int64 v55; // r13
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // r13
  _BYTE *v59; // rax
  __int64 v60; // rbx
  unsigned __int64 v61; // r14
  __int64 v62; // r15
  __int64 v63; // rax
  __int64 *v64; // rdx
  __int64 v65; // rsi
  unsigned int v66; // edi
  unsigned int v67; // ecx
  __int64 v68; // rax
  int v69; // ecx
  __int64 v70; // rbx
  __int64 v71; // rcx
  __int64 v72; // r9
  int v73; // r8d
  int v74; // r9d
  int v75; // r8d
  int v76; // r9d
  __int64 v77; // r9
  __int64 v78; // r12
  int v79; // r14d
  unsigned __int64 v80; // rdx
  unsigned int v81; // eax
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rcx
  __int64 v85; // r13
  unsigned int v86; // ebx
  __int64 v87; // rcx
  _QWORD *v88; // rax
  _QWORD *v89; // rsi
  __int64 v90; // rdx
  unsigned int v91; // r12d
  unsigned int v92; // ecx
  __int64 v93; // [rsp+18h] [rbp-568h]
  int v94; // [rsp+18h] [rbp-568h]
  __int64 v95; // [rsp+20h] [rbp-560h]
  __int64 *v96; // [rsp+20h] [rbp-560h]
  __int64 v97; // [rsp+20h] [rbp-560h]
  __int64 v98; // [rsp+38h] [rbp-548h]
  unsigned __int8 v99; // [rsp+38h] [rbp-548h]
  __int64 v100; // [rsp+38h] [rbp-548h]
  unsigned int v101; // [rsp+50h] [rbp-530h]
  unsigned int v102; // [rsp+58h] [rbp-528h]
  int v103; // [rsp+58h] [rbp-528h]
  __int64 v104; // [rsp+60h] [rbp-520h]
  int v105; // [rsp+60h] [rbp-520h]
  __int64 v106; // [rsp+68h] [rbp-518h]
  __int64 v107; // [rsp+68h] [rbp-518h]
  __int64 v108; // [rsp+68h] [rbp-518h]
  _BYTE *v109; // [rsp+70h] [rbp-510h] BYREF
  __int64 v110; // [rsp+78h] [rbp-508h]
  _BYTE v111[32]; // [rsp+80h] [rbp-500h] BYREF
  __int64 *v112; // [rsp+A0h] [rbp-4E0h] BYREF
  __int64 v113; // [rsp+A8h] [rbp-4D8h]
  _BYTE v114[64]; // [rsp+B0h] [rbp-4D0h] BYREF
  unsigned __int64 v115[2]; // [rsp+F0h] [rbp-490h] BYREF
  _BYTE v116[64]; // [rsp+100h] [rbp-480h] BYREF
  unsigned __int64 v117[2]; // [rsp+140h] [rbp-440h] BYREF
  _BYTE v118[128]; // [rsp+150h] [rbp-430h] BYREF
  __int64 v119; // [rsp+1D0h] [rbp-3B0h] BYREF
  int v120; // [rsp+1D8h] [rbp-3A8h]
  int v121; // [rsp+1DCh] [rbp-3A4h]
  int v122; // [rsp+1E0h] [rbp-3A0h]
  char v123; // [rsp+1E4h] [rbp-39Ch]
  char v124; // [rsp+1E5h] [rbp-39Bh]
  unsigned __int64 *v125; // [rsp+1E8h] [rbp-398h]
  __int64 v126; // [rsp+1F0h] [rbp-390h]
  __int64 v127; // [rsp+1F8h] [rbp-388h]
  __int64 v128; // [rsp+200h] [rbp-380h]
  __int64 v129; // [rsp+208h] [rbp-378h]
  _BYTE *v130; // [rsp+210h] [rbp-370h] BYREF
  __int64 v131; // [rsp+218h] [rbp-368h]
  _BYTE s[32]; // [rsp+220h] [rbp-360h] BYREF
  _BYTE *v133; // [rsp+240h] [rbp-340h] BYREF
  __int64 v134; // [rsp+248h] [rbp-338h]
  _BYTE v135[320]; // [rsp+250h] [rbp-330h] BYREF
  __int64 v136; // [rsp+390h] [rbp-1F0h] BYREF
  int v137; // [rsp+398h] [rbp-1E8h]
  int v138; // [rsp+39Ch] [rbp-1E4h]
  int v139; // [rsp+3A0h] [rbp-1E0h]
  char v140; // [rsp+3A4h] [rbp-1DCh]
  char v141; // [rsp+3A5h] [rbp-1DBh]
  unsigned __int64 *v142; // [rsp+3A8h] [rbp-1D8h]
  __int64 v143; // [rsp+3B0h] [rbp-1D0h]
  __int64 v144; // [rsp+3B8h] [rbp-1C8h]
  __int64 v145; // [rsp+3C0h] [rbp-1C0h]
  __int64 v146; // [rsp+3C8h] [rbp-1B8h]
  _BYTE *v147; // [rsp+3D0h] [rbp-1B0h] BYREF
  __int64 v148; // [rsp+3D8h] [rbp-1A8h]
  _BYTE v149[32]; // [rsp+3E0h] [rbp-1A0h] BYREF
  _BYTE *v150; // [rsp+400h] [rbp-180h] BYREF
  __int64 v151; // [rsp+408h] [rbp-178h]
  _BYTE v152[368]; // [rsp+410h] [rbp-170h] BYREF

  v2 = a1;
  v3 = a2;
  v4 = *(_DWORD *)(a2 + 12);
  v5 = *(_QWORD *)(a1 + 272);
  v117[0] = (unsigned __int64)v118;
  v117[1] = 0x1000000000LL;
  v6 = *(unsigned int *)(v5 + 408);
  v7 = v4 & 0x7FFFFFFF;
  v8 = v4 & 0x7FFFFFFF;
  if ( (v4 & 0x7FFFFFFFu) >= (unsigned int)v6 || (v104 = *(_QWORD *)(*(_QWORD *)(v5 + 400) + 8LL * v7)) == 0 )
  {
    v39 = v7 + 1;
    if ( (unsigned int)v6 < v7 + 1 )
    {
      v43 = v39;
      if ( v39 < v6 )
      {
        *(_DWORD *)(v5 + 408) = v39;
      }
      else if ( v39 > v6 )
      {
        if ( v39 > (unsigned __int64)*(unsigned int *)(v5 + 412) )
        {
          v102 = v7 + 1;
          v105 = v4;
          v107 = v39;
          sub_16CD150(v5 + 400, (const void *)(v5 + 416), v39, 8, v4, v39);
          v6 = *(unsigned int *)(v5 + 408);
          v39 = v102;
          v4 = v105;
          v43 = v107;
        }
        v40 = *(_QWORD *)(v5 + 400);
        v44 = *(_QWORD *)(v5 + 416);
        v45 = (_QWORD *)(v40 + 8 * v43);
        v46 = (_QWORD *)(v40 + 8 * v6);
        if ( v45 != v46 )
        {
          do
            *v46++ = v44;
          while ( v45 != v46 );
          v40 = *(_QWORD *)(v5 + 400);
        }
        *(_DWORD *)(v5 + 408) = v39;
        goto LABEL_48;
      }
    }
    v40 = *(_QWORD *)(v5 + 400);
LABEL_48:
    *(_QWORD *)(v40 + 8 * v8) = sub_1DBA290(v4);
    v104 = *(_QWORD *)(*(_QWORD *)(v5 + 400) + 8 * v8);
    sub_1DBB110((_QWORD *)v5, v104);
    v5 = *(_QWORD *)(a1 + 272);
    v6 = *(unsigned int *)(v5 + 408);
  }
  v9 = *(_DWORD *)(v3 + 8);
  v10 = v9 & 0x7FFFFFFF;
  v11 = v9 & 0x7FFFFFFF;
  if ( (v9 & 0x7FFFFFFFu) >= (unsigned int)v6 || (v106 = *(_QWORD *)(*(_QWORD *)(v5 + 400) + 8LL * v10)) == 0 )
  {
    v37 = v10 + 1;
    if ( v37 > (unsigned int)v6 )
    {
      v47 = v37;
      if ( v37 < v6 )
      {
        *(_DWORD *)(v5 + 408) = v37;
      }
      else if ( v37 > v6 )
      {
        if ( v37 > (unsigned __int64)*(unsigned int *)(v5 + 412) )
        {
          v101 = v37;
          v103 = *(_DWORD *)(v3 + 8);
          v108 = v37;
          sub_16CD150(v5 + 400, (const void *)(v5 + 416), v37, 8, v9, v37);
          v6 = *(unsigned int *)(v5 + 408);
          v37 = v101;
          v9 = v103;
          v47 = v108;
        }
        v38 = *(_QWORD *)(v5 + 400);
        v48 = *(_QWORD *)(v5 + 416);
        v49 = (_QWORD *)(v38 + 8 * v47);
        v50 = (_QWORD *)(v38 + 8 * v6);
        if ( v49 != v50 )
        {
          do
            *v50++ = v48;
          while ( v49 != v50 );
          v38 = *(_QWORD *)(v5 + 400);
        }
        *(_DWORD *)(v5 + 408) = v37;
        goto LABEL_45;
      }
    }
    v38 = *(_QWORD *)(v5 + 400);
LABEL_45:
    *(_QWORD *)(v38 + 8 * v11) = sub_1DBA290(v9);
    v106 = *(_QWORD *)(*(_QWORD *)(v5 + 400) + 8 * v11);
    sub_1DBB110((_QWORD *)v5, v106);
    v5 = *(_QWORD *)(a1 + 272);
  }
  v12 = *(_BYTE *)(*(_QWORD *)(a1 + 248) + 16LL);
  if ( v12 )
    v12 = *(_BYTE *)(*(_QWORD *)(v3 + 32) + 29LL);
  v13 = *(_DWORD *)(v3 + 20);
  v14 = *(_DWORD *)(v3 + 12);
  v127 = v5;
  v15 = s;
  v16 = *(_QWORD *)(v2 + 256);
  v122 = 0;
  v121 = v13;
  v17 = *(_QWORD *)(v5 + 272);
  v18 = *(unsigned int *)(v104 + 72);
  v120 = v14;
  v129 = v16;
  v119 = v104;
  v123 = 0;
  v124 = v12;
  v125 = v117;
  v126 = v3;
  v128 = v17;
  v130 = s;
  v131 = 0x800000000LL;
  if ( (unsigned int)v18 > 8 )
  {
    sub_16CD150((__int64)&v130, s, v18, 4, v9, v10);
    v15 = v130;
  }
  LODWORD(v131) = v18;
  if ( 4 * v18 )
    memset(v15, 255, 4 * v18);
  v19 = *(unsigned int *)(v104 + 72);
  v20 = v135;
  v134 = 0x800000000LL;
  v133 = v135;
  if ( (unsigned int)v19 > 8 )
  {
    sub_16CD150((__int64)&v133, v135, v19, 40, v9, v10);
    v20 = v133;
  }
  LODWORD(v134) = v19;
  for ( i = &v20[40 * v19]; i != v20; v20 += 40 )
  {
    if ( v20 )
    {
      *(_DWORD *)v20 = 0;
      *((_DWORD *)v20 + 1) = 0;
      *((_DWORD *)v20 + 2) = 0;
      *((_QWORD *)v20 + 2) = 0;
      *((_QWORD *)v20 + 3) = 0;
      v20[32] = 0;
      v20[33] = 0;
      v20[34] = 0;
      v20[35] = 0;
    }
  }
  v22 = *(_QWORD *)(v2 + 272);
  v23 = *(_DWORD *)(v3 + 16);
  v141 = v12;
  v24 = *(_DWORD *)(v3 + 8);
  v139 = 0;
  v144 = v22;
  v25 = *(_QWORD *)(v22 + 272);
  v26 = *(unsigned int *)(v106 + 72);
  v138 = v23;
  v27 = *(_QWORD *)(v2 + 256);
  v136 = v106;
  v28 = v149;
  v145 = v25;
  v137 = v24;
  v140 = 0;
  v142 = v117;
  v143 = v3;
  v146 = v27;
  v147 = v149;
  v148 = 0x800000000LL;
  if ( (unsigned int)v26 > 8 )
  {
    sub_16CD150((__int64)&v147, v149, v26, 4, v9, v10);
    v28 = v147;
  }
  LODWORD(v148) = v26;
  if ( 4 * v26 )
    memset(v28, 255, 4 * v26);
  v29 = 0x800000000LL;
  v30 = *(unsigned int *)(v106 + 72);
  v31 = v152;
  v151 = 0x800000000LL;
  v150 = v152;
  if ( (unsigned int)v30 > 8 )
  {
    sub_16CD150((__int64)&v150, v152, v30, 40, v9, v10);
    v31 = v150;
  }
  LODWORD(v151) = v30;
  for ( j = &v31[40 * v30]; j != v31; v31 += 40 )
  {
    if ( v31 )
    {
      *(_DWORD *)v31 = 0;
      *((_DWORD *)v31 + 1) = 0;
      *((_DWORD *)v31 + 2) = 0;
      *((_QWORD *)v31 + 2) = 0;
      *((_QWORD *)v31 + 3) = 0;
      v31[32] = 0;
      v31[33] = 0;
      v31[34] = 0;
      v31[35] = 0;
    }
  }
  v33 = *(_DWORD *)(v136 + 72);
  if ( v33 )
  {
    v98 = v3;
    for ( k = 0; k != v33; ++k )
    {
      sub_1EDB260((__int64)&v136, k, (__int64)&v119, v29, v9);
      v35 = v150;
      if ( *(_DWORD *)&v150[40 * k] == 5 )
      {
        v99 = 0;
        goto LABEL_32;
      }
    }
    v3 = v98;
  }
  v41 = *(_DWORD *)(v119 + 72);
  if ( v41 )
  {
    v100 = v3;
    for ( m = 0; m != v41; ++m )
    {
      sub_1EDB260((__int64)&v119, m, (__int64)&v136, v29, v9);
      v29 = (__int64)v133;
      if ( *(_DWORD *)&v133[40 * m] == 5 )
      {
        v99 = 0;
        v35 = v150;
        goto LABEL_32;
      }
    }
    v3 = v100;
  }
  v99 = sub_1EDA110((__int64)&v136, (__int64)&v119);
  if ( !v99 )
    goto LABEL_73;
  v99 = sub_1EDA110((__int64)&v119, (__int64)&v136);
  if ( !v99 )
    goto LABEL_73;
  v51 = *(_QWORD *)(v106 + 104);
  if ( !*(_QWORD *)(v104 + 104) )
  {
    if ( !v51 )
      goto LABEL_110;
    LODWORD(v52) = *(_DWORD *)(v3 + 16);
    if ( !(_DWORD)v52 )
    {
      v53 = *(unsigned int *)(v3 + 20);
      goto LABEL_141;
    }
    goto LABEL_135;
  }
  v52 = *(unsigned int *)(v3 + 16);
  if ( v51 )
  {
    if ( !(_DWORD)v52 )
    {
      LODWORD(v53) = *(_DWORD *)(v3 + 20);
      v54 = v106;
      v55 = *(_QWORD *)(v104 + 104);
      goto LABEL_79;
    }
LABEL_135:
    v91 = v52;
    do
    {
      *(_DWORD *)(v51 + 112) = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v2 + 256) + 128LL))(
                                 *(_QWORD *)(v2 + 256),
                                 v91,
                                 *(unsigned int *)(v51 + 112));
      v51 = *(_QWORD *)(v51 + 104);
    }
    while ( v51 );
    v72 = *(_QWORD *)(v104 + 104);
    goto LABEL_103;
  }
  if ( (_DWORD)v52 )
    v69 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 256) + 248LL) + 4 * v52);
  else
    v69 = *(_DWORD *)(*(_QWORD *)(v3 + 32) + 24LL);
  v94 = v69;
  v96 = (__int64 *)(*(_QWORD *)(v2 + 272) + 296LL);
  v70 = sub_145CBF0(v96, 120, 16);
  *(_QWORD *)v70 = v70 + 16;
  *(_QWORD *)(v70 + 8) = 0x200000000LL;
  *(_QWORD *)(v70 + 64) = v70 + 80;
  *(_QWORD *)(v70 + 72) = 0x200000000LL;
  *(_QWORD *)(v70 + 96) = 0;
  sub_1EDCA90(v70, (__int64 *)v106, v96, v71, (int)v96);
  *(_DWORD *)(v70 + 112) = v94;
  *(_QWORD *)(v70 + 104) = *(_QWORD *)(v106 + 104);
  *(_QWORD *)(v106 + 104) = v70;
  v72 = *(_QWORD *)(v104 + 104);
LABEL_103:
  v53 = *(unsigned int *)(v3 + 20);
  if ( !v72 )
  {
LABEL_141:
    if ( (_DWORD)v53 )
      v92 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 256) + 248LL) + 4 * v53);
    else
      v92 = *(_DWORD *)(*(_QWORD *)(v3 + 32) + 24LL);
    sub_1ED83F0(v2, v106, v104, v92, v3);
    goto LABEL_82;
  }
  v54 = v106;
  v55 = v72;
  do
  {
LABEL_79:
    v56 = *(unsigned int *)(v55 + 112);
    if ( (_DWORD)v53 )
      LODWORD(v56) = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(v2 + 256) + 128LL))(
                       *(_QWORD *)(v2 + 256),
                       (unsigned int)v53,
                       *(unsigned int *)(v55 + 112),
                       v56);
    sub_1ED83F0(v2, v54, v55, v56, v3);
    v55 = *(_QWORD *)(v55 + 104);
  }
  while ( v55 );
LABEL_82:
  v57 = *(unsigned int *)(v136 + 72);
  v95 = 8 * v57;
  if ( (_DWORD)v57 )
  {
    v93 = v2;
    v58 = 0;
    while ( 1 )
    {
      v59 = &v150[5 * v58];
      if ( *(_DWORD *)v59 )
        goto LABEL_107;
      v60 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v136 + 64) + v58) + 8LL);
      v61 = v60 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v60 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v60 & 6) == 0 )
        goto LABEL_107;
      v62 = *(_QWORD *)(v106 + 104);
      if ( v62 )
        break;
LABEL_106:
      v59[33] = 1;
      *(_BYTE *)(v93 + 396) = 1;
LABEL_107:
      v58 += 8;
      if ( v58 == v95 )
      {
        v2 = v93;
        goto LABEL_109;
      }
    }
    while ( 1 )
    {
      v64 = (__int64 *)sub_1DB3C70((__int64 *)v62, v61);
      v65 = *(_QWORD *)v62 + 24LL * *(unsigned int *)(v62 + 8);
      if ( v64 != (__int64 *)v65 )
      {
        v66 = *(_DWORD *)(v61 + 24);
        v67 = *(_DWORD *)((*v64 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        if ( (unsigned __int64)(v67 | (*v64 >> 1) & 3) > v66 || v61 != (v64[1] & 0xFFFFFFFFFFFFFFF8LL) )
          goto LABEL_89;
        if ( (__int64 *)v65 != v64 + 3 )
          break;
      }
LABEL_92:
      v62 = *(_QWORD *)(v62 + 104);
      if ( !v62 )
      {
        v59 = &v150[5 * v58];
        goto LABEL_106;
      }
    }
    v68 = v64[3];
    v64 += 3;
    v67 = *(_DWORD *)((v68 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_89:
    if ( v66 >= v67 )
    {
      v63 = v64[2];
      if ( v63 )
      {
        if ( v60 == *(_QWORD *)(v63 + 8) )
          goto LABEL_107;
      }
    }
    goto LABEL_92;
  }
LABEL_109:
  sub_1ED9CC0(&v136, v106, (_DWORD *)(v2 + 392));
  sub_1ED9CC0(&v119, v106, (_DWORD *)(v2 + 392));
LABEL_110:
  v112 = (__int64 *)v114;
  v113 = 0x800000000LL;
  sub_1ED7EC0(&v136, &v119, (__int64)&v112, 1);
  sub_1ED7EC0(&v119, &v136, (__int64)&v112, 1);
  v109 = v111;
  v110 = 0x800000000LL;
  sub_1ED9360(&v136, v2 + 560, (__int64)&v109, v106, v73, v74);
  sub_1ED9360(&v119, v2 + 560, (__int64)&v109, 0, v75, v76);
  while ( (_DWORD)v110 )
  {
    v78 = *(_QWORD *)(v2 + 272);
    v79 = *(_DWORD *)&v109[4 * (unsigned int)v110 - 4];
    LODWORD(v110) = v110 - 1;
    v80 = *(unsigned int *)(v78 + 408);
    v81 = v79 & 0x7FFFFFFF;
    v82 = v79 & 0x7FFFFFFF;
    v83 = 8 * v82;
    if ( (v79 & 0x7FFFFFFFu) < (unsigned int)v80 )
    {
      v84 = *(_QWORD *)(v78 + 400);
      v85 = *(_QWORD *)(v84 + 8LL * v81);
      if ( v85 )
        goto LABEL_114;
    }
    v86 = v81 + 1;
    if ( (unsigned int)v80 < v81 + 1 )
    {
      if ( v86 >= v80 )
      {
        if ( v86 > v80 )
        {
          if ( v86 > (unsigned __int64)*(unsigned int *)(v78 + 412) )
          {
            sub_16CD150(v78 + 400, (const void *)(v78 + 416), v86, 8, v82, 8 * v79);
            v80 = *(unsigned int *)(v78 + 408);
            v82 = v79 & 0x7FFFFFFF;
            v83 = 8 * v82;
          }
          v87 = *(_QWORD *)(v78 + 400);
          v88 = (_QWORD *)(v87 + 8 * v80);
          v89 = (_QWORD *)(v87 + 8LL * v86);
          v90 = *(_QWORD *)(v78 + 416);
          if ( v89 != v88 )
          {
            do
              *v88++ = v90;
            while ( v89 != v88 );
            v87 = *(_QWORD *)(v78 + 400);
          }
          *(_DWORD *)(v78 + 408) = v86;
          goto LABEL_119;
        }
      }
      else
      {
        *(_DWORD *)(v78 + 408) = v86;
      }
    }
    v87 = *(_QWORD *)(v78 + 400);
LABEL_119:
    v97 = v82;
    *(_QWORD *)(v87 + v83) = sub_1DBA290(v79);
    v85 = *(_QWORD *)(*(_QWORD *)(v78 + 400) + 8 * v97);
    sub_1DBB110((_QWORD *)v78, v85);
    v78 = *(_QWORD *)(v2 + 272);
LABEL_114:
    if ( (unsigned __int8)sub_1DC0580((_QWORD *)v78, v85, 0, v84, v82, v83) )
    {
      v115[1] = 0x800000000LL;
      v115[0] = (unsigned __int64)v116;
      sub_1DBEB50(v78, v85, (__int64)v115);
      if ( (_BYTE *)v115[0] != v116 )
        _libc_free(v115[0]);
    }
  }
  sub_1DB9000(v106, (__int64 *)v104, (__int64)v147, (__int64)v130, (__int64)v117, v77);
  sub_1E69E80(*(_QWORD *)(v2 + 248), *(_DWORD *)(v106 + 112));
  sub_1E69E80(*(_QWORD *)(v2 + 248), *(_DWORD *)(v104 + 112));
  if ( (_DWORD)v113 )
    sub_1DBC0D0(*(_QWORD **)(v2 + 272), v106, v112, (unsigned int)v113, 0, 0);
  if ( v109 != v111 )
    _libc_free((unsigned __int64)v109);
  if ( v112 != (__int64 *)v114 )
    _libc_free((unsigned __int64)v112);
LABEL_73:
  v35 = v150;
LABEL_32:
  if ( v35 != v152 )
    _libc_free((unsigned __int64)v35);
  if ( v147 != v149 )
    _libc_free((unsigned __int64)v147);
  if ( v133 != v135 )
    _libc_free((unsigned __int64)v133);
  if ( v130 != s )
    _libc_free((unsigned __int64)v130);
  if ( (_BYTE *)v117[0] != v118 )
    _libc_free(v117[0]);
  return v99;
}
