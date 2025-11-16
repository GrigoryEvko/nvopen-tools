// Function: sub_13751A0
// Address: 0x13751a0
//
__int64 __fastcall sub_13751A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  _BYTE *v5; // rax
  char *v6; // r13
  char *v7; // rbx
  int v8; // esi
  _QWORD *v9; // rdi
  unsigned int v10; // edx
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // r12
  unsigned int v14; // esi
  unsigned int v15; // edx
  unsigned int v16; // edi
  unsigned int v17; // r8d
  char v18; // si
  char v19; // di
  unsigned int v20; // ecx
  _BYTE *v21; // r15
  _BYTE *v22; // rbx
  _QWORD *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  _DWORD *v26; // r12
  _QWORD *v27; // rax
  __int64 v28; // rcx
  _QWORD *v29; // r13
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // r10
  int v33; // esi
  int v34; // ecx
  _QWORD *v35; // r11
  _QWORD **v36; // r14
  int v37; // ecx
  unsigned int v38; // edx
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // r9
  char v44; // di
  _BYTE *v45; // r14
  _BYTE *v46; // r15
  _QWORD *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  _BYTE *v50; // r13
  unsigned int *v51; // r13
  __int64 v52; // rbx
  unsigned int *v53; // r12
  unsigned __int64 v54; // rax
  unsigned int *v55; // rbx
  unsigned int v56; // ecx
  unsigned int *v57; // rax
  unsigned int *v58; // rdx
  unsigned int *v59; // r13
  __int64 v60; // rbx
  unsigned int *v61; // r12
  unsigned __int64 v62; // rax
  unsigned int *v63; // rbx
  unsigned int v64; // ecx
  unsigned int *v65; // rax
  unsigned int *v66; // rdx
  __int64 v67; // rax
  unsigned int *v68; // rbx
  unsigned int *v69; // r14
  __int64 v70; // rcx
  __int64 v71; // rax
  _DWORD *v72; // rdi
  __int64 v73; // r13
  __int64 v74; // r12
  bool v75; // al
  unsigned int *v76; // rbx
  _QWORD *v77; // rax
  __int64 v78; // rsi
  _QWORD *v79; // r10
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  int v83; // ecx
  _QWORD **v84; // r12
  _QWORD *v85; // rdi
  unsigned int v86; // edi
  _DWORD *v87; // rdx
  unsigned int v88; // r11d
  _QWORD *v89; // rcx
  _DWORD *v90; // rsi
  __int64 v91; // rax
  int v92; // ecx
  int v93; // r10d
  _QWORD *v94; // r9
  __int64 v95; // rsi
  __int64 v96; // r8
  int v97; // esi
  _QWORD *v98; // r8
  unsigned int v99; // ecx
  __int64 v100; // rdi
  int v101; // r9d
  _QWORD *v102; // rdx
  int v103; // esi
  _QWORD *v104; // r8
  unsigned int v105; // ecx
  __int64 v106; // rdi
  int v107; // r9d
  __int64 v108; // rbx
  unsigned __int64 v109; // rax
  unsigned int *v110; // rbx
  unsigned int v111; // ecx
  unsigned int *v112; // rax
  unsigned int *v113; // rdx
  unsigned int *v114; // rdx
  _QWORD *v115; // rbx
  __int64 v116; // rax
  __int64 v117; // rcx
  _QWORD *v119; // rdx
  __int64 v120; // rax
  unsigned int *v121; // rdx
  unsigned int *v122; // rdx
  __int64 v123; // [rsp+8h] [rbp-1F8h]
  int v127; // [rsp+48h] [rbp-1B8h]
  int v128; // [rsp+50h] [rbp-1B0h]
  int v129; // [rsp+58h] [rbp-1A8h]
  _QWORD *v130; // [rsp+58h] [rbp-1A8h]
  _BYTE *v131; // [rsp+60h] [rbp-1A0h]
  __int64 v132; // [rsp+68h] [rbp-198h]
  __int64 v133; // [rsp+70h] [rbp-190h] BYREF
  char *v134; // [rsp+78h] [rbp-188h] BYREF
  void *v135; // [rsp+80h] [rbp-180h] BYREF
  char *v136; // [rsp+88h] [rbp-178h] BYREF
  void *src; // [rsp+90h] [rbp-170h] BYREF
  __int64 v138; // [rsp+98h] [rbp-168h]
  _BYTE v139[16]; // [rsp+A0h] [rbp-160h] BYREF
  void *v140; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v141; // [rsp+B8h] [rbp-148h]
  _BYTE v142[16]; // [rsp+C0h] [rbp-140h] BYREF
  _QWORD v143[2]; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v144; // [rsp+E0h] [rbp-120h]
  __int64 v145; // [rsp+E8h] [rbp-118h]
  __int64 v146; // [rsp+F0h] [rbp-110h]
  __int64 v147; // [rsp+F8h] [rbp-108h]
  __int64 v148; // [rsp+100h] [rbp-100h]
  __int64 v149; // [rsp+108h] [rbp-F8h]
  char *v150; // [rsp+110h] [rbp-F0h]
  char *v151; // [rsp+118h] [rbp-E8h]
  __int64 v152; // [rsp+120h] [rbp-E0h]
  __int64 v153; // [rsp+128h] [rbp-D8h]
  __int64 v154; // [rsp+130h] [rbp-D0h]
  __int64 v155; // [rsp+138h] [rbp-C8h]
  char *v156; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v157; // [rsp+148h] [rbp-B8h]
  _QWORD *v158; // [rsp+150h] [rbp-B0h] BYREF
  unsigned int v159; // [rsp+158h] [rbp-A8h]
  _BYTE v160[48]; // [rsp+1D0h] [rbp-30h] BYREF

  v123 = a1 + 88;
  if ( a3 )
    v123 = *(_QWORD *)(a4 + 8);
  v4 = *(_QWORD *)(a2 + 16);
  v143[0] = 0;
  v143[1] = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  sub_13747A0((int *)v143, v4);
  while ( 1 )
  {
    sub_1374C50((__int64)v143);
    if ( v151 == v150 )
      break;
    if ( (unsigned __int64)(v151 - v150) <= 8 )
      continue;
    v156 = 0;
    v133 = a3;
    src = v139;
    v138 = 0x400000000LL;
    v141 = 0x400000000LL;
    v5 = &v158;
    v140 = v142;
    v157 = 1;
    do
    {
      *(_QWORD *)v5 = -8;
      v5 += 16;
    }
    while ( v5 != v160 );
    v6 = v151;
    v7 = v150;
    while ( v6 != v7 )
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)v7;
        if ( (v157 & 1) == 0 )
          break;
        v8 = 7;
        v9 = &v158;
LABEL_11:
        v10 = v8 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v11 = &v9[2 * v10];
        v12 = *v11;
        if ( v13 == *v11 )
          goto LABEL_12;
        v93 = 1;
        v94 = 0;
        while ( 1 )
        {
          if ( v12 == -8 )
          {
            v15 = v157;
            v17 = 24;
            v14 = 8;
            if ( v94 )
              v11 = v94;
            ++v156;
            v16 = ((unsigned int)v157 >> 1) + 1;
            if ( (v157 & 1) == 0 )
            {
              v14 = v159;
              goto LABEL_22;
            }
            goto LABEL_23;
          }
          if ( v94 || v12 != -16 )
            v11 = v94;
          v10 = v8 & (v93 + v10);
          v12 = v9[2 * v10];
          if ( v13 == v12 )
            break;
          ++v93;
          v94 = v11;
          v11 = &v9[2 * v10];
        }
        v11 = &v9[2 * v10];
LABEL_12:
        v7 += 8;
        *((_BYTE *)v11 + 8) = 0;
        if ( v6 == v7 )
          goto LABEL_28;
      }
      v14 = v159;
      v9 = v158;
      if ( v159 )
      {
        v8 = v159 - 1;
        goto LABEL_11;
      }
      v15 = v157;
      ++v156;
      v11 = 0;
      v16 = ((unsigned int)v157 >> 1) + 1;
LABEL_22:
      v17 = 3 * v14;
LABEL_23:
      if ( v17 <= 4 * v16 )
      {
        sub_1373FA0((__int64)&v156, 2 * v14);
        if ( (v157 & 1) != 0 )
        {
          v97 = 7;
          v98 = &v158;
        }
        else
        {
          v98 = v158;
          if ( !v159 )
            goto LABEL_213;
          v97 = v159 - 1;
        }
        v15 = v157;
        v99 = v97 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v11 = &v98[2 * v99];
        v100 = *v11;
        if ( v13 == *v11 )
          goto LABEL_25;
        v101 = 1;
        v102 = 0;
        while ( v100 != -8 )
        {
          if ( !v102 && v100 == -16 )
            v102 = v11;
          v99 = v97 & (v101 + v99);
          v11 = &v98[2 * v99];
          v100 = *v11;
          if ( v13 == *v11 )
            goto LABEL_157;
          ++v101;
        }
      }
      else
      {
        if ( v14 - HIDWORD(v157) - v16 > v14 >> 3 )
          goto LABEL_25;
        sub_1373FA0((__int64)&v156, v14);
        if ( (v157 & 1) != 0 )
        {
          v103 = 7;
          v104 = &v158;
        }
        else
        {
          v104 = v158;
          if ( !v159 )
          {
LABEL_213:
            LODWORD(v157) = (2 * ((unsigned int)v157 >> 1) + 2) | v157 & 1;
            BUG();
          }
          v103 = v159 - 1;
        }
        v15 = v157;
        v105 = v103 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v11 = &v104[2 * v105];
        v106 = *v11;
        if ( v13 == *v11 )
          goto LABEL_25;
        v107 = 1;
        v102 = 0;
        while ( v106 != -8 )
        {
          if ( v106 == -16 && !v102 )
            v102 = v11;
          v105 = v103 & (v107 + v105);
          v11 = &v104[2 * v105];
          v106 = *v11;
          if ( v13 == *v11 )
            goto LABEL_157;
          ++v107;
        }
      }
      if ( v102 )
        v11 = v102;
LABEL_157:
      v15 = v157;
LABEL_25:
      LODWORD(v157) = (2 * (v15 >> 1) + 2) | v15 & 1;
      if ( *v11 != -8 )
        --HIDWORD(v157);
      v7 += 8;
      *((_BYTE *)v11 + 8) = 0;
      *v11 = v13;
      *((_BYTE *)v11 + 8) = 0;
    }
LABEL_28:
    v18 = v157;
    v19 = v157 & 1;
    v20 = (unsigned int)v157 >> 1;
    if ( (unsigned int)v157 >> 1 )
    {
      if ( v19 )
      {
        v21 = v160;
        v22 = &v158;
      }
      else
      {
        v24 = v159;
        v23 = v158;
        v22 = v158;
        v96 = 2LL * v159;
        v21 = &v158[v96];
        if ( &v158[v96] == v158 )
          goto LABEL_36;
      }
      do
      {
        if ( *(_QWORD *)v22 != -16 && *(_QWORD *)v22 != -8 )
          break;
        v22 += 16;
      }
      while ( v22 != v21 );
    }
    else
    {
      if ( v19 )
      {
        v115 = &v158;
        v116 = 16;
      }
      else
      {
        v115 = v158;
        v116 = 2LL * v159;
      }
      v22 = &v115[v116];
      v21 = v22;
    }
    if ( !v19 )
    {
      v23 = v158;
      v24 = v159;
LABEL_36:
      v25 = 2 * v24;
      goto LABEL_37;
    }
    v23 = &v158;
    v25 = 16;
LABEL_37:
    v131 = &v23[v25];
    if ( &v23[v25] == (_QWORD *)v22 )
      goto LABEL_64;
    while ( 1 )
    {
      v26 = *(_DWORD **)v22;
      v27 = *(_QWORD **)(*(_QWORD *)v22 + 24LL);
      v28 = *(unsigned int *)(*(_QWORD *)v22 + 4LL);
      v29 = *(_QWORD **)(*(_QWORD *)v22 + 40LL);
      v30 = *(_QWORD *)(*(_QWORD *)v22 + 48LL);
      v31 = v28 + (((__int64)v27 - *(_QWORD *)(*(_QWORD *)v22 + 32LL)) >> 3);
      if ( v31 < 0 )
      {
        v41 = ~((unsigned __int64)~v31 >> 6);
        goto LABEL_61;
      }
      if ( v31 > 63 )
      {
        v41 = v31 >> 6;
LABEL_61:
        v32 = *(_QWORD *)(v30 + 8 * v41) + 8 * (v31 - (v41 << 6));
        goto LABEL_41;
      }
      v32 = (__int64)&v27[v28];
LABEL_41:
      v33 = v18 & 1;
      v34 = 8;
      v35 = &v158;
      v36 = (_QWORD **)(v30 + 8);
      if ( !v33 )
      {
        v35 = v158;
        v34 = v159;
      }
      v37 = v34 - 1;
LABEL_44:
      if ( (_QWORD *)v32 != v27 )
      {
        while ( (_BYTE)v33 || v159 )
        {
          v38 = v37 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
          v39 = v35[2 * v38];
          if ( *v27 != v39 )
          {
            v129 = 1;
            while ( v39 != -8 )
            {
              v38 = v37 & (v129 + v38);
              v39 = v35[2 * v38];
              if ( *v27 == v39 )
                goto LABEL_48;
              ++v129;
            }
            break;
          }
LABEL_48:
          if ( v29 != ++v27 )
            goto LABEL_44;
          v27 = *v36++;
          v29 = v27 + 64;
          if ( (_QWORD *)v32 == v27 )
            goto LABEL_50;
        }
        v22[8] = 1;
        v40 = (unsigned int)v138;
        if ( (unsigned int)v138 >= HIDWORD(v138) )
        {
          sub_16CD150(&src, v139, 0, 4);
          v40 = (unsigned int)v138;
        }
        *((_DWORD *)src + v40) = *v26;
        LODWORD(v138) = v138 + 1;
      }
      do
LABEL_50:
        v22 += 16;
      while ( v22 != v21 && (*(_QWORD *)v22 == -16 || *(_QWORD *)v22 == -8) );
      if ( v22 == v131 )
        break;
      v18 = v157;
    }
    v20 = (unsigned int)v157 >> 1;
LABEL_64:
    v42 = (unsigned int)v138;
    v43 = (unsigned int)v138;
    if ( v20 == (_DWORD)v138 )
    {
      v59 = (unsigned int *)src;
      v108 = 4LL * (unsigned int)v138;
      v61 = (unsigned int *)((char *)src + v108);
      if ( src == (char *)src + v108 )
        goto LABEL_93;
      _BitScanReverse64(&v109, v108 >> 2);
      sub_1370980((char *)src, (unsigned int *)src + (unsigned int)v138, 2LL * (int)(63 - (v109 ^ 0x3F)));
      if ( (unsigned __int64)v108 <= 0x40 )
        goto LABEL_167;
      v110 = v59 + 16;
      sub_1370B10(v59, v59 + 16);
      if ( v61 == v59 + 16 )
        goto LABEL_93;
      v111 = *v110;
      v112 = v59 + 15;
      if ( *v110 >= v59[15] )
        goto LABEL_165;
      while ( 1 )
      {
        do
        {
          v112[1] = *v112;
          v113 = v112--;
        }
        while ( v111 < *v112 );
        ++v110;
        *v113 = v111;
        if ( v61 == v110 )
          goto LABEL_93;
        while ( 1 )
        {
          v111 = *v110;
          v112 = v110 - 1;
          if ( *v110 < *(v110 - 1) )
            break;
LABEL_165:
          v114 = v110++;
          *v114 = v111;
          if ( v61 == v110 )
            goto LABEL_93;
        }
      }
    }
    v44 = v157 & 1;
    if ( !v20 )
    {
      if ( v44 )
      {
        v119 = &v158;
        v120 = 16;
      }
      else
      {
        v119 = v158;
        v120 = 2LL * v159;
      }
      v46 = &v119[v120];
      v45 = &v119[v120];
      goto LABEL_71;
    }
    if ( v44 )
    {
      v45 = v160;
      v46 = &v158;
      do
      {
LABEL_68:
        if ( *(_QWORD *)v46 != -16 && *(_QWORD *)v46 != -8 )
          break;
        v46 += 16;
      }
      while ( v46 != v45 );
LABEL_71:
      if ( !v44 )
      {
        v47 = v158;
        v48 = v159;
        goto LABEL_73;
      }
      v47 = &v158;
      v49 = 16;
    }
    else
    {
      v48 = v159;
      v47 = v158;
      v46 = v158;
      v117 = 2LL * v159;
      v45 = &v158[v117];
      if ( &v158[v117] != v158 )
        goto LABEL_68;
LABEL_73:
      v49 = 2 * v48;
    }
    v50 = &v47[v49];
    if ( &v47[v49] == (_QWORD *)v46 )
      goto LABEL_81;
    while ( 2 )
    {
      if ( v46[8] )
        goto LABEL_76;
      v76 = *(unsigned int **)v46;
      v77 = *(_QWORD **)(*(_QWORD *)v46 + 24LL);
      v78 = *(unsigned int *)(*(_QWORD *)v46 + 4LL);
      v79 = *(_QWORD **)(*(_QWORD *)v46 + 40LL);
      v80 = *(_QWORD *)(*(_QWORD *)v46 + 48LL);
      v81 = v78 + (((__int64)v77 - *(_QWORD *)(*(_QWORD *)v46 + 32LL)) >> 3);
      if ( v81 < 0 )
      {
        v95 = ~((unsigned __int64)~v81 >> 6);
      }
      else
      {
        if ( v81 <= 63 )
        {
          v82 = (__int64)&v77[v78];
          goto LABEL_108;
        }
        v95 = v81 >> 6;
      }
      v82 = *(_QWORD *)(v80 + 8 * v95) + 8 * (v81 - (v95 << 6));
LABEL_108:
      v83 = 8;
      v84 = (_QWORD **)(v80 + 8);
      v85 = &v158;
      if ( (v157 & 1) == 0 )
      {
        v85 = v158;
        v83 = v159;
      }
      v130 = v85;
      v128 = v83 - 1;
      v86 = *v76;
      while ( 2 )
      {
        if ( (_QWORD *)v82 == v77 )
          goto LABEL_119;
LABEL_112:
        v87 = (_DWORD *)*v77;
        if ( *(_DWORD *)*v77 < v86 )
        {
LABEL_117:
          if ( v79 != ++v77 )
            continue;
          v77 = *v84++;
          v79 = v77 + 64;
          if ( (_QWORD *)v82 == v77 )
            goto LABEL_119;
          goto LABEL_112;
        }
        break;
      }
      if ( (v157 & 1) == 0 && !v159 )
        goto LABEL_125;
      v88 = v128 & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
      v89 = &v130[2 * v88];
      v90 = (_DWORD *)*v89;
      if ( v87 == (_DWORD *)*v89 )
      {
LABEL_116:
        if ( !*((_BYTE *)v89 + 8) )
          goto LABEL_125;
        goto LABEL_117;
      }
      v92 = 1;
      while ( v90 != (_DWORD *)-8LL )
      {
        v88 = v128 & (v92 + v88);
        v127 = v92 + 1;
        v89 = &v130[2 * v88];
        v90 = (_DWORD *)*v89;
        if ( v87 == (_DWORD *)*v89 )
          goto LABEL_116;
        v92 = v127;
      }
LABEL_125:
      if ( HIDWORD(v138) <= (unsigned int)v43 )
      {
        sub_16CD150(&src, v139, 0, 4);
        v43 = (unsigned int)v138;
        v86 = *v76;
      }
      *((_DWORD *)src + v43) = v86;
      v43 = (unsigned int)(v138 + 1);
      LODWORD(v138) = v138 + 1;
      v86 = *v76;
LABEL_119:
      if ( *((_DWORD *)src + (unsigned int)v43 - 1) != v86 )
      {
        v91 = (unsigned int)v141;
        if ( (unsigned int)v141 >= HIDWORD(v141) )
        {
          sub_16CD150(&v140, v142, 0, 4);
          v91 = (unsigned int)v141;
          v86 = *v76;
        }
        *((_DWORD *)v140 + v91) = v86;
        v43 = (unsigned int)v138;
        LODWORD(v141) = v141 + 1;
      }
      do
LABEL_76:
        v46 += 16;
      while ( v46 != v45 && (*(_QWORD *)v46 == -16 || *(_QWORD *)v46 == -8) );
      if ( v46 != v50 )
        continue;
      break;
    }
    v42 = (unsigned int)v43;
LABEL_81:
    v51 = (unsigned int *)src;
    v52 = 4 * v42;
    v53 = (unsigned int *)((char *)src + 4 * v42);
    if ( src != v53 )
    {
      _BitScanReverse64(&v54, v52 >> 2);
      sub_1370980((char *)src, (unsigned int *)src + v42, 2LL * (int)(63 - (v54 ^ 0x3F)));
      if ( (unsigned __int64)v52 <= 0x40 )
      {
        sub_1370B10(v51, v53);
      }
      else
      {
        v55 = v51 + 16;
        sub_1370B10(v51, v51 + 16);
        if ( v53 != v51 + 16 )
        {
          do
          {
            while ( 1 )
            {
              v56 = *v55;
              v57 = v55 - 1;
              if ( *v55 < *(v55 - 1) )
                break;
              v122 = v55++;
              *v122 = v56;
              if ( v53 == v55 )
                goto LABEL_87;
            }
            do
            {
              v57[1] = *v57;
              v58 = v57--;
            }
            while ( v56 < *v57 );
            ++v55;
            *v58 = v56;
          }
          while ( v53 != v55 );
        }
      }
    }
LABEL_87:
    v59 = (unsigned int *)v140;
    v60 = 4LL * (unsigned int)v141;
    v61 = (unsigned int *)((char *)v140 + v60);
    if ( v140 != (char *)v140 + v60 )
    {
      _BitScanReverse64(&v62, v60 >> 2);
      sub_1370980((char *)v140, (unsigned int *)((char *)v140 + v60), 2LL * (int)(63 - (v62 ^ 0x3F)));
      if ( (unsigned __int64)v60 > 0x40 )
      {
        v63 = v59 + 16;
        sub_1370B10(v59, v59 + 16);
        if ( v61 != v59 + 16 )
        {
          do
          {
            while ( 1 )
            {
              v64 = *v63;
              v65 = v63 - 1;
              if ( *(v63 - 1) > *v63 )
                break;
              v121 = v63++;
              *v121 = v64;
              if ( v61 == v63 )
                goto LABEL_93;
            }
            do
            {
              v65[1] = *v65;
              v66 = v65--;
            }
            while ( v64 < *v65 );
            ++v63;
            *v66 = v64;
          }
          while ( v61 != v63 );
        }
        goto LABEL_93;
      }
LABEL_167:
      sub_1370B10(v59, v61);
    }
LABEL_93:
    if ( (v157 & 1) == 0 )
      j___libc_free_0(v158);
    v135 = v140;
    v134 = (char *)v140 + 4 * (unsigned int)v141;
    v156 = (char *)src;
    v136 = (char *)src + 4 * (unsigned int)v138;
    v67 = sub_1371440(
            a1 + 88,
            a4,
            &v133,
            (const void **)&v156,
            (const void **)&v136,
            (const void **)&v135,
            (const void **)&v134);
    v68 = *(unsigned int **)(v67 + 112);
    v69 = &v68[*(unsigned int *)(v67 + 120)];
    if ( v68 != v69 )
    {
      v70 = v67 + 16;
      while ( 1 )
      {
        v73 = *(_QWORD *)(a1 + 64) + 24LL * *v68;
        v74 = *(_QWORD *)(v73 + 8);
        if ( !v74 )
          break;
        v71 = *(unsigned int *)(v74 + 12);
        v72 = *(_DWORD **)(v74 + 96);
        if ( (unsigned int)v71 > 1 )
        {
          v132 = v70;
          v75 = sub_1369030(v72, &v72[v71], (_DWORD *)v73);
          v70 = v132;
          if ( !v75 )
          {
            *(_QWORD *)(v73 + 8) = v132;
            goto LABEL_100;
          }
        }
        else if ( *(_DWORD *)v73 != *v72 )
        {
          break;
        }
        *(_QWORD *)v74 = v70;
LABEL_100:
        if ( v69 == ++v68 )
          goto LABEL_16;
      }
      *(_QWORD *)(v73 + 8) = v70;
      goto LABEL_100;
    }
LABEL_16:
    if ( v140 != v142 )
      _libc_free((unsigned __int64)v140);
    if ( src != v139 )
      _libc_free((unsigned __int64)src);
  }
  if ( v153 )
    j_j___libc_free_0(v153, v155 - v153);
  if ( v150 )
    j_j___libc_free_0(v150, v152 - (_QWORD)v150);
  if ( v147 )
    j_j___libc_free_0(v147, v149 - v147);
  j___libc_free_0(v144);
  if ( a3 )
    return *(_QWORD *)v123;
  else
    return *(_QWORD *)(a1 + 88);
}
