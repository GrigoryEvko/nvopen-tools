// Function: sub_2D7D440
// Address: 0x2d7d440
//
__int64 __fastcall sub_2D7D440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rdx
  _BYTE *v10; // rdi
  _BYTE *v11; // r12
  __int64 v12; // rcx
  _BYTE *v13; // r10
  unsigned int i; // esi
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rbx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // r14
  __int64 v22; // r9
  unsigned __int64 v23; // rax
  int *v24; // rdx
  unsigned int j; // eax
  int *v26; // rdi
  int v27; // r8d
  _BYTE *v28; // rdi
  _BYTE *v29; // r13
  _BYTE *v30; // rbx
  int v31; // edx
  int v32; // r11d
  unsigned int k; // eax
  unsigned int *v34; // r8
  __int64 v35; // r9
  unsigned int v36; // eax
  __int64 v37; // r14
  _BYTE *v38; // r12
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // r15
  int v42; // r11d
  unsigned int v43; // ecx
  __int64 v44; // rax
  __int64 v45; // r10
  __int64 v46; // rax
  unsigned int **v47; // r12
  __int64 v48; // rax
  unsigned __int64 v49; // r15
  __int64 m; // r13
  __int64 v51; // rax
  __int64 v52; // r14
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  _QWORD *v56; // rax
  __int64 v57; // rax
  __int64 *v58; // rbx
  __int64 v59; // rax
  __int64 *v60; // r10
  __int64 v61; // r12
  __int64 v62; // rcx
  __int64 v63; // rdx
  _QWORD *v64; // rax
  __int64 v65; // rcx
  _QWORD *v66; // rdx
  __int64 v67; // rax
  __int64 v68; // r9
  __int64 v69; // r13
  unsigned int v70; // edx
  __int64 v71; // rax
  unsigned __int64 v72; // rdi
  _QWORD *v73; // rcx
  _BYTE *v74; // rdx
  unsigned __int64 v75; // rdx
  __int64 v76; // r8
  unsigned __int64 v77; // rcx
  unsigned int v78; // edx
  __int64 v79; // rbx
  __int64 v80; // r10
  __int64 v81; // rax
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rdx
  __int64 *v89; // r10
  __int64 *v90; // r10
  __int64 v91; // rax
  __int64 v92; // rdx
  unsigned __int64 v93; // rax
  __int64 v94; // r13
  __int64 *v95; // r10
  unsigned __int64 v96; // rax
  unsigned int **v97; // r12
  unsigned int **v98; // rbx
  unsigned __int64 v99; // rdi
  unsigned int v101; // eax
  __int64 v102; // r8
  int v103; // ecx
  int v104; // eax
  __int64 v105; // rax
  int v106; // edx
  _BYTE *v107; // rax
  __int64 v108; // rbx
  __int64 *v109; // rax
  int v110; // edi
  __int64 *v111; // rcx
  unsigned __int64 v112; // rdx
  __int64 v113; // rax
  unsigned __int64 v114; // rsi
  unsigned int **v115; // rsi
  __int64 *v116; // rdx
  __int64 v117; // rax
  __int64 v118; // [rsp+8h] [rbp-1C8h]
  __int64 *v119; // [rsp+10h] [rbp-1C0h]
  __int64 *v120; // [rsp+10h] [rbp-1C0h]
  __int64 *v121; // [rsp+10h] [rbp-1C0h]
  __int64 v122; // [rsp+20h] [rbp-1B0h]
  unsigned int **v123; // [rsp+38h] [rbp-198h]
  unsigned __int8 v124; // [rsp+40h] [rbp-190h]
  __int64 v125; // [rsp+48h] [rbp-188h]
  __int64 *v126; // [rsp+48h] [rbp-188h]
  __int64 *v127; // [rsp+48h] [rbp-188h]
  __int64 *v128; // [rsp+48h] [rbp-188h]
  unsigned int **v129; // [rsp+48h] [rbp-188h]
  __int64 *v130; // [rsp+50h] [rbp-180h]
  int v131; // [rsp+58h] [rbp-178h]
  unsigned int **v132; // [rsp+58h] [rbp-178h]
  _BYTE *v133; // [rsp+58h] [rbp-178h]
  _BYTE *v134; // [rsp+58h] [rbp-178h]
  _BYTE *v135; // [rsp+58h] [rbp-178h]
  __int64 v136; // [rsp+60h] [rbp-170h] BYREF
  int v137; // [rsp+68h] [rbp-168h]
  _BYTE *v138; // [rsp+70h] [rbp-160h] BYREF
  __int64 v139; // [rsp+78h] [rbp-158h]
  _BYTE v140[16]; // [rsp+80h] [rbp-150h] BYREF
  int *v141; // [rsp+90h] [rbp-140h] BYREF
  __int64 v142; // [rsp+98h] [rbp-138h]
  _BYTE v143[16]; // [rsp+A0h] [rbp-130h] BYREF
  unsigned __int64 v144; // [rsp+B0h] [rbp-120h] BYREF
  char *v145; // [rsp+B8h] [rbp-118h]
  __int64 v146; // [rsp+C0h] [rbp-110h]
  char v147; // [rsp+C8h] [rbp-108h] BYREF
  __int16 v148; // [rsp+D0h] [rbp-100h]
  __int64 v149; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v150; // [rsp+E8h] [rbp-E8h]
  __int64 v151; // [rsp+F0h] [rbp-E0h]
  unsigned int v152; // [rsp+F8h] [rbp-D8h]
  unsigned int **v153; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v154; // [rsp+108h] [rbp-C8h]
  unsigned int *v155; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v156; // [rsp+118h] [rbp-B8h]
  __int64 v157; // [rsp+120h] [rbp-B0h]
  unsigned int v158; // [rsp+128h] [rbp-A8h]
  _BYTE *v159; // [rsp+130h] [rbp-A0h] BYREF
  __int64 v160; // [rsp+138h] [rbp-98h]
  _BYTE v161[144]; // [rsp+140h] [rbp-90h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v138 = v140;
  v139 = 0x200000000LL;
  if ( !v6 )
    return 0;
  v7 = 0;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v6 + 24);
      if ( *(_BYTE *)v8 == 85 )
      {
        v9 = *(_QWORD *)(v8 - 32);
        if ( v9 )
        {
          if ( !*(_BYTE *)v9
            && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v8 + 80)
            && (*(_BYTE *)(v9 + 33) & 0x20) != 0
            && *(_DWORD *)(v9 + 36) == 149 )
          {
            break;
          }
        }
      }
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_13;
    }
    if ( v7 + 1 > (unsigned __int64)HIDWORD(v139) )
    {
      sub_C8D5F0((__int64)&v138, v140, v7 + 1, 8u, a5, a6);
      v7 = (unsigned int)v139;
    }
    *(_QWORD *)&v138[8 * v7] = v8;
    v7 = (unsigned int)(v139 + 1);
    LODWORD(v139) = v139 + 1;
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v6 );
LABEL_13:
  v10 = v138;
  if ( (unsigned int)v7 <= 1 )
  {
    v124 = 0;
    goto LABEL_117;
  }
  v11 = v138;
  v12 = 0;
  v149 = 0;
  v13 = &v138[8 * v7];
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v160 = 0;
  v153 = &v155;
  v159 = v161;
  for ( i = 0; ; i = v158 )
  {
    v15 = *(_QWORD *)v11;
    v16 = *(_DWORD *)(*(_QWORD *)v11 + 4LL) & 0x7FFFFFF;
    v17 = *(_QWORD *)(*(_QWORD *)v11 + 32 * (2 - v16));
    v18 = *(_QWORD **)(v17 + 24);
    if ( *(_DWORD *)(v17 + 32) > 0x40u )
      v18 = (_QWORD *)*v18;
    v19 = *(_QWORD *)(v15 + 32 * (1 - v16));
    v20 = *(_QWORD **)(v19 + 24);
    if ( *(_DWORD *)(v19 + 32) > 0x40u )
      v20 = (_QWORD *)*v20;
    v144 = __PAIR64__((unsigned int)v18, (unsigned int)v20);
    v21 = (unsigned int)v20;
    LODWORD(v145) = 0;
    if ( i )
    {
      v22 = i - 1;
      v131 = 1;
      v23 = 0xBF58476D1CE4E5B9LL
          * ((unsigned int)(37 * (_DWORD)v18) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v20) << 32));
      v24 = 0;
      for ( j = v22 & ((v23 >> 31) ^ v23); ; j = v22 & v101 )
      {
        v26 = (int *)(v12 + 12LL * j);
        v27 = *v26;
        if ( (_DWORD)v21 == *v26 && (_DWORD)v18 == v26[1] )
          break;
        if ( v27 == -1 )
        {
          if ( v26[1] == -1 )
          {
            if ( !v24 )
              v24 = (int *)(v12 + 12LL * j);
            v155 = (unsigned int *)((char *)v155 + 1);
            v104 = v157 + 1;
            v141 = v24;
            if ( 4 * ((int)v157 + 1) < 3 * i )
            {
              v102 = i - (v104 + HIDWORD(v157));
              v103 = v21;
              if ( (unsigned int)v102 <= i >> 3 )
              {
                v134 = v13;
                sub_2D7D1A0((__int64)&v155, i);
                sub_2D6B8A0((__int64)&v155, (int *)&v144, &v141);
                v103 = v144;
                v24 = v141;
                v13 = v134;
                v104 = v157 + 1;
              }
              goto LABEL_128;
            }
            goto LABEL_127;
          }
        }
        else if ( v27 == -2 && v26[1] == -2 && !v24 )
        {
          v24 = (int *)(v12 + 12LL * j);
        }
        v101 = v131 + j;
        ++v131;
      }
    }
    else
    {
      v155 = (unsigned int *)((char *)v155 + 1);
      v141 = 0;
LABEL_127:
      v133 = v13;
      sub_2D7D1A0((__int64)&v155, 2 * i);
      sub_2D6B8A0((__int64)&v155, (int *)&v144, &v141);
      v103 = v144;
      v24 = v141;
      v13 = v133;
      v104 = v157 + 1;
LABEL_128:
      LODWORD(v157) = v104;
      if ( *v24 != -1 || v24[1] != -1 )
        --HIDWORD(v157);
      *v24 = v103;
      v24[1] = HIDWORD(v144);
      v24[2] = (int)v145;
      v24[2] = v160;
      v105 = (unsigned int)v160;
      v106 = v160;
      if ( (unsigned int)v160 >= (unsigned __int64)HIDWORD(v160) )
      {
        v108 = v21 | ((_QWORD)v18 << 32);
        if ( HIDWORD(v160) < (unsigned __int64)(unsigned int)v160 + 1 )
        {
          v135 = v13;
          sub_C8D5F0((__int64)&v159, v161, (unsigned int)v160 + 1LL, 0x10u, v102, v22);
          v105 = (unsigned int)v160;
          v13 = v135;
        }
        v109 = (__int64 *)&v159[16 * v105];
        *v109 = v108;
        v109[1] = v15;
        LODWORD(v160) = v160 + 1;
      }
      else
      {
        v107 = &v159[16 * (unsigned int)v160];
        if ( v107 )
        {
          *(_DWORD *)v107 = v21;
          *((_DWORD *)v107 + 1) = (_DWORD)v18;
          *((_QWORD *)v107 + 1) = v15;
          v106 = v160;
        }
        LODWORD(v160) = v106 + 1;
      }
    }
    v11 += 8;
    if ( v13 == v11 )
      break;
    v12 = v156;
  }
  v28 = v159;
  v29 = &v159[16 * (unsigned int)v160];
  if ( v159 != v29 )
  {
    v30 = v159;
    while ( 1 )
    {
      v31 = *(_DWORD *)v30;
      if ( *(_DWORD *)v30 == *((_DWORD *)v30 + 1) || !v158 )
        goto LABEL_49;
      v32 = 1;
      for ( k = (v158 - 1)
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v31) | ((unsigned __int64)(unsigned int)(37 * v31) << 32))) >> 31)
               ^ (756364221 * v31)); ; k = (v158 - 1) & v36 )
      {
        v34 = (unsigned int *)(v156 + 12LL * k);
        v35 = *v34;
        if ( v31 == (_DWORD)v35 && v31 == v34[1] )
          break;
        if ( (_DWORD)v35 == -1 && v34[1] == -1 )
          goto LABEL_49;
        v36 = v32 + k;
        ++v32;
      }
      v37 = *((_QWORD *)v30 + 1);
      if ( v34 == (unsigned int *)(v156 + 12LL * v158) )
        goto LABEL_49;
      v38 = &v28[16 * v34[2]];
      if ( v38 == &v28[16 * (unsigned int)v160] )
        goto LABEL_49;
      v39 = *((_QWORD *)v38 + 1);
      v137 = 0;
      v136 = v39;
      if ( !v152 )
        break;
      v40 = v152 - 1;
      v41 = 0;
      v42 = 1;
      v43 = v40 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v44 = v150 + 16LL * v43;
      v45 = *(_QWORD *)v44;
      if ( v39 != *(_QWORD *)v44 )
      {
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v41 )
            v41 = v44;
          v35 = (unsigned int)(v42 + 1);
          v43 = v40 & (v42 + v43);
          v44 = v150 + 16LL * v43;
          v45 = *(_QWORD *)v44;
          if ( v39 == *(_QWORD *)v44 )
            goto LABEL_45;
          ++v42;
        }
        if ( !v41 )
          v41 = v44;
        ++v149;
        v110 = v151 + 1;
        v144 = v41;
        if ( 4 * ((int)v151 + 1) < 3 * v152 )
        {
          v40 = v152 >> 3;
          v111 = (__int64 *)&v144;
          if ( v152 - HIDWORD(v151) - v110 > (unsigned int)v40 )
          {
LABEL_149:
            LODWORD(v151) = v110;
            if ( *(_QWORD *)v41 != -4096 )
              --HIDWORD(v151);
            *(_QWORD *)v41 = v39;
            *(_DWORD *)(v41 + 8) = v137;
            v112 = (unsigned int)v154;
            v141 = (int *)v143;
            v142 = 0;
            v113 = *((_QWORD *)v38 + 1);
            v114 = (unsigned int)v154 + 1LL;
            v145 = &v147;
            v144 = v113;
            v46 = (unsigned int)v154;
            v146 = 0;
            if ( v114 > HIDWORD(v154) )
            {
              if ( v153 > (unsigned int **)&v144
                || (v129 = v153, v112 = (unsigned __int64)&v153[3 * (unsigned int)v154], (unsigned __int64)&v144 >= v112) )
              {
                sub_2D6B970((__int64)&v153, v114, v112, (__int64)&v144, v40, v35);
                v112 = (unsigned int)v154;
                v115 = v153;
                v111 = (__int64 *)&v144;
                v46 = (unsigned int)v154;
              }
              else
              {
                sub_2D6B970((__int64)&v153, v114, v112, (__int64)&v144, v40, v35);
                v115 = v153;
                v112 = (unsigned int)v154;
                v111 = (__int64 *)((char *)v153 + (char *)&v144 - (char *)v129);
                v46 = (unsigned int)v154;
              }
            }
            else
            {
              v115 = v153;
            }
            v116 = (__int64 *)&v115[3 * v112];
            if ( v116 )
            {
              v117 = *v111;
              v116[2] = 0;
              *v116 = v117;
              v116[1] = (__int64)(v116 + 3);
              if ( *((_DWORD *)v111 + 4) )
                sub_2D56E90((__int64)(v116 + 1), (char **)v111 + 1, (__int64)v116, (__int64)v111, v40, v35);
              v46 = (unsigned int)v154;
            }
            LODWORD(v154) = v46 + 1;
            if ( v145 != &v147 )
            {
              _libc_free((unsigned __int64)v145);
              v46 = (unsigned int)(v154 - 1);
            }
            *(_DWORD *)(v41 + 8) = v46;
            goto LABEL_46;
          }
          sub_2D747F0((__int64)&v149, v152);
LABEL_164:
          sub_2D6B060((__int64)&v149, &v136, &v144);
          v39 = v136;
          v41 = v144;
          v111 = (__int64 *)&v144;
          v110 = v151 + 1;
          goto LABEL_149;
        }
LABEL_163:
        sub_2D747F0((__int64)&v149, 2 * v152);
        goto LABEL_164;
      }
LABEL_45:
      v46 = *(unsigned int *)(v44 + 8);
LABEL_46:
      v47 = &v153[3 * v46];
      v48 = *((unsigned int *)v47 + 4);
      if ( v48 + 1 > (unsigned __int64)*((unsigned int *)v47 + 5) )
      {
        sub_C8D5F0((__int64)(v47 + 1), v47 + 3, v48 + 1, 8u, v40, v35);
        v48 = *((unsigned int *)v47 + 4);
      }
      *(_QWORD *)&v47[1][2 * v48] = v37;
      ++*((_DWORD *)v47 + 4);
      v28 = v159;
LABEL_49:
      v30 += 16;
      if ( v29 == v30 )
        goto LABEL_50;
    }
    ++v149;
    v144 = 0;
    goto LABEL_163;
  }
LABEL_50:
  if ( v28 != v161 )
    _libc_free((unsigned __int64)v28);
  sub_C7D6A0(v156, 12LL * v158, 4);
  if ( !(_DWORD)v154 )
  {
    v124 = 0;
    goto LABEL_113;
  }
  v132 = v153;
  v123 = &v153[3 * (unsigned int)v154];
  do
  {
    v49 = (unsigned __int64)*v132;
    for ( m = sub_AA5190(*((_QWORD *)*v132 + 5)); ; m = *(_QWORD *)(m + 8) )
    {
      if ( !m )
        BUG();
      if ( v49 == m - 24 )
        break;
      if ( *(_BYTE *)(m - 24) == 85 )
      {
        v51 = *(_QWORD *)(m - 56);
        if ( v51 )
        {
          if ( !*(_BYTE *)v51
            && *(_QWORD *)(v51 + 24) == *(_QWORD *)(m + 56)
            && (*(_BYTE *)(v51 + 33) & 0x20) != 0
            && *(_DWORD *)(v51 + 36) == 149 )
          {
            v52 = sub_B5B6B0(m - 24);
            if ( v52 == sub_B5B6B0(v49) )
            {
              v53 = *(_QWORD *)(m + 32 * (1LL - (*(_DWORD *)(m - 20) & 0x7FFFFFF)) - 24);
              if ( *(_DWORD *)(v53 + 32) <= 0x40u )
                v54 = *(_QWORD *)(v53 + 24);
              else
                v54 = **(_QWORD **)(v53 + 24);
              v55 = *(_QWORD *)(v49 + 32 * (1LL - (*(_DWORD *)(v49 + 4) & 0x7FFFFFF)));
              v56 = *(_QWORD **)(v55 + 24);
              if ( *(_DWORD *)(v55 + 32) > 0x40u )
                v56 = (_QWORD *)*v56;
              if ( (_DWORD)v54 == (_DWORD)v56 )
              {
                v57 = v122;
                LOWORD(v57) = 0;
                v122 = v57;
                sub_B444E0((_QWORD *)v49, m, v57);
                v124 = 1;
                goto LABEL_71;
              }
            }
          }
        }
      }
    }
    v124 = 0;
LABEL_71:
    v58 = (__int64 *)v132[1];
    v59 = *((unsigned int *)v132 + 4);
    if ( v58 != &v58[v59] )
    {
      v60 = &v58[v59];
      do
      {
        while ( 1 )
        {
          v61 = *v58;
          v62 = *(_DWORD *)(*v58 + 4) & 0x7FFFFFF;
          v63 = *(_QWORD *)(*v58 + 32 * (1 - v62));
          v64 = *(_QWORD **)(v63 + 24);
          if ( *(_DWORD *)(v63 + 32) > 0x40u )
            v64 = (_QWORD *)*v64;
          v65 = *(_QWORD *)(v61 + 32 * (2 - v62));
          v66 = *(_QWORD **)(v65 + 24);
          if ( *(_DWORD *)(v65 + 32) > 0x40u )
            v66 = (_QWORD *)*v66;
          if ( (_DWORD)v64 != (_DWORD)v66 && *(_QWORD *)(v49 + 40) == *(_QWORD *)(v61 + 40) )
          {
            v130 = v60;
            v125 = sub_B5B740(*v58);
            v67 = sub_B5B890(v61);
            v60 = v130;
            v69 = v67;
            if ( *(_BYTE *)v67 == 63 && v125 == *(_QWORD *)(v67 - 32LL * (*(_DWORD *)(v67 + 4) & 0x7FFFFFF)) )
              break;
          }
LABEL_73:
          if ( v60 == ++v58 )
            goto LABEL_108;
        }
        v141 = (int *)v143;
        v142 = 0x200000000LL;
        v70 = *(_DWORD *)(v67 + 4) & 0x7FFFFFF;
        if ( v70 > 1 )
        {
          v71 = v70;
          v72 = v69 + 32 * (v70 - 2 - (unsigned __int64)v70) + 64;
          v73 = (_QWORD *)(v69 - 32LL * v70 + 32);
          while ( 1 )
          {
            v74 = (_BYTE *)*v73;
            if ( *(_BYTE *)*v73 != 17 )
              goto LABEL_73;
            v75 = *((_DWORD *)v74 + 8) <= 0x40u ? *((_QWORD *)v74 + 3) : **((_QWORD **)v74 + 3);
            if ( v75 > 0x14 )
              goto LABEL_73;
            v73 += 4;
            if ( (_QWORD *)v72 == v73 )
            {
              v76 = 1;
              v77 = 2;
              v78 = 0;
              v119 = v58;
              v79 = 1;
              while ( 1 )
              {
                v80 = *(_QWORD *)(v69 + 32 * (v79 - v71));
                v81 = v78;
                v82 = v78 + 1LL;
                if ( v82 > v77 )
                {
                  v118 = v80;
                  sub_C8D5F0((__int64)&v141, v143, v82, 8u, v76, v68);
                  v81 = (unsigned int)v142;
                  v80 = v118;
                }
                ++v79;
                *(_QWORD *)&v141[2 * v81] = v80;
                v78 = v142 + 1;
                LODWORD(v142) = v142 + 1;
                v71 = *(_DWORD *)(v69 + 4) & 0x7FFFFFF;
                if ( (unsigned int)v71 <= (unsigned int)v79 )
                  break;
                v77 = HIDWORD(v142);
              }
              v60 = v130;
              v58 = v119;
              break;
            }
          }
        }
        v83 = *(_QWORD *)(v49 + 32);
        if ( v83 == *(_QWORD *)(v49 + 40) + 48LL || (v84 = v83 - 24, !v83) )
          v84 = 0;
        v120 = v60;
        sub_23D0AB0((__int64)&v155, v84, 0, 0, 0);
        v88 = *(_QWORD *)(v61 + 48);
        v89 = v120;
        v144 = v88;
        if ( v88 )
        {
          sub_2D572A0((__int64 *)&v144);
          v88 = v144;
          v89 = v120;
        }
        v121 = v89;
        sub_F80810((__int64)&v155, 0, v88, v85, v86, v87);
        v90 = v121;
        if ( v144 )
        {
          sub_B91220((__int64)&v144, v144);
          v90 = v121;
        }
        v91 = v125;
        v92 = v49;
        if ( *(_QWORD *)(v49 + 8) != *(_QWORD *)(v125 + 8) )
        {
          v148 = 257;
          v126 = v90;
          v93 = sub_2D5B7B0((__int64 *)&v155, 0x31u, v49, *(__int64 ***)(v91 + 8), (__int64)&v144, 0, v136, 0);
          v90 = v126;
          v92 = v93;
        }
        v148 = 257;
        v127 = v90;
        v94 = sub_921130(&v155, *(_QWORD *)(v69 + 72), v92, (_BYTE **)v141, (unsigned int)v142, (__int64)&v144, 0);
        sub_BD6B90((unsigned __int8 *)v94, (unsigned __int8 *)v61);
        v95 = v127;
        if ( *(_QWORD *)(v94 + 8) != *(_QWORD *)(v61 + 8) )
        {
          v148 = 257;
          v96 = sub_2D5B7B0((__int64 *)&v155, 0x31u, v94, *(__int64 ***)(v61 + 8), (__int64)&v144, 0, v136, 0);
          v95 = v127;
          v94 = v96;
        }
        v128 = v95;
        sub_BD84D0(v61, v94);
        sub_B43D60((_QWORD *)v61);
        sub_F94A20(&v155, v94);
        v60 = v128;
        if ( v141 != (int *)v143 )
        {
          _libc_free((unsigned __int64)v141);
          v60 = v128;
        }
        ++v58;
        v124 = 1;
      }
      while ( v60 != v58 );
    }
LABEL_108:
    v132 += 3;
  }
  while ( v123 != v132 );
  v97 = v153;
  v98 = &v153[3 * (unsigned int)v154];
  if ( v153 != v98 )
  {
    do
    {
      v98 -= 3;
      v99 = (unsigned __int64)v98[1];
      if ( (unsigned int **)v99 != v98 + 3 )
        _libc_free(v99);
    }
    while ( v97 != v98 );
LABEL_113:
    v97 = v153;
  }
  if ( v97 != &v155 )
    _libc_free((unsigned __int64)v97);
  sub_C7D6A0(v150, 16LL * v152, 8);
  v10 = v138;
LABEL_117:
  if ( v10 != v140 )
    _libc_free((unsigned __int64)v10);
  return v124;
}
