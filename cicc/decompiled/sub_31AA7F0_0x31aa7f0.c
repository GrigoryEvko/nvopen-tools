// Function: sub_31AA7F0
// Address: 0x31aa7f0
//
__int64 __fastcall sub_31AA7F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  char v8; // si
  char v9; // cl
  __int64 v10; // rdi
  int v11; // r10d
  unsigned int v12; // r8d
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // rdi
  int v22; // esi
  unsigned int v23; // edx
  __int64 v24; // r10
  int v25; // r11d
  __int64 *v26; // r8
  unsigned int v27; // eax
  int v28; // edx
  unsigned int v29; // ecx
  unsigned int v30; // esi
  char v31; // cl
  __int64 v32; // r12
  __int64 *v33; // r15
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r14
  __int64 *v37; // rbx
  char v38; // r15
  __int64 v39; // r12
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  _QWORD *v44; // rdi
  _QWORD *v45; // rax
  __int64 *v46; // r15
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // r14
  __int64 *v50; // rbx
  char v51; // r15
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 v54; // rax
  __int64 v55; // rcx
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  _QWORD *v58; // rdi
  unsigned __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // r9
  __int64 v62; // r10
  __int64 v63; // rax
  _QWORD *v64; // r13
  __int64 *v65; // r15
  __int64 v66; // rax
  __int64 v67; // r8
  __int64 v68; // r14
  __int64 *v69; // rbx
  char v70; // r15
  __int64 v71; // r12
  __int64 v72; // r13
  __int64 v73; // rax
  __int64 v74; // rcx
  unsigned __int64 v75; // rdx
  _QWORD *v76; // rax
  __int64 *v77; // r15
  __int64 v78; // rax
  __int64 v79; // r8
  __int64 v80; // r14
  __int64 *v81; // rbx
  char v82; // r15
  __int64 v83; // r12
  __int64 v84; // r13
  __int64 v85; // rax
  __int64 v86; // rcx
  unsigned __int64 v87; // rdx
  __int64 v88; // rdx
  unsigned __int64 v89; // rax
  __int64 *v90; // r15
  __int64 v91; // rax
  __int64 v92; // r8
  __int64 v93; // r14
  __int64 *v94; // rbx
  char v95; // r15
  __int64 v96; // r12
  __int64 v97; // r13
  __int64 v98; // rax
  __int64 v99; // rcx
  unsigned __int64 v100; // rdx
  __int64 v101; // rdx
  unsigned __int64 v102; // rax
  __int64 *v103; // r15
  __int64 v104; // rax
  __int64 v105; // r8
  __int64 v106; // r14
  __int64 *v107; // rbx
  char v108; // r15
  __int64 v109; // r12
  __int64 v110; // r13
  __int64 v111; // rax
  unsigned __int64 v112; // rdx
  _QWORD *v113; // rax
  __int64 v114; // r12
  __int64 v115; // r12
  __int64 *v116; // r15
  __int64 v117; // rax
  __int64 v118; // r8
  __int64 v119; // r14
  __int64 *v120; // rbx
  char v121; // r15
  __int64 v122; // r12
  __int64 v123; // r13
  __int64 v124; // rax
  unsigned __int64 v125; // rdx
  __int64 *v126; // rax
  __int64 v127; // r12
  __int64 v128; // r12
  __int64 v129; // rsi
  int v130; // edx
  unsigned int v131; // eax
  __int64 v132; // rcx
  __int64 v133; // r8
  __int64 v134; // r11
  __int64 v135; // r12
  __int64 v136; // r13
  _BYTE *v137; // rax
  unsigned __int64 v138; // r13
  __int64 v139; // r9
  __int64 *v140; // r15
  __int64 *v141; // r15
  __int64 *v142; // r13
  __int64 *v143; // rdi
  _QWORD *v144; // rax
  _QWORD *v145; // r13
  __int64 v146; // r13
  __int64 v147; // rsi
  int v148; // edx
  unsigned int v149; // eax
  __int64 v150; // rcx
  int v151; // r10d
  __int64 *v152; // rdi
  int v153; // edx
  int v154; // edx
  _QWORD *v155; // rdi
  _BYTE *v156; // rdi
  int v157; // r10d
  __int64 v158; // [rsp+8h] [rbp-B8h]
  size_t v159; // [rsp+10h] [rbp-B0h]
  signed __int64 v160; // [rsp+10h] [rbp-B0h]
  void *src; // [rsp+18h] [rbp-A8h]
  void *srca; // [rsp+18h] [rbp-A8h]
  void *srcb; // [rsp+18h] [rbp-A8h]
  void *srcc; // [rsp+18h] [rbp-A8h]
  void *srcd; // [rsp+18h] [rbp-A8h]
  void *srce; // [rsp+18h] [rbp-A8h]
  void *srcf; // [rsp+18h] [rbp-A8h]
  _BYTE *srcg; // [rsp+18h] [rbp-A8h]
  void *srch; // [rsp+18h] [rbp-A8h]
  void *srci; // [rsp+18h] [rbp-A8h]
  void *srcj; // [rsp+18h] [rbp-A8h]
  __int64 *v172; // [rsp+20h] [rbp-A0h]
  __int64 v173; // [rsp+20h] [rbp-A0h]
  __int64 v174; // [rsp+20h] [rbp-A0h]
  __int64 v175; // [rsp+20h] [rbp-A0h]
  __int64 *v176; // [rsp+28h] [rbp-98h]
  __int64 *v177; // [rsp+28h] [rbp-98h]
  __int64 v178; // [rsp+28h] [rbp-98h]
  __int64 *v179; // [rsp+28h] [rbp-98h]
  __int64 *v180; // [rsp+28h] [rbp-98h]
  __int64 *v181; // [rsp+28h] [rbp-98h]
  __int64 *v182; // [rsp+28h] [rbp-98h]
  __int64 *v183; // [rsp+28h] [rbp-98h]
  _BYTE *v184; // [rsp+30h] [rbp-90h] BYREF
  __int64 v185; // [rsp+38h] [rbp-88h]
  _BYTE v186[32]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v187; // [rsp+60h] [rbp-60h] BYREF
  __int64 v188; // [rsp+68h] [rbp-58h]
  _QWORD *v189; // [rsp+70h] [rbp-50h] BYREF
  _QWORD *v190; // [rsp+78h] [rbp-48h]

  v7 = a1;
  v8 = *(_BYTE *)(a1 + 16);
  v9 = v8 & 1;
  if ( (v8 & 1) != 0 )
  {
    v10 = a1 + 24;
    v11 = 3;
  }
  else
  {
    v18 = *(unsigned int *)(a1 + 32);
    v10 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v18 )
      goto LABEL_13;
    v11 = v18 - 1;
  }
  v12 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
    goto LABEL_4;
  v20 = 1;
  while ( v14 != -4096 )
  {
    a6 = (unsigned int)(v20 + 1);
    v12 = v11 & (v20 + v12);
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_4;
    v20 = a6;
  }
  if ( v9 )
  {
    v19 = 64;
    goto LABEL_14;
  }
  v18 = *(unsigned int *)(v7 + 32);
LABEL_13:
  v19 = 16 * v18;
LABEL_14:
  v13 = (__int64 *)(v10 + v19);
LABEL_4:
  v15 = 64;
  if ( !v9 )
    v15 = 16LL * *(unsigned int *)(v7 + 32);
  if ( v13 != (__int64 *)(v10 + v15) )
    return v13[1];
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
    case 1:
      goto LABEL_19;
    case 2:
      v16 = a2;
      if ( *(_BYTE *)(v7 + 104) )
        goto LABEL_20;
      v128 = *(_QWORD *)(a2 + 32);
      if ( !sub_DADE90(*(_QWORD *)v7, v128, *(_QWORD *)(v7 + 96)) )
        v128 = sub_31AA7F0(v7, v128);
      if ( *(_QWORD *)(a2 + 32) == v128 )
        goto LABEL_68;
      v16 = (__int64)sub_DC5200(*(_QWORD *)v7, v128, *(_QWORD *)(a2 + 40), 0);
      v9 = *(_BYTE *)(v7 + 16) & 1;
      goto LABEL_20;
    case 3:
      v16 = a2;
      if ( *(_BYTE *)(v7 + 104) )
        goto LABEL_20;
      v127 = *(_QWORD *)(a2 + 32);
      if ( !sub_DADE90(*(_QWORD *)v7, v127, *(_QWORD *)(v7 + 96)) )
        v127 = sub_31AA7F0(v7, v127);
      if ( *(_QWORD *)(a2 + 32) == v127 )
        goto LABEL_68;
      v16 = (__int64)sub_DC2B70(*(_QWORD *)v7, v127, *(_QWORD *)(a2 + 40), 0);
      v9 = *(_BYTE *)(v7 + 16) & 1;
      goto LABEL_20;
    case 4:
      v16 = a2;
      if ( *(_BYTE *)(v7 + 104) )
        goto LABEL_20;
      v114 = *(_QWORD *)(a2 + 32);
      if ( !sub_DADE90(*(_QWORD *)v7, v114, *(_QWORD *)(v7 + 96)) )
        v114 = sub_31AA7F0(v7, v114);
      if ( *(_QWORD *)(a2 + 32) == v114 )
        goto LABEL_68;
      v16 = (__int64)sub_DC5000(*(_QWORD *)v7, v114, *(_QWORD *)(a2 + 40), 0);
      v9 = *(_BYTE *)(v7 + 16) & 1;
      goto LABEL_20;
    case 5:
      v103 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v104 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v182 = &v103[v104];
      if ( v103 == v182 )
        goto LABEL_19;
      v105 = 0;
      srce = (void *)a2;
      v106 = v7;
      v107 = v103;
      v108 = 0;
      do
      {
        v109 = *v107;
        v110 = *v107;
        if ( !*(_BYTE *)(v106 + 104) && !sub_DADE90(*(_QWORD *)v106, *v107, *(_QWORD *)(v106 + 96)) )
          v110 = sub_31AA7F0(v106, v109);
        v111 = (unsigned int)v188;
        v112 = (unsigned int)v188 + 1LL;
        if ( v112 > HIDWORD(v188) )
        {
          sub_C8D5F0((__int64)&v187, &v189, v112, 8u, v105, a6);
          v111 = (unsigned int)v188;
        }
        v187[v111] = v110;
        v58 = v187;
        LODWORD(v188) = v188 + 1;
        ++v107;
        v108 |= v187[(unsigned int)v188 - 1] != v109;
      }
      while ( v182 != v107 );
      v7 = v106;
      a2 = (__int64)srce;
      v16 = (__int64)srce;
      if ( v108 )
      {
        v113 = sub_DC7EB0(*(__int64 **)v7, (__int64)&v187, 0, 0);
        v58 = v187;
        v16 = (__int64)v113;
      }
      goto LABEL_109;
    case 6:
      v116 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v117 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v183 = &v116[v117];
      if ( v116 == v183 )
        goto LABEL_19;
      v118 = 0;
      srcf = (void *)a2;
      v119 = v7;
      v120 = v116;
      v121 = 0;
      do
      {
        v122 = *v120;
        v123 = *v120;
        if ( !*(_BYTE *)(v119 + 104) && !sub_DADE90(*(_QWORD *)v119, *v120, *(_QWORD *)(v119 + 96)) )
          v123 = sub_31AA7F0(v119, v122);
        v124 = (unsigned int)v188;
        v125 = (unsigned int)v188 + 1LL;
        if ( v125 > HIDWORD(v188) )
        {
          sub_C8D5F0((__int64)&v187, &v189, v125, 8u, v118, a6);
          v124 = (unsigned int)v188;
        }
        v187[v124] = v123;
        v58 = v187;
        LODWORD(v188) = v188 + 1;
        ++v120;
        v121 |= v187[(unsigned int)v188 - 1] != v122;
      }
      while ( v183 != v120 );
      v7 = v119;
      a2 = (__int64)srcf;
      v16 = (__int64)srcf;
      if ( v121 )
      {
        v126 = sub_DC8BD0(*(__int64 **)v7, (__int64)&v187, 0, 0);
        v58 = v187;
        v16 = (__int64)v126;
      }
      goto LABEL_109;
    case 7:
      v115 = *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)(v7 + 104) )
        goto LABEL_118;
      if ( !sub_DADE90(*(_QWORD *)v7, *(_QWORD *)(a2 + 32), *(_QWORD *)(v7 + 96)) )
        v115 = sub_31AA7F0(v7, v115);
      v146 = *(_QWORD *)(a2 + 40);
      if ( *(_BYTE *)(v7 + 104) )
      {
        if ( v115 != *(_QWORD *)(a2 + 32) )
        {
LABEL_172:
          v16 = sub_DCB270(*(_QWORD *)v7, v115, v146);
          v9 = *(_BYTE *)(v7 + 16) & 1;
          goto LABEL_20;
        }
        v8 = *(_BYTE *)(v7 + 16);
LABEL_118:
        v16 = a2;
        v9 = v8 & 1;
      }
      else
      {
        if ( !sub_DADE90(*(_QWORD *)v7, *(_QWORD *)(a2 + 40), *(_QWORD *)(v7 + 96)) )
          v146 = sub_31AA7F0(v7, v146);
        if ( *(_QWORD *)(a2 + 32) != v115 || *(_QWORD *)(a2 + 40) != v146 )
          goto LABEL_172;
LABEL_68:
        v16 = a2;
        v9 = *(_BYTE *)(v7 + 16) & 1;
      }
LABEL_20:
      if ( v9 )
      {
        v21 = v7 + 24;
        v22 = 3;
      }
      else
      {
        v30 = *(_DWORD *)(v7 + 32);
        v21 = *(_QWORD *)(v7 + 24);
        if ( !v30 )
        {
          v27 = *(_DWORD *)(v7 + 16);
          ++*(_QWORD *)(v7 + 8);
          v26 = 0;
          v28 = (v27 >> 1) + 1;
          goto LABEL_142;
        }
        v22 = v30 - 1;
      }
      v23 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v21 + 16LL * v23);
      v24 = *v13;
      if ( *v13 == a2 )
        return v13[1];
      v25 = 1;
      v26 = 0;
      while ( v24 != -4096 )
      {
        if ( !v26 && v24 == -8192 )
          v26 = v13;
        v23 = v22 & (v25 + v23);
        v13 = (__int64 *)(v21 + 16LL * v23);
        v24 = *v13;
        if ( *v13 == a2 )
          return v13[1];
        ++v25;
      }
      if ( !v26 )
        v26 = v13;
      v27 = *(_DWORD *)(v7 + 16);
      ++*(_QWORD *)(v7 + 8);
      v28 = (v27 >> 1) + 1;
      if ( v9 )
      {
        v29 = 12;
        v30 = 4;
        goto LABEL_29;
      }
      v30 = *(_DWORD *)(v7 + 32);
LABEL_142:
      v29 = 3 * v30;
LABEL_29:
      if ( 4 * v28 >= v29 )
      {
        sub_DB0DD0(v7 + 8, 2 * v30);
        if ( (*(_BYTE *)(v7 + 16) & 1) != 0 )
        {
          v129 = v7 + 24;
          v130 = 3;
        }
        else
        {
          v153 = *(_DWORD *)(v7 + 32);
          v129 = *(_QWORD *)(v7 + 24);
          if ( !v153 )
            goto LABEL_211;
          v130 = v153 - 1;
        }
        v131 = v130 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v26 = (__int64 *)(v129 + 16LL * v131);
        v132 = *v26;
        if ( *v26 != a2 )
        {
          v157 = 1;
          v152 = 0;
          while ( v132 != -4096 )
          {
            if ( !v152 && v132 == -8192 )
              v152 = v26;
            v131 = v130 & (v157 + v131);
            v26 = (__int64 *)(v129 + 16LL * v131);
            v132 = *v26;
            if ( *v26 == a2 )
              goto LABEL_148;
            ++v157;
          }
          goto LABEL_178;
        }
LABEL_148:
        v27 = *(_DWORD *)(v7 + 16);
        goto LABEL_31;
      }
      if ( v30 - *(_DWORD *)(v7 + 20) - v28 <= v30 >> 3 )
      {
        sub_DB0DD0(v7 + 8, v30);
        if ( (*(_BYTE *)(v7 + 16) & 1) != 0 )
        {
          v147 = v7 + 24;
          v148 = 3;
          goto LABEL_175;
        }
        v154 = *(_DWORD *)(v7 + 32);
        v147 = *(_QWORD *)(v7 + 24);
        if ( v154 )
        {
          v148 = v154 - 1;
LABEL_175:
          v149 = v148 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v26 = (__int64 *)(v147 + 16LL * v149);
          v150 = *v26;
          if ( *v26 != a2 )
          {
            v151 = 1;
            v152 = 0;
            while ( v150 != -4096 )
            {
              if ( v150 == -8192 && !v152 )
                v152 = v26;
              v149 = v148 & (v151 + v149);
              v26 = (__int64 *)(v147 + 16LL * v149);
              v150 = *v26;
              if ( *v26 == a2 )
                goto LABEL_148;
              ++v151;
            }
LABEL_178:
            if ( v152 )
              v26 = v152;
            goto LABEL_148;
          }
          goto LABEL_148;
        }
LABEL_211:
        *(_DWORD *)(v7 + 16) = (2 * (*(_DWORD *)(v7 + 16) >> 1) + 2) | *(_DWORD *)(v7 + 16) & 1;
        BUG();
      }
LABEL_31:
      *(_DWORD *)(v7 + 16) = (2 * (v27 >> 1) + 2) | v27 & 1;
      if ( *v26 != -4096 )
        --*(_DWORD *)(v7 + 20);
      *v26 = a2;
      v26[1] = v16;
      return v16;
    case 8:
      v60 = sub_D95540(**(_QWORD **)(a2 + 32));
      v62 = *(_QWORD *)v7;
      v178 = v60;
      v63 = *(_QWORD *)(a2 + 40);
      if ( v63 == 2 )
      {
        v64 = *(_QWORD **)(*(_QWORD *)(a2 + 32) + 8LL);
        goto LABEL_66;
      }
      v133 = 8 * v63 - 8;
      v134 = *(_QWORD *)(a2 + 48);
      v135 = *(_QWORD *)(a2 + 32);
      v184 = v186;
      v185 = 0x300000000LL;
      v136 = v133 >> 3;
      if ( (unsigned __int64)v133 > 0x18 )
      {
        v160 = 8 * v63 - 8;
        srci = (void *)v134;
        v174 = v62;
        sub_C8D5F0((__int64)&v184, v186, v160 >> 3, 8u, v133, v61);
        v62 = v174;
        v134 = (__int64)srci;
        v133 = v160;
        v156 = &v184[8 * (unsigned int)v185];
      }
      else
      {
        v137 = v186;
        if ( !v133 )
          goto LABEL_151;
        v156 = v186;
      }
      srcj = (void *)v134;
      v175 = v62;
      memcpy(v156, (const void *)(v135 + 8), v133);
      v137 = v184;
      LODWORD(v133) = v185;
      v134 = (__int64)srcj;
      v62 = v175;
LABEL_151:
      LODWORD(v185) = v133 + v136;
      v138 = (unsigned int)(v133 + v136);
      v187 = &v189;
      v139 = 8 * v138;
      v188 = 0x400000000LL;
      if ( v138 > 4 )
      {
        v158 = v134;
        v159 = v62;
        srcg = v137;
        sub_C8D5F0((__int64)&v187, &v189, v138, 8u, (__int64)&v187, v139);
        v137 = srcg;
        v62 = v159;
        v134 = v158;
        v155 = &v187[(unsigned int)v188];
        v139 = 8 * v138;
      }
      else
      {
        if ( !v139 )
          goto LABEL_153;
        v155 = &v189;
      }
      srch = (void *)v134;
      v173 = v62;
      memcpy(v155, v137, v139);
      LODWORD(v139) = v188;
      v134 = (__int64)srch;
      v62 = v173;
LABEL_153:
      LODWORD(v188) = v138 + v139;
      v64 = sub_DBFF60(v62, (unsigned int *)&v187, v134, 0);
      if ( v187 != &v189 )
        _libc_free((unsigned __int64)v187);
      if ( v184 != v186 )
        _libc_free((unsigned __int64)v184);
      v62 = *(_QWORD *)v7;
LABEL_66:
      if ( !sub_DADE90(v62, (__int64)v64, *(_QWORD *)(v7 + 96)) )
      {
        *(_BYTE *)(v7 + 104) = 1;
        goto LABEL_68;
      }
      v140 = *(__int64 **)v7;
      v190 = sub_DA2C50(*(_QWORD *)v7, v178, *(unsigned int *)(v7 + 88), 0);
      v187 = &v189;
      v189 = v64;
      v188 = 0x200000002LL;
      v141 = sub_DC8BD0(v140, (__int64)&v187, 0, 0);
      if ( v187 != &v189 )
        _libc_free((unsigned __int64)v187);
      v172 = *(__int64 **)v7;
      v190 = sub_DA2C50(*(_QWORD *)v7, v178, *(unsigned int *)(v7 + 92), 0);
      v189 = v64;
      v187 = &v189;
      v188 = 0x200000002LL;
      v142 = sub_DC8BD0(v172, (__int64)&v187, 0, 0);
      if ( v187 != &v189 )
        _libc_free((unsigned __int64)v187);
      v143 = *(__int64 **)v7;
      v144 = **(_QWORD ***)(a2 + 32);
      v190 = v142;
      v187 = &v189;
      v189 = v144;
      v188 = 0x200000002LL;
      v145 = sub_DC7EB0(v143, (__int64)&v187, 0, 0);
      if ( v187 != &v189 )
        _libc_free((unsigned __int64)v187);
      v16 = (__int64)sub_DC1960(*(_QWORD *)v7, (__int64)v145, (__int64)v141, *(_QWORD *)(v7 + 96), 0);
      goto LABEL_111;
    case 9:
      v46 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v47 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v177 = &v46[v47];
      if ( v46 == v177 )
        goto LABEL_19;
      v48 = 0;
      srca = (void *)a2;
      v49 = v7;
      v50 = v46;
      v51 = 0;
      do
      {
        v52 = *v50;
        v53 = *v50;
        if ( !*(_BYTE *)(v49 + 104) && !sub_DADE90(*(_QWORD *)v49, *v50, *(_QWORD *)(v49 + 96)) )
          v53 = sub_31AA7F0(v49, v52);
        v54 = (unsigned int)v188;
        v55 = HIDWORD(v188);
        v56 = (unsigned int)v188 + 1LL;
        if ( v56 > HIDWORD(v188) )
        {
          sub_C8D5F0((__int64)&v187, &v189, v56, 8u, v48, a6);
          v54 = (unsigned int)v188;
        }
        v57 = (__int64)v187;
        v187[v54] = v53;
        v58 = v187;
        LODWORD(v188) = v188 + 1;
        ++v50;
        v51 |= v187[(unsigned int)v188 - 1] != v52;
      }
      while ( v177 != v50 );
      v7 = v49;
      a2 = (__int64)srca;
      v16 = (__int64)srca;
      if ( v51 )
      {
        v59 = sub_DCE040(*(__int64 **)v7, (__int64)&v187, v57, v55, v48);
        v58 = v187;
        v16 = v59;
      }
      goto LABEL_109;
    case 0xA:
      v90 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v91 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v181 = &v90[v91];
      if ( v90 == v181 )
        goto LABEL_19;
      v92 = 0;
      srcd = (void *)a2;
      v93 = v7;
      v94 = v90;
      v95 = 0;
      do
      {
        v96 = *v94;
        v97 = *v94;
        if ( !*(_BYTE *)(v93 + 104) && !sub_DADE90(*(_QWORD *)v93, *v94, *(_QWORD *)(v93 + 96)) )
          v97 = sub_31AA7F0(v93, v96);
        v98 = (unsigned int)v188;
        v99 = HIDWORD(v188);
        v100 = (unsigned int)v188 + 1LL;
        if ( v100 > HIDWORD(v188) )
        {
          sub_C8D5F0((__int64)&v187, &v189, v100, 8u, v92, a6);
          v98 = (unsigned int)v188;
        }
        v101 = (__int64)v187;
        v187[v98] = v97;
        v58 = v187;
        LODWORD(v188) = v188 + 1;
        ++v94;
        v95 |= v187[(unsigned int)v188 - 1] != v96;
      }
      while ( v181 != v94 );
      v7 = v93;
      a2 = (__int64)srcd;
      v16 = (__int64)srcd;
      if ( v95 )
      {
        v102 = sub_DCDF90(*(__int64 **)v7, (__int64)&v187, v101, v99, v92);
        v58 = v187;
        v16 = v102;
      }
      goto LABEL_109;
    case 0xB:
      v65 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v66 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v179 = &v65[v66];
      if ( v65 == v179 )
        goto LABEL_19;
      v67 = 0;
      srcb = (void *)a2;
      v68 = v7;
      v69 = v65;
      v70 = 0;
      do
      {
        v71 = *v69;
        v72 = *v69;
        if ( !*(_BYTE *)(v68 + 104) && !sub_DADE90(*(_QWORD *)v68, *v69, *(_QWORD *)(v68 + 96)) )
          v72 = sub_31AA7F0(v68, v71);
        v73 = (unsigned int)v188;
        v74 = HIDWORD(v188);
        v75 = (unsigned int)v188 + 1LL;
        if ( v75 > HIDWORD(v188) )
        {
          sub_C8D5F0((__int64)&v187, &v189, v75, 8u, v67, a6);
          v73 = (unsigned int)v188;
        }
        v187[v73] = v72;
        v58 = v187;
        LODWORD(v188) = v188 + 1;
        ++v69;
        v70 |= v187[(unsigned int)v188 - 1] != v71;
      }
      while ( v179 != v69 );
      v7 = v68;
      a2 = (__int64)srcb;
      v16 = (__int64)srcb;
      if ( v70 )
      {
        v76 = sub_DCEE50(*(__int64 **)v7, (__int64)&v187, 0, v74, v67);
        v58 = v187;
        v16 = (__int64)v76;
      }
      goto LABEL_109;
    case 0xC:
      v77 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v78 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v180 = &v77[v78];
      if ( v77 == v180 )
        goto LABEL_19;
      v79 = 0;
      srcc = (void *)a2;
      v80 = v7;
      v81 = v77;
      v82 = 0;
      do
      {
        v83 = *v81;
        v84 = *v81;
        if ( !*(_BYTE *)(v80 + 104) && !sub_DADE90(*(_QWORD *)v80, *v81, *(_QWORD *)(v80 + 96)) )
          v84 = sub_31AA7F0(v80, v83);
        v85 = (unsigned int)v188;
        v86 = HIDWORD(v188);
        v87 = (unsigned int)v188 + 1LL;
        if ( v87 > HIDWORD(v188) )
        {
          sub_C8D5F0((__int64)&v187, &v189, v87, 8u, v79, a6);
          v85 = (unsigned int)v188;
        }
        v88 = (__int64)v187;
        v187[v85] = v84;
        v58 = v187;
        LODWORD(v188) = v188 + 1;
        ++v81;
        v82 |= v187[(unsigned int)v188 - 1] != v83;
      }
      while ( v180 != v81 );
      v7 = v80;
      a2 = (__int64)srcc;
      v16 = (__int64)srcc;
      if ( v82 )
      {
        v89 = sub_DCE150(*(__int64 **)v7, (__int64)&v187, v88, v86, v79);
        v58 = v187;
        v16 = v89;
      }
LABEL_109:
      if ( v58 != &v189 )
        _libc_free((unsigned __int64)v58);
      goto LABEL_111;
    case 0xD:
      v33 = *(__int64 **)(a2 + 32);
      v188 = 0x200000000LL;
      v34 = *(_QWORD *)(a2 + 40);
      v187 = &v189;
      v176 = &v33[v34];
      if ( v33 == v176 )
      {
LABEL_19:
        v16 = a2;
      }
      else
      {
        v35 = 0;
        src = (void *)a2;
        v36 = v7;
        v37 = v33;
        v38 = 0;
        do
        {
          v39 = *v37;
          v40 = *v37;
          if ( !*(_BYTE *)(v36 + 104) && !sub_DADE90(*(_QWORD *)v36, *v37, *(_QWORD *)(v36 + 96)) )
            v40 = sub_31AA7F0(v36, v39);
          v41 = (unsigned int)v188;
          v42 = HIDWORD(v188);
          v43 = (unsigned int)v188 + 1LL;
          if ( v43 > HIDWORD(v188) )
          {
            sub_C8D5F0((__int64)&v187, &v189, v43, 8u, v35, a6);
            v41 = (unsigned int)v188;
          }
          v187[v41] = v40;
          v44 = v187;
          LODWORD(v188) = v188 + 1;
          ++v37;
          v38 |= v187[(unsigned int)v188 - 1] != v39;
        }
        while ( v176 != v37 );
        v7 = v36;
        a2 = (__int64)src;
        v16 = (__int64)src;
        if ( v38 )
        {
          v45 = sub_DCEE50(*(__int64 **)v7, (__int64)&v187, 1, v42, v35);
          v44 = v187;
          v16 = (__int64)v45;
        }
        if ( v44 == &v189 )
        {
LABEL_111:
          v9 = *(_BYTE *)(v7 + 16) & 1;
        }
        else
        {
          _libc_free((unsigned __int64)v44);
          v9 = *(_BYTE *)(v7 + 16) & 1;
        }
      }
      goto LABEL_20;
    case 0xE:
      v16 = a2;
      if ( *(_BYTE *)(v7 + 104) )
        goto LABEL_20;
      v32 = *(_QWORD *)(a2 + 32);
      if ( !sub_DADE90(*(_QWORD *)v7, v32, *(_QWORD *)(v7 + 96)) )
        v32 = sub_31AA7F0(v7, v32);
      if ( *(_QWORD *)(a2 + 32) == v32 )
        goto LABEL_68;
      v16 = (__int64)sub_DD3A70(*(_QWORD *)v7, v32, *(_QWORD *)(a2 + 40));
      v9 = *(_BYTE *)(v7 + 16) & 1;
      goto LABEL_20;
    case 0xF:
      v16 = a2;
      if ( sub_DADE90(*(_QWORD *)v7, a2, *(_QWORD *)(v7 + 96)) )
        goto LABEL_111;
      v31 = *(_BYTE *)(v7 + 16);
      *(_BYTE *)(v7 + 104) = 1;
      v9 = v31 & 1;
      goto LABEL_20;
    case 0x10:
      *(_BYTE *)(v7 + 104) = 1;
      v16 = a2;
      goto LABEL_20;
    default:
      BUG();
  }
}
