// Function: sub_1ADAFF0
// Address: 0x1adaff0
//
__int64 __fastcall sub_1ADAFF0(
        __int64 a1,
        unsigned __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 *v14; // rax
  __int64 v15; // rax
  bool v16; // zf
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 i; // r14
  char *v20; // r12
  __int64 v21; // r14
  _BYTE *v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rbx
  int v32; // r8d
  __int64 v33; // r9
  __int64 v34; // rax
  char *v35; // r12
  _QWORD *v36; // r8
  unsigned int v37; // ecx
  _QWORD *v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // r14
  __int64 v41; // rsi
  __int64 v42; // r13
  __int64 v43; // r9
  __int64 v44; // r13
  __int64 *v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r8
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 v52; // rcx
  __int64 v53; // r14
  __int64 v54; // rbx
  __int64 *v55; // rax
  unsigned int v56; // eax
  int v57; // ecx
  __int64 v58; // rsi
  __int64 *v59; // r14
  __int64 *v60; // r12
  __int64 *v61; // rax
  unsigned __int8 *v62; // r12
  double v63; // xmm4_8
  double v64; // xmm5_8
  unsigned int v65; // edi
  _QWORD *v66; // rax
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rdx
  __int64 v71; // rsi
  __int64 v72; // rax
  _QWORD *v73; // rbx
  _QWORD *v74; // r12
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  double v78; // xmm4_8
  double v79; // xmm5_8
  __int64 v80; // rbx
  _BYTE *v81; // r12
  __int64 v82; // rdi
  __int64 v84; // r12
  unsigned __int64 v85; // r14
  __int64 v86; // rbx
  __int64 v87; // r13
  __int64 v88; // rax
  int v89; // esi
  unsigned int v90; // ecx
  __int64 *v91; // rdx
  __int64 v92; // r8
  __int64 v93; // rdx
  __int64 v94; // rax
  __int64 v95; // rax
  int v96; // esi
  unsigned int v97; // ecx
  __int64 *v98; // rdx
  __int64 v99; // r8
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // r13
  __int64 v104; // rax
  __int64 v105; // r15
  __int64 v106; // r12
  unsigned int v107; // edi
  _QWORD *v108; // rax
  _BYTE *v109; // rcx
  _BYTE *v110; // rbx
  __int64 v111; // rax
  int v112; // ecx
  __int64 v113; // rax
  _QWORD *v114; // rdx
  int v115; // r10d
  unsigned int v116; // r14d
  _BYTE *v117; // rsi
  unsigned __int64 v118; // rax
  unsigned __int64 v119; // rax
  unsigned __int64 v120; // rcx
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rcx
  __int64 v124; // rcx
  __int64 v125; // rcx
  _QWORD *v126; // rax
  __int64 v127; // r13
  _BYTE *v128; // rbx
  __int64 v129; // rdi
  int v130; // r10d
  unsigned int v131; // eax
  __int64 v132; // rsi
  int v133; // r10d
  _QWORD *v134; // rdx
  int v135; // eax
  unsigned int v136; // ecx
  __int64 v137; // rsi
  int v138; // r11d
  _QWORD *v139; // rdi
  _QWORD *v140; // rsi
  unsigned int v141; // ebx
  int v142; // r10d
  __int64 v143; // rcx
  _QWORD *v144; // r11
  int v145; // r11d
  __int64 *v146; // rdi
  int v147; // ecx
  int v148; // r11d
  __int64 *v149; // rdi
  int v150; // ecx
  int v151; // r10d
  __int64 *v152; // r11
  int v153; // edi
  __int64 *v154; // r11
  int v155; // edi
  __int64 *v156; // r11
  int v157; // r10d
  unsigned __int64 v158; // [rsp+8h] [rbp-238h]
  __int64 v160; // [rsp+28h] [rbp-218h]
  char *v161; // [rsp+30h] [rbp-210h]
  unsigned int v162; // [rsp+30h] [rbp-210h]
  __int64 v163; // [rsp+30h] [rbp-210h]
  __int64 v164; // [rsp+38h] [rbp-208h]
  int v165; // [rsp+40h] [rbp-200h]
  __int64 v166; // [rsp+48h] [rbp-1F8h]
  char *v167; // [rsp+48h] [rbp-1F8h]
  char *v168; // [rsp+48h] [rbp-1F8h]
  __int64 v169; // [rsp+48h] [rbp-1F8h]
  __int64 v170; // [rsp+48h] [rbp-1F8h]
  __int64 v171; // [rsp+48h] [rbp-1F8h]
  __int64 v172; // [rsp+50h] [rbp-1F0h] BYREF
  __int64 *v173; // [rsp+58h] [rbp-1E8h] BYREF
  __int64 v174; // [rsp+60h] [rbp-1E0h] BYREF
  _QWORD *v175; // [rsp+68h] [rbp-1D8h]
  __int64 v176; // [rsp+70h] [rbp-1D0h]
  unsigned int v177; // [rsp+78h] [rbp-1C8h]
  __int64 *v178; // [rsp+80h] [rbp-1C0h] BYREF
  __int64 v179; // [rsp+88h] [rbp-1B8h]
  __int64 v180; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v181; // [rsp+98h] [rbp-1A8h]
  __int64 v182; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v183; // [rsp+B8h] [rbp-188h]
  __int64 v184; // [rsp+C0h] [rbp-180h]
  __int64 v185; // [rsp+C8h] [rbp-178h]
  char *v186; // [rsp+D0h] [rbp-170h]
  char *v187; // [rsp+D8h] [rbp-168h]
  __int64 v188; // [rsp+E0h] [rbp-160h]
  _BYTE *v189; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v190; // [rsp+F8h] [rbp-148h]
  _BYTE v191[128]; // [rsp+100h] [rbp-140h] BYREF
  _BYTE *v192; // [rsp+180h] [rbp-C0h] BYREF
  __int64 v193; // [rsp+188h] [rbp-B8h]
  _BYTE v194[176]; // [rsp+190h] [rbp-B0h] BYREF

  v158 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v14 = (__int64 *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  if ( (a1 & 4) != 0 )
    v14 = (__int64 *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v182 = 0;
  v183 = 0;
  v15 = *v14;
  v16 = *(_BYTE *)(v15 + 16) == 0;
  v164 = v15;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  if ( !v16 )
    BUG();
  v17 = *(_QWORD *)(v15 + 80);
  v166 = v15 + 72;
  if ( v15 + 72 == v17 )
    return j___libc_free_0(v183);
  do
  {
    if ( !v17 )
      BUG();
    v18 = *(_QWORD *)(v17 + 24);
    for ( i = v17 + 16; v18 != i; v18 = *(_QWORD *)(v18 + 8) )
    {
      while ( 1 )
      {
        if ( !v18 )
          BUG();
        if ( *(_QWORD *)(v18 + 24) || *(__int16 *)(v18 - 6) < 0 )
        {
          v192 = (_BYTE *)sub_1625790(v18 - 24, 7);
          if ( v192 )
            sub_1ADAD80((__int64)&v182, (__int64 *)&v192);
          if ( *(_QWORD *)(v18 + 24) )
            break;
        }
        if ( *(__int16 *)(v18 - 6) < 0 )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( v18 == i )
          goto LABEL_19;
      }
      v192 = (_BYTE *)sub_1625790(v18 - 24, 8);
      if ( v192 )
        sub_1ADAD80((__int64)&v182, (__int64 *)&v192);
    }
LABEL_19:
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v166 != v17 );
  v20 = v186;
  if ( v186 == v187 )
  {
    if ( v186 )
      j_j___libc_free_0(v186, v188 - (_QWORD)v186);
    return j___libc_free_0(v183);
  }
  v21 = v187 - v186;
  v22 = v191;
  v189 = v191;
  v23 = (v187 - v186) >> 3;
  v190 = 0x1000000000LL;
  if ( (unsigned __int64)(v187 - v186) <= 0x80
    || (sub_16CD150((__int64)&v189, v191, v21 >> 3, 8, a13, a14),
        v25 = v190,
        v22 = &v189[8 * (unsigned int)v190],
        v21 > 0) )
  {
    v24 = 0;
    do
    {
      *(_QWORD *)&v22[8 * v24] = *(_QWORD *)&v20[8 * v24];
      ++v24;
    }
    while ( v23 - v24 > 0 );
    v25 = v190;
  }
  LODWORD(v190) = v25 + v23;
  v26 = v25 + v23;
  if ( !(v25 + (_DWORD)v23) )
    goto LABEL_37;
  do
  {
    v27 = v26--;
    v28 = *(_QWORD *)&v189[8 * v27 - 8];
    LODWORD(v190) = v26;
    v29 = *(unsigned int *)(v28 + 8);
    if ( !(_DWORD)v29 )
      continue;
    v30 = *(unsigned int *)(v28 + 8);
    v31 = 0;
    while ( 1 )
    {
      if ( (unsigned __int8)(**(_BYTE **)(v28 + 8 * (v31 - v29)) - 4) <= 0x1Eu )
      {
        v192 = *(_BYTE **)(v28 + 8 * (v31 - v29));
        if ( (unsigned __int8)sub_1ADAD80((__int64)&v182, (__int64 *)&v192) )
          break;
      }
      if ( v30 == ++v31 )
        goto LABEL_35;
LABEL_29:
      v29 = *(unsigned int *)(v28 + 8);
    }
    v33 = (__int64)v192;
    v34 = (unsigned int)v190;
    if ( (unsigned int)v190 >= HIDWORD(v190) )
    {
      v169 = (__int64)v192;
      sub_16CD150((__int64)&v189, v191, 0, 8, v32, (int)v192);
      v34 = (unsigned int)v190;
      v33 = v169;
    }
    ++v31;
    *(_QWORD *)&v189[8 * v34] = v33;
    LODWORD(v190) = v190 + 1;
    if ( v30 != v31 )
      goto LABEL_29;
LABEL_35:
    v26 = v190;
  }
  while ( v26 );
LABEL_37:
  v35 = v186;
  v174 = 0;
  v192 = v194;
  v193 = 0x1000000000LL;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v167 = v187;
  if ( v187 != v186 )
  {
    while ( 1 )
    {
      v44 = *(_QWORD *)v35;
      v45 = (__int64 *)sub_15E0530(v164);
      v46 = 0;
      v47 = sub_1627350(v45, 0, 0, 2, 1);
      v52 = (unsigned int)v193;
      v53 = v47;
      if ( (unsigned int)v193 >= HIDWORD(v193) )
      {
        v162 = v193;
        v118 = (((((unsigned __int64)HIDWORD(v193) + 2) >> 1) | (HIDWORD(v193) + 2LL)) >> 2)
             | (((unsigned __int64)HIDWORD(v193) + 2) >> 1)
             | (HIDWORD(v193) + 2LL);
        v119 = (((v118 >> 4) | v118) >> 8) | (v118 >> 4) | v118;
        v120 = (v119 | (v119 >> 16) | HIDWORD(v119)) + 1;
        v121 = 0xFFFFFFFFLL;
        if ( v120 <= 0xFFFFFFFF )
          v121 = v120;
        v165 = v121;
        v122 = malloc(8 * v121);
        v123 = v162;
        v54 = v122;
        if ( !v122 )
        {
          v46 = 1;
          sub_16BD1C0("Allocation failed", 1u);
          v123 = (unsigned int)v193;
        }
        v48 = (__int64)v192;
        v124 = 8 * v123;
        v43 = (__int64)&v192[v124];
        if ( v192 != &v192[v124] )
        {
          v125 = v54 + v124;
          v126 = (_QWORD *)v54;
          do
          {
            if ( v126 )
            {
              v46 = *(_QWORD *)v48;
              *v126 = *(_QWORD *)v48;
              *(_QWORD *)v48 = 0;
            }
            ++v126;
            v48 += 8;
          }
          while ( v126 != (_QWORD *)v125 );
          v48 = (unsigned int)v193;
          v43 = (__int64)&v192[8 * (unsigned int)v193];
          if ( v192 != (_BYTE *)v43 )
          {
            v163 = v44;
            v127 = (__int64)v192;
            v160 = v54;
            v128 = &v192[8 * (unsigned int)v193];
            do
            {
              v129 = *((_QWORD *)v128 - 1);
              v128 -= 8;
              if ( v129 )
                sub_16307F0(v129, v46, v48, v125, v49, a3, a4, a5, a6, v50, v51, a9, a10);
            }
            while ( (_BYTE *)v127 != v128 );
            v44 = v163;
            v54 = v160;
            v43 = (__int64)v192;
          }
        }
        if ( (_BYTE *)v43 != v194 )
          _libc_free(v43);
        v192 = (_BYTE *)v54;
        v52 = (unsigned int)v193;
        HIDWORD(v193) = v165;
      }
      else
      {
        v54 = (__int64)v192;
      }
      v55 = (__int64 *)(v54 + 8LL * (unsigned int)v52);
      if ( v55 )
      {
        *v55 = v53;
        LODWORD(v193) = v193 + 1;
      }
      else
      {
        LODWORD(v193) = v52 + 1;
        if ( v53 )
          sub_16307F0(v53, v46, v48, v52, v49, a3, a4, a5, a6, v50, v51, a9, a10);
      }
      if ( v177 )
      {
        LODWORD(v36) = v177 - 1;
        v37 = (v177 - 1) & (((unsigned int)v44 >> 4) ^ ((unsigned int)v44 >> 9));
        v38 = &v175[2 * v37];
        v39 = *v38;
        if ( v44 == *v38 )
        {
LABEL_40:
          v40 = (__int64)(v38 + 1);
          v41 = v38[1];
          v42 = *(_QWORD *)&v192[8 * (unsigned int)v193 - 8];
          if ( v41 )
            sub_161E7C0((__int64)(v38 + 1), v41);
          goto LABEL_42;
        }
        v130 = 1;
        v43 = 0;
        while ( v39 != -8 )
        {
          if ( v39 != -16 || v43 )
            v38 = (_QWORD *)v43;
          LODWORD(v43) = v130 + 1;
          v37 = (unsigned int)v36 & (v130 + v37);
          v39 = v175[2 * v37];
          if ( v44 == v39 )
          {
            v38 = &v175[2 * v37];
            goto LABEL_40;
          }
          ++v130;
          v43 = (__int64)v38;
          v38 = &v175[2 * v37];
        }
        if ( v43 )
          v38 = (_QWORD *)v43;
        ++v174;
        v57 = v176 + 1;
        if ( 4 * ((int)v176 + 1) < 3 * v177 )
        {
          if ( v177 - HIDWORD(v176) - v57 > v177 >> 3 )
            goto LABEL_53;
          sub_1AD8C40((__int64)&v174, v177);
          if ( !v177 )
          {
LABEL_318:
            LODWORD(v176) = v176 + 1;
            BUG();
          }
          v36 = 0;
          LODWORD(v43) = 1;
          v131 = (v177 - 1) & (((unsigned int)v44 >> 4) ^ ((unsigned int)v44 >> 9));
          v57 = v176 + 1;
          v38 = &v175[2 * v131];
          v132 = *v38;
          if ( v44 == *v38 )
            goto LABEL_53;
          while ( v132 != -8 )
          {
            if ( v132 == -16 && !v36 )
              v36 = v38;
            v131 = (v177 - 1) & (v43 + v131);
            v38 = &v175[2 * v131];
            v132 = *v38;
            if ( v44 == *v38 )
              goto LABEL_53;
            LODWORD(v43) = v43 + 1;
          }
          goto LABEL_264;
        }
      }
      else
      {
        ++v174;
      }
      sub_1AD8C40((__int64)&v174, 2 * v177);
      if ( !v177 )
        goto LABEL_318;
      v56 = (v177 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v57 = v176 + 1;
      v38 = &v175[2 * v56];
      v58 = *v38;
      if ( v44 == *v38 )
        goto LABEL_53;
      LODWORD(v43) = 1;
      v36 = 0;
      while ( v58 != -8 )
      {
        if ( !v36 && v58 == -16 )
          v36 = v38;
        v56 = (v177 - 1) & (v43 + v56);
        v38 = &v175[2 * v56];
        v58 = *v38;
        if ( v44 == *v38 )
          goto LABEL_53;
        LODWORD(v43) = v43 + 1;
      }
LABEL_264:
      if ( v36 )
        v38 = v36;
LABEL_53:
      LODWORD(v176) = v57;
      if ( *v38 != -8 )
        --HIDWORD(v176);
      *v38 = v44;
      v40 = (__int64)(v38 + 1);
      v38[1] = 0;
      v42 = *(_QWORD *)&v192[8 * (unsigned int)v193 - 8];
LABEL_42:
      v38[1] = v42;
      if ( v42 )
        sub_1623A60(v40, v42, 2);
      v35 += 8;
      if ( v167 == v35 )
      {
        v161 = v187;
        if ( v187 == v186 )
          break;
        v168 = v186;
LABEL_132:
        v103 = *(_QWORD *)v168;
        v178 = &v180;
        v179 = 0x400000000LL;
        v104 = *(unsigned int *)(v103 + 8);
        if ( (_DWORD)v104 )
        {
          v105 = *(unsigned int *)(v103 + 8);
          v106 = 0;
          while ( 1 )
          {
            v110 = *(_BYTE **)(v103 + 8 * (v106 - v104));
            if ( (unsigned __int8)(*v110 - 4) > 0x1Eu )
              goto LABEL_136;
            if ( !v177 )
              break;
            LODWORD(v43) = v177 - 1;
            LODWORD(v36) = (_DWORD)v175;
            v107 = (v177 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
            v108 = &v175[2 * v107];
            v109 = (_BYTE *)*v108;
            if ( v110 == (_BYTE *)*v108 )
            {
              v110 = (_BYTE *)v108[1];
              goto LABEL_136;
            }
            v115 = 1;
            v114 = 0;
            while ( 1 )
            {
              if ( v109 == (_BYTE *)-8LL )
              {
                if ( !v114 )
                  v114 = v108;
                ++v174;
                v112 = v176 + 1;
                if ( 4 * ((int)v176 + 1) < 3 * v177 )
                {
                  if ( v177 - HIDWORD(v176) - v112 > v177 >> 3 )
                    goto LABEL_144;
                  sub_1AD8C40((__int64)&v174, v177);
                  if ( v177 )
                  {
                    v36 = 0;
                    v116 = (v177 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
                    LODWORD(v43) = 1;
                    v112 = v176 + 1;
                    v114 = &v175[2 * v116];
                    v117 = (_BYTE *)*v114;
                    if ( v110 != (_BYTE *)*v114 )
                    {
                      while ( v117 != (_BYTE *)-8LL )
                      {
                        if ( v117 == (_BYTE *)-16LL && !v36 )
                          v36 = v114;
                        v157 = v43 + 1;
                        LODWORD(v43) = (v177 - 1) & (v116 + v43);
                        v116 = v43;
                        v114 = &v175[2 * (unsigned int)v43];
                        v117 = (_BYTE *)*v114;
                        if ( v110 == (_BYTE *)*v114 )
                          goto LABEL_144;
                        LODWORD(v43) = v157;
                      }
                      if ( v36 )
                        v114 = v36;
                    }
LABEL_144:
                    LODWORD(v176) = v112;
                    if ( *v114 != -8 )
                      --HIDWORD(v176);
                    *v114 = v110;
                    v110 = 0;
                    v114[1] = 0;
                    v111 = (unsigned int)v179;
                    if ( (unsigned int)v179 < HIDWORD(v179) )
                      goto LABEL_137;
LABEL_147:
                    sub_16CD150((__int64)&v178, &v180, 0, 8, (int)v36, v43);
                    v111 = (unsigned int)v179;
                    goto LABEL_137;
                  }
LABEL_319:
                  LODWORD(v176) = v176 + 1;
                  BUG();
                }
LABEL_142:
                sub_1AD8C40((__int64)&v174, 2 * v177);
                if ( v177 )
                {
                  v112 = v176 + 1;
                  LODWORD(v113) = (v177 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
                  v114 = &v175[2 * (unsigned int)v113];
                  v36 = (_QWORD *)*v114;
                  if ( v110 != (_BYTE *)*v114 )
                  {
                    v151 = 1;
                    v43 = 0;
                    while ( v36 != (_QWORD *)-8LL )
                    {
                      if ( v36 == (_QWORD *)-16LL && !v43 )
                        v43 = (__int64)v114;
                      v113 = (v177 - 1) & ((_DWORD)v113 + v151);
                      v114 = &v175[2 * v113];
                      v36 = (_QWORD *)*v114;
                      if ( v110 == (_BYTE *)*v114 )
                        goto LABEL_144;
                      ++v151;
                    }
                    if ( v43 )
                      v114 = (_QWORD *)v43;
                  }
                  goto LABEL_144;
                }
                goto LABEL_319;
              }
              if ( v109 != (_BYTE *)-16LL || v114 )
                v108 = v114;
              v107 = v43 & (v115 + v107);
              v144 = &v175[2 * v107];
              v109 = (_BYTE *)*v144;
              if ( v110 == (_BYTE *)*v144 )
                break;
              ++v115;
              v114 = v108;
              v108 = &v175[2 * v107];
            }
            v110 = (_BYTE *)v144[1];
LABEL_136:
            v111 = (unsigned int)v179;
            if ( (unsigned int)v179 >= HIDWORD(v179) )
              goto LABEL_147;
LABEL_137:
            ++v106;
            v178[v111] = (__int64)v110;
            LODWORD(v179) = v179 + 1;
            if ( v105 == v106 )
            {
              v59 = v178;
              v60 = (__int64 *)(unsigned int)v179;
              goto LABEL_59;
            }
            v104 = *(unsigned int *)(v103 + 8);
          }
          ++v174;
          goto LABEL_142;
        }
        v59 = &v180;
        v60 = 0;
LABEL_59:
        v61 = (__int64 *)sub_15E0530(v164);
        v62 = (unsigned __int8 *)sub_1627350(v61, v59, v60, 0, 1);
        if ( v177 )
        {
          LODWORD(v43) = v177 - 1;
          LODWORD(v36) = (_DWORD)v175;
          v65 = (v177 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
          v66 = &v175[2 * v65];
          v67 = *v66;
          if ( v103 == *v66 )
          {
            v68 = v66[1];
            goto LABEL_62;
          }
          v133 = 1;
          v134 = 0;
          while ( v67 != -8 )
          {
            if ( v67 != -16 || v134 )
              v66 = v134;
            v65 = v43 & (v133 + v65);
            v152 = &v175[2 * v65];
            v67 = *v152;
            if ( v103 == *v152 )
            {
              v68 = v152[1];
              goto LABEL_62;
            }
            ++v133;
            v134 = v66;
            v66 = &v175[2 * v65];
          }
          if ( !v134 )
            v134 = v66;
          ++v174;
          v135 = v176 + 1;
          if ( 4 * ((int)v176 + 1) < 3 * v177 )
          {
            if ( v177 - HIDWORD(v176) - v135 <= v177 >> 3 )
            {
              sub_1AD8C40((__int64)&v174, v177);
              if ( !v177 )
              {
LABEL_322:
                LODWORD(v176) = v176 + 1;
                BUG();
              }
              LODWORD(v43) = v177 - 1;
              v140 = 0;
              v141 = (v177 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
              v142 = 1;
              v135 = v176 + 1;
              v134 = &v175[2 * v141];
              v143 = *v134;
              if ( v103 != *v134 )
              {
                while ( v143 != -8 )
                {
                  if ( v143 == -16 && !v140 )
                    v140 = v134;
                  LODWORD(v36) = v142 + 1;
                  v141 = v43 & (v142 + v141);
                  v134 = &v175[2 * v141];
                  v143 = *v134;
                  if ( v103 == *v134 )
                    goto LABEL_201;
                  ++v142;
                }
                if ( v140 )
                  v134 = v140;
              }
            }
LABEL_201:
            LODWORD(v176) = v135;
            if ( *v134 != -8 )
              --HIDWORD(v176);
            *v134 = v103;
            v68 = 0;
            v134[1] = 0;
LABEL_62:
            v69 = *(_QWORD *)(v68 + 16);
            if ( (v69 & 4) != 0 )
              sub_16302D0((const __m128i *)(v69 & 0xFFFFFFFFFFFFFFF8LL), v62, a3, a4, a5, a6, v63, v64, a9, a10);
            if ( v178 != &v180 )
              _libc_free((unsigned __int64)v178);
            v168 += 8;
            if ( v161 == v168 )
              break;
            goto LABEL_132;
          }
        }
        else
        {
          ++v174;
        }
        sub_1AD8C40((__int64)&v174, 2 * v177);
        if ( !v177 )
          goto LABEL_322;
        LODWORD(v36) = (_DWORD)v175;
        v136 = (v177 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
        v135 = v176 + 1;
        v134 = &v175[2 * v136];
        v137 = *v134;
        if ( v103 != *v134 )
        {
          v138 = 1;
          v139 = 0;
          while ( v137 != -8 )
          {
            if ( v137 == -16 && !v139 )
              v139 = v134;
            LODWORD(v43) = v138 + 1;
            v136 = (v177 - 1) & (v138 + v136);
            v134 = &v175[2 * v136];
            v137 = *v134;
            if ( v103 == *v134 )
              goto LABEL_201;
            ++v138;
          }
          if ( v139 )
            v134 = v139;
        }
        goto LABEL_201;
      }
    }
  }
  v70 = *(_QWORD *)(a2 + 8);
  v71 = *(unsigned int *)(a2 + 16);
  v72 = v70 + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6);
  if ( !(_DWORD)v71 )
  {
    v71 = a2;
    v179 = *(_QWORD *)a2;
    v180 = v72;
    v178 = (__int64 *)a2;
    v181 = v72;
    goto LABEL_69;
  }
  v179 = *(_QWORD *)a2;
  v180 = v70;
  v178 = (__int64 *)a2;
  v181 = v72;
  sub_1AD5930((__int64)&v178);
  v84 = v181;
  v85 = *(_QWORD *)(a2 + 8) + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6);
  v86 = v180;
  if ( v180 != v85 )
  {
    while ( 2 )
    {
      v87 = *(_QWORD *)(v86 + 56);
      if ( !v87 || *(_BYTE *)(v87 + 16) <= 0x17u )
        goto LABEL_109;
      if ( !*(_QWORD *)(v87 + 48) && *(__int16 *)(v87 + 18) >= 0 )
      {
        v172 = 0;
LABEL_116:
        if ( (unsigned __int8)sub_15F2ED0(v87) )
        {
          if ( *(_QWORD *)(v158 + 48) )
            goto LABEL_118;
LABEL_210:
          if ( *(__int16 *)(v158 + 18) < 0 )
          {
LABEL_118:
            v71 = 7;
            v93 = sub_1625790(v158, 7);
            if ( !v93 )
            {
              if ( !*(_QWORD *)(v87 + 48) )
              {
LABEL_120:
                if ( *(__int16 *)(v87 + 18) >= 0 )
                {
                  v172 = 0;
LABEL_122:
                  if ( (unsigned __int8)sub_15F2ED0(v87) )
                  {
                    if ( *(_QWORD *)(v158 + 48) )
                      goto LABEL_124;
LABEL_206:
                    if ( *(__int16 *)(v158 + 18) < 0 )
                    {
LABEL_124:
                      v71 = 8;
                      v100 = sub_1625790(v158, 8);
                      if ( !v100 )
                        goto LABEL_109;
LABEL_108:
                      v71 = 8;
                      sub_1625C10(v87, 8, v100);
                    }
                  }
                  else if ( (unsigned __int8)sub_15F3040(v87) )
                  {
                    if ( !*(_QWORD *)(v158 + 48) )
                      goto LABEL_206;
                    goto LABEL_124;
                  }
                  do
                  {
LABEL_109:
                    v86 += 64;
                    if ( v86 == v84 )
                      break;
                    v102 = *(_QWORD *)(v86 + 24);
                  }
                  while ( v102 == -8 || v102 == -16 );
                  if ( v86 == v85 )
                    goto LABEL_69;
                  continue;
                }
              }
LABEL_101:
              v71 = 8;
              v95 = sub_1625790(v87, 8);
              v172 = v95;
              if ( !v95 )
                goto LABEL_122;
              v96 = v177;
              if ( v177 )
              {
                v97 = (v177 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
                v98 = &v175[2 * v97];
                v99 = *v98;
                if ( v95 == *v98 )
                {
                  v100 = v98[1];
LABEL_105:
                  if ( *(_QWORD *)(v158 + 48) || *(__int16 *)(v158 + 18) < 0 )
                  {
                    v171 = v100;
                    v101 = sub_1625790(v158, 8);
                    v100 = v171;
                    if ( v101 )
                      v100 = sub_1631960(v171, v101);
                  }
                  goto LABEL_108;
                }
                v148 = 1;
                v149 = 0;
                while ( v99 != -8 )
                {
                  if ( v99 != -16 || v149 )
                    v98 = v149;
                  v155 = v148 + 1;
                  v97 = (v177 - 1) & (v148 + v97);
                  v156 = &v175[2 * v97];
                  v99 = *v156;
                  if ( v95 == *v156 )
                  {
                    v100 = v156[1];
                    goto LABEL_105;
                  }
                  v148 = v155;
                  v149 = v98;
                  v98 = &v175[2 * v97];
                }
                if ( !v149 )
                  v149 = v98;
                ++v174;
                v150 = v176 + 1;
                if ( 4 * ((int)v176 + 1) < 3 * v177 )
                {
                  if ( v177 - HIDWORD(v176) - v150 <= v177 >> 3 )
                    goto LABEL_254;
LABEL_249:
                  LODWORD(v176) = v150;
                  if ( *v149 != -8 )
                    --HIDWORD(v176);
                  *v149 = v95;
                  v100 = 0;
                  v149[1] = 0;
                  goto LABEL_105;
                }
              }
              else
              {
                ++v174;
              }
              v96 = 2 * v177;
LABEL_254:
              sub_1AD8C40((__int64)&v174, v96);
              sub_1AD5BF0((__int64)&v174, &v172, &v173);
              v149 = v173;
              v95 = v172;
              v150 = v176 + 1;
              goto LABEL_249;
            }
LABEL_99:
            v71 = 7;
            sub_1625C10(v87, 7, v93);
          }
        }
        else if ( (unsigned __int8)sub_15F3040(v87) )
        {
          if ( !*(_QWORD *)(v158 + 48) )
            goto LABEL_210;
          goto LABEL_118;
        }
        if ( !*(_QWORD *)(v87 + 48) )
          goto LABEL_120;
        goto LABEL_101;
      }
      break;
    }
    v71 = 7;
    v88 = sub_1625790(*(_QWORD *)(v86 + 56), 7);
    v172 = v88;
    if ( !v88 )
      goto LABEL_116;
    v89 = v177;
    if ( v177 )
    {
      v90 = (v177 - 1) & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
      v91 = &v175[2 * v90];
      v92 = *v91;
      if ( *v91 == v88 )
      {
        v93 = v91[1];
LABEL_96:
        if ( *(_QWORD *)(v158 + 48) || *(__int16 *)(v158 + 18) < 0 )
        {
          v170 = v93;
          v94 = sub_1625790(v158, 7);
          v93 = v170;
          if ( v94 )
            v93 = sub_1631960(v170, v94);
        }
        goto LABEL_99;
      }
      v145 = 1;
      v146 = 0;
      while ( v92 != -8 )
      {
        if ( v92 != -16 || v146 )
          v91 = v146;
        v153 = v145 + 1;
        v90 = (v177 - 1) & (v145 + v90);
        v154 = &v175[2 * v90];
        v92 = *v154;
        if ( v88 == *v154 )
        {
          v93 = v154[1];
          goto LABEL_96;
        }
        v145 = v153;
        v146 = v91;
        v91 = &v175[2 * v90];
      }
      if ( !v146 )
        v146 = v91;
      ++v174;
      v147 = v176 + 1;
      if ( 4 * ((int)v176 + 1) < 3 * v177 )
      {
        if ( v177 - HIDWORD(v176) - v147 <= v177 >> 3 )
          goto LABEL_242;
LABEL_237:
        LODWORD(v176) = v147;
        if ( *v146 != -8 )
          --HIDWORD(v176);
        *v146 = v88;
        v93 = 0;
        v146[1] = 0;
        goto LABEL_96;
      }
    }
    else
    {
      ++v174;
    }
    v89 = 2 * v177;
LABEL_242:
    sub_1AD8C40((__int64)&v174, v89);
    sub_1AD5BF0((__int64)&v174, &v172, &v173);
    v146 = v173;
    v88 = v172;
    v147 = v176 + 1;
    goto LABEL_237;
  }
LABEL_69:
  if ( v177 )
  {
    v73 = v175;
    v74 = &v175[2 * v177];
    do
    {
      if ( *v73 != -8 && *v73 != -16 )
      {
        v71 = v73[1];
        if ( v71 )
          sub_161E7C0((__int64)(v73 + 1), v71);
      }
      v73 += 2;
    }
    while ( v74 != v73 );
  }
  j___libc_free_0(v175);
  v80 = (__int64)v192;
  v81 = &v192[8 * (unsigned int)v193];
  if ( v192 != v81 )
  {
    do
    {
      v82 = *((_QWORD *)v81 - 1);
      v81 -= 8;
      if ( v82 )
        sub_16307F0(v82, v71, v75, v76, v77, a3, a4, a5, a6, v78, v79, a9, a10);
    }
    while ( (_BYTE *)v80 != v81 );
    v81 = v192;
  }
  if ( v81 != v194 )
    _libc_free((unsigned __int64)v81);
  if ( v189 != v191 )
    _libc_free((unsigned __int64)v189);
  if ( v186 )
    j_j___libc_free_0(v186, v188 - (_QWORD)v186);
  return j___libc_free_0(v183);
}
