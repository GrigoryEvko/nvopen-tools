// Function: sub_1DAEBC0
// Address: 0x1daebc0
//
__int64 __fastcall sub_1DAEBC0(__int64 a1)
{
  __int64 v1; // rsi
  unsigned __int64 k; // rdx
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  _DWORD *v17; // rsi
  __int64 *v18; // rax
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rax
  int v26; // r14d
  __int64 v27; // rax
  int v28; // eax
  _BYTE *v29; // rax
  unsigned int v30; // r13d
  __int64 v31; // rsi
  unsigned int v32; // r12d
  __int64 v33; // rcx
  __int64 v34; // rbx
  signed __int64 v35; // r12
  unsigned __int64 *v36; // r14
  char *v37; // rdi
  __int64 v38; // rax
  unsigned __int64 *v39; // r13
  unsigned __int64 v40; // rdx
  __int64 i; // rcx
  unsigned int v42; // ecx
  __int64 *v43; // rax
  __int64 v44; // r11
  unsigned __int64 j; // rdx
  __int64 *v46; // rax
  __int64 v47; // r11
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // r9
  int v52; // ecx
  __int64 v53; // r11
  __int64 *v54; // r10
  __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rax
  char v58; // r12
  unsigned __int64 v59; // rbx
  __int64 v60; // r12
  __int64 *v61; // rcx
  __int64 v62; // r9
  __int64 v63; // rax
  char *v64; // rax
  __int64 v65; // rcx
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // rbx
  __int64 v69; // r13
  __int64 v70; // rax
  _QWORD *v71; // rbx
  _QWORD *v72; // r12
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rdi
  __int64 result; // rax
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rbx
  __int64 *v79; // rax
  __int64 v80; // r8
  __int64 v81; // r8
  unsigned __int64 v82; // r9
  int v83; // eax
  __int64 v84; // rbx
  __int64 v85; // rax
  int v86; // r15d
  __int64 v87; // rbx
  __int64 v88; // r14
  unsigned __int64 v89; // r15
  __int64 v90; // r13
  __int64 v91; // rbx
  __int64 v92; // r12
  __int64 *v93; // rax
  __int64 v94; // rcx
  __int64 v95; // rax
  int v96; // eax
  int v97; // eax
  unsigned __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // rsi
  unsigned int v101; // ecx
  unsigned int v102; // edi
  unsigned __int64 *v103; // rax
  unsigned __int64 v104; // rcx
  unsigned __int64 v105; // rcx
  __int64 v106; // rdx
  __int64 v107; // rdi
  __int64 *v108; // rsi
  __int64 v109; // rdx
  __int64 v110; // rax
  __int64 *v111; // rdi
  __int64 v112; // rax
  __int64 v113; // rsi
  __int64 v114; // rax
  _DWORD *v115; // rax
  int v116; // eax
  __int64 v117; // r15
  __int64 v118; // r15
  __int64 *v119; // rdx
  __int64 v120; // rax
  __int64 *v121; // rax
  __int64 v122; // rsi
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // rsi
  __int64 *v126; // rdi
  unsigned __int64 v127; // rdx
  __int64 v128; // rax
  __int64 v129; // rdx
  unsigned int v130; // eax
  int v131; // ebx
  unsigned __int64 v132; // rax
  __int64 v133; // rdx
  int v134; // r8d
  int v135; // r9d
  unsigned __int64 v136; // rbx
  __int64 v137; // rax
  __int64 *v138; // rax
  unsigned int v139; // ecx
  __int64 v140; // r15
  unsigned int v141; // ecx
  signed __int64 *v142; // rax
  __int64 v143; // rax
  unsigned __int64 v144; // rsi
  __int64 v145; // rax
  int v146; // edx
  int v147; // eax
  _QWORD *v148; // rdi
  _QWORD *v149; // rcx
  _QWORD *v150; // rax
  __int64 v151; // rsi
  unsigned int v152; // eax
  int v153; // r10d
  __int64 v154; // rax
  int v155; // r10d
  __int64 v156; // rax
  int v157; // r10d
  __int64 v158; // rax
  char *v159; // rax
  __int64 v160; // [rsp+28h] [rbp-458h]
  __int64 v161; // [rsp+38h] [rbp-448h]
  int v162; // [rsp+40h] [rbp-440h]
  unsigned __int8 v163; // [rsp+47h] [rbp-439h]
  __int64 v164; // [rsp+58h] [rbp-428h]
  unsigned __int64 v165; // [rsp+78h] [rbp-408h]
  __int64 v167; // [rsp+98h] [rbp-3E8h]
  unsigned __int64 v168; // [rsp+98h] [rbp-3E8h]
  unsigned __int64 v169; // [rsp+98h] [rbp-3E8h]
  __int64 v170; // [rsp+98h] [rbp-3E8h]
  __int64 v171; // [rsp+A0h] [rbp-3E0h]
  __int64 v172; // [rsp+A8h] [rbp-3D8h]
  __int64 v173; // [rsp+B8h] [rbp-3C8h]
  unsigned int v174; // [rsp+C0h] [rbp-3C0h]
  unsigned int v175; // [rsp+C0h] [rbp-3C0h]
  unsigned int v176; // [rsp+C0h] [rbp-3C0h]
  __int64 v177; // [rsp+C8h] [rbp-3B8h]
  __int64 v178; // [rsp+D0h] [rbp-3B0h] BYREF
  __int64 *v179; // [rsp+D8h] [rbp-3A8h]
  __int64 v180; // [rsp+E0h] [rbp-3A0h]
  __int64 v181; // [rsp+E8h] [rbp-398h] BYREF
  unsigned __int64 v182; // [rsp+F0h] [rbp-390h]
  __int64 v183; // [rsp+130h] [rbp-350h] BYREF
  __int64 v184; // [rsp+138h] [rbp-348h] BYREF
  __int64 v185; // [rsp+140h] [rbp-340h] BYREF
  _BYTE v186[120]; // [rsp+148h] [rbp-338h] BYREF
  __int64 *v187; // [rsp+1C0h] [rbp-2C0h] BYREF
  unsigned __int64 v188; // [rsp+1C8h] [rbp-2B8h] BYREF
  __int64 v189; // [rsp+1D0h] [rbp-2B0h] BYREF
  _BYTE v190[120]; // [rsp+1D8h] [rbp-2A8h] BYREF
  __int64 v191; // [rsp+250h] [rbp-230h] BYREF
  __int64 *v192; // [rsp+258h] [rbp-228h] BYREF
  __int64 v193; // [rsp+260h] [rbp-220h]
  __int64 v194; // [rsp+268h] [rbp-218h]
  __int64 v195; // [rsp+270h] [rbp-210h]
  int v196; // [rsp+278h] [rbp-208h]
  __int64 v197; // [rsp+280h] [rbp-200h]
  __int64 v198; // [rsp+288h] [rbp-1F8h] BYREF
  void *s; // [rsp+290h] [rbp-1F0h]
  __int64 v200; // [rsp+298h] [rbp-1E8h]
  _QWORD *v201; // [rsp+2A0h] [rbp-1E0h]
  __int64 v202; // [rsp+2A8h] [rbp-1D8h]
  int v203; // [rsp+2B0h] [rbp-1D0h]
  __int64 v204; // [rsp+2B8h] [rbp-1C8h]
  __int64 v205; // [rsp+2C0h] [rbp-1C0h] BYREF
  __int64 *v206; // [rsp+2C8h] [rbp-1B8h] BYREF
  __int64 v207; // [rsp+2D0h] [rbp-1B0h]
  __int64 v208; // [rsp+2D8h] [rbp-1A8h]
  __int64 v209; // [rsp+2E0h] [rbp-1A0h]
  int v210; // [rsp+2E8h] [rbp-198h]
  __int64 v211; // [rsp+2F0h] [rbp-190h]
  __int64 v212; // [rsp+2F8h] [rbp-188h] BYREF
  _BYTE *v213; // [rsp+300h] [rbp-180h]
  __int64 v214; // [rsp+308h] [rbp-178h]
  _BYTE v215[32]; // [rsp+310h] [rbp-170h] BYREF
  __int64 v216; // [rsp+330h] [rbp-150h]
  _BYTE *v217; // [rsp+340h] [rbp-140h] BYREF
  __int64 v218; // [rsp+348h] [rbp-138h]
  _BYTE v219[304]; // [rsp+350h] [rbp-130h] BYREF

  v192 = &v198;
  v206 = &v212;
  s = &v205;
  v196 = 1065353216;
  v203 = 1065353216;
  v191 = 0;
  v193 = 1;
  v194 = 0;
  v195 = 0;
  v197 = 0;
  v198 = 0;
  v200 = 1;
  v201 = 0;
  v202 = 0;
  v204 = 0;
  v205 = 0;
  v207 = 1;
  v208 = 0;
  v209 = 0;
  v210 = 1065353216;
  v214 = 0x400000000LL;
  v1 = *(_QWORD *)(a1 + 120);
  v211 = 0;
  v212 = 0;
  v213 = v215;
  v216 = 0;
  sub_20FC0D0(&v191, v1);
  v5 = *(unsigned int *)(a1 + 160);
  if ( !(_DWORD)v5 )
    goto LABEL_74;
  v172 = 0;
  v160 = 8 * v5;
  do
  {
    v173 = *(_QWORD *)(a1 + 128);
    v177 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + v172);
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 120) + 40LL);
    v189 = 0x400000000LL;
    v161 = v6;
    v217 = v219;
    v218 = 0x1000000000LL;
    v187 = (__int64 *)(v177 + 216);
    v188 = (unsigned __int64)v190;
    sub_1DA9720((__int64 *)&v187, v1, k, a1, v3, v4);
    v185 = 0x400000000LL;
    v183 = (__int64)v187;
    v184 = (__int64)v186;
    if ( (_DWORD)v189 )
      sub_1DA8090((__int64)&v184, (char **)&v188, v7, v8, v9, v10);
    if ( (_BYTE *)v188 != v190 )
      _libc_free(v188);
    v11 = (unsigned int)v185;
    v12 = v184;
    if ( (_DWORD)v185 )
    {
      do
      {
        if ( *(_DWORD *)(v12 + 12) >= *(_DWORD *)(v12 + 8) )
          break;
        v13 = v12 + 16 * v11 - 16;
        v14 = *(_QWORD *)v13;
        v15 = *(unsigned int *)(v13 + 12);
        v16 = v15 + 36;
        if ( !*(_DWORD *)(v183 + 80) )
          v16 = v15 + 16;
        v17 = (_DWORD *)(v14 + 4 * v16);
        v18 = (__int64 *)(16 * v15 + v14);
        if ( (*v17 & 0x7FFFFFFF) != 0x7FFFFFFF )
        {
          v19 = *v18;
          v20 = (unsigned int)*v17;
          v21 = (unsigned int)v218;
          if ( (unsigned int)v218 >= HIDWORD(v218) )
          {
            sub_16CD150((__int64)&v217, v219, 0, 16, v9, v10);
            v21 = (unsigned int)v218;
          }
          v22 = &v217[16 * v21];
          *v22 = v19;
          v22[1] = v20;
          v12 = v184;
          LODWORD(v218) = v218 + 1;
        }
        v23 = v12 + 16LL * (unsigned int)v185 - 16;
        v24 = *(_DWORD *)(v23 + 12) + 1;
        *(_DWORD *)(v23 + 12) = v24;
        v11 = (unsigned int)v185;
        v12 = v184;
        if ( v24 == *(_DWORD *)(v184 + 16LL * (unsigned int)v185 - 8) )
        {
          v122 = *(unsigned int *)(v183 + 80);
          if ( (_DWORD)v122 )
          {
            sub_39460A0(&v184, v122);
            v11 = (unsigned int)v185;
            v12 = v184;
          }
        }
      }
      while ( (_DWORD)v11 );
    }
    if ( (_BYTE *)v12 != v186 )
      _libc_free(v12);
    v25 = 0;
    v26 = 0;
    if ( (_DWORD)v218 )
    {
      while ( 1 )
      {
        v29 = &v217[16 * v25];
        v30 = *((_DWORD *)v29 + 2);
        v31 = *(_QWORD *)v29;
        v32 = v30 & 0x7FFFFFFF;
        if ( (v30 & 0x7FFFFFFF) == 0x7FFFFFFF )
        {
          v32 = -1;
          v27 = *(_QWORD *)(v177 + 40) + 0x27FFFFFFD8LL;
          if ( *(_BYTE *)v27 )
            goto LABEL_26;
LABEL_22:
          v28 = *(_DWORD *)(v27 + 8);
          if ( v28 < 0 )
          {
            v76 = v28 & 0x7FFFFFFF;
            if ( (unsigned int)v76 >= *(_DWORD *)(v173 + 408)
              || (v77 = *(_QWORD *)(v173 + 400), (v78 = *(_QWORD *)(v77 + 8 * v76)) == 0) )
            {
              v183 = (__int64)&v185;
              v184 = 0x1000000000LL;
              sub_1DAD4A0(v177, v31, v30, 0, 0, (__int64)&v183, v173);
              goto LABEL_91;
            }
            v79 = (__int64 *)sub_1DB3C70(*(_QWORD *)(v77 + 8 * v76), v31);
            v80 = 0;
            if ( v79 != (__int64 *)(*(_QWORD *)v78 + 24LL * *(unsigned int *)(v78 + 8))
              && (*(_DWORD *)((*v79 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v79 >> 1) & 3) <= (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v31 >> 1) & 3) )
            {
              v80 = v79[2];
            }
            v183 = (__int64)&v185;
            v184 = 0x1000000000LL;
            sub_1DAD4A0(v177, v31, v30, v78, v80, (__int64)&v183, v173);
            if ( (_DWORD)v184 )
            {
              v83 = *(_DWORD *)(v78 + 112);
              if ( v83 < 0 )
              {
                v187 = &v189;
                v188 = 0x800000000LL;
                v84 = *(_QWORD *)(*(_QWORD *)(v161 + 24) + 16LL * (v83 & 0x7FFFFFFF) + 8);
                if ( v84 )
                {
                  while ( (*(_BYTE *)(v84 + 3) & 0x10) != 0 || (*(_BYTE *)(v84 + 4) & 8) != 0 )
                  {
                    v84 = *(_QWORD *)(v84 + 32);
                    if ( !v84 )
                      goto LABEL_91;
                  }
LABEL_104:
                  if ( (*(_DWORD *)v84 & 0xFFF00) != 0
                    || (v85 = *(_QWORD *)(v84 + 16), **(_WORD **)(v85 + 16) != 15)
                    || (v86 = *(_DWORD *)(*(_QWORD *)(v85 + 32) + 8LL), v86 >= 0) )
                  {
LABEL_107:
                    while ( 1 )
                    {
                      v84 = *(_QWORD *)(v84 + 32);
                      if ( !v84 )
                        break;
                      while ( (*(_BYTE *)(v84 + 3) & 0x10) == 0 )
                      {
                        if ( (*(_BYTE *)(v84 + 4) & 8) == 0 )
                          goto LABEL_104;
                        v84 = *(_QWORD *)(v84 + 32);
                        if ( !v84 )
                          goto LABEL_111;
                      }
                    }
LABEL_111:
                    v87 = (unsigned int)v188;
                    if ( !(_DWORD)v188 || !(_DWORD)v184 )
                      goto LABEL_172;
                    v171 = 0;
                    v162 = v26;
                    v164 = 8LL * (unsigned int)(v184 - 1);
                    v163 = HIBYTE(v30) >> 7;
LABEL_114:
                    v88 = *(_QWORD *)(v183 + v171);
                    if ( !(_DWORD)v87 )
                    {
LABEL_121:
                      v95 = v171;
                      if ( v164 == v171 )
                        goto LABEL_171;
                      goto LABEL_122;
                    }
                    v167 = (v88 >> 1) & 3;
                    v174 = (v88 >> 1) & 3;
                    v89 = 0;
                    v90 = 16 * v87;
                    while ( 1 )
                    {
                      v91 = v187[v89 / 8];
                      v92 = v187[v89 / 8 + 1];
                      v93 = (__int64 *)sub_1DB3C70(v91, v88);
                      v94 = 0;
                      if ( v93 != (__int64 *)(*(_QWORD *)v91 + 24LL * *(unsigned int *)(v91 + 8))
                        && (*(_DWORD *)((*v93 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v93 >> 1) & 3) <= (*(_DWORD *)((v88 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v174) )
                      {
                        v94 = v93[2];
                      }
                      if ( v92 == v94 )
                      {
                        v179 = &v181;
                        v178 = v177 + 216;
                        v180 = 0x400000000LL;
                        v123 = *(unsigned int *)(v177 + 296);
                        if ( (_DWORD)v123 )
                        {
                          sub_1DAAC30((__int64)&v178, v88, v123, v94, (__int64)&v181, (int)&v178);
                          v128 = (unsigned int)v180;
                          if ( !(_DWORD)v180 )
                            goto LABEL_164;
                          v126 = v179;
                        }
                        else
                        {
                          v124 = v177;
                          v125 = *(unsigned int *)(v177 + 300);
                          if ( (_DWORD)v125 )
                          {
                            v124 = v177 + 224;
                            do
                            {
                              if ( (*(_DWORD *)((*(_QWORD *)v124 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                  | (unsigned int)(*(__int64 *)v124 >> 1) & 3) > (*(_DWORD *)((v88 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                            + 24)
                                                                                | v174) )
                                break;
                              v123 = (unsigned int)(v123 + 1);
                              v124 += 16;
                            }
                            while ( (_DWORD)v125 != (_DWORD)v123 );
                          }
                          v126 = &v181;
                          LODWORD(v180) = 1;
                          v127 = v125 | (v123 << 32);
                          v182 = v127;
                          v181 = v177 + 216;
                          v128 = 1;
                        }
                        if ( *((_DWORD *)v126 + 3) >= *((_DWORD *)v126 + 2)
                          || (v129 = *(_QWORD *)(v126[2 * v128 - 2] + 16LL * HIDWORD(v126[2 * v128 - 1])),
                              v130 = *(_DWORD *)((v129 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v129 >> 1) & 3,
                              v127 = v88 & 0xFFFFFFFFFFFFFFF8LL,
                              v124 = *(_DWORD *)((v88 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v174,
                              v130 > (unsigned int)v124) )
                        {
LABEL_164:
                          if ( (*(_QWORD *)(v92 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                            BUG();
                          v131 = sub_1DA81D0(
                                   v177,
                                   *(const __m128i **)(*(_QWORD *)((*(_QWORD *)(v92 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16)
                                                     + 32LL),
                                   v127,
                                   v124,
                                   (int)&v181,
                                   (int)&v178)
                               & 0x7FFFFFFF;
                          v132 = v88 & 0xFFFFFFFFFFFFFFF8LL;
                          if ( v167 == 3 )
                            v133 = *(_QWORD *)(v132 + 8) & 0xFFFFFFFFFFFFFFF9LL;
                          else
                            v133 = v132 | (2 * v167 + 2);
                          sub_1DAD0A0((__int64)&v178, v88, v133, v131 | (v163 << 31));
                          v136 = ((unsigned __int64)(v163 & 1) << 31) | v131 & 0x7FFFFFFF | v165 & 0xFFFFFFFF00000000LL;
                          v137 = (unsigned int)v218;
                          v165 = v136;
                          if ( (unsigned int)v218 >= HIDWORD(v218) )
                          {
                            sub_16CD150((__int64)&v217, v219, 0, 16, v134, v135);
                            v137 = (unsigned int)v218;
                          }
                          v138 = (__int64 *)&v217[16 * v137];
                          *v138 = v88;
                          v138[1] = v136;
                          LODWORD(v218) = v218 + 1;
                          if ( v179 == &v181 )
                            goto LABEL_121;
                          _libc_free((unsigned __int64)v179);
                          v95 = v171;
                          if ( v164 == v171 )
                          {
LABEL_171:
                            v26 = v162;
LABEL_172:
                            if ( v187 != &v189 )
                              _libc_free((unsigned __int64)v187);
                            goto LABEL_91;
                          }
LABEL_122:
                          v87 = (unsigned int)v188;
                          v171 = v95 + 8;
                          goto LABEL_114;
                        }
                        if ( v126 != &v181 )
                          _libc_free((unsigned __int64)v126);
                      }
                      v89 += 16LL;
                      if ( v90 == v89 )
                        goto LABEL_121;
                    }
                  }
                  v98 = *(_QWORD *)(v84 + 16);
                  v99 = *(_QWORD *)(v173 + 272);
                  if ( (*(_BYTE *)(v85 + 46) & 4) != 0 )
                  {
                    do
                      v98 = *(_QWORD *)v98 & 0xFFFFFFFFFFFFFFF8LL;
                    while ( (*(_BYTE *)(v98 + 46) & 4) != 0 );
                  }
                  v100 = *(_QWORD *)(v99 + 368);
                  v101 = *(_DWORD *)(v99 + 384);
                  if ( v101 )
                  {
                    v81 = v101 - 1;
                    v102 = v81 & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
                    v103 = (unsigned __int64 *)(v100 + 16LL * v102);
                    v82 = *v103;
                    if ( *v103 == v98 )
                      goto LABEL_133;
                    v147 = 1;
                    while ( v82 != -8 )
                    {
                      v157 = v147 + 1;
                      v158 = (unsigned int)v81 & (v102 + v147);
                      v102 = v158;
                      v103 = (unsigned __int64 *)(v100 + 16 * v158);
                      v82 = *v103;
                      if ( *v103 == v98 )
                        goto LABEL_133;
                      v147 = v157;
                    }
                  }
                  v103 = (unsigned __int64 *)(v100 + 16LL * v101);
LABEL_133:
                  v104 = v103[1];
                  v179 = &v181;
                  v178 = v177 + 216;
                  v105 = v104 & 0xFFFFFFFFFFFFFFF8LL;
                  v180 = 0x400000000LL;
                  v106 = *(unsigned int *)(v177 + 296);
                  if ( (_DWORD)v106 )
                  {
                    v169 = v105;
                    sub_1DAAC30((__int64)&v178, v105 | 2, v106, v105, v81, v82);
                    v110 = (unsigned int)v180;
                    v111 = v179;
                    v105 = v169;
                    if ( (_DWORD)v180 && *((_DWORD *)v179 + 3) < *((_DWORD *)v179 + 2) )
                      goto LABEL_140;
                  }
                  else
                  {
                    v107 = *(unsigned int *)(v177 + 300);
                    if ( (_DWORD)v107 )
                    {
                      v108 = (__int64 *)(v177 + 224);
                      v81 = *(_DWORD *)(v105 + 24) | 1u;
                      do
                      {
                        v82 = *v108 & 0xFFFFFFFFFFFFFFF8LL;
                        if ( (*(_DWORD *)(v82 + 24) | (unsigned int)(*v108 >> 1) & 3) > (unsigned int)v81 )
                          break;
                        v106 = (unsigned int)(v106 + 1);
                        v108 += 2;
                      }
                      while ( (_DWORD)v107 != (_DWORD)v106 );
                    }
                    v109 = v106 << 32;
                    LODWORD(v180) = 1;
                    v181 = v177 + 216;
                    v182 = v109 | v107;
                    if ( HIDWORD(v109) >= ((unsigned int)v109 | (unsigned int)v107) )
                      goto LABEL_107;
                    v110 = 1;
                    v111 = &v181;
LABEL_140:
                    v112 = (__int64)&v111[2 * v110 - 2];
                    v113 = *(_QWORD *)v112;
                    v114 = *(unsigned int *)(v112 + 12);
                    if ( *(_DWORD *)(v178 + 80) )
                      v115 = (_DWORD *)(v113 + 4 * v114 + 144);
                    else
                      v115 = (_DWORD *)(v113 + 4 * v114 + 64);
                    v116 = *v115 & 0x7FFFFFFF;
                    if ( v116 == 0x7FFFFFFF )
                      v116 = -1;
                    if ( v32 == v116 )
                    {
                      v117 = v86 & 0x7FFFFFFF;
                      if ( (unsigned int)v117 < *(_DWORD *)(v173 + 408) )
                      {
                        v118 = *(_QWORD *)(*(_QWORD *)(v173 + 400) + 8 * v117);
                        if ( v118 )
                        {
                          v168 = v105;
                          v119 = (__int64 *)sub_1DB3C70(v118, v105 | 4);
                          if ( v119 == (__int64 *)(*(_QWORD *)v118 + 24LL * *(unsigned int *)(v118 + 8))
                            || (*(_DWORD *)((*v119 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v119 >> 1) & 3) > (*(_DWORD *)(v168 + 24) | 2u) )
                          {
                            v81 = 0;
                          }
                          else
                          {
                            v81 = v119[2];
                          }
                          v120 = (unsigned int)v188;
                          if ( (unsigned int)v188 >= HIDWORD(v188) )
                          {
                            v170 = v81;
                            sub_16CD150((__int64)&v187, &v189, 0, 16, v81, v82);
                            v120 = (unsigned int)v188;
                            v81 = v170;
                          }
                          v121 = &v187[2 * v120];
                          *v121 = v118;
                          v121[1] = v81;
                          v111 = v179;
                          LODWORD(v188) = v188 + 1;
                          if ( v179 == &v181 )
                            goto LABEL_107;
LABEL_189:
                          _libc_free((unsigned __int64)v111);
                          goto LABEL_107;
                        }
                      }
                    }
                  }
                  if ( v111 == &v181 )
                    goto LABEL_107;
                  goto LABEL_189;
                }
              }
            }
LABEL_91:
            if ( (__int64 *)v183 != &v185 )
              _libc_free(v183);
          }
          v25 = (unsigned int)(v26 + 1);
          v26 = v25;
          if ( (_DWORD)v218 == (_DWORD)v25 )
            break;
        }
        else
        {
          v27 = *(_QWORD *)(v177 + 40) + 40LL * v32;
          if ( !*(_BYTE *)v27 )
            goto LABEL_22;
LABEL_26:
          sub_1DAD4A0(v177, v31, v30, 0, 0, 0, v173);
          v25 = (unsigned int)(v26 + 1);
          v26 = v25;
          if ( (_DWORD)v218 == (_DWORD)v25 )
            break;
        }
      }
    }
    v1 = sub_15C70A0(v177 + 16);
    v34 = sub_20FAEB0(&v191, v1);
    if ( !v34 )
      goto LABEL_65;
    v35 = 0;
    v187 = (__int64 *)(v177 + 216);
    v188 = (unsigned __int64)v190;
    v189 = 0x400000000LL;
    sub_1DA9720((__int64 *)&v187, v1, k, v33, v3, v4);
    v36 = *(unsigned __int64 **)(v34 + 80);
    v37 = (char *)v188;
    v38 = 2LL * *(unsigned int *)(v34 + 88);
    v39 = &v36[v38];
    if ( v36 == &v36[v38] )
      goto LABEL_63;
    do
    {
      v40 = *v36;
      for ( i = *(_QWORD *)(v173 + 272); (*(_BYTE *)(v40 + 46) & 4) != 0; v40 = *(_QWORD *)v40 & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v1 = *(_QWORD *)(i + 368);
      v42 = *(_DWORD *)(i + 384);
      if ( v42 )
      {
        LODWORD(v3) = (v42 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v43 = (__int64 *)(v1 + 16LL * (unsigned int)v3);
        v44 = *v43;
        if ( v40 == *v43 )
          goto LABEL_33;
        v97 = 1;
        while ( v44 != -8 )
        {
          v153 = v97 + 1;
          v154 = (v42 - 1) & ((_DWORD)v3 + v97);
          LODWORD(v3) = v154;
          v43 = (__int64 *)(v1 + 16 * v154);
          v44 = *v43;
          if ( v40 == *v43 )
            goto LABEL_33;
          v97 = v153;
        }
      }
      v43 = (__int64 *)(v1 + 16LL * v42);
LABEL_33:
      v183 = v43[1];
      for ( j = v36[1]; (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
        ;
      if ( v42 )
      {
        LODWORD(v3) = (v42 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
        v46 = (__int64 *)(v1 + 16LL * (unsigned int)v3);
        v47 = *v46;
        if ( *v46 == j )
          goto LABEL_37;
        v96 = 1;
        while ( v47 != -8 )
        {
          v155 = v96 + 1;
          v156 = (v42 - 1) & ((_DWORD)v3 + v96);
          LODWORD(v3) = v156;
          v46 = (__int64 *)(v1 + 16 * v156);
          v47 = *v46;
          if ( *v46 == j )
            goto LABEL_37;
          v96 = v155;
        }
      }
      v46 = (__int64 *)(v1 + 16LL * v42);
LABEL_37:
      v48 = v46[1];
      v49 = (unsigned int)v189;
      k = v35 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v35 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_40;
      v50 = (__int64)&v37[16 * (unsigned int)v189 - 16];
      v1 = (v35 >> 1) & 3;
      v3 = *(_QWORD *)v50;
      v51 = *(unsigned int *)(v50 + 12);
      v52 = *(_DWORD *)(k + 24);
      v53 = 16 * v51;
      v54 = (__int64 *)(v3 + 16 * v51);
      v55 = *v54;
      if ( *((_DWORD *)v187 + 20) )
      {
        k = *(_DWORD *)((v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v55 >> 1) & 3;
        if ( (unsigned int)k >= ((unsigned int)v1 | v52) )
          goto LABEL_40;
        v140 = v54[1];
        v141 = *(_DWORD *)(v3 + 4 * v51 + 144);
        v142 = (signed __int64 *)(v3 + v53 + 8);
      }
      else
      {
        v139 = v1 | v52;
        v1 = v55 & 0xFFFFFFFFFFFFFFF8LL;
        k = *(_DWORD *)((v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v55 >> 1) & 3;
        if ( v139 <= (unsigned int)k )
          goto LABEL_40;
        v140 = v54[1];
        v141 = *(_DWORD *)(v3 + 4 * v51 + 64);
        v142 = (signed __int64 *)(v3 + v53 + 8);
      }
      *v142 = v35;
      v143 = (unsigned int)v189;
      v144 = v188;
      v3 = (unsigned int)(v189 - 1);
      if ( *(_DWORD *)(v188 + 16 * v3 + 12) == *(_DWORD *)(v188 + 16 * v3 + 8) - 1 )
      {
        v176 = v141;
        sub_1DA99F0((__int64)&v187, v3, v35);
        v144 = v188;
        v143 = (unsigned int)v189;
        v141 = v176;
      }
      v145 = v144 + 16 * v143 - 16;
      v146 = *(_DWORD *)(v145 + 12) + 1;
      *(_DWORD *)(v145 + 12) = v146;
      if ( v146 == *(_DWORD *)(v188 + 16LL * (unsigned int)v189 - 8) )
      {
        v151 = *((unsigned int *)v187 + 20);
        if ( (_DWORD)v151 )
        {
          v175 = v141;
          sub_39460A0(&v188, v151);
          v141 = v175;
        }
      }
      v1 = v183;
      k = *(_DWORD *)((v183 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v183 >> 1) & 3;
      if ( (unsigned int)k < (*(_DWORD *)((v140 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v140 >> 1) & 3) )
        sub_1DAD0A0((__int64)&v187, v183, v140, v141);
      v49 = (unsigned int)v189;
      v37 = (char *)v188;
LABEL_40:
      LODWORD(v4) = v183;
      if ( !(_DWORD)v49 )
        goto LABEL_63;
      if ( *((_DWORD *)v37 + 3) < *((_DWORD *)v37 + 2) )
      {
        if ( *((_DWORD *)v187 + 20) )
        {
          v1 = v183;
          sub_1DAAD40((__int64)&v187, v183);
          v49 = (unsigned int)v189;
          v37 = (char *)v188;
        }
        else
        {
          v1 = *((unsigned int *)v187 + 21);
          v3 = (unsigned __int64)&v37[16 * v49 - 16];
          for ( k = *(unsigned int *)(v3 + 12); (_DWORD)v1 != (_DWORD)k; k = (unsigned int)(k + 1) )
          {
            v56 = v187[2 * (unsigned int)k + 1];
            v57 = v56 >> 1;
            v4 = v56 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_DWORD *)(v4 + 24) | (unsigned int)(v57 & 3)) > (*(_DWORD *)((v183 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                   | (unsigned int)(v183 >> 1) & 3) )
              break;
          }
          *(_DWORD *)(v3 + 12) = k;
          v49 = (unsigned int)v189;
          v37 = (char *)v188;
        }
        if ( !(_DWORD)v49 )
          goto LABEL_63;
      }
      v1 = *((unsigned int *)v37 + 2);
      if ( *((_DWORD *)v37 + 3) >= (unsigned int)v1 )
        goto LABEL_63;
      v1 = v183;
      v58 = v48;
      v59 = v48 & 0xFFFFFFFFFFFFFFF8LL;
      v60 = v58 & 6;
      LODWORD(v4) = v183 & 0xFFFFFFF8;
      v61 = (__int64 *)(*(_QWORD *)&v37[16 * (unsigned int)v49 - 16]
                      + 16LL * *(unsigned int *)&v37[16 * (unsigned int)v49 - 4]);
      if ( (*(_DWORD *)((*v61 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v61 >> 1) & 3) >= (*(_DWORD *)((v183 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v183 >> 1)
                                                                                               & 3) )
      {
        v35 = *(_QWORD *)(v59 + 8) & 0xFFFFFFFFFFFFFFF9LL | v60;
        goto LABEL_52;
      }
      *v61 = v183;
      if ( *(_QWORD *)(v177 + 384) )
        goto LABEL_213;
      k = *(unsigned int *)(v177 + 320);
      v148 = *(_QWORD **)(v177 + 312);
      v149 = &v148[k];
      LODWORD(v3) = *(_DWORD *)(v177 + 320);
      if ( v148 == v149 )
        goto LABEL_209;
      v1 = v183;
      v150 = *(_QWORD **)(v177 + 312);
      while ( v183 != *v150 )
      {
        if ( v149 == ++v150 )
          goto LABEL_209;
      }
      if ( v149 == v150 )
      {
LABEL_209:
        if ( k > 1 )
        {
          while ( 1 )
          {
            sub_1DA9A50((_QWORD *)(v177 + 344), &v148[k - 1]);
            v152 = *(_DWORD *)(v177 + 320) - 1;
            *(_DWORD *)(v177 + 320) = v152;
            if ( !v152 )
              break;
            v148 = *(_QWORD **)(v177 + 312);
            k = v152;
          }
LABEL_213:
          v1 = (__int64)&v183;
          sub_1DA9A50((_QWORD *)(v177 + 344), &v183);
          goto LABEL_199;
        }
        if ( (unsigned int)v3 >= *(_DWORD *)(v177 + 324) )
        {
          v1 = v177 + 328;
          sub_16CD150(v177 + 312, (const void *)(v177 + 328), 0, 8, v3, v4);
          k = *(unsigned int *)(v177 + 320);
          v149 = (_QWORD *)(*(_QWORD *)(v177 + 312) + 8 * k);
        }
        *v149 = v183;
        ++*(_DWORD *)(v177 + 320);
      }
LABEL_199:
      v49 = (unsigned int)v189;
      v37 = (char *)v188;
      if ( !(_DWORD)v189 )
        goto LABEL_63;
      k = *(_QWORD *)(v59 + 8) & 0xFFFFFFFFFFFFFFF9LL;
      v35 = k | v60;
      if ( *(_DWORD *)(v188 + 12) >= *(_DWORD *)(v188 + 8) )
        goto LABEL_63;
LABEL_52:
      if ( *((_DWORD *)v187 + 20) )
      {
        v1 = v35;
        sub_1DAAD40((__int64)&v187, v35);
      }
      else
      {
        v1 = *((unsigned int *)v187 + 21);
        v3 = (unsigned __int64)&v37[16 * v49 - 16];
        for ( k = *(unsigned int *)(v3 + 12); (_DWORD)v1 != (_DWORD)k; k = (unsigned int)(k + 1) )
        {
          v62 = v187[2 * (unsigned int)k + 1];
          v63 = v62 >> 1;
          v4 = v62 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_DWORD *)(v4 + 24) | (unsigned int)(v63 & 3)) > (*(_DWORD *)((v35 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                 | (unsigned int)(v35 >> 1) & 3) )
            break;
        }
        *(_DWORD *)(v3 + 12) = k;
      }
      v37 = (char *)v188;
      if ( !(_DWORD)v189 || *(_DWORD *)(v188 + 12) >= *(_DWORD *)(v188 + 8) )
        goto LABEL_63;
      v36 += 2;
    }
    while ( v39 != v36 );
    v3 = v35 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v35 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v64 = (char *)(v188 + 16LL * (unsigned int)v189 - 16);
      v1 = *(_QWORD *)v64;
      v65 = 16LL * *((unsigned int *)v64 + 3);
      k = *(_DWORD *)(v3 + 24) | (unsigned int)(v35 >> 1) & 3;
      if ( (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v64 + v65) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (unsigned int)(*(__int64 *)(*(_QWORD *)v64 + v65) >> 1) & 3) < (unsigned int)k )
      {
        *(_QWORD *)(v1 + v65 + 8) = v35;
        v37 = (char *)v188;
        v1 = (unsigned int)(v189 - 1);
        v159 = (char *)(v188 + 16 * v1);
        k = (unsigned int)(*((_DWORD *)v159 + 2) - 1);
        if ( *((_DWORD *)v159 + 3) == (_DWORD)k )
        {
          sub_1DA99F0((__int64)&v187, v1, v35);
          v37 = (char *)v188;
        }
      }
    }
LABEL_63:
    if ( v37 != v190 )
      _libc_free((unsigned __int64)v37);
LABEL_65:
    if ( v217 != v219 )
      _libc_free((unsigned __int64)v217);
    v66 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + v172);
    v67 = *(unsigned int *)(v66 + 48);
    if ( (_DWORD)v67 )
    {
      v68 = 0;
      v69 = 40 * v67;
      do
      {
        while ( 1 )
        {
          v70 = v68 + *(_QWORD *)(v66 + 40);
          if ( !*(_BYTE *)v70 )
          {
            v1 = *(unsigned int *)(v70 + 8);
            if ( (int)v1 < 0 )
              break;
          }
          v68 += 40;
          if ( v69 == v68 )
            goto LABEL_73;
        }
        v68 += 40;
        sub_1DA9020(a1, v1, v66);
      }
      while ( v69 != v68 );
    }
LABEL_73:
    v172 += 8;
  }
  while ( v160 != v172 );
LABEL_74:
  if ( v213 != v215 )
    _libc_free((unsigned __int64)v213);
  sub_1DA2140((__int64)&v206);
  if ( v206 != &v212 )
    j_j___libc_free_0(v206, 8 * v207);
  v71 = v201;
  while ( v71 )
  {
    v72 = v71;
    v71 = (_QWORD *)*v71;
    v73 = v72[13];
    if ( (_QWORD *)v73 != v72 + 15 )
      _libc_free(v73);
    v74 = v72[7];
    if ( (_QWORD *)v74 != v72 + 9 )
      _libc_free(v74);
    j_j___libc_free_0(v72, 216);
  }
  memset(s, 0, 8 * v200);
  v202 = 0;
  v201 = 0;
  if ( s != &v205 )
    j_j___libc_free_0(s, 8 * v200);
  sub_1DA2140((__int64)&v192);
  result = v193;
  if ( v192 != &v198 )
    return j_j___libc_free_0(v192, 8 * v193);
  return result;
}
