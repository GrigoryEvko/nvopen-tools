// Function: sub_20DA7F0
// Address: 0x20da7f0
//
__int64 __fastcall sub_20DA7F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  _BYTE *v3; // r14
  unsigned int v4; // r12d
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rcx
  int v13; // r8d
  __int64 v14; // r13
  __int64 v15; // r10
  __int64 v16; // rbx
  __int64 v17; // r12
  int v18; // edi
  unsigned __int8 v19; // dl
  __int64 *v20; // rsi
  __int64 *v21; // rax
  __int64 *v22; // rcx
  __int64 *v23; // r10
  __int64 v24; // r8
  __int64 *v25; // rbx
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rbx
  _QWORD *v29; // r14
  _QWORD *v30; // r13
  _QWORD *v31; // r12
  unsigned __int64 *v32; // rsi
  unsigned __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // rdi
  __int64 v36; // rbx
  __int64 v37; // rdi
  __int64 v38; // rbx
  __int64 v39; // rdi
  __int64 v40; // rbx
  __int64 v41; // rdi
  __int64 v42; // rcx
  unsigned __int64 v43; // rbx
  __int64 i; // rbx
  unsigned __int64 v45; // rsi
  __int64 j; // rbx
  __int64 v47; // rdx
  __int64 v48; // r11
  _DWORD *v49; // r8
  unsigned int v50; // ecx
  _DWORD *v51; // rax
  __int64 v52; // r13
  __int64 v53; // r14
  __int64 v54; // rdi
  __int64 (*v55)(); // rax
  __int64 v56; // rdx
  __int64 v57; // r9
  unsigned int v58; // eax
  unsigned __int8 v59; // si
  _DWORD *v60; // rcx
  _DWORD *v61; // r10
  _DWORD *v62; // rcx
  _DWORD *v63; // r10
  __int64 v64; // r8
  __int64 v65; // rdx
  __int64 v66; // rbx
  __int64 v67; // r13
  char v68; // al
  unsigned int v69; // r9d
  _DWORD *v70; // rax
  _DWORD *v71; // rcx
  _QWORD *v72; // rcx
  unsigned int v73; // esi
  __int16 v74; // r9
  _WORD *v75; // rsi
  _WORD *v76; // rdi
  unsigned int v77; // eax
  __int64 v78; // rdx
  __int64 v79; // rbx
  _BYTE **v80; // rcx
  __int64 v81; // r13
  __int64 v82; // r15
  unsigned __int8 v83; // al
  int v84; // r12d
  __int64 v85; // rcx
  int v86; // r8d
  _DWORD *v87; // rcx
  _DWORD *v88; // r10
  _DWORD *v89; // rcx
  _DWORD *v90; // r10
  _DWORD *v91; // rcx
  _BYTE *v92; // rsi
  __int64 v93; // rcx
  int *v94; // rbx
  __int64 v95; // r11
  __int64 v96; // r10
  __int64 v97; // rcx
  int *v98; // rbx
  __int64 v99; // rcx
  int *v100; // rbx
  __int64 v101; // r11
  __int64 v102; // r10
  __int64 v103; // rcx
  int *v104; // rbx
  __int64 v105; // rcx
  int *v106; // r11
  __int64 v107; // rax
  int *v108; // rdi
  __int64 v109; // rax
  int *v110; // r9
  char v111; // al
  __int64 v112; // rcx
  int v113; // r8d
  __int64 (*v114)(); // rax
  __int64 v115; // r13
  __int64 v116; // rbx
  _BYTE **v117; // r12
  int v118; // edi
  bool v119; // al
  __int64 v120; // rcx
  __int64 v121; // r8
  int v122; // edi
  __int64 *v123; // r15
  _QWORD *v124; // r15
  __int16 v125; // ax
  __int16 *v126; // rax
  __int16 v127; // dx
  _BYTE **v128; // rax
  _BYTE *v129; // r9
  unsigned __int16 v130; // r12
  __int64 v131; // r14
  __int64 v132; // r15
  __int64 v133; // r13
  __int64 v134; // rbx
  __int16 v135; // ax
  char v136; // al
  __int64 v137; // [rsp+8h] [rbp-318h]
  unsigned __int8 v138; // [rsp+26h] [rbp-2FAh]
  unsigned __int8 v139; // [rsp+27h] [rbp-2F9h]
  _BYTE *v140; // [rsp+30h] [rbp-2F0h]
  __int64 v141; // [rsp+38h] [rbp-2E8h]
  __int64 *v142; // [rsp+58h] [rbp-2C8h]
  __int64 v143; // [rsp+58h] [rbp-2C8h]
  __int64 v144; // [rsp+68h] [rbp-2B8h]
  _BYTE *v145; // [rsp+68h] [rbp-2B8h]
  unsigned __int64 v146; // [rsp+70h] [rbp-2B0h]
  __int64 *v147; // [rsp+70h] [rbp-2B0h]
  __int64 v148; // [rsp+78h] [rbp-2A8h]
  _QWORD *v149; // [rsp+78h] [rbp-2A8h]
  __int64 v150; // [rsp+80h] [rbp-2A0h]
  __int64 v151; // [rsp+80h] [rbp-2A0h]
  unsigned __int8 v152; // [rsp+80h] [rbp-2A0h]
  unsigned __int8 v154; // [rsp+88h] [rbp-298h]
  __int64 *v155; // [rsp+88h] [rbp-298h]
  _BYTE *v156; // [rsp+88h] [rbp-298h]
  __int64 *v157; // [rsp+88h] [rbp-298h]
  __int64 v158; // [rsp+88h] [rbp-298h]
  __int64 v159; // [rsp+88h] [rbp-298h]
  char v160; // [rsp+97h] [rbp-289h] BYREF
  unsigned int v161; // [rsp+98h] [rbp-288h] BYREF
  unsigned int v162; // [rsp+9Ch] [rbp-284h] BYREF
  __int64 *v163; // [rsp+A0h] [rbp-280h] BYREF
  _QWORD *v164; // [rsp+A8h] [rbp-278h] BYREF
  __int64 v165; // [rsp+B0h] [rbp-270h] BYREF
  unsigned __int16 *v166; // [rsp+B8h] [rbp-268h]
  __int64 v167; // [rsp+C0h] [rbp-260h]
  unsigned __int16 v168; // [rsp+C8h] [rbp-258h] BYREF
  _WORD *v169; // [rsp+D0h] [rbp-250h]
  unsigned int v170; // [rsp+D8h] [rbp-248h]
  unsigned __int16 v171; // [rsp+E0h] [rbp-240h]
  unsigned __int64 v172; // [rsp+E8h] [rbp-238h]
  int v173; // [rsp+F0h] [rbp-230h]
  _BYTE *v174; // [rsp+100h] [rbp-220h] BYREF
  __int64 v175; // [rsp+108h] [rbp-218h]
  _BYTE v176[24]; // [rsp+110h] [rbp-210h] BYREF
  int v177; // [rsp+128h] [rbp-1F8h] BYREF
  __int64 v178; // [rsp+130h] [rbp-1F0h]
  int *v179; // [rsp+138h] [rbp-1E8h]
  int *v180; // [rsp+140h] [rbp-1E0h]
  __int64 v181; // [rsp+148h] [rbp-1D8h]
  _BYTE *v182; // [rsp+150h] [rbp-1D0h] BYREF
  __int64 v183; // [rsp+158h] [rbp-1C8h]
  _BYTE v184[24]; // [rsp+160h] [rbp-1C0h] BYREF
  int v185; // [rsp+178h] [rbp-1A8h] BYREF
  __int64 v186; // [rsp+180h] [rbp-1A0h]
  int *v187; // [rsp+188h] [rbp-198h]
  int *v188; // [rsp+190h] [rbp-190h]
  __int64 v189; // [rsp+198h] [rbp-188h]
  _BYTE *v190; // [rsp+1A0h] [rbp-180h] BYREF
  __int64 v191; // [rsp+1A8h] [rbp-178h]
  _BYTE v192[24]; // [rsp+1B0h] [rbp-170h] BYREF
  int v193; // [rsp+1C8h] [rbp-158h] BYREF
  __int64 v194; // [rsp+1D0h] [rbp-150h]
  int *v195; // [rsp+1D8h] [rbp-148h]
  int *v196; // [rsp+1E0h] [rbp-140h]
  __int64 v197; // [rsp+1E8h] [rbp-138h]
  _BYTE *v198; // [rsp+1F0h] [rbp-130h] BYREF
  __int64 v199; // [rsp+1F8h] [rbp-128h]
  _BYTE v200[24]; // [rsp+200h] [rbp-120h] BYREF
  int v201; // [rsp+218h] [rbp-108h] BYREF
  __int64 v202; // [rsp+220h] [rbp-100h]
  int *v203; // [rsp+228h] [rbp-F8h]
  int *v204; // [rsp+230h] [rbp-F0h]
  __int64 v205; // [rsp+238h] [rbp-E8h]
  _BYTE *v206; // [rsp+240h] [rbp-E0h] BYREF
  __int64 v207; // [rsp+248h] [rbp-D8h]
  _BYTE v208[208]; // [rsp+250h] [rbp-D0h] BYREF

  v2 = a1;
  v3 = v208;
  v4 = 0;
  v5 = *(_QWORD *)(a1 + 144);
  v163 = 0;
  v164 = 0;
  v206 = v208;
  v207 = 0x400000000LL;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 264LL);
  if ( v6 == sub_1D820E0 )
    return v4;
  v4 = ((__int64 (__fastcall *)(__int64, __int64, __int64 **, _QWORD **, _BYTE **, __int64))v6)(
         v5,
         a2,
         &v163,
         &v164,
         &v206,
         1);
  if ( (_BYTE)v4 || !v163 || !(_DWORD)v207 )
    goto LABEL_24;
  v8 = (__int64)v164;
  if ( !v164 )
  {
    v20 = *(__int64 **)(a2 + 96);
    v21 = *(__int64 **)(a2 + 88);
    if ( v21 == v20 )
      goto LABEL_24;
    while ( 1 )
    {
      v8 = *v21;
      if ( v163 != (__int64 *)*v21 )
        break;
      if ( v20 == ++v21 )
        goto LABEL_24;
    }
    v164 = (_QWORD *)*v21;
    if ( !v8 )
      goto LABEL_24;
  }
  if ( (unsigned int)((v163[9] - v163[8]) >> 3) > 1
    || (unsigned int)((__int64)(*(_QWORD *)(v8 + 72) - *(_QWORD *)(v8 + 64)) >> 3) > 1 )
  {
LABEL_24:
    v4 = 0;
    goto LABEL_25;
  }
  v174 = v176;
  v175 = 0x400000000LL;
  v179 = &v177;
  v180 = &v177;
  v183 = 0x400000000LL;
  v182 = v184;
  v9 = *(_QWORD *)(v2 + 144);
  v187 = &v185;
  v188 = &v185;
  v10 = *(_QWORD *)(v2 + 160);
  v148 = v9;
  v177 = 0;
  v178 = 0;
  v181 = 0;
  v185 = 0;
  v186 = 0;
  v189 = 0;
  v150 = v10;
  v11 = sub_1DD5EE0(a2);
  v142 = (__int64 *)v11;
  v139 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 664LL))(v9);
  if ( !v139 )
    goto LABEL_51;
  v14 = *(_QWORD *)(v11 + 32);
  v15 = a2;
  v16 = v14 + 40LL * *(unsigned int *)(v11 + 40);
  if ( v14 != v16 )
  {
    v154 = v4;
    v17 = *(_QWORD *)(v11 + 32);
    v144 = v15;
    do
    {
      if ( !*(_BYTE *)v17 )
      {
        v18 = *(_DWORD *)(v17 + 8);
        if ( v18 )
        {
          v19 = *(_BYTE *)(v17 + 3);
          if ( (v19 & 0x10) != 0 )
          {
            if ( (((v19 & 0x10) != 0) & (v19 >> 6)) == 0 )
            {
              v4 = v154;
              goto LABEL_51;
            }
            sub_20DA600(v18, v150, (__int64)&v182, v12, v13);
          }
          else
          {
            sub_20DA600(v18, v150, (__int64)&v174, v12, v13);
          }
        }
      }
      v17 += 40;
    }
    while ( v16 != v17 );
    v4 = v154;
    v15 = v144;
  }
  if ( (_DWORD)v175 || v181 )
  {
    v42 = *(_QWORD *)(v15 + 32);
    if ( v11 != v42 )
    {
      v43 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v43 )
        BUG();
      if ( (*(_QWORD *)v43 & 4) == 0 && (*(_BYTE *)(v43 + 46) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v43; ; i = *(_QWORD *)v43 )
        {
          v43 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v43 + 46) & 4) == 0 )
            break;
        }
      }
      while ( v42 != v43 && (unsigned __int16)(**(_WORD **)(v43 + 16) - 12) <= 1u )
      {
        v45 = *(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v45 )
          BUG();
        v43 = *(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v45 & 4) == 0 && (*(_BYTE *)(v45 + 46) & 4) != 0 )
        {
          for ( j = *(_QWORD *)v45; ; j = *(_QWORD *)v43 )
          {
            v43 = j & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v43 + 46) & 4) == 0 )
              break;
          }
        }
      }
      v47 = *(_QWORD *)(v43 + 32);
      v48 = v47 + 40LL * *(unsigned int *)(v43 + 40);
      if ( v47 != v48 )
      {
        v146 = v43;
        v49 = &v174[4 * (unsigned int)v175];
        while ( *(_BYTE *)v47 != 12 )
        {
          if ( !*(_BYTE *)v47 && (*(_BYTE *)(v47 + 3) & 0x10) != 0 )
          {
            v50 = *(_DWORD *)(v47 + 8);
            if ( v50 )
            {
              if ( v181 )
              {
                v109 = v178;
                if ( v178 )
                {
                  v110 = &v177;
                  do
                  {
                    if ( v50 > *(_DWORD *)(v109 + 32) )
                    {
                      v109 = *(_QWORD *)(v109 + 24);
                    }
                    else
                    {
                      v110 = (int *)v109;
                      v109 = *(_QWORD *)(v109 + 16);
                    }
                  }
                  while ( v109 );
                  if ( v110 != &v177 && v50 >= v110[8] )
                    goto LABEL_253;
                }
              }
              else if ( v174 != (_BYTE *)v49 )
              {
                v51 = v174;
                while ( v50 != *v51 )
                {
                  if ( v49 == ++v51 )
                    goto LABEL_89;
                }
                if ( v51 != v49 )
                {
LABEL_253:
                  v158 = v15;
                  LOBYTE(v165) = 1;
                  v111 = sub_1E17B50(v43, 0, &v165);
                  v15 = v158;
                  if ( !v111 )
                    goto LABEL_51;
                  v114 = *(__int64 (**)())(*(_QWORD *)v148 + 656LL);
                  if ( v114 != sub_1D918C0 )
                  {
                    v136 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v114)(v148, v43);
                    v15 = v158;
                    if ( v136 )
                      goto LABEL_51;
                  }
                  v115 = *(_QWORD *)(v43 + 32);
                  v159 = v115 + 40LL * *(unsigned int *)(v43 + 40);
                  if ( v115 != v159 )
                  {
                    v116 = v150;
                    v152 = v4;
                    v117 = &v174;
                    v143 = v15;
                    do
                    {
                      if ( !*(_BYTE *)v115 )
                      {
                        v118 = *(_DWORD *)(v115 + 8);
                        LODWORD(v190) = v118;
                        if ( v118 )
                        {
                          if ( (*(_BYTE *)(v115 + 3) & 0x10) != 0 )
                          {
                            v119 = sub_20D8970((__int64)v117, (unsigned int *)&v190);
                            v122 = (int)v190;
                            if ( v119 && (int)v190 > 0 )
                            {
                              if ( !v116 )
                                BUG();
                              v120 = 0;
                              v126 = (__int16 *)(*(_QWORD *)(v116 + 56)
                                               + 2LL
                                               * *(unsigned int *)(*(_QWORD *)(v116 + 8) + 24LL * (unsigned int)v190 + 4));
                              v127 = *v126;
                              if ( *v126 )
                                v120 = (__int64)(v126 + 1);
                              v128 = v117;
                              v129 = v3;
                              v130 = v127 + (_WORD)v190;
                              v131 = v116;
                              v121 = v2;
                              v132 = v115;
                              v133 = (__int64)v128;
LABEL_280:
                              v134 = v120;
                              while ( v134 )
                              {
                                v140 = v129;
                                v141 = v121;
                                v134 += 2;
                                LODWORD(v198) = v130;
                                sub_20D8970(v133, (unsigned int *)&v198);
                                v135 = *(_WORD *)(v134 - 2);
                                v120 = 0;
                                v121 = v141;
                                v129 = v140;
                                if ( !v135 )
                                  goto LABEL_280;
                                v130 += v135;
                              }
                              v116 = v131;
                              v117 = (_BYTE **)v133;
                              v122 = (int)v190;
                              v115 = v132;
                              v3 = v129;
                              v2 = v121;
                            }
                            sub_20DA600(v122, v116, (__int64)&v182, v120, v121);
                          }
                          else
                          {
                            sub_20DA600(v118, v116, (__int64)v117, v112, v113);
                          }
                        }
                      }
                      v115 += 40;
                    }
                    while ( v159 != v115 );
                    v43 = v146;
                    v4 = v152;
                    v15 = v143;
                  }
                  v142 = (__int64 *)v43;
                  break;
                }
              }
            }
          }
LABEL_89:
          v47 += 40;
          if ( v48 == v47 )
            break;
        }
      }
    }
  }
  if ( v142 == (__int64 *)(v15 + 24) )
    goto LABEL_51;
  v193 = 0;
  v190 = v192;
  v191 = 0x400000000LL;
  v199 = 0x400000000LL;
  v203 = &v201;
  v204 = &v201;
  v195 = &v193;
  v196 = &v193;
  v194 = 0;
  v197 = 0;
  v198 = v200;
  v201 = 0;
  v202 = 0;
  v205 = 0;
  v157 = v163 + 3;
  v149 = v164 + 3;
  if ( (__int64 *)v163[4] == v163 + 3 )
    goto LABEL_47;
  v147 = (__int64 *)v15;
  v52 = v164[4];
  v145 = v3;
  v53 = v163[4];
  while ( 2 )
  {
    if ( v149 == (_QWORD *)v52 )
      goto LABEL_29;
    for ( ; v157 != (__int64 *)v53; v53 = *(_QWORD *)(v53 + 8) )
    {
      if ( (unsigned __int16)(**(_WORD **)(v53 + 16) - 12) > 1u )
        break;
      if ( (*(_BYTE *)v53 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v53 + 46) & 8) != 0 )
          v53 = *(_QWORD *)(v53 + 8);
      }
    }
    while ( (unsigned __int16)(**(_WORD **)(v52 + 16) - 12) <= 1u )
    {
      if ( (*(_BYTE *)v52 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v52 + 46) & 8) != 0 )
          v52 = *(_QWORD *)(v52 + 8);
      }
      v52 = *(_QWORD *)(v52 + 8);
      if ( v149 == (_QWORD *)v52 )
        goto LABEL_29;
    }
    if ( v157 == (__int64 *)v53
      || v149 == (_QWORD *)v52
      || !(unsigned __int8)sub_1E15D60(v53, v52, 1u)
      || (v54 = *(_QWORD *)(v2 + 144), v55 = *(__int64 (**)())(*(_QWORD *)v54 + 656LL), v55 != sub_1D918C0)
      && ((unsigned __int8 (__fastcall *)(__int64, __int64))v55)(v54, v53) )
    {
LABEL_29:
      v22 = (__int64 *)v53;
      v23 = v147;
      v24 = v52;
      v3 = v145;
      if ( (_BYTE)v4 )
        goto LABEL_30;
      goto LABEL_43;
    }
    v56 = *(_QWORD *)(v53 + 32);
    v57 = v56 + 40LL * *(unsigned int *)(v53 + 40);
    while ( v57 != v56 )
    {
      if ( *(_BYTE *)v56 == 12 )
        goto LABEL_29;
      if ( !*(_BYTE *)v56 )
      {
        v58 = *(_DWORD *)(v56 + 8);
        if ( v58 )
        {
          v59 = *(_BYTE *)(v56 + 3);
          if ( (v59 & 0x10) != 0 )
          {
            if ( v181 )
            {
              v97 = v178;
              if ( v178 )
              {
                v98 = &v177;
                do
                {
                  if ( v58 > *(_DWORD *)(v97 + 32) )
                  {
                    v97 = *(_QWORD *)(v97 + 24);
                  }
                  else
                  {
                    v98 = (int *)v97;
                    v97 = *(_QWORD *)(v97 + 16);
                  }
                }
                while ( v97 );
                if ( v98 != &v177 && v58 >= v98[8] )
                  goto LABEL_29;
              }
            }
            else
            {
              v60 = v174;
              v61 = &v174[4 * (unsigned int)v175];
              if ( v174 != (_BYTE *)v61 )
              {
                while ( v58 != *v60 )
                {
                  if ( v61 == ++v60 )
                    goto LABEL_123;
                }
                if ( v60 != v61 )
                  goto LABEL_29;
              }
            }
LABEL_123:
            if ( v189 )
            {
              v99 = v186;
              if ( v186 )
              {
                v100 = &v185;
                do
                {
                  while ( 1 )
                  {
                    v101 = *(_QWORD *)(v99 + 16);
                    v102 = *(_QWORD *)(v99 + 24);
                    if ( v58 <= *(_DWORD *)(v99 + 32) )
                      break;
                    v99 = *(_QWORD *)(v99 + 24);
                    if ( !v102 )
                      goto LABEL_212;
                  }
                  v100 = (int *)v99;
                  v99 = *(_QWORD *)(v99 + 16);
                }
                while ( v101 );
LABEL_212:
                if ( v100 != &v185 && v58 >= v100[8] )
                {
LABEL_129:
                  if ( (((v59 & 0x10) != 0) & (v59 >> 6)) == 0 )
                    goto LABEL_29;
                }
              }
            }
            else
            {
              v62 = v182;
              v63 = &v182[4 * (unsigned int)v183];
              if ( v182 != (_BYTE *)v63 )
              {
                while ( v58 != *v62 )
                {
                  if ( v63 == ++v62 )
                    goto LABEL_130;
                }
                if ( v62 != v63 )
                  goto LABEL_129;
              }
            }
          }
          else
          {
            if ( v197 )
            {
              v93 = v194;
              if ( v194 )
              {
                v94 = &v193;
                do
                {
                  while ( 1 )
                  {
                    v95 = *(_QWORD *)(v93 + 16);
                    v96 = *(_QWORD *)(v93 + 24);
                    if ( v58 <= *(_DWORD *)(v93 + 32) )
                      break;
                    v93 = *(_QWORD *)(v93 + 24);
                    if ( !v96 )
                      goto LABEL_198;
                  }
                  v94 = (int *)v93;
                  v93 = *(_QWORD *)(v93 + 16);
                }
                while ( v95 );
LABEL_198:
                if ( v94 != &v193 && v58 >= v94[8] )
                  goto LABEL_130;
              }
            }
            else
            {
              v87 = v190;
              v88 = &v190[4 * (unsigned int)v191];
              if ( v190 != (_BYTE *)v88 )
              {
                while ( v58 != *v87 )
                {
                  if ( v88 == ++v87 )
                    goto LABEL_179;
                }
                if ( v88 != v87 )
                  goto LABEL_130;
              }
            }
LABEL_179:
            if ( v189 )
            {
              v103 = v186;
              if ( v186 )
              {
                v104 = &v185;
                do
                {
                  if ( v58 > *(_DWORD *)(v103 + 32) )
                  {
                    v103 = *(_QWORD *)(v103 + 24);
                  }
                  else
                  {
                    v104 = (int *)v103;
                    v103 = *(_QWORD *)(v103 + 16);
                  }
                }
                while ( v103 );
                if ( v104 != &v185 && v58 >= v104[8] )
                  goto LABEL_29;
              }
            }
            else
            {
              v89 = v182;
              v90 = &v182[4 * (unsigned int)v183];
              if ( v182 != (_BYTE *)v90 )
              {
                while ( v58 != *v89 )
                {
                  if ( v90 == ++v89 )
                    goto LABEL_185;
                }
                if ( v90 != v89 )
                  goto LABEL_29;
              }
            }
LABEL_185:
            if ( (*(_BYTE *)(v56 + 3) & 0x40) != 0 )
            {
              if ( v181 )
              {
                v105 = v178;
                if ( !v178 )
                  goto LABEL_130;
                v106 = &v177;
                do
                {
                  if ( v58 > *(_DWORD *)(v105 + 32) )
                  {
                    v105 = *(_QWORD *)(v105 + 24);
                  }
                  else
                  {
                    v106 = (int *)v105;
                    v105 = *(_QWORD *)(v105 + 16);
                  }
                }
                while ( v105 );
                if ( v106 == &v177 || v58 < v106[8] )
                  goto LABEL_130;
              }
              else
              {
                v91 = v174;
                v92 = &v174[4 * (unsigned int)v175];
                if ( v174 == v92 )
                  goto LABEL_130;
                while ( v58 != *v91 )
                {
                  if ( v92 == (_BYTE *)++v91 )
                    goto LABEL_130;
                }
                if ( v92 == (_BYTE *)v91 )
                  goto LABEL_130;
              }
              *(_BYTE *)(v56 + 3) &= ~0x40u;
            }
          }
        }
      }
LABEL_130:
      v56 += 40;
    }
    v160 = 1;
    v138 = sub_1E17B50(v53, 0, &v160);
    if ( !v138 )
      goto LABEL_29;
    v65 = *(_QWORD *)(v53 + 32);
    if ( v65 == v65 + 40LL * *(unsigned int *)(v53 + 40) )
      goto LABEL_163;
    v66 = v65 + 40LL * *(unsigned int *)(v53 + 40);
    v137 = v52;
    v67 = *(_QWORD *)(v53 + 32);
    do
    {
      if ( *(_BYTE *)v67 )
        goto LABEL_153;
      v68 = *(_BYTE *)(v67 + 3);
      if ( (v68 & 0x10) != 0 )
        goto LABEL_153;
      if ( (v68 & 0x40) == 0 )
        goto LABEL_153;
      v69 = *(_DWORD *)(v67 + 8);
      v161 = v69;
      if ( !v69 )
        goto LABEL_153;
      if ( v205 )
      {
        v107 = v202;
        if ( !v202 )
          goto LABEL_153;
        v108 = &v201;
        do
        {
          if ( v69 > *(_DWORD *)(v107 + 32) )
          {
            v107 = *(_QWORD *)(v107 + 24);
          }
          else
          {
            v108 = (int *)v107;
            v107 = *(_QWORD *)(v107 + 16);
          }
        }
        while ( v107 );
        if ( v108 == &v201 || v69 < v108[8] )
          goto LABEL_153;
LABEL_144:
        if ( (int)v69 <= 0 )
        {
          sub_20D8970((__int64)&v190, &v161);
        }
        else
        {
          LODWORD(v165) = v69;
          v72 = *(_QWORD **)(v2 + 160);
          if ( !v72 )
          {
            v166 = 0;
            LOBYTE(v167) = 1;
            v168 = 0;
            v169 = 0;
            v170 = 0;
            v171 = 0;
            v172 = 0;
            BUG();
          }
          v169 = 0;
          v166 = (unsigned __int16 *)(v72 + 1);
          v168 = 0;
          LOBYTE(v167) = 1;
          v170 = 0;
          v171 = 0;
          v172 = 0;
          v73 = *(_DWORD *)(v72[1] + 24LL * v69 + 16);
          v74 = (v73 & 0xF) * v69;
          v75 = (_WORD *)(v72[7] + 2LL * (v73 >> 4));
          v76 = v75 + 1;
          v168 = *v75 + v74;
          v169 = v75 + 1;
          while ( v76 )
          {
            v170 = *(_DWORD *)(v72[6] + 4LL * v168);
            v77 = (unsigned __int16)v170;
            if ( (_WORD)v170 )
            {
              while ( 1 )
              {
                v64 = *(unsigned int *)(v72[1] + 24LL * (unsigned __int16)v77 + 8);
                v78 = v72[7];
                v171 = v77;
                v172 = v78 + 2 * v64;
                if ( v172 )
                  break;
                v170 = HIWORD(v170);
                v77 = v170;
                if ( !(_WORD)v170 )
                  goto LABEL_271;
              }
              while ( 1 )
              {
                v162 = v77;
                sub_20D8970((__int64)&v190, &v162);
                sub_1E1D5E0((__int64)&v165);
                if ( !v169 )
                  break;
                v77 = v171;
              }
              goto LABEL_153;
            }
LABEL_271:
            v169 = ++v76;
            v125 = *(v76 - 1);
            v168 += v125;
            if ( !v125 )
            {
              v169 = 0;
              goto LABEL_153;
            }
          }
        }
        goto LABEL_153;
      }
      v70 = v198;
      v71 = &v198[4 * (unsigned int)v199];
      if ( v198 != (_BYTE *)v71 )
      {
        while ( v69 != *v70 )
        {
          if ( v71 == ++v70 )
            goto LABEL_153;
        }
        if ( v70 != v71 )
          goto LABEL_144;
      }
LABEL_153:
      v67 += 40;
    }
    while ( v66 != v67 );
    v52 = v137;
    v79 = *(_QWORD *)(v53 + 32);
    if ( v79 != v79 + 40LL * *(unsigned int *)(v53 + 40) )
    {
      v80 = &v190;
      v81 = v2;
      v82 = v79 + 40LL * *(unsigned int *)(v53 + 40);
      do
      {
        if ( !*(_BYTE *)v79 )
        {
          v83 = *(_BYTE *)(v79 + 3);
          if ( (v83 & 0x10) != 0 && (((v83 & 0x10) != 0) & (v83 >> 6)) == 0 )
          {
            v84 = *(_DWORD *)(v79 + 8);
            if ( v84 > 0 )
            {
              sub_20DA600(v84, *(_QWORD *)(v81 + 160), (__int64)&v190, (__int64)v80, v64);
              sub_20DA600(v84, *(_QWORD *)(v81 + 160), (__int64)&v198, v85, v86);
            }
          }
        }
        v79 += 40;
      }
      while ( v82 != v79 );
      v2 = v81;
      v52 = v137;
    }
LABEL_163:
    if ( (*(_BYTE *)v53 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v53 + 46) & 8) != 0 )
        v53 = *(_QWORD *)(v53 + 8);
    }
    v53 = *(_QWORD *)(v53 + 8);
    if ( (*(_BYTE *)v52 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v52 + 46) & 8) != 0 )
        v52 = *(_QWORD *)(v52 + 8);
    }
    v52 = *(_QWORD *)(v52 + 8);
    if ( v157 != (__int64 *)v53 )
    {
      v4 = v138;
      continue;
    }
    break;
  }
  v22 = (__int64 *)v53;
  v23 = v147;
  v24 = v52;
  v3 = v145;
LABEL_30:
  v25 = (__int64 *)v163[4];
  if ( v22 != v25 && v22 != v142 )
  {
    if ( v163 != v23 )
    {
      v151 = v24;
      v155 = v22;
      sub_1DD5C00(v23 + 2, (__int64)(v163 + 2), v163[4], (__int64)v22);
      v24 = v151;
      v22 = v155;
    }
    if ( v22 != v142 && v22 != v25 )
    {
      v26 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*v25 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v22;
      *v22 = *v22 & 7 | *v25 & 0xFFFFFFFFFFFFFFF8LL;
      v27 = *v142;
      *(_QWORD *)(v26 + 8) = v142;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      *v25 = v27 | *v25 & 7;
      *(_QWORD *)(v27 + 8) = v25;
      *v142 = v26 | *v142 & 7;
    }
  }
  v28 = (__int64)(v164 + 2);
  if ( v164[4] != v24 )
  {
    v156 = v3;
    v29 = (_QWORD *)v164[4];
    v30 = (_QWORD *)v24;
    do
    {
      v31 = v29;
      v29 = (_QWORD *)v29[1];
      sub_1DD5BC0(v28, (__int64)v31);
      v32 = (unsigned __int64 *)v31[1];
      v33 = *v31 & 0xFFFFFFFFFFFFFFF8LL;
      *v32 = v33 | *v32 & 7;
      *(_QWORD *)(v33 + 8) = v32;
      *v31 &= 7uLL;
      v31[1] = 0;
      sub_1DD5C20(v28);
    }
    while ( v30 != v29 );
    v3 = v156;
  }
  v4 = *(unsigned __int8 *)(v2 + 139);
  if ( (_BYTE)v4 )
  {
    v123 = v163;
    v167 = 0x800000000LL;
    v165 = 0;
    v166 = &v168;
    v172 = 0;
    v173 = 0;
    sub_1DD77B0((__int64)v163);
    sub_1DC3250((__int64)&v165, v123);
    _libc_free(v172);
    if ( v166 != &v168 )
      _libc_free((unsigned __int64)v166);
    v124 = v164;
    v166 = &v168;
    v167 = 0x800000000LL;
    v165 = 0;
    v172 = 0;
    v173 = 0;
    sub_1DD77B0((__int64)v164);
    sub_1DC3250((__int64)&v165, v124);
    _libc_free(v172);
    if ( v166 != &v168 )
      _libc_free((unsigned __int64)v166);
  }
  else
  {
    v4 = v139;
  }
LABEL_43:
  v34 = v202;
  while ( v34 )
  {
    sub_20D63D0(*(_QWORD *)(v34 + 24));
    v35 = v34;
    v34 = *(_QWORD *)(v34 + 16);
    j_j___libc_free_0(v35, 40);
  }
  if ( v198 != v200 )
    _libc_free((unsigned __int64)v198);
LABEL_47:
  v36 = v194;
  while ( v36 )
  {
    sub_20D63D0(*(_QWORD *)(v36 + 24));
    v37 = v36;
    v36 = *(_QWORD *)(v36 + 16);
    j_j___libc_free_0(v37, 40);
  }
  if ( v190 != v192 )
    _libc_free((unsigned __int64)v190);
LABEL_51:
  v38 = v186;
  while ( v38 )
  {
    sub_20D63D0(*(_QWORD *)(v38 + 24));
    v39 = v38;
    v38 = *(_QWORD *)(v38 + 16);
    j_j___libc_free_0(v39, 40);
  }
  if ( v182 != v184 )
    _libc_free((unsigned __int64)v182);
  v40 = v178;
  while ( v40 )
  {
    sub_20D63D0(*(_QWORD *)(v40 + 24));
    v41 = v40;
    v40 = *(_QWORD *)(v40 + 16);
    j_j___libc_free_0(v41, 40);
  }
  if ( v174 != v176 )
    _libc_free((unsigned __int64)v174);
LABEL_25:
  if ( v206 != v3 )
    _libc_free((unsigned __int64)v206);
  return v4;
}
