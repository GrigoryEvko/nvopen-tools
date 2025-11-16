// Function: sub_19996C0
// Address: 0x19996c0
//
void __fastcall sub_19996C0(
        __int64 *a1,
        __m128i a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // r15
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 *v14; // r12
  __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 *v20; // rdx
  __int64 *v21; // r13
  __int64 v22; // r15
  unsigned __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r11
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 *v28; // rdx
  unsigned __int64 v29; // rcx
  unsigned __int64 *v30; // r12
  unsigned __int64 v31; // rax
  unsigned __int64 *v32; // rbx
  unsigned __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // r10
  __int64 v39; // rdx
  int v40; // r11d
  unsigned int v41; // ecx
  __int64 *v42; // rdi
  __int64 v43; // r9
  __int64 *v44; // rcx
  __int64 v45; // rdx
  unsigned int v46; // r9d
  __int64 *v47; // rdi
  __int64 v48; // r14
  __int64 v49; // rcx
  __int64 v50; // rdi
  unsigned __int64 *v51; // rdx
  __int64 v52; // r12
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // r14
  bool v56; // al
  _QWORD *v57; // r11
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // r14
  __int64 v61; // rdx
  __int64 v62; // rsi
  __int64 v63; // rcx
  __int64 *v64; // rax
  int v65; // edi
  int v66; // r13d
  int v67; // edi
  __int64 v68; // r15
  __int64 v69; // rbx
  __int64 v70; // rax
  __int64 v71; // r8
  __int64 v72; // r14
  __int64 v73; // rax
  __int64 v74; // r14
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r9
  __int64 v78; // r8
  __int64 v79; // r14
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // rax
  unsigned __int64 v83; // rax
  __int64 v84; // r14
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // r14
  unsigned int v89; // edx
  __int64 v90; // rdi
  int v91; // eax
  bool v92; // al
  unsigned int v93; // edx
  __int64 v94; // r8
  __int64 v95; // rcx
  __int64 v96; // rsi
  unsigned int v97; // eax
  unsigned __int64 v98; // r14
  __int64 *v99; // rsi
  __int64 *v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  bool v104; // al
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 *v108; // rax
  char v109; // al
  __int64 v110; // rax
  __int64 v111; // rsi
  __int64 v112; // rax
  char v113; // al
  __int64 v114; // rdx
  __int64 v115; // rcx
  unsigned int v116; // r8d
  __int64 v117; // rax
  __int64 *v118; // rdx
  __int64 v119; // rax
  bool v120; // al
  __int64 v121; // rdx
  __int64 v122; // rdi
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 *v125; // rdx
  __int64 v126; // rax
  bool v127; // al
  int v128; // eax
  __int64 v129; // rax
  _QWORD *v130; // rax
  double v131; // xmm4_8
  double v132; // xmm5_8
  __int64 v133; // r11
  _QWORD **v134; // rax
  __int64 *v135; // rax
  __int64 v136; // rax
  __int64 v137; // r11
  __int64 v138; // rax
  _QWORD *v139; // rdi
  _QWORD *v140; // r11
  int v141; // r13d
  __int64 v142; // rax
  __int64 *v143; // rax
  __int64 v144; // rax
  __int64 v145; // rax
  bool v146; // al
  __int16 v147; // ax
  __int16 v148; // ax
  __int64 v149; // rax
  __int64 *v150; // rax
  __int64 v151; // rcx
  unsigned int v152; // r8d
  __int64 v153; // rdx
  __int64 v154; // rdx
  __int64 v155; // rax
  __int64 v156; // [rsp+0h] [rbp-160h]
  _QWORD *v157; // [rsp+8h] [rbp-158h]
  __int64 v158; // [rsp+10h] [rbp-150h]
  __int64 *v159; // [rsp+10h] [rbp-150h]
  __int64 *v160; // [rsp+10h] [rbp-150h]
  __int64 v161; // [rsp+10h] [rbp-150h]
  __int64 v162; // [rsp+18h] [rbp-148h]
  __int64 v163; // [rsp+18h] [rbp-148h]
  __int64 v164; // [rsp+18h] [rbp-148h]
  __int64 v165; // [rsp+18h] [rbp-148h]
  __int64 v166; // [rsp+18h] [rbp-148h]
  __int64 v167; // [rsp+18h] [rbp-148h]
  __int64 v168; // [rsp+18h] [rbp-148h]
  _QWORD *v169; // [rsp+18h] [rbp-148h]
  __int64 v170; // [rsp+18h] [rbp-148h]
  __int64 v171; // [rsp+18h] [rbp-148h]
  __int64 v172; // [rsp+20h] [rbp-140h]
  __int64 v173; // [rsp+20h] [rbp-140h]
  __int64 v174; // [rsp+20h] [rbp-140h]
  __int64 v175; // [rsp+20h] [rbp-140h]
  __int64 v176; // [rsp+20h] [rbp-140h]
  __int64 v177; // [rsp+28h] [rbp-138h]
  __int64 v178; // [rsp+28h] [rbp-138h]
  unsigned int v179; // [rsp+28h] [rbp-138h]
  unsigned int v180; // [rsp+28h] [rbp-138h]
  __int64 v181; // [rsp+28h] [rbp-138h]
  __int64 v182; // [rsp+28h] [rbp-138h]
  __int64 v183; // [rsp+28h] [rbp-138h]
  __int64 v184; // [rsp+28h] [rbp-138h]
  __int64 v185; // [rsp+28h] [rbp-138h]
  __int64 v186; // [rsp+28h] [rbp-138h]
  __int64 v187; // [rsp+30h] [rbp-130h]
  __int64 v188; // [rsp+30h] [rbp-130h]
  unsigned __int64 v189; // [rsp+30h] [rbp-130h]
  unsigned int v190; // [rsp+30h] [rbp-130h]
  __int64 v191; // [rsp+30h] [rbp-130h]
  __int64 v192; // [rsp+38h] [rbp-128h]
  _QWORD *v193; // [rsp+40h] [rbp-120h]
  __int64 v194; // [rsp+40h] [rbp-120h]
  __int64 v195; // [rsp+40h] [rbp-120h]
  __int64 v196; // [rsp+40h] [rbp-120h]
  _QWORD *v197; // [rsp+40h] [rbp-120h]
  __int64 v198; // [rsp+40h] [rbp-120h]
  __int64 v199; // [rsp+40h] [rbp-120h]
  int v200; // [rsp+40h] [rbp-120h]
  _QWORD *v201; // [rsp+40h] [rbp-120h]
  _QWORD *v202; // [rsp+40h] [rbp-120h]
  _QWORD *v203; // [rsp+40h] [rbp-120h]
  __int64 *v204; // [rsp+50h] [rbp-110h]
  __int64 *v205; // [rsp+58h] [rbp-108h]
  _QWORD v206[2]; // [rsp+60h] [rbp-100h] BYREF
  __int64 v207[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v208; // [rsp+80h] [rbp-E0h]
  __int64 v209; // [rsp+90h] [rbp-D0h] BYREF
  unsigned __int64 *v210; // [rsp+98h] [rbp-C8h]
  unsigned __int64 *v211; // [rsp+A0h] [rbp-C0h]
  __int64 v212; // [rsp+A8h] [rbp-B8h]
  int v213; // [rsp+B0h] [rbp-B0h]
  _BYTE v214[40]; // [rsp+B8h] [rbp-A8h] BYREF
  __int64 *v215; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v216; // [rsp+E8h] [rbp-78h]
  _BYTE v217[112]; // [rsp+F0h] [rbp-70h] BYREF

  v9 = a1;
  v10 = a1[5];
  v209 = 0;
  v210 = (unsigned __int64 *)v214;
  v211 = (unsigned __int64 *)v214;
  v212 = 4;
  v213 = 0;
  v11 = sub_13FCB50(v10);
  v12 = v9[5];
  v13 = v11;
  v215 = (__int64 *)v217;
  v216 = 0x800000000LL;
  sub_13F9CA0(v12, (__int64)&v215);
  v14 = v215;
  v15 = 8LL * (unsigned int)v216;
  v16 = &v215[(unsigned __int64)v15 / 8];
  v17 = v15 >> 3;
  v18 = v15 >> 5;
  v205 = v16;
  if ( !v18 )
  {
    v19 = v215;
LABEL_75:
    if ( v17 != 2 )
    {
      if ( v17 != 3 )
      {
        if ( v17 != 1 )
          goto LABEL_79;
        goto LABEL_78;
      }
      if ( v13 == *v19 )
        goto LABEL_8;
      ++v19;
    }
    if ( v13 == *v19 )
      goto LABEL_8;
    ++v19;
LABEL_78:
    if ( v13 != *v19 )
      goto LABEL_79;
    goto LABEL_8;
  }
  v19 = v215;
  v20 = &v215[4 * v18];
  while ( 1 )
  {
    if ( v13 == *v19 )
      goto LABEL_8;
    if ( v13 == v19[1] )
    {
      ++v19;
      goto LABEL_8;
    }
    if ( v13 == v19[2] )
    {
      v19 += 2;
      goto LABEL_8;
    }
    if ( v13 == v19[3] )
      break;
    v19 += 4;
    if ( v20 == v19 )
    {
      v17 = v205 - v19;
      goto LABEL_75;
    }
  }
  v19 += 3;
LABEL_8:
  if ( v205 != v19 )
  {
    v21 = v215;
    if ( v215 != v205 )
    {
      v192 = v13;
      v204 = v9;
      while ( 1 )
      {
        v22 = *v21;
        v23 = sub_157EBA0(*v21);
        v24 = v23;
        if ( *(_BYTE *)(v23 + 16) != 26 )
          goto LABEL_14;
        if ( (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) == 1 )
          goto LABEL_14;
        v25 = *(_QWORD *)(v23 - 72);
        if ( *(_BYTE *)(v25 + 16) != 75 )
          goto LABEL_14;
        v52 = *(_QWORD *)(*v204 + 216);
        v53 = *v204 + 208;
        if ( v53 == v52 )
          goto LABEL_14;
        while ( 1 )
        {
          if ( !v52 )
            BUG();
          if ( *(_QWORD *)(v52 - 8) == v25 )
            break;
          v52 = *(_QWORD *)(v52 + 8);
          if ( v52 == v53 )
            goto LABEL_14;
        }
        v54 = *(unsigned __int16 *)(v25 + 18);
        BYTE1(v54) &= ~0x80u;
        if ( (unsigned int)(v54 - 32) <= 1 )
        {
          v55 = *(_QWORD *)(v25 - 24);
          if ( *(_BYTE *)(v55 + 16) == 79 )
          {
            v103 = *(_QWORD *)(v55 + 8);
            if ( v103 )
            {
              v191 = *(_QWORD *)(v103 + 8);
              if ( !v191 )
              {
                v181 = v25;
                v198 = sub_1481F60((_QWORD *)v204[1], v204[5], a2, a3);
                v104 = sub_14562D0(v198);
                v25 = v181;
                if ( !v104 )
                {
                  v163 = v181;
                  v182 = v204[1];
                  v105 = sub_1456040(v198);
                  v173 = sub_145CF80(v182, v105, 1, 0);
                  v183 = v198;
                  v199 = sub_13A5B00(v204[1], v173, v198, 0, 0);
                  v106 = sub_146F1B0(v204[1], v55);
                  v107 = v199;
                  v25 = v163;
                  if ( v199 == v106 )
                  {
                    if ( *(_WORD *)(v183 + 24) == 9 )
                    {
                      v200 = 41;
                      v107 = v183;
                    }
                    else
                    {
                      v200 = 40;
                      v148 = *(_WORD *)(v107 + 24);
                      if ( v148 != 9 )
                      {
                        if ( v148 != 8 )
                          goto LABEL_64;
                        v200 = 36;
                      }
                    }
                    if ( *(_QWORD *)(v107 + 40) == 2 )
                    {
                      v108 = *(__int64 **)(v107 + 32);
                      v158 = *v108;
                      if ( *v108 )
                      {
                        v184 = v108[1];
                        v109 = sub_15FF820(v200);
                        v25 = v163;
                        if ( v109 )
                        {
                          v146 = sub_14560B0(v158);
                          v25 = v163;
                          if ( !v146 )
                            goto LABEL_64;
                        }
                        else if ( v173 != v158 )
                        {
                          goto LABEL_64;
                        }
                        v164 = v25;
                        v110 = sub_146F1B0(v204[1], *(_QWORD *)(v25 - 48));
                        v25 = v164;
                        if ( *(_WORD *)(v110 + 24) == 7
                          && *(_QWORD *)(v110 + 40) == 2
                          && v173 == **(_QWORD **)(v110 + 32) )
                        {
                          v111 = v204[1];
                          v112 = sub_13A5BC0((_QWORD *)v110, v111);
                          v25 = v164;
                          if ( v173 == v112 )
                            break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
LABEL_64:
        v193 = (_QWORD *)v25;
        v56 = sub_15CC8F0(v204[2], v22, v192);
        v57 = v193;
        if ( v56 )
        {
          v172 = v52 - 32;
          if ( v192 != v22 && *(_QWORD *)(*v204 + 216) != *v204 + 208 )
          {
            v196 = v22;
            v68 = *(_QWORD *)(*v204 + 216);
            v156 = v24;
            v69 = *v204 + 208;
            v157 = v57;
            while ( 1 )
            {
              if ( !v68 )
                BUG();
              if ( v68 - 32 != v172 && !sub_15CC890(v204[2], *(_QWORD *)(*(_QWORD *)(v68 - 8) + 40LL), v196) )
              {
                v187 = sub_13CA540(*v204, v172, v204[5]);
                v70 = sub_13CA540(*v204, v68 - 32, v204[5]);
                v71 = v187;
                if ( v187 )
                {
                  v188 = v70;
                  if ( v70 )
                  {
                    v72 = v204[1];
                    v162 = v71;
                    v73 = sub_1456040(v71);
                    v177 = sub_1456C90(v72, v73);
                    v74 = v204[1];
                    v75 = sub_1456040(v188);
                    v76 = sub_1456C90(v74, v75);
                    v77 = v188;
                    v78 = v162;
                    if ( v177 != v76 )
                    {
                      v178 = v188;
                      v79 = v204[1];
                      v80 = sub_1456040(v162);
                      v189 = sub_1456C90(v79, v80);
                      v81 = v204[1];
                      v82 = sub_1456040(v178);
                      v83 = sub_1456C90(v81, v82);
                      v84 = v204[1];
                      if ( v189 <= v83 )
                      {
                        v101 = sub_1456040(v178);
                        v102 = sub_147B0D0(v84, v162, v101, 0);
                        v77 = v178;
                        v78 = v102;
                      }
                      else
                      {
                        v85 = sub_1456040(v162);
                        v86 = sub_147B0D0(v84, v178, v85, 0);
                        v78 = v162;
                        v77 = v86;
                      }
                    }
                    v87 = sub_1999100(v77, v78, (_QWORD *)v204[1], 0, a2, a3);
                    if ( v87 )
                    {
                      if ( !*(_WORD *)(v87 + 24) )
                      {
                        v88 = *(_QWORD *)(v87 + 32);
                        v89 = *(_DWORD *)(v88 + 32);
                        v90 = v88 + 24;
                        if ( v89 <= 0x40 )
                        {
                          v92 = *(_QWORD *)(v88 + 24) == 1;
                        }
                        else
                        {
                          v179 = *(_DWORD *)(v88 + 32);
                          v91 = sub_16A57B0(v90);
                          v89 = v179;
                          v90 = v88 + 24;
                          v92 = v179 - 1 == v91;
                        }
                        if ( v92 )
                          goto LABEL_14;
                        if ( v89 <= 0x40 )
                        {
                          if ( *(_QWORD *)(v88 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v89) )
                            goto LABEL_14;
                          v190 = v89;
                          if ( (unsigned int)sub_1997E70(v90) > 0x3F )
                            goto LABEL_14;
                          v95 = v190 - 1;
                          if ( *(_QWORD *)(v88 + 24) == 1LL << ((unsigned __int8)v190 - 1) )
                            goto LABEL_14;
                        }
                        else
                        {
                          v180 = v89;
                          if ( v89 == (unsigned int)sub_16A58F0(v90) )
                            goto LABEL_14;
                          if ( (unsigned int)sub_1997E70(v90) > 0x3F )
                            goto LABEL_14;
                          v93 = v180 - 1;
                          v94 = *(_QWORD *)(v88 + 24);
                          v95 = v180 - 1;
                          if ( (*(_QWORD *)(v94 + 8LL * (v93 >> 6)) & (1LL << v93)) != 0
                            && v93 == (unsigned int)sub_16A58A0(v90) )
                          {
                            goto LABEL_14;
                          }
                        }
                        if ( (unsigned __int8)sub_1994130(
                                                v204[4],
                                                *(_QWORD *)(v68 - 8),
                                                *(_QWORD *)(v68 + 40),
                                                v95,
                                                v94) )
                        {
                          v96 = sub_19927B0(v204[4], *(__int64 **)(v68 - 8), *(__int64 **)(v68 + 40));
                          v97 = *(_DWORD *)(v88 + 32);
                          v98 = v97 > 0x40
                              ? **(_QWORD **)(v88 + 24)
                              : (__int64)(*(_QWORD *)(v88 + 24) << (64 - (unsigned __int8)v97)) >> (64
                                                                                                  - (unsigned __int8)v97);
                          if ( sub_14A2A90((__int64 *)v204[4], v96, 0, 0, 0, v98)
                            || sub_14A2A90((__int64 *)v204[4], v96, 0, 0, 0, -(__int64)v98) )
                          {
                            goto LABEL_14;
                          }
                        }
                      }
                    }
                  }
                }
              }
              v68 = *(_QWORD *)(v68 + 8);
              if ( v69 == v68 )
              {
                v22 = v196;
                v57 = v157;
                v24 = v156;
                break;
              }
            }
          }
          if ( !v57 )
            BUG();
          v58 = v57[4];
          if ( !v58 || v24 != v58 - 24 )
          {
            v59 = v57[1];
            if ( v59 && !*(_QWORD *)(v59 + 8) )
            {
              v197 = v57;
              sub_15F22F0(v57, v24);
              v57 = v197;
            }
            else
            {
              v194 = (__int64)v57;
              v60 = sub_15F4880((__int64)v57);
              v206[0] = sub_1649960(**(_QWORD **)(v204[5] + 32));
              v208 = 773;
              v207[1] = (__int64)".termcond";
              v206[1] = v61;
              v207[0] = (__int64)v206;
              sub_164B780(v60, v207);
              sub_157E9D0(v22 + 40, v60);
              v62 = *(_QWORD *)(v60 + 24);
              v63 = *(_QWORD *)(v24 + 24);
              *(_QWORD *)(v60 + 32) = v24 + 24;
              v63 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v60 + 24) = v63 | v62 & 7;
              *(_QWORD *)(v63 + 8) = v60 + 24;
              *(_QWORD *)(v24 + 24) = *(_QWORD *)(v24 + 24) & 7LL | (v60 + 24);
              v172 = sub_13C9CF0(*v204, v60, *(_QWORD *)(v52 + 40));
              sub_1648780(v24, v194, v60);
              v57 = (_QWORD *)v60;
            }
          }
          v195 = (__int64)v57;
          sub_13CA580(v172, v204[5]);
          v64 = (__int64 *)v210;
          *((_BYTE *)v204 + 48) = 1;
          if ( v211 != (unsigned __int64 *)v64 )
            goto LABEL_73;
          v99 = &v64[HIDWORD(v212)];
          if ( v64 != v99 )
          {
            v100 = 0;
            while ( v195 != *v64 )
            {
              if ( *v64 == -2 )
                v100 = v64;
              if ( v99 == ++v64 )
              {
                if ( !v100 )
                  goto LABEL_185;
                *v100 = v195;
                --v213;
                ++v209;
                goto LABEL_14;
              }
            }
            goto LABEL_14;
          }
LABEL_185:
          if ( HIDWORD(v212) < (unsigned int)v212 )
          {
            ++HIDWORD(v212);
            *v99 = v195;
            ++v209;
          }
          else
          {
LABEL_73:
            sub_16CCBA0((__int64)&v209, v195);
          }
        }
LABEL_14:
        if ( v205 == ++v21 )
        {
          v9 = v204;
          goto LABEL_16;
        }
      }
      v113 = sub_15FF820(v200);
      v25 = v164;
      if ( !v113 )
      {
        v144 = sub_146F1B0(v204[1], *(_QWORD *)(v55 - 48));
        v25 = v164;
        if ( v184 == v144 )
        {
          v191 = *(_QWORD *)(v55 - 48);
        }
        else
        {
          v145 = sub_146F1B0(v204[1], *(_QWORD *)(v55 - 24));
          v25 = v164;
          if ( v184 == v145 )
          {
            v191 = *(_QWORD *)(v55 - 24);
          }
          else
          {
            if ( *(_WORD *)(v184 + 24) != 10 )
              goto LABEL_64;
            v191 = *(_QWORD *)(v184 - 8);
          }
        }
        goto LABEL_166;
      }
      v174 = *(_QWORD *)(v55 - 48);
      if ( !v174 )
        goto LABEL_161;
      if ( !(unsigned __int8)sub_1992FE0(v174, v111, v114, v115, v116) )
        goto LABEL_161;
      v165 = v25;
      v117 = sub_13CF970(v174);
      v25 = v165;
      v118 = (__int64 *)v117;
      v119 = *(_QWORD *)(v117 + 24);
      v159 = v118;
      if ( *(_BYTE *)(v119 + 16) != 13 )
        goto LABEL_161;
      v120 = sub_1455000(v119 + 24);
      v25 = v165;
      if ( v120 && (v111 = *v159, v149 = sub_146F1B0(v204[1], *v159), v25 = v165, v184 == v149) )
      {
        v150 = (__int64 *)sub_13CF970(v174);
        v153 = *(_QWORD *)(v55 - 24);
        v25 = v165;
        v191 = *v150;
        if ( !v153 )
          goto LABEL_165;
        if ( !(unsigned __int8)sub_1992FE0(v153, v111, v153, v151, v152) )
          goto LABEL_165;
        v171 = v25;
        v175 = v154;
        v155 = sub_13CF970(v154);
        v25 = v171;
        v125 = (__int64 *)v155;
        v126 = *(_QWORD *)(v155 + 24);
        if ( *(_BYTE *)(v126 + 16) != 13 )
          goto LABEL_165;
      }
      else
      {
LABEL_161:
        v121 = *(_QWORD *)(v55 - 24);
        if ( !v121 )
          goto LABEL_64;
        v122 = *(_QWORD *)(v55 - 24);
        if ( !(unsigned __int8)sub_1992FE0(v122, v111, v121, v115, v116) )
          goto LABEL_64;
        v166 = v25;
        v175 = v123;
        v124 = sub_13CF970(v122);
        v25 = v166;
        v125 = (__int64 *)v124;
        v126 = *(_QWORD *)(v124 + 24);
        if ( *(_BYTE *)(v126 + 16) != 13 )
          goto LABEL_64;
      }
      v160 = v125;
      v167 = v25;
      v127 = sub_1455000(v126 + 24);
      v25 = v167;
      if ( v127 )
      {
        v142 = sub_146F1B0(v204[1], *v160);
        v25 = v167;
        if ( v184 == v142 )
        {
          v143 = (__int64 *)sub_13CF970(v175);
          v25 = v167;
          v191 = *v143;
        }
      }
LABEL_165:
      if ( !v191 )
        goto LABEL_64;
LABEL_166:
      v128 = *(unsigned __int16 *)(v25 + 18);
      BYTE1(v128) &= ~0x80u;
      if ( v128 == 32 )
      {
        v186 = v25;
        v147 = sub_15FF0F0(v200);
        v25 = v186;
        LOWORD(v200) = v147;
      }
      v129 = *(_QWORD *)(v25 - 48);
      v168 = v25;
      v207[0] = (__int64)"scmp";
      v176 = v129;
      v208 = 259;
      v130 = sub_1648A60(56, 2u);
      v133 = v168;
      v185 = (__int64)v130;
      if ( v130 )
      {
        v134 = *(_QWORD ***)v176;
        if ( *(_BYTE *)(*(_QWORD *)v176 + 8LL) == 16 )
        {
          v161 = v168;
          v169 = v134[4];
          v135 = (__int64 *)sub_1643320(*v134);
          v136 = (__int64)sub_16463B0(v135, (unsigned int)v169);
          v137 = v161;
        }
        else
        {
          v136 = sub_1643320(*v134);
          v137 = v168;
        }
        v170 = v137;
        sub_15FEC10(v185, v136, 51, v200, v176, v191, (__int64)v207, v137);
        v133 = v170;
      }
      v201 = (_QWORD *)v133;
      sub_164D160(v133, v185, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v131, v132, a8, a9);
      v138 = *(_QWORD *)(v52 - 8);
      v139 = (_QWORD *)(v52 - 24);
      v140 = v201;
      if ( v185 != v138 )
      {
        if ( v138 != 0 && v138 != -8 && v138 != -16 )
        {
          sub_1649B30(v139);
          v140 = v201;
          v139 = (_QWORD *)(v52 - 24);
        }
        *(_QWORD *)(v52 - 8) = v185;
        if ( v185 != 0 && v185 != -8 && v185 != -16 )
        {
          v202 = v140;
          sub_164C220((__int64)v139);
          v140 = v202;
        }
      }
      v203 = *(_QWORD **)(v55 - 72);
      sub_15F20C0(v140);
      sub_15F20C0((_QWORD *)v55);
      v25 = v185;
      if ( !v203[1] )
      {
        sub_15F20C0(v203);
        v25 = v185;
      }
      goto LABEL_64;
    }
LABEL_16:
    v26 = sub_13FCB50(v9[5]);
    v27 = sub_157EBA0(v26);
    v28 = v211;
    v9[7] = v27;
    v29 = v27;
    if ( v28 == v210 )
      v30 = &v28[HIDWORD(v212)];
    else
      v30 = &v28[(unsigned int)v212];
    if ( v28 != v30 )
    {
      while ( 1 )
      {
        v31 = *v28;
        v32 = v28;
        if ( *v28 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v30 == ++v28 )
          goto LABEL_21;
      }
      if ( v30 != v28 )
      {
        while ( 1 )
        {
          v34 = *(_QWORD *)(v29 + 40);
          v35 = *(_QWORD *)(v31 + 40);
          v36 = *(_QWORD *)(*(_QWORD *)(v34 + 56) + 80LL);
          if ( v36 )
            v36 -= 24;
          if ( v34 != v36 )
            break;
LABEL_54:
          if ( v35 == v36 )
            goto LABEL_46;
          if ( v34 != v36 )
            goto LABEL_45;
LABEL_47:
          v51 = v32 + 1;
          if ( v32 + 1 == v30 )
            goto LABEL_21;
          v31 = *v51;
          for ( ++v32; *v51 >= 0xFFFFFFFFFFFFFFFELL; v32 = v51 )
          {
            if ( v30 == ++v51 )
              goto LABEL_21;
            v31 = *v51;
          }
          if ( v30 == v32 )
            goto LABEL_21;
          v29 = v9[7];
        }
        if ( v35 == v36 )
        {
LABEL_46:
          v9[7] = v31;
          goto LABEL_47;
        }
        v37 = v9[2];
        v38 = *(_QWORD *)(v37 + 32);
        v39 = *(unsigned int *)(v37 + 48);
        if ( !(_DWORD)v39 )
        {
LABEL_44:
          v36 = 0;
          if ( v35 )
LABEL_45:
            v31 = sub_157EBA0(v36);
          goto LABEL_46;
        }
        v40 = v39 - 1;
        v41 = (v39 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v42 = (__int64 *)(v38 + 16LL * v41);
        v43 = *v42;
        if ( v34 == *v42 )
        {
LABEL_33:
          v44 = (__int64 *)(v38 + 16 * v39);
          if ( v44 != v42 )
          {
            v45 = v42[1];
            goto LABEL_35;
          }
        }
        else
        {
          v67 = 1;
          while ( v43 != -8 )
          {
            v141 = v67 + 1;
            v41 = v40 & (v67 + v41);
            v42 = (__int64 *)(v38 + 16LL * v41);
            v43 = *v42;
            if ( v34 == *v42 )
              goto LABEL_33;
            v67 = v141;
          }
          v44 = (__int64 *)(v38 + 16LL * (unsigned int)v39);
        }
        v45 = 0;
LABEL_35:
        v46 = v40 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v47 = (__int64 *)(v38 + 16LL * v46);
        v48 = *v47;
        if ( v35 == *v47 )
        {
LABEL_36:
          if ( v47 != v44 )
          {
            v49 = v47[1];
            if ( v49 )
            {
              while ( v45 )
              {
                if ( v45 == v49 )
                {
                  v36 = *(_QWORD *)v45;
                  goto LABEL_54;
                }
                if ( *(_DWORD *)(v45 + 16) < *(_DWORD *)(v49 + 16) )
                {
                  v50 = v45;
                  v45 = v49;
                  v49 = v50;
                }
                v45 = *(_QWORD *)(v45 + 8);
              }
            }
          }
        }
        else
        {
          v65 = 1;
          while ( v48 != -8 )
          {
            v66 = v65 + 1;
            v46 = v40 & (v65 + v46);
            v47 = (__int64 *)(v38 + 16LL * v46);
            v48 = *v47;
            if ( v35 == *v47 )
              goto LABEL_36;
            v65 = v66;
          }
        }
        goto LABEL_44;
      }
    }
LABEL_21:
    if ( v215 != (__int64 *)v217 )
      _libc_free((unsigned __int64)v215);
    v33 = (unsigned __int64)v211;
    if ( v211 != v210 )
      goto LABEL_24;
    return;
  }
LABEL_79:
  v9[7] = sub_157EBA0(v13);
  if ( v14 != (__int64 *)v217 )
    _libc_free((unsigned __int64)v14);
  v33 = (unsigned __int64)v211;
  if ( v210 != v211 )
LABEL_24:
    _libc_free(v33);
}
