// Function: sub_14B0C10
// Address: 0x14b0c10
//
unsigned __int64 __fastcall sub_14B0C10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8,
        int a9)
{
  __int64 v9; // r14
  __int64 v12; // rbx
  int v13; // r15d
  bool v14; // cl
  char v15; // dl
  _QWORD *v16; // rsi
  _QWORD **v17; // rdi
  bool v18; // al
  __int64 v19; // rdx
  _BOOL8 v20; // rcx
  int v21; // eax
  int v22; // eax
  int v23; // eax
  unsigned int v24; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // al
  unsigned int v29; // r9d
  __int64 **v30; // rax
  unsigned int v31; // r15d
  int v32; // edx
  unsigned int v33; // r9d
  __int64 **v34; // rax
  unsigned int v35; // edx
  int v36; // eax
  _QWORD *v37; // rax
  __int64 v38; // rax
  unsigned __int8 v39; // al
  unsigned int v40; // r15d
  bool v41; // al
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 v44; // rdx
  char v45; // al
  __int64 v46; // rax
  __int64 v47; // rdx
  unsigned __int8 v48; // al
  __int64 v49; // r11
  bool v50; // zf
  __int64 v51; // rax
  _BYTE *v52; // rcx
  _QWORD *v53; // r15
  _QWORD *v54; // rdx
  _BYTE *v55; // rsi
  int v56; // eax
  int v57; // eax
  __int64 v58; // rax
  int v59; // eax
  unsigned int v60; // eax
  unsigned __int8 v61; // al
  unsigned int v62; // r15d
  bool v63; // al
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned int v67; // eax
  __int64 *v68; // rdx
  __int64 *v69; // rcx
  unsigned __int8 v70; // al
  __int64 v71; // r14
  char v72; // al
  unsigned __int64 v73; // r15
  __int64 v74; // rax
  _QWORD *v75; // rax
  unsigned __int64 v76; // rax
  __int64 v77; // rcx
  int v78; // eax
  _QWORD *v79; // rdx
  __int64 *v80; // r12
  unsigned int v81; // edx
  unsigned int v82; // r15d
  unsigned int v83; // eax
  __int64 *v84; // r8
  __int64 *v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rax
  char v88; // al
  __int64 v89; // rax
  __int64 v90; // rax
  unsigned int v91; // r15d
  unsigned __int64 v92; // rax
  void *v93; // rdx
  __int64 v94; // rax
  unsigned int v95; // r15d
  unsigned __int16 v96; // cx
  unsigned int v97; // r15d
  bool v98; // al
  unsigned int v99; // eax
  unsigned int v100; // eax
  __int64 v101; // rax
  unsigned int v102; // edx
  __int64 v103; // rax
  char v104; // al
  char v105; // al
  char v106; // al
  unsigned int v107; // r15d
  __int64 v108; // rax
  char v109; // dl
  __int64 v110; // rsi
  __int64 v111; // rdx
  __int64 v112; // rcx
  unsigned int v113; // r8d
  _BYTE *v114; // rcx
  _QWORD *v115; // r15
  _QWORD *v116; // rdi
  _BYTE *v117; // rsi
  int v118; // eax
  char v119; // al
  unsigned int v120; // r15d
  __int64 v121; // rax
  char v122; // dl
  bool v123; // al
  _BYTE *v124; // rcx
  _QWORD *v125; // r15
  _QWORD *v126; // rdi
  _BYTE *v127; // rsi
  int v128; // eax
  char v129; // al
  _BYTE *v130; // rcx
  _QWORD *v131; // r15
  _QWORD *v132; // rdi
  _BYTE *v133; // rsi
  int v134; // eax
  char v135; // al
  char v136; // al
  __int64 *v137; // r14
  unsigned int v138; // eax
  unsigned int v139; // ecx
  char v140; // al
  char v141; // al
  unsigned int v142; // r15d
  __int64 v143; // rax
  char v144; // cl
  bool v145; // al
  __int64 v146; // rax
  char v147; // dl
  unsigned int v148; // r15d
  char v150; // al
  __int64 v151; // rdi
  int v152; // eax
  char v153; // al
  __int64 v154; // rdi
  int v155; // eax
  char v156; // al
  __int64 v157; // rdi
  int v158; // eax
  __int64 v159; // [rsp+8h] [rbp-D8h]
  __int64 v160; // [rsp+8h] [rbp-D8h]
  __int64 v161; // [rsp+8h] [rbp-D8h]
  char v162; // [rsp+10h] [rbp-D0h]
  int v163; // [rsp+10h] [rbp-D0h]
  __int64 v164; // [rsp+10h] [rbp-D0h]
  __int64 v165; // [rsp+10h] [rbp-D0h]
  __int64 v166; // [rsp+10h] [rbp-D0h]
  bool v167; // [rsp+18h] [rbp-C8h]
  char v168; // [rsp+18h] [rbp-C8h]
  char v169; // [rsp+18h] [rbp-C8h]
  __int64 v170; // [rsp+18h] [rbp-C8h]
  __int64 v171; // [rsp+18h] [rbp-C8h]
  __int64 *v172; // [rsp+18h] [rbp-C8h]
  __int64 *v173; // [rsp+18h] [rbp-C8h]
  __int64 *v174; // [rsp+18h] [rbp-C8h]
  int v175; // [rsp+18h] [rbp-C8h]
  int v176; // [rsp+18h] [rbp-C8h]
  __int64 v177; // [rsp+18h] [rbp-C8h]
  __int64 v178; // [rsp+18h] [rbp-C8h]
  __int64 *v179; // [rsp+18h] [rbp-C8h]
  __int64 *v180; // [rsp+18h] [rbp-C8h]
  __int64 *v181; // [rsp+18h] [rbp-C8h]
  int v182; // [rsp+18h] [rbp-C8h]
  int v183; // [rsp+18h] [rbp-C8h]
  __int64 v184; // [rsp+18h] [rbp-C8h]
  __int64 v185; // [rsp+18h] [rbp-C8h]
  __int64 v186; // [rsp+18h] [rbp-C8h]
  unsigned int v187; // [rsp+18h] [rbp-C8h]
  __int64 v188; // [rsp+18h] [rbp-C8h]
  unsigned int v189; // [rsp+18h] [rbp-C8h]
  __int64 v190; // [rsp+18h] [rbp-C8h]
  unsigned int v191; // [rsp+18h] [rbp-C8h]
  char v192; // [rsp+20h] [rbp-C0h]
  unsigned int v193; // [rsp+20h] [rbp-C0h]
  unsigned int v194; // [rsp+20h] [rbp-C0h]
  unsigned int v195; // [rsp+20h] [rbp-C0h]
  char v196; // [rsp+20h] [rbp-C0h]
  char v197; // [rsp+20h] [rbp-C0h]
  __int64 *v198; // [rsp+20h] [rbp-C0h]
  __int64 *v199; // [rsp+20h] [rbp-C0h]
  __int64 *v200; // [rsp+20h] [rbp-C0h]
  __int64 *v201; // [rsp+20h] [rbp-C0h]
  __int64 *v202; // [rsp+20h] [rbp-C0h]
  int v203; // [rsp+20h] [rbp-C0h]
  int v204; // [rsp+20h] [rbp-C0h]
  unsigned int v205; // [rsp+20h] [rbp-C0h]
  unsigned int v206; // [rsp+20h] [rbp-C0h]
  __int64 *v207; // [rsp+20h] [rbp-C0h]
  unsigned int v208; // [rsp+20h] [rbp-C0h]
  __int64 *v209; // [rsp+20h] [rbp-C0h]
  __int64 *v210; // [rsp+20h] [rbp-C0h]
  unsigned int v211; // [rsp+20h] [rbp-C0h]
  unsigned int v212; // [rsp+20h] [rbp-C0h]
  unsigned int v213; // [rsp+20h] [rbp-C0h]
  __int64 v214; // [rsp+20h] [rbp-C0h]
  unsigned int v215; // [rsp+20h] [rbp-C0h]
  __int64 v216; // [rsp+20h] [rbp-C0h]
  unsigned int v217; // [rsp+20h] [rbp-C0h]
  __int64 v218; // [rsp+20h] [rbp-C0h]
  unsigned int v219; // [rsp+2Ch] [rbp-B4h]
  __int64 *v220; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v221; // [rsp+48h] [rbp-98h] BYREF
  __int64 *v222; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v223; // [rsp+58h] [rbp-88h] BYREF
  __int64 **v224; // [rsp+60h] [rbp-80h] BYREF
  int v225; // [rsp+68h] [rbp-78h]
  __int64 **v226; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v227; // [rsp+78h] [rbp-68h]
  __int64 **v228; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v229; // [rsp+88h] [rbp-58h]
  unsigned __int64 v230; // [rsp+90h] [rbp-50h] BYREF
  _QWORD *v231; // [rsp+98h] [rbp-48h] BYREF
  __int64 v232; // [rsp+A0h] [rbp-40h]
  __int64 ***v233; // [rsp+A8h] [rbp-38h]

  v9 = a4;
  v12 = a3;
  v219 = a1;
  v192 = a2;
  *a7 = a3;
  *a8 = a4;
  if ( (unsigned int)a1 <= 0xD && (a4 = (unsigned int)a1, ((1LL << a1) & 0x2828) != 0) )
  {
    a2 &= 8u;
    if ( !(_DWORD)a2 )
    {
      if ( *(_BYTE *)(a3 + 16) == 14 )
      {
        v42 = *(_QWORD *)(a3 + 32);
        v43 = sub_16982C0(a1, a2, a3, (unsigned int)a1);
        v44 = v12 + 32;
        if ( v42 == v43 )
        {
          a2 = *(_QWORD *)(v12 + 40);
          v44 = a2 + 8;
        }
        a3 = *(_BYTE *)(v44 + 18) & 7;
        if ( (_BYTE)a3 != 3 )
        {
          if ( (v192 & 2) == 0 )
            goto LABEL_71;
LABEL_64:
          v13 = 3;
          goto LABEL_4;
        }
      }
      if ( *(_BYTE *)(v9 + 16) != 14 )
        goto LABEL_18;
      v26 = *(_QWORD *)(v9 + 32) == sub_16982C0(a1, a2, a3, a4) ? *(_QWORD *)(v9 + 40) + 8LL : v9 + 32;
      if ( (*(_BYTE *)(v26 + 18) & 7) == 3 )
        goto LABEL_18;
    }
  }
  else
  {
    v13 = 0;
    if ( (unsigned int)a1 > 0xF )
      goto LABEL_4;
  }
  if ( (v192 & 2) != 0 )
    goto LABEL_64;
  a3 = 0;
  if ( *(_BYTE *)(v12 + 16) != 14 )
  {
    if ( *(_BYTE *)(v9 + 16) == 14 )
      goto LABEL_29;
LABEL_65:
    v169 = a3;
    v45 = sub_15FF800((unsigned int)a1);
    v15 = v169;
    if ( v45 )
    {
      if ( v169 )
        goto LABEL_67;
    }
    else if ( v169 )
    {
      goto LABEL_127;
    }
LABEL_18:
    v230 = 0;
    LOBYTE(v231) = 0;
    return v230;
  }
  v42 = *(_QWORD *)(v12 + 32);
  v43 = sub_16982C0(a1, a2, 0, a4);
LABEL_71:
  if ( v42 == v43 )
    v46 = *(_QWORD *)(v12 + 40) + 8LL;
  else
    v46 = v12 + 32;
  LOBYTE(a3) = (*(_BYTE *)(v46 + 18) & 7) != 1;
  if ( *(_BYTE *)(v9 + 16) != 14 )
    goto LABEL_65;
LABEL_29:
  v168 = a3;
  if ( *(_QWORD *)(v9 + 32) == sub_16982C0(a1, a2, a3, a4) )
    v27 = *(_QWORD *)(v9 + 40) + 8LL;
  else
    v27 = v9 + 32;
  if ( (*(_BYTE *)(v27 + 18) & 7) != 1 )
  {
    v13 = 3;
    if ( v168 )
    {
LABEL_4:
      v14 = a6 == v12;
      v15 = a6 == v12 && a5 == v9;
      if ( v15 )
      {
        v60 = sub_15FF5D0((unsigned int)a1);
        v15 = a6 == v12 && a5 == v9;
        v219 = v60;
        goto LABEL_105;
      }
      goto LABEL_5;
    }
  }
  v162 = *(_BYTE *)(v27 + 18) & 7;
  v28 = sub_15FF800((unsigned int)a1);
  v15 = v168;
  if ( !v28 )
  {
    if ( !v168 )
    {
      if ( v162 == 1 )
        goto LABEL_18;
LABEL_67:
      v14 = a6 == v12;
      if ( a5 == v9 && a6 == v12 )
      {
        v196 = v15;
        v13 = 2;
        v219 = sub_15FF5D0((unsigned int)a1);
        v15 = v196 ^ 1;
        goto LABEL_105;
      }
      v13 = 1;
      goto LABEL_5;
    }
LABEL_127:
    v15 = 0;
    goto LABEL_128;
  }
  if ( v168 )
    goto LABEL_67;
  v15 = v28;
  if ( v162 == 1 )
    goto LABEL_18;
LABEL_128:
  v14 = a6 == v12;
  if ( a5 == v9 && a6 == v12 )
  {
    v197 = v15;
    v13 = 1;
    v219 = sub_15FF5D0((unsigned int)a1);
    v15 = v197 ^ 1;
    goto LABEL_105;
  }
  v13 = 2;
LABEL_5:
  if ( a5 != v12 )
  {
    v16 = (_QWORD *)a6;
    v17 = (_QWORD **)a5;
    v167 = v14;
    v18 = sub_14B0710(a5, a6, 0);
    v20 = v167;
    if ( !v18 )
      goto LABEL_15;
    v21 = *(unsigned __int8 *)(a5 + 16);
    if ( (unsigned __int8)v21 > 0x17u )
    {
      v36 = v21 - 24;
    }
    else
    {
      if ( (_BYTE)v21 != 5 )
      {
LABEL_9:
        if ( v20 )
        {
LABEL_109:
          v17 = (_QWORD **)&v230;
          v16 = (_QWORD *)v12;
          v231 = (_QWORD *)a5;
          *a7 = a6;
          *a8 = a5;
          if ( sub_13D52E0((__int64)&v230, v12) )
          {
            v19 = *a8;
            v65 = *a7;
            *a7 = *a8;
            v16 = a8;
            *a8 = v65;
          }
          if ( v219 != 38 )
          {
            if ( v219 == 40 )
            {
              if ( (unsigned __int8)sub_14A95E0(v9) )
              {
LABEL_118:
                v230 = 7;
                LOBYTE(v231) = 0;
                return v230;
              }
              v61 = *(_BYTE *)(v9 + 16);
              if ( v61 == 13 )
              {
                v62 = *(_DWORD *)(v9 + 32);
                if ( v62 <= 0x40 )
                  v63 = *(_QWORD *)(v9 + 24) == 1;
                else
                  v63 = v62 - 1 == (unsigned int)sub_16A57B0(v9 + 24);
              }
              else
              {
                if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 || v61 > 0x10u )
                  goto LABEL_39;
                v94 = sub_15A1020(v9);
                if ( !v94 || *(_BYTE *)(v94 + 16) != 13 )
                {
                  v120 = 0;
                  v204 = *(_DWORD *)(*(_QWORD *)v9 + 32LL);
                  while ( v204 != v120 )
                  {
                    v121 = sub_15A0A60(v9, v120);
                    if ( !v121 )
                      goto LABEL_39;
                    v122 = *(_BYTE *)(v121 + 16);
                    if ( v122 != 9 )
                    {
                      if ( v122 != 13 )
                        goto LABEL_39;
                      if ( *(_DWORD *)(v121 + 32) <= 0x40u )
                      {
                        v123 = *(_QWORD *)(v121 + 24) == 1;
                      }
                      else
                      {
                        v176 = *(_DWORD *)(v121 + 32);
                        v123 = v176 - 1 == (unsigned int)sub_16A57B0(v121 + 24);
                      }
                      if ( !v123 )
                        goto LABEL_39;
                    }
                    ++v120;
                  }
                  goto LABEL_118;
                }
                v95 = *(_DWORD *)(v94 + 32);
                if ( v95 <= 0x40 )
                  v63 = *(_QWORD *)(v94 + 24) == 1;
                else
                  v63 = v95 - 1 == (unsigned int)sub_16A57B0(v94 + 24);
              }
              if ( v63 )
                goto LABEL_118;
LABEL_39:
              *a7 = a5;
              *a8 = a6;
              if ( a5 == v9 )
              {
                v29 = v219;
                v47 = a6;
              }
              else
              {
                v29 = sub_15FF5D0(v219);
                if ( a6 != v9 )
                  goto LABEL_41;
                v47 = a5;
              }
              v48 = *(_BYTE *)(v9 + 16);
              v49 = v9 + 24;
              if ( v48 != 13 )
              {
                if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
                  goto LABEL_41;
                if ( v48 > 0x10u )
                  goto LABEL_41;
                v171 = v47;
                v195 = v29;
                v66 = sub_15A1020(v9);
                if ( !v66 || *(_BYTE *)(v66 + 16) != 13 )
                  goto LABEL_41;
                v47 = v171;
                v29 = v195;
                v49 = v66 + 24;
              }
              v50 = *(_BYTE *)(v47 + 16) == 79;
              v230 = v12;
              v231 = &v228;
              if ( !v50 )
                goto LABEL_41;
              v51 = *(_QWORD *)(v47 - 72);
              if ( *(_BYTE *)(v51 + 16) != 75 )
                goto LABEL_79;
              v114 = *(_BYTE **)(v47 - 48);
              v115 = *(_QWORD **)(v51 - 48);
              v116 = *(_QWORD **)(v47 - 24);
              v117 = *(_BYTE **)(v51 - 24);
              if ( v114 == (_BYTE *)v115 && v116 == (_QWORD *)v117 )
              {
                v118 = *(unsigned __int16 *)(v51 + 18);
              }
              else
              {
                if ( v114 != v117 || v116 != v115 )
                {
LABEL_79:
                  v230 = v12;
                  v231 = &v228;
                  goto LABEL_80;
                }
                v118 = *(unsigned __int16 *)(v51 + 18);
                if ( v114 != (_BYTE *)v115 )
                {
                  v160 = v47;
                  v178 = v49;
                  v206 = v29;
                  v118 = sub_15FF0F0(v118 & 0xFFFF7FFF);
                  v47 = v160;
                  v49 = v178;
                  v29 = v206;
LABEL_322:
                  if ( (unsigned int)(v118 - 40) <= 1 && v115 == (_QWORD *)v230 )
                  {
                    v164 = v47;
                    v186 = v49;
                    v213 = v29;
                    v150 = sub_13D2630(&v231, v117);
                    v29 = v213;
                    v49 = v186;
                    v47 = v164;
                    if ( v150 )
                    {
                      v151 = v186;
                      v187 = v213;
                      v214 = v49;
                      v152 = sub_16AEA10(v151, v228);
                      v49 = v214;
                      v29 = v187;
                      v47 = v164;
                      if ( v152 < 0 )
                      {
                        v31 = 3;
                        if ( v187 == 40 )
                          goto LABEL_46;
                      }
                    }
                  }
                  v119 = *(_BYTE *)(v47 + 16);
                  v230 = v12;
                  v231 = &v228;
                  if ( v119 != 79 )
                    goto LABEL_41;
                  v51 = *(_QWORD *)(v47 - 72);
LABEL_80:
                  if ( *(_BYTE *)(v51 + 16) != 75 )
                  {
LABEL_81:
                    v230 = v12;
                    v231 = &v228;
                    goto LABEL_82;
                  }
                  v130 = *(_BYTE **)(v47 - 48);
                  v131 = *(_QWORD **)(v51 - 48);
                  v132 = *(_QWORD **)(v47 - 24);
                  v133 = *(_BYTE **)(v51 - 24);
                  if ( v130 == (_BYTE *)v131 && v132 == (_QWORD *)v133 )
                  {
                    v134 = *(unsigned __int16 *)(v51 + 18);
                  }
                  else
                  {
                    if ( v130 != v133 || v132 != v131 )
                      goto LABEL_81;
                    v134 = *(unsigned __int16 *)(v51 + 18);
                    if ( v130 != (_BYTE *)v131 )
                    {
                      v159 = v47;
                      v177 = v49;
                      v205 = v29;
                      v134 = sub_15FF0F0(v134 & 0xFFFF7FFF);
                      v47 = v159;
                      v49 = v177;
                      v29 = v205;
LABEL_362:
                      if ( (unsigned int)(v134 - 38) <= 1 && v131 == (_QWORD *)v230 )
                      {
                        v165 = v47;
                        v188 = v49;
                        v215 = v29;
                        v153 = sub_13D2630(&v231, v133);
                        v29 = v215;
                        v49 = v188;
                        v47 = v165;
                        if ( v153 )
                        {
                          v154 = v188;
                          v189 = v215;
                          v216 = v49;
                          v155 = sub_16AEA10(v154, v228);
                          v29 = v189;
                          v49 = v216;
                          v47 = v165;
                          if ( v189 == 38 )
                          {
                            v31 = 1;
                            if ( v155 > 0 )
                              goto LABEL_46;
                          }
                        }
                      }
                      v135 = *(_BYTE *)(v47 + 16);
                      v230 = v12;
                      v231 = &v228;
                      if ( v135 != 79 )
                        goto LABEL_41;
                      v51 = *(_QWORD *)(v47 - 72);
LABEL_82:
                      if ( *(_BYTE *)(v51 + 16) != 75 )
                      {
LABEL_83:
                        v230 = v12;
                        v231 = &v228;
                        goto LABEL_84;
                      }
                      v124 = *(_BYTE **)(v47 - 48);
                      v125 = *(_QWORD **)(v51 - 48);
                      v126 = *(_QWORD **)(v47 - 24);
                      v127 = *(_BYTE **)(v51 - 24);
                      if ( v124 == (_BYTE *)v125 && v126 == (_QWORD *)v127 )
                      {
                        v128 = *(unsigned __int16 *)(v51 + 18);
                      }
                      else
                      {
                        if ( v124 != v127 || v126 != v125 )
                          goto LABEL_83;
                        v128 = *(unsigned __int16 *)(v51 + 18);
                        if ( v124 != (_BYTE *)v125 )
                        {
                          v161 = v47;
                          v184 = v49;
                          v211 = v29;
                          v128 = sub_15FF0F0(v128 & 0xFFFF7FFF);
                          v47 = v161;
                          v49 = v184;
                          v29 = v211;
LABEL_352:
                          if ( (unsigned int)(v128 - 36) <= 1 && v125 == (_QWORD *)v230 )
                          {
                            v166 = v47;
                            v190 = v49;
                            v217 = v29;
                            v156 = sub_13D2630(&v231, v127);
                            v29 = v217;
                            v49 = v190;
                            v47 = v166;
                            if ( v156 )
                            {
                              v157 = v190;
                              v191 = v217;
                              v218 = v49;
                              v158 = sub_16A9900(v157, v228);
                              v49 = v218;
                              v29 = v191;
                              v47 = v166;
                              if ( v158 < 0 )
                              {
                                v31 = 4;
                                if ( v191 == 36 )
                                  goto LABEL_46;
                              }
                            }
                          }
                          v129 = *(_BYTE *)(v47 + 16);
                          v230 = v12;
                          v231 = &v228;
                          if ( v129 != 79 )
                            goto LABEL_41;
                          v51 = *(_QWORD *)(v47 - 72);
LABEL_84:
                          if ( *(_BYTE *)(v51 + 16) != 75 )
                            goto LABEL_41;
                          v52 = *(_BYTE **)(v47 - 48);
                          v53 = *(_QWORD **)(v51 - 48);
                          v54 = *(_QWORD **)(v47 - 24);
                          v55 = *(_BYTE **)(v51 - 24);
                          if ( v52 == (_BYTE *)v53 && v54 == (_QWORD *)v55 )
                          {
                            v56 = *(unsigned __int16 *)(v51 + 18);
                          }
                          else
                          {
                            if ( v52 != v55 || v54 != v53 )
                              goto LABEL_41;
                            v56 = *(unsigned __int16 *)(v51 + 18);
                            if ( v52 != (_BYTE *)v53 )
                            {
                              v185 = v49;
                              v212 = v29;
                              v56 = sub_15FF0F0(v56 & 0xFFFF7FFF);
                              v49 = v185;
                              v29 = v212;
                              goto LABEL_89;
                            }
                          }
                          BYTE1(v56) &= ~0x80u;
LABEL_89:
                          if ( (unsigned int)(v56 - 34) <= 1 && v53 == (_QWORD *)v230 )
                          {
                            v170 = v49;
                            v194 = v29;
                            if ( (unsigned __int8)sub_13D2630(&v231, v55) )
                            {
                              v57 = sub_16A9900(v170, v228);
                              if ( v194 == 34 )
                              {
                                v31 = 2;
                                if ( v57 > 0 )
                                  goto LABEL_46;
                              }
                            }
                          }
LABEL_41:
                          v30 = (__int64 **)sub_14B2890(a5, &v220, &v221, 0, (unsigned int)(a9 + 1));
                          v224 = v30;
                          v31 = (unsigned int)v30;
                          v225 = v32;
                          v33 = v219 - 38;
                          if ( (unsigned int)((_DWORD)v30 - 7) <= 1 )
                            goto LABEL_44;
                          if ( !(_DWORD)v30 )
                            goto LABEL_44;
                          v34 = (__int64 **)sub_14B2890(a6, &v222, &v223, 0, (unsigned int)(a9 + 1));
                          v33 = v219 - 38;
                          v226 = v34;
                          v227 = v35;
                          if ( v31 != (_DWORD)v34 )
                            goto LABEL_44;
                          if ( v31 == 3 )
                          {
                            if ( v219 - 40 <= 1 )
                            {
                              v100 = sub_15FF5D0(v219);
                              v68 = (__int64 *)v12;
                              v69 = (__int64 *)v9;
                            }
                            else
                            {
                              v100 = v219;
                              v68 = (__int64 *)v9;
                              v69 = (__int64 *)v12;
                            }
                            if ( v100 - 38 > 1 )
                              goto LABEL_146;
                          }
                          else if ( v31 > 3 )
                          {
                            if ( v31 != 4 )
                              goto LABEL_44;
                            if ( v219 - 36 <= 1 )
                            {
                              v83 = sub_15FF5D0(v219);
                              v68 = (__int64 *)v12;
                              v69 = (__int64 *)v9;
                            }
                            else
                            {
                              v83 = v219;
                              v68 = (__int64 *)v9;
                              v69 = (__int64 *)v12;
                            }
                            v33 = v219 - 38;
                            if ( v83 - 34 > 1 )
                              goto LABEL_44;
                          }
                          else if ( v31 == 1 )
                          {
                            if ( v33 <= 1 )
                            {
                              v99 = sub_15FF5D0(v219);
                              v33 = v219 - 38;
                              v68 = (__int64 *)v12;
                              v69 = (__int64 *)v9;
                            }
                            else
                            {
                              v99 = v219;
                              v68 = (__int64 *)v9;
                              v69 = (__int64 *)v12;
                            }
                            if ( v99 - 40 > 1 )
                              goto LABEL_44;
                          }
                          else
                          {
                            if ( v219 - 34 <= 1 )
                            {
                              v67 = sub_15FF5D0(v219);
                              v68 = (__int64 *)v12;
                              v69 = (__int64 *)v9;
                            }
                            else
                            {
                              v67 = v219;
                              v68 = (__int64 *)v9;
                              v69 = (__int64 *)v12;
                            }
                            if ( v67 - 36 > 1 )
                            {
LABEL_146:
                              v33 = v219 - 38;
                              goto LABEL_44;
                            }
                          }
                          v84 = v223;
                          v85 = v221;
                          if ( v223 == v221 )
                          {
                            if ( v69 == v220 && v68 == v222 )
                              goto LABEL_46;
                            v172 = v68;
                            v230 = (unsigned __int64)v69;
                            v200 = v69;
                            v104 = sub_13D1F50((__int64 *)&v230, (__int64)v222);
                            v69 = v200;
                            v68 = v172;
                            if ( v104 )
                            {
                              v228 = (__int64 **)v172;
                              v181 = v200;
                              v210 = v68;
                              v141 = sub_13D1F50((__int64 *)&v228, (__int64)v220);
                              v68 = v210;
                              v69 = v181;
                              if ( v141 )
                                goto LABEL_46;
                            }
                            v85 = v221;
                            v84 = v223;
                          }
                          v86 = (__int64)v222;
                          if ( v85 == v222 )
                          {
                            if ( v69 == v220 && v68 == v84 )
                              goto LABEL_46;
                            v230 = (unsigned __int64)v69;
                            v174 = v68;
                            v202 = v69;
                            v106 = sub_13D1F50((__int64 *)&v230, (__int64)v84);
                            v69 = v202;
                            v68 = v174;
                            if ( v106 )
                            {
                              v228 = (__int64 **)v174;
                              v180 = v202;
                              v209 = v68;
                              v140 = sub_13D1F50((__int64 *)&v228, (__int64)v220);
                              v68 = v209;
                              v69 = v180;
                              if ( v140 )
                                goto LABEL_46;
                            }
                            v84 = v223;
                            v86 = (__int64)v222;
                          }
                          v87 = (__int64)v220;
                          if ( v220 == v84 )
                          {
                            if ( v69 == v221 && (__int64 *)v86 == v68 )
                              goto LABEL_46;
                            v173 = v68;
                            v230 = (unsigned __int64)v69;
                            v201 = v69;
                            v105 = sub_13D1F50((__int64 *)&v230, v86);
                            v69 = v201;
                            v68 = v173;
                            if ( v105 )
                            {
                              v228 = (__int64 **)v173;
                              v179 = v201;
                              v207 = v68;
                              v136 = sub_13D1F50((__int64 *)&v228, (__int64)v221);
                              v68 = v207;
                              v69 = v179;
                              if ( v136 )
                                goto LABEL_46;
                            }
                            v86 = (__int64)v222;
                            v87 = (__int64)v220;
                          }
                          v33 = v219 - 38;
                          if ( v86 == v87 )
                          {
                            if ( v69 == v221 && v68 == v223 )
                              goto LABEL_46;
                            v198 = v68;
                            v230 = (unsigned __int64)v69;
                            v88 = sub_13D1F50((__int64 *)&v230, (__int64)v223);
                            v33 = v219 - 38;
                            if ( v88 )
                            {
                              v228 = (__int64 **)v198;
                              if ( sub_13D1F50((__int64 *)&v228, (__int64)v221) )
                                goto LABEL_46;
                              v33 = v219 - 38;
                            }
                          }
LABEL_44:
                          v193 = v33 & 0xFFFFFFFD;
                          if ( (v33 & 0xFFFFFFFD) != 0 )
                          {
LABEL_45:
                            v31 = 0;
LABEL_46:
                            v230 = v31;
                            LOBYTE(v231) = 0;
                            return v230;
                          }
                          if ( *(_BYTE *)(a5 + 16) > 0x10u )
                            goto LABEL_201;
                          if ( (unsigned __int8)sub_1593BB0(a5) )
                            goto LABEL_172;
                          if ( *(_BYTE *)(a5 + 16) == 13 )
                          {
                            v82 = *(_DWORD *)(a5 + 32);
                            if ( v82 <= 0x40 )
                            {
                              if ( *(_QWORD *)(a5 + 24) )
                                goto LABEL_201;
                            }
                            else if ( v82 != (unsigned int)sub_16A57B0(a5 + 24) )
                            {
LABEL_201:
                              LOBYTE(v73) = *(_BYTE *)(a6 + 16);
                              goto LABEL_178;
                            }
                          }
                          else
                          {
                            if ( *(_BYTE *)(*(_QWORD *)a5 + 8LL) != 16 )
                              goto LABEL_201;
                            v101 = sub_15A1020(a5);
                            if ( v101 && *(_BYTE *)(v101 + 16) == 13 )
                            {
                              if ( !sub_13D01C0(v101 + 24) )
                                goto LABEL_201;
                            }
                            else
                            {
                              v142 = 0;
                              v182 = *(_DWORD *)(*(_QWORD *)a5 + 32LL);
                              while ( v182 != v142 )
                              {
                                v143 = sub_15A0A60(a5, v142);
                                if ( !v143 )
                                  goto LABEL_201;
                                v144 = *(_BYTE *)(v143 + 16);
                                if ( v144 != 9 )
                                {
                                  if ( v144 != 13 )
                                    goto LABEL_201;
                                  if ( *(_DWORD *)(v143 + 32) <= 0x40u )
                                  {
                                    v145 = *(_QWORD *)(v143 + 24) == 0;
                                  }
                                  else
                                  {
                                    v163 = *(_DWORD *)(v143 + 32);
                                    v145 = v163 == (unsigned int)sub_16A57B0(v143 + 24);
                                  }
                                  if ( !v145 )
                                    goto LABEL_201;
                                }
                                ++v142;
                              }
                            }
                          }
LABEL_172:
                          v73 = *(unsigned __int8 *)(a6 + 16);
                          if ( (unsigned __int8)v73 <= 0x17u )
                          {
                            if ( (_BYTE)v73 != 5 )
                              goto LABEL_178;
                            v92 = *(unsigned __int16 *)(a6 + 18);
                            if ( (unsigned __int16)v92 > 0x17u )
                              goto LABEL_179;
                            v93 = &loc_80A800;
                            if ( !_bittest64((const __int64 *)&v93, v92) || (_WORD)v92 != 13 )
                              goto LABEL_179;
                          }
                          else
                          {
                            if ( (unsigned __int8)v73 > 0x2Fu )
                              goto LABEL_189;
                            v74 = 0x80A800000000LL;
                            if ( !_bittest64(&v74, v73) || (_BYTE)v73 != 37 )
                              goto LABEL_189;
                          }
                          if ( (*(_BYTE *)(a6 + 17) & 4) != 0 )
                          {
                            v75 = (_QWORD *)sub_13CF970(a6);
                            if ( v12 == *v75 && v9 == v75[3] )
                              goto LABEL_278;
                          }
LABEL_178:
                          if ( (unsigned __int8)v73 > 0x10u )
                            goto LABEL_189;
LABEL_179:
                          if ( !(unsigned __int8)sub_1593BB0(a6) )
                          {
                            if ( *(_BYTE *)(a6 + 16) == 13 )
                            {
                              v97 = *(_DWORD *)(a6 + 32);
                              if ( v97 <= 0x40 )
                                v98 = *(_QWORD *)(a6 + 24) == 0;
                              else
                                v98 = v97 == (unsigned int)sub_16A57B0(a6 + 24);
                              if ( !v98 )
                                goto LABEL_189;
                            }
                            else
                            {
                              if ( *(_BYTE *)(*(_QWORD *)a6 + 8LL) != 16 )
                                goto LABEL_189;
                              v103 = sub_15A1020(a6);
                              if ( v103 && *(_BYTE *)(v103 + 16) == 13 )
                              {
                                if ( !sub_13D01C0(v103 + 24) )
                                  goto LABEL_189;
                              }
                              else
                              {
                                v183 = *(_DWORD *)(*(_QWORD *)a6 + 32LL);
                                while ( v183 != v193 )
                                {
                                  v146 = sub_15A0A60(a6, v193);
                                  if ( !v146 )
                                    goto LABEL_189;
                                  v147 = *(_BYTE *)(v146 + 16);
                                  if ( v147 != 9 )
                                  {
                                    if ( v147 != 13 )
                                      goto LABEL_189;
                                    v148 = *(_DWORD *)(v146 + 32);
                                    if ( !(v148 <= 0x40
                                         ? *(_QWORD *)(v146 + 24) == 0
                                         : v148 == (unsigned int)sub_16A57B0(v146 + 24)) )
                                      goto LABEL_189;
                                  }
                                  ++v193;
                                }
                              }
                            }
                          }
                          v76 = *(unsigned __int8 *)(a5 + 16);
                          if ( (unsigned __int8)v76 <= 0x17u )
                          {
                            if ( (_BYTE)v76 != 5 )
                              goto LABEL_189;
                            v96 = *(_WORD *)(a5 + 18);
                            if ( v96 > 0x17u )
                              goto LABEL_189;
                            v78 = v96;
                            if ( (((unsigned __int64)&loc_80A800 >> v96) & 1) == 0 )
                              goto LABEL_189;
                          }
                          else
                          {
                            if ( (unsigned __int8)v76 > 0x2Fu )
                              goto LABEL_189;
                            v77 = 0x80A800000000LL;
                            if ( !_bittest64(&v77, v76) )
                              goto LABEL_189;
                            v78 = (unsigned __int8)v76 - 24;
                          }
                          if ( v78 == 13 && (*(_BYTE *)(a5 + 17) & 4) != 0 )
                          {
                            v79 = (*(_BYTE *)(a5 + 23) & 0x40) != 0
                                ? *(_QWORD **)(a5 - 8)
                                : (_QWORD *)(a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
                            if ( v12 == *v79 && v9 == v79[3] )
                              goto LABEL_341;
                          }
LABEL_189:
                          v230 = (unsigned __int64)&v222;
                          if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v230, (_BYTE *)v9) )
                            goto LABEL_45;
                          if ( a5 == v12
                            && (v110 = a6,
                                v230 = (unsigned __int64)&v223,
                                (unsigned __int8)sub_13D2630((_QWORD **)&v230, (_BYTE *)a6))
                            || a6 == v12 && (v110 = a5, v228 = &v223, (unsigned __int8)sub_13D2630(&v228, (_BYTE *)a5)) )
                          {
                            if ( v219 == 40 )
                            {
                              if ( !sub_13D01C0((__int64)v222) )
                                goto LABEL_192;
                              v137 = v223;
                              v138 = *((_DWORD *)v223 + 2);
                              v139 = v138 - 1;
                              if ( v138 <= 0x40 )
                              {
                                if ( *v223 != (1LL << v139) - 1 )
                                  goto LABEL_192;
                              }
                              else
                              {
                                v208 = v138 - 1;
                                if ( sub_13D0200(v223, v139) || v208 != (unsigned int)sub_16A58F0(v137) )
                                  goto LABEL_192;
                              }
                              v31 = 2 * (a5 == v12) + 2;
                              goto LABEL_46;
                            }
                            if ( v219 == 38
                              && sub_1454FB0((__int64)v222)
                              && (unsigned __int8)sub_13CFF40(v223, v110, v111, v112, v113) )
                            {
                              v31 = 2 * (a6 == v12) + 2;
                              goto LABEL_46;
                            }
                          }
LABEL_192:
                          v230 = v12;
                          if ( !sub_13D1F50((__int64 *)&v230, a5)
                            || (v224 = &v223, !(unsigned __int8)sub_13D2630(&v224, (_BYTE *)a6)) )
                          {
LABEL_193:
                            v230 = v12;
                            if ( !sub_13D1F50((__int64 *)&v230, a6) )
                              goto LABEL_45;
                            v224 = &v223;
                            if ( !(unsigned __int8)sub_13D2630(&v224, (_BYTE *)a5) )
                              goto LABEL_45;
                            v80 = v223;
                            sub_13A38D0((__int64)&v226, (__int64)v222);
                            sub_13D0570((__int64)&v226);
                            v81 = v227;
                            v227 = 0;
                            v229 = v81;
                            v228 = v226;
                            if ( v81 <= 0x40 )
                            {
                              if ( v226 != (__int64 **)*v80 )
                                goto LABEL_197;
                            }
                            else if ( !(unsigned __int8)sub_16A5220(&v228, v80) )
                            {
LABEL_197:
                              sub_135E100((__int64 *)&v228);
                              sub_135E100((__int64 *)&v226);
                              goto LABEL_45;
                            }
                            sub_135E100((__int64 *)&v228);
                            sub_135E100((__int64 *)&v226);
LABEL_341:
                            v31 = 2 * (v219 == 38) + 1;
                            goto LABEL_46;
                          }
                          v199 = v223;
                          sub_13A38D0((__int64)&v226, (__int64)v222);
                          sub_13D0570((__int64)&v226);
                          v102 = v227;
                          v227 = 0;
                          v229 = v102;
                          v228 = v226;
                          if ( v102 <= 0x40 )
                          {
                            if ( v226 != (__int64 **)*v199 )
                              goto LABEL_275;
                          }
                          else if ( !(unsigned __int8)sub_16A5220(&v228, v199) )
                          {
LABEL_275:
                            sub_135E100((__int64 *)&v228);
                            sub_135E100((__int64 *)&v226);
                            goto LABEL_193;
                          }
                          sub_135E100((__int64 *)&v228);
                          sub_135E100((__int64 *)&v226);
LABEL_278:
                          v31 = 2 * (v219 != 38) + 1;
                          goto LABEL_46;
                        }
                      }
                      BYTE1(v128) &= ~0x80u;
                      goto LABEL_352;
                    }
                  }
                  BYTE1(v134) &= ~0x80u;
                  goto LABEL_362;
                }
              }
              BYTE1(v118) &= ~0x80u;
              goto LABEL_322;
            }
            goto LABEL_15;
          }
          if ( !(unsigned __int8)sub_14A95E0(v9) && !(unsigned __int8)sub_14A9710(v9) )
            goto LABEL_39;
          goto LABEL_59;
        }
        v22 = *(unsigned __int8 *)(a6 + 16);
        if ( (unsigned __int8)v22 > 0x17u )
        {
          v23 = v22 - 24;
        }
        else
        {
          if ( (_BYTE)v22 != 5 )
            goto LABEL_15;
          v23 = *(unsigned __int16 *)(a6 + 18);
        }
        if ( v23 == 38 )
        {
          v17 = (_QWORD **)a6;
          if ( v12 == *(_QWORD *)sub_13CF970(a6) )
            goto LABEL_109;
        }
LABEL_15:
        v24 = v219;
        goto LABEL_16;
      }
      v36 = *(unsigned __int16 *)(a5 + 18);
    }
    if ( v36 != 38 )
      goto LABEL_9;
    v17 = (_QWORD **)a5;
    v37 = (_QWORD *)sub_13CF970(a5);
    v20 = v167;
    if ( v12 != *v37 )
      goto LABEL_9;
LABEL_50:
    v17 = (_QWORD **)&v230;
    v16 = (_QWORD *)v12;
    v231 = (_QWORD *)a6;
    *a7 = a5;
    *a8 = a6;
    if ( sub_13D52E0((__int64)&v230, v12) )
    {
      v19 = *a8;
      v38 = *a7;
      *a7 = *a8;
      v16 = a8;
      *a8 = v38;
    }
    if ( v219 == 38 )
    {
      if ( (unsigned __int8)sub_14A95E0(v9) || (unsigned __int8)sub_14A9710(v9) )
        goto LABEL_118;
      goto LABEL_39;
    }
    v24 = v219;
    if ( v219 != 40 )
    {
LABEL_16:
      if ( v24 - 32 <= 9 )
        goto LABEL_39;
      if ( v13 == 3 )
      {
        if ( (v192 & 8) != 0
          || *(_BYTE *)(v12 + 16) == 14
          && (*(_QWORD *)(v12 + 32) == sub_16982C0(v17, v16, v19, v20)
            ? (v58 = *(_QWORD *)(v12 + 40) + 8LL)
            : (v58 = v12 + 32),
              (*(_BYTE *)(v58 + 18) & 7) != 3)
          || *(_BYTE *)(v9 + 16) == 14
          && (*(_QWORD *)(v9 + 32) == sub_16982C0(v17, v16, v19, v20)
            ? (v64 = *(_QWORD *)(v9 + 40) + 8LL)
            : (v64 = v9 + 32),
              (*(_BYTE *)(v64 + 18) & 7) != 3) )
        {
          if ( a6 == v9 )
          {
            v17 = (_QWORD **)v219;
            v219 = sub_15FF0F0(v219);
            *a7 = v9;
            *a8 = a5;
          }
          else
          {
            *a7 = a5;
            *a8 = a6;
            if ( a5 != v9 )
              goto LABEL_102;
            a5 = a6;
          }
          v70 = *(_BYTE *)(v9 + 16);
          if ( v70 == 14 )
          {
            v71 = v9 + 24;
          }
          else
          {
            if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
              goto LABEL_102;
            if ( v70 > 0x10u )
              goto LABEL_102;
            v17 = (_QWORD **)v9;
            v89 = sub_15A1020(v9);
            if ( !v89 || *(_BYTE *)(v89 + 16) != 14 )
              goto LABEL_102;
            v71 = v89 + 24;
          }
          if ( *(_QWORD *)(v71 + 8) == sub_16982C0(v17, v16, v19, v20) )
          {
            v72 = *(_BYTE *)(*(_QWORD *)(v71 + 16) + 26LL) & 7;
            if ( v72 == 1 )
              goto LABEL_102;
          }
          else
          {
            v72 = *(_BYTE *)(v71 + 26) & 7;
            if ( v72 == 1 )
              goto LABEL_102;
          }
          if ( !v72 )
            goto LABEL_102;
          if ( v219 > 0xB )
          {
            if ( v219 - 12 > 1 )
              goto LABEL_102;
          }
          else
          {
            if ( v219 > 9 )
            {
LABEL_163:
              v230 = v12;
              v231 = &v228;
              v232 = v12;
              v233 = &v228;
              if ( (unsigned __int8)sub_14B0280((__int64)&v230, a5)
                && (unsigned int)sub_14A9E40(v71, (__int64)v228) == 2 )
              {
                v59 = 5;
                goto LABEL_103;
              }
LABEL_102:
              v13 = 0;
              v59 = 0;
LABEL_103:
              v230 = __PAIR64__(v13, v59);
              LOBYTE(v231) = 0;
              return v230;
            }
            if ( v219 <= 3 )
            {
              if ( v219 <= 1 )
                goto LABEL_102;
              goto LABEL_163;
            }
            if ( v219 - 4 > 1 )
              goto LABEL_102;
          }
          v230 = v12;
          v231 = &v228;
          v232 = v12;
          v233 = &v228;
          if ( (unsigned __int8)sub_14B00C0((__int64)&v230, a5) && !(unsigned int)sub_14A9E40(v71, (__int64)v228) )
          {
            v59 = 6;
            goto LABEL_103;
          }
          goto LABEL_102;
        }
      }
      goto LABEL_18;
    }
    if ( !(unsigned __int8)sub_14A95E0(v9) )
    {
      v39 = *(_BYTE *)(v9 + 16);
      if ( v39 == 13 )
      {
        v40 = *(_DWORD *)(v9 + 32);
        if ( v40 <= 0x40 )
          v41 = *(_QWORD *)(v9 + 24) == 1;
        else
          v41 = v40 - 1 == (unsigned int)sub_16A57B0(v9 + 24);
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 || v39 > 0x10u )
          goto LABEL_39;
        v90 = sub_15A1020(v9);
        if ( !v90 || *(_BYTE *)(v90 + 16) != 13 )
        {
          v203 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
          if ( v203 )
          {
            v107 = 0;
            while ( 1 )
            {
              v108 = sub_15A0A60(v9, v107);
              if ( !v108 )
                goto LABEL_39;
              v109 = *(_BYTE *)(v108 + 16);
              if ( v109 != 9 )
              {
                if ( v109 != 13 )
                  goto LABEL_39;
                if ( *(_DWORD *)(v108 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v108 + 24) != 1 )
                    goto LABEL_39;
                }
                else
                {
                  v175 = *(_DWORD *)(v108 + 32);
                  if ( (unsigned int)sub_16A57B0(v108 + 24) != v175 - 1 )
                    goto LABEL_39;
                }
              }
              if ( v203 == ++v107 )
                goto LABEL_59;
            }
          }
          goto LABEL_59;
        }
        v91 = *(_DWORD *)(v90 + 32);
        if ( v91 <= 0x40 )
          v41 = *(_QWORD *)(v90 + 24) == 1;
        else
          v41 = v91 - 1 == (unsigned int)sub_16A57B0(v90 + 24);
      }
      if ( !v41 )
        goto LABEL_39;
    }
LABEL_59:
    v230 = 8;
    LOBYTE(v231) = 0;
    return v230;
  }
  if ( a6 != v9 )
  {
    v16 = (_QWORD *)a6;
    v17 = (_QWORD **)a5;
    if ( !sub_14B0710(a5, a6, 0) )
      goto LABEL_15;
    goto LABEL_50;
  }
LABEL_105:
  switch ( v219 )
  {
    case 2u:
    case 3u:
    case 0xAu:
    case 0xBu:
      LODWORD(v230) = 6;
      HIDWORD(v230) = v13;
      LOBYTE(v231) = v15;
      break;
    case 4u:
    case 5u:
    case 0xCu:
    case 0xDu:
      LODWORD(v230) = 5;
      HIDWORD(v230) = v13;
      LOBYTE(v231) = v15;
      break;
    case 0x22u:
    case 0x23u:
      v230 = 4;
      LOBYTE(v231) = 0;
      break;
    case 0x24u:
    case 0x25u:
      v230 = 2;
      LOBYTE(v231) = 0;
      break;
    case 0x26u:
    case 0x27u:
      v230 = 3;
      LOBYTE(v231) = 0;
      break;
    case 0x28u:
    case 0x29u:
      v230 = 1;
      LOBYTE(v231) = 0;
      break;
    default:
      goto LABEL_18;
  }
  return v230;
}
