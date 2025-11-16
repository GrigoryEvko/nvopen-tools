// Function: sub_1B4A120
// Address: 0x1b4a120
//
__int64 __fastcall sub_1B4A120(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int8 v5; // si
  _QWORD *v6; // rdx
  __int64 v7; // rbx
  void **p_base; // r12
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  int v16; // edx
  __int64 v17; // rax
  char v18; // dl
  __int16 v19; // dx
  __int64 v20; // rdx
  __int64 *v21; // rbx
  _BYTE *v22; // rdi
  _BYTE *v23; // rsi
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // r13
  char *v27; // rdi
  __int64 v29; // r10
  char v30; // r15
  __int64 v31; // rsi
  char *v32; // rsi
  char *v33; // rbx
  __int64 v34; // rcx
  char *v35; // rax
  unsigned int v36; // eax
  bool v37; // dl
  __int64 v38; // rbx
  _QWORD *v39; // r12
  _QWORD *v40; // r11
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // rdx
  __int64 *v45; // r13
  __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 *v48; // rdi
  __int64 v49; // r14
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // r13
  __int64 v54; // rsi
  __int64 i; // r13
  __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // r8
  unsigned int v59; // eax
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 v63; // rcx
  __int64 v64; // rbx
  int v65; // r15d
  int v66; // r14d
  __int64 v67; // rcx
  __int64 v68; // rdx
  _QWORD *v69; // rax
  __int64 v70; // rdi
  unsigned __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 v76; // rax
  int v77; // edx
  _QWORD *v78; // rdx
  char *v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rax
  char v82; // al
  __int64 *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  int v86; // r8d
  int v87; // r9d
  bool v88; // bl
  __int64 v89; // r15
  __int64 v90; // r8
  unsigned __int64 v91; // rbx
  char v92; // dl
  char v93; // r8
  __int64 *v94; // rdx
  __int64 *v95; // rax
  __int64 *v96; // rdx
  __int64 v97; // rsi
  char v98; // dl
  int v99; // r8d
  int v100; // r9d
  __int64 *v101; // rdi
  unsigned int v102; // ebx
  __int64 *v103; // rax
  __int64 *v104; // rsi
  __int64 *v105; // rax
  __int64 *v106; // rdi
  __int64 *v107; // rcx
  __int64 v108; // rax
  __int64 v109; // rbx
  int v110; // r8d
  int v111; // r9d
  __int64 v112; // rax
  unsigned __int8 *v113; // rdi
  __int64 v114; // rdx
  unsigned int v115; // edx
  unsigned __int64 v116; // rsi
  unsigned __int64 v117; // rax
  __int64 v118; // rdi
  unsigned __int64 v119; // rax
  const void *v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rcx
  int v123; // r8d
  int v124; // r9d
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rcx
  __int64 v128; // rdx
  _BYTE *v129; // rdi
  __int64 v130; // rdx
  unsigned int v131; // eax
  unsigned int v132; // eax
  unsigned __int8 v133; // al
  __int64 v134; // rax
  unsigned int v135; // eax
  unsigned int v136; // eax
  __int64 v137; // rdx
  __int64 v138; // rax
  unsigned int v139; // edx
  char v140; // dl
  const void *v141; // rax
  __int64 v142; // rdx
  __int64 v143; // rcx
  int v144; // r8d
  int v145; // r9d
  unsigned int v146; // edx
  __int64 v147; // rdi
  const void *v148; // rax
  __int64 *v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rcx
  int v152; // r8d
  int v153; // r9d
  __int64 *v154; // rdi
  __int64 v155; // rax
  unsigned int v156; // edx
  const void *v157; // rax
  __int64 *v158; // rax
  __int64 v159; // rdx
  __int64 v160; // rcx
  int v161; // r8d
  int v162; // r9d
  unsigned int v163; // edx
  const void **v164; // [rsp+8h] [rbp-238h]
  __int64 v165; // [rsp+10h] [rbp-230h]
  unsigned int v166; // [rsp+10h] [rbp-230h]
  const void **v167; // [rsp+10h] [rbp-230h]
  __int64 v168; // [rsp+18h] [rbp-228h]
  __int64 v169; // [rsp+18h] [rbp-228h]
  __int64 v170; // [rsp+18h] [rbp-228h]
  __int64 v171; // [rsp+20h] [rbp-220h]
  __int64 *v172; // [rsp+20h] [rbp-220h]
  __int64 *v173; // [rsp+20h] [rbp-220h]
  int v174; // [rsp+30h] [rbp-210h]
  __int64 v175; // [rsp+30h] [rbp-210h]
  __int64 v176; // [rsp+30h] [rbp-210h]
  unsigned __int8 v177; // [rsp+38h] [rbp-208h]
  unsigned __int64 v178; // [rsp+38h] [rbp-208h]
  unsigned __int64 v179; // [rsp+38h] [rbp-208h]
  __int64 v181; // [rsp+48h] [rbp-1F8h]
  int v182; // [rsp+48h] [rbp-1F8h]
  __int64 v184; // [rsp+60h] [rbp-1E0h]
  int v185; // [rsp+68h] [rbp-1D8h]
  __int64 v186; // [rsp+68h] [rbp-1D8h]
  __int64 v187; // [rsp+68h] [rbp-1D8h]
  void **v188; // [rsp+68h] [rbp-1D8h]
  __int64 v189; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v190; // [rsp+78h] [rbp-1C8h] BYREF
  unsigned __int64 v191; // [rsp+80h] [rbp-1C0h] BYREF
  unsigned int v192; // [rsp+88h] [rbp-1B8h]
  __int64 v193; // [rsp+90h] [rbp-1B0h] BYREF
  unsigned int v194; // [rsp+98h] [rbp-1A8h]
  __int64 v195; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned int v196; // [rsp+A8h] [rbp-198h]
  const void *v197; // [rsp+B0h] [rbp-190h] BYREF
  unsigned int v198; // [rsp+B8h] [rbp-188h]
  const void *v199; // [rsp+C0h] [rbp-180h] BYREF
  unsigned int v200; // [rsp+C8h] [rbp-178h]
  const void *v201; // [rsp+D0h] [rbp-170h] BYREF
  unsigned int v202; // [rsp+D8h] [rbp-168h]
  _QWORD *v203; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v204; // [rsp+E8h] [rbp-158h]
  _QWORD v205[8]; // [rsp+F0h] [rbp-150h] BYREF
  const char *v206; // [rsp+130h] [rbp-110h] BYREF
  __int64 *v207; // [rsp+138h] [rbp-108h]
  __int64 *v208; // [rsp+140h] [rbp-100h]
  __int64 v209; // [rsp+148h] [rbp-F8h]
  int v210; // [rsp+150h] [rbp-F0h]
  _QWORD v211[9]; // [rsp+158h] [rbp-E8h] BYREF
  __int64 v212; // [rsp+1A0h] [rbp-A0h]
  __int64 *v213; // [rsp+1A8h] [rbp-98h]
  void **v214; // [rsp+1B0h] [rbp-90h]
  void *base; // [rsp+1B8h] [rbp-88h] BYREF
  __int64 v216; // [rsp+1C0h] [rbp-80h]
  _BYTE v217[64]; // [rsp+1C8h] [rbp-78h] BYREF
  unsigned int v218; // [rsp+208h] [rbp-38h]

  v3 = *(_QWORD *)(a1 - 72);
  if ( *(_BYTE *)(v3 + 16) <= 0x17u )
  {
    LODWORD(p_base) = 0;
    return (unsigned int)p_base;
  }
  v218 = 0;
  base = v217;
  v214 = 0;
  v216 = 0x800000000LL;
  v5 = *(_BYTE *)(v3 + 16);
  v212 = a3;
  v213 = 0;
  v177 = v5;
  if ( v5 <= 0x17u )
    BUG();
  v6 = v205;
  v210 = 0;
  LODWORD(v7) = 1;
  v207 = v211;
  v208 = v211;
  v209 = 0x100000008LL;
  v204 = 0x800000001LL;
  v203 = v205;
  v185 = (v5 == 51) + 26;
  v211[0] = v3;
  v205[0] = v3;
  v174 = (v5 != 51) + 32;
  v206 = (const char *)1;
  while ( 1 )
  {
    p_base = (void **)v6[(unsigned int)v7 - 1];
    LODWORD(v204) = v7 - 1;
    v9 = *((_BYTE *)p_base + 16);
    if ( v9 <= 0x17u )
      goto LABEL_25;
    if ( v9 - 24 != v185 )
      break;
    v90 = *(_QWORD *)(sub_13CF970((__int64)p_base) + 24);
    if ( v208 != v207 )
      goto LABEL_142;
    v101 = &v207[HIDWORD(v209)];
    v102 = HIDWORD(v209);
    if ( v207 == v101 )
      goto LABEL_257;
    v103 = v207;
    v104 = 0;
    do
    {
      if ( v90 == *v103 )
      {
        v173 = v207;
        v105 = (__int64 *)sub_13CF970((__int64)p_base);
        v96 = v173;
        v97 = *v105;
        goto LABEL_157;
      }
      if ( *v103 == -2 )
        v104 = v103;
      ++v103;
    }
    while ( v101 != v103 );
    if ( !v104 )
    {
LABEL_257:
      if ( HIDWORD(v209) >= (unsigned int)v209 )
      {
LABEL_142:
        sub_16CCBA0((__int64)&v206, v90);
        v91 = (unsigned __int64)v208;
        v93 = v92;
        v94 = v207;
        if ( !v93 )
          goto LABEL_143;
        goto LABEL_168;
      }
      ++HIDWORD(v209);
      *v101 = v90;
      ++v206;
    }
    else
    {
      *v104 = v90;
      --v210;
      ++v206;
    }
LABEL_168:
    v109 = *(_QWORD *)(sub_13CF970((__int64)p_base) + 24);
    v112 = (unsigned int)v204;
    if ( (unsigned int)v204 >= HIDWORD(v204) )
    {
      sub_16CD150((__int64)&v203, v205, 0, 8, v110, v111);
      v112 = (unsigned int)v204;
    }
    v203[v112] = v109;
    v91 = (unsigned __int64)v208;
    LODWORD(v204) = v204 + 1;
    v94 = v207;
LABEL_143:
    v172 = v94;
    v95 = (__int64 *)sub_13CF970((__int64)p_base);
    v96 = v172;
    v97 = *v95;
    if ( (__int64 *)v91 != v172 )
      goto LABEL_144;
    v102 = HIDWORD(v209);
LABEL_157:
    v106 = &v96[v102];
    if ( v96 == v106 )
      goto LABEL_259;
    v107 = 0;
    do
    {
      if ( v97 == *v96 )
        goto LABEL_133;
      if ( *v96 == -2 )
        v107 = v96;
      ++v96;
    }
    while ( v106 != v96 );
    if ( !v107 )
    {
LABEL_259:
      if ( v102 >= (unsigned int)v209 )
      {
LABEL_144:
        sub_16CCBA0((__int64)&v206, v97);
        v7 = (unsigned int)v204;
        if ( !v98 )
          goto LABEL_134;
        goto LABEL_145;
      }
      HIDWORD(v209) = v102 + 1;
      *v106 = v97;
      v7 = (unsigned int)v204;
      ++v206;
    }
    else
    {
      *v107 = v97;
      v7 = (unsigned int)v204;
      --v210;
      ++v206;
    }
LABEL_145:
    p_base = *(void ***)sub_13CF970((__int64)p_base);
    if ( HIDWORD(v204) <= (unsigned int)v7 )
    {
      sub_16CD150((__int64)&v203, v205, 0, 8, v99, v100);
      v7 = (unsigned int)v204;
    }
    v203[v7] = p_base;
    LODWORD(v7) = v204 + 1;
    LODWORD(v204) = v204 + 1;
LABEL_134:
    if ( !(_DWORD)v7 )
      goto LABEL_27;
    v6 = v203;
  }
  if ( v9 != 75 )
    goto LABEL_25;
  v10 = sub_13CF970((__int64)p_base);
  v11 = v212;
  v12 = sub_1B42400(*(__int64 ****)(v10 + 24), v212);
  v189 = v12;
  if ( !v12 )
    goto LABEL_25;
  v16 = *((unsigned __int16 *)p_base + 9);
  BYTE1(v16) &= ~0x80u;
  if ( v16 == v174 )
  {
    v17 = (__int64)*(p_base - 6);
    v18 = *(_BYTE *)(v17 + 16);
    switch ( v18 )
    {
      case 50:
        v11 = *(_QWORD *)(v17 - 48);
        v168 = v11;
        if ( !v11 )
          goto LABEL_23;
        v113 = *(unsigned __int8 **)(v17 - 24);
        v114 = v113[16];
        if ( (_BYTE)v114 != 13 )
        {
          v13 = *(_QWORD *)v113;
          if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) != 16 || (unsigned __int8)v114 > 0x10u )
            goto LABEL_23;
          goto LABEL_252;
        }
        break;
      case 5:
        v19 = *(_WORD *)(v17 + 18);
        if ( v19 != 26 )
        {
LABEL_12:
          if ( v19 == 27 )
          {
            v20 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
            v11 = 4 * v20;
            v13 = -3 * v20;
            v21 = *(__int64 **)(v17 - 24 * v20);
            if ( v21 )
            {
              v13 = 1 - v20;
              v22 = *(_BYTE **)(v17 + 24 * (1 - v20));
              if ( v22[16] == 13 )
                goto LABEL_15;
              v137 = *(_QWORD *)v22;
              if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) == 16 )
              {
LABEL_223:
                v138 = sub_15A1020(v22, v11, v137, v13);
                if ( v138 && *(_BYTE *)(v138 + 16) == 13 )
                {
                  v23 = (_BYTE *)(v138 + 24);
LABEL_16:
                  v192 = *((_DWORD *)v23 + 2);
                  if ( v192 <= 0x40 )
                  {
                    v24 = *(_QWORD *)v23;
                    v191 = *(_QWORD *)v23;
LABEL_18:
                    if ( !v24 || (v24 & (v24 - 1)) != 0 )
                      goto LABEL_20;
LABEL_241:
                    v170 = v189;
                    v167 = (const void **)(v189 + 24);
                    sub_13A38D0((__int64)&v195, v189 + 24);
                    if ( v196 <= 0x40 )
                    {
                      v200 = v196;
                      v141 = (const void *)(v191 | v195);
                      v196 = 0;
                      v195 = (__int64)v141;
                      v199 = v141;
                      goto LABEL_243;
                    }
                    sub_16A89F0(&v195, (__int64 *)&v191);
                    v163 = v196;
                    v141 = (const void *)v195;
                    v196 = 0;
                    v200 = v163;
                    v199 = (const void *)v195;
                    if ( v163 <= 0x40 )
                    {
LABEL_243:
                      if ( v141 != *(const void **)(v170 + 24) )
                      {
LABEL_269:
                        sub_135E100((__int64 *)&v199);
                        sub_135E100(&v195);
                        goto LABEL_20;
                      }
                    }
                    else if ( !sub_16A5220((__int64)&v199, v167) )
                    {
                      goto LABEL_269;
                    }
                    sub_135E100((__int64 *)&v199);
                    sub_135E100(&v195);
                    if ( v213 )
                    {
                      if ( v21 == v213 )
                        goto LABEL_246;
LABEL_185:
                      sub_135E100((__int64 *)&v191);
                      goto LABEL_25;
                    }
                    v213 = v21;
LABEL_246:
                    p_base = &base;
                    sub_1B47640((__int64)&base, &v189, v142, v143, v144, v145);
                    sub_13A38D0((__int64)&v193, (__int64)&v191);
                    sub_13D0570((__int64)&v193);
                    v146 = v194;
                    v194 = 0;
                    v147 = v189;
                    v196 = v146;
                    v195 = v193;
                    if ( v146 > 0x40 )
                    {
                      sub_16A8890(&v195, (__int64 *)(v189 + 24));
                      v146 = v196;
                      v148 = (const void *)v195;
                      v147 = v189;
                    }
                    else
                    {
                      v148 = (const void *)(*(_QWORD *)(v189 + 24) & v193);
                      v195 = (__int64)v148;
                    }
                    v200 = v146;
                    v199 = v148;
                    v196 = 0;
                    v149 = (__int64 *)sub_16498A0(v147);
                    v190 = sub_159C0E0(v149, (__int64)&v199);
                    sub_1B47640((__int64)&base, &v190, v150, v151, v152, v153);
                    sub_135E100((__int64 *)&v199);
                    sub_135E100(&v195);
                    v154 = &v193;
LABEL_249:
                    sub_135E100(v154);
                    ++v218;
                    sub_135E100((__int64 *)&v191);
                    goto LABEL_133;
                  }
                  sub_16A4FD0((__int64)&v191, (const void **)v23);
                  if ( v192 <= 0x40 )
                  {
                    v24 = v191;
                    goto LABEL_18;
                  }
                  if ( (unsigned int)sub_16A5940((__int64)&v191) == 1 )
                    goto LABEL_241;
LABEL_20:
                  sub_135E100((__int64 *)&v191);
                }
                v17 = (__int64)*(p_base - 6);
              }
LABEL_22:
              if ( !v17 )
                goto LABEL_25;
            }
          }
          goto LABEL_23;
        }
        v125 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
        v13 = -3 * v125;
        v11 = *(_QWORD *)(v17 - 24 * v125);
        v168 = v11;
        if ( !v11 )
          goto LABEL_23;
        v13 = 1 - v125;
        v113 = *(unsigned __int8 **)(v17 + 24 * (1 - v125));
        if ( v113[16] != 13 )
        {
          v114 = *(_QWORD *)v113;
          if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) != 16 )
            goto LABEL_192;
LABEL_252:
          v155 = sub_15A1020(v113, v11, v114, v13);
          if ( !v155 || (v11 = v155 + 24, *(_BYTE *)(v155 + 16) != 13) )
          {
LABEL_236:
            v17 = (__int64)*(p_base - 6);
            v140 = *(_BYTE *)(v17 + 16);
            if ( v140 != 51 )
            {
              if ( v140 != 5 )
                goto LABEL_23;
LABEL_192:
              v19 = *(_WORD *)(v17 + 18);
              goto LABEL_12;
            }
LABEL_219:
            v21 = *(__int64 **)(v17 - 48);
            if ( v21 )
            {
              v22 = *(_BYTE **)(v17 - 24);
              v137 = (unsigned __int8)v22[16];
              if ( (_BYTE)v137 == 13 )
              {
LABEL_15:
                v23 = v22 + 24;
                goto LABEL_16;
              }
              v13 = *(_QWORD *)v22;
              if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) == 16 && (unsigned __int8)v137 <= 0x10u )
                goto LABEL_223;
              goto LABEL_22;
            }
LABEL_23:
            v25 = (__int64)v213;
            if ( v213 )
            {
              if ( (__int64 *)v17 != v213 )
                goto LABEL_25;
            }
            else
            {
              v213 = (__int64 *)v17;
            }
            ++v218;
            sub_1B47640((__int64)&base, &v189, v25, v13, v14, v15);
            v88 = *(p_base - 6) != 0;
            goto LABEL_132;
          }
LABEL_174:
          v115 = *(_DWORD *)(v11 + 8);
          v200 = v115;
          if ( v115 <= 0x40 )
          {
            v116 = *(_QWORD *)v11;
            goto LABEL_176;
          }
          sub_16A4FD0((__int64)&v199, (const void **)v11);
          v115 = v200;
          if ( v200 <= 0x40 )
          {
            v116 = (unsigned __int64)v199;
LABEL_176:
            v11 = ~v116;
            v192 = v115;
            v117 = v11 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v115);
            v191 = v117;
          }
          else
          {
            sub_16A8F40((__int64 *)&v199);
            v115 = v200;
            v117 = (unsigned __int64)v199;
            v192 = v200;
            v191 = (unsigned __int64)v199;
            if ( v200 > 0x40 )
            {
              v166 = v200;
              if ( (unsigned int)sub_16A5940((__int64)&v191) != 1 )
              {
LABEL_235:
                sub_135E100((__int64 *)&v191);
                goto LABEL_236;
              }
              v194 = v166;
              v165 = v189;
              v164 = (const void **)(v189 + 24);
              sub_16A4FD0((__int64)&v193, (const void **)&v191);
              v115 = v194;
              if ( v194 > 0x40 )
              {
                sub_16A8F40(&v193);
                v115 = v194;
                v119 = v193;
                v194 = 0;
                v118 = v189;
                v196 = v115;
                v195 = v193;
                if ( v115 > 0x40 )
                {
                  sub_16A8890(&v195, (__int64 *)(v189 + 24));
                  v139 = v196;
                  v120 = (const void *)v195;
                  v196 = 0;
                  v200 = v139;
                  v199 = (const void *)v195;
                  if ( v139 > 0x40 )
                  {
                    v11 = (__int64)v164;
                    if ( !sub_16A5220((__int64)&v199, v164) )
                    {
LABEL_234:
                      sub_135E100((__int64 *)&v199);
                      sub_135E100(&v195);
                      sub_135E100(&v193);
                      goto LABEL_235;
                    }
LABEL_183:
                    sub_135E100((__int64 *)&v199);
                    sub_135E100(&v195);
                    sub_135E100(&v193);
                    if ( v213 )
                    {
                      if ( (__int64 *)v168 != v213 )
                        goto LABEL_185;
                    }
                    else
                    {
                      v213 = (__int64 *)v168;
                    }
                    p_base = &base;
                    sub_1B47640((__int64)&base, &v189, v121, v122, v123, v124);
                    sub_13A38D0((__int64)&v195, v189 + 24);
                    v156 = v196;
                    if ( v196 > 0x40 )
                    {
                      sub_16A89F0(&v195, (__int64 *)&v191);
                      v156 = v196;
                      v157 = (const void *)v195;
                    }
                    else
                    {
                      v157 = (const void *)(v191 | v195);
                      v195 |= v191;
                    }
                    v200 = v156;
                    v199 = v157;
                    v196 = 0;
                    v158 = (__int64 *)sub_16498A0(v189);
                    v193 = sub_159C0E0(v158, (__int64)&v199);
                    sub_1B47640((__int64)&base, &v193, v159, v160, v161, v162);
                    sub_135E100((__int64 *)&v199);
                    v154 = &v195;
                    goto LABEL_249;
                  }
LABEL_182:
                  v11 = v165;
                  if ( v120 != *(const void **)(v165 + 24) )
                    goto LABEL_234;
                  goto LABEL_183;
                }
LABEL_181:
                v120 = (const void *)(*(_QWORD *)(v118 + 24) & v119);
                v200 = v115;
                v195 = (__int64)v120;
                v199 = v120;
                v196 = 0;
                goto LABEL_182;
              }
              v117 = v193;
              v118 = v189;
LABEL_180:
              v194 = 0;
              v119 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v115) & ~v117;
              v193 = v119;
              goto LABEL_181;
            }
          }
          if ( !v117 || (v117 & (v117 - 1)) != 0 )
            goto LABEL_235;
          v165 = v189;
          v118 = v189;
          goto LABEL_180;
        }
        break;
      case 51:
        goto LABEL_219;
      default:
        goto LABEL_23;
    }
    v11 = (__int64)(v113 + 24);
    goto LABEL_174;
  }
  v194 = *(_DWORD *)(v12 + 32);
  if ( v194 > 0x40 )
    sub_16A4FD0((__int64)&v193, (const void **)(v12 + 24));
  else
    v193 = *(_QWORD *)(v12 + 24);
  sub_1589870((__int64)&v199, &v193);
  v80 = *((_WORD *)p_base + 9) & 0x7FFF;
  sub_158AE10((__int64)&v195, v80, (__int64)&v199);
  if ( v202 > 0x40 && v201 )
    j_j___libc_free_0_0(v201);
  if ( v200 > 0x40 && v199 )
    j_j___libc_free_0_0(v199);
  if ( v194 > 0x40 && v193 )
    j_j___libc_free_0_0(v193);
  v81 = sub_13CF970((__int64)p_base);
  v171 = *(_QWORD *)v81;
  v82 = *(_BYTE *)(*(_QWORD *)v81 + 16LL);
  if ( v82 == 35 )
  {
    v127 = *(_QWORD *)(v171 - 48);
    if ( v127 )
    {
      v129 = *(_BYTE **)(v171 - 24);
      v133 = v129[16];
      if ( v133 == 13 )
      {
LABEL_195:
        v130 = (__int64)(v129 + 24);
LABEL_196:
        v171 = v127;
        sub_158BC30((__int64)&v199, &v195, v130);
        if ( v196 > 0x40 && v195 )
          j_j___libc_free_0_0(v195);
        v195 = (__int64)v199;
        v131 = v200;
        v200 = 0;
        v196 = v131;
        if ( v198 > 0x40 && v197 )
          j_j___libc_free_0_0(v197);
        v197 = v201;
        v132 = v202;
        v202 = 0;
        v198 = v132;
        sub_135E100((__int64 *)&v201);
        sub_135E100((__int64 *)&v199);
      }
      else
      {
        v128 = *(_QWORD *)v129;
        if ( *(_BYTE *)(*(_QWORD *)v129 + 8LL) == 16 && v133 <= 0x10u )
          goto LABEL_207;
      }
    }
  }
  else if ( v82 == 5 && *(_WORD *)(v171 + 18) == 11 )
  {
    v80 = v171;
    v126 = *(_DWORD *)(v171 + 20) & 0xFFFFFFF;
    v127 = *(_QWORD *)(v171 - 24 * v126);
    if ( v127 )
    {
      v128 = 1 - v126;
      v129 = *(_BYTE **)(v171 + 24 * (1 - v126));
      if ( v129[16] == 13 )
        goto LABEL_195;
      if ( *(_BYTE *)(*(_QWORD *)v129 + 8LL) == 16 )
      {
LABEL_207:
        v169 = v127;
        v134 = sub_15A1020(v129, v80, v128, v127);
        if ( v134 && *(_BYTE *)(v134 + 16) == 13 )
        {
          v127 = v169;
          v130 = v134 + 24;
          goto LABEL_196;
        }
      }
    }
  }
  if ( v177 != 51 )
  {
    sub_1590E70((__int64)&v199, (__int64)&v195);
    if ( v196 > 0x40 && v195 )
      j_j___libc_free_0_0(v195);
    v195 = (__int64)v199;
    v135 = v200;
    v200 = 0;
    v196 = v135;
    if ( v198 > 0x40 && v197 )
      j_j___libc_free_0_0(v197);
    v197 = v201;
    v136 = v202;
    v202 = 0;
    v198 = v136;
    sub_135E100((__int64 *)&v201);
    sub_135E100((__int64 *)&v199);
  }
  if ( (unsigned __int8)sub_158A820((__int64)&v195, 8u) || sub_158A120((__int64)&v195) )
    goto LABEL_125;
  if ( v213 )
  {
    if ( v213 == (__int64 *)v171 )
      goto LABEL_117;
LABEL_125:
    v88 = 0;
    goto LABEL_126;
  }
  v213 = (__int64 *)v171;
LABEL_117:
  sub_13A38D0((__int64)&v199, (__int64)&v195);
  while ( 2 )
  {
    if ( v200 > 0x40 )
    {
      if ( sub_16A5220((__int64)&v199, &v197) )
        break;
      goto LABEL_119;
    }
    if ( v199 != v197 )
    {
LABEL_119:
      v83 = (__int64 *)sub_16498A0((__int64)p_base);
      v193 = sub_159C0E0(v83, (__int64)&v199);
      sub_1B47640((__int64)&base, &v193, v84, v85, v86, v87);
      sub_16A7400((__int64)&v199);
      continue;
    }
    break;
  }
  v88 = 1;
  sub_135E100((__int64 *)&v199);
  ++v218;
LABEL_126:
  if ( v198 > 0x40 && v197 )
    j_j___libc_free_0_0(v197);
  if ( v196 > 0x40 && v195 )
    j_j___libc_free_0_0(v195);
LABEL_132:
  if ( v88 )
  {
LABEL_133:
    LODWORD(v7) = v204;
    goto LABEL_134;
  }
LABEL_25:
  if ( !v214 )
  {
    v214 = p_base;
    goto LABEL_133;
  }
  v213 = 0;
LABEL_27:
  if ( v208 != v207 )
    _libc_free((unsigned __int64)v208);
  if ( v203 != v205 )
    _libc_free((unsigned __int64)v203);
  v26 = v213;
  v27 = (char *)base;
  LOBYTE(p_base) = v213 == 0 || v218 <= 1;
  if ( (_BYTE)p_base )
  {
    LODWORD(p_base) = 0;
    goto LABEL_33;
  }
  v29 = (__int64)v214;
  v30 = *(_BYTE *)(v3 + 16);
  v31 = 8LL * (unsigned int)v216;
  if ( (unsigned int)v216 > 1uLL )
  {
    v188 = v214;
    qsort(base, v31 >> 3, 8u, (__compar_fn_t)sub_1B424C0);
    v27 = (char *)base;
    v29 = (__int64)v188;
    v31 = 8LL * (unsigned int)v216;
  }
  v32 = &v27[v31];
  if ( v32 == v27 )
  {
    v37 = 1;
    v36 = 0;
  }
  else
  {
    v33 = v27;
    do
    {
      v35 = v33;
      v33 += 8;
      if ( v32 == v33 )
      {
        v36 = (v32 - v27) >> 3;
        v37 = v36 <= 1;
        goto LABEL_43;
      }
      v34 = *((_QWORD *)v33 - 1);
    }
    while ( v34 != *(_QWORD *)v33 );
    if ( v32 == v35 )
    {
      v36 = (v32 - v27) >> 3;
      v37 = v36 <= 1;
    }
    else
    {
      v78 = v35 + 16;
      if ( v32 != v35 + 16 )
      {
        while ( 1 )
        {
          if ( v34 != *v78 )
          {
            *((_QWORD *)v35 + 1) = *v78;
            v35 += 8;
          }
          if ( v32 == (char *)++v78 )
            break;
          v34 = *(_QWORD *)v35;
        }
        v27 = (char *)base;
        v79 = (char *)((_BYTE *)base + 8 * (unsigned int)v216 - v32);
        v33 = &v79[(_QWORD)(v35 + 8)];
        if ( v32 != (char *)base + 8 * (unsigned int)v216 )
        {
          v187 = v29;
          memmove(v35 + 8, v32, (size_t)v79);
          v27 = (char *)base;
          v29 = v187;
          LODWORD(v216) = (v33 - (_BYTE *)base) >> 3;
          v37 = (unsigned int)v216 <= 1;
          if ( !v187 )
            goto LABEL_45;
          goto LABEL_44;
        }
      }
      v36 = (v33 - v27) >> 3;
      v37 = v36 <= 1;
    }
  }
LABEL_43:
  LODWORD(v216) = v36;
  if ( !v29 )
    goto LABEL_45;
LABEL_44:
  if ( v37 )
    goto LABEL_33;
LABEL_45:
  v38 = *(_QWORD *)(a1 - 48);
  v39 = *(_QWORD **)(a1 + 40);
  v186 = *(_QWORD *)(a1 - 24);
  if ( v30 == 51 )
  {
    if ( v29 )
    {
      v176 = v29;
      v206 = "switch.early.test";
      LOWORD(v208) = 259;
      v181 = sub_157FBF0(v39, (__int64 *)(a1 + 24), (__int64)&v206);
      v179 = sub_157EBA0((__int64)v39);
      sub_17050D0(a2, v179);
      v89 = v186;
      sub_1B48C20(a2, v176, v186, v181, 0, 0);
      v186 = v38;
      v38 = v89;
      v40 = (_QWORD *)v179;
LABEL_48:
      sub_15F20C0(v40);
      v41 = (__int64)v39;
      v39 = (_QWORD *)v181;
      sub_1B44430(v38, v41, v181);
    }
    else
    {
      v186 = *(_QWORD *)(a1 - 48);
      v38 = *(_QWORD *)(a1 - 24);
    }
  }
  else if ( v29 )
  {
    v175 = v29;
    v206 = "switch.early.test";
    LOWORD(v208) = 259;
    v181 = sub_157FBF0(v39, (__int64 *)(a1 + 24), (__int64)&v206);
    v178 = sub_157EBA0((__int64)v39);
    sub_17050D0(a2, v178);
    sub_1B48C20(a2, v175, v181, v38, 0, 0);
    v40 = (_QWORD *)v178;
    goto LABEL_48;
  }
  sub_17050D0(a2, a1);
  if ( *(_BYTE *)(*v26 + 8) == 15 )
  {
    v206 = "magicptr";
    LOWORD(v208) = 259;
    v108 = sub_15A9650(a3, *v26);
    v26 = (__int64 *)sub_12AA3B0(a2, 0x2Du, (__int64)v26, v108, (__int64)&v206);
  }
  LOWORD(v208) = 257;
  v182 = v216;
  v42 = sub_1648B60(64);
  v43 = v42;
  if ( v42 )
    sub_15FFAB0(v42, (__int64)v26, v186, v182, 0);
  v44 = a2[1];
  if ( v44 )
  {
    v45 = (__int64 *)a2[2];
    sub_157E9D0(v44 + 40, v43);
    v46 = *(_QWORD *)(v43 + 24);
    v47 = *v45;
    *(_QWORD *)(v43 + 32) = v45;
    v47 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v43 + 24) = v47 | v46 & 7;
    *(_QWORD *)(v47 + 8) = v43 + 24;
    *v45 = *v45 & 7 | (v43 + 24);
  }
  sub_164B780(v43, (__int64 *)&v206);
  v48 = a2;
  v49 = 0;
  sub_12A86E0(v48, v43);
  v53 = 8LL * (unsigned int)v216;
  if ( (_DWORD)v216 )
  {
    do
    {
      v54 = *(_QWORD *)((char *)base + v49);
      v49 += 8;
      sub_15FFFB0(v43, v54, v38, v50, v51, v52);
    }
    while ( v49 != v53 );
  }
  for ( i = *(_QWORD *)(v38 + 48); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v56 = i - 24;
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v57 = 0x17FFFFFFE8LL;
    v58 = *(_BYTE *)(i - 1) & 0x40;
    v59 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v59 )
    {
      v52 = v56 - 24LL * v59;
      v60 = 24LL * *(unsigned int *)(i + 32) + 8;
      v61 = 0;
      do
      {
        v62 = v56 - 24LL * v59;
        if ( (_BYTE)v58 )
          v62 = *(_QWORD *)(i - 32);
        if ( v39 == *(_QWORD **)(v62 + v60) )
        {
          v57 = 24 * v61;
          goto LABEL_67;
        }
        ++v61;
        v60 += 8;
      }
      while ( v59 != (_DWORD)v61 );
      v57 = 0x17FFFFFFE8LL;
    }
LABEL_67:
    if ( (_BYTE)v58 )
      v63 = *(_QWORD *)(i - 32);
    else
      v63 = v56 - 24LL * v59;
    v64 = *(_QWORD *)(v63 + v57);
    v65 = v216 - 1;
    if ( (_DWORD)v216 != 1 )
    {
      v66 = 0;
      v67 = v64 + 8;
      while ( 1 )
      {
        if ( v59 == *(_DWORD *)(i + 32) )
        {
          v184 = v67;
          sub_15F55D0(v56, v56, v57, v67, v58, v52);
          v67 = v184;
          v59 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
        }
        v76 = (v59 + 1) & 0xFFFFFFF;
        v77 = v76 | *(_DWORD *)(i - 4) & 0xF0000000;
        *(_DWORD *)(i - 4) = v77;
        if ( (v77 & 0x40000000) != 0 )
          v68 = *(_QWORD *)(i - 32);
        else
          v68 = v56 - 24 * v76;
        v69 = (_QWORD *)(v68 + 24LL * (unsigned int)(v76 - 1));
        if ( *v69 )
        {
          v70 = v69[1];
          v71 = v69[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v71 = v70;
          if ( v70 )
          {
            v58 = *(_QWORD *)(v70 + 16) & 3LL;
            *(_QWORD *)(v70 + 16) = v58 | v71;
          }
        }
        *v69 = v64;
        if ( v64 )
        {
          v72 = *(_QWORD *)(v64 + 8);
          v69[1] = v72;
          if ( v72 )
          {
            v58 = (__int64)(v69 + 1);
            *(_QWORD *)(v72 + 16) = (unsigned __int64)(v69 + 1) | *(_QWORD *)(v72 + 16) & 3LL;
          }
          v69[2] = v67 | v69[2] & 3LL;
          *(_QWORD *)(v64 + 8) = v69;
        }
        v73 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
        v74 = (unsigned int)(v73 - 1);
        v75 = (*(_BYTE *)(i - 1) & 0x40) != 0 ? *(_QWORD *)(i - 32) : v56 - 24 * v73;
        ++v66;
        v57 = 3LL * *(unsigned int *)(i + 32);
        *(_QWORD *)(v75 + 8 * v74 + 24LL * *(unsigned int *)(i + 32) + 8) = v39;
        if ( v65 == v66 )
          break;
        v59 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
      }
    }
  }
  LODWORD(p_base) = 1;
  sub_1B44FE0(a1);
  v27 = (char *)base;
LABEL_33:
  if ( v27 != v217 )
    _libc_free((unsigned __int64)v27);
  return (unsigned int)p_base;
}
