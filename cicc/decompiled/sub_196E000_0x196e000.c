// Function: sub_196E000
// Address: 0x196e000
//
__int64 __fastcall sub_196E000(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 *v13; // rax
  unsigned int v15; // r12d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rbx
  unsigned int v22; // r14d
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rax
  unsigned int v25; // r14d
  unsigned int v26; // r14d
  bool v27; // al
  int v28; // r8d
  int v29; // r9d
  unsigned int v30; // ebx
  __int64 v31; // rax
  unsigned int *v32; // rbx
  __int64 v33; // rax
  unsigned __int8 *v34; // rdi
  __int64 v35; // r14
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // r14
  unsigned int v39; // eax
  unsigned __int64 v40; // rbx
  _QWORD *v41; // rax
  int v42; // eax
  __int64 v43; // r14
  _QWORD *v44; // r12
  __int64 *v45; // rbx
  _QWORD *v46; // r8
  int v47; // ecx
  unsigned int v48; // edx
  _QWORD *v49; // rax
  __int64 v50; // rdi
  unsigned int v51; // esi
  unsigned int v52; // edx
  unsigned int v53; // ecx
  unsigned int v54; // r8d
  __int64 v55; // rdx
  __int64 *v56; // r13
  unsigned int v57; // edx
  __int64 v58; // rsi
  int v59; // eax
  __int64 v60; // rdi
  unsigned int v61; // r14d
  __int64 v62; // r12
  int v63; // edx
  unsigned int v64; // eax
  __int64 v65; // rsi
  _QWORD *v66; // rax
  _BYTE *v67; // rbx
  __int64 *v68; // r12
  __int64 v69; // rbx
  __int64 v70; // rax
  unsigned int v71; // ecx
  unsigned __int64 v72; // rax
  unsigned int v73; // ecx
  unsigned int v74; // ecx
  unsigned __int64 v76; // rdx
  double v77; // xmm4_8
  double v78; // xmm5_8
  unsigned int v79; // edx
  char v80; // dl
  unsigned __int8 v81; // r12
  __int64 *v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rbx
  __int64 v85; // rsi
  __int64 v86; // rax
  _QWORD *v87; // r8
  int v88; // esi
  unsigned int v89; // edx
  __int64 *v90; // rax
  __int64 v91; // r10
  int v92; // r8d
  unsigned int v93; // eax
  __int64 v94; // rsi
  __int64 v95; // rax
  unsigned int v96; // eax
  __int64 v97; // rdi
  __int64 v98; // rsi
  __int64 v99; // r9
  unsigned __int64 v100; // r10
  __int64 v101; // rax
  unsigned int v102; // esi
  __int64 *v103; // rdi
  __int64 *v104; // rsi
  unsigned int v105; // eax
  __int64 *v106; // r9
  unsigned int v107; // edx
  unsigned int v108; // r8d
  int v109; // r11d
  int v110; // r8d
  _QWORD *v111; // r8
  int v112; // esi
  unsigned int v113; // edx
  __int64 v114; // rdi
  int v115; // r10d
  __int64 *v116; // rax
  _QWORD *v117; // r8
  int v118; // esi
  unsigned int v119; // edx
  __int64 v120; // rdi
  int v121; // r10d
  int v122; // eax
  char *v123; // rdi
  _BYTE *v124; // rdx
  __int64 *v125; // rax
  __int64 *v126; // r14
  __int64 v127; // rsi
  unsigned __int64 v128; // r8
  _BYTE *v129; // r9
  __int64 *v130; // r12
  __int64 *v131; // r14
  __int64 *v132; // rax
  __int64 *v133; // r10
  __int64 *v134; // rax
  __int64 *v135; // rdi
  int v136; // r9d
  __int64 v137; // rax
  int v138; // eax
  unsigned int v139; // eax
  __int64 v140; // rdi
  __int64 v141; // rsi
  _QWORD *v142; // rax
  int v143; // eax
  int v144; // r14d
  _QWORD *v145; // r11
  __int64 *v146; // rbx
  _QWORD *v147; // r8
  int v148; // r10d
  __int64 v149; // rcx
  __int64 v150; // rsi
  int v151; // r11d
  _QWORD *v152; // rdx
  __int64 v153; // rax
  __int64 v154; // rax
  int v155; // eax
  _QWORD *v156; // rax
  int v157; // eax
  __int64 v158; // rax
  _QWORD *v159; // r8
  int v160; // r10d
  __int64 v161; // rcx
  __int64 v162; // rsi
  int v163; // r11d
  int v164; // eax
  _QWORD *v165; // rax
  __int64 v166; // rax
  int v167; // eax
  __int64 v168; // rax
  __int64 v169; // [rsp+20h] [rbp-320h]
  __int64 v170; // [rsp+28h] [rbp-318h]
  __int64 v171; // [rsp+28h] [rbp-318h]
  __int64 v172; // [rsp+30h] [rbp-310h]
  __int64 v173; // [rsp+30h] [rbp-310h]
  __int64 v174; // [rsp+30h] [rbp-310h]
  __int64 v176; // [rsp+40h] [rbp-300h]
  __int64 v177; // [rsp+40h] [rbp-300h]
  __int64 v178; // [rsp+48h] [rbp-2F8h]
  unsigned __int64 v179; // [rsp+48h] [rbp-2F8h]
  __int64 v180; // [rsp+50h] [rbp-2F0h]
  char *v181; // [rsp+50h] [rbp-2F0h]
  __int64 v182; // [rsp+50h] [rbp-2F0h]
  __int64 v183; // [rsp+50h] [rbp-2F0h]
  const void **v184; // [rsp+58h] [rbp-2E8h]
  bool v185; // [rsp+58h] [rbp-2E8h]
  __int64 v186; // [rsp+58h] [rbp-2E8h]
  __int64 v187; // [rsp+58h] [rbp-2E8h]
  __int64 v188; // [rsp+58h] [rbp-2E8h]
  __int64 v189; // [rsp+58h] [rbp-2E8h]
  __int64 v190; // [rsp+58h] [rbp-2E8h]
  unsigned int v191; // [rsp+58h] [rbp-2E8h]
  unsigned __int64 v192; // [rsp+58h] [rbp-2E8h]
  unsigned int v193; // [rsp+6Ch] [rbp-2D4h]
  unsigned __int8 v194; // [rsp+6Ch] [rbp-2D4h]
  bool v196; // [rsp+70h] [rbp-2D0h]
  __int64 v197; // [rsp+70h] [rbp-2D0h]
  unsigned __int64 v198; // [rsp+70h] [rbp-2D0h]
  __int64 v199; // [rsp+70h] [rbp-2D0h]
  unsigned __int64 v200; // [rsp+70h] [rbp-2D0h]
  unsigned __int64 v201; // [rsp+70h] [rbp-2D0h]
  unsigned __int8 *v202; // [rsp+78h] [rbp-2C8h]
  unsigned int *v203; // [rsp+78h] [rbp-2C8h]
  __int64 v204; // [rsp+78h] [rbp-2C8h]
  char *v205; // [rsp+78h] [rbp-2C8h]
  char v206; // [rsp+78h] [rbp-2C8h]
  __int64 v207; // [rsp+78h] [rbp-2C8h]
  __int64 v208; // [rsp+78h] [rbp-2C8h]
  __int64 v209; // [rsp+78h] [rbp-2C8h]
  __int64 v210; // [rsp+78h] [rbp-2C8h]
  unsigned int v211; // [rsp+80h] [rbp-2C0h]
  unsigned __int8 *v212; // [rsp+80h] [rbp-2C0h]
  __int64 v213; // [rsp+80h] [rbp-2C0h]
  __int64 v214; // [rsp+88h] [rbp-2B8h]
  __int64 *v215; // [rsp+88h] [rbp-2B8h]
  __int64 v216; // [rsp+98h] [rbp-2A8h]
  __int64 v217; // [rsp+98h] [rbp-2A8h]
  unsigned __int8 v218; // [rsp+98h] [rbp-2A8h]
  _QWORD *v219; // [rsp+A0h] [rbp-2A0h] BYREF
  unsigned int v220; // [rsp+A8h] [rbp-298h]
  char *v221; // [rsp+B0h] [rbp-290h] BYREF
  unsigned int v222; // [rsp+B8h] [rbp-288h]
  char *v223; // [rsp+C0h] [rbp-280h] BYREF
  unsigned int v224; // [rsp+C8h] [rbp-278h]
  __int64 v225; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v226; // [rsp+D8h] [rbp-268h]
  __int64 v227; // [rsp+E0h] [rbp-260h]
  __int64 v228; // [rsp+E8h] [rbp-258h]
  __int64 *v229; // [rsp+F0h] [rbp-250h]
  __int64 *v230; // [rsp+F8h] [rbp-248h]
  __int64 v231; // [rsp+100h] [rbp-240h]
  __int64 v232; // [rsp+110h] [rbp-230h] BYREF
  __int64 v233; // [rsp+118h] [rbp-228h]
  __int64 v234; // [rsp+120h] [rbp-220h]
  __int64 v235; // [rsp+128h] [rbp-218h]
  __int64 v236; // [rsp+130h] [rbp-210h]
  __int64 v237; // [rsp+138h] [rbp-208h]
  __int64 v238; // [rsp+140h] [rbp-200h]
  __int64 v239; // [rsp+150h] [rbp-1F0h] BYREF
  __int64 v240; // [rsp+158h] [rbp-1E8h]
  _QWORD *v241; // [rsp+160h] [rbp-1E0h] BYREF
  unsigned int v242; // [rsp+168h] [rbp-1D8h]
  unsigned int *v243; // [rsp+1A0h] [rbp-1A0h] BYREF
  __int64 v244; // [rsp+1A8h] [rbp-198h]
  _BYTE v245[64]; // [rsp+1B0h] [rbp-190h] BYREF
  _QWORD *v246; // [rsp+1F0h] [rbp-150h] BYREF
  __int64 *v247; // [rsp+1F8h] [rbp-148h]
  __int64 *v248; // [rsp+200h] [rbp-140h]
  __int64 v249; // [rsp+208h] [rbp-138h]
  int v250; // [rsp+210h] [rbp-130h]
  _BYTE v251[72]; // [rsp+218h] [rbp-128h] BYREF
  char *v252; // [rsp+260h] [rbp-E0h] BYREF
  _BYTE *v253; // [rsp+268h] [rbp-D8h]
  _BYTE *v254; // [rsp+270h] [rbp-D0h]
  __int64 v255; // [rsp+278h] [rbp-C8h]
  int v256; // [rsp+280h] [rbp-C0h]
  _BYTE v257[184]; // [rsp+288h] [rbp-B8h] BYREF

  v13 = (unsigned __int64 *)&v241;
  v225 = 0;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  v236 = 0;
  v237 = 0;
  v238 = 0;
  v239 = 0;
  v240 = 1;
  do
  {
    *v13 = -8;
    v13 += 2;
  }
  while ( v13 != (unsigned __int64 *)&v243 );
  v243 = (unsigned int *)v245;
  v244 = 0x1000000000LL;
  v193 = *(_DWORD *)(a2 + 8);
  if ( !v193 )
  {
    v194 = 0;
    goto LABEL_123;
  }
  v211 = 1;
  v214 = 0;
  while ( 2 )
  {
    v15 = v214;
    v216 = 8 * v214;
    v16 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v214);
    v202 = *(unsigned __int8 **)(v16 - 48);
    v17 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(sub_146F1B0(*(_QWORD *)(a1 + 32), *(_QWORD *)(v16 - 24)) + 32) + 8LL)
                    + 32LL);
    v224 = *(_DWORD *)(v17 + 32);
    if ( v224 > 0x40 )
      sub_16A4FD0((__int64)&v223, (const void **)(v17 + 24));
    else
      v223 = *(char **)(v17 + 24);
    v18 = *(_QWORD *)(a1 + 56);
    v19 = 1;
    v20 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)a2 + 8 * v214) - 48LL);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v20 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v36 = *(_QWORD *)(v20 + 32);
          v20 = *(_QWORD *)(v20 + 24);
          v19 *= v36;
          continue;
        case 1:
          v21 = 16;
          break;
        case 2:
          v21 = 32;
          break;
        case 3:
        case 9:
          v21 = 64;
          break;
        case 4:
          v21 = 80;
          break;
        case 5:
        case 6:
          v21 = 128;
          break;
        case 7:
          v190 = v19;
          v42 = sub_15A9520(v18, 0);
          v19 = v190;
          v21 = (unsigned int)(8 * v42);
          break;
        case 0xB:
          v21 = *(_DWORD *)(v20 + 8) >> 8;
          break;
        case 0xD:
          v189 = v19;
          v41 = (_QWORD *)sub_15A9930(v18, v20);
          v19 = v189;
          v21 = 8LL * *v41;
          break;
        case 0xE:
          v38 = *(_QWORD *)(v20 + 24);
          v170 = v19;
          v172 = *(_QWORD *)(v20 + 32);
          v39 = sub_15A9FE0(v18, v38);
          v188 = 1;
          v19 = v170;
          v40 = v39;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v38 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v154 = v188 * *(_QWORD *)(v38 + 32);
                v38 = *(_QWORD *)(v38 + 24);
                v188 = v154;
                continue;
              case 1:
                v153 = 16;
                goto LABEL_317;
              case 2:
                v153 = 32;
                goto LABEL_317;
              case 3:
              case 9:
                v153 = 64;
                goto LABEL_317;
              case 4:
                v153 = 80;
                goto LABEL_317;
              case 5:
              case 6:
                v153 = 128;
                goto LABEL_317;
              case 7:
                v155 = sub_15A9520(v18, 0);
                v19 = v170;
                v153 = (unsigned int)(8 * v155);
                goto LABEL_317;
              case 0xB:
                v153 = *(_DWORD *)(v38 + 8) >> 8;
                goto LABEL_317;
              case 0xD:
                v156 = (_QWORD *)sub_15A9930(v18, v38);
                v19 = v170;
                v153 = 8LL * *v156;
                goto LABEL_317;
              case 0xE:
                sub_15A9FE0(v18, *(_QWORD *)(v38 + 24));
                JUMPOUT(0x196FAA4);
              case 0xF:
                v157 = sub_15A9520(v18, *(_DWORD *)(v38 + 8) >> 8);
                v19 = v170;
                v153 = (unsigned int)(8 * v157);
LABEL_317:
                v21 = 8 * v172 * v40 * ((v40 + ((unsigned __int64)(v188 * v153 + 7) >> 3) - 1) / v40);
                break;
            }
            break;
          }
          break;
        case 0xF:
          v187 = v19;
          v37 = sub_15A9520(v18, *(_DWORD *)(v20 + 8) >> 8);
          v19 = v187;
          v21 = (unsigned int)(8 * v37);
          break;
      }
      break;
    }
    v22 = v224;
    v23 = (unsigned int)((unsigned __int64)(v21 * v19 + 7) >> 3);
    if ( v224 <= 0x40 )
    {
      v24 = (unsigned __int64)v223;
      if ( (char *)v23 == v223 )
        goto LABEL_77;
      LODWORD(v247) = v224;
      goto LABEL_13;
    }
    if ( v22 - (unsigned int)sub_16A57B0((__int64)&v223) <= 0x40 && v23 == *(_QWORD *)v223 )
      goto LABEL_77;
    LODWORD(v247) = v22;
    sub_16A4FD0((__int64)&v246, (const void **)&v223);
    LOBYTE(v22) = (_BYTE)v247;
    if ( (unsigned int)v247 <= 0x40 )
    {
      v24 = (unsigned __int64)v246;
LABEL_13:
      v246 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v24);
      goto LABEL_14;
    }
    sub_16A8F40((__int64 *)&v246);
LABEL_14:
    sub_16A7400((__int64)&v246);
    v25 = (unsigned int)v247;
    LODWORD(v247) = 0;
    LODWORD(v253) = v25;
    v252 = (char *)v246;
    if ( v25 <= 0x40 )
    {
      if ( (_QWORD *)v23 != v246 )
        goto LABEL_22;
LABEL_77:
      sub_196DDA0((__int64)&v225, (char **)(*(_QWORD *)a2 + v216));
      if ( v224 > 0x40 )
        goto LABEL_47;
      goto LABEL_49;
    }
    v184 = (const void **)v246;
    v26 = v25 - sub_16A57B0((__int64)&v252);
    v27 = 0;
    if ( v26 <= 0x40 )
      v27 = v23 == (_QWORD)*v184;
    if ( v252 )
    {
      v185 = v27;
      j_j___libc_free_0_0(v252);
      v27 = v185;
      if ( (unsigned int)v247 > 0x40 )
      {
        if ( v246 )
        {
          j_j___libc_free_0_0(v246);
          v27 = v185;
        }
      }
    }
    if ( v27 )
      goto LABEL_77;
LABEL_22:
    if ( a4 )
    {
      v180 = 0;
      v186 = sub_14ABE30(v202);
    }
    else
    {
      v186 = 0;
      v180 = sub_1969620((__int64)v202, *(_BYTE **)(a1 + 56));
    }
    LODWORD(v244) = 0;
    if ( v193 <= v211 )
    {
      if ( !v214 )
        goto LABEL_46;
      v31 = 0;
    }
    else
    {
      v30 = v211;
      v31 = 0;
      do
      {
        if ( HIDWORD(v244) <= (unsigned int)v31 )
        {
          sub_16CD150((__int64)&v243, v245, 0, 4, v28, v29);
          v31 = (unsigned int)v244;
        }
        v243[v31] = v30++;
        v31 = (unsigned int)(v244 + 1);
        LODWORD(v244) = v244 + 1;
      }
      while ( v30 < v193 );
      if ( !v214 )
        goto LABEL_33;
    }
    do
    {
      --v15;
      if ( HIDWORD(v244) <= (unsigned int)v31 )
      {
        sub_16CD150((__int64)&v243, v245, 0, 4, v28, v29);
        v31 = (unsigned int)v244;
      }
      v243[v31] = v15;
      v31 = (unsigned int)(v244 + 1);
      LODWORD(v244) = v244 + 1;
    }
    while ( v15 );
LABEL_33:
    v32 = v243;
    if ( v243 == &v243[v31] )
      goto LABEL_46;
    v203 = &v243[v31];
    while ( 1 )
    {
      v33 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(sub_146F1B0(
                                                  *(_QWORD *)(a1 + 32),
                                                  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8LL * *v32) - 24LL))
                                              + 32)
                                  + 8LL)
                      + 32LL);
      LODWORD(v253) = *(_DWORD *)(v33 + 32);
      if ( (unsigned int)v253 > 0x40 )
        sub_16A4FD0((__int64)&v252, (const void **)(v33 + 24));
      else
        v252 = *(char **)(v33 + 24);
      if ( v224 <= 0x40 )
      {
        if ( v223 != v252 )
          goto LABEL_42;
      }
      else if ( !sub_16A5220((__int64)&v223, (const void **)&v252) )
      {
        goto LABEL_42;
      }
      v34 = *(unsigned __int8 **)(*(_QWORD *)(*(_QWORD *)a2 + 8LL * *v32) - 48LL);
      if ( a4 )
      {
        v35 = sub_14ABE30(v34);
        if ( (unsigned __int8)sub_385F290(
                                *(_QWORD *)(*(_QWORD *)a2 + 8 * v214),
                                *(_QWORD *)(*(_QWORD *)a2 + 8LL * *v32),
                                *(_QWORD *)(a1 + 56),
                                *(_QWORD *)(a1 + 32),
                                0)
          && v186 == v35 )
        {
          break;
        }
        goto LABEL_42;
      }
      v43 = sub_1969620((__int64)v34, *(_BYTE **)(a1 + 56));
      if ( (unsigned __int8)sub_385F290(
                              *(_QWORD *)(*(_QWORD *)a2 + 8 * v214),
                              *(_QWORD *)(*(_QWORD *)a2 + 8LL * *v32),
                              *(_QWORD *)(a1 + 56),
                              *(_QWORD *)(a1 + 32),
                              0) )
      {
        if ( v180 == v43 )
          break;
      }
LABEL_42:
      if ( (unsigned int)v253 > 0x40 && v252 )
        j_j___libc_free_0_0(v252);
      if ( v203 == ++v32 )
        goto LABEL_46;
    }
    sub_196DDA0((__int64)&v232, (char **)(*(_QWORD *)a2 + 8LL * *v32));
    sub_196DDA0((__int64)&v225, (char **)(v216 + *(_QWORD *)a2));
    v44 = (_QWORD *)(*(_QWORD *)a2 + 8LL * *v32);
    v45 = (__int64 *)(*(_QWORD *)a2 + v216);
    if ( (v240 & 1) != 0 )
    {
      v46 = &v241;
      v47 = 3;
      goto LABEL_68;
    }
    v51 = v242;
    v46 = v241;
    v47 = v242 - 1;
    if ( !v242 )
    {
      v52 = v240;
      ++v239;
      v49 = 0;
      v53 = ((unsigned int)v240 >> 1) + 1;
      goto LABEL_81;
    }
LABEL_68:
    v48 = v47 & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
    v49 = &v46[2 * v48];
    v50 = *v49;
    if ( *v49 != *v45 )
    {
      v144 = 1;
      v145 = 0;
      while ( v50 != -8 )
      {
        if ( v50 == -16 && !v145 )
          v145 = v49;
        v48 = v47 & (v144 + v48);
        v49 = &v46[2 * v48];
        v50 = *v49;
        if ( *v45 == *v49 )
          goto LABEL_69;
        ++v144;
      }
      v52 = v240;
      v54 = 12;
      v51 = 4;
      if ( v145 )
        v49 = v145;
      ++v239;
      v53 = ((unsigned int)v240 >> 1) + 1;
      if ( (v240 & 1) == 0 )
      {
        v51 = v242;
LABEL_81:
        v54 = 3 * v51;
      }
      if ( 4 * v53 < v54 )
      {
        if ( v51 - HIDWORD(v240) - v53 > v51 >> 3 )
        {
LABEL_84:
          LODWORD(v240) = (2 * (v52 >> 1) + 2) | v52 & 1;
          if ( *v49 != -8 )
            --HIDWORD(v240);
          v55 = *v45;
          v49[1] = 0;
          *v49 = v55;
          goto LABEL_69;
        }
        sub_196D4F0((__int64)&v239, v51);
        if ( (v240 & 1) != 0 )
        {
          v159 = &v241;
          v160 = 3;
LABEL_334:
          v52 = v240;
          LODWORD(v161) = v160 & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
          v49 = &v159[2 * (unsigned int)v161];
          v162 = *v49;
          if ( *v45 == *v49 )
            goto LABEL_84;
          v163 = 1;
          v152 = 0;
          while ( v162 != -8 )
          {
            if ( v162 == -16 && !v152 )
              v152 = v49;
            v161 = v160 & (unsigned int)(v161 + v163);
            v49 = &v159[2 * v161];
            v162 = *v49;
            if ( *v45 == *v49 )
              goto LABEL_339;
            ++v163;
          }
          goto LABEL_337;
        }
        v159 = v241;
        if ( v242 )
        {
          v160 = v242 - 1;
          goto LABEL_334;
        }
LABEL_370:
        LODWORD(v240) = (2 * ((unsigned int)v240 >> 1) + 2) | v240 & 1;
        BUG();
      }
      sub_196D4F0((__int64)&v239, 2 * v51);
      if ( (v240 & 1) != 0 )
      {
        v147 = &v241;
        v148 = 3;
      }
      else
      {
        v147 = v241;
        if ( !v242 )
          goto LABEL_370;
        v148 = v242 - 1;
      }
      v52 = v240;
      LODWORD(v149) = v148 & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
      v49 = &v147[2 * (unsigned int)v149];
      v150 = *v49;
      if ( *v49 == *v45 )
        goto LABEL_84;
      v151 = 1;
      v152 = 0;
      while ( v150 != -8 )
      {
        if ( v150 == -16 && !v152 )
          v152 = v49;
        v149 = v148 & (unsigned int)(v149 + v151);
        v49 = &v147[2 * v149];
        v150 = *v49;
        if ( *v45 == *v49 )
          goto LABEL_339;
        ++v151;
      }
LABEL_337:
      if ( v152 )
        v49 = v152;
LABEL_339:
      v52 = v240;
      goto LABEL_84;
    }
LABEL_69:
    v49[1] = *v44;
    if ( (unsigned int)v253 > 0x40 && v252 )
      j_j___libc_free_0_0(v252);
LABEL_46:
    if ( v224 > 0x40 )
    {
LABEL_47:
      if ( v223 )
        j_j___libc_free_0_0(v223);
    }
LABEL_49:
    ++v214;
    if ( v193 > v211 )
    {
      ++v211;
      continue;
    }
    break;
  }
  v252 = 0;
  v253 = v257;
  v215 = v230;
  v254 = v257;
  v255 = 16;
  v256 = 0;
  v194 = 0;
  if ( v230 == v229 )
    goto LABEL_123;
  v56 = v229;
  while ( 2 )
  {
    while ( 2 )
    {
      v59 = v235;
      v60 = v233;
      if ( (_DWORD)v235 )
      {
        v57 = (v235 - 1) & (((unsigned int)*v56 >> 9) ^ ((unsigned int)*v56 >> 4));
        v58 = *(_QWORD *)(v233 + 8LL * v57);
        if ( *v56 != v58 )
        {
          v136 = 1;
          while ( v58 != -8 )
          {
            v57 = (v235 - 1) & (v136 + v57);
            v58 = *(_QWORD *)(v233 + 8LL * v57);
            if ( *v56 == v58 )
              goto LABEL_94;
            ++v136;
          }
          goto LABEL_96;
        }
        goto LABEL_94;
      }
LABEL_96:
      v61 = 0;
      v246 = 0;
      v247 = (__int64 *)v251;
      v248 = (__int64 *)v251;
      v249 = 8;
      v250 = 0;
      v217 = *v56;
      v62 = *v56;
      if ( (_DWORD)v235 )
      {
LABEL_97:
        v63 = v59 - 1;
        v64 = (v59 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v65 = *(_QWORD *)(v60 + 8LL * v64);
        if ( v65 == v62 )
          goto LABEL_98;
        v110 = 1;
        while ( v65 != -8 )
        {
          v64 = v63 & (v110 + v64);
          v65 = *(_QWORD *)(v60 + 8LL * v64);
          if ( v65 == v62 )
            goto LABEL_98;
          ++v110;
        }
      }
      while ( 2 )
      {
        if ( (_DWORD)v228 )
        {
          v92 = 1;
          v93 = (v228 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v94 = *(_QWORD *)(v226 + 8LL * v93);
          if ( v94 != v62 )
          {
            while ( v94 != -8 )
            {
              v93 = (v228 - 1) & (v92 + v93);
              v94 = *(_QWORD *)(v226 + 8LL * v93);
              if ( v94 == v62 )
                goto LABEL_98;
              ++v92;
            }
            break;
          }
LABEL_98:
          v66 = v253;
          if ( v254 == v253 )
          {
            v67 = &v253[8 * HIDWORD(v255)];
            if ( v253 == v67 )
            {
              v124 = v253;
            }
            else
            {
              do
              {
                if ( *v66 == v62 )
                  break;
                ++v66;
              }
              while ( v67 != (_BYTE *)v66 );
              v124 = &v253[8 * HIDWORD(v255)];
            }
          }
          else
          {
            v67 = &v254[8 * (unsigned int)v255];
            v66 = sub_16CC9F0((__int64)&v252, v62);
            if ( *v66 == v62 )
            {
              if ( v254 == v253 )
                v124 = &v254[8 * HIDWORD(v255)];
              else
                v124 = &v254[8 * (unsigned int)v255];
            }
            else
            {
              if ( v254 != v253 )
              {
                v66 = &v254[8 * (unsigned int)v255];
                goto LABEL_102;
              }
              v66 = &v254[8 * HIDWORD(v255)];
              v124 = v66;
            }
          }
          while ( v124 != (_BYTE *)v66 && *v66 >= 0xFFFFFFFFFFFFFFFELL )
            ++v66;
LABEL_102:
          if ( v67 != (_BYTE *)v66 )
            break;
          v82 = v247;
          if ( v248 == v247 )
          {
            v103 = &v247[HIDWORD(v249)];
            if ( v247 == v103 )
            {
LABEL_209:
              if ( HIDWORD(v249) >= (unsigned int)v249 )
                goto LABEL_147;
              ++HIDWORD(v249);
              *v103 = v62;
              v246 = (_QWORD *)((char *)v246 + 1);
            }
            else
            {
              v104 = 0;
              while ( *v82 != v62 )
              {
                if ( *v82 == -2 )
                  v104 = v82;
                if ( v103 == ++v82 )
                {
                  if ( !v104 )
                    goto LABEL_209;
                  *v104 = v62;
                  --v250;
                  v246 = (_QWORD *)((char *)v246 + 1);
                  break;
                }
              }
            }
          }
          else
          {
LABEL_147:
            sub_16CCBA0((__int64)&v246, v62);
          }
          v83 = *(_QWORD *)(a1 + 56);
          v84 = 1;
          v85 = **(_QWORD **)(v62 - 48);
LABEL_149:
          switch ( *(_BYTE *)(v85 + 8) )
          {
            case 1:
              v86 = 16;
              goto LABEL_151;
            case 2:
              v86 = 32;
              goto LABEL_151;
            case 3:
            case 9:
              v86 = 64;
              goto LABEL_151;
            case 4:
              v86 = 80;
              goto LABEL_151;
            case 5:
            case 6:
              v86 = 128;
              goto LABEL_151;
            case 7:
              v86 = 8 * (unsigned int)sub_15A9520(v83, 0);
              goto LABEL_151;
            case 0xB:
              v86 = *(_DWORD *)(v85 + 8) >> 8;
              goto LABEL_151;
            case 0xD:
              v86 = 8LL * *(_QWORD *)sub_15A9930(v83, v85);
              goto LABEL_151;
            case 0xE:
              v204 = *(_QWORD *)(a1 + 56);
              v197 = *(_QWORD *)(v85 + 24);
              v213 = *(_QWORD *)(v85 + 32);
              v96 = sub_15A9FE0(v83, v197);
              v97 = v204;
              v98 = v197;
              v99 = 1;
              v100 = v96;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v98 + 8) )
                {
                  case 1:
                    v101 = 16;
                    goto LABEL_173;
                  case 2:
                    v101 = 32;
                    goto LABEL_173;
                  case 3:
                  case 9:
                    v101 = 64;
                    goto LABEL_173;
                  case 4:
                    v101 = 80;
                    goto LABEL_173;
                  case 5:
                  case 6:
                    v101 = 128;
                    goto LABEL_173;
                  case 7:
                    v201 = v100;
                    v210 = v99;
                    v143 = sub_15A9520(v97, 0);
                    v99 = v210;
                    v100 = v201;
                    v101 = (unsigned int)(8 * v143);
                    goto LABEL_173;
                  case 0xB:
                    v101 = *(_DWORD *)(v98 + 8) >> 8;
                    goto LABEL_173;
                  case 0xD:
                    v200 = v100;
                    v209 = v99;
                    v142 = (_QWORD *)sub_15A9930(v97, v98);
                    v99 = v209;
                    v100 = v200;
                    v101 = 8LL * *v142;
                    goto LABEL_173;
                  case 0xE:
                    v173 = v100;
                    v176 = v99;
                    v178 = *(_QWORD *)(v98 + 24);
                    v182 = v204;
                    v199 = *(_QWORD *)(v98 + 32);
                    v139 = sub_15A9FE0(v204, v178);
                    v140 = v204;
                    v208 = 1;
                    v100 = v173;
                    v141 = v178;
                    v192 = v139;
                    v99 = v176;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v141 + 8) )
                      {
                        case 1:
                          v158 = 16;
                          goto LABEL_331;
                        case 2:
                          v158 = 32;
                          goto LABEL_331;
                        case 3:
                        case 9:
                          v158 = 64;
                          goto LABEL_331;
                        case 4:
                          v158 = 80;
                          goto LABEL_331;
                        case 5:
                        case 6:
                          v158 = 128;
                          goto LABEL_331;
                        case 7:
                          v164 = sub_15A9520(v140, 0);
                          v99 = v176;
                          v100 = v173;
                          v158 = (unsigned int)(8 * v164);
                          goto LABEL_331;
                        case 0xB:
                          v158 = *(_DWORD *)(v141 + 8) >> 8;
                          goto LABEL_331;
                        case 0xD:
                          v165 = (_QWORD *)sub_15A9930(v140, v141);
                          v99 = v176;
                          v100 = v173;
                          v158 = 8LL * *v165;
                          goto LABEL_331;
                        case 0xE:
                          v169 = v173;
                          v171 = v176;
                          v174 = *(_QWORD *)(v141 + 24);
                          v177 = v182;
                          v183 = *(_QWORD *)(v141 + 32);
                          v179 = (unsigned int)sub_15A9FE0(v140, v174);
                          v166 = sub_127FA20(v177, v174);
                          v99 = v171;
                          v100 = v169;
                          v158 = 8 * v183 * v179 * ((v179 + ((unsigned __int64)(v166 + 7) >> 3) - 1) / v179);
                          goto LABEL_331;
                        case 0xF:
                          v167 = sub_15A9520(v140, *(_DWORD *)(v141 + 8) >> 8);
                          v99 = v176;
                          v100 = v173;
                          v158 = (unsigned int)(8 * v167);
LABEL_331:
                          v101 = 8 * v192 * v199 * ((v192 + ((unsigned __int64)(v208 * v158 + 7) >> 3) - 1) / v192);
                          goto LABEL_173;
                        case 0x10:
                          v168 = v208 * *(_QWORD *)(v141 + 32);
                          v141 = *(_QWORD *)(v141 + 24);
                          v208 = v168;
                          continue;
                        default:
                          goto LABEL_371;
                      }
                    }
                  case 0xF:
                    v198 = v100;
                    v207 = v99;
                    v138 = sub_15A9520(v97, *(_DWORD *)(v98 + 8) >> 8);
                    v99 = v207;
                    v100 = v198;
                    v101 = (unsigned int)(8 * v138);
LABEL_173:
                    v86 = 8 * v213 * v100 * ((v100 + ((unsigned __int64)(v101 * v99 + 7) >> 3) - 1) / v100);
                    goto LABEL_151;
                  case 0x10:
                    v137 = *(_QWORD *)(v98 + 32);
                    v98 = *(_QWORD *)(v98 + 24);
                    v99 *= v137;
                    continue;
                  default:
                    goto LABEL_371;
                }
              }
            case 0xF:
              v86 = 8 * (unsigned int)sub_15A9520(v83, *(_DWORD *)(v85 + 8) >> 8);
LABEL_151:
              v61 += (unsigned __int64)(v86 * v84 + 7) >> 3;
              if ( (v240 & 1) != 0 )
              {
                v87 = &v241;
                v88 = 3;
              }
              else
              {
                v102 = v242;
                v87 = v241;
                if ( !v242 )
                {
                  v105 = v240;
                  ++v239;
                  v106 = 0;
                  v107 = ((unsigned int)v240 >> 1) + 1;
LABEL_196:
                  v108 = 3 * v102;
                  goto LABEL_197;
                }
                v88 = v242 - 1;
              }
              v89 = v88 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
              v90 = &v87[2 * v89];
              v91 = *v90;
              if ( *v90 == v62 )
              {
                v62 = v90[1];
                goto LABEL_155;
              }
              v109 = 1;
              v106 = 0;
              while ( v91 != -8 )
              {
                if ( v91 != -16 || v106 )
                  v90 = v106;
                v89 = v88 & (v109 + v89);
                v146 = &v87[2 * v89];
                v91 = *v146;
                if ( *v146 == v62 )
                {
                  v62 = v146[1];
                  goto LABEL_155;
                }
                v106 = v90;
                ++v109;
                v90 = &v87[2 * v89];
              }
              v108 = 12;
              v102 = 4;
              if ( !v106 )
                v106 = v90;
              v105 = v240;
              ++v239;
              v107 = ((unsigned int)v240 >> 1) + 1;
              if ( (v240 & 1) == 0 )
              {
                v102 = v242;
                goto LABEL_196;
              }
LABEL_197:
              if ( 4 * v107 >= v108 )
              {
                sub_196D4F0((__int64)&v239, 2 * v102);
                if ( (v240 & 1) != 0 )
                {
                  v111 = &v241;
                  v112 = 3;
                }
                else
                {
                  v111 = v241;
                  if ( !v242 )
                    goto LABEL_369;
                  v112 = v242 - 1;
                }
                v105 = v240;
                v113 = v112 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
                v106 = &v111[2 * v113];
                v114 = *v106;
                if ( v62 != *v106 )
                {
                  v115 = 1;
                  v116 = 0;
                  while ( v114 != -8 )
                  {
                    if ( !v116 && v114 == -16 )
                      v116 = v106;
                    v113 = v112 & (v115 + v113);
                    v106 = &v111[2 * v113];
                    v114 = *v106;
                    if ( *v106 == v62 )
                      goto LABEL_222;
                    ++v115;
                  }
                  goto LABEL_220;
                }
              }
              else if ( v102 - HIDWORD(v240) - v107 <= v102 >> 3 )
              {
                sub_196D4F0((__int64)&v239, v102);
                if ( (v240 & 1) != 0 )
                {
                  v117 = &v241;
                  v118 = 3;
                }
                else
                {
                  v117 = v241;
                  if ( !v242 )
                  {
LABEL_369:
                    LODWORD(v240) = (2 * ((unsigned int)v240 >> 1) + 2) | v240 & 1;
                    BUG();
                  }
                  v118 = v242 - 1;
                }
                v105 = v240;
                v119 = v118 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
                v106 = &v117[2 * v119];
                v120 = *v106;
                if ( *v106 != v62 )
                {
                  v121 = 1;
                  v116 = 0;
                  while ( v120 != -8 )
                  {
                    if ( v120 == -16 && !v116 )
                      v116 = v106;
                    v119 = v118 & (v121 + v119);
                    v106 = &v117[2 * v119];
                    v120 = *v106;
                    if ( *v106 == v62 )
                      goto LABEL_222;
                    ++v121;
                  }
LABEL_220:
                  if ( v116 )
                    v106 = v116;
LABEL_222:
                  v105 = v240;
                }
              }
              LODWORD(v240) = (2 * (v105 >> 1) + 2) | v105 & 1;
              if ( *v106 != -8 )
                --HIDWORD(v240);
              *v106 = v62;
              v62 = 0;
              v106[1] = 0;
LABEL_155:
              v59 = v235;
              v60 = v233;
              if ( !(_DWORD)v235 )
                continue;
              goto LABEL_97;
            case 0x10:
              v95 = *(_QWORD *)(v85 + 32);
              v85 = *(_QWORD *)(v85 + 24);
              v84 *= v95;
              goto LABEL_149;
            default:
LABEL_371:
              BUG();
          }
        }
        break;
      }
      v68 = *(__int64 **)(v217 - 24);
      v212 = *(unsigned __int8 **)(v217 - 48);
      v69 = sub_146F1B0(*(_QWORD *)(a1 + 32), (__int64)v68);
      v70 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v69 + 32) + 8LL) + 32LL);
      v71 = *(_DWORD *)(v70 + 32);
      v220 = v71;
      if ( v71 <= 0x40 )
      {
        v219 = *(_QWORD **)(v70 + 24);
        goto LABEL_105;
      }
      sub_16A4FD0((__int64)&v219, (const void **)(v70 + 24));
      v71 = v220;
      if ( v220 <= 0x40 )
      {
LABEL_105:
        v72 = (unsigned __int64)v219;
        if ( v219 == (_QWORD *)v61 )
        {
          v74 = v220;
          goto LABEL_133;
        }
        v222 = v71;
        goto LABEL_107;
      }
      v191 = v220;
      v122 = sub_16A57B0((__int64)&v219);
      v74 = v191;
      if ( v191 - v122 <= 0x40 && *v219 == v61 )
        goto LABEL_133;
      v222 = v191;
      sub_16A4FD0((__int64)&v221, (const void **)&v219);
      LOBYTE(v71) = v222;
      if ( v222 <= 0x40 )
      {
        v72 = (unsigned __int64)v221;
LABEL_107:
        v221 = (char *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v71) & ~v72);
        goto LABEL_108;
      }
      sub_16A8F40((__int64 *)&v221);
LABEL_108:
      sub_16A7400((__int64)&v221);
      v73 = v222;
      v222 = 0;
      v224 = v73;
      v223 = v221;
      if ( v73 > 0x40 )
      {
        v181 = v221;
        v196 = 1;
        if ( v73 - (unsigned int)sub_16A57B0((__int64)&v223) <= 0x40 )
          v196 = *(_QWORD *)v181 != v61;
        if ( v223 )
        {
          j_j___libc_free_0_0(v223);
          if ( v222 > 0x40 )
          {
            if ( v221 )
              j_j___libc_free_0_0(v221);
          }
        }
        v74 = v220;
        if ( v196 )
          goto LABEL_116;
LABEL_133:
        v222 = v74;
        if ( v74 <= 0x40 )
        {
          v76 = (unsigned __int64)v219;
          goto LABEL_135;
        }
        sub_16A4FD0((__int64)&v221, (const void **)&v219);
        LOBYTE(v74) = v222;
        if ( v222 <= 0x40 )
        {
          v76 = (unsigned __int64)v221;
LABEL_135:
          v221 = (char *)(~v76 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v74));
        }
        else
        {
          sub_16A8F40((__int64 *)&v221);
        }
        sub_16A7400((__int64)&v221);
        v79 = v222;
        v222 = 0;
        v224 = v79;
        v223 = v221;
        if ( v79 <= 0x40 )
        {
          v80 = v221 == (char *)v61;
          goto LABEL_138;
        }
        v205 = v221;
        if ( v79 - (unsigned int)sub_16A57B0((__int64)&v223) <= 0x40 && *(_QWORD *)v205 == v61 )
        {
          v80 = 1;
LABEL_241:
          v123 = v205;
          v206 = v80;
          j_j___libc_free_0_0(v123);
          v80 = v206;
          if ( v222 > 0x40 && v221 )
          {
            j_j___libc_free_0_0(v221);
            v80 = v206;
          }
        }
        else
        {
          v80 = 0;
          if ( v205 )
            goto LABEL_241;
        }
LABEL_138:
        v81 = sub_196B740(
                a1,
                v68,
                v61,
                1 << (*(unsigned __int16 *)(v217 + 18) >> 1) >> 1,
                v212,
                v217,
                a5,
                a6,
                a7,
                a8,
                v77,
                v78,
                a11,
                a12,
                (__int64)&v246,
                v69,
                a3,
                v80,
                0);
        if ( v81 )
        {
          v125 = v248;
          if ( v248 == v247 )
            v126 = &v248[HIDWORD(v249)];
          else
            v126 = &v248[(unsigned int)v249];
          if ( v248 != v126 )
          {
            while ( 1 )
            {
              v127 = *v125;
              if ( (unsigned __int64)*v125 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v126 == ++v125 )
                goto LABEL_140;
            }
            if ( v125 != v126 )
            {
              v128 = (unsigned __int64)v254;
              v129 = v253;
              v218 = v81;
              v130 = v126;
              v131 = v125;
              if ( v254 != v253 )
              {
LABEL_258:
                sub_16CCBA0((__int64)&v252, v127);
                v128 = (unsigned __int64)v254;
                v129 = v253;
                goto LABEL_259;
              }
              while ( 1 )
              {
                v133 = (__int64 *)(v128 + 8LL * HIDWORD(v255));
                if ( v133 == (__int64 *)v128 )
                {
LABEL_292:
                  if ( HIDWORD(v255) >= (unsigned int)v255 )
                    goto LABEL_258;
                  ++HIDWORD(v255);
                  *v133 = v127;
                  v129 = v253;
                  ++v252;
                  v128 = (unsigned __int64)v254;
                }
                else
                {
                  v134 = (__int64 *)v128;
                  v135 = 0;
                  while ( *v134 != v127 )
                  {
                    if ( *v134 == -2 )
                      v135 = v134;
                    if ( v133 == ++v134 )
                    {
                      if ( !v135 )
                        goto LABEL_292;
                      *v135 = v127;
                      v128 = (unsigned __int64)v254;
                      --v256;
                      v129 = v253;
                      ++v252;
                      break;
                    }
                  }
                }
LABEL_259:
                v132 = v131 + 1;
                if ( v131 + 1 == v130 )
                  break;
                while ( 1 )
                {
                  v127 = *v132;
                  v131 = v132;
                  if ( (unsigned __int64)*v132 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v130 == ++v132 )
                    goto LABEL_262;
                }
                if ( v132 == v130 )
                  break;
                if ( (_BYTE *)v128 != v129 )
                  goto LABEL_258;
              }
LABEL_262:
              v81 = v218;
            }
          }
        }
        else
        {
          v81 = v194;
        }
LABEL_140:
        if ( v220 > 0x40 && v219 )
          j_j___libc_free_0_0(v219);
        if ( v248 != v247 )
          _libc_free((unsigned __int64)v248);
        v194 = v81;
LABEL_94:
        if ( v215 == ++v56 )
          goto LABEL_121;
        continue;
      }
      break;
    }
    v74 = v220;
    if ( v221 == (char *)v61 )
      goto LABEL_133;
LABEL_116:
    if ( v74 > 0x40 && v219 )
      j_j___libc_free_0_0(v219);
    if ( v248 == v247 )
      goto LABEL_94;
    _libc_free((unsigned __int64)v248);
    if ( v215 != ++v56 )
      continue;
    break;
  }
LABEL_121:
  if ( v254 != v253 )
    _libc_free((unsigned __int64)v254);
LABEL_123:
  if ( v243 != (unsigned int *)v245 )
    _libc_free((unsigned __int64)v243);
  if ( (v240 & 1) == 0 )
    j___libc_free_0(v241);
  if ( v236 )
    j_j___libc_free_0(v236, v238 - v236);
  j___libc_free_0(v233);
  if ( v229 )
    j_j___libc_free_0(v229, v231 - (_QWORD)v229);
  j___libc_free_0(v226);
  return v194;
}
