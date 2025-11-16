// Function: sub_29613A0
// Address: 0x29613a0
//
__int64 __fastcall sub_29613A0(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r13
  __int64 *v10; // r14
  __int64 v11; // r15
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r8
  __int64 *v18; // rbx
  __int64 v19; // rdx
  unsigned __int64 v20; // r8
  __int64 *v21; // rdx
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // r15
  __int64 *v29; // rax
  char *v30; // rdi
  __int64 *v31; // r13
  __int64 *v32; // rbx
  int v33; // esi
  char *v34; // rax
  __int64 v35; // r15
  __int64 *v36; // rax
  __int64 v37; // r11
  __int64 *v38; // r14
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdi
  __int64 *v43; // rbx
  __int64 v44; // r8
  __int64 v45; // r13
  __int64 *v46; // r12
  __int64 v47; // rsi
  __int64 *v48; // rax
  bool v49; // al
  int v50; // eax
  __int64 v51; // r15
  __int64 *v52; // r14
  __int64 **v53; // rax
  __int64 *v54; // r13
  __int64 *v55; // r15
  __int64 *v56; // rbx
  __int64 i; // rcx
  __int64 v58; // rsi
  _QWORD *v59; // rdi
  _QWORD *v60; // rdx
  _QWORD *v61; // rax
  __int64 v62; // r8
  __int64 *v63; // r15
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 *v66; // rbx
  __int64 v67; // rsi
  _QWORD *v68; // rax
  _QWORD *v69; // rdx
  __int64 *j; // rbx
  __int64 v71; // rsi
  _QWORD *v72; // rax
  _QWORD *v73; // rdx
  __int64 v74; // rcx
  unsigned int v75; // esi
  unsigned int v76; // edx
  char *v77; // rax
  __int64 v78; // r11
  __int64 **v79; // rax
  __int64 *v80; // r8
  _BYTE *v81; // rax
  __int64 v82; // r8
  _QWORD *v83; // r12
  __int64 *v84; // r12
  __int64 *v85; // r14
  __int64 v86; // rax
  __int64 *v87; // r15
  __int64 v88; // rsi
  __int64 *v89; // rax
  __int64 v90; // rsi
  __int64 *v91; // r13
  __int64 v92; // rsi
  _QWORD *v93; // rax
  __int64 v94; // rsi
  int v95; // eax
  int v96; // edi
  __int64 **v97; // r12
  __int64 **v98; // r15
  __int64 *v99; // rdx
  char *v100; // r12
  __int64 *v101; // rbx
  __int64 v102; // rsi
  char *v103; // r12
  __int64 v104; // rax
  char *v105; // r13
  __int64 v106; // r15
  char *v107; // r13
  char *v108; // rax
  unsigned __int64 v109; // rbx
  __int64 *k; // rdx
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 *v114; // r12
  __int64 v115; // rdi
  __int64 v116; // rcx
  unsigned __int64 v117; // rsi
  unsigned int v118; // edx
  __int64 *v119; // rax
  __int64 *v120; // rax
  __int64 v121; // rdx
  __int64 *v122; // r15
  __int64 v123; // rsi
  __int64 *v124; // rbx
  __int64 v125; // rax
  unsigned __int64 v126; // rdx
  __int64 v127; // rdi
  unsigned int v128; // ecx
  __int64 v129; // rdx
  __int64 *v130; // rax
  __int64 v131; // r15
  __int64 v132; // rdx
  __int64 v133; // r12
  __int64 *v134; // rdx
  _QWORD *v135; // rax
  __int64 v136; // rcx
  __int64 *v137; // rdx
  _QWORD *v138; // rax
  __int64 v139; // rax
  unsigned __int64 v140; // rdx
  __int64 v141; // rdx
  __int64 *v142; // rax
  char *v143; // r13
  __int64 v144; // rax
  __int64 *v145; // rbx
  unsigned int v146; // eax
  __int64 *v147; // rax
  __int64 v148; // rdx
  __int64 *v149; // r14
  __int64 v150; // rsi
  __int64 *v151; // rbx
  __int64 *v152; // rsi
  __int64 *v153; // rcx
  __int64 *v154; // rax
  char *v155; // r14
  char *v156; // r12
  __int64 v157; // rax
  __int64 v158; // rcx
  char *v159; // r15
  __int64 v160; // rsi
  __int64 *v161; // rax
  __int64 *v162; // rcx
  char *v163; // r13
  __int64 v164; // rsi
  __int64 *v165; // rcx
  __int64 v166; // rsi
  __int64 *v167; // rax
  __int64 v168; // rsi
  __int64 *v169; // rdi
  __int64 *v170; // rax
  __int64 *v171; // rax
  __int64 *v172; // rax
  __int64 *v173; // rdi
  __int64 *v174; // rax
  __int64 *v175; // rax
  __int64 *v176; // rax
  unsigned int v177; // ecx
  char *v178; // rdx
  unsigned __int64 v179; // r10
  char *v180; // rdi
  _QWORD *v181; // rdi
  unsigned __int64 v182; // rdi
  int v183; // r10d
  char *v184; // r11
  unsigned int v185; // r11d
  int v186; // edi
  int v187; // edi
  int v188; // ecx
  int v189; // eax
  __int64 v190; // r10
  __int64 v191; // rbx
  __int64 v192; // rax
  char *v193; // r15
  __int64 v194; // r9
  char **v195; // r12
  char **v196; // rbx
  int v197; // esi
  unsigned int v198; // edx
  __int64 *v199; // rax
  __int64 v200; // r10
  __int64 v201; // rax
  _BYTE *v202; // rsi
  __int64 v203; // rax
  char *v204; // r14
  __int64 v205; // rcx
  int v206; // esi
  __int64 v207; // rdi
  char *v208; // rdx
  unsigned int v209; // r12d
  int v211; // eax
  __int64 v212; // rax
  int v213; // r10d
  __int64 v214; // r12
  int v215; // r10d
  __int64 *v216; // rdi
  __int64 *v217; // rsi
  _QWORD *v218; // rsi
  __int64 *v219; // rsi
  int v220; // esi
  _QWORD *v221; // r15
  unsigned int v222; // r13d
  int v223; // esi
  int v224; // r14d
  __int64 *v225; // rax
  __int64 *v226; // r8
  __int64 *v227; // rdx
  _QWORD *v228; // rsi
  __int64 *v229; // rax
  __int64 *v230; // r8
  _QWORD *v231; // rax
  __int64 *v232; // rax
  __int64 v233; // rsi
  __int64 *v234; // rax
  __int64 v235; // r10
  __int64 v236; // rbx
  __int64 v237; // rax
  __int64 *v238; // r15
  __int64 v239; // r9
  int v240; // edx
  int v241; // edi
  int v242; // edi
  __int64 *v243; // r8
  _BYTE *v244; // rax
  __int64 v245; // r8
  _QWORD *v246; // rbx
  _BYTE *v247; // rsi
  __int64 v248; // rsi
  _QWORD *v249; // rdi
  unsigned int v250; // esi
  unsigned int v251; // edx
  __int64 **v252; // rax
  __int64 *v253; // r8
  int v254; // eax
  int v255; // edi
  __int64 v256; // r15
  __int64 v257; // rdx
  int v258; // edi
  int v259; // r8d
  char *v260; // rdi
  int v261; // edi
  __int64 v262; // rax
  __int64 v265; // [rsp+20h] [rbp-360h]
  __int64 v266; // [rsp+28h] [rbp-358h]
  int v267; // [rsp+28h] [rbp-358h]
  __int64 *v268; // [rsp+40h] [rbp-340h]
  __int64 v269; // [rsp+40h] [rbp-340h]
  __int64 v270; // [rsp+48h] [rbp-338h]
  char *v271; // [rsp+48h] [rbp-338h]
  __int64 v272; // [rsp+48h] [rbp-338h]
  __int64 *v273; // [rsp+48h] [rbp-338h]
  __int64 v274; // [rsp+48h] [rbp-338h]
  __int64 v275; // [rsp+48h] [rbp-338h]
  __int64 *v276; // [rsp+50h] [rbp-330h]
  __int64 *v279; // [rsp+68h] [rbp-318h]
  __int64 v280; // [rsp+68h] [rbp-318h]
  char *v281; // [rsp+68h] [rbp-318h]
  __int64 v282; // [rsp+68h] [rbp-318h]
  char *v283; // [rsp+68h] [rbp-318h]
  __int64 v284; // [rsp+70h] [rbp-310h] BYREF
  char *v285; // [rsp+78h] [rbp-308h] BYREF
  _BYTE *v286; // [rsp+80h] [rbp-300h] BYREF
  __int64 v287; // [rsp+88h] [rbp-2F8h]
  _BYTE v288[32]; // [rsp+90h] [rbp-2F0h] BYREF
  void *src; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 v290; // [rsp+B8h] [rbp-2C8h]
  _BYTE v291[32]; // [rsp+C0h] [rbp-2C0h] BYREF
  _BYTE *v292; // [rsp+E0h] [rbp-2A0h] BYREF
  __int64 v293; // [rsp+E8h] [rbp-298h]
  _BYTE v294[128]; // [rsp+F0h] [rbp-290h] BYREF
  __int64 v295; // [rsp+170h] [rbp-210h] BYREF
  __int64 *v296; // [rsp+178h] [rbp-208h]
  __int64 v297; // [rsp+180h] [rbp-200h]
  int v298; // [rsp+188h] [rbp-1F8h]
  char v299; // [rsp+18Ch] [rbp-1F4h]
  char v300; // [rsp+190h] [rbp-1F0h] BYREF
  __int64 *v301; // [rsp+210h] [rbp-170h] BYREF
  __int64 *v302; // [rsp+218h] [rbp-168h]
  __int64 v303; // [rsp+220h] [rbp-160h]
  int v304; // [rsp+228h] [rbp-158h]
  char v305; // [rsp+22Ch] [rbp-154h]
  char v306; // [rsp+230h] [rbp-150h] BYREF
  char *v307; // [rsp+2B0h] [rbp-D0h] BYREF
  void *s; // [rsp+2B8h] [rbp-C8h]
  _BYTE v309[12]; // [rsp+2C0h] [rbp-C0h] BYREF
  unsigned __int8 v310; // [rsp+2CCh] [rbp-B4h]
  char v311; // [rsp+2D0h] [rbp-B0h] BYREF

  v276 = (__int64 *)sub_D4B130(a1);
  v286 = v288;
  v287 = 0x400000000LL;
  src = v291;
  v290 = 0x400000000LL;
  if ( a3 > 4 )
  {
    sub_C8D5F0((__int64)&src, v291, a3, 8u, v7, v8);
    v279 = &a2[a3];
    if ( a2 != v279 )
      goto LABEL_3;
  }
  else
  {
    v279 = &a2[a3];
    if ( a2 != v279 )
    {
LABEL_3:
      v9 = a2;
      v10 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v11 = *v9;
          v12 = *(_DWORD *)(a4 + 24);
          v13 = *(_QWORD *)(a4 + 8);
          if ( v12 )
          {
            v14 = v12 - 1;
            v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v16 = (__int64 *)(v13 + 16LL * v15);
            v17 = *v16;
            if ( v11 != *v16 )
            {
              v95 = 1;
              while ( v17 != -4096 )
              {
                v96 = v95 + 1;
                v15 = v14 & (v95 + v15);
                v16 = (__int64 *)(v13 + 16LL * v15);
                v17 = *v16;
                if ( v11 == *v16 )
                  goto LABEL_7;
                v95 = v96;
              }
              goto LABEL_4;
            }
LABEL_7:
            v18 = (__int64 *)v16[1];
            if ( v18 )
              break;
          }
LABEL_4:
          if ( v279 == ++v9 )
            goto LABEL_16;
        }
        v19 = (unsigned int)v287;
        v20 = (unsigned int)v287 + 1LL;
        if ( v20 > HIDWORD(v287) )
        {
          sub_C8D5F0((__int64)&v286, v288, (unsigned int)v287 + 1LL, 8u, v20, v8);
          v19 = (unsigned int)v287;
        }
        *(_QWORD *)&v286[8 * v19] = v18;
        LODWORD(v287) = v287 + 1;
        sub_B1A4E0((__int64)&src, v11);
        if ( v10 )
        {
          if ( v18 != v10 )
          {
            v21 = v18;
            while ( 1 )
            {
              v21 = (__int64 *)*v21;
              if ( v10 == v21 )
                goto LABEL_15;
              if ( !v21 )
                goto LABEL_4;
            }
          }
          goto LABEL_4;
        }
LABEL_15:
        v10 = v18;
        if ( v279 == ++v9 )
          goto LABEL_16;
      }
    }
  }
  v10 = 0;
LABEL_16:
  v295 = 0;
  v296 = (__int64 *)&v300;
  v299 = 1;
  v297 = 16;
  v298 = 0;
  v270 = sub_D4B130(a1);
  v26 = **(_QWORD **)(a1 + 32);
  v307 = v309;
  s = (void *)0x1000000000LL;
  v27 = *(_QWORD *)(v26 + 16);
  if ( v27 )
  {
    while ( 1 )
    {
      v22 = *(_QWORD *)(v27 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        goto LABEL_50;
    }
    while ( 1 )
    {
      v28 = *(_QWORD *)(v22 + 40);
      if ( v270 == v28 )
        goto LABEL_24;
      if ( !v299 )
        goto LABEL_52;
      v29 = v296;
      v23 = HIDWORD(v297);
      v22 = (__int64)&v296[HIDWORD(v297)];
      if ( v296 == (__int64 *)v22 )
        break;
      while ( v28 != *v29 )
      {
        if ( (__int64 *)v22 == ++v29 )
          goto LABEL_60;
      }
LABEL_24:
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
      {
LABEL_27:
        v30 = v307;
        if ( HIDWORD(v297) != v298 )
        {
LABEL_28:
          v268 = v10;
          v31 = (__int64 *)&v301;
          v32 = &v295;
          v33 = (int)s;
LABEL_30:
          v34 = &v30[8 * v33];
          while ( v33 )
          {
            v35 = *((_QWORD *)v34 - 1);
            --v33;
            v34 -= 8;
            LODWORD(s) = v33;
            if ( v26 != v35 )
            {
              v24 = *(unsigned int *)(a4 + 24);
              v25 = *(_QWORD *)(a4 + 8);
              if ( (_DWORD)v24 )
              {
                v24 = (unsigned int)(v24 - 1);
                v22 = (unsigned int)v24 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
                v36 = (__int64 *)(v25 + 16 * v22);
                v37 = *v36;
                if ( v35 == *v36 )
                {
LABEL_35:
                  v24 = v36[1];
                  if ( a1 != v24 && v24 )
                  {
                    v266 = v36[1];
                    v38 = (__int64 *)sub_D4B130(v266);
                    sub_26C2C80((__int64)v31, (__int64)v32, v38, v39, v40, v41);
                    v24 = v266;
                    if ( !v306 )
                      goto LABEL_29;
                    if ( *(_QWORD *)(v266 + 32) == *(_QWORD *)(v266 + 40) )
                      goto LABEL_48;
                    v42 = (__int64)v32;
                    v43 = *(__int64 **)(v266 + 32);
                    v44 = (__int64)v31;
                    v45 = v26;
                    v46 = *(__int64 **)(v266 + 40);
                    while ( 1 )
                    {
LABEL_40:
                      v47 = *v43;
                      if ( v35 == *v43 )
                        goto LABEL_46;
                      if ( v299 )
                      {
                        v48 = v296;
                        v23 = HIDWORD(v297);
                        v22 = (__int64)&v296[HIDWORD(v297)];
                        if ( v296 != (__int64 *)v22 )
                        {
                          while ( v47 != *v48 )
                          {
                            if ( (__int64 *)v22 == ++v48 )
                              goto LABEL_57;
                          }
LABEL_46:
                          if ( v46 == ++v43 )
                            goto LABEL_47;
                          continue;
                        }
LABEL_57:
                        if ( HIDWORD(v297) < (unsigned int)v297 )
                          break;
                      }
                      ++v43;
                      v265 = v44;
                      sub_C8CC70(v42, v47, v22, v23, v44, v25);
                      v44 = v265;
                      if ( v46 == v43 )
                        goto LABEL_47;
                    }
                    v23 = (unsigned int)(HIDWORD(v297) + 1);
                    ++v43;
                    ++HIDWORD(v297);
                    *(_QWORD *)v22 = v47;
                    ++v295;
                    if ( v46 != v43 )
                      goto LABEL_40;
LABEL_47:
                    v26 = v45;
                    v32 = (__int64 *)v42;
                    v31 = (__int64 *)v44;
LABEL_48:
                    sub_B1A4E0((__int64)&v307, (__int64)v38);
                    v30 = v307;
                    v33 = (int)s;
                    goto LABEL_30;
                  }
                }
                else
                {
                  v50 = 1;
                  while ( v37 != -4096 )
                  {
                    v23 = (unsigned int)(v50 + 1);
                    v262 = (unsigned int)v24 & ((_DWORD)v22 + v50);
                    v22 = (unsigned int)v262;
                    v36 = (__int64 *)(v25 + 16 * v262);
                    v37 = *v36;
                    if ( v35 == *v36 )
                      goto LABEL_35;
                    v50 = v23;
                  }
                }
              }
              v51 = *(_QWORD *)(v35 + 16);
              if ( v51 )
              {
                while ( 1 )
                {
                  v22 = *(_QWORD *)(v51 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
                    break;
                  v51 = *(_QWORD *)(v51 + 8);
                  if ( !v51 )
                    goto LABEL_30;
                }
LABEL_66:
                v52 = *(__int64 **)(v22 + 40);
                if ( *(_BYTE *)(a1 + 84) )
                {
                  v53 = *(__int64 ***)(a1 + 64);
                  v22 = (__int64)&v53[*(unsigned int *)(a1 + 76)];
                  if ( v53 == (__int64 **)v22 )
                    goto LABEL_73;
                  while ( v52 != *v53 )
                  {
                    if ( (__int64 **)v22 == ++v53 )
                      goto LABEL_73;
                  }
                }
                else if ( !sub_C8CA60(a1 + 56, *(_QWORD *)(v22 + 40)) )
                {
                  goto LABEL_73;
                }
                sub_26C2C80((__int64)v31, (__int64)v32, v52, v23, v24, v25);
                if ( v306 )
                  sub_B1A4E0((__int64)&v307, (__int64)v52);
LABEL_73:
                while ( 1 )
                {
                  v51 = *(_QWORD *)(v51 + 8);
                  if ( !v51 )
                    break;
                  v22 = *(_QWORD *)(v51 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
                    goto LABEL_66;
                }
LABEL_29:
                v30 = v307;
                v33 = (int)s;
                goto LABEL_30;
              }
              goto LABEL_30;
            }
          }
          v10 = v268;
        }
        if ( v30 != v309 )
          _libc_free((unsigned __int64)v30);
        goto LABEL_83;
      }
      while ( 1 )
      {
        v22 = *(_QWORD *)(v27 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
          break;
        v27 = *(_QWORD *)(v27 + 8);
        if ( !v27 )
          goto LABEL_27;
      }
    }
LABEL_60:
    if ( HIDWORD(v297) >= (unsigned int)v297 )
    {
LABEL_52:
      sub_C8CC70((__int64)&v295, v28, v22, v23, v24, v25);
      v49 = v22 & (v26 != v28);
    }
    else
    {
      v49 = v26 != v28;
      v23 = (unsigned int)++HIDWORD(v297);
      *(_QWORD *)v22 = v28;
      ++v295;
    }
    if ( v49 )
      sub_B1A4E0((__int64)&v307, v28);
    goto LABEL_24;
  }
LABEL_50:
  if ( v298 != HIDWORD(v297) )
  {
    v30 = v309;
    goto LABEL_28;
  }
LABEL_83:
  if ( HIDWORD(v297) == v298 )
  {
LABEL_146:
    v85 = *(__int64 **)(a1 + 32);
    goto LABEL_147;
  }
  v54 = *(__int64 **)a1;
  if ( *(__int64 **)a1 == v10 )
    goto LABEL_120;
  v280 = a1 + 56;
  do
  {
    sub_25DDDB0((__int64)(v54 + 7), (__int64)v276);
    v55 = *(__int64 **)(a1 + 40);
    v56 = *(__int64 **)(a1 + 32);
    for ( i = (__int64)(v54 + 7); v55 != v56; ++v56 )
    {
      v58 = *v56;
      if ( *((_BYTE *)v54 + 84) )
      {
        v59 = (_QWORD *)v54[8];
        v60 = &v59[*((unsigned int *)v54 + 19)];
        v61 = v59;
        if ( v59 != v60 )
        {
          while ( v58 != *v61 )
          {
            if ( v60 == ++v61 )
              goto LABEL_93;
          }
          v62 = (unsigned int)(*((_DWORD *)v54 + 19) - 1);
          *((_DWORD *)v54 + 19) = v62;
          *v61 = v59[v62];
          ++v54[7];
        }
      }
      else
      {
        v275 = i;
        v225 = sub_C8CA60(i, v58);
        i = v275;
        if ( v225 )
        {
          *v225 = -2;
          ++*((_DWORD *)v54 + 20);
          ++v54[7];
        }
      }
LABEL_93:
      ;
    }
    v63 = (__int64 *)v54[4];
    v271 = (char *)v54[5];
    v64 = (v271 - (char *)v63) >> 5;
    v65 = (v271 - (char *)v63) >> 3;
    if ( v64 <= 0 )
      goto LABEL_408;
    v66 = &v63[4 * v64];
    do
    {
      v67 = *v63;
      if ( v276 == (__int64 *)*v63 )
        goto LABEL_102;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v68 = *(_QWORD **)(a1 + 64);
        i = (__int64)&v68[*(unsigned int *)(a1 + 76)];
        if ( v68 != (_QWORD *)i )
        {
          v69 = *(_QWORD **)(a1 + 64);
          while ( v67 != *v69 )
          {
            if ( (_QWORD *)i == ++v69 )
            {
              v226 = (__int64 *)v63[1];
              v227 = v63 + 1;
              if ( v276 == v226 )
                goto LABEL_399;
              goto LABEL_396;
            }
          }
          goto LABEL_102;
        }
        v227 = v63 + 1;
        if ( v276 == (__int64 *)v63[1] )
          goto LABEL_399;
LABEL_404:
        v227 = v63 + 2;
        if ( v276 == (__int64 *)v63[2] )
          goto LABEL_399;
LABEL_405:
        v227 = v63 + 3;
        if ( v276 == (__int64 *)v63[3] )
          goto LABEL_399;
        goto LABEL_406;
      }
      if ( sub_C8CA60(v280, v67) )
        goto LABEL_102;
      v226 = (__int64 *)v63[1];
      v227 = v63 + 1;
      if ( v276 == v226 )
        goto LABEL_399;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v68 = *(_QWORD **)(a1 + 64);
        i = (__int64)&v68[*(unsigned int *)(a1 + 76)];
        if ( v68 != (_QWORD *)i )
        {
LABEL_396:
          v228 = v68;
          while ( (__int64 *)*v68 != v226 )
          {
            if ( ++v68 == (_QWORD *)i )
            {
              v230 = (__int64 *)v63[2];
              v227 = v63 + 2;
              if ( v276 != v230 )
                goto LABEL_416;
              break;
            }
          }
LABEL_399:
          v63 = v227;
          goto LABEL_102;
        }
        goto LABEL_404;
      }
      v229 = sub_C8CA60(v280, v63[1]);
      v227 = v63 + 1;
      if ( v229 )
        goto LABEL_399;
      v230 = (__int64 *)v63[2];
      v227 = v63 + 2;
      if ( v276 == v230 )
        goto LABEL_399;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v228 = *(_QWORD **)(a1 + 64);
        i = (__int64)&v228[*(unsigned int *)(a1 + 76)];
        if ( (_QWORD *)i != v228 )
        {
LABEL_416:
          v231 = v228;
          while ( (__int64 *)*v228 != v230 )
          {
            if ( ++v228 == (_QWORD *)i )
            {
              v233 = v63[3];
              v227 = v63 + 3;
              if ( v276 == (__int64 *)v233 )
                goto LABEL_399;
LABEL_426:
              while ( *v231 != v233 )
              {
                if ( ++v231 == (_QWORD *)i )
                  goto LABEL_406;
              }
              v63 = v227;
              goto LABEL_102;
            }
          }
          goto LABEL_399;
        }
        goto LABEL_405;
      }
      v232 = sub_C8CA60(v280, v63[2]);
      v227 = v63 + 2;
      if ( v232 )
        goto LABEL_399;
      v233 = v63[3];
      v227 = v63 + 3;
      if ( v276 == (__int64 *)v233 )
        goto LABEL_399;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v231 = *(_QWORD **)(a1 + 64);
        i = (__int64)&v231[*(unsigned int *)(a1 + 76)];
        if ( (_QWORD *)i != v231 )
          goto LABEL_426;
      }
      else
      {
        v234 = sub_C8CA60(v280, v233);
        v227 = v63 + 3;
        if ( v234 )
          goto LABEL_399;
      }
LABEL_406:
      v63 += 4;
    }
    while ( v66 != v63 );
    v65 = (v271 - (char *)v63) >> 3;
LABEL_408:
    if ( v65 == 2 )
      goto LABEL_474;
    if ( v65 == 3 )
    {
      if ( v276 == (__int64 *)*v63 || (unsigned __int8)sub_B19060(v280, *v63, 3, i) )
        goto LABEL_102;
      ++v63;
LABEL_474:
      if ( v276 != (__int64 *)*v63 && !(unsigned __int8)sub_B19060(v280, *v63, v65, i) )
      {
        ++v63;
        goto LABEL_477;
      }
      goto LABEL_102;
    }
    if ( v65 != 1 )
      goto LABEL_411;
LABEL_477:
    if ( v276 != (__int64 *)*v63 && !(unsigned __int8)sub_B19060(v280, *v63, v65, i) )
    {
LABEL_411:
      v63 = (__int64 *)v271;
      goto LABEL_111;
    }
LABEL_102:
    if ( v271 != (char *)v63 )
    {
      for ( j = v63 + 1; v271 != (char *)j; ++v63 )
      {
LABEL_104:
        v71 = *j;
        if ( v276 == (__int64 *)*j )
          goto LABEL_110;
        if ( *(_BYTE *)(a1 + 84) )
        {
          v72 = *(_QWORD **)(a1 + 64);
          v73 = &v72[*(unsigned int *)(a1 + 76)];
          if ( v72 != v73 )
          {
            while ( v71 != *v72 )
            {
              if ( v73 == ++v72 )
                goto LABEL_391;
            }
LABEL_110:
            if ( v271 == (char *)++j )
              break;
            goto LABEL_104;
          }
        }
        else
        {
          if ( sub_C8CA60(v280, v71) )
            goto LABEL_110;
          v71 = *j;
        }
LABEL_391:
        *v63 = v71;
        ++j;
      }
    }
LABEL_111:
    sub_295D210((__int64)(v54 + 4), (char *)v63, v271);
    v54 = (__int64 *)*v54;
  }
  while ( v54 != v10 );
  v301 = v276;
  v74 = *(_QWORD *)(a4 + 8);
  v75 = *(_DWORD *)(a4 + 24);
  if ( !v10 )
  {
    if ( v75 )
    {
      v250 = v75 - 1;
      v251 = v250 & (((unsigned int)v276 >> 9) ^ ((unsigned int)v276 >> 4));
      v252 = (__int64 **)(v74 + 16LL * v251);
      v253 = *v252;
      if ( v276 == *v252 )
      {
LABEL_503:
        *v252 = (__int64 *)-8192LL;
        --*(_DWORD *)(a4 + 16);
        ++*(_DWORD *)(a4 + 20);
      }
      else
      {
        v254 = 1;
        while ( v253 != (__int64 *)-4096LL )
        {
          v255 = v254 + 1;
          v251 = v250 & (v254 + v251);
          v252 = (__int64 **)(v74 + 16LL * v251);
          v253 = *v252;
          if ( v276 == *v252 )
            goto LABEL_503;
          v254 = v255;
        }
      }
    }
    goto LABEL_117;
  }
  if ( !v75 )
  {
    v307 = 0;
    ++*(_QWORD *)a4;
LABEL_516:
    v256 = a4;
    sub_D4F150(a4, 2 * v75);
    goto LABEL_517;
  }
  v76 = (v75 - 1) & (((unsigned int)v276 >> 9) ^ ((unsigned int)v276 >> 4));
  v77 = (char *)(v74 + 16LL * v76);
  v78 = *(_QWORD *)v77;
  if ( v276 == *(__int64 **)v77 )
  {
LABEL_115:
    v79 = (__int64 **)(v77 + 8);
    goto LABEL_116;
  }
  v259 = 1;
  v260 = 0;
  while ( v78 != -4096 )
  {
    if ( v78 == -8192 && !v260 )
      v260 = v77;
    v76 = (v75 - 1) & (v259 + v76);
    v77 = (char *)(v74 + 16LL * v76);
    v78 = *(_QWORD *)v77;
    if ( v276 == *(__int64 **)v77 )
      goto LABEL_115;
    ++v259;
  }
  if ( v260 )
    v77 = v260;
  ++*(_QWORD *)a4;
  v261 = *(_DWORD *)(a4 + 16);
  v307 = v77;
  v258 = v261 + 1;
  if ( 4 * v258 >= 3 * v75 )
    goto LABEL_516;
  v257 = (__int64)v276;
  if ( v75 - *(_DWORD *)(a4 + 20) - v258 <= v75 >> 3 )
  {
    v256 = a4;
    sub_D4F150(a4, v75);
LABEL_517:
    sub_D4C730(v256, (__int64 *)&v301, &v307);
    v257 = (__int64)v301;
    v258 = *(_DWORD *)(v256 + 16) + 1;
    v77 = v307;
  }
  *(_DWORD *)(a4 + 16) = v258;
  if ( *(_QWORD *)v77 != -4096 )
    --*(_DWORD *)(a4 + 20);
  *(_QWORD *)v77 = v257;
  v79 = (__int64 **)(v77 + 8);
  *v79 = 0;
LABEL_116:
  *v79 = v10;
LABEL_117:
  v80 = *(__int64 **)a1;
  v307 = (char *)a1;
  v81 = sub_29578B0((_QWORD *)v80[1], v80[2], (__int64 *)&v307);
  v83 = *(_QWORD **)v81;
  sub_D4C9B0(v82 + 8, v81);
  *v83 = 0;
  if ( v10 )
  {
    *(_QWORD *)a1 = v10;
    v307 = (char *)a1;
    sub_D4C980((__int64)(v10 + 1), &v307);
  }
  else
  {
    v307 = (char *)a1;
    sub_D4C980(a4 + 32, &v307);
  }
  if ( HIDWORD(v297) == v298 )
    goto LABEL_146;
LABEL_120:
  v84 = *(__int64 **)(a1 + 40);
  v85 = *(__int64 **)(a1 + 32);
  v86 = ((char *)v84 - (char *)v85) >> 5;
  v23 = v84 - v85;
  if ( v86 <= 0 )
  {
LABEL_138:
    if ( v23 != 2 )
    {
      if ( v23 != 3 )
      {
        if ( v23 != 1 )
        {
LABEL_141:
          v85 = v84;
          goto LABEL_147;
        }
LABEL_532:
        if ( !(unsigned __int8)sub_B19060((__int64)&v295, *v85, v22, v23) )
          goto LABEL_457;
        goto LABEL_141;
      }
      if ( !(unsigned __int8)sub_B19060((__int64)&v295, *v85, v22, 3) )
        goto LABEL_457;
      ++v85;
    }
    if ( !(unsigned __int8)sub_B19060((__int64)&v295, *v85, v22, v23) )
      goto LABEL_457;
    ++v85;
    goto LABEL_532;
  }
  v87 = &v85[4 * v86];
  while ( 1 )
  {
    v88 = *v85;
    if ( v299 )
      break;
    if ( !sub_C8CA60((__int64)&v295, v88) )
      goto LABEL_457;
    v90 = v85[1];
    v91 = v85 + 1;
    if ( v299 )
    {
      v89 = v296;
      v22 = (__int64)&v296[HIDWORD(v297)];
      if ( v296 != (__int64 *)v22 )
      {
        v23 = (unsigned __int64)v296;
LABEL_129:
        while ( *v89 != v90 )
        {
          if ( ++v89 == (__int64 *)v22 )
            goto LABEL_456;
        }
        v92 = v85[2];
        v91 = v85 + 2;
        v93 = (_QWORD *)v23;
        do
        {
LABEL_132:
          if ( *(_QWORD *)v23 == v92 )
          {
            v94 = v85[3];
            v91 = v85 + 3;
            goto LABEL_135;
          }
          v23 += 8LL;
        }
        while ( v23 != v22 );
      }
      goto LABEL_456;
    }
    if ( !sub_C8CA60((__int64)&v295, v90) )
      goto LABEL_456;
    v92 = v85[2];
    v91 = v85 + 2;
    if ( v299 )
    {
      v23 = (unsigned __int64)v296;
      v22 = (__int64)&v296[HIDWORD(v297)];
      if ( (__int64 *)v22 != v296 )
      {
        v93 = v296;
        goto LABEL_132;
      }
LABEL_456:
      v85 = v91;
      goto LABEL_457;
    }
    if ( !sub_C8CA60((__int64)&v295, v92) )
      goto LABEL_456;
    v94 = v85[3];
    v91 = v85 + 3;
    if ( v299 )
    {
      v93 = v296;
      v22 = (__int64)&v296[HIDWORD(v297)];
      if ( (__int64 *)v22 == v296 )
        goto LABEL_456;
LABEL_135:
      while ( *v93 != v94 )
      {
        if ( ++v93 == (_QWORD *)v22 )
          goto LABEL_456;
      }
      v85 += 4;
      if ( v87 == v85 )
        goto LABEL_137;
    }
    else
    {
      if ( !sub_C8CA60((__int64)&v295, v94) )
        goto LABEL_456;
      v85 += 4;
      if ( v87 == v85 )
      {
LABEL_137:
        v23 = v84 - v85;
        goto LABEL_138;
      }
    }
  }
  v89 = v296;
  v22 = (__int64)&v296[HIDWORD(v297)];
  if ( v296 != (__int64 *)v22 )
  {
    v23 = (unsigned __int64)v296;
    do
    {
      if ( v88 == *(_QWORD *)v23 )
      {
        v90 = v85[1];
        v91 = v85 + 1;
        v23 = (unsigned __int64)v296;
        goto LABEL_129;
      }
      v23 += 8LL;
    }
    while ( v22 != v23 );
  }
LABEL_457:
  if ( v84 != v85 )
  {
    v235 = v84 - v85;
    if ( (char *)v84 - (char *)v85 <= 0 )
    {
LABEL_540:
      v238 = 0;
      v239 = 0;
    }
    else
    {
      v236 = v84 - v85;
      while ( 1 )
      {
        v269 = v235;
        v237 = sub_2207800(8 * v236);
        v235 = v269;
        v238 = (__int64 *)v237;
        if ( v237 )
          break;
        v236 >>= 1;
        if ( !v236 )
          goto LABEL_540;
      }
      v239 = v236;
    }
    v85 = sub_295BED0(v85, v84, (__int64)&v295, v235, v238, v239);
    j_j___libc_free_0((unsigned __int64)v238);
  }
LABEL_147:
  v97 = (__int64 **)v85;
  v301 = 0;
  v305 = 1;
  v303 = 16;
  v98 = *(__int64 ***)(a1 + 40);
  v302 = (__int64 *)&v306;
  v304 = 0;
  if ( v98 != (__int64 **)v85 )
  {
    do
    {
      v99 = *v97++;
      sub_D695C0((__int64)&v307, (__int64)&v301, v99, v23, v24, v25);
    }
    while ( v98 != v97 );
  }
  if ( HIDWORD(v297) == v298 )
    sub_D695C0((__int64)&v307, (__int64)&v301, v276, v23, v24, v25);
  v100 = *(char **)(a1 + 40);
  if ( v85 != (__int64 *)v100 )
  {
    v101 = v85;
    do
    {
      v102 = *v101++;
      sub_25DDDB0(a1 + 56, v102);
    }
    while ( v100 != (char *)v101 );
    v100 = *(char **)(a1 + 40);
  }
  sub_295D210(a1 + 32, (char *)v85, v100);
  v103 = (char *)src;
  v104 = 8LL * (unsigned int)v290;
  v105 = (char *)src + v104;
  v106 = v104 >> 3;
  if ( v104 )
  {
    v281 = (char *)src + v104;
    v107 = (char *)src;
    do
    {
      v108 = (char *)sub_2207800(8 * v106);
      v109 = (unsigned __int64)v108;
      if ( v108 )
      {
        sub_295B550(v107, v281, v108, v106, a4);
        goto LABEL_159;
      }
      v106 >>= 1;
    }
    while ( v106 );
    v103 = v107;
    v105 = v281;
  }
  v109 = 0;
  sub_295A930(v103, v105, a4);
LABEL_159:
  j_j___libc_free_0(v109);
  v307 = 0;
  s = &v311;
  *(_QWORD *)v309 = 16;
  v114 = *(__int64 **)a1;
  *(_DWORD *)&v309[8] = 0;
  v292 = v294;
  v293 = 0x1000000000LL;
  v310 = 1;
  if ( HIDWORD(v303) != v304 )
  {
LABEL_160:
    if ( !(_DWORD)v290 )
      goto LABEL_210;
    v115 = *((_QWORD *)src + (unsigned int)v290 - 1);
    LODWORD(v290) = v290 - 1;
    v272 = v115;
    v116 = *(unsigned int *)(a4 + 24);
    v117 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v116 )
    {
      v116 = (unsigned int)(v116 - 1);
      v118 = v116 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
      v119 = (__int64 *)(v117 + 16LL * v118);
      v112 = *v119;
      if ( v115 == *v119 )
      {
LABEL_163:
        v282 = v119[1];
LABEL_164:
        if ( (__int64 *)v282 != v114 )
        {
          do
          {
            v120 = v302;
            if ( v305 )
            {
              v121 = HIDWORD(v303);
              v122 = &v302[HIDWORD(v303)];
            }
            else
            {
              v121 = (unsigned int)v303;
              v122 = &v302[(unsigned int)v303];
            }
            if ( v302 != v122 )
            {
              while ( 1 )
              {
                v123 = *v120;
                v124 = v120;
                if ( (unsigned __int64)*v120 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v122 == ++v120 )
                  goto LABEL_170;
              }
              if ( v122 != v120 )
              {
                if ( *((_BYTE *)v114 + 84) )
                  goto LABEL_248;
                while ( 1 )
                {
                  v172 = sub_C8CA60((__int64)(v114 + 7), v123);
                  if ( v172 )
                  {
                    *v172 = -2;
                    ++*((_DWORD *)v114 + 20);
                    ++v114[7];
                  }
LABEL_253:
                  v171 = v124 + 1;
                  if ( v124 + 1 == v122 )
                    goto LABEL_170;
                  v123 = *v171;
                  ++v124;
                  if ( (unsigned __int64)*v171 >= 0xFFFFFFFFFFFFFFFELL )
                    break;
LABEL_257:
                  if ( v122 == v124 )
                    goto LABEL_170;
                  if ( *((_BYTE *)v114 + 84) )
                  {
LABEL_248:
                    v169 = (__int64 *)v114[8];
                    v121 = (__int64)&v169[*((unsigned int *)v114 + 19)];
                    v116 = *((unsigned int *)v114 + 19);
                    v170 = v169;
                    if ( v169 != (__int64 *)v121 )
                    {
                      while ( *v170 != v123 )
                      {
                        if ( (__int64 *)v121 == ++v170 )
                          goto LABEL_253;
                      }
                      v116 = (unsigned int)(v116 - 1);
                      *((_DWORD *)v114 + 19) = v116;
                      v121 = v169[v116];
                      *v170 = v121;
                      ++v114[7];
                    }
                    goto LABEL_253;
                  }
                }
                while ( v122 != ++v171 )
                {
                  v123 = *v171;
                  v124 = v171;
                  if ( (unsigned __int64)*v171 < 0xFFFFFFFFFFFFFFFELL )
                    goto LABEL_257;
                }
              }
            }
LABEL_170:
            v117 = (unsigned __int64)&v301;
            sub_295D270(v114 + 4, (__int64)&v301, v121, v116);
            v114 = (__int64 *)*v114;
          }
          while ( v114 != (__int64 *)v282 );
        }
        v125 = (unsigned int)v293;
        v126 = (unsigned int)v293 + 1LL;
        if ( v126 > HIDWORD(v293) )
        {
          v117 = (unsigned __int64)v294;
          sub_C8D5F0((__int64)&v292, v294, v126, 8u, v112, v113);
          v125 = (unsigned int)v293;
        }
        v127 = v272;
        v273 = v114;
        *(_QWORD *)&v292[8 * v125] = v127;
        v128 = v293 + 1;
        LODWORD(v293) = v293 + 1;
LABEL_175:
        while ( 2 )
        {
          v129 = v128--;
          v130 = *(__int64 **)&v292[8 * v129 - 8];
          LODWORD(v293) = v128;
          if ( v276 == v130 || (v131 = v130[2]) == 0 )
          {
LABEL_174:
            if ( !v128 )
              goto LABEL_199;
            continue;
          }
          break;
        }
        while ( 1 )
        {
          v132 = *(_QWORD *)(v131 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v132 - 30) <= 0xAu )
            break;
          v131 = *(_QWORD *)(v131 + 8);
          if ( !v131 )
          {
            if ( !v128 )
            {
LABEL_199:
              v111 = v310;
              v114 = v273;
              k = (__int64 *)s;
              if ( v310 )
              {
                v143 = (char *)s + 8 * *(unsigned int *)&v309[4];
                if ( s != v143 )
                  goto LABEL_201;
                ++v307;
LABEL_208:
                *(_QWORD *)&v309[4] = 0;
                goto LABEL_209;
              }
              v143 = (char *)s + 8 * *(unsigned int *)v309;
              if ( s != v143 )
              {
LABEL_201:
                while ( 1 )
                {
                  v144 = *k;
                  v145 = k;
                  if ( (unsigned __int64)*k < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v143 == (char *)++k )
                    goto LABEL_203;
                }
                if ( v143 == (char *)k )
                {
LABEL_203:
                  ++v307;
                  if ( (_BYTE)v111 )
                    goto LABEL_208;
                  goto LABEL_204;
                }
                while ( 2 )
                {
                  v117 = *(unsigned int *)(a4 + 24);
                  v112 = *(_QWORD *)(a4 + 8);
                  if ( !(_DWORD)v117 )
                    goto LABEL_287;
                  v113 = (unsigned int)(v117 - 1);
                  v177 = v113 & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
                  v178 = (char *)(v112 + 16LL * v177);
                  v179 = *(_QWORD *)v178;
                  v180 = v178;
                  if ( v144 != *(_QWORD *)v178 )
                  {
                    v185 = v113 & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
                    v186 = 1;
                    while ( v179 != -4096 )
                    {
                      v185 = v113 & (v186 + v185);
                      v267 = v186 + 1;
                      v180 = (char *)(v112 + 16LL * v185);
                      v179 = *(_QWORD *)v180;
                      if ( *(_QWORD *)v180 == v144 )
                        goto LABEL_282;
                      v186 = v267;
                    }
                    goto LABEL_287;
                  }
LABEL_282:
                  v181 = (_QWORD *)*((_QWORD *)v180 + 1);
                  if ( !v181 )
                    goto LABEL_287;
                  if ( (_QWORD *)a1 != v181 )
                  {
                    do
                    {
                      v181 = (_QWORD *)*v181;
                      if ( (_QWORD *)a1 == v181 )
                        goto LABEL_287;
                    }
                    while ( v181 );
                  }
                  v284 = v144;
                  if ( !v282 )
                  {
                    v117 = *(_QWORD *)v178;
                    if ( v144 == *(_QWORD *)v178 )
                    {
LABEL_301:
                      *(_QWORD *)v178 = -8192;
                      --*(_DWORD *)(a4 + 16);
                      ++*(_DWORD *)(a4 + 20);
                    }
                    else
                    {
                      v240 = 1;
                      while ( v117 != -4096 )
                      {
                        v241 = v240 + 1;
                        v177 = v113 & (v177 + v240);
                        v178 = (char *)(v112 + 16LL * v177);
                        v117 = *(_QWORD *)v178;
                        if ( *(_QWORD *)v178 == v144 )
                          goto LABEL_301;
                        v240 = v241;
                      }
                    }
                    goto LABEL_287;
                  }
                  v182 = *(_QWORD *)v178;
                  v183 = 1;
                  v184 = 0;
                  if ( v144 == *(_QWORD *)v178 )
                    goto LABEL_295;
                  while ( v182 != -4096 )
                  {
                    if ( v182 == -8192 && !v184 )
                      v184 = v178;
                    v177 = v113 & (v183 + v177);
                    v178 = (char *)(v112 + 16LL * v177);
                    v182 = *(_QWORD *)v178;
                    if ( *(_QWORD *)v178 == v144 )
                      goto LABEL_295;
                    ++v183;
                  }
                  v187 = *(_DWORD *)(a4 + 16);
                  if ( v184 )
                    v178 = v184;
                  ++*(_QWORD *)a4;
                  v188 = v187 + 1;
                  v113 = (unsigned int)(4 * (v187 + 1));
                  v285 = v178;
                  if ( (unsigned int)v113 >= 3 * (int)v117 )
                  {
                    LODWORD(v117) = 2 * v117;
                  }
                  else
                  {
                    v112 = (unsigned int)v117 >> 3;
                    if ( (int)v117 - *(_DWORD *)(a4 + 20) - v188 > (unsigned int)v112 )
                    {
LABEL_315:
                      *(_DWORD *)(a4 + 16) = v188;
                      if ( *(_QWORD *)v178 != -4096 )
                        --*(_DWORD *)(a4 + 20);
                      *(_QWORD *)v178 = v144;
                      *((_QWORD *)v178 + 1) = 0;
LABEL_295:
                      *((_QWORD *)v178 + 1) = v282;
LABEL_287:
                      k = v145 + 1;
                      if ( v145 + 1 == (__int64 *)v143 )
                        goto LABEL_290;
                      while ( 1 )
                      {
                        v144 = *k;
                        v145 = k;
                        if ( (unsigned __int64)*k < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( v143 == (char *)++k )
                          goto LABEL_290;
                      }
                      if ( k == (__int64 *)v143 )
                      {
LABEL_290:
                        v114 = v273;
                        v111 = v310;
                        goto LABEL_203;
                      }
                      continue;
                    }
                  }
                  break;
                }
                sub_D4F150(a4, v117);
                v117 = (unsigned __int64)&v284;
                sub_D4C730(a4, &v284, &v285);
                v144 = v284;
                v178 = v285;
                v188 = *(_DWORD *)(a4 + 16) + 1;
                goto LABEL_315;
              }
              ++v307;
LABEL_204:
              v146 = 4 * (*(_DWORD *)&v309[4] - *(_DWORD *)&v309[8]);
              if ( v146 < 0x20 )
                v146 = 32;
              if ( v146 >= *(_DWORD *)v309 )
              {
                memset(s, -1, 8LL * *(unsigned int *)v309);
                goto LABEL_208;
              }
              sub_C8C990((__int64)&v307, v117);
LABEL_209:
              if ( HIDWORD(v303) == v304 )
                goto LABEL_210;
              goto LABEL_160;
            }
            goto LABEL_175;
          }
        }
        v133 = *(_QWORD *)(v132 + 40);
        if ( !v305 )
          goto LABEL_194;
LABEL_179:
        v117 = (unsigned __int64)v302;
        v134 = &v302[HIDWORD(v303)];
        v135 = v302;
        if ( v302 != v134 )
        {
          while ( v133 != *v135 )
          {
            if ( v134 == ++v135 )
              goto LABEL_191;
          }
          --HIDWORD(v303);
          v136 = HIDWORD(v303);
          v137 = (__int64 *)v302[HIDWORD(v303)];
          *v135 = v137;
          v301 = (__int64 *)((char *)v301 + 1);
          if ( !v310 )
            goto LABEL_196;
          goto LABEL_184;
        }
        do
        {
          do
          {
LABEL_191:
            v131 = *(_QWORD *)(v131 + 8);
            if ( !v131 )
            {
              v128 = v293;
              goto LABEL_174;
            }
            v141 = *(_QWORD *)(v131 + 24);
          }
          while ( (unsigned __int8)(*(_BYTE *)v141 - 30) > 0xAu );
          v133 = *(_QWORD *)(v141 + 40);
          if ( v305 )
            goto LABEL_179;
LABEL_194:
          v117 = v133;
          v142 = sub_C8CA60((__int64)&v301, v133);
        }
        while ( !v142 );
        *v142 = -2;
        ++v304;
        v301 = (__int64 *)((char *)v301 + 1);
        if ( v310 )
        {
LABEL_184:
          v138 = s;
          v136 = *(unsigned int *)&v309[4];
          v137 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v309[4]);
          if ( s == v137 )
          {
LABEL_261:
            if ( *(_DWORD *)&v309[4] >= *(_DWORD *)v309 )
              goto LABEL_196;
            ++*(_DWORD *)&v309[4];
            *v137 = v133;
            ++v307;
          }
          else
          {
            while ( v133 != *v138 )
            {
              if ( v137 == ++v138 )
                goto LABEL_261;
            }
          }
        }
        else
        {
LABEL_196:
          v117 = v133;
          sub_C8CC70((__int64)&v307, v133, (__int64)v137, v136, v112, v113);
        }
        v139 = (unsigned int)v293;
        v140 = (unsigned int)v293 + 1LL;
        if ( v140 > HIDWORD(v293) )
        {
          v117 = (unsigned __int64)v294;
          sub_C8D5F0((__int64)&v292, v294, v140, 8u, v112, v113);
          v139 = (unsigned int)v293;
        }
        *(_QWORD *)&v292[8 * v139] = v133;
        LODWORD(v293) = v293 + 1;
        goto LABEL_191;
      }
      v189 = 1;
      while ( v112 != -4096 )
      {
        v242 = v189 + 1;
        v118 = v116 & (v189 + v118);
        v119 = (__int64 *)(v117 + 16LL * v118);
        v112 = *v119;
        if ( v272 == *v119 )
          goto LABEL_163;
        v189 = v242;
      }
    }
    v282 = 0;
    goto LABEL_164;
  }
LABEL_210:
  while ( v114 )
  {
    v147 = v302;
    if ( v305 )
    {
      v148 = HIDWORD(v303);
      v149 = &v302[HIDWORD(v303)];
    }
    else
    {
      v148 = (unsigned int)v303;
      v149 = &v302[(unsigned int)v303];
    }
    if ( v302 != v149 )
    {
      while ( 1 )
      {
        v150 = *v147;
        v151 = v147;
        if ( (unsigned __int64)*v147 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v149 == ++v147 )
          goto LABEL_216;
      }
      if ( v147 != v149 )
      {
        if ( *((_BYTE *)v114 + 84) )
        {
LABEL_266:
          v173 = (__int64 *)v114[8];
          v148 = (__int64)&v173[*((unsigned int *)v114 + 19)];
          v111 = *((unsigned int *)v114 + 19);
          v174 = v173;
          if ( v173 != (__int64 *)v148 )
          {
            while ( *v174 != v150 )
            {
              if ( (__int64 *)v148 == ++v174 )
                goto LABEL_271;
            }
            v111 = (unsigned int)(v111 - 1);
            *((_DWORD *)v114 + 19) = v111;
            v148 = v173[v111];
            *v174 = v148;
            ++v114[7];
          }
          goto LABEL_271;
        }
        while ( 1 )
        {
          v176 = sub_C8CA60((__int64)(v114 + 7), v150);
          if ( v176 )
          {
            *v176 = -2;
            ++*((_DWORD *)v114 + 20);
            ++v114[7];
          }
LABEL_271:
          v175 = v151 + 1;
          if ( v151 + 1 == v149 )
            break;
          v150 = *v175;
          for ( ++v151; (unsigned __int64)*v175 >= 0xFFFFFFFFFFFFFFFELL; v151 = v175 )
          {
            if ( v149 == ++v175 )
              goto LABEL_216;
            v150 = *v175;
          }
          if ( v151 == v149 )
            break;
          if ( *((_BYTE *)v114 + 84) )
            goto LABEL_266;
        }
      }
    }
LABEL_216:
    sub_295D270(v114 + 4, (__int64)&v301, v148, v111);
    v114 = (__int64 *)*v114;
  }
  v152 = v302;
  if ( v305 )
    v153 = &v302[HIDWORD(v303)];
  else
    v153 = &v302[(unsigned int)v303];
  if ( v302 != v153 )
  {
    while ( 1 )
    {
      k = (__int64 *)*v152;
      v154 = v152;
      if ( (unsigned __int64)*v152 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v153 == ++v152 )
        goto LABEL_222;
    }
    if ( v152 != v153 )
    {
      do
      {
        v213 = *(_DWORD *)(a4 + 24);
        v214 = *(_QWORD *)(a4 + 8);
        if ( v213 )
        {
          v215 = v213 - 1;
          v112 = v215 & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
          v216 = (__int64 *)(v214 + 16 * v112);
          v113 = *v216;
          v217 = v216;
          if ( (__int64 *)*v216 == k )
          {
LABEL_372:
            v218 = (_QWORD *)v217[1];
            if ( v218 )
            {
              if ( (_QWORD *)a1 == v218 )
              {
LABEL_383:
                v220 = 1;
                if ( (__int64 *)v113 == k )
                {
LABEL_384:
                  *v216 = -8192;
                  --*(_DWORD *)(a4 + 16);
                  ++*(_DWORD *)(a4 + 20);
                }
                else
                {
                  while ( v113 != -4096 )
                  {
                    v112 = v215 & (unsigned int)(v220 + v112);
                    v216 = (__int64 *)(v214 + 16LL * (unsigned int)v112);
                    v113 = *v216;
                    if ( k == (__int64 *)*v216 )
                      goto LABEL_384;
                    ++v220;
                  }
                }
              }
              else
              {
                while ( 1 )
                {
                  v218 = (_QWORD *)*v218;
                  if ( (_QWORD *)a1 == v218 )
                    break;
                  if ( !v218 )
                    goto LABEL_383;
                }
              }
            }
          }
          else
          {
            v221 = (_QWORD *)*v216;
            v222 = v112;
            v223 = 1;
            while ( v221 != (_QWORD *)-4096LL )
            {
              v224 = v223 + 1;
              v222 = v215 & (v223 + v222);
              v217 = (__int64 *)(v214 + 16LL * v222);
              v221 = (_QWORD *)*v217;
              if ( k == (__int64 *)*v217 )
                goto LABEL_372;
              v223 = v224;
            }
          }
        }
        if ( ++v154 == v153 )
          break;
        v219 = v154;
        for ( k = (__int64 *)*v154; (unsigned __int64)*v219 >= 0xFFFFFFFFFFFFFFFELL; v154 = v219 )
        {
          if ( v153 == ++v219 )
            goto LABEL_222;
          k = (__int64 *)*v219;
        }
      }
      while ( v154 != v153 );
    }
  }
LABEL_222:
  v155 = *(char **)(a1 + 8);
  if ( HIDWORD(v297) == v298 )
    goto LABEL_334;
  v156 = *(char **)(a1 + 16);
  v157 = (v156 - v155) >> 5;
  v158 = (v156 - v155) >> 3;
  if ( v157 <= 0 )
  {
LABEL_241:
    if ( v158 != 2 )
    {
      if ( v158 != 3 )
      {
        if ( v158 != 1 )
          goto LABEL_333;
        goto LABEL_509;
      }
      if ( !(unsigned __int8)sub_B19060((__int64)&v295, **(_QWORD **)(*(_QWORD *)v155 + 32LL), (__int64)k, 3) )
        goto LABEL_327;
      v155 += 8;
    }
    if ( !(unsigned __int8)sub_B19060((__int64)&v295, **(_QWORD **)(*(_QWORD *)v155 + 32LL), (__int64)k, v158) )
      goto LABEL_327;
    v155 += 8;
LABEL_509:
    if ( !(unsigned __int8)sub_B19060((__int64)&v295, **(_QWORD **)(*(_QWORD *)v155 + 32LL), (__int64)k, v158) )
      goto LABEL_327;
    goto LABEL_333;
  }
  v159 = &v155[32 * v157];
  while ( 1 )
  {
    v160 = **(_QWORD **)(*(_QWORD *)v155 + 32LL);
    if ( v299 )
      break;
    if ( !sub_C8CA60((__int64)&v295, v160) )
      goto LABEL_327;
    v163 = v155 + 8;
    v164 = **(_QWORD **)(*((_QWORD *)v155 + 1) + 32LL);
    if ( v299 )
    {
      v161 = v296;
      k = &v296[HIDWORD(v297)];
      if ( k != v296 )
      {
        v165 = v296;
LABEL_232:
        while ( v164 != *v161 )
        {
          if ( k == ++v161 )
            goto LABEL_326;
        }
        v163 = v155 + 16;
        v166 = **(_QWORD **)(*((_QWORD *)v155 + 2) + 32LL);
        v167 = v165;
        do
        {
LABEL_235:
          if ( v166 == *v165 )
          {
            v163 = v155 + 24;
            v168 = **(_QWORD **)(*((_QWORD *)v155 + 3) + 32LL);
            goto LABEL_238;
          }
          ++v165;
        }
        while ( k != v165 );
      }
      goto LABEL_326;
    }
    if ( !sub_C8CA60((__int64)&v295, v164) )
      goto LABEL_326;
    v163 = v155 + 16;
    v166 = **(_QWORD **)(*((_QWORD *)v155 + 2) + 32LL);
    if ( v299 )
    {
      v165 = v296;
      k = &v296[HIDWORD(v297)];
      if ( k != v296 )
      {
        v167 = v296;
        goto LABEL_235;
      }
LABEL_326:
      v155 = v163;
      goto LABEL_327;
    }
    if ( !sub_C8CA60((__int64)&v295, v166) )
      goto LABEL_326;
    v163 = v155 + 24;
    v168 = **(_QWORD **)(*((_QWORD *)v155 + 3) + 32LL);
    if ( v299 )
    {
      v167 = v296;
      k = &v296[HIDWORD(v297)];
      if ( v296 == k )
        goto LABEL_326;
LABEL_238:
      while ( v168 != *v167 )
      {
        if ( k == ++v167 )
          goto LABEL_326;
      }
      v155 += 32;
      if ( v155 == v159 )
        goto LABEL_240;
    }
    else
    {
      if ( !sub_C8CA60((__int64)&v295, v168) )
        goto LABEL_326;
      v155 += 32;
      if ( v155 == v159 )
      {
LABEL_240:
        v158 = (v156 - v155) >> 3;
        goto LABEL_241;
      }
    }
  }
  v161 = v296;
  k = &v296[HIDWORD(v297)];
  if ( v296 != k )
  {
    v162 = v296;
    do
    {
      if ( v160 == *v162 )
      {
        v163 = v155 + 8;
        v164 = **(_QWORD **)(*((_QWORD *)v155 + 1) + 32LL);
        v165 = v296;
        goto LABEL_232;
      }
      ++v162;
    }
    while ( k != v162 );
  }
LABEL_327:
  if ( v155 != v156 )
  {
    v190 = (v156 - v155) >> 3;
    if ( v156 - v155 <= 0 )
    {
LABEL_544:
      v193 = 0;
      v194 = 0;
    }
    else
    {
      v191 = (v156 - v155) >> 3;
      while ( 1 )
      {
        v274 = v190;
        v192 = sub_2207800(8 * v191);
        v190 = v274;
        v193 = (char *)v192;
        if ( v192 )
          break;
        v191 >>= 1;
        if ( !v191 )
          goto LABEL_544;
      }
      v194 = v191;
    }
    v156 = sub_295C1F0(v155, v156, (__int64)&v295, v190, v193, v194);
    j_j___libc_free_0((unsigned __int64)v193);
  }
LABEL_333:
  v155 = v156;
LABEL_334:
  v195 = *(char ***)(a1 + 16);
  if ( v195 != (char **)v155 )
  {
    v283 = v155;
    v196 = (char **)v155;
    while ( 1 )
    {
      v203 = *(unsigned int *)(a5 + 8);
      v204 = *v196;
      if ( v203 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        sub_C8D5F0(a5, (const void *)(a5 + 16), v203 + 1, 8u, v112, v113);
        v203 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v203) = v204;
      ++*(_DWORD *)(a5 + 8);
      *(_QWORD *)v204 = 0;
      v205 = sub_D4B130((__int64)v204);
      v206 = *(_DWORD *)(a4 + 24);
      v207 = *(_QWORD *)(a4 + 8);
      if ( !v206 )
        goto LABEL_346;
      v197 = v206 - 1;
      v198 = v197 & (((unsigned int)v205 >> 9) ^ ((unsigned int)v205 >> 4));
      v199 = (__int64 *)(v207 + 16LL * v198);
      v200 = *v199;
      if ( v205 != *v199 )
      {
        v211 = 1;
        while ( v200 != -4096 )
        {
          v112 = (unsigned int)(v211 + 1);
          v212 = v197 & (v198 + v211);
          v198 = v212;
          v199 = (__int64 *)(v207 + 16 * v212);
          v200 = *v199;
          if ( v205 == *v199 )
            goto LABEL_337;
          v211 = v112;
        }
        goto LABEL_346;
      }
LABEL_337:
      v201 = v199[1];
      if ( v201 )
      {
        v285 = v204;
        *(_QWORD *)v204 = v201;
        v202 = *(_BYTE **)(v201 + 16);
        if ( v202 == *(_BYTE **)(v201 + 24) )
        {
          sub_D4C7F0(v201 + 8, v202, &v285);
        }
        else
        {
          if ( v202 )
          {
            *(_QWORD *)v202 = v285;
            v202 = *(_BYTE **)(v201 + 16);
          }
          *(_QWORD *)(v201 + 16) = v202 + 8;
        }
        if ( v195 == ++v196 )
        {
LABEL_347:
          v155 = v283;
          v208 = *(char **)(a1 + 16);
          goto LABEL_348;
        }
      }
      else
      {
LABEL_346:
        ++v196;
        v285 = v204;
        sub_D4C980(a4 + 32, &v285);
        if ( v195 == v196 )
          goto LABEL_347;
      }
    }
  }
  v208 = v155;
LABEL_348:
  v209 = 1;
  sub_295D580(a1 + 8, v155, v208);
  if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)(a1 + 32) )
  {
    v243 = *(__int64 **)a1;
    if ( *(_QWORD *)a1 )
    {
      v285 = (char *)a1;
      v244 = sub_29578B0((_QWORD *)v243[1], v243[2], (__int64 *)&v285);
      v246 = *(_QWORD **)v244;
      v247 = v244;
      sub_D4C9B0(v245 + 8, v244);
      *v246 = 0;
    }
    else
    {
      v248 = *(_QWORD *)(a4 + 40);
      v249 = *(_QWORD **)(a4 + 32);
      v285 = (char *)a1;
      v247 = sub_29578B0(v249, v248, (__int64 *)&v285);
      sub_D4C9B0(a4 + 32, v247);
    }
    if ( a6 )
    {
      v247 = 0;
      sub_D9D700(a6, 0);
    }
    v209 = 0;
    sub_D47BB0(a1, (__int64)v247);
  }
  if ( v292 != v294 )
    _libc_free((unsigned __int64)v292);
  if ( !v310 )
    _libc_free((unsigned __int64)s);
  if ( !v305 )
    _libc_free((unsigned __int64)v302);
  if ( !v299 )
    _libc_free((unsigned __int64)v296);
  if ( src != v291 )
    _libc_free((unsigned __int64)src);
  if ( v286 != v288 )
    _libc_free((unsigned __int64)v286);
  return v209;
}
