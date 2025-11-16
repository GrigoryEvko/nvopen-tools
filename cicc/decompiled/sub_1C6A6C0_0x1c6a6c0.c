// Function: sub_1C6A6C0
// Address: 0x1c6a6c0
//
_BOOL8 __fastcall sub_1C6A6C0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // r8
  int v13; // edx
  int v14; // edx
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // r12
  char *v24; // rax
  __int64 v25; // rdx
  int v26; // esi
  int v27; // r8d
  unsigned int v28; // ecx
  char **v29; // rdx
  char *v30; // r10
  char **v31; // r9
  char *v32; // r10
  char *v33; // rax
  char *v34; // rdx
  int v35; // esi
  __int64 v36; // r8
  unsigned int v37; // ecx
  char **v38; // rax
  char *v39; // r10
  __int64 v40; // rsi
  _BOOL4 v41; // r12d
  int v42; // r14d
  unsigned int v43; // edx
  _QWORD *v44; // rbx
  __int64 v45; // rax
  _QWORD *v46; // r13
  unsigned __int64 v47; // rdi
  int v48; // eax
  unsigned int v49; // ecx
  __int64 v50; // rdx
  _QWORD *v51; // rax
  _QWORD *k; // rdx
  int v53; // eax
  __int64 v54; // rdx
  _QWORD *v55; // rax
  _QWORD *m; // rdx
  int v57; // eax
  _QWORD *v58; // rdi
  unsigned int v59; // edx
  __int64 v60; // rsi
  _QWORD *v61; // rdx
  _QWORD *ii; // rax
  int v63; // eax
  __int64 v64; // rdx
  _QWORD *v65; // rax
  _QWORD *kk; // rdx
  void *v67; // rdi
  int v69; // eax
  char *v70; // r9
  unsigned int v71; // eax
  unsigned int v72; // r11d
  int v73; // r9d
  int v74; // r11d
  char **v75; // r10
  char *v76; // rax
  int v77; // eax
  int v78; // edi
  int v79; // r11d
  char **v80; // r9
  int v81; // ecx
  unsigned __int64 v82; // rdi
  unsigned int v83; // ecx
  unsigned int v84; // eax
  int v85; // r14d
  unsigned int v86; // eax
  unsigned int v87; // ecx
  _QWORD *v88; // rdi
  unsigned int v89; // eax
  int v90; // r13d
  unsigned int v91; // eax
  _QWORD *v92; // rax
  __int64 v93; // rdx
  _QWORD *n; // rdx
  _QWORD *v95; // r13
  _QWORD *v96; // rbx
  _QWORD *v97; // r14
  int v98; // eax
  __int64 v99; // rax
  __int64 v100; // rbx
  double v101; // xmm4_8
  double v102; // xmm5_8
  unsigned __int64 v103; // rdx
  __int64 v104; // rdi
  unsigned __int64 v105; // r13
  __int64 v106; // rax
  __int64 v107; // r15
  __int64 *v108; // rbx
  __int64 v109; // r13
  __int64 v110; // r15
  bool v111; // al
  __int64 v112; // rax
  char *v113; // rdi
  char *v114; // rax
  _BYTE *v115; // rsi
  int v116; // r8d
  unsigned int v117; // esi
  char **v118; // rcx
  char *v119; // r9
  __int64 v120; // r12
  __int64 v121; // rax
  __int64 v122; // r14
  __int64 v123; // rax
  __int64 v124; // r12
  char v125; // al
  int v126; // r13d
  __int64 v127; // rax
  __int64 v128; // rdx
  __int64 v129; // r15
  int v130; // r15d
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // r13
  __int64 v134; // r15
  __int64 v135; // rax
  __int64 v136; // rax
  int v137; // r13d
  __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // r15
  int v141; // r15d
  __int64 v142; // rax
  __int64 v143; // rdx
  int v144; // eax
  int v145; // r15d
  __int64 v146; // r13
  __int64 v147; // rbx
  __int64 *v148; // rbx
  __int64 v149; // rsi
  __int64 v150; // rdx
  __int64 v151; // rax
  bool v152; // zf
  _BYTE *v153; // rsi
  char *v154; // r8
  int v155; // r9d
  unsigned int v156; // edx
  char **v157; // rax
  char *v158; // rcx
  __int64 v159; // rbx
  __int64 v160; // rax
  int v161; // ecx
  _QWORD *v162; // rax
  __int64 *v163; // rsi
  __int64 *v164; // rbx
  __int64 v165; // rdi
  int v166; // esi
  unsigned int v167; // edx
  char *v168; // rax
  __int64 v169; // r8
  __int64 v170; // r15
  __int64 v171; // r14
  int v172; // r12d
  unsigned int v173; // r13d
  char v174; // cl
  __int64 v175; // rdx
  unsigned int v176; // edi
  __int64 v177; // rax
  __int64 *v178; // rdx
  __int64 v179; // rbx
  __int64 v180; // rax
  unsigned int v181; // edx
  __int64 *v182; // rax
  __int64 v183; // rdx
  __int64 v184; // rdx
  unsigned int v185; // eax
  __int64 v186; // rdx
  unsigned __int64 v187; // r12
  _QWORD *v188; // rax
  _QWORD *v189; // r13
  __int64 v190; // rcx
  __int64 v191; // rdx
  __int64 v192; // rdx
  __int64 v193; // r13
  __int64 *v194; // r12
  __int64 v195; // rax
  int v196; // r11d
  char *v197; // rcx
  int v198; // edx
  __int64 v199; // rdx
  _QWORD *v200; // rax
  _QWORD *v201; // rdx
  unsigned __int64 v202; // r12
  __int64 v203; // r13
  char **v204; // rax
  char v205; // dl
  char *v206; // rbx
  char **v207; // rax
  char **v208; // rcx
  char **v209; // rsi
  __int64 v210; // rsi
  unsigned __int64 v211; // rcx
  char v212; // r8
  char v213; // al
  bool v214; // al
  __int64 *v215; // r13
  __int64 *j; // rbx
  __int64 v217; // r15
  unsigned __int64 *v218; // r14
  __int64 v219; // rdi
  __int64 v220; // rax
  __int64 v221; // rbx
  __int64 v222; // r13
  __int64 v223; // rax
  __int64 v224; // rdi
  __int64 v225; // rdx
  __int64 *v226; // rsi
  unsigned int v227; // ecx
  _QWORD *v228; // rdi
  unsigned int v229; // eax
  int v230; // eax
  unsigned __int64 v231; // r12
  unsigned int v232; // eax
  _QWORD *v233; // rax
  _QWORD *i; // rdx
  unsigned int v235; // edx
  char *v236; // rax
  __int64 v237; // r10
  __int64 v238; // r13
  __int64 v239; // rax
  __int64 v240; // rdx
  __int64 *v241; // rsi
  int v242; // esi
  char *v243; // r9
  int v244; // edx
  __int64 v245; // rax
  int v246; // r11d
  void *v247; // rax
  __int64 v248; // rax
  __int64 v249; // rax
  __int64 v250; // r12
  unsigned __int64 v251; // rdx
  const char *v252; // r15
  size_t v253; // r14
  unsigned __int64 v254; // rax
  _QWORD *v255; // rdx
  size_t v256; // rdx
  char *v257; // rsi
  __int64 v258; // rax
  __int64 v259; // rax
  _QWORD *v260; // rax
  __int64 v261; // rdx
  _BOOL8 v262; // rdi
  int v263; // eax
  _QWORD *v264; // rax
  int v265; // r12d
  char **v266; // r9
  int v267; // eax
  int v268; // r10d
  int v269; // edx
  int v270; // r13d
  unsigned int v271; // eax
  unsigned int v272; // eax
  int v273; // eax
  _QWORD *v274; // rdi
  unsigned int v275; // eax
  int v276; // r14d
  unsigned int v277; // eax
  _QWORD *v278; // rdx
  unsigned int v279; // eax
  int v280; // eax
  unsigned __int64 v281; // rax
  unsigned __int64 v282; // rax
  int v283; // ebx
  __int64 v284; // r13
  _QWORD *v285; // rax
  __int64 v286; // rdx
  _QWORD *jj; // rdx
  _QWORD *v288; // rax
  int v289; // r10d
  __int64 v290; // rax
  _QWORD *v291; // rax
  __int64 v292; // [rsp+18h] [rbp-288h]
  unsigned int v293; // [rsp+34h] [rbp-26Ch]
  __int64 v295; // [rsp+40h] [rbp-260h]
  __int64 v296; // [rsp+48h] [rbp-258h]
  __int64 v297; // [rsp+48h] [rbp-258h]
  __int64 *v298; // [rsp+48h] [rbp-258h]
  __int64 v299; // [rsp+48h] [rbp-258h]
  _QWORD *v300; // [rsp+48h] [rbp-258h]
  _QWORD *v301; // [rsp+48h] [rbp-258h]
  int v302; // [rsp+50h] [rbp-250h]
  _QWORD *v303; // [rsp+58h] [rbp-248h]
  __int64 v304; // [rsp+58h] [rbp-248h]
  __int64 v305; // [rsp+58h] [rbp-248h]
  int v306; // [rsp+58h] [rbp-248h]
  __int64 v308; // [rsp+68h] [rbp-238h]
  __int64 v309; // [rsp+68h] [rbp-238h]
  __int64 *v310; // [rsp+68h] [rbp-238h]
  __int64 *v311; // [rsp+78h] [rbp-228h] BYREF
  __int64 *v312; // [rsp+80h] [rbp-220h] BYREF
  __int64 *v313; // [rsp+88h] [rbp-218h]
  __int64 *v314; // [rsp+90h] [rbp-210h]
  __int64 *v315; // [rsp+A0h] [rbp-200h] BYREF
  __int64 *v316; // [rsp+A8h] [rbp-1F8h]
  __int64 *v317; // [rsp+B0h] [rbp-1F0h]
  __int64 v318; // [rsp+C0h] [rbp-1E0h] BYREF
  _BYTE *v319; // [rsp+C8h] [rbp-1D8h]
  _BYTE *v320; // [rsp+D0h] [rbp-1D0h]
  __int64 v321; // [rsp+E0h] [rbp-1C0h] BYREF
  void *src; // [rsp+E8h] [rbp-1B8h]
  __int64 v323; // [rsp+F0h] [rbp-1B0h]
  __int64 v324; // [rsp+F8h] [rbp-1A8h]
  __int64 v325; // [rsp+100h] [rbp-1A0h] BYREF
  _QWORD *v326; // [rsp+108h] [rbp-198h]
  __int64 v327; // [rsp+110h] [rbp-190h]
  unsigned int v328; // [rsp+118h] [rbp-188h]
  char *v329; // [rsp+120h] [rbp-180h] BYREF
  __int64 v330; // [rsp+128h] [rbp-178h]
  _BYTE v331[64]; // [rsp+130h] [rbp-170h] BYREF
  _QWORD *v332; // [rsp+170h] [rbp-130h] BYREF
  char **v333; // [rsp+178h] [rbp-128h]
  char **v334; // [rsp+180h] [rbp-120h]
  __int64 v335; // [rsp+188h] [rbp-118h]
  int v336; // [rsp+190h] [rbp-110h]
  _BYTE v337[64]; // [rsp+198h] [rbp-108h] BYREF
  unsigned __int64 v338; // [rsp+1D8h] [rbp-C8h] BYREF
  unsigned __int64 v339; // [rsp+1E0h] [rbp-C0h]
  __int64 v340; // [rsp+1E8h] [rbp-B8h]
  _QWORD v341[22]; // [rsp+1F0h] [rbp-B0h] BYREF

  if ( byte_4FBC840 )
    sub_1C5A4D0(a2);
  v293 = 0;
  v10 = *(_QWORD *)(a2 + 80);
  v308 = a2 + 72;
  if ( v10 != a2 + 72 )
  {
    while ( 1 )
    {
      v11 = v10 - 24;
      if ( !v10 )
        v11 = 0;
      v12 = *(_QWORD *)(a1 + 192);
      v13 = *(_DWORD *)(v12 + 24);
      if ( v13 )
      {
        v14 = v13 - 1;
        v15 = *(_QWORD *)(v12 + 8);
        v16 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v11 == *v17 )
        {
LABEL_8:
          v19 = v17[1];
          if ( v19 && *(_QWORD *)(v19 + 8) == *(_QWORD *)(v19 + 16) && **(_QWORD **)(v19 + 32) == v11 )
          {
            v71 = sub_1BF8310(v11, 1, v12);
            if ( v293 >= v71 )
              v71 = v293;
            v293 = v71;
          }
        }
        else
        {
          v77 = 1;
          while ( v18 != -8 )
          {
            v78 = v77 + 1;
            v16 = v14 & (v77 + v16);
            v17 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v17;
            if ( v11 == *v17 )
              goto LABEL_8;
            v77 = v78;
          }
        }
      }
      memset(v341, 0, 28);
      v20 = v11 + 40;
      v21 = 0;
      v22 = *(_QWORD *)(v20 + 8);
      if ( v22 != v20 )
        break;
LABEL_27:
      j___libc_free_0(v21);
      v10 = *(_QWORD *)(v10 + 8);
      if ( v308 == v10 )
        goto LABEL_28;
    }
    while ( 1 )
    {
      v23 = v22;
      v22 = *(_QWORD *)(v22 + 8);
      if ( *(_BYTE *)(v23 - 8) != 54 )
        goto LABEL_12;
      v24 = *(char **)(v23 - 48);
      v25 = *(_QWORD *)v24;
      if ( *(_BYTE *)(*(_QWORD *)v24 + 8LL) == 16 )
        v25 = **(_QWORD **)(v25 + 16);
      if ( *(_DWORD *)(v25 + 8) >> 8 != 4 )
        goto LABEL_12;
      v26 = v341[3];
      if ( !LODWORD(v341[3]) )
      {
        ++v341[0];
        v329 = v24;
        goto LABEL_75;
      }
      v27 = LODWORD(v341[3]) - 1;
      v28 = (LODWORD(v341[3]) - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v29 = (char **)(v21 + 16LL * v28);
      v30 = *v29;
      v31 = v29;
      if ( v24 != *v29 )
      {
        v72 = (LODWORD(v341[3]) - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v73 = 1;
        while ( v30 != (char *)-8LL )
        {
          v72 = v27 & (v73 + v72);
          v306 = v73 + 1;
          v31 = (char **)(v21 + 16LL * v72);
          v30 = *v31;
          if ( v24 == *v31 )
            goto LABEL_19;
          v73 = v306;
        }
        v329 = *(char **)(v23 - 48);
        v28 = v27 & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
        v29 = (char **)(v21 + 16LL * v28);
        v70 = *v29;
        if ( v24 == *v29 )
          goto LABEL_78;
        goto LABEL_86;
      }
LABEL_19:
      if ( v31 == (char **)(v21 + 16LL * LODWORD(v341[3])) )
      {
        v329 = *(char **)(v23 - 48);
        v70 = *v29;
        if ( v24 == *v29 )
        {
LABEL_78:
          v29[1] = (char *)(v23 - 24);
          v21 = v341[1];
          goto LABEL_12;
        }
LABEL_86:
        v74 = 1;
        v75 = 0;
        while ( v70 != (char *)-8LL )
        {
          if ( v70 == (char *)-16LL && !v75 )
            v75 = v29;
          v28 = v27 & (v74 + v28);
          v29 = (char **)(v21 + 16LL * v28);
          v70 = *v29;
          if ( v24 == *v29 )
            goto LABEL_78;
          ++v74;
        }
        if ( v75 )
          v29 = v75;
        ++v341[0];
        v69 = LODWORD(v341[2]) + 1;
        if ( 4 * (LODWORD(v341[2]) + 1) < (unsigned int)(3 * LODWORD(v341[3])) )
        {
          if ( (unsigned int)(LODWORD(v341[3]) - HIDWORD(v341[2]) - v69) > LODWORD(v341[3]) >> 3 )
            goto LABEL_92;
          goto LABEL_76;
        }
LABEL_75:
        v26 = 2 * LODWORD(v341[3]);
LABEL_76:
        sub_1C5A760((__int64)v341, v26);
        sub_1C56A20((__int64)v341, (__int64 *)&v329, &v332);
        v29 = (char **)v332;
        v69 = LODWORD(v341[2]) + 1;
LABEL_92:
        LODWORD(v341[2]) = v69;
        if ( *v29 != (char *)-8LL )
          --HIDWORD(v341[2]);
        v76 = v329;
        v29[1] = 0;
        *v29 = v76;
        goto LABEL_78;
      }
      v329 = *(char **)(v23 - 48);
      v32 = *v29;
      if ( v24 != *v29 )
        break;
LABEL_21:
      v33 = v29[1];
LABEL_22:
      if ( *(_QWORD *)v33 == *(_QWORD *)(v23 - 24) )
      {
        v34 = *(char **)(v23 - 48);
        v35 = v341[3];
        v36 = v23 - 24;
        v329 = v34;
        if ( !LODWORD(v341[3]) )
        {
          ++v341[0];
          goto LABEL_467;
        }
        v37 = (LODWORD(v341[3]) - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v38 = (char **)(v21 + 16LL * v37);
        v39 = *v38;
        if ( v34 != *v38 )
        {
          v265 = 1;
          v266 = 0;
          while ( v39 != (char *)-8LL )
          {
            if ( !v266 && v39 == (char *)-16LL )
              v266 = v38;
            v37 = (LODWORD(v341[3]) - 1) & (v265 + v37);
            v38 = (char **)(v21 + 16LL * v37);
            v39 = *v38;
            if ( v34 == *v38 )
              goto LABEL_25;
            ++v265;
          }
          if ( !v266 )
            v266 = v38;
          ++v341[0];
          v267 = LODWORD(v341[2]) + 1;
          if ( 4 * (LODWORD(v341[2]) + 1) < (unsigned int)(3 * LODWORD(v341[3])) )
          {
            if ( (unsigned int)(LODWORD(v341[3]) - HIDWORD(v341[2]) - v267) > LODWORD(v341[3]) >> 3 )
            {
LABEL_447:
              LODWORD(v341[2]) = v267;
              if ( *v266 != (char *)-8LL )
                --HIDWORD(v341[2]);
              *v266 = v34;
              v40 = 0;
              v266[1] = 0;
              goto LABEL_26;
            }
            v305 = v36;
LABEL_465:
            sub_1C5A760((__int64)v341, v35);
            sub_1C56A20((__int64)v341, (__int64 *)&v329, &v332);
            v266 = (char **)v332;
            v34 = v329;
            v36 = v305;
            v267 = LODWORD(v341[2]) + 1;
            goto LABEL_447;
          }
LABEL_467:
          v305 = v36;
          v35 = 2 * LODWORD(v341[3]);
          goto LABEL_465;
        }
LABEL_25:
        v40 = (__int64)v38[1];
LABEL_26:
        v303 = (_QWORD *)v36;
        sub_164D160(
          v36,
          v40,
          a3,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128_u64,
          *(double *)a6.m128_u64,
          a7,
          a8,
          a9,
          a10);
        sub_15F20C0(v303);
        v21 = v341[1];
        if ( v20 == v22 )
          goto LABEL_27;
      }
      else
      {
LABEL_12:
        if ( v20 == v22 )
          goto LABEL_27;
      }
    }
    v79 = 1;
    v80 = 0;
    while ( v32 != (char *)-8LL )
    {
      if ( v32 == (char *)-16LL && !v80 )
        v80 = v29;
      v28 = v27 & (v79 + v28);
      v29 = (char **)(v21 + 16LL * v28);
      v32 = *v29;
      if ( v24 == *v29 )
        goto LABEL_21;
      ++v79;
    }
    if ( !v80 )
      v80 = v29;
    ++v341[0];
    v81 = LODWORD(v341[2]) + 1;
    if ( 4 * (LODWORD(v341[2]) + 1) >= (unsigned int)(3 * LODWORD(v341[3])) )
    {
      v26 = 2 * LODWORD(v341[3]);
    }
    else if ( (unsigned int)(LODWORD(v341[3]) - HIDWORD(v341[2]) - v81) > LODWORD(v341[3]) >> 3 )
    {
LABEL_105:
      LODWORD(v341[2]) = v81;
      if ( *v80 != (char *)-8LL )
        --HIDWORD(v341[2]);
      *v80 = v24;
      v33 = 0;
      v80[1] = 0;
      v21 = v341[1];
      goto LABEL_22;
    }
    sub_1C5A760((__int64)v341, v26);
    sub_1C56A20((__int64)v341, (__int64 *)&v329, &v332);
    v80 = (char **)v332;
    v24 = v329;
    v81 = LODWORD(v341[2]) + 1;
    goto LABEL_105;
  }
LABEL_28:
  if ( !dword_4FBD020 )
    goto LABEL_29;
  v247 = sub_16E8CB0();
  v248 = sub_1263B40((__int64)v247, "phi maxLoopInd = ");
  v249 = sub_16E7A90(v248, v293);
  v250 = sub_1263B40(v249, ": Function ");
  v252 = sub_1649960(a2);
  v253 = v251;
  if ( v252 )
  {
    v332 = (_QWORD *)v251;
    v254 = v251;
    v341[0] = &v341[2];
    if ( v251 > 0xF )
    {
      v341[0] = sub_22409D0(v341, &v332, 0);
      v274 = (_QWORD *)v341[0];
      v341[2] = v332;
    }
    else
    {
      if ( v251 == 1 )
      {
        LOBYTE(v341[2]) = *v252;
        v255 = &v341[2];
LABEL_427:
        v341[1] = v254;
        *((_BYTE *)v255 + v254) = 0;
        v256 = v341[1];
        v257 = (char *)v341[0];
        goto LABEL_428;
      }
      if ( !v251 )
      {
        v255 = &v341[2];
        goto LABEL_427;
      }
      v274 = &v341[2];
    }
    memcpy(v274, v252, v253);
    v254 = (unsigned __int64)v332;
    v255 = (_QWORD *)v341[0];
    goto LABEL_427;
  }
  LOBYTE(v341[2]) = 0;
  v256 = 0;
  v341[0] = &v341[2];
  v257 = (char *)&v341[2];
  v341[1] = 0;
LABEL_428:
  v258 = sub_16E7EE0(v250, v257, v256);
  sub_1263B40(v258, "\n");
  if ( (_QWORD *)v341[0] != &v341[2] )
    j_j___libc_free_0(v341[0], v341[2] + 1LL);
LABEL_29:
  v41 = 0;
  v321 = 0;
  src = 0;
  v323 = 0;
  v324 = 0;
  if ( !dword_4FBD1E0 )
    goto LABEL_30;
  v325 = 0;
  v326 = 0;
  v99 = *(_QWORD *)(a1 + 200);
  v327 = 0;
  v328 = 0;
  v312 = 0;
  v100 = *(_QWORD *)(v99 + 56);
  v313 = 0;
  v314 = 0;
  v315 = 0;
  v316 = 0;
  v317 = 0;
  v318 = 0;
  v319 = 0;
  v320 = 0;
  v332 = 0;
  v333 = (char **)v337;
  v334 = (char **)v337;
  v335 = 8;
  v336 = 0;
  v338 = 0;
  v339 = 0;
  v340 = 0;
  sub_1412190((__int64)&v332, v100);
  v341[0] = v100;
  LOBYTE(v341[2]) = 0;
  sub_13B8390(&v338, (__int64)v341);
  memset(v341, 0, 0x80u);
  v103 = v339;
  v104 = 0;
  v105 = v338;
  v341[1] = &v341[5];
  v341[2] = &v341[5];
  LODWORD(v341[3]) = 8;
  v302 = 0;
  v292 = a1 + 72;
  if ( v339 == v338 )
    goto LABEL_319;
  do
  {
LABEL_169:
    v106 = **(_QWORD **)(v103 - 24);
    v304 = v106 + 40;
    v309 = *(_QWORD *)(v106 + 48);
    if ( v309 != v106 + 40 )
    {
      while ( 1 )
      {
        v122 = v309;
        v123 = *(_QWORD *)(v309 + 8);
        v311 = 0;
        v124 = v309 - 24;
        v309 = v123;
        v125 = *(_BYTE *)(v122 - 8);
        if ( v125 == 54 || v125 == 55 )
        {
          v107 = *(_QWORD *)(v122 - 48);
          v108 = (__int64 *)sub_22077B0(32);
          if ( v108 )
          {
            v109 = *(_QWORD *)(a1 + 184);
            v108[3] = v107;
            v108[2] = v124;
            v110 = sub_146F1B0(v109, v107);
            v111 = sub_14560B0(v110);
            v108[1] = v110;
            if ( v111 )
            {
              v136 = sub_1456040(v110);
              *v108 = sub_145CF80(v109, v136, 0, 0);
            }
            else
            {
              v330 = 0x800000000LL;
              v329 = v331;
              v112 = sub_1456040(v110);
              *v108 = sub_145CF80(v109, v112, 0, 0);
              sub_1C54710(v110, 0, (__int64)&v329, v109, v108, (__m128i)a3, a4);
              if ( (_DWORD)v330 )
              {
                if ( (unsigned int)v330 == 1 )
                {
                  v113 = v329;
                  v108[1] = *(_QWORD *)v329;
                }
                else
                {
                  v108[1] = (__int64)sub_147DD40(v109, (__int64 *)&v329, 0, 0, (__m128i)a3, a4);
                  v113 = v329;
                }
                if ( v113 == v331 )
                  goto LABEL_178;
              }
              else
              {
                v108[1] = 0;
                v113 = v329;
                if ( v329 == v331 )
                  goto LABEL_178;
              }
              _libc_free((unsigned __int64)v113);
            }
          }
        }
        else
        {
          if ( v125 != 78 )
            goto LABEL_190;
          v126 = *(_DWORD *)(v122 - 4) & 0xFFFFFFF;
          if ( *(char *)(v122 - 1) < 0 )
          {
            v127 = sub_1648A40(v124);
            v129 = v127 + v128;
            if ( *(char *)(v122 - 1) >= 0 )
            {
              if ( (unsigned int)(v129 >> 4) )
LABEL_563:
                BUG();
            }
            else if ( (unsigned int)((v129 - sub_1648A40(v124)) >> 4) )
            {
              if ( *(char *)(v122 - 1) >= 0 )
                goto LABEL_563;
              v130 = *(_DWORD *)(sub_1648A40(v124) + 8);
              if ( *(char *)(v122 - 1) >= 0 )
                goto LABEL_564;
              v131 = sub_1648A40(v124);
              v126 += v130 - *(_DWORD *)(v131 + v132 - 4);
            }
          }
          if ( v126 != 2 )
            goto LABEL_404;
          v133 = *(_QWORD *)(v124 - 24LL * (*(_DWORD *)(v122 - 4) & 0xFFFFFFF));
          if ( !v133 )
            BUG();
          if ( *(_BYTE *)(*(_QWORD *)v133 + 8LL) != 15 )
          {
LABEL_404:
            v108 = v311;
            goto LABEL_179;
          }
          v108 = (__int64 *)sub_22077B0(32);
          if ( v108 )
          {
            v108[2] = v124;
            v108[3] = v133;
            v134 = *(_QWORD *)(a1 + 184);
            v135 = sub_146F1B0(v134, v133);
            sub_1C54F70(v108, v135, v134, (__m128i)a3, a4);
          }
        }
LABEL_178:
        v311 = v108;
LABEL_179:
        if ( v108 )
        {
          v114 = (char *)v108[1];
          v329 = v114;
          if ( v114 )
          {
            v115 = v319;
            if ( v319 == v320 )
            {
              sub_1C510A0((__int64)&v318, v319, &v311);
              v114 = v329;
            }
            else
            {
              if ( v319 )
              {
                *(_QWORD *)v319 = v108;
                v115 = v319;
                v114 = v329;
              }
              v319 = v115 + 8;
            }
            ++v302;
            if ( v328 )
            {
              v116 = v328 - 1;
              v117 = (v328 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
              v118 = (char **)&v326[2 * v117];
              v119 = *v118;
              if ( v114 == *v118 )
              {
LABEL_187:
                if ( v118 != &v326[2 * v328] )
                {
                  v120 = (__int64)v118[1];
                  v121 = *(unsigned int *)(v120 + 8);
                  if ( (unsigned int)v121 < *(_DWORD *)(v120 + 12) )
                  {
LABEL_189:
                    *(_QWORD *)(*(_QWORD *)v120 + 8 * v121) = v311;
                    ++*(_DWORD *)(v120 + 8);
                    goto LABEL_190;
                  }
LABEL_240:
                  sub_16CD150(v120, (const void *)(v120 + 16), 0, 8, v116, (int)v119);
                  v121 = *(unsigned int *)(v120 + 8);
                  goto LABEL_189;
                }
              }
              else
              {
                v161 = 1;
                while ( v119 != (char *)-8LL )
                {
                  v268 = v161 + 1;
                  v117 = v116 & (v161 + v117);
                  v118 = (char **)&v326[2 * v117];
                  v119 = *v118;
                  if ( v114 == *v118 )
                    goto LABEL_187;
                  v161 = v268;
                }
              }
            }
            v162 = (_QWORD *)sub_22077B0(80);
            if ( v162 )
            {
              *v162 = v162 + 2;
              v162[1] = 0x800000000LL;
            }
            sub_1C53170((__int64)&v325, (__int64 *)&v329)[1] = (__int64)v162;
            v163 = v313;
            if ( v313 == v314 )
            {
              sub_1C55B50((__int64)&v312, v313, &v329);
            }
            else
            {
              if ( v313 )
              {
                *v313 = (__int64)v329;
                v163 = v313;
              }
              v313 = v163 + 1;
            }
            v120 = sub_1C53170((__int64)&v325, (__int64 *)&v329)[1];
            v121 = *(unsigned int *)(v120 + 8);
            if ( (unsigned int)v121 < *(_DWORD *)(v120 + 12) )
              goto LABEL_189;
            goto LABEL_240;
          }
          j_j___libc_free_0(v108, 32);
          if ( v304 == v309 )
            break;
        }
        else
        {
          if ( *(_BYTE *)(v122 - 8) == 78 )
          {
            v137 = *(_DWORD *)(v122 - 4) & 0xFFFFFFF;
            if ( *(char *)(v122 - 1) < 0 )
            {
              v138 = sub_1648A40(v124);
              v140 = v138 + v139;
              if ( *(char *)(v122 - 1) >= 0 )
              {
                if ( (unsigned int)(v140 >> 4) )
                  goto LABEL_563;
              }
              else if ( (unsigned int)((v140 - sub_1648A40(v124)) >> 4) )
              {
                if ( *(char *)(v122 - 1) >= 0 )
                  goto LABEL_563;
                v141 = *(_DWORD *)(sub_1648A40(v124) + 8);
                if ( *(char *)(v122 - 1) >= 0 )
LABEL_564:
                  BUG();
                v142 = sub_1648A40(v124);
                v144 = *(_DWORD *)(v142 + v143 - 4) - v141;
LABEL_213:
                v145 = v137 - 1 - v144;
                if ( v137 - 1 == v144 )
                  goto LABEL_190;
                v146 = 0;
                while ( 2 )
                {
                  v147 = *(_QWORD *)(v124 + 24 * (v146 - (*(_DWORD *)(v122 - 4) & 0xFFFFFFF)));
                  if ( *(_BYTE *)(v147 + 16) <= 0x10u
                    || !sub_1642F90(*(_QWORD *)v147, 32)
                    && !sub_1642F90(*(_QWORD *)v147, 64)
                    && *(_BYTE *)(*(_QWORD *)v147 + 8LL) != 15
                    || (unsigned int)sub_1C5A920(
                                       *(_QWORD *)(v124 + 24 * (v146 - (*(_DWORD *)(v122 - 4) & 0xFFFFFFF))),
                                       dword_4FBC300) > dword_4FBC300 )
                  {
                    goto LABEL_215;
                  }
                  v296 = *(_QWORD *)(v124 + 24 * (v146 - (*(_DWORD *)(v122 - 4) & 0xFFFFFFF)));
                  v148 = (__int64 *)sub_22077B0(32);
                  if ( v148 )
                  {
                    v149 = v296;
                    v148[2] = v124;
                    v150 = *(_QWORD *)(a1 + 184);
                    v148[3] = v296;
                    v297 = v150;
                    v151 = sub_146F1B0(v150, v149);
                    sub_1C54F70(v148, v151, v297, (__m128i)a3, a4);
                  }
                  v152 = v148[1] == 0;
                  v311 = v148;
                  if ( v152 )
                  {
                    j_j___libc_free_0(v148, 32);
                    goto LABEL_215;
                  }
                  v153 = v319;
                  if ( v319 == v320 )
                  {
                    sub_1C510A0((__int64)&v318, v319, &v311);
                    v148 = v311;
                  }
                  else
                  {
                    if ( v319 )
                    {
                      *(_QWORD *)v319 = v148;
                      v153 = v319;
                    }
                    v319 = v153 + 8;
                  }
                  v154 = (char *)v148[1];
                  ++v302;
                  v329 = v154;
                  if ( v328 )
                  {
                    v155 = v328 - 1;
                    v156 = (v328 - 1) & (((unsigned int)v154 >> 9) ^ ((unsigned int)v154 >> 4));
                    v157 = (char **)&v326[2 * v156];
                    v158 = *v157;
                    if ( v154 == *v157 )
                    {
LABEL_228:
                      if ( v157 != &v326[2 * v328] )
                      {
                        v159 = (__int64)v157[1];
                        v160 = *(unsigned int *)(v159 + 8);
                        if ( (unsigned int)v160 < *(_DWORD *)(v159 + 12) )
                          goto LABEL_230;
LABEL_440:
                        sub_16CD150(v159, (const void *)(v159 + 16), 0, 8, (int)v154, v155);
                        v160 = *(unsigned int *)(v159 + 8);
LABEL_230:
                        *(_QWORD *)(*(_QWORD *)v159 + 8 * v160) = v311;
                        ++*(_DWORD *)(v159 + 8);
LABEL_215:
                        if ( ++v146 == v145 )
                          goto LABEL_190;
                        continue;
                      }
                    }
                    else
                    {
                      v263 = 1;
                      while ( v158 != (char *)-8LL )
                      {
                        v289 = v263 + 1;
                        v290 = v155 & (v156 + v263);
                        v156 = v290;
                        v157 = (char **)&v326[2 * v290];
                        v158 = *v157;
                        if ( v154 == *v157 )
                          goto LABEL_228;
                        v263 = v289;
                      }
                    }
                  }
                  break;
                }
                v264 = (_QWORD *)sub_22077B0(80);
                if ( v264 )
                {
                  *v264 = v264 + 2;
                  v264[1] = 0x800000000LL;
                }
                sub_1C53170((__int64)&v325, (__int64 *)&v329)[1] = (__int64)v264;
                sub_1C55CE0((__int64)&v312, &v329);
                v159 = sub_1C53170((__int64)&v325, (__int64 *)&v329)[1];
                v160 = *(unsigned int *)(v159 + 8);
                if ( (unsigned int)v160 >= *(_DWORD *)(v159 + 12) )
                  goto LABEL_440;
                goto LABEL_230;
              }
            }
            v144 = 0;
            goto LABEL_213;
          }
LABEL_190:
          if ( v304 == v309 )
            break;
        }
      }
    }
    v164 = v312;
    v310 = v313;
    if ( !dword_4FBCAE0 )
    {
      if ( v312 == v313 )
        goto LABEL_295;
      while ( 1 )
      {
        v242 = v328;
        if ( !v328 )
          break;
        v235 = (v328 - 1) & (((unsigned int)*v164 >> 9) ^ ((unsigned int)*v164 >> 4));
        v236 = (char *)&v326[2 * v235];
        v237 = *(_QWORD *)v236;
        if ( *(_QWORD *)v236 != *v164 )
        {
          v246 = 1;
          v243 = 0;
          while ( v237 != -8 )
          {
            if ( v237 == -16 && !v243 )
              v243 = v236;
            v235 = (v328 - 1) & (v246 + v235);
            v236 = (char *)&v326[2 * v235];
            v237 = *(_QWORD *)v236;
            if ( *v164 == *(_QWORD *)v236 )
              goto LABEL_388;
            ++v246;
          }
          if ( !v243 )
            v243 = v236;
          ++v325;
          v244 = v327 + 1;
          if ( 4 * ((int)v327 + 1) < 3 * v328 )
          {
            if ( v328 - HIDWORD(v327) - v244 <= v328 >> 3 )
            {
LABEL_399:
              sub_1C52FC0((__int64)&v325, v242);
              sub_1C50640((__int64)&v325, v164, &v329);
              v243 = v329;
              v244 = v327 + 1;
            }
            LODWORD(v327) = v244;
            if ( *(_QWORD *)v243 != -8 )
              --HIDWORD(v327);
            v245 = *v164;
            v238 = 0;
            *((_QWORD *)v243 + 1) = 0;
            *(_QWORD *)v243 = v245;
            goto LABEL_389;
          }
LABEL_398:
          v242 = 2 * v328;
          goto LABEL_399;
        }
LABEL_388:
        v238 = *((_QWORD *)v236 + 1);
LABEL_389:
        v239 = sub_22077B0(16);
        if ( v239 )
        {
          v240 = *v164;
          *(_QWORD *)(v239 + 8) = v238;
          *(_QWORD *)v239 = v240;
        }
        v329 = (char *)v239;
        v241 = v316;
        if ( v316 == v317 )
        {
          sub_1C50F10((__int64)&v315, v316, &v329);
        }
        else
        {
          if ( v316 )
          {
            *v316 = v239;
            v241 = v316;
          }
          v316 = v241 + 1;
        }
        if ( v310 == ++v164 )
          goto LABEL_295;
      }
      ++v325;
      goto LABEL_398;
    }
    if ( v312 != v313 )
    {
      while ( 1 )
      {
        v165 = *v164;
        v166 = v328;
        v311 = (__int64 *)*v164;
        if ( !v328 )
          break;
        v167 = (v328 - 1) & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
        v168 = (char *)&v326[2 * v167];
        v169 = *(_QWORD *)v168;
        if ( v165 != *(_QWORD *)v168 )
        {
          v196 = 1;
          v197 = 0;
          while ( v169 != -8 )
          {
            if ( !v197 && v169 == -16 )
              v197 = v168;
            v167 = (v328 - 1) & (v196 + v167);
            v168 = (char *)&v326[2 * v167];
            v169 = *(_QWORD *)v168;
            if ( v165 == *(_QWORD *)v168 )
              goto LABEL_251;
            ++v196;
          }
          if ( !v197 )
            v197 = v168;
          ++v325;
          v198 = v327 + 1;
          if ( 4 * ((int)v327 + 1) < 3 * v328 )
          {
            if ( v328 - HIDWORD(v327) - v198 > v328 >> 3 )
            {
LABEL_291:
              LODWORD(v327) = v198;
              if ( *(_QWORD *)v197 != -8 )
                --HIDWORD(v327);
              *(_QWORD *)v197 = v165;
              *((_QWORD *)v197 + 1) = 0;
              goto LABEL_294;
            }
LABEL_409:
            sub_1C52FC0((__int64)&v325, v166);
            sub_1C50640((__int64)&v325, (__int64 *)&v311, &v329);
            v197 = v329;
            v165 = (__int64)v311;
            v198 = v327 + 1;
            goto LABEL_291;
          }
LABEL_408:
          v166 = 2 * v328;
          goto LABEL_409;
        }
LABEL_251:
        v170 = *((_QWORD *)v168 + 1);
        if ( v170 && *(_DWORD *)(v170 + 8) )
        {
          if ( *(_BYTE *)(sub_1456040(v165) + 8) != 15 )
            goto LABEL_359;
          v171 = ***(_QWORD ***)v170;
          if ( sub_14560B0(v171) )
            goto LABEL_359;
          v172 = *(_DWORD *)(v170 + 8);
          if ( v172 != 1 )
          {
            v298 = v164;
            v173 = 1;
            while ( 1 )
            {
              v179 = **(_QWORD **)(*(_QWORD *)v170 + 8LL * v173);
              if ( sub_14560B0(v179) )
              {
                v164 = v298;
                goto LABEL_359;
              }
              v180 = *(_QWORD *)(v171 + 32);
              v181 = *(_DWORD *)(v180 + 32);
              v182 = *(__int64 **)(v180 + 24);
              if ( v181 <= 0x40 )
              {
                v174 = 64 - v181;
                v175 = *(_QWORD *)(v179 + 32);
                v176 = *(_DWORD *)(v175 + 32);
                v177 = (__int64)((_QWORD)v182 << v174) >> v174;
                v178 = *(__int64 **)(v175 + 24);
                if ( v176 <= 0x40 )
                  goto LABEL_258;
LABEL_264:
                if ( *v178 < v177 )
                  v171 = v179;
                if ( v172 == ++v173 )
                  goto LABEL_267;
              }
              else
              {
                v183 = *(_QWORD *)(v179 + 32);
                v177 = *v182;
                v176 = *(_DWORD *)(v183 + 32);
                v178 = *(__int64 **)(v183 + 24);
                if ( v176 > 0x40 )
                  goto LABEL_264;
LABEL_258:
                if ( (__int64)((_QWORD)v178 << (64 - (unsigned __int8)v176)) >> (64 - (unsigned __int8)v176) < v177 )
                  v171 = v179;
                if ( v172 == ++v173 )
                {
LABEL_267:
                  v164 = v298;
                  break;
                }
              }
            }
          }
          if ( dword_4FBCAE0 <= 1 )
            goto LABEL_272;
          v184 = *(_QWORD *)(v171 + 32);
          v185 = *(_DWORD *)(v184 + 32);
          v186 = *(_QWORD *)(v184 + 24);
          if ( v185 > 0x40 )
            v186 = *(_QWORD *)(v186 + 8LL * ((v185 - 1) >> 6));
          if ( (v186 & (1LL << ((unsigned __int8)v185 - 1))) != 0 || sub_1C51340((__int64)v311) )
          {
LABEL_272:
            v187 = sub_13A5B00(*(_QWORD *)(a1 + 184), (__int64)v311, v171, 0, 0);
            v188 = *(_QWORD **)(a1 + 80);
            v189 = (_QWORD *)(a1 + 72);
            if ( !v188 )
              goto LABEL_430;
            do
            {
              while ( 1 )
              {
                v190 = v188[2];
                v191 = v188[3];
                if ( v188[4] >= v187 )
                  break;
                v188 = (_QWORD *)v188[3];
                if ( !v191 )
                  goto LABEL_277;
              }
              v189 = v188;
              v188 = (_QWORD *)v188[2];
            }
            while ( v190 );
LABEL_277:
            if ( v189 == (_QWORD *)v292 || v189[4] > v187 )
            {
LABEL_430:
              v300 = v189;
              v259 = sub_22077B0(48);
              *(_QWORD *)(v259 + 32) = v187;
              v189 = (_QWORD *)v259;
              *(_QWORD *)(v259 + 40) = 0;
              v260 = sub_1C57250((_QWORD *)(a1 + 64), v300, (unsigned __int64 *)(v259 + 32));
              if ( v261 )
              {
                v262 = v260 || v292 == v261 || v187 < *(_QWORD *)(v261 + 32);
                sub_220F040(v262, v189, v261, v292);
                ++*(_QWORD *)(a1 + 104);
              }
              else
              {
                v301 = v260;
                j_j___libc_free_0(v189, 48);
                v189 = v301;
              }
            }
            v189[5] = v311;
            v192 = *(unsigned int *)(v170 + 8);
            v193 = 0;
            v295 = 8 * v192;
            if ( (_DWORD)v192 )
            {
              v299 = v187;
              do
              {
                v194 = *(__int64 **)(*(_QWORD *)v170 + v193);
                v193 += 8;
                *v194 = sub_14806B0(*(_QWORD *)(a1 + 184), *v194, v171, 0, 0);
                v194[1] = v299;
              }
              while ( v295 != v193 );
              v187 = v299;
            }
            v195 = sub_22077B0(16);
            if ( v195 )
            {
              *(_QWORD *)v195 = v187;
              *(_QWORD *)(v195 + 8) = v170;
            }
          }
          else
          {
LABEL_359:
            v195 = sub_22077B0(16);
            if ( v195 )
            {
              v225 = (__int64)v311;
              *(_QWORD *)(v195 + 8) = v170;
              *(_QWORD *)v195 = v225;
            }
          }
          v329 = (char *)v195;
          v226 = v316;
          if ( v316 == v317 )
          {
            ++v164;
            sub_1C50F10((__int64)&v315, v316, &v329);
            if ( v310 == v164 )
              goto LABEL_295;
          }
          else
          {
            if ( v316 )
            {
              *v316 = v195;
              v226 = v316;
            }
            ++v164;
            v316 = v226 + 1;
            if ( v310 == v164 )
              goto LABEL_295;
          }
        }
        else
        {
LABEL_294:
          if ( v310 == ++v164 )
            goto LABEL_295;
        }
      }
      ++v325;
      goto LABEL_408;
    }
LABEL_295:
    ++v325;
    if ( (_DWORD)v327 )
    {
      v227 = 4 * v327;
      v199 = v328;
      if ( (unsigned int)(4 * v327) < 0x40 )
        v227 = 64;
      if ( v328 <= v227 )
        goto LABEL_298;
      v228 = v326;
      if ( (_DWORD)v327 == 1 )
      {
        v231 = 86;
      }
      else
      {
        _BitScanReverse(&v229, v327 - 1);
        v230 = 1 << (33 - (v229 ^ 0x1F));
        if ( v230 < 64 )
          v230 = 64;
        if ( v328 == v230 )
        {
          v327 = 0;
          v288 = &v326[2 * v328];
          do
          {
            if ( v228 )
              *v228 = -8;
            v228 += 2;
          }
          while ( v288 != v228 );
          goto LABEL_301;
        }
        v231 = 4 * v230 / 3u + 1;
      }
      j___libc_free_0(v326);
      v232 = sub_1454B60(v231);
      v328 = v232;
      if ( v232 )
      {
        v233 = (_QWORD *)sub_22077B0(16LL * v232);
        v327 = 0;
        v326 = v233;
        for ( i = &v233[2 * v328]; i != v233; v233 += 2 )
        {
          if ( v233 )
            *v233 = -8;
        }
        goto LABEL_301;
      }
      goto LABEL_474;
    }
    if ( HIDWORD(v327) )
    {
      v199 = v328;
      if ( v328 <= 0x40 )
      {
LABEL_298:
        v200 = v326;
        v201 = &v326[2 * v199];
        if ( v326 != v201 )
        {
          do
          {
            *v200 = -8;
            v200 += 2;
          }
          while ( v201 != v200 );
        }
        goto LABEL_300;
      }
      j___libc_free_0(v326);
      v328 = 0;
LABEL_474:
      v326 = 0;
LABEL_300:
      v327 = 0;
    }
LABEL_301:
    if ( v312 != v313 )
      v313 = v312;
    v202 = v339;
    do
    {
      v203 = *(_QWORD *)(v202 - 24);
      if ( !*(_BYTE *)(v202 - 8) )
      {
        v204 = *(char ***)(v203 + 24);
        *(_BYTE *)(v202 - 8) = 1;
        *(_QWORD *)(v202 - 16) = v204;
        goto LABEL_308;
      }
      while ( 1 )
      {
        v204 = *(char ***)(v202 - 16);
LABEL_308:
        if ( v204 == *(char ***)(v203 + 32) )
          break;
        *(_QWORD *)(v202 - 16) = v204 + 1;
        v206 = *v204;
        v207 = v333;
        if ( v334 != v333 )
          goto LABEL_306;
        v208 = &v333[HIDWORD(v335)];
        if ( v333 == v208 )
        {
LABEL_366:
          if ( HIDWORD(v335) < (unsigned int)v335 )
          {
            ++HIDWORD(v335);
            *v208 = v206;
            v332 = (_QWORD *)((char *)v332 + 1);
LABEL_317:
            v329 = v206;
            v331[0] = 0;
            sub_13B8390(&v338, (__int64)&v329);
            v105 = v338;
            v103 = v339;
            goto LABEL_318;
          }
LABEL_306:
          sub_16CCBA0((__int64)&v332, (__int64)v206);
          if ( v205 )
            goto LABEL_317;
        }
        else
        {
          v209 = 0;
          while ( v206 != *v207 )
          {
            if ( *v207 == (char *)-2LL )
            {
              v209 = v207;
              if ( v207 + 1 == v208 )
                goto LABEL_316;
              ++v207;
            }
            else if ( v208 == ++v207 )
            {
              if ( !v209 )
                goto LABEL_366;
LABEL_316:
              *v209 = v206;
              --v336;
              v332 = (_QWORD *)((char *)v332 + 1);
              goto LABEL_317;
            }
          }
        }
      }
      v339 -= 24LL;
      v105 = v338;
      v202 = v339;
    }
    while ( v339 != v338 );
    v103 = v338;
LABEL_318:
    v104 = v341[13];
  }
  while ( v103 - v105 != v341[14] - v341[13] );
LABEL_319:
  if ( v105 != v103 )
  {
    v210 = v104;
    v211 = v105;
    while ( *(_QWORD *)v211 == *(_QWORD *)v210 )
    {
      v212 = *(_BYTE *)(v211 + 16);
      v213 = *(_BYTE *)(v210 + 16);
      if ( v212 && v213 )
        v214 = *(_QWORD *)(v211 + 8) == *(_QWORD *)(v210 + 8);
      else
        v214 = v212 == v213;
      if ( !v214 )
        break;
      v211 += 24LL;
      v210 += 24;
      if ( v211 == v103 )
        goto LABEL_327;
    }
    goto LABEL_169;
  }
LABEL_327:
  if ( v104 )
  {
    j_j___libc_free_0(v104, v341[15] - v104);
    v105 = v338;
  }
  if ( v341[2] != v341[1] )
    _libc_free(v341[2]);
  if ( v105 )
    j_j___libc_free_0(v105, v340 - v105);
  if ( v334 != v333 )
    _libc_free((unsigned __int64)v334);
  v41 = 0;
  if ( v302 > 5 )
    v41 = sub_1C67780(a1, a2, v293, &v315, (__int64)&v321, (__m128i)a3, a4, a5, a6, v101, v102, a9, a10);
  v215 = v316;
  for ( j = v315; v215 != j; ++j )
  {
    v217 = *j;
    v218 = *(unsigned __int64 **)(*j + 8);
    if ( v218 )
    {
      if ( (unsigned __int64 *)*v218 != v218 + 2 )
        _libc_free(*v218);
      j_j___libc_free_0(v218, 80);
    }
    j_j___libc_free_0(v217, 16);
  }
  v219 = v318;
  v220 = (__int64)&v319[-v318] >> 3;
  if ( (_DWORD)v220 )
  {
    v221 = 0;
    v222 = 8LL * (unsigned int)v220;
    v223 = v318;
    do
    {
      v224 = *(_QWORD *)(v223 + v221);
      if ( v224 )
      {
        j_j___libc_free_0(v224, 32);
        v223 = v318;
      }
      v221 += 8;
    }
    while ( v222 != v221 );
    v219 = v223;
  }
  if ( v219 )
    j_j___libc_free_0(v219, &v320[-v219]);
  if ( v315 )
    j_j___libc_free_0(v315, (char *)v317 - (char *)v315);
  if ( v312 )
    j_j___libc_free_0(v312, (char *)v314 - (char *)v312);
  j___libc_free_0(v326);
LABEL_30:
  if ( dword_4FBD2C0 )
  {
    memset(v341, 0, 32);
    if ( dword_4FBD2C0 == 1 )
    {
      j___libc_free_0(0);
      LODWORD(v341[3]) = v324;
      if ( (_DWORD)v324 )
      {
        v341[1] = sub_22077B0(8LL * (unsigned int)v324);
        v341[2] = v323;
        memcpy((void *)v341[1], src, 8LL * LODWORD(v341[3]));
      }
      else
      {
        v341[1] = 0;
        v341[2] = 0;
      }
    }
    if ( dword_4FBCD80 )
    {
      v273 = sub_1C5FDC0((_QWORD *)a1, a2, (__int64)v341, 1, (__m128i)a3, a4);
      if ( (_BYTE)v273 )
        v41 = v273;
    }
    v98 = sub_1C5FDC0((_QWORD *)a1, a2, (__int64)v341, 0, (__m128i)a3, a4);
    if ( (_BYTE)v98 )
      v41 = v98;
    j___libc_free_0(v341[1]);
  }
  if ( !byte_4FBC840 )
    sub_1C5A4D0(a2);
  v42 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v42 || *(_DWORD *)(a1 + 20) )
  {
    v43 = 4 * v42;
    v44 = *(_QWORD **)(a1 + 8);
    v45 = *(unsigned int *)(a1 + 24);
    v46 = &v44[4 * v45];
    if ( (unsigned int)(4 * v42) < 0x40 )
      v43 = 64;
    if ( (unsigned int)v45 <= v43 )
    {
      for ( ; v44 != v46; v44 += 4 )
      {
        if ( *v44 != -8 )
        {
          if ( *v44 != -16 )
          {
            v47 = v44[1];
            if ( (_QWORD *)v47 != v44 + 3 )
              _libc_free(v47);
          }
          *v44 = -8;
        }
      }
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_45;
    }
    do
    {
      if ( *v44 != -16 && *v44 != -8 )
      {
        v82 = v44[1];
        if ( (_QWORD *)v82 != v44 + 3 )
          _libc_free(v82);
      }
      v44 += 4;
    }
    while ( v44 != v46 );
    v269 = *(_DWORD *)(a1 + 24);
    if ( v42 )
    {
      v270 = 64;
      if ( v42 != 1 )
      {
        _BitScanReverse(&v271, v42 - 1);
        v270 = 1 << (33 - (v271 ^ 0x1F));
        if ( v270 < 64 )
          v270 = 64;
      }
      if ( v270 != v269 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        v272 = sub_1C521A0(v270);
        *(_DWORD *)(a1 + 24) = v272;
        if ( v272 )
        {
          *(_QWORD *)(a1 + 8) = sub_22077B0(32LL * v272);
          sub_1C55ED0(a1);
          goto LABEL_45;
        }
        goto LABEL_505;
      }
    }
    else if ( v269 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_DWORD *)(a1 + 24) = 0;
LABEL_505:
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_45;
    }
    sub_1C55ED0(a1);
  }
LABEL_45:
  v48 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( !v48 )
  {
    if ( !*(_DWORD *)(a1 + 132) )
      goto LABEL_52;
    v50 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)v50 <= 0x40 )
      goto LABEL_49;
    j___libc_free_0(*(_QWORD *)(a1 + 120));
    *(_DWORD *)(a1 + 136) = 0;
    goto LABEL_113;
  }
  v49 = 4 * v48;
  v50 = *(unsigned int *)(a1 + 136);
  if ( (unsigned int)(4 * v48) < 0x40 )
    v49 = 64;
  if ( (unsigned int)v50 > v49 )
  {
    v275 = v48 - 1;
    if ( v275 )
    {
      _BitScanReverse(&v275, v275);
      v276 = 1 << (33 - (v275 ^ 0x1F));
      if ( v276 < 64 )
        v276 = 64;
      if ( (_DWORD)v50 == v276 )
        goto LABEL_521;
    }
    else
    {
      v276 = 64;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 120));
    v277 = sub_1C521A0(v276);
    *(_DWORD *)(a1 + 136) = v277;
    if ( v277 )
    {
      *(_QWORD *)(a1 + 120) = sub_22077B0(32LL * v277);
LABEL_521:
      sub_1C55F50(a1 + 112);
      goto LABEL_52;
    }
LABEL_113:
    *(_QWORD *)(a1 + 120) = 0;
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_52;
  }
LABEL_49:
  v51 = *(_QWORD **)(a1 + 120);
  for ( k = &v51[4 * v50]; k != v51; *(v51 - 3) = -8 )
  {
    *v51 = -8;
    v51 += 4;
  }
  *(_QWORD *)(a1 + 128) = 0;
LABEL_52:
  v53 = *(_DWORD *)(a1 + 160);
  ++*(_QWORD *)(a1 + 144);
  if ( v53 )
  {
    v87 = 4 * v53;
    v54 = *(unsigned int *)(a1 + 168);
    if ( (unsigned int)(4 * v53) < 0x40 )
      v87 = 64;
    if ( v87 >= (unsigned int)v54 )
    {
LABEL_55:
      v55 = *(_QWORD **)(a1 + 152);
      for ( m = &v55[4 * v54]; m != v55; *(v55 - 3) = -8 )
      {
        *v55 = -8;
        v55 += 4;
      }
      *(_QWORD *)(a1 + 160) = 0;
      goto LABEL_58;
    }
    v88 = *(_QWORD **)(a1 + 152);
    v89 = v53 - 1;
    if ( v89 )
    {
      _BitScanReverse(&v89, v89);
      v90 = 1 << (33 - (v89 ^ 0x1F));
      if ( v90 < 64 )
        v90 = 64;
      if ( (_DWORD)v54 == v90 )
      {
        *(_QWORD *)(a1 + 160) = 0;
        v291 = &v88[4 * (unsigned int)v54];
        do
        {
          if ( v88 )
          {
            *v88 = -8;
            v88[1] = -8;
          }
          v88 += 4;
        }
        while ( v291 != v88 );
        goto LABEL_58;
      }
    }
    else
    {
      v90 = 64;
    }
    j___libc_free_0(v88);
    v91 = sub_1C521A0(v90);
    *(_DWORD *)(a1 + 168) = v91;
    if ( !v91 )
      goto LABEL_507;
    v92 = (_QWORD *)sub_22077B0(32LL * v91);
    v93 = *(unsigned int *)(a1 + 168);
    *(_QWORD *)(a1 + 152) = v92;
    *(_QWORD *)(a1 + 160) = 0;
    for ( n = &v92[4 * v93]; n != v92; v92 += 4 )
    {
      if ( v92 )
      {
        *v92 = -8;
        v92[1] = -8;
      }
    }
  }
  else if ( *(_DWORD *)(a1 + 164) )
  {
    v54 = *(unsigned int *)(a1 + 168);
    if ( (unsigned int)v54 <= 0x40 )
      goto LABEL_55;
    j___libc_free_0(*(_QWORD *)(a1 + 152));
    *(_DWORD *)(a1 + 168) = 0;
LABEL_507:
    *(_QWORD *)(a1 + 152) = 0;
    *(_QWORD *)(a1 + 160) = 0;
  }
LABEL_58:
  sub_1C51850(*(_QWORD *)(a1 + 328));
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = a1 + 320;
  *(_QWORD *)(a1 + 344) = a1 + 320;
  v57 = *(_DWORD *)(a1 + 248);
  *(_QWORD *)(a1 + 352) = 0;
  if ( v57 )
  {
    v58 = *(_QWORD **)(a1 + 240);
    v95 = &v58[*(unsigned int *)(a1 + 256)];
    v96 = v58;
    if ( v58 != v95 )
    {
      while ( *v96 == -8 || *v96 == -16 )
      {
        if ( v95 == ++v96 )
          goto LABEL_59;
      }
      if ( v95 != v96 )
      {
        do
        {
          v97 = (_QWORD *)*v96;
          if ( *v96 )
          {
            if ( *v97 )
              j_j___libc_free_0(*v97, v97[2] - *v97);
            j_j___libc_free_0(v97, 24);
          }
          if ( ++v96 == v95 )
            break;
          while ( *v96 == -8 || *v96 == -16 )
          {
            if ( v95 == ++v96 )
              goto LABEL_162;
          }
        }
        while ( v96 != v95 );
LABEL_162:
        v57 = *(_DWORD *)(a1 + 248);
        goto LABEL_59;
      }
      ++*(_QWORD *)(a1 + 232);
LABEL_61:
      v59 = 4 * v57;
      v60 = *(unsigned int *)(a1 + 256);
      if ( (unsigned int)(4 * v57) < 0x40 )
        v59 = 64;
      if ( v59 >= (unsigned int)v60 )
      {
LABEL_64:
        v61 = &v58[v60];
        for ( ii = v58; v61 != ii; ++ii )
          *ii = -8;
        *(_QWORD *)(a1 + 248) = 0;
        goto LABEL_67;
      }
      v278 = v58;
      v279 = v57 - 1;
      if ( v279 )
      {
        _BitScanReverse(&v279, v279);
        v280 = 1 << (33 - (v279 ^ 0x1F));
        if ( v280 < 64 )
          v280 = 64;
        if ( (_DWORD)v60 == v280 )
        {
          *(_QWORD *)(a1 + 248) = 0;
          do
          {
            if ( v278 )
              *v278 = -8;
            ++v278;
          }
          while ( &v58[v60] != v278 );
          goto LABEL_67;
        }
        v281 = (4 * v280 / 3u + 1) | ((unsigned __int64)(4 * v280 / 3u + 1) >> 1);
        v282 = ((v281 | (v281 >> 2)) >> 4)
             | v281
             | (v281 >> 2)
             | ((((v281 | (v281 >> 2)) >> 4) | v281 | (v281 >> 2)) >> 8);
        v283 = (v282 | (v282 >> 16)) + 1;
        v284 = 8 * ((v282 | (v282 >> 16)) + 1);
      }
      else
      {
        v284 = 1024;
        v283 = 128;
      }
      j___libc_free_0(v58);
      *(_DWORD *)(a1 + 256) = v283;
      v285 = (_QWORD *)sub_22077B0(v284);
      v286 = *(unsigned int *)(a1 + 256);
      *(_QWORD *)(a1 + 248) = 0;
      *(_QWORD *)(a1 + 240) = v285;
      for ( jj = &v285[v286]; jj != v285; ++v285 )
      {
        if ( v285 )
          *v285 = -8;
      }
      goto LABEL_67;
    }
  }
LABEL_59:
  ++*(_QWORD *)(a1 + 232);
  if ( v57 )
  {
    v58 = *(_QWORD **)(a1 + 240);
    goto LABEL_61;
  }
  if ( *(_DWORD *)(a1 + 252) )
  {
    v60 = *(unsigned int *)(a1 + 256);
    if ( (unsigned int)v60 <= 0x40 )
    {
      v58 = *(_QWORD **)(a1 + 240);
      goto LABEL_64;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 240));
    *(_QWORD *)(a1 + 240) = 0;
    *(_QWORD *)(a1 + 248) = 0;
    *(_DWORD *)(a1 + 256) = 0;
  }
LABEL_67:
  v63 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  if ( v63 )
  {
    v83 = 4 * v63;
    v64 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)(4 * v63) < 0x40 )
      v83 = 64;
    if ( v83 >= (unsigned int)v64 )
    {
LABEL_70:
      v65 = *(_QWORD **)(a1 + 40);
      for ( kk = &v65[2 * v64]; kk != v65; v65 += 2 )
        *v65 = -8;
      *(_QWORD *)(a1 + 48) = 0;
      goto LABEL_73;
    }
    v84 = v63 - 1;
    if ( v84 )
    {
      _BitScanReverse(&v84, v84);
      v85 = 1 << (33 - (v84 ^ 0x1F));
      if ( v85 < 64 )
        v85 = 64;
      if ( (_DWORD)v64 == v85 )
      {
LABEL_131:
        sub_1C55F10(a1 + 32);
        goto LABEL_73;
      }
    }
    else
    {
      v85 = 64;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    v86 = sub_1C521A0(v85);
    *(_DWORD *)(a1 + 56) = v86;
    if ( !v86 )
      goto LABEL_509;
    *(_QWORD *)(a1 + 40) = sub_22077B0(16LL * v86);
    goto LABEL_131;
  }
  if ( *(_DWORD *)(a1 + 52) )
  {
    v64 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v64 <= 0x40 )
      goto LABEL_70;
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    *(_DWORD *)(a1 + 56) = 0;
LABEL_509:
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 48) = 0;
  }
LABEL_73:
  sub_1C51A20(*(_QWORD *)(a1 + 80));
  *(_QWORD *)(a1 + 80) = 0;
  v67 = src;
  *(_QWORD *)(a1 + 88) = a1 + 72;
  *(_QWORD *)(a1 + 96) = a1 + 72;
  *(_QWORD *)(a1 + 104) = 0;
  j___libc_free_0(v67);
  return v41;
}
