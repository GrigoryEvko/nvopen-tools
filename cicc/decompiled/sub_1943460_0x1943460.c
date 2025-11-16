// Function: sub_1943460
// Address: 0x1943460
//
__int64 __fastcall sub_1943460(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // r14
  int v12; // eax
  __int64 v13; // r11
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // r8
  char v23; // di
  unsigned int v24; // esi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r8
  char v33; // di
  unsigned int v34; // esi
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rbx
  __int64 *v41; // rsi
  __int64 **v42; // r12
  __int64 *v43; // rax
  __int64 v44; // rsi
  double v45; // xmm4_8
  double v46; // xmm5_8
  unsigned int v47; // eax
  __int64 v48; // rbx
  __int64 v49; // rdx
  __int64 v50; // r12
  __int64 v51; // rcx
  __int64 v52; // r13
  char v53; // r15
  unsigned __int8 v54; // al
  __int64 v55; // r8
  __int64 v56; // rdi
  __int64 v57; // rax
  int v58; // edx
  int v59; // edx
  __int64 v60; // rsi
  unsigned int v61; // ecx
  __int64 *v62; // rax
  __int64 v63; // r11
  __int64 *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rsi
  __int64 v71; // r14
  __int64 v72; // rdx
  __int64 v73; // rdx
  int v74; // eax
  __int64 v75; // rax
  int v76; // ecx
  __int64 v77; // rcx
  __int64 *v78; // rax
  __int64 v79; // rsi
  unsigned __int64 v80; // rdi
  __int64 v81; // rdi
  __int64 v82; // rcx
  __int64 v83; // rsi
  __int64 v84; // rax
  unsigned __int64 v85; // r13
  __int64 v86; // rax
  double v87; // xmm4_8
  double v88; // xmm5_8
  __int64 v89; // rax
  __int64 *v90; // rsi
  __int64 **v91; // rdx
  __int64 v92; // r13
  unsigned int v93; // eax
  _QWORD *v94; // rdi
  __int64 *v95; // rsi
  __int64 v96; // rsi
  _QWORD *v97; // r12
  __int64 *v98; // rax
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 *v103; // rdi
  __int64 *v104; // rax
  __int64 v105; // r9
  unsigned __int64 v106; // rsi
  __int64 v107; // rsi
  __int64 result; // rax
  __int64 v109; // rax
  __int64 v110; // r15
  __int64 *v111; // rdi
  unsigned int v112; // eax
  __int64 v113; // rdx
  __int64 v114; // rbx
  __int64 v115; // rbx
  __int64 v116; // r14
  int v117; // r13d
  int v118; // r13d
  _QWORD *v119; // rax
  int v120; // r8d
  int v121; // r9d
  __int64 v122; // rsi
  __int64 v123; // rcx
  __int64 v124; // r12
  unsigned int v125; // edx
  __int64 *v126; // rax
  __int64 v127; // rdi
  _QWORD *v128; // rax
  _QWORD *v129; // rdx
  __int64 *v130; // rax
  __int64 *v131; // rsi
  char v132; // dl
  __int64 v133; // rax
  unsigned __int64 v134; // rax
  __int64 v135; // rcx
  int v136; // eax
  __int64 v137; // rdx
  _BYTE *v138; // rdi
  unsigned __int8 v139; // al
  _BYTE *v140; // rdx
  unsigned int v141; // eax
  __int64 v142; // rdx
  bool v143; // zf
  __int64 v144; // rcx
  _QWORD *v145; // r13
  _QWORD *v146; // r12
  __int64 v147; // rax
  __int64 v148; // rsi
  __int64 v149; // rdx
  __int64 v150; // rax
  __int64 v151; // r8
  int v152; // r10d
  unsigned int v153; // esi
  __int64 *v154; // rdx
  __int64 v155; // r9
  __int64 *v156; // rdi
  __int64 *v157; // rax
  __int64 *i; // r12
  unsigned __int64 v159; // rax
  unsigned __int64 v160; // r13
  __int64 v161; // rax
  _QWORD *v162; // r14
  _QWORD *v163; // r12
  __int64 v164; // rax
  __int64 v165; // rsi
  __int64 v166; // rbx
  __int64 *v167; // rcx
  unsigned int v168; // r14d
  __int64 v169; // rax
  __int64 *v170; // rsi
  __int64 **v171; // rdx
  __int64 v172; // rax
  __int64 *v173; // rsi
  __int64 v174; // r13
  unsigned int v175; // eax
  _QWORD *v176; // rdi
  __int64 *v177; // r11
  __int64 v178; // r14
  int v179; // eax
  __int64 v180; // r11
  __int64 v181; // rdi
  unsigned __int64 v182; // r15
  __int64 *v183; // r14
  __int64 v184; // rdi
  __int64 v185; // rax
  __int64 *v186; // rsi
  __int64 v187; // rax
  __int64 v188; // r9
  __int64 v189; // rax
  __int64 v190; // r9
  __int64 v191; // rdx
  _QWORD *v192; // rdi
  __int64 v193; // rax
  __int64 *v194; // rdi
  int v195; // eax
  __int64 v196; // r12
  unsigned int v197; // eax
  _QWORD *v198; // rdi
  __int64 v199; // r14
  __int64 v200; // rax
  __int64 v201; // r11
  char v202; // r15
  __int64 *v203; // rax
  __int64 *v204; // r14
  __int64 v205; // rsi
  __int64 v206; // rsi
  __int64 *v207; // rax
  __int64 v208; // rdx
  __int64 v209; // r14
  __int64 v210; // rax
  __int64 *v211; // rsi
  __int64 *v212; // r13
  __int64 v213; // rax
  __int64 v214; // rcx
  __int64 v215; // rsi
  unsigned __int8 *v216; // rsi
  __int64 v217; // rsi
  unsigned int v218; // esi
  __int64 v219; // rdi
  unsigned int v220; // edx
  __int64 *v221; // rax
  __int64 v222; // rcx
  _QWORD *v223; // rax
  int v224; // r14d
  char v225; // r13
  __int64 v226; // r9
  char v227; // al
  __int64 v228; // rsi
  __int64 v229; // rax
  __int64 *v230; // r15
  __int64 v231; // rsi
  __int64 v232; // rax
  __int64 v233; // r14
  const char *v234; // rax
  __int64 v235; // rdx
  __int64 v236; // rax
  __int64 v237; // r15
  __int64 v238; // rax
  __int64 v239; // r9
  __int64 *v240; // rsi
  __int64 *v241; // r15
  __int64 v242; // rax
  __int64 v243; // rcx
  __int64 v244; // r9
  __int64 v245; // rsi
  unsigned __int8 *v246; // rsi
  __int64 v247; // r12
  unsigned int v248; // eax
  _QWORD *v249; // rdi
  __int64 v250; // rax
  int v251; // r14d
  __int64 *v252; // rcx
  unsigned int v253; // edi
  __int64 *v254; // rdx
  int v255; // eax
  __int64 v256; // rsi
  unsigned __int8 *v257; // rsi
  __int64 *v258; // rax
  int v259; // r10d
  __int64 *v260; // r15
  int v261; // edi
  int v262; // edx
  __int64 *v263; // r13
  __int64 v264; // rax
  __int64 v265; // rcx
  __int64 v266; // rsi
  unsigned __int8 *v267; // rsi
  int v268; // r13d
  int v269; // r13d
  __int64 v270; // r9
  unsigned int v271; // ecx
  __int64 v272; // r8
  int v273; // edi
  __int64 *v274; // rsi
  char v275; // al
  __int64 v276; // rax
  __int64 v277; // rax
  int v278; // r11d
  int v279; // r11d
  int v280; // esi
  __int64 v281; // r8
  unsigned int v282; // r13d
  __int64 *v283; // rcx
  __int64 v284; // rdi
  __int64 v285; // r11
  __int64 v286; // r11
  unsigned int v287; // r13d
  int v288; // edi
  int v289; // r12d
  __int64 *v290; // r14
  __int64 v291; // rax
  __int64 v292; // rcx
  __int64 v293; // rsi
  unsigned __int8 *v294; // rsi
  int v295; // edx
  int v296; // edi
  int v297; // r9d
  __int64 v298; // rax
  __int64 v299; // [rsp+8h] [rbp-1B8h]
  __int64 v300; // [rsp+8h] [rbp-1B8h]
  __int64 v301; // [rsp+8h] [rbp-1B8h]
  bool v302; // [rsp+10h] [rbp-1B0h]
  __int64 v303; // [rsp+10h] [rbp-1B0h]
  __int64 v304; // [rsp+10h] [rbp-1B0h]
  __int64 v305; // [rsp+10h] [rbp-1B0h]
  __int64 v306; // [rsp+10h] [rbp-1B0h]
  __int64 v307; // [rsp+10h] [rbp-1B0h]
  __int64 v308; // [rsp+10h] [rbp-1B0h]
  __int64 v310; // [rsp+20h] [rbp-1A0h]
  int v311; // [rsp+20h] [rbp-1A0h]
  unsigned int v312; // [rsp+20h] [rbp-1A0h]
  __int64 v313; // [rsp+20h] [rbp-1A0h]
  __int64 *v314; // [rsp+28h] [rbp-198h]
  __int64 v315; // [rsp+28h] [rbp-198h]
  __int64 **v316; // [rsp+30h] [rbp-190h]
  __int64 v317; // [rsp+30h] [rbp-190h]
  __int64 v318; // [rsp+30h] [rbp-190h]
  int v319; // [rsp+30h] [rbp-190h]
  unsigned int v320; // [rsp+30h] [rbp-190h]
  __int64 v321; // [rsp+30h] [rbp-190h]
  __int64 v322; // [rsp+30h] [rbp-190h]
  unsigned __int64 v323; // [rsp+30h] [rbp-190h]
  __int64 v324; // [rsp+30h] [rbp-190h]
  __int64 v325; // [rsp+38h] [rbp-188h]
  __int64 v326; // [rsp+38h] [rbp-188h]
  __int64 v327; // [rsp+38h] [rbp-188h]
  __int64 v328; // [rsp+40h] [rbp-180h] BYREF
  __int64 v329; // [rsp+48h] [rbp-178h] BYREF
  __int64 v330; // [rsp+50h] [rbp-170h] BYREF
  __int64 *v331; // [rsp+58h] [rbp-168h] BYREF
  unsigned __int8 *v332; // [rsp+60h] [rbp-160h] BYREF
  __int64 v333; // [rsp+68h] [rbp-158h]
  __int64 *v334; // [rsp+70h] [rbp-150h] BYREF
  __int64 v335; // [rsp+78h] [rbp-148h]
  __int64 **v336; // [rsp+80h] [rbp-140h]
  __int64 *v337; // [rsp+88h] [rbp-138h]
  __int64 *v338; // [rsp+90h] [rbp-130h]
  __int64 *v339; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v340; // [rsp+A8h] [rbp-118h]
  _QWORD v341[6]; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v342; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v343; // [rsp+E8h] [rbp-D8h]
  __int64 *v344; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v345; // [rsp+F8h] [rbp-C8h]
  __int64 v346; // [rsp+100h] [rbp-C0h]
  __int64 v347; // [rsp+108h] [rbp-B8h] BYREF
  __int64 v348; // [rsp+110h] [rbp-B0h]
  __int64 v349; // [rsp+118h] [rbp-A8h]

  v11 = (__int64)a1;
  if ( *(_WORD *)(sub_146F1B0(a1[4], *a1) + 24) != 7 )
    return 0;
  v12 = sub_193EAF0((__int64)a1, *a1);
  v14 = a1[4];
  v15 = *(_QWORD *)(v11 + 8);
  v310 = v12 == 1 ? sub_147B0D0(v14, v13, v15, 0) : sub_14747F0(v14, v13, v15, 0);
  if ( *(_WORD *)(v310 + 24) != 7 )
    return 0;
  v16 = *(_QWORD *)(v11 + 24);
  if ( v16 != *(_QWORD *)(v310 + 48) )
    return 0;
  if ( byte_4FAF780 )
  {
    v109 = *(_QWORD *)v11;
    v110 = v11;
    v340 = 0x600000001LL;
    v111 = v341;
    v345 = 0x100000010LL;
    v343 = (__int64)&v347;
    v344 = &v347;
    v339 = v341;
    LODWORD(v346) = 0;
    v342 = 1;
    v341[0] = v109;
    v347 = v109;
    v112 = 1;
    while ( 1 )
    {
      v113 = v112--;
      v114 = v111[v113 - 1];
      LODWORD(v340) = v112;
      v318 = v114;
      v115 = *(_QWORD *)(v114 + 8);
      if ( !v115 )
        goto LABEL_164;
      do
      {
        v116 = *(_QWORD *)(v110 + 16);
        v117 = *(_DWORD *)(v116 + 24);
        if ( !v117 )
          goto LABEL_162;
        v118 = v117 - 1;
        v119 = sub_1648700(v115);
        v122 = *(_QWORD *)(v116 + 8);
        v123 = v119[5];
        v124 = (__int64)v119;
        v125 = v118 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
        v126 = (__int64 *)(v122 + 16LL * v125);
        v127 = *v126;
        if ( v123 != *v126 )
        {
          v195 = 1;
          while ( v127 != -8 )
          {
            v120 = v195 + 1;
            v125 = v118 & (v195 + v125);
            v126 = (__int64 *)(v122 + 16LL * v125);
            v127 = *v126;
            if ( v123 == *v126 )
              goto LABEL_98;
            v195 = v120;
          }
          goto LABEL_162;
        }
LABEL_98:
        v128 = (_QWORD *)v126[1];
        if ( !v128 )
          goto LABEL_162;
        v129 = *(_QWORD **)(v110 + 24);
        if ( v129 != v128 )
        {
          while ( 1 )
          {
            v128 = (_QWORD *)*v128;
            if ( v129 == v128 )
              break;
            if ( !v128 )
              goto LABEL_162;
          }
        }
        v130 = (__int64 *)v343;
        if ( v344 != (__int64 *)v343 )
          goto LABEL_104;
        v131 = (__int64 *)(v343 + 8LL * HIDWORD(v345));
        if ( (__int64 *)v343 != v131 )
        {
          v167 = 0;
          do
          {
            if ( v124 == *v130 )
              goto LABEL_162;
            if ( *v130 == -2 )
              v167 = v130;
            ++v130;
          }
          while ( v131 != v130 );
          if ( v167 )
          {
            *v167 = v124;
            LODWORD(v346) = v346 - 1;
            ++v342;
            goto LABEL_105;
          }
        }
        if ( HIDWORD(v345) < (unsigned int)v345 )
        {
          ++HIDWORD(v345);
          *v131 = v124;
          ++v342;
        }
        else
        {
LABEL_104:
          v131 = (__int64 *)v124;
          sub_16CCBA0((__int64)&v342, v124);
          if ( !v132 )
            goto LABEL_162;
        }
LABEL_105:
        v133 = (unsigned int)v340;
        if ( (unsigned int)v340 >= HIDWORD(v340) )
        {
          v131 = v341;
          sub_16CD150((__int64)&v339, v341, 0, 8, v120, v121);
          v133 = (unsigned int)v340;
        }
        v339[v133] = v124;
        LODWORD(v340) = v340 + 1;
        v328 = v318;
        v329 = v124;
        v134 = *(unsigned __int8 *)(v318 + 16);
        if ( (unsigned __int8)v134 <= 0x2Fu )
        {
          v135 = 0x80A800000000LL;
          if ( _bittest64(&v135, v134) )
          {
            v136 = (unsigned __int8)v134 <= 0x17u ? *(unsigned __int16 *)(v318 + 18) : (unsigned __int8)v134 - 24;
            if ( v136 == 11 && (*(_BYTE *)(v318 + 17) & 4) != 0 )
            {
              if ( (*(_BYTE *)(v318 + 23) & 0x40) != 0 )
              {
                if ( !**(_QWORD **)(v318 - 8) )
                  goto LABEL_162;
                v330 = **(_QWORD **)(v318 - 8);
                v137 = *(_QWORD *)(v318 - 8);
              }
              else
              {
                v135 = v318 - 24LL * (*(_DWORD *)(v318 + 20) & 0xFFFFFFF);
                v137 = v135;
                if ( !*(_QWORD *)v135 )
                  goto LABEL_162;
                v330 = *(_QWORD *)v135;
              }
              v138 = *(_BYTE **)(v137 + 24);
              v139 = v138[16];
              if ( v139 == 13 )
              {
                v140 = v138 + 24;
                v331 = (__int64 *)(v138 + 24);
              }
              else
              {
                if ( *(_BYTE *)(*(_QWORD *)v138 + 8LL) != 16 )
                  goto LABEL_162;
                if ( v139 > 0x10u )
                  goto LABEL_162;
                v277 = sub_15A1020(v138, (__int64)v131, *(_QWORD *)v138, v135);
                if ( !v277 || *(_BYTE *)(v277 + 16) != 13 )
                  goto LABEL_162;
                v140 = (_BYTE *)(v277 + 24);
                v331 = (__int64 *)(v277 + 24);
              }
              v141 = *((_DWORD *)v140 + 2);
              v142 = *(_QWORD *)v140;
              if ( v141 > 0x40 )
                v142 = *(_QWORD *)(v142 + 8LL * ((v141 - 1) >> 6));
              if ( (v142 & (1LL << ((unsigned __int8)v141 - 1))) == 0 )
              {
                v143 = *(_BYTE *)(v110 + 48) == 0;
                v335 = v110;
                v334 = &v330;
                v336 = &v331;
                v337 = &v328;
                v338 = &v329;
                v144 = *(_QWORD *)(v329 + 40);
                if ( !v143 )
                {
                  v145 = (_QWORD *)(v144 + 40);
                  v146 = (_QWORD *)(v329 + 24);
                  if ( v144 + 40 != v329 + 24 )
                  {
                    while ( 1 )
                    {
                      if ( *((_BYTE *)v146 - 8) == 78 )
                      {
                        v147 = *(v146 - 6);
                        if ( !*(_BYTE *)(v147 + 16) && *(_DWORD *)(v147 + 36) == 79 )
                        {
                          v148 = *(_QWORD *)(((unsigned __int64)(v146 - 3) & 0xFFFFFFFFFFFFFFF8LL)
                                           - 24LL
                                           * (*(_DWORD *)(((unsigned __int64)(v146 - 3) & 0xFFFFFFFFFFFFFFF8LL) + 20)
                                            & 0xFFFFFFF));
                          if ( v148 )
                            sub_1942B80((__int64)&v334, v148, 1);
                        }
                      }
                      v146 = (_QWORD *)(*v146 & 0xFFFFFFFFFFFFFFF8LL);
                      if ( v145 == v146 )
                        break;
                      if ( !v146 )
                        BUG();
                    }
                    v144 = *(_QWORD *)(v329 + 40);
                  }
                }
                v149 = *(_QWORD *)(v110 + 40);
                v150 = *(unsigned int *)(v149 + 48);
                if ( (_DWORD)v150 )
                {
                  v151 = *(_QWORD *)(v149 + 32);
                  v152 = v150 - 1;
                  v153 = (v150 - 1) & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
                  v154 = (__int64 *)(v151 + 16LL * v153);
                  v155 = *v154;
                  v156 = v154;
                  if ( v144 == *v154 )
                  {
LABEL_134:
                    v157 = (__int64 *)(v151 + 16 * v150);
                    if ( v157 != v156 && v156[1] )
                    {
                      if ( v144 != v155 )
                      {
                        v295 = 1;
                        while ( v155 != -8 )
                        {
                          v296 = v295 + 1;
                          v153 = v152 & (v295 + v153);
                          v154 = (__int64 *)(v151 + 16LL * v153);
                          v155 = *v154;
                          if ( v144 == *v154 )
                            goto LABEL_137;
                          v295 = v296;
                        }
LABEL_444:
                        BUG();
                      }
LABEL_137:
                      if ( v157 == v154 )
                        goto LABEL_444;
                      v299 = v115;
                      for ( i = *(__int64 **)(v154[1] + 8);
                            sub_1377F70(*(_QWORD *)(v110 + 24) + 56LL, *i);
                            i = (__int64 *)i[1] )
                      {
                        v327 = *i;
                        v159 = sub_157EBA0(*i);
                        v160 = v159;
                        if ( *(_BYTE *)(v110 + 48) )
                        {
                          v161 = *(_QWORD *)(v159 + 40);
                          v162 = (_QWORD *)(v160 + 24);
                          if ( v161 + 40 != v160 + 24 )
                          {
                            v314 = i;
                            v163 = (_QWORD *)(v161 + 40);
                            while ( 1 )
                            {
                              if ( *((_BYTE *)v162 - 8) == 78 )
                              {
                                v164 = *(v162 - 6);
                                if ( !*(_BYTE *)(v164 + 16) && *(_DWORD *)(v164 + 36) == 79 )
                                {
                                  v165 = *(_QWORD *)(((unsigned __int64)(v162 - 3) & 0xFFFFFFFFFFFFFFF8LL)
                                                   - 24LL
                                                   * (*(_DWORD *)(((unsigned __int64)(v162 - 3) & 0xFFFFFFFFFFFFFFF8LL)
                                                                + 20)
                                                    & 0xFFFFFFF));
                                  if ( v165 )
                                    sub_1942B80((__int64)&v334, v165, 1);
                                }
                              }
                              v162 = (_QWORD *)(*v162 & 0xFFFFFFFFFFFFFFF8LL);
                              if ( v163 == v162 )
                                break;
                              if ( !v162 )
                                BUG();
                            }
                            i = v314;
                          }
                        }
                        if ( *(_BYTE *)(v160 + 16) == 26 && (*(_DWORD *)(v160 + 20) & 0xFFFFFFF) == 3 )
                        {
                          v166 = *(_QWORD *)(v160 - 48);
                          v315 = v329;
                          v333 = *(_QWORD *)(v160 - 24);
                          v332 = (unsigned __int8 *)v327;
                          if ( (unsigned __int8)sub_15CC350((__int64 *)&v332)
                            && (unsigned __int8)sub_15CCD40(
                                                  *(_QWORD *)(v110 + 40),
                                                  (__int64 *)&v332,
                                                  *(_QWORD *)(v315 + 40)) )
                          {
                            sub_1942B80((__int64)&v334, *(_QWORD *)(v160 - 72), 1);
                          }
                          v333 = v166;
                          v332 = (unsigned __int8 *)v327;
                          if ( (unsigned __int8)sub_15CC350((__int64 *)&v332)
                            && (unsigned __int8)sub_15CCD40(
                                                  *(_QWORD *)(v110 + 40),
                                                  (__int64 *)&v332,
                                                  *(_QWORD *)(v315 + 40)) )
                          {
                            sub_1942B80((__int64)&v334, *(_QWORD *)(v160 - 72), 0);
                          }
                        }
                      }
                      v115 = v299;
                    }
                  }
                  else
                  {
                    v286 = *v154;
                    v287 = (v150 - 1) & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
                    v288 = 1;
                    while ( v286 != -8 )
                    {
                      v289 = v288 + 1;
                      v287 = v152 & (v288 + v287);
                      v156 = (__int64 *)(v151 + 16LL * v287);
                      v286 = *v156;
                      if ( v144 == *v156 )
                        goto LABEL_134;
                      v288 = v289;
                    }
                  }
                }
              }
            }
          }
        }
LABEL_162:
        v115 = *(_QWORD *)(v115 + 8);
      }
      while ( v115 );
      v112 = v340;
      v111 = v339;
LABEL_164:
      if ( !v112 )
      {
        v11 = v110;
        if ( v111 != v341 )
          _libc_free((unsigned __int64)v111);
        if ( v344 != (__int64 *)v343 )
          _libc_free((unsigned __int64)v344);
        v16 = *(_QWORD *)(v110 + 24);
        break;
      }
    }
  }
  v17 = *(_QWORD *)(**(_QWORD **)(v16 + 32) + 48LL);
  if ( v17 )
    v17 -= 24;
  v18 = sub_38767A0(a2, v310, *(_QWORD *)(v11 + 8), v17);
  v19 = *(_QWORD *)(v11 + 24);
  *(_QWORD *)(v11 + 56) = v18;
  v20 = sub_13FCB50(v19);
  if ( !v20 )
    goto LABEL_32;
  v21 = 0x17FFFFFFE8LL;
  v22 = *(_QWORD *)(v11 + 56);
  v23 = *(_BYTE *)(v22 + 23) & 0x40;
  v24 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
  if ( v24 )
  {
    v25 = 24LL * *(unsigned int *)(v22 + 56) + 8;
    v26 = 0;
    do
    {
      v27 = v22 - 24LL * v24;
      if ( v23 )
        v27 = *(_QWORD *)(v22 - 8);
      if ( v20 == *(_QWORD *)(v27 + v25) )
      {
        v21 = 24 * v26;
        goto LABEL_17;
      }
      ++v26;
      v25 += 8;
    }
    while ( v24 != (_DWORD)v26 );
    v21 = 0x17FFFFFFE8LL;
  }
LABEL_17:
  if ( v23 )
    v28 = *(_QWORD *)(v22 - 8);
  else
    v28 = v22 - 24LL * v24;
  v29 = *(_QWORD *)(v28 + v21);
  v30 = *(_QWORD *)(v11 + 32);
  *(_QWORD *)(v11 + 64) = v29;
  v31 = sub_146F1B0(v30, v29);
  v32 = *(_QWORD *)v11;
  *(_QWORD *)(v11 + 72) = v31;
  v33 = *(_BYTE *)(v32 + 23) & 0x40;
  v34 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
  if ( v34 )
  {
    v35 = 24LL * *(unsigned int *)(v32 + 56) + 8;
    v36 = 0;
    while ( 1 )
    {
      v37 = v32 - 24LL * v34;
      if ( v33 )
        v37 = *(_QWORD *)(v32 - 8);
      if ( v20 == *(_QWORD *)(v37 + v35) )
        break;
      ++v36;
      v35 += 8;
      if ( v34 == (_DWORD)v36 )
        goto LABEL_343;
    }
    v38 = 24 * v36;
  }
  else
  {
LABEL_343:
    v38 = 0x17FFFFFFE8LL;
  }
  if ( v33 )
    v39 = *(_QWORD *)(v32 - 8);
  else
    v39 = v32 - 24LL * v34;
  v40 = *(_QWORD *)(v11 + 64);
  v41 = *(__int64 **)(*(_QWORD *)(v39 + v38) + 48LL);
  v42 = (__int64 **)(v40 + 48);
  v342 = (__int64)v41;
  if ( !v41 )
  {
    if ( v42 == (__int64 **)&v342 )
      goto LABEL_33;
    v256 = *(_QWORD *)(v40 + 48);
    if ( !v256 )
      goto LABEL_33;
LABEL_347:
    sub_161E7C0(v40 + 48, v256);
    goto LABEL_348;
  }
  sub_1623A60((__int64)&v342, (__int64)v41, 2);
  if ( v42 == (__int64 **)&v342 )
  {
    if ( v342 )
      sub_161E7C0(v40 + 48, v342);
    goto LABEL_32;
  }
  v256 = *(_QWORD *)(v40 + 48);
  if ( v256 )
    goto LABEL_347;
LABEL_348:
  v257 = (unsigned __int8 *)v342;
  *(_QWORD *)(v40 + 48) = v342;
  if ( v257 )
  {
    sub_1623210((__int64)&v342, v257, v40 + 48);
    v32 = *(_QWORD *)v11;
    goto LABEL_33;
  }
LABEL_32:
  v32 = *(_QWORD *)v11;
LABEL_33:
  v43 = *(__int64 **)(v11 + 96);
  if ( *(__int64 **)(v11 + 104) != v43 )
    goto LABEL_34;
  v252 = &v43[*(unsigned int *)(v11 + 116)];
  v253 = *(_DWORD *)(v11 + 116);
  if ( v43 != v252 )
  {
    v254 = 0;
    while ( 1 )
    {
      v44 = *v43;
      if ( v32 == *v43 )
        goto LABEL_35;
      if ( v44 == -2 )
        v254 = v43;
      if ( v252 == ++v43 )
      {
        if ( !v254 )
          break;
        *v254 = v32;
        v44 = v32;
        --*(_DWORD *)(v11 + 120);
        ++*(_QWORD *)(v11 + 88);
        goto LABEL_35;
      }
    }
  }
  if ( v253 < *(_DWORD *)(v11 + 112) )
  {
    *(_DWORD *)(v11 + 116) = v253 + 1;
    *v252 = v32;
    v44 = *(_QWORD *)v11;
    ++*(_QWORD *)(v11 + 88);
  }
  else
  {
LABEL_34:
    sub_16CCBA0(v11 + 88, v32);
    v44 = *(_QWORD *)v11;
  }
LABEL_35:
  sub_193F7B0(v11, v44, *(_QWORD *)(v11 + 56));
  v47 = *(_DWORD *)(v11 + 264);
  v48 = v11;
  if ( !v47 )
    goto LABEL_79;
  do
  {
    v49 = *(_QWORD *)(v48 + 256) + 32LL * v47 - 32;
    v50 = *(_QWORD *)(v49 + 8);
    v51 = *(_QWORD *)v49;
    v52 = *(_QWORD *)(v49 + 16);
    v53 = *(_BYTE *)(v49 + 24);
    *(_DWORD *)(v48 + 264) = v47 - 1;
    v54 = *(_BYTE *)(v50 + 16);
    v325 = v51;
    switch ( v54 )
    {
      case 'M':
        v55 = *(_QWORD *)(v48 + 16);
        v56 = *(_QWORD *)(v50 + 40);
        v57 = 0;
        v58 = *(_DWORD *)(v55 + 24);
        if ( v58 )
        {
          v59 = v58 - 1;
          v60 = *(_QWORD *)(v55 + 8);
          v61 = v59 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v62 = (__int64 *)(v60 + 16LL * v61);
          v63 = *v62;
          if ( v56 == *v62 )
          {
LABEL_39:
            v57 = v62[1];
          }
          else
          {
            v255 = 1;
            while ( v63 != -8 )
            {
              v297 = v255 + 1;
              v298 = v59 & (v61 + v255);
              v61 = v298;
              v62 = (__int64 *)(v60 + 16 * v298);
              v63 = *v62;
              if ( v56 == *v62 )
                goto LABEL_39;
              v255 = v297;
            }
            v57 = 0;
          }
        }
        if ( *(_QWORD *)(v48 + 24) != v57 )
        {
          if ( (*(_DWORD *)(v50 + 20) & 0xFFFFFFF) == 1 )
          {
            if ( *(_BYTE *)(sub_157EBA0(v56) + 16) != 34 )
            {
              v64 = (__int64 *)sub_1649960(v50);
              LOWORD(v344) = 773;
              v339 = v64;
              v340 = v65;
              v342 = (__int64)&v339;
              v343 = (__int64)".wide";
              v316 = *(__int64 ***)v52;
              v66 = sub_1648B60(64);
              v70 = (__int64)v316;
              v71 = v66;
              if ( v66 )
              {
                v317 = v66;
                sub_15F1EA0(v66, v70, 53, 0, 0, v50);
                *(_DWORD *)(v71 + 56) = 1;
                sub_164B780(v71, &v342);
                v70 = *(unsigned int *)(v71 + 56);
                sub_1648880(v71, v70, 1);
              }
              else
              {
                v317 = 0;
              }
              if ( (*(_BYTE *)(v50 + 23) & 0x40) != 0 )
                v72 = *(_QWORD *)(v50 - 8);
              else
                v72 = v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF);
              v73 = *(_QWORD *)(v72 + 24LL * *(unsigned int *)(v50 + 56) + 8);
              v74 = *(_DWORD *)(v71 + 20) & 0xFFFFFFF;
              if ( v74 == *(_DWORD *)(v71 + 56) )
              {
                v313 = v73;
                sub_15F55D0(v71, v70, v73, v67, v68, v69);
                v73 = v313;
                v74 = *(_DWORD *)(v71 + 20) & 0xFFFFFFF;
              }
              v75 = (v74 + 1) & 0xFFFFFFF;
              v76 = v75 | *(_DWORD *)(v71 + 20) & 0xF0000000;
              *(_DWORD *)(v71 + 20) = v76;
              if ( (v76 & 0x40000000) != 0 )
                v77 = *(_QWORD *)(v71 - 8);
              else
                v77 = v317 - 24 * v75;
              v78 = (__int64 *)(v77 + 24LL * (unsigned int)(v75 - 1));
              if ( *v78 )
              {
                v79 = v78[1];
                v80 = v78[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v80 = v79;
                if ( v79 )
                  *(_QWORD *)(v79 + 16) = v80 | *(_QWORD *)(v79 + 16) & 3LL;
              }
              *v78 = v52;
              v81 = *(_QWORD *)(v52 + 8);
              v78[1] = v81;
              if ( v81 )
                *(_QWORD *)(v81 + 16) = (unsigned __int64)(v78 + 1) | *(_QWORD *)(v81 + 16) & 3LL;
              v78[2] = (v52 + 8) | v78[2] & 3;
              *(_QWORD *)(v52 + 8) = v78;
              v82 = *(_DWORD *)(v71 + 20) & 0xFFFFFFF;
              if ( (*(_BYTE *)(v71 + 23) & 0x40) != 0 )
                v83 = *(_QWORD *)(v71 - 8);
              else
                v83 = v317 - 24 * v82;
              *(_QWORD *)(v83 + 8LL * (unsigned int)(v82 - 1) + 24LL * *(unsigned int *)(v71 + 56) + 8) = v73;
              v84 = sub_157EE30(*(_QWORD *)(v71 + 40));
              v85 = v84;
              if ( !v84 )
              {
                v10 = sub_16498A0(0);
                v342 = 0;
                v344 = 0;
                v345 = v10;
                v346 = 0;
                LODWORD(v347) = 0;
                v348 = 0;
                v349 = 0;
                v343 = 0;
                BUG();
              }
              v86 = sub_16498A0(v84 - 24);
              v342 = 0;
              v345 = v86;
              v346 = 0;
              LODWORD(v347) = 0;
              v348 = 0;
              v349 = 0;
              v89 = *(_QWORD *)(v85 + 16);
              v344 = (__int64 *)v85;
              v343 = v89;
              v90 = *(__int64 **)(v85 + 24);
              v339 = v90;
              if ( v90 )
              {
                sub_1623A60((__int64)&v339, (__int64)v90, 2);
                if ( v342 )
                  sub_161E7C0((__int64)&v342, v342);
                v342 = (__int64)v339;
                if ( v339 )
                  sub_1623210((__int64)&v339, (unsigned __int8 *)v339, (__int64)&v342);
              }
              LOWORD(v336) = 257;
              v91 = *(__int64 ***)v325;
              if ( *(_QWORD *)v325 != *(_QWORD *)v71 )
              {
                if ( *(_BYTE *)(v71 + 16) > 0x10u )
                {
                  LOWORD(v341[0]) = 257;
                  v71 = sub_15FDBD0(36, v71, (__int64)v91, (__int64)&v339, 0);
                  if ( v343 )
                  {
                    v263 = v344;
                    sub_157E9D0(v343 + 40, v71);
                    v264 = *(_QWORD *)(v71 + 24);
                    v265 = *v263;
                    *(_QWORD *)(v71 + 32) = v263;
                    v265 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v71 + 24) = v265 | v264 & 7;
                    *(_QWORD *)(v265 + 8) = v71 + 24;
                    *v263 = *v263 & 7 | (v71 + 24);
                  }
                  sub_164B780(v71, (__int64 *)&v334);
                  if ( v342 )
                  {
                    v332 = (unsigned __int8 *)v342;
                    sub_1623A60((__int64)&v332, v342, 2);
                    v266 = *(_QWORD *)(v71 + 48);
                    if ( v266 )
                      sub_161E7C0(v71 + 48, v266);
                    v267 = v332;
                    *(_QWORD *)(v71 + 48) = v332;
                    if ( v267 )
                      sub_1623210((__int64)&v332, v267, v71 + 48);
                  }
                }
                else
                {
                  v71 = sub_15A46C0(36, (__int64 ***)v71, v91, 0);
                }
              }
              sub_164D160(v50, v71, a3, *(double *)a4.m128i_i64, a5, a6, v87, v88, a9, a10);
              v92 = *(_QWORD *)(v48 + 80);
              v93 = *(_DWORD *)(v92 + 8);
              if ( v93 >= *(_DWORD *)(v92 + 12) )
              {
                sub_170B450(*(_QWORD *)(v48 + 80), 0);
                v93 = *(_DWORD *)(v92 + 8);
              }
              v94 = (_QWORD *)(*(_QWORD *)v92 + 24LL * v93);
              if ( v94 )
              {
                *v94 = 6;
                v94[1] = 0;
                v94[2] = v50;
                if ( v50 != -8 && v50 != -16 )
                  sub_164C220((__int64)v94);
                v93 = *(_DWORD *)(v92 + 8);
              }
              *(_DWORD *)(v92 + 8) = v93 + 1;
              v95 = (__int64 *)v342;
              if ( v342 )
                goto LABEL_75;
            }
          }
          else
          {
            sub_193EF70((__int64 ***)v325, v50, v52, *(_QWORD *)(v48 + 40), v55);
          }
          goto LABEL_76;
        }
        goto LABEL_221;
      case '>':
        if ( !v53 && (unsigned int)sub_193EAF0(v48, v51) != 1 )
        {
LABEL_221:
          v181 = *(_QWORD *)(v48 + 32);
          goto LABEL_197;
        }
LABEL_173:
        if ( *(_QWORD *)v50 == *(_QWORD *)(v48 + 8) )
        {
LABEL_185:
          if ( v50 != v52 )
          {
            sub_164D160(v50, v52, a3, *(double *)a4.m128i_i64, a5, a6, v45, v46, a9, a10);
            v174 = *(_QWORD *)(v48 + 80);
            v175 = *(_DWORD *)(v174 + 8);
            if ( v175 >= *(_DWORD *)(v174 + 12) )
            {
              sub_170B450(*(_QWORD *)(v48 + 80), 0);
              v175 = *(_DWORD *)(v174 + 8);
            }
            v176 = (_QWORD *)(*(_QWORD *)v174 + 24LL * v175);
            if ( v176 )
            {
              *v176 = 6;
              v176[1] = 0;
              v176[2] = v50;
              if ( v50 != -16 && v50 != -8 )
                sub_164C220((__int64)v176);
              v175 = *(_DWORD *)(v174 + 8);
            }
            *(_DWORD *)(v174 + 8) = v175 + 1;
          }
          goto LABEL_76;
        }
        v168 = sub_1456C90(*(_QWORD *)(v48 + 32), *(_QWORD *)v50);
        if ( v168 >= (unsigned int)sub_1456C90(*(_QWORD *)(v48 + 32), *(_QWORD *)(v48 + 8)) )
        {
          sub_1648780(v50, v325, v52);
          goto LABEL_76;
        }
        v169 = sub_16498A0(v50);
        v342 = 0;
        v345 = v169;
        v346 = 0;
        LODWORD(v347) = 0;
        v348 = 0;
        v349 = 0;
        v343 = *(_QWORD *)(v50 + 40);
        v344 = (__int64 *)(v50 + 24);
        v170 = *(__int64 **)(v50 + 48);
        v339 = v170;
        if ( v170 )
        {
          sub_1623A60((__int64)&v339, (__int64)v170, 2);
          if ( v342 )
            sub_161E7C0((__int64)&v342, v342);
          v342 = (__int64)v339;
          if ( v339 )
            sub_1623210((__int64)&v339, (unsigned __int8 *)v339, (__int64)&v342);
        }
        LOWORD(v336) = 257;
        v171 = *(__int64 ***)v50;
        if ( *(_QWORD *)v50 != *(_QWORD *)v52 )
        {
          if ( *(_BYTE *)(v52 + 16) <= 0x10u )
          {
            v172 = sub_15A46C0(36, (__int64 ***)v52, v171, 0);
            v173 = (__int64 *)v342;
            v52 = v172;
            goto LABEL_183;
          }
          LOWORD(v341[0]) = 257;
          v52 = sub_15FDBD0(36, v52, (__int64)v171, (__int64)&v339, 0);
          if ( v343 )
          {
            v290 = v344;
            sub_157E9D0(v343 + 40, v52);
            v291 = *(_QWORD *)(v52 + 24);
            v292 = *v290;
            *(_QWORD *)(v52 + 32) = v290;
            v292 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v52 + 24) = v292 | v291 & 7;
            *(_QWORD *)(v292 + 8) = v52 + 24;
            *v290 = *v290 & 7 | (v52 + 24);
          }
          sub_164B780(v52, (__int64 *)&v334);
          if ( !v342 )
            goto LABEL_185;
          v332 = (unsigned __int8 *)v342;
          sub_1623A60((__int64)&v332, v342, 2);
          v293 = *(_QWORD *)(v52 + 48);
          if ( v293 )
            sub_161E7C0(v52 + 48, v293);
          v294 = v332;
          *(_QWORD *)(v52 + 48) = v332;
          if ( v294 )
            sub_1623210((__int64)&v332, v294, v52 + 48);
        }
        v173 = (__int64 *)v342;
LABEL_183:
        if ( v173 )
          sub_161E7C0((__int64)&v342, (__int64)v173);
        goto LABEL_185;
      case '=':
        if ( v53 || !(unsigned int)sub_193EAF0(v48, v51) )
          goto LABEL_173;
        goto LABEL_221;
    }
    v319 = v54 - 24;
    if ( (v54 & 0xFB) != 0x23 && v54 != 37 )
      goto LABEL_221;
    if ( (*(_BYTE *)(v50 + 23) & 0x40) != 0 )
      v177 = *(__int64 **)(v50 - 8);
    else
      v177 = (__int64 *)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF));
    v178 = *v177;
    v302 = *v177 == v51;
    v179 = sub_193EAF0(v48, v51);
    v181 = *(_QWORD *)(v48 + 32);
    v311 = v179;
    if ( v179 == 1 )
    {
      if ( (*(_BYTE *)(v50 + 17) & 4) == 0 )
        goto LABEL_197;
      v300 = *(_QWORD *)(v48 + 8);
      v187 = sub_146F1B0(v181, *(_QWORD *)(v180 + 24LL * v302));
      v188 = sub_147B0D0(v181, v187, v300, 0);
LABEL_213:
      v303 = v188;
      v189 = sub_146F1B0(*(_QWORD *)(v48 + 32), v52);
      v190 = v303;
      v191 = v189;
      if ( v178 == v325 )
      {
        v190 = v189;
        v191 = v303;
      }
      v192 = *(_QWORD **)(v48 + 32);
      if ( v319 == 11 )
      {
        v345 = v191;
        v344 = (__int64 *)v190;
        v342 = (__int64)&v344;
        v343 = 0x200000002LL;
        v258 = sub_147DD40((__int64)v192, &v342, 0, 0, (__m128i)a3, a4);
        v194 = (__int64 *)v342;
        v321 = (__int64)v258;
        if ( (__int64 **)v342 == &v344 )
          goto LABEL_219;
      }
      else
      {
        if ( v319 == 13 )
        {
          v321 = sub_14806B0((__int64)v192, v190, v191, 0, 0);
          goto LABEL_219;
        }
        v345 = v191;
        v344 = (__int64 *)v190;
        v342 = (__int64)&v344;
        v343 = 0x200000002LL;
        v193 = sub_147EE30(v192, (__int64 **)&v342, 0, 0, (__m128i)a3, a4);
        v194 = (__int64 *)v342;
        v321 = v193;
        if ( (__int64 **)v342 == &v344 )
        {
LABEL_219:
          if ( *(_WORD *)(v321 + 24) == 7 && *(_QWORD *)(v321 + 48) == *(_QWORD *)(v48 + 24) )
            goto LABEL_239;
          goto LABEL_221;
        }
      }
      _libc_free((unsigned __int64)v194);
      goto LABEL_219;
    }
    if ( !v179 && (*(_BYTE *)(v50 + 17) & 2) != 0 )
    {
      v301 = *(_QWORD *)(v48 + 8);
      v250 = sub_146F1B0(v181, *(_QWORD *)(v180 + 24LL * v302));
      v188 = sub_14747F0(v181, v250, v301, 0);
      goto LABEL_213;
    }
LABEL_197:
    if ( !sub_1456C80(v181, *(_QWORD *)v50)
      || (v199 = sub_146F1B0(*(_QWORD *)(v48 + 32), v50),
          v322 = *(_QWORD *)(v48 + 32),
          v200 = sub_1456040(v199),
          v323 = sub_1456C90(v322, v200),
          v323 >= sub_1456C90(*(_QWORD *)(v48 + 32), *(_QWORD *)(v48 + 8))) )
    {
LABEL_198:
      if ( *(_BYTE *)(v50 + 16) == 75
        && (v53 || (v251 = sub_193EAF0(v48, v325), (v251 == 1) == sub_15FF7F0(*(_WORD *)(v50 + 18) & 0x7FFF))) )
      {
        v182 = v50;
        v183 = *(__int64 **)(v50 + 24LL * (*(_QWORD *)(v50 - 48) == v325) - 48);
        v320 = sub_1456C90(*(_QWORD *)(v48 + 32), *v183);
        v184 = v50;
        v312 = sub_1456C90(*(_QWORD *)(v48 + 32), *(_QWORD *)(v48 + 8));
        if ( *(_BYTE *)(v50 + 16) == 77 )
        {
          v182 = sub_193EB70(v50, v325, *(_QWORD *)(v48 + 40), *(_QWORD *)(v48 + 16));
          v184 = v182;
        }
        v185 = sub_16498A0(v184);
        v342 = 0;
        v345 = v185;
        v346 = 0;
        LODWORD(v347) = 0;
        v348 = 0;
        v349 = 0;
        v343 = *(_QWORD *)(v182 + 40);
        v344 = (__int64 *)(v182 + 24);
        v186 = *(__int64 **)(v182 + 48);
        v339 = v186;
        if ( v186 )
        {
          sub_1623A60((__int64)&v339, (__int64)v186, 2);
          if ( v342 )
            sub_161E7C0((__int64)&v342, v342);
          v342 = (__int64)v339;
          if ( v339 )
            sub_1623210((__int64)&v339, (unsigned __int8 *)v339, (__int64)&v342);
        }
        sub_1648780(v50, v325, v52);
        if ( v320 < v312 )
        {
          v275 = sub_15FF7F0(*(_WORD *)(v50 + 18) & 0x7FFF);
          v276 = sub_193FBC0(v48, (__int64)v183, *(__int64 ***)(v48 + 8), v275, v50);
          sub_1648780(v50, (__int64)v183, v276);
        }
        v95 = (__int64 *)v342;
        if ( v342 )
        {
LABEL_75:
          sub_161E7C0((__int64)&v342, (__int64)v95);
          goto LABEL_76;
        }
      }
      else
      {
        sub_193EF70((__int64 ***)v325, v50, v52, *(_QWORD *)(v48 + 40), *(_QWORD *)(v48 + 16));
      }
      goto LABEL_76;
    }
    if ( v53 )
    {
      v311 = 1;
      v321 = sub_147B0D0(*(_QWORD *)(v48 + 32), v199, *(_QWORD *)(v48 + 8), 0);
      if ( *(_WORD *)(v321 + 24) == 7 )
        goto LABEL_238;
      v311 = 0;
      v321 = sub_14747F0(*(_QWORD *)(v48 + 32), v199, *(_QWORD *)(v48 + 8), 0);
    }
    else
    {
      v324 = *(_QWORD *)(v48 + 8);
      v311 = sub_193EAF0(v48, v325);
      if ( v311 == 1 )
      {
        v321 = sub_147B0D0(v285, v199, v324, 0);
      }
      else
      {
        v311 = 0;
        v321 = sub_14747F0(v285, v199, v324, 0);
      }
    }
    if ( *(_WORD *)(v321 + 24) != 7 )
      goto LABEL_198;
LABEL_238:
    if ( *(_QWORD *)(v321 + 48) != *(_QWORD *)(v48 + 24) )
      goto LABEL_198;
LABEL_239:
    if ( *(_QWORD *)(v48 + 72) != v321 || !(unsigned __int8)sub_3871F10(a2, *(_QWORD *)(v48 + 64), v50) )
    {
      switch ( *(_BYTE *)(v50 + 16) )
      {
        case '#':
        case '%':
        case '\'':
        case ')':
          v330 = v50;
          v331 = (__int64 *)v52;
          v329 = v321;
          if ( (*(_BYTE *)(v50 + 23) & 0x40) != 0 )
            v223 = *(_QWORD **)(v50 - 8);
          else
            v223 = (_QWORD *)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF));
          v143 = *v223 == v325;
          v339 = (__int64 *)v48;
          LODWORD(v328) = !v143;
          v340 = (__int64)&v328;
          v341[0] = &v331;
          v341[1] = &v330;
          v341[2] = &v329;
          v224 = sub_193EAF0(v48, v325);
          v225 = v224 == 1;
          if ( !sub_193F3C0(&v339, v224 == 1, (__m128i)a3, a4) )
          {
            v225 = v224 != 1;
            if ( !sub_193F3C0(&v339, v224 != 1, (__m128i)a3, a4) )
              goto LABEL_76;
          }
          v226 = v330;
          v227 = *(_BYTE *)(v330 + 23) & 0x40;
          if ( v227 )
          {
            v228 = **(_QWORD **)(v330 - 8);
            if ( v228 != v325 )
              goto LABEL_273;
          }
          else
          {
            v228 = *(_QWORD *)(v330 - 24LL * (*(_DWORD *)(v330 + 20) & 0xFFFFFFF));
            if ( v228 != v325 )
            {
LABEL_273:
              v229 = sub_193FBC0(v48, v228, *(__int64 ***)(v48 + 8), v225, v330);
              v226 = v330;
              v230 = (__int64 *)v229;
              v227 = *(_BYTE *)(v330 + 23) & 0x40;
              goto LABEL_274;
            }
          }
          v230 = v331;
LABEL_274:
          if ( v227 )
          {
            v231 = *(_QWORD *)(*(_QWORD *)(v226 - 8) + 24LL);
            if ( v231 != v325 )
              goto LABEL_276;
          }
          else
          {
            v231 = *(_QWORD *)(v226 - 24LL * (*(_DWORD *)(v226 + 20) & 0xFFFFFFF) + 24);
            if ( v231 != v325 )
            {
LABEL_276:
              v232 = sub_193FBC0(v48, v231, *(__int64 ***)(v48 + 8), v225, v226);
              v226 = v330;
              v233 = v232;
              goto LABEL_277;
            }
          }
          v233 = (__int64)v331;
LABEL_277:
          v304 = v226;
          v234 = sub_1649960(v226);
          v335 = v235;
          LOWORD(v344) = 261;
          v334 = (__int64 *)v234;
          v342 = (__int64)&v334;
          v236 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v304 + 16) - 24, v230, v233, (__int64)&v342, 0);
          v237 = v330;
          v209 = v236;
          v238 = sub_16498A0(v330);
          v239 = v304;
          v342 = 0;
          v345 = v238;
          v346 = 0;
          LODWORD(v347) = 0;
          v348 = 0;
          v349 = 0;
          v343 = *(_QWORD *)(v237 + 40);
          v344 = (__int64 *)(v237 + 24);
          v334 = *(__int64 **)(v237 + 48);
          if ( v334 )
          {
            sub_1623A60((__int64)&v334, (__int64)v334, 2);
            v239 = v304;
            if ( v342 )
            {
              sub_161E7C0((__int64)&v342, v342);
              v240 = v334;
              v239 = v304;
            }
            else
            {
              v240 = v334;
            }
            v342 = (__int64)v240;
            if ( v240 )
            {
              v305 = v239;
              sub_1623210((__int64)&v334, (unsigned __int8 *)v240, (__int64)&v342);
              v239 = v305;
            }
          }
          LOWORD(v336) = 257;
          if ( v343 )
          {
            v241 = v344;
            v306 = v239;
            sub_157E9D0(v343 + 40, v209);
            v242 = *(_QWORD *)(v209 + 24);
            v239 = v306;
            v243 = *v241;
            *(_QWORD *)(v209 + 32) = v241;
            v243 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v209 + 24) = v243 | v242 & 7;
            *(_QWORD *)(v243 + 8) = v209 + 24;
            *v241 = *v241 & 7 | (v209 + 24);
          }
          v307 = v239;
          sub_164B780(v209, (__int64 *)&v334);
          v244 = v307;
          if ( v342 )
          {
            v332 = (unsigned __int8 *)v342;
            sub_1623A60((__int64)&v332, v342, 2);
            v245 = *(_QWORD *)(v209 + 48);
            v244 = v307;
            if ( v245 )
            {
              sub_161E7C0(v209 + 48, v245);
              v244 = v307;
            }
            v246 = v332;
            *(_QWORD *)(v209 + 48) = v332;
            if ( v246 )
            {
              v308 = v244;
              sub_1623210((__int64)&v332, v246, v209 + 48);
              v244 = v308;
            }
          }
          v217 = v244;
LABEL_260:
          sub_15F2530((unsigned __int8 *)v209, v217, 1);
          if ( v342 )
            sub_161E7C0((__int64)&v342, v342);
          if ( v209 )
            goto LABEL_263;
          goto LABEL_76;
        case '/':
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
          v202 = (unsigned int)sub_193EAF0(v48, v325) == 1;
          if ( (*(_BYTE *)(v50 + 23) & 0x40) != 0 )
          {
            v203 = *(__int64 **)(v50 - 8);
            v204 = (__int64 *)v52;
            v205 = *v203;
            if ( v201 == *v203 )
              goto LABEL_245;
          }
          else
          {
            v204 = (__int64 *)v52;
            v203 = (__int64 *)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF));
            v205 = *v203;
            if ( v325 == *v203 )
              goto LABEL_245;
          }
          v204 = (__int64 *)sub_193FBC0(v48, v205, *(__int64 ***)(v48 + 8), v202, v50);
          if ( (*(_BYTE *)(v50 + 23) & 0x40) != 0 )
            v203 = *(__int64 **)(v50 - 8);
          else
            v203 = (__int64 *)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF));
LABEL_245:
          v206 = v203[3];
          if ( v206 != v325 )
            v52 = sub_193FBC0(v48, v206, *(__int64 ***)(v48 + 8), v202, v50);
          v207 = (__int64 *)sub_1649960(v50);
          v340 = v208;
          v339 = v207;
          LOWORD(v344) = 261;
          v342 = (__int64)&v339;
          v209 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v50 + 16) - 24, v204, v52, (__int64)&v342, 0);
          v210 = sub_16498A0(v50);
          v342 = 0;
          v345 = v210;
          v346 = 0;
          LODWORD(v347) = 0;
          v348 = 0;
          v349 = 0;
          v343 = *(_QWORD *)(v50 + 40);
          v344 = (__int64 *)(v50 + 24);
          v211 = *(__int64 **)(v50 + 48);
          v339 = v211;
          if ( v211 )
          {
            sub_1623A60((__int64)&v339, (__int64)v211, 2);
            if ( v342 )
              sub_161E7C0((__int64)&v342, v342);
            v342 = (__int64)v339;
            if ( v339 )
              sub_1623210((__int64)&v339, (unsigned __int8 *)v339, (__int64)&v342);
          }
          LOWORD(v341[0]) = 257;
          if ( v343 )
          {
            v212 = v344;
            sub_157E9D0(v343 + 40, v209);
            v213 = *(_QWORD *)(v209 + 24);
            v214 = *v212;
            *(_QWORD *)(v209 + 32) = v212;
            v214 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v209 + 24) = v214 | v213 & 7;
            *(_QWORD *)(v214 + 8) = v209 + 24;
            *v212 = *v212 & 7 | (v209 + 24);
          }
          sub_164B780(v209, (__int64 *)&v339);
          if ( v342 )
          {
            v334 = (__int64 *)v342;
            sub_1623A60((__int64)&v334, v342, 2);
            v215 = *(_QWORD *)(v209 + 48);
            if ( v215 )
              sub_161E7C0(v209 + 48, v215);
            v216 = (unsigned __int8 *)v334;
            *(_QWORD *)(v209 + 48) = v334;
            if ( v216 )
              sub_1623210((__int64)&v334, v216, v209 + 48);
          }
          v217 = v50;
          goto LABEL_260;
        default:
          goto LABEL_76;
      }
    }
    v209 = *(_QWORD *)(v48 + 64);
LABEL_263:
    if ( v321 == sub_146F1B0(*(_QWORD *)(v48 + 32), v209) )
    {
      v218 = *(_DWORD *)(v48 + 552);
      if ( v218 )
      {
        v219 = *(_QWORD *)(v48 + 536);
        v220 = (v218 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
        v221 = (__int64 *)(v219 + 16LL * v220);
        v222 = *v221;
        if ( v50 == *v221 )
        {
LABEL_266:
          *((_DWORD *)v221 + 2) = v311;
          if ( v209 )
            sub_193F7B0(v48, v50, v209);
          goto LABEL_76;
        }
        v259 = 1;
        v260 = 0;
        while ( v222 != -8 )
        {
          if ( !v260 && v222 == -16 )
            v260 = v221;
          v220 = (v218 - 1) & (v259 + v220);
          v221 = (__int64 *)(v219 + 16LL * v220);
          v222 = *v221;
          if ( v50 == *v221 )
            goto LABEL_266;
          ++v259;
        }
        v261 = *(_DWORD *)(v48 + 544);
        if ( v260 )
          v221 = v260;
        ++*(_QWORD *)(v48 + 528);
        v262 = v261 + 1;
        if ( 4 * (v261 + 1) < 3 * v218 )
        {
          if ( v218 - *(_DWORD *)(v48 + 548) - v262 <= v218 >> 3 )
          {
            sub_193E940(v48 + 528, v218);
            v278 = *(_DWORD *)(v48 + 552);
            if ( !v278 )
            {
LABEL_441:
              ++*(_DWORD *)(v48 + 544);
              BUG();
            }
            v279 = v278 - 1;
            v280 = 1;
            v281 = *(_QWORD *)(v48 + 536);
            v282 = v279 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
            v262 = *(_DWORD *)(v48 + 544) + 1;
            v283 = 0;
            v221 = (__int64 *)(v281 + 16LL * v282);
            v284 = *v221;
            if ( v50 != *v221 )
            {
              while ( v284 != -8 )
              {
                if ( !v283 && v284 == -16 )
                  v283 = v221;
                v282 = v279 & (v280 + v282);
                v221 = (__int64 *)(v281 + 16LL * v282);
                v284 = *v221;
                if ( v50 == *v221 )
                  goto LABEL_371;
                ++v280;
              }
              if ( v283 )
                v221 = v283;
            }
          }
          goto LABEL_371;
        }
      }
      else
      {
        ++*(_QWORD *)(v48 + 528);
      }
      sub_193E940(v48 + 528, 2 * v218);
      v268 = *(_DWORD *)(v48 + 552);
      if ( !v268 )
        goto LABEL_441;
      v269 = v268 - 1;
      v270 = *(_QWORD *)(v48 + 536);
      v262 = *(_DWORD *)(v48 + 544) + 1;
      v271 = v269 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v221 = (__int64 *)(v270 + 16LL * v271);
      v272 = *v221;
      if ( v50 != *v221 )
      {
        v273 = 1;
        v274 = 0;
        while ( v272 != -8 )
        {
          if ( v272 == -16 && !v274 )
            v274 = v221;
          v271 = v269 & (v273 + v271);
          v221 = (__int64 *)(v270 + 16LL * v271);
          v272 = *v221;
          if ( v50 == *v221 )
            goto LABEL_371;
          ++v273;
        }
        if ( v274 )
          v221 = v274;
      }
LABEL_371:
      *(_DWORD *)(v48 + 544) = v262;
      if ( *v221 != -8 )
        --*(_DWORD *)(v48 + 548);
      *v221 = v50;
      *((_DWORD *)v221 + 2) = 0;
      goto LABEL_266;
    }
    v247 = *(_QWORD *)(v48 + 80);
    v248 = *(_DWORD *)(v247 + 8);
    if ( v248 >= *(_DWORD *)(v247 + 12) )
    {
      sub_170B450(*(_QWORD *)(v48 + 80), 0);
      v248 = *(_DWORD *)(v247 + 8);
    }
    v249 = (_QWORD *)(*(_QWORD *)v247 + 24LL * v248);
    if ( v249 )
    {
      *v249 = 6;
      v249[1] = 0;
      v249[2] = v209;
      if ( v209 != 0 && v209 != -8 && v209 != -16 )
        sub_164C220((__int64)v249);
      v248 = *(_DWORD *)(v247 + 8);
    }
    *(_DWORD *)(v247 + 8) = v248 + 1;
LABEL_76:
    if ( !*(_QWORD *)(v325 + 8) )
    {
      v196 = *(_QWORD *)(v48 + 80);
      v197 = *(_DWORD *)(v196 + 8);
      if ( v197 >= *(_DWORD *)(v196 + 12) )
      {
        sub_170B450(*(_QWORD *)(v48 + 80), 0);
        v197 = *(_DWORD *)(v196 + 8);
      }
      v198 = (_QWORD *)(*(_QWORD *)v196 + 24LL * v197);
      if ( v198 )
      {
        *v198 = 6;
        v198[1] = 0;
        v198[2] = v325;
        if ( v325 != -16 && v325 != -8 )
          sub_164C220((__int64)v198);
        v197 = *(_DWORD *)(v196 + 8);
      }
      *(_DWORD *)(v196 + 8) = v197 + 1;
    }
    v47 = *(_DWORD *)(v48 + 264);
  }
  while ( v47 );
  v11 = v48;
LABEL_79:
  v96 = *(_QWORD *)v11;
  v343 = 0x100000000LL;
  v342 = (__int64)&v344;
  sub_1AEA1F0(&v342, v96);
  v97 = sub_1624210(*(_QWORD *)(v11 + 56));
  v98 = (__int64 *)sub_16498A0(*(_QWORD *)(v11 + 56));
  v99 = sub_1628DA0(v98, (__int64)v97);
  v100 = v342;
  v101 = v99;
  v102 = v99 + 8;
  v103 = (__int64 *)(v342 + 8LL * (unsigned int)v343);
  if ( (__int64 *)v342 != v103 )
  {
    do
    {
      v104 = (__int64 *)(*(_QWORD *)v100 - 24LL * (*(_DWORD *)(*(_QWORD *)v100 + 20LL) & 0xFFFFFFF));
      if ( *v104 )
      {
        v105 = v104[1];
        v106 = v104[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v106 = v105;
        if ( v105 )
          *(_QWORD *)(v105 + 16) = *(_QWORD *)(v105 + 16) & 3LL | v106;
      }
      *v104 = v101;
      if ( v101 )
      {
        v107 = *(_QWORD *)(v101 + 8);
        v104[1] = v107;
        if ( v107 )
          *(_QWORD *)(v107 + 16) = (unsigned __int64)(v104 + 1) | *(_QWORD *)(v107 + 16) & 3LL;
        v104[2] = v102 | v104[2] & 3;
        *(_QWORD *)(v101 + 8) = v104;
      }
      v100 += 8;
    }
    while ( v103 != (__int64 *)v100 );
    v103 = (__int64 *)v342;
  }
  result = *(_QWORD *)(v11 + 56);
  if ( v103 != (__int64 *)&v344 )
  {
    v326 = *(_QWORD *)(v11 + 56);
    _libc_free((unsigned __int64)v103);
    return v326;
  }
  return result;
}
