// Function: sub_2F9DAC0
// Address: 0x2f9dac0
//
__int64 __fastcall sub_2F9DAC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rdx
  unsigned __int64 v6; // r15
  const __m128i *v7; // rbx
  const __m128i *v8; // r13
  int v9; // esi
  __int64 *v10; // rdi
  unsigned int v11; // eax
  __int64 *v12; // r9
  __int64 v13; // r8
  __int64 v14; // r12
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 *v17; // rcx
  unsigned int v18; // edi
  __m128i v19; // xmm1
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 *v23; // rbx
  __int64 *v24; // r13
  int v25; // esi
  __int64 *v26; // rdi
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // r9
  __int64 v30; // r12
  unsigned int v31; // esi
  unsigned int v32; // eax
  __int64 *v33; // r8
  unsigned int v34; // edx
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rbx
  _QWORD *v39; // r8
  _QWORD *v40; // r14
  unsigned __int64 v41; // r12
  unsigned __int64 v42; // r15
  _BYTE *v43; // rdx
  unsigned int v44; // ecx
  __int64 v45; // r13
  _BYTE *v46; // r10
  unsigned __int64 v47; // r14
  __int64 **v48; // r10
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rbx
  _BYTE *v54; // rsi
  int v55; // ecx
  unsigned __int8 **v56; // r11
  __int64 v57; // rax
  __int64 v58; // rax
  int v59; // edx
  unsigned __int64 v60; // rbx
  unsigned __int64 *v61; // rbx
  unsigned __int64 *v62; // r12
  unsigned __int64 v63; // rdi
  char v64; // r14
  __int64 *v65; // rbx
  unsigned __int64 v66; // r13
  __int16 v67; // r14
  unsigned __int64 v68; // r12
  int v69; // r11d
  unsigned int v70; // r8d
  __int64 *v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // rdi
  unsigned __int64 v74; // r8
  __int16 v75; // cx
  __int64 *v76; // r15
  __int64 v77; // r15
  __int64 v78; // rcx
  int v79; // edx
  __int64 v80; // rdi
  __int16 v81; // r12
  unsigned __int64 v82; // rbx
  char v83; // si
  int v84; // eax
  __int64 *v85; // rcx
  unsigned int v86; // edx
  __int64 *v87; // rdi
  __int64 v88; // r8
  __int64 v89; // rax
  int v90; // edi
  __int64 *v91; // rcx
  unsigned int v92; // edx
  __int64 *v93; // rax
  __int64 v94; // r8
  unsigned __int64 v95; // rax
  __int64 v96; // rdx
  unsigned __int64 v97; // rax
  __int64 v98; // rdx
  unsigned __int16 v99; // dx
  __int16 v100; // bx
  unsigned __int64 v101; // r8
  __int16 v102; // r9
  unsigned __int16 v103; // dx
  __int16 v104; // ax
  unsigned __int64 v105; // r12
  __int16 v106; // r14
  __int16 v107; // si
  unsigned __int64 v108; // rbx
  _BYTE *v109; // rdx
  unsigned int v110; // ecx
  __int64 v111; // rsi
  _BYTE *v112; // r9
  __int16 v113; // cx
  unsigned __int64 v114; // rdx
  __int16 v115; // ax
  unsigned int v116; // r14d
  unsigned __int64 v117; // r12
  bool v118; // sf
  unsigned __int64 *v119; // rax
  __int128 v120; // rax
  int v121; // r11d
  __int64 *v122; // rdx
  unsigned int v123; // edi
  _QWORD *v124; // rax
  __int64 v125; // rcx
  unsigned __int64 *v126; // rax
  int v127; // eax
  __int64 v128; // rdx
  int v129; // r9d
  int v130; // r10d
  int v131; // esi
  __int64 *v132; // rdi
  __int64 v133; // rdx
  __int64 v134; // r8
  int v135; // r9d
  __int64 *v136; // rax
  int v137; // r10d
  int v138; // ecx
  __int64 *v139; // rsi
  __int64 v140; // rdx
  __int64 v141; // rdi
  int v142; // r9d
  __int64 *v143; // rax
  __int64 v144; // rax
  int v145; // esi
  int v146; // esi
  __int64 *v147; // rdi
  __int64 v148; // rdx
  __int64 v149; // r8
  int v150; // r9d
  int v151; // ecx
  __int64 *v152; // rsi
  __int64 v153; // rdx
  __int64 v154; // rdi
  int v155; // r9d
  __int64 v156; // rdi
  unsigned __int64 v157; // rbx
  unsigned __int64 v158; // rax
  unsigned __int64 v159; // r14
  unsigned __int16 v160; // dx
  unsigned __int16 v161; // r12
  unsigned __int64 v162; // rax
  unsigned __int16 v163; // dx
  __int16 v164; // dx
  unsigned __int64 v165; // rax
  int v166; // ebx
  int v167; // r15d
  int v168; // eax
  __int64 v169; // r8
  unsigned __int64 *v170; // rbx
  unsigned __int64 *v171; // r12
  unsigned __int64 v172; // rdi
  int v174; // edi
  __int64 v175; // rax
  unsigned int v176; // ecx
  __int64 v177; // rdi
  int v178; // r11d
  __int64 *v179; // r10
  int v180; // eax
  __int64 v181; // r8
  unsigned __int64 v182; // r13
  unsigned __int64 v183; // r15
  __int8 *v184; // rsi
  size_t v185; // rdx
  unsigned __int64 *v186; // rbx
  unsigned __int64 *v187; // r12
  unsigned __int64 v188; // rdi
  __int64 *v189; // r9
  int v190; // r11d
  __int64 v191; // rcx
  __int64 v192; // rdi
  __int64 v193; // rdx
  __int64 v194; // rcx
  __int64 v195; // r8
  __int64 v196; // r9
  unsigned __int64 *v197; // rbx
  unsigned __int64 v198; // rdi
  int v199; // r10d
  __int64 *v200; // r9
  unsigned int v201; // r14d
  __int64 v202; // rsi
  __int16 v203; // bx
  int v204; // eax
  unsigned __int64 v205; // rbx
  unsigned __int64 v206; // r13
  char v207; // r15
  int v208; // eax
  int v209; // eax
  __int64 v210; // rax
  __int64 v211; // rdx
  unsigned __int64 v212; // rax
  unsigned __int64 v213; // r13
  unsigned __int64 v214; // rbx
  __int64 v215; // rdx
  char v216; // r15
  int v217; // eax
  __int8 *v218; // rax
  __int16 v219; // dx
  __int64 v220; // rax
  unsigned __int64 v221; // r13
  __int16 v222; // bx
  int v223; // r9d
  int v224; // r9d
  int v225; // r11d
  int v226; // eax
  int v227; // r10d
  unsigned __int64 v228; // rax
  __int16 v229; // dx
  unsigned __int64 v230; // r15
  unsigned __int64 v231; // r13
  __int16 v232; // bx
  int v233; // eax
  unsigned __int64 v234; // rbx
  unsigned __int64 v235; // r13
  char v236; // r15
  int v237; // eax
  __int8 *v238; // rax
  size_t v239; // rdx
  __int128 v240; // rax
  __int16 v241; // dx
  __int64 v242; // rax
  unsigned __int64 v243; // rbx
  unsigned __int64 v244; // r15
  int v245; // r13d
  int v246; // eax
  int v247; // eax
  int v248; // eax
  int v249; // eax
  unsigned __int64 v250; // rbx
  char v251; // r15
  int v252; // eax
  int v253; // eax
  int v254; // eax
  __int64 v257; // [rsp+18h] [rbp-3C8h]
  __int64 v259; // [rsp+30h] [rbp-3B0h]
  __int16 v260; // [rsp+3Ch] [rbp-3A4h]
  unsigned __int64 v261; // [rsp+40h] [rbp-3A0h]
  _QWORD *v262; // [rsp+48h] [rbp-398h]
  __int64 v263; // [rsp+50h] [rbp-390h]
  unsigned __int64 v264; // [rsp+58h] [rbp-388h]
  __int16 v265; // [rsp+58h] [rbp-388h]
  __int64 v266; // [rsp+58h] [rbp-388h]
  __int64 v267; // [rsp+60h] [rbp-380h]
  __int16 v268; // [rsp+60h] [rbp-380h]
  __int64 v269; // [rsp+60h] [rbp-380h]
  __int64 v270; // [rsp+68h] [rbp-378h]
  int v271; // [rsp+70h] [rbp-370h]
  __int64 **v272; // [rsp+70h] [rbp-370h]
  unsigned __int64 *v273; // [rsp+78h] [rbp-368h]
  __int64 v275; // [rsp+88h] [rbp-358h]
  __int64 v276; // [rsp+90h] [rbp-350h]
  bool v277; // [rsp+98h] [rbp-348h]
  unsigned __int64 v278; // [rsp+98h] [rbp-348h]
  __int64 v279; // [rsp+A0h] [rbp-340h]
  __int16 v280; // [rsp+A0h] [rbp-340h]
  __int16 v281; // [rsp+A0h] [rbp-340h]
  __int16 v282; // [rsp+A0h] [rbp-340h]
  __int16 v283; // [rsp+A0h] [rbp-340h]
  __int16 v284; // [rsp+A0h] [rbp-340h]
  __int16 v285; // [rsp+A8h] [rbp-338h]
  unsigned __int64 v286; // [rsp+A8h] [rbp-338h]
  __int16 v287; // [rsp+A8h] [rbp-338h]
  __int16 v288; // [rsp+A8h] [rbp-338h]
  __int16 v289; // [rsp+A8h] [rbp-338h]
  __int16 v290; // [rsp+A8h] [rbp-338h]
  __int16 v291; // [rsp+A8h] [rbp-338h]
  __int16 v292; // [rsp+A8h] [rbp-338h]
  __int16 v293; // [rsp+B0h] [rbp-330h]
  __int64 *v294; // [rsp+B0h] [rbp-330h]
  unsigned __int64 v295; // [rsp+B0h] [rbp-330h]
  unsigned __int64 v296; // [rsp+B0h] [rbp-330h]
  char v297; // [rsp+B0h] [rbp-330h]
  __int16 v298; // [rsp+B0h] [rbp-330h]
  __int16 v299; // [rsp+B0h] [rbp-330h]
  char v300; // [rsp+B0h] [rbp-330h]
  __int16 v301; // [rsp+B0h] [rbp-330h]
  __int16 v302; // [rsp+B0h] [rbp-330h]
  __int64 v303; // [rsp+B8h] [rbp-328h]
  __int16 v304; // [rsp+B8h] [rbp-328h]
  int v305; // [rsp+B8h] [rbp-328h]
  int v306; // [rsp+B8h] [rbp-328h]
  int v307; // [rsp+B8h] [rbp-328h]
  int v308; // [rsp+B8h] [rbp-328h]
  int v309; // [rsp+B8h] [rbp-328h]
  int v310; // [rsp+B8h] [rbp-328h]
  int v311; // [rsp+B8h] [rbp-328h]
  unsigned __int16 v312; // [rsp+C4h] [rbp-31Ch] BYREF
  unsigned __int16 v313; // [rsp+C6h] [rbp-31Ah] BYREF
  unsigned __int64 v314; // [rsp+C8h] [rbp-318h] BYREF
  unsigned __int64 v315; // [rsp+D0h] [rbp-310h] BYREF
  unsigned __int64 v316; // [rsp+D8h] [rbp-308h] BYREF
  __m128i v317; // [rsp+E0h] [rbp-300h] BYREF
  unsigned __int64 v318; // [rsp+F0h] [rbp-2F0h] BYREF
  __int16 v319; // [rsp+F8h] [rbp-2E8h]
  __int128 v320; // [rsp+100h] [rbp-2E0h] BYREF
  __int64 v321; // [rsp+110h] [rbp-2D0h] BYREF
  __int64 v322; // [rsp+118h] [rbp-2C8h]
  __int64 v323; // [rsp+120h] [rbp-2C0h]
  unsigned int v324; // [rsp+128h] [rbp-2B8h]
  unsigned __int64 v325; // [rsp+130h] [rbp-2B0h] BYREF
  __int64 v326; // [rsp+138h] [rbp-2A8h]
  unsigned __int64 v327; // [rsp+140h] [rbp-2A0h] BYREF
  __int64 v328; // [rsp+148h] [rbp-298h]
  __int8 *v329; // [rsp+150h] [rbp-290h] BYREF
  size_t v330; // [rsp+158h] [rbp-288h]
  __int64 v331; // [rsp+160h] [rbp-280h] BYREF
  unsigned int v332; // [rsp+168h] [rbp-278h]
  __int64 v333; // [rsp+170h] [rbp-270h]
  unsigned __int64 v334; // [rsp+180h] [rbp-260h] BYREF
  __int64 v335; // [rsp+188h] [rbp-258h]
  unsigned __int64 v336; // [rsp+190h] [rbp-250h] BYREF
  __int16 v337; // [rsp+198h] [rbp-248h]
  unsigned __int64 v338; // [rsp+1A0h] [rbp-240h] BYREF
  __int64 v339; // [rsp+1A8h] [rbp-238h]
  unsigned __int64 v340; // [rsp+1B0h] [rbp-230h] BYREF
  __int16 v341; // [rsp+1B8h] [rbp-228h]
  unsigned __int64 v342; // [rsp+1C0h] [rbp-220h] BYREF
  size_t v343; // [rsp+1C8h] [rbp-218h]
  __int64 v344; // [rsp+1D0h] [rbp-210h] BYREF
  unsigned int v345; // [rsp+1D8h] [rbp-208h]
  __int64 v346; // [rsp+1E8h] [rbp-1F8h]
  unsigned __int64 v347; // [rsp+200h] [rbp-1E0h] BYREF
  __int64 v348; // [rsp+208h] [rbp-1D8h]
  _BYTE v349[64]; // [rsp+210h] [rbp-1D0h] BYREF
  unsigned __int64 *v350; // [rsp+250h] [rbp-190h]
  unsigned int v351; // [rsp+258h] [rbp-188h]
  _BYTE v352[384]; // [rsp+260h] [rbp-180h] BYREF

  LOWORD(v335) = 0;
  v4 = *(_QWORD *)a3;
  v5 = *(unsigned int *)(a3 + 8);
  v337 = 0;
  LOWORD(v339) = 0;
  v341 = 0;
  v6 = v4 + 56 * v5;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v334 = 0;
  v336 = 0;
  v338 = 0;
  v340 = 0;
  v342 = 0;
  v343 = 1;
  v344 = -4096;
  v346 = -4096;
  if ( v4 != v6 )
  {
    while ( 1 )
    {
      v7 = *(const __m128i **)(v4 + 8);
      v8 = &v7[*(unsigned int *)(v4 + 16)];
      if ( v7 != v8 )
        break;
LABEL_16:
      v4 += 56LL;
      if ( v6 == v4 )
      {
        v329 = 0;
        v330 = 1;
        v20 = *(unsigned int *)(a3 + 8);
        v21 = *(_QWORD *)a3;
        v331 = -4096;
        v333 = -4096;
        v22 = v21 + 56 * v20;
        if ( v21 != v22 )
        {
          while ( 1 )
          {
            v23 = *(__int64 **)(v21 + 8);
            v24 = &v23[2 * *(unsigned int *)(v21 + 16)];
            if ( v23 != v24 )
              break;
LABEL_32:
            v21 += 56;
            if ( v21 == v22 )
              goto LABEL_33;
          }
          while ( 1 )
          {
            v30 = *v23;
            if ( (v330 & 1) != 0 )
            {
              v25 = 1;
              v26 = &v331;
            }
            else
            {
              v31 = v332;
              v26 = (__int64 *)v331;
              if ( !v332 )
              {
                v32 = v330;
                ++v329;
                v33 = 0;
                v34 = ((unsigned int)v330 >> 1) + 1;
                goto LABEL_27;
              }
              v25 = v332 - 1;
            }
            v27 = v25 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v28 = &v26[2 * v27];
            v29 = *v28;
            if ( v30 == *v28 )
            {
LABEL_22:
              v23 += 2;
              if ( v24 == v23 )
                goto LABEL_32;
            }
            else
            {
              v137 = 1;
              v33 = 0;
              while ( v29 != -4096 )
              {
                if ( v33 || v29 != -8192 )
                  v28 = v33;
                v27 = v25 & (v137 + v27);
                v29 = v26[2 * v27];
                if ( v30 == v29 )
                  goto LABEL_22;
                ++v137;
                v33 = v28;
                v28 = &v26[2 * v27];
              }
              if ( !v33 )
                v33 = v28;
              v32 = v330;
              ++v329;
              v34 = ((unsigned int)v330 >> 1) + 1;
              if ( (v330 & 1) == 0 )
              {
                v31 = v332;
LABEL_27:
                if ( 4 * v34 < 3 * v31 )
                  goto LABEL_28;
                goto LABEL_188;
              }
              v31 = 2;
              if ( 4 * v34 < 6 )
              {
LABEL_28:
                if ( v31 - HIDWORD(v330) - v34 > v31 >> 3 )
                  goto LABEL_29;
                sub_2F9B420((__int64)&v329, v31);
                if ( (v330 & 1) != 0 )
                {
                  v151 = 1;
                  v152 = &v331;
                }
                else
                {
                  v152 = (__int64 *)v331;
                  if ( !v332 )
                  {
LABEL_462:
                    LODWORD(v330) = (2 * ((unsigned int)v330 >> 1) + 2) | v330 & 1;
                    BUG();
                  }
                  v151 = v332 - 1;
                }
                v32 = v330;
                LODWORD(v153) = v151 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v33 = &v152[2 * (unsigned int)v153];
                v154 = *v33;
                if ( v30 == *v33 )
                  goto LABEL_29;
                v155 = 1;
                v143 = 0;
                while ( v154 != -4096 )
                {
                  if ( !v143 && v154 == -8192 )
                    v143 = v33;
                  v153 = v151 & (unsigned int)(v153 + v155);
                  v33 = &v152[2 * v153];
                  v154 = *v33;
                  if ( v30 == *v33 )
                    goto LABEL_195;
                  ++v155;
                }
                goto LABEL_193;
              }
LABEL_188:
              sub_2F9B420((__int64)&v329, 2 * v31);
              if ( (v330 & 1) != 0 )
              {
                v138 = 1;
                v139 = &v331;
              }
              else
              {
                v139 = (__int64 *)v331;
                if ( !v332 )
                  goto LABEL_462;
                v138 = v332 - 1;
              }
              v32 = v330;
              LODWORD(v140) = v138 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v33 = &v139[2 * (unsigned int)v140];
              v141 = *v33;
              if ( v30 == *v33 )
                goto LABEL_29;
              v142 = 1;
              v143 = 0;
              while ( v141 != -4096 )
              {
                if ( !v143 && v141 == -8192 )
                  v143 = v33;
                v140 = v138 & (unsigned int)(v140 + v142);
                v33 = &v139[2 * v140];
                v141 = *v33;
                if ( v30 == *v33 )
                  goto LABEL_195;
                ++v142;
              }
LABEL_193:
              if ( v143 )
                v33 = v143;
LABEL_195:
              v32 = v330;
LABEL_29:
              LODWORD(v330) = (2 * (v32 >> 1) + 2) | v32 & 1;
              if ( *v33 != -4096 )
                --HIDWORD(v330);
              v23 += 2;
              *v33 = v30;
              v33[1] = v21;
              if ( v24 == v23 )
                goto LABEL_32;
            }
          }
        }
        goto LABEL_33;
      }
    }
    while ( 1 )
    {
      v14 = v7->m128i_i64[0];
      if ( (v343 & 1) != 0 )
      {
        v9 = 1;
        v10 = &v344;
      }
      else
      {
        v15 = v345;
        v10 = (__int64 *)v344;
        if ( !v345 )
        {
          v16 = v343;
          ++v342;
          v17 = 0;
          v18 = ((unsigned int)v343 >> 1) + 1;
          goto LABEL_11;
        }
        v9 = v345 - 1;
      }
      v11 = v9 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v12 = &v10[3 * v11];
      v13 = *v12;
      if ( v14 == *v12 )
      {
LABEL_6:
        if ( v8 == ++v7 )
          goto LABEL_16;
      }
      else
      {
        v130 = 1;
        v17 = 0;
        while ( v13 != -4096 )
        {
          if ( v13 != -8192 || v17 )
            v12 = v17;
          v11 = v9 & (v130 + v11);
          v13 = v10[3 * v11];
          if ( v14 == v13 )
            goto LABEL_6;
          v17 = v12;
          ++v130;
          v12 = &v10[3 * v11];
        }
        v16 = v343;
        if ( !v17 )
          v17 = v12;
        ++v342;
        v18 = ((unsigned int)v343 >> 1) + 1;
        if ( (v343 & 1) == 0 )
        {
          v15 = v345;
LABEL_11:
          if ( 4 * v18 < 3 * v15 )
            goto LABEL_12;
          goto LABEL_174;
        }
        v15 = 2;
        if ( 4 * v18 < 6 )
        {
LABEL_12:
          if ( v15 - HIDWORD(v343) - v18 > v15 >> 3 )
            goto LABEL_13;
          sub_2F9AFE0((__int64)&v342, v15);
          if ( (v343 & 1) != 0 )
          {
            v146 = 1;
            v147 = &v344;
          }
          else
          {
            v147 = (__int64 *)v344;
            if ( !v345 )
            {
LABEL_461:
              LODWORD(v343) = (2 * ((unsigned int)v343 >> 1) + 2) | v343 & 1;
              BUG();
            }
            v146 = v345 - 1;
          }
          LODWORD(v148) = v146 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v17 = &v147[3 * (unsigned int)v148];
          v16 = v343;
          v149 = *v17;
          if ( v14 == *v17 )
            goto LABEL_13;
          v150 = 1;
          v136 = 0;
          while ( v149 != -4096 )
          {
            if ( v149 == -8192 && !v136 )
              v136 = v17;
            v148 = v146 & (unsigned int)(v148 + v150);
            v17 = &v147[3 * v148];
            v149 = *v17;
            if ( v14 == *v17 )
              goto LABEL_181;
            ++v150;
          }
          goto LABEL_179;
        }
LABEL_174:
        sub_2F9AFE0((__int64)&v342, 2 * v15);
        if ( (v343 & 1) != 0 )
        {
          v131 = 1;
          v132 = &v344;
        }
        else
        {
          v132 = (__int64 *)v344;
          if ( !v345 )
            goto LABEL_461;
          v131 = v345 - 1;
        }
        LODWORD(v133) = v131 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v17 = &v132[3 * (unsigned int)v133];
        v16 = v343;
        v134 = *v17;
        if ( v14 == *v17 )
          goto LABEL_13;
        v135 = 1;
        v136 = 0;
        while ( v134 != -4096 )
        {
          if ( !v136 && v134 == -8192 )
            v136 = v17;
          v133 = v131 & (unsigned int)(v133 + v135);
          v17 = &v132[3 * v133];
          v134 = *v17;
          if ( v14 == *v17 )
            goto LABEL_181;
          ++v135;
        }
LABEL_179:
        if ( v136 )
          v17 = v136;
LABEL_181:
        v16 = v343;
LABEL_13:
        LODWORD(v343) = (2 * (v16 >> 1) + 2) | v16 & 1;
        if ( *v17 != -4096 )
          --HIDWORD(v343);
        *v17 = v14;
        v19 = _mm_loadu_si128(v7++);
        *(__m128i *)(v17 + 1) = v19;
        if ( v8 == v7 )
          goto LABEL_16;
      }
    }
  }
  v329 = 0;
  v330 = 1;
  v331 = -4096;
  v333 = -4096;
LABEL_33:
  v273 = &v334;
  while ( 1 )
  {
    v257 = *(_QWORD *)(a2 + 40);
    v259 = *(_QWORD *)(a2 + 32);
    if ( v259 != v257 )
      break;
LABEL_223:
    v273 += 4;
    if ( &v342 == v273 )
    {
      v277 = 1;
      goto LABEL_74;
    }
  }
  while ( 1 )
  {
    v270 = *(_QWORD *)v259 + 48LL;
    v303 = *(_QWORD *)(*(_QWORD *)v259 + 56LL);
    if ( v303 != v270 )
      break;
LABEL_222:
    v259 += 8;
    if ( v257 == v259 )
      goto LABEL_223;
  }
  while ( 1 )
  {
    v35 = v303 - 24;
    if ( !v303 )
      v35 = 0;
    v277 = sub_B46AA0(v35);
    if ( v277 )
      goto LABEL_37;
    v36 = 32LL * (*(_DWORD *)(v35 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v35 + 7) & 0x40) != 0 )
    {
      v37 = *(_QWORD *)(v35 - 8);
      v38 = (_QWORD *)(v37 + v36);
      v39 = (_QWORD *)v37;
      if ( v37 == v37 + v36 )
      {
        v42 = 0;
        v285 = 0;
        v47 = 0;
        v293 = 0;
        v48 = *(__int64 ***)(a1 + 24);
        goto LABEL_55;
      }
    }
    else
    {
      v38 = (_QWORD *)v35;
      v39 = (_QWORD *)(v35 - v36);
      if ( v35 == v35 - v36 )
      {
        v42 = 0;
        v285 = 0;
        v47 = 0;
        v293 = 0;
        v48 = *(__int64 ***)(a1 + 24);
        goto LABEL_163;
      }
    }
    v276 = v35;
    v40 = v39;
    v293 = 0;
    v41 = 0;
    v285 = 0;
    v42 = 0;
    do
    {
      v43 = (_BYTE *)*v40;
      if ( *(_BYTE *)*v40 > 0x1Cu && v324 )
      {
        v44 = (v324 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v45 = v322 + 40LL * v44;
        v46 = *(_BYTE **)v45;
        if ( v43 == *(_BYTE **)v45 )
        {
LABEL_47:
          if ( v45 != v322 + 40LL * v324 )
          {
            if ( (int)sub_D788E0(v42, v285, *(_QWORD *)(v45 + 8), *(_WORD *)(v45 + 16)) < 0 )
            {
              v42 = *(_QWORD *)(v45 + 8);
              v285 = *(_WORD *)(v45 + 16);
            }
            if ( (int)sub_D788E0(v41, v293, *(_QWORD *)(v45 + 24), *(_WORD *)(v45 + 32)) < 0 )
            {
              v41 = *(_QWORD *)(v45 + 24);
              v293 = *(_WORD *)(v45 + 32);
            }
          }
        }
        else
        {
          v129 = 1;
          while ( v46 != (_BYTE *)-4096LL )
          {
            v44 = (v324 - 1) & (v129 + v44);
            v45 = v322 + 40LL * v44;
            v46 = *(_BYTE **)v45;
            if ( v43 == *(_BYTE **)v45 )
              goto LABEL_47;
            ++v129;
          }
        }
      }
      v40 += 4;
    }
    while ( v38 != v40 );
    v35 = v276;
    v47 = v41;
    v48 = *(__int64 ***)(a1 + 24);
    v36 = 32LL * (*(_DWORD *)(v276 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v276 + 7) & 0x40) != 0 )
    {
      v37 = *(_QWORD *)(v276 - 8);
LABEL_55:
      v49 = v37;
      v50 = v37 + v36;
      goto LABEL_56;
    }
LABEL_163:
    v49 = v35 - v36;
    v50 = v35;
LABEL_56:
    v51 = v50 - v49;
    v348 = 0x400000000LL;
    v52 = v51 >> 5;
    v347 = (unsigned __int64)v349;
    v53 = v51 >> 5;
    if ( (unsigned __int64)v51 > 0x80 )
    {
      v266 = v51;
      v269 = v49;
      v272 = v48;
      v275 = v51 >> 5;
      sub_C8D5F0((__int64)&v347, v349, v52, 8u, v49, (__int64)v349);
      v56 = (unsigned __int8 **)v347;
      v55 = v348;
      LODWORD(v52) = v275;
      v48 = v272;
      v49 = v269;
      v54 = (_BYTE *)(v347 + 8LL * (unsigned int)v348);
      v51 = v266;
    }
    else
    {
      v54 = v349;
      v55 = 0;
      v56 = (unsigned __int8 **)v349;
    }
    if ( v51 > 0 )
    {
      v57 = 0;
      do
      {
        *(_QWORD *)&v54[v57] = *(_QWORD *)(v49 + 4 * v57);
        v57 += 8;
        --v53;
      }
      while ( v53 );
      v56 = (unsigned __int8 **)v347;
      v55 = v348;
    }
    LODWORD(v348) = v55 + v52;
    v58 = sub_DFCEF0(v48, (unsigned __int8 *)v35, v56, (unsigned int)(v55 + v52), 1);
    v271 = v59;
    v60 = v58;
    if ( (_BYTE *)v347 != v349 )
      _libc_free(v347);
    if ( v271 )
      break;
    v325 = v42;
    v347 = v60;
    LOWORD(v318) = v285;
    LOWORD(v320) = 0;
    v287 = sub_FDCA70(&v325, (unsigned __int16 *)&v318, &v347, (unsigned __int16 *)&v320);
    v278 = v325 + v347;
    if ( __CFADD__(v325, v347) )
    {
      ++v287;
      v278 = (v278 >> 1) | 0x8000000000000000LL;
    }
    if ( v287 > 0x3FFF )
    {
      v278 = -1;
      v287 = 0x3FFF;
    }
    v347 = v60;
    LOWORD(v320) = 0;
    v325 = v47;
    LOWORD(v318) = v293;
    v81 = sub_FDCA70(&v325, (unsigned __int16 *)&v318, &v347, (unsigned __int16 *)&v320);
    v82 = v325 + v347;
    if ( __CFADD__(v325, v347) )
    {
      ++v81;
      v82 = (v82 >> 1) | 0x8000000000000000LL;
    }
    if ( v81 > 0x3FFF )
    {
      v81 = 0x3FFF;
      v82 = -1;
    }
    v83 = v343 & 1;
    if ( (v343 & 1) != 0 )
    {
      v84 = 1;
      v85 = &v344;
    }
    else
    {
      v128 = v345;
      v85 = (__int64 *)v344;
      if ( !v345 )
        goto LABEL_235;
      v84 = v345 - 1;
    }
    v86 = v84 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v87 = &v85[3 * v86];
    v88 = *v87;
    if ( v35 == *v87 )
      goto LABEL_108;
    v174 = 1;
    while ( v88 != -4096 )
    {
      v224 = v174 + 1;
      v86 = v84 & (v174 + v86);
      v87 = &v85[3 * v86];
      v88 = *v87;
      if ( v35 == *v87 )
        goto LABEL_108;
      v174 = v224;
    }
    if ( v83 )
    {
      v156 = 6;
      goto LABEL_236;
    }
    v128 = v345;
LABEL_235:
    v156 = 3 * v128;
LABEL_236:
    v87 = &v85[v156];
LABEL_108:
    v89 = 6;
    if ( !v83 )
      v89 = 3LL * v345;
    if ( v87 != &v85[v89] )
    {
      v317 = _mm_loadu_si128((const __m128i *)(v87 + 1));
      if ( (v330 & 1) != 0 )
      {
        v90 = 1;
        v91 = &v331;
      }
      else
      {
        v144 = v332;
        v91 = (__int64 *)v331;
        if ( !v332 )
          goto LABEL_266;
        v90 = v332 - 1;
      }
      v92 = v90 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v93 = &v91[2 * v92];
      v94 = *v93;
      if ( v35 == *v93 )
      {
LABEL_114:
        v262 = (_QWORD *)v93[1];
        v95 = sub_2F9C740((unsigned __int8 **)&v317, 1, (__int64)&v321, *(_QWORD *)(a1 + 24));
        v267 = v96;
        v295 = v95;
        v97 = sub_2F9C740((unsigned __int8 **)&v317, 0, (__int64)&v321, *(_QWORD *)(a1 + 24));
        v320 = 0;
        v264 = v97;
        v263 = v98;
        if ( *(_BYTE *)v317.m128i_i64[0] == 86
          && (unsigned __int8)sub_BC8C50(v317.m128i_i64[0], &v314, &v315)
          && (v157 = v314 + v315) != 0 )
        {
          v347 = v315;
          LOWORD(v348) = 0;
          v158 = sub_2F9DA20(v264, v263, (__int64)&v347);
          LOWORD(v326) = 0;
          v159 = v158;
          v161 = v160;
          v325 = v314;
          v162 = sub_2F9DA20(v295, v267, (__int64)&v325);
          v318 = v159;
          v312 = v163;
          v316 = v162;
          v313 = v161;
          v164 = sub_FDCA70(&v316, &v312, &v318, &v313);
          v165 = v316 + v318;
          if ( __CFADD__(v316, v318) )
          {
            ++v164;
            v165 = (v165 >> 1) | 0x8000000000000000LL;
          }
          if ( v164 > 0x3FFF )
          {
            v164 = 0x3FFF;
            v165 = -1;
          }
          *(_QWORD *)&v320 = v165;
          WORD4(v320) = v164;
          v347 = v157;
        }
        else
        {
          v347 = 3;
          LOWORD(v348) = 0;
          v318 = sub_2F9DA20(v264, v263, (__int64)&v347);
          v313 = v99;
          v325 = v295;
          LOWORD(v316) = v267;
          v100 = sub_FDCA70(&v318, &v313, &v325, (unsigned __int16 *)&v316);
          v101 = v318 + v325;
          if ( __CFADD__(v318, v325) )
          {
            ++v100;
            v101 = (v101 >> 1) | 0x8000000000000000LL;
          }
          v102 = v100;
          if ( v100 > 0x3FFF )
          {
            v102 = 0x3FFF;
            v100 = 0x3FFF;
            v101 = -1;
          }
          v260 = v102;
          v261 = v101;
          v325 = 3;
          LOWORD(v326) = 0;
          v316 = sub_2F9DA20(v295, v267, (__int64)&v325);
          v312 = v103;
          v318 = v264;
          v313 = v263;
          v104 = sub_FDCA70(&v316, &v312, &v318, &v313);
          v105 = v316 + v318;
          v106 = v104;
          if ( __CFADD__(v316, v318) )
          {
            v106 = v104 + 1;
            v105 = (v105 >> 1) | 0x8000000000000000LL;
          }
          v107 = v106;
          if ( v106 > 0x3FFF )
          {
            v107 = 0x3FFF;
            v106 = 0x3FFF;
            v105 = -1;
          }
          if ( (int)sub_D788E0(v105, v107, v261, v260) < 0 )
          {
            *(_QWORD *)&v320 = v261;
            WORD4(v320) = v100;
          }
          else
          {
            *(_QWORD *)&v320 = v105;
            WORD4(v320) = v106;
          }
          v347 = 4;
        }
        LOWORD(v348) = 0;
        sub_FDE760((__int64)&v320, (__int64)&v347);
        v108 = v320;
        v265 = WORD4(v320);
        v109 = (_BYTE *)*v262;
        if ( *(_BYTE *)*v262 > 0x1Cu && v324 )
        {
          v110 = (v324 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
          v111 = v322 + 40LL * v110;
          v112 = *(_BYTE **)v111;
          if ( v109 == *(_BYTE **)v111 )
          {
LABEL_129:
            if ( v111 != v322 + 40LL * v324 )
            {
              v113 = *(_WORD *)(v111 + 32);
              v114 = *(_QWORD *)(v111 + 24);
              v115 = v113;
LABEL_131:
              v268 = v113;
              v318 = v114;
              v116 = qword_5025768;
              v117 = *(unsigned int *)(a1 + 84);
              v296 = v114;
              v319 = v115;
              if ( (unsigned __int8)sub_2F9A6D0(a1, v317.m128i_i64[0]) )
                v116 = 0;
              v347 = v117;
              v325 = v116;
              LOWORD(v326) = 0;
              LOWORD(v348) = 0;
              v118 = (int)sub_D788E0(v117, 0, v296, v268) < 0;
              v119 = &v318;
              if ( !v118 )
                v119 = &v347;
              *(_QWORD *)&v120 = sub_2F9DA20(*v119, v119[1], (__int64)&v325);
              v347 = 100;
              v320 = v120;
              LOWORD(v348) = 0;
              sub_FDE760((__int64)&v320, (__int64)&v347);
              v325 = v108;
              LOWORD(v318) = v265;
              v347 = v320;
              LOWORD(v320) = WORD4(v320);
              v81 = sub_FDCA70(&v325, (unsigned __int16 *)&v318, &v347, (unsigned __int16 *)&v320);
              v82 = v325 + v347;
              if ( __CFADD__(v325, v347) )
              {
                ++v81;
                v82 = (v82 >> 1) | 0x8000000000000000LL;
              }
              if ( v81 > 0x3FFF )
              {
                v81 = 0x3FFF;
                v82 = -1;
              }
              goto LABEL_139;
            }
          }
          else
          {
            v145 = 1;
            while ( v112 != (_BYTE *)-4096LL )
            {
              v227 = v145 + 1;
              v110 = (v324 - 1) & (v145 + v110);
              v111 = v322 + 40LL * v110;
              v112 = *(_BYTE **)v111;
              if ( v109 == *(_BYTE **)v111 )
                goto LABEL_129;
              v145 = v227;
            }
          }
        }
        v113 = 0;
        v115 = 0;
        v114 = 0;
        goto LABEL_131;
      }
      v180 = 1;
      while ( v94 != -4096 )
      {
        v223 = v180 + 1;
        v92 = v90 & (v180 + v92);
        v93 = &v91[2 * v92];
        v94 = *v93;
        if ( v35 == *v93 )
          goto LABEL_114;
        v180 = v223;
      }
      if ( (v330 & 1) != 0 )
      {
        v175 = 4;
        goto LABEL_267;
      }
      v144 = v332;
LABEL_266:
      v175 = 2 * v144;
LABEL_267:
      v93 = &v91[v175];
      goto LABEL_114;
    }
LABEL_139:
    if ( !v324 )
    {
      ++v321;
      goto LABEL_270;
    }
    v121 = 1;
    v122 = 0;
    v123 = (v324 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v124 = (_QWORD *)(v322 + 40LL * v123);
    v125 = *v124;
    if ( v35 != *v124 )
    {
      while ( v125 != -4096 )
      {
        if ( v125 == -8192 && !v122 )
          v122 = v124;
        v123 = (v324 - 1) & (v121 + v123);
        v124 = (_QWORD *)(v322 + 40LL * v123);
        v125 = *v124;
        if ( v35 == *v124 )
          goto LABEL_141;
        ++v121;
      }
      if ( !v122 )
        v122 = v124;
      ++v321;
      v127 = v323 + 1;
      if ( 4 * ((int)v323 + 1) < 3 * v324 )
      {
        if ( v324 - HIDWORD(v323) - v127 <= v324 >> 3 )
        {
          sub_2F9ADF0((__int64)&v321, v324);
          if ( !v324 )
          {
LABEL_460:
            LODWORD(v323) = v323 + 1;
            BUG();
          }
          v199 = 1;
          v200 = 0;
          v201 = (v324 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v122 = (__int64 *)(v322 + 40LL * v201);
          v202 = *v122;
          v127 = v323 + 1;
          if ( v35 != *v122 )
          {
            while ( v202 != -4096 )
            {
              if ( v202 == -8192 && !v200 )
                v200 = v122;
              v201 = (v324 - 1) & (v199 + v201);
              v122 = (__int64 *)(v322 + 40LL * v201);
              v202 = *v122;
              if ( v35 == *v122 )
                goto LABEL_156;
              ++v199;
            }
            if ( v200 )
              v122 = v200;
          }
        }
        goto LABEL_156;
      }
LABEL_270:
      sub_2F9ADF0((__int64)&v321, 2 * v324);
      if ( !v324 )
        goto LABEL_460;
      v176 = (v324 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v122 = (__int64 *)(v322 + 40LL * v176);
      v177 = *v122;
      v127 = v323 + 1;
      if ( v35 != *v122 )
      {
        v178 = 1;
        v179 = 0;
        while ( v177 != -4096 )
        {
          if ( !v179 && v177 == -8192 )
            v179 = v122;
          v176 = (v324 - 1) & (v178 + v176);
          v122 = (__int64 *)(v322 + 40LL * v176);
          v177 = *v122;
          if ( v35 == *v122 )
            goto LABEL_156;
          ++v178;
        }
        if ( v179 )
          v122 = v179;
      }
LABEL_156:
      LODWORD(v323) = v127;
      if ( *v122 != -4096 )
        --HIDWORD(v323);
      *v122 = v35;
      v126 = (unsigned __int64 *)(v122 + 1);
      v122[1] = 0;
      *((_WORD *)v122 + 8) = 0;
      v122[3] = 0;
      *((_WORD *)v122 + 16) = 0;
      goto LABEL_142;
    }
LABEL_141:
    v126 = v124 + 1;
LABEL_142:
    v126[2] = v82;
    *((_WORD *)v126 + 12) = v81;
    *v126 = v278;
    *((_WORD *)v126 + 4) = v287;
    if ( (int)sub_D788E0(*v273, *((_WORD *)v273 + 4), v278, v287) < 0 )
    {
      *v273 = v278;
      *((_WORD *)v273 + 4) = v287;
    }
    if ( (int)sub_D788E0(v273[2], *((_WORD *)v273 + 12), v82, v81) < 0 )
    {
      v273[2] = v82;
      *((_WORD *)v273 + 12) = v81;
    }
LABEL_37:
    v303 = *(_QWORD *)(v303 + 8);
    if ( v270 == v303 )
      goto LABEL_222;
  }
  sub_B176B0((__int64)&v347, (__int64)"select-optimize", (__int64)"SelectOpti", 10, v35);
  sub_B18290(
    (__int64)&v347,
    "Invalid instruction cost preventing analysis and optimization of the inner-most loop containing this instruction. ",
    0x72u);
  sub_1049740(*(__int64 **)(a1 + 56), (__int64)&v347);
  v61 = v350;
  v347 = (unsigned __int64)&unk_49D9D40;
  v62 = &v350[10 * v351];
  if ( v350 != v62 )
  {
    do
    {
      v62 -= 10;
      v63 = v62[4];
      if ( (unsigned __int64 *)v63 != v62 + 6 )
        j_j___libc_free_0(v63);
      if ( (unsigned __int64 *)*v62 != v62 + 2 )
        j_j___libc_free_0(*v62);
    }
    while ( v61 != v62 );
    v62 = v350;
  }
  if ( v62 != (unsigned __int64 *)v352 )
    _libc_free((unsigned __int64)v62);
LABEL_74:
  if ( (v330 & 1) == 0 )
    sub_C7D6A0(v331, 16LL * v332, 8);
  if ( (v343 & 1) == 0 )
    sub_C7D6A0(v344, 24LL * v345, 8);
  if ( !v277 )
    return sub_C7D6A0(v322, 40LL * v324, 8);
  v64 = qword_5025688;
  if ( (_BYTE)qword_5025688 )
    goto LABEL_80;
  v181 = sub_AA4FF0(**(_QWORD **)(a2 + 32));
  if ( v181 )
    v181 -= 24;
  sub_B176B0((__int64)&v347, (__int64)"select-optimize", (__int64)"SelectOpti", 10, v181);
  v182 = v336;
  if ( !v336 )
  {
LABEL_343:
    v205 = v340;
    v206 = v338;
    if ( v340 )
    {
      if ( !v338 )
        goto LABEL_293;
      v207 = v341;
      v281 = v339;
      v298 = v339;
      v289 = v341;
      v306 = sub_D788C0(v340, v341);
      v208 = sub_D788C0(v206, v298);
      if ( v306 == v208 )
      {
        if ( v289 >= v281 )
          v209 = -(int)sub_F042F0(v206, v205, v207 - (unsigned __int8)v298);
        else
          v209 = sub_F042F0(v205, v206, (unsigned __int8)v298 - v207);
        if ( v209 >= 0 )
          goto LABEL_293;
      }
      else if ( v306 >= v208 )
      {
        goto LABEL_293;
      }
    }
    else if ( !v338 )
    {
      goto LABEL_293;
    }
    v210 = sub_2F9CA30(v334, v335, (__int64)&v336);
    v326 = v211;
    v325 = v210;
    v212 = sub_2F9CA30(v338, v339, (__int64)&v340);
    v213 = (unsigned int)qword_5025928;
    v327 = v212;
    v214 = v212;
    v328 = v215;
    if ( v212 )
    {
      if ( (_DWORD)qword_5025928 )
      {
        v216 = v328;
        v299 = v328;
        v307 = sub_D788C0(v212, v328);
        v217 = sub_D788C0(v213, 0);
        if ( v307 == v217 )
        {
          if ( v299 >= 0 )
            v247 = -(int)sub_F042F0(v213, v214, v216);
          else
            v247 = sub_F042F0(v214, v213, -v216);
          if ( v247 < 0 )
            goto LABEL_353;
        }
        else if ( v307 < v217 )
        {
          goto LABEL_353;
        }
      }
    }
    else if ( (_DWORD)qword_5025928 )
    {
      goto LABEL_353;
    }
    LOWORD(v343) = 0;
    v342 = (unsigned int)dword_5025848;
    v228 = sub_2F9DA20(v327, v328, (__int64)&v342);
    v230 = v338;
    v231 = v228;
    if ( v228 )
    {
      if ( !v338 )
        goto LABEL_387;
      v232 = v339;
      v290 = v229;
      v300 = v229;
      v282 = v339;
      v308 = sub_D788C0(v228, v229);
      v233 = sub_D788C0(v230, v232);
      if ( v308 != v233 )
      {
        if ( v308 >= v233 )
          goto LABEL_387;
LABEL_353:
        v218 = (__int8 *)sub_2F9DA20(0x64u, 0, (__int64)&v327);
        LOWORD(v343) = v219;
        v342 = (unsigned __int64)v218;
        v220 = sub_FDE760((__int64)&v342, (__int64)&v338);
        v221 = *(_QWORD *)v220;
        v222 = *(_WORD *)(v220 + 8);
        sub_B18290(
          (__int64)&v347,
          "No select conversion in the loop due to small reduction of loop's critical path. Gain=",
          0x56u);
        sub_F04320((__int64 *)&v342, v327, (__int16)v328, 64, 0xAu);
        sub_B18290((__int64)&v347, (__int8 *)v342, v343);
        sub_B18290((__int64)&v347, ", RelativeGain=", 0xFu);
        sub_F04320((__int64 *)&v329, v221, v222, 64, 0xAu);
        sub_B18290((__int64)&v347, v329, v330);
        sub_B18290((__int64)&v347, "%. ", 3u);
        if ( v329 != (__int8 *)&v331 )
          j_j___libc_free_0((unsigned __int64)v329);
        goto LABEL_355;
      }
      if ( v290 >= v282 )
        v248 = -(int)sub_F042F0(v230, v231, v300 - (unsigned __int8)v232);
      else
        v248 = sub_F042F0(v231, v230, (unsigned __int8)v232 - v300);
      if ( v248 < 0 )
        goto LABEL_353;
    }
    else if ( v338 )
    {
      goto LABEL_353;
    }
LABEL_387:
    v234 = v327;
    v235 = v325;
    if ( v327 )
    {
      if ( !v325 )
        goto LABEL_394;
      v236 = v328;
      v283 = v326;
      v301 = v326;
      v291 = v328;
      v309 = sub_D788C0(v327, v328);
      v237 = sub_D788C0(v235, v301);
      if ( v309 != v237 )
      {
        if ( v309 < v237 )
        {
LABEL_391:
          v184 = "No select conversion in the loop due to negative gradient gain. ";
          v185 = 64;
          goto LABEL_294;
        }
LABEL_394:
        v238 = (__int8 *)sub_2F9CA30(v338, v339, (__int64)&v334);
        v330 = v239;
        v329 = v238;
        *(_QWORD *)&v240 = sub_2F9CA30(v327, v328, (__int64)&v325);
        v320 = v240;
        v342 = sub_2F9DA20(0x64u, 0, (__int64)&v320);
        LOWORD(v343) = v241;
        v242 = sub_FDE760((__int64)&v342, (__int64)&v329);
        v243 = (unsigned int)dword_5025A08;
        v244 = *(_QWORD *)v242;
        v245 = *(__int16 *)(v242 + 8);
        if ( *(_QWORD *)v242 )
        {
          if ( dword_5025A08 )
          {
            v310 = sub_D788C0(v244, v245);
            v246 = sub_D788C0(v243, 0);
            if ( v310 == v246 )
            {
              if ( (v245 & 0x8000u) == 0 )
                v254 = -(int)sub_F042F0(v243, v244, v245);
              else
                v254 = sub_F042F0(v244, v243, -(char)v245);
              if ( v254 < 0 )
                goto LABEL_398;
            }
            else if ( v310 < v246 )
            {
LABEL_398:
              sub_B18290(
                (__int64)&v347,
                "No select conversion in the loop due to small gradient gain. GradientGain=",
                0x4Au);
              sub_F04320((__int64 *)&v342, v244, v245, 64, 0xAu);
              sub_B18290((__int64)&v347, (__int8 *)v342, v343);
              sub_B18290((__int64)&v347, "%. ", 3u);
LABEL_355:
              if ( (__int64 *)v342 != &v344 )
                j_j___libc_free_0(v342);
              goto LABEL_295;
            }
          }
        }
        else if ( dword_5025A08 )
        {
          goto LABEL_398;
        }
LABEL_410:
        v64 = v277;
        goto LABEL_296;
      }
      if ( v291 >= v283 )
        v249 = -(int)sub_F042F0(v235, v234, v236 - (unsigned __int8)v301);
      else
        v249 = sub_F042F0(v234, v235, (unsigned __int8)v301 - v236);
      if ( v249 > 0 )
        goto LABEL_394;
      v250 = v327;
      v235 = v325;
      if ( v327 )
      {
        if ( v325 )
        {
          v251 = v328;
          v284 = v326;
          v302 = v326;
          v292 = v328;
          v311 = sub_D788C0(v327, v328);
          v252 = sub_D788C0(v235, v302);
          if ( v311 == v252 )
          {
            if ( v292 >= v284 )
              v253 = -(int)sub_F042F0(v235, v250, v251 - (unsigned __int8)v302);
            else
              v253 = sub_F042F0(v250, v235, (unsigned __int8)v302 - v251);
            if ( v253 < 0 )
              goto LABEL_391;
          }
          else if ( v311 < v252 )
          {
            goto LABEL_391;
          }
        }
        goto LABEL_410;
      }
    }
    if ( v235 )
      goto LABEL_391;
    goto LABEL_410;
  }
  v183 = v334;
  if ( !v334 )
    goto LABEL_293;
  v203 = v335;
  v288 = v337;
  v297 = v337;
  v280 = v335;
  v305 = sub_D788C0(v336, v337);
  v204 = sub_D788C0(v183, v203);
  if ( v305 == v204 )
  {
    if ( v288 < v280 )
      v226 = sub_F042F0(v182, v183, (unsigned __int8)v203 - v297);
    else
      v226 = -(int)sub_F042F0(v183, v182, v297 - (unsigned __int8)v203);
    if ( v226 > 0 )
      goto LABEL_293;
    goto LABEL_343;
  }
  if ( v305 < v204 )
    goto LABEL_343;
LABEL_293:
  v184 = "No select conversion in the loop due to no reduction of loop's critical path. ";
  v185 = 78;
LABEL_294:
  sub_B18290((__int64)&v347, v184, v185);
LABEL_295:
  sub_1049740(*(__int64 **)(a1 + 56), (__int64)&v347);
LABEL_296:
  v186 = v350;
  v347 = (unsigned __int64)&unk_49D9D40;
  v187 = &v350[10 * v351];
  if ( v350 != v187 )
  {
    do
    {
      v187 -= 10;
      v188 = v187[4];
      if ( (unsigned __int64 *)v188 != v187 + 6 )
        j_j___libc_free_0(v188);
      if ( (unsigned __int64 *)*v187 != v187 + 2 )
        j_j___libc_free_0(*v187);
    }
    while ( v186 != v187 );
    v187 = v350;
  }
  if ( v187 != (unsigned __int64 *)v352 )
    _libc_free((unsigned __int64)v187);
  if ( v64 )
  {
LABEL_80:
    v286 = *(_QWORD *)a3;
    v279 = *(_QWORD *)a3 + 56LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v279 )
    {
      do
      {
        v65 = *(__int64 **)(v286 + 8);
        v294 = &v65[2 * *(unsigned int *)(v286 + 16)];
        if ( v294 != v65 )
        {
          v66 = 0;
          v67 = 0;
          v68 = 0;
          v304 = 0;
          while ( 1 )
          {
            v77 = *v65;
            if ( !v324 )
              break;
            v69 = 1;
            v70 = (v324 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
            v71 = (__int64 *)(v322 + 40LL * v70);
            v72 = 0;
            v73 = *v71;
            if ( v77 != *v71 )
            {
              while ( v73 != -4096 )
              {
                if ( v73 == -8192 && !v72 )
                  v72 = v71;
                v70 = (v324 - 1) & (v69 + v70);
                v71 = (__int64 *)(v322 + 40LL * v70);
                v73 = *v71;
                if ( v77 == *v71 )
                  goto LABEL_84;
                ++v69;
              }
              if ( !v72 )
                v72 = v71;
              ++v321;
              v79 = v323 + 1;
              if ( 4 * ((int)v323 + 1) < 3 * v324 )
              {
                if ( v324 - HIDWORD(v323) - v79 <= v324 >> 3 )
                {
                  sub_2F9ADF0((__int64)&v321, v324);
                  if ( !v324 )
                  {
LABEL_463:
                    LODWORD(v323) = v323 + 1;
                    BUG();
                  }
                  v189 = 0;
                  v190 = 1;
                  LODWORD(v191) = (v324 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
                  v79 = v323 + 1;
                  v72 = (__int64 *)(v322 + 40LL * (unsigned int)v191);
                  v192 = *v72;
                  if ( v77 != *v72 )
                  {
                    while ( v192 != -4096 )
                    {
                      if ( v192 == -8192 && !v189 )
                        v189 = v72;
                      v191 = (v324 - 1) & ((_DWORD)v191 + v190);
                      v72 = (__int64 *)(v322 + 40 * v191);
                      v192 = *v72;
                      if ( v77 == *v72 )
                        goto LABEL_94;
                      ++v190;
                    }
LABEL_320:
                    if ( v189 )
                      v72 = v189;
                  }
                }
LABEL_94:
                LODWORD(v323) = v79;
                if ( *v72 != -4096 )
                  --HIDWORD(v323);
                *v72 = v77;
                v75 = 0;
                *((_WORD *)v72 + 8) = 0;
                v76 = v72 + 1;
                v74 = 0;
                v72[1] = 0;
                v72[3] = 0;
                *((_WORD *)v72 + 16) = 0;
                goto LABEL_85;
              }
LABEL_92:
              sub_2F9ADF0((__int64)&v321, 2 * v324);
              if ( !v324 )
                goto LABEL_463;
              LODWORD(v78) = (v324 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
              v79 = v323 + 1;
              v72 = (__int64 *)(v322 + 40LL * (unsigned int)v78);
              v80 = *v72;
              if ( v77 != *v72 )
              {
                v225 = 1;
                v189 = 0;
                while ( v80 != -4096 )
                {
                  if ( v80 == -8192 && !v189 )
                    v189 = v72;
                  v78 = (v324 - 1) & ((_DWORD)v78 + v225);
                  v72 = (__int64 *)(v322 + 40 * v78);
                  v80 = *v72;
                  if ( v77 == *v72 )
                    goto LABEL_94;
                  ++v225;
                }
                goto LABEL_320;
              }
              goto LABEL_94;
            }
LABEL_84:
            v74 = v71[1];
            v75 = *((_WORD *)v71 + 8);
            v76 = v71 + 1;
LABEL_85:
            if ( (int)sub_D788E0(v68, v67, v74, v75) < 0 )
            {
              v68 = *v76;
              v67 = *((_WORD *)v76 + 4);
            }
            if ( (int)sub_D788E0(v66, v304, v76[2], *((_WORD *)v76 + 12)) < 0 )
            {
              v66 = v76[2];
              v304 = *((_WORD *)v76 + 12);
            }
            v65 += 2;
            if ( v294 == v65 )
            {
              v166 = v304;
              v167 = v67;
              goto LABEL_245;
            }
          }
          ++v321;
          goto LABEL_92;
        }
        v166 = 0;
        v167 = 0;
        v66 = 0;
        v68 = 0;
LABEL_245:
        v168 = sub_D788E0(v66, v166, v68, v167);
        v169 = **(_QWORD **)(v286 + 8);
        if ( v168 < 0 )
        {
          sub_B174A0((__int64)&v347, (__int64)"select-optimize", (__int64)"SelectOpti", 10, v169);
          sub_B18290((__int64)&v347, "Profitable to convert to branch (loop analysis). BranchCost=", 0x3Cu);
          sub_F04320((__int64 *)&v329, v66, v166, 64, 0xAu);
          sub_B18290((__int64)&v347, v329, v330);
          sub_B18290((__int64)&v347, ", SelectCost=", 0xDu);
          sub_F04320((__int64 *)&v342, v68, v167, 64, 0xAu);
          sub_B18290((__int64)&v347, (__int8 *)v342, v343);
          sub_B18290((__int64)&v347, ". ", 2u);
          if ( (__int64 *)v342 != &v344 )
            j_j___libc_free_0(v342);
          if ( v329 != (__int8 *)&v331 )
            j_j___libc_free_0((unsigned __int64)v329);
          sub_1049740(*(__int64 **)(a1 + 56), (__int64)&v347);
          sub_2F9A860(a4, v286, v193, v194, v195, v196);
          v197 = v350;
          v347 = (unsigned __int64)&unk_49D9D40;
          v171 = &v350[10 * v351];
          if ( v350 == v171 )
            goto LABEL_257;
          do
          {
            v171 -= 10;
            v198 = v171[4];
            if ( (unsigned __int64 *)v198 != v171 + 6 )
              j_j___libc_free_0(v198);
            if ( (unsigned __int64 *)*v171 != v171 + 2 )
              j_j___libc_free_0(*v171);
          }
          while ( v197 != v171 );
        }
        else
        {
          sub_B176B0((__int64)&v347, (__int64)"select-optimize", (__int64)"SelectOpti", 10, v169);
          sub_B18290((__int64)&v347, "Select is more profitable (loop analysis). BranchCost=", 0x36u);
          sub_F04320((__int64 *)&v329, v66, v166, 64, 0xAu);
          sub_B18290((__int64)&v347, v329, v330);
          sub_B18290((__int64)&v347, ", SelectCost=", 0xDu);
          sub_F04320((__int64 *)&v342, v68, v167, 64, 0xAu);
          sub_B18290((__int64)&v347, (__int8 *)v342, v343);
          sub_B18290((__int64)&v347, ". ", 2u);
          if ( (__int64 *)v342 != &v344 )
            j_j___libc_free_0(v342);
          if ( v329 != (__int8 *)&v331 )
            j_j___libc_free_0((unsigned __int64)v329);
          sub_1049740(*(__int64 **)(a1 + 56), (__int64)&v347);
          v170 = v350;
          v347 = (unsigned __int64)&unk_49D9D40;
          v171 = &v350[10 * v351];
          if ( v350 == v171 )
            goto LABEL_257;
          do
          {
            v171 -= 10;
            v172 = v171[4];
            if ( (unsigned __int64 *)v172 != v171 + 6 )
              j_j___libc_free_0(v172);
            if ( (unsigned __int64 *)*v171 != v171 + 2 )
              j_j___libc_free_0(*v171);
          }
          while ( v170 != v171 );
        }
        v171 = v350;
LABEL_257:
        if ( v171 != (unsigned __int64 *)v352 )
          _libc_free((unsigned __int64)v171);
        v286 += 56LL;
      }
      while ( v279 != v286 );
    }
  }
  return sub_C7D6A0(v322, 40LL * v324, 8);
}
