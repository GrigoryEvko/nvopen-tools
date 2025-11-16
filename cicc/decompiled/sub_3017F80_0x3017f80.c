// Function: sub_3017F80
// Address: 0x3017f80
//
__int64 **__fastcall sub_3017F80(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 **result; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  char v11; // cl
  __int64 *v12; // r14
  __int64 v13; // r10
  int v14; // r8d
  __int64 *v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned int v22; // esi
  int v23; // r9d
  int v24; // r9d
  __int64 v25; // r10
  int v26; // ecx
  unsigned int v27; // esi
  __int64 v28; // r8
  const char *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  char *v33; // r12
  __int64 v34; // rax
  _QWORD *v35; // r9
  int v36; // ecx
  unsigned int v37; // edx
  _QWORD *v38; // rbx
  __int64 v39; // rsi
  int v40; // edx
  __int64 v41; // rdx
  __int64 *v42; // rdx
  _QWORD *v43; // rbx
  char *v44; // rax
  const __m128i *v45; // rsi
  const __m128i *v46; // r13
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 *v49; // rsi
  unsigned int v50; // esi
  __int64 v51; // rax
  __int64 v52; // r9
  int v53; // edi
  __int64 *v54; // rcx
  unsigned int v55; // edx
  __int64 *v56; // r12
  const char *v57; // r8
  __int64 v58; // rax
  unsigned __int64 *v59; // rdx
  unsigned __int64 v60; // r8
  __int64 v61; // rax
  __int64 *v62; // rdx
  __int64 *v63; // rcx
  __int64 *v64; // r12
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 *v68; // rax
  __int64 *v69; // rax
  __int64 *v70; // rax
  __int64 *v71; // rcx
  unsigned int v72; // esi
  __int64 v73; // r9
  int v74; // edx
  __int64 *v75; // rax
  unsigned int v76; // r8d
  __int64 *v77; // rdi
  __int64 v78; // rcx
  __int64 v79; // r15
  __int64 j; // r12
  unsigned __int8 *v81; // r13
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  const __m128i *v86; // r13
  __int64 **v87; // r10
  __int64 v88; // r8
  const __m128i *v89; // r15
  __int64 v90; // rax
  __int64 v91; // r12
  __int64 v92; // rbx
  __int64 v93; // rdx
  __int64 v94; // rdx
  unsigned __int64 v95; // rax
  __int64 v96; // r9
  __int64 *v97; // rdx
  __int64 *v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rcx
  __int64 v101; // rcx
  __int64 v102; // r11
  __int64 v103; // rax
  __int64 v104; // r11
  __int64 v105; // rdx
  __int64 v106; // r13
  __int64 v107; // r14
  unsigned int v108; // ebx
  int v109; // r15d
  bool v110; // al
  __int64 v111; // r12
  unsigned __int64 v112; // rax
  unsigned int v113; // esi
  __int64 v114; // r9
  int v115; // r11d
  _QWORD *v116; // rcx
  unsigned int v117; // r8d
  _QWORD *v118; // rax
  __int64 v119; // rdi
  __int64 v120; // rdx
  _QWORD **v121; // rax
  __int64 v122; // rdx
  __int64 v123; // r8
  int v124; // edi
  _QWORD *v125; // rcx
  int v126; // ecx
  unsigned int v127; // edx
  __int64 v128; // rsi
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // r13
  unsigned int v134; // ebx
  int v135; // r15d
  bool v136; // al
  __int64 v137; // r12
  unsigned __int64 v138; // rax
  unsigned int v139; // esi
  __int64 v140; // r9
  int v141; // r11d
  _QWORD *v142; // rcx
  unsigned int v143; // r8d
  _QWORD *v144; // rax
  __int64 v145; // rdi
  __int64 v146; // rdx
  _QWORD **v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // r12
  _QWORD *v151; // rdx
  unsigned __int64 v152; // rax
  unsigned int m; // r15d
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rdx
  __int64 v157; // r14
  __int64 v158; // rbx
  int v159; // eax
  __int64 v160; // rcx
  __int64 v161; // rdx
  __int64 v162; // rdi
  __int64 v163; // r13
  unsigned int v164; // r8d
  _QWORD *v165; // rsi
  __int64 v166; // r10
  int v167; // eax
  __int64 v168; // rcx
  __int64 v169; // rax
  __int64 v170; // rax
  __int64 v171; // rax
  __int64 v172; // rax
  _QWORD *v173; // r14
  __int64 v174; // rax
  _QWORD *v175; // rbx
  __int64 v176; // rdx
  __int64 v177; // rax
  __int64 v178; // rax
  int v179; // esi
  int v180; // eax
  int v181; // eax
  int v182; // eax
  int v183; // eax
  int v184; // r11d
  int v185; // r11d
  __int64 v186; // r9
  unsigned int v187; // edx
  __int64 v188; // r8
  int v189; // edi
  _QWORD *v190; // rsi
  int v191; // r11d
  int v192; // r11d
  __int64 v193; // r9
  unsigned int v194; // edx
  __int64 v195; // r8
  int v196; // edi
  _QWORD *v197; // rsi
  int v198; // r11d
  int v199; // r11d
  __int64 v200; // r9
  int v201; // edi
  unsigned int v202; // edx
  __int64 v203; // r8
  int v204; // r11d
  int v205; // r11d
  __int64 v206; // r9
  int v207; // edi
  unsigned int v208; // edx
  __int64 v209; // r8
  __int64 v210; // rax
  unsigned __int64 v211; // rdx
  int v212; // edi
  int v213; // edx
  int v214; // r9d
  int v215; // r9d
  __int64 v216; // r10
  __int64 v217; // rdx
  int v218; // eax
  __int64 v219; // r8
  int v220; // esi
  __int64 *v221; // rcx
  int v222; // r8d
  int v223; // r8d
  __int64 v224; // r10
  unsigned int v225; // ecx
  __int64 v226; // r9
  int v227; // edi
  __int64 *v228; // rsi
  __int64 v229; // rax
  __int64 v230; // r8
  __int64 v231; // rdx
  unsigned __int64 v232; // rax
  __int64 v233; // rdx
  _QWORD *v234; // r14
  __int64 v235; // rax
  __int64 *v236; // rax
  __int64 v237; // r11
  __int64 v238; // rbx
  __int64 v239; // r8
  __int64 v240; // rdi
  int v241; // r10d
  __int64 v242; // r9
  unsigned int v243; // ecx
  __int64 *v244; // rax
  __int64 v245; // rdx
  _QWORD *v246; // rcx
  __int64 v247; // rax
  __int64 v248; // rdx
  unsigned __int64 v249; // rax
  __int64 v250; // rax
  unsigned __int64 v251; // rdx
  unsigned int v252; // esi
  __int64 v253; // r12
  int v254; // r9d
  __int64 v255; // r10
  __int64 v256; // rcx
  int v257; // edx
  __int64 v258; // rax
  int v259; // edi
  int v260; // edi
  int v261; // edi
  int v262; // esi
  __int64 v263; // r13
  __int64 *v264; // rcx
  unsigned __int8 *v265; // rax
  size_t v266; // rdx
  int k; // eax
  __int64 v268; // rsi
  __int64 v269; // rax
  _QWORD *v270; // r12
  _QWORD *v271; // rbx
  __int64 v272; // rsi
  int v273; // eax
  int v274; // r8d
  int v275; // r8d
  __int64 v276; // r9
  __int64 *v277; // rdx
  __int64 v278; // r12
  int v279; // ecx
  __int64 v280; // rsi
  _QWORD *v281; // r14
  __int64 v282; // rax
  _QWORD *v283; // rbx
  __int64 v284; // rdx
  __int64 v285; // rax
  int v286; // r9d
  int v287; // r9d
  __int64 v288; // r10
  int v289; // edi
  unsigned int v290; // ecx
  __int64 v291; // r8
  int v292; // edi
  int v293; // r9d
  int v294; // r9d
  __int64 v295; // r10
  __int64 *v296; // r11
  int v297; // edi
  unsigned int v298; // esi
  __int64 v299; // r8
  __int64 *v300; // rax
  __int64 v301; // rax
  _QWORD *v302; // r12
  _QWORD *v303; // rbx
  __int64 v304; // rsi
  int v305; // edi
  __int64 *v306; // rsi
  int v307; // edi
  __int64 v308; // [rsp+0h] [rbp-220h]
  __int64 v309; // [rsp+8h] [rbp-218h]
  __int64 v310; // [rsp+8h] [rbp-218h]
  int v311; // [rsp+8h] [rbp-218h]
  unsigned int v312; // [rsp+8h] [rbp-218h]
  __int64 v313; // [rsp+8h] [rbp-218h]
  const __m128i *v314; // [rsp+10h] [rbp-210h]
  const __m128i *v315; // [rsp+10h] [rbp-210h]
  __int64 **v316; // [rsp+18h] [rbp-208h]
  __int64 v318; // [rsp+30h] [rbp-1F0h]
  __int64 v319; // [rsp+30h] [rbp-1F0h]
  const __m128i *v320; // [rsp+30h] [rbp-1F0h]
  unsigned int v321; // [rsp+30h] [rbp-1F0h]
  __int64 v322; // [rsp+30h] [rbp-1F0h]
  unsigned __int64 v323; // [rsp+30h] [rbp-1F0h]
  __int64 **v325; // [rsp+40h] [rbp-1E0h]
  __int64 v326; // [rsp+48h] [rbp-1D8h]
  __int64 *v327; // [rsp+50h] [rbp-1D0h]
  __int64 *v328; // [rsp+50h] [rbp-1D0h]
  const __m128i *v329; // [rsp+50h] [rbp-1D0h]
  __int64 v330; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v331; // [rsp+50h] [rbp-1D0h]
  __int64 v332; // [rsp+50h] [rbp-1D0h]
  __int64 *v333; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v334; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v335; // [rsp+50h] [rbp-1D0h]
  __int64 v336; // [rsp+58h] [rbp-1C8h]
  const __m128i *v337; // [rsp+58h] [rbp-1C8h]
  __int64 *v338; // [rsp+58h] [rbp-1C8h]
  __int64 v339; // [rsp+58h] [rbp-1C8h]
  __int64 v340; // [rsp+58h] [rbp-1C8h]
  int v341; // [rsp+58h] [rbp-1C8h]
  __int64 **v342; // [rsp+58h] [rbp-1C8h]
  __int64 v343; // [rsp+60h] [rbp-1C0h]
  __int64 v344; // [rsp+68h] [rbp-1B8h]
  __int64 v345; // [rsp+68h] [rbp-1B8h]
  _QWORD *v346; // [rsp+68h] [rbp-1B8h]
  const __m128i *v347; // [rsp+70h] [rbp-1B0h] BYREF
  const __m128i *v348; // [rsp+78h] [rbp-1A8h]
  const __m128i *v349; // [rsp+80h] [rbp-1A0h]
  __int64 *v350; // [rsp+90h] [rbp-190h] BYREF
  __int64 v351; // [rsp+98h] [rbp-188h]
  _BYTE v352[16]; // [rsp+A0h] [rbp-180h] BYREF
  __int64 *v353; // [rsp+B0h] [rbp-170h] BYREF
  unsigned __int64 v354[2]; // [rsp+B8h] [rbp-168h] BYREF
  __int64 v355; // [rsp+C8h] [rbp-158h]
  char *v356; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v357; // [rsp+D8h] [rbp-148h] BYREF
  __int64 v358; // [rsp+E0h] [rbp-140h]
  __int64 v359; // [rsp+E8h] [rbp-138h]
  __int64 v360; // [rsp+F0h] [rbp-130h]
  __int64 v361; // [rsp+110h] [rbp-110h] BYREF
  _QWORD *v362; // [rsp+118h] [rbp-108h]
  __int64 v363; // [rsp+120h] [rbp-100h]
  unsigned int v364; // [rsp+128h] [rbp-F8h]
  _QWORD *v365; // [rsp+138h] [rbp-E8h]
  unsigned int v366; // [rsp+148h] [rbp-D8h]
  char v367; // [rsp+150h] [rbp-D0h]
  const char *v368; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v369; // [rsp+168h] [rbp-B8h] BYREF
  const char *v370; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v371; // [rsp+178h] [rbp-A8h]
  __int64 *i; // [rsp+180h] [rbp-A0h]

  v2 = a1;
  result = *(__int64 ***)(a1 + 80);
  v4 = 4LL * *(unsigned int *)(a1 + 88);
  v316 = &result[v4];
  if ( &result[v4] == result )
    return result;
  ++result;
  do
  {
    v325 = result;
    v5 = (__int64)*(result - 1);
    v6 = *(_QWORD *)(a2 + 80);
    v343 = v5;
    if ( v6 )
      v6 -= 24;
    if ( v5 == v6 )
    {
      v300 = (__int64 *)sub_B2BE50(a2);
      v344 = sub_AC3540(v300);
    }
    else
    {
      v7 = sub_AA4FF0(v5);
      v8 = v7 - 24;
      if ( !v7 )
        v8 = 0;
      v344 = v8;
    }
    v347 = 0;
    v348 = 0;
    v349 = 0;
    v361 = 0;
    v364 = 128;
    v9 = (_QWORD *)sub_C7D670(0x2000, 8);
    v363 = 0;
    v362 = v9;
    v369 = 2;
    v10 = &v9[8 * (unsigned __int64)v364];
    v368 = (const char *)&unk_49DD7B0;
    v370 = 0;
    v371 = -4096;
    for ( i = 0; v10 != v9; v9 += 8 )
    {
      if ( v9 )
      {
        v11 = v369;
        v9[2] = 0;
        v9[3] = -4096;
        *v9 = &unk_49DD7B0;
        v9[1] = v11 & 6;
        v9[4] = i;
      }
    }
    v367 = 0;
    v336 = v2 + 16;
    v12 = *v325;
    v327 = v325[1];
    if ( *v325 != v327 )
    {
      while ( 1 )
      {
        v21 = *v12;
        v22 = *(_DWORD *)(v2 + 40);
        v353 = (__int64 *)*v12;
        if ( !v22 )
          break;
        v13 = *(_QWORD *)(v2 + 24);
        v14 = 1;
        v15 = 0;
        v16 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v17 = (__int64 *)(v13 + 16LL * v16);
        v18 = *v17;
        if ( v21 != *v17 )
        {
          while ( v18 != -4096 )
          {
            if ( v18 == -8192 && !v15 )
              v15 = v17;
            v16 = (v22 - 1) & (v14 + v16);
            v17 = (__int64 *)(v13 + 16LL * v16);
            v18 = *v17;
            if ( v21 == *v17 )
              goto LABEL_16;
            ++v14;
          }
          if ( v15 )
            v17 = v15;
          v292 = *(_DWORD *)(v2 + 32);
          ++*(_QWORD *)(v2 + 16);
          v26 = v292 + 1;
          if ( 4 * (v292 + 1) < 3 * v22 )
          {
            if ( v22 - *(_DWORD *)(v2 + 36) - v26 <= v22 >> 3 )
            {
              sub_B2ACE0(v336, v22);
              v293 = *(_DWORD *)(v2 + 40);
              if ( !v293 )
              {
LABEL_600:
                ++*(_DWORD *)(a1 + 32);
                BUG();
              }
              v294 = v293 - 1;
              v295 = *(_QWORD *)(v2 + 24);
              v296 = 0;
              v21 = (__int64)v353;
              v26 = *(_DWORD *)(v2 + 32) + 1;
              v297 = 1;
              v298 = v294 & (((unsigned int)v353 >> 9) ^ ((unsigned int)v353 >> 4));
              v17 = (__int64 *)(v295 + 16LL * v298);
              v299 = *v17;
              if ( v353 != (__int64 *)*v17 )
              {
                while ( v299 != -4096 )
                {
                  if ( !v296 && v299 == -8192 )
                    v296 = v17;
                  v298 = v294 & (v297 + v298);
                  v17 = (__int64 *)(v295 + 16LL * v298);
                  v299 = *v17;
                  if ( v353 == (__int64 *)*v17 )
                    goto LABEL_26;
                  ++v297;
                }
LABEL_520:
                if ( v296 )
                  v17 = v296;
              }
            }
LABEL_26:
            *(_DWORD *)(v2 + 32) = v26;
            if ( *v17 != -4096 )
              --*(_DWORD *)(v2 + 36);
            *v17 = v21;
            v17[1] = 0;
            goto LABEL_29;
          }
LABEL_24:
          sub_B2ACE0(v336, 2 * v22);
          v23 = *(_DWORD *)(v2 + 40);
          if ( !v23 )
            goto LABEL_600;
          v21 = (__int64)v353;
          v24 = v23 - 1;
          v25 = *(_QWORD *)(v2 + 24);
          v26 = *(_DWORD *)(v2 + 32) + 1;
          v27 = v24 & (((unsigned int)v353 >> 9) ^ ((unsigned int)v353 >> 4));
          v17 = (__int64 *)(v25 + 16LL * v27);
          v28 = *v17;
          if ( v353 != (__int64 *)*v17 )
          {
            v307 = 1;
            v296 = 0;
            while ( v28 != -4096 )
            {
              if ( !v296 && v28 == -8192 )
                v296 = v17;
              v27 = v24 & (v307 + v27);
              v17 = (__int64 *)(v25 + 16LL * v27);
              v28 = *v17;
              if ( v353 == (__int64 *)*v17 )
                goto LABEL_26;
              ++v307;
            }
            goto LABEL_520;
          }
          goto LABEL_26;
        }
LABEL_16:
        v19 = v17[1];
        v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0
          && ((v19 & 4) == 0 || *(_DWORD *)(v20 + 8))
          && (((v19 >> 2) & 1) == 0 || *(_DWORD *)(v20 + 8) == 1) )
        {
          goto LABEL_21;
        }
LABEL_29:
        v29 = sub_BD5D20(v343);
        LOWORD(i) = 1283;
        v368 = ".for.";
        v371 = v30;
        v370 = v29;
        v356 = (char *)sub_F4B360((__int64)v353, (__int64)&v361, (__int64 *)&v368, 0, 0);
        v31 = v353[4];
        if ( v31 == v353[9] + 72 || !v31 )
          v32 = 0;
        else
          v32 = v31 - 24;
        sub_AA4C60((__int64)v356, a2, v32);
        v369 = 2;
        v370 = 0;
        v33 = v356;
        v371 = (__int64)v353;
        if ( v353 != 0 && v353 + 512 != 0 && v353 != (__int64 *)-8192LL )
          sub_BD73F0((__int64)&v369);
        i = &v361;
        v368 = (const char *)&unk_49DD7B0;
        if ( !v364 )
        {
          ++v361;
          goto LABEL_37;
        }
        v34 = v371;
        v122 = (v364 - 1) & (((unsigned int)v371 >> 9) ^ ((unsigned int)v371 >> 4));
        v38 = &v362[8 * v122];
        v123 = v38[3];
        if ( v371 != v123 )
        {
          v124 = 1;
          v125 = 0;
          while ( v123 != -4096 )
          {
            if ( v123 == -8192 && !v125 )
              v125 = v38;
            LODWORD(v122) = (v364 - 1) & (v124 + v122);
            v38 = &v362[8 * (unsigned __int64)(unsigned int)v122];
            v123 = v38[3];
            if ( v371 == v123 )
              goto LABEL_50;
            ++v124;
          }
          if ( v125 )
            v38 = v125;
          ++v361;
          v40 = v363 + 1;
          if ( 4 * ((int)v363 + 1) >= 3 * v364 )
          {
LABEL_37:
            sub_CF32C0((__int64)&v361, 2 * v364);
            if ( !v364 )
              goto LABEL_506;
            v34 = v371;
            v35 = 0;
            v36 = 1;
            v37 = (v364 - 1) & (((unsigned int)v371 >> 9) ^ ((unsigned int)v371 >> 4));
            v38 = &v362[8 * (unsigned __int64)v37];
            v39 = v38[3];
            if ( v371 != v39 )
            {
              while ( v39 != -4096 )
              {
                if ( v39 == -8192 && !v35 )
                  v35 = v38;
                v37 = (v364 - 1) & (v36 + v37);
                v38 = &v362[8 * (unsigned __int64)v37];
                v39 = v38[3];
                if ( v371 == v39 )
                  goto LABEL_39;
                ++v36;
              }
              goto LABEL_566;
            }
LABEL_39:
            v40 = v363 + 1;
          }
          else if ( v364 - HIDWORD(v363) - v40 <= v364 >> 3 )
          {
            sub_CF32C0((__int64)&v361, v364);
            if ( v364 )
            {
              v34 = v371;
              v35 = 0;
              v126 = 1;
              v127 = (v364 - 1) & (((unsigned int)v371 >> 9) ^ ((unsigned int)v371 >> 4));
              v38 = &v362[8 * (unsigned __int64)v127];
              v128 = v38[3];
              if ( v371 != v128 )
              {
                while ( v128 != -4096 )
                {
                  if ( v128 == -8192 && !v35 )
                    v35 = v38;
                  v127 = (v364 - 1) & (v126 + v127);
                  v38 = &v362[8 * (unsigned __int64)v127];
                  v128 = v38[3];
                  if ( v371 == v128 )
                    goto LABEL_39;
                  ++v126;
                }
LABEL_566:
                if ( v35 )
                  v38 = v35;
              }
              goto LABEL_39;
            }
LABEL_506:
            v34 = v371;
            v38 = 0;
            goto LABEL_39;
          }
          LODWORD(v363) = v40;
          v41 = v38[3];
          if ( v41 == -4096 )
          {
            if ( v34 != -4096 )
              goto LABEL_45;
          }
          else
          {
            --HIDWORD(v363);
            if ( v41 != v34 )
            {
              if ( v41 != -8192 && v41 )
              {
                sub_BD60C0(v38 + 1);
                v34 = v371;
              }
LABEL_45:
              v38[3] = v34;
              if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
                sub_BD6050(v38 + 1, v369 & 0xFFFFFFFFFFFFFFF8LL);
              v34 = v371;
            }
          }
          v42 = i;
          v38[5] = 6;
          v38[6] = 0;
          v38[4] = v42;
          v38[7] = 0;
        }
LABEL_50:
        v43 = v38 + 5;
        v368 = (const char *)&unk_49DB368;
        if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
          sub_BD60C0(&v369);
        v44 = (char *)v43[2];
        if ( v33 != v44 )
        {
          if ( v44 != 0 && v44 + 4096 != 0 && v44 != (char *)-8192LL )
            sub_BD60C0(v43);
          v43[2] = v33;
          if ( v33 + 4096 != 0 && v33 != 0 && v33 != (char *)-8192LL )
            sub_BD73F0((__int64)v43);
        }
        v45 = v348;
        if ( v348 == v349 )
        {
          sub_3012990((unsigned __int64 *)&v347, v348, &v353, &v356);
LABEL_21:
          if ( v327 == ++v12 )
            goto LABEL_64;
        }
        else
        {
          if ( v348 )
          {
            v348->m128i_i64[0] = (__int64)v353;
            v45->m128i_i64[1] = (__int64)v356;
            v45 = v348;
          }
          ++v12;
          v348 = v45 + 1;
          if ( v327 == v12 )
            goto LABEL_64;
        }
      }
      ++*(_QWORD *)(v2 + 16);
      goto LABEL_24;
    }
LABEL_64:
    v46 = v347;
    v337 = v348;
    if ( v347 != v348 )
    {
      v326 = v2 + 16;
      while ( 1 )
      {
        v47 = v46->m128i_i64[1];
        v48 = v46->m128i_i64[0];
        v368 = (const char *)v47;
        v49 = v325[1];
        if ( v49 == v325[2] )
        {
          sub_9319A0((__int64)v325, v49, &v368);
          v50 = *(_DWORD *)(v2 + 40);
          if ( !v50 )
            goto LABEL_362;
        }
        else
        {
          if ( v49 )
          {
            *v49 = v47;
            v49 = v325[1];
          }
          v325[1] = v49 + 1;
          v50 = *(_DWORD *)(v2 + 40);
          if ( !v50 )
          {
LABEL_362:
            ++*(_QWORD *)(v2 + 16);
            goto LABEL_363;
          }
        }
        v51 = (__int64)v368;
        v52 = *(_QWORD *)(v2 + 24);
        v53 = 1;
        v54 = 0;
        v55 = (v50 - 1) & (((unsigned int)v368 >> 9) ^ ((unsigned int)v368 >> 4));
        v56 = (__int64 *)(v52 + 16LL * v55);
        v57 = (const char *)*v56;
        if ( v368 != (const char *)*v56 )
          break;
LABEL_71:
        v58 = v56[1];
        v59 = (unsigned __int64 *)(v56 + 1);
        v60 = v58 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v58 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (v58 & 4) == 0 )
          {
            v331 = v56[1] & 0xFFFFFFFFFFFFFFF8LL;
            v229 = sub_22077B0(0x30u);
            v230 = v331;
            if ( v229 )
            {
              *(_QWORD *)v229 = v229 + 16;
              *(_QWORD *)(v229 + 8) = 0x400000000LL;
            }
            v231 = v229;
            v232 = v229 & 0xFFFFFFFFFFFFFFF8LL;
            v56[1] = v231 | 4;
            v233 = *(unsigned int *)(v232 + 8);
            v52 = v233 + 1;
            if ( v233 + 1 > (unsigned __int64)*(unsigned int *)(v232 + 12) )
            {
              v323 = v331;
              v335 = v232;
              sub_C8D5F0(v232, (const void *)(v232 + 16), v233 + 1, 8u, v230, v52);
              v232 = v335;
              v230 = v323;
              v233 = *(unsigned int *)(v335 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v232 + 8 * v233) = v230;
            ++*(_DWORD *)(v232 + 8);
            v60 = v56[1] & 0xFFFFFFFFFFFFFFF8LL;
          }
          v61 = *(unsigned int *)(v60 + 8);
          if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(v60 + 12) )
          {
            v334 = v60;
            sub_C8D5F0(v60, (const void *)(v60 + 16), v61 + 1, 8u, v60, v52);
            v60 = v334;
            v61 = *(unsigned int *)(v334 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v60 + 8 * v61) = v343;
          ++*(_DWORD *)(v60 + 8);
          goto LABEL_76;
        }
LABEL_346:
        *v59 = v343 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_76:
        v62 = v325[1];
        v63 = *v325;
        v64 = v62;
        v65 = (char *)v62 - (char *)*v325;
        v66 = v65 >> 5;
        v67 = v65 >> 3;
        if ( v66 <= 0 )
          goto LABEL_348;
        v68 = &v63[4 * v66];
        do
        {
          if ( v48 == *v63 )
            goto LABEL_83;
          if ( v48 == v63[1] )
          {
            ++v63;
            goto LABEL_83;
          }
          if ( v48 == v63[2] )
          {
            v63 += 2;
            goto LABEL_83;
          }
          if ( v48 == v63[3] )
          {
            v63 += 3;
            goto LABEL_83;
          }
          v63 += 4;
        }
        while ( v68 != v63 );
        v67 = v62 - v63;
LABEL_348:
        if ( v67 != 2 )
        {
          if ( v67 != 3 )
          {
            if ( v67 == 1 )
            {
              if ( v48 != *v63 )
                goto LABEL_352;
LABEL_83:
              if ( v63 != v62 )
              {
                v69 = v63 + 1;
                if ( v62 == v63 + 1 )
                  goto LABEL_90;
                do
                {
                  if ( v48 != *v69 )
                    *v63++ = *v69;
                  ++v69;
                }
                while ( v62 != v69 );
                if ( v62 != v63 )
                {
                  v62 = v325[1];
LABEL_90:
                  if ( v64 != v62 )
                  {
                    v70 = (__int64 *)memmove(v63, v64, (char *)v62 - (char *)v64);
                    v62 = v325[1];
                    v63 = v70;
                  }
                  v71 = (__int64 *)((char *)v63 + (char *)v62 - (char *)v64);
                  if ( v62 != v71 )
                    v325[1] = v71;
                }
              }
            }
            v72 = *(_DWORD *)(v2 + 40);
            if ( !v72 )
              goto LABEL_353;
            goto LABEL_95;
          }
          if ( v48 == *v63 )
            goto LABEL_83;
          ++v63;
        }
        if ( v48 == *v63 )
          goto LABEL_83;
        if ( v48 == *++v63 )
          goto LABEL_83;
LABEL_352:
        v72 = *(_DWORD *)(v2 + 40);
        if ( !v72 )
        {
LABEL_353:
          ++*(_QWORD *)(v2 + 16);
          goto LABEL_354;
        }
LABEL_95:
        v73 = *(_QWORD *)(v2 + 24);
        v74 = 1;
        v75 = 0;
        v76 = (v72 - 1) & (((unsigned int)v48 >> 4) ^ ((unsigned int)v48 >> 9));
        v77 = (__int64 *)(v73 + 16LL * v76);
        v78 = *v77;
        if ( v48 != *v77 )
        {
          while ( v78 != -4096 )
          {
            if ( v78 == -8192 && !v75 )
              v75 = v77;
            v76 = (v72 - 1) & (v74 + v76);
            v77 = (__int64 *)(v73 + 16LL * v76);
            v78 = *v77;
            if ( v48 == *v77 )
              goto LABEL_96;
            ++v74;
          }
          if ( v75 )
            v77 = v75;
          v273 = *(_DWORD *)(v2 + 32);
          ++*(_QWORD *)(v2 + 16);
          v218 = v273 + 1;
          if ( 4 * v218 >= 3 * v72 )
          {
LABEL_354:
            sub_B2ACE0(v326, 2 * v72);
            v214 = *(_DWORD *)(v2 + 40);
            if ( !v214 )
              goto LABEL_600;
            v215 = v214 - 1;
            v216 = *(_QWORD *)(v2 + 24);
            LODWORD(v217) = v215 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v218 = *(_DWORD *)(v2 + 32) + 1;
            v77 = (__int64 *)(v216 + 16LL * (unsigned int)v217);
            v219 = *v77;
            if ( v48 != *v77 )
            {
              v220 = 1;
              v221 = 0;
              while ( v219 != -4096 )
              {
                if ( !v221 && v219 == -8192 )
                  v221 = v77;
                v217 = v215 & (unsigned int)(v217 + v220);
                v77 = (__int64 *)(v216 + 16 * v217);
                v219 = *v77;
                if ( v48 == *v77 )
                  goto LABEL_475;
                ++v220;
              }
              if ( v221 )
                v77 = v221;
            }
          }
          else if ( v72 - *(_DWORD *)(v2 + 36) - v218 <= v72 >> 3 )
          {
            sub_B2ACE0(v326, v72);
            v274 = *(_DWORD *)(v2 + 40);
            if ( !v274 )
              goto LABEL_600;
            v275 = v274 - 1;
            v276 = *(_QWORD *)(v2 + 24);
            v277 = 0;
            LODWORD(v278) = v275 & (((unsigned int)v48 >> 4) ^ ((unsigned int)v48 >> 9));
            v279 = 1;
            v218 = *(_DWORD *)(v2 + 32) + 1;
            v77 = (__int64 *)(v276 + 16LL * (unsigned int)v278);
            v280 = *v77;
            if ( v48 != *v77 )
            {
              while ( v280 != -4096 )
              {
                if ( !v277 && v280 == -8192 )
                  v277 = v77;
                v278 = v275 & (unsigned int)(v278 + v279);
                v77 = (__int64 *)(v276 + 16 * v278);
                v280 = *v77;
                if ( v48 == *v77 )
                  goto LABEL_475;
                ++v279;
              }
              if ( v277 )
                v77 = v277;
            }
          }
LABEL_475:
          *(_DWORD *)(v2 + 32) = v218;
          if ( *v77 != -4096 )
            --*(_DWORD *)(v2 + 36);
          *v77 = v48;
          v77[1] = 0;
        }
LABEL_96:
        ++v46;
        sub_3012780(v77 + 1, v343);
        if ( v337 == v46 )
        {
          v328 = v325[1];
          v338 = *v325;
          if ( v328 != *v325 )
          {
            v318 = v2;
            do
            {
              v79 = *(_QWORD *)(*v338 + 56);
              for ( j = *v338 + 48; v79 != j; v79 = *(_QWORD *)(v79 + 8) )
              {
                v81 = (unsigned __int8 *)(v79 - 24);
                if ( !v79 )
                  v81 = 0;
                sub_FC75A0((__int64 *)&v368, (__int64)&v361, 3, 0, 0, 0);
                sub_FCD280((__int64 *)&v368, v81, v82, v83, v84, v85);
                sub_FC7680((__int64 *)&v368, (__int64)v81);
              }
              ++v338;
            }
            while ( v328 != v338 );
            v2 = v318;
          }
          v86 = v347;
          v87 = &v350;
          v350 = (__int64 *)v352;
          v351 = 0x200000000LL;
          if ( v348 == v347 )
            goto LABEL_239;
          v88 = v2;
          v89 = v348;
          while ( 2 )
          {
            while ( 1 )
            {
              v90 = v86->m128i_i64[0];
              v91 = v86->m128i_i64[1];
              LODWORD(v351) = 0;
              v92 = *(_QWORD *)(v90 + 16);
              if ( v92 )
                break;
LABEL_126:
              if ( v89 == ++v86 )
                goto LABEL_127;
            }
            do
            {
              v93 = *(_QWORD *)(v92 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v93 - 30) <= 0xAu )
              {
LABEL_111:
                v94 = *(_QWORD *)(v93 + 40);
                v95 = *(_QWORD *)(v94 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v95 == v94 + 48
                  || !v95
                  || (v96 = v95 - 24, (unsigned int)*(unsigned __int8 *)(v95 - 24) - 30 > 0xA) )
                {
LABEL_599:
                  BUG();
                }
                if ( *(_BYTE *)(v95 - 24) == 38
                  && v344 == **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v95 - 88) - 32LL) - 8LL) )
                {
                  v210 = (unsigned int)v351;
                  v211 = (unsigned int)v351 + 1LL;
                  if ( v211 > HIDWORD(v351) )
                  {
                    v322 = v88;
                    v332 = v96;
                    v342 = v87;
                    sub_C8D5F0((__int64)v87, v352, v211, 8u, v88, v96);
                    v210 = (unsigned int)v351;
                    v88 = v322;
                    v96 = v332;
                    v87 = v342;
                  }
                  v350[v210] = v96;
                  LODWORD(v351) = v351 + 1;
                }
                while ( 1 )
                {
                  v92 = *(_QWORD *)(v92 + 8);
                  if ( !v92 )
                    break;
                  v93 = *(_QWORD *)(v92 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v93 - 30) <= 0xAu )
                    goto LABEL_111;
                }
                v97 = v350;
                v98 = &v350[(unsigned int)v351];
                if ( v98 != v350 )
                {
                  do
                  {
                    v99 = *v97;
                    if ( *(_QWORD *)(*v97 - 32) )
                    {
                      v100 = *(_QWORD *)(v99 - 24);
                      **(_QWORD **)(v99 - 16) = v100;
                      if ( v100 )
                        *(_QWORD *)(v100 + 16) = *(_QWORD *)(v99 - 16);
                    }
                    *(_QWORD *)(v99 - 32) = v91;
                    if ( v91 )
                    {
                      v101 = *(_QWORD *)(v91 + 16);
                      *(_QWORD *)(v99 - 24) = v101;
                      if ( v101 )
                        *(_QWORD *)(v101 + 16) = v99 - 24;
                      *(_QWORD *)(v99 - 16) = v91 + 16;
                      *(_QWORD *)(v91 + 16) = v99 - 32;
                    }
                    ++v97;
                  }
                  while ( v98 != v97 );
                }
                goto LABEL_126;
              }
              v92 = *(_QWORD *)(v92 + 8);
            }
            while ( v92 );
            if ( v89 != ++v86 )
              continue;
            break;
          }
LABEL_127:
          v2 = v88;
          v314 = v348;
          if ( v348 == v347 )
          {
LABEL_239:
            if ( (_DWORD)v363 )
            {
              v234 = v362;
              v346 = &v362[8 * (unsigned __int64)v364];
              if ( v362 != v346 )
              {
                while ( 1 )
                {
                  v235 = v234[3];
                  if ( v235 != -8192 && v235 != -4096 )
                    break;
                  v234 += 8;
                  if ( v346 == v234 )
                    goto LABEL_240;
                }
                if ( v234 != v346 )
                {
                  while ( 1 )
                  {
                    v236 = (__int64 *)v234[3];
                    v354[0] = 6;
                    v354[1] = 0;
                    v353 = v236;
                    v237 = v234[7];
                    v333 = v236;
                    v355 = v237;
                    if ( v237 == -4096 || v237 == 0 || v237 == -8192 )
                    {
                      v368 = (const char *)&v370;
                      v369 = 0x1000000000LL;
                      if ( *(_BYTE *)v236 <= 0x1Cu )
                        goto LABEL_414;
                    }
                    else
                    {
                      sub_BD6050(v354, v234[5] & 0xFFFFFFFFFFFFFFF8LL);
                      v237 = v355;
                      v368 = (const char *)&v370;
                      v369 = 0x1000000000LL;
                      v333 = v353;
                      if ( *(_BYTE *)v353 <= 0x1Cu )
                        goto LABEL_425;
                    }
                    v238 = v333[2];
                    if ( !v238 )
                      goto LABEL_424;
                    v313 = v237;
                    do
                    {
                      v252 = *(_DWORD *)(v2 + 40);
                      v253 = *(_QWORD *)(*(_QWORD *)(v238 + 24) + 40LL);
                      if ( !v252 )
                      {
                        ++*(_QWORD *)(v2 + 16);
                        goto LABEL_404;
                      }
                      v239 = v252 - 1;
                      v240 = *(_QWORD *)(v2 + 24);
                      v241 = 1;
                      v242 = 0;
                      v243 = v239 & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
                      v244 = (__int64 *)(v240 + 16LL * v243);
                      v245 = *v244;
                      if ( v253 != *v244 )
                      {
                        while ( v245 != -4096 )
                        {
                          if ( !v242 && v245 == -8192 )
                            v242 = (__int64)v244;
                          v243 = v239 & (v241 + v243);
                          v244 = (__int64 *)(v240 + 16LL * v243);
                          v245 = *v244;
                          if ( v253 == *v244 )
                            goto LABEL_393;
                          ++v241;
                        }
                        v259 = *(_DWORD *)(v2 + 32);
                        if ( v242 )
                          v244 = (__int64 *)v242;
                        ++*(_QWORD *)(v2 + 16);
                        v257 = v259 + 1;
                        if ( 4 * (v259 + 1) >= 3 * v252 )
                        {
LABEL_404:
                          sub_B2ACE0(v326, 2 * v252);
                          v254 = *(_DWORD *)(v2 + 40);
                          if ( !v254 )
                            goto LABEL_600;
                          v242 = (unsigned int)(v254 - 1);
                          v255 = *(_QWORD *)(v2 + 24);
                          LODWORD(v256) = v242 & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
                          v257 = *(_DWORD *)(v2 + 32) + 1;
                          v244 = (__int64 *)(v255 + 16LL * (unsigned int)v256);
                          v239 = *v244;
                          if ( v253 != *v244 )
                          {
                            v305 = 1;
                            v306 = 0;
                            while ( v239 != -4096 )
                            {
                              if ( !v306 && v239 == -8192 )
                                v306 = v244;
                              v256 = (unsigned int)v242 & ((_DWORD)v256 + v305);
                              v244 = (__int64 *)(v255 + 16 * v256);
                              v239 = *v244;
                              if ( v253 == *v244 )
                                goto LABEL_406;
                              ++v305;
                            }
                            if ( v306 )
                              v244 = v306;
                          }
                        }
                        else if ( v252 - *(_DWORD *)(v2 + 36) - v257 <= v252 >> 3 )
                        {
                          sub_B2ACE0(v326, v252);
                          v260 = *(_DWORD *)(v2 + 40);
                          if ( !v260 )
                            goto LABEL_600;
                          v261 = v260 - 1;
                          v242 = *(_QWORD *)(v2 + 24);
                          v262 = 1;
                          LODWORD(v263) = v261 & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
                          v257 = *(_DWORD *)(v2 + 32) + 1;
                          v264 = 0;
                          v244 = (__int64 *)(v242 + 16LL * (unsigned int)v263);
                          v239 = *v244;
                          if ( v253 != *v244 )
                          {
                            while ( v239 != -4096 )
                            {
                              if ( !v264 && v239 == -8192 )
                                v264 = v244;
                              v263 = v261 & (unsigned int)(v263 + v262);
                              v244 = (__int64 *)(v242 + 16 * v263);
                              v239 = *v244;
                              if ( v253 == *v244 )
                                goto LABEL_406;
                              ++v262;
                            }
                            if ( v264 )
                              v244 = v264;
                          }
                        }
LABEL_406:
                        *(_DWORD *)(v2 + 32) = v257;
                        if ( *v244 != -4096 )
                          --*(_DWORD *)(v2 + 36);
                        *v244 = v253;
                        v246 = v244 + 1;
                        v244[1] = 0;
                        goto LABEL_409;
                      }
LABEL_393:
                      v246 = v244 + 1;
                      v247 = v244[1];
                      v248 = (v247 >> 2) & 1;
                      v249 = v247 & 0xFFFFFFFFFFFFFFF8LL;
                      if ( !v249 || (_BYTE)v248 && !*(_DWORD *)(v249 + 8) )
                      {
                        if ( (_DWORD)v248 == 1 )
LABEL_412:
                          v246 = *(_QWORD **)v249;
LABEL_409:
                        if ( v343 == *v246 )
                          goto LABEL_401;
                        goto LABEL_398;
                      }
                      if ( !v248 )
                        goto LABEL_409;
                      if ( *(_DWORD *)(v249 + 8) <= 1u )
                        goto LABEL_412;
LABEL_398:
                      v250 = (unsigned int)v369;
                      v251 = (unsigned int)v369 + 1LL;
                      if ( v251 > HIDWORD(v369) )
                      {
                        sub_C8D5F0((__int64)&v368, &v370, v251, 8u, v239, v242);
                        v250 = (unsigned int)v369;
                      }
                      *(_QWORD *)&v368[8 * v250] = v238;
                      LODWORD(v369) = v369 + 1;
LABEL_401:
                      v238 = *(_QWORD *)(v238 + 8);
                    }
                    while ( v238 );
                    if ( (_DWORD)v369 )
                    {
                      sub_11D2BF0((__int64)&v356, 0);
                      v265 = (unsigned __int8 *)sub_BD5D20((__int64)v333);
                      sub_11D2C80((__int64 *)&v356, v333[1], v265, v266);
                      sub_11D33F0((__int64 *)&v356, v333[5], (__int64)v333);
                      sub_11D33F0((__int64 *)&v356, *(_QWORD *)(v313 + 40), v313);
                      for ( k = v369; (_DWORD)v369; k = v369 )
                      {
                        v268 = *(_QWORD *)&v368[8 * k - 8];
                        LODWORD(v369) = k - 1;
                        sub_11D9830((__int64 *)&v356, v268);
                      }
                      sub_11D2C20((__int64 *)&v356);
                      if ( v368 != (const char *)&v370 )
                        _libc_free((unsigned __int64)v368);
                      if ( v355 != 0 && v355 != -4096 && v355 != -8192 )
                        goto LABEL_427;
                    }
                    else
                    {
                      if ( v368 != (const char *)&v370 )
                        _libc_free((unsigned __int64)v368);
LABEL_424:
                      v237 = v355;
LABEL_425:
                      if ( v237 != -4096 && v237 != 0 && v237 != -8192 )
LABEL_427:
                        sub_BD60C0(v354);
                    }
LABEL_414:
                    v234 += 8;
                    if ( v234 != v346 )
                    {
                      while ( 1 )
                      {
                        v258 = v234[3];
                        if ( v258 != -8192 && v258 != -4096 )
                          break;
                        v234 += 8;
                        if ( v346 == v234 )
                          goto LABEL_240;
                      }
                      if ( v346 != v234 )
                        continue;
                    }
                    break;
                  }
                }
              }
            }
LABEL_240:
            if ( v350 != (__int64 *)v352 )
              _libc_free((unsigned __int64)v350);
            if ( v367 )
            {
              v269 = v366;
              v367 = 0;
              if ( v366 )
              {
                v270 = v365;
                v271 = &v365[2 * v366];
                do
                {
                  if ( *v270 != -8192 && *v270 != -4096 )
                  {
                    v272 = v270[1];
                    if ( v272 )
                      sub_B91220((__int64)(v270 + 1), v272);
                  }
                  v270 += 2;
                }
                while ( v271 != v270 );
                v269 = v366;
              }
              sub_C7D6A0((__int64)v365, 16 * v269, 8);
            }
            v172 = v364;
            if ( v364 )
            {
              v173 = v362;
              v357 = 2;
              v358 = 0;
              v174 = -4096;
              v175 = &v362[8 * (unsigned __int64)v364];
              v359 = -4096;
              v356 = (char *)&unk_49DD7B0;
              v360 = 0;
              v369 = 2;
              v370 = 0;
              v371 = -8192;
              v368 = (const char *)&unk_49DD7B0;
              i = 0;
              while ( 1 )
              {
                v176 = v173[3];
                if ( v176 != v174 )
                {
                  v174 = v371;
                  if ( v176 != v371 )
                  {
                    v177 = v173[7];
                    if ( v177 != -4096 && v177 != 0 && v177 != -8192 )
                    {
                      sub_BD60C0(v173 + 5);
                      v176 = v173[3];
                    }
                    v174 = v176;
                  }
                }
                *v173 = &unk_49DB368;
                if ( v174 != 0 && v174 != -4096 && v174 != -8192 )
                  sub_BD60C0(v173 + 1);
                v173 += 8;
                if ( v175 == v173 )
                  break;
                v174 = v359;
              }
              v368 = (const char *)&unk_49DB368;
              v178 = v371;
              if ( v371 == 0 || v371 == -4096 )
                goto LABEL_259;
              goto LABEL_257;
            }
            goto LABEL_263;
          }
          v329 = v347;
          v102 = v88;
LABEL_129:
          v309 = v102;
          v319 = v329->m128i_i64[1];
          v103 = sub_AA5930(v329->m128i_i64[0]);
          v104 = v309;
          v339 = v105;
          v106 = v103;
          if ( v103 == v105 )
            goto LABEL_175;
          v107 = v309;
          while ( 1 )
          {
            v108 = 0;
            v109 = *(_DWORD *)(v106 + 4) & 0x7FFFFFF;
            if ( v109 )
              break;
LABEL_170:
            v130 = *(_QWORD *)(v106 + 32);
            if ( !v130 )
              goto LABEL_599;
            v106 = 0;
            if ( *(_BYTE *)(v130 - 24) == 84 )
              v106 = v130 - 24;
            if ( v339 == v106 )
            {
              v104 = v107;
LABEL_175:
              v310 = v104;
              v131 = sub_AA5930(v319);
              v102 = v310;
              v340 = v132;
              v133 = v131;
              if ( v131 == v132 )
                goto LABEL_205;
              while ( 1 )
              {
                v134 = 0;
                v135 = *(_DWORD *)(v133 + 4) & 0x7FFFFFF;
                if ( v135 )
                  break;
LABEL_200:
                v149 = *(_QWORD *)(v133 + 32);
                if ( !v149 )
                  goto LABEL_599;
                v133 = 0;
                if ( *(_BYTE *)(v149 - 24) == 84 )
                  v133 = v149 - 24;
                if ( v340 == v133 )
                {
                  v102 = v310;
LABEL_205:
                  if ( v314 == ++v329 )
                  {
                    v2 = v102;
                    v315 = v348;
                    if ( v348 != v347 )
                    {
                      v320 = v347;
                      v308 = v102;
                      do
                      {
                        v150 = v320->m128i_i64[0];
                        v151 = (_QWORD *)(v320->m128i_i64[1] + 48);
                        v345 = v320->m128i_i64[1];
                        v152 = *v151 & 0xFFFFFFFFFFFFFFF8LL;
                        if ( (_QWORD *)v152 != v151 )
                        {
                          if ( !v152 )
                            BUG();
                          v330 = v152 - 24;
                          if ( (unsigned int)*(unsigned __int8 *)(v152 - 24) - 30 <= 0xA )
                          {
                            v341 = sub_B46E30(v152 - 24);
                            if ( v341 )
                            {
                              for ( m = 0; m != v341; ++m )
                              {
                                v154 = sub_B46EC0(v330, m);
                                v155 = sub_AA5930(v154);
                                v157 = v156;
                                v158 = v155;
                                while ( v157 != v158 )
                                {
                                  v159 = *(_DWORD *)(v158 + 4) & 0x7FFFFFF;
                                  if ( !v159 )
                                    break;
                                  v160 = *(_QWORD *)(v158 - 8);
                                  v161 = 0;
                                  v162 = *(unsigned int *)(v158 + 72);
                                  while ( v150 != *(_QWORD *)(v160 + 32 * v162 + 8 * v161) )
                                  {
                                    if ( v159 == (_DWORD)++v161 )
                                      goto LABEL_236;
                                  }
                                  v163 = *(_QWORD *)(v160 + 32 * v161);
                                  if ( *(_BYTE *)v163 > 0x1Cu && v364 )
                                  {
                                    v164 = (v364 - 1) & (((unsigned int)v163 >> 9) ^ ((unsigned int)v163 >> 4));
                                    v165 = &v362[8 * (unsigned __int64)v164];
                                    v166 = v165[3];
                                    if ( v163 == v166 )
                                    {
LABEL_221:
                                      if ( v165 != &v362[8 * (unsigned __int64)v364] )
                                        v163 = v165[7];
                                    }
                                    else
                                    {
                                      v179 = 1;
                                      while ( v166 != -4096 )
                                      {
                                        v164 = (v364 - 1) & (v179 + v164);
                                        v311 = v179 + 1;
                                        v165 = &v362[8 * (unsigned __int64)v164];
                                        v166 = v165[3];
                                        if ( v163 == v166 )
                                          goto LABEL_221;
                                        v179 = v311;
                                      }
                                    }
                                  }
                                  if ( v159 == (_DWORD)v162 )
                                  {
                                    sub_B48D90(v158);
                                    v160 = *(_QWORD *)(v158 - 8);
                                    v159 = *(_DWORD *)(v158 + 4) & 0x7FFFFFF;
                                  }
                                  v167 = (v159 + 1) & 0x7FFFFFF;
                                  *(_DWORD *)(v158 + 4) = v167 | *(_DWORD *)(v158 + 4) & 0xF8000000;
                                  v168 = 32LL * (unsigned int)(v167 - 1) + v160;
                                  if ( *(_QWORD *)v168 )
                                  {
                                    v169 = *(_QWORD *)(v168 + 8);
                                    **(_QWORD **)(v168 + 16) = v169;
                                    if ( v169 )
                                      *(_QWORD *)(v169 + 16) = *(_QWORD *)(v168 + 16);
                                  }
                                  *(_QWORD *)v168 = v163;
                                  if ( v163 )
                                  {
                                    v170 = *(_QWORD *)(v163 + 16);
                                    *(_QWORD *)(v168 + 8) = v170;
                                    if ( v170 )
                                      *(_QWORD *)(v170 + 16) = v168 + 8;
                                    *(_QWORD *)(v168 + 16) = v163 + 16;
                                    *(_QWORD *)(v163 + 16) = v168;
                                  }
                                  *(_QWORD *)(*(_QWORD *)(v158 - 8)
                                            + 32LL * *(unsigned int *)(v158 + 72)
                                            + 8LL * ((*(_DWORD *)(v158 + 4) & 0x7FFFFFFu) - 1)) = v345;
                                  v171 = *(_QWORD *)(v158 + 32);
                                  if ( !v171 )
                                    goto LABEL_599;
                                  v158 = 0;
                                  if ( *(_BYTE *)(v171 - 24) == 84 )
                                    v158 = v171 - 24;
                                }
LABEL_236:
                                ;
                              }
                            }
                          }
                        }
                        ++v320;
                      }
                      while ( v315 != v320 );
                      v2 = v308;
                    }
                    goto LABEL_239;
                  }
                  goto LABEL_129;
                }
              }
              while ( 1 )
              {
                v137 = *(_QWORD *)(*(_QWORD *)(v133 - 8) + 32LL * *(unsigned int *)(v133 + 72) + 8LL * v134);
                v138 = *(_QWORD *)(v137 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v138 == v137 + 48 || !v138 || (unsigned int)*(unsigned __int8 *)(v138 - 24) - 30 > 0xA )
                  goto LABEL_599;
                if ( *(_BYTE *)(v138 - 24) != 38 )
                  break;
                v136 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v138 - 88) - 32LL) - 8LL) == v344;
LABEL_179:
                if ( v136 )
                {
                  if ( v135 == ++v134 )
                    goto LABEL_200;
                }
                else
                {
                  if ( (*(_DWORD *)(v133 + 4) & 0x7FFFFFF) != 0 )
                  {
                    v148 = 0;
                    while ( v137 != *(_QWORD *)(*(_QWORD *)(v133 - 8) + 32LL * *(unsigned int *)(v133 + 72) + 8 * v148) )
                    {
                      if ( (*(_DWORD *)(v133 + 4) & 0x7FFFFFF) == (_DWORD)++v148 )
                        goto LABEL_267;
                    }
                    --v135;
                    sub_B48BF0(v133, v148, 0);
                  }
                  else
                  {
LABEL_267:
                    --v135;
                    sub_B48BF0(v133, 0xFFFFFFFF, 0);
                  }
                  if ( v135 == v134 )
                    goto LABEL_200;
                }
              }
              v139 = *(_DWORD *)(v310 + 40);
              if ( v139 )
              {
                v140 = *(_QWORD *)(v310 + 24);
                v141 = 1;
                v142 = 0;
                v143 = (v139 - 1) & (((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4));
                v144 = (_QWORD *)(v140 + 16LL * v143);
                v145 = *v144;
                if ( v137 == *v144 )
                {
LABEL_187:
                  v146 = v144[1];
                  v147 = (_QWORD **)(v146 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (v146 & 4) != 0 )
                    v147 = (_QWORD **)**v147;
LABEL_189:
                  v136 = v343 == (_QWORD)v147;
                  goto LABEL_179;
                }
                while ( v145 != -4096 )
                {
                  if ( !v142 && v145 == -8192 )
                    v142 = v144;
                  v143 = (v139 - 1) & (v141 + v143);
                  v144 = (_QWORD *)(v140 + 16LL * v143);
                  v145 = *v144;
                  if ( v137 == *v144 )
                    goto LABEL_187;
                  ++v141;
                }
                if ( !v142 )
                  v142 = v144;
                v180 = *(_DWORD *)(v310 + 32);
                ++*(_QWORD *)(v310 + 16);
                v181 = v180 + 1;
                if ( 4 * v181 < 3 * v139 )
                {
                  if ( v139 - *(_DWORD *)(v310 + 36) - v181 > v139 >> 3 )
                    goto LABEL_282;
                  v321 = ((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4);
                  sub_B2ACE0(v326, v139);
                  v204 = *(_DWORD *)(v310 + 40);
                  if ( !v204 )
                    goto LABEL_600;
                  v205 = v204 - 1;
                  v206 = *(_QWORD *)(v310 + 24);
                  v197 = 0;
                  v207 = 1;
                  v208 = v205 & v321;
                  v181 = *(_DWORD *)(v310 + 32) + 1;
                  v142 = (_QWORD *)(v206 + 16LL * (v205 & v321));
                  v209 = *v142;
                  if ( v137 == *v142 )
                    goto LABEL_282;
                  while ( v209 != -4096 )
                  {
                    if ( v209 == -8192 && !v197 )
                      v197 = v142;
                    v208 = v205 & (v207 + v208);
                    v142 = (_QWORD *)(v206 + 16LL * v208);
                    v209 = *v142;
                    if ( v137 == *v142 )
                      goto LABEL_282;
                    ++v207;
                  }
                  goto LABEL_311;
                }
              }
              else
              {
                ++*(_QWORD *)(v310 + 16);
              }
              sub_B2ACE0(v326, 2 * v139);
              v191 = *(_DWORD *)(v310 + 40);
              if ( !v191 )
                goto LABEL_600;
              v192 = v191 - 1;
              v193 = *(_QWORD *)(v310 + 24);
              v194 = v192 & (((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4));
              v181 = *(_DWORD *)(v310 + 32) + 1;
              v142 = (_QWORD *)(v193 + 16LL * v194);
              v195 = *v142;
              if ( v137 == *v142 )
                goto LABEL_282;
              v196 = 1;
              v197 = 0;
              while ( v195 != -4096 )
              {
                if ( !v197 && v195 == -8192 )
                  v197 = v142;
                v194 = v192 & (v196 + v194);
                v142 = (_QWORD *)(v193 + 16LL * v194);
                v195 = *v142;
                if ( v137 == *v142 )
                  goto LABEL_282;
                ++v196;
              }
LABEL_311:
              if ( v197 )
                v142 = v197;
LABEL_282:
              *(_DWORD *)(v310 + 32) = v181;
              if ( *v142 != -4096 )
                --*(_DWORD *)(v310 + 36);
              *v142 = v137;
              v147 = 0;
              v142[1] = 0;
              goto LABEL_189;
            }
          }
          while ( 2 )
          {
            v111 = *(_QWORD *)(*(_QWORD *)(v106 - 8) + 32LL * *(unsigned int *)(v106 + 72) + 8LL * v108);
            v112 = *(_QWORD *)(v111 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v112 == v111 + 48 || !v112 || (unsigned int)*(unsigned __int8 *)(v112 - 24) - 30 > 0xA )
              goto LABEL_599;
            if ( *(_BYTE *)(v112 - 24) == 38 )
            {
              v110 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v112 - 88) - 32LL) - 8LL) == v344;
            }
            else
            {
              v113 = *(_DWORD *)(v107 + 40);
              if ( v113 )
              {
                v114 = *(_QWORD *)(v107 + 24);
                v115 = 1;
                v116 = 0;
                v117 = (v113 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                v118 = (_QWORD *)(v114 + 16LL * v117);
                v119 = *v118;
                if ( v111 == *v118 )
                {
LABEL_142:
                  v120 = v118[1];
                  v121 = (_QWORD **)(v120 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (v120 & 4) != 0 )
                    v121 = (_QWORD **)**v121;
                  goto LABEL_144;
                }
                while ( v119 != -4096 )
                {
                  if ( !v116 && v119 == -8192 )
                    v116 = v118;
                  v117 = (v113 - 1) & (v115 + v117);
                  v118 = (_QWORD *)(v114 + 16LL * v117);
                  v119 = *v118;
                  if ( v111 == *v118 )
                    goto LABEL_142;
                  ++v115;
                }
                if ( !v116 )
                  v116 = v118;
                v182 = *(_DWORD *)(v107 + 32);
                ++*(_QWORD *)(v107 + 16);
                v183 = v182 + 1;
                if ( 4 * v183 < 3 * v113 )
                {
                  if ( v113 - *(_DWORD *)(v107 + 36) - v183 <= v113 >> 3 )
                  {
                    v312 = ((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4);
                    sub_B2ACE0(v326, v113);
                    v198 = *(_DWORD *)(v107 + 40);
                    if ( !v198 )
                      goto LABEL_600;
                    v199 = v198 - 1;
                    v200 = *(_QWORD *)(v107 + 24);
                    v190 = 0;
                    v201 = 1;
                    v202 = v199 & v312;
                    v183 = *(_DWORD *)(v107 + 32) + 1;
                    v116 = (_QWORD *)(v200 + 16LL * (v199 & v312));
                    v203 = *v116;
                    if ( v111 != *v116 )
                    {
                      while ( v203 != -4096 )
                      {
                        if ( !v190 && v203 == -8192 )
                          v190 = v116;
                        v202 = v199 & (v201 + v202);
                        v116 = (_QWORD *)(v200 + 16LL * v202);
                        v203 = *v116;
                        if ( v111 == *v116 )
                          goto LABEL_295;
                        ++v201;
                      }
                      goto LABEL_303;
                    }
                  }
                  goto LABEL_295;
                }
              }
              else
              {
                ++*(_QWORD *)(v107 + 16);
              }
              sub_B2ACE0(v326, 2 * v113);
              v184 = *(_DWORD *)(v107 + 40);
              if ( !v184 )
                goto LABEL_600;
              v185 = v184 - 1;
              v186 = *(_QWORD *)(v107 + 24);
              v187 = v185 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
              v183 = *(_DWORD *)(v107 + 32) + 1;
              v116 = (_QWORD *)(v186 + 16LL * v187);
              v188 = *v116;
              if ( v111 != *v116 )
              {
                v189 = 1;
                v190 = 0;
                while ( v188 != -4096 )
                {
                  if ( !v190 && v188 == -8192 )
                    v190 = v116;
                  v187 = v185 & (v189 + v187);
                  v116 = (_QWORD *)(v186 + 16LL * v187);
                  v188 = *v116;
                  if ( v111 == *v116 )
                    goto LABEL_295;
                  ++v189;
                }
LABEL_303:
                if ( v190 )
                  v116 = v190;
              }
LABEL_295:
              *(_DWORD *)(v107 + 32) = v183;
              if ( *v116 != -4096 )
                --*(_DWORD *)(v107 + 36);
              *v116 = v111;
              v121 = 0;
              v116[1] = 0;
LABEL_144:
              v110 = v343 == (_QWORD)v121;
            }
            if ( v110 )
            {
              if ( (*(_DWORD *)(v106 + 4) & 0x7FFFFFF) != 0 )
              {
                v129 = 0;
                while ( v111 != *(_QWORD *)(*(_QWORD *)(v106 - 8) + 32LL * *(unsigned int *)(v106 + 72) + 8 * v129) )
                {
                  if ( (*(_DWORD *)(v106 + 4) & 0x7FFFFFF) == (_DWORD)++v129 )
                    goto LABEL_190;
                }
                --v109;
                sub_B48BF0(v106, v129, 0);
              }
              else
              {
LABEL_190:
                --v109;
                sub_B48BF0(v106, 0xFFFFFFFF, 0);
              }
              if ( v109 == v108 )
                goto LABEL_170;
            }
            else if ( v109 == ++v108 )
            {
              goto LABEL_170;
            }
            continue;
          }
        }
      }
      while ( v57 != (const char *)-4096LL )
      {
        if ( v57 == (const char *)-8192LL && !v54 )
          v54 = v56;
        v55 = (v50 - 1) & (v53 + v55);
        v56 = (__int64 *)(v52 + 16LL * v55);
        v57 = (const char *)*v56;
        if ( v368 == (const char *)*v56 )
          goto LABEL_71;
        ++v53;
      }
      v212 = *(_DWORD *)(v2 + 32);
      if ( v54 )
        v56 = v54;
      ++*(_QWORD *)(v2 + 16);
      v213 = v212 + 1;
      if ( 4 * (v212 + 1) >= 3 * v50 )
      {
LABEL_363:
        sub_B2ACE0(v326, 2 * v50);
        v222 = *(_DWORD *)(v2 + 40);
        if ( !v222 )
          goto LABEL_600;
        v51 = (__int64)v368;
        v223 = v222 - 1;
        v224 = *(_QWORD *)(v2 + 24);
        v213 = *(_DWORD *)(v2 + 32) + 1;
        v225 = v223 & (((unsigned int)v368 >> 9) ^ ((unsigned int)v368 >> 4));
        v56 = (__int64 *)(v224 + 16LL * v225);
        v226 = *v56;
        if ( v368 == (const char *)*v56 )
          goto LABEL_343;
        v227 = 1;
        v228 = 0;
        while ( v226 != -4096 )
        {
          if ( !v228 && v226 == -8192 )
            v228 = v56;
          v225 = v223 & (v227 + v225);
          v56 = (__int64 *)(v224 + 16LL * v225);
          v226 = *v56;
          if ( v368 == (const char *)*v56 )
            goto LABEL_343;
          ++v227;
        }
      }
      else
      {
        if ( v50 - *(_DWORD *)(v2 + 36) - v213 > v50 >> 3 )
          goto LABEL_343;
        sub_B2ACE0(v326, v50);
        v286 = *(_DWORD *)(v2 + 40);
        if ( !v286 )
          goto LABEL_600;
        v287 = v286 - 1;
        v288 = *(_QWORD *)(v2 + 24);
        v228 = 0;
        v51 = (__int64)v368;
        v213 = *(_DWORD *)(v2 + 32) + 1;
        v289 = 1;
        v290 = v287 & (((unsigned int)v368 >> 9) ^ ((unsigned int)v368 >> 4));
        v56 = (__int64 *)(v288 + 16LL * v290);
        v291 = *v56;
        if ( v368 == (const char *)*v56 )
          goto LABEL_343;
        while ( v291 != -4096 )
        {
          if ( !v228 && v291 == -8192 )
            v228 = v56;
          v290 = v287 & (v289 + v290);
          v56 = (__int64 *)(v288 + 16LL * v290);
          v291 = *v56;
          if ( v368 == (const char *)*v56 )
            goto LABEL_343;
          ++v289;
        }
      }
      if ( v228 )
        v56 = v228;
LABEL_343:
      *(_DWORD *)(v2 + 32) = v213;
      if ( *v56 != -4096 )
        --*(_DWORD *)(v2 + 36);
      *v56 = v51;
      v59 = (unsigned __int64 *)(v56 + 1);
      v56[1] = 0;
      goto LABEL_346;
    }
    if ( v367 )
    {
      v301 = v366;
      v367 = 0;
      if ( v366 )
      {
        v302 = v365;
        v303 = &v365[2 * v366];
        do
        {
          if ( *v302 != -8192 && *v302 != -4096 )
          {
            v304 = v302[1];
            if ( v304 )
              sub_B91220((__int64)(v302 + 1), v304);
          }
          v302 += 2;
        }
        while ( v303 != v302 );
        v301 = v366;
      }
      sub_C7D6A0((__int64)v365, 16 * v301, 8);
    }
    v172 = v364;
    if ( v364 )
    {
      v281 = v362;
      v357 = 2;
      v358 = 0;
      v282 = -4096;
      v283 = &v362[8 * (unsigned __int64)v364];
      v359 = -4096;
      v356 = (char *)&unk_49DD7B0;
      v360 = 0;
      v369 = 2;
      v370 = 0;
      v371 = -8192;
      v368 = (const char *)&unk_49DD7B0;
      i = 0;
      while ( 1 )
      {
        v284 = v281[3];
        if ( v284 != v282 )
        {
          v282 = v371;
          if ( v284 != v371 )
          {
            v285 = v281[7];
            if ( v285 != -4096 && v285 != 0 && v285 != -8192 )
            {
              sub_BD60C0(v281 + 5);
              v284 = v281[3];
            }
            v282 = v284;
          }
        }
        *v281 = &unk_49DB368;
        if ( v282 != 0 && v282 != -4096 && v282 != -8192 )
          sub_BD60C0(v281 + 1);
        v281 += 8;
        if ( v283 == v281 )
          break;
        v282 = v359;
      }
      v368 = (const char *)&unk_49DB368;
      v178 = v371;
      if ( v371 == -4096 || v371 == 0 )
        goto LABEL_259;
LABEL_257:
      if ( v178 != -8192 )
        sub_BD60C0(&v369);
LABEL_259:
      v356 = (char *)&unk_49DB368;
      if ( v359 != 0 && v359 != -4096 && v359 != -8192 )
        sub_BD60C0(&v357);
      v172 = v364;
    }
LABEL_263:
    sub_C7D6A0((__int64)v362, v172 << 6, 8);
    if ( v347 )
      j_j___libc_free_0((unsigned __int64)v347);
    result = v325 + 4;
  }
  while ( v316 != v325 + 3 );
  return result;
}
