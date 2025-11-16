// Function: sub_18F6D00
// Address: 0x18f6d00
//
__int64 __fastcall sub_18F6D00(__int64 a1, __int64 i, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int32 v16; // eax
  unsigned __int16 v17; // ax
  __int64 v18; // r8
  __int64 v19; // rbx
  unsigned __int64 v20; // rax
  int v21; // eax
  unsigned __int64 v22; // r15
  bool v23; // bl
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rbx
  _QWORD *v27; // rax
  __int64 v28; // r12
  unsigned __int64 v29; // rax
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 *v34; // rcx
  __int64 v35; // r9
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // rcx
  _QWORD *v39; // r8
  _QWORD *v40; // r9
  __int64 *v41; // rbx
  __int64 *v42; // r12
  __int64 v44; // r14
  __int64 v45; // rdx
  unsigned int v46; // ebx
  __int64 v47; // rcx
  __int64 *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r12
  unsigned __int8 v53; // al
  __int64 v54; // r12
  int v55; // eax
  __int64 v56; // r10
  __int64 v57; // rcx
  unsigned int v58; // edx
  __int64 *v59; // rax
  __int64 v60; // r9
  unsigned __int16 v61; // ax
  __int64 v62; // rax
  __m128i v63; // xmm0
  __m128i v64; // xmm1
  int v65; // eax
  unsigned __int64 v66; // r12
  bool v67; // r15
  int v68; // eax
  __int64 v69; // rax
  _QWORD **v70; // rdx
  __m128i *v71; // rax
  __int64 v72; // r12
  __int64 v73; // rbx
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // rdx
  __int64 v77; // rbx
  __int64 n; // r13
  __int64 v79; // rdx
  __int64 v80; // rcx
  _QWORD *v81; // r8
  _QWORD *v82; // r9
  __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // rdi
  bool v86; // al
  __int64 v87; // r12
  bool v88; // bl
  __int64 v89; // r13
  unsigned __int16 v90; // ax
  __int64 v91; // rsi
  char v92; // bl
  unsigned __int8 v93; // dl
  __int64 v94; // rcx
  char v95; // dl
  unsigned int v96; // eax
  __m128i v97; // xmm2
  __m128i v98; // xmm3
  _BYTE *v99; // rdi
  __int64 v100; // rsi
  __int64 v101; // r9
  int v102; // eax
  _QWORD *v103; // rdx
  __m128i *v104; // r11
  int v105; // r15d
  int v106; // r10d
  __int64 v107; // rax
  __int64 v108; // rsi
  __int64 v109; // rdi
  __int64 *v110; // rax
  __int64 *v111; // r15
  __int64 *v112; // rbx
  __int64 v113; // r12
  const __m128i *v114; // rdi
  __m128i *v115; // rax
  __m128i *v116; // rcx
  __m128i *v117; // rdx
  __m128i *v118; // rax
  __m128i *v119; // rdx
  __int64 v120; // rax
  unsigned __int16 *v121; // rax
  __int64 v122; // rax
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r12
  __int64 v126; // rdi
  __int64 *v127; // rdx
  int v128; // r11d
  int v129; // eax
  unsigned __int16 v130; // ax
  __int64 v131; // rax
  __int64 v132; // r12
  unsigned __int8 v133; // bl
  __m128i v134; // xmm5
  char v135; // al
  __int64 v136; // rax
  __int64 v137; // r12
  __int64 v138; // rdi
  __int64 v139; // rax
  __int64 v140; // rcx
  __int64 v141; // r8
  char v142; // r12
  __int64 v143; // rax
  __int64 v144; // rdx
  __int64 v145; // rcx
  __m128i *v146; // r8
  int v147; // edi
  unsigned int v148; // eax
  __int64 *v149; // rdx
  __int64 v150; // rsi
  _QWORD *v151; // rax
  int v152; // r8d
  char *v153; // r9
  unsigned __int64 v154; // rsi
  unsigned __int8 v155; // dl
  _QWORD *v156; // rbx
  __m128i *v157; // rcx
  __int64 v158; // rdx
  __int64 v159; // rsi
  __int64 *v160; // rax
  __int64 v161; // rdx
  _QWORD *v162; // rsi
  __int64 v163; // rdx
  __int64 v164; // rsi
  __int64 v165; // rsi
  _QWORD *v166; // r13
  _QWORD *v167; // r15
  __int64 v168; // rdi
  __int64 v169; // rcx
  char v170; // al
  __int64 v171; // rdx
  __int64 v172; // rsi
  unsigned __int64 v173; // rax
  unsigned __int8 v174; // cl
  _QWORD *v175; // r13
  _QWORD *v176; // r15
  __int64 v177; // rdi
  __int64 v178; // rcx
  char v179; // al
  __int64 v180; // rdx
  __int64 v181; // rsi
  unsigned __int64 v182; // rax
  unsigned __int8 v183; // cl
  _QWORD *v184; // r13
  _QWORD *v185; // r15
  __int64 v186; // rdi
  __int64 v187; // rcx
  char v188; // al
  __int64 v189; // rdx
  __int64 v190; // rsi
  unsigned __int64 v191; // rax
  unsigned __int8 v192; // cl
  __int64 v193; // rcx
  _QWORD *v194; // r13
  __int64 v195; // rdi
  _QWORD *v196; // r15
  char v197; // al
  __int64 v198; // rdx
  unsigned __int64 v199; // rax
  unsigned __int8 v200; // cl
  __m128i *v201; // rdx
  __m128i *v202; // r8
  int v203; // esi
  unsigned int v204; // ecx
  __int64 *v205; // rax
  __int64 v206; // r10
  unsigned __int32 v207; // eax
  _QWORD **v208; // r15
  unsigned __int64 v209; // rsi
  _QWORD *v210; // r12
  char v211; // al
  __int64 v212; // rdx
  unsigned __int8 v213; // cl
  __m128i *v214; // rdx
  __m128i *v215; // r8
  int v216; // esi
  __int64 v217; // rdi
  unsigned int v218; // ecx
  __int64 *v219; // rax
  __int64 v220; // r10
  unsigned __int32 v221; // eax
  __m128i *v222; // r8
  int v223; // esi
  __int64 v224; // rdi
  unsigned int v225; // ecx
  __int64 v226; // r10
  int v227; // eax
  int v228; // r9d
  __m128i *v229; // r8
  int v230; // esi
  __int64 v231; // rdi
  unsigned int v232; // ecx
  __int64 v233; // r10
  int v234; // eax
  int v235; // r9d
  __m128i *v236; // rdi
  int v237; // edx
  unsigned int v238; // eax
  __int64 *v239; // rcx
  _QWORD *v240; // r10
  int v241; // edx
  __int32 v242; // esi
  __int32 v243; // esi
  __int32 v244; // esi
  __int32 v245; // esi
  __m128i *v246; // r9
  int v247; // r8d
  __int64 **v248; // rsi
  unsigned int v249; // r11d
  int v250; // edx
  __int64 *v251; // rdi
  _QWORD *v252; // rax
  int v253; // r8d
  char *v254; // r9
  __int64 v255; // rax
  unsigned int v256; // eax
  __int64 *v257; // r11
  __int64 v258; // rax
  unsigned int v259; // ecx
  __int64 v260; // rdi
  int v261; // ecx
  unsigned int v262; // ebx
  __int64 v263; // rdi
  __int64 v264; // rdi
  unsigned __int16 v265; // ax
  unsigned __int8 v266; // r15
  __int32 v267; // eax
  unsigned int v268; // ebx
  __int64 v269; // rdx
  unsigned __int32 v270; // edx
  __int32 v271; // eax
  __int32 v272; // eax
  unsigned int v273; // eax
  _QWORD *v274; // rax
  char v275; // cl
  unsigned int v276; // ecx
  int v277; // r9d
  unsigned int v278; // edx
  unsigned int j; // r8d
  _QWORD *v280; // rsi
  __int64 v281; // r9
  int k; // r8d
  _QWORD *v283; // rsi
  __int64 v284; // rax
  int v285; // ecx
  int v286; // r8d
  int v287; // eax
  int v288; // r9d
  int v289; // eax
  int v290; // r9d
  int v291; // edx
  int v292; // r9d
  int v293; // r8d
  __int64 v294; // rax
  unsigned int v295; // edx
  unsigned int v296; // r8d
  __int64 v297; // [rsp+18h] [rbp-3B8h]
  __int64 v298; // [rsp+20h] [rbp-3B0h]
  unsigned __int64 v299; // [rsp+30h] [rbp-3A0h]
  unsigned int v300; // [rsp+38h] [rbp-398h]
  _BYTE *v301; // [rsp+40h] [rbp-390h]
  unsigned __int8 v303; // [rsp+54h] [rbp-37Ch]
  int v304; // [rsp+54h] [rbp-37Ch]
  __int64 v306; // [rsp+68h] [rbp-368h]
  __int64 v307; // [rsp+70h] [rbp-360h]
  bool v308; // [rsp+78h] [rbp-358h]
  _QWORD *v309; // [rsp+80h] [rbp-350h]
  bool v310; // [rsp+80h] [rbp-350h]
  _QWORD *v311; // [rsp+88h] [rbp-348h]
  unsigned __int64 v312; // [rsp+90h] [rbp-340h]
  __int64 v313; // [rsp+90h] [rbp-340h]
  __int64 v314; // [rsp+90h] [rbp-340h]
  __m128i *v315; // [rsp+90h] [rbp-340h]
  __int64 v316; // [rsp+98h] [rbp-338h]
  _QWORD *v317; // [rsp+98h] [rbp-338h]
  char v318; // [rsp+98h] [rbp-338h]
  _QWORD *v319; // [rsp+98h] [rbp-338h]
  __int64 v320; // [rsp+98h] [rbp-338h]
  __int64 v321; // [rsp+A0h] [rbp-330h]
  char v322; // [rsp+A0h] [rbp-330h]
  _QWORD *v323; // [rsp+A0h] [rbp-330h]
  unsigned int v324; // [rsp+A0h] [rbp-330h]
  int v325; // [rsp+A8h] [rbp-328h]
  __int64 v326; // [rsp+A8h] [rbp-328h]
  __int64 v327; // [rsp+A8h] [rbp-328h]
  __int64 v328; // [rsp+A8h] [rbp-328h]
  __int64 v329; // [rsp+A8h] [rbp-328h]
  __int64 v330; // [rsp+A8h] [rbp-328h]
  _QWORD *v331; // [rsp+A8h] [rbp-328h]
  unsigned __int64 m; // [rsp+B8h] [rbp-318h]
  __int64 *v334; // [rsp+B8h] [rbp-318h]
  __int64 v335; // [rsp+B8h] [rbp-318h]
  __int64 v336; // [rsp+B8h] [rbp-318h]
  __int64 v337; // [rsp+B8h] [rbp-318h]
  __int64 v338; // [rsp+B8h] [rbp-318h]
  __int64 v339; // [rsp+B8h] [rbp-318h]
  __int64 v340; // [rsp+B8h] [rbp-318h]
  __m128i v341; // [rsp+C0h] [rbp-310h] BYREF
  __m128i v342; // [rsp+D0h] [rbp-300h] BYREF
  __int64 v343; // [rsp+E0h] [rbp-2F0h]
  int v344; // [rsp+F4h] [rbp-2DCh] BYREF
  __int64 v345; // [rsp+F8h] [rbp-2D8h] BYREF
  __int64 v346; // [rsp+100h] [rbp-2D0h] BYREF
  __int64 v347; // [rsp+108h] [rbp-2C8h] BYREF
  __int64 v348; // [rsp+110h] [rbp-2C0h] BYREF
  unsigned int v349; // [rsp+118h] [rbp-2B8h]
  __int64 v350; // [rsp+120h] [rbp-2B0h] BYREF
  unsigned int v351; // [rsp+128h] [rbp-2A8h]
  __int64 v352[2]; // [rsp+130h] [rbp-2A0h] BYREF
  _QWORD *v353; // [rsp+140h] [rbp-290h] BYREF
  __int32 v354; // [rsp+148h] [rbp-288h]
  __int64 v355; // [rsp+150h] [rbp-280h] BYREF
  __int64 v356; // [rsp+158h] [rbp-278h]
  __int64 v357; // [rsp+160h] [rbp-270h]
  unsigned int v358; // [rsp+168h] [rbp-268h]
  __int64 v359; // [rsp+170h] [rbp-260h] BYREF
  __int64 *v360; // [rsp+178h] [rbp-258h]
  __int64 v361; // [rsp+180h] [rbp-250h]
  unsigned int v362; // [rsp+188h] [rbp-248h]
  __m128i v363; // [rsp+190h] [rbp-240h] BYREF
  __m128i v364; // [rsp+1A0h] [rbp-230h] BYREF
  __m128i *v365; // [rsp+1B0h] [rbp-220h]
  __m128i v366[3]; // [rsp+1C0h] [rbp-210h] BYREF
  __m128i v367; // [rsp+1F0h] [rbp-1E0h] BYREF
  __int64 v368; // [rsp+200h] [rbp-1D0h]
  __int64 v369; // [rsp+208h] [rbp-1C8h]
  __int64 v370; // [rsp+210h] [rbp-1C0h]
  __m128i v371; // [rsp+220h] [rbp-1B0h] BYREF
  __int64 v372; // [rsp+230h] [rbp-1A0h]
  __int64 v373; // [rsp+238h] [rbp-198h]
  __int64 v374; // [rsp+240h] [rbp-190h]
  __m128i v375; // [rsp+250h] [rbp-180h] BYREF
  __m128i v376; // [rsp+260h] [rbp-170h] BYREF
  __int64 v377; // [rsp+270h] [rbp-160h]
  __m128i *v378; // [rsp+278h] [rbp-158h]
  __m128i v379; // [rsp+280h] [rbp-150h] BYREF
  __m128i v380; // [rsp+290h] [rbp-140h] BYREF
  __m128i *v381; // [rsp+2A0h] [rbp-130h]
  __m128i *v382; // [rsp+2A8h] [rbp-128h]
  __int64 v383; // [rsp+2B0h] [rbp-120h]
  _QWORD *v384; // [rsp+310h] [rbp-C0h] BYREF
  __int64 v385; // [rsp+318h] [rbp-B8h]
  _BYTE v386[176]; // [rsp+320h] [rbp-B0h] BYREF

  v5 = (__int64)a5;
  v309 = (_QWORD *)i;
  v8 = sub_157EB90(a1);
  v9 = sub_1632FA0(v8);
  v10 = *(_QWORD *)(a1 + 48);
  v303 = 0;
  v301 = (_BYTE *)v9;
  v355 = 0;
  v356 = 0;
  v357 = 0;
  v358 = 0;
  v359 = 0;
  v360 = 0;
  v361 = 0;
  v362 = 0;
  v345 = v10;
  v306 = a1 + 40;
  v307 = 1;
  v299 = 0;
  if ( v10 != a1 + 40 )
  {
    while ( 1 )
    {
      i = (__int64)a5;
      if ( v10 )
        v10 -= 24;
      v11 = sub_140B650(v10, a5);
      v12 = (__int64 *)v11;
      if ( !v11 )
      {
        v44 = v345;
        i = v358;
        v45 = v345 - 24;
        v335 = v345 - 24;
        v345 = *(_QWORD *)(v345 + 8);
        if ( !v358 )
        {
          ++v355;
          goto LABEL_367;
        }
        v46 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
        LODWORD(v47) = (v358 - 1) & v46;
        v48 = (__int64 *)(v356 + 16LL * (unsigned int)v47);
        v49 = *v48;
        if ( v45 == *v48 )
          goto LABEL_60;
        v127 = 0;
        v128 = 1;
        while ( 2 )
        {
          if ( v49 == -8 )
          {
            if ( !v127 )
              v127 = v48;
            ++v355;
            v129 = v357 + 1;
            if ( 4 * ((int)v357 + 1) < 3 * v358 )
            {
              if ( v358 - HIDWORD(v357) - v129 <= v358 >> 3 )
              {
                sub_18F61D0((__int64)&v355, v358);
                if ( !v358 )
                {
LABEL_497:
                  LODWORD(v357) = v357 + 1;
                  BUG();
                }
                i = v358 - 1;
                v261 = 1;
                v262 = i & v46;
                v129 = v357 + 1;
                v127 = (__int64 *)(v356 + 16LL * v262);
                v263 = *v127;
                if ( v335 != *v127 )
                {
                  while ( 2 )
                  {
                    if ( v263 == -8 )
                      goto LABEL_385;
                    if ( !v12 && v263 == -16 )
                      v12 = v127;
                    v262 = i & (v261 + v262);
                    v127 = (__int64 *)(v356 + 16LL * v262);
                    v263 = *v127;
                    if ( v335 != *v127 )
                    {
                      ++v261;
                      continue;
                    }
                    break;
                  }
                }
              }
              goto LABEL_203;
            }
LABEL_367:
            i = 2 * v358;
            sub_18F61D0((__int64)&v355, i);
            if ( !v358 )
              goto LABEL_497;
            v259 = (v358 - 1) & (((unsigned int)v335 >> 9) ^ ((unsigned int)v335 >> 4));
            v129 = v357 + 1;
            v127 = (__int64 *)(v356 + 16LL * v259);
            v260 = *v127;
            if ( v335 != *v127 )
            {
              for ( i = 1; ; i = (unsigned int)(i + 1) )
              {
                if ( v260 == -8 )
                {
LABEL_385:
                  if ( v12 )
                    v127 = v12;
                  break;
                }
                if ( v260 == -16 && !v12 )
                  v12 = v127;
                v259 = (v358 - 1) & (i + v259);
                v127 = (__int64 *)(v356 + 16LL * v259);
                v260 = *v127;
                if ( v335 == *v127 )
                  break;
              }
            }
LABEL_203:
            LODWORD(v357) = v129;
            if ( *v127 != -8 )
              --HIDWORD(v357);
            *v127 = v335;
            v127[1] = v307;
          }
          else
          {
            if ( !v127 && v49 == -16 )
              v127 = v48;
            v47 = (v358 - 1) & ((_DWORD)v47 + v128);
            v48 = (__int64 *)(v356 + 16 * v47);
            v49 = *v48;
            if ( v335 != *v48 )
            {
              ++v128;
              continue;
            }
          }
          break;
        }
LABEL_60:
        if ( sub_15F3330(v335) )
        {
          v10 = v345;
          v299 = v307++;
          goto LABEL_45;
        }
        i = (__int64)a5;
        if ( !sub_18F47E0(v335, a5) )
          goto LABEL_97;
        if ( *(_BYTE *)(v44 - 8) != 55 )
          goto LABEL_66;
        v52 = *(_QWORD *)(v44 - 72);
        v53 = *(_BYTE *)(v52 + 16);
        if ( v53 != 54 )
        {
LABEL_64:
          if ( v53 > 0x10u || !sub_1593BB0(v52, i, v50, v51) )
            goto LABEL_66;
          if ( *(_BYTE *)(v44 - 8) == 55 )
          {
            v130 = *(_WORD *)(v44 - 6);
            if ( ((v130 >> 7) & 6) == 0 && (v130 & 1) == 0 )
              goto LABEL_209;
          }
          else if ( (unsigned __int8)sub_18F2D80(v335) )
          {
LABEL_209:
            v131 = sub_14AD280(*(_QWORD *)(v44 - 48), (unsigned __int64)v301, 6u);
            v132 = v131;
            if ( *(_BYTE *)(v131 + 16) > 0x17u )
            {
              if ( (unsigned __int8)sub_140B100(v131, a5, 0) )
              {
                v133 = sub_18F4CD0(v132, v335, v309);
                if ( v133 )
                {
LABEL_212:
                  i = (__int64)&v345;
                  sub_18F35E0(v335, &v345, a3, (__int64)a5, (__int64)&v359, (__int64)&v355, 0);
                  v303 = v133;
                  v10 = v345;
                  ++v307;
                  goto LABEL_45;
                }
              }
            }
          }
LABEL_66:
          i = v335;
          v54 = sub_141C430(a3, v335, 1u);
          if ( (unsigned int)(v54 & 7) - 1 > 1 )
            goto LABEL_97;
          i = v335;
          if ( *(_BYTE *)(v44 - 8) == 55 )
            sub_141EDF0(&v363, v335);
          else
            sub_18F2E70(&v363, v335);
          if ( !v363.m128i_i64[0] )
            goto LABEL_97;
          v55 = sub_1414210(a3);
          v56 = v54;
          v344 = v55;
          while ( 2 )
          {
            v65 = v56 & 7;
            if ( v65 != 2 && v65 != 1 )
              goto LABEL_97;
            i = (__int64)a5;
            v66 = v56 & 0xFFFFFFFFFFFFFFF8LL;
            v67 = sub_18F47E0(v56 & 0xFFFFFFFFFFFFFFF8LL, a5);
            if ( !v67 )
              goto LABEL_97;
            i = v66;
            if ( *(_BYTE *)(v66 + 16) == 55 )
              sub_141EDF0(v366, v66);
            else
              sub_18F2E70(v366, v66);
            if ( !v366[0].m128i_i64[0] )
              goto LABEL_97;
            if ( v358 )
            {
              v57 = v358 - 1;
              v58 = v57 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
              v59 = (__int64 *)(v356 + 16LL * v58);
              v60 = *v59;
              if ( *v59 == v66 )
              {
LABEL_75:
                i = v299;
                if ( v59[1] > v299 )
                  goto LABEL_76;
              }
              else
              {
                v68 = 1;
                while ( v60 != -8 )
                {
                  v293 = v68 + 1;
                  v294 = (unsigned int)v57 & (v58 + v68);
                  v58 = v294;
                  v59 = (__int64 *)(v356 + 16 * v294);
                  v60 = *v59;
                  if ( v66 == *v59 )
                    goto LABEL_75;
                  v68 = v293;
                }
              }
            }
            i = (__int64)v301;
            v69 = sub_14AD280(v366[0].m128i_i64[0], (unsigned __int64)v301, 6u);
            if ( *(_BYTE *)(v69 + 16) != 53 )
            {
              i = (__int64)a5;
              v316 = v69;
              if ( !(unsigned __int8)sub_140B1C0(v69, a5, 0) )
                goto LABEL_97;
              i = 0;
              if ( (unsigned __int8)sub_139D0F0(v316, 0) )
                goto LABEL_97;
            }
LABEL_76:
            if ( *(_BYTE *)(v66 + 16) != 55 )
            {
              if ( (unsigned __int8)sub_18F2D80(v66) )
                goto LABEL_79;
              goto LABEL_82;
            }
            v61 = *(_WORD *)(v66 + 18);
            if ( ((v61 >> 7) & 6) != 0 || (v61 & 1) != 0 )
              goto LABEL_82;
LABEL_79:
            sub_18F2CB0(&v367, v335);
            if ( v367.m128i_i64[0] )
            {
              i = (__int64)&v367;
              if ( (unsigned __int8)sub_134CB50((__int64)v309, (__int64)&v367, (__int64)&v363) )
              {
                if ( *(_BYTE *)(v44 - 8) != 78 )
                  goto LABEL_82;
                v255 = *(_QWORD *)(v44 - 48);
                if ( *(_BYTE *)(v255 + 16) )
                  goto LABEL_82;
                if ( (*(_BYTE *)(v255 + 33) & 0x20) == 0 )
                  goto LABEL_82;
                if ( (unsigned int)(*(_DWORD *)(v255 + 36) - 133) > 1 )
                  goto LABEL_82;
                i = v66;
                sub_18F2CB0(&v371, v66);
                if ( !v371.m128i_i64[0] )
                  goto LABEL_82;
                v379.m128i_i64[0] = v371.m128i_i64[0];
                i = (__int64)&v375;
                v379.m128i_i64[1] = 1;
                v380 = 0u;
                v381 = 0;
                v375.m128i_i64[0] = v367.m128i_i64[0];
                v375.m128i_i64[1] = 1;
                v376 = 0u;
                v377 = 0;
                if ( (unsigned __int8)sub_134CB50((__int64)v309, (__int64)&v375, (__int64)&v379) != 3 )
                  goto LABEL_82;
              }
            }
            i = (__int64)v366;
            v256 = sub_18F5A80(
                     v363.m128i_i64,
                     v366[0].m128i_i64,
                     (unsigned __int64)v301,
                     (__int64)a5,
                     &v347,
                     &v346,
                     v66,
                     (__int64)&v359,
                     (__int64)v309,
                     *(_QWORD *)(a1 + 56));
            v257 = &v347;
            v57 = v256;
            switch ( v256 )
            {
              case 1u:
                sub_18F35E0(v66, &v345, a3, (__int64)a5, (__int64)&v359, (__int64)&v355, 0);
                i = v335;
                v303 = v67;
                v56 = sub_141C430(a3, v335, 1u);
                continue;
              case 2u:
                if ( sub_18F2D30(v66) )
                  goto LABEL_365;
                goto LABEL_82;
              case 0u:
                if ( *(_BYTE *)(v66 + 16) == 78 )
                {
                  v258 = *(_QWORD *)(v66 - 24);
                  if ( !*(_BYTE *)(v258 + 16)
                    && (*(_BYTE *)(v258 + 33) & 0x20) != 0
                    && (unsigned int)(*(_DWORD *)(v258 + 36) - 137) <= 1 )
                  {
LABEL_365:
                    i = (__int64)v257;
                    v379.m128i_i64[0] = v366[0].m128i_i64[1];
                    v303 |= sub_18F3020(v66, v257, &v379, v346, v363.m128i_i64[1], v57 == 2);
                  }
                }
LABEL_82:
                v62 = *(_QWORD *)(a1 + 48);
                if ( v62 && v66 == v62 - 24 )
                  goto LABEL_97;
                v63 = _mm_loadu_si128(&v363);
                i = v66;
                v64 = _mm_loadu_si128(&v364);
                LOBYTE(v382) = 1;
                v379 = v63;
                v381 = v365;
                v380 = v64;
                if ( (sub_13575E0(v309, v66, &v379, v57) & 1) != 0 )
                  goto LABEL_97;
                i = (__int64)&v363;
                v56 = sub_141C340(a3, &v363, 0, (_QWORD *)(v66 + 24), a1, 0, &v344, 1u);
                continue;
            }
          }
          if ( !byte_4FAE400 )
            goto LABEL_82;
          if ( v256 != 3 )
            goto LABEL_82;
          if ( *(_BYTE *)(v66 + 16) != 55 )
            goto LABEL_82;
          if ( *(_BYTE *)(v44 - 8) != 55 )
            goto LABEL_82;
          if ( *(_BYTE *)(*(_QWORD *)(v66 - 48) + 16LL) != 13 )
            goto LABEL_82;
          if ( *(_BYTE *)(*(_QWORD *)(v44 - 72) + 16LL) != 13 )
            goto LABEL_82;
          i = v335;
          v266 = sub_18F4CD0(v66, v335, v309);
          if ( !v266 )
            goto LABEL_82;
          sub_13A38D0((__int64)&v348, *(_QWORD *)(v66 - 48) + 24LL);
          sub_13A38D0((__int64)&v350, *(_QWORD *)(v44 - 72) + 24LL);
          v324 = v351;
          sub_16A5C50((__int64)&v379, (const void **)&v350, v349);
          if ( v351 > 0x40 && v350 )
            j_j___libc_free_0_0(v350);
          v350 = v379.m128i_i64[0];
          v267 = v379.m128i_i32[2];
          v379.m128i_i32[2] = 0;
          v351 = v267;
          sub_135E100(v379.m128i_i64);
          v268 = 8 * (v346 - v347);
          if ( *v301 )
            v268 = v349 - v324 - v268;
          sub_18F4C10((__int64)v352, v349, v268, v268 + v324);
          sub_13A38D0((__int64)&v379, (__int64)&v350);
          if ( v379.m128i_i32[2] > 0x40u )
          {
            sub_16A7DC0(v379.m128i_i64, v268);
          }
          else
          {
            if ( v268 == v379.m128i_i32[2] )
              v269 = 0;
            else
              v269 = v379.m128i_i64[0] << v268;
            v379.m128i_i64[0] = v269 & (0xFFFFFFFFFFFFFFFFLL >> -v379.m128i_i8[8]);
          }
          sub_13A38D0((__int64)&v367, (__int64)v352);
          sub_13D0570((__int64)&v367);
          v270 = v367.m128i_u32[2];
          v367.m128i_i32[2] = 0;
          v371.m128i_i32[2] = v270;
          v371.m128i_i64[0] = v367.m128i_i64[0];
          if ( v270 > 0x40 )
            sub_16A8890(v371.m128i_i64, &v348);
          else
            v371.m128i_i64[0] = v348 & v367.m128i_i64[0];
          v271 = v371.m128i_i32[2];
          v371.m128i_i32[2] = 0;
          v375.m128i_i32[2] = v271;
          v375.m128i_i64[0] = v371.m128i_i64[0];
          if ( v379.m128i_i32[2] > 0x40u )
            sub_16A89F0(v379.m128i_i64, v375.m128i_i64);
          else
            v379.m128i_i64[0] |= v371.m128i_i64[0];
          v272 = v379.m128i_i32[2];
          v379.m128i_i32[2] = 0;
          v354 = v272;
          v353 = (_QWORD *)v379.m128i_i64[0];
          sub_135E100(v375.m128i_i64);
          sub_135E100(v371.m128i_i64);
          sub_135E100(v367.m128i_i64);
          sub_135E100(v379.m128i_i64);
          v297 = sub_15A1070(**(_QWORD **)(v66 - 48), (__int64)&v353);
          v273 = *(unsigned __int16 *)(v66 + 18);
          v298 = *(_QWORD *)(v66 - 24);
          v304 = (v273 >> 7) & 7;
          v300 = 1 << (v273 >> 1) >> 1;
          v318 = *(_BYTE *)(v66 + 56);
          v274 = sub_1648A60(64, 2u);
          if ( v274 )
          {
            v275 = v318;
            v319 = v274;
            sub_15F9480((__int64)v274, v297, v298, 0, v300, v304, v275, v66);
            v274 = v319;
          }
          v320 = (__int64)v274;
          v375.m128i_i64[0] = 0x100000000LL;
          v375.m128i_i64[1] = 0x800000007LL;
          v376.m128i_i32[0] = 9;
          sub_15F4370((__int64)v274, v66, v375.m128i_i32, 5);
          if ( v358 )
          {
            v276 = v358 - 1;
            v277 = 1;
            v278 = (v358 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
            for ( j = v278; ; j = v276 & v296 )
            {
              v280 = (_QWORD *)(v356 + 16LL * j);
              if ( v66 == *v280 )
                break;
              if ( *v280 == -8 )
              {
                v281 = 0;
                goto LABEL_435;
              }
              v296 = v277 + j;
              ++v277;
            }
            v281 = v280[1];
LABEL_435:
            for ( k = 1; ; ++k )
            {
              v283 = (_QWORD *)(v356 + 16LL * v278);
              if ( v66 == *v283 )
                break;
              if ( *v283 == -8 )
                goto LABEL_438;
              v295 = k + v278;
              v278 = v276 & v295;
            }
            *v283 = -16;
            LODWORD(v357) = v357 - 1;
            ++HIDWORD(v357);
          }
          else
          {
            v281 = 0;
          }
LABEL_438:
          v371.m128i_i64[1] = v281;
          v371.m128i_i64[0] = v320;
          sub_18F6390((__int64)&v379, (__int64)&v355, v371.m128i_i64, &v371.m128i_i64[1]);
          sub_18F35E0(v335, &v345, a3, (__int64)a5, (__int64)&v359, (__int64)&v355, 0);
          i = (__int64)&v345;
          sub_18F35E0(v66, &v345, a3, (__int64)a5, (__int64)&v359, (__int64)&v355, 0);
          sub_135E100((__int64 *)&v353);
          sub_135E100(v352);
          sub_135E100(&v350);
          sub_135E100(&v348);
          v303 = v266;
LABEL_97:
          v10 = v345;
          ++v307;
          goto LABEL_45;
        }
        if ( *(_QWORD *)(v44 - 48) != *(_QWORD *)(v52 - 24) )
        {
          sub_141EDF0(&v379, v335);
          sub_141EB40(&v375, (__int64 *)v52);
          i = (__int64)&v375;
          if ( (unsigned __int8)sub_134CB50((__int64)v309, (__int64)&v375, (__int64)&v379) != 3 )
          {
LABEL_218:
            v52 = *(_QWORD *)(v44 - 72);
            v53 = *(_BYTE *)(v52 + 16);
            goto LABEL_64;
          }
        }
        if ( *(_BYTE *)(v44 - 8) == 55 )
        {
          v265 = *(_WORD *)(v44 - 6);
          v50 = (v265 >> 7) & 6;
          if ( ((v265 >> 7) & 6) != 0 || (v265 & 1) != 0 )
            goto LABEL_218;
        }
        else if ( !(unsigned __int8)sub_18F2D80(v335) )
        {
          goto LABEL_218;
        }
        i = v335;
        v133 = sub_18F4CD0(v52, v335, v309);
        if ( v133 )
          goto LABEL_212;
        goto LABEL_218;
      }
      v13 = v11;
      v14 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      v367.m128i_i64[1] = -1;
      v368 = 0;
      v367.m128i_i64[0] = v14;
      v369 = 0;
      v370 = 0;
      v379.m128i_i64[0] = (__int64)&v380;
      v380.m128i_i64[0] = v12[5];
      v379.m128i_i64[1] = 0x1000000001LL;
      v15 = sub_15F2050(v13);
      v312 = sub_1632FA0(v15);
      v16 = v379.m128i_u32[2];
      if ( !v379.m128i_i32[2] )
        goto LABEL_42;
      v308 = 0;
      do
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(v379.m128i_i64[0] + 8LL * v16 - 8);
          v379.m128i_i32[2] = v16 - 1;
          v19 = v18;
          v321 = v18;
          v20 = sub_157EBA0(v18);
          if ( v19 == v12[5] )
            v20 = (unsigned __int64)v12;
          i = (__int64)&v367;
          for ( m = sub_141C340(a3, &v367, 0, (_QWORD *)(v20 + 24), v19, 0, 0, 1u);
                ;
                m = sub_141C340(a3, &v367, 0, v375.m128i_i64[0], v321, 0, 0, 1u) )
          {
            v21 = m & 7;
            v325 = v21;
            if ( v21 == 2 )
            {
              v22 = m & 0xFFFFFFFFFFFFFFF8LL;
            }
            else
            {
              if ( v21 != 1 )
                break;
              v22 = m & 0xFFFFFFFFFFFFFFF8LL;
            }
            i = (__int64)a5;
            v23 = sub_18F47E0(v22, a5);
            if ( !v23 )
              break;
            if ( *(_BYTE *)(v22 + 16) == 55 )
            {
              v17 = *(_WORD *)(v22 + 18);
              if ( ((v17 >> 7) & 6) != 0 || (v17 & 1) != 0 )
                break;
              sub_141EDF0(&v375, v22);
            }
            else
            {
              if ( !(unsigned __int8)sub_18F2D80(v22) )
                break;
              sub_18F2E70(&v375, v22);
            }
            i = (__int64)&v371;
            v24 = sub_14AD280(v375.m128i_i64[0], v312, 6u);
            v25 = v12[-3 * (*((_DWORD *)v12 + 5) & 0xFFFFFFF)];
            v375.m128i_i64[0] = v24;
            v375.m128i_i64[1] = 1;
            v376 = 0u;
            v377 = 0;
            v371.m128i_i64[0] = v25;
            v371.m128i_i64[1] = 1;
            v372 = 0;
            v373 = 0;
            v374 = 0;
            if ( (unsigned __int8)sub_134CB50((__int64)v309, (__int64)&v371, (__int64)&v375) != 3 )
              break;
            v375.m128i_i64[0] = v22 + 24;
            sub_18F35E0(v22, v375.m128i_i64, a3, (__int64)a5, (__int64)&v359, (__int64)&v355, 0);
            i = (__int64)&v367;
            v308 = v23;
          }
          if ( v325 == 3 && m >> 61 == 1 )
          {
            v26 = *(_QWORD *)(v321 + 8);
            if ( v26 )
              break;
          }
LABEL_10:
          v16 = v379.m128i_u32[2];
          if ( !v379.m128i_i32[2] )
            goto LABEL_41;
        }
        while ( 1 )
        {
          v27 = sub_1648700(v26);
          i = *((unsigned __int8 *)v27 + 16);
          if ( (unsigned __int8)(i - 25) <= 9u )
            break;
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_10;
        }
        v334 = v12;
        while ( 1 )
        {
          v28 = v27[5];
          if ( v321 == v28 )
            goto LABEL_29;
          v29 = sub_157EBA0(v27[5]);
          v30 = sub_15F4D60(v29);
          if ( v30 != 1 )
            goto LABEL_29;
          i = a4;
          v32 = *(unsigned int *)(a4 + 48);
          if ( !(_DWORD)v32 )
            goto LABEL_29;
          v33 = *(_QWORD *)(a4 + 32);
          i = ((_DWORD)v32 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v34 = (__int64 *)(v33 + 16 * i);
          v35 = *v34;
          if ( *v34 != v28 )
          {
            while ( v35 != -8 )
            {
              v31 = v30 + 1;
              i = ((_DWORD)v32 - 1) & (unsigned int)(i + v30);
              v34 = (__int64 *)(v33 + 16 * i);
              v35 = *v34;
              if ( v28 == *v34 )
                goto LABEL_35;
              ++v30;
            }
            goto LABEL_29;
          }
LABEL_35:
          if ( v34 == (__int64 *)(v33 + 16 * v32) || !v34[1] )
          {
LABEL_29:
            v26 = *(_QWORD *)(v26 + 8);
            if ( !v26 )
              break;
            goto LABEL_30;
          }
          v36 = v379.m128i_u32[2];
          if ( v379.m128i_i32[2] >= (unsigned __int32)v379.m128i_i32[3] )
          {
            i = (__int64)&v380;
            sub_16CD150((__int64)&v379, &v380, 0, 8, v31, v35);
            v36 = v379.m128i_u32[2];
          }
          *(_QWORD *)(v379.m128i_i64[0] + 8 * v36) = v28;
          ++v379.m128i_i32[2];
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            break;
LABEL_30:
          v27 = sub_1648700(v26);
          if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) > 9u )
            goto LABEL_29;
        }
        v16 = v379.m128i_u32[2];
        v12 = v334;
      }
      while ( v379.m128i_i32[2] );
LABEL_41:
      v303 |= v308;
LABEL_42:
      if ( (__m128i *)v379.m128i_i64[0] != &v380 )
        _libc_free(v379.m128i_u64[0]);
      v10 = *(_QWORD *)(v345 + 8);
      v345 = v10;
LABEL_45:
      if ( v306 == v10 )
      {
        v5 = (__int64)a5;
        break;
      }
    }
  }
  if ( byte_4FAE4E0 )
  {
    if ( (_DWORD)v361 )
    {
      v110 = v360;
      v111 = &v360[7 * v362];
      if ( v360 != v111 )
      {
        while ( 1 )
        {
          v112 = v110;
          if ( *v110 != -16 && *v110 != -8 )
            break;
          v110 += 7;
          if ( v111 == v110 )
            goto LABEL_48;
        }
        v322 = 0;
        if ( v111 != v110 )
        {
          v326 = v5;
          do
          {
            v113 = *v112;
            v381 = &v380;
            v380.m128i_i32[0] = 0;
            v379.m128i_i64[0] = v113;
            v380.m128i_i64[1] = 0;
            v382 = &v380;
            v383 = 0;
            v114 = (const __m128i *)v112[3];
            if ( v114 )
            {
              v115 = sub_18F2AD0(v114, (__int64)&v380);
              v116 = v115;
              do
              {
                v117 = v115;
                v115 = (__m128i *)v115[1].m128i_i64[0];
              }
              while ( v115 );
              v381 = v117;
              v118 = v116;
              do
              {
                v119 = v118;
                v118 = (__m128i *)v118[1].m128i_i64[1];
              }
              while ( v118 );
              v382 = v119;
              v120 = v112[6];
              v380.m128i_i64[1] = (__int64)v116;
              v113 = v379.m128i_i64[0];
              v383 = v120;
            }
            if ( *(_BYTE *)(v113 + 16) == 55 )
              sub_141EDF0(&v375, v113);
            else
              sub_18F2E70(&v375, v113);
            v121 = (unsigned __int16 *)sub_1649C60(v375.m128i_i64[0]);
            i = (__int64)&v367;
            v367.m128i_i64[0] = 0;
            v371.m128i_i64[0] = v375.m128i_i64[1];
            sub_14AC610(v121, v367.m128i_i64, (__int64)v301);
            if ( !v383 )
              goto LABEL_225;
            if ( !sub_18F2D30(v113) )
              goto LABEL_504;
            v122 = sub_220EF80(&v380);
            v123 = *(_QWORD *)(v122 + 40);
            if ( v123 <= v367.m128i_i64[0] )
              goto LABEL_504;
            v124 = *(_QWORD *)(v122 + 32);
            if ( v124 < v371.m128i_i64[0] + v367.m128i_i64[0] || v123 >= v371.m128i_i64[0] + v367.m128i_i64[0] )
              goto LABEL_504;
            i = (__int64)&v367;
            v314 = v122;
            v135 = sub_18F3020(v113, &v367, &v371, v123, v124 - v123, 1u);
            if ( v135 )
            {
              v322 = v135;
              v284 = sub_220F330(v314, &v380);
              i = 48;
              j_j___libc_free_0(v284, 48);
              v136 = --v383;
            }
            else
            {
              v136 = v383;
            }
            if ( v136 )
            {
LABEL_504:
              if ( *(_BYTE *)(v113 + 16) == 78 )
              {
                v139 = *(_QWORD *)(v113 - 24);
                if ( !*(_BYTE *)(v139 + 16)
                  && (*(_BYTE *)(v139 + 33) & 0x20) != 0
                  && (unsigned int)(*(_DWORD *)(v139 + 36) - 137) <= 1 )
                {
                  v140 = v381[2].m128i_i64[1];
                  v141 = v381[2].m128i_i64[0];
                  if ( v140 <= v367.m128i_i64[0] && v141 > v367.m128i_i64[0] )
                  {
                    i = (__int64)&v367;
                    v315 = v381;
                    v142 = sub_18F3020(v113, &v367, &v371, v140, v141 - v140, 0);
                    if ( v142 )
                    {
                      v143 = sub_220F330(v315, &v380);
                      i = 48;
                      j_j___libc_free_0(v143, 48);
                      --v383;
                      v322 = v142;
                    }
                  }
                }
              }
              v125 = v380.m128i_i64[1];
              while ( v125 )
              {
                sub_18F3410(*(_QWORD *)(v125 + 24));
                v126 = v125;
                v125 = *(_QWORD *)(v125 + 16);
                i = 48;
                j_j___libc_free_0(v126, 48);
              }
            }
            else
            {
LABEL_225:
              v137 = v380.m128i_i64[1];
              while ( v137 )
              {
                sub_18F3410(*(_QWORD *)(v137 + 24));
                v138 = v137;
                v137 = *(_QWORD *)(v137 + 16);
                i = 48;
                j_j___libc_free_0(v138, 48);
              }
            }
            v112 += 7;
            if ( v112 == v111 )
              break;
            while ( *v112 == -16 || *v112 == -8 )
            {
              v112 += 7;
              if ( v111 == v112 )
                goto LABEL_196;
            }
          }
          while ( v112 != v111 );
LABEL_196:
          v5 = v326;
          v303 |= v322;
        }
      }
    }
  }
LABEL_48:
  v37 = sub_157EBA0(a1);
  if ( (unsigned int)sub_15F4D60(v37) )
    goto LABEL_49;
  v363.m128i_i64[0] = v5;
  v70 = &v384;
  v379.m128i_i64[0] = 0;
  v353 = v309;
  v71 = &v380;
  v379.m128i_i64[1] = 1;
  do
  {
    v71->m128i_i64[0] = -8;
    v71 = (__m128i *)((char *)v71 + 8);
  }
  while ( v71 != (__m128i *)&v384 );
  v384 = v386;
  v385 = 0x1000000000LL;
  v72 = *(_QWORD *)(a1 + 56);
  v73 = *(_QWORD *)(v72 + 80);
  if ( !v73 )
    BUG();
  v74 = *(_QWORD *)(v73 + 24);
  v75 = v73 + 16;
  if ( v74 != v75 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v74 )
          BUG();
        if ( *(_BYTE *)(v74 - 8) == 53 )
          break;
        i = v363.m128i_i64[0];
        if ( (unsigned __int8)sub_140B1C0(v74 - 24, v363.m128i_i64[0], 0) )
        {
          i = 1;
          if ( !(unsigned __int8)sub_139D0F0(v74 - 24, 1) )
            break;
        }
        v74 = *(_QWORD *)(v74 + 8);
        if ( v75 == v74 )
          goto LABEL_112;
      }
      i = (__int64)&v375;
      v375.m128i_i64[0] = v74 - 24;
      sub_18F6A00((__int64)&v379, &v375, (__int64)v70, v38, v39, v40);
      v74 = *(_QWORD *)(v74 + 8);
    }
    while ( v75 != v74 );
LABEL_112:
    v72 = *(_QWORD *)(a1 + 56);
  }
  if ( (*(_BYTE *)(v72 + 18) & 1) != 0 )
  {
    sub_15E08E0(v72, i);
    v76 = *(_QWORD *)(v72 + 88);
    v77 = v76 + 40LL * *(_QWORD *)(v72 + 96);
    if ( (*(_BYTE *)(v72 + 18) & 1) != 0 )
    {
      sub_15E08E0(v72, i);
      v76 = *(_QWORD *)(v72 + 88);
    }
  }
  else
  {
    v76 = *(_QWORD *)(v72 + 88);
    v77 = v76 + 40LL * *(_QWORD *)(v72 + 96);
  }
  for ( n = v76; v77 != n; n += 40 )
  {
    while ( !(unsigned __int8)sub_15E0300(n) )
    {
      n += 40;
      if ( v77 == n )
        goto LABEL_120;
    }
    v375.m128i_i64[0] = n;
    sub_18F6A00((__int64)&v379, &v375, v79, v80, v81, v82);
  }
LABEL_120:
  v83 = sub_157EB90(a1);
  v313 = sub_1632FA0(v83);
  v366[0].m128i_i64[0] = v306;
  if ( v306 == *(_QWORD *)(a1 + 48) )
  {
LABEL_146:
    v99 = v384;
    goto LABEL_332;
  }
  v310 = 0;
  v84 = (_QWORD *)v306;
  while ( 1 )
  {
    while ( 1 )
    {
      v366[0].m128i_i64[0] = *v84 & 0xFFFFFFFFFFFFFFF8LL;
      v85 = v366[0].m128i_i64[0] - 24;
      if ( !v366[0].m128i_i64[0] )
        v85 = 0;
      v86 = sub_18F47E0(v85, (__int64 *)v363.m128i_i64[0]);
      v87 = v366[0].m128i_i64[0];
      v88 = v86;
      if ( v86 )
      {
        if ( !v366[0].m128i_i64[0] )
          BUG();
        v89 = v366[0].m128i_i64[0] - 24;
        if ( *(_BYTE *)(v366[0].m128i_i64[0] - 8) == 55 )
        {
          v90 = *(_WORD *)(v366[0].m128i_i64[0] - 6);
          if ( ((v90 >> 7) & 6) != 0 || (v90 & 1) != 0 )
          {
LABEL_129:
            v91 = v363.m128i_i64[0];
LABEL_130:
            v87 = v89;
            goto LABEL_131;
          }
        }
        else if ( !(unsigned __int8)sub_18F2D80(v366[0].m128i_i64[0] - 24) )
        {
          goto LABEL_129;
        }
        v100 = v87 - 24;
        v375.m128i_i64[0] = (__int64)&v376;
        v375.m128i_i64[1] = 0x400000000LL;
        if ( *(_BYTE *)(v87 - 8) == 55 )
          sub_141EDF0(&v371, v100);
        else
          sub_18F2E70(&v371, v100);
        sub_14AD470(v371.m128i_i64[0], (__int64)&v375, v313, 0, 6u);
        v101 = v375.m128i_i64[0] + 8LL * v375.m128i_u32[2];
        if ( v375.m128i_i64[0] == v101 )
        {
LABEL_398:
          v264 = v366[0].m128i_i64[0];
          if ( v366[0].m128i_i64[0] )
            v264 = v366[0].m128i_i64[0] - 24;
          sub_18F35E0(v264, v366[0].m128i_i64, a3, v363.m128i_i64[0], (__int64)&v359, (__int64)&v355, (__int64)&v379);
          if ( (__m128i *)v375.m128i_i64[0] != &v376 )
            _libc_free(v375.m128i_u64[0]);
          v310 = v88;
          v84 = (_QWORD *)v366[0].m128i_i64[0];
          goto LABEL_144;
        }
        v102 = 16;
        v103 = (_QWORD *)v375.m128i_i64[0];
        v104 = &v380;
        if ( (v379.m128i_i8[8] & 1) == 0 )
        {
          v104 = (__m128i *)v380.m128i_i64[0];
          v102 = v380.m128i_i32[2];
        }
        v105 = v102 - 1;
        while ( (v379.m128i_i8[8] & 1) != 0 || v380.m128i_i32[2] )
        {
          v106 = 1;
          LODWORD(v107) = v105 & (((unsigned int)*v103 >> 9) ^ ((unsigned int)*v103 >> 4));
          v108 = v104->m128i_i64[(unsigned int)v107];
          if ( *v103 != v108 )
          {
            while ( v108 != -8 )
            {
              v107 = v105 & (unsigned int)(v107 + v106);
              v108 = v104->m128i_i64[v107];
              if ( *v103 == v108 )
                goto LABEL_157;
              ++v106;
            }
            break;
          }
LABEL_157:
          if ( (_QWORD *)v101 == ++v103 )
            goto LABEL_398;
        }
        if ( (__m128i *)v375.m128i_i64[0] != &v376 )
          _libc_free(v375.m128i_u64[0]);
        v87 = v366[0].m128i_i64[0];
      }
      v91 = v363.m128i_i64[0];
      if ( v87 )
      {
        v89 = v87 - 24;
        goto LABEL_130;
      }
LABEL_131:
      v92 = sub_1AE9990(v87, v91);
      if ( !v92 )
        break;
      v109 = v366[0].m128i_i64[0];
      if ( v366[0].m128i_i64[0] )
        v109 = v366[0].m128i_i64[0] - 24;
      sub_18F35E0(v109, v366[0].m128i_i64, a3, v363.m128i_i64[0], (__int64)&v359, (__int64)&v355, (__int64)&v379);
      v310 = v92;
      v84 = (_QWORD *)v366[0].m128i_i64[0];
      if ( *(_QWORD *)(a1 + 48) == v366[0].m128i_i64[0] )
      {
LABEL_145:
        v303 |= v310;
        goto LABEL_146;
      }
    }
    v84 = (_QWORD *)v366[0].m128i_i64[0];
    if ( !v366[0].m128i_i64[0] )
      BUG();
    v93 = *(_BYTE *)(v366[0].m128i_i64[0] - 8);
    v94 = v366[0].m128i_i64[0] - 24;
    if ( v93 == 53 )
    {
      v375.m128i_i64[0] = v366[0].m128i_i64[0] - 24;
      if ( (v379.m128i_i8[8] & 1) != 0 )
      {
        v246 = &v380;
        v247 = 15;
      }
      else
      {
        v246 = (__m128i *)v380.m128i_i64[0];
        if ( !v380.m128i_i32[2] )
          goto LABEL_144;
        v247 = v380.m128i_i32[2] - 1;
      }
      v248 = (__int64 **)v246 + (v247 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4)));
      v249 = v247 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
      v250 = 1;
      v251 = *v248;
      if ( (__int64 *)v94 == *v248 )
      {
LABEL_350:
        *v248 = (__int64 *)-16LL;
        ++v379.m128i_i32[3];
        v379.m128i_i32[2] = (2 * ((unsigned __int32)v379.m128i_i32[2] >> 1) - 2) | v379.m128i_i8[8] & 1;
        v252 = sub_18F2A10(v384, (__int64)&v384[(unsigned int)v385], v375.m128i_i64);
        if ( v252 + 1 != (_QWORD *)v254 )
        {
          memmove(v252, v252 + 1, v254 - (char *)(v252 + 1));
          v253 = v385;
        }
        LODWORD(v385) = v253 - 1;
        v84 = (_QWORD *)v366[0].m128i_i64[0];
      }
      else
      {
        while ( v251 != (__int64 *)-8LL )
        {
          v249 = v247 & (v250 + v249);
          v248 = (__int64 **)v246 + v249;
          v251 = *v248;
          if ( (__int64 *)v94 == *v248 )
            goto LABEL_350;
          ++v250;
        }
      }
      goto LABEL_144;
    }
    if ( v93 > 0x17u )
    {
      if ( v93 == 78 )
      {
        v144 = v94 | 4;
        goto LABEL_238;
      }
      if ( v93 == 29 )
        break;
    }
    if ( v93 != 57 )
      goto LABEL_138;
LABEL_144:
    if ( *(_QWORD **)(a1 + 48) == v84 )
      goto LABEL_145;
  }
  v144 = v94 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_238:
  if ( (v144 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
LABEL_138:
    v375.m128i_i64[0] = 0;
    v375.m128i_i64[1] = -1;
    v376 = 0u;
    v377 = 0;
    v95 = *(_BYTE *)(v366[0].m128i_i64[0] - 8);
    if ( v95 == 54 )
    {
      v96 = *(unsigned __int16 *)(v366[0].m128i_i64[0] - 6);
      if ( ((v96 >> 7) & 6) != 0 || (v96 & 1) != 0 )
      {
LABEL_377:
        v99 = v384;
        goto LABEL_331;
      }
      sub_141EB40(&v341, (__int64 *)v94);
      v97 = _mm_loadu_si128(&v341);
      v98 = _mm_loadu_si128(&v342);
      v377 = v343;
      v375 = v97;
      v376 = v98;
    }
    else
    {
      if ( v95 != 82 )
      {
        if ( (unsigned __int8)sub_15F2ED0(v366[0].m128i_i64[0] - 24) )
          goto LABEL_377;
        goto LABEL_143;
      }
      sub_141F0A0(&v341, v366[0].m128i_i64[0] - 24);
      v134 = _mm_loadu_si128(&v342);
      v375 = _mm_loadu_si128(&v341);
      v377 = v343;
      v376 = v134;
    }
    sub_18F3DB0(v375.m128i_i64, (__int64)&v379, v313, (__int64)v353, v363.m128i_i64[0], *(_QWORD *)(a1 + 56));
    if ( !(_DWORD)v385 )
      goto LABEL_377;
LABEL_143:
    v84 = (_QWORD *)v366[0].m128i_i64[0];
    goto LABEL_144;
  }
  v367.m128i_i64[0] = v144;
  if ( (unsigned __int8)sub_140B1C0(v366[0].m128i_i64[0] - 24, v363.m128i_i64[0], 0) )
  {
    v145 = v366[0].m128i_i64[0];
    if ( v366[0].m128i_i64[0] )
      v145 = v366[0].m128i_i64[0] - 24;
    v375.m128i_i64[0] = v145;
    if ( (v379.m128i_i8[8] & 1) != 0 )
    {
      v146 = &v380;
      v147 = 15;
LABEL_244:
      v148 = v147 & (((unsigned int)v145 >> 9) ^ ((unsigned int)v145 >> 4));
      v149 = &v146->m128i_i64[v148];
      v150 = *v149;
      if ( v145 == *v149 )
      {
LABEL_245:
        *v149 = -16;
        ++v379.m128i_i32[3];
        v379.m128i_i32[2] = (2 * ((unsigned __int32)v379.m128i_i32[2] >> 1) - 2) | v379.m128i_i8[8] & 1;
        v151 = sub_18F2A10(v384, (__int64)&v384[(unsigned int)v385], v375.m128i_i64);
        if ( v151 + 1 != (_QWORD *)v153 )
        {
          memmove(v151, v151 + 1, v153 - (char *)(v151 + 1));
          v152 = v385;
        }
        LODWORD(v385) = v152 - 1;
      }
      else
      {
        v291 = 1;
        while ( v150 != -8 )
        {
          v292 = v291 + 1;
          v148 = v147 & (v291 + v148);
          v149 = &v146->m128i_i64[v148];
          v150 = *v149;
          if ( v145 == *v149 )
            goto LABEL_245;
          v291 = v292;
        }
      }
    }
    else
    {
      v146 = (__m128i *)v380.m128i_i64[0];
      if ( v380.m128i_i32[2] )
      {
        v147 = v380.m128i_i32[2] - 1;
        goto LABEL_244;
      }
    }
  }
  v154 = 0;
  v155 = *(_BYTE *)((v367.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v155 > 0x17u )
  {
    if ( v155 == 78 )
    {
      v154 = v367.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL | 4;
    }
    else
    {
      v154 = 0;
      if ( v155 == 29 )
        v154 = v367.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  if ( (unsigned int)sub_134CC90((__int64)v353, v154) == 4 )
    goto LABEL_143;
  v156 = v384;
  v375.m128i_i64[1] = (__int64)&v367;
  v157 = &v363;
  v158 = 8LL * (unsigned int)v385;
  v378 = &v379;
  v376.m128i_i64[0] = v313;
  v317 = &v384[(unsigned __int64)v158 / 8];
  v159 = v158 >> 3;
  v160 = (__int64 *)&v353;
  v161 = v158 >> 5;
  v375.m128i_i64[0] = (__int64)&v353;
  v376.m128i_i64[1] = (__int64)&v363;
  v377 = a1;
  if ( !v161 )
    goto LABEL_462;
  v162 = &v384[4 * v161];
  v163 = v313;
  v311 = v162;
  v164 = a1;
  while ( 2 )
  {
    v193 = v157->m128i_i64[0];
    v194 = (_QWORD *)*v156;
    v330 = v163;
    v195 = *(_QWORD *)(v164 + 56);
    v196 = (_QWORD *)*v160;
    LOWORD(v352[0]) = 0;
    v339 = v193;
    BYTE2(v352[0]) = sub_15E4690(v195, 0);
    v197 = sub_140E950(v194, &v371, v330, v339, v352[0]);
    v198 = -1;
    if ( v197 )
      v198 = v371.m128i_i64[0];
    v165 = 0;
    v199 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
    v200 = *(_BYTE *)(v199 + 16);
    if ( v200 > 0x17u )
    {
      if ( v200 == 78 )
      {
        v165 = v199 | 4;
      }
      else
      {
        v165 = 0;
        if ( v200 == 29 )
          v165 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      }
    }
    v371.m128i_i64[1] = v198;
    v371.m128i_i64[0] = (__int64)v194;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    if ( (sub_134F0E0(v196, v165, (__int64)&v371) & 1) != 0 )
    {
      v201 = v378;
      if ( (v378->m128i_i8[8] & 1) != 0 )
      {
        v202 = v378 + 1;
        v203 = 15;
      }
      else
      {
        v242 = v378[1].m128i_i32[2];
        v202 = (__m128i *)v378[1].m128i_i64[0];
        if ( !v242 )
          goto LABEL_293;
        v203 = v242 - 1;
      }
      v204 = v203 & (((unsigned int)*v156 >> 9) ^ ((unsigned int)*v156 >> 4));
      v205 = &v202->m128i_i64[v204];
      v206 = *v205;
      if ( *v156 == *v205 )
      {
LABEL_292:
        *v205 = -16;
        v207 = v201->m128i_u32[2];
        ++v201->m128i_i32[3];
        v201->m128i_i32[2] = (2 * (v207 >> 1) - 2) | v207 & 1;
      }
      else
      {
        v289 = 1;
        while ( v206 != -8 )
        {
          v290 = v289 + 1;
          v204 = v203 & (v289 + v204);
          v205 = &v202->m128i_i64[v204];
          v206 = *v205;
          if ( *v156 == *v205 )
            goto LABEL_292;
          v289 = v290;
        }
      }
LABEL_293:
      if ( v317 == v156 )
        goto LABEL_328;
      v208 = (_QWORD **)(v156 + 1);
      if ( v317 == v156 + 1 )
        goto LABEL_328;
      while ( 1 )
      {
        while ( 1 )
        {
          v210 = *v208;
          v371.m128i_i16[0] = 0;
          v331 = v353;
          v340 = v363.m128i_i64[0];
          v371.m128i_i8[2] = sub_15E4690(*(_QWORD *)(a1 + 56), 0);
          v211 = sub_140E950(v210, &v375, v313, v340, v371.m128i_i32[0]);
          v212 = -1;
          if ( v211 )
            v212 = v375.m128i_i64[0];
          v209 = 0;
          v213 = *(_BYTE *)((v367.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 16);
          if ( v213 > 0x17u )
          {
            if ( v213 == 78 )
            {
              v209 = v367.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL | 4;
            }
            else
            {
              v209 = 0;
              if ( v213 == 29 )
                v209 = v367.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
            }
          }
          v375.m128i_i64[1] = v212;
          v375.m128i_i64[0] = (__int64)v210;
          v376 = 0u;
          v377 = 0;
          if ( (sub_134F0E0(v331, v209, (__int64)&v375) & 1) != 0 )
            break;
          *v156++ = *v208;
LABEL_300:
          if ( v317 == ++v208 )
            goto LABEL_328;
        }
        if ( (v379.m128i_i8[8] & 1) != 0 )
        {
          v236 = &v380;
          v237 = 15;
        }
        else
        {
          v236 = (__m128i *)v380.m128i_i64[0];
          if ( !v380.m128i_i32[2] )
            goto LABEL_300;
          v237 = v380.m128i_i32[2] - 1;
        }
        v238 = v237 & (((unsigned int)*v208 >> 9) ^ ((unsigned int)*v208 >> 4));
        v239 = &v236->m128i_i64[v238];
        v240 = (_QWORD *)*v239;
        if ( (_QWORD *)*v239 != *v208 )
        {
          v285 = 1;
          while ( v240 != (_QWORD *)-8LL )
          {
            v286 = v285 + 1;
            v238 = v237 & (v285 + v238);
            v239 = &v236->m128i_i64[v238];
            v240 = (_QWORD *)*v239;
            if ( *v208 == (_QWORD *)*v239 )
              goto LABEL_327;
            v285 = v286;
          }
          goto LABEL_300;
        }
LABEL_327:
        *v239 = -16;
        ++v208;
        ++v379.m128i_i32[3];
        v379.m128i_i32[2] = (2 * ((unsigned __int32)v379.m128i_i32[2] >> 1) - 2) | v379.m128i_i8[8] & 1;
        if ( v317 == v208 )
          goto LABEL_328;
      }
    }
    v323 = v156 + 1;
    v166 = (_QWORD *)v156[1];
    v336 = v376.m128i_i64[0];
    v167 = *(_QWORD **)v375.m128i_i64[0];
    v168 = *(_QWORD *)(v377 + 56);
    v169 = *(_QWORD *)v376.m128i_i64[1];
    LOWORD(v352[0]) = 0;
    v327 = v169;
    BYTE2(v352[0]) = sub_15E4690(v168, 0);
    v170 = sub_140E950(v166, &v371, v336, v327, v352[0]);
    v171 = -1;
    if ( v170 )
      v171 = v371.m128i_i64[0];
    v172 = 0;
    v173 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
    v174 = *(_BYTE *)(v173 + 16);
    if ( v174 > 0x17u )
    {
      if ( v174 == 78 )
      {
        v172 = v173 | 4;
      }
      else
      {
        v172 = 0;
        if ( v174 == 29 )
          v172 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      }
    }
    v371.m128i_i64[1] = v171;
    v371.m128i_i64[0] = (__int64)v166;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    if ( (sub_134F0E0(v167, v172, (__int64)&v371) & 1) != 0 )
    {
      v214 = v378;
      if ( (v378->m128i_i8[8] & 1) != 0 )
      {
        v215 = v378 + 1;
        v216 = 15;
        goto LABEL_308;
      }
      v243 = v378[1].m128i_i32[2];
      v215 = (__m128i *)v378[1].m128i_i64[0];
      if ( v243 )
      {
        v216 = v243 - 1;
LABEL_308:
        v217 = v156[1];
        v218 = v216 & (((unsigned int)v217 >> 9) ^ ((unsigned int)v217 >> 4));
        v219 = &v215->m128i_i64[v218];
        v220 = *v219;
        if ( v217 == *v219 )
        {
LABEL_309:
          *v219 = -16;
          v221 = v214->m128i_u32[2];
          ++v214->m128i_i32[3];
          v214->m128i_i32[2] = (2 * (v221 >> 1) - 2) | v221 & 1;
          v156 = v323;
          goto LABEL_293;
        }
        v287 = 1;
        while ( v220 != -8 )
        {
          v288 = v287 + 1;
          v218 = v216 & (v287 + v218);
          v219 = &v215->m128i_i64[v218];
          v220 = *v219;
          if ( v217 == *v219 )
            goto LABEL_309;
          v287 = v288;
        }
      }
      goto LABEL_340;
    }
    v323 = v156 + 2;
    v175 = (_QWORD *)v156[2];
    v337 = v376.m128i_i64[0];
    v176 = *(_QWORD **)v375.m128i_i64[0];
    v177 = *(_QWORD *)(v377 + 56);
    v178 = *(_QWORD *)v376.m128i_i64[1];
    LOWORD(v352[0]) = 0;
    v328 = v178;
    BYTE2(v352[0]) = sub_15E4690(v177, 0);
    v179 = sub_140E950(v175, &v371, v337, v328, v352[0]);
    v180 = -1;
    if ( v179 )
      v180 = v371.m128i_i64[0];
    v181 = 0;
    v182 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
    v183 = *(_BYTE *)(v182 + 16);
    if ( v183 > 0x17u )
    {
      if ( v183 == 78 )
      {
        v181 = v182 | 4;
      }
      else
      {
        v181 = 0;
        if ( v183 == 29 )
          v181 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      }
    }
    v371.m128i_i64[1] = v180;
    v371.m128i_i64[0] = (__int64)v175;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    if ( (sub_134F0E0(v176, v181, (__int64)&v371) & 1) != 0 )
    {
      v214 = v378;
      if ( (v378->m128i_i8[8] & 1) != 0 )
      {
        v222 = v378 + 1;
        v223 = 15;
      }
      else
      {
        v245 = v378[1].m128i_i32[2];
        v222 = (__m128i *)v378[1].m128i_i64[0];
        if ( !v245 )
          goto LABEL_340;
        v223 = v245 - 1;
      }
      v224 = v156[2];
      v225 = v223 & (((unsigned int)v224 >> 9) ^ ((unsigned int)v224 >> 4));
      v219 = &v222->m128i_i64[v225];
      v226 = *v219;
      if ( *v219 == v224 )
        goto LABEL_309;
      v227 = 1;
      while ( v226 != -8 )
      {
        v228 = v227 + 1;
        v225 = v223 & (v227 + v225);
        v219 = &v222->m128i_i64[v225];
        v226 = *v219;
        if ( v224 == *v219 )
          goto LABEL_309;
        v227 = v228;
      }
      goto LABEL_340;
    }
    v323 = v156 + 3;
    v184 = (_QWORD *)v156[3];
    v338 = v376.m128i_i64[0];
    v185 = *(_QWORD **)v375.m128i_i64[0];
    v186 = *(_QWORD *)(v377 + 56);
    v187 = *(_QWORD *)v376.m128i_i64[1];
    LOWORD(v352[0]) = 0;
    v329 = v187;
    BYTE2(v352[0]) = sub_15E4690(v186, 0);
    v188 = sub_140E950(v184, &v371, v338, v329, v352[0]);
    v189 = -1;
    if ( v188 )
      v189 = v371.m128i_i64[0];
    v190 = 0;
    v191 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
    v192 = *(_BYTE *)(v191 + 16);
    if ( v192 > 0x17u )
    {
      if ( v192 == 78 )
      {
        v190 = v191 | 4;
      }
      else
      {
        v190 = 0;
        if ( v192 == 29 )
          v190 = *(_QWORD *)v375.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      }
    }
    v371.m128i_i64[1] = v189;
    v371.m128i_i64[0] = (__int64)v184;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    if ( (sub_134F0E0(v185, v190, (__int64)&v371) & 1) != 0 )
    {
      v214 = v378;
      if ( (v378->m128i_i8[8] & 1) != 0 )
      {
        v229 = v378 + 1;
        v230 = 15;
LABEL_319:
        v231 = v156[3];
        v232 = v230 & (((unsigned int)v231 >> 9) ^ ((unsigned int)v231 >> 4));
        v219 = &v229->m128i_i64[v232];
        v233 = *v219;
        if ( *v219 == v231 )
          goto LABEL_309;
        v234 = 1;
        while ( v233 != -8 )
        {
          v235 = v234 + 1;
          v232 = v230 & (v234 + v232);
          v219 = &v229->m128i_i64[v232];
          v233 = *v219;
          if ( v231 == *v219 )
            goto LABEL_309;
          v234 = v235;
        }
        goto LABEL_340;
      }
      v244 = v378[1].m128i_i32[2];
      v229 = (__m128i *)v378[1].m128i_i64[0];
      if ( v244 )
      {
        v230 = v244 - 1;
        goto LABEL_319;
      }
LABEL_340:
      v156 = v323;
      goto LABEL_293;
    }
    v156 += 4;
    if ( v311 != v156 )
    {
      v160 = (__int64 *)v375.m128i_i64[0];
      v164 = v377;
      v157 = (__m128i *)v376.m128i_i64[1];
      v163 = v376.m128i_i64[0];
      continue;
    }
    break;
  }
  v159 = v317 - v156;
LABEL_462:
  if ( v159 != 2 )
  {
    if ( v159 != 3 )
    {
      if ( v159 != 1 )
        goto LABEL_465;
      goto LABEL_470;
    }
    if ( (unsigned __int8)sub_18F3C00((__int64)&v375, (_QWORD **)v156) )
      goto LABEL_293;
    ++v156;
  }
  if ( (unsigned __int8)sub_18F3C00((__int64)&v375, (_QWORD **)v156) )
    goto LABEL_293;
  ++v156;
LABEL_470:
  if ( (unsigned __int8)sub_18F3C00((__int64)&v375, (_QWORD **)v156) )
    goto LABEL_293;
LABEL_465:
  v156 = v317;
LABEL_328:
  v99 = v384;
  v241 = v385;
  if ( v156 != &v384[(unsigned int)v385] )
  {
    LODWORD(v385) = v156 - v384;
    v241 = v385;
  }
  if ( v241 )
    goto LABEL_143;
LABEL_331:
  v303 |= v310;
LABEL_332:
  if ( v99 != v386 )
    _libc_free((unsigned __int64)v99);
  if ( (v379.m128i_i8[8] & 1) == 0 )
    j___libc_free_0(v380.m128i_i64[0]);
LABEL_49:
  if ( v362 )
  {
    v41 = v360;
    v42 = &v360[7 * v362];
    do
    {
      if ( *v41 != -16 && *v41 != -8 )
        sub_18F3410(v41[3]);
      v41 += 7;
    }
    while ( v42 != v41 );
  }
  j___libc_free_0(v360);
  j___libc_free_0(v356);
  return v303;
}
