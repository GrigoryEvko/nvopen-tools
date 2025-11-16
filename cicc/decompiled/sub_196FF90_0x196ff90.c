// Function: sub_196FF90
// Address: 0x196ff90
//
__int64 __fastcall sub_196FF90(
        __int64 a1,
        __m128i a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned int v12; // r13d
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rbx
  int v19; // edx
  __int64 v20; // rax
  int v21; // edi
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // r12
  __int64 v27; // rbx
  int v28; // eax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  _QWORD *i; // rdx
  __int64 v32; // rbx
  __int64 v33; // r14
  __int64 v34; // r12
  unsigned __int64 v35; // rdi
  double v36; // xmm4_8
  double v37; // xmm5_8
  __int64 v38; // rbx
  __int64 v39; // r14
  __int64 v40; // r12
  unsigned __int64 v41; // rdi
  __int64 v42; // rbx
  __int64 v43; // r13
  unsigned int v44; // r14d
  __int64 *v45; // r10
  __int64 v46; // rdi
  __int64 v47; // r11
  char v48; // si
  _DWORD *v49; // rax
  __int64 v50; // rcx
  int v51; // edx
  _DWORD *v52; // r15
  __int64 v53; // r12
  __int64 v54; // rcx
  _DWORD *v55; // rcx
  __int64 v56; // r12
  __int64 v57; // rbx
  char v58; // r14
  char v59; // al
  __int64 v60; // r12
  __int64 v61; // rbx
  char v62; // r14
  char v63; // al
  __int64 *v64; // rax
  __int64 v65; // r15
  __int64 v66; // rbx
  __int64 v67; // rsi
  __int64 v68; // r13
  __int64 v69; // r14
  __int64 v70; // rsi
  __int64 v71; // rax
  unsigned __int64 v72; // r14
  __int64 v73; // rax
  __int64 v74; // rax
  unsigned int v75; // ecx
  unsigned __int64 v76; // rax
  unsigned int v77; // ecx
  __int64 v78; // r8
  __int64 v79; // r9
  char v80; // r12
  unsigned __int64 v81; // rax
  __int64 v82; // r15
  __int64 v83; // rbx
  __int64 v84; // rdx
  __int64 v85; // r9
  __int64 v86; // r12
  __int64 v87; // rdi
  unsigned int v88; // r14d
  int v89; // eax
  bool v90; // al
  __int64 *v91; // r12
  __int64 v92; // rax
  bool v93; // r10
  __int64 v94; // rax
  char v95; // cl
  unsigned __int64 v96; // rdx
  unsigned int v97; // r12d
  _QWORD *v98; // r13
  __int64 v99; // r12
  unsigned __int64 v100; // r13
  _QWORD *v101; // rax
  unsigned __int8 *v102; // rsi
  __int64 v103; // rax
  const char *v104; // r13
  __int64 v105; // rax
  __int64 v106; // rbx
  __int64 v107; // rax
  unsigned int v108; // r13d
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  _QWORD *v112; // r9
  __int64 v113; // rdx
  __int64 v114; // r9
  __int64 v115; // rax
  int v116; // r13d
  __int64 v117; // rax
  _QWORD *v118; // r13
  _QWORD *v119; // rcx
  __int64 v120; // rbx
  unsigned __int64 v121; // rax
  __int64 *v122; // rbx
  unsigned int v123; // eax
  _QWORD *v124; // rbx
  const void **v125; // r12
  unsigned __int8 *v126; // rsi
  __int64 v127; // rax
  double v128; // xmm4_8
  double v129; // xmm5_8
  _QWORD *v130; // rbx
  _QWORD *v131; // r12
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // r12
  unsigned int v135; // eax
  __int64 v136; // rsi
  __int64 v137; // r8
  unsigned __int64 v138; // rcx
  _QWORD *v139; // rbx
  _QWORD *v140; // r12
  __int64 v141; // rax
  unsigned __int64 *v142; // rax
  unsigned __int64 *v143; // r13
  __int64 v145; // r12
  __int64 v146; // r15
  __int64 v147; // r14
  __int64 v148; // rsi
  unsigned __int8 *v149; // rsi
  unsigned int v150; // ecx
  _QWORD *v151; // rdi
  unsigned int v152; // eax
  int v153; // eax
  unsigned __int64 v154; // rax
  unsigned __int64 v155; // rax
  int v156; // ebx
  __int64 v157; // r12
  _QWORD *v158; // rax
  __int64 v159; // rdx
  _QWORD *j; // rdx
  int v161; // eax
  __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // r10
  __int64 v166; // r14
  unsigned __int16 v167; // ax
  __int64 v168; // rax
  __int64 v169; // rax
  unsigned __int64 v170; // r15
  __int64 v171; // r15
  unsigned int v172; // eax
  __int64 v173; // rax
  __int64 v174; // rax
  int v175; // eax
  __int64 v176; // rax
  _QWORD *v177; // rax
  __int64 v178; // rax
  int v179; // eax
  __int64 v180; // rax
  _QWORD *v181; // rax
  unsigned int v182; // esi
  int v183; // eax
  __int64 v184; // rax
  int v185; // eax
  __int64 v186; // rax
  __int64 v187; // r9
  __int64 v188; // r8
  char v189; // al
  unsigned __int64 v190; // rsi
  unsigned int v191; // ecx
  __int64 v192; // r8
  __int64 v193; // r9
  int v194; // eax
  unsigned int v195; // ecx
  unsigned int v196; // eax
  double v197; // xmm4_8
  double v198; // xmm5_8
  int v199; // r9d
  int v200; // eax
  bool v201; // r10
  _QWORD *v202; // rax
  char v203; // al
  bool v204; // r15
  __int64 v205; // rdx
  __int64 v206; // rcx
  __int64 v207; // r8
  int v208; // r9d
  __int64 v209; // rdi
  int v210; // r8d
  int v211; // r9d
  __int64 v212; // r14
  __int64 v213; // rax
  bool v214; // al
  int v215; // r8d
  int v216; // r9d
  __int64 v217; // rax
  __int64 *v218; // [rsp+10h] [rbp-380h]
  __int64 *v219; // [rsp+18h] [rbp-378h]
  __int64 v220; // [rsp+20h] [rbp-370h]
  _QWORD *v221; // [rsp+28h] [rbp-368h]
  unsigned __int8 v222; // [rsp+36h] [rbp-35Ah]
  char v223; // [rsp+37h] [rbp-359h]
  __int64 v224; // [rsp+38h] [rbp-358h]
  unsigned __int64 v225; // [rsp+40h] [rbp-350h]
  __int64 v226; // [rsp+40h] [rbp-350h]
  bool v227; // [rsp+50h] [rbp-340h]
  unsigned __int64 v228; // [rsp+50h] [rbp-340h]
  __int64 *v229; // [rsp+50h] [rbp-340h]
  unsigned __int64 v230; // [rsp+50h] [rbp-340h]
  __int64 v231; // [rsp+58h] [rbp-338h]
  __int64 v232; // [rsp+58h] [rbp-338h]
  __int64 v233; // [rsp+58h] [rbp-338h]
  bool v234; // [rsp+60h] [rbp-330h]
  __int64 v235; // [rsp+60h] [rbp-330h]
  unsigned __int8 *v236; // [rsp+60h] [rbp-330h]
  __int64 v237; // [rsp+60h] [rbp-330h]
  __int64 v238; // [rsp+60h] [rbp-330h]
  __int64 *v239; // [rsp+68h] [rbp-328h]
  __int64 v240; // [rsp+70h] [rbp-320h]
  __int64 *v241; // [rsp+78h] [rbp-318h]
  unsigned int v242; // [rsp+78h] [rbp-318h]
  __int64 v243; // [rsp+78h] [rbp-318h]
  unsigned __int64 v244; // [rsp+78h] [rbp-318h]
  unsigned int v245; // [rsp+78h] [rbp-318h]
  __int64 v246; // [rsp+78h] [rbp-318h]
  _QWORD *v247; // [rsp+78h] [rbp-318h]
  __int64 v248; // [rsp+78h] [rbp-318h]
  __int64 v249; // [rsp+80h] [rbp-310h]
  unsigned __int8 *v250; // [rsp+88h] [rbp-308h]
  char v251; // [rsp+88h] [rbp-308h]
  __int64 *v252; // [rsp+88h] [rbp-308h]
  __int64 v253; // [rsp+88h] [rbp-308h]
  __int64 *v254; // [rsp+88h] [rbp-308h]
  __int64 *v255; // [rsp+88h] [rbp-308h]
  unsigned __int64 v256; // [rsp+88h] [rbp-308h]
  unsigned __int64 v257; // [rsp+88h] [rbp-308h]
  unsigned __int64 v258; // [rsp+88h] [rbp-308h]
  __int64 v259; // [rsp+88h] [rbp-308h]
  __int64 v260; // [rsp+88h] [rbp-308h]
  __int64 v261; // [rsp+88h] [rbp-308h]
  __int64 v262; // [rsp+88h] [rbp-308h]
  __int64 v263; // [rsp+88h] [rbp-308h]
  __int64 v264; // [rsp+88h] [rbp-308h]
  __int64 v265; // [rsp+90h] [rbp-300h]
  __int64 v266; // [rsp+90h] [rbp-300h]
  char v267; // [rsp+90h] [rbp-300h]
  unsigned __int64 v268; // [rsp+90h] [rbp-300h]
  __int64 v269; // [rsp+90h] [rbp-300h]
  __int64 v270; // [rsp+90h] [rbp-300h]
  __int64 v271; // [rsp+90h] [rbp-300h]
  __int64 v272; // [rsp+90h] [rbp-300h]
  unsigned __int8 *v273; // [rsp+90h] [rbp-300h]
  _QWORD *v274; // [rsp+90h] [rbp-300h]
  __int64 v275; // [rsp+90h] [rbp-300h]
  __int64 v276; // [rsp+98h] [rbp-2F8h]
  __int64 v277; // [rsp+98h] [rbp-2F8h]
  __int64 v278; // [rsp+98h] [rbp-2F8h]
  __int64 v279; // [rsp+98h] [rbp-2F8h]
  __int64 v280; // [rsp+98h] [rbp-2F8h]
  unsigned int v281; // [rsp+98h] [rbp-2F8h]
  char v282; // [rsp+98h] [rbp-2F8h]
  bool v283; // [rsp+98h] [rbp-2F8h]
  unsigned __int8 *v284; // [rsp+A0h] [rbp-2F0h] BYREF
  unsigned int v285; // [rsp+A8h] [rbp-2E8h]
  _QWORD *v286; // [rsp+B0h] [rbp-2E0h] BYREF
  unsigned int v287; // [rsp+B8h] [rbp-2D8h]
  __int16 v288; // [rsp+C0h] [rbp-2D0h] BYREF
  __int64 v289; // [rsp+C8h] [rbp-2C8h]
  _QWORD *v290; // [rsp+D0h] [rbp-2C0h]
  __int64 v291; // [rsp+D8h] [rbp-2B8h]
  unsigned int v292; // [rsp+E0h] [rbp-2B0h]
  unsigned __int64 v293; // [rsp+F0h] [rbp-2A0h] BYREF
  __int64 *v294; // [rsp+F8h] [rbp-298h]
  __int64 *v295; // [rsp+100h] [rbp-290h]
  __int64 v296; // [rsp+108h] [rbp-288h]
  int v297; // [rsp+110h] [rbp-280h]
  __int64 v298; // [rsp+118h] [rbp-278h] BYREF
  unsigned __int64 v299; // [rsp+120h] [rbp-270h] BYREF
  __int64 v300; // [rsp+128h] [rbp-268h]
  unsigned __int64 v301; // [rsp+130h] [rbp-260h]
  _QWORD *v302; // [rsp+138h] [rbp-258h]
  __int64 v303; // [rsp+140h] [rbp-250h]
  int v304; // [rsp+148h] [rbp-248h]
  __int64 v305; // [rsp+150h] [rbp-240h]
  __int64 v306; // [rsp+158h] [rbp-238h]
  _BYTE *v307; // [rsp+170h] [rbp-220h] BYREF
  __int64 v308; // [rsp+178h] [rbp-218h]
  _BYTE v309[64]; // [rsp+180h] [rbp-210h] BYREF
  __int64 v310; // [rsp+1C0h] [rbp-1D0h] BYREF
  const char *v311; // [rsp+1C8h] [rbp-1C8h]
  const char *v312; // [rsp+1D0h] [rbp-1C0h]
  __int64 v313; // [rsp+1D8h] [rbp-1B8h]
  _QWORD *v314; // [rsp+1E0h] [rbp-1B0h]
  __int64 v315; // [rsp+1E8h] [rbp-1A8h] BYREF
  unsigned int v316; // [rsp+1F0h] [rbp-1A0h]
  __int64 v317; // [rsp+1F8h] [rbp-198h]
  __int64 v318; // [rsp+200h] [rbp-190h]
  __int64 v319; // [rsp+208h] [rbp-188h]
  __int64 v320; // [rsp+210h] [rbp-180h]
  __int64 v321; // [rsp+218h] [rbp-178h]
  __int64 v322; // [rsp+220h] [rbp-170h]
  __int64 v323; // [rsp+228h] [rbp-168h]
  __int64 v324; // [rsp+230h] [rbp-160h]
  __int64 v325; // [rsp+238h] [rbp-158h]
  __int64 v326; // [rsp+240h] [rbp-150h]
  __int64 v327; // [rsp+248h] [rbp-148h]
  int v328; // [rsp+250h] [rbp-140h]
  __int64 v329; // [rsp+258h] [rbp-138h]
  _BYTE *v330; // [rsp+260h] [rbp-130h]
  _BYTE *v331; // [rsp+268h] [rbp-128h]
  __int64 v332; // [rsp+270h] [rbp-120h]
  int v333; // [rsp+278h] [rbp-118h]
  _BYTE v334[16]; // [rsp+280h] [rbp-110h] BYREF
  __int64 v335; // [rsp+290h] [rbp-100h]
  __int64 v336; // [rsp+298h] [rbp-F8h]
  __int64 v337; // [rsp+2A0h] [rbp-F0h]
  __int64 v338; // [rsp+2A8h] [rbp-E8h]
  __int64 v339; // [rsp+2B0h] [rbp-E0h]
  __int64 v340; // [rsp+2B8h] [rbp-D8h]
  __int16 v341; // [rsp+2C0h] [rbp-D0h]
  __int64 v342[5]; // [rsp+2C8h] [rbp-C8h] BYREF
  int v343; // [rsp+2F0h] [rbp-A0h]
  __int64 v344; // [rsp+2F8h] [rbp-98h]
  __int64 v345; // [rsp+300h] [rbp-90h]
  const char *v346; // [rsp+308h] [rbp-88h]
  _BYTE *v347; // [rsp+310h] [rbp-80h]
  __int64 v348; // [rsp+318h] [rbp-78h]
  _BYTE v349[112]; // [rsp+320h] [rbp-70h] BYREF

  v10 = sub_1481F60(*(_QWORD **)(a1 + 32), *(_QWORD *)a1, a2, a3);
  v240 = v10;
  if ( *(_WORD *)(v10 + 24) )
    goto LABEL_4;
  v11 = *(_QWORD *)(v10 + 32);
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 > 0x40 )
  {
    if ( v12 - (unsigned int)sub_16A57B0(v11 + 24) > 0x40 || **(_QWORD **)(v11 + 24) )
      goto LABEL_4;
    return 0;
  }
  if ( !*(_QWORD *)(v11 + 24) )
    return 0;
LABEL_4:
  v13 = *(_QWORD *)a1;
  v307 = v309;
  v308 = 0x800000000LL;
  sub_13FA0E0(v13, (__int64)&v307);
  v14 = *(_QWORD *)a1;
  v288 = 0;
  v289 = 0;
  v290 = 0;
  v291 = 0;
  v292 = 0;
  sub_1436EA0((__int64)&v288, v14);
  if ( !(_BYTE)v288 )
  {
    v15 = *(_QWORD *)a1;
    v218 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
    if ( *(__int64 **)(*(_QWORD *)a1 + 32LL) != v218 )
    {
      v239 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
      v16 = a1;
      v222 = 0;
      while ( 1 )
      {
        v17 = *(_QWORD *)(v16 + 24);
        v18 = *v239;
        v19 = *(_DWORD *)(v17 + 24);
        v20 = 0;
        v220 = *v239;
        if ( v19 )
        {
          v21 = v19 - 1;
          v22 = *(_QWORD *)(v17 + 8);
          v23 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v24 = (__int64 *)(v22 + 16LL * v23);
          v25 = *v24;
          if ( v18 == *v24 )
          {
LABEL_11:
            v20 = v24[1];
          }
          else
          {
            v161 = 1;
            while ( v25 != -8 )
            {
              v199 = v161 + 1;
              v23 = v21 & (v161 + v23);
              v24 = (__int64 *)(v22 + 16LL * v23);
              v25 = *v24;
              if ( v220 == *v24 )
                goto LABEL_11;
              v161 = v199;
            }
            v20 = 0;
          }
        }
        if ( v20 == v15 )
          break;
LABEL_7:
        if ( v218 == ++v239 )
          goto LABEL_180;
        v15 = *(_QWORD *)v16;
      }
      v26 = 0;
      v27 = 8LL * (unsigned int)v308;
      if ( (_DWORD)v308 )
      {
        while ( sub_15CC8F0(*(_QWORD *)(v16 + 16), v220, *(_QWORD *)&v307[v26]) )
        {
          v26 += 8;
          if ( v27 == v26 )
            goto LABEL_16;
        }
        goto LABEL_7;
      }
LABEL_16:
      ++*(_QWORD *)(v16 + 72);
      v276 = v16 + 72;
      v28 = *(_DWORD *)(v16 + 88);
      if ( v28 )
      {
        v150 = 4 * v28;
        v29 = *(unsigned int *)(v16 + 96);
        if ( (unsigned int)(4 * v28) < 0x40 )
          v150 = 64;
        if ( v150 >= (unsigned int)v29 )
        {
LABEL_19:
          v30 = *(_QWORD **)(v16 + 80);
          for ( i = &v30[2 * v29]; i != v30; v30 += 2 )
            *v30 = -8;
          *(_QWORD *)(v16 + 88) = 0;
          goto LABEL_22;
        }
        v151 = *(_QWORD **)(v16 + 80);
        v152 = v28 - 1;
        if ( v152 )
        {
          _BitScanReverse(&v152, v152);
          v153 = 1 << (33 - (v152 ^ 0x1F));
          if ( v153 < 64 )
            v153 = 64;
          if ( (_DWORD)v29 == v153 )
          {
            *(_QWORD *)(v16 + 88) = 0;
            v202 = &v151[2 * (unsigned int)v29];
            do
            {
              if ( v151 )
                *v151 = -8;
              v151 += 2;
            }
            while ( v202 != v151 );
            goto LABEL_22;
          }
          v154 = (4 * v153 / 3u + 1) | ((unsigned __int64)(4 * v153 / 3u + 1) >> 1);
          v155 = ((v154 | (v154 >> 2)) >> 4)
               | v154
               | (v154 >> 2)
               | ((((v154 | (v154 >> 2)) >> 4) | v154 | (v154 >> 2)) >> 8);
          v156 = (v155 | (v155 >> 16)) + 1;
          v157 = 16 * ((v155 | (v155 >> 16)) + 1);
        }
        else
        {
          v157 = 2048;
          v156 = 128;
        }
        j___libc_free_0(v151);
        *(_DWORD *)(v16 + 96) = v156;
        v158 = (_QWORD *)sub_22077B0(v157);
        v159 = *(unsigned int *)(v16 + 96);
        *(_QWORD *)(v16 + 88) = 0;
        *(_QWORD *)(v16 + 80) = v158;
        for ( j = &v158[2 * v159]; j != v158; v158 += 2 )
        {
          if ( v158 )
            *v158 = -8;
        }
        goto LABEL_22;
      }
      if ( *(_DWORD *)(v16 + 92) )
      {
        v29 = *(unsigned int *)(v16 + 96);
        if ( (unsigned int)v29 <= 0x40 )
          goto LABEL_19;
        j___libc_free_0(*(_QWORD *)(v16 + 80));
        *(_QWORD *)(v16 + 80) = 0;
        *(_QWORD *)(v16 + 88) = 0;
        *(_DWORD *)(v16 + 96) = 0;
      }
LABEL_22:
      v32 = *(_QWORD *)(v16 + 104);
      v33 = *(_QWORD *)(v16 + 112);
      if ( v32 != v33 )
      {
        v34 = *(_QWORD *)(v16 + 104);
        do
        {
          v35 = *(_QWORD *)(v34 + 8);
          if ( v35 != v34 + 24 )
            _libc_free(v35);
          v34 += 88;
        }
        while ( v33 != v34 );
        *(_QWORD *)(v16 + 112) = v32;
      }
      v265 = v16 + 128;
      sub_196A810(v16 + 128);
      v38 = *(_QWORD *)(v16 + 160);
      v39 = *(_QWORD *)(v16 + 168);
      if ( v38 != v39 )
      {
        v40 = *(_QWORD *)(v16 + 160);
        do
        {
          v41 = *(_QWORD *)(v40 + 8);
          if ( v41 != v40 + 24 )
            _libc_free(v41);
          v40 += 88;
        }
        while ( v39 != v40 );
        *(_QWORD *)(v16 + 168) = v38;
      }
      *(_DWORD *)(v16 + 192) = 0;
      v249 = v220 + 40;
      if ( *(_QWORD *)(v220 + 48) != v220 + 40 )
      {
        v42 = v16;
        v43 = *(_QWORD *)(v220 + 48);
        while ( 1 )
        {
          if ( !v43 )
            BUG();
          if ( *(_BYTE *)(v43 - 8) != 55 )
            goto LABEL_49;
          v44 = *(unsigned __int16 *)(v43 - 6);
          if ( (v44 & 1) != 0 || ((v44 >> 7) & 6) != 0 )
            goto LABEL_49;
          v45 = *(__int64 **)(v43 - 72);
          v46 = *(_QWORD *)(v42 + 56);
          v47 = *v45;
          v48 = *(_BYTE *)(*v45 + 8);
          if ( v48 != 15 )
            goto LABEL_211;
          v49 = *(_DWORD **)(v46 + 408);
          v50 = 4LL * *(unsigned int *)(v46 + 416);
          v51 = *(_DWORD *)(v47 + 8) >> 8;
          v52 = &v49[(unsigned __int64)v50 / 4];
          v53 = v50 >> 2;
          v54 = v50 >> 4;
          if ( v54 )
          {
            v55 = &v49[4 * v54];
            while ( v51 != *v49 )
            {
              if ( v51 == v49[1] )
              {
                ++v49;
                goto LABEL_48;
              }
              if ( v51 == v49[2] )
              {
                v49 += 2;
                goto LABEL_48;
              }
              if ( v51 == v49[3] )
              {
                v49 += 3;
                goto LABEL_48;
              }
              v49 += 4;
              if ( v55 == v49 )
              {
                v53 = v52 - v49;
                goto LABEL_207;
              }
            }
            goto LABEL_48;
          }
LABEL_207:
          if ( v53 == 2 )
            goto LABEL_301;
          if ( v53 != 3 )
          {
            if ( v53 != 1 )
              goto LABEL_211;
            goto LABEL_210;
          }
          if ( v51 != *v49 )
            break;
LABEL_48:
          if ( v52 != v49 )
            goto LABEL_49;
LABEL_211:
          v145 = v43 - 24;
          if ( *(_QWORD *)(v43 + 24) || (v44 & 0x8000u) != 0 )
          {
            if ( sub_1625790(v43 - 24, 9) )
              goto LABEL_49;
            v45 = *(__int64 **)(v43 - 72);
            v46 = *(_QWORD *)(v42 + 56);
            v47 = *v45;
            v48 = *(_BYTE *)(*v45 + 8);
          }
          v146 = *(_QWORD *)(v43 - 48);
          v147 = 1;
          while ( 2 )
          {
            switch ( v48 )
            {
              case 0:
              case 8:
              case 10:
              case 12:
              case 16:
                v178 = *(_QWORD *)(v47 + 32);
                v47 = *(_QWORD *)(v47 + 24);
                v147 *= v178;
                v48 = *(_BYTE *)(v47 + 8);
                continue;
              case 1:
                v162 = 16;
                break;
              case 2:
                v162 = 32;
                break;
              case 3:
              case 9:
                v162 = 64;
                break;
              case 4:
                v162 = 80;
                break;
              case 5:
              case 6:
                v162 = 128;
                break;
              case 7:
                v255 = v45;
                v179 = sub_15A9520(v46, 0);
                v45 = v255;
                v162 = (unsigned int)(8 * v179);
                break;
              case 11:
                v162 = *(_DWORD *)(v47 + 8) >> 8;
                break;
              case 13:
                v254 = v45;
                v177 = (_QWORD *)sub_15A9930(v46, v47);
                v45 = v254;
                v162 = 8LL * *v177;
                break;
              case 14:
                v229 = v45;
                v232 = *(_QWORD *)(v47 + 24);
                v253 = *(_QWORD *)(v47 + 32);
                v244 = (unsigned int)sub_15A9FE0(v46, v232);
                v176 = sub_127FA20(v46, v232);
                v45 = v229;
                v162 = 8 * v253 * v244 * ((v244 + ((unsigned __int64)(v176 + 7) >> 3) - 1) / v244);
                break;
              case 15:
                v252 = v45;
                v175 = sub_15A9520(v46, *(_DWORD *)(v47 + 8) >> 8);
                v45 = v252;
                v162 = (unsigned int)(8 * v175);
                break;
            }
            break;
          }
          v250 = (unsigned __int8 *)v45;
          if ( ((unsigned __int64)(v162 * v147) >> 32) | ((_BYTE)v162 * (_BYTE)v147) & 7 )
            goto LABEL_49;
          v163 = sub_146F1B0(*(_QWORD *)(v42 + 32), v146);
          v243 = v163;
          if ( *(_WORD *)(v163 + 24) != 7
            || *(_QWORD *)(v163 + 48) != *(_QWORD *)v42
            || *(_QWORD *)(v163 + 40) != 2
            || *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v163 + 32) + 8LL) + 24LL) )
          {
            goto LABEL_49;
          }
          v164 = sub_14ABE30(v250);
          v165 = (__int64)v250;
          v166 = v164;
          v167 = *(_WORD *)(v43 - 6);
          if ( ((v167 >> 7) & 6) == 0 && (v167 & 1) == 0 )
          {
            v236 = v250;
            v251 = sub_15F32D0(v43 - 24);
            if ( v251 )
              goto LABEL_263;
            v251 = *(_BYTE *)(v43 - 6) & 1;
            if ( v251 )
              goto LABEL_263;
            v165 = (__int64)v236;
          }
          if ( *(_BYTE *)(v42 + 264) )
          {
            if ( v166 )
            {
              v264 = v165;
              v214 = sub_13FC1A0(*(_QWORD *)v42, v166);
              v165 = v264;
              if ( v214 )
              {
                v310 = sub_14AD280(*(_QWORD *)(v43 - 48), *(_QWORD *)(v42 + 56), 6u);
                v209 = v276;
                goto LABEL_357;
              }
            }
          }
          if ( *(_BYTE *)(v42 + 265) )
          {
            v168 = *(_QWORD *)v146;
            if ( *(_BYTE *)(*(_QWORD *)v146 + 8LL) == 16 )
              v168 = **(_QWORD **)(v168 + 16);
            if ( !(*(_DWORD *)(v168 + 8) >> 8) && sub_1969620(v165, *(_BYTE **)(v42 + 56)) )
            {
              v310 = sub_14AD280(*(_QWORD *)(v43 - 48), *(_QWORD *)(v42 + 56), 6u);
              v209 = v265;
LABEL_357:
              v212 = sub_196D8D0(v209, &v310, v205, v206, v207, v208);
              v213 = *(unsigned int *)(v212 + 8);
              if ( (unsigned int)v213 >= *(_DWORD *)(v212 + 12) )
              {
                sub_16CD150(v212, (const void *)(v212 + 16), 0, 8, v210, v211);
                v213 = *(unsigned int *)(v212 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v212 + 8 * v213) = v145;
              ++*(_DWORD *)(v212 + 8);
              goto LABEL_49;
            }
          }
          v251 = 0;
LABEL_263:
          if ( *(_BYTE *)(v42 + 266) )
          {
            v169 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v243 + 32) + 8LL) + 32LL);
            LODWORD(v294) = *(_DWORD *)(v169 + 32);
            if ( (unsigned int)v294 > 0x40 )
              sub_16A4FD0((__int64)&v293, (const void **)(v169 + 24));
            else
              v293 = *(_QWORD *)(v169 + 24);
            v170 = (unsigned int)((unsigned __int64)(sub_127FA20(*(_QWORD *)(v42 + 56), **(_QWORD **)(v43 - 72)) + 7) >> 3);
            if ( sub_13A38F0((__int64)&v293, (_QWORD *)v170) )
              goto LABEL_267;
            v203 = (char)v294;
            LODWORD(v300) = (_DWORD)v294;
            if ( (unsigned int)v294 <= 0x40 )
            {
              v299 = v293;
              goto LABEL_349;
            }
            sub_16A4FD0((__int64)&v299, (const void **)&v293);
            v203 = v300;
            if ( (unsigned int)v300 <= 0x40 )
LABEL_349:
              v299 = ~v299 & (0xFFFFFFFFFFFFFFFFLL >> -v203);
            else
              sub_16A8F40((__int64 *)&v299);
            sub_16A7400((__int64)&v299);
            LODWORD(v311) = v300;
            LODWORD(v300) = 0;
            v310 = v299;
            v204 = sub_13A38F0((__int64)&v310, (_QWORD *)v170);
            sub_135E100(&v310);
            sub_135E100((__int64 *)&v299);
            if ( v204 )
            {
LABEL_267:
              v171 = *(_QWORD *)(v43 - 72);
              if ( *(_BYTE *)(v171 + 16) == 54 )
              {
                v172 = *(unsigned __int16 *)(v171 + 18);
                if ( (v172 & 1) == 0 && ((v172 >> 7) & 6) == 0 )
                {
                  v173 = sub_146F1B0(*(_QWORD *)(v42 + 32), *(_QWORD *)(v171 - 24));
                  if ( *(_WORD *)(v173 + 24) == 7
                    && *(_QWORD *)(v173 + 48) == *(_QWORD *)v42
                    && *(_QWORD *)(v173 + 40) == 2
                    && *(_QWORD *)(*(_QWORD *)(v243 + 32) + 8LL) == *(_QWORD *)(*(_QWORD *)(v173 + 32) + 8LL) )
                  {
                    if ( !v251 )
                      sub_15F32D0(v171);
                    sub_135E100((__int64 *)&v293);
                    v217 = *(unsigned int *)(v42 + 192);
                    if ( (unsigned int)v217 >= *(_DWORD *)(v42 + 196) )
                    {
                      sub_16CD150(v42 + 184, (const void *)(v42 + 200), 0, 8, v215, v216);
                      v217 = *(unsigned int *)(v42 + 192);
                    }
                    *(_QWORD *)(*(_QWORD *)(v42 + 184) + 8 * v217) = v145;
                    ++*(_DWORD *)(v42 + 192);
                    goto LABEL_49;
                  }
                }
              }
            }
            sub_135E100((__int64 *)&v293);
          }
LABEL_49:
          v43 = *(_QWORD *)(v43 + 8);
          if ( v249 == v43 )
          {
            v16 = v42;
            goto LABEL_51;
          }
        }
        ++v49;
LABEL_301:
        if ( v51 != *v49 )
        {
          ++v49;
LABEL_210:
          if ( v51 != *v49 )
            goto LABEL_211;
          goto LABEL_48;
        }
        goto LABEL_48;
      }
LABEL_51:
      v56 = *(_QWORD *)(v16 + 104);
      v57 = *(_QWORD *)(v16 + 112);
      v58 = 0;
      v223 = 0;
      if ( v56 != v57 )
      {
        do
        {
          v59 = sub_196E000(v16, v56 + 8, v240, 1, a2, a3, a4, a5, v36, v37, a8, a9);
          v56 += 88;
          v58 |= v59;
        }
        while ( v57 != v56 );
        v223 = v58;
      }
      v60 = *(_QWORD *)(v16 + 160);
      v61 = *(_QWORD *)(v16 + 168);
      if ( v60 != v61 )
      {
        v62 = v223;
        do
        {
          v63 = sub_196E000(v16, v60 + 8, v240, 0, a2, a3, a4, a5, v36, v37, a8, a9);
          v60 += 88;
          v62 |= v63;
        }
        while ( v61 != v60 );
        v223 = v62;
      }
      v64 = *(__int64 **)(v16 + 184);
      v219 = &v64[*(unsigned int *)(v16 + 192)];
      if ( v64 != v219 )
      {
        v241 = *(__int64 **)(v16 + 184);
        v65 = v16;
        while ( 2 )
        {
          v277 = *v241;
          v66 = sub_146F1B0(*(_QWORD *)(v65 + 32), *(_QWORD *)(*v241 - 24));
          v67 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v66 + 32) + 8LL) + 32LL);
          v287 = *(_DWORD *)(v67 + 32);
          if ( v287 > 0x40 )
            sub_16A4FD0((__int64)&v286, (const void **)(v67 + 24));
          else
            v286 = *(_QWORD **)(v67 + 24);
          v68 = *(_QWORD *)(v65 + 56);
          v69 = 1;
          v70 = **(_QWORD **)(v277 - 48);
LABEL_63:
          switch ( *(_BYTE *)(v70 + 8) )
          {
            case 1:
              v94 = 16;
              goto LABEL_109;
            case 2:
              v94 = 32;
              goto LABEL_109;
            case 3:
            case 9:
              v94 = 64;
              goto LABEL_109;
            case 4:
              v94 = 80;
              goto LABEL_109;
            case 5:
            case 6:
              v94 = 128;
              goto LABEL_109;
            case 7:
              v94 = 8 * (unsigned int)sub_15A9520(*(_QWORD *)(v65 + 56), 0);
              goto LABEL_109;
            case 0xB:
              v94 = *(_DWORD *)(v70 + 8) >> 8;
              goto LABEL_109;
            case 0xD:
              v94 = 8LL * *(_QWORD *)sub_15A9930(*(_QWORD *)(v65 + 56), v70);
              goto LABEL_109;
            case 0xE:
              v134 = *(_QWORD *)(v70 + 32);
              v269 = *(_QWORD *)(v70 + 24);
              v135 = sub_15A9FE0(*(_QWORD *)(v65 + 56), v269);
              v136 = v269;
              v137 = 1;
              v138 = v135;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v136 + 8) )
                {
                  case 1:
                    v174 = 16;
                    goto LABEL_275;
                  case 2:
                    v174 = 32;
                    goto LABEL_275;
                  case 3:
                  case 9:
                    v174 = 64;
                    goto LABEL_275;
                  case 4:
                    v174 = 80;
                    goto LABEL_275;
                  case 5:
                  case 6:
                    v174 = 128;
                    goto LABEL_275;
                  case 7:
                    v258 = v138;
                    v182 = 0;
                    v272 = v137;
                    goto LABEL_289;
                  case 0xB:
                    v174 = *(_DWORD *)(v136 + 8) >> 8;
                    goto LABEL_275;
                  case 0xD:
                    v257 = v138;
                    v271 = v137;
                    v181 = (_QWORD *)sub_15A9930(v68, v136);
                    v137 = v271;
                    v138 = v257;
                    v174 = 8LL * *v181;
                    goto LABEL_275;
                  case 0xE:
                    v230 = v138;
                    v233 = v137;
                    v237 = *(_QWORD *)(v136 + 24);
                    v270 = *(_QWORD *)(v136 + 32);
                    v256 = (unsigned int)sub_15A9FE0(v68, v237);
                    v180 = sub_127FA20(v68, v237);
                    v137 = v233;
                    v138 = v230;
                    v174 = 8 * v270 * v256 * ((v256 + ((unsigned __int64)(v180 + 7) >> 3) - 1) / v256);
                    goto LABEL_275;
                  case 0xF:
                    v258 = v138;
                    v272 = v137;
                    v182 = *(_DWORD *)(v136 + 8) >> 8;
LABEL_289:
                    v183 = sub_15A9520(v68, v182);
                    v137 = v272;
                    v138 = v258;
                    v174 = (unsigned int)(8 * v183);
LABEL_275:
                    v94 = 8 * v138 * v134 * ((v138 + ((unsigned __int64)(v174 * v137 + 7) >> 3) - 1) / v138);
                    goto LABEL_109;
                  case 0x10:
                    v184 = *(_QWORD *)(v136 + 32);
                    v136 = *(_QWORD *)(v136 + 24);
                    v137 *= v184;
                    continue;
                  default:
                    goto LABEL_363;
                }
              }
            case 0xF:
              v94 = 8 * (unsigned int)sub_15A9520(*(_QWORD *)(v65 + 56), *(_DWORD *)(v70 + 8) >> 8);
LABEL_109:
              v95 = v287;
              LODWORD(v300) = v287;
              v268 = (unsigned __int64)(v94 * v69 + 7) >> 3;
              if ( v287 <= 0x40 )
              {
                v96 = (unsigned __int64)v286;
LABEL_111:
                v299 = ~v96 & (0xFFFFFFFFFFFFFFFFLL >> -v95);
                goto LABEL_112;
              }
              sub_16A4FD0((__int64)&v299, (const void **)&v286);
              v95 = v300;
              if ( (unsigned int)v300 <= 0x40 )
              {
                v96 = v299;
                goto LABEL_111;
              }
              sub_16A8F40((__int64 *)&v299);
LABEL_112:
              sub_16A7400((__int64)&v299);
              v97 = v300;
              v98 = (_QWORD *)v299;
              LODWORD(v300) = 0;
              LODWORD(v311) = v97;
              v310 = v299;
              if ( v97 <= 0x40 )
              {
                v227 = (unsigned int)v268 == v299;
                goto LABEL_114;
              }
              if ( v97 - (unsigned int)sub_16A57B0((__int64)&v310) <= 0x40 && (unsigned int)v268 == *v98 )
              {
                v227 = 1;
              }
              else
              {
                v227 = 0;
                if ( !v98 )
                  goto LABEL_114;
              }
              j_j___libc_free_0_0(v98);
              if ( (unsigned int)v300 > 0x40 && v299 )
                j_j___libc_free_0_0(v299);
LABEL_114:
              v231 = *(_QWORD *)(v277 - 48);
              v224 = sub_146F1B0(*(_QWORD *)(v65 + 32), *(_QWORD *)(v231 - 24));
              v99 = sub_13FC520(*(_QWORD *)v65);
              v100 = sub_157EBA0(v99);
              v101 = (_QWORD *)sub_16498A0(v100);
              v299 = 0;
              v302 = v101;
              v303 = 0;
              v304 = 0;
              v305 = 0;
              v306 = 0;
              v300 = *(_QWORD *)(v100 + 40);
              v301 = v100 + 24;
              v102 = *(unsigned __int8 **)(v100 + 48);
              v310 = (__int64)v102;
              if ( v102 )
              {
                sub_1623A60((__int64)&v310, (__int64)v102, 2);
                if ( v299 )
                  sub_161E7C0((__int64)&v299, v299);
                v299 = v310;
                if ( v310 )
                  sub_1623210((__int64)&v310, (unsigned __int8 *)v310, (__int64)&v299);
              }
              v103 = *(_QWORD *)(v65 + 32);
              v104 = *(const char **)(v65 + 56);
              v330 = v334;
              v331 = v334;
              v310 = v103;
              v312 = "loop-idiom";
              v311 = v104;
              v313 = 0;
              v314 = 0;
              v315 = 0;
              v316 = 0;
              v317 = 0;
              v318 = 0;
              v319 = 0;
              v320 = 0;
              v321 = 0;
              v322 = 0;
              v323 = 0;
              v324 = 0;
              v325 = 0;
              v326 = 0;
              v327 = 0;
              v328 = 0;
              v329 = 0;
              v332 = 2;
              v333 = 0;
              v335 = 0;
              v336 = 0;
              v337 = 0;
              v338 = 0;
              v339 = 0;
              v340 = 0;
              v341 = 1;
              v105 = sub_15E0530(*(_QWORD *)(v103 + 24));
              v346 = v104;
              v342[3] = v105;
              v347 = v349;
              memset(v342, 0, 24);
              v342[4] = 0;
              v343 = 0;
              v344 = 0;
              v345 = 0;
              v348 = 0x800000000LL;
              v106 = **(_QWORD **)(v66 + 32);
              v107 = **(_QWORD **)(v277 - 24);
              if ( *(_BYTE *)(v107 + 8) == 16 )
                v107 = **(_QWORD **)(v107 + 16);
              v108 = *(_DWORD *)(v107 + 8) >> 8;
              v109 = sub_15A9620(*(_QWORD *)(v65 + 56), (__int64)v302, v108);
              v235 = v109;
              if ( v227 )
                v106 = sub_1969510(v106, v240, v109, v268, *(_QWORD **)(v65 + 32), a2, a3);
              v225 = sub_157EBA0(v99);
              v110 = sub_16471D0(v302, v108);
              v111 = sub_38767A0(&v310, v106, v110, v225);
              v112 = *(_QWORD **)(v65 + 8);
              v113 = *(_QWORD *)v65;
              v221 = (_QWORD *)v111;
              v294 = &v298;
              v295 = &v298;
              v296 = 0x100000001LL;
              v297 = 0;
              v298 = v277;
              v293 = 1;
              if ( (unsigned __int8)sub_1969150(v111, 7u, v113, v240, v268, v112, (__int64)&v293) )
              {
                sub_196A390((__int64)&v310);
                sub_1AEB370(v221, *(_QWORD *)(v65 + 40));
              }
              else
              {
                v114 = **(_QWORD **)(v224 + 32);
                v115 = **(_QWORD **)(v231 - 24);
                if ( *(_BYTE *)(v115 + 8) == 16 )
                  v115 = **(_QWORD **)(v115 + 16);
                v116 = *(_DWORD *)(v115 + 8) >> 8;
                if ( v227 )
                  v114 = sub_1969510(**(_QWORD **)(v224 + 32), v240, v235, v268, *(_QWORD **)(v65 + 32), a2, a3);
                v226 = v114;
                v228 = sub_157EBA0(v99);
                v117 = sub_16471D0(v302, v116);
                v118 = (_QWORD *)sub_38767A0(&v310, v226, v117, v228);
                if ( (unsigned __int8)sub_1969150(
                                        (__int64)v118,
                                        6u,
                                        *(_QWORD *)v65,
                                        v240,
                                        v268,
                                        *(_QWORD **)(v65 + 8),
                                        (__int64)&v293) )
                {
                  sub_196A390((__int64)&v310);
                  sub_1AEB370(v118, *(_QWORD *)(v65 + 40));
                  sub_1AEB370(v221, *(_QWORD *)(v65 + 40));
                }
                else
                {
                  v119 = *(_QWORD **)v65;
                  if ( !*(_BYTE *)(v65 + 64) || (unsigned int)((__int64)(v119[5] - v119[4]) >> 3) <= 1 || *v119 )
                  {
                    v120 = sub_19699C0(
                             v240,
                             v235,
                             v268,
                             (__int64)v119,
                             *(_QWORD *)(v65 + 56),
                             *(_QWORD **)(v65 + 32),
                             a2,
                             a3);
                    v121 = sub_157EBA0(v99);
                    v122 = (__int64 *)sub_38767A0(&v310, v120, v235, v121);
                    if ( !sub_15F32D0(v277) && !sub_15F32D0(v231) )
                    {
                      v124 = sub_15E7430(
                               (__int64 *)&v299,
                               v221,
                               1 << (*(unsigned __int16 *)(v277 + 18) >> 1) >> 1,
                               v118,
                               1 << (*(unsigned __int16 *)(v231 + 18) >> 1) >> 1,
                               v122,
                               0,
                               0,
                               0,
                               0,
                               0);
LABEL_138:
                      v125 = (const void **)(v124 + 6);
                      v126 = *(unsigned __int8 **)(v277 + 48);
                      v284 = v126;
                      if ( v126 )
                      {
                        sub_1623A60((__int64)&v284, (__int64)v126, 2);
                        if ( v125 == (const void **)&v284 )
                        {
                          if ( v284 )
                            sub_161E7C0((__int64)(v124 + 6), (__int64)v284);
                          goto LABEL_142;
                        }
                        v148 = v124[6];
                        if ( !v148 )
                        {
LABEL_225:
                          v149 = v284;
                          v124[6] = v284;
                          if ( v149 )
                            sub_1623210((__int64)&v284, v149, (__int64)(v124 + 6));
                          goto LABEL_142;
                        }
                      }
                      else if ( v125 == (const void **)&v284 || (v148 = v124[6]) == 0 )
                      {
LABEL_142:
                        v127 = sub_1599EF0(*(__int64 ***)v277);
                        sub_164D160(v277, v127, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v128, v129, a8, a9);
                        sub_15F20C0((_QWORD *)v277);
                        v223 = 1;
                        goto LABEL_143;
                      }
                      sub_161E7C0((__int64)(v124 + 6), v148);
                      goto LABEL_225;
                    }
                    v123 = 1 << (*(unsigned __int16 *)(v231 + 18) >> 1) >> 1;
                    if ( v123 > 1 << (*(unsigned __int16 *)(v277 + 18) >> 1) >> 1 )
                      v123 = 1 << (*(unsigned __int16 *)(v277 + 18) >> 1) >> 1;
                    if ( (unsigned int)v268 <= v123
                      && (unsigned int)sub_14A3710(*(_QWORD *)(v65 + 48)) >= (unsigned int)v268 )
                    {
                      v124 = sub_15E76C0(
                               (__int64 *)&v299,
                               v221,
                               1 << (*(unsigned __int16 *)(v277 + 18) >> 1) >> 1,
                               v118,
                               1 << (*(unsigned __int16 *)(v231 + 18) >> 1) >> 1,
                               v122,
                               v268,
                               0,
                               0,
                               0,
                               0);
                      goto LABEL_138;
                    }
                  }
                }
              }
LABEL_143:
              if ( v295 != v294 )
                _libc_free((unsigned __int64)v295);
              if ( v347 != v349 )
                _libc_free((unsigned __int64)v347);
              if ( v342[0] )
                sub_161E7C0((__int64)v342, v342[0]);
              j___libc_free_0(v338);
              if ( v331 != v330 )
                _libc_free((unsigned __int64)v331);
              j___libc_free_0(v326);
              j___libc_free_0(v322);
              j___libc_free_0(v318);
              if ( v316 )
              {
                v130 = v314;
                v131 = &v314[5 * v316];
                do
                {
                  while ( *v130 == -8 )
                  {
                    if ( v130[1] != -8 )
                      goto LABEL_154;
                    v130 += 5;
                    if ( v131 == v130 )
                      goto LABEL_161;
                  }
                  if ( *v130 != -16 || v130[1] != -16 )
                  {
LABEL_154:
                    v132 = v130[4];
                    if ( v132 != -8 && v132 != 0 && v132 != -16 )
                      sub_1649B30(v130 + 2);
                  }
                  v130 += 5;
                }
                while ( v131 != v130 );
              }
LABEL_161:
              j___libc_free_0(v314);
              if ( v299 )
                sub_161E7C0((__int64)&v299, v299);
              if ( v287 > 0x40 && v286 )
                j_j___libc_free_0_0(v286);
              if ( v219 != ++v241 )
                continue;
              v16 = v65;
              break;
            case 0x10:
              v133 = *(_QWORD *)(v70 + 32);
              v70 = *(_QWORD *)(v70 + 24);
              v69 *= v133;
              goto LABEL_63;
            default:
LABEL_363:
              BUG();
          }
          break;
        }
      }
LABEL_84:
      v82 = *(_QWORD *)(v220 + 48);
LABEL_85:
      if ( v82 == v249 )
      {
LABEL_107:
        v222 |= v223;
        goto LABEL_7;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v83 = v82;
          v82 = *(_QWORD *)(v82 + 8);
          if ( *(_BYTE *)(v83 - 8) == 78 )
          {
            v84 = *(_QWORD *)(v83 - 48);
            if ( !*(_BYTE *)(v84 + 16) && (*(_BYTE *)(v84 + 33) & 0x20) != 0 && *(_DWORD *)(v84 + 36) == 137 )
              break;
          }
          if ( v82 == v249 )
            goto LABEL_107;
        }
        if ( v82 )
        {
          v299 = 6;
          v300 = 0;
          v301 = v82 - 24;
          if ( v82 != 8 && v82 != 16 )
            sub_164C220((__int64)&v299);
        }
        else
        {
          v299 = 6;
          v300 = 0;
          v301 = 0;
        }
        v85 = v83 - 24;
        v86 = *(_DWORD *)(v83 - 4) & 0xFFFFFFF;
        v87 = *(_QWORD *)(v83 - 24 + 24 * (3 - v86));
        v88 = *(_DWORD *)(v87 + 32);
        if ( v88 <= 0x40 )
        {
          v90 = *(_QWORD *)(v87 + 24) == 0;
        }
        else
        {
          v89 = sub_16A57B0(v87 + 24);
          v85 = v83 - 24;
          v90 = v88 == v89;
        }
        if ( !v90 )
          goto LABEL_103;
        if ( *(_BYTE *)(*(_QWORD *)(v85 + 24 * (2 - v86)) + 16LL) != 13 )
          goto LABEL_103;
        if ( !*(_BYTE *)(v16 + 264) )
          goto LABEL_103;
        v267 = *(_BYTE *)(v16 + 264);
        v279 = v85;
        v91 = (__int64 *)sub_1649C60(*(_QWORD *)(v85 - 24 * v86));
        v92 = sub_146F1B0(*(_QWORD *)(v16 + 32), (__int64)v91);
        v79 = v279;
        v93 = v267;
        v78 = v92;
        if ( *(_WORD *)(v92 + 24) != 7 || *(_QWORD *)(v92 + 48) != *(_QWORD *)v16 || *(_QWORD *)(v92 + 40) != 2 )
          goto LABEL_103;
        v71 = *(_QWORD *)(v279 + 24 * (2LL - (*(_DWORD *)(v83 - 4) & 0xFFFFFFF)));
        v72 = *(_QWORD *)(v71 + 24);
        if ( *(_DWORD *)(v71 + 32) > 0x40u )
          v72 = *(_QWORD *)v72;
        if ( HIDWORD(v72) || (v73 = *(_QWORD *)(*(_QWORD *)(v78 + 32) + 8LL), *(_WORD *)(v73 + 24)) )
        {
LABEL_103:
          v81 = v301;
          goto LABEL_104;
        }
        v74 = *(_QWORD *)(v73 + 32);
        v75 = *(_DWORD *)(v74 + 32);
        v285 = v75;
        if ( v75 <= 0x40 )
          break;
        v259 = v78;
        sub_16A4FD0((__int64)&v284, (const void **)(v74 + 24));
        v75 = v285;
        v93 = v267;
        v78 = v259;
        v79 = v279;
        if ( v285 <= 0x40 )
          goto LABEL_74;
        v245 = v285;
        v185 = sub_16A57B0((__int64)&v284);
        v78 = v259;
        v79 = v279;
        if ( v245 - v185 <= 0x40 && v72 == *(_QWORD *)v284 )
          goto LABEL_311;
        LODWORD(v294) = v245;
        sub_16A4FD0((__int64)&v293, (const void **)&v284);
        LOBYTE(v75) = (_BYTE)v294;
        v93 = v267;
        v78 = v259;
        v79 = v279;
        if ( (unsigned int)v294 <= 0x40 )
        {
          v76 = v293;
          goto LABEL_76;
        }
        sub_16A8F40((__int64 *)&v293);
        v79 = v279;
        v78 = v259;
        v93 = v267;
LABEL_77:
        v266 = v79;
        v278 = v78;
        v234 = v93;
        sub_16A7400((__int64)&v293);
        v77 = (unsigned int)v294;
        LODWORD(v294) = 0;
        v78 = v278;
        v79 = v266;
        LODWORD(v311) = v77;
        v310 = v293;
        v242 = v77;
        if ( v77 > 0x40 )
        {
          v263 = v266;
          v274 = (_QWORD *)v293;
          v200 = sub_16A57B0((__int64)&v310);
          v78 = v278;
          v79 = v263;
          v201 = v234;
          if ( v242 - v200 <= 0x40 )
            v201 = *v274 != v72;
          if ( v310 )
          {
            v275 = v278;
            v283 = v201;
            j_j___libc_free_0_0(v310);
            v201 = v283;
            v78 = v275;
            v79 = v263;
            if ( (unsigned int)v294 > 0x40 )
            {
              if ( v293 )
              {
                j_j___libc_free_0_0(v293);
                v79 = v263;
                v78 = v275;
                v201 = v283;
              }
            }
          }
          if ( v201 )
          {
LABEL_79:
            v80 = 0;
            if ( v285 <= 0x40 )
              goto LABEL_103;
LABEL_80:
            if ( v284 )
              j_j___libc_free_0_0(v284);
            goto LABEL_82;
          }
        }
        else if ( v72 != v293 )
        {
          goto LABEL_79;
        }
LABEL_311:
        v260 = v78;
        v280 = v79;
        v186 = *(_QWORD *)(v79 + 24 * (1LL - (*(_DWORD *)(v83 - 4) & 0xFFFFFFF)));
        v273 = (unsigned __int8 *)v186;
        if ( !v186 || !sub_13FC1A0(*(_QWORD *)v16, v186) )
          goto LABEL_79;
        v187 = v280;
        LODWORD(v314) = 0;
        v311 = (const char *)&v315;
        v188 = v260;
        v312 = (const char *)&v315;
        v313 = 0x100000001LL;
        v189 = v285;
        v315 = v280;
        v287 = v285;
        v310 = 1;
        if ( v285 <= 0x40 )
        {
          v190 = (unsigned __int64)v284;
LABEL_315:
          v286 = (_QWORD *)(~v190 & (0xFFFFFFFFFFFFFFFFLL >> -v189));
          goto LABEL_316;
        }
        sub_16A4FD0((__int64)&v286, (const void **)&v284);
        v189 = v287;
        v188 = v260;
        v187 = v280;
        if ( v287 <= 0x40 )
        {
          v190 = (unsigned __int64)v286;
          goto LABEL_315;
        }
        sub_16A8F40((__int64 *)&v286);
        v187 = v280;
        v188 = v260;
LABEL_316:
        v246 = v187;
        v261 = v188;
        sub_16A7400((__int64)&v286);
        v191 = v287;
        v287 = 0;
        v192 = v261;
        v193 = v246;
        LODWORD(v294) = v191;
        v293 = (unsigned __int64)v286;
        v281 = v191;
        if ( v191 <= 0x40 )
        {
          v282 = v72 == (_QWORD)v286;
        }
        else
        {
          v238 = v246;
          v247 = v286;
          v194 = sub_16A57B0((__int64)&v293);
          v195 = v281;
          v192 = v261;
          v282 = 0;
          v193 = v238;
          if ( v195 - v194 <= 0x40 )
            v282 = *v247 == v72;
          if ( v247 )
          {
            j_j___libc_free_0_0(v247);
            v192 = v261;
            v193 = v238;
            if ( v287 > 0x40 )
            {
              if ( v286 )
              {
                j_j___libc_free_0_0(v286);
                v193 = v238;
                v192 = v261;
              }
            }
          }
        }
        v248 = v193;
        v262 = v192;
        v196 = sub_15603A0((_QWORD *)(v83 + 32), 0);
        v80 = sub_196B740(
                v16,
                v91,
                v72,
                v196,
                v273,
                v248,
                a2,
                a3,
                a4,
                a5,
                v197,
                v198,
                a8,
                a9,
                (__int64)&v310,
                v262,
                v240,
                v282,
                1);
        if ( v312 != v311 )
          _libc_free((unsigned __int64)v312);
        if ( v285 > 0x40 )
          goto LABEL_80;
LABEL_82:
        v81 = v301;
        if ( v80 )
        {
          v223 = v80;
          if ( !v301 )
            goto LABEL_84;
        }
LABEL_104:
        if ( v81 == -8 || v81 == 0 || v81 == -16 )
          goto LABEL_85;
        sub_1649B30(&v299);
        if ( v82 == v249 )
          goto LABEL_107;
      }
      v284 = *(unsigned __int8 **)(v74 + 24);
LABEL_74:
      v76 = (unsigned __int64)v284;
      if ( (unsigned __int8 *)v72 == v284 )
        goto LABEL_311;
      LODWORD(v294) = v75;
LABEL_76:
      v293 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v75) & ~v76;
      goto LABEL_77;
    }
  }
  v222 = 0;
LABEL_180:
  if ( v292 )
  {
    v139 = v290;
    v140 = &v290[2 * v292];
    do
    {
      if ( *v139 != -16 && *v139 != -8 )
      {
        v141 = v139[1];
        if ( (v141 & 4) != 0 )
        {
          v142 = (unsigned __int64 *)(v141 & 0xFFFFFFFFFFFFFFF8LL);
          v143 = v142;
          if ( v142 )
          {
            if ( (unsigned __int64 *)*v142 != v142 + 2 )
              _libc_free(*v142);
            j_j___libc_free_0(v143, 48);
          }
        }
      }
      v139 += 2;
    }
    while ( v140 != v139 );
  }
  j___libc_free_0(v290);
  if ( v307 != v309 )
    _libc_free((unsigned __int64)v307);
  return v222;
}
