// Function: sub_26AF660
// Address: 0x26af660
//
__int64 __fastcall sub_26AF660(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v4; // r12d
  unsigned __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // r14
  signed __int64 v9; // rax
  int v10; // ecx
  int v11; // edx
  __int64 v12; // r15
  __int64 i; // r13
  __int64 v14; // rdi
  signed __int64 v15; // rax
  int v16; // edx
  bool v17; // of
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rbx
  unsigned __int64 v26; // rbx
  __int64 v27; // r14
  __int64 v28; // rsi
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rax
  unsigned int v31; // r12d
  __int64 *v32; // r15
  __int64 v33; // r14
  __int64 v34; // r15
  unsigned __int64 v35; // rdi
  unsigned int v36; // esi
  int v37; // eax
  __int64 v39; // rax
  char *v40; // rbx
  char v41; // al
  __int64 v42; // rsi
  __int64 v43; // rbx
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 v47; // r14
  __int64 v48; // r14
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __m128i v52; // xmm2
  __m128i v53; // xmm5
  __int64 v54; // r9
  unsigned __int64 *v55; // r14
  __int64 v56; // rax
  unsigned __int64 *v57; // r15
  unsigned __int64 v58; // rdi
  __int64 *v59; // r12
  __int64 v60; // rax
  __int64 v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rsi
  _BYTE *v66; // rbx
  char v67; // al
  __int64 v68; // rax
  __int64 v69; // r14
  __int64 v70; // r13
  unsigned __int8 *v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r9
  _QWORD *v76; // rbx
  _QWORD *v77; // r12
  __int64 v78; // rax
  unsigned __int64 *v79; // rbx
  unsigned __int64 *v80; // r12
  unsigned __int64 v81; // rdi
  __int64 *v82; // r12
  __int64 **v83; // r13
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __m128i v89; // xmm6
  __m128i v90; // xmm7
  __m128i v91; // xmm3
  __int64 *v92; // r12
  unsigned __int8 *v93; // r13
  __int64 v94; // rax
  __int64 v95; // rdi
  __int64 *v96; // r14
  __int64 v97; // r12
  __int64 v98; // rax
  __int64 v99; // rax
  char v100; // al
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r15
  __int64 v104; // rax
  signed int v105; // r12d
  __int64 v106; // r12
  __int64 v107; // rax
  __int64 v108; // r12
  __int64 v109; // r12
  __int64 v110; // r12
  __int64 v111; // r12
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 v114; // r9
  __m128i v115; // xmm0
  __m128i v116; // xmm2
  __int64 v117; // rdx
  unsigned __int64 *v118; // r12
  unsigned __int64 *v119; // r15
  unsigned __int64 v120; // rdi
  _BYTE *v121; // rdi
  _QWORD *v122; // rbx
  _QWORD *v123; // r12
  __int64 v124; // rax
  unsigned __int64 *v125; // rbx
  unsigned __int64 *v126; // r12
  unsigned __int64 v127; // rdi
  __int64 *v128; // r12
  __int64 v129; // r12
  __int64 v130; // rax
  __int64 v131; // r12
  __int64 v132; // r12
  __int64 v133; // r12
  __int64 v134; // r12
  __int64 v135; // rdx
  __int64 v136; // rcx
  __int64 v137; // r8
  __int64 v138; // r9
  __m128i v139; // xmm3
  __m128i v140; // xmm5
  __int64 *v141; // r12
  __int64 *v142; // r15
  unsigned __int64 v143; // rdi
  __int64 v144; // rax
  __int64 v145; // rbx
  __int64 v146; // rax
  __int64 v147; // rdx
  unsigned __int64 v148; // r13
  __int64 v149; // r14
  __int64 *v150; // r13
  void *v151; // r15
  _QWORD *v152; // r10
  int v153; // r11d
  unsigned int v154; // edx
  _QWORD *v155; // rax
  _BYTE *v156; // rcx
  _BYTE *v157; // rbx
  char v158; // al
  __int64 (__fastcall *v159)(_QWORD, __int64); // rax
  void *v160; // rax
  unsigned __int8 *v161; // rdx
  _QWORD *v162; // r9
  int v163; // r11d
  __int64 v164; // rdx
  _QWORD *v165; // rax
  _BYTE *v166; // rcx
  __int64 v167; // r14
  __int64 v168; // rax
  __int64 v169; // r14
  __int64 v170; // r13
  __int64 v171; // r13
  __int64 v172; // r12
  __int64 v173; // rdx
  __int64 v174; // r8
  __int64 v175; // r9
  __m128i v176; // xmm4
  __m128i v177; // xmm6
  __int64 v178; // rcx
  __int64 *v179; // r12
  __int64 v180; // rax
  unsigned __int64 v181; // rdi
  unsigned int v182; // r10d
  _QWORD *v183; // r9
  _BYTE *v184; // rsi
  unsigned __int64 v185; // rax
  __int64 v186; // r12
  __int64 v187; // rax
  __int64 v188; // r12
  __int64 v189; // rdx
  __int64 v190; // rcx
  __int64 v191; // r9
  __m128i v192; // xmm7
  __m128i v193; // xmm1
  __int64 v194; // r8
  __int64 v195; // r12
  __int64 v196; // rax
  __int64 v197; // rcx
  __int64 v198; // r8
  __int64 v199; // r9
  __int64 v200; // rax
  __m128i v201; // xmm7
  __m128i v202; // xmm1
  char *v203; // rax
  char *v204; // rax
  __int64 v205; // rsi
  __int64 *v206; // rax
  unsigned __int64 *v207; // r14
  unsigned __int64 *v208; // r15
  unsigned __int64 v209; // rbx
  __int64 *v210; // r12
  __int64 *v211; // r13
  __int64 v212; // rdi
  __int64 v213; // rax
  __int64 v214; // rax
  unsigned int v215; // eax
  __int64 v216; // rdx
  char v217; // al
  unsigned __int64 v218; // rdi
  unsigned __int64 v219; // rdi
  __int64 *v220; // r14
  __int64 *v221; // r12
  __int64 v222; // rsi
  __int64 v223; // rdi
  __int64 *v224; // r12
  __int64 *v225; // r12
  __int64 *v226; // r14
  unsigned __int64 v227; // r15
  unsigned __int64 v228; // rdi
  __int64 v229; // rax
  __int64 v230; // rax
  __int64 v231; // rax
  __int64 v232; // rax
  __int64 *v233; // rax
  __int64 v234; // r12
  __int64 *v235; // r14
  __int64 v236; // rdi
  __int64 v237; // rsi
  unsigned int v238; // ecx
  __int64 *v239; // r14
  __int64 v240; // rsi
  __int64 v241; // rdi
  __int64 v242; // rax
  __int64 v243; // rax
  void *v244; // rax
  int v245; // edx
  int v246; // edx
  __int64 v247; // rax
  __int64 v248; // rax
  unsigned int v249; // r12d
  __int64 v250; // rsi
  unsigned int v251; // ecx
  __int64 v252; // rdi
  _QWORD *v253; // rsi
  int v254; // edx
  unsigned int v255; // ecx
  __int64 v256; // r11
  _QWORD *v257; // rsi
  int v258; // edi
  _QWORD *v259; // rcx
  unsigned int v260; // r12d
  int v261; // esi
  __int64 v262; // r8
  __int64 v263; // rax
  __int64 v264; // rax
  __int64 v265; // rax
  __int64 v266; // rax
  int v267; // edi
  __int64 v268; // rax
  __int64 v269; // rax
  bool v270; // cc
  unsigned __int64 v271; // rax
  _QWORD *v272; // rcx
  int v273; // edx
  __int64 *v274; // [rsp+8h] [rbp-628h]
  __int64 v275; // [rsp+18h] [rbp-618h]
  char v276; // [rsp+27h] [rbp-609h]
  unsigned __int64 v277; // [rsp+30h] [rbp-600h]
  unsigned __int64 v278; // [rsp+38h] [rbp-5F8h]
  unsigned __int64 v279; // [rsp+40h] [rbp-5F0h]
  __int64 *v280; // [rsp+48h] [rbp-5E8h]
  _BYTE *v281; // [rsp+48h] [rbp-5E8h]
  unsigned __int64 v282; // [rsp+50h] [rbp-5E0h]
  unsigned __int8 v283; // [rsp+58h] [rbp-5D8h]
  __int64 v284; // [rsp+58h] [rbp-5D8h]
  _QWORD *v285; // [rsp+60h] [rbp-5D0h]
  __int64 v287; // [rsp+70h] [rbp-5C0h]
  __int64 *v288; // [rsp+70h] [rbp-5C0h]
  int v289; // [rsp+78h] [rbp-5B8h]
  char v290; // [rsp+78h] [rbp-5B8h]
  __int64 *v291; // [rsp+78h] [rbp-5B8h]
  _QWORD *v293; // [rsp+88h] [rbp-5A8h]
  __int64 *v294; // [rsp+88h] [rbp-5A8h]
  __int64 *v295; // [rsp+88h] [rbp-5A8h]
  unsigned __int64 v296; // [rsp+90h] [rbp-5A0h] BYREF
  int v297; // [rsp+98h] [rbp-598h]
  char v298; // [rsp+A0h] [rbp-590h]
  __int64 v299[2]; // [rsp+B0h] [rbp-580h] BYREF
  __int64 *v300; // [rsp+C0h] [rbp-570h]
  __int64 v301; // [rsp+D0h] [rbp-560h] BYREF
  __int64 v302; // [rsp+D8h] [rbp-558h]
  __int64 v303; // [rsp+E0h] [rbp-550h]
  unsigned int v304; // [rsp+E8h] [rbp-548h]
  __m128i v305; // [rsp+F0h] [rbp-540h] BYREF
  unsigned __int64 v306; // [rsp+108h] [rbp-528h]
  unsigned int v307; // [rsp+110h] [rbp-520h]
  unsigned __int64 v308; // [rsp+118h] [rbp-518h]
  unsigned int v309; // [rsp+120h] [rbp-510h]
  char v310; // [rsp+128h] [rbp-508h]
  __int64 v311[2]; // [rsp+130h] [rbp-500h] BYREF
  __int64 *v312[2]; // [rsp+140h] [rbp-4F0h] BYREF
  _QWORD *v313; // [rsp+150h] [rbp-4E0h]
  _QWORD v314[4]; // [rsp+160h] [rbp-4D0h] BYREF
  __int64 v315[2]; // [rsp+180h] [rbp-4B0h] BYREF
  _QWORD v316[2]; // [rsp+190h] [rbp-4A0h] BYREF
  _QWORD *v317; // [rsp+1A0h] [rbp-490h]
  _QWORD v318[4]; // [rsp+1B0h] [rbp-480h] BYREF
  unsigned __int64 v319[2]; // [rsp+1D0h] [rbp-460h] BYREF
  _QWORD v320[2]; // [rsp+1E0h] [rbp-450h] BYREF
  _QWORD *v321; // [rsp+1F0h] [rbp-440h] BYREF
  _QWORD v322[4]; // [rsp+200h] [rbp-430h] BYREF
  __int64 **v323; // [rsp+220h] [rbp-410h] BYREF
  __int64 v324; // [rsp+228h] [rbp-408h]
  __int64 *v325; // [rsp+230h] [rbp-400h] BYREF
  __int64 *v326; // [rsp+238h] [rbp-3F8h]
  unsigned __int64 v327; // [rsp+240h] [rbp-3F0h] BYREF
  __int64 v328; // [rsp+248h] [rbp-3E8h] BYREF
  _QWORD v329[2]; // [rsp+250h] [rbp-3E0h] BYREF
  __m128i v330[2]; // [rsp+260h] [rbp-3D0h] BYREF
  __int64 v331; // [rsp+280h] [rbp-3B0h]
  __int64 v332; // [rsp+288h] [rbp-3A8h]
  char v333; // [rsp+290h] [rbp-3A0h]
  __int64 v334; // [rsp+294h] [rbp-39Ch]
  unsigned __int8 *v335; // [rsp+2A0h] [rbp-390h] BYREF
  __int64 v336; // [rsp+2A8h] [rbp-388h]
  __int64 v337; // [rsp+2B0h] [rbp-380h] BYREF
  __m128i v338; // [rsp+2B8h] [rbp-378h] BYREF
  unsigned __int64 *v339; // [rsp+2C8h] [rbp-368h]
  __m128i v340; // [rsp+2D0h] [rbp-360h] BYREF
  __m128i v341; // [rsp+2E0h] [rbp-350h]
  __int64 *v342; // [rsp+2F0h] [rbp-340h] BYREF
  __int64 v343; // [rsp+2F8h] [rbp-338h] BYREF
  __int64 v344; // [rsp+300h] [rbp-330h] BYREF
  _BYTE v345[16]; // [rsp+308h] [rbp-328h] BYREF
  __int64 *v346; // [rsp+318h] [rbp-318h]
  unsigned int v347; // [rsp+320h] [rbp-310h]
  __int64 v348; // [rsp+328h] [rbp-308h] BYREF
  _BYTE *v349; // [rsp+3C8h] [rbp-268h]
  __int64 v350; // [rsp+3D0h] [rbp-260h]
  _BYTE v351[104]; // [rsp+3D8h] [rbp-258h] BYREF
  char v352; // [rsp+440h] [rbp-1F0h]
  int v353; // [rsp+444h] [rbp-1ECh]
  __int64 v354; // [rsp+448h] [rbp-1E8h]
  void *v355; // [rsp+450h] [rbp-1E0h] BYREF
  unsigned __int8 *v356; // [rsp+458h] [rbp-1D8h] BYREF
  __int64 v357; // [rsp+460h] [rbp-1D0h]
  __m128i v358; // [rsp+468h] [rbp-1C8h] BYREF
  unsigned __int64 *v359; // [rsp+478h] [rbp-1B8h]
  __m128i v360; // [rsp+480h] [rbp-1B0h] BYREF
  __m128i v361; // [rsp+490h] [rbp-1A0h] BYREF
  unsigned __int64 *v362; // [rsp+4A0h] [rbp-190h] BYREF
  __int64 v363; // [rsp+4A8h] [rbp-188h]
  __int64 v364; // [rsp+4B0h] [rbp-180h] BYREF
  char v365; // [rsp+4B8h] [rbp-178h] BYREF
  _QWORD v366[2]; // [rsp+4F8h] [rbp-138h] BYREF
  char v367; // [rsp+508h] [rbp-128h] BYREF
  _BYTE v368[140]; // [rsp+568h] [rbp-C8h] BYREF
  int v369; // [rsp+5F4h] [rbp-3Ch]
  __int64 v370; // [rsp+5F8h] [rbp-38h]

  v2 = *(unsigned int *)(a2 + 24);
  v283 = 0;
  if ( !(_DWORD)v2 )
    return v283;
  v4 = 0;
  v5 = 0;
  v289 = 0;
  v293 = *(_QWORD **)(a2 + 16);
  v285 = &v293[2 * v2];
  v287 = 0;
  do
  {
    v6 = *v293;
    v7 = v293[1];
    v8 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 40))(*(_QWORD *)(a1 + 48), *v293);
    v9 = sub_26ABAF0(v7, v8);
    v10 = 1;
    if ( v11 != 1 )
      v10 = v289;
    v289 = v10;
    if ( __OFADD__(v9, v287) )
    {
      v270 = v9 <= 0;
      v271 = 0x8000000000000000LL;
      if ( !v270 )
        v271 = 0x7FFFFFFFFFFFFFFFLL;
      v287 = v271;
    }
    else
    {
      v287 += v9;
    }
    v12 = *(_QWORD *)(v6 + 80);
    for ( i = v6 + 72; i != v12; v12 = *(_QWORD *)(v12 + 8) )
    {
      v14 = v12 - 24;
      if ( !v12 )
        v14 = 0;
      v15 = sub_26ABAF0(v14, v8);
      if ( v16 == 1 )
        v4 = 1;
      v17 = __OFADD__(v15, v5);
      v5 += v15;
      if ( v17 )
      {
        v5 = 0x8000000000000000LL;
        if ( v15 > 0 )
          v5 = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
    v293 += 2;
  }
  while ( v285 != v293 );
  v18 = sub_30D4FD0();
  v19 = *(unsigned int *)(a2 + 24);
  v20 = v19 * 2 * v18;
  v17 = __OFSUB__(v5, v20);
  v21 = v5 - v20;
  if ( v17 )
  {
    v21 = 0x8000000000000000LL;
    if ( v20 <= 0 )
      v21 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v22 = *(_QWORD *)(a2 + 104);
  if ( *(_DWORD *)(a2 + 112) != 1 )
  {
    v17 = __OFSUB__(v21, v22);
    v23 = v21 - v22;
    if ( !v17 )
      goto LABEL_19;
    if ( v22 > 0 )
    {
      if ( v4 != 1 )
      {
        v25 = v287 + 0x8000000000000000LL;
        if ( !__OFADD__(v287, 0x8000000000000000LL) )
          goto LABEL_21;
        goto LABEL_253;
      }
LABEL_256:
      v24 = v287 + 0x8000000000000000LL;
      if ( !__OFADD__(v287, 0x8000000000000000LL) )
        goto LABEL_20;
LABEL_253:
      v26 = (unsigned int)qword_4FF55A8 + 0x8000000000000000LL;
      goto LABEL_22;
    }
LABEL_318:
    v25 = 0x7FFFFFFFFFFFFFFFLL;
    v24 = v287 + 0x7FFFFFFFFFFFFFFFLL;
    if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v287) )
      goto LABEL_21;
    goto LABEL_20;
  }
  v17 = __OFSUB__(v21, v22);
  v23 = v21 - v22;
  if ( v17 )
  {
    if ( v22 > 0 )
      goto LABEL_256;
    goto LABEL_318;
  }
LABEL_19:
  v24 = v23 + v287;
  if ( __OFADD__(v23, v287) )
  {
    if ( v23 > 0 )
    {
      v25 = 0x7FFFFFFFFFFFFFFFLL;
      goto LABEL_21;
    }
    goto LABEL_253;
  }
LABEL_20:
  v25 = v24;
LABEL_21:
  v17 = __OFADD__((unsigned int)qword_4FF55A8, v25);
  v26 = (unsigned int)qword_4FF55A8 + v25;
  if ( v17 )
  {
    v26 = 0x7FFFFFFFFFFFFFFFLL;
    if ( !(_DWORD)qword_4FF55A8 )
      v26 = 0x8000000000000000LL;
  }
LABEL_22:
  if ( *(_QWORD *)(a2 + 120) )
  {
    v27 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16 * v19 - 8);
    v28 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 80LL);
    if ( v28 )
      v28 -= 24;
    v29 = sub_FDD860(*(__int64 **)(a2 + 136), v28);
    v30 = sub_FDD860(*(__int64 **)(a2 + 136), v27);
    if ( v29 <= v30 )
      v30 = v29;
    v31 = sub_F02DD0(v30, v29);
    v32 = *(__int64 **)(a2 + 120);
    sub_B2EE70((__int64)&v355, *(_QWORD *)a2, 0);
    if ( !(_BYTE)v357 )
    {
      v33 = *v32;
      v34 = *v32 + 8LL * *((unsigned int *)v32 + 2);
      if ( v33 == v34 )
      {
LABEL_36:
        sub_F02DB0(&v355, 0x2Du, 0x64u);
        if ( (unsigned int)v355 <= v31 )
        {
          sub_F02DB0(&v355, qword_4FF5688, 0x64u);
          if ( v31 < (unsigned int)v355 )
            v31 = (unsigned int)v355;
        }
      }
      else
      {
        while ( 1 )
        {
          v35 = *(_QWORD *)(*(_QWORD *)v33 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v35 == *(_QWORD *)v33 + 48LL )
            goto LABEL_503;
          if ( !v35 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v35 - 24) - 30 > 0xA )
LABEL_503:
            BUG();
          if ( *(_BYTE *)(v35 - 24) == 31
            && (*(_DWORD *)(v35 - 20) & 0x7FFFFFF) != 1
            && (unsigned __int8)sub_BC8700(v35 - 24) )
          {
            break;
          }
          v33 += 8;
          if ( v34 == v33 )
            goto LABEL_36;
        }
      }
    }
    v36 = v31;
  }
  else
  {
    sub_F02DB0(&v355, 0, 1u);
    v36 = (unsigned int)v355;
  }
  v355 = (void *)v26;
  v277 = sub_1098D20((unsigned __int64 *)&v355, v36);
  v283 = qword_4FF5BC8;
  if ( !(_BYTE)qword_4FF5BC8 )
  {
    v37 = *(_DWORD *)(a2 + 112);
    if ( v37 == v289 ? *(_QWORD *)(a2 + 104) < v287 : v289 > v37 )
    {
      sub_1049690(v311, *(_QWORD *)a2);
      v39 = *(_QWORD *)(a2 + 8);
      v299[0] = 0;
      v40 = *(char **)(*(_QWORD *)(v39 + 16) + 24LL);
      v41 = *v40;
      if ( (unsigned __int8)*v40 > 0x1Cu && (v41 == 85 || v41 == 34) )
      {
        v42 = *((_QWORD *)v40 + 6);
        v335 = (unsigned __int8 *)v42;
        if ( v42 )
        {
          sub_B96E90((__int64)&v335, v42, 1);
          v355 = (void *)*((_QWORD *)v40 + 5);
          v356 = v335;
          if ( v335 )
          {
            sub_B96E90((__int64)&v356, (__int64)v335, 1);
            if ( v335 )
              sub_B91220((__int64)&v335, (__int64)v335);
          }
        }
        else
        {
          v244 = (void *)*((_QWORD *)v40 + 5);
          v356 = 0;
          v355 = v244;
        }
        v299[0] = (__int64)v356;
        if ( v356 )
          sub_B976B0((__int64)&v356, v356, (__int64)v299);
        v43 = (__int64)v355;
        v44 = v311[0];
        v45 = sub_B2BE50(v311[0]);
        if ( sub_B6EA50(v45)
          || (v268 = sub_B2BE50(v44),
              v269 = sub_B6F970(v268),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v269 + 48LL))(v269)) )
        {
          sub_B157E0((__int64)&v305, v299);
          sub_B17850((__int64)&v355, (__int64)"partial-inlining", (__int64)"OutlineRegionTooSmall", 21, &v305, v43);
          sub_B16080((__int64)&v323, "Function", 8, *(unsigned __int8 **)a2);
          v46 = sub_26AC990((__int64)&v355, (__int64)&v323);
          sub_B18290(v46, " not partially inlined into callers (Original Size = ", 0x35u);
          sub_B16D50((__int64)v319, "OutlinedRegionOriginalSize", 26, *(_QWORD *)(a2 + 104), *(_QWORD *)(a2 + 112));
          v47 = sub_B826F0(v46, (__int64)v319);
          sub_B18290(v47, ", Size of call sequence to outlined function = ", 0x2Fu);
          sub_B16D50((__int64)v315, "NewSize", 7, v287, (unsigned int)v289);
          v48 = sub_B826F0(v47, (__int64)v315);
          sub_B18290(v48, ")", 1u);
          LODWORD(v336) = *(_DWORD *)(v48 + 8);
          BYTE4(v336) = *(_BYTE *)(v48 + 12);
          v337 = *(_QWORD *)(v48 + 16);
          v52 = _mm_loadu_si128((const __m128i *)(v48 + 24));
          v335 = (unsigned __int8 *)&unk_49D9D40;
          v338 = v52;
          v339 = *(unsigned __int64 **)(v48 + 40);
          v340 = _mm_loadu_si128((const __m128i *)(v48 + 48));
          v53 = _mm_loadu_si128((const __m128i *)(v48 + 64));
          v342 = &v344;
          v343 = 0x400000000LL;
          v341 = v53;
          v54 = *(unsigned int *)(v48 + 88);
          if ( (_DWORD)v54 )
            sub_26ACA40((__int64)&v342, v48 + 80, v49, v50, v51, v54);
          v352 = *(_BYTE *)(v48 + 416);
          v353 = *(_DWORD *)(v48 + 420);
          v354 = *(_QWORD *)(v48 + 424);
          v335 = (unsigned __int8 *)&unk_49D9DE8;
          if ( v317 != v318 )
            j_j___libc_free_0((unsigned __int64)v317);
          if ( (_QWORD *)v315[0] != v316 )
            j_j___libc_free_0(v315[0]);
          if ( v321 != v322 )
            j_j___libc_free_0((unsigned __int64)v321);
          if ( (_QWORD *)v319[0] != v320 )
            j_j___libc_free_0(v319[0]);
          if ( (_QWORD *)v327 != v329 )
            j_j___libc_free_0(v327);
          if ( v323 != &v325 )
            j_j___libc_free_0((unsigned __int64)v323);
          v55 = v362;
          v355 = &unk_49D9D40;
          v56 = 10LL * (unsigned int)v363;
          v57 = &v362[v56];
          if ( v362 != &v362[v56] )
          {
            do
            {
              v57 -= 10;
              v58 = v57[4];
              if ( (unsigned __int64 *)v58 != v57 + 6 )
                j_j___libc_free_0(v58);
              if ( (unsigned __int64 *)*v57 != v57 + 2 )
                j_j___libc_free_0(*v57);
            }
            while ( v55 != v57 );
            v57 = v362;
          }
          if ( v57 != (unsigned __int64 *)&v364 )
            _libc_free((unsigned __int64)v57);
          sub_1049740(v311, (__int64)&v335);
          v335 = (unsigned __int8 *)&unk_49D9D40;
          sub_23FD590((__int64)&v342);
        }
        if ( v299[0] )
          sub_B91220((__int64)v299, v299[0]);
        v59 = v312[0];
        if ( v312[0] )
        {
          sub_FDC110(v312[0]);
          j_j___libc_free_0((unsigned __int64)v59);
        }
        return v283;
      }
LABEL_506:
      BUG();
    }
  }
  v60 = *(_QWORD *)(a2 + 8);
  v61 = *(_QWORD *)(v60 + 16);
  if ( v61 )
  {
    v62 = *(_QWORD *)(v60 + 16);
    v63 = 0;
    do
    {
      v62 = *(_QWORD *)(v62 + 8);
      ++v63;
    }
    while ( v62 );
    if ( v63 > 0xFFFFFFFFFFFFFFFLL )
      goto LABEL_494;
    v64 = sub_22077B0(8 * v63);
    v279 = v64;
    do
    {
      v64 += 8;
      *(_QWORD *)(v64 - 8) = *(_QWORD *)(v61 + 24);
      v61 = *(_QWORD *)(v61 + 8);
    }
    while ( v61 );
    v288 = (__int64 *)v64;
  }
  else
  {
    v279 = 0;
    v288 = 0;
  }
  v301 = 0;
  v302 = 0;
  v303 = 0;
  v65 = *(_QWORD *)a2;
  v304 = 0;
  sub_B2EE70((__int64)&v296, v65, 0);
  v276 = v298;
  if ( !v298 )
  {
    v278 = 0;
    goto LABEL_91;
  }
  v144 = *(_QWORD *)(a2 + 8);
  v145 = *(_QWORD *)(v144 + 16);
  if ( !v145 )
    goto LABEL_285;
  v146 = *(_QWORD *)(v144 + 16);
  v147 = 0;
  do
  {
    v146 = *(_QWORD *)(v146 + 8);
    ++v147;
  }
  while ( v146 );
  if ( v147 > 0xFFFFFFFFFFFFFFFLL )
LABEL_494:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v282 = sub_22077B0(8 * v147);
  v148 = v282;
  do
  {
    v148 += 8LL;
    *(_QWORD *)(v148 - 8) = *(_QWORD *)(v145 + 24);
    v145 = *(_QWORD *)(v145 + 8);
  }
  while ( v145 );
  if ( v148 == v282 )
    goto LABEL_283;
  v291 = (__int64 *)v148;
  v149 = 0;
  v150 = (__int64 *)v282;
  v280 = 0;
  v295 = 0;
  do
  {
    while ( 1 )
    {
      v157 = (_BYTE *)*v150;
      v158 = *(_BYTE *)*v150;
      if ( v158 == 4 )
        goto LABEL_270;
      if ( v158 != 85 && v158 != 34 )
        goto LABEL_506;
      v284 = sub_B491C0(*v150);
      if ( v284 != v149 )
      {
        v159 = *(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 56);
        if ( v159 )
        {
          v149 = v284;
          v295 = (__int64 *)v159(*(_QWORD *)(a1 + 64), v284);
          goto LABEL_277;
        }
        v333 = 0;
        v323 = &v325;
        v324 = 0x100000000LL;
        v326 = &v328;
        v327 = 0x600000000LL;
        v334 = 0;
        v331 = 0;
        v332 = v284;
        HIDWORD(v334) = *(_DWORD *)(v284 + 92);
        sub_B1F440((__int64)&v323);
        sub_D51D90((__int64)&v335, (__int64)&v323);
        v203 = &v365;
        v355 = 0;
        v356 = 0;
        v357 = 0;
        v358 = 0u;
        v359 = 0;
        v360 = 0u;
        v361 = 0u;
        v362 = 0;
        v363 = 0;
        v364 = 1;
        do
        {
          *(_QWORD *)v203 = -4096;
          v203 += 16;
        }
        while ( v203 != (char *)v366 );
        v204 = &v367;
        v366[0] = 0;
        v366[1] = 1;
        do
        {
          *(_QWORD *)v204 = -4096;
          v204 += 24;
          *((_DWORD *)v204 - 4) = 0x7FFFFFFF;
        }
        while ( v204 != v368 );
        v205 = v284;
        sub_FF9360(&v355, v284, (__int64)&v335, 0, 0, 0);
        v206 = (__int64 *)sub_22077B0(8u);
        v295 = v206;
        if ( v206 )
        {
          v205 = v284;
          sub_FE7FB0(v206, (const char *)v284, (__int64)&v355, (__int64)&v335);
        }
        if ( v280 )
        {
          sub_FDC110(v280);
          v205 = 8;
          j_j___libc_free_0((unsigned __int64)v280);
        }
        sub_D77880((__int64)&v355);
        sub_D786F0((__int64)&v335);
        v207 = (unsigned __int64 *)v338.m128i_i64[1];
        if ( (unsigned __int64 *)v338.m128i_i64[1] != v339 )
        {
          v281 = v157;
          v208 = v339;
          v274 = v150;
          do
          {
            v209 = *v207;
            v210 = *(__int64 **)(*v207 + 16);
            if ( *(__int64 **)(*v207 + 8) == v210 )
            {
              *(_BYTE *)(v209 + 152) = 1;
            }
            else
            {
              v211 = *(__int64 **)(*v207 + 8);
              do
              {
                v212 = *v211++;
                sub_D47BB0(v212, v205);
              }
              while ( v210 != v211 );
              *(_BYTE *)(v209 + 152) = 1;
              v213 = *(_QWORD *)(v209 + 8);
              if ( *(_QWORD *)(v209 + 16) != v213 )
                *(_QWORD *)(v209 + 16) = v213;
            }
            v214 = *(_QWORD *)(v209 + 32);
            if ( v214 != *(_QWORD *)(v209 + 40) )
              *(_QWORD *)(v209 + 40) = v214;
            ++*(_QWORD *)(v209 + 56);
            if ( *(_BYTE *)(v209 + 84) )
            {
              *(_QWORD *)v209 = 0;
            }
            else
            {
              v215 = 4 * (*(_DWORD *)(v209 + 76) - *(_DWORD *)(v209 + 80));
              v216 = *(unsigned int *)(v209 + 72);
              if ( v215 < 0x20 )
                v215 = 32;
              if ( v215 < (unsigned int)v216 )
              {
                sub_C8C990(v209 + 56, v205);
              }
              else
              {
                v205 = 0xFFFFFFFFLL;
                memset(*(void **)(v209 + 64), -1, 8 * v216);
              }
              v217 = *(_BYTE *)(v209 + 84);
              *(_QWORD *)v209 = 0;
              if ( !v217 )
                _libc_free(*(_QWORD *)(v209 + 64));
            }
            v218 = *(_QWORD *)(v209 + 32);
            if ( v218 )
            {
              v205 = *(_QWORD *)(v209 + 48) - v218;
              j_j___libc_free_0(v218);
            }
            v219 = *(_QWORD *)(v209 + 8);
            if ( v219 )
            {
              v205 = *(_QWORD *)(v209 + 24) - v219;
              j_j___libc_free_0(v219);
            }
            ++v207;
          }
          while ( v208 != v207 );
          v157 = v281;
          v150 = v274;
          if ( (unsigned __int64 *)v338.m128i_i64[1] != v339 )
            v339 = (unsigned __int64 *)v338.m128i_i64[1];
        }
        v220 = v346;
        v221 = &v346[2 * v347];
        if ( v346 != v221 )
        {
          do
          {
            v222 = v220[1];
            v223 = *v220;
            v220 += 2;
            sub_C7D6A0(v223, v222, 16);
          }
          while ( v221 != v220 );
        }
        v347 = 0;
        if ( (_DWORD)v342 )
        {
          v233 = (__int64 *)v341.m128i_i64[1];
          v348 = 0;
          v234 = v341.m128i_i64[1] + 8LL * (unsigned int)v342;
          v235 = (__int64 *)(v341.m128i_i64[1] + 8);
          v340.m128i_i64[1] = *(_QWORD *)v341.m128i_i64[1];
          for ( v341.m128i_i64[0] = v340.m128i_i64[1] + 4096; (__int64 *)v234 != v235; v233 = (__int64 *)v341.m128i_i64[1] )
          {
            v236 = *v235;
            v237 = 0x40000000000LL;
            v238 = (unsigned int)(v235 - v233) >> 7;
            if ( v238 < 0x1E )
              v237 = 4096LL << v238;
            ++v235;
            sub_C7D6A0(v236, v237, 16);
          }
          LODWORD(v342) = 1;
          sub_C7D6A0(*v233, 4096, 16);
          v239 = v346;
          v224 = &v346[2 * v347];
          if ( v346 == v224 )
            goto LABEL_363;
          do
          {
            v240 = v239[1];
            v241 = *v239;
            v239 += 2;
            sub_C7D6A0(v241, v240, 16);
          }
          while ( v224 != v239 );
        }
        v224 = v346;
LABEL_363:
        if ( v224 != &v348 )
          _libc_free((unsigned __int64)v224);
        if ( (__int64 *)v341.m128i_i64[1] != &v343 )
          _libc_free(v341.m128i_u64[1]);
        if ( v338.m128i_i64[1] )
          j_j___libc_free_0(v338.m128i_u64[1]);
        sub_C7D6A0(v336, 16LL * v338.m128i_u32[0], 8);
        v225 = v326;
        v226 = &v326[(unsigned int)v327];
        if ( v326 != v226 )
        {
          do
          {
            v227 = *--v226;
            if ( v227 )
            {
              v228 = *(_QWORD *)(v227 + 24);
              if ( v228 != v227 + 40 )
                _libc_free(v228);
              j_j___libc_free_0(v227);
            }
          }
          while ( v225 != v226 );
          v226 = v326;
        }
        if ( v226 != &v328 )
          _libc_free((unsigned __int64)v226);
        if ( v323 != &v325 )
          _libc_free((unsigned __int64)v323);
        v149 = v284;
        v280 = v295;
      }
LABEL_277:
      v160 = (void *)sub_FDD2C0(v295, *((_QWORD *)v157 + 5), 0);
      v356 = v161;
      v355 = v160;
      if ( !(_BYTE)v161 )
        break;
      v151 = v355;
      if ( !v304 )
      {
        ++v301;
        goto LABEL_433;
      }
      v152 = 0;
      v153 = 1;
      v154 = (v304 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
      v155 = (_QWORD *)(v302 + 16LL * v154);
      v156 = (_BYTE *)*v155;
      if ( v157 != (_BYTE *)*v155 )
      {
        while ( v156 != (_BYTE *)-4096LL )
        {
          if ( !v152 && v156 == (_BYTE *)-8192LL )
            v152 = v155;
          v154 = (v304 - 1) & (v153 + v154);
          v155 = (_QWORD *)(v302 + 16LL * v154);
          v156 = (_BYTE *)*v155;
          if ( v157 == (_BYTE *)*v155 )
            goto LABEL_269;
          ++v153;
        }
        if ( v152 )
          v155 = v152;
        ++v301;
        v246 = v303 + 1;
        if ( 4 * ((int)v303 + 1) < 3 * v304 )
        {
          if ( v304 - HIDWORD(v303) - v246 > v304 >> 3 )
          {
LABEL_424:
            LODWORD(v303) = v246;
            if ( *v155 != -4096 )
              --HIDWORD(v303);
            *v155 = v157;
            v155[1] = 0;
            goto LABEL_269;
          }
          sub_26AF480((__int64)&v301, v304);
          if ( !v304 )
            goto LABEL_505;
          v249 = (v304 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
          v155 = (_QWORD *)(v302 + 16LL * v249);
          v250 = *v155;
          if ( v157 != (_BYTE *)*v155 )
          {
            v272 = 0;
            v273 = 1;
            while ( v250 != -4096 )
            {
              if ( v250 == -8192 && !v272 )
                v272 = v155;
              v249 = (v304 - 1) & (v273 + v249);
              v155 = (_QWORD *)(v302 + 16LL * v249);
              v250 = *v155;
              if ( v157 == (_BYTE *)*v155 )
                goto LABEL_431;
              ++v273;
            }
            if ( v272 )
            {
              v246 = v303 + 1;
              v155 = v272;
            }
            else
            {
LABEL_477:
              v246 = v303 + 1;
            }
            goto LABEL_424;
          }
LABEL_431:
          v246 = v303 + 1;
          goto LABEL_424;
        }
LABEL_433:
        sub_26AF480((__int64)&v301, 2 * v304);
        if ( !v304 )
        {
LABEL_505:
          LODWORD(v303) = v303 + 1;
          BUG();
        }
        v251 = (v304 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
        v246 = v303 + 1;
        v155 = (_QWORD *)(v302 + 16LL * v251);
        v252 = *v155;
        if ( (_BYTE *)*v155 == v157 )
          goto LABEL_424;
        v253 = 0;
        v254 = 1;
        while ( v252 != -4096 )
        {
          if ( v252 == -8192 && !v253 )
            v253 = v155;
          v251 = (v304 - 1) & (v254 + v251);
          v155 = (_QWORD *)(v302 + 16LL * v251);
          v252 = *v155;
          if ( v157 == (_BYTE *)*v155 )
            goto LABEL_477;
          ++v254;
        }
        if ( v253 )
        {
          v246 = v303 + 1;
          v155 = v253;
          goto LABEL_424;
        }
        goto LABEL_431;
      }
LABEL_269:
      v155[1] = v151;
LABEL_270:
      if ( v291 == ++v150 )
        goto LABEL_281;
    }
    if ( v304 )
    {
      v162 = 0;
      v163 = 1;
      LODWORD(v164) = (v304 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
      v165 = (_QWORD *)(v302 + 16LL * (unsigned int)v164);
      v166 = (_BYTE *)*v165;
      if ( v157 == (_BYTE *)*v165 )
        goto LABEL_280;
      while ( v166 != (_BYTE *)-4096LL )
      {
        if ( v166 == (_BYTE *)-8192LL && !v162 )
          v162 = v165;
        v164 = (v304 - 1) & ((_DWORD)v164 + v163);
        v165 = (_QWORD *)(v302 + 16 * v164);
        v166 = (_BYTE *)*v165;
        if ( v157 == (_BYTE *)*v165 )
          goto LABEL_280;
        ++v163;
      }
      if ( v162 )
        v165 = v162;
      ++v301;
      v245 = v303 + 1;
      if ( 4 * ((int)v303 + 1) < 3 * v304 )
      {
        if ( v304 - HIDWORD(v303) - v245 <= v304 >> 3 )
        {
          sub_26AF480((__int64)&v301, v304);
          if ( !v304 )
          {
LABEL_504:
            LODWORD(v303) = v303 + 1;
            BUG();
          }
          v259 = 0;
          v260 = (v304 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
          v245 = v303 + 1;
          v261 = 1;
          v165 = (_QWORD *)(v302 + 16LL * v260);
          v262 = *v165;
          if ( v157 != (_BYTE *)*v165 )
          {
            while ( v262 != -4096 )
            {
              if ( !v259 && v262 == -8192 )
                v259 = v165;
              v260 = (v304 - 1) & (v261 + v260);
              v165 = (_QWORD *)(v302 + 16LL * v260);
              v262 = *v165;
              if ( v157 == (_BYTE *)*v165 )
                goto LABEL_411;
              ++v261;
            }
            if ( v259 )
              v165 = v259;
          }
        }
        goto LABEL_411;
      }
    }
    else
    {
      ++v301;
    }
    sub_26AF480((__int64)&v301, 2 * v304);
    if ( !v304 )
      goto LABEL_504;
    v255 = (v304 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
    v245 = v303 + 1;
    v165 = (_QWORD *)(v302 + 16LL * v255);
    v256 = *v165;
    if ( v157 != (_BYTE *)*v165 )
    {
      v257 = 0;
      v258 = 1;
      while ( v256 != -4096 )
      {
        if ( v256 == -8192 && !v257 )
          v257 = v165;
        v255 = (v304 - 1) & (v258 + v255);
        v165 = (_QWORD *)(v302 + 16LL * v255);
        v256 = *v165;
        if ( v157 == (_BYTE *)*v165 )
          goto LABEL_411;
        ++v258;
      }
      if ( v257 )
        v165 = v257;
    }
LABEL_411:
    LODWORD(v303) = v245;
    if ( *v165 != -4096 )
      --HIDWORD(v303);
    *v165 = v157;
    v165[1] = 0;
LABEL_280:
    ++v150;
    v165[1] = 0;
  }
  while ( v291 != v150 );
LABEL_281:
  if ( v280 )
  {
    sub_FDC110(v280);
    j_j___libc_free_0((unsigned __int64)v280);
  }
LABEL_283:
  if ( v282 )
    j_j___libc_free_0(v282);
LABEL_285:
  v278 = v296;
LABEL_91:
  if ( v288 == (__int64 *)v279 )
  {
    v283 = 0;
    goto LABEL_149;
  }
  v283 = 0;
  v294 = (__int64 *)v279;
  while ( 2 )
  {
    v66 = (_BYTE *)*v294;
    v67 = *(_BYTE *)*v294;
    if ( v67 == 4 )
      goto LABEL_139;
    if ( v67 != 85 && v67 != 34 )
      goto LABEL_506;
    if ( (_DWORD)qword_4FF5768 != -1 && (int)qword_4FF5768 <= *(_DWORD *)a1 )
      goto LABEL_139;
    v68 = sub_B491C0(*v294);
    sub_1049690(v299, v68);
    v69 = *((_QWORD *)v66 - 4);
    if ( v69 )
    {
      if ( *(_BYTE *)v69 )
      {
        v69 = 0;
      }
      else if ( *(_QWORD *)(v69 + 24) != *((_QWORD *)v66 + 10) )
      {
        v69 = 0;
      }
    }
    v290 = qword_4FF5BC8;
    if ( (_BYTE)qword_4FF5BC8 )
    {
      v290 = sub_30D6380(v69) == 0;
      goto LABEL_104;
    }
    v93 = (unsigned __int8 *)sub_B491C0((__int64)v66);
    v94 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 40))(*(_QWORD *)(a1 + 48), v69);
    v95 = v69;
    v96 = 0;
    v97 = v94;
    v98 = sub_B2BE50(v95);
    v99 = sub_B6F970(v98);
    v100 = (*(__int64 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v99 + 32LL))(v99, "partial-inlining", 16);
    v103 = *(_QWORD *)(a1 + 88);
    if ( v100 )
      v96 = v299;
    sub_30D6B30(&v355, "partial-inlining", v101, v102);
    sub_30DF350(
      (unsigned int)&v305,
      (_DWORD)v66,
      (unsigned int)&v355,
      v97,
      *(_QWORD *)(a1 + 8),
      *(_QWORD *)(a1 + 16),
      *(_QWORD *)(a1 + 72),
      *(_QWORD *)(a1 + 80),
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64),
      v103,
      (__int64)v96);
    if ( v305.m128i_i32[0] == 0x80000000 )
    {
      v186 = v299[0];
      v187 = sub_B2BE50(v299[0]);
      if ( sub_B6EA50(v187)
        || (v265 = sub_B2BE50(v186),
            v266 = sub_B6F970(v265),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v266 + 48LL))(v266)) )
      {
        sub_B178C0((__int64)&v355, (__int64)"partial-inlining", (__int64)"AlwaysInline", 12, (__int64)v66);
        sub_B16080((__int64)&v323, "Callee", 6, *(unsigned __int8 **)a2);
        v188 = sub_26AC990((__int64)&v355, (__int64)&v323);
        sub_B18290(v188, " should always be fully inlined, not partially", 0x2Eu);
        LODWORD(v336) = *(_DWORD *)(v188 + 8);
        BYTE4(v336) = *(_BYTE *)(v188 + 12);
        v337 = *(_QWORD *)(v188 + 16);
        v192 = _mm_loadu_si128((const __m128i *)(v188 + 24));
        v335 = (unsigned __int8 *)&unk_49D9D40;
        v338 = v192;
        v339 = *(unsigned __int64 **)(v188 + 40);
        v340 = _mm_loadu_si128((const __m128i *)(v188 + 48));
        v193 = _mm_loadu_si128((const __m128i *)(v188 + 64));
        v342 = &v344;
        v343 = 0x400000000LL;
        v341 = v193;
        v194 = *(unsigned int *)(v188 + 88);
        if ( (_DWORD)v194 )
          sub_26ACA40((__int64)&v342, v188 + 80, v189, v190, v194, v191);
        v352 = *(_BYTE *)(v188 + 416);
        v353 = *(_DWORD *)(v188 + 420);
        v354 = *(_QWORD *)(v188 + 424);
        v335 = (unsigned __int8 *)&unk_49D9DE8;
        sub_2240A30(&v327);
        sub_2240A30((unsigned __int64 *)&v323);
        v355 = &unk_49D9D40;
        sub_23FD590((__int64)&v362);
        sub_1049740(v299, (__int64)&v335);
        v335 = (unsigned __int8 *)&unk_49D9D40;
        sub_23FD590((__int64)&v342);
      }
      goto LABEL_187;
    }
    if ( v305.m128i_i32[0] == 0x7FFFFFFF )
    {
      v195 = v299[0];
      v196 = sub_B2BE50(v299[0]);
      if ( sub_B6EA50(v196)
        || (v263 = sub_B2BE50(v195),
            v264 = sub_B6F970(v263),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v264 + 48LL))(v264)) )
      {
        sub_B176B0((__int64)&v355, (__int64)"partial-inlining", (__int64)"NeverInline", 11, (__int64)v66);
        sub_B16080((__int64)&v323, "Callee", 6, *(unsigned __int8 **)a2);
        v335 = (unsigned __int8 *)&v337;
        sub_26AB380((__int64 *)&v335, v323, (__int64)&v323[(unsigned __int64)v324 / 8]);
        v338.m128i_i64[1] = (__int64)&v340;
        sub_26AB380(&v338.m128i_i64[1], (_BYTE *)v327, v327 + v328);
        v341 = _mm_loadu_si128(v330);
        sub_B180C0((__int64)&v355, (unsigned __int64)&v335);
        sub_2240A30(&v338.m128i_u64[1]);
        sub_2240A30((unsigned __int64 *)&v335);
        sub_B18290((__int64)&v355, " not partially inlined into ", 0x1Cu);
        sub_B16080((__int64)v319, "Caller", 6, v93);
        v275 = sub_2445430((__int64)&v355, (__int64)v319);
        sub_B18290(v275, " because it should never be inlined (cost=never)", 0x30u);
        v200 = v275;
        LODWORD(v336) = *(_DWORD *)(v275 + 8);
        BYTE4(v336) = *(_BYTE *)(v275 + 12);
        v337 = *(_QWORD *)(v275 + 16);
        v201 = _mm_loadu_si128((const __m128i *)(v275 + 24));
        v335 = (unsigned __int8 *)&unk_49D9D40;
        v338 = v201;
        v339 = *(unsigned __int64 **)(v275 + 40);
        v340 = _mm_loadu_si128((const __m128i *)(v275 + 48));
        v202 = _mm_loadu_si128((const __m128i *)(v275 + 64));
        v342 = &v344;
        v343 = 0x400000000LL;
        v341 = v202;
        if ( *(_DWORD *)(v275 + 88) )
        {
          sub_26ACA40((__int64)&v342, v275 + 80, (__int64)&v344, v197, v198, v199);
          v200 = v275;
        }
        v352 = *(_BYTE *)(v200 + 416);
        v353 = *(_DWORD *)(v200 + 420);
        v354 = *(_QWORD *)(v200 + 424);
        v335 = (unsigned __int8 *)&unk_49D9DB0;
        sub_2240A30((unsigned __int64 *)&v321);
        sub_2240A30(v319);
        sub_2240A30(&v327);
        sub_2240A30((unsigned __int64 *)&v323);
        v355 = &unk_49D9D40;
        sub_23FD590((__int64)&v362);
        sub_1049740(v299, (__int64)&v335);
        v335 = (unsigned __int8 *)&unk_49D9D40;
        sub_23FD590((__int64)&v342);
      }
      goto LABEL_187;
    }
    if ( v305.m128i_i32[0] >= v305.m128i_i32[1] )
    {
      v129 = v299[0];
      v130 = sub_B2BE50(v299[0]);
      if ( !sub_B6EA50(v130) )
      {
        v229 = sub_B2BE50(v129);
        v230 = sub_B6F970(v229);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v230 + 48LL))(v230) )
          goto LABEL_187;
      }
      sub_B178C0((__int64)&v355, (__int64)"partial-inlining", (__int64)"TooCostly", 9, (__int64)v66);
      sub_B16080((__int64)&v323, "Callee", 6, *(unsigned __int8 **)a2);
      v131 = sub_26AC990((__int64)&v355, (__int64)&v323);
      sub_B18290(v131, " not partially inlined into ", 0x1Cu);
      sub_B16080((__int64)v319, "Caller", 6, v93);
      v132 = sub_B826F0(v131, (__int64)v319);
      sub_B18290(v132, " because too costly to inline (cost=", 0x24u);
      sub_B16530(v315, "Cost", 4, v305.m128i_i32[0]);
      v133 = sub_B826F0(v132, (__int64)v315);
      sub_B18290(v133, ", threshold=", 0xCu);
      sub_B16530(v311, "Threshold", 9, v305.m128i_i32[1]);
      v134 = sub_B826F0(v133, (__int64)v311);
      sub_B18290(v134, ")", 1u);
      LODWORD(v336) = *(_DWORD *)(v134 + 8);
      BYTE4(v336) = *(_BYTE *)(v134 + 12);
      v337 = *(_QWORD *)(v134 + 16);
      v139 = _mm_loadu_si128((const __m128i *)(v134 + 24));
      v335 = (unsigned __int8 *)&unk_49D9D40;
      v338 = v139;
      v339 = *(unsigned __int64 **)(v134 + 40);
      v340 = _mm_loadu_si128((const __m128i *)(v134 + 48));
      v140 = _mm_loadu_si128((const __m128i *)(v134 + 64));
      v342 = &v344;
      v343 = 0x400000000LL;
      v341 = v140;
      if ( *(_DWORD *)(v134 + 88) )
        sub_26ACA40((__int64)&v342, v134 + 80, v135, v136, v137, v138);
      v352 = *(_BYTE *)(v134 + 416);
      v353 = *(_DWORD *)(v134 + 420);
      v354 = *(_QWORD *)(v134 + 424);
      v335 = (unsigned __int8 *)&unk_49D9DE8;
      if ( v313 != v314 )
        j_j___libc_free_0((unsigned __int64)v313);
      if ( (__int64 **)v311[0] != v312 )
        j_j___libc_free_0(v311[0]);
      if ( v317 != v318 )
        j_j___libc_free_0((unsigned __int64)v317);
      if ( (_QWORD *)v315[0] != v316 )
        j_j___libc_free_0(v315[0]);
      if ( v321 != v322 )
        j_j___libc_free_0((unsigned __int64)v321);
      if ( (_QWORD *)v319[0] != v320 )
        j_j___libc_free_0(v319[0]);
      if ( (_QWORD *)v327 != v329 )
        j_j___libc_free_0(v327);
      if ( v323 != &v325 )
        j_j___libc_free_0((unsigned __int64)v323);
      v141 = (__int64 *)v362;
      v355 = &unk_49D9D40;
      v142 = (__int64 *)&v362[10 * (unsigned int)v363];
      if ( v362 == (unsigned __int64 *)v142 )
      {
LABEL_246:
        if ( v142 != &v364 )
          _libc_free((unsigned __int64)v142);
        sub_1049740(v299, (__int64)&v335);
        v335 = (unsigned __int8 *)&unk_49D9D40;
        sub_23FD590((__int64)&v342);
        goto LABEL_187;
      }
      do
      {
        v142 -= 10;
        v143 = v142[4];
        if ( (__int64 *)v143 != v142 + 6 )
          j_j___libc_free_0(v143);
        if ( (__int64 *)*v142 != v142 + 2 )
          j_j___libc_free_0(*v142);
      }
      while ( v141 != v142 );
LABEL_245:
      v142 = (__int64 *)v362;
      goto LABEL_246;
    }
    v104 = sub_B2BEC0((__int64)v93);
    v105 = sub_30D4FE0(v97, v66, v104);
    if ( v105 >= v277 )
    {
      v106 = v299[0];
      v107 = sub_B2BE50(v299[0]);
      if ( sub_B6EA50(v107)
        || (v242 = sub_B2BE50(v106),
            v243 = sub_B6F970(v242),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v243 + 48LL))(v243)) )
      {
        sub_B178C0((__int64)&v355, (__int64)"partial-inlining", (__int64)"CanBePartiallyInlined", 21, (__int64)v66);
        sub_B16080((__int64)&v323, "Callee", 6, *(unsigned __int8 **)a2);
        v108 = sub_26AC990((__int64)&v355, (__int64)&v323);
        sub_B18290(v108, " can be partially inlined into ", 0x1Fu);
        sub_B16080((__int64)v319, "Caller", 6, v93);
        v109 = sub_B826F0(v108, (__int64)v319);
        sub_B18290(v109, " with cost=", 0xBu);
        sub_B16530(v315, "Cost", 4, v305.m128i_i32[0]);
        v110 = sub_B826F0(v109, (__int64)v315);
        sub_B18290(v110, " (threshold=", 0xCu);
        sub_B16530(v311, "Threshold", 9, v305.m128i_i32[1]);
        v111 = sub_B826F0(v110, (__int64)v311);
        sub_B18290(v111, ")", 1u);
        LODWORD(v336) = *(_DWORD *)(v111 + 8);
        BYTE4(v336) = *(_BYTE *)(v111 + 12);
        v337 = *(_QWORD *)(v111 + 16);
        v115 = _mm_loadu_si128((const __m128i *)(v111 + 24));
        v335 = (unsigned __int8 *)&unk_49D9D40;
        v338 = v115;
        v339 = *(unsigned __int64 **)(v111 + 40);
        v340 = _mm_loadu_si128((const __m128i *)(v111 + 48));
        v116 = _mm_loadu_si128((const __m128i *)(v111 + 64));
        v342 = &v344;
        v343 = 0x400000000LL;
        v341 = v116;
        v117 = *(unsigned int *)(v111 + 88);
        if ( (_DWORD)v117 )
          sub_26ACA40((__int64)&v342, v111 + 80, v117, v112, v113, v114);
        v352 = *(_BYTE *)(v111 + 416);
        v353 = *(_DWORD *)(v111 + 420);
        v354 = *(_QWORD *)(v111 + 424);
        v335 = (unsigned __int8 *)&unk_49D9DE8;
        if ( v313 != v314 )
          j_j___libc_free_0((unsigned __int64)v313);
        if ( (__int64 **)v311[0] != v312 )
          j_j___libc_free_0(v311[0]);
        if ( v317 != v318 )
          j_j___libc_free_0((unsigned __int64)v317);
        if ( (_QWORD *)v315[0] != v316 )
          j_j___libc_free_0(v315[0]);
        if ( v321 != v322 )
          j_j___libc_free_0((unsigned __int64)v321);
        if ( (_QWORD *)v319[0] != v320 )
          j_j___libc_free_0(v319[0]);
        if ( (_QWORD *)v327 != v329 )
          j_j___libc_free_0(v327);
        if ( v323 != &v325 )
          j_j___libc_free_0((unsigned __int64)v323);
        v118 = v362;
        v355 = &unk_49D9D40;
        v119 = &v362[10 * (unsigned int)v363];
        if ( v362 != v119 )
        {
          do
          {
            v119 -= 10;
            v120 = v119[4];
            if ( (unsigned __int64 *)v120 != v119 + 6 )
              j_j___libc_free_0(v120);
            if ( (unsigned __int64 *)*v119 != v119 + 2 )
              j_j___libc_free_0(*v119);
          }
          while ( v118 != v119 );
          v119 = v362;
        }
        if ( v119 != (unsigned __int64 *)&v364 )
          _libc_free((unsigned __int64)v119);
        sub_1049740(v299, (__int64)&v335);
        v335 = (unsigned __int8 *)&unk_49D9D40;
        sub_23FD590((__int64)&v342);
      }
      v290 = 1;
      goto LABEL_187;
    }
    v167 = v299[0];
    v168 = sub_B2BE50(v299[0]);
    if ( sub_B6EA50(v168)
      || (v247 = sub_B2BE50(v167),
          v248 = sub_B6F970(v247),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v248 + 48LL))(v248)) )
    {
      sub_B178C0((__int64)&v355, (__int64)"partial-inlining", (__int64)"OutliningCallcostTooHigh", 24, (__int64)v66);
      sub_B16080((__int64)&v323, "Callee", 6, *(unsigned __int8 **)a2);
      v169 = sub_26AC990((__int64)&v355, (__int64)&v323);
      sub_B18290(v169, " not partially inlined into ", 0x1Cu);
      sub_B16080((__int64)v319, "Caller", 6, v93);
      v170 = sub_B826F0(v169, (__int64)v319);
      sub_B18290(v170, " runtime overhead (overhead=", 0x1Cu);
      sub_B169E0(v315, "Overhead", 8, v277);
      v171 = sub_B826F0(v170, (__int64)v315);
      sub_B18290(v171, ", savings=", 0xAu);
      sub_B169E0(v311, "Savings", 7, v105);
      v172 = sub_B826F0(v171, (__int64)v311);
      sub_B18290(v172, ")", 1u);
      sub_B18290(v172, " of making the outlined call is too high", 0x28u);
      LODWORD(v336) = *(_DWORD *)(v172 + 8);
      BYTE4(v336) = *(_BYTE *)(v172 + 12);
      v337 = *(_QWORD *)(v172 + 16);
      v176 = _mm_loadu_si128((const __m128i *)(v172 + 24));
      v335 = (unsigned __int8 *)&unk_49D9D40;
      v338 = v176;
      v339 = *(unsigned __int64 **)(v172 + 40);
      v340 = _mm_loadu_si128((const __m128i *)(v172 + 48));
      v177 = _mm_loadu_si128((const __m128i *)(v172 + 64));
      v342 = &v344;
      v343 = 0x400000000LL;
      v341 = v177;
      v178 = *(unsigned int *)(v172 + 88);
      if ( (_DWORD)v178 )
        sub_26ACA40((__int64)&v342, v172 + 80, v173, v178, v174, v175);
      v352 = *(_BYTE *)(v172 + 416);
      v353 = *(_DWORD *)(v172 + 420);
      v354 = *(_QWORD *)(v172 + 424);
      v335 = (unsigned __int8 *)&unk_49D9DE8;
      if ( v313 != v314 )
        j_j___libc_free_0((unsigned __int64)v313);
      if ( (__int64 **)v311[0] != v312 )
        j_j___libc_free_0(v311[0]);
      if ( v317 != v318 )
        j_j___libc_free_0((unsigned __int64)v317);
      if ( (_QWORD *)v315[0] != v316 )
        j_j___libc_free_0(v315[0]);
      if ( v321 != v322 )
        j_j___libc_free_0((unsigned __int64)v321);
      if ( (_QWORD *)v319[0] != v320 )
        j_j___libc_free_0(v319[0]);
      if ( (_QWORD *)v327 != v329 )
        j_j___libc_free_0(v327);
      if ( v323 != &v325 )
        j_j___libc_free_0((unsigned __int64)v323);
      v179 = (__int64 *)v362;
      v355 = &unk_49D9D40;
      v180 = 10LL * (unsigned int)v363;
      v142 = (__int64 *)&v362[v180];
      if ( v362 == &v362[v180] )
        goto LABEL_246;
      do
      {
        v142 -= 10;
        v181 = v142[4];
        if ( (__int64 *)v181 != v142 + 6 )
          j_j___libc_free_0(v181);
        if ( (__int64 *)*v142 != v142 + 2 )
          j_j___libc_free_0(*v142);
      }
      while ( v179 != v142 );
      goto LABEL_245;
    }
LABEL_187:
    if ( v310 )
    {
      v310 = 0;
      if ( v309 > 0x40 && v308 )
        j_j___libc_free_0_0(v308);
      if ( v307 > 0x40 && v306 )
        j_j___libc_free_0_0(v306);
    }
LABEL_104:
    if ( v290 )
    {
      sub_B174A0((__int64)&v355, (__int64)"partial-inlining", (__int64)"PartiallyInlined", 16, (__int64)v66);
      sub_B16080((__int64)&v323, "Callee", 6, *(unsigned __int8 **)a2);
      v70 = sub_23FD640((__int64)&v355, (__int64)&v323);
      sub_B18290(v70, " partially inlined into ", 0x18u);
      v71 = (unsigned __int8 *)sub_B491C0((__int64)v66);
      sub_B16080((__int64)&v335, "Caller", 6, v71);
      sub_23FD640(v70, (__int64)&v335);
      if ( (__m128i *)v338.m128i_i64[1] != &v340 )
        j_j___libc_free_0(v338.m128i_u64[1]);
      if ( v335 != (unsigned __int8 *)&v337 )
        j_j___libc_free_0((unsigned __int64)v335);
      if ( (_QWORD *)v327 != v329 )
        j_j___libc_free_0(v327);
      if ( v323 != &v325 )
        j_j___libc_free_0((unsigned __int64)v323);
      v338 = 0u;
      v339 = &v340.m128i_u64[1];
      v72 = *(_QWORD *)(a1 + 8);
      v73 = *(_QWORD *)(a1 + 88);
      v74 = *(_QWORD *)(a1 + 16);
      v343 = (__int64)v345;
      v335 = (unsigned __int8 *)v72;
      v337 = v73;
      v75 = *(_QWORD *)(a2 + 120);
      v340.m128i_i64[0] = 0x400000000LL;
      v336 = v74;
      v344 = 0x800000000LL;
      v349 = v351;
      v350 = 0x800000000LL;
      v351[64] = 1;
      if ( v75 )
        v75 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * *(unsigned int *)(a2 + 24) - 16);
      if ( sub_29F2700(v66, &v335, 0, 0, 1, v75) )
      {
        if ( v349 != v351 )
          _libc_free((unsigned __int64)v349);
        v76 = (_QWORD *)v343;
        v77 = (_QWORD *)(v343 + 24LL * (unsigned int)v344);
        if ( (_QWORD *)v343 != v77 )
        {
          do
          {
            v78 = *(v77 - 1);
            v77 -= 3;
            if ( v78 != 0 && v78 != -4096 && v78 != -8192 )
              sub_BD60C0(v77);
          }
          while ( v76 != v77 );
          v77 = (_QWORD *)v343;
        }
        if ( v77 != (_QWORD *)v345 )
          _libc_free((unsigned __int64)v77);
        if ( v339 != &v340.m128i_u64[1] )
          _libc_free((unsigned __int64)v339);
        v79 = v362;
        v355 = &unk_49D9D40;
        v80 = &v362[10 * (unsigned int)v363];
        if ( v362 != v80 )
        {
          do
          {
            v80 -= 10;
            v81 = v80[4];
            if ( (unsigned __int64 *)v81 != v80 + 6 )
              j_j___libc_free_0(v81);
            if ( (unsigned __int64 *)*v80 != v80 + 2 )
              j_j___libc_free_0(*v80);
          }
          while ( v79 != v80 );
          v80 = v362;
        }
        if ( v80 != (unsigned __int64 *)&v364 )
          _libc_free((unsigned __int64)v80);
        goto LABEL_137;
      }
      sub_1049740(v299, (__int64)&v355);
      if ( v278 && v304 )
      {
        v182 = (v304 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v183 = (_QWORD *)(v302 + 16LL * v182);
        v184 = (_BYTE *)*v183;
        if ( v66 == (_BYTE *)*v183 )
        {
LABEL_314:
          if ( v183 != (_QWORD *)(v302 + 16LL * v304) )
          {
            v185 = v278;
            if ( v183[1] <= v278 )
              v185 = v183[1];
            v278 -= v185;
          }
        }
        else
        {
          v267 = 1;
          while ( v184 != (_BYTE *)-4096LL )
          {
            v182 = (v304 - 1) & (v267 + v182);
            v183 = (_QWORD *)(v302 + 16LL * v182);
            v184 = (_BYTE *)*v183;
            if ( v66 == (_BYTE *)*v183 )
              goto LABEL_314;
            ++v267;
          }
        }
      }
      v121 = v349;
      ++*(_DWORD *)a1;
      if ( v121 != v351 )
        _libc_free((unsigned __int64)v121);
      v122 = (_QWORD *)v343;
      v123 = (_QWORD *)(v343 + 24LL * (unsigned int)v344);
      if ( (_QWORD *)v343 != v123 )
      {
        do
        {
          v124 = *(v123 - 1);
          v123 -= 3;
          if ( v124 != -4096 && v124 != 0 && v124 != -8192 )
            sub_BD60C0(v123);
        }
        while ( v122 != v123 );
        v123 = (_QWORD *)v343;
      }
      if ( v123 != (_QWORD *)v345 )
        _libc_free((unsigned __int64)v123);
      if ( v339 != &v340.m128i_u64[1] )
        _libc_free((unsigned __int64)v339);
      v125 = v362;
      v355 = &unk_49D9D40;
      v126 = &v362[10 * (unsigned int)v363];
      if ( v362 != v126 )
      {
        do
        {
          v126 -= 10;
          v127 = v126[4];
          if ( (unsigned __int64 *)v127 != v126 + 6 )
            j_j___libc_free_0(v127);
          if ( (unsigned __int64 *)*v126 != v126 + 2 )
            j_j___libc_free_0(*v126);
        }
        while ( v125 != v126 );
        v126 = v362;
      }
      if ( v126 != (unsigned __int64 *)&v364 )
        _libc_free((unsigned __int64)v126);
      v128 = v300;
      if ( v300 )
      {
        sub_FDC110(v300);
        j_j___libc_free_0((unsigned __int64)v128);
      }
      v283 = v290;
    }
    else
    {
LABEL_137:
      v82 = v300;
      if ( v300 )
      {
        sub_FDC110(v300);
        j_j___libc_free_0((unsigned __int64)v82);
      }
    }
LABEL_139:
    if ( v288 != ++v294 )
      continue;
    break;
  }
  if ( v283 )
  {
    *(_BYTE *)(a2 + 96) = 1;
    if ( v276 )
      sub_B2F4C0(*(_QWORD *)a2, v278, v297, 0);
    sub_1049690((__int64 *)&v323, *(_QWORD *)a2);
    v83 = v323;
    v84 = sub_B2BE50((__int64)v323);
    if ( sub_B6EA50(v84)
      || (v231 = sub_B2BE50((__int64)v83),
          v232 = sub_B6F970(v231),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v232 + 48LL))(v232)) )
    {
      sub_B17560((__int64)&v355, (__int64)"partial-inlining", (__int64)"PartiallyInlined", 16, *(_QWORD *)a2);
      sub_B18290((__int64)&v355, "Partially inlined into at least one caller", 0x2Au);
      v89 = _mm_loadu_si128(&v358);
      v90 = _mm_loadu_si128(&v360);
      LODWORD(v336) = (_DWORD)v356;
      v91 = _mm_loadu_si128(&v361);
      v338 = v89;
      BYTE4(v336) = BYTE4(v356);
      v340 = v90;
      v337 = v357;
      v335 = (unsigned __int8 *)&unk_49D9D40;
      v341 = v91;
      v339 = v359;
      v342 = &v344;
      v343 = 0x400000000LL;
      if ( (_DWORD)v363 )
        sub_26ACA40((__int64)&v342, (__int64)&v362, v85, v86, v87, v88);
      v355 = &unk_49D9D40;
      v352 = v368[136];
      v353 = v369;
      v354 = v370;
      v335 = (unsigned __int8 *)&unk_49D9D78;
      sub_23FD590((__int64)&v362);
      sub_1049740((__int64 *)&v323, (__int64)&v335);
      v335 = (unsigned __int8 *)&unk_49D9D40;
      sub_23FD590((__int64)&v342);
    }
    v92 = v325;
    if ( v325 )
    {
      sub_FDC110(v325);
      j_j___libc_free_0((unsigned __int64)v92);
    }
  }
LABEL_149:
  sub_C7D6A0(v302, 16LL * v304, 8);
  if ( v279 )
    j_j___libc_free_0(v279);
  return v283;
}
