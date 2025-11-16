// Function: sub_3521FF0
// Address: 0x3521ff0
//
__int64 __fastcall sub_3521FF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 *v3; // rdx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 (*v17)(void); // rdx
  __int64 v18; // rax
  __int64 (*v19)(void); // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rax
  _QWORD *v30; // r13
  _QWORD *v31; // r12
  unsigned __int64 v32; // rsi
  _QWORD *v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  _QWORD *v41; // r13
  _QWORD *v42; // r12
  unsigned __int64 v43; // rsi
  _QWORD *v44; // rax
  _QWORD *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  _QWORD *v49; // rdi
  __int64 v50; // rcx
  __int64 v51; // rdx
  char v52; // al
  __int64 v53; // rcx
  char v54; // r13
  __int64 v55; // rax
  unsigned int v56; // edx
  unsigned int v57; // edx
  char v58; // r12
  char v59; // r14
  __int64 *v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  int v63; // eax
  __int64 v64; // rax
  unsigned int v65; // edx
  __int64 v66; // rax
  const char *v67; // rbx
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 *v70; // r14
  __int64 v71; // rdi
  __int64 v72; // r13
  __int64 (*v73)(); // rax
  __int64 v74; // rdi
  __int64 (*v75)(); // rax
  size_t v76; // r12
  __int64 *v77; // r12
  const char *v78; // rax
  __int64 v79; // rdx
  _QWORD *v81; // r12
  _QWORD *v82; // r13
  unsigned __int64 v83; // rdx
  _QWORD *v84; // rax
  _QWORD *v85; // rdi
  __int64 v86; // rsi
  __int64 v87; // rcx
  __int64 v88; // rax
  _QWORD *v89; // rdi
  __int64 v90; // r13
  __int64 (__fastcall *v91)(__int64, int); // r12
  __int64 v92; // rsi
  int v93; // eax
  _QWORD *v94; // r12
  _QWORD *v95; // r13
  unsigned __int64 v96; // rsi
  _QWORD *v97; // rax
  _QWORD *v98; // rdi
  __int64 v99; // rcx
  __int64 v100; // rdx
  __int64 v101; // rax
  _QWORD *v102; // rdi
  __int64 v103; // rcx
  __int64 v104; // rdx
  _QWORD *v105; // r12
  _QWORD *v106; // r13
  unsigned __int64 v107; // rsi
  _QWORD *v108; // rax
  _QWORD *v109; // rdi
  __int64 v110; // rcx
  __int64 v111; // rdx
  __int64 v112; // rax
  _QWORD *v113; // rdi
  __int64 v114; // rcx
  __int64 v115; // rdx
  _QWORD *v116; // r12
  _QWORD *v117; // r13
  unsigned __int64 v118; // rdx
  _QWORD *v119; // rax
  _QWORD *v120; // rcx
  __int64 v121; // rsi
  __int64 v122; // rax
  __int64 v123; // rdx
  int v124; // ecx
  unsigned __int64 v125; // rdx
  __int64 v126; // rax
  unsigned int v127; // edx
  unsigned __int64 v128; // rsi
  __int64 v129; // rax
  const char *v130; // rbx
  const char *v131; // rax
  __int64 v132; // r12
  const char *v133; // r15
  __int64 v134; // r9
  __int64 v135; // r8
  __int64 *v136; // rax
  unsigned int v137; // ecx
  __int64 *v138; // rdx
  __int64 v139; // rdi
  __int64 *v140; // rax
  _BYTE *v141; // rsi
  __int64 v142; // r14
  __int64 v143; // rcx
  int v144; // edx
  __int64 v145; // r10
  __int64 v146; // rax
  __int64 v147; // rdx
  __int64 *v148; // rdx
  __int64 v149; // r12
  __int64 v150; // r13
  __int64 i; // r12
  const char *v152; // rax
  unsigned __int64 v153; // rax
  __int64 v154; // rdx
  __int64 v155; // r13
  __int64 v156; // rax
  __int64 v157; // rdx
  unsigned int v158; // ebx
  __int64 *v159; // rax
  __int64 *j; // rdx
  __int64 v161; // rax
  __int64 v162; // rax
  unsigned int v163; // ebx
  const __m128i **v164; // rax
  const __m128i **k; // rdx
  __int64 v166; // rax
  __int64 v167; // rbx
  __int64 v168; // rax
  int v169; // r11d
  __int64 *v170; // rdx
  unsigned int v171; // ecx
  __int64 *v172; // rax
  __int64 v173; // rdi
  __int64 v174; // rax
  __int64 v175; // rsi
  __int64 v176; // rax
  __int16 v177; // dx
  __int64 v178; // rbx
  __int16 v179; // dx
  unsigned int v180; // esi
  int v181; // r11d
  __int64 *v182; // rdx
  unsigned int v183; // ecx
  __int64 *v184; // rax
  __int64 v185; // rdi
  __int64 v186; // rax
  __int64 v187; // rdi
  __int64 (*v188)(); // rax
  __int64 **v189; // rdx
  __int64 v190; // rcx
  double v191; // rbx
  __int64 v192; // rax
  __int64 v193; // rcx
  __int64 v194; // rax
  unsigned int v195; // edx
  unsigned __int64 v196; // rsi
  __int64 *v197; // r14
  __int64 *v198; // r13
  _BYTE *v199; // rsi
  __int64 *v200; // rdx
  double v201; // xmm0_8
  __int64 v202; // rdx
  const char *v203; // rax
  unsigned __int64 v204; // r14
  const char *v205; // rax
  __int64 v206; // r9
  __int64 v207; // r8
  const char *v208; // r15
  const char *v209; // r12
  __int64 v210; // rax
  __int64 v211; // rdi
  unsigned int v212; // ecx
  int v213; // eax
  __int64 v214; // r8
  int v215; // edi
  __int64 *v216; // rsi
  __int64 *v217; // r12
  unsigned int v218; // r8d
  int v219; // r11d
  __int64 *v220; // rdx
  __int64 v221; // rdi
  unsigned int v222; // ecx
  __int64 *v223; // rax
  __int64 v224; // r9
  __int64 v225; // r9
  __int64 *v226; // rax
  __int64 v227; // r10
  _BYTE *v228; // rax
  __int64 v229; // rax
  unsigned __int64 v230; // r8
  const __m128i *v231; // rdx
  const __m128i *v232; // rbx
  __m128i *v233; // rax
  __int64 v234; // rdx
  __int64 v235; // rbx
  unsigned int v236; // eax
  unsigned __int64 v237; // rax
  unsigned int v238; // esi
  _BYTE *v239; // r15
  unsigned int v240; // r8d
  int v241; // eax
  __int64 v242; // rsi
  unsigned int v243; // r8d
  int v244; // eax
  __int64 *v245; // rdx
  __int64 v246; // rsi
  signed __int64 v247; // rbx
  __int64 v248; // rax
  __int64 v249; // r8
  __int64 v250; // r9
  __int64 v251; // rbx
  __int64 v252; // r12
  __int64 v253; // rax
  unsigned __int64 v254; // rbx
  __int64 v255; // rax
  __int64 *v256; // rbx
  __int64 v257; // r15
  __int64 v258; // r8
  unsigned int v259; // edi
  int v260; // r14d
  __int64 *v261; // rdx
  __int64 v262; // rcx
  unsigned int v263; // r9d
  __int64 *v264; // rax
  __int64 v265; // r10
  int v266; // r11d
  _QWORD *v267; // rdx
  unsigned int v268; // r9d
  _QWORD *v269; // rax
  __int64 v270; // r10
  _BYTE *v271; // rax
  __int64 v272; // rax
  const __m128i *v273; // r12
  unsigned __int64 v274; // r9
  const __m128i *v275; // rdx
  __m128i *v276; // rax
  __int64 v277; // rdx
  unsigned int v278; // esi
  __int64 v279; // r12
  unsigned int v280; // r9d
  int v281; // eax
  __int64 v282; // rdi
  unsigned int v283; // ecx
  int v284; // eax
  __int64 v285; // r9
  signed __int64 v286; // r12
  __int64 *v287; // rcx
  __int64 v288; // r8
  __int64 v289; // rsi
  __int64 *v290; // rcx
  int v291; // r9d
  unsigned int v292; // r8d
  __int64 v293; // rsi
  _QWORD *v294; // rcx
  int v295; // esi
  unsigned int v296; // r14d
  __int64 v297; // rdi
  __int64 *v298; // rcx
  int v299; // esi
  unsigned int v300; // r9d
  __int64 v301; // rdi
  __int64 *v302; // rcx
  __int64 v303; // r13
  int v304; // esi
  int v305; // eax
  const void *v306; // r13
  const char *v307; // rdi
  __int64 v308; // rdx
  __int64 *v309; // rcx
  unsigned int v310; // r12d
  int v311; // esi
  __int64 v312; // rdi
  unsigned int v313; // ecx
  __int64 v314; // r8
  int v315; // edi
  __int64 *v316; // rsi
  __int64 *v317; // rcx
  unsigned int v318; // r12d
  int v319; // esi
  __int64 v320; // rdi
  __int64 v321; // rax
  _QWORD *v322; // rdx
  __int64 v323; // rbx
  __int64 v324; // rax
  __int64 v325; // rdi
  const char *v326; // rbx
  const char *v327; // r13
  __int64 v328; // rsi
  __int64 *v329; // rax
  int v330; // r9d
  int v331; // esi
  int v332; // edi
  _QWORD *v333; // rsi
  int v334; // edi
  __int64 *v335; // rsi
  int v336; // [rsp+4h] [rbp-29Ch]
  __int64 v337; // [rsp+18h] [rbp-288h]
  __int64 v338; // [rsp+18h] [rbp-288h]
  __int64 v339; // [rsp+18h] [rbp-288h]
  __int64 v340; // [rsp+18h] [rbp-288h]
  __int64 v341; // [rsp+18h] [rbp-288h]
  __int64 v342; // [rsp+38h] [rbp-268h]
  __int64 v343; // [rsp+40h] [rbp-260h]
  unsigned int v344; // [rsp+40h] [rbp-260h]
  unsigned int v345; // [rsp+50h] [rbp-250h]
  __int64 *v346; // [rsp+50h] [rbp-250h]
  __int64 v347; // [rsp+58h] [rbp-248h]
  unsigned int v348; // [rsp+58h] [rbp-248h]
  __int64 v350; // [rsp+68h] [rbp-238h]
  __int64 v351; // [rsp+68h] [rbp-238h]
  __int64 *v352; // [rsp+68h] [rbp-238h]
  __int64 *v353; // [rsp+68h] [rbp-238h]
  const char *v354; // [rsp+70h] [rbp-230h]
  char v355; // [rsp+78h] [rbp-228h]
  __int64 v356; // [rsp+78h] [rbp-228h]
  __int64 v357; // [rsp+80h] [rbp-220h] BYREF
  __int64 v358; // [rsp+88h] [rbp-218h] BYREF
  __int64 *v359; // [rsp+90h] [rbp-210h] BYREF
  __int64 v360; // [rsp+98h] [rbp-208h]
  const __m128i **v361; // [rsp+A0h] [rbp-200h] BYREF
  __int64 v362; // [rsp+A8h] [rbp-1F8h]
  __m128i *v363; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v364; // [rsp+B8h] [rbp-1E8h]
  __int64 *v365; // [rsp+C0h] [rbp-1E0h] BYREF
  _BYTE *v366; // [rsp+C8h] [rbp-1D8h]
  _BYTE *v367; // [rsp+D0h] [rbp-1D0h]
  __int64 *v368; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 *v369; // [rsp+E8h] [rbp-1B8h]
  __int64 *v370; // [rsp+100h] [rbp-1A0h] BYREF
  _BYTE *v371; // [rsp+108h] [rbp-198h]
  _BYTE *v372; // [rsp+110h] [rbp-190h]
  __int64 v373; // [rsp+120h] [rbp-180h] BYREF
  __int64 v374; // [rsp+128h] [rbp-178h]
  __int64 v375; // [rsp+130h] [rbp-170h]
  unsigned int v376; // [rsp+138h] [rbp-168h]
  __int64 *v377; // [rsp+140h] [rbp-160h] BYREF
  __int64 v378; // [rsp+148h] [rbp-158h]
  _BYTE v379[32]; // [rsp+150h] [rbp-150h] BYREF
  const char *v380; // [rsp+170h] [rbp-130h] BYREF
  __int64 v381; // [rsp+178h] [rbp-128h]
  _QWORD v382[2]; // [rsp+180h] [rbp-120h] BYREF
  unsigned __int64 v383; // [rsp+190h] [rbp-110h]
  char v384; // [rsp+1A4h] [rbp-FCh]
  __int64 v385; // [rsp+1C0h] [rbp-E0h]
  unsigned int v386; // [rsp+1D0h] [rbp-D0h]
  unsigned __int64 v387; // [rsp+1D8h] [rbp-C8h]
  char *v388; // [rsp+220h] [rbp-80h]
  char v389; // [rsp+238h] [rbp-68h] BYREF
  unsigned __int64 v390; // [rsp+248h] [rbp-58h]

  v2 = a2 + 320;
  if ( a2 + 320 == *(_QWORD *)(*(_QWORD *)(a2 + 328) + 8LL) )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 520) = a2;
  v4 = a1;
  v5 = *v3;
  v6 = v3[1];
  if ( v5 == v6 )
    goto LABEL_664;
  while ( *(_UNKNOWN **)v5 != &unk_501F1C8 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_664;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_501F1C8);
  v8 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 528) = v7 + 169;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
    goto LABEL_664;
  while ( *(_UNKNOWN **)v9 != &unk_501EC08 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_664;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_501EC08)
      + 200;
  v12 = sub_22077B0(0x28u);
  if ( v12 )
  {
    *(_QWORD *)v12 = v11;
    *(_QWORD *)(v12 + 8) = 0;
    *(_QWORD *)(v12 + 16) = 0;
    *(_QWORD *)(v12 + 24) = 0;
    *(_DWORD *)(v12 + 32) = 0;
  }
  v13 = *(_QWORD *)(a1 + 536);
  *(_QWORD *)(a1 + 536) = v12;
  if ( v13 )
  {
    sub_C7D6A0(*(_QWORD *)(v13 + 16), 16LL * *(unsigned int *)(v13 + 32), 8);
    j_j___libc_free_0(v13);
  }
  v14 = *(__int64 **)(a1 + 8);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
    goto LABEL_664;
  while ( *(_UNKNOWN **)v15 != &unk_50208AC )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_664;
  }
  *(_QWORD *)(a1 + 544) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
                            *(_QWORD *)(v15 + 8),
                            &unk_50208AC)
                        + 200;
  v17 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v18 = 0;
  if ( v17 != sub_2DAC790 )
    v18 = v17();
  *(_QWORD *)(a1 + 560) = v18;
  v19 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 144LL);
  v20 = 0;
  if ( v19 != sub_2C8F680 )
    v20 = v19();
  v21 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 568) = v20;
  *(_QWORD *)(a1 + 576) = 0;
  v22 = *v21;
  v23 = v21[1];
  if ( v22 == v23 )
    goto LABEL_664;
  while ( *(_UNKNOWN **)v22 != &unk_4F87C64 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_664;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F87C64);
  v25 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 584) = *(_QWORD *)(v24 + 176);
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_664:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_5027190 )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_664;
  }
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(*(_QWORD *)(v26 + 8), &unk_5027190);
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 592) = v28;
  v29 = *(__int64 **)(a1 + 520);
  *(_QWORD *)(a1 + 776) = 0;
  sub_B2EE70((__int64)&v380, *v29, 0);
  if ( LOBYTE(v382[0]) )
  {
    v146 = sub_D844E0(*(_QWORD *)(a1 + 584));
    if ( v146 == -1 )
    {
      v149 = *(_QWORD *)(a1 + 520);
      v380 = 0;
      v150 = *(_QWORD *)(v149 + 328);
      for ( i = v149 + 320; i != v150; v150 = *(_QWORD *)(v150 + 8) )
      {
        v152 = (const char *)sub_2F06CB0(*(_QWORD *)(a1 + 536), v150);
        if ( v380 < v152 )
          v380 = v152;
      }
      sub_F02DB0(&v377, qword_503C2C8, 0x64u);
      v153 = sub_1098D20((unsigned __int64 *)&v380, (unsigned int)v377);
      *(_BYTE *)(a1 + 788) = 0;
      *(_QWORD *)(a1 + 776) = v153;
    }
    else
    {
      v147 = (unsigned int)qword_503C1E8;
      *(_BYTE *)(a1 + 788) = 1;
      *(_QWORD *)(a1 + 776) = v146 * v147 / 0x64uLL;
    }
  }
  *(_DWORD *)(a1 + 784) = qword_503C488;
  v30 = sub_C52410();
  v31 = v30 + 1;
  v32 = sub_C959E0();
  v33 = (_QWORD *)v30[2];
  if ( v33 )
  {
    v34 = v30 + 1;
    do
    {
      while ( 1 )
      {
        v35 = v33[2];
        v36 = v33[3];
        if ( v32 <= v33[4] )
          break;
        v33 = (_QWORD *)v33[3];
        if ( !v36 )
          goto LABEL_36;
      }
      v34 = v33;
      v33 = (_QWORD *)v33[2];
    }
    while ( v35 );
LABEL_36:
    if ( v31 != v34 && v32 >= v34[4] )
      v31 = v34;
  }
  if ( v31 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v37 = v31[7];
    if ( v37 )
    {
      v38 = v31 + 6;
      do
      {
        while ( 1 )
        {
          v39 = *(_QWORD *)(v37 + 16);
          v40 = *(_QWORD *)(v37 + 24);
          if ( *(_DWORD *)(v37 + 32) >= dword_503C328 )
            break;
          v37 = *(_QWORD *)(v37 + 24);
          if ( !v40 )
            goto LABEL_45;
        }
        v38 = (_QWORD *)v37;
        v37 = *(_QWORD *)(v37 + 16);
      }
      while ( v39 );
LABEL_45:
      if ( v31 + 6 != v38 && dword_503C328 >= *((_DWORD *)v38 + 8) && *((_DWORD *)v38 + 9) )
      {
        v94 = sub_C52410();
        v95 = v94 + 1;
        v96 = sub_C959E0();
        v97 = (_QWORD *)v94[2];
        if ( v97 )
        {
          v98 = v94 + 1;
          do
          {
            while ( 1 )
            {
              v99 = v97[2];
              v100 = v97[3];
              if ( v96 <= v97[4] )
                break;
              v97 = (_QWORD *)v97[3];
              if ( !v100 )
                goto LABEL_145;
            }
            v98 = v97;
            v97 = (_QWORD *)v97[2];
          }
          while ( v99 );
LABEL_145:
          if ( v95 != v98 && v96 >= v98[4] )
            v95 = v98;
        }
        if ( v95 == (_QWORD *)((char *)sub_C52410() + 8) )
          goto LABEL_157;
        v101 = v95[7];
        if ( !v101 )
          goto LABEL_157;
        v102 = v95 + 6;
        do
        {
          while ( 1 )
          {
            v103 = *(_QWORD *)(v101 + 16);
            v104 = *(_QWORD *)(v101 + 24);
            if ( *(_DWORD *)(v101 + 32) >= dword_503C408 )
              break;
            v101 = *(_QWORD *)(v101 + 24);
            if ( !v104 )
              goto LABEL_154;
          }
          v102 = (_QWORD *)v101;
          v101 = *(_QWORD *)(v101 + 16);
        }
        while ( v103 );
LABEL_154:
        if ( v95 + 6 == v102 || dword_503C408 < *((_DWORD *)v102 + 8) || !*((_DWORD *)v102 + 9) )
LABEL_157:
          *(_DWORD *)(v4 + 784) = qword_503C3A8;
      }
    }
  }
  if ( (int)sub_2FF0570(*(_QWORD *)(v4 + 592)) > 2 )
  {
    v105 = sub_C52410();
    v106 = v105 + 1;
    v107 = sub_C959E0();
    v108 = (_QWORD *)v105[2];
    if ( v108 )
    {
      v109 = v105 + 1;
      do
      {
        while ( 1 )
        {
          v110 = v108[2];
          v111 = v108[3];
          if ( v107 <= v108[4] )
            break;
          v108 = (_QWORD *)v108[3];
          if ( !v111 )
            goto LABEL_165;
        }
        v109 = v108;
        v108 = (_QWORD *)v108[2];
      }
      while ( v110 );
LABEL_165:
      if ( v106 != v109 && v107 >= v109[4] )
        v106 = v109;
    }
    if ( v106 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_186;
    v112 = v106[7];
    if ( !v112 )
      goto LABEL_186;
    v113 = v106 + 6;
    do
    {
      while ( 1 )
      {
        v114 = *(_QWORD *)(v112 + 16);
        v115 = *(_QWORD *)(v112 + 24);
        if ( *(_DWORD *)(v112 + 32) >= dword_503C408 )
          break;
        v112 = *(_QWORD *)(v112 + 24);
        if ( !v115 )
          goto LABEL_174;
      }
      v113 = (_QWORD *)v112;
      v112 = *(_QWORD *)(v112 + 16);
    }
    while ( v114 );
LABEL_174:
    if ( v106 + 6 == v113 || dword_503C408 < *((_DWORD *)v113 + 8) || !*((_DWORD *)v113 + 9) )
      goto LABEL_186;
    v116 = sub_C52410();
    v117 = v116 + 1;
    v118 = sub_C959E0();
    v119 = (_QWORD *)v116[2];
    if ( v119 )
    {
      v120 = v116 + 1;
      do
      {
        if ( v118 > v119[4] )
        {
          v119 = (_QWORD *)v119[3];
        }
        else
        {
          v120 = v119;
          v119 = (_QWORD *)v119[2];
        }
      }
      while ( v119 );
      if ( v117 != v120 && v118 >= v120[4] )
        v117 = v120;
    }
    if ( v117 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v321 = v117[7];
      if ( v321 )
      {
        v322 = v117 + 6;
        do
        {
          if ( *(_DWORD *)(v321 + 32) < dword_503C328 )
          {
            v321 = *(_QWORD *)(v321 + 24);
          }
          else
          {
            v322 = (_QWORD *)v321;
            v321 = *(_QWORD *)(v321 + 16);
          }
        }
        while ( v321 );
        if ( v117 + 6 != v322 && dword_503C328 >= *((_DWORD *)v322 + 8) && *((_DWORD *)v322 + 9) )
LABEL_186:
          *(_DWORD *)(v4 + 784) = qword_503C3A8;
      }
    }
  }
  v41 = sub_C52410();
  v42 = v41 + 1;
  v43 = sub_C959E0();
  v44 = (_QWORD *)v41[2];
  if ( v44 )
  {
    v45 = v41 + 1;
    do
    {
      while ( 1 )
      {
        v46 = v44[2];
        v47 = v44[3];
        if ( v43 <= v44[4] )
          break;
        v44 = (_QWORD *)v44[3];
        if ( !v47 )
          goto LABEL_53;
      }
      v45 = v44;
      v44 = (_QWORD *)v44[2];
    }
    while ( v46 );
LABEL_53:
    if ( v42 != v45 && v43 >= v45[4] )
      v42 = v45;
  }
  if ( v42 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_667;
  v48 = v42[7];
  if ( !v48 )
    goto LABEL_667;
  v49 = v42 + 6;
  do
  {
    while ( 1 )
    {
      v50 = *(_QWORD *)(v48 + 16);
      v51 = *(_QWORD *)(v48 + 24);
      if ( *(_DWORD *)(v48 + 32) >= dword_503C408 )
        break;
      v48 = *(_QWORD *)(v48 + 24);
      if ( !v51 )
        goto LABEL_62;
    }
    v49 = (_QWORD *)v48;
    v48 = *(_QWORD *)(v48 + 16);
  }
  while ( v50 );
LABEL_62:
  if ( v42 + 6 == v49 || dword_503C408 < *((_DWORD *)v49 + 8) || !*((_DWORD *)v49 + 9) )
  {
LABEL_667:
    if ( (int)sub_2FF0570(*(_QWORD *)(v4 + 592)) <= 2 )
      goto LABEL_136;
    v81 = sub_C52410();
    v82 = v81 + 1;
    v83 = sub_C959E0();
    v84 = (_QWORD *)v81[2];
    if ( v84 )
    {
      v85 = v81 + 1;
      do
      {
        while ( 1 )
        {
          v86 = v84[2];
          v87 = v84[3];
          if ( v83 <= v84[4] )
            break;
          v84 = (_QWORD *)v84[3];
          if ( !v87 )
            goto LABEL_118;
        }
        v85 = v84;
        v84 = (_QWORD *)v84[2];
      }
      while ( v86 );
LABEL_118:
      if ( v82 != v85 && v83 >= v85[4] )
        v82 = v85;
    }
    if ( v82 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_136;
    v88 = v82[7];
    if ( !v88 )
      goto LABEL_136;
    v89 = v82 + 6;
    do
    {
      if ( *(_DWORD *)(v88 + 32) < dword_503C328 )
      {
        v88 = *(_QWORD *)(v88 + 24);
      }
      else
      {
        v89 = (_QWORD *)v88;
        v88 = *(_QWORD *)(v88 + 16);
      }
    }
    while ( v88 );
    if ( v82 + 6 == v89 || dword_503C328 < *((_DWORD *)v89 + 8) || !*((_DWORD *)v89 + 9) )
    {
LABEL_136:
      v90 = *(_QWORD *)(v4 + 560);
      v91 = *(__int64 (__fastcall **)(__int64, int))(*(_QWORD *)v90 + 1488LL);
      v92 = (unsigned int)sub_2FF0570(*(_QWORD *)(v4 + 592));
      if ( v91 == sub_2FDC800 )
        v93 = 2 * ((int)v92 > 2) + 2;
      else
        v93 = v91(v90, v92);
      *(_DWORD *)(v4 + 784) = v93;
    }
  }
  v52 = sub_2EE6520((__int64 *)a2, *(_QWORD *)(v4 + 584), **(__int64 ***)(v4 + 536));
  v53 = *(_QWORD *)(a2 + 328);
  v54 = v52;
  if ( v2 == v53 )
    goto LABEL_158;
  v55 = *(_QWORD *)(a2 + 328);
  v56 = 0;
  do
  {
    v55 = *(_QWORD *)(v55 + 8);
    ++v56;
  }
  while ( v2 != v55 );
  if ( v56 <= 2 )
  {
LABEL_158:
    v58 = 0;
  }
  else
  {
    v57 = 0;
    do
    {
      v53 = *(_QWORD *)(v53 + 8);
      ++v57;
    }
    while ( v55 != v53 );
    v58 = 0;
    if ( (unsigned int)qword_503BF48 >= v57 )
    {
      v58 = qword_5008A68[8];
      if ( LOBYTE(qword_5008A68[8]) )
      {
        v58 = qword_5008988[8];
        if ( !LOBYTE(qword_5008988[8]) )
        {
          sub_B2EE70((__int64)&v380, *(_QWORD *)a2, 0);
          v58 = v382[0];
        }
      }
      if ( v54 && (_BYTE)qword_503BE68 )
      {
        v59 = qword_503C648;
        if ( !(_BYTE)qword_503C648 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 + 520) + 8LL) + 688LL) & 1) != 0 )
          goto LABEL_81;
        v148 = *(__int64 **)(v4 + 8);
        v61 = *v148;
        v62 = v148[1];
        if ( v62 == v61 )
          BUG();
        goto LABEL_77;
      }
    }
  }
  if ( !(_BYTE)qword_503C648 || (v59 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 + 520) + 8LL) + 688LL) & 1) != 0 )
  {
LABEL_129:
    sub_3521900(v4);
    v355 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(a2 + 8) + 688LL) & 1) != 0 )
      goto LABEL_88;
    v355 = *(_BYTE *)(*(_QWORD *)(v4 + 592) + 274LL);
    if ( !v355 )
      goto LABEL_88;
    v355 = byte_503C568;
    if ( !byte_503C568 )
      goto LABEL_88;
    v355 = 0;
    v64 = *(_QWORD *)(a2 + 328);
    if ( v2 == v64 )
      goto LABEL_88;
    goto LABEL_85;
  }
  v60 = *(__int64 **)(v4 + 8);
  v61 = *v60;
  v62 = v60[1];
  if ( v61 == v62 )
    goto LABEL_664;
LABEL_77:
  while ( *(_UNKNOWN **)v61 != &unk_50209DC )
  {
    v61 += 16;
    if ( v62 == v61 )
      goto LABEL_664;
  }
  *(_QWORD *)(v4 + 576) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v61 + 8) + 104LL))(
                            *(_QWORD *)(v61 + 8),
                            &unk_50209DC)
                        + 200;
  if ( v54 )
  {
    *(_DWORD *)(v4 + 784) = 1;
    v63 = 1;
  }
  else
  {
    v63 = *(_DWORD *)(v4 + 784);
  }
  sub_2FD5DC0(v4 + 600, a2, 0, *(_QWORD *)(v4 + 528), *(_QWORD *)(v4 + 536), *(_QWORD *)(v4 + 584), 1, v63);
  if ( !v59 )
  {
    sub_35185B0((_QWORD *)v4);
    goto LABEL_129;
  }
LABEL_81:
  v355 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 688LL) & 1;
  if ( v355 )
  {
LABEL_188:
    v66 = *(_QWORD *)(v4 + 520);
LABEL_189:
    v374 = 0;
    v121 = v66 + 320;
    v375 = 0;
    v376 = 0;
    v122 = *(_QWORD *)(v66 + 328);
    if ( v121 == v122 )
      goto LABEL_229;
    v123 = v122;
    v124 = 0;
    do
    {
      v123 = *(_QWORD *)(v123 + 8);
      ++v124;
    }
    while ( v121 != v123 );
    if ( !v124 )
    {
LABEL_229:
      v373 = 1;
    }
    else
    {
      v373 = 1;
      v125 = ((((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
              | (4 * v124 / 3u + 1)
              | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
            | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
            | (4 * v124 / 3u + 1)
            | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 8;
      sub_2E3E470(
        (__int64)&v373,
        (((v125
         | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
           | (4 * v124 / 3u + 1)
           | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
         | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
         | (4 * v124 / 3u + 1)
         | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 16)
       | v125
       | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
         | (4 * v124 / 3u + 1)
         | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
       | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
       | (4 * v124 / 3u + 1)
       | ((4 * v124 / 3u + 1) >> 1))
      + 1);
      v126 = *(_QWORD *)(v4 + 520);
      v121 = v126 + 320;
      v122 = *(_QWORD *)(v126 + 328);
    }
    v365 = 0;
    v366 = 0;
    v367 = 0;
    if ( v122 == v121 )
    {
      v128 = 0;
    }
    else
    {
      v127 = 0;
      do
      {
        v122 = *(_QWORD *)(v122 + 8);
        ++v127;
      }
      while ( v122 != v121 );
      v128 = v127;
    }
    sub_2E3A980((__int64)&v365, v128);
    v129 = *(_QWORD *)(v4 + 520);
    v130 = *(const char **)(v129 + 328);
    v131 = (const char *)(v129 + 320);
    v342 = (__int64)v131;
    if ( v130 == v131 )
      goto LABEL_562;
    v351 = v4;
    v132 = 0;
    v133 = v131;
    while ( 1 )
    {
      v142 = v132++;
      if ( !v376 )
        break;
      v134 = v374;
      v135 = 1;
      v136 = 0;
      v137 = (v376 - 1) & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
      v138 = (__int64 *)(v374 + 16LL * v137);
      v139 = *v138;
      if ( v130 != (const char *)*v138 )
      {
        while ( v139 != -4096 )
        {
          if ( !v136 && v139 == -8192 )
            v136 = v138;
          v137 = (v376 - 1) & (v135 + v137);
          v138 = (__int64 *)(v374 + 16LL * v137);
          v139 = *v138;
          if ( v130 == (const char *)*v138 )
            goto LABEL_201;
          v135 = (unsigned int)(v135 + 1);
        }
        if ( !v136 )
          v136 = v138;
        ++v373;
        v144 = v375 + 1;
        if ( 4 * ((int)v375 + 1) < 3 * v376 )
        {
          if ( v376 - HIDWORD(v375) - v144 <= v376 >> 3 )
          {
            sub_2E3E470((__int64)&v373, v376);
            if ( !v376 )
              goto LABEL_663;
            v302 = 0;
            LODWORD(v303) = (v376 - 1) & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
            v144 = v375 + 1;
            v304 = 1;
            v136 = (__int64 *)(v374 + 16LL * (unsigned int)v303);
            v134 = *v136;
            if ( v130 != (const char *)*v136 )
            {
              while ( v134 != -4096 )
              {
                if ( v134 == -8192 && !v302 )
                  v302 = v136;
                v135 = (unsigned int)(v304 + 1);
                v303 = (v376 - 1) & ((_DWORD)v303 + v304);
                v136 = (__int64 *)(v374 + 16 * v303);
                v134 = *v136;
                if ( v130 == (const char *)*v136 )
                  goto LABEL_210;
                ++v304;
              }
              if ( v302 )
                v136 = v302;
            }
          }
          goto LABEL_210;
        }
LABEL_208:
        sub_2E3E470((__int64)&v373, 2 * v376);
        if ( !v376 )
          goto LABEL_663;
        v134 = v376 - 1;
        LODWORD(v143) = v134 & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
        v144 = v375 + 1;
        v136 = (__int64 *)(v374 + 16LL * (unsigned int)v143);
        v145 = *v136;
        if ( v130 != (const char *)*v136 )
        {
          v334 = 1;
          v335 = 0;
          while ( v145 != -4096 )
          {
            if ( v145 == -8192 && !v335 )
              v335 = v136;
            v135 = (unsigned int)(v334 + 1);
            v143 = (unsigned int)v134 & ((_DWORD)v143 + v334);
            v136 = (__int64 *)(v374 + 16 * v143);
            v145 = *v136;
            if ( v130 == (const char *)*v136 )
              goto LABEL_210;
            ++v334;
          }
          if ( v335 )
            v136 = v335;
        }
LABEL_210:
        LODWORD(v375) = v144;
        if ( *v136 != -4096 )
          --HIDWORD(v375);
        *v136 = (__int64)v130;
        v140 = v136 + 1;
        *v140 = 0;
        goto LABEL_202;
      }
LABEL_201:
      v140 = v138 + 1;
LABEL_202:
      *v140 = v142;
      v141 = v366;
      v380 = v130;
      if ( v366 == v367 )
      {
        sub_2E3CE90((__int64)&v365, v366, &v380);
        v130 = (const char *)*((_QWORD *)v130 + 1);
        if ( v133 == v130 )
          goto LABEL_231;
      }
      else
      {
        if ( v366 )
        {
          *(_QWORD *)v366 = v130;
          v141 = v366;
        }
        v366 = v141 + 8;
        v130 = (const char *)*((_QWORD *)v130 + 1);
        if ( v133 == v130 )
        {
LABEL_231:
          v4 = v351;
          v154 = *(_QWORD *)(v351 + 520);
          v155 = *(_QWORD *)(v154 + 328);
          v156 = v154 + 320;
          if ( v154 + 320 != v155 )
          {
            v157 = *(_QWORD *)(v154 + 328);
            v158 = 0;
            do
            {
              v157 = *(_QWORD *)(v157 + 8);
              ++v158;
            }
            while ( v156 != v157 );
            v342 = v157;
            v359 = (__int64 *)&v361;
            v360 = 0;
            if ( !v158 )
              goto LABEL_240;
            sub_C8D5F0((__int64)&v359, &v361, v158, 8u, v135, v134);
            v159 = &v359[(unsigned int)v360];
            for ( j = &v359[v158]; j != v159; ++v159 )
            {
              if ( v159 )
                *v159 = 0;
            }
            v161 = *(_QWORD *)(v351 + 520);
            LODWORD(v360) = v158;
            v155 = *(_QWORD *)(v161 + 328);
            v342 = v161 + 320;
            if ( v155 != v161 + 320 )
            {
LABEL_240:
              v162 = v155;
              v163 = 0;
              do
              {
                v162 = *(_QWORD *)(v162 + 8);
                ++v163;
              }
              while ( v162 != v342 );
              v362 = 0;
              v361 = (const __m128i **)&v363;
              if ( v163 )
              {
                sub_C8D5F0((__int64)&v361, &v363, v163, 8u, v135, v134);
                v164 = &v361[(unsigned int)v362];
                for ( k = &v361[v163]; k != v164; ++v164 )
                {
                  if ( v164 )
                    *v164 = 0;
                }
                v166 = *(_QWORD *)(v351 + 520);
                LODWORD(v362) = v163;
                v155 = *(_QWORD *)(v166 + 328);
                v342 = v166 + 320;
              }
LABEL_248:
              v364 = 0;
              v363 = (__m128i *)&v365;
              v380 = (const char *)v382;
              v381 = 0x400000000LL;
              v377 = (__int64 *)v379;
              v378 = 0x400000000LL;
              if ( v342 == v155 )
              {
                v189 = &v365;
                v190 = 0;
                goto LABEL_273;
              }
              v346 = (__int64 *)v4;
              while ( 2 )
              {
                v167 = 1;
                v168 = sub_2F06CB0(v346[67], v155);
                v357 = v168;
                if ( !v355 )
                  v167 = v168;
                if ( v376 )
                {
                  v169 = 1;
                  v170 = 0;
                  v171 = (v376 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
                  v172 = (__int64 *)(v374 + 16LL * v171);
                  v173 = *v172;
                  if ( *v172 == v155 )
                  {
LABEL_254:
                    v174 = v172[1];
                    goto LABEL_255;
                  }
                  while ( v173 != -4096 )
                  {
                    if ( v173 == -8192 && !v170 )
                      v170 = v172;
                    v171 = (v376 - 1) & (v169 + v171);
                    v172 = (__int64 *)(v374 + 16LL * v171);
                    v173 = *v172;
                    if ( *v172 == v155 )
                      goto LABEL_254;
                    ++v169;
                  }
                  if ( !v170 )
                    v170 = v172;
                  ++v373;
                  v305 = v375 + 1;
                  if ( 4 * ((int)v375 + 1) < 3 * v376 )
                  {
                    if ( v376 - HIDWORD(v375) - v305 <= v376 >> 3 )
                    {
                      sub_2E3E470((__int64)&v373, v376);
                      if ( !v376 )
                        goto LABEL_663;
                      v309 = 0;
                      v310 = (v376 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
                      v311 = 1;
                      v305 = v375 + 1;
                      v170 = (__int64 *)(v374 + 16LL * v310);
                      v312 = *v170;
                      if ( v155 != *v170 )
                      {
                        while ( v312 != -4096 )
                        {
                          if ( v312 == -8192 && !v309 )
                            v309 = v170;
                          v310 = (v376 - 1) & (v311 + v310);
                          v170 = (__int64 *)(v374 + 16LL * v310);
                          v312 = *v170;
                          if ( *v170 == v155 )
                            goto LABEL_490;
                          ++v311;
                        }
                        if ( v309 )
                          v170 = v309;
                      }
                    }
LABEL_490:
                    LODWORD(v375) = v305;
                    if ( *v170 != -4096 )
                      --HIDWORD(v375);
                    *v170 = v155;
                    v174 = 0;
                    v170[1] = 0;
LABEL_255:
                    v175 = v155 + 48;
                    v359[v174] = v167;
                    v176 = *(_QWORD *)(v155 + 56);
                    if ( v155 + 48 == v176 )
                    {
LABEL_418:
                      v178 = 0;
                      goto LABEL_266;
                    }
                    while ( 1 )
                    {
                      v177 = *(_WORD *)(v176 + 68);
                      if ( (unsigned __int16)(v177 - 14) > 4u && v177 != 24 )
                        break;
                      v176 = *(_QWORD *)(v176 + 8);
                      if ( v175 == v176 )
                        goto LABEL_418;
                    }
                    v178 = 0;
                    if ( v175 == v176 )
                    {
LABEL_266:
                      v180 = v376;
                      if ( !v376 )
                        goto LABEL_316;
                    }
                    else
                    {
                      while ( 1 )
                      {
                        v176 = *(_QWORD *)(v176 + 8);
                        if ( v175 == v176 )
                          break;
                        v179 = *(_WORD *)(v176 + 68);
                        if ( (unsigned __int16)(v179 - 14) > 4u && v179 != 24 )
                        {
                          ++v178;
                          if ( v175 == v176 )
                          {
                            v178 *= 4;
                            goto LABEL_266;
                          }
                        }
                      }
                      v180 = v376;
                      v178 = 4 * v178 + 4;
                      if ( !v376 )
                      {
LABEL_316:
                        ++v373;
                        goto LABEL_317;
                      }
                    }
                    v181 = 1;
                    v182 = 0;
                    v183 = (v180 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
                    v184 = (__int64 *)(v374 + 16LL * v183);
                    v185 = *v184;
                    if ( *v184 == v155 )
                    {
LABEL_268:
                      v186 = v184[1];
                    }
                    else
                    {
                      while ( v185 != -4096 )
                      {
                        if ( v185 == -8192 && !v182 )
                          v182 = v184;
                        v183 = (v180 - 1) & (v181 + v183);
                        v184 = (__int64 *)(v374 + 16LL * v183);
                        v185 = *v184;
                        if ( *v184 == v155 )
                          goto LABEL_268;
                        ++v181;
                      }
                      if ( !v182 )
                        v182 = v184;
                      ++v373;
                      v213 = v375 + 1;
                      if ( 4 * ((int)v375 + 1) >= 3 * v180 )
                      {
LABEL_317:
                        sub_2E3E470((__int64)&v373, 2 * v180);
                        if ( !v376 )
                          goto LABEL_663;
                        v212 = (v376 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
                        v213 = v375 + 1;
                        v182 = (__int64 *)(v374 + 16LL * v212);
                        v214 = *v182;
                        if ( *v182 != v155 )
                        {
                          v215 = 1;
                          v216 = 0;
                          while ( v214 != -4096 )
                          {
                            if ( v214 == -8192 && !v216 )
                              v216 = v182;
                            v212 = (v376 - 1) & (v215 + v212);
                            v182 = (__int64 *)(v374 + 16LL * v212);
                            v214 = *v182;
                            if ( *v182 == v155 )
                              goto LABEL_477;
                            ++v215;
                          }
                          if ( v216 )
                            v182 = v216;
                        }
                      }
                      else if ( v180 - HIDWORD(v375) - v213 <= v180 >> 3 )
                      {
                        sub_2E3E470((__int64)&v373, v180);
                        if ( !v376 )
                          goto LABEL_663;
                        v317 = 0;
                        v318 = (v376 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
                        v319 = 1;
                        v213 = v375 + 1;
                        v182 = (__int64 *)(v374 + 16LL * v318);
                        v320 = *v182;
                        if ( *v182 != v155 )
                        {
                          while ( v320 != -4096 )
                          {
                            if ( v320 == -8192 && !v317 )
                              v317 = v182;
                            v318 = (v376 - 1) & (v319 + v318);
                            v182 = (__int64 *)(v374 + 16LL * v318);
                            v320 = *v182;
                            if ( *v182 == v155 )
                              goto LABEL_477;
                            ++v319;
                          }
                          if ( v317 )
                            v182 = v317;
                        }
                      }
LABEL_477:
                      LODWORD(v375) = v213;
                      if ( *v182 != -4096 )
                        --HIDWORD(v375);
                      *v182 = v155;
                      v186 = 0;
                      v182[1] = 0;
                    }
                    v361[v186] = (const __m128i *)v178;
                    if ( v355 )
                    {
                      LODWORD(v381) = 0;
                      v368 = 0;
                      v187 = v346[70];
                      v358 = 0;
                      v188 = *(__int64 (**)())(*(_QWORD *)v187 + 344LL);
                      if ( v188 != sub_2DB1AE0
                        && !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 **, const char **, _QWORD))v188)(
                              v187,
                              v155,
                              &v358,
                              &v368,
                              &v380,
                              0) )
                      {
                        v248 = sub_2E32300((__int64 *)v155, 1);
                        v251 = v358;
                        LODWORD(v378) = 0;
                        v252 = v248;
                        if ( v248 == v358 || !v358 )
                        {
                          v254 = (unsigned __int64)v368;
                          v255 = 0;
                          if ( v368 )
                            goto LABEL_356;
                          if ( v252 )
                            goto LABEL_414;
                        }
                        else
                        {
                          v253 = 0;
                          if ( !HIDWORD(v378) )
                          {
                            sub_C8D5F0((__int64)&v377, v379, 1u, 8u, v249, v250);
                            v253 = (unsigned int)v378;
                          }
                          v377[v253] = v251;
                          v254 = (unsigned __int64)v368;
                          v255 = (unsigned int)(v378 + 1);
                          LODWORD(v378) = v378 + 1;
                          if ( v368 )
                          {
LABEL_356:
                            if ( v252 != v254 )
                            {
                              if ( HIDWORD(v378) < (unsigned __int64)(v255 + 1) )
                              {
                                sub_C8D5F0((__int64)&v377, v379, v255 + 1, 8u, v249, v250);
                                v255 = (unsigned int)v378;
                              }
                              v377[v255] = v254;
                              v255 = (unsigned int)(v378 + 1);
                              LODWORD(v378) = v378 + 1;
                              goto LABEL_360;
                            }
LABEL_414:
                            if ( v255 + 1 > (unsigned __int64)HIDWORD(v378) )
                            {
                              sub_C8D5F0((__int64)&v377, v379, v255 + 1, 8u, v249, v250);
                              v255 = (unsigned int)v378;
                            }
                            v377[v255] = v252;
                            v255 = (unsigned int)(v378 + 1);
                            LODWORD(v378) = v378 + 1;
                          }
                          else
                          {
LABEL_360:
                            if ( v252 )
                              goto LABEL_414;
                          }
                          v256 = v377;
                          v257 = 110;
                          if ( v255 != 1 )
                            v257 = 100;
                          v353 = &v377[v255];
                          if ( v377 != v353 )
                          {
                            v258 = v257;
                            v348 = ((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4);
                            do
                            {
                              v278 = v376;
                              v279 = *v256;
                              if ( v376 )
                              {
                                v259 = v376 - 1;
                                v260 = 1;
                                v261 = 0;
                                v262 = v374;
                                v263 = (v376 - 1) & v348;
                                v264 = (__int64 *)(v374 + 16LL * v263);
                                v265 = *v264;
                                if ( *v264 == v155 )
                                {
LABEL_366:
                                  v370 = (__int64 *)v264[1];
                                  goto LABEL_367;
                                }
                                while ( v265 != -4096 )
                                {
                                  if ( v265 == -8192 && !v261 )
                                    v261 = v264;
                                  v263 = v259 & (v260 + v263);
                                  v264 = (__int64 *)(v374 + 16LL * v263);
                                  v265 = *v264;
                                  if ( *v264 == v155 )
                                    goto LABEL_366;
                                  ++v260;
                                }
                                if ( !v261 )
                                  v261 = v264;
                                ++v373;
                                v281 = v375 + 1;
                                if ( 4 * ((int)v375 + 1) < 3 * v376 )
                                {
                                  if ( v376 - HIDWORD(v375) - v281 > v376 >> 3 )
                                    goto LABEL_375;
                                  v341 = v258;
                                  sub_2E3E470((__int64)&v373, v376);
                                  if ( !v376 )
                                    goto LABEL_663;
                                  v298 = 0;
                                  v258 = v341;
                                  v299 = 1;
                                  v300 = (v376 - 1) & v348;
                                  v281 = v375 + 1;
                                  v261 = (__int64 *)(v374 + 16LL * v300);
                                  v301 = *v261;
                                  if ( *v261 == v155 )
                                    goto LABEL_375;
                                  while ( v301 != -4096 )
                                  {
                                    if ( v301 == -8192 && !v298 )
                                      v298 = v261;
                                    v300 = (v376 - 1) & (v299 + v300);
                                    v261 = (__int64 *)(v374 + 16LL * v300);
                                    v301 = *v261;
                                    if ( *v261 == v155 )
                                      goto LABEL_375;
                                    ++v299;
                                  }
                                  goto LABEL_448;
                                }
                              }
                              else
                              {
                                ++v373;
                              }
                              v337 = v258;
                              sub_2E3E470((__int64)&v373, 2 * v376);
                              if ( !v376 )
                                goto LABEL_663;
                              v258 = v337;
                              v280 = (v376 - 1) & v348;
                              v281 = v375 + 1;
                              v261 = (__int64 *)(v374 + 16LL * v280);
                              v282 = *v261;
                              if ( *v261 == v155 )
                                goto LABEL_375;
                              v331 = 1;
                              v298 = 0;
                              while ( v282 != -4096 )
                              {
                                if ( v282 == -8192 && !v298 )
                                  v298 = v261;
                                v280 = (v376 - 1) & (v331 + v280);
                                v261 = (__int64 *)(v374 + 16LL * v280);
                                v282 = *v261;
                                if ( *v261 == v155 )
                                  goto LABEL_375;
                                ++v331;
                              }
LABEL_448:
                              if ( v298 )
                                v261 = v298;
LABEL_375:
                              LODWORD(v375) = v281;
                              if ( *v261 != -4096 )
                                --HIDWORD(v375);
                              *v261 = v155;
                              v261[1] = 0;
                              v278 = v376;
                              v370 = 0;
                              v262 = v374;
                              if ( !v376 )
                              {
                                ++v373;
                                goto LABEL_379;
                              }
                              v259 = v376 - 1;
LABEL_367:
                              v266 = 1;
                              v267 = 0;
                              v268 = v259 & (((unsigned int)v279 >> 9) ^ ((unsigned int)v279 >> 4));
                              v269 = (_QWORD *)(v262 + 16LL * v268);
                              v270 = *v269;
                              if ( *v269 != v279 )
                              {
                                while ( v270 != -4096 )
                                {
                                  if ( v270 == -8192 && !v267 )
                                    v267 = v269;
                                  v268 = v259 & (v266 + v268);
                                  v269 = (_QWORD *)(v262 + 16LL * v268);
                                  v270 = *v269;
                                  if ( v279 == *v269 )
                                    goto LABEL_368;
                                  ++v266;
                                }
                                if ( !v267 )
                                  v267 = v269;
                                ++v373;
                                v284 = v375 + 1;
                                if ( 4 * ((int)v375 + 1) >= 3 * v278 )
                                {
LABEL_379:
                                  v338 = v258;
                                  sub_2E3E470((__int64)&v373, 2 * v278);
                                  if ( !v376 )
                                    goto LABEL_663;
                                  v258 = v338;
                                  v283 = (v376 - 1) & (((unsigned int)v279 >> 9) ^ ((unsigned int)v279 >> 4));
                                  v284 = v375 + 1;
                                  v267 = (_QWORD *)(v374 + 16LL * v283);
                                  v285 = *v267;
                                  if ( v279 != *v267 )
                                  {
                                    v332 = 1;
                                    v333 = 0;
                                    while ( v285 != -4096 )
                                    {
                                      if ( !v333 && v285 == -8192 )
                                        v333 = v267;
                                      v283 = (v376 - 1) & (v332 + v283);
                                      v267 = (_QWORD *)(v374 + 16LL * v283);
                                      v285 = *v267;
                                      if ( v279 == *v267 )
                                        goto LABEL_381;
                                      ++v332;
                                    }
                                    if ( v333 )
                                      v267 = v333;
                                  }
                                }
                                else if ( v278 - (v284 + HIDWORD(v375)) <= v278 >> 3 )
                                {
                                  v340 = v258;
                                  sub_2E3E470((__int64)&v373, v278);
                                  if ( !v376 )
                                    goto LABEL_663;
                                  v294 = 0;
                                  v295 = 1;
                                  v296 = (v376 - 1) & (((unsigned int)v279 >> 9) ^ ((unsigned int)v279 >> 4));
                                  v258 = v340;
                                  v284 = v375 + 1;
                                  v267 = (_QWORD *)(v374 + 16LL * v296);
                                  v297 = *v267;
                                  if ( v279 != *v267 )
                                  {
                                    while ( v297 != -4096 )
                                    {
                                      if ( !v294 && v297 == -8192 )
                                        v294 = v267;
                                      v296 = (v376 - 1) & (v295 + v296);
                                      v267 = (_QWORD *)(v374 + 16LL * v296);
                                      v297 = *v267;
                                      if ( v279 == *v267 )
                                        goto LABEL_381;
                                      ++v295;
                                    }
                                    if ( v294 )
                                      v267 = v294;
                                  }
                                }
LABEL_381:
                                LODWORD(v375) = v284;
                                if ( *v267 != -4096 )
                                  --HIDWORD(v375);
                                *v267 = v279;
                                v271 = 0;
                                v267[1] = 0;
                                goto LABEL_369;
                              }
LABEL_368:
                              v271 = (_BYTE *)v269[1];
LABEL_369:
                              v371 = v271;
                              v272 = (unsigned int)v364;
                              v273 = (const __m128i *)&v370;
                              v372 = (_BYTE *)v258;
                              v274 = (unsigned int)v364 + 1LL;
                              v275 = v363;
                              if ( v274 > HIDWORD(v364) )
                              {
                                v339 = v258;
                                if ( v363 > (__m128i *)&v370 || &v370 >= (__int64 **)v363 + 3 * (unsigned int)v364 )
                                {
                                  v273 = (const __m128i *)&v370;
                                  sub_C8D5F0((__int64)&v363, &v365, (unsigned int)v364 + 1LL, 0x18u, v258, v274);
                                  v275 = v363;
                                  v272 = (unsigned int)v364;
                                  v258 = v339;
                                }
                                else
                                {
                                  v286 = (char *)&v370 - (char *)v363;
                                  sub_C8D5F0((__int64)&v363, &v365, (unsigned int)v364 + 1LL, 0x18u, v258, v274);
                                  v275 = v363;
                                  v272 = (unsigned int)v364;
                                  v258 = v339;
                                  v273 = (__m128i *)((char *)v363 + v286);
                                }
                              }
                              ++v256;
                              v276 = (__m128i *)((char *)v275 + 24 * v272);
                              *v276 = _mm_loadu_si128(v273);
                              v277 = v273[1].m128i_i64[0];
                              LODWORD(v364) = v364 + 1;
                              v276[1].m128i_i64[0] = v277;
                            }
                            while ( v353 != v256 );
                          }
                        }
                      }
                    }
                    else
                    {
                      v217 = *(__int64 **)(v155 + 112);
                      v352 = &v217[*(unsigned int *)(v155 + 120)];
                      if ( v217 != v352 )
                      {
                        v344 = ((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4);
                        while ( 1 )
                        {
                          v235 = *v217;
                          v236 = sub_2E441D0(v346[66], v155, *v217);
                          v237 = sub_1098D20((unsigned __int64 *)&v357, v236);
                          v238 = v376;
                          v239 = (_BYTE *)v237;
                          if ( !v376 )
                            break;
                          v218 = v376 - 1;
                          v219 = 1;
                          v220 = 0;
                          v221 = v374;
                          v222 = (v376 - 1) & v344;
                          v223 = (__int64 *)(v374 + 16LL * v222);
                          v224 = *v223;
                          if ( v155 == *v223 )
                            goto LABEL_327;
                          while ( 1 )
                          {
                            if ( v224 == -4096 )
                            {
                              if ( !v220 )
                                v220 = v223;
                              ++v373;
                              v241 = v375 + 1;
                              if ( 4 * ((int)v375 + 1) < 3 * v376 )
                              {
                                if ( v376 - HIDWORD(v375) - v241 <= v376 >> 3 )
                                {
                                  sub_2E3E470((__int64)&v373, v376);
                                  if ( !v376 )
                                    goto LABEL_663;
                                  v290 = 0;
                                  v291 = 1;
                                  v292 = (v376 - 1) & v344;
                                  v241 = v375 + 1;
                                  v220 = (__int64 *)(v374 + 16LL * v292);
                                  v293 = *v220;
                                  if ( *v220 != v155 )
                                  {
                                    while ( v293 != -4096 )
                                    {
                                      if ( !v290 && v293 == -8192 )
                                        v290 = v220;
                                      v292 = (v376 - 1) & (v291 + v292);
                                      v220 = (__int64 *)(v374 + 16LL * v292);
                                      v293 = *v220;
                                      if ( *v220 == v155 )
                                        goto LABEL_336;
                                      ++v291;
                                    }
LABEL_408:
                                    if ( v290 )
                                      v220 = v290;
                                  }
                                }
LABEL_336:
                                LODWORD(v375) = v241;
                                if ( *v220 != -4096 )
                                  --HIDWORD(v375);
                                *v220 = v155;
                                v220[1] = 0;
                                v238 = v376;
                                v370 = 0;
                                v221 = v374;
                                if ( v376 )
                                {
                                  v218 = v376 - 1;
                                  goto LABEL_328;
                                }
                                ++v373;
                                goto LABEL_340;
                              }
LABEL_334:
                              sub_2E3E470((__int64)&v373, 2 * v376);
                              if ( !v376 )
                                goto LABEL_663;
                              v240 = (v376 - 1) & v344;
                              v241 = v375 + 1;
                              v220 = (__int64 *)(v374 + 16LL * v240);
                              v242 = *v220;
                              if ( *v220 != v155 )
                              {
                                v330 = 1;
                                v290 = 0;
                                while ( v242 != -4096 )
                                {
                                  if ( !v290 && v242 == -8192 )
                                    v290 = v220;
                                  v240 = (v376 - 1) & (v330 + v240);
                                  v220 = (__int64 *)(v374 + 16LL * v240);
                                  v242 = *v220;
                                  if ( *v220 == v155 )
                                    goto LABEL_336;
                                  ++v330;
                                }
                                goto LABEL_408;
                              }
                              goto LABEL_336;
                            }
                            if ( v224 != -8192 || v220 )
                              v223 = v220;
                            v222 = v218 & (v219 + v222);
                            v224 = *(_QWORD *)(v374 + 16LL * v222);
                            if ( v224 == v155 )
                              break;
                            ++v219;
                            v220 = v223;
                            v223 = (__int64 *)(v374 + 16LL * v222);
                          }
                          v223 = (__int64 *)(v374 + 16LL * v222);
LABEL_327:
                          v370 = (__int64 *)v223[1];
LABEL_328:
                          v225 = v218 & (((unsigned int)v235 >> 9) ^ ((unsigned int)v235 >> 4));
                          v226 = (__int64 *)(v221 + 16 * v225);
                          v227 = *v226;
                          if ( v235 == *v226 )
                            goto LABEL_329;
                          v336 = 1;
                          v245 = 0;
                          while ( 2 )
                          {
                            if ( v227 == -4096 )
                            {
                              if ( !v245 )
                                v245 = v226;
                              ++v373;
                              v244 = v375 + 1;
                              if ( 4 * ((int)v375 + 1) < 3 * v238 )
                              {
                                if ( v238 - (v244 + HIDWORD(v375)) > v238 >> 3 )
                                  goto LABEL_342;
                                sub_2E3E470((__int64)&v373, v238);
                                if ( v376 )
                                {
                                  v225 = 1;
                                  v244 = v375 + 1;
                                  v287 = 0;
                                  v288 = (v376 - 1) & (((unsigned int)v235 >> 9) ^ ((unsigned int)v235 >> 4));
                                  v245 = (__int64 *)(v374 + 16 * v288);
                                  v289 = *v245;
                                  if ( v235 != *v245 )
                                  {
                                    while ( v289 != -4096 )
                                    {
                                      if ( !v287 && v289 == -8192 )
                                        v287 = v245;
                                      LODWORD(v288) = (v376 - 1) & (v225 + v288);
                                      v245 = (__int64 *)(v374 + 16LL * (unsigned int)v288);
                                      v289 = *v245;
                                      if ( v235 == *v245 )
                                        goto LABEL_342;
                                      v225 = (unsigned int)(v225 + 1);
                                    }
LABEL_396:
                                    if ( v287 )
                                      v245 = v287;
                                  }
LABEL_342:
                                  LODWORD(v375) = v244;
                                  if ( *v245 != -4096 )
                                    --HIDWORD(v375);
                                  *v245 = v235;
                                  v228 = 0;
                                  v245[1] = 0;
                                  goto LABEL_330;
                                }
LABEL_663:
                                LODWORD(v375) = v375 + 1;
                                goto LABEL_664;
                              }
LABEL_340:
                              sub_2E3E470((__int64)&v373, 2 * v238);
                              if ( v376 )
                              {
                                v243 = (v376 - 1) & (((unsigned int)v235 >> 9) ^ ((unsigned int)v235 >> 4));
                                v244 = v375 + 1;
                                v245 = (__int64 *)(v374 + 16LL * v243);
                                v246 = *v245;
                                if ( v235 != *v245 )
                                {
                                  v225 = 1;
                                  v287 = 0;
                                  while ( v246 != -4096 )
                                  {
                                    if ( !v287 && v246 == -8192 )
                                      v287 = v245;
                                    v243 = (v376 - 1) & (v225 + v243);
                                    v245 = (__int64 *)(v374 + 16LL * v243);
                                    v246 = *v245;
                                    if ( v235 == *v245 )
                                      goto LABEL_342;
                                    v225 = (unsigned int)(v225 + 1);
                                  }
                                  goto LABEL_396;
                                }
                                goto LABEL_342;
                              }
                              goto LABEL_663;
                            }
                            if ( v227 != -8192 || v245 )
                              v226 = v245;
                            v225 = v218 & (v336 + (_DWORD)v225);
                            v227 = *(_QWORD *)(v221 + 16LL * (unsigned int)v225);
                            if ( v235 != v227 )
                            {
                              ++v336;
                              v245 = v226;
                              v226 = (__int64 *)(v221 + 16LL * (unsigned int)v225);
                              continue;
                            }
                            break;
                          }
                          v226 = (__int64 *)(v221 + 16LL * (unsigned int)v225);
LABEL_329:
                          v228 = (_BYTE *)v226[1];
LABEL_330:
                          v371 = v228;
                          v229 = (unsigned int)v364;
                          v372 = v239;
                          v230 = (unsigned int)v364 + 1LL;
                          v231 = v363;
                          v232 = (const __m128i *)&v370;
                          if ( v230 > HIDWORD(v364) )
                          {
                            if ( v363 > (__m128i *)&v370 || &v370 >= (__int64 **)v363 + 3 * (unsigned int)v364 )
                            {
                              sub_C8D5F0((__int64)&v363, &v365, (unsigned int)v364 + 1LL, 0x18u, v230, v225);
                              v231 = v363;
                              v229 = (unsigned int)v364;
                              v232 = (const __m128i *)&v370;
                            }
                            else
                            {
                              v247 = (char *)&v370 - (char *)v363;
                              sub_C8D5F0((__int64)&v363, &v365, (unsigned int)v364 + 1LL, 0x18u, v230, v225);
                              v231 = v363;
                              v229 = (unsigned int)v364;
                              v232 = (__m128i *)((char *)v363 + v247);
                            }
                          }
                          ++v217;
                          v233 = (__m128i *)((char *)v231 + 24 * v229);
                          *v233 = _mm_loadu_si128(v232);
                          v234 = v232[1].m128i_i64[0];
                          LODWORD(v364) = v364 + 1;
                          v233[1].m128i_i64[0] = v234;
                          if ( v352 == v217 )
                            goto LABEL_271;
                        }
                        ++v373;
                        goto LABEL_334;
                      }
                    }
LABEL_271:
                    v155 = *(_QWORD *)(v155 + 8);
                    if ( v155 == v342 )
                    {
                      v4 = (__int64)v346;
                      v189 = (__int64 **)v363;
                      v190 = (unsigned int)v364;
LABEL_273:
                      v191 = sub_29BAF70((__int64)v361, (unsigned int)v362, (__int64)v189, v190);
                      sub_29BB2B0(
                        &v368,
                        (__int64)v361,
                        (unsigned int)v362,
                        (__int64)v359,
                        (unsigned int)v360,
                        (unsigned int)v362,
                        v363,
                        (unsigned int)v364);
                      v192 = *(_QWORD *)(v4 + 520);
                      v370 = 0;
                      v371 = 0;
                      v372 = 0;
                      v193 = v192 + 320;
                      v194 = *(_QWORD *)(v192 + 328);
                      if ( v193 == v194 )
                      {
                        v196 = 0;
                      }
                      else
                      {
                        v195 = 0;
                        do
                        {
                          v194 = *(_QWORD *)(v194 + 8);
                          ++v195;
                        }
                        while ( v193 != v194 );
                        v196 = v195;
                      }
                      sub_2E3A980((__int64)&v370, v196);
                      v197 = v368;
                      v198 = v369;
                      if ( v368 == v369 )
                      {
                        v329 = v369;
                      }
                      else
                      {
                        do
                        {
                          while ( 1 )
                          {
                            v199 = v371;
                            v200 = &v365[*v197];
                            if ( v371 != v372 )
                              break;
                            ++v197;
                            sub_2E417A0((__int64)&v370, v371, v200);
                            if ( v198 == v197 )
                              goto LABEL_284;
                          }
                          if ( v371 )
                          {
                            *(_QWORD *)v371 = *v200;
                            v199 = v371;
                          }
                          ++v197;
                          v371 = v199 + 8;
                        }
                        while ( v198 != v197 );
LABEL_284:
                        v198 = v368;
                        v329 = v369;
                      }
                      v201 = sub_29BAC40(
                               v198,
                               v329 - v198,
                               (__int64)v361,
                               (unsigned int)v362,
                               (__int64)v363,
                               (unsigned int)v364);
                      if ( v355 && v191 > v201 )
                        sub_3519A10(v4, &v365);
                      else
                        sub_3519A10(v4, &v370);
                      if ( v370 )
                        j_j___libc_free_0((unsigned __int64)v370);
                      if ( v368 )
                        j_j___libc_free_0((unsigned __int64)v368);
                      if ( v377 != (__int64 *)v379 )
                        _libc_free((unsigned __int64)v377);
                      if ( v380 != (const char *)v382 )
                        _libc_free((unsigned __int64)v380);
                      if ( v363 != (__m128i *)&v365 )
                        _libc_free((unsigned __int64)v363);
                      if ( v361 != (const __m128i **)&v363 )
                        _libc_free((unsigned __int64)v361);
                      if ( v359 != (__int64 *)&v361 )
                        _libc_free((unsigned __int64)v359);
                      if ( v365 )
                        j_j___libc_free_0((unsigned __int64)v365);
                      sub_C7D6A0(v374, 16LL * v376, 8);
                      v356 = v4 + 888;
                      sub_3511770(v4 + 888);
                      v347 = v4 + 488;
                      sub_35142F0(v4 + 488);
                      v350 = v4 + 792;
                      sub_3510940(v4 + 792);
                      v202 = *(_QWORD *)(v4 + 792);
                      v203 = *(const char **)(*(_QWORD *)(v4 + 520) + 328LL);
                      *(_QWORD *)(v4 + 872) += 64LL;
                      v204 = (v202 + 7) & 0xFFFFFFFFFFFFFFF8LL;
                      v354 = v203;
                      if ( *(_QWORD *)(v4 + 800) >= v204 + 64 && v202 )
                        *(_QWORD *)(v4 + 792) = v204 + 64;
                      else
                        v204 = sub_9D1E70(v350, 64, 64, 3);
                      v380 = v354;
                      *(_QWORD *)v204 = v204 + 16;
                      v205 = v380;
                      *(_QWORD *)(v204 + 8) = 0x400000001LL;
                      *(_QWORD *)(v204 + 16) = v205;
                      *(_QWORD *)(v204 + 48) = v356;
                      *(_DWORD *)(v204 + 56) = 0;
                      *sub_3515040(v356, (__int64 *)&v380) = v204;
                      v207 = *(_QWORD *)(v4 + 520);
                      v67 = (const char *)(v207 + 320);
                      if ( *(_QWORD *)(v207 + 328) != v207 + 320 )
                      {
                        v343 = v4;
                        v208 = *(const char **)(v207 + 328);
                        v209 = (const char *)(v207 + 320);
                        do
                        {
                          if ( v354 != v208 )
                          {
                            v380 = v208;
                            v210 = *(unsigned int *)(v204 + 8);
                            if ( v210 + 1 > (unsigned __int64)*(unsigned int *)(v204 + 12) )
                            {
                              sub_C8D5F0(v204, (const void *)(v204 + 16), v210 + 1, 8u, v207, v206);
                              v210 = *(unsigned int *)(v204 + 8);
                            }
                            *(_QWORD *)(*(_QWORD *)v204 + 8 * v210) = v208;
                            v211 = *(_QWORD *)(v204 + 48);
                            ++*(_DWORD *)(v204 + 8);
                            *sub_3515040(v211, (__int64 *)&v380) = v204;
                          }
                          v208 = (const char *)*((_QWORD *)v208 + 1);
                        }
                        while ( v209 != v208 );
                        v4 = v343;
                        v67 = *(const char **)(*(_QWORD *)(v343 + 520) + 328LL);
                      }
                      goto LABEL_90;
                    }
                    continue;
                  }
                }
                else
                {
                  ++v373;
                }
                break;
              }
              sub_2E3E470((__int64)&v373, 2 * v376);
              if ( !v376 )
                goto LABEL_663;
              v313 = (v376 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
              v305 = v375 + 1;
              v170 = (__int64 *)(v374 + 16LL * v313);
              v314 = *v170;
              if ( *v170 != v155 )
              {
                v315 = 1;
                v316 = 0;
                while ( v314 != -4096 )
                {
                  if ( !v316 && v314 == -8192 )
                    v316 = v170;
                  v313 = (v376 - 1) & (v315 + v313);
                  v170 = (__int64 *)(v374 + 16LL * v313);
                  v314 = *v170;
                  if ( *v170 == v155 )
                    goto LABEL_490;
                  ++v315;
                }
                if ( v316 )
                  v170 = v316;
              }
              goto LABEL_490;
            }
LABEL_563:
            v362 = 0;
            v361 = (const __m128i **)&v363;
            goto LABEL_248;
          }
          v342 = v154 + 320;
LABEL_562:
          v155 = v342;
          v360 = 0;
          v359 = (__int64 *)&v361;
          goto LABEL_563;
        }
      }
    }
    ++v373;
    goto LABEL_208;
  }
  v355 = *(_BYTE *)(*(_QWORD *)(v4 + 592) + 274LL);
  if ( !v355 )
  {
    v355 = 1;
    goto LABEL_188;
  }
  v58 = byte_503C568;
  if ( !byte_503C568 )
    goto LABEL_188;
  v355 = byte_503C568;
  v64 = *(_QWORD *)(a2 + 328);
  if ( v2 == v64 )
    goto LABEL_188;
LABEL_85:
  v65 = 0;
  do
  {
    v64 = *(_QWORD *)(v64 + 8);
    ++v65;
  }
  while ( v2 != v64 );
  if ( v65 > 3 )
  {
    sub_34BEDF0(
      (__int64)&v380,
      1,
      0,
      *(_QWORD *)(v4 + 536),
      *(_QWORD *)(v4 + 528),
      *(_QWORD *)(v4 + 584),
      *(_DWORD *)(v4 + 784) + 1);
    v323 = *(_QWORD *)(v4 + 544);
    v324 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
    if ( (unsigned __int8)sub_34C7080((__int64)&v380, (_QWORD *)a2, *(_QWORD *)(v4 + 560), v324, v323, 1) )
    {
      v325 = *(_QWORD *)(v4 + 576);
      if ( v325 )
      {
        *(_QWORD *)(v325 + 128) = a2;
        *(_DWORD *)(v325 + 144) = *(_DWORD *)(a2 + 120);
        sub_2EBA550(v325);
      }
      if ( !v355 )
      {
        sub_3511770(v4 + 888);
        sub_35142F0(v4 + 488);
        sub_3510940(v4 + 792);
        sub_3521900(v4);
      }
    }
    if ( v390 )
      _libc_free(v390);
    if ( v388 != &v389 )
      _libc_free((unsigned __int64)v388);
    if ( v387 )
      j_j___libc_free_0(v387);
    sub_C7D6A0(v385, 16LL * v386, 8);
    if ( !v384 )
      _libc_free(v383);
    v326 = (const char *)v381;
    v327 = v380;
    if ( (const char *)v381 != v380 )
    {
      do
      {
        v328 = *((_QWORD *)v327 + 2);
        if ( v328 )
          sub_B91220((__int64)(v327 + 16), v328);
        v327 += 24;
      }
      while ( v326 != v327 );
      v327 = v380;
    }
    if ( v327 )
      j_j___libc_free_0((unsigned __int64)v327);
  }
LABEL_88:
  v66 = *(_QWORD *)(v4 + 520);
  v67 = *(const char **)(v66 + 328);
  if ( v58 )
    goto LABEL_189;
  v356 = v4 + 888;
  v347 = v4 + 488;
  v350 = v4 + 792;
LABEL_90:
  v380 = v67;
  v68 = *sub_3515040(v356, (__int64 *)&v380);
  v380 = (const char *)v382;
  v381 = 0x400000000LL;
  v69 = *(_QWORD *)v68 + 8LL * *(unsigned int *)(v68 + 8);
  if ( *(_QWORD *)v68 != v69 )
  {
    v70 = *(__int64 **)v68;
    do
    {
      while ( 1 )
      {
        v71 = *(_QWORD *)(v4 + 560);
        v72 = *v70;
        LODWORD(v381) = 0;
        v370 = 0;
        v373 = 0;
        v73 = *(__int64 (**)())(*(_QWORD *)v71 + 344LL);
        if ( v73 != sub_2DB1AE0
          && !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 **, __int64 *, const char **, __int64))v73)(
                v71,
                v72,
                &v370,
                &v373,
                &v380,
                1) )
        {
          if ( v370 )
          {
            if ( v373 )
            {
              if ( (_DWORD)v381 )
              {
                if ( !sub_2EE6AD0(v72, *(_QWORD *)(v4 + 584), *(__int64 ***)(v4 + 536)) )
                {
                  v345 = sub_2E441D0(*(_QWORD *)(v4 + 528), v72, v373);
                  if ( v345 > (unsigned int)sub_2E441D0(*(_QWORD *)(v4 + 528), v72, (__int64)v370) )
                  {
                    v74 = *(_QWORD *)(v4 + 560);
                    v75 = *(__int64 (**)())(*(_QWORD *)v74 + 880LL);
                    if ( v75 != sub_2DB1B20
                      && !((unsigned __int8 (__fastcall *)(__int64, const char **))v75)(v74, &v380) )
                    {
                      sub_2E32880((__int64 *)&v377, v72);
                      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v4 + 560) + 360LL))(
                        *(_QWORD *)(v4 + 560),
                        v72,
                        0);
                      (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64 *, const char *, _QWORD, __int64 **, _QWORD))(**(_QWORD **)(v4 + 560) + 368LL))(
                        *(_QWORD *)(v4 + 560),
                        v72,
                        v373,
                        v370,
                        v380,
                        (unsigned int)v381,
                        &v377,
                        0);
                      if ( v377 )
                        break;
                    }
                  }
                }
              }
            }
          }
        }
        if ( (__int64 *)v69 == ++v70 )
          goto LABEL_104;
      }
      ++v70;
      sub_B91220((__int64)&v377, (__int64)v377);
    }
    while ( (__int64 *)v69 != v70 );
LABEL_104:
    if ( v380 != (const char *)v382 )
      _libc_free((unsigned __int64)v380);
  }
  sub_3516980(v4);
  sub_3511770(v356);
  sub_35142F0(v347);
  sub_3510940(v350);
  if ( LODWORD(qword_501ED48[8]) )
  {
    v76 = qword_4F8DF28[9];
    if ( !qword_4F8DF28[9]
      || (v306 = (const void *)qword_4F8DF28[8], v307 = sub_BD5D20(**(_QWORD **)(v4 + 520)), v76 == v308)
      && !memcmp(v307, v306, v76) )
    {
      if ( byte_503C028 )
        sub_2E7A760(a2, 0);
      v77 = *(__int64 **)(v4 + 536);
      v78 = sub_2E791E0((__int64 *)a2);
      v382[1] = v79;
      LOWORD(v383) = 1283;
      v380 = "MBP.";
      v382[0] = v78;
      sub_2F06DC0(v77, (void **)&v380, 0);
    }
  }
  return 1;
}
