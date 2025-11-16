// Function: sub_1EF5620
// Address: 0x1ef5620
//
__int64 __fastcall sub_1EF5620(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r15
  _QWORD *v32; // rax
  __int64 v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // rax
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // r13
  __int64 v39; // r15
  __int64 i; // r14
  __int64 ***v41; // r8
  __int64 ***v42; // r15
  __int64 ***j; // r14
  __int64 v44; // r13
  __int64 v45; // rsi
  __int64 v46; // rdx
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // r12
  __int64 v52; // rbx
  __int64 v53; // rax
  unsigned __int8 *v54; // rsi
  __int64 v55; // r12
  unsigned __int64 *v56; // r12
  unsigned __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rsi
  unsigned __int8 *v60; // rsi
  __int64 v61; // r12
  __int64 v62; // r13
  unsigned __int64 *v63; // r12
  unsigned __int64 v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rsi
  unsigned __int8 *v67; // rsi
  __int64 *v68; // r12
  unsigned int v69; // r13d
  _QWORD *v70; // rax
  unsigned __int64 *v71; // r12
  unsigned __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rsi
  unsigned __int8 *v75; // rsi
  _QWORD *v76; // rax
  _QWORD *v77; // r12
  unsigned __int64 *v78; // r13
  __int64 v79; // rax
  unsigned __int64 v80; // rcx
  double v81; // xmm4_8
  double v82; // xmm5_8
  __int64 v83; // rsi
  unsigned __int8 *v84; // rsi
  __int64 v85; // rax
  double v86; // xmm4_8
  double v87; // xmm5_8
  _QWORD *v88; // r12
  __int64 v89; // rax
  unsigned __int8 *v90; // rsi
  __int64 v91; // r12
  __int64 *v92; // rax
  __int64 v93; // rax
  __int64 v94; // r12
  __int64 v95; // r13
  _QWORD *v96; // rax
  unsigned __int8 *v97; // rsi
  _QWORD *v98; // rax
  __int64 v99; // r15
  __int64 *v100; // r12
  __int64 v101; // rax
  __int64 v102; // rcx
  __int64 v103; // rsi
  unsigned __int8 *v104; // rsi
  _QWORD *v105; // r12
  unsigned int v106; // r15d
  __int64 v107; // rax
  __int64 v108; // rdi
  __int64 v109; // rsi
  unsigned __int64 v110; // r9
  _QWORD *v111; // rax
  _QWORD **v112; // rax
  __int64 *v113; // rax
  __int64 v114; // rsi
  unsigned __int64 *v115; // r15
  __int64 v116; // rax
  unsigned __int64 v117; // rsi
  __int64 v118; // rsi
  unsigned __int8 *v119; // rsi
  _QWORD *v120; // r12
  __int64 v121; // rax
  double v122; // xmm4_8
  double v123; // xmm5_8
  __int64 v124; // r13
  __int64 v125; // r9
  unsigned __int64 v126; // rcx
  unsigned __int64 v127; // r13
  _QWORD *v128; // r15
  unsigned __int64 *v129; // r14
  __int64 v130; // rax
  unsigned __int64 v131; // rsi
  __int64 v132; // rsi
  unsigned __int8 *v133; // rsi
  _QWORD *v134; // rax
  _QWORD *v135; // r14
  unsigned __int64 *v136; // r15
  __int64 v137; // rax
  unsigned __int64 v138; // rsi
  __int64 v139; // rsi
  unsigned __int8 *v140; // rsi
  __int64 v141; // rax
  __int64 v142; // rdx
  unsigned __int8 *v143; // rsi
  unsigned __int8 *v144; // rsi
  unsigned __int64 v145; // r12
  unsigned __int8 *v146; // rsi
  _QWORD *v147; // rax
  _QWORD *v148; // r14
  unsigned __int64 v149; // rsi
  __int64 v150; // rax
  __int64 v151; // rsi
  __int64 v152; // rdx
  unsigned __int8 *v153; // rsi
  __int64 v154; // rax
  unsigned __int8 *v155; // rsi
  __int64 v156; // rdx
  _QWORD *v157; // rax
  _QWORD *v158; // rdx
  __int64 *v159; // r14
  __int64 *v160; // r13
  __int64 v161; // r15
  __int64 *v162; // rbx
  __int64 *v163; // r12
  __int64 v164; // rdi
  __int64 v165; // rax
  __int64 v166; // rax
  void *v167; // rdi
  unsigned int v168; // eax
  __int64 v169; // rdx
  unsigned __int64 v170; // rdi
  __int64 v171; // rax
  __int64 v172; // rdi
  __int64 v173; // rdi
  unsigned __int64 *v174; // rbx
  unsigned __int64 *v175; // r13
  unsigned __int64 v176; // rdi
  unsigned __int64 *v177; // rdx
  unsigned __int64 *v178; // r13
  unsigned __int64 *v179; // rbx
  unsigned __int64 v180; // rdi
  unsigned __int64 *v181; // rbx
  unsigned __int64 *v182; // r13
  unsigned __int64 v183; // rdi
  _QWORD *v184; // rbx
  _QWORD *v185; // r13
  __int64 v186; // r12
  __int64 v187; // rdi
  char v189; // al
  __int64 v190; // rbx
  unsigned __int64 v191; // rax
  __int64 v192; // rax
  __int64 v193; // rax
  __int64 v194; // rax
  __int64 v195; // rax
  __int64 v196; // rax
  __int64 v197; // rax
  unsigned int v198; // ecx
  _QWORD *v199; // rdi
  unsigned int v200; // eax
  int v201; // eax
  unsigned __int64 v202; // rax
  unsigned __int64 v203; // rax
  int v204; // ebx
  __int64 v205; // r13
  _QWORD *v206; // rax
  _QWORD *k; // rdx
  __int64 v208; // r12
  __int64 *v209; // rax
  __int64 v210; // rax
  __int64 v211; // r12
  __int64 *v212; // r14
  unsigned int v213; // edx
  _QWORD *v214; // rax
  unsigned __int64 *v215; // r14
  __int64 v216; // rax
  unsigned __int64 v217; // rcx
  __int64 v218; // rsi
  unsigned __int8 *v219; // rsi
  _QWORD *v220; // rax
  _QWORD *v221; // r14
  unsigned __int64 v222; // rsi
  __int64 v223; // rax
  __int64 v224; // rsi
  __int64 v225; // rdx
  unsigned __int8 *v226; // rsi
  __int64 v227; // rax
  char v228; // r13
  unsigned __int8 v229; // cl
  unsigned __int64 v230; // rdx
  __int64 *v231; // r12
  __int64 v232; // r15
  unsigned __int64 v233; // r14
  bool v234; // zf
  _QWORD *v235; // r12
  _QWORD *v236; // rdi
  __int64 v237; // rax
  __int64 v238; // rax
  _QWORD *v239; // rax
  int v240; // eax
  int v241; // eax
  unsigned int v242; // eax
  __int64 v243; // rdi
  __int64 v244; // rsi
  unsigned __int64 v245; // rax
  __int64 v246; // rcx
  unsigned __int64 v247; // rax
  int v248; // ecx
  double v249; // xmm4_8
  double v250; // xmm5_8
  __int64 v251; // rax
  _QWORD *v252; // rbx
  _QWORD *v253; // r14
  __int64 v254; // rax
  __int64 v255; // rax
  _QWORD *v256; // rax
  int v257; // eax
  __int64 v258; // rax
  _QWORD *v259; // rax
  __int64 v260; // rax
  int v261; // eax
  _QWORD *v262; // [rsp+48h] [rbp-978h]
  _QWORD *v263; // [rsp+50h] [rbp-970h]
  __int64 v264; // [rsp+50h] [rbp-970h]
  __int64 v265; // [rsp+58h] [rbp-968h]
  __int64 *v266; // [rsp+60h] [rbp-960h]
  __int64 v267; // [rsp+60h] [rbp-960h]
  __int64 v268; // [rsp+60h] [rbp-960h]
  _QWORD *v269; // [rsp+68h] [rbp-958h]
  __int64 v270; // [rsp+68h] [rbp-958h]
  unsigned __int64 v271; // [rsp+68h] [rbp-958h]
  __int64 v272; // [rsp+70h] [rbp-950h]
  __int64 v273; // [rsp+70h] [rbp-950h]
  unsigned int v274; // [rsp+78h] [rbp-948h]
  __int64 v275; // [rsp+78h] [rbp-948h]
  unsigned __int64 v276; // [rsp+78h] [rbp-948h]
  __int64 v277; // [rsp+80h] [rbp-940h]
  _QWORD *v278; // [rsp+88h] [rbp-938h]
  __int64 v279; // [rsp+88h] [rbp-938h]
  __int64 v280; // [rsp+88h] [rbp-938h]
  unsigned __int64 v281; // [rsp+88h] [rbp-938h]
  unsigned __int64 v282; // [rsp+88h] [rbp-938h]
  unsigned __int64 v283; // [rsp+88h] [rbp-938h]
  __int64 v284; // [rsp+88h] [rbp-938h]
  __int64 v285; // [rsp+90h] [rbp-930h]
  __int64 v286; // [rsp+90h] [rbp-930h]
  __int64 v287; // [rsp+90h] [rbp-930h]
  unsigned __int64 v288; // [rsp+90h] [rbp-930h]
  __int64 *v289; // [rsp+90h] [rbp-930h]
  unsigned int v290; // [rsp+90h] [rbp-930h]
  unsigned __int64 *v291; // [rsp+90h] [rbp-930h]
  __int64 *v292; // [rsp+98h] [rbp-928h]
  __int64 v293; // [rsp+98h] [rbp-928h]
  unsigned __int64 v294; // [rsp+98h] [rbp-928h]
  __int64 v295; // [rsp+98h] [rbp-928h]
  __int64 v296; // [rsp+98h] [rbp-928h]
  unsigned __int64 *v297; // [rsp+98h] [rbp-928h]
  unsigned __int8 v298; // [rsp+98h] [rbp-928h]
  __int64 v299; // [rsp+98h] [rbp-928h]
  unsigned __int8 *v300; // [rsp+A8h] [rbp-918h] BYREF
  unsigned __int8 *v301[2]; // [rsp+B0h] [rbp-910h] BYREF
  _QWORD v302[2]; // [rsp+C0h] [rbp-900h] BYREF
  __int64 *v303; // [rsp+D0h] [rbp-8F0h] BYREF
  __int64 v304; // [rsp+D8h] [rbp-8E8h]
  _BYTE v305[32]; // [rsp+E0h] [rbp-8E0h] BYREF
  __int64 ****v306; // [rsp+100h] [rbp-8C0h] BYREF
  __int64 v307; // [rsp+108h] [rbp-8B8h]
  _BYTE v308[32]; // [rsp+110h] [rbp-8B0h] BYREF
  __int64 *v309; // [rsp+130h] [rbp-890h] BYREF
  __int64 v310; // [rsp+138h] [rbp-888h]
  _BYTE v311[32]; // [rsp+140h] [rbp-880h] BYREF
  _BYTE *v312; // [rsp+160h] [rbp-860h] BYREF
  __int64 v313; // [rsp+168h] [rbp-858h]
  _BYTE v314[32]; // [rsp+170h] [rbp-850h] BYREF
  __int64 v315; // [rsp+190h] [rbp-830h] BYREF
  __int64 v316; // [rsp+198h] [rbp-828h]
  __int64 v317; // [rsp+1A0h] [rbp-820h]
  _BYTE *v318; // [rsp+1A8h] [rbp-818h]
  __int64 *v319; // [rsp+1B0h] [rbp-810h]
  __int64 v320; // [rsp+1B8h] [rbp-808h]
  __int64 v321; // [rsp+1C0h] [rbp-800h]
  __int64 v322; // [rsp+1C8h] [rbp-7F8h]
  __int64 v323; // [rsp+1D0h] [rbp-7F0h]
  unsigned __int8 *v324; // [rsp+1E0h] [rbp-7E0h] BYREF
  __int64 v325; // [rsp+1E8h] [rbp-7D8h]
  unsigned __int64 *v326; // [rsp+1F0h] [rbp-7D0h]
  __int64 v327; // [rsp+1F8h] [rbp-7C8h]
  __int64 v328; // [rsp+200h] [rbp-7C0h]
  int v329; // [rsp+208h] [rbp-7B8h]
  __int64 v330; // [rsp+210h] [rbp-7B0h]
  __int64 v331; // [rsp+218h] [rbp-7A8h]
  unsigned __int8 *v332; // [rsp+230h] [rbp-790h] BYREF
  __int64 v333; // [rsp+238h] [rbp-788h]
  __int64 *v334; // [rsp+240h] [rbp-780h]
  _QWORD *v335; // [rsp+248h] [rbp-778h]
  __int64 v336; // [rsp+250h] [rbp-770h]
  int v337; // [rsp+258h] [rbp-768h]
  __int64 v338; // [rsp+260h] [rbp-760h]
  __int64 v339; // [rsp+268h] [rbp-758h]
  unsigned __int64 v340[2]; // [rsp+280h] [rbp-740h] BYREF
  char v341; // [rsp+290h] [rbp-730h] BYREF
  __int64 v342; // [rsp+298h] [rbp-728h]
  _QWORD *v343; // [rsp+2A0h] [rbp-720h]
  __int64 v344; // [rsp+2A8h] [rbp-718h]
  unsigned int v345; // [rsp+2B0h] [rbp-710h]
  __int64 v346; // [rsp+2C0h] [rbp-700h]
  char v347; // [rsp+2C8h] [rbp-6F8h]
  int v348; // [rsp+2CCh] [rbp-6F4h]
  __int64 *v349; // [rsp+2D0h] [rbp-6F0h] BYREF
  __int64 v350; // [rsp+2D8h] [rbp-6E8h]
  _BYTE v351[128]; // [rsp+2E0h] [rbp-6E0h] BYREF
  __int64 v352; // [rsp+360h] [rbp-660h] BYREF
  _QWORD *v353; // [rsp+368h] [rbp-658h]
  __int64 v354; // [rsp+370h] [rbp-650h]
  unsigned int v355; // [rsp+378h] [rbp-648h]
  __int64 *v356; // [rsp+380h] [rbp-640h]
  __int64 *v357; // [rsp+388h] [rbp-638h]
  __int64 v358; // [rsp+390h] [rbp-630h]
  unsigned __int64 v359; // [rsp+398h] [rbp-628h]
  unsigned __int64 v360; // [rsp+3A0h] [rbp-620h]
  unsigned __int64 *v361; // [rsp+3A8h] [rbp-618h]
  unsigned int v362; // [rsp+3B0h] [rbp-610h]
  char v363; // [rsp+3B8h] [rbp-608h] BYREF
  unsigned __int64 *v364; // [rsp+3D8h] [rbp-5E8h]
  unsigned int v365; // [rsp+3E0h] [rbp-5E0h]
  __int64 v366; // [rsp+3E8h] [rbp-5D8h] BYREF
  unsigned __int8 *v367; // [rsp+400h] [rbp-5C0h] BYREF
  __int64 v368; // [rsp+408h] [rbp-5B8h]
  _QWORD *v369; // [rsp+410h] [rbp-5B0h] BYREF
  __int64 v370; // [rsp+418h] [rbp-5A8h]
  __int64 v371; // [rsp+420h] [rbp-5A0h]
  _QWORD *v372; // [rsp+428h] [rbp-598h]
  __int64 v373; // [rsp+430h] [rbp-590h]
  _QWORD v374[4]; // [rsp+438h] [rbp-588h] BYREF
  _BYTE *v375; // [rsp+458h] [rbp-568h]
  __int64 v376; // [rsp+460h] [rbp-560h]
  _BYTE v377[192]; // [rsp+468h] [rbp-558h] BYREF
  _BYTE *v378; // [rsp+528h] [rbp-498h]
  __int64 v379; // [rsp+530h] [rbp-490h]
  _BYTE v380[72]; // [rsp+538h] [rbp-488h] BYREF
  _BYTE v381[1088]; // [rsp+580h] [rbp-440h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
    goto LABEL_425;
  while ( *(_UNKNOWN **)v11 != &unk_4FCBA30 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_425;
  }
  v15 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
                      *(_QWORD *)(v11 + 8),
                      &unk_4FCBA30)
                  + 208);
  *(_QWORD *)(a1 + 160) = v15;
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 16LL);
  if ( v16 == sub_16FF750 )
    goto LABEL_425;
  v17 = ((__int64 (__fastcall *)(__int64, __int64))v16)(v15, a2);
  v18 = *(__int64 (**)())(*(_QWORD *)v17 + 56LL);
  if ( v18 == sub_1D12D20 || (v285 = ((__int64 (__fastcall *)(__int64))v18)(v17)) == 0 )
    sub_16BD130("TargetLowering instance is required", 1u);
  v19 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
    goto LABEL_425;
  while ( *(_UNKNOWN **)v22 != &unk_4F9B6E8 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_425;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9B6E8);
  v25 = *(__int64 **)(a1 + 8);
  v26 = v24 + 360;
  v27 = *v25;
  v28 = v25[1];
  if ( v27 == v28 )
LABEL_425:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F9D764 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_425;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F9D764);
  v30 = sub_14CF090(v29, a2);
  v342 = 0;
  v31 = v30;
  v343 = 0;
  v340[0] = (unsigned __int64)&v341;
  v340[1] = 0x100000000LL;
  v344 = 0;
  v345 = 0;
  v347 = 0;
  v348 = 0;
  v346 = a2;
  sub_15D3930((__int64)v340);
  sub_14019E0((__int64)&v352, (__int64)v340);
  sub_1457DF0((__int64)v381, a2, v26, v31, (__int64)v340, (__int64)&v352);
  v317 = v21;
  v315 = a2;
  v316 = v285;
  v318 = v381;
  v32 = (_QWORD *)sub_15E0530(a2);
  v319 = (__int64 *)sub_16471D0(v32, 0);
  v33 = sub_15E0530(a2);
  v320 = sub_15A9620(v21, v33, 0);
  v34 = (_QWORD *)sub_15E0530(a2);
  v321 = sub_1643350(v34);
  v35 = (_QWORD *)sub_15E0530(a2);
  v323 = 0;
  v322 = sub_1643330(v35);
  v349 = (__int64 *)v351;
  v350 = 0x1000000000LL;
  v303 = (__int64 *)v305;
  v304 = 0x400000000LL;
  v307 = 0x400000000LL;
  v310 = 0x400000000LL;
  v313 = 0x400000000LL;
  v306 = (__int64 ****)v308;
  v38 = *(_QWORD *)(a2 + 80);
  v39 = a2 + 72;
  v309 = (__int64 *)v311;
  v312 = v314;
  if ( a2 + 72 != v38 )
  {
    while ( v38 )
    {
      i = *(_QWORD *)(v38 + 24);
      if ( i != v38 + 16 )
        goto LABEL_273;
      v38 = *(_QWORD *)(v38 + 8);
      if ( v39 == v38 )
        goto LABEL_21;
    }
LABEL_427:
    BUG();
  }
  i = 0;
LABEL_273:
  if ( v39 != v38 )
  {
    if ( !i )
LABEL_290:
      BUG();
    while ( 1 )
    {
      v189 = *(_BYTE *)(i - 8);
      v190 = i - 24;
      switch ( v189 )
      {
        case 53:
          v191 = sub_1EEF2C0(v317, i - 24);
          v33 = i - 24;
          if ( !(unsigned __int8)sub_1EF49F0((__int64)&v315, i - 24, v191, a3, a4) )
          {
            if ( (unsigned __int8)sub_15F8F00(i - 24) )
            {
              v192 = (unsigned int)v350;
              if ( (unsigned int)v350 >= HIDWORD(v350) )
              {
                v33 = (__int64)v351;
                sub_16CD150((__int64)&v349, v351, 0, 8, v36, v37);
                v192 = (unsigned int)v350;
              }
              v349[v192] = v190;
              LODWORD(v350) = v350 + 1;
            }
            else
            {
              v195 = (unsigned int)v304;
              if ( (unsigned int)v304 >= HIDWORD(v304) )
              {
                v33 = (__int64)v305;
                sub_16CD150((__int64)&v303, v305, 0, 8, v36, v37);
                v195 = (unsigned int)v304;
              }
              v303[v195] = v190;
              LODWORD(v304) = v304 + 1;
            }
          }
          break;
        case 25:
          v194 = (unsigned int)v310;
          if ( (unsigned int)v310 >= HIDWORD(v310) )
          {
            v33 = (__int64)v311;
            sub_16CD150((__int64)&v309, v311, 0, 8, v36, v37);
            v194 = (unsigned int)v310;
          }
          v309[v194] = v190;
          LODWORD(v310) = v310 + 1;
          break;
        case 78:
          if ( !*(_BYTE *)(*(_QWORD *)(i - 48) + 16LL) )
          {
            v33 = 0xFFFFFFFFLL;
            if ( (unsigned __int8)sub_1560260((_QWORD *)(i + 32), -1, 39)
              || (v196 = *(_QWORD *)(i - 48), !*(_BYTE *)(v196 + 16))
              && (v33 = 0xFFFFFFFFLL,
                  v367 = *(unsigned __int8 **)(v196 + 112),
                  (unsigned __int8)sub_1560260(&v367, -1, 39)) )
            {
LABEL_303:
              v197 = (unsigned int)v313;
              if ( (unsigned int)v313 >= HIDWORD(v313) )
              {
                v33 = (__int64)v314;
                sub_16CD150((__int64)&v312, v314, 0, 8, v36, v37);
                v197 = (unsigned int)v313;
              }
              *(_QWORD *)&v312[8 * v197] = v190;
              LODWORD(v313) = v313 + 1;
            }
          }
          break;
        case 88:
          goto LABEL_303;
      }
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v38 + 24) )
      {
        v193 = v38 - 24;
        if ( !v38 )
          v193 = 0;
        if ( i != v193 + 40 )
          break;
        v38 = *(_QWORD *)(v38 + 8);
        if ( v39 == v38 )
          goto LABEL_21;
        if ( !v38 )
          goto LABEL_427;
      }
      if ( v39 == v38 )
        break;
      if ( !i )
        goto LABEL_290;
    }
  }
LABEL_21:
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, v33);
    v41 = *(__int64 ****)(a2 + 88);
    v42 = &v41[5 * *(_QWORD *)(a2 + 96)];
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, v33);
      v41 = *(__int64 ****)(a2 + 88);
    }
  }
  else
  {
    v41 = *(__int64 ****)(a2 + 88);
    v42 = &v41[5 * *(_QWORD *)(a2 + 96)];
  }
  for ( j = v41; v42 != j; j += 5 )
  {
    if ( (unsigned __int8)sub_15E0450((__int64)j) )
    {
      v44 = 1;
      v45 = *(*j)[2];
      while ( 2 )
      {
        switch ( *(_BYTE *)(v45 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v107 = *(_QWORD *)(v45 + 32);
            v45 = *(_QWORD *)(v45 + 24);
            v44 *= v107;
            continue;
          case 1:
            v46 = 16;
            break;
          case 2:
            v46 = 32;
            break;
          case 3:
          case 9:
            v46 = 64;
            break;
          case 4:
            v46 = 80;
            break;
          case 5:
          case 6:
            v46 = 128;
            break;
          case 7:
            v46 = 8 * (unsigned int)sub_15A9520(v317, 0);
            break;
          case 0xB:
            v46 = *(_DWORD *)(v45 + 8) >> 8;
            break;
          case 0xD:
            v46 = 8LL * *(_QWORD *)sub_15A9930(v317, v45);
            break;
          case 0xE:
            v279 = v317;
            v287 = *(_QWORD *)(v45 + 32);
            v108 = v317;
            v293 = 1;
            v109 = *(_QWORD *)(v45 + 24);
            v110 = (unsigned int)sub_15A9FE0(v317, v109);
            while ( 2 )
            {
              switch ( *(_BYTE *)(v109 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v238 = v293 * *(_QWORD *)(v109 + 32);
                  v109 = *(_QWORD *)(v109 + 24);
                  v293 = v238;
                  continue;
                case 1:
                  v227 = 16;
                  break;
                case 2:
                  v227 = 32;
                  break;
                case 3:
                case 9:
                  v227 = 64;
                  break;
                case 4:
                  v227 = 80;
                  break;
                case 5:
                case 6:
                  v227 = 128;
                  break;
                case 7:
                  v282 = v110;
                  v240 = sub_15A9520(v108, 0);
                  v110 = v282;
                  v227 = (unsigned int)(8 * v240);
                  break;
                case 0xB:
                  v227 = *(_DWORD *)(v109 + 8) >> 8;
                  break;
                case 0xD:
                  v281 = v110;
                  v239 = (_QWORD *)sub_15A9930(v108, v109);
                  v110 = v281;
                  v227 = 8LL * *v239;
                  break;
                case 0xE:
                  v267 = v110;
                  v272 = v279;
                  v270 = *(_QWORD *)(v109 + 24);
                  v277 = *(_QWORD *)(v109 + 32);
                  v242 = sub_15A9FE0(v279, v270);
                  v243 = v279;
                  v284 = 1;
                  v110 = v267;
                  v244 = v270;
                  v276 = v242;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v244 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v260 = v284 * *(_QWORD *)(v244 + 32);
                        v244 = *(_QWORD *)(v244 + 24);
                        v284 = v260;
                        continue;
                      case 1:
                        v255 = 16;
                        break;
                      case 2:
                        v255 = 32;
                        break;
                      case 3:
                      case 9:
                        v255 = 64;
                        break;
                      case 4:
                        v255 = 80;
                        break;
                      case 5:
                      case 6:
                        v255 = 128;
                        break;
                      case 7:
                        v261 = sub_15A9520(v243, 0);
                        v110 = v267;
                        v255 = (unsigned int)(8 * v261);
                        break;
                      case 0xB:
                        v255 = *(_DWORD *)(v244 + 8) >> 8;
                        break;
                      case 0xD:
                        v259 = (_QWORD *)sub_15A9930(v243, v244);
                        v110 = v267;
                        v255 = 8LL * *v259;
                        break;
                      case 0xE:
                        v264 = v267;
                        v268 = v272;
                        v265 = *(_QWORD *)(v244 + 24);
                        v273 = *(_QWORD *)(v244 + 32);
                        v271 = (unsigned int)sub_15A9FE0(v243, v265);
                        v258 = sub_127FA20(v268, v265);
                        v110 = v264;
                        v255 = 8 * v273 * v271 * ((v271 + ((unsigned __int64)(v258 + 7) >> 3) - 1) / v271);
                        break;
                      case 0xF:
                        v257 = sub_15A9520(v243, *(_DWORD *)(v244 + 8) >> 8);
                        v110 = v267;
                        v255 = (unsigned int)(8 * v257);
                        break;
                    }
                    break;
                  }
                  v227 = 8 * v276 * v277 * ((v276 + ((unsigned __int64)(v284 * v255 + 7) >> 3) - 1) / v276);
                  break;
                case 0xF:
                  v283 = v110;
                  v241 = sub_15A9520(v108, *(_DWORD *)(v109 + 8) >> 8);
                  v110 = v283;
                  v227 = (unsigned int)(8 * v241);
                  break;
              }
              break;
            }
            v46 = 8 * v110 * v287 * ((v110 + ((unsigned __int64)(v293 * v227 + 7) >> 3) - 1) / v110);
            break;
          case 0xF:
            v46 = 8 * (unsigned int)sub_15A9520(v317, *(_DWORD *)(v45 + 8) >> 8);
            break;
        }
        break;
      }
      if ( !(unsigned __int8)sub_1EF49F0((__int64)&v315, (__int64)j, (unsigned __int64)(v46 * v44 + 7) >> 3, a3, a4) )
      {
        v49 = (unsigned int)v307;
        if ( (unsigned int)v307 >= HIDWORD(v307) )
        {
          sub_16CD150((__int64)&v306, v308, 0, 8, v47, v48);
          v49 = (unsigned int)v307;
        }
        v306[v49] = j;
        LODWORD(v307) = v307 + 1;
      }
    }
  }
  if ( !((unsigned int)v313 | (unsigned int)v307 | (unsigned int)v304 | (unsigned int)v350) )
  {
    v298 = 0;
    goto LABEL_205;
  }
  v50 = *(_QWORD *)(v315 + 80);
  if ( v50 )
    v50 -= 24;
  v51 = sub_157EE30(v50);
  v52 = *(_QWORD *)(v315 + 80);
  if ( v52 )
    v52 -= 24;
  v53 = sub_157E9C0(v52);
  v325 = v52;
  v324 = 0;
  v327 = v53;
  v328 = 0;
  v329 = 0;
  v330 = 0;
  v331 = 0;
  v326 = (unsigned __int64 *)v51;
  if ( v51 != v52 + 40 )
  {
    if ( !v51 )
      goto LABEL_427;
    v54 = *(unsigned __int8 **)(v51 + 24);
    v367 = v54;
    if ( v54 )
    {
      sub_1623A60((__int64)&v367, (__int64)v54, 2);
      if ( v324 )
        sub_161E7C0((__int64)&v324, (__int64)v324);
      v324 = v367;
      if ( v367 )
        sub_1623210((__int64)&v367, v367, (__int64)&v324);
    }
  }
  if ( byte_4FCA2C0 )
  {
    v208 = *(_QWORD *)(v315 + 40);
    v209 = (__int64 *)sub_1647190(v319, 0);
    v367 = (unsigned __int8 *)&v369;
    v368 = 0;
    v210 = sub_1644EA0(v209, &v369, 0, 0);
    v211 = sub_1632080(v208, (__int64)"__safestack_pointer_address", 27, v210, 0);
    if ( v367 != (unsigned __int8 *)&v369 )
      _libc_free((unsigned __int64)v367);
    LOWORD(v369) = 257;
    v323 = sub_1285290((__int64 *)&v324, *(_QWORD *)(*(_QWORD *)v211 + 24LL), v211, 0, 0, (__int64)&v367, 0);
    v55 = v323;
  }
  else
  {
    v323 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 **))(*(_QWORD *)v316 + 560LL))(v316, &v324);
    v55 = v323;
  }
  v367 = (unsigned __int8 *)"unsafe_stack_ptr";
  LOWORD(v369) = 259;
  v263 = sub_1648A60(64, 1u);
  if ( v263 )
    sub_15F9210((__int64)v263, *(_QWORD *)(*(_QWORD *)v55 + 24LL), v55, 0, 0, 0);
  if ( v325 )
  {
    v56 = v326;
    sub_157E9D0(v325 + 40, (__int64)v263);
    v57 = *v56;
    v58 = v263[3];
    v263[4] = v56;
    v57 &= 0xFFFFFFFFFFFFFFF8LL;
    v263[3] = v57 | v58 & 7;
    *(_QWORD *)(v57 + 8) = v263 + 3;
    *v56 = *v56 & 7 | (unsigned __int64)(v263 + 3);
  }
  sub_164B780((__int64)v263, (__int64 *)&v367);
  if ( v324 )
  {
    v332 = v324;
    sub_1623A60((__int64)&v332, (__int64)v324, 2);
    v59 = v263[6];
    if ( v59 )
      sub_161E7C0((__int64)(v263 + 6), v59);
    v60 = v332;
    v263[6] = v332;
    if ( v60 )
      sub_1623210((__int64)&v332, v60, (__int64)(v263 + 6));
  }
  if ( (unsigned __int8)sub_1560180(v315 + 112, 49)
    || (unsigned __int8)sub_1560180(v315 + 112, 51)
    || (unsigned __int8)sub_1560180(v315 + 112, 50) )
  {
    v61 = v315;
    v62 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 **))(*(_QWORD *)v316 + 520LL))(v316, &v324);
    if ( !v62 )
      v62 = sub_1632210(*(_QWORD *)(v61 + 40), (__int64)"__stack_chk_guard", 17, (__int64)v319);
    v367 = "StackGuard";
    LOWORD(v369) = 259;
    v278 = sub_1648A60(64, 1u);
    if ( v278 )
      sub_15F9210((__int64)v278, *(_QWORD *)(*(_QWORD *)v62 + 24LL), v62, 0, 0, 0);
    if ( v325 )
    {
      v63 = v326;
      sub_157E9D0(v325 + 40, (__int64)v278);
      v64 = *v63;
      v65 = v278[3];
      v278[4] = v63;
      v64 &= 0xFFFFFFFFFFFFFFF8LL;
      v278[3] = v64 | v65 & 7;
      *(_QWORD *)(v64 + 8) = v278 + 3;
      *v63 = *v63 & 7 | (unsigned __int64)(v278 + 3);
    }
    sub_164B780((__int64)v278, (__int64 *)&v367);
    if ( v324 )
    {
      v332 = v324;
      sub_1623A60((__int64)&v332, (__int64)v324, 2);
      v66 = v278[6];
      if ( v66 )
        sub_161E7C0((__int64)(v278 + 6), v66);
      v67 = v332;
      v278[6] = v332;
      if ( v67 )
        sub_1623210((__int64)&v332, v67, (__int64)(v278 + 6));
    }
    v68 = v319;
    LOWORD(v334) = 257;
    v69 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v325 + 56) + 40LL)) + 4);
    LOWORD(v369) = 257;
    v70 = sub_1648A60(64, 1u);
    v269 = v70;
    if ( v70 )
      sub_15F8BC0((__int64)v70, v68, v69, 0, (__int64)&v367, 0);
    if ( v325 )
    {
      v71 = v326;
      sub_157E9D0(v325 + 40, (__int64)v269);
      v72 = *v71;
      v73 = v269[3];
      v269[4] = v71;
      v72 &= 0xFFFFFFFFFFFFFFF8LL;
      v269[3] = v72 | v73 & 7;
      *(_QWORD *)(v72 + 8) = v269 + 3;
      *v71 = *v71 & 7 | (unsigned __int64)(v269 + 3);
    }
    sub_164B780((__int64)v269, (__int64 *)&v332);
    if ( v324 )
    {
      v301[0] = v324;
      sub_1623A60((__int64)v301, (__int64)v324, 2);
      v74 = v269[6];
      if ( v74 )
        sub_161E7C0((__int64)(v269 + 6), v74);
      v75 = v301[0];
      v269[6] = v301[0];
      if ( v75 )
        sub_1623210((__int64)v301, v75, (__int64)(v269 + 6));
    }
    LOWORD(v369) = 257;
    v76 = sub_1648A60(64, 2u);
    v77 = v76;
    if ( v76 )
      sub_15F9650((__int64)v76, (__int64)v278, (__int64)v269, 0, 0);
    if ( v325 )
    {
      v78 = v326;
      sub_157E9D0(v325 + 40, (__int64)v77);
      v79 = v77[3];
      v80 = *v78;
      v77[4] = v78;
      v80 &= 0xFFFFFFFFFFFFFFF8LL;
      v77[3] = v80 | v79 & 7;
      *(_QWORD *)(v80 + 8) = v77 + 3;
      *v78 = *v78 & 7 | (unsigned __int64)(v77 + 3);
    }
    sub_164B780((__int64)v77, (__int64 *)&v367);
    if ( v324 )
    {
      v332 = v324;
      sub_1623A60((__int64)&v332, (__int64)v324, 2);
      v83 = v77[6];
      if ( v83 )
        sub_161E7C0((__int64)(v77 + 6), v83);
      v84 = v332;
      v77[6] = v332;
      if ( v84 )
        sub_1623210((__int64)&v332, v84, (__int64)(v77 + 6));
    }
    v266 = &v309[(unsigned int)v310];
    if ( v309 == v266 )
      goto LABEL_149;
    v292 = v309;
    while ( 1 )
    {
      v95 = *v292;
      v96 = (_QWORD *)sub_16498A0(*v292);
      v332 = 0;
      v335 = v96;
      v336 = 0;
      v337 = 0;
      v338 = 0;
      v339 = 0;
      v333 = *(_QWORD *)(v95 + 40);
      v334 = (__int64 *)(v95 + 24);
      v97 = *(unsigned __int8 **)(v95 + 48);
      v367 = v97;
      if ( v97 )
      {
        sub_1623A60((__int64)&v367, (__int64)v97, 2);
        if ( v332 )
          sub_161E7C0((__int64)&v332, (__int64)v332);
        v332 = v367;
        if ( v367 )
          sub_1623210((__int64)&v367, v367, (__int64)&v332);
      }
      v286 = v315;
      LOWORD(v369) = 257;
      v98 = sub_1648A60(64, 1u);
      v99 = (__int64)v98;
      if ( v98 )
        sub_15F9210((__int64)v98, *(_QWORD *)(*v269 + 24LL), (__int64)v269, 0, 0, 0);
      if ( v333 )
      {
        v100 = v334;
        sub_157E9D0(v333 + 40, v99);
        v101 = *(_QWORD *)(v99 + 24);
        v102 = *v100;
        *(_QWORD *)(v99 + 32) = v100;
        v102 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v99 + 24) = v102 | v101 & 7;
        *(_QWORD *)(v102 + 8) = v99 + 24;
        *v100 = *v100 & 7 | (v99 + 24);
      }
      sub_164B780(v99, (__int64 *)&v367);
      if ( v332 )
      {
        v301[0] = v332;
        sub_1623A60((__int64)v301, (__int64)v332, 2);
        v103 = *(_QWORD *)(v99 + 48);
        if ( v103 )
          sub_161E7C0(v99 + 48, v103);
        v104 = v301[0];
        *(unsigned __int8 **)(v99 + 48) = v301[0];
        if ( v104 )
          sub_1623210((__int64)v301, v104, v99 + 48);
      }
      LOWORD(v302[0]) = 257;
      if ( *((_BYTE *)v278 + 16) <= 0x10u && *(_BYTE *)(v99 + 16) <= 0x10u )
        break;
      LOWORD(v369) = 257;
      v111 = sub_1648A60(56, 2u);
      v105 = v111;
      if ( v111 )
      {
        v275 = (__int64)v111;
        v112 = (_QWORD **)*v278;
        if ( *(_BYTE *)(*v278 + 8LL) == 16 )
        {
          v262 = v112[4];
          v113 = (__int64 *)sub_1643320(*v112);
          v114 = (__int64)sub_16463B0(v113, (unsigned int)v262);
        }
        else
        {
          v114 = sub_1643320(*v112);
        }
        sub_15FEC10((__int64)v105, v114, 51, 33, (__int64)v278, v99, (__int64)&v367, 0);
      }
      else
      {
        v275 = 0;
      }
      if ( v333 )
      {
        v115 = (unsigned __int64 *)v334;
        sub_157E9D0(v333 + 40, (__int64)v105);
        v116 = v105[3];
        v117 = *v115;
        v105[4] = v115;
        v117 &= 0xFFFFFFFFFFFFFFF8LL;
        v105[3] = v117 | v116 & 7;
        *(_QWORD *)(v117 + 8) = v105 + 3;
        *v115 = *v115 & 7 | (unsigned __int64)(v105 + 3);
      }
      sub_164B780(v275, (__int64 *)v301);
      if ( !v332 )
        goto LABEL_118;
      v300 = v332;
      sub_1623A60((__int64)&v300, (__int64)v332, 2);
      v118 = v105[6];
      if ( v118 )
        sub_161E7C0((__int64)(v105 + 6), v118);
      v119 = v300;
      v105[6] = v300;
      if ( !v119 )
        goto LABEL_118;
      sub_1623210((__int64)&v300, v119, (__int64)(v105 + 6));
      if ( !byte_4FCA2E8[0] )
      {
LABEL_144:
        if ( (unsigned int)sub_2207590(byte_4FCA2E8) )
        {
          sub_16AF710(dword_4FCA2F0, 0xFFFFFu, 0x100000u);
          sub_2207640(byte_4FCA2E8);
        }
      }
LABEL_119:
      v106 = dword_4FCA2F0[0];
      if ( !byte_4FCA2E8[0] && (unsigned int)sub_2207590(byte_4FCA2E8) )
      {
        sub_16AF710(dword_4FCA2F0, 0xFFFFFu, 0x100000u);
        sub_2207640(byte_4FCA2E8);
      }
      v274 = 0x80000000 - dword_4FCA2F0[0];
      v367 = (unsigned __int8 *)sub_15E0530(v286);
      v85 = sub_161BE60(&v367, v106, v274);
      v88 = sub_1AA92B0(
              (__int64)v105,
              v95,
              1,
              v85,
              0,
              0,
              (__m128)a3,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              a6,
              v86,
              v87,
              a9,
              a10);
      v89 = sub_16498A0((__int64)v88);
      v367 = 0;
      v370 = v89;
      v371 = 0;
      LODWORD(v372) = 0;
      v373 = 0;
      v374[0] = 0;
      v368 = v88[5];
      v369 = v88 + 3;
      v90 = (unsigned __int8 *)v88[6];
      v301[0] = v90;
      if ( v90 )
      {
        sub_1623A60((__int64)v301, (__int64)v90, 2);
        if ( v367 )
          sub_161E7C0((__int64)&v367, (__int64)v367);
        v367 = v301[0];
        if ( v301[0] )
          sub_1623210((__int64)v301, v301[0], (__int64)&v367);
      }
      v91 = *(_QWORD *)(v286 + 40);
      v92 = (__int64 *)sub_1643270(v335);
      v301[1] = 0;
      v301[0] = (unsigned __int8 *)v302;
      v93 = sub_1644EA0(v92, v302, 0, 0);
      v94 = sub_1632080(v91, (__int64)"__stack_chk_fail", 16, v93, 0);
      if ( (_QWORD *)v301[0] != v302 )
        _libc_free((unsigned __int64)v301[0]);
      LOWORD(v302[0]) = 257;
      sub_1285290((__int64 *)&v367, *(_QWORD *)(*(_QWORD *)v94 + 24LL), v94, 0, 0, (__int64)v301, 0);
      if ( v367 )
        sub_161E7C0((__int64)&v367, (__int64)v367);
      if ( v332 )
        sub_161E7C0((__int64)&v332, (__int64)v332);
      if ( v266 == ++v292 )
        goto LABEL_149;
    }
    v105 = (_QWORD *)sub_15A37B0(0x21u, v278, (_QWORD *)v99, 0);
LABEL_118:
    if ( !byte_4FCA2E8[0] )
      goto LABEL_144;
    goto LABEL_119;
  }
  v269 = 0;
LABEL_149:
  v120 = 0;
  v121 = sub_1EF16D0(
           &v315,
           (__int64)&v324,
           v315,
           v349,
           (unsigned int)v350,
           (__int64)v263,
           (__m128)a3,
           *(double *)a4.m128i_i64,
           *(double *)a5.m128i_i64,
           a6,
           v81,
           v82,
           a9,
           a10,
           v306,
           (unsigned int)v307,
           (__int64)v269);
  v124 = (unsigned int)v313;
  v125 = (unsigned int)v304;
  v280 = v121;
  v294 = (unsigned __int64)v312;
  if ( !(_DWORD)v313 )
    goto LABEL_183;
  if ( (_DWORD)v304 )
  {
    v212 = v319;
    v332 = "unsafe_stack_dynamic_ptr";
    LOWORD(v334) = 259;
    v213 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v325 + 56) + 40LL)) + 4);
    LOWORD(v369) = 257;
    v290 = v213;
    v214 = sub_1648A60(64, 1u);
    v120 = v214;
    if ( v214 )
      sub_15F8BC0((__int64)v214, v212, v290, 0, (__int64)&v367, 0);
    if ( v325 )
    {
      v215 = v326;
      sub_157E9D0(v325 + 40, (__int64)v120);
      v216 = v120[3];
      v217 = *v215;
      v120[4] = v215;
      v217 &= 0xFFFFFFFFFFFFFFF8LL;
      v120[3] = v217 | v216 & 7;
      *(_QWORD *)(v217 + 8) = v120 + 3;
      *v215 = *v215 & 7 | (unsigned __int64)(v120 + 3);
    }
    sub_164B780((__int64)v120, (__int64 *)&v332);
    if ( v324 )
    {
      v301[0] = v324;
      sub_1623A60((__int64)v301, (__int64)v324, 2);
      v218 = v120[6];
      if ( v218 )
        sub_161E7C0((__int64)(v120 + 6), v218);
      v219 = v301[0];
      v120[6] = v301[0];
      if ( v219 )
        sub_1623210((__int64)v301, v219, (__int64)(v120 + 6));
    }
    LOWORD(v369) = 257;
    v220 = sub_1648A60(64, 2u);
    v221 = v220;
    if ( v220 )
      sub_15F9650((__int64)v220, v280, (__int64)v120, 0, 0);
    if ( v325 )
    {
      v291 = v326;
      sub_157E9D0(v325 + 40, (__int64)v221);
      v222 = *v291;
      v223 = v221[3] & 7LL;
      v221[4] = v291;
      v222 &= 0xFFFFFFFFFFFFFFF8LL;
      v221[3] = v222 | v223;
      *(_QWORD *)(v222 + 8) = v221 + 3;
      *v291 = *v291 & 7 | (unsigned __int64)(v221 + 3);
    }
    sub_164B780((__int64)v221, (__int64 *)&v367);
    if ( v324 )
    {
      v332 = v324;
      sub_1623A60((__int64)&v332, (__int64)v324, 2);
      v224 = v221[6];
      v225 = (__int64)(v221 + 6);
      if ( v224 )
      {
        sub_161E7C0((__int64)(v221 + 6), v224);
        v225 = (__int64)(v221 + 6);
      }
      v226 = v332;
      v221[6] = v332;
      if ( v226 )
        sub_1623210((__int64)&v332, v226, v225);
    }
  }
  v126 = v294 + 8 * v124;
  v127 = v294;
  v288 = v126;
  do
  {
    v141 = *(_QWORD *)(*(_QWORD *)v127 + 32LL);
    if ( v141 == *(_QWORD *)(*(_QWORD *)v127 + 40LL) + 40LL || !v141 )
      BUG();
    v142 = *(_QWORD *)(v141 + 16);
    v143 = *(unsigned __int8 **)(v141 + 24);
    v326 = *(unsigned __int64 **)(*(_QWORD *)v127 + 32LL);
    v325 = v142;
    v367 = v143;
    if ( v143 )
    {
      sub_1623A60((__int64)&v367, (__int64)v143, 2);
      v144 = v324;
      if ( !v324 )
        goto LABEL_176;
    }
    else
    {
      v144 = v324;
      if ( !v324 )
        goto LABEL_178;
    }
    sub_161E7C0((__int64)&v324, (__int64)v144);
LABEL_176:
    v324 = v367;
    if ( v367 )
      sub_1623210((__int64)&v367, v367, (__int64)&v324);
LABEL_178:
    if ( v120 )
    {
      LOWORD(v369) = 257;
      v128 = sub_1648A60(64, 1u);
      if ( v128 )
        sub_15F9210((__int64)v128, *(_QWORD *)(*v120 + 24LL), (__int64)v120, 0, 0, 0);
      if ( v325 )
      {
        v129 = v326;
        sub_157E9D0(v325 + 40, (__int64)v128);
        v130 = v128[3];
        v131 = *v129;
        v128[4] = v129;
        v131 &= 0xFFFFFFFFFFFFFFF8LL;
        v128[3] = v131 | v130 & 7;
        *(_QWORD *)(v131 + 8) = v128 + 3;
        *v129 = *v129 & 7 | (unsigned __int64)(v128 + 3);
      }
      sub_164B780((__int64)v128, (__int64 *)&v367);
      if ( v324 )
      {
        v332 = v324;
        sub_1623A60((__int64)&v332, (__int64)v324, 2);
        v132 = v128[6];
        if ( v132 )
          sub_161E7C0((__int64)(v128 + 6), v132);
        v133 = v332;
        v128[6] = v332;
        if ( v133 )
          sub_1623210((__int64)&v332, v133, (__int64)(v128 + 6));
      }
    }
    else
    {
      v128 = (_QWORD *)v280;
    }
    LOWORD(v369) = 257;
    v295 = v323;
    v134 = sub_1648A60(64, 2u);
    v135 = v134;
    if ( v134 )
      sub_15F9650((__int64)v134, (__int64)v128, v295, 0, 0);
    if ( v325 )
    {
      v136 = v326;
      sub_157E9D0(v325 + 40, (__int64)v135);
      v137 = v135[3];
      v138 = *v136;
      v135[4] = v136;
      v138 &= 0xFFFFFFFFFFFFFFF8LL;
      v135[3] = v138 | v137 & 7;
      *(_QWORD *)(v138 + 8) = v135 + 3;
      *v136 = *v136 & 7 | (unsigned __int64)(v135 + 3);
    }
    sub_164B780((__int64)v135, (__int64 *)&v367);
    if ( v324 )
    {
      v332 = v324;
      sub_1623A60((__int64)&v332, (__int64)v324, 2);
      v139 = v135[6];
      if ( v139 )
        sub_161E7C0((__int64)(v135 + 6), v139);
      v140 = v332;
      v135[6] = v332;
      if ( v140 )
        sub_1623210((__int64)&v332, v140, (__int64)(v135 + 6));
    }
    v127 += 8LL;
  }
  while ( v288 != v127 );
  v125 = (unsigned int)v304;
LABEL_183:
  sub_1EEF920(
    &v315,
    v315,
    v323,
    (__int64)v120,
    v303,
    v125,
    (__m128)a3,
    *(double *)a4.m128i_i64,
    *(double *)a5.m128i_i64,
    a6,
    v122,
    v123,
    a9,
    a10);
  v145 = (unsigned __int64)v309;
  v289 = &v309[(unsigned int)v310];
  if ( v309 != v289 )
  {
    while ( 2 )
    {
      v154 = *(_QWORD *)v145;
      v155 = *(unsigned __int8 **)(*(_QWORD *)v145 + 48LL);
      v325 = *(_QWORD *)(*(_QWORD *)v145 + 40LL);
      v326 = (unsigned __int64 *)(v154 + 24);
      v367 = v155;
      if ( v155 )
      {
        sub_1623A60((__int64)&v367, (__int64)v155, 2);
        v146 = v324;
        if ( v324 )
          goto LABEL_186;
      }
      else
      {
        v146 = v324;
        if ( !v324 )
          goto LABEL_189;
LABEL_186:
        sub_161E7C0((__int64)&v324, (__int64)v146);
      }
      v324 = v367;
      if ( v367 )
        sub_1623210((__int64)&v367, v367, (__int64)&v324);
LABEL_189:
      v296 = v323;
      LOWORD(v369) = 257;
      v147 = sub_1648A60(64, 2u);
      v148 = v147;
      if ( v147 )
        sub_15F9650((__int64)v147, (__int64)v263, v296, 0, 0);
      if ( v325 )
      {
        v297 = v326;
        sub_157E9D0(v325 + 40, (__int64)v148);
        v149 = *v297;
        v150 = v148[3] & 7LL;
        v148[4] = v297;
        v149 &= 0xFFFFFFFFFFFFFFF8LL;
        v148[3] = v149 | v150;
        *(_QWORD *)(v149 + 8) = v148 + 3;
        *v297 = *v297 & 7 | (unsigned __int64)(v148 + 3);
      }
      sub_164B780((__int64)v148, (__int64 *)&v367);
      if ( v324 )
      {
        v332 = v324;
        sub_1623A60((__int64)&v332, (__int64)v324, 2);
        v151 = v148[6];
        v152 = (__int64)(v148 + 6);
        if ( v151 )
        {
          sub_161E7C0((__int64)(v148 + 6), v151);
          v152 = (__int64)(v148 + 6);
        }
        v153 = v332;
        v148[6] = v332;
        if ( v153 )
          sub_1623210((__int64)&v332, v153, v152);
      }
      v145 += 8LL;
      if ( v289 == (__int64 *)v145 )
        break;
      continue;
    }
  }
  if ( *(_BYTE *)(v323 + 16) != 78 )
    goto LABEL_203;
  v228 = sub_1560180(v315 + 112, 35);
  if ( v228 )
    goto LABEL_203;
  v229 = *(_BYTE *)(v323 + 16);
  if ( v229 <= 0x17u )
  {
    v299 = 0;
    v230 = 0;
  }
  else
  {
    if ( v229 == 78 )
    {
      v245 = v323 | 4;
      v299 = v323 | 4;
    }
    else
    {
      v299 = 0;
      v230 = 0;
      if ( v229 != 29 )
        goto LABEL_353;
      v245 = v323 & 0xFFFFFFFFFFFFFFFBLL;
      v299 = v323 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v246 = v245;
    v247 = v245 & 0xFFFFFFFFFFFFFFF8LL;
    v230 = v247;
    v248 = (v246 >> 2) & 1;
    if ( v248 )
    {
      v231 = (__int64 *)(v247 - 24);
      v228 = v248;
      goto LABEL_354;
    }
  }
LABEL_353:
  v231 = (__int64 *)(v230 - 72);
LABEL_354:
  if ( *(_BYTE *)(*v231 + 16) || sub_15E4F60(*v231) )
    goto LABEL_203;
  v232 = *v231;
  v233 = v299 & 0xFFFFFFFFFFFFFFF8LL;
  v234 = *(_BYTE *)(*v231 + 16) == 0;
  v235 = (_QWORD *)((v299 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( !v234 )
    v232 = 0;
  v236 = (_QWORD *)((v299 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( !v228 )
  {
    if ( !(unsigned __int8)sub_1560260(v236, -1, 3) )
    {
      v237 = *(_QWORD *)(v233 - 72);
      if ( *(_BYTE *)(v237 + 16) )
        goto LABEL_362;
LABEL_361:
      v367 = *(unsigned __int8 **)(v237 + 112);
      if ( !(unsigned __int8)sub_1560260(&v367, -1, 3) )
        goto LABEL_362;
    }
    goto LABEL_408;
  }
  if ( (unsigned __int8)sub_1560260(v236, -1, 3) )
  {
LABEL_408:
    if ( !(unsigned __int8)sub_3850BA0(v232) )
      goto LABEL_362;
    goto LABEL_383;
  }
  v237 = *(_QWORD *)(v233 - 24);
  if ( !*(_BYTE *)(v237 + 16) )
    goto LABEL_361;
LABEL_362:
  switch ( *(_BYTE *)(v232 + 32) & 0xF )
  {
    case 0:
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
      if ( (unsigned __int8)sub_1560180(v232 + 112, 26) )
        goto LABEL_203;
      if ( v228 )
      {
        if ( (unsigned __int8)sub_1560260(v235, -1, 26) )
          goto LABEL_203;
        v251 = *(_QWORD *)(v233 - 24);
        if ( *(_BYTE *)(v251 + 16) )
          break;
        goto LABEL_382;
      }
      if ( (unsigned __int8)sub_1560260(v235, -1, 26) )
        goto LABEL_203;
      v251 = *(_QWORD *)(v233 - 72);
      if ( !*(_BYTE *)(v251 + 16) )
      {
LABEL_382:
        v367 = *(unsigned __int8 **)(v251 + 112);
        if ( (unsigned __int8)sub_1560260(&v367, -1, 26) )
          goto LABEL_203;
        break;
      }
      break;
    case 2:
    case 4:
    case 9:
    case 0xA:
      goto LABEL_203;
    default:
      goto LABEL_425;
  }
LABEL_383:
  v373 = 0x400000000LL;
  v367 = 0;
  v368 = 0;
  v369 = 0;
  v370 = 0;
  v371 = 0;
  v372 = v374;
  v375 = v377;
  v376 = 0x800000000LL;
  v378 = v380;
  v379 = 0x800000000LL;
  sub_1ADC640(v299, (__int64)&v367, 0, 1, 0, (__m128)a3, a4, a5, a6, v249, v250, a9, a10);
  if ( v378 != v380 )
    _libc_free((unsigned __int64)v378);
  v252 = v375;
  v253 = &v375[24 * (unsigned int)v376];
  if ( v375 != (_BYTE *)v253 )
  {
    do
    {
      v254 = *(v253 - 1);
      v253 -= 3;
      if ( v254 != 0 && v254 != -8 && v254 != -16 )
        sub_1649B30(v253);
    }
    while ( v252 != v253 );
    v253 = v375;
  }
  if ( v253 != (_QWORD *)v377 )
    _libc_free((unsigned __int64)v253);
  if ( v372 != v374 )
    _libc_free((unsigned __int64)v372);
LABEL_203:
  v298 = 1;
  if ( v324 )
    sub_161E7C0((__int64)&v324, (__int64)v324);
LABEL_205:
  if ( v312 != v314 )
    _libc_free((unsigned __int64)v312);
  if ( v309 != (__int64 *)v311 )
    _libc_free((unsigned __int64)v309);
  if ( v306 != (__int64 ****)v308 )
    _libc_free((unsigned __int64)v306);
  if ( v303 != (__int64 *)v305 )
    _libc_free((unsigned __int64)v303);
  if ( v349 != (__int64 *)v351 )
    _libc_free((unsigned __int64)v349);
  sub_14602B0((__int64)v381);
  ++v352;
  if ( (_DWORD)v354 )
  {
    v198 = 4 * v354;
    v156 = v355;
    if ( (unsigned int)(4 * v354) < 0x40 )
      v198 = 64;
    if ( v198 >= v355 )
    {
LABEL_218:
      v157 = v353;
      v158 = &v353[2 * v156];
      if ( v353 != v158 )
      {
        do
        {
          *v157 = -8;
          v157 += 2;
        }
        while ( v158 != v157 );
      }
      v354 = 0;
      goto LABEL_221;
    }
    v199 = v353;
    if ( (_DWORD)v354 == 1 )
    {
      v205 = 2048;
      v204 = 128;
    }
    else
    {
      _BitScanReverse(&v200, v354 - 1);
      v201 = 1 << (33 - (v200 ^ 0x1F));
      if ( v201 < 64 )
        v201 = 64;
      if ( v355 == v201 )
      {
        v354 = 0;
        v256 = &v353[2 * v355];
        do
        {
          if ( v199 )
            *v199 = -8;
          v199 += 2;
        }
        while ( v256 != v199 );
        goto LABEL_221;
      }
      v202 = (4 * v201 / 3u + 1) | ((unsigned __int64)(4 * v201 / 3u + 1) >> 1);
      v203 = ((v202 | (v202 >> 2)) >> 4)
           | v202
           | (v202 >> 2)
           | ((((v202 | (v202 >> 2)) >> 4) | v202 | (v202 >> 2)) >> 8);
      v204 = (v203 | (v203 >> 16)) + 1;
      v205 = 16 * ((v203 | (v203 >> 16)) + 1);
    }
    j___libc_free_0(v353);
    v355 = v204;
    v206 = (_QWORD *)sub_22077B0(v205);
    v354 = 0;
    v353 = v206;
    for ( k = &v206[2 * v355]; k != v206; v206 += 2 )
    {
      if ( v206 )
        *v206 = -8;
    }
    goto LABEL_221;
  }
  if ( HIDWORD(v354) )
  {
    v156 = v355;
    if ( v355 <= 0x40 )
      goto LABEL_218;
    j___libc_free_0(v353);
    v353 = 0;
    v354 = 0;
    v355 = 0;
  }
LABEL_221:
  v159 = v357;
  v160 = v356;
  if ( v356 != v357 )
  {
    do
    {
      v161 = *v160;
      v162 = *(__int64 **)(*v160 + 8);
      v163 = *(__int64 **)(*v160 + 16);
      if ( v162 == v163 )
      {
        *(_BYTE *)(v161 + 160) = 1;
      }
      else
      {
        do
        {
          v164 = *v162++;
          sub_13FACC0(v164);
        }
        while ( v163 != v162 );
        *(_BYTE *)(v161 + 160) = 1;
        v165 = *(_QWORD *)(v161 + 8);
        if ( *(_QWORD *)(v161 + 16) != v165 )
          *(_QWORD *)(v161 + 16) = v165;
      }
      v166 = *(_QWORD *)(v161 + 32);
      if ( v166 != *(_QWORD *)(v161 + 40) )
        *(_QWORD *)(v161 + 40) = v166;
      ++*(_QWORD *)(v161 + 56);
      v167 = *(void **)(v161 + 72);
      if ( v167 == *(void **)(v161 + 64) )
      {
        *(_QWORD *)v161 = 0;
      }
      else
      {
        v168 = 4 * (*(_DWORD *)(v161 + 84) - *(_DWORD *)(v161 + 88));
        v169 = *(unsigned int *)(v161 + 80);
        if ( v168 < 0x20 )
          v168 = 32;
        if ( (unsigned int)v169 > v168 )
          sub_16CC920(v161 + 56);
        else
          memset(v167, -1, 8 * v169);
        v170 = *(_QWORD *)(v161 + 72);
        v171 = *(_QWORD *)(v161 + 64);
        *(_QWORD *)v161 = 0;
        if ( v171 != v170 )
          _libc_free(v170);
      }
      v172 = *(_QWORD *)(v161 + 32);
      if ( v172 )
        j_j___libc_free_0(v172, *(_QWORD *)(v161 + 48) - v172);
      v173 = *(_QWORD *)(v161 + 8);
      if ( v173 )
        j_j___libc_free_0(v173, *(_QWORD *)(v161 + 24) - v173);
      ++v160;
    }
    while ( v159 != v160 );
    if ( v356 != v357 )
      v357 = v356;
  }
  v174 = v364;
  v175 = &v364[2 * v365];
  if ( v364 != v175 )
  {
    do
    {
      v176 = *v174;
      v174 += 2;
      _libc_free(v176);
    }
    while ( v175 != v174 );
  }
  v365 = 0;
  if ( v362 )
  {
    v177 = v361;
    v366 = 0;
    v178 = &v361[v362];
    v179 = v361 + 1;
    v359 = *v361;
    v360 = v359 + 4096;
    if ( v178 != v361 + 1 )
    {
      do
      {
        v180 = *v179++;
        _libc_free(v180);
      }
      while ( v178 != v179 );
      v177 = v361;
    }
    v362 = 1;
    _libc_free(*v177);
    v181 = v364;
    v182 = &v364[2 * v365];
    if ( v364 != v182 )
    {
      do
      {
        v183 = *v181;
        v181 += 2;
        _libc_free(v183);
      }
      while ( v182 != v181 );
      goto LABEL_250;
    }
  }
  else
  {
LABEL_250:
    v182 = v364;
  }
  if ( v182 != (unsigned __int64 *)&v366 )
    _libc_free((unsigned __int64)v182);
  if ( v361 != (unsigned __int64 *)&v363 )
    _libc_free((unsigned __int64)v361);
  if ( v356 )
    j_j___libc_free_0(v356, v358 - (_QWORD)v356);
  j___libc_free_0(v353);
  if ( v345 )
  {
    v184 = v343;
    v185 = &v343[2 * v345];
    do
    {
      if ( *v184 != -8 && *v184 != -16 )
      {
        v186 = v184[1];
        if ( v186 )
        {
          v187 = *(_QWORD *)(v186 + 24);
          if ( v187 )
            j_j___libc_free_0(v187, *(_QWORD *)(v186 + 40) - v187);
          j_j___libc_free_0(v186, 56);
        }
      }
      v184 += 2;
    }
    while ( v185 != v184 );
  }
  j___libc_free_0(v343);
  if ( (char *)v340[0] != &v341 )
    _libc_free(v340[0]);
  return v298;
}
