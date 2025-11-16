// Function: sub_1B8BC40
// Address: 0x1b8bc40
//
__int64 __fastcall sub_1B8BC40(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdi
  int v32; // r15d
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // r14
  bool v39; // zf
  __int64 v40; // r10
  __int64 v41; // r12
  char v42; // al
  unsigned int v43; // ecx
  int v44; // esi
  __int64 v45; // r8
  __int64 v46; // rbx
  __int64 *v47; // rax
  const __m128i *v48; // rax
  unsigned __int64 *v49; // rax
  char *v50; // rax
  unsigned __int64 *v51; // rax
  unsigned __int64 *v52; // rax
  unsigned __int64 *v53; // rax
  __m128 *v54; // rax
  __int64 *v55; // rax
  __int64 *v56; // rax
  unsigned __int8 **v57; // rdi
  __int64 v58; // rsi
  __int64 v59; // rdx
  unsigned __int64 *v60; // rcx
  const __m128i *v61; // r8
  char *v62; // rbx
  __int64 v63; // rax
  __int64 v64; // rdi
  __m128 *v65; // rdx
  const __m128i *v66; // rax
  const __m128i *v67; // rcx
  const __m128i *v68; // r8
  unsigned __int64 v69; // rbx
  __int64 v70; // rax
  __m128 *v71; // rdi
  __m128 *v72; // rdx
  const __m128i *v73; // rax
  const __m128i *v74; // rcx
  const __m128i *v75; // rax
  __int64 v76; // rbx
  __int64 v77; // rbx
  __int64 v78; // r13
  __int64 v79; // r15
  char v80; // al
  __int64 v81; // rax
  int v82; // edx
  __int64 v83; // r14
  __int64 v84; // r13
  __int64 v85; // rdi
  unsigned int v86; // r10d
  __int64 v87; // rax
  unsigned int v88; // eax
  __int64 v89; // r13
  _QWORD *v90; // rax
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  int v95; // r9d
  __int64 *v96; // rdi
  __int64 *v97; // rax
  __int64 v98; // r13
  __int64 v99; // rdi
  unsigned int v100; // r10d
  __int64 v101; // rax
  unsigned int v102; // eax
  __int64 i; // r13
  _QWORD *v104; // rax
  __int64 v105; // rax
  int v106; // r8d
  int v107; // r9d
  __int64 v108; // r13
  __int64 v109; // rax
  __int64 v110; // rcx
  int v111; // r8d
  int v112; // r9d
  unsigned int v113; // r14d
  __int64 *v114; // kr08_8
  __int64 *v115; // r12
  __int64 *v116; // rbx
  unsigned __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // r13
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rdi
  __int64 v123; // rcx
  int v124; // r8d
  int v125; // r9d
  size_t v126; // r12
  unsigned __int8 **v127; // rax
  unsigned __int64 v128; // kr00_8
  __int64 *v129; // r15
  unsigned __int64 v130; // rbx
  __int64 v131; // rax
  __int64 v132; // r12
  __int64 v133; // rbx
  __int64 v134; // rax
  __int64 v135; // rsi
  __int64 v136; // rdi
  __int64 v137; // rbx
  unsigned __int64 v138; // rdi
  __int64 *v139; // rbx
  __int64 *v140; // r15
  unsigned __int64 v141; // rdi
  __int64 v142; // rbx
  __int64 v143; // r14
  unsigned __int64 v144; // r15
  __int64 v145; // r13
  unsigned __int64 v146; // rdi
  __int64 v147; // r13
  __int64 v148; // rbx
  unsigned __int64 v149; // r14
  __int64 v150; // r12
  unsigned __int64 v151; // rdi
  double v152; // xmm4_8
  double v153; // xmm5_8
  char v154; // bl
  double v155; // xmm4_8
  double v156; // xmm5_8
  char v157; // al
  __int64 v158; // r12
  __int64 v159; // rbx
  unsigned __int64 v160; // rdi
  __int64 v161; // rbx
  __int64 v162; // r12
  unsigned __int64 v163; // rdi
  __m128 *v164; // rdx
  __int64 v165; // rbx
  __int64 v166; // r12
  __int64 v167; // rdi
  unsigned int v169; // ecx
  __int64 v170; // rsi
  _QWORD *v171; // rax
  unsigned __int8 *v172; // rsi
  __int64 v173; // rax
  __int64 v174; // rax
  __int64 v175; // r8
  int v176; // r9d
  __int64 v177; // rax
  void *v178; // rdx
  void *v179; // rdx
  void *v180; // rdx
  unsigned __int64 *v181; // r12
  unsigned __int64 *v182; // r13
  unsigned __int64 *v183; // r12
  unsigned __int64 v184; // rdi
  unsigned __int64 *v185; // r12
  unsigned __int64 *v186; // r13
  unsigned __int64 *v187; // r12
  unsigned __int64 v188; // rdi
  unsigned __int64 v189; // rdi
  unsigned __int8 **v190; // rax
  __int64 v191; // rax
  __int64 v192; // rax
  _QWORD *v193; // rax
  _QWORD *v194; // rdi
  char v195; // al
  __int64 v196; // rax
  _QWORD *v197; // rdi
  char v198; // al
  _QWORD *v199; // rax
  __int64 v200; // rsi
  __int64 v201; // rax
  __int64 v202; // r8
  int v203; // r9d
  __int64 v204; // rax
  void *v205; // rdx
  double v206; // xmm4_8
  double v207; // xmm5_8
  int v208; // r8d
  int v209; // r9d
  __int64 v210; // rax
  __int64 v211; // rdx
  _QWORD *v212; // rax
  unsigned __int64 v213; // rax
  unsigned __int64 v214; // rax
  unsigned __int64 v215; // r12
  __int64 v216; // rax
  unsigned int v217; // eax
  unsigned __int64 v218; // r15
  unsigned int v219; // r14d
  __int64 v220; // rdi
  unsigned __int8 v221; // [rsp+27h] [rbp-529h]
  unsigned __int8 **v222; // [rsp+28h] [rbp-528h]
  __int64 v223; // [rsp+30h] [rbp-520h]
  unsigned __int8 **v224; // [rsp+38h] [rbp-518h]
  __int64 v225; // [rsp+38h] [rbp-518h]
  __int64 v226; // [rsp+38h] [rbp-518h]
  __int64 v227; // [rsp+40h] [rbp-510h]
  __int64 v228; // [rsp+40h] [rbp-510h]
  __int64 v229; // [rsp+48h] [rbp-508h]
  __int64 v230; // [rsp+48h] [rbp-508h]
  __int64 v231; // [rsp+48h] [rbp-508h]
  __int64 v232; // [rsp+50h] [rbp-500h]
  unsigned __int64 v233; // [rsp+58h] [rbp-4F8h]
  __int64 v234; // [rsp+58h] [rbp-4F8h]
  __int64 v235; // [rsp+58h] [rbp-4F8h]
  __int64 v236; // [rsp+58h] [rbp-4F8h]
  unsigned __int64 v237; // [rsp+58h] [rbp-4F8h]
  int v238; // [rsp+60h] [rbp-4F0h]
  __int64 v239; // [rsp+60h] [rbp-4F0h]
  __int64 v240; // [rsp+60h] [rbp-4F0h]
  int v241; // [rsp+60h] [rbp-4F0h]
  __int64 v242; // [rsp+68h] [rbp-4E8h]
  __int64 v243; // [rsp+68h] [rbp-4E8h]
  unsigned __int64 v244; // [rsp+68h] [rbp-4E8h]
  unsigned int v245; // [rsp+70h] [rbp-4E0h]
  unsigned int v246; // [rsp+70h] [rbp-4E0h]
  __int64 v247; // [rsp+70h] [rbp-4E0h]
  __int64 v248; // [rsp+78h] [rbp-4D8h]
  unsigned int v249; // [rsp+78h] [rbp-4D8h]
  unsigned int v250; // [rsp+78h] [rbp-4D8h]
  __int64 v251; // [rsp+78h] [rbp-4D8h]
  __int64 v252; // [rsp+88h] [rbp-4C8h] BYREF
  __int64 v253; // [rsp+90h] [rbp-4C0h] BYREF
  unsigned __int8 **v254; // [rsp+98h] [rbp-4B8h]
  __int64 *v255; // [rsp+A0h] [rbp-4B0h]
  unsigned int v256; // [rsp+A8h] [rbp-4A8h]
  __int64 v257; // [rsp+B0h] [rbp-4A0h]
  __int64 v258; // [rsp+B8h] [rbp-498h]
  unsigned __int64 j; // [rsp+C0h] [rbp-490h]
  __int64 v260; // [rsp+D0h] [rbp-480h] BYREF
  unsigned __int8 **v261; // [rsp+D8h] [rbp-478h]
  unsigned __int64 v262; // [rsp+E0h] [rbp-470h]
  int v263; // [rsp+E8h] [rbp-468h]
  __int64 v264; // [rsp+F0h] [rbp-460h]
  __int64 v265; // [rsp+F8h] [rbp-458h]
  unsigned __int64 k; // [rsp+100h] [rbp-450h]
  __int64 *v267[16]; // [rsp+110h] [rbp-440h] BYREF
  __int64 v268; // [rsp+190h] [rbp-3C0h] BYREF
  void *src; // [rsp+198h] [rbp-3B8h]
  __int64 *v270; // [rsp+1A0h] [rbp-3B0h]
  __int64 v271; // [rsp+1A8h] [rbp-3A8h]
  __int64 *v272; // [rsp+1B0h] [rbp-3A0h]
  __int64 v273; // [rsp+1B8h] [rbp-398h] BYREF
  __int64 v274; // [rsp+1C0h] [rbp-390h]
  const __m128i *v275; // [rsp+1F8h] [rbp-358h] BYREF
  unsigned __int64 *v276; // [rsp+200h] [rbp-350h]
  char *v277; // [rsp+208h] [rbp-348h]
  unsigned __int8 *v278; // [rsp+210h] [rbp-340h] BYREF
  __int64 v279; // [rsp+218h] [rbp-338h]
  unsigned __int64 v280; // [rsp+220h] [rbp-330h]
  _QWORD *v281; // [rsp+228h] [rbp-328h]
  __int64 v282; // [rsp+230h] [rbp-320h]
  int v283; // [rsp+238h] [rbp-318h] BYREF
  __int64 v284; // [rsp+240h] [rbp-310h]
  __int64 v285; // [rsp+248h] [rbp-308h]
  const __m128i *v286; // [rsp+278h] [rbp-2D8h]
  unsigned __int64 *v287; // [rsp+280h] [rbp-2D0h]
  char *v288; // [rsp+288h] [rbp-2C8h]
  unsigned __int8 *v289; // [rsp+290h] [rbp-2C0h] BYREF
  _QWORD **v290; // [rsp+298h] [rbp-2B8h]
  unsigned __int64 v291; // [rsp+2A0h] [rbp-2B0h]
  _QWORD *v292[2]; // [rsp+2A8h] [rbp-2A8h] BYREF
  _QWORD *v293; // [rsp+2B8h] [rbp-298h] BYREF
  __int64 v294; // [rsp+2C0h] [rbp-290h]
  _QWORD v295[4]; // [rsp+2C8h] [rbp-288h] BYREF
  __int64 v296; // [rsp+2E8h] [rbp-268h]
  __int64 v297; // [rsp+2F0h] [rbp-260h]
  __m128 *v298; // [rsp+2F8h] [rbp-258h]
  __int64 *v299; // [rsp+300h] [rbp-250h]
  __int64 *v300; // [rsp+308h] [rbp-248h]
  __m128i v301; // [rsp+310h] [rbp-240h] BYREF
  unsigned __int64 v302; // [rsp+320h] [rbp-230h] BYREF
  _BYTE v303[16]; // [rsp+328h] [rbp-228h] BYREF
  _BYTE *v304; // [rsp+338h] [rbp-218h] BYREF
  __int64 v305; // [rsp+340h] [rbp-210h]
  _BYTE v306[32]; // [rsp+348h] [rbp-208h] BYREF
  __int64 v307; // [rsp+368h] [rbp-1E8h]
  __int64 v308; // [rsp+370h] [rbp-1E0h]
  const __m128i *v309; // [rsp+378h] [rbp-1D8h]
  __m128i *v310; // [rsp+380h] [rbp-1D0h] BYREF
  unsigned __int64 *v311; // [rsp+388h] [rbp-1C8h]
  __int64 v312; // [rsp+390h] [rbp-1C0h] BYREF
  unsigned __int64 *v313; // [rsp+398h] [rbp-1B8h] BYREF
  unsigned __int64 *v314; // [rsp+3A0h] [rbp-1B0h]
  __int64 v315; // [rsp+3A8h] [rbp-1A8h]
  __int64 v316; // [rsp+3B0h] [rbp-1A0h]
  _QWORD v317[8]; // [rsp+3B8h] [rbp-198h] BYREF
  const __m128i *v318; // [rsp+3F8h] [rbp-158h]
  const __m128i *v319; // [rsp+400h] [rbp-150h]
  __int64 *v320; // [rsp+408h] [rbp-148h]
  _QWORD v321[4]; // [rsp+410h] [rbp-140h] BYREF
  __int64 *v322; // [rsp+430h] [rbp-120h]
  unsigned __int8 *v323; // [rsp+438h] [rbp-118h]
  __int64 v324[5]; // [rsp+440h] [rbp-110h] BYREF
  int v325; // [rsp+468h] [rbp-E8h]
  __int64 v326; // [rsp+470h] [rbp-E0h]
  __int64 v327; // [rsp+478h] [rbp-D8h]
  int v328; // [rsp+488h] [rbp-C8h]
  char v329; // [rsp+48Ch] [rbp-C4h]
  __int64 v330; // [rsp+490h] [rbp-C0h]
  __int64 v331; // [rsp+498h] [rbp-B8h]
  __int64 v332; // [rsp+4A0h] [rbp-B0h]
  int v333; // [rsp+4A8h] [rbp-A8h]
  int v334; // [rsp+4B8h] [rbp-98h] BYREF
  __int64 v335; // [rsp+4C0h] [rbp-90h]
  int *v336; // [rsp+4C8h] [rbp-88h]
  int *v337; // [rsp+4D0h] [rbp-80h]
  __int64 v338; // [rsp+4D8h] [rbp-78h]
  __int64 v339; // [rsp+4E0h] [rbp-70h]
  __int64 v340; // [rsp+4E8h] [rbp-68h]
  __int64 v341; // [rsp+4F0h] [rbp-60h]
  int v342; // [rsp+4F8h] [rbp-58h]
  __int64 v343; // [rsp+500h] [rbp-50h]
  __int64 v344; // [rsp+508h] [rbp-48h]
  __int64 v345; // [rsp+510h] [rbp-40h]
  __int64 v346; // [rsp+518h] [rbp-38h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
    goto LABEL_357;
  while ( *(_UNKNOWN **)v11 != &unk_4F96DB4 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_357;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F96DB4);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD *)(v14 + 160);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
    goto LABEL_357;
  while ( *(_UNKNOWN **)v17 != &unk_4F9E06C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_357;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9E06C);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19 + 160;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
    goto LABEL_357;
  while ( *(_UNKNOWN **)v22 != &unk_4F9A488 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_357;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9A488);
  v25 = *(__int64 **)(a1 + 8);
  v26 = *(_QWORD *)(v24 + 160);
  v27 = *v25;
  v28 = v25[1];
  if ( v27 == v28 )
LABEL_357:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F9D3C0 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_357;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F9D3C0);
  v30 = (__int64 *)sub_14A4050(v29, a2);
  v31 = *(_QWORD *)(a2 + 40);
  v32 = *(_DWORD *)(a1 + 156);
  v321[0] = a2;
  v321[1] = v16;
  v321[2] = v21;
  v321[3] = v26;
  v322 = v30;
  v323 = (unsigned __int8 *)sub_1632FA0(v31);
  v33 = sub_15E0530(*(_QWORD *)(v26 + 24));
  memset(v324, 0, 24);
  v324[3] = v33;
  v324[4] = 0;
  v325 = 0;
  v326 = 0;
  v327 = 0;
  v328 = v32;
  v329 = 0;
  v330 = 0;
  v331 = 0;
  v332 = 0;
  v333 = 0;
  v334 = 0;
  v335 = 0;
  v336 = &v334;
  v337 = &v334;
  v338 = 0;
  v339 = 0;
  v340 = 0;
  v341 = 0;
  v342 = 0;
  v343 = 0;
  v344 = 0;
  v345 = 0;
  v346 = 0;
  if ( byte_4FB7DA0 )
  {
    v34 = v321[0];
    v248 = v321[0] + 72LL;
    v35 = *(_QWORD *)(v321[0] + 80LL);
    if ( v35 != v321[0] + 72LL )
    {
      do
      {
        v36 = v35;
        v35 = *(_QWORD *)(v35 + 8);
        v37 = v36 + 16;
        v38 = *(_QWORD *)(v36 + 24);
        if ( v38 != v36 + 16 )
        {
          do
          {
            while ( 1 )
            {
              v39 = v38 == 0;
              v40 = v38 - 24;
              v38 = *(_QWORD *)(v38 + 8);
              if ( v39 )
                v40 = 0;
              v41 = v40;
              if ( !(unsigned __int8)sub_15F2ED0(v40) && !(unsigned __int8)sub_15F3040(v41) )
                goto LABEL_32;
              v42 = *(_BYTE *)(v41 + 16);
              if ( v42 == 54 )
                break;
              if ( v42 == 55
                && !sub_15F32D0(v41)
                && (v169 = *(unsigned __int16 *)(v41 + 18), (v169 & 1) == 0)
                && (unsigned int)(1 << (v169 >> 1)) >> 1
                && (unsigned __int8)sub_14A3910((__int64)v322)
                && (v170 = **(_QWORD **)(v41 - 48), (unsigned int)*(unsigned __int8 *)(v170 + 8) - 13 <= 1)
                && (unsigned __int8)sub_1B7D7E0(**(_QWORD **)(v41 - 48))
                && (unsigned int)sub_127FA20((__int64)v323, v170) > 7 )
              {
                v307 = 0;
                v301.m128i_i64[0] = (__int64)v323;
                v301.m128i_i64[1] = (__int64)v303;
                v302 = 0x400000000LL;
                v304 = v306;
                v305 = 0x400000000LL;
                v308 = 0;
                v309 = 0;
                v310 = 0;
                v311 = 0;
                v312 = 0;
                v313 = 0;
                v314 = 0;
                v315 = 0;
                v316 = 4;
                v317[0] = 2;
                v317[2] = 0;
                v171 = (_QWORD *)sub_16498A0(v41);
                v289 = 0;
                v292[0] = v171;
                v292[1] = 0;
                LODWORD(v293) = 0;
                v294 = 0;
                v295[0] = 0;
                v290 = *(_QWORD ***)(v41 + 40);
                v291 = v41 + 24;
                v172 = *(unsigned __int8 **)(v41 + 48);
                v278 = v172;
                if ( v172 )
                {
                  sub_1623A60((__int64)&v278, (__int64)v172, 2);
                  if ( v289 )
                    sub_161E7C0((__int64)&v289, (__int64)v289);
                  v289 = v278;
                  if ( v278 )
                    sub_1623210((__int64)&v278, v278, (__int64)&v289);
                }
                else
                {
                  v289 = 0;
                }
                v267[0] = *(__int64 **)(v41 - 48);
                v234 = *v267[0];
                v173 = *(_QWORD *)(v41 - 24);
                v308 = *v267[0];
                v307 = v173;
                v309 = (const __m128i *)(unsigned int)(1 << (*(unsigned __int16 *)(v41 + 18) >> 1) >> 1);
                v174 = sub_1643350(v292[0]);
                v175 = sub_159C470(v174, 0, 0);
                v177 = (unsigned int)v305;
                if ( (unsigned int)v305 >= HIDWORD(v305) )
                {
                  v231 = v175;
                  sub_16CD150((__int64)&v304, v306, 0, 8, v175, v176);
                  v177 = (unsigned int)v305;
                  v175 = v231;
                }
                *(_QWORD *)&v304[8 * v177] = v175;
                LODWORD(v305) = v305 + 1;
                v268 = (__int64)sub_1649960(v41);
                src = v178;
                LOWORD(v280) = 261;
                v278 = (unsigned __int8 *)&v268;
                sub_1B82200((__int64)&v301, (__int64)&v289, v234, v267, (__int64)&v278);
                v268 = (__int64)sub_1649960(v41);
                src = v179;
                LOWORD(v280) = 261;
                v278 = (unsigned __int8 *)&v268;
                sub_1B81290((__int64)&v301, (__int64 *)&v289, v267, (__int64)&v278, (unsigned __int64 **)&v310, 0);
                v268 = (__int64)sub_1649960(v41);
                src = v180;
                LOWORD(v280) = 261;
                v278 = (unsigned __int8 *)&v268;
                sub_1B81290((__int64)&v301, (__int64 *)&v289, v267, (__int64)&v278, &v313, 1);
                sub_15F20C0((_QWORD *)v41);
                LODWORD(v302) = 0;
                LODWORD(v305) = 0;
                if ( v289 )
                  sub_161E7C0((__int64)&v289, (__int64)v289);
                v181 = v313;
                if ( v314 != v313 )
                {
                  v235 = v35;
                  v182 = v313;
                  v183 = v314;
                  do
                  {
                    v184 = v182[4];
                    if ( (unsigned __int64 *)v184 != v182 + 6 )
                      _libc_free(v184);
                    if ( (unsigned __int64 *)*v182 != v182 + 2 )
                      _libc_free(*v182);
                    v182 += 11;
                  }
                  while ( v183 != v182 );
                  v35 = v235;
                  v181 = v313;
                }
                if ( v181 )
                  j_j___libc_free_0(v181, v315 - (_QWORD)v181);
                v185 = (unsigned __int64 *)v310;
                if ( v311 != (unsigned __int64 *)v310 )
                {
                  v236 = v35;
                  v186 = (unsigned __int64 *)v310;
                  v187 = v311;
                  do
                  {
                    v188 = v186[4];
                    if ( (unsigned __int64 *)v188 != v186 + 6 )
                      _libc_free(v188);
                    if ( (unsigned __int64 *)*v186 != v186 + 2 )
                      _libc_free(*v186);
                    v186 += 11;
                  }
                  while ( v187 != v186 );
                  v35 = v236;
                  v185 = (unsigned __int64 *)v310;
                }
                if ( v185 )
                  j_j___libc_free_0(v185, v312 - (_QWORD)v185);
                if ( v304 != v306 )
                  _libc_free((unsigned __int64)v304);
                v189 = v301.m128i_u64[1];
                if ( (_BYTE *)v301.m128i_i64[1] != v303 )
                  goto LABEL_322;
                if ( v37 == v38 )
                  goto LABEL_33;
              }
              else
              {
LABEL_32:
                if ( v37 == v38 )
                  goto LABEL_33;
              }
            }
            if ( sub_15F32D0(v41) )
              goto LABEL_32;
            v43 = *(unsigned __int16 *)(v41 + 18);
            if ( (v43 & 1) != 0 )
              goto LABEL_32;
            if ( !((unsigned int)(1 << (v43 >> 1)) >> 1) )
              goto LABEL_32;
            if ( !(unsigned __int8)sub_14A38E0((__int64)v322) )
              goto LABEL_32;
            v44 = *(unsigned __int8 *)(*(_QWORD *)v41 + 8LL);
            if ( (unsigned int)(v44 - 13) > 1 || !(unsigned __int8)sub_1B7D7E0(*(_QWORD *)v41) )
              goto LABEL_32;
            switch ( (char)v44 )
            {
              case 0:
                v210 = 8LL * *(_QWORD *)sub_15A9930((__int64)v323, v45);
                break;
              case 1:
                v232 = (__int64)v323;
                v230 = *(_QWORD *)(v45 + 24);
                v240 = *(_QWORD *)(v45 + 32);
                v237 = (unsigned int)sub_15A9FE0((__int64)v323, v230);
                v210 = 8 * v240 * v237 * ((v237 + ((unsigned __int64)(sub_127FA20(v232, v230) + 7) >> 3) - 1) / v237);
                break;
              default:
                goto LABEL_32;
            }
            if ( (unsigned int)v210 <= 7 )
              goto LABEL_32;
            v296 = 0;
            v289 = v323;
            v290 = v292;
            v291 = 0x400000000LL;
            v293 = v295;
            v294 = 0x400000000LL;
            v297 = 0;
            v298 = 0;
            v199 = (_QWORD *)sub_16498A0(v41);
            v278 = 0;
            v281 = v199;
            v282 = 0;
            v283 = 0;
            v284 = 0;
            v285 = 0;
            v279 = *(_QWORD *)(v41 + 40);
            v280 = v41 + 24;
            v200 = *(_QWORD *)(v41 + 48);
            v229 = v41 + 24;
            v301.m128i_i64[0] = v200;
            if ( v200 )
            {
              sub_1623A60((__int64)&v301, v200, 2);
              if ( v278 )
                sub_161E7C0((__int64)&v278, (__int64)v278);
              v278 = (unsigned __int8 *)v301.m128i_i64[0];
              if ( v301.m128i_i64[0] )
                sub_1623210((__int64)&v301, (unsigned __int8 *)v301.m128i_i64[0], (__int64)&v278);
            }
            v239 = *(_QWORD *)v41;
            v267[0] = (__int64 *)sub_1599EF0(*(__int64 ***)v41);
            v296 = *(_QWORD *)(v41 - 24);
            v297 = v239;
            v298 = (__m128 *)(unsigned int)(1 << (*(unsigned __int16 *)(v41 + 18) >> 1) >> 1);
            v201 = sub_1643350(v281);
            v202 = sub_159C470(v201, 0, 0);
            v204 = (unsigned int)v294;
            if ( (unsigned int)v294 >= HIDWORD(v294) )
            {
              v228 = v202;
              sub_16CD150((__int64)&v293, v295, 0, 8, v202, v203);
              v204 = (unsigned int)v294;
              v202 = v228;
            }
            v293[v204] = v202;
            LODWORD(v294) = v294 + 1;
            v227 = *(_QWORD *)(v41 + 24);
            v268 = (__int64)sub_1649960(v41);
            src = v205;
            LOWORD(v302) = 261;
            v301.m128i_i64[0] = (__int64)&v268;
            sub_1B7E8C0((__int64)&v289, (__int64)&v278, v239, v267, (__int64)&v301);
            sub_164D160(v41, (__int64)v267[0], (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v206, v207, a9, a10);
            if ( (unsigned __int8)sub_1C300D0(v41) && *(_BYTE *)(v239 + 8) == 14 )
            {
              v301.m128i_i64[0] = (__int64)&v302;
              v301.m128i_i64[1] = 0x1000000000LL;
              v213 = (v227 & 0xFFFFFFFFFFFFFFF8LL) - 24;
              if ( (v227 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                v213 = 0;
              v214 = v213 + 24;
              if ( v214 != v229 )
              {
                v225 = v41;
                v215 = v214;
                while ( 1 )
                {
                  if ( *(_BYTE *)(v215 - 8) == 54 )
                  {
                    v216 = v301.m128i_u32[2];
                    if ( v301.m128i_i32[2] >= (unsigned __int32)v301.m128i_i32[3] )
                    {
                      sub_16CD150((__int64)&v301, &v302, 0, 8, v208, v209);
                      v216 = v301.m128i_u32[2];
                    }
                    *(_QWORD *)(v301.m128i_i64[0] + 8 * v216) = v215 - 24;
                    ++v301.m128i_i32[2];
                  }
                  v215 = *(_QWORD *)(v215 + 8);
                  if ( v215 == v229 )
                    break;
                  if ( !v215 )
                    BUG();
                }
                v41 = v225;
              }
              v241 = (unsigned __int64)sub_127FA20((__int64)v289, **(_QWORD **)(v239 + 16)) >> 3;
              v217 = sub_1C30110(v41);
              if ( v301.m128i_i32[2] )
              {
                v226 = v37;
                v218 = 0;
                v223 = v38;
                v219 = v217;
                do
                {
                  v220 = *(_QWORD *)(v301.m128i_i64[0] + 8 * v218++);
                  sub_1C30170(v220, v219 & ~(-1 << v241));
                  v219 >>= v241;
                }
                while ( v301.m128i_u32[2] > v218 );
                v37 = v226;
                v38 = v223;
              }
              if ( (unsigned __int64 *)v301.m128i_i64[0] != &v302 )
                _libc_free(v301.m128i_u64[0]);
            }
            LODWORD(v291) = 0;
            LODWORD(v294) = 0;
            if ( v278 )
              sub_161E7C0((__int64)&v278, (__int64)v278);
            if ( v293 != v295 )
              _libc_free((unsigned __int64)v293);
            v189 = (unsigned __int64)v290;
            if ( v290 == v292 )
              goto LABEL_32;
LABEL_322:
            _libc_free(v189);
          }
          while ( v37 != v38 );
        }
LABEL_33:
        ;
      }
      while ( v248 != v35 );
      v34 = v321[0];
    }
  }
  else
  {
    v34 = v321[0];
  }
  memset(v267, 0, sizeof(v267));
  LODWORD(v267[3]) = 8;
  v267[1] = (__int64 *)&v267[5];
  v267[2] = (__int64 *)&v267[5];
  v46 = *(_QWORD *)(v34 + 80);
  v275 = 0;
  v276 = 0;
  v277 = 0;
  if ( v46 )
    v46 -= 24;
  src = &v273;
  v270 = &v273;
  v273 = v46;
  v271 = 0x100000008LL;
  LODWORD(v272) = 0;
  v268 = 1;
  v301.m128i_i64[1] = sub_157EBA0(v46);
  v301.m128i_i64[0] = v46;
  LODWORD(v302) = 0;
  sub_13FDF40(&v275, 0, &v301);
  sub_1B88860((__int64)&v268);
  sub_16CCEE0(&v289, (__int64)&v293, 8, (__int64)v267);
  v47 = v267[13];
  memset(&v267[13], 0, 24);
  v298 = (__m128 *)v47;
  v299 = v267[14];
  v300 = v267[15];
  sub_16CCEE0(&v278, (__int64)&v283, 8, (__int64)&v268);
  v48 = v275;
  v275 = 0;
  v286 = v48;
  v49 = v276;
  v276 = 0;
  v287 = v49;
  v50 = v277;
  v277 = 0;
  v288 = v50;
  sub_16CCEE0(&v301, (__int64)&v304, 8, (__int64)&v278);
  v51 = (unsigned __int64 *)v286;
  v286 = 0;
  v309 = (const __m128i *)v51;
  v52 = v287;
  v287 = 0;
  v310 = (__m128i *)v52;
  v53 = (unsigned __int64 *)v288;
  v288 = 0;
  v311 = v53;
  sub_16CCEE0(&v312, (__int64)v317, 8, (__int64)&v289);
  v54 = v298;
  v298 = 0;
  v318 = (const __m128i *)v54;
  v55 = v299;
  v299 = 0;
  v319 = (const __m128i *)v55;
  v56 = v300;
  v300 = 0;
  v320 = v56;
  if ( v286 )
    j_j___libc_free_0(v286, v288 - (char *)v286);
  if ( v280 != v279 )
    _libc_free(v280);
  if ( v298 )
    j_j___libc_free_0(v298, (char *)v300 - (char *)v298);
  if ( (_QWORD **)v291 != v290 )
    _libc_free(v291);
  if ( v275 )
    j_j___libc_free_0(v275, v277 - (char *)v275);
  if ( v270 != src )
    _libc_free((unsigned __int64)v270);
  if ( v267[13] )
    j_j___libc_free_0(v267[13], (char *)v267[15] - (char *)v267[13]);
  if ( v267[2] != v267[1] )
    _libc_free((unsigned __int64)v267[2]);
  v57 = &v278;
  v58 = (__int64)&v283;
  sub_16CCCB0(&v278, (__int64)&v283, (__int64)&v301);
  v60 = (unsigned __int64 *)v310;
  v61 = v309;
  v286 = 0;
  v287 = 0;
  v288 = 0;
  v62 = (char *)((char *)v310 - (char *)v309);
  if ( v310 == v309 )
  {
    v62 = 0;
    v64 = 0;
  }
  else
  {
    if ( (unsigned __int64)v62 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_348;
    v63 = sub_22077B0((char *)v310 - (char *)v309);
    v60 = (unsigned __int64 *)v310;
    v61 = v309;
    v64 = v63;
  }
  v286 = (const __m128i *)v64;
  v287 = (unsigned __int64 *)v64;
  v288 = &v62[v64];
  if ( v61 != (const __m128i *)v60 )
  {
    v65 = (__m128 *)v64;
    v66 = v61;
    do
    {
      if ( v65 )
      {
        a3 = _mm_loadu_si128(v66);
        *v65 = (__m128)a3;
        v65[1].m128_u64[0] = v66[1].m128i_u64[0];
      }
      v66 = (const __m128i *)((char *)v66 + 24);
      v65 = (__m128 *)((char *)v65 + 24);
    }
    while ( v66 != (const __m128i *)v60 );
    v64 += 8 * ((unsigned __int64)((char *)&v66[-2].m128i_u64[1] - (char *)v61) >> 3) + 24;
  }
  v287 = (unsigned __int64 *)v64;
  v58 = (__int64)&v293;
  v57 = &v289;
  sub_16CCCB0(&v289, (__int64)&v293, (__int64)&v312);
  v67 = v319;
  v68 = v318;
  v298 = 0;
  v299 = 0;
  v300 = 0;
  v69 = (char *)v319 - (char *)v318;
  if ( v319 != v318 )
  {
    if ( v69 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v70 = sub_22077B0((char *)v319 - (char *)v318);
      v67 = v319;
      v68 = v318;
      v71 = (__m128 *)v70;
      goto LABEL_65;
    }
LABEL_348:
    sub_4261EA(v57, v58, v59);
  }
  v69 = 0;
  v71 = 0;
LABEL_65:
  v298 = v71;
  v72 = v71;
  v299 = (__int64 *)v71;
  v300 = (__int64 *)((char *)v71 + v69);
  if ( v68 != v67 )
  {
    v73 = v68;
    do
    {
      if ( v72 )
      {
        a4 = _mm_loadu_si128(v73);
        *v72 = (__m128)a4;
        v72[1].m128_u64[0] = v73[1].m128i_u64[0];
      }
      v73 = (const __m128i *)((char *)v73 + 24);
      v72 = (__m128 *)((char *)v72 + 24);
    }
    while ( v73 != v67 );
    v72 = (__m128 *)((char *)v71 + 8 * ((unsigned __int64)((char *)&v73[-2].m128i_u64[1] - (char *)v68) >> 3) + 24);
  }
  v74 = (const __m128i *)v287;
  v75 = v286;
  v299 = (__int64 *)v72;
  v221 = 0;
  v58 = (char *)v287 - (char *)v286;
  if ( (char *)v287 - (char *)v286 == (char *)v72 - (char *)v71 )
    goto LABEL_203;
  while ( 1 )
  {
    do
    {
      v76 = v74[-2].m128i_i64[1];
      j = 0;
      k = 0;
      memset(v267, 0, 28);
      v77 = v76 + 40;
      memset(&v267[4], 0, 24);
      v268 = 0;
      src = 0;
      v270 = 0;
      LODWORD(v271) = 0;
      v272 = 0;
      v273 = 0;
      v274 = 0;
      v78 = *(_QWORD *)(v77 + 8);
      v253 = 0;
      v254 = 0;
      v79 = v78;
      v255 = 0;
      v256 = 0;
      v257 = 0;
      v258 = 0;
      v260 = 0;
      v261 = 0;
      v262 = 0;
      v263 = 0;
      v264 = 0;
      v265 = 0;
      if ( v78 != v77 )
      {
        while ( 1 )
        {
          v83 = v79 - 24;
          if ( !v79 )
            v83 = 0;
          if ( (unsigned __int8)sub_15F2ED0(v83) )
          {
            v80 = *(_BYTE *)(v83 + 16);
            if ( v80 != 54 )
              goto LABEL_75;
LABEL_87:
            if ( sub_15F32D0(v83) || (*(_BYTE *)(v83 + 18) & 1) != 0 )
              goto LABEL_81;
LABEL_89:
            if ( *(_BYTE *)(v83 + 16) == 54 )
            {
              v58 = v83;
              if ( !(unsigned __int8)sub_14A38E0((__int64)v322) )
                goto LABEL_81;
            }
            v84 = *(_QWORD *)v83;
            v85 = *(_QWORD *)v83;
            if ( *(_BYTE *)(*(_QWORD *)v83 + 8LL) == 16 )
              v85 = **(_QWORD **)(v84 + 16);
            if ( !(unsigned __int8)sub_1643F10(v85) )
              goto LABEL_81;
            v58 = v84;
            v245 = sub_127FA20((__int64)v323, v84);
            if ( (v245 & 7) != 0 || *(_BYTE *)(v84 + 8) == 16 && *(_BYTE *)(**(_QWORD **)(v84 + 16) + 8LL) == 15 )
              goto LABEL_81;
            v242 = sub_1B7CA20(v83);
            v87 = *(_QWORD *)v242;
            if ( *(_BYTE *)(*(_QWORD *)v242 + 8LL) == 16 )
              v87 = **(_QWORD **)(v87 + 16);
            v249 = v86;
            v58 = *(_DWORD *)(v87 + 8) >> 8;
            v88 = sub_14A38B0((__int64)v322);
            if ( *(_BYTE *)(v84 + 8) == 16 )
            {
              if ( v249 <= v88 >> 1 )
              {
                v58 = v88 / v245;
                if ( (unsigned int)sub_14A39A0(v322, v58, v249, v249 >> 3, v84) )
                {
                  if ( *(_BYTE *)(v84 + 8) == 16 )
                  {
                    v89 = *(_QWORD *)(v83 + 8);
                    if ( v89 )
                    {
                      while ( 1 )
                      {
                        v90 = sub_1648700(v89);
                        if ( *((_BYTE *)v90 + 16) != 83 || *(_BYTE *)(*(v90 - 3) + 16LL) != 13 )
                          goto LABEL_81;
                        v89 = *(_QWORD *)(v89 + 8);
                        if ( !v89 )
                          goto LABEL_106;
                      }
                    }
                  }
                  goto LABEL_106;
                }
              }
              goto LABEL_81;
            }
            if ( v88 >> 1 < v249 )
              goto LABEL_81;
LABEL_106:
            v91 = sub_14AD280(v242, (unsigned __int64)v323, 8u);
            if ( *(_BYTE *)(v91 + 16) == 79 )
              v91 = *(_QWORD *)(v91 - 72);
            v252 = v91;
            v96 = (__int64 *)v267;
            v58 = (__int64)&v252;
LABEL_135:
            v108 = sub_1B8B920((__int64)v96, &v252, v92, v93, v94, v95);
            v109 = *(unsigned int *)(v108 + 8);
            if ( (unsigned int)v109 >= *(_DWORD *)(v108 + 12) )
            {
              v58 = v108 + 16;
              sub_16CD150(v108, (const void *)(v108 + 16), 0, 8, v106, v107);
              v109 = *(unsigned int *)(v108 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v108 + 8 * v109) = v83;
            ++*(_DWORD *)(v108 + 8);
            v79 = *(_QWORD *)(v79 + 8);
            if ( v77 == v79 )
              break;
          }
          else
          {
            if ( !(unsigned __int8)sub_15F3040(v83) )
              goto LABEL_81;
            v80 = *(_BYTE *)(v83 + 16);
            if ( v80 == 54 )
              goto LABEL_87;
LABEL_75:
            if ( v80 == 78 )
            {
              v81 = *(_QWORD *)(v83 - 24);
              if ( *(_BYTE *)(v81 + 16) )
                goto LABEL_81;
              v82 = *(_DWORD *)(v81 + 36);
              if ( v82 == 4057 || v82 == 4085 )
              {
                v192 = *(_QWORD *)(v83 - 24LL * (*(_DWORD *)(v83 + 20) & 0xFFFFFFF));
                v58 = *(unsigned int *)(v192 + 32);
                v193 = *(_QWORD **)(v192 + 24);
                if ( v82 == 4085 )
                {
                  if ( (unsigned int)v58 > 0x40 )
                    v193 = (_QWORD *)*v193;
                  v195 = (((unsigned __int8)v193 >> 4) ^ 1) & 1;
                }
                else
                {
                  v194 = v193;
                  if ( (unsigned int)v58 > 0x40 )
                    v194 = (_QWORD *)*v193;
                  v195 = (((unsigned __int8)((unsigned __int16)sub_1C278B0(v194) >> 8) >> 1) ^ 1) & 1;
                }
                if ( !v195 )
                  goto LABEL_81;
                goto LABEL_89;
              }
              if ( v82 != 4492 && v82 != 4503 )
                goto LABEL_81;
              if ( v82 == 4503 )
              {
                v211 = *(_QWORD *)(v83 - 24LL * (*(_DWORD *)(v83 + 20) & 0xFFFFFFF));
                v212 = *(_QWORD **)(v211 + 24);
                if ( *(_DWORD *)(v211 + 32) > 0x40u )
                  v212 = (_QWORD *)*v212;
                v198 = (((unsigned __int8)v212 >> 4) ^ 1) & 1;
              }
              else
              {
                v196 = *(_QWORD *)(v83 - 24LL * (*(_DWORD *)(v83 + 20) & 0xFFFFFFF));
                v197 = *(_QWORD **)(v196 + 24);
                if ( *(_DWORD *)(v196 + 32) > 0x40u )
                  v197 = (_QWORD *)*v197;
                v198 = (((unsigned __int8)((unsigned __int16)sub_1C278B0(v197) >> 8) >> 1) ^ 1) & 1;
              }
              if ( !v198 )
                goto LABEL_81;
            }
            else if ( v80 != 55 || sub_15F32D0(v83) || (*(_BYTE *)(v83 + 18) & 1) != 0 )
            {
              goto LABEL_81;
            }
            if ( *(_BYTE *)(v83 + 16) != 55 )
              goto LABEL_278;
            v58 = v83;
            if ( !(unsigned __int8)sub_14A3910((__int64)v322) )
              goto LABEL_81;
            if ( *(_BYTE *)(v83 + 16) == 55 )
            {
              v97 = *(__int64 **)(v83 - 48);
            }
            else
            {
LABEL_278:
              if ( (*(_BYTE *)(v83 + 23) & 0x40) != 0 )
                v191 = *(_QWORD *)(v83 - 8);
              else
                v191 = v83 - 24LL * (*(_DWORD *)(v83 + 20) & 0xFFFFFFF);
              v97 = *(__int64 **)(v191 + 24);
            }
            v98 = *v97;
            v99 = *v97;
            if ( *(_BYTE *)(*v97 + 8) == 16 )
              v99 = **(_QWORD **)(v98 + 16);
            if ( !(unsigned __int8)sub_1643F10(v99)
              || *(_BYTE *)(v98 + 8) == 16 && *(_BYTE *)(**(_QWORD **)(v98 + 16) + 8LL) == 15 )
            {
              goto LABEL_81;
            }
            v58 = v98;
            v246 = sub_127FA20((__int64)v323, v98);
            if ( (v246 & 7) != 0 )
              goto LABEL_81;
            v243 = sub_1B7CA20(v83);
            v101 = *(_QWORD *)v243;
            if ( *(_BYTE *)(*(_QWORD *)v243 + 8LL) == 16 )
              v101 = **(_QWORD **)(v101 + 16);
            v250 = v100;
            v58 = *(_DWORD *)(v101 + 8) >> 8;
            v102 = sub_14A38B0((__int64)v322);
            if ( *(_BYTE *)(v98 + 8) == 16 )
            {
              if ( v250 <= v102 >> 1 )
              {
                v58 = v102 / v246;
                if ( (unsigned int)sub_14A3A00(v322, v58, v250, v250 >> 3, v98) )
                {
                  if ( *(_BYTE *)(v98 + 8) == 16 )
                  {
                    for ( i = *(_QWORD *)(v83 + 8); i; i = *(_QWORD *)(i + 8) )
                    {
                      v104 = sub_1648700(i);
                      if ( *((_BYTE *)v104 + 16) != 83 || *(_BYTE *)(*(v104 - 3) + 16LL) != 13 )
                        goto LABEL_81;
                    }
                  }
LABEL_132:
                  v105 = sub_14AD280(v243, (unsigned __int64)v323, 8u);
                  if ( *(_BYTE *)(v105 + 16) == 79 )
                    v105 = *(_QWORD *)(v105 - 72);
                  v252 = v105;
                  v96 = &v268;
                  v58 = (__int64)&v252;
                  goto LABEL_135;
                }
              }
            }
            else if ( v250 <= v102 >> 1 )
            {
              goto LABEL_132;
            }
LABEL_81:
            v79 = *(_QWORD *)(v79 + 8);
            if ( v77 == v79 )
              break;
          }
        }
      }
      v57 = 0;
      j___libc_free_0(0);
      v113 = (unsigned int)v267[3];
      if ( LODWORD(v267[3]) )
      {
        v190 = (unsigned __int8 **)sub_22077B0(16LL * LODWORD(v267[3]));
        v58 = (__int64)v267[1];
        v57 = v190;
        v224 = v190;
        v114 = v267[2];
        memcpy(v190, v267[1], 16LL * v113);
      }
      else
      {
        v224 = 0;
        v114 = 0;
      }
      v115 = v267[5];
      v116 = v267[4];
      v117 = (char *)v267[5] - (char *)v267[4];
      if ( v267[5] == v267[4] )
      {
        v251 = 0;
      }
      else
      {
        if ( v117 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_348;
        v118 = sub_22077B0((char *)v267[5] - (char *)v267[4]);
        v115 = v267[5];
        v116 = v267[4];
        v251 = v118;
      }
      v244 = v251 + v117;
      if ( v116 == v115 )
      {
        v119 = v251;
      }
      else
      {
        v119 = v251;
        do
        {
          while ( 1 )
          {
            if ( v119 )
            {
              v120 = *v116;
              *(_DWORD *)(v119 + 16) = 0;
              *(_DWORD *)(v119 + 20) = 8;
              *(_QWORD *)v119 = v120;
              *(_QWORD *)(v119 + 8) = v119 + 24;
              v121 = *((unsigned int *)v116 + 4);
              if ( (_DWORD)v121 )
                break;
            }
            v116 += 11;
            v119 += 88;
            if ( v115 == v116 )
              goto LABEL_149;
          }
          v58 = (__int64)(v116 + 1);
          v122 = v119 + 8;
          v116 += 11;
          v119 += 88;
          sub_1B7CED0(v122, v58, v121, v110, v111, v112);
        }
        while ( v115 != v116 );
      }
LABEL_149:
      v57 = 0;
      j___libc_free_0(0);
      v238 = v271;
      if ( (_DWORD)v271 )
      {
        v126 = 16LL * (unsigned int)v271;
        v127 = (unsigned __int8 **)sub_22077B0(v126);
        v58 = (__int64)src;
        v57 = v127;
        v222 = v127;
        v128 = (unsigned __int64)v270;
        memcpy(v127, src, v126);
      }
      else
      {
        v222 = 0;
        v128 = 0;
      }
      v59 = v273;
      v129 = v272;
      v130 = v273 - (_QWORD)v272;
      if ( (__int64 *)v273 == v272 )
      {
        v247 = 0;
      }
      else
      {
        if ( v130 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_348;
        v131 = sub_22077B0(v273 - (_QWORD)v272);
        v59 = v273;
        v129 = v272;
        v247 = v131;
      }
      v132 = v247;
      v233 = v247 + v130;
      if ( (__int64 *)v59 == v129 )
      {
        v132 = v247;
      }
      else
      {
        v133 = v59;
        do
        {
          while ( 1 )
          {
            if ( v132 )
            {
              v134 = *v129;
              *(_DWORD *)(v132 + 16) = 0;
              *(_DWORD *)(v132 + 20) = 8;
              *(_QWORD *)v132 = v134;
              *(_QWORD *)(v132 + 8) = v132 + 24;
              if ( *((_DWORD *)v129 + 4) )
                break;
            }
            v129 += 11;
            v132 += 88;
            if ( (__int64 *)v133 == v129 )
              goto LABEL_160;
          }
          v135 = (__int64)(v129 + 1);
          v136 = v132 + 8;
          v129 += 11;
          v132 += 88;
          sub_1B7CED0(v136, v135, v59, v123, v124, v125);
        }
        while ( (__int64 *)v133 != v129 );
LABEL_160:
        v137 = v273;
        v129 = v272;
        if ( (__int64 *)v273 != v272 )
        {
          do
          {
            v138 = v129[1];
            if ( (__int64 *)v138 != v129 + 3 )
              _libc_free(v138);
            v129 += 11;
          }
          while ( v129 != (__int64 *)v137 );
          v129 = v272;
        }
      }
      if ( v129 )
        j_j___libc_free_0(v129, v274 - (_QWORD)v129);
      j___libc_free_0(src);
      v139 = v267[5];
      v140 = v267[4];
      if ( v267[5] != v267[4] )
      {
        do
        {
          v141 = v140[1];
          if ( (__int64 *)v141 != v140 + 3 )
            _libc_free(v141);
          v140 += 11;
        }
        while ( v139 != v140 );
        v140 = v267[4];
      }
      if ( v140 )
        j_j___libc_free_0(v140, (char *)v267[6] - (char *)v140);
      j___libc_free_0(v267[1]);
      j___libc_free_0(v254);
      v256 = v113;
      v142 = v258;
      v143 = v257;
      v258 = v119;
      v254 = v224;
      v144 = j;
      ++v253;
      v145 = v257;
      v255 = v114;
      v257 = v251;
      for ( j = v244; v142 != v145; v145 += 88 )
      {
        v146 = *(_QWORD *)(v145 + 8);
        if ( v146 != v145 + 24 )
          _libc_free(v146);
      }
      if ( v143 )
        j_j___libc_free_0(v143, v144 - v143);
      j___libc_free_0(v261);
      v147 = v264;
      v148 = v265;
      v149 = k;
      v265 = v132;
      v261 = v222;
      v150 = v264;
      ++v260;
      v262 = v128;
      v263 = v238;
      v264 = v247;
      for ( k = v233; v148 != v150; v150 += 88 )
      {
        v151 = *(_QWORD *)(v150 + 8);
        if ( v151 != v150 + 24 )
          _libc_free(v151);
      }
      if ( v147 )
        j_j___libc_free_0(v147, v149 - v147);
      j___libc_free_0(0);
      j___libc_free_0(0);
      v154 = sub_1B89510((__int64)v321, (__int64)&v253, a3, a4, a5, a6, v152, v153, a9, a10);
      v157 = sub_1B89510((__int64)v321, (__int64)&v260, a3, a4, a5, a6, v155, v156, a9, a10);
      v158 = v264;
      v221 |= v157 | v154;
      v159 = v265;
      if ( v265 != v264 )
      {
        do
        {
          v160 = *(_QWORD *)(v158 + 8);
          if ( v160 != v158 + 24 )
            _libc_free(v160);
          v158 += 88;
        }
        while ( v159 != v158 );
        v158 = v264;
      }
      if ( v158 )
        j_j___libc_free_0(v158, k - v158);
      j___libc_free_0(v261);
      v161 = v258;
      v162 = v257;
      if ( v258 != v257 )
      {
        do
        {
          v163 = *(_QWORD *)(v162 + 8);
          if ( v163 != v162 + 24 )
            _libc_free(v163);
          v162 += 88;
        }
        while ( v161 != v162 );
        v162 = v257;
      }
      if ( v162 )
        j_j___libc_free_0(v162, j - v162);
      j___libc_free_0(v254);
      v75 = v286;
      v287 -= 3;
      v74 = v286;
      if ( v287 != (unsigned __int64 *)v286 )
      {
        sub_1B88860((__int64)&v278);
        v75 = v286;
        v74 = (const __m128i *)v287;
      }
      v71 = v298;
      v58 = (char *)v74 - (char *)v75;
    }
    while ( (char *)v74 - (char *)v75 != (char *)v299 - (char *)v298 );
LABEL_203:
    if ( v75 == v74 )
      break;
    v164 = v71;
    while ( 1 )
    {
      v58 = v164->m128_u64[0];
      if ( v75->m128i_i64[0] != v164->m128_u64[0] )
        break;
      v58 = v164[1].m128_u32[0];
      if ( v75[1].m128i_i32[0] != (_DWORD)v58 )
        break;
      v75 = (const __m128i *)((char *)v75 + 24);
      v164 = (__m128 *)((char *)v164 + 24);
      if ( v75 == v74 )
        goto LABEL_208;
    }
  }
LABEL_208:
  if ( v71 )
    j_j___libc_free_0(v71, (char *)v300 - (char *)v71);
  if ( (_QWORD **)v291 != v290 )
    _libc_free(v291);
  if ( v286 )
    j_j___libc_free_0(v286, v288 - (char *)v286);
  if ( v280 != v279 )
    _libc_free(v280);
  if ( v318 )
    j_j___libc_free_0(v318, (char *)v320 - (char *)v318);
  if ( v314 != v313 )
    _libc_free((unsigned __int64)v314);
  if ( v309 )
    j_j___libc_free_0(v309, (char *)v311 - (char *)v309);
  if ( v302 != v301.m128i_i64[1] )
    _libc_free(v302);
  j___libc_free_0(v344);
  j___libc_free_0(v340);
  v165 = v335;
  while ( v165 )
  {
    v166 = v165;
    sub_1B7DEF0(*(_QWORD *)(v165 + 24));
    v165 = *(_QWORD *)(v165 + 16);
    if ( *(_DWORD *)(v166 + 48) > 0x40u )
    {
      v167 = *(_QWORD *)(v166 + 40);
      if ( v167 )
        j_j___libc_free_0_0(v167);
    }
    j_j___libc_free_0(v166, 64);
  }
  j___libc_free_0(v331);
  if ( v324[0] )
    sub_161E7C0((__int64)v324, v324[0]);
  return v221;
}
