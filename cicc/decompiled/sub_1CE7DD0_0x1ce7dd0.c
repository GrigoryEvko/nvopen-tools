// Function: sub_1CE7DD0
// Address: 0x1ce7dd0
//
__int64 __fastcall sub_1CE7DD0(
        __int64 *a1,
        __int64 a2,
        __m128 si128,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r12
  int v13; // r14d
  _QWORD *v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rdx
  char *v18; // rsi
  const char **v19; // r13
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  double v40; // xmm4_8
  double v41; // xmm5_8
  unsigned int *v42; // rdi
  unsigned int v43; // r13d
  unsigned int v44; // eax
  __int64 v45; // rdx
  int v46; // eax
  __int64 v47; // rdi
  __int64 v48; // rsi
  __int64 v49; // rax
  _QWORD *v50; // rcx
  unsigned int v51; // edx
  _QWORD *v52; // rax
  int v53; // ebx
  __int64 v54; // r15
  __int64 v55; // r12
  int v56; // r12d
  unsigned __int64 v57; // rdx
  const char *v58; // r9
  size_t v59; // r8
  size_t v60; // rax
  _QWORD *v61; // rdx
  const char *v62; // rdx
  __int64 v63; // r12
  void *v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // rbx
  __int64 v67; // r12
  unsigned int v68; // ecx
  _DWORD *v69; // rdx
  __int64 v70; // r8
  __int64 v71; // rax
  int v72; // esi
  _DWORD *v73; // rdi
  int v74; // ecx
  const char **v75; // rsi
  void *v76; // r12
  __int64 v77; // rax
  __int64 v78; // r14
  __int64 v79; // r12
  __int64 v80; // r13
  int v81; // r8d
  int v82; // r9d
  __int64 v83; // r15
  unsigned int v84; // ecx
  __int64 v85; // rax
  unsigned int v86; // ecx
  __int64 v87; // rax
  _QWORD *v88; // rax
  int v89; // r12d
  char *v90; // r14
  char *v91; // r15
  const char **v92; // rdx
  __int64 v93; // r14
  unsigned int v94; // esi
  __int64 v95; // rax
  __int64 v97; // rdi
  __int64 v98; // rax
  int v99; // ecx
  _QWORD *v100; // rbx
  int v101; // esi
  __int64 v102; // rax
  int v103; // ecx
  __int64 v104; // r8
  __int64 v105; // rax
  int v106; // r15d
  char v107; // al
  _QWORD *v108; // r13
  unsigned __int64 v109; // r12
  __int64 v110; // r15
  __int64 v111; // rcx
  __int64 *v112; // r14
  __int64 v113; // rdi
  __int64 *v114; // rdx
  __int64 v115; // r8
  char v116; // al
  __int64 v117; // rax
  unsigned int v118; // edx
  __int64 *v119; // rax
  __int64 v120; // rax
  bool v121; // zf
  int v122; // edx
  __int64 v123; // rax
  __int64 *v124; // rax
  __int64 v125; // rax
  double v126; // xmm4_8
  double v127; // xmm5_8
  __int64 v128; // r8
  int v129; // ecx
  int v130; // ecx
  __int64 v131; // rsi
  unsigned int v132; // edx
  const char **v133; // rax
  const char *v134; // r10
  const char *v135; // rax
  int v136; // ebx
  __int64 *v137; // rax
  unsigned int v138; // edi
  unsigned int v139; // eax
  __int64 v140; // rdx
  _QWORD *v141; // rcx
  __int64 v142; // rsi
  __int64 v143; // rcx
  _QWORD *v144; // r12
  _QWORD *v145; // r13
  int v146; // eax
  char *v147; // rdi
  __int64 v148; // rcx
  int v149; // r11d
  char *v150; // r10
  int v151; // edx
  unsigned int v152; // eax
  int v154; // r12d
  char *v155; // rax
  _BYTE *v156; // rsi
  unsigned int v157; // eax
  unsigned int v158; // r10d
  unsigned int v159; // esi
  int v160; // ecx
  unsigned __int64 v161; // r8
  __int64 v162; // rdx
  unsigned __int64 v163; // r8
  __int64 *v166; // rbx
  __int64 v167; // rdi
  unsigned int v168; // edx
  const char **v169; // rax
  const char *v170; // r8
  const char *v171; // rax
  const char *v172; // r12
  int v173; // esi
  int v174; // r8d
  int v175; // r12d
  const char **v176; // rcx
  __int64 v177; // rdx
  _QWORD *v178; // r11
  unsigned int v179; // edx
  _QWORD *v180; // rdi
  __int64 v181; // r8
  __int64 v182; // rax
  int v183; // esi
  int v184; // ecx
  _QWORD *v185; // rax
  __int64 v186; // rax
  int v187; // esi
  unsigned int v188; // edx
  _QWORD *v189; // rcx
  __int64 v190; // rdi
  int v191; // r11d
  _QWORD *v192; // r10
  int v193; // edx
  _QWORD *v194; // rdi
  __int64 v195; // r10
  __int64 v196; // rdx
  signed int v197; // r14d
  __int64 v198; // rbx
  unsigned int v199; // esi
  __int64 v200; // rcx
  __int64 v201; // rdx
  __int64 v202; // rax
  __int64 v203; // r9
  signed int v204; // eax
  __int64 v205; // rax
  int v206; // r15d
  char *v207; // rax
  __int64 v208; // rdx
  const char *v209; // rdx
  __int64 v210; // rdi
  int v211; // edx
  __int64 *v212; // rax
  const char **v213; // r12
  const char *v214; // rdi
  const char **v215; // r12
  const char *v216; // rdi
  unsigned int v217; // ebx
  unsigned int *v218; // rdi
  __int64 v219; // rax
  __int64 v220; // rax
  __int64 v221; // rcx
  int v222; // r11d
  char *v223; // r9
  int v224; // edi
  int v225; // eax
  char *v226; // rsi
  __int64 v227; // r15
  __int64 v228; // rdi
  __int64 v229; // r15
  unsigned __int64 v230; // r14
  __int64 v231; // r12
  __int64 v232; // rax
  unsigned __int64 v233; // r12
  void *v234; // rax
  unsigned int v235; // edx
  unsigned int v236; // ecx
  int v237; // r11d
  int v238; // r11d
  const char **v239; // rcx
  int v240; // r8d
  const char **v241; // rax
  const char **v242; // r12
  const char *v243; // r14
  void *v244; // r12
  __int64 v245; // rdi
  __m128 *v246; // rax
  __int64 v247; // rax
  _WORD *v248; // rdx
  __int64 v249; // rdi
  __int64 v250; // rax
  _WORD *v251; // rdx
  const char **v252; // rax
  __int64 v253; // rdx
  unsigned __int64 v254; // r12
  __int64 v255; // r13
  const char *v256; // rax
  size_t v257; // rdx
  _BYTE *v258; // rdi
  char *v259; // rsi
  _BYTE *v260; // rax
  size_t v261; // r15
  __int64 v262; // r15
  __int64 v263; // r14
  _BYTE *v264; // rax
  void *v265; // rax
  _QWORD *v266; // rdi
  const char **v267; // r12
  const char **v268; // rbx
  const char *v269; // rdi
  const char **v270; // r12
  const char **v271; // rbx
  _QWORD *v272; // rax
  _QWORD *v273; // r12
  _QWORD *v274; // r14
  __int64 v275; // rcx
  unsigned int v276; // edx
  _QWORD *v277; // rax
  unsigned int v278; // ebx
  __int64 v279; // r15
  int v280; // eax
  int v281; // eax
  void *v282; // rax
  __int64 v283; // rax
  __int64 v284; // rax
  unsigned int v285; // eax
  unsigned int v286; // r13d
  __int64 v287; // rdi
  unsigned int v288; // r14d
  __int64 v289; // rax
  int v290; // eax
  unsigned int v291; // r12d
  char *v292; // r15
  __int64 *v293; // r13
  __int64 v294; // rax
  _QWORD *v295; // rdx
  unsigned int v296; // eax
  _QWORD *v297; // rsi
  int v298; // edx
  _QWORD *v299; // rbx
  int v300; // eax
  unsigned int v301; // eax
  unsigned int v302; // r8d
  unsigned int v303; // eax
  __int64 *v304; // rax
  __int64 v305; // rdx
  __int64 *v306; // rax
  __int64 v307; // rsi
  _QWORD *v308; // rdx
  __int64 *v309; // r14
  __int64 v310; // rax
  __int64 v311; // rbx
  __int64 v312; // r10
  unsigned int v313; // edx
  __int64 v314; // rcx
  __int64 *v315; // rax
  __int64 v316; // rax
  __int64 *v317; // rcx
  __int64 v318; // rax
  int v319; // r14d
  __int64 v320; // rdx
  int v321; // eax
  int v322; // r9d
  _QWORD *v323; // rax
  __int64 v324; // rdx
  _QWORD *v325; // r14
  _QWORD *v326; // r15
  __int64 v327; // rcx
  unsigned int v328; // edx
  _QWORD *v329; // rax
  unsigned int v330; // r13d
  __int64 v331; // rbx
  __int64 v332; // r12
  __int64 v333; // rax
  __int64 v334; // rax
  int v335; // [rsp+10h] [rbp-2E0h]
  int v336; // [rsp+18h] [rbp-2D8h]
  unsigned int v337; // [rsp+1Ch] [rbp-2D4h]
  size_t v338; // [rsp+20h] [rbp-2D0h]
  unsigned int v339; // [rsp+38h] [rbp-2B8h]
  unsigned int v340; // [rsp+40h] [rbp-2B0h]
  signed int v341; // [rsp+44h] [rbp-2ACh]
  __int64 v342; // [rsp+48h] [rbp-2A8h]
  __int64 *v343; // [rsp+50h] [rbp-2A0h]
  const char **v345; // [rsp+68h] [rbp-288h]
  int v346; // [rsp+70h] [rbp-280h]
  const char **v347; // [rsp+78h] [rbp-278h]
  unsigned __int8 v348; // [rsp+78h] [rbp-278h]
  unsigned int v349; // [rsp+78h] [rbp-278h]
  unsigned int v350; // [rsp+78h] [rbp-278h]
  size_t v351; // [rsp+80h] [rbp-270h]
  bool v352; // [rsp+80h] [rbp-270h]
  unsigned int v353; // [rsp+80h] [rbp-270h]
  unsigned int v354; // [rsp+80h] [rbp-270h]
  size_t v355; // [rsp+80h] [rbp-270h]
  size_t v356; // [rsp+80h] [rbp-270h]
  unsigned int *src; // [rsp+88h] [rbp-268h]
  unsigned int srca; // [rsp+88h] [rbp-268h]
  unsigned int srcf; // [rsp+88h] [rbp-268h]
  __int64 *srcb; // [rsp+88h] [rbp-268h]
  unsigned int srcc; // [rsp+88h] [rbp-268h]
  const char *srcg; // [rsp+88h] [rbp-268h]
  char *srcd; // [rsp+88h] [rbp-268h]
  void *srce; // [rsp+88h] [rbp-268h]
  int v366; // [rsp+90h] [rbp-260h]
  char v367; // [rsp+90h] [rbp-260h]
  _QWORD *v368; // [rsp+90h] [rbp-260h]
  int v369; // [rsp+90h] [rbp-260h]
  __int64 *v370; // [rsp+90h] [rbp-260h]
  _BYTE *v371; // [rsp+98h] [rbp-258h]
  __int64 v372; // [rsp+98h] [rbp-258h]
  _QWORD *v373; // [rsp+98h] [rbp-258h]
  __int64 *v374; // [rsp+98h] [rbp-258h]
  unsigned int v375; // [rsp+98h] [rbp-258h]
  unsigned int v376; // [rsp+98h] [rbp-258h]
  int v377; // [rsp+A4h] [rbp-24Ch] BYREF
  __int64 v378; // [rsp+A8h] [rbp-248h] BYREF
  const char **v379; // [rsp+B0h] [rbp-240h] BYREF
  const char **v380; // [rsp+B8h] [rbp-238h]
  const char **v381; // [rsp+C0h] [rbp-230h]
  void *dest; // [rsp+D0h] [rbp-220h] BYREF
  unsigned __int64 v383; // [rsp+D8h] [rbp-218h]
  unsigned int v384; // [rsp+E0h] [rbp-210h]
  _BYTE *v385; // [rsp+F0h] [rbp-200h] BYREF
  _BYTE *v386; // [rsp+F8h] [rbp-1F8h]
  _BYTE *v387; // [rsp+100h] [rbp-1F0h]
  char **v388; // [rsp+110h] [rbp-1E0h] BYREF
  char **v389; // [rsp+118h] [rbp-1D8h]
  __int64 v390; // [rsp+120h] [rbp-1D0h]
  __int64 v391; // [rsp+130h] [rbp-1C0h] BYREF
  __int64 v392; // [rsp+138h] [rbp-1B8h]
  __int64 v393; // [rsp+140h] [rbp-1B0h]
  unsigned int v394; // [rsp+148h] [rbp-1A8h]
  __int64 v395; // [rsp+150h] [rbp-1A0h] BYREF
  char *v396; // [rsp+158h] [rbp-198h]
  __int64 v397; // [rsp+160h] [rbp-190h]
  __int64 v398; // [rsp+168h] [rbp-188h]
  __int64 v399; // [rsp+170h] [rbp-180h] BYREF
  __int64 v400; // [rsp+178h] [rbp-178h]
  __int64 v401; // [rsp+180h] [rbp-170h]
  __int64 v402; // [rsp+188h] [rbp-168h]
  __int64 v403; // [rsp+190h] [rbp-160h] BYREF
  __int64 v404; // [rsp+198h] [rbp-158h]
  __int64 v405; // [rsp+1A0h] [rbp-150h]
  __int64 v406; // [rsp+1A8h] [rbp-148h]
  __int64 v407; // [rsp+1B0h] [rbp-140h] BYREF
  __int64 v408; // [rsp+1B8h] [rbp-138h]
  __int64 v409; // [rsp+1C0h] [rbp-130h]
  __int64 v410; // [rsp+1C8h] [rbp-128h]
  __int64 v411; // [rsp+1D0h] [rbp-120h] BYREF
  const char **v412; // [rsp+1D8h] [rbp-118h]
  __int64 v413; // [rsp+1E0h] [rbp-110h]
  unsigned int v414; // [rsp+1E8h] [rbp-108h]
  __int64 v415; // [rsp+1F0h] [rbp-100h] BYREF
  const char **v416; // [rsp+1F8h] [rbp-F8h]
  __int64 v417; // [rsp+200h] [rbp-F0h]
  unsigned int v418; // [rsp+208h] [rbp-E8h]
  char *v419; // [rsp+210h] [rbp-E0h] BYREF
  __int64 v420; // [rsp+218h] [rbp-D8h]
  __int64 v421; // [rsp+220h] [rbp-D0h]
  __int64 v422; // [rsp+228h] [rbp-C8h]
  const char *v423; // [rsp+230h] [rbp-C0h] BYREF
  const char **v424; // [rsp+238h] [rbp-B8h]
  __int64 v425; // [rsp+240h] [rbp-B0h]
  unsigned int v426; // [rsp+248h] [rbp-A8h]
  const char *v427; // [rsp+250h] [rbp-A0h] BYREF
  __int64 v428; // [rsp+258h] [rbp-98h]
  _QWORD v429[2]; // [rsp+260h] [rbp-90h] BYREF
  void *s1; // [rsp+270h] [rbp-80h] BYREF
  size_t n; // [rsp+278h] [rbp-78h]
  _QWORD v432[4]; // [rsp+280h] [rbp-70h] BYREF
  _QWORD *v433; // [rsp+2A0h] [rbp-50h]
  _QWORD *v434; // [rsp+2A8h] [rbp-48h]
  __int64 v435; // [rsp+2B0h] [rbp-40h]
  _QWORD *v436; // [rsp+2B8h] [rbp-38h]

  v10 = a1;
  if ( qword_4FC04E0 != qword_4FC04E8 )
  {
    v11 = qword_4FC04E0;
    v12 = 0;
    v13 = 0;
    do
    {
      v15 = 32 * v12;
      v16 = v15 + v11;
      v18 = (char *)sub_1649960(a2);
      s1 = v432;
      if ( v18 )
      {
        sub_1CD0120((__int64 *)&s1, v18, (__int64)&v18[v17]);
        v14 = s1;
        if ( *(_QWORD *)(v16 + 8) == n )
        {
          if ( !n || (v14 = s1, !memcmp(s1, *(const void **)v16, n)) )
          {
            if ( v14 != v432 )
              j_j___libc_free_0(v14, v432[0] + 1LL);
LABEL_14:
            LODWORD(v19) = 0;
            if ( dword_4FC0240 )
            {
              v282 = sub_16E8CB0();
              v283 = sub_1263B40((__int64)v282, "\tSkip rematerialization on ");
              v284 = sub_16E7EE0(v283, *(char **)(qword_4FC04E0 + v15), *(_QWORD *)(qword_4FC04E0 + v15 + 8));
              sub_1263B40(v284, "\n");
            }
            return (unsigned int)v19;
          }
        }
        if ( v14 != v432 )
          j_j___libc_free_0(v14, v432[0] + 1LL);
      }
      else
      {
        n = 0;
        LOBYTE(v432[0]) = 0;
        if ( !*(_QWORD *)(v16 + 8) )
          goto LABEL_14;
      }
      v11 = qword_4FC04E0;
      v12 = (unsigned int)++v13;
    }
    while ( v13 != (qword_4FC04E8 - qword_4FC04E0) >> 5 );
    v10 = a1;
  }
  v21 = (__int64 *)v10[1];
  v22 = *v21;
  v23 = v21[1];
  if ( v22 == v23 )
LABEL_658:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4FB9E2C )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_658;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4FB9E2C);
  v25 = (__int64 *)v10[1];
  v10[21] = v24 + 156;
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_659:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_4F9920C )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_659;
  }
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(*(_QWORD *)(v26 + 8), &unk_4F9920C);
  v29 = (__int64 *)v10[1];
  v10[24] = v28 + 160;
  v30 = *v29;
  v31 = v29[1];
  if ( v30 == v31 )
LABEL_663:
    BUG();
  while ( *(_UNKNOWN **)v30 != &unk_4F99CCC )
  {
    v30 += 16;
    if ( v31 == v30 )
      goto LABEL_663;
  }
  v32 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v30 + 8) + 104LL))(*(_QWORD *)(v30 + 8), &unk_4F99CCC);
  v33 = (__int64 *)v10[1];
  v10[22] = v32 + 160;
  v34 = *v33;
  v35 = v33[1];
  if ( v34 == v35 )
LABEL_664:
    BUG();
  while ( *(_UNKNOWN **)v34 != &unk_4F9E06C )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_664;
  }
  v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(*(_QWORD *)(v34 + 8), &unk_4F9E06C);
  v37 = (__int64 *)v10[1];
  v10[23] = v36 + 160;
  v38 = *v37;
  v39 = v37[1];
  if ( v38 == v39 )
LABEL_662:
    BUG();
  while ( *(_UNKNOWN **)v38 != &unk_4FB9E34 )
  {
    v38 += 16;
    if ( v39 == v38 )
      goto LABEL_662;
  }
  v10[25] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(
              *(_QWORD *)(v38 + 8),
              &unk_4FB9E34)
          + 160;
  if ( !(dword_4FBFA60 | dword_4FBFB40 | dword_4FC05C0) )
  {
    LODWORD(v19) = 0;
    goto LABEL_135;
  }
  sub_1CDA600((__int64)v10, a2);
  sub_1C007A0(v10[25], a2, 1, 1, 1, 0);
  if ( dword_4FC0240 )
  {
    v318 = v10[25];
    srce = (void *)v318;
    if ( !*(_DWORD *)(v318 + 96) )
      goto LABEL_583;
    v323 = *(_QWORD **)(v318 + 88);
    v324 = 2LL * *(unsigned int *)(v10[25] + 104);
    v325 = &v323[v324];
    if ( v323 == &v323[v324] )
      goto LABEL_583;
    while ( 1 )
    {
      v326 = v323;
      if ( *v323 != -16 && *v323 != -8 )
        break;
      v323 += 2;
      if ( v325 == v323 )
        goto LABEL_583;
    }
    if ( v325 == v323 )
    {
LABEL_583:
      v376 = 0;
    }
    else
    {
      v376 = 0;
      v370 = v10;
      do
      {
        v327 = *(_QWORD *)(v326[1] + 8LL);
        v328 = (unsigned int)(*(_DWORD *)(v327 + 16) + 63) >> 6;
        if ( v328 )
        {
          v329 = *(_QWORD **)v327;
          v330 = 0;
          v331 = *(_QWORD *)v327 + 8LL;
          v332 = v331 + 8LL * (v328 - 1);
          while ( 1 )
          {
            v330 += sub_39FAC40(*v329);
            v329 = (_QWORD *)v331;
            if ( v332 == v331 )
              break;
            v331 += 8;
          }
          if ( v376 >= v330 )
            v330 = v376;
          v376 = v330;
        }
        v326 += 2;
        if ( v326 == v325 )
          break;
        while ( *v326 == -8 || *v326 == -16 )
        {
          v326 += 2;
          if ( v325 == v326 )
            goto LABEL_626;
        }
      }
      while ( v325 != v326 );
LABEL_626:
      v10 = v370;
    }
    v319 = sub_1CD1FD0((__int64)srce);
    v427 = sub_1649960(a2);
    v428 = v320;
    sub_12C70A0((__int64 *)&s1, (__int64)&v427);
    fprintf(stderr, "\nFunc %s: maxIn = %d, maxOut = %d\n", (const char *)s1, v319, v376);
    sub_2240A30(&s1);
  }
  v42 = (unsigned int *)v10[21];
  v43 = dword_4FBF600;
  v44 = v42[1];
  if ( dword_4FBF600 && dword_4FBF600 < v44 )
    goto LABEL_555;
  if ( v44 )
  {
    v43 = v42[1];
    goto LABEL_555;
  }
  v339 = dword_4FBFD00;
  if ( dword_4FBF8A0 )
  {
    LODWORD(v423) = 0;
    if ( dword_4FBF600 )
    {
      v285 = sub_1BFBA30(v42, a2, &v423);
      v286 = dword_4FBF600;
      if ( v285 > dword_4FBF600 )
      {
LABEL_525:
        if ( dword_4FBF8A0 >= v286 )
          goto LABEL_44;
        v287 = v10[25];
        s1 = v432;
        n = 0x200000000LL;
        v288 = sub_1C01730(v287, a2, (__int64)&s1);
        v289 = sub_1632FA0(*(_QWORD *)(a2 + 40));
        v290 = sub_15A9520(v289, 0);
        v291 = n;
        v369 = v290;
        if ( (_DWORD)n )
        {
          v350 = v286;
          v291 = 0;
          v292 = 0;
          srcd = (char *)(8LL * (unsigned int)n);
          v293 = v10;
          do
          {
            v294 = sub_1BFDF20(v293[25], *(_QWORD *)&v292[(_QWORD)s1]);
            v295 = *(_QWORD **)v294;
            v296 = (unsigned int)(*(_DWORD *)(*(_QWORD *)v294 + 16LL) + 63) >> 6;
            if ( v296 )
            {
              v297 = (_QWORD *)*v295;
              v298 = 0;
              v299 = v297 + 1;
              v356 = (size_t)&v297[v296];
              while ( 1 )
              {
                v346 = v298;
                v300 = sub_39FAC40(*v297);
                v297 = v299;
                v298 = v300 + v346;
                if ( (_QWORD *)v356 == v299 )
                  break;
                ++v299;
              }
              v301 = v288 - v298;
            }
            else
            {
              v301 = v288;
            }
            if ( v291 < v301 )
              v291 = v301;
            v292 += 8;
          }
          while ( srcd != v292 );
          v10 = v293;
          v286 = v350;
        }
        if ( v369 > 4 )
          v288 = (3 * v288) >> 1;
        LODWORD(v427) = 0;
        if ( !dword_4FBF600
          || (v302 = sub_1BFBB30((_DWORD *)v10[21], a2, v288, (unsigned int *)&v427),
              v303 = dword_4FBF600,
              v302 <= dword_4FBF600) )
        {
          v303 = sub_1BFBB30((_DWORD *)v10[21], a2, v288, (unsigned int *)&v427);
        }
        if ( !v303 )
          v303 = v286;
        if ( (unsigned int)v423 >= v303 )
          v303 = (unsigned int)v423;
        v43 = v303;
        if ( v369 > 4 )
          v43 = 2 * v303 / 3;
        if ( s1 != v432 )
          _libc_free((unsigned __int64)s1);
        if ( !v43 )
          goto LABEL_44;
        if ( v291 && v43 > v291 )
        {
          v339 = v43 - v291;
          goto LABEL_44;
        }
LABEL_555:
        v339 = 8 * v43 / 0xA;
        goto LABEL_44;
      }
      v42 = (unsigned int *)v10[21];
    }
    v286 = sub_1BFBA30(v42, a2, &v423);
    goto LABEL_525;
  }
LABEL_44:
  if ( !dword_4FC05C0 )
  {
    LODWORD(v19) = 0;
LABEL_132:
    if ( dword_4FBFB40 )
      goto LABEL_495;
LABEL_133:
    if ( dword_4FBFA60 )
      goto LABEL_498;
    goto LABEL_134;
  }
  v343 = v10;
  v335 = 5;
  src = (unsigned int *)v10[25];
  v345 = (const char **)(a2 + 72);
  do
  {
    v391 = 0;
    v392 = 0;
    v336 = dword_4FC0080;
    v393 = 0;
    v45 = v343[21];
    v394 = 0;
    v337 = *src;
    v46 = dword_4FBF600;
    if ( dword_4FBF600 )
    {
      if ( (unsigned int)dword_4FBF600 >= *(_DWORD *)(v45 + 4) )
        v46 = *(_DWORD *)(v45 + 4);
      v366 = v46;
    }
    else
    {
      v366 = *(_DWORD *)(v45 + 4);
    }
    v47 = (__int64)src;
    v19 = *(const char ***)(a2 + 80);
    if ( v19 == v345 )
    {
LABEL_591:
      v341 = 0;
      goto LABEL_74;
    }
    while ( 1 )
    {
      v48 = (__int64)(v19 - 3);
      if ( !v19 )
        v48 = 0;
      v423 = (const char *)v48;
      v49 = sub_1BFDF20(v47, v48);
      v50 = *(_QWORD **)v49;
      v51 = (unsigned int)(*(_DWORD *)(*(_QWORD *)v49 + 16LL) + 63) >> 6;
      if ( v51 )
      {
        v52 = (_QWORD *)*v50;
        v53 = 0;
        v54 = *v50 + 8LL;
        v55 = v54 + 8LL * (v51 - 1);
        while ( 1 )
        {
          v53 += sub_39FAC40(*v52);
          v52 = (_QWORD *)v54;
          if ( v55 == v54 )
            break;
          v54 += 8;
        }
      }
      else
      {
        v53 = 0;
      }
      *((_DWORD *)sub_1CD4900((__int64)&v391, (__int64 *)&v423) + 2) = v53;
      if ( v366 )
      {
        v128 = v343[24];
        v129 = *(_DWORD *)(v128 + 24);
        if ( v129 )
        {
          v130 = v129 - 1;
          v131 = *(_QWORD *)(v128 + 8);
          v132 = v130 & (((unsigned int)v423 >> 9) ^ ((unsigned int)v423 >> 4));
          v133 = (const char **)(v131 + 16LL * v132);
          v134 = *v133;
          if ( v423 == *v133 )
          {
LABEL_175:
            v135 = v133[1];
            if ( v135 && v423 == **((const char ***)v135 + 4) )
            {
              v136 = sub_1BF8310((__int64)v423, 0, v343[24]);
              v137 = sub_1CD4900((__int64)&v391, (__int64 *)&v423);
              *((_DWORD *)v137 + 2) += v136;
            }
          }
          else
          {
            v321 = 1;
            while ( v134 != (const char *)-8LL )
            {
              v322 = v321 + 1;
              v132 = v130 & (v321 + v132);
              v133 = (const char **)(v131 + 16LL * v132);
              v134 = *v133;
              if ( v423 == *v133 )
                goto LABEL_175;
              v321 = v322;
            }
          }
        }
      }
      if ( (unsigned int)dword_4FC0240 > 1 )
        break;
LABEL_57:
      v19 = (const char **)v19[1];
      if ( v19 == v345 )
        goto LABEL_66;
LABEL_58:
      v47 = v343[25];
    }
    v56 = *((_DWORD *)sub_1CD4900((__int64)&v391, (__int64 *)&v423) + 2);
    v58 = sub_1649960((__int64)v423);
    v59 = v57;
    if ( v58 )
    {
      v427 = (const char *)v57;
      v60 = v57;
      s1 = v432;
      if ( v57 > 0xF )
      {
        v355 = v57;
        srcg = v58;
        v265 = (void *)sub_22409D0(&s1, &v427, 0);
        v58 = srcg;
        v59 = v355;
        s1 = v265;
        v266 = v265;
        v432[0] = v427;
      }
      else
      {
        if ( v57 == 1 )
        {
          LOBYTE(v432[0]) = *v58;
          v61 = v432;
LABEL_63:
          n = v60;
          *((_BYTE *)v61 + v60) = 0;
          v62 = (const char *)s1;
          goto LABEL_64;
        }
        if ( !v57 )
        {
          v61 = v432;
          goto LABEL_63;
        }
        v266 = v432;
      }
      memcpy(v266, v58, v59);
      v60 = (size_t)v427;
      v61 = s1;
      goto LABEL_63;
    }
    n = 0;
    s1 = v432;
    v62 = (const char *)v432;
    LOBYTE(v432[0]) = 0;
LABEL_64:
    fprintf(stderr, "Block %s: live-in = %d\n", v62, v56);
    if ( s1 == v432 )
      goto LABEL_57;
    j_j___libc_free_0(s1, v432[0] + 1LL);
    v19 = (const char **)v19[1];
    if ( v19 != v345 )
      goto LABEL_58;
LABEL_66:
    v63 = *(_QWORD *)(a2 + 80);
    if ( (const char **)v63 == v345 )
      goto LABEL_591;
    LODWORD(v19) = 0;
    do
    {
      while ( 1 )
      {
        v64 = (void *)(v63 - 24);
        if ( !v63 )
          v64 = 0;
        s1 = v64;
        if ( *((_DWORD *)sub_1CD4900((__int64)&v391, (__int64 *)&s1) + 2) >= (int)v19 )
          break;
        v63 = *(_QWORD *)(v63 + 8);
        if ( (const char **)v63 == v345 )
          goto LABEL_73;
      }
      v65 = sub_1CD4900((__int64)&v391, (__int64 *)&s1);
      v63 = *(_QWORD *)(v63 + 8);
      LODWORD(v19) = *((_DWORD *)v65 + 2);
    }
    while ( (const char **)v63 != v345 );
LABEL_73:
    v341 = (int)v19;
LABEL_74:
    if ( v341 <= (int)v339 )
    {
      j___libc_free_0(v392);
      v367 = 0;
      goto LABEL_130;
    }
    v367 = 0;
    v66 = v343;
    v338 = 8LL * ((v337 + 63) >> 6);
    while ( 2 )
    {
      v379 = 0;
      v380 = 0;
      v381 = 0;
      v67 = *(_QWORD *)(a2 + 80);
      if ( (const char **)v67 != v345 )
      {
        while ( 1 )
        {
          v71 = v67 - 24;
          v72 = v394;
          if ( !v67 )
            v71 = 0;
          v427 = (const char *)v71;
          if ( !v394 )
            break;
          v68 = (v394 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
          v69 = (_DWORD *)(v392 + 16LL * v68);
          v70 = *(_QWORD *)v69;
          if ( v71 == *(_QWORD *)v69 )
          {
LABEL_79:
            if ( v69[2] == v341 )
              goto LABEL_90;
LABEL_80:
            v67 = *(_QWORD *)(v67 + 8);
            if ( (const char **)v67 == v345 )
              goto LABEL_94;
          }
          else
          {
            v237 = 1;
            v73 = 0;
            while ( v70 != -8 )
            {
              if ( !v73 && v70 == -16 )
                v73 = v69;
              v68 = (v394 - 1) & (v237 + v68);
              v69 = (_DWORD *)(v392 + 16LL * v68);
              v70 = *(_QWORD *)v69;
              if ( v71 == *(_QWORD *)v69 )
                goto LABEL_79;
              ++v237;
            }
            if ( !v73 )
              v73 = v69;
            ++v391;
            v74 = v393 + 1;
            if ( 4 * ((int)v393 + 1) >= 3 * v394 )
              goto LABEL_85;
            if ( v394 - HIDWORD(v393) - v74 <= v394 >> 3 )
              goto LABEL_86;
LABEL_87:
            LODWORD(v393) = v74;
            if ( *(_QWORD *)v73 != -8 )
              --HIDWORD(v393);
            *(_QWORD *)v73 = v71;
            v73[2] = 0;
            if ( v341 )
              goto LABEL_80;
LABEL_90:
            v75 = v380;
            if ( v380 == v381 )
            {
              sub_1292090((__int64)&v379, v380, &v427);
              goto LABEL_80;
            }
            if ( v380 )
            {
              *v380 = v427;
              v75 = v380;
            }
            v380 = v75 + 1;
            v67 = *(_QWORD *)(v67 + 8);
            if ( (const char **)v67 == v345 )
              goto LABEL_94;
          }
        }
        ++v391;
LABEL_85:
        v72 = 2 * v394;
LABEL_86:
        sub_18EEB70((__int64)&v391, v72);
        sub_1CD3300((__int64)&v391, (__int64 *)&v427, &s1);
        v73 = s1;
        v71 = (__int64)v427;
        v74 = v393 + 1;
        goto LABEL_87;
      }
LABEL_94:
      if ( dword_4FC0240 )
      {
        v245 = (__int64)sub_16E8CB0();
        v246 = *(__m128 **)(v245 + 24);
        if ( *(_QWORD *)(v245 + 16) - (_QWORD)v246 <= 0x10u )
        {
          v245 = sub_16E7EE0(v245, "Current max-in = ", 0x11u);
        }
        else
        {
          si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42E19C0);
          v246[1].m128_i8[0] = 32;
          *v246 = si128;
          *(_QWORD *)(v245 + 24) += 17LL;
        }
        v247 = sub_16E7AB0(v245, v341);
        v248 = *(_WORD **)(v247 + 24);
        v249 = v247;
        if ( *(_QWORD *)(v247 + 16) - (_QWORD)v248 <= 1u )
        {
          v249 = sub_16E7EE0(v247, ", ", 2u);
        }
        else
        {
          *v248 = 8236;
          *(_QWORD *)(v247 + 24) += 2LL;
        }
        v250 = sub_16E7A90(v249, v380 - v379);
        v251 = *(_WORD **)(v250 + 24);
        if ( *(_QWORD *)(v250 + 16) - (_QWORD)v251 <= 1u )
        {
          sub_16E7EE0(v250, ",\n", 2u);
        }
        else
        {
          *v251 = 2604;
          *(_QWORD *)(v250 + 24) += 2LL;
        }
        v252 = v379;
        v253 = v380 - v379;
        if ( (_DWORD)v253 )
        {
          v254 = 0;
          v255 = 8LL * (unsigned int)(v253 - 1);
          while ( 1 )
          {
            v262 = (__int64)v252[v254 / 8];
            v263 = (__int64)sub_16E8CB0();
            v264 = *(_BYTE **)(v263 + 24);
            if ( *(_BYTE **)(v263 + 16) == v264 )
            {
              v263 = sub_16E7EE0(v263, "\t", 1u);
            }
            else
            {
              *v264 = 9;
              ++*(_QWORD *)(v263 + 24);
            }
            v256 = sub_1649960(v262);
            v258 = *(_BYTE **)(v263 + 24);
            v259 = (char *)v256;
            v260 = *(_BYTE **)(v263 + 16);
            v261 = v257;
            if ( v260 - v258 < v257 )
            {
              v263 = sub_16E7EE0(v263, v259, v257);
              v260 = *(_BYTE **)(v263 + 16);
              v258 = *(_BYTE **)(v263 + 24);
            }
            else if ( v257 )
            {
              memcpy(v258, v259, v257);
              v260 = *(_BYTE **)(v263 + 16);
              v258 = (_BYTE *)(v261 + *(_QWORD *)(v263 + 24));
              *(_QWORD *)(v263 + 24) = v258;
            }
            if ( v260 == v258 )
            {
              sub_16E7EE0(v263, "\n", 1u);
              if ( v255 == v254 )
                break;
            }
            else
            {
              *v258 = 10;
              ++*(_QWORD *)(v263 + 24);
              if ( v255 == v254 )
                break;
            }
            v252 = v379;
            v254 += 8LL;
          }
        }
        fputc(10, stderr);
      }
      dest = 0;
      v383 = 0;
      v384 = v337;
      v76 = (void *)malloc(v338);
      if ( !v76 )
      {
        if ( v338 || (v334 = malloc(1u)) == 0 )
          sub_16BD1C0("Allocation failed", 1u);
        else
          v76 = (void *)v334;
      }
      dest = v76;
      v383 = (v337 + 63) >> 6;
      if ( (v337 + 63) >> 6 )
      {
        memset(v76, 0, v338);
        memset(v76, 0, v338);
      }
      v19 = v379;
      v395 = 0;
      v396 = 0;
      v347 = v380;
      v371 = (_BYTE *)((char *)v380 - (char *)v379);
      v77 = v380 - v379;
      v397 = 0;
      v398 = 0;
      LODWORD(v351) = v77;
      if ( !(_DWORD)v77 )
        goto LABEL_109;
      v78 = 0;
      v372 = (unsigned int)v77;
      do
      {
        v79 = (__int64)v19[v78];
        v80 = *(_QWORD *)sub_1BFDF20(v66[25], v79);
        v83 = sub_1CDB400((__int64)v66, v79);
        if ( !v78 )
        {
          v84 = *(_DWORD *)(v80 + 16);
          if ( v384 < v84 )
          {
            if ( v84 <= v383 << 6 )
              goto LABEL_209;
            v349 = *(_DWORD *)(v80 + 16);
            v233 = (v84 + 63) >> 6;
            v353 = v383;
            if ( v233 < 2 * v383 )
              v233 = 2 * v383;
            v234 = realloc((unsigned __int64)dest, 8 * v233, v383, v84, v81, v82);
            v235 = v353;
            v236 = v349;
            if ( !v234 )
            {
              if ( 8 * v233 )
              {
                sub_16BD1C0("Allocation failed", 1u);
                v236 = v349;
                v234 = 0;
                v235 = v353;
              }
              else
              {
                v234 = (void *)malloc(1u);
                v235 = v353;
                v236 = v349;
                if ( !v234 )
                {
                  sub_16BD1C0("Allocation failed", 1u);
                  v235 = v353;
                  v234 = 0;
                  v236 = v349;
                }
              }
            }
            v383 = v233;
            v354 = v236;
            srcc = v235;
            dest = v234;
            sub_13A4C60((__int64)&dest, 0);
            v84 = v354;
            if ( v383 != srcc )
            {
              memset((char *)dest + 8 * srcc, 0, 8 * (v383 - srcc));
              v84 = v354;
            }
            v152 = v384;
            if ( v84 > v384 )
            {
LABEL_209:
              srcf = v84;
              sub_13A4C60((__int64)&dest, 0);
              v152 = v384;
              v84 = srcf;
            }
            v384 = v84;
            if ( v84 < v152 )
              sub_13A4C60((__int64)&dest, 0);
            v84 = *(_DWORD *)(v80 + 16);
          }
          v85 = 0;
          v86 = (v84 + 63) >> 6;
          if ( v86 )
          {
            do
            {
              *((_QWORD *)dest + v85) |= *(_QWORD *)(*(_QWORD *)v80 + 8 * v85);
              ++v85;
            }
            while ( v86 != v85 );
          }
          if ( (__int64 *)v83 != &v395 )
          {
            j___libc_free_0(v396);
            v87 = *(unsigned int *)(v83 + 24);
            LODWORD(v398) = v87;
            if ( (_DWORD)v87 )
            {
              v396 = (char *)sub_22077B0(8 * v87);
              v397 = *(_QWORD *)(v83 + 16);
              memcpy(v396, *(const void **)(v83 + 8), 8LL * (unsigned int)v398);
            }
            else
            {
              v396 = 0;
              v397 = 0;
            }
          }
          goto LABEL_107;
        }
        v138 = (v384 + 63) >> 6;
        v139 = (unsigned int)(*(_DWORD *)(v80 + 16) + 63) >> 6;
        if ( v139 > v138 )
          v139 = (v384 + 63) >> 6;
        v140 = 0;
        if ( v139 )
        {
          do
          {
            v141 = (char *)dest + v140;
            v142 = *(_QWORD *)(*(_QWORD *)v80 + v140);
            v140 += 8;
            *v141 &= v142;
          }
          while ( 8LL * v139 != v140 );
        }
        for ( ; v138 != v139; *((_QWORD *)dest + v143) = 0 )
          v143 = v139++;
        v144 = *(_QWORD **)(v83 + 8);
        v145 = &v144[*(unsigned int *)(v83 + 24)];
        if ( *(_DWORD *)(v83 + 16) && v144 != v145 )
        {
          while ( *v144 == -16 || *v144 == -8 )
          {
            if ( ++v144 == v145 )
              goto LABEL_107;
          }
LABEL_191:
          if ( v144 != v145 )
          {
            if ( (_DWORD)v398 )
            {
              v146 = (v398 - 1) & (((unsigned int)*v144 >> 9) ^ ((unsigned int)*v144 >> 4));
              v147 = &v396[8 * v146];
              v148 = *(_QWORD *)v147;
              if ( *v144 == *(_QWORD *)v147 )
                goto LABEL_194;
              v149 = 1;
              v150 = 0;
              while ( v148 != -8 )
              {
                if ( v150 || v148 != -16 )
                  v147 = v150;
                v146 = (v398 - 1) & (v149 + v146);
                v148 = *(_QWORD *)&v396[8 * v146];
                if ( *v144 == v148 )
                {
LABEL_194:
                  while ( 1 )
                  {
                    if ( ++v144 == v145 )
                      goto LABEL_107;
                    if ( *v144 != -8 && *v144 != -16 )
                      goto LABEL_191;
                  }
                }
                ++v149;
                v150 = v147;
                v147 = &v396[8 * v146];
              }
              if ( !v150 )
                v150 = v147;
              ++v395;
              v151 = v397 + 1;
              if ( 4 * ((int)v397 + 1) < (unsigned int)(3 * v398) )
              {
                if ( (int)v398 - HIDWORD(v397) - v151 > (unsigned int)v398 >> 3 )
                  goto LABEL_205;
                sub_13B3D40((__int64)&v395, v398);
                if ( (_DWORD)v398 )
                {
                  v224 = 1;
                  v225 = (v398 - 1) & (((unsigned int)*v144 >> 9) ^ ((unsigned int)*v144 >> 4));
                  v150 = &v396[8 * v225];
                  v151 = v397 + 1;
                  v226 = 0;
                  v227 = *(_QWORD *)v150;
                  if ( *v144 != *(_QWORD *)v150 )
                  {
                    while ( v227 != -8 )
                    {
                      if ( !v226 && v227 == -16 )
                        v226 = v150;
                      v225 = (v398 - 1) & (v225 + v224);
                      v150 = &v396[8 * v225];
                      v227 = *(_QWORD *)v150;
                      if ( *v144 == *(_QWORD *)v150 )
                        goto LABEL_205;
                      ++v224;
                    }
                    if ( v226 )
                      v150 = v226;
                  }
                  goto LABEL_205;
                }
                goto LABEL_657;
              }
            }
            else
            {
              ++v395;
            }
            sub_13B3D40((__int64)&v395, 2 * v398);
            if ( (_DWORD)v398 )
            {
              LODWORD(v220) = (v398 - 1) & (((unsigned int)*v144 >> 9) ^ ((unsigned int)*v144 >> 4));
              v150 = &v396[8 * (unsigned int)v220];
              v221 = *(_QWORD *)v150;
              v151 = v397 + 1;
              if ( *v144 != *(_QWORD *)v150 )
              {
                v222 = 1;
                v223 = 0;
                while ( v221 != -8 )
                {
                  if ( v221 == -16 && !v223 )
                    v223 = v150;
                  v220 = ((_DWORD)v398 - 1) & (unsigned int)(v220 + v222);
                  v150 = &v396[8 * v220];
                  v221 = *(_QWORD *)v150;
                  if ( *v144 == *(_QWORD *)v150 )
                    goto LABEL_205;
                  ++v222;
                }
                if ( v223 )
                  v150 = v223;
              }
LABEL_205:
              LODWORD(v397) = v151;
              if ( *(_QWORD *)v150 != -8 )
                --HIDWORD(v397);
              *(_QWORD *)v150 = *v144;
              goto LABEL_194;
            }
LABEL_657:
            LODWORD(v397) = v397 + 1;
            BUG();
          }
        }
LABEL_107:
        v19 = v379;
        ++v78;
      }
      while ( v372 != v78 );
      v347 = v380;
      v371 = (_BYTE *)((char *)v380 - (char *)v379);
      v351 = v380 - v379;
LABEL_109:
      v399 = 0;
      v400 = 0;
      srca = v384;
      v401 = 0;
      v402 = 0;
      if ( !((v384 + 63) >> 6) )
        goto LABEL_347;
      v88 = dest;
      v89 = 0;
      v90 = (char *)dest + 8;
      v91 = (char *)dest + 8 * ((v384 + 63) >> 6);
      while ( 1 )
      {
        v89 += sub_39FAC40(*v88);
        v88 = v90;
        if ( v91 == v90 )
          break;
        v90 += 8;
      }
      if ( v89 )
      {
        if ( (_DWORD)v351 )
        {
          v92 = v19;
          v93 = 0;
          v19 = &v427;
          while ( 1 )
          {
            v427 = v92[v93];
            sub_1C0B2E0((__int64)&s1, (__int64)&v399, (__int64 *)&v427);
            if ( (unsigned int)(v351 - 1) == v93 )
              break;
            v92 = v379;
            ++v93;
          }
          srca = v384;
        }
      }
      else
      {
LABEL_347:
        if ( v371 )
        {
          if ( (unsigned __int64)v371 > 8 && v19 + 1 != v347 )
            v380 = v19 + 1;
        }
        else
        {
          sub_14F2040((__int64)&v379, 1u);
          v19 = v379;
        }
        v228 = v66[25];
        v427 = *v19;
        v229 = *(_QWORD *)sub_1BFDF20(v228, (__int64)v427);
        if ( (void **)v229 != &dest )
        {
          v384 = *(_DWORD *)(v229 + 16);
          v230 = (v384 + 63) >> 6;
          if ( v384 > v383 << 6 )
          {
            v19 = (const char **)(8 * v230);
            v244 = (void *)malloc(8 * v230);
            if ( !v244 )
            {
              if ( v19 || (v333 = malloc(1u)) == 0 )
                sub_16BD1C0("Allocation failed", 1u);
              else
                v244 = (void *)v333;
            }
            memcpy(v244, *(const void **)v229, 8 * v230);
            _libc_free((unsigned __int64)dest);
            dest = v244;
            v383 = v230;
          }
          else
          {
            if ( v384 )
              memcpy(dest, *(const void **)v229, 8 * v230);
            sub_13A4C60((__int64)&dest, 0);
          }
        }
        v231 = sub_1CDB400((__int64)v66, (__int64)v427);
        if ( (__int64 *)v231 != &v395 )
        {
          j___libc_free_0(v396);
          v232 = *(unsigned int *)(v231 + 24);
          LODWORD(v398) = v232;
          if ( (_DWORD)v232 )
          {
            v396 = (char *)sub_22077B0(8 * v232);
            v397 = *(_QWORD *)(v231 + 16);
            memcpy(v396, *(const void **)(v231 + 8), 8LL * (unsigned int)v398);
          }
          else
          {
            v396 = 0;
            v397 = 0;
          }
        }
        sub_1C0B2E0((__int64)&s1, (__int64)&v399, (__int64 *)&v427);
        srca = v384;
      }
      v385 = 0;
      v386 = 0;
      v387 = 0;
      v403 = 0;
      v404 = 0;
      v405 = 0;
      v406 = 0;
      if ( !srca )
        goto LABEL_125;
      v94 = (srca - 1) >> 6;
      v95 = 0;
      while ( 1 )
      {
        _RDX = *((_QWORD *)dest + v95);
        if ( v94 == (_DWORD)v95 )
          _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)srca) & *((_QWORD *)dest + v95);
        if ( _RDX )
          break;
        if ( v94 + 1 == ++v95 )
          goto LABEL_125;
      }
      __asm { tzcnt   rdx, rdx }
      v154 = _RDX + ((_DWORD)v95 << 6);
      if ( v154 == -1 )
      {
LABEL_125:
        v97 = 0;
        goto LABEL_126;
      }
      while ( 2 )
      {
        v155 = (char *)sub_1CE0F00(v66[25], v154);
        v419 = v155;
        if ( (unsigned __int8)v155[16] <= 0x17u )
        {
          v423 = 0;
          sub_1C0AE50((__int64)&s1, (__int64)&v403, (__int64 *)&v419);
        }
        else
        {
          v423 = v155;
          if ( sub_1CD06C0(v155, 1) )
          {
            v156 = v386;
            if ( v386 == v387 )
            {
              sub_170B610((__int64)&v385, v386, &v423);
            }
            else
            {
              if ( v386 )
              {
                *(_QWORD *)v386 = v423;
                v156 = v386;
              }
              v386 = v156 + 8;
            }
            goto LABEL_220;
          }
          v186 = (__int64)v423;
          v187 = v406;
          v427 = v423;
          if ( !(_DWORD)v406 )
          {
            ++v403;
            goto LABEL_424;
          }
          v188 = (v406 - 1) & (((unsigned int)v423 >> 9) ^ ((unsigned int)v423 >> 4));
          v189 = (_QWORD *)(v404 + 8LL * v188);
          v190 = *v189;
          if ( v423 != (const char *)*v189 )
          {
            v191 = 1;
            v192 = 0;
            while ( v190 != -8 )
            {
              if ( !v192 && v190 == -16 )
                v192 = v189;
              LODWORD(v19) = v191 + 1;
              v188 = (v406 - 1) & (v191 + v188);
              v189 = (_QWORD *)(v404 + 8LL * v188);
              v190 = *v189;
              if ( v423 == (const char *)*v189 )
                goto LABEL_220;
              ++v191;
            }
            if ( v192 )
              v189 = v192;
            ++v403;
            v193 = v405 + 1;
            if ( 4 * ((int)v405 + 1) < (unsigned int)(3 * v406) )
            {
              if ( (int)v406 - HIDWORD(v405) - v193 > (unsigned int)v406 >> 3 )
              {
LABEL_272:
                LODWORD(v405) = v193;
                if ( *v189 != -8 )
                  --HIDWORD(v405);
                *v189 = v186;
                goto LABEL_220;
              }
LABEL_425:
              sub_1353F00((__int64)&v403, v187);
              sub_1A97120((__int64)&v403, (__int64 *)&v427, &s1);
              v189 = s1;
              v186 = (__int64)v427;
              v193 = v405 + 1;
              goto LABEL_272;
            }
LABEL_424:
            v187 = 2 * v406;
            goto LABEL_425;
          }
        }
LABEL_220:
        v157 = v154 + 1;
        if ( v384 != v154 + 1 )
        {
          v158 = v157 >> 6;
          v159 = (v384 - 1) >> 6;
          if ( v157 >> 6 <= v159 )
          {
            v160 = 64 - (v157 & 0x3F);
            v161 = 0xFFFFFFFFFFFFFFFFLL >> v160;
            v162 = v158;
            if ( v160 == 64 )
              v161 = 0;
            v163 = ~v161;
            while ( 1 )
            {
              _RAX = *((_QWORD *)dest + v162);
              if ( v158 == (_DWORD)v162 )
                _RAX = v163 & *((_QWORD *)dest + v162);
              if ( v159 == (_DWORD)v162 )
                _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v384;
              if ( _RAX )
                break;
              if ( v159 < (unsigned int)++v162 )
                goto LABEL_232;
            }
            __asm { tzcnt   rax, rax }
            v154 = ((_DWORD)v162 << 6) + _RAX;
            if ( v154 != -1 )
              continue;
          }
        }
        break;
      }
LABEL_232:
      if ( v386 == v385 )
        goto LABEL_473;
      v340 = v341;
      v388 = 0;
      v389 = 0;
      v390 = 0;
      v407 = 0;
      v408 = 0;
      v409 = 0;
      v410 = 0;
      v411 = 0;
      v412 = 0;
      v413 = 0;
      v414 = 0;
      v415 = 0;
      v416 = 0;
      v417 = 0;
      v418 = 0;
      v377 = 0;
      sub_1CE3AF0(
        (__int64)v66,
        (__int64 *)&v385,
        (__int64)&v407,
        (__int64)&v388,
        (__int64)&v403,
        (__int64)&v411,
        (__int64)&v395,
        v341 - v339,
        (float *)&v377,
        v336,
        (__int64)&v415);
      if ( !(_DWORD)v409 )
        goto LABEL_459;
      if ( (unsigned int)dword_4FC0240 > 3 )
        fprintf(stderr, "Reducing %d live-ins\n", v409);
      LODWORD(v19) = (_DWORD)v388;
      v419 = 0;
      v420 = 0;
      v421 = 0;
      v422 = 0;
      v423 = 0;
      v424 = 0;
      v425 = 0;
      v426 = 0;
      if ( v388 != v389 )
      {
        srcb = v66;
        v166 = (__int64 *)v388;
        v19 = (const char **)v389;
        while ( 1 )
        {
          v427 = (const char *)*v166;
          v171 = (const char *)sub_22077B0(24);
          v172 = v171;
          if ( v171 )
          {
            *(_QWORD *)v171 = 0;
            *((_QWORD *)v171 + 1) = 0;
            *((_QWORD *)v171 + 2) = 0;
          }
          v173 = v426;
          if ( !v426 )
            break;
          v167 = (__int64)v427;
          v168 = (v426 - 1) & (((unsigned int)v427 >> 9) ^ ((unsigned int)v427 >> 4));
          v169 = &v424[2 * v168];
          v170 = *v169;
          if ( *v169 == v427 )
          {
LABEL_239:
            ++v166;
            v169[1] = v172;
            if ( v19 == (const char **)v166 )
              goto LABEL_249;
          }
          else
          {
            v238 = 1;
            v239 = 0;
            while ( v170 != (const char *)-8LL )
            {
              if ( v170 == (const char *)-16LL && !v239 )
                v239 = v169;
              v168 = (v426 - 1) & (v238 + v168);
              v169 = &v424[2 * v168];
              v170 = *v169;
              if ( v427 == *v169 )
                goto LABEL_239;
              ++v238;
            }
            if ( v239 )
              v169 = v239;
            ++v423;
            v174 = v425 + 1;
            if ( 4 * ((int)v425 + 1) < 3 * v426 )
            {
              if ( v426 - HIDWORD(v425) - v174 > v426 >> 3 )
                goto LABEL_246;
              goto LABEL_245;
            }
LABEL_244:
            v173 = 2 * v426;
LABEL_245:
            sub_1CD3DF0((__int64)&v423, v173);
            sub_1CD30F0((__int64)&v423, (__int64 *)&v427, &s1);
            v169 = (const char **)s1;
            v167 = (__int64)v427;
            v174 = v425 + 1;
LABEL_246:
            LODWORD(v425) = v174;
            if ( *v169 != (const char *)-8LL )
              --HIDWORD(v425);
            ++v166;
            v169[1] = 0;
            *v169 = (const char *)v167;
            v169[1] = v172;
            if ( v19 == (const char **)v166 )
            {
LABEL_249:
              v66 = srcb;
              goto LABEL_250;
            }
          }
        }
        ++v423;
        goto LABEL_244;
      }
LABEL_250:
      s1 = 0;
      v175 = 0;
      n = 0;
      memset(v432, 0, sizeof(v432));
      v433 = 0;
      v434 = 0;
      v435 = 0;
      v436 = 0;
      sub_1C08D60((__int64 *)&s1, 0);
      v176 = v379;
      v177 = 0;
      if ( v379 != v380 )
      {
        while ( 1 )
        {
          v182 = (__int64)v176[v177];
          v183 = v422;
          v378 = v182;
          if ( !(_DWORD)v422 )
            break;
          LODWORD(v19) = 1;
          v178 = 0;
          v179 = (v422 - 1) & (((unsigned int)v182 >> 9) ^ ((unsigned int)v182 >> 4));
          v180 = (_QWORD *)(v420 + 8LL * v179);
          v181 = *v180;
          if ( v182 != *v180 )
          {
            while ( v181 != -8 )
            {
              if ( v181 == -16 && !v178 )
                v178 = v180;
              v179 = (v422 - 1) & ((_DWORD)v19 + v179);
              v180 = (_QWORD *)(v420 + 8LL * v179);
              v181 = *v180;
              if ( v182 == *v180 )
                goto LABEL_253;
              LODWORD(v19) = (_DWORD)v19 + 1;
            }
            if ( !v178 )
              v178 = v180;
            ++v419;
            v184 = v421 + 1;
            if ( 4 * ((int)v421 + 1) < (unsigned int)(3 * v422) )
            {
              if ( (int)v422 - HIDWORD(v421) - v184 <= (unsigned int)v422 >> 3 )
              {
LABEL_257:
                v19 = (const char **)&v419;
                sub_13B3D40((__int64)&v419, v183);
                sub_1898220((__int64)&v419, &v378, &v427);
                v178 = v427;
                v182 = v378;
                v184 = v421 + 1;
              }
              LODWORD(v421) = v184;
              if ( *v178 != -8 )
                --HIDWORD(v421);
              *v178 = v182;
              v185 = v433;
              if ( v433 == (_QWORD *)(v435 - 8) )
              {
                sub_1B4ECC0((__int64 *)&s1, &v378);
                v176 = v379;
              }
              else
              {
                if ( v433 )
                {
                  *v433 = v378;
                  v185 = v433;
                }
                v176 = v379;
                v433 = v185 + 1;
              }
              goto LABEL_253;
            }
LABEL_256:
            v183 = 2 * v422;
            goto LABEL_257;
          }
LABEL_253:
          v177 = (unsigned int)++v175;
          if ( v175 == v380 - v176 )
            goto LABEL_276;
        }
        ++v419;
        goto LABEL_256;
      }
LABEL_276:
      v194 = v433;
      if ( (_QWORD *)v432[0] != v433 )
      {
        v19 = (const char **)&v388;
        do
        {
          if ( v434 == v194 )
          {
            v342 = *(_QWORD *)(*(v436 - 1) + 504LL);
            j_j___libc_free_0(v194, 512);
            v195 = v342;
            v196 = *--v436 + 512LL;
            v434 = (_QWORD *)*v436;
            v435 = v196;
            v433 = v434 + 63;
          }
          else
          {
            v195 = *(v194 - 1);
            v433 = v194 - 1;
          }
          sub_1CE67D0(
            v66,
            v195,
            (__int64 *)&s1,
            (__int64)&v407,
            &v388,
            (__int64)&v411,
            (__int64)&v419,
            (__int64)&v391,
            (__int64)&v415,
            (__int64)&v423,
            (__int64)&v399);
          v194 = v433;
        }
        while ( v433 != (_QWORD *)v432[0] );
      }
      if ( (_DWORD)v425 )
      {
        v241 = v424;
        v19 = &v424[2 * v426];
        if ( v424 != v19 )
        {
          while ( 1 )
          {
            v242 = v241;
            if ( *v241 != (const char *)-16LL && *v241 != (const char *)-8LL )
              break;
            v241 += 2;
            if ( v19 == v241 )
              goto LABEL_283;
          }
          while ( v242 != v19 )
          {
            v243 = v242[1];
            if ( v243 )
            {
              if ( *(_QWORD *)v243 )
                j_j___libc_free_0(*(_QWORD *)v243, *((_QWORD *)v243 + 2) - *(_QWORD *)v243);
              j_j___libc_free_0(v243, 24);
            }
            v242 += 2;
            if ( v242 == v19 )
              break;
            while ( *v242 == (const char *)-8LL || *v242 == (const char *)-16LL )
            {
              v242 += 2;
              if ( v19 == v242 )
                goto LABEL_283;
            }
          }
        }
      }
LABEL_283:
      v197 = 0;
      if ( *(const char ***)(a2 + 80) == v345 )
        goto LABEL_302;
      v374 = v66;
      v198 = *(_QWORD *)(a2 + 80);
      do
      {
        while ( 1 )
        {
          v205 = v198 - 24;
          if ( !v198 )
            v205 = 0;
          v378 = v205;
          if ( (unsigned int)dword_4FC0240 <= 1 )
            goto LABEL_285;
          v19 = (const char **)v429;
          v206 = *((_DWORD *)sub_1CD4900((__int64)&v391, &v378) + 2);
          v207 = (char *)sub_1649960(v378);
          v427 = (const char *)v429;
          if ( v207 )
          {
            sub_1CD0120((__int64 *)&v427, v207, (__int64)&v207[v208]);
            v209 = v427;
          }
          else
          {
            v428 = 0;
            v209 = (const char *)v429;
            LOBYTE(v429[0]) = 0;
          }
          fprintf(stderr, "Block %s: live-in = %d\n", v209, v206);
          if ( v427 == (const char *)v429 )
          {
LABEL_285:
            v199 = v394;
            if ( !v394 )
              goto LABEL_297;
          }
          else
          {
            j_j___libc_free_0(v427, v429[0] + 1LL);
            v199 = v394;
            if ( !v394 )
            {
LABEL_297:
              ++v391;
              goto LABEL_298;
            }
          }
          v200 = v378;
          LODWORD(v201) = (v199 - 1) & (((unsigned int)v378 >> 9) ^ ((unsigned int)v378 >> 4));
          v202 = v392 + 16LL * (unsigned int)v201;
          v203 = *(_QWORD *)v202;
          if ( v378 == *(_QWORD *)v202 )
          {
LABEL_287:
            v204 = *(_DWORD *)(v202 + 8);
            goto LABEL_288;
          }
          v240 = 1;
          v210 = 0;
          while ( v203 != -8 )
          {
            if ( !v210 && v203 == -16 )
              v210 = v202;
            LODWORD(v19) = v240 + 1;
            v201 = (v199 - 1) & ((_DWORD)v201 + v240);
            v202 = v392 + 16 * v201;
            v203 = *(_QWORD *)v202;
            if ( v378 == *(_QWORD *)v202 )
              goto LABEL_287;
            ++v240;
          }
          if ( !v210 )
            v210 = v202;
          ++v391;
          v211 = v393 + 1;
          if ( 4 * ((int)v393 + 1) < 3 * v199 )
          {
            if ( v199 - HIDWORD(v393) - v211 > v199 >> 3 )
              goto LABEL_387;
            goto LABEL_299;
          }
LABEL_298:
          v199 *= 2;
LABEL_299:
          sub_18EEB70((__int64)&v391, v199);
          sub_1CD3300((__int64)&v391, &v378, &v427);
          v210 = (__int64)v427;
          v200 = v378;
          v211 = v393 + 1;
LABEL_387:
          LODWORD(v393) = v211;
          if ( *(_QWORD *)v210 != -8 )
            --HIDWORD(v393);
          *(_QWORD *)v210 = v200;
          v204 = 0;
          *(_DWORD *)(v210 + 8) = 0;
LABEL_288:
          if ( v204 > v197 )
            break;
          v198 = *(_QWORD *)(v198 + 8);
          if ( (const char **)v198 == v345 )
            goto LABEL_301;
        }
        v212 = sub_1CD4900((__int64)&v391, &v378);
        v198 = *(_QWORD *)(v198 + 8);
        v197 = *((_DWORD *)v212 + 2);
      }
      while ( (const char **)v198 != v345 );
LABEL_301:
      v66 = v374;
LABEL_302:
      v367 = 0;
      if ( v197 < v341 )
      {
        v340 = v197;
        v341 = v197;
        v367 = 1;
      }
      if ( v339 <= v340 )
      {
        sub_1C08CE0((__int64 *)&s1);
        j___libc_free_0(v424);
        j___libc_free_0(v420);
        if ( v418 )
        {
          v213 = v416;
          v19 = &v416[4 * v418];
          do
          {
            if ( *v213 != (const char *)-16LL && *v213 != (const char *)-8LL )
            {
              v214 = v213[1];
              if ( v214 )
                j_j___libc_free_0(v214, v213[3] - v214);
            }
            v213 += 4;
          }
          while ( v19 != v213 );
        }
        j___libc_free_0(v416);
        if ( v414 )
        {
          v215 = v412;
          v19 = &v412[4 * v414];
          do
          {
            if ( *v215 != (const char *)-8LL && *v215 != (const char *)-16LL )
            {
              v216 = v215[1];
              if ( v216 )
                j_j___libc_free_0(v216, v215[3] - v216);
            }
            v215 += 4;
          }
          while ( v19 != v215 );
        }
        j___libc_free_0(v412);
        j___libc_free_0(v408);
        if ( v388 )
          j_j___libc_free_0(v388, v390 - (_QWORD)v388);
        j___libc_free_0(v404);
        if ( v385 )
          j_j___libc_free_0(v385, v387 - v385);
        j___libc_free_0(v400);
        j___libc_free_0(v396);
        _libc_free((unsigned __int64)dest);
        if ( v379 )
          j_j___libc_free_0(v379, (char *)v381 - (char *)v379);
        if ( !v367 )
        {
LABEL_326:
          v217 = 0;
          v218 = (unsigned int *)v343[25];
          if ( *v218 )
          {
            do
            {
              v219 = sub_1CE0F00((__int64)v218, v217);
              if ( *(_BYTE *)(v219 + 16) > 0x17u && !*(_QWORD *)(v219 + 8) )
                sub_15F20C0((_QWORD *)v219);
              ++v217;
              v218 = (unsigned int *)v343[25];
            }
            while ( *v218 > v217 );
          }
          j___libc_free_0(v392);
          sub_1C007A0(v343[25], a2, 1, 1, 1, 0);
          v367 = 1;
          goto LABEL_130;
        }
        continue;
      }
      break;
    }
    sub_1C08CE0((__int64 *)&s1);
    j___libc_free_0(v424);
    j___libc_free_0(v420);
    v367 = 1;
LABEL_459:
    if ( v418 )
    {
      v267 = v416;
      v268 = &v416[4 * v418];
      do
      {
        if ( *v267 != (const char *)-16LL && *v267 != (const char *)-8LL )
        {
          v269 = v267[1];
          if ( v269 )
            j_j___libc_free_0(v269, v267[3] - v269);
        }
        v267 += 4;
      }
      while ( v268 != v267 );
    }
    j___libc_free_0(v416);
    if ( v414 )
    {
      v270 = v412;
      v271 = &v412[4 * v414];
      do
      {
        if ( *v270 != (const char *)-16LL && *v270 != (const char *)-8LL )
          sub_1CD1DE0(v270 + 1);
        v270 += 4;
      }
      while ( v271 != v270 );
    }
    j___libc_free_0(v412);
    j___libc_free_0(v408);
    sub_1CD1DE0(&v388);
LABEL_473:
    v97 = v404;
LABEL_126:
    j___libc_free_0(v97);
    sub_1CD1DE0(&v385);
    j___libc_free_0(v400);
    j___libc_free_0(v396);
    _libc_free((unsigned __int64)dest);
    if ( v379 )
      j_j___libc_free_0(v379, (char *)v381 - (char *)v379);
    if ( v367 )
      goto LABEL_326;
    j___libc_free_0(v392);
LABEL_130:
    v98 = v343[25];
    src = (unsigned int *)v98;
    if ( !*(_DWORD *)(v98 + 96) )
      goto LABEL_131;
    v272 = *(_QWORD **)(v98 + 88);
    v273 = &v272[2 * *(unsigned int *)(v343[25] + 104)];
    if ( v272 == v273 )
      goto LABEL_131;
    while ( 1 )
    {
      v274 = v272;
      if ( *v272 != -8 && *v272 != -16 )
        break;
      v272 += 2;
      if ( v273 == v272 )
        goto LABEL_131;
    }
    if ( v273 == v272 )
      goto LABEL_131;
    v375 = 0;
    do
    {
      v275 = *(_QWORD *)v274[1];
      v276 = (unsigned int)(*(_DWORD *)(v275 + 16) + 63) >> 6;
      if ( v276 )
      {
        v277 = *(_QWORD **)v275;
        v278 = 0;
        v279 = *(_QWORD *)v275 + 8LL;
        v19 = (const char **)(v279 + 8LL * (v276 - 1));
        while ( 1 )
        {
          v278 += sub_39FAC40(*v277);
          v277 = (_QWORD *)v279;
          if ( (const char **)v279 == v19 )
            break;
          v279 += 8;
        }
        if ( v375 >= v278 )
          v278 = v375;
        v375 = v278;
      }
      v274 += 2;
      if ( v274 == v273 )
        break;
      while ( *v274 == -16 || *v274 == -8 )
      {
        v274 += 2;
        if ( v273 == v274 )
          goto LABEL_492;
      }
    }
    while ( v274 != v273 );
LABEL_492:
    LOBYTE(v19) = v367 & (v339 < v375);
    if ( !(_BYTE)v19 )
    {
LABEL_131:
      v10 = v343;
      LODWORD(v19) = 1;
      goto LABEL_132;
    }
    --v335;
  }
  while ( v335 );
  v10 = v343;
  if ( !dword_4FBFB40 )
    goto LABEL_133;
LABEL_495:
  v280 = sub_1CD74B0((__int64)v10, a2, si128, a4, a5, a6, v40, v41, a9, a10);
  if ( (_BYTE)v280 )
    LODWORD(v19) = v280;
  if ( dword_4FBFA60 )
  {
LABEL_498:
    v281 = sub_1CDE4D0(a2);
    if ( (_BYTE)v281 )
      LODWORD(v19) = v281;
  }
LABEL_134:
  sub_1CD2540((__int64)v10);
LABEL_135:
  if ( !dword_4FBF980 )
    return (unsigned int)v19;
  v368 = *(_QWORD **)(a2 + 80);
  if ( v368 == (_QWORD *)(a2 + 72) )
    return (unsigned int)v19;
  v348 = (unsigned __int8)v19;
  v99 = 0;
  while ( 2 )
  {
    if ( !v368 )
      BUG();
    v373 = (_QWORD *)v368[3];
    v100 = (_QWORD *)(v368[2] & 0xFFFFFFFFFFFFFFF8LL);
    if ( v373 != v100 )
    {
      v101 = v99;
      while ( 1 )
      {
        if ( !v100 )
          BUG();
        v107 = *((_BYTE *)v100 - 8);
        v108 = v100 - 3;
        if ( v107 == 77 )
        {
LABEL_168:
          v99 = v101;
          goto LABEL_169;
        }
        v109 = *v100 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !*(v100 - 2) )
          goto LABEL_144;
        if ( v107 != 35 )
          break;
        v110 = (__int64)(v100 - 3);
        v111 = 1;
        v112 = 0;
        do
        {
          v113 = *(_QWORD *)(v110 - 48);
          v114 = *(__int64 **)(v110 - 24);
          v115 = v111 - 1;
          v116 = *(_BYTE *)(v113 + 16);
          if ( (unsigned __int8)(v116 - 35) > 0x11u )
          {
            v116 = *((_BYTE *)v114 + 16);
            if ( (unsigned __int8)(v116 - 35) > 0x11u || v112 && (__int64 *)v113 != v112 )
              goto LABEL_561;
            v115 = v111;
            v110 = *(_QWORD *)(v110 - 24);
            v112 = (__int64 *)v113;
          }
          else
          {
            if ( v112 )
            {
              if ( !v114 || v114 != v112 )
                goto LABEL_561;
              v114 = v112;
            }
            v115 = v111;
            v110 = *(_QWORD *)(v110 - 48);
            v112 = v114;
          }
          ++v111;
        }
        while ( v116 == 35 );
        if ( v116 != 47
          || !v112
          || !(v352 = *(_QWORD *)(v110 - 48) != 0 && v112 == *(__int64 **)(v110 - 48))
          || (v117 = *(_QWORD *)(v110 - 24), *(_BYTE *)(v117 + 16) != 13)
          || ((v118 = *(_DWORD *)(v117 + 32), v119 = *(__int64 **)(v117 + 24), v118 > 0x40)
            ? (v120 = *v119)
            : (v120 = (__int64)((_QWORD)v119 << (64 - (unsigned __int8)v118)) >> (64 - (unsigned __int8)v118)),
              (unsigned int)(v120 - 1) > 7) )
        {
LABEL_561:
          if ( v115 <= 1 )
            goto LABEL_144;
          s1 = "factor";
          LOWORD(v432[0]) = 259;
          v304 = (__int64 *)sub_15A0680(*v112, v115, 0);
          v305 = sub_15FB440(15, v304, (__int64)v112, (__int64)&s1, (__int64)(v100 - 3));
          if ( (*((_BYTE *)v100 - 1) & 0x40) != 0 )
          {
            v306 = (__int64 *)*(v100 - 4);
            v307 = *v306;
            if ( (__int64 *)*v306 != v112 )
              goto LABEL_564;
          }
          else
          {
            v306 = &v108[-3 * (*((_DWORD *)v100 - 1) & 0xFFFFFFF)];
            v307 = *v306;
            if ( (__int64 *)*v306 != v112 )
            {
LABEL_564:
              sub_1648780((__int64)(v100 - 3), v307, v305);
              if ( (*((_BYTE *)v100 - 1) & 0x40) != 0 )
                v308 = (_QWORD *)*(v100 - 4);
              else
                v308 = &v108[-3 * (*((_DWORD *)v100 - 1) & 0xFFFFFFF)];
              sub_1648780((__int64)(v100 - 3), v308[3], v110);
              v101 = 1;
              goto LABEL_144;
            }
          }
          sub_1648780((__int64)(v100 - 3), v306[3], v305);
          if ( (*((_BYTE *)v100 - 1) & 0x40) != 0 )
            v317 = (__int64 *)*(v100 - 4);
          else
            v317 = &v108[-3 * (*((_DWORD *)v100 - 1) & 0xFFFFFFF)];
          sub_1648780((__int64)(v100 - 3), *v317, v110);
          v101 = 1;
          goto LABEL_144;
        }
        v121 = (_DWORD)v120 == 0;
        v122 = v120;
        v123 = 1;
        if ( !v121 )
        {
          do
          {
            v123 *= 2;
            --v122;
          }
          while ( v122 );
        }
        s1 = "factor";
        LOWORD(v432[0]) = 259;
        v124 = (__int64 *)sub_15A0680(*v112, v123 + v115, 0);
        v125 = sub_15FB440(15, v124, (__int64)v112, (__int64)&s1, (__int64)v108);
        sub_164D160((__int64)v108, v125, si128, a4, a5, a6, v126, v127, a9, a10);
        sub_15F20C0(v108);
        v101 = v352;
        if ( v373 == (_QWORD *)v109 )
          goto LABEL_168;
LABEL_145:
        v100 = (_QWORD *)v109;
      }
      if ( v107 == 56 )
      {
        v102 = *((_DWORD *)v100 - 1) & 0xFFFFFFF;
        v103 = 4 * v102;
        v104 = v108[-3 * v102];
        v105 = *(_QWORD *)(v104 + 8);
        if ( v105 )
        {
          LOBYTE(v103) = (*((_DWORD *)v100 - 1) & 0xFFFFFFF) == 2 && *(_QWORD *)(v105 + 8) == 0;
          v106 = v103;
          if ( (_BYTE)v103 )
          {
            if ( *(_BYTE *)(v104 + 16) == 56 )
            {
              v309 = (__int64 *)*(v100 - 6);
              v310 = v104;
              v311 = 1;
              while ( 1 )
              {
                v312 = v310;
                v313 = *(_DWORD *)(v310 + 20) & 0xFFFFFFF;
                v310 = *(_QWORD *)(v310 - 24LL * v313);
                v314 = *(_QWORD *)(v310 + 8);
                if ( !v314 || *(_QWORD *)(v314 + 8) || v313 != 2 || v309 != *(__int64 **)(v312 - 24) )
                  break;
                ++v311;
                if ( *(_BYTE *)(v310 + 16) != 56 )
                {
                  v312 = v310;
LABEL_575:
                  sub_1648780((__int64)v108, v104, v312);
                  LOWORD(v432[0]) = 259;
                  s1 = "factor";
                  v315 = (__int64 *)sub_15A0680(*v309, v311, 0);
                  v316 = sub_15FB440(15, v315, (__int64)v309, (__int64)&s1, (__int64)v108);
                  sub_1648780((__int64)v108, (__int64)v309, v316);
                  v101 = v106;
                  goto LABEL_144;
                }
              }
              if ( v311 != 1 )
                goto LABEL_575;
            }
          }
        }
      }
LABEL_144:
      if ( v373 == (_QWORD *)v109 )
        goto LABEL_168;
      goto LABEL_145;
    }
LABEL_169:
    v368 = (_QWORD *)v368[1];
    if ( (_QWORD *)(a2 + 72) != v368 )
      continue;
    break;
  }
  LODWORD(v19) = v348;
  if ( (_BYTE)v99 )
    LODWORD(v19) = v99;
  return (unsigned int)v19;
}
