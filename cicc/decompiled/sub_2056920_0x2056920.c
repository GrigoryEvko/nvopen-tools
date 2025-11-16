// Function: sub_2056920
// Address: 0x2056920
//
__int64 __fastcall sub_2056920(__int64 a1, __m128i *a2, __m128i *a3, __m128i a4, __m128i si128, __m128i a6)
{
  __m128i *v6; // r15
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  int v10; // r8d
  int v11; // r9d
  _OWORD *p_src; // r13
  char v13; // dl
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 m128i_i64; // rsi
  unsigned __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rdx
  __int64 (*v21)(); // rax
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // r14
  __m128i *v25; // r10
  __int64 (__fastcall *v26)(__int64, __int64, __int64, __int64, __int64); // rax
  __int8 v27; // r10
  __int64 (__fastcall *v28)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v29; // r11
  int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rdx
  int v33; // r14d
  __int64 v34; // rax
  __int8 v35; // r13
  int v36; // r12d
  char v37; // cl
  __m128i *v38; // rdx
  __int64 (*v39)(); // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  unsigned int v42; // edx
  __m128i *v43; // r14
  unsigned int *v44; // rbx
  __int64 v45; // rax
  __int64 (*v46)(); // rax
  unsigned int *v47; // r15
  _QWORD **v48; // rax
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 (__fastcall *v51)(__int64, __int64, __int64); // rbx
  __int64 v52; // rdx
  __int64 v53; // rcx
  int v54; // r8d
  int v55; // r9d
  __int64 v56; // rdx
  __int64 v57; // rcx
  int v58; // r8d
  int v59; // r9d
  __int64 v60; // rdx
  __int64 v61; // rcx
  int v62; // r8d
  int v63; // r9d
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // r8d
  int v67; // r9d
  unsigned int v68; // ebx
  __int64 v69; // r9
  __int64 v70; // rdx
  char v71; // al
  char v72; // cl
  bool v73; // r13
  __int64 *v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // r12
  __int64 (__fastcall *v78)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v79; // rbx
  __int64 (__fastcall *v80)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v81; // r12
  __int64 v82; // rbx
  unsigned int v83; // ebx
  __m128i *v84; // r8
  __m128i *v85; // rax
  __int64 v86; // rsi
  unsigned int v87; // edx
  char v88; // al
  unsigned __int64 v89; // rcx
  __int64 v90; // rax
  char v91; // di
  __int64 v92; // rax
  unsigned __int64 v93; // rcx
  __m128i *v94; // r8
  int v95; // eax
  bool v96; // r12
  int v97; // eax
  __int64 v98; // rdi
  const __m128i *v99; // rax
  __int64 v100; // r15
  const __m128i *v101; // rbx
  int v102; // eax
  unsigned int *v103; // rcx
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r11
  __int64 v107; // rax
  const __m128i *v108; // r11
  __int64 v109; // rax
  char v110; // di
  __int64 v111; // rax
  bool v112; // al
  __int64 v113; // r12
  unsigned int v114; // eax
  __int64 v115; // r8
  unsigned __int64 v116; // r11
  __int64 v117; // rsi
  __int64 v118; // rax
  unsigned int v119; // eax
  __int64 v120; // rax
  unsigned int v121; // esi
  int v122; // eax
  unsigned int v123; // eax
  __int64 v124; // r9
  __int64 v125; // rsi
  unsigned __int64 v126; // rcx
  _QWORD *v127; // rax
  __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // rbx
  __int64 v131; // rbx
  __int64 v132; // rbx
  __int64 v133; // rax
  __int8 v134; // al
  int v135; // ebx
  __int128 v136; // kr00_16
  __int64 v137; // rax
  __int64 v138; // rax
  char v139; // di
  unsigned int v140; // eax
  __int8 v141; // al
  __int64 v142; // rax
  __int64 v143; // rax
  char v144; // di
  unsigned int v145; // eax
  __int64 v146; // rax
  int v147; // r8d
  int v148; // r9d
  int v149; // edx
  unsigned __int64 v150; // rdx
  int v151; // ebx
  unsigned __int64 v152; // rax
  __int64 v153; // r12
  __int64 v154; // rdx
  bool v155; // zf
  char v156; // al
  bool v157; // al
  int v158; // eax
  __int64 v159; // r14
  unsigned __int8 v160; // bl
  __int64 (__fastcall *v161)(__int64, __int64, __int64, __int64, __int64); // r9
  __int64 v162; // r10
  unsigned int v163; // r10d
  __int64 *v164; // rdi
  __int64 *v165; // rax
  int v166; // r8d
  int v167; // r9d
  unsigned int v168; // r10d
  __int64 *v169; // r12
  __int64 v170; // rax
  __int64 *v171; // rdx
  __int64 *v172; // r13
  __int64 **v173; // rax
  unsigned int v174; // eax
  __int64 v175; // rdx
  unsigned int v176; // r11d
  __int64 v177; // r12
  __int64 v178; // r13
  __int64 *v179; // rax
  __int64 (__fastcall *v180)(__int64, __int64, __int64, __int64, __int64); // r9
  __int64 v181; // rbx
  __int64 v182; // rax
  __int64 v183; // rax
  unsigned int v184; // esi
  int v185; // eax
  __int64 v186; // rax
  _QWORD *v187; // rax
  unsigned int v188; // eax
  unsigned int v189; // eax
  __int128 v190; // kr10_16
  __int64 v191; // rax
  char v192; // di
  unsigned int v193; // eax
  __int64 v194; // rax
  __int64 v195; // r14
  __int64 *v196; // rdi
  __int64 *v197; // rbx
  __int64 v198; // r12
  __int64 v199; // r13
  const void ***v200; // rax
  int v201; // edx
  __int64 v202; // r9
  __int64 *v203; // rax
  __m128i v204; // xmm7
  __int64 v205; // rdx
  int v207; // ebx
  _BYTE *v208; // rdi
  __int64 v209; // r12
  __int64 v210; // rbx
  __int64 v211; // rdx
  unsigned __int8 v212; // al
  __int64 v213; // r13
  __int64 v214; // r14
  char v215; // r14
  unsigned int v216; // r15d
  __int64 v217; // r13
  unsigned int v218; // eax
  int v219; // esi
  char *v220; // rdx
  __int64 v221; // rdx
  int v222; // r14d
  unsigned int v223; // r12d
  unsigned int v224; // eax
  __int64 v225; // r15
  unsigned int v226; // ebx
  __int64 v227; // r13
  __int64 v228; // rbx
  __int64 v229; // r12
  __int64 v230; // r13
  __int64 v231; // rax
  int v232; // r8d
  int v233; // r9d
  unsigned __int64 v234; // r13
  __int32 v235; // r12d
  __int64 v236; // rax
  __int64 v237; // rbx
  __m128i *v238; // rax
  __m128i *v239; // rbx
  __int64 v240; // rbx
  __int64 *v241; // r13
  __int128 v242; // rax
  __int64 v243; // r12
  __int128 v244; // rax
  _QWORD *v245; // r13
  __int64 v246; // rax
  int v247; // edx
  int v248; // r8d
  __int64 v249; // rdx
  __int64 v250; // r12
  __int64 *v251; // rax
  __m128i *v252; // rdi
  int v253; // edx
  __m128i v254; // xmm5
  __int64 v255; // rsi
  unsigned __int64 v256; // r12
  __int64 v257; // r12
  unsigned int v258; // eax
  __int16 v259; // bx
  __int64 v260; // rax
  _QWORD *v261; // r14
  __int64 v262; // r12
  unsigned int v263; // edx
  unsigned __int8 v264; // al
  _QWORD *v265; // rax
  unsigned __int32 v266; // edx
  __int64 v267; // rax
  char v268; // al
  unsigned __int64 v269; // rdx
  __int64 v270; // r14
  __int64 v271; // rbx
  unsigned int v272; // edx
  __int8 v273; // al
  unsigned int v274; // edx
  __int8 v275; // al
  __int64 v276; // rax
  __m128i *v277; // rax
  __int64 v278; // rax
  __int64 i; // rdx
  __int64 v280; // r15
  char v281; // r10
  __int64 v282; // r11
  int v283; // eax
  __int64 v284; // r15
  int v285; // r13d
  __int64 v286; // rax
  char v287; // di
  unsigned int v288; // eax
  __int64 v289; // rax
  _BYTE *v290; // rsi
  size_t v291; // rdx
  __int128 v292; // [rsp-20h] [rbp-710h]
  __int64 v293; // [rsp-10h] [rbp-700h]
  __int128 v294; // [rsp-10h] [rbp-700h]
  int v295; // [rsp-10h] [rbp-700h]
  __int128 v296; // [rsp-10h] [rbp-700h]
  __m128i *v297; // [rsp-10h] [rbp-700h]
  __m128i *v298; // [rsp-8h] [rbp-6F8h]
  __int64 v299; // [rsp+0h] [rbp-6F0h]
  __int64 v300; // [rsp+8h] [rbp-6E8h]
  __int64 v301; // [rsp+10h] [rbp-6E0h]
  __int64 v302; // [rsp+18h] [rbp-6D8h]
  __int64 *v303; // [rsp+20h] [rbp-6D0h]
  __int64 v305; // [rsp+38h] [rbp-6B8h]
  unsigned int v306; // [rsp+48h] [rbp-6A8h]
  unsigned int v307; // [rsp+4Ch] [rbp-6A4h]
  unsigned __int64 v308; // [rsp+58h] [rbp-698h]
  unsigned __int64 v309; // [rsp+68h] [rbp-688h]
  __int64 v310; // [rsp+70h] [rbp-680h]
  __int64 v311; // [rsp+88h] [rbp-668h]
  __int64 v312; // [rsp+90h] [rbp-660h]
  unsigned __int64 v313; // [rsp+98h] [rbp-658h]
  __m128i *v314; // [rsp+98h] [rbp-658h]
  __m128i *v315; // [rsp+A0h] [rbp-650h]
  char v316; // [rsp+ABh] [rbp-645h]
  int v317; // [rsp+ACh] [rbp-644h]
  __int64 v318; // [rsp+140h] [rbp-5B0h]
  int v319; // [rsp+154h] [rbp-59Ch]
  unsigned int v320; // [rsp+158h] [rbp-598h]
  char v321; // [rsp+160h] [rbp-590h]
  __int64 v322; // [rsp+160h] [rbp-590h]
  unsigned __int8 v323; // [rsp+168h] [rbp-588h]
  __int64 v324; // [rsp+168h] [rbp-588h]
  bool v325; // [rsp+170h] [rbp-580h]
  unsigned __int64 v326; // [rsp+170h] [rbp-580h]
  bool v327; // [rsp+178h] [rbp-578h]
  __int64 v328; // [rsp+178h] [rbp-578h]
  __int64 v329; // [rsp+178h] [rbp-578h]
  __int64 v330; // [rsp+178h] [rbp-578h]
  bool v331; // [rsp+185h] [rbp-56Bh]
  unsigned __int8 v332; // [rsp+186h] [rbp-56Ah]
  bool v333; // [rsp+187h] [rbp-569h]
  unsigned __int8 v334; // [rsp+188h] [rbp-568h]
  __int64 v335; // [rsp+188h] [rbp-568h]
  __int64 v336; // [rsp+190h] [rbp-560h]
  bool v337; // [rsp+190h] [rbp-560h]
  __int64 v338; // [rsp+198h] [rbp-558h]
  bool v339; // [rsp+198h] [rbp-558h]
  __int64 v340; // [rsp+1A0h] [rbp-550h]
  bool v341; // [rsp+1A0h] [rbp-550h]
  __int64 v342; // [rsp+1A8h] [rbp-548h]
  bool v343; // [rsp+1A8h] [rbp-548h]
  __int64 v344; // [rsp+1B8h] [rbp-538h]
  __int32 v345; // [rsp+1B8h] [rbp-538h]
  unsigned int v346; // [rsp+1C0h] [rbp-530h]
  int v347; // [rsp+1C0h] [rbp-530h]
  unsigned int v348; // [rsp+1C0h] [rbp-530h]
  unsigned int v349; // [rsp+1C0h] [rbp-530h]
  __int64 v350; // [rsp+1C8h] [rbp-528h]
  __int64 v351; // [rsp+1D0h] [rbp-520h]
  __int8 v352; // [rsp+1D0h] [rbp-520h]
  __int64 v353; // [rsp+1D8h] [rbp-518h]
  __int64 v354; // [rsp+1D8h] [rbp-518h]
  unsigned int v355; // [rsp+1D8h] [rbp-518h]
  __int64 v356; // [rsp+1D8h] [rbp-518h]
  __int64 v357; // [rsp+1D8h] [rbp-518h]
  __int64 v359; // [rsp+1E0h] [rbp-510h]
  const void **v360; // [rsp+1E0h] [rbp-510h]
  __int8 v361; // [rsp+1E8h] [rbp-508h]
  bool v362; // [rsp+1E8h] [rbp-508h]
  __int64 v363; // [rsp+1E8h] [rbp-508h]
  __int64 v364; // [rsp+1E8h] [rbp-508h]
  __int64 v365; // [rsp+1E8h] [rbp-508h]
  unsigned int v366; // [rsp+1E8h] [rbp-508h]
  __int64 v367; // [rsp+1E8h] [rbp-508h]
  __int64 v368; // [rsp+1E8h] [rbp-508h]
  __int64 v369; // [rsp+1E8h] [rbp-508h]
  int v370; // [rsp+1E8h] [rbp-508h]
  __int64 v371; // [rsp+1E8h] [rbp-508h]
  __int64 v372; // [rsp+1E8h] [rbp-508h]
  __int64 v373; // [rsp+1F0h] [rbp-500h]
  __int64 v374; // [rsp+1F0h] [rbp-500h]
  const __m128i *v375; // [rsp+1F0h] [rbp-500h]
  __int64 v376; // [rsp+1F0h] [rbp-500h]
  __int128 v377; // [rsp+1F0h] [rbp-500h]
  unsigned int v378; // [rsp+1F0h] [rbp-500h]
  __int8 v379; // [rsp+1F8h] [rbp-4F8h]
  __int64 v380; // [rsp+1F8h] [rbp-4F8h]
  __int64 v381; // [rsp+1F8h] [rbp-4F8h]
  unsigned __int8 v382; // [rsp+200h] [rbp-4F0h]
  int v383; // [rsp+200h] [rbp-4F0h]
  __int64 v384; // [rsp+200h] [rbp-4F0h]
  unsigned __int64 v385; // [rsp+200h] [rbp-4F0h]
  unsigned __int64 v386; // [rsp+200h] [rbp-4F0h]
  unsigned __int64 v387; // [rsp+200h] [rbp-4F0h]
  __int64 v388; // [rsp+200h] [rbp-4F0h]
  unsigned int v389; // [rsp+200h] [rbp-4F0h]
  int v390; // [rsp+208h] [rbp-4E8h]
  __int64 v391; // [rsp+208h] [rbp-4E8h]
  __int64 v392; // [rsp+208h] [rbp-4E8h]
  __int64 v393; // [rsp+208h] [rbp-4E8h]
  __int64 v394; // [rsp+208h] [rbp-4E8h]
  __int64 v395; // [rsp+208h] [rbp-4E8h]
  __int64 v396; // [rsp+208h] [rbp-4E8h]
  __m128i *v397; // [rsp+208h] [rbp-4E8h]
  __int64 v398; // [rsp+208h] [rbp-4E8h]
  __int64 v399; // [rsp+210h] [rbp-4E0h]
  __int64 v400; // [rsp+210h] [rbp-4E0h]
  __int64 v401; // [rsp+210h] [rbp-4E0h]
  unsigned __int64 v402; // [rsp+210h] [rbp-4E0h]
  __int64 v403; // [rsp+210h] [rbp-4E0h]
  unsigned __int64 v404; // [rsp+210h] [rbp-4E0h]
  __int64 v405; // [rsp+210h] [rbp-4E0h]
  __int64 v406; // [rsp+210h] [rbp-4E0h]
  __int64 v407; // [rsp+210h] [rbp-4E0h]
  __int64 v408; // [rsp+210h] [rbp-4E0h]
  int v409; // [rsp+210h] [rbp-4E0h]
  __int8 v410; // [rsp+210h] [rbp-4E0h]
  __int64 v411; // [rsp+210h] [rbp-4E0h]
  __int64 v412; // [rsp+210h] [rbp-4E0h]
  __int128 v413; // [rsp+210h] [rbp-4E0h]
  __int64 v414; // [rsp+228h] [rbp-4C8h]
  int v415; // [rsp+268h] [rbp-488h]
  __int8 v416; // [rsp+27Bh] [rbp-475h] BYREF
  unsigned int v417; // [rsp+27Ch] [rbp-474h] BYREF
  __m128i v418; // [rsp+280h] [rbp-470h] BYREF
  __int64 v419; // [rsp+290h] [rbp-460h] BYREF
  __int64 v420; // [rsp+298h] [rbp-458h]
  __int128 v421; // [rsp+2A0h] [rbp-450h] BYREF
  __int64 v422; // [rsp+2B0h] [rbp-440h]
  __int128 v423; // [rsp+2C0h] [rbp-430h] BYREF
  __int64 v424; // [rsp+2D0h] [rbp-420h]
  __int128 v425; // [rsp+2E0h] [rbp-410h] BYREF
  _BYTE v426[16]; // [rsp+2F0h] [rbp-400h] BYREF
  _BYTE *v427; // [rsp+300h] [rbp-3F0h] BYREF
  __int64 v428; // [rsp+308h] [rbp-3E8h]
  _BYTE v429[32]; // [rsp+310h] [rbp-3E0h] BYREF
  char *v430; // [rsp+330h] [rbp-3C0h] BYREF
  __int64 v431; // [rsp+338h] [rbp-3B8h]
  _BYTE v432[64]; // [rsp+340h] [rbp-3B0h] BYREF
  __int128 src; // [rsp+380h] [rbp-370h] BYREF
  __int8 v434; // [rsp+390h] [rbp-360h] BYREF
  __int64 v435; // [rsp+398h] [rbp-358h]
  bool v436; // [rsp+3A0h] [rbp-350h]
  int v437; // [rsp+3A4h] [rbp-34Ch]
  int v438; // [rsp+3A8h] [rbp-348h]
  __int128 v439; // [rsp+3D0h] [rbp-320h] BYREF
  __int64 v440[8]; // [rsp+3E0h] [rbp-310h] BYREF
  _BYTE *v441; // [rsp+420h] [rbp-2D0h] BYREF
  __int64 v442; // [rsp+428h] [rbp-2C8h]
  _BYTE v443[192]; // [rsp+430h] [rbp-2C0h] BYREF
  __m128i v444; // [rsp+4F0h] [rbp-200h] BYREF
  __m128i v445; // [rsp+500h] [rbp-1F0h] BYREF
  __m128i v446; // [rsp+510h] [rbp-1E0h] BYREF
  _QWORD v447[2]; // [rsp+520h] [rbp-1D0h] BYREF
  char v448; // [rsp+530h] [rbp-1C0h] BYREF
  _QWORD v449[2]; // [rsp+5B0h] [rbp-140h] BYREF
  char v450; // [rsp+5C0h] [rbp-130h] BYREF
  _QWORD v451[2]; // [rsp+5D0h] [rbp-120h] BYREF
  char v452; // [rsp+5E0h] [rbp-110h] BYREF
  __int64 v453; // [rsp+680h] [rbp-70h]
  _QWORD v454[2]; // [rsp+688h] [rbp-68h] BYREF
  char v455; // [rsp+698h] [rbp-58h] BYREF

  v6 = a3;
  v7 = (__int64 *)a3[1].m128i_i64[0];
  a3[137].m128i_i32[2] = 0;
  v303 = v7;
  v430 = v432;
  v427 = v429;
  v8 = *(_QWORD *)(a3[5].m128i_i64[0] + 32);
  v431 = 0x400000000LL;
  v428 = 0x400000000LL;
  v344 = sub_1E0A0C0(v8);
  sub_20C7CE0(a2, v344, v6[1].m128i_i64[0], &v430, &v427, 0);
  if ( !v6[1].m128i_i8[10] )
  {
    p_src = &src;
    goto LABEL_3;
  }
  v444.m128i_i64[1] = 0x400000000LL;
  v444.m128i_i64[0] = (__int64)&v445;
  if ( (_DWORD)v431 )
    sub_20449C0((__int64)&v444, &v430, v9, (unsigned int)v431, v10, v11);
  v207 = v428;
  v441 = v443;
  v442 = 0x400000000LL;
  if ( !(_DWORD)v428 )
  {
    p_src = &src;
    v381 = v444.m128i_u32[2];
    if ( !v444.m128i_i32[2] )
      goto LABEL_271;
    v208 = v443;
    goto LABEL_254;
  }
  v208 = v427;
  if ( v427 != v429 )
  {
    v441 = v427;
    v442 = v428;
    v427 = v429;
    v428 = 0;
    goto LABEL_374;
  }
  if ( (unsigned int)v428 > 4 )
  {
    sub_16CD150((__int64)&v441, v443, (unsigned int)v428, 8, v10, v11);
    v208 = v441;
    v290 = v427;
    v291 = 8LL * (unsigned int)v428;
    if ( !v291 )
      goto LABEL_373;
  }
  else
  {
    v290 = v429;
    v208 = v443;
    v291 = 8LL * (unsigned int)v428;
  }
  memcpy(v208, v290, v291);
  v208 = v441;
LABEL_373:
  LODWORD(v442) = v207;
  LODWORD(v428) = 0;
LABEL_374:
  p_src = &src;
  v381 = v444.m128i_u32[2];
  if ( !v444.m128i_i32[2] )
    goto LABEL_269;
LABEL_254:
  v412 = 0;
  v209 = (__int64)a2;
  v397 = v6;
  do
  {
    v210 = *(_QWORD *)&v208[8 * v412];
    v211 = v444.m128i_i64[0] + 16 * v412;
    v212 = *(_BYTE *)v211;
    v213 = *(_QWORD *)(v211 + 8);
    v214 = *(_QWORD *)v397[1].m128i_i64[0];
    LOBYTE(v425) = v212;
    *((_QWORD *)&v425 + 1) = v213;
    if ( v212 )
    {
      v215 = *(_BYTE *)(v209 + v212 + 1155);
      v216 = *(unsigned __int8 *)(v209 + v212 + 1040);
    }
    else
    {
      if ( sub_1F58D20((__int64)&v425) )
      {
        LOBYTE(v439) = 0;
        *((_QWORD *)&v439 + 1) = 0;
        LOBYTE(v423) = 0;
        sub_1F426C0(v209, v214, (unsigned int)v425, v213, (__int64)&v439, (unsigned int *)&src, &v423);
        v215 = v423;
        v282 = *(_QWORD *)v397[1].m128i_i64[0];
      }
      else
      {
        sub_1F40D10((__int64)&v439, v209, v214, v425, *((__int64 *)&v425 + 1));
        v280 = v440[0];
        LOBYTE(src) = BYTE8(v439);
        *((_QWORD *)&src + 1) = v440[0];
        if ( BYTE8(v439) )
        {
          v281 = *(_BYTE *)(v209 + BYTE8(v439) + 1155);
        }
        else if ( sub_1F58D20((__int64)&src) )
        {
          LOBYTE(v439) = 0;
          *((_QWORD *)&v439 + 1) = 0;
          LOBYTE(v421) = 0;
          sub_1F426C0(v209, v214, (unsigned int)src, v280, (__int64)&v439, (unsigned int *)&v423, &v421);
          v281 = v421;
        }
        else
        {
          sub_1F40D10((__int64)&v439, v209, v214, src, *((__int64 *)&src + 1));
          v289 = v336;
          LOBYTE(v289) = BYTE8(v439);
          v336 = v289;
          v281 = sub_1D5E9F0(v209, v214, (unsigned int)v289, v440[0]);
        }
        v215 = v281;
        v282 = *(_QWORD *)v397[1].m128i_i64[0];
      }
      v356 = v282;
      LOBYTE(v425) = 0;
      *((_QWORD *)&v425 + 1) = v213;
      if ( sub_1F58D20((__int64)&v425) )
      {
        LOBYTE(v439) = 0;
        *((_QWORD *)&v439 + 1) = 0;
        LOBYTE(v423) = 0;
        v216 = sub_1F426C0(v209, v356, (unsigned int)v425, v213, (__int64)&v439, (unsigned int *)&src, &v423);
      }
      else
      {
        v335 = v356;
        v283 = sub_1F58D40((__int64)&v425);
        v284 = *((_QWORD *)&v425 + 1);
        v285 = v283;
        src = v425;
        v357 = v425;
        if ( sub_1F58D20((__int64)&src) )
        {
          LOBYTE(v439) = 0;
          *((_QWORD *)&v439 + 1) = 0;
          LOBYTE(v421) = 0;
          sub_1F426C0(v209, v335, (unsigned int)src, v284, (__int64)&v439, (unsigned int *)&v423, &v421);
          v287 = v421;
        }
        else
        {
          sub_1F40D10((__int64)&v439, v209, v335, v357, v284);
          v286 = v338;
          LOBYTE(v286) = BYTE8(v439);
          v338 = v286;
          v287 = sub_1D5E9F0(v209, v335, (unsigned int)v286, v440[0]);
        }
        v288 = sub_2045180(v287);
        v216 = (v288 + v285 - 1) / v288;
      }
    }
    v217 = v216;
    v218 = (unsigned int)sub_2045180(v215) >> 3;
    v219 = v431;
    if ( v216 > HIDWORD(v431) - (unsigned __int64)(unsigned int)v431 )
    {
      v389 = v218;
      sub_16CD150((__int64)&v430, v432, v216 + (unsigned __int64)(unsigned int)v431, 16, v10, v11);
      v218 = v389;
      v219 = v431;
      v220 = &v430[16 * (unsigned int)v431];
      if ( !v216 )
      {
LABEL_351:
        LODWORD(v431) = v219;
        goto LABEL_267;
      }
    }
    else
    {
      v220 = &v430[16 * (unsigned int)v431];
      if ( !v216 )
        goto LABEL_351;
    }
    do
    {
      if ( v220 )
      {
        *v220 = v215;
        *((_QWORD *)v220 + 1) = 0;
      }
      v220 += 16;
      --v217;
    }
    while ( v217 );
    v11 = 0;
    v221 = (unsigned int)v428;
    v222 = 0;
    LODWORD(v431) = v216 + v431;
    v388 = v209;
    v223 = v218;
    v224 = v216;
    v225 = v210;
    v226 = 0;
    do
    {
      v227 = v225 + v226;
      if ( HIDWORD(v428) <= (unsigned int)v221 )
      {
        v355 = v224;
        sub_16CD150((__int64)&v427, v429, 0, 8, v10, v11);
        v221 = (unsigned int)v428;
        v224 = v355;
      }
      ++v222;
      v226 += v223;
      *(_QWORD *)&v427[8 * v221] = v227;
      v221 = (unsigned int)(v428 + 1);
      LODWORD(v428) = v428 + 1;
    }
    while ( v222 != v224 );
    v209 = v388;
LABEL_267:
    ++v412;
    v208 = v441;
  }
  while ( v412 != v381 );
  v6 = v397;
  p_src = &src;
LABEL_269:
  if ( v208 != v443 )
    _libc_free((unsigned __int64)v208);
LABEL_271:
  if ( (__m128i *)v444.m128i_i64[0] != &v445 )
    _libc_free(v444.m128i_u64[0]);
LABEL_3:
  v13 = v6[1].m128i_i8[8];
  v441 = v443;
  v442 = 0x400000000LL;
  v444.m128i_i64[0] = (__int64)&v445;
  v444.m128i_i64[1] = 0x200000000LL;
  v14 = 0;
  if ( (v13 & 1) != 0 )
  {
    v445.m128i_i32[0] = 40;
    v14 = 1;
    v444.m128i_i32[2] = 1;
  }
  if ( (v13 & 2) != 0 )
  {
    v445.m128i_i32[v14] = 58;
    v14 = (unsigned int)++v444.m128i_i32[2];
    if ( (v13 & 8) != 0 )
    {
      if ( (unsigned int)v14 >= v444.m128i_i32[3] )
      {
        sub_16CD150((__int64)&v444, &v445, 0, 4, v10, v11);
        v14 = v444.m128i_u32[2];
      }
      goto LABEL_223;
    }
  }
  else
  {
    if ( (v13 & 8) == 0 )
      goto LABEL_7;
LABEL_223:
    *(_DWORD *)(v444.m128i_i64[0] + 4 * v14) = 12;
    LODWORD(v14) = ++v444.m128i_i32[2];
  }
LABEL_7:
  v15 = sub_1560040(*(__int64 **)v6[1].m128i_i64[0], 0, (unsigned int *)v444.m128i_i64[0], (unsigned int)v14);
  if ( (__m128i *)v444.m128i_i64[0] != &v445 )
    _libc_free(v444.m128i_u64[0]);
  m128i_i64 = v6[1].m128i_i64[0];
  sub_1F431C0(v6[2].m128i_u32[0], (__int64 *)m128i_i64, v15, (__int64)&v441, a2, v344);
  v20 = (_QWORD *)a2->m128i_i64[0];
  v21 = *(__int64 (**)())(a2->m128i_i64[0] + 1216);
  if ( v21 == sub_1FD3420 )
  {
LABEL_10:
    v306 = v431;
    if ( (_DWORD)v431 )
    {
      v353 = 16LL * (unsigned int)v431;
      v22 = 0;
      while ( 1 )
      {
        v17 = v6[1].m128i_u64[0];
        v19 = v6[2].m128i_u32[0];
        v23 = *(_QWORD *)&v430[v22];
        v24 = *(_QWORD *)&v430[v22 + 8];
        v25 = *(__m128i **)v17;
        v26 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v20[48];
        if ( v26 != sub_1F42DB0 )
          break;
        LOBYTE(src) = *(_QWORD *)&v430[v22];
        *((_QWORD *)&src + 1) = v24;
        if ( !(_BYTE)v23 )
        {
          v408 = (__int64)v25;
          if ( sub_1F58D20((__int64)&src) )
          {
            v444.m128i_i8[0] = 0;
            v444.m128i_i64[1] = 0;
            LOBYTE(v425) = 0;
            sub_1F426C0((__int64)a2, v408, (unsigned int)src, v24, (__int64)&v444, (unsigned int *)&v439, &v425);
            v19 = v6[2].m128i_u32[0];
            v27 = v425;
            v17 = v6[1].m128i_u64[0];
            v20 = (_QWORD *)a2->m128i_i64[0];
            m128i_i64 = (__int64)v298;
            goto LABEL_15;
          }
          m128i_i64 = (__int64)a2;
          sub_1F40D10((__int64)&v444, (__int64)a2, v408, src, *((__int64 *)&src + 1));
          LOBYTE(v439) = v444.m128i_i8[8];
          *((_QWORD *)&v439 + 1) = v445.m128i_i64[0];
          if ( v444.m128i_i8[8] )
          {
            v27 = a2[72].m128i_i8[v444.m128i_u8[8] + 3];
          }
          else
          {
            v380 = v445.m128i_i64[0];
            if ( sub_1F58D20((__int64)&v439) )
            {
              m128i_i64 = v408;
              v444.m128i_i8[0] = 0;
              v444.m128i_i64[1] = 0;
              LOBYTE(v423) = 0;
              sub_1F426C0((__int64)a2, v408, (unsigned int)v439, v380, (__int64)&v444, (unsigned int *)&v425, &v423);
              v27 = v423;
            }
            else
            {
              sub_1F40D10((__int64)&v444, (__int64)a2, v408, v439, *((__int64 *)&v439 + 1));
              v182 = v340;
              LOBYTE(v182) = v444.m128i_i8[8];
              m128i_i64 = v408;
              v340 = v182;
              v27 = sub_1D5E9F0((__int64)a2, v408, (unsigned int)v182, v445.m128i_i64[0]);
            }
          }
          goto LABEL_167;
        }
        m128i_i64 = (__int64)a2;
        v27 = a2[72].m128i_i8[(unsigned __int8)v23 + 3];
LABEL_15:
        v28 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v20[49];
        v29 = *(_QWORD *)v17;
        if ( v28 == sub_1F42F80 )
        {
          LOBYTE(src) = v23;
          *((_QWORD *)&src + 1) = v24;
          if ( (_BYTE)v23 )
          {
            v30 = a2[65].m128i_u8[(unsigned __int8)v23];
          }
          else
          {
            v379 = v27;
            v407 = v29;
            if ( sub_1F58D20((__int64)&src) )
            {
              m128i_i64 = v407;
              v444.m128i_i8[0] = 0;
              v444.m128i_i64[1] = 0;
              LOBYTE(v425) = 0;
              v30 = sub_1F426C0((__int64)a2, v407, (unsigned int)src, v24, (__int64)&v444, (unsigned int *)&v439, &v425);
              v19 = v293;
              v27 = v379;
            }
            else
            {
              v365 = v407;
              v409 = sub_1F58D40((__int64)&src);
              v439 = src;
              v377 = src;
              if ( sub_1F58D20((__int64)&v439) )
              {
                v444.m128i_i8[0] = 0;
                v444.m128i_i64[1] = 0;
                LOBYTE(v423) = 0;
                sub_1F426C0(
                  (__int64)a2,
                  v365,
                  (unsigned int)v439,
                  *((__int64 *)&v439 + 1),
                  (__int64)&v444,
                  (unsigned int *)&v425,
                  &v423);
                m128i_i64 = (__int64)v297;
                v144 = v423;
              }
              else
              {
                sub_1F40D10((__int64)&v444, (__int64)a2, v365, v377, *((__int64 *)&v377 + 1));
                v143 = v342;
                LOBYTE(v143) = v444.m128i_i8[8];
                m128i_i64 = v365;
                v342 = v143;
                v144 = sub_1D5E9F0((__int64)a2, v365, (unsigned int)v143, v445.m128i_i64[0]);
              }
              v145 = sub_2045180(v144);
              v17 = v145;
              v30 = (v145 + v409 - 1) / v145;
            }
          }
        }
        else
        {
          v410 = v27;
          m128i_i64 = *(_QWORD *)v17;
          v30 = v28((__int64)a2, v29, (unsigned int)v19, (unsigned int)v23, v24);
          v27 = v410;
        }
        if ( v30 )
        {
          v31 = v24;
          v32 = v6[137].m128i_u32[2];
          v399 = v22;
          v33 = v30;
          v34 = v23;
          v35 = v27;
          v36 = 0;
          v18 = v31;
          do
          {
            v37 = v6[1].m128i_i8[8];
            v444.m128i_i8[8] = v35;
            v444.m128i_i64[0] = 0;
            v445.m128i_i64[0] = v34;
            v445.m128i_i64[1] = v18;
            m128i_i64 = (v37 & 0x20) != 0;
            v446.m128i_i8[0] = (v37 & 0x20) != 0;
            if ( (v37 & 1) != 0 )
              v444.m128i_i8[0] |= 2u;
            if ( (v37 & 2) != 0 )
              v444.m128i_i8[0] |= 1u;
            v17 = v37 & 8;
            if ( (_DWORD)v17 )
              v444.m128i_i8[0] |= 4u;
            if ( v6[137].m128i_i32[3] <= (unsigned int)v32 )
            {
              m128i_i64 = (__int64)v6[138].m128i_i64;
              v364 = v18;
              v376 = v34;
              sub_16CD150((__int64)v6[137].m128i_i64, &v6[138], 0, 48, v18, v19);
              v32 = v6[137].m128i_u32[2];
              v18 = v364;
              v34 = v376;
            }
            ++v36;
            v38 = (__m128i *)(v6[137].m128i_i64[0] + 48 * v32);
            *v38 = _mm_load_si128(&v444);
            v38[1] = _mm_load_si128(&v445);
            v38[2] = _mm_load_si128(&v446);
            v32 = (unsigned int)(v6[137].m128i_i32[2] + 1);
            v6[137].m128i_i32[2] = v32;
          }
          while ( v33 != v36 );
          v22 = v399;
        }
        v22 += 16;
        v20 = (_QWORD *)a2->m128i_i64[0];
        if ( v353 == v22 )
        {
          v316 = 1;
          p_src = &src;
          v306 = 0;
          v305 = 0;
          v307 = -100;
          goto LABEL_32;
        }
      }
      m128i_i64 = *(_QWORD *)v17;
      v27 = v26((__int64)a2, *(_QWORD *)v17, (unsigned int)v19, (unsigned int)v23, *(_QWORD *)&v430[v22 + 8]);
LABEL_167:
      v19 = v6[2].m128i_u32[0];
      v17 = v6[1].m128i_u64[0];
      v20 = (_QWORD *)a2->m128i_i64[0];
      goto LABEL_15;
    }
    v316 = 1;
    v305 = 0;
    v307 = -100;
LABEL_32:
    v39 = (__int64 (*)())v20[145];
    if ( v39 == sub_1D45FE0 )
    {
LABEL_33:
      v40 = v6[3].m128i_i64[1];
      v41 = v6[4].m128i_i64[0];
      goto LABEL_34;
    }
  }
  else
  {
    m128i_i64 = v6[2].m128i_u32[0];
    v316 = ((__int64 (__fastcall *)(__m128i *, __int64, _QWORD, bool, _BYTE **, _QWORD))v21)(
             a2,
             m128i_i64,
             *(_QWORD *)(v6[5].m128i_i64[0] + 32),
             (v6[1].m128i_i8[8] & 4) != 0,
             &v441,
             *(_QWORD *)v6[1].m128i_i64[0]);
    if ( v316 )
    {
      v20 = (_QWORD *)a2->m128i_i64[0];
      goto LABEL_10;
    }
    v255 = v6[1].m128i_i64[0];
    v256 = (unsigned int)sub_15A9FE0(v344, v255);
    v257 = (v256 + ((unsigned __int64)(sub_127FA20(v344, v255) + 7) >> 3) - 1) / v256 * v256;
    v258 = sub_15AAE50(v344, v6[1].m128i_i64[0]);
    v259 = v258;
    v307 = sub_1E090F0(*(_QWORD *)(*(_QWORD *)(v6[5].m128i_i64[0] + 32) + 56LL), v257, v258, 0, 0, 0);
    v260 = sub_1646BA0((__int64 *)v6[1].m128i_i64[0], *(_DWORD *)(v344 + 4));
    v261 = (_QWORD *)v6[5].m128i_i64[0];
    v262 = v260;
    v263 = 8 * sub_15A9520(v344, *(_DWORD *)(v344 + 4));
    if ( v263 == 32 )
    {
      v264 = 5;
    }
    else if ( v263 > 0x20 )
    {
      v264 = 6;
      if ( v263 != 64 )
      {
        v264 = 7;
        if ( v263 != 128 )
          v264 = 0;
      }
    }
    else
    {
      v264 = 3;
      if ( v263 != 8 )
        v264 = 4 * (v263 == 16);
    }
    v265 = sub_1D299D0(v261, v307, v264, 0, 0);
    v445.m128i_i64[1] = v262;
    v306 = v266;
    v445.m128i_i32[0] = v266;
    m128i_i64 = v6[3].m128i_i64[1];
    v446.m128i_i16[0] = 8;
    v305 = (__int64)v265;
    v309 = v266;
    v444.m128i_i64[1] = (__int64)v265;
    v444.m128i_i64[0] = 0;
    v446.m128i_i16[1] = v259;
    sub_2056800((__int64)&v6[3].m128i_i64[1], (__m128i *)m128i_i64, &v444);
    ++v6[1].m128i_i32[3];
    v267 = sub_1643270(*(_QWORD **)v6[1].m128i_i64[0]);
    v6[1].m128i_i8[9] = 0;
    v6[1].m128i_i64[0] = v267;
    v20 = (_QWORD *)a2->m128i_i64[0];
    v39 = *(__int64 (**)())(a2->m128i_i64[0] + 1160);
    if ( v39 == sub_1D45FE0 )
      goto LABEL_33;
  }
  v268 = ((__int64 (__fastcall *)(__m128i *, __int64, _QWORD *, unsigned __int64, __int64, __int64))v39)(
           a2,
           m128i_i64,
           v20,
           v17,
           v18,
           v19);
  v41 = v6[4].m128i_i64[0];
  v155 = v268 == 0;
  v40 = v6[3].m128i_i64[1];
  if ( v155 )
  {
LABEL_34:
    v42 = -858993459 * ((v41 - v40) >> 3);
    goto LABEL_35;
  }
  v17 = 0xCCCCCCCCCCCCCCCDLL;
  v269 = 0xCCCCCCCCCCCCCCCDLL * ((v41 - v40) >> 3);
  if ( (_DWORD)v269 )
  {
    v270 = 0;
    v271 = 40LL * (unsigned int)v269;
    do
    {
      while ( (*(_BYTE *)(v40 + v270 + 33) & 2) == 0 )
      {
        v270 += 40;
        if ( v271 == v270 )
          goto LABEL_320;
      }
      v444.m128i_i64[0] = 0;
      v445.m128i_i64[1] = 0;
      v446.m128i_i8[0] = 0;
      v272 = 8 * sub_15A9520(v344, 0);
      if ( v272 == 32 )
      {
        v273 = 5;
      }
      else if ( v272 > 0x20 )
      {
        v273 = 6;
        if ( v272 != 64 )
        {
          v273 = 0;
          if ( v272 == 128 )
            v273 = 7;
        }
      }
      else
      {
        v273 = 3;
        if ( v272 != 8 )
          v273 = 4 * (v272 == 16);
      }
      v444.m128i_i8[8] = v273;
      v274 = 8 * sub_15A9520(v344, 0);
      if ( v274 == 32 )
      {
        v275 = 5;
      }
      else if ( v274 > 0x20 )
      {
        v275 = 6;
        if ( v274 != 64 )
        {
          v275 = 0;
          if ( v274 == 128 )
            v275 = 7;
        }
      }
      else
      {
        v275 = 3;
        if ( v274 != 8 )
          v275 = 4 * (v274 == 16);
      }
      v445.m128i_i8[0] = v275;
      v276 = v6[137].m128i_u32[2];
      v444.m128i_i8[1] |= 8u;
      if ( (unsigned int)v276 >= v6[137].m128i_i32[3] )
      {
        sub_16CD150((__int64)v6[137].m128i_i64, &v6[138], 0, 48, v18, v19);
        v276 = v6[137].m128i_u32[2];
      }
      v270 += 40;
      v277 = (__m128i *)(v6[137].m128i_i64[0] + 48 * v276);
      *v277 = _mm_load_si128(&v444);
      v277[1] = _mm_load_si128(&v445);
      v277[2] = _mm_load_si128(&v446);
      v40 = v6[3].m128i_i64[1];
      ++v6[137].m128i_i32[2];
    }
    while ( v271 != v270 );
LABEL_320:
    v42 = -858993459 * ((v6[4].m128i_i64[0] - v40) >> 3);
LABEL_35:
    v6[7].m128i_i32[2] = 0;
    v6[104].m128i_i32[2] = 0;
    if ( !v42 )
      goto LABEL_174;
    v312 = 0;
    v43 = v6;
    v310 = v42;
    v44 = (unsigned int *)&src;
    while ( 1 )
    {
      *(_QWORD *)&v439 = v440;
      *((_QWORD *)&v439 + 1) = 0x400000000LL;
      v351 = 40 * v312;
      sub_20C7CE0(a2, v344, *(_QWORD *)(v40 + 40 * v312 + 24), &v439, 0, 0);
      v45 = 40 * v312 + v43[3].m128i_i64[1];
      v311 = *(_QWORD *)(v45 + 24);
      if ( (*(_BYTE *)(v45 + 32) & 0x20) != 0 )
        v311 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + 24LL);
      LOBYTE(v319) = 0;
      v46 = *(__int64 (**)())(a2->m128i_i64[0] + 1272);
      if ( v46 != sub_1FD3430 )
        LOBYTE(v319) = ((__int64 (__fastcall *)(__m128i *, __int64, _QWORD, bool))v46)(
                         a2,
                         v311,
                         v43[2].m128i_u32[0],
                         (v43[1].m128i_i8[8] & 4) != 0);
      if ( DWORD2(v439) )
        break;
LABEL_101:
      if ( (__int64 *)v439 != v440 )
        _libc_free(v439);
      if ( v310 == ++v312 )
      {
        v6 = v43;
        p_src = v44;
        goto LABEL_174;
      }
      v40 = v43[3].m128i_i64[1];
    }
    v318 = DWORD2(v439);
    v47 = v44;
    v354 = 0;
    v317 = DWORD2(v439) - 1;
    while ( 2 )
    {
      v48 = (_QWORD **)v43[1].m128i_i64[0];
      v418 = _mm_loadu_si128((const __m128i *)(v439 + 16 * v354));
      v49 = sub_1F58E60((__int64)&v418, *v48);
      v50 = v43[3].m128i_i64[1] + v351;
      v346 = *(_DWORD *)(v50 + 16) + v354;
      v373 = *(_QWORD *)(v50 + 8);
      v51 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a2->m128i_i64[0] + 400);
      v447[0] = &v448;
      v447[1] = 0x1000000000LL;
      v450 = 0;
      v449[0] = &v450;
      v445.m128i_i64[1] = (__int64)&v446.m128i_i64[1];
      v451[0] = &v452;
      v446.m128i_i64[0] = 0x800000000LL;
      v454[0] = &v455;
      v451[1] = 0x800000000LL;
      v454[1] = 0x800000000LL;
      v449[1] = 0;
      v453 = 0;
      sub_15A9210((__int64)&v444);
      sub_2240AE0(v449, v344 + 192);
      v444.m128i_i8[0] = *(_BYTE *)v344;
      *(__int64 *)((char *)v444.m128i_i64 + 4) = *(_QWORD *)(v344 + 4);
      v444.m128i_i32[3] = *(_DWORD *)(v344 + 12);
      v445.m128i_i32[0] = *(_DWORD *)(v344 + 16);
      sub_2044F40((__int64)&v445.m128i_i64[1], v344 + 24, v52, v53, v54, v55);
      sub_2044E60((__int64)v447, v344 + 48, v56, v57, v58, v59);
      sub_2044D80((__int64)v451, v344 + 224, v60, v61, v62, v63);
      sub_2045020((__int64)v454, v344 + 408, v64, v65, v66, v67);
      if ( v51 == sub_1F3CCF0 )
        v68 = sub_15A9FE0((__int64)&v444, v49);
      else
        v68 = v51((__int64)a2, v49, (__int64)&v444);
      sub_15A93E0(&v444);
      v70 = v43[3].m128i_i64[1] + v351;
      v343 = 0;
      v71 = *(_BYTE *)(v70 + 32);
      v341 = 0;
      v337 = (v71 & 2) != 0;
      v334 = v71 & 1;
      v339 = (v71 & 4) != 0;
      if ( (v71 & 4) != 0 && v43[2].m128i_i32[0] == 80 && *(_BYTE *)(v311 + 8) == 13 )
      {
        v341 = (v71 & 4) != 0;
        v343 = (_DWORD)v354 == 0;
      }
      v72 = *(_BYTE *)(v70 + 33);
      v73 = (v71 & 0x40) != 0;
      v333 = (v71 & 8) != 0;
      v332 = v72 & 1;
      v331 = (v72 & 2) != 0;
      if ( (v71 & 0x20) != 0 || (v71 & 0x40) != 0 )
      {
        v113 = 1;
        v401 = *(_QWORD *)(*(_QWORD *)(v70 + 24) + 24LL);
        v114 = sub_15A9FE0(v344, v401);
        v115 = v401;
        v116 = v114;
        v117 = v401;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v117 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v120 = *(_QWORD *)(v117 + 32);
              v117 = *(_QWORD *)(v117 + 24);
              v113 *= v120;
              continue;
            case 1:
              v118 = 16;
              goto LABEL_110;
            case 2:
              v118 = 32;
              goto LABEL_110;
            case 3:
            case 9:
              v118 = 64;
              goto LABEL_110;
            case 4:
              v118 = 80;
              goto LABEL_110;
            case 5:
            case 6:
              v118 = 128;
              goto LABEL_110;
            case 7:
              v391 = v401;
              v121 = 0;
              v402 = v116;
              goto LABEL_118;
            case 0xB:
              v118 = *(_DWORD *)(v117 + 8) >> 8;
              goto LABEL_110;
            case 0xD:
              v393 = v401;
              v404 = v116;
              v127 = (_QWORD *)sub_15A9930(v344, v117);
              v116 = v404;
              v115 = v393;
              v118 = 8LL * *v127;
              goto LABEL_110;
            case 0xE:
              v363 = v401;
              v384 = v116;
              v403 = *(_QWORD *)(v117 + 32);
              v392 = *(_QWORD *)(v117 + 24);
              v123 = sub_15A9FE0(v344, v392);
              v115 = v363;
              v116 = v384;
              v124 = 1;
              v125 = v392;
              v126 = v123;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v125 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v183 = *(_QWORD *)(v125 + 32);
                    v125 = *(_QWORD *)(v125 + 24);
                    v124 *= v183;
                    continue;
                  case 1:
                    v128 = 16;
                    goto LABEL_127;
                  case 2:
                    v128 = 32;
                    goto LABEL_127;
                  case 3:
                  case 9:
                    v128 = 64;
                    goto LABEL_127;
                  case 4:
                    v128 = 80;
                    goto LABEL_127;
                  case 5:
                  case 6:
                    v128 = 128;
                    goto LABEL_127;
                  case 7:
                    v328 = v363;
                    v184 = 0;
                    v367 = v384;
                    v385 = v126;
                    v394 = v124;
                    goto LABEL_213;
                  case 0xB:
                    v128 = *(_DWORD *)(v125 + 8) >> 8;
                    goto LABEL_127;
                  case 0xD:
                    v330 = v363;
                    v369 = v384;
                    v387 = v126;
                    v396 = v124;
                    v187 = (_QWORD *)sub_15A9930(v344, v125);
                    v124 = v396;
                    v126 = v387;
                    v116 = v369;
                    v115 = v330;
                    v128 = 8LL * *v187;
                    goto LABEL_127;
                  case 0xE:
                    v322 = v363;
                    v324 = v384;
                    v326 = v126;
                    v329 = v124;
                    v368 = *(_QWORD *)(v125 + 24);
                    v395 = *(_QWORD *)(v125 + 32);
                    v386 = (unsigned int)sub_15A9FE0(v344, v368);
                    v186 = sub_127FA20(v344, v368);
                    v124 = v329;
                    v126 = v326;
                    v116 = v324;
                    v115 = v322;
                    v128 = 8 * v395 * v386 * ((v386 + ((unsigned __int64)(v186 + 7) >> 3) - 1) / v386);
                    goto LABEL_127;
                  case 0xF:
                    v328 = v363;
                    v367 = v384;
                    v385 = v126;
                    v184 = *(_DWORD *)(v125 + 8) >> 8;
                    v394 = v124;
LABEL_213:
                    v185 = sub_15A9520(v344, v184);
                    v124 = v394;
                    v126 = v385;
                    v116 = v367;
                    v115 = v328;
                    v128 = (unsigned int)(8 * v185);
LABEL_127:
                    v69 = v128 * v124;
                    v118 = 8 * v403 * v126 * ((v126 + ((unsigned __int64)(v69 + 7) >> 3) - 1) / v126);
                    break;
                }
                goto LABEL_110;
              }
            case 0xF:
              v391 = v401;
              v402 = v116;
              v121 = *(_DWORD *)(v117 + 8) >> 8;
LABEL_118:
              v122 = sub_15A9520(v344, v121);
              v116 = v402;
              v115 = v391;
              v118 = (unsigned int)(8 * v122);
LABEL_110:
              v70 = v43[3].m128i_i64[1] + v351;
              v390 = v116 * ((v116 + ((unsigned __int64)(v118 * v113 + 7) >> 3) - 1) / v116);
              v119 = *(unsigned __int16 *)(v70 + 34);
              if ( (_WORD)v119
                || (v119 = (*(__int64 (__fastcall **)(__m128i *, __int64, __int64))(a2->m128i_i64[0] + 376))(
                             a2,
                             v115,
                             v344),
                    v70 = v43[3].m128i_i64[1] + v351,
                    v119) )
              {
                _BitScanReverse(&v119, v119);
                v323 = -(v119 ^ 0x1F) & 0xF;
              }
              else
              {
                v323 = 0;
              }
              v325 = v73;
              v73 = 1;
              break;
          }
          break;
        }
      }
      else
      {
        v325 = 0;
        v390 = 0;
        v323 = 0;
      }
      v382 = 0;
      v327 = (*(_BYTE *)(v70 + 32) & 0x10) != 0;
      if ( v68 )
      {
        _BitScanReverse(&v68, v68);
        v382 = -(v68 ^ 0x1F) & 0x1F;
      }
      v74 = (__int64 *)v43[1].m128i_i64[0];
      v75 = v43[2].m128i_u32[0];
      v76 = a2->m128i_i64[0];
      v77 = *v74;
      v78 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(a2->m128i_i64[0] + 384);
      if ( v78 != sub_1F42DB0 )
      {
        v141 = v78((__int64)a2, v77, v75, v418.m128i_u32[0], v418.m128i_i64[1]);
        v76 = a2->m128i_i64[0];
        v75 = v43[2].m128i_u32[0];
        v361 = v141;
        v74 = (__int64 *)v43[1].m128i_i64[0];
        goto LABEL_55;
      }
      v79 = v418.m128i_i64[1];
      LOBYTE(v419) = v418.m128i_i8[0];
      v420 = v418.m128i_i64[1];
      if ( v418.m128i_i8[0] )
      {
        v361 = a2[72].m128i_i8[v418.m128i_u8[0] + 3];
        goto LABEL_55;
      }
      if ( sub_1F58D20((__int64)&v419) )
      {
        v444.m128i_i8[0] = 0;
        v444.m128i_i64[1] = 0;
        LOBYTE(v425) = 0;
        sub_1F426C0((__int64)a2, v77, (unsigned int)v419, v79, (__int64)&v444, v47, &v425);
        v76 = a2->m128i_i64[0];
        v75 = v43[2].m128i_u32[0];
        v361 = v425;
        v74 = (__int64 *)v43[1].m128i_i64[0];
        goto LABEL_55;
      }
      sub_1F40D10((__int64)&v444, (__int64)a2, v77, v419, v420);
      v129 = v444.m128i_u8[8];
      v130 = v445.m128i_i64[0];
      LOBYTE(v421) = v444.m128i_i8[8];
      *((_QWORD *)&v421 + 1) = v445.m128i_i64[0];
      if ( v444.m128i_i8[8] )
        goto LABEL_151;
      if ( sub_1F58D20((__int64)&v421) )
      {
        v444.m128i_i8[0] = 0;
        v444.m128i_i64[1] = 0;
        LOBYTE(v425) = 0;
        sub_1F426C0((__int64)a2, v77, (unsigned int)v421, v130, (__int64)&v444, v47, &v425);
      }
      else
      {
        sub_1F40D10((__int64)&v444, (__int64)a2, v77, v421, *((__int64 *)&v421 + 1));
        v129 = v444.m128i_u8[8];
        v131 = v445.m128i_i64[0];
        LOBYTE(v423) = v444.m128i_i8[8];
        *((_QWORD *)&v423 + 1) = v445.m128i_i64[0];
        if ( v444.m128i_i8[8] )
          goto LABEL_151;
        if ( !sub_1F58D20((__int64)&v423) )
        {
          sub_1F40D10((__int64)&v444, (__int64)a2, v77, v423, *((__int64 *)&v423 + 1));
          v129 = v444.m128i_u8[8];
          v132 = v445.m128i_i64[0];
          LOBYTE(v425) = v444.m128i_i8[8];
          *((_QWORD *)&v425 + 1) = v445.m128i_i64[0];
          if ( !v444.m128i_i8[8] )
          {
            if ( sub_1F58D20((__int64)&v425) )
            {
              v444.m128i_i8[0] = 0;
              v444.m128i_i64[1] = 0;
              LOBYTE(v417) = 0;
              sub_1F426C0((__int64)a2, v77, (unsigned int)v425, v132, (__int64)&v444, v47, &v417);
              LODWORD(v69) = v295;
              v361 = v417;
            }
            else
            {
              sub_1F40D10((__int64)&v444, (__int64)a2, v77, v425, *((__int64 *)&v425 + 1));
              LOBYTE(src) = v444.m128i_i8[8];
              *((_QWORD *)&src + 1) = v445.m128i_i64[0];
              if ( v444.m128i_i8[8] )
              {
                v134 = a2[72].m128i_i8[v444.m128i_u8[8] + 3];
              }
              else if ( sub_1F58D20((__int64)v47) )
              {
                v444.m128i_i8[0] = 0;
                v444.m128i_i64[1] = 0;
                v416 = 0;
                sub_1F426C0((__int64)a2, v77, (unsigned int)src, *((__int64 *)&src + 1), (__int64)&v444, &v417, &v416);
                v134 = v416;
              }
              else
              {
                sub_1F40D10((__int64)&v444, (__int64)a2, v77, src, *((__int64 *)&src + 1));
                v133 = v299;
                LOBYTE(v133) = v444.m128i_i8[8];
                v299 = v133;
                v134 = sub_1D5E9F0((__int64)a2, v77, (unsigned int)v133, v445.m128i_i64[0]);
              }
              v361 = v134;
            }
LABEL_152:
            v75 = v43[2].m128i_u32[0];
            v74 = (__int64 *)v43[1].m128i_i64[0];
            v76 = a2->m128i_i64[0];
LABEL_55:
            v80 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(v76 + 392);
            v81 = *v74;
            if ( v80 == sub_1F42F80 )
            {
              v82 = v418.m128i_i64[1];
              LOBYTE(v421) = v418.m128i_i8[0];
              *((_QWORD *)&v421 + 1) = v418.m128i_i64[1];
              if ( v418.m128i_i8[0] )
              {
                v83 = a2[65].m128i_u8[v418.m128i_u8[0]];
                goto LABEL_58;
              }
              if ( sub_1F58D20((__int64)&v421) )
              {
                v444.m128i_i8[0] = 0;
                v444.m128i_i64[1] = 0;
                LOBYTE(v425) = 0;
                v83 = sub_1F426C0((__int64)a2, v81, (unsigned int)v421, v82, (__int64)&v444, v47, &v425);
                goto LABEL_58;
              }
              v135 = sub_1F58D40((__int64)&v421);
              v423 = v421;
              v136 = v421;
              if ( sub_1F58D20((__int64)&v423) )
              {
                v444.m128i_i8[0] = 0;
                v444.m128i_i64[1] = 0;
                LOBYTE(v425) = 0;
                sub_1F426C0((__int64)a2, v81, (unsigned int)v423, *((__int64 *)&v423 + 1), (__int64)&v444, v47, &v425);
                v139 = v425;
                goto LABEL_147;
              }
              sub_1F40D10((__int64)&v444, (__int64)a2, v81, v136, *((__int64 *)&v136 + 1));
              v137 = v444.m128i_u8[8];
              LOBYTE(v425) = v444.m128i_i8[8];
              *((_QWORD *)&v425 + 1) = v445.m128i_i64[0];
              if ( v444.m128i_i8[8] )
                goto LABEL_158;
              v405 = v445.m128i_i64[0];
              if ( sub_1F58D20((__int64)&v425) )
              {
                v444.m128i_i8[0] = 0;
                v444.m128i_i64[1] = 0;
                LOBYTE(v419) = 0;
                sub_1F426C0((__int64)a2, v81, (unsigned int)v425, v405, (__int64)&v444, v47, &v419);
                v139 = v419;
                goto LABEL_147;
              }
              sub_1F40D10((__int64)&v444, (__int64)a2, v81, v425, *((__int64 *)&v425 + 1));
              v137 = v444.m128i_u8[8];
              LOBYTE(src) = v444.m128i_i8[8];
              *((_QWORD *)&src + 1) = v445.m128i_i64[0];
              if ( v444.m128i_i8[8] )
              {
LABEL_158:
                v139 = a2[72].m128i_i8[v137 + 3];
                goto LABEL_147;
              }
              v406 = v445.m128i_i64[0];
              if ( sub_1F58D20((__int64)v47) )
              {
                v444.m128i_i8[0] = 0;
                v444.m128i_i64[1] = 0;
                LOBYTE(v417) = 0;
                sub_1F426C0((__int64)a2, v81, (unsigned int)src, v406, (__int64)&v444, (unsigned int *)&v419, &v417);
                v139 = v417;
              }
              else
              {
                sub_1F40D10((__int64)&v444, (__int64)a2, v81, src, *((__int64 *)&src + 1));
                v138 = v300;
                LOBYTE(v138) = v444.m128i_i8[8];
                v300 = v138;
                v139 = sub_1D5E9F0((__int64)a2, v81, (unsigned int)v138, v445.m128i_i64[0]);
              }
LABEL_147:
              v140 = sub_2045180(v139);
              v83 = (v140 + v135 - 1) / v140;
            }
            else
            {
              v83 = v80((__int64)a2, v81, v75, v418.m128i_u32[0], v418.m128i_i64[1]);
            }
LABEL_58:
            v84 = &v445;
            v400 = v83;
            v444.m128i_i64[0] = (__int64)&v445;
            v444.m128i_i64[1] = 0x400000000LL;
            if ( v83 > 4 )
            {
              sub_16CD150((__int64)&v444, &v445, v83, 16, (int)&v445, v69);
              v84 = (__m128i *)v444.m128i_i64[0];
            }
            v444.m128i_i32[2] = v83;
            v85 = &v84[v83];
            if ( v85 != v84 )
            {
              do
              {
                if ( v84 )
                {
                  v84->m128i_i64[0] = 0;
                  v84->m128i_i32[2] = 0;
                }
                ++v84;
              }
              while ( v85 != v84 );
              v84 = (__m128i *)v444.m128i_i64[0];
            }
            v86 = v43[3].m128i_i64[1] + v351;
            v87 = 142;
            v88 = *(_BYTE *)(v86 + 32);
            v321 = v88 & 1;
            if ( (v88 & 1) == 0 )
              v87 = ((v88 & 2) == 0) + 143;
            v89 = v346;
            if ( v88 >= 0 )
              goto LABEL_106;
            v90 = *(_QWORD *)(v373 + 40) + 16LL * v346;
            v91 = *(_BYTE *)v90;
            v92 = *(_QWORD *)(v90 + 8);
            LOBYTE(src) = v91;
            *((_QWORD *)&src + 1) = v92;
            if ( v91 )
            {
              if ( (unsigned __int8)(v91 - 14) > 0x5Fu )
                goto LABEL_70;
LABEL_106:
              v96 = 0;
            }
            else
            {
              v313 = v346;
              v315 = v84;
              v349 = v87;
              v112 = sub_1F58D20((__int64)v47);
              v87 = v349;
              v84 = v315;
              v89 = v313;
              if ( v112 )
                goto LABEL_106;
LABEL_70:
              if ( !v316 )
                goto LABEL_106;
              v320 = v87;
              v347 = v83 * sub_2045180(v361);
              if ( v418.m128i_i8[0] )
              {
                v95 = sub_2045180(v418.m128i_i8[0]);
              }
              else
              {
                v308 = v93;
                v314 = v94;
                v95 = sub_1F58D40((__int64)&v418);
                v89 = v308;
                v84 = v314;
              }
              v87 = v320;
              v96 = v316;
              if ( v347 != v95 )
              {
                v96 = 0;
                if ( v320 != 144 && (v43[1].m128i_i8[8] & 1) == v321 )
                  v96 = ((v43[1].m128i_i8[8] & 2) != 0) == ((*(_BYTE *)(v86 + 32) & 2) != 0);
              }
            }
            v97 = v43[2].m128i_i32[0];
            v98 = v43[5].m128i_i64[0];
            BYTE4(src) = 1;
            LODWORD(src) = v97;
            sub_204A2F0(
              v98,
              (__int64)&v43[5].m128i_i64[1],
              v373,
              v89,
              (unsigned __int64)v84,
              v83,
              a4,
              *(double *)si128.m128i_i64,
              a6,
              v361,
              v43[6].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL,
              (__int64)v47,
              v87);
            if ( !v83 )
            {
LABEL_94:
              if ( (_BYTE)v319 && v317 == (_DWORD)v354 )
              {
                v142 = v43[7].m128i_i64[0] + 16 * (3LL * v43[7].m128i_u32[2] - 3);
                *(_BYTE *)(v142 + 3) |= 1u;
              }
              if ( (__m128i *)v444.m128i_i64[0] != &v445 )
                _libc_free(v444.m128i_u64[0]);
              if ( v318 == ++v354 )
              {
                v44 = v47;
                goto LABEL_101;
              }
              continue;
            }
            v348 = v83 - 1;
            v362 = v83 > 1;
            v99 = (const __m128i *)v47;
            v100 = 0;
            v101 = v99;
            v383 = ((v319 << 25)
                  | (v382 << 19)
                  | (v323 << 15)
                  | (v343 << 13)
                  | (v331 << 11)
                  | (v325 << 8)
                  | (v96 << 6)
                  | (32 * v327)
                  | (16 * v73)
                  | (8 * v333)
                  | (4 * v339)
                  | v337
                  | (2 * v334)
                  | (v332 << 10)
                  | (v341 << 12))
                 & 0x7FFFFFF;
            while ( 2 )
            {
              v109 = *(_QWORD *)(*(_QWORD *)(16 * v100 + v444.m128i_i64[0]) + 40LL)
                   + 16LL * *(unsigned int *)(16 * v100 + v444.m128i_i64[0] + 8);
              v110 = *(_BYTE *)v109;
              v111 = *(_QWORD *)(v109 + 8);
              LOBYTE(v425) = v110;
              *((_QWORD *)&v425 + 1) = v111;
              if ( v110 )
              {
                v102 = sub_2045180(v110);
              }
              else
              {
                v374 = 16 * v100 + v444.m128i_i64[0];
                v102 = sub_1F58D40((__int64)&v425);
                v103 = (unsigned int *)v374;
              }
              v17 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v103 + 40LL) + 16LL * v103[2]);
              v436 = v43[1].m128i_i32[3] > (unsigned int)v312;
              v437 = v312;
              v438 = v100 * ((unsigned int)(v102 + 7) >> 3);
              LODWORD(src) = v383 | src & 0xF8000000;
              v434 = v418.m128i_i8[0];
              DWORD1(src) = v390;
              BYTE8(src) = v17;
              v435 = v418.m128i_i64[1];
              if ( !(_DWORD)v100 && v362 )
              {
                LOBYTE(src) = src | 0x80;
                goto LABEL_83;
              }
              if ( v100 && (BYTE2(src) = BYTE2(src) & 7 | 8, (_DWORD)v100 == v348) )
              {
                BYTE1(src) |= 2u;
                v104 = v43[7].m128i_u32[2];
                if ( (unsigned int)v104 >= v43[7].m128i_i32[3] )
                {
LABEL_92:
                  sub_16CD150((__int64)v43[7].m128i_i64, &v43[8], 0, 48, v18, v19);
                  v104 = v43[7].m128i_u32[2];
                }
              }
              else
              {
LABEL_83:
                v104 = v43[7].m128i_u32[2];
                if ( (unsigned int)v104 >= v43[7].m128i_i32[3] )
                  goto LABEL_92;
              }
              si128 = _mm_load_si128(v101);
              v105 = v43[7].m128i_i64[0] + 48 * v104;
              *(__m128i *)v105 = si128;
              a6 = _mm_load_si128(v101 + 1);
              *(__m128i *)(v105 + 16) = a6;
              *(__m128i *)(v105 + 32) = _mm_load_si128(v101 + 2);
              v106 = v444.m128i_i64[0];
              v107 = v43[104].m128i_u32[2];
              ++v43[7].m128i_i32[2];
              v108 = (const __m128i *)(16 * v100 + v106);
              if ( (unsigned int)v107 >= v43[104].m128i_i32[3] )
              {
                v375 = v108;
                sub_16CD150((__int64)v43[104].m128i_i64, &v43[105], 0, 16, v18, v19);
                v107 = v43[104].m128i_u32[2];
                v108 = v375;
              }
              a4 = _mm_loadu_si128(v108);
              ++v100;
              *(__m128i *)(v43[104].m128i_i64[0] + 16 * v107) = a4;
              ++v43[104].m128i_i32[2];
              if ( v400 == v100 )
              {
                v47 = (unsigned int *)v101;
                goto LABEL_94;
              }
              continue;
            }
          }
LABEL_151:
          v361 = a2[72].m128i_i8[v129 + 3];
          goto LABEL_152;
        }
        v444.m128i_i8[0] = 0;
        v444.m128i_i64[1] = 0;
        LOBYTE(v425) = 0;
        sub_1F426C0((__int64)a2, v77, (unsigned int)v423, v131, (__int64)&v444, v47, &v425);
      }
      break;
    }
    v361 = v425;
    goto LABEL_152;
  }
  v6[7].m128i_i32[2] = 0;
  v6[104].m128i_i32[2] = 0;
LABEL_174:
  *(_QWORD *)&src = &v434;
  *((_QWORD *)&src + 1) = 0x400000000LL;
  v146 = (*(__int64 (__fastcall **)(__m128i *, __m128i *, _OWORD *, unsigned __int64, __int64, __int64))(a2->m128i_i64[0] + 1200))(
           a2,
           v6,
           p_src,
           v17,
           v18,
           v19);
  v415 = v149;
  v150 = DWORD2(src);
  v6->m128i_i64[0] = v146;
  v151 = v150;
  v6->m128i_i32[2] = v415;
  v152 = v6[234].m128i_u32[2];
  if ( v150 > v152 )
  {
    if ( v150 > v6[234].m128i_u32[3] )
    {
      v153 = 0;
      v6[234].m128i_i32[2] = 0;
      sub_16CD150((__int64)v6[234].m128i_i64, &v6[235], v150, 16, v147, v148);
      v150 = DWORD2(src);
    }
    else
    {
      v153 = 16 * v152;
      if ( v6[234].m128i_i32[2] )
      {
        memmove((void *)v6[234].m128i_i64[0], (const void *)src, 16 * v152);
        v150 = DWORD2(src);
      }
    }
    v154 = 16 * v150;
    if ( (_QWORD)src + v153 != v154 + (_QWORD)src )
      memcpy((void *)(v153 + v6[234].m128i_i64[0]), (const void *)(src + v153), v154 - v153);
LABEL_180:
    v155 = v6[1].m128i_i8[9] == 0;
    v6[234].m128i_i32[2] = v151;
    if ( v155 )
      goto LABEL_181;
LABEL_275:
    v228 = v6->m128i_i64[0];
    v229 = v6[5].m128i_i64[0];
    v230 = v6->m128i_i64[1];
    if ( v6->m128i_i64[0] )
    {
      nullsub_686();
      *(_QWORD *)(v229 + 176) = v228;
      *(_DWORD *)(v229 + 184) = v230;
      sub_1D23870();
    }
    else
    {
      v414 = v6->m128i_i64[1];
      *(_QWORD *)(v229 + 176) = 0;
      *(_DWORD *)(v229 + 184) = v414;
    }
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_240;
  }
  if ( !v150 )
    goto LABEL_180;
  memmove((void *)v6[234].m128i_i64[0], (const void *)src, 16 * v150);
  v155 = v6[1].m128i_i8[9] == 0;
  v6[234].m128i_i32[2] = v151;
  if ( !v155 )
    goto LABEL_275;
LABEL_181:
  *(_QWORD *)&v439 = v440;
  *((_QWORD *)&v439 + 1) = 0x400000000LL;
  if ( !v316 )
  {
    *(_QWORD *)&v425 = v426;
    *((_QWORD *)&v425 + 1) = 0x100000000LL;
    v231 = sub_1647190(v303, *(_DWORD *)(v344 + 4));
    sub_20C7CE0(a2, v344, v231, &v425, 0, 0);
    v234 = (unsigned int)v431;
    v235 = v431;
    v360 = *(const void ***)(v425 + 8);
    v236 = DWORD2(v439);
    v372 = *(_QWORD *)v425;
    if ( (unsigned int)v431 >= (unsigned __int64)DWORD2(v439) )
    {
      v237 = (unsigned int)v431;
      if ( (unsigned int)v431 > (unsigned __int64)DWORD2(v439) )
      {
        if ( (unsigned int)v431 > (unsigned __int64)HIDWORD(v439) )
        {
          sub_16CD150((__int64)&v439, v440, (unsigned int)v431, 16, v232, v233);
          v236 = DWORD2(v439);
        }
        v278 = v439 + 16 * v236;
        for ( i = v237 * 16 + v439; i != v278; v278 += 16 )
        {
          if ( v278 )
          {
            *(_QWORD *)v278 = 0;
            *(_DWORD *)(v278 + 8) = 0;
          }
        }
        DWORD2(v439) = v235;
      }
    }
    else
    {
      DWORD2(v439) = v431;
      v237 = (unsigned int)v431;
    }
    v238 = &v445;
    v444.m128i_i64[1] = 0x400000000LL;
    v444.m128i_i64[0] = (__int64)&v445;
    if ( v234 > 4 )
    {
      sub_16CD150((__int64)&v444, &v445, v234, 16, v232, v233);
      v238 = (__m128i *)v444.m128i_i64[0];
    }
    v239 = &v238[v237];
    for ( v444.m128i_i32[2] = v235; v239 != v238; ++v238 )
    {
      if ( v238 )
      {
        v238->m128i_i64[0] = 0;
        v238->m128i_i32[2] = 0;
      }
    }
    v195 = (__int64)&v6[5].m128i_i64[1];
    if ( v235 )
    {
      v398 = 8 * v234;
      v240 = 0;
      do
      {
        v241 = (__int64 *)v6[5].m128i_i64[0];
        *(_QWORD *)&v242 = sub_1D38BB0(
                             (__int64)v241,
                             *(_QWORD *)&v427[v240],
                             (__int64)&v6[5].m128i_i64[1],
                             (unsigned int)v372,
                             v360,
                             0,
                             a4,
                             *(double *)si128.m128i_i64,
                             a6,
                             0);
        v309 = v306 | v309 & 0xFFFFFFFF00000000LL;
        v243 = 2 * v240;
        *(_QWORD *)&v244 = sub_1D332F0(
                             v241,
                             52,
                             (__int64)&v6[5].m128i_i64[1],
                             (unsigned int)v372,
                             v360,
                             3u,
                             *(double *)a4.m128i_i64,
                             *(double *)si128.m128i_i64,
                             a6,
                             v305,
                             v309,
                             v242);
        v245 = (_QWORD *)v6[5].m128i_i64[0];
        v421 = 0u;
        v413 = v244;
        v422 = 0;
        sub_1E341E0((__int64)&v423, v245[4], v307, *(_QWORD *)&v427[v240]);
        v246 = sub_1D2B730(
                 v245,
                 *(unsigned int *)&v430[2 * v240],
                 *(_QWORD *)&v430[2 * v240 + 8],
                 (__int64)&v6[5].m128i_i64[1],
                 v6->m128i_i64[0],
                 v6->m128i_i64[1],
                 v413,
                 *((__int64 *)&v413 + 1),
                 v423,
                 v424,
                 1,
                 0,
                 (__int64)&v421,
                 0);
        v248 = v247;
        v249 = v439;
        *(_QWORD *)(v439 + 2 * v240) = v246;
        v240 += 8;
        *(_DWORD *)(v249 + v243 + 8) = v248;
        v250 = v444.m128i_i64[0] + v243;
        *(_QWORD *)v250 = v246;
        *(_DWORD *)(v250 + 8) = 1;
      }
      while ( v398 != v240 );
    }
    *((_QWORD *)&v296 + 1) = v444.m128i_u32[2];
    *(_QWORD *)&v296 = v444.m128i_i64[0];
    v251 = sub_1D359D0(
             (__int64 *)v6[5].m128i_i64[0],
             2,
             (__int64)&v6[5].m128i_i64[1],
             1,
             0,
             0,
             *(double *)a4.m128i_i64,
             *(double *)si128.m128i_i64,
             a6,
             v296);
    v252 = (__m128i *)v444.m128i_i64[0];
    v6->m128i_i64[0] = (__int64)v251;
    v6->m128i_i32[2] = v253;
    if ( v252 != &v445 )
      _libc_free((unsigned __int64)v252);
    if ( (_BYTE *)v425 != v426 )
      _libc_free(v425);
    v174 = DWORD2(v439);
LABEL_237:
    v197 = (__int64 *)v6[5].m128i_i64[0];
    v198 = v439;
    v199 = v174;
    v200 = (const void ***)sub_1D25C30((__int64)v197, (unsigned __int8 *)v430, (unsigned int)v431);
    *((_QWORD *)&v294 + 1) = v199;
    *(_QWORD *)&v294 = v198;
    v203 = sub_1D36D80(v197, 51, v195, v200, v201, *(double *)a4.m128i_i64, *(double *)si128.m128i_i64, a6, v202, v294);
    v204 = _mm_loadu_si128(v6);
    v196 = (__int64 *)v439;
    *(_QWORD *)a1 = v203;
    *(_QWORD *)(a1 + 8) = v205;
    *(__m128i *)(a1 + 16) = v204;
    goto LABEL_238;
  }
  v156 = v6[1].m128i_i8[8];
  if ( (v156 & 1) != 0 )
  {
    v352 = 1;
    v345 = 3;
  }
  else
  {
    v157 = (v156 & 2) != 0;
    v155 = !v157;
    v352 = v157;
    v158 = 4;
    if ( v155 )
      v158 = 0;
    v345 = v158;
  }
  if ( !(_DWORD)v431 )
  {
    v196 = v440;
    goto LABEL_296;
  }
  v159 = (__int64)a2;
  v411 = 0;
  v350 = 16LL * (unsigned int)v431;
  v378 = 0;
  while ( 2 )
  {
    v175 = *(_QWORD *)v159;
    v176 = v6[2].m128i_u32[0];
    v177 = *(_QWORD *)&v430[v411];
    v178 = *(_QWORD *)&v430[v411 + 8];
    v179 = (__int64 *)v6[1].m128i_i64[0];
    v180 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v159 + 384LL);
    v181 = *v179;
    if ( v180 != sub_1F42DB0 )
    {
      v160 = v180(v159, *v179, v176, *(_QWORD *)&v430[v411], *(_QWORD *)&v430[v411 + 8]);
      goto LABEL_202;
    }
    LOBYTE(v423) = *(_QWORD *)&v430[v411];
    *((_QWORD *)&v423 + 1) = v178;
    if ( (_BYTE)v177 )
    {
      v160 = *(_BYTE *)(v159 + (unsigned __int8)v177 + 1155);
    }
    else if ( sub_1F58D20((__int64)&v423) )
    {
      v444.m128i_i8[0] = 0;
      v444.m128i_i64[1] = 0;
      LOBYTE(v421) = 0;
      sub_1F426C0(v159, v181, (unsigned int)v423, v178, (__int64)&v444, (unsigned int *)&v425, &v421);
      v160 = v421;
      v175 = *(_QWORD *)v159;
      v176 = v6[2].m128i_u32[0];
      v179 = (__int64 *)v6[1].m128i_i64[0];
    }
    else
    {
      sub_1F40D10((__int64)&v444, v159, v181, v423, *((__int64 *)&v423 + 1));
      LOBYTE(v425) = v444.m128i_i8[8];
      *((_QWORD *)&v425 + 1) = v445.m128i_i64[0];
      if ( v444.m128i_i8[8] )
      {
        v160 = *(_BYTE *)(v159 + v444.m128i_u8[8] + 1155);
      }
      else
      {
        v371 = v445.m128i_i64[0];
        if ( sub_1F58D20((__int64)&v425) )
        {
          v444.m128i_i8[0] = 0;
          v444.m128i_i64[1] = 0;
          LOBYTE(v419) = 0;
          sub_1F426C0(v159, v181, (unsigned int)v425, v371, (__int64)&v444, (unsigned int *)&v421, &v419);
          v160 = v419;
        }
        else
        {
          sub_1F40D10((__int64)&v444, v159, v181, v425, *((__int64 *)&v425 + 1));
          v194 = v301;
          LOBYTE(v194) = v444.m128i_i8[8];
          v301 = v194;
          v160 = sub_1D5E9F0(v159, v181, (unsigned int)v194, v445.m128i_i64[0]);
        }
      }
LABEL_202:
      v175 = *(_QWORD *)v159;
      v176 = v6[2].m128i_u32[0];
      v179 = (__int64 *)v6[1].m128i_i64[0];
    }
    v161 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(v175 + 392);
    v162 = *v179;
    if ( v161 == sub_1F42F80 )
    {
      LOBYTE(v423) = v177;
      *((_QWORD *)&v423 + 1) = v178;
      if ( (_BYTE)v177 )
      {
        v163 = *(unsigned __int8 *)(v159 + (unsigned __int8)v177 + 1040);
      }
      else
      {
        v359 = v162;
        if ( sub_1F58D20((__int64)&v423) )
        {
          v444.m128i_i8[0] = 0;
          v444.m128i_i64[1] = 0;
          LOBYTE(v421) = 0;
          v188 = sub_1F426C0(v159, v359, (unsigned int)v423, v178, (__int64)&v444, (unsigned int *)&v425, &v421);
          v176 = v6[2].m128i_u32[0];
          v163 = v188;
        }
        else
        {
          v370 = sub_1F58D40((__int64)&v423);
          v425 = v423;
          v190 = v423;
          if ( sub_1F58D20((__int64)&v425) )
          {
            v444.m128i_i8[0] = 0;
            v444.m128i_i64[1] = 0;
            LOBYTE(v419) = 0;
            sub_1F426C0(
              v159,
              v359,
              (unsigned int)v425,
              *((__int64 *)&v425 + 1),
              (__int64)&v444,
              (unsigned int *)&v421,
              &v419);
            v192 = v419;
          }
          else
          {
            sub_1F40D10((__int64)&v444, v159, v359, v190, *((__int64 *)&v190 + 1));
            v191 = v302;
            LOBYTE(v191) = v444.m128i_i8[8];
            v302 = v191;
            v192 = sub_1D5E9F0(v159, v359, (unsigned int)v191, v445.m128i_i64[0]);
          }
          v193 = sub_2045180(v192);
          v176 = v6[2].m128i_u32[0];
          v163 = (v193 + v370 - 1) / v193;
        }
      }
    }
    else
    {
      v189 = v161(v159, *v179, v176, v177, v178);
      v176 = v6[2].m128i_u32[0];
      v163 = v189;
    }
    v444.m128i_i8[4] = v352;
    if ( v352 )
      v444.m128i_i32[0] = v345;
    v164 = (__int64 *)v6[5].m128i_i64[0];
    *((_QWORD *)&v292 + 1) = v178;
    *(_QWORD *)&v292 = v177;
    v366 = v163;
    BYTE4(v425) = 1;
    LODWORD(v425) = v176;
    v165 = sub_204AFD0(
             v164,
             (__int64)&v6[5].m128i_i64[1],
             src + 16LL * v378,
             v163,
             v160,
             0,
             a4,
             si128,
             a6,
             v292,
             (__int64)&v425,
             (unsigned int *)&v444);
    v168 = v366;
    v169 = v165;
    v170 = DWORD2(v439);
    v172 = v171;
    if ( DWORD2(v439) >= HIDWORD(v439) )
    {
      sub_16CD150((__int64)&v439, v440, 0, 16, v166, v167);
      v170 = DWORD2(v439);
      v168 = v366;
    }
    v173 = (__int64 **)(v439 + 16 * v170);
    v411 += 16;
    *v173 = v169;
    v173[1] = v172;
    v378 += v168;
    v174 = ++DWORD2(v439);
    if ( v350 != v411 )
      continue;
    break;
  }
  v195 = (__int64)&v6[5].m128i_i64[1];
  v196 = (__int64 *)v439;
  if ( v174 )
    goto LABEL_237;
LABEL_296:
  v254 = _mm_loadu_si128(v6);
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(__m128i *)(a1 + 16) = v254;
LABEL_238:
  if ( v196 != v440 )
    _libc_free((unsigned __int64)v196);
LABEL_240:
  if ( (__int8 *)src != &v434 )
    _libc_free(src);
  if ( v441 != v443 )
    _libc_free((unsigned __int64)v441);
  if ( v427 != v429 )
    _libc_free((unsigned __int64)v427);
  if ( v430 != v432 )
    _libc_free((unsigned __int64)v430);
  return a1;
}
