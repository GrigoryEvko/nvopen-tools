// Function: sub_20A2AF0
// Address: 0x20a2af0
//
__int64 __fastcall sub_20A2AF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const void ***a4,
        _QWORD *a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        int a10,
        int a11)
{
  unsigned int v14; // r13d
  __m128i *v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  bool v20; // cc
  __int32 v21; // eax
  __int64 v22; // rdi
  __int16 v23; // dx
  __int64 v24; // rax
  unsigned int v25; // ebx
  __int64 v26; // rax
  unsigned __int64 v27; // r12
  unsigned __int8 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned int v33; // r15d
  void *v34; // rax
  _QWORD *v35; // rdi
  _QWORD *v36; // rbx
  int v37; // edx
  int v38; // r14d
  int v39; // eax
  __int64 v40; // rdx
  int v41; // ecx
  int v42; // r8d
  int v43; // r9d
  __int64 v44; // rbx
  __int64 v45; // r13
  unsigned int v46; // ebx
  __int64 v47; // rsi
  unsigned int v48; // ebx
  unsigned __int64 v49; // rax
  __int64 v50; // r13
  int v51; // edx
  __int64 v52; // rax
  __int64 v53; // rsi
  int v54; // edx
  __int64 v55; // rax
  unsigned int v56; // r11d
  unsigned __int64 v57; // rax
  unsigned int v58; // ebx
  unsigned __int64 v59; // rax
  unsigned __int16 v60; // ax
  unsigned __int64 v61; // r9
  __int64 *v62; // rax
  int v63; // edx
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned int v67; // eax
  unsigned int v68; // ebx
  int v69; // r15d
  unsigned int v70; // ebx
  char v71; // bl
  __int64 v72; // rax
  __int8 v73; // dl
  __int64 v74; // rax
  unsigned int v75; // ebx
  __int64 v76; // rax
  __int64 *v77; // rax
  int v78; // edx
  __int64 v79; // rax
  int v80; // r8d
  int v81; // r9d
  __int64 v82; // rbx
  __int64 v83; // rcx
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  int v86; // ecx
  unsigned int v87; // ebx
  __int64 v88; // rax
  unsigned __int64 v89; // rax
  char v90; // al
  unsigned int v91; // eax
  unsigned int v92; // eax
  int v93; // ebx
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 v96; // rdx
  __int64 v97; // rax
  int v98; // r9d
  _QWORD *v99; // rcx
  __int64 v100; // rdx
  unsigned __int64 v101; // r8
  unsigned int v102; // ebx
  unsigned int v103; // edx
  int v104; // eax
  __int64 v105; // r8
  __int64 v106; // r9
  unsigned int v107; // ecx
  unsigned __int64 v108; // rax
  unsigned int v109; // ecx
  unsigned __int64 v110; // rax
  __int64 v111; // rdx
  char v112; // al
  __int64 v113; // rdx
  unsigned int v114; // r15d
  unsigned __int64 v115; // rdx
  unsigned int v116; // eax
  int v117; // ebx
  __int64 v118; // rsi
  unsigned __int64 v119; // rdx
  unsigned __int64 v120; // rdx
  _QWORD *v121; // rax
  unsigned int v122; // eax
  unsigned __int64 v123; // rsi
  unsigned __int64 v124; // rdx
  __int64 v125; // rdi
  unsigned __int8 *v126; // rax
  __int64 v127; // rdx
  __int64 v128; // rax
  unsigned int v129; // ebx
  _QWORD *v130; // rax
  unsigned __int64 v131; // rax
  __int32 v132; // edx
  unsigned __int64 v133; // rax
  unsigned __int64 v134; // rax
  __int32 v135; // edx
  unsigned __int64 v136; // rax
  unsigned int v137; // ecx
  unsigned __int64 v138; // rdx
  unsigned int v139; // ecx
  unsigned __int64 v140; // rdx
  __int64 v141; // rbx
  int v142; // eax
  __int64 v143; // r12
  __int64 v144; // rax
  unsigned int v145; // r15d
  unsigned __int64 v146; // rdx
  unsigned __int64 v147; // rax
  __int32 v148; // edx
  __int64 v149; // rcx
  __int32 v150; // r15d
  unsigned __int64 v151; // rax
  unsigned int v152; // ebx
  __int32 v153; // eax
  __int64 v154; // rdx
  __int64 v155; // rdi
  __int64 v156; // rax
  __int64 v157; // rax
  unsigned int v158; // eax
  __int64 v159; // rax
  __int32 v160; // ebx
  __int64 v161; // r13
  __int64 v162; // rdi
  __int64 v163; // rax
  __int64 v164; // rbx
  __int64 v165; // rbx
  __int64 *v166; // rax
  __int64 v167; // rsi
  __int64 v168; // rdx
  __int64 v169; // rdi
  __int64 v170; // rax
  unsigned int v171; // edx
  unsigned __int64 v172; // rsi
  unsigned __int64 v173; // rax
  unsigned __int64 v174; // rax
  unsigned __int64 v175; // r15
  const void *v176; // r15
  unsigned __int32 v177; // edx
  __int64 v178; // rsi
  unsigned __int64 v179; // rax
  unsigned __int64 v180; // rax
  _QWORD *v181; // rax
  char v182; // al
  char v183; // bl
  __int32 v184; // edx
  __int64 v185; // rax
  unsigned __int8 *v186; // rax
  __int64 v187; // rdx
  __int64 v188; // rax
  unsigned int v189; // eax
  unsigned int v190; // r15d
  unsigned int v191; // ebx
  unsigned __int64 v192; // rax
  unsigned int v193; // edx
  __int32 v194; // eax
  __int32 v195; // eax
  __int64 v196; // rdi
  int v197; // eax
  __int64 v198; // rax
  int v199; // edx
  __int64 v200; // rsi
  __int64 v201; // rbx
  unsigned int v202; // edx
  int v203; // eax
  __int64 v204; // rax
  __int8 v205; // dl
  __int64 v206; // rax
  bool v207; // bl
  int v208; // eax
  unsigned int v209; // eax
  unsigned int v210; // ebx
  unsigned __int64 v211; // rax
  unsigned int v212; // ebx
  __int32 v213; // eax
  __int64 v214; // rdx
  __int64 v215; // rdi
  int v216; // eax
  __int64 v217; // rax
  int v218; // edx
  __int64 v219; // rdx
  unsigned __int64 v220; // rax
  __int32 v221; // eax
  __int64 v222; // rdi
  __int64 v223; // rcx
  __int64 v224; // rax
  int v225; // edx
  __int64 v226; // rax
  __int64 v227; // rdx
  unsigned int v228; // ebx
  unsigned __int64 v229; // rbx
  __int64 v230; // rdx
  unsigned int v231; // eax
  unsigned int v232; // eax
  unsigned int v233; // ebx
  unsigned __int64 v234; // rax
  int v235; // edx
  __int64 v236; // rax
  __int64 v237; // rcx
  __int64 v238; // rax
  __int64 v239; // rcx
  unsigned int v240; // edx
  unsigned __int64 v241; // rax
  unsigned __int64 v242; // rax
  unsigned __int32 v243; // edx
  unsigned __int64 v244; // rax
  unsigned __int64 v245; // rax
  _QWORD *v246; // rax
  char v247; // bl
  __int32 v248; // edx
  __int64 v249; // rax
  int v250; // eax
  bool v251; // zf
  __int128 v252; // rax
  __int64 *v253; // rax
  int v254; // edx
  unsigned __int64 v255; // rax
  int v256; // eax
  bool v257; // cl
  int v258; // eax
  __int64 v259; // rax
  __int8 v260; // cl
  __int64 v261; // rax
  int v262; // eax
  bool v263; // al
  __int64 v264; // rax
  __int64 *v265; // r13
  unsigned int v266; // eax
  __int64 v267; // rdx
  __int64 *v268; // rax
  int v269; // edx
  __int64 v270; // rdx
  unsigned int v271; // eax
  __int64 v272; // rax
  bool v273; // r13
  __int64 v274; // r13
  _QWORD *v275; // rax
  unsigned int v276; // r14d
  signed int v277; // edx
  __int128 v278; // rax
  __int64 *v279; // rax
  int v280; // edx
  __int64 v281; // rax
  int v282; // edx
  __int64 v283; // rdx
  _QWORD *v284; // rax
  __int64 v285; // rax
  const void **v286; // rdx
  unsigned int v287; // edx
  __int64 v288; // rdx
  _QWORD *v289; // rax
  __int64 v290; // rax
  __int64 v291; // rsi
  unsigned int v292; // ebx
  bool v293; // bl
  __int64 v294; // rax
  unsigned __int64 v295; // rdx
  __int64 *v296; // rax
  int v297; // edx
  __int64 *v298; // rax
  int v299; // edx
  unsigned int v300; // eax
  int v301; // r10d
  unsigned int v302; // r13d
  __int64 v303; // rdx
  unsigned __int8 *v304; // rdx
  int v305; // eax
  __int64 v306; // rdx
  int v307; // eax
  __int64 v308; // rdx
  __int64 v309; // rsi
  __m128i v310; // kr10_16
  __int128 v311; // rax
  unsigned int v312; // eax
  unsigned int v313; // edx
  int v314; // eax
  __int128 v315; // rax
  __int64 *v316; // rax
  int v317; // edx
  __int64 v318; // rax
  __int32 v319; // eax
  _QWORD *v320; // rax
  unsigned int v321; // r14d
  int v322; // edx
  __int128 v323; // rax
  __int64 *v324; // rax
  int v325; // edx
  unsigned int v326; // eax
  int v327; // eax
  __int64 *v328; // rax
  int v329; // edx
  unsigned __int64 v330; // rdx
  __int32 v331; // edx
  __int64 v332; // rax
  const __m128i *v333; // rax
  __int64 v334; // rcx
  __m128i v335; // xmm4
  __int64 v336; // rax
  unsigned __int8 *v337; // rax
  __int64 v338; // rdx
  __int64 v339; // rax
  int v340; // eax
  __int64 v341; // rdx
  int v342; // ecx
  int v343; // r8d
  int v344; // r9d
  __int64 v345; // rax
  unsigned int v346; // ecx
  unsigned int v347; // eax
  int v348; // eax
  const void ***v349; // rax
  __int64 v350; // r14
  __int64 v351; // rdx
  __int64 v352; // r15
  __int64 v353; // rax
  unsigned __int64 v354; // rdx
  __int64 *v355; // rax
  int v356; // edx
  bool v357; // al
  __int64 v358; // rdx
  __int32 v359; // eax
  char v360; // bl
  __int64 *v361; // rax
  int v362; // edx
  __int32 v363; // eax
  int v364; // eax
  int v365; // eax
  __int32 v366; // eax
  bool v367; // bl
  __int128 v368; // rax
  __int64 *v369; // rax
  int v370; // edx
  __int32 v371; // eax
  unsigned int v372; // eax
  __int32 v373; // eax
  __int32 v374; // eax
  unsigned int v375; // eax
  __int32 v376; // eax
  __int32 v377; // eax
  __int64 v378; // rdx
  int v379; // ecx
  int v380; // r8d
  int v381; // r9d
  int v382; // eax
  __int64 v383; // rax
  _QWORD *v384; // r13
  int v385; // eax
  __int64 *v386; // rax
  int v387; // edx
  const void ***v388; // rax
  __int128 v389; // rax
  __int64 *v390; // rax
  int v391; // edx
  __int32 v392; // eax
  __int64 v393; // rdx
  __int64 v394; // rdx
  __int64 v395; // rcx
  __int64 v396; // r8
  __int64 v397; // r9
  unsigned int v398; // r13d
  int v399; // eax
  unsigned int v400; // ebx
  __int64 *v401; // r15
  __int128 v402; // rax
  __int128 v403; // rax
  __int64 v404; // rax
  int v405; // edx
  __int32 v406; // eax
  int v407; // eax
  __int64 v408; // rax
  int v409; // edx
  int v410; // eax
  __int64 v411; // rdi
  int v412; // eax
  __int32 v413; // eax
  __int64 v414; // rdx
  __int64 *v415; // rax
  int v416; // edx
  __int32 v417; // eax
  __int32 v418; // eax
  char v419; // al
  int v420; // [rsp-10h] [rbp-620h]
  __int128 v421; // [rsp-10h] [rbp-620h]
  int v422; // [rsp-8h] [rbp-618h]
  __int64 v423; // [rsp-8h] [rbp-618h]
  __int64 v424; // [rsp+0h] [rbp-610h]
  unsigned __int64 v425; // [rsp+8h] [rbp-608h]
  char v426; // [rsp+10h] [rbp-600h]
  __int64 v427; // [rsp+10h] [rbp-600h]
  __int64 v428; // [rsp+10h] [rbp-600h]
  unsigned __int64 v429; // [rsp+10h] [rbp-600h]
  bool v430; // [rsp+10h] [rbp-600h]
  __m128i v431; // [rsp+10h] [rbp-600h]
  unsigned int v432; // [rsp+10h] [rbp-600h]
  __int64 v433; // [rsp+10h] [rbp-600h]
  unsigned __int64 v434; // [rsp+18h] [rbp-5F8h]
  __int64 v435; // [rsp+20h] [rbp-5F0h]
  __int64 v436; // [rsp+28h] [rbp-5E8h]
  int v437; // [rsp+28h] [rbp-5E8h]
  __int64 v438; // [rsp+28h] [rbp-5E8h]
  int v439; // [rsp+28h] [rbp-5E8h]
  int v440; // [rsp+30h] [rbp-5E0h]
  unsigned int v441; // [rsp+30h] [rbp-5E0h]
  __m128i *v442; // [rsp+30h] [rbp-5E0h]
  __int64 v443; // [rsp+30h] [rbp-5E0h]
  int v444; // [rsp+30h] [rbp-5E0h]
  unsigned int v445; // [rsp+30h] [rbp-5E0h]
  __int64 v446; // [rsp+30h] [rbp-5E0h]
  __int64 v447; // [rsp+30h] [rbp-5E0h]
  __int128 v448; // [rsp+30h] [rbp-5E0h]
  int v449; // [rsp+40h] [rbp-5D0h]
  __int64 v450; // [rsp+40h] [rbp-5D0h]
  _QWORD *v451; // [rsp+40h] [rbp-5D0h]
  int v452; // [rsp+40h] [rbp-5D0h]
  __int64 v453; // [rsp+40h] [rbp-5D0h]
  unsigned int v454; // [rsp+40h] [rbp-5D0h]
  __int16 v455; // [rsp+48h] [rbp-5C8h]
  int v456; // [rsp+48h] [rbp-5C8h]
  unsigned __int64 v457; // [rsp+48h] [rbp-5C8h]
  unsigned int v458; // [rsp+48h] [rbp-5C8h]
  __int64 v459; // [rsp+48h] [rbp-5C8h]
  unsigned int v460; // [rsp+48h] [rbp-5C8h]
  unsigned int v461; // [rsp+48h] [rbp-5C8h]
  unsigned int v462; // [rsp+48h] [rbp-5C8h]
  bool v463; // [rsp+48h] [rbp-5C8h]
  __int64 v464; // [rsp+50h] [rbp-5C0h]
  __int64 v465; // [rsp+50h] [rbp-5C0h]
  __int64 v466; // [rsp+50h] [rbp-5C0h]
  __int64 *v467; // [rsp+50h] [rbp-5C0h]
  char v468; // [rsp+50h] [rbp-5C0h]
  unsigned int v469; // [rsp+60h] [rbp-5B0h]
  const void **v470; // [rsp+68h] [rbp-5A8h]
  __int128 v472; // [rsp+70h] [rbp-5A0h]
  unsigned int v473; // [rsp+70h] [rbp-5A0h]
  __int64 v474; // [rsp+70h] [rbp-5A0h]
  __int128 v475; // [rsp+70h] [rbp-5A0h]
  unsigned int v476; // [rsp+80h] [rbp-590h]
  unsigned __int64 v477; // [rsp+80h] [rbp-590h]
  __int64 v478; // [rsp+80h] [rbp-590h]
  __int64 v479; // [rsp+80h] [rbp-590h]
  __int64 v480; // [rsp+80h] [rbp-590h]
  bool v481; // [rsp+80h] [rbp-590h]
  __int64 v482; // [rsp+80h] [rbp-590h]
  unsigned int v483; // [rsp+80h] [rbp-590h]
  unsigned int v484; // [rsp+80h] [rbp-590h]
  __m128i v485; // [rsp+80h] [rbp-590h]
  __int64 v486; // [rsp+80h] [rbp-590h]
  __int64 v487; // [rsp+80h] [rbp-590h]
  unsigned __int64 v489; // [rsp+90h] [rbp-580h]
  __int64 v491; // [rsp+98h] [rbp-578h]
  __int32 v492; // [rsp+98h] [rbp-578h]
  unsigned __int64 v493; // [rsp+98h] [rbp-578h]
  __m128i v494; // [rsp+1E0h] [rbp-430h]
  __m128i v495; // [rsp+380h] [rbp-290h]
  __m128i v496; // [rsp+3A0h] [rbp-270h]
  __m128i v497; // [rsp+3C0h] [rbp-250h]
  __m128i v498; // [rsp+3E0h] [rbp-230h]
  __m128i v499; // [rsp+420h] [rbp-1F0h]
  __m128i v500; // [rsp+440h] [rbp-1D0h]
  unsigned __int64 v501; // [rsp+4C0h] [rbp-150h] BYREF
  unsigned int v502; // [rsp+4C8h] [rbp-148h]
  __int64 v503; // [rsp+4D0h] [rbp-140h] BYREF
  int v504; // [rsp+4D8h] [rbp-138h]
  __m128i v505; // [rsp+4E0h] [rbp-130h] BYREF
  char v506[8]; // [rsp+4F0h] [rbp-120h] BYREF
  __int64 v507; // [rsp+4F8h] [rbp-118h]
  __int64 v508; // [rsp+500h] [rbp-110h] BYREF
  unsigned int v509; // [rsp+508h] [rbp-108h]
  __int64 v510; // [rsp+510h] [rbp-100h] BYREF
  unsigned int v511; // [rsp+518h] [rbp-F8h]
  unsigned __int64 v512; // [rsp+520h] [rbp-F0h] BYREF
  unsigned int v513; // [rsp+528h] [rbp-E8h]
  unsigned __int64 v514; // [rsp+530h] [rbp-E0h] BYREF
  __int64 v515; // [rsp+538h] [rbp-D8h]
  unsigned __int64 v516; // [rsp+540h] [rbp-D0h] BYREF
  unsigned int v517; // [rsp+548h] [rbp-C8h]
  __m128i v518; // [rsp+550h] [rbp-C0h] BYREF
  unsigned __int64 v519; // [rsp+560h] [rbp-B0h] BYREF
  __int64 v520; // [rsp+568h] [rbp-A8h]
  unsigned __int64 v521; // [rsp+570h] [rbp-A0h] BYREF
  __int64 v522; // [rsp+578h] [rbp-98h]
  __int64 v523; // [rsp+580h] [rbp-90h] BYREF
  __int64 v524; // [rsp+588h] [rbp-88h]
  __int64 v525; // [rsp+590h] [rbp-80h] BYREF
  __int64 v526; // [rsp+598h] [rbp-78h]
  __m128i v527; // [rsp+5A0h] [rbp-70h] BYREF
  const void *v528; // [rsp+5B0h] [rbp-60h] BYREF
  __int64 v529; // [rsp+5B8h] [rbp-58h]
  __m128i v530; // [rsp+5C0h] [rbp-50h] BYREF
  __int64 v531; // [rsp+5D0h] [rbp-40h] BYREF
  __int64 v532; // [rsp+5D8h] [rbp-38h]

  v14 = *((_DWORD *)a4 + 2);
  LODWORD(v15) = a11;
  v476 = a3;
  v502 = v14;
  if ( v14 > 0x40 )
    sub_16A4FD0((__int64)&v501, (const void **)a4);
  else
    v501 = (unsigned __int64)*a4;
  v16 = *(_QWORD *)(a2 + 72);
  v503 = v16;
  if ( v16 )
    sub_1623A60((__int64)&v503, v16, 2);
  v504 = *(_DWORD *)(a2 + 64);
  v17 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a6 + 32LL));
  v530.m128i_i32[2] = v14;
  v464 = v17;
  if ( v14 > 0x40 )
  {
    sub_16A4EF0((__int64)&v530, 0, 0);
    v16 = 0;
    LODWORD(v532) = v14;
    sub_16A4EF0((__int64)&v531, 0, 0);
  }
  else
  {
    v530.m128i_i64[0] = 0;
    LODWORD(v532) = v14;
    v531 = 0;
  }
  if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
    j_j___libc_free_0_0(*a5);
  v20 = *((_DWORD *)a5 + 6) <= 0x40u;
  *a5 = v530.m128i_i64[0];
  v21 = v530.m128i_i32[2];
  v530.m128i_i32[2] = 0;
  *((_DWORD *)a5 + 2) = v21;
  v470 = (const void **)(a5 + 2);
  if ( v20 || (v22 = a5[2]) == 0 )
  {
    a5[2] = v531;
    *((_DWORD *)a5 + 6) = v532;
    goto LABEL_34;
  }
  j_j___libc_free_0_0(v22);
  v20 = v530.m128i_i32[2] <= 0x40u;
  a5[2] = v531;
  *((_DWORD *)a5 + 6) = v532;
  if ( v20 || !v530.m128i_i64[0] )
  {
LABEL_34:
    v23 = *(_WORD *)(a2 + 24);
    if ( v23 == 10 )
      goto LABEL_15;
    goto LABEL_35;
  }
  j_j___libc_free_0_0(v530.m128i_i64[0]);
  v23 = *(_WORD *)(a2 + 24);
  if ( v23 == 10 )
  {
LABEL_15:
    v24 = *(_QWORD *)(a2 + 88);
    if ( *((_DWORD *)a5 + 6) <= 0x40u && *(_DWORD *)(v24 + 32) <= 0x40u )
    {
      v219 = *(_QWORD *)(v24 + 24);
      a5[2] = v219;
      v25 = *(_DWORD *)(v24 + 32);
      *((_DWORD *)a5 + 6) = v25;
      v220 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v25;
      if ( v25 <= 0x40 )
      {
        a5[2] = v220 & v219;
        goto LABEL_19;
      }
      v223 = (unsigned int)(((unsigned __int64)v25 + 63) >> 6) - 1;
      *(_QWORD *)(v219 + 8 * v223) &= v220;
      v25 = *((_DWORD *)a5 + 6);
    }
    else
    {
      sub_16A51C0((__int64)v470, v24 + 24);
      v25 = *((_DWORD *)a5 + 6);
    }
    v530.m128i_i32[2] = v25;
    if ( v25 > 0x40 )
    {
      sub_16A4FD0((__int64)&v530, v470);
      v25 = v530.m128i_u32[2];
      if ( v530.m128i_i32[2] > 0x40u )
      {
        sub_16A8F40(v530.m128i_i64);
        v27 = v530.m128i_i64[0];
        v25 = v530.m128i_u32[2];
        goto LABEL_21;
      }
      v26 = v530.m128i_i64[0];
LABEL_20:
      v27 = ~v26 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v25);
      v530.m128i_i64[0] = v27;
LABEL_21:
      v20 = *((_DWORD *)a5 + 2) <= 0x40u;
      v530.m128i_i32[2] = 0;
      if ( v20 || !*a5 )
      {
        *a5 = v27;
        *((_DWORD *)a5 + 2) = v25;
      }
      else
      {
        j_j___libc_free_0_0(*a5);
        v20 = v530.m128i_i32[2] <= 0x40u;
        *a5 = v27;
        *((_DWORD *)a5 + 2) = v25;
        if ( !v20 && v530.m128i_i64[0] )
          j_j___libc_free_0_0(v530.m128i_i64[0]);
      }
      goto LABEL_26;
    }
LABEL_19:
    v26 = a5[2];
    goto LABEL_20;
  }
LABEL_35:
  v469 = v476;
  v29 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * v476);
  v30 = *v29;
  v505.m128i_i64[1] = *((_QWORD *)v29 + 1);
  v31 = *(_QWORD *)(a2 + 48);
  v505.m128i_i8[0] = v30;
  if ( (!v31 || *(_QWORD *)(v31 + 32)) && !(_BYTE)a11 )
  {
    if ( a10 )
    {
      LODWORD(v15) = 0;
      sub_1D1F820(*(_QWORD *)a6, a2, a3, a5, a10);
      goto LABEL_27;
    }
    v530.m128i_i32[2] = v14;
    if ( v14 <= 0x40 )
    {
      v30 = -v14;
      v530.m128i_i64[0] = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
    }
    else
    {
      v16 = -1;
      sub_16A4EF0((__int64)&v530, -1, 1);
    }
    if ( v502 > 0x40 && v501 )
      j_j___libc_free_0_0(v501);
    v23 = *(_WORD *)(a2 + 24);
    v501 = v530.m128i_i64[0];
    v502 = v530.m128i_u32[2];
LABEL_44:
    v519 = 0;
    v520 = 1;
    v521 = 0;
    v522 = 1;
    v523 = 0;
    v524 = 1;
    v525 = 0;
    v526 = 1;
    v32 = (unsigned __int16)(v23 - 4);
    switch ( (__int16)v32 )
    {
      case 0:
        v111 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
        v112 = *(_BYTE *)(v111 + 88);
        v113 = *(_QWORD *)(v111 + 96);
        LOBYTE(v514) = v112;
        v515 = v113;
        if ( v112 )
          v114 = sub_1F3E310(&v514);
        else
          v114 = sub_1F58D40((__int64)&v514);
        v517 = v14;
        if ( v14 > 0x40 )
          sub_16A4EF0((__int64)&v516, 0, 0);
        else
          v516 = 0;
        if ( !v114 )
          goto LABEL_460;
        if ( v114 > 0x40 )
        {
          sub_16A5260(&v516, 0, v114);
LABEL_460:
          v116 = v517;
        }
        else
        {
          v115 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v114);
          v116 = v517;
          if ( v517 <= 0x40 )
          {
            v516 |= v115;
            v117 = a10 + 1;
            goto LABEL_187;
          }
          *(_QWORD *)v516 |= v115;
          v116 = v517;
        }
        v518.m128i_i32[2] = v116;
        v117 = a10 + 1;
        if ( v116 > 0x40 )
        {
          sub_16A4FD0((__int64)&v518, (const void **)&v516);
          v116 = v518.m128i_u32[2];
          if ( v518.m128i_i32[2] > 0x40u )
          {
            sub_16A8F40(v518.m128i_i64);
            v116 = v518.m128i_u32[2];
            v119 = v518.m128i_i64[0];
            v518.m128i_i32[2] = 0;
            v527.m128i_i32[2] = v116;
            v527.m128i_i64[0] = v518.m128i_i64[0];
            if ( v116 > 0x40 )
            {
              sub_16A89F0(v527.m128i_i64, (__int64 *)&v501);
              v116 = v527.m128i_u32[2];
              v120 = v527.m128i_i64[0];
              goto LABEL_190;
            }
LABEL_189:
            v120 = v501 | v119;
            v527.m128i_i64[0] = v120;
LABEL_190:
            v530.m128i_i32[2] = v116;
            v121 = *(_QWORD **)(a2 + 32);
            v530.m128i_i64[0] = v120;
            v527.m128i_i32[2] = 0;
            LODWORD(v15) = sub_20A2AF0(a1, *v121, v121[1], (unsigned int)&v530, (_DWORD)a5, a6, v117, 0);
            if ( v530.m128i_i32[2] > 0x40u && v530.m128i_i64[0] )
              j_j___libc_free_0_0(v530.m128i_i64[0]);
            if ( v527.m128i_i32[2] > 0x40u && v527.m128i_i64[0] )
              j_j___libc_free_0_0(v527.m128i_i64[0]);
            if ( v518.m128i_i32[2] > 0x40u && v518.m128i_i64[0] )
              j_j___libc_free_0_0(v518.m128i_i64[0]);
            if ( (_BYTE)v15 )
            {
              sub_135E100((__int64 *)&v516);
              goto LABEL_90;
            }
            v122 = v517;
            v527.m128i_i32[2] = v517;
            if ( v517 > 0x40 )
            {
              sub_16A4FD0((__int64)&v527, (const void **)&v516);
              v122 = v527.m128i_u32[2];
              if ( v527.m128i_i32[2] > 0x40u )
              {
                sub_16A8F40(v527.m128i_i64);
                v122 = v527.m128i_u32[2];
                v124 = v527.m128i_i64[0];
LABEL_203:
                v20 = *((_DWORD *)a5 + 2) <= 0x40u;
                v530.m128i_i32[2] = v122;
                v530.m128i_i64[0] = v124;
                v527.m128i_i32[2] = 0;
                if ( v20 )
                {
                  *a5 |= v124;
                }
                else
                {
                  sub_16A89F0(a5, v530.m128i_i64);
                  v122 = v530.m128i_u32[2];
                }
                if ( v122 > 0x40 && v530.m128i_i64[0] )
                  j_j___libc_free_0_0(v530.m128i_i64[0]);
                if ( v527.m128i_i32[2] > 0x40u && v527.m128i_i64[0] )
                  j_j___libc_free_0_0(v527.m128i_i64[0]);
                if ( v517 <= 0x40 )
                  goto LABEL_72;
                v125 = v516;
                if ( !v516 )
                  goto LABEL_72;
                goto LABEL_213;
              }
              v123 = v527.m128i_i64[0];
            }
            else
            {
              v123 = v516;
            }
            v124 = ~v123 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v122);
            v527.m128i_i64[0] = v124;
            goto LABEL_203;
          }
          v118 = v518.m128i_i64[0];
LABEL_188:
          v518.m128i_i32[2] = 0;
          v119 = ~v118 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v116);
          v518.m128i_i64[0] = v119;
          goto LABEL_189;
        }
LABEL_187:
        v118 = v516;
        goto LABEL_188;
      case 46:
        v126 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
        v127 = *v126;
        v128 = *((_QWORD *)v126 + 1);
        v506[0] = v127;
        v507 = v128;
        v129 = sub_1D159C0((__int64)v506, v16, v127, v30, v18, v19);
        sub_16A88B0((__int64)&v530, (__int64)&v501, v129);
        sub_16A5A50((__int64)&v508, v530.m128i_i64, v129);
        if ( v530.m128i_i32[2] > 0x40u && v530.m128i_i64[0] )
          j_j___libc_free_0_0(v530.m128i_i64[0]);
        sub_16A8130((__int64)&v530, (__int64)&v501, v129);
        sub_16A5A50((__int64)&v510, v530.m128i_i64, v129);
        if ( v530.m128i_i32[2] > 0x40u && v530.m128i_i64[0] )
          j_j___libc_free_0_0(v530.m128i_i64[0]);
        v527.m128i_i64[0] = 0;
        v527.m128i_i64[1] = 1;
        v130 = *(_QWORD **)(a2 + 32);
        v528 = 0;
        v529 = 1;
        v530.m128i_i64[0] = 0;
        v530.m128i_i64[1] = 1;
        v531 = 0;
        v532 = 1;
        if ( !(unsigned __int8)sub_20A2AF0(a1, *v130, v130[1], (unsigned int)&v508, (unsigned int)&v527, a6, a10 + 1, 0)
          && !(unsigned __int8)sub_20A2AF0(
                                 a1,
                                 *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                                 (unsigned int)&v510,
                                 (unsigned int)&v530,
                                 a6,
                                 a10 + 1,
                                 0) )
        {
          v15 = (__m128i *)&v516;
          sub_16A5C50((__int64)&v514, (const void **)&v530, v14);
          sub_13A38D0((__int64)&v516, (__int64)&v514);
          if ( v517 > 0x40 )
          {
            sub_16A7DC0((__int64 *)&v516, v129);
          }
          else
          {
            v131 = 0;
            if ( v129 != v517 )
              v131 = (v516 << v129) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v517);
            v516 = v131;
          }
          sub_16A5C50((__int64)&v512, (const void **)&v527, v14);
          v132 = v517;
          if ( v517 > 0x40 )
          {
            sub_16A89F0((__int64 *)&v516, (__int64 *)&v512);
            v132 = v517;
            v133 = v516;
          }
          else
          {
            v133 = v512 | v516;
            v516 |= v512;
          }
          v518.m128i_i64[0] = v133;
          v518.m128i_i32[2] = v132;
          v517 = 0;
          sub_14A9CA0(a5, v518.m128i_i64);
          sub_135E100(v518.m128i_i64);
          sub_135E100((__int64 *)&v512);
          sub_135E100((__int64 *)&v516);
          sub_135E100((__int64 *)&v514);
          sub_16A5C50((__int64)&v514, (const void **)&v531, v14);
          sub_13A38D0((__int64)&v516, (__int64)&v514);
          if ( v517 > 0x40 )
          {
            sub_16A7DC0((__int64 *)&v516, v129);
          }
          else
          {
            v134 = 0;
            if ( v129 != v517 )
              v134 = (v516 << v129) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v517);
            v516 = v134;
          }
          sub_16A5C50((__int64)&v512, &v528, v14);
          v135 = v517;
          if ( v517 > 0x40 )
          {
            sub_16A89F0((__int64 *)&v516, (__int64 *)&v512);
            v135 = v517;
            v136 = v516;
          }
          else
          {
            v136 = v512 | v516;
            v516 |= v512;
          }
          v518.m128i_i32[2] = v135;
          v518.m128i_i64[0] = v136;
          v517 = 0;
          sub_14A9CA0((__int64 *)v470, v518.m128i_i64);
          sub_135E100(v518.m128i_i64);
          sub_135E100((__int64 *)&v512);
          sub_135E100((__int64 *)&v516);
          sub_135E100((__int64 *)&v514);
          sub_135E100(&v531);
          sub_135E100(v530.m128i_i64);
          sub_135E100((__int64 *)&v528);
          sub_135E100(v527.m128i_i64);
          sub_135E100(&v510);
          sub_135E100(&v508);
          goto LABEL_72;
        }
        if ( (unsigned int)v532 > 0x40 && v531 )
          j_j___libc_free_0_0(v531);
        if ( v530.m128i_i32[2] > 0x40u && v530.m128i_i64[0] )
          j_j___libc_free_0_0(v530.m128i_i64[0]);
        if ( (unsigned int)v529 > 0x40 && v528 )
          j_j___libc_free_0_0(v528);
        if ( v527.m128i_i32[2] > 0x40u && v527.m128i_i64[0] )
          j_j___libc_free_0_0(v527.m128i_i64[0]);
        if ( v511 > 0x40 && v510 )
          j_j___libc_free_0_0(v510);
        if ( v509 <= 0x40 )
          goto LABEL_118;
        v64 = v508;
        if ( !v508 )
          goto LABEL_118;
        goto LABEL_117;
      case 48:
      case 49:
      case 50:
        v55 = *(_QWORD *)(a2 + 32);
        v56 = v502;
        a7 = _mm_loadu_si128((const __m128i *)v55);
        a8 = _mm_loadu_si128((const __m128i *)(v55 + 40));
        if ( v502 > 0x40 )
        {
          v56 = sub_16A57B0((__int64)&v501);
        }
        else if ( v501 )
        {
          _BitScanReverse64(&v57, v501);
          v56 = v502 - 64 + (v57 ^ 0x3F);
        }
        v517 = v14;
        v58 = v14 - v56;
        if ( v14 > 0x40 )
        {
          v15 = (__m128i *)&v516;
          v458 = v56;
          sub_16A4EF0((__int64)&v516, 0, 0);
          v56 = v458;
        }
        else
        {
          v516 = 0;
          v15 = (__m128i *)&v516;
        }
        if ( v58 )
        {
          if ( v58 > 0x40 )
          {
            v462 = v56;
            sub_16A5260(&v516, 0, v58);
            v56 = v462;
          }
          else
          {
            v59 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v56 - (unsigned __int8)v14 + 64);
            if ( v517 > 0x40 )
              *(_QWORD *)v516 |= v59;
            else
              v516 |= v59;
          }
        }
        v440 = v56;
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                a7.m128i_i32[0],
                                a7.m128i_i32[2],
                                (unsigned int)&v516,
                                (unsigned int)&v519,
                                a6,
                                a10 + 1,
                                0)
          || (unsigned __int8)sub_20A2AF0(
                                a1,
                                a8.m128i_i32[0],
                                a8.m128i_i32[2],
                                (unsigned int)&v516,
                                (unsigned int)&v519,
                                a6,
                                a10 + 1,
                                0)
          || (unsigned __int8)sub_20A2630(
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9,
                                a1,
                                a2,
                                a3,
                                v14,
                                (__int64)&v501,
                                (__int64 *)a6) )
        {
          v60 = *(_WORD *)(a2 + 80);
          v61 = v60;
          if ( (*(_BYTE *)(a2 + 80) & 4) != 0 || (*(_BYTE *)(a2 + 80) & 2) != 0 )
          {
            LOBYTE(v61) = v60 & 0xF8 | 1;
            v62 = sub_1D332F0(
                    *(__int64 **)a6,
                    *(unsigned __int16 *)(a2 + 24),
                    (__int64)&v503,
                    v505.m128i_u32[0],
                    (const void **)v505.m128i_i64[1],
                    v61,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    a7.m128i_i64[0],
                    a7.m128i_u64[1],
                    *(_OWORD *)&a8);
            *(_QWORD *)(a6 + 16) = a2;
            *(_DWORD *)(a6 + 24) = a3;
            *(_QWORD *)(a6 + 32) = v62;
            *(_DWORD *)(a6 + 40) = v63;
          }
          goto LABEL_115;
        }
        v44 = sub_1D1ADA0(a8.m128i_i64[0], a8.m128i_u32[2], v40, v41, v42, v43);
        sub_171A350((__int64)&v518, v502, v440);
        if ( !v44 )
          goto LABEL_70;
        v45 = *(_QWORD *)(v44 + 88);
        if ( sub_1454FB0(v45 + 24) )
          goto LABEL_70;
        v46 = *(_DWORD *)(v45 + 32);
        v47 = v45 + 24;
        if ( v46 <= 0x40 )
        {
          if ( *(_QWORD *)(v45 + 24) == 1 )
            goto LABEL_70;
        }
        else
        {
          v47 = v45 + 24;
          if ( (unsigned int)sub_16A57B0(v45 + 24) == v46 - 1 )
            goto LABEL_70;
        }
        sub_13A38D0((__int64)&v527, v47);
        sub_20A0CC0(v527.m128i_i64, v518.m128i_i64);
        v366 = v527.m128i_i32[2];
        v527.m128i_i32[2] = 0;
        v530.m128i_i32[2] = v366;
        v530.m128i_i64[0] = v527.m128i_i64[0];
        v367 = sub_1454FB0((__int64)&v530);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( v367 )
        {
          *(_QWORD *)&v368 = sub_1D389D0(
                               *(_QWORD *)a6,
                               (__int64)&v503,
                               v505.m128i_u32[0],
                               (const void **)v505.m128i_i64[1],
                               0,
                               0,
                               a7,
                               *(double *)a8.m128i_i64,
                               a9);
          v369 = sub_1D332F0(
                   *(__int64 **)a6,
                   *(unsigned __int16 *)(a2 + 24),
                   (__int64)&v503,
                   v505.m128i_u32[0],
                   (const void **)v505.m128i_i64[1],
                   1u,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   a7.m128i_i64[0],
                   a7.m128i_u64[1],
                   v368);
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v369;
          *(_DWORD *)(a6 + 40) = v370;
          sub_135E100(v518.m128i_i64);
          goto LABEL_115;
        }
LABEL_70:
        sub_135E100(v518.m128i_i64);
        sub_135E100((__int64 *)&v516);
LABEL_71:
        sub_1D1F820(*(_QWORD *)a6, a2, a3, a5, a10);
        goto LABEL_72;
      case 100:
        v137 = *((_DWORD *)a5 + 2);
        if ( v137 <= 0x40 )
        {
          *a5 = -1;
          v138 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v137;
LABEL_237:
          *a5 &= v138;
          goto LABEL_238;
        }
        memset((void *)*a5, -1, 8 * (((unsigned __int64)v137 + 63) >> 6));
        v236 = *((unsigned int *)a5 + 2);
        v138 = 0xFFFFFFFFFFFFFFFFLL >> -*((_BYTE *)a5 + 8);
        if ( (unsigned int)v236 <= 0x40 )
          goto LABEL_237;
        v237 = (unsigned int)((unsigned __int64)(v236 + 63) >> 6) - 1;
        *(_QWORD *)(*a5 + 8 * v237) &= v138;
LABEL_238:
        v139 = *((_DWORD *)a5 + 6);
        if ( v139 <= 0x40 )
        {
          a5[2] = -1;
          v140 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v139;
LABEL_240:
          a5[2] &= v140;
          goto LABEL_241;
        }
        memset((void *)a5[2], -1, 8 * (((unsigned __int64)v139 + 63) >> 6));
        v238 = *((unsigned int *)a5 + 6);
        v140 = 0xFFFFFFFFFFFFFFFFLL >> -*((_BYTE *)a5 + 24);
        if ( (unsigned int)v238 <= 0x40 )
          goto LABEL_240;
        v239 = (unsigned int)((unsigned __int64)(v238 + 63) >> 6) - 1;
        *(_QWORD *)(a5[2] + 8 * v239) &= v140;
LABEL_241:
        v141 = *(_QWORD *)(a2 + 32);
        v142 = 5 * *(_DWORD *)(a2 + 56);
        v143 = v141 + 40LL * *(unsigned int *)(a2 + 56);
        if ( v141 == v143 )
          goto LABEL_89;
        while ( 1 )
        {
          LOBYTE(v142) = *(_WORD *)(*(_QWORD *)v141 + 24LL) == 10 || *(_WORD *)(*(_QWORD *)v141 + 24LL) == 32;
          if ( !(_BYTE)v142 )
          {
            v530.m128i_i32[2] = v14;
            LODWORD(v15) = v142;
            if ( v14 > 0x40 )
            {
              sub_16A4EF0((__int64)&v530, 0, 0);
              LODWORD(v532) = v14;
              sub_16A4EF0((__int64)&v531, 0, 0);
            }
            else
            {
              v530.m128i_i64[0] = 0;
              LODWORD(v532) = v14;
              v531 = 0;
            }
            if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
              j_j___libc_free_0_0(*a5);
            v20 = *((_DWORD *)a5 + 6) <= 0x40u;
            *a5 = v530.m128i_i64[0];
            v221 = v530.m128i_i32[2];
            v530.m128i_i32[2] = 0;
            *((_DWORD *)a5 + 2) = v221;
            if ( v20 || (v222 = a5[2]) == 0 )
            {
              a5[2] = v531;
              *((_DWORD *)a5 + 6) = v532;
            }
            else
            {
              j_j___libc_free_0_0(v222);
              v20 = v530.m128i_i32[2] <= 0x40u;
              a5[2] = v531;
              *((_DWORD *)a5 + 6) = v532;
              if ( !v20 && v530.m128i_i64[0] )
                j_j___libc_free_0_0(v530.m128i_i64[0]);
            }
            goto LABEL_90;
          }
          v144 = *(_QWORD *)(*(_QWORD *)v141 + 88LL);
          if ( (unsigned int)v522 <= 0x40 )
          {
            v145 = *(_DWORD *)(v144 + 32);
            if ( v145 <= 0x40 )
              break;
          }
          sub_16A51C0((__int64)&v521, v144 + 24);
          v145 = v522;
          v530.m128i_i32[2] = v522;
          if ( (unsigned int)v522 <= 0x40 )
            goto LABEL_246;
          sub_16A4FD0((__int64)&v530, (const void **)&v521);
          v145 = v530.m128i_u32[2];
          if ( v530.m128i_i32[2] <= 0x40u )
          {
            v146 = v530.m128i_i64[0];
            goto LABEL_247;
          }
          sub_16A8F40(v530.m128i_i64);
          v147 = v530.m128i_i64[0];
          v145 = v530.m128i_u32[2];
LABEL_248:
          v530.m128i_i32[2] = 0;
          if ( (unsigned int)v520 > 0x40 && v519 )
          {
            v491 = v147;
            j_j___libc_free_0_0(v519);
            LODWORD(v520) = v145;
            v519 = v491;
            if ( v530.m128i_i32[2] > 0x40u && v530.m128i_i64[0] )
              j_j___libc_free_0_0(v530.m128i_i64[0]);
          }
          else
          {
            v519 = v147;
            LODWORD(v520) = v145;
          }
          if ( (_DWORD)v522 != v14 )
          {
            sub_16A5A50((__int64)&v530, (__int64 *)&v521, v14);
            sub_16A5A50((__int64)&v527, (__int64 *)&v519, v14);
            v148 = v527.m128i_i32[2];
            v149 = v527.m128i_i64[0];
            v150 = v530.m128i_i32[2];
            v151 = v530.m128i_i64[0];
            if ( (unsigned int)v520 > 0x40 && v519 )
            {
              v489 = v530.m128i_i64[0];
              v479 = v527.m128i_i64[0];
              v492 = v527.m128i_i32[2];
              j_j___libc_free_0_0(v519);
              v151 = v489;
              v149 = v479;
              v148 = v492;
            }
            v519 = v149;
            LODWORD(v520) = v148;
            if ( (unsigned int)v522 > 0x40 && v521 )
            {
              v493 = v151;
              j_j___libc_free_0_0(v521);
              v151 = v493;
            }
            v521 = v151;
            LODWORD(v522) = v150;
          }
          if ( *((_DWORD *)a5 + 6) > 0x40u )
            sub_16A8890((__int64 *)v470, (__int64 *)&v521);
          else
            a5[2] &= v521;
          if ( *((_DWORD *)a5 + 2) > 0x40u )
          {
            sub_16A8890(a5, (__int64 *)&v519);
          }
          else
          {
            v142 = v519;
            *a5 &= v519;
          }
          v141 += 40;
          if ( v143 == v141 )
            goto LABEL_89;
        }
        v270 = *(_QWORD *)(v144 + 24);
        LODWORD(v522) = *(_DWORD *)(v144 + 32);
        v521 = v270 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v145);
LABEL_246:
        v146 = v521;
LABEL_247:
        v147 = ~v146 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v145);
        v530.m128i_i64[0] = v147;
        goto LABEL_248;
      case 114:
        v163 = *(_QWORD *)(a2 + 48);
        v164 = *(_QWORD *)(a2 + 32);
        if ( v163 )
        {
          if ( !*(_QWORD *)(v163 + 32) && (unsigned int)*(unsigned __int16 *)(*(_QWORD *)(v163 + 16) + 24LL) - 123 <= 1 )
          {
            v264 = *(_QWORD *)(v164 + 40);
            v32 = *(unsigned __int16 *)(v264 + 24);
            if ( ((_DWORD)v32 == 10 || (_DWORD)v32 == 32)
              && sub_179D670((_DWORD *)(*(_QWORD *)(v264 + 88) + 24LL), 0xFFFFFFFFFFFFFFFFLL) == 31 )
            {
              LODWORD(v15) = 0;
              goto LABEL_93;
            }
          }
        }
        v165 = sub_1D1ADA0(*(_QWORD *)(v164 + 40), *(_QWORD *)(v164 + 48), v32, v30, v18, v19);
        if ( !v165 )
          goto LABEL_474;
        v166 = *(__int64 **)(a2 + 32);
        v167 = *v166;
        v168 = v166[1];
        LODWORD(v166) = *((_DWORD *)v166 + 2);
        v530.m128i_i64[0] = 0;
        v444 = v168;
        v437 = (int)v166;
        v466 = v167;
        v169 = *(_QWORD *)a6;
        v450 = v167;
        v530.m128i_i64[1] = 1;
        v531 = 0;
        v532 = 1;
        sub_1D1F820(v169, v167, v168, (unsigned __int64 *)&v530, a10);
        v170 = *(_QWORD *)(v165 + 88);
        v517 = *(_DWORD *)(v170 + 32);
        v171 = v517;
        if ( v517 <= 0x40 )
        {
          v172 = *(_QWORD *)(v170 + 24);
LABEL_292:
          v517 = 0;
          v173 = ~v172 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v171);
          v516 = v173;
LABEL_293:
          v174 = v501 & v173;
          v518.m128i_i64[0] = v174;
          goto LABEL_294;
        }
        sub_16A4FD0((__int64)&v516, (const void **)(v170 + 24));
        v171 = v517;
        if ( v517 <= 0x40 )
        {
          v172 = v516;
          goto LABEL_292;
        }
        sub_16A8F40((__int64 *)&v516);
        v171 = v517;
        v173 = v516;
        v517 = 0;
        v518.m128i_i32[2] = v171;
        v518.m128i_i64[0] = v516;
        if ( v171 <= 0x40 )
          goto LABEL_293;
        sub_16A8890(v518.m128i_i64, (__int64 *)&v501);
        v174 = v518.m128i_i64[0];
        v171 = v518.m128i_u32[2];
LABEL_294:
        v527.m128i_i32[2] = v171;
        v527.m128i_i64[0] = v174;
        v518.m128i_i32[2] = 0;
        v513 = v530.m128i_u32[2];
        if ( v530.m128i_i32[2] <= 0x40u )
        {
          v175 = v530.m128i_i64[0];
LABEL_296:
          v176 = (const void *)(v501 & v175);
LABEL_297:
          v481 = v176 == (const void *)v174;
          goto LABEL_298;
        }
        sub_16A4FD0((__int64)&v512, (const void **)&v530);
        if ( v513 <= 0x40 )
        {
          v175 = v512;
          v174 = v527.m128i_i64[0];
          goto LABEL_296;
        }
        sub_16A8890((__int64 *)&v512, (__int64 *)&v501);
        v271 = v513;
        v176 = (const void *)v512;
        v513 = 0;
        LODWORD(v515) = v271;
        v514 = v512;
        if ( v271 <= 0x40 )
        {
          v174 = v527.m128i_i64[0];
          goto LABEL_297;
        }
        v481 = sub_16A5220((__int64)&v514, (const void **)&v527);
        if ( v176 )
        {
          j_j___libc_free_0_0(v176);
          if ( v513 > 0x40 )
          {
            if ( v512 )
              j_j___libc_free_0_0(v512);
          }
        }
LABEL_298:
        if ( v527.m128i_i32[2] > 0x40u && v527.m128i_i64[0] )
          j_j___libc_free_0_0(v527.m128i_i64[0]);
        if ( v518.m128i_i32[2] > 0x40u && v518.m128i_i64[0] )
          j_j___libc_free_0_0(v518.m128i_i64[0]);
        if ( v517 > 0x40 && v516 )
          j_j___libc_free_0_0(v516);
        if ( v481 )
        {
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 16) = a2;
          *(_QWORD *)(a6 + 32) = v466;
          *(_DWORD *)(a6 + 40) = v444;
LABEL_309:
          if ( (unsigned int)v532 > 0x40 && v531 )
            j_j___libc_free_0_0(v531);
          if ( v530.m128i_i32[2] > 0x40u )
          {
            v64 = v530.m128i_i64[0];
            if ( v530.m128i_i64[0] )
LABEL_117:
              j_j___libc_free_0_0(v64);
          }
          goto LABEL_118;
        }
        sub_13A38D0((__int64)&v516, (__int64)&v530);
        v240 = v517;
        if ( v517 <= 0x40 )
        {
          v517 = 0;
          v241 = ~v516 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v240);
          v516 = v241;
LABEL_470:
          v242 = v501 & v241;
          v518.m128i_i64[0] = v242;
          goto LABEL_471;
        }
        sub_16A8F40((__int64 *)&v516);
        v240 = v517;
        v241 = v516;
        v517 = 0;
        v518.m128i_i32[2] = v240;
        v518.m128i_i64[0] = v516;
        if ( v240 <= 0x40 )
          goto LABEL_470;
        sub_16A8890(v518.m128i_i64, (__int64 *)&v501);
        v242 = v518.m128i_i64[0];
        v240 = v518.m128i_u32[2];
LABEL_471:
        v527.m128i_i32[2] = v240;
        v527.m128i_i64[0] = v242;
        v518.m128i_i32[2] = 0;
        v426 = sub_20A2230(a1, a2, a3, (__int64)&v527, a6, a7, *(double *)a8.m128i_i64, a9);
        sub_135E100(v527.m128i_i64);
        sub_135E100(v518.m128i_i64);
        sub_135E100((__int64 *)&v516);
        if ( v426 )
          goto LABEL_309;
        if ( sub_1D18970(v466) && sub_1D18C00(v450, 1, v437) )
        {
          sub_13A38D0((__int64)&v518, *(_QWORD *)(v165 + 88) + 24LL);
          sub_13D0570((__int64)&v518);
          v359 = v518.m128i_i32[2];
          v518.m128i_i32[2] = 0;
          v527.m128i_i32[2] = v359;
          v527.m128i_i64[0] = v518.m128i_i64[0];
          v360 = sub_1455820((__int64)&v531, &v527);
          sub_135E100(v527.m128i_i64);
          sub_135E100(v518.m128i_i64);
          if ( v360 )
          {
            v361 = sub_1D332F0(
                     *(__int64 **)a6,
                     120,
                     (__int64)&v503,
                     v505.m128i_u32[0],
                     (const void **)v505.m128i_i64[1],
                     0,
                     *(double *)a7.m128i_i64,
                     *(double *)a8.m128i_i64,
                     a9,
                     **(_QWORD **)(v450 + 32),
                     *(_QWORD *)(*(_QWORD *)(v450 + 32) + 8LL),
                     *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
            *(_QWORD *)(a6 + 16) = a2;
            *(_DWORD *)(a6 + 24) = a3;
            *(_QWORD *)(a6 + 32) = v361;
            *(_DWORD *)(a6 + 40) = v362;
            goto LABEL_309;
          }
        }
        sub_135E100(&v531);
        sub_135E100(v530.m128i_i64);
LABEL_474:
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                                (unsigned int)&v501,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_118;
        sub_13A38D0((__int64)&v518, (__int64)a5);
        v243 = v518.m128i_u32[2];
        if ( v518.m128i_i32[2] <= 0x40u )
        {
          v518.m128i_i32[2] = 0;
          v244 = ~v518.m128i_i64[0] & (0xFFFFFFFFFFFFFFFFLL >> -(char)v243);
          v518.m128i_i64[0] = v244;
LABEL_477:
          v245 = v501 & v244;
          v527.m128i_i64[0] = v245;
          goto LABEL_478;
        }
        sub_16A8F40(v518.m128i_i64);
        v243 = v518.m128i_u32[2];
        v244 = v518.m128i_i64[0];
        v518.m128i_i32[2] = 0;
        v527.m128i_i32[2] = v243;
        v527.m128i_i64[0] = v518.m128i_i64[0];
        if ( v243 <= 0x40 )
          goto LABEL_477;
        sub_16A8890(v527.m128i_i64, (__int64 *)&v501);
        v245 = v527.m128i_i64[0];
        v243 = v527.m128i_u32[2];
LABEL_478:
        v530.m128i_i64[0] = v245;
        v246 = *(_QWORD **)(a2 + 32);
        v15 = (__m128i *)&v519;
        v530.m128i_i32[2] = v243;
        v527.m128i_i32[2] = 0;
        v247 = sub_20A2AF0(a1, *v246, v246[1], (unsigned int)&v530, (unsigned int)&v519, a6, a10 + 1, 0);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        sub_135E100(v518.m128i_i64);
        if ( v247 )
          goto LABEL_118;
        sub_13A38D0((__int64)&v527, (__int64)&v519);
        v248 = v527.m128i_i32[2];
        if ( v527.m128i_i32[2] > 0x40u )
        {
          sub_16A89F0(v527.m128i_i64, (__int64 *)v470);
          v249 = v527.m128i_i64[0];
          v248 = v527.m128i_i32[2];
        }
        else
        {
          v249 = a5[2] | v527.m128i_i64[0];
          v527.m128i_i64[0] = v249;
        }
        v530.m128i_i32[2] = v248;
        v530.m128i_i64[0] = v249;
        v527.m128i_i32[2] = 0;
        if ( v502 <= 0x40 )
          LOBYTE(v15) = (v501 & ~v249) == 0;
        else
          LODWORD(v15) = sub_16A5A00((__int64 *)&v501, v530.m128i_i64);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( (_BYTE)v15 )
        {
          v500 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v500.m128i_i64[0];
          *(_DWORD *)(a6 + 40) = v500.m128i_i32[2];
          goto LABEL_90;
        }
        sub_13A38D0((__int64)&v527, (__int64)a5);
        sub_20A0CC0(v527.m128i_i64, (__int64 *)&v521);
        v363 = v527.m128i_i32[2];
        v527.m128i_i32[2] = 0;
        v530.m128i_i32[2] = v363;
        v530.m128i_i64[0] = v527.m128i_i64[0];
        LOBYTE(v364) = sub_13D0550((__int64)&v501, &v530);
        LODWORD(v15) = v364;
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( (_BYTE)v15 )
        {
          v499 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL));
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v499.m128i_i64[0];
          *(_DWORD *)(a6 + 40) = v499.m128i_i32[2];
          goto LABEL_90;
        }
        sub_13A38D0((__int64)&v527, (__int64)a5);
        sub_20A0CC0(v527.m128i_i64, (__int64 *)&v519);
        v406 = v527.m128i_i32[2];
        v527.m128i_i32[2] = 0;
        v530.m128i_i32[2] = v406;
        v530.m128i_i64[0] = v527.m128i_i64[0];
        LOBYTE(v407) = sub_13D0550((__int64)&v501, &v530);
        LODWORD(v15) = v407;
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( (_BYTE)v15 )
        {
          v408 = sub_1D38BB0(
                   *(_QWORD *)a6,
                   0,
                   (__int64)&v503,
                   v505.m128i_u32[0],
                   (const void **)v505.m128i_i64[1],
                   0,
                   a7,
                   *(double *)a8.m128i_i64,
                   a9,
                   0);
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v408;
          *(_DWORD *)(a6 + 40) = v409;
          goto LABEL_90;
        }
        sub_13A38D0((__int64)&v518, (__int64)&v519);
        sub_13D0570((__int64)&v518);
        v417 = v518.m128i_i32[2];
        v518.m128i_i32[2] = 0;
        v527.m128i_i32[2] = v417;
        v527.m128i_i64[0] = v518.m128i_i64[0];
        sub_20A0CE0(v527.m128i_i64, (__int64 *)&v501);
        v418 = v527.m128i_i32[2];
        v527.m128i_i32[2] = 0;
        v530.m128i_i32[2] = v418;
        v530.m128i_i64[0] = v527.m128i_i64[0];
        LODWORD(v15) = sub_20A2230(a1, a2, a3, (__int64)&v530, a6, a7, *(double *)a8.m128i_i64, a9);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        sub_135E100(v518.m128i_i64);
        if ( !(_BYTE)v15
          && !(unsigned __int8)sub_20A2630(
                                 *(double *)a7.m128i_i64,
                                 *(double *)a8.m128i_i64,
                                 a9,
                                 a1,
                                 a2,
                                 a3,
                                 v14,
                                 (__int64)&v501,
                                 (__int64 *)a6) )
        {
          sub_20A0CE0((__int64 *)v470, (__int64 *)&v521);
          sub_20A0CC0(a5, (__int64 *)&v519);
          goto LABEL_72;
        }
        goto LABEL_118;
      case 115:
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                                (unsigned int)&v501,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_118;
        v518.m128i_i32[2] = *((_DWORD *)a5 + 6);
        v177 = v518.m128i_u32[2];
        if ( v518.m128i_i32[2] <= 0x40u )
        {
          v178 = a5[2];
LABEL_318:
          v518.m128i_i32[2] = 0;
          v179 = ~v178 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v177);
          v518.m128i_i64[0] = v179;
LABEL_319:
          v180 = v501 & v179;
          v527.m128i_i64[0] = v180;
          goto LABEL_320;
        }
        sub_16A4FD0((__int64)&v518, v470);
        v177 = v518.m128i_u32[2];
        if ( v518.m128i_i32[2] <= 0x40u )
        {
          v178 = v518.m128i_i64[0];
          goto LABEL_318;
        }
        sub_16A8F40(v518.m128i_i64);
        v177 = v518.m128i_u32[2];
        v179 = v518.m128i_i64[0];
        v518.m128i_i32[2] = 0;
        v527.m128i_i32[2] = v177;
        v527.m128i_i64[0] = v518.m128i_i64[0];
        if ( v177 <= 0x40 )
          goto LABEL_319;
        sub_16A8890(v527.m128i_i64, (__int64 *)&v501);
        v180 = v527.m128i_i64[0];
        v177 = v527.m128i_u32[2];
LABEL_320:
        v530.m128i_i64[0] = v180;
        v181 = *(_QWORD **)(a2 + 32);
        v530.m128i_i32[2] = v177;
        v527.m128i_i32[2] = 0;
        v182 = sub_20A2AF0(a1, *v181, v181[1], (unsigned int)&v530, (unsigned int)&v519, a6, a10 + 1, 0);
        LODWORD(v15) = v420;
        v183 = v182;
        if ( v530.m128i_i32[2] > 0x40u && v530.m128i_i64[0] )
          j_j___libc_free_0_0(v530.m128i_i64[0]);
        sub_135E100(v527.m128i_i64);
        if ( v518.m128i_i32[2] > 0x40u && v518.m128i_i64[0] )
          j_j___libc_free_0_0(v518.m128i_i64[0]);
        if ( v183 )
          goto LABEL_118;
        sub_13A38D0((__int64)&v527, (__int64)&v521);
        v184 = v527.m128i_i32[2];
        if ( v527.m128i_i32[2] > 0x40u )
        {
          sub_16A89F0(v527.m128i_i64, a5);
          v184 = v527.m128i_i32[2];
          v185 = v527.m128i_i64[0];
        }
        else
        {
          v185 = *a5 | v527.m128i_i64[0];
          v527.m128i_i64[0] = v185;
        }
        v530.m128i_i32[2] = v184;
        v530.m128i_i64[0] = v185;
        v527.m128i_i32[2] = 0;
        if ( v502 <= 0x40 )
          LOBYTE(v15) = (v501 & ~v185) == 0;
        else
          LODWORD(v15) = sub_16A5A00((__int64 *)&v501, v530.m128i_i64);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( (_BYTE)v15 )
        {
          v498 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v498.m128i_i64[0];
          *(_DWORD *)(a6 + 40) = v498.m128i_i32[2];
          goto LABEL_90;
        }
        sub_13A38D0((__int64)&v527, (__int64)v470);
        v331 = v527.m128i_i32[2];
        if ( v527.m128i_i32[2] > 0x40u )
        {
          sub_16A89F0(v527.m128i_i64, (__int64 *)&v519);
          v331 = v527.m128i_i32[2];
          v332 = v527.m128i_i64[0];
        }
        else
        {
          v332 = v519 | v527.m128i_i64[0];
          v527.m128i_i64[0] |= v519;
        }
        v530.m128i_i32[2] = v331;
        v530.m128i_i64[0] = v332;
        v527.m128i_i32[2] = 0;
        if ( v502 <= 0x40 )
          LOBYTE(v15) = (v501 & ~v332) == 0;
        else
          LODWORD(v15) = sub_16A5A00((__int64 *)&v501, v530.m128i_i64);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( (_BYTE)v15 )
        {
          v497 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL));
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v497.m128i_i64[0];
          *(_DWORD *)(a6 + 40) = v497.m128i_i32[2];
          goto LABEL_90;
        }
        LODWORD(v15) = a1;
        if ( (unsigned __int8)sub_20A2230(a1, a2, a3, (__int64)&v501, a6, a7, *(double *)a8.m128i_i64, a9)
          || (unsigned __int8)sub_20A2630(
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9,
                                a1,
                                a2,
                                a3,
                                v14,
                                (__int64)&v501,
                                (__int64 *)a6) )
        {
          goto LABEL_118;
        }
        sub_20A0CE0(a5, (__int64 *)&v519);
        sub_20A0CC0((__int64 *)v470, (__int64 *)&v521);
        goto LABEL_72;
      case 116:
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                                (unsigned int)&v501,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0)
          || (unsigned __int8)sub_20A2AF0(
                                a1,
                                **(_QWORD **)(a2 + 32),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                (unsigned int)&v501,
                                (unsigned int)&v519,
                                a6,
                                a10 + 1,
                                0) )
        {
          goto LABEL_118;
        }
        if ( v502 <= 0x40 )
        {
          if ( (v501 & ~*a5) != 0 )
          {
            if ( (v501 & ~v519) == 0 )
              goto LABEL_338;
LABEL_628:
            if ( !(unsigned __int8)sub_20A2630(
                                     *(double *)a7.m128i_i64,
                                     *(double *)a8.m128i_i64,
                                     a9,
                                     a1,
                                     a2,
                                     a3,
                                     v14,
                                     (__int64)&v501,
                                     (__int64 *)a6) )
            {
              sub_13A38D0((__int64)&v518, (__int64)&v519);
              sub_13D0570((__int64)&v518);
              v527.m128i_i32[2] = v518.m128i_i32[2];
              v518.m128i_i32[2] = 0;
              v527.m128i_i64[0] = v518.m128i_i64[0];
              sub_13A38D0((__int64)&v512, (__int64)a5);
              sub_13D0570((__int64)&v512);
              v326 = v513;
              v513 = 0;
              LODWORD(v515) = v326;
              v514 = v512;
              sub_20A0CE0((__int64 *)&v514, (__int64 *)&v501);
              v517 = v515;
              LODWORD(v515) = 0;
              v516 = v514;
              sub_20A0CE0(v527.m128i_i64, (__int64 *)&v516);
              v530.m128i_i32[2] = v527.m128i_i32[2];
              v527.m128i_i32[2] = 0;
              v530.m128i_i64[0] = v527.m128i_i64[0];
              LOBYTE(v327) = sub_13A38F0((__int64)&v530, 0);
              LODWORD(v15) = v327;
              sub_135E100(v530.m128i_i64);
              sub_135E100((__int64 *)&v516);
              sub_135E100((__int64 *)&v514);
              sub_135E100((__int64 *)&v512);
              sub_135E100(v527.m128i_i64);
              sub_135E100(v518.m128i_i64);
              if ( (_BYTE)v15 )
              {
                v328 = sub_1D332F0(
                         *(__int64 **)a6,
                         119,
                         (__int64)&v503,
                         v505.m128i_u32[0],
                         (const void **)v505.m128i_i64[1],
                         0,
                         *(double *)a7.m128i_i64,
                         *(double *)a8.m128i_i64,
                         a9,
                         **(_QWORD **)(a2 + 32),
                         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                         *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
                *(_QWORD *)(a6 + 16) = a2;
                *(_DWORD *)(a6 + 24) = a3;
                *(_QWORD *)(a6 + 32) = v328;
                *(_DWORD *)(a6 + 40) = v329;
                goto LABEL_90;
              }
              v15 = (__m128i *)&v521;
              sub_13A38D0((__int64)&v518, (__int64)v470);
              sub_20A0CE0(v518.m128i_i64, (__int64 *)&v521);
              v371 = v518.m128i_i32[2];
              v518.m128i_i32[2] = 0;
              v527.m128i_i32[2] = v371;
              v527.m128i_i64[0] = v518.m128i_i64[0];
              sub_13A38D0((__int64)&v514, (__int64)a5);
              sub_20A0CE0((__int64 *)&v514, (__int64 *)&v519);
              v372 = v515;
              LODWORD(v515) = 0;
              v517 = v372;
              v516 = v514;
              sub_20A0CC0(v527.m128i_i64, (__int64 *)&v516);
              v373 = v527.m128i_i32[2];
              v527.m128i_i32[2] = 0;
              v530.m128i_i32[2] = v373;
              v530.m128i_i64[0] = v527.m128i_i64[0];
              sub_14A9CA0(&v523, v530.m128i_i64);
              sub_135E100(v530.m128i_i64);
              sub_135E100((__int64 *)&v516);
              sub_135E100((__int64 *)&v514);
              sub_135E100(v527.m128i_i64);
              sub_135E100(v518.m128i_i64);
              sub_13A38D0((__int64)&v518, (__int64)v470);
              sub_20A0CE0(v518.m128i_i64, (__int64 *)&v519);
              v374 = v518.m128i_i32[2];
              v518.m128i_i32[2] = 0;
              v527.m128i_i32[2] = v374;
              v527.m128i_i64[0] = v518.m128i_i64[0];
              sub_13A38D0((__int64)&v514, (__int64)a5);
              sub_20A0CE0((__int64 *)&v514, (__int64 *)&v521);
              v375 = v515;
              LODWORD(v515) = 0;
              v517 = v375;
              v516 = v514;
              sub_20A0CC0(v527.m128i_i64, (__int64 *)&v516);
              v376 = v527.m128i_i32[2];
              v527.m128i_i32[2] = 0;
              v530.m128i_i32[2] = v376;
              v530.m128i_i64[0] = v527.m128i_i64[0];
              sub_14A9CA0(&v525, v530.m128i_i64);
              sub_135E100(v530.m128i_i64);
              sub_135E100((__int64 *)&v516);
              sub_135E100((__int64 *)&v514);
              sub_135E100(v527.m128i_i64);
              sub_135E100(v518.m128i_i64);
              sub_13A38D0((__int64)&v527, (__int64)a5);
              sub_20A0CC0(v527.m128i_i64, (__int64 *)v470);
              v377 = v527.m128i_i32[2];
              v527.m128i_i32[2] = 0;
              v530.m128i_i32[2] = v377;
              v530.m128i_i64[0] = v527.m128i_i64[0];
              v468 = sub_13D0550((__int64)&v501, &v530);
              sub_135E100(v530.m128i_i64);
              sub_135E100(v527.m128i_i64);
              if ( v468 )
              {
                LOBYTE(v382) = sub_1455820((__int64)v470, &v521);
                LODWORD(v15) = v382;
                if ( (_BYTE)v382 )
                {
                  v474 = *(_QWORD *)a6;
                  sub_13A38D0((__int64)&v518, (__int64)v470);
                  sub_13D0570((__int64)&v518);
                  v413 = v518.m128i_i32[2];
                  v518.m128i_i32[2] = 0;
                  v527.m128i_i32[2] = v413;
                  v527.m128i_i64[0] = v518.m128i_i64[0];
                  sub_20A0CE0(v527.m128i_i64, (__int64 *)&v501);
                  v530.m128i_i32[2] = v527.m128i_i32[2];
                  v527.m128i_i32[2] = 0;
                  v530.m128i_i64[0] = v527.m128i_i64[0];
                  *(_QWORD *)&v475 = sub_1D38970(
                                       v474,
                                       (__int64)&v530,
                                       (__int64)&v503,
                                       v505.m128i_u32[0],
                                       (const void **)v505.m128i_i64[1],
                                       0,
                                       a7,
                                       *(double *)a8.m128i_i64,
                                       a9,
                                       0);
                  *((_QWORD *)&v475 + 1) = v414;
                  sub_135E100(v530.m128i_i64);
                  sub_135E100(v527.m128i_i64);
                  sub_135E100(v518.m128i_i64);
                  v415 = sub_1D332F0(
                           *(__int64 **)a6,
                           118,
                           (__int64)&v503,
                           v505.m128i_u32[0],
                           (const void **)v505.m128i_i64[1],
                           0,
                           *(double *)a7.m128i_i64,
                           *(double *)a8.m128i_i64,
                           a9,
                           **(_QWORD **)(a2 + 32),
                           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                           v475);
                  *(_QWORD *)(a6 + 16) = a2;
                  *(_DWORD *)(a6 + 24) = a3;
                  *(_QWORD *)(a6 + 32) = v415;
                  *(_DWORD *)(a6 + 40) = v416;
LABEL_90:
                  if ( (unsigned int)v526 > 0x40 && v525 )
                    j_j___libc_free_0_0(v525);
LABEL_93:
                  if ( (unsigned int)v524 > 0x40 && v523 )
                    j_j___libc_free_0_0(v523);
                  if ( (unsigned int)v522 > 0x40 && v521 )
                    j_j___libc_free_0_0(v521);
                  if ( (unsigned int)v520 > 0x40 && v519 )
                    j_j___libc_free_0_0(v519);
                  goto LABEL_27;
                }
              }
              v383 = sub_1D1ADA0(
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                       v378,
                       v379,
                       v380,
                       v381);
              if ( !v383 )
                goto LABEL_755;
              v384 = (_QWORD *)(*(_QWORD *)(v383 + 88) + 24LL);
              if ( sub_1454FB0((__int64)v384) )
                goto LABEL_755;
              LOBYTE(v385) = sub_13D0550((__int64)&v501, v384);
              LODWORD(v15) = v385;
              if ( (_BYTE)v385 )
              {
                v386 = sub_1D3C080(
                         *(__int64 **)a6,
                         (__int64)&v503,
                         **(_QWORD **)(a2 + 32),
                         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                         v505.m128i_u32[0],
                         (const void **)v505.m128i_i64[1],
                         a7,
                         *(double *)a8.m128i_i64,
                         a9);
                *(_QWORD *)(a6 + 16) = a2;
                *(_DWORD *)(a6 + 24) = a3;
                *(_QWORD *)(a6 + 32) = v386;
                *(_DWORD *)(a6 + 40) = v387;
                goto LABEL_90;
              }
              if ( !(unsigned __int8)sub_20A2230(a1, a2, a3, (__int64)&v501, a6, a7, *(double *)a8.m128i_i64, a9) )
              {
LABEL_755:
                if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
                  j_j___libc_free_0_0(*a5);
                v20 = *((_DWORD *)a5 + 6) <= 0x40u;
                *a5 = v523;
                v410 = v524;
                LODWORD(v524) = 0;
                *((_DWORD *)a5 + 2) = v410;
                if ( !v20 )
                {
                  v411 = a5[2];
                  if ( v411 )
                    j_j___libc_free_0_0(v411);
                }
                a5[2] = v525;
                v412 = v526;
                LODWORD(v526) = 0;
                *((_DWORD *)a5 + 6) = v412;
                goto LABEL_72;
              }
            }
LABEL_118:
            LODWORD(v15) = 1;
            goto LABEL_90;
          }
        }
        else if ( !(unsigned __int8)sub_16A5A00((__int64 *)&v501, a5) )
        {
          if ( (unsigned __int8)sub_16A5A00((__int64 *)&v501, (__int64 *)&v519) )
          {
LABEL_338:
            LODWORD(v15) = 1;
            v495 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL));
            *(_QWORD *)(a6 + 16) = a2;
            *(_DWORD *)(a6 + 24) = a3;
            *(_QWORD *)(a6 + 32) = v495.m128i_i64[0];
            *(_DWORD *)(a6 + 40) = v495.m128i_i32[2];
            goto LABEL_90;
          }
          goto LABEL_628;
        }
        LODWORD(v15) = 1;
        v496 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
        *(_QWORD *)(a6 + 16) = a2;
        *(_DWORD *)(a6 + 24) = a3;
        *(_QWORD *)(a6 + 32) = v496.m128i_i64[0];
        *(_DWORD *)(a6 + 40) = v496.m128i_i32[2];
        goto LABEL_90;
      case 118:
        v97 = sub_1D1ADA0(
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                (unsigned __int16)v32,
                v30,
                v18,
                v19);
        if ( !v97 )
          goto LABEL_72;
        v99 = *(_QWORD **)(a2 + 32);
        v100 = *(_QWORD *)(v97 + 88);
        v101 = v14;
        v15 = (__m128i *)*v99;
        v478 = v99[1];
        v102 = *(_DWORD *)(v100 + 32);
        if ( v102 > 0x40 )
        {
          v451 = *(_QWORD **)(a2 + 32);
          v459 = *(_QWORD *)(v97 + 88);
          v250 = sub_16A57B0(v100 + 24);
          v100 = v459;
          v101 = v14;
          v99 = v451;
          if ( v102 - v250 > 0x40 )
            goto LABEL_72;
          v457 = **(_QWORD **)(v459 + 24);
          if ( v14 <= v457 )
            goto LABEL_72;
        }
        else
        {
          v457 = *(_QWORD *)(v100 + 24);
          if ( v14 <= v457 )
            goto LABEL_72;
        }
        v449 = *((_DWORD *)v99 + 2);
        v443 = *v99;
        if ( *(_WORD *)(*v99 + 24LL) == 124 )
        {
          v429 = v101;
          v318 = sub_1D1ADA0(
                   *(_QWORD *)(*(_QWORD *)(*v99 + 32LL) + 40LL),
                   *(_QWORD *)(*(_QWORD *)(*v99 + 32LL) + 48LL),
                   v100,
                   (int)v99,
                   v101,
                   v98);
          if ( v318 )
          {
            if ( (_DWORD)v457 )
            {
              v425 = v429;
              v424 = v318;
              sub_13D0120((__int64)&v527, v14, v457);
              sub_20A0CE0(v527.m128i_i64, (__int64 *)&v501);
              v319 = v527.m128i_i32[2];
              v527.m128i_i32[2] = 0;
              v530.m128i_i32[2] = v319;
              v530.m128i_i64[0] = v527.m128i_i64[0];
              v430 = sub_13A38F0((__int64)&v530, 0);
              sub_135E100(v530.m128i_i64);
              sub_135E100(v527.m128i_i64);
              if ( v430 )
              {
                v438 = *(_QWORD *)(v424 + 88);
                if ( sub_13D0480(v438 + 24, v425) )
                {
                  v320 = *(_QWORD **)(v438 + 24);
                  if ( *(_DWORD *)(v438 + 32) > 0x40u )
                    v320 = (_QWORD *)*v320;
                  v321 = 122;
                  v322 = v457 - (_DWORD)v320;
                  if ( (int)v457 - (int)v320 < 0 )
                  {
                    v322 = (_DWORD)v320 - v457;
                    v321 = 124;
                  }
                  *(_QWORD *)&v323 = sub_1D38BB0(
                                       *(_QWORD *)a6,
                                       v322,
                                       (__int64)&v503,
                                       *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                                                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL)),
                                       *(const void ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                                                       + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL)
                                                       + 8),
                                       0,
                                       a7,
                                       *(double *)a8.m128i_i64,
                                       a9,
                                       0);
                  v324 = sub_1D332F0(
                           *(__int64 **)a6,
                           v321,
                           (__int64)&v503,
                           v505.m128i_u32[0],
                           (const void **)v505.m128i_i64[1],
                           0,
                           *(double *)a7.m128i_i64,
                           *(double *)a8.m128i_i64,
                           a9,
                           **(_QWORD **)(v443 + 32),
                           *(_QWORD *)(*(_QWORD *)(v443 + 32) + 8LL),
                           v323);
                  *(_QWORD *)(a6 + 16) = a2;
                  *(_DWORD *)(a6 + 24) = a3;
                  *(_QWORD *)(a6 + 32) = v324;
                  *(_DWORD *)(a6 + 40) = v325;
                  goto LABEL_118;
                }
              }
            }
          }
        }
        v103 = v502;
        v104 = a10 + 1;
        v530.m128i_i32[2] = v502;
        if ( v502 > 0x40 )
        {
          sub_16A4FD0((__int64)&v530, (const void **)&v501);
          v103 = v530.m128i_u32[2];
          v104 = a10 + 1;
          if ( v530.m128i_i32[2] > 0x40u )
          {
            sub_16A8110((__int64)&v530, v457);
            v104 = a10 + 1;
            goto LABEL_168;
          }
        }
        else
        {
          v530.m128i_i64[0] = v501;
        }
        if ( v103 == (_DWORD)v457 )
          v530.m128i_i64[0] = 0;
        else
          v530.m128i_i64[0] = (unsigned __int64)v530.m128i_i64[0] >> v457;
LABEL_168:
        LODWORD(v15) = sub_20A2AF0(a1, (_DWORD)v15, v478, (unsigned int)&v530, (_DWORD)a5, a6, v104, 0);
        sub_135E100(v530.m128i_i64);
        if ( (_BYTE)v15 )
          goto LABEL_118;
        if ( *(_WORD *)(v443 + 24) != 144 )
          goto LABEL_170;
        v333 = *(const __m128i **)(v443 + 32);
        v334 = v333->m128i_i64[0];
        v335 = _mm_loadu_si128(v333);
        v336 = v333->m128i_u32[2];
        v487 = v334;
        v439 = v336;
        v337 = (unsigned __int8 *)(*(_QWORD *)(v334 + 40) + 16 * v336);
        v338 = *v337;
        v339 = *((_QWORD *)v337 + 1);
        v431 = v335;
        v518.m128i_i8[0] = v338;
        v518.m128i_i64[1] = v339;
        LODWORD(v15) = sub_1D159C0((__int64)&v518, v423, v338, v334, v105, v106);
        if ( (unsigned int)v457 < (unsigned int)v15 )
        {
          v340 = sub_1455840((__int64)&v501);
          if ( (unsigned int)v15 >= v502 - v340 )
          {
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 1136LL))(
                   a1,
                   122,
                   v518.m128i_u32[0],
                   v518.m128i_i64[1]) )
            {
              v392 = sub_1F40B60(a1, v518.m128i_u32[0], v518.m128i_i64[1], v464, 1);
              v527.m128i_i64[1] = v393;
              v527.m128i_i32[0] = v392;
              sub_135E0D0((__int64)&v530, v14, (unsigned int)v457, 0);
              v398 = sub_1D159A0(
                       v527.m128i_i8,
                       v14,
                       v394,
                       v395,
                       v396,
                       v397,
                       v424,
                       v425,
                       v431.m128i_i32[0],
                       v431.m128i_i64[1]);
              v399 = sub_1455840((__int64)&v530);
              v400 = v530.m128i_i32[2] - v399;
              sub_135E100(v530.m128i_i64);
              if ( v398 < v400 )
                v527 = _mm_loadu_si128(&v518);
              v401 = *(__int64 **)a6;
              *(_QWORD *)&v402 = sub_1D38BB0(
                                   *(_QWORD *)a6,
                                   (unsigned int)v457,
                                   (__int64)&v503,
                                   v527.m128i_u32[0],
                                   (const void **)v527.m128i_i64[1],
                                   0,
                                   a7,
                                   *(double *)a8.m128i_i64,
                                   a9,
                                   0);
              *(_QWORD *)&v403 = sub_1D332F0(
                                   v401,
                                   122,
                                   (__int64)&v503,
                                   v518.m128i_u32[0],
                                   (const void **)v518.m128i_i64[1],
                                   0,
                                   *(double *)a7.m128i_i64,
                                   *(double *)a8.m128i_i64,
                                   a9,
                                   v433,
                                   v434,
                                   v402);
              v404 = sub_1D309E0(
                       *(__int64 **)a6,
                       144,
                       (__int64)&v503,
                       v505.m128i_u32[0],
                       (const void **)v505.m128i_i64[1],
                       0,
                       *(double *)a7.m128i_i64,
                       *(double *)a8.m128i_i64,
                       *(double *)a9.m128i_i64,
                       v403);
              *(_QWORD *)(a6 + 16) = a2;
              *(_DWORD *)(a6 + 24) = a3;
              *(_QWORD *)(a6 + 32) = v404;
              *(_DWORD *)(a6 + 40) = v405;
              goto LABEL_118;
            }
          }
        }
        if ( sub_1D18C00(v443, 1, v449) && *(_WORD *)(v487 + 24) == 124 && sub_1D18C00(v487, 1, v439) )
        {
          v345 = sub_1D1ADA0(
                   *(_QWORD *)(*(_QWORD *)(v487 + 32) + 40LL),
                   *(_QWORD *)(*(_QWORD *)(v487 + 32) + 48LL),
                   v341,
                   v342,
                   v343,
                   v344);
          if ( v345 )
          {
            v346 = sub_179D670((_DWORD *)(*(_QWORD *)(v345 + 88) + 24LL), (unsigned int)v15);
            v347 = (unsigned int)v15;
            v473 = v346;
            if ( (unsigned int)v457 <= (unsigned int)v15 )
              v347 = v457;
            if ( v346 < v347 )
            {
              v348 = sub_1455840((__int64)&v501);
              if ( (_DWORD)v457 + (_DWORD)v15 - v473 >= v502 - v348
                && (unsigned int)sub_1455870((__int64 *)&v501) >= (unsigned int)v457 )
              {
                v349 = (const void ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                                      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
                v350 = sub_1D38BB0(
                         *(_QWORD *)a6,
                         (unsigned int)v457 - v473,
                         (__int64)&v503,
                         *(unsigned __int8 *)v349,
                         v349[1],
                         0,
                         a7,
                         *(double *)a8.m128i_i64,
                         a9,
                         0);
                v352 = v351;
                v353 = sub_1D309E0(
                         *(__int64 **)a6,
                         144,
                         (__int64)&v503,
                         v505.m128i_u32[0],
                         (const void **)v505.m128i_i64[1],
                         0,
                         *(double *)a7.m128i_i64,
                         *(double *)a8.m128i_i64,
                         *(double *)a9.m128i_i64,
                         *(_OWORD *)*(_QWORD *)(v487 + 32));
                *((_QWORD *)&v421 + 1) = v352;
                *(_QWORD *)&v421 = v350;
                v355 = sub_1D332F0(
                         *(__int64 **)a6,
                         122,
                         (__int64)&v503,
                         v505.m128i_u32[0],
                         (const void **)v505.m128i_i64[1],
                         0,
                         *(double *)a7.m128i_i64,
                         *(double *)a8.m128i_i64,
                         a9,
                         v353,
                         v354,
                         v421);
                *(_QWORD *)(a6 + 16) = a2;
                *(_DWORD *)(a6 + 24) = a3;
                *(_QWORD *)(a6 + 32) = v355;
                *(_DWORD *)(a6 + 40) = v356;
                goto LABEL_118;
              }
            }
          }
        }
LABEL_170:
        v107 = *((_DWORD *)a5 + 2);
        if ( v107 > 0x40 )
        {
          sub_16A7DC0(a5, v457);
        }
        else
        {
          v108 = 0;
          if ( (_DWORD)v457 != v107 )
            v108 = (*a5 << v457) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v107);
          *a5 = v108;
        }
        v109 = *((_DWORD *)a5 + 6);
        if ( v109 > 0x40 )
        {
          sub_16A7DC0((__int64 *)v470, v457);
        }
        else
        {
          v110 = 0;
          if ( (_DWORD)v457 != v109 )
            v110 = (a5[2] << v457) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v109);
          a5[2] = v110;
        }
        sub_14A9D90((__int64)a5, 0, (unsigned int)v457);
        goto LABEL_72;
      case 119:
        v75 = v502;
        if ( v502 <= 0x40 )
          LOBYTE(v15) = v501 == 1;
        else
          LOBYTE(v15) = v75 - 1 == (unsigned int)sub_16A57B0((__int64)&v501);
        v76 = *(_QWORD *)(a2 + 32);
        if ( (_BYTE)v15 )
        {
          v77 = sub_1D332F0(
                  *(__int64 **)a6,
                  124,
                  (__int64)&v503,
                  v505.m128i_u32[0],
                  (const void **)v505.m128i_i64[1],
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  *(_QWORD *)v76,
                  *(_QWORD *)(v76 + 8),
                  *(_OWORD *)(v76 + 40));
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v77;
          *(_DWORD *)(a6 + 40) = v78;
          goto LABEL_90;
        }
        v226 = sub_1D1ADA0(*(_QWORD *)(v76 + 40), *(_QWORD *)(v76 + 48), v32, v30, v18, v19);
        if ( !v226 )
          goto LABEL_72;
        v227 = *(_QWORD *)(v226 + 88);
        LODWORD(v15) = v14;
        v228 = *(_DWORD *)(v227 + 32);
        if ( v228 > 0x40 )
        {
          v486 = *(_QWORD *)(v226 + 88);
          if ( v228 - (unsigned int)sub_16A57B0(v227 + 24) > 0x40 )
            goto LABEL_72;
          v229 = **(_QWORD **)(v486 + 24);
          if ( v14 <= v229 )
            goto LABEL_72;
        }
        else
        {
          v229 = *(_QWORD *)(v227 + 24);
          if ( v14 <= v229 )
            goto LABEL_72;
        }
        LODWORD(v15) = v229;
        sub_13A38D0((__int64)&v530, (__int64)&v501);
        if ( v530.m128i_i32[2] > 0x40u )
        {
          sub_16A7DC0(v530.m128i_i64, v229);
        }
        else
        {
          v230 = 0;
          if ( (_DWORD)v229 != v530.m128i_i32[2] )
            v230 = v530.m128i_i64[0] << v229;
          v530.m128i_i64[0] = v230 & (0xFFFFFFFFFFFFFFFFLL >> -v530.m128i_i8[8]);
        }
        if ( (*(_BYTE *)(a2 + 80) & 8) != 0 )
          sub_14A9D90((__int64)&v530, 0, (unsigned int)v229);
        if ( (unsigned int)sub_1455840((__int64)&v501) < (unsigned int)v229 )
          sub_14A9D60(v530.m128i_i64, v530.m128i_i32[2] - 1);
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                **(_QWORD **)(a2 + 32),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                (unsigned int)&v530,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_585;
        v231 = *((_DWORD *)a5 + 2);
        if ( v231 > 0x40 )
        {
          sub_16A8110((__int64)a5, v229);
        }
        else if ( (_DWORD)v229 == v231 )
        {
          *a5 = 0;
        }
        else
        {
          *a5 >>= v229;
        }
        v232 = *((_DWORD *)a5 + 6);
        if ( v232 > 0x40 )
        {
          sub_16A8110((__int64)v470, v229);
        }
        else if ( (_DWORD)v229 == v232 )
        {
          a5[2] = 0;
        }
        else
        {
          a5[2] >>= v229;
        }
        v484 = v14 - 1 - v229;
        if ( sub_13D0200(a5, v484) || (unsigned int)sub_1455840((__int64)&v501) >= (unsigned int)v229 )
        {
          v298 = sub_1D332F0(
                   *(__int64 **)a6,
                   124,
                   (__int64)&v503,
                   v505.m128i_u32[0],
                   (const void **)v505.m128i_i64[1],
                   *(_BYTE *)(a2 + 80) & 8 | 1u,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   **(_QWORD **)(a2 + 32),
                   *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                   *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v298;
          *(_DWORD *)(a6 + 40) = v299;
        }
        else
        {
          v233 = v502;
          if ( v502 > 0x40 )
          {
            if ( (unsigned int)sub_16A5940((__int64)&v501) != 1 )
              goto LABEL_448;
            v235 = sub_16A57B0((__int64)&v501);
          }
          else
          {
            if ( !v501 || (v501 & (v501 - 1)) != 0 )
              goto LABEL_448;
            _BitScanReverse64(&v234, v501);
            v235 = v502 + (v234 ^ 0x3F) - 64;
          }
          if ( (int)(v233 - 1 - v235) < 0 )
          {
LABEL_448:
            if ( sub_13D0200((__int64 *)v470, v484) )
              sub_14A9D90((__int64)v470, (unsigned int)(*((_DWORD *)a5 + 6) - (_DWORD)v15), *((unsigned int *)a5 + 6));
            sub_135E100(v530.m128i_i64);
            goto LABEL_72;
          }
          v388 = (const void ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                                + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
          *(_QWORD *)&v389 = sub_1D38BB0(
                               *(_QWORD *)a6,
                               v14 - v233 + v235,
                               (__int64)&v503,
                               *(unsigned __int8 *)v388,
                               v388[1],
                               0,
                               a7,
                               *(double *)a8.m128i_i64,
                               a9,
                               0);
          v390 = sub_1D332F0(
                   *(__int64 **)a6,
                   124,
                   (__int64)&v503,
                   v505.m128i_u32[0],
                   (const void **)v505.m128i_i64[1],
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   **(_QWORD **)(a2 + 32),
                   *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                   v389);
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v390;
          *(_DWORD *)(a6 + 40) = v391;
        }
LABEL_585:
        LODWORD(v15) = 1;
        sub_135E100(v530.m128i_i64);
        goto LABEL_90;
      case 120:
        v79 = sub_1D1ADA0(
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                (unsigned __int16)v32,
                v30,
                v18,
                v19);
        if ( !v79 )
          goto LABEL_72;
        v82 = *(_QWORD *)(a2 + 32);
        v15 = *(__m128i **)v82;
        v465 = *(_QWORD *)(v82 + 8);
        v83 = *(_QWORD *)(v79 + 88);
        v477 = v14;
        v84 = *(unsigned int *)(v83 + 32);
        if ( (unsigned int)v84 > 0x40 )
        {
          v446 = *(_QWORD *)(v79 + 88);
          v456 = *(_DWORD *)(v83 + 32);
          v84 = v456 - (unsigned int)sub_16A57B0(v83 + 24);
          if ( (unsigned int)v84 > 0x40 )
            goto LABEL_72;
          v85 = **(_QWORD **)(v446 + 24);
          if ( v14 <= v85 )
            goto LABEL_72;
        }
        else
        {
          v85 = *(_QWORD *)(v83 + 24);
          if ( v14 <= v85 )
            goto LABEL_72;
        }
        v86 = v502;
        v442 = *(__m128i **)v82;
        v87 = v85;
        v518.m128i_i32[2] = v502;
        if ( v502 > 0x40 )
        {
          v454 = v85;
          sub_16A4FD0((__int64)&v518, (const void **)&v501);
          v86 = v518.m128i_i32[2];
          if ( v518.m128i_i32[2] > 0x40u )
          {
            sub_16A7DC0(v518.m128i_i64, v454);
            goto LABEL_144;
          }
        }
        else
        {
          v518.m128i_i64[0] = v501;
        }
        v88 = 0;
        if ( v87 != v86 )
        {
          v89 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v86;
          v86 = v87;
          v84 = v518.m128i_i64[0] << v87;
          v88 = (v518.m128i_i64[0] << v87) & v89;
        }
        v518.m128i_i64[0] = v88;
LABEL_144:
        if ( (*(_BYTE *)(a2 + 80) & 8) != 0 )
          sub_14A9D90((__int64)&v518, 0, v87);
        if ( v442[1].m128i_i16[4] != 122 )
          goto LABEL_147;
        v272 = sub_1D1ADA0(
                 *(_QWORD *)(v442[2].m128i_i64[0] + 40),
                 *(_QWORD *)(v442[2].m128i_i64[0] + 48),
                 v84,
                 v86,
                 v80,
                 v81);
        if ( !v272 )
          goto LABEL_147;
        if ( !v87 )
          goto LABEL_147;
        v427 = v272;
        sub_171A350((__int64)&v527, v14, v87);
        sub_20A0CE0(v527.m128i_i64, (__int64 *)&v501);
        v530.m128i_i32[2] = v527.m128i_i32[2];
        v527.m128i_i32[2] = 0;
        v530.m128i_i64[0] = v527.m128i_i64[0];
        v273 = sub_13A38F0((__int64)&v530, 0);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( !v273 )
          goto LABEL_147;
        v274 = *(_QWORD *)(v427 + 88);
        if ( sub_13D0480(v274 + 24, v477) )
        {
          v275 = *(_QWORD **)(v274 + 24);
          if ( *(_DWORD *)(v274 + 32) > 0x40u )
            v275 = (_QWORD *)*v275;
          v276 = 124;
          v277 = v87 - (_DWORD)v275;
          if ( (int)(v87 - (_DWORD)v275) < 0 )
          {
            v277 = (_DWORD)v275 - v87;
            v276 = 122;
          }
          *(_QWORD *)&v278 = sub_1D38BB0(
                               *(_QWORD *)a6,
                               v277,
                               (__int64)&v503,
                               *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                                                  + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL)),
                               *(const void ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                                               + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL)
                                               + 8),
                               0,
                               a7,
                               *(double *)a8.m128i_i64,
                               a9,
                               0);
          v279 = sub_1D332F0(
                   *(__int64 **)a6,
                   v276,
                   (__int64)&v503,
                   v505.m128i_u32[0],
                   (const void **)v505.m128i_i64[1],
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   *(_QWORD *)v442[2].m128i_i64[0],
                   *(_QWORD *)(v442[2].m128i_i64[0] + 8),
                   v278);
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v279;
          *(_DWORD *)(a6 + 40) = v280;
        }
        else
        {
LABEL_147:
          v90 = sub_20A2AF0(a1, (_DWORD)v15, v465, (unsigned int)&v518, (_DWORD)a5, a6, a10 + 1, 0);
          LODWORD(v15) = v422;
          if ( !v90 )
          {
            v91 = *((_DWORD *)a5 + 2);
            if ( v91 > 0x40 )
            {
              sub_16A8110((__int64)a5, v87);
            }
            else if ( v87 == v91 )
            {
              *a5 = 0;
            }
            else
            {
              *a5 >>= v87;
            }
            v92 = *((_DWORD *)a5 + 6);
            if ( v92 > 0x40 )
            {
              sub_16A8110((__int64)v470, v87);
            }
            else if ( v87 == v92 )
            {
              a5[2] = 0;
            }
            else
            {
              a5[2] >>= v87;
            }
            sub_14A9D90((__int64)a5, *((_DWORD *)a5 + 2) - v87, *((unsigned int *)a5 + 2));
            sub_135E100(v518.m128i_i64);
            goto LABEL_72;
          }
        }
        LODWORD(v15) = 1;
        sub_135E100(v518.m128i_i64);
        goto LABEL_90;
      case 130:
        v15 = (__m128i *)&v501;
        v93 = a10 + 1;
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL),
                                (unsigned int)&v501,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_118;
        v94 = *(_QWORD *)(a2 + 32);
        v95 = *(_QWORD *)(v94 + 40);
        v96 = *(_QWORD *)(v94 + 48);
        goto LABEL_157;
      case 132:
        v15 = (__m128i *)&v501;
        v93 = a10 + 1;
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 128LL),
                                (unsigned int)&v501,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_118;
        v157 = *(_QWORD *)(a2 + 32);
        v95 = *(_QWORD *)(v157 + 80);
        v96 = *(_QWORD *)(v157 + 88);
LABEL_157:
        if ( (unsigned __int8)sub_20A2AF0(a1, v95, v96, (unsigned int)&v501, (unsigned int)&v519, a6, v93, 0)
          || (unsigned __int8)sub_20A2230(a1, a2, a3, (__int64)&v501, a6, a7, *(double *)a8.m128i_i64, a9) )
        {
          goto LABEL_118;
        }
        sub_20A0CE0((__int64 *)v470, (__int64 *)&v521);
        sub_20A0CE0(a5, (__int64 *)&v519);
        goto LABEL_72;
      case 133:
        v200 = *(_QWORD *)(a2 + 32);
        v201 = *(_QWORD *)v200;
        v202 = *(_DWORD *)(v200 + 8);
        v482 = *(_QWORD *)(v200 + 40);
        v467 = *(__int64 **)(v200 + 48);
        LODWORD(v15) = v502 - 1;
        if ( v502 <= 0x40 )
        {
          if ( v501 != 1LL << (char)v15 )
            goto LABEL_358;
        }
        else
        {
          if ( (*(_QWORD *)(v501 + 8LL * ((unsigned int)v15 >> 6)) & (1LL << (char)v15)) == 0 )
            goto LABEL_358;
          v445 = *(_DWORD *)(v200 + 8);
          v203 = sub_16A58A0((__int64)&v501);
          v202 = v445;
          if ( v203 != (_DWORD)v15 )
            goto LABEL_358;
        }
        LODWORD(v15) = *(_DWORD *)(v200 + 48);
        v460 = v202;
        v447 = *(_QWORD *)(v200 + 40);
        v452 = *(_DWORD *)(*(_QWORD *)(v200 + 80) + 84LL);
        v256 = sub_20A1760(v201, v202);
        v202 = v460;
        if ( v256 != v14 )
          goto LABEL_358;
        v530 = _mm_loadu_si128(&v505);
        if ( v505.m128i_i8[0] )
        {
          if ( (unsigned __int8)(v505.m128i_i8[0] - 14) > 0x5Fu )
          {
            v257 = (unsigned __int8)(v505.m128i_i8[0] - 86) <= 0x17u || (unsigned __int8)(v505.m128i_i8[0] - 8) <= 5u;
            goto LABEL_509;
          }
        }
        else
        {
          v432 = v460;
          v463 = sub_1F58CD0((__int64)&v530);
          v357 = sub_1F58D20((__int64)&v530);
          v257 = v463;
          v202 = v432;
          if ( !v357 )
          {
LABEL_509:
            if ( v257 )
              v258 = *(_DWORD *)(a1 + 64);
            else
              v258 = *(_DWORD *)(a1 + 60);
LABEL_511:
            if ( v258 == 2 && v452 == 20 )
            {
              v461 = v202;
              v259 = *(_QWORD *)(v447 + 40) + 16LL * (unsigned int)v15;
              v260 = *(_BYTE *)v259;
              v261 = *(_QWORD *)(v259 + 8);
              v530.m128i_i8[0] = v260;
              v530.m128i_i64[1] = v261;
              LOBYTE(v262) = sub_20A18F0((__int64)&v530);
              v202 = v461;
              LODWORD(v15) = v262;
              if ( (_BYTE)v262 )
              {
                v263 = sub_1D185B0(v482);
                v202 = v461;
                if ( v263 || (v419 = sub_1D16620(v447, v467), v202 = v461, v419) )
                {
                  *(_QWORD *)(a6 + 16) = a2;
                  *(_DWORD *)(a6 + 24) = a3;
                  *(_QWORD *)(a6 + 32) = v201;
                  *(_DWORD *)(a6 + 40) = v202;
                  goto LABEL_90;
                }
              }
            }
LABEL_358:
            v204 = *(_QWORD *)(v201 + 40) + 16LL * v202;
            v205 = *(_BYTE *)v204;
            v206 = *(_QWORD *)(v204 + 8);
            v530.m128i_i8[0] = v205;
            v530.m128i_i64[1] = v206;
            if ( v205 )
            {
              if ( (unsigned __int8)(v205 - 14) > 0x5Fu )
              {
                v207 = (unsigned __int8)(v205 - 86) <= 0x17u || (unsigned __int8)(v205 - 8) <= 5u;
                goto LABEL_361;
              }
            }
            else
            {
              v15 = &v530;
              v207 = sub_1F58CD0((__int64)&v530);
              if ( !sub_1F58D20((__int64)&v530) )
              {
LABEL_361:
                if ( v207 )
                  v208 = *(_DWORD *)(a1 + 64);
                else
                  v208 = *(_DWORD *)(a1 + 60);
LABEL_363:
                if ( v14 > 1 && v208 == 1 )
                  sub_14A9D90((__int64)a5, 1, *((unsigned int *)a5 + 2));
                goto LABEL_72;
              }
            }
            v208 = *(_DWORD *)(a1 + 68);
            goto LABEL_363;
          }
        }
        v258 = *(_DWORD *)(a1 + 68);
        goto LABEL_511;
      case 138:
        v186 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
        v187 = *v186;
        v188 = *((_QWORD *)v186 + 1);
        v530.m128i_i8[0] = v187;
        v530.m128i_i64[1] = v188;
        v189 = sub_1D159C0((__int64)&v530, v16, v187, v30, v18, v19);
        v190 = v502;
        v191 = v189;
        if ( v502 > 0x40 )
        {
          v193 = v190 - sub_16A57B0((__int64)&v501);
        }
        else
        {
          if ( !v501 )
            goto LABEL_417;
          _BitScanReverse64(&v192, v501);
          v193 = 64 - (v192 ^ 0x3F);
        }
        if ( v191 >= v193 )
          goto LABEL_417;
        v15 = (__m128i *)&v516;
        sub_16A5A50((__int64)&v516, (__int64 *)&v501, v191);
        if ( v517 > 0x40 )
          *(_QWORD *)(v516 + 8LL * ((v191 - 1) >> 6)) |= 1LL << ((unsigned __int8)v191 - 1);
        else
          v516 |= 1LL << ((unsigned __int8)v191 - 1);
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                **(_QWORD **)(a2 + 32),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                (unsigned int)&v516,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_354;
        sub_16A5B10((__int64)&v527, v470, v14);
        sub_16A5B10((__int64)&v518, a5, v14);
        v530.m128i_i32[2] = v518.m128i_i32[2];
        v530.m128i_i64[0] = v518.m128i_i64[0];
        v194 = v527.m128i_i32[2];
        v527.m128i_i32[2] = 0;
        LODWORD(v532) = v194;
        v531 = v527.m128i_i64[0];
        sub_135E100(v527.m128i_i64);
        if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
          j_j___libc_free_0_0(*a5);
        v20 = *((_DWORD *)a5 + 6) <= 0x40u;
        *a5 = v530.m128i_i64[0];
        v195 = v530.m128i_i32[2];
        v530.m128i_i32[2] = 0;
        *((_DWORD *)a5 + 2) = v195;
        if ( !v20 )
        {
          v196 = a5[2];
          if ( v196 )
            j_j___libc_free_0_0(v196);
        }
        a5[2] = v531;
        v197 = v532;
        LODWORD(v532) = 0;
        *((_DWORD *)a5 + 6) = v197;
        sub_135E100(&v531);
        sub_135E100(v530.m128i_i64);
        if ( !sub_13D0200(a5, *((_DWORD *)a5 + 2) - 1) )
          goto LABEL_275;
        v198 = sub_1D309E0(
                 *(__int64 **)a6,
                 143,
                 (__int64)&v503,
                 v505.m128i_u32[0],
                 (const void **)v505.m128i_i64[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 *(_OWORD *)*(_QWORD *)(a2 + 32));
        *(_QWORD *)(a6 + 16) = a2;
        *(_DWORD *)(a6 + 24) = a3;
        *(_QWORD *)(a6 + 32) = v198;
        *(_DWORD *)(a6 + 40) = v199;
        goto LABEL_354;
      case 139:
        v209 = sub_20A1760(**(_QWORD **)(a2 + 32), *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
        v210 = v502;
        v483 = v209;
        if ( v502 > 0x40 )
        {
          v212 = v210 - sub_16A57B0((__int64)&v501);
        }
        else
        {
          if ( !v501 )
            goto LABEL_417;
          _BitScanReverse64(&v211, v501);
          v212 = 64 - (v211 ^ 0x3F);
        }
        if ( v483 < v212 )
        {
          v15 = (__m128i *)&v516;
          sub_16A5A50((__int64)&v516, (__int64 *)&v501, v483);
          if ( (unsigned __int8)sub_20A2AF0(
                                  a1,
                                  **(_QWORD **)(a2 + 32),
                                  *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                  (unsigned int)&v516,
                                  (_DWORD)a5,
                                  a6,
                                  a10 + 1,
                                  0) )
          {
LABEL_354:
            sub_135E100((__int64 *)&v516);
            goto LABEL_118;
          }
          sub_16A5C50((__int64)&v527, v470, v14);
          sub_16A5C50((__int64)&v518, (const void **)a5, v14);
          v213 = v518.m128i_i32[2];
          v214 = v518.m128i_i64[0];
          v20 = *((_DWORD *)a5 + 2) <= 0x40u;
          LODWORD(v532) = v527.m128i_i32[2];
          v530.m128i_i32[2] = v518.m128i_i32[2];
          v530.m128i_i64[0] = v518.m128i_i64[0];
          v531 = v527.m128i_i64[0];
          if ( !v20 && *a5 )
          {
            j_j___libc_free_0_0(*a5);
            v214 = v530.m128i_i64[0];
            v213 = v530.m128i_i32[2];
          }
          v20 = *((_DWORD *)a5 + 6) <= 0x40u;
          *a5 = v214;
          *((_DWORD *)a5 + 2) = v213;
          v530.m128i_i32[2] = 0;
          if ( !v20 )
          {
            v215 = a5[2];
            if ( v215 )
              j_j___libc_free_0_0(v215);
          }
          a5[2] = v531;
          v216 = v532;
          LODWORD(v532) = 0;
          *((_DWORD *)a5 + 6) = v216;
          sub_135E100(&v531);
          sub_135E100(v530.m128i_i64);
          sub_14A9D90((__int64)a5, v483, *((unsigned int *)a5 + 2));
          sub_135E100((__int64 *)&v516);
LABEL_72:
          v48 = *((_DWORD *)a5 + 2);
          v527.m128i_i32[2] = v48;
          if ( v48 > 0x40 )
          {
            sub_16A4FD0((__int64)&v527, (const void **)a5);
            v48 = v527.m128i_u32[2];
            if ( v527.m128i_i32[2] > 0x40u )
            {
              sub_16A89F0(v527.m128i_i64, (__int64 *)v470);
              v48 = v527.m128i_u32[2];
              v50 = v527.m128i_i64[0];
LABEL_75:
              v530.m128i_i32[2] = v48;
              v530.m128i_i64[0] = v50;
              v527.m128i_i32[2] = 0;
              if ( v502 <= 0x40 )
                LOBYTE(v15) = (v501 & ~v50) == 0;
              else
                LODWORD(v15) = sub_16A5A00((__int64 *)&v501, v530.m128i_i64);
              if ( v48 > 0x40 )
              {
                if ( v50 )
                {
                  j_j___libc_free_0_0(v50);
                  if ( v527.m128i_i32[2] > 0x40u )
                  {
                    if ( v527.m128i_i64[0] )
                      j_j___libc_free_0_0(v527.m128i_i64[0]);
                  }
                }
              }
              if ( (_BYTE)v15 )
              {
                v51 = *(_DWORD *)(a2 + 56);
                if ( v51 )
                {
                  v52 = *(_QWORD *)(a2 + 32);
                  v53 = v52 + 40LL * (unsigned int)(v51 - 1) + 40;
                  while ( 1 )
                  {
                    v54 = *(unsigned __int16 *)(*(_QWORD *)v52 + 24LL);
                    if ( (v54 == 32 || v54 == 10) && (*(_BYTE *)(*(_QWORD *)v52 + 26LL) & 8) != 0 )
                      break;
                    v52 += 40;
                    if ( v52 == v53 )
                      goto LABEL_381;
                  }
LABEL_89:
                  LODWORD(v15) = 0;
                }
                else
                {
LABEL_381:
                  v217 = sub_1D38970(
                           *(_QWORD *)a6,
                           (__int64)v470,
                           (__int64)&v503,
                           v505.m128i_u32[0],
                           (const void **)v505.m128i_i64[1],
                           0,
                           a7,
                           *(double *)a8.m128i_i64,
                           a9,
                           0);
                  *(_QWORD *)(a6 + 16) = a2;
                  *(_DWORD *)(a6 + 24) = v469;
                  *(_QWORD *)(a6 + 32) = v217;
                  *(_DWORD *)(a6 + 40) = v218;
                }
              }
              goto LABEL_90;
            }
            v49 = v527.m128i_i64[0];
          }
          else
          {
            v49 = *a5;
          }
          v527.m128i_i64[0] = a5[2] | v49;
          v50 = v527.m128i_i64[0];
          goto LABEL_75;
        }
LABEL_417:
        v224 = sub_1D309E0(
                 *(__int64 **)a6,
                 144,
                 (__int64)&v503,
                 v505.m128i_u32[0],
                 (const void **)v505.m128i_i64[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 *(_OWORD *)*(_QWORD *)(a2 + 32));
        *(_QWORD *)(a6 + 16) = a2;
        *(_DWORD *)(a6 + 24) = a3;
        *(_QWORD *)(a6 + 32) = v224;
        *(_DWORD *)(a6 + 40) = v225;
        goto LABEL_118;
      case 140:
        v158 = sub_20A1760(**(_QWORD **)(a2 + 32), *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
        sub_16A5A50((__int64)&v518, (__int64 *)&v501, v158);
        LODWORD(v15) = sub_20A2AF0(
                         a1,
                         **(_QWORD **)(a2 + 32),
                         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                         (unsigned int)&v518,
                         (_DWORD)a5,
                         a6,
                         a10 + 1,
                         0);
        if ( (_BYTE)v15 )
        {
          sub_135E100(v518.m128i_i64);
          goto LABEL_90;
        }
        sub_16A5C50((__int64)&v530, v470, v14);
        sub_16A5C50((__int64)&v527, (const void **)a5, v14);
        LODWORD(v15) = v527.m128i_i32[2];
        v159 = v527.m128i_i64[0];
        v160 = v530.m128i_i32[2];
        v161 = v530.m128i_i64[0];
        if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
        {
          v480 = v527.m128i_i64[0];
          j_j___libc_free_0_0(*a5);
          v159 = v480;
        }
        v20 = *((_DWORD *)a5 + 6) <= 0x40u;
        *a5 = v159;
        *((_DWORD *)a5 + 2) = (_DWORD)v15;
        if ( !v20 )
        {
          v162 = a5[2];
          if ( v162 )
            j_j___libc_free_0_0(v162);
        }
        v20 = v518.m128i_i32[2] <= 0x40u;
        a5[2] = v161;
        *((_DWORD *)a5 + 6) = v160;
        if ( !v20 )
        {
          v125 = v518.m128i_i64[0];
          if ( v518.m128i_i64[0] )
LABEL_213:
            j_j___libc_free_0_0(v125);
        }
        goto LABEL_72;
      case 141:
        v15 = (__m128i *)&v516;
        v152 = sub_20A1760(**(_QWORD **)(a2 + 32), *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
        sub_16A5C50((__int64)&v516, (const void **)&v501, v152);
        if ( (unsigned __int8)sub_20A2AF0(
                                a1,
                                **(_QWORD **)(a2 + 32),
                                *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                (unsigned int)&v516,
                                (_DWORD)a5,
                                a6,
                                a10 + 1,
                                0) )
          goto LABEL_115;
        sub_16A5A50((__int64)&v527, (__int64 *)v470, v14);
        sub_16A5A50((__int64)&v518, a5, v14);
        v153 = v518.m128i_i32[2];
        v154 = v518.m128i_i64[0];
        v20 = *((_DWORD *)a5 + 2) <= 0x40u;
        LODWORD(v532) = v527.m128i_i32[2];
        v530.m128i_i32[2] = v518.m128i_i32[2];
        v530.m128i_i64[0] = v518.m128i_i64[0];
        v531 = v527.m128i_i64[0];
        if ( !v20 && *a5 )
        {
          j_j___libc_free_0_0(*a5);
          v154 = v530.m128i_i64[0];
          v153 = v530.m128i_i32[2];
        }
        v20 = *((_DWORD *)a5 + 6) <= 0x40u;
        *a5 = v154;
        *((_DWORD *)a5 + 2) = v153;
        v530.m128i_i32[2] = 0;
        if ( !v20 )
        {
          v155 = a5[2];
          if ( v155 )
            j_j___libc_free_0_0(v155);
        }
        a5[2] = v531;
        *((_DWORD *)a5 + 6) = v532;
        sub_135E100(v530.m128i_i64);
        v436 = **(_QWORD **)(a2 + 32);
        v156 = *(_QWORD *)(v436 + 48);
        if ( !v156
          || *(_QWORD *)(v156 + 32)
          || *(_WORD *)(v436 + 24) != 124
          || *(_BYTE *)(a6 + 8)
          && !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 1136LL))(
                a1,
                124,
                v505.m128i_u32[0],
                v505.m128i_i64[1]) )
        {
          goto LABEL_275;
        }
        v281 = *(_QWORD *)(v436 + 32);
        v282 = *(unsigned __int16 *)(*(_QWORD *)(v281 + 40) + 24LL);
        v453 = *(_QWORD *)(v281 + 40);
        if ( v282 != 32 && v282 != 10 )
          goto LABEL_275;
        v448 = (__int128)_mm_loadu_si128((const __m128i *)(v281 + 40));
        if ( *(_BYTE *)(a6 + 8) )
        {
          v283 = *(_QWORD *)(v453 + 88);
          v284 = *(_QWORD **)(v283 + 24);
          if ( *(_DWORD *)(v283 + 32) > 0x40u )
            v284 = (_QWORD *)*v284;
          v428 = (__int64)v284;
          v435 = *(_QWORD *)a6;
          v285 = sub_1F40B60(a1, v505.m128i_u32[0], v505.m128i_i64[1], v464, 1);
          *(_QWORD *)&v448 = sub_1D38BB0(v435, v428, (__int64)&v503, v285, v286, 0, a7, *(double *)a8.m128i_i64, a9, 0);
          *((_QWORD *)&v448 + 1) = v287 | *((_QWORD *)&v448 + 1) & 0xFFFFFFFF00000000LL;
        }
        v288 = *(_QWORD *)(v453 + 88);
        v289 = *(_QWORD **)(v288 + 24);
        if ( *(_DWORD *)(v288 + 32) > 0x40u )
          v289 = (_QWORD *)*v289;
        if ( v14 <= (unsigned __int64)v289 )
          goto LABEL_275;
        sub_171A350((__int64)&v518, v152, v152 - v14);
        v290 = *(_QWORD *)(v453 + 88);
        if ( *(_DWORD *)(v290 + 32) <= 0x40u )
          v291 = *(_QWORD *)(v290 + 24);
        else
          v291 = **(_QWORD **)(v290 + 24);
        sub_17A2760((__int64)&v518, v291);
        sub_16A5A50((__int64)&v530, v518.m128i_i64, v14);
        sub_14A9CA0(v518.m128i_i64, v530.m128i_i64);
        sub_135E100(v530.m128i_i64);
        sub_13A38D0((__int64)&v527, (__int64)&v518);
        sub_20A0CE0(v527.m128i_i64, (__int64 *)&v501);
        v292 = v527.m128i_u32[2];
        v527.m128i_i32[2] = 0;
        v530.m128i_i32[2] = v292;
        v530.m128i_i64[0] = v527.m128i_i64[0];
        if ( v292 <= 0x40 )
          v293 = v527.m128i_i64[0] == 0;
        else
          v293 = v292 == (unsigned int)sub_16A57B0((__int64)&v530);
        sub_135E100(v530.m128i_i64);
        sub_135E100(v527.m128i_i64);
        if ( !v293 )
        {
          sub_135E100(v518.m128i_i64);
LABEL_275:
          sub_135E100((__int64 *)&v516);
          goto LABEL_72;
        }
        v294 = sub_1D309E0(
                 *(__int64 **)a6,
                 145,
                 (__int64)&v503,
                 v505.m128i_u32[0],
                 (const void **)v505.m128i_i64[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 *(_OWORD *)*(_QWORD *)(v436 + 32));
        v296 = sub_1D332F0(
                 *(__int64 **)a6,
                 124,
                 (__int64)&v503,
                 v505.m128i_u32[0],
                 (const void **)v505.m128i_i64[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 a9,
                 v294,
                 v295,
                 v448);
        *(_QWORD *)(a6 + 16) = a2;
        *(_DWORD *)(a6 + 24) = a3;
        *(_QWORD *)(a6 + 32) = v296;
        *(_DWORD *)(a6 + 40) = v297;
        sub_135E100(v518.m128i_i64);
LABEL_115:
        if ( v517 <= 0x40 )
          goto LABEL_118;
        v64 = v516;
        if ( !v516 )
          goto LABEL_118;
        goto LABEL_117;
      case 144:
        v65 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
        v66 = *(_QWORD *)(v65 + 96);
        v518.m128i_i8[0] = *(_BYTE *)(v65 + 88);
        v518.m128i_i64[1] = v66;
        v67 = sub_1D159C0((__int64)&v518, v16, v66, v30, v18, v19);
        v68 = v502;
        v441 = v67;
        if ( v502 <= 0x40 )
        {
          if ( v501 != 1LL << ((unsigned __int8)v502 - 1) )
            goto LABEL_500;
        }
        else
        {
          if ( (*(_QWORD *)(v501 + 8LL * ((v502 - 1) >> 6)) & (1LL << ((unsigned __int8)v502 - 1))) == 0 )
            goto LABEL_122;
          v69 = v502 - 1;
          if ( v69 != (unsigned int)sub_16A58A0((__int64)&v501) )
            goto LABEL_122;
        }
        v485 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
        if ( (unsigned int)sub_1D23330(*(_QWORD *)a6, v485.m128i_i64[0], v485.m128i_i64[1], 0) < v14 - v441 + 1 )
        {
          v251 = *(_BYTE *)(a6 + 8) == 0;
          v530 = _mm_loadu_si128(&v505);
          if ( !v251 )
          {
            if ( v530.m128i_i8[0] )
            {
              if ( (unsigned __int8)(v530.m128i_i8[0] - 14) <= 0x5Fu )
                goto LABEL_499;
            }
            else if ( sub_1F58D20((__int64)&v530) )
            {
              goto LABEL_499;
            }
            v530.m128i_i32[0] = sub_1F40B60(a1, v530.m128i_u32[0], v530.m128i_i64[1], v464, 1);
            v530.m128i_i64[1] = v358;
          }
LABEL_499:
          *(_QWORD *)&v252 = sub_1D38BB0(
                               *(_QWORD *)a6,
                               v14 - v441,
                               (__int64)&v503,
                               v530.m128i_u32[0],
                               (const void **)v530.m128i_i64[1],
                               0,
                               a7,
                               *(double *)a8.m128i_i64,
                               a9,
                               0);
          v253 = sub_1D332F0(
                   *(__int64 **)a6,
                   122,
                   (__int64)&v503,
                   v505.m128i_u32[0],
                   (const void **)v505.m128i_i64[1],
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   v485.m128i_i64[0],
                   v485.m128i_u64[1],
                   v252);
          *(_QWORD *)(a6 + 16) = a2;
          *(_DWORD *)(a6 + 24) = a3;
          *(_QWORD *)(a6 + 32) = v253;
          *(_DWORD *)(a6 + 40) = v254;
          goto LABEL_118;
        }
        v68 = v502;
        if ( v502 > 0x40 )
        {
LABEL_122:
          v70 = v68 - sub_16A57B0((__int64)&v501);
          goto LABEL_123;
        }
LABEL_500:
        if ( !v501 )
          goto LABEL_124;
        _BitScanReverse64(&v255, v501);
        v70 = 64 - (v255 ^ 0x3F);
LABEL_123:
        if ( v441 < v70 )
        {
          LODWORD(v15) = v441 - 1;
          sub_16A88B0((__int64)&v527, (__int64)&v501, v441);
          sub_14A9D60(v527.m128i_i64, v441 - 1);
          if ( !(unsigned __int8)sub_20A2AF0(
                                   a1,
                                   **(_QWORD **)(a2 + 32),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                   (unsigned int)&v527,
                                   (_DWORD)a5,
                                   a6,
                                   a10 + 1,
                                   0) )
          {
            if ( !sub_13D0200(a5, (unsigned int)v15) )
            {
              v530.m128i_i32[2] = v14;
              if ( v14 > 0x40 )
                sub_16A4EF0((__int64)&v530, 0, 0);
              else
                v530.m128i_i64[0] = 0;
              if ( v441 )
              {
                if ( v441 > 0x40 )
                {
                  sub_16A5260(&v530, 0, v441);
                }
                else
                {
                  v330 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v441);
                  if ( v530.m128i_i32[2] > 0x40u )
                    *(_QWORD *)v530.m128i_i64[0] |= v330;
                  else
                    v530.m128i_i64[0] |= v330;
                }
              }
              if ( sub_13D0200((__int64 *)v470, (unsigned int)v15) )
              {
                sub_14A9D90((__int64)v470, v441, *((unsigned int *)a5 + 6));
                sub_20A0CE0(a5, v530.m128i_i64);
              }
              else
              {
                sub_20A0CE0(a5, v530.m128i_i64);
                sub_20A0CE0((__int64 *)v470, v530.m128i_i64);
              }
              sub_135E100(v530.m128i_i64);
              sub_135E100(v527.m128i_i64);
              goto LABEL_72;
            }
            v265 = *(__int64 **)a6;
            LOBYTE(v266) = sub_1D15870(v518.m128i_i8);
            v268 = sub_1D3BC50(
                     v265,
                     **(_QWORD **)(a2 + 32),
                     *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                     (__int64)&v503,
                     v266,
                     v267,
                     a7,
                     *(double *)a8.m128i_i64,
                     a9);
            *(_QWORD *)(a6 + 16) = a2;
            *(_DWORD *)(a6 + 24) = a3;
            *(_QWORD *)(a6 + 32) = v268;
            *(_DWORD *)(a6 + 40) = v269;
          }
          sub_135E100(v527.m128i_i64);
          goto LABEL_118;
        }
LABEL_124:
        v494 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
        *(_QWORD *)(a6 + 16) = a2;
        *(_DWORD *)(a6 + 24) = a3;
        *(_QWORD *)(a6 + 32) = v494.m128i_i64[0];
        *(_DWORD *)(a6 + 40) = v494.m128i_i32[2];
        goto LABEL_118;
      case 154:
        v71 = *(_BYTE *)(a6 + 9);
        if ( v71 )
          goto LABEL_130;
        if ( v505.m128i_i8[0] )
        {
          if ( (unsigned __int8)(v505.m128i_i8[0] - 14) <= 0x5Fu )
            goto LABEL_130;
        }
        else if ( sub_1F58D20((__int64)&v505) )
        {
          goto LABEL_130;
        }
        v72 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
        v73 = *(_BYTE *)v72;
        v74 = *(_QWORD *)(v72 + 8);
        v518.m128i_i8[0] = v73;
        v518.m128i_i64[1] = v74;
        if ( v73 )
        {
          if ( (unsigned __int8)(v73 - 14) <= 0x5Fu )
            goto LABEL_130;
        }
        else if ( sub_1F58D20((__int64)&v518) )
        {
          goto LABEL_130;
        }
        v300 = sub_20A1720(a2, v476);
        v527.m128i_i32[2] = v300;
        v302 = v300;
        if ( v300 <= 0x40 )
        {
          v527.m128i_i64[0] = 0;
          v303 = 1LL << ((unsigned __int8)v300 - 1);
LABEL_589:
          v527.m128i_i64[0] |= v303;
          goto LABEL_590;
        }
        sub_16A4EF0((__int64)&v527, 0, 0);
        v303 = 1LL << ((unsigned __int8)v302 - 1);
        if ( v527.m128i_i32[2] <= 0x40u )
          goto LABEL_589;
        *(_QWORD *)(v527.m128i_i64[0] + 8LL * ((v302 - 1) >> 6)) |= v303;
LABEL_590:
        if ( v502 <= 0x40 )
        {
          if ( v501 != v527.m128i_i64[0] )
            goto LABEL_672;
        }
        else if ( !sub_16A5220((__int64)&v501, (const void **)&v527) )
        {
          goto LABEL_672;
        }
        v304 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
        v305 = *v304;
        v306 = *((_QWORD *)v304 + 1);
        v530.m128i_i8[0] = v305;
        v530.m128i_i64[1] = v306;
        if ( (_BYTE)v305 )
        {
          LOBYTE(v301) = (unsigned __int8)(v305 - 8) <= 5u;
          v307 = v305 - 86;
          LOBYTE(v307) = (unsigned __int8)v307 <= 0x17u;
          LODWORD(v15) = v307 | v301;
        }
        else
        {
          LOBYTE(v365) = sub_1F58CD0((__int64)&v530);
          LODWORD(v15) = v365;
        }
        if ( (_BYTE)v15 )
        {
          if ( v527.m128i_i32[2] > 0x40u && v527.m128i_i64[0] )
            j_j___libc_free_0_0(v527.m128i_i64[0]);
          v308 = 1;
          if ( v505.m128i_i8[0] == 1
            || v505.m128i_i8[0] && (v308 = v505.m128i_u8[0], *(_QWORD *)(a1 + 8LL * v505.m128i_u8[0] + 120)) )
          {
            if ( (*(_BYTE *)(a1 + 259 * v308 + 2524) & 0xFB) == 0 )
            {
              if ( !*(_QWORD *)(a1 + 160) || (v71 = (char)v15, (*(_BYTE *)(a1 + 3819) & 0xFB) != 0) )
                v71 = (char)v15;
LABEL_601:
              if ( v505.m128i_i8[0] )
              {
                v309 = *(_QWORD *)(a2 + 32);
                if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v309 + 40LL) + 16LL * *(unsigned int *)(v309 + 8)) & 0xFB) != 8 )
                {
                  if ( v71 )
                    v310 = v505;
                  else
                    v310 = (__m128i)5uLL;
                  *(_QWORD *)&v311 = sub_1D309E0(
                                       *(__int64 **)a6,
                                       102,
                                       (__int64)&v503,
                                       v310.m128i_i64[0],
                                       (const void **)v310.m128i_i64[1],
                                       0,
                                       *(double *)a7.m128i_i64,
                                       *(double *)a8.m128i_i64,
                                       *(double *)a9.m128i_i64,
                                       *(_OWORD *)v309);
                  v472 = v311;
                  v312 = sub_20A1720(a2, v476);
                  if ( v71 != 1 && v312 > 0x20 )
                  {
                    *(_QWORD *)&v472 = sub_1D309E0(
                                         *(__int64 **)a6,
                                         143,
                                         (__int64)&v503,
                                         v505.m128i_u32[0],
                                         (const void **)v505.m128i_i64[1],
                                         0,
                                         *(double *)a7.m128i_i64,
                                         *(double *)a8.m128i_i64,
                                         *(double *)a9.m128i_i64,
                                         v472);
                    *((_QWORD *)&v472 + 1) = v313 | *((_QWORD *)&v472 + 1) & 0xFFFFFFFF00000000LL;
                  }
                  v314 = sub_20A1720(a2, v476);
                  *(_QWORD *)&v315 = sub_1D38BB0(
                                       *(_QWORD *)a6,
                                       (unsigned int)(v314 - 1),
                                       (__int64)&v503,
                                       v505.m128i_u32[0],
                                       (const void **)v505.m128i_i64[1],
                                       0,
                                       a7,
                                       *(double *)a8.m128i_i64,
                                       a9,
                                       0);
                  v316 = sub_1D332F0(
                           *(__int64 **)a6,
                           122,
                           (__int64)&v503,
                           v505.m128i_u32[0],
                           (const void **)v505.m128i_i64[1],
                           0,
                           *(double *)a7.m128i_i64,
                           *(double *)a8.m128i_i64,
                           a9,
                           v472,
                           *((unsigned __int64 *)&v472 + 1),
                           v315);
                  *(_QWORD *)(a6 + 16) = a2;
                  *(_DWORD *)(a6 + 24) = a3;
                  *(_QWORD *)(a6 + 32) = v316;
                  *(_DWORD *)(a6 + 40) = v317;
                  goto LABEL_90;
                }
              }
              goto LABEL_130;
            }
          }
          if ( *(_QWORD *)(a1 + 160) && (*(_BYTE *)(a1 + 3819) & 0xFB) == 0 )
            goto LABEL_601;
LABEL_130:
          if ( !a10 )
            goto LABEL_72;
          LODWORD(v15) = 0;
          sub_1D1F820(*(_QWORD *)a6, a2, v476 | a3 & 0xFFFFFFFF00000000LL, a5, a10);
          goto LABEL_90;
        }
LABEL_672:
        if ( v527.m128i_i32[2] > 0x40u && v527.m128i_i64[0] )
          j_j___libc_free_0_0(v527.m128i_i64[0]);
        goto LABEL_130;
      default:
        goto LABEL_71;
    }
  }
  v33 = *((_DWORD *)a4 + 2);
  if ( v33 > 0x40 )
  {
    v455 = v23;
    v39 = sub_16A57B0((__int64)a4);
    v23 = v455;
    if ( v33 - v39 > 0x40 )
    {
LABEL_49:
      LODWORD(v15) = 0;
      if ( a10 == 6 )
        goto LABEL_27;
      goto LABEL_44;
    }
    v34 = (void *)**a4;
  }
  else
  {
    v34 = *a4;
  }
  if ( v34 )
    goto LABEL_49;
  if ( v23 == 48 )
  {
LABEL_26:
    LODWORD(v15) = 0;
    goto LABEL_27;
  }
  v530.m128i_i64[0] = 0;
  v35 = *(_QWORD **)a6;
  v530.m128i_i32[2] = 0;
  v36 = sub_1D2B300(v35, 0x30u, (__int64)&v530, v505.m128i_u32[0], v505.m128i_i64[1], v19);
  v38 = v37;
  if ( v530.m128i_i64[0] )
    sub_161E7C0((__int64)&v530, v530.m128i_i64[0]);
  LODWORD(v15) = 1;
  *(_QWORD *)(a6 + 16) = a2;
  *(_DWORD *)(a6 + 24) = a3;
  *(_QWORD *)(a6 + 32) = v36;
  *(_DWORD *)(a6 + 40) = v38;
LABEL_27:
  if ( v503 )
    sub_161E7C0((__int64)&v503, v503);
  if ( v502 > 0x40 && v501 )
    j_j___libc_free_0_0(v501);
  return (unsigned int)v15;
}
