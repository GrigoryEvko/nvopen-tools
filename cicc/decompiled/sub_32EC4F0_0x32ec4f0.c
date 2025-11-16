// Function: sub_32EC4F0
// Address: 0x32ec4f0
//
__int64 __fastcall sub_32EC4F0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  __int64 *v3; // rax
  __int64 v4; // r15
  __m128i v5; // xmm1
  __int64 v6; // rcx
  int v7; // ebx
  __int16 *v8; // rax
  __int16 v9; // dx
  __int32 v10; // r13d
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // r14
  __int128 v15; // rdi
  int v16; // eax
  char v17; // r10
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rcx
  unsigned int v21; // ebx
  __int64 v22; // r11
  int v23; // eax
  __int64 v24; // r13
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // edx
  __int64 v31; // rax
  int v32; // r9d
  __int64 (__fastcall *v33)(__int64, unsigned int); // rax
  char (__fastcall *v34)(__int64, unsigned int); // rax
  __int64 v35; // rdi
  __int64 *v36; // rax
  __int64 v37; // r14
  __int64 v38; // r15
  __int64 v39; // rax
  __int16 v40; // dx
  __int64 v41; // rax
  int v42; // eax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rcx
  __int64 v50; // rax
  __int16 v51; // r9
  __int64 v52; // r8
  __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rbx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rax
  _QWORD *v59; // rcx
  unsigned int v60; // eax
  __int64 v61; // rdx
  __int64 v62; // rcx
  int v63; // eax
  int v64; // eax
  unsigned int v65; // edx
  __int64 v66; // rdi
  __int64 (*v67)(); // rax
  int v68; // eax
  __int64 v69; // rbx
  __int64 v70; // rax
  signed __int64 v71; // rdx
  unsigned int v72; // ebx
  __int64 v73; // rdx
  __int64 v74; // r13
  __int64 v75; // rsi
  __int64 v76; // rdx
  __int64 v77; // r8
  __int64 v78; // rdi
  __int64 (*v79)(); // rax
  bool v80; // r13
  char v81; // al
  __int64 v82; // rbx
  __int64 v83; // r15
  __int64 v84; // rdx
  __int64 v85; // rbx
  __int64 (*v86)(); // rax
  __int64 v87; // r8
  unsigned __int16 *v88; // rax
  int v89; // r9d
  char v90; // r10
  __int64 v91; // r11
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rcx
  __int128 v95; // xmm3
  unsigned int v96; // edi
  __m128i v97; // xmm4
  __int64 v98; // rax
  char v99; // al
  __int64 v100; // r8
  unsigned int v101; // edx
  int v102; // r9d
  unsigned int v103; // edx
  __int64 v104; // rbx
  unsigned __int64 v105; // rdi
  __int64 v106; // rdi
  char v107; // al
  int v108; // r9d
  __int64 v109; // rdi
  __int64 v110; // rbx
  __int64 (*v111)(); // rax
  char v112; // al
  char v113; // al
  unsigned __int64 v114; // rcx
  __int64 v115; // r12
  unsigned int v116; // r14d
  __int64 v117; // rsi
  char v118; // al
  __int64 v119; // rdx
  _QWORD *v120; // rax
  char v121; // al
  __int64 v122; // r8
  __int64 v123; // r9
  __int64 v124; // rdx
  _QWORD *v125; // rax
  __int64 v126; // rbx
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // r9
  char v130; // al
  bool v131; // al
  __int64 v132; // r12
  __int128 v133; // rax
  __int64 v134; // rax
  bool v135; // cc
  _QWORD *v136; // rax
  int v137; // edi
  __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rax
  __int64 v141; // rcx
  __int64 v142; // rbx
  unsigned __int16 *v143; // rdx
  int v144; // eax
  __int16 *v145; // rax
  __int16 v146; // r13
  __int64 v147; // rax
  signed __int64 v148; // rax
  __int64 v149; // rdx
  _QWORD *v150; // rdx
  unsigned int v151; // eax
  int v152; // r14d
  unsigned __int32 v153; // r12d
  __int64 v154; // rdx
  __int64 v155; // rdx
  __int64 v156; // r9
  __int64 v157; // r14
  unsigned __int16 *v158; // rax
  unsigned __int16 v159; // dx
  __int64 v160; // rax
  __int64 v161; // rdx
  unsigned __int64 v162; // rdx
  char v163; // al
  int v164; // edx
  unsigned __int64 v165; // rax
  __int64 v166; // r15
  int v167; // eax
  unsigned __int16 *v168; // rdx
  int v169; // eax
  __int64 v170; // rdx
  __int64 v171; // rax
  _QWORD *v172; // rax
  __int64 *v173; // rdi
  unsigned __int64 *v174; // rax
  __int64 v175; // r8
  __int64 v176; // r9
  unsigned __int16 *v177; // rax
  __int64 v178; // rdx
  __int64 v179; // rax
  int v180; // r8d
  int v181; // eax
  int v182; // ecx
  unsigned int v183; // edx
  unsigned int v184; // edx
  __int128 v185; // rax
  int v186; // r9d
  __int64 v187; // rax
  __int64 v188; // rdx
  __int64 v189; // rdx
  __int16 v190; // ax
  __int64 v191; // rdx
  unsigned __int16 v192; // dx
  int v193; // ecx
  char v194; // al
  char v195; // al
  unsigned int *v196; // rax
  __int64 v197; // rbx
  __int64 v198; // rdx
  __int64 v199; // rax
  _QWORD *v200; // rdx
  unsigned __int32 v201; // r11d
  int v202; // r10d
  unsigned __int64 v203; // rax
  unsigned int v204; // eax
  __int64 v205; // rdx
  char v206; // bl
  __int64 v207; // rax
  bool v208; // al
  __int64 v209; // rdx
  __int64 v210; // rcx
  __int64 v211; // r8
  __int64 v212; // rax
  __int16 v213; // dx
  __int64 v214; // rdx
  __int64 v215; // rcx
  __int64 v216; // r8
  __int64 v217; // rdx
  bool v218; // zf
  __int64 v219; // rax
  __int64 v220; // rdx
  __int64 v221; // rcx
  __int64 v222; // r8
  char v223; // al
  int v224; // edx
  __int64 v225; // rax
  unsigned __int64 v226; // rbx
  __int64 v227; // r14
  int v228; // r15d
  int v229; // eax
  __int64 v230; // rax
  int v231; // edx
  unsigned __int16 v232; // ax
  int v233; // eax
  __int64 v234; // rax
  unsigned __int64 v235; // rbx
  __int64 v236; // rax
  __int64 v237; // rdx
  __int64 v238; // rax
  unsigned int v239; // eax
  __int64 v240; // rdx
  __int64 v241; // rax
  __int64 v242; // rdx
  unsigned __int64 v243; // rax
  unsigned int v244; // eax
  __int64 v245; // r13
  unsigned int v246; // ebx
  __int64 v247; // rax
  unsigned __int64 *v248; // r13
  unsigned __int64 v249; // rax
  __int128 v250; // rax
  int v251; // r9d
  __int64 v252; // rax
  __int64 v253; // rdx
  char v254; // al
  char v255; // al
  __int64 v256; // rax
  char v257; // cl
  __int64 v258; // rax
  __int64 v259; // rax
  char v260; // cl
  __int64 v261; // rax
  __int64 v262; // rbx
  __int64 v263; // rax
  __int64 v264; // rcx
  __int64 v265; // rax
  const __m128i *v266; // rdi
  __m128i *v267; // rax
  int v268; // r15d
  __int64 v269; // rax
  __int64 v270; // rcx
  bool v271; // al
  unsigned int *v272; // rax
  __int64 v273; // rax
  __int64 v274; // rax
  __int64 v275; // rdx
  __int32 v276; // eax
  unsigned int v277; // edx
  __int128 v278; // rax
  int v279; // r9d
  char v280; // bl
  bool v281; // al
  __int64 v282; // rdx
  unsigned __int64 *v283; // rax
  unsigned __int64 *v284; // rbx
  __int64 v285; // rax
  unsigned __int64 *v286; // rdx
  const __m128i *v287; // rax
  int v288; // r12d
  __int64 v289; // rax
  __int64 v290; // rax
  const __m128i *v291; // rdi
  const __m128i *v292; // rax
  int v293; // r15d
  unsigned int v294; // esi
  __int64 v295; // rax
  unsigned int *v296; // rax
  __int64 v297; // rsi
  unsigned __int16 *v298; // rax
  unsigned __int64 *v299; // rax
  __int64 v300; // rdx
  __int64 v301; // rax
  unsigned int v302; // eax
  unsigned int v303; // edx
  __int64 v304; // r13
  __int128 v305; // rax
  unsigned int v306; // edx
  __int64 v307; // rax
  __int16 v308; // ax
  __int64 v309; // rdx
  __int64 v310; // rcx
  __int64 v311; // r8
  __int64 v312; // rax
  __int16 v313; // dx
  __int64 v314; // rax
  unsigned int v315; // eax
  unsigned int v316; // edx
  __int64 v317; // rsi
  unsigned int v318; // ebx
  __int64 v319; // rax
  __int64 v320; // rbx
  __int64 *v321; // rax
  __int64 v322; // rdx
  __int64 v323; // r14
  __int64 v324; // r15
  __int64 v325; // rcx
  __int64 v326; // r8
  int v327; // eax
  int v328; // edx
  int v329; // r9d
  __int128 v330; // rax
  int v331; // r9d
  __int64 v332; // rbx
  __int64 v333; // rax
  __int64 v334; // r13
  unsigned __int16 *v335; // rax
  __int64 v336; // rdx
  unsigned __int16 v337; // ax
  __int128 v338; // [rsp-30h] [rbp-6A0h]
  __int128 v339; // [rsp-20h] [rbp-690h]
  __int128 v340; // [rsp-10h] [rbp-680h]
  unsigned __int32 v341; // [rsp+0h] [rbp-670h]
  const __m128i *v342; // [rsp+0h] [rbp-670h]
  __int64 v343; // [rsp+8h] [rbp-668h]
  __int64 v344; // [rsp+10h] [rbp-660h]
  int v345; // [rsp+20h] [rbp-650h]
  int v346; // [rsp+24h] [rbp-64Ch]
  int v347; // [rsp+24h] [rbp-64Ch]
  int v348; // [rsp+24h] [rbp-64Ch]
  __int64 v349; // [rsp+28h] [rbp-648h]
  int v350; // [rsp+28h] [rbp-648h]
  __int64 v351; // [rsp+28h] [rbp-648h]
  unsigned int v352; // [rsp+30h] [rbp-640h]
  char v353; // [rsp+30h] [rbp-640h]
  __int128 v354; // [rsp+30h] [rbp-640h]
  __int64 v355; // [rsp+30h] [rbp-640h]
  int v356; // [rsp+40h] [rbp-630h]
  unsigned __int64 v357; // [rsp+40h] [rbp-630h]
  __int16 v358; // [rsp+42h] [rbp-62Eh]
  __int64 v359; // [rsp+48h] [rbp-628h]
  __int64 v360; // [rsp+48h] [rbp-628h]
  __int64 v361; // [rsp+48h] [rbp-628h]
  unsigned __int64 *v362; // [rsp+48h] [rbp-628h]
  __int64 v363; // [rsp+48h] [rbp-628h]
  __int64 v364; // [rsp+50h] [rbp-620h]
  __int64 v365; // [rsp+50h] [rbp-620h]
  int v366; // [rsp+50h] [rbp-620h]
  unsigned int v367; // [rsp+50h] [rbp-620h]
  unsigned __int16 v368; // [rsp+58h] [rbp-618h]
  __int64 v369; // [rsp+58h] [rbp-618h]
  __int64 v370; // [rsp+58h] [rbp-618h]
  unsigned int v371; // [rsp+58h] [rbp-618h]
  char v372; // [rsp+58h] [rbp-618h]
  unsigned int v373; // [rsp+58h] [rbp-618h]
  __int64 v374; // [rsp+58h] [rbp-618h]
  __int64 v375; // [rsp+60h] [rbp-610h]
  __int64 v376; // [rsp+60h] [rbp-610h]
  __int64 v377; // [rsp+60h] [rbp-610h]
  __int64 v378; // [rsp+60h] [rbp-610h]
  __int64 v379; // [rsp+60h] [rbp-610h]
  __int128 v380; // [rsp+60h] [rbp-610h]
  int v381; // [rsp+70h] [rbp-600h]
  __int64 v382; // [rsp+70h] [rbp-600h]
  int v383; // [rsp+70h] [rbp-600h]
  unsigned int v384; // [rsp+70h] [rbp-600h]
  __int64 v385; // [rsp+70h] [rbp-600h]
  __int64 v386; // [rsp+70h] [rbp-600h]
  __int64 v387; // [rsp+70h] [rbp-600h]
  char v388; // [rsp+70h] [rbp-600h]
  __int64 v389; // [rsp+70h] [rbp-600h]
  __int64 v390; // [rsp+70h] [rbp-600h]
  __int64 v391; // [rsp+70h] [rbp-600h]
  __int64 v392; // [rsp+70h] [rbp-600h]
  __int64 v393; // [rsp+78h] [rbp-5F8h]
  unsigned int v394; // [rsp+80h] [rbp-5F0h]
  __int64 v395; // [rsp+80h] [rbp-5F0h]
  __int64 v396; // [rsp+80h] [rbp-5F0h]
  __int128 v397; // [rsp+80h] [rbp-5F0h]
  __int64 v398; // [rsp+80h] [rbp-5F0h]
  char v399; // [rsp+80h] [rbp-5F0h]
  char v400; // [rsp+80h] [rbp-5F0h]
  __int64 v401; // [rsp+80h] [rbp-5F0h]
  unsigned int v402; // [rsp+80h] [rbp-5F0h]
  __int64 v403; // [rsp+80h] [rbp-5F0h]
  unsigned int v404; // [rsp+80h] [rbp-5F0h]
  int v405; // [rsp+80h] [rbp-5F0h]
  unsigned int v406; // [rsp+90h] [rbp-5E0h]
  __int64 v407; // [rsp+90h] [rbp-5E0h]
  char v408; // [rsp+90h] [rbp-5E0h]
  __int128 v409; // [rsp+90h] [rbp-5E0h]
  int v410; // [rsp+90h] [rbp-5E0h]
  int v411; // [rsp+90h] [rbp-5E0h]
  __int64 v412; // [rsp+90h] [rbp-5E0h]
  __int128 v413; // [rsp+90h] [rbp-5E0h]
  __int128 v414; // [rsp+90h] [rbp-5E0h]
  unsigned int v415; // [rsp+90h] [rbp-5E0h]
  __int128 v416; // [rsp+A0h] [rbp-5D0h]
  __int16 v417; // [rsp+A0h] [rbp-5D0h]
  __int64 v418; // [rsp+A0h] [rbp-5D0h]
  _QWORD *v419; // [rsp+A0h] [rbp-5D0h]
  __int64 v420; // [rsp+A0h] [rbp-5D0h]
  __int64 v421; // [rsp+A0h] [rbp-5D0h]
  unsigned int v422; // [rsp+A0h] [rbp-5D0h]
  __int64 v423; // [rsp+A0h] [rbp-5D0h]
  _QWORD *v424; // [rsp+A0h] [rbp-5D0h]
  unsigned int v425; // [rsp+A0h] [rbp-5D0h]
  __int64 v426; // [rsp+B0h] [rbp-5C0h]
  unsigned int v427; // [rsp+B0h] [rbp-5C0h]
  int v428; // [rsp+B0h] [rbp-5C0h]
  __int64 v429; // [rsp+B0h] [rbp-5C0h]
  __int16 v430; // [rsp+B0h] [rbp-5C0h]
  _DWORD *v431; // [rsp+B8h] [rbp-5B8h]
  unsigned int v432; // [rsp+B8h] [rbp-5B8h]
  __int64 v433; // [rsp+B8h] [rbp-5B8h]
  __int64 v434; // [rsp+C0h] [rbp-5B0h]
  __int64 v435; // [rsp+C0h] [rbp-5B0h]
  unsigned int v436; // [rsp+C8h] [rbp-5A8h]
  unsigned __int64 v437; // [rsp+D8h] [rbp-598h]
  __int64 v438; // [rsp+E0h] [rbp-590h]
  __int128 v439; // [rsp+E0h] [rbp-590h]
  __int64 v440; // [rsp+E8h] [rbp-588h]
  __m128i v441; // [rsp+150h] [rbp-520h] BYREF
  __int64 v442; // [rsp+160h] [rbp-510h] BYREF
  __int64 v443; // [rsp+168h] [rbp-508h]
  unsigned int v444; // [rsp+170h] [rbp-500h] BYREF
  __int64 v445; // [rsp+178h] [rbp-4F8h]
  __int64 v446; // [rsp+180h] [rbp-4F0h] BYREF
  int v447; // [rsp+188h] [rbp-4E8h]
  unsigned __int16 v448; // [rsp+190h] [rbp-4E0h] BYREF
  signed __int64 v449; // [rsp+198h] [rbp-4D8h]
  __int16 v450; // [rsp+1A0h] [rbp-4D0h] BYREF
  __int64 v451; // [rsp+1A8h] [rbp-4C8h]
  __int64 v452; // [rsp+1B0h] [rbp-4C0h] BYREF
  char v453; // [rsp+1B8h] [rbp-4B8h]
  __int64 v454; // [rsp+1C0h] [rbp-4B0h] BYREF
  __int64 v455; // [rsp+1C8h] [rbp-4A8h]
  __int64 v456; // [rsp+1D0h] [rbp-4A0h]
  signed __int64 v457; // [rsp+1D8h] [rbp-498h]
  __int64 v458; // [rsp+1E0h] [rbp-490h]
  __int64 v459; // [rsp+1E8h] [rbp-488h]
  __int64 v460; // [rsp+1F0h] [rbp-480h]
  __int64 v461; // [rsp+1F8h] [rbp-478h]
  unsigned __int64 v462; // [rsp+200h] [rbp-470h] BYREF
  __int64 v463; // [rsp+208h] [rbp-468h]
  unsigned __int64 v464; // [rsp+210h] [rbp-460h] BYREF
  __int64 v465; // [rsp+218h] [rbp-458h]
  unsigned __int64 v466; // [rsp+220h] [rbp-450h] BYREF
  __int64 v467; // [rsp+228h] [rbp-448h]
  __int64 v468; // [rsp+230h] [rbp-440h] BYREF
  int v469; // [rsp+238h] [rbp-438h]
  int v470; // [rsp+23Ch] [rbp-434h]
  unsigned __int64 *v471; // [rsp+430h] [rbp-240h] BYREF
  __int64 v472; // [rsp+438h] [rbp-238h]
  unsigned __int64 v473; // [rsp+440h] [rbp-230h] BYREF
  unsigned int v474; // [rsp+448h] [rbp-228h]

  v2 = a1;
  v3 = (__int64 *)a2[5];
  v4 = *v3;
  v5 = _mm_loadu_si128((const __m128i *)(v3 + 5));
  v6 = v3[5];
  LODWORD(v3) = *((_DWORD *)v3 + 12);
  v441 = _mm_loadu_si128((const __m128i *)a2[5]);
  v7 = *(_DWORD *)(v4 + 24);
  v438 = v6;
  v436 = (unsigned int)v3;
  v8 = (__int16 *)a2[6];
  v437 = v5.m128i_u64[1];
  v9 = *v8;
  v443 = *((_QWORD *)v8 + 1);
  LOWORD(v442) = v9;
  v10 = v441.m128i_i32[2];
  v11 = *(_QWORD *)(v4 + 48) + 16LL * v441.m128i_u32[2];
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LOWORD(v444) = v12;
  v445 = v13;
  if ( v7 != 51 )
  {
    v14 = (__int64)a2;
    *((_QWORD *)&v15 + 1) = a2[10];
    v446 = *((_QWORD *)&v15 + 1);
    if ( *((_QWORD *)&v15 + 1) )
    {
      sub_B96E90((__int64)&v446, *((__int64 *)&v15 + 1), 1);
      v7 = *(_DWORD *)(v4 + 24);
    }
    v447 = *(_DWORD *)(v14 + 72);
    if ( v7 == 157 )
    {
      v26 = *(_QWORD *)(v4 + 40);
      if ( v438 == *(_QWORD *)(v26 + 80) && *(_DWORD *)(v26 + 88) == v436 )
      {
        v24 = *(_QWORD *)(v26 + 40);
        v55 = *(_QWORD *)(v26 + 48);
        sub_32B3670((__int64)a1, *(_QWORD *)(v4 + 56));
        if ( (_WORD)v444 )
        {
          if ( (unsigned __int16)(v444 - 2) > 7u
            && (unsigned __int16)(v444 - 17) > 0x6Cu
            && (unsigned __int16)(v444 - 176) > 0x1Fu )
          {
            goto LABEL_14;
          }
        }
        else if ( !sub_3007070((__int64)&v444) )
        {
          goto LABEL_14;
        }
        v24 = sub_33FAFB0(*a1, v24, v55, &v446, (unsigned int)v442, v443);
        goto LABEL_14;
      }
      v27 = *(_DWORD *)(v438 + 24);
      if ( v27 != 35 && v27 != 11 )
      {
        v434 = 0;
        goto LABEL_9;
      }
    }
    else
    {
      if ( v7 == 167 )
      {
        if ( !(unsigned __int8)sub_33DE9F0(*a1, v5.m128i_i64[0], v5.m128i_i64[1], 0) )
        {
          v36 = *(__int64 **)(v4 + 40);
          v37 = *v36;
          v38 = v36[1];
          v39 = *(_QWORD *)(*v36 + 48) + 16LL * *((unsigned int *)v36 + 2);
          v40 = *(_WORD *)v39;
          v41 = *(_QWORD *)(v39 + 8);
          if ( v40 == (_WORD)v442 && (v443 == v41 || v40) )
          {
            v24 = v37;
          }
          else
          {
            LOWORD(v471) = v40;
            v472 = v41;
            v107 = sub_3280A00((__int64)&v471, (unsigned int)v442, v443);
            v109 = *a1;
            *((_QWORD *)&v340 + 1) = v38;
            *(_QWORD *)&v340 = v37;
            if ( v107 )
              v24 = sub_33FAF80(v109, 216, (unsigned int)&v446, v442, v443, v108, v340);
            else
              v24 = sub_33FAF80(v109, 215, (unsigned int)&v446, v442, v443, v108, v340);
          }
          goto LABEL_14;
        }
        goto LABEL_154;
      }
      v16 = *(_DWORD *)(v438 + 24);
      if ( v16 != 11 )
      {
        v434 = 0;
        if ( v16 != 35 )
        {
          if ( v7 != 168 )
            goto LABEL_9;
          goto LABEL_57;
        }
      }
    }
    v28 = (unsigned __int16)v444;
    if ( (_WORD)v444 )
    {
      if ( (unsigned __int16)(v444 - 17) > 0x9Eu )
      {
        if ( v7 == 156 )
        {
          v45 = a1[1];
          v434 = v438;
LABEL_59:
          if ( *(_QWORD *)(v45 + 8 * v28 + 112) )
          {
            v46 = 0;
            if ( *(_DWORD *)(v4 + 24) == 156 )
            {
              v119 = *(_QWORD *)(v434 + 96);
              v120 = *(_QWORD **)(v119 + 24);
              if ( *(_DWORD *)(v119 + 32) > 0x40u )
                v120 = (_QWORD *)*v120;
              v46 = 40LL * (unsigned int)v120;
            }
            v47 = (__int64 *)(*(_QWORD *)(v4 + 40) + v46);
            v48 = *v47;
            v49 = v47[1];
            v50 = *(_QWORD *)(*v47 + 48) + 16LL * *((unsigned int *)v47 + 2);
            v51 = *(_WORD *)v50;
            v52 = *(_QWORD *)(v50 + 8);
            v53 = *(_QWORD *)(v4 + 56);
            if ( !v53 )
              goto LABEL_164;
            v54 = 1;
            do
            {
              if ( v441.m128i_i32[2] == *(_DWORD *)(v53 + 8) )
              {
                if ( !v54 )
                  goto LABEL_164;
                v53 = *(_QWORD *)(v53 + 32);
                if ( !v53 )
                  goto LABEL_151;
                if ( v441.m128i_i32[2] == *(_DWORD *)(v53 + 8) )
                  goto LABEL_164;
                v54 = 0;
              }
              v53 = *(_QWORD *)(v53 + 32);
            }
            while ( v53 );
            if ( v54 == 1 )
            {
LABEL_164:
              v111 = *(__int64 (**)())(*(_QWORD *)v45 + 1712LL);
              if ( v111 == sub_2FE35E0 )
                goto LABEL_528;
              v418 = v49;
              *((_QWORD *)&v15 + 1) = v444;
              v430 = v51;
              v433 = v52;
              v113 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v111)(v45, v444, v445);
              v52 = v433;
              v51 = v430;
              v49 = v418;
              if ( !v113 )
              {
LABEL_528:
                *((_QWORD *)&v15 + 1) = v49;
                v417 = v51;
                v429 = v52;
                v112 = sub_33CF170(v48, v49);
                v52 = v429;
                v51 = v417;
                if ( !v112 )
                  goto LABEL_9;
              }
            }
LABEL_151:
            if ( v51 == (_WORD)v442 && (v51 || v52 == v443) )
            {
              v24 = v48;
              goto LABEL_14;
            }
          }
LABEL_9:
          v17 = *((_BYTE *)v2 + 34);
          v426 = *v2;
          v431 = *(_DWORD **)(*v2 + 16LL);
          v18 = *(__int64 **)(v14 + 40);
          v19 = *v18;
          v20 = v18[1];
          v21 = *((_DWORD *)v18 + 2);
          v22 = *v18;
          v416 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 5));
          v23 = *(_DWORD *)(v18[5] + 24);
          if ( v23 != 35 && v23 != 11 )
            goto LABEL_11;
          v29 = *(_QWORD *)(v19 + 56);
          v30 = 1;
          if ( !v29 )
            goto LABEL_11;
          do
          {
            while ( *(_DWORD *)(v29 + 8) != v21 )
            {
              v29 = *(_QWORD *)(v29 + 32);
              if ( !v29 )
                goto LABEL_32;
            }
            if ( !v30 )
              goto LABEL_11;
            v31 = *(_QWORD *)(v29 + 32);
            if ( !v31 )
              goto LABEL_33;
            if ( *(_DWORD *)(v31 + 8) == v21 )
              goto LABEL_11;
            v29 = *(_QWORD *)(v31 + 32);
            v30 = 0;
          }
          while ( v29 );
LABEL_32:
          if ( v30 == 1 )
          {
LABEL_11:
            if ( (_WORD)v444 )
            {
              if ( (unsigned __int16)(v444 - 176) <= 0x34u )
                goto LABEL_13;
              v428 = word_4456340[(unsigned __int16)v444 - 1];
            }
            else
            {
              if ( sub_3007100((__int64)&v444) )
                goto LABEL_13;
              v428 = sub_3007130((__int64)&v444, *((__int64 *)&v15 + 1));
            }
            v406 = sub_32844A0((unsigned __int16 *)&v444, *((__int64 *)&v15 + 1));
            if ( !v434 )
              goto LABEL_106;
            if ( (_WORD)v442 )
            {
              if ( (unsigned __int16)(v442 - 10) <= 6u
                || (unsigned __int16)(v442 - 126) <= 0x31u
                || (unsigned __int16)(v442 - 208) <= 0x14u )
              {
LABEL_82:
                v58 = *(_QWORD *)(v434 + 96);
                v59 = *(_QWORD **)(v58 + 24);
                if ( *(_DWORD *)(v58 + 32) > 0x40u )
                  v59 = (_QWORD *)*v59;
                v60 = (unsigned int)v59;
                v61 = 1LL << (char)v59;
                LODWORD(v465) = v428;
                v62 = 1LL << (char)v59;
                if ( (unsigned int)v428 > 0x40 )
                {
                  v422 = v60;
                  v403 = v61;
                  sub_C43690((__int64)&v464, 0, 0);
                  v62 = v403;
                  if ( (unsigned int)v465 > 0x40 )
                  {
                    *(_QWORD *)(v464 + 8LL * (v422 >> 6)) |= v403;
LABEL_87:
                    *((_QWORD *)&v15 + 1) = *v2;
                    sub_33D4EF0(&v471, *v2, v441.m128i_i64[0], v441.m128i_i64[1], &v464, 0);
                    v394 = v472;
                    if ( (unsigned int)v472 > 0x40 )
                      v63 = sub_C44630((__int64)&v471);
                    else
                      v63 = sub_39FAC40(v471);
                    v381 = v63;
                    if ( v474 > 0x40 )
                    {
                      if ( v63 + (unsigned int)sub_C44630((__int64)&v473) != v394 )
                      {
LABEL_95:
                        if ( v473 )
                          j_j___libc_free_0_0(v473);
LABEL_97:
                        v65 = v472;
LABEL_98:
                        if ( v65 > 0x40 && v471 )
                          j_j___libc_free_0_0((unsigned __int64)v471);
                        if ( (unsigned int)v465 > 0x40 && v464 )
                          j_j___libc_free_0_0(v464);
                        goto LABEL_104;
                      }
                    }
                    else
                    {
                      v64 = sub_39FAC40(v473);
                      v65 = v394;
                      v56 = (__int64)&v473;
                      if ( v394 != v381 + v64 )
                        goto LABEL_98;
                    }
                    *((_QWORD *)&v15 + 1) = sub_300AC80((unsigned __int16 *)&v442, *((__int64 *)&v15 + 1));
                    if ( *((void **)&v15 + 1) == sub_C33340() )
                      sub_C3C640(&v466, *((__int64 *)&v15 + 1), &v473);
                    else
                      sub_C3B160((__int64)&v466, *((_DWORD **)&v15 + 1), (__int64 *)&v473);
                    v66 = v2[1];
                    v67 = *(__int64 (**)())(*(_QWORD *)v66 + 616LL);
                    if ( v67 != sub_2FE3170 )
                    {
                      *((_QWORD *)&v15 + 1) = &v466;
                      if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64 *, _QWORD, __int64, _QWORD))v67)(
                             v66,
                             &v466,
                             (unsigned int)v442,
                             v443,
                             0) )
                      {
                        v24 = sub_33FE6E0(*v2, &v466, &v446, (unsigned int)v442, v443, 0);
                        sub_91D830(&v466);
                        sub_969240((__int64 *)&v473);
                        sub_969240((__int64 *)&v471);
                        sub_969240((__int64 *)&v464);
                        goto LABEL_14;
                      }
                    }
                    sub_91D830(&v466);
                    if ( v474 <= 0x40 )
                      goto LABEL_97;
                    goto LABEL_95;
                  }
                }
                else
                {
                  v464 = 0;
                }
                v464 |= v62;
                goto LABEL_87;
              }
            }
            else if ( (unsigned __int8)sub_3007030((__int64)&v442) )
            {
              goto LABEL_82;
            }
LABEL_104:
            v68 = *(_DWORD *)(v4 + 24);
            if ( v68 != 234 )
            {
LABEL_105:
              if ( v68 != 165 )
              {
LABEL_106:
                v69 = *(_QWORD *)(v4 + 56);
                if ( v69 )
                {
                  v70 = *(_QWORD *)(v4 + 56);
                  while ( 1 )
                  {
                    v71 = *(_QWORD *)(v70 + 16);
                    if ( *(_DWORD *)(v71 + 24) != 158 )
                      break;
                    v71 = *(_QWORD *)(v71 + 40);
                    if ( v4 != *(_QWORD *)v71 )
                      break;
                    if ( v10 != *(_DWORD *)(v71 + 8) )
                      break;
                    v71 = *(unsigned int *)(*(_QWORD *)(v71 + 40) + 24LL);
                    if ( (_DWORD)v71 != 35 && (_DWORD)v71 != 11 )
                      break;
                    v70 = *(_QWORD *)(v70 + 32);
                    if ( !v70 )
                      goto LABEL_179;
                  }
LABEL_109:
                  if ( (unsigned int)(*((_DWORD *)v2 + 6) - 1) > 1 )
                    goto LABEL_110;
                  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(*v2 + 40LL)) )
                    goto LABEL_110;
                  v56 = *(_QWORD *)(v14 + 40);
                  v142 = *(_QWORD *)(v56 + 40);
                  v344 = *(_QWORD *)v56;
                  v343 = *(_QWORD *)(v56 + 8);
                  v143 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v56 + 48LL) + 16LL * *(unsigned int *)(v56 + 8));
                  v144 = *v143;
                  *((_QWORD *)&v15 + 1) = *((_QWORD *)v143 + 1);
                  v71 = *(unsigned int *)(v142 + 24);
                  v448 = v144;
                  v449 = *((_QWORD *)&v15 + 1);
                  LOBYTE(v56) = (_DWORD)v71 == 11;
                  LOBYTE(v71) = (_DWORD)v71 == 35;
                  LOBYTE(v56) = v71 | v56;
                  v372 = v56;
                  if ( !(_BYTE)v56 )
                    goto LABEL_110;
                  if ( (_WORD)v144 )
                  {
                    if ( (unsigned __int16)(v144 - 17) > 0xD3u )
                    {
                      LOWORD(v454) = v144;
                      v455 = *((_QWORD *)&v15 + 1);
                      goto LABEL_248;
                    }
                    LOWORD(v144) = word_4456580[v144 - 1];
                    v282 = 0;
                  }
                  else
                  {
                    if ( !sub_30070B0((__int64)&v448) )
                    {
                      v455 = *((_QWORD *)&v15 + 1);
                      LOWORD(v454) = 0;
                      goto LABEL_355;
                    }
                    LOWORD(v144) = sub_3009970((__int64)&v448, *((__int64 *)&v15 + 1), v209, v210, v211);
                  }
                  LOWORD(v454) = v144;
                  v455 = v282;
                  if ( (_WORD)v144 )
                  {
LABEL_248:
                    if ( (_WORD)v144 == 1 || (unsigned __int16)(v144 - 504) <= 7u )
LABEL_525:
                      BUG();
                    v71 = (signed __int64)byte_444C4A0;
                    v360 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v144 - 16];
LABEL_251:
                    v145 = *(__int16 **)(v14 + 48);
                    v56 = v448;
                    v146 = *v145;
                    v147 = *((_QWORD *)v145 + 1);
                    v450 = v146;
                    v423 = v147;
                    v451 = v147;
                    if ( v448 )
                    {
                      if ( (unsigned __int16)(v448 - 17) <= 0xD3u )
                      {
                        v71 = (signed __int64)word_4456580;
                        v56 = (unsigned __int16)word_4456580[v448 - 1];
                        v148 = 0;
                        goto LABEL_254;
                      }
                    }
                    else
                    {
                      v404 = v448;
                      v208 = sub_30070B0((__int64)&v448);
                      v56 = v404;
                      if ( v208 )
                      {
                        v56 = (unsigned int)sub_3009970((__int64)&v448, *((__int64 *)&v15 + 1), v71, v404, v57);
                        v148 = v71;
                        goto LABEL_254;
                      }
                    }
                    v148 = v449;
LABEL_254:
                    if ( v146 != (_WORD)v56 )
                      goto LABEL_110;
                    if ( v146 )
                    {
                      if ( (unsigned __int16)(v146 - 2) > 7u )
                        goto LABEL_110;
                    }
                    else if ( v423 != v148 || !sub_30070A0((__int64)&v450) )
                    {
                      goto LABEL_110;
                    }
                    v466 = (unsigned __int64)&v468;
                    v467 = 0x2000000000LL;
                    v472 = 0x2000000000LL;
                    v471 = &v473;
                    v149 = *(_QWORD *)(v142 + 96);
                    v135 = *(_DWORD *)(v149 + 32) <= 0x40u;
                    v150 = *(_QWORD **)(v149 + 24);
                    if ( !v135 )
                      v150 = (_QWORD *)*v150;
                    v468 = v14;
                    v469 = (_DWORD)v150 * v360;
                    LODWORD(v467) = 1;
                    v424 = v2;
                    v412 = v14;
                    v389 = v4;
                    v470 = v360;
                    v151 = 1;
                    while ( 1 )
                    {
                      v71 = v466 + 16LL * v151 - 16;
                      v56 = *(_QWORD *)v71;
                      v152 = *(_DWORD *)(v71 + 12);
                      v153 = *(_DWORD *)(v71 + 8);
                      LODWORD(v467) = v151 - 1;
                      v365 = v56;
                      if ( v152 <= 0 )
                        break;
                      if ( v448 )
                      {
                        if ( v448 == 1 || (unsigned __int16)(v448 - 504) <= 7u )
                          goto LABEL_525;
                        v259 = 16LL * (v448 - 1);
                        v260 = byte_444C4A0[v259 + 8];
                        v261 = *(_QWORD *)&byte_444C4A0[v259];
                        LOBYTE(v459) = v260;
                        v458 = v261;
                      }
                      else
                      {
                        v458 = sub_3007260((__int64)&v448);
                        v459 = v154;
                      }
                      v452 = v458;
                      v453 = v459;
                      if ( v153 >= (unsigned __int64)sub_CA1930(&v452) )
                        break;
                      if ( v448 )
                      {
                        if ( v448 == 1 || (unsigned __int16)(v448 - 504) <= 7u )
                          goto LABEL_525;
                        v256 = 16LL * (v448 - 1);
                        v257 = byte_444C4A0[v256 + 8];
                        v258 = *(_QWORD *)&byte_444C4A0[v256];
                        LOBYTE(v461) = v257;
                        v460 = v258;
                      }
                      else
                      {
                        v460 = sub_3007260((__int64)&v448);
                        v461 = v155;
                      }
                      v462 = v460;
                      LOBYTE(v463) = v461;
                      if ( v152 + v153 > (unsigned __int64)sub_CA1930(&v462) )
                        break;
                      v353 = 0;
                      if ( *(_QWORD *)(v365 + 56) )
                      {
                        v347 = v152;
                        v157 = *(_QWORD *)(v365 + 56);
                        do
                        {
                          v166 = *(_QWORD *)(v157 + 16);
                          v167 = *(_DWORD *)(v166 + 24);
                          switch ( v167 )
                          {
                            case 192:
                              v172 = *(_QWORD **)(v166 + 40);
                              v71 = v172[5];
                              v56 = *(unsigned int *)(v71 + 24);
                              if ( (_DWORD)v56 != 11 && (_DWORD)v56 != 35 )
                                goto LABEL_289;
                              v56 = v365;
                              if ( v365 != *v172 )
                                goto LABEL_289;
                              v199 = *(_QWORD *)(v71 + 96);
                              v200 = *(_QWORD **)(v199 + 24);
                              if ( *(_DWORD *)(v199 + 32) > 0x40u )
                                v200 = (_QWORD *)*v200;
                              v201 = v153 + (_DWORD)v200;
                              v202 = v347 - (_DWORD)v200;
                              v71 = HIDWORD(v467);
                              if ( (unsigned int)v467 >= (unsigned __int64)HIDWORD(v467) )
                              {
                                *((_QWORD *)&v15 + 1) = &v468;
                                v345 = v202;
                                v341 = v201;
                                v71 = sub_C8D7D0((__int64)&v466, (__int64)&v468, 0, 0x10u, &v462, v156);
                                v289 = (unsigned int)v467;
                                v56 = v71 + 16LL * (unsigned int)v467;
                                if ( v56 )
                                {
                                  *(_QWORD *)v56 = v166;
                                  *(_DWORD *)(v56 + 8) = v341;
                                  *(_DWORD *)(v56 + 12) = v345;
                                  v289 = (unsigned int)v467;
                                }
                                v290 = 16 * v289;
                                v291 = (const __m128i *)v466;
                                if ( v290 )
                                {
                                  v292 = (const __m128i *)(v71 + v290);
                                  v56 = v71;
                                  do
                                  {
                                    if ( v56 )
                                      *(__m128i *)v56 = _mm_loadu_si128(v291);
                                    v56 += 16;
                                    ++v291;
                                  }
                                  while ( v292 != (const __m128i *)v56 );
                                  v291 = (const __m128i *)v466;
                                }
                                v293 = v462;
                                if ( v291 != (const __m128i *)&v468 )
                                {
                                  v342 = (const __m128i *)v71;
                                  _libc_free((unsigned __int64)v291);
                                  v71 = (signed __int64)v342;
                                }
                                LODWORD(v467) = v467 + 1;
                                v466 = v71;
                                HIDWORD(v467) = v293;
                              }
                              else
                              {
                                v203 = v466 + 16LL * (unsigned int)v467;
                                if ( v203 )
                                {
                                  *(_QWORD *)v203 = v166;
                                  *(_DWORD *)(v203 + 8) = v201;
                                  *(_DWORD *)(v203 + 12) = v202;
                                }
                                LODWORD(v467) = v467 + 1;
                              }
                              break;
                            case 216:
                              v158 = *(unsigned __int16 **)(v166 + 48);
                              v159 = *v158;
                              v160 = *((_QWORD *)v158 + 1);
                              LOWORD(v462) = v159;
                              v463 = v160;
                              if ( v159 )
                              {
                                if ( v159 == 1 || (unsigned __int16)(v159 - 504) <= 7u )
                                  goto LABEL_525;
                                v207 = 16LL * (v159 - 1);
                                v162 = *(_QWORD *)&byte_444C4A0[v207];
                                v163 = byte_444C4A0[v207 + 8];
                              }
                              else
                              {
                                v464 = sub_3007260((__int64)&v462);
                                v465 = v161;
                                v162 = v464;
                                v163 = v465;
                              }
                              LOBYTE(v463) = v163;
                              v462 = v162;
                              if ( (unsigned int)v467 >= HIDWORD(v467) )
                              {
                                v262 = sub_C8D7D0(
                                         (__int64)&v466,
                                         (__int64)&v468,
                                         0,
                                         0x10u,
                                         (unsigned __int64 *)&v454,
                                         v156);
                                *((_QWORD *)&v15 + 1) = sub_CA1930(&v462);
                                v263 = (unsigned int)v467;
                                v264 = v262 + 16LL * (unsigned int)v467;
                                if ( v264 )
                                {
                                  *(_QWORD *)v264 = v166;
                                  *(_DWORD *)(v264 + 8) = v153;
                                  *(_DWORD *)(v264 + 12) = DWORD2(v15);
                                  v263 = (unsigned int)v467;
                                }
                                v265 = 16 * v263;
                                v218 = v265 == 0;
                                v266 = (const __m128i *)v466;
                                v56 = v262 + v265;
                                v267 = (__m128i *)v262;
                                if ( !v218 )
                                {
                                  do
                                  {
                                    if ( v267 )
                                      *v267 = _mm_loadu_si128(v266);
                                    ++v267;
                                    ++v266;
                                  }
                                  while ( (__m128i *)v56 != v267 );
                                  v266 = (const __m128i *)v466;
                                }
                                v268 = v454;
                                if ( v266 != (const __m128i *)&v468 )
                                  _libc_free((unsigned __int64)v266);
                                LODWORD(v467) = v467 + 1;
                                v466 = v262;
                                HIDWORD(v467) = v268;
                              }
                              else
                              {
                                v56 = sub_CA1930(&v462);
                                v164 = v467;
                                v165 = v466 + 16LL * (unsigned int)v467;
                                if ( v165 )
                                {
                                  *(_QWORD *)v165 = v166;
                                  *(_DWORD *)(v165 + 8) = v153;
                                  *(_DWORD *)(v165 + 12) = v56;
                                  v164 = v467;
                                }
                                v71 = (unsigned int)(v164 + 1);
                                LODWORD(v467) = v71;
                              }
                              break;
                            case 156:
                              v353 = v372;
                              break;
                            default:
                              goto LABEL_289;
                          }
                          v157 = *(_QWORD *)(v157 + 32);
                        }
                        while ( v157 );
                        if ( v353 )
                        {
                          v71 = HIDWORD(v472);
                          if ( (unsigned int)v472 >= (unsigned __int64)HIDWORD(v472) )
                          {
                            *((_QWORD *)&v15 + 1) = &v473;
                            v284 = (unsigned __int64 *)sub_C8D7D0((__int64)&v471, (__int64)&v473, 0, 0x10u, &v462, v156);
                            v285 = (unsigned int)v472;
                            v286 = &v284[2 * (unsigned int)v472];
                            if ( v286 )
                            {
                              *((_DWORD *)v286 + 2) = v153;
                              *((_DWORD *)v286 + 3) = v347;
                              *v286 = v365;
                              v285 = (unsigned int)v472;
                            }
                            v71 = (signed __int64)v471;
                            v56 = (__int64)v284;
                            v287 = (const __m128i *)&v471[2 * v285];
                            while ( (const __m128i *)v71 != v287 )
                            {
                              if ( v56 )
                                *(__m128i *)v56 = _mm_loadu_si128((const __m128i *)v71);
                              v56 += 16;
                              v71 += 16LL;
                            }
                            v288 = v462;
                            if ( v471 != &v473 )
                              _libc_free((unsigned __int64)v471);
                            LODWORD(v472) = v472 + 1;
                            v471 = v284;
                            HIDWORD(v472) = v288;
                          }
                          else
                          {
                            v283 = &v471[2 * (unsigned int)v472];
                            if ( v283 )
                            {
                              v56 = v365;
                              *((_DWORD *)v283 + 2) = v153;
                              *((_DWORD *)v283 + 3) = v347;
                              *v283 = v365;
                            }
                            LODWORD(v472) = v472 + 1;
                          }
                        }
                      }
                      v151 = v467;
                      if ( !(_DWORD)v467 )
                      {
                        v173 = (__int64 *)v471;
                        v14 = v412;
                        v2 = v424;
                        v4 = v389;
                        v235 = *((unsigned int *)v471 + 3);
                        v415 = v235;
                        if ( (_DWORD)v235 != (_DWORD)v360 )
                        {
                          v236 = sub_2D5B750(&v448);
                          v463 = v237;
                          v462 = v236;
                          v71 = sub_CA1930(&v462) % v235;
                          if ( !v71 )
                          {
                            *((_QWORD *)&v15 + 1) = &v471[2 * (unsigned int)v472];
                            v238 = sub_3278280((__int64)v471, *((__int64 *)&v15 + 1), v235);
                            v56 = *((_QWORD *)&v15 + 1);
                            if ( *((_QWORD *)&v15 + 1) == v238 )
                            {
                              v239 = sub_327FC40(*(_QWORD **)(*v424 + 64LL), v235);
                              v392 = v240;
                              v425 = v239;
                              v241 = sub_2D5B750(&v448);
                              v463 = v242;
                              v462 = v241;
                              v243 = sub_CA1930(&v462);
                              *((_QWORD *)&v15 + 1) = v425;
                              v244 = sub_327FCF0(*(__int64 **)(*v2 + 64LL), v425, v392, v243 / v235, 0);
                              v245 = v71;
                              v246 = v244;
                              if ( !*((_BYTE *)v2 + 34)
                                || (_WORD)v425
                                && (v247 = v2[1],
                                    v71 = (unsigned __int16)v425,
                                    *(_QWORD *)(v247 + 8LL * (unsigned __int16)v425 + 112))
                                && (_WORD)v246
                                && (v71 = (unsigned __int16)v246, *(_QWORD *)(v247 + 8LL * (unsigned __int16)v246 + 112)) )
                              {
                                if ( !*((_BYTE *)v2 + 33)
                                  || (*((_QWORD *)&v15 + 1) = 234,
                                      *(_QWORD *)&v15 = v2[1],
                                      (unsigned __int8)sub_328A020(v15, 0xEAu, v246, v245, 0))
                                  && (*((_QWORD *)&v15 + 1) = 158,
                                      (unsigned __int8)sub_328A020(v15, 0x9Eu, v246, v245, 0)) )
                                {
                                  *((_QWORD *)&v15 + 1) = v246;
                                  *(_QWORD *)&v354 = sub_33FB890(*v2, v246, v245, v344, v343);
                                  *((_QWORD *)&v354 + 1) = v71;
                                  v248 = v471;
                                  v362 = &v471[2 * (unsigned int)v472];
                                  while ( v248 != v362 )
                                  {
                                    v249 = *v248;
                                    v454 = *(_QWORD *)(*v248 + 80);
                                    if ( v454 )
                                    {
                                      v357 = v249;
                                      sub_325F5D0(&v454);
                                      v249 = v357;
                                    }
                                    *(_QWORD *)&v15 = *v2;
                                    v248 += 2;
                                    LODWORD(v455) = *(_DWORD *)(v249 + 72);
                                    *(_QWORD *)&v250 = sub_3400EE0(v15, *((_DWORD *)v248 - 2) / v415, &v454, 0, v57);
                                    v252 = sub_3406EB0(v15, 158, (unsigned int)&v454, v425, v392, v251, v354, v250);
                                    *((_QWORD *)&v15 + 1) = *(v248 - 2);
                                    v463 = v253;
                                    v462 = v252;
                                    sub_32EB790((__int64)v2, *((__int64 *)&v15 + 1), (__int64 *)&v462, 1, 1);
                                    sub_9C6650(&v454);
                                  }
                                  v173 = (__int64 *)v471;
LABEL_292:
                                  if ( v173 != (__int64 *)&v473 )
                                    _libc_free((unsigned __int64)v173);
                                  if ( (__int64 *)v466 != &v468 )
                                    _libc_free(v466);
                                  if ( v372 )
                                  {
                                    v24 = v14;
                                    goto LABEL_14;
                                  }
LABEL_110:
                                  LODWORD(v466) = sub_3281170(&v444, *((__int64 *)&v15 + 1), v71, v56, v57);
                                  v72 = v466;
                                  v74 = v73;
                                  v467 = v73;
                                  v75 = (unsigned int)v466;
                                  v407 = v466;
                                  v395 = v73;
                                  if ( sub_3280B30((__int64)&v442, (unsigned int)v466, v73) )
                                  {
                                    v78 = v2[1];
                                    v79 = *(__int64 (**)())(*(_QWORD *)v78 + 1392LL);
                                    if ( v79 == sub_2FE3480 )
                                      goto LABEL_13;
                                    v75 = v72;
                                    if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v79)(
                                            v78,
                                            v72,
                                            v74,
                                            (unsigned int)v442,
                                            v443) )
                                      goto LABEL_13;
                                  }
                                  v80 = 0;
                                  if ( *(_DWORD *)(v4 + 24) != 234 )
                                    goto LABEL_114;
                                  if ( !(unsigned __int8)sub_3286E00(&v441) )
                                    goto LABEL_13;
                                  v212 = *(_QWORD *)(**(_QWORD **)(v4 + 40) + 48LL)
                                       + 16LL * *(unsigned int *)(*(_QWORD *)(v4 + 40) + 8LL);
                                  v213 = *(_WORD *)v212;
                                  v472 = *(_QWORD *)(v212 + 8);
                                  LOWORD(v471) = v213;
                                  if ( !sub_32801E0((__int64)&v471) )
                                    goto LABEL_13;
                                  v75 = (unsigned int)sub_3281170(&v471, v75, v214, v215, v216);
                                  if ( sub_3280A00((__int64)&v466, v75, v217) )
                                    goto LABEL_13;
                                  v218 = (unsigned int)sub_3281500(&v471, v75) == v428;
                                  v219 = *(_QWORD *)(v4 + 40);
                                  v80 = !v218;
                                  v220 = *(_QWORD *)v219;
                                  LODWORD(v219) = *(_DWORD *)(v219 + 8);
                                  v441.m128i_i64[0] = v220;
                                  v441.m128i_i32[2] = v219;
                                  LODWORD(v466) = sub_3281170(&v471, v75, v220, v221, v222);
                                  v467 = v76;
LABEL_114:
                                  v81 = *((_BYTE *)v2 + 33);
                                  if ( v81 )
                                    goto LABEL_367;
                                  if ( v434 && (unsigned int)(*(_DWORD *)(*(_QWORD *)*v2 + 544LL) - 42) > 1 )
                                    goto LABEL_13;
                                  if ( !(unsigned __int8)sub_3286E00(&v441) )
                                    goto LABEL_13;
                                  v82 = v441.m128i_i64[0];
                                  if ( *(_DWORD *)(v441.m128i_i64[0] + 24) != 298
                                    || (*(_BYTE *)(v441.m128i_i64[0] + 33) & 0xC) != 0
                                    || (*(_WORD *)(v441.m128i_i64[0] + 32) & 0x380) != 0 )
                                  {
                                    goto LABEL_13;
                                  }
                                  v75 = v441.m128i_i64[0];
                                  if ( !(unsigned __int8)sub_33CFFC0(v438, v441.m128i_i64[0])
                                    && *(_DWORD *)(v82 + 24) == 298
                                    && (unsigned __int8)sub_3287C60(v82) )
                                  {
                                    v83 = v2[1];
                                    v84 = *v2;
                                    v471 = *(unsigned __int64 **)(v14 + 80);
                                    if ( v471 )
                                    {
                                      v382 = v84;
                                      sub_325F5D0((__int64 *)&v471);
                                      v84 = v382;
                                    }
                                    v75 = (unsigned int)v442;
                                    LODWORD(v472) = *(_DWORD *)(v14 + 72);
                                    v85 = sub_3472160(
                                            v83,
                                            v442,
                                            v443,
                                            (unsigned int)&v471,
                                            v444,
                                            v445,
                                            v5.m128i_i64[0],
                                            v5.m128i_i64[1],
                                            v82,
                                            v84);
                                    sub_9C6650(&v471);
                                    if ( v85 )
                                    {
                                      v24 = v85;
                                      goto LABEL_14;
                                    }
                                  }
                                  v81 = *((_BYTE *)v2 + 33);
LABEL_367:
                                  if ( !v434 || v81 != 1 )
                                    goto LABEL_13;
                                  v225 = *(_QWORD *)(v434 + 96);
                                  v226 = *(_QWORD *)(v225 + 24);
                                  if ( *(_DWORD *)(v225 + 32) > 0x40u )
                                    v226 = *(_QWORD *)v226;
                                  v227 = v441.m128i_i64[0];
                                  v228 = v226;
                                  v229 = *(_DWORD *)(v441.m128i_i64[0] + 24);
                                  switch ( v229 )
                                  {
                                    case 298:
                                      if ( (*(_BYTE *)(v441.m128i_i64[0] + 33) & 0xC) == 0
                                        && (*(_WORD *)(v441.m128i_i64[0] + 32) & 0x380) == 0 )
                                      {
LABEL_374:
                                        v230 = *(_QWORD *)(v227 + 56);
                                        if ( v230 )
                                        {
                                          v231 = 1;
                                          do
                                          {
                                            if ( !*(_DWORD *)(v230 + 8) )
                                            {
                                              if ( !v231 )
                                                goto LABEL_13;
                                              v230 = *(_QWORD *)(v230 + 32);
                                              if ( !v230 )
                                                goto LABEL_514;
                                              if ( !*(_DWORD *)(v230 + 8) )
                                                goto LABEL_13;
                                              v231 = 0;
                                            }
                                            v230 = *(_QWORD *)(v230 + 32);
                                          }
                                          while ( v230 );
                                          if ( v231 == 1 )
                                            break;
LABEL_514:
                                          if ( (unsigned __int8)sub_3287C60(v227) )
                                          {
                                            if ( v228 == -1 )
                                            {
                                              v24 = sub_3288990(*v2, v407, v395);
                                            }
                                            else
                                            {
                                              v332 = 0;
                                              v333 = sub_3472160(
                                                       v2[1],
                                                       v407,
                                                       v395,
                                                       (unsigned int)&v446,
                                                       v444,
                                                       v445,
                                                       v438,
                                                       v436 | v437 & 0xFFFFFFFF00000000LL,
                                                       v227,
                                                       *v2);
                                              if ( v333 )
                                                v332 = v333;
                                              v24 = v332;
                                            }
                                            goto LABEL_14;
                                          }
                                        }
                                      }
                                      break;
                                    case 167:
                                      v272 = *(unsigned int **)(v441.m128i_i64[0] + 40);
                                      v227 = *(_QWORD *)v272;
                                      v273 = *(_QWORD *)(*(_QWORD *)v272 + 48LL) + 16LL * v272[2];
                                      if ( (_WORD)v466 == *(_WORD *)v273
                                        && (*(_QWORD *)(v273 + 8) == v467 || *(_WORD *)v273)
                                        && *(_DWORD *)(v227 + 24) == 298
                                        && (*(_BYTE *)(v227 + 33) & 0xC) == 0
                                        && (*(_WORD *)(v227 + 32) & 0x380) == 0
                                        && (unsigned __int8)sub_3286E00(&v441) )
                                      {
                                        goto LABEL_374;
                                      }
                                      break;
                                    case 165:
                                      if ( (unsigned __int8)sub_3286E00(&v441) && !v80 )
                                      {
                                        if ( v428 < (int)v226 )
                                        {
                                          v228 = -1;
                                        }
                                        else
                                        {
                                          v226 = (unsigned int)v226;
                                          v228 = *(_DWORD *)(*(_QWORD *)(v227 + 96) + 4LL * (unsigned int)v226);
                                        }
                                        v274 = *(_QWORD *)(v227 + 40);
                                        if ( v428 <= v228 )
                                        {
                                          v275 = *(_QWORD *)(v274 + 40);
                                          v276 = *(_DWORD *)(v274 + 48);
                                        }
                                        else
                                        {
                                          v275 = *(_QWORD *)v274;
                                          v276 = *(_DWORD *)(v274 + 8);
                                        }
                                        v441.m128i_i64[0] = v275;
                                        v441.m128i_i32[2] = v276;
                                        v227 = v275;
                                        if ( *(_DWORD *)(v275 + 24) == 234 )
                                        {
                                          if ( !(unsigned __int8)sub_3286E00(&v441) )
                                            break;
                                          v307 = *(_QWORD *)(v227 + 40);
                                          v227 = *(_QWORD *)v307;
                                          LODWORD(v307) = *(_DWORD *)(v307 + 8);
                                          v441.m128i_i64[0] = v227;
                                          v441.m128i_i32[2] = v307;
                                        }
                                        if ( *(_DWORD *)(v227 + 24) == 298
                                          && (*(_BYTE *)(v227 + 33) & 0xC) == 0
                                          && (*(_WORD *)(v227 + 32) & 0x380) == 0 )
                                        {
                                          if ( v428 <= v228 )
                                            v228 -= v428;
                                          v438 = sub_3400BD0(
                                                   *v2,
                                                   v228,
                                                   (unsigned int)&v446,
                                                   *(unsigned __int16 *)(*(_QWORD *)(v438 + 48) + 16LL * v436),
                                                   *(_QWORD *)(*(_QWORD *)(v438 + 48) + 16LL * v436 + 8),
                                                   0,
                                                   0,
                                                   v226);
                                          v436 = v277;
                                          v437 = v277 | v5.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                                          goto LABEL_374;
                                        }
                                      }
                                      break;
                                    default:
                                      if ( v229 == 159 && !v80 )
                                      {
                                        v308 = sub_3281170(&v444, v75, v76, v434, v77);
                                        if ( (_WORD)v442 == v308 && (v308 || v443 == v309) )
                                        {
                                          if ( !*((_BYTE *)v2 + 34)
                                            || (v334 = v2[1],
                                                v335 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v227 + 40) + 48LL)
                                                                          + 16LL
                                                                          * *(unsigned int *)(*(_QWORD *)(v227 + 40)
                                                                                            + 8LL)),
                                                v336 = *v335,
                                                v472 = *((_QWORD *)v335 + 1),
                                                LOWORD(v471) = v336,
                                                (v337 = sub_3281170(&v471, v75, v336, v310, v311)) != 0)
                                            && *(_QWORD *)(v334 + 8LL * v337 + 112) )
                                          {
                                            v312 = *(_QWORD *)(**(_QWORD **)(v227 + 40) + 48LL)
                                                 + 16LL * *(unsigned int *)(*(_QWORD *)(v227 + 40) + 8LL);
                                            v313 = *(_WORD *)v312;
                                            v314 = *(_QWORD *)(v312 + 8);
                                            LOWORD(v471) = v313;
                                            v472 = v314;
                                            v315 = sub_3281500(&v471, v75);
                                            v316 = (unsigned int)v226 % v315;
                                            v317 = (unsigned int)v226 % v315;
                                            v318 = (unsigned int)v226 / v315;
                                            *(_QWORD *)&v439 = sub_3400BD0(
                                                                 *v2,
                                                                 v316,
                                                                 (unsigned int)&v446,
                                                                 *(unsigned __int16 *)(*(_QWORD *)(v438 + 48)
                                                                                     + 16LL * v436),
                                                                 *(_QWORD *)(*(_QWORD *)(v438 + 48) + 16LL * v436 + 8),
                                                                 0,
                                                                 0,
                                                                 v315);
                                            v319 = v318;
                                            v320 = *v2;
                                            v321 = (__int64 *)(*(_QWORD *)(v227 + 40) + 40 * v319);
                                            *((_QWORD *)&v439 + 1) = v322;
                                            v323 = *v321;
                                            v324 = v321[1];
                                            v327 = sub_3281170(&v471, v317, v322, v325, v326);
                                            *((_QWORD *)&v338 + 1) = v324;
                                            *(_QWORD *)&v338 = v323;
                                            *(_QWORD *)&v330 = sub_3406EB0(
                                                                 v320,
                                                                 158,
                                                                 (unsigned int)&v446,
                                                                 v327,
                                                                 v328,
                                                                 v329,
                                                                 v338,
                                                                 v439);
                                            v24 = sub_33FAF80(*v2, 234, (unsigned int)&v446, v442, v443, v331, v330);
                                            goto LABEL_14;
                                          }
                                        }
                                      }
                                      break;
                                  }
LABEL_13:
                                  v24 = 0;
                                  goto LABEL_14;
                                }
                              }
                            }
                          }
LABEL_290:
                          v173 = (__int64 *)v471;
                        }
                        v372 = 0;
                        goto LABEL_292;
                      }
                    }
LABEL_289:
                    v2 = v424;
                    v14 = v412;
                    v4 = v389;
                    goto LABEL_290;
                  }
LABEL_355:
                  v456 = sub_3007260((__int64)&v454);
                  v457 = v71;
                  LODWORD(v360) = v456;
                  goto LABEL_251;
                }
LABEL_179:
                LODWORD(v467) = v428;
                if ( (unsigned int)v428 > 0x40 )
                {
                  sub_C43690((__int64)&v466, 0, 0);
                  v69 = *(_QWORD *)(v4 + 56);
                }
                else
                {
                  v466 = 0;
                }
                if ( !v69 )
                {
LABEL_206:
                  if ( (unsigned __int8)sub_32E2DA0((__int64)v2, v441.m128i_i64[0], v441.m128i_i32[2], (int)&v466, 1u) )
                  {
                    if ( *(_DWORD *)(v14 + 24) )
                      sub_32B3E80((__int64)v2, v14, 1, 0, v122, v123);
                    v24 = v14;
                  }
                  else
                  {
                    LODWORD(v472) = v406;
                    if ( v406 > 0x40 )
                    {
                      sub_C43690((__int64)&v471, -1, 1);
                    }
                    else
                    {
                      v174 = (unsigned __int64 *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v406);
                      if ( !v406 )
                        v174 = 0;
                      v471 = v174;
                    }
                    *((_QWORD *)&v15 + 1) = v441.m128i_i64[0];
                    if ( !(unsigned __int8)sub_32D0760(
                                             (__int64)v2,
                                             v441.m128i_i64[0],
                                             v441.m128i_i32[2],
                                             (int)&v471,
                                             (int)&v466,
                                             1u) )
                    {
                      sub_969240((__int64 *)&v471);
                      sub_969240((__int64 *)&v466);
                      goto LABEL_109;
                    }
                    if ( *(_DWORD *)(v14 + 24) )
                      sub_32B3E80((__int64)v2, v14, 1, 0, v175, v176);
                    v24 = v14;
                    sub_969240((__int64 *)&v471);
                  }
                  if ( (unsigned int)v467 > 0x40 && v466 )
                    j_j___libc_free_0_0(v466);
                  goto LABEL_14;
                }
                v419 = v2;
                v398 = v14;
                while ( 1 )
                {
                  v115 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v69 + 16) + 40LL) + 40LL) + 96LL);
                  v116 = *(_DWORD *)(v115 + 32);
                  if ( v116 > 0x40 )
                    break;
                  v114 = *(_QWORD *)(v115 + 24);
                  if ( v114 < (unsigned int)v428 )
                    goto LABEL_188;
LABEL_184:
                  v69 = *(_QWORD *)(v69 + 32);
                  if ( !v69 )
                  {
                    v2 = v419;
                    v14 = v398;
                    goto LABEL_206;
                  }
                }
                if ( v116 - (unsigned int)sub_C444A0(v115 + 24) > 0x40 )
                  goto LABEL_184;
                v114 = **(_QWORD **)(v115 + 24);
                if ( v114 >= (unsigned int)v428 )
                  goto LABEL_184;
LABEL_188:
                v117 = 1LL << v114;
                if ( (unsigned int)v467 > 0x40 )
                  *(_QWORD *)(v466 + 8LL * ((unsigned int)v114 >> 6)) |= v117;
                else
                  v466 |= v117;
                goto LABEL_184;
              }
              v124 = *(_QWORD *)(v434 + 96);
              v125 = *(_QWORD **)(v124 + 24);
              if ( *(_DWORD *)(v124 + 32) > 0x40u )
                v125 = (_QWORD *)*v125;
              v126 = *(unsigned int *)(*(_QWORD *)(v4 + 96) + 4LL * (unsigned int)v125);
              if ( (_DWORD)v126 == -1 )
              {
                v24 = sub_3288990(*v2, (unsigned int)v442, v443);
                goto LABEL_14;
              }
              v127 = *(_QWORD *)(v4 + 40);
              if ( v428 <= (int)v126 )
              {
                v128 = *(unsigned int *)(v127 + 48);
                v57 = *(_QWORD *)(v127 + 40);
                v126 = (unsigned int)(v126 - v428);
                v420 = v57;
              }
              else
              {
                v57 = *(_QWORD *)v127;
                v128 = *(unsigned int *)(v127 + 8);
                v420 = *(_QWORD *)v127;
              }
              v129 = v128;
              v130 = *((_BYTE *)v2 + 33);
              if ( *(_DWORD *)(v420 + 24) != 156 )
              {
                if ( !v130
                  || (*((_QWORD *)&v15 + 1) = 158,
                      v387 = v57,
                      v393 = v128,
                      v401 = v2[1],
                      v131 = sub_328D6E0(v401, 0x9Eu, v444),
                      v57 = v387,
                      v129 = v393,
                      v131)
                  || !(_WORD)v444
                  || !*(_QWORD *)(v401 + 8LL * (unsigned __int16)v444 + 112)
                  || *(_BYTE *)(v401 + 500LL * (unsigned __int16)v444 + 6579) == 2 )
                {
                  v132 = *v2;
                  v440 = v129;
                  *(_QWORD *)&v133 = sub_3400EE0(v132, (int)v126, &v446, 0, v57);
                  *((_QWORD *)&v339 + 1) = v440;
                  *(_QWORD *)&v339 = v420;
                  v24 = sub_3406EB0(v132, 158, (unsigned int)&v446, v442, v443, v440, v339, v133);
                  goto LABEL_14;
                }
                goto LABEL_106;
              }
              if ( v130 && sub_328D6E0(v2[1], 0xA5u, v444) && !(unsigned __int8)sub_3286E00(&v441) )
                goto LABEL_13;
              v196 = (unsigned int *)(*(_QWORD *)(v420 + 40) + 40 * v126);
              v197 = *(_QWORD *)v196;
              v198 = *(_QWORD *)(*(_QWORD *)v196 + 48LL) + 16LL * v196[2];
              if ( *(_WORD *)v198 != (_WORD)v442 || v443 != *(_QWORD *)(v198 + 8) && !*(_WORD *)v198 )
                v197 = sub_33FB160(*v2, *(_QWORD *)v196, *((_QWORD *)v196 + 1), &v446, (unsigned int)v442, v443);
              v24 = v197;
              goto LABEL_14;
            }
            if ( !sub_3280180((__int64)&v444) || !(unsigned __int8)sub_3286E00(&v441) )
              goto LABEL_106;
            v388 = *(_BYTE *)sub_2E79000(*(__int64 **)(*v2 + 40LL)) ^ 1;
            v134 = *(_QWORD *)(v434 + 96);
            v135 = *(_DWORD *)(v134 + 32) <= 0x40u;
            v136 = *(_QWORD **)(v134 + 24);
            if ( !v135 )
              v136 = (_QWORD *)*v136;
            v402 = (unsigned int)v136;
            v137 = (int)v136;
            v56 = (unsigned int)(v428 - 1);
            if ( v388 )
              v56 = 0;
            v138 = *(_QWORD *)(v4 + 40);
            *((_QWORD *)&v15 + 1) = *(_QWORD *)v138;
            v57 = *(unsigned int *)(v138 + 8);
            v421 = *(_QWORD *)(v138 + 8);
            v139 = *(_QWORD *)v138;
            if ( (_DWORD)v56 == v137 )
            {
              v269 = *(_QWORD *)(*((_QWORD *)&v15 + 1) + 48LL) + 16LL * (unsigned int)v57;
              v373 = v57;
              v270 = *(_QWORD *)(v269 + 8);
              LOWORD(v471) = *(_WORD *)v269;
              v472 = v270;
              v271 = sub_32801C0((__int64)&v471);
              v139 = *((_QWORD *)&v15 + 1);
              v57 = v373;
              if ( v271 )
              {
                v24 = sub_33FAFB0(*v2, *((_QWORD *)&v15 + 1), v421, &v446, (unsigned int)v442, v443);
                goto LABEL_14;
              }
            }
            if ( !*((_BYTE *)v2 + 34) )
              goto LABEL_237;
            v371 = v57;
            v140 = *(_QWORD *)(v139 + 48) + 16LL * (unsigned int)v57;
            v378 = v139;
            v141 = *(_QWORD *)(v140 + 8);
            LOWORD(v471) = *(_WORD *)v140;
            v472 = v141;
            if ( !sub_3280180((__int64)&v471) )
              goto LABEL_237;
            v57 = v371;
            if ( *(_DWORD *)(v378 + 24) != 167 )
              goto LABEL_237;
            v294 = v371;
            v374 = v378;
            v379 = sub_3263630(v378, v294);
            v295 = *(_QWORD *)(v374 + 40);
            *((_QWORD *)&v15 + 1) = *(unsigned int *)(v295 + 8);
            if ( v379 != sub_3263630(*(_QWORD *)v295, DWORD2(v15)) )
              goto LABEL_237;
            v296 = *(unsigned int **)(v374 + 40);
            v297 = v296[2];
            *(_QWORD *)&v15 = *(_QWORD *)v296;
            v355 = v297;
            v298 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v296 + 48LL) + 16 * v297);
            v380 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(v374 + 40));
            v363 = *((_QWORD *)v298 + 1);
            v367 = *v298;
            v299 = (unsigned __int64 *)sub_3262090(v15, v297);
            v472 = v300;
            v471 = v299;
            v301 = sub_CA1930(&v471);
            v57 = v15;
            *((_QWORD *)&v15 + 1) = v301;
            v56 = v355;
            v302 = (unsigned int)v301 / v406;
            v303 = 0;
            if ( !v388 )
              v303 = v302 - 1;
            if ( v402 >= v302 || v406 >= DWORD2(v15) )
            {
LABEL_237:
              v68 = *(_DWORD *)(v4 + 24);
              goto LABEL_105;
            }
            if ( v303 != v402 )
            {
              if ( !v388 )
                v402 = ~v402 + v302;
              v304 = *v2;
              *(_QWORD *)&v305 = sub_3400E40(*v2, v406 * v402, v367, v363, &v446);
              v57 = sub_3406EB0(v304, 192, (unsigned int)&v446, v367, v363, v367, v380, v305);
              v56 = v306;
            }
            v24 = sub_33FAFB0(
                    *v2,
                    v57,
                    v56 | *((_QWORD *)&v380 + 1) & 0xFFFFFFFF00000000LL,
                    &v446,
                    (unsigned int)v442,
                    v443);
LABEL_14:
            if ( v446 )
              sub_B91220((__int64)&v446, v446);
            return v24;
          }
LABEL_33:
          v32 = *(_DWORD *)(v19 + 24);
          v33 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v431 + 1368LL);
          if ( v33 != sub_2FE4300 )
          {
            v369 = v20;
            *((_QWORD *)&v15 + 1) = (unsigned int)v32;
            v376 = v19;
            v385 = v19;
            v399 = *((_BYTE *)v2 + 34);
            v410 = *(_DWORD *)(v19 + 24);
            v118 = v33((__int64)v431, v32);
            v32 = v410;
            v22 = v385;
            v17 = v399;
            v19 = v376;
            v20 = v369;
            if ( v118 )
              goto LABEL_131;
            goto LABEL_130;
          }
          v34 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v431 + 1360LL);
          if ( v34 == sub_2FE3400 )
          {
            if ( v32 <= 98 )
            {
              if ( v32 > 55 )
              {
                switch ( v32 )
                {
                  case '8':
                  case ':':
                  case '?':
                  case '@':
                  case 'D':
                  case 'F':
                  case 'L':
                  case 'M':
                  case 'R':
                  case 'S':
                  case '`':
                  case 'b':
                    goto LABEL_131;
                  default:
                    break;
                }
              }
LABEL_37:
              if ( v32 > 56 )
              {
LABEL_38:
                switch ( v32 )
                {
                  case '9':
                  case ';':
                  case '<':
                  case '=':
                  case '>':
                  case 'T':
                  case 'U':
                  case 'a':
                  case 'c':
                  case 'd':
                    goto LABEL_131;
                  default:
                    break;
                }
              }
LABEL_130:
              if ( v32 != 208 )
                goto LABEL_11;
LABEL_131:
              if ( *(_DWORD *)(v22 + 68) != 1 )
                goto LABEL_11;
              v86 = *(__int64 (**)())(*(_QWORD *)v431 + 1688LL);
              if ( v86 == sub_2FE35C0 )
                goto LABEL_11;
              v383 = v32;
              *((_QWORD *)&v15 + 1) = v19;
              v396 = v22;
              v408 = v17;
              if ( !((unsigned __int8 (__fastcall *)(_DWORD *, __int64, __int64))v86)(v431, v19, v20) )
                goto LABEL_11;
              v88 = *(unsigned __int16 **)(v14 + 48);
              v89 = v383;
              v90 = v408;
              v91 = v396;
              v92 = *v88;
              v368 = *v88;
              v375 = *((_QWORD *)v88 + 1);
              if ( v383 == 208 )
              {
                v168 = (unsigned __int16 *)(*(_QWORD *)(v396 + 48) + 16LL * v21);
                v169 = *v168;
                v170 = *((_QWORD *)v168 + 1);
                LOWORD(v471) = v169;
                v472 = v170;
                if ( (_WORD)v169 )
                {
                  *((_QWORD *)&v15 + 1) = (unsigned __int16)word_4456580[v169 - 1];
                  v171 = 0;
                }
                else
                {
                  v204 = sub_3009970((__int64)&v471, *((__int64 *)&v15 + 1), v170, v92, v87);
                  v89 = 208;
                  v91 = v396;
                  *((_QWORD *)&v15 + 1) = v204;
                  v90 = v408;
                  v171 = v205;
                }
                if ( v368 != WORD4(v15) || !v368 && v375 != v171 || v90 )
                  goto LABEL_11;
              }
              v93 = *(_QWORD *)(v91 + 40);
              v94 = *(_QWORD *)v93;
              v95 = (__int128)_mm_loadu_si128((const __m128i *)v93);
              v96 = *(_DWORD *)(v93 + 8);
              v97 = _mm_loadu_si128((const __m128i *)(v93 + 40));
              v98 = *(_QWORD *)(v93 + 40);
              v364 = v94;
              LODWORD(v467) = 1;
              v466 = 0;
              v352 = v96;
              v359 = v98;
              v397 = (__int128)v97;
              if ( *(_DWORD *)(v94 + 24) != 156 )
                goto LABEL_529;
              v350 = v89;
              v390 = v91;
              v15 = v95;
              v194 = sub_326A930(v15, DWORD2(v15), 1u);
              v91 = v390;
              v89 = v350;
              if ( !v194 )
              {
                v195 = sub_33CA720(v364);
                v91 = v390;
                v89 = v350;
                if ( !v195 )
                {
LABEL_529:
                  v346 = v89;
                  *((_QWORD *)&v15 + 1) = &v466;
                  v349 = v91;
                  v99 = sub_33D1410(v364, &v466);
                  v91 = v349;
                  v89 = v346;
                  if ( !v99 )
                  {
                    if ( *(_DWORD *)(v359 + 24) != 156
                      || (v15 = v397, v254 = sub_326A930(v15, DWORD2(v15), 1u), v91 = v349, v89 = v346, !v254)
                      && (v255 = sub_33CA720(v359), v91 = v349, v89 = v346, !v255) )
                    {
                      *((_QWORD *)&v15 + 1) = &v466;
                      v348 = v89;
                      v351 = v91;
                      v223 = sub_33D1410(v359, &v466);
                      v91 = v351;
                      v89 = v348;
                      if ( !v223 )
                      {
                        if ( (unsigned int)v467 <= 0x40 )
                          goto LABEL_11;
                        v105 = v466;
                        v104 = 0;
                        if ( !v466 )
                          goto LABEL_11;
                        goto LABEL_141;
                      }
                    }
                  }
                }
              }
              if ( v89 != 208 )
              {
                v384 = v89;
                *(_QWORD *)&v409 = sub_3406EB0(v426, 158, (unsigned int)&v446, v368, v375, v89, v95, v416);
                *((_QWORD *)&v409 + 1) = v101 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
                *(_QWORD *)&v397 = sub_3406EB0(v426, 158, (unsigned int)&v446, v368, v375, v102, v397, v416);
                *((_QWORD *)&v397 + 1) = v103 | *((_QWORD *)&v397 + 1) & 0xFFFFFFFF00000000LL;
                *((_QWORD *)&v15 + 1) = v384;
                v104 = sub_3406EB0(v426, v384, (unsigned int)&v446, v368, v375, v384, v409, v397);
LABEL_139:
                if ( (unsigned int)v467 <= 0x40 || (v105 = v466) == 0 )
                {
LABEL_142:
                  if ( v104 )
                  {
                    v24 = v104;
                    goto LABEL_14;
                  }
                  goto LABEL_11;
                }
LABEL_141:
                j_j___libc_free_0_0(v105);
                goto LABEL_142;
              }
              v177 = (unsigned __int16 *)(*(_QWORD *)(v364 + 48) + 16LL * v352);
              v178 = *v177;
              v179 = *((_QWORD *)v177 + 1);
              LOWORD(v471) = v178;
              v472 = v179;
              if ( (_WORD)v178 )
              {
                v180 = 0;
                LOWORD(v181) = word_4456580[(unsigned __int16)v178 - 1];
              }
              else
              {
                v391 = v91;
                v181 = sub_3009970((__int64)&v471, *((__int64 *)&v15 + 1), v178, v364, v100);
                v91 = v391;
                v358 = HIWORD(v181);
                v180 = v224;
              }
              HIWORD(v182) = v358;
              LOWORD(v182) = v181;
              v361 = v91;
              v366 = v180;
              v356 = v182;
              *(_QWORD *)&v413 = sub_3406EB0(v426, 158, (unsigned int)&v446, v182, v180, v89, v95, v416);
              *((_QWORD *)&v413 + 1) = v183 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v397 = sub_3406EB0(v426, 158, (unsigned int)&v446, v356, v366, 0, v397, v416);
              *((_QWORD *)&v397 + 1) = v184 | *((_QWORD *)&v397 + 1) & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v185 = sub_33ED040(v426, *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v361 + 40) + 80LL) + 96LL));
              *((_QWORD *)&v15 + 1) = 208;
              v187 = sub_340F900(v426, 208, (unsigned int)&v446, v368, v375, v186, v413, v397, v185);
              *((_QWORD *)&v414 + 1) = v188;
              v189 = *(_QWORD *)(v361 + 48) + 16LL * v21;
              *(_QWORD *)&v414 = v187;
              v190 = *(_WORD *)v189;
              v191 = *(_QWORD *)(v189 + 8);
              LOWORD(v471) = v190;
              v472 = v191;
              if ( v190 )
              {
                v192 = v190 - 17;
                if ( (unsigned __int16)(v190 - 10) > 6u && (unsigned __int16)(v190 - 126) > 0x31u )
                {
                  if ( v192 > 0xD3u )
                  {
LABEL_313:
                    v193 = v431[15];
LABEL_314:
                    if ( !v193 || v368 == 2 )
                    {
LABEL_316:
                      v104 = v414;
                      goto LABEL_139;
                    }
                    LOWORD(v471) = v368;
                    v472 = v375;
                    if ( v368 )
                    {
                      v232 = v368 - 17;
                      if ( (unsigned __int16)(v368 - 10) > 6u && (unsigned __int16)(v368 - 126) > 0x31u )
                      {
                        if ( v232 > 0xD3u )
                        {
LABEL_386:
                          v233 = v431[15];
                          goto LABEL_387;
                        }
                        goto LABEL_456;
                      }
                      if ( v232 <= 0xD3u )
                      {
LABEL_456:
                        v233 = v431[17];
                        goto LABEL_387;
                      }
                    }
                    else
                    {
                      v405 = v193;
                      v280 = sub_3007030((__int64)&v471);
                      v281 = sub_30070B0((__int64)&v471);
                      v193 = v405;
                      if ( v281 )
                        goto LABEL_456;
                      if ( !v280 )
                        goto LABEL_386;
                    }
                    v233 = v431[16];
LABEL_387:
                    if ( v193 != v233 )
                    {
                      if ( v193 == 2 )
                      {
                        *(_QWORD *)&v278 = sub_33F7D60(v426, 2, 0);
                        *((_QWORD *)&v15 + 1) = 222;
                        v234 = sub_3406EB0(v426, 222, (unsigned int)&v446, v368, v375, v279, v414, v278);
                      }
                      else
                      {
                        *((_QWORD *)&v15 + 1) = v414;
                        v234 = sub_34070B0(v426, v414, *((_QWORD *)&v414 + 1), &v446, 2, 0);
                      }
                      *(_QWORD *)&v414 = v234;
                    }
                    goto LABEL_316;
                  }
LABEL_350:
                  v193 = v431[17];
                  goto LABEL_314;
                }
                if ( v192 <= 0xD3u )
                  goto LABEL_350;
              }
              else
              {
                v206 = sub_3007030((__int64)&v471);
                if ( sub_30070B0((__int64)&v471) )
                  goto LABEL_350;
                if ( !v206 )
                  goto LABEL_313;
              }
              v193 = v431[16];
              goto LABEL_314;
            }
            if ( v32 > 188 )
            {
              if ( (unsigned int)(v32 - 279) <= 7 )
                goto LABEL_131;
            }
            else
            {
              if ( v32 > 185 || (unsigned int)(v32 - 172) <= 0xB )
                goto LABEL_131;
              if ( v32 <= 100 )
                goto LABEL_38;
            }
          }
          else
          {
            v370 = v20;
            *((_QWORD *)&v15 + 1) = (unsigned int)v32;
            v377 = v19;
            v386 = v19;
            v400 = *((_BYTE *)v2 + 34);
            v411 = *(_DWORD *)(v19 + 24);
            v121 = v34((__int64)v431, v32);
            v32 = v411;
            v22 = v386;
            v17 = v400;
            v19 = v377;
            v20 = v370;
            if ( v121 )
              goto LABEL_131;
            if ( v411 <= 100 )
              goto LABEL_37;
          }
          if ( (unsigned int)(v32 - 190) <= 4 )
            goto LABEL_131;
          goto LABEL_130;
        }
        if ( v7 == 168 )
        {
          v434 = v438;
LABEL_58:
          v45 = a1[1];
          goto LABEL_59;
        }
LABEL_23:
        v434 = v438;
        goto LABEL_9;
      }
      v110 = *(_QWORD *)(v438 + 96);
      v435 = v110 + 24;
    }
    else
    {
      if ( !sub_30070D0((__int64)&v444) )
      {
        if ( v7 == 156 )
          goto LABEL_23;
        goto LABEL_45;
      }
      v110 = *(_QWORD *)(v438 + 96);
      v435 = v110 + 24;
      if ( !sub_3007100((__int64)&v444)
        || (sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead"),
            !(_WORD)v444) )
      {
        v43 = (unsigned int)sub_3007130((__int64)&v444, *((__int64 *)&v15 + 1));
LABEL_160:
        v432 = *(_DWORD *)(v110 + 32);
        if ( v432 > 0x40 )
        {
          v427 = v43;
          v42 = sub_C444A0(v435);
          v43 = v427;
          if ( v432 - v42 > 0x40 )
            goto LABEL_154;
          v44 = **(_QWORD **)(v110 + 24);
        }
        else
        {
          v44 = *(_QWORD *)(v110 + 24);
        }
        if ( v43 > v44 )
        {
          v7 = *(_DWORD *)(v4 + 24);
          if ( v7 == 156 )
          {
            v434 = v438;
LABEL_57:
            v28 = (unsigned __int16)v444;
            if ( !(_WORD)v444 )
              goto LABEL_9;
            goto LABEL_58;
          }
LABEL_45:
          v434 = v438;
          if ( v7 != 168 )
            goto LABEL_9;
          goto LABEL_57;
        }
LABEL_154:
        v106 = *a1;
        v471 = 0;
        LODWORD(v472) = 0;
        v24 = sub_33F17F0(v106, 51, &v471, v442, v443);
        if ( v471 )
          sub_B91220((__int64)&v471, (__int64)v471);
        goto LABEL_14;
      }
      if ( (unsigned __int16)(v444 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
    }
    v43 = word_4456340[(unsigned __int16)v444 - 1];
    goto LABEL_160;
  }
  v35 = *a1;
  v471 = 0;
  LODWORD(v472) = 0;
  v24 = sub_33F17F0(v35, 51, &v471, v442, v443);
  if ( v471 )
    sub_B91220((__int64)&v471, (__int64)v471);
  return v24;
}
