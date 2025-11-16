// Function: sub_1D54C20
// Address: 0x1d54c20
//
__int64 __fastcall sub_1D54C20(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 (*v6)(); // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  const __m128i *v9; // rsi
  __m128i *v10; // rdi
  __int64 v11; // rdx
  const __m128i *v12; // rcx
  const __m128i *v13; // r8
  __int64 v14; // r14
  __int64 v15; // rax
  __m128i *v16; // rdi
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  __m128i *v19; // rax
  __m128i *v20; // rax
  __int8 *v21; // rax
  const __m128i *v22; // r8
  signed __int64 v23; // r12
  __int64 v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  __m128i *v28; // rax
  __m128i *v29; // rax
  __int8 *v30; // rax
  __int64 v31; // r12
  __int64 v32; // r15
  unsigned int v33; // esi
  __int64 v34; // r8
  __int64 v35; // rcx
  __int64 *v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // rax
  __m128i v41; // xmm2
  __int64 *v42; // rsi
  __int64 v43; // r12
  int v44; // edx
  __int64 v45; // r14
  int v46; // r13d
  _QWORD *v47; // r13
  _QWORD *v48; // r12
  _QWORD *v49; // rdi
  __int64 v50; // rax
  _QWORD *v51; // rdx
  unsigned __int64 v52; // rax
  __int64 i; // rax
  __int64 v54; // r12
  __int64 (*v55)(); // rax
  __int64 v56; // r14
  __int64 v57; // r15
  __int64 v58; // r8
  unsigned int v59; // edi
  __int64 *v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // rcx
  _QWORD *v67; // r13
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rdx
  void *v72; // rdi
  unsigned int v73; // eax
  __int64 v74; // rdx
  __int64 v75; // r14
  __int64 v76; // rax
  unsigned int v77; // esi
  int v78; // esi
  int v79; // esi
  __int64 v80; // r9
  unsigned int v81; // ecx
  int v82; // eax
  __int64 *v83; // rdx
  __int64 v84; // rdi
  __int64 v85; // rax
  __int64 v86; // r12
  __int64 *v87; // rax
  __int64 v88; // rbx
  __int64 v89; // rsi
  int v90; // ecx
  int v91; // eax
  __int64 v92; // rsi
  unsigned __int8 v93; // al
  int v94; // eax
  __int64 v95; // rax
  int v96; // eax
  unsigned int v97; // eax
  __int64 v98; // rsi
  __int64 v99; // r8
  unsigned __int64 v100; // r10
  _QWORD *v101; // rax
  unsigned __int64 v102; // rax
  unsigned int v103; // esi
  int v104; // eax
  __int64 v105; // rax
  unsigned int v106; // eax
  __int64 v107; // rsi
  int v108; // ebx
  __int64 v109; // rdx
  __int64 v110; // r8
  int v111; // r9d
  __int64 v112; // r12
  unsigned int v113; // edx
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v116; // rdi
  unsigned int v117; // ecx
  __int64 *v118; // rdx
  __int64 v119; // r10
  __int64 v120; // rax
  unsigned __int8 (*v121)(void); // rax
  __int64 v122; // r12
  _QWORD *v123; // rbx
  _QWORD *v124; // rax
  __int64 v125; // rax
  _QWORD *v126; // r14
  _QWORD *v127; // rdx
  int v128; // eax
  char v129; // al
  __int64 v130; // r8
  __int64 v131; // rax
  unsigned int v132; // esi
  __int64 v133; // rdi
  __int64 v134; // r9
  unsigned int v135; // edx
  __int64 v136; // rbx
  _QWORD *v137; // rcx
  __int64 v138; // rbx
  _QWORD *v139; // rbx
  _QWORD *v140; // r12
  _QWORD *v141; // rdi
  __int64 v142; // rax
  __int64 v143; // rdx
  __int64 v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // r11
  int v147; // eax
  int v148; // eax
  __int64 v149; // rbx
  _QWORD *v150; // rax
  _QWORD *v151; // r14
  _QWORD *v152; // rax
  __int64 v153; // rax
  __int64 v154; // r12
  __int64 v155; // r13
  _QWORD *v156; // rdx
  _QWORD *v157; // rdx
  __int64 v158; // r12
  __int64 v159; // rdx
  __int64 v160; // r13
  __int64 v161; // rsi
  __int64 v162; // rax
  __int64 v163; // rbx
  __int64 v164; // rax
  __int64 v165; // rsi
  unsigned int v166; // ecx
  __int64 *v167; // rdx
  __int64 v168; // r8
  int v169; // eax
  __int64 v170; // rax
  unsigned __int64 v171; // rdx
  unsigned int v172; // ecx
  __int64 v173; // r8
  __int64 v174; // r8
  __int64 v175; // rdx
  __int64 v176; // r12
  __int64 v177; // r13
  unsigned int v178; // ecx
  unsigned int v179; // ecx
  unsigned int v180; // ecx
  __int64 v181; // rdi
  __int64 v182; // rdx
  __int64 v183; // rsi
  __int64 v184; // rax
  __int64 v185; // rdx
  __int64 v186; // r12
  __int64 v187; // rbx
  __int64 v188; // rbx
  __int64 v189; // rsi
  __int64 *v190; // rax
  __int64 result; // rax
  __int64 v192; // rdi
  _QWORD *v193; // rbx
  _QWORD *v194; // r14
  _QWORD *v195; // rdi
  __int64 *v196; // rdi
  unsigned int v197; // r8d
  __int64 *v198; // rcx
  int v199; // r11d
  int v200; // eax
  int v201; // r10d
  int v202; // r10d
  __int64 v203; // r8
  __int64 *v204; // rcx
  unsigned int v205; // ebx
  int v206; // esi
  __int64 v207; // rdi
  __int64 v208; // rbx
  bool v209; // zf
  __int64 v210; // rax
  __int64 v211; // r12
  _QWORD *v212; // rbx
  __int64 v213; // r14
  unsigned __int8 v214; // al
  __int64 v215; // r13
  __int64 v216; // r14
  __int64 v217; // rax
  __int64 v218; // rax
  unsigned __int64 v219; // rax
  __int64 *v220; // rdx
  __int64 v221; // r15
  __int64 *v222; // rbx
  __int64 *v223; // r14
  __int64 v224; // rax
  int v225; // edx
  int v226; // r9d
  int v227; // edx
  int v228; // r9d
  __int64 v229; // rdx
  __int64 v230; // rbx
  __int64 v231; // r12
  __int64 v232; // rdi
  __int64 v233; // rdi
  __int64 v234; // rax
  __int64 v235; // rax
  __int64 (__fastcall *v236)(__int64, unsigned __int8); // r13
  unsigned int v237; // eax
  __int64 v238; // rsi
  __int64 *v239; // rdx
  __int64 *v240; // r15
  __int64 v241; // rax
  __int64 v242; // r14
  unsigned __int32 v243; // r12d
  __int64 v244; // r13
  __int64 v245; // rsi
  __int64 v246; // r8
  __int64 v247; // r13
  __int64 v248; // rax
  unsigned __int64 v249; // rax
  __int64 j; // rax
  __int64 v251; // r12
  const char *v252; // rsi
  char v253; // r14
  _QWORD *v254; // rbx
  _QWORD *v255; // rdi
  _QWORD *v256; // rdx
  int v257; // r14d
  __int64 *v258; // rax
  int v259; // ecx
  int v260; // edx
  __int64 *v261; // r13
  __int64 v262; // rax
  unsigned int v263; // esi
  int v264; // eax
  unsigned int v265; // eax
  __int64 v266; // rsi
  _QWORD *v267; // rax
  __int64 v268; // rax
  int v269; // r11d
  int v270; // r11d
  __int64 v271; // r10
  unsigned int v272; // ecx
  __int64 v273; // r8
  int v274; // edi
  __int64 *v275; // rsi
  int v276; // r10d
  int v277; // r10d
  __int64 v278; // r9
  int v279; // esi
  unsigned int v280; // r13d
  __int64 *v281; // rcx
  __int64 v282; // rdi
  int v283; // eax
  int v284; // r10d
  int v285; // esi
  __int64 v286; // rcx
  __int64 v287; // r11
  unsigned int v288; // edx
  _QWORD *v289; // rdi
  int v290; // r11d
  __int64 *v291; // r8
  int v292; // eax
  int v293; // r10d
  __int64 v294; // r11
  unsigned int v295; // edx
  _QWORD *v296; // rdi
  int v297; // esi
  __int64 v298; // rax
  __int64 v299; // rax
  unsigned int v300; // esi
  int v301; // eax
  __int64 v302; // rax
  _QWORD *v303; // rax
  int v304; // [rsp+0h] [rbp-590h]
  __int64 v305; // [rsp+8h] [rbp-588h]
  int v306; // [rsp+10h] [rbp-580h]
  __int64 v307; // [rsp+10h] [rbp-580h]
  __int64 v308; // [rsp+18h] [rbp-578h]
  int v309; // [rsp+18h] [rbp-578h]
  __int64 v310; // [rsp+18h] [rbp-578h]
  __int64 v311; // [rsp+20h] [rbp-570h]
  unsigned __int64 v312; // [rsp+20h] [rbp-570h]
  unsigned __int64 v313; // [rsp+20h] [rbp-570h]
  unsigned __int64 v314; // [rsp+20h] [rbp-570h]
  int v315; // [rsp+28h] [rbp-568h]
  __int64 v316; // [rsp+28h] [rbp-568h]
  __int64 v317; // [rsp+28h] [rbp-568h]
  __int64 v318; // [rsp+28h] [rbp-568h]
  __int64 v319; // [rsp+28h] [rbp-568h]
  __int64 v320; // [rsp+30h] [rbp-560h]
  int v321; // [rsp+30h] [rbp-560h]
  unsigned __int64 v322; // [rsp+30h] [rbp-560h]
  __int64 v323; // [rsp+38h] [rbp-558h]
  unsigned __int64 v324; // [rsp+38h] [rbp-558h]
  __int64 v325; // [rsp+38h] [rbp-558h]
  unsigned __int64 v326; // [rsp+40h] [rbp-550h]
  int v327; // [rsp+48h] [rbp-548h]
  __int64 v328; // [rsp+48h] [rbp-548h]
  __int64 v329; // [rsp+50h] [rbp-540h]
  __int64 v330; // [rsp+50h] [rbp-540h]
  int v331; // [rsp+60h] [rbp-530h]
  unsigned __int64 v332; // [rsp+60h] [rbp-530h]
  __int64 v333; // [rsp+60h] [rbp-530h]
  __int64 v334; // [rsp+60h] [rbp-530h]
  __int64 v335; // [rsp+60h] [rbp-530h]
  __int64 v336; // [rsp+60h] [rbp-530h]
  __int64 v337; // [rsp+60h] [rbp-530h]
  __int64 v338; // [rsp+60h] [rbp-530h]
  __int64 v340; // [rsp+70h] [rbp-520h]
  __int64 v341; // [rsp+70h] [rbp-520h]
  unsigned int v342; // [rsp+70h] [rbp-520h]
  _QWORD *v343; // [rsp+70h] [rbp-520h]
  int v344; // [rsp+70h] [rbp-520h]
  __int64 v345; // [rsp+70h] [rbp-520h]
  __int64 v346; // [rsp+78h] [rbp-518h]
  __int64 v347; // [rsp+78h] [rbp-518h]
  __int64 v348; // [rsp+78h] [rbp-518h]
  __int64 v349; // [rsp+78h] [rbp-518h]
  __int32 v350; // [rsp+78h] [rbp-518h]
  __int64 v351; // [rsp+78h] [rbp-518h]
  __int64 v352; // [rsp+78h] [rbp-518h]
  __m128i *v353; // [rsp+80h] [rbp-510h]
  int v354; // [rsp+80h] [rbp-510h]
  __int64 v355; // [rsp+80h] [rbp-510h]
  __int64 v356; // [rsp+90h] [rbp-500h]
  __int64 v357; // [rsp+90h] [rbp-500h]
  _QWORD *v358; // [rsp+98h] [rbp-4F8h]
  __int64 v359; // [rsp+A0h] [rbp-4F0h]
  unsigned int v360; // [rsp+A0h] [rbp-4F0h]
  unsigned int v361; // [rsp+A0h] [rbp-4F0h]
  _QWORD *v362; // [rsp+A0h] [rbp-4F0h]
  unsigned int v363; // [rsp+A0h] [rbp-4F0h]
  unsigned int v364; // [rsp+A0h] [rbp-4F0h]
  __int64 *v365; // [rsp+A0h] [rbp-4F0h]
  __int64 v366; // [rsp+A0h] [rbp-4F0h]
  __int64 v367; // [rsp+A8h] [rbp-4E8h]
  int v368; // [rsp+A8h] [rbp-4E8h]
  int v369; // [rsp+A8h] [rbp-4E8h]
  __int64 v370; // [rsp+A8h] [rbp-4E8h]
  int v371; // [rsp+A8h] [rbp-4E8h]
  _QWORD *v372; // [rsp+A8h] [rbp-4E8h]
  __int64 v373; // [rsp+A8h] [rbp-4E8h]
  __int64 v374; // [rsp+A8h] [rbp-4E8h]
  unsigned int v375; // [rsp+A8h] [rbp-4E8h]
  __int64 *v376; // [rsp+A8h] [rbp-4E8h]
  __int64 v377; // [rsp+B0h] [rbp-4E0h]
  __int64 v378; // [rsp+B0h] [rbp-4E0h]
  __int64 v379; // [rsp+B0h] [rbp-4E0h]
  __int64 v380; // [rsp+B0h] [rbp-4E0h]
  __int64 v381; // [rsp+B0h] [rbp-4E0h]
  __int64 v382; // [rsp+B0h] [rbp-4E0h]
  __int64 v383; // [rsp+B8h] [rbp-4D8h]
  __int64 v384; // [rsp+B8h] [rbp-4D8h]
  __int64 v385; // [rsp+B8h] [rbp-4D8h]
  __int64 v386; // [rsp+E0h] [rbp-4B0h] BYREF
  __int64 v387; // [rsp+E8h] [rbp-4A8h]
  __int64 v388; // [rsp+F0h] [rbp-4A0h]
  __int64 v389; // [rsp+100h] [rbp-490h] BYREF
  _QWORD *v390; // [rsp+108h] [rbp-488h]
  _QWORD *v391; // [rsp+110h] [rbp-480h]
  __int64 v392; // [rsp+118h] [rbp-478h]
  int v393; // [rsp+120h] [rbp-470h]
  _QWORD v394[8]; // [rsp+128h] [rbp-468h] BYREF
  const __m128i *v395; // [rsp+168h] [rbp-428h] BYREF
  const __m128i *v396; // [rsp+170h] [rbp-420h]
  __int64 v397; // [rsp+178h] [rbp-418h]
  _QWORD v398[16]; // [rsp+180h] [rbp-410h] BYREF
  _QWORD v399[2]; // [rsp+200h] [rbp-390h] BYREF
  unsigned __int64 v400; // [rsp+210h] [rbp-380h]
  char v401[64]; // [rsp+228h] [rbp-368h] BYREF
  __m128i *v402; // [rsp+268h] [rbp-328h]
  __m128i *v403; // [rsp+270h] [rbp-320h]
  __int8 *v404; // [rsp+278h] [rbp-318h]
  unsigned __int64 *v405; // [rsp+280h] [rbp-310h] BYREF
  __int64 v406; // [rsp+288h] [rbp-308h]
  unsigned __int64 v407[2]; // [rsp+290h] [rbp-300h] BYREF
  __int64 *v408; // [rsp+2A0h] [rbp-2F0h]
  __int64 v409; // [rsp+2A8h] [rbp-2E8h] BYREF
  __int64 v410; // [rsp+2B0h] [rbp-2E0h] BYREF
  __m128i v411; // [rsp+2C0h] [rbp-2D0h] BYREF
  __int64 v412; // [rsp+2D0h] [rbp-2C0h]
  __m128i *v413; // [rsp+2E8h] [rbp-2A8h]
  __m128i *v414; // [rsp+2F0h] [rbp-2A0h]
  __int8 *v415; // [rsp+2F8h] [rbp-298h]
  __m128i v416; // [rsp+300h] [rbp-290h] BYREF
  unsigned __int64 v417; // [rsp+310h] [rbp-280h] BYREF
  __int64 v418; // [rsp+318h] [rbp-278h]
  _QWORD *v419; // [rsp+320h] [rbp-270h] BYREF
  unsigned __int64 **v420; // [rsp+328h] [rbp-268h] BYREF
  _QWORD v421[2]; // [rsp+330h] [rbp-260h] BYREF
  __m128i v422; // [rsp+340h] [rbp-250h]
  __int64 v423; // [rsp+350h] [rbp-240h]
  __m128i *v424; // [rsp+368h] [rbp-228h]
  __m128i *v425; // [rsp+370h] [rbp-220h]
  __int8 *v426; // [rsp+378h] [rbp-218h]
  __m128i v427; // [rsp+380h] [rbp-210h] BYREF
  unsigned __int64 v428; // [rsp+390h] [rbp-200h]
  __int64 v429; // [rsp+398h] [rbp-1F8h]
  __int64 v430; // [rsp+3A0h] [rbp-1F0h]
  char v431[48]; // [rsp+3A8h] [rbp-1E8h] BYREF
  _QWORD *v432; // [rsp+3D8h] [rbp-1B8h]
  unsigned int v433; // [rsp+3E0h] [rbp-1B0h]
  _QWORD v434[2]; // [rsp+3E8h] [rbp-1A8h] BYREF
  __int8 *v435; // [rsp+3F8h] [rbp-198h]

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 232);
  *(_BYTE *)(a1 + 328) = 0;
  v4 = *(_QWORD *)(a1 + 320);
  v358 = 0;
  v5 = *(_QWORD *)(a1 + 248);
  if ( (*(_BYTE *)(v3 + 800) & 2) != 0 )
  {
    v6 = *(__int64 (**)())(*(_QWORD *)v4 + 1336LL);
    if ( v6 != sub_1D46000 )
    {
      v234 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v6)(v4, *(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 240));
      v5 = *(_QWORD *)(a1 + 248);
      v4 = *(_QWORD *)(a1 + 320);
      v358 = (_QWORD *)v234;
    }
  }
  sub_1D4A810((__int64)a2, v4, v5);
  v7 = a2[10];
  v386 = 0;
  v387 = 0;
  v388 = 0;
  if ( v7 )
    v7 -= 24;
  memset(v398, 0, sizeof(v398));
  LODWORD(v398[3]) = 8;
  v398[1] = &v398[5];
  v398[2] = &v398[5];
  v390 = v394;
  v391 = v394;
  v394[0] = v7;
  v395 = 0;
  v396 = 0;
  v397 = 0;
  v392 = 0x100000008LL;
  v393 = 0;
  v389 = 1;
  v8 = sub_157EBA0(v7);
  v427.m128i_i64[0] = v7;
  v427.m128i_i64[1] = v8;
  v353 = &v427;
  LODWORD(v428) = 0;
  sub_136D560(&v395, 0, &v427);
  sub_136D710((__int64)&v389);
  v9 = (const __m128i *)&v420;
  v10 = &v416;
  sub_16CCCB0(&v416, (__int64)&v420, (__int64)v398);
  v12 = (const __m128i *)v398[14];
  v13 = (const __m128i *)v398[13];
  v424 = 0;
  v425 = 0;
  v426 = 0;
  if ( v398[14] == v398[13] )
  {
    v14 = 0;
    v16 = 0;
  }
  else
  {
    v14 = v398[14] - v398[13];
    if ( v398[14] - v398[13] > 0x7FFFFFFFFFFFFFF8uLL )
      goto LABEL_555;
    v15 = sub_22077B0(v398[14] - v398[13]);
    v12 = (const __m128i *)v398[14];
    v13 = (const __m128i *)v398[13];
    v16 = (__m128i *)v15;
  }
  v424 = v16;
  v425 = v16;
  v426 = &v16->m128i_i8[v14];
  if ( v13 != v12 )
  {
    v17 = v16;
    v18 = v13;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1].m128i_i64[0] = v18[1].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 24);
      v17 = (__m128i *)((char *)v17 + 24);
    }
    while ( v18 != v12 );
    v16 = (__m128i *)((char *)v16 + 8 * ((unsigned __int64)((char *)&v18[-2].m128i_u64[1] - (char *)v13) >> 3) + 24);
  }
  v425 = v16;
  sub_16CCEE0(&v427, (__int64)v431, 8, (__int64)&v416);
  v19 = v424;
  v10 = (__m128i *)v399;
  v424 = 0;
  v434[0] = v19;
  v20 = v425;
  v425 = 0;
  v434[1] = v20;
  v21 = v426;
  v426 = 0;
  v435 = v21;
  sub_16CCCB0(v399, (__int64)v401, (__int64)&v389);
  v9 = v396;
  v22 = v395;
  v402 = 0;
  v403 = 0;
  v404 = 0;
  if ( v396 != v395 )
  {
    v23 = (char *)v396 - (char *)v395;
    if ( (unsigned __int64)((char *)v396 - (char *)v395) <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v24 = sub_22077B0((char *)v396 - (char *)v395);
      v9 = v396;
      v22 = v395;
      v25 = (__m128i *)v24;
      goto LABEL_18;
    }
LABEL_555:
    sub_4261EA(v10, v9, v11);
  }
  v23 = 0;
  v25 = 0;
LABEL_18:
  v402 = v25;
  v403 = v25;
  v404 = &v25->m128i_i8[v23];
  if ( v9 != v22 )
  {
    v26 = v25;
    v27 = v22;
    do
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(v27);
        v26[1].m128i_i64[0] = v27[1].m128i_i64[0];
      }
      v27 = (const __m128i *)((char *)v27 + 24);
      v26 = (__m128i *)((char *)v26 + 24);
    }
    while ( v27 != v9 );
    v25 = (__m128i *)((char *)v25 + 8 * ((unsigned __int64)((char *)&v27[-2].m128i_u64[1] - (char *)v22) >> 3) + 24);
  }
  v403 = v25;
  sub_16CCEE0(&v405, (__int64)&v409, 8, (__int64)v399);
  v28 = v402;
  v402 = 0;
  v413 = v28;
  v29 = v403;
  v403 = 0;
  v414 = v29;
  v30 = v404;
  v404 = 0;
  v415 = v30;
  sub_136DA30((__int64)&v405, (__int64)&v427, (__int64)&v386);
  if ( v413 )
    j_j___libc_free_0(v413, v415 - (__int8 *)v413);
  if ( v407[0] != v406 )
    _libc_free(v407[0]);
  if ( v402 )
    j_j___libc_free_0(v402, v404 - (__int8 *)v402);
  if ( v400 != v399[1] )
    _libc_free(v400);
  if ( v434[0] )
    j_j___libc_free_0(v434[0], &v435[-v434[0]]);
  if ( v428 != v427.m128i_i64[1] )
    _libc_free(v428);
  if ( v424 )
    j_j___libc_free_0(v424, v426 - (__int8 *)v424);
  if ( v417 != v416.m128i_i64[1] )
    _libc_free(v417);
  if ( v395 )
    j_j___libc_free_0(v395, v397 - (_QWORD)v395);
  if ( v391 != v390 )
    _libc_free((unsigned __int64)v391);
  if ( v398[13] )
    j_j___libc_free_0(v398[13], v398[15] - v398[13]);
  if ( v398[2] != v398[1] )
    _libc_free(v398[2]);
  v31 = *(_QWORD *)(v2 + 248);
  v32 = a2[10];
  v33 = *(_DWORD *)(v31 + 72);
  if ( v32 )
    v32 -= 24;
  if ( v33 )
  {
    v34 = *(_QWORD *)(v31 + 56);
    v35 = (v33 - 1) & (((unsigned int)v32 >> 4) ^ ((unsigned int)v32 >> 9));
    v36 = (__int64 *)(v34 + 16 * v35);
    v37 = *v36;
    if ( v32 == *v36 )
    {
LABEL_52:
      v38 = v36[1];
      goto LABEL_53;
    }
    v257 = 1;
    v258 = 0;
    while ( v37 != -8 )
    {
      if ( v37 == -16 && !v258 )
        v258 = v36;
      LODWORD(v35) = (v33 - 1) & (v257 + v35);
      v36 = (__int64 *)(v34 + 16LL * (unsigned int)v35);
      v37 = *v36;
      if ( v32 == *v36 )
        goto LABEL_52;
      ++v257;
    }
    v259 = *(_DWORD *)(v31 + 64);
    if ( !v258 )
      v258 = v36;
    ++*(_QWORD *)(v31 + 48);
    v260 = v259 + 1;
    if ( 4 * (v259 + 1) < 3 * v33 )
    {
      if ( v33 - *(_DWORD *)(v31 + 68) - v260 <= v33 >> 3 )
      {
        sub_1D52F30(v31 + 48, v33);
        v276 = *(_DWORD *)(v31 + 72);
        if ( !v276 )
          goto LABEL_598;
        v277 = v276 - 1;
        v278 = *(_QWORD *)(v31 + 56);
        v279 = 1;
        v280 = v277 & (((unsigned int)v32 >> 4) ^ ((unsigned int)v32 >> 9));
        v260 = *(_DWORD *)(v31 + 64) + 1;
        v281 = 0;
        v258 = (__int64 *)(v278 + 16LL * v280);
        v282 = *v258;
        if ( v32 != *v258 )
        {
          while ( v282 != -8 )
          {
            if ( !v281 && v282 == -16 )
              v281 = v258;
            v280 = v277 & (v279 + v280);
            v258 = (__int64 *)(v278 + 16LL * v280);
            v282 = *v258;
            if ( v32 == *v258 )
              goto LABEL_479;
            ++v279;
          }
          if ( v281 )
            v258 = v281;
        }
      }
      goto LABEL_479;
    }
  }
  else
  {
    ++*(_QWORD *)(v31 + 48);
  }
  sub_1D52F30(v31 + 48, 2 * v33);
  v269 = *(_DWORD *)(v31 + 72);
  if ( !v269 )
    goto LABEL_598;
  v270 = v269 - 1;
  v271 = *(_QWORD *)(v31 + 56);
  v272 = v270 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
  v260 = *(_DWORD *)(v31 + 64) + 1;
  v258 = (__int64 *)(v271 + 16LL * v272);
  v273 = *v258;
  if ( v32 != *v258 )
  {
    v274 = 1;
    v275 = 0;
    while ( v273 != -8 )
    {
      if ( !v275 && v273 == -16 )
        v275 = v258;
      v272 = v270 & (v274 + v272);
      v258 = (__int64 *)(v271 + 16LL * v272);
      v273 = *v258;
      if ( v32 == *v258 )
        goto LABEL_479;
      ++v274;
    }
    if ( v275 )
      v258 = v275;
  }
LABEL_479:
  *(_DWORD *)(v31 + 64) = v260;
  if ( *v258 != -8 )
    --*(_DWORD *)(v31 + 68);
  *v258 = v32;
  v258[1] = 0;
  v38 = 0;
  v31 = *(_QWORD *)(v2 + 248);
LABEL_53:
  *(_QWORD *)(v31 + 784) = v38;
  *(_QWORD *)(*(_QWORD *)(v2 + 248) + 792LL) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 248) + 784LL) + 32LL);
  *(_QWORD *)(*(_QWORD *)(v2 + 272) + 72LL) = *(_QWORD *)(v2 + 248);
  if ( v358 )
  {
    sub_1FD39C0(v358);
    if ( !(unsigned __int8)sub_1FD47F0(v358) )
    {
      *(_BYTE *)(v2 + 328) = 1;
      v39 = a2[10];
      if ( v39 )
        v39 -= 24;
      v40 = sub_1626D20((__int64)a2);
      sub_15C9150((const char **)&v416, v40);
      sub_15CA540((__int64)&v427, (__int64)"sdagisel", (__int64)"FastISelFailure", 15, &v416, v39);
      sub_15CAB20((__int64)&v427, "FastISel didn't lower all arguments: ", 0x25u);
      sub_15C9730((__int64)&v405, "Prototype", 9, *a2);
      v416.m128i_i64[0] = (__int64)&v417;
      sub_1D46650(v416.m128i_i64, v405, (__int64)v405 + v406);
      v419 = v421;
      sub_1D46650((__int64 *)&v419, v408, (__int64)v408 + v409);
      v41 = _mm_loadu_si128(&v411);
      v423 = v412;
      v422 = v41;
      sub_15CAC60((__int64)&v427, &v416);
      if ( v419 != v421 )
        j_j___libc_free_0(v419, v421[0] + 1LL);
      if ( (unsigned __int64 *)v416.m128i_i64[0] != &v417 )
        j_j___libc_free_0(v416.m128i_i64[0], v417 + 1);
      if ( v408 != &v410 )
        j_j___libc_free_0(v408, v410 + 1);
      if ( v405 != v407 )
        j_j___libc_free_0(v405, v407[0] + 1);
      sub_1D472F0(*(_QWORD *)(v2 + 256), *(_QWORD **)(v2 + 408), (__int64)&v427, dword_4FC1E80 > 1);
      v42 = a2;
      sub_208CF60(v2, a2);
      v43 = *(_QWORD *)(v2 + 272);
      v45 = sub_2051DF0(*(_QWORD *)(v2 + 280));
      v46 = v44;
      if ( v45 )
      {
        nullsub_686();
        v42 = 0;
        *(_QWORD *)(v43 + 176) = v45;
        *(_DWORD *)(v43 + 184) = v46;
        sub_1D23870();
      }
      else
      {
        *(_QWORD *)(v43 + 176) = 0;
        *(_DWORD *)(v43 + 184) = v44;
      }
      sub_20515F0(*(_QWORD *)(v2 + 280), v42);
      sub_1D50350(v2);
      v47 = v432;
      v427.m128i_i64[0] = (__int64)&unk_49ECF68;
      v48 = &v432[11 * v433];
      if ( v432 != v48 )
      {
        do
        {
          v48 -= 11;
          v49 = (_QWORD *)v48[4];
          if ( v49 != v48 + 6 )
            j_j___libc_free_0(v49, v48[6] + 1LL);
          if ( (_QWORD *)*v48 != v48 + 2 )
            j_j___libc_free_0(*v48, v48[2] + 1LL);
        }
        while ( v47 != v48 );
        v48 = v432;
      }
      if ( v48 != v434 )
        _libc_free((unsigned __int64)v48);
    }
    v50 = *(_QWORD *)(v2 + 248);
    v51 = *(_QWORD **)(v50 + 792);
    if ( *(_QWORD **)(*(_QWORD *)(v50 + 784) + 32LL) == v51 )
    {
      v358[19] = 0;
      v358[18] = 0;
    }
    else
    {
      v52 = *v51 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v52 )
LABEL_594:
        BUG();
      if ( (*(_QWORD *)v52 & 4) == 0 && (*(_BYTE *)(v52 + 46) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v52; ; i = *(_QWORD *)v52 )
        {
          v52 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v52 + 46) & 4) == 0 )
            break;
        }
      }
      v358[19] = v52;
      v358[18] = v52;
    }
  }
  else
  {
    sub_208CF60(v2, a2);
  }
  v54 = *(_QWORD *)(v2 + 320);
  v356 = *(_QWORD *)(v2 + 248);
  v55 = *(__int64 (**)())(*(_QWORD *)v54 + 1160LL);
  if ( v55 != sub_1D45FE0 )
  {
    v345 = *(_QWORD *)(v2 + 280);
    v336 = *(_QWORD *)(v2 + 312);
    if ( ((unsigned __int8 (__fastcall *)(__int64))v55)(v54) && *(_DWORD *)(v356 + 192) )
    {
      v235 = sub_1E0A0C0(*(_QWORD *)(v356 + 8));
      v236 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v54 + 288LL);
      v237 = 8 * sub_15A9520(v235, 0);
      if ( v237 == 32 )
      {
        v238 = 5;
      }
      else if ( v237 > 0x20 )
      {
        v238 = 6;
        if ( v237 != 64 )
        {
          v238 = 0;
          if ( v237 == 128 )
            v238 = 7;
        }
      }
      else
      {
        v238 = 3;
        if ( v237 != 8 )
        {
          LOBYTE(v238) = v237 == 16;
          v238 = (unsigned int)(4 * v238);
        }
      }
      v352 = v236 == sub_1D45FB0 ? *(_QWORD *)(v54 + 8 * (v238 & 7) + 120) : v236(v54, v238);
      v239 = *(__int64 **)(v356 + 184);
      v240 = v239;
      v365 = &v239[*(unsigned int *)(v356 + 192)];
      if ( v239 != v365 )
      {
        do
        {
          v241 = *(_QWORD *)(v356 + 176);
          v242 = *v240;
          if ( !v241 || v242 != v241 )
          {
            v243 = sub_1E6B9A0(*(_QWORD *)(*(_QWORD *)(v356 + 8) + 40LL), v352, byte_3F871B3, 0);
            v244 = *(_QWORD *)(v336 + 8) + 576LL;
            if ( *(_QWORD *)v345 )
            {
              v245 = *(_QWORD *)(*(_QWORD *)v345 + 48LL);
              v416.m128i_i64[0] = v245;
              if ( v245 )
                sub_1623A60((__int64)&v416, v245, 2);
            }
            else
            {
              v416.m128i_i64[0] = 0;
            }
            v376 = (__int64 *)sub_1DD5D10(*(_QWORD *)(v356 + 784));
            v382 = *(_QWORD *)(v356 + 784);
            v385 = *(_QWORD *)(v382 + 56);
            v247 = sub_1E0B640(v385, v244, &v416, 0, v246);
            sub_1DD5BA0(v382 + 16, v247);
            v248 = *v376;
            *(_QWORD *)(v247 + 8) = v376;
            *(_QWORD *)v247 = v248 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v247 & 7LL;
            *(_QWORD *)((v248 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v247;
            *v376 = v247 | *v376 & 7;
            v427.m128i_i64[0] = 0x10000000;
            v428 = 0;
            v427.m128i_i32[2] = v243;
            v429 = 0;
            v430 = 0;
            sub_1E1A9C0(v247, v385, &v427);
            if ( v416.m128i_i64[0] )
              sub_161E7C0((__int64)&v416, v416.m128i_i64[0]);
            if ( v358 )
            {
              v249 = **(_QWORD **)(v356 + 792) & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v249 )
                goto LABEL_594;
              if ( (*(_QWORD *)v249 & 4) == 0 && (*(_BYTE *)(v249 + 46) & 4) != 0 )
              {
                for ( j = *(_QWORD *)v249; ; j = *(_QWORD *)v249 )
                {
                  v249 = j & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_BYTE *)(v249 + 46) & 4) == 0 )
                    break;
                }
              }
              v358[19] = v249;
              v358[18] = v249;
            }
            sub_1FE5190(v356, *(_QWORD *)(v356 + 784), v242, v243);
          }
          ++v240;
        }
        while ( v365 != v240 );
      }
    }
    v356 = *(_QWORD *)(v2 + 248);
  }
  v383 = *(_QWORD *)(v356 + 8);
  v377 = sub_1E0A0C0(v383);
  v340 = *(_QWORD *)v356 + 72LL;
  v359 = *(_QWORD *)(*(_QWORD *)v356 + 80LL);
  if ( v359 != v340 )
  {
    v329 = v2;
    do
    {
      if ( !v359 )
        BUG();
      v56 = *(_QWORD *)(v359 + 24);
      if ( v56 != v359 + 16 )
      {
        while ( 1 )
        {
          if ( !v56 )
            goto LABEL_597;
          if ( *(_BYTE *)(v56 - 8) != 78 )
            goto LABEL_92;
          v85 = *(_QWORD *)(v56 - 48);
          if ( *(_BYTE *)(v85 + 16) )
            goto LABEL_92;
          if ( (*(_BYTE *)(v85 + 33) & 0x20) == 0 )
            goto LABEL_92;
          if ( *(_DWORD *)(v85 + 36) != 36 )
            goto LABEL_92;
          v86 = v56 - 24;
          v87 = (__int64 *)sub_1601A30(v56 - 24, 1);
          v88 = (__int64)v87;
          if ( !v87 )
            goto LABEL_92;
          v89 = *v87;
          v90 = 1;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v89 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v95 = *(_QWORD *)(v89 + 32);
                v89 = *(_QWORD *)(v89 + 24);
                v90 *= (_DWORD)v95;
                continue;
              case 1:
                v91 = 16;
                goto LABEL_131;
              case 2:
                v91 = 32;
                goto LABEL_131;
              case 3:
              case 9:
                v91 = 64;
                goto LABEL_131;
              case 4:
                v91 = 80;
                goto LABEL_131;
              case 5:
              case 6:
                v427.m128i_i32[2] = v90 << 7;
                if ( (unsigned int)(v90 << 7) > 0x40 )
                  goto LABEL_139;
                goto LABEL_132;
              case 7:
                v368 = v90;
                v94 = sub_15A9520(v377, 0);
                v90 = v368;
                v91 = 8 * v94;
                goto LABEL_131;
              case 0xB:
                v91 = *(_DWORD *)(v89 + 8) >> 8;
                goto LABEL_131;
              case 0xD:
                v371 = v90;
                v101 = (_QWORD *)sub_15A9930(v377, v89);
                v90 = v371;
                v91 = 8 * *v101;
                goto LABEL_131;
              case 0xE:
                v331 = v90;
                v370 = *(_QWORD *)(v89 + 32);
                v347 = *(_QWORD *)(v89 + 24);
                v97 = sub_15A9FE0(v377, v347);
                v98 = v347;
                v90 = v331;
                v99 = 1;
                v100 = v97;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v98 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v105 = *(_QWORD *)(v98 + 32);
                      v98 = *(_QWORD *)(v98 + 24);
                      v99 *= v105;
                      continue;
                    case 1:
                      v102 = 16;
                      goto LABEL_151;
                    case 2:
                      v102 = 32;
                      goto LABEL_151;
                    case 3:
                    case 9:
                      v102 = 64;
                      goto LABEL_151;
                    case 4:
                      v102 = 80;
                      goto LABEL_151;
                    case 5:
                    case 6:
                      v102 = 128;
                      goto LABEL_151;
                    case 7:
                      v327 = v331;
                      v103 = 0;
                      v332 = v100;
                      v348 = v99;
                      goto LABEL_154;
                    case 0xB:
                      v102 = *(_DWORD *)(v98 + 8) >> 8;
                      goto LABEL_151;
                    case 0xD:
                      JUMPOUT(0x1D55C1B);
                    case 0xE:
                      v315 = v331;
                      v320 = v100;
                      v323 = v99;
                      v333 = *(_QWORD *)(v98 + 24);
                      v328 = *(_QWORD *)(v98 + 32);
                      v106 = sub_15A9FE0(v377, v333);
                      v90 = v315;
                      v349 = 1;
                      v100 = v320;
                      v107 = v333;
                      v326 = v106;
                      v99 = v323;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v107 + 8) )
                        {
                          case 0:
                          case 8:
                          case 0xA:
                          case 0xC:
                          case 0x10:
                            v268 = v349 * *(_QWORD *)(v107 + 32);
                            v107 = *(_QWORD *)(v107 + 24);
                            v349 = v268;
                            continue;
                          case 1:
                            v262 = 16;
                            goto LABEL_488;
                          case 2:
                            v262 = 32;
                            goto LABEL_488;
                          case 3:
                          case 9:
                            v262 = 64;
                            goto LABEL_488;
                          case 4:
                            v262 = 80;
                            goto LABEL_488;
                          case 5:
                          case 6:
                            v262 = 128;
                            goto LABEL_488;
                          case 7:
                            v321 = v315;
                            v263 = 0;
                            v324 = v100;
                            v337 = v99;
                            goto LABEL_492;
                          case 0xB:
                            v262 = *(_DWORD *)(v107 + 8) >> 8;
                            goto LABEL_488;
                          case 0xD:
                            v267 = (_QWORD *)sub_15A9930(v377, v107);
                            v99 = v323;
                            v100 = v320;
                            v90 = v315;
                            v262 = 8LL * *v267;
                            goto LABEL_488;
                          case 0xE:
                            v306 = v315;
                            v308 = v320;
                            v311 = v323;
                            v316 = *(_QWORD *)(v107 + 24);
                            v325 = *(_QWORD *)(v107 + 32);
                            v265 = sub_15A9FE0(v377, v316);
                            v90 = v306;
                            v338 = 1;
                            v100 = v320;
                            v99 = v311;
                            v322 = v265;
                            v266 = v316;
                            while ( 2 )
                            {
                              switch ( *(_BYTE *)(v266 + 8) )
                              {
                                case 0:
                                case 8:
                                case 0xA:
                                case 0xC:
                                case 0x10:
                                  v299 = v338 * *(_QWORD *)(v266 + 32);
                                  v266 = *(_QWORD *)(v266 + 24);
                                  v338 = v299;
                                  continue;
                                case 1:
                                  v298 = 16;
                                  goto LABEL_574;
                                case 2:
                                  v298 = 32;
                                  goto LABEL_574;
                                case 3:
                                case 9:
                                  v298 = 64;
                                  goto LABEL_574;
                                case 4:
                                  v298 = 80;
                                  goto LABEL_574;
                                case 5:
                                case 6:
                                  v298 = 128;
                                  goto LABEL_574;
                                case 7:
                                  v309 = v306;
                                  v300 = 0;
                                  v312 = v100;
                                  v317 = v99;
                                  goto LABEL_577;
                                case 0xB:
                                  v298 = *(_DWORD *)(v266 + 8) >> 8;
                                  goto LABEL_574;
                                case 0xD:
                                  v314 = v100;
                                  v319 = v99;
                                  v303 = (_QWORD *)sub_15A9930(v377, v266);
                                  v99 = v319;
                                  v100 = v314;
                                  v90 = v306;
                                  v298 = 8LL * *v303;
                                  goto LABEL_574;
                                case 0xE:
                                  v304 = v306;
                                  v305 = v308;
                                  v307 = v311;
                                  v310 = *(_QWORD *)(v266 + 24);
                                  v318 = *(_QWORD *)(v266 + 32);
                                  v313 = (unsigned int)sub_15A9FE0(v377, v310);
                                  v302 = sub_127FA20(v377, v310);
                                  v99 = v307;
                                  v100 = v305;
                                  v90 = v304;
                                  v298 = 8 * v318 * v313 * ((v313 + ((unsigned __int64)(v302 + 7) >> 3) - 1) / v313);
                                  goto LABEL_574;
                                case 0xF:
                                  v309 = v306;
                                  v312 = v100;
                                  v317 = v99;
                                  v300 = *(_DWORD *)(v266 + 8) >> 8;
LABEL_577:
                                  v301 = sub_15A9520(v377, v300);
                                  v99 = v317;
                                  v100 = v312;
                                  v90 = v309;
                                  v298 = (unsigned int)(8 * v301);
LABEL_574:
                                  v262 = 8
                                       * v322
                                       * v325
                                       * ((v322 + ((unsigned __int64)(v338 * v298 + 7) >> 3) - 1)
                                        / v322);
                                  break;
                              }
                              goto LABEL_488;
                            }
                          case 0xF:
                            v321 = v315;
                            v324 = v100;
                            v337 = v99;
                            v263 = *(_DWORD *)(v107 + 8) >> 8;
LABEL_492:
                            v264 = sub_15A9520(v377, v263);
                            v99 = v337;
                            v100 = v324;
                            v90 = v321;
                            v262 = (unsigned int)(8 * v264);
LABEL_488:
                            v102 = 8 * v326 * v328 * ((v326 + ((unsigned __int64)(v349 * v262 + 7) >> 3) - 1) / v326);
                            break;
                        }
                        goto LABEL_151;
                      }
                    case 0xF:
                      v327 = v331;
                      v332 = v100;
                      v348 = v99;
                      v103 = *(_DWORD *)(v98 + 8) >> 8;
LABEL_154:
                      v104 = sub_15A9520(v377, v103);
                      v99 = v348;
                      v100 = v332;
                      v90 = v327;
                      v102 = (unsigned int)(8 * v104);
LABEL_151:
                      v91 = 8 * v370 * v100 * ((v100 + ((v102 * v99 + 7) >> 3) - 1) / v100);
                      break;
                  }
                  goto LABEL_131;
                }
              case 0xF:
                v369 = v90;
                v96 = sub_15A9520(v377, *(_DWORD *)(v89 + 8) >> 8);
                v90 = v369;
                v91 = 8 * v96;
LABEL_131:
                v427.m128i_i32[2] = v91 * v90;
                if ( (unsigned int)(v91 * v90) > 0x40 )
LABEL_139:
                  sub_16A4EF0((__int64)&v427, 0, 0);
                else
LABEL_132:
                  v427.m128i_i64[0] = 0;
                v92 = sub_164A410(v88, v377, (__int64)&v427);
                v93 = *(_BYTE *)(v92 + 16);
                if ( v93 <= 0x17u )
                {
                  if ( v93 != 17 )
                    goto LABEL_135;
                  v108 = sub_1FDEA40(v356, v92);
                }
                else
                {
                  if ( v93 != 53 )
                    goto LABEL_135;
                  v115 = *(unsigned int *)(v356 + 360);
                  if ( !(_DWORD)v115 )
                    goto LABEL_135;
                  v116 = *(_QWORD *)(v356 + 344);
                  v117 = (v115 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
                  v118 = (__int64 *)(v116 + 16LL * v117);
                  v119 = *v118;
                  if ( v92 != *v118 )
                  {
                    v227 = 1;
                    while ( v119 != -8 )
                    {
                      v228 = v227 + 1;
                      v117 = (v115 - 1) & (v117 + v227);
                      v118 = (__int64 *)(v116 + 16LL * v117);
                      v119 = *v118;
                      if ( v92 == *v118 )
                        goto LABEL_178;
                      v227 = v228;
                    }
LABEL_135:
                    if ( v427.m128i_i32[2] <= 0x40u )
                      goto LABEL_92;
LABEL_136:
                    if ( v427.m128i_i64[0] )
                      j_j___libc_free_0_0(v427.m128i_i64[0]);
                    goto LABEL_92;
                  }
LABEL_178:
                  if ( v118 == (__int64 *)(v116 + 16 * v115) )
                    goto LABEL_135;
                  v108 = *((_DWORD *)v118 + 2);
                }
                if ( v108 == 0x7FFFFFFF )
                  goto LABEL_135;
                v372 = *(_QWORD **)(*(_QWORD *)(v86 + 24 * (2LL - (*(_DWORD *)(v56 - 4) & 0xFFFFFFF))) + 24LL);
                if ( v427.m128i_i32[2] > 0x40u )
                {
                  v350 = v427.m128i_i32[2];
                  if ( v350 == (unsigned int)sub_16A57B0((__int64)&v427) )
                    goto LABEL_170;
                  v109 = *(_QWORD *)v427.m128i_i64[0];
LABEL_169:
                  v372 = (_QWORD *)sub_15C48E0(v372, 0, v109, 0, 0);
                  goto LABEL_170;
                }
                v109 = v427.m128i_i64[0];
                if ( v427.m128i_i64[0] )
                  goto LABEL_169;
LABEL_170:
                v110 = sub_15C70A0(v56 + 24);
                v112 = *(_QWORD *)(*(_QWORD *)(v86 + 24 * (1LL - (*(_DWORD *)(v56 - 4) & 0xFFFFFFF))) + 24LL);
                v113 = *(_DWORD *)(v383 + 616);
                if ( v113 >= *(_DWORD *)(v383 + 620) )
                {
                  v351 = v110;
                  sub_16CD150(v383 + 608, (const void *)(v383 + 624), 0, 32, v110, v111);
                  v110 = v351;
                  v113 = *(_DWORD *)(v383 + 616);
                }
                v114 = *(_QWORD *)(v383 + 608) + 32LL * v113;
                if ( v114 )
                {
                  *(_QWORD *)v114 = v112;
                  *(_DWORD *)(v114 + 16) = v108;
                  *(_QWORD *)(v114 + 8) = v372;
                  *(_QWORD *)(v114 + 24) = v110;
                  v113 = *(_DWORD *)(v383 + 616);
                }
                *(_DWORD *)(v383 + 616) = v113 + 1;
                if ( v427.m128i_i32[2] > 0x40u )
                  goto LABEL_136;
LABEL_92:
                v56 = *(_QWORD *)(v56 + 8);
                if ( v359 + 16 == v56 )
                  goto LABEL_93;
                break;
            }
            break;
          }
        }
      }
LABEL_93:
      v359 = *(_QWORD *)(v359 + 8);
    }
    while ( v340 != v359 );
    v2 = v329;
  }
  v57 = v2;
  v346 = v2 + 336;
  v384 = v387;
  v357 = v386;
  if ( v387 == v386 )
    goto LABEL_321;
  while ( 2 )
  {
    v75 = *(_QWORD *)(v384 - 8);
    if ( *(_DWORD *)(v57 + 304) )
    {
      v149 = *(_QWORD *)(v75 + 8);
      if ( !v149 )
        goto LABEL_307;
      while ( 1 )
      {
        v150 = sub_1648700(v149);
        if ( (unsigned __int8)(*((_BYTE *)v150 + 16) - 25) <= 9u )
          break;
        v149 = *(_QWORD *)(v149 + 8);
        if ( !v149 )
          goto LABEL_307;
      }
      v379 = v75;
LABEL_264:
      v154 = *(_QWORD *)(v57 + 248);
      v155 = v150[5];
      v156 = *(_QWORD **)(v154 + 848);
      v152 = *(_QWORD **)(v154 + 840);
      if ( v156 == v152 )
      {
        v157 = &v152[*(unsigned int *)(v154 + 860)];
        if ( v152 == v157 )
        {
          v151 = *(_QWORD **)(v154 + 840);
        }
        else
        {
          do
          {
            if ( v155 == *v152 )
              break;
            ++v152;
          }
          while ( v157 != v152 );
          v151 = v157;
        }
      }
      else
      {
        v151 = &v156[*(unsigned int *)(v154 + 856)];
        v152 = sub_16CC9F0(v154 + 832, v155);
        if ( v155 == *v152 )
        {
          v182 = *(_QWORD *)(v154 + 848);
          if ( v182 == *(_QWORD *)(v154 + 840) )
            v183 = *(unsigned int *)(v154 + 860);
          else
            v183 = *(unsigned int *)(v154 + 856);
          v157 = (_QWORD *)(v182 + 8 * v183);
        }
        else
        {
          v153 = *(_QWORD *)(v154 + 848);
          if ( v153 != *(_QWORD *)(v154 + 840) )
          {
            v152 = (_QWORD *)(v153 + 8LL * *(unsigned int *)(v154 + 856));
            goto LABEL_261;
          }
          v157 = (_QWORD *)(v153 + 8LL * *(unsigned int *)(v154 + 860));
          v152 = v157;
        }
      }
      if ( v152 != v157 )
      {
        do
        {
          if ( *v152 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_261;
          ++v152;
        }
        while ( v157 != v152 );
        if ( v152 == v151 )
        {
LABEL_274:
          v75 = v379;
          v158 = sub_157F280(v379);
          v160 = v159;
          if ( v158 == v159 )
            goto LABEL_318;
          while ( 1 )
          {
            v163 = *(_QWORD *)(v57 + 248);
            v164 = *(unsigned int *)(v163 + 232);
            if ( (_DWORD)v164 )
            {
              v165 = *(_QWORD *)(v163 + 216);
              v166 = (v164 - 1) & (((unsigned int)v158 >> 9) ^ ((unsigned int)v158 >> 4));
              v167 = (__int64 *)(v165 + 16LL * v166);
              v168 = *v167;
              if ( v158 != *v167 )
              {
                v225 = 1;
                while ( v168 != -8 )
                {
                  v226 = v225 + 1;
                  v166 = (v164 - 1) & (v166 + v225);
                  v167 = (__int64 *)(v165 + 16LL * v166);
                  v168 = *v167;
                  if ( v158 == *v167 )
                    goto LABEL_285;
                  v225 = v226;
                }
                goto LABEL_278;
              }
LABEL_285:
              if ( v167 != (__int64 *)(v165 + 16 * v164) )
              {
                v169 = *((_DWORD *)v167 + 2);
                if ( v169 )
                  break;
              }
            }
LABEL_278:
            if ( !v158 )
              BUG();
            v162 = *(_QWORD *)(v158 + 32);
            if ( !v162 )
LABEL_597:
              BUG();
            v158 = 0;
            if ( *(_BYTE *)(v162 - 8) == 77 )
              v158 = v162 - 24;
            if ( v160 == v158 )
              goto LABEL_318;
          }
          v170 = v169 & 0x7FFFFFFF;
          v171 = *(unsigned int *)(v163 + 952);
          v172 = v170 + 1;
          if ( (int)v170 + 1 > (unsigned int)v171 )
          {
            v173 = v172;
            if ( v172 < v171 )
            {
              v161 = *(_QWORD *)(v163 + 944);
              v229 = v161 + 40 * v171;
              if ( v229 != v161 + 40LL * v172 )
              {
                v363 = v170;
                v344 = v170 + 1;
                v380 = *(_QWORD *)(v57 + 248);
                v230 = v161 + 40LL * v172;
                v374 = v158;
                v231 = v229;
                do
                {
                  v231 -= 40;
                  if ( *(_DWORD *)(v231 + 32) > 0x40u )
                  {
                    v232 = *(_QWORD *)(v231 + 24);
                    if ( v232 )
                      j_j___libc_free_0_0(v232);
                  }
                  if ( *(_DWORD *)(v231 + 16) > 0x40u )
                  {
                    v233 = *(_QWORD *)(v231 + 8);
                    if ( v233 )
                      j_j___libc_free_0_0(v233);
                  }
                }
                while ( v230 != v231 );
                v158 = v374;
                v170 = v363;
                *(_DWORD *)(v380 + 952) = v344;
                v161 = *(_QWORD *)(v380 + 944);
                goto LABEL_277;
              }
              goto LABEL_302;
            }
            if ( v172 > v171 )
            {
              if ( v172 > (unsigned __int64)*(unsigned int *)(v163 + 956) )
              {
                v364 = v170 + 1;
                v375 = v170;
                v381 = v172;
                sub_1D4FA80(v163 + 944, v172);
                v171 = *(unsigned int *)(v163 + 952);
                v172 = v364;
                v170 = v375;
                v173 = v381;
              }
              v161 = *(_QWORD *)(v163 + 944);
              v174 = v161 + 40 * v173;
              v175 = v161 + 40 * v171;
              if ( v174 != v175 )
              {
                v373 = v158;
                v176 = v174;
                v334 = v160;
                v177 = v175;
                v361 = v170;
                v342 = v172;
                while ( 1 )
                {
LABEL_297:
                  if ( !v177 )
                    goto LABEL_296;
                  *(_DWORD *)v177 = *(_DWORD *)(v163 + 960);
                  v179 = *(_DWORD *)(v163 + 976);
                  *(_DWORD *)(v177 + 16) = v179;
                  if ( v179 > 0x40 )
                    break;
                  *(_QWORD *)(v177 + 8) = *(_QWORD *)(v163 + 968);
                  v178 = *(_DWORD *)(v163 + 992);
                  *(_DWORD *)(v177 + 32) = v178;
                  if ( v178 <= 0x40 )
                    goto LABEL_295;
LABEL_300:
                  v181 = v177 + 24;
                  v177 += 40;
                  sub_16A4FD0(v181, (const void **)(v163 + 984));
                  if ( v176 == v177 )
                  {
LABEL_301:
                    v158 = v373;
                    v170 = v361;
                    v172 = v342;
                    v160 = v334;
                    v161 = *(_QWORD *)(v163 + 944);
                    goto LABEL_302;
                  }
                }
                sub_16A4FD0(v177 + 8, (const void **)(v163 + 968));
                v180 = *(_DWORD *)(v163 + 992);
                *(_DWORD *)(v177 + 32) = v180;
                if ( v180 > 0x40 )
                  goto LABEL_300;
LABEL_295:
                *(_QWORD *)(v177 + 24) = *(_QWORD *)(v163 + 984);
LABEL_296:
                v177 += 40;
                if ( v176 == v177 )
                  goto LABEL_301;
                goto LABEL_297;
              }
LABEL_302:
              *(_DWORD *)(v163 + 952) = v172;
              goto LABEL_277;
            }
          }
          v161 = *(_QWORD *)(v163 + 944);
LABEL_277:
          *(_BYTE *)(v161 + 40 * v170 + 3) &= ~0x80u;
          goto LABEL_278;
        }
LABEL_262:
        while ( 1 )
        {
          v149 = *(_QWORD *)(v149 + 8);
          if ( !v149 )
            break;
          v150 = sub_1648700(v149);
          if ( (unsigned __int8)(*((_BYTE *)v150 + 16) - 25) <= 9u )
            goto LABEL_264;
        }
        v75 = v379;
LABEL_307:
        v184 = sub_157F280(v75);
        v186 = v185;
        v187 = v184;
        if ( v184 != v185 )
        {
          while ( 1 )
          {
            sub_1FE0410(*(_QWORD *)(v57 + 248), v187);
            if ( !v187 )
              goto LABEL_315;
            v188 = *(_QWORD *)(v187 + 32);
            if ( !v188 )
              goto LABEL_597;
            if ( *(_BYTE *)(v188 - 8) != 77 )
              break;
            v187 = v188 - 24;
            if ( v186 == v187 )
              goto LABEL_318;
          }
          if ( v186 )
          {
            sub_1FE0410(*(_QWORD *)(v57 + 248), 0);
LABEL_315:
            BUG();
          }
        }
LABEL_318:
        v189 = *(_QWORD *)(v57 + 248);
        v190 = *(__int64 **)(v189 + 840);
        if ( *(__int64 **)(v189 + 848) == v190 )
        {
          v196 = &v190[*(unsigned int *)(v189 + 860)];
          v197 = *(_DWORD *)(v189 + 860);
          if ( v190 != v196 )
          {
            v198 = 0;
            while ( v75 != *v190 )
            {
              if ( *v190 == -2 )
                v198 = v190;
              if ( v196 == ++v190 )
              {
                if ( !v198 )
                  goto LABEL_471;
                *v198 = v75;
                --*(_DWORD *)(v189 + 864);
                ++*(_QWORD *)(v189 + 832);
                goto LABEL_117;
              }
            }
            goto LABEL_117;
          }
LABEL_471:
          if ( v197 < *(_DWORD *)(v189 + 856) )
          {
            *(_DWORD *)(v189 + 860) = v197 + 1;
            *v196 = v75;
            ++*(_QWORD *)(v189 + 832);
            goto LABEL_117;
          }
        }
        sub_16CCBA0(v189 + 832, v75);
        goto LABEL_117;
      }
LABEL_261:
      if ( v152 != v151 )
        goto LABEL_262;
      goto LABEL_274;
    }
LABEL_117:
    v76 = sub_157ED20(v75);
    v31 = *(_QWORD *)(v57 + 248);
    v367 = v76;
    v77 = *(_DWORD *)(v31 + 72);
    if ( !v77 )
    {
      ++*(_QWORD *)(v31 + 48);
LABEL_119:
      sub_1D52F30(v31 + 48, 2 * v77);
      v78 = *(_DWORD *)(v31 + 72);
      if ( !v78 )
        goto LABEL_598;
      v79 = v78 - 1;
      v80 = *(_QWORD *)(v31 + 56);
      v81 = v79 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
      v82 = *(_DWORD *)(v31 + 64) + 1;
      v83 = (__int64 *)(v80 + 16LL * v81);
      v84 = *v83;
      if ( v75 != *v83 )
      {
        v290 = 1;
        v291 = 0;
        while ( v84 != -8 )
        {
          if ( v84 == -16 && !v291 )
            v291 = v83;
          v81 = v79 & (v290 + v81);
          v83 = (__int64 *)(v80 + 16LL * v81);
          v84 = *v83;
          if ( v75 == *v83 )
            goto LABEL_121;
          ++v290;
        }
        if ( v291 )
          v83 = v291;
      }
      goto LABEL_121;
    }
    v58 = *(_QWORD *)(v31 + 56);
    v59 = (v77 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
    v60 = (__int64 *)(v58 + 16LL * v59);
    v61 = *v60;
    if ( v75 == *v60 )
    {
      v62 = v60[1];
      goto LABEL_99;
    }
    v199 = 1;
    v83 = 0;
    while ( v61 != -8 )
    {
      if ( v83 || v61 != -16 )
        v60 = v83;
      v59 = (v77 - 1) & (v199 + v59);
      v261 = (__int64 *)(v58 + 16LL * v59);
      v61 = *v261;
      if ( v75 == *v261 )
      {
        v62 = v261[1];
        goto LABEL_99;
      }
      ++v199;
      v83 = v60;
      v60 = (__int64 *)(v58 + 16LL * v59);
    }
    if ( !v83 )
      v83 = v60;
    v200 = *(_DWORD *)(v31 + 64);
    ++*(_QWORD *)(v31 + 48);
    v82 = v200 + 1;
    if ( 4 * v82 >= 3 * v77 )
      goto LABEL_119;
    if ( v77 - *(_DWORD *)(v31 + 68) - v82 <= v77 >> 3 )
    {
      sub_1D52F30(v31 + 48, v77);
      v201 = *(_DWORD *)(v31 + 72);
      if ( v201 )
      {
        v202 = v201 - 1;
        v203 = *(_QWORD *)(v31 + 56);
        v204 = 0;
        v205 = v202 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v206 = 1;
        v82 = *(_DWORD *)(v31 + 64) + 1;
        v83 = (__int64 *)(v203 + 16LL * v205);
        v207 = *v83;
        if ( v75 != *v83 )
        {
          while ( v207 != -8 )
          {
            if ( !v204 && v207 == -16 )
              v204 = v83;
            v205 = v202 & (v206 + v205);
            v83 = (__int64 *)(v203 + 16LL * v205);
            v207 = *v83;
            if ( v75 == *v83 )
              goto LABEL_121;
            ++v206;
          }
          if ( v204 )
            v83 = v204;
        }
        goto LABEL_121;
      }
LABEL_598:
      ++*(_DWORD *)(v31 + 64);
      BUG();
    }
LABEL_121:
    *(_DWORD *)(v31 + 64) = v82;
    if ( *v83 != -8 )
      --*(_DWORD *)(v31 + 68);
    *v83 = v75;
    v62 = 0;
    v83[1] = 0;
    v31 = *(_QWORD *)(v57 + 248);
LABEL_99:
    *(_QWORD *)(v31 + 784) = v62;
    v63 = *(_QWORD *)(v57 + 248);
    v64 = *(_QWORD *)(v63 + 784);
    if ( !v64 )
      goto LABEL_115;
    *(_QWORD *)(v63 + 792) = v64 + 24;
    *(_DWORD *)(*(_QWORD *)(v57 + 248) + 932LL) = 0;
    *(_DWORD *)(*(_QWORD *)(v57 + 248) + 936LL) = 0;
    v65 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v75) + 16) - 34;
    if ( (unsigned int)v65 <= 0x36 )
    {
      v66 = 0x40018000000001LL;
      if ( _bittest64(&v66, v65) )
      {
        if ( !(unsigned __int8)sub_1D52720(v57) )
          goto LABEL_115;
      }
    }
    v67 = (_QWORD *)(v75 + 40);
    v378 = v367 + 24;
    if ( !v358 )
    {
      if ( (_QWORD *)v378 != v67 )
      {
        sub_1D50960(v57, v378, v75 + 40, v353);
        if ( v427.m128i_i8[0] )
        {
          v68 = *(_QWORD *)(v57 + 248);
          v69 = *(_QWORD *)(v68 + 784) + 24LL;
          if ( *(_QWORD *)(v68 + 792) != v69 )
            sub_1FD3B40(0, *(_QWORD *)(v68 + 792), v69);
        }
      }
      goto LABEL_107;
    }
    v120 = a2[10];
    if ( v120 )
      v120 -= 24;
    if ( v75 != v120 )
      sub_1FD39C0(v358);
    v121 = *(unsigned __int8 (**)(void))(**(_QWORD **)(v57 + 320) + 1160LL);
    if ( (char *)v121 != (char *)sub_1D45FE0 )
    {
      v211 = *(_QWORD *)(v57 + 248);
      if ( v121() )
      {
        if ( *(_DWORD *)(v211 + 192) )
        {
          v212 = (_QWORD *)(v367 + 24);
          if ( (_QWORD *)v378 == v67 )
            goto LABEL_405;
          v335 = v75;
          v362 = (_QWORD *)(v75 + 40);
          v330 = v57;
          while ( 1 )
          {
            v416.m128i_i64[0] = 0;
            v214 = *((_BYTE *)v212 - 8);
            v215 = (__int64)(v212 - 3);
            if ( v214 > 0x17u )
            {
              switch ( v214 )
              {
                case 0x4Eu:
                  v218 = v215 | 4;
LABEL_386:
                  v416.m128i_i64[0] = v218;
                  if ( (v218 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                  {
                    v219 = sub_1D4AD40(&v416);
                    if ( (__int64 *)v219 != v220 )
                    {
                      v343 = v212;
                      v221 = 0;
                      v222 = (__int64 *)v219;
                      v223 = v220;
                      do
                      {
                        if ( (unsigned __int8)sub_1649A90(*v222) )
                        {
                          v221 = *v222;
                          sub_1FE6270(v211, v215, *(_QWORD *)(v211 + 784), *v222);
                        }
                        v222 += 3;
                      }
                      while ( v223 != v222 );
                      v212 = v343;
                      if ( v221 )
                      {
                        v224 = sub_1FE5EB0(v211, v215);
                        sub_1FE5190(v211, *(_QWORD *)(v211 + 784), v221, v224);
                      }
                    }
                  }
                  goto LABEL_378;
                case 0x1Du:
                  v218 = v215 & 0xFFFFFFFFFFFFFFFBLL;
                  goto LABEL_386;
                case 0x36u:
                  v213 = *(v212 - 6);
                  if ( (unsigned __int8)sub_1649A90(v213) )
                    sub_1FE6270(v211, v212 - 3, *(_QWORD *)(v211 + 784), v213);
                  goto LABEL_378;
              }
            }
            if ( v214 == 55 )
            {
              v216 = *(v212 - 6);
              if ( (unsigned __int8)sub_1649A90(v216) )
              {
                v217 = sub_1FE5EB0(v211, v212 - 3);
                sub_1FE5190(v211, *(_QWORD *)(v211 + 784), v216, v217);
                v212 = (_QWORD *)v212[1];
                if ( v362 == v212 )
                {
LABEL_384:
                  v67 = v362;
                  v57 = v330;
                  v122 = (__int64)v353;
                  v341 = v335;
                  goto LABEL_196;
                }
                goto LABEL_379;
              }
            }
            else if ( v214 == 25 )
            {
              v427.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v212[2] + 56LL) + 112LL);
              if ( (unsigned __int8)sub_1560490(v353, 54, 0) )
                sub_1FE6270(v211, v212 - 3, *(_QWORD *)(v211 + 784), *(_QWORD *)(v211 + 176));
            }
LABEL_378:
            v212 = (_QWORD *)v212[1];
            if ( v362 == v212 )
              goto LABEL_384;
LABEL_379:
            if ( !v212 )
            {
              v416.m128i_i64[0] = 0;
              goto LABEL_597;
            }
          }
        }
      }
    }
    if ( (_QWORD *)v378 == v67 )
      goto LABEL_405;
    v341 = v75;
    v122 = (__int64)v353;
LABEL_196:
    while ( 2 )
    {
      v126 = 0;
      if ( (*v67 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        v126 = (_QWORD *)((*v67 & 0xFFFFFFFFFFFFFFF8LL) - 24);
      if ( (unsigned __int8)sub_1D47920((__int64)v126, *(_QWORD *)(v57 + 248)) )
        goto LABEL_195;
      v127 = *(_QWORD **)(v57 + 352);
      v124 = *(_QWORD **)(v57 + 344);
      if ( v127 == v124 )
      {
        v123 = &v124[*(unsigned int *)(v57 + 364)];
        if ( v124 == v123 )
        {
          v256 = *(_QWORD **)(v57 + 344);
        }
        else
        {
          do
          {
            if ( v126 == (_QWORD *)*v124 )
              break;
            ++v124;
          }
          while ( v123 != v124 );
          v256 = v123;
        }
      }
      else
      {
        v123 = &v127[*(unsigned int *)(v57 + 360)];
        v124 = sub_16CC9F0(v346, (__int64)v126);
        if ( v126 == (_QWORD *)*v124 )
        {
          v144 = *(_QWORD *)(v57 + 352);
          if ( v144 == *(_QWORD *)(v57 + 344) )
            v145 = *(unsigned int *)(v57 + 364);
          else
            v145 = *(unsigned int *)(v57 + 360);
          v256 = (_QWORD *)(v144 + 8 * v145);
        }
        else
        {
          v125 = *(_QWORD *)(v57 + 352);
          if ( v125 != *(_QWORD *)(v57 + 344) )
          {
            v124 = (_QWORD *)(v125 + 8LL * *(unsigned int *)(v57 + 360));
            goto LABEL_194;
          }
          v124 = (_QWORD *)(v125 + 8LL * *(unsigned int *)(v57 + 364));
          v256 = v124;
        }
      }
      if ( v124 != v256 )
      {
        while ( *v124 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( v256 == ++v124 )
          {
            if ( v124 != v123 )
              goto LABEL_195;
            goto LABEL_209;
          }
        }
      }
LABEL_194:
      if ( v124 != v123 )
        goto LABEL_195;
LABEL_209:
      sub_1FD3A30(v358);
      if ( (unsigned __int8)sub_1FDD7A0(v358, v126) )
      {
        v208 = (__int64)v126;
        do
        {
          if ( v367 == v208 )
            break;
          if ( !v208 )
            goto LABEL_593;
          v209 = (*(_QWORD *)(v208 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0;
          v208 = (*(_QWORD *)(v208 + 24) & 0xFFFFFFFFFFFFFFF8LL) - 24;
          if ( v209 )
            v208 = 0;
        }
        while ( (unsigned __int8)sub_1D47920(v208, *(_QWORD *)(v57 + 248)) );
        if ( v126 != (_QWORD *)v208 && *(_BYTE *)(v208 + 16) == 54 )
        {
          v210 = *(_QWORD *)(v208 + 8);
          if ( v210 )
          {
            if ( !*(_QWORD *)(v210 + 8) && (unsigned __int8)sub_1FDCE70(v358, v208, v126) )
              v67 = *(_QWORD **)(v208 + 32);
          }
        }
        goto LABEL_195;
      }
      *(_BYTE *)(v57 + 328) = 1;
      if ( *((_BYTE *)v126 + 16) != 78 || sub_1642D30((__int64)v126) || sub_1642DB0((__int64)v126) )
      {
        v353 = (__m128i *)v122;
        v251 = (__int64)v126;
        sub_15C9090((__int64)&v416, v126 + 6);
        sub_15CA540((__int64)v353, (__int64)"sdagisel", (__int64)"FastISelFailure", 15, &v416, v341);
        if ( (unsigned int)*((unsigned __int8 *)v126 + 16) - 25 > 9 )
        {
          v252 = "FastISel missed";
          v253 = dword_4FC1E80 != 0;
          sub_15CAB20((__int64)v353, "FastISel missed", 0xFu);
        }
        else
        {
          v252 = "FastISel missed terminator";
          sub_15CAB20((__int64)v353, "FastISel missed terminator", 0x1Au);
          v253 = dword_4FC1E80 > 2;
        }
        if ( (unsigned __int8)sub_15C8060((__int64)v353, (__int64)v252) || dword_4FC1E80 )
        {
          v406 = 0;
          v405 = v407;
          v416.m128i_i64[0] = (__int64)&unk_49EFBE0;
          LOBYTE(v407[0]) = 0;
          v420 = &v405;
          LODWORD(v419) = 1;
          v418 = 0;
          v417 = 0;
          v416.m128i_i64[1] = 0;
          sub_155C2B0(v251, (__int64)&v416, 0);
          sub_15CAB20((__int64)v353, ": ", 2u);
          if ( v418 != v416.m128i_i64[1] )
            sub_16E7BA0(v416.m128i_i64);
          sub_15CAB20((__int64)v353, *v420, (size_t)v420[1]);
          sub_16E7BC0(v416.m128i_i64);
          if ( v405 != v407 )
            j_j___libc_free_0(v405, v407[0] + 1);
        }
        sub_1D472F0(*(_QWORD *)(v57 + 256), *(_QWORD **)(v57 + 408), (__int64)v353, v253);
        v254 = v432;
        v427.m128i_i64[0] = (__int64)&unk_49ECF68;
        v140 = &v432[11 * v433];
        if ( v432 != v140 )
        {
          do
          {
            v140 -= 11;
            v255 = (_QWORD *)v140[4];
            if ( v255 != v140 + 6 )
              j_j___libc_free_0(v255, v140[6] + 1LL);
            if ( (_QWORD *)*v140 != v140 + 2 )
              j_j___libc_free_0(*v140, v140[2] + 1LL);
          }
          while ( v254 != v140 );
LABEL_233:
          v140 = v432;
        }
LABEL_234:
        if ( v140 != v434 )
          _libc_free((unsigned __int64)v140);
        sub_1FD3A30(v358);
        if ( (_QWORD *)v378 != v67 )
        {
          sub_1D50960(v57, v378, (__int64)v67, v353);
          if ( v427.m128i_i8[0] )
          {
            v142 = *(_QWORD *)(v57 + 248);
            v143 = *(_QWORD *)(v142 + 784) + 24LL;
            if ( *(_QWORD *)(v142 + 792) != v143 )
              sub_1FD3B40(v358, *(_QWORD *)(v142 + 792), v143);
          }
        }
        goto LABEL_240;
      }
      sub_15C9090((__int64)&v416, v126 + 6);
      sub_15CA540(v122, (__int64)"sdagisel", (__int64)"FastISelFailure", 15, &v416, v341);
      sub_15CAB20(v122, "FastISel missed call", 0x14u);
      if ( (unsigned __int8)sub_15C8060(v122, (__int64)"FastISel missed call") || (v128 = dword_4FC1E80) != 0 )
      {
        v406 = 0;
        v405 = v407;
        v416.m128i_i64[0] = (__int64)&unk_49EFBE0;
        LOBYTE(v407[0]) = 0;
        v420 = &v405;
        LODWORD(v419) = 1;
        v418 = 0;
        v417 = 0;
        v416.m128i_i64[1] = 0;
        sub_155C2B0((__int64)v126, (__int64)&v416, 0);
        sub_15CAB20(v122, ": ", 2u);
        if ( v418 != v416.m128i_i64[1] )
          sub_16E7BA0(v416.m128i_i64);
        sub_15CAB20(v122, *v420, (size_t)v420[1]);
        sub_16E7BC0(v416.m128i_i64);
        if ( v405 != v407 )
          j_j___libc_free_0(v405, v407[0] + 1);
        v128 = dword_4FC1E80;
      }
      sub_1D472F0(*(_QWORD *)(v57 + 256), *(_QWORD **)(v57 + 408), v122, v128 > 2);
      v129 = *(_BYTE *)(*v126 + 8LL);
      if ( v129 == 10 || !v129 )
      {
LABEL_254:
        v131 = *(_QWORD *)(v57 + 248);
        goto LABEL_226;
      }
      v130 = *(_QWORD *)(v57 + 248);
      v131 = v130;
      if ( !v126[1] )
        goto LABEL_226;
      v132 = *(_DWORD *)(v130 + 232);
      v133 = v130 + 208;
      if ( !v132 )
      {
        ++*(_QWORD *)(v130 + 208);
        goto LABEL_528;
      }
      v134 = *(_QWORD *)(v130 + 216);
      v360 = ((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4);
      v135 = (v132 - 1) & v360;
      v136 = v134 + 16LL * v135;
      v137 = *(_QWORD **)v136;
      if ( v126 == *(_QWORD **)v136 )
        goto LABEL_225;
      v354 = 1;
      v146 = 0;
      while ( 1 )
      {
        if ( v137 == (_QWORD *)-8LL )
        {
          v147 = *(_DWORD *)(v130 + 224);
          if ( v146 )
            v136 = v146;
          ++*(_QWORD *)(v130 + 208);
          v148 = v147 + 1;
          if ( 4 * v148 < 3 * v132 )
          {
            if ( v132 - *(_DWORD *)(v130 + 228) - v148 > v132 >> 3 )
            {
LABEL_250:
              *(_DWORD *)(v130 + 224) = v148;
              if ( *(_QWORD *)v136 != -8 )
                --*(_DWORD *)(v130 + 228);
              *(_QWORD *)v136 = v126;
              *(_DWORD *)(v136 + 8) = 0;
              v130 = *(_QWORD *)(v57 + 248);
LABEL_253:
              *(_DWORD *)(v136 + 8) = sub_1FDE000(v130, *v126);
              goto LABEL_254;
            }
            v355 = v130;
            sub_1542080(v133, v132);
            v130 = v355;
            v283 = *(_DWORD *)(v355 + 232);
            if ( v283 )
            {
              v284 = v283 - 1;
              v285 = 1;
              v286 = 0;
              v287 = *(_QWORD *)(v355 + 216);
              v288 = v284 & v360;
              v148 = *(_DWORD *)(v355 + 224) + 1;
              v136 = v287 + 16LL * (v284 & v360);
              v289 = *(_QWORD **)v136;
              if ( v126 == *(_QWORD **)v136 )
                goto LABEL_250;
              while ( v289 != (_QWORD *)-8LL )
              {
                if ( !v286 && v289 == (_QWORD *)-16LL )
                  v286 = v136;
                v288 = v284 & (v285 + v288);
                v136 = v287 + 16LL * v288;
                v289 = *(_QWORD **)v136;
                if ( v126 == *(_QWORD **)v136 )
                  goto LABEL_250;
                ++v285;
              }
LABEL_519:
              if ( v286 )
                v136 = v286;
              goto LABEL_250;
            }
            goto LABEL_592;
          }
LABEL_528:
          v366 = v130;
          sub_1542080(v133, 2 * v132);
          v130 = v366;
          v292 = *(_DWORD *)(v366 + 232);
          if ( v292 )
          {
            v293 = v292 - 1;
            v294 = *(_QWORD *)(v366 + 216);
            v295 = (v292 - 1) & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
            v148 = *(_DWORD *)(v366 + 224) + 1;
            v136 = v294 + 16LL * v295;
            v296 = *(_QWORD **)v136;
            if ( v126 == *(_QWORD **)v136 )
              goto LABEL_250;
            v297 = 1;
            v286 = 0;
            while ( v296 != (_QWORD *)-8LL )
            {
              if ( v296 == (_QWORD *)-16LL && !v286 )
                v286 = v136;
              v295 = v293 & (v297 + v295);
              v136 = v294 + 16LL * v295;
              v296 = *(_QWORD **)v136;
              if ( v126 == *(_QWORD **)v136 )
                goto LABEL_250;
              ++v297;
            }
            goto LABEL_519;
          }
LABEL_592:
          ++*(_DWORD *)(v130 + 224);
LABEL_593:
          BUG();
        }
        if ( v137 == (_QWORD *)-16LL && !v146 )
          v146 = v136;
        v135 = (v132 - 1) & (v354 + v135);
        v136 = v134 + 16LL * v135;
        v137 = *(_QWORD **)v136;
        if ( v126 == *(_QWORD **)v136 )
          break;
        ++v354;
      }
      v131 = *(_QWORD *)(v57 + 248);
LABEL_225:
      if ( !*(_DWORD *)(v136 + 8) )
        goto LABEL_253;
LABEL_226:
      v416.m128i_i8[0] = 0;
      v138 = *(_QWORD *)(v131 + 792);
      sub_1D50960(v57, (__int64)(v126 + 3), (__int64)v67, &v416);
      if ( v416.m128i_i8[0] )
      {
        v353 = (__m128i *)v122;
        sub_1FD3B40(v358, v138, *(_QWORD *)(*(_QWORD *)(v57 + 248) + 784LL) + 24LL);
        v139 = v432;
        v67 = (_QWORD *)(*v67 & 0xFFFFFFFFFFFFFFF8LL);
        v427.m128i_i64[0] = (__int64)&unk_49ECF68;
        v140 = &v432[11 * v433];
        if ( v432 != v140 )
        {
          do
          {
            v140 -= 11;
            v141 = (_QWORD *)v140[4];
            if ( v141 != v140 + 6 )
              j_j___libc_free_0(v141, v140[6] + 1LL);
            if ( (_QWORD *)*v140 != v140 + 2 )
              j_j___libc_free_0(*v140, v140[2] + 1LL);
          }
          while ( v139 != v140 );
          goto LABEL_233;
        }
        goto LABEL_234;
      }
      v193 = v432;
      v427.m128i_i64[0] = (__int64)&unk_49ECF68;
      v194 = &v432[11 * v433];
      if ( v432 != v194 )
      {
        do
        {
          v194 -= 11;
          v195 = (_QWORD *)v194[4];
          if ( v195 != v194 + 6 )
            j_j___libc_free_0(v195, v194[6] + 1LL);
          if ( (_QWORD *)*v194 != v194 + 2 )
            j_j___libc_free_0(*v194, v194[2] + 1LL);
        }
        while ( v193 != v194 );
        v194 = v432;
      }
      if ( v194 != v434 )
        _libc_free((unsigned __int64)v194);
LABEL_195:
      v67 = (_QWORD *)(*v67 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)v378 != v67 )
        continue;
      break;
    }
    v353 = (__m128i *)v122;
LABEL_405:
    sub_1FD3A30(v358);
LABEL_240:
    sub_1FD5CB0(v358);
LABEL_107:
    sub_1D50BB0((_QWORD *)v57);
    v70 = *(_QWORD *)(v57 + 248);
    v71 = *(_QWORD *)(v70 + 904);
    if ( v71 != *(_QWORD *)(v70 + 912) )
      *(_QWORD *)(v70 + 912) = v71;
    ++*(_QWORD *)(v57 + 336);
    v72 = *(void **)(v57 + 352);
    if ( v72 == *(void **)(v57 + 344) )
    {
LABEL_114:
      *(_QWORD *)(v57 + 364) = 0;
    }
    else
    {
      v73 = 4 * (*(_DWORD *)(v57 + 364) - *(_DWORD *)(v57 + 368));
      v74 = *(unsigned int *)(v57 + 360);
      if ( v73 < 0x20 )
        v73 = 32;
      if ( v73 >= (unsigned int)v74 )
      {
        memset(v72, -1, 8 * v74);
        goto LABEL_114;
      }
      sub_16CC920(v346);
    }
LABEL_115:
    v384 -= 8;
    if ( v357 != v384 )
      continue;
    break;
  }
  v2 = v57;
LABEL_321:
  sub_1D53D30(*(_QWORD *)(v2 + 248));
  if ( v358 )
    (*(void (__fastcall **)(_QWORD *))(*v358 + 8LL))(v358);
  sub_2051990(*(_QWORD *)(v2 + 280));
  result = *(_QWORD *)(v2 + 280);
  v192 = v386;
  *(_QWORD *)(result + 672) = 0;
  if ( v192 )
    return j_j___libc_free_0(v192, v388 - v192);
  return result;
}
