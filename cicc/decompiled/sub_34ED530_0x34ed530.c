// Function: sub_34ED530
// Address: 0x34ed530
//
__int64 __fastcall sub_34ED530(__m128i *a1, _QWORD *a2)
{
  _QWORD *m128i_i64; // r13
  _QWORD *v3; // r12
  __int64 (*v4)(); // rdx
  __int64 v5; // rax
  __int64 (*v6)(); // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // r9
  __int64 v23; // r14
  __int64 v24; // rbx
  _QWORD *v25; // r8
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // rax
  int v28; // eax
  bool v29; // zf
  bool v30; // sf
  _QWORD *v31; // r15
  __int64 v32; // r12
  __int64 *v33; // r10
  __int64 v34; // r14
  char v35; // al
  char v36; // al
  __int64 v37; // rdi
  __int64 (*v38)(); // r11
  char v39; // al
  int v40; // edx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  char v43; // al
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rbx
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // r14
  __int64 v58; // r13
  __int64 v59; // rax
  void *v60; // r9
  __int64 v61; // rcx
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // r10
  __int64 v64; // r8
  unsigned __int64 v65; // rdx
  __int64 v66; // rdi
  __int64 (*v67)(); // rax
  int v68; // eax
  int v69; // edx
  __int64 v70; // rdi
  char v71; // al
  unsigned __int64 *v72; // rbx
  unsigned __int64 *v73; // r15
  _QWORD *v74; // r14
  __int64 v75; // r13
  __int64 v76; // rbx
  unsigned __int64 *v77; // rax
  unsigned __int64 *v78; // r12
  __int64 v79; // rcx
  unsigned __int64 *v80; // rbx
  unsigned __int64 *v81; // r14
  unsigned __int64 *v82; // rax
  unsigned __int64 v83; // rdx
  unsigned __int64 *v84; // rax
  unsigned __int64 v85; // rdx
  unsigned __int64 v86; // rdi
  unsigned __int64 *v87; // rbx
  __int64 v88; // rax
  unsigned __int64 v89; // r14
  unsigned __int64 *v90; // rdx
  char *v91; // rbx
  char v92; // al
  bool v93; // r10
  unsigned int v94; // r13d
  unsigned int v95; // ecx
  unsigned int v96; // r9d
  void *v97; // r11
  __int64 v98; // rsi
  __int64 v99; // r8
  __int64 v100; // rax
  __int64 v101; // r15
  __int64 v102; // r15
  __int64 v103; // r8
  __int64 v104; // rcx
  size_t v105; // r12
  __int64 v106; // rax
  __int64 v107; // r9
  __int64 v108; // rax
  char v109; // r12
  size_t v110; // r9
  int v111; // eax
  unsigned __int64 *v112; // r12
  __int64 v113; // rbx
  unsigned __int64 *v114; // r14
  __int64 v115; // r12
  __int64 v116; // rbx
  __int64 v117; // r14
  unsigned __int64 v118; // rdi
  unsigned __int64 v119; // rdi
  int v120; // eax
  __int64 v121; // rbx
  unsigned __int64 *v122; // r12
  unsigned int v123; // r14d
  _BYTE *v125; // rdi
  unsigned int v126; // r12d
  __int64 v127; // rdx
  __int64 v128; // rdi
  __int64 (*v129)(); // rax
  char v130; // r12
  unsigned int v131; // r13d
  __int64 v132; // rax
  unsigned __int64 v133; // rdi
  __int64 v134; // rdx
  __int64 v135; // rdi
  __int64 (*v136)(); // rax
  char v137; // r12
  __int64 v138; // rax
  unsigned __int64 v139; // rdi
  __int64 v140; // rdx
  __int64 v141; // rdi
  __int64 (*v142)(); // rax
  char v143; // r12
  __int64 v144; // rax
  unsigned __int64 v145; // rdi
  __int64 v146; // rdx
  __int64 v147; // rdi
  __int64 (*v148)(); // rax
  char v149; // r12
  __int64 v150; // rax
  unsigned __int64 v151; // rdi
  char v152; // al
  unsigned int v153; // esi
  unsigned int v154; // ecx
  __int64 v155; // rdx
  __int64 v156; // rdi
  char v157; // al
  char v158; // dl
  __int64 v159; // rdi
  __int64 v160; // r8
  __int64 v161; // rsi
  __int64 v162; // rcx
  __int64 v163; // rax
  __int64 v164; // r13
  __int64 v165; // r13
  void *v166; // r11
  __int64 v167; // rax
  size_t v168; // r12
  __int64 v169; // rax
  __int64 v170; // r9
  __int64 v171; // r9
  char v172; // r15
  char v173; // r12
  __int64 v174; // r15
  __int64 v175; // r13
  char *v176; // r13
  unsigned __int64 v177; // rax
  unsigned int v178; // r9d
  bool v179; // r10
  unsigned int v180; // ecx
  __int64 v181; // rsi
  char v182; // al
  __int64 v183; // rsi
  __int64 v184; // rdx
  __int64 *v185; // rax
  __int64 *v186; // rcx
  __int64 v187; // rdx
  char v188; // al
  void *v189; // r11
  unsigned __int64 v190; // rdx
  int v191; // r13d
  _QWORD *v192; // rdi
  __int64 v193; // rdi
  __int64 (*v194)(); // rax
  char v195; // al
  __int64 v196; // rdx
  __int64 v197; // rdi
  __int64 (*v198)(); // rax
  char v199; // r14
  __int64 v200; // rax
  unsigned __int64 v201; // rdi
  __int64 v202; // rdx
  __int64 v203; // rdi
  __int64 (*v204)(); // rax
  __int64 v205; // rax
  unsigned __int64 v206; // rdi
  char v207; // r13
  unsigned __int8 v208; // dl
  char v209; // si
  unsigned int v210; // edx
  char v211; // si
  __int64 v212; // r12
  __int64 v213; // r15
  char v214; // al
  __int64 v215; // r11
  __int64 v216; // rdx
  __int64 v217; // rax
  unsigned __int64 v218; // rax
  __int64 v219; // rcx
  __int64 v220; // r8
  __int64 v221; // r9
  __int64 v222; // r11
  __int64 v223; // rdx
  __int64 (*v224)(); // rax
  unsigned int v225; // eax
  __int64 v226; // rax
  char v227; // r12
  bool v228; // r12
  __int64 v229; // rax
  unsigned __int64 v230; // rdi
  _QWORD *v231; // rsi
  __int64 *v232; // rdi
  _QWORD *v233; // rdi
  __int64 v234; // r8
  __int64 v235; // rcx
  __int64 v236; // r12
  __int64 v237; // rdi
  __int64 (*v238)(); // rax
  char v239; // al
  __int64 v240; // rax
  unsigned int v241; // r13d
  unsigned int v242; // eax
  __int64 v243; // rax
  unsigned __int64 v244; // rdi
  bool v245; // al
  bool v246; // r10
  char v247; // r9
  __int64 v248; // rax
  __int64 v249; // rbx
  unsigned __int64 v250; // r12
  __int64 v251; // rsi
  __int64 v252; // r12
  __int64 v253; // rdi
  __int64 (*v254)(); // rax
  char v255; // al
  __int64 v256; // rax
  unsigned int v257; // ecx
  unsigned int v258; // eax
  __int64 v259; // rax
  unsigned __int64 v260; // rdi
  unsigned int v261; // eax
  bool v262; // r10
  __int64 v263; // r9
  bool v264; // r10
  __int64 v265; // rax
  void *v266; // r11
  size_t v267; // r9
  __int64 v268; // rax
  __int64 v269; // rdi
  __int64 (*v270)(); // rax
  __int64 v271; // r11
  __int64 v272; // rax
  _QWORD *v273; // rax
  int v274; // r8d
  __int64 v275; // r11
  bool v276; // r10
  _QWORD *v277; // r9
  __int64 v278; // rcx
  __int64 v279; // r8
  __int64 v280; // r9
  bool v281; // r10
  bool v282; // al
  bool v283; // r10
  __int64 v284; // rax
  __int64 v285; // rbx
  unsigned __int64 v286; // r12
  __int64 v287; // rsi
  __int64 v288; // rbx
  __int64 v289; // r12
  unsigned __int64 v290; // rdi
  unsigned __int64 v291; // rdi
  char v292; // al
  __int64 v293; // rax
  __int64 i; // rsi
  _BYTE *v295; // rdx
  char v296; // al
  unsigned __int64 v297; // r15
  unsigned __int64 v298; // rax
  bool v299; // al
  __int64 v300; // rsi
  _QWORD *v301; // rdi
  char v302; // r12
  __int64 v303; // rax
  __int64 v304; // rdx
  bool v305; // cf
  unsigned __int64 v306; // rax
  unsigned __int64 v307; // rbx
  __int64 v308; // rax
  __int64 v309; // rcx
  char *v310; // rax
  char *v311; // rdx
  unsigned __int16 *v312; // r15
  __int64 v313; // r14
  __int64 v314; // rdx
  __int64 v315; // rax
  unsigned __int16 *v316; // r14
  unsigned __int64 v317; // rdi
  unsigned __int64 v318; // rdi
  __int64 v319; // [rsp-8h] [rbp-668h]
  __int64 v320; // [rsp+0h] [rbp-660h]
  bool v322; // [rsp+18h] [rbp-648h]
  bool v323; // [rsp+20h] [rbp-640h]
  bool v324; // [rsp+20h] [rbp-640h]
  __int64 v325; // [rsp+28h] [rbp-638h]
  __int64 v326; // [rsp+30h] [rbp-630h]
  unsigned __int8 v327; // [rsp+3Eh] [rbp-622h]
  bool v328; // [rsp+3Fh] [rbp-621h]
  _QWORD *v329; // [rsp+40h] [rbp-620h]
  unsigned __int64 v330; // [rsp+48h] [rbp-618h]
  unsigned int v331; // [rsp+4Ch] [rbp-614h]
  int v332; // [rsp+50h] [rbp-610h]
  unsigned int v333; // [rsp+50h] [rbp-610h]
  size_t v334; // [rsp+70h] [rbp-5F0h]
  __int64 v335; // [rsp+78h] [rbp-5E8h]
  bool v336; // [rsp+78h] [rbp-5E8h]
  bool v337; // [rsp+78h] [rbp-5E8h]
  bool v338; // [rsp+78h] [rbp-5E8h]
  _DWORD *v339; // [rsp+80h] [rbp-5E0h]
  void *v340; // [rsp+80h] [rbp-5E0h]
  bool v341; // [rsp+88h] [rbp-5D8h]
  char v342; // [rsp+88h] [rbp-5D8h]
  __int64 v343; // [rsp+88h] [rbp-5D8h]
  bool v344; // [rsp+88h] [rbp-5D8h]
  bool v345; // [rsp+88h] [rbp-5D8h]
  bool v346; // [rsp+90h] [rbp-5D0h]
  char v347; // [rsp+90h] [rbp-5D0h]
  char v348; // [rsp+90h] [rbp-5D0h]
  bool v349; // [rsp+90h] [rbp-5D0h]
  bool v350; // [rsp+90h] [rbp-5D0h]
  bool v351; // [rsp+90h] [rbp-5D0h]
  __int64 v352; // [rsp+90h] [rbp-5D0h]
  unsigned int v353; // [rsp+90h] [rbp-5D0h]
  char v354; // [rsp+90h] [rbp-5D0h]
  bool v355; // [rsp+98h] [rbp-5C8h]
  bool v356; // [rsp+98h] [rbp-5C8h]
  void *v357; // [rsp+98h] [rbp-5C8h]
  void *v358; // [rsp+98h] [rbp-5C8h]
  int v359; // [rsp+98h] [rbp-5C8h]
  void *v360; // [rsp+98h] [rbp-5C8h]
  int v361; // [rsp+98h] [rbp-5C8h]
  bool v362; // [rsp+98h] [rbp-5C8h]
  bool v363; // [rsp+98h] [rbp-5C8h]
  void *v364; // [rsp+98h] [rbp-5C8h]
  bool v365; // [rsp+98h] [rbp-5C8h]
  bool v366; // [rsp+98h] [rbp-5C8h]
  unsigned int v367; // [rsp+A0h] [rbp-5C0h]
  bool v368; // [rsp+A0h] [rbp-5C0h]
  __int64 v369; // [rsp+A0h] [rbp-5C0h]
  bool v370; // [rsp+A0h] [rbp-5C0h]
  int v371; // [rsp+A0h] [rbp-5C0h]
  bool v372; // [rsp+A0h] [rbp-5C0h]
  bool v373; // [rsp+A0h] [rbp-5C0h]
  size_t v374; // [rsp+A0h] [rbp-5C0h]
  bool v375; // [rsp+A0h] [rbp-5C0h]
  bool v376; // [rsp+A0h] [rbp-5C0h]
  __int64 v377; // [rsp+A0h] [rbp-5C0h]
  __int64 v378; // [rsp+A0h] [rbp-5C0h]
  __int64 v379; // [rsp+A0h] [rbp-5C0h]
  unsigned int v380; // [rsp+A0h] [rbp-5C0h]
  bool v381; // [rsp+A0h] [rbp-5C0h]
  bool v382; // [rsp+A0h] [rbp-5C0h]
  __int64 v383; // [rsp+A0h] [rbp-5C0h]
  char n; // [rsp+A8h] [rbp-5B8h]
  size_t nk; // [rsp+A8h] [rbp-5B8h]
  unsigned int na; // [rsp+A8h] [rbp-5B8h]
  unsigned int nb; // [rsp+A8h] [rbp-5B8h]
  unsigned int nc; // [rsp+A8h] [rbp-5B8h]
  unsigned int nd; // [rsp+A8h] [rbp-5B8h]
  unsigned int nl; // [rsp+A8h] [rbp-5B8h]
  size_t nm; // [rsp+A8h] [rbp-5B8h]
  bool ne; // [rsp+A8h] [rbp-5B8h]
  size_t nn; // [rsp+A8h] [rbp-5B8h]
  size_t no; // [rsp+A8h] [rbp-5B8h]
  size_t np; // [rsp+A8h] [rbp-5B8h]
  unsigned int nq; // [rsp+A8h] [rbp-5B8h]
  size_t nr; // [rsp+A8h] [rbp-5B8h]
  size_t ns; // [rsp+A8h] [rbp-5B8h]
  size_t nf; // [rsp+A8h] [rbp-5B8h]
  int nu; // [rsp+A8h] [rbp-5B8h]
  unsigned int ng; // [rsp+A8h] [rbp-5B8h]
  bool nh; // [rsp+A8h] [rbp-5B8h]
  size_t ni; // [rsp+A8h] [rbp-5B8h]
  size_t nt; // [rsp+A8h] [rbp-5B8h]
  int nj; // [rsp+A8h] [rbp-5B8h]
  unsigned int v406; // [rsp+B0h] [rbp-5B0h]
  bool v407; // [rsp+B0h] [rbp-5B0h]
  unsigned __int64 v408; // [rsp+B0h] [rbp-5B0h]
  unsigned int src; // [rsp+B8h] [rbp-5A8h]
  void *srcg; // [rsp+B8h] [rbp-5A8h]
  bool srch; // [rsp+B8h] [rbp-5A8h]
  void *srci; // [rsp+B8h] [rbp-5A8h]
  char *srcj; // [rsp+B8h] [rbp-5A8h]
  unsigned __int64 *srca; // [rsp+B8h] [rbp-5A8h]
  bool srck; // [rsp+B8h] [rbp-5A8h]
  bool srcl; // [rsp+B8h] [rbp-5A8h]
  char *srcm; // [rsp+B8h] [rbp-5A8h]
  unsigned int srcn; // [rsp+B8h] [rbp-5A8h]
  bool srcb; // [rsp+B8h] [rbp-5A8h]
  bool srco; // [rsp+B8h] [rbp-5A8h]
  void *srcp; // [rsp+B8h] [rbp-5A8h]
  unsigned int srcc; // [rsp+B8h] [rbp-5A8h]
  void *srcd; // [rsp+B8h] [rbp-5A8h]
  void *srcq; // [rsp+B8h] [rbp-5A8h]
  void *srce; // [rsp+B8h] [rbp-5A8h]
  void *srcf; // [rsp+B8h] [rbp-5A8h]
  bool srcr; // [rsp+B8h] [rbp-5A8h]
  bool srcs; // [rsp+B8h] [rbp-5A8h]
  __int64 v429; // [rsp+C0h] [rbp-5A0h]
  bool v430; // [rsp+C0h] [rbp-5A0h]
  unsigned __int64 v431; // [rsp+C0h] [rbp-5A0h]
  unsigned __int64 v432; // [rsp+C0h] [rbp-5A0h]
  __int64 *v433; // [rsp+C0h] [rbp-5A0h]
  __int64 *v434; // [rsp+C0h] [rbp-5A0h]
  __int64 *v435; // [rsp+C0h] [rbp-5A0h]
  __int64 v436; // [rsp+C0h] [rbp-5A0h]
  __int64 v437; // [rsp+C0h] [rbp-5A0h]
  _BYTE *v438; // [rsp+C0h] [rbp-5A0h]
  _QWORD *v439; // [rsp+C0h] [rbp-5A0h]
  unsigned __int16 *v440; // [rsp+C0h] [rbp-5A0h]
  char v441; // [rsp+C8h] [rbp-598h]
  char v442; // [rsp+C8h] [rbp-598h]
  __int64 *v443; // [rsp+C8h] [rbp-598h]
  __int64 *v444; // [rsp+C8h] [rbp-598h]
  __int64 v445; // [rsp+C8h] [rbp-598h]
  __int64 v446; // [rsp+C8h] [rbp-598h]
  unsigned int v447; // [rsp+C8h] [rbp-598h]
  char v448; // [rsp+C8h] [rbp-598h]
  __int64 v449; // [rsp+C8h] [rbp-598h]
  unsigned int v450; // [rsp+D8h] [rbp-588h] BYREF
  int v451; // [rsp+DCh] [rbp-584h] BYREF
  __int64 v452; // [rsp+E0h] [rbp-580h] BYREF
  __int64 v453; // [rsp+E8h] [rbp-578h] BYREF
  __int64 v454; // [rsp+F0h] [rbp-570h] BYREF
  unsigned __int64 v455; // [rsp+F8h] [rbp-568h] BYREF
  unsigned __int64 *v456; // [rsp+100h] [rbp-560h] BYREF
  __int64 v457; // [rsp+108h] [rbp-558h]
  unsigned __int64 *v458; // [rsp+110h] [rbp-550h]
  _QWORD v459[2]; // [rsp+120h] [rbp-540h] BYREF
  __int64 v460; // [rsp+130h] [rbp-530h]
  __int64 v461; // [rsp+138h] [rbp-528h]
  unsigned int v462; // [rsp+140h] [rbp-520h]
  _BYTE *v463; // [rsp+150h] [rbp-510h] BYREF
  __int64 v464; // [rsp+158h] [rbp-508h]
  _BYTE v465[160]; // [rsp+160h] [rbp-500h] BYREF
  __int64 *v466; // [rsp+200h] [rbp-460h] BYREF
  __int64 v467; // [rsp+208h] [rbp-458h]
  __int64 v468; // [rsp+210h] [rbp-450h] BYREF
  char v469; // [rsp+218h] [rbp-448h]
  unsigned __int64 v470; // [rsp+310h] [rbp-350h] BYREF
  __int64 v471; // [rsp+318h] [rbp-348h]
  _QWORD v472[3]; // [rsp+320h] [rbp-340h] BYREF
  _BYTE *v473; // [rsp+338h] [rbp-328h]
  __int64 v474; // [rsp+340h] [rbp-320h]
  _BYTE v475[160]; // [rsp+348h] [rbp-318h] BYREF
  _BYTE *v476; // [rsp+3E8h] [rbp-278h]
  __int64 v477; // [rsp+3F0h] [rbp-270h]
  _BYTE v478[168]; // [rsp+3F8h] [rbp-268h] BYREF
  unsigned __int64 v479; // [rsp+4A0h] [rbp-1C0h] BYREF
  __int64 v480; // [rsp+4A8h] [rbp-1B8h]
  _QWORD v481[2]; // [rsp+4B0h] [rbp-1B0h] BYREF
  unsigned __int64 v482; // [rsp+4C0h] [rbp-1A0h]
  _BYTE *v483; // [rsp+4C8h] [rbp-198h]
  __int64 v484; // [rsp+4D0h] [rbp-190h]
  _BYTE v485[24]; // [rsp+4D8h] [rbp-188h] BYREF
  __int64 v486; // [rsp+4F0h] [rbp-170h]
  unsigned int v487; // [rsp+500h] [rbp-160h]
  unsigned __int64 v488; // [rsp+508h] [rbp-158h]
  _BYTE *v489; // [rsp+550h] [rbp-110h]
  _BYTE v490[16]; // [rsp+568h] [rbp-F8h] BYREF
  _BYTE *v491; // [rsp+578h] [rbp-E8h]
  __int64 v492; // [rsp+580h] [rbp-E0h]
  _BYTE v493[216]; // [rsp+588h] [rbp-D8h] BYREF

  m128i_i64 = a1->m128i_i64;
  v3 = (_QWORD *)a2[2];
  v4 = *(__int64 (**)())(*v3 + 144LL);
  v5 = 0;
  if ( v4 != sub_2C8F680 )
    v5 = ((__int64 (__fastcall *)(_QWORD *))v4)(v3);
  a1[32].m128i_i64[1] = v5;
  v6 = *(__int64 (**)())(*v3 + 128LL);
  v7 = 0;
  if ( v6 != sub_2DAC790 )
    v7 = ((__int64 (__fastcall *)(_QWORD *))v6)(v3);
  a1[33].m128i_i64[0] = v7;
  v8 = (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 200LL))(v3);
  v9 = (__int64 *)a1->m128i_i64[1];
  a1[33].m128i_i64[1] = v8;
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
    goto LABEL_552;
  while ( *(_UNKNOWN **)v10 != &unk_501EC08 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_552;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_501EC08);
  v13 = (__int64 *)a1->m128i_i64[1];
  v459[1] = 0;
  v462 = 0;
  v460 = 0;
  v461 = 0;
  v459[0] = v12 + 200;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
    goto LABEL_552;
  while ( *(_UNKNOWN **)v14 != &unk_501F1C8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_552;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_501F1C8);
  v17 = (__int64 *)a1->m128i_i64[1];
  a1[34].m128i_i64[0] = v16 + 169;
  v18 = *v17;
  v19 = v17[1];
  if ( v18 == v19 )
LABEL_552:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F87C64 )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_552;
  }
  v320 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(
                       *(_QWORD *)(v18 + 8),
                       &unk_4F87C64)
                   + 176);
  a1[34].m128i_i64[1] = a2[4];
  sub_2FF7BB0(a1 + 14, v3);
  if ( !a1[33].m128i_i64[0] )
    goto LABEL_367;
  v327 = 0;
  v20 = *(_QWORD *)(*(_QWORD *)a1[34].m128i_i64[1] + 344LL) & 1LL;
  a1[39].m128i_i8[0] = v20;
  if ( !(_BYTE)v20 )
  {
    sub_34BEDF0((__int64)&v479, 1, 0, (__int64)v459, a1[34].m128i_i64[0], v320, 0);
    v248 = (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 200LL))(v3);
    v327 = sub_34C7080((__int64)&v479, a2, a1[33].m128i_i64[0], v248, 0, 0);
    if ( v491 )
      _libc_free((unsigned __int64)v491);
    if ( v489 != v490 )
      _libc_free((unsigned __int64)v489);
    if ( v488 )
      j_j___libc_free_0(v488);
    sub_C7D6A0(v486, 16LL * v487, 8);
    if ( !BYTE4(v484) )
      _libc_free(v482);
    v249 = v480;
    v250 = v479;
    if ( v480 != v479 )
    {
      do
      {
        v251 = *(_QWORD *)(v250 + 16);
        if ( v251 )
          sub_B91220(v250 + 16, v251);
        v250 += 24LL;
      }
      while ( v249 != v250 );
      v250 = v479;
    }
    if ( v250 )
      j_j___libc_free_0(v250);
  }
  v21 = a1[39].m128i_i32[1];
  if ( v21 < (int)qword_503BAA8 || (_DWORD)qword_503B9C8 != -1 && v21 > (int)qword_503B9C8 )
  {
LABEL_367:
    v123 = 0;
  }
  else
  {
    sub_2E7A760((__int64)a2, 0);
    v23 = a1[13].m128i_i64[0];
    v24 = a1[12].m128i_i64[1];
    v25 = (_QWORD *)(v23 - v24);
    v26 = (unsigned int)((__int64)(a2[13] - a2[12]) >> 3);
    v27 = 0x7D6343EB1A1F58D1LL * ((v23 - v24) >> 3);
    if ( v26 > v27 )
    {
      v297 = v26 - v27;
      if ( v26 - v27 > 0x7D6343EB1A1F58D1LL * ((a1[13].m128i_i64[1] - v23) >> 3) )
      {
        if ( v297 > 0x5397829CBC14E5LL - v27 )
          sub_4262D8((__int64)"vector::_M_default_append");
        v304 = v26 - v27;
        if ( v27 >= v297 )
          v304 = 0x7D6343EB1A1F58D1LL * ((a1[13].m128i_i64[0] - v24) >> 3);
        v305 = __CFADD__(v304, v27);
        v306 = v304 + v27;
        if ( v305 )
        {
          v307 = 0x7FFFFFFFFFFFFEA8LL;
        }
        else
        {
          if ( v306 > 0x5397829CBC14E5LL )
            v306 = 0x5397829CBC14E5LL;
          v307 = 392 * v306;
        }
        v439 = v25;
        v308 = sub_22077B0(v307);
        v25 = v439;
        v449 = v308;
        v310 = (char *)v439 + v308;
        v311 = &v310[392 * v297];
        do
        {
          if ( v310 )
          {
            *(_WORD *)v310 &= 0xFC00u;
            *((_QWORD *)v310 + 5) = v310 + 56;
            v309 = (__int64)(v310 + 232);
            *((_DWORD *)v310 + 1) = 0;
            *((_DWORD *)v310 + 2) = 0;
            *((_DWORD *)v310 + 3) = 0;
            *((_QWORD *)v310 + 2) = 0;
            *((_QWORD *)v310 + 3) = 0;
            *((_QWORD *)v310 + 4) = 0;
            *((_DWORD *)v310 + 12) = 0;
            *((_DWORD *)v310 + 13) = 4;
            *((_QWORD *)v310 + 27) = v310 + 232;
            *((_DWORD *)v310 + 56) = 0;
            *((_DWORD *)v310 + 57) = 4;
          }
          v310 += 392;
        }
        while ( v310 != v311 );
        v312 = (unsigned __int16 *)a1[12].m128i_i64[1];
        v440 = (unsigned __int16 *)a1[13].m128i_i64[0];
        if ( v440 != v312 )
        {
          v313 = v449;
          do
          {
            if ( v313 )
            {
              v314 = *v312;
              LOWORD(v314) = v314 & 0x3FF;
              *(_WORD *)v313 = v314 | *(_WORD *)v313 & 0xFC00;
              *(_DWORD *)(v313 + 4) = *((_DWORD *)v312 + 1);
              *(_DWORD *)(v313 + 8) = *((_DWORD *)v312 + 2);
              *(_DWORD *)(v313 + 12) = *((_DWORD *)v312 + 3);
              *(_QWORD *)(v313 + 16) = *((_QWORD *)v312 + 2);
              *(_QWORD *)(v313 + 24) = *((_QWORD *)v312 + 3);
              v315 = *((_QWORD *)v312 + 4);
              *(_DWORD *)(v313 + 48) = 0;
              *(_QWORD *)(v313 + 32) = v315;
              *(_QWORD *)(v313 + 40) = v313 + 56;
              *(_DWORD *)(v313 + 52) = 4;
              if ( *((_DWORD *)v312 + 12) )
                sub_34E6680(v313 + 40, (__int64)(v312 + 20), v314, v309, (__int64)v25, v22);
              *(_DWORD *)(v313 + 224) = 0;
              *(_QWORD *)(v313 + 216) = v313 + 232;
              *(_DWORD *)(v313 + 228) = 4;
              if ( *((_DWORD *)v312 + 56) )
                sub_34E6680(v313 + 216, (__int64)(v312 + 108), v314, v309, (__int64)v25, v22);
            }
            v312 += 196;
            v313 += 392;
          }
          while ( v440 != v312 );
          v316 = (unsigned __int16 *)a1[13].m128i_i64[0];
          v312 = (unsigned __int16 *)a1[12].m128i_i64[1];
          if ( v316 != v312 )
          {
            do
            {
              v317 = *((_QWORD *)v312 + 27);
              if ( (unsigned __int16 *)v317 != v312 + 116 )
                _libc_free(v317);
              v318 = *((_QWORD *)v312 + 5);
              if ( (unsigned __int16 *)v318 != v312 + 28 )
                _libc_free(v318);
              v312 += 196;
            }
            while ( v316 != v312 );
            v312 = (unsigned __int16 *)m128i_i64[25];
          }
        }
        if ( v312 )
          j_j___libc_free_0((unsigned __int64)v312);
        m128i_i64[25] = v449;
        m128i_i64[27] = v449 + v307;
        m128i_i64[26] = v449 + 392 * v26;
      }
      else
      {
        v298 = v23 + 392 * v297;
        do
        {
          if ( v23 )
          {
            *(_WORD *)v23 &= 0xFC00u;
            *(_QWORD *)(v23 + 40) = v23 + 56;
            *(_DWORD *)(v23 + 4) = 0;
            *(_DWORD *)(v23 + 8) = 0;
            *(_DWORD *)(v23 + 12) = 0;
            *(_QWORD *)(v23 + 16) = 0;
            *(_QWORD *)(v23 + 24) = 0;
            *(_QWORD *)(v23 + 32) = 0;
            *(_DWORD *)(v23 + 48) = 0;
            *(_DWORD *)(v23 + 52) = 4;
            *(_QWORD *)(v23 + 216) = v23 + 232;
            *(_DWORD *)(v23 + 224) = 0;
            *(_DWORD *)(v23 + 228) = 4;
          }
          v23 += 392;
        }
        while ( v23 != v298 );
        a1[13].m128i_i64[0] = v23;
      }
    }
    else if ( v26 < v27 )
    {
      v288 = 392 * v26 + v24;
      if ( v23 != v288 )
      {
        v289 = v288;
        do
        {
          v290 = *(_QWORD *)(v289 + 216);
          if ( v290 != v289 + 232 )
            _libc_free(v290);
          v291 = *(_QWORD *)(v289 + 40);
          if ( v291 != v289 + 56 )
            _libc_free(v291);
          v289 += 392;
        }
        while ( v23 != v289 );
        m128i_i64[26] = v288;
      }
    }
    v28 = qword_503B8E8;
    v456 = 0;
    v457 = 0;
    v29 = (_DWORD)qword_503B8E8 == 0;
    v30 = (int)qword_503B8E8 < 0;
    *((_BYTE *)m128i_i64 + 625) = 0;
    v458 = 0;
    v328 = v28 == -1 || !v30 && !v29;
    if ( v328 )
    {
      v329 = a2 + 40;
      while ( 1 )
      {
        v31 = m128i_i64;
        v335 = a2[41];
        if ( (_QWORD *)v335 != v329 )
        {
          while ( 1 )
          {
            v32 = v335;
            v33 = &v468;
            v469 = 0;
            v441 = 0;
            v34 = 1;
            v466 = &v468;
            v467 = 0x1000000001LL;
            v468 = v335;
            while ( 1 )
            {
              v55 = v31[25];
              v56 = v55 + 392LL * *(int *)(v32 + 24);
              if ( v441 )
                break;
              v35 = *(_BYTE *)v56;
              if ( (*(_BYTE *)v56 & 4) != 0 || (v35 & 2) != 0 )
              {
                v34 = (unsigned int)(v34 - 1);
                LODWORD(v467) = v34;
                goto LABEL_52;
              }
              v36 = v35 | 2;
              *(_QWORD *)(v56 + 16) = v32;
              *(_BYTE *)v56 = v36;
              v442 = v36 & 1;
              if ( (v36 & 1) == 0 )
              {
                *(_QWORD *)(v56 + 32) = 0;
                *(_QWORD *)(v56 + 24) = 0;
                *(_DWORD *)(v56 + 48) = 0;
                v37 = v31[66];
                v38 = *(__int64 (**)())(*(_QWORD *)v37 + 344LL);
                if ( v38 == sub_2DB1AE0 )
                {
                  *(_BYTE *)v56 = v36 & 0xEF;
                  goto LABEL_32;
                }
                v433 = v33;
                v188 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v38)(
                         v37,
                         v32,
                         v56 + 24,
                         v56 + 32,
                         v56 + 40,
                         0);
                v33 = v433;
                v342 = v188 ^ 1;
                *(_BYTE *)v56 = (16 * ((v188 ^ 1) & 1)) | *(_BYTE *)v56 & 0xEF;
                if ( v188 != 1 )
                {
                  v189 = *(void **)(v56 + 40);
                  v25 = v481;
                  v190 = 5LL * *(unsigned int *)(v56 + 48);
                  v479 = (unsigned __int64)v481;
                  v480 = 0x400000000LL;
                  v191 = -858993459 * v190;
                  v22 = 8 * v190;
                  if ( v190 > 0x14 )
                  {
                    nm = 8 * v190;
                    srcp = v189;
                    sub_C8D5F0((__int64)&v479, v481, 0xCCCCCCCCCCCCCCCDLL * v190, 0x28u, (__int64)v481, v22);
                    v189 = srcp;
                    v22 = nm;
                    v33 = v433;
                    v192 = (_QWORD *)(v479 + 40LL * (unsigned int)v480);
                    goto LABEL_261;
                  }
                  if ( v22 )
                  {
                    v192 = v481;
LABEL_261:
                    v434 = v33;
                    memcpy(v192, v189, v22);
                    v25 = v481;
                    v33 = v434;
                  }
                  LODWORD(v480) = v480 + v191;
                  if ( (_DWORD)v480 )
                  {
                    v193 = v31[66];
                    v194 = *(__int64 (**)())(*(_QWORD *)v193 + 880LL);
                    if ( v194 != sub_2DB1B20 )
                    {
                      v435 = v33;
                      v195 = ((__int64 (__fastcall *)(__int64, unsigned __int64 *))v194)(v193, &v479);
                      v33 = v435;
                      v25 = v481;
                      v442 = v195 ^ 1;
                    }
                  }
                  else
                  {
                    v442 = v342;
                  }
                }
                else
                {
LABEL_32:
                  *(_QWORD *)(v56 + 24) = 0;
                  v25 = v481;
                  *(_QWORD *)(v56 + 32) = 0;
                  *(_DWORD *)(v56 + 48) = 0;
                  v480 = 0x400000000LL;
                  v479 = (unsigned __int64)v481;
                  v442 = v328;
                }
                v39 = (32 * (v442 & 1)) | *(_BYTE *)v56 & 0xDF;
                v40 = *(_DWORD *)(v56 + 48);
                *(_BYTE *)v56 = v39;
                if ( (v39 & 0x10) != 0 )
                {
                  if ( *(_QWORD *)(v56 + 32) )
                  {
                    *(_BYTE *)v56 = v39 & 0xBF;
                    goto LABEL_36;
                  }
                  *(_BYTE *)v56 = v39 | 0x40;
                  if ( v40 )
                  {
LABEL_252:
                    v184 = *(_QWORD *)(v56 + 16);
                    v185 = *(__int64 **)(v184 + 112);
                    v186 = &v185[*(unsigned int *)(v184 + 120)];
                    if ( v185 == v186 )
                    {
LABEL_324:
                      *(_QWORD *)(v56 + 32) = 0;
                    }
                    else
                    {
                      while ( 1 )
                      {
                        v187 = *v185;
                        if ( *(_QWORD *)(v56 + 24) != *v185 )
                          break;
                        if ( v186 == ++v185 )
                          goto LABEL_324;
                      }
                      *(_QWORD *)(v56 + 32) = v187;
                      if ( v187 )
                        goto LABEL_36;
                    }
                    *(_BYTE *)v56 |= 0x80u;
                  }
                }
                else
                {
                  *(_BYTE *)v56 = v39 & 0xBF;
                  if ( v40 && !*(_QWORD *)(v56 + 32) )
                    goto LABEL_252;
                }
LABEL_36:
                if ( (_QWORD *)v479 != v481 )
                {
                  v443 = v33;
                  _libc_free(v479);
                  v33 = v443;
                }
              }
              v41 = *(_QWORD *)(v56 + 16);
              v42 = *(_QWORD *)(v41 + 56);
              v479 = v41 + 48;
              v470 = v42;
              v43 = *(_BYTE *)v56;
              if ( (*(_BYTE *)v56 & 1) == 0 && v43 >= 0 )
              {
                v444 = v33;
                sub_34E85D0((__int64)v31, v56, (__int64 *)&v470, (__int64 *)&v479, 0);
                v43 = *(_BYTE *)v56;
                v33 = v444;
              }
              if ( (v43 & 0x10) == 0 )
                goto LABEL_141;
              if ( !*(_DWORD *)(v56 + 48) )
                goto LABEL_141;
              if ( (v43 & 1) != 0 )
                goto LABEL_141;
              if ( *(_QWORD *)(v56 + 24) == v32 )
                goto LABEL_141;
              v44 = *(_QWORD *)(v56 + 32);
              if ( v44 == v32 )
                goto LABEL_141;
              if ( !v44 )
                goto LABEL_142;
              v45 = v326;
              LOBYTE(v33[2 * v34 - 1]) = 1;
              LOBYTE(v45) = 0;
              v46 = *(_QWORD *)(v56 + 32);
              v326 = v45;
              v47 = (unsigned int)v467;
              v48 = (unsigned int)v467 + 1LL;
              if ( v48 > HIDWORD(v467) )
              {
                sub_C8D5F0((__int64)&v466, &v468, v48, 0x10u, (__int64)v25, v22);
                v47 = (unsigned int)v467;
              }
              v49 = v325;
              v50 = &v466[2 * v47];
              *v50 = v46;
              LOBYTE(v49) = 0;
              v50[1] = v326;
              v325 = v49;
              LODWORD(v467) = v467 + 1;
              v51 = (unsigned int)v467;
              v52 = *(_QWORD *)(v56 + 24);
              if ( (unsigned __int64)(unsigned int)v467 + 1 > HIDWORD(v467) )
              {
                sub_C8D5F0((__int64)&v466, &v468, (unsigned int)v467 + 1LL, 0x10u, (__int64)v25, v22);
                v51 = (unsigned int)v467;
              }
              v53 = &v466[2 * v51];
              *v53 = v52;
              v53[1] = v325;
              v33 = v466;
              v34 = (unsigned int)(v467 + 1);
              LODWORD(v467) = v467 + 1;
LABEL_52:
              if ( !(_DWORD)v34 )
                goto LABEL_69;
LABEL_53:
              v54 = (__int64)&v33[2 * (unsigned int)v34 - 2];
              v32 = *(_QWORD *)v54;
              v441 = *(_BYTE *)(v54 + 8);
            }
            v57 = v55 + 392LL * *(int *)(*(_QWORD *)(v56 + 24) + 24LL);
            v58 = v55 + 392LL * *(int *)(*(_QWORD *)(v56 + 32) + 24LL);
            if ( (*(_BYTE *)v57 & 1) == 0 || (*(_BYTE *)v58 & 1) == 0 )
            {
              v59 = *(unsigned int *)(v56 + 48);
              v60 = *(void **)(v56 + 40);
              v61 = 0x400000000LL;
              v464 = 0x400000000LL;
              v62 = 5 * v59;
              v63 = 0xCCCCCCCCCCCCCCCDLL * v62;
              v463 = v465;
              v64 = 8 * v62;
              v65 = 0xCCCCCCCCCCCCCCCDLL * v62;
              if ( v62 > 0x14 )
              {
                nk = 8 * v62;
                srci = v60;
                v431 = 0xCCCCCCCCCCCCCCCDLL * v62;
                sub_C8D5F0((__int64)&v463, v465, v65, 0x28u, v64, (__int64)v60);
                v63 = v431;
                v60 = srci;
                v64 = nk;
                v125 = &v463[40 * (unsigned int)v464];
              }
              else
              {
                if ( !v64 )
                  goto LABEL_59;
                v125 = v465;
              }
              v432 = v63;
              memcpy(v125, v60, v64);
              v65 = v432 + (unsigned int)v464;
LABEL_59:
              v66 = v31[66];
              LODWORD(v464) = v65;
              v67 = *(__int64 (**)())(*(_QWORD *)v66 + 880LL);
              if ( v67 == sub_2DB1B20 )
              {
                v68 = *(_DWORD *)(v57 + 224);
                v69 = *(_DWORD *)(v58 + 224);
                v450 = 0;
                v70 = v31[68];
                v451 = 0;
                v355 = v68 != 0;
                v341 = v69 != 0;
                v441 = 0;
                src = sub_2E441D0(v70, v32, *(_QWORD *)(v57 + 16));
                n = 0;
                goto LABEL_61;
              }
              v152 = ((__int64 (__fastcall *)(__int64, _BYTE **, unsigned __int64, __int64, __int64, void *))v67)(
                       v66,
                       &v463,
                       v65,
                       v61,
                       v64,
                       v60);
              v153 = *(_DWORD *)(v57 + 224);
              v154 = *(_DWORD *)(v58 + 224);
              v450 = 0;
              v155 = *(_QWORD *)(v57 + 16);
              n = v152;
              v156 = v31[68];
              v355 = v153 != 0;
              v341 = v154 != 0;
              v451 = 0;
              v330 = __PAIR64__(v153, v154);
              src = sub_2E441D0(v156, v32, v155);
              if ( n )
              {
                v441 = 0;
                n = 0;
                v71 = sub_34E6540((__int64)v31, (char *)v57, v58, 0, &v450, src);
              }
              else
              {
                v470 = 0;
                v473 = v475;
                v476 = v478;
                v471 = 0;
                memset(v472, 0, sizeof(v472));
                v474 = 0x400000000LL;
                v477 = 0x400000000LL;
                v479 = 0;
                v480 = 0;
                v481[0] = 0;
                v481[1] = 0;
                v482 = 0;
                v483 = v485;
                v484 = 0x400000000LL;
                v491 = v493;
                v492 = 0x400000000LL;
                v451 = 0;
                v450 = 0;
                v157 = *(_BYTE *)v57;
                if ( (*(_BYTE *)v57 & 2) == 0 && (v157 & 1) == 0 )
                {
                  v158 = *(_BYTE *)v58;
                  if ( (*(_BYTE *)v58 & 2) == 0 && (v158 & 1) == 0 )
                  {
                    v159 = *(_QWORD *)(v57 + 16);
                    v160 = *(_QWORD *)(v58 + 16);
                    if ( v159 != v160 )
                    {
                      v161 = *(_QWORD *)(v57 + 24);
                      v162 = *(_QWORD *)(v58 + 24);
                      if ( v161
                        || (v157 & 0x10) != 0 && (v161 = *(_QWORD *)(v159 + 8), v161 != *(_QWORD *)(v159 + 32) + 320LL) )
                      {
                        if ( !v162 && (v158 & 0x10) != 0 )
                          goto LABEL_340;
                        goto LABEL_204;
                      }
                      if ( !v162 )
                      {
                        if ( (v158 & 0x10) == 0 )
                          goto LABEL_333;
                        v161 = 0;
LABEL_340:
                        v162 = *(_QWORD *)(v160 + 8);
                        if ( v162 == *(_QWORD *)(v160 + 32) + 320LL )
                          v162 = 0;
LABEL_204:
                        if ( v162 == v161 )
                        {
                          if ( v162 )
                          {
LABEL_206:
                            if ( *(_DWORD *)(v159 + 72) <= 1u
                              && *(_DWORD *)(v160 + 72) <= 1u
                              && !*(_QWORD *)(v57 + 32)
                              && !*(_QWORD *)(v58 + 32) )
                            {
                              v247 = 0;
                              if ( (v157 & 0x10) != 0 )
                                v247 = (v158 & 0x10) != 0;
                              v452 = *(_QWORD *)(v159 + 56);
                              v453 = *(_QWORD *)(v160 + 56);
                              v454 = v159 + 48;
                              v455 = v160 + 48;
                              if ( (unsigned __int8)sub_34E9A00(
                                                      (__int64)v31,
                                                      &v452,
                                                      &v453,
                                                      &v454,
                                                      &v455,
                                                      &v450,
                                                      &v451,
                                                      v159,
                                                      v160,
                                                      v247) )
                              {
                                v472[0] = *(_QWORD *)(v57 + 16);
                                v481[0] = *(_QWORD *)(v58 + 16);
                                LOBYTE(v470) = *(_BYTE *)v57 & 0x10 | v470 & 0xEF;
                                LOBYTE(v479) = *(_BYTE *)v58 & 0x10 | v479 & 0xEF;
                                if ( sub_34E8A70(
                                       (__int64)v31,
                                       &v452,
                                       &v453,
                                       &v454,
                                       (__int64 *)&v455,
                                       (char *)&v470,
                                       (char *)&v479) )
                                {
                                  HIDWORD(v470) = *(_DWORD *)(v57 + 4);
                                  HIDWORD(v479) = *(_DWORD *)(v58 + 4);
                                  v302 = sub_34EA620(
                                           (__int64)v31,
                                           (__int64)&v470,
                                           (__int64)&v479,
                                           v32,
                                           v451 + v450,
                                           src,
                                           0);
                                  if ( (*(_BYTE *)v57 & 1) == 0 )
                                  {
                                    v354 = sub_34E95D0((__int64)v31, v57, v56 + 40, 0, 0, 1u);
                                    if ( (*(_BYTE *)v58 & 1) == 0 )
                                    {
                                      v348 = sub_34E95D0((__int64)v31, v58, (__int64)&v463, 0, 0, 1u) & v354 & v302;
                                      if ( v348 )
                                      {
                                        v333 = v450;
                                        v322 = (v479 & 0x200) != 0;
                                        v324 = (v470 & 0x200) != 0;
                                        nj = v451;
                                        v303 = sub_22077B0(0x18u);
                                        v230 = v303;
                                        if ( v303 )
                                        {
                                          *(_QWORD *)v303 = v56;
                                          *(_DWORD *)(v303 + 8) = 7;
                                          *(_DWORD *)(v303 + 12) = v333;
                                          *(_DWORD *)(v303 + 16) = nj;
                                          *(_BYTE *)(v303 + 20) = *(_BYTE *)(v303 + 20) & 0xF8
                                                                | (4 * v322)
                                                                | (2 * v324)
                                                                | (v330 != 0);
                                        }
                                        v455 = v303;
                                        v231 = (_QWORD *)v457;
                                        if ( (unsigned __int64 *)v457 == v458 )
                                        {
LABEL_507:
                                          sub_34E6A90(
                                            (unsigned __int64 *)&v456,
                                            (unsigned __int64 *)v457,
                                            (__int64 *)&v455);
                                          v230 = v455;
                                          goto LABEL_327;
                                        }
LABEL_312:
                                        if ( v231 )
                                        {
                                          *v231 = v230;
                                          v457 += 8;
                                        }
                                        else
                                        {
                                          v457 = 8;
LABEL_327:
                                          if ( v230 )
                                            j_j___libc_free_0(v230);
                                        }
                                        n = v348;
                                      }
                                    }
                                    goto LABEL_210;
                                  }
LABEL_315:
                                  if ( (*(_BYTE *)v58 & 1) == 0 )
                                  {
                                    sub_34E95D0((__int64)v31, v58, (__int64)&v463, 0, 0, 1u);
                                    n = 0;
                                  }
                                  goto LABEL_210;
                                }
                              }
                            }
                          }
                          else
                          {
LABEL_333:
                            if ( (v157 & 0x10) == 0 && (v158 & 0x10) == 0 )
                              goto LABEL_206;
                          }
                        }
                      }
                    }
                  }
                }
                if ( !sub_34EA220((__int64)v31, v57, v58, &v450, &v451, (__int64)&v470, (__int64)&v479) )
                  goto LABEL_210;
                v227 = sub_34EA620((__int64)v31, (__int64)&v470, (__int64)&v479, v32, v451 + v450, src, 1);
                if ( (*(_BYTE *)v57 & 1) != 0 )
                  goto LABEL_315;
                v347 = sub_34E95D0((__int64)v31, v57, v56 + 40, 0, 0, 1u);
                if ( (*(_BYTE *)v58 & 1) == 0 )
                {
                  v348 = sub_34E95D0((__int64)v31, v58, (__int64)&v463, 0, 0, 1u) & v347 & v227;
                  if ( v348 )
                  {
                    v228 = (v479 & 0x200) != 0;
                    v29 = v330 == 0;
                    v331 = v450;
                    v323 = (v470 & 0x200) != 0;
                    v332 = v451;
                    ne = !v29;
                    v229 = sub_22077B0(0x18u);
                    v230 = v229;
                    if ( v229 )
                    {
                      *(_QWORD *)v229 = v56;
                      *(_DWORD *)(v229 + 8) = 8;
                      *(_DWORD *)(v229 + 16) = v332;
                      *(_DWORD *)(v229 + 12) = v331;
                      *(_BYTE *)(v229 + 20) = *(_BYTE *)(v229 + 20) & 0xF8 | (4 * v228) | ne | (2 * v323);
                    }
                    v455 = v229;
                    v231 = (_QWORD *)v457;
                    if ( (unsigned __int64 *)v457 == v458 )
                      goto LABEL_507;
                    goto LABEL_312;
                  }
                }
LABEL_210:
                if ( v491 != v493 )
                  _libc_free((unsigned __int64)v491);
                if ( v483 != v485 )
                  _libc_free((unsigned __int64)v483);
                if ( v476 != v478 )
                  _libc_free((unsigned __int64)v476);
                if ( v473 != v475 )
                  _libc_free((unsigned __int64)v473);
LABEL_61:
                v71 = sub_34E6540((__int64)v31, (char *)v57, v58, 0, &v450, src);
              }
              if ( v71 )
              {
                v146 = (unsigned int)(*(_DWORD *)(v57 + 4) + *(_DWORD *)(v57 + 8));
                if ( (_DWORD)v146 )
                {
                  v147 = v31[66];
                  v148 = *(__int64 (**)())(*(_QWORD *)v147 + 416LL);
                  if ( v148 != sub_2DB1AF0 )
                  {
                    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v148)(
                           v147,
                           *(_QWORD *)(v57 + 16),
                           v146,
                           *(unsigned int *)(v57 + 12),
                           src) )
                    {
                      if ( (*(_BYTE *)v57 & 1) == 0 && *(char *)v57 >= 0 )
                      {
                        v149 = sub_34E95D0((__int64)v31, v57, v56 + 40, 1, 0, 0);
                        if ( v149 )
                        {
                          nc = v450;
                          v150 = sub_22077B0(0x18u);
                          v151 = v150;
                          if ( v150 )
                          {
                            *(_QWORD *)v150 = v56;
                            *(_DWORD *)(v150 + 8) = 6;
                            *(_QWORD *)(v150 + 12) = nc;
                            *(_BYTE *)(v150 + 20) = v355 | *(_BYTE *)(v150 + 20) & 0xF8;
                          }
                          v479 = v150;
                          if ( (unsigned __int64 *)v457 == v458 )
                          {
                            sub_34E6A90((unsigned __int64 *)&v456, (unsigned __int64 *)v457, (__int64 *)&v479);
                            v151 = v479;
                          }
                          else
                          {
                            if ( v457 )
                            {
                              *(_QWORD *)v457 = v150;
                              v457 += 8;
                              goto LABEL_195;
                            }
                            v457 = 8;
                          }
                          if ( v151 )
                            j_j___libc_free_0(v151);
LABEL_195:
                          n = v149;
                        }
                      }
                    }
                  }
                }
              }
              if ( (unsigned __int8)sub_34E6540((__int64)v31, (char *)v57, v58, 1, &v450, src) )
              {
                v140 = (unsigned int)(*(_DWORD *)(v57 + 4) + *(_DWORD *)(v57 + 8));
                if ( (_DWORD)v140 )
                {
                  v141 = v31[66];
                  v142 = *(__int64 (**)())(*(_QWORD *)v141 + 416LL);
                  if ( v142 != sub_2DB1AF0 )
                  {
                    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v142)(
                           v141,
                           *(_QWORD *)(v57 + 16),
                           v140,
                           *(unsigned int *)(v57 + 12),
                           src) )
                    {
                      if ( (*(_BYTE *)v57 & 1) == 0 && *(char *)v57 >= 0 )
                      {
                        v143 = sub_34E95D0((__int64)v31, v57, v56 + 40, 1, 1, 0);
                        if ( v143 )
                        {
                          nb = v450;
                          v144 = sub_22077B0(0x18u);
                          v145 = v144;
                          if ( v144 )
                          {
                            *(_QWORD *)v144 = v56;
                            *(_DWORD *)(v144 + 8) = 4;
                            *(_QWORD *)(v144 + 12) = nb;
                            *(_BYTE *)(v144 + 20) = v355 | *(_BYTE *)(v144 + 20) & 0xF8;
                          }
                          v479 = v144;
                          if ( (unsigned __int64 *)v457 == v458 )
                          {
                            sub_34E6A90((unsigned __int64 *)&v456, (unsigned __int64 *)v457, (__int64 *)&v479);
                            v145 = v479;
                          }
                          else
                          {
                            if ( v457 )
                            {
                              *(_QWORD *)v457 = v144;
                              v457 += 8;
                              goto LABEL_183;
                            }
                            v457 = 8;
                          }
                          if ( v145 )
                            j_j___libc_free_0(v145);
LABEL_183:
                          n = v143;
                        }
                      }
                    }
                  }
                }
              }
              if ( (unsigned __int8)sub_34E68B0((__int64)v31, (char *)v57, &v450) )
              {
                v134 = (unsigned int)(*(_DWORD *)(v57 + 4) + *(_DWORD *)(v57 + 8));
                if ( (_DWORD)v134 )
                {
                  v135 = v31[66];
                  v136 = *(__int64 (**)())(*(_QWORD *)v135 + 416LL);
                  if ( v136 != sub_2DB1AF0 )
                  {
                    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v136)(
                           v135,
                           *(_QWORD *)(v57 + 16),
                           v134,
                           *(unsigned int *)(v57 + 12),
                           src) )
                    {
                      if ( (*(_BYTE *)v57 & 1) == 0 && *(char *)v57 >= 0 )
                      {
                        v137 = sub_34E95D0((__int64)v31, v57, v56 + 40, 0, 0, 0);
                        if ( v137 )
                        {
                          na = v450;
                          v138 = sub_22077B0(0x18u);
                          v139 = v138;
                          if ( v138 )
                          {
                            *(_QWORD *)v138 = v56;
                            *(_DWORD *)(v138 + 8) = 2;
                            *(_QWORD *)(v138 + 12) = na;
                            *(_BYTE *)(v138 + 20) = v355 | *(_BYTE *)(v138 + 20) & 0xF8;
                          }
                          v479 = v138;
                          if ( (unsigned __int64 *)v457 == v458 )
                          {
                            sub_34E6A90((unsigned __int64 *)&v456, (unsigned __int64 *)v457, (__int64 *)&v479);
                            v139 = v479;
                          }
                          else
                          {
                            if ( v457 )
                            {
                              *(_QWORD *)v457 = v138;
                              v457 += 8;
                              goto LABEL_171;
                            }
                            v457 = 8;
                          }
                          if ( v139 )
                            j_j___libc_free_0(v139);
LABEL_171:
                          n = v137;
                        }
                      }
                    }
                  }
                }
              }
              if ( !v441 )
              {
LABEL_66:
                *(_BYTE *)v56 = *(_BYTE *)v56 & 0xF1 | (8 * n + 4) & 0xE;
                v34 = (unsigned int)(v467 - 1);
                LODWORD(v467) = v467 - 1;
                if ( v463 != v465 )
                {
                  _libc_free((unsigned __int64)v463);
                  v34 = (unsigned int)v467;
                }
                goto LABEL_68;
              }
              v126 = 0x80000000 - src;
              if ( (unsigned __int8)sub_34E6540((__int64)v31, (char *)v58, v57, 0, &v450, 0x80000000 - src) )
              {
                v202 = (unsigned int)(*(_DWORD *)(v58 + 4) + *(_DWORD *)(v58 + 8));
                if ( (_DWORD)v202 )
                {
                  v203 = v31[66];
                  v204 = *(__int64 (**)())(*(_QWORD *)v203 + 416LL);
                  if ( v204 != sub_2DB1AF0 )
                  {
                    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v204)(
                           v203,
                           *(_QWORD *)(v58 + 16),
                           v202,
                           *(unsigned int *)(v58 + 12),
                           v126) )
                    {
                      if ( (*(_BYTE *)v58 & 1) == 0 && *(char *)v58 >= 0 )
                      {
                        v448 = sub_34E95D0((__int64)v31, v58, (__int64)&v463, 1, 0, 0);
                        if ( v448 )
                        {
                          srcc = v450;
                          v205 = sub_22077B0(0x18u);
                          v206 = v205;
                          if ( v205 )
                          {
                            *(_QWORD *)v205 = v56;
                            *(_DWORD *)(v205 + 8) = 5;
                            *(_QWORD *)(v205 + 12) = srcc;
                            *(_BYTE *)(v205 + 20) = v341 | *(_BYTE *)(v205 + 20) & 0xF8;
                          }
                          v479 = v205;
                          if ( (unsigned __int64 *)v457 == v458 )
                          {
                            sub_34E6A90((unsigned __int64 *)&v456, (unsigned __int64 *)v457, (__int64 *)&v479);
                            v206 = v479;
                          }
                          else
                          {
                            if ( v457 )
                            {
                              *(_QWORD *)v457 = v205;
                              v457 += 8;
                              goto LABEL_288;
                            }
                            v457 = 8;
                          }
                          if ( v206 )
                            j_j___libc_free_0(v206);
LABEL_288:
                          n = v448;
                        }
                      }
                    }
                  }
                }
              }
              if ( (unsigned __int8)sub_34E6540((__int64)v31, (char *)v58, v57, 1, &v450, v126) )
              {
                v196 = (unsigned int)(*(_DWORD *)(v58 + 4) + *(_DWORD *)(v58 + 8));
                if ( (_DWORD)v196 )
                {
                  v197 = v31[66];
                  v198 = *(__int64 (**)())(*(_QWORD *)v197 + 416LL);
                  if ( v198 != sub_2DB1AF0 )
                  {
                    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v198)(
                           v197,
                           *(_QWORD *)(v58 + 16),
                           v196,
                           *(unsigned int *)(v58 + 12),
                           v126) )
                    {
                      if ( (*(_BYTE *)v58 & 1) == 0 && *(char *)v58 >= 0 )
                      {
                        v199 = sub_34E95D0((__int64)v31, v58, (__int64)&v463, 1, 1, 0);
                        if ( v199 )
                        {
                          v447 = v450;
                          v200 = sub_22077B0(0x18u);
                          v201 = v200;
                          if ( v200 )
                          {
                            *(_QWORD *)v200 = v56;
                            *(_DWORD *)(v200 + 8) = 3;
                            *(_QWORD *)(v200 + 12) = v447;
                            *(_BYTE *)(v200 + 20) = v341 | *(_BYTE *)(v200 + 20) & 0xF8;
                          }
                          v479 = v200;
                          if ( (unsigned __int64 *)v457 == v458 )
                          {
                            sub_34E6A90((unsigned __int64 *)&v456, (unsigned __int64 *)v457, (__int64 *)&v479);
                            v201 = v479;
                          }
                          else
                          {
                            if ( v457 )
                            {
                              *(_QWORD *)v457 = v200;
                              v457 += 8;
                              goto LABEL_276;
                            }
                            v457 = 8;
                          }
                          if ( v201 )
                            j_j___libc_free_0(v201);
LABEL_276:
                          n = v199;
                        }
                      }
                    }
                  }
                }
              }
              if ( !(unsigned __int8)sub_34E68B0((__int64)v31, (char *)v58, &v450) )
                goto LABEL_66;
              v127 = (unsigned int)(*(_DWORD *)(v58 + 4) + *(_DWORD *)(v58 + 8));
              if ( !(_DWORD)v127 )
                goto LABEL_66;
              v128 = v31[66];
              v129 = *(__int64 (**)())(*(_QWORD *)v128 + 416LL);
              if ( v129 == sub_2DB1AF0 )
                goto LABEL_66;
              if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v129)(
                      v128,
                      *(_QWORD *)(v58 + 16),
                      v127,
                      *(unsigned int *)(v58 + 12),
                      v126) )
                goto LABEL_66;
              if ( (*(_BYTE *)v58 & 1) != 0 )
                goto LABEL_66;
              if ( *(char *)v58 < 0 )
                goto LABEL_66;
              v130 = sub_34E95D0((__int64)v31, v58, (__int64)&v463, 0, 0, 0);
              if ( !v130 )
                goto LABEL_66;
              v131 = v450;
              v132 = sub_22077B0(0x18u);
              v133 = v132;
              if ( v132 )
              {
                *(_QWORD *)v132 = v56;
                *(_DWORD *)(v132 + 8) = 1;
                *(_DWORD *)(v132 + 12) = v131;
                *(_DWORD *)(v132 + 16) = 0;
                *(_BYTE *)(v132 + 20) = v341 | *(_BYTE *)(v132 + 20) & 0xF8;
              }
              v479 = v132;
              if ( (unsigned __int64 *)v457 == v458 )
              {
                sub_34E6A90((unsigned __int64 *)&v456, (unsigned __int64 *)v457, (__int64 *)&v479);
                v133 = v479;
              }
              else
              {
                if ( v457 )
                {
                  *(_QWORD *)v457 = v132;
                  v457 += 8;
LABEL_159:
                  n = v130;
                  goto LABEL_66;
                }
                v457 = 8;
              }
              if ( v133 )
                j_j___libc_free_0(v133);
              goto LABEL_159;
            }
LABEL_141:
            v43 = *(_BYTE *)v56;
LABEL_142:
            *(_BYTE *)v56 = v43 & 0xF9 | 4;
            v34 = (unsigned int)(v467 - 1);
            LODWORD(v467) = v467 - 1;
LABEL_68:
            v33 = v466;
            if ( (_DWORD)v34 )
              goto LABEL_53;
LABEL_69:
            if ( v33 != &v468 )
              _libc_free((unsigned __int64)v33);
            v335 = *(_QWORD *)(v335 + 8);
            if ( (_QWORD *)v335 == v329 )
            {
              m128i_i64 = v31;
              break;
            }
          }
        }
        v72 = (unsigned __int64 *)v457;
        v73 = v456;
        if ( v457 - (__int64)v456 <= 0 )
          goto LABEL_345;
        v74 = m128i_i64;
        v75 = v457;
        v76 = (v457 - (__int64)v456) >> 3;
        do
        {
          v445 = v76;
          v77 = (unsigned __int64 *)sub_2207800(8 * v76);
          v78 = v77;
          if ( v77 )
          {
            v79 = v76;
            v80 = (unsigned __int64 *)v75;
            m128i_i64 = v74;
            v81 = &v77[v445];
            *v77 = *v73;
            v82 = v77 + 1;
            *v73 = 0;
            if ( v81 == v78 + 1 )
            {
              v84 = v78;
            }
            else
            {
              do
              {
                v83 = *(v82 - 1);
                *(v82++ - 1) = 0;
                *(v82 - 1) = v83;
              }
              while ( v81 != v82 );
              v84 = &v78[v445 - 1];
            }
            v85 = *v84;
            *v84 = 0;
            v86 = *v73;
            *v73 = v85;
            if ( v86 )
            {
              v429 = v79;
              j_j___libc_free_0(v86);
              sub_34E94D0(v73, v80, v78, v429, (__int64)sub_34E6220);
            }
            else
            {
              sub_34E94D0(v73, v80, v78, v79, (__int64)sub_34E6220);
            }
            v87 = v78;
            do
            {
              if ( *v87 )
                j_j___libc_free_0(*v87);
              ++v87;
            }
            while ( v81 != v87 );
            goto LABEL_85;
          }
          v76 >>= 1;
        }
        while ( v76 );
        v72 = (unsigned __int64 *)v75;
        m128i_i64 = v74;
LABEL_345:
        v78 = 0;
        sub_34EAEA0(v73, v72, (__int64)sub_34E6220);
LABEL_85:
        j_j___libc_free_0((unsigned __int64)v78);
        v88 = v457;
        if ( (unsigned __int64 *)v457 == v456 )
          break;
        v430 = 0;
        v446 = (__int64)m128i_i64;
        do
        {
          v89 = *(_QWORD *)(v88 - 8);
          *(_QWORD *)(v88 - 8) = 0;
          v90 = (unsigned __int64 *)(v457 - 8);
          v457 = (__int64)v90;
          if ( *v90 )
            j_j___libc_free_0(*v90);
          v91 = *(char **)v89;
          v92 = **(_BYTE **)v89;
          if ( (v92 & 1) != 0 )
          {
            *v91 = v92 & 0xF7;
            goto LABEL_88;
          }
          v93 = (v92 & 8) != 0;
          if ( (v92 & 8) != 0 )
          {
            v94 = *(_DWORD *)(v89 + 8);
            v95 = *(_DWORD *)(v89 + 12);
            v96 = *(_DWORD *)(v89 + 16);
            *v91 = v92 & 0xF7;
            if ( v94 != 7 )
            {
              if ( v94 > 7 )
              {
                if ( v94 != 8 )
                  goto LABEL_552;
                if ( byte_503B2C8 )
                  goto LABEL_111;
                v356 = (v92 & 8) != 0;
                v367 = v96;
                v172 = *(_BYTE *)(v89 + 20);
                nd = v95;
                v173 = (v172 & 2) != 0;
                v406 = (v172 & 4) != 0;
                srcm = *(char **)(v446 + 200);
                v174 = (__int64)&srcm[392 * *(int *)(*((_QWORD *)v91 + 3) + 24LL)];
                v175 = *(int *)(*((_QWORD *)v91 + 4) + 24LL);
                v479 = 0;
                v176 = &srcm[392 * v175];
                v177 = sub_2E313E0(*(_QWORD *)(v174 + 16));
                v178 = v367;
                v179 = v356;
                v180 = nd;
                if ( v177 != *(_QWORD *)(v174 + 16) + 48LL && &v479 != (unsigned __int64 *)(v177 + 56) )
                {
                  if ( v479 )
                  {
                    v346 = v356;
                    v357 = (void *)v177;
                    sub_B91220((__int64)&v479, v479);
                    v179 = v346;
                    v177 = (unsigned __int64)v357;
                    v178 = v367;
                    v180 = nd;
                  }
                  v181 = *(_QWORD *)(v177 + 56);
                  v479 = v181;
                  if ( v181 )
                  {
                    v368 = v179;
                    nl = v178;
                    srcn = v180;
                    sub_B96E90((__int64)&v479, v181, 1);
                    v179 = v368;
                    v178 = nl;
                    v180 = srcn;
                  }
                }
                srcb = v179;
                v182 = sub_34EBFC0(v446, v91, v174, (__int64)v176, v180, v178, v173, v406, 1, 1);
                v93 = srcb;
                v109 = v182;
                if ( v182 )
                {
                  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, unsigned __int64 *, _QWORD))(**(_QWORD **)(v446 + 528) + 368LL))(
                    *(_QWORD *)(v446 + 528),
                    *((_QWORD *)v91 + 2),
                    *(_QWORD *)(v174 + 24),
                    *(_QWORD *)(v174 + 32),
                    *(_QWORD *)(v174 + 40),
                    *(unsigned int *)(v174 + 48),
                    &v479,
                    0);
                  *v176 |= 1u;
                  *(_BYTE *)v174 |= 1u;
                  v183 = *((_QWORD *)v91 + 2);
                  *v91 |= 1u;
                  sub_34E62C0(v446, v183);
                  v93 = srcb;
                  v103 = v319;
                }
                v98 = v479;
                if ( v479 )
                {
                  srco = v93;
                  sub_B91220((__int64)&v479, v479);
                  v93 = srco;
                }
                goto LABEL_110;
              }
              if ( v94 <= 2 )
              {
                if ( !v94 )
                  goto LABEL_552;
                if ( v94 != 1 )
                {
                  if ( !(_BYTE)qword_503B808 )
                    goto LABEL_99;
LABEL_111:
                  if ( (int)qword_503B8E8 <= 0 && (_DWORD)qword_503B8E8 != -1 )
                  {
                    m128i_i64 = (_QWORD *)v446;
                    j_j___libc_free_0(v89);
                    goto LABEL_114;
                  }
                  goto LABEL_88;
                }
                if ( byte_503B728 )
                  goto LABEL_111;
LABEL_99:
                v97 = (void *)*((_QWORD *)v91 + 5);
                v98 = 0xCCCCCCCCCCCCCCCDLL;
                srcg = *(void **)(v446 + 200);
                v99 = *(int *)(*((_QWORD *)v91 + 4) + 24LL);
                v100 = *((unsigned int *)v91 + 12);
                v101 = 392LL * *(int *)(*((_QWORD *)v91 + 3) + 24LL);
                v466 = &v468;
                v102 = (__int64)srcg + v101;
                v103 = (__int64)srcg + 392 * v99;
                v104 = 0x400000000LL;
                v467 = 0x400000000LL;
                v106 = 40 * v100;
                v105 = v106;
                v107 = 0xCCCCCCCCCCCCCCCDLL * (v106 >> 3);
                if ( (unsigned __int64)v106 > 0xA0 )
                {
                  v349 = v93;
                  v358 = v97;
                  v369 = v103;
                  nn = 0xCCCCCCCCCCCCCCCDLL * (v106 >> 3);
                  sub_C8D5F0((__int64)&v466, &v468, nn, 0x28u, v103, v107);
                  LODWORD(v107) = nn;
                  v103 = v369;
                  v97 = v358;
                  v93 = v349;
                  v232 = &v466[5 * (unsigned int)v467];
                }
                else
                {
                  if ( !v106 )
                    goto LABEL_101;
                  v232 = &v468;
                }
                v98 = (__int64)v97;
                v359 = v107;
                v370 = v93;
                no = v103;
                memcpy(v232, v97, v105);
                LODWORD(v106) = v467;
                LODWORD(v107) = v359;
                v93 = v370;
                v103 = no;
LABEL_101:
                LODWORD(v467) = v107 + v106;
                if ( v94 == 1 )
                {
                  v108 = v102;
                  v102 = v103;
                  v103 = v108;
                }
                v109 = *(_BYTE *)v102 & 1;
                if ( v109
                  || (v110 = *(_QWORD *)(v102 + 16), (*(_BYTE *)(v102 + 1) & 1) != 0) && *(_DWORD *)(v110 + 72) > 1u )
                {
                  *v91 &= ~4u;
                  v109 = 0;
                  *(_BYTE *)v102 &= ~4u;
                }
                else if ( !*(_BYTE *)(v110 + 217) )
                {
                  v109 = 0;
                  if ( !*(_QWORD *)(v110 + 224) )
                  {
                    v236 = *(_QWORD *)(v103 + 16);
                    if ( v94 == 1 )
                    {
                      v237 = *(_QWORD *)(v446 + 528);
                      v238 = *(__int64 (**)())(*(_QWORD *)v237 + 880LL);
                      if ( v238 == sub_2DB1B20 )
                        goto LABEL_552;
                      v373 = v93;
                      nr = *(_QWORD *)(v102 + 16);
                      v239 = ((__int64 (__fastcall *)(__int64, __int64 **))v238)(v237, &v466);
                      v110 = nr;
                      v93 = v373;
                      if ( v239 )
                        goto LABEL_552;
                    }
                    v240 = *(_QWORD *)(v446 + 536);
                    *(_QWORD *)(v446 + 576) = 0;
                    *(_QWORD *)(v446 + 560) = v240;
                    v241 = *(_DWORD *)(v240 + 16);
                    v242 = *(_DWORD *)(v446 + 616);
                    if ( v241 < v242 >> 2 || v241 > v242 )
                    {
                      v362 = v93;
                      v374 = v110;
                      v243 = (__int64)_libc_calloc(v241, 1u);
                      v110 = v374;
                      v93 = v362;
                      if ( !v243 )
                      {
                        if ( v241 )
                          goto LABEL_546;
                        v243 = malloc(1u);
                        v110 = v374;
                        v93 = v362;
                        if ( !v243 )
                          goto LABEL_546;
                      }
                      v244 = *(_QWORD *)(v446 + 608);
                      *(_QWORD *)(v446 + 608) = v243;
                      if ( v244 )
                      {
                        v375 = v93;
                        ns = v110;
                        _libc_free(v244);
                        v93 = v375;
                        v110 = ns;
                      }
                      *(_DWORD *)(v446 + 616) = v241;
                    }
                    if ( (*(_BYTE *)(**(_QWORD **)(v446 + 552) + 344LL) & 4) != 0 )
                    {
                      v382 = v93;
                      nt = v110;
                      sub_3508750(v446 + 560, v110);
                      sub_3508750(v446 + 560, v236);
                      v93 = v382;
                      v110 = nt;
                    }
                    v376 = v93;
                    nf = v110;
                    *((_DWORD *)v91 + 1) -= (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v446 + 528)
                                                                                              + 360LL))(
                                              *(_QWORD *)(v446 + 528),
                                              *((_QWORD *)v91 + 2),
                                              0);
                    if ( *(_DWORD *)(nf + 72) <= 1u )
                    {
                      sub_34E7EB0(v446, v102, nf + 48, (__int64)&v466, 0, nf);
                      sub_34EB320(v446, v91, v102, 1);
                    }
                    else
                    {
                      sub_34E8180(v446, (__int64)v91, v102, (__int64)&v466, 0, nf);
                      sub_2E33650(*((_QWORD *)v91 + 2), nf);
                    }
                    v245 = sub_34E6820(*((_QWORD *)v91 + 2), v236);
                    v246 = v376;
                    if ( !v245 )
                    {
                      sub_34E6930(*((_QWORD *)v91 + 2), v236, *(__int64 **)(v446 + 528));
                      v246 = v376;
                      *v91 = *v91 & 0xBE | 1;
                    }
                    v98 = *((_QWORD *)v91 + 2);
                    v109 = v246;
                    sub_34E62C0(v446, v98);
                    *(_BYTE *)v102 |= 1u;
                  }
                }
                if ( v466 != &v468 )
                {
                  srch = v93;
                  _libc_free((unsigned __int64)v466);
                  v93 = srch;
                }
LABEL_110:
                if ( !v109 )
                  goto LABEL_111;
LABEL_304:
                v430 = v93;
                if ( (*(_BYTE *)(**(_QWORD **)(v446 + 552) + 344LL) & 4) != 0 )
                  sub_3509090(*((_QWORD *)v91 + 2), v98, v90, v104, v103);
                goto LABEL_111;
              }
              v103 = v94 - 3;
              if ( (_BYTE)qword_503B648 && v94 == 6
                || byte_503B568 && (unsigned int)v103 <= 1
                || byte_503B488 && v94 == 5 )
              {
                goto LABEL_111;
              }
              v104 = 0x400000000LL;
              srcj = *(char **)(v446 + 200);
              v163 = *(int *)(*((_QWORD *)v91 + 4) + 24LL);
              v164 = 392LL * *(int *)(*((_QWORD *)v91 + 3) + 24LL);
              v463 = 0;
              v165 = (__int64)&srcj[v164];
              v166 = (void *)*((_QWORD *)v91 + 5);
              srca = (unsigned __int64 *)&srcj[392 * v163];
              v167 = *((unsigned int *)v91 + 12);
              v470 = (unsigned __int64)v472;
              v471 = 0x400000000LL;
              v169 = 40 * v167;
              v168 = v169;
              v170 = 0xCCCCCCCCCCCCCCCDLL * (v169 >> 3);
              if ( (unsigned __int64)v169 > 0xA0 )
              {
                v350 = v93;
                v360 = v166;
                v371 = v103;
                np = 0xCCCCCCCCCCCCCCCDLL * (v169 >> 3);
                sub_C8D5F0((__int64)&v470, v472, np, 0x28u, v103, v170);
                LODWORD(v170) = np;
                LODWORD(v103) = v371;
                v166 = v360;
                v93 = v350;
                v233 = (_QWORD *)(v470 + 40LL * (unsigned int)v471);
              }
              else
              {
                if ( !v169 )
                  goto LABEL_226;
                v233 = v472;
              }
              v361 = v170;
              v372 = v93;
              nq = v103;
              memcpy(v233, v166, v168);
              LODWORD(v169) = v471;
              LODWORD(v170) = v361;
              v93 = v372;
              v103 = nq;
LABEL_226:
              LODWORD(v471) = v169 + v170;
              if ( (v103 & 0xFFFFFFFD) == 0 )
              {
                v90 = (unsigned __int64 *)v165;
                v165 = (__int64)srca;
                srca = v90;
              }
              v109 = *(_BYTE *)v165 & 1;
              if ( v109
                || (v171 = *(_QWORD *)(v165 + 16), (*(_BYTE *)(v165 + 1) & 1) != 0) && *(_DWORD *)(v171 + 72) > 1u )
              {
                *v91 &= ~4u;
                v109 = 0;
                *(_BYTE *)v165 &= ~4u;
LABEL_233:
                if ( (_QWORD *)v470 != v472 )
                {
                  srck = v93;
                  _libc_free(v470);
                  v93 = srck;
                }
                v98 = (__int64)v463;
                if ( v463 )
                {
                  srcl = v93;
                  sub_B91220((__int64)&v463, (__int64)v463);
                  v93 = srcl;
                }
                goto LABEL_110;
              }
              if ( *(_BYTE *)(v171 + 217) )
                goto LABEL_233;
              v109 = 0;
              if ( *(_QWORD *)(v171 + 224) )
                goto LABEL_233;
              v252 = srca[2];
              if ( (v103 & 0xFFFFFFFD) == 0 )
              {
                v253 = *(_QWORD *)(v446 + 528);
                v254 = *(__int64 (**)())(*(_QWORD *)v253 + 880LL);
                if ( v254 == sub_2DB1B20 )
                  goto LABEL_552;
                v363 = v93;
                v377 = *(_QWORD *)(v165 + 16);
                nu = v103;
                v255 = ((__int64 (__fastcall *)(__int64, unsigned __int64 *))v254)(v253, &v470);
                LODWORD(v103) = nu;
                v171 = v377;
                v93 = v363;
                if ( v255 )
                  goto LABEL_552;
              }
              if ( (unsigned int)v103 <= 1 )
              {
                v381 = v93;
                ni = v171;
                v292 = sub_34E69B0(v446, v165);
                v171 = ni;
                v93 = v381;
                if ( v292 )
                {
                  v293 = *(_QWORD *)(ni + 64);
                  for ( i = v293 + 8LL * *(unsigned int *)(ni + 72); v293 != i; v293 += 8 )
                  {
                    if ( *(_QWORD *)v293 != *((_QWORD *)v91 + 2) )
                    {
                      v295 = (_BYTE *)(*(_QWORD *)(v446 + 200) + 392LL * *(int *)(*(_QWORD *)v293 + 24LL));
                      if ( (*v295 & 8) != 0 )
                        *v295 &= 0xF3u;
                    }
                  }
                }
              }
              v256 = *(_QWORD *)(v446 + 536);
              *(_QWORD *)(v446 + 576) = 0;
              *(_QWORD *)(v446 + 560) = v256;
              v257 = *(_DWORD *)(v256 + 16);
              v258 = *(_DWORD *)(v446 + 616);
              ng = v257;
              if ( v257 < v258 >> 2 || v257 > v258 )
              {
                v351 = v93;
                v364 = (void *)v171;
                v259 = (__int64)_libc_calloc(v257, 1u);
                v171 = (__int64)v364;
                v93 = v351;
                if ( !v259 )
                {
                  if ( ng || (v259 = malloc(1u), v171 = (__int64)v364, v93 = v351, !v259) )
LABEL_546:
                    sub_C64F00("Allocation failed", 1u);
                }
                v260 = *(_QWORD *)(v446 + 608);
                *(_QWORD *)(v446 + 608) = v259;
                if ( v260 )
                {
                  v365 = v93;
                  v378 = v171;
                  _libc_free(v260);
                  v93 = v365;
                  v171 = v378;
                }
                *(_DWORD *)(v446 + 616) = ng;
              }
              if ( (*(_BYTE *)(**(_QWORD **)(v446 + 552) + 344LL) & 4) != 0 )
              {
                v366 = v93;
                v383 = v171;
                sub_3508750(v446 + 560, v171);
                sub_3508750(v446 + 560, v252);
                v93 = v366;
                v171 = v383;
              }
              v343 = *(_QWORD *)(v165 + 32);
              if ( v343 )
              {
                v336 = v93;
                v379 = v171;
                sub_2E441D0(*(_QWORD *)(v446 + 544), v171, v252);
                v352 = v379;
                v380 = sub_2E441D0(*(_QWORD *)(v446 + 544), v379, *(_QWORD *)(v165 + 32));
                sub_2E441D0(*(_QWORD *)(v446 + 544), *((_QWORD *)v91 + 2), v252);
                v261 = sub_2E441D0(*(_QWORD *)(v446 + 544), *((_QWORD *)v91 + 2), v352);
                v171 = v352;
                v93 = v336;
                v353 = v261;
              }
              else
              {
                v353 = -1;
                v380 = -1;
              }
              v337 = v93;
              v339 = (_DWORD *)v171;
              *((_DWORD *)v91 + 1) -= (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v446 + 528)
                                                                                        + 360LL))(
                                        *(_QWORD *)(v446 + 528),
                                        *((_QWORD *)v91 + 2),
                                        0);
              if ( v339[18] > 1u )
              {
                sub_34E8180(v446, (__int64)v91, v165, (__int64)&v470, 1, (__int64)v339);
                v263 = (__int64)v339;
                v262 = v337;
              }
              else
              {
                *(_DWORD *)(v165 + 4) -= (*(__int64 (__fastcall **)(_QWORD, _DWORD *, _QWORD))(**(_QWORD **)(v446 + 528)
                                                                                             + 360LL))(
                                           *(_QWORD *)(v446 + 528),
                                           v339,
                                           0);
                sub_34E7EB0(v446, v165, (__int64)(v339 + 12), (__int64)&v470, 0, (__int64)v339);
                sub_34EB320(v446, v91, v165, 0);
                v262 = v337;
                v263 = (__int64)v339;
              }
              v407 = v262;
              sub_2E33650(*((_QWORD *)v91 + 2), v263);
              v264 = v407;
              if ( v343 )
              {
                v265 = *(unsigned int *)(v165 + 48);
                v266 = *(void **)(v165 + 40);
                v479 = (unsigned __int64)v481;
                v480 = 0x400000000LL;
                v268 = 40 * v265;
                v267 = v268;
                v408 = 0xCCCCCCCCCCCCCCCDLL * (v268 >> 3);
                if ( (unsigned __int64)v268 > 0xA0 )
                {
                  v334 = v268;
                  v338 = v264;
                  v340 = v266;
                  sub_C8D5F0((__int64)&v479, v481, 0xCCCCCCCCCCCCCCCDLL * (v268 >> 3), 0x28u, (__int64)&v479, v268);
                  v266 = v340;
                  v264 = v338;
                  v267 = v334;
                  v301 = (_QWORD *)(v479 + 40LL * (unsigned int)v480);
                }
                else
                {
                  if ( !v268 )
                  {
LABEL_436:
                    LODWORD(v480) = v408 + v268;
                    v269 = *(_QWORD *)(v446 + 528);
                    v270 = *(__int64 (**)())(*(_QWORD *)v269 + 880LL);
                    if ( v270 == sub_2DB1B20 )
                      goto LABEL_552;
                    v344 = v264;
                    if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64 *))v270)(v269, &v479) )
                      goto LABEL_552;
                    v271 = *((_QWORD *)v91 + 2);
                    v272 = *(_QWORD *)(v271 + 8);
                    if ( v272 == *(_QWORD *)(v271 + 32) + 320LL )
                      v272 = 0;
                    v466 = (__int64 *)v272;
                    v273 = sub_34E6760(
                             *(_QWORD **)(v271 + 112),
                             *(_QWORD *)(v271 + 112) + 8LL * *(unsigned int *)(v271 + 120),
                             (__int64 *)&v466);
                    v276 = v344;
                    if ( v277 != v273 )
                    {
                      sub_2E32F90(v275, (__int64)v273, v274);
                      v275 = *((_QWORD *)v91 + 2);
                      v276 = v344;
                    }
                    nh = v276;
                    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, unsigned __int64, _QWORD, _BYTE **, _QWORD))(**(_QWORD **)(v446 + 528) + 368LL))(
                      *(_QWORD *)(v446 + 528),
                      v275,
                      *(_QWORD *)(v165 + 32),
                      0,
                      v479,
                      (unsigned int)v480,
                      &v463,
                      0);
                    sub_2E33F80(
                      *((_QWORD *)v91 + 2),
                      *(_QWORD *)(v165 + 32),
                      (v353 * (unsigned __int64)v380 + 0x40000000) >> 31,
                      v278,
                      v279,
                      v280);
                    v281 = nh;
                    if ( (_QWORD *)v479 != v481 )
                    {
                      _libc_free(v479);
                      v281 = nh;
                    }
                    srcr = v281;
                    v282 = sub_34E6820(*((_QWORD *)v91 + 2), v252);
                    v283 = srcr;
                    if ( v282 )
                    {
LABEL_445:
                      v109 = v283;
                      sub_34E62C0(v446, *((_QWORD *)v91 + 2));
                      *(_BYTE *)v165 |= 1u;
                      goto LABEL_233;
                    }
LABEL_497:
                    srcs = v283;
                    sub_34E6930(*((_QWORD *)v91 + 2), v252, *(__int64 **)(v446 + 528));
                    v300 = *((_QWORD *)v91 + 2);
                    *v91 = *v91 & 0xBE | 1;
                    sub_34E62C0(v446, v300);
                    v93 = srcs;
                    *(_BYTE *)v165 |= 1u;
                    v109 = srcs;
                    goto LABEL_233;
                  }
                  v301 = v481;
                }
                v345 = v264;
                memcpy(v301, v266, v267);
                LODWORD(v268) = v480;
                v264 = v345;
                goto LABEL_436;
              }
              v299 = sub_34E6820(*((_QWORD *)v91 + 2), v252);
              v283 = v407;
              if ( v299 )
                goto LABEL_445;
              if ( *(_DWORD *)(v252 + 72) == 1
                && (*(_BYTE *)srca & 0x40) == 0
                && !*(_BYTE *)(v252 + 217)
                && !*(_QWORD *)(v252 + 224) )
              {
                sub_34EB320(v446, v91, (__int64)srca, 1);
                *v91 |= 1u;
                sub_34E62C0(v446, *((_QWORD *)v91 + 2));
                v93 = v407;
                *(_BYTE *)v165 |= 1u;
                *(_BYTE *)srca |= 1u;
                v109 = v407;
                goto LABEL_233;
              }
              goto LABEL_497;
            }
            v207 = qword_503B3A8;
            if ( (_BYTE)qword_503B3A8 )
              goto LABEL_111;
            v208 = *(_BYTE *)(v89 + 20);
            v209 = v208 >> 1;
            v210 = (v208 & 4) != 0;
            v211 = v209 & 1;
            srcd = *(void **)(v446 + 200);
            v212 = (__int64)srcd + 392 * *(int *)(*((_QWORD *)v91 + 3) + 24LL);
            v213 = (__int64)srcd + 392 * *(int *)(*((_QWORD *)v91 + 4) + 24LL);
            if ( *(_QWORD *)(v212 + 24) )
            {
              srcq = *(void **)(v212 + 24);
              v214 = sub_34EBFC0(v446, v91, v212, v213, v95, v96, v211, v210, (*(_BYTE *)v212 & 0x10) != 0, 0);
              v215 = (__int64)srcq;
              if ( !v214 )
                goto LABEL_111;
LABEL_292:
              v436 = v215;
              sub_2E33650(*((_QWORD *)v91 + 2), *(_QWORD *)(v212 + 16));
              sub_2E33650(*((_QWORD *)v91 + 2), *(_QWORD *)(v213 + 16));
              v216 = *(_QWORD *)(v446 + 200) + 392LL * *(int *)(v436 + 24);
              if ( (*(_BYTE *)v216 & 0x40) == 0 )
              {
                v217 = *(_QWORD *)(v216 + 16);
                v207 = 0;
                if ( !*(_BYTE *)(v217 + 217) )
                  v207 = *(_QWORD *)(v217 + 224) == 0;
              }
              srce = (void *)(*(_QWORD *)(v446 + 200) + 392LL * *(int *)(v436 + 24));
              v218 = sub_2E313E0(*((_QWORD *)v91 + 2));
              v222 = v436;
              v223 = (__int64)srce;
              if ( (v218 == *((_QWORD *)v91 + 2) + 48LL
                 || (v224 = *(__int64 (**)())(**(_QWORD **)(v446 + 528) + 920LL), v224 == sub_2DB1B30)
                 || (v296 = v224(), v222 = v436, v223 = (__int64)srce, !v296))
                && (v225 = *(_DWORD *)(v222 + 72), v225 <= 1) )
              {
                if ( (v225 & 1) != 0 )
                {
                  if ( !v207 )
                    goto LABEL_302;
                  v226 = **(_QWORD **)(v222 + 64);
                  if ( v226 != *(_QWORD *)(v212 + 16) && v226 != *(_QWORD *)(v213 + 16) )
                    goto LABEL_302;
                }
                else if ( !v207 )
                {
                  goto LABEL_302;
                }
                v438 = (_BYTE *)v223;
                sub_34EB320(v446, v91, v223, 1);
                *v438 |= 1u;
              }
              else
              {
LABEL_302:
                v437 = v222;
                sub_2E33F80(*((_QWORD *)v91 + 2), v222, 0x80000000, v219, v220, v221);
                sub_34E6930(*((_QWORD *)v91 + 2), v437, *(__int64 **)(v446 + 528));
                *v91 &= ~0x40u;
              }
            }
            else if ( (*(_BYTE *)v212 & 0x10) != 0 )
            {
              v234 = v95;
              v235 = (__int64)srcd + 392 * *(int *)(*((_QWORD *)v91 + 4) + 24LL);
              srcf = *(void **)(v213 + 24);
              if ( !(unsigned __int8)sub_34EBFC0(v446, v91, v212, v235, v234, v96, v211, v210, 1, srcf == 0) )
                goto LABEL_111;
              v215 = (__int64)srcf;
              if ( srcf )
                goto LABEL_292;
            }
            else if ( !(unsigned __int8)sub_34EBFC0(v446, v91, v212, v213, v95, v96, v211, v210, 0, 1) )
            {
              goto LABEL_111;
            }
            *(_BYTE *)v213 |= 1u;
            *(_BYTE *)v212 |= 1u;
            v98 = *((_QWORD *)v91 + 2);
            *v91 |= 1u;
            sub_34E62C0(v446, v98);
            goto LABEL_304;
          }
LABEL_88:
          j_j___libc_free_0(v89);
          v88 = v457;
        }
        while ( (unsigned __int64 *)v457 != v456 );
        m128i_i64 = (_QWORD *)v446;
LABEL_114:
        if ( v430 )
        {
          v111 = qword_503B8E8;
          *((_BYTE *)m128i_i64 + 625) = 1;
          if ( v111 == -1 || v111 > 0 )
            continue;
        }
        v112 = v456;
        v113 = v457;
        if ( v456 != (unsigned __int64 *)v457 )
        {
          v114 = v456;
          do
          {
            if ( *v114 )
              j_j___libc_free_0(*v114);
            ++v114;
          }
          while ( (unsigned __int64 *)v113 != v114 );
          v457 = (__int64)v112;
        }
        break;
      }
    }
    v115 = m128i_i64[25];
    v116 = m128i_i64[26];
    if ( v115 != v116 )
    {
      v117 = m128i_i64[25];
      do
      {
        v118 = *(_QWORD *)(v117 + 216);
        if ( v118 != v117 + 232 )
          _libc_free(v118);
        v119 = *(_QWORD *)(v117 + 40);
        if ( v119 != v117 + 56 )
          _libc_free(v119);
        v117 += 392;
      }
      while ( v116 != v117 );
      m128i_i64[26] = v115;
    }
    v120 = *((unsigned __int8 *)m128i_i64 + 625);
    if ( (_BYTE)v120 && (_BYTE)qword_503B1E8 )
    {
      sub_34BEDF0((__int64)&v479, 0, 0, (__int64)v459, m128i_i64[68], v320, 0);
      v284 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)a2[2] + 200LL))(a2[2]);
      sub_34C7080((__int64)&v479, a2, m128i_i64[66], v284, 0, 0);
      if ( v491 )
        _libc_free((unsigned __int64)v491);
      if ( v489 != v490 )
        _libc_free((unsigned __int64)v489);
      if ( v488 )
        j_j___libc_free_0(v488);
      sub_C7D6A0(v486, 16LL * v487, 8);
      if ( !BYTE4(v484) )
        _libc_free(v482);
      v285 = v480;
      v286 = v479;
      if ( v480 != v479 )
      {
        do
        {
          v287 = *(_QWORD *)(v286 + 16);
          if ( v287 )
            sub_B91220(v286 + 16, v287);
          v286 += 24LL;
        }
        while ( v285 != v286 );
        v286 = v479;
      }
      if ( v286 )
        j_j___libc_free_0(v286);
      v120 = *((unsigned __int8 *)m128i_i64 + 625);
    }
    v121 = v457;
    v122 = v456;
    v123 = v120 | v327;
    *((_BYTE *)m128i_i64 + 625) = v120 | v327;
    if ( (unsigned __int64 *)v121 != v122 )
    {
      do
      {
        if ( *v122 )
          j_j___libc_free_0(*v122);
        ++v122;
      }
      while ( (unsigned __int64 *)v121 != v122 );
      v122 = v456;
    }
    if ( v122 )
      j_j___libc_free_0((unsigned __int64)v122);
  }
  sub_C7D6A0(v460, 16LL * v462, 8);
  return v123;
}
