// Function: sub_17EEF60
// Address: 0x17eef60
//
__int64 __fastcall sub_17EEF60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 (__fastcall *a4)(__int64, __int64 *),
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64 *),
        __int64 a8)
{
  __int64 *v8; // r12
  char v9; // al
  int **v10; // r8
  unsigned __int64 v11; // r8
  _QWORD *v12; // r14
  __m128i *v13; // rsi
  void **v14; // r15
  void **v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  int **v19; // rbx
  __int64 (__fastcall *v20)(__int64); // rax
  __int64 v21; // r12
  unsigned __int64 v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // r13
  unsigned __int64 v25; // r14
  _QWORD *v26; // rax
  __int64 i; // r13
  __int64 v28; // r15
  unsigned __int64 v29; // r14
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // r15
  __int64 v34; // rbx
  __int64 *v35; // r12
  char *v36; // rax
  __int64 v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r15
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // rbx
  unsigned int j; // r12d
  __int64 v45; // r13
  unsigned __int64 v46; // r15
  __int64 v47; // r9
  __int64 v48; // r15
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdx
  char *v53; // rbx
  char *v54; // r15
  __int64 v55; // r14
  __int64 v56; // r13
  __int64 *v57; // rax
  __int64 *v58; // r12
  __int64 v59; // rcx
  __int64 *v60; // r14
  __int64 *v61; // rax
  __int64 v62; // rdx
  __int64 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rdi
  __int64 *v66; // rbx
  char *v67; // r14
  char *v68; // rbx
  __int64 v69; // rax
  __int64 v70; // rdi
  char *v71; // rdx
  char *v72; // r14
  __int64 *v73; // rax
  __int64 v74; // r10
  unsigned int v75; // edi
  __int64 v76; // r8
  _QWORD *v77; // rsi
  unsigned int v78; // r9d
  unsigned int v79; // ecx
  __int64 *v80; // rax
  __int64 v81; // r11
  _QWORD ******v82; // rcx
  _QWORD ******v83; // r10
  _QWORD *****v84; // rsi
  _QWORD ****v85; // r9
  _QWORD ***v86; // rbx
  _QWORD **v87; // r11
  _QWORD *v88; // r13
  _QWORD *****v89; // r12
  _QWORD ***v90; // rax
  unsigned int v91; // r10d
  __int64 *v92; // rax
  __int64 v93; // r11
  _QWORD *v94; // r8
  _QWORD *v95; // rsi
  _QWORD *****v96; // r9
  _QWORD ****v97; // r10
  _QWORD ***v98; // rbx
  _QWORD **v99; // r11
  _QWORD *v100; // r13
  _QWORD *****v101; // r12
  _QWORD ***v102; // rax
  unsigned int v103; // eax
  _BYTE *v104; // rdi
  __int64 v105; // rsi
  __int64 v106; // rdi
  unsigned int v107; // r13d
  unsigned __int64 v108; // r12
  int v109; // ebx
  __int64 v110; // rsi
  unsigned int v111; // ecx
  __int64 *v112; // rdx
  __int64 v113; // r9
  __int64 v114; // rax
  unsigned int v115; // r14d
  int k; // ecx
  size_t v117; // rsi
  size_t v118; // rsi
  size_t v119; // rdx
  __int64 v120; // rsi
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  int v124; // r9d
  const char *v125; // rdi
  size_t v126; // rdx
  const char *v127; // rax
  __int64 v128; // rdx
  __int64 v129; // r8
  _QWORD *v130; // r12
  _QWORD *v131; // r15
  __int64 v132; // rbx
  __int64 v133; // rdi
  _QWORD *v134; // rax
  __int64 v135; // r12
  _QWORD *v136; // rax
  __int64 v137; // rdi
  __int64 *v138; // rax
  __int64 v139; // rdx
  __int64 *v140; // r14
  unsigned __int64 v141; // rdi
  unsigned __int64 v142; // rdi
  __int64 v143; // rax
  _QWORD *v144; // r14
  __int64 v145; // r13
  _QWORD *v146; // r12
  __int64 v147; // rax
  __int64 *v148; // r15
  __int64 *v149; // r12
  __int64 v150; // r13
  __int64 *v151; // rbx
  __int64 *v152; // r14
  __int64 v153; // rdi
  __int64 v154; // rax
  __int64 v155; // rax
  void *v156; // rdi
  unsigned int v157; // eax
  __int64 v158; // rdx
  unsigned __int64 v159; // rdi
  __int64 v160; // rax
  __int64 v161; // rdi
  __int64 v162; // rdi
  unsigned __int64 *v163; // r12
  unsigned __int64 *v164; // rbx
  unsigned __int64 v165; // rdi
  unsigned __int64 *v166; // rax
  unsigned __int64 *v167; // rbx
  unsigned __int64 *v168; // r12
  unsigned __int64 v169; // rdi
  unsigned __int64 *v170; // rbx
  unsigned __int64 *v171; // r12
  unsigned __int64 v172; // rdi
  int v173; // eax
  __int64 v174; // r13
  _QWORD **v175; // rbx
  _QWORD **v176; // r12
  _QWORD *v177; // r14
  _QWORD *v178; // rdi
  _QWORD **v179; // rbx
  _QWORD **v180; // r12
  _QWORD *v181; // r14
  _QWORD *v182; // rdi
  _QWORD *v183; // r12
  _QWORD *v184; // rbx
  __int64 v185; // r13
  unsigned __int64 v186; // rdi
  unsigned __int64 v187; // rdi
  char *v188; // rbx
  char *v189; // r12
  char *v190; // rbx
  char *v191; // r12
  __int64 v192; // rax
  __int64 *v193; // rbx
  __int64 *m; // r12
  __int64 v195; // rdi
  __int64 *v196; // rdi
  __int64 *v197; // rbx
  __int64 *v198; // r12
  __int64 v199; // rdi
  _QWORD *v200; // rbx
  _QWORD *v201; // rdi
  __int64 v202; // rdx
  int v204; // edx
  int v205; // r10d
  __int64 *v206; // rax
  unsigned int v207; // edi
  _QWORD *v208; // rcx
  __int64 v209; // rsi
  __int64 v210; // r9
  unsigned int v211; // r8d
  unsigned int v212; // edx
  __int64 *v213; // rax
  __int64 v214; // r11
  _QWORD *v215; // rdx
  _QWORD ***********v216; // r9
  _QWORD **********v217; // r12
  _QWORD *********v218; // rcx
  _QWORD ********v219; // r13
  _QWORD *******v220; // r8
  _QWORD ******v221; // r10
  _QWORD *****v222; // r11
  _QWORD ****v223; // rdi
  _QWORD ***v224; // rax
  _QWORD *v225; // r11
  unsigned int v226; // r9d
  __int64 *v227; // rax
  __int64 v228; // r11
  _QWORD *v229; // r11
  _QWORD *v230; // rcx
  _QWORD **********v231; // r12
  _QWORD *********v232; // rsi
  _QWORD ********v233; // r13
  _QWORD *******v234; // r8
  _QWORD ******v235; // r9
  _QWORD *****v236; // r10
  _QWORD ****v237; // rdi
  _QWORD ***v238; // rax
  _QWORD *v239; // r10
  unsigned int v240; // eax
  __int64 v241; // rdx
  char *v242; // rsi
  __m128i v243; // rax
  char v244; // al
  signed __int64 *v245; // rdx
  __int64 v246; // r13
  _QWORD **v247; // rbx
  _QWORD **v248; // r12
  _QWORD *v249; // r14
  _QWORD *v250; // rdi
  _QWORD **v251; // rbx
  _QWORD **v252; // r12
  _QWORD *v253; // r14
  _QWORD *v254; // rdi
  _QWORD *v255; // r12
  _QWORD *v256; // rbx
  __int64 v257; // r13
  unsigned __int64 v258; // rdi
  unsigned __int64 v259; // rdi
  char *v260; // rbx
  char *v261; // r12
  char *v262; // rbx
  int v263; // eax
  int v264; // eax
  int v265; // ebx
  size_t v266; // rdx
  _QWORD *v267; // rax
  __int64 v268; // rcx
  char v269; // al
  __m128i *v270; // rdx
  _BYTE *v271; // rdi
  __int64 v272; // rsi
  __int64 v273; // rcx
  _QWORD *v274; // rdi
  __m128i v275; // rax
  __int64 v276; // r8
  char v277; // al
  __m128i *v278; // rdx
  const char *v279; // rax
  __int64 v280; // rdx
  __int64 v281; // r8
  _QWORD *v282; // rdx
  _QWORD *v283; // r12
  _QWORD *v284; // r15
  __int64 v285; // rdx
  char *v286; // rsi
  __int64 v287; // rdx
  char v288; // al
  __m128i *v289; // rdx
  __int64 v290; // r13
  char v291; // al
  bool v292; // zf
  __int64 v293; // rdx
  __int64 v294; // rax
  __int64 v295; // r15
  const char *v296; // rax
  size_t v297; // rdx
  __int64 v298; // rdi
  __int64 v299; // rdx
  int v300; // eax
  int v301; // r10d
  __m128i v302; // rax
  size_t v303; // rdx
  int v304; // eax
  __int64 v305; // rax
  __int64 v306; // r12
  const char *v307; // rax
  size_t v308; // rdx
  __int64 v309; // rax
  __int64 v310; // rdx
  __int64 v311; // rax
  int v312; // ebx
  __int64 v313; // rax
  __int64 *v314; // rdx
  __int64 v315; // rcx
  int v316; // r10d
  __int64 v317; // [rsp+20h] [rbp-5B0h]
  unsigned __int8 v320; // [rsp+4Fh] [rbp-581h]
  int **v321; // [rsp+50h] [rbp-580h]
  __int64 v322; // [rsp+60h] [rbp-570h]
  __int64 v324; // [rsp+78h] [rbp-558h]
  unsigned __int64 v325; // [rsp+80h] [rbp-550h]
  __int64 *v326; // [rsp+88h] [rbp-548h]
  __int64 v327; // [rsp+98h] [rbp-538h]
  __int64 v328; // [rsp+A0h] [rbp-530h]
  unsigned __int64 v329; // [rsp+A8h] [rbp-528h]
  __int64 *v330; // [rsp+B0h] [rbp-520h]
  __int64 *v331; // [rsp+C0h] [rbp-510h]
  __int64 v332; // [rsp+C8h] [rbp-508h]
  unsigned __int64 v333; // [rsp+D0h] [rbp-500h]
  bool v334; // [rsp+D8h] [rbp-4F8h]
  int v335; // [rsp+E8h] [rbp-4E8h]
  __int64 v336; // [rsp+E8h] [rbp-4E8h]
  __int64 v337; // [rsp+E8h] [rbp-4E8h]
  __int64 v338; // [rsp+F0h] [rbp-4E0h]
  unsigned __int64 src; // [rsp+F8h] [rbp-4D8h]
  __int64 v340; // [rsp+100h] [rbp-4D0h]
  __int64 *v341; // [rsp+100h] [rbp-4D0h]
  __int64 v342; // [rsp+100h] [rbp-4D0h]
  __int64 v343; // [rsp+100h] [rbp-4D0h]
  __int64 v344; // [rsp+100h] [rbp-4D0h]
  unsigned __int64 v345; // [rsp+108h] [rbp-4C8h]
  __int64 v346; // [rsp+108h] [rbp-4C8h]
  char v347; // [rsp+118h] [rbp-4B8h]
  __int64 v348; // [rsp+118h] [rbp-4B8h]
  int v349; // [rsp+118h] [rbp-4B8h]
  _QWORD v350[2]; // [rsp+120h] [rbp-4B0h] BYREF
  int **v351; // [rsp+130h] [rbp-4A0h] BYREF
  char v352; // [rsp+138h] [rbp-498h]
  __int64 *v353; // [rsp+140h] [rbp-490h] BYREF
  __int64 *v354; // [rsp+148h] [rbp-488h]
  __int64 *v355; // [rsp+150h] [rbp-480h]
  __int64 *v356; // [rsp+160h] [rbp-470h] BYREF
  __int64 *v357; // [rsp+168h] [rbp-468h]
  __int64 *v358; // [rsp+170h] [rbp-460h]
  __m128i v359; // [rsp+180h] [rbp-450h] BYREF
  char v360; // [rsp+190h] [rbp-440h]
  char v361; // [rsp+191h] [rbp-43Fh]
  __m128i v362; // [rsp+1A0h] [rbp-430h] BYREF
  __int16 v363; // [rsp+1B0h] [rbp-420h]
  __m128i v364[2]; // [rsp+1C0h] [rbp-410h] BYREF
  __m128i v365; // [rsp+1E0h] [rbp-3F0h] BYREF
  char v366; // [rsp+1F0h] [rbp-3E0h]
  char v367; // [rsp+1F1h] [rbp-3DFh]
  __m128i v368[2]; // [rsp+200h] [rbp-3D0h] BYREF
  __m128i v369; // [rsp+220h] [rbp-3B0h] BYREF
  __int16 v370; // [rsp+230h] [rbp-3A0h]
  __m128i v371; // [rsp+240h] [rbp-390h] BYREF
  __int64 v372; // [rsp+250h] [rbp-380h]
  __m128i v373; // [rsp+260h] [rbp-370h] BYREF
  __int64 v374; // [rsp+270h] [rbp-360h]
  __m128i v375; // [rsp+280h] [rbp-350h] BYREF
  __int64 v376; // [rsp+290h] [rbp-340h]
  __m128i v377; // [rsp+2A0h] [rbp-330h] BYREF
  _QWORD v378[2]; // [rsp+2B0h] [rbp-320h] BYREF
  __m128i v379; // [rsp+2C0h] [rbp-310h] BYREF
  __int64 v380; // [rsp+2D0h] [rbp-300h] BYREF
  __int64 v381; // [rsp+2D8h] [rbp-2F8h]
  __int64 v382; // [rsp+2E0h] [rbp-2F0h]
  void *s; // [rsp+2F0h] [rbp-2E0h] BYREF
  __int64 v384; // [rsp+2F8h] [rbp-2D8h]
  _QWORD *v385; // [rsp+300h] [rbp-2D0h]
  __int64 v386; // [rsp+308h] [rbp-2C8h]
  int v387; // [rsp+310h] [rbp-2C0h]
  __int64 v388; // [rsp+318h] [rbp-2B8h]
  _QWORD v389[2]; // [rsp+320h] [rbp-2B0h] BYREF
  signed __int64 v390; // [rsp+330h] [rbp-2A0h] BYREF
  size_t v391; // [rsp+338h] [rbp-298h] BYREF
  __int64 v392; // [rsp+340h] [rbp-290h] BYREF
  __int64 v393; // [rsp+348h] [rbp-288h]
  _QWORD *v394; // [rsp+350h] [rbp-280h]
  __int64 v395; // [rsp+358h] [rbp-278h]
  unsigned int v396; // [rsp+360h] [rbp-270h]
  __int64 *v397; // [rsp+370h] [rbp-260h]
  char v398; // [rsp+378h] [rbp-258h]
  int v399; // [rsp+37Ch] [rbp-254h]
  size_t n[2]; // [rsp+380h] [rbp-250h] BYREF
  _QWORD v401[2]; // [rsp+390h] [rbp-240h] BYREF
  __int64 *v402; // [rsp+3A0h] [rbp-230h]
  __int64 *v403; // [rsp+3A8h] [rbp-228h]
  __int64 v404; // [rsp+3B0h] [rbp-220h]
  unsigned __int64 v405; // [rsp+3B8h] [rbp-218h]
  unsigned __int64 v406; // [rsp+3C0h] [rbp-210h]
  unsigned __int64 *v407; // [rsp+3C8h] [rbp-208h]
  unsigned int v408; // [rsp+3D0h] [rbp-200h]
  char v409; // [rsp+3D8h] [rbp-1F8h] BYREF
  unsigned __int64 *v410; // [rsp+3F8h] [rbp-1D8h]
  unsigned int v411; // [rsp+400h] [rbp-1D0h]
  __int64 v412; // [rsp+408h] [rbp-1C8h] BYREF
  __int64 *v413; // [rsp+420h] [rbp-1B0h] BYREF
  __int64 v414; // [rsp+428h] [rbp-1A8h]
  __int64 *v415; // [rsp+430h] [rbp-1A0h]
  __int64 *v416; // [rsp+438h] [rbp-198h]
  void **p_s; // [rsp+440h] [rbp-190h]
  char *v418; // [rsp+448h] [rbp-188h]
  char *v419; // [rsp+450h] [rbp-180h]
  char *v420; // [rsp+458h] [rbp-178h]
  __int64 *v421; // [rsp+460h] [rbp-170h]
  __int64 v422; // [rsp+468h] [rbp-168h]
  __int64 v423; // [rsp+470h] [rbp-160h]
  int v424; // [rsp+478h] [rbp-158h]
  __int64 v425; // [rsp+480h] [rbp-150h]
  __int64 v426; // [rsp+488h] [rbp-148h]
  __int64 v427; // [rsp+490h] [rbp-140h]
  __int64 *v428; // [rsp+498h] [rbp-138h]
  __int64 v429; // [rsp+4A0h] [rbp-130h]
  __int64 v430; // [rsp+4A8h] [rbp-128h]
  __int64 v431; // [rsp+4B0h] [rbp-120h]
  __int64 v432; // [rsp+4B8h] [rbp-118h]
  __int64 v433; // [rsp+4C0h] [rbp-110h]
  __int64 v434; // [rsp+4C8h] [rbp-108h]
  __int64 v435; // [rsp+4D0h] [rbp-100h]
  __int64 v436; // [rsp+4D8h] [rbp-F8h]
  void *dest; // [rsp+4E0h] [rbp-F0h] BYREF
  size_t v438; // [rsp+4E8h] [rbp-E8h]
  _QWORD v439[3]; // [rsp+4F0h] [rbp-E0h] BYREF
  unsigned __int64 v440; // [rsp+508h] [rbp-C8h] BYREF
  __int64 *v441; // [rsp+510h] [rbp-C0h] BYREF
  char *v442; // [rsp+518h] [rbp-B8h]
  char *v443; // [rsp+520h] [rbp-B0h]
  __int64 v444; // [rsp+528h] [rbp-A8h]
  __int64 v445; // [rsp+530h] [rbp-A0h]
  _QWORD *v446; // [rsp+538h] [rbp-98h]
  __int64 v447; // [rsp+540h] [rbp-90h]
  unsigned int v448; // [rsp+548h] [rbp-88h]
  char v449; // [rsp+550h] [rbp-80h]
  __int64 v450; // [rsp+558h] [rbp-78h]
  __int64 *v451; // [rsp+560h] [rbp-70h]
  __int64 v452; // [rsp+568h] [rbp-68h]
  __int64 v453; // [rsp+570h] [rbp-60h]
  __int64 v454; // [rsp+578h] [rbp-58h]
  __int64 v455; // [rsp+580h] [rbp-50h]
  __int64 v456; // [rsp+588h] [rbp-48h]
  __int64 v457; // [rsp+590h] [rbp-40h]
  int v458; // [rsp+598h] [rbp-38h]

  v8 = *(__int64 **)a1;
  v350[0] = a2;
  v350[1] = a3;
  LOWORD(v415) = 261;
  v413 = v350;
  sub_393B110(&v351, &v413);
  v9 = v352;
  v352 &= ~2u;
  if ( (v9 & 1) == 0 )
  {
    v371.m128i_i64[0] = 1;
    v413 = 0;
    sub_14ECA90((__int64 *)&v413);
    v19 = v351;
    v371.m128i_i64[0] = 0;
    v321 = v351;
    sub_14ECA90(v371.m128i_i64);
    v351 = 0;
    if ( !v19 )
    {
LABEL_538:
      v13 = (__m128i *)&v413;
      v390 = (signed __int64)"Cannot get PGOReader";
      n[0] = (size_t)&v390;
      v391 = 20;
      LOWORD(v401[0]) = 261;
      v413 = (__int64 *)&unk_49ECF40;
      v414 = 18;
      v415 = (__int64 *)v350[0];
      v416 = (__int64 *)n;
      sub_16027F0((__int64)v8, (__int64)&v413);
      v320 = 0;
      goto LABEL_331;
    }
    v20 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v321 + 4);
    if ( v20 == sub_17E1DB0 )
      v320 = (*(__int64 (__fastcall **)(int *))(*(_QWORD *)v321[4] + 64LL))(v321[4]);
    else
      v320 = v20((__int64)v321);
    if ( !v320 )
    {
      v13 = (__m128i *)&v413;
      n[0] = (size_t)"Not an IR level instrumentation profile";
      LOWORD(v401[0]) = 259;
      v414 = 18;
      v413 = (__int64 *)&unk_49ECF40;
      v415 = (__int64 *)v350[0];
      v416 = (__int64 *)n;
      sub_16027F0((__int64)v8, (__int64)&v413);
LABEL_330:
      (*((void (__fastcall **)(int **, __m128i *))*v321 + 1))(v321, v13);
      goto LABEL_331;
    }
    v384 = 1;
    s = v389;
    v385 = 0;
    v386 = 0;
    v387 = 1065353216;
    v388 = 0;
    v389[0] = 0;
    v322 = a1 + 24;
    if ( byte_4FA5660 )
    {
      v21 = *(_QWORD *)(a1 + 32);
      if ( v21 != v322 )
      {
        while ( v21 )
        {
          v22 = *(_QWORD *)(v21 - 8);
          if ( v22 )
          {
            v23 = (_QWORD *)sub_22077B0(24);
            if ( v23 )
              *v23 = 0;
            v23[1] = v22;
            v23[2] = v21 - 56;
            sub_17ED2A0(&s, 0, v23 + 1, v22, (__int64)v23);
          }
          v21 = *(_QWORD *)(v21 + 8);
          if ( v21 == v322 )
            goto LABEL_28;
        }
LABEL_583:
        BUG();
      }
LABEL_28:
      v24 = *(_QWORD *)(a1 + 16);
      if ( a1 + 8 != v24 )
      {
        while ( v24 )
        {
          v25 = *(_QWORD *)(v24 - 8);
          if ( v25 )
          {
            v26 = (_QWORD *)sub_22077B0(24);
            if ( v26 )
              *v26 = 0;
            v26[1] = v25;
            v26[2] = v24 - 56;
            sub_17ED2A0(&s, 0, v26 + 1, v25, (__int64)v26);
          }
          v24 = *(_QWORD *)(v24 + 8);
          if ( a1 + 8 == v24 )
            goto LABEL_35;
        }
        goto LABEL_583;
      }
LABEL_35:
      for ( i = *(_QWORD *)(a1 + 48); a1 + 40 != i; i = *(_QWORD *)(i + 8) )
      {
        v28 = i - 48;
        if ( !i )
          v28 = 0;
        v29 = sub_15E4F10(v28);
        if ( v29 )
        {
          v30 = (_QWORD *)sub_22077B0(24);
          if ( v30 )
            *v30 = 0;
          v30[1] = v29;
          v30[2] = v28;
          sub_17ED2A0(&s, 0, v30 + 1, v29, (__int64)v30);
        }
      }
    }
    v353 = 0;
    v354 = 0;
    v31 = *(_QWORD *)(a1 + 32);
    v355 = 0;
    v356 = 0;
    v357 = 0;
    v358 = 0;
    v328 = v31;
    if ( v31 == v322 )
      goto LABEL_317;
LABEL_44:
    v32 = 0;
    if ( v328 )
      v32 = v328 - 56;
    v326 = (__int64 *)v32;
    v33 = (__int64 *)v32;
    if ( sub_15E4F60(v32) )
      goto LABEL_316;
    v34 = a4(a5, v33);
    v35 = (__int64 *)a7(a8, v33);
    sub_1AAD1B0(v33, v34, v35);
    v413 = v33;
    v415 = v35;
    v414 = a1;
    v416 = v33;
    p_s = &s;
    v418 = 0;
    v419 = 0;
    v420 = 0;
    v36 = (char *)sub_22077B0(48);
    v418 = v36;
    v420 = v36 + 48;
    if ( v36 )
    {
      *(_QWORD *)v36 = 0;
      *((_QWORD *)v36 + 1) = 0;
      *((_QWORD *)v36 + 2) = 0;
    }
    *((_QWORD *)v36 + 3) = 0;
    *((_QWORD *)v36 + 4) = 0;
    *((_QWORD *)v36 + 5) = 0;
    v419 = v36 + 48;
    v421 = v326;
    v428 = v326;
    dest = v439;
    v422 = 0;
    v441 = v416;
    v423 = 0;
    v424 = 0;
    v425 = 0;
    v426 = 0;
    v427 = 0;
    v429 = 0;
    v430 = 0;
    v431 = 0;
    v432 = 0;
    v433 = 0;
    v434 = 0;
    v435 = 0;
    v436 = 0;
    v438 = 0;
    LOBYTE(v439[0]) = 0;
    v440 = 0;
    v442 = 0;
    v443 = 0;
    v444 = 0;
    v445 = 0;
    v446 = 0;
    v447 = 0;
    v448 = 0;
    v449 = 0;
    v450 = v34;
    v451 = v35;
    v37 = v416[10];
    v325 = 2;
    v38 = v37 - 24;
    if ( !v37 )
      v38 = 0;
    v327 = v38;
    if ( v35 )
      v325 = sub_1368DC0((__int64)v35);
    v317 = sub_17E4990((__int64)&v441, 0, v327, v325);
    v39 = sub_157EBA0(v327);
    if ( !v39 || !(unsigned int)sub_15F4D60(v39) )
    {
      sub_17E4990((__int64)&v441, v327, 0, v325);
      goto LABEL_88;
    }
    v330 = v441 + 9;
    v331 = (__int64 *)v441[10];
    if ( v331 == v441 + 9 )
    {
      v332 = 0;
      v345 = 0;
      v333 = 0;
      v340 = 0;
    }
    else
    {
      v333 = 0;
      v345 = 0;
      v332 = 0;
      v324 = 0;
      v340 = 0;
      v329 = 0;
      do
      {
        v41 = (__int64)(v331 - 3);
        if ( !v331 )
          v41 = 0;
        src = 2;
        v42 = sub_157EBA0(v41);
        if ( v451 )
          src = sub_1368AA0(v451, v41);
        v335 = sub_15F4D60(v42);
        if ( v335 )
        {
          v338 = v41;
          v43 = 2;
          v334 = v327 == v41;
          for ( j = 0; j != v335; ++j )
          {
            v45 = sub_15F4DF0(v42, j);
            v347 = sub_137E040(v42, j, 0);
            v46 = src;
            if ( v347 )
            {
              v47 = -1;
              if ( src <= 0x4189374BC6A7EELL )
                v47 = 1000 * src;
              v46 = v47;
            }
            if ( v450 )
            {
              LODWORD(n[0]) = sub_13774B0(v450, v338, v45);
              v43 = sub_16AF780((unsigned int *)n, v46);
            }
            v48 = sub_17E4990((__int64)&v441, v338, v45, v43);
            *(_BYTE *)(v48 + 26) = v347;
            v49 = v345;
            if ( v43 > v345 )
            {
              if ( v334 )
                v49 = v43;
              v345 = v49;
              v50 = v340;
              if ( v334 )
                v50 = v48;
              v340 = v50;
            }
            v51 = sub_157EBA0(v45);
            if ( v51 && !(unsigned int)sub_15F4D60(v51) )
            {
              v52 = v333;
              if ( v43 > v333 )
                v52 = v43;
              else
                v48 = v332;
              v332 = v48;
              v333 = v52;
            }
          }
        }
        else
        {
          v449 = 1;
          v40 = sub_17E4990((__int64)&v441, v41, 0, src);
          if ( v329 < src )
          {
            v329 = src;
            v324 = v40;
          }
        }
        v331 = (__int64 *)v331[1];
      }
      while ( v330 != v331 );
      if ( v325 >= v329 && 2 * v325 < 3 * v329 )
      {
        *(_QWORD *)(v317 + 16) = v329;
        *(_QWORD *)(v324 + 16) = v325 + 1;
      }
      if ( v345 < v333 )
      {
LABEL_88:
        v53 = v443;
        v54 = v442;
        if ( v443 - v442 <= 0 )
        {
LABEL_458:
          v58 = 0;
          v56 = 0;
          sub_17E4160(v54, v53);
        }
        else
        {
          v55 = (v443 - v442) >> 3;
          while ( 1 )
          {
            v56 = v55;
            v57 = (__int64 *)sub_2207800(8 * v55, &unk_435FF63);
            v58 = v57;
            if ( v57 )
              break;
            v55 >>= 1;
            if ( !v55 )
              goto LABEL_458;
          }
          v59 = v55;
          v60 = &v57[v56];
          *v57 = *(_QWORD *)v54;
          v61 = v57 + 1;
          *(_QWORD *)v54 = 0;
          if ( v60 == v58 + 1 )
          {
            v63 = v58;
          }
          else
          {
            do
            {
              v62 = *(v61 - 1);
              *(v61++ - 1) = 0;
              *(v61 - 1) = v62;
            }
            while ( v60 != v61 );
            v63 = &v58[v56 - 1];
          }
          v64 = *v63;
          *v63 = 0;
          v65 = *(_QWORD *)v54;
          *(_QWORD *)v54 = v64;
          if ( v65 )
          {
            v348 = v59;
            j_j___libc_free_0(v65, 40);
            sub_17E5DC0(v54, v53, v58, v348);
          }
          else
          {
            sub_17E5DC0(v54, v53, v58, v59);
          }
          v66 = v58;
          do
          {
            if ( *v66 )
              j_j___libc_free_0(*v66, 40);
            ++v66;
          }
          while ( v60 != v66 );
        }
        j_j___libc_free_0(v58, v56 * 8);
        v67 = v443;
        v68 = v442;
        if ( v442 == v443 )
          goto LABEL_152;
        while ( 1 )
        {
          v69 = *(_QWORD *)v68;
          if ( !*(_BYTE *)(*(_QWORD *)v68 + 25LL) )
          {
            if ( *(_BYTE *)(v69 + 26) )
            {
              v70 = *(_QWORD *)(v69 + 8);
              if ( v70 )
              {
                if ( sub_157F790(v70) )
                  break;
              }
            }
          }
LABEL_105:
          v68 += 8;
          if ( v67 == v68 )
          {
            v71 = v442;
            v72 = v443;
            if ( v442 != v443 )
            {
              while ( 1 )
              {
                v73 = *(__int64 **)v71;
                if ( !*(_BYTE *)(*(_QWORD *)v71 + 25LL) )
                {
                  v74 = *v73;
                  if ( v449 || v74 )
                    break;
                }
LABEL_151:
                v71 += 8;
                if ( v72 == v71 )
                  goto LABEL_152;
              }
              v75 = v448;
              v76 = v73[1];
              v77 = v446;
              if ( v448 )
              {
                v78 = v448 - 1;
                v79 = (v448 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
                v80 = &v446[2 * v79];
                v81 = *v80;
                if ( v74 == *v80 )
                {
LABEL_112:
                  v82 = (_QWORD ******)v80[1];
                  v83 = (_QWORD ******)*v82;
                  if ( *v82 != v82 )
                    goto LABEL_113;
LABEL_130:
                  v91 = v78 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                  v92 = &v77[2 * v91];
                  v93 = *v92;
                  if ( v76 == *v92 )
                  {
LABEL_131:
                    v94 = (_QWORD *)v92[1];
                    v95 = (_QWORD *)*v94;
                    if ( v94 != (_QWORD *)*v94 )
                    {
                      v96 = (_QWORD *****)*v95;
                      if ( v95 != (_QWORD *)*v95 )
                      {
                        v97 = *v96;
                        if ( v96 != *v96 )
                        {
                          v98 = *v97;
                          if ( v97 != *v97 )
                          {
                            v99 = *v98;
                            if ( v98 != *v98 )
                            {
                              v100 = *v99;
                              if ( v99 != *v99 )
                              {
                                v101 = (_QWORD *****)*v100;
                                if ( v100 != (_QWORD *)*v100 )
                                {
                                  if ( v101 != *v101 )
                                  {
                                    v102 = sub_17E2800(*v101);
                                    *v101 = (_QWORD ****)v102;
                                    v101 = (_QWORD *****)v102;
                                  }
                                  *v100 = v101;
                                }
                                *v99 = v101;
                                v99 = v101;
                              }
                              *v98 = v99;
                            }
                            *v97 = (_QWORD ***)v99;
                            v97 = (_QWORD ****)v99;
                          }
                          *v96 = v97;
                        }
                        *v95 = v97;
                        v95 = v97;
                      }
                      *v94 = v95;
                    }
                    if ( v95 != v82 )
                    {
                      v103 = *((_DWORD *)v95 + 3);
                      if ( *((_DWORD *)v82 + 3) >= v103 )
                      {
                        *v95 = v82;
                        if ( v103 == *((_DWORD *)v82 + 3) )
                          *((_DWORD *)v82 + 3) = v103 + 1;
                      }
                      else
                      {
                        *v82 = (_QWORD *****)v95;
                      }
                      *(_BYTE *)(*(_QWORD *)v71 + 24LL) = 1;
                    }
                    goto LABEL_151;
                  }
                  v264 = 1;
                  while ( v93 != -8 )
                  {
                    v265 = v264 + 1;
                    v91 = v78 & (v264 + v91);
                    v92 = &v77[2 * v91];
                    v93 = *v92;
                    if ( v76 == *v92 )
                      goto LABEL_131;
                    v264 = v265;
                  }
LABEL_343:
                  v92 = &v77[2 * v75];
                  goto LABEL_131;
                }
                v263 = 1;
                while ( v81 != -8 )
                {
                  v312 = v263 + 1;
                  v79 = v78 & (v263 + v79);
                  v80 = &v446[2 * v79];
                  v81 = *v80;
                  if ( *v80 == v74 )
                    goto LABEL_112;
                  v263 = v312;
                }
                v82 = (_QWORD ******)v446[2 * v448 + 1];
                v83 = (_QWORD ******)*v82;
                if ( *v82 == v82 )
                {
LABEL_129:
                  v82 = v83;
                  goto LABEL_130;
                }
              }
              else
              {
                v82 = (_QWORD ******)v446[1];
                v83 = (_QWORD ******)*v82;
                if ( *v82 == v82 )
                  goto LABEL_343;
              }
LABEL_113:
              v84 = *v83;
              if ( *v83 != v83 )
              {
                v85 = *v84;
                if ( v84 != *v84 )
                {
                  v86 = *v85;
                  if ( v85 != *v85 )
                  {
                    v87 = *v86;
                    if ( v86 != *v86 )
                    {
                      v88 = *v87;
                      if ( v87 != *v87 )
                      {
                        v89 = (_QWORD *****)*v88;
                        if ( v88 != (_QWORD *)*v88 )
                        {
                          if ( v89 != *v89 )
                          {
                            v90 = sub_17E2800(*v89);
                            *v89 = (_QWORD ****)v90;
                            v89 = (_QWORD *****)v90;
                          }
                          *v88 = v89;
                        }
                        *v87 = v89;
                        v87 = v89;
                      }
                      *v86 = v87;
                    }
                    *v85 = (_QWORD ***)v87;
                    v85 = (_QWORD ****)v87;
                  }
                  *v84 = v85;
                }
                *v83 = (_QWORD *****)v85;
                v83 = (_QWORD ******)v85;
              }
              *v82 = v83;
              v75 = v448;
              v77 = v446;
              if ( !v448 )
              {
                v82 = v83;
                goto LABEL_343;
              }
              v78 = v448 - 1;
              goto LABEL_129;
            }
LABEL_152:
            sub_1695660((__int64)n, (__int64)v416, 0);
            v104 = dest;
            if ( (_QWORD *)n[0] == v401 )
            {
              v266 = n[1];
              if ( n[1] )
              {
                if ( n[1] == 1 )
                  *(_BYTE *)dest = v401[0];
                else
                  memcpy(dest, v401, n[1]);
                v266 = n[1];
                v104 = dest;
              }
              v438 = v266;
              v104[v266] = 0;
              v104 = (_BYTE *)n[0];
            }
            else
            {
              if ( dest == v439 )
              {
                dest = (void *)n[0];
                v438 = n[1];
                v439[0] = v401[0];
              }
              else
              {
                v105 = v439[0];
                dest = (void *)n[0];
                v438 = n[1];
                v439[0] = v401[0];
                if ( v104 )
                {
                  n[0] = (size_t)v104;
                  v401[0] = v105;
                  goto LABEL_156;
                }
              }
              n[0] = (size_t)v401;
              v104 = v401;
            }
LABEL_156:
            n[1] = 0;
            *v104 = 0;
            if ( (_QWORD *)n[0] != v401 )
              j_j___libc_free_0(n[0], v401[0] + 1LL);
            n[0] = 0;
            n[1] = 0;
            v401[0] = 0;
            LODWORD(v390) = -1;
            v341 = v416 + 9;
            if ( (__int64 *)v416[10] == v416 + 9 )
            {
              v119 = 0;
              v118 = 0;
            }
            else
            {
              v346 = v416[10];
              do
              {
                v106 = v346 - 24;
                if ( !v346 )
                  v106 = 0;
                v107 = 0;
                v108 = sub_157EBA0(v106);
                v109 = sub_15F4D60(v108);
                if ( v109 )
                {
                  do
                  {
                    v110 = sub_15F4DF0(v108, v107);
                    if ( v448 )
                    {
                      v111 = (v448 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
                      v112 = &v446[2 * v111];
                      v113 = *v112;
                      if ( v110 == *v112 )
                      {
LABEL_165:
                        if ( v112 != &v446[2 * v448] )
                        {
                          v114 = v112[1];
                          if ( v114 )
                          {
                            v115 = *(_DWORD *)(v114 + 8);
                            for ( k = 0; k != 32; k += 8 )
                            {
                              v117 = n[1];
                              v379.m128i_i8[0] = v115 >> k;
                              if ( n[1] == v401[0] )
                              {
                                v349 = k;
                                sub_17EB120((__int64)n, (const void *)n[1], v379.m128i_i8);
                                k = v349;
                              }
                              else
                              {
                                if ( n[1] )
                                {
                                  *(_BYTE *)n[1] = v115 >> k;
                                  v117 = n[1];
                                }
                                n[1] = v117 + 1;
                              }
                            }
                          }
                        }
                      }
                      else
                      {
                        v204 = 1;
                        while ( v113 != -8 )
                        {
                          v205 = v204 + 1;
                          v111 = (v448 - 1) & (v204 + v111);
                          v112 = &v446[2 * v111];
                          v113 = *v112;
                          if ( v110 == *v112 )
                            goto LABEL_165;
                          v204 = v205;
                        }
                      }
                    }
                    ++v107;
                  }
                  while ( v109 != v107 );
                }
                v346 = *(_QWORD *)(v346 + 8);
              }
              while ( v341 != (__int64 *)v346 );
              v118 = n[0];
              v119 = n[1] - n[0];
            }
            sub_3946250(&v390, v118, v119);
            v440 = ((__int64)(*((_QWORD *)v418 + 1) - *(_QWORD *)v418) >> 3 << 48)
                 | (unsigned int)v390
                 | ((unsigned __int64)(unsigned int)v422 << 56)
                 | ((v443 - v442) >> 3 << 32);
            if ( n[0] )
              j_j___libc_free_0(n[0], v401[0] - n[0]);
            if ( !v386 || !(unsigned __int8)sub_17E85A0((__int64)v416, p_s) )
              goto LABEL_179;
            v242 = (char *)sub_1649960((__int64)v416);
            if ( v242 )
            {
              v377.m128i_i64[0] = (__int64)v378;
              sub_17E2210(v377.m128i_i64, v242, (__int64)&v242[v241]);
            }
            else
            {
              LOBYTE(v378[0]) = 0;
              v377.m128i_i64[0] = (__int64)v378;
              v377.m128i_i64[1] = 0;
            }
            LOWORD(v392) = 267;
            v390 = (signed __int64)&v440;
            v243.m128i_i64[0] = (__int64)sub_1649960((__int64)v416);
            v373 = v243;
            v375.m128i_i64[0] = (__int64)&v373;
            v375.m128i_i64[1] = (__int64)".";
            v244 = v392;
            LOWORD(v376) = 773;
            if ( (_BYTE)v392 )
            {
              if ( (_BYTE)v392 == 1 )
              {
                *(__m128i *)n = _mm_loadu_si128(&v375);
                v401[0] = v376;
              }
              else
              {
                v245 = (signed __int64 *)v390;
                if ( BYTE1(v392) != 1 )
                {
                  v245 = &v390;
                  v244 = 2;
                }
                n[1] = (size_t)v245;
                n[0] = (size_t)&v375;
                LOBYTE(v401[0]) = 2;
                BYTE1(v401[0]) = v244;
              }
            }
            else
            {
              LOWORD(v401[0]) = 256;
            }
            sub_16E2FC0(v379.m128i_i64, (__int64)n);
            n[0] = (size_t)&v379;
            LOWORD(v401[0]) = 260;
            sub_164B780((__int64)v416, (__int64 *)n);
            LOWORD(v401[0]) = 260;
            n[0] = (size_t)&v377;
            sub_15E5880(4, (__int64)n, v416);
            v390 = (signed __int64)&v392;
            LOWORD(v374) = 267;
            v373.m128i_i64[0] = (__int64)&v440;
            sub_17E2330(&v390, dest, (__int64)dest + v438);
            if ( v391 == 0x3FFFFFFFFFFFFFFFLL )
              sub_4262D8((__int64)"basic_string::append");
            sub_2241490(&v390, ".", 1, v268);
            v269 = v374;
            if ( (_BYTE)v374 )
            {
              if ( (_BYTE)v374 == 1 )
              {
                v375.m128i_i64[0] = (__int64)&v390;
                LOWORD(v376) = 260;
              }
              else
              {
                v270 = (__m128i *)v373.m128i_i64[0];
                if ( BYTE1(v374) != 1 )
                {
                  v270 = &v373;
                  v269 = 2;
                }
                v375.m128i_i64[0] = (__int64)&v390;
                v375.m128i_i64[1] = (__int64)v270;
                LOBYTE(v376) = 4;
                BYTE1(v376) = v269;
              }
            }
            else
            {
              LOWORD(v376) = 256;
            }
            sub_16E2FC0((__int64 *)n, (__int64)&v375);
            v271 = dest;
            if ( (_QWORD *)n[0] == v401 )
            {
              v303 = n[1];
              if ( n[1] )
              {
                if ( n[1] == 1 )
                  *(_BYTE *)dest = v401[0];
                else
                  memcpy(dest, v401, n[1]);
                v303 = n[1];
                v271 = dest;
              }
              v438 = v303;
              v271[v303] = 0;
              v271 = (_BYTE *)n[0];
              goto LABEL_490;
            }
            if ( dest == v439 )
            {
              dest = (void *)n[0];
              v438 = n[1];
              v439[0] = v401[0];
            }
            else
            {
              v272 = v439[0];
              dest = (void *)n[0];
              v438 = n[1];
              v439[0] = v401[0];
              if ( v271 )
              {
                n[0] = (size_t)v271;
                v401[0] = v272;
LABEL_490:
                n[1] = 0;
                *v271 = 0;
                if ( (_QWORD *)n[0] != v401 )
                  j_j___libc_free_0(n[0], v401[0] + 1LL);
                if ( (__int64 *)v390 != &v392 )
                  j_j___libc_free_0(v390, v392 + 1);
                v273 = v416[6];
                if ( v273 )
                {
                  v274 = (_QWORD *)v416[6];
                  v336 = v416[5];
                  LOWORD(v376) = 267;
                  v368[0].m128i_i64[0] = v273;
                  v342 = v273;
                  v375.m128i_i64[0] = (__int64)&v440;
                  v275.m128i_i64[0] = sub_1580C70(v274);
                  v276 = v336;
                  v371 = v275;
                  LOWORD(v374) = 773;
                  v373.m128i_i64[0] = (__int64)&v371;
                  v373.m128i_i64[1] = (__int64)".";
                  v277 = v376;
                  if ( (_BYTE)v376 )
                  {
                    if ( (_BYTE)v376 == 1 )
                    {
                      *(__m128i *)n = _mm_loadu_si128(&v373);
                      v401[0] = v374;
                    }
                    else
                    {
                      v278 = (__m128i *)v375.m128i_i64[0];
                      if ( BYTE1(v376) != 1 )
                      {
                        v278 = &v375;
                        v277 = 2;
                      }
                      n[1] = (size_t)v278;
                      LOBYTE(v401[0]) = 2;
                      n[0] = (size_t)&v373;
                      BYTE1(v401[0]) = v277;
                    }
                  }
                  else
                  {
                    LOWORD(v401[0]) = 256;
                  }
                  v337 = v342;
                  v343 = v276;
                  sub_16E2FC0(&v390, (__int64)n);
                  v344 = sub_1633B90(v343, (void *)v390, v391);
                  *(_DWORD *)(v344 + 8) = *(_DWORD *)(v337 + 8);
                  v283 = (_QWORD *)sub_17E8510(p_s, (unsigned __int64 *)v368);
                  if ( v283 != v282 )
                  {
                    v284 = v282;
                    do
                    {
                      while ( 1 )
                      {
                        v290 = v283[2];
                        v291 = *(_BYTE *)(v290 + 16);
                        if ( v291 == 1 )
                          break;
                        if ( v291 )
                        {
                          MEMORY[0x30] = v344;
                          BUG();
                        }
                        *(_QWORD *)(v290 + 48) = v344;
                        v283 = (_QWORD *)*v283;
                        if ( v284 == v283 )
                          goto LABEL_521;
                      }
                      v286 = (char *)sub_1649960(v283[2]);
                      n[0] = (size_t)v401;
                      if ( v286 )
                      {
                        sub_17E2210((__int64 *)n, v286, (__int64)&v286[v285]);
                      }
                      else
                      {
                        n[1] = 0;
                        LOBYTE(v401[0]) = 0;
                      }
                      LOWORD(v374) = 267;
                      v373.m128i_i64[0] = (__int64)&v440;
                      v369.m128i_i64[0] = (__int64)sub_1649960(v290);
                      LOWORD(v372) = 773;
                      v369.m128i_i64[1] = v287;
                      v371.m128i_i64[0] = (__int64)&v369;
                      v371.m128i_i64[1] = (__int64)".";
                      v288 = v374;
                      if ( (_BYTE)v374 )
                      {
                        if ( (_BYTE)v374 == 1 )
                        {
                          v375 = _mm_loadu_si128(&v371);
                          v376 = v372;
                        }
                        else
                        {
                          v289 = (__m128i *)v373.m128i_i64[0];
                          if ( BYTE1(v374) != 1 )
                          {
                            v289 = &v373;
                            v288 = 2;
                          }
                          v375.m128i_i64[1] = (__int64)v289;
                          LOBYTE(v376) = 2;
                          v375.m128i_i64[0] = (__int64)&v371;
                          BYTE1(v376) = v288;
                        }
                      }
                      else
                      {
                        LOWORD(v376) = 256;
                      }
                      sub_164B780(v290, v375.m128i_i64);
                      LOWORD(v376) = 260;
                      v375.m128i_i64[0] = (__int64)n;
                      sub_15E5880(4, (__int64)&v375, (__int64 *)v290);
                      if ( (_QWORD *)n[0] != v401 )
                        j_j___libc_free_0(n[0], v401[0] + 1LL);
                      v283 = (_QWORD *)*v283;
                    }
                    while ( v284 != v283 );
                  }
LABEL_521:
                  if ( (__int64 *)v390 != &v392 )
                    j_j___libc_free_0(v390, v392 + 1);
                  if ( (__int64 *)v379.m128i_i64[0] != &v380 )
                    j_j___libc_free_0(v379.m128i_i64[0], v380 + 1);
                  if ( (_QWORD *)v377.m128i_i64[0] != v378 )
                    j_j___libc_free_0(v377.m128i_i64[0], v378[0] + 1LL);
                }
                else
                {
                  v313 = sub_1633B90(v416[5], (void *)v379.m128i_i64[0], v379.m128i_u64[1]);
                  v314 = v416;
                  v315 = v313;
                  LOBYTE(v313) = v416[4] & 0xF0 | 3;
                  *((_BYTE *)v416 + 32) = v313;
                  if ( (v313 & 0x30) != 0 )
                    *((_BYTE *)v314 + 33) |= 0x40u;
                  v314[6] = v315;
                  sub_2240A30(&v379);
                  sub_2240A30(&v377);
                }
LABEL_179:
                v120 = (__int64)v321;
                v452 = 0;
                v453 = 0;
                v454 = 0;
                v455 = 0;
                v456 = 0;
                v457 = 0;
                v458 = 0;
                if ( (unsigned __int8)sub_17E8B00((__int64)&v413, (__int64)v321) )
                {
                  sub_17EA700((__int64)&v413, (__int64)v321, v121, v122, v123, v124);
                  sub_17EAE60((__int64)&v413);
                  if ( !byte_4FA5900 )
                  {
                    sub_1695830((__int64)v413, dest, v438);
                    sub_17E6910((__int64)&v413, 0);
                    v120 = 1;
                    sub_17E6910((__int64)&v413, 1u);
                  }
                  v125 = (const char *)&v413;
                  sub_17E7E50((__int64)&v413);
                  if ( v458 == 1 )
                  {
                    v120 = (__int64)v357;
                    n[0] = (size_t)v326;
                    if ( v357 == v358 )
                    {
                      v125 = (const char *)&v356;
                      sub_17E9700((__int64)&v356, v357, n);
                    }
                    else
                    {
                      if ( v357 )
                      {
                        *v357 = (__int64)v326;
                        v120 = (__int64)v357;
                      }
                      v120 += 8;
                      v357 = (__int64 *)v120;
                    }
                  }
                  else if ( v458 == 2 )
                  {
                    v120 = (__int64)v354;
                    n[0] = (size_t)v326;
                    if ( v354 == v355 )
                    {
                      v125 = (const char *)&v353;
                      sub_17E9700((__int64)&v353, v354, n);
                    }
                    else
                    {
                      if ( v354 )
                      {
                        *v354 = (__int64)v326;
                        v120 = (__int64)v354;
                      }
                      v120 += 8;
                      v354 = (__int64 *)v120;
                    }
                  }
                  if ( unk_4F98100 )
                  {
                    if ( !qword_4F983A0[21]
                      || (v127 = sub_1649960((__int64)v326),
                          v129 = v128,
                          v126 = qword_4F983A0[21],
                          v125 = v127,
                          v126 == v129)
                      && (!v126 || (v120 = qword_4F983A0[20], !memcmp(v127, (const void *)v120, v126))) )
                    {
                      v393 = 0;
                      v391 = 0x100000000LL;
                      v390 = (signed __int64)&v392;
                      v397 = v326;
                      v394 = 0;
                      v395 = 0;
                      v396 = 0;
                      v398 = 0;
                      v399 = 0;
                      sub_15D3930((__int64)&v390);
                      v120 = (__int64)&v390;
                      sub_14019E0((__int64)n, (__int64)&v390);
                      if ( v396 )
                      {
                        v130 = v394;
                        v131 = &v394[2 * v396];
                        do
                        {
                          if ( *v130 != -8 && *v130 != -16 )
                          {
                            v132 = v130[1];
                            if ( v132 )
                            {
                              v133 = *(_QWORD *)(v132 + 24);
                              if ( v133 )
                                j_j___libc_free_0(v133, *(_QWORD *)(v132 + 40) - v133);
                              v120 = 56;
                              j_j___libc_free_0(v132, 56);
                            }
                          }
                          v130 += 2;
                        }
                        while ( v131 != v130 );
                      }
                      j___libc_free_0(v394);
                      if ( (__int64 *)v390 != &v392 )
                        _libc_free(v390);
                      v134 = (_QWORD *)sub_22077B0(408);
                      v135 = (__int64)v134;
                      if ( v134 )
                      {
                        *v134 = 0;
                        v136 = v134 + 14;
                        *(v136 - 13) = 0;
                        *(v136 - 12) = 0;
                        v120 = (__int64)v326;
                        *((_DWORD *)v136 - 22) = 0;
                        *(v136 - 10) = 0;
                        *(v136 - 9) = 0;
                        *(v136 - 8) = 0;
                        *((_DWORD *)v136 - 14) = 0;
                        *(_QWORD *)(v135 + 80) = v136;
                        *(_QWORD *)(v135 + 88) = v136;
                        *(_QWORD *)(v135 + 72) = 0;
                        *(_QWORD *)(v135 + 96) = 16;
                        *(_DWORD *)(v135 + 104) = 0;
                        *(_QWORD *)(v135 + 240) = 0;
                        *(_QWORD *)(v135 + 248) = v135 + 280;
                        *(_QWORD *)(v135 + 256) = v135 + 280;
                        *(_QWORD *)(v135 + 264) = 16;
                        *(_DWORD *)(v135 + 272) = 0;
                        sub_137CAE0(v135, v326, (__int64)n, 0);
                      }
                      v137 = 8;
                      v138 = (__int64 *)sub_22077B0(8);
                      v140 = v138;
                      if ( v138 )
                      {
                        v120 = (__int64)v326;
                        v137 = (__int64)v138;
                        sub_13702A0(v138, v326, v135, (__int64)n);
                      }
                      if ( unk_4F98100 == 1 )
                      {
                        sub_136FFE0(v140);
                      }
                      else if ( unk_4F98100 == 2 )
                      {
                        v294 = sub_16BA580(v137, v120, v139);
                        v295 = sub_1263B40(v294, "pgo-view-counts: ");
                        v296 = sub_1649960((__int64)v413);
                        v298 = sub_1549FF0(v295, v296, v297);
                        sub_1263B40(v298, "\n");
                        v120 = sub_16BA580(v298, (__int64)"\n", v299);
                        sub_1368E20(v140);
                      }
                      if ( v140 )
                      {
                        sub_1368A00(v140);
                        v120 = 8;
                        j_j___libc_free_0(v140, 8);
                      }
                      if ( v135 )
                      {
                        v141 = *(_QWORD *)(v135 + 256);
                        if ( v141 != *(_QWORD *)(v135 + 248) )
                          _libc_free(v141);
                        v142 = *(_QWORD *)(v135 + 88);
                        if ( v142 != *(_QWORD *)(v135 + 80) )
                          _libc_free(v142);
                        j___libc_free_0(*(_QWORD *)(v135 + 40));
                        v143 = *(unsigned int *)(v135 + 24);
                        if ( (_DWORD)v143 )
                        {
                          v144 = *(_QWORD **)(v135 + 8);
                          v379.m128i_i64[1] = 2;
                          v380 = 0;
                          v381 = -8;
                          v379.m128i_i64[0] = (__int64)&unk_49E8A80;
                          v382 = 0;
                          v391 = 2;
                          v392 = 0;
                          v393 = -16;
                          v390 = (signed __int64)&unk_49E8A80;
                          v394 = 0;
                          v145 = v135;
                          v146 = &v144[5 * v143];
                          do
                          {
                            v147 = v144[3];
                            *v144 = &unk_49EE2B0;
                            if ( v147 != 0 && v147 != -8 && v147 != -16 )
                              sub_1649B30(v144 + 1);
                            v144 += 5;
                          }
                          while ( v146 != v144 );
                          v135 = v145;
                          v390 = (signed __int64)&unk_49EE2B0;
                          if ( v393 != 0 && v393 != -8 && v393 != -16 )
                            sub_1649B30(&v391);
                          v379.m128i_i64[0] = (__int64)&unk_49EE2B0;
                          if ( v381 != 0 && v381 != -8 && v381 != -16 )
                            sub_1649B30(&v379.m128i_i64[1]);
                        }
                        j___libc_free_0(*(_QWORD *)(v135 + 8));
                        v120 = 408;
                        j_j___libc_free_0(v135, 408);
                      }
                      sub_142D890((__int64)n);
                      v148 = v403;
                      v149 = v402;
                      if ( v402 != v403 )
                      {
                        do
                        {
                          v150 = *v149;
                          v151 = *(__int64 **)(*v149 + 8);
                          v152 = *(__int64 **)(*v149 + 16);
                          if ( v151 == v152 )
                          {
                            *(_BYTE *)(v150 + 160) = 1;
                          }
                          else
                          {
                            do
                            {
                              v153 = *v151++;
                              sub_13FACC0(v153);
                            }
                            while ( v152 != v151 );
                            *(_BYTE *)(v150 + 160) = 1;
                            v154 = *(_QWORD *)(v150 + 8);
                            if ( *(_QWORD *)(v150 + 16) != v154 )
                              *(_QWORD *)(v150 + 16) = v154;
                          }
                          v155 = *(_QWORD *)(v150 + 32);
                          if ( v155 != *(_QWORD *)(v150 + 40) )
                            *(_QWORD *)(v150 + 40) = v155;
                          ++*(_QWORD *)(v150 + 56);
                          v156 = *(void **)(v150 + 72);
                          if ( v156 == *(void **)(v150 + 64) )
                          {
                            *(_QWORD *)v150 = 0;
                          }
                          else
                          {
                            v157 = 4 * (*(_DWORD *)(v150 + 84) - *(_DWORD *)(v150 + 88));
                            v158 = *(unsigned int *)(v150 + 80);
                            if ( v157 < 0x20 )
                              v157 = 32;
                            if ( (unsigned int)v158 > v157 )
                            {
                              sub_16CC920(v150 + 56);
                            }
                            else
                            {
                              v120 = 0xFFFFFFFFLL;
                              memset(v156, -1, 8 * v158);
                            }
                            v159 = *(_QWORD *)(v150 + 72);
                            v160 = *(_QWORD *)(v150 + 64);
                            *(_QWORD *)v150 = 0;
                            if ( v159 != v160 )
                              _libc_free(v159);
                          }
                          v161 = *(_QWORD *)(v150 + 32);
                          if ( v161 )
                          {
                            v120 = *(_QWORD *)(v150 + 48) - v161;
                            j_j___libc_free_0(v161, v120);
                          }
                          v162 = *(_QWORD *)(v150 + 8);
                          if ( v162 )
                          {
                            v120 = *(_QWORD *)(v150 + 24) - v162;
                            j_j___libc_free_0(v162, v120);
                          }
                          ++v149;
                        }
                        while ( v148 != v149 );
                        if ( v402 != v403 )
                          v403 = v402;
                      }
                      v163 = v410;
                      v164 = &v410[2 * v411];
                      if ( v410 != v164 )
                      {
                        do
                        {
                          v165 = *v163;
                          v163 += 2;
                          _libc_free(v165);
                        }
                        while ( v164 != v163 );
                      }
                      v411 = 0;
                      if ( v408 )
                      {
                        v166 = v407;
                        v412 = 0;
                        v167 = &v407[v408];
                        v168 = v407 + 1;
                        v405 = *v407;
                        v406 = v405 + 4096;
                        if ( v167 != v407 + 1 )
                        {
                          do
                          {
                            v169 = *v168++;
                            _libc_free(v169);
                          }
                          while ( v167 != v168 );
                          v166 = v407;
                        }
                        v408 = 1;
                        _libc_free(*v166);
                        v170 = v410;
                        v171 = &v410[2 * v411];
                        if ( v410 != v171 )
                        {
                          do
                          {
                            v172 = *v170;
                            v170 += 2;
                            _libc_free(v172);
                          }
                          while ( v170 != v171 );
                          goto LABEL_256;
                        }
                      }
                      else
                      {
LABEL_256:
                        v171 = v410;
                      }
                      if ( v171 != (unsigned __int64 *)&v412 )
                        _libc_free((unsigned __int64)v171);
                      if ( v407 != (unsigned __int64 *)&v409 )
                        _libc_free((unsigned __int64)v407);
                      if ( v402 )
                      {
                        v120 = v404 - (_QWORD)v402;
                        j_j___libc_free_0(v402, v404 - (_QWORD)v402);
                      }
                      v125 = (const char *)n[1];
                      j___libc_free_0(n[1]);
                    }
                  }
                  v173 = dword_4FA5080;
                  if ( dword_4FA5080 )
                  {
                    if ( qword_4F983A0[21] )
                    {
                      v279 = sub_1649960((__int64)v326);
                      v281 = v280;
                      v126 = qword_4F983A0[21];
                      v125 = v279;
                      if ( v126 == v281 )
                      {
                        if ( !v126 || (v120 = qword_4F983A0[20], !memcmp(v279, (const void *)v120, v126)) )
                        {
                          v173 = dword_4FA5080;
                          goto LABEL_266;
                        }
                      }
                    }
                    else
                    {
LABEL_266:
                      if ( v173 == 1 )
                      {
                        v292 = qword_4F983A0[21] == 0;
                        LOWORD(v392) = 257;
                        if ( v292 )
                        {
                          v377.m128i_i64[0] = (__int64)sub_1649960((__int64)v413);
                          v377.m128i_i64[1] = v293;
                          v379.m128i_i64[0] = (__int64)"PGORawCounts_";
                          LOWORD(v380) = 1283;
                          v379.m128i_i64[1] = (__int64)&v377;
                          v375.m128i_i64[0] = (__int64)&v413;
                          sub_17ED5F0((__m128i *)n, (__int64 **)&v375, (__int64)&v379, 0, (__int64)&v390);
                          sub_2240A30(n);
                        }
                        else
                        {
                          v302.m128i_i64[0] = (__int64)sub_1649960((__int64)v413);
                          v373 = v302;
                          LOWORD(v378[0]) = 261;
                          v377.m128i_i64[0] = (__int64)&v373;
                          v375.m128i_i64[0] = (__int64)"PGORawCounts_";
                          LOWORD(v376) = 259;
                          sub_14EC200(&v379, &v375, &v377);
                          v371.m128i_i64[0] = (__int64)&v413;
                          sub_17ED5F0((__m128i *)n, (__int64 **)&v371, (__int64)&v379, 0, (__int64)&v390);
                          if ( n[1] )
                            sub_16BED90((_BYTE *)n[0], n[1], 0, 0);
                          sub_2240A30(n);
                        }
                      }
                      else if ( v173 == 2 )
                      {
                        v305 = sub_16BA580((__int64)v125, v120, v126);
                        v306 = sub_1263B40(v305, "pgo-view-raw-counts: ");
                        v307 = sub_1649960((__int64)v413);
                        v309 = sub_1549FF0(v306, v307, v308);
                        sub_1263B40(v309, "\n");
                        sub_17E3D90(&v390, byte_3F871B3);
                        n[0] = (size_t)v401;
                        sub_17E2330((__int64 *)n, (_BYTE *)v390, v390 + v391);
                        LOWORD(v378[0]) = 260;
                        v373.m128i_i64[0] = (__int64)"\t";
                        v369.m128i_i64[0] = (__int64)&v440;
                        v365.m128i_i64[0] = (__int64)" Hash: ";
                        v370 = 267;
                        v362.m128i_i64[0] = (__int64)&dest;
                        v363 = 260;
                        v359.m128i_i64[0] = (__int64)"Dump Function ";
                        v377.m128i_i64[0] = (__int64)n;
                        LOWORD(v374) = 259;
                        v367 = 1;
                        v366 = 3;
                        v361 = 1;
                        v360 = 3;
                        sub_14EC200(v364, &v359, &v362);
                        sub_14EC200(v368, v364, &v365);
                        sub_14EC200(&v371, v368, &v369);
                        sub_14EC200(&v375, &v371, &v373);
                        sub_14EC200(&v379, &v375, &v377);
                        v311 = sub_16BA580((__int64)&v379, (__int64)&v375, v310);
                        sub_17E6B50((__int64)&v441, v311, (__int64)&v379);
                        sub_2240A30(n);
                        sub_2240A30(&v390);
                      }
                    }
                  }
                  v174 = v457;
                  if ( v457 )
                  {
                    v175 = *(_QWORD ***)(v457 + 32);
                    v176 = *(_QWORD ***)(v457 + 24);
                    if ( v175 != v176 )
                    {
                      do
                      {
                        v177 = *v176;
                        while ( v176 != v177 )
                        {
                          v178 = v177;
                          v177 = (_QWORD *)*v177;
                          j_j___libc_free_0(v178, 32);
                        }
                        v176 += 3;
                      }
                      while ( v175 != v176 );
                      v176 = *(_QWORD ***)(v174 + 24);
                    }
                    if ( v176 )
                      j_j___libc_free_0(v176, *(_QWORD *)(v174 + 40) - (_QWORD)v176);
                    v179 = *(_QWORD ***)(v174 + 8);
                    v180 = *(_QWORD ***)v174;
                    if ( v179 != *(_QWORD ***)v174 )
                    {
                      do
                      {
                        v181 = *v180;
                        while ( v180 != v181 )
                        {
                          v182 = v181;
                          v181 = (_QWORD *)*v181;
                          j_j___libc_free_0(v182, 32);
                        }
                        v180 += 3;
                      }
                      while ( v179 != v180 );
                      v180 = *(_QWORD ***)v174;
                    }
                    if ( v180 )
                      j_j___libc_free_0(v180, *(_QWORD *)(v174 + 16) - (_QWORD)v180);
                    j_j___libc_free_0(v174, 48);
                  }
                  if ( v454 )
                    j_j___libc_free_0(v454, v456 - v454);
                  if ( v448 )
                  {
                    v183 = v446;
                    v184 = &v446[2 * v448];
                    do
                    {
                      if ( *v183 != -16 && *v183 != -8 )
                      {
                        v185 = v183[1];
                        if ( v185 )
                        {
                          v186 = *(_QWORD *)(v185 + 72);
                          if ( v186 != v185 + 88 )
                            _libc_free(v186);
                          v187 = *(_QWORD *)(v185 + 40);
                          if ( v187 != v185 + 56 )
                            _libc_free(v187);
                          j_j___libc_free_0(v185, 104);
                        }
                      }
                      v183 += 2;
                    }
                    while ( v184 != v183 );
                  }
                  j___libc_free_0(v446);
                  v188 = v443;
                  v189 = v442;
                  if ( v443 != v442 )
                  {
                    do
                    {
                      if ( *(_QWORD *)v189 )
                        j_j___libc_free_0(*(_QWORD *)v189, 40);
                      v189 += 8;
                    }
                    while ( v188 != v189 );
                    v189 = v442;
                  }
                  if ( v189 )
                    j_j___libc_free_0(v189, v444 - (_QWORD)v189);
                  if ( dest != v439 )
                    j_j___libc_free_0(dest, v439[0] + 1LL);
                  if ( v434 )
                    j_j___libc_free_0(v434, v436 - v434);
                  v190 = v419;
                  v191 = v418;
                  if ( v419 != v418 )
                  {
                    do
                    {
                      if ( *(_QWORD *)v191 )
                        j_j___libc_free_0(*(_QWORD *)v191, *((_QWORD *)v191 + 2) - *(_QWORD *)v191);
                      v191 += 24;
                    }
                    while ( v190 != v191 );
                    goto LABEL_313;
                  }
                }
                else
                {
                  v246 = v457;
                  if ( v457 )
                  {
                    v247 = *(_QWORD ***)(v457 + 32);
                    v248 = *(_QWORD ***)(v457 + 24);
                    if ( v247 != v248 )
                    {
                      do
                      {
                        v249 = *v248;
                        while ( v249 != v248 )
                        {
                          v250 = v249;
                          v249 = (_QWORD *)*v249;
                          j_j___libc_free_0(v250, 32);
                        }
                        v248 += 3;
                      }
                      while ( v247 != v248 );
                      v248 = *(_QWORD ***)(v246 + 24);
                    }
                    if ( v248 )
                      j_j___libc_free_0(v248, *(_QWORD *)(v246 + 40) - (_QWORD)v248);
                    v251 = *(_QWORD ***)(v246 + 8);
                    v252 = *(_QWORD ***)v246;
                    if ( v251 != *(_QWORD ***)v246 )
                    {
                      do
                      {
                        v253 = *v252;
                        while ( v253 != v252 )
                        {
                          v254 = v253;
                          v253 = (_QWORD *)*v253;
                          j_j___libc_free_0(v254, 32);
                        }
                        v252 += 3;
                      }
                      while ( v251 != v252 );
                      v252 = *(_QWORD ***)v246;
                    }
                    if ( v252 )
                      j_j___libc_free_0(v252, *(_QWORD *)(v246 + 16) - (_QWORD)v252);
                    j_j___libc_free_0(v246, 48);
                  }
                  if ( v454 )
                    j_j___libc_free_0(v454, v456 - v454);
                  if ( v448 )
                  {
                    v255 = v446;
                    v256 = &v446[2 * v448];
                    do
                    {
                      if ( *v255 != -16 && *v255 != -8 )
                      {
                        v257 = v255[1];
                        if ( v257 )
                        {
                          v258 = *(_QWORD *)(v257 + 72);
                          if ( v258 != v257 + 88 )
                            _libc_free(v258);
                          v259 = *(_QWORD *)(v257 + 40);
                          if ( v259 != v257 + 56 )
                            _libc_free(v259);
                          j_j___libc_free_0(v257, 104);
                        }
                      }
                      v255 += 2;
                    }
                    while ( v256 != v255 );
                  }
                  j___libc_free_0(v446);
                  v260 = v443;
                  v261 = v442;
                  if ( v443 != v442 )
                  {
                    do
                    {
                      if ( *(_QWORD *)v261 )
                        j_j___libc_free_0(*(_QWORD *)v261, 40);
                      v261 += 8;
                    }
                    while ( v260 != v261 );
                    v261 = v442;
                  }
                  if ( v261 )
                    j_j___libc_free_0(v261, v444 - (_QWORD)v261);
                  if ( dest != v439 )
                    j_j___libc_free_0(dest, v439[0] + 1LL);
                  if ( v434 )
                    j_j___libc_free_0(v434, v436 - v434);
                  v262 = v419;
                  v191 = v418;
                  if ( v419 != v418 )
                  {
                    do
                    {
                      if ( *(_QWORD *)v191 )
                        j_j___libc_free_0(*(_QWORD *)v191, *((_QWORD *)v191 + 2) - *(_QWORD *)v191);
                      v191 += 24;
                    }
                    while ( v262 != v191 );
LABEL_313:
                    v191 = v418;
                  }
                }
                if ( v191 )
                  j_j___libc_free_0(v191, v420 - v191);
LABEL_316:
                v328 = *(_QWORD *)(v328 + 8);
                if ( v328 == v322 )
                {
LABEL_317:
                  v192 = sub_163B6D0(v321[5], *(__int64 **)a1);
                  sub_1633D70((__int64 **)a1, v192);
                  v193 = v354;
                  for ( m = v353; v193 != m; ++m )
                  {
                    v195 = *m;
                    sub_15E0D50(v195, -1, 15);
                  }
                  v196 = v356;
                  v197 = v357;
                  v198 = v356;
                  if ( v357 != v356 )
                  {
                    do
                    {
                      v199 = *v198++;
                      sub_15E0D50(v199, -1, 7);
                    }
                    while ( v197 != v198 );
                    v196 = v356;
                  }
                  if ( v196 )
                    j_j___libc_free_0(v196, (char *)v358 - (char *)v196);
                  if ( v353 )
                    j_j___libc_free_0(v353, (char *)v355 - (char *)v353);
                  v200 = v385;
                  while ( v200 )
                  {
                    v201 = v200;
                    v200 = (_QWORD *)*v200;
                    j_j___libc_free_0(v201, 24);
                  }
                  memset(s, 0, 8 * v384);
                  v13 = (__m128i *)v384;
                  v386 = 0;
                  v385 = 0;
                  if ( s != v389 )
                  {
                    v13 = (__m128i *)(8 * v384);
                    j_j___libc_free_0(s, 8 * v384);
                  }
                  goto LABEL_330;
                }
                goto LABEL_44;
              }
            }
            n[0] = (size_t)v401;
            v271 = v401;
            goto LABEL_490;
          }
        }
        v206 = *(__int64 **)v68;
        v207 = v448;
        v208 = v446;
        v209 = *(_QWORD *)(*(_QWORD *)v68 + 8LL);
        if ( v448 )
        {
          v210 = *v206;
          v211 = v448 - 1;
          v212 = (v448 - 1) & (((unsigned int)*v206 >> 9) ^ ((unsigned int)*v206 >> 4));
          v213 = &v446[2 * v212];
          v214 = *v213;
          if ( v210 == *v213 )
          {
LABEL_347:
            v215 = (_QWORD *)v213[1];
            v216 = (_QWORD ***********)*v215;
            if ( (_QWORD *)*v215 == v215 )
            {
LABEL_365:
              v226 = v211 & (((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4));
              v227 = &v208[2 * v226];
              v228 = *v227;
              if ( v209 == *v227 )
              {
LABEL_366:
                v229 = (_QWORD *)v227[1];
                v230 = (_QWORD *)*v229;
                if ( v229 != (_QWORD *)*v229 )
                {
                  v231 = (_QWORD **********)*v230;
                  if ( v230 != (_QWORD *)*v230 )
                  {
                    v232 = *v231;
                    if ( v231 != *v231 )
                    {
                      v233 = *v232;
                      if ( v232 != *v232 )
                      {
                        v234 = *v233;
                        if ( v233 != *v233 )
                        {
                          v235 = *v234;
                          if ( v234 != *v234 )
                          {
                            v236 = *v235;
                            if ( v235 != *v235 )
                            {
                              v237 = *v236;
                              if ( v236 != *v236 )
                              {
                                v238 = sub_17E2800(v237);
                                *v239 = v238;
                                v237 = (_QWORD ****)v238;
                              }
                              *v235 = (_QWORD *****)v237;
                              v236 = (_QWORD *****)v237;
                            }
                            *v234 = (_QWORD ******)v236;
                            v235 = (_QWORD ******)v236;
                          }
                          *v233 = (_QWORD *******)v235;
                          v234 = (_QWORD *******)v235;
                        }
                        *v232 = (_QWORD ********)v234;
                        v232 = (_QWORD *********)v234;
                      }
                      *v231 = v232;
                    }
                    *v230 = v232;
                    v230 = v232;
                  }
                  *v229 = v230;
                }
                if ( v215 != v230 )
                {
                  v240 = *((_DWORD *)v230 + 3);
                  if ( *((_DWORD *)v215 + 3) >= v240 )
                  {
                    *v230 = v215;
                    if ( v240 == *((_DWORD *)v215 + 3) )
                      *((_DWORD *)v215 + 3) = v240 + 1;
                  }
                  else
                  {
                    *v215 = v230;
                  }
                  *(_BYTE *)(*(_QWORD *)v68 + 24LL) = 1;
                }
                goto LABEL_105;
              }
              v300 = 1;
              while ( v228 != -8 )
              {
                v301 = v300 + 1;
                v226 = v211 & (v300 + v226);
                v227 = &v208[2 * v226];
                v228 = *v227;
                if ( v209 == *v227 )
                  goto LABEL_366;
                v300 = v301;
              }
LABEL_471:
              v227 = &v208[2 * v207];
              goto LABEL_366;
            }
          }
          else
          {
            v304 = 1;
            while ( v214 != -8 )
            {
              v316 = v304 + 1;
              v212 = v211 & (v304 + v212);
              v213 = &v446[2 * v212];
              v214 = *v213;
              if ( v210 == *v213 )
                goto LABEL_347;
              v304 = v316;
            }
            v215 = (_QWORD *)v446[2 * v448 + 1];
            v216 = (_QWORD ***********)*v215;
            if ( v215 == (_QWORD *)*v215 )
            {
LABEL_364:
              v215 = v216;
              goto LABEL_365;
            }
          }
        }
        else
        {
          v267 = (_QWORD *)v446[1];
          v215 = (_QWORD *)*v267;
          if ( v267 == (_QWORD *)*v267 )
            goto LABEL_471;
          v216 = (_QWORD ***********)*v267;
          v215 = (_QWORD *)v446[1];
        }
        v217 = *v216;
        if ( *v216 != v216 )
        {
          v218 = *v217;
          if ( v217 != *v217 )
          {
            v219 = *v218;
            if ( v218 != *v218 )
            {
              v220 = *v219;
              if ( v219 != *v219 )
              {
                v221 = *v220;
                if ( v220 != *v220 )
                {
                  v222 = *v221;
                  if ( v221 != *v221 )
                  {
                    v223 = *v222;
                    if ( v222 != *v222 )
                    {
                      v224 = sub_17E2800(v223);
                      *v225 = v224;
                      v223 = (_QWORD ****)v224;
                    }
                    *v221 = (_QWORD *****)v223;
                    v222 = (_QWORD *****)v223;
                  }
                  *v220 = (_QWORD ******)v222;
                  v221 = (_QWORD ******)v222;
                }
                *v219 = (_QWORD *******)v221;
                v220 = (_QWORD *******)v221;
              }
              *v218 = (_QWORD ********)v220;
              v218 = (_QWORD *********)v220;
            }
            *v217 = v218;
          }
          *v216 = (_QWORD **********)v218;
          v216 = (_QWORD ***********)v218;
        }
        *v215 = v216;
        v207 = v448;
        v208 = v446;
        if ( !v448 )
        {
          v215 = v216;
          goto LABEL_471;
        }
        v211 = v448 - 1;
        goto LABEL_364;
      }
    }
    if ( 2 * v345 < 3 * v333 )
    {
      *(_QWORD *)(v340 + 16) = v333;
      *(_QWORD *)(v332 + 16) = v345 + 1;
    }
    goto LABEL_88;
  }
  v10 = v351;
  v351 = 0;
  v11 = (unsigned __int64)v10 & 0xFFFFFFFFFFFFFFFELL;
  v12 = (_QWORD *)v11;
  if ( !v11 )
  {
    v371.m128i_i64[0] = 0;
    sub_14ECA90(v371.m128i_i64);
    goto LABEL_538;
  }
  v413 = v8;
  v13 = (__m128i *)&unk_4FA032A;
  v414 = (__int64)v350;
  v371.m128i_i64[0] = 0;
  v373.m128i_i64[0] = 0;
  v375.m128i_i64[0] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v11 + 48LL))(v11, &unk_4FA032A) )
  {
    v377.m128i_i64[0] = 1;
    v14 = (void **)v12[1];
    v15 = (void **)v12[2];
    if ( v14 == v15 )
    {
      v18 = 1;
    }
    else
    {
      do
      {
        s = *v14;
        *v14 = 0;
        sub_17E5460(&v390, &s, (__int64 *)&v413);
        v16 = v377.m128i_i64[0];
        v13 = &v379;
        v377.m128i_i64[0] = 0;
        v379.m128i_i64[0] = v16 | 1;
        sub_12BEC00(n, (unsigned __int64 *)&v379, (unsigned __int64 *)&v390);
        if ( (v377.m128i_i8[0] & 1) != 0 || (v377.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v377, (__int64)&v379, v17);
        v377.m128i_i64[0] |= n[0] | 1;
        if ( (v379.m128i_i8[0] & 1) != 0 || (v379.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v379, (__int64)&v379, v17);
        if ( (v390 & 1) != 0 || (v390 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v390, (__int64)&v379, v17);
        if ( s )
          (*(void (__fastcall **)(void *))(*(_QWORD *)s + 8LL))(s);
        ++v14;
      }
      while ( v15 != v14 );
      v18 = v377.m128i_i64[0] | 1;
    }
    v390 = v18;
    (*(void (__fastcall **)(_QWORD *))(*v12 + 8LL))(v12);
  }
  else
  {
    v13 = (__m128i *)n;
    n[0] = (size_t)v12;
    sub_17E5460(&v390, n, (__int64 *)&v413);
    if ( n[0] )
      (*(void (__fastcall **)(size_t))(*(_QWORD *)n[0] + 8LL))(n[0]);
  }
  v390 = ((v390 & 0xFFFFFFFFFFFFFFFELL) != 0) | v390 & 0xFFFFFFFFFFFFFFFELL;
  sub_14ECA90(&v390);
  sub_14ECA90(v375.m128i_i64);
  sub_14ECA90(v373.m128i_i64);
  sub_14ECA90(v371.m128i_i64);
  v320 = 0;
LABEL_331:
  if ( (v352 & 2) != 0 )
    sub_17E9690(&v351, (__int64)v13, v202);
  if ( v351 )
    (*((void (__fastcall **)(int **))*v351 + 1))(v351);
  return v320;
}
