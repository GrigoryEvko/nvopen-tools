// Function: sub_24B8750
// Address: 0x24b8750
//
__int64 __fastcall sub_24B8750(
        size_t a1,
        __int64 a2,
        __int64 a3,
        __m128i *a4,
        __int64 **a5,
        __int64 *a6,
        int a7,
        __int64 *a8,
        int a9,
        __int64 *a10,
        int a11,
        __int64 *a12,
        int a13,
        __int64 *a14,
        __int64 *a15,
        unsigned __int8 a16)
{
  __int64 *v16; // r12
  __m128i *v17; // rsi
  char v18; // al
  unsigned int **v19; // rax
  unsigned __int64 v20; // rax
  unsigned int **v22; // rax
  __int64 (__fastcall *v23)(__int64); // rax
  unsigned __int8 v24; // al
  unsigned int *v25; // rdx
  __int64 (__fastcall *v26)(__int64); // rax
  __int64 (__fastcall *v27)(__int64); // rax
  char v28; // al
  __int64 v29; // r13
  size_t i; // r12
  int v31; // r12d
  int *v32; // rdi
  unsigned __int8 *v33; // rax
  __int64 (__fastcall *v34)(__int64); // rax
  _QWORD *v35; // r12
  _QWORD *v36; // rbx
  unsigned __int64 v37; // rsi
  _QWORD *v38; // rax
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rdx
  _QWORD *v43; // r8
  __int64 v44; // rax
  _QWORD *v45; // rdi
  __int64 v46; // rcx
  __int64 (__fastcall *v47)(__int64); // rax
  _QWORD *v48; // r12
  _QWORD *v49; // rbx
  unsigned __int64 v50; // rsi
  _QWORD *v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rdx
  _QWORD *v56; // r8
  __int64 v57; // rax
  _QWORD *v58; // rdi
  __int64 v59; // rcx
  __int64 (__fastcall *v60)(__int64); // rax
  __m128i *v61; // rbx
  __int64 v62; // r13
  __int64 v63; // r12
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // r14
  unsigned __int64 *v67; // rax
  unsigned __int64 *v68; // rdx
  unsigned __int64 *v69; // r10
  unsigned __int64 *v70; // r9
  unsigned __int64 *v71; // r14
  __int64 v72; // r15
  unsigned __int64 *v73; // r12
  unsigned __int64 *v74; // rax
  unsigned __int64 *v75; // r13
  __int64 v76; // rcx
  unsigned __int64 *v77; // r15
  unsigned __int64 *v78; // rax
  unsigned __int64 v79; // rdx
  unsigned __int64 *v80; // rax
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rdi
  unsigned __int64 *v83; // r14
  __int64 **v84; // r14
  __int64 **v85; // r13
  __int64 *v86; // rax
  __int64 v87; // rdi
  __int64 **v88; // r14
  unsigned __int64 *v89; // r13
  __int64 *v90; // rax
  __int64 v91; // rsi
  unsigned __int64 *v92; // rax
  unsigned __int64 v93; // rcx
  __m128i *v94; // r15
  __m128i *v95; // rax
  __int64 *v96; // r13
  __int64 *v97; // r14
  __int64 *v98; // rsi
  int v99; // eax
  unsigned __int64 *v100; // rax
  unsigned __int64 v101; // rdi
  unsigned __int64 *v102; // rdi
  unsigned __int64 v103; // rsi
  unsigned __int64 *v104; // rdi
  unsigned __int64 v105; // rsi
  __int64 *v106; // rax
  char v107; // al
  __int64 v108; // rax
  __int64 v109; // rsi
  unsigned __int64 *v110; // r14
  unsigned __int64 v111; // rdi
  unsigned __int64 *v112; // r13
  unsigned __int64 *v113; // r15
  const char *v114; // rdi
  size_t v115; // r13
  const void *v116; // r14
  __int64 v117; // rdx
  __int64 v118; // rsi
  unsigned __int64 *v119; // r13
  unsigned __int64 *v120; // r14
  unsigned __int64 v121; // r8
  unsigned __int64 v122; // rdi
  _QWORD *v123; // rax
  _QWORD *v124; // r13
  _QWORD *v125; // rdx
  _QWORD *v126; // rax
  _QWORD *v127; // rax
  __int64 v128; // rdi
  __int64 *v129; // rax
  __int64 *v130; // r14
  unsigned __int64 v131; // r14
  unsigned __int64 v132; // r15
  __int64 v133; // rbx
  __int64 *v134; // r13
  __int64 *v135; // r12
  __int64 v136; // rdi
  __int64 v137; // rax
  __int64 v138; // rax
  unsigned int v139; // eax
  __int64 v140; // rdx
  char v141; // al
  unsigned __int64 v142; // rdi
  unsigned __int64 v143; // rdi
  __int64 *v144; // r14
  __int64 v145; // r13
  __int64 v146; // rsi
  __int64 v147; // rdi
  __int64 *v148; // rax
  __int64 *v149; // r13
  __int64 *v150; // r15
  __int64 v151; // rdi
  unsigned int v152; // ecx
  __int64 v153; // rax
  __int64 *v154; // r14
  __int64 *v155; // r13
  __int64 v156; // rsi
  __int64 v157; // rdi
  int v158; // eax
  size_t v159; // r14
  __int64 v160; // rax
  __int64 v161; // r13
  const char *v162; // rax
  size_t v163; // rdx
  __int64 v164; // rax
  __int64 v165; // rcx
  __int64 v166; // r9
  __int64 v167; // rcx
  __int64 v168; // r8
  __int64 v169; // r9
  __int64 v170; // rcx
  __int64 v171; // r8
  __int64 v172; // r9
  __int64 v173; // rcx
  __int64 v174; // r8
  __int64 v175; // r9
  __int64 v176; // rcx
  __int64 v177; // r8
  __int64 v178; // r9
  __int64 v179; // r12
  unsigned __int64 *v180; // r15
  unsigned __int64 v181; // r14
  unsigned __int64 v182; // rdi
  __m128i **v183; // rax
  char *v184; // rax
  unsigned __int64 v185; // r8
  unsigned __int64 v186; // r12
  __int64 v187; // rsi
  unsigned __int64 v188; // r13
  unsigned __int64 v189; // r12
  __int64 v190; // r15
  __int64 *v191; // rbx
  __int64 *v192; // r14
  __int64 v193; // rdi
  __int64 v194; // rax
  __int64 v195; // rax
  unsigned int v196; // eax
  __int64 v197; // rdx
  char v198; // al
  unsigned __int64 v199; // rdi
  unsigned __int64 v200; // rdi
  __int64 *v201; // r12
  __int64 v202; // rbx
  __int64 v203; // rsi
  __int64 v204; // rdi
  __int64 *v205; // r12
  __int64 *v206; // rbx
  __int64 *ii; // r12
  __int64 v208; // rdi
  __int64 *v209; // r12
  __int64 *v210; // rbx
  __int64 v211; // rdi
  __int64 v212; // rdx
  char *v213; // rsi
  unsigned __int64 v214; // rax
  _QWORD *v215; // rbx
  unsigned __int64 v216; // rdi
  __int64 v217; // rax
  __int64 v218; // r14
  unsigned __int64 *v219; // r13
  unsigned __int64 v220; // rdi
  unsigned __int64 *v221; // rax
  unsigned __int64 v222; // rdi
  unsigned __int64 *v223; // rax
  unsigned __int64 v224; // rdi
  __int64 v225; // rdx
  __int64 v226; // rdx
  __int64 *v227; // rax
  unsigned __int64 v228; // rdi
  __int64 v229; // rax
  unsigned __int64 v230; // rax
  __int64 v231; // rax
  unsigned __int64 v232; // rdi
  unsigned __int64 v233; // rax
  unsigned __int64 v234; // rax
  unsigned __int64 *v235; // rax
  unsigned __int64 v236; // rcx
  unsigned __int64 *v237; // r14
  unsigned __int64 v238; // rdi
  unsigned __int64 *v239; // r13
  unsigned __int64 *v240; // r15
  __int64 v241; // rdx
  __int64 v242; // rcx
  __int64 *v243; // rax
  __int64 v244; // rsi
  __int64 v245; // rcx
  __int64 v246; // rdx
  __int64 v247; // rcx
  __int64 v248; // r8
  __int64 v249; // r9
  __int64 v250; // rax
  __int64 v251; // r14
  unsigned __int64 *v252; // rcx
  __int64 v253; // r15
  unsigned __int64 v254; // rax
  unsigned __int64 *v255; // rax
  unsigned __int64 v256; // rcx
  unsigned __int64 v257; // rax
  unsigned __int64 v258; // r13
  int k; // r12d
  int v260; // r9d
  unsigned int v261; // r8d
  __int64 *v262; // rax
  __int64 *v263; // rbx
  __int64 *v264; // r13
  __int64 v265; // rdi
  unsigned int v266; // ecx
  __int64 v267; // rax
  __int64 *v268; // rbx
  __int64 v269; // rsi
  __int64 v270; // rdi
  __int64 *v271; // r13
  char *v272; // rax
  __int64 v273; // rdx
  __int64 v274; // rcx
  __int64 v275; // r8
  __int64 v276; // r9
  char *v277; // rax
  __int64 v278; // rcx
  __int64 v279; // r8
  __int64 v280; // r9
  __int64 v281; // rcx
  __int64 v282; // r8
  __int64 v283; // r9
  __int64 v284; // rcx
  __int64 v285; // r8
  __int64 v286; // r9
  __int64 *v287; // rax
  unsigned __int64 v288; // rdx
  __int64 v289; // rcx
  const char *v290; // r9
  size_t v291; // r8
  _QWORD *v292; // rax
  __int64 v293; // rcx
  __int64 v294; // r9
  __int64 *v295; // rax
  unsigned __int64 *v296; // rax
  unsigned __int64 v297; // rdi
  __m128i v298; // rax
  __int64 v299; // rcx
  __int64 v300; // r8
  __int64 v301; // r9
  const void *v302; // r15
  __int64 v303; // rdx
  __int64 v304; // rax
  _QWORD *v305; // rdi
  __int64 v306; // rax
  __int64 v307; // r15
  const char *v308; // rax
  size_t v309; // rdx
  __int64 v310; // rdi
  __m128i v311; // rax
  __int64 v312; // rcx
  __int64 v313; // r8
  __int64 v314; // r9
  size_t v315; // [rsp+8h] [rbp-718h]
  size_t v316; // [rsp+8h] [rbp-718h]
  unsigned __int64 *v317; // [rsp+10h] [rbp-710h]
  const char *v318; // [rsp+10h] [rbp-710h]
  bool v319; // [rsp+28h] [rbp-6F8h]
  __int64 v320; // [rsp+48h] [rbp-6D8h]
  int v321; // [rsp+48h] [rbp-6D8h]
  unsigned __int64 v322; // [rsp+50h] [rbp-6D0h]
  __int64 v323; // [rsp+90h] [rbp-690h]
  char v324; // [rsp+90h] [rbp-690h]
  __m128i *v325; // [rsp+90h] [rbp-690h]
  unsigned __int64 v326; // [rsp+90h] [rbp-690h]
  unsigned __int64 v327; // [rsp+90h] [rbp-690h]
  __int64 j; // [rsp+A0h] [rbp-680h]
  char v330; // [rsp+ABh] [rbp-675h]
  unsigned __int8 v331; // [rsp+ACh] [rbp-674h]
  unsigned __int8 v332; // [rsp+ADh] [rbp-673h]
  char v333; // [rsp+AEh] [rbp-672h]
  char *src; // [rsp+B0h] [rbp-670h]
  __int64 *srca; // [rsp+B0h] [rbp-670h]
  unsigned int **v336; // [rsp+C0h] [rbp-660h]
  __int64 *v337; // [rsp+C8h] [rbp-658h]
  _QWORD v338[2]; // [rsp+D0h] [rbp-650h] BYREF
  unsigned int **v339; // [rsp+E0h] [rbp-640h] BYREF
  char v340; // [rsp+E8h] [rbp-638h]
  __int64 *v341; // [rsp+F0h] [rbp-630h] BYREF
  __int64 *v342; // [rsp+F8h] [rbp-628h]
  __int64 *v343; // [rsp+100h] [rbp-620h]
  __int64 *v344; // [rsp+110h] [rbp-610h] BYREF
  __int64 *v345; // [rsp+118h] [rbp-608h]
  __int64 *v346; // [rsp+120h] [rbp-600h]
  __m128i v347; // [rsp+130h] [rbp-5F0h] BYREF
  _QWORD v348[4]; // [rsp+140h] [rbp-5E0h] BYREF
  __m128i v349; // [rsp+160h] [rbp-5C0h] BYREF
  __int64 *v350; // [rsp+170h] [rbp-5B0h]
  __m128i *v351; // [rsp+178h] [rbp-5A8h]
  __int16 v352; // [rsp+180h] [rbp-5A0h]
  __m128i v353[2]; // [rsp+190h] [rbp-590h] BYREF
  char v354; // [rsp+1B0h] [rbp-570h]
  char v355; // [rsp+1B1h] [rbp-56Fh]
  __m128i v356; // [rsp+1C0h] [rbp-560h] BYREF
  __int64 *v357; // [rsp+1D0h] [rbp-550h]
  __m128i *v358; // [rsp+1D8h] [rbp-548h]
  __int16 v359; // [rsp+1E0h] [rbp-540h]
  __m128i v360; // [rsp+1F0h] [rbp-530h] BYREF
  _QWORD v361[4]; // [rsp+200h] [rbp-520h] BYREF
  __m128i v362[2]; // [rsp+220h] [rbp-500h] BYREF
  __int16 v363; // [rsp+240h] [rbp-4E0h]
  __m128i v364[2]; // [rsp+250h] [rbp-4D0h] BYREF
  char v365; // [rsp+270h] [rbp-4B0h]
  char v366; // [rsp+271h] [rbp-4AFh]
  __m128i v367[2]; // [rsp+280h] [rbp-4A0h] BYREF
  __int16 v368; // [rsp+2A0h] [rbp-480h]
  __m128i v369[2]; // [rsp+2B0h] [rbp-470h] BYREF
  __int16 v370; // [rsp+2D0h] [rbp-450h]
  void *s; // [rsp+2E0h] [rbp-440h] BYREF
  __int64 v372; // [rsp+2E8h] [rbp-438h]
  _QWORD *v373; // [rsp+2F0h] [rbp-430h]
  __int64 v374; // [rsp+2F8h] [rbp-428h]
  int v375; // [rsp+300h] [rbp-420h]
  __int64 v376; // [rsp+308h] [rbp-418h]
  _QWORD v377[2]; // [rsp+310h] [rbp-410h] BYREF
  __m128i v378; // [rsp+320h] [rbp-400h] BYREF
  __int64 v379; // [rsp+330h] [rbp-3F0h] BYREF
  unsigned __int64 *v380; // [rsp+338h] [rbp-3E8h]
  unsigned __int64 v381; // [rsp+340h] [rbp-3E0h]
  unsigned __int64 v382; // [rsp+348h] [rbp-3D8h] BYREF
  __int64 v383; // [rsp+358h] [rbp-3C8h]
  __int64 n; // [rsp+360h] [rbp-3C0h]
  __int64 *v385; // [rsp+368h] [rbp-3B8h]
  unsigned int v386; // [rsp+370h] [rbp-3B0h]
  char v387; // [rsp+378h] [rbp-3A8h] BYREF
  __int64 v388; // [rsp+380h] [rbp-3A0h]
  __m128i *v389; // [rsp+388h] [rbp-398h]
  char v390; // [rsp+390h] [rbp-390h]
  _BYTE v391[12]; // [rsp+394h] [rbp-38Ch]
  unsigned int v392; // [rsp+3A0h] [rbp-380h]
  __int64 v393; // [rsp+3A8h] [rbp-378h] BYREF
  __m128i v394; // [rsp+3C0h] [rbp-360h] BYREF
  unsigned __int64 v395; // [rsp+3D0h] [rbp-350h] BYREF
  unsigned __int64 *v396; // [rsp+3D8h] [rbp-348h]
  unsigned __int64 v397; // [rsp+3E0h] [rbp-340h]
  unsigned __int64 v398; // [rsp+3E8h] [rbp-338h] BYREF
  unsigned __int64 *v399; // [rsp+3F0h] [rbp-330h]
  __int64 v400; // [rsp+3F8h] [rbp-328h]
  __int64 m; // [rsp+400h] [rbp-320h]
  __int64 *v402; // [rsp+408h] [rbp-318h]
  __int64 v403; // [rsp+410h] [rbp-310h]
  __int64 v404; // [rsp+418h] [rbp-308h] BYREF
  __int64 v405; // [rsp+420h] [rbp-300h]
  __m128i *v406; // [rsp+428h] [rbp-2F8h] BYREF
  char v407; // [rsp+430h] [rbp-2F0h]
  _BYTE v408[12]; // [rsp+434h] [rbp-2ECh]
  unsigned int v409; // [rsp+440h] [rbp-2E0h]
  __int64 v410; // [rsp+448h] [rbp-2D8h] BYREF
  _QWORD v411[2]; // [rsp+468h] [rbp-2B8h] BYREF
  char v412; // [rsp+478h] [rbp-2A8h] BYREF
  char v413; // [rsp+4D8h] [rbp-248h] BYREF
  __m128i *v414; // [rsp+4E0h] [rbp-240h] BYREF
  size_t v415; // [rsp+4E8h] [rbp-238h]
  __int64 v416; // [rsp+4F0h] [rbp-230h]
  __int64 *v417; // [rsp+4F8h] [rbp-228h]
  __m128i *v418; // [rsp+500h] [rbp-220h] BYREF
  unsigned __int8 v419; // [rsp+508h] [rbp-218h]
  void **p_s; // [rsp+510h] [rbp-210h]
  char v421[8]; // [rsp+518h] [rbp-208h] BYREF
  __int64 v422; // [rsp+520h] [rbp-200h]
  unsigned __int64 *v423; // [rsp+528h] [rbp-1F8h]
  unsigned __int64 *v424; // [rsp+530h] [rbp-1F0h]
  _QWORD *v425; // [rsp+538h] [rbp-1E8h]
  __m128i *v426; // [rsp+540h] [rbp-1E0h]
  __int64 v427; // [rsp+548h] [rbp-1D8h]
  __int64 v428; // [rsp+550h] [rbp-1D0h]
  int v429; // [rsp+558h] [rbp-1C8h]
  __int64 v430; // [rsp+560h] [rbp-1C0h]
  __int64 v431; // [rsp+568h] [rbp-1B8h]
  __int64 v432; // [rsp+570h] [rbp-1B0h]
  char v433; // [rsp+578h] [rbp-1A8h]
  __m128i dest; // [rsp+580h] [rbp-1A0h] BYREF
  _QWORD v435[2]; // [rsp+590h] [rbp-190h] BYREF
  __m128i v436; // [rsp+5A0h] [rbp-180h]
  _QWORD v437[3]; // [rsp+5B0h] [rbp-170h] BYREF
  __int64 v438; // [rsp+5C8h] [rbp-158h] BYREF
  __int64 v439; // [rsp+5D0h] [rbp-150h] BYREF
  unsigned __int64 *v440; // [rsp+5D8h] [rbp-148h]
  unsigned __int64 *v441; // [rsp+5E0h] [rbp-140h]
  __int64 v442; // [rsp+5E8h] [rbp-138h]
  __int64 v443[3]; // [rsp+5F0h] [rbp-130h] BYREF
  int v444; // [rsp+608h] [rbp-118h]
  char v445; // [rsp+610h] [rbp-110h]
  __int64 v446; // [rsp+618h] [rbp-108h]
  __int64 v447; // [rsp+620h] [rbp-100h]
  __int64 v448; // [rsp+628h] [rbp-F8h]
  unsigned __int8 v449; // [rsp+630h] [rbp-F0h]
  char v450; // [rsp+631h] [rbp-EFh]
  _QWORD v451[11]; // [rsp+638h] [rbp-E8h] BYREF
  __int64 v452; // [rsp+690h] [rbp-90h]
  __int64 v453; // [rsp+698h] [rbp-88h]
  __int64 *v454; // [rsp+6A0h] [rbp-80h] BYREF
  __int64 v455; // [rsp+6A8h] [rbp-78h]
  unsigned __int64 v456; // [rsp+6B0h] [rbp-70h]
  unsigned __int64 v457; // [rsp+6B8h] [rbp-68h]
  unsigned __int64 v458; // [rsp+6C0h] [rbp-60h]
  __int64 v459; // [rsp+6C8h] [rbp-58h]
  unsigned __int64 v460; // [rsp+6D0h] [rbp-50h]
  int v461; // [rsp+6D8h] [rbp-48h]
  unsigned __int8 v462; // [rsp+6DCh] [rbp-44h]
  _BYTE v463[64]; // [rsp+6E0h] [rbp-40h] BYREF

  v338[0] = a2;
  v16 = *(__int64 **)a1;
  v338[1] = a3;
  v414 = a4;
  v394.m128i_i64[0] = a2;
  v17 = &v394;
  v394.m128i_i64[1] = a3;
  LOWORD(v418) = 261;
  v415 = (size_t)a5;
  LOWORD(v397) = 261;
  sub_EDC610((__int64)&v339, (void **)&v394, a6, (void **)&v414);
  v18 = v340;
  v340 &= ~2u;
  if ( (v18 & 1) == 0 )
  {
    v369[0].m128i_i64[0] = 1;
    v414 = 0;
    sub_9C66B0((__int64 *)&v414);
LABEL_10:
    v369[0].m128i_i64[0] = 0;
    sub_9C66B0(v369[0].m128i_i64);
    v332 = (v340 & 2) != 0;
    if ( (v340 & 2) != 0 )
LABEL_403:
      sub_248B690(&v339, (__int64)v17);
    v22 = v339;
    v339 = 0;
    v336 = v22;
    if ( !v22 )
    {
      v17 = &v394;
      v414 = (__m128i *)"Cannot get PGOReader";
      LOWORD(v418) = 261;
      v415 = 20;
      v394.m128i_i64[0] = (__int64)&unk_49D9CA8;
      v394.m128i_i64[1] = 23;
      v395 = v338[0];
      v396 = (unsigned __int64 *)&v414;
      sub_B6EB20((__int64)v16, (__int64)&v394);
      goto LABEL_5;
    }
    v23 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v22 + 8);
    if ( v23 == sub_ED6C40 )
      v24 = (*(__int64 (__fastcall **)(unsigned int *))(*(_QWORD *)v336[16] + 72LL))(v336[16]);
    else
      v24 = v23((__int64)v336);
    v25 = *v336;
    if ( (a16 & (v24 ^ 1)) == 0 )
    {
      v26 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v25 + 7);
      if ( v26 == sub_ED6C10 )
        v332 = (*(__int64 (__fastcall **)(unsigned int *))(*(_QWORD *)v336[16] + 64LL))(v336[16]);
      else
        v332 = v26((__int64)v336);
      if ( v332 )
      {
        v27 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v336 + 12);
        if ( v27 == sub_ED6D00 )
          v28 = (*(__int64 (__fastcall **)(unsigned int *))(*(_QWORD *)v336[16] + 104LL))(v336[16]);
        else
          v28 = v27((__int64)v336);
        if ( v28 )
        {
          v17 = &v394;
          v414 = (__m128i *)"Function entry profiles are not yet supported for optimization";
          LOWORD(v418) = 259;
          v394.m128i_i64[1] = 23;
          v394.m128i_i64[0] = (__int64)&unk_49D9CA8;
          v395 = v338[0];
          v396 = (unsigned __int64 *)&v414;
          sub_B6EB20((__int64)v16, (__int64)&v394);
          v332 = 0;
          v25 = *v336;
        }
        else
        {
          if ( LOBYTE(qword_4F8A488[8]) )
          {
            v29 = *(_QWORD *)(a1 + 16);
            for ( i = a1 + 8; i != v29; v29 = *(_QWORD *)(v29 + 8) )
            {
              while ( 1 )
              {
                if ( !v29 )
                  BUG();
                if ( (*(_BYTE *)(v29 - 49) & 0x30) == 0x30 && sub_B91C10(v29 - 56, 19) )
                  break;
                v29 = *(_QWORD *)(v29 + 8);
                if ( i == v29 )
                  goto LABEL_29;
              }
              sub_ED15E0((__int64 *)&v414, v29 - 56, 0);
              sub_ED2B20(v29 - 56, v414, v415);
              sub_2240A30((unsigned __int64 *)&v414);
            }
          }
LABEL_29:
          if ( a16 )
          {
            v31 = 1;
            v32 = (int *)v336[19];
          }
          else
          {
            v31 = 0;
            v32 = (int *)v336[18];
          }
          v33 = (unsigned __int8 *)sub_BCA5C0(v32, *(__int64 **)a1, 1, 1);
          sub_BAA660((__int64 **)a1, v33, v31);
          sub_D84780(a15);
          s = v377;
          v372 = 1;
          v373 = 0;
          v374 = 0;
          v375 = 1065353216;
          v376 = 0;
          v377[0] = 0;
          if ( byte_4FEC008 )
            sub_24B1CF0((_QWORD *)a1, (unsigned __int64 *)&s);
          v341 = 0;
          v342 = 0;
          v343 = 0;
          v344 = 0;
          v345 = 0;
          v346 = 0;
          v34 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v336 + 9);
          if ( v34 == sub_ED6C70 )
            v331 = (*(__int64 (__fastcall **)(unsigned int *))(*(_QWORD *)v336[16] + 80LL))(v336[16]);
          else
            v331 = v34((__int64)v336);
          v35 = sub_C52410();
          v36 = v35 + 1;
          v37 = sub_C959E0();
          v38 = (_QWORD *)v35[2];
          if ( v38 )
          {
            v39 = v35 + 1;
            do
            {
              while ( 1 )
              {
                v40 = v38[2];
                v41 = v38[3];
                if ( v37 <= v38[4] )
                  break;
                v38 = (_QWORD *)v38[3];
                if ( !v41 )
                  goto LABEL_40;
              }
              v39 = v38;
              v38 = (_QWORD *)v38[2];
            }
            while ( v40 );
LABEL_40:
            if ( v36 != v39 && v37 >= v39[4] )
              v36 = v39;
          }
          if ( v36 != (_QWORD *)((char *)sub_C52410() + 8) )
          {
            v44 = v36[7];
            v43 = v36 + 6;
            if ( v44 )
            {
              v37 = (unsigned int)dword_4FEB708;
              v45 = v36 + 6;
              do
              {
                while ( 1 )
                {
                  v46 = *(_QWORD *)(v44 + 16);
                  v42 = *(_QWORD *)(v44 + 24);
                  if ( *(_DWORD *)(v44 + 32) >= dword_4FEB708 )
                    break;
                  v44 = *(_QWORD *)(v44 + 24);
                  if ( !v42 )
                    goto LABEL_49;
                }
                v45 = (_QWORD *)v44;
                v44 = *(_QWORD *)(v44 + 16);
              }
              while ( v46 );
LABEL_49:
              if ( v45 != v43 && dword_4FEB708 >= *((_DWORD *)v45 + 8) && *((int *)v45 + 9) > 0 )
                v331 = byte_4FEB788;
            }
          }
          v47 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v336 + 10);
          if ( v47 == sub_ED6CA0 )
            v330 = (*(__int64 (__fastcall **)(unsigned int *, unsigned __int64))(*(_QWORD *)v336[16] + 88LL))(
                     v336[16],
                     v37);
          else
            v330 = ((__int64 (__fastcall *)(unsigned int **, unsigned __int64, __int64, unsigned int **, _QWORD *))v47)(
                     v336,
                     v37,
                     v42,
                     v336,
                     v43);
          v48 = sub_C52410();
          v49 = v48 + 1;
          v50 = sub_C959E0();
          v51 = (_QWORD *)v48[2];
          if ( v51 )
          {
            v52 = v48 + 1;
            do
            {
              while ( 1 )
              {
                v53 = v51[2];
                v54 = v51[3];
                if ( v50 <= v51[4] )
                  break;
                v51 = (_QWORD *)v51[3];
                if ( !v54 )
                  goto LABEL_60;
              }
              v52 = v51;
              v51 = (_QWORD *)v51[2];
            }
            while ( v53 );
LABEL_60:
            if ( v52 != v49 && v50 >= v52[4] )
              v49 = v52;
          }
          if ( v49 != (_QWORD *)((char *)sub_C52410() + 8) )
          {
            v57 = v49[7];
            v56 = v49 + 6;
            if ( v57 )
            {
              v50 = (unsigned int)dword_4FEB628;
              v58 = v49 + 6;
              do
              {
                while ( 1 )
                {
                  v59 = *(_QWORD *)(v57 + 16);
                  v55 = *(_QWORD *)(v57 + 24);
                  if ( *(_DWORD *)(v57 + 32) >= dword_4FEB628 )
                    break;
                  v57 = *(_QWORD *)(v57 + 24);
                  if ( !v55 )
                    goto LABEL_69;
                }
                v58 = (_QWORD *)v57;
                v57 = *(_QWORD *)(v57 + 16);
              }
              while ( v59 );
LABEL_69:
              if ( v58 != v56 && dword_4FEB628 >= *((_DWORD *)v58 + 8) )
              {
                v55 = *((unsigned int *)v58 + 9);
                if ( (int)v55 > 0 )
                  v330 = qword_4FEB6A8;
              }
            }
          }
          v60 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v336 + 11);
          if ( v60 == sub_ED6CD0 )
            v333 = (*(__int64 (__fastcall **)(unsigned int *, unsigned __int64))(*(_QWORD *)v336[16] + 96LL))(
                     v336[16],
                     v50);
          else
            v333 = ((__int64 (__fastcall *)(unsigned int **, unsigned __int64, __int64, unsigned int **, _QWORD *))v60)(
                     v336,
                     v50,
                     v55,
                     v336,
                     v56);
          for ( j = *(_QWORD *)(a1 + 32); a1 + 24 != j; j = *(_QWORD *)(j + 8) )
          {
            v61 = (__m128i *)(j - 56);
            if ( !j )
              v61 = 0;
            if ( sub_B2FC80((__int64)v61) || sub_24A3530((__int64)v61) )
              continue;
            v323 = sub_BC1CD0(*a8, &unk_4F6D3F8, (__int64)v61) + 8;
            v62 = sub_BC1CD0(*a10, &unk_4F8E5A8, (__int64)v61) + 8;
            v63 = sub_BC1CD0(*a12, &unk_4F8D9A8, (__int64)v61) + 8;
            v66 = sub_BC1CD0(*a14, &unk_4F875F0, (__int64)v61) + 8;
            if ( !v333 )
              sub_F429C0((__int64)v61, 0, v62, v63, v64, v65);
            v414 = v61;
            v416 = v63;
            v415 = a1;
            v418 = v61;
            v417 = a15;
            v419 = a16;
            p_s = &s;
            sub_24DAB80(v421);
            v422 = v323;
            v423 = 0;
            v424 = 0;
            v425 = 0;
            v67 = (unsigned __int64 *)sub_22077B0(0x48u);
            v68 = v67 + 9;
            v423 = v67;
            v425 = v67 + 9;
            do
            {
              if ( v67 )
              {
                *v67 = 0;
                v67[1] = 0;
                v67[2] = 0;
              }
              v67 += 3;
            }
            while ( v67 != v68 );
            v424 = v67;
            v447 = v63;
            v433 = v333;
            dest.m128i_i64[0] = (__int64)v435;
            v436.m128i_i64[0] = (__int64)v437;
            v426 = v61;
            v427 = 0;
            v428 = 0;
            v429 = 0;
            v430 = 0;
            v431 = 0;
            v432 = 0;
            dest.m128i_i64[1] = 0;
            LOBYTE(v435[0]) = 0;
            v436.m128i_i64[1] = 0;
            LOBYTE(v437[0]) = 0;
            v438 = 0;
            v439 = (__int64)v418;
            v440 = 0;
            v441 = 0;
            v442 = 0;
            memset(v443, 0, sizeof(v443));
            v444 = 0;
            v445 = 0;
            v446 = v62;
            v448 = v66;
            v449 = v331;
            v450 = v330;
            sub_24A5080((__int64)&v439);
            v69 = v441;
            v70 = v440;
            if ( (char *)v441 - (char *)v440 <= 0 )
            {
LABEL_367:
              v75 = 0;
              sub_24A48A0(v70, v69);
            }
            else
            {
              v71 = v440;
              v72 = v441 - v440;
              v73 = v441;
              while ( 1 )
              {
                src = (char *)(8 * v72);
                v74 = (unsigned __int64 *)sub_2207800(8 * v72);
                v75 = v74;
                if ( v74 )
                  break;
                v72 >>= 1;
                if ( !v72 )
                {
                  v69 = v73;
                  v70 = v71;
                  goto LABEL_367;
                }
              }
              v76 = v72;
              v77 = (unsigned __int64 *)&src[(_QWORD)v74];
              *v74 = *v71;
              v78 = v74 + 1;
              *v71 = 0;
              if ( v77 == v75 + 1 )
              {
                v80 = v75;
              }
              else
              {
                do
                {
                  v79 = *(v78 - 1);
                  *(v78++ - 1) = 0;
                  *(v78 - 1) = v79;
                }
                while ( v77 != v78 );
                v80 = (unsigned __int64 *)&src[(_QWORD)v75 - 8];
              }
              v81 = *v80;
              *v80 = 0;
              v82 = *v71;
              *v71 = v81;
              if ( v82 )
              {
                v315 = v76;
                j_j___libc_free_0(v82);
                sub_24A7210(v71, v73, v75, v315);
              }
              else
              {
                sub_24A7210(v71, v73, v75, v76);
              }
              v83 = v75;
              do
              {
                if ( *v83 )
                  j_j___libc_free_0(*v83);
                ++v83;
              }
              while ( v77 != v83 );
            }
            j_j___libc_free_0((unsigned __int64)v75);
            v84 = (__int64 **)v441;
            v85 = (__int64 **)v440;
            if ( v440 != v441 )
            {
              do
              {
                v86 = *v85;
                if ( !*((_BYTE *)*v85 + 25) )
                {
                  if ( *((_BYTE *)v86 + 26) )
                  {
                    v87 = v86[1];
                    if ( v87 )
                    {
                      if ( sub_AA5E90(v87) && (unsigned __int8)sub_24A73E0((__int64)&v439, **v85, (*v85)[1]) )
                        *((_BYTE *)*v85 + 24) = 1;
                    }
                  }
                }
                ++v85;
              }
              while ( v84 != v85 );
              v88 = (__int64 **)v440;
              v89 = v441;
              if ( v440 != v441 )
              {
                do
                {
                  v90 = *v88;
                  if ( !*((_BYTE *)*v88 + 25) )
                  {
                    v91 = *v90;
                    if ( v445 || v91 )
                    {
                      if ( (unsigned __int8)sub_24A73E0((__int64)&v439, v91, v90[1]) )
                        *((_BYTE *)*v88 + 24) = 1;
                    }
                  }
                  ++v88;
                }
                while ( v89 != (unsigned __int64 *)v88 );
                v92 = v441;
                if ( (unsigned __int64)((char *)v441 - (char *)v440) > 8 && v331 )
                {
                  v93 = *v440;
                  *v440 = *(v441 - 1);
                  *(v92 - 1) = v93;
                }
              }
            }
            if ( v333 )
            {
              sub_315C560(&v394, v61, v331);
              ++v395;
              v451[0] = v394.m128i_i64[0];
              v399 = (unsigned __int64 *)((char *)v399 + 1);
              LOBYTE(v451[1]) = v394.m128i_i8[8];
              v451[2] = 1;
              v451[3] = v396;
              v396 = 0;
              v451[4] = v397;
              v397 = 0;
              LODWORD(v451[5]) = v398;
              LODWORD(v398) = 0;
              v451[7] = v400;
              v451[6] = 1;
              v451[8] = m;
              v400 = 0;
              LODWORD(v451[9]) = (_DWORD)v402;
              m = 0;
              LODWORD(v402) = 0;
              LOBYTE(v451[10]) = 1;
              sub_C7D6A0(0, 0, 8);
              v217 = (unsigned int)v398;
              if ( (_DWORD)v398 )
              {
                v218 = (__int64)v396;
                v219 = &v396[11 * (unsigned int)v398];
                do
                {
                  if ( *(_QWORD *)v218 != -8192 && *(_QWORD *)v218 != -4096 )
                  {
                    v220 = *(_QWORD *)(v218 + 40);
                    if ( v220 != v218 + 56 )
                      _libc_free(v220);
                    sub_C7D6A0(*(_QWORD *)(v218 + 16), 8LL * *(unsigned int *)(v218 + 32), 8);
                  }
                  v218 += 88;
                }
                while ( v219 != (unsigned __int64 *)v218 );
                v217 = (unsigned int)v398;
              }
              sub_C7D6A0((__int64)v396, 88 * v217, 8);
              if ( LOBYTE(v451[10]) && (_BYTE)qword_4FEB408 )
                sub_315DEA0(v451, 0);
            }
            else
            {
              memset(v451, 0, sizeof(v451));
            }
            v427 = 0;
            if ( &v426[4].m128i_u64[1] != (unsigned __int64 *)v426[5].m128i_i64[0] )
            {
              v94 = (__m128i *)v426[5].m128i_i64[0];
              do
              {
                v95 = v94;
                v94 = (__m128i *)v94->m128i_i64[1];
                v96 = (__int64 *)v95[2].m128i_i64[0];
                v97 = &v95[1].m128i_i64[1];
LABEL_120:
                while ( v97 != v96 )
                {
                  while ( 1 )
                  {
                    v98 = v96;
                    v96 = (__int64 *)v96[1];
                    v99 = *((unsigned __int8 *)v98 - 24);
                    if ( v99 == 86 )
                    {
                      if ( (_BYTE)qword_4FEBC88
                        && !(_BYTE)qword_4FEB5C8
                        && !v433
                        && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(v98 - 15) + 8) + 8LL) - 17 > 1 )
                      {
                        LODWORD(v427) = v427 + 1;
                      }
                      goto LABEL_120;
                    }
                    if ( (unsigned int)(v99 - 29) <= 0x39 )
                      break;
                    if ( (unsigned int)(v99 - 87) > 9 )
                      goto LABEL_503;
                    if ( v97 == v96 )
                      goto LABEL_125;
                  }
                  if ( (unsigned int)(v99 - 30) > 0x37 )
LABEL_503:
                    BUG();
                }
LABEL_125:
                ;
              }
              while ( &v426[4].m128i_u64[1] != (unsigned __int64 *)v94 );
            }
            sub_24DB2C0(&v394, v421, 1);
            v100 = v423;
            v101 = v423[3];
            *(__m128i *)(v423 + 3) = v394;
            v100[5] = v395;
            v394 = 0u;
            v395 = 0;
            if ( v101 )
            {
              j_j___libc_free_0(v101);
              if ( v394.m128i_i64[0] )
                j_j___libc_free_0(v394.m128i_u64[0]);
            }
            if ( !a16 )
            {
              sub_24DB2C0(&v394, v421, 0);
              v221 = v423;
              v222 = *v423;
              *(__m128i *)v423 = v394;
              v221[2] = v395;
              v394 = 0u;
              v395 = 0;
              if ( v222 )
              {
                j_j___libc_free_0(v222);
                if ( v394.m128i_i64[0] )
                  j_j___libc_free_0(v394.m128i_u64[0]);
              }
              if ( LOBYTE(qword_4F8A568[8]) )
              {
                sub_24DB2C0(&v394, v421, 2);
                v223 = v423;
                v224 = v423[6];
                *((__m128i *)v423 + 3) = v394;
                v223[8] = v395;
                v394 = 0u;
                v395 = 0;
                if ( v224 )
                {
                  j_j___libc_free_0(v224);
                  if ( v394.m128i_i64[0] )
                    j_j___libc_free_0(v394.m128i_u64[0]);
                }
              }
            }
            sub_ED29C0(v394.m128i_i64, (__int64)v418, 0);
            v102 = (unsigned __int64 *)dest.m128i_i64[0];
            if ( (unsigned __int64 *)v394.m128i_i64[0] == &v395 )
            {
              v225 = v394.m128i_i64[1];
              if ( v394.m128i_i64[1] )
              {
                if ( v394.m128i_i64[1] == 1 )
                  *(_BYTE *)dest.m128i_i64[0] = v395;
                else
                  memcpy((void *)dest.m128i_i64[0], &v395, v394.m128i_u64[1]);
                v225 = v394.m128i_i64[1];
                v102 = (unsigned __int64 *)dest.m128i_i64[0];
              }
              dest.m128i_i64[1] = v225;
              *((_BYTE *)v102 + v225) = 0;
              v102 = (unsigned __int64 *)v394.m128i_i64[0];
            }
            else
            {
              if ( (_QWORD *)dest.m128i_i64[0] == v435 )
              {
                dest = v394;
                v435[0] = v395;
              }
              else
              {
                v103 = v435[0];
                dest = v394;
                v435[0] = v395;
                if ( v102 )
                {
                  v394.m128i_i64[0] = (__int64)v102;
                  v395 = v103;
                  goto LABEL_134;
                }
              }
              v394.m128i_i64[0] = (__int64)&v395;
              v102 = &v395;
            }
LABEL_134:
            v394.m128i_i64[1] = 0;
            *(_BYTE *)v102 = 0;
            if ( (unsigned __int64 *)v394.m128i_i64[0] != &v395 )
              j_j___libc_free_0(v394.m128i_u64[0]);
            sub_ED2A00(v394.m128i_i64, (__int64)v418, 0);
            v104 = (unsigned __int64 *)v436.m128i_i64[0];
            if ( (unsigned __int64 *)v394.m128i_i64[0] == &v395 )
            {
              v226 = v394.m128i_i64[1];
              if ( v394.m128i_i64[1] )
              {
                if ( v394.m128i_i64[1] == 1 )
                  *(_BYTE *)v436.m128i_i64[0] = v395;
                else
                  memcpy((void *)v436.m128i_i64[0], &v395, v394.m128i_u64[1]);
                v226 = v394.m128i_i64[1];
                v104 = (unsigned __int64 *)v436.m128i_i64[0];
              }
              v436.m128i_i64[1] = v226;
              *((_BYTE *)v104 + v226) = 0;
              v104 = (unsigned __int64 *)v394.m128i_i64[0];
            }
            else
            {
              if ( (_QWORD *)v436.m128i_i64[0] == v437 )
              {
                v436 = v394;
                v437[0] = v395;
              }
              else
              {
                v105 = v437[0];
                v436 = v394;
                v437[0] = v395;
                if ( v104 )
                {
                  v394.m128i_i64[0] = (__int64)v104;
                  v395 = v105;
                  goto LABEL_140;
                }
              }
              v394.m128i_i64[0] = (__int64)&v395;
              v104 = &v395;
            }
LABEL_140:
            v394.m128i_i64[1] = 0;
            *(_BYTE *)v104 = 0;
            if ( (unsigned __int64 *)v394.m128i_i64[0] != &v395 )
              j_j___libc_free_0(v394.m128i_u64[0]);
            sub_24B12F0(&v418);
            if ( v374 )
              sub_24ABCF0((__int64)&v418);
            v453 = 0;
            v454 = 0;
            v462 = a16;
            v455 = 0;
            v456 = 0;
            v457 = 0;
            v458 = 0;
            v459 = 0;
            v460 = 0;
            v461 = 0;
            sub_24DAB80(v463);
            if ( v333 )
            {
              v109 = (__int64)v336;
              sub_24B4BC0((__int64)&v414, v336);
              goto LABEL_303;
            }
            v106 = *(__int64 **)v415;
            v353[0].m128i_i64[0] = 0;
            v320 = (__int64)v106;
            sub_ED9F50(
              (__int64)&v394,
              (__int64)v336,
              dest.m128i_i64[0],
              dest.m128i_i64[1],
              v438,
              (unsigned __int64 *)v353,
              v436.m128i_i8[0]);
            v107 = v400;
            LOBYTE(v400) = v400 & 0xFD;
            if ( (v107 & 1) != 0 )
            {
              v108 = v394.m128i_i64[0];
              v394.m128i_i64[0] = 0;
              v369[0].m128i_i64[0] = v108 | 1;
              if ( (v108 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              {
                v378.m128i_i64[0] = v108 | 1;
                v109 = (__int64)&v378;
                v369[0].m128i_i64[0] = 0;
                sub_24AC500((__int64 *)&v414, v378.m128i_i64, v353[0].m128i_i64[0]);
                sub_9C66B0(v378.m128i_i64);
                sub_9C66B0(v369[0].m128i_i64);
                v319 = 0;
                v324 = 0;
                v321 = 0;
                goto LABEL_148;
              }
              v227 = 0;
            }
            else
            {
              v369[0].m128i_i64[0] = 1;
              v378.m128i_i64[0] = 0;
              sub_9C66B0(v378.m128i_i64);
              v227 = (__int64 *)v394.m128i_i64[0];
            }
            v228 = (unsigned __int64)v454;
            v454 = v227;
            v229 = v394.m128i_i64[1];
            v394 = 0u;
            v455 = v229;
            v230 = v395;
            v395 = 0;
            v456 = v230;
            if ( v228 )
              j_j___libc_free_0(v228);
            v231 = (__int64)v396;
            v232 = v457;
            v396 = 0;
            v109 = v459;
            v457 = v231;
            v233 = v397;
            v397 = 0;
            v458 = v233;
            v234 = v398;
            v398 = 0;
            v459 = v234;
            if ( v232 )
            {
              v109 -= v232;
              j_j___libc_free_0(v232);
            }
            v235 = v399;
            v236 = v460;
            v399 = 0;
            v326 = v460;
            v460 = (unsigned __int64)v235;
            if ( v326 )
            {
              v237 = (unsigned __int64 *)(v236 + 72);
              do
              {
                v238 = *(v237 - 3);
                v239 = (unsigned __int64 *)*(v237 - 2);
                v237 -= 3;
                v240 = (unsigned __int64 *)v238;
                if ( v239 != (unsigned __int64 *)v238 )
                {
                  do
                  {
                    if ( *v240 )
                      j_j___libc_free_0(*v240);
                    v240 += 3;
                  }
                  while ( v239 != v240 );
                  v238 = *v237;
                }
                if ( v238 )
                  j_j___libc_free_0(v238);
              }
              while ( (unsigned __int64 *)v326 != v237 );
              v109 = 72;
              j_j___libc_free_0(v326);
            }
            v241 = *v454;
            if ( *v454 == -1 )
            {
              v319 = 0;
              v321 = 1;
              v324 = v332;
              goto LABEL_148;
            }
            if ( v241 == -2 )
            {
              v319 = 0;
              v321 = 2;
              v324 = v332;
            }
            else
            {
              v242 = (v455 - (__int64)v454) >> 3;
              if ( (_DWORD)v242 )
              {
                v243 = v454 + 1;
                v244 = (__int64)&v454[(unsigned int)(v242 - 1) + 1];
                v245 = 0;
                while ( 1 )
                {
                  v245 += v241;
                  if ( (__int64 *)v244 == v243 )
                    break;
                  v241 = *v243++;
                }
                v319 = v245 == 0;
              }
              else
              {
                v319 = v332;
              }
              sub_24A2C40((__int64 **)&v378, v443, 0);
              *(_DWORD *)(*(_QWORD *)(v379 + 8) + 36LL) = 2;
              sub_24A2C40((__int64 **)&v378, v443, 0);
              v109 = (__int64)&v454;
              *(_DWORD *)(*(_QWORD *)(v379 + 8) + 32LL) = 2;
              v324 = sub_24AEB60((__int64)&v414, (__int64 *)&v454, v246, v247, v248, v249);
              if ( !v324 )
              {
                v366 = 1;
                v364[0].m128i_i64[0] = (__int64)": the profile may be stale or there is a function name collision.";
                v365 = 3;
                v290 = sub_BD5D20((__int64)v414);
                v291 = v288;
                if ( v290 )
                {
                  v292 = v361;
                  v378.m128i_i64[0] = v288;
                  v360.m128i_i64[0] = (__int64)v361;
                  if ( v288 > 0xF )
                  {
                    v316 = v288;
                    v318 = v290;
                    v304 = sub_22409D0((__int64)&v360, (unsigned __int64 *)&v378, 0);
                    v290 = v318;
                    v291 = v316;
                    v360.m128i_i64[0] = v304;
                    v305 = (_QWORD *)v304;
                    v361[0] = v378.m128i_i64[0];
                    goto LABEL_494;
                  }
                  if ( v288 == 1 )
                  {
                    v289 = *(unsigned __int8 *)v290;
                    LOBYTE(v361[0]) = *v290;
                    goto LABEL_468;
                  }
                  if ( v288 )
                  {
                    v305 = v361;
LABEL_494:
                    memcpy(v305, v290, v291);
                    v288 = v378.m128i_i64[0];
                    v292 = (_QWORD *)v360.m128i_i64[0];
                  }
LABEL_468:
                  v360.m128i_i64[1] = v288;
                  *((_BYTE *)v292 + v288) = 0;
                }
                else
                {
                  LOBYTE(v361[0]) = 0;
                  v360.m128i_i64[0] = (__int64)v361;
                  v360.m128i_i64[1] = 0;
                }
                v370 = 260;
                v378.m128i_i64[0] = (__int64)"Inconsistent number of counts in ";
                v369[0].m128i_i64[0] = (__int64)&v360;
                LOWORD(v381) = 259;
                sub_9C6370(v367, &v378, v369, v289, (__int64)v367, (__int64)v290);
                sub_9C6370(v362, v367, v364, v293, (__int64)v367, v294);
                v109 = (__int64)&v356;
                v295 = *(__int64 **)(v415 + 168);
                v356.m128i_i64[1] = 0x100000017LL;
                v356.m128i_i64[0] = (__int64)&unk_49D9CA8;
                v357 = v295;
                v358 = v362;
                sub_B6EB20(v320, (__int64)&v356);
                sub_2240A30((unsigned __int64 *)&v360);
                v321 = 0;
                goto LABEL_148;
              }
              if ( v462 )
                v250 = *((_QWORD *)v336[19] + 7);
              else
                v250 = *((_QWORD *)v336[18] + 7);
              v452 = v250;
              v321 = 0;
            }
LABEL_148:
            if ( (v400 & 2) != 0 )
              sub_EDE470(&v394, v109);
            if ( (v400 & 1) != 0 )
            {
              if ( v394.m128i_i64[0] )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v394.m128i_i64[0] + 8LL))(v394.m128i_i64[0]);
            }
            else
            {
              v317 = v399;
              if ( v399 )
              {
                v110 = v399 + 9;
                do
                {
                  v111 = *(v110 - 3);
                  v112 = (unsigned __int64 *)*(v110 - 2);
                  v110 -= 3;
                  v113 = (unsigned __int64 *)v111;
                  if ( v112 != (unsigned __int64 *)v111 )
                  {
                    do
                    {
                      if ( *v113 )
                        j_j___libc_free_0(*v113);
                      v113 += 3;
                    }
                    while ( v112 != v113 );
                    v111 = *v110;
                  }
                  if ( v111 )
                    j_j___libc_free_0(v111);
                }
                while ( v317 != v110 );
                v109 = 72;
                j_j___libc_free_0((unsigned __int64)v317);
              }
              if ( v396 )
              {
                v109 = v398 - (_QWORD)v396;
                j_j___libc_free_0((unsigned __int64)v396);
              }
              if ( v394.m128i_i64[0] )
              {
                v109 = v395 - v394.m128i_i64[0];
                j_j___libc_free_0(v394.m128i_u64[0]);
              }
            }
            if ( !v324 )
              goto LABEL_303;
            if ( v319 )
            {
              v109 = 0;
              sub_B2F4C0((__int64)v61, 0, 0, 0);
              if ( v452 )
              {
                v394.m128i_i64[0] = (__int64)v61;
                v109 = (__int64)v345;
                if ( v345 == v346 )
                {
                  sub_24147A0((__int64)&v344, v345, &v394);
                }
                else
                {
                  if ( v345 )
                  {
                    *v345 = (__int64)v61;
                    v109 = (__int64)v345;
                  }
                  v109 += 8;
                  v345 = (__int64 *)v109;
                }
              }
              goto LABEL_303;
            }
            if ( v321 )
            {
              v109 = 5;
              if ( (unsigned __int8)sub_B2D610((__int64)v61, 5) )
              {
                v109 = 5;
                sub_B2D470((__int64)v61, 5);
              }
              if ( v321 == 1 )
              {
                v109 = 12;
                sub_B2CD30((__int64)v61, 12);
              }
              goto LABEL_303;
            }
            sub_24AE100((__int64)&v414);
            sub_24AE680((__int64)&v414);
            if ( !(_BYTE)qword_4FEC2A8 )
            {
              v109 = dest.m128i_i64[0];
              v251 = 0;
              sub_ED2B00((__int64)v414, (const void *)dest.m128i_i64[0], dest.m128i_u64[1]);
              do
              {
                v252 = v423;
                v253 = 0;
                if ( v460 )
                {
                  v109 = 3 * v251;
                  v254 = 0xAAAAAAAAAAAAAAABLL
                       * ((__int64)(*(_QWORD *)(v460 + 24 * v251 + 8) - *(_QWORD *)(v460 + 24 * v251)) >> 3);
                  v253 = (unsigned int)v254;
                  if ( (_DWORD)v254 )
                  {
                    if ( (_DWORD)v251 == 2
                      && 0xAAAAAAAAAAAAAAABLL * ((__int64)(v423[7] - v423[6]) >> 3) != (unsigned int)v254 )
                    {
                      v109 = LODWORD(qword_4F8EA48[8]);
                      if ( LODWORD(qword_4F8EA48[8]) )
                      {
                        sub_24DB2C0(&v394, v463, 2);
                        v296 = v423;
                        v297 = v423[6];
                        v109 = v423[8];
                        *((__m128i *)v423 + 3) = v394;
                        v296[8] = v395;
                        v394 = 0u;
                        v395 = 0;
                        if ( v297 )
                        {
                          j_j___libc_free_0(v297);
                          v109 = v395 - v394.m128i_i64[0];
                          if ( v394.m128i_i64[0] )
                            j_j___libc_free_0(v394.m128i_u64[0]);
                        }
                        v252 = v423;
                      }
                    }
                  }
                }
                v255 = &v252[3 * v251];
                v256 = v255[1];
                v257 = *v255;
                v327 = v256;
                if ( v253 == 0xAAAAAAAAAAAAAAABLL * ((__int64)(v256 - v257) >> 3) )
                {
                  if ( v256 != v257 )
                  {
                    v258 = v257 + 24;
                    for ( k = 1; ; ++k )
                    {
                      v260 = dword_4FEC0E8;
                      v261 = k - 1;
                      if ( v251 == 1 || (v260 = qword_4FEC1C8, (_DWORD)v251 != 2) )
                      {
                        v109 = *(_QWORD *)(v258 - 8);
                        sub_ED2550((__int64 **)v415, v109, (__int64)&v454, v251, v261, v260);
                        if ( v327 == v258 )
                          break;
                      }
                      else
                      {
                        v109 = *(_QWORD *)(v258 - 8);
                        sub_ED2550((__int64 **)v415, v109, (__int64)&v454, 2u, v261, qword_4F8EA48[8]);
                        if ( v258 == v327 )
                          goto LABEL_169;
                      }
                      v258 += 24LL;
                    }
                  }
                }
                else
                {
                  v271 = *(__int64 **)v415;
                  v378.m128i_i64[0] = (__int64)"\", possibly due to the use of a stale profile.";
                  LOWORD(v381) = 259;
                  v272 = (char *)sub_BD5D20((__int64)v414);
                  if ( v272 )
                  {
                    v347.m128i_i64[0] = (__int64)v348;
                    sub_24A2F70(v347.m128i_i64, v272, (__int64)&v272[v273]);
                  }
                  else
                  {
                    LOBYTE(v348[0]) = 0;
                    v347.m128i_i64[0] = (__int64)v348;
                    v347.m128i_i64[1] = 0;
                  }
                  v367[0].m128i_i64[0] = (__int64)&v347;
                  v368 = 260;
                  v362[0].m128i_i64[0] = (__int64)" profiling in \"";
                  v277 = off_49D3980[v251];
                  v363 = 259;
                  v359 = 257;
                  if ( *v277 )
                  {
                    v356.m128i_i64[0] = (__int64)v277;
                    LOBYTE(v359) = 3;
                  }
                  v355 = 1;
                  v353[0].m128i_i64[0] = (__int64)"Inconsistent number of value sites for ";
                  v354 = 3;
                  sub_9C6370(&v360, v353, &v356, v274, v275, v276);
                  sub_9C6370(v364, &v360, v362, v278, v279, v280);
                  sub_9C6370(v369, v364, v367, v281, v282, v283);
                  sub_9C6370(&v394, v369, &v378, v284, v285, v286);
                  v109 = (__int64)&v349;
                  v287 = *(__int64 **)(v415 + 168);
                  v349.m128i_i64[1] = 0x100000017LL;
                  v349.m128i_i64[0] = (__int64)&unk_49D9CA8;
                  v350 = v287;
                  v351 = &v394;
                  sub_B6EB20((__int64)v271, (__int64)&v349);
                  if ( (_QWORD *)v347.m128i_i64[0] != v348 )
                  {
                    v109 = v348[0] + 1LL;
                    j_j___libc_free_0(v347.m128i_u64[0]);
                  }
                }
                ++v251;
              }
              while ( v251 != 3 );
            }
LABEL_169:
            v114 = (const char *)&v414;
            sub_24A9C40((__int64)&v414);
            if ( v461 == 1 )
            {
              v394.m128i_i64[0] = (__int64)v61;
              v109 = (__int64)v345;
              if ( v345 == v346 )
              {
                v114 = (const char *)&v344;
                sub_24147A0((__int64)&v344, v345, &v394);
              }
              else
              {
                if ( v345 )
                {
                  *v345 = (__int64)v61;
                  v109 = (__int64)v345;
                }
                v109 += 8;
                v345 = (__int64 *)v109;
              }
            }
            else if ( v461 == 2 )
            {
              v394.m128i_i64[0] = (__int64)v61;
              v109 = (__int64)v342;
              if ( v342 == v343 )
              {
                v114 = (const char *)&v341;
                sub_24147A0((__int64)&v341, v342, &v394);
              }
              else
              {
                if ( v342 )
                {
                  *v342 = (__int64)v61;
                  v109 = (__int64)v342;
                }
                v109 += 8;
                v342 = (__int64 *)v109;
              }
            }
            if ( unk_4F8DC28 )
            {
              v115 = qword_4F8DF28[9];
              if ( !qword_4F8DF28[9]
                || (v116 = (const void *)qword_4F8DF28[8], v114 = sub_BD5D20((__int64)v61), v115 == v117)
                && (v109 = (__int64)v116, !memcmp(v114, v116, v115)) )
              {
                *(_QWORD *)v391 = 0;
                v378.m128i_i64[1] = 0x100000000LL;
                v380 = &v382;
                v378.m128i_i64[0] = (__int64)&v379;
                v381 = 0x600000000LL;
                v388 = 0;
                v390 = 0;
                v389 = v61;
                *(_DWORD *)&v391[4] = v61[5].m128i_i32[3];
                sub_B1F440((__int64)&v378);
                v118 = (__int64)&v378;
                sub_D51D90((__int64)&v394, (__int64)&v378);
                v119 = v380;
                v120 = &v380[(unsigned int)v381];
                if ( v380 != v120 )
                {
                  do
                  {
                    v121 = *--v120;
                    if ( v121 )
                    {
                      v122 = *(_QWORD *)(v121 + 24);
                      if ( v122 != v121 + 40 )
                      {
                        v322 = v121;
                        _libc_free(v122);
                        v121 = v322;
                      }
                      v118 = 80;
                      j_j___libc_free_0(v121);
                    }
                  }
                  while ( v119 != v120 );
                  v120 = v380;
                }
                if ( v120 != &v382 )
                  _libc_free((unsigned __int64)v120);
                if ( (__int64 *)v378.m128i_i64[0] != &v379 )
                  _libc_free(v378.m128i_u64[0]);
                v123 = (_QWORD *)sub_22077B0(0x118u);
                v124 = v123;
                if ( v123 )
                {
                  *v123 = 0;
                  v125 = v123 + 21;
                  v126 = v123 + 13;
                  *(v126 - 12) = 0;
                  *(v126 - 11) = 0;
                  *((_DWORD *)v126 - 20) = 0;
                  *(v126 - 9) = 0;
                  *(v126 - 8) = 0;
                  *(v126 - 7) = 0;
                  *((_DWORD *)v126 - 12) = 0;
                  *(v126 - 5) = 0;
                  *(v126 - 4) = 0;
                  *(v126 - 3) = 0;
                  *(v126 - 2) = 0;
                  *(v126 - 1) = 1;
                  do
                  {
                    if ( v126 )
                      *v126 = -4096;
                    v126 += 2;
                  }
                  while ( v126 != v125 );
                  v127 = v124 + 23;
                  v124[21] = 0;
                  v124[22] = 1;
                  do
                  {
                    if ( v127 )
                    {
                      *v127 = -4096;
                      *((_DWORD *)v127 + 2) = 0x7FFFFFFF;
                    }
                    v127 += 3;
                  }
                  while ( v127 != v124 + 35 );
                  v118 = (__int64)v61;
                  sub_FF9360(v124, (__int64)v61, (__int64)&v394, 0, 0, 0);
                }
                v128 = 8;
                v129 = (__int64 *)sub_22077B0(8u);
                v130 = v129;
                if ( v129 )
                {
                  v118 = (__int64)v61;
                  v128 = (__int64)v129;
                  sub_FE7FB0(v129, v61->m128i_i8, (__int64)v124, (__int64)&v394);
                }
                if ( unk_4F8DC28 == 1 )
                {
                  v118 = (__int64)"BlockFrequencyDAGs";
                  sub_FE7C90((__int64)v130, "BlockFrequencyDAGs", (void *)0x12);
                }
                else if ( unk_4F8DC28 == 2 )
                {
                  v306 = sub_C5F790(v128, v118);
                  v307 = sub_904010(v306, "pgo-view-counts: ");
                  v308 = sub_BD5D20((__int64)v414);
                  v310 = sub_A51340(v307, v308, v309);
                  sub_904010(v310, "\n");
                  v118 = sub_C5F790(v310, (__int64)"\n");
                  sub_FDC540(v130);
                }
                if ( v130 )
                {
                  sub_FDC110(v130);
                  v118 = 8;
                  j_j___libc_free_0((unsigned __int64)v130);
                }
                if ( v124 )
                {
                  sub_D77880((__int64)v124);
                  v118 = 280;
                  j_j___libc_free_0((unsigned __int64)v124);
                }
                sub_D786F0((__int64)&v394);
                v131 = v397;
                if ( v397 != v398 )
                {
                  v325 = v61;
                  v132 = v398;
                  do
                  {
                    v133 = *(_QWORD *)v131;
                    v134 = *(__int64 **)(*(_QWORD *)v131 + 16LL);
                    if ( *(__int64 **)(*(_QWORD *)v131 + 8LL) == v134 )
                    {
                      *(_BYTE *)(v133 + 152) = 1;
                    }
                    else
                    {
                      v135 = *(__int64 **)(*(_QWORD *)v131 + 8LL);
                      do
                      {
                        v136 = *v135++;
                        sub_D47BB0(v136, v118);
                      }
                      while ( v134 != v135 );
                      *(_BYTE *)(v133 + 152) = 1;
                      v137 = *(_QWORD *)(v133 + 8);
                      if ( v137 != *(_QWORD *)(v133 + 16) )
                        *(_QWORD *)(v133 + 16) = v137;
                    }
                    v138 = *(_QWORD *)(v133 + 32);
                    if ( v138 != *(_QWORD *)(v133 + 40) )
                      *(_QWORD *)(v133 + 40) = v138;
                    ++*(_QWORD *)(v133 + 56);
                    if ( *(_BYTE *)(v133 + 84) )
                    {
                      *(_QWORD *)v133 = 0;
                    }
                    else
                    {
                      v139 = 4 * (*(_DWORD *)(v133 + 76) - *(_DWORD *)(v133 + 80));
                      v140 = *(unsigned int *)(v133 + 72);
                      if ( v139 < 0x20 )
                        v139 = 32;
                      if ( v139 < (unsigned int)v140 )
                      {
                        sub_C8C990(v133 + 56, v118);
                      }
                      else
                      {
                        v118 = 0xFFFFFFFFLL;
                        memset(*(void **)(v133 + 64), -1, 8 * v140);
                      }
                      v141 = *(_BYTE *)(v133 + 84);
                      *(_QWORD *)v133 = 0;
                      if ( !v141 )
                        _libc_free(*(_QWORD *)(v133 + 64));
                    }
                    v142 = *(_QWORD *)(v133 + 32);
                    if ( v142 )
                    {
                      v118 = *(_QWORD *)(v133 + 48) - v142;
                      j_j___libc_free_0(v142);
                    }
                    v143 = *(_QWORD *)(v133 + 8);
                    if ( v143 )
                    {
                      v118 = *(_QWORD *)(v133 + 24) - v143;
                      j_j___libc_free_0(v143);
                    }
                    v131 += 8LL;
                  }
                  while ( v132 != v131 );
                  v61 = v325;
                  if ( v397 != v398 )
                    v398 = v397;
                }
                v144 = *(__int64 **)&v408[4];
                v145 = *(_QWORD *)&v408[4] + 16LL * v409;
                if ( *(_QWORD *)&v408[4] != v145 )
                {
                  do
                  {
                    v146 = v144[1];
                    v147 = *v144;
                    v144 += 2;
                    sub_C7D6A0(v147, v146, 16);
                  }
                  while ( (__int64 *)v145 != v144 );
                }
                v409 = 0;
                if ( (_DWORD)v403 )
                {
                  v148 = v402;
                  v410 = 0;
                  v149 = &v402[(unsigned int)v403];
                  v150 = v402 + 1;
                  v400 = *v402;
                  for ( m = v400 + 4096; v149 != v150; v148 = v402 )
                  {
                    v151 = *v150;
                    v152 = (unsigned int)(v150 - v148) >> 7;
                    v153 = 4096LL << v152;
                    if ( v152 >= 0x1E )
                      v153 = 0x40000000000LL;
                    ++v150;
                    sub_C7D6A0(v151, v153, 16);
                  }
                  LODWORD(v403) = 1;
                  sub_C7D6A0(*v148, 4096, 16);
                  v154 = *(__int64 **)&v408[4];
                  v155 = (__int64 *)(*(_QWORD *)&v408[4] + 16LL * v409);
                  if ( *(__int64 **)&v408[4] != v155 )
                  {
                    do
                    {
                      v156 = v154[1];
                      v157 = *v154;
                      v154 += 2;
                      sub_C7D6A0(v157, v156, 16);
                    }
                    while ( v155 != v154 );
                    goto LABEL_237;
                  }
                }
                else
                {
LABEL_237:
                  v155 = *(__int64 **)&v408[4];
                }
                if ( v155 != &v410 )
                  _libc_free((unsigned __int64)v155);
                if ( v402 != &v404 )
                  _libc_free((unsigned __int64)v402);
                if ( v397 )
                  j_j___libc_free_0(v397);
                v114 = (const char *)v394.m128i_i64[1];
                v109 = 16LL * (unsigned int)v396;
                sub_C7D6A0(v394.m128i_i64[1], v109, 8);
              }
            }
            v158 = dword_4FEBA28;
            if ( !dword_4FEBA28 )
              goto LABEL_250;
            v159 = qword_4F8DF28[9];
            if ( qword_4F8DF28[9] )
            {
              v302 = (const void *)qword_4F8DF28[8];
              v114 = sub_BD5D20((__int64)v61);
              if ( v159 != v303 )
                goto LABEL_250;
              v109 = (__int64)v302;
              if ( memcmp(v114, v302, v159) )
                goto LABEL_250;
              v158 = dword_4FEBA28;
            }
            if ( v158 == 1 )
            {
              if ( qword_4F8DF28[9] )
              {
                LOWORD(v397) = 257;
                v311.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v414);
                v369[0] = v311;
                v370 = 261;
                v367[0].m128i_i64[0] = (__int64)"PGORawCounts_";
                v368 = 259;
                sub_9C6370(&v378, v367, v369, v312, v313, v314);
                v360.m128i_i64[0] = (__int64)&v414;
                sub_24A46A0(v364[0].m128i_i64, byte_3F871B3);
                sub_24B7560((__int64)v362, (__int64 **)&v360, (void **)&v378, 0, (void **)&v394, (__int64)v364);
                sub_2240A30((unsigned __int64 *)v364);
                v109 = v362[0].m128i_i64[1];
                if ( v362[0].m128i_i64[1] )
                  sub_C67930(v362[0].m128i_i64[0], v362[0].m128i_i64[1], 0, 0);
                sub_2240A30((unsigned __int64 *)v362);
              }
              else
              {
                sub_24A46A0(v362[0].m128i_i64, byte_3F871B3);
                LOWORD(v397) = 257;
                v298.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v414);
                v370 = 261;
                v369[0] = v298;
                v367[0].m128i_i64[0] = (__int64)"PGORawCounts_";
                v368 = 259;
                sub_9C6370(&v378, v367, v369, v299, v300, v301);
                v109 = (__int64)&v360;
                v360.m128i_i64[0] = (__int64)&v414;
                sub_24B7560((__int64)v364, (__int64 **)&v360, (void **)&v378, 0, (void **)&v394, (__int64)v362);
                sub_2240A30((unsigned __int64 *)v364);
                sub_2240A30((unsigned __int64 *)v362);
              }
            }
            else if ( v158 == 2 )
            {
              v160 = sub_C5F790((__int64)v114, v109);
              v161 = sub_904010(v160, "pgo-view-raw-counts: ");
              v162 = sub_BD5D20((__int64)v414);
              v164 = sub_A51340(v161, v162, v163);
              sub_904010(v164, "\n");
              v349.m128i_i64[0] = (__int64)byte_3F871B3;
              v356.m128i_i64[0] = (__int64)"\t";
              v362[0].m128i_i64[0] = (__int64)&v438;
              v352 = 261;
              v367[0].m128i_i64[0] = (__int64)" Hash: ";
              v363 = 267;
              v378.m128i_i64[0] = (__int64)&dest;
              LOWORD(v381) = 260;
              v394.m128i_i64[0] = (__int64)"Dump Function ";
              v349.m128i_i64[1] = 0;
              v359 = 259;
              v368 = 259;
              LOWORD(v397) = 259;
              sub_9C6370(v369, &v394, &v378, v165, 260, v166);
              sub_9C6370(v364, v369, v367, v167, v168, v169);
              sub_9C6370(&v360, v364, v362, v170, v171, v172);
              sub_9C6370(v353, &v360, &v356, v173, v174, v175);
              sub_9C6370(&v347, v353, &v349, v176, v177, v178);
              v109 = sub_C5F790((__int64)&v347, (__int64)v353);
              sub_24A7CA0((__int64)&v439, v109, (void **)&v347);
            }
LABEL_250:
            if ( (_BYTE)qword_4FEB088 || byte_4FEB168 || (_BYTE)qword_4FEB248 )
            {
              *(_QWORD *)v408 = 0;
              v405 = 0;
              v394.m128i_i64[0] = (__int64)&v395;
              v394.m128i_i64[1] = 0x100000000LL;
              v396 = &v398;
              v397 = 0x600000000LL;
              v407 = 0;
              v406 = v61;
              *(_DWORD *)&v408[4] = v61[5].m128i_i32[3];
              sub_B1F440((__int64)&v394);
              sub_D51D90((__int64)&v378, (__int64)&v394);
              v179 = (__int64)v396;
              v180 = &v396[(unsigned int)v397];
              if ( v396 != v180 )
              {
                do
                {
                  v181 = *--v180;
                  if ( v181 )
                  {
                    v182 = *(_QWORD *)(v181 + 24);
                    if ( v182 != v181 + 40 )
                      _libc_free(v182);
                    j_j___libc_free_0(v181);
                  }
                }
                while ( (unsigned __int64 *)v179 != v180 );
                v180 = v396;
              }
              if ( v180 != &v398 )
                _libc_free((unsigned __int64)v180);
              if ( (unsigned __int64 *)v394.m128i_i64[0] != &v395 )
                _libc_free(v394.m128i_u64[0]);
              v394 = 0u;
              v183 = &v406;
              v395 = 0;
              v396 = 0;
              v397 = 0;
              v398 = 0;
              v399 = 0;
              v400 = 0;
              m = 0;
              v402 = 0;
              v403 = 0;
              v404 = 0;
              v405 = 1;
              do
              {
                *v183 = (__m128i *)-4096LL;
                v183 += 2;
              }
              while ( v183 != v411 );
              v184 = &v412;
              v411[0] = 0;
              v411[1] = 1;
              do
              {
                *(_QWORD *)v184 = -4096;
                v184 += 24;
                *((_DWORD *)v184 - 4) = 0x7FFFFFFF;
              }
              while ( v184 != &v413 );
              sub_FF9360(&v394, (__int64)v61, (__int64)&v378, 0, 0, 0);
              if ( (_BYTE)qword_4FEB248 )
                sub_24A91D0((__int64)&v414, (__int64)&v378, (__int64)&v394);
              v185 = 0;
              v186 = 0;
              if ( byte_4FEB168 )
              {
                v186 = sub_D844E0((__int64)a15);
                v185 = sub_D84500((__int64)a15);
              }
              v187 = (__int64)&v378;
              sub_24AC9E0((__int64)&v414, (__int64)&v378, (__int64)&v394, v186, v185);
              sub_D77880((__int64)&v394);
              sub_D786F0((__int64)&v378);
              v188 = v382;
              v189 = v381;
              if ( v381 != v382 )
              {
                do
                {
                  v190 = *(_QWORD *)v189;
                  v191 = *(__int64 **)(*(_QWORD *)v189 + 8LL);
                  v192 = *(__int64 **)(*(_QWORD *)v189 + 16LL);
                  if ( v191 == v192 )
                  {
                    *(_BYTE *)(v190 + 152) = 1;
                  }
                  else
                  {
                    do
                    {
                      v193 = *v191++;
                      sub_D47BB0(v193, v187);
                    }
                    while ( v192 != v191 );
                    *(_BYTE *)(v190 + 152) = 1;
                    v194 = *(_QWORD *)(v190 + 8);
                    if ( v194 != *(_QWORD *)(v190 + 16) )
                      *(_QWORD *)(v190 + 16) = v194;
                  }
                  v195 = *(_QWORD *)(v190 + 32);
                  if ( v195 != *(_QWORD *)(v190 + 40) )
                    *(_QWORD *)(v190 + 40) = v195;
                  ++*(_QWORD *)(v190 + 56);
                  if ( *(_BYTE *)(v190 + 84) )
                  {
                    *(_QWORD *)v190 = 0;
                  }
                  else
                  {
                    v196 = 4 * (*(_DWORD *)(v190 + 76) - *(_DWORD *)(v190 + 80));
                    v197 = *(unsigned int *)(v190 + 72);
                    if ( v196 < 0x20 )
                      v196 = 32;
                    if ( v196 < (unsigned int)v197 )
                    {
                      sub_C8C990(v190 + 56, v187);
                    }
                    else
                    {
                      v187 = 0xFFFFFFFFLL;
                      memset(*(void **)(v190 + 64), -1, 8 * v197);
                    }
                    v198 = *(_BYTE *)(v190 + 84);
                    *(_QWORD *)v190 = 0;
                    if ( !v198 )
                      _libc_free(*(_QWORD *)(v190 + 64));
                  }
                  v199 = *(_QWORD *)(v190 + 32);
                  if ( v199 )
                  {
                    v187 = *(_QWORD *)(v190 + 48) - v199;
                    j_j___libc_free_0(v199);
                  }
                  v200 = *(_QWORD *)(v190 + 8);
                  if ( v200 )
                  {
                    v187 = *(_QWORD *)(v190 + 24) - v200;
                    j_j___libc_free_0(v200);
                  }
                  v189 += 8LL;
                }
                while ( v188 != v189 );
                if ( v381 != v382 )
                  v382 = v381;
              }
              v201 = *(__int64 **)&v391[4];
              v202 = *(_QWORD *)&v391[4] + 16LL * v392;
              if ( *(_QWORD *)&v391[4] != v202 )
              {
                do
                {
                  v203 = v201[1];
                  v204 = *v201;
                  v201 += 2;
                  sub_C7D6A0(v204, v203, 16);
                }
                while ( (__int64 *)v202 != v201 );
              }
              v392 = 0;
              if ( v386 )
              {
                v262 = v385;
                v393 = 0;
                v263 = &v385[v386];
                v264 = v385 + 1;
                v383 = *v385;
                for ( n = v383 + 4096; v263 != v264; v262 = v385 )
                {
                  v265 = *v264;
                  v266 = (unsigned int)(v264 - v262) >> 7;
                  v267 = 4096LL << v266;
                  if ( v266 >= 0x1E )
                    v267 = 0x40000000000LL;
                  ++v264;
                  sub_C7D6A0(v265, v267, 16);
                }
                v386 = 1;
                sub_C7D6A0(*v262, 4096, 16);
                v268 = *(__int64 **)&v391[4];
                v205 = (__int64 *)(*(_QWORD *)&v391[4] + 16LL * v392);
                if ( *(__int64 **)&v391[4] != v205 )
                {
                  do
                  {
                    v269 = v268[1];
                    v270 = *v268;
                    v268 += 2;
                    sub_C7D6A0(v270, v269, 16);
                  }
                  while ( v205 != v268 );
                  goto LABEL_295;
                }
              }
              else
              {
LABEL_295:
                v205 = *(__int64 **)&v391[4];
              }
              if ( v205 != &v393 )
                _libc_free((unsigned __int64)v205);
              if ( v385 != (__int64 *)&v387 )
                _libc_free((unsigned __int64)v385);
              if ( v381 )
                j_j___libc_free_0(v381);
              v109 = 16LL * (unsigned int)v380;
              sub_C7D6A0(v378.m128i_i64[1], v109, 8);
            }
LABEL_303:
            sub_24A5B80((__int64)&v414, v109);
          }
          v206 = v342;
          for ( ii = v341; v206 != ii; ++ii )
          {
            v208 = *ii;
            sub_B2CD30(v208, 16);
          }
          v209 = v345;
          v210 = v344;
          if ( v344 != v345 )
          {
            srca = v345;
            do
            {
              while ( !(unsigned __int8)sub_B2D610(*v210, 12) )
              {
                v211 = *v210++;
                sub_B2CD30(v211, 5);
                if ( srca == v210 )
                  goto LABEL_314;
              }
              v337 = *(__int64 **)a1;
              sub_24A46A0((__int64 *)&v414, " is annotated as a hot function but the profile is cold");
              v213 = (char *)sub_BD5D20(*v210);
              if ( v213 )
              {
                v378.m128i_i64[0] = (__int64)&v379;
                sub_24A2F70(v378.m128i_i64, v213, (__int64)&v213[v212]);
              }
              else
              {
                LOBYTE(v379) = 0;
                v378.m128i_i64[0] = (__int64)&v379;
                v378.m128i_i64[1] = 0;
              }
              ++v210;
              sub_24A46A0(v369[0].m128i_i64, "Function ");
              sub_8FD5D0(&v394, (__int64)v369, &v378);
              sub_8FD5D0(v367, (__int64)&v394, &v414);
              sub_2240A30((unsigned __int64 *)&v394);
              sub_2240A30((unsigned __int64 *)v369);
              sub_2240A30((unsigned __int64 *)&v378);
              sub_2240A30((unsigned __int64 *)&v414);
              LOWORD(v418) = 260;
              v214 = *(_QWORD *)(a1 + 168);
              v394.m128i_i64[1] = 0x100000017LL;
              v414 = v367;
              v394.m128i_i64[0] = (__int64)&unk_49D9CA8;
              v395 = v214;
              v396 = (unsigned __int64 *)&v414;
              sub_B6EB20((__int64)v337, (__int64)&v394);
              sub_2240A30((unsigned __int64 *)v367);
            }
            while ( srca != v210 );
LABEL_314:
            v209 = v344;
          }
          if ( v209 )
            j_j___libc_free_0((unsigned __int64)v209);
          if ( v341 )
            j_j___libc_free_0((unsigned __int64)v341);
          v215 = v373;
          while ( v215 )
          {
            v216 = (unsigned __int64)v215;
            v215 = (_QWORD *)*v215;
            j_j___libc_free_0(v216);
          }
          memset(s, 0, 8 * v372);
          v17 = (__m128i *)v372;
          v374 = 0;
          v373 = 0;
          if ( s != v377 )
          {
            v17 = (__m128i *)(8 * v372);
            j_j___libc_free_0((unsigned __int64)s);
          }
          v25 = *v336;
        }
      }
      else
      {
        v17 = &v394;
        v414 = (__m128i *)"Not an IR level instrumentation profile";
        LOWORD(v418) = 259;
        v394.m128i_i64[1] = 23;
        v394.m128i_i64[0] = (__int64)&unk_49D9CA8;
        v395 = v338[0];
        v396 = (unsigned __int64 *)&v414;
        sub_B6EB20((__int64)v16, (__int64)&v394);
        v25 = *v336;
      }
    }
    (*((void (__fastcall **)(unsigned int **))v25 + 1))(v336);
    goto LABEL_5;
  }
  v19 = v339;
  v339 = 0;
  v414 = 0;
  v369[0].m128i_i64[0] = (unsigned __int64)v19 | 1;
  sub_9C8CB0((__int64 *)&v414);
  if ( (v369[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_10;
  v414 = (__m128i *)v16;
  v20 = v369[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  v415 = (size_t)v338;
  v17 = &v378;
  v369[0].m128i_i64[0] = 0;
  v378.m128i_i64[0] = v20;
  s = 0;
  sub_24AC320(v394.m128i_i64, &v378, (__int64)&v414);
  if ( (v394.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  v394.m128i_i64[0] = 0;
  sub_9C66B0(v394.m128i_i64);
  sub_9C66B0(v378.m128i_i64);
  sub_9C66B0((__int64 *)&s);
  sub_9C66B0(v369[0].m128i_i64);
  v332 = 0;
LABEL_5:
  if ( (v340 & 2) != 0 )
    goto LABEL_403;
  if ( v339 )
    (*((void (__fastcall **)(unsigned int **))*v339 + 1))(v339);
  return v332;
}
