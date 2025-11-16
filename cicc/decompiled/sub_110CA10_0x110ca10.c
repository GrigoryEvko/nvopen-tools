// Function: sub_110CA10
// Address: 0x110ca10
//
unsigned __int8 *__fastcall sub_110CA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r13
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rbx
  char v11; // al
  unsigned __int8 v12; // al
  unsigned __int8 *v13; // r10
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // r11
  __m128i v18; // xmm0
  unsigned __int64 v19; // xmm2_8
  __m128i v20; // xmm1
  __m128i v21; // xmm3
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r14
  int v25; // edi
  int v26; // esi
  unsigned __int8 *result; // rax
  unsigned __int8 v28; // al
  _BYTE *v29; // rax
  unsigned __int8 *v30; // r10
  unsigned __int64 v31; // rdx
  __m128i *v32; // rax
  int v33; // r8d
  __m128i *v34; // rcx
  __m128i *i; // rdx
  __int64 v36; // rax
  __m128i *v37; // rdi
  __int64 v38; // r11
  __int64 v39; // r15
  __int64 v40; // r12
  unsigned int **v41; // r14
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rax
  __int64 v45; // rbx
  _QWORD *v46; // rax
  unsigned int *v47; // r15
  __int64 v48; // r14
  __int64 v49; // rdx
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 *v52; // rax
  int v53; // eax
  int v54; // esi
  char v55; // dl
  int v56; // eax
  __int64 v57; // rax
  __int64 v58; // rax
  int v59; // eax
  int v60; // eax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rbx
  unsigned __int8 *v64; // rax
  __int64 v65; // r10
  __int64 v66; // r14
  __int64 v67; // rcx
  __int64 v68; // rcx
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // rcx
  __int64 v72; // rcx
  unsigned int v73; // eax
  __int64 v74; // rdx
  unsigned int v75; // r15d
  __int64 v76; // rsi
  _QWORD *v77; // rdx
  unsigned __int8 *v78; // rax
  __int64 *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdx
  bool v82; // al
  __int64 v83; // rdx
  int v84; // eax
  __int64 *v85; // rdi
  __int64 v86; // rax
  unsigned int **v87; // rdi
  __int64 v88; // r14
  __int64 v89; // rax
  __int64 v90; // r13
  _QWORD *v91; // rax
  __int64 v92; // rax
  __int64 v93; // r12
  __int64 v94; // r15
  __int64 v95; // r15
  __int64 v96; // rax
  __int64 v97; // rdx
  int v98; // edi
  unsigned __int8 v99; // si
  __int64 v100; // r14
  __int64 v101; // rbx
  __int64 v102; // r12
  char v103; // dl
  unsigned __int8 v104; // al
  unsigned __int8 *v105; // rdi
  __int64 v106; // rax
  int v107; // eax
  int v108; // eax
  __int64 *v109; // rax
  unsigned __int8 *v110; // rdi
  __int64 v111; // rax
  int v112; // eax
  int v113; // eax
  __int64 *v114; // rax
  int v115; // eax
  __int64 v116; // rsi
  int v117; // eax
  __int64 v118; // rdx
  __int64 v119; // rax
  __int64 v120; // rbx
  __int64 v121; // r9
  int v122; // eax
  __int64 v123; // rax
  __int64 v124; // rbx
  __int64 v125; // r15
  unsigned __int8 *v126; // rax
  unsigned __int8 *v127; // rdi
  __int64 v128; // rax
  __int64 v129; // rbx
  __int64 v130; // rax
  char v131; // dl
  __int64 v132; // rax
  __int64 v133; // rdx
  int v134; // r14d
  __int64 v135; // r14
  __int64 v136; // rbx
  __int64 v137; // rdx
  unsigned int v138; // esi
  _QWORD *v139; // rax
  __int64 v140; // rbx
  __int64 v141; // r12
  __int64 v142; // rdx
  unsigned int v143; // esi
  bool v144; // r11
  _BYTE *v145; // r10
  unsigned int **v146; // rdi
  __int64 v147; // rsi
  __int64 v148; // rax
  unsigned int **v149; // rdi
  __int64 v150; // r12
  __int64 v151; // rax
  __int64 v152; // r15
  void *v153; // r14
  __int64 v154; // r13
  __int64 v155; // rax
  __int64 v156; // rcx
  __int64 v157; // r9
  __int64 v158; // r8
  void *v159; // rdx
  char *v160; // rax
  __int64 v161; // rdx
  int v162; // eax
  int v163; // eax
  __int64 *v164; // rdx
  __int64 v165; // rax
  __int64 v166; // rax
  __int64 v167; // rbx
  unsigned __int8 *v168; // rax
  __int64 v169; // r10
  __int64 v170; // rdx
  __int64 v171; // rdx
  __int64 v172; // rdx
  __int64 v173; // rdx
  __int64 v174; // rdx
  __int64 v175; // rdx
  int v176; // eax
  int v177; // eax
  __int64 *v178; // rax
  __int64 v179; // rax
  __int64 v180; // rdx
  __int64 v181; // rax
  void **v182; // r11
  __int64 v183; // rbx
  __int64 *v184; // rax
  __int64 v185; // rdx
  __int64 v186; // rax
  void **v187; // r11
  __int64 v188; // rbx
  char v189; // al
  __int64 v190; // r11
  __int64 v191; // rcx
  __int64 v192; // r15
  __int64 v193; // rdx
  __int64 v194; // rax
  __int64 v195; // rbx
  char v196; // al
  __int64 v197; // r9
  __int64 v198; // rdx
  int v199; // ebx
  __int64 v200; // rbx
  __int64 v201; // r13
  __int64 v202; // r12
  __int64 v203; // rdx
  unsigned int v204; // esi
  __int64 v205; // rdx
  unsigned __int8 *v206; // rax
  __int64 v207; // rcx
  unsigned __int8 *v208; // rdi
  __int64 v209; // rcx
  unsigned __int8 *v210; // rsi
  unsigned int v211; // r14d
  __int64 *v212; // rax
  __int64 v213; // rax
  unsigned int **v214; // rdi
  __int64 v215; // r12
  char *v216; // rax
  __int64 v217; // rdi
  __int64 v218; // rax
  unsigned int **v219; // r10
  __int64 v220; // rax
  void **v221; // r11
  int v222; // ecx
  char *v223; // rdx
  __int64 v224; // rax
  __int64 v225; // rax
  char *v226; // rcx
  __int64 v227; // r15
  unsigned __int8 *v228; // rax
  __int64 v229; // rsi
  char v230; // al
  __int64 v231; // r9
  __int64 v232; // rdx
  int v233; // ebx
  __int64 v234; // rbx
  __int64 v235; // r13
  __int64 v236; // r12
  __int64 v237; // rdx
  unsigned int v238; // esi
  __int64 v239; // rdx
  int v240; // r14d
  __int64 v241; // r14
  __int64 v242; // rbx
  __int64 v243; // r12
  __int64 v244; // r15
  __int64 v245; // rt0
  __int64 v246; // rdx
  unsigned int v247; // esi
  __int64 v248; // rt1
  char v249; // al
  unsigned __int8 *v250; // r10
  __int64 v251; // rdx
  int v252; // r15d
  __int64 v253; // r15
  __int64 v254; // r14
  __int64 v255; // rdx
  unsigned int v256; // esi
  __int64 *v257; // rax
  char v258; // al
  unsigned __int8 *v259; // r10
  __int64 v260; // rdx
  int v261; // r8d
  __int64 v262; // r12
  __int64 v263; // rdx
  unsigned int v264; // esi
  __int64 v265; // rax
  unsigned int v266; // r15d
  __int64 v267; // r8
  __int64 v268; // r9
  __int64 v269; // r11
  unsigned __int64 v270; // rdx
  char *v271; // r8
  signed __int64 v272; // rdx
  char *v273; // r9
  __int64 v274; // rcx
  __int64 v275; // r15
  char *v276; // rax
  char v277; // al
  unsigned __int8 *v278; // r10
  __int64 v279; // rdx
  int v280; // r8d
  __int64 v281; // r12
  __int64 v282; // rdx
  unsigned int v283; // esi
  char v284; // al
  unsigned __int8 *v285; // r10
  __int64 v286; // rdx
  int v287; // r15d
  __int64 v288; // r15
  __int64 v289; // r14
  unsigned __int8 *v290; // r12
  __int64 v291; // rdx
  unsigned int v292; // esi
  unsigned __int64 v293; // rdx
  char *v294; // rax
  char *v295; // rdx
  signed __int64 v296; // rcx
  char v297; // al
  unsigned __int8 *v298; // r10
  __int64 v299; // rdx
  int v300; // r15d
  __int64 v301; // r15
  __int64 v302; // r14
  __int64 v303; // rdx
  unsigned int v304; // esi
  char v305; // al
  unsigned __int8 *v306; // r10
  __int64 v307; // rdx
  int v308; // r15d
  __int64 v309; // r15
  __int64 v310; // r14
  __int64 v311; // rdx
  unsigned int v312; // esi
  int v313; // eax
  char v314; // al
  unsigned int **v315; // r10
  __int64 v316; // rdx
  int v317; // r8d
  __int64 v318; // r15
  __int64 v319; // r12
  unsigned int *v320; // rbx
  __int64 v321; // rdx
  char *v322; // rax
  char *v323; // rdx
  size_t v324; // r15
  size_t v325; // rax
  char *v326; // r10
  char *v327; // rax
  size_t v328; // [rsp+8h] [rbp-198h]
  __int64 v329; // [rsp+10h] [rbp-190h]
  char *v330; // [rsp+10h] [rbp-190h]
  char *v331; // [rsp+10h] [rbp-190h]
  __int64 v332; // [rsp+18h] [rbp-188h]
  __int64 v333; // [rsp+18h] [rbp-188h]
  __int64 v334; // [rsp+18h] [rbp-188h]
  char *v335; // [rsp+18h] [rbp-188h]
  char *v336; // [rsp+18h] [rbp-188h]
  __int64 v337; // [rsp+20h] [rbp-180h]
  __int64 v338; // [rsp+20h] [rbp-180h]
  void **v339; // [rsp+20h] [rbp-180h]
  char v340; // [rsp+28h] [rbp-178h]
  char *v341; // [rsp+28h] [rbp-178h]
  char *v342; // [rsp+28h] [rbp-178h]
  char *v343; // [rsp+28h] [rbp-178h]
  int v344; // [rsp+30h] [rbp-170h]
  unsigned __int8 *v345; // [rsp+30h] [rbp-170h]
  unsigned int v346; // [rsp+30h] [rbp-170h]
  char *v347; // [rsp+30h] [rbp-170h]
  void **v348; // [rsp+30h] [rbp-170h]
  __int64 v349; // [rsp+30h] [rbp-170h]
  char *v350; // [rsp+30h] [rbp-170h]
  unsigned __int8 *v351; // [rsp+38h] [rbp-168h]
  __int64 v352; // [rsp+38h] [rbp-168h]
  unsigned __int8 *v353; // [rsp+38h] [rbp-168h]
  __int64 v354; // [rsp+38h] [rbp-168h]
  __int64 v355; // [rsp+38h] [rbp-168h]
  unsigned __int8 *v356; // [rsp+38h] [rbp-168h]
  unsigned __int8 *v357; // [rsp+38h] [rbp-168h]
  unsigned __int8 *v358; // [rsp+38h] [rbp-168h]
  __int64 v359; // [rsp+38h] [rbp-168h]
  int v360; // [rsp+38h] [rbp-168h]
  __int64 v361; // [rsp+38h] [rbp-168h]
  __m128i *v362; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v363; // [rsp+40h] [rbp-160h]
  __int64 v364; // [rsp+40h] [rbp-160h]
  __int64 v365; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v366; // [rsp+40h] [rbp-160h]
  __int64 v367; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v368; // [rsp+40h] [rbp-160h]
  int v369; // [rsp+40h] [rbp-160h]
  __int64 *v370; // [rsp+40h] [rbp-160h]
  __int64 v371; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v372; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v373; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v374; // [rsp+40h] [rbp-160h]
  __int64 v375; // [rsp+40h] [rbp-160h]
  __int64 v376; // [rsp+40h] [rbp-160h]
  unsigned int **v377; // [rsp+40h] [rbp-160h]
  unsigned int v378; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v379; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v380; // [rsp+40h] [rbp-160h]
  __int64 v381; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v382; // [rsp+40h] [rbp-160h]
  __int64 v383; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v384; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v385; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v386; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v387; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v388; // [rsp+40h] [rbp-160h]
  unsigned __int8 *v389; // [rsp+40h] [rbp-160h]
  __int64 v390; // [rsp+40h] [rbp-160h]
  unsigned int **v391; // [rsp+40h] [rbp-160h]
  unsigned int **v392; // [rsp+40h] [rbp-160h]
  __int64 v393; // [rsp+40h] [rbp-160h]
  __int64 v394; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v395; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v396; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v397; // [rsp+48h] [rbp-158h]
  __int64 v398; // [rsp+48h] [rbp-158h]
  __int64 v399; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v400; // [rsp+48h] [rbp-158h]
  __int64 v401; // [rsp+48h] [rbp-158h]
  __int64 v402; // [rsp+48h] [rbp-158h]
  unsigned int v403; // [rsp+48h] [rbp-158h]
  unsigned __int64 v404; // [rsp+48h] [rbp-158h]
  __int64 v405; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v406; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v407; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v408; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v409; // [rsp+48h] [rbp-158h]
  __int64 v410; // [rsp+48h] [rbp-158h]
  __int64 v411; // [rsp+48h] [rbp-158h]
  __int64 v412; // [rsp+48h] [rbp-158h]
  __int64 v413; // [rsp+48h] [rbp-158h]
  void *v414; // [rsp+48h] [rbp-158h]
  __int64 v415; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v416; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v417; // [rsp+48h] [rbp-158h]
  int v418; // [rsp+48h] [rbp-158h]
  __int64 v419; // [rsp+48h] [rbp-158h]
  unsigned __int64 v420; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v421; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v422; // [rsp+48h] [rbp-158h]
  __int64 v423; // [rsp+48h] [rbp-158h]
  void **v424; // [rsp+48h] [rbp-158h]
  void **v425; // [rsp+48h] [rbp-158h]
  __int64 v426; // [rsp+50h] [rbp-150h]
  __int64 v427; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v428; // [rsp+50h] [rbp-150h]
  __int64 v429; // [rsp+50h] [rbp-150h]
  __int64 v430; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v431; // [rsp+50h] [rbp-150h]
  __int64 v432; // [rsp+50h] [rbp-150h]
  __int64 v433; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v434; // [rsp+50h] [rbp-150h]
  __int64 v435; // [rsp+50h] [rbp-150h]
  __int64 v436; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v437; // [rsp+50h] [rbp-150h]
  __int64 v438; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v439; // [rsp+50h] [rbp-150h]
  __int64 v440; // [rsp+50h] [rbp-150h]
  __int64 v441; // [rsp+50h] [rbp-150h]
  __int64 v442; // [rsp+50h] [rbp-150h]
  __int64 v443; // [rsp+50h] [rbp-150h]
  __int64 v444; // [rsp+50h] [rbp-150h]
  __int64 v445; // [rsp+50h] [rbp-150h]
  __int64 v446; // [rsp+50h] [rbp-150h]
  unsigned int v447; // [rsp+50h] [rbp-150h]
  __int64 v448; // [rsp+50h] [rbp-150h]
  __int64 v449; // [rsp+50h] [rbp-150h]
  __int64 v450; // [rsp+50h] [rbp-150h]
  __m128i *v451; // [rsp+50h] [rbp-150h]
  __int64 v452; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v453; // [rsp+50h] [rbp-150h]
  __int64 v454; // [rsp+50h] [rbp-150h]
  __int64 v455; // [rsp+50h] [rbp-150h]
  __int64 v456; // [rsp+50h] [rbp-150h]
  __m128i *v457; // [rsp+50h] [rbp-150h]
  __int64 v458; // [rsp+50h] [rbp-150h]
  int v459; // [rsp+50h] [rbp-150h]
  __int64 v460; // [rsp+50h] [rbp-150h]
  __int64 v461; // [rsp+50h] [rbp-150h]
  __int64 v462; // [rsp+68h] [rbp-138h] BYREF
  __int64 v463; // [rsp+70h] [rbp-130h] BYREF
  __int64 v464; // [rsp+78h] [rbp-128h] BYREF
  __int64 v465; // [rsp+80h] [rbp-120h]
  char *v466; // [rsp+88h] [rbp-118h]
  char v467[32]; // [rsp+90h] [rbp-110h] BYREF
  __int16 v468; // [rsp+B0h] [rbp-F0h]
  _BYTE v469[32]; // [rsp+C0h] [rbp-E0h] BYREF
  __int16 v470; // [rsp+E0h] [rbp-C0h]
  char *v471; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v472; // [rsp+F8h] [rbp-A8h]
  __int64 *v473; // [rsp+100h] [rbp-A0h]
  __int16 v474; // [rsp+110h] [rbp-90h]
  void *src[2]; // [rsp+120h] [rbp-80h] BYREF
  __m128i v476; // [rsp+130h] [rbp-70h] BYREF
  unsigned __int64 v477; // [rsp+140h] [rbp-60h]
  __int64 v478; // [rsp+148h] [rbp-58h]
  __m128i v479; // [rsp+150h] [rbp-50h]
  __int64 v480; // [rsp+160h] [rbp-40h]

  v6 = (__m128i *)a1;
  v7 = a2;
  v8 = *(_QWORD *)(a2 - 32);
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(_QWORD *)(v8 + 8);
  v462 = v9;
  if ( v10 == v9 )
    return sub_F162A0(a1, a2, v8);
  v11 = *(_BYTE *)(v10 + 8);
  if ( *(_BYTE *)(v9 + 8) == 17 && v11 == 12 )
  {
    v28 = *(_BYTE *)v8;
    if ( (unsigned __int8)(*(_BYTE *)v8 - 67) > 1u )
    {
      v30 = (unsigned __int8 *)v8;
      goto LABEL_143;
    }
    v29 = *(_BYTE **)(v8 - 32);
    v30 = (unsigned __int8 *)v8;
    if ( *v29 != 78 )
      goto LABEL_24;
    v155 = *((_QWORD *)v29 - 4);
    v156 = *(_QWORD *)(v155 + 8);
    v438 = v155;
    v405 = v156;
    if ( (unsigned int)*(unsigned __int8 *)(v156 + 8) - 17 > 1 )
      goto LABEL_24;
    v157 = *(_QWORD *)(v156 + 24);
    v158 = *(_QWORD *)(v9 + 24);
    if ( v157 != v158 )
    {
      v370 = *(__int64 **)(v9 + 24);
      v354 = *(_QWORD *)(v156 + 24);
      src[0] = (void *)sub_BCAE30((__int64)v370);
      src[1] = v159;
      v160 = (char *)sub_BCAE30(v354);
      a5 = (__int64)v370;
      v471 = v160;
      v30 = (unsigned __int8 *)v8;
      v472 = v161;
      if ( v160 != src[0] || (_BYTE)v472 != LOBYTE(src[1]) )
      {
        v28 = *(_BYTE *)v8;
        goto LABEL_143;
      }
      a2 = *(unsigned int *)(v405 + 32);
      v218 = sub_BCDA70(v370, a2);
      v219 = *(unsigned int ***)(a1 + 32);
      v474 = 257;
      v405 = v218;
      if ( v218 != *(_QWORD *)(v438 + 8) )
      {
        v377 = v219;
        a2 = 49;
        v220 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64))(*(_QWORD *)v219[10] + 120LL))(
                 v219[10],
                 49,
                 v438,
                 v218);
        if ( !v220 )
        {
          LOWORD(v477) = 257;
          v438 = sub_B51D30(49, v438, v405, (__int64)src, 0, 0);
          v314 = sub_920620(v438);
          v315 = v377;
          if ( v314 )
          {
            v316 = (__int64)v377[12];
            v317 = *((_DWORD *)v377 + 26);
            if ( v316 )
            {
              v360 = *((_DWORD *)v377 + 26);
              sub_B99FD0(v438, 3u, v316);
              v317 = v360;
              v315 = v377;
            }
            v391 = v315;
            sub_B45150(v438, v317);
            v315 = v391;
          }
          v392 = v315;
          a2 = v438;
          (*(void (__fastcall **)(unsigned int *, __int64, char **, unsigned int *, unsigned int *))(*(_QWORD *)v315[11] + 16LL))(
            v315[11],
            v438,
            &v471,
            v315[7],
            v315[8]);
          v221 = src;
          v361 = v9;
          v318 = v7;
          v319 = v10;
          v320 = *v392;
          v393 = (__int64)&(*v392)[4 * *((unsigned int *)v392 + 2)];
          while ( (unsigned int *)v393 != v320 )
          {
            v321 = *((_QWORD *)v320 + 1);
            a2 = *v320;
            v320 += 4;
            v348 = v221;
            sub_B99FD0(v438, a2, v321);
            v221 = v348;
          }
          v10 = v319;
          v7 = v318;
          v9 = v361;
LABEL_344:
          v346 = *(_DWORD *)(v9 + 32);
          v340 = **(_BYTE **)(a1 + 88);
          v378 = *(_DWORD *)(v405 + 32);
          src[0] = &v476;
          src[1] = (void *)0x1000000000LL;
          if ( (unsigned __int64)(int)v378 > 0x10 )
          {
            a2 = (__int64)&v476;
            v339 = v221;
            sub_C8D5F0((__int64)v221, &v476, (int)v378, 4u, v158, v157);
            v221 = v339;
          }
          v222 = (int)src[1];
          v223 = (char *)src[0] + 4 * LODWORD(src[1]);
          v224 = 0;
          if ( (int)v378 > 0LL )
          {
            do
            {
              *(_DWORD *)&v223[4 * v224] = v224;
              ++v224;
            }
            while ( v378 != v224 );
            v222 = (int)src[1];
          }
          LODWORD(src[1]) = v378 + v222;
          if ( v378 > v346 )
          {
            v225 = sub_ACADE0((__int64 **)v405);
            v226 = (char *)src[0];
            v338 = v225;
            v227 = LODWORD(src[1]);
            if ( v340 )
            {
              if ( LODWORD(src[1]) > (unsigned __int64)v346 )
              {
                v226 = (char *)src[0] + 4 * (LODWORD(src[1]) - (unsigned __int64)v346);
                v227 = v346;
              }
            }
            else if ( LODWORD(src[1]) > (unsigned __int64)v346 )
            {
              v227 = v346;
            }
            goto LABEL_353;
          }
          v333 = (__int64)v221;
          v266 = v346 - v378;
          v338 = sub_AD6530(v405, a2);
          v420 = v346 - v378;
          v269 = v333;
          if ( v340 )
          {
            v270 = LODWORD(src[1]) + (unsigned __int64)(v346 - v378);
            if ( LODWORD(src[1]) )
            {
              if ( HIDWORD(src[1]) < v270 )
              {
                sub_C8D5F0(v333, &v476, v270, 4u, v267, v268);
                v269 = v333;
              }
              v271 = (char *)src[0];
              v272 = 4LL * LODWORD(src[1]);
              v273 = (char *)src[0] + v272;
              v274 = v272 >> 2;
              if ( v272 >> 2 >= v420 )
              {
                v324 = 4 * (LODWORD(src[1]) - v420);
                v325 = v272 - v324;
                v326 = (char *)src[0] + v324;
                v349 = (__int64)(v272 - v324) >> 2;
                if ( (unsigned __int64)LODWORD(src[1]) + v349 > HIDWORD(src[1]) )
                {
                  v328 = v272 - v324;
                  v331 = (char *)src[0] + v324;
                  v336 = (char *)src[0] + v272;
                  v343 = (char *)src[0];
                  sub_C8D5F0(v269, &v476, LODWORD(src[1]) + v349, 4u, (__int64)src[0], (__int64)v273);
                  v325 = v328;
                  v326 = v331;
                  v273 = v336;
                  v271 = v343;
                }
                if ( v273 != v326 )
                {
                  v330 = v273;
                  v335 = v271;
                  v342 = v326;
                  memmove((char *)src[0] + 4 * LODWORD(src[1]), v326, v325);
                  v273 = v330;
                  v271 = v335;
                  v326 = v342;
                }
                LODWORD(src[1]) += v349;
                if ( v271 != v326 )
                {
                  v350 = v271;
                  memmove(&v273[-v324], v271, v324);
                  v271 = v350;
                }
                if ( v420 )
                {
                  v327 = &v271[4 * v420];
                  while ( v327 != v271 )
                  {
                    v271 += 4;
                    *((_DWORD *)v271 - 1) = v378;
                  }
                }
              }
              else
              {
                v275 = LODWORD(src[1]) + v266;
                LODWORD(src[1]) = v275;
                if ( src[0] != v273 )
                {
                  v334 = v272 >> 2;
                  v341 = (char *)src[0] + v272;
                  v347 = (char *)src[0];
                  memcpy((char *)src[0] + 4 * v275 - v272, src[0], v272);
                  v274 = v334;
                  v273 = v341;
                  v271 = v347;
                }
                if ( v274 )
                {
                  while ( v273 != v271 )
                  {
                    v271 += 4;
                    *((_DWORD *)v271 - 1) = v378;
                  }
                }
                v276 = &v273[4 * (v420 - v274)];
                while ( v276 != v273 )
                {
                  v273 += 4;
                  *((_DWORD *)v273 - 1) = v378;
                }
              }
              goto LABEL_415;
            }
            if ( HIDWORD(src[1]) < v270 )
              sub_C8D5F0(v333, &v476, v270, 4u, v267, v268);
            if ( v346 != v378 )
            {
              v322 = (char *)src[0] + 4 * LODWORD(src[1]);
              v323 = &v322[4 * v420];
              while ( v323 != v322 )
              {
                v322 += 4;
                *((_DWORD *)v322 - 1) = v378;
              }
            }
          }
          else
          {
            v293 = v420 + LODWORD(src[1]);
            if ( v293 > HIDWORD(src[1]) )
              sub_C8D5F0(v333, &v476, v293, 4u, v267, v268);
            if ( v346 != v378 )
            {
              v294 = (char *)src[0] + 4 * LODWORD(src[1]);
              v295 = &v294[4 * v420];
              do
              {
                v294 += 4;
                *((_DWORD *)v294 - 1) = v378;
              }
              while ( v295 != v294 );
            }
          }
          LODWORD(src[1]) += v266;
LABEL_415:
          v226 = (char *)src[0];
          v227 = LODWORD(src[1]);
LABEL_353:
          v414 = v226;
          v474 = 257;
          a2 = unk_3F1FE60;
          v228 = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
          if ( v228 )
          {
            v229 = v438;
            v453 = v228;
            sub_B4E9E0((__int64)v228, v229, v338, v414, v227, (__int64)&v471, 0, 0);
            result = v453;
            if ( src[0] != &v476 )
            {
              _libc_free(src[0], v229);
              return v453;
            }
            return result;
          }
          if ( src[0] != &v476 )
            _libc_free(src[0], unk_3F1FE60);
          v30 = *(unsigned __int8 **)(v7 - 32);
          v9 = *(_QWORD *)(v7 + 8);
          v28 = *v30;
LABEL_143:
          if ( v28 == 12 || v28 == 13 )
          {
LABEL_144:
            v11 = *(_BYTE *)(v10 + 8);
            goto LABEL_3;
          }
LABEL_24:
          v31 = *(unsigned int *)(v9 + 32);
          v32 = &v476;
          src[0] = &v476;
          v33 = v31;
          src[1] = (void *)0x800000000LL;
          if ( v31 )
          {
            v34 = &v476;
            if ( v31 > 8 )
            {
              v345 = v30;
              v369 = v31;
              v404 = v31;
              sub_C8D5F0((__int64)src, &v476, v31, 8u, v31, a6);
              v34 = (__m128i *)src[0];
              v30 = v345;
              v33 = v369;
              v31 = v404;
              v32 = (__m128i *)((char *)src[0] + 8 * LODWORD(src[1]));
            }
            for ( i = (__m128i *)((char *)v34 + 8 * v31); i != v32; v32 = (__m128i *)((char *)v32 + 8) )
            {
              if ( v32 )
                v32->m128i_i64[0] = 0;
            }
            LODWORD(src[1]) = v33;
          }
          a2 = 0;
          if ( sub_10FDEB0(v30, 0, (__int64)src, *(__int64 ***)(v9 + 24), **(_BYTE **)(a1 + 88)) )
          {
            v36 = sub_AD6530(*(_QWORD *)(v7 + 8), 0);
            v37 = (__m128i *)src[0];
            v38 = v36;
            if ( LODWORD(src[1]) )
            {
              v394 = LODWORD(src[1]);
              a4 = 0;
              v39 = v36;
              v329 = v7;
              v40 = 0;
              v337 = v8;
              v332 = v10;
              v362 = v6;
              do
              {
                if ( v37->m128i_i64[v40] )
                {
                  v470 = 257;
                  v41 = (unsigned int **)v362[2].m128i_i64[0];
                  v42 = sub_BCB2D0(v41[9]);
                  a2 = v39;
                  v426 = sub_ACD640(v42, v40, 0);
                  v43 = *((_QWORD *)src[0] + v40);
                  v44 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64))(*(_QWORD *)v41[10] + 104LL))(
                          v41[10],
                          v39,
                          v43);
                  a4 = v426;
                  v45 = v44;
                  if ( !v44 )
                  {
                    v474 = 257;
                    v46 = sub_BD2C40(72, 3u);
                    v45 = (__int64)v46;
                    if ( v46 )
                      sub_B4DFA0((__int64)v46, v39, v43, v426, (__int64)&v471, 0, 0, 0);
                    a2 = v45;
                    (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)v41[11] + 16LL))(
                      v41[11],
                      v45,
                      v469,
                      v41[7],
                      v41[8]);
                    v47 = *v41;
                    v48 = (__int64)&(*v41)[4 * *((unsigned int *)v41 + 2)];
                    while ( (unsigned int *)v48 != v47 )
                    {
                      v49 = *((_QWORD *)v47 + 1);
                      a2 = *v47;
                      v47 += 4;
                      sub_B99FD0(v45, a2, v49);
                    }
                  }
                  v37 = (__m128i *)src[0];
                  v39 = v45;
                }
                ++v40;
              }
              while ( v394 != v40 );
              v8 = v337;
              v10 = v332;
              v38 = v39;
              v6 = v362;
              v7 = v329;
            }
            if ( v37 != &v476 )
            {
              v442 = v38;
              _libc_free(v37, a2);
              v38 = v442;
            }
            if ( v38 )
              return sub_F162A0((__int64)v6, v7, v38);
          }
          else if ( src[0] != &v476 )
          {
            _libc_free(src[0], 0);
          }
          goto LABEL_144;
        }
        v438 = v220;
      }
    }
    v221 = src;
    goto LABEL_344;
  }
LABEL_3:
  if ( v11 != 17 )
    goto LABEL_8;
  if ( *(_DWORD *)(v10 + 32) != 1 )
  {
LABEL_7:
    if ( !(unsigned __int8)sub_F0C3D0((__int64)v6) )
    {
      v73 = sub_BCB060(v462);
      v74 = *(_QWORD *)(v8 + 16);
      v75 = v73;
      v12 = *(_BYTE *)v8;
      if ( !v74 || *(_QWORD *)(v74 + 8) || v12 != 91 )
      {
LABEL_9:
        if ( v12 != 92 )
        {
LABEL_10:
          if ( v12 == 84 )
          {
            result = sub_110B960(v6->m128i_i64, (_BYTE *)v7, v8, a4, a5, a6);
            if ( result )
              return result;
          }
          goto LABEL_11;
        }
        v95 = *(_QWORD *)(v8 - 64);
        v96 = *(_QWORD *)(v8 + 8);
        a5 = v462;
        a4 = *(_QWORD *)(v8 + 16);
        v97 = *(_QWORD *)(v95 + 8);
        v98 = *(_DWORD *)(v96 + 32);
        v99 = *(_BYTE *)(v462 + 8);
        a6 = *(unsigned int *)(v97 + 32);
        if ( !a4 )
          goto LABEL_11;
        if ( !*(_QWORD *)(a4 + 8) && (unsigned int)v99 - 17 <= 1 )
        {
          v144 = *(_BYTE *)(v96 + 8) == 18;
          if ( *(_DWORD *)(v462 + 32) == v98
            && (_DWORD)a6 == v98
            && (v99 == 18) == v144
            && (*(_BYTE *)(v97 + 8) == 18) == v144 )
          {
            if ( (v145 = *(_BYTE **)(v8 - 32), *(_BYTE *)v95 == 78) && *(_QWORD *)(*(_QWORD *)(v95 - 32) + 8LL) == v462
              || *v145 == 78 && *(_QWORD *)(*((_QWORD *)v145 - 4) + 8LL) == v462 )
            {
              v146 = (unsigned int **)v6[2].m128i_i64[0];
              v147 = *(_QWORD *)(v8 - 64);
              v436 = *(_QWORD *)(v8 - 32);
              LOWORD(v477) = 257;
              v148 = sub_A83570(v146, v147, v462, (__int64)src);
              v149 = (unsigned int **)v6[2].m128i_i64[0];
              LOWORD(v477) = 257;
              v150 = v148;
              v151 = sub_A83570(v149, v436, v462, (__int64)src);
              v152 = *(unsigned int *)(v8 + 80);
              v153 = *(void **)(v8 + 72);
              v154 = v151;
              LOWORD(v477) = 257;
              result = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
              if ( result )
              {
                v437 = result;
                sub_B4E9E0((__int64)result, v150, v154, v153, v152, (__int64)src, 0, 0);
                return v437;
              }
              return result;
            }
          }
        }
        if ( v99 != 12 || (v98 & 1) != 0 || *(_QWORD *)(a4 + 8) || (_DWORD)a6 != *(_DWORD *)(v8 + 80) )
          goto LABEL_11;
        if ( !(unsigned __int8)sub_B4EDA0(*(int **)(v8 + 72), (unsigned int)a6, a6) )
        {
LABEL_132:
          v12 = *(_BYTE *)v8;
          goto LABEL_10;
        }
        v452 = v6[5].m128i_i64[1];
        v205 = (unsigned int)sub_BCB060(v462);
        v206 = *(unsigned __int8 **)(v452 + 32);
        v207 = *(_QWORD *)(v452 + 40);
        v208 = &v206[v207];
        v209 = v207 >> 2;
        if ( v209 > 0 )
        {
          v210 = &v206[4 * v209];
          while ( v205 != *v206 )
          {
            if ( v205 == v206[1] )
            {
              ++v206;
              break;
            }
            if ( v205 == v206[2] )
            {
              v206 += 2;
              break;
            }
            if ( v205 == v206[3] )
            {
              v206 += 3;
              break;
            }
            v206 += 4;
            if ( v210 == v206 )
              goto LABEL_450;
          }
LABEL_332:
          if ( v208 != v206 && (unsigned int)sub_BCB060(v10) == 8 )
          {
            v211 = 15;
            goto LABEL_336;
          }
LABEL_334:
          if ( (unsigned int)sub_BCB060(v10) != 1 )
            goto LABEL_132;
          v211 = 14;
LABEL_336:
          v212 = (__int64 *)sub_B43CA0(v7);
          v213 = sub_B6E160(v212, v211, (__int64)&v462, 1);
          v214 = (unsigned int **)v6[2].m128i_i64[0];
          v215 = v213;
          LOWORD(v477) = 257;
          v216 = (char *)sub_A83570(v214, v95, v462, (__int64)src);
          LOWORD(v477) = 257;
          v471 = v216;
          if ( v215 )
            v217 = *(_QWORD *)(v215 + 24);
          else
            v217 = 0;
          return (unsigned __int8 *)sub_10E0510(v217, v215, (__int64 *)&v471, 1, 0, 0, (__int64)src, 0, 0);
        }
LABEL_450:
        v296 = v208 - v206;
        if ( v208 - v206 != 2 )
        {
          if ( v296 != 3 )
          {
            if ( v296 != 1 )
              goto LABEL_334;
            goto LABEL_453;
          }
          if ( v205 == *v206 )
            goto LABEL_332;
          ++v206;
        }
        if ( v205 == *v206 )
          goto LABEL_332;
        ++v206;
LABEL_453:
        if ( v205 != *v206 )
          goto LABEL_334;
        goto LABEL_332;
      }
      v76 = v462;
      v77 = (_QWORD *)sub_986520(v8);
      v78 = (unsigned __int8 *)*v77;
      a4 = *(_QWORD *)(*v77 + 16LL);
      if ( !a4 || *(_QWORD *)(a4 + 8) )
        goto LABEL_11;
      a4 = *v78;
      if ( (unsigned __int8)a4 > 0x1Cu )
      {
        a4 = (unsigned int)(a4 - 29);
      }
      else
      {
        if ( (_BYTE)a4 != 5 )
          goto LABEL_11;
        a4 = *((unsigned __int16 *)v78 + 1);
      }
      if ( (_DWORD)a4 != 49 )
        goto LABEL_11;
      if ( (v78[7] & 0x40) != 0 )
      {
        v79 = (__int64 *)*((_QWORD *)v78 - 1);
      }
      else
      {
        a4 = 32LL * (*((_DWORD *)v78 + 1) & 0x7FFFFFF);
        v79 = (__int64 *)&v78[-a4];
      }
      a5 = *v79;
      if ( !*v79 )
        goto LABEL_11;
      a6 = v77[4];
      if ( !a6 )
        goto LABEL_11;
      v80 = v77[8];
      if ( *(_BYTE *)v80 != 17 )
        goto LABEL_11;
      a4 = *(unsigned int *)(v80 + 32);
      v344 = *(_DWORD *)(v80 + 32);
      if ( (unsigned int)a4 > 0x40 )
      {
        v359 = *v79;
        v390 = a6;
        v461 = v80;
        v313 = sub_C444A0(v80 + 24);
        v76 = v462;
        a6 = v390;
        a4 = (unsigned int)(v344 - v313);
        a5 = v359;
        if ( (unsigned int)a4 > 0x40 )
          goto LABEL_11;
        v81 = **(_QWORD **)(v461 + 24);
      }
      else
      {
        v81 = *(_QWORD *)(v80 + 24);
      }
      if ( *(_BYTE *)(v76 + 8) != 12 || v76 != *(_QWORD *)(a5 + 8) || *(_BYTE *)(*(_QWORD *)(a6 + 8) + 8LL) != 12 )
      {
LABEL_11:
        v13 = *(unsigned __int8 **)(v7 - 32);
        v14 = *((_QWORD *)v13 + 2);
        if ( !v14 )
        {
          v15 = v6[2].m128i_i64[0];
          v16 = *(_QWORD *)(v7 + 8);
LABEL_13:
          v17 = v16;
          goto LABEL_14;
        }
        if ( !*(_QWORD *)(v14 + 8) && *v13 == 90 )
        {
          if ( (v13[7] & 0x40) != 0 )
          {
            a4 = *((_QWORD *)v13 - 1);
            v100 = *(_QWORD *)a4;
            if ( *(_QWORD *)a4 )
              goto LABEL_136;
          }
          else
          {
            a4 = (__int64)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)];
            v100 = *(_QWORD *)a4;
            if ( *(_QWORD *)a4 )
            {
LABEL_136:
              a4 = *(_QWORD *)(a4 + 32);
              v16 = *(_QWORD *)(v7 + 8);
              v399 = a4;
              if ( a4 )
              {
                v101 = *(_QWORD *)(v100 + 8);
                if ( (unsigned __int8)sub_BCBCB0(v16) )
                {
                  v122 = *(_DWORD *)(v101 + 32);
                  BYTE4(v465) = *(_BYTE *)(v101 + 8) == 18;
                  LODWORD(v465) = v122;
                  v123 = sub_BCE1B0((__int64 *)v16, v465);
                  v124 = v6[2].m128i_i64[0];
                  v474 = 259;
                  v471 = "bc";
                  if ( v123 == *(_QWORD *)(v100 + 8) )
                  {
                    v125 = v100;
                  }
                  else
                  {
                    v433 = v123;
                    v125 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v124 + 80)
                                                                                        + 120LL))(
                             *(_QWORD *)(v124 + 80),
                             49,
                             v100,
                             v123);
                    if ( !v125 )
                    {
                      LOWORD(v477) = 257;
                      v125 = sub_B51D30(49, v100, v433, (__int64)src, 0, 0);
                      if ( (unsigned __int8)sub_920620(v125) )
                      {
                        v133 = *(_QWORD *)(v124 + 96);
                        v134 = *(_DWORD *)(v124 + 104);
                        if ( v133 )
                          sub_B99FD0(v125, 3u, v133);
                        sub_B45150(v125, v134);
                      }
                      (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v124 + 88) + 16LL))(
                        *(_QWORD *)(v124 + 88),
                        v125,
                        &v471,
                        *(_QWORD *)(v124 + 56),
                        *(_QWORD *)(v124 + 64));
                      v135 = *(_QWORD *)v124 + 16LL * *(unsigned int *)(v124 + 8);
                      if ( *(_QWORD *)v124 != v135 )
                      {
                        v136 = *(_QWORD *)v124;
                        do
                        {
                          v137 = *(_QWORD *)(v136 + 8);
                          v138 = *(_DWORD *)v136;
                          v136 += 16;
                          sub_B99FD0(v125, v138, v137);
                        }
                        while ( v135 != v136 );
                      }
                    }
                  }
                  LOWORD(v477) = 257;
                  v126 = (unsigned __int8 *)sub_BD2C40(72, 2u);
                  if ( !v126 )
                    goto LABEL_140;
                  v434 = v126;
                  sub_B4DE80((__int64)v126, v125, v399, (__int64)src, 0, 0);
                  result = v434;
                }
                else
                {
                  if ( *(_BYTE *)(v101 + 8) != 17
                    || (unsigned __int8)(*(_BYTE *)(v16 + 8) - 17) > 1u
                    || *(_DWORD *)(v101 + 32) != 1 )
                  {
                    goto LABEL_140;
                  }
                  LOWORD(v477) = 257;
                  result = (unsigned __int8 *)sub_B51D30(49, v100, v16, (__int64)src, 0, 0);
                }
                if ( result )
                  return result;
LABEL_140:
                v13 = *(unsigned __int8 **)(v7 - 32);
                v50 = v6[2].m128i_i64[0];
                v16 = *(_QWORD *)(v7 + 8);
                v14 = *((_QWORD *)v13 + 2);
                if ( !v14 )
                {
                  v15 = v6[2].m128i_i64[0];
                  goto LABEL_13;
                }
LABEL_47:
                if ( *(_QWORD *)(v14 + 8) )
                  goto LABEL_48;
                if ( *v13 <= 0x1Cu )
                  goto LABEL_48;
                if ( (unsigned __int8)(*v13 - 57) > 2u )
                  goto LABEL_48;
                a4 = *(unsigned __int8 *)(v16 + 8);
                v103 = *(_BYTE *)(v16 + 8);
                if ( (_BYTE)a4 != 18 && (_DWORD)a4 != 17 )
                  goto LABEL_48;
                if ( (unsigned __int8)(*(_BYTE *)(*((_QWORD *)v13 + 1) + 8LL) - 17) > 1u )
                  goto LABEL_48;
                v104 = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
                if ( v104 <= 3u || v104 == 5 || (v104 & 0xFD) == 4 )
                {
                  v105 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
                  v106 = *((_QWORD *)v105 + 2);
                  if ( !v106 || *(_QWORD *)(v106 + 8) )
                    goto LABEL_48;
                  v107 = *v105;
                  if ( (unsigned __int8)v107 > 0x1Cu )
                  {
                    v108 = v107 - 29;
                  }
                  else
                  {
                    if ( (_BYTE)v107 != 5 )
                      goto LABEL_48;
                    v108 = *((unsigned __int16 *)v105 + 1);
                  }
                  if ( v108 != 49 )
                    goto LABEL_48;
                  v400 = v13;
                  v109 = (__int64 *)sub_986520((__int64)v105);
                  v13 = v400;
                  v432 = *v109;
                  if ( !*v109 )
                    goto LABEL_48;
                  v110 = (unsigned __int8 *)*((_QWORD *)v400 - 4);
                  v111 = *((_QWORD *)v110 + 2);
                  if ( !v111 || *(_QWORD *)(v111 + 8) )
                    goto LABEL_48;
                  v112 = *v110;
                  if ( (unsigned __int8)v112 > 0x1Cu )
                  {
                    v113 = v112 - 29;
                  }
                  else
                  {
                    if ( (_BYTE)v112 != 5 )
                      goto LABEL_48;
                    v113 = *((unsigned __int16 *)v110 + 1);
                  }
                  if ( v113 == 49 )
                  {
                    v366 = v400;
                    v114 = (__int64 *)sub_986520((__int64)v110);
                    v13 = v400;
                    v401 = *v114;
                    if ( *v114 )
                    {
                      a4 = *(_QWORD *)(v432 + 8);
                      v115 = *(unsigned __int8 *)(a4 + 8);
                      if ( (unsigned int)(v115 - 17) <= 1 )
                        LOBYTE(v115) = *(_BYTE *)(**(_QWORD **)(a4 + 16) + 8LL);
                      if ( (unsigned __int8)v115 > 3u && (_BYTE)v115 != 5 && (v115 & 0xFD) != 4 )
                        goto LABEL_300;
                      v116 = *(_QWORD *)(v401 + 8);
                      a4 = *(unsigned __int8 *)(v116 + 8);
                      if ( (unsigned int)(a4 - 17) <= 1 )
                        a4 = *(unsigned __int8 *)(**(_QWORD **)(v116 + 16) + 8LL);
                      if ( (_BYTE)a4 == 12 )
                      {
                        v474 = 257;
                        v193 = *((_QWORD *)v366 - 8);
                        v446 = v193;
                        v376 = *(_QWORD *)(v401 + 8);
                        if ( v376 == *(_QWORD *)(v193 + 8) )
                        {
                          v195 = v193;
                        }
                        else
                        {
                          v356 = v13;
                          v194 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v50 + 80)
                                                                                              + 120LL))(
                                   *(_QWORD *)(v50 + 80),
                                   49,
                                   v193,
                                   v376);
                          v13 = v356;
                          v195 = v194;
                          if ( !v194 )
                          {
                            LOWORD(v477) = 257;
                            v195 = sub_B51D30(49, v446, v376, (__int64)src, 0, 0);
                            v277 = sub_920620(v195);
                            v278 = v356;
                            if ( v277 )
                            {
                              v279 = *(_QWORD *)(v50 + 96);
                              v280 = *(_DWORD *)(v50 + 104);
                              if ( v279 )
                              {
                                v459 = *(_DWORD *)(v50 + 104);
                                sub_B99FD0(v195, 3u, v279);
                                v278 = v356;
                                v280 = v459;
                              }
                              v382 = v278;
                              sub_B45150(v195, v280);
                              v278 = v382;
                            }
                            v358 = v278;
                            (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88)
                                                                                             + 16LL))(
                              *(_QWORD *)(v50 + 88),
                              v195,
                              &v471,
                              *(_QWORD *)(v50 + 56),
                              *(_QWORD *)(v50 + 64));
                            v13 = v358;
                            v460 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                            if ( *(_QWORD *)v50 != v460 )
                            {
                              v383 = v7;
                              v281 = *(_QWORD *)v50;
                              do
                              {
                                v282 = *(_QWORD *)(v281 + 8);
                                v283 = *(_DWORD *)v281;
                                v281 += 16;
                                sub_B99FD0(v195, v283, v282);
                              }
                              while ( v460 != v281 );
                              v7 = v383;
                              v13 = v358;
                            }
                          }
                        }
                        v474 = 257;
                        v447 = *v13 - 29;
                        v121 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v50 + 80)
                                                                                           + 16LL))(
                                 *(_QWORD *)(v50 + 80),
                                 v447,
                                 v195,
                                 v401);
                        if ( !v121 )
                        {
                          LOWORD(v477) = 257;
                          v448 = sub_B504D0(v447, v195, v401, (__int64)src, 0, 0);
                          v196 = sub_920620(v448);
                          v197 = v448;
                          if ( v196 )
                          {
                            v198 = *(_QWORD *)(v50 + 96);
                            v199 = *(_DWORD *)(v50 + 104);
                            if ( v198 )
                            {
                              sub_B99FD0(v448, 3u, v198);
                              v197 = v448;
                            }
                            v449 = v197;
                            sub_B45150(v197, v199);
                            v197 = v449;
                          }
                          v450 = v197;
                          (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88)
                                                                                           + 16LL))(
                            *(_QWORD *)(v50 + 88),
                            v197,
                            &v471,
                            *(_QWORD *)(v50 + 56),
                            *(_QWORD *)(v50 + 64));
                          v121 = v450;
                          v200 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                          if ( *(_QWORD *)v50 != v200 )
                          {
                            v451 = v6;
                            v201 = v121;
                            v413 = v7;
                            v202 = *(_QWORD *)v50;
                            do
                            {
                              v203 = *(_QWORD *)(v202 + 8);
                              v204 = *(_DWORD *)v202;
                              v202 += 16;
                              sub_B99FD0(v201, v204, v203);
                            }
                            while ( v200 != v202 );
                            v121 = v201;
                            v7 = v413;
                            v6 = v451;
                          }
                        }
                      }
                      else
                      {
LABEL_300:
                        if ( (_BYTE)v115 != 12 )
                          goto LABEL_48;
                        a4 = *(_QWORD *)(v401 + 8);
                        v117 = *(unsigned __int8 *)(a4 + 8);
                        if ( (unsigned int)(v117 - 17) <= 1 )
                          LOBYTE(v117) = *(_BYTE *)(**(_QWORD **)(a4 + 16) + 8LL);
                        if ( (unsigned __int8)v117 > 3u && (_BYTE)v117 != 5 && (v117 & 0xFD) != 4 )
                          goto LABEL_48;
                        v474 = 257;
                        v118 = *((_QWORD *)v366 - 4);
                        v402 = v118;
                        v367 = *(_QWORD *)(v432 + 8);
                        if ( v367 == *(_QWORD *)(v118 + 8) )
                        {
                          v120 = v118;
                        }
                        else
                        {
                          v353 = v13;
                          v119 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v50 + 80)
                                                                                              + 120LL))(
                                   *(_QWORD *)(v50 + 80),
                                   49,
                                   v118,
                                   v367);
                          v13 = v353;
                          v120 = v119;
                          if ( !v119 )
                          {
                            LOWORD(v477) = 257;
                            v120 = sub_B51D30(49, v402, v367, (__int64)src, 0, 0);
                            v258 = sub_920620(v120);
                            v259 = v353;
                            if ( v258 )
                            {
                              v260 = *(_QWORD *)(v50 + 96);
                              v261 = *(_DWORD *)(v50 + 104);
                              if ( v260 )
                              {
                                v418 = *(_DWORD *)(v50 + 104);
                                sub_B99FD0(v120, 3u, v260);
                                v259 = v353;
                                v261 = v418;
                              }
                              v380 = v259;
                              sub_B45150(v120, v261);
                              v259 = v380;
                            }
                            v357 = v259;
                            (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88)
                                                                                             + 16LL))(
                              *(_QWORD *)(v50 + 88),
                              v120,
                              &v471,
                              *(_QWORD *)(v50 + 56),
                              *(_QWORD *)(v50 + 64));
                            v13 = v357;
                            v419 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                            if ( *(_QWORD *)v50 != v419 )
                            {
                              v381 = v7;
                              v262 = *(_QWORD *)v50;
                              do
                              {
                                v263 = *(_QWORD *)(v262 + 8);
                                v264 = *(_DWORD *)v262;
                                v262 += 16;
                                sub_B99FD0(v120, v264, v263);
                              }
                              while ( v419 != v262 );
                              v7 = v381;
                              v13 = v357;
                            }
                          }
                        }
                        v474 = 257;
                        v403 = *v13 - 29;
                        v121 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v50 + 80)
                                                                                           + 16LL))(
                                 *(_QWORD *)(v50 + 80),
                                 v403,
                                 v120,
                                 v432);
                        if ( !v121 )
                        {
                          LOWORD(v477) = 257;
                          v454 = sub_B504D0(v403, v120, v432, (__int64)src, 0, 0);
                          v230 = sub_920620(v454);
                          v231 = v454;
                          if ( v230 )
                          {
                            v232 = *(_QWORD *)(v50 + 96);
                            v233 = *(_DWORD *)(v50 + 104);
                            if ( v232 )
                            {
                              sub_B99FD0(v454, 3u, v232);
                              v231 = v454;
                            }
                            v455 = v231;
                            sub_B45150(v231, v233);
                            v231 = v455;
                          }
                          v456 = v231;
                          (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88)
                                                                                           + 16LL))(
                            *(_QWORD *)(v50 + 88),
                            v231,
                            &v471,
                            *(_QWORD *)(v50 + 56),
                            *(_QWORD *)(v50 + 64));
                          v121 = v456;
                          v234 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                          if ( *(_QWORD *)v50 != v234 )
                          {
                            v457 = v6;
                            v235 = *(_QWORD *)v50;
                            v415 = v7;
                            v236 = v121;
                            do
                            {
                              v237 = *(_QWORD *)(v235 + 8);
                              v238 = *(_DWORD *)v235;
                              v235 += 16;
                              sub_B99FD0(v236, v238, v237);
                            }
                            while ( v234 != v235 );
                            v121 = v236;
                            v6 = v457;
                            v7 = v415;
                          }
                        }
                      }
                      LOWORD(v477) = 257;
                      result = (unsigned __int8 *)sub_B52260(v121, v16, (__int64)src, 0, 0);
LABEL_181:
                      if ( result )
                        return result;
                      v13 = *(unsigned __int8 **)(v7 - 32);
                      v16 = *(_QWORD *)(v7 + 8);
                      v50 = v6[2].m128i_i64[0];
                      v51 = *((_QWORD *)v13 + 2);
LABEL_49:
                      v15 = v50;
                      if ( !v51 || *(_QWORD *)(v51 + 8) || *v13 != 86 )
                        goto LABEL_13;
                      if ( (v13[7] & 0x40) != 0 )
                      {
                        v52 = (__int64 *)*((_QWORD *)v13 - 1);
                        a4 = *v52;
                        v427 = *v52;
                        if ( !*v52 )
                          goto LABEL_13;
                      }
                      else
                      {
                        v52 = (__int64 *)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)];
                        a4 = *v52;
                        v427 = *v52;
                        if ( !*v52 )
                          goto LABEL_13;
                      }
                      a4 = v52[4];
                      v395 = (unsigned __int8 *)a4;
                      if ( !a4 )
                        goto LABEL_13;
                      v17 = v16;
                      v363 = (unsigned __int8 *)v52[8];
                      if ( !v363 )
                        goto LABEL_13;
                      a4 = *(_QWORD *)(v427 + 8);
                      v53 = *(unsigned __int8 *)(v16 + 8);
                      v54 = *(unsigned __int8 *)(a4 + 8);
                      if ( (unsigned int)(v54 - 17) > 1 )
                      {
                        if ( (unsigned int)(v53 - 17) > 1 )
                        {
                          v55 = 0;
                          v56 = *(unsigned __int8 *)(*((_QWORD *)v395 + 1) + 8LL);
                          if ( v56 == 18 )
                            goto LABEL_14;
LABEL_61:
                          if ( (v56 == 17) != v55 )
                            goto LABEL_14;
LABEL_62:
                          v57 = *((_QWORD *)v395 + 2);
                          if ( v57 && !*(_QWORD *)(v57 + 8) )
                          {
                            v162 = *v395;
                            if ( (unsigned __int8)v162 > 0x1Cu )
                            {
                              v163 = v162 - 29;
                            }
                            else
                            {
                              if ( (_BYTE)v162 != 5 )
                                goto LABEL_63;
                              v163 = *((unsigned __int16 *)v395 + 1);
                            }
                            if ( v163 == 49 )
                            {
                              if ( (v395[7] & 0x40) != 0 )
                              {
                                v164 = (__int64 *)*((_QWORD *)v395 - 1);
                              }
                              else
                              {
                                a4 = (__int64)&v395[-32 * (*((_DWORD *)v395 + 1) & 0x7FFFFFF)];
                                v164 = (__int64 *)a4;
                              }
                              v165 = *v164;
                              v355 = *v164;
                              if ( *v164 )
                              {
                                if ( v16 == *(_QWORD *)(v165 + 8) && *(_BYTE *)v165 > 0x15u )
                                {
                                  v474 = 257;
                                  if ( v16 == *((_QWORD *)v363 + 1) )
                                  {
                                    v167 = (__int64)v363;
                                  }
                                  else
                                  {
                                    v406 = v13;
                                    v166 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64))(**(_QWORD **)(v50 + 80) + 120LL))(
                                             *(_QWORD *)(v50 + 80),
                                             49,
                                             v363,
                                             v16);
                                    v13 = v406;
                                    v167 = v166;
                                    if ( !v166 )
                                    {
                                      LOWORD(v477) = 257;
                                      v167 = sub_B51D30(49, (__int64)v363, v16, (__int64)src, 0, 0);
                                      v284 = sub_920620(v167);
                                      v285 = v406;
                                      if ( v284 )
                                      {
                                        v286 = *(_QWORD *)(v50 + 96);
                                        v287 = *(_DWORD *)(v50 + 104);
                                        if ( v286 )
                                        {
                                          sub_B99FD0(v167, 3u, v286);
                                          v285 = v406;
                                        }
                                        v421 = v285;
                                        sub_B45150(v167, v287);
                                        v285 = v421;
                                      }
                                      v422 = v285;
                                      (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88) + 16LL))(
                                        *(_QWORD *)(v50 + 88),
                                        v167,
                                        &v471,
                                        *(_QWORD *)(v50 + 56),
                                        *(_QWORD *)(v50 + 64));
                                      v13 = v422;
                                      v288 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                                      if ( *(_QWORD *)v50 != v288 )
                                      {
                                        v423 = v7;
                                        v289 = *(_QWORD *)v50;
                                        v290 = v13;
                                        do
                                        {
                                          v291 = *(_QWORD *)(v289 + 8);
                                          v292 = *(_DWORD *)v289;
                                          v289 += 16;
                                          sub_B99FD0(v167, v292, v291);
                                        }
                                        while ( v288 != v289 );
                                        v13 = v290;
                                        v7 = v423;
                                      }
                                    }
                                  }
                                  v407 = v13;
                                  LOWORD(v477) = 257;
                                  v168 = (unsigned __int8 *)sub_BD2C40(72, 3u);
                                  v169 = (__int64)v407;
                                  if ( v168 )
                                  {
                                    v371 = (__int64)v407;
                                    v408 = v168;
                                    sub_B44260((__int64)v168, *(_QWORD *)(v355 + 8), 57, 3u, 0, 0);
                                    if ( *((_QWORD *)v408 - 12) )
                                    {
                                      v170 = *((_QWORD *)v408 - 11);
                                      **((_QWORD **)v408 - 10) = v170;
                                      if ( v170 )
                                        *(_QWORD *)(v170 + 16) = *((_QWORD *)v408 - 10);
                                    }
                                    *((_QWORD *)v408 - 12) = v427;
                                    v171 = *(_QWORD *)(v427 + 16);
                                    *((_QWORD *)v408 - 11) = v171;
                                    if ( v171 )
                                      *(_QWORD *)(v171 + 16) = v408 - 88;
                                    *((_QWORD *)v408 - 10) = v427 + 16;
                                    *(_QWORD *)(v427 + 16) = v408 - 96;
                                    if ( *((_QWORD *)v408 - 8) )
                                    {
                                      v172 = *((_QWORD *)v408 - 7);
                                      **((_QWORD **)v408 - 6) = v172;
                                      if ( v172 )
                                        *(_QWORD *)(v172 + 16) = *((_QWORD *)v408 - 6);
                                    }
                                    *((_QWORD *)v408 - 8) = v355;
                                    v173 = *(_QWORD *)(v355 + 16);
                                    *((_QWORD *)v408 - 7) = v173;
                                    if ( v173 )
                                      *(_QWORD *)(v173 + 16) = v408 - 56;
                                    *((_QWORD *)v408 - 6) = v355 + 16;
                                    *(_QWORD *)(v355 + 16) = v408 - 64;
                                    if ( *((_QWORD *)v408 - 4) )
                                    {
                                      v174 = *((_QWORD *)v408 - 3);
                                      **((_QWORD **)v408 - 2) = v174;
                                      if ( v174 )
                                        *(_QWORD *)(v174 + 16) = *((_QWORD *)v408 - 2);
                                    }
                                    *((_QWORD *)v408 - 4) = v167;
                                    if ( v167 )
                                    {
                                      v175 = *(_QWORD *)(v167 + 16);
                                      *((_QWORD *)v408 - 3) = v175;
                                      if ( v175 )
                                        *(_QWORD *)(v175 + 16) = v408 - 24;
                                      *((_QWORD *)v408 - 2) = v167 + 16;
                                      *(_QWORD *)(v167 + 16) = v408 - 32;
                                    }
                                    sub_BD6B50(v408, (const char **)src);
                                    v168 = v408;
                                    v169 = v371;
                                  }
                                  v439 = v168;
                                  sub_B47C00((__int64)v168, v169, 0, 0);
                                  result = v439;
                                  goto LABEL_94;
                                }
                              }
                            }
                          }
LABEL_63:
                          v58 = *((_QWORD *)v363 + 2);
                          if ( !v58 || *(_QWORD *)(v58 + 8) )
                            goto LABEL_14;
                          v59 = *v363;
                          if ( (unsigned __int8)v59 > 0x1Cu )
                          {
                            v60 = v59 - 29;
                          }
                          else
                          {
                            if ( (_BYTE)v59 != 5 )
                              goto LABEL_14;
                            v60 = *((unsigned __int16 *)v363 + 1);
                          }
                          if ( v60 != 49
                            || ((v363[7] & 0x40) == 0
                              ? (a4 = (__int64)&v363[-32 * (*((_DWORD *)v363 + 1) & 0x7FFFFFF)], v61 = a4)
                              : (v61 = *((_QWORD *)v363 - 1)),
                                (v364 = *(_QWORD *)v61) == 0
                             || v16 != *(_QWORD *)(*(_QWORD *)v61 + 8LL)
                             || *(_BYTE *)v364 <= 0x15u) )
                          {
LABEL_14:
                            v18 = _mm_loadu_si128(v6 + 6);
                            v19 = _mm_loadu_si128(v6 + 8).m128i_u64[0];
                            v20 = _mm_loadu_si128(v6 + 7);
                            v21 = _mm_loadu_si128(v6 + 9);
                            v480 = v6[10].m128i_i64[0];
                            v477 = v19;
                            *(__m128i *)src = v18;
                            v478 = v7;
                            v476 = v20;
                            v479 = v21;
                            v22 = *(unsigned __int8 *)(v17 + 8);
                            v23 = (unsigned int)(v22 - 17);
                            if ( (unsigned int)v23 <= 1 )
                              LOBYTE(v22) = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
                            if ( (unsigned __int8)v22 > 3u && (_BYTE)v22 != 5 && (v22 & 0xFD) != 4 )
                              return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                            v24 = *(_QWORD *)(v7 - 32);
                            v472 = 0;
                            v471 = (char *)&v463;
                            v473 = &v464;
                            a4 = *(_QWORD *)(v24 + 8);
                            v25 = *(unsigned __int8 *)(v17 + 8);
                            v23 = (unsigned int)(v25 - 17);
                            v26 = *(unsigned __int8 *)(a4 + 8);
                            LOBYTE(a5) = (unsigned int)v23 <= 1;
                            LOBYTE(v23) = (unsigned int)(v26 - 17) <= 1;
                            if ( (_BYTE)a5 != (_BYTE)v23 )
                              return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                            if ( (unsigned int)(v26 - 17) <= 1 )
                            {
                              if ( *(_DWORD *)(v17 + 32) != *(_DWORD *)(a4 + 32) )
                                return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                              LOBYTE(v23) = (_BYTE)v25 == 18;
                              if ( ((_BYTE)v25 == 18) != ((_BYTE)v26 == 18) )
                                return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                            }
                            if ( *(_BYTE *)v24 != 58 )
                              return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                            v443 = v17;
                            v189 = sub_1105FC0((_QWORD **)&v471, 28, *(unsigned __int8 **)(v24 - 64));
                            v23 = *(_QWORD *)(v24 - 32);
                            v190 = v443;
                            if ( v189 && v23 )
                            {
                              *v473 = v23;
                            }
                            else
                            {
                              if ( !(unsigned __int8)sub_1105FC0((_QWORD **)&v471, 28, (unsigned __int8 *)v23) )
                                return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                              v265 = *(_QWORD *)(v24 - 64);
                              if ( !v265 )
                                return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                              v23 = (__int64)v473;
                              v190 = v443;
                              *v473 = v265;
                            }
                            v444 = v190;
                            if ( *(_QWORD *)(v463 + 8) == v190 && (unsigned __int8)sub_9AC470(v464, (__m128i *)src, 0) )
                            {
                              HIDWORD(v466) = 0;
                              v470 = 257;
                              v468 = 257;
                              v412 = v463;
                              if ( *(_QWORD *)(v464 + 8) == v444 )
                              {
                                v192 = v464;
                              }
                              else
                              {
                                v191 = v444;
                                v375 = v444;
                                v445 = v464;
                                v192 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v15 + 80) + 120LL))(
                                         *(_QWORD *)(v15 + 80),
                                         49,
                                         v464,
                                         v191);
                                if ( !v192 )
                                {
                                  v474 = 257;
                                  v192 = sub_B51D30(49, v445, v375, (__int64)&v471, 0, 0);
                                  if ( (unsigned __int8)sub_920620(v192) )
                                  {
                                    v239 = *(_QWORD *)(v15 + 96);
                                    v240 = *(_DWORD *)(v15 + 104);
                                    if ( v239 )
                                      sub_B99FD0(v192, 3u, v239);
                                    sub_B45150(v192, v240);
                                  }
                                  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v15 + 88)
                                                                                                  + 16LL))(
                                    *(_QWORD *)(v15 + 88),
                                    v192,
                                    v467,
                                    *(_QWORD *)(v15 + 56),
                                    *(_QWORD *)(v15 + 64));
                                  if ( *(_QWORD *)v15 != *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8) )
                                  {
                                    v458 = v15;
                                    v241 = *(_QWORD *)v15;
                                    v242 = *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8);
                                    v245 = v192;
                                    v244 = v7;
                                    v243 = v245;
                                    do
                                    {
                                      v246 = *(_QWORD *)(v241 + 8);
                                      v247 = *(_DWORD *)v241;
                                      v241 += 16;
                                      sub_B99FD0(v243, v247, v246);
                                    }
                                    while ( v242 != v241 );
                                    v15 = v458;
                                    v248 = v243;
                                    v7 = v244;
                                    v192 = v248;
                                  }
                                }
                              }
                              v471 = v466;
                              v23 = sub_B33C40(v15, 0x1Au, v192, v412, (__int64)v466, (__int64)v469);
                              if ( v23 )
                                return sub_F162A0((__int64)v6, v7, v23);
                            }
                            return sub_11005E0(v6, (unsigned __int8 *)v7, v23, a4, a5, (__int64 *)a6);
                          }
                          v474 = 257;
                          if ( v16 == *((_QWORD *)v395 + 1) )
                          {
                            v63 = (__int64)v395;
                          }
                          else
                          {
                            v351 = v13;
                            v62 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64))(**(_QWORD **)(v50 + 80) + 120LL))(
                                    *(_QWORD *)(v50 + 80),
                                    49,
                                    v395,
                                    v16);
                            v13 = v351;
                            v63 = v62;
                            if ( !v62 )
                            {
                              LOWORD(v477) = 257;
                              v63 = sub_B51D30(49, (__int64)v395, v16, (__int64)src, 0, 0);
                              v249 = sub_920620(v63);
                              v250 = v351;
                              if ( v249 )
                              {
                                v251 = *(_QWORD *)(v50 + 96);
                                v252 = *(_DWORD *)(v50 + 104);
                                if ( v251 )
                                {
                                  sub_B99FD0(v63, 3u, v251);
                                  v250 = v351;
                                }
                                v416 = v250;
                                sub_B45150(v63, v252);
                                v250 = v416;
                              }
                              v417 = v250;
                              (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88)
                                                                                               + 16LL))(
                                *(_QWORD *)(v50 + 88),
                                v63,
                                &v471,
                                *(_QWORD *)(v50 + 56),
                                *(_QWORD *)(v50 + 64));
                              v253 = *(_QWORD *)v50;
                              v13 = v417;
                              v254 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                              if ( v253 != v254 )
                              {
                                do
                                {
                                  v255 = *(_QWORD *)(v253 + 8);
                                  v256 = *(_DWORD *)v253;
                                  v253 += 16;
                                  sub_B99FD0(v63, v256, v255);
                                }
                                while ( v254 != v253 );
                                v13 = v417;
                              }
                            }
                          }
                          v396 = v13;
                          LOWORD(v477) = 257;
                          v64 = (unsigned __int8 *)sub_BD2C40(72, 3u);
                          v65 = (__int64)v396;
                          if ( v64 )
                          {
                            v66 = (__int64)v64;
                            v352 = (__int64)v396;
                            v397 = v64;
                            sub_B44260((__int64)v64, *(_QWORD *)(v63 + 8), 57, 3u, 0, 0);
                            if ( *((_QWORD *)v397 - 12) )
                            {
                              v67 = *((_QWORD *)v397 - 11);
                              **((_QWORD **)v397 - 10) = v67;
                              if ( v67 )
                                *(_QWORD *)(v67 + 16) = *((_QWORD *)v397 - 10);
                            }
                            *((_QWORD *)v397 - 12) = v427;
                            v68 = *(_QWORD *)(v427 + 16);
                            *((_QWORD *)v397 - 11) = v68;
                            if ( v68 )
                              *(_QWORD *)(v68 + 16) = v397 - 88;
                            *((_QWORD *)v397 - 10) = v427 + 16;
                            *(_QWORD *)(v427 + 16) = v397 - 96;
                            if ( *((_QWORD *)v397 - 8) )
                            {
                              v69 = *((_QWORD *)v397 - 7);
                              **((_QWORD **)v397 - 6) = v69;
                              if ( v69 )
                                *(_QWORD *)(v69 + 16) = *((_QWORD *)v397 - 6);
                            }
                            *((_QWORD *)v397 - 8) = v63;
                            v70 = *(_QWORD *)(v63 + 16);
                            *((_QWORD *)v397 - 7) = v70;
                            if ( v70 )
                              *(_QWORD *)(v70 + 16) = v397 - 56;
                            *((_QWORD *)v397 - 6) = v63 + 16;
                            *(_QWORD *)(v63 + 16) = v397 - 64;
                            if ( *((_QWORD *)v397 - 4) )
                            {
                              v71 = *((_QWORD *)v397 - 3);
                              **((_QWORD **)v397 - 2) = v71;
                              if ( v71 )
                                *(_QWORD *)(v71 + 16) = *((_QWORD *)v397 - 2);
                            }
                            *((_QWORD *)v397 - 4) = v364;
                            v72 = *(_QWORD *)(v364 + 16);
                            *((_QWORD *)v397 - 3) = v72;
                            if ( v72 )
                              *(_QWORD *)(v72 + 16) = v397 - 24;
                            *((_QWORD *)v397 - 2) = v364 + 16;
                            *(_QWORD *)(v364 + 16) = v397 - 32;
                            sub_BD6B50(v397, (const char **)src);
                            v64 = v397;
                            v65 = v352;
                          }
                          else
                          {
                            v66 = 0;
                          }
                          v428 = v64;
                          sub_B47C00(v66, v65, 0, 0);
                          result = v428;
LABEL_94:
                          if ( result )
                            return result;
                          v15 = v6[2].m128i_i64[0];
                          v17 = *(_QWORD *)(v7 + 8);
                          goto LABEL_14;
                        }
                      }
                      else if ( (unsigned int)(v53 - 17) > 1
                             || *(_DWORD *)(v16 + 32) != *(_DWORD *)(a4 + 32)
                             || (*(_BYTE *)(v16 + 8) == 18) != ((_BYTE)v54 == 18) )
                      {
                        goto LABEL_14;
                      }
                      v55 = 1;
                      v56 = *(unsigned __int8 *)(*((_QWORD *)v395 + 1) + 8LL);
                      if ( v56 == 18 )
                        goto LABEL_62;
                      goto LABEL_61;
                    }
                  }
LABEL_48:
                  v51 = *((_QWORD *)v13 + 2);
                  goto LABEL_49;
                }
                a4 = (unsigned int)(a4 - 17);
                if ( (unsigned int)a4 <= 1 )
                  v103 = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
                if ( v103 != 12 )
                  goto LABEL_48;
                v127 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
                v128 = *((_QWORD *)v127 + 2);
                if ( v128 && !*(_QWORD *)(v128 + 8) )
                {
                  v176 = *v127;
                  if ( (unsigned __int8)v176 > 0x1Cu )
                  {
                    v177 = v176 - 29;
                  }
                  else
                  {
                    if ( (_BYTE)v176 != 5 )
                      goto LABEL_197;
                    v177 = *((unsigned __int16 *)v127 + 1);
                  }
                  if ( v177 == 49 )
                  {
                    v409 = v13;
                    v178 = (__int64 *)sub_986520((__int64)v127);
                    v13 = v409;
                    v179 = *v178;
                    v440 = v179;
                    if ( v179 )
                    {
                      if ( v16 == *(_QWORD *)(v179 + 8) && *(_BYTE *)v179 > 0x15u )
                      {
                        v474 = 257;
                        v180 = *((_QWORD *)v409 - 4);
                        v410 = v180;
                        if ( v16 == *(_QWORD *)(v180 + 8) )
                        {
                          v183 = v180;
                          v182 = src;
                        }
                        else
                        {
                          v372 = v13;
                          v181 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v50 + 80)
                                                                                              + 120LL))(
                                   *(_QWORD *)(v50 + 80),
                                   49,
                                   v180,
                                   v16);
                          v13 = v372;
                          v182 = src;
                          v183 = v181;
                          if ( !v181 )
                          {
                            LOWORD(v477) = 257;
                            v183 = sub_B51D30(49, v410, v16, (__int64)src, 0, 0);
                            v297 = sub_920620(v183);
                            v298 = v372;
                            if ( v297 )
                            {
                              v299 = *(_QWORD *)(v50 + 96);
                              v300 = *(_DWORD *)(v50 + 104);
                              if ( v299 )
                              {
                                sub_B99FD0(v183, 3u, v299);
                                v298 = v372;
                              }
                              v384 = v298;
                              sub_B45150(v183, v300);
                              v298 = v384;
                            }
                            v385 = v298;
                            (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88)
                                                                                             + 16LL))(
                              *(_QWORD *)(v50 + 88),
                              v183,
                              &v471,
                              *(_QWORD *)(v50 + 56),
                              *(_QWORD *)(v50 + 64));
                            v182 = src;
                            v13 = v385;
                            v301 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                            if ( *(_QWORD *)v50 != v301 )
                            {
                              v302 = *(_QWORD *)v50;
                              do
                              {
                                v303 = *(_QWORD *)(v302 + 8);
                                v304 = *(_DWORD *)v302;
                                v302 += 16;
                                v386 = v13;
                                v424 = v182;
                                sub_B99FD0(v183, v304, v303);
                                v182 = v424;
                                v13 = v386;
                              }
                              while ( v301 != v302 );
                            }
                          }
                        }
                        LOWORD(v477) = 257;
                        result = (unsigned __int8 *)sub_B504D0((unsigned int)*v13 - 29, v440, v183, (__int64)v182, 0, 0);
                        goto LABEL_181;
                      }
                    }
                  }
                }
LABEL_197:
                v129 = *((_QWORD *)v13 - 4);
                v130 = *(_QWORD *)(v129 + 16);
                v131 = *(_BYTE *)v129;
                if ( v130 && !*(_QWORD *)(v130 + 8) )
                {
                  if ( (unsigned __int8)v131 > 0x1Cu )
                  {
                    if ( v131 != 78 )
                      goto LABEL_48;
                    v379 = v13;
                    v257 = (__int64 *)sub_986520(*((_QWORD *)v13 - 4));
                    v131 = 78;
                    v13 = v379;
                    v441 = *v257;
                    if ( !*v257 )
                      goto LABEL_48;
                  }
                  else
                  {
                    if ( v131 != 5 )
                      goto LABEL_198;
                    if ( *(_WORD *)(v129 + 2) != 49 )
                      goto LABEL_199;
                    v373 = v13;
                    v184 = (__int64 *)sub_986520(*((_QWORD *)v13 - 4));
                    v131 = 5;
                    v13 = v373;
                    v441 = *v184;
                    if ( !*v184 )
                      goto LABEL_199;
                  }
                  if ( v16 == *(_QWORD *)(v441 + 8) && *(_BYTE *)v441 > 0x15u )
                  {
                    v474 = 257;
                    v185 = *((_QWORD *)v13 - 8);
                    v411 = v185;
                    if ( v16 == *(_QWORD *)(v185 + 8) )
                    {
                      v188 = *((_QWORD *)v13 - 8);
                      v187 = src;
                    }
                    else
                    {
                      v374 = v13;
                      v186 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v50 + 80)
                                                                                          + 120LL))(
                               *(_QWORD *)(v50 + 80),
                               49,
                               v185,
                               v16);
                      v13 = v374;
                      v187 = src;
                      v188 = v186;
                      if ( !v186 )
                      {
                        LOWORD(v477) = 257;
                        v188 = sub_B51D30(49, v411, v16, (__int64)src, 0, 0);
                        v305 = sub_920620(v188);
                        v306 = v374;
                        if ( v305 )
                        {
                          v307 = *(_QWORD *)(v50 + 96);
                          v308 = *(_DWORD *)(v50 + 104);
                          if ( v307 )
                          {
                            sub_B99FD0(v188, 3u, v307);
                            v306 = v374;
                          }
                          v387 = v306;
                          sub_B45150(v188, v308);
                          v306 = v387;
                        }
                        v388 = v306;
                        (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v50 + 88) + 16LL))(
                          *(_QWORD *)(v50 + 88),
                          v188,
                          &v471,
                          *(_QWORD *)(v50 + 56),
                          *(_QWORD *)(v50 + 64));
                        v187 = src;
                        v13 = v388;
                        v309 = *(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 8);
                        if ( *(_QWORD *)v50 != v309 )
                        {
                          v310 = *(_QWORD *)v50;
                          do
                          {
                            v311 = *(_QWORD *)(v310 + 8);
                            v312 = *(_DWORD *)v310;
                            v310 += 16;
                            v389 = v13;
                            v425 = v187;
                            sub_B99FD0(v188, v312, v311);
                            v187 = v425;
                            v13 = v389;
                          }
                          while ( v309 != v310 );
                        }
                      }
                    }
                    LOWORD(v477) = 257;
                    result = (unsigned __int8 *)sub_B504D0((unsigned int)*v13 - 29, v188, v441, (__int64)v187, 0, 0);
                    goto LABEL_181;
                  }
                }
LABEL_198:
                if ( (unsigned __int8)v131 <= 0x15u )
                {
LABEL_199:
                  v368 = v13;
                  LOWORD(v477) = 257;
                  v435 = sub_10FF770((__int64 *)v50, 49, *((_QWORD *)v13 - 8), v16, (__int64)src, 0, (int)v471, 0);
                  LOWORD(v477) = 257;
                  v132 = sub_10FF770((__int64 *)v50, 49, v129, v16, (__int64)src, 0, (int)v471, 0);
                  LOWORD(v477) = 257;
                  result = (unsigned __int8 *)sub_B504D0((unsigned int)*v368 - 29, v435, v132, (__int64)src, 0, 0);
                  goto LABEL_181;
                }
                goto LABEL_48;
              }
LABEL_46:
              v50 = v6[2].m128i_i64[0];
              goto LABEL_47;
            }
          }
        }
        v16 = *(_QWORD *)(v7 + 8);
        goto LABEL_46;
      }
      v365 = a5;
      v398 = a6;
      v429 = v81;
      v82 = sub_F0C740((__int64)v6, v75);
      v83 = v429;
      a6 = v398;
      a5 = v365;
      if ( v82 )
      {
        if ( *(_BYTE *)v6[5].m128i_i64[1] )
          v83 = (unsigned int)(*(_DWORD *)(v10 + 32) - 1) - v429;
        if ( !v83 )
        {
          v84 = sub_BCB060(*(_QWORD *)(v398 + 8));
          sub_109DDE0((__int64)&v471, v75, v75 - v84);
          v85 = (__int64 *)v6[2].m128i_i64[0];
          LOWORD(v477) = 257;
          v86 = sub_10BC480(v85, v365, (__int64)&v471, (__int64)src);
          v87 = (unsigned int **)v6[2].m128i_i64[0];
          v88 = v86;
          LOWORD(v477) = 257;
          v89 = sub_A82F30(v87, v398, v462, (__int64)src, 0);
          LOWORD(v477) = 257;
          v430 = sub_B504D0(29, v88, v89, (__int64)src, 0, 0);
          sub_969240((__int64 *)&v471);
          return (unsigned __int8 *)v430;
        }
      }
    }
LABEL_8:
    v12 = *(_BYTE *)v8;
    goto LABEL_9;
  }
  if ( (unsigned int)*(unsigned __int8 *)(v462 + 8) - 17 > 1 )
  {
    v90 = v6[2].m128i_i64[0];
    v474 = 257;
    v91 = (_QWORD *)sub_BD5C60(v7);
    v92 = sub_BCB2D0(v91);
    v93 = sub_AD6530(v92, a2);
    v94 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v90 + 80) + 96LL))(
            *(_QWORD *)(v90 + 80),
            v8,
            v93);
    if ( !v94 )
    {
      LOWORD(v477) = 257;
      v139 = sub_BD2C40(72, 2u);
      v94 = (__int64)v139;
      if ( v139 )
        sub_B4DE80((__int64)v139, v8, v93, (__int64)src, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v90 + 88) + 16LL))(
        *(_QWORD *)(v90 + 88),
        v94,
        &v471,
        *(_QWORD *)(v90 + 56),
        *(_QWORD *)(v90 + 64));
      v140 = *(_QWORD *)v90;
      v141 = *(_QWORD *)v90 + 16LL * *(unsigned int *)(v90 + 8);
      if ( *(_QWORD *)v90 != v141 )
      {
        do
        {
          v142 = *(_QWORD *)(v140 + 8);
          v143 = *(_DWORD *)v140;
          v140 += 16;
          sub_B99FD0(v94, v143, v142);
        }
        while ( v141 != v140 );
      }
    }
    LOWORD(v477) = 257;
    return (unsigned __int8 *)sub_B51D30(49, v94, v462, (__int64)src, 0, 0);
  }
  else
  {
    if ( *(_BYTE *)v8 != 91 )
      goto LABEL_7;
    v102 = *(_QWORD *)(v8 - 64);
    LOWORD(v477) = 257;
    result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
    if ( result )
    {
      v431 = result;
      sub_B51BF0((__int64)result, v102, v462, (__int64)src, 0, 0);
      return v431;
    }
  }
  return result;
}
