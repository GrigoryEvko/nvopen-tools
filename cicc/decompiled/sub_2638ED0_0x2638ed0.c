// Function: sub_2638ED0
// Address: 0x2638ed0
//
__int64 __fastcall sub_2638ED0(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // rax
  __int64 **v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 i; // r13
  __int64 v7; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // r12
  char *v18; // rax
  size_t v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 *v24; // r15
  __int64 *v25; // rbx
  __int64 v26; // rsi
  __int64 *v27; // r15
  __int64 v28; // rbx
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  _QWORD **v31; // r14
  _QWORD **v32; // rbx
  _QWORD *v33; // rdi
  __int64 v34; // r12
  char *v35; // rax
  size_t v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rdi
  __int64 v42; // rax
  bool v43; // zf
  __int64 v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // r14
  unsigned __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rbx
  unsigned int v50; // eax
  __int64 *v51; // rdi
  __int64 v52; // rcx
  int v53; // esi
  __int64 v54; // rdx
  __int64 *v55; // r10
  int v56; // ecx
  int v57; // r11d
  int v58; // esi
  _QWORD *v59; // rdx
  int v60; // eax
  unsigned int v61; // r12d
  unsigned int v62; // eax
  __int64 v63; // r8
  __int64 v64; // rax
  char v65; // dl
  __int64 v66; // rax
  _BYTE *v67; // r15
  __int64 v68; // r14
  __int64 *v69; // rax
  _BYTE *v70; // r13
  __int64 v71; // rdx
  __int64 v72; // rbx
  _BYTE *v73; // rax
  __int64 v74; // rdx
  unsigned __int8 *v75; // rax
  unsigned __int8 *v76; // r15
  __int64 v77; // rdi
  __int64 v78; // rsi
  char v79; // cl
  bool v80; // r9
  __int64 v81; // r10
  _QWORD *v82; // rax
  _QWORD *v83; // rsi
  __int64 v84; // rsi
  __m128i v85; // xmm1
  __m128i *v86; // r14
  __m128i *v87; // r12
  __m128i *v88; // rdi
  __int64 (__fastcall *v89)(__int64); // rax
  __m128i *v90; // r13
  _BYTE *v91; // rax
  _BYTE *v92; // rbx
  __int64 v93; // rdi
  char v94; // r14
  char *v95; // rax
  __int64 v96; // rdx
  unsigned int v97; // r15d
  int v98; // eax
  unsigned int v99; // r9d
  char v100; // r8
  size_t v101; // rsi
  unsigned int v102; // r15d
  _BYTE *v103; // r10
  char *v104; // r14
  __m128i *v105; // r11
  int j; // ebx
  __int64 v107; // r13
  const void *v108; // rdx
  bool v109; // al
  __int64 v110; // r8
  _BYTE *v111; // rax
  char v112; // r8
  __int64 v113; // r9
  void *v114; // r10
  size_t v115; // r15
  unsigned __int64 v116; // rcx
  int v117; // esi
  int v118; // r9d
  __int64 *v119; // r8
  unsigned int v120; // edx
  __int64 *v121; // rax
  _BYTE *v122; // rdi
  char *v123; // r14
  char *v124; // r15
  __int64 v125; // rax
  int v126; // edx
  _BYTE *v127; // r13
  __int64 v128; // rax
  __int64 *v129; // rax
  _BYTE *v130; // rsi
  __m128i *v131; // r14
  __m128i *v132; // r13
  __int64 *v133; // rdi
  __int64 (__fastcall *v134)(__int64); // rax
  __int64 v135; // rdi
  __int64 *v136; // r13
  __int64 k; // r12
  __int64 v138; // rcx
  __int64 v139; // rax
  __int64 v140; // rsi
  __int64 v141; // rax
  __int64 v142; // rax
  _BYTE *v143; // rsi
  unsigned __int8 *v144; // r12
  int v145; // edx
  __int64 v146; // r13
  __int64 v147; // rax
  __int64 v148; // rdx
  __int64 v149; // rbx
  int v150; // ebx
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 v153; // rdx
  int v154; // edx
  unsigned int v155; // r14d
  int v156; // eax
  __int64 v157; // r13
  __int64 v158; // rax
  __int64 v159; // rdx
  __int64 v160; // rbx
  int v161; // ebx
  __int64 v162; // rax
  __int64 v163; // rdx
  __int64 v164; // rcx
  __int64 v165; // r13
  __int64 v166; // rbx
  unsigned __int8 *v167; // r13
  bool v168; // bl
  int v169; // esi
  unsigned __int8 **v170; // rcx
  unsigned int v171; // r10d
  unsigned int v172; // edx
  unsigned __int8 **v173; // rax
  unsigned __int8 *v174; // rdi
  __int64 v175; // rcx
  __int64 v176; // rsi
  _QWORD *v177; // r13
  unsigned __int64 v178; // rcx
  unsigned __int64 v179; // rdx
  _QWORD *v180; // rax
  bool v181; // si
  bool v182; // r8
  _QWORD *v183; // rax
  unsigned __int64 v184; // rbx
  _QWORD *v185; // rsi
  unsigned __int64 v186; // rdx
  _BYTE **v187; // rcx
  _BYTE *v188; // rax
  _QWORD *v189; // rcx
  int v190; // edx
  int v191; // edx
  __int64 v192; // rax
  const void *v193; // r14
  size_t v194; // r13
  __int64 v195; // rsi
  unsigned __int64 v196; // rbx
  __int64 v197; // r8
  __int64 v198; // rax
  _QWORD *v199; // rdx
  unsigned __int64 v200; // rax
  __int64 v201; // rax
  __int64 v202; // rax
  int v203; // edx
  __int64 **v204; // rax
  __int64 *v205; // r13
  __int64 *v206; // rax
  int v207; // esi
  __int32 v208; // edx
  __int64 v209; // rdx
  _QWORD **v210; // rdx
  __int64 v211; // rax
  __int64 v212; // rbx
  _QWORD **v213; // r15
  _QWORD *v214; // rsi
  unsigned __int64 v215; // rax
  char *v216; // rdi
  unsigned __int64 v217; // r15
  unsigned __int64 v218; // r13
  int v219; // r12d
  __int64 v220; // rbx
  unsigned __int8 *v221; // rax
  unsigned __int8 *v222; // r14
  unsigned __int8 v223; // dl
  __m128i v224; // rax
  __int64 v225; // r14
  unsigned int v226; // r13d
  __int64 *v227; // rax
  unsigned __int64 v228; // rax
  unsigned int v229; // r12d
  unsigned int v230; // eax
  __int64 v231; // rdx
  __int64 v232; // rdx
  __int64 v233; // rcx
  _QWORD *v234; // r14
  __int64 v235; // rbx
  __int64 v236; // rax
  unsigned __int64 *v237; // rax
  unsigned __int64 v238; // r12
  __int64 v239; // rdx
  unsigned __int64 v240; // rax
  __int64 v241; // rdx
  __int64 v242; // rsi
  __int64 v243; // rax
  __int64 v244; // rax
  int v245; // r9d
  __int64 v246; // rdx
  __int64 v247; // r12
  unsigned __int64 v248; // r12
  unsigned __int32 v249; // eax
  __int64 v250; // rax
  __int64 v251; // rdx
  unsigned __int64 v252; // rax
  unsigned __int32 v253; // eax
  const __m128i *v254; // rsi
  __m128i *v255; // r12
  const __m128i *v256; // r14
  __int64 v257; // rbx
  unsigned __int64 v258; // rax
  __m128i *v259; // r11
  __m128i *m; // rdx
  __int64 v261; // r8
  unsigned __int32 v262; // esi
  __m128i *v263; // rcx
  __m128i *v264; // rax
  __int64 v265; // rdi
  const __m128i *v266; // r12
  unsigned __int64 v267; // r15
  __int64 v268; // rsi
  unsigned __int64 v269; // rax
  __int64 v270; // rdx
  __int64 v271; // rsi
  __int64 v272; // rsi
  __int64 v273; // r9
  char *v274; // r13
  unsigned __int128 v275; // kr10_16
  __int64 v276; // r15
  unsigned __int64 v277; // rax
  char *v278; // r15
  __int64 v279; // rdx
  __int64 v280; // rcx
  char *v281; // rax
  char *v282; // rsi
  __int64 v283; // rax
  __int64 v284; // r15
  unsigned int n; // r14d
  _BYTE *v286; // r12
  __int64 *v287; // rax
  void *v288; // rdx
  _BYTE *v289; // rax
  __m128i v290; // rax
  __int64 v291; // rdx
  __int64 v292; // rsi
  _BYTE *v293; // rax
  __int64 v294; // rdx
  unsigned __int8 *v295; // rax
  unsigned __int8 *v296; // rbx
  _BYTE *v297; // rax
  __int64 v298; // rdx
  unsigned __int8 *v299; // rax
  _BYTE *v300; // rax
  unsigned __int8 *v301; // r12
  unsigned __int8 *v302; // rax
  unsigned __int8 *v303; // rbx
  __int64 v304; // rax
  __int64 v305; // r15
  unsigned int ii; // r14d
  __int64 v307; // rbx
  __int64 v308; // rcx
  __int64 v309; // r8
  __int64 v310; // r9
  __int64 v311; // rcx
  __int64 v312; // r8
  __int64 v313; // r9
  __int64 v314; // rax
  _BYTE *v315; // r12
  __int64 *v316; // rax
  __int64 v317; // rdx
  _BYTE *v318; // rax
  __int64 v319; // r12
  __int64 v320; // rdx
  __int64 v321; // rbx
  __int64 v322; // rax
  __int64 v323; // rcx
  __int64 v324; // r8
  __int64 v325; // r9
  char v326; // al
  char v327; // al
  char v328; // al
  __int64 v329; // rdx
  __int64 *v330; // rbx
  char v331; // r13
  __int64 *v332; // r12
  unsigned __int64 v333; // rdi
  __int64 v334; // rsi
  int v335; // edx
  __int64 v336; // rdi
  __int64 *v337; // rax
  __int64 *v338; // r14
  __int64 v339; // rbx
  __int64 *v340; // r12
  __int64 *v341; // r15
  __int64 v342; // rax
  __int64 v343; // rdx
  __int64 v344; // r9
  unsigned __int64 *v345; // r8
  unsigned __int64 v346; // r13
  __int64 v347; // rax
  _QWORD *v348; // r8
  __int64 v349; // rax
  __int64 v350; // rax
  unsigned __int64 v351; // r14
  __int64 v352; // rax
  unsigned int v353; // r15d
  __m128i *v354; // [rsp+0h] [rbp-460h]
  unsigned int v355; // [rsp+Ch] [rbp-454h]
  char *v356; // [rsp+10h] [rbp-450h]
  _BYTE *v357; // [rsp+10h] [rbp-450h]
  __int64 v358; // [rsp+10h] [rbp-450h]
  size_t v359; // [rsp+18h] [rbp-448h]
  char v360; // [rsp+18h] [rbp-448h]
  void *v361; // [rsp+18h] [rbp-448h]
  __int64 *v362; // [rsp+20h] [rbp-440h]
  __int64 v363; // [rsp+20h] [rbp-440h]
  unsigned __int64 v364; // [rsp+20h] [rbp-440h]
  char v365; // [rsp+20h] [rbp-440h]
  unsigned __int64 v366; // [rsp+20h] [rbp-440h]
  __int64 v367; // [rsp+48h] [rbp-418h]
  __int64 v368; // [rsp+50h] [rbp-410h]
  __int64 v369; // [rsp+58h] [rbp-408h]
  __int128 v370; // [rsp+58h] [rbp-408h]
  __int64 v371; // [rsp+60h] [rbp-400h]
  __int64 v372; // [rsp+68h] [rbp-3F8h]
  unsigned __int64 v373; // [rsp+68h] [rbp-3F8h]
  __int64 v374; // [rsp+70h] [rbp-3F0h]
  __int64 v375; // [rsp+70h] [rbp-3F0h]
  __int64 v376; // [rsp+70h] [rbp-3F0h]
  __int64 *v377; // [rsp+80h] [rbp-3E0h]
  int v378; // [rsp+80h] [rbp-3E0h]
  _QWORD *v379; // [rsp+88h] [rbp-3D8h]
  __int64 v380; // [rsp+88h] [rbp-3D8h]
  __int64 v381; // [rsp+90h] [rbp-3D0h]
  unsigned int v382; // [rsp+90h] [rbp-3D0h]
  __int64 *v383; // [rsp+90h] [rbp-3D0h]
  _QWORD *v384; // [rsp+98h] [rbp-3C8h]
  unsigned __int64 v385; // [rsp+98h] [rbp-3C8h]
  __int64 *v386; // [rsp+98h] [rbp-3C8h]
  unsigned __int8 *v387; // [rsp+98h] [rbp-3C8h]
  int v388; // [rsp+A0h] [rbp-3C0h]
  char v389; // [rsp+A0h] [rbp-3C0h]
  _BYTE *v390; // [rsp+A0h] [rbp-3C0h]
  __int64 v391; // [rsp+A0h] [rbp-3C0h]
  __int64 *v392; // [rsp+A8h] [rbp-3B8h]
  __int64 *v393; // [rsp+A8h] [rbp-3B8h]
  __int64 v394; // [rsp+A8h] [rbp-3B8h]
  int v395; // [rsp+A8h] [rbp-3B8h]
  char *v396; // [rsp+B0h] [rbp-3B0h]
  void *v397; // [rsp+B0h] [rbp-3B0h]
  int v398; // [rsp+B0h] [rbp-3B0h]
  __int64 *v399; // [rsp+B8h] [rbp-3A8h]
  unsigned __int64 v400; // [rsp+B8h] [rbp-3A8h]
  _QWORD *v401; // [rsp+B8h] [rbp-3A8h]
  unsigned __int64 v402; // [rsp+B8h] [rbp-3A8h]
  __int64 v403; // [rsp+B8h] [rbp-3A8h]
  __int64 *v404; // [rsp+B8h] [rbp-3A8h]
  const __m128i *v405; // [rsp+B8h] [rbp-3A8h]
  __int64 v406; // [rsp+B8h] [rbp-3A8h]
  __int64 *v407; // [rsp+B8h] [rbp-3A8h]
  __int64 *v408; // [rsp+C0h] [rbp-3A0h]
  __int64 **v409; // [rsp+C0h] [rbp-3A0h]
  char v410; // [rsp+C0h] [rbp-3A0h]
  char v411; // [rsp+C0h] [rbp-3A0h]
  unsigned __int8 v412; // [rsp+C0h] [rbp-3A0h]
  __int64 *v413; // [rsp+C0h] [rbp-3A0h]
  __m128i v414; // [rsp+D0h] [rbp-390h] BYREF
  __int64 v415[4]; // [rsp+E0h] [rbp-380h] BYREF
  const __m128i *v416; // [rsp+100h] [rbp-360h] BYREF
  __m128i *v417; // [rsp+108h] [rbp-358h]
  const __m128i *v418; // [rsp+110h] [rbp-350h]
  __int64 v419; // [rsp+120h] [rbp-340h] BYREF
  __int64 *v420; // [rsp+128h] [rbp-338h]
  __int64 v421; // [rsp+130h] [rbp-330h]
  unsigned int v422; // [rsp+138h] [rbp-328h]
  void *src; // [rsp+140h] [rbp-320h] BYREF
  __int64 v424; // [rsp+148h] [rbp-318h]
  _BYTE v425[16]; // [rsp+150h] [rbp-310h] BYREF
  __int64 v426; // [rsp+160h] [rbp-300h] BYREF
  __int64 v427; // [rsp+168h] [rbp-2F8h]
  __int64 v428; // [rsp+170h] [rbp-2F0h]
  unsigned int v429; // [rsp+178h] [rbp-2E8h]
  char *v430; // [rsp+180h] [rbp-2E0h] BYREF
  size_t v431; // [rsp+188h] [rbp-2D8h]
  __m128i v432[2]; // [rsp+1A0h] [rbp-2C0h] BYREF
  char v433; // [rsp+1C0h] [rbp-2A0h]
  char v434; // [rsp+1C1h] [rbp-29Fh]
  __m128i v435; // [rsp+1D0h] [rbp-290h] BYREF
  __int16 v436; // [rsp+1F0h] [rbp-270h]
  __m128i v437[3]; // [rsp+200h] [rbp-260h] BYREF
  __m128i v438; // [rsp+230h] [rbp-230h] BYREF
  __m128i v439; // [rsp+240h] [rbp-220h]
  char v440; // [rsp+250h] [rbp-210h]
  char v441; // [rsp+251h] [rbp-20Fh]
  __m128i v442; // [rsp+260h] [rbp-200h] BYREF
  __int64 (__fastcall *v443)(__int64 *); // [rsp+270h] [rbp-1F0h]
  __int64 v444; // [rsp+278h] [rbp-1E8h]
  char v445; // [rsp+280h] [rbp-1E0h] BYREF
  __m128i v446; // [rsp+290h] [rbp-1D0h] BYREF
  __int64 (__fastcall *v447)(_QWORD *); // [rsp+2A0h] [rbp-1C0h]
  __int64 v448; // [rsp+2A8h] [rbp-1B8h]
  __int16 v449; // [rsp+2B0h] [rbp-1B0h] BYREF
  _QWORD **v450; // [rsp+2C0h] [rbp-1A0h] BYREF
  _QWORD **v451; // [rsp+2C8h] [rbp-198h] BYREF
  _QWORD *v452; // [rsp+2D0h] [rbp-190h]
  _QWORD *v453; // [rsp+2D8h] [rbp-188h]
  _QWORD *v454; // [rsp+2E0h] [rbp-180h]
  __int64 v455; // [rsp+2E8h] [rbp-178h]
  __int64 *v456; // [rsp+2F0h] [rbp-170h] BYREF
  __int64 v457; // [rsp+2F8h] [rbp-168h]
  __int64 v458; // [rsp+300h] [rbp-160h] BYREF
  unsigned int v459; // [rsp+308h] [rbp-158h]
  _BYTE *v460; // [rsp+310h] [rbp-150h]
  __int64 v461; // [rsp+318h] [rbp-148h]
  _BYTE v462[32]; // [rsp+320h] [rbp-140h] BYREF
  unsigned __int128 v463; // [rsp+340h] [rbp-120h] BYREF
  __m128i v464; // [rsp+350h] [rbp-110h] BYREF
  unsigned __int64 v465; // [rsp+360h] [rbp-100h]
  unsigned __int64 v466; // [rsp+368h] [rbp-F8h]
  __int128 v467; // [rsp+370h] [rbp-F0h]
  unsigned __int64 v468; // [rsp+390h] [rbp-D0h] BYREF
  unsigned __int64 v469; // [rsp+398h] [rbp-C8h]
  char *v470; // [rsp+3A0h] [rbp-C0h]
  __int64 v471; // [rsp+3A8h] [rbp-B8h]
  char v472; // [rsp+3B0h] [rbp-B0h] BYREF
  _QWORD *v473; // [rsp+3D0h] [rbp-90h]
  __int64 v474; // [rsp+3D8h] [rbp-88h]
  _QWORD v475[16]; // [rsp+3E0h] [rbp-80h] BYREF

  v1 = (__int64 *)a1;
  v2 = sub_B6AC80(*(_QWORD *)a1, 358);
  v381 = v2;
  if ( *(_DWORD *)(a1 + 24) )
  {
    v3 = *(__int64 ***)a1;
    if ( v2 )
    {
      sub_261A0B0(v3, v2);
      v4 = sub_B6AC80(*v1, 300);
      if ( !v4 )
        goto LABEL_5;
      goto LABEL_4;
    }
    v4 = sub_B6AC80((__int64)v3, 300);
    if ( v4 )
    {
LABEL_4:
      sub_261A0B0((__int64 **)*v1, v4);
LABEL_5:
      v5 = *(_QWORD *)(*v1 + 16);
      for ( i = *v1 + 8; v5 != i; v5 = *(_QWORD *)(v5 + 8) )
      {
        v7 = v5 - 56;
        if ( !v5 )
          v7 = 0;
        sub_B98000(v7, 28);
      }
      return 1;
    }
    return 0;
  }
  v9 = *(_QWORD *)(a1 + 8);
  if ( v9 && *(_BYTE *)(v9 + 346) )
    return 0;
  v10 = *(_QWORD *)(a1 + 16);
  if ( v10 )
  {
    if ( *(_BYTE *)(v10 + 346) )
      return 0;
  }
  v374 = sub_B6AC80(*(_QWORD *)a1, 194);
  if ( v381 )
  {
    v11 = *(_QWORD *)(v381 + 16);
    if ( v11 )
    {
      if ( *(_QWORD *)(a1 + 16) )
        goto LABEL_17;
      goto LABEL_58;
    }
  }
  if ( (!v374 || !*(_QWORD *)(v374 + 16)) && !*(_QWORD *)(a1 + 8) )
  {
    if ( !*(_QWORD *)(a1 + 16) )
      return 0;
LABEL_48:
    if ( !v381 || (v11 = *(_QWORD *)(v381 + 16)) == 0 )
    {
LABEL_18:
      if ( v374 && *(_QWORD *)(v374 + 16) )
        sub_C64ED0("unexpected call to llvm.icall.branch.funnel during import phase", 1u);
      v13 = *(_QWORD *)a1;
      v456 = &v458;
      v14 = v13 + 24;
      v457 = 0x800000000LL;
      *((_QWORD *)&v463 + 1) = 0x800000000LL;
      *(_QWORD *)&v463 = &v464;
      v15 = *(_QWORD *)(v13 + 32);
      if ( v15 == v13 + 24 )
      {
LABEL_30:
        v450 = 0;
        v451 = 0;
        v452 = 0;
        sub_262AB70((__int64 *)&v468, v13);
        v24 = v456;
        v25 = &v456[(unsigned int)v457];
        if ( v456 != v25 )
        {
          do
          {
            v26 = *v24++;
            sub_26330D0((__int64 **)a1, v26, 1u, (__int64)&v450);
          }
          while ( v25 != v24 );
        }
        v27 = (__int64 *)v463;
        v28 = v463 + 8LL * DWORD2(v463);
        if ( (_QWORD)v463 != v28 )
        {
          do
          {
            v29 = *v27++;
            sub_26330D0((__int64 **)a1, v29, 0, (__int64)&v450);
          }
          while ( (__int64 *)v28 != v27 );
        }
        sub_261AE10((__int64)&v468);
        v30 = (unsigned __int64)v450;
        v31 = v451;
        v32 = v450;
        if ( v451 != v450 )
        {
          do
          {
            v33 = *v32++;
            sub_B30340(v33);
          }
          while ( v31 != v32 );
          v30 = (unsigned __int64)v450;
        }
        if ( v30 )
          j_j___libc_free_0(v30);
        if ( (__m128i *)v463 != &v464 )
          _libc_free(v463);
        if ( v456 != &v458 )
        {
          _libc_free((unsigned __int64)v456);
          return 1;
        }
        return 1;
      }
      while ( 1 )
      {
        if ( !v15 )
          BUG();
        if ( (*(_BYTE *)(v15 - 24) & 0xFu) - 7 <= 1 )
          goto LABEL_22;
        v16 = v15 - 56;
        v17 = *(_QWORD *)(a1 + 16) + 352LL;
        v18 = (char *)sub_BD5D20(v15 - 56);
        if ( sub_2625AD0(v17, v18, v19) )
        {
          v22 = (unsigned int)v457;
          v23 = (unsigned int)v457 + 1LL;
          if ( v23 > HIDWORD(v457) )
          {
            sub_C8D5F0((__int64)&v456, &v458, v23, 8u, v20, v21);
            v22 = (unsigned int)v457;
          }
          v456[v22] = v16;
          LODWORD(v457) = v457 + 1;
          v15 = *(_QWORD *)(v15 + 8);
          if ( v15 == v14 )
          {
LABEL_29:
            v13 = *(_QWORD *)a1;
            goto LABEL_30;
          }
        }
        else
        {
          v34 = *(_QWORD *)(a1 + 16) + 384LL;
          v35 = (char *)sub_BD5D20(v15 - 56);
          if ( sub_2625AD0(v34, v35, v36) )
          {
            v39 = DWORD2(v463);
            v40 = DWORD2(v463) + 1LL;
            if ( v40 > HIDWORD(v463) )
            {
              sub_C8D5F0((__int64)&v463, &v464, v40, 8u, v37, v38);
              v39 = DWORD2(v463);
            }
            *(_QWORD *)(v463 + 8 * v39) = v16;
            ++DWORD2(v463);
          }
LABEL_22:
          v15 = *(_QWORD *)(v15 + 8);
          if ( v15 == v14 )
            goto LABEL_29;
        }
      }
    }
    do
    {
LABEL_17:
      v12 = v11;
      v11 = *(_QWORD *)(v11 + 8);
      sub_26278B0((__int64 *)a1, *(_QWORD *)(v12 + 24));
    }
    while ( v11 );
    goto LABEL_18;
  }
  if ( *(_QWORD *)(a1 + 16) )
    goto LABEL_48;
LABEL_58:
  v41 = *(_QWORD *)a1;
  v453 = &v451;
  v454 = &v451;
  v470 = &v472;
  v471 = 0x400000000LL;
  v473 = v475;
  src = v425;
  LODWORD(v451) = 0;
  v452 = 0;
  v455 = 0;
  v468 = 0;
  v469 = 0;
  v474 = 0;
  v475[0] = 0;
  v475[1] = 1;
  v419 = 0;
  v420 = 0;
  v421 = 0;
  v422 = 0;
  v424 = 0x200000000LL;
  v42 = sub_BA91D0(v41, "Cross-DSO CFI", 0xDu);
  v43 = v1[1] == 0;
  v456 = 0;
  v368 = v42;
  v457 = 0;
  v458 = 0;
  v459 = 0;
  v460 = v462;
  v461 = 0;
  if ( v43 )
    goto LABEL_123;
  v371 = sub_BA8DC0(*v1, (__int64)"cfi.functions", 13);
  if ( !v371 )
    goto LABEL_123;
  v446 = 0u;
  v44 = v1[1];
  v447 = 0;
  v448 = 0;
  v369 = v44 + 8;
  v372 = *(_QWORD *)(v44 + 24);
  if ( v372 == v44 + 8 )
    goto LABEL_96;
  v362 = v1;
  do
  {
    v377 = *(__int64 **)(v372 + 64);
    v399 = *(__int64 **)(v372 + 56);
    if ( v399 != v377 )
    {
LABEL_65:
      v45 = *v399;
      if ( *(char *)(*v399 + 12) >= 0 )
        goto LABEL_64;
      v46 = *(_QWORD **)(v45 + 40);
      v384 = &v46[*(unsigned int *)(v45 + 48)];
      if ( v384 == v46 )
        goto LABEL_64;
      while ( 1 )
      {
        v438.m128i_i64[0] = *(_QWORD *)(*v46 & 0xFFFFFFFFFFFFFFF8LL);
        if ( !(unsigned __int8)sub_A27FA0((__int64)&v446, v438.m128i_i64, &v442) )
          break;
LABEL_68:
        v47 = *v46 & 0xFFFFFFFFFFFFFFF8LL;
        v48 = *(_QWORD *)(v47 + 32);
        v49 = *(_QWORD *)(v47 + 24);
        if ( v48 != v49 )
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(*(_QWORD *)v49 + 8LL) )
              goto LABEL_71;
            v53 = v448;
            v54 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v49 + 56LL) & 0xFFFFFFFFFFFFFFF8LL);
            v442.m128i_i64[0] = v54;
            if ( !(_DWORD)v448 )
            {
              ++v446.m128i_i64[0];
              *(_QWORD *)&v463 = 0;
              goto LABEL_75;
            }
            v50 = (v448 - 1) & (((0xBF58476D1CE4E5B9LL * v54) >> 31) ^ (484763065 * v54));
            v51 = (__int64 *)(v446.m128i_i64[1] + 8LL * v50);
            v52 = *v51;
            if ( v54 == *v51 )
            {
LABEL_71:
              v49 += 8;
              if ( v48 == v49 )
                break;
            }
            else
            {
              v57 = 1;
              v55 = 0;
              while ( v52 != -1 )
              {
                if ( v55 || v52 != -2 )
                  v51 = v55;
                v50 = (v448 - 1) & (v57 + v50);
                v52 = *(_QWORD *)(v446.m128i_i64[1] + 8LL * v50);
                if ( v54 == v52 )
                  goto LABEL_71;
                ++v57;
                v55 = v51;
                v51 = (__int64 *)(v446.m128i_i64[1] + 8LL * v50);
              }
              if ( !v55 )
                v55 = v51;
              ++v446.m128i_i64[0];
              v56 = (_DWORD)v447 + 1;
              *(_QWORD *)&v463 = v55;
              if ( 4 * ((int)v447 + 1) < (unsigned int)(3 * v448) )
              {
                if ( (int)v448 - HIDWORD(v447) - v56 > (unsigned int)v448 >> 3 )
                  goto LABEL_77;
                goto LABEL_76;
              }
LABEL_75:
              v53 = 2 * v448;
LABEL_76:
              sub_A32210((__int64)&v446, v53);
              sub_A27FA0((__int64)&v446, v442.m128i_i64, &v463);
              v54 = v442.m128i_i64[0];
              v55 = (__int64 *)v463;
              v56 = (_DWORD)v447 + 1;
LABEL_77:
              LODWORD(v447) = v56;
              if ( *v55 != -1 )
                --HIDWORD(v447);
              v49 += 8;
              *v55 = v54;
              if ( v48 == v49 )
                break;
            }
          }
        }
        if ( v384 == ++v46 )
        {
LABEL_64:
          if ( v377 == ++v399 )
            goto LABEL_94;
          goto LABEL_65;
        }
      }
      v58 = v448;
      v59 = (_QWORD *)v442.m128i_i64[0];
      ++v446.m128i_i64[0];
      v60 = (_DWORD)v447 + 1;
      *(_QWORD *)&v463 = v442.m128i_i64[0];
      if ( 4 * ((int)v447 + 1) >= (unsigned int)(3 * v448) )
      {
        v58 = 2 * v448;
      }
      else if ( (int)v448 - HIDWORD(v447) - v60 > (unsigned int)v448 >> 3 )
      {
LABEL_91:
        LODWORD(v447) = v60;
        if ( *v59 != -1 )
          --HIDWORD(v447);
        *v59 = v438.m128i_i64[0];
        goto LABEL_68;
      }
      sub_A32210((__int64)&v446, v58);
      sub_A27FA0((__int64)&v446, v438.m128i_i64, &v463);
      v59 = (_QWORD *)v463;
      v60 = (_DWORD)v447 + 1;
      goto LABEL_91;
    }
LABEL_94:
    v372 = sub_220EEE0(v372);
  }
  while ( v369 != v372 );
  v1 = v362;
LABEL_96:
  v61 = 0;
  v388 = sub_B91A00(v371);
  if ( !v388 )
    goto LABEL_339;
  v392 = v1;
  do
  {
    v66 = sub_B91A10(v371, v61);
    v67 = (_BYTE *)(v66 - 16);
    v68 = v66;
    v69 = (__int64 *)sub_A17150((_BYTE *)(v66 - 16));
    v70 = (_BYTE *)sub_B91420(*v69);
    v72 = v71;
    v73 = sub_A17150(v67);
    v75 = sub_AD8340(*(unsigned __int8 **)(*((_QWORD *)v73 + 1) + 136LL), v61, v74);
    if ( *((_DWORD *)v75 + 2) > 0x40u )
      v75 = *(unsigned __int8 **)v75;
    v76 = *(unsigned __int8 **)v75;
    v77 = (__int64)v70;
    v78 = v72;
    if ( v72 && *v70 == 1 )
    {
      v78 = v72 - 1;
      v77 = (__int64)(v70 + 1);
    }
    v400 = sub_B2F650(v77, v78);
    v79 = sub_BAEF70(v392[1], v400);
    if ( v79 )
    {
      if ( (_DWORD)v448 )
      {
        v62 = (v448 - 1) & (((0xBF58476D1CE4E5B9LL * v400) >> 31) ^ (484763065 * v400));
        v63 = *(_QWORD *)(v446.m128i_i64[1] + 8LL * v62);
        if ( v400 == v63 )
          goto LABEL_99;
        v245 = 1;
        while ( v63 != -1 )
        {
          v62 = (v448 - 1) & (v245 + v62);
          v63 = *(_QWORD *)(v446.m128i_i64[1] + 8LL * v62);
          if ( v400 == v63 )
            goto LABEL_99;
          ++v245;
        }
      }
      v80 = (_DWORD)v76 != 0 || v368 == 0;
      if ( !v80 )
      {
        v81 = v392[1];
        v82 = *(_QWORD **)(v81 + 16);
        if ( v82 )
        {
          v83 = (_QWORD *)(v81 + 8);
          do
          {
            if ( v400 > v82[4] )
            {
              v82 = (_QWORD *)v82[3];
            }
            else
            {
              v83 = v82;
              v82 = (_QWORD *)v82[2];
            }
          }
          while ( v82 );
          if ( (_QWORD *)(v81 + 8) != v83 && v400 >= v83[4] )
          {
            v239 = *(unsigned __int8 *)(v81 + 343);
            v240 = v239 & 0xFFFFFFFFFFFFFFF8LL | (unsigned __int64)(v83 + 4) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v240 )
            {
              v241 = *(_QWORD *)((v239 & 0xFFFFFFFFFFFFFFF8LL | (unsigned __int64)(v83 + 4) & 0xFFFFFFFFFFFFFFF8LL)
                               + 0x18);
              v242 = *(_QWORD *)(v240 + 32);
              if ( v241 != v242 )
              {
                v243 = *(_QWORD *)(v240 + 24);
                do
                {
                  if ( *(char *)(*(_QWORD *)v243 + 12LL) < 0 && (*(_BYTE *)(*(_QWORD *)v243 + 12LL) & 0xFu) - 7 >= 2 )
                    v80 = v79;
                  v243 += 8;
                }
                while ( v242 != v243 );
                if ( v80 )
                {
LABEL_99:
                  *(_QWORD *)&v463 = v70;
                  *((_QWORD *)&v463 + 1) = v72;
                  v464.m128i_i32[0] = (int)v76;
                  v464.m128i_i64[1] = v68;
                  v64 = sub_26336E0((__int64)&v456, (const __m128i *)&v463, &v464);
                  if ( !v65 && *(_DWORD *)(v64 + 16) )
                  {
                    *(_DWORD *)(v64 + 16) = (_DWORD)v76;
                    *(_QWORD *)(v64 + 24) = v68;
                  }
                }
              }
            }
          }
        }
      }
    }
    ++v61;
  }
  while ( v388 != v61 );
  v1 = v392;
LABEL_339:
  v217 = (unsigned __int64)v460;
  v390 = &v460[32 * (unsigned int)v461];
  if ( v390 == v460 )
    goto LABEL_368;
  v404 = v1;
  while ( 2 )
  {
    v218 = *(_QWORD *)(v217 + 8);
    v219 = *(_DWORD *)(v217 + 16);
    v220 = *(_QWORD *)(v217 + 24);
    v394 = *(_QWORD *)v217;
    v221 = sub_BA8CB0(*v404, *(_QWORD *)v217, v218);
    v222 = v221;
    if ( v221 )
    {
      v223 = v221[32] & 0xF;
      if ( (unsigned int)v223 - 7 <= 1 )
      {
        v224.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v221);
        v463 = (unsigned __int128)v224;
        LOWORD(v465) = 773;
        v464.m128i_i64[0] = (__int64)".1";
        sub_BD6B50(v222, (const char **)&v463);
        goto LABEL_347;
      }
    }
    else
    {
LABEL_347:
      *((_QWORD *)&v463 + 1) = v218;
      v225 = *v404;
      LOWORD(v465) = 261;
      *(_QWORD *)&v463 = v394;
      v226 = *(_DWORD *)(v225 + 320);
      v227 = (__int64 *)sub_BCB120(*(_QWORD **)v225);
      v228 = sub_BCF640(v227, 0);
      v222 = (unsigned __int8 *)sub_2624F00(v228, 0, v226, (__int64)&v463, v225);
      v223 = v222[32] & 0xF;
    }
    if ( v223 == 1 )
    {
      v222[32] &= 0xF0u;
      if ( (unsigned __int8)sub_2624ED0((__int64)v222) )
        v222[33] |= 0x40u;
      sub_B2CA40((__int64)v222, 0);
      v222[32] &= 0xF0u;
      if ( (unsigned __int8)sub_2624ED0((__int64)v222) )
        v222[33] |= 0x40u;
      sub_B2F990((__int64)v222, 0, v232, v233);
      sub_B91E30((__int64)v222, 0);
    }
    if ( v219 )
    {
      if ( sub_B2FC80((__int64)v222) )
      {
        if ( v219 == 2 )
        {
          v222[32] = v222[32] & 0xF0 | 9;
          if ( (unsigned __int8)sub_2624ED0((__int64)v222) )
            v222[33] |= 0x40u;
        }
        goto LABEL_354;
      }
    }
    else
    {
      if ( (v222[32] & 0xF) == 9 )
      {
        v222[32] &= 0xF0u;
        if ( (unsigned __int8)sub_2624ED0((__int64)v222) )
          v222[33] |= 0x40u;
      }
      if ( sub_B2FC80((__int64)v222) )
      {
LABEL_354:
        v229 = 2;
        sub_B98000((__int64)v222, 19);
        while ( 1 )
        {
          v230 = (*(_BYTE *)(v220 - 16) & 2) != 0 ? *(_DWORD *)(v220 - 24) : (*(_WORD *)(v220 - 16) >> 6) & 0xF;
          if ( v229 >= v230 )
            break;
          v231 = *(_QWORD *)&sub_A17150((_BYTE *)(v220 - 16))[8 * v229++];
          sub_B994D0((__int64)v222, 19, v231);
        }
      }
    }
    v217 += 32LL;
    if ( v390 != (_BYTE *)v217 )
      continue;
    break;
  }
  v1 = v404;
LABEL_368:
  sub_C7D6A0(v446.m128i_i64[1], 8LL * (unsigned int)v448, 8);
LABEL_123:
  v84 = *v1;
  v426 = 0;
  v427 = 0;
  v428 = 0;
  v429 = 0;
  sub_BA9600(&v463, v84);
  v378 = 0;
  v85 = _mm_loadu_si128(&v464);
  v367 = (__int64)v1;
  v373 = v465;
  v438 = _mm_loadu_si128((const __m128i *)&v463);
  v385 = v466;
  v439 = v85;
  v370 = v467;
  while ( 2 )
  {
    if ( *(_OWORD *)&v438 != __PAIR128__(v385, v373) || *(_OWORD *)&v439 != v370 )
    {
      v86 = &v442;
      v87 = &v438;
      v444 = 0;
      v88 = &v438;
      v443 = sub_25AC5E0;
      v89 = sub_25AC5C0;
      v90 = &v442;
      if ( ((unsigned __int8)sub_25AC5C0 & 1) == 0 )
        goto LABEL_127;
LABEL_126:
      v89 = *(__int64 (__fastcall **)(__int64))((char *)v89 + v88->m128i_i64[0] - 1);
LABEL_127:
      while ( 1 )
      {
        v91 = (_BYTE *)v89((__int64)v88);
        v92 = v91;
        if ( v91 )
          break;
        if ( &v445 == (char *)++v86 )
          goto LABEL_586;
        v93 = v90[1].m128i_i64[1];
        v89 = (__int64 (__fastcall *)(__int64))v90[1].m128i_i64[0];
        v90 = v86;
        v88 = (__m128i *)((char *)&v438 + v93);
        if ( ((unsigned __int8)v89 & 1) != 0 )
          goto LABEL_126;
      }
      if ( *v91 == 3 && ((v91[32] & 0xF) == 1 || sub_B2FC80((__int64)v91)) )
        goto LABEL_165;
      LODWORD(v424) = 0;
      sub_B91D10((__int64)v92, 19, (__int64)&src);
      if ( *v92 )
      {
        v112 = 0;
        v94 = 0;
      }
      else
      {
        v94 = sub_2626710((__int64)v92);
        v95 = (char *)sub_BD5D20((__int64)v92);
        v97 = v459;
        v363 = v457;
        if ( v459 )
        {
          v356 = v95;
          v359 = v96;
          v98 = sub_C94890(v95, v96);
          v99 = v97 - 1;
          v100 = v94;
          v101 = v359;
          v102 = (v97 - 1) & v98;
          v103 = v92;
          v104 = v356;
          v105 = &v438;
          for ( j = 1; ; ++j )
          {
            v107 = v363 + 24LL * v102;
            v108 = *(const void **)v107;
            if ( *(_QWORD *)v107 == -1 )
              break;
            v109 = v104 + 2 == 0;
            if ( v108 != (const void *)-2LL )
            {
              v354 = v105;
              v355 = v99;
              v357 = v103;
              v360 = v100;
              v109 = sub_9691B0(v104, v101, v108, *(_QWORD *)(v107 + 8));
              v100 = v360;
              v103 = v357;
              v99 = v355;
              v105 = v354;
            }
            if ( v109 )
            {
              v94 = v100;
              v92 = v103;
              v110 = v363 + 24LL * v102;
              v87 = v105;
              goto LABEL_142;
            }
            v353 = j + v102;
            v102 = v99 & v353;
          }
          v216 = v104;
          v92 = v103;
          v94 = v100;
          v87 = v105;
          v110 = v363 + 24LL * v102;
          if ( v216 != (char *)-1LL )
            goto LABEL_333;
LABEL_142:
          if ( v110 != v457 + 24LL * v459 )
          {
            v111 = &v460[32 * *(unsigned int *)(v110 + 16)];
            if ( v111 != &v460[32 * (unsigned int)v461] )
            {
              v112 = 1;
              v94 |= *((_DWORD *)v111 + 4) == 0;
              goto LABEL_145;
            }
          }
        }
LABEL_333:
        if ( (unsigned __int8)sub_B2DDD0((__int64)v92, 0, 0, 1, 0, 0, 0) )
        {
          v112 = 0;
        }
        else
        {
          v112 = v94 ^ 1 | (v368 == 0);
          if ( v112 || (v94 = 1, (v92[32] & 0xFu) - 7 <= 1) )
          {
LABEL_165:
            v131 = &v446;
            v448 = 0;
            v132 = &v446;
            v133 = (__int64 *)v87;
            v447 = sub_25AC590;
            v134 = sub_25AC560;
            if ( ((unsigned __int8)sub_25AC560 & 1) == 0 )
              goto LABEL_167;
LABEL_166:
            v134 = *(__int64 (__fastcall **)(__int64))((char *)v134 + *v133 - 1);
LABEL_167:
            while ( !(unsigned __int8)v134((__int64)v133) )
            {
              if ( &v449 == (__int16 *)++v131 )
                goto LABEL_586;
              v135 = v132[1].m128i_i64[1];
              v134 = (__int64 (__fastcall *)(__int64))v132[1].m128i_i64[0];
              v132 = v131;
              v133 = (__int64 *)((char *)v87->m128i_i64 + v135);
              if ( ((unsigned __int8)v134 & 1) != 0 )
                goto LABEL_166;
            }
            continue;
          }
        }
      }
LABEL_145:
      v113 = (unsigned int)v424;
      v114 = src;
      v115 = 8LL * (unsigned int)v424;
      v116 = (v468 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v475[0] += v115 + 24;
      if ( v469 >= v115 + 24 + v116 && v468 )
      {
        v468 = v115 + 24 + v116;
      }
      else
      {
        v358 = (unsigned int)v424;
        v361 = src;
        v365 = v112;
        v244 = sub_9D1E70((__int64)&v468, v115 + 24, v115 + 24, 3);
        v113 = v358;
        v114 = v361;
        v112 = v365;
        v116 = v244;
      }
      *(_QWORD *)v116 = v92;
      *(_QWORD *)(v116 + 8) = v113;
      *(_BYTE *)(v116 + 16) = v94;
      *(_BYTE *)(v116 + 17) = v112;
      if ( v115 )
      {
        v364 = v116;
        memmove((void *)(v116 + 24), v114, v115);
        v116 = v364;
      }
      v117 = v429;
      v437[0].m128i_i64[0] = v116;
      v442.m128i_i64[0] = (__int64)v92;
      if ( v429 )
      {
        v118 = 1;
        v119 = 0;
        v120 = (v429 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
        v121 = (__int64 *)(v427 + 16LL * v120);
        v122 = (_BYTE *)*v121;
        if ( (_BYTE *)*v121 == v92 )
        {
LABEL_152:
          v121[1] = v116;
          v123 = (char *)src;
          v124 = (char *)src + 8 * (unsigned int)v424;
          if ( v124 != src )
          {
            do
            {
              v125 = *(_QWORD *)v123;
              if ( (*(_BYTE *)(*(_QWORD *)v123 - 16LL) & 2) != 0 )
                v126 = *(_DWORD *)(v125 - 24);
              else
                v126 = (*(_WORD *)(v125 - 16) >> 6) & 0xF;
              if ( v126 != 2 )
                sub_C64ED0("All operands of type metadata must have 2 elements", 1u);
              if ( (v92[33] & 0x1C) != 0 )
                sub_C64ED0("Bit set element may not be thread-local", 1u);
              if ( *v92 == 3 && (v92[35] & 4) != 0 )
                sub_C64ED0("A member of a type identifier may not have an explicit section", 1u);
              v127 = (_BYTE *)(v125 - 16);
              v128 = *(_QWORD *)sub_A17150((_BYTE *)(v125 - 16));
              if ( *(_BYTE *)v128 != 1 )
                sub_C64ED0("Type offset must be a constant", 1u);
              if ( **(_BYTE **)(v128 + 136) != 17 )
                sub_C64ED0("Type offset must be an integer constant", 1u);
              v446.m128i_i64[0] = *((_QWORD *)sub_A17150(v127) + 1);
              v129 = sub_26238D0((__int64)&v419, v446.m128i_i64);
              ++v378;
              v130 = (_BYTE *)v129[2];
              *(_DWORD *)v129 = v378;
              if ( v130 == (_BYTE *)v129[3] )
              {
                sub_26194D0((__int64)(v129 + 1), v130, v437);
              }
              else
              {
                if ( v130 )
                {
                  *(_QWORD *)v130 = v437[0].m128i_i64[0];
                  v130 = (_BYTE *)v129[2];
                }
                v129[2] = (__int64)(v130 + 8);
              }
              v123 += 8;
            }
            while ( v124 != v123 );
          }
          goto LABEL_165;
        }
        while ( v122 != (_BYTE *)-4096LL )
        {
          if ( !v119 && v122 == (_BYTE *)-8192LL )
            v119 = v121;
          v120 = (v429 - 1) & (v118 + v120);
          v121 = (__int64 *)(v427 + 16LL * v120);
          v122 = (_BYTE *)*v121;
          if ( v92 == (_BYTE *)*v121 )
            goto LABEL_152;
          ++v118;
        }
        if ( v119 )
          v121 = v119;
        ++v426;
        v335 = v428 + 1;
        v446.m128i_i64[0] = (__int64)v121;
        if ( 4 * ((int)v428 + 1) < 3 * v429 )
        {
          v336 = (__int64)v92;
          if ( v429 - HIDWORD(v428) - v335 > v429 >> 3 )
          {
LABEL_535:
            LODWORD(v428) = v335;
            if ( *v121 != -4096 )
              --HIDWORD(v428);
            *v121 = v336;
            v121[1] = 0;
            goto LABEL_152;
          }
          v366 = v116;
LABEL_540:
          sub_261CB80((__int64)&v426, v117);
          sub_2618E40((__int64)&v426, v442.m128i_i64, &v446);
          v336 = v442.m128i_i64[0];
          v116 = v366;
          v335 = v428 + 1;
          v121 = (__int64 *)v446.m128i_i64[0];
          goto LABEL_535;
        }
      }
      else
      {
        ++v426;
        v446.m128i_i64[0] = 0;
      }
      v366 = v116;
      v117 = 2 * v429;
      goto LABEL_540;
    }
    break;
  }
  v136 = (__int64 *)v367;
  v415[1] = (__int64)&v450;
  v415[0] = v367;
  v415[2] = (__int64)&v419;
  if ( v381 )
  {
    for ( k = *(_QWORD *)(v381 + 16); k; k = *(_QWORD *)(k + 8) )
    {
      v138 = *(_QWORD *)(k + 24);
      *(_QWORD *)&v463 = v138;
      v139 = *(_QWORD *)(v138 + 16);
      if ( v139 )
      {
        while ( 1 )
        {
          v140 = *(_QWORD *)(v139 + 24);
          if ( *(_BYTE *)v140 != 85 )
            break;
          v246 = *(_QWORD *)(v140 - 32);
          if ( !v246
            || *(_BYTE *)v246
            || *(_QWORD *)(v246 + 24) != *(_QWORD *)(v140 + 80)
            || (*(_BYTE *)(v246 + 33) & 0x20) == 0
            || *(_DWORD *)(v246 + 36) != 11 )
          {
            break;
          }
          v139 = *(_QWORD *)(v139 + 8);
          if ( !v139 )
            goto LABEL_182;
        }
      }
      v141 = *(_QWORD *)(v138 + 32 * (1LL - (*(_DWORD *)(v138 + 4) & 0x7FFFFFF)));
      if ( *(_BYTE *)v141 != 24 )
        sub_C64ED0("Second argument of llvm.type.test must be metadata", 1u);
      v142 = sub_2622110(v415, *(_QWORD **)(v141 + 24));
      v143 = *(_BYTE **)(v142 + 8);
      if ( v143 == *(_BYTE **)(v142 + 16) )
      {
        sub_2628C60(v142, v143, &v463);
      }
      else
      {
        if ( v143 )
        {
          *(_QWORD *)v143 = v463;
          v143 = *(_BYTE **)(v142 + 8);
        }
        *(_QWORD *)(v142 + 8) = v143 + 8;
      }
LABEL_182:
      ;
    }
  }
  if ( v374 )
  {
    v375 = *(_QWORD *)(v374 + 16);
    if ( v375 )
    {
      while ( 2 )
      {
        if ( *(_DWORD *)(v367 + 28) != 39 )
          sub_C64ED0("llvm.icall.branch.funnel not supported on this target", 1u);
        v144 = *(unsigned __int8 **)(v375 + 24);
        v446 = 0u;
        v447 = 0;
        v145 = *v144;
        if ( v145 == 40 )
        {
          v146 = 32LL * (unsigned int)sub_B491D0((__int64)v144);
        }
        else
        {
          v146 = 0;
          if ( v145 != 85 )
          {
            if ( v145 != 34 )
              goto LABEL_586;
            v146 = 64;
          }
        }
        if ( (v144[7] & 0x80u) != 0 )
        {
          v147 = sub_BD2BC0((__int64)v144);
          v149 = v147 + v148;
          if ( (v144[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v149 >> 4) )
              goto LABEL_587;
          }
          else if ( (unsigned int)((v149 - sub_BD2BC0((__int64)v144)) >> 4) )
          {
            if ( (v144[7] & 0x80u) == 0 )
              goto LABEL_587;
            v150 = *(_DWORD *)(sub_BD2BC0((__int64)v144) + 8);
            if ( (v144[7] & 0x80u) == 0 )
              goto LABEL_586;
            v151 = sub_BD2BC0((__int64)v144);
            v153 = 32LL * (unsigned int)(*(_DWORD *)(v151 + v152 - 4) - v150);
            goto LABEL_196;
          }
        }
        v153 = 0;
LABEL_196:
        v382 = ((32LL * (*((_DWORD *)v144 + 1) & 0x7FFFFFF) - 32 - v146 - v153) >> 5) & 1;
        if ( !v382 )
          sub_C64ED0("number of arguments should be odd", 1u);
        v154 = *v144;
        v155 = v382;
        v156 = v154 - 29;
        if ( v154 != 40 )
        {
LABEL_198:
          v157 = 0;
          if ( v156 == 56 )
            goto LABEL_201;
          if ( v156 == 5 )
          {
            v157 = 64;
LABEL_201:
            if ( (v144[7] & 0x80u) == 0 )
              goto LABEL_246;
LABEL_202:
            v158 = sub_BD2BC0((__int64)v144);
            v160 = v158 + v159;
            if ( (v144[7] & 0x80u) == 0 )
            {
              if ( !(unsigned int)(v160 >> 4) )
                goto LABEL_246;
            }
            else
            {
              if ( !(unsigned int)((v160 - sub_BD2BC0((__int64)v144)) >> 4) )
                goto LABEL_246;
              if ( (v144[7] & 0x80u) != 0 )
              {
                v161 = *(_DWORD *)(sub_BD2BC0((__int64)v144) + 8);
                if ( (v144[7] & 0x80u) != 0 )
                {
                  v162 = sub_BD2BC0((__int64)v144);
                  v164 = 32LL * (unsigned int)(*(_DWORD *)(v162 + v163 - 4) - v161);
                  goto LABEL_207;
                }
                goto LABEL_586;
              }
            }
LABEL_587:
            BUG();
          }
LABEL_586:
          BUG();
        }
LABEL_245:
        v157 = 32LL * (unsigned int)sub_B491D0((__int64)v144);
        if ( (v144[7] & 0x80u) != 0 )
          goto LABEL_202;
LABEL_246:
        v164 = 0;
LABEL_207:
        if ( v155 == (unsigned int)((32LL * (*((_DWORD *)v144 + 1) & 0x7FFFFFF) - 32 - v157 - v164) >> 5) )
        {
          v193 = (const void *)v446.m128i_i64[0];
          ++v378;
          v194 = v446.m128i_i64[1] - v446.m128i_i64[0];
          v195 = v446.m128i_i64[1] - v446.m128i_i64[0] + 24;
          v475[0] += v195;
          v196 = (v468 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          v197 = (v446.m128i_i64[1] - v446.m128i_i64[0]) >> 3;
          if ( v469 >= v195 + v196 && v468 )
          {
            v468 = v195 + v196;
            v198 = v196;
          }
          else
          {
            v403 = (v446.m128i_i64[1] - v446.m128i_i64[0]) >> 3;
            v198 = sub_9D1E70((__int64)&v468, v195, v446.m128i_i64[1] - v446.m128i_i64[0] + 24, 3);
            v197 = v403;
            v196 = v198 & 0xFFFFFFFFFFFFFFFCLL;
          }
          *(_QWORD *)v198 = v144;
          *(_QWORD *)(v198 + 16) = v197;
          *(_DWORD *)(v198 + 8) = v378;
          if ( v194 )
            memmove((void *)(v198 + 24), v193, v194);
          v199 = sub_2619BA0(&v450, v196 | 2);
          v200 = 0;
          if ( v199 != &v451 )
            v200 = (unsigned __int64)sub_261C730((_BYTE *)v199 + 32);
          if ( (_QWORD *)v200 != v379 )
          {
            *(_QWORD *)(*v379 + 8LL) = v200 | *(_QWORD *)(*v379 + 8LL) & 1LL;
            *v379 = *(_QWORD *)v200;
            *(_QWORD *)(v200 + 8) &= ~1uLL;
            *(_QWORD *)v200 = v379;
          }
          if ( v446.m128i_i64[0] )
            j_j___libc_free_0(v446.m128i_u64[0]);
          v375 = *(_QWORD *)(v375 + 8);
          if ( !v375 )
          {
            v136 = (__int64 *)v367;
            goto LABEL_291;
          }
          continue;
        }
        break;
      }
      v165 = *(_QWORD *)v367 + 312LL;
      v166 = *(_QWORD *)&v144[32 * (v155 - (unsigned __int64)(*((_DWORD *)v144 + 1) & 0x7FFFFFF))];
      DWORD2(v463) = sub_AE43F0(v165, *(_QWORD *)(v166 + 8));
      if ( DWORD2(v463) > 0x40 )
        sub_C43690((__int64)&v463, 0, 0);
      else
        *(_QWORD *)&v463 = 0;
      v167 = sub_BD45C0((unsigned __int8 *)v166, v165, (__int64)&v463, 1, 0, 0, 0, 0);
      if ( DWORD2(v463) > 0x40 && (_QWORD)v463 )
        j_j___libc_free_0_0(v463);
      v168 = *v167 == 0 || (unsigned __int8)(*v167 - 2) <= 1u;
      if ( !v168 )
      {
        v438.m128i_i64[0] = 0;
        sub_C64ED0("Expected branch funnel operand to be global value", 1u);
      }
      v169 = v429;
      v438.m128i_i64[0] = (__int64)v167;
      if ( v429 )
      {
        v170 = 0;
        v171 = v382;
        v172 = (v429 - 1) & (((unsigned int)v167 >> 9) ^ ((unsigned int)v167 >> 4));
        v173 = (unsigned __int8 **)(v427 + 16LL * v172);
        v174 = *v173;
        if ( *v173 == v167 )
        {
LABEL_216:
          v175 = (__int64)v173[1];
          goto LABEL_217;
        }
        while ( v174 != (unsigned __int8 *)-4096LL )
        {
          if ( !v170 && v174 == (unsigned __int8 *)-8192LL )
            v170 = v173;
          v172 = (v429 - 1) & (v171 + v172);
          v173 = (unsigned __int8 **)(v427 + 16LL * v172);
          v174 = *v173;
          if ( v167 == *v173 )
            goto LABEL_216;
          ++v171;
        }
        if ( !v170 )
          v170 = v173;
        ++v426;
        v191 = v428 + 1;
        *(_QWORD *)&v463 = v170;
        if ( 4 * ((int)v428 + 1) < 3 * v429 )
        {
          if ( v429 - HIDWORD(v428) - v191 > v429 >> 3 )
          {
LABEL_257:
            LODWORD(v428) = v191;
            if ( *v170 != (unsigned __int8 *)-4096LL )
              --HIDWORD(v428);
            *v170 = v167;
            v170[1] = 0;
            v175 = 0;
LABEL_217:
            v442.m128i_i64[0] = v175;
            v176 = v446.m128i_i64[1];
            if ( (__int64 (__fastcall *)(_QWORD *))v446.m128i_i64[1] == v447 )
            {
              sub_26194D0((__int64)&v446, (_BYTE *)v446.m128i_i64[1], &v442);
              v175 = v442.m128i_i64[0];
            }
            else
            {
              if ( v446.m128i_i64[1] )
              {
                *(_QWORD *)v446.m128i_i64[1] = v175;
                v176 = v446.m128i_i64[1];
              }
              v446.m128i_i64[1] = v176 + 8;
            }
            v177 = v452;
            v178 = v175 & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)&v463 = &v463;
            *((_QWORD *)&v463 + 1) = 1;
            v464.m128i_i64[0] = v178;
            if ( v452 )
            {
              while ( 1 )
              {
                v179 = v177[6];
                v180 = (_QWORD *)v177[3];
                v181 = 0;
                if ( v179 > v178 )
                {
                  v180 = (_QWORD *)v177[2];
                  v181 = v168;
                }
                if ( !v180 )
                  break;
                v177 = v180;
              }
              if ( !v181 )
              {
                if ( v178 > v179 )
                {
LABEL_229:
                  v182 = 1;
                  if ( v177 != &v451 )
                    v182 = v178 < v177[6];
LABEL_231:
                  v389 = v182;
                  v183 = (_QWORD *)sub_22077B0(0x38u);
                  v184 = (unsigned __int64)(v183 + 4);
                  v183[5] = 1;
                  v183[4] = v183 + 4;
                  v401 = v183;
                  v183[6] = v464.m128i_i64[0];
                  sub_220F040(v389, (__int64)v183, v177, &v451);
                  ++v455;
                  v185 = v401;
LABEL_232:
                  if ( (v185[5] & 1) == 0 )
                  {
                    v184 = v185[4];
                    if ( (*(_BYTE *)(v184 + 8) & 1) == 0 )
                    {
                      v186 = *(_QWORD *)v184;
                      if ( (*(_BYTE *)(*(_QWORD *)v184 + 8LL) & 1) != 0 )
                      {
                        v184 = *(_QWORD *)v184;
                      }
                      else
                      {
                        v187 = *(_BYTE ***)v186;
                        if ( (*(_BYTE *)(*(_QWORD *)v186 + 8LL) & 1) == 0 )
                        {
                          if ( ((*v187)[8] & 1) != 0 )
                          {
                            v187 = (_BYTE **)*v187;
                          }
                          else
                          {
                            v188 = sub_261C730(*v187);
                            *v189 = v188;
                            v187 = (_BYTE **)v188;
                          }
                          *(_QWORD *)v186 = v187;
                        }
                        *(_QWORD *)v184 = v187;
                        v184 = (unsigned __int64)v187;
                      }
                      v185[4] = v184;
                    }
                  }
LABEL_241:
                  if ( v155 == 1 )
                  {
                    v379 = (_QWORD *)v184;
                  }
                  else if ( (_QWORD *)v184 != v379 )
                  {
                    *(_QWORD *)(*v379 + 8LL) = v184 | *(_QWORD *)(*v379 + 8LL) & 1LL;
                    *v379 = *(_QWORD *)v184;
                    *(_QWORD *)(v184 + 8) &= ~1uLL;
                    *(_QWORD *)v184 = v379;
                  }
                  v190 = *v144;
                  v155 += 2;
                  v156 = v190 - 29;
                  if ( v190 != 40 )
                    goto LABEL_198;
                  goto LABEL_245;
                }
LABEL_262:
                if ( v177 == &v451 )
                {
                  v184 = 0;
                  goto LABEL_241;
                }
                goto LABEL_267;
              }
              if ( v453 == v177 )
                goto LABEL_229;
            }
            else
            {
              v177 = &v451;
              if ( v453 == &v451 )
              {
                v177 = &v451;
                v182 = 1;
                goto LABEL_231;
              }
            }
            v402 = v178;
            v192 = sub_220EF80((__int64)v177);
            v178 = v402;
            if ( *(_QWORD *)(v192 + 48) >= v402 )
            {
              v177 = (_QWORD *)v192;
              goto LABEL_262;
            }
            if ( v177 )
              goto LABEL_229;
LABEL_267:
            v184 = (unsigned __int64)(v177 + 4);
            v185 = v177;
            goto LABEL_232;
          }
LABEL_277:
          sub_261CB80((__int64)&v426, v169);
          sub_2618E40((__int64)&v426, v438.m128i_i64, &v463);
          v167 = (unsigned __int8 *)v438.m128i_i64[0];
          v170 = (unsigned __int8 **)v463;
          v191 = v428 + 1;
          goto LABEL_257;
        }
      }
      else
      {
        ++v426;
        *(_QWORD *)&v463 = 0;
      }
      v169 = 2 * v429;
      goto LABEL_277;
    }
  }
LABEL_291:
  if ( !v136[1] )
    goto LABEL_409;
  v463 = 0u;
  v464.m128i_i64[0] = 0;
  v464.m128i_i32[2] = 0;
  if ( (_DWORD)v421 )
  {
    v337 = v420;
    v338 = &v420[5 * v422];
    if ( v420 != v338 )
    {
      while ( 1 )
      {
        v339 = *v337;
        v340 = v337;
        if ( *v337 != -4096 && v339 != -8192 )
          break;
        v337 += 5;
        if ( v338 == v337 )
          goto LABEL_293;
      }
      if ( v338 != v337 )
      {
        v413 = v136;
        v341 = &v420[5 * v422];
        while ( 1 )
        {
          if ( !*(_BYTE *)v339 )
          {
            v342 = sub_B91420(v339);
            v442.m128i_i64[0] = sub_B2F650(v342, v343);
            if ( (unsigned __int8)sub_2628DF0((__int64)&v463, v442.m128i_i64, &v446) )
            {
              v345 = (unsigned __int64 *)(v446.m128i_i64[0] + 8);
            }
            else
            {
              v348 = sub_262D4F0((__int64)&v463, &v442, v446.m128i_i64[0]);
              v349 = v442.m128i_i64[0];
              v345 = v348 + 1;
              *v345 = 0;
              *(v345 - 1) = v349;
            }
            v346 = *v345 & 0xFFFFFFFFFFFFFFFCLL;
            if ( v346 )
            {
              if ( (*v345 & 2) == 0 )
              {
                v407 = (__int64 *)v345;
                v350 = sub_22077B0(0x30u);
                v345 = (unsigned __int64 *)v407;
                if ( v350 )
                {
                  *(_QWORD *)v350 = v350 + 16;
                  *(_QWORD *)(v350 + 8) = 0x400000000LL;
                }
                v351 = v350 & 0xFFFFFFFFFFFFFFFCLL;
                *v407 = v350 | 2;
                v352 = *(unsigned int *)((v350 & 0xFFFFFFFFFFFFFFFCLL) + 8);
                if ( v352 + 1 > (unsigned __int64)*(unsigned int *)(v351 + 12) )
                {
                  sub_C8D5F0(v351, (const void *)(v351 + 16), v352 + 1, 8u, (__int64)v407, v344);
                  v352 = *(unsigned int *)(v351 + 8);
                  v345 = (unsigned __int64 *)v407;
                }
                *(_QWORD *)(*(_QWORD *)v351 + 8 * v352) = v346;
                ++*(_DWORD *)(v351 + 8);
                v346 = *v345 & 0xFFFFFFFFFFFFFFFCLL;
              }
              v347 = *(unsigned int *)(v346 + 8);
              if ( v347 + 1 > (unsigned __int64)*(unsigned int *)(v346 + 12) )
              {
                sub_C8D5F0(v346, (const void *)(v346 + 16), v347 + 1, 8u, (__int64)v345, v344);
                v347 = *(unsigned int *)(v346 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v346 + 8 * v347) = v339;
              ++*(_DWORD *)(v346 + 8);
            }
            else
            {
              *v345 = v339 & 0xFFFFFFFFFFFFFFFDLL;
            }
          }
          v340 += 5;
          if ( v340 == v341 )
            break;
          while ( *v340 == -8192 || *v340 == -4096 )
          {
            v340 += 5;
            if ( v341 == v340 )
              goto LABEL_568;
          }
          if ( v341 == v340 )
            break;
          v339 = *v340;
        }
LABEL_568:
        v136 = v413;
      }
    }
  }
LABEL_293:
  v201 = v136[1];
  v376 = v201 + 8;
  v380 = *(_QWORD *)(v201 + 24);
  if ( v201 + 8 == v380 )
    goto LABEL_372;
  v386 = v136;
  while ( 2 )
  {
    v383 = *(__int64 **)(v380 + 64);
    v408 = *(__int64 **)(v380 + 56);
    if ( v383 != v408 )
    {
      while ( 1 )
      {
        v202 = *v408;
        if ( !*(_BYTE *)(v386[1] + 336) || *(char *)(v202 + 12) < 0 )
        {
          v203 = *(_DWORD *)(v202 + 8);
          if ( !v203 )
          {
            v202 = *(_QWORD *)(v202 + 64);
            v203 = *(_DWORD *)(v202 + 8);
          }
          if ( v203 == 1 )
          {
            v204 = *(__int64 ***)(v202 + 80);
            if ( v204 )
            {
              v205 = *v204;
              v393 = v204[1];
              if ( *v204 != v393 )
                break;
            }
          }
        }
LABEL_298:
        if ( v383 == ++v408 )
          goto LABEL_370;
      }
      while ( 1 )
      {
        v438.m128i_i64[0] = *v205;
        v43 = (unsigned __int8)sub_2628DF0((__int64)&v463, v438.m128i_i64, &v442) == 0;
        v206 = (__int64 *)v442.m128i_i64[0];
        if ( !v43 )
          goto LABEL_311;
        v207 = v464.m128i_i32[2];
        v446.m128i_i64[0] = v442.m128i_i64[0];
        *(_QWORD *)&v463 = v463 + 1;
        v208 = v464.m128i_i32[0] + 1;
        if ( 4 * (v464.m128i_i32[0] + 1) >= (unsigned int)(3 * v464.m128i_i32[2]) )
          break;
        if ( v464.m128i_i32[2] - v464.m128i_i32[1] - v208 <= (unsigned __int32)v464.m128i_i32[2] >> 3 )
          goto LABEL_324;
LABEL_308:
        v464.m128i_i32[0] = v208;
        if ( *v206 != -1 )
          --v464.m128i_i32[1];
        v209 = v438.m128i_i64[0];
        v206[1] = 0;
        *v206 = v209;
LABEL_311:
        v210 = (_QWORD **)(v206 + 1);
        v211 = v206[1];
        if ( (v211 & 2) != 0 )
        {
          v215 = v211 & 0xFFFFFFFFFFFFFFFCLL;
          v210 = *(_QWORD ***)v215;
          v212 = *(_QWORD *)v215 + 8LL * *(unsigned int *)(v215 + 8);
LABEL_314:
          if ( v210 != (_QWORD **)v212 )
          {
            v213 = v210;
            do
            {
              v214 = *v213++;
              *(_BYTE *)(sub_2622110(v415, v214) + 24) = 1;
            }
            while ( (_QWORD **)v212 != v213 );
          }
          goto LABEL_317;
        }
        if ( (v211 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
        {
          v212 = (__int64)(v210 + 1);
          goto LABEL_314;
        }
LABEL_317:
        if ( v393 == ++v205 )
          goto LABEL_298;
      }
      v207 = 2 * v464.m128i_i32[2];
LABEL_324:
      sub_262D290((__int64)&v463, v207);
      sub_2628DF0((__int64)&v463, v438.m128i_i64, &v446);
      v208 = v464.m128i_i32[0] + 1;
      v206 = (__int64 *)v446.m128i_i64[0];
      goto LABEL_308;
    }
LABEL_370:
    v380 = sub_220EEE0(v380);
    if ( v376 != v380 )
      continue;
    break;
  }
  v136 = v386;
LABEL_372:
  if ( v464.m128i_i32[2] )
  {
    v234 = (_QWORD *)*((_QWORD *)&v463 + 1);
    v235 = *((_QWORD *)&v463 + 1) + 16LL * v464.m128i_u32[2];
    do
    {
      if ( *v234 <= 0xFFFFFFFFFFFFFFFDLL )
      {
        v236 = v234[1];
        if ( v236 )
        {
          if ( (v236 & 2) != 0 )
          {
            v237 = (unsigned __int64 *)(v236 & 0xFFFFFFFFFFFFFFFCLL);
            v238 = (unsigned __int64)v237;
            if ( v237 )
            {
              if ( (unsigned __int64 *)*v237 != v237 + 2 )
                _libc_free(*v237);
              j_j___libc_free_0(v238);
            }
          }
        }
      }
      v234 += 2;
    }
    while ( (_QWORD *)v235 != v234 );
  }
  sub_C7D6A0(*((__int64 *)&v463 + 1), 16LL * v464.m128i_u32[2], 8);
LABEL_409:
  if ( v455 )
  {
    v247 = (__int64)v453;
    v416 = 0;
    v417 = 0;
    v418 = 0;
    for ( v446.m128i_i64[0] = (__int64)v453; (_QWORD ***)v446.m128i_i64[0] != &v451; v247 = v446.m128i_i64[0] )
    {
      if ( (*(_BYTE *)(v247 + 40) & 1) != 0 )
      {
        v442.m128i_i32[0] = 0;
        v248 = v247 + 32;
        do
        {
          v250 = *(_QWORD *)(v248 + 16);
          if ( v250 )
          {
            v251 = *(_QWORD *)(v248 + 16) & 3LL;
            if ( v251 == 1 )
            {
              *(_QWORD *)&v463 = v250 & 0xFFFFFFFFFFFFFFFCLL;
              if ( (v250 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
              {
                v249 = *(_DWORD *)sub_26238D0((__int64)&v419, (__int64 *)&v463);
                if ( v442.m128i_i32[0] >= v249 )
                  v249 = v442.m128i_i32[0];
                v442.m128i_i32[0] = v249;
              }
            }
            else if ( v251 == 2 )
            {
              v252 = v250 & 0xFFFFFFFFFFFFFFFCLL;
              if ( v252 )
              {
                v253 = *(_DWORD *)(v252 + 8);
                if ( v442.m128i_i32[0] >= v253 )
                  v253 = v442.m128i_i32[0];
                v442.m128i_i32[0] = v253;
              }
            }
          }
          v248 = *(_QWORD *)(v248 + 8) & 0xFFFFFFFFFFFFFFFELL;
        }
        while ( v248 );
        v254 = v417;
        if ( v417 == v418 )
        {
          sub_26197E0((unsigned __int64 *)&v416, v417, &v446, &v442);
        }
        else
        {
          if ( v417 )
          {
            v417->m128i_i64[0] = v446.m128i_i64[0];
            v254->m128i_i32[2] = v442.m128i_i32[0];
            v254 = v417;
          }
          v417 = (__m128i *)&v254[1];
        }
      }
      v446.m128i_i64[0] = sub_220EF30(v446.m128i_i64[0]);
    }
    v255 = v417;
    v256 = v416;
    if ( v417 != v416 )
    {
      v257 = (char *)v417 - (char *)v416;
      _BitScanReverse64(&v258, v417 - v416);
      sub_2619E90((__int64)v416, (unsigned __int64)v417, 2LL * (int)(63 - (v258 ^ 0x3F)));
      if ( v257 <= 256 )
      {
        sub_261AA90((__int64)v256, v255->m128i_i64);
      }
      else
      {
        sub_261AA90((__int64)v256, v256[16].m128i_i64);
        for ( m = v259; v255 != m; v263->m128i_i32[2] = v262 )
        {
          v261 = m->m128i_i64[0];
          v262 = m->m128i_u32[2];
          v263 = m;
          if ( m[-1].m128i_i32[2] > v262 )
          {
            v264 = m - 1;
            do
            {
              v265 = v264->m128i_i64[0];
              v263 = v264--;
              v264[2].m128i_i64[0] = v265;
              v264[2].m128i_i32[2] = v264[1].m128i_i32[2];
            }
            while ( v262 < v264->m128i_i32[2] );
          }
          ++m;
          v263->m128i_i64[0] = v261;
        }
      }
    }
    v266 = v416;
    v405 = v417;
    if ( v417 != v416 )
    {
      v409 = (__int64 **)v136;
      do
      {
        v442 = 0u;
        v443 = 0;
        v446 = 0u;
        v447 = 0;
        v463 = 0u;
        v464.m128i_i64[0] = 0;
        if ( (*(_BYTE *)(v266->m128i_i64[0] + 40) & 1) != 0 )
        {
          v267 = v266->m128i_i64[0] + 32;
          do
          {
            v269 = *(_QWORD *)(v267 + 16) & 0xFFFFFFFFFFFFFFFCLL;
            v270 = *(_QWORD *)(v267 + 16) & 3LL;
            v438.m128i_i64[0] = v269;
            if ( v270 == 1 )
            {
              v272 = v442.m128i_i64[1];
              if ( (__int64 (__fastcall *)(__int64 *))v442.m128i_i64[1] == v443 )
              {
                sub_914280((__int64)&v442, (_BYTE *)v442.m128i_i64[1], &v438);
              }
              else
              {
                if ( v442.m128i_i64[1] )
                {
                  *(_QWORD *)v442.m128i_i64[1] = v269;
                  v272 = v442.m128i_i64[1];
                }
                v442.m128i_i64[1] = v272 + 8;
              }
            }
            else if ( v270 )
            {
              v271 = *((_QWORD *)&v463 + 1);
              if ( *((_QWORD *)&v463 + 1) == v464.m128i_i64[0] )
              {
                sub_2619340((__int64)&v463, *((_BYTE **)&v463 + 1), &v438);
              }
              else
              {
                if ( *((_QWORD *)&v463 + 1) )
                {
                  **((_QWORD **)&v463 + 1) = v269;
                  v271 = *((_QWORD *)&v463 + 1);
                }
                *((_QWORD *)&v463 + 1) = v271 + 8;
              }
            }
            else
            {
              v268 = v446.m128i_i64[1];
              if ( (__int64 (__fastcall *)(_QWORD *))v446.m128i_i64[1] == v447 )
              {
                sub_26194D0((__int64)&v446, (_BYTE *)v446.m128i_i64[1], &v438);
              }
              else
              {
                if ( v446.m128i_i64[1] )
                {
                  *(_QWORD *)v446.m128i_i64[1] = v269;
                  v268 = v446.m128i_i64[1];
                }
                v446.m128i_i64[1] = v268 + 8;
              }
            }
            v267 = *(_QWORD *)(v267 + 8) & 0xFFFFFFFFFFFFFFFELL;
          }
          while ( v267 );
        }
        sub_2623260((char **)&v442, (__int64)&v419);
        v274 = (char *)*((_QWORD *)&v463 + 1);
        v275 = __PAIR128__(v463, v463);
        if ( *((_QWORD *)&v463 + 1) != (_QWORD)v463 )
        {
          v396 = (char *)v463;
          v276 = *((_QWORD *)&v463 + 1) - v463;
          _BitScanReverse64(&v277, (__int64)(*((_QWORD *)&v463 + 1) - v463) >> 3);
          sub_2619CC0((char *)v463, *((char **)&v463 + 1), 2LL * (int)(63 - (v277 ^ 0x3F)));
          if ( v276 <= 128 )
          {
            sub_2619970(v396, v274);
          }
          else
          {
            v278 = v396 + 128;
            sub_2619970(v396, v396 + 128);
            if ( v274 != v396 + 128 )
            {
              do
              {
                v279 = *((_QWORD *)v278 - 1);
                v280 = *(_QWORD *)v278;
                v281 = v278 - 8;
                if ( *(_DWORD *)(*(_QWORD *)v278 + 8LL) >= *(_DWORD *)(v279 + 8) )
                {
                  v282 = v278;
                }
                else
                {
                  do
                  {
                    *((_QWORD *)v281 + 1) = v279;
                    v282 = v281;
                    v279 = *((_QWORD *)v281 - 1);
                    v281 -= 8;
                  }
                  while ( *(_DWORD *)(v280 + 8) < *(_DWORD *)(v279 + 8) );
                }
                v278 += 8;
                *(_QWORD *)v282 = v280;
              }
              while ( v274 != v278 );
            }
          }
          v275 = v463;
        }
        sub_2635FD0(
          v409,
          (__int64 *)v442.m128i_i64[0],
          (v442.m128i_i64[1] - v442.m128i_i64[0]) >> 3,
          v446.m128i_i64[0],
          (v446.m128i_i64[1] - v446.m128i_i64[0]) >> 3,
          v273,
          (__int64 *)v275,
          (__int64)(*((_QWORD *)&v275 + 1) - v275) >> 3);
        if ( (_QWORD)v463 )
          j_j___libc_free_0(v463);
        if ( v446.m128i_i64[0] )
          j_j___libc_free_0(v446.m128i_u64[0]);
        if ( v442.m128i_i64[0] )
          j_j___libc_free_0(v442.m128i_u64[0]);
        ++v266;
      }
      while ( v405 != v266 );
      v136 = (__int64 *)v409;
    }
    sub_26282B0(v136);
    if ( v136[1] )
    {
      v283 = sub_BA8DC0(*v136, (__int64)"aliases", 7);
      v284 = v283;
      if ( v283 )
      {
        v395 = sub_B91A00(v283);
        if ( v395 )
        {
          for ( n = 0; n != v395; ++n )
          {
            v286 = (_BYTE *)(sub_B91A10(v284, n) - 16);
            v287 = (__int64 *)sub_A17150(v286);
            v406 = sub_B91420(*v287);
            v397 = v288;
            v289 = sub_A17150(v286);
            v290.m128i_i64[0] = sub_B91420(*((_QWORD *)v289 + 1));
            v446 = v290;
            v291 = sub_261D670((__int64)&v456, (__int64)&v446);
            if ( (_BYTE *)v291 != &v460[32 * (unsigned int)v461] && !*(_DWORD *)(v291 + 16) )
            {
              v292 = v446.m128i_i64[0];
              if ( sub_BA8DA0(*v136, v446.m128i_i64[0], v446.m128i_u64[1]) )
              {
                v293 = sub_A17150(v286);
                v295 = sub_AD8340(*(unsigned __int8 **)(*((_QWORD *)v293 + 2) + 136LL), v292, v294);
                if ( *((_DWORD *)v295 + 2) > 0x40u )
                  v295 = *(unsigned __int8 **)v295;
                v296 = *(unsigned __int8 **)v295;
                v297 = sub_A17150(v286);
                v299 = sub_AD8340(*(unsigned __int8 **)(*((_QWORD *)v297 + 3) + 136LL), v292, v298);
                if ( *((_DWORD *)v299 + 2) > 0x40u )
                  v299 = *(unsigned __int8 **)v299;
                v387 = *(unsigned __int8 **)v299;
                v300 = sub_BA8DA0(*v136, v446.m128i_i64[0], v446.m128i_u64[1]);
                LOWORD(v465) = 257;
                v301 = (unsigned __int8 *)sub_B305C0((__int64)&v463, (__int64)v300);
                v301[32] = v301[32] & 0xCF | (16 * ((unsigned __int8)v296 & 3));
                if ( (unsigned __int8)sub_2624ED0((__int64)v301) )
                  v301[33] |= 0x40u;
                if ( v387 )
                {
                  v301[32] = v301[32] & 0xF0 | 4;
                  if ( (unsigned __int8)sub_2624ED0((__int64)v301) )
                    v301[33] |= 0x40u;
                }
                v302 = sub_BA8CB0(*v136, v406, (unsigned __int64)v397);
                v303 = v302;
                if ( v302 )
                {
                  sub_BD6B90(v301, v302);
                  sub_BD84D0((__int64)v303, (__int64)v301);
                  sub_B2E860(v303);
                }
                else
                {
                  LOWORD(v465) = 261;
                  *(_QWORD *)&v463 = v406;
                  *((_QWORD *)&v463 + 1) = v397;
                  sub_BD6B50(v301, (const char **)&v463);
                }
              }
            }
          }
        }
      }
      if ( v136[1] )
      {
        v304 = sub_BA8DC0(*v136, (__int64)"symvers", 7);
        v305 = v304;
        if ( v304 )
        {
          v398 = sub_B91A00(v304);
          if ( v398 )
          {
            for ( ii = 0; ii != v398; ++ii )
            {
              v315 = (_BYTE *)(sub_B91A10(v305, ii) - 16);
              v316 = (__int64 *)sub_A17150(v315);
              v414.m128i_i64[0] = sub_B91420(*v316);
              v414.m128i_i64[1] = v317;
              v318 = sub_A17150(v315);
              v319 = sub_B91420(*((_QWORD *)v318 + 1));
              v391 = v320;
              v321 = v457 + 24LL * v459;
              v322 = sub_262B840((__int64)&v456, (__int64)&v414);
              if ( !v322 )
              {
                v323 = 3LL * v459;
                v322 = v457 + 24LL * v459;
              }
              if ( v322 != v321 )
              {
                v446.m128i_i64[1] = v391;
                v449 = 261;
                v307 = *v136;
                v438.m128i_i64[0] = (__int64)", ";
                v436 = 261;
                v435 = v414;
                v446.m128i_i64[0] = v319;
                v432[0].m128i_i64[0] = (__int64)".symver ";
                v441 = 1;
                v440 = 3;
                v434 = 1;
                v433 = 3;
                sub_9C6370(v437, v432, &v435, v323, v324, v325);
                sub_9C6370(&v442, v437, &v438, v308, v309, v310);
                sub_9C6370((__m128i *)&v463, &v442, &v446, v311, v312, v313);
                sub_CA0F50((__int64 *)&v430, (void **)&v463);
                if ( v431 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(v307 + 96) )
                  sub_4262D8((__int64)"basic_string::append");
                sub_2241490((unsigned __int64 *)(v307 + 88), v430, v431);
                v314 = *(_QWORD *)(v307 + 96);
                if ( v314 && *(_BYTE *)(*(_QWORD *)(v307 + 88) + v314 - 1) != 10 )
                  sub_2240F50((unsigned __int64 *)(v307 + 88), 10);
                sub_2240A30((unsigned __int64 *)&v430);
              }
            }
          }
        }
      }
    }
    if ( v416 )
      j_j___libc_free_0((unsigned __int64)v416);
    v326 = 1;
  }
  else
  {
    v326 = 0;
  }
  v410 = v326;
  sub_C7D6A0(v427, 16LL * v429, 8);
  v327 = v410;
  if ( v460 != v462 )
  {
    _libc_free((unsigned __int64)v460);
    v327 = v410;
  }
  v411 = v327;
  sub_C7D6A0(v457, 24LL * v459, 8);
  v328 = v411;
  if ( src != v425 )
  {
    _libc_free((unsigned __int64)src);
    v328 = v411;
  }
  v329 = v422;
  if ( v422 )
  {
    v330 = v420;
    v331 = v328;
    v332 = &v420[5 * v422];
    do
    {
      if ( *v330 != -8192 && *v330 != -4096 )
      {
        v333 = v330[2];
        if ( v333 )
          j_j___libc_free_0(v333);
      }
      v330 += 5;
    }
    while ( v332 != v330 );
    v329 = v422;
    v328 = v331;
  }
  v412 = v328;
  v334 = 40 * v329;
  sub_C7D6A0((__int64)v420, 40 * v329, 8);
  sub_B72320((__int64)&v468, v334);
  sub_261B4A0((unsigned __int64)v452);
  return v412;
}
