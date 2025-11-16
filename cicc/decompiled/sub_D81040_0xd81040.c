// Function: sub_D81040
// Address: 0xd81040
//
__int64 __fastcall sub_D81040(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  bool v6; // bl
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  bool v11; // cl
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rbx
  __int64 *v19; // r13
  __int64 v20; // r12
  __int64 *v21; // rax
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // r14
  __int64 v29; // r12
  __int64 v30; // r15
  void (__fastcall *v31)(__int64 **, __int64, __int64); // rax
  __int64 v32; // r9
  _BYTE *v33; // rbx
  _BYTE *v34; // r12
  __int64 v35; // r13
  __int64 v36; // rdi
  __m128i *v37; // r12
  __int64 v38; // r9
  __int64 v39; // rsi
  __int8 v40; // bl
  __int64 v41; // rdx
  unsigned __int8 v42; // al
  bool v43; // r15
  char v44; // bl
  __int64 v45; // r8
  __int64 v46; // r9
  char v47; // r13
  bool v48; // al
  __int64 v49; // rcx
  __int64 v50; // rdx
  _QWORD *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r14
  __int64 v58; // rdx
  char v59; // dl
  __m128i *v60; // rdi
  __m128i *v61; // rsi
  _QWORD *v62; // r12
  _BYTE *v63; // r15
  __int64 v64; // rdx
  __int16 v65; // r14
  unsigned __int8 v66; // r13
  __int16 v67; // bx
  __int64 v68; // rax
  __int64 v69; // rcx
  __int16 v70; // ax
  __int64 v71; // rax
  __int64 v72; // rcx
  unsigned __int64 v73; // r14
  _QWORD *v74; // rax
  _QWORD *v75; // rdi
  __int64 v76; // rsi
  __int64 v77; // rdx
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // rax
  __int64 v80; // rdx
  _QWORD *k; // r14
  __int64 v82; // rdi
  __int64 *v83; // rax
  __int64 *v84; // r14
  __int64 v85; // rsi
  __int64 *v86; // r13
  const char *v87; // rsi
  __int64 v88; // rdi
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r14
  int v93; // edx
  __int64 v94; // r8
  int v95; // r15d
  char *v96; // rax
  __int64 v97; // r8
  char *v98; // rcx
  __int64 v99; // rdx
  __int64 v100; // r8
  char *v101; // r8
  int v102; // ecx
  __int64 v103; // rdi
  int v104; // edx
  __int64 v105; // r11
  __int64 *v106; // rax
  __int64 v107; // r9
  _QWORD *v108; // rax
  __int64 v109; // r8
  _QWORD *v110; // r15
  __int64 v111; // rdx
  __int64 v112; // r8
  int v113; // ecx
  _QWORD *v114; // r13
  __int64 v115; // rdi
  int v116; // edx
  __int64 v117; // rdi
  int v118; // edx
  int v119; // r10d
  int v120; // ecx
  __int64 v121; // r8
  _QWORD *v122; // rdi
  __int64 v123; // r8
  unsigned int v124; // eax
  int v125; // eax
  unsigned __int64 v126; // rax
  unsigned __int64 v127; // rax
  __int64 v128; // r14
  int v129; // eax
  __int64 v130; // r14
  _QWORD *v131; // rax
  _QWORD *j; // rdx
  __int64 v133; // rax
  __int64 v134; // rcx
  __int64 v135; // r14
  __int64 v136; // rbx
  __int64 v137; // rax
  __int64 v138; // rdx
  unsigned __int8 v139; // al
  _QWORD *v140; // rax
  _BYTE *v141; // rdi
  __int64 v142; // rax
  _QWORD *v143; // rax
  size_t v144; // rdx
  char *v145; // rax
  __int64 v146; // rbx
  _QWORD *v147; // r12
  char *v148; // r14
  size_t v149; // rdx
  size_t v150; // r13
  size_t v151; // r15
  size_t v152; // rdx
  int v153; // eax
  size_t v154; // rbx
  size_t v155; // rdx
  int v156; // eax
  _QWORD *v157; // rax
  _QWORD *v158; // rsi
  __int64 v159; // rax
  _QWORD *v160; // rdx
  _QWORD *v161; // r15
  _BOOL8 v162; // rdi
  __int64 v163; // rbx
  _QWORD *v164; // rax
  _QWORD *v165; // rbx
  _QWORD *v166; // r14
  _QWORD *v167; // r15
  unsigned __int64 v168; // r13
  char *v169; // rax
  char *v170; // rax
  __int64 v171; // rsi
  __int64 v172; // rax
  char *v173; // r15
  char *v174; // r13
  __int64 v175; // r12
  __int64 *v176; // rbx
  __int64 *v177; // r14
  __int64 v178; // rdi
  __int64 v179; // rax
  __int64 v180; // rax
  unsigned int v181; // eax
  __int64 v182; // rdx
  char v183; // al
  __int64 v184; // rdi
  __int64 v185; // rdi
  __int64 *v186; // r15
  __int64 *v187; // rbx
  __int64 v188; // rdi
  __int64 *v189; // r15
  __m128i *v190; // rax
  const __m128i *v191; // rdx
  __int8 *v192; // r15
  __m128i *v193; // r13
  __m128i *v194; // rbx
  __m128i **v195; // rax
  _QWORD *v196; // r15
  int v197; // r13d
  char *v198; // r12
  __int64 v199; // rdi
  int v200; // edx
  __int64 v201; // r11
  int v202; // r13d
  __int64 v203; // rdi
  int v204; // edx
  __int64 v205; // r11
  int v206; // r13d
  int v207; // edx
  __int64 v208; // r8
  int v209; // edx
  int v210; // ecx
  __int64 v211; // rdi
  int v212; // r11d
  _QWORD *v213; // r12
  __int64 v214; // rdi
  int v215; // edx
  __int64 v216; // r8
  int v217; // r11d
  __int64 v218; // rdi
  int v219; // edx
  __int64 v220; // r8
  int v221; // r11d
  __int64 v222; // rdi
  int v223; // edx
  __int64 v224; // r8
  int v225; // r11d
  __int64 v226; // rdx
  __int64 v227; // r12
  __int64 v228; // rsi
  int v230; // r12d
  __int64 v231; // rdi
  int v232; // edx
  __int64 v233; // r11
  unsigned __int64 v234; // r12
  __int64 v235; // rax
  __int64 *v236; // rax
  _QWORD *v237; // rdi
  int v238; // r11d
  char *v239; // r10
  size_t v240; // rbx
  size_t v241; // rdx
  unsigned int v242; // edi
  __int64 v243; // r10
  int v244; // ecx
  __int64 v245; // r8
  int v246; // edi
  __int64 v247; // r10
  int v248; // ecx
  __int64 v249; // r8
  int v250; // edi
  __int64 *v251; // rax
  __int64 *v252; // rbx
  __int64 *v253; // r15
  __int64 v254; // rdi
  unsigned int v255; // ecx
  __int64 v256; // rsi
  __int64 *v257; // rbx
  __int64 v258; // rdi
  __int64 v259; // rdi
  int v260; // ecx
  __int64 v261; // r8
  int v262; // r10d
  __int64 v263; // r10
  int v264; // ecx
  __int64 v265; // r8
  int v266; // edi
  unsigned int v267; // eax
  __int64 v268; // rdi
  int v269; // esi
  char *v270; // rdi
  unsigned int v271; // eax
  __int64 v272; // rsi
  unsigned int v273; // r11d
  __int64 v274; // rcx
  __int64 v275; // r8
  __int64 v276; // r9
  __int64 v277; // rcx
  __int64 v278; // r8
  __int64 v279; // r9
  __int64 v280; // rcx
  __int64 v281; // r8
  __int64 v282; // r9
  __int64 v283; // rcx
  __int64 v284; // r8
  __int64 v285; // r9
  __int64 *v286; // [rsp+10h] [rbp-550h]
  __int64 v287; // [rsp+18h] [rbp-548h]
  __int64 v288; // [rsp+20h] [rbp-540h]
  _QWORD *v289; // [rsp+28h] [rbp-538h]
  unsigned __int8 v290; // [rsp+38h] [rbp-528h]
  _QWORD *v291; // [rsp+38h] [rbp-528h]
  __int64 v292; // [rsp+40h] [rbp-520h]
  __int8 v295; // [rsp+50h] [rbp-510h]
  unsigned __int8 v298; // [rsp+6Eh] [rbp-4F2h]
  char v299; // [rsp+6Fh] [rbp-4F1h]
  __int64 v300; // [rsp+70h] [rbp-4F0h]
  bool v301; // [rsp+70h] [rbp-4F0h]
  int v302; // [rsp+78h] [rbp-4E8h]
  __int64 *v303; // [rsp+78h] [rbp-4E8h]
  __int64 v304; // [rsp+88h] [rbp-4D8h]
  __int64 v305; // [rsp+90h] [rbp-4D0h]
  bool v306; // [rsp+90h] [rbp-4D0h]
  _BYTE *v307; // [rsp+90h] [rbp-4D0h]
  unsigned __int8 v308; // [rsp+90h] [rbp-4D0h]
  __int64 v309; // [rsp+90h] [rbp-4D0h]
  __int64 v310; // [rsp+90h] [rbp-4D0h]
  _QWORD *v311; // [rsp+98h] [rbp-4C8h]
  _QWORD *v312; // [rsp+A0h] [rbp-4C0h]
  __int64 v313; // [rsp+A0h] [rbp-4C0h]
  char *v314; // [rsp+A0h] [rbp-4C0h]
  __int64 v315; // [rsp+A8h] [rbp-4B8h]
  bool v316; // [rsp+A8h] [rbp-4B8h]
  __int64 v317; // [rsp+A8h] [rbp-4B8h]
  __int64 v318; // [rsp+A8h] [rbp-4B8h]
  __int64 v319; // [rsp+A8h] [rbp-4B8h]
  unsigned __int64 v320; // [rsp+A8h] [rbp-4B8h]
  unsigned __int8 v321; // [rsp+BFh] [rbp-4A1h] BYREF
  int v322; // [rsp+C0h] [rbp-4A0h] BYREF
  __int64 v323; // [rsp+C8h] [rbp-498h]
  __int64 v324; // [rsp+D0h] [rbp-490h] BYREF
  const char *v325; // [rsp+D8h] [rbp-488h]
  __int64 v326; // [rsp+E0h] [rbp-480h]
  __int64 v327; // [rsp+E8h] [rbp-478h]
  __int64 *v328; // [rsp+F0h] [rbp-470h] BYREF
  __int64 v329; // [rsp+F8h] [rbp-468h]
  _BYTE v330[16]; // [rsp+100h] [rbp-460h] BYREF
  _BYTE v331[32]; // [rsp+110h] [rbp-450h] BYREF
  __m128i v332[2]; // [rsp+130h] [rbp-430h] BYREF
  char v333; // [rsp+150h] [rbp-410h]
  char v334; // [rsp+151h] [rbp-40Fh]
  __m128i v335[2]; // [rsp+160h] [rbp-400h] BYREF
  __int16 v336; // [rsp+180h] [rbp-3E0h]
  __m128i v337[3]; // [rsp+190h] [rbp-3D0h] BYREF
  __m128i v338[2]; // [rsp+1C0h] [rbp-3A0h] BYREF
  char v339; // [rsp+1E0h] [rbp-380h]
  char v340; // [rsp+1E1h] [rbp-37Fh]
  __m128i v341[3]; // [rsp+1F0h] [rbp-370h] BYREF
  __m128i v342[2]; // [rsp+220h] [rbp-340h] BYREF
  __int16 v343; // [rsp+240h] [rbp-320h]
  __m128i v344; // [rsp+250h] [rbp-310h] BYREF
  __int64 v345; // [rsp+260h] [rbp-300h]
  __int64 *v346; // [rsp+280h] [rbp-2E0h] BYREF
  __int64 v347; // [rsp+288h] [rbp-2D8h]
  _BYTE v348[32]; // [rsp+290h] [rbp-2D0h] BYREF
  __int64 v349; // [rsp+2B0h] [rbp-2B0h] BYREF
  __int64 *v350; // [rsp+2B8h] [rbp-2A8h]
  __int64 v351; // [rsp+2C0h] [rbp-2A0h]
  int v352; // [rsp+2C8h] [rbp-298h]
  char v353; // [rsp+2CCh] [rbp-294h]
  char v354; // [rsp+2D0h] [rbp-290h] BYREF
  __m128i v355; // [rsp+2F0h] [rbp-270h] BYREF
  _QWORD *v356; // [rsp+300h] [rbp-260h] BYREF
  _BYTE *v357; // [rsp+308h] [rbp-258h]
  __int64 v358; // [rsp+310h] [rbp-250h]
  _BYTE v359[56]; // [rsp+318h] [rbp-248h] BYREF
  __int64 v360; // [rsp+350h] [rbp-210h]
  __int64 v361; // [rsp+358h] [rbp-208h]
  char v362; // [rsp+360h] [rbp-200h]
  __int64 v363; // [rsp+364h] [rbp-1FCh]
  __m128i v364; // [rsp+370h] [rbp-1F0h] BYREF
  __int64 v365; // [rsp+380h] [rbp-1E0h]
  __int64 v366; // [rsp+388h] [rbp-1D8h]
  char *v367; // [rsp+390h] [rbp-1D0h] BYREF
  char *v368; // [rsp+398h] [rbp-1C8h]
  __int64 v369; // [rsp+3A0h] [rbp-1C0h] BYREF
  __int64 v370; // [rsp+3A8h] [rbp-1B8h]
  __int64 i; // [rsp+3B0h] [rbp-1B0h]
  __int64 *v372; // [rsp+3B8h] [rbp-1A8h]
  unsigned int v373; // [rsp+3C0h] [rbp-1A0h]
  char v374; // [rsp+3C8h] [rbp-198h] BYREF
  __int64 *v375; // [rsp+3E8h] [rbp-178h]
  unsigned int v376; // [rsp+3F0h] [rbp-170h]
  __int64 v377; // [rsp+3F8h] [rbp-168h] BYREF
  __int64 *v378; // [rsp+410h] [rbp-150h] BYREF
  _QWORD *v379; // [rsp+418h] [rbp-148h]
  __int64 v380; // [rsp+420h] [rbp-140h] BYREF
  __int64 v381; // [rsp+428h] [rbp-138h]
  _QWORD v382[9]; // [rsp+430h] [rbp-130h] BYREF
  char v383; // [rsp+478h] [rbp-E8h] BYREF
  _QWORD v384[2]; // [rsp+4B8h] [rbp-A8h] BYREF
  char v385; // [rsp+4C8h] [rbp-98h] BYREF
  char v386; // [rsp+528h] [rbp-38h] BYREF

  v5 = a1;
  v6 = 0;
  v7 = sub_BA91D0((__int64)a2, "EnableSplitLTOUnit", 0x12u);
  if ( v7 )
  {
    v8 = *(_QWORD *)(v7 + 136);
    v6 = 0;
    if ( v8 )
    {
      if ( *(_DWORD *)(v8 + 32) <= 0x40u )
        v9 = *(_QWORD *)(v8 + 24);
      else
        v9 = **(_QWORD **)(v8 + 24);
      v6 = v9 != 0;
    }
  }
  v10 = sub_BA91D0((__int64)a2, "UnifiedLTO", 0xAu);
  v11 = 0;
  if ( v10 )
  {
    v12 = *(_QWORD *)(v10 + 136);
    v11 = 0;
    if ( v12 )
    {
      if ( *(_DWORD *)(v12 + 32) <= 0x40u )
        v13 = *(_QWORD *)(v12 + 24);
      else
        v13 = **(_QWORD **)(v12 + 24);
      v11 = v13 != 0;
    }
  }
  *(_DWORD *)(a1 + 8) = 0;
  v311 = (_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 64) = 0x2000000000LL;
  *(_QWORD *)(a1 + 168) = a1 + 72;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 232) = a1 + 216;
  *(_QWORD *)(a1 + 240) = a1 + 216;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  v289 = (_QWORD *)(a1 + 264);
  *(_QWORD *)(a1 + 280) = a1 + 264;
  *(_QWORD *)(a1 + 288) = a1 + 264;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 1;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 432) = a1 + 448;
  *(_QWORD *)(a1 + 336) = 0x100000000000000LL;
  *(_QWORD *)(a1 + 480) = a1 + 496;
  *(_BYTE *)(a1 + 344) = v6;
  *(_BYTE *)(a1 + 345) = v11;
  *(_WORD *)(a1 + 346) = 0;
  *(_QWORD *)(a1 + 440) = 0x400000000LL;
  *(_QWORD *)(a1 + 512) = a1 + 416;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 1;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_DWORD *)(a1 + 576) = 0;
  v350 = (__int64 *)&v354;
  v346 = (__int64 *)v348;
  v349 = 0;
  v351 = 4;
  v352 = 0;
  v353 = 1;
  v347 = 0x400000000LL;
  sub_BAA9B0((__int64)a2, (__int64)&v346, 0);
  sub_BAA9B0((__int64)a2, (__int64)&v346, 1);
  v18 = v346;
  v324 = 0;
  v325 = 0;
  v326 = 0;
  v19 = &v346[(unsigned int)v347];
  v327 = 0;
  if ( v346 != v19 )
  {
    do
    {
      v20 = *v18;
      if ( (*(_BYTE *)(*v18 + 32) & 0xFu) - 7 > 1 )
        goto LABEL_22;
      if ( !v353 )
        goto LABEL_162;
      v21 = v350;
      v15 = HIDWORD(v351);
      v14 = (__int64)&v350[HIDWORD(v351)];
      if ( v350 != (__int64 *)v14 )
      {
        while ( v20 != *v21 )
        {
          if ( (__int64 *)v14 == ++v21 )
            goto LABEL_163;
        }
        goto LABEL_18;
      }
LABEL_163:
      if ( HIDWORD(v351) < (unsigned int)v351 )
      {
        ++HIDWORD(v351);
        *(_QWORD *)v14 = v20;
        ++v349;
      }
      else
      {
LABEL_162:
        sub_C8CC70((__int64)&v349, *v18, v14, v15, v16, v17);
      }
LABEL_18:
      sub_B2F930(&v378, v20);
      v22 = sub_B2F650((__int64)v378, (__int64)v379);
      if ( v378 != &v380 )
        j_j___libc_free_0(v378, v380 + 1);
      if ( !(_DWORD)v327 )
      {
        ++v324;
        goto LABEL_427;
      }
      v17 = (unsigned int)(v327 - 1);
      v16 = (__int64)v325;
      v23 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (484763065 * (_DWORD)v22)) & ((_DWORD)v327 - 1);
      v14 = (__int64)&v325[8 * v23];
      v15 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 != v22 )
      {
        v238 = 1;
        v239 = 0;
        while ( v15 != -1 )
        {
          if ( v239 || v15 != -2 )
            v14 = (__int64)v239;
          LODWORD(v23) = v17 & (v238 + v23);
          v15 = *(_QWORD *)&v325[8 * (unsigned int)v23];
          if ( v22 == v15 )
            goto LABEL_22;
          ++v238;
          v239 = (char *)v14;
          v14 = (__int64)&v325[8 * (unsigned int)v23];
        }
        if ( !v239 )
          v239 = (char *)v14;
        ++v324;
        v15 = (unsigned int)(v326 + 1);
        if ( 4 * (int)v15 >= (unsigned int)(3 * v327) )
        {
LABEL_427:
          sub_A32210((__int64)&v324, 2 * v327);
          if ( !(_DWORD)v327 )
            goto LABEL_477;
          v16 = (unsigned int)(v327 - 1);
          v17 = (__int64)v325;
          v15 = (unsigned int)(v326 + 1);
          v267 = v16 & (((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (484763065 * v22));
          v14 = v267;
          v239 = (char *)&v325[8 * v267];
          v268 = *(_QWORD *)v239;
          if ( v22 != *(_QWORD *)v239 )
          {
            v269 = 1;
            v14 = 0;
            while ( v268 != -1 )
            {
              if ( !v14 && v268 == -2 )
                v14 = (__int64)v239;
              v267 = v16 & (v269 + v267);
              v239 = (char *)&v325[8 * v267];
              v268 = *(_QWORD *)v239;
              if ( v22 == *(_QWORD *)v239 )
                goto LABEL_372;
              ++v269;
            }
            if ( v14 )
              v239 = (char *)v14;
          }
        }
        else
        {
          v14 = (unsigned int)(v327 - HIDWORD(v326) - v15);
          if ( (unsigned int)v14 <= (unsigned int)v327 >> 3 )
          {
            v320 = ((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (0xBF58476D1CE4E5B9LL * v22);
            sub_A32210((__int64)&v324, v327);
            if ( !(_DWORD)v327 )
            {
LABEL_477:
              LODWORD(v326) = v326 + 1;
              BUG();
            }
            v16 = (unsigned int)(v327 - 1);
            v270 = 0;
            v17 = (__int64)v325;
            v271 = v16 & v320;
            v15 = (unsigned int)(v326 + 1);
            v239 = (char *)&v325[8 * ((unsigned int)v16 & (unsigned int)v320)];
            v14 = 1;
            v272 = *(_QWORD *)v239;
            if ( *(_QWORD *)v239 != v22 )
            {
              while ( v272 != -1 )
              {
                if ( v272 == -2 && !v270 )
                  v270 = v239;
                v273 = v14 + 1;
                v271 = v16 & (v14 + v271);
                v14 = v271;
                v239 = (char *)&v325[8 * v271];
                v272 = *(_QWORD *)v239;
                if ( v22 == *(_QWORD *)v239 )
                  goto LABEL_372;
                v14 = v273;
              }
              if ( v270 )
                v239 = v270;
            }
          }
        }
LABEL_372:
        LODWORD(v326) = v15;
        if ( *(_QWORD *)v239 != -1 )
          --HIDWORD(v326);
        *(_QWORD *)v239 = v22;
      }
LABEL_22:
      ++v18;
    }
    while ( v19 != v18 );
  }
  v321 = 0;
  if ( a2[12] )
  {
    v381 = v5;
    v378 = (__int64 *)&v321;
    v379 = a2;
    v380 = (__int64)&v324;
    sub_C136E0((__int64)a2, (__int64 (__fastcall *)(__int64, const char *, __int64, __int64))sub_D7C550, (__int64)&v378);
  }
  v24 = (__int64)"ThinLTO";
  v25 = sub_BA91D0((__int64)a2, "ThinLTO", 7u);
  if ( v25 )
  {
    v26 = *(_QWORD *)(v25 + 136);
    v299 = 1;
    if ( v26 )
    {
      if ( *(_DWORD *)(v26 + 32) <= 0x40u )
        v27 = *(_QWORD *)(v26 + 24);
      else
        v27 = **(_QWORD **)(v26 + 24);
      v299 = v27 != 0;
    }
  }
  else
  {
    v299 = 1;
  }
  v28 = (_QWORD *)a2[4];
  if ( a2 + 3 != v28 )
  {
    v300 = v5;
    while ( 1 )
    {
      v29 = (__int64)(v28 - 7);
      if ( !v28 )
        v29 = 0;
      if ( sub_B2FC80(v29) )
        goto LABEL_56;
      v362 = 0;
      v355.m128i_i64[0] = (__int64)&v356;
      v355.m128i_i64[1] = 0x100000000LL;
      v357 = v359;
      v363 = 0;
      v358 = 0x600000000LL;
      v360 = 0;
      v361 = v29;
      HIDWORD(v363) = *(_DWORD *)(v29 + 92);
      sub_B1F440((__int64)&v355);
      if ( *(_QWORD *)(a3 + 16) )
      {
        v30 = 0;
        v305 = (*(__int64 (__fastcall **)(__int64, __int64))(a3 + 24))(a3, v29);
      }
      else
      {
        v30 = 0;
        sub_B2EE70((__int64)&v378, v29, 0);
        v305 = 0;
        if ( (_BYTE)v380 )
        {
          sub_D51D90((__int64)&v364, (__int64)&v355);
          v169 = &v383;
          v378 = 0;
          v379 = 0;
          v380 = 0;
          v381 = 0;
          memset(v382, 0, 64);
          v382[8] = 1;
          do
          {
            *(_QWORD *)v169 = -4096;
            v169 += 16;
          }
          while ( v169 != (char *)v384 );
          v384[0] = 0;
          v170 = &v385;
          v384[1] = 1;
          do
          {
            *(_QWORD *)v170 = -4096;
            v170 += 24;
            *((_DWORD *)v170 - 4) = 0x7FFFFFFF;
          }
          while ( v170 != &v386 );
          v171 = v29;
          sub_FF9360(&v378, v29, &v364, 0, 0, 0);
          v172 = sub_22077B0(8);
          v305 = v172;
          if ( v172 )
          {
            v171 = v29;
            sub_FE7FB0(v172, v29, &v378, &v364);
          }
          sub_D77880((__int64)&v378);
          sub_D786F0((__int64)&v364);
          v173 = v368;
          v174 = v367;
          if ( v367 != v368 )
          {
            v292 = v29;
            v291 = v28;
            do
            {
              v175 = *(_QWORD *)v174;
              v176 = *(__int64 **)(*(_QWORD *)v174 + 16LL);
              if ( *(__int64 **)(*(_QWORD *)v174 + 8LL) == v176 )
              {
                *(_BYTE *)(v175 + 152) = 1;
              }
              else
              {
                v177 = *(__int64 **)(*(_QWORD *)v174 + 8LL);
                do
                {
                  v178 = *v177++;
                  sub_D47BB0(v178, v171);
                }
                while ( v176 != v177 );
                *(_BYTE *)(v175 + 152) = 1;
                v179 = *(_QWORD *)(v175 + 8);
                if ( *(_QWORD *)(v175 + 16) != v179 )
                  *(_QWORD *)(v175 + 16) = v179;
              }
              v180 = *(_QWORD *)(v175 + 32);
              if ( v180 != *(_QWORD *)(v175 + 40) )
                *(_QWORD *)(v175 + 40) = v180;
              ++*(_QWORD *)(v175 + 56);
              if ( *(_BYTE *)(v175 + 84) )
              {
                *(_QWORD *)v175 = 0;
              }
              else
              {
                v181 = 4 * (*(_DWORD *)(v175 + 76) - *(_DWORD *)(v175 + 80));
                v182 = *(unsigned int *)(v175 + 72);
                if ( v181 < 0x20 )
                  v181 = 32;
                if ( (unsigned int)v182 > v181 )
                {
                  sub_C8C990(v175 + 56, v171);
                }
                else
                {
                  v171 = 0xFFFFFFFFLL;
                  memset(*(void **)(v175 + 64), -1, 8 * v182);
                }
                v183 = *(_BYTE *)(v175 + 84);
                *(_QWORD *)v175 = 0;
                if ( !v183 )
                  _libc_free(*(_QWORD *)(v175 + 64), v171);
              }
              v184 = *(_QWORD *)(v175 + 32);
              if ( v184 )
              {
                v171 = *(_QWORD *)(v175 + 48) - v184;
                j_j___libc_free_0(v184, v171);
              }
              v185 = *(_QWORD *)(v175 + 8);
              if ( v185 )
              {
                v171 = *(_QWORD *)(v175 + 24) - v185;
                j_j___libc_free_0(v185, v171);
              }
              v174 += 8;
            }
            while ( v173 != v174 );
            v29 = v292;
            v28 = v291;
            if ( v367 != v368 )
              v368 = v367;
          }
          v186 = v375;
          v187 = &v375[2 * v376];
          if ( v375 != v187 )
          {
            do
            {
              v171 = v186[1];
              v188 = *v186;
              v186 += 2;
              sub_C7D6A0(v188, v171, 16);
            }
            while ( v187 != v186 );
          }
          v376 = 0;
          if ( v373 )
          {
            v251 = v372;
            v377 = 0;
            v252 = &v372[v373];
            v253 = v372 + 1;
            v370 = *v372;
            for ( i = v370 + 4096; v252 != v253; v251 = v372 )
            {
              v254 = *v253;
              v255 = (unsigned int)(v253 - v251) >> 7;
              v256 = 4096LL << v255;
              if ( v255 >= 0x1E )
                v256 = 0x40000000000LL;
              ++v253;
              sub_C7D6A0(v254, v256, 16);
            }
            v373 = 1;
            v171 = 4096;
            sub_C7D6A0(*v251, 4096, 16);
            v257 = v375;
            v189 = &v375[2 * v376];
            if ( v375 != v189 )
            {
              do
              {
                v171 = v257[1];
                v258 = *v257;
                v257 += 2;
                sub_C7D6A0(v258, v171, 16);
              }
              while ( v189 != v257 );
              goto LABEL_271;
            }
          }
          else
          {
LABEL_271:
            v189 = v375;
          }
          if ( v189 != &v377 )
            _libc_free(v189, v171);
          if ( v372 != (__int64 *)&v374 )
            _libc_free(v372, v171);
          if ( v367 )
            j_j___libc_free_0(v367, v369 - (_QWORD)v367);
          sub_C7D6A0(v364.m128i_i64[1], 16LL * (unsigned int)v366, 8);
          v30 = v305;
        }
      }
      v380 = 0;
      v31 = *(void (__fastcall **)(__int64 **, __int64, __int64))(a5 + 16);
      if ( v31 )
      {
        v31(&v378, a5, 2);
        v381 = *(_QWORD *)(a5 + 24);
        v380 = *(_QWORD *)(a5 + 16);
      }
      v32 = 1;
      if ( HIDWORD(v351) == v352 )
        v32 = v321;
      v24 = v29;
      sub_D7D4E0(v300, v29, v305, a4, (__int64)&v355, v32, (__int64)&v324, v299, (__int64)&v378);
      if ( v380 )
      {
        v24 = (__int64)&v378;
        ((void (__fastcall *)(__int64 **, __int64 **, __int64))v380)(&v378, &v378, 3);
      }
      if ( v30 )
      {
        sub_FDC110(v30);
        v24 = 8;
        j_j___libc_free_0(v30, 8);
      }
      v33 = v357;
      v34 = &v357[8 * (unsigned int)v358];
      if ( v357 != v34 )
      {
        do
        {
          v35 = *((_QWORD *)v34 - 1);
          v34 -= 8;
          if ( v35 )
          {
            v36 = *(_QWORD *)(v35 + 24);
            if ( v36 != v35 + 40 )
              _libc_free(v36, v24);
            v24 = 80;
            j_j___libc_free_0(v35, 80);
          }
        }
        while ( v33 != v34 );
        v34 = v357;
      }
      if ( v34 != v359 )
        _libc_free(v34, v24);
      if ( (_QWORD **)v355.m128i_i64[0] != &v356 )
        _libc_free(v355.m128i_i64[0], v24);
LABEL_56:
      v28 = (_QWORD *)v28[1];
      if ( a2 + 3 == v28 )
      {
        v5 = v300;
        break;
      }
    }
  }
  v328 = (__int64 *)v330;
  v329 = 0x200000000LL;
  v312 = (_QWORD *)a2[2];
  if ( a2 + 1 != v312 )
  {
    v315 = v5;
    while ( 1 )
    {
      v37 = 0;
      if ( v312 )
        v37 = (__m128i *)(v312 - 7);
      if ( sub_B2FC80((__int64)v37) )
        goto LABEL_99;
      v39 = (__int64)v37;
      v367 = (char *)&v369;
      v364 = 0u;
      v365 = 0;
      v366 = 0;
      v368 = 0;
      v378 = 0;
      v379 = v382;
      v380 = 8;
      LODWORD(v381) = 0;
      BYTE4(v381) = 1;
      v341[0].m128i_i8[0] = 0;
      v40 = sub_D7B190(v315, v37, (__m128i **)&v364, (__int64)&v378, v341, v38);
      if ( !v40 )
        v40 = v341[0].m128i_i8[0];
      v295 = v40;
      sub_B32650(v37, (__int64)v37);
      v301 = 0;
      if ( v41 )
        v301 = (v37[2].m128i_i8[0] & 0xFu) - 7 <= 1;
      LOBYTE(a5) = sub_B2FE60(v37);
      LOBYTE(v292) = (v37[2].m128i_i8[1] & 0x40) != 0;
      v42 = v37[2].m128i_u8[0];
      v344 = 0u;
      v298 = v42 & 0xF;
      v345 = 0;
      v290 = (v42 >> 4) & 3;
      if ( !*(_BYTE *)(v315 + 344) )
      {
        v39 = 19;
        LODWORD(v329) = 0;
        sub_B91D10((__int64)v37, 19, (__int64)&v328);
        v137 = (unsigned int)v329;
        if ( (_DWORD)v329 )
        {
          if ( (v37[5].m128i_i8[0] & 1) != 0 )
          {
            v39 = 0;
            sub_D79860((unsigned __int8 *)v37[-2].m128i_i64[0], 0, (__int64)a2, v315, (__int64)&v344, (__int64)v37);
            v137 = (unsigned int)v329;
          }
          v286 = &v328[v137];
          if ( v328 != v286 )
          {
            v303 = v328;
            v287 = (__int64)v37;
            while ( 1 )
            {
              v138 = *v303;
              v139 = *(_BYTE *)(*v303 - 16);
              if ( (v139 & 2) != 0 )
              {
                v140 = *(_QWORD **)(v138 - 32);
                v141 = (_BYTE *)v140[1];
              }
              else
              {
                v226 = v138 - 8LL * ((v139 >> 2) & 0xF);
                v141 = *(_BYTE **)(v226 - 8);
                v140 = (_QWORD *)(v226 - 16);
              }
              v142 = *(_QWORD *)(*v140 + 136LL);
              if ( *(_DWORD *)(v142 + 32) <= 0x40u )
                v288 = *(_QWORD *)(v142 + 24);
              else
                v288 = **(_QWORD **)(v142 + 24);
              if ( *v141 )
                goto LABEL_232;
              v143 = (_QWORD *)sub_B91420((__int64)v141);
              v145 = sub_C94910(v315 + 168, v143, v144);
              v146 = *(_QWORD *)(v315 + 272);
              v147 = v289;
              v148 = v145;
              v150 = v149;
              if ( !v146 )
                goto LABEL_212;
              do
              {
                while ( 1 )
                {
                  v151 = *(_QWORD *)(v146 + 40);
                  v152 = v151;
                  if ( v150 <= v151 )
                    v152 = v150;
                  if ( v152 )
                  {
                    v153 = memcmp(*(const void **)(v146 + 32), v148, v152);
                    if ( v153 )
                      break;
                  }
                  if ( v150 != v151 && v150 > v151 )
                  {
                    v146 = *(_QWORD *)(v146 + 24);
                    goto LABEL_204;
                  }
LABEL_196:
                  v147 = (_QWORD *)v146;
                  v146 = *(_QWORD *)(v146 + 16);
                  if ( !v146 )
                    goto LABEL_205;
                }
                if ( v153 >= 0 )
                  goto LABEL_196;
                v146 = *(_QWORD *)(v146 + 24);
LABEL_204:
                ;
              }
              while ( v146 );
LABEL_205:
              if ( v289 == v147 )
                goto LABEL_212;
              v154 = v147[5];
              v155 = v150;
              if ( v154 <= v150 )
                v155 = v147[5];
              if ( v155 )
              {
                v156 = memcmp(v148, (const void *)v147[4], v155);
                if ( v156 )
                {
                  if ( v156 >= 0 )
                    goto LABEL_217;
LABEL_212:
                  v157 = (_QWORD *)sub_22077B0(72);
                  v158 = v147;
                  v157[4] = v148;
                  v147 = v157;
                  v157[5] = v150;
                  v157[6] = 0;
                  v157[7] = 0;
                  v157[8] = 0;
                  v159 = sub_9D5590((_QWORD *)(v315 + 256), v158, (__int64)(v157 + 4));
                  v161 = v160;
                  if ( v160 )
                  {
                    if ( v289 == v160 || v159 )
                    {
                      v162 = 1;
                    }
                    else
                    {
                      v240 = v160[5];
                      v241 = v150;
                      if ( v240 <= v150 )
                        v241 = v240;
                      if ( v241 && (v242 = memcmp(v148, (const void *)v161[4], v241)) != 0 )
                      {
                        v162 = v242 >> 31;
                      }
                      else
                      {
                        v162 = v240 > v150;
                        if ( v240 == v150 )
                          v162 = 0;
                      }
                    }
                    sub_220F040(v162, v147, v161, v289);
                    ++*(_QWORD *)(v315 + 296);
                  }
                  else
                  {
                    v237 = v147;
                    v147 = (_QWORD *)v159;
                    j_j___libc_free_0(v237, 72);
                  }
                  goto LABEL_217;
                }
              }
              if ( v154 != v150 && v154 > v150 )
                goto LABEL_212;
LABEL_217:
              sub_B2F930(&v355, v287);
              v163 = sub_B2F650(v355.m128i_i64[0], v355.m128i_i64[1]);
              if ( (_QWORD **)v355.m128i_i64[0] != &v356 )
                j_j___libc_free_0(v355.m128i_i64[0], (char *)v356 + 1);
              v342[0].m128i_i64[0] = v163;
              if ( *(_BYTE *)(v315 + 343) )
              {
                v355.m128i_i64[0] = 0;
              }
              else
              {
                v355.m128i_i64[1] = 0;
                v355.m128i_i64[0] = (__int64)byte_3F871B3;
              }
              v356 = 0;
              v357 = 0;
              v358 = 0;
              v164 = sub_9CA390((_QWORD *)v315, (unsigned __int64 *)v342, &v355);
              v165 = v357;
              v166 = v356;
              v167 = v164;
              v168 = (unsigned __int64)(v164 + 4);
              if ( v357 != (_BYTE *)v356 )
              {
                do
                {
                  if ( *v166 )
                    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v166 + 8LL))(*v166);
                  ++v166;
                }
                while ( v165 != v166 );
                v166 = v356;
              }
              if ( v166 )
                j_j___libc_free_0(v166, v358 - (_QWORD)v166);
              v167[5] = v287;
              v355.m128i_i64[1] = *(unsigned __int8 *)(v315 + 343) | v168 & 0xFFFFFFFFFFFFFFF8LL;
              v355.m128i_i64[0] = v288;
              v39 = v147[7];
              if ( v39 == v147[8] )
              {
                sub_9D2E70((const __m128i **)v147 + 6, (const __m128i *)v39, &v355);
              }
              else
              {
                if ( v39 )
                {
                  *(__m128i *)v39 = _mm_loadu_si128(&v355);
                  v39 = v147[7];
                }
                v39 += 16;
                v147[7] = v39;
              }
LABEL_232:
              if ( v286 == ++v303 )
              {
                v37 = (__m128i *)v287;
                if ( !*(_QWORD *)(v287 + 48) )
                  goto LABEL_234;
                goto LABEL_69;
              }
            }
          }
        }
      }
      if ( v37[3].m128i_i64[0] )
        goto LABEL_69;
LABEL_234:
      if ( (v37[2].m128i_i8[0] & 0xF) == 6
        || (unsigned __int8)sub_B2F6B0((__int64)v37)
        || (v37[2].m128i_i8[0] & 0xF) == 1 )
      {
LABEL_69:
        v43 = 0;
      }
      else
      {
        v43 = (v37[2].m128i_i8[1] & 3) != 2;
      }
      v44 = v37[5].m128i_i8[0] & 1;
      v47 = sub_B92110((__int64)v37);
      v48 = 0;
      if ( !v44 )
        v48 = v43;
      ++v364.m128i_i64[0];
      v306 = v48;
      if ( !(_DWORD)v365 )
      {
        v49 = HIDWORD(v365);
        if ( !HIDWORD(v365) )
          goto LABEL_78;
        v50 = (unsigned int)v366;
        if ( (unsigned int)v366 > 0x40 )
        {
          v39 = 8LL * (unsigned int)v366;
          sub_C7D6A0(v364.m128i_i64[1], v39, 8);
          v364.m128i_i64[1] = 0;
          v365 = 0;
          LODWORD(v366) = 0;
          goto LABEL_78;
        }
LABEL_75:
        v51 = (_QWORD *)v364.m128i_i64[1];
        v52 = v364.m128i_i64[1] + 8 * v50;
        if ( v364.m128i_i64[1] != v52 )
        {
          do
            *v51++ = -8;
          while ( (_QWORD *)v52 != v51 );
        }
        v365 = 0;
        goto LABEL_78;
      }
      v49 = (unsigned int)(4 * v365);
      v39 = 64;
      v50 = (unsigned int)v366;
      if ( (unsigned int)v49 < 0x40 )
        v49 = 64;
      if ( (unsigned int)v49 >= (unsigned int)v366 )
        goto LABEL_75;
      v122 = (_QWORD *)v364.m128i_i64[1];
      v123 = 8LL * (unsigned int)v366;
      if ( (_DWORD)v365 == 1 )
        break;
      _BitScanReverse(&v124, v365 - 1);
      v49 = 33 - (v124 ^ 0x1F);
      v125 = 1 << (33 - (v124 ^ 0x1F));
      if ( v125 < 64 )
        v125 = 64;
      if ( v125 != (_DWORD)v366 )
      {
        v126 = (4 * v125 / 3u + 1) | ((unsigned __int64)(4 * v125 / 3u + 1) >> 1);
        v127 = ((v126 | (v126 >> 2)) >> 4)
             | v126
             | (v126 >> 2)
             | ((((v126 | (v126 >> 2)) >> 4) | v126 | (v126 >> 2)) >> 8);
        v128 = (v127 | (v127 >> 16)) + 1;
        v129 = (v127 | (v127 >> 16)) + 1;
        v130 = 8 * v128;
        goto LABEL_173;
      }
      v365 = 0;
      v45 = v364.m128i_i64[1] + v123;
      do
      {
        if ( v122 )
          *v122 = -8;
        ++v122;
      }
      while ( (_QWORD *)v45 != v122 );
LABEL_78:
      v355 = (__m128i)(unsigned __int64)&v356;
      if ( (_DWORD)v368 )
      {
        v39 = (__int64)&v367;
        sub_D76D40((__int64)&v355, &v367, (unsigned int)v368, v49, v45, v46);
      }
      v53 = sub_22077B0(72);
      v57 = v53;
      if ( v53 )
      {
        *(_DWORD *)(v53 + 8) = 2;
        *(_QWORD *)(v53 + 16) = 0;
        *(_QWORD *)(v53 + 24) = 0;
        *(_QWORD *)v53 = &unk_49D9770;
        *(_QWORD *)(v53 + 32) = 0;
        *(_QWORD *)(v53 + 48) = 0;
        v58 = *(unsigned __int16 *)(v53 + 12);
        LOWORD(v58) = v58 & 0xF800;
        *(_WORD *)(v53 + 12) = v58
                             | (((_WORD)a5 << 9)
                              | ((_WORD)v292 << 8)
                              | (v301 << 6)
                              | v298
                              | (unsigned __int16)(16 * v290))
                             & 0x7FF;
        *(_QWORD *)(v53 + 40) = v53 + 56;
        if ( v355.m128i_i32[2] )
        {
          v39 = (__int64)&v355;
          sub_D76D40(v53 + 40, (char **)&v355, v58, v54, v55, v56);
        }
        v59 = *(_BYTE *)(v57 + 64);
        *(_QWORD *)(v57 + 56) = 0;
        *(_QWORD *)v57 = &unk_49D97D0;
        *(_BYTE *)(v57 + 64) = v59 & 0xE0 | ((8 * (v47 & 3)) | (4 * v44) | v43 | (2 * v306)) & 0x1F;
      }
      v60 = (__m128i *)v355.m128i_i64[0];
      if ( (_QWORD **)v355.m128i_i64[0] != &v356 )
        _libc_free(v355.m128i_i64[0], v39);
      if ( v301 )
      {
        sub_B2F930(&v355, (__int64)v37);
        v136 = sub_B2F650(v355.m128i_i64[0], v355.m128i_i64[1]);
        if ( (_QWORD **)v355.m128i_i64[0] != &v356 )
          j_j___libc_free_0(v355.m128i_i64[0], (char *)v356 + 1);
        v60 = &v355;
        v39 = (__int64)&v324;
        v342[0].m128i_i64[0] = v136;
        sub_D7AC80((__int64)&v355, (__int64)&v324, v342[0].m128i_i64);
      }
      if ( v295 )
        *(_BYTE *)(v57 + 12) |= 0x40u;
      if ( v344.m128i_i64[1] == v344.m128i_i64[0] )
        goto LABEL_90;
      v310 = v344.m128i_i64[1] - v344.m128i_i64[0];
      if ( v344.m128i_i64[1] - v344.m128i_i64[0] > 0x7FFFFFFFFFFFFFF0uLL )
        sub_4261EA(v60, v39, v344.m128i_i64[1] - v344.m128i_i64[0]);
      v190 = (__m128i *)sub_22077B0(v310);
      v191 = (const __m128i *)v344.m128i_i64[0];
      v192 = &v190->m128i_i8[v310];
      v193 = v190;
      if ( v344.m128i_i64[1] == v344.m128i_i64[0] )
      {
        v194 = v190;
      }
      else
      {
        v194 = (__m128i *)((char *)v190 + v344.m128i_i64[1] - v344.m128i_i64[0]);
        do
        {
          if ( v190 )
            *v190 = _mm_loadu_si128(v191);
          ++v190;
          ++v191;
        }
        while ( v190 != v194 );
      }
      v195 = (__m128i **)sub_22077B0(24);
      if ( !v195 )
      {
        v196 = *(_QWORD **)(v57 + 56);
        *(_QWORD *)(v57 + 56) = 0;
        if ( !v196 )
        {
LABEL_291:
          if ( v193 )
            j_j___libc_free_0(v193, v310);
          goto LABEL_90;
        }
LABEL_288:
        if ( *v196 )
          j_j___libc_free_0(*v196, v196[2] - *v196);
        j_j___libc_free_0(v196, 24);
        goto LABEL_291;
      }
      v195[2] = (__m128i *)v192;
      v196 = *(_QWORD **)(v57 + 56);
      *v195 = v193;
      v195[1] = v194;
      *(_QWORD *)(v57 + 56) = v195;
      if ( v196 )
      {
        v310 = 0;
        v193 = 0;
        goto LABEL_288;
      }
LABEL_90:
      v61 = v37;
      v355.m128i_i64[0] = v57;
      sub_D7A690(v315, (__int64)v37, v355.m128i_i64);
      if ( v355.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v355.m128i_i64[0] + 8LL))(v355.m128i_i64[0]);
      if ( v344.m128i_i64[0] )
      {
        v61 = (__m128i *)(v345 - v344.m128i_i64[0]);
        j_j___libc_free_0(v344.m128i_i64[0], v345 - v344.m128i_i64[0]);
      }
      if ( !BYTE4(v381) )
        _libc_free(v379, v61);
      if ( v367 != (char *)&v369 )
        _libc_free(v367, v61);
      v24 = 8LL * (unsigned int)v366;
      sub_C7D6A0(v364.m128i_i64[1], v24, 8);
LABEL_99:
      v312 = (_QWORD *)v312[1];
      if ( a2 + 1 == v312 )
      {
        v5 = v315;
        goto LABEL_101;
      }
    }
    v130 = 1024;
    v129 = 128;
LABEL_173:
    v302 = v129;
    sub_C7D6A0(v364.m128i_i64[1], v123, 8);
    v39 = 8;
    LODWORD(v366) = v302;
    v131 = (_QWORD *)sub_C7D670(v130, 8);
    v365 = 0;
    v364.m128i_i64[1] = (__int64)v131;
    for ( j = &v131[(unsigned int)v366]; j != v131; ++v131 )
    {
      if ( v131 )
        *v131 = -8;
    }
    goto LABEL_78;
  }
LABEL_101:
  if ( a2 + 5 != (_QWORD *)a2[6] )
  {
    v313 = v5;
    v62 = (_QWORD *)a2[6];
    do
    {
      v63 = 0;
      if ( v62 )
        v63 = v62 - 6;
      v307 = (_BYTE *)sub_B325F0((__int64)v63);
      if ( *v307 != 2 )
      {
        sub_B32650(v63, v24);
        v316 = 0;
        if ( v64 )
          v316 = (v63[32] & 0xFu) - 7 <= 1;
        v304 = (__int64)v307;
        v65 = sub_B2FE60(v63);
        v66 = v63[32] & 0xF;
        v67 = (v63[33] & 0x40) != 0;
        v308 = (v63[32] >> 4) & 3;
        v68 = sub_22077B0(72);
        v69 = v68;
        if ( v68 )
        {
          *(_DWORD *)(v68 + 8) = 0;
          v70 = *(_WORD *)(v68 + 12);
          *(_QWORD *)(v69 + 16) = 0;
          *(_QWORD *)(v69 + 24) = 0;
          *(_QWORD *)(v69 + 32) = 0;
          *(_QWORD *)(v69 + 48) = 0;
          *(_QWORD *)(v69 + 56) = 0;
          *(_QWORD *)(v69 + 64) = 0;
          *(_QWORD *)(v69 + 40) = v69 + 56;
          *(_WORD *)(v69 + 12) = v70 & 0xF800 | ((v316 << 6) | (16 * v308) | v66 | (v67 << 8) | (v65 << 9)) & 0x7FF;
          *(_QWORD *)v69 = &unk_49D9790;
        }
        v309 = v69;
        sub_B2F930(&v378, v304);
        v71 = sub_B2F650((__int64)v378, (__int64)v379);
        v72 = v309;
        v73 = v71;
        if ( v378 != &v380 )
        {
          j_j___libc_free_0(v378, v380 + 1);
          v72 = v309;
        }
        v74 = *(_QWORD **)(v313 + 16);
        if ( v74 )
        {
          v75 = v311;
          do
          {
            while ( 1 )
            {
              v76 = v74[2];
              v77 = v74[3];
              if ( v73 <= v74[4] )
                break;
              v74 = (_QWORD *)v74[3];
              if ( !v77 )
                goto LABEL_117;
            }
            v75 = v74;
            v74 = (_QWORD *)v74[2];
          }
          while ( v76 );
LABEL_117:
          v78 = 0;
          if ( v311 != v75 && v73 >= v75[4] )
            v78 = (unsigned __int64)(v75 + 4) & 0xFFFFFFFFFFFFFFF8LL;
        }
        else
        {
          v78 = 0;
        }
        v79 = v78 | *(unsigned __int8 *)(v313 + 343);
        v80 = **(_QWORD **)((v79 & 0xFFFFFFFFFFFFFFF8LL) + 0x18);
        *(_QWORD *)(v72 + 56) = v79;
        *(_QWORD *)(v72 + 64) = v80;
        if ( v316 )
        {
          v318 = v72;
          sub_B2F930(&v378, (__int64)v63);
          v133 = sub_B2F650((__int64)v378, (__int64)v379);
          v134 = v318;
          v135 = v133;
          if ( v378 != &v380 )
          {
            j_j___libc_free_0(v378, v380 + 1);
            v134 = v318;
          }
          v319 = v134;
          v364.m128i_i64[0] = v135;
          sub_D7AC80((__int64)&v378, (__int64)&v324, v364.m128i_i64);
          v72 = v319;
        }
        v24 = (__int64)v63;
        v378 = (__int64 *)v72;
        sub_D7A690(v313, (__int64)v63, (__int64 *)&v378);
        if ( v378 )
          (*(void (__fastcall **)(__int64 *))(*v378 + 8))(v378);
      }
      v62 = (_QWORD *)v62[1];
    }
    while ( a2 + 5 != v62 );
    v5 = v313;
  }
  for ( k = (_QWORD *)a2[8]; a2 + 7 != k; k = (_QWORD *)k[1] )
  {
    v82 = (__int64)(k - 7);
    v378 = (__int64 *)v5;
    if ( !k )
      v82 = 0;
    sub_B32A80(v82, (__int64)sub_D77010, (__int64)&v378);
  }
  v83 = v350;
  if ( v353 )
    v84 = &v350[HIDWORD(v351)];
  else
    v84 = &v350[(unsigned int)v351];
  if ( v350 != v84 )
  {
    while ( 1 )
    {
      v85 = *v83;
      v86 = v83;
      if ( (unsigned __int64)*v83 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v84 == ++v83 )
        goto LABEL_134;
    }
    if ( v84 != v83 )
    {
      do
      {
        sub_B2F930(&v378, v85);
        v234 = sub_B2F650((__int64)v378, (__int64)v379);
        if ( v378 != &v380 )
          j_j___libc_free_0(v378, v380 + 1);
        v235 = sub_BAEEF0(v5, v234);
        *(_BYTE *)(v235 + 12) |= 0x40u;
        v236 = v86 + 1;
        if ( v86 + 1 == v84 )
          break;
        v85 = *v236;
        for ( ++v86; (unsigned __int64)*v236 >= 0xFFFFFFFFFFFFFFFELL; v86 = v236 )
        {
          if ( v84 == ++v236 )
            goto LABEL_134;
          v85 = *v236;
        }
      }
      while ( v84 != v86 );
    }
  }
LABEL_134:
  sub_D76CA0(v5, (__int64)"llvm.used", 9);
  sub_D76CA0(v5, (__int64)"llvm.compiler.used", 18);
  sub_D76CA0(v5, (__int64)"llvm.global_ctors", 17);
  sub_D76CA0(v5, (__int64)"llvm.global_dtors", 17);
  v87 = "llvm.global.annotations";
  v88 = v5;
  sub_D76CA0(v5, (__int64)"llvm.global.annotations", 23);
  v92 = *(_QWORD *)(v5 + 24);
  if ( (_QWORD *)v92 == v311 )
    goto LABEL_338;
  v317 = v5;
  do
  {
    v106 = *(__int64 **)(v92 + 56);
    if ( *(__int64 **)(v92 + 64) == v106 )
      goto LABEL_145;
    v107 = *v106;
    if ( !v299 )
      goto LABEL_144;
    v108 = *(_QWORD **)(v107 + 40);
    v109 = 8LL * *(unsigned int *)(v107 + 48);
    v110 = &v108[(unsigned __int64)v109 / 8];
    v111 = v109 >> 3;
    v112 = v109 >> 5;
    if ( v112 )
    {
      v87 = v325;
      v113 = v327 - 1;
      v114 = &v108[4 * v112];
      while ( 1 )
      {
        v115 = *(_QWORD *)(*v108 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (_DWORD)v327 )
        {
          v93 = v113 & (((0xBF58476D1CE4E5B9LL * v115) >> 31) ^ (484763065 * v115));
          v94 = *(_QWORD *)&v325[8 * v93];
          if ( v115 == v94 )
            goto LABEL_137;
          v212 = 1;
          while ( v94 != -1 )
          {
            v93 = v113 & (v212 + v93);
            v94 = *(_QWORD *)&v325[8 * v93];
            if ( v115 == v94 )
              goto LABEL_137;
            ++v212;
          }
          v213 = v108 + 1;
          v214 = *(_QWORD *)(v108[1] & 0xFFFFFFFFFFFFFFF8LL);
          v215 = v113
               & (((0xBF58476D1CE4E5B9LL * v214) >> 31)
                ^ (484763065 * *(_DWORD *)(v108[1] & 0xFFFFFFFFFFFFFFF8LL)));
          v216 = *(_QWORD *)&v325[8 * v215];
          if ( v214 == v216 )
            goto LABEL_322;
          v217 = 1;
          while ( v216 != -1 )
          {
            v215 = v113 & (v217 + v215);
            v216 = *(_QWORD *)&v325[8 * v215];
            if ( v214 == v216 )
              goto LABEL_322;
            ++v217;
          }
          v213 = v108 + 2;
          v218 = *(_QWORD *)(v108[2] & 0xFFFFFFFFFFFFFFF8LL);
          v219 = v113
               & (((0xBF58476D1CE4E5B9LL * v218) >> 31)
                ^ (484763065 * *(_DWORD *)(v108[2] & 0xFFFFFFFFFFFFFFF8LL)));
          v220 = *(_QWORD *)&v325[8 * v219];
          if ( v220 == v218 )
            goto LABEL_322;
          v221 = 1;
          while ( v220 != -1 )
          {
            v219 = v113 & (v221 + v219);
            v220 = *(_QWORD *)&v325[8 * v219];
            if ( v220 == v218 )
              goto LABEL_322;
            ++v221;
          }
          v213 = v108 + 3;
          v222 = *(_QWORD *)(v108[3] & 0xFFFFFFFFFFFFFFF8LL);
          v223 = v113
               & (((0xBF58476D1CE4E5B9LL * v222) >> 31)
                ^ (484763065 * *(_DWORD *)(v108[3] & 0xFFFFFFFFFFFFFFF8LL)));
          v224 = *(_QWORD *)&v325[8 * v223];
          if ( v222 == v224 )
          {
LABEL_322:
            v108 = v213;
            goto LABEL_137;
          }
          v225 = 1;
          while ( v224 != -1 )
          {
            v223 = v113 & (v225 + v223);
            v224 = *(_QWORD *)&v325[8 * v223];
            if ( v224 == v222 )
              goto LABEL_322;
            ++v225;
          }
        }
        v108 += 4;
        if ( v108 == v114 )
        {
          v111 = v110 - v108;
          break;
        }
      }
    }
    switch ( v111 )
    {
      case 2LL:
        v87 = v325;
        v116 = v327;
        break;
      case 3LL:
        v87 = v325;
        v263 = *(_QWORD *)(*v108 & 0xFFFFFFFFFFFFFFF8LL);
        v116 = v327;
        if ( (_DWORD)v327 )
        {
          v264 = (v327 - 1) & (((0xBF58476D1CE4E5B9LL * v263) >> 31) ^ (484763065 * v263));
          v265 = *(_QWORD *)&v325[8 * v264];
          if ( v265 == v263 )
            goto LABEL_137;
          v266 = 1;
          while ( v265 != -1 )
          {
            v264 = (v327 - 1) & (v266 + v264);
            v265 = *(_QWORD *)&v325[8 * v264];
            if ( v263 == v265 )
              goto LABEL_137;
            ++v266;
          }
        }
        ++v108;
        break;
      case 1LL:
        v87 = v325;
        v116 = v327;
        goto LABEL_157;
      default:
        goto LABEL_138;
    }
    v259 = *(_QWORD *)(*v108 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v116 )
      goto LABEL_419;
    v260 = (v116 - 1) & (((0xBF58476D1CE4E5B9LL * v259) >> 31) ^ (484763065 * v259));
    v261 = *(_QWORD *)&v87[8 * v260];
    if ( v261 != v259 )
    {
      v262 = 1;
      while ( v261 != -1 )
      {
        v260 = (v116 - 1) & (v262 + v260);
        v261 = *(_QWORD *)&v87[8 * v260];
        if ( v259 == v261 )
          goto LABEL_137;
        ++v262;
      }
LABEL_419:
      ++v108;
LABEL_157:
      v117 = *(_QWORD *)(*v108 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v116 )
      {
        v118 = v116 - 1;
        v119 = 1;
        v120 = v118 & (((0xBF58476D1CE4E5B9LL * v117) >> 31) ^ (484763065 * v117));
        v121 = *(_QWORD *)&v87[8 * v120];
        if ( v117 == v121 )
          goto LABEL_137;
        while ( v121 != -1 )
        {
          v120 = v118 & (v119 + v120);
          v121 = *(_QWORD *)&v87[8 * v120];
          if ( v117 == v121 )
            goto LABEL_137;
          ++v119;
        }
      }
      goto LABEL_138;
    }
LABEL_137:
    if ( v110 != v108 )
      goto LABEL_144;
LABEL_138:
    v95 = *(_DWORD *)(v107 + 8);
    if ( v95 != 1 )
      goto LABEL_145;
    v96 = *(char **)(v107 + 64);
    v97 = 16LL * *(unsigned int *)(v107 + 72);
    v98 = &v96[v97];
    v99 = v97 >> 4;
    v100 = v97 >> 6;
    v314 = v98;
    if ( v100 )
    {
      v87 = v325;
      v101 = &v96[64 * v100];
      v102 = v327 - 1;
      while ( 1 )
      {
        v103 = *(_QWORD *)(*(_QWORD *)v96 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (_DWORD)v327 )
        {
          v104 = v102 & (((0xBF58476D1CE4E5B9LL * v103) >> 31) ^ (484763065 * v103));
          v105 = *(_QWORD *)&v325[8 * v104];
          if ( v105 == v103 )
            goto LABEL_143;
          v230 = 1;
          while ( v105 != -1 )
          {
            v104 = v102 & (v230 + v104);
            v105 = *(_QWORD *)&v325[8 * v104];
            if ( v103 == v105 )
              goto LABEL_143;
            ++v230;
          }
          v198 = v96 + 16;
          v231 = *(_QWORD *)(*((_QWORD *)v96 + 2) & 0xFFFFFFFFFFFFFFF8LL);
          v232 = v102
               & (((0xBF58476D1CE4E5B9LL * v231) >> 31)
                ^ (484763065 * *(_DWORD *)(*((_QWORD *)v96 + 2) & 0xFFFFFFFFFFFFFFF8LL)));
          v233 = *(_QWORD *)&v325[8 * v232];
          if ( v233 == v231 )
            goto LABEL_349;
          v197 = 1;
          while ( v233 != -1 )
          {
            v232 = v102 & (v197 + v232);
            v233 = *(_QWORD *)&v325[8 * v232];
            if ( v233 == v231 )
              goto LABEL_349;
            ++v197;
          }
          v198 = v96 + 32;
          v199 = *(_QWORD *)(*((_QWORD *)v96 + 4) & 0xFFFFFFFFFFFFFFF8LL);
          v200 = v102
               & (((0xBF58476D1CE4E5B9LL * v199) >> 31)
                ^ (484763065 * *(_DWORD *)(*((_QWORD *)v96 + 4) & 0xFFFFFFFFFFFFFFF8LL)));
          v201 = *(_QWORD *)&v325[8 * v200];
          if ( v201 == v199 )
            goto LABEL_349;
          v202 = 1;
          while ( v201 != -1 )
          {
            v200 = v102 & (v202 + v200);
            v201 = *(_QWORD *)&v325[8 * v200];
            if ( v201 == v199 )
              goto LABEL_349;
            ++v202;
          }
          v198 = v96 + 48;
          v203 = *(_QWORD *)(*((_QWORD *)v96 + 6) & 0xFFFFFFFFFFFFFFF8LL);
          v204 = v102
               & (((0xBF58476D1CE4E5B9LL * v203) >> 31)
                ^ (484763065 * *(_DWORD *)(*((_QWORD *)v96 + 6) & 0xFFFFFFFFFFFFFFF8LL)));
          v205 = *(_QWORD *)&v325[8 * v204];
          if ( v205 == v203 )
          {
LABEL_349:
            v96 = v198;
            goto LABEL_143;
          }
          v206 = 1;
          while ( v205 != -1 )
          {
            v204 = v102 & (v206 + v204);
            v205 = *(_QWORD *)&v325[8 * v204];
            if ( v205 == v203 )
              goto LABEL_349;
            ++v206;
          }
        }
        v96 += 64;
        if ( v101 == v96 )
        {
          v99 = (v314 - v96) >> 4;
          break;
        }
      }
    }
    if ( v99 == 2 )
    {
      v87 = v325;
      v207 = v327;
LABEL_399:
      v247 = *(_QWORD *)(*(_QWORD *)v96 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v207 )
      {
        v248 = (v207 - 1) & (((0xBF58476D1CE4E5B9LL * v247) >> 31) ^ (484763065 * v247));
        v249 = *(_QWORD *)&v87[8 * v248];
        if ( v249 == v247 )
        {
LABEL_143:
          if ( v314 == v96 )
            goto LABEL_145;
LABEL_144:
          *(_BYTE *)(v107 + 12) |= 0x40u;
          goto LABEL_145;
        }
        v250 = 1;
        while ( v249 != -1 )
        {
          v248 = (v207 - 1) & (v250 + v248);
          v249 = *(_QWORD *)&v87[8 * v248];
          if ( v247 == v249 )
            goto LABEL_143;
          ++v250;
        }
      }
      v96 += 16;
      goto LABEL_312;
    }
    if ( v99 == 3 )
    {
      v87 = v325;
      v243 = *(_QWORD *)(*(_QWORD *)v96 & 0xFFFFFFFFFFFFFFF8LL);
      v207 = v327;
      if ( (_DWORD)v327 )
      {
        v244 = (v327 - 1) & (((0xBF58476D1CE4E5B9LL * v243) >> 31) ^ (484763065 * v243));
        v245 = *(_QWORD *)&v325[8 * v244];
        if ( v243 == v245 )
          goto LABEL_143;
        v246 = 1;
        while ( v245 != -1 )
        {
          v244 = (v327 - 1) & (v246 + v244);
          v245 = *(_QWORD *)&v325[8 * v244];
          if ( v243 == v245 )
            goto LABEL_143;
          ++v246;
        }
      }
      v96 += 16;
      goto LABEL_399;
    }
    if ( v99 != 1 )
      goto LABEL_145;
    v87 = v325;
    v207 = v327;
LABEL_312:
    v208 = *(_QWORD *)(*(_QWORD *)v96 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v207 )
    {
      v209 = v207 - 1;
      v210 = v209 & (((0xBF58476D1CE4E5B9LL * v208) >> 31) ^ (484763065 * v208));
      v211 = *(_QWORD *)&v87[8 * v210];
      if ( v211 == v208 )
        goto LABEL_143;
      while ( v211 != -1 )
      {
        v210 = v209 & (v95 + v210);
        v211 = *(_QWORD *)&v87[8 * v210];
        if ( v208 == v211 )
          goto LABEL_143;
        ++v95;
      }
    }
LABEL_145:
    v88 = v92;
    v92 = sub_220EEE0(v92);
  }
  while ( (_QWORD *)v92 != v311 );
  v5 = v317;
LABEL_338:
  v227 = qword_4F87990;
  if ( qword_4F87990 )
  {
    v322 = 0;
    v323 = sub_2241E40(v88, v87, v89, v90, v91);
    sub_CB7060((__int64)&v378, (_BYTE *)qword_4F87988, v227, (__int64)&v322, 1u);
    if ( v322 )
    {
      v355.m128i_i64[0] = (__int64)"\n";
      LOWORD(v358) = 259;
      (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v323 + 32LL))(v331);
      v338[0].m128i_i64[0] = (__int64)": ";
      v335[0].m128i_i64[0] = (__int64)&qword_4F87988;
      v332[0].m128i_i64[0] = (__int64)"Failed to open dot file ";
      v343 = 260;
      v342[0].m128i_i64[0] = (__int64)v331;
      v340 = 1;
      v339 = 3;
      v336 = 260;
      v334 = 1;
      v333 = 3;
      sub_9C6370(v337, v332, v335, v274, v275, v276);
      sub_9C6370(v341, v337, v338, v277, v278, v279);
      sub_9C6370(&v344, v341, v342, v280, v281, v282);
      sub_9C6370(&v364, &v344, &v355, v283, v284, v285);
      sub_C64D30((__int64)&v364, 1u);
    }
    v364 = 0u;
    v365 = 0;
    v366 = 0;
    sub_BB0D30(v5, (__int64)&v378, (__int64)&v364);
    v87 = (const char *)(8LL * (unsigned int)v366);
    sub_C7D6A0(v364.m128i_i64[1], (__int64)v87, 8);
    sub_CB5B00((int *)&v378, (__int64)v87);
  }
  if ( v328 != (__int64 *)v330 )
    _libc_free(v328, v87);
  v228 = 8LL * (unsigned int)v327;
  sub_C7D6A0((__int64)v325, v228, 8);
  if ( v346 != (__int64 *)v348 )
    _libc_free(v346, v228);
  if ( !v353 )
    _libc_free(v350, v228);
  return v5;
}
