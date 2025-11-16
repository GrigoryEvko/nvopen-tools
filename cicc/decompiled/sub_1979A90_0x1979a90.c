// Function: sub_1979A90
// Address: 0x1979a90
//
__int64 __fastcall sub_1979A90(
        __int64 a1,
        __m128i a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        __m128i a8,
        __m128 a9)
{
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
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  char **v29; // r14
  bool v30; // al
  int v31; // r9d
  __int64 v32; // rax
  __int64 *v33; // r13
  _QWORD *v34; // rbx
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rcx
  _BYTE *v38; // r8
  __int64 v39; // rdi
  _QWORD *v40; // r14
  __int64 **v41; // rbx
  __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rdi
  unsigned int v45; // eax
  __int64 v46; // rax
  unsigned __int64 *v47; // rdx
  unsigned __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdx
  _QWORD *v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r12
  __int64 v59; // rbx
  __int64 v60; // r15
  _QWORD *v61; // rax
  _QWORD *v62; // rbx
  int v63; // r13d
  _QWORD *v64; // r14
  _QWORD *v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rdx
  _BYTE *v68; // rdi
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // r12
  __int64 v72; // r13
  __int64 v73; // rbx
  __int64 v74; // r14
  char v75; // al
  char *v76; // rbx
  char *v77; // r12
  __int64 *v78; // rdi
  __int64 v79; // rax
  __int64 *v80; // r15
  __int64 v81; // rdx
  __int64 v82; // rcx
  unsigned int v83; // r13d
  unsigned int v84; // ebx
  _BYTE *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // rdi
  __int64 v90; // rax
  bool v91; // al
  _BYTE *v92; // rsi
  __int64 v93; // r13
  signed __int64 v94; // rax
  char *v95; // rbx
  char *v96; // rax
  char *v97; // rdx
  char *v98; // rcx
  __int64 v99; // r13
  char *v100; // rbx
  __int64 (__fastcall *v101)(__int64); // rax
  __int64 v102; // rax
  _QWORD *v103; // rcx
  _BYTE *v104; // rdi
  __int64 v106; // rcx
  int v107; // eax
  __int64 v108; // r15
  _BYTE *v109; // rdx
  __int64 v110; // rbx
  __int64 v111; // rdi
  __int64 *v112; // r14
  __int64 v113; // rcx
  unsigned __int8 v114; // al
  __int64 v115; // rsi
  __int64 v116; // rsi
  unsigned __int64 v117; // rax
  __int64 v118; // rbx
  __int64 v119; // r11
  __int64 v120; // r9
  __int64 v121; // rcx
  int v122; // edi
  char v123; // r8
  unsigned int v124; // eax
  char v125; // dl
  __int64 v126; // rax
  __int64 v127; // rbx
  _QWORD *v128; // r15
  _QWORD *v129; // rbx
  _QWORD *v130; // r12
  _QWORD *v131; // rdi
  __int64 v132; // rdx
  __m128i *v133; // rdi
  __int64 v134; // rdx
  __int64 v135; // rax
  __m128i *v136; // r15
  __int64 v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rax
  __int64 v140; // rdx
  __int64 v141; // rax
  char v142; // r9
  __int64 v143; // rdi
  __int64 v144; // rax
  __int64 *v145; // r14
  __int64 v146; // rax
  __int64 v147; // r12
  _QWORD *v148; // r15
  _QWORD *v149; // rbx
  _QWORD *v150; // r12
  _QWORD *v151; // rdi
  _QWORD *v152; // r12
  _QWORD *v153; // rdi
  __int64 v154; // rax
  __int64 v155; // rax
  _QWORD *v156; // rbx
  _QWORD *v157; // rdi
  unsigned __int64 v158; // rax
  unsigned __int64 v159; // rax
  unsigned __int64 v160; // r12
  __int64 v161; // rax
  __int64 v162; // rdx
  __int64 v163; // rdx
  _BYTE *v164; // r13
  __int64 v165; // rbx
  int v166; // r14d
  __int64 v167; // r12
  char **v168; // rsi
  __int64 v169; // rdi
  int v170; // r13d
  __int64 v171; // r12
  unsigned __int64 *v172; // rbx
  __int64 v173; // rax
  __int64 v174; // rax
  __int64 v175; // r12
  __int64 v176; // rax
  __int64 v177; // rdi
  __int64 v178; // rbx
  __int64 v179; // rax
  __int64 v180; // r12
  __int64 v181; // rbx
  __int64 v182; // r13
  unsigned __int64 v183; // rax
  __int64 v184; // r14
  unsigned int v185; // r12d
  int v186; // eax
  __int64 v187; // r15
  int v188; // ebx
  __int64 v189; // rax
  __int64 *v190; // r14
  __int64 v191; // rax
  __int64 v192; // rbx
  __m128i v193; // xmm4
  __m128i v194; // xmm5
  _QWORD *v195; // r15
  _QWORD *v196; // rbx
  _QWORD *v197; // rdi
  int v198; // eax
  bool v199; // zf
  char v200; // al
  __int64 (__fastcall *v201)(__int64); // rax
  __int64 v202; // rax
  __int64 v203; // rax
  _QWORD *v204; // rdi
  __int64 v205; // r14
  __int64 v206; // rbx
  __int64 v207; // r13
  _QWORD *v208; // rax
  _QWORD *v209; // rdi
  __int64 v210; // rbx
  __int64 v211; // r12
  __int64 v212; // rax
  __int64 v213; // rax
  __int64 v214; // rdx
  __int64 v215; // rbx
  __int64 *v216; // r14
  __int64 v217; // rax
  __int64 v218; // rbx
  _QWORD *v219; // r15
  _QWORD *v220; // rbx
  _QWORD *v221; // rdi
  __int64 v222; // rax
  __int64 v223; // r12
  __int64 v224; // r15
  unsigned int v225; // eax
  __int64 v226; // rbx
  __int64 v227; // r14
  __int64 i; // r15
  __int64 v229; // rax
  _QWORD *v230; // rax
  int v231; // r14d
  char *v232; // rax
  char v233; // cl
  __int64 v234; // rax
  __int64 v235; // r13
  __int64 v236; // r13
  __int64 v237; // r14
  _QWORD *v238; // r15
  _QWORD *v239; // r14
  _QWORD *v240; // rdi
  _QWORD *v241; // rbx
  _QWORD *v242; // rdi
  __int64 v243; // rax
  __int64 v244; // r13
  _QWORD *v245; // r14
  double v246; // xmm4_8
  double v247; // xmm5_8
  _QWORD *v248; // rbx
  _QWORD *v249; // r12
  _QWORD *v250; // rdi
  __int64 v251; // rax
  __int64 v252; // rdx
  __int64 v253; // rcx
  _QWORD *v254; // rdx
  __int64 v255; // rcx
  char *v256; // rdx
  unsigned __int64 v257; // rax
  __int64 v258; // rdi
  __int64 v259; // rax
  __int64 v260; // rdi
  __int64 v261; // rcx
  _BYTE *v262; // rdx
  __int64 v263; // rdx
  _QWORD *v264; // rbx
  _QWORD *v265; // rdi
  __int64 v266; // r13
  __int64 v267; // r12
  unsigned __int64 v268; // r14
  unsigned __int64 v269; // rax
  __int64 v270; // rax
  __int64 v271; // r14
  __int64 v272; // r12
  __int64 v273; // rbx
  __int64 v274; // r13
  __int64 v275; // rdx
  _QWORD *v276; // rax
  __int64 v277; // r12
  __int64 v278; // rsi
  __int64 v279; // rax
  __m128i v280; // kr00_16
  __int64 v281; // r12
  __int64 v282; // rax
  __int64 v283; // rax
  __int64 v284; // rax
  __int64 v285; // rax
  __int64 v286; // rax
  int v287; // edx
  __int64 v288; // r13
  unsigned int v289; // eax
  __int64 v290; // rax
  __int64 v291; // rax
  __int64 v292; // r14
  __int64 v293; // rax
  __int64 j; // r14
  __int64 v295; // rax
  _QWORD *v296; // rdi
  __int64 v297; // rax
  __int64 v298; // rax
  __int64 v299; // [rsp+8h] [rbp-9B8h]
  __int64 v300; // [rsp+18h] [rbp-9A8h]
  __int64 v301; // [rsp+40h] [rbp-980h]
  int v302; // [rsp+4Ch] [rbp-974h]
  __int64 *v303; // [rsp+58h] [rbp-968h]
  __int64 v304; // [rsp+70h] [rbp-950h]
  __int64 v305; // [rsp+80h] [rbp-940h]
  int v306; // [rsp+88h] [rbp-938h]
  _QWORD *v307; // [rsp+88h] [rbp-938h]
  unsigned __int8 v308; // [rsp+A0h] [rbp-920h]
  __int64 *v310; // [rsp+B0h] [rbp-910h]
  int v311; // [rsp+B0h] [rbp-910h]
  unsigned int v312; // [rsp+B8h] [rbp-908h]
  __int64 v313; // [rsp+B8h] [rbp-908h]
  char **v314; // [rsp+B8h] [rbp-908h]
  __int64 v315; // [rsp+B8h] [rbp-908h]
  __int64 *v316; // [rsp+C0h] [rbp-900h]
  __int64 *v317; // [rsp+C0h] [rbp-900h]
  __int64 v318; // [rsp+C0h] [rbp-900h]
  __int64 *v319; // [rsp+C0h] [rbp-900h]
  char v320; // [rsp+C0h] [rbp-900h]
  char v321; // [rsp+C0h] [rbp-900h]
  __int64 *v322; // [rsp+C8h] [rbp-8F8h]
  __int64 v323; // [rsp+C8h] [rbp-8F8h]
  __int64 *v324; // [rsp+C8h] [rbp-8F8h]
  unsigned int v325; // [rsp+C8h] [rbp-8F8h]
  _QWORD *v326; // [rsp+C8h] [rbp-8F8h]
  __int64 v327; // [rsp+C8h] [rbp-8F8h]
  __int64 v328; // [rsp+C8h] [rbp-8F8h]
  __int64 v329; // [rsp+C8h] [rbp-8F8h]
  _QWORD *v330; // [rsp+D0h] [rbp-8F0h]
  __int64 v331; // [rsp+D0h] [rbp-8F0h]
  __int64 *v332; // [rsp+D0h] [rbp-8F0h]
  __int64 v333; // [rsp+D0h] [rbp-8F0h]
  __int64 v334; // [rsp+D0h] [rbp-8F0h]
  char v335; // [rsp+D0h] [rbp-8F0h]
  __int64 *v336; // [rsp+D0h] [rbp-8F0h]
  __int64 v337; // [rsp+D8h] [rbp-8E8h]
  __int64 v338; // [rsp+D8h] [rbp-8E8h]
  __int64 *v339; // [rsp+D8h] [rbp-8E8h]
  unsigned __int64 v340; // [rsp+D8h] [rbp-8E8h]
  _QWORD *v341; // [rsp+D8h] [rbp-8E8h]
  _QWORD *v342; // [rsp+D8h] [rbp-8E8h]
  char *v343; // [rsp+E0h] [rbp-8E0h] BYREF
  char *v344; // [rsp+E8h] [rbp-8D8h]
  char *v345; // [rsp+F0h] [rbp-8D0h]
  __m128i v346[2]; // [rsp+100h] [rbp-8C0h] BYREF
  __m128i v347; // [rsp+120h] [rbp-8A0h] BYREF
  _BYTE v348[16]; // [rsp+130h] [rbp-890h] BYREF
  void (__fastcall *v349)(_BYTE *, _BYTE *, __int64); // [rsp+140h] [rbp-880h]
  unsigned __int8 (__fastcall *v350)(_BYTE *, __int64); // [rsp+148h] [rbp-878h]
  __int64 v351; // [rsp+150h] [rbp-870h] BYREF
  __int64 v352; // [rsp+158h] [rbp-868h]
  __int64 v353; // [rsp+160h] [rbp-860h]
  __int64 v354; // [rsp+168h] [rbp-858h]
  __int64 v355; // [rsp+170h] [rbp-850h]
  unsigned __int8 v356; // [rsp+178h] [rbp-848h]
  __int64 *v357; // [rsp+180h] [rbp-840h]
  char v358; // [rsp+188h] [rbp-838h]
  _BYTE *v359; // [rsp+190h] [rbp-830h] BYREF
  __int64 v360; // [rsp+198h] [rbp-828h]
  _BYTE v361[64]; // [rsp+1A0h] [rbp-820h] BYREF
  _QWORD *v362; // [rsp+1E0h] [rbp-7E0h] BYREF
  __int64 v363; // [rsp+1E8h] [rbp-7D8h]
  _BYTE v364[64]; // [rsp+1F0h] [rbp-7D0h] BYREF
  unsigned __int64 v365[2]; // [rsp+230h] [rbp-790h] BYREF
  _BYTE v366[64]; // [rsp+240h] [rbp-780h] BYREF
  __int64 v367[2]; // [rsp+280h] [rbp-740h] BYREF
  _QWORD v368[2]; // [rsp+290h] [rbp-730h] BYREF
  __int64 *v369; // [rsp+2A0h] [rbp-720h]
  __int64 v370; // [rsp+2A8h] [rbp-718h]
  __int64 v371; // [rsp+2B0h] [rbp-710h] BYREF
  __m128i v372; // [rsp+2E0h] [rbp-6E0h] BYREF
  _QWORD v373[2]; // [rsp+2F0h] [rbp-6D0h] BYREF
  _QWORD *v374; // [rsp+300h] [rbp-6C0h]
  unsigned __int8 (__fastcall *v375)(_BYTE *, __int64); // [rsp+308h] [rbp-6B8h]
  _QWORD v376[2]; // [rsp+310h] [rbp-6B0h] BYREF
  _BYTE v377[16]; // [rsp+320h] [rbp-6A0h] BYREF
  _QWORD *v378; // [rsp+330h] [rbp-690h]
  __int64 v379; // [rsp+338h] [rbp-688h]
  void *src; // [rsp+340h] [rbp-680h] BYREF
  _BYTE *v381; // [rsp+348h] [rbp-678h]
  _BYTE *v382; // [rsp+350h] [rbp-670h]
  __m128 v383; // [rsp+358h] [rbp-668h]
  __int64 v384; // [rsp+368h] [rbp-658h]
  __int64 v385; // [rsp+370h] [rbp-650h]
  __m128 v386; // [rsp+378h] [rbp-648h]
  _BYTE *v387; // [rsp+388h] [rbp-638h]
  char v388; // [rsp+390h] [rbp-630h]
  _BYTE *v389; // [rsp+398h] [rbp-628h] BYREF
  __int64 v390; // [rsp+3A0h] [rbp-620h]
  _BYTE v391[352]; // [rsp+3A8h] [rbp-618h] BYREF
  char v392; // [rsp+508h] [rbp-4B8h]
  int v393; // [rsp+50Ch] [rbp-4B4h]
  __int64 v394; // [rsp+510h] [rbp-4B0h]
  unsigned __int64 v395; // [rsp+520h] [rbp-4A0h] BYREF
  __int64 v396; // [rsp+528h] [rbp-498h]
  __int64 v397; // [rsp+530h] [rbp-490h] BYREF
  __m128i v398; // [rsp+538h] [rbp-488h] BYREF
  __int64 v399; // [rsp+548h] [rbp-478h]
  __int64 v400; // [rsp+550h] [rbp-470h]
  __m128i v401; // [rsp+558h] [rbp-468h] BYREF
  _BYTE *v402; // [rsp+568h] [rbp-458h]
  _BYTE *v403; // [rsp+570h] [rbp-450h]
  __int64 v404; // [rsp+578h] [rbp-448h] BYREF
  unsigned int v405; // [rsp+580h] [rbp-440h]
  _BYTE v406[352]; // [rsp+588h] [rbp-438h] BYREF
  char v407; // [rsp+6E8h] [rbp-2D8h]
  int v408; // [rsp+6ECh] [rbp-2D4h]
  __int64 v409; // [rsp+6F0h] [rbp-2D0h]
  _BYTE *v410; // [rsp+700h] [rbp-2C0h]
  __int64 v411; // [rsp+708h] [rbp-2B8h]
  _BYTE v412[688]; // [rsp+710h] [rbp-2B0h] BYREF

  v9 = *(__int64 **)(a1 + 8);
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
    goto LABEL_580;
  while ( *(_UNKNOWN **)v10 != &unk_4F9A488 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_580;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F9A488);
  v13 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 160) = *(_QWORD *)(v12 + 160);
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
    goto LABEL_580;
  while ( *(_UNKNOWN **)v14 != &unk_4F9920C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_580;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9920C);
  v17 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 168) = v16 + 160;
  v18 = *v17;
  v19 = v17[1];
  if ( v18 == v19 )
    goto LABEL_580;
  while ( *(_UNKNOWN **)v18 != &unk_4F98D2D )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_580;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F98D2D);
  v21 = sub_13A6090(v20);
  v22 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 176) = v21;
  v23 = *v22;
  v24 = v22[1];
  if ( v23 == v24 )
    goto LABEL_580;
  while ( *(_UNKNOWN **)v23 != &unk_4F9E06C )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_580;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F9E06C);
  v26 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 184) = v25 + 160;
  v27 = *v26;
  v28 = v26[1];
  if ( v27 == v28 )
LABEL_580:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F99CB0 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_580;
  }
  v29 = (char **)&v395;
  *(_QWORD *)(a1 + 200) = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(
                                        *(_QWORD *)(v27 + 8),
                                        &unk_4F99CB0)
                                    + 160);
  v30 = sub_1636850(a1, (__int64)&unk_4FB65F4);
  v411 = 0x800000000LL;
  *(_BYTE *)(a1 + 192) = v30;
  v410 = v412;
  v32 = *(_QWORD *)(a1 + 168);
  v33 = &v397;
  v34 = *(_QWORD **)(v32 + 32);
  v330 = *(_QWORD **)(v32 + 40);
  if ( v34 == v330 )
    return 1;
  do
  {
    v35 = *v34;
    v395 = (unsigned __int64)v33;
    v396 = 0x800000000LL;
    v36 = *(_QWORD *)(v35 + 16);
    v37 = *(_QWORD *)(v35 + 8);
    LODWORD(v38) = v35 + 8;
    if ( v36 != v37 )
    {
      v39 = (__int64)v29;
      v40 = v34;
      v41 = (__int64 **)(v35 + 8);
      while ( v36 - v37 == 8 )
      {
        v43 = (unsigned int)v396;
        if ( (unsigned int)v396 >= HIDWORD(v396) )
        {
          sub_16CD150(v39, v33, 0, 8, (int)v38, v31);
          v43 = (unsigned int)v396;
        }
        *(_QWORD *)(v395 + 8 * v43) = v35;
        v42 = (unsigned int)(v396 + 1);
        LODWORD(v396) = v396 + 1;
        v35 = **v41;
        v36 = *(_QWORD *)(v35 + 16);
        v37 = *(_QWORD *)(v35 + 8);
        v41 = (__int64 **)(v35 + 8);
        if ( v36 == v37 )
        {
          v34 = v40;
          v29 = (char **)v39;
          if ( (unsigned int)v42 >= HIDWORD(v396) )
          {
            sub_16CD150(v39, v33, 0, 8, (int)v38, v31);
            v42 = (unsigned int)v396;
          }
          v53 = (__int64 *)(v395 + 8 * v42);
          goto LABEL_44;
        }
      }
      v34 = v40;
      v29 = (char **)v39;
      v44 = v395;
      if ( (__int64 *)v395 == v33 )
        goto LABEL_30;
LABEL_29:
      _libc_free(v44);
      goto LABEL_30;
    }
    v53 = v33;
LABEL_44:
    *v53 = v35;
    v54 = (unsigned int)v411;
    LODWORD(v396) = v396 + 1;
    if ( (unsigned int)v411 >= HIDWORD(v411) )
    {
      v325 = v411;
      v158 = (((HIDWORD(v411) + 2LL) | (((unsigned __int64)HIDWORD(v411) + 2) >> 1)) >> 2)
           | (HIDWORD(v411) + 2LL)
           | (((unsigned __int64)HIDWORD(v411) + 2) >> 1);
      v159 = (v158 >> 4) | v158;
      v160 = ((v159 >> 8) | v159 | (((v159 >> 8) | v159) >> 16) | (((v159 >> 8) | v159) >> 32)) + 1;
      if ( v160 > 0xFFFFFFFF )
        v160 = 0xFFFFFFFFLL;
      v161 = malloc(80 * v160);
      v162 = v325;
      v337 = v161;
      if ( !v161 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v162 = (unsigned int)v411;
      }
      v38 = &v410[80 * v162];
      if ( v410 != v38 )
      {
        v163 = v337;
        v326 = v34;
        v319 = v33;
        v164 = v38;
        v314 = v29;
        v165 = v337;
        v166 = v160;
        v167 = (__int64)v410;
        do
        {
          while ( 1 )
          {
            if ( v165 )
            {
              *(_DWORD *)(v165 + 8) = 0;
              *(_QWORD *)v165 = v165 + 16;
              *(_DWORD *)(v165 + 12) = 8;
              if ( *(_DWORD *)(v167 + 8) )
                break;
            }
            v167 += 80;
            v165 += 80;
            if ( v164 == (_BYTE *)v167 )
              goto LABEL_301;
          }
          v168 = (char **)v167;
          v169 = v165;
          v167 += 80;
          v165 += 80;
          sub_19742B0(v169, v168, v163, v37, (int)v38, v31);
        }
        while ( v164 != (_BYTE *)v167 );
LABEL_301:
        v38 = v410;
        LODWORD(v160) = v166;
        v34 = v326;
        v33 = v319;
        v29 = v314;
        if ( &v410[80 * (unsigned int)v411] != v410 )
        {
          v170 = v160;
          v171 = (__int64)v410;
          v172 = (unsigned __int64 *)&v410[80 * (unsigned int)v411];
          do
          {
            v172 -= 10;
            if ( (unsigned __int64 *)*v172 != v172 + 2 )
              _libc_free(*v172);
          }
          while ( v172 != (unsigned __int64 *)v171 );
          LODWORD(v160) = v170;
          v34 = v326;
          v33 = v319;
          v29 = v314;
          v38 = v410;
        }
      }
      if ( v38 != v412 )
        _libc_free((unsigned __int64)v38);
      HIDWORD(v411) = v160;
      v54 = (unsigned int)v411;
      v410 = (_BYTE *)v337;
    }
    else
    {
      v337 = (__int64)v410;
    }
    v55 = (_QWORD *)(v337 + 80LL * (unsigned int)v54);
    if ( v55 )
    {
      v55[1] = 0x800000000LL;
      *v55 = v55 + 2;
      if ( (_DWORD)v396 )
        sub_19742B0((__int64)v55, v29, v54, v37, (int)v38, v31);
      LODWORD(v54) = v411;
    }
    v44 = v395;
    LODWORD(v411) = v54 + 1;
    if ( (__int64 *)v395 != v33 )
      goto LABEL_29;
LABEL_30:
    ++v34;
  }
  while ( v330 != v34 );
  v45 = v411;
  if ( !(_DWORD)v411 )
  {
    v308 = 1;
    v104 = v410;
    goto LABEL_161;
  }
  while ( 2 )
  {
    v51 = (__int64)v410;
    v50 = 80LL * v45;
    v359 = v361;
    v52 = (__int64)&v410[v50 - 80];
    v360 = 0x800000000LL;
    if ( *(_DWORD *)(v52 + 8) )
    {
      sub_19742B0((__int64)&v359, (char **)v52, (__int64)v410, v50, (int)v38, v31);
      v51 = (__int64)v410;
      v45 = v411;
    }
    v46 = v45 - 1;
    LODWORD(v411) = v46;
    v47 = (unsigned __int64 *)(80 * v46 + v51);
    v48 = *v47;
    v49 = (__int64)(v47 + 2);
    if ( v48 != v49 )
      _libc_free(v48);
    v31 = v360;
    v308 = 0;
    v362 = v364;
    v363 = 0x800000000LL;
    if ( !(_DWORD)v360 )
      goto LABEL_36;
    sub_19741D0((__int64)&v362, (__int64)&v359, v49, v50, (int)v38, v360);
    v312 = v363;
    if ( (unsigned int)(v363 - 2) > 8 )
      goto LABEL_104;
    v395 = (unsigned __int64)&v397;
    v396 = 0x800000000LL;
    sub_19741D0((__int64)&v395, (__int64)&v362, v56, v57, (int)v38, v31);
    v316 = (__int64 *)(v395 + 8LL * (unsigned int)v396);
    if ( (__int64 *)v395 != v316 )
    {
      v322 = (__int64 *)v395;
      while ( 1 )
      {
        v58 = *v322;
        v59 = sub_1481F60(*(_QWORD **)(a1 + 160), *v322, a2, a3);
        if ( v59 == sub_1456E90(*(_QWORD *)(a1 + 160)) || (v60 = *(_QWORD *)(**(_QWORD **)(v58 + 32) + 8LL)) == 0 )
        {
LABEL_78:
          if ( (__int64 *)v395 != &v397 )
            _libc_free(v395);
          v308 = 0;
          v68 = v362;
          goto LABEL_81;
        }
        while ( 1 )
        {
          v61 = sub_1648700(v60);
          if ( (unsigned __int8)(*((_BYTE *)v61 + 16) - 25) <= 9u )
            break;
          v60 = *(_QWORD *)(v60 + 8);
          if ( !v60 )
            goto LABEL_78;
        }
        v62 = *(_QWORD **)(v58 + 72);
        v63 = 0;
LABEL_65:
        v66 = v61[5];
        v65 = *(_QWORD **)(v58 + 64);
        if ( v62 == v65 )
        {
          v67 = *(unsigned int *)(v58 + 84);
          v64 = &v62[v67];
          if ( v62 == v64 )
          {
            v103 = v62;
          }
          else
          {
            do
            {
              if ( v66 == *v65 )
                break;
              ++v65;
            }
            while ( v64 != v65 );
            v103 = &v62[v67];
          }
        }
        else
        {
          v338 = v66;
          v64 = &v62[*(unsigned int *)(v58 + 80)];
          v65 = sub_16CC9F0(v58 + 56, v66);
          if ( v338 == *v65 )
          {
            v62 = *(_QWORD **)(v58 + 72);
            if ( v62 == *(_QWORD **)(v58 + 64) )
              v103 = &v62[*(unsigned int *)(v58 + 84)];
            else
              v103 = &v62[*(unsigned int *)(v58 + 80)];
          }
          else
          {
            v62 = *(_QWORD **)(v58 + 72);
            if ( v62 != *(_QWORD **)(v58 + 64) )
            {
              v65 = &v62[*(unsigned int *)(v58 + 80)];
              goto LABEL_62;
            }
            v65 = &v62[*(unsigned int *)(v58 + 84)];
            v103 = v65;
          }
        }
        while ( v103 != v65 && *v65 >= 0xFFFFFFFFFFFFFFFELL )
          ++v65;
LABEL_62:
        v63 += v65 != v64;
        while ( 1 )
        {
          v60 = *(_QWORD *)(v60 + 8);
          if ( !v60 )
            break;
          v61 = sub_1648700(v60);
          if ( (unsigned __int8)(*((_BYTE *)v61 + 16) - 25) <= 9u )
            goto LABEL_65;
        }
        if ( v63 != 1 || !sub_13F9E70(v58) )
          goto LABEL_78;
        if ( v316 == ++v322 )
        {
          v316 = (__int64 *)v395;
          break;
        }
      }
    }
    if ( v316 != &v397 )
      _libc_free((unsigned __int64)v316);
    v343 = 0;
    v344 = 0;
    v69 = *(_QWORD *)(a1 + 176);
    v345 = 0;
    v70 = *v362;
    v323 = v69;
    v395 = (unsigned __int64)&v397;
    v396 = 0x1000000000LL;
    v71 = *(_QWORD *)(v70 + 32);
    v72 = *(_QWORD *)(v70 + 40);
    v331 = v70;
    if ( v71 != v72 )
    {
      while ( 2 )
      {
        v73 = *(_QWORD *)(*(_QWORD *)v71 + 48LL);
        v74 = *(_QWORD *)v71 + 40LL;
        if ( v73 != v74 )
        {
          while ( v73 )
          {
            v75 = *(_BYTE *)(v73 - 8);
            if ( v75 == 54 || v75 == 55 )
            {
              if ( sub_15F32D0(v73 - 24) || (*(_BYTE *)(v73 - 6) & 1) != 0 )
                goto LABEL_95;
              v102 = (unsigned int)v396;
              if ( (unsigned int)v396 >= HIDWORD(v396) )
              {
                sub_16CD150((__int64)&v395, &v397, 0, 8, (int)v38, v31);
                v102 = (unsigned int)v396;
              }
              *(_QWORD *)(v395 + 8 * v102) = v73 - 24;
              LODWORD(v396) = v396 + 1;
            }
            v73 = *(_QWORD *)(v73 + 8);
            if ( v74 == v73 )
              goto LABEL_107;
          }
LABEL_581:
          BUG();
        }
LABEL_107:
        v71 += 8;
        if ( v72 != v71 )
          continue;
        break;
      }
      v78 = (__int64 *)v395;
      v79 = (unsigned int)v396;
      v317 = (__int64 *)(v395 + 8LL * (unsigned int)v396);
      if ( (__int64 *)v395 == v317 )
        goto LABEL_164;
      v80 = (__int64 *)v395;
      while ( 2 )
      {
        v310 = &v78[v79];
        if ( v310 == v80 )
          goto LABEL_152;
        v339 = v80;
LABEL_112:
        src = 0;
        v381 = 0;
        v382 = 0;
        v81 = *v80;
        v82 = *v339;
        if ( *v80 == *v339 || *(_BYTE *)(v81 + 16) == 54 && *(_BYTE *)(v82 + 16) == 54 )
          goto LABEL_150;
        sub_13B1040((__m128i **)&v372, v323, v81, v82, 1);
        if ( !v372.m128i_i64[0] )
          goto LABEL_148;
        v83 = 1;
        v84 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v372.m128i_i64[0] + 40LL))(v372.m128i_i64[0]);
        if ( !v84 )
        {
LABEL_131:
          v92 = v381;
          v93 = v312;
          if ( v381 - (_BYTE *)src == v312 )
          {
            v93 = v381 - (_BYTE *)src;
          }
          else
          {
            do
            {
              while ( 1 )
              {
                LOBYTE(v367[0]) = 73;
                if ( v382 != v92 )
                  break;
                sub_17EB120((__int64)&src, v92, (char *)v367);
                v92 = v381;
                if ( v381 - (_BYTE *)src == v312 )
                  goto LABEL_138;
              }
              if ( v92 )
                *v92 = 73;
              v92 = v381 + 1;
              v94 = ++v381 - (_BYTE *)src;
            }
            while ( v94 != v312 );
          }
LABEL_138:
          v95 = v344;
          if ( v344 == v345 )
          {
            sub_1974CB0((__int64)&v343, v344, &src);
            v100 = v344;
          }
          else
          {
            if ( v344 )
            {
              *(_QWORD *)v344 = 0;
              *((_QWORD *)v95 + 1) = 0;
              *((_QWORD *)v95 + 2) = 0;
              v96 = (char *)sub_22077B0(v93);
              v97 = &v96[v93];
              *(_QWORD *)v95 = v96;
              v98 = v96;
              v99 = 0;
              *((_QWORD *)v95 + 1) = v96;
              *((_QWORD *)v95 + 2) = v97;
              if ( v381 != src )
              {
                v99 = v381 - (_BYTE *)src;
                v98 = (char *)memmove(v96, src, v381 - (_BYTE *)src);
              }
              *((_QWORD *)v95 + 1) = &v98[v99];
              v95 = v344;
            }
            v100 = v95 + 24;
            v344 = v100;
          }
          if ( (unsigned __int64)(v100 - v343) > 0x960 )
          {
            if ( v372.m128i_i64[0] )
            {
              v201 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v372.m128i_i64[0] + 8LL);
              if ( v201 == sub_13A31C0 )
                j_j___libc_free_0(v372.m128i_i64[0], 40);
              else
                v201(v372.m128i_i64[0]);
            }
            if ( src )
              j_j___libc_free_0(src, v382 - (_BYTE *)src);
LABEL_95:
            if ( (__int64 *)v395 != &v397 )
            {
              _libc_free(v395);
              v308 = 0;
              goto LABEL_97;
            }
LABEL_158:
            v308 = 0;
            goto LABEL_97;
          }
          if ( v372.m128i_i64[0] )
          {
            v101 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v372.m128i_i64[0] + 8LL);
            if ( v101 == sub_13A31C0 )
              j_j___libc_free_0(v372.m128i_i64[0], 40);
            else
              v101(v372.m128i_i64[0]);
          }
LABEL_148:
          if ( src )
            j_j___libc_free_0(src, v382 - (_BYTE *)src);
LABEL_150:
          if ( v310 == ++v339 )
          {
            v78 = (__int64 *)v395;
LABEL_152:
            if ( v317 == ++v80 )
            {
LABEL_164:
              if ( v78 != &v397 )
                _libc_free((unsigned __int64)v78);
              goto LABEL_166;
            }
            v79 = (unsigned int)v396;
            continue;
          }
          goto LABEL_112;
        }
        break;
      }
      while ( 2 )
      {
        v86 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v372.m128i_i64[0] + 56LL))(v372.m128i_i64[0], v83);
        if ( !v86 || *(_WORD *)(v86 + 24) )
        {
          if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v372.m128i_i64[0] + 88LL))(
                 v372.m128i_i64[0],
                 v83) )
          {
            LOBYTE(v365[0]) = 83;
            v85 = v381;
            if ( v381 != v382 )
            {
              if ( v381 )
              {
                *v381 = 83;
                v85 = v381;
              }
LABEL_123:
              ++v83;
              v381 = v85 + 1;
              if ( v84 < v83 )
                goto LABEL_131;
              continue;
            }
            goto LABEL_130;
          }
          v198 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v372.m128i_i64[0] + 48LL))(
                   v372.m128i_i64[0],
                   v83);
          if ( (v198 & 0xFFFFFFFD) == 1 )
            goto LABEL_119;
          if ( (v198 & 0xFFFFFFFD) == 4 )
          {
            LOBYTE(v365[0]) = 62;
          }
          else
          {
            v199 = v198 == 2;
            v200 = 42;
            if ( v199 )
              v200 = 61;
            LOBYTE(v365[0]) = v200;
          }
        }
        else
        {
          v87 = *(_QWORD *)(v86 + 32);
          v88 = *(_DWORD *)(v87 + 32);
          v89 = *(_QWORD *)(v87 + 24);
          v90 = 1LL << ((unsigned __int8)v88 - 1);
          if ( v88 <= 0x40 )
          {
            if ( (v90 & v89) == 0 )
            {
              v91 = v89 == 0;
              goto LABEL_129;
            }
          }
          else if ( (*(_QWORD *)(v89 + 8LL * ((v88 - 1) >> 6)) & v90) == 0 )
          {
            v306 = *(_DWORD *)(v87 + 32);
            v91 = v306 == (unsigned int)sub_16A57B0(v87 + 24);
LABEL_129:
            v85 = v381;
            LOBYTE(v365[0]) = !v91 + 61;
            if ( v381 != v382 )
            {
LABEL_121:
              if ( v85 )
              {
                *v85 = v365[0];
                v85 = v381;
              }
              goto LABEL_123;
            }
LABEL_130:
            ++v83;
            sub_1683630((__int64)&src, v85, (char *)v365);
            if ( v84 < v83 )
              goto LABEL_131;
            continue;
          }
LABEL_119:
          LOBYTE(v365[0]) = 60;
        }
        break;
      }
      v85 = v381;
      if ( v381 != v382 )
        goto LABEL_121;
      goto LABEL_130;
    }
LABEL_166:
    v300 = sub_13FA090(v331);
    if ( !v300 )
      goto LABEL_158;
    v107 = v363;
    if ( (_DWORD)v363 == 1 )
      goto LABEL_158;
    v308 = 0;
    v304 = (unsigned int)(v363 - 1);
    v108 = (unsigned int)(v363 - 2);
    while ( 2 )
    {
      v109 = v366;
      v365[1] = 0x800000000LL;
      v365[0] = (unsigned __int64)v366;
      if ( v107 )
      {
        sub_19741D0((__int64)v365, (__int64)&v362, (__int64)v366, v106, (int)v38, v31);
        v109 = (_BYTE *)v365[0];
      }
      v110 = *(_QWORD *)&v109[8 * v108];
      v111 = *(_QWORD *)&v109[8 * v304];
      v112 = *(__int64 **)(a1 + 200);
      v358 = 0;
      v113 = *(_QWORD *)(a1 + 168);
      v301 = v108;
      v114 = *(_BYTE *)(a1 + 192);
      v115 = *(_QWORD *)(a1 + 160);
      v355 = *(_QWORD *)(a1 + 184);
      v307 = (_QWORD *)v111;
      v353 = v115;
      v116 = (__int64)v343;
      v356 = v114;
      v340 = v110;
      v351 = v110;
      v352 = v111;
      v117 = 0xAAAAAAAAAAAAAAABLL * ((v344 - v343) >> 3);
      v354 = v113;
      v357 = v112;
      if ( (_DWORD)v117 )
      {
        v118 = 8396803;
        v119 = 4198401;
        v120 = (__int64)&v343[24 * (unsigned int)(v117 - 1) + 24];
        while ( 1 )
        {
          v121 = *(_QWORD *)v116;
          v122 = *(unsigned __int8 *)(*(_QWORD *)v116 + v304);
          v123 = *(_BYTE *)(*(_QWORD *)v116 + v108);
          if ( (_BYTE)v122 == 42 || v123 == 42 )
            break;
          v124 = 0;
          do
          {
            v125 = *(_BYTE *)(v121 + v124);
            if ( v125 == 60 )
              break;
            if ( v125 == 62 )
              goto LABEL_181;
            ++v124;
          }
          while ( v124 <= (unsigned int)v108 );
          if ( (_BYTE)v122 != v123 )
          {
            if ( (unsigned __int8)(v122 - 60) > 0x17u )
              break;
            if ( !_bittest64(&v118, (unsigned int)(v122 - 60)) )
            {
              if ( !(_DWORD)v108 || (_BYTE)v122 != 62 )
                break;
              v132 = 0;
              while ( (unsigned __int8)(*(_BYTE *)(v121 + v132) - 61) <= 0x16u
                   && _bittest64(&v119, (unsigned int)*(unsigned __int8 *)(v121 + v132) - 61) )
              {
                if ( (unsigned int)v108 <= (unsigned int)++v132 )
                  goto LABEL_181;
              }
            }
          }
          v116 += 24;
          if ( v120 == v116 )
            goto LABEL_207;
        }
LABEL_181:
        v126 = sub_15E0530(*v112);
        if ( !sub_1602790(v126) )
        {
          v173 = sub_15E0530(*v112);
          v174 = sub_16033E0(v173);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v174 + 48LL))(v174) )
            goto LABEL_270;
        }
        v127 = **(_QWORD **)(v352 + 32);
        sub_13FD840(v367, v352);
        sub_15C9090((__int64)&v372, v367);
        sub_15CA540((__int64)&v395, (__int64)"loop-interchange", (__int64)"Dependence", 10, &v372, v127);
        sub_15CAB20((__int64)&v395, "Cannot interchange loops due to dependences.", 0x2Cu);
        a2 = _mm_loadu_si128(&v398);
        a3 = _mm_loadu_si128(&v401);
        LODWORD(v381) = v396;
        v383 = (__m128)a2;
        BYTE4(v381) = BYTE4(v396);
        v386 = (__m128)a3;
        v382 = (_BYTE *)v397;
        v384 = v399;
        src = &unk_49ECF68;
        v385 = v400;
        v388 = (char)v403;
        if ( (_BYTE)v403 )
          v387 = v402;
        v389 = v391;
        v390 = 0x400000000LL;
        if ( v405 )
        {
          sub_1974F80((__int64)&v389, (__int64)&v404);
          v392 = v407;
          v393 = v408;
          v394 = v409;
          src = &unk_49ECFC8;
          v395 = (unsigned __int64)&unk_49ECF68;
          v128 = (_QWORD *)(v404 + 88LL * v405);
          if ( (_QWORD *)v404 != v128 )
          {
            v156 = (_QWORD *)v404;
            do
            {
              v128 -= 11;
              v157 = (_QWORD *)v128[4];
              if ( v157 != v128 + 6 )
                j_j___libc_free_0(v157, v128[6] + 1LL);
              if ( (_QWORD *)*v128 != v128 + 2 )
                j_j___libc_free_0(*v128, v128[2] + 1LL);
            }
            while ( v156 != v128 );
            v128 = (_QWORD *)v404;
          }
        }
        else
        {
          v128 = (_QWORD *)v404;
          v392 = v407;
          v393 = v408;
          v394 = v409;
          src = &unk_49ECFC8;
        }
        if ( v128 != (_QWORD *)v406 )
          _libc_free((unsigned __int64)v128);
        if ( v367[0] )
          sub_161E7C0((__int64)v367, v367[0]);
        sub_143AA50(v112, (__int64)&src);
        v129 = v389;
        src = &unk_49ECF68;
        v130 = &v389[88 * (unsigned int)v390];
        if ( v389 != (_BYTE *)v130 )
        {
          do
          {
            v130 -= 11;
            v131 = (_QWORD *)v130[4];
            if ( v131 != v130 + 6 )
              j_j___libc_free_0(v131, v130[6] + 1LL);
            if ( (_QWORD *)*v130 != v130 + 2 )
              j_j___libc_free_0(*v130, v130[2] + 1LL);
          }
          while ( v129 != v130 );
          goto LABEL_196;
        }
        goto LABEL_197;
      }
LABEL_207:
      v324 = *(__int64 **)(v340 + 40);
      if ( *(__int64 **)(v340 + 32) == v324 )
      {
        v143 = v340;
        goto LABEL_313;
      }
      v332 = *(__int64 **)(v340 + 32);
      v313 = v108;
      do
      {
        v133 = &v372;
        v116 = *v332;
        sub_1580910(&v372);
        v349 = 0;
        v347 = v372;
        if ( v374 )
        {
          v116 = (__int64)v373;
          v133 = (__m128i *)v348;
          ((void (__fastcall *)(_BYTE *, _QWORD *, __int64))v374)(v348, v373, 2);
          v350 = v375;
          v349 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v374;
        }
        v369 = 0;
        v367[0] = v376[0];
        v367[1] = v376[1];
        if ( v378 )
        {
          v116 = (__int64)v377;
          v133 = (__m128i *)v368;
          ((void (__fastcall *)(_QWORD *, _BYTE *, __int64))v378)(v368, v377, 2);
          v370 = v379;
          v369 = v378;
        }
        while ( 1 )
        {
          v135 = v347.m128i_i64[0];
          if ( v347.m128i_i64[0] == v367[0] )
            break;
          v136 = (__m128i *)v347.m128i_i64[0];
          while ( 1 )
          {
            if ( !v136 )
              BUG();
            if ( v136[-1].m128i_i8[8] == 78 )
            {
              v133 = v136 + 2;
              if ( (unsigned __int8)sub_1560260((__m128i *)v136[2].m128i_i64, -1, 36) )
                goto LABEL_242;
              if ( v136[-1].m128i_i8[15] < 0 )
              {
                v137 = sub_1648A40((__int64)&v136[-2].m128i_i64[1]);
                v139 = v138 + v137;
                v140 = 0;
                v318 = v139;
                if ( v136[-1].m128i_i8[15] < 0 )
                  v140 = sub_1648A40((__int64)&v136[-2].m128i_i64[1]);
                if ( (unsigned int)((v318 - v140) >> 4) )
                  goto LABEL_243;
              }
              v141 = v136[-3].m128i_i64[0];
              if ( *(_BYTE *)(v141 + 16)
                || (v133 = (__m128i *)&v395,
                    v395 = *(_QWORD *)(v141 + 112),
                    v142 = sub_1560260(&v395, -1, 36),
                    v135 = v347.m128i_i64[0],
                    !v142) )
              {
LABEL_243:
                v133 = v136 + 2;
                if ( !(unsigned __int8)sub_1560260((__m128i *)v136[2].m128i_i64, -1, 57) )
                {
                  v144 = v136[-3].m128i_i64[0];
                  if ( *(_BYTE *)(v144 + 16)
                    || (v133 = (__m128i *)&v395,
                        v395 = *(_QWORD *)(v144 + 112),
                        !(unsigned __int8)sub_1560260(&v395, -1, 57)) )
                  {
                    v145 = v357;
                    v146 = sub_15E0530(*v357);
                    if ( sub_1602790(v146)
                      || (v154 = sub_15E0530(*v145),
                          v155 = sub_16033E0(v154),
                          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v155 + 48LL))(v155)) )
                    {
                      v147 = v136[1].m128i_i64[0];
                      sub_15C9090((__int64)v346, &v136[1].m128i_i64[1]);
                      sub_15CA540((__int64)&v395, (__int64)"loop-interchange", (__int64)"CallInst", 8, v346, v147);
                      sub_15CAB20((__int64)&v395, "Cannot interchange loops due to call instruction.", 0x31u);
                      a4 = _mm_loadu_si128(&v398);
                      a5 = _mm_loadu_si128(&v401);
                      LODWORD(v381) = v396;
                      v383 = (__m128)a4;
                      BYTE4(v381) = BYTE4(v396);
                      v386 = (__m128)a5;
                      v382 = (_BYTE *)v397;
                      v384 = v399;
                      src = &unk_49ECF68;
                      v385 = v400;
                      v388 = (char)v403;
                      if ( (_BYTE)v403 )
                        v387 = v402;
                      v389 = v391;
                      v390 = 0x400000000LL;
                      if ( v405 )
                      {
                        sub_1974F80((__int64)&v389, (__int64)&v404);
                        v152 = (_QWORD *)v404;
                        v392 = v407;
                        v393 = v408;
                        v394 = v409;
                        src = &unk_49ECFC8;
                        v395 = (unsigned __int64)&unk_49ECF68;
                        v148 = (_QWORD *)(v404 + 88LL * v405);
                        if ( (_QWORD *)v404 != v148 )
                        {
                          do
                          {
                            v148 -= 11;
                            v153 = (_QWORD *)v148[4];
                            if ( v153 != v148 + 6 )
                              j_j___libc_free_0(v153, v148[6] + 1LL);
                            if ( (_QWORD *)*v148 != v148 + 2 )
                              j_j___libc_free_0(*v148, v148[2] + 1LL);
                          }
                          while ( v152 != v148 );
                          v148 = (_QWORD *)v404;
                        }
                      }
                      else
                      {
                        v148 = (_QWORD *)v404;
                        v392 = v407;
                        v393 = v408;
                        v394 = v409;
                        src = &unk_49ECFC8;
                      }
                      if ( v148 != (_QWORD *)v406 )
                        _libc_free((unsigned __int64)v148);
                      sub_143AA50(v145, (__int64)&src);
                      v149 = v389;
                      src = &unk_49ECF68;
                      v150 = &v389[88 * (unsigned int)v390];
                      if ( v389 != (_BYTE *)v150 )
                      {
                        do
                        {
                          v150 -= 11;
                          v151 = (_QWORD *)v150[4];
                          if ( v151 != v150 + 6 )
                            j_j___libc_free_0(v151, v150[6] + 1LL);
                          if ( (_QWORD *)*v150 != v150 + 2 )
                            j_j___libc_free_0(*v150, v150[2] + 1LL);
                        }
                        while ( v149 != v150 );
                        v150 = v389;
                      }
                      if ( v150 != (_QWORD *)v391 )
                        _libc_free((unsigned __int64)v150);
                    }
                    if ( v369 )
                      ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v369)(v368, v368, 3);
                    if ( v349 )
                      v349(v348, v348, 3);
                    if ( v378 )
                      ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v378)(v377, v377, 3);
                    if ( v374 )
                      ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v374)(v373, v373, 3);
                    goto LABEL_270;
                  }
                }
LABEL_242:
                v135 = v347.m128i_i64[0];
              }
            }
            v135 = *(_QWORD *)(v135 + 8);
            v347.m128i_i64[0] = v135;
            v136 = (__m128i *)v135;
            v116 = v135;
            if ( v135 != v347.m128i_i64[1] )
              break;
LABEL_231:
            if ( (__m128i *)v367[0] == v136 )
              goto LABEL_232;
          }
          while ( 1 )
          {
            if ( v116 )
              v116 -= 24;
            if ( !v349 )
              sub_4263D6(v133, v116, v134);
            v133 = (__m128i *)v348;
            if ( v350(v348, v116) )
              break;
            v116 = *(_QWORD *)(v347.m128i_i64[0] + 8);
            v347.m128i_i64[0] = v116;
            v135 = v116;
            if ( v347.m128i_i64[1] == v116 )
            {
              v136 = (__m128i *)v116;
              goto LABEL_231;
            }
          }
        }
LABEL_232:
        if ( v369 )
        {
          v116 = (__int64)v368;
          ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v369)(v368, v368, 3);
        }
        if ( v349 )
        {
          v116 = (__int64)v348;
          v349(v348, v348, 3);
        }
        if ( v378 )
        {
          v116 = (__int64)v377;
          ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v378)(v377, v377, 3);
        }
        if ( v374 )
        {
          v116 = (__int64)v373;
          ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v374)(v373, v373, 3);
        }
        ++v332;
      }
      while ( v324 != v332 );
      v108 = v313;
      v143 = v351;
LABEL_313:
      v175 = sub_13FC520(v143);
      v176 = sub_13FC520(v352);
      v177 = v351;
      v178 = v176;
      if ( !v175 || v175 == **(_QWORD **)(v351 + 32) )
      {
LABEL_319:
        v116 = v355;
        sub_1AF8F90(v177, v355, v354, v356);
        goto LABEL_320;
      }
      v179 = *(_QWORD *)(v175 + 48);
      if ( !v179 )
        goto LABEL_581;
      if ( *(_BYTE *)(v179 - 8) == 77 )
        goto LABEL_319;
      if ( !sub_157F120(v175) )
      {
        v177 = v351;
        goto LABEL_319;
      }
LABEL_320:
      if ( !v178 || v178 == **(_QWORD **)(v352 + 32) || v178 == **(_QWORD **)(v351 + 32) )
      {
        v116 = v355;
        sub_1AF8F90(v352, v355, v354, v356);
      }
      if ( (unsigned __int8)sub_1975210(&v351) )
        goto LABEL_270;
      v180 = v351;
      v181 = **(_QWORD **)(v351 + 32);
      v182 = sub_13FC520(v352);
      v333 = sub_13FCB50(v180);
      v183 = sub_157EBA0(v181);
      v184 = v183;
      if ( *(_BYTE *)(v183 + 16) != 26 )
        goto LABEL_330;
      v185 = 0;
      v186 = sub_15F4D60(v183);
      if ( v186 )
      {
        v327 = v108;
        v187 = v181;
        v188 = v186;
        do
        {
          v116 = v185;
          v189 = sub_15F4DF0(v184, v185);
          if ( v182 != v189 && v333 != v189 )
            goto LABEL_330;
          ++v185;
        }
        while ( v188 != v185 );
        v181 = v187;
        v108 = v327;
      }
      v205 = *(_QWORD *)(v181 + 48);
      v328 = v181 + 40;
      if ( v205 == v181 + 40 )
      {
LABEL_399:
        v210 = *(_QWORD *)(v333 + 48);
        if ( v210 != v333 + 40 )
        {
          while ( v210 )
          {
            if ( *(_BYTE *)(v210 - 8) == 55 )
            {
              if ( *(_BYTE *)(*(_QWORD *)(v210 - 72) + 16LL) != 77 )
                goto LABEL_330;
            }
            else if ( (unsigned __int8)sub_15F3040(v210 - 24)
                   || sub_15F3330(v210 - 24)
                   || (unsigned __int8)sub_15F2ED0(v210 - 24) )
            {
              goto LABEL_330;
            }
            v210 = *(_QWORD *)(v210 + 8);
            if ( v333 + 40 == v210 )
              goto LABEL_404;
          }
          goto LABEL_581;
        }
LABEL_404:
        v211 = v351;
        v212 = sub_13FA560(v351);
        v213 = sub_157F280(v212);
        v334 = v214;
        v215 = v213;
        if ( v213 != v214 )
        {
          while ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v215 + 8LL) - 1) > 5u )
          {
            v287 = *(_DWORD *)(v215 + 20);
            v288 = 0;
            v289 = v287 & 0xFFFFFFF;
            if ( (v287 & 0xFFFFFFF) != 0 )
            {
              do
              {
                if ( (*(_BYTE *)(v215 + 23) & 0x40) != 0 )
                  v290 = *(_QWORD *)(v215 - 8);
                else
                  v290 = v215 - 24LL * v289;
                v291 = *(_QWORD *)(v290 + 24 * v288);
                if ( *(_BYTE *)(v291 + 16) > 0x17u )
                {
                  v292 = *(_QWORD *)(v291 + 40);
                  if ( v292 == sub_13FCB50(v211) )
                  {
                    v293 = sub_13FCB50(v211);
                    if ( !sub_157F120(v293) )
                      goto LABEL_406;
                  }
                  v287 = *(_DWORD *)(v215 + 20);
                }
                ++v288;
                v289 = v287 & 0xFFFFFFF;
              }
              while ( (v287 & 0xFFFFFFFu) > (unsigned int)v288 );
            }
            v222 = *(_QWORD *)(v215 + 32);
            if ( !v222 )
              goto LABEL_581;
            v215 = 0;
            if ( *(_BYTE *)(v222 - 8) == 77 )
              v215 = v222 - 24;
            if ( v334 == v215 )
              goto LABEL_430;
          }
LABEL_406:
          v216 = v357;
          v217 = sub_15E0530(*v357);
          if ( !sub_1602790(v217) )
          {
            v297 = sub_15E0530(*v216);
            v298 = sub_16033E0(v297);
            if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v298 + 48LL))(v298, v116) )
              goto LABEL_270;
          }
          v218 = **(_QWORD **)(v351 + 32);
          sub_13FD840(v367, v351);
          sub_15C9090((__int64)&v372, v367);
          sub_15CA540((__int64)&v395, (__int64)"loop-interchange", (__int64)"UnsupportedExitPHI", 18, &v372, v218);
          sub_15CAB20((__int64)&v395, "Found unsupported PHI node in loop exit.", 0x28u);
          a8 = _mm_loadu_si128(&v398);
          a9 = (__m128)_mm_loadu_si128(&v401);
          LODWORD(v381) = v396;
          v383 = (__m128)a8;
          BYTE4(v381) = BYTE4(v396);
          v386 = a9;
          v382 = (_BYTE *)v397;
          v384 = v399;
          src = &unk_49ECF68;
          v385 = v400;
          v388 = (char)v403;
          if ( (_BYTE)v403 )
            v387 = v402;
          v389 = v391;
          v390 = 0x400000000LL;
          if ( v405 )
          {
            sub_1974F80((__int64)&v389, (__int64)&v404);
            v392 = v407;
            v342 = (_QWORD *)v404;
            v393 = v408;
            v394 = v409;
            src = &unk_49ECFC8;
            v395 = (unsigned __int64)&unk_49ECF68;
            v219 = (_QWORD *)(v404 + 88LL * v405);
            if ( (_QWORD *)v404 != v219 )
            {
              do
              {
                v219 -= 11;
                v296 = (_QWORD *)v219[4];
                if ( v296 != v219 + 6 )
                  j_j___libc_free_0(v296, v219[6] + 1LL);
                if ( (_QWORD *)*v219 != v219 + 2 )
                  j_j___libc_free_0(*v219, v219[2] + 1LL);
              }
              while ( v342 != v219 );
              v219 = (_QWORD *)v404;
            }
          }
          else
          {
            v219 = (_QWORD *)v404;
            v392 = v407;
            v393 = v408;
            v394 = v409;
            src = &unk_49ECFC8;
          }
          if ( v219 != (_QWORD *)v406 )
            _libc_free((unsigned __int64)v219);
          if ( v367[0] )
            sub_161E7C0((__int64)v367, v367[0]);
          sub_143AA50(v216, (__int64)&src);
          v220 = v389;
          src = &unk_49ECF68;
          v130 = &v389[88 * (unsigned int)v390];
          if ( v389 != (_BYTE *)v130 )
          {
            do
            {
              v130 -= 11;
              v221 = (_QWORD *)v130[4];
              if ( v221 != v130 + 6 )
                j_j___libc_free_0(v221, v130[6] + 1LL);
              if ( (_QWORD *)*v130 != v130 + 2 )
                j_j___libc_free_0(*v130, v130[2] + 1LL);
            }
            while ( v220 != v130 );
            goto LABEL_196;
          }
          goto LABEL_197;
        }
LABEL_430:
        v223 = *(_QWORD *)(a1 + 160);
        v303 = *(__int64 **)(a1 + 200);
        v305 = v307[5];
        if ( v307[4] != v305 )
        {
          v315 = v307[4];
          v302 = 0;
          v311 = 0;
          v299 = v108;
          while ( 1 )
          {
            v224 = *(_QWORD *)(*(_QWORD *)v315 + 48LL);
            v329 = *(_QWORD *)v315 + 40LL;
            if ( v224 != v329 )
              break;
LABEL_445:
            v315 += 8;
            if ( v305 == v315 )
            {
              v108 = v299;
              v231 = v311 - v302;
              goto LABEL_447;
            }
          }
          while ( 1 )
          {
            if ( !v224 )
              goto LABEL_581;
            if ( *(_BYTE *)(v224 - 8) == 56 )
            {
              v225 = *(_DWORD *)(v224 - 4) & 0xFFFFFFF;
              if ( v225 )
                break;
            }
LABEL_444:
            v224 = *(_QWORD *)(v224 + 8);
            if ( v329 == v224 )
              goto LABEL_445;
          }
          v321 = 0;
          v226 = v225 - 1;
          v335 = 0;
          v227 = v224;
          for ( i = 0; ; ++i )
          {
            v229 = sub_146F1B0(v223, *(_QWORD *)(v227 + 24 * (i - v225) - 24));
            if ( *(_WORD *)(v229 + 24) != 7 )
              goto LABEL_438;
            v230 = *(_QWORD **)(v229 + 48);
            if ( v307 == v230 )
              break;
            if ( (_QWORD *)v340 != v230 )
              goto LABEL_438;
            if ( v321 )
              goto LABEL_490;
            v335 = 1;
            if ( v226 == i )
            {
LABEL_487:
              v224 = v227;
              goto LABEL_444;
            }
LABEL_439:
            v225 = *(_DWORD *)(v227 - 4) & 0xFFFFFFF;
          }
          if ( v335 )
          {
            ++v311;
            v224 = v227;
            goto LABEL_444;
          }
          if ( (_QWORD *)v340 == v307 )
          {
LABEL_490:
            ++v302;
            v224 = v227;
            goto LABEL_444;
          }
          v321 = 1;
LABEL_438:
          if ( v226 == i )
            goto LABEL_487;
          goto LABEL_439;
        }
        v231 = 0;
LABEL_447:
        if ( v231 >= -dword_4FB07E0 )
        {
          v232 = v343;
          if ( v343 != v344 )
          {
            while ( 1 )
            {
              v233 = *(_BYTE *)(*(_QWORD *)v232 + v304);
              if ( v233 != 83 && v233 != 73 )
                break;
              if ( *(_BYTE *)(*(_QWORD *)v232 + v108) != 61 )
                break;
              v232 += 24;
              if ( v344 == v232 )
                goto LABEL_491;
            }
          }
          v234 = sub_15E0530(*v303);
          if ( !sub_1602790(v234) )
          {
            v285 = sub_15E0530(*v303);
            v286 = sub_16033E0(v285);
            if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v286 + 48LL))(v286) )
              goto LABEL_270;
          }
          v235 = *(_QWORD *)v307[4];
          sub_13FD840(v346, (__int64)v307);
          sub_15C9090((__int64)&v347, v346);
          sub_15CA540((__int64)&v395, (__int64)"loop-interchange", (__int64)"InterchangeNotProfitable", 24, &v347, v235);
          sub_15CAB20((__int64)&v395, "Interchanging loops is too costly (cost=", 0x28u);
          sub_15C9890((__int64)&v372, "Cost", 4, (unsigned int)v231);
          v236 = sub_17C21B0((__int64)&v395, (__int64)&v372);
          sub_15CAB20(v236, ", threshold=", 0xCu);
          sub_15C9890((__int64)v367, "Threshold", 9, (unsigned int)dword_4FB07E0);
          v237 = sub_17C21B0(v236, (__int64)v367);
          sub_15CAB20(v237, ") and it does not improve parallelism.", 0x26u);
          LODWORD(v381) = *(_DWORD *)(v237 + 8);
          BYTE4(v381) = *(_BYTE *)(v237 + 12);
          v382 = *(_BYTE **)(v237 + 16);
          a2 = _mm_loadu_si128((const __m128i *)(v237 + 24));
          v383 = (__m128)a2;
          v384 = *(_QWORD *)(v237 + 40);
          src = &unk_49ECF68;
          v385 = *(_QWORD *)(v237 + 48);
          a3 = _mm_loadu_si128((const __m128i *)(v237 + 56));
          v386 = (__m128)a3;
          v388 = *(_BYTE *)(v237 + 80);
          if ( v388 )
            v387 = *(_BYTE **)(v237 + 72);
          v389 = v391;
          v390 = 0x400000000LL;
          if ( *(_DWORD *)(v237 + 96) )
            sub_1974F80((__int64)&v389, v237 + 88);
          v392 = *(_BYTE *)(v237 + 456);
          v393 = *(_DWORD *)(v237 + 460);
          v394 = *(_QWORD *)(v237 + 464);
          src = &unk_49ECFC8;
          if ( v369 != &v371 )
            j_j___libc_free_0(v369, v371 + 1);
          if ( (_QWORD *)v367[0] != v368 )
            j_j___libc_free_0(v367[0], v368[0] + 1LL);
          if ( v374 != v376 )
            j_j___libc_free_0(v374, v376[0] + 1LL);
          if ( (_QWORD *)v372.m128i_i64[0] != v373 )
            j_j___libc_free_0(v372.m128i_i64[0], v373[0] + 1LL);
          v238 = (_QWORD *)v404;
          v395 = (unsigned __int64)&unk_49ECF68;
          v239 = (_QWORD *)(v404 + 88LL * v405);
          if ( (_QWORD *)v404 != v239 )
          {
            do
            {
              v239 -= 11;
              v240 = (_QWORD *)v239[4];
              if ( v240 != v239 + 6 )
                j_j___libc_free_0(v240, v239[6] + 1LL);
              if ( (_QWORD *)*v239 != v239 + 2 )
                j_j___libc_free_0(*v239, v239[2] + 1LL);
            }
            while ( v238 != v239 );
            v239 = (_QWORD *)v404;
          }
          if ( v239 != (_QWORD *)v406 )
            _libc_free((unsigned __int64)v239);
          if ( v346[0].m128i_i64[0] )
            sub_161E7C0((__int64)v346, v346[0].m128i_i64[0]);
          sub_143AA50(v303, (__int64)&src);
          v241 = v389;
          src = &unk_49ECF68;
          v130 = &v389[88 * (unsigned int)v390];
          if ( v389 != (_BYTE *)v130 )
          {
            do
            {
              v130 -= 11;
              v242 = (_QWORD *)v130[4];
              if ( v242 != v130 + 6 )
                j_j___libc_free_0(v242, v130[6] + 1LL);
              if ( (_QWORD *)*v130 != v130 + 2 )
                j_j___libc_free_0(*v130, v130[2] + 1LL);
            }
            while ( v241 != v130 );
LABEL_196:
            v130 = v389;
          }
LABEL_197:
          if ( v130 != (_QWORD *)v391 )
            _libc_free((unsigned __int64)v130);
          goto LABEL_270;
        }
LABEL_491:
        v336 = *(__int64 **)(a1 + 200);
        v243 = sub_15E0530(*v336);
        if ( sub_1602790(v243)
          || (v283 = sub_15E0530(*v336),
              v284 = sub_16033E0(v283),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v284 + 48LL))(v284)) )
        {
          v244 = *(_QWORD *)v307[4];
          sub_13FD840(v367, (__int64)v307);
          sub_15C9090((__int64)&v372, v367);
          sub_15CA330((__int64)&v395, (__int64)"loop-interchange", (__int64)"Interchanged", 12, &v372, v244);
          sub_15CAB20((__int64)&v395, "Loop interchanged with enclosing loop.", 0x26u);
          a4 = _mm_loadu_si128(&v398);
          a5 = _mm_loadu_si128(&v401);
          LODWORD(v381) = v396;
          v383 = (__m128)a4;
          BYTE4(v381) = BYTE4(v396);
          v386 = (__m128)a5;
          v382 = (_BYTE *)v397;
          v384 = v399;
          src = &unk_49ECF68;
          v385 = v400;
          v388 = (char)v403;
          if ( (_BYTE)v403 )
            v387 = v402;
          v390 = 0x400000000LL;
          v389 = v391;
          if ( v405 )
          {
            sub_1974F80((__int64)&v389, (__int64)&v404);
            v392 = v407;
            v393 = v408;
            v394 = v409;
            src = &unk_49ECF98;
            v395 = (unsigned __int64)&unk_49ECF68;
            v245 = (_QWORD *)(v404 + 88LL * v405);
            if ( (_QWORD *)v404 != v245 )
            {
              v264 = (_QWORD *)v404;
              do
              {
                v245 -= 11;
                v265 = (_QWORD *)v245[4];
                if ( v265 != v245 + 6 )
                  j_j___libc_free_0(v265, v245[6] + 1LL);
                if ( (_QWORD *)*v245 != v245 + 2 )
                  j_j___libc_free_0(*v245, v245[2] + 1LL);
              }
              while ( v264 != v245 );
              v245 = (_QWORD *)v404;
            }
          }
          else
          {
            v245 = (_QWORD *)v404;
            v392 = v407;
            v393 = v408;
            v394 = v409;
            src = &unk_49ECF98;
          }
          if ( v245 != (_QWORD *)v406 )
            _libc_free((unsigned __int64)v245);
          if ( v367[0] )
            sub_161E7C0((__int64)v367, v367[0]);
          sub_143AA50(v336, (__int64)&src);
          v248 = v389;
          src = &unk_49ECF68;
          v249 = &v389[88 * (unsigned int)v390];
          if ( v389 != (_BYTE *)v249 )
          {
            do
            {
              v249 -= 11;
              v250 = (_QWORD *)v249[4];
              if ( v250 != v249 + 6 )
                j_j___libc_free_0(v250, v249[6] + 1LL);
              if ( (_QWORD *)*v249 != v249 + 2 )
                j_j___libc_free_0(*v249, v249[2] + 1LL);
            }
            while ( v248 != v249 );
            v249 = v389;
          }
          if ( v249 != (_QWORD *)v391 )
            _libc_free((unsigned __int64)v249);
        }
        v251 = *(_QWORD *)(a1 + 184);
        v395 = v340;
        v252 = *(_QWORD *)(a1 + 168);
        v398.m128i_i64[1] = v251;
        v253 = *(_QWORD *)(a1 + 160);
        v396 = (__int64)v307;
        v399 = v300;
        v397 = v253;
        v398.m128i_i64[0] = v252;
        LOBYTE(v400) = v358;
        if ( v307[2] != v307[1] )
          goto LABEL_510;
        v270 = sub_13FC520((__int64)v307);
        v271 = v396;
        v272 = v397;
        v273 = v270;
        v274 = sub_13FCD20(v396);
        if ( !v274 )
        {
          if ( sub_13FCB50(v271) && sub_13FC470(v271) )
          {
            for ( j = *(_QWORD *)(**(_QWORD **)(v271 + 32) + 48LL); j; j = *(_QWORD *)(j + 8) )
            {
              v274 = j - 24;
              if ( *(_BYTE *)(j - 8) != 77
                || (*(_BYTE *)(*(_QWORD *)(j - 24) + 8LL) & 0xFB) != 0xB
                && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(j - 24) + 8LL) - 1) > 5u )
              {
                goto LABEL_512;
              }
              v295 = sub_146F1B0(v272, j - 24);
              if ( *(_WORD *)(v295 + 24) == 7
                && *(_QWORD *)(v295 + 40) == 2
                && !*(_WORD *)(*(_QWORD *)(*(_QWORD *)(v295 + 32) + 8LL) + 24LL) )
              {
                goto LABEL_529;
              }
            }
            goto LABEL_581;
          }
          goto LABEL_512;
        }
LABEL_529:
        v275 = 3LL * *(unsigned int *)(v274 + 56);
        if ( (*(_BYTE *)(v274 + 23) & 0x40) != 0 )
        {
          v276 = *(_QWORD **)(v274 - 8);
          if ( v276[v275 + 1] != v273 )
            goto LABEL_531;
        }
        else
        {
          v276 = (_QWORD *)(v274 - 24LL * (*(_DWORD *)(v274 + 20) & 0xFFFFFFF));
          if ( v276[v275 + 1] != v273 )
          {
LABEL_531:
            v277 = *v276;
            if ( *(_BYTE *)(*v276 + 16LL) <= 0x17u )
              v277 = 0;
LABEL_533:
            v278 = *(_QWORD *)(*(_QWORD *)(v274 + 40) + 48LL);
            if ( !v278 || (v278 -= 24, v274 != v278) )
              sub_15F22F0((_QWORD *)v274, v278);
            v279 = sub_13FCB50(v396);
            sub_1AA8CA0(v279, v277, v398.m128i_i64[1], v398.m128i_i64[0]);
            v280 = v398;
            v281 = **(_QWORD **)(v396 + 32);
            v282 = sub_157ED20(v281);
            sub_1AA8CA0(v281, v282, v280.m128i_i64[1], v280.m128i_i64[0]);
LABEL_510:
            if ( (unsigned __int8)sub_1978000(
                                    (__int64 *)&v395,
                                    (__m128)a2,
                                    *(double *)a3.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    *(double *)a5.m128i_i64,
                                    v246,
                                    v247,
                                    *(double *)a8.m128i_i64,
                                    a9) )
            {
              v266 = sub_13FC520(v395);
              v267 = sub_13FC520(v396);
              v268 = sub_157EBA0(v267);
              v269 = sub_157EBA0(**(_QWORD **)(v395 + 32));
              sub_1973F90(v267, v269);
              sub_1973F90(v266, v268);
            }
LABEL_512:
            if ( (_BYTE *)v365[0] != v366 )
              _libc_free(v365[0]);
            v254 = &v362[v304];
            v255 = v362[v301];
            v362[v301] = *v254;
            *v254 = v255;
            v256 = v343;
            v106 = 0xAAAAAAAAAAAAAAABLL;
            v257 = 0xAAAAAAAAAAAAAAABLL * ((v344 - v343) >> 3);
            if ( (_DWORD)v257 )
            {
              v258 = 3LL * (unsigned int)v257;
              v259 = 0;
              v260 = 8 * v258;
              while ( 1 )
              {
                v261 = *(_QWORD *)&v256[v259];
                v262 = (_BYTE *)(v261 + v108);
                v106 = *(unsigned __int8 *)(v261 + v304);
                LODWORD(v38) = (unsigned __int8)*v262;
                *v262 = v106;
                v263 = *(_QWORD *)&v343[v259];
                v259 += 24;
                *(_BYTE *)(v263 + v304) = (_BYTE)v38;
                if ( v260 == v259 )
                  break;
                v256 = v343;
              }
            }
            --v108;
            --v304;
            if ( v108 == -1 )
            {
              v308 = 1;
              goto LABEL_97;
            }
            v308 = 1;
            v107 = v363;
            continue;
          }
        }
        v277 = v276[3];
        if ( *(_BYTE *)(v277 + 16) <= 0x17u )
          v277 = 0;
        goto LABEL_533;
      }
      break;
    }
    while ( 2 )
    {
      if ( !v205 )
        goto LABEL_581;
      if ( *(_BYTE *)(v205 - 8) != 54 )
      {
        if ( (unsigned __int8)sub_15F3040(v205 - 24)
          || sub_15F3330(v205 - 24)
          || (unsigned __int8)sub_15F2ED0(v205 - 24) )
        {
          goto LABEL_330;
        }
        goto LABEL_398;
      }
      v206 = *(_QWORD *)(v205 - 16);
      v207 = v352;
      if ( !v206 )
      {
LABEL_398:
        v205 = *(_QWORD *)(v205 + 8);
        if ( v328 == v205 )
          goto LABEL_399;
        continue;
      }
      break;
    }
    while ( 1 )
    {
      v208 = sub_1648700(v206);
      v199 = *((_BYTE *)v208 + 16) == 77;
      v209 = v208;
      v395 = 6;
      if ( !v199 )
        v209 = 0;
      v396 = 0;
      v397 = 0;
      v398 = 0u;
      v399 = 0;
      v400 = 0;
      v401.m128i_i8[0] = 0;
      v401.m128i_i64[1] = 0;
      v402 = v406;
      v403 = v406;
      v404 = 8;
      v405 = 0;
      if ( !v209 )
        break;
      v116 = v207;
      v320 = sub_1B1CF40(v209, v207, &v395, 0, 0, 0);
      if ( v403 != v402 )
        _libc_free((unsigned __int64)v403);
      if ( v397 != -16 && v397 != 0 && v397 != -8 )
        sub_1649B30(&v395);
      if ( !v320 )
        break;
      v206 = *(_QWORD *)(v206 + 8);
      if ( !v206 )
        goto LABEL_398;
    }
LABEL_330:
    v190 = v357;
    v191 = sub_15E0530(*v357);
    if ( sub_1602790(v191)
      || (v202 = sub_15E0530(*v190),
          v203 = sub_16033E0(v202),
          (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v203 + 48LL))(v203, v116)) )
    {
      v192 = **(_QWORD **)(v352 + 32);
      sub_13FD840(v367, v352);
      sub_15C9090((__int64)&v372, v367);
      sub_15CA540((__int64)&v395, (__int64)"loop-interchange", (__int64)"NotTightlyNested", 16, &v372, v192);
      sub_15CAB20((__int64)&v395, "Cannot interchange loops because they are not tightly nested.", 0x3Du);
      v193 = _mm_loadu_si128(&v398);
      v194 = _mm_loadu_si128(&v401);
      LODWORD(v381) = v396;
      v383 = (__m128)v193;
      BYTE4(v381) = BYTE4(v396);
      v386 = (__m128)v194;
      v382 = (_BYTE *)v397;
      v384 = v399;
      src = &unk_49ECF68;
      v385 = v400;
      v388 = (char)v403;
      if ( (_BYTE)v403 )
        v387 = v402;
      v389 = v391;
      v390 = 0x400000000LL;
      if ( v405 )
      {
        sub_1974F80((__int64)&v389, (__int64)&v404);
        v392 = v407;
        v341 = (_QWORD *)v404;
        v393 = v408;
        v394 = v409;
        src = &unk_49ECFC8;
        v395 = (unsigned __int64)&unk_49ECF68;
        v195 = (_QWORD *)(v404 + 88LL * v405);
        if ( (_QWORD *)v404 != v195 )
        {
          do
          {
            v195 -= 11;
            v204 = (_QWORD *)v195[4];
            if ( v204 != v195 + 6 )
              j_j___libc_free_0(v204, v195[6] + 1LL);
            if ( (_QWORD *)*v195 != v195 + 2 )
              j_j___libc_free_0(*v195, v195[2] + 1LL);
          }
          while ( v341 != v195 );
          v195 = (_QWORD *)v404;
        }
      }
      else
      {
        v195 = (_QWORD *)v404;
        v392 = v407;
        v393 = v408;
        v394 = v409;
        src = &unk_49ECFC8;
      }
      if ( v195 != (_QWORD *)v406 )
        _libc_free((unsigned __int64)v195);
      if ( v367[0] )
        sub_161E7C0((__int64)v367, v367[0]);
      sub_143AA50(v190, (__int64)&src);
      v196 = v389;
      src = &unk_49ECF68;
      v130 = &v389[88 * (unsigned int)v390];
      if ( v389 != (_BYTE *)v130 )
      {
        do
        {
          v130 -= 11;
          v197 = (_QWORD *)v130[4];
          if ( v197 != v130 + 6 )
            j_j___libc_free_0(v197, v130[6] + 1LL);
          if ( (_QWORD *)*v130 != v130 + 2 )
            j_j___libc_free_0(*v130, v130[2] + 1LL);
        }
        while ( v196 != v130 );
        goto LABEL_196;
      }
      goto LABEL_197;
    }
LABEL_270:
    if ( (_BYTE *)v365[0] != v366 )
      _libc_free(v365[0]);
LABEL_97:
    v76 = v344;
    v77 = v343;
    if ( v344 != v343 )
    {
      do
      {
        if ( *(_QWORD *)v77 )
          j_j___libc_free_0(*(_QWORD *)v77, *((_QWORD *)v77 + 2) - *(_QWORD *)v77);
        v77 += 24;
      }
      while ( v76 != v77 );
      v77 = v343;
    }
    if ( v77 )
      j_j___libc_free_0(v77, v345 - v77);
LABEL_104:
    v68 = v362;
LABEL_81:
    if ( v68 != v364 )
      _libc_free((unsigned __int64)v68);
LABEL_36:
    if ( v359 != v361 )
      _libc_free((unsigned __int64)v359);
    v45 = v411;
    if ( (_DWORD)v411 )
      continue;
    break;
  }
  v104 = v410;
LABEL_161:
  if ( v104 != v412 )
    _libc_free((unsigned __int64)v104);
  return v308;
}
