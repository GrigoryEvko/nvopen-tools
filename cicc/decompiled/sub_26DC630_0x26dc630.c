// Function: sub_26DC630
// Address: 0x26dc630
//
__int64 __fastcall sub_26DC630(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __m128i a5)
{
  __int64 *v5; // rbx
  void *v6; // r12
  __int64 v7; // rax
  volatile signed __int32 *v8; // rdx
  unsigned __int8 v9; // r10
  __int64 v10; // rax
  __int8 v11; // di
  __int64 v12; // r8
  __int64 *v13; // rdx
  _BYTE *v14; // rcx
  __int64 v15; // rdx
  void *v16; // rdi
  __m128i *v17; // r12
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // r12
  _QWORD *v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rbx
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rdi
  void *v26; // rdi
  _QWORD *v27; // rbx
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rbx
  unsigned __int64 v30; // rdi
  bool v32; // zf
  __int64 *v33; // r15
  __int64 v34; // r12
  __int64 v35; // rax
  unsigned int v36; // esi
  __int64 v37; // r8
  unsigned int v38; // edx
  __int64 *v39; // rcx
  __int64 v40; // rdi
  __int64 v41; // r13
  __int64 v42; // rbx
  const char *v43; // r14
  __int64 v44; // rdx
  size_t v45; // r12
  __int64 v46; // rax
  unsigned int v47; // esi
  __int64 v48; // r8
  unsigned int v49; // edx
  __int64 *v50; // rcx
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int128 v53; // rax
  const void *v54; // rax
  __int64 v55; // r13
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // rcx
  _QWORD *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  int v62; // ecx
  __int64 *v63; // rsi
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // rbx
  __int64 v68; // r12
  _BYTE *v69; // r13
  size_t v70; // r15
  int *v71; // r12
  unsigned __int64 v72; // rax
  __int64 v73; // r13
  __int64 v74; // rdx
  __int128 v75; // rax
  int *v76; // rax
  size_t v77; // rdx
  int *v78; // rcx
  size_t v79; // r13
  size_t v80; // r13
  int *v81; // rdi
  __int64 v82; // rax
  __int64 *v83; // rdx
  __int64 *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r12
  size_t v87; // rdx
  __int64 *v88; // rax
  __int64 v89; // rbx
  unsigned __int64 v90; // rbx
  char *v91; // rax
  void *v92; // r14
  char *v93; // r13
  __int64 i; // r12
  __int64 *v95; // r14
  __int64 v96; // r12
  __int64 v97; // rdx
  _QWORD *v98; // rax
  _QWORD *v99; // rdx
  char v100; // al
  __int64 v101; // r13
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // r12
  int v105; // eax
  unsigned int v106; // ecx
  __int64 v107; // rdx
  _QWORD *v108; // rax
  _QWORD *jj; // rdx
  __int64 v110; // rsi
  __int64 v111; // rdx
  __int64 v112; // r12
  __int64 v113; // rbx
  _QWORD *v114; // rdi
  int v115; // eax
  _QWORD *v116; // rax
  int *v117; // rcx
  __int64 *v118; // rsi
  void *v119; // r13
  _QWORD *v120; // r13
  int *v121; // rax
  size_t v122; // rdx
  void *v123; // rax
  size_t v124; // rdx
  char *v125; // rax
  char *v126; // rbx
  unsigned int v127; // ecx
  unsigned int v128; // eax
  int v129; // r13d
  unsigned int v130; // eax
  __int64 v131; // rbx
  char *v132; // rax
  __int64 v133; // rdx
  size_t v134; // r13
  __int64 v135; // rbx
  int v136; // eax
  char *v137; // rdi
  int v138; // r9d
  int v139; // ecx
  unsigned int ii; // r8d
  __int64 v141; // rax
  const void *v142; // rsi
  unsigned int v143; // r8d
  __int64 v144; // rax
  __int64 v145; // rdx
  __int64 v146; // rdi
  unsigned int v147; // eax
  __int64 v148; // rsi
  const void *v149; // rbx
  int v150; // eax
  int v151; // eax
  char *v152; // rdx
  _QWORD *v153; // rax
  unsigned __int64 v154; // rdi
  unsigned __int64 v155; // r10
  unsigned __int64 v156; // r9
  _QWORD *v157; // r11
  unsigned __int64 v158; // rsi
  _QWORD *v159; // rax
  _QWORD *v160; // r8
  int v161; // eax
  bool v162; // al
  _QWORD *v163; // rdi
  __m128i *v164; // rax
  __m128i si128; // xmm0
  __m128i v166; // xmm0
  __int64 v167; // r12
  __int64 v168; // rax
  unsigned __int64 v169; // rbx
  __int64 k; // r13
  __int64 v171; // r12
  __int64 v172; // r14
  __int64 v173; // rdx
  __int64 v174; // r15
  __int128 v175; // rax
  __int64 v176; // rdx
  int *v177; // rsi
  __int64 *v178; // r13
  unsigned __int64 v179; // rbx
  _QWORD *v180; // r14
  __int64 v181; // r15
  __int64 v182; // r9
  __int64 v183; // rax
  __int64 v184; // rsi
  __int64 v185; // rdx
  __int64 *v186; // r15
  __int64 v187; // r10
  size_t v188; // rdx
  __int64 v189; // rdi
  __int64 (__fastcall *v190)(__int64 *, __int64); // rax
  unsigned __int8 *v191; // rsi
  __int64 v192; // r12
  __int64 v193; // r15
  __int64 v194; // rbx
  unsigned __int64 v195; // rax
  __int64 v196; // rcx
  int v197; // eax
  __int64 v198; // rax
  __int64 v199; // rax
  __int64 v200; // rdi
  unsigned __int8 v201; // al
  __int64 v202; // rdx
  unsigned int v203; // esi
  int v204; // eax
  int v205; // esi
  _QWORD *v206; // rax
  __int64 v207; // rsi
  unsigned __int8 *v208; // rsi
  _QWORD *v209; // r12
  size_t v210; // rbx
  _QWORD *v211; // rdi
  _QWORD *v212; // rsi
  size_t v213; // rsi
  int v214; // r11d
  __int64 v215; // r10
  int v216; // eax
  int v217; // edx
  __int64 v218; // rax
  volatile signed __int32 *v219; // rdi
  __int64 *v220; // rax
  __int64 *v221; // r12
  __int64 v222; // rdi
  __int64 *v223; // rbx
  __int64 v224; // rsi
  __int64 *v225; // rsi
  __m128i *v226; // rdi
  _BYTE *v227; // rdx
  _BYTE *v228; // rax
  _BYTE *v229; // r13
  unsigned __int64 v230; // r13
  __int64 ***v231; // rax
  __int64 ***v232; // r9
  signed __int64 v233; // r13
  __int64 v234; // r13
  __int64 ***m; // rbx
  size_t v236; // r15
  __int64 *v237; // r8
  __int64 *v238; // rax
  void *v239; // r15
  __int64 *v240; // rdx
  __int64 *v241; // rax
  __int64 v242; // rcx
  __int64 v243; // rsi
  _QWORD *v244; // r13
  unsigned __int64 v245; // r14
  unsigned __int64 v246; // r12
  unsigned __int64 v247; // rdi
  unsigned __int64 v248; // r12
  unsigned __int64 v249; // rdi
  int v250; // r8d
  __int64 v251; // rsi
  __int64 v252; // rax
  __int64 v253; // rax
  __int64 v254; // r12
  __int64 v255; // rax
  __int64 v256; // rax
  __int64 *v257; // r15
  __int64 v258; // r12
  __int64 **j; // r13
  __int64 *v260; // r14
  unsigned __int64 v261; // r15
  __int64 v262; // r13
  __int64 v263; // rax
  __int64 *v264; // r15
  __int64 v265; // r14
  __int64 v266; // rbx
  char v267; // di
  __int64 v268; // r9
  __int64 v269; // rax
  __int64 v270; // rdx
  __int64 v271; // rax
  unsigned int v272; // ecx
  _QWORD *v273; // r12
  __int64 v274; // r13
  __int64 v275; // rax
  __int64 v276; // rdx
  __int64 v277; // r13
  unsigned int v278; // eax
  _QWORD *v279; // r15
  _QWORD *v280; // r12
  __int64 v281; // r14
  __int64 v282; // rax
  __int64 v283; // rdx
  __int64 v284; // r14
  unsigned int v285; // eax
  _QWORD *v286; // r9
  _QWORD *v287; // r12
  __int64 v288; // r13
  __int64 v289; // rax
  __int64 v290; // rdx
  __int64 v291; // r15
  unsigned int v292; // eax
  __int64 v293; // r8
  __int64 v294; // r15
  __int64 v295; // r12
  __int64 v296; // rax
  int v297; // r11d
  __int64 v298; // r10
  int v299; // eax
  int v300; // eax
  __int64 v301; // rax
  int v302; // r10d
  char *v303; // rsi
  unsigned int v304; // eax
  int v305; // r13d
  unsigned int v306; // eax
  _QWORD *v307; // [rsp+0h] [rbp-950h]
  __int64 v308; // [rsp+8h] [rbp-948h]
  _QWORD *v309; // [rsp+10h] [rbp-940h]
  _QWORD *v310; // [rsp+20h] [rbp-930h]
  __int64 v311; // [rsp+28h] [rbp-928h]
  __int64 v312; // [rsp+30h] [rbp-920h]
  __int64 v313; // [rsp+38h] [rbp-918h]
  __int64 *v314; // [rsp+40h] [rbp-910h]
  __int64 *v315; // [rsp+48h] [rbp-908h]
  __int64 *v316; // [rsp+58h] [rbp-8F8h]
  _QWORD *v318; // [rsp+68h] [rbp-8E8h]
  _QWORD *v319; // [rsp+78h] [rbp-8D8h]
  unsigned int v320; // [rsp+80h] [rbp-8D0h]
  int v321; // [rsp+80h] [rbp-8D0h]
  __int64 v322; // [rsp+80h] [rbp-8D0h]
  void *v324; // [rsp+98h] [rbp-8B8h]
  char *v325; // [rsp+98h] [rbp-8B8h]
  char *v326; // [rsp+98h] [rbp-8B8h]
  unsigned int v327; // [rsp+98h] [rbp-8B8h]
  _QWORD *v328; // [rsp+98h] [rbp-8B8h]
  __int64 v330; // [rsp+A8h] [rbp-8A8h]
  char v331; // [rsp+A8h] [rbp-8A8h]
  unsigned __int64 v332; // [rsp+A8h] [rbp-8A8h]
  __int64 v333; // [rsp+A8h] [rbp-8A8h]
  __int64 *v334; // [rsp+B0h] [rbp-8A0h]
  __int64 v335; // [rsp+B0h] [rbp-8A0h]
  int *v336; // [rsp+B0h] [rbp-8A0h]
  int *v337; // [rsp+B0h] [rbp-8A0h]
  _QWORD *v338; // [rsp+B0h] [rbp-8A0h]
  size_t v339; // [rsp+B0h] [rbp-8A0h]
  __int64 v340; // [rsp+B0h] [rbp-8A0h]
  __int64 v341; // [rsp+B8h] [rbp-898h]
  __int64 v342; // [rsp+C0h] [rbp-890h]
  __int64 *v343; // [rsp+C0h] [rbp-890h]
  __int64 *v344; // [rsp+C0h] [rbp-890h]
  int *v345; // [rsp+C0h] [rbp-890h]
  __int64 v346; // [rsp+C0h] [rbp-890h]
  __int64 v347; // [rsp+C8h] [rbp-888h]
  __int64 *v348; // [rsp+D0h] [rbp-880h]
  __int64 v349; // [rsp+D0h] [rbp-880h]
  int v350; // [rsp+D0h] [rbp-880h]
  __int64 v351; // [rsp+D0h] [rbp-880h]
  size_t v352; // [rsp+D0h] [rbp-880h]
  int v353; // [rsp+D0h] [rbp-880h]
  __int64 v354; // [rsp+D0h] [rbp-880h]
  __int64 v355[2]; // [rsp+E0h] [rbp-870h] BYREF
  __int64 v356[2]; // [rsp+F0h] [rbp-860h] BYREF
  char *v357; // [rsp+100h] [rbp-850h]
  void *src; // [rsp+110h] [rbp-840h] BYREF
  __int64 *v359; // [rsp+118h] [rbp-838h]
  __int64 *v360; // [rsp+120h] [rbp-830h]
  __m128i v361; // [rsp+130h] [rbp-820h] BYREF
  char *v362; // [rsp+140h] [rbp-810h]
  __int64 (__fastcall *v363)(__int64 *, __int64); // [rsp+148h] [rbp-808h]
  __m128i v364; // [rsp+150h] [rbp-800h] BYREF
  __int64 (__fastcall *v365)(__m128i *, __m128i *, int); // [rsp+160h] [rbp-7F0h]
  __int64 (__fastcall *v366)(__int64 *, __int64); // [rsp+168h] [rbp-7E8h]
  __int64 v367; // [rsp+170h] [rbp-7E0h]
  unsigned __int64 v368; // [rsp+178h] [rbp-7D8h]
  __int64 v369; // [rsp+180h] [rbp-7D0h]
  __int64 v370; // [rsp+188h] [rbp-7C8h]
  void *v371; // [rsp+190h] [rbp-7C0h] BYREF
  _BYTE *v372; // [rsp+198h] [rbp-7B8h]
  __int64 v373; // [rsp+1A0h] [rbp-7B0h]
  unsigned __int64 v374; // [rsp+1A8h] [rbp-7A8h]
  __int64 v375; // [rsp+1B0h] [rbp-7A0h]
  __int64 v376; // [rsp+1B8h] [rbp-798h]
  void *s1; // [rsp+1C0h] [rbp-790h] BYREF
  size_t n[2]; // [rsp+1C8h] [rbp-788h] BYREF
  __int64 (__fastcall *v379)(__int64 *, __int64); // [rsp+1D8h] [rbp-778h]
  __int64 v380; // [rsp+1E0h] [rbp-770h]
  __m128i *v381; // [rsp+1E8h] [rbp-768h]
  char v382; // [rsp+1F0h] [rbp-760h] BYREF
  char *v383[13]; // [rsp+1F8h] [rbp-758h] BYREF
  __m128i v384; // [rsp+260h] [rbp-6F0h] BYREF
  _QWORD *v385; // [rsp+270h] [rbp-6E0h]
  __int64 v386; // [rsp+278h] [rbp-6D8h]
  unsigned int v387; // [rsp+280h] [rbp-6D0h]
  char v388; // [rsp+6C8h] [rbp-288h]
  __int64 v389; // [rsp+6D0h] [rbp-280h]
  char v390[8]; // [rsp+6D8h] [rbp-278h] BYREF
  char v391; // [rsp+6E0h] [rbp-270h] BYREF
  __int64 v392; // [rsp+708h] [rbp-248h]
  __int64 v393; // [rsp+710h] [rbp-240h]
  __int64 *v394; // [rsp+760h] [rbp-1F0h]
  __int64 v395; // [rsp+768h] [rbp-1E8h]
  void *v396; // [rsp+770h] [rbp-1E0h] BYREF
  unsigned __int64 v397; // [rsp+778h] [rbp-1D8h]
  _QWORD *v398; // [rsp+780h] [rbp-1D0h]
  __int64 v399; // [rsp+788h] [rbp-1C8h]
  char v400; // [rsp+7A0h] [rbp-1B0h] BYREF
  void *s; // [rsp+7A8h] [rbp-1A8h]
  __int64 v402; // [rsp+7B0h] [rbp-1A0h]
  _QWORD *v403; // [rsp+7B8h] [rbp-198h]
  __int64 v404; // [rsp+7C0h] [rbp-190h]
  char v405; // [rsp+7D8h] [rbp-178h] BYREF
  _BYTE v406[16]; // [rsp+7E0h] [rbp-170h] BYREF
  void (__fastcall *v407)(_BYTE *, _BYTE *, __int64); // [rsp+7F0h] [rbp-160h]
  _BYTE v408[16]; // [rsp+800h] [rbp-150h] BYREF
  void (__fastcall *v409)(_BYTE *, _BYTE *, __int64); // [rsp+810h] [rbp-140h]
  _BYTE v410[16]; // [rsp+820h] [rbp-130h] BYREF
  void (__fastcall *v411)(_BYTE *, _BYTE *, __int64); // [rsp+830h] [rbp-120h]
  __int64 v412; // [rsp+840h] [rbp-110h]
  unsigned __int64 v413; // [rsp+848h] [rbp-108h]
  __int64 *v414; // [rsp+858h] [rbp-F8h]
  __int64 v415; // [rsp+868h] [rbp-E8h] BYREF
  __int64 v416; // [rsp+878h] [rbp-D8h]
  volatile signed __int32 *v417; // [rsp+880h] [rbp-D0h]
  __int64 v418; // [rsp+888h] [rbp-C8h]
  __int64 *v419; // [rsp+898h] [rbp-B8h]
  int v420; // [rsp+8A0h] [rbp-B0h]
  unsigned int v421; // [rsp+8A8h] [rbp-A8h]
  char v422; // [rsp+8B0h] [rbp-A0h] BYREF
  __int64 v423; // [rsp+8B8h] [rbp-98h]
  unsigned int v424; // [rsp+8C8h] [rbp-88h]
  char *v425; // [rsp+8D0h] [rbp-80h] BYREF
  unsigned int v426; // [rsp+8D8h] [rbp-78h]
  int v427; // [rsp+8DCh] [rbp-74h]
  __int64 v428; // [rsp+8F0h] [rbp-60h]
  unsigned int v429; // [rsp+900h] [rbp-50h]
  char v430; // [rsp+908h] [rbp-48h]
  __int64 v431; // [rsp+910h] [rbp-40h]
  __m128i *v432; // [rsp+918h] [rbp-38h]

  v5 = (__int64 *)a2;
  v6 = *(void **)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  if ( !*(_QWORD *)(a2 + 72) )
  {
    sub_CA41E0(&v384);
    v219 = *(volatile signed __int32 **)(a2 + 72);
    *(_QWORD *)(a2 + 72) = v384.m128i_i64[0];
    v384.m128i_i64[0] = (__int64)v219;
    if ( v219 )
    {
      if ( !_InterlockedSub(v219 + 2, 1u) )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v219 + 8LL))(v219);
    }
  }
  v7 = sub_BC0510(a4, qword_4F86C48, (__int64)a3);
  v8 = *(volatile signed __int32 **)(a2 + 72);
  s1 = v6;
  v379 = sub_26B9E50;
  v9 = *(_BYTE *)(a2 + 81);
  v10 = v7 + 8;
  n[1] = (size_t)sub_26B9F90;
  v11 = *(_BYTE *)(a2 + 80);
  v366 = sub_26B9E90;
  v365 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_26B9FC0;
  v363 = sub_26B9E70;
  v364.m128i_i64[0] = (__int64)v6;
  v361.m128i_i64[0] = (__int64)v6;
  v362 = (char *)sub_26B9FF0;
  src = (void *)v8;
  if ( v8 )
    _InterlockedAdd(v8 + 2, 1u);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = (__int64 *)(a2 + 32);
  if ( !v12 )
  {
    v12 = qword_4FF8370;
    v13 = &qword_4FF8368;
  }
  v14 = (_BYTE *)*v13;
  v15 = *(_QWORD *)(a2 + 8);
  if ( !v15 )
  {
    v15 = qword_4FF8470;
    v5 = &qword_4FF8468;
  }
  sub_26BD440(
    &v384,
    (_BYTE *)*v5,
    v15,
    v14,
    v12,
    *(_DWORD *)(a2 + 64),
    (__int64 *)&src,
    &v361,
    &v364,
    (__m128i *)&s1,
    v10,
    v11,
    v9);
  v16 = src;
  if ( src && !_InterlockedSub((volatile signed __int32 *)src + 2, 1u) )
    (*(void (__fastcall **)(void *))(*(_QWORD *)v16 + 8LL))(v16);
  if ( v362 )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v362)(&v361, &v361, 3);
  if ( v365 )
    v365(&v364, &v364, 3);
  if ( n[1] )
    ((void (__fastcall *)(void **, void **, __int64))n[1])(&s1, &s1, 3);
  if ( !(unsigned __int8)sub_26CE0F0(&v384, a3, (__int64)v6) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_19;
  }
  v356[1] = (__int64)a3;
  v348 = (__int64 *)(sub_BC0510(a4, &unk_4F87C68, (__int64)a3) + 8);
  v32 = *(_BYTE *)(v389 + 204) == 0;
  v356[0] = v389;
  v357 = &v422;
  if ( v32 )
    goto LABEL_72;
  v33 = (__int64 *)a3[4];
  v334 = a3 + 3;
  if ( v33 == a3 + 3 )
  {
    v303 = &v422;
    goto LABEL_71;
  }
  do
  {
    while ( 1 )
    {
      v41 = (__int64)(v33 - 7);
      if ( !v33 )
        v41 = 0;
      v42 = (__int64)v357;
      v43 = sub_BD5D20(v41);
      v45 = v44;
      v46 = sub_B2F650((__int64)v43, v44);
      n[0] = (size_t)v43;
      s1 = (void *)v46;
      n[1] = v45;
      v47 = *((_DWORD *)v357 + 6);
      if ( !v47 )
      {
        v364.m128i_i64[0] = 0;
        ++*(_QWORD *)v357;
        goto LABEL_330;
      }
      v48 = *((_QWORD *)v357 + 1);
      v49 = (v47 - 1) & (((0xBF58476D1CE4E5B9LL * v46) >> 31) ^ (484763065 * v46));
      v50 = (__int64 *)(v48 + 24LL * v49);
      v51 = *v50;
      if ( v46 != *v50 )
      {
        v214 = 1;
        v215 = 0;
        while ( v51 != -1 )
        {
          if ( v215 || v51 != -2 )
            v50 = (__int64 *)v215;
          v49 = (v47 - 1) & (v214 + v49);
          v51 = *(_QWORD *)(v48 + 24LL * v49);
          if ( v46 == v51 )
            goto LABEL_66;
          ++v214;
          v215 = (__int64)v50;
          v50 = (__int64 *)(v48 + 24LL * v49);
        }
        if ( !v215 )
          v215 = (__int64)v50;
        v364.m128i_i64[0] = v215;
        v216 = *((_DWORD *)v357 + 4);
        ++*(_QWORD *)v357;
        v217 = v216 + 1;
        if ( 4 * (v216 + 1) < 3 * v47 )
        {
          if ( v47 - *(_DWORD *)(v42 + 20) - v217 > v47 >> 3 )
          {
LABEL_323:
            *(_DWORD *)(v42 + 16) = v217;
            v218 = v364.m128i_i64[0];
            if ( *(_QWORD *)v364.m128i_i64[0] != -1 )
              --*(_DWORD *)(v42 + 20);
            *(_QWORD *)v218 = s1;
            *(__m128i *)(v218 + 8) = _mm_loadu_si128((const __m128i *)n);
            goto LABEL_66;
          }
LABEL_331:
          sub_9E2150(v42, v47);
          sub_26C5480(v42, (__int64 *)&s1, &v364);
          v217 = *(_DWORD *)(v42 + 16) + 1;
          goto LABEL_323;
        }
LABEL_330:
        v47 *= 2;
        goto LABEL_331;
      }
LABEL_66:
      s1 = (void *)sub_B2D7E0(v41, "sample-profile-suffix-elision-policy", 0x24u);
      v342 = sub_A72240((__int64 *)&s1);
      v347 = v52;
      *(_QWORD *)&v53 = sub_BD5D20(v41);
      v54 = (const void *)sub_C16140(v53, v342, v347);
      v55 = (__int64)v54;
      v57 = v56;
      if ( v56 == v45 )
      {
        if ( !v45 )
          goto LABEL_61;
        if ( !memcmp(v54, v43, v45) )
          break;
      }
      v34 = (__int64)v357;
      v35 = sub_B2F650(v55, v57);
      n[0] = v55;
      s1 = (void *)v35;
      n[1] = v57;
      v36 = *((_DWORD *)v357 + 6);
      if ( !v36 )
      {
        v364.m128i_i64[0] = 0;
        ++*(_QWORD *)v357;
        goto LABEL_497;
      }
      v37 = *((_QWORD *)v357 + 1);
      v38 = (v36 - 1) & (((0xBF58476D1CE4E5B9LL * v35) >> 31) ^ (484763065 * v35));
      v39 = (__int64 *)(v37 + 24LL * v38);
      v40 = *v39;
      if ( v35 != *v39 )
      {
        v297 = 1;
        v298 = 0;
        while ( v40 != -1 )
        {
          if ( v40 == -2 && !v298 )
            v298 = (__int64)v39;
          v38 = (v36 - 1) & (v297 + v38);
          v39 = (__int64 *)(v37 + 24LL * v38);
          v40 = *v39;
          if ( v35 == *v39 )
            goto LABEL_61;
          ++v297;
        }
        if ( !v298 )
          v298 = (__int64)v39;
        v364.m128i_i64[0] = v298;
        v299 = *((_DWORD *)v357 + 4);
        ++*(_QWORD *)v357;
        v300 = v299 + 1;
        if ( 4 * v300 < 3 * v36 )
        {
          if ( v36 - *(_DWORD *)(v34 + 20) - v300 > v36 >> 3 )
          {
LABEL_493:
            *(_DWORD *)(v34 + 16) = v300;
            v301 = v364.m128i_i64[0];
            if ( *(_QWORD *)v364.m128i_i64[0] != -1 )
              --*(_DWORD *)(v34 + 20);
            *(_QWORD *)v301 = s1;
            *(__m128i *)(v301 + 8) = _mm_loadu_si128((const __m128i *)n);
            goto LABEL_61;
          }
LABEL_498:
          sub_9E2150(v34, v36);
          sub_26C5480(v34, (__int64 *)&s1, &v364);
          v300 = *(_DWORD *)(v34 + 16) + 1;
          goto LABEL_493;
        }
LABEL_497:
        v36 *= 2;
        goto LABEL_498;
      }
LABEL_61:
      v33 = (__int64 *)v33[1];
      if ( v334 == v33 )
        goto LABEL_70;
    }
    v33 = (__int64 *)v33[1];
  }
  while ( v334 != v33 );
LABEL_70:
  v303 = v357;
LABEL_71:
  sub_26C7D10(v356, (__int64)v303);
LABEL_72:
  v394 = v348;
  if ( !sub_BAA6A0((__int64)a3, 0) )
  {
    v191 = (unsigned __int8 *)sub_BCA5C0(*(int **)(v389 + 80), (__int64 *)*a3, 1, 1);
    sub_BAA660((__int64 **)a3, v191, 2);
    sub_D84780(v394);
  }
  if ( unk_4F838D4 )
  {
    v58 = v389;
    v178 = v394;
    if ( *(_QWORD *)(v389 + 24) )
    {
      v354 = 0;
      v179 = 0;
      v180 = *(_QWORD **)(v389 + 24);
      do
      {
        v181 = v392;
        v182 = v180[5];
        v345 = (int *)v180[4];
        if ( v345 )
        {
          v339 = v180[5];
          sub_C7D030(&s1);
          sub_C7D280((int *)&s1, v345, v339);
          sub_C7D290(&s1, &v364);
          v182 = v364.m128i_i64[0];
        }
        v183 = *(unsigned int *)(v181 + 24);
        v184 = *(_QWORD *)(v181 + 8);
        if ( (_DWORD)v183 )
        {
          LODWORD(v185) = (v183 - 1) & (((0xBF58476D1CE4E5B9LL * v182) >> 31) ^ (484763065 * v182));
          v186 = (__int64 *)(v184 + 24LL * (unsigned int)v185);
          v187 = *v186;
          if ( v182 == *v186 )
          {
LABEL_252:
            if ( v186 != (__int64 *)(v184 + 24 * v183) && sub_D85370((__int64)v178, dword_4FF7108, v180[9]) )
            {
              ++v179;
              v354 += v186[2] != v180[3];
            }
          }
          else
          {
            v250 = 1;
            while ( v187 != -1 )
            {
              v185 = ((_DWORD)v183 - 1) & (unsigned int)(v185 + v250);
              v186 = (__int64 *)(v184 + 24 * v185);
              v187 = *v186;
              if ( v182 == *v186 )
                goto LABEL_252;
              ++v250;
            }
          }
        }
        v180 = (_QWORD *)*v180;
      }
      while ( v180 );
      if ( (unsigned int)qword_4FF7028 > v179 || (unsigned int)qword_4FF6F48 * v179 > 100 * v354 )
        goto LABEL_75;
    }
    else if ( (_DWORD)qword_4FF7028 )
    {
      goto LABEL_78;
    }
    v364.m128i_i64[0] = (__int64)"The input profile significantly mismatches current source code. Please recollect profil"
                                 "e to avoid performance regression.";
    v188 = a3[21];
    v189 = *a3;
    LOWORD(v367) = 259;
    v190 = (__int64 (__fastcall *)(__int64 *, __int64))a3[22];
    n[0] = 12;
    s1 = &unk_49D9C78;
    v379 = v190;
    n[1] = v188;
    LODWORD(v380) = 0;
    v381 = &v364;
    sub_B6EB20(v189, (__int64)&s1);
    v331 = 0;
    goto LABEL_143;
  }
LABEL_75:
  v58 = v389;
  v59 = *(_QWORD **)(v389 + 24);
  if ( v59 )
  {
    v60 = v418;
    do
    {
      v60 += v59[9];
      v418 = v60;
      v59 = (_QWORD *)*v59;
    }
    while ( v59 );
  }
LABEL_78:
  v330 = *(_QWORD *)(v58 + 88);
  v61 = a3[15];
  v62 = *(_DWORD *)(v61 + 8);
  if ( v62 )
  {
    v63 = *(__int64 **)v61;
    v64 = **(_QWORD **)v61;
    if ( v64 && v64 != -8 )
    {
      v67 = v63;
    }
    else
    {
      v65 = v63 + 1;
      do
      {
        do
        {
          v66 = *v65;
          v67 = v65++;
        }
        while ( !v66 );
      }
      while ( v66 == -8 );
    }
    if ( &v63[v62] != v67 )
    {
      v343 = &v63[v62];
      while ( 2 )
      {
        v68 = *v67;
        v69 = *(_BYTE **)(*v67 + 8);
        v70 = *(_QWORD *)*v67;
        if ( *v69 )
          goto LABEL_96;
        v355[0] = *(_QWORD *)(*v67 + 8);
        if ( !v70 )
          goto LABEL_96;
        v71 = (int *)(v68 + 16);
        sub_C7D030(&s1);
        sub_C7D280((int *)&s1, v71, v70);
        sub_C7D290(&s1, &v364);
        s1 = (void *)v364.m128i_i64[0];
        v72 = (unsigned __int64)sub_26C56D0(&v396, (__int64 *)&s1);
        if ( !v72 )
        {
          v153 = (_QWORD *)sub_22077B0(0x18u);
          v154 = (unsigned __int64)v153;
          if ( v153 )
            *v153 = 0;
          v155 = (unsigned __int64)s1;
          v156 = v397;
          v153[2] = 0;
          v153[1] = v155;
          v157 = (_QWORD *)*((_QWORD *)v396 + v155 % v156);
          v158 = v155 % v156;
          if ( v157 )
          {
            v159 = (_QWORD *)*v157;
            if ( *(_QWORD *)(*v157 + 8LL) == v155 )
            {
LABEL_217:
              if ( *v157 )
              {
                v338 = (_QWORD *)*v157;
                j_j___libc_free_0(v154);
                v72 = (unsigned __int64)v338;
                goto LABEL_89;
              }
            }
            else
            {
              while ( 1 )
              {
                v160 = (_QWORD *)*v159;
                if ( !*v159 )
                  break;
                v157 = v159;
                if ( v158 != v160[1] % v156 )
                  break;
                v159 = (_QWORD *)*v159;
                if ( v160[1] == v155 )
                  goto LABEL_217;
              }
            }
          }
          v72 = sub_26BAF80((unsigned __int64 *)&v396, v158, v155, v154, 1);
        }
LABEL_89:
        *(_QWORD *)(v72 + 16) = v69;
        v73 = v355[0];
        s1 = (void *)sub_B2D7E0(v355[0], "sample-profile-suffix-elision-policy", 0x24u);
        v335 = sub_A72240((__int64 *)&s1);
        v341 = v74;
        *(_QWORD *)&v75 = sub_BD5D20(v73);
        v76 = (int *)sub_C16140(v75, v335, v341);
        v78 = v76;
        v79 = v77;
        if ( v70 == v77 )
        {
          v336 = v76;
          v115 = memcmp(v71, v76, v70);
          v78 = v336;
          if ( !v115 )
          {
LABEL_91:
            if ( v330 )
            {
              sub_C21B60((__int64)&s1, v330, (__int64)v71, v70);
              if ( LOBYTE(n[1]) )
              {
                v80 = n[0];
                v81 = (int *)s1;
                if ( v70 == n[0] )
                {
                  if ( n[0] )
                  {
                    v81 = (int *)s1;
                    if ( memcmp(s1, v71, n[0]) )
                    {
LABEL_244:
                      src = (void *)sub_26BA4C0(v81, v80);
                      if ( !sub_26C56D0(&v396, (__int64 *)&src) )
                      {
                        v364.m128i_i64[0] = (__int64)v355;
                        v361.m128i_i64[0] = (__int64)&src;
                        sub_26BBF70((unsigned __int64 *)&v396, (unsigned __int64 **)&v361, &v364);
                      }
                    }
                  }
                }
                else if ( n[0] )
                {
                  goto LABEL_244;
                }
              }
            }
LABEL_96:
            v82 = v67[1];
            v83 = v67 + 1;
            if ( v82 != -8 && v82 )
            {
              ++v67;
              if ( v343 == v83 )
                goto LABEL_102;
            }
            else
            {
              v84 = v67 + 2;
              do
              {
                do
                {
                  v85 = *v84;
                  v67 = v84++;
                }
                while ( v85 == -8 );
              }
              while ( !v85 );
              if ( v343 == v67 )
                goto LABEL_102;
            }
            continue;
          }
        }
        else if ( !v77 )
        {
          goto LABEL_91;
        }
        break;
      }
      v337 = v78;
      v361.m128i_i64[0] = sub_26BA4C0(v78, v79);
      v116 = sub_26C56D0(&v396, v361.m128i_i64);
      v117 = v337;
      if ( v116 )
      {
        v116[2] = 0;
      }
      else
      {
        s1 = v355;
        v364.m128i_i64[0] = (__int64)&v361;
        sub_26BBF70((unsigned __int64 *)&v396, (unsigned __int64 **)&v364, (_QWORD **)&s1);
        v117 = v337;
      }
      v70 = v79;
      v71 = v117;
      goto LABEL_91;
    }
  }
LABEL_102:
  if ( LOBYTE(qword_4FF8040[17]) || LOBYTE(qword_4FF7F60[17]) || LOBYTE(qword_4FF8200[17]) )
  {
    sub_26E86E0(v432);
    sub_26C0800(v432);
  }
  src = 0;
  v359 = 0;
  v86 = v412;
  v360 = 0;
  v87 = (size_t)(a3 + 3);
  v88 = (__int64 *)a3[4];
  v316 = a3 + 3;
  if ( a3 + 3 != v88 )
  {
    v89 = 0;
    do
    {
      v88 = (__int64 *)v88[1];
      ++v89;
    }
    while ( (__int64 *)v87 != v88 );
    if ( v89 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::reserve");
    v90 = 8 * v89;
    v91 = (char *)sub_22077B0(v90);
    v92 = src;
    v93 = v91;
    v87 = (char *)v359 - (_BYTE *)src;
    if ( (char *)v359 - (_BYTE *)src > 0 )
    {
      memmove(v91, src, v87);
    }
    else if ( !src )
    {
      goto LABEL_111;
    }
    j_j___libc_free_0((unsigned __int64)v92);
LABEL_111:
    src = v93;
    v359 = (__int64 *)v93;
    v360 = (__int64 *)&v93[v90];
  }
  if ( !byte_4FF7B88 )
  {
    if ( !(_BYTE)qword_4FF7AA8 )
      goto LABEL_114;
    v163 = sub_CB72A0();
    v164 = (__m128i *)v163[4];
    v87 = v163[3] - (_QWORD)v164;
    if ( v87 <= 0x66 )
    {
      sub_CB6200(
        (__int64)v163,
        "WARNING: -use-profiled-call-graph ignored, should be used together with -sample-profile-top-down-load.\n",
        0x67u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43912B0);
      v164[6].m128i_i32[0] = 1634692141;
      v164[6].m128i_i16[2] = 11876;
      *v164 = si128;
      v166 = _mm_load_si128((const __m128i *)&xmmword_43912C0);
      v164[6].m128i_i8[6] = 10;
      v164[1] = v166;
      v164[2] = _mm_load_si128((const __m128i *)&xmmword_43912D0);
      v164[3] = _mm_load_si128((const __m128i *)&xmmword_43912E0);
      v164[4] = _mm_load_si128((const __m128i *)&xmmword_43912F0);
      a5 = _mm_load_si128((const __m128i *)&xmmword_4391300);
      v164[5] = a5;
      v163[4] += 103LL;
    }
    if ( !byte_4FF7B88 )
    {
LABEL_114:
      if ( byte_4FF7C68 )
      {
        LOBYTE(s1) = 0;
        byte_4FF7C68 = 0;
        sub_26C3780((__int64)&unk_4FF7C88, (__int64)&s1, v87);
      }
      for ( i = a3[4]; v316 != (__int64 *)i; i = *(_QWORD *)(i + 8) )
      {
        v119 = (void *)(i - 56);
        if ( !i )
          v119 = 0;
        if ( !sub_B2FC80((__int64)v119) && (unsigned __int8)sub_B2D620((__int64)v119, "use-sample-profile", 0x12u) )
        {
          s1 = v119;
          v118 = v359;
          if ( v359 == v360 )
          {
            sub_24147A0((__int64)&src, v359, &s1);
          }
          else
          {
            if ( v359 )
            {
              *v359 = (__int64)v119;
              v118 = v359;
            }
            v359 = v118 + 1;
          }
        }
      }
      goto LABEL_117;
    }
  }
  if ( !(_BYTE)qword_4FF7AA8 && (!unk_4F838D3 || (unsigned int)sub_23DF0D0(&dword_4FF7A28)) )
  {
    sub_26C5130(v86, (__int64 *)&src);
    goto LABEL_117;
  }
  if ( !unk_4F838D3 )
  {
    v254 = v389;
    v255 = sub_22077B0(0x78u);
    v169 = v255;
    if ( !v255 )
      goto LABEL_232;
    *(_DWORD *)(v255 + 24) = 0;
    v256 = v255 + 24;
    *(_QWORD *)(v256 + 8) = 0;
    *(_QWORD *)(v256 - 24) = 0;
    *(_QWORD *)(v256 - 16) = 0;
    *(_QWORD *)(v169 + 40) = v256;
    *(_QWORD *)(v169 + 48) = v256;
    *(_QWORD *)(v169 + 56) = 0;
    *(_QWORD *)(v169 + 72) = v169 + 64;
    *(_QWORD *)(v169 + 64) = v169 + 64;
    *(_QWORD *)(v169 + 80) = 0;
    *(_QWORD *)(v169 + 88) = 0;
    *(_QWORD *)(v169 + 96) = 0;
    *(_QWORD *)(v169 + 104) = 0;
    *(_DWORD *)(v169 + 112) = 0;
    v257 = *(__int64 **)(v254 + 24);
    if ( !v257 )
      goto LABEL_232;
    while ( 1 )
    {
      sub_26D04C0(v169, (int *)v257[4], v257[5]);
      v258 = v257[14];
      if ( (__int64 *)v258 != v257 + 12 )
      {
        do
        {
          for ( j = *(__int64 ***)(v258 + 64); j; j = (__int64 **)*j )
          {
            sub_26D04C0(v169, (int *)j[1], (__int64)j[2]);
            sub_26C4ED0(v169, (int *)v257[4], v257[5], (int *)j[1], (size_t)j[2], (__int64)j[3]);
          }
          v258 = sub_220EF30(v258);
        }
        while ( v257 + 12 != (__int64 *)v258 );
      }
      v315 = v257 + 18;
      v322 = v257[20];
      if ( (__int64 *)v322 != v257 + 18 )
        break;
LABEL_465:
      v257 = (__int64 *)*v257;
      if ( !v257 )
        goto LABEL_232;
    }
    v260 = v257;
    v261 = v169;
LABEL_415:
    v262 = *(_QWORD *)(v322 + 64);
    if ( v262 == v322 + 48 )
      goto LABEL_463;
    v263 = v261;
    v264 = v260;
    v265 = v263;
    while ( 1 )
    {
      v266 = v262 + 32;
      sub_26D04C0(v265, *(int **)(v262 + 32), *(_QWORD *)(v262 + 40));
      v346 = v262 + 48;
      v267 = unk_4F838D3;
      if ( unk_4F838D3 )
      {
        v268 = *(_QWORD *)(v262 + 112);
        if ( v268 )
          goto LABEL_461;
      }
      v269 = *(_QWORD *)(v262 + 208);
      if ( *(_QWORD *)(v262 + 160) )
      {
        v270 = *(_QWORD *)(v262 + 144);
        if ( !v269
          || (v271 = *(_QWORD *)(v262 + 192), v272 = *(_DWORD *)(v271 + 32), *(_DWORD *)(v270 + 32) < v272)
          || *(_DWORD *)(v270 + 32) == v272 && *(_DWORD *)(v270 + 36) < *(_DWORD *)(v271 + 36) )
        {
          v268 = *(_QWORD *)(v270 + 40);
LABEL_460:
          if ( v268 )
            goto LABEL_461;
          goto LABEL_467;
        }
      }
      else
      {
        if ( !v269 )
          goto LABEL_467;
        v271 = *(_QWORD *)(v262 + 192);
      }
      v273 = *(_QWORD **)(v271 + 64);
      v328 = (_QWORD *)(v271 + 48);
      if ( v273 != (_QWORD *)(v271 + 48) )
      {
        v314 = v264;
        v313 = v262;
        v312 = v265;
        v333 = 0;
        v311 = v262 + 32;
        while ( 2 )
        {
          if ( v267 )
          {
            v274 = v273[14];
            if ( v274 )
              goto LABEL_458;
          }
          v275 = v273[26];
          if ( v273[20] )
          {
            v276 = v273[18];
            if ( !v275
              || (v277 = v273[24], v278 = *(_DWORD *)(v277 + 32), *(_DWORD *)(v276 + 32) < v278)
              || *(_DWORD *)(v276 + 32) == v278 && *(_DWORD *)(v276 + 36) < *(_DWORD *)(v277 + 36) )
            {
              v274 = *(_QWORD *)(v276 + 40);
LABEL_457:
              if ( v274 )
                goto LABEL_458;
LABEL_468:
              v274 = v273[13] != 0;
LABEL_458:
              v333 += v274;
              v273 = (_QWORD *)sub_220EF30((__int64)v273);
              if ( v328 == v273 )
              {
                v264 = v314;
                v262 = v313;
                v265 = v312;
                v266 = v311;
                v268 = v333;
                goto LABEL_460;
              }
              continue;
            }
          }
          else
          {
            if ( !v275 )
              goto LABEL_468;
            v277 = v273[24];
          }
          break;
        }
        v279 = *(_QWORD **)(v277 + 64);
        v319 = (_QWORD *)(v277 + 48);
        if ( v279 == (_QWORD *)(v277 + 48) )
          goto LABEL_468;
        v310 = v273;
        v274 = 0;
        v280 = v279;
        while ( 2 )
        {
          if ( v267 )
          {
            v281 = v280[14];
            if ( v281 )
            {
LABEL_455:
              v274 += v281;
              v280 = (_QWORD *)sub_220EF30((__int64)v280);
              if ( v319 == v280 )
              {
                v273 = v310;
                goto LABEL_457;
              }
              continue;
            }
          }
          break;
        }
        v282 = v280[26];
        if ( v280[20] )
        {
          v283 = v280[18];
          if ( !v282
            || (v284 = v280[24], v285 = *(_DWORD *)(v284 + 32), *(_DWORD *)(v283 + 32) < v285)
            || *(_DWORD *)(v283 + 32) == v285 && *(_DWORD *)(v283 + 36) < *(_DWORD *)(v284 + 36) )
          {
            v281 = *(_QWORD *)(v283 + 40);
LABEL_454:
            if ( v281 )
              goto LABEL_455;
LABEL_471:
            v281 = v280[13] != 0;
            goto LABEL_455;
          }
        }
        else
        {
          if ( !v282 )
            goto LABEL_471;
          v284 = v280[24];
        }
        v286 = *(_QWORD **)(v284 + 64);
        v318 = (_QWORD *)(v284 + 48);
        if ( v286 == (_QWORD *)(v284 + 48) )
          goto LABEL_471;
        v308 = v274;
        v281 = 0;
        v309 = v280;
        v287 = v286;
        while ( 2 )
        {
          if ( v267 )
          {
            v288 = v287[14];
            if ( v288 )
              goto LABEL_452;
          }
          v289 = v287[26];
          if ( v287[20] )
          {
            v290 = v287[18];
            if ( !v289
              || (v291 = v287[24], v292 = *(_DWORD *)(v291 + 32), *(_DWORD *)(v290 + 32) < v292)
              || *(_DWORD *)(v290 + 32) == v292 && *(_DWORD *)(v290 + 36) < *(_DWORD *)(v291 + 36) )
            {
              v288 = *(_QWORD *)(v290 + 40);
              goto LABEL_451;
            }
LABEL_447:
            v293 = *(_QWORD *)(v291 + 64);
            v294 = v291 + 48;
            if ( v293 != v294 )
            {
              v307 = v287;
              v288 = 0;
              v295 = v293;
              do
              {
                v288 += sub_EF9210((_QWORD *)(v295 + 48));
                v295 = sub_220EF30(v295);
              }
              while ( v294 != v295 );
              v287 = v307;
LABEL_451:
              if ( v288 )
              {
LABEL_452:
                v281 += v288;
                v287 = (_QWORD *)sub_220EF30((__int64)v287);
                if ( v318 == v287 )
                {
                  v280 = v309;
                  v274 = v308;
                  goto LABEL_454;
                }
                continue;
              }
            }
          }
          else if ( v289 )
          {
            v291 = v287[24];
            goto LABEL_447;
          }
          break;
        }
        v288 = v287[13] != 0;
        goto LABEL_452;
      }
LABEL_467:
      v268 = *(_QWORD *)(v262 + 104) != 0;
LABEL_461:
      sub_26C4ED0(v265, (int *)v264[4], v264[5], *(int **)v266, *(_QWORD *)(v266 + 8), v268);
      sub_26D1050(v265, v346);
      v262 = sub_220EF30(v262);
      if ( v322 + 48 == v262 )
      {
        v296 = v265;
        v260 = v264;
        v261 = v296;
LABEL_463:
        v322 = sub_220EF30(v322);
        if ( v315 == (__int64 *)v322 )
        {
          v169 = v261;
          v257 = v260;
          goto LABEL_465;
        }
        goto LABEL_415;
      }
    }
  }
  v167 = v413;
  v168 = sub_22077B0(0x78u);
  v169 = v168;
  if ( v168 )
    sub_26D0850(v168, v167, 0);
LABEL_232:
  for ( k = a3[4]; v316 != (__int64 *)k; k = *(_QWORD *)(k + 8) )
  {
    v171 = k - 56;
    if ( !k )
      v171 = 0;
    if ( !sub_B2FC80(v171) && (unsigned __int8)sub_B2D620(v171, "use-sample-profile", 0x12u) )
    {
      s1 = (void *)sub_B2D7E0(v171, "sample-profile-suffix-elision-policy", 0x24u);
      v172 = sub_A72240((__int64 *)&s1);
      v174 = v173;
      *(_QWORD *)&v175 = sub_BD5D20(v171);
      v177 = (int *)sub_C16140(v175, v172, v174);
      if ( v176 && unk_4F838D1 )
      {
        v253 = sub_B2F650((__int64)v177, v176);
        v177 = 0;
        v176 = v253;
      }
      sub_26D04C0(v169, v177, v176);
    }
  }
  v225 = (__int64 *)v169;
  v364 = 0u;
  v365 = 0;
  v366 = 0;
  v367 = 0;
  v368 = 0;
  v369 = 0;
  v370 = 0;
  v371 = 0;
  v372 = 0;
  v373 = 0;
  v374 = 0;
  v375 = 0;
  v376 = 0;
  sub_26D4D40(v364.m128i_i32, v169);
  v226 = &v364;
  sub_26D5040((__int64)&v364);
  v227 = v372;
  v228 = v371;
  if ( v372 != v371 )
  {
    v332 = v169;
    v229 = v372;
    do
    {
      v361 = 0u;
      v230 = v229 - v228;
      v362 = 0;
      if ( v230 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v226, v225, v227);
      v231 = (__int64 ***)sub_22077B0(v230);
      v225 = (__int64 *)v371;
      v361.m128i_i64[0] = (__int64)v231;
      v232 = v231;
      v361.m128i_i64[1] = (__int64)v231;
      v362 = (char *)v231 + v230;
      v233 = v372 - (_BYTE *)v371;
      if ( v372 != v371 )
        v232 = (__int64 ***)memmove(v231, v371, v372 - (_BYTE *)v371);
      v234 = (__int64)v232 + v233;
      v361.m128i_i64[1] = v234;
      if ( LOBYTE(qword_4FF7780[17]) )
      {
        sub_26D3D20((__int64)&s1, (char **)&v371);
        v225 = (__int64 *)v383;
        sub_26BA800((__int64)&v361, v383);
        if ( v383[0] )
        {
          v225 = (__int64 *)(v383[2] - v383[0]);
          j_j___libc_free_0((unsigned __int64)v383[0]);
        }
        sub_26C37A0((__int64)&s1);
        if ( s1 != &v382 )
        {
          v225 = (__int64 *)(8 * n[0]);
          j_j___libc_free_0((unsigned __int64)s1);
        }
        v234 = v361.m128i_i64[1];
        v232 = (__int64 ***)v361.m128i_i64[0];
      }
      for ( m = v232; (__int64 ***)v234 != m; ++m )
      {
        v225 = **m;
        v236 = (size_t)(*m)[1];
        if ( v225 )
        {
          sub_C7D030(&s1);
          sub_C7D280((int *)&s1, (int *)v225, v236);
          v225 = v355;
          sub_C7D290(&s1, v355);
          v236 = v355[0];
        }
        v237 = (__int64 *)*((_QWORD *)v396 + v236 % v397);
        if ( v237 )
        {
          v238 = (__int64 *)*v237;
          if ( v236 == *(_QWORD *)(*v237 + 8) )
          {
LABEL_360:
            if ( *v237 )
            {
              v239 = *(void **)(*v237 + 16);
              s1 = v239;
              if ( v239 )
              {
                if ( !sub_B2FC80((__int64)v239) )
                {
                  v225 = (__int64 *)"use-sample-profile";
                  if ( (unsigned __int8)sub_B2D620((__int64)v239, "use-sample-profile", 0x12u) )
                  {
                    v225 = v359;
                    if ( v359 == v360 )
                    {
                      sub_9CC5C0((__int64)&src, v359, &s1);
                    }
                    else
                    {
                      if ( v359 )
                      {
                        *v359 = (__int64)s1;
                        v225 = v359;
                      }
                      v359 = ++v225;
                    }
                  }
                }
              }
            }
          }
          else
          {
            while ( 1 )
            {
              v225 = (__int64 *)*v238;
              if ( !*v238 )
                break;
              v237 = v238;
              if ( v236 % v397 != v225[1] % v397 )
                break;
              v238 = (__int64 *)*v238;
              if ( v236 == v225[1] )
                goto LABEL_360;
            }
          }
        }
      }
      sub_26D5040((__int64)&v364);
      v226 = (__m128i *)v361.m128i_i64[0];
      if ( v361.m128i_i64[0] )
      {
        v225 = (__int64 *)&v362[-v361.m128i_i64[0]];
        j_j___libc_free_0(v361.m128i_u64[0]);
      }
      v229 = v372;
      v228 = v371;
    }
    while ( v372 != v371 );
    v169 = v332;
  }
  v240 = (__int64 *)src;
  if ( v359 != src )
  {
    v241 = v359 - 1;
    if ( src < v359 - 1 )
    {
      do
      {
        v242 = *v240;
        v243 = *v241;
        ++v240;
        --v241;
        *(v240 - 1) = v243;
        v241[1] = v242;
      }
      while ( v240 < v241 );
    }
  }
  if ( v374 )
    j_j___libc_free_0(v374);
  if ( v371 )
    j_j___libc_free_0((unsigned __int64)v371);
  if ( v368 )
    j_j___libc_free_0(v368);
  sub_C7D6A0((__int64)v365, 16LL * (unsigned int)v367, 8);
  if ( v169 )
  {
    sub_C7D6A0(*(_QWORD *)(v169 + 96), 16LL * *(unsigned int *)(v169 + 112), 8);
    v244 = *(_QWORD **)(v169 + 64);
    while ( (_QWORD *)(v169 + 64) != v244 )
    {
      v245 = (unsigned __int64)v244;
      v244 = (_QWORD *)*v244;
      v246 = *(_QWORD *)(v245 + 48);
      while ( v246 )
      {
        sub_26BBDA0(*(_QWORD *)(v246 + 24));
        v247 = v246;
        v246 = *(_QWORD *)(v246 + 16);
        j_j___libc_free_0(v247);
      }
      j_j___libc_free_0(v245);
    }
    v248 = *(_QWORD *)(v169 + 32);
    while ( v248 )
    {
      sub_26BBDA0(*(_QWORD *)(v248 + 24));
      v249 = v248;
      v248 = *(_QWORD *)(v248 + 16);
      j_j___libc_free_0(v249);
    }
    j_j___libc_free_0(v169);
  }
LABEL_117:
  v344 = v359;
  if ( src != v359 )
  {
    v331 = 0;
    v95 = (__int64 *)src;
    while ( 1 )
    {
      v96 = *v95;
      sub_26C8090((__int64)&v384, 1);
      ++v384.m128i_i64[1];
      if ( (_DWORD)v386 )
        break;
      if ( HIDWORD(v386) )
      {
        v97 = v387;
        if ( v387 <= 0x40 )
          goto LABEL_122;
        sub_C7D6A0((__int64)v385, 16LL * v387, 8);
        v387 = 0;
LABEL_315:
        v385 = 0;
        goto LABEL_124;
      }
LABEL_125:
      v100 = qword_4FF7D48;
      if ( (_BYTE)qword_4FF7D48 )
        v100 = v416 != 0;
      v430 = v100;
      if ( (_BYTE)qword_4FF7F08 || (unsigned __int8)sub_B2D620(v96, "profile-sample-accurate", 0x17u) )
      {
        v430 = 0;
        v101 = 0;
        v388 = 0;
        goto LABEL_129;
      }
      v388 = v430;
      if ( !v430 )
        goto LABEL_198;
      v131 = v416;
      v132 = (char *)sub_BD5D20(v96);
      v134 = v133;
      v350 = *(_DWORD *)(v131 + 32);
      if ( !v350 )
      {
LABEL_200:
        v101 = -1;
        goto LABEL_201;
      }
      v325 = v132;
      v135 = *(_QWORD *)(v131 + 16);
      v136 = sub_C94890(v132, v133);
      v137 = v325;
      v138 = 1;
      v139 = v350 - 1;
      for ( ii = (v350 - 1) & v136; ; ii = v139 & v143 )
      {
        v141 = v135 + 16LL * ii;
        v142 = *(const void **)v141;
        if ( *(_QWORD *)v141 == -1 )
          break;
        if ( v142 == (const void *)-2LL )
        {
          v162 = v137 + 2 == 0;
        }
        else
        {
          if ( *(_QWORD *)(v141 + 8) != v134 )
            goto LABEL_195;
          v321 = v138;
          v327 = ii;
          v353 = v139;
          if ( !v134 )
            goto LABEL_222;
          v161 = memcmp(v137, v142, v134);
          v139 = v353;
          ii = v327;
          v138 = v321;
          v162 = v161 == 0;
        }
        if ( v162 )
          goto LABEL_222;
        if ( v142 == (const void *)-1LL )
          goto LABEL_200;
LABEL_195:
        v143 = v138 + ii;
        ++v138;
      }
      if ( v137 != (char *)-1LL )
        goto LABEL_200;
LABEL_222:
      v101 = 0;
LABEL_201:
      v149 = (const void *)sub_26C07A0(v96);
      if ( !unk_4F838D1 )
      {
LABEL_202:
        v352 = v145;
        v326 = v425;
        v320 = v426;
        v150 = sub_C92610();
        v151 = sub_C92860((__int64 *)&v425, v149, v352, v150);
        if ( v151 == -1 )
          v152 = &v425[8 * v426];
        else
          v152 = &v425[8 * v151];
        if ( v152 != &v326[8 * v320] )
          v101 = -1;
        goto LABEL_129;
      }
      v351 = v145;
      v144 = sub_B2F650((__int64)v149, v145);
      v145 = v351;
      v146 = v144;
      if ( v429 )
      {
        v147 = (v429 - 1) & (((0xBF58476D1CE4E5B9LL * v144) >> 31) ^ (484763065 * v144));
        v148 = *(_QWORD *)(v428
                         + 8LL
                         * ((v429 - 1)
                          & ((unsigned int)((0xBF58476D1CE4E5B9LL * v146) >> 31)
                           ^ (484763065 * (_DWORD)v146))));
        if ( v148 == v146 )
        {
LABEL_198:
          v101 = -1;
          goto LABEL_129;
        }
        v302 = 1;
        while ( v148 != -1 )
        {
          v147 = (v429 - 1) & (v302 + v147);
          v148 = *(_QWORD *)(v428 + 8LL * v147);
          if ( v146 == v148 )
            goto LABEL_198;
          ++v302;
        }
      }
      if ( !unk_4F838D1 )
        goto LABEL_202;
LABEL_129:
      sub_B2EE70((__int64)&s1, v96, 0);
      if ( !LOBYTE(n[1]) )
        sub_B2F4C0(v96, v101, 0, 0);
      v102 = sub_BC0510(a4, &unk_4F82418, *(_QWORD *)(v96 + 40));
      v395 = sub_BC1CD0(*(_QWORD *)(v102 + 8), &unk_4F8FAE8, v96) + 8;
      if ( unk_4F838D3 )
      {
        v103 = sub_31810C0(v413, v96, 1);
        v393 = v103;
        goto LABEL_133;
      }
      v120 = (_QWORD *)v389;
      v121 = (int *)sub_26C07A0(v96);
      v103 = sub_26C7880(v120, v121, v122);
      v393 = v103;
      if ( v103 )
        goto LABEL_538;
      v123 = (void *)sub_26C07A0(v96);
      n[1] = 0;
      s1 = v123;
      v324 = v123;
      n[0] = v124;
      v349 = v124;
      v379 = 0;
      v380 = 0;
      v125 = (char *)sub_26C3A50((__int64)v390, (__int64)&s1);
      v126 = v125;
      if ( v125 == &v391 )
      {
        v251 = *(_QWORD *)(v389 + 88);
        if ( !v251 )
          goto LABEL_406;
        sub_C21B60((__int64)&v364, v251, (__int64)v324, v349);
        if ( (_BYTE)v365
          && (n[1] = 0,
              v379 = 0,
              s1 = (void *)v364.m128i_i64[0],
              v380 = 0,
              n[0] = v364.m128i_u64[1],
              v252 = sub_26C3A50((__int64)v390, (__int64)&s1),
              (char *)v252 != v126) )
        {
          v103 = v252 + 72;
          v393 = v103;
        }
        else
        {
LABEL_406:
          v103 = v393;
        }
      }
      else
      {
        v103 = (__int64)(v125 + 72);
        v393 = (__int64)(v126 + 72);
      }
LABEL_133:
      if ( v103 )
      {
LABEL_538:
        if ( *(_QWORD *)(v103 + 56) )
          v331 |= sub_26DC3F0(&v384, v96, a5);
      }
      if ( v344 == ++v95 )
      {
        v344 = (__int64 *)src;
        goto LABEL_138;
      }
    }
    v127 = 4 * v386;
    v97 = v387;
    if ( (unsigned int)(4 * v386) < 0x40 )
      v127 = 64;
    if ( v127 < v387 )
    {
      if ( (_DWORD)v386 == 1 )
      {
        v129 = 64;
      }
      else
      {
        _BitScanReverse(&v128, v386 - 1);
        v129 = 1 << (33 - (v128 ^ 0x1F));
        if ( v129 < 64 )
          v129 = 64;
        if ( v129 == v387 )
          goto LABEL_187;
      }
      sub_C7D6A0((__int64)v385, 16LL * v387, 8);
      v130 = sub_26BC060(v129);
      v387 = v130;
      if ( v130 )
      {
        v385 = (_QWORD *)sub_C7D670(16LL * v130, 8);
LABEL_187:
        sub_26C7FD0((__int64)&v384.m128i_i64[1]);
        goto LABEL_125;
      }
      goto LABEL_315;
    }
LABEL_122:
    v98 = v385;
    v99 = &v385[2 * v97];
    if ( v385 != v99 )
    {
      do
      {
        *v98 = -4096;
        v98 += 2;
      }
      while ( v99 != v98 );
    }
LABEL_124:
    v386 = 0;
    goto LABEL_125;
  }
  v331 = 0;
LABEL_138:
  if ( v344 )
    j_j___libc_free_0((unsigned __int64)v344);
  if ( !unk_4F838D3 )
  {
    if ( v420 )
    {
      v220 = v419;
      v221 = &v419[2 * v421];
      if ( v419 != v221 )
      {
        while ( 1 )
        {
          v222 = *v220;
          v223 = v220;
          if ( *v220 != -8192 && v222 != -4096 )
            break;
          v220 += 2;
          if ( v221 == v220 )
            goto LABEL_141;
        }
        while ( v223 != v221 )
        {
          v224 = v223[1];
          v223 += 2;
          sub_29E4190(v222, v224, 0);
          if ( v223 == v221 )
            break;
          while ( 1 )
          {
            v222 = *v223;
            if ( *v223 != -8192 && v222 != -4096 )
              break;
            v223 += 2;
            if ( v221 == v223 )
              goto LABEL_141;
          }
        }
      }
    }
  }
LABEL_141:
  if ( (_BYTE)qword_4FF6BC8 && unk_4F838D4 )
  {
    v340 = a3[4];
    if ( v316 == (__int64 *)v340 )
    {
LABEL_301:
      v212 = (_QWORD *)sub_BA8DC0((__int64)a3, (__int64)"llvm.pseudo_probe_desc", 22);
      if ( v212 )
        sub_BA9050((__int64)a3, v212);
      goto LABEL_143;
    }
    while ( 1 )
    {
      s1 = 0;
      n[0] = 0;
      n[1] = 0;
      if ( !v340 )
        BUG();
      v192 = *(_QWORD *)(v340 + 24);
      if ( v192 == v340 + 16 )
        goto LABEL_300;
      do
      {
        if ( !v192 )
          BUG();
        v193 = *(_QWORD *)(v192 + 32);
        v194 = v192 + 24;
        if ( v193 != v192 + 24 )
        {
          while ( 1 )
          {
            if ( !v193 )
              BUG();
            v197 = *(unsigned __int8 *)(v193 - 24);
            if ( (_BYTE)v197 == 85 )
            {
              v198 = *(_QWORD *)(v193 - 56);
              if ( v198
                && !*(_BYTE *)v198
                && *(_QWORD *)(v198 + 24) == *(_QWORD *)(v193 + 56)
                && (*(_BYTE *)(v198 + 33) & 0x20) != 0
                && *(_DWORD *)(v198 + 36) == 291 )
              {
                v213 = n[0];
                v364.m128i_i64[0] = v193 - 24;
                if ( n[0] == n[1] )
                {
                  sub_249A840((__int64)&s1, (_BYTE *)n[0], &v364);
                }
                else
                {
                  if ( n[0] )
                  {
                    *(_QWORD *)n[0] = v193 - 24;
                    v213 = n[0];
                  }
                  n[0] = v213 + 8;
                }
                goto LABEL_275;
              }
            }
            else
            {
              v195 = (unsigned int)(v197 - 34);
              if ( (unsigned __int8)v195 > 0x33u )
                goto LABEL_275;
              v196 = 0x8000000000041LL;
              if ( !_bittest64(&v196, v195) )
                goto LABEL_275;
            }
            v199 = sub_B10CD0(v193 + 24);
            v200 = v199;
            if ( !v199 )
              goto LABEL_275;
            v201 = *(_BYTE *)(v199 - 16);
            v202 = (v201 & 2) != 0 ? *(_QWORD *)(v200 - 32) : v200 - 16 - 8LL * ((v201 >> 2) & 0xF);
            if ( **(_BYTE **)v202 != 20 )
              goto LABEL_275;
            v203 = *(_DWORD *)(*(_QWORD *)v202 + 4LL);
            if ( (v203 & 7) != 7 || (v203 & 0xFFFFFFF8) == 0 )
              goto LABEL_275;
            v204 = HIWORD(v203) & 7;
            v205 = v203 & 0x10000000;
            if ( v205 )
              v205 = v204;
            v206 = sub_26BDBC0(v200, v205);
            sub_B10CB0(&v364, (__int64)v206);
            if ( (__m128i *)(v193 + 24) == &v364 )
            {
              if ( v364.m128i_i64[0] )
                sub_B91220((__int64)&v364, v364.m128i_i64[0]);
              goto LABEL_275;
            }
            v207 = *(_QWORD *)(v193 + 24);
            if ( v207 )
              sub_B91220(v193 + 24, v207);
            v208 = (unsigned __int8 *)v364.m128i_i64[0];
            *(_QWORD *)(v193 + 24) = v364.m128i_i64[0];
            if ( v208 )
            {
              sub_B976B0((__int64)&v364, v208, v193 + 24);
              v193 = *(_QWORD *)(v193 + 8);
              if ( v194 == v193 )
                break;
            }
            else
            {
LABEL_275:
              v193 = *(_QWORD *)(v193 + 8);
              if ( v194 == v193 )
                break;
            }
          }
        }
        v192 = *(_QWORD *)(v192 + 8);
      }
      while ( v340 + 16 != v192 );
      v209 = s1;
      v210 = n[0];
      if ( s1 != (void *)n[0] )
      {
        do
        {
          v211 = (_QWORD *)*v209++;
          sub_B43D60(v211);
        }
        while ( (_QWORD *)v210 != v209 );
        v209 = s1;
      }
      if ( v209 )
        j_j___libc_free_0((unsigned __int64)v209);
LABEL_300:
      v340 = *(_QWORD *)(v340 + 8);
      if ( v316 == (__int64 *)v340 )
        goto LABEL_301;
    }
  }
LABEL_143:
  if ( !*(_BYTE *)(v356[0] + 204) )
    goto LABEL_152;
  v104 = (__int64)v357;
  v105 = *((_DWORD *)v357 + 4);
  ++*(_QWORD *)v357;
  if ( !v105 )
  {
    if ( !*(_DWORD *)(v104 + 20) )
      goto LABEL_151;
    v107 = *(unsigned int *)(v104 + 24);
    if ( (unsigned int)v107 <= 0x40 )
      goto LABEL_148;
    sub_C7D6A0(*(_QWORD *)(v104 + 8), 24 * v107, 8);
    *(_DWORD *)(v104 + 24) = 0;
    goto LABEL_264;
  }
  v106 = 4 * v105;
  v107 = *(unsigned int *)(v104 + 24);
  if ( (unsigned int)(4 * v105) < 0x40 )
    v106 = 64;
  if ( v106 < (unsigned int)v107 )
  {
    v304 = v105 - 1;
    if ( v304 )
    {
      _BitScanReverse(&v304, v304);
      v305 = 1 << (33 - (v304 ^ 0x1F));
      if ( v305 < 64 )
        v305 = 64;
      if ( v305 == (_DWORD)v107 )
        goto LABEL_523;
    }
    else
    {
      v305 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(v104 + 8), 24 * v107, 8);
    v306 = sub_26BC060(v305);
    *(_DWORD *)(v104 + 24) = v306;
    if ( v306 )
    {
      *(_QWORD *)(v104 + 8) = sub_C7D670(24LL * v306, 8);
LABEL_523:
      sub_26C4930(v104);
      goto LABEL_151;
    }
LABEL_264:
    *(_QWORD *)(v104 + 8) = 0;
    goto LABEL_150;
  }
LABEL_148:
  v108 = *(_QWORD **)(v104 + 8);
  for ( jj = &v108[3 * v107]; jj != v108; v108 += 3 )
    *v108 = -1;
LABEL_150:
  *(_QWORD *)(v104 + 16) = 0;
LABEL_151:
  sub_26C7D10(v356, 0);
LABEL_152:
  v110 = a1 + 32;
  v111 = a1 + 80;
  if ( v331 )
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v110;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v111;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v110;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v111;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
  }
LABEL_19:
  v17 = v432;
  v384.m128i_i64[0] = (__int64)off_4A206A0;
  if ( v432 )
  {
    sub_26C2E30((__int64)v432);
    j_j___libc_free_0((unsigned __int64)v17);
  }
  if ( v431 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v431 + 8LL))(v431);
  sub_C7D6A0(v428, 8LL * v429, 8);
  if ( v427 )
  {
    v18 = (unsigned __int64)v425;
    if ( v426 )
    {
      v112 = 8LL * v426;
      v113 = 0;
      do
      {
        v114 = *(_QWORD **)(v18 + v113);
        if ( v114 != (_QWORD *)-8LL && v114 )
        {
          sub_C7D6A0((__int64)v114, *v114 + 9LL, 8);
          v18 = (unsigned __int64)v425;
        }
        v113 += 8;
      }
      while ( v112 != v113 );
    }
  }
  else
  {
    v18 = (unsigned __int64)v425;
  }
  _libc_free(v18);
  sub_C7D6A0(v423, 24LL * v424, 8);
  sub_C7D6A0((__int64)v419, 16LL * v421, 8);
  if ( v417 )
    sub_A191D0(v417);
  if ( v414 != &v415 )
    j_j___libc_free_0((unsigned __int64)v414);
  v19 = v413;
  if ( v413 )
  {
    sub_26BC0B0(*(_QWORD **)(v413 + 136));
    v20 = *(_QWORD **)(v19 + 72);
    while ( v20 )
    {
      v21 = (unsigned __int64)v20;
      v20 = (_QWORD *)*v20;
      j_j___libc_free_0(v21);
    }
    memset(*(void **)(v19 + 56), 0, 8LL * *(_QWORD *)(v19 + 64));
    v22 = *(_QWORD *)(v19 + 56);
    *(_QWORD *)(v19 + 80) = 0;
    *(_QWORD *)(v19 + 72) = 0;
    if ( v22 != v19 + 104 )
      j_j___libc_free_0(v22);
    v23 = *(_QWORD **)(v19 + 16);
    while ( v23 )
    {
      v24 = (unsigned __int64)v23;
      v23 = (_QWORD *)*v23;
      v25 = *(_QWORD *)(v24 + 16);
      if ( v25 )
        j_j___libc_free_0(v25);
      j_j___libc_free_0(v24);
    }
    memset(*(void **)v19, 0, 8LL * *(_QWORD *)(v19 + 8));
    v26 = *(void **)v19;
    *(_QWORD *)(v19 + 24) = 0;
    *(_QWORD *)(v19 + 16) = 0;
    if ( v26 != (void *)(v19 + 48) )
      j_j___libc_free_0((unsigned __int64)v26);
    j_j___libc_free_0(v19);
  }
  if ( v411 )
    v411(v410, v410, 3);
  if ( v409 )
    v409(v408, v408, 3);
  if ( v407 )
    v407(v406, v406, 3);
  v27 = v403;
  while ( v27 )
  {
    v28 = (unsigned __int64)v27;
    v27 = (_QWORD *)*v27;
    j_j___libc_free_0(v28);
  }
  memset(s, 0, 8 * v402);
  v404 = 0;
  v403 = 0;
  if ( s != &v405 )
    j_j___libc_free_0((unsigned __int64)s);
  v29 = v398;
  while ( v29 )
  {
    v30 = (unsigned __int64)v29;
    v29 = (_QWORD *)*v29;
    j_j___libc_free_0(v30);
  }
  memset(v396, 0, 8 * v397);
  v399 = 0;
  v398 = 0;
  if ( v396 != &v400 )
    j_j___libc_free_0((unsigned __int64)v396);
  sub_26C0C70((__int64)&v384);
  return a1;
}
