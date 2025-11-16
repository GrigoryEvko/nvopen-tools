// Function: sub_1D830E0
// Address: 0x1d830e0
//
_BOOL8 __fastcall sub_1D830E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r14
  __int64 (*v4)(); // rax
  __int64 v6; // r15
  __int64 (*v7)(); // rdx
  __int64 v8; // rax
  __int64 (*v9)(); // rdx
  __int64 v10; // rax
  const __m128i *v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 (*v23)(void); // rdx
  __int64 v24; // rax
  __int64 (*v25)(); // rax
  __int64 v26; // rax
  int v27; // r8d
  int v28; // r9d
  unsigned int v29; // edx
  unsigned int v30; // ebx
  int v31; // ecx
  __int64 v32; // r12
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rdx
  unsigned int v35; // r13d
  __int64 v36; // rbx
  __m128i v37; // rax
  const __m128i *v38; // rax
  const __m128i *v39; // rax
  __m128i *v40; // rax
  __int8 *v41; // rax
  __m128i *v42; // rax
  __m128i *v43; // rax
  __int8 *v44; // rax
  const __m128i *v45; // rax
  const __m128i *v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rdi
  _BYTE *v49; // rsi
  const __m128i *v50; // rcx
  const __m128i *v51; // rdx
  unsigned __int64 v52; // r13
  __m128i *v53; // rax
  __m128i *v54; // rcx
  const __m128i *v55; // rax
  unsigned __int64 v56; // rbx
  __int64 v57; // rax
  __m128i *v58; // rdi
  __m128i *v59; // rax
  const __m128i *v60; // rcx
  const __m128i *v61; // rdx
  bool v62; // cl
  const __m128i *v63; // rax
  __int64 v64; // rcx
  _QWORD *v65; // rax
  __int64 v66; // r12
  _QWORD *v67; // r13
  _QWORD *v68; // rbx
  _QWORD *v69; // rdx
  _QWORD *v70; // rax
  _QWORD *v71; // rcx
  int v72; // eax
  __int64 v73; // rdi
  __int64 (*v74)(); // rax
  int v75; // r8d
  int v76; // r9d
  _QWORD *v77; // r15
  _QWORD *v78; // rax
  unsigned __int64 v79; // r13
  void *v80; // rdi
  unsigned int v81; // eax
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rdi
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  unsigned __int64 v91; // rbx
  unsigned __int64 v92; // rax
  __int64 v93; // rdx
  _QWORD *v94; // rdx
  _QWORD *v95; // rax
  _QWORD *v96; // r13
  __int64 v97; // rax
  __int64 v98; // rax
  __m128i *v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rax
  unsigned int *v102; // rax
  unsigned int v103; // edx
  __int64 v104; // rsi
  __int64 v105; // rcx
  __int64 v106; // rdi
  _QWORD *v107; // r8
  __int64 v108; // rdi
  __int64 v109; // rdi
  __int64 (*v110)(); // r10
  __int64 *v111; // rax
  unsigned __int64 v112; // r12
  __int64 v113; // r14
  int v114; // eax
  unsigned int v115; // ecx
  int v116; // eax
  __int64 v117; // rdx
  unsigned int v118; // edx
  unsigned int v119; // eax
  unsigned __int64 v120; // rax
  unsigned int v121; // ebx
  unsigned __int64 v122; // rax
  unsigned __int64 v123; // rbx
  unsigned int v124; // r12d
  __int64 v125; // rax
  __int64 v126; // rdx
  __int64 v127; // rcx
  _QWORD *v128; // rdx
  __int64 v129; // r13
  __int64 v130; // r15
  int v131; // r14d
  char v132; // al
  __int64 v133; // rax
  __int64 v134; // rcx
  __int64 v135; // rax
  __int16 v136; // di
  unsigned int v137; // edx
  _WORD *v138; // rdx
  __int16 *v139; // rcx
  unsigned __int16 v140; // r13
  __int16 *v141; // r14
  __int16 v142; // ax
  _BYTE *v143; // rdx
  unsigned int v144; // ecx
  unsigned int v145; // eax
  __int64 v146; // rdi
  _DWORD *v147; // rsi
  __int64 v148; // rax
  __int16 v149; // ax
  __int64 v150; // rax
  unsigned int v151; // r13d
  __int64 v152; // rdi
  __int64 v153; // rsi
  __int64 v154; // rax
  __int64 v155; // rdi
  __int64 v156; // rsi
  __int64 v157; // rdx
  __int64 v158; // rax
  int v159; // r9d
  __int64 v160; // rdx
  unsigned int v161; // ebx
  int v162; // edx
  __int64 v163; // rbx
  __int64 v164; // rax
  unsigned int v165; // r15d
  __int64 v166; // rcx
  unsigned int v167; // r15d
  __int64 v168; // rsi
  unsigned int v169; // edx
  __int64 *v170; // rax
  __int64 v171; // r8
  __int64 v172; // rax
  __int64 v173; // rdx
  int v174; // r13d
  unsigned int v175; // eax
  unsigned int v176; // r8d
  int v177; // r13d
  unsigned int v178; // eax
  unsigned int v179; // r8d
  _QWORD *v180; // rbx
  unsigned int v181; // eax
  __int64 v182; // r9
  unsigned int v183; // r14d
  int v184; // edi
  __int64 v185; // r8
  int v186; // edi
  unsigned int v187; // esi
  __int64 *v188; // rax
  __int64 v189; // r11
  int v190; // esi
  unsigned int v191; // eax
  __int64 v192; // rcx
  unsigned int v193; // edx
  __int16 v194; // ax
  _WORD *v195; // rdx
  __int16 *v196; // rcx
  unsigned __int16 v197; // ax
  __int16 *v198; // rdi
  unsigned int v199; // esi
  unsigned int v200; // ecx
  _DWORD *v201; // rdx
  __int64 v202; // rsi
  _DWORD *v203; // rsi
  __int16 v204; // dx
  __int64 v205; // rdi
  __int64 v206; // rax
  __int64 *v207; // rbx
  __int64 v208; // rax
  __int64 *v209; // r14
  __int64 *v210; // r15
  unsigned __int64 v211; // rdx
  __int64 v212; // rcx
  __int64 v213; // rdi
  __int64 *v214; // rbx
  __int64 v215; // rax
  __int64 *v216; // r14
  unsigned __int64 *v217; // r15
  unsigned __int64 v218; // rdx
  unsigned __int64 v219; // rcx
  __int64 v220; // rax
  __int64 v221; // rdi
  __int64 v222; // rax
  __int64 v223; // rsi
  unsigned int *v224; // r10
  __int64 v225; // r15
  unsigned int *v226; // r14
  __int64 v227; // rdi
  unsigned int v228; // ebx
  __int64 v229; // rax
  __int64 v230; // r8
  __int64 v231; // rsi
  __int64 v232; // rdx
  __int64 v233; // rax
  __int64 v234; // rcx
  __int64 v235; // rdi
  __int64 v236; // rsi
  __int64 v237; // rdi
  __int64 v238; // rsi
  int v239; // r8d
  int v240; // r9d
  __int64 v241; // rdx
  __int64 v242; // rax
  __int64 v243; // rdi
  __int64 v244; // rax
  __int64 v245; // rdi
  __int64 v246; // rdi
  __int64 v247; // rsi
  __int64 v248; // rbx
  __int64 *v249; // r15
  int v250; // r14d
  __int64 v251; // r13
  __int64 v252; // rax
  __int64 v253; // rbx
  unsigned int v254; // ecx
  __int64 v255; // rsi
  unsigned int v256; // edx
  __int64 *v257; // rax
  __int64 v258; // rdi
  __int64 v259; // rax
  __m128i *v260; // r14
  __int64 v261; // rax
  __m128i *v262; // r15
  __int64 v263; // r13
  __int64 v264; // rax
  __int64 v265; // rdx
  __int64 v266; // rsi
  unsigned int v267; // ecx
  __int64 *v268; // rax
  __int64 v269; // r9
  __int64 v270; // r13
  __int64 v271; // rax
  __int64 v272; // r12
  __int64 v273; // rax
  _BYTE *v274; // rax
  int v275; // r9d
  _BYTE *v276; // rsi
  unsigned __int64 v277; // rdi
  __m128i *v278; // rsi
  __int64 v279; // r8
  __int64 v280; // rax
  __int64 v281; // r14
  __int64 v282; // rdx
  __int64 v283; // rdx
  __int64 *v284; // rbx
  __int64 *v285; // r12
  __int64 v286; // r13
  __int64 *v287; // r15
  __int64 v288; // rbx
  __int64 v289; // r13
  __int64 v290; // r13
  __int64 v291; // rax
  __int64 v292; // rcx
  int v293; // edi
  __int64 v294; // rsi
  unsigned int v295; // r11d
  unsigned int v296; // r8d
  __int64 *v297; // rdx
  __int64 v298; // r9
  __int64 v299; // rax
  __int64 v300; // r8
  _BYTE *v301; // rax
  __int64 v302; // r8
  int v303; // eax
  __int64 v304; // r15
  _QWORD *v305; // rdi
  __int64 *v306; // r13
  __int64 v307; // r8
  __int64 v308; // rax
  __int64 v309; // rsi
  int v310; // r10d
  unsigned int v311; // edx
  _QWORD *v312; // r14
  __int64 v313; // rbx
  __int64 *v314; // r12
  char *v315; // rdx
  char *v316; // rdi
  __int64 v317; // rsi
  __int64 v318; // rax
  char *v319; // rax
  _QWORD *v320; // rax
  __int64 v321; // rax
  __int64 v322; // rdx
  __int64 v323; // rsi
  _QWORD *v324; // rdx
  unsigned int v325; // r11d
  _QWORD *v326; // rax
  __int64 v327; // rdx
  __int64 v328; // r8
  __int64 v329; // rdi
  int v330; // eax
  int v331; // r10d
  __int64 v332; // r9
  int v333; // edx
  int v334; // r10d
  __int64 v335; // r9
  __int64 v336; // r11
  int v337; // eax
  int v338; // r8d
  __int64 v339; // rdx
  __int64 v340; // rax
  __int64 v341; // rsi
  __int64 v342; // r15
  int v343; // r14d
  unsigned int *v344; // rbx
  int v345; // eax
  int v346; // edx
  __int64 v347; // r11
  __int64 v348; // rax
  __int64 v349; // r13
  __int64 v350; // r14
  __int64 *v351; // rax
  __int64 *v352; // rbx
  unsigned __int64 v353; // rcx
  __int64 v354; // rsi
  int v355; // r8d
  int v356; // r9d
  __int64 v357; // rax
  __int64 v358; // rdi
  int v359; // eax
  int v360; // edi
  int v361; // eax
  int v362; // r9d
  __int64 v363; // [rsp+8h] [rbp-588h]
  __int64 v364; // [rsp+18h] [rbp-578h]
  __int64 v365; // [rsp+20h] [rbp-570h]
  __int64 v366; // [rsp+28h] [rbp-568h]
  __int64 v367; // [rsp+60h] [rbp-530h]
  __int64 *v368; // [rsp+60h] [rbp-530h]
  __int64 v369; // [rsp+68h] [rbp-528h]
  _QWORD *v370; // [rsp+68h] [rbp-528h]
  __int64 v371; // [rsp+70h] [rbp-520h]
  bool v372; // [rsp+88h] [rbp-508h]
  int v373; // [rsp+88h] [rbp-508h]
  __int64 v374; // [rsp+88h] [rbp-508h]
  __int64 v375; // [rsp+88h] [rbp-508h]
  __int64 v376; // [rsp+88h] [rbp-508h]
  __int64 v377; // [rsp+88h] [rbp-508h]
  _QWORD *v378; // [rsp+88h] [rbp-508h]
  bool v379; // [rsp+92h] [rbp-4FEh]
  bool v380; // [rsp+93h] [rbp-4FDh]
  __int64 v381; // [rsp+98h] [rbp-4F8h]
  unsigned int v382; // [rsp+98h] [rbp-4F8h]
  __int64 v383; // [rsp+98h] [rbp-4F8h]
  __int64 v384; // [rsp+98h] [rbp-4F8h]
  __int64 v385; // [rsp+A0h] [rbp-4F0h]
  __int64 v386; // [rsp+A0h] [rbp-4F0h]
  __int64 v387; // [rsp+A0h] [rbp-4F0h]
  __int64 v388; // [rsp+A0h] [rbp-4F0h]
  int v389; // [rsp+A0h] [rbp-4F0h]
  __int64 *v390; // [rsp+A0h] [rbp-4F0h]
  __int64 v391; // [rsp+A0h] [rbp-4F0h]
  __int64 *v392; // [rsp+A0h] [rbp-4F0h]
  _QWORD *v393; // [rsp+A8h] [rbp-4E8h]
  __m128i *v394; // [rsp+A8h] [rbp-4E8h]
  unsigned int v395; // [rsp+A8h] [rbp-4E8h]
  __int64 v396; // [rsp+A8h] [rbp-4E8h]
  __int64 v397; // [rsp+B0h] [rbp-4E0h] BYREF
  __int64 v398; // [rsp+B8h] [rbp-4D8h]
  __int64 v399; // [rsp+C0h] [rbp-4D0h] BYREF
  __int64 v400; // [rsp+C8h] [rbp-4C8h]
  _QWORD v401[16]; // [rsp+D0h] [rbp-4C0h] BYREF
  __int64 v402; // [rsp+150h] [rbp-440h] BYREF
  _QWORD *v403; // [rsp+158h] [rbp-438h]
  _QWORD *v404; // [rsp+160h] [rbp-430h]
  __int64 v405; // [rsp+168h] [rbp-428h]
  int v406; // [rsp+170h] [rbp-420h]
  _QWORD v407[8]; // [rsp+178h] [rbp-418h] BYREF
  const __m128i *v408; // [rsp+1B8h] [rbp-3D8h] BYREF
  const __m128i *v409; // [rsp+1C0h] [rbp-3D0h]
  __int8 *v410; // [rsp+1C8h] [rbp-3C8h]
  __int64 v411; // [rsp+1D0h] [rbp-3C0h] BYREF
  __int64 v412; // [rsp+1D8h] [rbp-3B8h]
  unsigned __int64 v413; // [rsp+1E0h] [rbp-3B0h]
  _BYTE v414[64]; // [rsp+1F8h] [rbp-398h] BYREF
  __m128i *v415; // [rsp+238h] [rbp-358h]
  __m128i *v416; // [rsp+240h] [rbp-350h]
  __int8 *v417; // [rsp+248h] [rbp-348h]
  _QWORD v418[2]; // [rsp+250h] [rbp-340h] BYREF
  unsigned __int64 v419; // [rsp+260h] [rbp-330h]
  char v420[64]; // [rsp+278h] [rbp-318h] BYREF
  const __m128i *v421; // [rsp+2B8h] [rbp-2D8h]
  const __m128i *v422; // [rsp+2C0h] [rbp-2D0h]
  __int8 *v423; // [rsp+2C8h] [rbp-2C8h]
  _QWORD v424[2]; // [rsp+2D0h] [rbp-2C0h] BYREF
  unsigned __int64 v425; // [rsp+2E0h] [rbp-2B0h]
  char v426[64]; // [rsp+2F8h] [rbp-298h] BYREF
  const __m128i *v427; // [rsp+338h] [rbp-258h]
  const __m128i *v428; // [rsp+340h] [rbp-250h]
  __int64 v429; // [rsp+348h] [rbp-248h]
  __m128i v430; // [rsp+350h] [rbp-240h] BYREF
  __m128i v431; // [rsp+360h] [rbp-230h] BYREF
  char v432[64]; // [rsp+378h] [rbp-218h] BYREF
  const __m128i *v433; // [rsp+3B8h] [rbp-1D8h]
  const __m128i *v434; // [rsp+3C0h] [rbp-1D0h]
  __int64 v435; // [rsp+3C8h] [rbp-1C8h]

  v3 = *(_QWORD **)(a2 + 16);
  v4 = *(__int64 (**)())(*v3 + 280LL);
  if ( v4 == sub_1D820D0 )
    return 0;
  v6 = a1;
  if ( !((unsigned __int8 (__fastcall *)(_QWORD))v4)(*(_QWORD *)(a2 + 16)) )
    return 0;
  v7 = *(__int64 (**)())(*v3 + 40LL);
  v8 = 0;
  if ( v7 != sub_1D00B00 )
    v8 = ((__int64 (__fastcall *)(_QWORD *))v7)(v3);
  *(_QWORD *)(a1 + 232) = v8;
  v9 = *(__int64 (**)())(*v3 + 112LL);
  v10 = 0;
  if ( v9 != sub_1D00B10 )
    v10 = ((__int64 (__fastcall *)(_QWORD *))v9)(v3);
  *(_QWORD *)(a1 + 240) = v10;
  v11 = (const __m128i *)v3[20];
  v12 = *(__int64 **)(a1 + 8);
  *(__m128i *)(a1 + 248) = _mm_loadu_si128(v11);
  *(__m128i *)(a1 + 264) = _mm_loadu_si128(v11 + 1);
  *(__m128i *)(a1 + 280) = _mm_loadu_si128(v11 + 2);
  *(__m128i *)(a1 + 296) = _mm_loadu_si128(v11 + 3);
  *(_QWORD *)(a1 + 312) = v11[4].m128i_i64[0];
  *(_QWORD *)(a1 + 320) = *(_QWORD *)(a2 + 40);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
    goto LABEL_519;
  while ( *(_UNKNOWN **)v13 != &unk_4FC62EC )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_519;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4FC62EC);
  v16 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v6 + 328) = v15;
  v17 = sub_160F9A0(v16, (__int64)&unk_4FC6A0C, 1u);
  v18 = v17;
  if ( v17 )
    v18 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v17 + 104LL))(v17, &unk_4FC6A0C);
  v19 = *(__int64 **)(v6 + 8);
  *(_QWORD *)(v6 + 336) = v18;
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
LABEL_519:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4FC820C )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_519;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4FC820C);
  *(_QWORD *)(v6 + 352) = 0;
  *(_QWORD *)(v6 + 344) = v22;
  v365 = v6 + 360;
  v23 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v24 = 0;
  if ( v23 != sub_1D00B00 )
    v24 = v23();
  *(_QWORD *)(v6 + 360) = v24;
  v25 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v25 == sub_1D00B10 )
  {
    *(_QWORD *)(v6 + 368) = 0;
    v2 = *(_QWORD *)(a2 + 40);
    *(_DWORD *)(v6 + 1000) = 0;
    *(_QWORD *)(v6 + 376) = v2;
    BUG();
  }
  v26 = v25();
  *(_QWORD *)(v6 + 368) = v26;
  v363 = v6 + 992;
  *(_QWORD *)(v6 + 376) = *(_QWORD *)(a2 + 40);
  v29 = *(_DWORD *)(v6 + 1048);
  *(_DWORD *)(v6 + 1000) = 0;
  v30 = *(_DWORD *)(v26 + 44);
  v31 = v29 >> 2;
  if ( v30 < v29 >> 2 || v30 > v29 )
  {
    _libc_free(*(_QWORD *)(v6 + 1040));
    v32 = (__int64)_libc_calloc(v30, 1u);
    if ( !v32 )
    {
      if ( v30 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v32 = sub_13A3880(1u);
    }
    *(_QWORD *)(v6 + 1040) = v32;
    v26 = *(_QWORD *)(v6 + 368);
    *(_DWORD *)(v6 + 1048) = v30;
  }
  v33 = *(_QWORD *)(v6 + 976);
  *(_DWORD *)(v6 + 984) = 0;
  v34 = *(unsigned int *)(v26 + 44);
  v35 = *(_DWORD *)(v26 + 44);
  if ( v34 > v33 << 6 )
  {
    v112 = (unsigned int)(v34 + 63) >> 6;
    if ( v112 < 2 * v33 )
      v112 = 2 * v33;
    v113 = (__int64)realloc(*(_QWORD *)(v6 + 968), 8 * v112, 8 * (int)v112, v31, v27, v28);
    if ( !v113 )
    {
      if ( 8 * v112 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v113 = sub_13A3880(1u);
    }
    v114 = *(_DWORD *)(v6 + 984);
    *(_QWORD *)(v6 + 968) = v113;
    *(_QWORD *)(v6 + 976) = v112;
    v115 = (unsigned int)(v114 + 63) >> 6;
    if ( v112 > v115 )
    {
      v395 = (unsigned int)(v114 + 63) >> 6;
      memset((void *)(v113 + 8LL * v115), 0, 8 * (v112 - v115));
      v114 = *(_DWORD *)(v6 + 984);
      v113 = *(_QWORD *)(v6 + 968);
      v115 = v395;
    }
    v116 = v114 & 0x3F;
    if ( v116 )
    {
      *(_QWORD *)(v113 + 8LL * (v115 - 1)) &= ~(-1LL << v116);
      v113 = *(_QWORD *)(v6 + 968);
    }
    v117 = *(_QWORD *)(v6 + 976) - (unsigned int)v33;
    if ( v117 )
      memset((void *)(v113 + 8LL * (unsigned int)v33), 0, 8 * v117);
    v118 = *(_DWORD *)(v6 + 984);
    v119 = v118;
    if ( v35 <= v118 )
      goto LABEL_170;
    v123 = *(_QWORD *)(v6 + 976);
    v124 = (v118 + 63) >> 6;
    v125 = v124;
    if ( v124 >= v123 )
      goto LABEL_177;
    v33 = v123 - v124;
    if ( !v33 )
      goto LABEL_177;
    goto LABEL_182;
  }
  if ( !(_DWORD)v34 )
    goto LABEL_28;
  v119 = 0;
  if ( v33 )
  {
    v125 = 0;
    v124 = 0;
LABEL_182:
    memset((void *)(*(_QWORD *)(v6 + 968) + 8 * v125), 0, 8 * v33);
    v118 = *(_DWORD *)(v6 + 984);
LABEL_177:
    v119 = v118;
    if ( (v118 & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(v6 + 968) + 8LL * (v124 - 1)) &= ~(-1LL << (v118 & 0x3F));
      v119 = *(_DWORD *)(v6 + 984);
    }
  }
LABEL_170:
  *(_DWORD *)(v6 + 984) = v35;
  if ( v35 < v119 )
  {
    v120 = *(_QWORD *)(v6 + 976);
    v121 = (v35 + 63) >> 6;
    if ( v120 > v121 )
    {
      v122 = v120 - v121;
      if ( v122 )
      {
        memset((void *)(*(_QWORD *)(v6 + 968) + 8LL * v121), 0, 8 * v122);
        v35 = *(_DWORD *)(v6 + 984);
      }
    }
    if ( (v35 & 0x3F) != 0 )
      *(_QWORD *)(*(_QWORD *)(v6 + 968) + 8LL * (v121 - 1)) &= ~(-1LL << (v35 & 0x3F));
  }
LABEL_28:
  v36 = *(_QWORD *)(v6 + 328);
  memset(v401, 0, sizeof(v401));
  LODWORD(v401[3]) = 8;
  v401[1] = &v401[5];
  v401[2] = &v401[5];
  sub_1E06620(v36);
  v37.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v36 + 1312) + 56LL);
  v405 = 0x100000008LL;
  v403 = v407;
  v407[0] = v37.m128i_i64[0];
  v404 = v407;
  v408 = 0;
  v409 = 0;
  v410 = 0;
  v406 = 0;
  v402 = 1;
  v37.m128i_i64[1] = *(_QWORD *)(v37.m128i_i64[0] + 24);
  v430 = v37;
  sub_1D82E20(&v408, 0, &v430);
  sub_1D82FA0((__int64)&v402);
  sub_16CCEE0(&v430, (__int64)v432, 8, (__int64)v401);
  v38 = (const __m128i *)v401[13];
  memset(&v401[13], 0, 24);
  v433 = v38;
  v434 = (const __m128i *)v401[14];
  v435 = v401[15];
  sub_16CCEE0(&v411, (__int64)v414, 8, (__int64)&v402);
  v39 = v408;
  v408 = 0;
  v415 = (__m128i *)v39;
  v40 = (__m128i *)v409;
  v409 = 0;
  v416 = v40;
  v41 = v410;
  v410 = 0;
  v417 = v41;
  sub_16CCEE0(v418, (__int64)v420, 8, (__int64)&v411);
  v42 = v415;
  v415 = 0;
  v421 = v42;
  v43 = v416;
  v416 = 0;
  v422 = v43;
  v44 = v417;
  v417 = 0;
  v423 = v44;
  sub_16CCEE0(v424, (__int64)v426, 8, (__int64)&v430);
  v45 = v433;
  v433 = 0;
  v427 = v45;
  v46 = v434;
  v434 = 0;
  v428 = v46;
  v47 = v435;
  v435 = 0;
  v429 = v47;
  if ( v415 )
    j_j___libc_free_0(v415, v417 - (__int8 *)v415);
  if ( v413 != v412 )
    _libc_free(v413);
  if ( v433 )
    j_j___libc_free_0(v433, v435 - (_QWORD)v433);
  if ( v431.m128i_i64[0] != v430.m128i_i64[1] )
    _libc_free(v431.m128i_u64[0]);
  if ( v408 )
    j_j___libc_free_0(v408, v410 - (__int8 *)v408);
  if ( v404 != v403 )
    _libc_free((unsigned __int64)v404);
  if ( v401[13] )
    j_j___libc_free_0(v401[13], v401[15] - v401[13]);
  if ( v401[2] != v401[1] )
    _libc_free(v401[2]);
  v48 = &v402;
  v49 = v407;
  sub_16CCCB0(&v402, (__int64)v407, (__int64)v418);
  v50 = v422;
  v51 = v421;
  v408 = 0;
  v409 = 0;
  v410 = 0;
  v52 = (char *)v422 - (char *)v421;
  if ( v422 == v421 )
  {
    v52 = 0;
    v53 = 0;
  }
  else
  {
    if ( v52 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_498;
    v53 = (__m128i *)sub_22077B0((char *)v422 - (char *)v421);
    v50 = v422;
    v51 = v421;
  }
  v408 = v53;
  v409 = v53;
  v410 = &v53->m128i_i8[v52];
  if ( v50 != v51 )
  {
    v54 = (__m128i *)((char *)v53 + (char *)v50 - (char *)v51);
    do
    {
      if ( v53 )
        *v53 = _mm_loadu_si128(v51);
      ++v53;
      ++v51;
    }
    while ( v53 != v54 );
  }
  v49 = v414;
  v48 = &v411;
  v409 = v53;
  sub_16CCCB0(&v411, (__int64)v414, (__int64)v424);
  v55 = v428;
  v51 = v427;
  v415 = 0;
  v416 = 0;
  v417 = 0;
  v56 = (char *)v428 - (char *)v427;
  if ( v428 == v427 )
  {
    v56 = 0;
    v58 = 0;
    goto LABEL_55;
  }
  if ( v56 > 0x7FFFFFFFFFFFFFF0LL )
LABEL_498:
    sub_4261EA(v48, v49, v51);
  v57 = sub_22077B0((char *)v428 - (char *)v427);
  v51 = v427;
  v58 = (__m128i *)v57;
  v55 = v428;
LABEL_55:
  v415 = v58;
  v416 = v58;
  v417 = &v58->m128i_i8[v56];
  if ( v55 == v51 )
  {
    v59 = v58;
  }
  else
  {
    v59 = (__m128i *)((char *)v58 + (char *)v55 - (char *)v51);
    do
    {
      if ( v58 )
        *v58 = _mm_loadu_si128(v51);
      ++v58;
      ++v51;
    }
    while ( v58 != v59 );
    v58 = v415;
  }
  v416 = v59;
  v60 = v408;
  v380 = 0;
  v61 = v409;
  while ( (char *)v61 - (char *)v60 != (char *)v59 - (char *)v58 )
  {
LABEL_71:
    v64 = *(_QWORD *)v61[-1].m128i_i64[0];
    *(_QWORD *)(v6 + 392) = 0;
    *(_QWORD *)(v6 + 408) = 0;
    *(_QWORD *)(v6 + 384) = v64;
    *(_QWORD *)(v6 + 400) = 0;
    v65 = *(_QWORD **)(v64 + 88);
    v364 = v64;
    if ( (unsigned int)((__int64)(*(_QWORD *)(v64 + 96) - (_QWORD)v65) >> 3) != 2 )
      goto LABEL_67;
    v372 = 0;
    v66 = v6;
    while ( 1 )
    {
      v67 = (_QWORD *)*v65;
      v68 = (_QWORD *)v65[1];
      if ( (unsigned int)((__int64)(*(_QWORD *)(*v65 + 72LL) - *(_QWORD *)(*v65 + 64LL)) >> 3) != 1 )
      {
        if ( (unsigned int)((__int64)(v68[9] - v68[8]) >> 3) != 1 )
          break;
        v67 = (_QWORD *)v65[1];
        v68 = (_QWORD *)*v65;
      }
      v69 = (_QWORD *)v67[11];
      if ( (unsigned int)((__int64)(v67[12] - (_QWORD)v69) >> 3) != 1 )
        break;
      v70 = (_QWORD *)*v69;
      *(_QWORD *)(v66 + 392) = *v69;
      if ( v68 != v70 )
      {
        if ( (unsigned int)((__int64)(v68[9] - v68[8]) >> 3) != 1 )
          break;
        v71 = (_QWORD *)v68[11];
        if ( (unsigned int)((__int64)(v68[12] - (_QWORD)v71) >> 3) != 1 || v70 != (_QWORD *)*v71 || v70[20] != v70[19] )
          break;
      }
      if ( v70 + 3 == (_QWORD *)(v70[3] & 0xFFFFFFFFFFFFFFF8LL) )
        break;
      v72 = **(unsigned __int16 **)(v70[4] + 16LL);
      v379 = v72 == 0 || v72 == 45;
      if ( !v379 )
        break;
      v73 = *(_QWORD *)(v66 + 360);
      *(_DWORD *)(v66 + 696) = 0;
      v74 = *(__int64 (**)())(*(_QWORD *)v73 + 264LL);
      if ( v74 == sub_1D820E0 )
        break;
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v74)(
             v73,
             v364,
             v66 + 400,
             v66 + 408,
             v66 + 688,
             0) )
      {
        break;
      }
      v77 = *(_QWORD **)(v66 + 400);
      if ( !v77 )
        break;
      v78 = *(_QWORD **)(v66 + 392);
      *(_DWORD *)(v66 + 424) = 0;
      if ( v67 != v77 )
        v68 = v67;
      *(_QWORD *)(v66 + 408) = v68;
      if ( v77 == v78 )
        v77 = *(_QWORD **)(v66 + 384);
      if ( v68 == v78 )
        v68 = *(_QWORD **)(v66 + 384);
      v79 = v78[4];
      v393 = v78 + 3;
      if ( (_QWORD *)v79 != v78 + 3 )
      {
        while ( 1 )
        {
          if ( **(_WORD **)(v79 + 16) && **(_WORD **)(v79 + 16) != 45 )
            goto LABEL_94;
          v430 = (__m128i)v79;
          v98 = *(unsigned int *)(v66 + 424);
          v431.m128i_i64[0] = 0;
          v431.m128i_i32[2] = 0;
          if ( (unsigned int)v98 >= *(_DWORD *)(v66 + 428) )
          {
            sub_16CD150(v66 + 416, (const void *)(v66 + 432), 0, 32, v75, v76);
            v98 = *(unsigned int *)(v66 + 424);
          }
          v99 = (__m128i *)(*(_QWORD *)(v66 + 416) + 32 * v98);
          *v99 = _mm_loadu_si128(&v430);
          v99[1] = _mm_loadu_si128(&v431);
          v100 = *(_QWORD *)(v66 + 416);
          v101 = (unsigned int)(*(_DWORD *)(v66 + 424) + 1);
          *(_DWORD *)(v66 + 424) = v101;
          v102 = (unsigned int *)(v100 + 32 * v101 - 32);
          v103 = 1;
          v104 = *(_QWORD *)v102;
          if ( *(_DWORD *)(*(_QWORD *)v102 + 40LL) != 1 )
            break;
LABEL_129:
          v109 = *(_QWORD *)(v66 + 360);
          v110 = *(__int64 (**)())(*(_QWORD *)v109 + 360LL);
          if ( v110 == sub_1D820F0
            || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, unsigned int *, unsigned int *, unsigned int *))v110)(
                  v109,
                  *(_QWORD *)(v66 + 384),
                  *(_QWORD *)(v66 + 688),
                  *(unsigned int *)(v66 + 696),
                  v102[2],
                  v102[3],
                  v102 + 4,
                  v102 + 5,
                  v102 + 6) )
          {
            goto LABEL_63;
          }
          if ( (*(_BYTE *)v79 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v79 + 46) & 8) != 0 )
              v79 = *(_QWORD *)(v79 + 8);
          }
          v79 = *(_QWORD *)(v79 + 8);
          if ( v393 == (_QWORD *)v79 )
            goto LABEL_94;
        }
        while ( 1 )
        {
          v105 = *(_QWORD *)(v104 + 32);
          v106 = 40LL * (v103 + 1);
          v107 = *(_QWORD **)(v105 + v106 + 24);
          if ( v107 == v77 )
          {
            v102[2] = *(_DWORD *)(v105 + 40LL * v103 + 8);
            v105 = *(_QWORD *)(v104 + 32);
            if ( v68 != *(_QWORD **)(v105 + v106 + 24) )
            {
LABEL_125:
              v103 += 2;
              if ( *(_DWORD *)(v104 + 40) == v103 )
                goto LABEL_129;
              continue;
            }
          }
          else if ( v68 != v107 )
          {
            goto LABEL_125;
          }
          v108 = v103;
          v103 += 2;
          v102[3] = *(_DWORD *)(v105 + 40 * v108 + 8);
          if ( *(_DWORD *)(v104 + 40) == v103 )
            goto LABEL_129;
        }
      }
LABEL_94:
      v80 = *(void **)(v66 + 880);
      ++*(_QWORD *)(v66 + 864);
      v385 = v66 + 864;
      if ( v80 != *(void **)(v66 + 872) )
      {
        v81 = 4 * (*(_DWORD *)(v66 + 892) - *(_DWORD *)(v66 + 896));
        v82 = *(unsigned int *)(v66 + 888);
        if ( v81 < 0x20 )
          v81 = 32;
        if ( (unsigned int)v82 > v81 )
        {
          sub_16CC920(v385);
          goto LABEL_100;
        }
        memset(v80, -1, 8 * v82);
      }
      *(_QWORD *)(v66 + 892) = 0;
LABEL_100:
      v83 = *(_QWORD *)(v66 + 976);
      if ( v83 )
        memset(*(void **)(v66 + 968), 0, 8 * v83);
      v84 = *(_QWORD *)(v66 + 400);
      v85 = *(_QWORD *)(v66 + 392);
      if ( v84 != v85 )
      {
        if ( *(_QWORD *)(v84 + 160) != *(_QWORD *)(v84 + 152) || !(unsigned __int8)sub_1D824C0(v365, v84) )
          break;
        v85 = *(_QWORD *)(v66 + 392);
      }
      v86 = *(_QWORD *)(v66 + 408);
      if ( v86 != v85 && (*(_QWORD *)(v86 + 160) != *(_QWORD *)(v86 + 152) || !(unsigned __int8)sub_1D824C0(v365, v86)) )
        break;
      v87 = *(_QWORD *)(v66 + 384);
      *(_DWORD *)(v66 + 1000) = 0;
      v394 = &v431;
      v430.m128i_i64[0] = (__int64)&v431;
      v430.m128i_i64[1] = 0x800000000LL;
      v371 = sub_1DD5EE0(v87);
      v90 = *(_QWORD *)(v66 + 384);
      v91 = v90 + 24;
      v381 = *(_QWORD *)(v90 + 32);
      if ( v90 + 24 == v381 )
        goto LABEL_117;
      while ( 1 )
      {
        v92 = *(_QWORD *)v91 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v92 )
          BUG();
        v93 = *(_QWORD *)v92;
        v91 = *(_QWORD *)v91 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v92 & 4) == 0 && (*(_BYTE *)(v92 + 46) & 4) != 0 )
        {
          while ( 1 )
          {
            v91 = v93 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)((v93 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
              break;
            v93 = *(_QWORD *)v91;
          }
        }
        v94 = *(_QWORD **)(v66 + 880);
        v95 = *(_QWORD **)(v66 + 872);
        if ( v94 == v95 )
        {
          v96 = &v95[*(unsigned int *)(v66 + 892)];
          if ( v95 == v96 )
          {
            v128 = *(_QWORD **)(v66 + 872);
          }
          else
          {
            do
            {
              if ( v91 == *v95 )
                break;
              ++v95;
            }
            while ( v96 != v95 );
            v128 = v96;
          }
        }
        else
        {
          v96 = &v94[*(unsigned int *)(v66 + 888)];
          v95 = sub_16CC9F0(v385, v91);
          if ( v91 == *v95 )
          {
            v126 = *(_QWORD *)(v66 + 880);
            v127 = v126 == *(_QWORD *)(v66 + 872) ? *(unsigned int *)(v66 + 892) : *(unsigned int *)(v66 + 888);
            v128 = (_QWORD *)(v126 + 8 * v127);
          }
          else
          {
            v97 = *(_QWORD *)(v66 + 880);
            if ( v97 != *(_QWORD *)(v66 + 872) )
            {
              v95 = (_QWORD *)(v97 + 8LL * *(unsigned int *)(v66 + 888));
              goto LABEL_116;
            }
            v95 = (_QWORD *)(v97 + 8LL * *(unsigned int *)(v66 + 892));
            v128 = v95;
          }
        }
        while ( v128 != v95 && *v95 >= 0xFFFFFFFFFFFFFFFELL )
          ++v95;
LABEL_116:
        if ( v96 != v95 )
          goto LABEL_117;
        v129 = *(_QWORD *)(v91 + 32);
        v130 = v129 + 40LL * *(unsigned int *)(v91 + 40);
        if ( v129 == v130 )
          goto LABEL_208;
        do
        {
          if ( !*(_BYTE *)v129 )
          {
            v131 = *(_DWORD *)(v129 + 8);
            if ( v131 > 0 )
            {
              if ( (*(_BYTE *)(v129 + 3) & 0x10) != 0 )
              {
                v192 = *(_QWORD *)(v66 + 368);
                if ( !v192 )
                  goto LABEL_519;
                v89 = *(_QWORD *)(v192 + 8);
                v193 = *(_DWORD *)(v89 + 24LL * (unsigned int)v131 + 16);
                v194 = v131 * (v193 & 0xF);
                v195 = (_WORD *)(*(_QWORD *)(v192 + 56) + 2LL * (v193 >> 4));
                v196 = v195 + 1;
                v197 = *v195 + v194;
LABEL_265:
                v198 = v196;
                while ( v198 )
                {
                  LODWORD(v89) = v197;
                  v199 = *(unsigned __int8 *)(*(_QWORD *)(v66 + 1040) + v197);
                  v200 = *(_DWORD *)(v66 + 1000);
                  if ( v199 < v200 )
                  {
                    v88 = *(_QWORD *)(v66 + 992);
                    while ( 1 )
                    {
                      v201 = (_DWORD *)(v88 + 4LL * v199);
                      if ( v197 == *v201 )
                        break;
                      v199 += 256;
                      if ( v200 <= v199 )
                        goto LABEL_275;
                    }
                    v202 = 4LL * v200;
                    LODWORD(v89) = v88 + v202;
                    if ( v201 != (_DWORD *)(v88 + v202) )
                    {
                      v203 = (_DWORD *)(v88 + v202 - 4);
                      if ( v201 != v203 )
                      {
                        *v201 = *v203;
                        v88 = *(_QWORD *)(v66 + 1040);
                        *(_BYTE *)(v88
                                 + *(unsigned int *)(*(_QWORD *)(v66 + 992) + 4LL * *(unsigned int *)(v66 + 1000) - 4)) = ((__int64)v201 - *(_QWORD *)(v66 + 992)) >> 2;
                        v200 = *(_DWORD *)(v66 + 1000);
                      }
                      *(_DWORD *)(v66 + 1000) = v200 - 1;
                    }
                  }
LABEL_275:
                  v204 = *v198;
                  v196 = 0;
                  ++v198;
                  if ( !v204 )
                    goto LABEL_265;
                  v197 += v204;
                }
              }
              v132 = *(_BYTE *)(v129 + 4);
              if ( (v132 & 1) == 0
                && (v132 & 2) == 0
                && ((*(_BYTE *)(v129 + 3) & 0x10) == 0 || (*(_DWORD *)v129 & 0xFFF00) != 0) )
              {
                v133 = v430.m128i_u32[2];
                if ( v430.m128i_i32[2] >= (unsigned __int32)v430.m128i_i32[3] )
                {
                  sub_16CD150((__int64)&v430, &v431, 0, 4, v88, v89);
                  v133 = v430.m128i_u32[2];
                }
                *(_DWORD *)(v430.m128i_i64[0] + 4 * v133) = v131;
                ++v430.m128i_i32[2];
              }
            }
          }
          v129 += 40;
        }
        while ( v130 != v129 );
LABEL_208:
        while ( v430.m128i_i32[2] )
        {
          v134 = *(_QWORD *)(v66 + 368);
          if ( !v134 )
          {
            --v430.m128i_i32[2];
            goto LABEL_519;
          }
          v135 = *(unsigned int *)(v430.m128i_i64[0] + 4LL * v430.m128i_u32[2] - 4);
          --v430.m128i_i32[2];
          v88 = *(_QWORD *)(v134 + 8);
          v136 = v135;
          v137 = *(_DWORD *)(v88 + 24 * v135 + 16);
          LOWORD(v135) = v137 & 0xF;
          v138 = (_WORD *)(*(_QWORD *)(v134 + 56) + 2LL * (v137 >> 4));
          v139 = v138 + 1;
          v140 = *v138 + v136 * v135;
          while ( 1 )
          {
            v141 = v139;
            if ( !v139 )
              break;
            while ( 1 )
            {
              if ( (*(_QWORD *)(*(_QWORD *)(v66 + 968) + 8LL * (v140 >> 6)) & (1LL << v140)) != 0 )
              {
                v143 = (_BYTE *)(*(_QWORD *)(v66 + 1040) + v140);
                v144 = *(_DWORD *)(v66 + 1000);
                v145 = (unsigned __int8)*v143;
                if ( v145 >= v144 )
                  goto LABEL_221;
                v146 = *(_QWORD *)(v66 + 992);
                while ( 1 )
                {
                  v147 = (_DWORD *)(v146 + 4LL * v145);
                  if ( v140 == *v147 )
                    break;
                  v145 += 256;
                  if ( v144 <= v145 )
                    goto LABEL_221;
                }
                if ( v147 == (_DWORD *)(v146 + 4LL * v144) )
                {
LABEL_221:
                  *v143 = v144;
                  v148 = *(unsigned int *)(v66 + 1000);
                  if ( (unsigned int)v148 >= *(_DWORD *)(v66 + 1004) )
                  {
                    sub_16CD150(v363, (const void *)(v66 + 1008), 0, 4, v88, v89);
                    v148 = *(unsigned int *)(v66 + 1000);
                  }
                  *(_DWORD *)(*(_QWORD *)(v66 + 992) + 4 * v148) = v140;
                  ++*(_DWORD *)(v66 + 1000);
                }
              }
              v142 = *v141;
              v139 = 0;
              ++v141;
              if ( !v142 )
                break;
              v140 += v142;
              if ( !v141 )
                goto LABEL_208;
            }
          }
        }
        if ( v91 == v371
          || ((v149 = *(_WORD *)(v91 + 46), (v149 & 4) != 0) || (v149 & 8) == 0
            ? (v150 = (*(_QWORD *)(*(_QWORD *)(v91 + 16) + 8LL) >> 6) & 1LL)
            : (LOBYTE(v150) = sub_1E15D00(v91, 64, 1)),
              !(_BYTE)v150) )
        {
          v151 = *(_DWORD *)(v66 + 1000);
          if ( !v151 )
            break;
        }
        if ( v381 == v91 )
          goto LABEL_117;
      }
      *(_QWORD *)(v66 + 1056) = v91;
      if ( (__m128i *)v430.m128i_i64[0] != &v431 )
        _libc_free(v430.m128i_u64[0]);
      if ( byte_4FC3400 )
        goto LABEL_284;
      v152 = *(_QWORD *)(v66 + 352);
      if ( !v152 )
      {
        v348 = sub_1E816F0(*(_QWORD *)(v66 + 344), 0);
        *(_QWORD *)(v66 + 352) = v348;
        v152 = v348;
      }
      v153 = *(_QWORD *)(v66 + 400);
      if ( v153 == *(_QWORD *)(v66 + 392) )
        v153 = *(_QWORD *)(v66 + 384);
      v154 = sub_1E84C70(v152, v153);
      v155 = *(_QWORD *)(v66 + 352);
      v156 = *(_QWORD *)(v66 + 408);
      v397 = v154;
      v398 = v157;
      if ( v156 == *(_QWORD *)(v66 + 392) )
        v156 = *(_QWORD *)(v66 + 384);
      v158 = sub_1E84C70(v155, v156);
      v400 = v160;
      v399 = v158;
      v161 = *(_DWORD *)(v398 + 36);
      if ( *(_DWORD *)(v160 + 36) <= v161 )
        v161 = *(_DWORD *)(v400 + 36);
      v162 = 0;
      v382 = *(_DWORD *)(v66 + 268) >> 1;
      v430.m128i_i64[0] = (__int64)&v431;
      v430.m128i_i64[1] = 0x100000000LL;
      if ( *(_QWORD *)(v66 + 400) != *(_QWORD *)(v66 + 392) )
      {
        v431.m128i_i64[0] = *(_QWORD *)(v66 + 400);
        v162 = 1;
        v430.m128i_i32[2] = 1;
      }
      if ( (unsigned int)sub_1E80E60((unsigned int)&v399, (unsigned int)&v431, v162, 0, 0, v159, 0, 0) > v382 + v161 )
      {
LABEL_117:
        v6 = v66;
        if ( (__m128i *)v430.m128i_i64[0] != &v431 )
          _libc_free(v430.m128i_u64[0]);
        goto LABEL_64;
      }
      v163 = sub_1E84C70(*(_QWORD *)(v66 + 352), *(_QWORD *)(v66 + 384));
      v164 = sub_1DD5EE0(*(_QWORD *)(v66 + 384));
      v165 = *(_DWORD *)(v163 + 400);
      v166 = v164;
      if ( v165 )
      {
        v167 = v165 - 1;
        v168 = *(_QWORD *)(v163 + 384);
        v169 = v167 & (((unsigned int)v164 >> 9) ^ ((unsigned int)v164 >> 4));
        v170 = (__int64 *)(v168 + 16LL * v169);
        v171 = *v170;
        if ( v166 == *v170 )
        {
LABEL_245:
          v165 = *((_DWORD *)v170 + 2);
        }
        else
        {
          v359 = 1;
          while ( v171 != -8 )
          {
            v360 = v359 + 1;
            v169 = v167 & (v359 + v169);
            v170 = (__int64 *)(v168 + 16LL * v169);
            v171 = *v170;
            if ( v166 == *v170 )
              goto LABEL_245;
            v359 = v360;
          }
          v165 = 0;
        }
      }
      v401[0] = sub_1E84C70(*(_QWORD *)(v66 + 352), *(_QWORD *)(v66 + 392));
      v172 = *(unsigned int *)(v66 + 424);
      v401[1] = v173;
      if ( (_DWORD)v172 )
      {
        v367 = 32 * v172;
        v386 = 0;
        while ( 1 )
        {
          v180 = (_QWORD *)(*(_QWORD *)(v66 + 416) + v386);
          v181 = sub_1E80C40(v401, *v180);
          v182 = *v180;
          v183 = v181;
          v184 = *(_DWORD *)(v401[0] + 400LL);
          if ( !v184 )
            goto LABEL_260;
          v185 = *(_QWORD *)(v401[0] + 384LL);
          v186 = v184 - 1;
          v187 = v186 & (((unsigned int)v182 >> 9) ^ ((unsigned int)v182 >> 4));
          v188 = (__int64 *)(v185 + 16LL * v187);
          v189 = *v188;
          if ( *v188 == v182 )
            goto LABEL_259;
          v345 = 1;
          if ( v189 != -8 )
            break;
LABEL_260:
          v190 = *((_DWORD *)v180 + 4);
          v191 = v190 + v165;
          if ( (v190 >= 0 || v191 <= v165) && v191 > v183 && v382 < v191 - v183 )
            goto LABEL_117;
          v174 = *((_DWORD *)v180 + 5);
          v175 = sub_1E80CC0(&v397, *v180);
          v176 = v174 + v175;
          if ( (v174 >= 0 || v175 >= v176) && v183 < v176 && v382 < v176 - v183 )
            goto LABEL_117;
          v177 = *((_DWORD *)v180 + 6);
          v178 = sub_1E80CC0(&v399, *v180);
          v179 = v177 + v178;
          if ( (v177 >= 0 || v178 >= v179) && v179 > v183 && v382 < v179 - v183 )
            goto LABEL_117;
          v386 += 32;
          if ( v386 == v367 )
          {
            v151 = 0;
            goto LABEL_282;
          }
        }
        while ( 1 )
        {
          v346 = v345 + 1;
          v187 = v186 & (v345 + v187);
          v188 = (__int64 *)(v185 + 16LL * v187);
          v347 = *v188;
          if ( v182 == *v188 )
            break;
          v345 = v346;
          if ( v347 == -8 )
            goto LABEL_260;
        }
LABEL_259:
        v183 += *((_DWORD *)v188 + 2);
        goto LABEL_260;
      }
LABEL_282:
      if ( (__m128i *)v430.m128i_i64[0] != &v431 )
        _libc_free(v430.m128i_u64[0]);
LABEL_284:
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v66 + 344) + 128LL))(*(_QWORD *)(v66 + 344));
      sub_1E80B50(*(_QWORD *)(v66 + 344), *(_QWORD *)(v66 + 384));
      sub_1E80B50(*(_QWORD *)(v66 + 344), *(_QWORD *)(v66 + 392));
      sub_1E80B50(*(_QWORD *)(v66 + 344), *(_QWORD *)(v66 + 400));
      sub_1E80B50(*(_QWORD *)(v66 + 344), *(_QWORD *)(v66 + 408));
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v66 + 344) + 128LL))(*(_QWORD *)(v66 + 344));
      v205 = *(_QWORD *)(v66 + 400);
      v401[0] = &v401[2];
      v401[1] = 0x400000000LL;
      v206 = *(_QWORD *)(v66 + 392);
      if ( v205 != v206 )
      {
        v387 = *(_QWORD *)(v66 + 384);
        v207 = (__int64 *)sub_1DD5EE0(v205);
        v208 = *(_QWORD *)(v66 + 400);
        v209 = *(__int64 **)(v208 + 32);
        if ( v207 != v209 )
        {
          v210 = *(__int64 **)(v66 + 1056);
          if ( v207 != v210 )
          {
            if ( v387 != v208 )
              sub_1DD5C00(v387 + 16, v208 + 16, *(_QWORD *)(v208 + 32), v207);
            if ( v210 != v207 && v209 != v207 )
            {
              v211 = *v207 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v209 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v207;
              *v207 = *v207 & 7 | *v209 & 0xFFFFFFFFFFFFFFF8LL;
              v212 = *v210;
              *(_QWORD *)(v211 + 8) = v210;
              v212 &= 0xFFFFFFFFFFFFFFF8LL;
              *v209 = v212 | *v209 & 7;
              *(_QWORD *)(v212 + 8) = v209;
              *v210 = v211 | *v210 & 7;
            }
          }
        }
        v206 = *(_QWORD *)(v66 + 392);
      }
      v213 = *(_QWORD *)(v66 + 408);
      if ( v206 != v213 )
      {
        v388 = *(_QWORD *)(v66 + 384);
        v214 = (__int64 *)sub_1DD5EE0(v213);
        v215 = *(_QWORD *)(v66 + 408);
        v216 = *(__int64 **)(v215 + 32);
        if ( v214 != v216 )
        {
          v217 = *(unsigned __int64 **)(v66 + 1056);
          if ( v217 != (unsigned __int64 *)v214 )
          {
            if ( v388 != v215 )
              sub_1DD5C00(v388 + 16, v215 + 16, *(_QWORD *)(v215 + 32), v214);
            if ( v217 != (unsigned __int64 *)v214 && v216 != v214 )
            {
              v218 = *v214 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v216 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v214;
              *v214 = *v214 & 7 | *v216 & 0xFFFFFFFFFFFFFFF8LL;
              v219 = *v217;
              *(_QWORD *)(v218 + 8) = v217;
              v219 &= 0xFFFFFFFFFFFFFFF8LL;
              *v216 = v219 | *v216 & 7;
              *(_QWORD *)(v219 + 8) = v216;
              *v217 = v218 | *v217 & 7;
            }
          }
        }
        v213 = *(_QWORD *)(v66 + 392);
      }
      v220 = (__int64)(*(_QWORD *)(v213 + 72) - *(_QWORD *)(v213 + 64)) >> 3;
      v221 = *(_QWORD *)(v66 + 384);
      v373 = v220;
      if ( (_DWORD)v220 == 2 )
      {
        v340 = sub_1DD5EE0(v221);
        v341 = *(_QWORD *)(v340 + 64);
        v342 = v340;
        v430.m128i_i64[0] = v341;
        if ( v341 )
        {
          sub_1623A60((__int64)&v430, v341, 2);
          v343 = *(_DWORD *)(v66 + 424);
          if ( !v343 )
            goto LABEL_318;
        }
        else
        {
          v343 = *(_DWORD *)(v66 + 424);
          if ( !v343 )
            goto LABEL_320;
        }
        do
        {
          v344 = (unsigned int *)(*(_QWORD *)(v66 + 416) + 32LL * v151++);
          (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __m128i *, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v66 + 360) + 368LL))(
            *(_QWORD *)(v66 + 360),
            *(_QWORD *)(v66 + 384),
            v342,
            &v430,
            *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v344 + 32LL) + 8LL),
            v344[2],
            *(_QWORD *)(v66 + 688),
            *(unsigned int *)(v66 + 696),
            v344[3]);
          sub_1E16240(*(_QWORD *)v344);
          *(_QWORD *)v344 = 0;
        }
        while ( v151 != v343 );
        goto LABEL_318;
      }
      v222 = sub_1DD5EE0(v221);
      v223 = *(_QWORD *)(v222 + 64);
      v383 = v222;
      v430.m128i_i64[0] = v223;
      if ( !v223 )
      {
        v389 = *(_DWORD *)(v66 + 424);
        if ( !v389 )
          goto LABEL_320;
        while ( 1 )
        {
LABEL_305:
          v224 = (unsigned int *)(*(_QWORD *)(v66 + 416) + 32LL * v151);
          LODWORD(v225) = v224[3];
          v226 = v224;
          if ( v224[2] != (_DWORD)v225 )
          {
            v225 = (unsigned int)sub_1E6B9A0(
                                   *(_QWORD *)(v66 + 376),
                                   *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v66 + 376) + 24LL)
                                             + 16LL
                                             * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)v224 + 32LL) + 8LL) & 0x7FFFFFFF))
                                 & 0xFFFFFFFFFFFFFFF8LL,
                                   byte_3F871B3,
                                   0);
            (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __m128i *, __int64, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v66 + 360) + 368LL))(
              *(_QWORD *)(v66 + 360),
              *(_QWORD *)(v66 + 384),
              v383,
              &v430,
              v225,
              v226[2],
              *(_QWORD *)(v66 + 688),
              *(unsigned int *)(v66 + 696),
              v226[3]);
          }
          v227 = *(_QWORD *)v226;
          v228 = *(_DWORD *)(*(_QWORD *)v226 + 40LL);
          if ( v228 != 1 )
            break;
LABEL_317:
          if ( v389 == ++v151 )
            goto LABEL_318;
        }
        while ( 1 )
        {
          v230 = *(_QWORD *)(v66 + 392);
          v231 = v228 - 1;
          v232 = *(_QWORD *)(v227 + 32) + 40 * v231;
          v233 = *(_QWORD *)(v66 + 400);
          v234 = *(_QWORD *)(v232 + 24);
          if ( v233 == v230 )
          {
            v228 -= 2;
            if ( v234 != *(_QWORD *)(v66 + 384) )
            {
LABEL_310:
              v229 = *(_QWORD *)(v66 + 408);
              if ( v230 == v229 )
              {
                if ( v234 == *(_QWORD *)(v66 + 384) )
                  goto LABEL_402;
              }
              else
              {
                if ( v234 != v229 )
                  goto LABEL_312;
LABEL_402:
                sub_1E16C90(v227, v231);
                sub_1E16C90(*(_QWORD *)v226, v228);
              }
LABEL_312:
              if ( v228 == 1 )
                goto LABEL_317;
              goto LABEL_313;
            }
          }
          else
          {
            v228 -= 2;
            if ( v234 != v233 )
              goto LABEL_310;
          }
          *(_QWORD *)(v232 + 24) = *(_QWORD *)(v66 + 384);
          sub_1E310D0(*(_QWORD *)(*(_QWORD *)v226 + 32LL) + 40LL * v228, (unsigned int)v225);
          if ( v228 == 1 )
            goto LABEL_317;
LABEL_313:
          v227 = *(_QWORD *)v226;
        }
      }
      sub_1623A60((__int64)&v430, v223, 2);
      v389 = *(_DWORD *)(v66 + 424);
      if ( v389 )
        goto LABEL_305;
LABEL_318:
      if ( v430.m128i_i64[0] )
        sub_161E7C0((__int64)&v430, v430.m128i_i64[0]);
LABEL_320:
      sub_1DD91B0(*(_QWORD *)(v66 + 384), *(_QWORD *)(v66 + 400), 0);
      sub_1DD91B0(*(_QWORD *)(v66 + 384), *(_QWORD *)(v66 + 408), 1);
      v235 = *(_QWORD *)(v66 + 400);
      v236 = *(_QWORD *)(v66 + 392);
      if ( v235 != v236 )
      {
        sub_1DD91B0(v235, v236, 1);
        v236 = *(_QWORD *)(v66 + 392);
      }
      v237 = *(_QWORD *)(v66 + 408);
      if ( v236 != v237 )
        sub_1DD91B0(v237, v236, 1);
      v238 = *(_QWORD *)(sub_1DD5EE0(*(_QWORD *)(v66 + 384)) + 64);
      v399 = v238;
      if ( v238 )
        sub_1623A60((__int64)&v399, v238, 2);
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v66 + 360) + 280LL))(
        *(_QWORD *)(v66 + 360),
        *(_QWORD *)(v66 + 384),
        0);
      v241 = *(_QWORD *)(v66 + 392);
      if ( *(_QWORD *)(v66 + 400) != v241 )
      {
        v242 = LODWORD(v401[1]);
        if ( LODWORD(v401[1]) >= HIDWORD(v401[1]) )
        {
          sub_16CD150((__int64)v401, &v401[2], 0, 8, v239, v240);
          v242 = LODWORD(v401[1]);
        }
        *(_QWORD *)(v401[0] + 8 * v242) = *(_QWORD *)(v66 + 400);
        v243 = *(_QWORD *)(v66 + 400);
        ++LODWORD(v401[1]);
        sub_1DD6E70(v243);
        v241 = *(_QWORD *)(v66 + 392);
      }
      if ( *(_QWORD *)(v66 + 408) != v241 )
      {
        v244 = LODWORD(v401[1]);
        if ( LODWORD(v401[1]) >= HIDWORD(v401[1]) )
        {
          sub_16CD150((__int64)v401, &v401[2], 0, 8, v239, v240);
          v244 = LODWORD(v401[1]);
        }
        *(_QWORD *)(v401[0] + 8 * v244) = *(_QWORD *)(v66 + 408);
        v245 = *(_QWORD *)(v66 + 408);
        ++LODWORD(v401[1]);
        sub_1DD6E70(v245);
        v241 = *(_QWORD *)(v66 + 392);
      }
      if ( v373 == 2 )
      {
        if ( !(unsigned __int8)sub_1DD69A0(*(_QWORD *)(v66 + 384), v241) )
        {
          v241 = *(_QWORD *)(v66 + 392);
          goto LABEL_335;
        }
        v349 = *(_QWORD *)(v66 + 392);
        v350 = *(_QWORD *)(v66 + 384);
        v351 = *(__int64 **)(v349 + 32);
        v352 = (__int64 *)(v349 + 24);
        if ( (__int64 *)(v349 + 24) != v351 && v352 != (__int64 *)(v350 + 24) )
        {
          if ( v350 != v349 )
          {
            v392 = *(__int64 **)(v349 + 32);
            sub_1DD5C00(v350 + 16, v349 + 16, v392, v349 + 24);
            v351 = v392;
          }
          if ( v352 != v351 )
          {
            v353 = *(_QWORD *)(v349 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v351 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v352;
            *(_QWORD *)(v349 + 24) = *(_QWORD *)(v349 + 24) & 7LL | *v351 & 0xFFFFFFFFFFFFFFF8LL;
            v354 = *(_QWORD *)(v350 + 24);
            *(_QWORD *)(v353 + 8) = v350 + 24;
            v354 &= 0xFFFFFFFFFFFFFFF8LL;
            *v351 = v354 | *v351 & 7;
            *(_QWORD *)(v354 + 8) = v351;
            *(_QWORD *)(v350 + 24) = v353 | *(_QWORD *)(v350 + 24) & 7LL;
          }
          v350 = *(_QWORD *)(v66 + 384);
          v349 = *(_QWORD *)(v66 + 392);
        }
        sub_1DD9280(v350, v349);
        v357 = LODWORD(v401[1]);
        if ( LODWORD(v401[1]) >= HIDWORD(v401[1]) )
        {
          sub_16CD150((__int64)v401, &v401[2], 0, 8, v355, v356);
          v357 = LODWORD(v401[1]);
        }
        *(_QWORD *)(v401[0] + 8 * v357) = *(_QWORD *)(v66 + 392);
        v358 = *(_QWORD *)(v66 + 392);
        ++LODWORD(v401[1]);
        sub_1DD6E70(v358);
      }
      else
      {
LABEL_335:
        v246 = *(_QWORD *)(v66 + 360);
        v430.m128i_i64[1] = 0;
        v247 = *(_QWORD *)(v66 + 384);
        v430.m128i_i64[0] = (__int64)&v431;
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, __m128i *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v246 + 288LL))(
          v246,
          v247,
          v241,
          0,
          &v431,
          0,
          &v399,
          0);
        sub_1DD8FE0(*(_QWORD *)(v66 + 384), *(_QWORD *)(v66 + 392), 0xFFFFFFFFLL);
        if ( (__m128i *)v430.m128i_i64[0] != &v431 )
          _libc_free(v430.m128i_u64[0]);
      }
      if ( v399 )
        sub_161E7C0((__int64)&v399, v399);
      v248 = *(_QWORD *)(v66 + 328);
      v249 = (__int64 *)v401[0];
      v250 = v401[1];
      v251 = *(_QWORD *)(v66 + 384);
      sub_1E06620(v248);
      v252 = *(_QWORD *)(v248 + 1312);
      v253 = 0;
      v254 = *(_DWORD *)(v252 + 48);
      if ( v254 )
      {
        v255 = *(_QWORD *)(v252 + 32);
        v256 = (v254 - 1) & (((unsigned int)v251 >> 9) ^ ((unsigned int)v251 >> 4));
        v257 = (__int64 *)(v255 + 16LL * v256);
        v258 = *v257;
        if ( v251 == *v257 )
        {
LABEL_341:
          if ( v257 != (__int64 *)(v255 + 16LL * v254) )
          {
            v253 = v257[1];
            goto LABEL_343;
          }
        }
        else
        {
          v361 = 1;
          while ( v258 != -8 )
          {
            v362 = v361 + 1;
            v256 = (v254 - 1) & (v361 + v256);
            v257 = (__int64 *)(v255 + 16LL * v256);
            v258 = *v257;
            if ( v251 == *v257 )
              goto LABEL_341;
            v361 = v362;
          }
        }
        v253 = 0;
      }
LABEL_343:
      if ( v250 )
      {
        v259 = (unsigned int)(v250 - 1);
        v384 = v66;
        v260 = &v431;
        v261 = (__int64)&v249[v259 + 1];
        v390 = v249;
        v262 = &v430;
        v368 = (__int64 *)v261;
        while ( 1 )
        {
          v263 = *(_QWORD *)(v384 + 328);
          v374 = *v390;
          sub_1E06620(v263);
          v264 = *(_QWORD *)(v263 + 1312);
          v265 = *(unsigned int *)(v264 + 48);
          if ( !(_DWORD)v265 )
LABEL_522:
            BUG();
          v266 = *(_QWORD *)(v264 + 32);
          v267 = (v265 - 1) & (((unsigned int)v374 >> 9) ^ ((unsigned int)v374 >> 4));
          v268 = (__int64 *)(v266 + 16LL * v267);
          v269 = *v268;
          if ( *v268 != v374 )
          {
            v330 = 1;
            if ( v269 == -8 )
              goto LABEL_522;
            while ( 1 )
            {
              v331 = v330 + 1;
              v267 = (v265 - 1) & (v330 + v267);
              v268 = (__int64 *)(v266 + 16LL * v267);
              v332 = *v268;
              if ( v374 == *v268 )
                break;
              v330 = v331;
              if ( v332 == -8 )
                goto LABEL_522;
            }
          }
          if ( v268 == (__int64 *)(v266 + 16 * v265) )
            goto LABEL_522;
          v270 = v268[1];
          v271 = *(_QWORD *)(v270 + 32);
          if ( *(_QWORD *)(v270 + 24) != v271 )
          {
            v369 = v253 + 24;
            do
            {
              v272 = *(_QWORD *)(v271 - 8);
              v375 = *(_QWORD *)(v384 + 328);
              sub_1E06620(v375);
              *(_BYTE *)(*(_QWORD *)(v375 + 1312) + 72LL) = 0;
              v273 = *(_QWORD *)(v272 + 8);
              if ( v253 != v273 )
              {
                v430.m128i_i64[0] = v272;
                v274 = sub_1D82110(*(_QWORD **)(v273 + 24), *(_QWORD *)(v273 + 32), v262->m128i_i64);
                sub_1D82C50(*(_QWORD *)(v272 + 8) + 24LL, v274);
                *(_QWORD *)(v272 + 8) = v253;
                v430.m128i_i64[0] = v272;
                v276 = *(_BYTE **)(v253 + 32);
                if ( v276 == *(_BYTE **)(v253 + 40) )
                {
                  sub_1D82C90(v369, v276, v262);
                }
                else
                {
                  if ( v276 )
                  {
                    *(_QWORD *)v276 = v272;
                    v276 = *(_BYTE **)(v253 + 32);
                  }
                  *(_QWORD *)(v253 + 32) = v276 + 8;
                }
                if ( *(_DWORD *)(v272 + 16) != *(_DWORD *)(*(_QWORD *)(v272 + 8) + 16LL) + 1 )
                {
                  v277 = (unsigned __int64)v260;
                  v278 = v260;
                  v279 = v270;
                  v431.m128i_i64[0] = v272;
                  v430.m128i_i64[0] = (__int64)v394;
                  v430.m128i_i64[1] = 0x4000000001LL;
                  LODWORD(v280) = 1;
                  v394 = v260;
                  v281 = v253;
                  do
                  {
                    v282 = (unsigned int)v280;
                    v280 = (unsigned int)(v280 - 1);
                    v283 = *(_QWORD *)(v277 + 8 * v282 - 8);
                    v430.m128i_i32[2] = v280;
                    v284 = *(__int64 **)(v283 + 32);
                    v285 = *(__int64 **)(v283 + 24);
                    *(_DWORD *)(v283 + 16) = *(_DWORD *)(*(_QWORD *)(v283 + 8) + 16LL) + 1;
                    if ( v285 != v284 )
                    {
                      v366 = v279;
                      v286 = (__int64)v262;
                      v287 = v284;
                      do
                      {
                        v288 = *v285;
                        if ( *(_DWORD *)(*v285 + 16) != *(_DWORD *)(*(_QWORD *)(*v285 + 8) + 16LL) + 1 )
                        {
                          if ( (unsigned int)v280 >= v430.m128i_i32[3] )
                          {
                            sub_16CD150(v286, v278, 0, 8, v279, v275);
                            v280 = v430.m128i_u32[2];
                          }
                          *(_QWORD *)(v430.m128i_i64[0] + 8 * v280) = v288;
                          v280 = (unsigned int)++v430.m128i_i32[2];
                        }
                        ++v285;
                      }
                      while ( v287 != v285 );
                      v279 = v366;
                      v262 = (__m128i *)v286;
                      v277 = v430.m128i_i64[0];
                    }
                  }
                  while ( (_DWORD)v280 );
                  v253 = v281;
                  v270 = v279;
                  v260 = v278;
                  if ( (__m128i *)v277 != v278 )
                    _libc_free(v277);
                }
              }
              v271 = *(_QWORD *)(v270 + 32);
            }
            while ( v271 != *(_QWORD *)(v270 + 24) );
          }
          v289 = *(_QWORD *)(v384 + 328);
          v376 = *v390;
          sub_1E06620(v289);
          v290 = *(_QWORD *)(v289 + 1312);
          v291 = *(unsigned int *)(v290 + 48);
          if ( !(_DWORD)v291 )
          {
            v336 = v290;
LABEL_521:
            v430.m128i_i64[0] = 0;
            *(_BYTE *)(v336 + 72) = 0;
            BUG();
          }
          v292 = v376;
          v293 = v291 - 1;
          v294 = *(_QWORD *)(v290 + 32);
          v295 = ((unsigned int)v376 >> 4) ^ ((unsigned int)v376 >> 9);
          v296 = (v291 - 1) & v295;
          v297 = (__int64 *)(v294 + 16LL * v296);
          v298 = *v297;
          if ( v376 != *v297 )
          {
            v333 = 1;
            if ( v298 == -8 )
            {
LABEL_444:
              v336 = v290;
              goto LABEL_521;
            }
            while ( 1 )
            {
              v334 = v333 + 1;
              v296 = v293 & (v333 + v296);
              v297 = (__int64 *)(v294 + 16LL * v296);
              v335 = *v297;
              if ( v376 == *v297 )
                break;
              v333 = v334;
              if ( v335 == -8 )
                goto LABEL_444;
            }
          }
          if ( v297 == (__int64 *)(v294 + 16 * v291) )
          {
            v336 = v290;
            goto LABEL_521;
          }
          v430.m128i_i64[0] = v297[1];
          v299 = v430.m128i_i64[0];
          *(_BYTE *)(v290 + 72) = 0;
          v300 = *(_QWORD *)(v299 + 8);
          if ( v300 )
          {
            v301 = sub_1D82110(*(_QWORD **)(v300 + 24), *(_QWORD *)(v300 + 32), v262->m128i_i64);
            sub_1D82C50(v302 + 24, v301);
            v303 = *(_DWORD *)(v290 + 48);
            v294 = *(_QWORD *)(v290 + 32);
            if ( !v303 )
              goto LABEL_374;
            v295 = ((unsigned int)v376 >> 4) ^ ((unsigned int)v376 >> 9);
            v292 = v376;
            v293 = v303 - 1;
          }
          v325 = v293 & v295;
          v326 = (_QWORD *)(v294 + 16LL * v325);
          v327 = *v326;
          if ( v292 == *v326 )
            goto LABEL_419;
          v337 = 1;
          if ( v327 != -8 )
          {
            while ( 1 )
            {
              v338 = v337 + 1;
              v325 = v293 & (v337 + v325);
              v326 = (_QWORD *)(v294 + 16LL * v325);
              v339 = *v326;
              if ( v292 == *v326 )
                break;
              v337 = v338;
              if ( v339 == -8 )
                goto LABEL_374;
            }
LABEL_419:
            v328 = v326[1];
            if ( v328 )
            {
              v329 = *(_QWORD *)(v328 + 24);
              if ( v329 )
              {
                v370 = v326;
                v377 = v326[1];
                j_j___libc_free_0(v329, *(_QWORD *)(v328 + 40) - v329);
                v326 = v370;
                v328 = v377;
              }
              v378 = v326;
              j_j___libc_free_0(v328, 56);
              v326 = v378;
            }
            *v326 = -16;
            --*(_DWORD *)(v290 + 40);
            ++*(_DWORD *)(v290 + 44);
          }
LABEL_374:
          if ( ++v390 == v368 )
          {
            v66 = v384;
            break;
          }
        }
      }
      v304 = *(_QWORD *)(v66 + 336);
      v305 = (_QWORD *)v401[0];
      if ( !v304 || !LODWORD(v401[1]) )
        goto LABEL_433;
      v306 = (__int64 *)v401[0];
      v307 = v66;
      v396 = v401[0] + 8LL * (unsigned int)(LODWORD(v401[1]) - 1) + 8;
      while ( 2 )
      {
        v308 = *(unsigned int *)(v304 + 256);
        if ( !(_DWORD)v308 )
          goto LABEL_399;
        v309 = *(_QWORD *)(v304 + 240);
        v310 = 1;
        v311 = (v308 - 1) & (((unsigned int)*v306 >> 9) ^ ((unsigned int)*v306 >> 4));
        v312 = (_QWORD *)(v309 + 16LL * v311);
        if ( *v306 != *v312 )
        {
          if ( *v312 == -8 )
            goto LABEL_399;
          while ( 1 )
          {
            v311 = (v308 - 1) & (v310 + v311);
            v312 = (_QWORD *)(v309 + 16LL * v311);
            if ( *v306 == *v312 )
              break;
            ++v310;
            if ( *v312 == -8 )
              goto LABEL_399;
          }
        }
        if ( v312 == (_QWORD *)(v309 + 16 * v308) )
          goto LABEL_399;
        if ( !v312[1] )
          goto LABEL_398;
        v391 = v307;
        v313 = *v306;
        v314 = (__int64 *)v312[1];
        while ( 2 )
        {
          v315 = (char *)v314[5];
          v316 = (char *)v314[4];
          v317 = (v315 - v316) >> 5;
          v318 = (v315 - v316) >> 3;
          if ( v317 <= 0 )
          {
LABEL_409:
            if ( v318 != 2 )
            {
              if ( v318 != 3 )
              {
                if ( v318 != 1 )
                {
                  v316 = (char *)v314[5];
                  goto LABEL_391;
                }
LABEL_429:
                if ( v313 != *(_QWORD *)v316 )
                  v316 = (char *)v314[5];
                goto LABEL_391;
              }
              if ( v313 == *(_QWORD *)v316 )
                goto LABEL_391;
              v316 += 8;
            }
            if ( v313 == *(_QWORD *)v316 )
              goto LABEL_391;
            v316 += 8;
            goto LABEL_429;
          }
          v319 = &v316[32 * v317];
          while ( v313 != *(_QWORD *)v316 )
          {
            if ( v313 == *((_QWORD *)v316 + 1) )
            {
              v316 += 8;
              break;
            }
            if ( v313 == *((_QWORD *)v316 + 2) )
            {
              v316 += 16;
              break;
            }
            if ( v313 == *((_QWORD *)v316 + 3) )
            {
              v316 += 24;
              break;
            }
            v316 += 32;
            if ( v319 == v316 )
            {
              v318 = (v315 - v316) >> 3;
              goto LABEL_409;
            }
          }
LABEL_391:
          if ( v316 + 8 != v315 )
          {
            memmove(v316, v316 + 8, v315 - (v316 + 8));
            v315 = (char *)v314[5];
          }
          v320 = (_QWORD *)v314[8];
          v314[5] = (__int64)(v315 - 8);
          if ( (_QWORD *)v314[9] == v320 )
          {
            v324 = &v320[*((unsigned int *)v314 + 21)];
            if ( v320 == v324 )
            {
LABEL_467:
              v320 = v324;
            }
            else
            {
              while ( v313 != *v320 )
              {
                if ( v324 == ++v320 )
                  goto LABEL_467;
              }
            }
          }
          else
          {
            v320 = sub_16CC9F0((__int64)(v314 + 7), v313);
            if ( v313 == *v320 )
            {
              v322 = v314[9];
              if ( v322 == v314[8] )
                v323 = *((unsigned int *)v314 + 21);
              else
                v323 = *((unsigned int *)v314 + 20);
              v324 = (_QWORD *)(v322 + 8 * v323);
            }
            else
            {
              v321 = v314[9];
              if ( v321 != v314[8] )
                goto LABEL_396;
              v320 = (_QWORD *)(v321 + 8LL * *((unsigned int *)v314 + 21));
              v324 = v320;
            }
          }
          if ( v320 != v324 )
          {
            *v320 = -2;
            ++*((_DWORD *)v314 + 22);
          }
LABEL_396:
          v314 = (__int64 *)*v314;
          if ( v314 )
            continue;
          break;
        }
        v307 = v391;
LABEL_398:
        *v312 = -16;
        --*(_DWORD *)(v304 + 248);
        ++*(_DWORD *)(v304 + 252);
LABEL_399:
        if ( (__int64 *)v396 != ++v306 )
        {
          v304 = *(_QWORD *)(v307 + 336);
          continue;
        }
        break;
      }
      v305 = (_QWORD *)v401[0];
      v66 = v307;
LABEL_433:
      if ( v305 != &v401[2] )
        _libc_free((unsigned __int64)v305);
      *(_QWORD *)(v66 + 392) = 0;
      *(_QWORD *)(v66 + 408) = 0;
      *(_QWORD *)(v66 + 384) = v364;
      *(_QWORD *)(v66 + 400) = 0;
      v65 = *(_QWORD **)(v364 + 88);
      if ( (unsigned int)((__int64)(*(_QWORD *)(v364 + 96) - (_QWORD)v65) >> 3) != 2 )
      {
        v61 = v409;
        v6 = v66;
        v380 = v379;
        goto LABEL_67;
      }
      v372 = v379;
    }
LABEL_63:
    v6 = v66;
LABEL_64:
    v62 = v380;
    v61 = v409;
    if ( v372 )
      v62 = v372;
    v380 = v62;
LABEL_67:
    v63 = v61 - 1;
    v61 = v408;
    v409 = v63;
    v60 = v408;
    if ( v63 != v408 )
    {
      sub_1D82FA0((__int64)&v402);
      v60 = v408;
      v61 = v409;
    }
    v58 = v415;
    v59 = v416;
  }
  if ( v61 != v60 )
  {
    v111 = (__int64 *)v58;
    while ( v60->m128i_i64[0] == *v111 && v60->m128i_i64[1] == v111[1] )
    {
      ++v60;
      v111 += 2;
      if ( v61 == v60 )
        goto LABEL_142;
    }
    goto LABEL_71;
  }
LABEL_142:
  if ( v58 )
    j_j___libc_free_0(v58, v417 - (__int8 *)v58);
  if ( v413 != v412 )
    _libc_free(v413);
  if ( v408 )
    j_j___libc_free_0(v408, v410 - (__int8 *)v408);
  if ( v404 != v403 )
    _libc_free((unsigned __int64)v404);
  if ( v427 )
    j_j___libc_free_0(v427, v429 - (_QWORD)v427);
  if ( v425 != v424[1] )
    _libc_free(v425);
  if ( v421 )
    j_j___libc_free_0(v421, v423 - (__int8 *)v421);
  if ( v419 != v418[1] )
    _libc_free(v419);
  return v380;
}
