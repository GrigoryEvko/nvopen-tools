// Function: sub_1D7AA30
// Address: 0x1d7aa30
//
__int64 __fastcall sub_1D7AA30(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128 a6,
        double a7,
        double a8,
        __m128i a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 v11; // rax
  void *v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // rdx
  int v15; // eax
  unsigned int v16; // esi
  _QWORD *v17; // rdx
  _QWORD *i; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 (*v26)(); // rdx
  __int64 v27; // rax
  __int64 (*v28)(void); // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rax
  __int64 v43; // r12
  _QWORD *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r13
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  _QWORD *v49; // rbx
  _QWORD *v50; // r14
  __int64 v51; // rdi
  __int64 *v52; // rax
  __int64 *v53; // rbx
  __int64 *v54; // r12
  char v55; // al
  __int64 *v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  double v64; // xmm4_8
  double v65; // xmm5_8
  __int64 v66; // r12
  char v67; // bl
  double v68; // xmm4_8
  double v69; // xmm5_8
  __int64 v70; // r13
  __int64 v71; // r14
  _QWORD *v72; // r12
  __int64 v73; // rax
  _QWORD *v74; // r15
  __int64 v75; // rcx
  double v76; // xmm4_8
  double v77; // xmm5_8
  int v78; // eax
  __int64 v79; // rdx
  _QWORD *v80; // rax
  _QWORD *j; // rdx
  int v82; // r13d
  _QWORD *v83; // rbx
  _QWORD *v84; // r12
  unsigned int v85; // eax
  unsigned __int64 v86; // rdi
  void *v87; // rdi
  unsigned int v88; // eax
  __int64 v89; // rdx
  int v90; // r13d
  _QWORD *v91; // rbx
  unsigned int v92; // edx
  __int64 v93; // rax
  _QWORD *v94; // r12
  unsigned __int64 v95; // rdi
  int v96; // eax
  __int64 v97; // rdx
  _QWORD *v98; // rax
  _QWORD *ii; // rdx
  __int64 v100; // rcx
  _QWORD *v101; // rax
  int v102; // r12d
  _QWORD *v103; // r14
  unsigned __int8 v104; // al
  _QWORD *v105; // rbx
  __int64 v106; // rsi
  __int64 v107; // r12
  unsigned __int8 v108; // al
  unsigned int v109; // edx
  __int64 v110; // r13
  char v111; // al
  double v112; // xmm4_8
  double v113; // xmm5_8
  char v114; // bl
  bool v115; // zf
  unsigned int v116; // edx
  unsigned int v117; // eax
  int v118; // eax
  unsigned __int64 v119; // r13
  unsigned int v120; // eax
  unsigned __int64 v121; // rdi
  unsigned int v122; // edx
  unsigned __int64 v123; // rdi
  int v124; // edx
  char v125; // bl
  __int64 v126; // r13
  __int64 v127; // r13
  __int64 v128; // rdi
  __int64 v129; // rdi
  __int64 v130; // rcx
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // r12
  char v134; // al
  __int64 v135; // rsi
  __int64 *v136; // rax
  __int64 v137; // rdx
  __int64 *v138; // r12
  __int64 v139; // rdi
  __int64 *v140; // rbx
  __int64 *v141; // rax
  __int64 v142; // r14
  __int64 v143; // rbx
  __int64 mm; // r15
  __int64 v145; // r12
  int v146; // r8d
  int v147; // r9d
  __int64 v148; // rax
  __int64 *v149; // rbx
  _QWORD *v150; // r13
  unsigned __int8 v151; // r14
  __int64 v152; // rax
  __int64 v153; // r15
  unsigned int v154; // r13d
  _QWORD *v155; // rax
  int v156; // r8d
  int v157; // r9d
  __int64 v158; // rcx
  double v159; // xmm4_8
  double v160; // xmm5_8
  _QWORD *v161; // r8
  __int64 v162; // rax
  __int64 *v163; // r15
  __int64 *v164; // rcx
  __int64 v165; // rdi
  __int64 *v166; // r13
  _QWORD *v167; // r15
  unsigned __int64 v168; // rdi
  unsigned int v170; // eax
  _QWORD *v171; // rbx
  _QWORD *v172; // r12
  __int64 v173; // rsi
  __int64 v174; // rax
  __int64 v175; // rax
  __int64 v176; // rsi
  unsigned __int8 v177; // al
  unsigned __int64 v178; // rax
  __int64 v179; // rcx
  _QWORD *v180; // rdi
  unsigned int v181; // eax
  int v182; // eax
  unsigned __int64 v183; // rax
  unsigned __int64 v184; // rax
  int v185; // ebx
  __int64 v186; // r12
  _QWORD *v187; // rdi
  unsigned int v188; // eax
  int v189; // eax
  unsigned __int64 v190; // rax
  unsigned __int64 v191; // rax
  int v192; // ebx
  __int64 v193; // r12
  _QWORD *v194; // rax
  __int64 v195; // rdx
  _QWORD *k; // rdx
  __int64 v197; // rsi
  int v198; // r8d
  int v199; // r9d
  _QWORD *v200; // r13
  _QWORD *v201; // rbx
  __int64 v202; // r12
  __int64 v203; // rdi
  char v204; // cl
  __int64 *v205; // rax
  __int64 v206; // rbx
  unsigned int v207; // eax
  _QWORD *v208; // rdi
  unsigned __int64 v209; // rax
  __int64 v210; // rax
  _QWORD *v211; // rax
  __int64 v212; // rdx
  _QWORD *m; // rdx
  unsigned __int64 *v214; // rax
  __int64 v215; // r14
  __int64 v216; // rax
  __int64 v217; // rbx
  unsigned __int64 v218; // rdi
  unsigned __int64 v219; // rax
  int v220; // r8d
  int v221; // r9d
  __int64 v222; // r12
  int v223; // eax
  __int64 *v224; // r13
  unsigned int kk; // ebx
  __int64 v226; // rax
  __int64 v227; // rdx
  __int64 v228; // rcx
  __int64 v229; // r8
  __int64 *v230; // r9
  __int64 v231; // rax
  char v232; // bl
  __int64 v233; // r12
  _QWORD *v234; // rdi
  int v235; // edx
  unsigned int v236; // eax
  __int64 *v237; // rcx
  __int64 v238; // rsi
  unsigned __int64 v239; // rdi
  double v240; // xmm4_8
  double v241; // xmm5_8
  unsigned __int64 v242; // rax
  int v243; // r8d
  int v244; // r9d
  unsigned __int64 v245; // r13
  int v246; // eax
  __int64 *v247; // rbx
  __int64 v248; // rax
  unsigned int v249; // r14d
  __int64 v250; // r12
  __int64 v251; // r13
  __int64 v252; // rax
  __int64 v253; // rdx
  __int64 v254; // rcx
  __int64 v255; // r8
  __int64 *v256; // r9
  __int64 v257; // rbx
  __int64 **v258; // r12
  __int64 v259; // r13
  __int64 *v260; // rsi
  __int64 *v261; // r12
  __int64 *v262; // rax
  unsigned int v263; // eax
  unsigned __int64 *v264; // rdi
  __int64 v265; // rax
  __int64 v266; // r12
  __int64 **v267; // r13
  __int64 v268; // r14
  __int64 v269; // rax
  unsigned __int64 v270; // rax
  double v271; // xmm4_8
  double v272; // xmm5_8
  __int64 v273; // r12
  __int64 v274; // rax
  unsigned __int8 v275; // r12
  __int64 v276; // r12
  __int64 **v277; // r13
  __int64 v278; // rbx
  __int64 *v279; // rsi
  unsigned int v280; // eax
  _QWORD *v281; // r12
  _QWORD *v282; // rbx
  __int64 v283; // rsi
  _QWORD *v284; // rax
  __int64 v285; // r12
  _QWORD *v286; // r15
  __int64 *v287; // r11
  __int64 *v288; // r15
  __int64 v289; // r13
  _QWORD *v290; // r12
  _QWORD *v291; // rax
  __int64 v292; // rax
  _QWORD *v293; // rdx
  __int64 v294; // r14
  _QWORD *v295; // rax
  __int64 v296; // rax
  __int64 v297; // rax
  __int64 v298; // r13
  __int64 *v299; // r12
  __int64 v300; // r14
  char v301; // al
  double v302; // xmm4_8
  double v303; // xmm5_8
  char v304; // al
  double v305; // xmm4_8
  double v306; // xmm5_8
  __int64 v307; // rax
  __int64 v308; // rdx
  __int64 v309; // rcx
  __int64 v310; // rax
  _QWORD *v311; // r14
  __int64 v312; // rdi
  _QWORD *v313; // rdx
  __int64 v314; // rax
  __int64 v315; // r14
  __int64 v316; // rsi
  __int64 v317; // rax
  __int64 v318; // rax
  char v319; // bl
  __int64 v320; // rax
  unsigned __int64 v321; // rax
  __int64 v322; // r12
  __int64 v323; // rdx
  char v324; // r14
  unsigned __int64 v325; // rax
  __int64 v326; // r13
  __int16 v327; // ax
  __int64 v328; // r9
  __int64 v329; // rdx
  __int64 v330; // r14
  __int64 v331; // rax
  int v332; // eax
  int v333; // eax
  __int64 v334; // rax
  __int64 v335; // r8
  __int64 *v336; // rax
  __int64 v337; // rdx
  _QWORD *v338; // rax
  __int64 v339; // r9
  __int64 v340; // rbx
  __int64 v341; // rax
  _QWORD *v342; // rax
  __int64 v343; // r12
  __int64 v344; // rsi
  __int64 v345; // rax
  __int64 v346; // rcx
  __int64 v347; // rax
  __int64 v348; // rdx
  __int64 v349; // r8
  unsigned int v350; // esi
  int v351; // edx
  __int64 v352; // rcx
  __int64 v353; // rdi
  int v354; // ebx
  unsigned int v355; // eax
  _QWORD *v356; // rdi
  unsigned __int64 v357; // rdx
  unsigned __int64 v358; // rax
  _QWORD *v359; // rax
  __int64 v360; // rdx
  _QWORD *n; // rdx
  int v362; // ecx
  int v363; // r8d
  _QWORD *v364; // rax
  _QWORD *v365; // rax
  _QWORD *v366; // rax
  _QWORD *v367; // rax
  __int64 v368; // rdx
  _QWORD *jj; // rdx
  _QWORD *v370; // rax
  __int64 v371; // rdx
  __int64 v372; // rdx
  __int64 v373; // r8
  __int64 v374; // r9
  __int64 v375; // r14
  __int64 v376; // rcx
  __int64 v377; // rbx
  __int64 v378; // rdx
  char v379; // si
  unsigned int v380; // eax
  __int64 v381; // rdi
  __int64 v382; // rdx
  __int64 v383; // rsi
  __int64 v384; // rax
  __int64 v385; // rax
  __int64 v386; // rax
  __int64 v387; // rdx
  __int64 v388; // rax
  __int64 v389; // rax
  __int64 v390; // rax
  __int64 v391; // [rsp+10h] [rbp-2F0h]
  __int64 *v392; // [rsp+18h] [rbp-2E8h]
  __int64 v393; // [rsp+18h] [rbp-2E8h]
  __int64 v394; // [rsp+20h] [rbp-2E0h]
  __int64 v395; // [rsp+20h] [rbp-2E0h]
  unsigned __int8 v396; // [rsp+28h] [rbp-2D8h]
  __int64 v397; // [rsp+28h] [rbp-2D8h]
  __int64 *v398; // [rsp+28h] [rbp-2D8h]
  __int64 v399; // [rsp+38h] [rbp-2C8h]
  __int64 *v401; // [rsp+48h] [rbp-2B8h]
  char v402; // [rsp+50h] [rbp-2B0h]
  _QWORD *v403; // [rsp+50h] [rbp-2B0h]
  __int64 v404; // [rsp+50h] [rbp-2B0h]
  bool v405; // [rsp+50h] [rbp-2B0h]
  unsigned __int8 v406; // [rsp+58h] [rbp-2A8h]
  _QWORD *v407; // [rsp+58h] [rbp-2A8h]
  int v408; // [rsp+58h] [rbp-2A8h]
  __int64 v409; // [rsp+58h] [rbp-2A8h]
  __int64 v410; // [rsp+60h] [rbp-2A0h]
  char v411; // [rsp+60h] [rbp-2A0h]
  __int64 *v412; // [rsp+60h] [rbp-2A0h]
  __int64 v413; // [rsp+60h] [rbp-2A0h]
  char v414; // [rsp+68h] [rbp-298h]
  _QWORD *v415; // [rsp+68h] [rbp-298h]
  char v416; // [rsp+68h] [rbp-298h]
  int v417; // [rsp+68h] [rbp-298h]
  _QWORD *v418; // [rsp+68h] [rbp-298h]
  __int64 v419; // [rsp+68h] [rbp-298h]
  __int64 v420; // [rsp+68h] [rbp-298h]
  __int64 v421; // [rsp+70h] [rbp-290h]
  __int64 v422; // [rsp+70h] [rbp-290h]
  _QWORD *v423; // [rsp+70h] [rbp-290h]
  __int64 v424; // [rsp+70h] [rbp-290h]
  char v425; // [rsp+78h] [rbp-288h]
  int v426; // [rsp+78h] [rbp-288h]
  int v427; // [rsp+78h] [rbp-288h]
  __int64 *v428; // [rsp+78h] [rbp-288h]
  int v429; // [rsp+78h] [rbp-288h]
  __int64 v430; // [rsp+80h] [rbp-280h] BYREF
  __int64 v431; // [rsp+88h] [rbp-278h]
  _QWORD v432[2]; // [rsp+90h] [rbp-270h] BYREF
  __int64 *v433; // [rsp+A0h] [rbp-260h] BYREF
  __int64 v434; // [rsp+A8h] [rbp-258h] BYREF
  _QWORD *v435; // [rsp+B0h] [rbp-250h] BYREF
  __int64 v436; // [rsp+B8h] [rbp-248h]
  _QWORD *v437; // [rsp+C0h] [rbp-240h]
  __int64 v438; // [rsp+C8h] [rbp-238h]
  unsigned int v439; // [rsp+D0h] [rbp-230h]
  __int64 *v440; // [rsp+E0h] [rbp-220h]
  char v441; // [rsp+E8h] [rbp-218h]
  int v442; // [rsp+ECh] [rbp-214h]
  _BYTE *v443; // [rsp+F0h] [rbp-210h] BYREF
  __int64 v444; // [rsp+F8h] [rbp-208h]
  _BYTE v445[64]; // [rsp+100h] [rbp-200h] BYREF
  __int64 **v446; // [rsp+140h] [rbp-1C0h] BYREF
  __int64 v447; // [rsp+148h] [rbp-1B8h] BYREF
  __int64 *v448; // [rsp+150h] [rbp-1B0h] BYREF
  __int64 v449; // [rsp+158h] [rbp-1A8h]
  __int64 v450; // [rsp+160h] [rbp-1A0h]
  int v451; // [rsp+168h] [rbp-198h]
  __int64 v452; // [rsp+170h] [rbp-190h]
  __int64 v453; // [rsp+178h] [rbp-188h]

  v10 = a1;
  v11 = sub_1632FA0(a2[5]);
  ++*(_QWORD *)(a1 + 320);
  v12 = *(void **)(a1 + 336);
  *(_QWORD *)(v10 + 904) = v11;
  if ( v12 == *(void **)(v10 + 328) )
    goto LABEL_6;
  v13 = 4 * (*(_DWORD *)(v10 + 348) - *(_DWORD *)(v10 + 352));
  v14 = *(unsigned int *)(v10 + 344);
  if ( v13 < 0x20 )
    v13 = 32;
  if ( (unsigned int)v14 <= v13 )
  {
    memset(v12, -1, 8 * v14);
LABEL_6:
    *(_QWORD *)(v10 + 348) = 0;
    goto LABEL_7;
  }
  sub_16CC920(v10 + 320);
LABEL_7:
  v15 = *(_DWORD *)(v10 + 504);
  ++*(_QWORD *)(v10 + 488);
  if ( !v15 )
  {
    if ( !*(_DWORD *)(v10 + 508) )
      goto LABEL_13;
    v16 = *(_DWORD *)(v10 + 512);
    if ( v16 <= 0x40 )
      goto LABEL_10;
    j___libc_free_0(*(_QWORD *)(v10 + 496));
    *(_DWORD *)(v10 + 512) = 0;
LABEL_586:
    *(_QWORD *)(v10 + 496) = 0;
LABEL_12:
    *(_QWORD *)(v10 + 504) = 0;
    goto LABEL_13;
  }
  v116 = 4 * v15;
  v16 = *(_DWORD *)(v10 + 512);
  if ( (unsigned int)(4 * v15) < 0x40 )
    v116 = 64;
  if ( v16 <= v116 )
  {
LABEL_10:
    v17 = *(_QWORD **)(v10 + 496);
    for ( i = &v17[2 * v16]; i != v17; v17 += 2 )
      *v17 = -8;
    goto LABEL_12;
  }
  v117 = v15 - 1;
  if ( v117 )
  {
    _BitScanReverse(&v117, v117);
    v118 = 1 << (33 - (v117 ^ 0x1F));
    if ( v118 < 64 )
      v118 = 64;
    if ( v16 == v118 )
      goto LABEL_145;
    v119 = 4 * v118 / 3u + 1;
  }
  else
  {
    v119 = 86;
  }
  j___libc_free_0(*(_QWORD *)(v10 + 496));
  v120 = sub_1454B60(v119);
  *(_DWORD *)(v10 + 512) = v120;
  if ( !v120 )
    goto LABEL_586;
  *(_QWORD *)(v10 + 496) = sub_22077B0(16LL * v120);
LABEL_145:
  sub_1D5A4D0(v10 + 488);
LABEL_13:
  v19 = *(_QWORD *)(v10 + 8);
  *(_BYTE *)(v10 + 896) = 0;
  v20 = sub_160F9A0(v19, (__int64)&unk_4FCBA30, 1u);
  if ( v20 )
  {
    v21 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v20 + 104LL))(v20, &unk_4FCBA30);
    if ( v21 )
    {
      v22 = *(_QWORD *)(v21 + 208);
      *(_QWORD *)(v10 + 160) = v22;
      v23 = *(__int64 (**)())(*(_QWORD *)v22 + 16LL);
      if ( v23 == sub_16FF750 )
      {
        *(_QWORD *)(v10 + 168) = 0;
        BUG();
      }
      v24 = ((__int64 (__fastcall *)(__int64, __int64 *))v23)(v22, a2);
      *(_QWORD *)(v10 + 168) = v24;
      v25 = v24;
      v26 = *(__int64 (**)())(*(_QWORD *)v24 + 56LL);
      v27 = 0;
      if ( v26 != sub_1D12D20 )
      {
        v27 = ((__int64 (__fastcall *)(__int64))v26)(v25);
        v25 = *(_QWORD *)(v10 + 168);
      }
      *(_QWORD *)(v10 + 176) = v27;
      v28 = *(__int64 (**)(void))(*(_QWORD *)v25 + 112LL);
      v29 = 0;
      if ( v28 != sub_1D00B10 )
        v29 = v28();
      *(_QWORD *)(v10 + 184) = v29;
    }
  }
  v30 = *(__int64 **)(v10 + 8);
  v31 = *v30;
  v32 = v30[1];
  if ( v31 == v32 )
LABEL_649:
    BUG();
  while ( *(_UNKNOWN **)v31 != &unk_4F9B6E8 )
  {
    v31 += 16;
    if ( v32 == v31 )
      goto LABEL_649;
  }
  v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(*(_QWORD *)(v31 + 8), &unk_4F9B6E8);
  v34 = *(__int64 **)(v10 + 8);
  *(_QWORD *)(v10 + 200) = v33 + 360;
  v35 = *v34;
  v36 = v34[1];
  if ( v35 == v36 )
LABEL_655:
    BUG();
  while ( *(_UNKNOWN **)v35 != &unk_4F9D3C0 )
  {
    v35 += 16;
    if ( v36 == v35 )
      goto LABEL_655;
  }
  v37 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v35 + 8) + 104LL))(*(_QWORD *)(v35 + 8), &unk_4F9D3C0);
  v38 = sub_14A4050(v37, (__int64)a2);
  v39 = *(__int64 **)(v10 + 8);
  *(_QWORD *)(v10 + 192) = v38;
  v40 = *v39;
  v41 = v39[1];
  if ( v40 == v41 )
LABEL_654:
    BUG();
  while ( *(_UNKNOWN **)v40 != &unk_4F9920C )
  {
    v40 += 16;
    if ( v41 == v40 )
      goto LABEL_654;
  }
  *(_QWORD *)(v10 + 208) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v40 + 8) + 104LL))(
                             *(_QWORD *)(v40 + 8),
                             &unk_4F9920C)
                         + 160;
  v42 = (_QWORD *)sub_22077B0(408);
  v43 = (__int64)v42;
  if ( v42 )
  {
    *v42 = 0;
    v44 = v42 + 14;
    *(v44 - 13) = 0;
    v45 = *(_QWORD *)(v10 + 208);
    *(v44 - 12) = 0;
    *((_DWORD *)v44 - 22) = 0;
    *(v44 - 10) = 0;
    *(v44 - 9) = 0;
    *(v44 - 8) = 0;
    *((_DWORD *)v44 - 14) = 0;
    *(_QWORD *)(v43 + 80) = v44;
    *(_QWORD *)(v43 + 88) = v44;
    *(_QWORD *)(v43 + 72) = 0;
    *(_QWORD *)(v43 + 96) = 16;
    *(_DWORD *)(v43 + 104) = 0;
    *(_QWORD *)(v43 + 240) = 0;
    *(_QWORD *)(v43 + 248) = v43 + 280;
    *(_QWORD *)(v43 + 256) = v43 + 280;
    *(_QWORD *)(v43 + 264) = 16;
    *(_DWORD *)(v43 + 272) = 0;
    sub_137CAE0(v43, a2, v45, 0);
  }
  v46 = *(_QWORD *)(v10 + 224);
  *(_QWORD *)(v10 + 224) = v43;
  if ( v46 )
  {
    v47 = *(_QWORD *)(v46 + 256);
    if ( v47 != *(_QWORD *)(v46 + 248) )
      _libc_free(v47);
    v48 = *(_QWORD *)(v46 + 88);
    if ( v48 != *(_QWORD *)(v46 + 80) )
      _libc_free(v48);
    j___libc_free_0(*(_QWORD *)(v46 + 40));
    if ( *(_DWORD *)(v46 + 24) )
    {
      v434 = 2;
      v435 = 0;
      v436 = -8;
      v433 = (__int64 *)&unk_49E8A80;
      v437 = 0;
      v447 = 2;
      v448 = 0;
      v449 = -16;
      v446 = (__int64 **)&unk_49E8A80;
      v450 = 0;
      v49 = *(_QWORD **)(v46 + 8);
      v50 = &v49[5 * *(unsigned int *)(v46 + 24)];
      while ( v50 != v49 )
      {
        *v49 = &unk_49EE2B0;
        v51 = (__int64)(v49 + 1);
        v49 += 5;
        sub_1455FA0(v51);
      }
      v446 = (__int64 **)&unk_49EE2B0;
      sub_1455FA0((__int64)&v447);
      v433 = (__int64 *)&unk_49EE2B0;
      sub_1455FA0((__int64)&v434);
    }
    j___libc_free_0(*(_QWORD *)(v46 + 8));
    j_j___libc_free_0(v46, 408);
    v43 = *(_QWORD *)(v10 + 224);
  }
  v52 = (__int64 *)sub_22077B0(8);
  v53 = v52;
  if ( v52 )
    sub_13702A0(v52, a2, v43, *(_QWORD *)(v10 + 208));
  v54 = *(__int64 **)(v10 + 216);
  *(_QWORD *)(v10 + 216) = v53;
  if ( v54 )
  {
    sub_1368A00(v54);
    j_j___libc_free_0(v54, 8);
  }
  v55 = sub_1560180((__int64)(a2 + 14), 34);
  if ( !v55 )
    v55 = sub_1560180((__int64)(a2 + 14), 17);
  v56 = *(__int64 **)(v10 + 8);
  *(_BYTE *)(v10 + 897) = v55;
  v57 = *v56;
  v58 = v56[1];
  if ( v57 == v58 )
LABEL_653:
    BUG();
  while ( *(_UNKNOWN **)v57 != &unk_4F99CCD )
  {
    v57 += 16;
    if ( v58 == v57 )
      goto LABEL_653;
  }
  v59 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v57 + 8) + 104LL))(*(_QWORD *)(v57 + 8), &unk_4F99CCD);
  v66 = *(_QWORD *)(v59 + 160);
  if ( byte_4FC2A40 )
  {
    if ( (unsigned __int8)sub_1441EC0(*(_QWORD *)(v59 + 160), (__int64)a2, *(__int64 **)(v10 + 216)) )
    {
      sub_15E45F0((__int64)a2, (__int64)".hot", 4);
    }
    else if ( (unsigned __int8)sub_14420B0(v66, (__int64)a2, *(__int64 **)(v10 + 216)) )
    {
      sub_15E45F0((__int64)a2, (__int64)".unlikely", 9);
    }
  }
  if ( !*(_BYTE *)(v10 + 897)
    && (v125 = sub_1441C90(v66)) == 0
    && (v126 = *(_QWORD *)(v10 + 176)) != 0
    && *(_DWORD *)(v126 + 40)
    && (v127 = v126 + 24, (v128 = a2[10]) != 0) )
  {
    v129 = v128 - 24;
    while ( 1 )
    {
      v130 = *(_QWORD *)(v129 + 56);
      v131 = *(_QWORD *)(v129 + 32);
      v132 = v130 + 72;
      if ( v131 == v130 + 72 || !v131 )
        break;
      v133 = v131 - 24;
      v134 = sub_394CF20(v129, v127, v132, v130, v62, v63);
      v129 = v133;
      v125 |= v134;
    }
    v67 = sub_394CF20(v129, v127, v132, v130, v62, v63) | v125;
  }
  else
  {
    v67 = 0;
  }
  v402 = v67
       | sub_1D64280(
           v10,
           (__int64)a2,
           a3,
           a4,
           a5,
           *(double *)a6.m128_u64,
           v64,
           v65,
           *(double *)a9.m128i_i64,
           a10,
           v60,
           v61,
           v62,
           v63);
  v401 = a2 + 9;
  if ( (__int64 *)a2[10] != a2 + 9 )
  {
    v425 = 0;
    v70 = a2[10];
    v421 = v10;
    do
    {
      if ( !v70 )
        BUG();
      v71 = *(_QWORD *)(v70 + 24);
      v72 = 0;
      while ( v70 + 16 != v71 )
      {
        v73 = v71;
        v71 = *(_QWORD *)(v71 + 8);
        v74 = (_QWORD *)(v73 - 24);
        if ( *(_BYTE *)(v73 - 8) != 78
          || (v174 = *(_QWORD *)(v73 - 48), *(_BYTE *)(v174 + 16))
          || (*(_BYTE *)(v174 + 33) & 0x20) == 0
          || *(_DWORD *)(v174 + 36) != 38
          || sub_1601A30((__int64)v74, 0) && *(_BYTE *)(sub_1601A30((__int64)v74, 0) + 16) == 53 )
        {
          v72 = v74;
        }
        else
        {
          v175 = sub_1601A30((__int64)v74, 0);
          v176 = v175;
          if ( v175 )
          {
            v177 = *(_BYTE *)(v175 + 16);
            if ( v177 > 0x17u && (_QWORD *)v176 != v72 && (unsigned int)v177 - 25 > 9 )
            {
              if ( v177 != 77
                || (v178 = (unsigned int)*(unsigned __int8 *)(sub_157EBA0(*(_QWORD *)(v176 + 40)) + 16) - 34,
                    (unsigned int)v178 > 0x36)
                || (v179 = 0x40018000000001LL, !_bittest64(&v179, v178)) )
              {
                sub_15F2070(v74);
                if ( *(_BYTE *)(v176 + 16) == 77 )
                {
                  v197 = sub_157EE30(*(_QWORD *)(v176 + 40));
                  if ( v197 )
                    v197 -= 24;
                  sub_15F2120((__int64)v74, v197);
                  v425 = 1;
                }
                else
                {
                  sub_15F2180((__int64)v74, v176);
                  v425 = 1;
                }
              }
            }
          }
        }
      }
      v70 = *(_QWORD *)(v70 + 8);
    }
    while ( v401 != (__int64 *)v70 );
    v402 |= v425;
    v10 = v421;
  }
  if ( !byte_4FC3300 )
  {
    v317 = *(_QWORD *)(v10 + 160);
    if ( v317 )
    {
      if ( (*(_BYTE *)(v317 + 800) & 2) != 0 )
      {
        v318 = *(_QWORD *)(v10 + 176);
        if ( v318 )
        {
          v319 = *(_BYTE *)(v318 + 56);
          if ( !v319 )
          {
            v409 = a2[10];
            if ( v401 != (__int64 *)v409 )
            {
              while ( 1 )
              {
                v320 = v409 - 24;
                if ( !v409 )
                  v320 = 0;
                v424 = v320;
                v321 = sub_157EBA0(v320);
                if ( *(_BYTE *)(v321 + 16) != 26 )
                  goto LABEL_493;
                if ( (*(_DWORD *)(v321 + 20) & 0xFFFFFFF) != 3 )
                  goto LABEL_493;
                v322 = *(_QWORD *)(v321 - 72);
                v323 = *(_QWORD *)(v322 + 8);
                if ( !v323 )
                  goto LABEL_493;
                if ( *(_QWORD *)(v323 + 8) )
                  goto LABEL_493;
                v324 = *(_BYTE *)(v322 + 16);
                if ( (unsigned __int8)(v324 - 35) > 0x11u )
                  goto LABEL_493;
                v419 = *(_QWORD *)(v321 - 24);
                v413 = *(_QWORD *)(v321 - 48);
                v325 = sub_157EBA0(v424);
                v326 = v325;
                if ( !*(_QWORD *)(v325 + 48) && *(__int16 *)(v325 + 18) >= 0 )
                  break;
                if ( sub_1625790(v325, 15) )
                  goto LABEL_493;
                v324 = *(_BYTE *)(v322 + 16);
                if ( v324 == 50 )
                  goto LABEL_631;
                if ( v324 != 5 )
                  goto LABEL_626;
                v327 = *(_WORD *)(v322 + 18);
                if ( v327 != 26 )
                {
                  if ( v327 != 27 )
                    goto LABEL_493;
                  v328 = *(_QWORD *)(v322 - 24LL * (*(_DWORD *)(v322 + 20) & 0xFFFFFFF));
                  v329 = *(_QWORD *)(v328 + 8);
                  if ( !v329 )
                    goto LABEL_493;
                  if ( *(_QWORD *)(v329 + 8) )
                    goto LABEL_493;
                  v330 = *(_QWORD *)(v322 + 24 * (1LL - (*(_DWORD *)(v322 + 20) & 0xFFFFFFF)));
                  v331 = *(_QWORD *)(v330 + 8);
                  if ( !v331 )
                    goto LABEL_493;
                  goto LABEL_510;
                }
                v328 = *(_QWORD *)(v322 - 24LL * (*(_DWORD *)(v322 + 20) & 0xFFFFFFF));
                v387 = *(_QWORD *)(v328 + 8);
                if ( !v387 )
                  goto LABEL_493;
                if ( *(_QWORD *)(v387 + 8) )
                  goto LABEL_493;
                v330 = *(_QWORD *)(v322 + 24 * (1LL - (*(_DWORD *)(v322 + 20) & 0xFFFFFFF)));
                v388 = *(_QWORD *)(v330 + 8);
                if ( !v388 )
                  goto LABEL_493;
LABEL_622:
                if ( !*(_QWORD *)(v388 + 8) )
                {
                  v429 = 26;
LABEL_512:
                  v332 = *(unsigned __int8 *)(v328 + 16);
                  if ( (unsigned __int8)v332 > 0x17u
                    && ((unsigned __int8)(v332 - 75) <= 1u || (unsigned int)(v332 - 35) <= 0x11) )
                  {
                    v333 = *(unsigned __int8 *)(v330 + 16);
                    if ( (unsigned __int8)v333 > 0x17u
                      && ((unsigned __int8)(v333 - 75) <= 1u || (unsigned int)(v333 - 35) <= 0x11) )
                    {
                      v334 = *(_QWORD *)(v424 + 32);
                      if ( !v334 || v334 == *(_QWORD *)(v424 + 56) + 72LL )
                        v335 = 0;
                      else
                        v335 = v334 - 24;
                      v391 = v335;
                      v393 = v328;
                      v395 = *(_QWORD *)(v424 + 56);
                      v336 = (__int64 *)sub_1649960(v424);
                      LOWORD(v448) = 773;
                      v433 = v336;
                      v446 = &v433;
                      v434 = v337;
                      v447 = (__int64)".cond.split";
                      v397 = sub_157E9C0(v424);
                      v338 = (_QWORD *)sub_22077B0(64);
                      v339 = v393;
                      v340 = (__int64)v338;
                      if ( v338 )
                      {
                        sub_157FB60(v338, v397, (__int64)&v446, v395, v391);
                        v339 = v393;
                      }
                      sub_1593B40((_QWORD *)(v326 - 72), v339);
                      sub_15F20C0((_QWORD *)v322);
                      if ( v429 == 26 )
                        sub_1593B40((_QWORD *)(v326 - 24), v340);
                      else
                        sub_1593B40((_QWORD *)(v326 - 48), v340);
                      v341 = sub_157E9C0(v340);
                      LOWORD(v435) = 257;
                      v449 = v341;
                      v446 = 0;
                      v450 = 0;
                      v451 = 0;
                      v452 = 0;
                      v453 = 0;
                      v447 = v340;
                      v448 = (__int64 *)(v340 + 40);
                      v342 = sub_1648A60(56, 3u);
                      v343 = (__int64)v342;
                      if ( v342 )
                        sub_15F83E0((__int64)v342, v419, v413, v330, 0);
                      if ( v447 )
                      {
                        v398 = v448;
                        sub_157E9D0(v447 + 40, v343);
                        v344 = *v398;
                        v345 = *(_QWORD *)(v343 + 24) & 7LL;
                        *(_QWORD *)(v343 + 32) = v398;
                        v344 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(v343 + 24) = v344 | v345;
                        *(_QWORD *)(v344 + 8) = v343 + 24;
                        *v398 = *v398 & 7 | (v343 + 24);
                      }
                      sub_164B780(v343, (__int64 *)&v433);
                      sub_12A86E0((__int64 *)&v446, v343);
                      sub_17CD270((__int64 *)&v446);
                      if ( *(_BYTE *)(v330 + 16) > 0x17u )
                      {
                        sub_15F2070((_QWORD *)v330);
                        sub_15F2120(v330, v343);
                      }
                      if ( v429 != 27 )
                      {
                        v346 = v419;
                        v419 = v413;
                        v413 = v346;
                      }
                      v347 = sub_157F280(v413);
                      v349 = v348;
                      while ( v349 != v347 )
                      {
                        while ( 1 )
                        {
                          v350 = *(_DWORD *)(v347 + 20) & 0xFFFFFFF;
                          if ( !v350 )
                            break;
                          v351 = 0;
                          v352 = 24LL * *(unsigned int *)(v347 + 56) + 8;
                          while ( 1 )
                          {
                            v353 = v347 - 24LL * v350;
                            if ( (*(_BYTE *)(v347 + 23) & 0x40) != 0 )
                              v353 = *(_QWORD *)(v347 - 8);
                            if ( v424 == *(_QWORD *)(v353 + v352) )
                              break;
                            ++v351;
                            v352 += 8;
                            if ( v350 == v351 )
                              goto LABEL_588;
                          }
                          if ( v351 < 0 )
                            break;
                          *(_QWORD *)(v353 + 8LL * v351 + 24LL * *(unsigned int *)(v347 + 56) + 8) = v340;
                        }
LABEL_588:
                        v371 = *(_QWORD *)(v347 + 32);
                        if ( !v371 )
                          BUG();
                        v347 = 0;
                        if ( *(_BYTE *)(v371 - 8) == 77 )
                          v347 = v371 - 24;
                      }
                      v375 = sub_157F280(v419);
                      if ( v375 != v372 )
                      {
                        v376 = v340;
                        v377 = v372;
                        do
                        {
                          v378 = 0x17FFFFFFE8LL;
                          v379 = *(_BYTE *)(v375 + 23) & 0x40;
                          v380 = *(_DWORD *)(v375 + 20) & 0xFFFFFFF;
                          if ( v380 )
                          {
                            v374 = v375 - 24LL * v380;
                            v381 = 24LL * *(unsigned int *)(v375 + 56) + 8;
                            v382 = 0;
                            do
                            {
                              v373 = v375 - 24LL * v380;
                              if ( v379 )
                                v373 = *(_QWORD *)(v375 - 8);
                              if ( v424 == *(_QWORD *)(v373 + v381) )
                              {
                                v378 = 24 * v382;
                                goto LABEL_601;
                              }
                              ++v382;
                              v381 += 8;
                            }
                            while ( v380 != (_DWORD)v382 );
                            v378 = 0x17FFFFFFE8LL;
                          }
LABEL_601:
                          if ( v379 )
                            v383 = *(_QWORD *)(v375 - 8);
                          else
                            v383 = v375 - 24LL * v380;
                          v420 = v376;
                          sub_1704F80(v375, *(_QWORD *)(v383 + v378), v376, v376, v373, v374);
                          v384 = *(_QWORD *)(v375 + 32);
                          if ( !v384 )
                            BUG();
                          v375 = 0;
                          v376 = v420;
                          if ( *(_BYTE *)(v384 - 8) == 77 )
                            v375 = v384 - 24;
                        }
                        while ( v377 != v375 );
                      }
                      if ( (unsigned __int8)sub_1625AE0(v326, &v430, &v433) )
                      {
                        v446 = (__int64 **)sub_16498A0(v326);
                        v385 = sub_161BE60(&v446, v430, (unsigned int)v433);
                        sub_1625C10(v326, 2, v385);
                        v446 = (__int64 **)sub_16498A0(v343);
                        v386 = sub_161BE60(&v446, v430, (unsigned int)v433);
                        sub_1625C10(v343, 2, v386);
                      }
                      v319 = 1;
                      *(_BYTE *)(v10 + 896) = 1;
                    }
                  }
                }
LABEL_493:
                v409 = *(_QWORD *)(v409 + 8);
                if ( v401 == (__int64 *)v409 )
                {
                  v402 |= v319;
                  goto LABEL_68;
                }
              }
              if ( v324 != 50 )
              {
LABEL_626:
                if ( v324 != 51 )
                  goto LABEL_493;
                v328 = *(_QWORD *)(v322 - 48);
                v389 = *(_QWORD *)(v328 + 8);
                if ( !v389 )
                  goto LABEL_493;
                if ( *(_QWORD *)(v389 + 8) )
                  goto LABEL_493;
                v330 = *(_QWORD *)(v322 - 24);
                v331 = *(_QWORD *)(v330 + 8);
                if ( !v331 )
                  goto LABEL_493;
LABEL_510:
                if ( !*(_QWORD *)(v331 + 8) )
                {
                  v429 = 27;
                  goto LABEL_512;
                }
                goto LABEL_493;
              }
LABEL_631:
              v328 = *(_QWORD *)(v322 - 48);
              v390 = *(_QWORD *)(v328 + 8);
              if ( !v390 )
                goto LABEL_493;
              if ( *(_QWORD *)(v390 + 8) )
                goto LABEL_493;
              v330 = *(_QWORD *)(v322 - 24);
              v388 = *(_QWORD *)(v330 + 8);
              if ( !v388 )
                goto LABEL_493;
              goto LABEL_622;
            }
          }
        }
      }
    }
  }
LABEL_68:
  v396 = v402
       | sub_1AAD1B0(
           (__int64)a2,
           0,
           0,
           a3,
           *(double *)a4.m128i_i64,
           *(double *)a5.m128i_i64,
           *(double *)a6.m128_u64,
           v68,
           v69,
           *(double *)a9.m128i_i64,
           a10);
  v394 = v10 + 520;
  v399 = v10 + 240;
  while ( 1 )
  {
    v78 = *(_DWORD *)(v10 + 704);
    ++*(_QWORD *)(v10 + 688);
    if ( !v78 )
    {
      if ( !*(_DWORD *)(v10 + 708) )
        goto LABEL_75;
      v79 = *(unsigned int *)(v10 + 712);
      if ( (unsigned int)v79 > 0x40 )
      {
        j___libc_free_0(*(_QWORD *)(v10 + 696));
        *(_QWORD *)(v10 + 696) = 0;
        *(_QWORD *)(v10 + 704) = 0;
        *(_DWORD *)(v10 + 712) = 0;
        goto LABEL_75;
      }
LABEL_72:
      v80 = *(_QWORD **)(v10 + 696);
      for ( j = &v80[2 * v79]; j != v80; v80 += 2 )
        *v80 = -8;
      *(_QWORD *)(v10 + 704) = 0;
      goto LABEL_75;
    }
    v75 = (unsigned int)(4 * v78);
    v79 = *(unsigned int *)(v10 + 712);
    if ( (unsigned int)v75 < 0x40 )
      v75 = 64;
    if ( (unsigned int)v75 >= (unsigned int)v79 )
      goto LABEL_72;
    v187 = *(_QWORD **)(v10 + 696);
    v188 = v78 - 1;
    if ( !v188 )
    {
      v193 = 2048;
      v192 = 128;
LABEL_280:
      j___libc_free_0(v187);
      *(_DWORD *)(v10 + 712) = v192;
      v194 = (_QWORD *)sub_22077B0(v193);
      v195 = *(unsigned int *)(v10 + 712);
      *(_QWORD *)(v10 + 704) = 0;
      *(_QWORD *)(v10 + 696) = v194;
      for ( k = &v194[2 * v195]; k != v194; v194 += 2 )
      {
        if ( v194 )
          *v194 = -8;
      }
      goto LABEL_75;
    }
    _BitScanReverse(&v188, v188);
    v75 = 33 - (v188 ^ 0x1F);
    v189 = 1 << (33 - (v188 ^ 0x1F));
    if ( v189 < 64 )
      v189 = 64;
    if ( (_DWORD)v79 != v189 )
    {
      v190 = (4 * v189 / 3u + 1) | ((unsigned __int64)(4 * v189 / 3u + 1) >> 1);
      v191 = ((v190 | (v190 >> 2)) >> 4)
           | v190
           | (v190 >> 2)
           | ((((v190 | (v190 >> 2)) >> 4) | v190 | (v190 >> 2)) >> 8);
      v192 = (v191 | (v191 >> 16)) + 1;
      v193 = 16 * ((v191 | (v191 >> 16)) + 1);
      goto LABEL_280;
    }
    *(_QWORD *)(v10 + 704) = 0;
    v370 = &v187[2 * (unsigned int)v79];
    do
    {
      if ( v187 )
        *v187 = -8;
      v187 += 2;
    }
    while ( v370 != v187 );
LABEL_75:
    v82 = *(_DWORD *)(v10 + 880);
    ++*(_QWORD *)(v10 + 864);
    if ( v82 || *(_DWORD *)(v10 + 884) )
    {
      v83 = *(_QWORD **)(v10 + 872);
      v75 = 64;
      v84 = &v83[19 * *(unsigned int *)(v10 + 888)];
      v85 = 4 * v82;
      if ( (unsigned int)(4 * v82) < 0x40 )
        v85 = 64;
      if ( *(_DWORD *)(v10 + 888) <= v85 )
      {
        while ( v83 != v84 )
        {
          if ( *v83 != -8 )
          {
            if ( *v83 != -16 )
            {
              v86 = v83[1];
              if ( (_QWORD *)v86 != v83 + 3 )
                _libc_free(v86);
            }
            *v83 = -8;
          }
          v83 += 19;
        }
LABEL_88:
        *(_QWORD *)(v10 + 880) = 0;
        goto LABEL_89;
      }
      do
      {
        if ( *v83 != -8 && *v83 != -16 )
        {
          v123 = v83[1];
          if ( (_QWORD *)v123 != v83 + 3 )
            _libc_free(v123);
        }
        v83 += 19;
      }
      while ( v84 != v83 );
      v124 = *(_DWORD *)(v10 + 888);
      if ( v82 )
      {
        v206 = 64;
        if ( v82 != 1 )
        {
          _BitScanReverse(&v207, v82 - 1);
          v75 = 33 - (v207 ^ 0x1F);
          v206 = (unsigned int)(1 << (33 - (v207 ^ 0x1F)));
          if ( (int)v206 < 64 )
            v206 = 64;
        }
        v208 = *(_QWORD **)(v10 + 872);
        if ( (_DWORD)v206 == v124 )
        {
          *(_QWORD *)(v10 + 880) = 0;
          v365 = &v208[19 * v206];
          do
          {
            if ( v208 )
              *v208 = -8;
            v208 += 19;
          }
          while ( v365 != v208 );
        }
        else
        {
          j___libc_free_0(v208);
          v209 = (((((((4 * (int)v206 / 3u + 1) | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v206 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v206 / 3u + 1) | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v206 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v206 / 3u + 1) | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v206 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v206 / 3u + 1) | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v206 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v206 / 3u + 1) >> 1);
          v210 = ((v209 >> 16) | v209) + 1;
          *(_DWORD *)(v10 + 888) = v210;
          v211 = (_QWORD *)sub_22077B0(152 * v210);
          v212 = *(unsigned int *)(v10 + 888);
          *(_QWORD *)(v10 + 880) = 0;
          *(_QWORD *)(v10 + 872) = v211;
          v75 = 9 * v212;
          for ( m = &v211[19 * v212]; m != v211; v211 += 19 )
          {
            if ( v211 )
              *v211 = -8;
          }
        }
      }
      else
      {
        if ( !v124 )
          goto LABEL_88;
        j___libc_free_0(*(_QWORD *)(v10 + 872));
        *(_QWORD *)(v10 + 872) = 0;
        *(_QWORD *)(v10 + 880) = 0;
        *(_DWORD *)(v10 + 888) = 0;
      }
    }
LABEL_89:
    ++*(_QWORD *)(v10 + 520);
    v87 = *(void **)(v10 + 536);
    if ( v87 != *(void **)(v10 + 528) )
    {
      v88 = 4 * (*(_DWORD *)(v10 + 548) - *(_DWORD *)(v10 + 552));
      v89 = *(unsigned int *)(v10 + 544);
      if ( v88 < 0x20 )
        v88 = 32;
      if ( (unsigned int)v89 > v88 )
      {
        sub_16CC920(v394);
        goto LABEL_95;
      }
      memset(v87, -1, 8 * v89);
    }
    *(_QWORD *)(v10 + 548) = 0;
LABEL_95:
    v90 = *(_DWORD *)(v10 + 736);
    ++*(_QWORD *)(v10 + 720);
    if ( v90 || *(_DWORD *)(v10 + 740) )
    {
      v91 = *(_QWORD **)(v10 + 728);
      v75 = 64;
      v92 = 4 * v90;
      v93 = *(unsigned int *)(v10 + 744);
      v94 = &v91[67 * v93];
      if ( (unsigned int)(4 * v90) < 0x40 )
        v92 = 64;
      if ( (unsigned int)v93 <= v92 )
      {
        for ( ; v91 != v94; v91 += 67 )
        {
          if ( *v91 != -8 )
          {
            if ( *v91 != -16 )
            {
              v95 = v91[1];
              if ( (_QWORD *)v95 != v91 + 3 )
                _libc_free(v95);
            }
            *v91 = -8;
          }
        }
LABEL_107:
        *(_QWORD *)(v10 + 736) = 0;
        goto LABEL_108;
      }
      do
      {
        if ( *v91 != -16 && *v91 != -8 )
        {
          v121 = v91[1];
          if ( (_QWORD *)v121 != v91 + 3 )
            _libc_free(v121);
        }
        v91 += 67;
      }
      while ( v91 != v94 );
      v122 = *(_DWORD *)(v10 + 744);
      if ( v90 )
      {
        v354 = 64;
        if ( v90 != 1 )
        {
          _BitScanReverse(&v355, v90 - 1);
          v75 = 33 - (v355 ^ 0x1F);
          v354 = 1 << (33 - (v355 ^ 0x1F));
          if ( v354 < 64 )
            v354 = 64;
        }
        v356 = *(_QWORD **)(v10 + 728);
        if ( v122 == v354 )
        {
          *(_QWORD *)(v10 + 736) = 0;
          v364 = &v356[67 * v122];
          do
          {
            if ( v356 )
              *v356 = -8;
            v356 += 67;
          }
          while ( v364 != v356 );
        }
        else
        {
          j___libc_free_0(v356);
          v357 = ((((((((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                    | (4 * v354 / 3u + 1)
                    | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                  | (4 * v354 / 3u + 1)
                  | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                  | (4 * v354 / 3u + 1)
                  | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 4)
                | (((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                | (4 * v354 / 3u + 1)
                | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 16;
          v358 = (v357
                | (((((((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                    | (4 * v354 / 3u + 1)
                    | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                  | (4 * v354 / 3u + 1)
                  | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                  | (4 * v354 / 3u + 1)
                  | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 4)
                | (((4 * v354 / 3u + 1) | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1)) >> 2)
                | (4 * v354 / 3u + 1)
                | ((unsigned __int64)(4 * v354 / 3u + 1) >> 1))
               + 1;
          *(_DWORD *)(v10 + 744) = v358;
          v359 = (_QWORD *)sub_22077B0(536 * v358);
          v360 = *(unsigned int *)(v10 + 744);
          *(_QWORD *)(v10 + 736) = 0;
          *(_QWORD *)(v10 + 728) = v359;
          for ( n = &v359[67 * v360]; n != v359; v359 += 67 )
          {
            if ( v359 )
              *v359 = -8;
          }
        }
      }
      else
      {
        if ( !v122 )
          goto LABEL_107;
        j___libc_free_0(*(_QWORD *)(v10 + 728));
        *(_QWORD *)(v10 + 728) = 0;
        *(_QWORD *)(v10 + 736) = 0;
        *(_DWORD *)(v10 + 744) = 0;
      }
    }
LABEL_108:
    v96 = *(_DWORD *)(v10 + 848);
    ++*(_QWORD *)(v10 + 832);
    if ( !v96 )
    {
      if ( !*(_DWORD *)(v10 + 852) )
        goto LABEL_114;
      v97 = *(unsigned int *)(v10 + 856);
      if ( (unsigned int)v97 > 0x40 )
      {
        j___libc_free_0(*(_QWORD *)(v10 + 840));
        *(_QWORD *)(v10 + 840) = 0;
        *(_QWORD *)(v10 + 848) = 0;
        *(_DWORD *)(v10 + 856) = 0;
        goto LABEL_114;
      }
LABEL_111:
      v98 = *(_QWORD **)(v10 + 840);
      for ( ii = &v98[2 * v97]; ii != v98; v98 += 2 )
        *v98 = -8;
      *(_QWORD *)(v10 + 848) = 0;
      goto LABEL_114;
    }
    v75 = (unsigned int)(4 * v96);
    v97 = *(unsigned int *)(v10 + 856);
    if ( (unsigned int)v75 < 0x40 )
      v75 = 64;
    if ( (unsigned int)v97 <= (unsigned int)v75 )
      goto LABEL_111;
    v180 = *(_QWORD **)(v10 + 840);
    v181 = v96 - 1;
    if ( !v181 )
    {
      v186 = 2048;
      v185 = 128;
LABEL_574:
      j___libc_free_0(v180);
      *(_DWORD *)(v10 + 856) = v185;
      v367 = (_QWORD *)sub_22077B0(v186);
      v368 = *(unsigned int *)(v10 + 856);
      *(_QWORD *)(v10 + 848) = 0;
      *(_QWORD *)(v10 + 840) = v367;
      for ( jj = &v367[2 * v368]; jj != v367; v367 += 2 )
      {
        if ( v367 )
          *v367 = -8;
      }
      goto LABEL_114;
    }
    _BitScanReverse(&v181, v181);
    v75 = 33 - (v181 ^ 0x1F);
    v182 = 1 << (33 - (v181 ^ 0x1F));
    if ( v182 < 64 )
      v182 = 64;
    if ( (_DWORD)v97 != v182 )
    {
      v183 = (4 * v182 / 3u + 1) | ((unsigned __int64)(4 * v182 / 3u + 1) >> 1);
      v184 = ((v183 | (v183 >> 2)) >> 4)
           | v183
           | (v183 >> 2)
           | ((((v183 | (v183 >> 2)) >> 4) | v183 | (v183 >> 2)) >> 8);
      v185 = (v184 | (v184 >> 16)) + 1;
      v186 = 16 * ((v184 | (v184 >> 16)) + 1);
      goto LABEL_574;
    }
    *(_QWORD *)(v10 + 848) = 0;
    v366 = &v180[2 * (unsigned int)v97];
    do
    {
      if ( v180 )
        *v180 = -8;
      v180 += 2;
    }
    while ( v366 != v180 );
LABEL_114:
    v406 = 0;
    v410 = a2[10];
    if ( (__int64 *)v410 != v401 )
    {
      while ( 2 )
      {
        v422 = v410;
        v100 = *(_QWORD *)(v410 + 8);
        v101 = (_QWORD *)(v410 - 24);
        LOBYTE(v433) = 0;
        v410 = v100;
        v403 = v101;
        sub_1D672E0(v399);
        if ( *(_BYTE *)(v10 + 304) )
        {
          v170 = *(_DWORD *)(v10 + 296);
          if ( v170 )
          {
            v171 = *(_QWORD **)(v10 + 280);
            v172 = &v171[2 * v170];
            do
            {
              if ( *v171 != -4 && *v171 != -8 )
              {
                v173 = v171[1];
                if ( v173 )
                  sub_161E7C0((__int64)(v171 + 1), v173);
              }
              v171 += 2;
            }
            while ( v172 != v171 );
          }
          j___libc_free_0(*(_QWORD *)(v10 + 280));
          *(_BYTE *)(v10 + 304) = 0;
        }
        v102 = 0;
        v103 = *(_QWORD **)(v422 + 24);
        *(_QWORD *)(v10 + 232) = v103;
        while ( (_QWORD *)(v422 + 16) != v103 )
        {
          *(_QWORD *)(v10 + 232) = v103[1];
          v102 |= sub_1D779D0(v10, (__int64)(v103 - 3), &v433, a3, (__m128)a4, (__m128)a5, a6, v76, v77, a9, a10);
          v104 = (unsigned __int8)v433;
          if ( (_BYTE)v433 )
            goto LABEL_271;
          v103 = *(_QWORD **)(v10 + 232);
        }
        v414 = v102;
        if ( *(_QWORD *)(v10 + 176) )
        {
LABEL_121:
          v105 = (_QWORD *)(*(_QWORD *)(v422 + 16) & 0xFFFFFFFFFFFFFFF8LL);
          if ( v103 == v105 )
            goto LABEL_172;
          while ( 1 )
          {
            if ( !v105 )
              BUG();
            v106 = *(v105 - 3);
            if ( *(_BYTE *)(v106 + 8) != 11 )
              goto LABEL_123;
            v107 = *(_QWORD *)(v10 + 176);
            v108 = sub_1D5D7E0(*(_QWORD *)(v10 + 904), (__int64 *)v106, 1u);
            v109 = 1;
            if ( v108 != 1 )
            {
              if ( !v108 )
                goto LABEL_123;
              v109 = v108;
              if ( !*(_QWORD *)(v107 + 8LL * v108 + 120) )
                goto LABEL_123;
            }
            if ( (*(_BYTE *)(v107 + 259LL * v109 + 2553) & 0xFB) != 0 )
              goto LABEL_123;
            v110 = (__int64)(v105 - 3);
            v446 = &v448;
            v447 = 0x400000000LL;
            v111 = sub_1AECB30((unsigned __int64)(v105 - 3), 0, 1u, (__int64)&v446);
            if ( v111 )
            {
              v114 = v111;
              sub_164D160(
                v110,
                (__int64)v446[(unsigned int)v447 - 1],
                a3,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(double *)a6.m128_u64,
                v112,
                v113,
                *(double *)a9.m128i_i64,
                a10);
              sub_1AEB370(v110, 0);
              if ( v446 != &v448 )
                _libc_free((unsigned __int64)v446);
              v115 = *(_QWORD *)(v10 + 176) == 0;
              LOBYTE(v433) = 1;
              if ( v115 )
              {
                v414 = v114;
                break;
              }
              v414 = v114;
              goto LABEL_121;
            }
            if ( v446 != &v448 )
            {
              _libc_free((unsigned __int64)v446);
              v105 = (_QWORD *)(*v105 & 0xFFFFFFFFFFFFFFF8LL);
              if ( v103 == v105 )
                break;
            }
            else
            {
LABEL_123:
              v105 = (_QWORD *)(*v105 & 0xFFFFFFFFFFFFFFF8LL);
              if ( v103 == v105 )
                break;
            }
          }
        }
LABEL_172:
        v406 |= (unsigned __int8)sub_1D639B0(v10, v403) | v414;
        if ( (_BYTE)v433 )
        {
          v104 = v406;
LABEL_271:
          v406 = v104;
          break;
        }
        v75 = (__int64)(a2 + 9);
        if ( (__int64 *)v410 != v401 )
          continue;
        break;
      }
    }
    if ( byte_4FC27A0 && *(_DWORD *)(v10 + 880) )
    {
      v436 = 0;
      v433 = (__int64 *)&v435;
      v434 = 0x100000000LL;
      v437 = 0;
      v438 = 0;
      v439 = 0;
      v441 = 0;
      v442 = 0;
      v440 = a2;
      sub_15D3930((__int64)&v433);
      if ( *(_DWORD *)(v10 + 880) )
      {
        v284 = *(_QWORD **)(v10 + 872);
        v418 = &v284[19 * *(unsigned int *)(v10 + 888)];
        if ( v284 != v418 )
        {
          while ( 1 )
          {
            v405 = *v284 == -16 || *v284 == -8;
            if ( !v405 )
              break;
            v284 += 19;
            if ( v418 == v284 )
              goto LABEL_291;
          }
          if ( v418 != v284 )
          {
            v285 = v10;
            v286 = v284;
            while ( 1 )
            {
              v446 = &v448;
              v447 = 0x1000000000LL;
              v287 = (__int64 *)v286[1];
              v428 = &v287[*((unsigned int *)v286 + 4)];
              if ( v287 != v428 )
                break;
LABEL_471:
              v286 += 19;
              if ( v286 != v418 )
              {
                while ( *v286 == -8 || *v286 == -16 )
                {
                  v286 += 19;
                  if ( v418 == v286 )
                    goto LABEL_475;
                }
                if ( v418 != v286 )
                  continue;
              }
LABEL_475:
              v10 = v285;
              v406 |= v405;
              goto LABEL_291;
            }
            v423 = v286;
            v288 = (__int64 *)v286[1];
            v289 = v285;
            while ( 1 )
            {
              v293 = *(_QWORD **)(v289 + 536);
              v291 = *(_QWORD **)(v289 + 528);
              v294 = *v288;
              if ( v293 == v291 )
              {
                v290 = &v291[*(unsigned int *)(v289 + 548)];
                if ( v291 == v290 )
                {
                  v313 = *(_QWORD **)(v289 + 528);
                }
                else
                {
                  do
                  {
                    if ( v294 == *v291 )
                      break;
                    ++v291;
                  }
                  while ( v290 != v291 );
                  v313 = v290;
                }
LABEL_454:
                while ( v313 != v291 )
                {
                  if ( *v291 < 0xFFFFFFFFFFFFFFFELL )
                    goto LABEL_444;
                  ++v291;
                }
                if ( v290 != v291 )
                  goto LABEL_445;
              }
              else
              {
                v290 = &v293[*(unsigned int *)(v289 + 544)];
                v291 = sub_16CC9F0(v394, *v288);
                if ( v294 == *v291 )
                {
                  v308 = *(_QWORD *)(v289 + 536);
                  if ( v308 == *(_QWORD *)(v289 + 528) )
                    v309 = *(unsigned int *)(v289 + 548);
                  else
                    v309 = *(unsigned int *)(v289 + 544);
                  v313 = (_QWORD *)(v308 + 8 * v309);
                  goto LABEL_454;
                }
                v292 = *(_QWORD *)(v289 + 536);
                if ( v292 == *(_QWORD *)(v289 + 528) )
                {
                  v291 = (_QWORD *)(v292 + 8LL * *(unsigned int *)(v289 + 548));
                  v313 = v291;
                  goto LABEL_454;
                }
                v291 = (_QWORD *)(v292 + 8LL * *(unsigned int *)(v289 + 544));
LABEL_444:
                if ( v290 != v291 )
                  goto LABEL_445;
              }
              if ( *(_BYTE *)(v294 + 16) != 62 )
                goto LABEL_445;
              v295 = (*(_BYTE *)(v294 + 23) & 0x40) != 0
                   ? *(_QWORD **)(v294 - 8)
                   : (_QWORD *)(v294 - 24LL * (*(_DWORD *)(v294 + 20) & 0xFFFFFFF));
              if ( *v295 != *v423 )
                goto LABEL_445;
              v296 = (unsigned int)v447;
              v392 = (__int64 *)&v446[(unsigned int)v447];
              if ( v446 != (__int64 **)v392 )
              {
                v297 = v289;
                v298 = v294;
                v299 = (__int64 *)v446;
                v300 = v297;
                while ( 1 )
                {
                  v301 = sub_15CCEE0((__int64)&v433, v298, *v299);
                  if ( v301 )
                    break;
                  v304 = sub_15CCEE0((__int64)&v433, *v299, v298);
                  if ( v304 )
                  {
                    v405 = v304;
                    v310 = v300;
                    v311 = (_QWORD *)v298;
                    v312 = v298;
                    v289 = v310;
                    sub_164D160(
                      v312,
                      *v299,
                      a3,
                      *(double *)a4.m128i_i64,
                      *(double *)a5.m128i_i64,
                      *(double *)a6.m128_u64,
                      v305,
                      v306,
                      *(double *)a9.m128i_i64,
                      a10);
                    sub_1412190(v394, (__int64)v311);
                    sub_15F2070(v311);
                    goto LABEL_445;
                  }
                  if ( v392 == ++v299 )
                  {
                    v307 = v300;
                    v294 = v298;
                    v289 = v307;
                    v296 = (unsigned int)v447;
                    goto LABEL_466;
                  }
                }
                v405 = v301;
                v314 = v300;
                v315 = v298;
                v316 = v298;
                v289 = v314;
                sub_164D160(
                  *v299,
                  v316,
                  a3,
                  *(double *)a4.m128i_i64,
                  *(double *)a5.m128i_i64,
                  *(double *)a6.m128_u64,
                  v302,
                  v303,
                  *(double *)a9.m128i_i64,
                  a10);
                sub_1412190(v394, *v299);
                sub_15F2070((_QWORD *)*v299);
                *v299 = v315;
LABEL_445:
                if ( v428 == ++v288 )
                  goto LABEL_469;
                continue;
              }
LABEL_466:
              if ( (unsigned int)v296 >= HIDWORD(v447) )
              {
                sub_16CD150((__int64)&v446, &v448, 0, 8, v198, v199);
                v296 = (unsigned int)v447;
              }
              ++v288;
              v446[v296] = (__int64 *)v294;
              LODWORD(v447) = v447 + 1;
              if ( v428 == v288 )
              {
LABEL_469:
                v286 = v423;
                v285 = v289;
                if ( v446 != &v448 )
                  _libc_free((unsigned __int64)v446);
                goto LABEL_471;
              }
            }
          }
        }
      }
LABEL_291:
      if ( v439 )
      {
        v200 = v437;
        v201 = &v437[2 * v439];
        do
        {
          if ( *v200 != -8 && *v200 != -16 )
          {
            v202 = v200[1];
            if ( v202 )
            {
              v203 = *(_QWORD *)(v202 + 24);
              if ( v203 )
                j_j___libc_free_0(v203, *(_QWORD *)(v202 + 40) - v203);
              j_j___libc_free_0(v202, 56);
            }
          }
          v200 += 2;
        }
        while ( v201 != v200 );
      }
      j___libc_free_0(v437);
      if ( v433 != (__int64 *)&v435 )
        _libc_free((unsigned __int64)v433);
    }
    v135 = *(unsigned int *)(v10 + 736);
    if ( (_DWORD)v135 )
    {
      v406 |= sub_1D6E200(
                v10,
                a3,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(double *)a6.m128_u64,
                v76,
                v77,
                *(double *)a9.m128i_i64,
                a10);
      v136 = *(__int64 **)(v10 + 536);
      if ( v136 != *(__int64 **)(v10 + 528) )
      {
LABEL_178:
        v137 = *(unsigned int *)(v10 + 544);
        v138 = &v136[v137];
        goto LABEL_179;
      }
    }
    else
    {
      v136 = *(__int64 **)(v10 + 536);
      if ( v136 != *(__int64 **)(v10 + 528) )
        goto LABEL_178;
    }
    v137 = *(unsigned int *)(v10 + 548);
    v138 = &v136[v137];
LABEL_179:
    if ( v136 != v138 )
    {
      while ( 1 )
      {
        v139 = *v136;
        v140 = v136;
        if ( (unsigned __int64)*v136 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v138 == ++v136 )
          goto LABEL_182;
      }
      while ( v138 != v140 )
      {
        sub_164BEC0(
          v139,
          v135,
          v137,
          v75,
          a3,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128_u64,
          v76,
          v77,
          *(double *)a9.m128i_i64,
          a10);
        v141 = v140 + 1;
        if ( v140 + 1 == v138 )
          break;
        while ( 1 )
        {
          v139 = *v141;
          v140 = v141;
          if ( (unsigned __int64)*v141 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v138 == ++v141 )
          {
            if ( v406 )
              goto LABEL_183;
            goto LABEL_189;
          }
        }
      }
    }
LABEL_182:
    if ( !v406 )
      break;
LABEL_183:
    v396 = v406;
  }
LABEL_189:
  sub_1D672E0(v399);
  if ( *(_BYTE *)(v10 + 304) )
  {
    v280 = *(_DWORD *)(v10 + 296);
    if ( v280 )
    {
      v281 = *(_QWORD **)(v10 + 280);
      v282 = &v281[2 * v280];
      do
      {
        if ( *v281 != -8 && *v281 != -4 )
        {
          v283 = v281[1];
          if ( v283 )
            sub_161E7C0((__int64)(v281 + 1), v283);
        }
        v281 += 2;
      }
      while ( v282 != v281 );
    }
    j___libc_free_0(*(_QWORD *)(v10 + 280));
    *(_BYTE *)(v10 + 304) = 0;
  }
  v411 = byte_4FC3300;
  if ( !byte_4FC3300 )
  {
    v214 = (unsigned __int64 *)&v435;
    v433 = 0;
    v434 = 1;
    do
      *v214++ = -8;
    while ( v214 != (unsigned __int64 *)&v443 );
    v443 = v445;
    v444 = 0x800000000LL;
    v215 = a2[10];
    v216 = v215;
    if ( (__int64 *)v215 == v401 )
    {
      if ( !v396 )
      {
        v275 = 0;
LABEL_403:
        if ( (v434 & 1) == 0 )
          j___libc_free_0(v435);
        v396 = v275;
        goto LABEL_191;
      }
      v232 = 0;
    }
    else
    {
      v416 = 0;
      do
      {
        v217 = v215 - 24;
        if ( !v215 )
          v217 = 0;
        v404 = v217;
        v218 = sub_157EBA0(v217);
        if ( v218 )
        {
          v426 = sub_15F4D60(v218);
          v219 = sub_157EBA0(v217);
          v446 = &v448;
          v222 = v219;
          v408 = v426;
          v447 = 0x200000000LL;
          if ( (unsigned __int64)v426 <= 2 )
          {
            v224 = (__int64 *)&v448;
            v223 = 0;
          }
          else
          {
            sub_16CD150((__int64)&v446, &v448, v426, 8, v220, v221);
            v223 = v447;
            v224 = (__int64 *)&v446[(unsigned int)v447];
          }
          if ( v426 )
          {
            for ( kk = 0; kk != v426; ++kk )
            {
              v226 = sub_15F4DF0(v222, kk);
              if ( v224 )
                *v224 = v226;
              ++v224;
            }
            v223 = v447;
          }
        }
        else
        {
          v446 = &v448;
          v223 = 0;
          HIDWORD(v447) = 2;
          v408 = 0;
        }
        LODWORD(v447) = v408 + v223;
        v416 |= sub_1AEE9C0(v404, 1u, 0, 0);
        if ( v416 )
        {
          v276 = (__int64)v446;
          v277 = &v446[(unsigned int)v447];
          if ( v446 != v277 )
          {
            do
            {
              while ( 1 )
              {
                v278 = *(_QWORD *)(*(_QWORD *)v276 + 8LL);
                if ( v278 )
                  break;
LABEL_415:
                v279 = (__int64 *)v276;
                v276 += 8;
                sub_1D6FA20((__int64)&v433, v279, v227, v228, v229, v230);
                if ( v277 == (__int64 **)v276 )
                  goto LABEL_416;
              }
              while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v278) + 16) - 25) > 9u )
              {
                v278 = *(_QWORD *)(v278 + 8);
                if ( !v278 )
                  goto LABEL_415;
              }
              v276 += 8;
            }
            while ( v277 != (__int64 **)v276 );
LABEL_416:
            v277 = v446;
          }
          if ( v277 != &v448 )
            _libc_free((unsigned __int64)v277);
        }
        else if ( v446 != &v448 )
        {
          _libc_free((unsigned __int64)v446);
        }
        v215 = *(_QWORD *)(v215 + 8);
      }
      while ( (__int64 *)v215 != v401 );
      v231 = (unsigned int)v444;
      v232 = v416 | ((_DWORD)v444 != 0);
      if ( (_DWORD)v444 )
      {
        while ( 1 )
        {
          v233 = *(_QWORD *)&v443[8 * v231 - 8];
          if ( (v434 & 1) != 0 )
            break;
          v234 = v435;
          if ( (_DWORD)v436 )
          {
            v235 = v436 - 1;
LABEL_343:
            v236 = v235 & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
            v237 = &v234[v236];
            v238 = *v237;
            if ( *v237 == v233 )
            {
LABEL_344:
              *v237 = -16;
              ++HIDWORD(v434);
              LODWORD(v434) = (2 * ((unsigned int)v434 >> 1) - 2) | v434 & 1;
            }
            else
            {
              v362 = 1;
              while ( v238 != -8 )
              {
                v363 = v362 + 1;
                v236 = v235 & (v362 + v236);
                v237 = &v234[v236];
                v238 = *v237;
                if ( v233 == *v237 )
                  goto LABEL_344;
                v362 = v363;
              }
            }
          }
          LODWORD(v444) = v444 - 1;
          v239 = sub_157EBA0(v233);
          if ( v239 )
          {
            v427 = sub_15F4D60(v239);
            v242 = sub_157EBA0(v233);
            v446 = &v448;
            v245 = v242;
            v417 = v427;
            v447 = 0x200000000LL;
            if ( (unsigned __int64)v427 <= 2 )
            {
              v247 = (__int64 *)&v448;
              v246 = 0;
            }
            else
            {
              sub_16CD150((__int64)&v446, &v448, v427, 8, v243, v244);
              v246 = v447;
              v247 = (__int64 *)&v446[(unsigned int)v447];
            }
            if ( v427 )
            {
              v248 = v233;
              v249 = 0;
              v250 = v245;
              v251 = v248;
              do
              {
                v252 = sub_15F4DF0(v250, v249);
                if ( v247 )
                  *v247 = v252;
                ++v247;
                ++v249;
              }
              while ( v249 != v427 );
              v246 = v447;
              v233 = v251;
            }
          }
          else
          {
            v446 = &v448;
            v246 = 0;
            HIDWORD(v447) = 2;
            v417 = 0;
          }
          LODWORD(v447) = v417 + v246;
          sub_1AA7270(
            v233,
            0,
            a3,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128_u64,
            v240,
            v241,
            *(double *)a9.m128i_i64,
            a10);
          v257 = (__int64)v446;
          v258 = &v446[(unsigned int)v447];
          if ( v446 != v258 )
          {
            do
            {
              while ( 1 )
              {
                v259 = *(_QWORD *)(*(_QWORD *)v257 + 8LL);
                if ( v259 )
                  break;
LABEL_360:
                v260 = (__int64 *)v257;
                v257 += 8;
                sub_1D6FA20((__int64)&v433, v260, v253, v254, v255, v256);
                if ( v258 == (__int64 **)v257 )
                  goto LABEL_361;
              }
              while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v259) + 16) - 25) > 9u )
              {
                v259 = *(_QWORD *)(v259 + 8);
                if ( !v259 )
                  goto LABEL_360;
              }
              v257 += 8;
            }
            while ( v258 != (__int64 **)v257 );
LABEL_361:
            v258 = v446;
          }
          if ( v258 != &v448 )
            _libc_free((unsigned __int64)v258);
          v231 = (unsigned int)v444;
          if ( !(_DWORD)v444 )
          {
            v232 = 1;
            v216 = a2[10];
            goto LABEL_366;
          }
        }
        v234 = &v435;
        v235 = 7;
        goto LABEL_343;
      }
      v275 = v232 | v396;
      if ( !((unsigned __int8)v232 | v396) )
        goto LABEL_401;
      v216 = a2[10];
    }
LABEL_366:
    v446 = &v448;
    v447 = 0x1000000000LL;
    v261 = *(__int64 **)(v216 + 8);
    if ( v401 != v261 )
    {
      do
      {
        v262 = v261 - 3;
        v430 = 6;
        if ( !v261 )
          v262 = 0;
        v431 = 0;
        v432[0] = v262;
        if ( v262 + 1 != 0 && v262 != 0 && v262 != (__int64 *)-16LL )
          sub_164C220((__int64)&v430);
        v263 = v447;
        if ( (unsigned int)v447 >= HIDWORD(v447) )
        {
          sub_170B450((__int64)&v446, 0);
          v263 = v447;
        }
        v264 = (unsigned __int64 *)&v446[3 * v263];
        if ( v264 )
        {
          *v264 = 6;
          v264[1] = 0;
          v265 = v432[0];
          v115 = v432[0] == -8;
          v264[2] = v432[0];
          if ( v265 != 0 && !v115 && v265 != -16 )
            sub_1649AC0(v264, v430 & 0xFFFFFFFFFFFFFFF8LL);
          v263 = v447;
        }
        LODWORD(v447) = v263 + 1;
        if ( v432[0] != 0 && v432[0] != -8 && v432[0] != -16 )
          sub_1649B30(&v430);
        v261 = (__int64 *)v261[1];
      }
      while ( v401 != v261 );
      v266 = (__int64)v446;
      v267 = &v446[3 * (unsigned int)v447];
      if ( v446 != v267 )
      {
        do
        {
          v268 = *(_QWORD *)(v266 + 16);
          if ( v268 )
          {
            v269 = sub_157F0B0(*(_QWORD *)(v266 + 16));
            if ( v269 )
            {
              if ( v268 != v269 && !*(_WORD *)(v268 + 18) )
              {
                v270 = sub_157EBA0(v269);
                if ( *(_BYTE *)(v270 + 16) == 26 && (*(_DWORD *)(v270 + 20) & 0xFFFFFFF) != 3 )
                {
                  sub_1AA7EA0(
                    v268,
                    0,
                    0,
                    0,
                    0,
                    a3,
                    a4,
                    a5,
                    *(double *)a6.m128_u64,
                    v271,
                    v272,
                    *(double *)a9.m128i_i64,
                    a10);
                  v411 = 1;
                }
              }
            }
          }
          v266 += 24;
        }
        while ( v267 != (__int64 **)v266 );
        v273 = (__int64)v446;
        v232 |= v411;
        v267 = &v446[3 * (unsigned int)v447];
        if ( v446 != v267 )
        {
          do
          {
            v274 = (__int64)*(v267 - 1);
            v267 -= 3;
            if ( v274 != -8 && v274 != 0 && v274 != -16 )
              sub_1649B30(v267);
          }
          while ( (__int64 **)v273 != v267 );
          v267 = v446;
        }
      }
      if ( v267 != &v448 )
        _libc_free((unsigned __int64)v267);
    }
    v275 = v232 | v396;
LABEL_401:
    if ( v443 != v445 )
      _libc_free((unsigned __int64)v443);
    goto LABEL_403;
  }
LABEL_191:
  if ( !byte_4FC3220 )
  {
    v430 = (__int64)v432;
    v431 = 0x200000000LL;
    v142 = a2[10];
    if ( (__int64 *)v142 != v401 )
    {
      do
      {
        if ( !v142 )
          BUG();
        v143 = *(_QWORD *)(v142 + 24);
        for ( mm = v142 + 16; mm != v143; v143 = *(_QWORD *)(v143 + 8) )
        {
          while ( 1 )
          {
            v145 = v143 - 24;
            if ( !v143 )
              v145 = 0;
            if ( sub_1642D70(v145) )
              break;
            v143 = *(_QWORD *)(v143 + 8);
            if ( mm == v143 )
              goto LABEL_203;
          }
          v148 = (unsigned int)v431;
          if ( (unsigned int)v431 >= HIDWORD(v431) )
          {
            sub_16CD150((__int64)&v430, v432, 0, 8, v146, v147);
            v148 = (unsigned int)v431;
          }
          *(_QWORD *)(v430 + 8 * v148) = v145;
          LODWORD(v431) = v431 + 1;
        }
LABEL_203:
        v142 = *(_QWORD *)(v142 + 8);
      }
      while ( (__int64 *)v142 != v401 );
      v149 = (__int64 *)v430;
      v150 = (_QWORD *)(v430 + 8LL * (unsigned int)v431);
      if ( (_QWORD *)v430 != v150 )
      {
        v412 = (__int64 *)(v430 + 8LL * (unsigned int)v431);
        v151 = v396;
        do
        {
          v152 = *v149;
          v434 = 0x200000000LL;
          v433 = (__int64 *)&v435;
          v153 = *(_QWORD *)(v152 + 8);
          if ( v153 )
          {
            v154 = 0;
            do
            {
              while ( 1 )
              {
                v155 = sub_1648700(v153);
                if ( *((_BYTE *)v155 + 16) == 78 )
                {
                  v158 = *(v155 - 3);
                  if ( !*(_BYTE *)(v158 + 16) && (*(_BYTE *)(v158 + 33) & 0x20) != 0 && *(_DWORD *)(v158 + 36) == 76 )
                    break;
                }
                v153 = *(_QWORD *)(v153 + 8);
                if ( !v153 )
                  goto LABEL_216;
              }
              if ( HIDWORD(v434) <= v154 )
              {
                v407 = v155;
                sub_16CD150((__int64)&v433, &v435, 0, 8, v156, v157);
                v154 = v434;
                v155 = v407;
              }
              v433[v154] = (__int64)v155;
              v154 = v434 + 1;
              LODWORD(v434) = v434 + 1;
              v153 = *(_QWORD *)(v153 + 8);
            }
            while ( v153 );
LABEL_216:
            if ( v154 > 1 )
            {
              v446 = 0;
              v447 = 0;
              v448 = 0;
              LODWORD(v449) = 0;
              sub_1D68B30(&v433, (__int64)&v446);
              if ( (_DWORD)v448 )
              {
                v161 = (_QWORD *)v447;
                v162 = (unsigned int)v449;
                v163 = (__int64 *)(v447 + 40LL * (unsigned int)v449);
                if ( (__int64 *)v447 != v163 )
                {
                  v164 = (__int64 *)v447;
                  while ( 1 )
                  {
                    v165 = *v164;
                    v166 = v164;
                    if ( *v164 != -8 && v165 != -16 )
                      break;
                    v164 += 5;
                    if ( v163 == v164 )
                      goto LABEL_223;
                  }
                  if ( v164 != v163 )
                  {
                    do
                    {
                      v204 = sub_1D65210(
                               v165,
                               (__int64)(v166 + 1),
                               a3,
                               *(double *)a4.m128i_i64,
                               *(double *)a5.m128i_i64,
                               *(double *)a6.m128_u64,
                               v159,
                               v160,
                               *(double *)a9.m128i_i64,
                               a10);
                      v205 = v166 + 5;
                      if ( v166 + 5 == v163 )
                        break;
                      while ( 1 )
                      {
                        v165 = *v205;
                        v166 = v205;
                        if ( *v205 != -16 && v165 != -8 )
                          break;
                        v205 += 5;
                        if ( v163 == v205 )
                          goto LABEL_308;
                      }
                    }
                    while ( v205 != v163 );
LABEL_308:
                    v162 = (unsigned int)v449;
                    v161 = (_QWORD *)v447;
                    v151 |= v204;
                  }
                }
              }
              else
              {
                v162 = (unsigned int)v449;
                v161 = (_QWORD *)v447;
              }
LABEL_223:
              if ( (_DWORD)v162 )
              {
                v167 = &v161[5 * v162];
                do
                {
                  if ( *v161 != -16 && *v161 != -8 )
                  {
                    v168 = v161[1];
                    if ( (_QWORD *)v168 != v161 + 3 )
                    {
                      v415 = v161;
                      _libc_free(v168);
                      v161 = v415;
                    }
                  }
                  v161 += 5;
                }
                while ( v167 != v161 );
                v161 = (_QWORD *)v447;
              }
              j___libc_free_0(v161);
            }
            if ( v433 != (__int64 *)&v435 )
              _libc_free((unsigned __int64)v433);
          }
          ++v149;
        }
        while ( v412 != v149 );
        v396 = v151;
        v150 = (_QWORD *)v430;
      }
      if ( v150 != v432 )
        _libc_free((unsigned __int64)v150);
    }
  }
  return v396;
}
