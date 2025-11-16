// Function: sub_17B2B40
// Address: 0x17b2b40
//
__int64 __fastcall sub_17B2B40(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  __int64 ***v12; // r12
  size_t v13; // rax
  unsigned __int8 *v14; // rdi
  _QWORD *v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rdx
  __int64 **v18; // rcx
  __m128 v19; // xmm0
  __m128i v20; // xmm1
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r12
  _QWORD *v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  unsigned __int8 *v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // r14
  __int64 v33; // rbx
  _QWORD *v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // rax
  __int64 v38; // r11
  __int64 v39; // r9
  unsigned __int8 v40; // dl
  int v41; // r8d
  __int64 v42; // rax
  int v43; // r15d
  __int64 v44; // rax
  _BYTE *v45; // r12
  __int64 v46; // rsi
  char v47; // al
  __int64 v48; // r10
  __int64 v49; // r9
  __int64 *v50; // r11
  __int64 v51; // rax
  unsigned __int8 *v52; // r15
  __int64 v53; // r15
  unsigned __int8 v54; // cl
  __int64 v55; // r9
  __int64 v56; // r8
  __int64 v57; // r10
  int v58; // eax
  unsigned int v59; // r12d
  char v60; // al
  __int64 *v61; // r10
  __int64 *v62; // r9
  __int64 v63; // r11
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int8 *v66; // rax
  unsigned __int8 *v67; // rax
  unsigned __int8 *v68; // r12
  __int64 v69; // rax
  __int64 v70; // rdi
  char v71; // al
  __int64 v72; // rax
  char v73; // al
  __int64 v74; // r15
  unsigned int *v75; // r12
  char v76; // r8
  unsigned int v77; // eax
  unsigned int v78; // ecx
  unsigned int v79; // ecx
  int *v80; // rdx
  unsigned int v81; // ecx
  int v82; // esi
  __int64 v83; // r12
  int v84; // r8d
  __int64 v85; // r9
  __int64 v86; // rax
  unsigned int *v87; // rcx
  int v88; // eax
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rcx
  unsigned __int64 v92; // rsi
  __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rcx
  unsigned __int64 v96; // rsi
  __int64 v97; // rcx
  __int64 v98; // rax
  __int64 v99; // rcx
  unsigned __int64 v100; // rsi
  __int64 v101; // rcx
  __int64 v102; // r12
  __int64 v103; // rbx
  _QWORD *v104; // rax
  double v105; // xmm4_8
  double v106; // xmm5_8
  __int64 **v107; // r15
  __int64 v108; // rsi
  __int64 v109; // r12
  __int64 *v110; // rax
  __int64 v111; // r15
  __int64 v112; // r14
  __int64 v113; // r13
  _QWORD *v114; // rax
  int v115; // r8d
  int v116; // r9d
  __int64 v117; // rax
  unsigned int v118; // r14d
  __int64 **v119; // r12
  __int64 *v120; // r13
  int v121; // r14d
  __int64 v122; // rdi
  __int64 v123; // r8
  unsigned int v124; // r15d
  __int64 v125; // rsi
  __int64 v126; // rax
  __int64 v127; // rax
  _QWORD *v128; // rax
  unsigned int v129; // r14d
  unsigned int v130; // edx
  __int64 ***v131; // rax
  __int64 **v132; // rcx
  unsigned __int8 *v133; // r15
  __int64 v134; // rdx
  __int64 v135; // rax
  unsigned __int8 *v136; // rax
  __int64 v137; // r14
  __int64 v138; // r15
  _QWORD *v139; // rax
  double v140; // xmm4_8
  double v141; // xmm5_8
  __int64 v142; // r12
  __int64 v143; // rbx
  _QWORD *v144; // rax
  double v145; // xmm4_8
  double v146; // xmm5_8
  char v147; // al
  __int64 v148; // rax
  __int64 v149; // r14
  __int64 v150; // rbx
  _QWORD *v151; // rax
  __int64 v152; // rdx
  __int64 v153; // r15
  __int64 ***v154; // rcx
  size_t v155; // r9
  __int64 ***v156; // rax
  __int64 ***v157; // rdi
  bool v158; // r10
  __int64 v159; // rdx
  __int64 v160; // rcx
  int v161; // r9d
  bool v162; // r10
  __int64 v163; // rdx
  __int64 v164; // rcx
  int v165; // r8d
  int v166; // r9d
  int v167; // eax
  unsigned int v168; // ecx
  __int64 v169; // r12
  __int64 v170; // rdx
  size_t v171; // r10
  __m128 *v172; // r11
  __int64 v173; // r8
  int v174; // eax
  int v175; // r9d
  int v176; // r13d
  __int64 v177; // rbx
  __int64 v178; // rsi
  __m128 *v179; // r14
  __int64 *v180; // rbx
  unsigned int v181; // r11d
  __int64 v182; // rcx
  void *v183; // rsi
  void *v184; // r12
  size_t v185; // rdx
  __int64 v186; // r13
  __int64 v187; // r15
  int v188; // r8d
  __int64 v189; // r9
  __int64 v190; // rax
  int v191; // eax
  __int64 v192; // rax
  _QWORD *v193; // r13
  _QWORD *v194; // rax
  size_t v195; // rdx
  int v196; // eax
  int v197; // eax
  double v198; // xmm4_8
  double v199; // xmm5_8
  __int64 v200; // rax
  __int64 v201; // r12
  __int64 v202; // rbx
  _QWORD *v203; // rax
  __int64 v204; // r15
  __int64 v205; // rdx
  __int64 v206; // rdx
  unsigned __int64 v207; // rcx
  __int64 v208; // rax
  __int64 v209; // r8
  int v210; // r9d
  __int64 *v211; // rax
  __int64 *v212; // rdx
  int v213; // r15d
  __int64 v214; // r12
  unsigned int v215; // ebx
  __int64 *v216; // r14
  __int64 *v217; // r13
  __int64 v218; // rax
  __int64 v219; // r15
  const char *v220; // rax
  __int64 v221; // rdx
  __int64 v222; // rax
  void *v223; // rsi
  int v224; // eax
  int v225; // eax
  int v226; // r9d
  __int64 v227; // [rsp+0h] [rbp-300h]
  __m128i *v228; // [rsp+0h] [rbp-300h]
  __int64 v229; // [rsp+8h] [rbp-2F8h]
  __int64 v230; // [rsp+10h] [rbp-2F0h]
  __m128 *v231; // [rsp+18h] [rbp-2E8h]
  unsigned __int32 v232; // [rsp+20h] [rbp-2E0h]
  __int64 v233; // [rsp+20h] [rbp-2E0h]
  unsigned int v234; // [rsp+28h] [rbp-2D8h]
  size_t v235; // [rsp+28h] [rbp-2D8h]
  int v236; // [rsp+30h] [rbp-2D0h]
  __int64 **v237; // [rsp+30h] [rbp-2D0h]
  __int64 v238; // [rsp+30h] [rbp-2D0h]
  __int64 v239; // [rsp+38h] [rbp-2C8h]
  unsigned __int64 v240; // [rsp+38h] [rbp-2C8h]
  char v241; // [rsp+38h] [rbp-2C8h]
  size_t v242; // [rsp+38h] [rbp-2C8h]
  __int64 v243; // [rsp+38h] [rbp-2C8h]
  __int64 v244; // [rsp+40h] [rbp-2C0h]
  __int64 v245; // [rsp+40h] [rbp-2C0h]
  __int64 v246; // [rsp+40h] [rbp-2C0h]
  __int64 v247; // [rsp+48h] [rbp-2B8h]
  __int64 v248; // [rsp+48h] [rbp-2B8h]
  unsigned __int8 *v249; // [rsp+48h] [rbp-2B8h]
  __int64 v250; // [rsp+48h] [rbp-2B8h]
  int v251; // [rsp+50h] [rbp-2B0h]
  __int64 *v252; // [rsp+50h] [rbp-2B0h]
  __int64 v253; // [rsp+50h] [rbp-2B0h]
  __int64 v254; // [rsp+50h] [rbp-2B0h]
  __int64 v255; // [rsp+50h] [rbp-2B0h]
  __int64 v256; // [rsp+50h] [rbp-2B0h]
  __int64 v257; // [rsp+50h] [rbp-2B0h]
  __int64 *v258; // [rsp+50h] [rbp-2B0h]
  __int64 **v259; // [rsp+50h] [rbp-2B0h]
  bool v260; // [rsp+50h] [rbp-2B0h]
  int v261; // [rsp+50h] [rbp-2B0h]
  __int64 v262; // [rsp+58h] [rbp-2A8h]
  __int64 v263; // [rsp+58h] [rbp-2A8h]
  __int64 v264; // [rsp+58h] [rbp-2A8h]
  _QWORD *v265; // [rsp+58h] [rbp-2A8h]
  __int64 v266; // [rsp+60h] [rbp-2A0h]
  __int64 v267; // [rsp+60h] [rbp-2A0h]
  __int64 *v268; // [rsp+60h] [rbp-2A0h]
  unsigned __int8 v269; // [rsp+68h] [rbp-298h]
  int v270; // [rsp+68h] [rbp-298h]
  int v271; // [rsp+68h] [rbp-298h]
  __int64 v272; // [rsp+70h] [rbp-290h]
  int v273; // [rsp+70h] [rbp-290h]
  __int64 v274; // [rsp+70h] [rbp-290h]
  __int64 v275; // [rsp+70h] [rbp-290h]
  __int64 v276; // [rsp+70h] [rbp-290h]
  int v277; // [rsp+70h] [rbp-290h]
  char v278; // [rsp+70h] [rbp-290h]
  unsigned int v279; // [rsp+70h] [rbp-290h]
  __int64 v280; // [rsp+70h] [rbp-290h]
  __int64 v281; // [rsp+70h] [rbp-290h]
  __int64 v282; // [rsp+70h] [rbp-290h]
  __int64 v283; // [rsp+78h] [rbp-288h]
  __int64 *v284; // [rsp+78h] [rbp-288h]
  __int64 v285; // [rsp+78h] [rbp-288h]
  _BYTE *v286; // [rsp+78h] [rbp-288h]
  __int64 v287; // [rsp+78h] [rbp-288h]
  int v288; // [rsp+78h] [rbp-288h]
  __int64 v289; // [rsp+78h] [rbp-288h]
  __int64 *v290; // [rsp+80h] [rbp-280h]
  __int64 v291; // [rsp+80h] [rbp-280h]
  bool v292; // [rsp+80h] [rbp-280h]
  __int64 *v293; // [rsp+80h] [rbp-280h]
  __int64 *v294; // [rsp+80h] [rbp-280h]
  unsigned __int8 v295; // [rsp+80h] [rbp-280h]
  __int64 v296; // [rsp+80h] [rbp-280h]
  __int64 v297; // [rsp+80h] [rbp-280h]
  unsigned int v298; // [rsp+88h] [rbp-278h]
  int v299; // [rsp+88h] [rbp-278h]
  __int64 v300; // [rsp+88h] [rbp-278h]
  int v301; // [rsp+88h] [rbp-278h]
  char v302; // [rsp+88h] [rbp-278h]
  int v303; // [rsp+88h] [rbp-278h]
  __int64 v304; // [rsp+88h] [rbp-278h]
  __int64 v305; // [rsp+88h] [rbp-278h]
  __int64 **v306; // [rsp+90h] [rbp-270h]
  __int64 ***v307; // [rsp+90h] [rbp-270h]
  __int64 **v308; // [rsp+98h] [rbp-268h]
  __int64 ***v309; // [rsp+A0h] [rbp-260h]
  unsigned int v310; // [rsp+A0h] [rbp-260h]
  unsigned int v311; // [rsp+A0h] [rbp-260h]
  size_t n; // [rsp+B0h] [rbp-250h]
  size_t na; // [rsp+B0h] [rbp-250h]
  size_t nd; // [rsp+B0h] [rbp-250h]
  size_t nb; // [rsp+B0h] [rbp-250h]
  unsigned int ne; // [rsp+B0h] [rbp-250h]
  size_t nc; // [rsp+B0h] [rbp-250h]
  __int64 v318; // [rsp+B8h] [rbp-248h]
  unsigned int v319; // [rsp+B8h] [rbp-248h]
  int v320; // [rsp+B8h] [rbp-248h]
  __int64 v321; // [rsp+B8h] [rbp-248h]
  __int64 v322; // [rsp+C0h] [rbp-240h] BYREF
  unsigned int v323; // [rsp+C8h] [rbp-238h]
  unsigned __int64 v324; // [rsp+D0h] [rbp-230h] BYREF
  unsigned int v325; // [rsp+D8h] [rbp-228h]
  _QWORD v326[2]; // [rsp+E0h] [rbp-220h] BYREF
  __int16 v327; // [rsp+F0h] [rbp-210h]
  void *v328; // [rsp+100h] [rbp-200h] BYREF
  __int64 v329; // [rsp+108h] [rbp-1F8h]
  _BYTE v330[64]; // [rsp+110h] [rbp-1F0h] BYREF
  void *s2; // [rsp+150h] [rbp-1B0h] BYREF
  __int64 v332; // [rsp+158h] [rbp-1A8h]
  _WORD v333[32]; // [rsp+160h] [rbp-1A0h] BYREF
  void *v334; // [rsp+1A0h] [rbp-160h] BYREF
  __int64 v335; // [rsp+1A8h] [rbp-158h]
  __int64 v336; // [rsp+1B0h] [rbp-150h] BYREF
  unsigned int v337; // [rsp+1B8h] [rbp-148h]
  void *s1; // [rsp+1F0h] [rbp-110h] BYREF
  __int64 v339; // [rsp+1F8h] [rbp-108h]
  _QWORD v340[8]; // [rsp+200h] [rbp-100h] BYREF
  __m128 v341; // [rsp+240h] [rbp-C0h] BYREF
  __m128i v342; // [rsp+250h] [rbp-B0h] BYREF
  __int64 v343; // [rsp+260h] [rbp-A0h]

  v10 = a2;
  v12 = *(__int64 ****)(a2 - 72);
  v13 = *(_QWORD *)(a2 - 48);
  v14 = *(unsigned __int8 **)(a2 - 24);
  v328 = v330;
  n = v13;
  v309 = v12;
  v329 = 0x1000000000LL;
  sub_15FAA20(v14, (__int64)&v328);
  v318 = a2;
  v15 = (_QWORD *)sub_16498A0(a2);
  v16 = sub_1643350(v15);
  v17 = *(_BYTE **)(a2 - 24);
  v18 = *(__int64 ***)a2;
  v19 = (__m128)_mm_loadu_si128(a1 + 167);
  v20 = _mm_loadu_si128(a1 + 168);
  v308 = (__int64 **)v16;
  v343 = a2;
  v341 = v19;
  v342 = v20;
  v21 = sub_13D1880(v12, n, v17, (__int64)v18);
  if ( v21 )
  {
    v22 = *(_QWORD *)(a2 + 8);
    if ( v22 )
    {
      v23 = a1->m128i_i64[0];
      v24 = v21;
      do
      {
        v25 = sub_1648700(v22);
        sub_170B990(v23, (__int64)v25);
        v22 = *(_QWORD *)(v22 + 8);
      }
      while ( v22 );
      if ( a2 == v24 )
        v24 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v24, v19, *(double *)v20.m128i_i64, a5, a6, v26, v27, a9, a10);
    }
    else
    {
      v318 = 0;
    }
    goto LABEL_8;
  }
  v29 = *(unsigned __int8 **)(a2 - 24);
  if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) != *(_DWORD *)(*(_QWORD *)v29 + 32LL) )
    goto LABEL_12;
  v37 = a1[166].m128i_i64[1];
  v341.m128_u64[1] = 0x1000000000LL;
  v341.m128_u64[0] = (unsigned __int64)&v342;
  v266 = v37;
  v262 = a1->m128i_i64[1];
  sub_15FAA20(v29, (__int64)&v341);
  v269 = sub_15FACD0((int *)v341.m128_u64[0], v341.m128_i32[2]);
  if ( (__m128i *)v341.m128_u64[0] != &v342 )
    _libc_free(v341.m128_u64[0]);
  if ( !v269 )
    goto LABEL_12;
  v38 = *(_QWORD *)(a2 - 72);
  v39 = *(_QWORD *)(a2 - 48);
  v40 = *(_BYTE *)(v38 + 16);
  if ( v40 > 0x17u
    && (unsigned int)v40 - 35 <= 0x11
    && v39 == *(_QWORD *)(v38 - 48)
    && *(_BYTE *)(*(_QWORD *)(v38 - 24) + 16LL) <= 0x10u )
  {
    v43 = v40 - 24;
    v276 = *(_QWORD *)(v38 - 24);
    v287 = *(_QWORD *)(a2 - 72);
    v293 = *(__int64 **)(a2 - 48);
    v303 = v40;
    v69 = sub_15A14F0(v43, *(__int64 ***)a2, 1);
    if ( v69 )
    {
      v45 = *(_BYTE **)(a2 - 24);
      v70 = v276;
      v277 = v303;
      v304 = sub_15A3950(v70, v69, v45, 0);
      v71 = sub_15A0F20((__int64)v45);
      v48 = v304;
      if ( v71 )
      {
        if ( (unsigned int)(v277 - 44) <= 1 || (unsigned int)(v277 - 41) <= 1 || (unsigned int)(v277 - 47) <= 2 )
        {
          v72 = sub_17ADE40(v43, v304, 1u);
          v342.m128i_i16[0] = 257;
          v52 = (unsigned __int8 *)sub_15FB440(v43, v293, v72, (__int64)&v341, 0);
          sub_15F2530(v52, v287, 1);
          sub_15A0F20((__int64)v45);
          goto LABEL_46;
        }
        v50 = v293;
        v49 = v287;
      }
      else
      {
        v50 = v293;
        v49 = v287;
      }
LABEL_87:
      v305 = v49;
      v342.m128i_i16[0] = 257;
      v52 = (unsigned __int8 *)sub_15FB440(v43, v50, v48, (__int64)&v341, 0);
      sub_15F2530(v52, v305, 1);
      if ( (unsigned __int8)sub_15A0F20((__int64)v45) )
        sub_15F2390((__int64)v52);
LABEL_46:
      if ( v52 )
      {
        v318 = (__int64)v52;
        goto LABEL_8;
      }
      goto LABEL_48;
    }
    goto LABEL_48;
  }
  v41 = *(unsigned __int8 *)(v39 + 16);
  if ( (unsigned __int8)v41 > 0x17u && (unsigned int)(v41 - 35) <= 0x11 )
  {
    v42 = *(_QWORD *)(v39 - 48);
    if ( v38 == v42 )
    {
      if ( v42 )
      {
        v272 = *(_QWORD *)(a2 - 48);
        v283 = *(_QWORD *)(v39 - 24);
        if ( *(_BYTE *)(v283 + 16) <= 0x10u )
        {
          v43 = v41 - 24;
          v290 = *(__int64 **)(a2 - 72);
          v299 = *(unsigned __int8 *)(v39 + 16);
          v44 = sub_15A14F0(v41 - 24, *(__int64 ***)a2, 1);
          if ( v44 )
          {
            v45 = *(_BYTE **)(a2 - 24);
            v46 = v283;
            v284 = v290;
            v291 = v272;
            v273 = v299;
            v300 = sub_15A3950(v44, v46, *(_BYTE **)(v10 - 24), 0);
            v47 = sub_15A0F20((__int64)v45);
            v48 = v300;
            v49 = v291;
            v50 = v284;
            if ( v47 )
            {
              if ( (unsigned int)(v273 - 44) <= 1 || (unsigned int)(v273 - 41) <= 1 || (unsigned int)(v273 - 47) <= 2 )
              {
                v51 = sub_17ADE40(v43, v300, 1u);
                v342.m128i_i16[0] = 257;
                v52 = (unsigned __int8 *)sub_15FB440(v43, v284, v51, (__int64)&v341, 0);
                sub_15F2530(v52, v291, 1);
                sub_15A0F20((__int64)v45);
                goto LABEL_46;
              }
              v49 = v291;
            }
            goto LABEL_87;
          }
LABEL_48:
          v38 = *(_QWORD *)(v10 - 72);
          v40 = *(_BYTE *)(v38 + 16);
        }
      }
    }
  }
  if ( v40 <= 0x17u )
    goto LABEL_12;
  if ( (unsigned int)v40 - 35 > 0x11 )
    goto LABEL_12;
  v53 = *(_QWORD *)(v10 - 48);
  v54 = *(_BYTE *)(v53 + 16);
  if ( v54 <= 0x17u || (unsigned int)v54 - 35 > 0x11 )
    goto LABEL_12;
  v55 = *(_QWORD *)(v38 - 48);
  if ( !v55 )
  {
    if ( MEMORY[0x10] > 0x10u )
      goto LABEL_12;
    v56 = *(_QWORD *)(v38 - 24);
    if ( !v56 )
      goto LABEL_12;
    goto LABEL_303;
  }
  v56 = *(_QWORD *)(v38 - 24);
  if ( *(_BYTE *)(v56 + 16) > 0x10u
    || (v274 = *(_QWORD *)(v53 - 48)) == 0
    || (v57 = *(_QWORD *)(v53 - 24), *(_BYTE *)(v57 + 16) > 0x10u) )
  {
    if ( *(_BYTE *)(v55 + 16) > 0x10u )
      goto LABEL_12;
LABEL_303:
    v57 = *(_QWORD *)(v53 - 48);
    if ( *(_BYTE *)(v57 + 16) > 0x10u )
      goto LABEL_12;
    v274 = *(_QWORD *)(v53 - 24);
    if ( !v274 )
      goto LABEL_12;
    v59 = v40 - 24;
    v55 = v56;
    v58 = v54 - 24;
    v269 = 0;
    v56 = *(_QWORD *)(v38 - 48);
    v292 = 0;
    goto LABEL_60;
  }
  v58 = v54 - 24;
  v301 = v40 - 24;
  if ( v54 == v40 )
  {
    v292 = 0;
    v59 = v40 - 24;
  }
  else
  {
    v239 = *(_QWORD *)(v38 - 24);
    v244 = *(_QWORD *)(v38 - 48);
    v247 = *(_QWORD *)(v53 - 24);
    v251 = v54 - 24;
    v285 = v38;
    v292 = v58 == 23 || v301 == 23;
    sub_17ADBD0((__int64)&s1, v38, v266, *(double *)v19.m128_u64, *(double *)v20.m128i_i64, a5);
    v59 = (unsigned int)s1;
    v38 = v285;
    v58 = v251;
    v57 = v247;
    v55 = v244;
    if ( (_DWORD)s1 )
    {
      v56 = v340[0];
    }
    else
    {
      v246 = v285;
      v250 = v55;
      v289 = v57;
      sub_17ADBD0((__int64)&v341, v53, v266, *(double *)v19.m128_u64, *(double *)v20.m128i_i64, a5);
      v57 = v289;
      v58 = v251;
      v55 = v250;
      v38 = v246;
      v56 = v239;
      if ( v341.m128_i32[0] )
      {
        v57 = v342.m128i_i64[0];
        v58 = v341.m128_i32[0];
      }
      v59 = v301;
    }
  }
LABEL_60:
  v248 = v38;
  v252 = (__int64 *)v55;
  if ( v59 != v58 )
    goto LABEL_12;
  v286 = *(_BYTE **)(v10 - 24);
  v267 = sub_15A3950(v56, v57, v286, 0);
  v60 = sub_15A0F20((__int64)v286);
  v61 = (__int64 *)v267;
  v62 = v252;
  v302 = v60;
  v63 = v248;
  if ( v60 )
  {
    if ( v59 <= 0x19
      && (((unsigned __int64)"p16,1) (__edg_scalable_vector_type__(unsigned long,1)) __edg_throw__()" >> v59) & 1) != 0 )
    {
      v208 = sub_17ADE40(v59, v267, v269);
      v62 = v252;
      v63 = v248;
      v61 = (__int64 *)v208;
    }
    else
    {
      v302 = 0;
    }
  }
  if ( (__int64 *)v274 != v62 )
  {
    v64 = *(_QWORD *)(v63 + 8);
    if ( !v64 || *(_QWORD *)(v64 + 8) )
    {
      v65 = *(_QWORD *)(v53 + 8);
      if ( !v65 || *(_QWORD *)(v65 + 8) )
        goto LABEL_12;
    }
    if ( v269 != 1 && v302 )
      goto LABEL_12;
    v253 = v63;
    v268 = v61;
    v342.m128i_i16[0] = 257;
    v66 = sub_17AF270(v262, (__int64)v62, v274, (__int64)v286, (__int64 *)&v341);
    v63 = v253;
    v61 = v268;
    v62 = (__int64 *)v66;
  }
  v275 = v63;
  v342.m128i_i16[0] = 257;
  if ( v269 )
    v67 = (unsigned __int8 *)sub_15FB440(v59, v62, (__int64)v61, (__int64)&v341, 0);
  else
    v67 = (unsigned __int8 *)sub_15FB440(v59, v61, (__int64)v62, (__int64)&v341, 0);
  v68 = v67;
  sub_15F2530(v67, v275, 1);
  sub_15F2780(v68, v53);
  if ( v292 )
    sub_15F2330((__int64)v68, 0);
  if ( (unsigned __int8)sub_15A0F20((__int64)v286) == 1 && !v302 )
    sub_15F2390((__int64)v68);
  if ( v68 )
  {
    v318 = (__int64)v68;
    goto LABEL_8;
  }
LABEL_12:
  v30 = *(_QWORD *)(*(_QWORD *)v10 + 32LL);
  v298 = v30;
  v323 = v30;
  if ( (unsigned int)v30 > 0x40 )
  {
    sub_16A4EF0((__int64)&v322, 0, 0);
    v325 = v298;
    sub_16A4EF0((__int64)&v324, -1, 1);
    v341.m128_i32[2] = v325;
    if ( v325 > 0x40 )
    {
      sub_16A4FD0((__int64)&v341, (const void **)&v324);
      goto LABEL_15;
    }
  }
  else
  {
    v322 = 0;
    v325 = v30;
    v341.m128_i32[2] = v30;
    v324 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
  }
  v341.m128_u64[0] = v324;
LABEL_15:
  v31 = (__int64)sub_17A4D70(a1->m128i_i64, (_BYTE *)v10, (__int64)&v341, &v322, 0);
  if ( v341.m128_i32[2] > 0x40u && v341.m128_u64[0] )
    j_j___libc_free_0_0(v341.m128_u64[0]);
  if ( v31 )
  {
    if ( v10 == v31 )
      goto LABEL_24;
    v32 = *(_QWORD *)(v10 + 8);
    if ( v32 )
    {
      v33 = a1->m128i_i64[0];
      do
      {
        v34 = sub_1648700(v32);
        sub_170B990(v33, (__int64)v34);
        v32 = *(_QWORD *)(v32 + 8);
      }
      while ( v32 );
LABEL_23:
      sub_164D160(v10, v31, v19, *(double *)v20.m128i_i64, a5, a6, v35, v36, a9, a10);
      goto LABEL_24;
    }
    goto LABEL_109;
  }
  v294 = (*v309)[4];
  v288 = (int)v294;
  v73 = *((_BYTE *)v309 + 16);
  if ( (__int64 ***)n == v309 )
  {
    if ( v73 == 9 )
    {
      if ( v298 != (_DWORD)v294 )
        v309 = (__int64 ***)sub_1599EF0(*(__int64 ***)v10);
      v102 = *(_QWORD *)(v10 + 8);
      if ( !v102 )
        goto LABEL_109;
      v103 = a1->m128i_i64[0];
      do
      {
        v104 = sub_1648700(v102);
        sub_170B990(v103, (__int64)v104);
        v102 = *(_QWORD *)(v102 + 8);
      }
      while ( v102 );
      goto LABEL_153;
    }
  }
  else
  {
    v278 = 0;
    if ( v73 != 9 )
      goto LABEL_96;
  }
  v341.m128_u64[1] = 0x1000000000LL;
  v341.m128_u64[0] = (unsigned __int64)&v342;
  if ( v298 )
  {
    v83 = 0;
    do
    {
      v87 = (unsigned int *)((char *)v328 + v83);
      v88 = *(_DWORD *)((char *)v328 + v83);
      if ( v88 >= 0 )
      {
        if ( v88 >= (int)v294 )
        {
          if ( *(_BYTE *)(n + 16) != 9 )
            goto LABEL_115;
        }
        else if ( *((_BYTE *)v309 + 16) != 9 )
        {
LABEL_115:
          *v87 = v88 % (unsigned int)v294;
          v85 = sub_15A0680((__int64)v308, *(int *)((char *)v328 + v83), 0);
          v86 = v341.m128_u32[2];
          if ( v341.m128_i32[2] >= (unsigned __int32)v341.m128_i32[3] )
            goto LABEL_122;
          goto LABEL_116;
        }
        *v87 = -1;
      }
      v85 = sub_1599EF0(v308);
      v86 = v341.m128_u32[2];
      if ( v341.m128_i32[2] >= (unsigned __int32)v341.m128_i32[3] )
      {
LABEL_122:
        v263 = v85;
        sub_16CD150((__int64)&v341, &v342, 0, 8, v84, v85);
        v86 = v341.m128_u32[2];
        v85 = v263;
      }
LABEL_116:
      v83 += 4;
      *(_QWORD *)(v341.m128_u64[0] + 8 * v86) = v85;
      ++v341.m128_i32[2];
    }
    while ( 4LL * v298 != v83 );
  }
  v89 = *(_QWORD *)(v10 - 48);
  v90 = *(_QWORD *)(v10 - 72);
  if ( v89 )
  {
    if ( v90 )
    {
      v91 = *(_QWORD *)(v10 - 64);
      v92 = *(_QWORD *)(v10 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v92 = v91;
      if ( v91 )
        *(_QWORD *)(v91 + 16) = v92 | *(_QWORD *)(v91 + 16) & 3LL;
    }
    *(_QWORD *)(v10 - 72) = v89;
    v93 = *(_QWORD *)(v89 + 8);
    *(_QWORD *)(v10 - 64) = v93;
    if ( v93 )
      *(_QWORD *)(v93 + 16) = (v10 - 64) | *(_QWORD *)(v93 + 16) & 3LL;
    *(_QWORD *)(v10 - 56) = (v89 + 8) | *(_QWORD *)(v10 - 56) & 3LL;
    *(_QWORD *)(v89 + 8) = v10 - 72;
  }
  else if ( v90 )
  {
    v206 = *(_QWORD *)(v10 - 64);
    v207 = *(_QWORD *)(v10 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v207 = v206;
    if ( v206 )
      *(_QWORD *)(v206 + 16) = v207 | *(_QWORD *)(v206 + 16) & 3LL;
    *(_QWORD *)(v10 - 72) = 0;
  }
  v94 = sub_1599EF0(*(__int64 ***)n);
  if ( *(_QWORD *)(v10 - 48) )
  {
    v95 = *(_QWORD *)(v10 - 40);
    v96 = *(_QWORD *)(v10 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v96 = v95;
    if ( v95 )
      *(_QWORD *)(v95 + 16) = v96 | *(_QWORD *)(v95 + 16) & 3LL;
  }
  *(_QWORD *)(v10 - 48) = v94;
  if ( v94 )
  {
    v97 = *(_QWORD *)(v94 + 8);
    *(_QWORD *)(v10 - 40) = v97;
    if ( v97 )
      *(_QWORD *)(v97 + 16) = (v10 - 40) | *(_QWORD *)(v97 + 16) & 3LL;
    *(_QWORD *)(v10 - 32) = (v94 + 8) | *(_QWORD *)(v10 - 32) & 3LL;
    *(_QWORD *)(v94 + 8) = v10 - 48;
  }
  v98 = sub_15A01B0((__int64 *)v341.m128_u64[0], v341.m128_u32[2]);
  if ( *(_QWORD *)(v10 - 24) )
  {
    v99 = *(_QWORD *)(v10 - 16);
    v100 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v100 = v99;
    if ( v99 )
      *(_QWORD *)(v99 + 16) = v100 | *(_QWORD *)(v99 + 16) & 3LL;
  }
  *(_QWORD *)(v10 - 24) = v98;
  if ( v98 )
  {
    v101 = *(_QWORD *)(v98 + 8);
    *(_QWORD *)(v10 - 16) = v101;
    if ( v101 )
      *(_QWORD *)(v101 + 16) = (v10 - 16) | *(_QWORD *)(v101 + 16) & 3LL;
    *(_QWORD *)(v10 - 8) = (v98 + 8) | *(_QWORD *)(v10 - 8) & 3LL;
    *(_QWORD *)(v98 + 8) = v10 - 24;
  }
  v309 = *(__int64 ****)(v10 - 72);
  n = *(_QWORD *)(v10 - 48);
  if ( (__m128i *)v341.m128_u64[0] != &v342 )
    _libc_free(v341.m128_u64[0]);
  v278 = 1;
LABEL_96:
  if ( v298 != (_DWORD)v294 )
  {
LABEL_97:
    v74 = (unsigned int)v329;
    v75 = (unsigned int *)v328;
    v76 = *(_BYTE *)(n + 16);
    v270 = v329;
    if ( v76 == 9 )
    {
      v147 = sub_17ADFE0((__int64)v309, v328, (unsigned int)v329, 5);
      v76 = 9;
      if ( v147 )
      {
        v148 = sub_17AE700((__int64)a1, v309, v75, v74);
        v149 = *(_QWORD *)(v10 + 8);
        v31 = v148;
        if ( v149 )
        {
          v150 = a1->m128i_i64[0];
          do
          {
            v151 = sub_1648700(v149);
            sub_170B990(v150, (__int64)v151);
            v149 = *(_QWORD *)(v149 + 8);
          }
          while ( v149 );
          if ( v10 == v31 )
            v31 = sub_1599EF0(*(__int64 ***)v10);
          goto LABEL_23;
        }
        goto LABEL_109;
      }
    }
    v77 = *v75;
    v78 = v75[v74 - 1];
    if ( v78 < *(_DWORD *)(**(_QWORD **)(v10 - 72) + 32LL) && v77 <= v78 )
    {
      v79 = v78 - v77;
      if ( v79 == v270 - 1 )
      {
        if ( v270 )
        {
          v80 = (int *)(v75 + 1);
          v81 = v77 + v79;
          while ( v81 != v77 )
          {
            v82 = *v80;
            ++v77;
            ++v80;
            if ( v82 != v77 )
              goto LABEL_105;
          }
        }
        v107 = *v309;
        v108 = (__int64)(*v309)[3];
        v109 = 0;
        v319 = *((_DWORD *)v107 + 8) * sub_1643030(v108);
        v234 = sub_127FA20(a1[166].m128i_i64[1], v108);
        v110 = v107[4];
        v334 = 0;
        v111 = *(_QWORD *)(v10 + 8);
        v232 = (unsigned int)v110;
        s1 = v340;
        v339 = 0x800000000LL;
        v335 = 0;
        v336 = 0;
        v337 = 0;
        if ( v111 )
        {
          v112 = v10;
          v113 = v111;
          do
          {
            v114 = sub_1648700(v113);
            if ( *((_BYTE *)v114 + 16) == 71 && v114[1] )
            {
              if ( HIDWORD(v339) <= (unsigned int)v109 )
              {
                v265 = v114;
                sub_16CD150((__int64)&s1, v340, 0, 8, v115, v116);
                v109 = (unsigned int)v339;
                v114 = v265;
              }
              *((_QWORD *)s1 + v109) = v114;
              v109 = (unsigned int)(v339 + 1);
              LODWORD(v339) = v339 + 1;
            }
            v113 = *(_QWORD *)(v113 + 8);
          }
          while ( v113 );
          v10 = v112;
          v306 = (__int64 **)((char *)s1 + 8 * (unsigned int)v109);
          if ( s1 != v306 )
          {
            v119 = (__int64 **)s1;
            v245 = v112;
            v229 = 8LL * v232;
            v249 = (unsigned __int8 *)v309;
            while ( 1 )
            {
              v120 = *v119;
              v121 = 1;
              v122 = a1[166].m128i_i64[1];
              v123 = **v119;
              v124 = *(_DWORD *)v328;
              v125 = v123;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v125 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v126 = *(_QWORD *)(v125 + 32);
                    v125 = *(_QWORD *)(v125 + 24);
                    v121 *= (_DWORD)v126;
                    continue;
                  case 1:
                    LODWORD(v117) = 16;
                    break;
                  case 2:
                    LODWORD(v117) = 32;
                    break;
                  case 3:
                  case 9:
                    LODWORD(v117) = 64;
                    break;
                  case 4:
                    LODWORD(v117) = 80;
                    break;
                  case 5:
                  case 6:
                    LODWORD(v117) = 128;
                    break;
                  case 7:
                    v254 = **v119;
                    LODWORD(v117) = sub_15A9520(v122, 0);
                    v123 = v254;
                    LODWORD(v117) = 8 * v117;
                    break;
                  case 0xB:
                    LODWORD(v117) = *(_DWORD *)(v125 + 8) >> 8;
                    break;
                  case 0xD:
                    v257 = **v119;
                    v128 = (_QWORD *)sub_15A9930(v122, v125);
                    v123 = v257;
                    v117 = 8LL * *v128;
                    break;
                  case 0xE:
                    v227 = **v119;
                    v230 = *(_QWORD *)(v125 + 24);
                    v256 = *(_QWORD *)(v125 + 32);
                    v240 = (unsigned int)sub_15A9FE0(v122, v230);
                    v127 = sub_127FA20(v122, v230);
                    v123 = v227;
                    v117 = 8 * v256 * v240 * ((v240 + ((unsigned __int64)(v127 + 7) >> 3) - 1) / v240);
                    break;
                  case 0xF:
                    v255 = **v119;
                    LODWORD(v117) = sub_15A9520(v122, *(_DWORD *)(v125 + 8) >> 8);
                    v123 = v255;
                    LODWORD(v117) = 8 * v117;
                    break;
                }
                break;
              }
              v118 = v117 * v121;
              if ( !v118 )
                goto LABEL_167;
              if ( v319 / v118 * v118 != v319 )
                goto LABEL_167;
              v258 = (__int64 *)v123;
              v241 = sub_1643F10(v123);
              if ( !v241 )
                goto LABEL_167;
              v279 = v124;
              v259 = (__int64 **)sub_16463B0(v258, v319 / v118);
              if ( v124 * v234 % v118 )
              {
                v209 = sub_1599EF0(v308);
                v341.m128_u64[0] = (unsigned __int64)&v342;
                v341.m128_u64[1] = 0x1000000000LL;
                if ( v232 > 0x10uLL )
                {
                  v238 = v209;
                  sub_16CD150((__int64)&v341, &v342, v232, 8, v209, v210);
                  v209 = v238;
                }
                v341.m128_i32[2] = v232;
                v211 = (__int64 *)v341.m128_u64[0];
                v212 = (__int64 *)(v341.m128_u64[0] + v229);
                if ( v341.m128_u64[0] != v341.m128_u64[0] + v229 )
                {
                  do
                    *v211++ = v209;
                  while ( v212 != v211 );
                }
                v213 = v270 + v124;
                if ( v270 )
                {
                  v237 = v119;
                  v214 = 0;
                  v228 = a1;
                  v215 = v118;
                  v216 = v120;
                  do
                  {
                    v217 = (__int64 *)(v214 + v341.m128_u64[0]);
                    v214 += 8;
                    v218 = sub_15A0680((__int64)v308, v279++, 0);
                    *v217 = v218;
                  }
                  while ( v279 != v213 );
                  v120 = v216;
                  v119 = v237;
                  v118 = v215;
                  a1 = v228;
                }
                v219 = a1->m128i_i64[1];
                v220 = sub_1649960(v245);
                v333[0] = 773;
                v326[0] = v220;
                s2 = v326;
                v326[1] = v221;
                v332 = (__int64)".extract";
                v282 = sub_15A01B0((__int64 *)v341.m128_u64[0], v341.m128_u32[2]);
                v222 = sub_1599EF0(*(__int64 ***)v249);
                v249 = sub_17AF270(v219, (__int64)v249, v222, v282, (__int64 *)&s2);
                if ( (__m128i *)v341.m128_u64[0] != &v342 )
                  _libc_free(v341.m128_u64[0]);
                v279 = 0;
              }
              v129 = v279 / (v118 / v234);
              if ( v337 )
              {
                v130 = (v337 - 1) & (((unsigned int)v259 >> 9) ^ ((unsigned int)v259 >> 4));
                v131 = (__int64 ***)(v335 + 16LL * v130);
                v132 = *v131;
                if ( v259 == *v131 )
                {
LABEL_190:
                  if ( v131 != (__int64 ***)(v335 + 16LL * v337) )
                  {
                    v341.m128_u64[0] = (unsigned __int64)v259;
                    v133 = (unsigned __int8 *)sub_17B2900((__int64)&v334, (__int64 *)&v341)[1];
                    goto LABEL_192;
                  }
                }
                else
                {
                  v225 = 1;
                  while ( v132 != (__int64 **)-8LL )
                  {
                    v226 = v225 + 1;
                    v130 = (v337 - 1) & (v225 + v130);
                    v131 = (__int64 ***)(v335 + 16LL * v130);
                    v132 = *v131;
                    if ( v259 == *v131 )
                      goto LABEL_190;
                    v225 = v226;
                  }
                }
              }
              v204 = a1->m128i_i64[1];
              s2 = (void *)sub_1649960(v245);
              v332 = v205;
              v342.m128i_i16[0] = 773;
              v341.m128_u64[0] = (unsigned __int64)&s2;
              v341.m128_u64[1] = (unsigned __int64)".bc";
              v133 = sub_1708970(v204, 47, (__int64)v249, v259, (__int64 *)&v341);
              v341.m128_u64[0] = (unsigned __int64)v259;
              sub_17B2900((__int64)&v334, (__int64 *)&v341)[1] = v133;
LABEL_192:
              v280 = a1->m128i_i64[1];
              s2 = (void *)sub_1649960(v245);
              v342.m128i_i16[0] = 773;
              v332 = v134;
              v341.m128_u64[0] = (unsigned __int64)&s2;
              v341.m128_u64[1] = (unsigned __int64)".extract";
              v135 = sub_15A0680((__int64)v308, v129, 0);
              v136 = sub_17AF100(v280, (__int64)v133, v135, (__int64 *)&v341);
              v137 = v120[1];
              v281 = (__int64)v136;
              if ( v137 )
              {
                v138 = a1->m128i_i64[0];
                do
                {
                  v139 = sub_1648700(v137);
                  sub_170B990(v138, (__int64)v139);
                  v137 = *(_QWORD *)(v137 + 8);
                }
                while ( v137 );
                if ( v120 == (__int64 *)v281 )
                  v281 = sub_1599EF0((__int64 **)*v120);
                sub_164D160((__int64)v120, v281, v19, *(double *)v20.m128i_i64, a5, a6, v140, v141, a9, a10);
                v278 = v241;
              }
              else
              {
                v278 = v241;
              }
LABEL_167:
              if ( v306 == ++v119 )
              {
                v10 = v245;
                break;
              }
            }
          }
        }
        j___libc_free_0(v335);
        if ( s1 != v340 )
          _libc_free((unsigned __int64)s1);
        v76 = *(_BYTE *)(n + 16);
      }
    }
LABEL_105:
    if ( *((_BYTE *)v309 + 16) != 85 )
    {
      if ( v76 != 85 )
        goto LABEL_108;
      goto LABEL_107;
    }
    v152 = (__int64)*(v309 - 6);
    if ( v76 == 85 )
    {
      if ( *(_BYTE *)(v152 + 16) != 9 )
      {
LABEL_107:
        if ( *(_BYTE *)(*(_QWORD *)(n - 48) + 16LL) != 9 )
          goto LABEL_108;
        v153 = n;
        v154 = 0;
        v157 = 0;
        v264 = 0;
        v320 = 0;
        v156 = *(__int64 ****)(n - 72);
        v155 = n;
        v271 = *((_DWORD *)*v156 + 8);
        v307 = v309;
        goto LABEL_217;
      }
      if ( *(_BYTE *)(*(_QWORD *)(n - 48) + 16LL) == 9 )
      {
        v155 = n;
        v264 = (__int64)*(v309 - 6);
        v154 = (__int64 ***)*(v309 - 9);
        v320 = *((_DWORD *)*v154 + 8);
        v156 = *(__int64 ****)(n - 72);
        v271 = *((_DWORD *)*v156 + 8);
        goto LABEL_282;
      }
    }
    else if ( *(_BYTE *)(v152 + 16) != 9 && v76 != 9 )
    {
      goto LABEL_108;
    }
    v153 = (__int64)*(v309 - 6);
    v307 = (__int64 ***)*(v309 - 9);
    v320 = *((_DWORD *)*v307 + 8);
    if ( v76 == 9 )
    {
      v264 = (__int64)*(v309 - 6);
      v154 = (__int64 ***)*(v309 - 9);
      v155 = 0;
      v156 = 0;
      v271 = 0;
      v157 = v309;
      goto LABEL_217;
    }
    v264 = (__int64)*(v309 - 6);
    v154 = (__int64 ***)*(v309 - 9);
    v156 = 0;
    v155 = 0;
    v271 = 0;
LABEL_282:
    if ( v320 == (_DWORD)v294 )
    {
      v307 = v154;
      v157 = v309;
      v153 = n;
    }
    else
    {
      v153 = n;
      v307 = v309;
      v157 = v309;
    }
LABEL_217:
    v158 = v155 != 0 && (_DWORD)v294 == v271;
    if ( v158 )
      v153 = (__int64)v156;
    else
      v158 = v155 != 0;
    if ( v156 == v154 )
    {
      v153 = 0;
      if ( v309 != v156 )
      {
        v307 = v156;
LABEL_221:
        v242 = v155;
        s2 = v333;
        v332 = 0x1000000000LL;
        v334 = &v336;
        v335 = 0x1000000000LL;
        v341.m128_u64[1] = 0x1000000000LL;
        v341.m128_u64[0] = (unsigned __int64)&v342;
        v260 = v158;
        sub_15FAA20((unsigned __int8 *)*(v157 - 3), (__int64)&v341);
        sub_17ADD00((__int64)&s2, (char **)&v341, v159, v160, (int)&s2, v161);
        v162 = v260;
        v155 = v242;
        if ( (__m128i *)v341.m128_u64[0] != &v342 )
        {
          _libc_free(v341.m128_u64[0]);
          v155 = v242;
          v162 = v260;
        }
        v261 = v320;
        if ( !v162 )
          goto LABEL_230;
        if ( n == v153 )
          goto LABEL_227;
        goto LABEL_225;
      }
      v307 = v309;
LABEL_307:
      s2 = v333;
      v332 = 0x1000000000LL;
      v334 = &v336;
      v335 = 0x1000000000LL;
      if ( !v158 )
      {
        v261 = (int)v294;
LABEL_230:
        s1 = v340;
        v339 = 0x1000000000LL;
        if ( !v298 )
        {
          v341.m128_u64[1] = 0x1000000000LL;
          v341.m128_u64[0] = (unsigned __int64)&v342;
LABEL_268:
          if ( !v153 )
            v153 = sub_1599EF0(*v307);
          v192 = sub_15A01B0((__int64 *)v341.m128_u64[0], v341.m128_u32[2]);
          v327 = 257;
          v193 = (_QWORD *)v192;
          v194 = sub_1648A60(56, 3u);
          v318 = (__int64)v194;
          if ( v194 )
            sub_15FA660((__int64)v194, v307, v153, v193, (__int64)v326, 0);
          if ( (__m128i *)v341.m128_u64[0] != &v342 )
            _libc_free(v341.m128_u64[0]);
LABEL_274:
          if ( s1 != v340 )
            _libc_free((unsigned __int64)s1);
          if ( v334 != &v336 )
            _libc_free((unsigned __int64)v334);
          if ( s2 != v333 )
            _libc_free((unsigned __int64)s2);
          goto LABEL_24;
        }
        v168 = 16;
        v169 = 0;
        v170 = 0;
        v171 = n;
        v172 = &v341;
        na = (size_t)a1;
        v173 = 4LL * (v298 - 1);
        v174 = -1;
        v175 = 1;
        v243 = v10;
        v176 = (int)v294;
        while ( 1 )
        {
          LODWORD(v177) = -1;
          v178 = *(int *)((char *)v328 + v169);
          if ( (int)v178 < 0 )
            goto LABEL_242;
          if ( (int)v178 < v176 )
            break;
          if ( *(_BYTE *)(v171 + 16) == 9 )
            goto LABEL_242;
          v177 = (unsigned int)(v178 - v288);
          if ( v171 == v153 )
          {
            LODWORD(v178) = v178 - v288;
          }
          else
          {
            LODWORD(v178) = *((_DWORD *)v334 + v177);
            LODWORD(v177) = -1;
            if ( (int)v178 >= v271 )
              goto LABEL_242;
            LODWORD(v177) = v178;
          }
          if ( (int)v178 >= 0 )
          {
            if ( v153 && v307 != (__int64 ***)v153 )
            {
              LODWORD(v177) = v261 + v177;
              LODWORD(v178) = v177;
              goto LABEL_238;
            }
LABEL_239:
            if ( v174 == (_DWORD)v178 || v174 < 0 )
            {
              v174 = v178;
            }
            else
            {
              v174 = v178;
              v175 = 0;
            }
          }
LABEL_242:
          if ( v168 <= (unsigned int)v170 )
          {
            v231 = v172;
            v233 = v173;
            v235 = v171;
            v236 = v174;
            v295 = v175;
            sub_16CD150((__int64)&s1, v340, 0, 4, v173, v175);
            v170 = (unsigned int)v339;
            v172 = v231;
            v173 = v233;
            v171 = v235;
            v174 = v236;
            v175 = v295;
          }
          *((_DWORD *)s1 + v170) = v177;
          v170 = (unsigned int)(v339 + 1);
          LODWORD(v339) = v339 + 1;
          if ( v169 == v173 )
          {
            v179 = v172;
            v180 = (__int64 *)na;
            v181 = v170;
            if ( (_BYTE)v175 )
              goto LABEL_260;
            v182 = (unsigned int)v170;
            if ( (_DWORD)v332 == (_DWORD)v170 )
            {
              v296 = (unsigned int)v170;
              v195 = 4LL * (unsigned int)v170;
              if ( !(4 * v182) )
                goto LABEL_260;
              v184 = s1;
              v310 = v181;
              nb = 4 * v182;
              v196 = memcmp(s1, s2, v195);
              v181 = v310;
              if ( !v196 )
                goto LABEL_260;
              v182 = v296;
              v185 = nb;
              if ( v296 == (unsigned int)v335 )
              {
                v223 = v334;
                goto LABEL_339;
              }
              if ( (_DWORD)v329 != v310 )
                goto LABEL_294;
            }
            else
            {
              if ( (unsigned int)v170 != (unsigned __int64)(unsigned int)v335 )
              {
                if ( (unsigned int)v170 != (unsigned __int64)(unsigned int)v329 )
                  goto LABEL_294;
                v183 = v328;
                v184 = s1;
                v185 = 4LL * (unsigned int)v170;
                if ( !v185 )
                  goto LABEL_260;
LABEL_293:
                ne = v181;
                v197 = memcmp(v184, v183, v185);
                v181 = ne;
                if ( v197 )
                {
LABEL_294:
                  sub_17ADB80((__int64)&s1, v326, v179);
                  if ( LOBYTE(v326[0]) && v320 == v298 )
                  {
                    v318 = sub_170E100(
                             v180,
                             v243,
                             (__int64)v307,
                             v19,
                             *(double *)v20.m128i_i64,
                             a5,
                             a6,
                             v198,
                             v199,
                             a9,
                             a10);
                  }
                  else if ( v341.m128_i8[0] && v271 == v298 )
                  {
                    v318 = sub_170E100(v180, v243, v153, v19, *(double *)v20.m128i_i64, a5, a6, v198, v199, a9, a10);
                  }
                  else
                  {
                    v200 = 0;
                    if ( v278 )
                      v200 = v243;
                    v318 = v200;
                  }
                  goto LABEL_274;
                }
LABEL_260:
                v341.m128_u64[1] = 0x1000000000LL;
                v341.m128_u64[0] = (unsigned __int64)&v342;
                if ( !v181 )
                  goto LABEL_268;
                v321 = v153;
                v186 = 4LL * v181;
                v187 = 0;
                while ( 2 )
                {
                  v191 = *(_DWORD *)((char *)s1 + v187);
                  if ( v191 >= 0 )
                  {
                    v189 = sub_15A0680((__int64)v308, v191, 0);
                    v190 = v341.m128_u32[2];
                    if ( v341.m128_i32[2] >= (unsigned __int32)v341.m128_i32[3] )
                      goto LABEL_266;
                  }
                  else
                  {
                    v189 = sub_1599EF0(v308);
                    v190 = v341.m128_u32[2];
                    if ( v341.m128_i32[2] >= (unsigned __int32)v341.m128_i32[3] )
                    {
LABEL_266:
                      nd = v189;
                      sub_16CD150((__int64)v179, &v342, 0, 8, v188, v189);
                      v190 = v341.m128_u32[2];
                      v189 = nd;
                    }
                  }
                  v187 += 4;
                  *(_QWORD *)(v341.m128_u64[0] + 8 * v190) = v189;
                  ++v341.m128_i32[2];
                  if ( v186 == v187 )
                  {
                    v153 = v321;
                    goto LABEL_268;
                  }
                  continue;
                }
              }
              v223 = v334;
              v184 = s1;
              v185 = 4LL * (unsigned int)v170;
              if ( !(4 * v182) )
                goto LABEL_260;
LABEL_339:
              v311 = v181;
              v297 = v182;
              nc = v185;
              v224 = memcmp(v184, v223, v185);
              v181 = v311;
              if ( !v224 )
                goto LABEL_260;
              v185 = nc;
              if ( v297 != (unsigned int)v329 )
                goto LABEL_294;
            }
            v183 = v328;
            goto LABEL_293;
          }
          v168 = HIDWORD(v339);
          v169 += 4;
        }
        if ( v309 == v307 )
        {
          LODWORD(v177) = *(_DWORD *)((char *)v328 + v169);
        }
        else
        {
          LODWORD(v178) = *((_DWORD *)s2 + v178);
          if ( (int)v178 >= v320 && *(_BYTE *)(v264 + 16) == 9 )
            goto LABEL_242;
          LODWORD(v177) = v178;
LABEL_238:
          if ( (int)v178 < 0 )
            goto LABEL_242;
        }
        goto LABEL_239;
      }
LABEL_225:
      v341.m128_u64[0] = (unsigned __int64)&v342;
      v341.m128_u64[1] = 0x1000000000LL;
      sub_15FAA20(*(unsigned __int8 **)(v155 - 24), (__int64)&v341);
      sub_17ADD00((__int64)&v334, (char **)&v341, v163, v164, v165, v166);
      if ( (__m128i *)v341.m128_u64[0] != &v342 )
        _libc_free(v341.m128_u64[0]);
LABEL_227:
      v167 = (int)v294;
      if ( v309 != v307 )
        v167 = v320;
      v261 = v167;
      goto LABEL_230;
    }
    if ( v307 != v309 )
      goto LABEL_221;
    if ( v153 != n )
      goto LABEL_307;
LABEL_108:
    v318 = v10;
    if ( v278 )
      goto LABEL_24;
    goto LABEL_109;
  }
  sub_17ADB80((__int64)&v328, &s1, &v341);
  if ( (_BYTE)s1 )
  {
    v201 = *(_QWORD *)(v10 + 8);
    if ( !v201 )
      goto LABEL_109;
    v202 = a1->m128i_i64[0];
    do
    {
      v203 = sub_1648700(v201);
      sub_170B990(v202, (__int64)v203);
      v201 = *(_QWORD *)(v201 + 8);
    }
    while ( v201 );
LABEL_153:
    if ( (__int64 ***)v10 == v309 )
      v309 = (__int64 ***)sub_1599EF0(*(__int64 ***)v10);
    sub_164D160(v10, (__int64)v309, v19, *(double *)v20.m128i_i64, a5, a6, v105, v106, a9, a10);
    goto LABEL_24;
  }
  if ( !v341.m128_i8[0] )
    goto LABEL_97;
  v142 = *(_QWORD *)(v10 + 8);
  if ( v142 )
  {
    v143 = a1->m128i_i64[0];
    do
    {
      v144 = sub_1648700(v142);
      sub_170B990(v143, (__int64)v144);
      v142 = *(_QWORD *)(v142 + 8);
    }
    while ( v142 );
    if ( v10 == n )
      n = sub_1599EF0(*(__int64 ***)v10);
    sub_164D160(v10, n, v19, *(double *)v20.m128i_i64, a5, a6, v145, v146, a9, a10);
    goto LABEL_24;
  }
LABEL_109:
  v318 = 0;
LABEL_24:
  if ( v325 > 0x40 && v324 )
    j_j___libc_free_0_0(v324);
  if ( v323 > 0x40 && v322 )
    j_j___libc_free_0_0(v322);
LABEL_8:
  if ( v328 != v330 )
    _libc_free((unsigned __int64)v328);
  return v318;
}
