// Function: sub_1FC9570
// Address: 0x1fc9570
//
__int64 __fastcall sub_1FC9570(
        _QWORD *a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned int v9; // r15d
  __int64 v11; // rbx
  const __m128i *v12; // rax
  __m128 v13; // xmm0
  __m128i v14; // xmm1
  __int16 v15; // dx
  __int16 v16; // cx
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int32 v20; // r12d
  __int64 v21; // r15
  __int64 v22; // rax
  const __m128i *v23; // rax
  __int64 v24; // rdx
  char v25; // cl
  __int64 v26; // rcx
  __int64 result; // rax
  unsigned int v28; // r12d
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  unsigned __int8 v31; // r12
  __int64 v32; // r10
  unsigned __int8 *v33; // rcx
  __int64 v34; // r11
  __int64 v35; // rcx
  unsigned int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 v41; // rsi
  _QWORD *v42; // r12
  unsigned int v43; // r14d
  __int64 *v44; // r8
  bool v45; // al
  __int64 v46; // r10
  __int64 v47; // rax
  __int64 v48; // rax
  __int8 v49; // dl
  __int64 v50; // rax
  __int64 v52; // rax
  __int64 *v53; // r12
  unsigned int v54; // r15d
  int v55; // r8d
  int v56; // r9d
  unsigned __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r12
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rsi
  _QWORD *v65; // r13
  __int64 v66; // rcx
  __int64 v67; // r14
  __int64 v68; // r15
  int v69; // eax
  __int64 (*v70)(); // rax
  __int64 v71; // rax
  __int64 *v72; // r15
  __int64 v73; // rax
  __int64 v74; // rdx
  unsigned __int8 v75; // si
  _QWORD *v76; // r14
  __int64 v77; // rcx
  __int64 v78; // r12
  __int64 v79; // rsi
  __int64 v80; // r13
  const __m128i *v81; // rax
  unsigned int v82; // r15d
  __int32 v83; // r12d
  __int64 v84; // r13
  __int64 v85; // rcx
  __int64 v86; // rax
  char v87; // dl
  __int64 v88; // rax
  __int64 v89; // r15
  __int64 v90; // r13
  int v91; // r8d
  __int16 v92; // ax
  __int64 v93; // rax
  __int64 v94; // rdx
  char v95; // r13
  __int64 *v96; // rbx
  __int64 v97; // rax
  __m128 *v98; // rax
  __int64 v99; // r15
  __int64 **v100; // rax
  __int64 *v101; // r12
  __int64 *v102; // rax
  __int64 v103; // rdx
  __int64 v104; // r11
  _QWORD *v105; // r10
  __int64 v106; // rsi
  __int64 v107; // r8
  __int64 v108; // r9
  __int16 v109; // r15
  char v110; // al
  char v111; // al
  __int64 v112; // rbx
  __int64 v113; // r9
  unsigned int v114; // edx
  __int64 v115; // r10
  __int64 *v116; // r13
  __int64 v117; // rdx
  __int64 v118; // r12
  __int64 v119; // r13
  __int64 v120; // rsi
  __int64 v121; // rdx
  _QWORD *v122; // rdi
  __int64 *v123; // rax
  unsigned __int64 v124; // rdi
  __int64 v125; // rax
  __int64 v126; // rsi
  __int64 v127; // r13
  __int64 v128; // r15
  __int64 v129; // r15
  __int64 v130; // rdx
  __int8 v131; // al
  __int64 v132; // rdx
  __int64 v133; // r13
  __int64 v134; // rax
  unsigned int v135; // r12d
  unsigned __int64 v136; // rdx
  char v137; // r12
  __int64 v138; // rcx
  __int64 v139; // rax
  __int64 v140; // r8
  int v141; // edx
  unsigned int v142; // r12d
  __int64 v143; // rax
  __int64 v144; // r15
  char v145; // di
  __int64 v146; // rax
  __int64 v147; // r8
  __int64 v148; // rax
  int v149; // edx
  __int64 v150; // r15
  __int64 v151; // rax
  unsigned int v152; // r15d
  __int64 v153; // rax
  __int64 v154; // r8
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // rdx
  __int64 v158; // rax
  __int64 v159; // r8
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // rdx
  unsigned int *v163; // rax
  unsigned __int8 *v164; // rax
  __int64 v165; // rsi
  __int64 v166; // rdx
  unsigned int *v167; // rax
  __int64 v168; // r8
  __int64 v169; // rcx
  __int64 v170; // rax
  unsigned __int16 v171; // di
  unsigned int v172; // eax
  __int64 v173; // rcx
  const void **v174; // rdx
  const void **v175; // r15
  __int64 v176; // r13
  unsigned int v177; // edx
  const __m128i *v178; // rax
  unsigned int v179; // edx
  __int64 v180; // rdx
  __int64 v181; // r13
  __int128 v182; // rax
  unsigned int v183; // edx
  int v184; // r12d
  __int64 v185; // rdx
  char v186; // al
  __int64 v187; // rdx
  unsigned int v188; // eax
  int v189; // eax
  __int64 v190; // r8
  __int64 v191; // rdx
  unsigned int v192; // r12d
  __int64 v193; // rax
  unsigned int v194; // eax
  __int64 v195; // r11
  __int64 v196; // rax
  __int64 v197; // rdx
  __int64 v198; // r15
  __int64 v199; // rcx
  _QWORD *v200; // r11
  __int64 v201; // rdx
  __int64 v202; // rsi
  __int64 v203; // r9
  __int64 v204; // r10
  __int64 v205; // r12
  __int64 v206; // rcx
  __int64 v207; // rdi
  __int64 v208; // rax
  __int64 v209; // r15
  bool v210; // al
  __int64 v211; // rax
  int v212; // r12d
  __int64 v213; // rax
  __int64 v214; // r12
  unsigned int v215; // eax
  unsigned __int8 *v216; // rax
  unsigned __int64 v217; // rax
  int v218; // eax
  __int64 v219; // rsi
  unsigned int v220; // r12d
  const void **v221; // rdx
  __int64 v222; // rcx
  __int64 v223; // r8
  __int64 v224; // r9
  __int64 v225; // r9
  unsigned int v226; // ecx
  char v227; // r12
  int v228; // eax
  __int64 v229; // r15
  unsigned int v230; // eax
  __int64 v231; // rax
  const void **v232; // r8
  __int64 v233; // rcx
  unsigned __int8 *v234; // rdi
  __int64 v235; // rdx
  const void **v236; // r8
  __int64 v237; // rcx
  __int64 v238; // rdx
  __int64 v239; // rsi
  __int64 *v240; // r13
  unsigned __int64 v241; // rdx
  __int128 v242; // rax
  __int64 *v243; // rax
  __int64 v244; // rdx
  __int64 v245; // r13
  __int64 v246; // rax
  __int64 v247; // rsi
  __int64 v248; // rbx
  __int64 v249; // rdx
  __int64 v250; // rdi
  unsigned __int64 v251; // rax
  unsigned int v252; // [rsp-18h] [rbp-358h]
  __int128 v253; // [rsp-10h] [rbp-350h]
  int v254; // [rsp+0h] [rbp-340h]
  __int64 v255; // [rsp+0h] [rbp-340h]
  __int64 v256; // [rsp+8h] [rbp-338h]
  __int64 v257; // [rsp+8h] [rbp-338h]
  int v258; // [rsp+10h] [rbp-330h]
  __int64 v259; // [rsp+10h] [rbp-330h]
  __int64 v260; // [rsp+18h] [rbp-328h]
  char *v261; // [rsp+18h] [rbp-328h]
  __int64 v262; // [rsp+18h] [rbp-328h]
  __m128i v263; // [rsp+20h] [rbp-320h]
  __int64 v264; // [rsp+20h] [rbp-320h]
  __int64 *v265; // [rsp+20h] [rbp-320h]
  __int64 v266; // [rsp+30h] [rbp-310h]
  char v267; // [rsp+30h] [rbp-310h]
  __int64 v268; // [rsp+38h] [rbp-308h]
  __int64 v269; // [rsp+38h] [rbp-308h]
  _QWORD *v270; // [rsp+38h] [rbp-308h]
  _QWORD *v271; // [rsp+40h] [rbp-300h]
  __int64 v272; // [rsp+40h] [rbp-300h]
  __int64 v273; // [rsp+40h] [rbp-300h]
  __int64 v274; // [rsp+50h] [rbp-2F0h]
  __int64 v275; // [rsp+50h] [rbp-2F0h]
  __int64 v276; // [rsp+60h] [rbp-2E0h]
  __int64 v277; // [rsp+60h] [rbp-2E0h]
  unsigned int v278; // [rsp+60h] [rbp-2E0h]
  const void **v279; // [rsp+60h] [rbp-2E0h]
  __int128 v280; // [rsp+60h] [rbp-2E0h]
  __int64 v281; // [rsp+60h] [rbp-2E0h]
  __int128 v282; // [rsp+60h] [rbp-2E0h]
  __int64 v283; // [rsp+70h] [rbp-2D0h]
  __int64 v284; // [rsp+70h] [rbp-2D0h]
  int v285; // [rsp+78h] [rbp-2C8h]
  __int64 v286; // [rsp+78h] [rbp-2C8h]
  __int64 v287; // [rsp+78h] [rbp-2C8h]
  unsigned int v288; // [rsp+78h] [rbp-2C8h]
  __int64 v289; // [rsp+78h] [rbp-2C8h]
  __int64 v290; // [rsp+80h] [rbp-2C0h]
  unsigned int v291; // [rsp+80h] [rbp-2C0h]
  unsigned int v292; // [rsp+80h] [rbp-2C0h]
  unsigned int v293; // [rsp+80h] [rbp-2C0h]
  unsigned __int32 v294; // [rsp+80h] [rbp-2C0h]
  __int64 v295; // [rsp+80h] [rbp-2C0h]
  __int64 v296; // [rsp+80h] [rbp-2C0h]
  _QWORD *v297; // [rsp+80h] [rbp-2C0h]
  _QWORD *v298; // [rsp+80h] [rbp-2C0h]
  unsigned __int64 v299; // [rsp+88h] [rbp-2B8h]
  __int64 v300; // [rsp+90h] [rbp-2B0h]
  __int64 v301; // [rsp+90h] [rbp-2B0h]
  __int64 v302; // [rsp+90h] [rbp-2B0h]
  __int64 v303; // [rsp+90h] [rbp-2B0h]
  unsigned int v304; // [rsp+90h] [rbp-2B0h]
  unsigned int v305; // [rsp+90h] [rbp-2B0h]
  __int64 v306; // [rsp+90h] [rbp-2B0h]
  _QWORD *v307; // [rsp+90h] [rbp-2B0h]
  __int64 v308; // [rsp+90h] [rbp-2B0h]
  const void **v309; // [rsp+90h] [rbp-2B0h]
  unsigned __int16 v310; // [rsp+90h] [rbp-2B0h]
  __int64 v311; // [rsp+90h] [rbp-2B0h]
  __int64 v312; // [rsp+98h] [rbp-2A8h]
  unsigned int v313; // [rsp+A0h] [rbp-2A0h]
  __int64 v314; // [rsp+A0h] [rbp-2A0h]
  __int64 *v315; // [rsp+A0h] [rbp-2A0h]
  __int64 v316; // [rsp+A0h] [rbp-2A0h]
  __int64 *v317; // [rsp+A0h] [rbp-2A0h]
  unsigned int v318; // [rsp+A0h] [rbp-2A0h]
  __int64 *v319; // [rsp+A0h] [rbp-2A0h]
  unsigned int v320; // [rsp+A0h] [rbp-2A0h]
  __int64 v321; // [rsp+A0h] [rbp-2A0h]
  __int64 v322; // [rsp+A0h] [rbp-2A0h]
  __int64 v323; // [rsp+A0h] [rbp-2A0h]
  unsigned int v324; // [rsp+A0h] [rbp-2A0h]
  __int64 v325; // [rsp+A0h] [rbp-2A0h]
  __int64 *v326; // [rsp+A0h] [rbp-2A0h]
  __int64 *v327; // [rsp+A0h] [rbp-2A0h]
  __int64 v328; // [rsp+A8h] [rbp-298h]
  __int64 v329; // [rsp+B0h] [rbp-290h]
  __int64 v330; // [rsp+B0h] [rbp-290h]
  _QWORD *v331; // [rsp+B0h] [rbp-290h]
  unsigned int v332; // [rsp+C0h] [rbp-280h]
  __int64 v333; // [rsp+C0h] [rbp-280h]
  __int64 v334; // [rsp+C0h] [rbp-280h]
  __int64 v335; // [rsp+C0h] [rbp-280h]
  unsigned int v336; // [rsp+C0h] [rbp-280h]
  __int64 v337; // [rsp+C0h] [rbp-280h]
  __int64 v338; // [rsp+C8h] [rbp-278h]
  unsigned int v339; // [rsp+C8h] [rbp-278h]
  int v340; // [rsp+C8h] [rbp-278h]
  _QWORD *v341; // [rsp+C8h] [rbp-278h]
  __int64 v342; // [rsp+C8h] [rbp-278h]
  __int64 v343; // [rsp+C8h] [rbp-278h]
  __int64 v344; // [rsp+C8h] [rbp-278h]
  __int64 v345; // [rsp+C8h] [rbp-278h]
  int v346; // [rsp+C8h] [rbp-278h]
  __int64 v347; // [rsp+C8h] [rbp-278h]
  __int64 v348; // [rsp+C8h] [rbp-278h]
  __int64 *v349; // [rsp+C8h] [rbp-278h]
  __int64 *v350; // [rsp+D0h] [rbp-270h]
  __int64 v351; // [rsp+D0h] [rbp-270h]
  __int64 v352; // [rsp+D0h] [rbp-270h]
  __int64 v353; // [rsp+D0h] [rbp-270h]
  __int64 v354; // [rsp+D0h] [rbp-270h]
  __int64 v355; // [rsp+D0h] [rbp-270h]
  unsigned int v356; // [rsp+D0h] [rbp-270h]
  __m128i v357; // [rsp+D0h] [rbp-270h]
  __int64 v358; // [rsp+D0h] [rbp-270h]
  __int64 v359; // [rsp+F0h] [rbp-250h]
  __m128 v360; // [rsp+130h] [rbp-210h] BYREF
  unsigned int v361; // [rsp+140h] [rbp-200h] BYREF
  __int64 v362; // [rsp+148h] [rbp-1F8h]
  __int64 v363; // [rsp+150h] [rbp-1F0h] BYREF
  unsigned int v364; // [rsp+158h] [rbp-1E8h]
  unsigned int v365; // [rsp+160h] [rbp-1E0h] BYREF
  const void **v366; // [rsp+168h] [rbp-1D8h]
  __int64 v367[2]; // [rsp+170h] [rbp-1D0h] BYREF
  __int64 v368; // [rsp+180h] [rbp-1C0h] BYREF
  unsigned int v369; // [rsp+188h] [rbp-1B8h]
  __int64 v370; // [rsp+190h] [rbp-1B0h] BYREF
  int v371; // [rsp+198h] [rbp-1A8h]
  __m128i v372; // [rsp+1A0h] [rbp-1A0h] BYREF
  unsigned __int64 v373; // [rsp+1B0h] [rbp-190h]
  __int128 v374; // [rsp+1C0h] [rbp-180h] BYREF
  __int64 v375; // [rsp+1D0h] [rbp-170h]
  __int128 v376; // [rsp+1F0h] [rbp-150h] BYREF
  __int64 v377[8]; // [rsp+200h] [rbp-140h] BYREF
  __m128i v378; // [rsp+240h] [rbp-100h] BYREF
  __int64 v379; // [rsp+250h] [rbp-F0h] BYREF
  _QWORD *v380; // [rsp+258h] [rbp-E8h]

  v11 = a2;
  v12 = *(const __m128i **)(a2 + 32);
  v13 = (__m128)_mm_loadu_si128(v12);
  v14 = _mm_loadu_si128(v12 + 5);
  v15 = (*(_WORD *)(a2 + 26) >> 7) & 7;
  v276 = v12[2].m128i_i64[1];
  v283 = v12[3].m128i_i64[0];
  v16 = *(_WORD *)(v276 + 24);
  v332 = v12[3].m128i_u32[0];
  v17 = v12[5].m128i_i64[0];
  v18 = v12[5].m128i_u32[2];
  v360 = v13;
  v290 = v17;
  v285 = v18;
  if ( v16 == 158 )
  {
    if ( (*(_BYTE *)(v11 + 27) & 4) != 0 )
      goto LABEL_3;
    if ( (_BYTE)v15 )
      goto LABEL_5;
    v29 = (_QWORD *)a1[1];
    v30 = *(_QWORD *)(**(_QWORD **)(v276 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v276 + 32) + 8LL);
    v31 = *(_BYTE *)v30;
    if ( *((_BYTE *)a1 + 24) || (*(_BYTE *)(v11 + 26) & 8) != 0 )
    {
      if ( (v18 = 1, v31 != 1) && (!v31 || (a8 = v31, v18 = v31, !v29[v31 + 15]))
        || (a8 = 129 * v18, v18 *= 259, (*((_BYTE *)v29 + v18 + 2608) & 0xFB) != 0) )
      {
LABEL_77:
        if ( !*((_DWORD *)a1 + 5) )
          goto LABEL_5;
        goto LABEL_30;
      }
    }
    v32 = *(_QWORD *)(v30 + 8);
    v33 = (unsigned __int8 *)(*(_QWORD *)(v276 + 40) + 16LL * v332);
    a9 = *(_QWORD *)(*v29 + 120LL);
    a8 = *v33;
    v34 = *((_QWORD *)v33 + 1);
    v18 = a8;
    if ( (__int64 (__fastcall *)(_QWORD *, unsigned __int8, __int64, char))a9 == sub_1F3D4A0 )
    {
      a9 = *(_QWORD *)(*v29 + 112LL);
      v35 = v31;
      if ( (__int64 (__fastcall *)(__int64, unsigned __int8, __int64, char))a9 == sub_1F3CFB0 )
      {
        if ( (_BYTE)a8 && v31 && *((_BYTE *)v29 + 259 * (unsigned __int8)a8 + 2607) == 1 )
        {
          v138 = v29[9258];
          if ( !v138 )
            goto LABEL_337;
          a9 = (__int64)(v29 + 9257);
          do
          {
            v18 = *(unsigned int *)(v138 + 32);
            if ( (unsigned int)v18 <= 0xB8 || (_DWORD)v18 == 185 && (unsigned __int8)a8 > *(_BYTE *)(v138 + 36) )
            {
              v138 = *(_QWORD *)(v138 + 24);
            }
            else
            {
              a9 = v138;
              v138 = *(_QWORD *)(v138 + 16);
            }
          }
          while ( v138 );
          if ( (_QWORD *)a9 == v29 + 9257
            || *(_DWORD *)(a9 + 32) > 0xB9u
            || *(_DWORD *)(a9 + 32) == 185 && (unsigned __int8)a8 < *(_BYTE *)(a9 + 36) )
          {
LABEL_337:
            while ( 1 )
            {
              do
                LOBYTE(a8) = a8 + 1;
              while ( !(_BYTE)a8 );
              if ( v29[(unsigned __int8)a8 + 15] )
              {
                v18 = 129LL * (unsigned __int8)a8;
                if ( *((_BYTE *)v29 + 259 * (unsigned __int8)a8 + 2607) != 1 )
                  break;
              }
            }
          }
          else
          {
            a8 = *(unsigned __int8 *)(a9 + 40);
          }
          if ( v31 == (_BYTE)a8 )
            goto LABEL_77;
        }
        goto LABEL_41;
      }
    }
    else
    {
      v9 = v31;
      v35 = v31;
    }
    v342 = v32;
    v110 = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, __int64, __int64))a9)(v29, a8, v34, v35, v32);
    v32 = v342;
    if ( !v110 )
    {
LABEL_142:
      v16 = *(_WORD *)(v276 + 24);
      v15 = (*(_WORD *)(v11 + 26) >> 7) & 7;
      goto LABEL_2;
    }
LABEL_41:
    v300 = v32;
    LOBYTE(v9) = v31;
    v36 = sub_1E34390(*(_QWORD *)(v11 + 104));
    v37 = *(_QWORD *)(v11 + 104);
    v38 = a1[1];
    LOBYTE(v374) = 0;
    v339 = v36;
    v313 = sub_1E340A0(v37);
    v39 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
    v18 = *(_QWORD *)(*a1 + 48LL);
    if ( (unsigned __int8)sub_1F43CC0(v38, v18, v39, v9, v300, v313, v339, &v374) && (_BYTE)v374 )
    {
      v40 = *(_QWORD *)(v11 + 104);
      v41 = *(_QWORD *)(v11 + 72);
      v42 = (_QWORD *)*a1;
      v378 = _mm_loadu_si128((const __m128i *)(v40 + 40));
      v379 = *(_QWORD *)(v40 + 56);
      v43 = *(unsigned __int16 *)(v40 + 32);
      v44 = *(__int64 **)(v276 + 32);
      *(_QWORD *)&v376 = v41;
      if ( v41 )
      {
        v350 = v44;
        sub_1623A60((__int64)&v376, v41, 2);
        v44 = v350;
      }
      DWORD2(v376) = *(_DWORD *)(v11 + 64);
      result = sub_1D2BF40(
                 v42,
                 v360.m128_i64[0],
                 v360.m128_i64[1],
                 (__int64)&v376,
                 *v44,
                 v44[1],
                 v14.m128i_i64[0],
                 v14.m128i_i64[1],
                 *(_OWORD *)v40,
                 *(_QWORD *)(v40 + 16),
                 v339,
                 v43,
                 (__int64)&v378);
      if ( (_QWORD)v376 )
      {
        v351 = result;
        sub_161E7C0((__int64)&v376, v376);
        return v351;
      }
      return result;
    }
    goto LABEL_142;
  }
LABEL_2:
  if ( v16 == 48 )
  {
    if ( !(_BYTE)v15 )
      return v360.m128_u64[0];
    goto LABEL_4;
  }
LABEL_3:
  if ( *((_DWORD *)a1 + 5) && !(_BYTE)v15 )
  {
LABEL_30:
    v18 = v14.m128i_i64[0];
    v28 = sub_1D1FC50(*a1, v14.m128i_i64[0]);
    if ( v28 )
    {
      if ( v28 > (unsigned int)sub_1E34390(*(_QWORD *)(v11 + 104)) )
      {
        v104 = *(_QWORD *)(v11 + 104);
        if ( !(*(_QWORD *)(v104 + 8) % (__int64)v28) )
        {
          v105 = (_QWORD *)*a1;
          v106 = *(_QWORD *)(v11 + 72);
          v107 = *(unsigned __int8 *)(v11 + 88);
          v378 = _mm_loadu_si128((const __m128i *)(v104 + 40));
          v108 = *(_QWORD *)(v11 + 96);
          v379 = *(_QWORD *)(v104 + 56);
          v109 = *(_WORD *)(v104 + 32);
          *(_QWORD *)&v376 = v106;
          if ( v106 )
          {
            v303 = v107;
            v312 = v108;
            v316 = v104;
            v341 = v105;
            sub_1623A60((__int64)&v376, v106, 2);
            v107 = v303;
            v108 = v312;
            v104 = v316;
            v105 = v341;
          }
          DWORD2(v376) = *(_DWORD *)(v11 + 64);
          sub_1D2C750(
            v105,
            v360.m128_i64[0],
            v360.m128_i64[1],
            (__int64)&v376,
            v276,
            v283,
            v14.m128i_i64[0],
            v14.m128i_i64[1],
            *(_OWORD *)v104,
            *(_QWORD *)(v104 + 16),
            v107,
            v108,
            v28,
            v109,
            (__int64)&v378);
          v18 = v376;
          if ( (_QWORD)v376 )
            sub_161E7C0((__int64)&v376, v376);
        }
      }
    }
    v15 = (*(_WORD *)(v11 + 26) >> 7) & 7;
  }
LABEL_4:
  v12 = *(const __m128i **)(v11 + 32);
LABEL_5:
  v19 = v12->m128i_i64[0];
  v20 = v12->m128i_i32[2];
  v21 = v12[2].m128i_i64[1];
  v338 = v12->m128i_i64[0];
  if ( *(_WORD *)(v11 + 24) != 186 || (*(_BYTE *)(v11 + 27) & 4) != 0 )
    goto LABEL_7;
  if ( (_BYTE)v15 )
    goto LABEL_8;
  if ( *(_WORD *)(v21 + 24) == 185 && (*(_BYTE *)(v21 + 27) & 0xC) == 0 && (*(_WORD *)(v21 + 26) & 0x380) == 0 )
  {
    v18 = 1;
    v45 = sub_1D18C00(v21, 1, v12[3].m128i_i32[0]);
    v19 = v338;
    v46 = v21;
    LOBYTE(v19) = v338 == v21;
    if ( v20 == 1 && v338 == v21 && v45 )
    {
      v184 = *(unsigned __int8 *)(v21 + 88);
      v185 = *(_QWORD *)(v21 + 96);
      v372.m128i_i8[0] = v184;
      v372.m128i_i64[1] = v185;
      if ( (_BYTE)v184 )
      {
        v19 = (unsigned int)(v184 - 86);
        LOBYTE(v19) = (unsigned __int8)(v184 - 86) <= 0x17u;
        v186 = v19 | ((unsigned __int8)(v184 - 8) <= 5u);
      }
      else
      {
        v323 = v185;
        v186 = sub_1F58CD0((__int64)&v372);
        v185 = v323;
        v46 = v21;
      }
      if ( v186
        && (_BYTE)v184 == *(_BYTE *)(v11 + 88)
        && (*(_QWORD *)(v11 + 96) == v185 || (_BYTE)v184)
        && (*(_BYTE *)(v46 + 26) & 0x10) == 0
        && (*(_BYTE *)(v11 + 26) & 0x10) == 0 )
      {
        v347 = v46;
        if ( !(unsigned int)sub_1E340A0(*(_QWORD *)(v46 + 104)) && !(unsigned int)sub_1E340A0(*(_QWORD *)(v11 + 104)) )
        {
          v188 = sub_1D159A0(v372.m128i_i8, 1, v187, v19, a8, a9, v254, v256, v258, v260);
          v189 = sub_1F7DE30(*(_QWORD **)(*a1 + 48LL), v188);
          v190 = a1[1];
          v18 = 185;
          *((_QWORD *)&v374 + 1) = v191;
          LODWORD(v374) = v189;
          if ( sub_1F6C830(v190, 0xB9u, v189) )
          {
            v18 = 186;
            if ( sub_1F6C830(a8, 0xBAu, a9) )
            {
              v18 = 185;
              if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a8 + 1144LL))(
                     a8,
                     185,
                     v372.m128i_u32[0],
                     v372.m128i_i64[1]) )
              {
                v18 = 186;
                if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(*(_QWORD *)a1[1] + 1144LL))(
                       a1[1],
                       186,
                       v372.m128i_u32[0],
                       v372.m128i_i64[1]) )
                {
                  v306 = v347;
                  v320 = sub_1E34390(*(_QWORD *)(v347 + 104));
                  v192 = sub_1E34390(*(_QWORD *)(v11 + 104));
                  v348 = sub_1F58E60((__int64)&v374, *(_QWORD **)(*a1 + 48LL));
                  v193 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
                  v18 = v348;
                  a8 = (unsigned int)sub_15A9FE0(v193, v348);
                  v194 = v320;
                  if ( v320 > v192 )
                    v194 = v192;
                  if ( (unsigned int)a8 <= v194 )
                  {
                    v195 = *a1;
                    v378 = 0u;
                    v379 = 0;
                    v270 = (_QWORD *)v195;
                    v272 = *(_QWORD *)(v306 + 104);
                    v349 = *(__int64 **)(v306 + 32);
                    sub_1F80610((__int64)&v376, v21);
                    v196 = sub_1D2B730(
                             v270,
                             (unsigned int)v374,
                             *((__int64 *)&v374 + 1),
                             (__int64)&v376,
                             *v349,
                             v349[1],
                             v349[5],
                             v349[6],
                             *(_OWORD *)v272,
                             *(_QWORD *)(v272 + 16),
                             v320,
                             0,
                             (__int64)&v378,
                             0);
                    v321 = v197;
                    v198 = v196;
                    sub_17CD270((__int64 *)&v376);
                    v199 = *(_QWORD *)(v11 + 72);
                    v200 = (_QWORD *)*a1;
                    v378 = 0u;
                    v201 = *(_QWORD *)(v11 + 104);
                    *(_QWORD *)&v376 = v199;
                    v202 = *(_QWORD *)(v11 + 32);
                    v203 = v321;
                    v204 = v306;
                    v379 = 0;
                    if ( v199 )
                    {
                      v264 = v321;
                      v273 = v201;
                      v307 = v200;
                      v322 = v204;
                      sub_1F6CA20((__int64 *)&v376);
                      v203 = v264;
                      v201 = v273;
                      v200 = v307;
                      v204 = v322;
                    }
                    DWORD2(v376) = *(_DWORD *)(v11 + 64);
                    v308 = v204;
                    v205 = sub_1D2BF40(
                             v200,
                             v198,
                             1,
                             (__int64)&v376,
                             v198,
                             v203,
                             *(_QWORD *)(v202 + 80),
                             *(_QWORD *)(v202 + 88),
                             *(_OWORD *)v201,
                             *(_QWORD *)(v201 + 16),
                             v192,
                             0,
                             (__int64)&v378);
                    sub_17CD270((__int64 *)&v376);
                    sub_1F81BC0((__int64)a1, v198);
                    sub_1F81BC0((__int64)a1, v205);
                    v206 = *(_QWORD *)(*a1 + 664LL);
                    v379 = *a1;
                    v18 = v308;
                    v378.m128i_i64[1] = v206;
                    *(_QWORD *)(v379 + 664) = &v378;
                    v207 = *a1;
                    v378.m128i_i64[0] = (__int64)off_49FFF30;
                    v380 = a1;
                    sub_1D44C70(v207, v308, 1, v198, 1u);
                    v19 = v378.m128i_i64[1];
                    *(_QWORD *)(v379 + 664) = v378.m128i_i64[1];
                    if ( v205 )
                      return v205;
                  }
                }
              }
            }
          }
        }
      }
    }
    v15 = (*(_WORD *)(v11 + 26) >> 7) & 7;
LABEL_7:
    if ( (_BYTE)v15 )
      goto LABEL_8;
  }
  if ( *((_DWORD *)a1 + 5) )
  {
    v18 = v11;
    sub_2043720(&v374, v11, *a1);
    if ( (_QWORD)v374 )
    {
      if ( *(_WORD *)(v374 + 24) != 48 )
      {
        v377[0] = v11;
        v90 = v11;
        *(_QWORD *)&v376 = v377;
        *((_QWORD *)&v376 + 1) = 0x800000001LL;
        while ( (*(_BYTE *)(v90 + 26) & 8) == 0 )
        {
          if ( (*(_WORD *)(v90 + 26) & 0x380) != 0 )
            break;
          sub_2043720(&v378, v90, *a1);
          v18 = (__int64)&v378;
          if ( !(unsigned __int8)sub_2043540(&v374, &v378, *a1, &v372) )
            break;
          while ( 1 )
          {
            v90 = **(_QWORD **)(v90 + 32);
            v92 = *(_WORD *)(v90 + 24);
            if ( v92 == 186 )
              break;
            if ( v92 != 185 )
              goto LABEL_180;
          }
          v93 = DWORD2(v376);
          if ( (*(_BYTE *)(v90 + 26) & 8) == 0 && (*(_WORD *)(v90 + 26) & 0x380) == 0 )
          {
            if ( HIDWORD(v376) <= DWORD2(v376) )
            {
              v18 = (__int64)v377;
              sub_16CD150((__int64)&v376, v377, 0, 8, v91, a9);
              v93 = DWORD2(v376);
            }
            *(_QWORD *)(v376 + 8 * v93) = v90;
            v93 = (unsigned int)++DWORD2(v376);
            if ( v11 == v90 )
              continue;
            v94 = *(_QWORD *)(v90 + 48);
            if ( v94 )
            {
              if ( !*(_QWORD *)(v94 + 32) )
                continue;
            }
          }
          goto LABEL_120;
        }
LABEL_180:
        v93 = DWORD2(v376);
LABEL_120:
        a8 = v376;
        v19 = 0x800000000LL;
        v378.m128i_i64[0] = (__int64)&v379;
        v378.m128i_i64[1] = 0x800000000LL;
        v315 = (__int64 *)(v376 + 8 * v93);
        if ( (__int64 *)v376 != v315 )
        {
          v302 = v11;
          v95 = 0;
          v96 = (__int64 *)v376;
          do
          {
            v99 = *v96;
            v100 = *(__int64 ***)(*v96 + 32);
            v101 = *v100;
            v340 = *((_DWORD *)v100 + 2);
            v102 = sub_1F71D20(a1, *v96, *v100, v100[1], (_DWORD *)a8, v13, *(double *)v14.m128i_i64, a5);
            if ( v101 != v102 || v340 != (_DWORD)v103 )
            {
              v372.m128i_i64[1] = (__int64)v102;
              v372.m128i_i64[0] = v99;
              if ( v302 == v99 )
                v95 = 1;
              v97 = v378.m128i_u32[2];
              v373 = v103;
              if ( v378.m128i_i32[2] >= (unsigned __int32)v378.m128i_i32[3] )
              {
                sub_16CD150((__int64)&v378, &v379, 0, 24, a8, a9);
                v97 = v378.m128i_u32[2];
              }
              a5 = _mm_loadu_si128(&v372);
              v98 = (__m128 *)(v378.m128i_i64[0] + 24 * v97);
              *v98 = (__m128)a5;
              v98[1].m128_u64[0] = v373;
              ++v378.m128i_i32[2];
            }
            ++v96;
          }
          while ( v315 != v96 );
          v267 = v95;
          v11 = v302;
          v18 = v378.m128i_i64[0] + 24LL * v378.m128i_u32[2];
          v317 = (__int64 *)v18;
          if ( v378.m128i_i64[0] != v18 )
          {
            v112 = v378.m128i_i64[0];
            do
            {
              v118 = *(_QWORD *)v112;
              v119 = *(unsigned int *)(v112 + 16);
              v120 = *(_QWORD *)(*(_QWORD *)v112 + 72LL);
              v344 = *(_QWORD *)(v112 + 8);
              v370 = v120;
              if ( v120 )
                sub_1623A60((__int64)&v370, v120, 2);
              v371 = *(_DWORD *)(v118 + 64);
              v121 = *(_QWORD *)(v118 + 104);
              v122 = (_QWORD *)*a1;
              v123 = *(__int64 **)(v118 + 32);
              if ( (*(_BYTE *)(v118 + 27) & 4) != 0 )
              {
                v113 = v274;
                LOBYTE(v113) = *(_BYTE *)(v118 + 88);
                v274 = v113;
                v115 = sub_1D2C2D0(
                         v122,
                         v344,
                         v119,
                         (__int64)&v370,
                         v123[5],
                         v123[6],
                         v123[10],
                         v123[11],
                         v113,
                         *(_QWORD *)(v118 + 96),
                         v121);
              }
              else
              {
                v115 = sub_1D2BB40(v122, v344, v119, (__int64)&v370, v123[5], v123[6], v123[10], v123[11], v121);
              }
              *((_QWORD *)&v253 + 1) = v114;
              *(_QWORD *)&v253 = v115;
              v116 = sub_1D332F0(
                       (__int64 *)*a1,
                       2,
                       (__int64)&v370,
                       1,
                       0,
                       0,
                       *(double *)v13.m128_u64,
                       *(double *)v14.m128i_i64,
                       a5,
                       **(_QWORD **)(v118 + 32),
                       *(_QWORD *)(*(_QWORD *)(v118 + 32) + 8LL),
                       v253);
              v343 = v117;
              sub_1F81BC0((__int64)a1, (__int64)v116);
              v372.m128i_i64[0] = (__int64)v116;
              v372.m128i_i64[1] = v343;
              sub_1F994A0((__int64)a1, v118, v372.m128i_i64, 1, 0);
              v18 = v370;
              if ( v370 )
                sub_161E7C0((__int64)&v370, v370);
              v112 += 24;
            }
            while ( v317 != (__int64 *)v112 );
            v11 = v302;
            v317 = (__int64 *)v378.m128i_i64[0];
          }
          v19 = (__int64)&v379;
          if ( v317 != &v379 )
            _libc_free((unsigned __int64)v317);
          v124 = v376;
          if ( (__int64 *)v376 == v377 )
          {
LABEL_167:
            if ( v267 )
              return v11;
            goto LABEL_55;
          }
LABEL_166:
          _libc_free(v124);
          goto LABEL_167;
        }
        v267 = 0;
        v124 = v376;
        if ( (__int64 *)v376 != v377 )
          goto LABEL_166;
      }
    }
  }
LABEL_55:
  v47 = *(_QWORD *)(v11 + 32);
  v360.m128_u64[0] = *(_QWORD *)v47;
  v360.m128_i32[2] = *(_DWORD *)(v47 + 8);
  if ( (*(_BYTE *)(v11 + 27) & 4) != 0 && (*(_WORD *)(v11 + 26) & 0x380) == 0 )
  {
    v48 = *(_QWORD *)(v276 + 40) + 16LL * v332;
    v49 = *(_BYTE *)v48;
    v50 = *(_QWORD *)(v48 + 8);
    v378.m128i_i8[0] = v49;
    v378.m128i_i64[1] = v50;
    if ( v49 ? (unsigned __int8)(v49 - 14) <= 0x47u || (unsigned __int8)(v49 - 2) <= 5u : sub_1F58CF0((__int64)&v378) )
    {
      v52 = *(_QWORD *)(v11 + 96);
      v53 = (__int64 *)*a1;
      LOBYTE(v376) = *(_BYTE *)(v11 + 88);
      *((_QWORD *)&v376 + 1) = v52;
      v54 = sub_1D159C0((__int64)&v376, v18, (unsigned __int8)v376, v19, a8, a9);
      v378.m128i_i32[2] = sub_1F701D0(v276, v332);
      if ( v378.m128i_i32[2] > 0x40u )
        sub_16A4EF0((__int64)&v378, 0, 0);
      else
        v378.m128i_i64[0] = 0;
      if ( v54 )
      {
        if ( v54 > 0x40 )
        {
          sub_16A5260(&v378, 0, v54);
        }
        else
        {
          v57 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v54);
          if ( v378.m128i_i32[2] > 0x40u )
            *(_QWORD *)v378.m128i_i64[0] |= v57;
          else
            v378.m128i_i64[0] |= v57;
        }
      }
      v58 = sub_1D3CC80(v53, v276, v283, (__int64)&v378, v55, v56, (__m128i)v13, *(double *)v14.m128i_i64, a5);
      v314 = v59;
      v60 = v58;
      if ( v378.m128i_i32[2] > 0x40u && v378.m128i_i64[0] )
        j_j___libc_free_0_0(v378.m128i_i64[0]);
      sub_1F81BC0((__int64)a1, v276);
      if ( v60 )
      {
        v64 = *(_QWORD *)(v11 + 72);
        v65 = (_QWORD *)*a1;
        v66 = *(_QWORD *)(v11 + 104);
        v67 = *(unsigned __int8 *)(v11 + 88);
        v378.m128i_i64[0] = v64;
        v68 = *(_QWORD *)(v11 + 96);
        if ( v64 )
        {
          v352 = v66;
          sub_1623A60((__int64)&v378, v64, 2);
          v66 = v352;
        }
        v378.m128i_i32[2] = *(_DWORD *)(v11 + 64);
        result = sub_1D2C2D0(
                   v65,
                   v360.m128_i64[0],
                   v360.m128_i64[1],
                   (__int64)&v378,
                   v60,
                   v314,
                   v14.m128i_i64[0],
                   v14.m128i_i64[1],
                   v67,
                   v68,
                   v66);
        goto LABEL_73;
      }
      v134 = *(_QWORD *)(v11 + 96);
      LOBYTE(v376) = *(_BYTE *)(v11 + 88);
      *((_QWORD *)&v376 + 1) = v134;
      v135 = sub_1D159C0((__int64)&v376, v276, (unsigned __int8)v376, v61, v62, v63);
      v378.m128i_i32[2] = sub_1F701D0(v276, v332);
      if ( v378.m128i_i32[2] > 0x40u )
        sub_16A4EF0((__int64)&v378, 0, 0);
      else
        v378.m128i_i64[0] = 0;
      if ( v135 )
      {
        if ( v135 > 0x40 )
        {
          sub_16A5260(&v378, 0, v135);
        }
        else
        {
          v136 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v135);
          if ( v378.m128i_i32[2] > 0x40u )
            *(_QWORD *)v378.m128i_i64[0] |= v136;
          else
            v378.m128i_i64[0] |= v136;
        }
      }
      v137 = sub_1FB1C90((__int64)a1, v276, v283, (int)&v378);
      if ( v378.m128i_i32[2] > 0x40u && v378.m128i_i64[0] )
        j_j___libc_free_0_0(v378.m128i_i64[0]);
      if ( v137 )
      {
        if ( *(_WORD *)(v11 + 24) )
          sub_1F81BC0((__int64)a1, v11);
        return v11;
      }
    }
  }
LABEL_8:
  if ( *(_WORD *)(v276 + 24) == 185 )
  {
    v22 = *(_QWORD *)(v276 + 32);
    if ( v290 == *(_QWORD *)(v22 + 40) && *(_DWORD *)(v22 + 48) == v285 )
    {
      v111 = *(_BYTE *)(v11 + 88);
      if ( v111 == *(_BYTE *)(v276 + 88)
        && (*(_QWORD *)(v11 + 96) == *(_QWORD *)(v276 + 96) || v111)
        && (*(_WORD *)(v11 + 26) & 0x380) == 0
        && (*(_BYTE *)(v11 + 26) & 8) == 0
        && sub_1D18E30(&v360, v276, 1, 2) )
      {
        return v360.m128_u64[0];
      }
    }
  }
  if ( *(_WORD *)(v360.m128_u64[0] + 24) == 186
    && (*(_WORD *)(v11 + 26) & 0x380) == 0
    && (*(_BYTE *)(v11 + 26) & 8) == 0
    && (*(_WORD *)(v360.m128_u64[0] + 26) & 0x380) == 0
    && (*(_BYTE *)(v360.m128_u64[0] + 26) & 8) == 0 )
  {
    v23 = *(const __m128i **)(v360.m128_u64[0] + 32);
    v24 = v23[5].m128i_i64[0];
    if ( v290 == v24 && v23[5].m128i_i32[2] == v285 )
    {
      v25 = *(_BYTE *)(v11 + 88);
      if ( *(_BYTE *)(v360.m128_u64[0] + 88) == v25
        && (*(_QWORD *)(v360.m128_u64[0] + 96) == *(_QWORD *)(v11 + 96) || v25) )
      {
        if ( v23[2].m128i_i64[1] == v276 && v23[3].m128i_i32[0] == v332 )
          return v360.m128_u64[0];
        if ( *((_DWORD *)a1 + 5) )
        {
          v26 = *(_QWORD *)(v360.m128_u64[0] + 48);
          if ( v26 )
          {
            if ( !*(_QWORD *)(v26 + 32) && *(_WORD *)(v24 + 24) != 48 )
            {
              v378 = _mm_loadu_si128(v23);
              sub_1F994A0((__int64)a1, v360.m128_i64[0], v378.m128i_i64, 1, 1);
              return 0;
            }
          }
        }
      }
    }
  }
  v69 = *(unsigned __int16 *)(v276 + 24);
  if ( v69 == 145 || v69 == 154 )
  {
    v71 = *(_QWORD *)(v276 + 48);
    if ( v71 )
    {
      if ( !*(_QWORD *)(v71 + 32) && (*(_WORD *)(v11 + 26) & 0x380) == 0 )
      {
        v72 = *(__int64 **)(v276 + 32);
        v73 = *(unsigned __int8 *)(*(_QWORD *)(*v72 + 40) + 16LL * *((unsigned int *)v72 + 2));
        if ( (_BYTE)v73 )
        {
          v74 = a1[1];
          v75 = *(_BYTE *)(v11 + 88);
          if ( *(_QWORD *)(v74 + 8 * v73 + 120) )
          {
            if ( v75 && !*(_BYTE *)(v75 + 115LL * (unsigned __int8)v73 + v74 + 58658) )
            {
              v76 = (_QWORD *)*a1;
              v77 = *(_QWORD *)(v11 + 104);
              v78 = v75;
              v79 = *(_QWORD *)(v11 + 72);
              v80 = *(_QWORD *)(v11 + 96);
              v378.m128i_i64[0] = v79;
              if ( v79 )
              {
                v354 = v77;
                sub_1623A60((__int64)&v378, v79, 2);
                v77 = v354;
              }
              v378.m128i_i32[2] = *(_DWORD *)(v11 + 64);
              result = sub_1D2C2D0(
                         v76,
                         v360.m128_i64[0],
                         v360.m128_i64[1],
                         (__int64)&v378,
                         *v72,
                         v72[1],
                         v14.m128i_i64[0],
                         v14.m128i_i64[1],
                         v78,
                         v80,
                         v77);
LABEL_73:
              if ( v378.m128i_i64[0] )
              {
                v353 = result;
                sub_161E7C0((__int64)&v378, v378.m128i_i64[0]);
                return v353;
              }
              return result;
            }
          }
        }
      }
    }
  }
  if ( !*((_BYTE *)a1 + 25)
    || (v70 = *(__int64 (**)())(*(_QWORD *)a1[1] + 136LL), v70 == sub_1F3C9D0)
    || (unsigned __int8)v70() )
  {
    while ( (unsigned __int8)sub_1FC66B0(a1, v11, (__m128i)v13, v14, a5) )
    {
      if ( *(_WORD *)(v11 + 24) != 186 )
        return v11;
    }
  }
  if ( (unsigned __int8)sub_1F8F9B0(a1, v11, (__m128i)v13, *(double *)v14.m128i_i64, a5)
    || *((int *)a1 + 4) > 2 && (unsigned __int8)sub_1F81F00(a1, v11) )
  {
    return v11;
  }
  if ( !*((_DWORD *)a1 + 5) )
    goto LABEL_101;
  v125 = *(_QWORD *)(v11 + 32);
  v126 = *(_QWORD *)(v11 + 72);
  v127 = *(_QWORD *)(v125 + 40);
  v128 = *(unsigned int *)(v125 + 48);
  *(_QWORD *)&v374 = v126;
  if ( v126 )
    sub_1623A60((__int64)&v374, v126, 2);
  v129 = 16 * v128;
  DWORD2(v374) = *(_DWORD *)(v11 + 64);
  v130 = v129 + *(_QWORD *)(v127 + 40);
  v131 = *(_BYTE *)v130;
  v132 = *(_QWORD *)(v130 + 8);
  v378.m128i_i8[0] = v131;
  v378.m128i_i64[1] = v132;
  if ( v131 )
  {
    if ( (unsigned __int8)(v131 - 2) > 5u )
    {
LABEL_174:
      v133 = 0;
      goto LABEL_175;
    }
  }
  else if ( !sub_1F58D10((__int64)&v378) )
  {
    goto LABEL_174;
  }
  if ( *(_WORD *)(v127 + 24) != 119 )
    goto LABEL_174;
  v139 = *(_QWORD *)(v127 + 32);
  v140 = *(_QWORD *)v139;
  v141 = *(_DWORD *)(v139 + 8);
  v142 = *(_DWORD *)(v139 + 48);
  v355 = *(_QWORD *)(v139 + 40);
  if ( *(_WORD *)(*(_QWORD *)v139 + 24LL) != 122 )
  {
    if ( *(_WORD *)(*(_QWORD *)(v139 + 40) + 24LL) != 122 )
      goto LABEL_174;
    v140 = *(_QWORD *)(v139 + 40);
    v142 = *(_DWORD *)(v139 + 8);
    v141 = *(_DWORD *)(v139 + 48);
    v355 = *(_QWORD *)v139;
  }
  v143 = *(_QWORD *)(v140 + 32);
  v345 = v140;
  v329 = *(_QWORD *)v143;
  v304 = *(_DWORD *)(v143 + 8);
  if ( !sub_1D18C00(v140, 1, v141) )
    goto LABEL_174;
  v144 = *(_QWORD *)(v127 + 40) + v129;
  v145 = *(_BYTE *)v144;
  v146 = *(_QWORD *)(v144 + 8);
  v378.m128i_i8[0] = v145;
  v378.m128i_i64[1] = v146;
  if ( v145 )
  {
    v318 = sub_1F6C8D0(v145);
  }
  else
  {
    v215 = sub_1F58D40((__int64)&v378);
    v147 = v345;
    v318 = v215;
  }
  v148 = *(_QWORD *)(*(_QWORD *)(v147 + 32) + 40LL);
  v149 = *(unsigned __int16 *)(v148 + 24);
  if ( v149 != 32 && v149 != 10 )
    goto LABEL_174;
  v150 = *(_QWORD *)(v148 + 88);
  if ( *(_DWORD *)(v150 + 32) > 0x40u )
  {
    v346 = *(_DWORD *)(v150 + 32);
    if ( v346 - (unsigned int)sub_16A57B0(v150 + 24) > 0x40 )
      goto LABEL_174;
    v151 = **(_QWORD **)(v150 + 24);
  }
  else
  {
    v151 = *(_QWORD *)(v150 + 24);
  }
  v152 = v318 >> 1;
  if ( v318 >> 1 != v151 )
    goto LABEL_174;
  if ( *(_WORD *)(v355 + 24) != 143 )
    goto LABEL_174;
  if ( !sub_1D18C00(v355, 1, v142) )
    goto LABEL_174;
  v153 = *(_QWORD *)(v355 + 32);
  v154 = *(_QWORD *)v153;
  v155 = *(unsigned int *)(v153 + 8);
  v292 = v155;
  v156 = *(_QWORD *)(v154 + 40) + 16 * v155;
  v157 = *(_QWORD *)(v156 + 8);
  v333 = v154;
  v378.m128i_i8[0] = *(_BYTE *)v156;
  v378.m128i_i64[1] = v157;
  if ( !sub_1F7E0B0((__int64)&v378) )
    goto LABEL_174;
  if ( v152 < (unsigned int)sub_1F6D5D0(v333, v292) )
    goto LABEL_174;
  if ( *(_WORD *)(v329 + 24) != 143 )
    goto LABEL_174;
  if ( !sub_1D18C00(v329, 1, v304) )
    goto LABEL_174;
  v158 = *(_QWORD *)(v329 + 32);
  v159 = *(_QWORD *)v158;
  v160 = *(unsigned int *)(v158 + 8);
  v293 = v160;
  v161 = *(_QWORD *)(v159 + 40) + 16 * v160;
  v162 = *(_QWORD *)(v161 + 8);
  v334 = v159;
  LOBYTE(v376) = *(_BYTE *)v161;
  *((_QWORD *)&v376 + 1) = v162;
  if ( !sub_1F7E0B0((__int64)&v376) || v152 < (unsigned int)sub_1F6D5D0(v334, v293) )
    goto LABEL_174;
  v163 = *(unsigned int **)(v355 + 32);
  v164 = (unsigned __int8 *)(*(_WORD *)(*(_QWORD *)v163 + 24LL) == 158
                           ? *(_QWORD *)(*(_QWORD *)v163 + 40LL) + 16LL * v163[2]
                           : *(_QWORD *)(v355 + 40) + 16LL * v142);
  v165 = *v164;
  v166 = *((_QWORD *)v164 + 1);
  v167 = *(unsigned int **)(v329 + 32);
  if ( *(_WORD *)(*(_QWORD *)v167 + 24LL) == 158 )
  {
    v216 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v167 + 40LL) + 16LL * v167[2]);
    v168 = *((_QWORD *)v216 + 1);
    v169 = *v216;
  }
  else
  {
    v168 = *(_QWORD *)(*(_QWORD *)(v329 + 40) + 16LL * v304 + 8);
    v169 = *(unsigned __int8 *)(*(_QWORD *)(v329 + 40) + 16LL * v304);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[1] + 184LL))(
          a1[1],
          v165,
          v166,
          v169,
          v168) )
    goto LABEL_174;
  v305 = sub_1E34390(*(_QWORD *)(v11 + 104));
  v170 = *(_QWORD *)(v11 + 104);
  v171 = *(_WORD *)(v170 + 32);
  v378 = _mm_loadu_si128((const __m128i *)(v170 + 40));
  v379 = *(_QWORD *)(v170 + 56);
  v172 = sub_1F7DE30(*(_QWORD **)(*a1 + 48LL), v152);
  v173 = v355;
  v175 = v174;
  v356 = v172;
  v176 = sub_1D309E0(
           (__int64 *)*a1,
           143,
           (__int64)&v374,
           v172,
           v174,
           0,
           *(double *)v13.m128_u64,
           *(double *)v14.m128i_i64,
           *(double *)a5.m128i_i64,
           *(_OWORD *)*(_QWORD *)(v173 + 32));
  v286 = v177;
  v359 = sub_1D309E0(
           (__int64 *)*a1,
           143,
           (__int64)&v374,
           v356,
           v175,
           0,
           *(double *)v13.m128_u64,
           *(double *)v14.m128i_i64,
           *(double *)a5.m128i_i64,
           *(_OWORD *)*(_QWORD *)(v329 + 32));
  v178 = *(const __m128i **)(v11 + 32);
  v277 = v179;
  v275 = v178->m128i_i64[1];
  v330 = v178[5].m128i_i64[0];
  v294 = v178[5].m128i_u32[2];
  v357 = _mm_loadu_si128(v178 + 5);
  v269 = v178->m128i_i64[0];
  v271 = (_QWORD *)*a1;
  v252 = sub_1E34390(*(_QWORD *)(v11 + 104));
  v287 = sub_1D2BF40(
           v271,
           v269,
           v275,
           (__int64)&v374,
           v176,
           v286,
           v357.m128i_i64[0],
           v357.m128i_i64[1],
           *(_OWORD *)*(_QWORD *)(v11 + 104),
           *(_QWORD *)(*(_QWORD *)(v11 + 104) + 16LL),
           v252,
           v171,
           (__int64)&v378);
  LODWORD(v176) = v318;
  v335 = v180;
  v319 = (__int64 *)*a1;
  v181 = (unsigned int)v176 >> 4;
  *(_QWORD *)&v182 = sub_1D38BB0(
                       *a1,
                       v181,
                       (__int64)&v374,
                       *(unsigned __int8 *)(*(_QWORD *)(v330 + 40) + 16LL * v294),
                       *(const void ***)(*(_QWORD *)(v330 + 40) + 16LL * v294 + 8),
                       0,
                       (__m128i)v13,
                       *(double *)v14.m128i_i64,
                       a5,
                       0);
  v357.m128i_i64[0] = (__int64)sub_1D332F0(
                                 v319,
                                 52,
                                 (__int64)&v374,
                                 *(unsigned __int8 *)(*(_QWORD *)(v330 + 40) + 16LL * v294),
                                 *(const void ***)(*(_QWORD *)(v330 + 40) + 16LL * v294 + 8),
                                 0,
                                 *(double *)v13.m128_u64,
                                 *(double *)v14.m128i_i64,
                                 a5,
                                 v357.m128i_i64[0],
                                 v357.m128i_u64[1],
                                 v182);
  v357.m128i_i64[1] = v183 | v357.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v331 = (_QWORD *)*a1;
  sub_1F7DDA0((__int64)&v372, *(_QWORD *)(v11 + 104), v181);
  v133 = sub_1D2BF40(
           v331,
           v287,
           v335,
           (__int64)&v374,
           v359,
           v277,
           v357.m128i_i64[0],
           v357.m128i_i64[1],
           *(_OWORD *)&v372,
           v373,
           v305 >> 1,
           v171,
           (__int64)&v378);
LABEL_175:
  if ( (_QWORD)v374 )
    sub_161E7C0((__int64)&v374, v374);
  result = v133;
  if ( !v133 )
  {
LABEL_101:
    if ( (*(_BYTE *)(v11 + 26) & 8) != 0 )
      return 0;
    v81 = *(const __m128i **)(v11 + 32);
    v82 = v81[3].m128i_u32[0];
    v284 = v81->m128i_i64[0];
    v266 = v81->m128i_i64[1];
    v83 = v81->m128i_i32[2];
    v84 = v81[2].m128i_i64[1];
    v268 = v84;
    v85 = v81[5].m128i_i64[0];
    v263 = _mm_loadu_si128(v81 + 5);
    v259 = v81[3].m128i_i64[0];
    v291 = v81[5].m128i_u32[2];
    v86 = *(_QWORD *)(v84 + 40) + 16LL * v82;
    v301 = v85;
    v87 = *(_BYTE *)v86;
    v88 = *(_QWORD *)(v86 + 8);
    LOBYTE(v361) = v87;
    v362 = v88;
    if ( (*(_BYTE *)(v11 + 27) & 4) != 0 )
      return 0;
    if ( v87 )
    {
      if ( (unsigned __int8)(v87 - 14) <= 0x5Fu )
        return 0;
    }
    else if ( sub_1F58D20((__int64)&v361) )
    {
      return 0;
    }
    if ( sub_1D18C00(v84, 1, v82) )
    {
      v288 = *(unsigned __int16 *)(v84 + 24);
      if ( v288 == 119 )
      {
        v217 = sub_1F6D610(**(_QWORD **)(v84 + 32), *(_QWORD *)(*(_QWORD *)(v84 + 32) + 8LL), v301, v291, v284);
        if ( (_DWORD)v217 )
        {
          v89 = sub_1F729C0(
                  v217,
                  HIDWORD(v217),
                  *(_QWORD *)(*(_QWORD *)(v84 + 32) + 40LL),
                  *(_QWORD *)(*(_QWORD *)(v84 + 32) + 48LL),
                  v11,
                  (__int64)a1,
                  (__m128i)v13,
                  v14,
                  a5);
          if ( v89 )
            return v89;
        }
        v251 = sub_1F6D610(
                 *(_QWORD *)(*(_QWORD *)(v84 + 32) + 40LL),
                 *(_QWORD *)(*(_QWORD *)(v84 + 32) + 48LL),
                 v301,
                 v291,
                 v284);
        if ( (_DWORD)v251 )
        {
          v89 = sub_1F729C0(
                  v251,
                  HIDWORD(v251),
                  **(_QWORD **)(v84 + 32),
                  *(_QWORD *)(*(_QWORD *)(v84 + 32) + 8LL),
                  v11,
                  (__int64)a1,
                  (__m128i)v13,
                  v14,
                  a5);
          if ( v89 )
            return v89;
        }
      }
      else if ( v288 - 118 > 2 )
      {
        return 0;
      }
      v208 = *(_QWORD *)(v84 + 32);
      if ( *(_WORD *)(*(_QWORD *)(v208 + 40) + 24LL) == 10 )
      {
        v257 = *(_QWORD *)v208;
        v209 = *(_QWORD *)v208;
        v358 = *(_QWORD *)v208;
        v255 = *(_QWORD *)(v208 + 8);
        if ( *(_WORD *)(*(_QWORD *)v208 + 24LL) == 185
          && (*(_BYTE *)(v257 + 27) & 0xC) == 0
          && (*(_WORD *)(v257 + 26) & 0x380) == 0 )
        {
          v210 = sub_1D18C00(v257, 1, *(_DWORD *)(v208 + 8));
          if ( v284 == v209 && v83 == 1 && v210 )
          {
            v211 = *(_QWORD *)(v358 + 32);
            if ( v301 == *(_QWORD *)(v211 + 40) && v291 == *(_DWORD *)(v211 + 48) )
            {
              v212 = sub_1E340A0(*(_QWORD *)(v358 + 104));
              if ( v212 == (unsigned int)sub_1E340A0(*(_QWORD *)(v11 + 104)) )
              {
                v213 = *(_QWORD *)(v84 + 32);
                v214 = *(_QWORD *)(v213 + 40);
                v278 = sub_1F6D5D0(v214, *(_DWORD *)(v213 + 48));
                sub_13A38D0((__int64)&v363, *(_QWORD *)(v214 + 88) + 24LL);
                if ( v288 == 118 )
                {
                  sub_135E0D0((__int64)&v378, v278, -1, 1u);
                  if ( v364 > 0x40 )
                    sub_16A8F00(&v363, v378.m128i_i64);
                  else
                    v363 ^= v378.m128i_i64[0];
                  sub_135E100(v378.m128i_i64);
                }
                if ( !sub_13A38F0((__int64)&v363, 0) && !sub_1454FB0((__int64)&v363) )
                {
                  v324 = sub_1455870(&v363);
                  v218 = sub_1455840((__int64)&v363);
                  v219 = sub_1454B60(v278 - 1 - v218 - v324);
                  v220 = v219;
                  v365 = sub_1F7DE30(*(_QWORD **)(*a1 + 48LL), v219);
                  v366 = v221;
                  v261 = (char *)&v365;
                  while ( v278 > v220 )
                  {
                    if ( v220 == (((unsigned int)sub_1D159A0(
                                                   v261,
                                                   v219,
                                                   (__int64)v221,
                                                   v222,
                                                   v223,
                                                   v224,
                                                   v255,
                                                   v257,
                                                   v259,
                                                   (__int64)v261)
                                 + 7)
                                & 0xFFFFFFF8)
                      && sub_1F6C880(a1[1], v288, v365)
                      && (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, const void **))(*(_QWORD *)v225 + 928LL))(
                           v225,
                           v361,
                           v362,
                           v365,
                           v366) )
                    {
                      v336 = v220;
                      if ( v324 % v220 )
                        v324 = v220 * ((v324 + v220 - 1) / v220 - 1);
                      v226 = v324 + v220;
                      if ( v324 + v220 > v278 )
                        v226 = v278;
                      sub_18F4C10((__int64)v367, v278, v324, v226);
                      sub_13A38D0((__int64)&v376, (__int64)&v363);
                      sub_1F6C9C0((__int64 *)&v376, v367);
                      v378.m128i_i32[2] = DWORD2(v376);
                      DWORD2(v376) = 0;
                      v378.m128i_i64[0] = v376;
                      v227 = sub_1455820((__int64)&v378, &v363);
                      sub_135E100(v378.m128i_i64);
                      sub_135E100((__int64 *)&v376);
                      if ( !v227 )
                      {
                        sub_135E100(v367);
                        sub_135E100(&v363);
                        return 0;
                      }
                      sub_13A38D0((__int64)&v374, (__int64)&v363);
                      sub_1F6C9C0((__int64 *)&v374, v367);
                      v228 = DWORD2(v374);
                      DWORD2(v374) = 0;
                      DWORD2(v376) = v228;
                      *(_QWORD *)&v376 = v374;
                      sub_13A38D0((__int64)&v378, (__int64)&v376);
                      sub_17A2760((__int64)&v378, v324);
                      sub_16A5A50((__int64)&v368, v378.m128i_i64, v336);
                      sub_135E100(v378.m128i_i64);
                      sub_135E100((__int64 *)&v376);
                      sub_135E100((__int64 *)&v374);
                      if ( v288 == 118 )
                      {
                        sub_135E0D0((__int64)&v378, v336, -1, 1u);
                        if ( v369 > 0x40 )
                          sub_16A8F00(&v368, v378.m128i_i64);
                        else
                          v368 ^= v378.m128i_i64[0];
                        sub_135E100(v378.m128i_i64);
                      }
                      v229 = v324 >> 3;
                      if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL)) )
                        v229 = ((v278 + 7 - v336) >> 3) - v229;
                      v230 = sub_1E34390(*(_QWORD *)(v358 + 104));
                      v337 = (v229 | v230) & -(v229 | v230);
                      v325 = sub_1F58E60((__int64)v261, *(_QWORD **)(*a1 + 48LL));
                      v231 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
                      if ( (unsigned int)sub_15A9FE0(v231, v325) > (unsigned int)v337 )
                      {
                        v89 = 0;
                      }
                      else
                      {
                        v295 = 16LL * v291;
                        v232 = *(const void ***)(*(_QWORD *)(v301 + 40) + v295 + 8);
                        v233 = *(unsigned __int8 *)(*(_QWORD *)(v301 + 40) + v295);
                        v326 = (__int64 *)*a1;
                        v378.m128i_i64[0] = *(_QWORD *)(v358 + 72);
                        if ( v378.m128i_i64[0] )
                        {
                          v262 = v233;
                          v279 = v232;
                          sub_1F6CA20(v378.m128i_i64);
                          v233 = v262;
                          v232 = v279;
                        }
                        v378.m128i_i32[2] = *(_DWORD *)(v358 + 64);
                        *(_QWORD *)&v280 = sub_1D38BB0(
                                             (__int64)v326,
                                             v229,
                                             (__int64)&v378,
                                             v233,
                                             v232,
                                             0,
                                             (__m128i)v13,
                                             *(double *)v14.m128i_i64,
                                             a5,
                                             0);
                        v234 = (unsigned __int8 *)(*(_QWORD *)(v301 + 40) + v295);
                        *((_QWORD *)&v280 + 1) = v235;
                        v236 = (const void **)*((_QWORD *)v234 + 1);
                        v237 = *v234;
                        *(_QWORD *)&v376 = *(_QWORD *)(v358 + 72);
                        if ( (_QWORD)v376 )
                        {
                          v296 = v237;
                          v309 = v236;
                          sub_1F6CA20((__int64 *)&v376);
                          v237 = v296;
                          v236 = v309;
                        }
                        DWORD2(v376) = *(_DWORD *)(v358 + 64);
                        v327 = sub_1D332F0(
                                 v326,
                                 52,
                                 (__int64)&v376,
                                 v237,
                                 v236,
                                 0,
                                 *(double *)v13.m128_u64,
                                 *(double *)v14.m128i_i64,
                                 a5,
                                 v263.m128i_i64[0],
                                 v263.m128i_u64[1],
                                 v280);
                        v328 = v238;
                        sub_17CD270((__int64 *)&v376);
                        sub_17CD270(v378.m128i_i64);
                        v239 = *(_QWORD *)(v358 + 104);
                        v297 = (_QWORD *)*a1;
                        v378 = _mm_loadu_si128((const __m128i *)(v239 + 40));
                        v379 = *(_QWORD *)(v239 + 56);
                        v310 = *(_WORD *)(v239 + 32);
                        sub_1F7DDA0((__int64)&v376, v239, v229);
                        v240 = *(__int64 **)(v358 + 32);
                        sub_1F80610((__int64)&v374, v257);
                        v311 = sub_1D2B730(
                                 v297,
                                 v365,
                                 (__int64)v366,
                                 (__int64)&v374,
                                 *v240,
                                 v240[1],
                                 (__int64)v327,
                                 v328,
                                 v376,
                                 v377[0],
                                 v337,
                                 v310,
                                 (__int64)&v378,
                                 0);
                        v299 = v241;
                        sub_17CD270((__int64 *)&v374);
                        v281 = *a1;
                        sub_1F80610((__int64)&v378, v268);
                        v265 = (__int64 *)v281;
                        *(_QWORD *)&v242 = sub_1D38970(
                                             v281,
                                             (__int64)&v368,
                                             (__int64)&v378,
                                             v365,
                                             v366,
                                             0,
                                             (__m128i)v13,
                                             *(double *)v14.m128i_i64,
                                             a5,
                                             0);
                        v282 = v242;
                        sub_1F80610((__int64)&v374, v268);
                        v243 = sub_1D332F0(
                                 v265,
                                 v288,
                                 (__int64)&v374,
                                 v365,
                                 v366,
                                 0,
                                 *(double *)v13.m128_u64,
                                 *(double *)v14.m128i_i64,
                                 a5,
                                 v311,
                                 v299,
                                 v282);
                        v289 = v244;
                        v245 = (__int64)v243;
                        sub_17CD270((__int64 *)&v374);
                        sub_17CD270(v378.m128i_i64);
                        v246 = *a1;
                        v247 = *(_QWORD *)(v11 + 104);
                        v378 = 0u;
                        v298 = (_QWORD *)v246;
                        v379 = 0;
                        sub_1F7DDA0((__int64)&v374, v247, v229);
                        v370 = *(_QWORD *)(v11 + 72);
                        if ( v370 )
                          sub_1F6CA20(&v370);
                        v371 = *(_DWORD *)(v11 + 64);
                        v248 = sub_1D2BF40(
                                 v298,
                                 v284,
                                 v266 & 0xFFFFFFFF00000000LL | 1,
                                 (__int64)&v370,
                                 v245,
                                 v289,
                                 (__int64)v327,
                                 v328,
                                 v374,
                                 v375,
                                 v337,
                                 0,
                                 (__int64)&v378);
                        sub_17CD270(&v370);
                        sub_1F81BC0((__int64)a1, (__int64)v327);
                        sub_1F81BC0((__int64)a1, v311);
                        sub_1F81BC0((__int64)a1, v245);
                        v89 = v248;
                        v249 = *(_QWORD *)(*a1 + 664LL);
                        v379 = *a1;
                        v378.m128i_i64[1] = v249;
                        *(_QWORD *)(v379 + 664) = &v378;
                        v250 = *a1;
                        v378.m128i_i64[0] = (__int64)off_49FFF30;
                        v380 = a1;
                        sub_1D44C70(v250, v358, 1, v311, 1u);
                        *(_QWORD *)(v379 + 664) = v378.m128i_i64[1];
                      }
                      sub_135E100(&v368);
                      sub_135E100(v367);
                      goto LABEL_288;
                    }
                    v219 = sub_1454B60(v220);
                    v220 = v219;
                    v365 = sub_1F7DE30(*(_QWORD **)(*a1 + 48LL), v219);
                    v366 = v221;
                  }
                }
                v89 = 0;
LABEL_288:
                sub_135E100(&v363);
                return v89;
              }
            }
          }
        }
      }
    }
    return 0;
  }
  return result;
}
