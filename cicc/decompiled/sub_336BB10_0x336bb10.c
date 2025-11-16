// Function: sub_336BB10
// Address: 0x336bb10
//
__int64 __fastcall sub_336BB10(
        __int64 a1,
        __int64 a2,
        const __m128i *a3,
        unsigned int a4,
        unsigned int a5,
        int *a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        __int128 a9,
        __int64 a10,
        int a11,
        char a12)
{
  __int64 v12; // r15
  __int64 v16; // r14
  __int64 (*v17)(); // rax
  unsigned __int16 v18; // cx
  bool v19; // al
  bool v20; // al
  __int64 v21; // rdx
  __int64 v22; // rdx
  char v23; // al
  unsigned int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned __int32 v30; // edx
  __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int16 v33; // r13
  __int64 v34; // rax
  __int64 result; // rax
  __int64 v36; // r9
  __int64 v37; // r8
  __int64 v39; // rdx
  __int64 v40; // rax
  char v41; // dl
  __int64 v42; // rax
  __int64 v43; // rdx
  int v44; // eax
  int v45; // r9d
  unsigned int v46; // ecx
  unsigned int v47; // edx
  unsigned int v48; // ecx
  unsigned int v49; // esi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r8
  __int64 v53; // rsi
  unsigned int v54; // edx
  unsigned int v55; // edx
  __int64 v56; // rcx
  _BYTE *v57; // rax
  int v58; // r9d
  __int64 v59; // rcx
  __int64 v60; // rdx
  unsigned __int32 v61; // edx
  __int64 v62; // rdi
  __int64 v63; // rsi
  __int64 (__fastcall *v64)(__int64, __int64, __int64, unsigned int, unsigned __int64, __int64 *, unsigned int *, unsigned __int16 *); // rax
  unsigned int v65; // eax
  __int64 v66; // r8
  _QWORD *v67; // r9
  __int64 v68; // rdx
  const char *v69; // rax
  const char *v70; // rcx
  int v71; // r14d
  const char *i; // rdx
  unsigned __int16 v73; // bx
  char v74; // di
  __int64 v75; // r13
  __int64 v76; // rdx
  char v77; // cl
  __int64 v78; // rsi
  __int64 *v79; // rax
  unsigned int v80; // r14d
  int v81; // eax
  __int64 v82; // r8
  __int64 v83; // r9
  int v84; // ebx
  int v85; // edi
  int v86; // r13d
  __int64 v87; // r8
  __int64 v88; // r9
  __int32 v89; // edx
  unsigned __int16 v90; // r14
  unsigned __int16 *v91; // rcx
  int v92; // r13d
  __int64 v93; // rdx
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rbx
  __int64 v97; // rdx
  __int64 v98; // rax
  unsigned __int16 *v99; // rdx
  __int64 v100; // r8
  unsigned __int16 *v101; // rdx
  __int64 v102; // rax
  int v103; // eax
  __int64 v104; // rbx
  unsigned int v105; // edx
  int v106; // r9d
  unsigned int v107; // edx
  unsigned int v108; // r13d
  _BYTE *v109; // rax
  __int64 v110; // rcx
  unsigned int v111; // r10d
  __int64 v112; // rax
  unsigned __int32 v113; // edx
  unsigned int v114; // eax
  __int64 v115; // rax
  __int64 v116; // rax
  bool v117; // al
  __int64 v118; // rax
  __int64 v119; // rsi
  __int64 v120; // rdx
  char v121; // bl
  __int64 v122; // rdx
  bool v123; // al
  char v124; // al
  int v125; // r9d
  int v126; // r9d
  int v127; // r9d
  bool v128; // al
  bool v129; // al
  bool v130; // al
  unsigned int v131; // r14d
  unsigned int v132; // r13d
  int v133; // eax
  int v134; // edi
  bool v135; // al
  __int64 v136; // rdx
  __int64 v137; // r8
  bool v138; // al
  __int64 v139; // rcx
  __int64 v140; // rdx
  __int64 v141; // rdx
  int v142; // ebx
  unsigned __int64 v143; // rdx
  unsigned __int64 v144; // rax
  __int64 v145; // rcx
  unsigned __int16 *v146; // rdx
  __int64 v147; // r8
  unsigned __int64 v148; // rsi
  int v149; // eax
  __int64 v150; // r13
  unsigned __int16 v151; // ax
  __int64 *v152; // r14
  unsigned int v153; // ebx
  __int16 v154; // ax
  __int64 v155; // r8
  __int64 v156; // r9
  __int64 v157; // rdx
  __int128 v158; // rax
  int v159; // r9d
  __int64 v160; // rax
  unsigned __int16 v161; // bx
  unsigned __int16 v162; // r13
  __int32 v163; // edx
  int v164; // eax
  int v165; // edx
  unsigned __int16 v166; // r14
  __int64 v167; // rbx
  __int64 v168; // rdx
  unsigned __int16 v169; // r13
  __int64 v170; // rdx
  __int64 v171; // rax
  unsigned __int16 *v172; // rdx
  __int64 v173; // rdi
  __int64 v174; // r12
  unsigned int v175; // r13d
  __int64 v176; // rbx
  __int64 v177; // rax
  int v178; // edx
  int v179; // edi
  __int64 v180; // rdx
  __int64 v181; // rax
  int v182; // r9d
  __int64 (__fastcall *v183)(__int64, __int64, unsigned int); // r13
  __int64 v184; // rax
  unsigned __int16 v185; // ax
  __int64 v186; // r13
  __int64 v187; // rdx
  __int64 v188; // r14
  int v189; // r9d
  int v190; // eax
  int v191; // edx
  int v192; // r9d
  __int64 v193; // rax
  unsigned int v194; // edx
  __int64 v195; // rax
  __int64 v196; // rax
  __int64 v197; // rdx
  __int64 v198; // rax
  unsigned __int32 v199; // edx
  unsigned __int32 v200; // ebx
  __int64 v201; // r13
  _BYTE *v202; // rax
  _QWORD *v203; // rdi
  bool v204; // zf
  unsigned int v205; // eax
  __int64 v206; // rbx
  unsigned int v207; // ebx
  __int64 v208; // rdx
  __int64 v209; // r13
  int v210; // r9d
  unsigned int v211; // edx
  __int64 v212; // rax
  int v213; // ecx
  int v214; // edx
  __int64 v215; // rax
  __int64 v216; // rax
  __int64 v217; // rdx
  int v218; // eax
  __int128 v219; // rax
  int v220; // r9d
  unsigned int v221; // edx
  int v222; // r9d
  unsigned int v223; // edx
  int v224; // r9d
  unsigned __int32 v225; // edx
  __int64 v226; // r14
  char v227; // bl
  __int64 v228; // rcx
  __int64 v229; // rdx
  __int64 v230; // rax
  __int64 v231; // rdx
  int v232; // edx
  __int64 v233; // rax
  int v234; // edx
  unsigned int v235; // r8d
  unsigned int v236; // r14d
  int v237; // r13d
  int v238; // ebx
  int v239; // edx
  __int64 v240; // rax
  int v241; // edx
  int v242; // edi
  __int64 v243; // rdx
  __int64 v244; // rax
  char *v245; // rax
  unsigned __int64 v246; // rbx
  __int64 v247; // rdx
  char v248; // r14
  __int64 v249; // rdx
  __int64 v250; // rax
  unsigned __int16 *v251; // rdx
  __int64 v252; // rax
  __int64 v253; // rdx
  __int64 v254; // rdx
  __int64 v255; // rax
  __int64 v256; // rsi
  int v257; // eax
  int v258; // edx
  int v259; // r8d
  int v260; // ebx
  __int32 v261; // edx
  bool v262; // al
  __int64 v263; // rdx
  __int64 v264; // rdx
  __int64 v265; // rax
  __int64 v266; // rdx
  unsigned int v267; // eax
  __int64 v268; // rdx
  int v269; // r9d
  __int64 v270; // rax
  unsigned __int32 v271; // edx
  __int16 v272; // ax
  __int64 v273; // rdx
  __int128 v274; // rax
  int v275; // r9d
  unsigned __int16 v276; // ax
  __int64 v277; // rdx
  __int64 v278; // rax
  __int64 v279; // rdx
  unsigned int v280; // ebx
  __int64 v281; // rax
  __int64 v282; // rdx
  int v283; // eax
  int v284; // edx
  int v285; // r9d
  __int32 v286; // edx
  __int64 v287; // rax
  __int32 v288; // edx
  __int64 v289; // rbx
  __int32 v290; // edx
  __int32 v291; // eax
  __int32 v292; // edx
  __int32 v293; // edx
  __int128 v294; // [rsp-20h] [rbp-370h]
  __int128 v295; // [rsp-10h] [rbp-360h]
  __int128 v296; // [rsp-10h] [rbp-360h]
  __int128 v297; // [rsp-10h] [rbp-360h]
  __int128 v298; // [rsp+0h] [rbp-350h]
  __int128 v299; // [rsp+0h] [rbp-350h]
  __int128 v300; // [rsp+0h] [rbp-350h]
  __int128 v301; // [rsp+0h] [rbp-350h]
  __int128 v302; // [rsp+0h] [rbp-350h]
  unsigned __int128 v303; // [rsp+0h] [rbp-350h]
  __int128 v304; // [rsp+0h] [rbp-350h]
  __int64 v305; // [rsp+8h] [rbp-348h]
  __int32 v306; // [rsp+8h] [rbp-348h]
  int v307; // [rsp+10h] [rbp-340h]
  unsigned __int32 v308; // [rsp+10h] [rbp-340h]
  __int16 v309; // [rsp+12h] [rbp-33Eh]
  int v310; // [rsp+18h] [rbp-338h]
  __int16 v311; // [rsp+1Ah] [rbp-336h]
  __int16 v312; // [rsp+22h] [rbp-32Eh]
  int v313; // [rsp+28h] [rbp-328h]
  unsigned int v314; // [rsp+30h] [rbp-320h]
  __int64 v315; // [rsp+30h] [rbp-320h]
  __int64 v316; // [rsp+38h] [rbp-318h]
  __int64 v317; // [rsp+38h] [rbp-318h]
  __int64 v318; // [rsp+40h] [rbp-310h]
  __int64 v319; // [rsp+40h] [rbp-310h]
  unsigned int v320; // [rsp+40h] [rbp-310h]
  int v321; // [rsp+40h] [rbp-310h]
  int v322; // [rsp+40h] [rbp-310h]
  __int128 v323; // [rsp+40h] [rbp-310h]
  int v324; // [rsp+40h] [rbp-310h]
  __int64 v325; // [rsp+48h] [rbp-308h]
  unsigned __int64 v326; // [rsp+48h] [rbp-308h]
  unsigned int v327; // [rsp+50h] [rbp-300h]
  __int64 v328; // [rsp+50h] [rbp-300h]
  __int64 v329; // [rsp+50h] [rbp-300h]
  __int64 v330; // [rsp+58h] [rbp-2F8h]
  unsigned __int32 v331; // [rsp+58h] [rbp-2F8h]
  __int64 v332; // [rsp+58h] [rbp-2F8h]
  unsigned int v333; // [rsp+58h] [rbp-2F8h]
  __int64 v334; // [rsp+58h] [rbp-2F8h]
  unsigned __int32 v335; // [rsp+58h] [rbp-2F8h]
  __m128i v336; // [rsp+60h] [rbp-2F0h]
  int v337; // [rsp+60h] [rbp-2F0h]
  __int64 v338; // [rsp+60h] [rbp-2F0h]
  char v339; // [rsp+60h] [rbp-2F0h]
  __int64 v340; // [rsp+70h] [rbp-2E0h]
  unsigned int v341; // [rsp+70h] [rbp-2E0h]
  char v342; // [rsp+70h] [rbp-2E0h]
  unsigned int v343; // [rsp+70h] [rbp-2E0h]
  int v344; // [rsp+70h] [rbp-2E0h]
  __int64 v345; // [rsp+70h] [rbp-2E0h]
  int *v346; // [rsp+78h] [rbp-2D8h]
  unsigned int v347; // [rsp+78h] [rbp-2D8h]
  unsigned __int16 v348; // [rsp+78h] [rbp-2D8h]
  char v349; // [rsp+78h] [rbp-2D8h]
  unsigned __int32 v350; // [rsp+78h] [rbp-2D8h]
  unsigned int v351; // [rsp+78h] [rbp-2D8h]
  int v352; // [rsp+78h] [rbp-2D8h]
  __int64 *v354; // [rsp+80h] [rbp-2D0h]
  char v355; // [rsp+80h] [rbp-2D0h]
  unsigned __int16 v356; // [rsp+80h] [rbp-2D0h]
  unsigned __int16 v357; // [rsp+80h] [rbp-2D0h]
  unsigned __int16 v358; // [rsp+80h] [rbp-2D0h]
  __int64 v359; // [rsp+80h] [rbp-2D0h]
  __int64 *v360; // [rsp+80h] [rbp-2D0h]
  __int64 v361; // [rsp+80h] [rbp-2D0h]
  char v362; // [rsp+80h] [rbp-2D0h]
  char v363; // [rsp+80h] [rbp-2D0h]
  __int64 v364; // [rsp+80h] [rbp-2D0h]
  unsigned int v365; // [rsp+80h] [rbp-2D0h]
  char v366; // [rsp+80h] [rbp-2D0h]
  __int64 v367; // [rsp+88h] [rbp-2C8h]
  __int64 v368; // [rsp+148h] [rbp-208h]
  __int64 v369; // [rsp+150h] [rbp-200h]
  unsigned __int64 v370; // [rsp+160h] [rbp-1F0h]
  unsigned __int64 v371; // [rsp+170h] [rbp-1E0h] BYREF
  unsigned __int64 v372; // [rsp+178h] [rbp-1D8h]
  __m128i v373; // [rsp+180h] [rbp-1D0h] BYREF
  int v374; // [rsp+190h] [rbp-1C0h] BYREF
  __int64 v375; // [rsp+198h] [rbp-1B8h]
  __int64 v376; // [rsp+1A0h] [rbp-1B0h]
  __int64 v377; // [rsp+1A8h] [rbp-1A8h]
  __int64 v378; // [rsp+1B0h] [rbp-1A0h]
  __int64 v379; // [rsp+1B8h] [rbp-198h]
  __int64 v380; // [rsp+1C0h] [rbp-190h]
  __int64 v381; // [rsp+1C8h] [rbp-188h]
  __int64 v382; // [rsp+1D0h] [rbp-180h]
  __int64 v383; // [rsp+1D8h] [rbp-178h]
  unsigned __int16 *v384; // [rsp+1E0h] [rbp-170h]
  __int64 v385; // [rsp+1E8h] [rbp-168h]
  __int64 v386; // [rsp+1F0h] [rbp-160h]
  __int64 v387; // [rsp+1F8h] [rbp-158h]
  unsigned __int16 *v388; // [rsp+200h] [rbp-150h]
  __int64 v389; // [rsp+208h] [rbp-148h]
  __int64 v390; // [rsp+210h] [rbp-140h]
  __int64 v391; // [rsp+218h] [rbp-138h]
  unsigned __int16 *v392; // [rsp+220h] [rbp-130h]
  __int64 v393; // [rsp+228h] [rbp-128h]
  _QWORD v394[2]; // [rsp+230h] [rbp-120h] BYREF
  __int64 v395; // [rsp+240h] [rbp-110h] BYREF
  __int64 v396; // [rsp+248h] [rbp-108h]
  _QWORD v397[2]; // [rsp+250h] [rbp-100h] BYREF
  _QWORD v398[2]; // [rsp+260h] [rbp-F0h] BYREF
  __int64 v399; // [rsp+270h] [rbp-E0h]
  __int64 v400; // [rsp+278h] [rbp-D8h]
  __int64 v401; // [rsp+280h] [rbp-D0h] BYREF
  __int64 v402; // [rsp+288h] [rbp-C8h]
  const char *v403; // [rsp+290h] [rbp-C0h] BYREF
  __int64 v404; // [rsp+298h] [rbp-B8h]
  _BYTE v405[176]; // [rsp+2A0h] [rbp-B0h] BYREF

  v12 = a2;
  v16 = *(_QWORD *)(a1 + 16);
  v346 = a6;
  v17 = *(__int64 (**)())(*(_QWORD *)v16 + 2288LL);
  if ( v17 != sub_302E200 )
  {
    v36 = a5;
    v37 = a4;
    v39 = a2;
    a2 = a1;
    result = ((__int64 (__fastcall *)(__int64, __int64, __int64, const __m128i *, __int64, __int64, unsigned __int64, unsigned __int64, __int64))v17)(
               v16,
               a1,
               v39,
               a3,
               v37,
               v36,
               a7,
               a8,
               a10);
    if ( result )
      return result;
  }
  v18 = a7;
  if ( (_WORD)a7 )
  {
    if ( (unsigned __int16)(a7 - 17) > 0xD3u )
    {
      v336 = _mm_loadu_si128(a3);
      if ( a4 <= 1 )
      {
        v31 = a3->m128i_u32[2];
        v340 = a3->m128i_i64[0];
        v115 = *(_QWORD *)(a3->m128i_i64[0] + 48) + 16 * v31;
        v331 = a3->m128i_u32[2];
        v33 = *(_WORD *)v115;
        v116 = *(_QWORD *)(v115 + 8);
        LOWORD(v395) = v33;
        v396 = v116;
        if ( (_WORD)a7 == v33 )
          return v340;
        goto LABEL_107;
      }
      if ( (unsigned __int16)(a7 - 2) > 7u && (unsigned __int16)(a7 - 176) > 0x1Fu )
      {
LABEL_6:
        if ( (unsigned __int16)(a5 - 10) <= 6u
          || (unsigned __int16)(a5 - 126) <= 0x31u
          || (unsigned __int16)(a5 - 208) <= 0x14u )
        {
          v104 = sub_33FAF80(a1, 234, v12, 13, 0, (_DWORD)a6);
          v320 = v105;
          v347 = v105;
          v334 = sub_33FAF80(a1, 234, v12, 13, 0, v106);
          v108 = v107;
          v343 = v107;
          v109 = (_BYTE *)sub_2E79000(*(__int64 **)(a1 + 40));
          v110 = v334;
          v111 = v347;
          if ( *v109 != 1 && (_WORD)a7 != 16 )
          {
            v110 = v104;
            v108 = v320;
            v111 = v343;
            v104 = v334;
          }
          *((_QWORD *)&v301 + 1) = v111;
          *(_QWORD *)&v301 = v104;
          *((_QWORD *)&v296 + 1) = v108;
          *(_QWORD *)&v296 = v110;
          v112 = sub_3406EB0(a1, 54, v12, a7, a8, v111, v296, v301);
          v18 = a7;
          v340 = v112;
          v331 = v113;
        }
        else
        {
          if ( v18 )
          {
            if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
              goto LABEL_397;
            v195 = 16LL * (v18 - 1);
            v22 = *(_QWORD *)&byte_444C4A0[v195];
            v23 = byte_444C4A0[v195 + 8];
          }
          else
          {
            v399 = sub_3007260((__int64)&a7);
            v400 = v21;
            v22 = v399;
            v23 = v400;
          }
          v403 = (const char *)v22;
          LOBYTE(v404) = v23;
          v24 = sub_CA1930(&v403);
          switch ( v24 )
          {
            case 1u:
              LOWORD(v25) = 2;
              v27 = 0;
              break;
            case 2u:
              LOWORD(v25) = 3;
              v27 = 0;
              break;
            case 4u:
              LOWORD(v25) = 4;
              v27 = 0;
              break;
            case 8u:
              LOWORD(v25) = 5;
              v27 = 0;
              break;
            case 0x10u:
              LOWORD(v25) = 6;
              v27 = 0;
              break;
            case 0x20u:
              LOWORD(v25) = 7;
              v27 = 0;
              break;
            case 0x40u:
              LOWORD(v25) = 8;
              v27 = 0;
              break;
            case 0x80u:
              LOWORD(v25) = 9;
              v27 = 0;
              break;
            default:
              v25 = sub_3007020(*(_QWORD **)(a1 + 64), v24);
              v330 = v25;
              v27 = v26;
              break;
          }
          v28 = v330;
          LOWORD(v28) = v25;
          v29 = sub_336BB10(
                  a1,
                  v12,
                  (_DWORD)a3,
                  a4,
                  a5,
                  (_DWORD)v346,
                  v28,
                  v27,
                  a9,
                  *((__int64 *)&a9 + 1),
                  a10,
                  (_DWORD)v403,
                  0);
          v18 = a7;
          v340 = v29;
          v331 = v30;
        }
LABEL_21:
        v31 = v331;
        v32 = *(_QWORD *)(v340 + 48) + 16LL * v331;
        v33 = *(_WORD *)v32;
        v34 = *(_QWORD *)(v32 + 8);
        LOWORD(v395) = v33;
        v396 = v34;
        if ( v33 == v18 )
        {
          if ( a8 == v34 || v18 )
            return v340;
          if ( !sub_3007070((__int64)&v395) )
            goto LABEL_118;
          goto LABEL_166;
        }
LABEL_107:
        if ( v33 )
        {
          if ( (unsigned __int16)(v33 - 2) > 7u
            && (unsigned __int16)(v33 - 17) > 0x6Cu
            && (unsigned __int16)(v33 - 176) > 0x1Fu )
          {
            goto LABEL_117;
          }
        }
        else
        {
          v358 = v18;
          v128 = sub_3007070((__int64)&v395);
          v18 = v358;
          if ( !v128 )
            goto LABEL_117;
        }
        if ( v18 )
        {
          if ( (unsigned __int16)(v18 - 10) > 6u
            && (unsigned __int16)(v18 - 126) > 0x31u
            && (unsigned __int16)(v18 - 208) > 0x14u )
          {
            goto LABEL_146;
          }
          goto LABEL_115;
        }
LABEL_166:
        if ( !(unsigned __int8)sub_3007030((__int64)&a7) )
          goto LABEL_119;
        v18 = 0;
LABEL_115:
        v356 = v18;
        v117 = sub_3280B30((__int64)&a7, (unsigned int)v395, v396);
        v18 = v356;
        if ( v117 )
        {
          v265 = sub_2D5B750((unsigned __int16 *)&a7);
          v404 = v266;
          v403 = (const char *)v265;
          v267 = sub_CA1930(&v403);
          LODWORD(v395) = sub_327FC40(*(_QWORD **)(a1 + 64), v267);
          v396 = v268;
          v336.m128i_i64[1] = v31 | v336.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v270 = sub_33FAF80(a1, 216, v12, v395, v268, v269);
          v18 = a7;
          v340 = v270;
          v331 = v271;
        }
LABEL_117:
        if ( !v18 )
        {
LABEL_118:
          v33 = v395;
LABEL_119:
          v118 = sub_3007260((__int64)&a7);
          v18 = 0;
          v403 = (const char *)v118;
          v119 = v118;
          v404 = v120;
          v121 = v120;
LABEL_120:
          if ( !v33 )
          {
            v348 = v18;
            v401 = sub_3007260((__int64)&v395);
            v402 = v122;
            if ( v401 == v119 && (_BYTE)v402 == v121 )
              return sub_33FAF80(a1, 234, v12, a7, a8, (_DWORD)a6);
            v123 = sub_3007070((__int64)&v395);
            v18 = v348;
            if ( !v123 )
            {
LABEL_123:
              v357 = v18;
              v124 = sub_3007030((__int64)&v395);
              v18 = v357;
              if ( !v124 )
                goto LABEL_124;
              goto LABEL_133;
            }
LABEL_125:
            if ( v18 )
            {
              if ( (unsigned __int16)(v18 - 2) <= 7u
                || (unsigned __int16)(v18 - 17) <= 0x6Cu
                || (unsigned __int16)(v18 - 176) <= 0x1Fu )
              {
LABEL_129:
                if ( !sub_3280B30((__int64)&a7, (unsigned int)v395, v396) )
                  return sub_33FAF80(a1, 215, v12, a7, a8, v125);
                if ( a12 )
                {
                  *(_QWORD *)&v274 = sub_33F7D60(a1, (unsigned int)a7, a8);
                  sub_3406EB0(
                    a1,
                    a11,
                    v12,
                    v395,
                    v396,
                    v275,
                    __PAIR128__(v331 | v336.m128i_i64[1] & 0xFFFFFFFF00000000LL, v340),
                    v274);
                }
                return sub_33FAF80(a1, 216, v12, a7, a8, v125);
              }
            }
            else
            {
              v129 = sub_3007070((__int64)&a7);
              v18 = 0;
              if ( v129 )
                goto LABEL_129;
            }
            if ( !v33 )
              goto LABEL_123;
LABEL_155:
            if ( (unsigned __int16)(v33 - 10) > 6u
              && (unsigned __int16)(v33 - 126) > 0x31u
              && (unsigned __int16)(v33 - 208) > 0x14u )
            {
              if ( v33 != 261 )
                goto LABEL_124;
              if ( v18 )
              {
LABEL_138:
                if ( (unsigned __int16)(v18 - 2) > 7u
                  && (unsigned __int16)(v18 - 17) > 0x6Cu
                  && (unsigned __int16)(v18 - 176) > 0x1Fu )
                {
                  goto LABEL_124;
                }
                goto LABEL_141;
              }
              goto LABEL_160;
            }
LABEL_133:
            if ( v18 )
            {
              if ( (unsigned __int16)(v18 - 10) > 6u
                && (unsigned __int16)(v18 - 126) > 0x31u
                && (unsigned __int16)(v18 - 208) > 0x14u )
              {
                if ( v33 != 261 )
                  goto LABEL_124;
                goto LABEL_138;
              }
              goto LABEL_218;
            }
            if ( (unsigned __int8)sub_3007030((__int64)&a7) )
            {
LABEL_218:
              if ( !sub_3280B30(
                      (__int64)&a7,
                      *(unsigned __int16 *)(*(_QWORD *)(v340 + 48) + 16LL * v331),
                      *(_QWORD *)(*(_QWORD *)(v340 + 48) + 16LL * v331 + 8)) )
                return sub_33FAF80(a1, 233, v12, a7, a8, v182);
              v183 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v16 + 32LL);
              v184 = sub_2E79000(*(__int64 **)(a1 + 40));
              if ( v183 == sub_2D42F30 )
                v185 = sub_3366D50(v184, 0);
              else
                v185 = v183(v16, v184, 0);
              v186 = sub_3400BD0(a1, 1, v12, v185, 0, 1, 0);
              v188 = v187;
              v394[0] = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 120LL);
              if ( (unsigned __int8)sub_A73ED0(v394, 72) )
              {
                v190 = sub_33E5110(a1, (unsigned int)a7, a8, 1, 0);
                *((_QWORD *)&v302 + 1) = v188;
                *(_QWORD *)&v302 = v186;
                return sub_3412970(
                         a1,
                         145,
                         v12,
                         v190,
                         v191,
                         v192,
                         a9,
                         __PAIR128__(v331 | v336.m128i_i64[1] & 0xFFFFFFFF00000000LL, v340),
                         v302);
              }
              else
              {
                *((_QWORD *)&v304 + 1) = v188;
                *(_QWORD *)&v304 = v186;
                return sub_3406EB0(
                         a1,
                         230,
                         v12,
                         a7,
                         a8,
                         v189,
                         __PAIR128__(v331 | v336.m128i_i64[1] & 0xFFFFFFFF00000000LL, v340),
                         v304);
              }
            }
            if ( v33 != 261 )
              goto LABEL_124;
LABEL_160:
            if ( !sub_3007070((__int64)&a7) )
              goto LABEL_124;
LABEL_141:
            if ( sub_3280B30((__int64)&a7, (unsigned int)v395, v396) )
            {
              sub_33FAF80(a1, 234, v12, 8, 0, v126);
              return sub_33FAF80(a1, 216, v12, a7, a8, v127);
            }
LABEL_124:
            sub_C64ED0("Unknown mismatch in getCopyFromParts!", 1u);
          }
          if ( v33 != 1 && (unsigned __int16)(v33 - 504) > 7u )
          {
            if ( *(_QWORD *)&byte_444C4A0[16 * v33 - 16] == v119 && v121 == byte_444C4A0[16 * v33 - 8] )
              return sub_33FAF80(a1, 234, v12, a7, a8, (_DWORD)a6);
            if ( (unsigned __int16)(v33 - 2) > 7u
              && (unsigned __int16)(v33 - 17) > 0x6Cu
              && (unsigned __int16)(v33 - 176) > 0x1Fu )
            {
              goto LABEL_155;
            }
            goto LABEL_125;
          }
LABEL_397:
          BUG();
        }
LABEL_146:
        if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
          goto LABEL_397;
        v33 = v395;
        v119 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
        v121 = byte_444C4A0[16 * v18 - 8];
        goto LABEL_120;
      }
LABEL_30:
      if ( (unsigned __int16)a5 <= 1u || (unsigned __int16)(a5 - 504) <= 7u )
        goto LABEL_397;
      v40 = 16LL * ((unsigned __int16)a5 - 1);
      v41 = byte_444C4A0[v40 + 8];
      v403 = *(const char **)&byte_444C4A0[v40];
      LOBYTE(v404) = v41;
      v313 = sub_CA1930(&v403);
      if ( (_WORD)a7 )
      {
        if ( (_WORD)a7 == 1 || (unsigned __int16)(a7 - 504) <= 7u )
          goto LABEL_397;
        v43 = 16LL * ((unsigned __int16)a7 - 1);
        v42 = *(_QWORD *)&byte_444C4A0[v43];
        LOBYTE(v43) = byte_444C4A0[v43 + 8];
      }
      else
      {
        v42 = sub_3007260((__int64)&a7);
        v397[0] = v42;
        v397[1] = v43;
      }
      LOBYTE(v404) = v43;
      v403 = (const char *)v42;
      v44 = sub_CA1930(&v403);
      _BitScanReverse(&v46, a4);
      v47 = 0x80000000 >> (v46 ^ 0x1F);
      v48 = v313 * v47;
      v327 = v47;
      if ( v313 * v47 == v44 )
      {
        v310 = a7;
        v307 = a8;
      }
      else
      {
        v344 = v313 * v47;
        v164 = sub_327FC40(*(_QWORD **)(a1 + 64), v48);
        v48 = v344;
        v310 = v164;
        v307 = v165;
      }
      v49 = v48 >> 1;
      if ( v48 >> 1 == 1 )
      {
        LOWORD(v50) = 2;
      }
      else
      {
        switch ( v49 )
        {
          case 2u:
            LOWORD(v50) = 3;
            break;
          case 4u:
            LOWORD(v50) = 4;
            break;
          case 8u:
            LOWORD(v50) = 5;
            break;
          case 0x10u:
            LOWORD(v50) = 6;
            break;
          case 0x20u:
            LOWORD(v50) = 7;
            break;
          case 0x40u:
            LOWORD(v50) = 8;
            break;
          case 0x80u:
            LOWORD(v50) = 9;
            break;
          default:
            v50 = sub_3007020(*(_QWORD **)(a1 + 64), v49);
            v318 = v50;
            v52 = v51;
LABEL_45:
            v53 = v318;
            LOWORD(v53) = v50;
            if ( v327 == 2 )
            {
              v322 = v52;
              v193 = sub_33FAF80(a1, 234, v12, v53, v52, v45);
              v341 = v194;
              v332 = v193;
              v56 = sub_33FAF80(a1, 234, v12, v53, v322, v53);
            }
            else
            {
              BYTE4(v401) = 0;
              v316 = v52;
              v332 = sub_336BB10(
                       a1,
                       v12,
                       (_DWORD)a3,
                       v327 >> 1,
                       a5,
                       (_DWORD)v346,
                       v53,
                       v52,
                       a9,
                       *((__int64 *)&a9 + 1),
                       v401,
                       (_DWORD)v403,
                       0);
              v341 = v54;
              BYTE4(v401) = 0;
              v56 = sub_336BB10(
                      a1,
                      v12,
                      (unsigned int)a3 + 16 * (v327 >> 1),
                      v327 >> 1,
                      a5,
                      (_DWORD)v346,
                      v53,
                      v316,
                      a9,
                      *((__int64 *)&a9 + 1),
                      v401,
                      (_DWORD)v403,
                      0);
            }
            v314 = v55;
            v319 = v56;
            v57 = (_BYTE *)sub_2E79000(*(__int64 **)(a1 + 40));
            v59 = v319;
            v60 = v314;
            if ( *v57 )
            {
              v60 = v341;
              v341 = v314;
              v59 = v332;
              v332 = v319;
            }
            v325 = v60;
            *((_QWORD *)&v298 + 1) = v60;
            *(_QWORD *)&v298 = v59;
            v317 = v341;
            *((_QWORD *)&v295 + 1) = v341;
            *(_QWORD *)&v295 = v332;
            v340 = sub_3406EB0(a1, 54, v12, v310, v307, v58, v295, v298);
            v331 = v61;
            if ( a4 > v327 )
            {
              v308 = v61;
              v196 = sub_327FC40(*(_QWORD **)(a1 + 64), (a4 - v327) * v313);
              v198 = sub_336BB10(
                       a1,
                       v12,
                       (unsigned int)a3 + 16 * v327,
                       a4 - v327,
                       a5,
                       (_DWORD)v346,
                       v196,
                       v197,
                       a9,
                       *((__int64 *)&a9 + 1),
                       a10,
                       (_DWORD)v403,
                       0);
              v200 = v199;
              v350 = v199;
              v201 = v198;
              v202 = (_BYTE *)sub_2E79000(*(__int64 **)(a1 + 40));
              v203 = *(_QWORD **)(a1 + 64);
              v204 = *v202 == 0;
              if ( !*v202 )
                v200 = v331;
              v205 = v308;
              if ( v204 )
                v205 = v350;
              v335 = v200;
              v351 = v205;
              v206 = v340;
              if ( v204 )
              {
                v206 = v201;
                v201 = v340;
              }
              v329 = v206;
              v345 = v201;
              v207 = sub_327FC40(v203, v313 * a4);
              v209 = v208;
              v326 = v351 | v325 & 0xFFFFFFFF00000000LL;
              v303 = __PAIR128__(v326, v329);
              *(_QWORD *)&v323 = sub_33FAF80(a1, 215, v12, v207, v208, v210);
              *((_QWORD *)&v323 + 1) = v211 | v326 & 0xFFFFFFFF00000000LL;
              v212 = sub_2E79000(*(__int64 **)(a1 + 40));
              v213 = sub_2FE6750(v16, v207, v209, v212);
              LODWORD(v329) = v214;
              v215 = *(_QWORD *)(v345 + 48) + 16LL * v335;
              v352 = v213;
              LOWORD(v214) = *(_WORD *)v215;
              v216 = *(_QWORD *)(v215 + 8);
              LOWORD(v403) = v214;
              v404 = v216;
              v398[0] = sub_2D5B750((unsigned __int16 *)&v403);
              v398[1] = v217;
              v403 = (const char *)v398[0];
              LOBYTE(v404) = v217;
              v218 = sub_CA1930(&v403);
              LODWORD(v303) = 0;
              *(_QWORD *)&v219 = sub_3400BD0(a1, v218, v12, v352, v329, 0, v303);
              *(_QWORD *)&v323 = sub_3406EB0(a1, 190, v12, v207, v209, v220, v323, v219);
              *((_QWORD *)&v297 + 1) = v221 | *((_QWORD *)&v323 + 1) & 0xFFFFFFFF00000000LL;
              v315 = sub_33FAF80(a1, 214, v12, v207, v209, v222);
              *(_QWORD *)&v297 = v323;
              *((_QWORD *)&v294 + 1) = v223 | v317 & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v294 = v315;
              v340 = sub_3406EB0(a1, 187, v12, v207, v209, v224, v294, v297);
              v331 = v225;
            }
            v18 = a7;
            goto LABEL_21;
        }
      }
      v52 = 0;
      goto LABEL_45;
    }
  }
  else
  {
    v19 = sub_30070B0((__int64)&a7);
    v18 = 0;
    if ( !v19 )
    {
      v336 = _mm_loadu_si128(a3);
      if ( a4 <= 1 )
      {
        v340 = a3->m128i_i64[0];
        v331 = a3->m128i_u32[2];
        goto LABEL_21;
      }
      v20 = sub_3007070((__int64)&a7);
      v18 = 0;
      if ( !v20 )
        goto LABEL_6;
      goto LABEL_30;
    }
  }
  v62 = *(_QWORD *)(a1 + 16);
  HIDWORD(v368) = HIDWORD(a10);
  v371 = a7;
  v337 = a10;
  v372 = a8;
  v342 = BYTE4(a10);
  v328 = v62;
  v373 = _mm_loadu_si128(a3);
  if ( a4 > 1 )
  {
    v402 = 0;
    LOWORD(v401) = 0;
    v63 = *(_QWORD *)(a1 + 64);
    LOWORD(v397[0]) = 0;
    if ( BYTE4(a10) )
    {
      v64 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int, unsigned __int64, __int64 *, unsigned int *, unsigned __int16 *))(*(_QWORD *)v62 + 600LL);
      if ( v64 == sub_2FE9890 )
      {
        v65 = sub_2FE8D10(v62, v63, a7, a8, &v401, (unsigned int *)v398, (unsigned __int16 *)v397);
      }
      else
      {
        v65 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, unsigned __int64))v64)(
                v62,
                v63,
                (unsigned int)a10,
                (unsigned int)v371,
                v372);
        v67 = v398;
      }
      v333 = v65;
    }
    else
    {
      v114 = sub_2FE8D10(v62, v63, (unsigned int)v371, v372, &v401, (unsigned int *)v398, (unsigned __int16 *)v397);
      v66 = v305;
      v333 = v114;
    }
    v68 = LODWORD(v398[0]);
    v69 = v405;
    v70 = v405;
    v403 = v405;
    v71 = v398[0];
    v404 = 0x800000000LL;
    if ( LODWORD(v398[0]) )
    {
      if ( LODWORD(v398[0]) > 8uLL )
      {
        v364 = LODWORD(v398[0]);
        sub_C8D5F0((__int64)&v403, v405, LODWORD(v398[0]), 0x10u, v66, (__int64)v67);
        v70 = v403;
        v68 = v364;
        v69 = &v403[16 * (unsigned int)v404];
      }
      for ( i = &v70[16 * v68]; i != v69; v69 += 16 )
      {
        if ( v69 )
        {
          *(_QWORD *)v69 = 0;
          *((_DWORD *)v69 + 2) = 0;
        }
      }
      LODWORD(v404) = v71;
      if ( LODWORD(v398[0]) == v333 )
      {
        if ( v333 )
        {
          v321 = (int)a3;
          v175 = a5;
          v176 = 0;
          do
          {
            LODWORD(v368) = v337;
            BYTE4(v368) = v342;
            v177 = sub_336BB10(
                     a1,
                     v12,
                     v321 + (int)v176,
                     1,
                     v175,
                     (_DWORD)v346,
                     v401,
                     v402,
                     a9,
                     *((__int64 *)&a9 + 1),
                     v368,
                     v399,
                     0);
            v179 = v178;
            v180 = v177;
            v181 = (__int64)v403;
            *(_QWORD *)&v403[v176] = v180;
            *(_DWORD *)(v181 + v176 + 8) = v179;
            v176 += 16;
          }
          while ( v176 != 16LL * v333 );
        }
      }
      else if ( v333 )
      {
        v324 = (int)a3;
        v235 = a5;
        v236 = 0;
        v237 = 0;
        v238 = v333 / LODWORD(v398[0]);
        do
        {
          v239 = 16 * v237;
          LODWORD(v368) = v337;
          v237 += v238;
          v365 = v235;
          BYTE4(v368) = v342;
          v240 = sub_336BB10(
                   a1,
                   v12,
                   v324 + v239,
                   v238,
                   v235,
                   (_DWORD)v346,
                   v401,
                   v402,
                   a9,
                   *((__int64 *)&a9 + 1),
                   v368,
                   v399,
                   0);
          v235 = v365;
          v242 = v241;
          v243 = v240;
          v244 = v236++;
          v245 = (char *)&v403[16 * v244];
          *(_QWORD *)v245 = v243;
          *((_DWORD *)v245 + 2) = v242;
        }
        while ( LODWORD(v398[0]) != v236 );
      }
    }
    v73 = v401;
    if ( (_WORD)v401 )
    {
      if ( (unsigned __int16)(v401 - 17) <= 0xD3u )
      {
        v74 = (unsigned __int16)(v401 - 176) <= 0x34u;
        v75 = 0;
        v76 = (unsigned __int16)v401 - 1;
        v77 = v74;
        LODWORD(v78) = v333 * word_4456340[v76];
        v73 = word_4456580[v76];
LABEL_68:
        v79 = *(__int64 **)(a1 + 64);
        LODWORD(v399) = v78;
        v80 = v73;
        BYTE4(v399) = v77;
        v354 = v79;
        if ( v74 )
          LOWORD(v81) = sub_2D43AD0(v73, v78);
        else
          LOWORD(v81) = sub_2D43050(v73, v78);
        v84 = 0;
        if ( !(_WORD)v81 )
        {
          v81 = sub_3009450(v354, v80, v75, v399, v82, v83);
          v311 = HIWORD(v81);
          v84 = v234;
        }
        HIWORD(v85) = v311;
        LOWORD(v85) = v81;
        v86 = v85;
LABEL_73:
        v87 = (__int64)v403;
        v88 = (unsigned int)v404;
        if ( (_WORD)v401 )
        {
          a2 = (unsigned __int16)(v401 - 17) < 0xD4u ? 159 : 156;
        }
        else
        {
          v359 = (__int64)v403;
          v367 = (unsigned int)v404;
          v130 = sub_30070B0((__int64)&v401);
          v87 = v359;
          v88 = v367;
          a2 = !v130 ? 156 : 159;
        }
        *((_QWORD *)&v299 + 1) = v88;
        *(_QWORD *)&v299 = v87;
        v373.m128i_i64[0] = sub_33FC220(a1, a2, v12, v86, v84, v88, v299);
        v373.m128i_i32[2] = v89;
        if ( v403 != v405 )
          _libc_free((unsigned __int64)v403);
        goto LABEL_77;
      }
    }
    else if ( sub_30070B0((__int64)&v401) )
    {
      v369 = sub_3007240((__int64)&v401);
      v135 = sub_30070B0((__int64)&v401);
      v78 = v333 * (unsigned int)v369;
      v77 = BYTE4(v369);
      if ( v135 )
      {
        v276 = sub_3009970((__int64)&v401, v78, v136, BYTE4(v369), v137);
        v77 = BYTE4(v369);
        LODWORD(v78) = v333 * v369;
        v73 = v276;
        v75 = v277;
      }
      else
      {
        v75 = v402;
      }
      v74 = BYTE4(v369);
      goto LABEL_68;
    }
    v131 = v398[0];
    v132 = v73;
    v84 = 0;
    v338 = v402;
    v360 = *(__int64 **)(a1 + 64);
    LOWORD(v133) = sub_2D43050(v132, v398[0]);
    if ( !(_WORD)v133 )
    {
      v133 = sub_3009400(v360, v132, v338, v131, 0);
      v312 = HIWORD(v133);
      v84 = v232;
    }
    HIWORD(v134) = v312;
    LOWORD(v134) = v133;
    v86 = v134;
    goto LABEL_73;
  }
LABEL_77:
  result = v373.m128i_i64[0];
  v90 = v371;
  v91 = (unsigned __int16 *)(*(_QWORD *)(v373.m128i_i64[0] + 48) + 16LL * v373.m128i_u32[2]);
  v92 = *v91;
  v93 = *((_QWORD *)v91 + 1);
  LOWORD(v374) = v92;
  v375 = v93;
  if ( (_WORD)v92 == (_WORD)v371 )
  {
    if ( v372 == v93 || (_WORD)v92 )
      return result;
  }
  else if ( (_WORD)v92 )
  {
    if ( (unsigned __int16)(v92 - 17) > 0xD3u )
      goto LABEL_80;
    v139 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v92 - 16];
    v349 = byte_444C4A0[16 * (unsigned __int16)v92 - 8];
LABEL_178:
    if ( (_WORD)v371 )
    {
      if ( (_WORD)v371 == 1 || (unsigned __int16)(v371 - 504) <= 7u )
        goto LABEL_397;
      v233 = (unsigned __int16)v371 - 1;
      if ( v139 == *(_QWORD *)&byte_444C4A0[16 * v233] && v349 == byte_444C4A0[16 * v233 + 8] )
        return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
      v142 = word_4456340[v233];
      LOBYTE(v143) = (unsigned __int16)(v371 - 176) <= 0x34u;
    }
    else
    {
      v361 = v139;
      v376 = sub_3007260((__int64)&v371);
      v377 = v141;
      if ( v361 == v376 && v349 == (_BYTE)v377 )
        return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
      v370 = sub_3007240((__int64)&v371);
      v142 = v370;
      v143 = HIDWORD(v370);
    }
    if ( (_WORD)v92 )
    {
      LODWORD(v144) = word_4456340[(unsigned __int16)v92 - 1];
      v145 = (unsigned int)(v92 - 176);
      LOBYTE(v145) = (unsigned __int16)(v92 - 176) <= 0x34u;
    }
    else
    {
      v363 = v143;
      v144 = sub_3007240((__int64)&v374);
      LOBYTE(v143) = v363;
      v145 = HIDWORD(v144);
    }
    if ( (_DWORD)v144 == v142 && (_BYTE)v145 == (_BYTE)v143 )
      return sub_33FAFB0(a1, v373.m128i_i64[0], v373.m128i_i64[1], v12, (unsigned int)v371, v372);
    if ( v90 )
    {
      v146 = word_4456340;
      LOBYTE(v145) = (unsigned __int16)(v90 - 176) <= 0x34u;
      v147 = (unsigned int)v145;
      v148 = word_4456340[v90 - 1];
    }
    else
    {
      v148 = sub_3007240((__int64)&v371);
      v147 = HIDWORD(v148);
      v145 = HIDWORD(v148);
    }
    if ( (_WORD)v92 )
    {
      v149 = (unsigned __int16)v92;
      v150 = 0;
      v151 = word_4456580[v149 - 1];
    }
    else
    {
      v339 = v145;
      v366 = v147;
      v151 = sub_3009970((__int64)&v374, v148, (__int64)v146, v145, v147);
      LOBYTE(v145) = v339;
      LOBYTE(v147) = v366;
      v150 = v264;
    }
    LODWORD(v403) = v148;
    v152 = *(__int64 **)(a1 + 64);
    v153 = v151;
    BYTE4(v403) = v147;
    if ( (_BYTE)v145 )
      v154 = sub_2D43AD0(v151, v148);
    else
      v154 = sub_2D43050(v151, v148);
    v157 = 0;
    if ( !v154 )
      v154 = sub_3009450(v152, v153, v150, (__int64)v403, v155, v156);
    v375 = v157;
    LOWORD(v374) = v154;
    *(_QWORD *)&v158 = sub_3400EE0(a1, 0, v12, 0, v155);
    v160 = sub_3406EB0(a1, 161, v12, v374, v375, v159, *(_OWORD *)&v373, v158);
    v161 = v374;
    v162 = v371;
    v373.m128i_i64[0] = v160;
    v373.m128i_i32[2] = v163;
    if ( (_WORD)v371 == (_WORD)v374 )
    {
      if ( (_WORD)v371 || v372 == v375 )
        return v373.m128i_i64[0];
      if ( !sub_3007070((__int64)&v374) )
        goto LABEL_303;
    }
    else
    {
      if ( (_WORD)v374 )
      {
        if ( (unsigned __int16)(v374 - 2) > 7u
          && (unsigned __int16)(v374 - 17) > 0x6Cu
          && (unsigned __int16)(v374 - 176) > 0x1Fu )
        {
          goto LABEL_249;
        }
      }
      else
      {
        v262 = sub_3007070((__int64)&v374);
        a6 = &v374;
        if ( !v262 )
        {
LABEL_303:
          v382 = sub_3007260((__int64)&v374);
          v226 = v382;
          v383 = v263;
          v227 = v263;
          goto LABEL_252;
        }
      }
      if ( v162 )
      {
        if ( (unsigned __int16)(v162 - 10) <= 6u
          || (unsigned __int16)(v162 - 126) <= 0x31u
          || (unsigned __int16)(v162 - 208) <= 0x14u )
        {
          return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
        }
        goto LABEL_302;
      }
    }
    if ( (unsigned __int8)sub_3007030((__int64)&v371) )
      return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
LABEL_302:
    a6 = &v374;
    if ( !v161 )
      goto LABEL_303;
LABEL_249:
    if ( v161 == 1 || (unsigned __int16)(v161 - 504) <= 7u )
      goto LABEL_397;
    v226 = *(_QWORD *)&byte_444C4A0[16 * v161 - 16];
    v227 = byte_444C4A0[16 * v161 - 8];
LABEL_252:
    if ( v162 )
    {
      if ( v162 == 1 || (unsigned __int16)(v162 - 504) <= 7u )
        goto LABEL_397;
      v231 = *(_QWORD *)&byte_444C4A0[16 * v162 - 16];
      LOBYTE(v230) = byte_444C4A0[16 * v162 - 8];
    }
    else
    {
      v228 = sub_3007260((__int64)&v371);
      v230 = v229;
      v380 = v228;
      v231 = v228;
      v381 = v230;
    }
    if ( v231 != v226 || (_BYTE)v230 != v227 )
      return sub_33FAFB0(a1, v373.m128i_i64[0], v373.m128i_i64[1], v12, (unsigned int)v371, v372);
    return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
  }
  v138 = sub_30070B0((__int64)&v374);
  LODWORD(a6) = (unsigned int)&v374;
  if ( v138 )
  {
    v378 = sub_3007260((__int64)&v374);
    v139 = v378;
    v379 = v140;
    v349 = v140;
    goto LABEL_178;
  }
LABEL_80:
  if ( (_WORD)v371 )
  {
    if ( (_WORD)v371 == 1 || (unsigned __int16)(v371 - 504) <= 7u )
      goto LABEL_397;
    v96 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v371 - 16];
    v355 = byte_444C4A0[16 * (unsigned __int16)v371 - 8];
  }
  else
  {
    v94 = sub_3007260((__int64)&v371);
    a2 = v95;
    v386 = v94;
    v96 = v94;
    v387 = v95;
    v355 = v95;
  }
  if ( (_WORD)v92 )
  {
    if ( (_WORD)v92 == 1 || (unsigned __int16)(v92 - 504) <= 7u )
      goto LABEL_397;
    v99 = *(unsigned __int16 **)&byte_444C4A0[16 * (unsigned __int16)v92 - 16];
    LOBYTE(v98) = byte_444C4A0[16 * (unsigned __int16)v92 - 8];
  }
  else
  {
    v91 = (unsigned __int16 *)sub_3007260((__int64)&v374);
    v98 = v97;
    v384 = v91;
    v99 = v91;
    v385 = v98;
  }
  if ( (unsigned __int16 *)v96 == v99 && (_BYTE)v98 == v355 )
  {
    if ( v90 )
    {
      if ( *(_QWORD *)(v328 + 8LL * v90 + 112) )
        return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
LABEL_86:
      if ( (unsigned __int16)(v90 - 176) > 0x34u )
        goto LABEL_317;
      goto LABEL_87;
    }
  }
  else if ( v90 )
  {
    goto LABEL_86;
  }
  if ( !sub_3007100((__int64)&v371) )
  {
LABEL_202:
    if ( (unsigned int)sub_3007130((__int64)&v371, a2) == 1 )
      goto LABEL_322;
    goto LABEL_203;
  }
LABEL_87:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  v90 = v371;
  if ( !(_WORD)v371 )
    goto LABEL_202;
  if ( (unsigned __int16)(v371 - 176) > 0x34u )
  {
LABEL_317:
    v102 = v90 - 1;
    if ( word_4456340[v102] == 1 )
      goto LABEL_91;
    goto LABEL_203;
  }
  sub_CA17B0(
    "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::ge"
    "tVectorElementCount() instead");
  v101 = word_4456340;
  v90 = v371;
  v102 = (unsigned __int16)v371 - 1;
  if ( word_4456340[v102] == 1 )
  {
    if ( (_WORD)v371 )
    {
LABEL_91:
      v402 = 0;
      LOWORD(v401) = word_4456580[v102];
      goto LABEL_92;
    }
LABEL_322:
    v272 = sub_3009970((__int64)&v371, a2, (__int64)v101, (__int64)v91, v100);
    v90 = v371;
    LOWORD(v401) = v272;
    v402 = v273;
    if ( !(_WORD)v371 )
    {
      if ( !sub_3007100((__int64)&v371) )
        goto LABEL_324;
      goto LABEL_343;
    }
LABEL_92:
    if ( (unsigned __int16)(v90 - 176) > 0x34u )
    {
LABEL_93:
      v103 = word_4456340[(unsigned __int16)v371 - 1];
LABEL_94:
      if ( v103 != 1 || (_WORD)v374 == (_WORD)v401 && ((_WORD)v401 || v375 == v402) )
        goto LABEL_95;
      v278 = sub_2D5B750((unsigned __int16 *)&v401);
      v404 = v279;
      v403 = (const char *)v278;
      v280 = sub_CA1930(&v403);
      v281 = sub_2D5B750((unsigned __int16 *)&v374);
      v404 = v282;
      v403 = (const char *)v281;
      if ( v280 == sub_CA1930(&v403) )
      {
        v373.m128i_i64[0] = sub_33FAF80(a1, 234, v12, v401, v402, (unsigned int)&v374);
        v373.m128i_i32[2] = v293;
        goto LABEL_95;
      }
      if ( (_WORD)v401 )
      {
        if ( (unsigned __int16)(v401 - 10) > 6u
          && (unsigned __int16)(v401 - 126) > 0x31u
          && (unsigned __int16)(v401 - 208) > 0x14u )
        {
          goto LABEL_374;
        }
      }
      else if ( !(unsigned __int8)sub_3007030((__int64)&v401) )
      {
        goto LABEL_374;
      }
      if ( (_WORD)v374 )
      {
        if ( (unsigned __int16)(v374 - 2) <= 7u
          || (unsigned __int16)(v374 - 17) <= 0x6Cu
          || (unsigned __int16)(v374 - 176) <= 0x1Fu )
        {
          goto LABEL_358;
        }
      }
      else if ( sub_3007070((__int64)&v374) )
      {
LABEL_358:
        v283 = sub_327FC40(*(_QWORD **)(a1 + 64), v280);
        v306 = v373.m128i_i32[2];
        v373.m128i_i64[0] = sub_33FAF80(a1, 216, v12, v283, v284, v285);
        v373.m128i_i32[2] = v286;
        v287 = sub_33FB890(a1, (unsigned int)v401, v402, v373.m128i_i64[0], v373.m128i_i64[1]);
        LODWORD(a6) = v306;
        v373.m128i_i64[0] = v287;
        v373.m128i_i32[2] = v288;
LABEL_95:
        *((_QWORD *)&v300 + 1) = 1;
        *(_QWORD *)&v300 = &v373;
        return sub_33FC220(a1, 156, v12, v371, v372, (_DWORD)a6, v300);
      }
LABEL_374:
      if ( (_WORD)v371 )
      {
        if ( (unsigned __int16)(v371 - 10) <= 6u
          || (unsigned __int16)(v371 - 126) <= 0x31u
          || (unsigned __int16)(v371 - 208) <= 0x14u )
        {
          goto LABEL_378;
        }
      }
      else if ( (unsigned __int8)sub_3007030((__int64)&v371) )
      {
LABEL_378:
        v289 = sub_3406EE0(a1, v373.m128i_i64[0], v373.m128i_i64[1], v12, (unsigned int)v401, v402);
        v291 = v290;
LABEL_379:
        v373.m128i_i64[0] = v289;
        v373.m128i_i32[2] = v291;
        goto LABEL_95;
      }
      v289 = sub_33FAFB0(a1, v373.m128i_i64[0], v373.m128i_i64[1], v12, (unsigned int)v401, v402);
      v291 = v292;
      goto LABEL_379;
    }
LABEL_343:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v371 )
    {
      if ( (unsigned __int16)(v371 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_93;
    }
LABEL_324:
    v103 = sub_3007130((__int64)&v371, a2);
    goto LABEL_94;
  }
LABEL_203:
  v166 = v374;
  if ( (_WORD)v374 )
  {
    if ( (_WORD)v374 == 1 || (unsigned __int16)(v374 - 504) <= 7u )
      goto LABEL_397;
    v167 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v374 - 16];
    v362 = byte_444C4A0[16 * (unsigned __int16)v374 - 8];
  }
  else
  {
    v390 = sub_3007260((__int64)&v374);
    v167 = v390;
    v391 = v168;
    v362 = v168;
  }
  v169 = v371;
  if ( (_WORD)v371 )
  {
    if ( (_WORD)v371 == 1 || (unsigned __int16)(v371 - 504) <= 7u )
      goto LABEL_397;
    v172 = *(unsigned __int16 **)&byte_444C4A0[16 * (unsigned __int16)v371 - 16];
    LOBYTE(v171) = byte_444C4A0[16 * (unsigned __int16)v371 - 8];
  }
  else
  {
    v91 = (unsigned __int16 *)sub_3007260((__int64)&v371);
    v171 = v170;
    v388 = v91;
    v172 = v91;
    v389 = v171;
  }
  if ( (unsigned __int16 *)v167 == v172 && (_BYTE)v171 == v362 )
    return sub_33FAF80(a1, 234, v12, v371, v372, (_DWORD)a6);
  if ( v169 == v166 )
  {
    if ( v169 || v372 == v375 )
      goto LABEL_210;
    v402 = v375;
    LOWORD(v401) = 0;
    goto LABEL_279;
  }
  LOWORD(v401) = v166;
  v402 = v375;
  if ( !v166 )
  {
LABEL_279:
    v394[0] = sub_3007260((__int64)&v401);
    v246 = v394[0];
    v394[1] = v247;
    v248 = v247;
    goto LABEL_280;
  }
  if ( v166 == 1 || (unsigned __int16)(v166 - 504) <= 7u )
    goto LABEL_397;
  v246 = *(_QWORD *)&byte_444C4A0[16 * v166 - 16];
  v248 = byte_444C4A0[16 * v166 - 8];
LABEL_280:
  if ( v169 )
  {
    if ( v169 == 1 || (unsigned __int16)(v169 - 504) <= 7u )
      goto LABEL_397;
    v251 = *(unsigned __int16 **)&byte_444C4A0[16 * v169 - 16];
    LOBYTE(v250) = byte_444C4A0[16 * v169 - 8];
  }
  else
  {
    v91 = (unsigned __int16 *)sub_3007260((__int64)&v371);
    v250 = v249;
    v392 = v91;
    v251 = v91;
    v393 = v250;
  }
  if ( (!(_BYTE)v250 || v248) && (unsigned __int64)v251 < v246 )
  {
    if ( v169 )
    {
      if ( (unsigned __int16)(v169 - 504) <= 7u )
        goto LABEL_397;
      v256 = *(_QWORD *)&byte_444C4A0[16 * v169 - 16];
    }
    else
    {
      v252 = sub_3007260((__int64)&v371);
      v256 = v253;
      v254 = v252;
      v255 = v256;
      v395 = v254;
      LODWORD(v256) = v254;
      v396 = v255;
    }
    switch ( (_DWORD)v256 )
    {
      case 1:
        LOWORD(v257) = 2;
        break;
      case 2:
        LOWORD(v257) = 3;
        break;
      case 4:
        LOWORD(v257) = 4;
        break;
      case 8:
        LOWORD(v257) = 5;
        break;
      case 0x10:
        LOWORD(v257) = 6;
        break;
      case 0x20:
        LOWORD(v257) = 7;
        break;
      case 0x40:
        LOWORD(v257) = 8;
        break;
      case 0x80:
        LOWORD(v257) = 9;
        break;
      default:
        v257 = sub_3007020(*(_QWORD **)(a1 + 64), v256);
        v309 = HIWORD(v257);
        v259 = v258;
LABEL_296:
        HIWORD(v260) = v309;
        LOWORD(v260) = v257;
        v373.m128i_i64[0] = sub_33FAF80(a1, 216, v12, v260, v259, (_DWORD)a6);
        v373.m128i_i32[2] = v261;
        return sub_33FB890(a1, (unsigned int)v371, v372, v373.m128i_i64[0], v373.m128i_i64[1]);
    }
    v259 = 0;
    goto LABEL_296;
  }
LABEL_210:
  v173 = *(_QWORD *)(a1 + 64);
  v403 = "non-trivial scalar-to-vector conversion";
  v405[17] = 1;
  v405[16] = 3;
  sub_33681A0(v173, v346, (__int64)&v403, (__int64)v91);
  v403 = 0;
  LODWORD(v404) = 0;
  v174 = sub_33F17F0(a1, 51, &v403, v371, v372);
  if ( v403 )
    sub_B91220((__int64)&v403, (__int64)v403);
  return v174;
}
