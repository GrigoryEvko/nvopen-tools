// Function: sub_3461D10
// Address: 0x3461d10
//
_QWORD *__fastcall sub_3461D10(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  int v5; // r14d
  unsigned int v6; // r15d
  __int64 v9; // rax
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  __int16 *v13; // rax
  unsigned __int16 v14; // dx
  __int16 v15; // si
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int8 v21; // al
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int16 v25; // cx
  __int64 v26; // r11
  __int64 v27; // r10
  __int64 v28; // rax
  __int64 (__fastcall *v29)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rax
  unsigned __int16 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  char v35; // al
  bool v36; // al
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int8 v39; // al
  unsigned int v40; // r12d
  unsigned int v41; // r15d
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int8 v45; // cl
  __int64 v46; // rax
  __int64 v47; // r9
  _QWORD *v48; // rax
  __int64 v49; // r10
  __int64 v50; // r8
  __int64 v51; // rcx
  __int64 v52; // rdx
  unsigned __int16 *v53; // rax
  unsigned __int8 *v54; // rax
  __int64 v55; // rdx
  unsigned __int8 *v56; // rax
  __int64 v57; // r10
  __int64 v58; // rdx
  __int64 v59; // rdi
  unsigned int v60; // r10d
  __int64 v61; // rcx
  int v62; // edx
  __m128i *v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // r14
  __int16 v66; // r10d^2
  __int64 v67; // rax
  __int64 v68; // r8
  unsigned __int8 v69; // al
  __m128i *v70; // rax
  __int16 v71; // r10d^2
  __int64 v72; // r8
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r9
  unsigned __int64 v76; // rdx
  __int64 *v77; // rax
  unsigned __int8 *v78; // rax
  unsigned int v79; // edx
  unsigned int v80; // r14d
  unsigned int v81; // edx
  __int64 v82; // rax
  __m128i v83; // xmm0
  __int16 v84; // si
  __int16 v85; // bx
  unsigned __int64 v86; // rdx
  char v87; // r8
  __int64 v88; // r14
  int v89; // eax
  __int64 v90; // rdx
  int v91; // eax
  __int64 v92; // r15
  unsigned int v93; // esi
  __int64 v94; // rax
  unsigned __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rbx
  __int64 v98; // rax
  __int16 v99; // si
  __int16 v100; // dx
  unsigned __int64 v101; // rdi
  __int64 v102; // rbx
  int v103; // ecx
  __m128i *v104; // rax
  unsigned __int64 v105; // rdx
  unsigned __int64 v106; // rbx
  __int64 v107; // rcx
  __int64 v108; // r8
  __int64 v109; // r9
  unsigned __int8 v110; // al
  __m128i *v111; // rax
  __int64 v112; // r9
  __m128i *v113; // rdx
  __m128i *v114; // r15
  __int64 v115; // rdx
  __m128i *v116; // r14
  unsigned __int64 v117; // r8
  __m128i **v118; // rdx
  unsigned __int8 *v119; // r14
  __int64 v120; // rdx
  __int64 v121; // r15
  __m128i *v122; // rax
  _OWORD *v123; // rdi
  unsigned int v124; // edx
  char v126; // r8
  __int64 v127; // rbx
  int v128; // eax
  __m128i *v129; // rax
  __m128i *v130; // r14
  int v131; // r9d
  unsigned __int8 *v132; // rax
  unsigned __int64 v133; // rdx
  int v134; // r9d
  unsigned __int8 *v135; // r10
  unsigned __int64 v136; // r11
  __int64 v137; // rsi
  unsigned int v138; // edx
  __int64 v139; // rax
  __int64 v140; // rcx
  int v141; // eax
  __int64 v142; // rbx
  unsigned int v143; // r12d
  char v144; // bl
  __int16 v145; // dx
  __int64 v146; // rax
  unsigned int v147; // edx
  int v148; // eax
  __int64 v149; // r14
  __int64 v150; // rax
  int v151; // eax
  __int64 v152; // rcx
  unsigned __int8 *v153; // rax
  unsigned int v154; // edx
  __int16 v155; // dx
  __int64 v156; // rax
  __m128i *v157; // rbx
  unsigned int v158; // edx
  unsigned int v159; // edx
  unsigned __int8 *v160; // r8
  __int16 v161; // cx
  __int64 v162; // rax
  __int16 v163; // r9
  __int16 v164; // si
  unsigned __int64 v165; // rdi
  __int32 v166; // ecx
  __int64 v167; // r14
  unsigned int v168; // edx
  __int128 v169; // rax
  __int64 v170; // r9
  __int128 v171; // rax
  __int64 v172; // r15
  __int64 v173; // r9
  unsigned __int8 *v174; // r14
  unsigned int v175; // edx
  unsigned __int64 v176; // r15
  __int64 v177; // r9
  unsigned __int8 *v178; // rax
  __int64 v179; // rdx
  __int64 v180; // r8
  unsigned int v181; // r14d
  __int64 v182; // rax
  __int16 v183; // cx
  __int16 v184; // si
  unsigned __int64 v185; // rdi
  int v186; // edx
  __int64 v187; // rdi
  unsigned int v188; // edx
  __int64 v189; // r11
  __int8 v190; // r10
  __int64 v191; // r10
  char v192; // r9
  __int64 v193; // rax
  __int64 v194; // rax
  int v195; // edx
  __int64 (__fastcall *v196)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v197; // rcx
  __int64 v198; // rdx
  unsigned __int64 v199; // rdx
  char v200; // al
  __int64 v201; // rcx
  __int64 *v202; // rdx
  unsigned int v203; // edx
  __int64 v204; // rax
  unsigned __int64 v205; // rdx
  __int128 v206; // [rsp-50h] [rbp-3F0h]
  __int128 v207; // [rsp-10h] [rbp-3B0h]
  __int128 v208; // [rsp-10h] [rbp-3B0h]
  __int128 v209; // [rsp+0h] [rbp-3A0h]
  int v210; // [rsp+0h] [rbp-3A0h]
  __int128 v211; // [rsp+0h] [rbp-3A0h]
  __int128 v212; // [rsp+0h] [rbp-3A0h]
  __int64 v213; // [rsp+10h] [rbp-390h]
  unsigned __int64 v214; // [rsp+18h] [rbp-388h]
  __int64 v215; // [rsp+20h] [rbp-380h]
  __int64 v216; // [rsp+28h] [rbp-378h]
  int v217; // [rsp+30h] [rbp-370h]
  __int64 v219; // [rsp+50h] [rbp-350h]
  __int64 v220; // [rsp+58h] [rbp-348h]
  __int64 v221; // [rsp+68h] [rbp-338h]
  __int64 v222; // [rsp+70h] [rbp-330h]
  unsigned int v223; // [rsp+7Ch] [rbp-324h]
  __int64 v224; // [rsp+90h] [rbp-310h]
  unsigned int v225; // [rsp+98h] [rbp-308h]
  unsigned __int16 v226; // [rsp+9Eh] [rbp-302h]
  __int64 v228; // [rsp+A8h] [rbp-2F8h]
  __int64 v229; // [rsp+B0h] [rbp-2F0h]
  unsigned int v230; // [rsp+B0h] [rbp-2F0h]
  __int64 v231; // [rsp+B0h] [rbp-2F0h]
  int v232; // [rsp+B0h] [rbp-2F0h]
  unsigned __int64 v233; // [rsp+B0h] [rbp-2F0h]
  __int16 v234; // [rsp+B2h] [rbp-2EEh]
  __int16 v235; // [rsp+B2h] [rbp-2EEh]
  unsigned int v236; // [rsp+B8h] [rbp-2E8h]
  __int64 v237; // [rsp+C0h] [rbp-2E0h]
  unsigned __int64 v238; // [rsp+C0h] [rbp-2E0h]
  __int64 v239; // [rsp+C0h] [rbp-2E0h]
  unsigned __int8 *v240; // [rsp+C0h] [rbp-2E0h]
  __int64 v241; // [rsp+C8h] [rbp-2D8h]
  unsigned __int16 v242; // [rsp+D0h] [rbp-2D0h]
  __int64 v243; // [rsp+D0h] [rbp-2D0h]
  __int16 v244; // [rsp+D0h] [rbp-2D0h]
  char v245; // [rsp+D0h] [rbp-2D0h]
  __int64 v246; // [rsp+D8h] [rbp-2C8h]
  int v247; // [rsp+D8h] [rbp-2C8h]
  int v248; // [rsp+D8h] [rbp-2C8h]
  unsigned __int64 v249; // [rsp+D8h] [rbp-2C8h]
  __m128i *v250; // [rsp+D8h] [rbp-2C8h]
  unsigned __int64 v251; // [rsp+D8h] [rbp-2C8h]
  unsigned int v252; // [rsp+E0h] [rbp-2C0h]
  unsigned __int64 v253; // [rsp+E0h] [rbp-2C0h]
  unsigned int v254; // [rsp+E0h] [rbp-2C0h]
  __int64 v255; // [rsp+E0h] [rbp-2C0h]
  __int64 v256; // [rsp+E8h] [rbp-2B8h]
  __int16 v257; // [rsp+F0h] [rbp-2B0h]
  __int64 v258; // [rsp+F0h] [rbp-2B0h]
  unsigned __int8 *v259; // [rsp+F0h] [rbp-2B0h]
  __int64 v260; // [rsp+F8h] [rbp-2A8h]
  __int64 v261; // [rsp+100h] [rbp-2A0h]
  __int64 v262; // [rsp+100h] [rbp-2A0h]
  __int64 v263; // [rsp+100h] [rbp-2A0h]
  unsigned __int64 v264; // [rsp+108h] [rbp-298h]
  unsigned __int64 v265; // [rsp+108h] [rbp-298h]
  __int64 v266; // [rsp+108h] [rbp-298h]
  __m128i v267; // [rsp+110h] [rbp-290h]
  unsigned __int64 v268; // [rsp+118h] [rbp-288h]
  unsigned __int64 v269; // [rsp+118h] [rbp-288h]
  unsigned int v270; // [rsp+190h] [rbp-210h] BYREF
  __int64 v271; // [rsp+198h] [rbp-208h]
  __int64 v272; // [rsp+1A0h] [rbp-200h] BYREF
  __int64 v273; // [rsp+1A8h] [rbp-1F8h]
  __int64 v274; // [rsp+1B0h] [rbp-1F0h] BYREF
  int v275; // [rsp+1B8h] [rbp-1E8h]
  __int64 v276; // [rsp+1C0h] [rbp-1E0h]
  __int64 v277; // [rsp+1C8h] [rbp-1D8h]
  __int64 v278; // [rsp+1D0h] [rbp-1D0h]
  __int64 v279; // [rsp+1D8h] [rbp-1C8h]
  __int64 v280; // [rsp+1E0h] [rbp-1C0h]
  __int64 v281; // [rsp+1E8h] [rbp-1B8h]
  __int64 v282; // [rsp+1F0h] [rbp-1B0h]
  __int64 v283; // [rsp+1F8h] [rbp-1A8h]
  __int128 v284; // [rsp+200h] [rbp-1A0h]
  __int64 v285; // [rsp+210h] [rbp-190h]
  __m128i v286; // [rsp+220h] [rbp-180h] BYREF
  __int64 v287; // [rsp+230h] [rbp-170h]
  __int128 v288; // [rsp+240h] [rbp-160h]
  __int64 v289; // [rsp+250h] [rbp-150h]
  __m128i v290; // [rsp+260h] [rbp-140h] BYREF
  __int64 v291; // [rsp+270h] [rbp-130h]
  __int128 v292; // [rsp+280h] [rbp-120h] BYREF
  __int64 v293; // [rsp+290h] [rbp-110h]
  __m128i v294; // [rsp+2A0h] [rbp-100h] BYREF
  __int64 v295; // [rsp+2B0h] [rbp-F0h]
  __m128i v296; // [rsp+2C0h] [rbp-E0h] BYREF
  __m128i v297; // [rsp+2D0h] [rbp-D0h]
  __m128i v298; // [rsp+2E0h] [rbp-C0h] BYREF
  _OWORD v299[11]; // [rsp+2F0h] [rbp-B0h] BYREF

  v9 = *(_QWORD *)(a3 + 40);
  v11 = _mm_loadu_si128((const __m128i *)v9);
  v12 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v237 = *(_QWORD *)(v9 + 40);
  v252 = *(_DWORD *)(v9 + 48);
  v13 = *(__int16 **)(a3 + 48);
  v14 = *(_WORD *)(a3 + 96);
  v15 = *v13;
  v16 = *((_QWORD *)v13 + 1);
  v267 = v12;
  LOWORD(v272) = v14;
  v257 = v15;
  v246 = v16;
  LOWORD(v270) = v15;
  v17 = *(_QWORD *)(a3 + 80);
  v271 = v16;
  v18 = *(_QWORD *)(a3 + 104);
  v274 = v17;
  v273 = v18;
  if ( v17 )
  {
    sub_B96E90((__int64)&v274, v17, 1);
    v14 = v272;
  }
  v275 = *(_DWORD *)(a3 + 72);
  if ( v257 )
  {
    if ( (unsigned __int16)(v257 - 10) <= 0xDAu )
      goto LABEL_5;
LABEL_28:
    if ( v14 )
    {
      if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
        goto LABEL_169;
      v139 = 16LL * (v14 - 1);
      v38 = *(_QWORD *)&byte_444C4A0[v139];
      v39 = byte_444C4A0[v139 + 8];
    }
    else
    {
      v280 = sub_3007260((__int64)&v272);
      v281 = v37;
      v38 = v280;
      v39 = v281;
    }
    v298.m128i_i8[8] = v39;
    v298.m128i_i64[0] = v38;
    v40 = sub_CA1930(&v298);
    v41 = v40 >> 1;
    if ( v40 >> 1 == 1 )
    {
      LOWORD(v42) = 2;
    }
    else
    {
      switch ( v41 )
      {
        case 2u:
          LOWORD(v42) = 3;
          break;
        case 4u:
          LOWORD(v42) = 4;
          break;
        case 8u:
          LOWORD(v42) = 5;
          break;
        case 0x10u:
          LOWORD(v42) = 6;
          break;
        case 0x20u:
          LOWORD(v42) = 7;
          break;
        case 0x40u:
          LOWORD(v42) = 8;
          break;
        case 0x80u:
          LOWORD(v42) = 9;
          break;
        default:
          v42 = sub_3007020(*(_QWORD **)(a4 + 64), v41);
          v229 = v42;
          goto LABEL_106;
      }
    }
    v43 = 0;
LABEL_106:
    v142 = v229;
    v143 = v40 >> 4;
    v256 = v43;
    LOWORD(v142) = v42;
    v255 = v142;
    v245 = *(_BYTE *)(*(_QWORD *)(a3 + 112) + 34LL);
    v144 = (*(_BYTE *)(a3 + 33) >> 2) & 3;
    if ( !v144 )
      v144 = 3;
    if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a4 + 40)) )
    {
      HIBYTE(v155) = 1;
      LOBYTE(v155) = v245;
      v156 = *(_QWORD *)(a3 + 112);
      v298 = _mm_loadu_si128((const __m128i *)(v156 + 40));
      v299[0] = _mm_loadu_si128((const __m128i *)(v156 + 56));
      v157 = sub_33F1DB0(
               (__int64 *)a4,
               v144,
               (__int64)&v274,
               v270,
               v271,
               v155,
               *(_OWORD *)&v11,
               v267.m128i_i64[0],
               v267.m128i_i64[1],
               *(_OWORD *)v156,
               *(_QWORD *)(v156 + 16),
               v255,
               v256,
               *(_WORD *)(v156 + 32),
               (__int64)&v298);
      v298.m128i_i8[8] = 0;
      v260 = v158;
      v298.m128i_i64[0] = v143;
      v160 = sub_3409320((_QWORD *)a4, v267.m128i_i64[0], v267.m128i_i64[1], v143, 0, (__int64)&v274, a5, 1);
      LOBYTE(v161) = v245;
      v162 = *(_QWORD *)(a3 + 112);
      HIBYTE(v161) = 1;
      v163 = v161;
      v298 = _mm_loadu_si128((const __m128i *)(v162 + 40));
      v299[0] = _mm_loadu_si128((const __m128i *)(v162 + 56));
      v164 = *(_WORD *)(v162 + 32);
      v165 = *(_QWORD *)v162 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v165 )
      {
        v189 = *(_QWORD *)(v162 + 8) + v143;
        v190 = *(_BYTE *)(v162 + 20);
        if ( (*(_QWORD *)v162 & 4) != 0 )
        {
          v296.m128i_i64[1] = *(_QWORD *)(v162 + 8) + v143;
          v297.m128i_i8[4] = v190;
          v296.m128i_i64[0] = v165 | 4;
          v297.m128i_i32[0] = *(_DWORD *)(v165 + 12);
        }
        else
        {
          v296.m128i_i64[0] = *(_QWORD *)v162 & 0xFFFFFFFFFFFFFFF8LL;
          v296.m128i_i64[1] = v189;
          v297.m128i_i8[4] = v190;
          v193 = *(_QWORD *)(v165 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v193 + 8) - 17 <= 1 )
            v193 = **(_QWORD **)(v193 + 16);
          v297.m128i_i32[0] = *(_DWORD *)(v193 + 8) >> 8;
        }
      }
      else
      {
        v166 = *(_DWORD *)(v162 + 16);
        v167 = *(_QWORD *)(v162 + 8) + v143;
        v297.m128i_i8[4] = 0;
        v296.m128i_i64[0] = 0;
        v296.m128i_i64[1] = v167;
        v297.m128i_i32[0] = v166;
      }
      v250 = sub_33F1DB0(
               (__int64 *)a4,
               3,
               (__int64)&v274,
               v270,
               v271,
               v163,
               *(_OWORD *)&v11,
               (__int64)v160,
               v159 | v267.m128i_i64[1] & 0xFFFFFFFF00000000LL,
               *(_OWORD *)&v296,
               v297.m128i_i64[0],
               v255,
               v256,
               v164,
               (__int64)&v298);
      v266 = v168;
      goto LABEL_121;
    }
    HIBYTE(v145) = 1;
    LOBYTE(v145) = v245;
    v146 = *(_QWORD *)(a3 + 112);
    v298 = _mm_loadu_si128((const __m128i *)(v146 + 40));
    v299[0] = _mm_loadu_si128((const __m128i *)(v146 + 56));
    v250 = sub_33F1DB0(
             (__int64 *)a4,
             3,
             (__int64)&v274,
             v270,
             v271,
             v145,
             *(_OWORD *)&v11,
             v267.m128i_i64[0],
             v267.m128i_i64[1],
             *(_OWORD *)v146,
             *(_QWORD *)(v146 + 16),
             v255,
             v256,
             *(_WORD *)(v146 + 32),
             (__int64)&v298);
    v266 = v147;
    if ( *(_DWORD *)(v237 + 24) == 56
      && ((v148 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v237 + 40) + 40LL) + 24LL), v148 == 11) || v148 == 35) )
    {
      v231 = *(_QWORD *)(*(_QWORD *)(v237 + 40) + 40LL);
      v296.m128i_i8[8] = 0;
      v296.m128i_i64[0] = v143;
      v149 = sub_CA1930(&v296);
      v150 = *(_QWORD *)(v231 + 96);
      v298.m128i_i32[2] = *(_DWORD *)(v150 + 32);
      if ( v298.m128i_i32[2] > 0x40u )
        sub_C43780((__int64)&v298, (const void **)(v150 + 24));
      else
        v298.m128i_i64[0] = *(_QWORD *)(v150 + 24);
      sub_C46A40((__int64)&v298, v149);
      v232 = *(_DWORD *)(v237 + 28);
      v151 = v232 & 1;
      if ( v298.m128i_i32[2] > 0x40u )
      {
        v201 = *(_QWORD *)v298.m128i_i64[0];
        LOBYTE(v277) = 0;
        v202 = *(__int64 **)(v237 + 40);
        v233 = v298.m128i_i64[0];
        v276 = v201;
        v240 = sub_3409320((_QWORD *)a4, *v202, v202[1], v201, 0, (__int64)&v274, a5, v151);
        v181 = v203;
        j_j___libc_free_0_0(v233);
        v180 = (__int64)v240;
        goto LABEL_124;
      }
      v152 = 0;
      if ( v298.m128i_i32[2] )
        v152 = v298.m128i_i64[0] << (64 - v298.m128i_i8[8]) >> (64 - v298.m128i_i8[8]);
      LOBYTE(v277) = 0;
      v276 = v152;
      v153 = sub_3409320(
               (_QWORD *)a4,
               **(_QWORD **)(v237 + 40),
               *(_QWORD *)(*(_QWORD *)(v237 + 40) + 8LL),
               v152,
               0,
               (__int64)&v274,
               a5,
               v232 & 1);
    }
    else
    {
      v298.m128i_i8[8] = 0;
      v298.m128i_i64[0] = v143;
      v153 = sub_3409320((_QWORD *)a4, v267.m128i_i64[0], v267.m128i_i64[1], v143, 0, (__int64)&v274, a5, 1);
    }
    v180 = (__int64)v153;
    v181 = v154;
LABEL_124:
    LOBYTE(v183) = v245;
    v182 = *(_QWORD *)(a3 + 112);
    HIBYTE(v183) = 1;
    v298 = _mm_loadu_si128((const __m128i *)(v182 + 40));
    v299[0] = _mm_loadu_si128((const __m128i *)(v182 + 56));
    v184 = *(_WORD *)(v182 + 32);
    v185 = *(_QWORD *)v182 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v185 )
    {
      v191 = *(_QWORD *)(v182 + 8) + v143;
      v192 = *(_BYTE *)(v182 + 20);
      if ( (*(_QWORD *)v182 & 4) != 0 )
      {
        v294.m128i_i64[1] = *(_QWORD *)(v182 + 8) + v143;
        BYTE4(v295) = v192;
        v294.m128i_i64[0] = v185 | 4;
        LODWORD(v295) = *(_DWORD *)(v185 + 12);
      }
      else
      {
        v194 = *(_QWORD *)(v185 + 8);
        v294.m128i_i64[0] = v185;
        v294.m128i_i64[1] = v191;
        v195 = *(unsigned __int8 *)(v194 + 8);
        BYTE4(v295) = v192;
        if ( (unsigned int)(v195 - 17) <= 1 )
          v194 = **(_QWORD **)(v194 + 16);
        LODWORD(v295) = *(_DWORD *)(v194 + 8) >> 8;
      }
    }
    else
    {
      v186 = *(_DWORD *)(v182 + 16);
      v294.m128i_i64[0] = 0;
      v187 = *(_QWORD *)(v182 + 8) + v143;
      BYTE4(v295) = 0;
      v294.m128i_i64[1] = v187;
      LODWORD(v295) = v186;
    }
    v157 = sub_33F1DB0(
             (__int64 *)a4,
             v144,
             (__int64)&v274,
             v270,
             v271,
             v183,
             *(_OWORD *)&v11,
             v180,
             v181 | v267.m128i_i64[1] & 0xFFFFFFFF00000000LL,
             *(_OWORD *)&v294,
             v295,
             v255,
             v256,
             v184,
             (__int64)&v298);
    v260 = v188;
LABEL_121:
    *(_QWORD *)&v169 = sub_3400E40(a4, v41, v270, v271, (__int64)&v274, a5);
    *((_QWORD *)&v207 + 1) = v260;
    *(_QWORD *)&v207 = v157;
    *(_QWORD *)&v171 = sub_3406EB0((_QWORD *)a4, 0xBEu, (__int64)&v274, v270, v271, v170, v207, v169);
    *((_QWORD *)&v211 + 1) = v266;
    v172 = *((_QWORD *)&v171 + 1);
    *(_QWORD *)&v211 = v250;
    v174 = sub_3406EB0((_QWORD *)a4, 0xBBu, (__int64)&v274, v270, v271, v173, v171, v211);
    v176 = v175 | v172 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v212 + 1) = 1;
    *(_QWORD *)&v212 = v157;
    *((_QWORD *)&v208 + 1) = 1;
    *(_QWORD *)&v208 = v250;
    v178 = sub_3406EB0((_QWORD *)a4, 2u, (__int64)&v274, 1, 0, v177, v208, v212);
    *a1 = v174;
    a1[1] = v176;
    a1[2] = v178;
    a1[3] = v179;
    goto LABEL_79;
  }
  v242 = v14;
  v35 = sub_3007030((__int64)&v270);
  v14 = v242;
  if ( !v35 )
  {
    v36 = sub_30070B0((__int64)&v270);
    v14 = v242;
    if ( !v36 )
      goto LABEL_28;
  }
LABEL_5:
  v224 = *(_QWORD *)(a4 + 40);
  if ( !v14 )
  {
    v278 = sub_3007260((__int64)&v272);
    v279 = v19;
    v20 = v278;
    v21 = v279;
    goto LABEL_7;
  }
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
LABEL_169:
    BUG();
  v96 = 16LL * (v14 - 1);
  v20 = *(_QWORD *)&byte_444C4A0[v96];
  v21 = byte_444C4A0[v96 + 8];
LABEL_7:
  v298.m128i_i8[8] = v21;
  v298.m128i_i64[0] = v20;
  v22 = sub_CA1930(&v298);
  switch ( v22 )
  {
    case 1u:
      LODWORD(v23) = 2;
      v25 = 2;
LABEL_40:
      LODWORD(v27) = v25;
      v26 = 0;
LABEL_41:
      v32 = v272;
      if ( !*(_QWORD *)(a2 + 8LL * (int)v23 + 112)
        || !(_WORD)v272
        || !*(_QWORD *)(a2 + 8LL * (unsigned __int16)v272 + 112) )
      {
        v226 = *(_WORD *)(a2 + 2LL * ((int)v23 + 1426));
        goto LABEL_44;
      }
      LOWORD(v27) = v25;
      v244 = v272;
      v254 = v27;
      v263 = v26;
      if ( !(unsigned __int8)sub_328A020(a2, 0x12Au, v25, v26, 0) && (unsigned __int16)(v244 - 17) <= 0xD3u )
      {
        sub_3460140((__int64)a1, a5, a2, a3, a4);
        goto LABEL_79;
      }
      v129 = sub_33F1A60(
               (__int64 *)a4,
               v254,
               v263,
               (__int64)&v274,
               v11.m128i_i64[0],
               v11.m128i_i64[1],
               v267.m128i_i64[0],
               v267.m128i_i64[1],
               *(const __m128i **)(a3 + 112));
      v210 = (int)v129;
      v130 = v129;
      v132 = sub_33FAF80(a4, 234, (__int64)&v274, (unsigned int)v272, v273, v131, a5);
      v134 = v210;
      v135 = v132;
      v136 = v133;
      if ( (_WORD)v272 == v257 )
      {
        if ( v257 || v273 == v246 )
          goto LABEL_135;
      }
      else if ( v257 )
      {
        if ( (unsigned __int16)(v257 - 10) <= 6u || (unsigned __int16)(v257 - 126) <= 0x31u )
          v137 = 233;
        else
          v137 = (unsigned __int16)(v257 - 208) < 0x15u ? 233 : 215;
        goto LABEL_93;
      }
      v269 = v133;
      v200 = sub_3007030((__int64)&v270);
      v136 = v269;
      v137 = v200 == 0 ? 215 : 233;
LABEL_93:
      v268 = v136;
      v135 = sub_33FAF80(a4, v137, (__int64)&v274, v270, v271, v134, a5);
      v136 = v138 | v268 & 0xFFFFFFFF00000000LL;
LABEL_135:
      *a1 = v135;
      a1[1] = v136;
      a1[2] = v130;
      *((_DWORD *)a1 + 6) = 1;
      goto LABEL_79;
    case 2u:
      LODWORD(v23) = 3;
      v25 = 3;
      goto LABEL_40;
    case 4u:
      LODWORD(v23) = 4;
      v25 = 4;
      goto LABEL_40;
    case 8u:
      LODWORD(v23) = 5;
      v25 = 5;
      goto LABEL_40;
    case 0x10u:
      LODWORD(v23) = 6;
      v25 = 6;
      goto LABEL_40;
    case 0x20u:
      LODWORD(v23) = 7;
      v25 = 7;
      goto LABEL_40;
    case 0x40u:
      LODWORD(v23) = 8;
      v25 = 8;
      goto LABEL_40;
    case 0x80u:
      LODWORD(v23) = 9;
      v25 = 9;
      goto LABEL_40;
  }
  v23 = sub_3007020(*(_QWORD **)(a4 + 64), v22);
  v25 = v23;
  v26 = v24;
  v27 = v23;
  if ( (_WORD)v23 )
  {
    LODWORD(v23) = (unsigned __int16)v23;
    goto LABEL_41;
  }
  v28 = *(_QWORD *)(a4 + 64);
  LOWORD(v27) = 0;
  v294.m128i_i64[1] = v24;
  v258 = v28;
  v294.m128i_i64[0] = v27;
  if ( sub_30070B0((__int64)&v294) )
  {
    v298.m128i_i16[0] = 0;
    v298.m128i_i64[1] = 0;
    LOWORD(v292) = 0;
    sub_2FE8D10(
      a2,
      v258,
      v294.m128i_u32[0],
      v294.m128i_u64[1],
      v298.m128i_i64,
      (unsigned int *)&v296,
      (unsigned __int16 *)&v292);
    v32 = v272;
    v226 = v292;
LABEL_44:
    if ( v32 )
      goto LABEL_23;
LABEL_45:
    v33 = sub_3007260((__int64)&v272);
    v282 = v33;
    v283 = v34;
    goto LABEL_46;
  }
  if ( !sub_3007070((__int64)&v294) )
    goto LABEL_170;
  v29 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a2 + 592LL);
  if ( v29 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v298, a2, v258, v294.m128i_i64[0], v294.m128i_i64[1]);
    v30 = *(_QWORD *)&v299[0];
    v31 = v298.m128i_u16[4];
  }
  else
  {
    v31 = v29(a2, v258, v294.m128i_u32[0], v294.m128i_i64[1]);
    v30 = v199;
  }
  v296.m128i_i64[0] = v31;
  v296.m128i_i64[1] = v30;
  if ( (_WORD)v31 )
  {
    v226 = *(_WORD *)(a2 + 2LL * (unsigned __int16)v31 + 2852);
    goto LABEL_22;
  }
  v251 = v30;
  if ( sub_30070B0((__int64)&v296) )
  {
    v298.m128i_i16[0] = 0;
    v290.m128i_i16[0] = 0;
    v298.m128i_i64[1] = 0;
    sub_2FE8D10(a2, v258, v296.m128i_u32[0], v251, v298.m128i_i64, (unsigned int *)&v292, (unsigned __int16 *)&v290);
    v226 = v290.m128i_i16[0];
    goto LABEL_22;
  }
  if ( !sub_3007070((__int64)&v296) )
LABEL_170:
    BUG();
  v196 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a2 + 592LL);
  if ( v196 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v298, a2, v258, v296.m128i_i64[0], v296.m128i_i64[1]);
    v197 = *(_QWORD *)&v299[0];
    v198 = v298.m128i_u16[4];
  }
  else
  {
    v204 = v196(a2, v258, v296.m128i_u32[0], v251);
    v197 = v205;
    v198 = v204;
  }
  v226 = sub_2FE98B0(a2, v258, v198, v197);
LABEL_22:
  v32 = v272;
  if ( !(_WORD)v272 )
    goto LABEL_45;
LABEL_23:
  if ( v32 == 1 || (unsigned __int16)(v32 - 504) <= 7u )
    goto LABEL_169;
  v33 = *(_QWORD *)&byte_444C4A0[16 * v32 - 16];
  LOBYTE(v34) = byte_444C4A0[16 * v32 - 8];
LABEL_46:
  v298.m128i_i8[8] = v34;
  v298.m128i_i64[0] = (unsigned __int64)(v33 + 7) >> 3;
  v217 = sub_CA1930(&v298);
  v247 = v217;
  if ( v226 <= 1u || (unsigned __int16)(v226 - 504) <= 7u )
    goto LABEL_169;
  v44 = 16LL * (v226 - 1);
  v45 = byte_444C4A0[v44 + 8];
  v46 = *(_QWORD *)&byte_444C4A0[v44];
  v298.m128i_i8[8] = v45;
  v298.m128i_i64[0] = v46;
  v214 = (unsigned __int64)sub_CA1930(&v298) >> 3;
  v223 = (v217 + (int)v214 - 1) / (unsigned int)v214;
  v48 = sub_33EE0D0(a4, (unsigned int)v272, v273, v226, 0, v47);
  v49 = v252;
  v215 = (__int64)v48;
  v225 = *((_DWORD *)v48 + 24);
  v298.m128i_i64[1] = 0x800000000LL;
  v259 = (unsigned __int8 *)v48;
  v298.m128i_i64[0] = (__int64)v299;
  v50 = *(_QWORD *)(*(_QWORD *)(v237 + 48) + 16LL * v252 + 8);
  v51 = *(unsigned __int16 *)(*(_QWORD *)(v237 + 48) + 16LL * v252);
  v253 = (unsigned int)v52;
  v216 = v52;
  v53 = (unsigned __int16 *)(16LL * (unsigned int)v52 + v48[6]);
  v228 = v49;
  v243 = *((_QWORD *)v53 + 1);
  v230 = *v53;
  v54 = sub_3400BD0(a4, (unsigned int)v214, (__int64)&v274, v51, v50, 0, a5, 0);
  v222 = v55;
  v220 = (__int64)v54;
  v56 = sub_3400BD0(a4, (unsigned int)v214, (__int64)&v274, v230, v243, 0, a5, 0);
  v57 = v228;
  v221 = v58;
  if ( v223 <= 1 )
  {
    v92 = 0;
  }
  else
  {
    v248 = 1;
    v59 = v228;
    HIWORD(v60) = HIWORD(v6);
    HIWORD(v6) = HIWORD(v5);
    v236 = 0;
    v219 = (__int64)v56;
    while ( 1 )
    {
      v82 = *(_QWORD *)(a3 + 112);
      v83 = _mm_loadu_si128((const __m128i *)(v82 + 40));
      v296 = v83;
      v297 = _mm_loadu_si128((const __m128i *)(v82 + 56));
      LOBYTE(v85) = *(_BYTE *)(v82 + 34);
      v84 = *(_WORD *)(v82 + 32);
      HIBYTE(v85) = 1;
      v86 = *(_QWORD *)v82 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v86 )
      {
        v87 = *(_BYTE *)(v82 + 20);
        v88 = v236 + *(_QWORD *)(v82 + 8);
        if ( (*(_QWORD *)v82 & 4) != 0 )
        {
          v89 = *(_DWORD *)(v86 + 12);
          *((_QWORD *)&v284 + 1) = v88;
          BYTE4(v285) = v87;
          *(_QWORD *)&v284 = v86 | 4;
          LODWORD(v285) = v89;
        }
        else
        {
          *(_QWORD *)&v284 = *(_QWORD *)v82 & 0xFFFFFFFFFFFFFFF8LL;
          v90 = *(_QWORD *)(v86 + 8);
          *((_QWORD *)&v284 + 1) = v88;
          v91 = *(unsigned __int8 *)(v90 + 8);
          BYTE4(v285) = v87;
          if ( (unsigned int)(v91 - 17) <= 1 )
            v90 = **(_QWORD **)(v90 + 16);
          LODWORD(v285) = *(_DWORD *)(v90 + 8) >> 8;
        }
      }
      else
      {
        v61 = *(_QWORD *)(v82 + 8);
        v62 = *(_DWORD *)(v82 + 16);
        *(_QWORD *)&v284 = 0;
        BYTE4(v285) = 0;
        LODWORD(v285) = v62;
        *((_QWORD *)&v284 + 1) = v236 + v61;
      }
      LOWORD(v60) = v226;
      v267.m128i_i64[0] = v237;
      v267.m128i_i64[1] = v59 | v267.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v63 = sub_33F1F00(
              (__int64 *)a4,
              v60,
              0,
              (__int64)&v274,
              v11.m128i_i64[0],
              v11.m128i_i64[1],
              v237,
              v267.m128i_i64[1],
              v284,
              v285,
              v85,
              v84,
              (__int64)&v296,
              0);
      v238 = v64;
      v65 = (unsigned __int64)v63;
      v234 = v66;
      v296 = 0u;
      v297 = 0u;
      sub_2EAC300((__int64)&v286, v224, v225, v236);
      v294 = _mm_load_si128(&v286);
      v264 = v253 | v264 & 0xFFFFFFFF00000000LL;
      v295 = v287;
      v67 = *(_QWORD *)(v65 + 48) + 16LL * (unsigned int)v238;
      LOWORD(v6) = *(_WORD *)v67;
      v69 = sub_33CC4A0(a4, v6, *(_QWORD *)(v67 + 8), 0xFFFFFFFF00000000LL, v68, v238);
      v70 = sub_33F4560(
              (_QWORD *)a4,
              v65,
              1u,
              (__int64)&v274,
              v65,
              v238,
              (unsigned __int64)v259,
              v264,
              *(_OWORD *)&v294,
              v295,
              v69,
              0,
              (__int64)&v296);
      v71 = v234;
      v72 = (__int64)v70;
      v73 = v298.m128i_u32[2];
      v75 = v74;
      v76 = v298.m128i_u32[2] + 1LL;
      if ( v76 > v298.m128i_u32[3] )
      {
        v239 = v72;
        v241 = v75;
        sub_C8D5F0((__int64)&v298, v299, v76, 0x10u, v72, v75);
        v73 = v298.m128i_u32[2];
        v71 = v234;
        v72 = v239;
        v75 = v241;
      }
      v77 = (__int64 *)(v298.m128i_i64[0] + 16 * v73);
      *v77 = v72;
      v77[1] = v75;
      v236 += v214;
      v235 = v71;
      ++v298.m128i_i32[2];
      v78 = sub_34092D0((_QWORD *)a4, v267.m128i_i64[0], v267.m128i_i64[1], v220, v222, (__int64)&v274, v83, 1);
      v80 = v79;
      v237 = (__int64)v78;
      ++v248;
      v259 = sub_34092D0((_QWORD *)a4, (__int64)v259, v253, v219, v221, (__int64)&v274, v83, 1);
      if ( v223 == v248 )
        break;
      HIWORD(v60) = v235;
      v59 = v80;
      v253 = v81;
    }
    v57 = v80;
    v92 = (unsigned int)v214 * (v223 - 1);
    v247 = v217 - v214 * (v223 - 1);
    v253 = v81;
  }
  v93 = 8 * v247;
  if ( 8 * v247 == 8 )
  {
    LOWORD(v94) = 5;
LABEL_72:
    v95 = 0;
    goto LABEL_73;
  }
  switch ( v93 )
  {
    case 0x10u:
      LOWORD(v94) = 6;
      goto LABEL_72;
    case 0x20u:
      LOWORD(v94) = 7;
      goto LABEL_72;
    case 0x40u:
      LOWORD(v94) = 8;
      goto LABEL_72;
    case 0x80u:
      LOWORD(v94) = 9;
      goto LABEL_72;
  }
  v261 = v57;
  v94 = sub_3007020(*(_QWORD **)(a4 + 64), v93);
  v57 = v261;
  v213 = v94;
LABEL_73:
  v97 = v213;
  v265 = v95;
  LOWORD(v97) = v94;
  v262 = v97;
  v98 = *(_QWORD *)(a3 + 112);
  v296 = _mm_loadu_si128((const __m128i *)(v98 + 40));
  v297 = _mm_loadu_si128((const __m128i *)(v98 + 56));
  LOBYTE(v100) = *(_BYTE *)(v98 + 34);
  v99 = *(_WORD *)(v98 + 32);
  HIBYTE(v100) = 1;
  v101 = *(_QWORD *)v98 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v101 )
  {
    v126 = *(_BYTE *)(v98 + 20);
    v127 = v92 + *(_QWORD *)(v98 + 8);
    if ( (*(_QWORD *)v98 & 4) != 0 )
    {
      v128 = *(_DWORD *)(v101 + 12);
      *((_QWORD *)&v288 + 1) = v127;
      BYTE4(v289) = v126;
      *(_QWORD *)&v288 = v101 | 4;
      LODWORD(v289) = v128;
    }
    else
    {
      v140 = *(_QWORD *)(v101 + 8);
      *(_QWORD *)&v288 = *(_QWORD *)v98 & 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)&v288 + 1) = v127;
      v141 = *(unsigned __int8 *)(v140 + 8);
      BYTE4(v289) = v126;
      if ( (unsigned int)(v141 - 17) <= 1 )
        v140 = **(_QWORD **)(v140 + 16);
      LODWORD(v289) = *(_DWORD *)(v140 + 8) >> 8;
    }
  }
  else
  {
    v102 = *(_QWORD *)(v98 + 8);
    v103 = *(_DWORD *)(v98 + 16);
    *(_QWORD *)&v288 = 0;
    BYTE4(v289) = 0;
    LODWORD(v289) = v103;
    *((_QWORD *)&v288 + 1) = v92 + v102;
  }
  v104 = sub_33F1DB0(
           (__int64 *)a4,
           1,
           (__int64)&v274,
           v226,
           0,
           v100,
           *(_OWORD *)&v11,
           v237,
           v57 | v267.m128i_i64[1] & 0xFFFFFFFF00000000LL,
           v288,
           v289,
           v262,
           v265,
           v99,
           (__int64)&v296);
  v106 = v105;
  v249 = (unsigned __int64)v104;
  v296 = 0u;
  v297 = 0u;
  sub_2EAC300((__int64)&v290, v224, v225, v92);
  v294 = _mm_load_si128(&v290);
  v295 = v291;
  v110 = sub_33CC4A0(a4, v262, v265, v107, v108, v109);
  v111 = sub_33F5040(
           (_QWORD *)a4,
           v249,
           1u,
           (__int64)&v274,
           v249,
           v106,
           (unsigned __int64)v259,
           v253,
           *(_OWORD *)&v294,
           v295,
           v262,
           v265,
           v110,
           0,
           (__int64)&v296);
  v114 = v113;
  v115 = v298.m128i_u32[2];
  v116 = v111;
  v117 = v298.m128i_u32[2] + 1LL;
  if ( v117 > v298.m128i_u32[3] )
  {
    sub_C8D5F0((__int64)&v298, v299, v298.m128i_u32[2] + 1LL, 0x10u, v117, v112);
    v115 = v298.m128i_u32[2];
  }
  v118 = (__m128i **)(v298.m128i_i64[0] + 16 * v115);
  *v118 = v116;
  v118[1] = v114;
  ++v298.m128i_i32[2];
  *((_QWORD *)&v209 + 1) = v298.m128i_u32[2];
  *(_QWORD *)&v209 = v298.m128i_i64[0];
  v119 = sub_33FC220((_QWORD *)a4, 2, (__int64)&v274, 1, 0, v112, v209);
  v121 = v120;
  v296 = 0u;
  v297 = 0u;
  sub_2EAC300((__int64)&v292, v224, v225, 0);
  *((_QWORD *)&v206 + 1) = v121;
  *(_QWORD *)&v206 = v119;
  v122 = sub_33F1DB0(
           (__int64 *)a4,
           (*(_BYTE *)(a3 + 33) >> 2) & 3,
           (__int64)&v274,
           v270,
           v271,
           0,
           v206,
           v215,
           v216,
           v292,
           v293,
           v272,
           v273,
           0,
           (__int64)&v296);
  v123 = (_OWORD *)v298.m128i_i64[0];
  *a1 = v122;
  a1[1] = v124 | v106 & 0xFFFFFFFF00000000LL;
  a1[2] = v119;
  a1[3] = v121;
  if ( v123 != v299 )
    _libc_free((unsigned __int64)v123);
LABEL_79:
  if ( v274 )
    sub_B91220((__int64)&v274, v274);
  return a1;
}
