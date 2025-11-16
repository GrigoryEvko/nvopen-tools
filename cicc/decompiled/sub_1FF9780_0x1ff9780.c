// Function: sub_1FF9780
// Address: 0x1ff9780
//
_QWORD *__fastcall sub_1FF9780(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // r15
  const __m128i *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __m128i v11; // xmm0
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  bool v17; // zf
  __m128i v18; // xmm1
  __int64 v19; // rax
  __int64 v20; // rax
  __m128i v21; // xmm2
  __int64 v22; // rcx
  __int64 v23; // rcx
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int16 v28; // cx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r15
  int v32; // edx
  __int64 v33; // r12
  __int64 v34; // r13
  _QWORD *v35; // rdi
  unsigned __int8 v36; // al
  char v37; // dl
  _QWORD *result; // rax
  int v39; // edx
  _QWORD *v40; // r12
  __int64 v41; // rdx
  __int64 v42; // rcx
  _QWORD *v43; // r8
  _QWORD *v44; // r9
  __int64 v45; // rdi
  __int64 v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rdi
  __m128i v49; // xmm4
  __int64 v50; // rcx
  int v51; // eax
  char v52; // di
  const void **v53; // rax
  unsigned int v54; // r12d
  int v55; // eax
  unsigned int v56; // edx
  unsigned int v57; // esi
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r13
  __int64 *v62; // rax
  _QWORD *v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // r12
  int v66; // edx
  __int64 v67; // rdx
  __int64 v68; // rcx
  _QWORD *v69; // r8
  _QWORD *v70; // r9
  __int64 v71; // rdi
  __int64 v72; // r12
  __int64 v73; // rdx
  __int64 v74; // rdi
  char v75; // dl
  __int64 v76; // rsi
  __int64 *v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // r8
  __int64 v81; // r9
  _QWORD *v82; // rdi
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 *v85; // r10
  __int64 v86; // r15
  __int64 v87; // rdi
  __int64 v88; // rdx
  __int64 v89; // rdi
  __int64 v90; // r12
  __int64 v91; // r9
  __int64 v92; // rax
  char v93; // al
  __int64 v94; // r15
  unsigned int v95; // r13d
  unsigned int v96; // r12d
  unsigned int v97; // eax
  __int64 v98; // r12
  int v99; // edx
  __int64 v100; // rdx
  __int64 v101; // rdi
  __int64 v102; // rdi
  __int64 v103; // r15
  unsigned int v104; // r12d
  unsigned int v105; // r13d
  __int64 v106; // rax
  __int64 v107; // r12
  int v108; // edx
  __int64 v109; // rdx
  __int64 v110; // rdi
  __int64 v111; // rdx
  __int64 v112; // rsi
  __int64 *v113; // rsi
  unsigned int v114; // ecx
  unsigned int v115; // r9d
  __int64 *v116; // r12
  unsigned int v117; // esi
  _QWORD *v118; // rdi
  unsigned int v119; // r9d
  __int64 *v120; // rax
  __int64 v121; // rdx
  __int64 *v122; // rcx
  __int64 v123; // rax
  __int64 v124; // rdx
  __int64 v125; // rcx
  __int64 v126; // r13
  __int64 v127; // rdi
  __int64 v128; // rax
  const void **v129; // rdx
  __int128 v130; // rax
  __int64 *v131; // rax
  _QWORD *v132; // rdi
  __int64 v133; // rdx
  __int64 v134; // rax
  __int64 *v135; // r12
  __int64 v136; // rdx
  __int64 v137; // rax
  __int128 v138; // rax
  __int64 *v139; // rax
  _QWORD *v140; // rdi
  __int64 v141; // rdx
  unsigned __int64 v142; // rax
  __int64 v143; // rdx
  unsigned __int64 v144; // rcx
  int v145; // eax
  __int64 v146; // rdx
  __int64 *v147; // r12
  int v148; // edx
  __int64 v149; // rdx
  __int64 v150; // rdi
  __int64 v151; // rdx
  __int64 v152; // rdx
  _QWORD *v153; // rsi
  unsigned int v154; // ecx
  __int64 v155; // rax
  _QWORD *v156; // rdi
  __int64 v157; // rdx
  __int64 v158; // r12
  int v159; // edx
  __int64 v160; // rdx
  __int64 v161; // rdi
  __int64 v162; // rdx
  __int64 v163; // rdx
  __int64 *v164; // r13
  __int64 v165; // rax
  _QWORD *v166; // rdi
  __int64 v167; // rdx
  int v168; // edx
  __int64 v169; // r12
  __int64 v170; // rdx
  __int64 v171; // rdi
  __int64 v172; // rax
  _QWORD *v173; // rdi
  __int64 v174; // rdx
  __int64 v175; // rax
  __int64 *v176; // rax
  __int64 *v177; // r12
  __int64 v178; // rdx
  __int128 v179; // rax
  __int64 *v180; // rax
  __int64 v181; // rdi
  __int64 v182; // rdx
  __int64 *v183; // r12
  __int64 v184; // rax
  const void **v185; // rdx
  __int128 v186; // rax
  __int64 *v187; // rax
  _QWORD *v188; // rdi
  __int64 v189; // rdx
  unsigned __int64 v190; // rax
  __int64 v191; // rdx
  unsigned __int64 v192; // rcx
  int v193; // eax
  __int64 v194; // rdx
  int v195; // edx
  _QWORD *v196; // r12
  __int64 v197; // rdx
  __int64 v198; // rdi
  __int64 v199; // rdx
  void *v200; // rax
  __int64 *v201; // rsi
  __int64 v202; // r12
  __int64 v203; // rax
  __int64 *v204; // r10
  unsigned __int64 v205; // rdx
  int v206; // eax
  __m128i v207; // rax
  __int64 v208; // r10
  _BYTE *v209; // rax
  __int64 v210; // r10
  __int64 v211; // rcx
  __int64 v212; // rcx
  _QWORD *v213; // rdi
  __int64 v214; // rcx
  __int64 v215; // rax
  __int64 v216; // rax
  __int64 *v217; // rdi
  __int64 v218; // rdx
  __int64 v219; // r15
  __int128 v220; // rax
  __int64 *v221; // rax
  __int64 *v222; // rcx
  _QWORD *v223; // rdi
  __int64 v224; // rdx
  __int64 v225; // rdx
  unsigned __int64 v226; // rsi
  int v227; // edx
  __int64 v228; // rax
  __int64 *v229; // rdi
  __int64 v230; // rdx
  __int64 *v231; // rax
  char v232; // si
  __int64 v233; // r11
  __int64 v234; // rax
  char v235; // si
  __int64 v236; // r11
  __int64 v237; // rax
  char v238; // r8
  __int64 v239; // rcx
  __int64 v240; // rdx
  __int128 v241; // [rsp-40h] [rbp-300h]
  __int128 v242; // [rsp-40h] [rbp-300h]
  __int128 v243; // [rsp-40h] [rbp-300h]
  __int128 v244; // [rsp-30h] [rbp-2F0h]
  __int64 v245; // [rsp-30h] [rbp-2F0h]
  __int128 v246; // [rsp-30h] [rbp-2F0h]
  __int64 v247; // [rsp-20h] [rbp-2E0h]
  __int64 v248; // [rsp-20h] [rbp-2E0h]
  int v249; // [rsp-18h] [rbp-2D8h]
  __int128 v250; // [rsp-10h] [rbp-2D0h]
  unsigned int v251; // [rsp-10h] [rbp-2D0h]
  unsigned __int64 v252; // [rsp+8h] [rbp-2B8h]
  __int64 v253; // [rsp+10h] [rbp-2B0h]
  __int64 v254; // [rsp+10h] [rbp-2B0h]
  __int64 v255; // [rsp+10h] [rbp-2B0h]
  __int64 v256; // [rsp+10h] [rbp-2B0h]
  __int64 v257; // [rsp+10h] [rbp-2B0h]
  unsigned int v258; // [rsp+10h] [rbp-2B0h]
  __int64 *v259; // [rsp+10h] [rbp-2B0h]
  __int64 v260; // [rsp+18h] [rbp-2A8h]
  __int64 v261; // [rsp+20h] [rbp-2A0h]
  __int64 v262; // [rsp+20h] [rbp-2A0h]
  __int64 v263; // [rsp+20h] [rbp-2A0h]
  __int64 v264; // [rsp+28h] [rbp-298h]
  unsigned __int64 v265; // [rsp+28h] [rbp-298h]
  unsigned __int64 v266; // [rsp+28h] [rbp-298h]
  __m128i v267; // [rsp+30h] [rbp-290h] BYREF
  __int64 *v268; // [rsp+40h] [rbp-280h]
  unsigned __int64 v269; // [rsp+48h] [rbp-278h]
  _OWORD *v270; // [rsp+50h] [rbp-270h]
  __int64 v271; // [rsp+58h] [rbp-268h]
  __int64 v272; // [rsp+60h] [rbp-260h]
  __int64 v273; // [rsp+68h] [rbp-258h]
  __int128 v274; // [rsp+70h] [rbp-250h]
  _BYTE *v275; // [rsp+80h] [rbp-240h]
  __int64 v276; // [rsp+88h] [rbp-238h]
  int v277; // [rsp+94h] [rbp-22Ch]
  __int64 v278; // [rsp+98h] [rbp-228h]
  __int64 v279; // [rsp+A0h] [rbp-220h]
  __int64 v280; // [rsp+A8h] [rbp-218h]
  __m128i v281; // [rsp+B0h] [rbp-210h]
  __m128i v282; // [rsp+C0h] [rbp-200h]
  __int64 v283; // [rsp+D0h] [rbp-1F0h]
  __int64 v284; // [rsp+D8h] [rbp-1E8h]
  __int64 v285; // [rsp+E0h] [rbp-1E0h]
  __int64 v286; // [rsp+E8h] [rbp-1D8h]
  __int64 *v287; // [rsp+F0h] [rbp-1D0h]
  __int64 v288; // [rsp+F8h] [rbp-1C8h]
  __int64 *v289; // [rsp+100h] [rbp-1C0h]
  __int64 v290; // [rsp+108h] [rbp-1B8h]
  __int64 v291; // [rsp+110h] [rbp-1B0h]
  __int64 v292; // [rsp+118h] [rbp-1A8h]
  __int64 *v293; // [rsp+120h] [rbp-1A0h]
  __int64 v294; // [rsp+128h] [rbp-198h]
  __int64 v295; // [rsp+130h] [rbp-190h]
  __int64 v296; // [rsp+138h] [rbp-188h]
  __int64 *v297; // [rsp+140h] [rbp-180h]
  __int64 v298; // [rsp+148h] [rbp-178h]
  __int64 *v299; // [rsp+150h] [rbp-170h]
  __int64 v300; // [rsp+158h] [rbp-168h]
  __int64 *v301; // [rsp+160h] [rbp-160h]
  __int64 v302; // [rsp+168h] [rbp-158h]
  __int64 *v303; // [rsp+170h] [rbp-150h]
  __int64 v304; // [rsp+178h] [rbp-148h]
  __int64 v305; // [rsp+180h] [rbp-140h]
  __int64 v306; // [rsp+188h] [rbp-138h]
  __int64 v307; // [rsp+190h] [rbp-130h]
  __int64 v308; // [rsp+198h] [rbp-128h]
  __int64 *v309; // [rsp+1A0h] [rbp-120h]
  __int64 v310; // [rsp+1A8h] [rbp-118h]
  __int64 v311; // [rsp+1B0h] [rbp-110h]
  __int64 v312; // [rsp+1B8h] [rbp-108h]
  __int64 v313; // [rsp+1C0h] [rbp-100h]
  unsigned __int64 v314; // [rsp+1C8h] [rbp-F8h]
  __m128i v315; // [rsp+1D0h] [rbp-F0h]
  __int64 v316; // [rsp+1E0h] [rbp-E0h] BYREF
  int v317; // [rsp+1E8h] [rbp-D8h]
  __int64 v318; // [rsp+1F0h] [rbp-D0h] BYREF
  int v319; // [rsp+1F8h] [rbp-C8h]
  __int64 v320; // [rsp+200h] [rbp-C0h] BYREF
  const void **v321; // [rsp+208h] [rbp-B8h]
  __m128i v322; // [rsp+210h] [rbp-B0h] BYREF
  __int64 v323; // [rsp+220h] [rbp-A0h]
  __int128 v324; // [rsp+230h] [rbp-90h] BYREF
  __int64 v325; // [rsp+240h] [rbp-80h]
  __int128 v326; // [rsp+250h] [rbp-70h] BYREF
  __int64 v327; // [rsp+260h] [rbp-60h]
  __int128 v328; // [rsp+270h] [rbp-50h] BYREF
  const void **v329; // [rsp+280h] [rbp-40h]

  v8 = *(const __m128i **)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = v8->m128i_i64[0];
  v11 = _mm_loadu_si128(v8 + 5);
  v316 = v9;
  v12 = v8[5].m128i_u32[2];
  v280 = v10;
  v13 = v8->m128i_i64[1];
  v281 = v11;
  v279 = v13;
  v276 = v8[5].m128i_i64[0];
  if ( v9 )
    sub_1623A60((__int64)&v316, v9, 2);
  v14 = *(_QWORD *)(a2 + 104);
  v317 = *(_DWORD *)(a2 + 64);
  v15 = sub_1E34390(v14);
  v16 = *(_QWORD *)(a2 + 104);
  v17 = (*(_BYTE *)(a2 + 27) & 4) == 0;
  v278 = v15;
  v18 = _mm_loadu_si128((const __m128i *)(v16 + 40));
  LOWORD(v277) = *(_WORD *)(v16 + 32);
  v19 = *(_QWORD *)(v16 + 56);
  v322 = v18;
  v323 = v19;
  v20 = *(_QWORD *)(a2 + 32);
  if ( !v17 )
  {
    v49 = _mm_loadu_si128((const __m128i *)(v20 + 40));
    v50 = *(_QWORD *)(v20 + 40);
    v51 = *(_DWORD *)(v20 + 48);
    v52 = *(_BYTE *)(a2 + 88);
    v273 = v50;
    LODWORD(v270) = v51;
    v53 = *(const void ***)(a2 + 96);
    LOBYTE(v320) = v52;
    v321 = v53;
    v282 = v49;
    if ( v52 )
      v54 = sub_1FEB8F0(v52);
    else
      v54 = sub_1F58D40((__int64)&v320);
    v275 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1[2] + 32));
    if ( (_BYTE)v320 )
    {
      LOBYTE(v274) = v320;
      v55 = sub_1FEB8F0(v320);
      v56 = (unsigned __int8)v274;
    }
    else
    {
      LOBYTE(v274) = 0;
      v55 = sub_1F58D40((__int64)&v320);
      v56 = 0;
    }
    v57 = (v55 + 7) & 0xFFFFFFF8;
    if ( v57 != v54 )
    {
      v58 = a1[2];
      if ( v57 == 32 )
      {
        LOBYTE(v59) = 5;
        goto LABEL_31;
      }
      if ( v57 > 0x20 )
      {
        if ( v57 == 64 )
        {
          LOBYTE(v59) = 6;
          goto LABEL_31;
        }
        if ( v57 == 128 )
        {
          LOBYTE(v59) = 7;
          goto LABEL_31;
        }
      }
      else
      {
        if ( v57 == 8 )
        {
          LOBYTE(v59) = 3;
          goto LABEL_31;
        }
        LOBYTE(v59) = 4;
        if ( v57 == 16 )
        {
LABEL_31:
          v60 = 0;
LABEL_32:
          LOBYTE(v5) = v59;
          v61 = v60;
          v62 = sub_1D3BC50(
                  (__int64 *)v58,
                  v282.m128i_i64[0],
                  v282.m128i_u64[1],
                  (__int64)&v316,
                  (unsigned int)v320,
                  (__int64)v321,
                  v11,
                  *(double *)v18.m128i_i64,
                  a5);
          v63 = (_QWORD *)a1[2];
          v303 = v62;
          v282.m128i_i64[0] = (__int64)v62;
          v304 = v64;
          v282.m128i_i64[1] = (unsigned int)v64 | v282.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v65 = sub_1D2C750(
                  v63,
                  v280,
                  v279,
                  (__int64)&v316,
                  (__int64)v62,
                  v282.m128i_i64[1],
                  v281.m128i_i64[0],
                  v281.m128i_i64[1],
                  *(_OWORD *)*(_QWORD *)(a2 + 104),
                  *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
                  v5,
                  v61,
                  v278,
                  v277,
                  (__int64)&v322);
          sub_1D44850(a1[2], a2, 0, v65, v66);
          v71 = a1[4];
          if ( v71 )
          {
            *(_QWORD *)&v328 = v65;
            sub_1FF5010(v71, &v328, v67, v68, v69, v70);
          }
          v72 = a1[3];
          *(_QWORD *)&v328 = a2;
          result = *(_QWORD **)(v72 + 8);
          if ( *(_QWORD **)(v72 + 16) != result )
            goto LABEL_35;
          v73 = (__int64)&result[*(unsigned int *)(v72 + 28)];
          if ( result != (_QWORD *)v73 )
          {
            while ( a2 != *result )
            {
              if ( (_QWORD *)v73 == ++result )
                goto LABEL_83;
            }
            goto LABEL_139;
          }
LABEL_83:
          result = (_QWORD *)v73;
          goto LABEL_139;
        }
      }
      v59 = sub_1F58CC0(*(_QWORD **)(v58 + 48), v57);
      v58 = a1[2];
      v5 = v59;
      goto LABEL_32;
    }
    if ( (v54 & (v54 - 1)) == 0 )
    {
      v91 = a1[1];
      v92 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
      if ( (_BYTE)v92 )
      {
        if ( !(_BYTE)v56 )
          goto LABEL_180;
        v93 = *(_BYTE *)(v56 + v91 + 115 * v92 + 58658);
        if ( v93 != 2 )
        {
          if ( v93 == 4 )
          {
            result = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)v91 + 1312LL))(
                                 a1[1],
                                 a2,
                                 0,
                                 a1[2]);
            v196 = result;
            if ( !result || (_QWORD *)a2 == result && !v195 )
              goto LABEL_39;
            sub_1D44850(a1[2], a2, 0, (__int64)result, v195);
            v198 = a1[4];
            if ( v198 )
            {
              *(_QWORD *)&v328 = v196;
              sub_1FF5010(v198, &v328, v197, v68, v69, v70);
            }
            v72 = a1[3];
            *(_QWORD *)&v328 = a2;
            result = *(_QWORD **)(v72 + 8);
            if ( *(_QWORD **)(v72 + 16) == result )
            {
              v73 = (__int64)&result[*(unsigned int *)(v72 + 28)];
              if ( result != (_QWORD *)v73 )
              {
                while ( a2 != *result )
                {
                  if ( (_QWORD *)v73 == ++result )
                    goto LABEL_83;
                }
                goto LABEL_139;
              }
              goto LABEL_83;
            }
          }
          else
          {
            v94 = *(_QWORD *)(a2 + 96);
            v95 = *(unsigned __int8 *)(a2 + 88);
            v96 = sub_1E340A0(*(_QWORD *)(a2 + 104));
            v97 = sub_1E34390(*(_QWORD *)(a2 + 104));
            result = (_QWORD *)sub_1F43CC0(a1[1], *(_QWORD *)(a1[2] + 48), (__int64)v275, v95, v94, v96, v97, 0);
            if ( (_BYTE)result )
              goto LABEL_39;
            v98 = sub_20BB820(a1[1], a2, a1[2]);
            sub_1D44850(a1[2], a2, 0, v98, v99);
            v101 = a1[4];
            if ( v101 )
            {
              *(_QWORD *)&v328 = v98;
              sub_1FF5010(v101, &v328, v100, v68, v69, v70);
            }
            v72 = a1[3];
            *(_QWORD *)&v328 = a2;
            result = *(_QWORD **)(v72 + 8);
            if ( *(_QWORD **)(v72 + 16) == result )
            {
              v73 = (__int64)&result[*(unsigned int *)(v72 + 28)];
              if ( result != (_QWORD *)v73 )
              {
                while ( a2 != *result )
                {
                  if ( (_QWORD *)v73 == ++result )
                    goto LABEL_83;
                }
                goto LABEL_139;
              }
              goto LABEL_83;
            }
          }
          goto LABEL_35;
        }
      }
      else if ( !(_BYTE)v56 )
      {
        goto LABEL_180;
      }
      if ( *(_QWORD *)(v91 + 8LL * (int)v56 + 120) )
      {
        v172 = sub_1D309E0(
                 (__int64 *)a1[2],
                 145,
                 (__int64)&v316,
                 (unsigned int)v320,
                 v321,
                 0,
                 *(double *)v11.m128i_i64,
                 *(double *)v18.m128i_i64,
                 *(double *)a5.m128i_i64,
                 *(_OWORD *)&v282);
        v173 = (_QWORD *)a1[2];
        v285 = v172;
        v282.m128i_i64[0] = v172;
        v286 = v174;
        v282.m128i_i64[1] = (unsigned int)v174 | v282.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v169 = sub_1D2BF40(
                 v173,
                 v280,
                 v279,
                 (__int64)&v316,
                 v172,
                 v282.m128i_i64[1],
                 v281.m128i_i64[0],
                 v281.m128i_i64[1],
                 *(_OWORD *)*(_QWORD *)(a2 + 104),
                 *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
                 v278,
                 (unsigned __int16)v277,
                 (__int64)&v322);
LABEL_181:
        sub_1D44850(a1[2], a2, 0, v169, v168);
        v171 = a1[4];
        if ( v171 )
        {
          *(_QWORD *)&v328 = v169;
          sub_1FF5010(v171, &v328, v170, v68, v69, v70);
        }
        v72 = a1[3];
        *(_QWORD *)&v328 = a2;
        result = *(_QWORD **)(v72 + 8);
        if ( *(_QWORD **)(v72 + 16) == result )
        {
          v73 = (__int64)&result[*(unsigned int *)(v72 + 28)];
          if ( result != (_QWORD *)v73 )
          {
            while ( a2 != *result )
            {
              if ( (_QWORD *)v73 == ++result )
                goto LABEL_83;
            }
            goto LABEL_139;
          }
          goto LABEL_83;
        }
LABEL_35:
        result = sub_16CC9F0(v72, a2);
        if ( a2 == *result )
        {
          v151 = *(_QWORD *)(v72 + 16);
          if ( v151 == *(_QWORD *)(v72 + 8) )
            v68 = *(unsigned int *)(v72 + 28);
          else
            v68 = *(unsigned int *)(v72 + 24);
          v73 = v151 + 8 * v68;
        }
        else
        {
          result = *(_QWORD **)(v72 + 16);
          if ( result != *(_QWORD **)(v72 + 8) )
            goto LABEL_37;
          result += *(unsigned int *)(v72 + 28);
          v73 = (__int64)result;
        }
LABEL_139:
        if ( (_QWORD *)v73 == result )
          goto LABEL_37;
        goto LABEL_140;
      }
LABEL_180:
      v164 = (__int64 *)a1[2];
      sub_1F40D10((__int64)&v328, a1[1], v164[6], v320, (__int64)v321);
      v165 = sub_1D309E0(
               v164,
               145,
               (__int64)&v316,
               BYTE8(v328),
               v329,
               0,
               *(double *)v11.m128i_i64,
               *(double *)v18.m128i_i64,
               *(double *)a5.m128i_i64,
               *(_OWORD *)&v282);
      v166 = (_QWORD *)a1[2];
      v283 = v165;
      v282.m128i_i64[0] = v165;
      v284 = v167;
      v282.m128i_i64[1] = (unsigned int)v167 | v282.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v169 = sub_1D2C750(
               v166,
               v280,
               v279,
               (__int64)&v316,
               v165,
               v282.m128i_i64[1],
               v281.m128i_i64[0],
               v281.m128i_i64[1],
               *(_OWORD *)*(_QWORD *)(a2 + 104),
               *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
               v320,
               (__int64)v321,
               v278,
               v277,
               (__int64)&v322);
      goto LABEL_181;
    }
    _BitScanReverse(&v114, v54);
    v115 = v54;
    v116 = (__int64 *)a1[2];
    v117 = 0x80000000 >> (v114 ^ 0x1F);
    v118 = (_QWORD *)v116[6];
    v267.m128i_i32[0] = v117;
    v119 = v115 - v117;
    if ( v117 == 32 )
    {
      LOBYTE(v120) = 5;
    }
    else if ( v117 > 0x20 )
    {
      if ( v267.m128i_i32[0] == 64 )
      {
        LOBYTE(v120) = 6;
      }
      else
      {
        if ( v267.m128i_i32[0] != 128 )
        {
LABEL_135:
          LODWORD(v274) = v119;
          v120 = (__int64 *)sub_1F58CC0(v118, v267.m128i_u32[0]);
          v116 = (__int64 *)a1[2];
          v119 = v274;
          v268 = v120;
          v118 = (_QWORD *)v116[6];
LABEL_107:
          v122 = v268;
          v264 = v121;
          LOBYTE(v122) = (_BYTE)v120;
          v261 = (__int64)v122;
          if ( v119 == 32 )
          {
            LOBYTE(v123) = 5;
          }
          else if ( v119 > 0x20 )
          {
            if ( v119 == 64 )
            {
              LOBYTE(v123) = 6;
            }
            else
            {
              if ( v119 != 128 )
              {
LABEL_133:
                LODWORD(v274) = v119;
                v123 = sub_1F58CC0(v118, v119);
                v116 = (__int64 *)a1[2];
                v119 = v274;
                v253 = v123;
LABEL_112:
                v125 = v253;
                v260 = v124;
                v272 = 16 * v12;
                LOBYTE(v125) = v123;
                v254 = v125;
                v268 = 0;
                v269 = 0;
                v126 = 16LL * (unsigned int)v270;
                v274 = 0u;
                v277 = (unsigned __int16)v277;
                v252 = (((unsigned __int32)v267.m128i_i32[0] >> 3) | (unsigned __int64)(unsigned int)v278)
                     & -(__int64)(((unsigned __int32)v267.m128i_i32[0] >> 3) | (unsigned __int64)(unsigned int)v278);
                if ( *v275 )
                {
                  v127 = a1[1];
                  v267.m128i_i64[0] = (unsigned __int32)v267.m128i_i32[0] >> 3;
                  LODWORD(v270) = v119;
                  v128 = sub_1F40B60(
                           v127,
                           *(unsigned __int8 *)(v126 + *(_QWORD *)(v273 + 40)),
                           *(_QWORD *)(v126 + *(_QWORD *)(v273 + 40) + 8),
                           (__int64)v275,
                           1);
                  *(_QWORD *)&v130 = sub_1D38BB0(
                                       (__int64)v116,
                                       (unsigned int)v270,
                                       (__int64)&v316,
                                       v128,
                                       v129,
                                       0,
                                       v11,
                                       *(double *)v18.m128i_i64,
                                       a5,
                                       0);
                  v131 = sub_1D332F0(
                           v116,
                           124,
                           (__int64)&v316,
                           *(unsigned __int8 *)(*(_QWORD *)(v273 + 40) + v126),
                           *(const void ***)(*(_QWORD *)(v273 + 40) + v126 + 8),
                           0,
                           *(double *)v11.m128i_i64,
                           *(double *)v18.m128i_i64,
                           a5,
                           v282.m128i_i64[0],
                           v282.m128i_u64[1],
                           v130);
                  v132 = (_QWORD *)a1[2];
                  v293 = v131;
                  *(_QWORD *)&v274 = v131;
                  v294 = v133;
                  v275 = &v322;
                  *((_QWORD *)&v274 + 1) = (unsigned int)v133 | *((_QWORD *)&v274 + 1) & 0xFFFFFFFF00000000LL;
                  v134 = sub_1D2C750(
                           v132,
                           v280,
                           v279,
                           (__int64)&v316,
                           (__int64)v131,
                           *((__int64 *)&v274 + 1),
                           v281.m128i_i64[0],
                           v281.m128i_i64[1],
                           *(_OWORD *)*(_QWORD *)(a2 + 104),
                           *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
                           v261,
                           v264,
                           v278,
                           v277,
                           (__int64)&v322);
                  v135 = (__int64 *)a1[2];
                  v291 = v134;
                  *(_QWORD *)&v274 = v134;
                  v292 = v136;
                  *((_QWORD *)&v274 + 1) = (unsigned int)v136 | *((_QWORD *)&v274 + 1) & 0xFFFFFFFF00000000LL;
                  v137 = *(_QWORD *)(v276 + 40);
                  v278 = v267.m128i_i64[0];
                  *(_QWORD *)&v138 = sub_1D38BB0(
                                       (__int64)v135,
                                       v267.m128i_i64[0],
                                       (__int64)&v316,
                                       *(unsigned __int8 *)(v272 + v137),
                                       *(const void ***)(v272 + v137 + 8),
                                       0,
                                       v11,
                                       *(double *)v18.m128i_i64,
                                       a5,
                                       0);
                  v139 = sub_1D332F0(
                           v135,
                           52,
                           (__int64)&v316,
                           *(unsigned __int8 *)(*(_QWORD *)(v276 + 40) + v272),
                           *(const void ***)(*(_QWORD *)(v276 + 40) + v272 + 8),
                           0,
                           *(double *)v11.m128i_i64,
                           *(double *)v18.m128i_i64,
                           a5,
                           v281.m128i_i64[0],
                           v281.m128i_u64[1],
                           v138);
                  v140 = (_QWORD *)a1[2];
                  v289 = v139;
                  v281.m128i_i64[0] = (__int64)v139;
                  v290 = v141;
                  v142 = (unsigned int)v141 | v281.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                  v143 = *(_QWORD *)(a2 + 104);
                  v281.m128i_i64[1] = v142;
                  v144 = *(_QWORD *)v143 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v144 )
                  {
                    v235 = *(_BYTE *)(v143 + 16);
                    v236 = *(_QWORD *)(v143 + 8) + v278;
                    if ( (*(_QWORD *)v143 & 4) != 0 )
                    {
                      *((_QWORD *)&v326 + 1) = *(_QWORD *)(v143 + 8) + v278;
                      LOBYTE(v327) = v235;
                      *(_QWORD *)&v326 = v144 | 4;
                      HIDWORD(v327) = *(_DWORD *)(v144 + 12);
                    }
                    else
                    {
                      *(_QWORD *)&v326 = *(_QWORD *)v143 & 0xFFFFFFFFFFFFFFF8LL;
                      *((_QWORD *)&v326 + 1) = v236;
                      LOBYTE(v327) = v235;
                      v237 = *(_QWORD *)v144;
                      if ( *(_BYTE *)(*(_QWORD *)v144 + 8LL) == 16 )
                        v237 = **(_QWORD **)(v237 + 16);
                      HIDWORD(v327) = *(_DWORD *)(v237 + 8) >> 8;
                    }
                  }
                  else
                  {
                    v145 = *(_DWORD *)(v143 + 20);
                    LODWORD(v327) = 0;
                    v326 = 0u;
                    HIDWORD(v327) = v145;
                  }
                  v287 = (__int64 *)sub_1D2C750(
                                      v140,
                                      v280,
                                      v279,
                                      (__int64)&v316,
                                      v282.m128i_i64[0],
                                      v282.m128i_i64[1],
                                      v281.m128i_i64[0],
                                      v281.m128i_i64[1],
                                      v326,
                                      v327,
                                      v254,
                                      v260,
                                      v252,
                                      v277,
                                      (__int64)v275);
                  v268 = v287;
                  v288 = v146;
                  v269 = (unsigned int)v146 | v269 & 0xFFFFFFFF00000000LL;
                }
                else
                {
                  v175 = *(_QWORD *)(a2 + 104);
                  v249 = v278;
                  v245 = *(_QWORD *)(v175 + 16);
                  v242 = *(_OWORD *)v175;
                  v270 = &v322;
                  v278 = (unsigned __int32)v267.m128i_i32[0] >> 3;
                  v176 = (__int64 *)sub_1D2C750(
                                      v116,
                                      v280,
                                      v279,
                                      (__int64)&v316,
                                      v282.m128i_i64[0],
                                      v282.m128i_i64[1],
                                      v281.m128i_i64[0],
                                      v281.m128i_i64[1],
                                      v242,
                                      v245,
                                      v261,
                                      v264,
                                      v249,
                                      v277,
                                      (__int64)&v322);
                  v177 = (__int64 *)a1[2];
                  v301 = v176;
                  v268 = v176;
                  v302 = v178;
                  v269 = (unsigned int)v178 | v269 & 0xFFFFFFFF00000000LL;
                  *(_QWORD *)&v179 = sub_1D38BB0(
                                       (__int64)v177,
                                       v278,
                                       (__int64)&v316,
                                       *(unsigned __int8 *)(*(_QWORD *)(v276 + 40) + v272),
                                       *(const void ***)(*(_QWORD *)(v276 + 40) + v272 + 8),
                                       0,
                                       v11,
                                       *(double *)v18.m128i_i64,
                                       a5,
                                       0);
                  v180 = sub_1D332F0(
                           v177,
                           52,
                           (__int64)&v316,
                           *(unsigned __int8 *)(*(_QWORD *)(v276 + 40) + v272),
                           *(const void ***)(*(_QWORD *)(v276 + 40) + v272 + 8),
                           0,
                           *(double *)v11.m128i_i64,
                           *(double *)v18.m128i_i64,
                           a5,
                           v281.m128i_i64[0],
                           v281.m128i_u64[1],
                           v179);
                  v181 = a1[1];
                  v299 = v180;
                  v281.m128i_i64[0] = (__int64)v180;
                  v300 = v182;
                  v183 = (__int64 *)a1[2];
                  v281.m128i_i64[1] = (unsigned int)v182 | v281.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                  v184 = sub_1F40B60(
                           v181,
                           *(unsigned __int8 *)(v126 + *(_QWORD *)(v273 + 40)),
                           *(_QWORD *)(v126 + *(_QWORD *)(v273 + 40) + 8),
                           (__int64)v275,
                           1);
                  *(_QWORD *)&v186 = sub_1D38BB0(
                                       (__int64)v183,
                                       v267.m128i_u32[0],
                                       (__int64)&v316,
                                       v184,
                                       v185,
                                       0,
                                       v11,
                                       *(double *)v18.m128i_i64,
                                       a5,
                                       0);
                  v187 = sub_1D332F0(
                           v183,
                           124,
                           (__int64)&v316,
                           *(unsigned __int8 *)(*(_QWORD *)(v273 + 40) + v126),
                           *(const void ***)(*(_QWORD *)(v273 + 40) + v126 + 8),
                           0,
                           *(double *)v11.m128i_i64,
                           *(double *)v18.m128i_i64,
                           a5,
                           v282.m128i_i64[0],
                           v282.m128i_u64[1],
                           v186);
                  v188 = (_QWORD *)a1[2];
                  v297 = v187;
                  *(_QWORD *)&v274 = v187;
                  v298 = v189;
                  v190 = (unsigned int)v189 | *((_QWORD *)&v274 + 1) & 0xFFFFFFFF00000000LL;
                  v191 = *(_QWORD *)(a2 + 104);
                  *((_QWORD *)&v274 + 1) = v190;
                  v192 = *(_QWORD *)v191 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v192 )
                  {
                    v232 = *(_BYTE *)(v191 + 16);
                    v233 = *(_QWORD *)(v191 + 8) + v278;
                    if ( (*(_QWORD *)v191 & 4) != 0 )
                    {
                      *((_QWORD *)&v324 + 1) = *(_QWORD *)(v191 + 8) + v278;
                      LOBYTE(v325) = v232;
                      *(_QWORD *)&v324 = v192 | 4;
                      HIDWORD(v325) = *(_DWORD *)(v192 + 12);
                    }
                    else
                    {
                      *(_QWORD *)&v324 = *(_QWORD *)v191 & 0xFFFFFFFFFFFFFFF8LL;
                      *((_QWORD *)&v324 + 1) = v233;
                      LOBYTE(v325) = v232;
                      v234 = *(_QWORD *)v192;
                      if ( *(_BYTE *)(*(_QWORD *)v192 + 8LL) == 16 )
                        v234 = **(_QWORD **)(v234 + 16);
                      HIDWORD(v325) = *(_DWORD *)(v234 + 8) >> 8;
                    }
                  }
                  else
                  {
                    v193 = *(_DWORD *)(v191 + 20);
                    LODWORD(v325) = 0;
                    v324 = 0u;
                    HIDWORD(v325) = v193;
                  }
                  v295 = sub_1D2C750(
                           v188,
                           v280,
                           v279,
                           (__int64)&v316,
                           v274,
                           *((__int64 *)&v274 + 1),
                           v281.m128i_i64[0],
                           v281.m128i_i64[1],
                           v324,
                           v325,
                           v254,
                           v260,
                           v252,
                           v277,
                           (__int64)v270);
                  *(_QWORD *)&v274 = v295;
                  v296 = v194;
                  *((_QWORD *)&v274 + 1) = (unsigned int)v194 | *((_QWORD *)&v274 + 1) & 0xFFFFFFFF00000000LL;
                }
                v147 = sub_1D332F0(
                         (__int64 *)a1[2],
                         2,
                         (__int64)&v316,
                         1,
                         0,
                         0,
                         *(double *)v11.m128i_i64,
                         *(double *)v18.m128i_i64,
                         a5,
                         (__int64)v268,
                         v269,
                         v274);
                sub_1D44850(a1[2], a2, 0, (__int64)v147, v148);
                v150 = a1[4];
                if ( v150 )
                {
                  *(_QWORD *)&v328 = v147;
                  sub_1FF5010(v150, &v328, v149, v68, v69, v70);
                }
                v72 = a1[3];
                *(_QWORD *)&v328 = a2;
                result = *(_QWORD **)(v72 + 8);
                if ( *(_QWORD **)(v72 + 16) == result )
                {
                  v73 = (__int64)&result[*(unsigned int *)(v72 + 28)];
                  if ( result == (_QWORD *)v73 )
                  {
LABEL_206:
                    result = (_QWORD *)v73;
                  }
                  else
                  {
                    while ( a2 != *result )
                    {
                      if ( (_QWORD *)v73 == ++result )
                        goto LABEL_206;
                    }
                  }
                }
                else
                {
                  result = sub_16CC9F0(v72, a2);
                  if ( a2 == *result )
                  {
                    v199 = *(_QWORD *)(v72 + 16);
                    if ( v199 == *(_QWORD *)(v72 + 8) )
                      v68 = *(unsigned int *)(v72 + 28);
                    else
                      v68 = *(unsigned int *)(v72 + 24);
                    v73 = v199 + 8 * v68;
                  }
                  else
                  {
                    result = *(_QWORD **)(v72 + 16);
                    if ( result != *(_QWORD **)(v72 + 8) )
                      goto LABEL_37;
                    result += *(unsigned int *)(v72 + 28);
                    v73 = (__int64)result;
                  }
                }
                if ( result == (_QWORD *)v73 )
                {
LABEL_37:
                  v74 = a1[4];
                  if ( v74 )
                    result = (_QWORD *)sub_1FF5010(v74, &v328, v73, v68, v69, v70);
                  goto LABEL_39;
                }
LABEL_140:
                *result = -2;
                ++*(_DWORD *)(v72 + 32);
                goto LABEL_37;
              }
              LOBYTE(v123) = 7;
            }
          }
          else if ( v119 == 8 )
          {
            LOBYTE(v123) = 3;
          }
          else
          {
            LOBYTE(v123) = 4;
            if ( v119 != 16 )
            {
              LOBYTE(v123) = 2;
              if ( v119 != 1 )
                goto LABEL_133;
            }
          }
          v124 = 0;
          goto LABEL_112;
        }
        LOBYTE(v120) = 7;
      }
    }
    else if ( v117 == 8 )
    {
      LOBYTE(v120) = 3;
    }
    else
    {
      LOBYTE(v120) = 4;
      if ( v117 != 16 )
      {
        LOBYTE(v120) = 2;
        if ( v117 != 1 )
          goto LABEL_135;
      }
    }
    v121 = 0;
    goto LABEL_107;
  }
  v21 = _mm_loadu_si128((const __m128i *)(v20 + 80));
  *(_QWORD *)&v274 = *(_QWORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  v282 = v21;
  v275 = (_BYTE *)v22;
  v23 = *(_QWORD *)(v20 + 80);
  v24 = *(_DWORD *)(v20 + 88);
  v270 = (_OWORD *)v23;
  LODWORD(v268) = v24;
  v25 = sub_1E34390(v16);
  v26 = *(_QWORD *)(a2 + 72);
  v276 = v25;
  v27 = *(_QWORD *)(a2 + 104);
  v318 = v26;
  v28 = *(_WORD *)(v27 + 32);
  v326 = (__int128)_mm_loadu_si128((const __m128i *)(v27 + 40));
  v29 = *(_QWORD *)(v27 + 56);
  LOWORD(v273) = v28;
  v327 = v29;
  if ( v26 )
    sub_1623A60((__int64)&v318, v26, 2);
  v319 = *(_DWORD *)(a2 + 64);
  v30 = *(_QWORD *)(a2 + 32);
  v31 = *(_QWORD *)(v30 + 40);
  v32 = *(unsigned __int16 *)(v31 + 24);
  if ( v32 == 11 || v32 == 33 )
  {
    v75 = **(_BYTE **)(v31 + 40);
    if ( v75 == 9 )
    {
      if ( !*(_QWORD *)(a1[1] + 160) )
        goto LABEL_8;
      v76 = *(_QWORD *)(v31 + 72);
      v270 = (_OWORD *)a1[2];
      *(_QWORD *)&v328 = v76;
      if ( v76 )
        sub_1623A60((__int64)&v328, v76, 2);
      DWORD2(v328) = *(_DWORD *)(v31 + 64);
      v77 = (__int64 *)(*(_QWORD *)(v31 + 88) + 32LL);
      if ( (void *)*v77 == sub_16982C0() )
        sub_169D930((__int64)&v320, (__int64)v77);
      else
        sub_169D7E0((__int64)&v320, v77);
      sub_16A5D10((__int64)&v324, (__int64)&v320, 0x20u);
      v78 = sub_1D38970((__int64)v270, (__int64)&v324, (__int64)&v328, 5u, 0, 0, v11, *(double *)v18.m128i_i64, v21, 0);
      goto LABEL_52;
    }
    if ( v75 != 10 )
      goto LABEL_8;
    v111 = a1[1];
    if ( *(_QWORD *)(v111 + 168) )
    {
      v112 = *(_QWORD *)(v31 + 72);
      v270 = (_OWORD *)a1[2];
      *(_QWORD *)&v328 = v112;
      if ( v112 )
        sub_1623A60((__int64)&v328, v112, 2);
      DWORD2(v328) = *(_DWORD *)(v31 + 64);
      v113 = (__int64 *)(*(_QWORD *)(v31 + 88) + 32LL);
      if ( (void *)*v113 == sub_16982C0() )
        sub_169D930((__int64)&v320, (__int64)v113);
      else
        sub_169D7E0((__int64)&v320, v113);
      sub_16A5D10((__int64)&v324, (__int64)&v320, 0x40u);
      v78 = sub_1D38970((__int64)v270, (__int64)&v324, (__int64)&v328, 6u, 0, 0, v11, *(double *)v18.m128i_i64, v21, 0);
LABEL_52:
      v80 = v78;
      v81 = v79;
      if ( DWORD2(v324) > 0x40 && (_QWORD)v324 )
      {
        v270 = (_OWORD *)v78;
        v271 = v79;
        j_j___libc_free_0_0(v324);
        v80 = (__int64)v270;
        v81 = v271;
      }
      if ( (unsigned int)v321 > 0x40 && v320 )
      {
        v270 = (_OWORD *)v80;
        v271 = v81;
        j_j___libc_free_0_0(v320);
        v80 = (__int64)v270;
        v81 = v271;
      }
      if ( (_QWORD)v328 )
      {
        v270 = (_OWORD *)v80;
        v271 = v81;
        sub_161E7C0((__int64)&v328, v328);
        v80 = (__int64)v270;
        v81 = v271;
      }
      v82 = (_QWORD *)a1[2];
      v83 = *(_QWORD *)(a2 + 104);
      v247 = *(_QWORD *)(v83 + 16);
      v244 = *(_OWORD *)v83;
      v241 = (__int128)v282;
      v282.m128i_i64[0] = (__int64)&v318;
      v84 = sub_1D2BF40(
              v82,
              v274,
              (__int64)v275,
              (__int64)&v318,
              v80,
              v81,
              v241,
              *((__int64 *)&v241 + 1),
              v244,
              v247,
              v276,
              (unsigned __int16)v273,
              (__int64)&v326);
      v85 = &v318;
      v86 = v84;
      goto LABEL_61;
    }
    if ( !*(_QWORD *)(v111 + 160) || (*(_BYTE *)(a2 + 26) & 8) != 0 )
      goto LABEL_8;
    v200 = sub_16982C0();
    v201 = (__int64 *)(*(_QWORD *)(v31 + 88) + 32LL);
    if ( (void *)*v201 == v200 )
      sub_169D930((__int64)&v320, (__int64)v201);
    else
      sub_169D7E0((__int64)&v320, v201);
    v202 = a1[2];
    sub_16A5A50((__int64)&v328, &v320, 0x20u);
    v267.m128i_i64[0] = (__int64)&v318;
    v203 = sub_1D38970(v202, (__int64)&v328, (__int64)&v318, 5u, 0, 0, v11, *(double *)v18.m128i_i64, v21, 0);
    v204 = &v318;
    v262 = v203;
    v265 = v205;
    if ( DWORD2(v328) > 0x40 && (_QWORD)v328 )
    {
      j_j___libc_free_0_0(v328);
      v204 = (__int64 *)v267.m128i_i64[0];
    }
    v267.m128i_i64[0] = a1[2];
    v206 = (int)v321;
    DWORD2(v324) = (_DWORD)v321;
    if ( (unsigned int)v321 > 0x40 )
    {
      v259 = v204;
      sub_16A4FD0((__int64)&v324, (const void **)&v320);
      v206 = DWORD2(v324);
      v204 = v259;
      if ( DWORD2(v324) > 0x40 )
      {
        sub_16A8110((__int64)&v324, 0x20u);
        v204 = v259;
LABEL_234:
        v255 = (__int64)v204;
        sub_16A5A50((__int64)&v328, (__int64 *)&v324, 0x20u);
        v207.m128i_i64[0] = sub_1D38970(
                              v267.m128i_i64[0],
                              (__int64)&v328,
                              v255,
                              5u,
                              0,
                              0,
                              v11,
                              *(double *)v18.m128i_i64,
                              v21,
                              0);
        v208 = v255;
        v267 = v207;
        if ( DWORD2(v328) > 0x40 && (_QWORD)v328 )
        {
          j_j___libc_free_0_0(v328);
          v208 = v255;
        }
        if ( DWORD2(v324) > 0x40 && (_QWORD)v324 )
        {
          v256 = v208;
          j_j___libc_free_0_0(v324);
          v208 = v256;
        }
        v257 = v208;
        v209 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1[2] + 32));
        v210 = v257;
        if ( *v209 )
        {
          v211 = v262;
          v315 = _mm_load_si128(&v267);
          v262 = v267.m128i_i64[0];
          v313 = v211;
          v314 = v265;
          v267.m128i_i64[0] = v211;
          v212 = (unsigned int)v265;
          v265 = v315.m128i_u32[2] | v265 & 0xFFFFFFFF00000000LL;
          v267.m128i_i64[1] = v212 | v267.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        }
        v213 = (_QWORD *)a1[2];
        v214 = v257;
        v251 = (unsigned __int16)v273;
        v215 = *(_QWORD *)(a2 + 104);
        v258 = (unsigned __int16)v273;
        v248 = *(_QWORD *)(v215 + 16);
        v246 = *(_OWORD *)v215;
        v273 = v210;
        v216 = sub_1D2BF40(
                 v213,
                 v274,
                 (__int64)v275,
                 v214,
                 v262,
                 v265,
                 v282.m128i_i64[0],
                 v282.m128i_i64[1],
                 v246,
                 v248,
                 v276,
                 v251,
                 (__int64)&v326);
        v217 = (__int64 *)a1[2];
        v311 = v216;
        v263 = v216;
        v312 = v218;
        v219 = (unsigned int)v268;
        v268 = v217;
        v266 = (unsigned int)v218 | v265 & 0xFFFFFFFF00000000LL;
        v219 *= 16;
        *(_QWORD *)&v220 = sub_1D38BB0(
                             (__int64)v217,
                             4,
                             v273,
                             *(unsigned __int8 *)(v219 + *((_QWORD *)v270 + 5)),
                             *(const void ***)(v219 + *((_QWORD *)v270 + 5) + 8),
                             0,
                             v11,
                             *(double *)v18.m128i_i64,
                             v21,
                             0);
        v221 = sub_1D332F0(
                 v268,
                 52,
                 v273,
                 *(unsigned __int8 *)(*((_QWORD *)v270 + 5) + v219),
                 *(const void ***)(*((_QWORD *)v270 + 5) + v219 + 8),
                 0,
                 *(double *)v11.m128i_i64,
                 *(double *)v18.m128i_i64,
                 v21,
                 v282.m128i_i64[0],
                 v282.m128i_u64[1],
                 v220);
        v222 = *(__int64 **)(a2 + 104);
        v309 = v221;
        v223 = (_QWORD *)a1[2];
        v282.m128i_i64[0] = (__int64)v221;
        v310 = v224;
        v282.m128i_i64[1] = (unsigned int)v224 | v282.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v225 = *v222;
        v226 = *v222 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v226 )
        {
          v238 = *((_BYTE *)v222 + 16);
          v239 = v222[1] + 4;
          if ( (v225 & 4) != 0 )
          {
            *((_QWORD *)&v328 + 1) = v239;
            LOBYTE(v329) = v238;
            *(_QWORD *)&v328 = v226 | 4;
            HIDWORD(v329) = *(_DWORD *)(v226 + 12);
          }
          else
          {
            *(_QWORD *)&v328 = v226;
            *((_QWORD *)&v328 + 1) = v239;
            LOBYTE(v329) = v238;
            v240 = *(_QWORD *)v226;
            if ( *(_BYTE *)(*(_QWORD *)v226 + 8LL) == 16 )
              v240 = **(_QWORD **)(v240 + 16);
            HIDWORD(v329) = *(_DWORD *)(v240 + 8) >> 8;
          }
        }
        else
        {
          v227 = *((_DWORD *)v222 + 5);
          LODWORD(v329) = 0;
          v328 = 0u;
          HIDWORD(v329) = v227;
        }
        v243 = (__int128)v282;
        v282.m128i_i64[0] = v273;
        v228 = sub_1D2BF40(
                 v223,
                 v274,
                 (__int64)v275,
                 v273,
                 v267.m128i_i64[0],
                 v267.m128i_i64[1],
                 v243,
                 *((__int64 *)&v243 + 1),
                 v328,
                 (__int64)v329,
                 -(v276 | 4) & (v276 | 4),
                 v258,
                 (__int64)&v326);
        v229 = (__int64 *)a1[2];
        v267.m128i_i64[0] = v228;
        v307 = v228;
        v308 = v230;
        v267.m128i_i64[1] = (unsigned int)v230 | v267.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v231 = sub_1D332F0(
                 v229,
                 2,
                 v282.m128i_i64[0],
                 1,
                 0,
                 0,
                 *(double *)v11.m128i_i64,
                 *(double *)v18.m128i_i64,
                 v21,
                 v263,
                 v266,
                 __PAIR128__(v267.m128i_u64[1], v228));
        v85 = (__int64 *)v282.m128i_i64[0];
        v86 = (__int64)v231;
        if ( (unsigned int)v321 > 0x40 && v320 )
        {
          j_j___libc_free_0_0(v320);
          v85 = (__int64 *)v282.m128i_i64[0];
        }
LABEL_61:
        if ( v318 )
          sub_161E7C0((__int64)v85, v318);
        if ( !v86 )
          goto LABEL_10;
        v87 = a1[2];
        *(_QWORD *)&v324 = v86;
        sub_1D444E0(v87, a2, v86);
        v89 = a1[4];
        if ( v89 )
          sub_1FF5010(v89, &v324, v88, v42, v43, v44);
        v90 = a1[3];
        *(_QWORD *)&v326 = a2;
        result = *(_QWORD **)(v90 + 8);
        if ( *(_QWORD **)(v90 + 16) == result )
        {
          v47 = (__int64)&result[*(unsigned int *)(v90 + 28)];
          if ( result == (_QWORD *)v47 )
          {
LABEL_202:
            result = (_QWORD *)v47;
          }
          else
          {
            while ( a2 != *result )
            {
              if ( (_QWORD *)v47 == ++result )
                goto LABEL_202;
            }
          }
        }
        else
        {
          result = sub_16CC9F0(v90, a2);
          if ( a2 == *result )
          {
            v163 = *(_QWORD *)(v90 + 16);
            if ( v163 == *(_QWORD *)(v90 + 8) )
              v42 = *(unsigned int *)(v90 + 28);
            else
              v42 = *(unsigned int *)(v90 + 24);
            v47 = v163 + 8 * v42;
          }
          else
          {
            result = *(_QWORD **)(v90 + 16);
            if ( result != *(_QWORD **)(v90 + 8) )
              goto LABEL_69;
            result += *(unsigned int *)(v90 + 28);
            v47 = (__int64)result;
          }
        }
        if ( (_QWORD *)v47 != result )
        {
          *result = -2;
          ++*(_DWORD *)(v90 + 32);
        }
LABEL_69:
        v48 = a1[4];
        if ( v48 )
        {
LABEL_21:
          result = (_QWORD *)sub_1FF5010(v48, &v326, v47, v42, v43, v44);
          goto LABEL_39;
        }
        goto LABEL_39;
      }
    }
    else
    {
      *(_QWORD *)&v324 = v320;
    }
    if ( v206 == 32 )
      *(_QWORD *)&v324 = 0;
    else
      *(_QWORD *)&v324 = DWORD1(v324);
    goto LABEL_234;
  }
LABEL_8:
  if ( v318 )
  {
    sub_161E7C0((__int64)&v318, v318);
LABEL_10:
    v30 = *(_QWORD *)(a2 + 32);
    v31 = *(_QWORD *)(v30 + 40);
  }
  v33 = *(_QWORD *)(v30 + 40);
  v34 = *(_QWORD *)(v30 + 48);
  v35 = (_QWORD *)a1[1];
  v36 = *(_BYTE *)(*(_QWORD *)(v31 + 40) + 16LL * *(unsigned int *)(v30 + 48));
  v37 = *((_BYTE *)v35 + 259 * v36 + 2608);
  if ( v37 == 1 )
  {
    v152 = v35[9258];
    if ( !v152 )
      goto LABEL_157;
    v153 = v35 + 9257;
    do
    {
      v154 = *(_DWORD *)(v152 + 32);
      if ( v154 <= 0xB9 || v154 == 186 && v36 > *(_BYTE *)(v152 + 36) )
      {
        v152 = *(_QWORD *)(v152 + 24);
      }
      else
      {
        v153 = (_QWORD *)v152;
        v152 = *(_QWORD *)(v152 + 16);
      }
    }
    while ( v152 );
    if ( v35 + 9257 == v153
      || *((_DWORD *)v153 + 8) > 0xBAu
      || *((_DWORD *)v153 + 8) == 186 && v36 < *((_BYTE *)v153 + 36) )
    {
LABEL_157:
      do
      {
        do
          ++v36;
        while ( !v36 );
      }
      while ( !v35[v36 + 15] || *((_BYTE *)v35 + 259 * v36 + 2608) == 1 );
    }
    else
    {
      v36 = *((_BYTE *)v153 + 40);
    }
    *((_QWORD *)&v250 + 1) = v34;
    *(_QWORD *)&v250 = v33;
    v155 = sub_1D309E0(
             (__int64 *)a1[2],
             158,
             (__int64)&v316,
             v36,
             0,
             0,
             *(double *)v11.m128i_i64,
             *(double *)v18.m128i_i64,
             *(double *)v21.m128i_i64,
             v250);
    v156 = (_QWORD *)a1[2];
    v305 = v155;
    v306 = v157;
    v158 = sub_1D2BF40(
             v156,
             v280,
             v279,
             (__int64)&v316,
             v155,
             (unsigned int)v157 | v34 & 0xFFFFFFFF00000000LL,
             v281.m128i_i64[0],
             v281.m128i_i64[1],
             *(_OWORD *)*(_QWORD *)(a2 + 104),
             *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
             v278,
             (unsigned __int16)v277,
             (__int64)&v322);
    sub_1D44850(a1[2], a2, 0, v158, v159);
    v161 = a1[4];
    if ( v161 )
    {
      *(_QWORD *)&v326 = v158;
      sub_1FF5010(v161, &v326, v160, v42, v43, v44);
    }
    v46 = a1[3];
    *(_QWORD *)&v326 = a2;
    result = *(_QWORD **)(v46 + 8);
    if ( *(_QWORD **)(v46 + 16) != result )
      goto LABEL_18;
    v47 = (__int64)&result[*(unsigned int *)(v46 + 28)];
    if ( result != (_QWORD *)v47 )
    {
      while ( a2 != *result )
      {
        if ( (_QWORD *)v47 == ++result )
          goto LABEL_93;
      }
      goto LABEL_174;
    }
    goto LABEL_93;
  }
  if ( v37 == 4 )
  {
    result = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, __int64))(*v35 + 1312LL))(
                         v35,
                         a2,
                         0,
                         a1[2]);
    v40 = result;
    if ( !result || (_QWORD *)a2 == result && !v39 )
      goto LABEL_39;
    sub_1D44850(a1[2], a2, 0, (__int64)result, v39);
    v45 = a1[4];
    if ( v45 )
    {
      *(_QWORD *)&v326 = v40;
      sub_1FF5010(v45, &v326, v41, v42, v43, v44);
    }
    v46 = a1[3];
    *(_QWORD *)&v326 = a2;
    result = *(_QWORD **)(v46 + 8);
    if ( *(_QWORD **)(v46 + 16) != result )
    {
LABEL_18:
      result = sub_16CC9F0(v46, a2);
      if ( a2 == *result )
      {
        v162 = *(_QWORD *)(v46 + 16);
        if ( v162 == *(_QWORD *)(v46 + 8) )
          v42 = *(unsigned int *)(v46 + 28);
        else
          v42 = *(unsigned int *)(v46 + 24);
        v47 = v162 + 8 * v42;
      }
      else
      {
        result = *(_QWORD **)(v46 + 16);
        if ( result != *(_QWORD **)(v46 + 8) )
          goto LABEL_20;
        result += *(unsigned int *)(v46 + 28);
        v47 = (__int64)result;
      }
LABEL_174:
      if ( (_QWORD *)v47 != result )
      {
        *result = -2;
        ++*(_DWORD *)(v46 + 32);
      }
LABEL_20:
      v48 = a1[4];
      if ( !v48 )
        goto LABEL_39;
      goto LABEL_21;
    }
    v47 = (__int64)&result[*(unsigned int *)(v46 + 28)];
    if ( result != (_QWORD *)v47 )
    {
      while ( a2 != *result )
      {
        if ( (_QWORD *)v47 == ++result )
          goto LABEL_93;
      }
      goto LABEL_174;
    }
LABEL_93:
    result = (_QWORD *)v47;
    goto LABEL_174;
  }
  v102 = *(_QWORD *)(a2 + 104);
  v103 = *(_QWORD *)(a2 + 96);
  v282.m128i_i64[0] = *(unsigned __int8 *)(a2 + 88);
  v104 = sub_1E340A0(v102);
  v105 = sub_1E34390(*(_QWORD *)(a2 + 104));
  v106 = sub_1E0A0C0(*(_QWORD *)(a1[2] + 32));
  result = (_QWORD *)sub_1F43CC0(a1[1], *(_QWORD *)(a1[2] + 48), v106, v282.m128i_i64[0], v103, v104, v105, 0);
  if ( !(_BYTE)result )
  {
    v107 = sub_20BB820(a1[1], a2, a1[2]);
    sub_1D44850(a1[2], a2, 0, v107, v108);
    v110 = a1[4];
    if ( v110 )
    {
      *(_QWORD *)&v326 = v107;
      sub_1FF5010(v110, &v326, v109, v42, v43, v44);
    }
    v46 = a1[3];
    *(_QWORD *)&v326 = a2;
    result = *(_QWORD **)(v46 + 8);
    if ( *(_QWORD **)(v46 + 16) != result )
      goto LABEL_18;
    v47 = (__int64)&result[*(unsigned int *)(v46 + 28)];
    if ( result != (_QWORD *)v47 )
    {
      while ( a2 != *result )
      {
        if ( (_QWORD *)v47 == ++result )
          goto LABEL_93;
      }
      goto LABEL_174;
    }
    goto LABEL_93;
  }
LABEL_39:
  if ( v316 )
    return (_QWORD *)sub_161E7C0((__int64)&v316, v316);
  return result;
}
