// Function: sub_92CBF0
// Address: 0x92cbf0
//
__m128i *__fastcall sub_92CBF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  _DWORD *v6; // rcx
  __m128i *result; // rax
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r12
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // rax
  char k; // dl
  __int64 v21; // rsi
  __int64 v22; // r12
  _BYTE *v23; // rbx
  _BYTE *v24; // rax
  __int64 v25; // rcx
  _BYTE *v26; // r15
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int **v30; // r12
  _BYTE *v31; // rbx
  __int64 v32; // rax
  unsigned int *v33; // rdi
  __m128i *v34; // r10
  __int64 (__fastcall *v35)(__int64, unsigned int, _BYTE *, __m128i *); // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r15
  unsigned int **v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r13
  unsigned int *v42; // rdi
  __int64 (__fastcall *v43)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v44; // rax
  __int64 v45; // r13
  unsigned int *v46; // rbx
  unsigned int *v47; // r13
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // r15
  __int64 v51; // r13
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // rax
  char v56; // dl
  __int64 v57; // rsi
  char v58; // cl
  unsigned __int8 v59; // dl
  __int64 *v60; // rbx
  __int64 *v61; // r13
  __int64 v62; // r12
  __int64 v63; // rsi
  __int64 v64; // rt0
  __int64 *v65; // rt1
  __int64 v66; // rbx
  __int64 v67; // r15
  __int64 v68; // r12
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // r15
  __int64 v73; // r13
  __int64 v74; // rbx
  __int64 v75; // r12
  __int64 v76; // rsi
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdi
  __int64 v80; // r13
  __int64 v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rbx
  unsigned int **v84; // rbx
  __m128i *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // r15
  unsigned int *v88; // r15
  __int64 v89; // rbx
  __int64 v90; // rdx
  __int64 v91; // rsi
  int v92; // eax
  int v93; // eax
  unsigned int v94; // edx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rbx
  int v99; // eax
  int v100; // eax
  unsigned int v101; // edx
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rbx
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // rbx
  __int64 *v108; // rdx
  __int64 v109; // rax
  __int64 v110; // r15
  __int64 v111; // rdi
  __int64 v112; // r12
  __int64 v113; // rax
  __int64 v114; // r13
  __int64 v115; // rbx
  __int64 v116; // rax
  __int64 v117; // rdi
  __int64 v118; // rax
  unsigned __int16 v119; // bx
  __int64 v120; // r15
  __int64 v121; // rax
  __int64 v122; // r13
  __int64 v123; // r15
  __int64 v124; // rcx
  __int64 v125; // rbx
  __int64 v126; // rcx
  int v127; // eax
  int v128; // eax
  unsigned int v129; // esi
  __int64 v130; // rax
  __int64 v131; // rsi
  __int64 v132; // rsi
  __int64 v133; // rbx
  __int64 v134; // r15
  int v135; // eax
  int v136; // eax
  unsigned int v137; // edx
  __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rdx
  __int64 v141; // rdi
  unsigned int **v142; // rbx
  __int64 v143; // r15
  unsigned int *v144; // rdi
  __int64 (__fastcall *v145)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v146; // rax
  __int64 v147; // r13
  unsigned int *v148; // rbx
  unsigned int *v149; // r13
  __int64 v150; // rdx
  __int64 v151; // rsi
  __int64 v152; // r15
  __int64 v153; // rdi
  __int64 v154; // r12
  __int64 v155; // rax
  __int64 v156; // r13
  __int64 v157; // rbx
  __int64 v158; // rax
  __int64 v159; // rdi
  __int64 v160; // rax
  unsigned __int16 v161; // bx
  __int64 v162; // r15
  __int64 v163; // rax
  __int64 v164; // r15
  __int64 v165; // rcx
  __int64 v166; // rbx
  __int64 v167; // rcx
  int v168; // eax
  int v169; // eax
  unsigned int v170; // esi
  __int64 v171; // rax
  __int64 v172; // rsi
  __int64 v173; // rsi
  __int64 v174; // rbx
  __int64 v175; // r15
  int v176; // eax
  int v177; // eax
  unsigned int v178; // edx
  __int64 v179; // rax
  __int64 v180; // rdx
  __int64 v181; // rdx
  __int64 v182; // rdi
  unsigned int **v183; // rbx
  __int64 v184; // r15
  unsigned int *v185; // rdi
  __int64 (__fastcall *v186)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v187; // rax
  __int64 v188; // r13
  unsigned int *v189; // rbx
  unsigned int *v190; // r13
  __int64 v191; // rdx
  __int64 v192; // rsi
  __int64 v193; // r15
  __int64 *v194; // rdx
  __int64 i; // rax
  __int64 v196; // rax
  __int64 v197; // rsi
  int v198; // ebx
  __int64 v199; // rdx
  __int64 v200; // rcx
  __int64 v201; // r9
  __int64 v202; // r8
  unsigned __int64 v203; // rax
  int v204; // r12d
  int v205; // edx
  unsigned __int64 v206; // rax
  unsigned __int64 v207; // rax
  char j; // dl
  void *v209; // rdx
  __int64 v210; // rbx
  _BYTE *v211; // rax
  __int64 v212; // rsi
  __int64 v213; // rdx
  _DWORD *v214; // rbx
  __int64 v215; // rdx
  __int64 v216; // rcx
  __int64 v217; // r8
  __int64 v218; // r9
  __int32 v219; // r15d
  __int64 v220; // r8
  _BYTE *v221; // rax
  __m128i *v222; // rsi
  __int32 *v223; // rdi
  __int64 v224; // rcx
  __int64 v225; // rcx
  __int64 v226; // r8
  __int64 v227; // r9
  __m128i v228; // xmm6
  __m128i v229; // xmm7
  __m128i v230; // xmm6
  __int64 v231; // r8
  __int64 v232; // rcx
  __int64 v233; // rdx
  _QWORD *v234; // rbx
  __int64 v235; // r15
  __int64 v236; // r12
  __int64 v237; // r15
  __int64 v238; // rax
  __int64 v239; // rdx
  __int64 v240; // rcx
  char v241; // dl
  __int64 v242; // rbx
  __int64 v243; // rax
  unsigned int **v244; // rdi
  __int64 v245; // r13
  __int64 v246; // rax
  unsigned int **v247; // rdi
  _BYTE *v248; // r12
  _BYTE *v249; // rax
  unsigned int **v250; // rdi
  _BYTE *v251; // r15
  __int64 v252; // rax
  unsigned int **v253; // r13
  _BYTE *v254; // rbx
  unsigned int *v255; // rdi
  __int64 (__fastcall *v256)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  unsigned int *v257; // rbx
  __int64 v258; // r13
  __int64 v259; // rdx
  __int64 v260; // rsi
  __int64 *v261; // r13
  __int64 *v262; // rbx
  __int64 v263; // r12
  _BYTE *v264; // rax
  __int64 v265; // rax
  unsigned int **v266; // r14
  _BYTE *v267; // r15
  __int64 v268; // rax
  unsigned int *v269; // rdi
  _BYTE *v270; // r13
  __int64 (__fastcall *v271)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  unsigned int *v272; // rbx
  __int64 v273; // r13
  __int64 v274; // rdx
  __int64 v275; // rsi
  __int64 *v276; // r12
  __int64 v277; // rax
  unsigned __int64 v278; // r15
  __int64 v279; // rbx
  char v280; // al
  __int64 v281; // rax
  __int64 v282; // rcx
  __int64 v283; // rbx
  int v284; // eax
  unsigned int **v285; // r13
  unsigned int *v286; // rdi
  unsigned int v287; // r15d
  __int64 (__fastcall *v288)(__int64, unsigned int, _BYTE *); // rax
  int v289; // eax
  __int64 v290; // r15
  __int64 v291; // rcx
  __int64 v292; // rax
  unsigned int **v293; // rdi
  unsigned int **v294; // r13
  unsigned int *v295; // rdi
  __int64 (__fastcall *v296)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  unsigned int *v297; // rbx
  __int64 v298; // r13
  __int64 v299; // rdx
  __int64 v300; // rsi
  unsigned int **v301; // r13
  unsigned int *v302; // rdi
  __int64 (__fastcall *v303)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  unsigned int *v304; // rbx
  __int64 v305; // r13
  __int64 v306; // rdx
  __int64 v307; // rsi
  unsigned int **v308; // r13
  unsigned int *v309; // rdi
  __int64 (__fastcall *v310)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  unsigned int *v311; // rbx
  __int64 v312; // r13
  __int64 v313; // rdx
  __int64 v314; // rsi
  unsigned int **v315; // r12
  _BYTE *v316; // rax
  char v317; // r9
  unsigned int **v318; // r13
  __int64 v319; // rax
  __int64 v320; // rbx
  __int64 v321; // rax
  __int64 v322; // rax
  unsigned int *v323; // rdi
  __int64 (__fastcall *v324)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v325; // rax
  __int64 v326; // r11
  _BYTE **v327; // rcx
  __int64 v328; // rax
  unsigned int *v329; // rbx
  __int64 v330; // r13
  __int64 v331; // rdx
  __int64 v332; // rsi
  __int64 v333; // rdi
  unsigned int *v334; // rbx
  __int64 v335; // r12
  __int64 v336; // rdx
  __int64 v337; // rsi
  __int64 v338; // rcx
  __int64 v339; // r9
  unsigned __int64 v340; // rsi
  __int64 v341; // rax
  __int64 v342; // rax
  __int64 v343; // rax
  __int64 v344; // rdi
  __int64 *v345; // rsi
  __int64 v346; // rax
  unsigned int *v347; // rdx
  unsigned int *v348; // rbx
  __int64 v349; // r13
  __int64 v350; // rdx
  __int64 v351; // rsi
  unsigned int *v352; // rdx
  unsigned int v353; // r15d
  __int64 v354; // rax
  __m128i *v355; // rsi
  __int64 v356; // rcx
  __int32 *v357; // rdi
  __int64 v358; // rcx
  int v359; // edx
  int v360; // edx
  int v361; // ecx
  __int64 v362; // rdi
  __m128i v363; // [rsp-40h] [rbp-1A0h] BYREF
  __m128i v364; // [rsp-30h] [rbp-190h]
  __m128i v365; // [rsp-20h] [rbp-180h]
  __int64 v366; // [rsp-10h] [rbp-170h]
  __int64 v367; // [rsp-8h] [rbp-168h]
  __m128i *v368; // [rsp+0h] [rbp-160h]
  __int64 v369; // [rsp+8h] [rbp-158h]
  __int64 v370; // [rsp+10h] [rbp-150h]
  __m128i *v371; // [rsp+18h] [rbp-148h]
  _BYTE *v372; // [rsp+28h] [rbp-138h] BYREF
  _BYTE *v373; // [rsp+30h] [rbp-130h]
  int v374; // [rsp+38h] [rbp-128h]
  char v375; // [rsp+3Ch] [rbp-124h]
  __int64 v376; // [rsp+40h] [rbp-120h]
  _BYTE *v377; // [rsp+50h] [rbp-110h] BYREF
  __int64 v378; // [rsp+58h] [rbp-108h]
  __int64 v379; // [rsp+60h] [rbp-100h] BYREF
  __m128i v380; // [rsp+70h] [rbp-F0h] BYREF
  __m128i v381; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v382; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v383; // [rsp+A0h] [rbp-C0h]
  __m128i v384; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v385; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v386; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v387; // [rsp+E0h] [rbp-80h]
  __m128i v388; // [rsp+F0h] [rbp-70h] BYREF
  __m128i v389; // [rsp+100h] [rbp-60h]
  __m128i v390; // [rsp+110h] [rbp-50h]
  __int64 v391; // [rsp+120h] [rbp-40h]

  v5 = *a1;
  while ( 2 )
  {
    v6 = &dword_4D04720;
    if ( !(dword_4D04720 | unk_4D04658) && (*(_WORD *)(a2 + 24) & 0x10FF) != 0x1002 )
    {
      v388.m128i_i64[0] = *(_QWORD *)(a2 + 36);
      if ( v388.m128i_i32[0] )
      {
        sub_92FD10(v5, &v388);
        sub_91CAC0(&v388);
      }
    }
    switch ( *(_BYTE *)(a2 + 24) )
    {
      case 1:
        switch ( *(_BYTE *)(a2 + 56) )
        {
          case 0:
            v27 = *(_QWORD *)(a2 + 72);
            v28 = sub_72B0F0(v27, 0);
            sub_91C930(v28, (_DWORD *)(a2 + 36));
            sub_926800((__int64)&v388, *a1, v27);
            return (__m128i *)v388.m128i_i64[1];
          case 3:
          case 6:
          case 8:
          case 0x5C:
          case 0x5E:
          case 0x5F:
            return (__m128i *)sub_9288E0(a1, a2);
          case 5:
            v276 = *(__int64 **)(a2 + 72);
            if ( !sub_91B770(*v276) )
            {
              v277 = sub_92CBF0(a1, v276);
              v278 = *(_QWORD *)a2;
              v279 = v277;
              v280 = sub_91B6F0(*v276);
              return (__m128i *)sub_92C930(a1, v279, v280, v278, (_DWORD *)(a2 + 36));
            }
            v345 = v276;
            v13 = 0;
            sub_947E80(*a1, v345, 0, 0, 0);
            return (__m128i *)v13;
          case 0x15:
            sub_926800((__int64)&v388, *a1, *(_QWORD *)(a2 + 72));
            v290 = v388.m128i_i64[1];
            if ( sub_8D23B0(**(_QWORD **)(a2 + 72)) )
            {
              v292 = sub_91A390(*(_QWORD *)(*a1 + 32) + 8LL, *(_QWORD *)a2, 0, v291);
              v293 = (unsigned int **)a1[1];
              v386.m128i_i16[0] = 257;
              return (__m128i *)sub_929600(v293, 0x31u, v290, v292, (__int64)&v384, 0, v380.m128i_u32[0], 0);
            }
            v318 = (unsigned int **)a1[1];
            v380.m128i_i64[0] = (__int64)"arraydecay";
            v319 = *a1;
            v382.m128i_i16[0] = 259;
            v320 = sub_91A390(*(_QWORD *)(v319 + 32) + 8LL, v389.m128i_u64[0], 0, v291);
            v321 = sub_BCB2D0(v318[9]);
            v377 = (_BYTE *)sub_ACD640(v321, 0, 0);
            v322 = sub_BCB2D0(v318[9]);
            v378 = sub_ACD640(v322, 0, 0);
            v323 = v318[10];
            v324 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v323 + 64LL);
            if ( v324 == sub_920540 )
            {
              if ( (unsigned __int8)sub_BCEA30(v320) )
                goto LABEL_293;
              if ( *(_BYTE *)v290 > 0x15u )
                goto LABEL_293;
              v325 = sub_928980(&v377, (__int64)&v379);
              if ( v327 != v325 )
                goto LABEL_293;
              v367 = v326;
              v386.m128i_i8[0] = 0;
              v13 = sub_AD9FD0(v320, v290, (unsigned int)&v377, 2, 7, (unsigned int)&v384, 0);
              if ( v386.m128i_i8[0] )
              {
                v386.m128i_i8[0] = 0;
                if ( v385.m128i_i32[2] > 0x40u && v385.m128i_i64[0] )
                  j_j___libc_free_0_0(v385.m128i_i64[0]);
                if ( v384.m128i_i32[2] > 0x40u && v384.m128i_i64[0] )
                  j_j___libc_free_0_0(v384.m128i_i64[0]);
              }
            }
            else
            {
              v13 = v324((__int64)v323, v320, (_BYTE *)v290, &v377, 2, 7);
            }
            if ( v13 )
              return (__m128i *)v13;
LABEL_293:
            v386.m128i_i16[0] = 257;
            v13 = sub_BD2C40(88, 3);
            if ( !v13 )
              goto LABEL_296;
            v328 = *(_QWORD *)(v290 + 8);
            if ( (unsigned int)*(unsigned __int8 *)(v328 + 8) - 17 <= 1 )
              goto LABEL_295;
            v358 = *((_QWORD *)v377 + 1);
            v359 = *(unsigned __int8 *)(v358 + 8);
            if ( v359 == 17 )
              goto LABEL_349;
            if ( v359 == 18 )
            {
              LOBYTE(v360) = 18;
            }
            else
            {
              v358 = *(_QWORD *)(v378 + 8);
              v360 = *(unsigned __int8 *)(v358 + 8);
              if ( v360 == 17 )
              {
LABEL_349:
                LOBYTE(v360) = 17;
              }
              else if ( v360 != 18 )
              {
                goto LABEL_295;
              }
            }
            v361 = *(_DWORD *)(v358 + 32);
            v362 = *(_QWORD *)(v290 + 8);
            BYTE4(v373) = (_BYTE)v360 == 18;
            LODWORD(v373) = v361;
            v328 = sub_BCE1B0(v362, v373);
LABEL_295:
            sub_B44260(v13, v328, 34, 3, 0, 0);
            *(_QWORD *)(v13 + 72) = v320;
            *(_QWORD *)(v13 + 80) = sub_B4DC50(v320, &v377, 2);
            sub_B4D9A0(v13, v290, &v377, 2, &v384);
LABEL_296:
            sub_B4DDE0(v13, 7);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v318[11] + 16LL))(
              v318[11],
              v13,
              &v380,
              v318[7],
              v318[8]);
            v329 = *v318;
            v330 = (__int64)&(*v318)[4 * *((unsigned int *)v318 + 2)];
            while ( (unsigned int *)v330 != v329 )
            {
              v331 = *((_QWORD *)v329 + 1);
              v332 = *v329;
              v329 += 4;
              sub_B99FD0(v13, v332, v331);
            }
            return (__m128i *)v13;
          case 0x19:
            a2 = *(_QWORD *)(a2 + 72);
            v5 = *a1;
            continue;
          case 0x1A:
            v281 = sub_92CBF0(a1, *(_QWORD *)(a2 + 72));
            v282 = *(_QWORD *)(v281 + 8);
            v283 = v281;
            v284 = *(unsigned __int8 *)(v282 + 8);
            if ( (unsigned int)(v284 - 17) <= 1 )
              LOBYTE(v284) = *(_BYTE *)(**(_QWORD **)(v282 + 16) + 8LL);
            if ( (unsigned __int8)v284 > 3u && (_BYTE)v284 != 5 && (v284 & 0xFD) != 4 )
            {
              if ( (unsigned __int8)sub_91B6F0(*(_QWORD *)a2) )
              {
                v315 = (unsigned int **)a1[1];
                v388.m128i_i64[0] = (__int64)"neg";
                v390.m128i_i16[0] = 259;
                v316 = (_BYTE *)sub_AD6530(*(_QWORD *)(v283 + 8));
                v317 = 1;
              }
              else
              {
                v315 = (unsigned int **)a1[1];
                v388.m128i_i64[0] = (__int64)"neg";
                v390.m128i_i16[0] = 259;
                v316 = (_BYTE *)sub_AD6530(*(_QWORD *)(v283 + 8));
                v317 = 0;
              }
              return (__m128i *)sub_929DE0(v315, v316, (_BYTE *)v283, (__int64)&v388, 0, v317);
            }
            v285 = (unsigned int **)a1[1];
            v384.m128i_i64[0] = (__int64)"neg";
            v386.m128i_i16[0] = 259;
            v286 = v285[10];
            v287 = *((_DWORD *)v285 + 26);
            v288 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*(_QWORD *)v286 + 48LL);
            if ( v288 == sub_9288C0 )
            {
              if ( *(_BYTE *)v283 > 0x15u )
                goto LABEL_311;
              v13 = sub_AAAFF0(12, v283);
            }
            else
            {
              v13 = ((__int64 (__fastcall *)(unsigned int *, __int64, __int64, _QWORD))v288)(v286, 12, v283, v287);
            }
            if ( v13 )
              goto LABEL_235;
            v287 = *((_DWORD *)v285 + 26);
LABEL_311:
            v390.m128i_i16[0] = 257;
            v346 = sub_B50340(12, v283, &v388, 0, 0);
            v347 = v285[12];
            v13 = v346;
            if ( v347 )
              sub_B99FD0(v346, 3, v347);
            sub_B45150(v13, v287);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v285[11] + 16LL))(
              v285[11],
              v13,
              &v384,
              v285[7],
              v285[8]);
            v348 = *v285;
            v349 = (__int64)&(*v285)[4 * *((unsigned int *)v285 + 2)];
            while ( (unsigned int *)v349 != v348 )
            {
              v350 = *((_QWORD *)v348 + 1);
              v351 = *v348;
              v348 += 4;
              sub_B99FD0(v13, v351, v350);
            }
LABEL_235:
            if ( unk_4D04700 && *(_BYTE *)v13 > 0x1Cu )
            {
              v289 = sub_B45210(v13);
              sub_B45150(v13, v289 | 1u);
            }
            return (__m128i *)v13;
          case 0x1C:
            v265 = sub_92CBF0(a1, *(_QWORD *)(a2 + 72));
            v266 = (unsigned int **)a1[1];
            v267 = (_BYTE *)v265;
            v386.m128i_i16[0] = 259;
            v384.m128i_i64[0] = (__int64)"not";
            v268 = sub_AD62B0(*(_QWORD *)(v265 + 8));
            v269 = v266[10];
            v270 = (_BYTE *)v268;
            v271 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v269 + 16LL);
            if ( v271 == sub_9202E0 )
            {
              if ( *v267 > 0x15u || *v270 > 0x15u )
                goto LABEL_222;
              if ( (unsigned __int8)sub_AC47B0(30) )
                v13 = sub_AD5570(30, v267, v270, 0, 0);
              else
                v13 = sub_AABE40(30, v267, v270);
            }
            else
            {
              v13 = v271((__int64)v269, 30u, v267, v270);
            }
            if ( v13 )
              return (__m128i *)v13;
LABEL_222:
            v390.m128i_i16[0] = 257;
            v13 = sub_B504D0(30, v267, v270, &v388, 0, 0);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v266[11] + 16LL))(
              v266[11],
              v13,
              &v384,
              v266[7],
              v266[8]);
            v272 = *v266;
            v273 = (__int64)&(*v266)[4 * *((unsigned int *)v266 + 2)];
            if ( *v266 != (unsigned int *)v273 )
            {
              do
              {
                v274 = *((_QWORD *)v272 + 1);
                v275 = *v272;
                v272 += 4;
                sub_B99FD0(v13, v275, v274);
              }
              while ( (unsigned int *)v273 != v272 );
            }
            return (__m128i *)v13;
          case 0x1D:
            v29 = sub_921E00(*a1, *(_QWORD *)(a2 + 72));
            v30 = (unsigned int **)a1[1];
            v31 = (_BYTE *)v29;
            v386.m128i_i16[0] = 259;
            v384.m128i_i64[0] = (__int64)"lnot";
            v32 = sub_AD62B0(*(_QWORD *)(v29 + 8));
            v33 = v30[10];
            v34 = (__m128i *)v32;
            v35 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __m128i *))(*(_QWORD *)v33 + 16LL);
            if ( (char *)v35 == (char *)sub_9202E0 )
            {
              if ( *v31 > 0x15u || v34->m128i_i8[0] > 0x15u )
                goto LABEL_300;
              v371 = v34;
              if ( (unsigned __int8)sub_AC47B0(30) )
                v36 = sub_AD5570(30, v31, v371, 0, 0);
              else
                v36 = sub_AABE40(30, v31, v371);
              v34 = v371;
              v38 = v36;
            }
            else
            {
              v371 = v34;
              v354 = v35((__int64)v33, 30u, v31, v34);
              v34 = v371;
              v38 = v354;
            }
            if ( v38 )
              goto LABEL_28;
LABEL_300:
            v390.m128i_i16[0] = 257;
            v38 = sub_B504D0(30, v31, v34, &v388, 0, 0);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v30[11] + 16LL))(
              v30[11],
              v38,
              &v384,
              v30[7],
              v30[8]);
            v334 = *v30;
            v335 = (__int64)&(*v30)[4 * *((unsigned int *)v30 + 2)];
            while ( (unsigned int *)v335 != v334 )
            {
              v336 = *((_QWORD *)v334 + 1);
              v337 = *v334;
              v334 += 4;
              sub_B99FD0(v38, v337, v336);
            }
LABEL_28:
            v39 = (unsigned int **)a1[1];
            v384.m128i_i64[0] = (__int64)"lnot.ext";
            v40 = *a1;
            v386.m128i_i16[0] = 259;
            v41 = sub_91A390(*(_QWORD *)(v40 + 32) + 8LL, *(_QWORD *)a2, 0, v37);
            if ( v41 == *(_QWORD *)(v38 + 8) )
              return (__m128i *)v38;
            v42 = v39[10];
            v43 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v42 + 120LL);
            if ( v43 == sub_920130 )
            {
              if ( *(_BYTE *)v38 > 0x15u )
                goto LABEL_34;
              if ( (unsigned __int8)sub_AC4810(39) )
                v13 = sub_ADAB70(39, v38, v41, 0);
              else
                v13 = sub_AA93C0(39, v38, v41);
            }
            else
            {
              v13 = v43((__int64)v42, 39u, (_BYTE *)v38, v41);
            }
            if ( v13 )
              return (__m128i *)v13;
LABEL_34:
            v390.m128i_i16[0] = 257;
            v44 = sub_BD2C40(72, unk_3F10A14);
            v13 = v44;
            if ( v44 )
              sub_B515B0(v44, v38, v41, &v388, 0, 0);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v39[11] + 16LL))(
              v39[11],
              v13,
              &v384,
              v39[7],
              v39[8]);
            v45 = 4LL * *((unsigned int *)v39 + 2);
            v46 = *v39;
            v47 = &v46[v45];
            while ( v47 != v46 )
            {
              v48 = *((_QWORD *)v46 + 1);
              v49 = *v46;
              v46 += 4;
              sub_B99FD0(v13, v49, v48);
            }
            return (__m128i *)v13;
          case 0x1E:
          case 0x1F:
          case 0x41:
          case 0x42:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x59:
          case 0x5A:
          case 0x5D:
          case 0x68:
            return (__m128i *)sub_91DAF0(*(_QWORD *)(*a1 + 32), *(_QWORD *)a2, a3, (__int64)v6);
          case 0x23:
            v58 = 0;
            v59 = 1;
            return (__m128i *)sub_92AEE0(a1, a2, v59, v58);
          case 0x24:
            v58 = 0;
            goto LABEL_43;
          case 0x25:
            v58 = 1;
            v59 = 1;
            return (__m128i *)sub_92AEE0(a1, a2, v59, v58);
          case 0x26:
            v58 = 1;
LABEL_43:
            v59 = 0;
            return (__m128i *)sub_92AEE0(a1, a2, v59, v58);
          case 0x27:
          case 0x28:
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x35:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
            v21 = *(_QWORD *)(a2 + 72);
            v22 = *(_QWORD *)(v21 + 16);
            v23 = (_BYTE *)sub_92CBF0(a1, v21);
            v24 = (_BYTE *)sub_92CBF0(a1, v22);
            v25 = *(_QWORD *)a2;
            v26 = v24;
            switch ( *(_BYTE *)(a2 + 56) )
            {
              case '\'':
                return (__m128i *)sub_92A460((__int64)a1, (__int64)v23, v24, v25);
              case '(':
                return (__m128i *)sub_929F70((__int64)a1, (__int64)v23, v24, v25);
              case ')':
                return (__m128i *)sub_929130((__int64)a1, (__int64)v23, v24, v25);
              case '*':
                return (__m128i *)sub_92B6A0(a1, (__int64)v23, v24, v25);
              case '+':
                return (__m128i *)sub_928EC0((__int64)a1, v23, v24, v25);
              case '5':
                return (__m128i *)sub_9297A0((__int64)a1, (__int64)v23, (__int64)v24);
              case '6':
                return (__m128i *)sub_929960((__int64)a1, (__int64)v23, (__int64)v24, v25);
              case '7':
                v308 = (unsigned int **)a1[1];
                v384.m128i_i64[0] = (__int64)"and";
                v386.m128i_i16[0] = 259;
                v309 = v308[10];
                v310 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v309 + 16LL);
                if ( v310 == sub_9202E0 )
                {
                  if ( *v23 > 0x15u || *v26 > 0x15u )
                    goto LABEL_274;
                  if ( (unsigned __int8)sub_AC47B0(28) )
                    v13 = sub_AD5570(28, v23, v26, 0, 0);
                  else
                    v13 = sub_AABE40(28, v23, v26);
                }
                else
                {
                  v13 = v310((__int64)v309, 28u, v23, v26);
                }
                if ( v13 )
                  return (__m128i *)v13;
LABEL_274:
                v390.m128i_i16[0] = 257;
                v13 = sub_B504D0(28, v23, v26, &v388, 0, 0);
                (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v308[11] + 16LL))(
                  v308[11],
                  v13,
                  &v384,
                  v308[7],
                  v308[8]);
                v311 = *v308;
                v312 = (__int64)&(*v308)[4 * *((unsigned int *)v308 + 2)];
                while ( (unsigned int *)v312 != v311 )
                {
                  v313 = *((_QWORD *)v311 + 1);
                  v314 = *v311;
                  v311 += 4;
                  sub_B99FD0(v13, v314, v313);
                }
                return (__m128i *)v13;
              case '8':
                v301 = (unsigned int **)a1[1];
                v384.m128i_i64[0] = (__int64)"or";
                v386.m128i_i16[0] = 259;
                v302 = v301[10];
                v303 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v302 + 16LL);
                if ( v303 == sub_9202E0 )
                {
                  if ( *v23 > 0x15u || *v26 > 0x15u )
                    goto LABEL_264;
                  if ( (unsigned __int8)sub_AC47B0(29) )
                    v13 = sub_AD5570(29, v23, v26, 0, 0);
                  else
                    v13 = sub_AABE40(29, v23, v26);
                }
                else
                {
                  v13 = v303((__int64)v302, 29u, v23, v26);
                }
                if ( v13 )
                  return (__m128i *)v13;
LABEL_264:
                v390.m128i_i16[0] = 257;
                v13 = sub_B504D0(29, v23, v26, &v388, 0, 0);
                (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v301[11] + 16LL))(
                  v301[11],
                  v13,
                  &v384,
                  v301[7],
                  v301[8]);
                v304 = *v301;
                v305 = (__int64)&(*v301)[4 * *((unsigned int *)v301 + 2)];
                while ( (unsigned int *)v305 != v304 )
                {
                  v306 = *((_QWORD *)v304 + 1);
                  v307 = *v304;
                  v304 += 4;
                  sub_B99FD0(v13, v307, v306);
                }
                return (__m128i *)v13;
              case '9':
                v294 = (unsigned int **)a1[1];
                v384.m128i_i64[0] = (__int64)"xor";
                v386.m128i_i16[0] = 259;
                v295 = v294[10];
                v296 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v295 + 16LL);
                if ( v296 == sub_9202E0 )
                {
                  if ( *v23 > 0x15u || *v26 > 0x15u )
                    goto LABEL_254;
                  if ( (unsigned __int8)sub_AC47B0(30) )
                    v13 = sub_AD5570(30, v23, v26, 0, 0);
                  else
                    v13 = sub_AABE40(30, v23, v26);
                }
                else
                {
                  v13 = v296((__int64)v295, 30u, v23, v26);
                }
                if ( v13 )
                  return (__m128i *)v13;
LABEL_254:
                v390.m128i_i16[0] = 257;
                v13 = sub_B504D0(30, v23, v26, &v388, 0, 0);
                (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v294[11] + 16LL))(
                  v294[11],
                  v13,
                  &v384,
                  v294[7],
                  v294[8]);
                v297 = *v294;
                v298 = (__int64)&(*v294)[4 * *((unsigned int *)v294 + 2)];
                while ( (unsigned int *)v298 != v297 )
                {
                  v299 = *((_QWORD *)v297 + 1);
                  v300 = *v297;
                  v297 += 4;
                  sub_B99FD0(v13, v300, v299);
                }
                break;
              default:
                sub_91B8A0("unsupported binary expression!", (_DWORD *)(a2 + 36), 1);
            }
            return (__m128i *)v13;
          case 0x32:
            v60 = *(__int64 **)(a2 + 72);
            v61 = (__int64 *)v60[2];
            v62 = sub_92CBF0(a1, v60);
            v63 = sub_92CBF0(a1, v61);
            if ( *(_BYTE *)(*(_QWORD *)(v62 + 8) + 8LL) == 14 )
            {
              v64 = v62;
              v62 = v63;
              v63 = v64;
              v65 = v61;
              v61 = v60;
              v60 = v65;
            }
            return (__m128i *)sub_92AA70(a1, v63, (_BYTE *)v62, *v60, *v61);
          case 0x33:
            v261 = *(__int64 **)(a2 + 72);
            v262 = (__int64 *)v261[2];
            v263 = sub_92CBF0(a1, v261);
            v264 = (_BYTE *)sub_92CBF0(a1, v262);
            return (__m128i *)sub_92A590(a1, v263, v264, *v262, *v261);
          case 0x34:
            v234 = *(_QWORD **)(a2 + 72);
            v235 = v234[2];
            v236 = sub_92CBF0(a1, v234);
            v237 = sub_92CBF0(a1, v235);
            v238 = *v234;
            if ( *(_BYTE *)(*v234 + 140LL) == 12 )
            {
              v239 = *v234;
              do
                v239 = *(_QWORD *)(v239 + 160);
              while ( *(_BYTE *)(v239 + 140) == 12 );
              v240 = *(_QWORD *)(v239 + 160);
              do
                v238 = *(_QWORD *)(v238 + 160);
              while ( *(_BYTE *)(v238 + 140) == 12 );
            }
            else
            {
              v240 = *(_QWORD *)(v238 + 160);
            }
            do
            {
              v238 = *(_QWORD *)(v238 + 160);
              v241 = *(_BYTE *)(v238 + 140);
            }
            while ( v241 == 12 );
            v242 = 1;
            if ( v241 != 1 && *(_BYTE *)(*(_QWORD *)(v236 + 8) + 8LL) != 13 )
            {
              while ( *(_BYTE *)(v240 + 140) == 12 )
                v240 = *(_QWORD *)(v240 + 160);
              v242 = *(_QWORD *)(v240 + 128);
            }
            v243 = sub_91A390(*(_QWORD *)(*a1 + 32) + 8LL, *(_QWORD *)a2, 0, v240);
            v244 = (unsigned int **)a1[1];
            v245 = v243;
            v390.m128i_i16[0] = 259;
            v388.m128i_i64[0] = (__int64)"sub.ptr.lhs.cast";
            v371 = &v388;
            v246 = sub_929600(v244, 0x2Fu, v236, v243, (__int64)&v388, 0, v384.m128i_u32[0], 0);
            v247 = (unsigned int **)a1[1];
            v248 = (_BYTE *)v246;
            v388.m128i_i64[0] = (__int64)"sub.ptr.rhs.cast";
            v390.m128i_i16[0] = 259;
            v249 = (_BYTE *)sub_929600(v247, 0x2Fu, v237, v245, (__int64)&v388, 0, v384.m128i_u32[0], 0);
            v250 = (unsigned int **)a1[1];
            v388.m128i_i64[0] = (__int64)"sub.ptr.sub";
            v390.m128i_i16[0] = 259;
            v251 = (_BYTE *)sub_929DE0(v250, v248, v249, (__int64)&v388, 0, 0);
            v13 = (__int64)v251;
            if ( v242 == 1 )
              return (__m128i *)v13;
            v252 = sub_AD64C0(v245, v242, 0);
            v253 = (unsigned int **)a1[1];
            v254 = (_BYTE *)v252;
            v386.m128i_i16[0] = 259;
            v384.m128i_i64[0] = (__int64)"sub.ptr.div";
            v255 = v253[10];
            v256 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v255 + 24LL);
            if ( v256 == sub_920250 )
            {
              if ( *v251 > 0x15u || *v254 > 0x15u )
                goto LABEL_210;
              if ( (unsigned __int8)sub_AC47B0(20) )
                v13 = sub_AD5570(20, v251, v254, 1, 0);
              else
                v13 = sub_AABE40(20, v251, v254);
            }
            else
            {
              v13 = v256((__int64)v255, 20u, v251, v254, 1u);
            }
            if ( v13 )
              return (__m128i *)v13;
LABEL_210:
            v390.m128i_i16[0] = 257;
            v13 = sub_B504D0(20, v251, v254, v371, 0, 0);
            sub_B448B0(v13, 1);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v253[11] + 16LL))(
              v253[11],
              v13,
              &v384,
              v253[7],
              v253[8]);
            v257 = *v253;
            v258 = (__int64)&(*v253)[4 * *((unsigned int *)v253 + 2)];
            while ( (unsigned int *)v258 != v257 )
            {
              v259 = *((_QWORD *)v257 + 1);
              v260 = *v257;
              v257 += 4;
              sub_B99FD0(v13, v260, v259);
            }
            return (__m128i *)v13;
          case 0x3A:
            v231 = 1;
            v232 = 32;
            v233 = 32;
            return (__m128i *)sub_92F480(a1, a2, v233, v232, v231);
          case 0x3B:
            v231 = 14;
            v232 = 33;
            v233 = 33;
            return (__m128i *)sub_92F480(a1, a2, v233, v232, v231);
          case 0x3C:
            v231 = 2;
            v232 = 38;
            v233 = 34;
            return (__m128i *)sub_92F480(a1, a2, v233, v232, v231);
          case 0x3D:
            v231 = 4;
            v232 = 40;
            v233 = 36;
            return (__m128i *)sub_92F480(a1, a2, v233, v232, v231);
          case 0x3E:
            v231 = 3;
            v232 = 39;
            v233 = 35;
            return (__m128i *)sub_92F480(a1, a2, v233, v232, v231);
          case 0x3F:
            v231 = 5;
            v232 = 41;
            v233 = 37;
            return (__m128i *)sub_92F480(a1, a2, v233, v232, v231);
          case 0x49:
            v210 = *(_QWORD *)(a2 + 72);
            v211 = (_BYTE *)sub_92CBF0(a1, *(_QWORD *)(v210 + 16));
            v212 = *a1;
            v213 = v210;
            v372 = v211;
            v214 = (_DWORD *)(v210 + 36);
            sub_926800((__int64)&v384, v212, v213);
            v219 = v384.m128i_i32[0];
            if ( v384.m128i_i32[0] == 1 )
            {
              v220 = *a1;
              v221 = v372;
              if ( (v387 & 1) == 0 )
              {
                v388.m128i_i8[12] &= ~1u;
                v355 = &v384;
                v356 = 14;
                v357 = &v363.m128i_i32[2];
                v388.m128i_i64[0] = (__int64)v372;
                v388.m128i_i32[2] = 0;
                v389.m128i_i32[0] = 0;
                while ( v356 )
                {
                  *v357 = v355->m128i_i32[0];
                  v355 = (__m128i *)((char *)v355 + 4);
                  ++v357;
                  --v356;
                }
                sub_923780(
                  v220,
                  v214,
                  &v372,
                  0,
                  v220,
                  v218,
                  (__int64)v221,
                  v388.m128i_i32[2],
                  v389.m128i_i32[0],
                  v363.m128i_i64[1],
                  v364.m128i_i64[0],
                  v364.m128i_i64[1],
                  v365.m128i_i64[0],
                  v365.m128i_i64[1],
                  v366,
                  v367);
                return (__m128i *)v372;
              }
              BYTE4(v378) &= ~1u;
              v222 = &v384;
              v223 = &v363.m128i_i32[2];
              v377 = v372;
              v224 = 14;
              LODWORD(v378) = 0;
              LODWORD(v379) = 0;
              while ( v224 )
              {
                *v223 = v222->m128i_i32[0];
                v222 = (__m128i *)((char *)v222 + 4);
                ++v223;
                --v224;
              }
              sub_923780(
                v220,
                v214,
                0,
                0,
                v220,
                v218,
                (__int64)v221,
                v378,
                v379,
                v363.m128i_i64[1],
                v364.m128i_i64[0],
                v364.m128i_i64[1],
                v365.m128i_i64[0],
                v365.m128i_i64[1],
                v366,
                v367);
            }
            else
            {
              v333 = *a1;
              v375 &= ~1u;
              v374 = 0;
              LODWORD(v376) = 0;
              v373 = v372;
              sub_925900(
                v333,
                v214,
                v215,
                v216,
                v217,
                v218,
                (__int64)v372,
                0,
                0,
                v384.m128i_i64[0],
                v384.m128i_u64[1],
                v385.m128i_i64[0],
                v385.m128i_i64[1],
                v386.m128i_i64[0],
                v386.m128i_i64[1],
                v387);
            }
            v13 = 0;
            if ( (*(_BYTE *)(a2 + 25) & 4) == 0 )
            {
              v384.m128i_i32[0] = v219;
              v228 = _mm_loadu_si128(&v384);
              v229 = _mm_loadu_si128(&v385);
              v391 = v387;
              v388 = v228;
              v230 = _mm_loadu_si128(&v386);
              v366 = v387;
              v389 = v229;
              v390 = v230;
              v365 = v230;
              v364 = v229;
              v363 = v388;
              sub_9286A0(
                (__int64)&v380,
                *a1,
                v214,
                v225,
                v226,
                v227,
                v363.m128i_i64[0],
                v363.m128i_u64[1],
                v229.m128i_u64[0],
                v229.m128i_i64[1],
                v230.m128i_i64[0],
                v230.m128i_i64[1],
                v387);
              return (__m128i *)v380.m128i_i64[0];
            }
            return (__m128i *)v13;
          case 0x4A:
            v209 = sub_92A460;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x4B:
            v209 = sub_929F70;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x4C:
            v209 = sub_929130;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x4D:
            v209 = sub_92B6A0;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x4E:
            v209 = sub_928EC0;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x4F:
            v209 = sub_9297A0;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x50:
            v209 = sub_929960;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x51:
            v209 = sub_928C20;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x52:
            v209 = sub_928AD0;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x53:
            v209 = sub_928D70;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x54:
            v209 = sub_92AA70;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x55:
            v209 = sub_92A590;
            return (__m128i *)sub_92F600(a1, a2, v209, 0);
          case 0x56:
            v193 = *(_QWORD *)(a2 + 72);
            v194 = *(__int64 **)(v193 + 16);
            for ( i = *v194; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            v196 = *(_QWORD *)(i + 128);
            v197 = *a1;
            v198 = 0;
            v371 = (__m128i *)*v194;
            v370 = v196;
            sub_926800((__int64)&v384, v197, (__int64)v194);
            v202 = (__int64)v371;
            if ( v385.m128i_i32[2] )
            {
              _BitScanReverse64(&v203, v385.m128i_u32[2]);
              LODWORD(v203) = v203 ^ 0x3F;
              v199 = (unsigned int)(63 - v203);
              LOBYTE(v198) = 63 - v203;
              BYTE1(v198) = 1;
            }
            v204 = v384.m128i_i32[2];
            v371 = &v388;
            if ( dword_4D04810 )
            {
              v369 = v202;
              if ( (unsigned int)sub_731770(v193, 0, v199, v200, v202, v201) )
              {
                v339 = *a1;
                v388.m128i_i64[0] = (__int64)"bassign.tmp";
                v340 = v369;
                v390.m128i_i16[0] = 259;
                v341 = *(_QWORD *)(v339 + 32);
                v369 = v339;
                v342 = sub_91A390(v341 + 8, v340, 0, v338);
                v343 = sub_921B80(v369, v342, (__int64)v371, v198, 0);
                v367 = v369;
                v366 = 0;
                v344 = *a1;
                v369 = v343;
                sub_92CB70(v344, v343, v204, v370, v198, v198, 0);
                v204 = v369;
              }
            }
            sub_926800((__int64)v371, *a1, v193);
            v205 = 0;
            if ( v389.m128i_i32[2] )
            {
              _BitScanReverse64(&v206, v389.m128i_u32[2]);
              LOBYTE(v205) = 63 - (v206 ^ 0x3F);
              BYTE1(v205) = 1;
            }
            sub_92CB70(*a1, v388.m128i_i32[2], v204, v370, v205, v198, 0);
            v207 = *(_QWORD *)a2;
            for ( j = *(_BYTE *)(*(_QWORD *)a2 + 140LL); j == 12; j = *(_BYTE *)(v207 + 140) )
              v207 = *(_QWORD *)(v207 + 160);
            if ( j != 1 )
              sub_91B8A0("expected result type of bassign to be void!", (_DWORD *)(a2 + 36), 1);
            return 0;
          case 0x57:
            v152 = *(_QWORD *)(a2 + 72);
            v153 = *a1;
            v371 = *(__m128i **)(v152 + 16);
            v154 = sub_945CA0(v153, "land.end", 0, 0);
            v155 = sub_945CA0(*a1, "land.rhs", 0, 0);
            v156 = *a1;
            v157 = v155;
            v370 = v155;
            v158 = sub_921E00(v156, v152);
            sub_945D00(v156, v158, v157, v154, 0);
            v368 = &v384;
            sub_B43C20(&v384, v154);
            v159 = a1[2];
            v390.m128i_i16[0] = 257;
            v160 = sub_BCB2A0(v159);
            v161 = v384.m128i_u16[4];
            v162 = v160;
            v369 = v384.m128i_i64[0];
            v163 = sub_BD2DA0(80);
            v122 = v163;
            if ( v163 )
            {
              sub_B44260(v163, v162, 55, 0x8000000, v369, v161);
              *(_DWORD *)(v122 + 72) = 2;
              sub_BD6B50(v122, &v388);
              sub_BD2A10(v122, *(unsigned int *)(v122 + 72), 1);
            }
            v164 = *(_QWORD *)(v154 + 16);
            if ( !v164 )
              goto LABEL_133;
            while ( 1 )
            {
              v165 = *(_QWORD *)(v164 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v165 - 30) <= 0xAu )
                break;
              v164 = *(_QWORD *)(v164 + 8);
              if ( !v164 )
                goto LABEL_133;
            }
LABEL_122:
            v166 = *(_QWORD *)(v165 + 40);
            v167 = sub_ACD720(a1[2]);
            v168 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
            if ( v168 == *(_DWORD *)(v122 + 72) )
            {
              v369 = v167;
              sub_B48D90(v122);
              v167 = v369;
              v168 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
            }
            v169 = (v168 + 1) & 0x7FFFFFF;
            v170 = v169 | *(_DWORD *)(v122 + 4) & 0xF8000000;
            v171 = *(_QWORD *)(v122 - 8) + 32LL * (unsigned int)(v169 - 1);
            *(_DWORD *)(v122 + 4) = v170;
            if ( *(_QWORD *)v171 )
            {
              v172 = *(_QWORD *)(v171 + 8);
              **(_QWORD **)(v171 + 16) = v172;
              if ( v172 )
                *(_QWORD *)(v172 + 16) = *(_QWORD *)(v171 + 16);
            }
            *(_QWORD *)v171 = v167;
            if ( v167 )
            {
              v173 = *(_QWORD *)(v167 + 16);
              *(_QWORD *)(v171 + 8) = v173;
              if ( v173 )
                *(_QWORD *)(v173 + 16) = v171 + 8;
              *(_QWORD *)(v171 + 16) = v167 + 16;
              *(_QWORD *)(v167 + 16) = v171;
            }
            *(_QWORD *)(*(_QWORD *)(v122 - 8)
                      + 32LL * *(unsigned int *)(v122 + 72)
                      + 8LL * ((*(_DWORD *)(v122 + 4) & 0x7FFFFFFu) - 1)) = v166;
            while ( 1 )
            {
              v164 = *(_QWORD *)(v164 + 8);
              if ( !v164 )
                break;
              v165 = *(_QWORD *)(v164 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v165 - 30) <= 0xAu )
                goto LABEL_122;
            }
LABEL_133:
            sub_92FEA0(*a1, v370, 0);
            v174 = sub_921E00(*a1, (__int64)v371);
            v175 = *(_QWORD *)(a1[1] + 48);
            sub_92FEA0(*a1, v154, 0);
            v176 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
            if ( v176 == *(_DWORD *)(v122 + 72) )
            {
              sub_B48D90(v122);
              v176 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
            }
            v177 = (v176 + 1) & 0x7FFFFFF;
            v178 = v177 | *(_DWORD *)(v122 + 4) & 0xF8000000;
            v179 = *(_QWORD *)(v122 - 8) + 32LL * (unsigned int)(v177 - 1);
            *(_DWORD *)(v122 + 4) = v178;
            if ( *(_QWORD *)v179 )
            {
              v180 = *(_QWORD *)(v179 + 8);
              **(_QWORD **)(v179 + 16) = v180;
              if ( v180 )
                *(_QWORD *)(v180 + 16) = *(_QWORD *)(v179 + 16);
            }
            *(_QWORD *)v179 = v174;
            if ( v174 )
            {
              v181 = *(_QWORD *)(v174 + 16);
              *(_QWORD *)(v179 + 8) = v181;
              if ( v181 )
                *(_QWORD *)(v181 + 16) = v179 + 8;
              *(_QWORD *)(v179 + 16) = v174 + 16;
              *(_QWORD *)(v174 + 16) = v179;
            }
            *(_QWORD *)(*(_QWORD *)(v122 - 8)
                      + 32LL * *(unsigned int *)(v122 + 72)
                      + 8LL * ((*(_DWORD *)(v122 + 4) & 0x7FFFFFFu) - 1)) = v175;
            sub_B33980(a1[1], v122);
            v182 = a1[2];
            v183 = (unsigned int **)a1[1];
            v384.m128i_i64[0] = (__int64)"land.ext";
            v386.m128i_i16[0] = 259;
            v184 = sub_BCB2D0(v182);
            if ( v184 == *(_QWORD *)(v122 + 8) )
              return (__m128i *)v122;
            v185 = v183[10];
            v186 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v185 + 120LL);
            if ( v186 == sub_920130 )
            {
              if ( *(_BYTE *)v122 > 0x15u )
                goto LABEL_148;
              if ( (unsigned __int8)sub_AC4810(39) )
                v13 = sub_ADAB70(39, v122, v184, 0);
              else
                v13 = sub_AA93C0(39, v122, v184);
            }
            else
            {
              v13 = v186((__int64)v185, 39u, (_BYTE *)v122, v184);
            }
            if ( v13 )
              return (__m128i *)v13;
LABEL_148:
            v390.m128i_i16[0] = 257;
            v187 = sub_BD2C40(72, unk_3F10A14);
            v13 = v187;
            if ( v187 )
              sub_B515B0(v187, v122, v184, &v388, 0, 0);
            (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v183[11] + 16LL))(
              v183[11],
              v13,
              v368,
              v183[7],
              v183[8]);
            v188 = 4LL * *((unsigned int *)v183 + 2);
            v189 = *v183;
            v190 = &v189[v188];
            while ( v190 != v189 )
            {
              v191 = *((_QWORD *)v189 + 1);
              v192 = *v189;
              v189 += 4;
              sub_B99FD0(v13, v192, v191);
            }
            return (__m128i *)v13;
          case 0x58:
            v110 = *(_QWORD *)(a2 + 72);
            v111 = *a1;
            v371 = *(__m128i **)(v110 + 16);
            v112 = sub_945CA0(v111, "lor.end", 0, 0);
            v113 = sub_945CA0(*a1, "lor.rhs", 0, 0);
            v114 = *a1;
            v115 = v113;
            v370 = v113;
            v116 = sub_921E00(v114, v110);
            sub_945D00(v114, v116, v112, v115, 0);
            v368 = &v384;
            sub_B43C20(&v384, v112);
            v117 = a1[2];
            v390.m128i_i16[0] = 257;
            v118 = sub_BCB2A0(v117);
            v119 = v384.m128i_u16[4];
            v120 = v118;
            v369 = v384.m128i_i64[0];
            v121 = sub_BD2DA0(80);
            v122 = v121;
            if ( v121 )
            {
              sub_B44260(v121, v120, 55, 0x8000000, v369, v119);
              *(_DWORD *)(v122 + 72) = 2;
              sub_BD6B50(v122, &v388);
              sub_BD2A10(v122, *(unsigned int *)(v122 + 72), 1);
            }
            v123 = *(_QWORD *)(v112 + 16);
            if ( !v123 )
              goto LABEL_96;
            break;
          case 0x5B:
            v108 = *(__int64 **)(a2 + 72);
            a2 = v108[2];
            sub_921EA0((__int64)&v388, *a1, v108, 0, 0, 0);
            v5 = *a1;
            if ( !*(_QWORD *)(*a1 + 96) )
            {
              v371 = (__m128i *)*a1;
              v109 = sub_945CA0(v5, byte_3F871B3, 0, 0);
              sub_92FEA0(v371, v109, 0);
              v5 = *a1;
            }
            continue;
          case 0x67:
            v71 = *a1;
            v72 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 16LL);
            v371 = *(__m128i **)(a2 + 72);
            v370 = *(_QWORD *)(v72 + 16);
            v73 = sub_945CA0(v71, "cond.true", 0, 0);
            v74 = sub_945CA0(*a1, "cond.false", 0, 0);
            v75 = sub_945CA0(*a1, "cond.end", 0, 0);
            v76 = (__int64)v371;
            v371 = (__m128i *)*a1;
            v77 = sub_921E00((__int64)v371, v76);
            sub_945D00(v371, v77, v73, v74, 0);
            sub_92FEA0(*a1, v73, 0);
            v78 = sub_92CBF0(a1, v72);
            v79 = *a1;
            v80 = v78;
            v369 = *(_QWORD *)(a1[1] + 48);
            sub_92FD90(v79, v75);
            sub_92FEA0(*a1, v74, 0);
            v81 = sub_92CBF0(a1, v370);
            v82 = *a1;
            v371 = (__m128i *)v81;
            v83 = v81;
            v370 = *(_QWORD *)(a1[1] + 48);
            sub_92FD90(v82, v75);
            sub_92FEA0(*a1, v75, 0);
            if ( !v80 )
              return v371;
            v13 = v80;
            if ( v83 )
            {
              v384.m128i_i64[0] = (__int64)"cond";
              v84 = (unsigned int **)a1[1];
              v386.m128i_i16[0] = 259;
              v85 = *(__m128i **)(v80 + 8);
              v390.m128i_i16[0] = 257;
              v368 = v85;
              v86 = sub_BD2DA0(80);
              v13 = v86;
              if ( v86 )
              {
                v87 = v86;
                sub_B44260(v86, v85, 55, 0x8000000, 0, 0);
                *(_DWORD *)(v13 + 72) = 2;
                sub_BD6B50(v13, &v388);
                sub_BD2A10(v13, *(unsigned int *)(v13 + 72), 1);
              }
              else
              {
                v87 = 0;
              }
              if ( (unsigned __int8)sub_920620(v87) )
              {
                v352 = v84[12];
                v353 = *((_DWORD *)v84 + 26);
                if ( v352 )
                  sub_B99FD0(v13, 3, v352);
                sub_B45150(v13, v353);
              }
              (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v84[11] + 16LL))(
                v84[11],
                v13,
                &v384,
                v84[7],
                v84[8]);
              v88 = *v84;
              v89 = (__int64)&(*v84)[4 * *((unsigned int *)v84 + 2)];
              while ( (unsigned int *)v89 != v88 )
              {
                v90 = *((_QWORD *)v88 + 1);
                v91 = *v88;
                v88 += 4;
                sub_B99FD0(v13, v91, v90);
              }
              v92 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
              if ( v92 == *(_DWORD *)(v13 + 72) )
              {
                sub_B48D90(v13);
                v92 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
              }
              v93 = (v92 + 1) & 0x7FFFFFF;
              v94 = v93 | *(_DWORD *)(v13 + 4) & 0xF8000000;
              v95 = *(_QWORD *)(v13 - 8) + 32LL * (unsigned int)(v93 - 1);
              *(_DWORD *)(v13 + 4) = v94;
              if ( *(_QWORD *)v95 )
              {
                v96 = *(_QWORD *)(v95 + 8);
                **(_QWORD **)(v95 + 16) = v96;
                if ( v96 )
                  *(_QWORD *)(v96 + 16) = *(_QWORD *)(v95 + 16);
              }
              *(_QWORD *)v95 = v80;
              v97 = *(_QWORD *)(v80 + 16);
              *(_QWORD *)(v95 + 8) = v97;
              if ( v97 )
                *(_QWORD *)(v97 + 16) = v95 + 8;
              *(_QWORD *)(v95 + 16) = v80 + 16;
              v98 = v369;
              *(_QWORD *)(v80 + 16) = v95;
              *(_QWORD *)(*(_QWORD *)(v13 - 8)
                        + 32LL * *(unsigned int *)(v13 + 72)
                        + 8LL * ((*(_DWORD *)(v13 + 4) & 0x7FFFFFFu) - 1)) = v98;
              v99 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
              if ( v99 == *(_DWORD *)(v13 + 72) )
              {
                sub_B48D90(v13);
                v99 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
              }
              v100 = (v99 + 1) & 0x7FFFFFF;
              v101 = v100 | *(_DWORD *)(v13 + 4) & 0xF8000000;
              v102 = *(_QWORD *)(v13 - 8) + 32LL * (unsigned int)(v100 - 1);
              *(_DWORD *)(v13 + 4) = v101;
              if ( *(_QWORD *)v102 )
              {
                v103 = *(_QWORD *)(v102 + 8);
                **(_QWORD **)(v102 + 16) = v103;
                if ( v103 )
                  *(_QWORD *)(v103 + 16) = *(_QWORD *)(v102 + 16);
              }
              v104 = (__int64)v371;
              *(_QWORD *)v102 = v371;
              v105 = *(_QWORD *)(v104 + 16);
              v106 = v104 + 16;
              *(_QWORD *)(v102 + 8) = v105;
              if ( v105 )
                *(_QWORD *)(v105 + 16) = v102 + 8;
              v107 = (__int64)v371;
              *(_QWORD *)(v102 + 16) = v106;
              *(_QWORD *)(v107 + 16) = v102;
              *(_QWORD *)(*(_QWORD *)(v13 - 8)
                        + 32LL * *(unsigned int *)(v13 + 72)
                        + 8LL * ((*(_DWORD *)(v13 + 4) & 0x7FFFFFFu) - 1)) = v370;
              sub_B33980(a1[1], v13);
            }
            return (__m128i *)v13;
          case 0x69:
            sub_926600((__int64)&v388);
            return (__m128i *)v388.m128i_i64[0];
          case 0x6F:
            v54 = *a1;
            v70 = sub_945C50(*a1, *(_QWORD *)(a2 + 72));
            v56 = 1;
            v57 = v70;
            return (__m128i *)sub_927810(v54, v57, v56);
          case 0x70:
            v66 = *a1;
            v67 = *(_QWORD *)(a2 + 72);
            v68 = sub_91A390(*(_QWORD *)(*a1 + 32) + 8LL, *(_QWORD *)a2, 0, (__int64)v6);
            v69 = sub_945C50(*a1, v67);
            return (__m128i *)sub_927EF0(v66, v69, v68);
          case 0x71:
            v54 = *a1;
            v55 = sub_945C50(*a1, *(_QWORD *)(a2 + 72));
            v56 = 0;
            v57 = v55;
            return (__m128i *)sub_927810(v54, v57, v56);
          case 0x72:
            v50 = *(_QWORD *)(a2 + 72);
            v51 = *a1;
            v52 = sub_945C50(*a1, *(_QWORD *)(v50 + 16));
            v53 = sub_945C50(*a1, v50);
            return (__m128i *)sub_927A80(v51, v53, v52);
          default:
            sub_91B8A0("unsupported operation expression!", (_DWORD *)(a2 + 36), 1);
        }
        return result;
      case 2:
        return (__m128i *)sub_91FFE0(*a1, *(const __m128i **)(a2 + 56), 0, (__int64)v6);
      case 3:
        sub_926800((__int64)&v380, *a1, a2);
        v8 = _mm_loadu_si128(&v381);
        v9 = _mm_loadu_si128(&v382);
        v388 = _mm_loadu_si128(&v380);
        v389 = v8;
        v390 = v9;
        v391 = v383;
        v363 = v388;
        sub_9286A0(
          (__int64)&v384,
          *a1,
          (_DWORD *)(a2 + 36),
          v10,
          v11,
          v12,
          v363.m128i_i64[0],
          v363.m128i_u64[1],
          v8.m128i_u64[0],
          v8.m128i_i64[1],
          v9.m128i_i64[0],
          v9.m128i_i64[1],
          v383);
        return (__m128i *)v384.m128i_i64[0];
      case 0x11:
        v19 = *(_QWORD *)a2;
        for ( k = *(_BYTE *)(*(_QWORD *)a2 + 140LL); k == 12; k = *(_BYTE *)(v19 + 140) )
          v19 = *(_QWORD *)(v19 + 160);
        sub_9365F0((unsigned int)&v388, *a1, *(_QWORD *)(a2 + 56), k != 1, 0, 0, 0);
        return (__m128i *)v388.m128i_i64[0];
      case 0x13:
        sub_927750((__int64)&v388, *a1, (__int64 *)a2);
        return (__m128i *)v388.m128i_i64[0];
      case 0x14:
        sub_926800((__int64)&v384, *a1, a2);
        v14 = _mm_loadu_si128(&v385);
        v15 = _mm_loadu_si128(&v386);
        v388 = _mm_loadu_si128(&v384);
        v389 = v14;
        v390 = v15;
        v391 = v387;
        v363 = v388;
        sub_9286A0(
          (__int64)&v380,
          *a1,
          (_DWORD *)(a2 + 36),
          v16,
          v17,
          v18,
          v363.m128i_i64[0],
          v363.m128i_u64[1],
          v14.m128i_u64[0],
          v14.m128i_i64[1],
          v15.m128i_i64[0],
          v15.m128i_i64[1],
          v387);
        return (__m128i *)v380.m128i_i64[0];
      default:
        sub_91B8A0("unsupported expression!", (_DWORD *)(a2 + 36), 1);
    }
  }
  while ( 1 )
  {
    v124 = *(_QWORD *)(v123 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v124 - 30) <= 0xAu )
      break;
    v123 = *(_QWORD *)(v123 + 8);
    if ( !v123 )
      goto LABEL_96;
  }
LABEL_85:
  v125 = *(_QWORD *)(v124 + 40);
  v126 = sub_ACD6D0(a1[2]);
  v127 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
  if ( v127 == *(_DWORD *)(v122 + 72) )
  {
    v369 = v126;
    sub_B48D90(v122);
    v126 = v369;
    v127 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
  }
  v128 = (v127 + 1) & 0x7FFFFFF;
  v129 = v128 | *(_DWORD *)(v122 + 4) & 0xF8000000;
  v130 = *(_QWORD *)(v122 - 8) + 32LL * (unsigned int)(v128 - 1);
  *(_DWORD *)(v122 + 4) = v129;
  if ( *(_QWORD *)v130 )
  {
    v131 = *(_QWORD *)(v130 + 8);
    **(_QWORD **)(v130 + 16) = v131;
    if ( v131 )
      *(_QWORD *)(v131 + 16) = *(_QWORD *)(v130 + 16);
  }
  *(_QWORD *)v130 = v126;
  if ( v126 )
  {
    v132 = *(_QWORD *)(v126 + 16);
    *(_QWORD *)(v130 + 8) = v132;
    if ( v132 )
      *(_QWORD *)(v132 + 16) = v130 + 8;
    *(_QWORD *)(v130 + 16) = v126 + 16;
    *(_QWORD *)(v126 + 16) = v130;
  }
  *(_QWORD *)(*(_QWORD *)(v122 - 8)
            + 32LL * *(unsigned int *)(v122 + 72)
            + 8LL * ((*(_DWORD *)(v122 + 4) & 0x7FFFFFFu) - 1)) = v125;
  while ( 1 )
  {
    v123 = *(_QWORD *)(v123 + 8);
    if ( !v123 )
      break;
    v124 = *(_QWORD *)(v123 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v124 - 30) <= 0xAu )
      goto LABEL_85;
  }
LABEL_96:
  sub_92FEA0(*a1, v370, 0);
  v133 = sub_921E00(*a1, (__int64)v371);
  v134 = *(_QWORD *)(a1[1] + 48);
  sub_92FEA0(*a1, v112, 0);
  v135 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
  if ( v135 == *(_DWORD *)(v122 + 72) )
  {
    sub_B48D90(v122);
    v135 = *(_DWORD *)(v122 + 4) & 0x7FFFFFF;
  }
  v136 = (v135 + 1) & 0x7FFFFFF;
  v137 = v136 | *(_DWORD *)(v122 + 4) & 0xF8000000;
  v138 = *(_QWORD *)(v122 - 8) + 32LL * (unsigned int)(v136 - 1);
  *(_DWORD *)(v122 + 4) = v137;
  if ( *(_QWORD *)v138 )
  {
    v139 = *(_QWORD *)(v138 + 8);
    **(_QWORD **)(v138 + 16) = v139;
    if ( v139 )
      *(_QWORD *)(v139 + 16) = *(_QWORD *)(v138 + 16);
  }
  *(_QWORD *)v138 = v133;
  if ( v133 )
  {
    v140 = *(_QWORD *)(v133 + 16);
    *(_QWORD *)(v138 + 8) = v140;
    if ( v140 )
      *(_QWORD *)(v140 + 16) = v138 + 8;
    *(_QWORD *)(v138 + 16) = v133 + 16;
    *(_QWORD *)(v133 + 16) = v138;
  }
  *(_QWORD *)(*(_QWORD *)(v122 - 8)
            + 32LL * *(unsigned int *)(v122 + 72)
            + 8LL * ((*(_DWORD *)(v122 + 4) & 0x7FFFFFFu) - 1)) = v134;
  sub_B33980(a1[1], v122);
  v141 = a1[2];
  v142 = (unsigned int **)a1[1];
  v384.m128i_i64[0] = (__int64)"lor.ext";
  v386.m128i_i16[0] = 259;
  v143 = sub_BCB2D0(v141);
  if ( v143 == *(_QWORD *)(v122 + 8) )
    return (__m128i *)v122;
  v144 = v142[10];
  v145 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v144 + 120LL);
  if ( v145 == sub_920130 )
  {
    if ( *(_BYTE *)v122 > 0x15u )
    {
LABEL_111:
      v390.m128i_i16[0] = 257;
      v146 = sub_BD2C40(72, unk_3F10A14);
      v13 = v146;
      if ( v146 )
        sub_B515B0(v146, v122, v143, &v388, 0, 0);
      (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v142[11]
                                                                                                 + 16LL))(
        v142[11],
        v13,
        v368,
        v142[7],
        v142[8]);
      v147 = 4LL * *((unsigned int *)v142 + 2);
      v148 = *v142;
      v149 = &v148[v147];
      while ( v149 != v148 )
      {
        v150 = *((_QWORD *)v148 + 1);
        v151 = *v148;
        v148 += 4;
        sub_B99FD0(v13, v151, v150);
      }
      return (__m128i *)v13;
    }
    if ( (unsigned __int8)sub_AC4810(39) )
      v13 = sub_ADAB70(39, v122, v143, 0);
    else
      v13 = sub_AA93C0(39, v122, v143);
  }
  else
  {
    v13 = v145((__int64)v144, 39u, (_BYTE *)v122, v143);
  }
  if ( !v13 )
    goto LABEL_111;
  return (__m128i *)v13;
}
