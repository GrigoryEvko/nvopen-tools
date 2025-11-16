// Function: sub_1B649E0
// Address: 0x1b649e0
//
_BOOL8 __fastcall sub_1B649E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15)
{
  int v16; // eax
  __int64 v18; // rcx
  int v19; // edi
  __int64 v20; // rsi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r8
  _QWORD *v25; // rbx
  int v26; // eax
  unsigned __int64 v27; // rdx
  __int64 *v28; // r14
  __int64 v29; // r12
  __int64 v30; // r13
  char v31; // r15
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  unsigned int v37; // eax
  _QWORD *v38; // rdi
  unsigned __int64 v39; // rcx
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 v42; // rcx
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // r15
  __int64 *v49; // rax
  __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  __int64 v52; // rdx
  int v53; // eax
  unsigned __int8 v54; // r15
  __int64 *v55; // rax
  __int64 *v56; // rax
  __int64 v57; // r13
  unsigned int v58; // r15d
  __int64 *v59; // rsi
  __int64 v60; // r13
  __int64 v61; // r15
  __int64 v62; // r15
  _QWORD *v63; // r13
  unsigned __int8 v64; // al
  int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // rcx
  unsigned int v68; // eax
  __int64 v69; // rax
  unsigned int v70; // r13d
  __int64 v71; // r10
  __int64 v72; // rdi
  _QWORD *v73; // rdx
  int v74; // ecx
  __int64 v75; // rsi
  __int64 v76; // r9
  int v77; // ecx
  unsigned int v78; // edx
  __int64 *v79; // rax
  __int64 v80; // r8
  __int64 v81; // r15
  __int64 *v82; // rax
  __int64 v83; // rax
  double v84; // xmm4_8
  double v85; // xmm5_8
  __int64 v86; // rsi
  unsigned int v87; // eax
  _QWORD *v88; // rdi
  unsigned int v89; // eax
  _QWORD *v90; // rdi
  unsigned int v91; // eax
  _QWORD *v92; // rdi
  unsigned int v93; // eax
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // r8
  __int64 v97; // rdi
  unsigned int v98; // esi
  __int64 *v99; // rcx
  __int64 v100; // r9
  __int64 *v101; // rax
  int v102; // r8d
  int v103; // r9d
  __int64 v104; // rax
  __int64 *v105; // rax
  __int64 v106; // rsi
  double v107; // xmm4_8
  double v108; // xmm5_8
  unsigned int v109; // eax
  _QWORD *v110; // rdi
  __int64 v111; // rbx
  __int64 v112; // r14
  int v113; // r12d
  int v114; // eax
  __int64 v115; // r15
  _QWORD *v116; // rax
  __int64 v117; // r15
  _QWORD *v118; // r14
  _QWORD **v119; // rax
  unsigned int v120; // eax
  __int64 *v121; // rax
  __int64 v122; // rax
  unsigned int v123; // r11d
  __int64 v124; // r13
  __int64 v125; // r10
  _QWORD *v126; // rdx
  int v127; // r9d
  __int64 v128; // rcx
  __int64 v129; // rdi
  int v130; // r9d
  unsigned int v131; // edx
  __int64 *v132; // rax
  __int64 v133; // rsi
  __int64 *v134; // r13
  __int64 v135; // rax
  bool v136; // zf
  __int64 v137; // rax
  unsigned int v138; // eax
  bool v139; // r10
  int v140; // r11d
  void *v141; // r9
  __int64 v142; // r13
  __int64 v143; // r15
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // r15
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // r13
  __int64 v150; // rbx
  __int64 v151; // r15
  _QWORD *v152; // rax
  double v153; // xmm4_8
  double v154; // xmm5_8
  __int64 v155; // r13
  int v156; // r8d
  int v157; // r9d
  __int64 v158; // rax
  int v159; // eax
  __int64 v160; // r13
  __int64 v161; // rax
  _QWORD *v162; // rdx
  __int64 v163; // rsi
  int v164; // r11d
  __int64 v165; // rcx
  __int64 v166; // r9
  int v167; // r11d
  unsigned int v168; // edx
  __int64 *v169; // rax
  __int64 v170; // rdi
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // r9
  __int64 *v174; // r13
  __int64 v175; // r15
  __int64 v176; // rdx
  __int64 v177; // rsi
  double v178; // xmm4_8
  double v179; // xmm5_8
  _QWORD *v180; // r15
  __int64 v181; // r13
  int v182; // r9d
  __int64 v183; // rcx
  __int64 v184; // rdi
  int v185; // r9d
  unsigned int v186; // edx
  __int64 *v187; // rax
  __int64 v188; // rsi
  __int64 v189; // r13
  __int64 v190; // rdx
  __int64 v191; // r13
  bool v192; // al
  double v193; // xmm4_8
  double v194; // xmm5_8
  __int64 v195; // rax
  unsigned __int64 *v196; // rdi
  unsigned __int64 v197; // rax
  __int64 **v198; // rax
  __int64 v199; // rdi
  unsigned int v200; // esi
  __int64 v201; // rdx
  int v202; // eax
  __int64 v203; // rcx
  __int64 v204; // r13
  __int64 v205; // r13
  __int64 v206; // rax
  __int64 v207; // rsi
  unsigned __int64 v208; // rdi
  __int64 v209; // rsi
  __int64 v210; // rcx
  unsigned __int64 v211; // rdi
  __int64 v212; // rcx
  unsigned int v213; // eax
  _QWORD *v214; // rax
  __int64 *v215; // rax
  __int64 v216; // rax
  double v217; // xmm4_8
  double v218; // xmm5_8
  unsigned int v219; // r13d
  _QWORD *v220; // rax
  __int64 v221; // r13
  __int64 v222; // r15
  __int64 v223; // rax
  __int64 v224; // r15
  __int64 v225; // rax
  __int64 v226; // rax
  __int64 v227; // rax
  __int64 v228; // rax
  __int64 v229; // r15
  __int64 v230; // rax
  __int64 v231; // r13
  __int64 v232; // rdi
  __int64 v233; // rsi
  int v234; // ecx
  __int64 v235; // r8
  int v236; // ecx
  unsigned int v237; // edx
  __int64 *v238; // rax
  __int64 v239; // r11
  _QWORD *v240; // rdi
  unsigned int v241; // edx
  __int64 *v242; // rax
  __int64 v243; // r11
  _QWORD *v244; // rax
  _QWORD **v245; // r13
  _QWORD **v246; // r15
  _QWORD *v247; // rdi
  int v248; // eax
  int v249; // ecx
  int v250; // r11d
  __int64 v251; // rax
  unsigned int v252; // r13d
  __int64 v253; // rax
  __int64 v254; // rax
  char v255; // al
  __int64 v256; // rax
  double v257; // xmm4_8
  double v258; // xmm5_8
  unsigned int v259; // eax
  _QWORD *v260; // rdi
  __int64 v261; // rsi
  __int64 *v262; // r13
  __int64 v263; // r15
  _QWORD *v264; // r14
  char v265; // al
  __int64 v266; // rax
  __int64 v267; // rax
  unsigned __int64 *v268; // rdi
  unsigned __int8 v269; // al
  unsigned __int64 *v270; // r13
  __int64 v271; // rax
  __int64 v272; // rax
  unsigned __int8 v273; // al
  __int64 v274; // rax
  __int64 v275; // rax
  __int64 v276; // rax
  __int64 v277; // rax
  char v278; // al
  unsigned __int64 v279; // rdi
  int v280; // r9d
  int v281; // eax
  __int64 v282; // rax
  __int64 v283; // rax
  int v284; // r11d
  int v285; // eax
  int v286; // eax
  int v287; // r8d
  __int64 v288; // rax
  __int64 v289; // r15
  _QWORD *v290; // r13
  _QWORD **v291; // rax
  __int64 *v292; // rax
  __int64 v293; // rsi
  _QWORD *v294; // rax
  double v295; // xmm4_8
  double v296; // xmm5_8
  __int64 v297; // r15
  __int64 v298; // rcx
  unsigned __int64 v299; // rsi
  __int64 v300; // rcx
  __int64 v301; // rcx
  unsigned __int64 v302; // rsi
  __int64 v303; // rcx
  __int64 v304; // rdx
  unsigned __int64 v305; // rcx
  __int64 v306; // rdx
  _QWORD *v307; // rdi
  int v308; // r8d
  __int64 v309; // rax
  int v310; // r8d
  int v311; // eax
  int v312; // r9d
  int v313; // eax
  int v314; // r9d
  __int64 *v315; // [rsp+8h] [rbp-2E8h]
  __int64 v316; // [rsp+10h] [rbp-2E0h]
  _QWORD *v317; // [rsp+18h] [rbp-2D8h]
  __int64 v318; // [rsp+18h] [rbp-2D8h]
  __int64 v319; // [rsp+20h] [rbp-2D0h]
  unsigned __int64 v320; // [rsp+28h] [rbp-2C8h]
  bool v321; // [rsp+30h] [rbp-2C0h]
  int v322; // [rsp+38h] [rbp-2B8h]
  __int64 v323; // [rsp+40h] [rbp-2B0h]
  unsigned int v324; // [rsp+40h] [rbp-2B0h]
  __int64 (__fastcall *v325)(_QWORD *, __int64, __int64, _QWORD, _QWORD); // [rsp+40h] [rbp-2B0h]
  __int64 v326; // [rsp+40h] [rbp-2B0h]
  __int64 v327; // [rsp+58h] [rbp-298h]
  __int64 *v328; // [rsp+58h] [rbp-298h]
  __int64 (__fastcall *v329)(_QWORD *, __int64, __int64, _QWORD, _QWORD); // [rsp+58h] [rbp-298h]
  __int64 v330; // [rsp+58h] [rbp-298h]
  __int64 v331; // [rsp+58h] [rbp-298h]
  __int64 v332; // [rsp+58h] [rbp-298h]
  bool v333; // [rsp+60h] [rbp-290h]
  __int64 v334; // [rsp+60h] [rbp-290h]
  _QWORD *v335; // [rsp+60h] [rbp-290h]
  __int64 v336; // [rsp+60h] [rbp-290h]
  __int64 v337; // [rsp+60h] [rbp-290h]
  __int64 v338; // [rsp+60h] [rbp-290h]
  __int64 *v339; // [rsp+60h] [rbp-290h]
  __int64 v341; // [rsp+78h] [rbp-278h]
  unsigned int v342; // [rsp+78h] [rbp-278h]
  _QWORD *v343; // [rsp+78h] [rbp-278h]
  __int64 v344; // [rsp+78h] [rbp-278h]
  unsigned int v345; // [rsp+78h] [rbp-278h]
  __int64 (__fastcall *v346)(__int64, __int64, __int64, unsigned int); // [rsp+78h] [rbp-278h]
  __int64 v347; // [rsp+78h] [rbp-278h]
  void *v348; // [rsp+78h] [rbp-278h]
  _QWORD *v349; // [rsp+78h] [rbp-278h]
  _QWORD *v350; // [rsp+78h] [rbp-278h]
  bool v353; // [rsp+97h] [rbp-259h]
  char v354; // [rsp+98h] [rbp-258h]
  bool v355; // [rsp+98h] [rbp-258h]
  unsigned int v356; // [rsp+98h] [rbp-258h]
  unsigned __int64 v357; // [rsp+98h] [rbp-258h]
  _QWORD *v358; // [rsp+98h] [rbp-258h]
  _QWORD *v359; // [rsp+98h] [rbp-258h]
  __int64 v360; // [rsp+98h] [rbp-258h]
  __int64 v361; // [rsp+98h] [rbp-258h]
  __int64 v362; // [rsp+98h] [rbp-258h]
  __int64 v363; // [rsp+98h] [rbp-258h]
  __int64 v364; // [rsp+98h] [rbp-258h]
  __int64 *v365; // [rsp+98h] [rbp-258h]
  __int64 v366; // [rsp+A0h] [rbp-250h]
  char v369; // [rsp+B8h] [rbp-238h]
  _QWORD *v370; // [rsp+B8h] [rbp-238h]
  unsigned int v371; // [rsp+CCh] [rbp-224h] BYREF
  __int64 v372; // [rsp+D0h] [rbp-220h] BYREF
  __int64 v373; // [rsp+D8h] [rbp-218h] BYREF
  __int64 v374; // [rsp+E0h] [rbp-210h] BYREF
  __int64 v375; // [rsp+E8h] [rbp-208h] BYREF
  char *v376; // [rsp+F0h] [rbp-200h] BYREF
  __int64 v377; // [rsp+F8h] [rbp-1F8h]
  __int16 v378; // [rsp+100h] [rbp-1F0h]
  unsigned __int64 v379; // [rsp+110h] [rbp-1E0h] BYREF
  __int64 v380; // [rsp+118h] [rbp-1D8h]
  __int64 *v381; // [rsp+120h] [rbp-1D0h] BYREF
  __int64 v382; // [rsp+128h] [rbp-1C8h]
  int v383; // [rsp+130h] [rbp-1C0h]
  _BYTE v384[40]; // [rsp+138h] [rbp-1B8h] BYREF
  char v385; // [rsp+160h] [rbp-190h] BYREF
  _BYTE *v386; // [rsp+180h] [rbp-170h] BYREF
  __int64 v387; // [rsp+188h] [rbp-168h]
  _BYTE v388[128]; // [rsp+190h] [rbp-160h] BYREF
  __int64 v389; // [rsp+210h] [rbp-E0h] BYREF
  _BYTE *v390; // [rsp+218h] [rbp-D8h]
  _BYTE *v391; // [rsp+220h] [rbp-D0h]
  __int64 v392; // [rsp+228h] [rbp-C8h]
  int v393; // [rsp+230h] [rbp-C0h]
  _BYTE v394[184]; // [rsp+238h] [rbp-B8h] BYREF

  v16 = *(_DWORD *)(a4 + 24);
  v366 = 0;
  if ( v16 )
  {
    v18 = a1[5];
    v19 = v16 - 1;
    v20 = *(_QWORD *)(a4 + 8);
    v21 = (v16 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v22 = (__int64 *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( v18 == *v22 )
    {
LABEL_3:
      v366 = v22[1];
    }
    else
    {
      v159 = 1;
      while ( v23 != -8 )
      {
        v280 = v159 + 1;
        v21 = v19 & (v159 + v21);
        v22 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v18 == *v22 )
          goto LABEL_3;
        v159 = v280;
      }
      v366 = 0;
    }
  }
  v353 = sub_1456C80(a2, *a1);
  if ( v353 )
  {
    v25 = (_QWORD *)a2;
    v390 = v394;
    v391 = v394;
    v386 = v388;
    v387 = 0x800000000LL;
    v389 = 0;
    v392 = 16;
    v393 = 0;
    sub_1B64100(a1, v366, (__int64)&v389, (__int64)&v386);
    v26 = v387;
    v369 = 0;
    if ( !(_DWORD)v387 )
    {
      v353 = 0;
LABEL_22:
      if ( v386 != v388 )
        _libc_free((unsigned __int64)v386);
      if ( v391 != v390 )
        _libc_free((unsigned __int64)v391);
      return v353;
    }
LABEL_7:
    v27 = (unsigned __int64)&v386[16 * v26 - 16];
    v28 = *(__int64 **)v27;
    v29 = *(_QWORD *)(v27 + 8);
    LODWORD(v387) = v26 - 1;
    v354 = sub_1AE9990((__int64)v28, 0);
    if ( v354 )
    {
      v89 = *(_DWORD *)(a5 + 8);
      if ( v89 >= *(_DWORD *)(a5 + 12) )
      {
        sub_170B450(a5, 0);
        v89 = *(_DWORD *)(a5 + 8);
      }
      v90 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v89);
      if ( v90 )
      {
        *v90 = 6;
        v90[1] = 0;
        v90[2] = v28;
        if ( v28 + 1 != 0 && v28 != 0 && v28 != (__int64 *)-16LL )
          sub_164C220((__int64)v90);
        v89 = *(_DWORD *)(a5 + 8);
      }
      *(_DWORD *)(a5 + 8) = v89 + 1;
      goto LABEL_20;
    }
    if ( a1 == v28 )
      goto LABEL_20;
    if ( sub_1456C80((__int64)v25, *v28) )
    {
      v30 = sub_146F1B0((__int64)v25, (__int64)v28);
      if ( sub_146CEE0((__int64)v25, v30, v366) )
      {
        v379 = 0;
        v380 = (__int64)v384;
        v381 = (__int64 *)v384;
        v382 = 8;
        v383 = 0;
        v31 = sub_3872990(a6, v30, v366, v28, &v379, 1);
        if ( v381 != (__int64 *)v380 )
          _libc_free((unsigned __int64)v381);
        if ( !v31 )
        {
          v32 = sub_13FC520(v366);
          v33 = (unsigned __int64)v28;
          if ( v32 )
            v33 = sub_157EBA0(v32);
          v34 = sub_38767A0(a6, v30, *v28, v33);
          sub_164D160((__int64)v28, v34, a7, *(double *)a8.m128i_i64, a9, a10, v35, v36, a13, a14);
          v37 = *(_DWORD *)(a5 + 8);
          if ( v37 >= *(_DWORD *)(a5 + 12) )
          {
            sub_170B450(a5, 0);
            v37 = *(_DWORD *)(a5 + 8);
          }
          v38 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v37);
          if ( v38 )
          {
            *v38 = 6;
            v38[1] = 0;
            v38[2] = v28;
            if ( v28 != (__int64 *)-8LL && v28 != (__int64 *)-16LL )
              sub_164C220((__int64)v38);
            v37 = *(_DWORD *)(a5 + 8);
          }
          v369 = 1;
          *(_DWORD *)(a5 + 8) = v37 + 1;
          goto LABEL_20;
        }
      }
    }
    while ( 1 )
    {
      if ( !v29 )
        goto LABEL_20;
      v53 = *((unsigned __int8 *)v28 + 16);
      v54 = *((_BYTE *)v28 + 16);
      if ( v53 != 41 && v53 != 48 )
        goto LABEL_84;
      v55 = (*((_BYTE *)v28 + 23) & 0x40) != 0 ? (__int64 *)*(v28 - 1) : &v28[-3 * (*((_DWORD *)v28 + 5) & 0xFFFFFFF)];
      if ( v29 != *v55 )
        goto LABEL_84;
      v45 = v55[3];
      if ( *(_BYTE *)(v45 + 16) != 13 || (unsigned int)*(unsigned __int8 *)(v29 + 16) - 35 > 0x11 )
        goto LABEL_84;
      v56 = (*(_BYTE *)(v29 + 23) & 0x40) != 0
          ? *(__int64 **)(v29 - 8)
          : (__int64 *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v56[3] + 16) != 13 )
        goto LABEL_84;
      v57 = *v56;
      if ( v54 == 48 )
        break;
LABEL_34:
      v46 = sub_146F1B0((__int64)v25, v45);
      v47 = sub_146F1B0((__int64)v25, v57);
      v48 = sub_1483CF0(v25, v47, v46, (__m128i)a7, a8);
      if ( !sub_1456C80((__int64)v25, *v28) || v48 != sub_146F1B0((__int64)v25, (__int64)v28) )
      {
        v54 = *((_BYTE *)v28 + 16);
        goto LABEL_84;
      }
      if ( (*((_BYTE *)v28 + 23) & 0x40) != 0 )
        v49 = (__int64 *)*(v28 - 1);
      else
        v49 = &v28[-3 * (*((_DWORD *)v28 + 5) & 0xFFFFFFF)];
      if ( *v49 )
      {
        v50 = v49[1];
        v51 = v49[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v51 = v50;
        if ( v50 )
          *(_QWORD *)(v50 + 16) = *(_QWORD *)(v50 + 16) & 3LL | v51;
      }
      *v49 = v57;
      if ( v57 )
      {
        v52 = *(_QWORD *)(v57 + 8);
        v49[1] = v52;
        if ( v52 )
          *(_QWORD *)(v52 + 16) = (unsigned __int64)(v49 + 1) | *(_QWORD *)(v52 + 16) & 3LL;
        v49[2] = (v57 + 8) | v49[2] & 3;
        *(_QWORD *)(v57 + 8) = v49;
        if ( *(_QWORD *)(v29 + 8) )
          goto LABEL_45;
      }
      else if ( *(_QWORD *)(v29 + 8) )
      {
        goto LABEL_120;
      }
      v91 = *(_DWORD *)(a5 + 8);
      if ( v91 >= *(_DWORD *)(a5 + 12) )
      {
        sub_170B450(a5, 0);
        v91 = *(_DWORD *)(a5 + 8);
      }
      v92 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v91);
      if ( v92 )
      {
        *v92 = 6;
        v92[1] = 0;
        v92[2] = v29;
        if ( v29 != -16 && v29 != -8 )
          sub_164C220((__int64)v92);
        v91 = *(_DWORD *)(a5 + 8);
      }
      *(_DWORD *)(a5 + 8) = v91 + 1;
      if ( !v57 )
      {
LABEL_120:
        v369 = 1;
        v54 = *((_BYTE *)v28 + 16);
LABEL_84:
        if ( v54 == 75 )
        {
          v68 = *((unsigned __int16 *)v28 + 9);
          BYTE1(v68) &= ~0x80u;
          v356 = v68;
          v69 = *(v28 - 6);
          if ( v69 && v29 == v69 )
          {
            v70 = v356;
            v71 = -24;
            v72 = 0x1FFFFFFFFFFFFFFALL;
          }
          else
          {
            v93 = sub_15FF5D0(v356);
            v71 = -48;
            v72 = 0x1FFFFFFFFFFFFFFDLL;
            v70 = v93;
          }
          v73 = 0;
          v74 = *(_DWORD *)(a4 + 24);
          if ( v74 )
          {
            v75 = v28[5];
            v76 = *(_QWORD *)(a4 + 8);
            v77 = v74 - 1;
            v78 = v77 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            v79 = (__int64 *)(v76 + 16LL * v78);
            v80 = *v79;
            if ( v75 == *v79 )
            {
LABEL_90:
              v73 = (_QWORD *)v79[1];
            }
            else
            {
              v248 = 1;
              while ( v80 != -8 )
              {
                v284 = v248 + 1;
                v78 = v77 & (v248 + v78);
                v79 = (__int64 *)(v76 + 16LL * v78);
                v80 = *v79;
                if ( v75 == *v79 )
                  goto LABEL_90;
                v248 = v284;
              }
              v73 = 0;
            }
          }
          v334 = v71;
          v343 = v73;
          v81 = sub_1472610((__int64)v25, v28[v72], v73);
          v344 = sub_1472610((__int64)v25, *(__int64 *)((char *)v28 + v334), v343);
          if ( (unsigned __int8)sub_147A340((__int64)v25, v70, v81, v344) )
          {
            v82 = (__int64 *)sub_16498A0((__int64)v28);
            v83 = sub_159C4F0(v82);
            goto LABEL_93;
          }
          v120 = sub_15FF0F0(v70);
          if ( (unsigned __int8)sub_147A340((__int64)v25, v120, v81, v344) )
          {
            v121 = (__int64 *)sub_16498A0((__int64)v28);
            v83 = sub_159C540(v121);
LABEL_93:
            v86 = v83;
LABEL_94:
            sub_164D160((__int64)v28, v86, a7, *(double *)a8.m128i_i64, a9, a10, v84, v85, a13, a14);
            v87 = *(_DWORD *)(a5 + 8);
            if ( v87 >= *(_DWORD *)(a5 + 12) )
            {
              sub_170B450(a5, 0);
              v87 = *(_DWORD *)(a5 + 8);
            }
            v88 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v87);
            if ( !v88 )
            {
LABEL_97:
              *(_DWORD *)(a5 + 8) = v87 + 1;
              goto LABEL_98;
            }
            *v88 = 6;
            v88[1] = 0;
            v88[2] = v28;
            if ( v28 != (__int64 *)-8LL && v28 != (__int64 *)-16LL )
LABEL_167:
              sub_164C220((__int64)v88);
LABEL_168:
            v87 = *(_DWORD *)(a5 + 8);
            goto LABEL_97;
          }
          v122 = *(v28 - 6);
          v123 = *((_WORD *)v28 + 9) & 0x7FFF;
          if ( !v122 || (v124 = 0x1FFFFFFFFFFFFFFDLL, v125 = 0x1FFFFFFFFFFFFFFALL, v29 != v122) )
          {
            v124 = 0x1FFFFFFFFFFFFFFALL;
            v213 = sub_15FF5D0(v123);
            v125 = 0x1FFFFFFFFFFFFFFDLL;
            v123 = v213;
          }
          v126 = 0;
          v127 = *(_DWORD *)(a4 + 24);
          if ( v127 )
          {
            v128 = v28[5];
            v129 = *(_QWORD *)(a4 + 8);
            v130 = v127 - 1;
            v131 = v130 & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
            v132 = (__int64 *)(v129 + 16LL * v131);
            v133 = *v132;
            if ( v128 == *v132 )
            {
LABEL_173:
              v126 = (_QWORD *)v132[1];
            }
            else
            {
              v281 = 1;
              while ( v133 != -8 )
              {
                v287 = v281 + 1;
                v131 = v130 & (v281 + v131);
                v132 = (__int64 *)(v129 + 16LL * v131);
                v133 = *v132;
                if ( v128 == *v132 )
                  goto LABEL_173;
                v281 = v287;
              }
              v126 = 0;
            }
          }
          v134 = &v28[v124];
          v324 = v123;
          v335 = v126;
          v328 = &v28[v125];
          v372 = sub_1472610((__int64)v25, v28[v125], v126);
          v135 = sub_1472610((__int64)v25, *v134, v335);
          v136 = *(_BYTE *)(v29 + 16) == 77;
          v373 = v135;
          if ( v136 && (unsigned __int8)sub_14798F0((__int64)v25, v324, v372, v135, v366, &v371, &v374, &v375) )
          {
            v198 = &v381;
            v379 = 0;
            v380 = 1;
            do
            {
              *v198 = (__int64 *)-8LL;
              v198 += 2;
            }
            while ( v198 != (__int64 **)&v385 );
            sub_1B64710((__int64)&v379, &v372)[1] = *v328;
            sub_1B64710((__int64)&v379, &v373)[1] = *v134;
            v199 = sub_13FC470(v366);
            if ( v199 )
            {
              v200 = *(_DWORD *)(v29 + 20) & 0xFFFFFFF;
              if ( v200 )
              {
                v201 = 24LL * *(unsigned int *)(v29 + 56) + 8;
                v202 = 0;
                while ( 1 )
                {
                  v203 = v29 - 24LL * v200;
                  if ( (*(_BYTE *)(v29 + 23) & 0x40) != 0 )
                    v203 = *(_QWORD *)(v29 - 8);
                  if ( v199 == *(_QWORD *)(v203 + v201) )
                    break;
                  ++v202;
                  v201 += 8;
                  if ( v200 == v202 )
                    goto LABEL_252;
                }
                if ( v202 >= 0 )
                {
                  v204 = *(_QWORD *)(v203 + 24LL * v202);
                  v376 = (char *)sub_146F1B0((__int64)v25, v204);
                  sub_1B64710((__int64)&v379, (__int64 *)&v376)[1] = v204;
                }
              }
            }
LABEL_252:
            v205 = sub_1B64710((__int64)&v379, &v374)[1];
            v206 = sub_1B64710((__int64)&v379, &v375)[1];
            if ( v205 )
            {
              if ( v206 )
              {
LABEL_254:
                v136 = *(v28 - 6) == 0;
                *((_WORD *)v28 + 9) = v371 | *((_WORD *)v28 + 9) & 0x8000;
                if ( !v136 )
                {
                  v207 = *(v28 - 5);
                  v208 = *(v28 - 4) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v208 = v207;
                  if ( v207 )
                    *(_QWORD *)(v207 + 16) = v208 | *(_QWORD *)(v207 + 16) & 3LL;
                }
                *(v28 - 6) = v205;
                v209 = *(_QWORD *)(v205 + 8);
                *(v28 - 5) = v209;
                if ( v209 )
                  *(_QWORD *)(v209 + 16) = (unsigned __int64)(v28 - 5) | *(_QWORD *)(v209 + 16) & 3LL;
                *(v28 - 4) = (v205 + 8) | *(v28 - 4) & 3;
                *(_QWORD *)(v205 + 8) = v28 - 6;
                if ( *(v28 - 3) )
                {
                  v210 = *(v28 - 2);
                  v211 = *(v28 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v211 = v210;
                  if ( v210 )
                    *(_QWORD *)(v210 + 16) = v211 | *(_QWORD *)(v210 + 16) & 3LL;
                }
                *(v28 - 3) = v206;
                v212 = *(_QWORD *)(v206 + 8);
                *(v28 - 2) = v212;
                if ( v212 )
                  *(_QWORD *)(v212 + 16) = (unsigned __int64)(v28 - 2) | *(_QWORD *)(v212 + 16) & 3LL;
                *(v28 - 1) = *(v28 - 1) & 3 | (v206 + 8);
                *(_QWORD *)(v206 + 8) = v28 - 3;
                if ( (v380 & 1) == 0 )
                  j___libc_free_0(v381);
                goto LABEL_98;
              }
LABEL_425:
              if ( !*(_WORD *)(v375 + 24) )
              {
                v206 = *(_QWORD *)(v375 + 32);
                goto LABEL_427;
              }
            }
            else if ( *(_WORD *)(v374 + 24) )
            {
              if ( !v206 )
                goto LABEL_425;
            }
            else
            {
              v205 = *(_QWORD *)(v374 + 32);
              if ( !v206 )
                goto LABEL_425;
LABEL_427:
              if ( v205 && v206 )
                goto LABEL_254;
            }
            if ( (v380 & 1) == 0 )
              j___libc_free_0(v381);
          }
          if ( !sub_15FF7F0(v356)
            || !(unsigned __int8)sub_1477BC0((__int64)v25, v81)
            || !(unsigned __int8)sub_1477BC0((__int64)v25, v344) )
          {
            goto LABEL_99;
          }
          *((_WORD *)v28 + 9) = sub_15FF470(v356) | *((_WORD *)v28 + 9) & 0x8000;
LABEL_98:
          v369 = 1;
          goto LABEL_99;
        }
        if ( (unsigned int)v54 - 35 <= 0x11 )
        {
          if ( (unsigned __int8)(v54 - 44) > 1u )
          {
            if ( v54 != 42 )
              goto LABEL_180;
            v180 = 0;
            v360 = sub_146F1B0((__int64)v25, *(v28 - 6));
            v181 = sub_146F1B0((__int64)v25, *(v28 - 3));
            v182 = *(_DWORD *)(a4 + 24);
            if ( v182 )
            {
              v183 = v28[5];
              v184 = *(_QWORD *)(a4 + 8);
              v185 = v182 - 1;
              v186 = v185 & (((unsigned int)v183 >> 9) ^ ((unsigned int)v183 >> 4));
              v187 = (__int64 *)(v184 + 16LL * v186);
              v188 = *v187;
              if ( *v187 == v183 )
              {
LABEL_224:
                v180 = (_QWORD *)v187[1];
              }
              else
              {
                v285 = 1;
                while ( v188 != -8 )
                {
                  v308 = v285 + 1;
                  v309 = v185 & (v186 + v285);
                  v186 = v309;
                  v187 = (__int64 *)(v184 + 16 * v309);
                  v188 = *v187;
                  if ( v183 == *v187 )
                    goto LABEL_224;
                  v285 = v308;
                }
                v180 = 0;
              }
            }
            v361 = sub_1472270((__int64)v25, v360, v180);
            v189 = sub_1472270((__int64)v25, v181, v180);
            if ( (unsigned __int8)sub_1477BC0((__int64)v25, v361) && (unsigned __int8)sub_1477BC0((__int64)v25, v189) )
            {
              v376 = (char *)sub_1649960((__int64)v28);
              v379 = (unsigned __int64)&v376;
              LOWORD(v381) = 773;
              v377 = v190;
              v380 = (__int64)".udiv";
              v191 = sub_15FB440(17, (__int64 *)*(v28 - 6), *(v28 - 3), (__int64)&v379, (__int64)v28);
              v192 = sub_15F23D0((__int64)v28);
              sub_15F2350(v191, v192);
              sub_164D160((__int64)v28, v191, a7, *(double *)a8.m128i_i64, a9, a10, v193, v194, a13, a14);
              v381 = v28;
              v379 = 6;
              v380 = 0;
              if ( v28 != (__int64 *)-16LL && v28 != (__int64 *)-8LL )
                sub_164C220((__int64)&v379);
              v195 = *(unsigned int *)(a5 + 8);
              if ( (unsigned int)v195 >= *(_DWORD *)(a5 + 12) )
              {
                sub_170B450(a5, 0);
                v195 = *(unsigned int *)(a5 + 8);
              }
              v196 = (unsigned __int64 *)(*(_QWORD *)a5 + 24 * v195);
              if ( v196 )
              {
                *v196 = 6;
                v196[1] = 0;
                v197 = (unsigned __int64)v381;
                v136 = v381 == 0;
                v196[2] = (unsigned __int64)v381;
                if ( v197 != -8 && !v136 && v197 != -16 )
                  sub_1649AC0(v196, v379 & 0xFFFFFFFFFFFFFFF8LL);
              }
              ++*(_DWORD *)(a5 + 8);
              if ( v381 != 0 && v381 + 1 != 0 && v381 != (__int64 *)-16LL )
                sub_1649B30(&v379);
              goto LABEL_98;
            }
            goto LABEL_70;
          }
          v160 = *(v28 - 6);
          if ( v29 != v160 && v54 != 45 )
            goto LABEL_99;
          v347 = *(v28 - 3);
          v161 = sub_146F1B0((__int64)v25, *(v28 - 6));
          v162 = 0;
          v163 = v161;
          v164 = *(_DWORD *)(a4 + 24);
          if ( v164 )
          {
            v165 = v28[5];
            v166 = *(_QWORD *)(a4 + 8);
            v167 = v164 - 1;
            v168 = v167 & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
            v169 = (__int64 *)(v166 + 16LL * v168);
            v170 = *v169;
            if ( v165 == *v169 )
            {
LABEL_209:
              v162 = (_QWORD *)v169[1];
            }
            else
            {
              v286 = 1;
              while ( v170 != -8 )
              {
                v310 = v286 + 1;
                v168 = v167 & (v286 + v168);
                v169 = (__int64 *)(v166 + 16LL * v168);
                v170 = *v169;
                if ( v165 == *v169 )
                  goto LABEL_209;
                v286 = v310;
              }
              v162 = 0;
            }
          }
          v359 = v162;
          v171 = sub_1472270((__int64)v25, v163, v162);
          v337 = v171;
          if ( v54 == 45 )
          {
            if ( !(unsigned __int8)sub_1477BC0((__int64)v25, v171) )
              goto LABEL_99;
            v172 = sub_146F1B0((__int64)v25, v347);
            v173 = sub_1472270((__int64)v25, v172, v359);
            if ( v29 != v160 )
            {
LABEL_213:
              if ( !(unsigned __int8)sub_1477BC0((__int64)v25, v173) )
                goto LABEL_99;
              v174 = (__int64 *)*(v28 - 6);
              v175 = *(v28 - 3);
              v376 = (char *)sub_1649960((__int64)v28);
              v379 = (unsigned __int64)&v376;
              v380 = (__int64)".urem";
              v377 = v176;
              LOWORD(v381) = 773;
              v177 = sub_15FB440(20, v174, v175, (__int64)&v379, (__int64)v28);
              goto LABEL_215;
            }
            v252 = 40;
          }
          else
          {
            v251 = sub_146F1B0((__int64)v25, v347);
            v173 = sub_1472270((__int64)v25, v251, v359);
            if ( v29 != v160 )
              goto LABEL_99;
            v252 = 36;
          }
          v363 = v173;
          if ( (unsigned __int8)sub_147A340((__int64)v25, v252, v337, v173) )
          {
            v86 = *(v28 - 6);
            goto LABEL_94;
          }
          v253 = sub_145CF80((__int64)v25, *v28, 1, 0);
          v254 = sub_14806B0((__int64)v25, v337, v253, 0, 0);
          v255 = sub_147A340((__int64)v25, v252, v254, v363);
          v173 = v363;
          if ( v255 )
          {
            v288 = *(v28 - 3);
            v289 = *v28;
            v290 = (_QWORD *)*(v28 - 6);
            LOWORD(v381) = 257;
            v364 = v288;
            v370 = sub_1648A60(56, 2u);
            if ( v370 )
            {
              v291 = (_QWORD **)*v290;
              if ( *(_BYTE *)(*v290 + 8LL) == 16 )
              {
                v349 = v291[4];
                v292 = (__int64 *)sub_1643320(*v291);
                v293 = (__int64)sub_16463B0(v292, (unsigned int)v349);
              }
              else
              {
                v293 = sub_1643320(*v291);
              }
              sub_15FEC10((__int64)v370, v293, 51, 32, (__int64)v290, v364, (__int64)&v379, (__int64)v28);
            }
            v379 = (unsigned __int64)"iv.rem";
            LOWORD(v381) = 259;
            v365 = (__int64 *)sub_15A0680(v289, 0, 0);
            v294 = sub_1648A60(56, 3u);
            v297 = (__int64)v294;
            if ( v294 )
            {
              v350 = v294 - 9;
              sub_15F1EA0((__int64)v294, *v365, 55, (__int64)(v294 - 9), 3, (__int64)v28);
              if ( *(_QWORD *)(v297 - 72) )
              {
                v298 = *(_QWORD *)(v297 - 64);
                v299 = *(_QWORD *)(v297 - 56) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v299 = v298;
                if ( v298 )
                  *(_QWORD *)(v298 + 16) = v299 | *(_QWORD *)(v298 + 16) & 3LL;
              }
              *(_QWORD *)(v297 - 72) = v370;
              if ( v370 )
              {
                v300 = v370[1];
                *(_QWORD *)(v297 - 64) = v300;
                if ( v300 )
                  *(_QWORD *)(v300 + 16) = (v297 - 64) | *(_QWORD *)(v300 + 16) & 3LL;
                *(_QWORD *)(v297 - 56) = (unsigned __int64)(v370 + 1) | *(_QWORD *)(v297 - 56) & 3LL;
                v370[1] = v350;
              }
              if ( *(_QWORD *)(v297 - 48) )
              {
                v301 = *(_QWORD *)(v297 - 40);
                v302 = *(_QWORD *)(v297 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v302 = v301;
                if ( v301 )
                  *(_QWORD *)(v301 + 16) = v302 | *(_QWORD *)(v301 + 16) & 3LL;
              }
              *(_QWORD *)(v297 - 48) = v365;
              v303 = v365[1];
              *(_QWORD *)(v297 - 40) = v303;
              if ( v303 )
                *(_QWORD *)(v303 + 16) = (v297 - 40) | *(_QWORD *)(v303 + 16) & 3LL;
              *(_QWORD *)(v297 - 32) = (unsigned __int64)(v365 + 1) | *(_QWORD *)(v297 - 32) & 3LL;
              v365[1] = v297 - 48;
              if ( *(_QWORD *)(v297 - 24) )
              {
                v304 = *(_QWORD *)(v297 - 16);
                v305 = *(_QWORD *)(v297 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v305 = v304;
                if ( v304 )
                  *(_QWORD *)(v304 + 16) = v305 | *(_QWORD *)(v304 + 16) & 3LL;
              }
              *(_QWORD *)(v297 - 24) = v290;
              if ( v290 )
              {
                v306 = v290[1];
                *(_QWORD *)(v297 - 16) = v306;
                if ( v306 )
                  *(_QWORD *)(v306 + 16) = (v297 - 16) | *(_QWORD *)(v306 + 16) & 3LL;
                *(_QWORD *)(v297 - 8) = (unsigned __int64)(v290 + 1) | *(_QWORD *)(v297 - 8) & 3LL;
                v290[1] = v297 - 24;
              }
              sub_164B780(v297, (__int64 *)&v379);
            }
            sub_164D160((__int64)v28, v297, a7, *(double *)a8.m128i_i64, a9, a10, v295, v296, a13, a14);
            if ( *(_DWORD *)(a5 + 8) >= *(_DWORD *)(a5 + 12) )
              sub_170B450(a5, 0);
            v307 = (_QWORD *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8));
            if ( v307 )
            {
              *v307 = 6;
              v307[1] = 0;
              v307[2] = v28;
              if ( v28 != (__int64 *)-16LL && v28 != (__int64 *)-8LL )
                sub_164C220((__int64)v307);
            }
            v369 = 1;
            ++*(_DWORD *)(a5 + 8);
            goto LABEL_99;
          }
          if ( v54 != 45 )
            goto LABEL_99;
          goto LABEL_213;
        }
LABEL_180:
        if ( v54 == 78 )
        {
          v137 = *(v28 - 3);
          if ( !*(_BYTE *)(v137 + 16) )
          {
            v138 = *(_DWORD *)(v137 + 36);
            if ( v138 == 209 )
            {
              v141 = sub_13A5B00;
              v139 = 0;
              v140 = 11;
              v346 = sub_14747F0;
              goto LABEL_187;
            }
            if ( v138 > 0xD1 )
            {
              if ( v138 == 211 )
              {
                v141 = sub_14806B0;
                v139 = 0;
                v140 = 13;
                v346 = sub_14747F0;
                goto LABEL_187;
              }
            }
            else
            {
              if ( v138 == 189 )
              {
                v139 = v353;
                v140 = 11;
                v141 = sub_13A5B00;
                v346 = sub_147B0D0;
                goto LABEL_187;
              }
              if ( v138 == 198 )
              {
                v139 = v353;
                v140 = 13;
                v141 = sub_14806B0;
                v346 = sub_147B0D0;
LABEL_187:
                v321 = v139;
                v322 = v140;
                v329 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, _QWORD))v141;
                v142 = sub_146F1B0((__int64)v25, v28[-3 * (*((_DWORD *)v28 + 5) & 0xFFFFFFF)]);
                v143 = sub_146F1B0((__int64)v25, v28[3 * (1LL - (*((_DWORD *)v28 + 5) & 0xFFFFFFF))]);
                v144 = sub_1456040(v142);
                v336 = sub_1644900(*(_QWORD **)v144, 2 * (*(_DWORD *)(v144 + 8) >> 8));
                v325 = v329;
                v145 = v329(v25, v142, v143, 0, 0);
                v330 = v346((__int64)v25, v145, v336, 0);
                v146 = v346((__int64)v25, v143, v336, 0);
                v147 = v346((__int64)v25, v142, v336, 0);
                if ( v330 == v325(v25, v147, v146, 0, 0) )
                {
                  LOWORD(v381) = 257;
                  v148 = sub_15FB440(
                           v322,
                           (__int64 *)v28[-3 * (*((_DWORD *)v28 + 5) & 0xFFFFFFF)],
                           v28[3 * (1LL - (*((_DWORD *)v28 + 5) & 0xFFFFFFF))],
                           (__int64)&v379,
                           (__int64)v28);
                  v149 = v148;
                  if ( v321 )
                    sub_15F2330(v148, 1);
                  else
                    sub_15F2310(v148, 1);
                  v379 = (unsigned __int64)&v381;
                  v380 = 0x400000000LL;
                  if ( !v28[1] )
                    goto LABEL_404;
                  v358 = v25;
                  v150 = v28[1];
                  v151 = v149;
                  do
                  {
                    v152 = sub_1648700(v150);
                    v155 = (__int64)v152;
                    if ( *((_BYTE *)v152 + 16) == 86 )
                    {
                      if ( *(_DWORD *)v152[7] == 1 )
                      {
                        v215 = (__int64 *)sub_16498A0((__int64)v28);
                        v216 = sub_159C540(v215);
                        sub_164D160(v155, v216, a7, *(double *)a8.m128i_i64, a9, a10, v217, v218, a13, a14);
                      }
                      else
                      {
                        sub_164D160((__int64)v152, v151, a7, *(double *)a8.m128i_i64, a9, a10, v153, v154, a13, a14);
                      }
                      v158 = (unsigned int)v380;
                      if ( (unsigned int)v380 >= HIDWORD(v380) )
                      {
                        sub_16CD150((__int64)&v379, &v381, 0, 8, v156, v157);
                        v158 = (unsigned int)v380;
                      }
                      *(_QWORD *)(v379 + 8 * v158) = v155;
                      LODWORD(v380) = v380 + 1;
                    }
                    v150 = *(_QWORD *)(v150 + 8);
                  }
                  while ( v150 );
                  v245 = (_QWORD **)v379;
                  v25 = v358;
                  v246 = (_QWORD **)(v379 + 8LL * (unsigned int)v380);
                  if ( (_QWORD **)v379 != v246 )
                  {
                    do
                    {
                      v247 = *v245++;
                      sub_15F20C0(v247);
                    }
                    while ( v246 != v245 );
                  }
                  if ( !v28[1] )
LABEL_404:
                    sub_15F20C0(v28);
                  goto LABEL_311;
                }
              }
            }
          }
        }
LABEL_61:
        if ( *((_BYTE *)v28 + 16) != 60 )
          goto LABEL_68;
        v59 = (__int64 *)*(v28 - 3);
        v323 = (__int64)v59;
        v327 = *v59;
        v60 = sub_146F1B0((__int64)v25, (__int64)v59);
        v61 = sub_146F1B0((__int64)v25, (__int64)v28);
        if ( v60 != sub_147B0D0((__int64)v25, v61, v327, 0) )
        {
          if ( v60 == sub_14747F0((__int64)v25, v61, v327, 0) )
          {
            v333 = v353;
            goto LABEL_64;
          }
LABEL_68:
          if ( !sub_1456C80((__int64)v25, *v28) )
            goto LABEL_70;
          if ( *v28 != *(_QWORD *)v29 )
            goto LABEL_70;
          v231 = sub_146F1B0((__int64)v25, (__int64)v28);
          if ( v231 != sub_146F1B0((__int64)v25, v29) )
            goto LABEL_70;
          if ( *((_BYTE *)v28 + 16) == 77 )
          {
            if ( !a3 )
              goto LABEL_126;
            if ( sub_15CCEE0(a3, v29, (__int64)v28) )
              goto LABEL_298;
LABEL_70:
            v65 = *((unsigned __int8 *)v28 + 16);
            v66 = (unsigned int)(v65 - 35);
            v67 = *((unsigned __int8 *)v28 + 16);
            if ( (unsigned int)v66 > 0x11 )
              goto LABEL_78;
            if ( (unsigned __int8)v65 > 0x2Fu
              || (v355 = ((0x80A800000000uLL >> v67) & 1) == 0, ((0x80A800000000uLL >> v67) & 1) == 0) )
            {
LABEL_126:
              if ( sub_1456C80((__int64)v25, *v28) )
              {
                v94 = sub_146F1B0((__int64)v25, (__int64)v28);
                if ( *(_WORD *)(v94 + 24) == 7 && v366 == *(_QWORD *)(v94 + 48) )
                  sub_1B64100(v28, v366, (__int64)&v389, (__int64)&v386);
              }
              goto LABEL_20;
            }
            if ( !sub_15F2370((__int64)v28) || !sub_15F2380((__int64)v28) )
            {
              v67 = *((unsigned __int8 *)v28 + 16);
              v65 = *((unsigned __int8 *)v28 + 16);
              v66 = (unsigned int)(v67 - 24);
              switch ( (_DWORD)v67 )
              {
                case '%':
                  v348 = sub_14806B0;
                  break;
                case '\'':
                  v348 = sub_13A5B60;
                  break;
                case '#':
                  v348 = sub_13A5B00;
                  break;
                default:
                  goto LABEL_77;
              }
              v219 = *(_DWORD *)(*v28 + 8);
              v220 = (_QWORD *)sub_16498A0((__int64)v28);
              v221 = sub_1644900(v220, 2 * (v219 >> 8));
              v338 = sub_146F1B0((__int64)v25, *(v28 - 6));
              v222 = sub_146F1B0((__int64)v25, *(v28 - 3));
              if ( sub_15F2370((__int64)v28)
                || (v226 = sub_146F1B0((__int64)v25, (__int64)v28),
                    v332 = sub_14747F0((__int64)v25, v226, v221, 0),
                    v326 = sub_14747F0((__int64)v25, v222, v221, 0),
                    v227 = sub_14747F0((__int64)v25, v338, v221, 0),
                    v332 != ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, _QWORD))v348)(
                              v25,
                              v227,
                              v326,
                              0,
                              0)) )
              {
                if ( sub_15F2380((__int64)v28) )
                  goto LABEL_278;
                v223 = sub_146F1B0((__int64)v25, (__int64)v28);
                v331 = sub_147B0D0((__int64)v25, v223, v221, 0);
                v224 = sub_147B0D0((__int64)v25, v222, v221, 0);
                v225 = sub_147B0D0((__int64)v25, v338, v221, 0);
                if ( v331 != ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, _QWORD))v348)(
                               v25,
                               v225,
                               v224,
                               0,
                               0) )
                  goto LABEL_278;
              }
              else
              {
                sub_15F2310((__int64)v28, 1);
                sub_1464C80((__int64)v25, (__int64)v28);
                if ( sub_15F2380((__int64)v28) )
                  goto LABEL_284;
                v228 = sub_146F1B0((__int64)v25, (__int64)v28);
                v362 = sub_147B0D0((__int64)v25, v228, v221, 0);
                v229 = sub_147B0D0((__int64)v25, v222, v221, 0);
                v230 = sub_147B0D0((__int64)v25, v338, v221, 0);
                if ( v362 != ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, _QWORD))v348)(
                               v25,
                               v230,
                               v229,
                               0,
                               0) )
                  goto LABEL_284;
              }
              sub_15F2330((__int64)v28, 1);
              sub_1464C80((__int64)v25, (__int64)v28);
              goto LABEL_284;
            }
LABEL_278:
            v65 = *((unsigned __int8 *)v28 + 16);
LABEL_77:
            if ( (_BYTE)v65 != 47 )
            {
LABEL_78:
              if ( (unsigned int)(v65 - 60) <= 0xC && a15 )
              {
                (*(void (__fastcall **)(__int64, __int64 *, __int64, __int64))(*(_QWORD *)a15 + 24LL))(
                  a15,
                  v28,
                  v66,
                  v67);
                goto LABEL_20;
              }
              goto LABEL_126;
            }
            v261 = sub_146F1B0((__int64)v25, v29);
            v262 = sub_1477920((__int64)v25, v261, 0);
            LODWORD(v380) = *((_DWORD *)v262 + 2);
            if ( (unsigned int)v380 > 0x40 )
            {
              v261 = (__int64)v262;
              sub_16A4FD0((__int64)&v379, (const void **)v262);
            }
            else
            {
              v379 = *v262;
            }
            LODWORD(v382) = *((_DWORD *)v262 + 6);
            if ( (unsigned int)v382 > 0x40 )
            {
              v261 = (__int64)(v262 + 2);
              sub_16A4FD0((__int64)&v381, (const void **)v262 + 2);
              v263 = v28[1];
              if ( !v263 )
                goto LABEL_387;
            }
            else
            {
              v381 = (__int64 *)v262[2];
              v263 = v28[1];
              if ( !v263 )
              {
                if ( (unsigned int)v380 > 0x40 )
                {
                  v279 = v379;
                  if ( v379 )
                    goto LABEL_392;
                }
                goto LABEL_285;
              }
            }
            v339 = v28;
            while ( 2 )
            {
              v264 = sub_1648700(v263);
              v265 = *((_BYTE *)v264 + 16);
              switch ( v265 )
              {
                case 49:
                  v266 = *(v264 - 6);
                  v66 = *(unsigned __int8 *)(v266 + 16);
                  if ( (_BYTE)v66 != 47 )
                  {
                    if ( (_BYTE)v66 == 5 && *(_WORD *)(v266 + 18) == 23 )
                    {
                      v67 = *(_DWORD *)(v266 + 20) & 0xFFFFFFF;
                      v66 = 3 * (1 - v67);
                      v267 = *(_QWORD *)(v266 + 24 * (1 - v67));
                      if ( v267 )
                      {
                        if ( v29 == v267 )
                          goto LABEL_348;
                      }
                    }
LABEL_341:
                    v263 = *(_QWORD *)(v263 + 8);
                    if ( !v263 )
                    {
                      v28 = v339;
LABEL_387:
                      if ( (unsigned int)v382 > 0x40 && v381 )
                        j_j___libc_free_0_0(v381);
                      if ( (unsigned int)v380 > 0x40 )
                      {
                        v279 = v379;
                        if ( v379 )
LABEL_392:
                          j_j___libc_free_0_0(v279);
                      }
                      if ( !v355 )
                      {
LABEL_285:
                        v65 = *((unsigned __int8 *)v28 + 16);
                        goto LABEL_78;
                      }
LABEL_284:
                      sub_1B64100((__int64 *)v29, v366, (__int64)&v389, (__int64)&v386);
                      goto LABEL_285;
                    }
                    continue;
                  }
                  v275 = *(_QWORD *)(v266 - 24);
                  if ( v29 != v275 || !v275 )
                    goto LABEL_341;
LABEL_348:
                  v268 = (unsigned __int64 *)*(v264 - 3);
                  v269 = *((_BYTE *)v268 + 16);
                  if ( v269 == 13 )
                  {
LABEL_349:
                    v270 = v268 + 3;
                    goto LABEL_350;
                  }
                  v66 = *v268;
                  if ( *(_BYTE *)(*v268 + 8) != 16 || v269 > 0x10u )
                    goto LABEL_341;
LABEL_378:
                  v277 = sub_15A1020(v268, v261, v66, v67);
                  if ( v277 )
                  {
                    v270 = (unsigned __int64 *)(v277 + 24);
                    if ( *(_BYTE *)(v277 + 16) == 13 )
                    {
LABEL_350:
                      if ( !sub_15F23D0((__int64)v264) )
                      {
                        sub_158AAD0((__int64)&v376, (__int64)&v379);
                        v261 = (__int64)v270;
                        if ( (int)sub_16A9900((__int64)&v376, v270) < 0 )
                        {
                          if ( (unsigned int)v377 > 0x40 && v376 )
                            j_j___libc_free_0_0(v376);
                        }
                        else
                        {
                          if ( (unsigned int)v377 > 0x40 && v376 )
                            j_j___libc_free_0_0(v376);
                          v261 = 1;
                          sub_15F2350((__int64)v264, 1);
                          v355 = v353;
                        }
                      }
                      goto LABEL_341;
                    }
                  }
                  v278 = *((_BYTE *)v264 + 16);
                  if ( v278 != 48 )
                  {
                    if ( v278 != 5 )
                      goto LABEL_341;
                    goto LABEL_340;
                  }
LABEL_357:
                  v271 = *(v264 - 6);
                  v66 = *(unsigned __int8 *)(v271 + 16);
                  if ( (_BYTE)v66 == 47 )
                  {
                    v282 = *(_QWORD *)(v271 - 24);
                    if ( !v282 || v29 != v282 )
                      goto LABEL_341;
                  }
                  else
                  {
                    if ( (_BYTE)v66 != 5 )
                      goto LABEL_341;
                    if ( *(_WORD *)(v271 + 18) != 23 )
                      goto LABEL_341;
                    v67 = *(_DWORD *)(v271 + 20) & 0xFFFFFFF;
                    v66 = 3 * (1 - v67);
                    v272 = *(_QWORD *)(v271 + 24 * (1 - v67));
                    if ( v29 != v272 || !v272 )
                      goto LABEL_341;
                  }
                  v268 = (unsigned __int64 *)*(v264 - 3);
                  v273 = *((_BYTE *)v268 + 16);
                  if ( v273 == 13 )
                    goto LABEL_349;
                  v66 = *v268;
                  if ( *(_BYTE *)(*v268 + 8) != 16 || v273 > 0x10u )
                    goto LABEL_341;
                  break;
                case 5:
                  if ( *((_WORD *)v264 + 9) != 25 )
                    goto LABEL_340;
                  v276 = *((_DWORD *)v264 + 5) & 0xFFFFFFF;
                  v67 = v264[-3 * v276];
                  v66 = *(unsigned __int8 *)(v67 + 16);
                  if ( (_BYTE)v66 == 47 )
                  {
                    v66 = *(_QWORD *)(v67 - 24);
                    if ( v29 != v66 || !v66 )
                      goto LABEL_340;
                  }
                  else
                  {
                    if ( (_BYTE)v66 != 5 )
                      goto LABEL_340;
                    if ( *(_WORD *)(v67 + 18) != 23 )
                      goto LABEL_340;
                    v261 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
                    v66 = *(_QWORD *)(v67 + 24 * (1 - v261));
                    if ( !v66 || v29 != v66 )
                      goto LABEL_340;
                  }
                  v66 = 1 - v276;
                  v268 = (unsigned __int64 *)v264[3 * (1 - v276)];
                  if ( *((_BYTE *)v268 + 16) == 13 )
                    goto LABEL_349;
                  if ( *(_BYTE *)(*v268 + 8) == 16 )
                    goto LABEL_378;
LABEL_340:
                  if ( *((_WORD *)v264 + 9) != 24 )
                    goto LABEL_341;
                  v283 = *((_DWORD *)v264 + 5) & 0xFFFFFFF;
                  v67 = v264[-3 * v283];
                  v66 = *(unsigned __int8 *)(v67 + 16);
                  if ( (_BYTE)v66 == 47 )
                  {
                    v66 = *(_QWORD *)(v67 - 24);
                    if ( !v66 || v29 != v66 )
                      goto LABEL_341;
                  }
                  else
                  {
                    if ( (_BYTE)v66 != 5 )
                      goto LABEL_341;
                    if ( *(_WORD *)(v67 + 18) != 23 )
                      goto LABEL_341;
                    v261 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
                    v66 = *(_QWORD *)(v67 + 24 * (1 - v261));
                    if ( v29 != v66 || !v66 )
                      goto LABEL_341;
                  }
                  v66 = 1 - v283;
                  v268 = (unsigned __int64 *)v264[3 * (1 - v283)];
                  if ( *((_BYTE *)v268 + 16) == 13 )
                    goto LABEL_349;
                  if ( *(_BYTE *)(*v268 + 8) != 16 )
                    goto LABEL_341;
                  break;
                case 48:
                  goto LABEL_357;
                default:
                  goto LABEL_341;
              }
              break;
            }
            v274 = sub_15A1020(v268, v261, v66, v67);
            if ( !v274 || *(_BYTE *)(v274 + 16) != 13 )
              goto LABEL_341;
            v270 = (unsigned __int64 *)(v274 + 24);
            goto LABEL_350;
          }
LABEL_298:
          if ( *(_BYTE *)(v29 + 16) > 0x17u )
          {
            v232 = *(_QWORD *)(v29 + 40);
            v233 = v28[5];
            if ( v232 != v233 )
            {
              v234 = *(_DWORD *)(a4 + 24);
              if ( v234 )
              {
                v235 = *(_QWORD *)(a4 + 8);
                v236 = v234 - 1;
                v237 = v236 & (((unsigned int)v232 >> 9) ^ ((unsigned int)v232 >> 4));
                v238 = (__int64 *)(v235 + 16LL * v237);
                v239 = *v238;
                if ( v232 == *v238 )
                {
LABEL_302:
                  v240 = (_QWORD *)v238[1];
                  if ( v240 )
                  {
                    v241 = v236 & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
                    v242 = (__int64 *)(v235 + 16LL * v241);
                    v243 = *v242;
                    if ( *v242 != v233 )
                    {
                      v311 = 1;
                      while ( v243 != -8 )
                      {
                        v312 = v311 + 1;
                        v241 = v236 & (v311 + v241);
                        v242 = (__int64 *)(v235 + 16LL * v241);
                        v243 = *v242;
                        if ( v233 == *v242 )
                          goto LABEL_304;
                        v311 = v312;
                      }
                      goto LABEL_70;
                    }
LABEL_304:
                    v244 = (_QWORD *)v242[1];
                    if ( v240 != v244 )
                    {
                      while ( v244 )
                      {
                        v244 = (_QWORD *)*v244;
                        if ( v240 == v244 )
                          goto LABEL_307;
                      }
                      goto LABEL_70;
                    }
                  }
                }
                else
                {
                  v313 = 1;
                  while ( v239 != -8 )
                  {
                    v314 = v313 + 1;
                    v237 = v236 & (v313 + v237);
                    v238 = (__int64 *)(v235 + 16LL * v237);
                    v239 = *v238;
                    if ( v232 == *v238 )
                      goto LABEL_302;
                    v313 = v314;
                  }
                }
              }
            }
          }
LABEL_307:
          v177 = v29;
LABEL_215:
          sub_164D160((__int64)v28, v177, a7, *(double *)a8.m128i_i64, a9, a10, v178, v179, a13, a14);
          v87 = *(_DWORD *)(a5 + 8);
          if ( v87 >= *(_DWORD *)(a5 + 12) )
          {
            sub_170B450(a5, 0);
            v87 = *(_DWORD *)(a5 + 8);
          }
          v88 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v87);
          if ( !v88 )
            goto LABEL_97;
          *v88 = 6;
          v88[1] = 0;
          v88[2] = v28;
          if ( v28 != (__int64 *)-16LL && v28 != (__int64 *)-8LL )
            goto LABEL_167;
          goto LABEL_168;
        }
        v333 = v60 == sub_14747F0((__int64)v25, v61, v327, 0);
        v354 = v353;
LABEL_64:
        v379 = (unsigned __int64)&v381;
        v380 = 0x400000000LL;
        v62 = v28[1];
        if ( !v62 )
          goto LABEL_328;
        while ( 2 )
        {
          v63 = sub_1648700(v62);
          v64 = *((_BYTE *)v63 + 16);
          if ( v64 <= 0x17u )
          {
LABEL_66:
            if ( (__int64 **)v379 != &v381 )
              _libc_free(v379);
            goto LABEL_68;
          }
          v95 = *(unsigned int *)(a3 + 48);
          if ( (_DWORD)v95 )
          {
            v96 = v63[5];
            v97 = *(_QWORD *)(a3 + 32);
            v98 = (v95 - 1) & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
            v99 = (__int64 *)(v97 + 16LL * v98);
            v100 = *v99;
            if ( v96 == *v99 )
            {
LABEL_132:
              if ( v99 != (__int64 *)(v97 + 16 * v95) && v99[1] )
              {
                if ( v64 != 75 )
                  goto LABEL_66;
                v101 = (__int64 *)*(v63 - 6);
                if ( !v101
                  || v28 != v101
                  || !sub_13FC1A0(v366, *(v63 - 3))
                  || sub_15FF7F0(*((_WORD *)v63 + 9) & 0x7FFF) && !v354 )
                {
                  goto LABEL_66;
                }
                if ( sub_15FF7E0(*((_WORD *)v63 + 9) & 0x7FFF) && !v333 )
                  goto LABEL_66;
                v104 = (unsigned int)v380;
                if ( (unsigned int)v380 >= HIDWORD(v380) )
                {
                  sub_16CD150((__int64)&v379, &v381, 0, 8, v102, v103);
                  v104 = (unsigned int)v380;
                }
                *(_QWORD *)(v379 + 8 * v104) = v63;
                LODWORD(v380) = v380 + 1;
              }
            }
            else
            {
              v249 = 1;
              while ( v100 != -8 )
              {
                v250 = v249 + 1;
                v98 = (v95 - 1) & (v249 + v98);
                v99 = (__int64 *)(v97 + 16LL * v98);
                v100 = *v99;
                if ( v96 == *v99 )
                  goto LABEL_132;
                v249 = v250;
              }
            }
          }
          v62 = *(_QWORD *)(v62 + 8);
          if ( v62 )
            continue;
          break;
        }
        v320 = v379 + 8LL * (unsigned int)v380;
        if ( v379 != v320 )
        {
          v357 = v379;
          v316 = v29;
          v315 = v28;
          v319 = (__int64)v25;
          do
          {
            v111 = *(_QWORD *)v357;
            v112 = *(_QWORD *)(*(_QWORD *)v357 - 24LL);
            v113 = *(_WORD *)(*(_QWORD *)v357 + 18LL) & 0x7FFF;
            if ( sub_15FF7E0(v113)
              || v333
              && ((v114 = *(unsigned __int16 *)(v111 + 18), BYTE1(v114) &= ~0x80u, (unsigned int)(v114 - 32) <= 1)
               || (v115 = sub_146F1B0(v319, *(_QWORD *)(v111 - 48)),
                   v318 = sub_146F1B0(v319, *(_QWORD *)(v111 - 24)),
                   (unsigned __int8)sub_1477BC0(v319, v115))
               && (unsigned __int8)sub_1477BC0(v319, v318)) )
            {
              v376 = "zext";
              v378 = 259;
              v214 = sub_1648A60(56, 1u);
              v117 = (__int64)v214;
              if ( v214 )
                sub_15FC690((__int64)v214, v112, v327, (__int64)&v376, v111);
              LOWORD(v113) = sub_15FF470(v113);
            }
            else
            {
              v376 = "sext";
              v378 = 259;
              v116 = sub_1648A60(56, 1u);
              v117 = (__int64)v116;
              if ( v116 )
                sub_15FC810((__int64)v116, v112, v327, (__int64)&v376, v111);
            }
            sub_13FC570(v366, v117, &v375, 0);
            v378 = 257;
            v118 = sub_1648A60(56, 2u);
            if ( v118 )
            {
              v119 = *(_QWORD ***)v323;
              if ( *(_BYTE *)(*(_QWORD *)v323 + 8LL) == 16 )
              {
                v317 = v119[4];
                v105 = (__int64 *)sub_1643320(*v119);
                v106 = (__int64)sub_16463B0(v105, (unsigned int)v317);
              }
              else
              {
                v106 = sub_1643320(*v119);
              }
              sub_15FEC10((__int64)v118, v106, 51, v113, v323, v117, (__int64)&v376, v111);
            }
            sub_164D160(v111, (__int64)v118, a7, *(double *)a8.m128i_i64, a9, a10, v107, v108, a13, a14);
            v109 = *(_DWORD *)(a5 + 8);
            if ( v109 >= *(_DWORD *)(a5 + 12) )
            {
              sub_170B450(a5, 0);
              v109 = *(_DWORD *)(a5 + 8);
            }
            v110 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v109);
            if ( v110 )
            {
              *v110 = 6;
              v110[1] = 0;
              v110[2] = v111;
              if ( v111 != -8 && v111 != -16 )
                sub_164C220((__int64)v110);
              v109 = *(_DWORD *)(a5 + 8);
            }
            v357 += 8LL;
            *(_DWORD *)(a5 + 8) = v109 + 1;
          }
          while ( v320 != v357 );
          v29 = v316;
          v28 = v315;
          v25 = (_QWORD *)v319;
        }
LABEL_328:
        v256 = sub_1599EF0((__int64 **)*v28);
        sub_164D160((__int64)v28, v256, a7, *(double *)a8.m128i_i64, a9, a10, v257, v258, a13, a14);
        v259 = *(_DWORD *)(a5 + 8);
        if ( v259 >= *(_DWORD *)(a5 + 12) )
        {
          sub_170B450(a5, 0);
          v259 = *(_DWORD *)(a5 + 8);
        }
        v260 = (_QWORD *)(*(_QWORD *)a5 + 24LL * v259);
        if ( v260 )
        {
          *v260 = 6;
          v260[1] = 0;
          v260[2] = v28;
          if ( v28 != (__int64 *)-16LL && v28 != (__int64 *)-8LL )
            sub_164C220((__int64)v260);
          v259 = *(_DWORD *)(a5 + 8);
        }
        *(_DWORD *)(a5 + 8) = v259 + 1;
LABEL_311:
        if ( (__int64 **)v379 != &v381 )
          _libc_free(v379);
LABEL_99:
        sub_1B64100((__int64 *)v29, v366, (__int64)&v389, (__int64)&v386);
LABEL_20:
        v26 = v387;
        if ( !(_DWORD)v387 )
        {
          v353 = v369;
          goto LABEL_22;
        }
        goto LABEL_7;
      }
LABEL_45:
      v369 = 1;
      if ( *(_BYTE *)(v57 + 16) <= 0x17u )
        goto LABEL_20;
      v29 = v57;
    }
    v342 = *(_DWORD *)(v45 + 32);
    v58 = *(_DWORD *)(*v28 + 8) >> 8;
    if ( v342 > 0x40 )
    {
      if ( v342 - (unsigned int)sub_16A57B0(v45 + 24) > 0x40 )
        goto LABEL_61;
      v39 = **(_QWORD **)(v45 + 24);
      if ( v58 <= v39 )
        goto LABEL_61;
    }
    else
    {
      v39 = *(_QWORD *)(v45 + 24);
      if ( v58 <= v39 )
        goto LABEL_61;
    }
    v40 = v39;
    LODWORD(v380) = v58;
    v41 = 1LL << v39;
    v42 = 1LL << v39;
    if ( v58 > 0x40 )
    {
      v345 = v40;
      sub_16A4EF0((__int64)&v379, 0, 0);
      v42 = v41;
      if ( (unsigned int)v380 > 0x40 )
      {
        *(_QWORD *)(v379 + 8LL * (v345 >> 6)) |= v41;
        goto LABEL_31;
      }
    }
    else
    {
      v379 = 0;
    }
    v379 |= v42;
LABEL_31:
    v43 = (__int64 *)sub_16498A0((__int64)v28);
    v44 = sub_159C0E0(v43, (__int64)&v379);
    v45 = v44;
    if ( (unsigned int)v380 > 0x40 && v379 )
    {
      v341 = v44;
      j_j___libc_free_0_0(v379);
      v45 = v341;
    }
    goto LABEL_34;
  }
  return v353;
}
