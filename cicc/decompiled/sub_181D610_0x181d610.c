// Function: sub_181D610
// Address: 0x181d610
//
__int64 *__fastcall sub_181D610(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  char v10; // al
  __int64 *v11; // r15
  unsigned __int64 v12; // rsi
  bool v13; // zf
  __int64 **v14; // rax
  __int64 *result; // rax
  char v16; // dl
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // r10
  __int64 v23; // r9
  unsigned int v24; // edi
  __int64 *v25; // rcx
  __int64 v26; // r12
  void **p_src; // rdi
  unsigned int v28; // eax
  char *v29; // rdx
  __int64 v30; // r9
  __int64 *v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rbx
  unsigned __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rdi
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 *v42; // rax
  char **v43; // r12
  int v44; // eax
  char **v45; // rbx
  char *v46; // rax
  __int64 *v47; // r12
  int v48; // eax
  unsigned __int64 v49; // rbx
  __int64 v50; // rsi
  unsigned __int64 v51; // rax
  __int64 v52; // r10
  __int64 v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rbx
  _QWORD *v56; // rax
  __int64 v57; // r14
  __int64 v58; // rsi
  unsigned __int64 v59; // r14
  __int64 *v60; // rbx
  unsigned __int64 v61; // rdx
  _QWORD *v62; // r14
  __int64 v63; // rax
  __int64 v64; // rax
  _QWORD *v65; // rax
  void *v66; // r11
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  double v71; // xmm4_8
  double v72; // xmm5_8
  __int64 v73; // r12
  unsigned int v74; // r13d
  __int64 j; // rdx
  unsigned int v76; // esi
  _QWORD *v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rbx
  __int64 v80; // rax
  __int64 v81; // rax
  _QWORD *v82; // rdi
  __int64 v83; // rsi
  __int64 *v84; // rbx
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // r13
  __int64 v89; // r8
  int v90; // eax
  _BYTE *v91; // rcx
  _BYTE *v92; // rsi
  __int64 v93; // r14
  __int64 v94; // r12
  __int64 v95; // rbx
  __int64 v96; // rax
  __int64 v97; // r10
  __int64 *v98; // rax
  int v99; // r8d
  __int64 v100; // r9
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // r15
  __int64 *v104; // rax
  __int64 v105; // rcx
  unsigned __int64 v106; // rdx
  __int64 v107; // rdx
  char *v108; // rax
  signed __int64 v109; // rdx
  __int64 v110; // r12
  __int64 v111; // rax
  __int64 v112; // rdi
  __int64 v113; // rdi
  __int64 v114; // rbx
  __int64 *v115; // rax
  __int64 v116; // rcx
  unsigned __int64 v117; // rdx
  __int64 v118; // rdx
  unsigned __int64 v119; // r12
  unsigned __int64 v120; // rax
  unsigned int v121; // r14d
  __int64 v122; // rax
  __int64 v123; // r12
  unsigned int v124; // r15d
  unsigned int v125; // r8d
  __int64 v126; // rax
  int v127; // ecx
  int v128; // ebx
  __int64 v129; // rax
  int v130; // ebx
  __int64 v131; // r13
  __int64 v132; // r12
  char *v133; // rax
  char *v134; // rsi
  __int64 v135; // r14
  __int64 v136; // r13
  __int64 v137; // rax
  void *v138; // r12
  unsigned __int64 v139; // rdx
  __int64 v140; // rcx
  const char *v141; // rsi
  __int64 v142; // rax
  unsigned __int64 v143; // rax
  __int64 *v144; // r15
  unsigned __int64 v145; // rcx
  _BYTE *v146; // rsi
  __int64 v147; // rcx
  __int64 v148; // rcx
  unsigned __int64 v149; // rdx
  __int64 v150; // rcx
  const char *v151; // rsi
  __int64 v152; // r14
  char **v153; // rbx
  __int64 v154; // r12
  __int64 v155; // rax
  char **v156; // rsi
  __int64 v157; // rax
  __int64 v158; // rbx
  __int64 v159; // rax
  __int64 v160; // rdx
  unsigned __int64 v161; // rax
  unsigned __int64 v162; // r12
  int v163; // eax
  __int64 *v164; // r14
  __int64 v165; // rsi
  char *v166; // rax
  char **i; // r14
  char *v168; // rax
  __int64 v169; // r12
  __int64 v170; // rbx
  __int64 v171; // rax
  char *v172; // rdx
  __int64 v173; // rax
  __int64 v174; // rbx
  char *v175; // rax
  __int64 v176; // r13
  __int64 v177; // rax
  __int64 v178; // rdx
  unsigned int v179; // eax
  unsigned int v180; // r12d
  unsigned int v181; // r14d
  char *v182; // rsi
  void *v183; // rax
  __int64 v184; // rax
  __int64 v185; // rbx
  double v186; // xmm4_8
  double v187; // xmm5_8
  int v188; // r9d
  int v189; // r14d
  __int64 *v190; // rax
  __int64 v191; // rax
  char v192; // al
  __int64 v193; // r14
  _QWORD *v194; // rax
  __int64 v195; // rax
  __int64 v196; // r12
  char *v197; // rbx
  __int64 v198; // r10
  _BOOL8 v199; // rsi
  __int64 v200; // rax
  unsigned __int64 v201; // rdx
  bool v202; // al
  __int64 v203; // rax
  int v204; // edx
  __int64 v205; // r12
  __int64 v206; // rbx
  __int64 *v207; // r11
  __int64 v208; // r14
  unsigned int v209; // edx
  _QWORD *v210; // rax
  __int64 v211; // rcx
  int v212; // esi
  __int64 v213; // rdi
  int v214; // ecx
  __int64 v215; // rax
  __int64 v216; // rsi
  unsigned int v217; // eax
  __int64 v218; // r12
  _QWORD *v219; // rdi
  __int64 v220; // rbx
  __int64 v221; // rax
  __int64 v222; // rax
  _QWORD *v223; // rax
  __int64 v224; // rdi
  unsigned __int64 v225; // rax
  unsigned int v226; // edx
  __int64 v227; // rax
  __int64 v228; // rax
  __int64 v229; // rbx
  _QWORD *v230; // rax
  _BYTE *v231; // r12
  unsigned int v232; // ebx
  __int64 *v233; // r14
  unsigned int v234; // r8d
  __int64 v235; // rax
  __int64 v236; // rbx
  __int64 v237; // rax
  int v238; // r10d
  _QWORD *v239; // r9
  unsigned int v240; // edx
  __int64 v241; // rax
  __int64 v242; // rax
  __int64 v243; // r12
  _QWORD *v244; // rax
  _QWORD *v245; // rbx
  __int64 v246; // [rsp+18h] [rbp-368h]
  int v247; // [rsp+24h] [rbp-35Ch]
  __int64 v248; // [rsp+30h] [rbp-350h]
  __int64 v249; // [rsp+30h] [rbp-350h]
  __int64 v250; // [rsp+30h] [rbp-350h]
  __int64 *v251; // [rsp+30h] [rbp-350h]
  __int64 v252; // [rsp+38h] [rbp-348h]
  __int64 *v253; // [rsp+38h] [rbp-348h]
  _QWORD *v254; // [rsp+38h] [rbp-348h]
  __int64 v255; // [rsp+40h] [rbp-340h]
  __int64 v256; // [rsp+40h] [rbp-340h]
  __int64 v257; // [rsp+58h] [rbp-328h]
  __int64 v258; // [rsp+70h] [rbp-310h]
  __int64 v259; // [rsp+78h] [rbp-308h]
  int v260; // [rsp+80h] [rbp-300h]
  __int64 v261; // [rsp+80h] [rbp-300h]
  __int64 v262; // [rsp+80h] [rbp-300h]
  __int64 v263; // [rsp+88h] [rbp-2F8h]
  unsigned int *dest; // [rsp+90h] [rbp-2F0h]
  size_t n; // [rsp+98h] [rbp-2E8h]
  __int64 *v266; // [rsp+A8h] [rbp-2D8h]
  __int64 *v267; // [rsp+A8h] [rbp-2D8h]
  __int64 **v268; // [rsp+A8h] [rbp-2D8h]
  __int64 v269; // [rsp+B0h] [rbp-2D0h]
  char *v270; // [rsp+B0h] [rbp-2D0h]
  __int64 v271; // [rsp+B8h] [rbp-2C8h]
  __int64 *v272; // [rsp+B8h] [rbp-2C8h]
  __int64 v273; // [rsp+B8h] [rbp-2C8h]
  __int64 v274; // [rsp+B8h] [rbp-2C8h]
  unsigned __int64 v275; // [rsp+B8h] [rbp-2C8h]
  __int64 v276; // [rsp+B8h] [rbp-2C8h]
  __int64 v277; // [rsp+B8h] [rbp-2C8h]
  __int64 *v278; // [rsp+B8h] [rbp-2C8h]
  __int64 v279; // [rsp+B8h] [rbp-2C8h]
  __int64 v280; // [rsp+B8h] [rbp-2C8h]
  unsigned int v281; // [rsp+C0h] [rbp-2C0h]
  __int64 *v282; // [rsp+C0h] [rbp-2C0h]
  _BYTE *v283; // [rsp+C0h] [rbp-2C0h]
  __int64 *v284; // [rsp+C0h] [rbp-2C0h]
  unsigned __int64 v285; // [rsp+C8h] [rbp-2B8h]
  __int64 v286; // [rsp+C8h] [rbp-2B8h]
  int v287; // [rsp+C8h] [rbp-2B8h]
  int v288; // [rsp+C8h] [rbp-2B8h]
  __int64 v289; // [rsp+C8h] [rbp-2B8h]
  __int64 v290; // [rsp+D0h] [rbp-2B0h]
  int v291; // [rsp+D0h] [rbp-2B0h]
  __int64 v292; // [rsp+D0h] [rbp-2B0h]
  __int64 v293; // [rsp+D8h] [rbp-2A8h]
  __int64 v294; // [rsp+D8h] [rbp-2A8h]
  __int64 v295; // [rsp+D8h] [rbp-2A8h]
  __int64 *v296; // [rsp+D8h] [rbp-2A8h]
  unsigned int v297; // [rsp+D8h] [rbp-2A8h]
  __int64 v298; // [rsp+D8h] [rbp-2A8h]
  int v299; // [rsp+D8h] [rbp-2A8h]
  unsigned int v300; // [rsp+D8h] [rbp-2A8h]
  __int64 v301; // [rsp+D8h] [rbp-2A8h]
  unsigned int v302; // [rsp+D8h] [rbp-2A8h]
  unsigned __int64 v303; // [rsp+E0h] [rbp-2A0h]
  __int64 v304; // [rsp+E0h] [rbp-2A0h]
  char *v305; // [rsp+E0h] [rbp-2A0h]
  __int64 v306; // [rsp+E0h] [rbp-2A0h]
  __int64 v307; // [rsp+F8h] [rbp-288h] BYREF
  __int64 v308; // [rsp+100h] [rbp-280h] BYREF
  char **v309; // [rsp+108h] [rbp-278h] BYREF
  __int64 v310; // [rsp+110h] [rbp-270h] BYREF
  __int64 v311; // [rsp+118h] [rbp-268h]
  __int64 v312; // [rsp+120h] [rbp-260h]
  __int64 v313; // [rsp+130h] [rbp-250h] BYREF
  __int64 v314; // [rsp+138h] [rbp-248h]
  _QWORD v315[2]; // [rsp+140h] [rbp-240h] BYREF
  __int64 *v316; // [rsp+150h] [rbp-230h] BYREF
  __int64 v317; // [rsp+158h] [rbp-228h]
  __int64 v318; // [rsp+160h] [rbp-220h] BYREF
  void *src; // [rsp+170h] [rbp-210h] BYREF
  __int64 v320; // [rsp+178h] [rbp-208h]
  char v321[32]; // [rsp+180h] [rbp-200h] BYREF
  __int64 v322[10]; // [rsp+1A0h] [rbp-1E0h] BYREF
  char *v323; // [rsp+1F0h] [rbp-190h] BYREF
  __int64 v324; // [rsp+1F8h] [rbp-188h]
  char *v325; // [rsp+200h] [rbp-180h] BYREF
  __int64 v326; // [rsp+208h] [rbp-178h]
  char v327; // [rsp+210h] [rbp-170h] BYREF
  __int64 v328; // [rsp+218h] [rbp-168h]
  __int64 v329; // [rsp+220h] [rbp-160h]
  __int64 v330; // [rsp+228h] [rbp-158h]
  int v331; // [rsp+230h] [rbp-150h]
  __int64 v332; // [rsp+240h] [rbp-140h]
  char v333; // [rsp+248h] [rbp-138h]
  int v334; // [rsp+24Ch] [rbp-134h]
  int v335; // [rsp+250h] [rbp-130h]
  char v336; // [rsp+254h] [rbp-12Ch]
  __int64 v337; // [rsp+258h] [rbp-128h]
  __int64 v338; // [rsp+260h] [rbp-120h]
  __int64 v339; // [rsp+268h] [rbp-118h]
  __int64 v340; // [rsp+270h] [rbp-110h] BYREF
  __int64 v341; // [rsp+278h] [rbp-108h]
  __int64 v342; // [rsp+280h] [rbp-100h]
  unsigned int v343; // [rsp+288h] [rbp-F8h]
  __int64 v344; // [rsp+290h] [rbp-F0h]
  __int64 v345; // [rsp+298h] [rbp-E8h]
  __int64 v346; // [rsp+2A0h] [rbp-E0h]
  int v347; // [rsp+2A8h] [rbp-D8h]
  __int64 v348; // [rsp+2B0h] [rbp-D0h]
  __int64 v349; // [rsp+2B8h] [rbp-C8h]
  __int64 v350; // [rsp+2C0h] [rbp-C0h]
  __int64 v351; // [rsp+2C8h] [rbp-B8h]
  __int64 v352; // [rsp+2D0h] [rbp-B0h]
  __int64 v353; // [rsp+2D8h] [rbp-A8h]
  __int64 v354; // [rsp+2E0h] [rbp-A0h]
  __int64 v355; // [rsp+2E8h] [rbp-98h]
  __int64 v356; // [rsp+2F0h] [rbp-90h]
  __int64 v357; // [rsp+2F8h] [rbp-88h]
  bool v358; // [rsp+300h] [rbp-80h]
  __int64 v359; // [rsp+308h] [rbp-78h]
  __int64 v360; // [rsp+310h] [rbp-70h]
  __int64 v361; // [rsp+318h] [rbp-68h]
  int v362; // [rsp+320h] [rbp-60h]
  __int64 v363; // [rsp+328h] [rbp-58h]
  __int64 v364; // [rsp+330h] [rbp-50h]
  __int64 v365; // [rsp+338h] [rbp-48h]
  int v366; // [rsp+340h] [rbp-40h]

  v10 = a2;
  v11 = a1;
  v307 = a2;
  v12 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = (v10 & 4) == 0;
  v14 = (__int64 **)(v12 - 24);
  if ( v13 )
    v14 = (__int64 **)(v12 - 72);
  result = *v14;
  v16 = *((_BYTE *)result + 16);
  if ( !v16 )
  {
    if ( (*((_BYTE *)result + 33) & 0x20) == 0 )
    {
      if ( *(__int64 **)(*(_QWORD *)*a1 + 376LL) == result )
        return result;
LABEL_6:
      sub_17CE510((__int64)v322, v12, 0, 0, 0);
      v17 = *(_QWORD *)*a1;
      v18 = v307;
      v19 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 24);
      v20 = (v307 >> 2) & 1;
      v303 = v307 & 0xFFFFFFFFFFFFFFF8LL;
      if ( ((v307 >> 2) & 1) == 0 )
        v19 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 72);
      v21 = *(unsigned int *)(v17 + 424);
      if ( (_DWORD)v21 )
      {
        v22 = *v19;
        v23 = *(_QWORD *)(v17 + 408);
        v24 = (v21 - 1) & (((unsigned int)*v19 >> 9) ^ ((unsigned int)*v19 >> 4));
        v25 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( v22 == *v25 )
        {
LABEL_10:
          if ( v25 == (__int64 *)(v23 + 16 * v21) )
            goto LABEL_26;
          p_src = (void **)v17;
          v293 = v25[1];
          v28 = sub_18151B0(v17, v293);
          if ( v28 == 2 )
          {
            v115 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 72);
            if ( (v307 & 4) != 0 )
              v115 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 24);
            if ( *v115 )
            {
              v116 = v115[1];
              v117 = v115[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v117 = v116;
              if ( v116 )
                *(_QWORD *)(v116 + 16) = *(_QWORD *)(v116 + 16) & 3LL | v117;
            }
            *v115 = v293;
            if ( v293 )
            {
              v118 = *(_QWORD *)(v293 + 8);
              v115[1] = v118;
              if ( v118 )
                *(_QWORD *)(v118 + 16) = (unsigned __int64)(v115 + 1) | *(_QWORD *)(v118 + 16) & 3LL;
              v115[2] = (v293 + 8) | v115[2] & 3;
              *(_QWORD *)(v293 + 8) = v115;
            }
            sub_181B3B0(v11, v307 & 0xFFFFFFFFFFFFFFF8LL);
            return (__int64 *)sub_17CD270(v322);
          }
          if ( v28 <= 2 )
          {
            if ( v28 )
            {
              v31 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 72);
              if ( (v307 & 4) != 0 )
                v31 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 24);
              if ( *v31 )
              {
                v32 = v31[1];
                v33 = v31[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v33 = v32;
                if ( v32 )
                  *(_QWORD *)(v32 + 16) = *(_QWORD *)(v32 + 16) & 3LL | v33;
              }
              *v31 = v293;
              if ( v293 )
              {
                v34 = *(_QWORD *)(v293 + 8);
                v31[1] = v34;
                if ( v34 )
                  *(_QWORD *)(v34 + 16) = (unsigned __int64)(v31 + 1) | *(_QWORD *)(v34 + 16) & 3LL;
                v31[2] = (v293 + 8) | v31[2] & 3;
                *(_QWORD *)(v293 + 8) = v31;
              }
              v35 = *v11 + 128;
              v36 = *(_QWORD *)(*(_QWORD *)*v11 + 200LL);
              v323 = (char *)(v307 & 0xFFFFFFFFFFFFFFF8LL);
              sub_176FB00(v35, (__int64 *)&v323)[1] = v36;
            }
            else
            {
              v104 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 72);
              if ( (v307 & 4) != 0 )
                v104 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 24);
              if ( *v104 )
              {
                v105 = v104[1];
                v106 = v104[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v106 = v105;
                if ( v105 )
                  *(_QWORD *)(v105 + 16) = *(_QWORD *)(v105 + 16) & 3LL | v106;
              }
              *v104 = v293;
              if ( v293 )
              {
                v107 = *(_QWORD *)(v293 + 8);
                v104[1] = v107;
                if ( v107 )
                  *(_QWORD *)(v107 + 16) = (unsigned __int64)(v104 + 1) | *(_QWORD *)(v107 + 16) & 3LL;
                v104[2] = (v293 + 8) | v104[2] & 3;
                *(_QWORD *)(v293 + 8) = v104;
              }
              LOWORD(v325) = 257;
              *(_WORD *)v321 = 257;
              v108 = (char *)sub_1649960(v293);
              v110 = sub_15E70A0((__int64)v322, v108, v109, (__int64)&src, 0);
              v111 = sub_1643350((_QWORD *)v322[3]);
              v316 = (__int64 *)sub_159C470(v111, 0, 0);
              v317 = (__int64)v316;
              v112 = *(_QWORD *)(v110 + 24);
              BYTE4(v313) = 0;
              v316 = (__int64 *)sub_15A2E80(v112, v110, &v316, 2u, 1u, (__int64)&v313, 0);
              sub_1285290(
                v322,
                *(_QWORD *)(**(_QWORD **)(*(_QWORD *)*v11 + 352LL) + 24LL),
                *(_QWORD *)(*(_QWORD *)*v11 + 352LL),
                (int)&v316,
                1,
                (__int64)&v323,
                0);
              v113 = *v11 + 128;
              v114 = *(_QWORD *)(*(_QWORD *)*v11 + 200LL);
              v323 = (char *)(v307 & 0xFFFFFFFFFFFFFFF8LL);
              sub_176FB00(v113, (__int64 *)&v323)[1] = v114;
            }
            return (__int64 *)sub_17CD270(v322);
          }
          v13 = v28 == 3;
          v18 = v307;
          if ( !v13 )
          {
            v303 = v307 & 0xFFFFFFFFFFFFFFF8LL;
            v20 = (v307 >> 2) & 1;
            goto LABEL_26;
          }
          v303 = v307 & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_BYTE *)((v307 & 0xFFFFFFFFFFFFFFF8LL) + 16) == 78 )
          {
            v88 = *(_QWORD *)(v293 + 24);
            v89 = *(_QWORD *)*v11;
            src = 0;
            v323 = (char *)&v325;
            v324 = 0x400000000LL;
            v320 = 0;
            *(_QWORD *)v321 = 0;
            v90 = *(_DWORD *)(v88 + 12);
            v292 = v88;
            if ( v90 == 1 )
            {
              if ( !(*(_DWORD *)(v88 + 8) >> 8) )
                goto LABEL_143;
              v236 = *(_QWORD *)(v89 + 184);
              v237 = (unsigned int)v324;
            }
            else
            {
              p_src = (void **)&v323;
              v91 = 0;
              v92 = 0;
              v93 = 8;
              v94 = v89;
              v95 = 8LL * (unsigned int)(v90 - 2) + 16;
              v282 = v11;
              v96 = 0;
              while ( 1 )
              {
                v103 = *(_QWORD *)(*(_QWORD *)(v88 + 16) + v93);
                if ( *(_BYTE *)(v103 + 8) == 15 && (v97 = *(_QWORD *)(v103 + 24), *(_BYTE *)(v97 + 8) == 12) )
                {
                  LODWORD(v316) = v96;
                  if ( v92 == v91 )
                  {
                    v277 = v97;
                    sub_C88AB0((__int64)&src, v92, &v316);
                    v97 = v277;
                  }
                  else
                  {
                    if ( v92 )
                    {
                      *(_DWORD *)v92 = v96;
                      v92 = (_BYTE *)v320;
                    }
                    v320 = (__int64)(v92 + 4);
                  }
                  v98 = (__int64 *)sub_18160E0(v94, v97);
                  v100 = sub_1647190(v98, 0);
                  v101 = (unsigned int)v324;
                  if ( (unsigned int)v324 >= HIDWORD(v324) )
                  {
                    v280 = v100;
                    sub_16CD150((__int64)&v323, &v325, 0, 8, v99, v100);
                    v101 = (unsigned int)v324;
                    v100 = v280;
                  }
                  *(_QWORD *)&v323[8 * v101] = v100;
                  LODWORD(v324) = v324 + 1;
                  p_src = *(void ***)(v94 + 168);
                  v30 = sub_16471D0(p_src, 0);
                  v102 = (unsigned int)v324;
                  if ( (unsigned int)v324 >= HIDWORD(v324) )
                  {
                    p_src = (void **)&v323;
                    v279 = v30;
                    sub_16CD150((__int64)&v323, &v325, 0, 8, v89, v30);
                    v102 = (unsigned int)v324;
                    v30 = v279;
                  }
                  v29 = v323;
                  *(_QWORD *)&v323[8 * v102] = v30;
                  LODWORD(v324) = v324 + 1;
                }
                else
                {
                  LODWORD(v316) = v96;
                  if ( v92 == v91 )
                  {
                    p_src = &src;
                    sub_C88AB0((__int64)&src, v92, &v316);
                    v96 = (unsigned int)v324;
                  }
                  else
                  {
                    if ( v92 )
                    {
                      *(_DWORD *)v92 = v96;
                      v92 = (_BYTE *)v320;
                      v96 = (unsigned int)v324;
                    }
                    v320 = (__int64)(v92 + 4);
                  }
                  if ( HIDWORD(v324) <= (unsigned int)v96 )
                  {
                    p_src = (void **)&v323;
                    sub_16CD150((__int64)&v323, &v325, 0, 8, v89, v30);
                    v96 = (unsigned int)v324;
                  }
                  v29 = v323;
                  *(_QWORD *)&v323[8 * v96] = v103;
                  LODWORD(v324) = v324 + 1;
                }
                v93 += 8;
                if ( v95 == v93 )
                  break;
                v96 = (unsigned int)v324;
                v92 = (_BYTE *)v320;
                v91 = *(_BYTE **)v321;
              }
              v11 = v282;
              v89 = v94;
              v287 = *(_DWORD *)(v88 + 12);
              if ( v287 != 1 )
              {
                v129 = (unsigned int)v324;
                v130 = 0;
                v131 = v94;
                do
                {
                  v132 = *(_QWORD *)(v131 + 176);
                  if ( HIDWORD(v324) <= (unsigned int)v129 )
                  {
                    p_src = (void **)&v323;
                    sub_16CD150((__int64)&v323, &v325, 0, 8, v89, v30);
                    v129 = (unsigned int)v324;
                  }
                  v29 = v323;
                  ++v130;
                  *(_QWORD *)&v323[8 * v129] = v132;
                  v129 = (unsigned int)(v324 + 1);
                  LODWORD(v324) = v324 + 1;
                }
                while ( v287 - 1 != v130 );
                v11 = v282;
                v89 = v131;
              }
              if ( !(*(_DWORD *)(v292 + 8) >> 8) )
              {
LABEL_143:
                if ( *(_BYTE *)(**(_QWORD **)(v292 + 16) + 8LL) )
                {
                  v220 = *(_QWORD *)(v89 + 184);
                  v221 = (unsigned int)v324;
                  if ( (unsigned int)v324 >= HIDWORD(v324) )
                  {
                    p_src = (void **)&v323;
                    sub_16CD150((__int64)&v323, &v325, 0, 8, v89, v30);
                    v221 = (unsigned int)v324;
                  }
                  v29 = v323;
                  *(_QWORD *)&v323[8 * v221] = v220;
                  LODWORD(v324) = v324 + 1;
                }
                v133 = (char *)v320;
                v134 = (char *)src;
                v135 = v320 - (_QWORD)src;
                v136 = v320 - (_QWORD)src;
                if ( (void *)v320 == src )
                {
                  n = 0;
                  v138 = 0;
                }
                else
                {
                  if ( (unsigned __int64)v135 > 0x7FFFFFFFFFFFFFFCLL )
                    goto LABEL_314;
                  v137 = sub_22077B0(v320 - (_QWORD)src);
                  v134 = (char *)src;
                  v138 = (void *)v137;
                  v133 = (char *)v320;
                  v135 = v320 - (_QWORD)src;
                  n = v320 - (_QWORD)src;
                }
                if ( v134 != v133 )
                  memmove(v138, v134, n);
                v134 = v323;
                p_src = **(void ****)(v292 + 16);
                v263 = sub_1644EA0((__int64 *)p_src, v323, (unsigned int)v324, *(_DWORD *)(v292 + 8) >> 8 != 0);
                v257 = v135 >> 2;
                if ( !v135 )
                {
                  dest = 0;
LABEL_152:
                  if ( n )
                  {
                    memcpy(dest, v138, n);
                  }
                  else if ( !v138 )
                  {
LABEL_155:
                    if ( src )
                      j_j___libc_free_0(src, *(_QWORD *)v321 - (_QWORD)src);
                    if ( v323 != (char *)&v325 )
                      _libc_free((unsigned __int64)v323);
                    v318 = 0x5F777366645F5FLL;
                    v316 = &v318;
                    v317 = 7;
                    v141 = sub_1649960(v293);
                    if ( v139 > 0x3FFFFFFFFFFFFFFFLL - v317 )
LABEL_317:
                      sub_4262D8((__int64)"basic_string::append");
                    sub_2241490(&v316, v141, v139, v140);
                    v142 = sub_1632190(*(_QWORD *)(*(_QWORD *)*v11 + 160LL), (__int64)v316, v317, v263);
                    v258 = v142;
                    if ( !*(_BYTE *)(v142 + 16) )
                    {
                      sub_15E4330(v142, v293);
                      if ( *(_BYTE *)(**(_QWORD **)(v292 + 16) + 8LL) )
                        sub_15E0EF0(v258, -1, (_QWORD *)(*(_QWORD *)*v11 + 432LL));
                    }
                    v310 = 0;
                    v311 = 0;
                    v312 = 0;
                    v143 = sub_165B7C0(&v307);
                    v267 = (__int64 *)v143;
                    v288 = *(_DWORD *)(v292 + 12) - 1;
                    if ( *(_DWORD *)(v292 + 12) == 1 )
                    {
LABEL_182:
                      if ( *(_DWORD *)(v292 + 8) >> 8 )
                      {
                        v225 = sub_1389B50(&v307);
                        v278 = sub_1645D80(
                                 *(__int64 **)(*(_QWORD *)*v11 + 176LL),
                                 -1431655765
                               * (unsigned int)((__int64)(v225
                                                        - ((v307 & 0xFFFFFFFFFFFFFFF8LL)
                                                         - 24LL
                                                         * (*(_DWORD *)((v307 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3)
                               - *(_DWORD *)(v292 + 12)
                               + 1);
                        v226 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*v11 + 8) + 40LL)) + 4);
                        v323 = "labelva";
                        v227 = *v11;
                        LOWORD(v325) = 259;
                        v300 = v226;
                        v228 = *(_QWORD *)(*(_QWORD *)(v227 + 8) + 80LL);
                        if ( !v228 )
                          goto LABEL_318;
                        v229 = *(_QWORD *)(v228 + 24);
                        if ( v229 )
                          v229 -= 24;
                        v230 = sub_1648A60(64, 1u);
                        v231 = v230;
                        if ( v230 )
                          sub_15F8BE0((__int64)v230, v278, v300, (__int64)&v323, v229);
                        v232 = 0;
                        v233 = v267;
                        while ( v233 != (__int64 *)sub_1389B50(&v307) )
                        {
                          v234 = v232;
                          v233 += 3;
                          ++v232;
                          LOWORD(v325) = 257;
                          v301 = sub_1286300(v322, (__int64)v278, v231, 0, v234, (__int64)&v323);
                          v235 = sub_1819D40((_QWORD *)*v11, *(v233 - 3));
                          sub_12A8F50(v322, v235, v301, 0);
                        }
                        LOWORD(v325) = 257;
                        src = (void *)sub_1286300(v322, (__int64)v278, v231, 0, 0, (__int64)&v323);
                        sub_15E88C0((__int64)&v310, &src);
                      }
                      if ( *(_BYTE *)(**(_QWORD **)(v292 + 16) + 8LL) )
                      {
                        v166 = *(char **)(*v11 + 120);
                        if ( !v166 )
                        {
                          v240 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*v11 + 8) + 40LL)) + 4);
                          v323 = "labelreturn";
                          v241 = *v11;
                          LOWORD(v325) = 259;
                          v302 = v240;
                          v242 = *(_QWORD *)(*(_QWORD *)(v241 + 8) + 80LL);
                          if ( !v242 )
                            goto LABEL_318;
                          v243 = *(_QWORD *)(v242 + 24);
                          if ( v243 )
                            v243 -= 24;
                          v244 = sub_1648A60(64, 1u);
                          v245 = v244;
                          if ( v244 )
                            sub_15F8BE0(
                              (__int64)v244,
                              *(_QWORD **)(*(_QWORD *)*v11 + 176LL),
                              v302,
                              (__int64)&v323,
                              v243);
                          *(_QWORD *)(*v11 + 120) = v245;
                          v166 = *(char **)(*v11 + 120);
                        }
                        v323 = v166;
                        sub_15E88C0((__int64)&v310, &v323);
                      }
                      for ( i = (char **)(sub_165B7C0(&v307) + 24LL * (unsigned int)(*(_DWORD *)(v292 + 12) - 1));
                            i != (char **)sub_1389B50(&v307);
                            i += 3 )
                      {
                        v168 = *i;
                        v323 = v168;
                        sub_15E88C0((__int64)&v310, &v323);
                      }
                      LOWORD(v325) = 257;
                      v169 = sub_1285290(
                               v322,
                               *(_QWORD *)(*(_QWORD *)v258 + 24LL),
                               v258,
                               v310,
                               (v311 - v310) >> 3,
                               (__int64)&v323,
                               0);
                      *(_WORD *)(v169 + 18) = *(_WORD *)(v169 + 18) & 0x8000
                                            | *(_WORD *)(v169 + 18) & 3
                                            | (4 * ((*(_WORD *)(v303 + 18) >> 2) & 0xDFFF));
                      v170 = *(_QWORD *)(v303 + 56);
                      v171 = sub_16498A0(v303);
                      v313 = v170;
                      v172 = 0;
                      v284 = (__int64 *)v171;
                      LODWORD(v171) = *(_DWORD *)(v263 + 12);
                      v323 = 0;
                      v324 = 0;
                      v325 = 0;
                      v173 = (unsigned int)(v171 - 1);
                      v174 = 8 * v173;
                      if ( v173 )
                      {
                        v175 = (char *)sub_22077B0(8 * v173);
                        v172 = &v175[v174];
                        v323 = v175;
                        v325 = &v175[v174];
                        do
                        {
                          if ( v175 )
                            *(_QWORD *)v175 = 0;
                          v175 += 8;
                        }
                        while ( v175 != v172 );
                      }
                      v324 = (__int64)v172;
                      if ( (_DWORD)v257 )
                      {
                        v176 = 0;
                        do
                        {
                          v177 = sub_1560230(&v313, v176);
                          v178 = dest[v176++];
                          *(_QWORD *)&v323[8 * v178] = v177;
                        }
                        while ( (unsigned int)v257 != v176 );
                      }
                      v297 = *(_DWORD *)(v292 + 12) - 1;
                      v179 = sub_15601D0((__int64)&v313);
                      if ( v297 < v179 )
                      {
                        v274 = v169;
                        v180 = v179;
                        v181 = v297;
                        do
                        {
                          v183 = (void *)sub_1560230(&v313, v181);
                          v182 = (char *)v324;
                          src = v183;
                          if ( (char *)v324 == v325 )
                          {
                            sub_17401C0(&v323, (char *)v324, &src);
                          }
                          else
                          {
                            if ( v324 )
                            {
                              *(_QWORD *)v324 = v183;
                              v182 = (char *)v324;
                            }
                            v324 = (__int64)(v182 + 8);
                          }
                          ++v181;
                        }
                        while ( v180 > v181 );
                        v169 = v274;
                      }
                      v270 = v323;
                      v275 = (v324 - (__int64)v323) >> 3;
                      v298 = sub_1560240(&v313);
                      v184 = sub_1560250(&v313);
                      v185 = sub_155FDB0(v284, v184, v298, v270, v275);
                      if ( v323 )
                        j_j___libc_free_0(v323, v325 - v323);
                      *(_QWORD *)(v169 + 56) = v185;
                      if ( *(_DWORD *)(v292 + 12) != 1 )
                      {
                        v188 = ~v288;
                        v189 = v288 + 1;
                        do
                        {
                          if ( *(_QWORD *)(*(_QWORD *)*v11 + 176LL) == **(_QWORD **)(v169
                                                                                   + 24
                                                                                   * ((unsigned int)(v189 - 1)
                                                                                    - (unsigned __int64)(*(_DWORD *)(v169 + 20) & 0xFFFFFFF))) )
                          {
                            v299 = v188;
                            v323 = *(char **)(v169 + 56);
                            v190 = (__int64 *)sub_16498A0(v169);
                            v191 = sub_1563AB0((__int64 *)&v323, v190, v189, 58);
                            v188 = v299;
                            *(_QWORD *)(v169 + 56) = v191;
                          }
                          ++v189;
                        }
                        while ( *(_DWORD *)(v292 + 12) - 1 > (unsigned int)(v188 + v189) );
                      }
                      if ( *(_BYTE *)(**(_QWORD **)(v292 + 16) + 8LL) )
                      {
                        v222 = *v11;
                        LOWORD(v325) = 257;
                        v223 = sub_156E5B0(v322, *(_QWORD *)(v222 + 120), (__int64)&v323);
                        v224 = *v11;
                        v323 = (char *)v169;
                        sub_176FB00(v224 + 128, (__int64 *)&v323)[1] = v223;
                      }
                      sub_164D160(v303, v169, a3, a4, a5, a6, v186, v187, a9, a10);
                      sub_15F20C0((_QWORD *)v303);
                      if ( v310 )
                        j_j___libc_free_0(v310, v312 - v310);
                      sub_2240A30(&v316);
                      if ( dest )
                        j_j___libc_free_0(dest, n);
                      return (__int64 *)sub_17CD270(v322);
                    }
                    v268 = (__int64 **)v11;
                    v144 = (__int64 *)v143;
                    while ( 1 )
                    {
                      v160 = *(_QWORD *)*v144;
                      if ( *(_BYTE *)(v160 + 8) != 15 || (v273 = *(_QWORD *)(v160 + 24), *(_BYTE *)(v273 + 8) != 12) )
                      {
                        v323 = (char *)*v144;
                        sub_15E88C0((__int64)&v310, &v323);
                        goto LABEL_173;
                      }
                      strcpy(v321, "dfst");
                      src = v321;
                      v320 = 4;
                      v145 = (unsigned int)(*(_DWORD *)(v292 + 12) + ~v288);
                      if ( *(_DWORD *)(v292 + 12) + ~v288 )
                      {
                        v146 = (char *)v315 + 5;
                        do
                        {
                          *--v146 = v145 % 0xA + 48;
                          v161 = v145;
                          v145 /= 0xAu;
                        }
                        while ( v161 > 9 );
                      }
                      else
                      {
                        BYTE4(v315[0]) = 48;
                        v146 = (char *)v315 + 4;
                      }
                      v323 = (char *)&v325;
                      sub_1814C60((__int64 *)&v323, v146, (__int64)v315 + 5);
                      sub_2241490(&src, v323, v324, v147);
                      sub_2240A30(&v323);
                      if ( v320 == 0x3FFFFFFFFFFFFFFFLL )
                        goto LABEL_317;
                      sub_2241490(&src, "$", 1, v148);
                      v151 = sub_1649960(v293);
                      if ( v149 > 0x3FFFFFFFFFFFFFFFLL - v320 )
                        goto LABEL_317;
                      sub_2241490(&src, v151, v149, v150);
                      v152 = v320;
                      v153 = (char **)src;
                      v154 = **v268;
                      v155 = sub_18160E0(v154, v273);
                      v156 = v153;
                      v157 = sub_1632190(*(_QWORD *)(v154 + 160), (__int64)v153, v152, v155);
                      v158 = v157;
                      if ( !*(_BYTE *)(v157 + 16) && sub_15E4F60(v157) )
                        break;
LABEL_172:
                      v323 = (char *)v158;
                      sub_15E88C0((__int64)&v310, &v323);
                      LOWORD(v325) = 257;
                      v159 = sub_16471D0(*(_QWORD **)(**v268 + 168), 0);
                      v313 = sub_12AA3B0(v322, 0x2Fu, *v144, v159, (__int64)&v323);
                      sub_15E88C0((__int64)&v310, &v313);
                      sub_2240A30(&src);
LABEL_173:
                      v144 += 3;
                      if ( !--v288 )
                      {
                        v11 = (__int64 *)v268;
                        v162 = sub_165B7C0(&v307);
                        v288 = (v311 - v310) >> 3;
                        v163 = *(_DWORD *)(v292 + 12);
                        if ( v163 == 1 )
                        {
                          v267 = (__int64 *)v162;
                        }
                        else
                        {
                          v164 = (__int64 *)v162;
                          v267 = (__int64 *)(v162 + 24LL * (unsigned int)(v163 - 1));
                          do
                          {
                            v165 = *v164;
                            v164 += 3;
                            v323 = (char *)sub_1819D40((_QWORD *)*v11, v165);
                            sub_15E88C0((__int64)&v310, &v323);
                          }
                          while ( v164 != v267 );
                        }
                        goto LABEL_182;
                      }
                    }
                    v192 = *(_BYTE *)(v158 + 32) & 0xF0 | 3;
                    *(_BYTE *)(v158 + 32) = v192;
                    if ( (v192 & 0x30) != 0 )
                      *(_BYTE *)(v158 + 33) |= 0x40u;
                    v323 = "entry";
                    LOWORD(v325) = 259;
                    v193 = *(_QWORD *)(v154 + 168);
                    v194 = (_QWORD *)sub_22077B0(64);
                    v255 = (__int64)v194;
                    if ( v194 )
                    {
                      v156 = (char **)v193;
                      sub_157FB60(v194, v193, (__int64)&v323, v158, 0);
                    }
                    v313 = 0;
                    v314 = 0;
                    v315[0] = 0;
                    if ( (*(_BYTE *)(v158 + 18) & 1) != 0 )
                      sub_15E08E0(v158, (__int64)v156);
                    v195 = *(_QWORD *)(v158 + 88);
                    v259 = v195 + 40;
                    v260 = *(_DWORD *)(v273 + 12);
                    if ( v260 != 1 )
                    {
                      v252 = v154;
                      v248 = v158;
                      v196 = v195 + 40LL * (unsigned int)(v260 - 2) + 80;
                      v197 = (char *)(v195 + 40);
                      do
                      {
                        v323 = v197;
                        v156 = &v323;
                        v197 += 40;
                        sub_15E88C0((__int64)&v313, &v323);
                      }
                      while ( v197 != (char *)v196 );
                      v154 = v252;
                      v158 = v248;
                      v259 += 40LL * (unsigned int)(v260 - 1);
                    }
                    LOWORD(v325) = 257;
                    v253 = (__int64 *)v313;
                    v198 = (v314 - v313) >> 3;
                    if ( (*(_BYTE *)(v158 + 18) & 1) != 0 )
                    {
                      v262 = (v314 - v313) >> 3;
                      sub_15E08E0(v158, (__int64)v156);
                      v198 = v262;
                    }
                    v249 = v198;
                    v247 = v198 + 1;
                    v246 = *(_QWORD *)(v158 + 88);
                    v261 = (__int64)sub_1648A60(72, (int)v198 + 1);
                    if ( v261 )
                    {
                      sub_15F1F50(
                        v261,
                        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v246 + 24LL) + 16LL),
                        54,
                        v261 - 24 * v249 - 24,
                        v247,
                        v255);
                      *(_QWORD *)(v261 + 56) = 0;
                      sub_15F5B40(v261, *(_QWORD *)(*(_QWORD *)v246 + 24LL), v246, v253, v249, (__int64)&v323, 0, 0);
                    }
                    v250 = *(_QWORD *)(v154 + 168);
                    if ( *(_BYTE *)(**(_QWORD **)(v273 + 16) + 8LL) )
                    {
                      v199 = v261 != 0;
                      v254 = sub_1648A60(56, v199);
                      if ( v254 )
                      {
                        v199 = v250;
                        sub_15F7090((__int64)v254, v250, v261, v255);
                      }
                    }
                    else
                    {
                      v199 = 0;
                      v254 = sub_1648A60(56, 0);
                      if ( v254 )
                      {
                        v199 = v250;
                        sub_15F7190((__int64)v254, v250, v255);
                      }
                    }
                    v323 = (char *)v154;
                    v325 = &v327;
                    v326 = 0x100000000LL;
                    v324 = v158;
                    v328 = 0;
                    v329 = 0;
                    v335 = (unsigned __int8)byte_4FA91C0 ^ 1;
                    v330 = 0;
                    v331 = 0;
                    v333 = 0;
                    v334 = 0;
                    v336 = 1;
                    v337 = 0;
                    v338 = 0;
                    v339 = 0;
                    v340 = 0;
                    v341 = 0;
                    v342 = 0;
                    v343 = 0;
                    v344 = 0;
                    v345 = 0;
                    v346 = 0;
                    v347 = 0;
                    v348 = 0;
                    v349 = 0;
                    v350 = 0;
                    v351 = 0;
                    v352 = 0;
                    v353 = 0;
                    v354 = 0;
                    v355 = 0;
                    v356 = 0;
                    v357 = 0;
                    v359 = 0;
                    v360 = 0;
                    v361 = 0;
                    v362 = 0;
                    v363 = 0;
                    v364 = 0;
                    v365 = 0;
                    v366 = 0;
                    v332 = v158;
                    sub_15D3930((__int64)&v325);
                    v200 = *(_QWORD *)(v158 + 80);
                    if ( v158 + 72 == v200 )
                    {
                      v202 = 0;
                    }
                    else
                    {
                      v201 = 0;
                      do
                      {
                        v200 = *(_QWORD *)(v200 + 8);
                        ++v201;
                      }
                      while ( v158 + 72 != v200 );
                      v202 = v201 > 0x3E8;
                    }
                    v358 = v202;
                    if ( (*(_BYTE *)(v158 + 18) & 1) != 0 )
                      sub_15E08E0(v158, v199);
                    v203 = *(_QWORD *)(v158 + 88);
                    v204 = *(_DWORD *)(v273 + 12);
                    v205 = v203 + 40;
                    if ( v204 == 1 )
                    {
LABEL_260:
                      v309 = &v323;
                      sub_181D610(&v309, v261 | 4);
                      if ( *(_BYTE *)(**(_QWORD **)(v273 + 16) + 8LL) )
                      {
                        v216 = 0;
                        v217 = *((_DWORD *)v254 + 5) & 0xFFFFFFF;
                        if ( v217 )
                          v216 = v254[-3 * v217];
                        v218 = sub_1819D40(&v323, v216);
                        if ( (*(_BYTE *)(v158 + 18) & 1) != 0 )
                          sub_15E08E0(v158, v216);
                        v276 = *(_QWORD *)(v158 + 88) + 40LL * *(_QWORD *)(v158 + 96) - 40;
                        v219 = sub_1648A60(64, 2u);
                        if ( v219 )
                          sub_15F9660((__int64)v219, v218, v276, (__int64)v254);
                      }
                      sub_1815030((__int64)&v323);
                      if ( v313 )
                        j_j___libc_free_0(v313, v315[0] - v313);
                      goto LABEL_172;
                    }
                    v256 = v158;
                    v206 = v259;
                    v207 = v144;
                    v208 = v203 + 40LL * (unsigned int)(v204 - 2) + 80;
                    while ( 1 )
                    {
                      v212 = v343;
                      v308 = v205;
                      if ( !v343 )
                        break;
                      v209 = (v343 - 1) & (((unsigned int)v205 >> 9) ^ ((unsigned int)v205 >> 4));
                      v210 = (_QWORD *)(v341 + 16LL * v209);
                      v211 = *v210;
                      if ( v205 != *v210 )
                      {
                        v238 = 1;
                        v239 = 0;
                        while ( v211 != -8 )
                        {
                          if ( !v239 && v211 == -16 )
                            v239 = v210;
                          v209 = (v343 - 1) & (v238 + v209);
                          v210 = (_QWORD *)(v341 + 16LL * v209);
                          v211 = *v210;
                          if ( v205 == *v210 )
                            goto LABEL_249;
                          ++v238;
                        }
                        if ( v239 )
                          v210 = v239;
                        ++v340;
                        v214 = v342 + 1;
                        if ( 4 * ((int)v342 + 1) < 3 * v343 )
                        {
                          v213 = v205;
                          if ( v343 - HIDWORD(v342) - v214 <= v343 >> 3 )
                          {
                            v251 = v207;
LABEL_253:
                            sub_176F940((__int64)&v340, v212);
                            sub_176A9A0((__int64)&v340, &v308, &v309);
                            v210 = v309;
                            v213 = v308;
                            v207 = v251;
                            v214 = v342 + 1;
                          }
                          LODWORD(v342) = v214;
                          if ( *v210 != -8 )
                            --HIDWORD(v342);
                          *v210 = v213;
                          v210[1] = 0;
                          goto LABEL_249;
                        }
LABEL_252:
                        v251 = v207;
                        v212 = 2 * v343;
                        goto LABEL_253;
                      }
LABEL_249:
                      v205 += 40;
                      v210[1] = v206;
                      v206 += 40;
                      if ( v205 == v208 )
                      {
                        v158 = v256;
                        v144 = v207;
                        goto LABEL_260;
                      }
                    }
                    ++v340;
                    goto LABEL_252;
                  }
                  j_j___libc_free_0(v138, v136);
                  goto LABEL_155;
                }
                if ( n <= 0x7FFFFFFFFFFFFFFCLL )
                {
                  dest = (unsigned int *)sub_22077B0(n);
                  goto LABEL_152;
                }
LABEL_314:
                sub_4261EA(p_src, v134, v29);
              }
              v236 = *(_QWORD *)(v89 + 184);
              v237 = (unsigned int)v324;
              if ( (unsigned int)v324 >= HIDWORD(v324) )
              {
                p_src = (void **)&v323;
                v289 = v89;
                sub_16CD150((__int64)&v323, &v325, 0, 8, v89, v30);
                v237 = (unsigned int)v324;
                v89 = v289;
              }
            }
            v29 = v323;
            *(_QWORD *)&v323[8 * v237] = v236;
            LODWORD(v324) = v324 + 1;
            goto LABEL_143;
          }
          v20 = (v307 >> 2) & 1;
        }
        else
        {
          v127 = 1;
          while ( v26 != -8 )
          {
            v128 = v127 + 1;
            v24 = (v21 - 1) & (v127 + v24);
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( v22 == *v25 )
              goto LABEL_10;
            v127 = v128;
          }
        }
      }
LABEL_26:
      v37 = v303 - 72;
      if ( (_BYTE)v20 )
        v37 = v303 - 24;
      v38 = **(_QWORD **)(**(_QWORD **)v37 + 16LL);
      v294 = v38;
      if ( byte_4FA91C0 )
      {
        if ( !*(_BYTE *)(*(_QWORD *)v303 + 8LL) )
        {
          v290 = 0;
          v39 = *(_QWORD *)*v11;
LABEL_31:
          v40 = (__int64 *)sub_1816300(v39, v294);
          LOWORD(v325) = 257;
          v41 = sub_1646BA0(v40, 0);
          if ( (v307 & 4) != 0 )
            v42 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 24);
          else
            v42 = (__int64 *)((v307 & 0xFFFFFFFFFFFFFFF8LL) - 72);
          v269 = sub_12AA3B0(v322, 0x2Fu, *v42, v41, (__int64)&v323);
          v316 = 0;
          v317 = 0;
          v318 = 0;
          v43 = (char **)sub_165B7C0(&v307);
          v285 = sub_1389B50(&v307);
          v44 = *(_DWORD *)(v294 + 12);
          if ( v44 == 1 )
          {
            v49 = sub_165B7C0(&v307);
          }
          else
          {
            v45 = &v43[3 * (unsigned int)(v44 - 1)];
            do
            {
              v46 = *v43;
              v43 += 3;
              v323 = v46;
              sub_15E88C0((__int64)&v316, &v323);
            }
            while ( v43 != v45 );
            v47 = (__int64 *)sub_165B7C0(&v307);
            v48 = *(_DWORD *)(v294 + 12);
            if ( v48 == 1 )
            {
              v49 = (unsigned __int64)v47;
            }
            else
            {
              v49 = (unsigned __int64)&v47[3 * (unsigned int)(v48 - 1)];
              do
              {
                v50 = *v47;
                v47 += 3;
                v323 = (char *)sub_1819D40((_QWORD *)*v11, v50);
                sub_15E88C0((__int64)&v316, &v323);
              }
              while ( (__int64 *)v49 != v47 );
            }
          }
          if ( !(*(_DWORD *)(v294 + 8) >> 8) )
          {
LABEL_40:
            v51 = v307 & 0xFFFFFFFFFFFFFFF8LL;
            v52 = (v317 - (__int64)v316) >> 3;
            if ( *(_BYTE *)((v307 & 0xFFFFFFFFFFFFFFF8LL) + 16) == 29 )
            {
              *(_WORD *)v321 = 257;
              v53 = *(_QWORD *)(v51 - 24);
              v54 = *(_QWORD *)(v51 - 48);
              LOWORD(v325) = 257;
              v286 = v53;
              v295 = v54;
              v266 = v316;
              v271 = (v317 - (__int64)v316) >> 3;
              v281 = v52 + 3;
              v55 = *(_QWORD *)(*(_QWORD *)v269 + 24LL);
              v56 = sub_1648AB0(72, (int)v52 + 3, 0);
              v57 = (__int64)v56;
              if ( v56 )
              {
                sub_15F1EA0((__int64)v56, **(_QWORD **)(v55 + 16), 5, (__int64)&v56[-3 * v281], v281, 0);
                *(_QWORD *)(v57 + 56) = 0;
                sub_15F6500(v57, v55, v269, v295, v286, (__int64)&v323, v266, v271, 0, 0);
              }
              sub_18149C0(v57, (__int64 *)&src, v322[1], (__int64 *)v322[2]);
              v58 = v57;
              v59 = v57 & 0xFFFFFFFFFFFFFFFBLL;
              sub_12A86E0(v322, v58);
            }
            else
            {
              LOWORD(v325) = 257;
              v59 = sub_1285290(
                      v322,
                      *(_QWORD *)(*(_QWORD *)v269 + 24LL),
                      v269,
                      (int)v316,
                      (v317 - (__int64)v316) >> 3,
                      (__int64)&v323,
                      0)
                  | 4;
            }
            v60 = (__int64 *)(v59 & 0xFFFFFFFFFFFFFFF8LL);
            v61 = v307 & 0xFFFFFFFFFFFFFFF8LL;
            *(_WORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 18) = *(_WORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x8000
                                                          | *(_WORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 3
                                                          | (4
                                                           * ((*(_WORD *)((v307 & 0xFFFFFFFFFFFFFFF8LL) + 18) >> 2)
                                                            & 0xDFFF));
            src = *(void **)(v61 + 56);
            sub_1560E30((__int64)&v323, *(_QWORD *)(v59 & 0xFFFFFFFFFFFFFFF8LL));
            *(_QWORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 56) = sub_1563330(
                                                               (__int64 *)&src,
                                                               *(__int64 **)(*(_QWORD *)*v11 + 168LL),
                                                               0,
                                                               &v323);
            sub_1814D10((_QWORD *)v326);
            if ( v290 )
            {
              LODWORD(src) = 0;
              LOWORD(v325) = 257;
              v62 = sub_1648A60(88, 1u);
              if ( v62 )
              {
                v63 = sub_15FB2A0(*v60, (unsigned int *)&src, 1);
                sub_15F1EA0((__int64)v62, v63, 62, (__int64)(v62 - 3), 1, v290);
                sub_1593B40(v62 - 3, (__int64)v60);
                v62[7] = v62 + 9;
                v62[8] = 0x400000000LL;
                sub_15FB110((__int64)v62, &src, 1, (__int64)&v323);
              }
              v64 = *v11;
              src = v62;
              sub_181D4C0((__int64)&v323, v64 + 216, (__int64 *)&src);
              LOWORD(v325) = 257;
              LODWORD(src) = 1;
              v65 = sub_1648A60(88, 1u);
              v66 = v65;
              if ( v65 )
              {
                v304 = (__int64)v65;
                v67 = sub_15FB2A0(*v60, (unsigned int *)&src, 1);
                sub_15F1EA0(v304, v67, 62, v304 - 24, 1, v290);
                sub_1593B40((_QWORD *)(v304 - 24), (__int64)v60);
                *(_QWORD *)(v304 + 56) = v304 + 72;
                *(_QWORD *)(v304 + 64) = 0x400000000LL;
                sub_15FB110(v304, &src, 1, (__int64)&v323);
                v66 = (void *)v304;
              }
              v68 = *v11;
              src = v66;
              v305 = (char *)v66;
              sub_181D4C0((__int64)&v323, v68 + 216, (__int64 *)&src);
              v69 = *v11;
              v323 = (char *)v62;
              sub_176FB00(v69 + 128, (__int64 *)&v323)[1] = v305;
              v70 = *v11;
              v323 = v305;
              sub_15E88C0(v70 + 248, &v323);
              sub_164D160(v307 & 0xFFFFFFFFFFFFFFF8LL, (__int64)v62, a3, a4, a5, a6, v71, v72, a9, a10);
            }
            sub_15F20C0((_QWORD *)(v307 & 0xFFFFFFFFFFFFFFF8LL));
            if ( v316 )
              j_j___libc_free_0(v316, v318 - (_QWORD)v316);
            return (__int64 *)sub_17CD270(v322);
          }
          v119 = sub_1389B50(&v307);
          v120 = sub_165B7C0(&v307);
          v296 = sub_1645D80(
                   *(__int64 **)(*(_QWORD *)*v11 + 176LL),
                   -1431655765 * (unsigned int)((__int64)(v119 - v120) >> 3) - *(_DWORD *)(v294 + 12) + 1);
          v121 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*v11 + 8) + 40LL)) + 4);
          LOWORD(v325) = 257;
          v122 = *(_QWORD *)(*(_QWORD *)(*v11 + 8) + 80LL);
          if ( v122 )
          {
            v123 = *(_QWORD *)(v122 + 24);
            if ( v123 )
              v123 -= 24;
            v283 = sub_1648A60(64, 1u);
            if ( v283 )
              sub_15F8BE0((__int64)v283, v296, v121, (__int64)&v323, v123);
            LOWORD(v325) = 257;
            src = (void *)sub_18174F0((__int64)v322, (__int64)v296, v283, 0, 0, (__int64 *)&v323);
            sub_15E88C0((__int64)&v316, &src);
            if ( v285 != v49 )
            {
              v272 = v11;
              v124 = 0;
              do
              {
                v125 = v124;
                v49 += 24LL;
                ++v124;
                LOWORD(v325) = 257;
                v306 = sub_18174F0((__int64)v322, (__int64)v296, v283, 0, v125, (__int64 *)&v323);
                v126 = sub_1819D40((_QWORD *)*v272, *(_QWORD *)(v49 - 24));
                sub_12A8F50(v322, v126, v306, 0);
                v323 = *(char **)(v49 - 24);
                sub_15E88C0((__int64)&v316, &v323);
              }
              while ( v285 != v49 );
              v11 = v272;
            }
            goto LABEL_40;
          }
LABEL_318:
          BUG();
        }
      }
      else
      {
        v291 = *(_DWORD *)(v38 + 12);
        if ( v291 == 1 )
        {
          if ( !*(_BYTE *)(*(_QWORD *)v303 + 8LL) )
            return (__int64 *)sub_17CD270(v322);
        }
        else
        {
          v73 = 0;
          v74 = 0;
          for ( j = v18; ; j = v307 )
          {
            v76 = v74++;
            v77 = sub_1817AE0((_QWORD *)*v11, v76, j & 0xFFFFFFFFFFFFFFF8LL);
            v78 = v73;
            v73 += 24;
            v79 = (__int64)v77;
            v80 = sub_1819D40(
                    (_QWORD *)*v11,
                    *(_QWORD *)((v307 & 0xFFFFFFFFFFFFFFF8LL)
                              + v78
                              - 24LL * (*(_DWORD *)((v307 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
            sub_12A8F50(v322, v80, v79, 0);
            if ( v74 == v291 - 1 )
              break;
          }
          v290 = 0;
          v303 = v307 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !*(_BYTE *)(*(_QWORD *)(v307 & 0xFFFFFFFFFFFFFFF8LL) + 8LL) )
          {
LABEL_69:
            if ( byte_4FA91C0 )
              goto LABEL_70;
            return (__int64 *)sub_17CD270(v322);
          }
        }
      }
      if ( *(_BYTE *)(v303 + 16) == 29 )
      {
        if ( sub_157F0B0(*(_QWORD *)(v303 - 48)) )
        {
          v81 = *(_QWORD *)(*(_QWORD *)(v303 - 48) + 48LL);
          v290 = v81 - 24;
          if ( v81 )
          {
LABEL_66:
            if ( byte_4FA91C0 )
            {
LABEL_70:
              v39 = *(_QWORD *)*v11;
              goto LABEL_31;
            }
            sub_17CE510((__int64)&v323, v290, 0, 0, 0);
            v82 = (_QWORD *)*v11;
            *(_WORD *)v321 = 257;
            v83 = v82[14];
            if ( !v83 )
            {
              v83 = *(_QWORD *)(*v82 + 232LL);
              if ( v83 )
                v82[14] = v83;
              else
                v83 = sub_1817880(v82);
            }
            v84 = sub_156E5B0((__int64 *)&v323, v83, (__int64)&src);
            v85 = *v11;
            v316 = v84;
            sub_181D4C0((__int64)&src, v85 + 216, (__int64 *)&v316);
            src = (void *)(v307 & 0xFFFFFFFFFFFFFFF8LL);
            sub_176FB00(*v11 + 128, (__int64 *)&src)[1] = v84;
            v86 = *v11;
            src = v84;
            sub_15E88C0(v86 + 248, &src);
            sub_17CD270((__int64 *)&v323);
            goto LABEL_69;
          }
        }
        else
        {
          v215 = *(_QWORD *)(sub_1AA91E0(*(_QWORD *)(v303 + 40), *(_QWORD *)(v303 - 48), *v11 + 16, 0) + 48);
          if ( v215 )
          {
            v290 = v215 - 24;
            goto LABEL_66;
          }
        }
      }
      else
      {
        v87 = *(_QWORD *)(v303 + 32);
        if ( v87 != *(_QWORD *)(v303 + 40) + 40LL && v87 )
        {
          v290 = v87 - 24;
          goto LABEL_66;
        }
      }
      v290 = 0;
      goto LABEL_66;
    }
    return sub_181B3B0(a1, v12);
  }
  if ( v16 == 20 )
    return sub_181B3B0(a1, v12);
  result = 0;
  if ( *(_QWORD *)(*(_QWORD *)*a1 + 376LL) )
    goto LABEL_6;
  return result;
}
