// Function: sub_26968A0
// Address: 0x26968a0
//
__int64 __fastcall sub_26968A0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // rbx
  __int64 v4; // r9
  __int64 v5; // rbx
  __int64 *j; // r13
  unsigned __int8 *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned int v11; // ecx
  __int64 v12; // rsi
  unsigned int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // r11
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v31; // r13
  __int64 v32; // rdx
  unsigned __int8 *v33; // rax
  _BYTE *v34; // rax
  unsigned __int64 v35; // r8
  __int64 v36; // r9
  unsigned int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 **v41; // r15
  __int64 v42; // r13
  __int64 *v43; // rax
  __int64 v44; // rcx
  __int64 *v45; // rdx
  char v46; // dl
  unsigned __int64 *v47; // r12
  __int64 v48; // r13
  unsigned __int64 *v49; // rbx
  __int64 v50; // r15
  __int64 v51; // r8
  unsigned int v52; // edx
  __int64 v53; // rsi
  unsigned int v54; // eax
  __int64 v55; // r10
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 *v58; // rax
  __int64 v59; // r13
  __int64 *v60; // r12
  __int64 v61; // rax
  _QWORD *v62; // rdi
  __int64 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r12
  __int64 v66; // rbx
  __int64 v67; // rax
  int v68; // ecx
  int v69; // ecx
  int v70; // r10d
  unsigned int i; // eax
  __int64 v72; // rdx
  unsigned int v73; // eax
  int v74; // edi
  int v75; // r8d
  __int64 v76; // rdi
  unsigned __int64 v77; // rdi
  __int64 v78; // rax
  _QWORD *v79; // r12
  _QWORD *v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // r14
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 *v87; // r15
  __int64 v88; // rcx
  __int64 v89; // r13
  _QWORD *v90; // rax
  _QWORD *v91; // rsi
  __int64 v92; // rdx
  _QWORD *v93; // rax
  _QWORD *v94; // r13
  __int64 *v95; // r8
  const char *v96; // rsi
  unsigned __int64 v97; // rax
  int v98; // edx
  _QWORD *v99; // rdi
  _QWORD *v100; // rax
  __int64 v101; // rdx
  _QWORD *v102; // rax
  __int64 v103; // rdx
  __int64 v104; // r13
  __int64 *v105; // r8
  const char *v106; // rsi
  __int64 v107; // rax
  __int64 v108; // r9
  __int64 v109; // rax
  __int64 v110; // r13
  const char *v111; // rsi
  __int64 v112; // rdx
  _QWORD *v113; // rdi
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // r14
  __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rax
  __int64 v120; // r15
  _QWORD *v121; // rax
  _QWORD *v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rbx
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 *v129; // rsi
  __int64 v130; // rax
  __int64 v131; // r13
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rax
  __int64 v135; // r13
  __int64 v136; // rax
  __int64 v137; // rdx
  __int64 *v138; // r12
  __int64 v139; // rax
  __int64 *v140; // r15
  unsigned __int64 v141; // rax
  unsigned __int64 v142; // rsi
  __int64 v143; // rax
  __int64 v144; // rcx
  __int64 v145; // r8
  __int64 v146; // r9
  __int64 v147; // rcx
  __int64 v148; // r8
  __int64 v149; // r9
  __int64 v150; // rcx
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  __int64 v156; // rcx
  __int64 v157; // r8
  __int64 v158; // r9
  __int64 v159; // r8
  __int64 v160; // r9
  __int64 *v161; // r14
  unsigned __int8 v162; // bl
  __int64 v163; // r12
  __int64 v164; // rax
  unsigned __int64 v165; // rax
  unsigned __int64 v166; // rdx
  __int64 v167; // rsi
  __int64 v168; // r13
  unsigned __int64 v169; // rax
  _QWORD *v170; // rdi
  __int64 v171; // r12
  __int64 v172; // rax
  unsigned __int16 v173; // r14
  _QWORD *v174; // rax
  _QWORD *v175; // rbx
  __int64 *v176; // r14
  unsigned __int64 v177; // rax
  int v178; // edx
  _QWORD *v179; // rdi
  _QWORD *v180; // rax
  unsigned __int64 v181; // rax
  __int64 v182; // rdx
  __int64 v183; // rbx
  __int64 v184; // rax
  unsigned __int8 *v185; // rsi
  __int64 v186; // r14
  __int64 v187; // rdx
  __int64 v188; // rax
  __int64 v189; // rdi
  unsigned __int8 *v190; // rbx
  __int64 (__fastcall *v191)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v192; // r10
  _QWORD *v193; // rax
  __int64 v194; // rbx
  __int64 v195; // r15
  __int64 v196; // r13
  __int64 v197; // rdx
  unsigned int v198; // esi
  void **v199; // r13
  __int64 v200; // r13
  __int64 v201; // rdx
  __int64 v202; // rbx
  __int64 v203; // rax
  __int16 v204; // dx
  char v205; // cl
  char v206; // dl
  __int64 v207; // rax
  __int64 v208; // r12
  void **v209; // r15
  __int64 v210; // r12
  __int64 v211; // rax
  __int64 v212; // rax
  __int64 v213; // rdi
  __int64 v214; // r14
  __int64 v215; // rax
  __int64 *v216; // rax
  __int64 v217; // r13
  __int64 *v218; // r12
  __int64 v219; // rax
  __int64 v220; // rdx
  __int64 v221; // rcx
  __int64 v222; // r8
  __int64 v223; // r9
  __int64 v224; // rdx
  __int64 v225; // rcx
  __int64 v226; // r8
  __int64 v227; // r9
  __int64 **v228; // r13
  __int64 *v229; // rdx
  __m128i *v230; // rax
  __int64 v231; // rdx
  __m128i *v232; // rsi
  __int8 v233; // cl
  _QWORD *v234; // r12
  unsigned __int64 v235; // rax
  unsigned __int64 v236; // r13
  __int64 v237; // r13
  _QWORD *v238; // rdi
  __int64 v239; // r13
  __int64 v240; // rdx
  unsigned __int64 v241; // rax
  unsigned __int64 v242; // r8
  unsigned __int8 *v243; // rax
  __int64 v244; // rcx
  unsigned __int8 *v245; // rbx
  const char *v246; // r12
  __int64 *v247; // r14
  __int64 v248; // rsi
  __int64 v249; // rsi
  unsigned __int8 *v250; // rsi
  __int64 v251; // rsi
  unsigned __int8 *v252; // rsi
  __int64 v253; // rsi
  unsigned __int8 *v254; // rsi
  __int64 v255; // r12
  __int64 v256; // r8
  __int64 v257; // r10
  __int64 v258; // r9
  __int64 v259; // r13
  __int64 v260; // r15
  __int64 v261; // rax
  unsigned __int64 v262; // rdx
  __int64 *v263; // rax
  __int64 v264; // rdx
  __int64 v265; // rbx
  __int64 v266; // rcx
  unsigned int v267; // eax
  __int64 *v268; // rsi
  __int64 v269; // rdi
  _QWORD *v270; // rax
  _QWORD *v271; // rax
  _QWORD *v272; // r10
  __int64 v273; // rdx
  int v274; // ecx
  int v275; // eax
  _QWORD *v276; // rdi
  __int64 *v277; // rax
  __int64 v278; // rax
  __int64 v279; // rax
  __int64 v280; // r15
  __int64 v281; // r12
  __int64 v282; // rbx
  __int64 v283; // rdx
  unsigned int v284; // esi
  unsigned __int64 v285; // rax
  __int64 v286; // rdx
  __int64 v287; // r9
  __int64 v288; // rax
  __int64 v289; // r12
  __int64 v290; // r13
  int v291; // esi
  __int64 v292; // rsi
  unsigned __int8 *v293; // rsi
  __int64 v294; // rax
  __int64 v295; // rax
  __int64 v296; // rsi
  unsigned __int8 *v297; // rsi
  __int64 v298; // rsi
  unsigned __int8 *v299; // rsi
  __int64 v300; // rsi
  unsigned __int8 *v301; // rsi
  __int64 v302; // rax
  __int64 v303; // rax
  _BYTE *v304; // [rsp+8h] [rbp-6D8h]
  __int64 v305; // [rsp+18h] [rbp-6C8h]
  __int64 *v306; // [rsp+28h] [rbp-6B8h]
  __int64 v307; // [rsp+30h] [rbp-6B0h]
  __int64 v308; // [rsp+38h] [rbp-6A8h]
  __int64 v309; // [rsp+40h] [rbp-6A0h]
  __int64 v310; // [rsp+48h] [rbp-698h]
  __int64 v311; // [rsp+50h] [rbp-690h]
  __int64 v313; // [rsp+60h] [rbp-680h]
  unsigned __int8 v314; // [rsp+6Fh] [rbp-671h]
  __int64 *v315; // [rsp+70h] [rbp-670h]
  __int64 *v316; // [rsp+78h] [rbp-668h]
  __int64 *v317; // [rsp+80h] [rbp-660h]
  __int64 v318; // [rsp+88h] [rbp-658h]
  unsigned __int8 v319; // [rsp+90h] [rbp-650h]
  __int64 v320; // [rsp+90h] [rbp-650h]
  __int64 v321; // [rsp+98h] [rbp-648h]
  __int64 v322; // [rsp+98h] [rbp-648h]
  __int64 v323; // [rsp+A0h] [rbp-640h]
  __int64 v324; // [rsp+A0h] [rbp-640h]
  __int64 v325; // [rsp+A0h] [rbp-640h]
  __int64 v326; // [rsp+A0h] [rbp-640h]
  __int64 *v327; // [rsp+A0h] [rbp-640h]
  __int64 v328; // [rsp+A0h] [rbp-640h]
  int v329; // [rsp+A0h] [rbp-640h]
  __int64 v330; // [rsp+A0h] [rbp-640h]
  unsigned __int64 v332; // [rsp+B8h] [rbp-628h]
  __int64 v333; // [rsp+B8h] [rbp-628h]
  __int64 v334; // [rsp+B8h] [rbp-628h]
  __int64 v335; // [rsp+B8h] [rbp-628h]
  __int64 *v336; // [rsp+B8h] [rbp-628h]
  __int64 *v337; // [rsp+C0h] [rbp-620h]
  __int64 v338; // [rsp+C0h] [rbp-620h]
  unsigned __int8 *v339; // [rsp+C0h] [rbp-620h]
  __int64 v340; // [rsp+C0h] [rbp-620h]
  unsigned int **v341; // [rsp+C0h] [rbp-620h]
  __int64 *v342; // [rsp+C0h] [rbp-620h]
  __int64 v343; // [rsp+C8h] [rbp-618h]
  __int64 v344; // [rsp+D0h] [rbp-610h]
  __int64 *v345; // [rsp+D0h] [rbp-610h]
  __int64 v346; // [rsp+D0h] [rbp-610h]
  __int64 *v347; // [rsp+D0h] [rbp-610h]
  __int64 v348; // [rsp+D8h] [rbp-608h]
  __int64 **v349; // [rsp+D8h] [rbp-608h]
  __int64 v350; // [rsp+D8h] [rbp-608h]
  __int64 v351; // [rsp+D8h] [rbp-608h]
  __int64 *v352; // [rsp+D8h] [rbp-608h]
  __int64 *v353; // [rsp+E0h] [rbp-600h]
  __int64 **v354; // [rsp+E0h] [rbp-600h]
  __int64 *v355; // [rsp+E0h] [rbp-600h]
  __int64 v356; // [rsp+E0h] [rbp-600h]
  const char *v357; // [rsp+E0h] [rbp-600h]
  const char *v358; // [rsp+E0h] [rbp-600h]
  __int64 v359; // [rsp+E0h] [rbp-600h]
  __int64 v360; // [rsp+E8h] [rbp-5F8h]
  __int64 v361; // [rsp+F0h] [rbp-5F0h] BYREF
  __int64 v362; // [rsp+F8h] [rbp-5E8h]
  __m128i *v363; // [rsp+100h] [rbp-5E0h] BYREF
  __int64 v364; // [rsp+108h] [rbp-5D8h]
  __m128i v365; // [rsp+110h] [rbp-5D0h] BYREF
  __m128i *v366; // [rsp+120h] [rbp-5C0h] BYREF
  __int64 v367; // [rsp+128h] [rbp-5B8h]
  __m128i v368; // [rsp+130h] [rbp-5B0h] BYREF
  const char *v369; // [rsp+140h] [rbp-5A0h] BYREF
  __int64 *v370; // [rsp+148h] [rbp-598h]
  const char *v371; // [rsp+150h] [rbp-590h]
  __m128i *v372; // [rsp+158h] [rbp-588h] BYREF
  __int16 v373; // [rsp+160h] [rbp-580h]
  __m128i *v374; // [rsp+170h] [rbp-570h] BYREF
  __int64 v375; // [rsp+178h] [rbp-568h]
  const char *v376; // [rsp+180h] [rbp-560h]
  __int16 v377; // [rsp+190h] [rbp-550h]
  const char *v378; // [rsp+1A0h] [rbp-540h] BYREF
  __int64 v379; // [rsp+1A8h] [rbp-538h]
  _BYTE v380[8]; // [rsp+1B0h] [rbp-530h] BYREF
  __int64 v381; // [rsp+1B8h] [rbp-528h] BYREF
  __int16 v382; // [rsp+1C0h] [rbp-520h]
  _BYTE v383[344]; // [rsp+1F0h] [rbp-4F0h] BYREF
  __int64 v384; // [rsp+348h] [rbp-398h]
  __int64 *v385; // [rsp+350h] [rbp-390h] BYREF
  __int64 v386; // [rsp+358h] [rbp-388h]
  __int64 v387; // [rsp+360h] [rbp-380h] BYREF
  __m128i v388; // [rsp+368h] [rbp-378h] BYREF
  __int64 v389; // [rsp+378h] [rbp-368h]
  __m128i v390; // [rsp+380h] [rbp-360h] BYREF
  __m128i v391; // [rsp+390h] [rbp-350h] BYREF
  char v392[8]; // [rsp+3A0h] [rbp-340h] BYREF
  int v393; // [rsp+3A8h] [rbp-338h]
  char v394; // [rsp+4F0h] [rbp-1F0h]
  int v395; // [rsp+4F4h] [rbp-1ECh]
  __int64 v396; // [rsp+4F8h] [rbp-1E8h]
  const char *v397; // [rsp+500h] [rbp-1E0h] BYREF
  __int64 *v398; // [rsp+508h] [rbp-1D8h]
  __int64 v399; // [rsp+510h] [rbp-1D0h]
  __m128i v400; // [rsp+518h] [rbp-1C8h] BYREF
  __int64 v401; // [rsp+528h] [rbp-1B8h]
  __m128i v402; // [rsp+530h] [rbp-1B0h]
  __m128i v403; // [rsp+540h] [rbp-1A0h]
  _QWORD v404[2]; // [rsp+550h] [rbp-190h] BYREF
  _BYTE v405[324]; // [rsp+560h] [rbp-180h] BYREF
  int v406; // [rsp+6A4h] [rbp-3Ch]
  __int64 v407; // [rsp+6A8h] [rbp-38h]

  v314 = *(_BYTE *)(a1 + 241);
  if ( v314 )
  {
    v31 = (__int64 *)(a1 + 72);
    v32 = sub_25096F0((_QWORD *)(a1 + 72));
    if ( (*(_BYTE *)(v32 + 32) & 0xFu) - 7 <= 1 )
      sub_B491C0(*(_QWORD *)(*(_QWORD *)(v32 + 16) + 24LL));
    v33 = (unsigned __int8 *)sub_2674090(*(_QWORD *)(a1 + 296), a2);
    v34 = sub_2674040(v33);
    v37 = *((_DWORD *)v34 + 8);
    v304 = v34;
    if ( v37 > 0x40 )
    {
      v39 = **((_QWORD **)v34 + 3);
    }
    else
    {
      v38 = *((_QWORD *)v34 + 3);
      if ( !v37 )
        return v314;
      v39 = v38 << (64 - (unsigned __int8)v37) >> (64 - (unsigned __int8)v37);
    }
    if ( (_BYTE)v39 != 1 )
      return v314;
    *a3 = 0;
    if ( !*(_DWORD *)(a1 + 160) && !*(_DWORD *)(a1 + 224) )
    {
      v77 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
        v77 = *(_QWORD *)(v77 + 24);
      v350 = sub_BD5C60(v77);
      v339 = sub_250CBE0(v31, a2);
      v78 = *(_QWORD *)(a1 + 296);
      v79 = *(_QWORD **)(v78 + 40);
      v397 = "main.thread.user_code";
      v400.m128i_i16[4] = 259;
      v80 = *(_QWORD **)(v78 + 32);
      if ( v79 + 6 == v80 || !v80 )
        v81 = 0;
      else
        v81 = v80 - 3;
      v82 = sub_AA8550(v79, v81 + 3, 0, (__int64)&v397, 0);
      v400.m128i_i16[4] = 259;
      v397 = "exit.threads";
      v83 = sub_22077B0(0x50u);
      v87 = (__int64 *)v83;
      if ( v83 )
        sub_AA4D50(v83, v350, (__int64)&v397, (__int64)v339, v82);
      v88 = a2;
      v89 = a2 + 3560;
      if ( !*(_BYTE *)(a2 + 3588) )
        goto LABEL_380;
      v90 = *(_QWORD **)(a2 + 3568);
      v84 = *(unsigned int *)(a2 + 3580);
      v91 = &v90[v84];
      v88 = *(unsigned int *)(a2 + 3580);
      if ( v90 != v91 )
      {
        v84 = (__int64)v90;
        while ( v79 != *(_QWORD **)v84 )
        {
          v84 += 8;
          if ( v91 == (_QWORD *)v84 )
            goto LABEL_389;
        }
        goto LABEL_88;
      }
LABEL_389:
      if ( (unsigned int)v88 < *(_DWORD *)(a2 + 3576) )
      {
        v88 = (unsigned int)(v88 + 1);
        *(_DWORD *)(a2 + 3580) = v88;
        *v91 = v79;
        v90 = *(_QWORD **)(a2 + 3568);
        ++*(_QWORD *)(a2 + 3560);
        v92 = *(unsigned __int8 *)(a2 + 3588);
      }
      else
      {
LABEL_380:
        sub_C8CC70(a2 + 3560, (__int64)v79, v84, v88, v85, v86);
        v92 = *(unsigned __int8 *)(a2 + 3588);
        v90 = *(_QWORD **)(a2 + 3568);
      }
      if ( !(_BYTE)v92 )
        goto LABEL_386;
      v88 = *(unsigned int *)(a2 + 3580);
LABEL_88:
      v92 = (__int64)&v90[(unsigned int)v88];
      if ( (_QWORD *)v92 != v90 )
      {
        while ( v82 != *v90 )
        {
          if ( (_QWORD *)v92 == ++v90 )
            goto LABEL_387;
        }
LABEL_92:
        sub_D695C0((__int64)&v397, v89, v87, v88, v85, v86);
        v356 = *(_QWORD *)(a1 + 296);
        sub_B43C20((__int64)&v397, (__int64)v87);
        v93 = sub_BD2C40(72, 0);
        v94 = v93;
        if ( v93 )
          sub_B4BB80((__int64)v93, v350, 0, 0, (__int64)v397, (unsigned __int16)v398);
        v95 = v94 + 6;
        v96 = *(const char **)(v356 + 48);
        v397 = v96;
        if ( v96 )
        {
          sub_B96E90((__int64)&v397, (__int64)v96, 1);
          v95 = v94 + 6;
          if ( v94 + 6 == &v397 )
          {
            if ( v397 )
              sub_B91220((__int64)&v397, (__int64)v397);
            goto LABEL_98;
          }
          v300 = v94[6];
          if ( !v300 )
          {
LABEL_378:
            v301 = (unsigned __int8 *)v397;
            v94[6] = v397;
            if ( v301 )
              sub_B976B0((__int64)&v397, v301, (__int64)v95);
LABEL_98:
            v97 = v79[6] & 0xFFFFFFFFFFFFFFF8LL;
            if ( v79 + 6 == (_QWORD *)v97 )
            {
              v99 = 0;
            }
            else
            {
              if ( !v97 )
                BUG();
              v98 = *(unsigned __int8 *)(v97 - 24);
              v99 = 0;
              v100 = (_QWORD *)(v97 - 24);
              if ( (unsigned int)(v98 - 30) < 0xB )
                v99 = v100;
            }
            sub_B43D60(v99);
            v333 = sub_312CF50(*(_QWORD *)(a2 + 208) + 400LL, *((_QWORD *)v339 + 5), 6);
            v346 = v101;
            sub_B43C20((__int64)&v385, (__int64)v79);
            v397 = "thread_id.in.block";
            v400.m128i_i16[4] = 259;
            v340 = (__int64)v385;
            v343 = (unsigned __int16)v386;
            v102 = sub_BD2C40(88, 1u);
            v103 = v346;
            v104 = (__int64)v102;
            if ( v102 )
            {
              sub_B4A410((__int64)v102, v333, v346, (__int64)&v397, 1u, v333, v340, v343);
              v103 = v346;
            }
            if ( !*(_BYTE *)v103 )
              *(_WORD *)(v104 + 2) = *(_WORD *)(v104 + 2) & 0xF003 | (4 * ((*(_WORD *)(v103 + 2) >> 4) & 0x3FF));
            v105 = (__int64 *)(v104 + 48);
            v106 = *(const char **)(v356 + 48);
            v397 = v106;
            if ( v106 )
            {
              sub_B96E90((__int64)&v397, (__int64)v106, 1);
              v105 = (__int64 *)(v104 + 48);
              if ( (const char **)(v104 + 48) == &v397 )
              {
                if ( v397 )
                  sub_B91220((__int64)&v397, (__int64)v397);
                goto LABEL_110;
              }
              v298 = *(_QWORD *)(v104 + 48);
              if ( !v298 )
              {
LABEL_373:
                v299 = (unsigned __int8 *)v397;
                *(_QWORD *)(v104 + 48) = v397;
                if ( v299 )
                  sub_B976B0((__int64)&v397, v299, (__int64)v105);
LABEL_110:
                sub_B43C20((__int64)&v385, (__int64)v79);
                v397 = "thread.is_main";
                v400.m128i_i16[4] = 259;
                v107 = sub_AD64C0(*(_QWORD *)(v104 + 8), 0, 0);
                v109 = sub_B52500(53, 33, v104, v107, (__int64)&v397, v108, (__int64)v385, v386);
                v110 = v109;
                v111 = *(const char **)(v356 + 48);
                v397 = v111;
                if ( v111 )
                {
                  sub_B96E90((__int64)&v397, (__int64)v111, 1);
                  v112 = v110 + 48;
                  if ( (const char **)(v110 + 48) == &v397 )
                  {
                    if ( v397 )
                      sub_B91220((__int64)&v397, (__int64)v397);
                    goto LABEL_114;
                  }
                  v296 = *(_QWORD *)(v110 + 48);
                  if ( !v296 )
                  {
LABEL_368:
                    v297 = (unsigned __int8 *)v397;
                    *(_QWORD *)(v110 + 48) = v397;
                    if ( v297 )
                      sub_B976B0((__int64)&v397, v297, v112);
                    goto LABEL_114;
                  }
                }
                else
                {
                  v112 = v109 + 48;
                  if ( (const char **)(v109 + 48) == &v397 || (v296 = *(_QWORD *)(v109 + 48)) == 0 )
                  {
LABEL_114:
                    sub_B43C20((__int64)&v397, (__int64)v79);
                    v357 = v397;
                    v360 = (unsigned __int16)v398;
                    v113 = sub_BD2C40(72, 3u);
                    if ( v113 )
                      sub_B4C9A0((__int64)v113, (__int64)v87, v82, v110, 3u, v360, (__int64)v357, v360);
LABEL_227:
                    v210 = sub_ACD640(*((_QWORD *)v304 + 1), 3, 0);
                    v211 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
                    LODWORD(v397) = 2;
                    v212 = sub_AAAE30(v211, v210, &v397, 1);
                    v213 = *(_QWORD *)(a1 + 304);
                    LODWORD(v397) = 0;
                    *(_QWORD *)(a1 + 304) = sub_AAAE30(v213, v212, &v397, 1);
                    if ( *(_QWORD *)(a2 + 4392) )
                    {
                      v214 = *(_QWORD *)(a1 + 296);
                      v215 = sub_B43CB0(v214);
                      v216 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 4392))(
                                          *(_QWORD *)(a2 + 4400),
                                          v215);
                      v217 = *v216;
                      v218 = v216;
                      v219 = sub_B2BE50(*v216);
                      if ( sub_B6EA50(v219)
                        || (v302 = sub_B2BE50(v217),
                            v303 = sub_B6F970(v302),
                            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v303 + 48LL))(v303)) )
                      {
                        sub_B174A0((__int64)&v385, *(_QWORD *)(a2 + 4408), (__int64)"OMP120", 6, v214);
                        sub_B18290((__int64)&v385, "Transformed generic-mode kernel to SPMD-mode.", 0x2Du);
                        sub_23FE290((__int64)&v397, (__int64)&v385, v220, v221, v222, v223);
                        v407 = v396;
                        v397 = (const char *)&unk_49D9D78;
                        sub_B18290((__int64)&v397, " [", 2u);
                        sub_B18290((__int64)&v397, "OMP120", 6u);
                        sub_B18290((__int64)&v397, "]", 1u);
                        sub_23FE290((__int64)&v378, (__int64)&v397, v224, v225, v226, v227);
                        v378 = (const char *)&unk_49D9D78;
                        v384 = v407;
                        v397 = (const char *)&unk_49D9D40;
                        sub_23FD590((__int64)v404);
                        v385 = (__int64 *)&unk_49D9D40;
                        sub_23FD590((__int64)v392);
                        sub_1049740(v218, (__int64)&v378);
                        v378 = (const char *)&unk_49D9D40;
                        sub_23FD590((__int64)v383);
                      }
                    }
                    return v314;
                  }
                }
                v359 = v112;
                sub_B91220(v112, v296);
                v112 = v359;
                goto LABEL_368;
              }
            }
            else
            {
              if ( v105 == (__int64 *)&v397 )
                goto LABEL_110;
              v298 = *(_QWORD *)(v104 + 48);
              if ( !v298 )
                goto LABEL_110;
            }
            v342 = v105;
            sub_B91220((__int64)v105, v298);
            v105 = v342;
            goto LABEL_373;
          }
        }
        else
        {
          if ( v95 == (__int64 *)&v397 )
            goto LABEL_98;
          v300 = v94[6];
          if ( !v300 )
            goto LABEL_98;
        }
        v352 = v95;
        sub_B91220((__int64)v95, v300);
        v95 = v352;
        goto LABEL_378;
      }
LABEL_387:
      if ( (unsigned int)v88 < *(_DWORD *)(a2 + 3576) )
      {
        v88 = (unsigned int)(v88 + 1);
        *(_DWORD *)(a2 + 3580) = v88;
        *(_QWORD *)v92 = v82;
        ++*(_QWORD *)(a2 + 3560);
        goto LABEL_92;
      }
LABEL_386:
      sub_C8CC70(v89, v82, v92, v88, v85, v86);
      goto LABEL_92;
    }
    v397 = 0;
    v399 = 8;
    v40 = *(_QWORD *)(a2 + 208);
    v400.m128i_i8[4] = 1;
    v400.m128i_i32[0] = 0;
    v344 = v40;
    v398 = &v400.m128i_i64[1];
    v41 = *(__int64 ***)(a1 + 280);
    v354 = &v41[*(unsigned int *)(a1 + 288)];
    if ( v41 == v354 )
      goto LABEL_225;
    v42 = (*v41)[5];
    while ( 1 )
    {
      v43 = v398;
      v44 = HIDWORD(v399);
      v45 = &v398[HIDWORD(v399)];
      if ( v398 == v45 )
        break;
      while ( v42 != *v43 )
      {
        if ( v45 == ++v43 )
          goto LABEL_126;
      }
      while ( 1 )
      {
LABEL_39:
        if ( v354 == ++v41 )
          goto LABEL_64;
LABEL_40:
        v45 = *v41;
        v42 = (*v41)[5];
        if ( v400.m128i_i8[4] )
          break;
LABEL_41:
        sub_C8CC70((__int64)&v397, v42, (__int64)v45, v44, v35, v36);
        if ( v46 )
          goto LABEL_42;
      }
    }
LABEL_126:
    if ( HIDWORD(v399) >= (unsigned int)v399 )
      goto LABEL_41;
    v44 = (unsigned int)++HIDWORD(v399);
    *v45 = v42;
    ++v397;
LABEL_42:
    v47 = (unsigned __int64 *)(v42 + 48);
    v385 = &v387;
    v386 = 0x300000000LL;
    v35 = *(_QWORD *)(*(_QWORD *)(v42 + 48) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v42 + 48 == v35 )
      goto LABEL_39;
    v338 = v3;
    v48 = 0;
    v49 = (unsigned __int64 *)v35;
    v349 = v41;
    while ( 1 )
    {
      v50 = 0;
      if ( v49 )
        v50 = (__int64)(v49 - 3);
      if ( !(unsigned __int8)sub_B46970((unsigned __int8 *)v50) && !(unsigned __int8)sub_B46420(v50) )
        goto LABEL_45;
      if ( *(_BYTE *)v50 == 85 )
      {
        if ( *(char *)(v50 + 7) >= 0
          || ((v114 = sub_BD2BC0(v50), v116 = v114 + v115, *(char *)(v50 + 7) >= 0)
            ? (v117 = v116 >> 4)
            : (LODWORD(v117) = (v116 - sub_BD2BC0(v50)) >> 4),
              !(_DWORD)v117) )
        {
          v118 = *(_QWORD *)(v344 + 32432);
          if ( v118 )
          {
            v119 = *(_QWORD *)(v50 - 32);
            if ( v119 )
            {
              if ( !*(_BYTE *)v119 && *(_QWORD *)(v119 + 24) == *(_QWORD *)(v50 + 80) && v118 == v119 )
                goto LABEL_45;
            }
          }
        }
      }
      if ( *(_QWORD *)(v50 + 16) )
        goto LABEL_44;
      v52 = *(_DWORD *)(a1 + 272);
      v53 = *(_QWORD *)(a1 + 256);
      if ( !v52 )
        goto LABEL_44;
      v54 = (v52 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v44 = v53 + 8LL * v54;
      v55 = *(_QWORD *)v44;
      if ( v50 != *(_QWORD *)v44 )
        break;
LABEL_54:
      if ( v44 == v53 + 8LL * v52 )
      {
LABEL_44:
        v48 = 0;
LABEL_45:
        v35 = *v49;
        v49 = (unsigned __int64 *)(*v49 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v47 == v49 )
          goto LABEL_59;
      }
      else
      {
        if ( !v48 )
        {
          v48 = v50;
          goto LABEL_45;
        }
        v56 = (unsigned int)v386;
        v44 = HIDWORD(v386);
        v57 = (unsigned int)v386 + 1LL;
        if ( v57 > HIDWORD(v386) )
        {
          sub_C8D5F0((__int64)&v385, &v387, v57, 0x10u, v51, v36);
          v56 = (unsigned int)v386;
        }
        v58 = &v385[2 * v56];
        v58[1] = v48;
        v48 = v50;
        *v58 = v50;
        LODWORD(v386) = v386 + 1;
        v35 = *v49;
        v49 = (unsigned __int64 *)(*v49 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v47 == v49 )
        {
LABEL_59:
          v59 = (__int64)v385;
          v41 = v349;
          v3 = v338;
          v60 = &v385[2 * (unsigned int)v386];
          if ( v385 != v60 )
          {
            do
            {
              v61 = *(_QWORD *)(v59 + 8);
              v62 = *(_QWORD **)v59;
              LOWORD(v3) = 0;
              v59 += 16;
              sub_B444E0(v62, v61 + 24, v3);
            }
            while ( v60 != (__int64 *)v59 );
            v60 = v385;
          }
          if ( v60 == &v387 )
            goto LABEL_39;
          v41 = v349 + 1;
          _libc_free((unsigned __int64)v60);
          if ( v354 != v349 + 1 )
            goto LABEL_40;
LABEL_64:
          v386 = 0x400000000LL;
          v63 = *(__int64 **)(a1 + 280);
          v64 = *(unsigned int *)(a1 + 288);
          v385 = &v387;
          v345 = &v63[v64];
          if ( v63 == v345 )
            goto LABEL_225;
          v355 = v63;
          while ( 2 )
          {
            v65 = *v355;
            v66 = *(_QWORD *)(*v355 + 40);
            v67 = sub_B43CB0(*v355);
            v379 = 0;
            v378 = (const char *)(v67 & 0xFFFFFFFFFFFFFFFCLL);
            nullsub_1518();
            v68 = *(_DWORD *)(a2 + 160);
            if ( !v68 )
LABEL_403:
              BUG();
            v69 = v68 - 1;
            v70 = 1;
            v332 = (unsigned __int64)(((unsigned int)&unk_438FC88 >> 9) ^ ((unsigned int)&unk_438FC88 >> 4)) << 32;
            for ( i = v69
                    & (((0xBF58476D1CE4E5B9LL
                       * (v332
                        | ((unsigned int)v379 >> 9)
                        ^ ((unsigned int)v379 >> 4)
                        ^ (16 * (((unsigned int)v378 >> 9) ^ ((unsigned int)v378 >> 4))))) >> 31)
                     ^ (484763065
                      * (v332
                       | ((unsigned int)v379 >> 9)
                       ^ ((unsigned int)v379 >> 4)
                       ^ (16 * (((unsigned int)v378 >> 9) ^ ((unsigned int)v378 >> 4)))))); ; i = v69 & v73 )
            {
              v72 = *(_QWORD *)(a2 + 144) + 32LL * i;
              if ( *(_UNKNOWN **)v72 == &unk_438FC88
                && *(const char **)(v72 + 8) == v378
                && *(_QWORD *)(v72 + 16) == v379 )
              {
                break;
              }
              if ( *(_QWORD *)v72 == -4096
                && *(_QWORD *)(v72 + 8) == qword_4FEE4D0
                && qword_4FEE4D8 == *(_QWORD *)(v72 + 16) )
              {
                goto LABEL_403;
              }
              v73 = v70 + i;
              ++v70;
            }
            v120 = *(_QWORD *)(v72 + 24);
            v351 = v120 + 472;
            if ( *(_BYTE *)(v120 + 500) )
            {
              v121 = *(_QWORD **)(v120 + 480);
              v122 = &v121[*(unsigned int *)(v120 + 492)];
              if ( v121 == v122 )
              {
LABEL_287:
                v255 = *(_QWORD *)(v66 + 56);
                v256 = v66 + 48;
                v257 = v120;
                v258 = 0;
                v259 = 0;
                v260 = v66 + 48;
                if ( v255 == v66 + 48 )
                  goto LABEL_135;
                while ( 2 )
                {
                  v264 = *(unsigned int *)(a1 + 272);
                  v265 = v255 - 24;
                  v266 = *(_QWORD *)(a1 + 256);
                  if ( !v255 )
                    v265 = 0;
                  if ( (_DWORD)v264 )
                  {
                    v256 = (unsigned int)(v264 - 1);
                    v267 = v256 & (((unsigned int)v265 >> 9) ^ ((unsigned int)v265 >> 4));
                    v268 = (__int64 *)(v266 + 8LL * v267);
                    v269 = *v268;
                    if ( v265 == *v268 )
                    {
LABEL_298:
                      if ( v268 != (__int64 *)(v266 + 8 * v264) )
                      {
                        if ( !*(_BYTE *)(v257 + 500) )
                          goto LABEL_315;
                        v270 = *(_QWORD **)(v257 + 480);
                        v266 = *(unsigned int *)(v257 + 492);
                        v264 = (__int64)&v270[v266];
                        if ( v270 == (_QWORD *)v264 )
                        {
LABEL_331:
                          if ( (unsigned int)v266 < *(_DWORD *)(v257 + 488) )
                          {
                            v258 = v265;
                            *(_DWORD *)(v257 + 492) = v266 + 1;
                            *(_QWORD *)v264 = v265;
                            ++*(_QWORD *)(v257 + 472);
                            if ( !v259 )
                              v259 = v265;
                            goto LABEL_293;
                          }
LABEL_315:
                          v328 = v257;
                          sub_C8CC70(v351, v265, v264, v266, v256, v258);
                          v258 = v265;
                          v257 = v328;
                          if ( !v259 )
                            v259 = v265;
                        }
                        else
                        {
                          while ( v265 != *v270 )
                          {
                            if ( (_QWORD *)v264 == ++v270 )
                              goto LABEL_331;
                          }
                          v258 = v265;
                          if ( !v259 )
                            v259 = v265;
                        }
LABEL_293:
                        v255 = *(_QWORD *)(v255 + 8);
                        if ( v260 == v255 )
                          goto LABEL_135;
                        continue;
                      }
                    }
                    else
                    {
                      v291 = 1;
                      while ( v269 != -4096 )
                      {
                        v267 = v256 & (v291 + v267);
                        v329 = v291 + 1;
                        v268 = (__int64 *)(v266 + 8LL * v267);
                        v269 = *v268;
                        if ( v265 == *v268 )
                          goto LABEL_298;
                        v291 = v329;
                      }
                    }
                  }
                  break;
                }
                if ( v259 )
                {
                  v261 = (unsigned int)v386;
                  v262 = (unsigned int)v386 + 1LL;
                  if ( v262 > HIDWORD(v386) )
                  {
                    v322 = v257;
                    v330 = v258;
                    sub_C8D5F0((__int64)&v385, &v387, v262, 0x10u, v256, v258);
                    v261 = (unsigned int)v386;
                    v257 = v322;
                    v258 = v330;
                  }
                  v263 = &v385[2 * v261];
                  *v263 = v259;
                  v259 = 0;
                  v263[1] = v258;
                  v258 = 0;
                  LODWORD(v386) = v386 + 1;
                }
                goto LABEL_293;
              }
              while ( v65 != *v121 )
              {
                if ( v122 == ++v121 )
                  goto LABEL_287;
              }
            }
            else if ( !sub_C8CA60(v351, v65) )
            {
              goto LABEL_287;
            }
LABEL_135:
            if ( v345 != ++v355 )
              continue;
            break;
          }
          v123 = 2LL * (unsigned int)v386;
          v306 = &v385[v123];
          if ( v385 != &v385[v123] )
          {
            v316 = v385;
            while ( 1 )
            {
              v124 = *v316;
              v125 = v316[1];
              v323 = *(_QWORD *)(*v316 + 40);
              v126 = *(_QWORD *)(*(_QWORD *)(v323 + 72) + 40LL);
              v382 = 259;
              v321 = v126;
              v378 = "region.guarded.end";
              v127 = *(_QWORD *)(v125 + 32);
              if ( v127 == *(_QWORD *)(v125 + 40) + 48LL || !v127 )
                v128 = 0;
              else
                v128 = v127 - 24;
              v129 = (__int64 *)(v128 + 24);
              v130 = v307;
              LOWORD(v130) = 0;
              v307 = v130;
              v131 = sub_F36960(v323, v129, v130, 0, 0, 0, (void **)&v378, 0);
              v378 = "region.barrier";
              v317 = (__int64 *)v131;
              v382 = 259;
              v132 = sub_AA5190(v131);
              if ( v132 )
                v132 -= 24;
              v133 = v308;
              LOWORD(v133) = 0;
              v308 = v133;
              v134 = sub_F36960(v131, (__int64 *)(v132 + 24), v133, 0, 0, 0, (void **)&v378, 0);
              v378 = "region.exit";
              v135 = v134;
              v347 = (__int64 *)v134;
              v382 = 259;
              v136 = sub_AA5190(v134);
              if ( v136 )
                v136 -= 24;
              v137 = v309;
              LOWORD(v137) = 0;
              v309 = v137;
              v138 = (__int64 *)sub_F36960(v135, (__int64 *)(v136 + 24), v137, 0, 0, 0, (void **)&v378, 0);
              v378 = "region.guarded";
              v139 = v310;
              v382 = 259;
              LOWORD(v139) = 0;
              v310 = v139;
              v140 = (__int64 *)sub_F36960(v323, (__int64 *)(v124 + 24), v139, 0, 0, 0, (void **)&v378, 0);
              v382 = 259;
              v378 = "region.check.tid";
              v313 = v323 + 48;
              v141 = *(_QWORD *)(v323 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v323 + 48 == v141 )
              {
                v142 = 0;
              }
              else
              {
                if ( !v141 )
                  BUG();
                v142 = v141 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v141 - 24) - 30 >= 0xB )
                  v142 = 0;
              }
              v143 = v311;
              LOWORD(v143) = 0;
              v311 = v143;
              v315 = (__int64 *)sub_F36960(v323, (__int64 *)(v142 + 24), v143, 0, 0, 0, (void **)&v378, 0);
              sub_D695C0((__int64)&v378, a2 + 3560, v317, v144, v145, v146);
              sub_D695C0((__int64)&v378, a2 + 3560, v347, v147, v148, v149);
              sub_D695C0((__int64)&v378, a2 + 3560, v138, v150, v151, v152);
              sub_D695C0((__int64)&v378, a2 + 3560, v140, v153, v154, v155);
              sub_D695C0((__int64)&v378, a2 + 3560, v315, v156, v157, v158);
              v161 = (__int64 *)v140[7];
              if ( v161 == v140 + 6 )
              {
                v319 = 0;
              }
              else
              {
                v162 = 0;
                do
                {
                  v378 = v380;
                  v379 = 0x400000000LL;
                  if ( !v161 )
                    BUG();
                  v163 = *(v161 - 1);
                  if ( v163 )
                  {
                    v164 = 0;
                    do
                    {
                      if ( v140 != *(__int64 **)(*(_QWORD *)(v163 + 24) + 40LL) )
                      {
                        if ( v164 + 1 > (unsigned __int64)HIDWORD(v379) )
                        {
                          sub_C8D5F0((__int64)&v378, v380, v164 + 1, 8u, v159, v160);
                          v164 = (unsigned int)v379;
                        }
                        *(_QWORD *)&v378[8 * v164] = v163;
                        v164 = (unsigned int)(v379 + 1);
                        LODWORD(v379) = v379 + 1;
                      }
                      v163 = *(_QWORD *)(v163 + 8);
                    }
                    while ( v163 );
                    if ( (_DWORD)v164 )
                    {
                      v228 = (__int64 **)*(v161 - 2);
                      v320 = sub_ACA8A0(v228);
                      v369 = sub_BD5D20((__int64)(v161 - 3));
                      v370 = v229;
                      v371 = ".guarded.output.alloc";
                      v373 = 773;
                      sub_CA0F50((__int64 *)&v363, (void **)&v369);
                      v230 = v363;
                      v231 = v364;
                      v232 = (__m128i *)((char *)v363 + v364);
                      if ( v363 != (__m128i *)&v363->m128i_i8[v364] )
                      {
                        do
                        {
                          v233 = v230->m128i_i8[0];
                          if ( (unsigned __int8)((v230->m128i_i8[0] & 0xDF) - 65) > 0x19u
                            && (unsigned __int8)(v233 - 48) > 9u
                            && v233 != 95 )
                          {
                            v230->m128i_i8[0] = 46;
                          }
                          v230 = (__m128i *)((char *)v230 + 1);
                        }
                        while ( v230 != v232 );
                        v232 = v363;
                        v231 = v364;
                      }
                      v366 = &v368;
                      if ( v232 == &v365 )
                      {
                        v368 = _mm_load_si128(&v365);
                      }
                      else
                      {
                        v366 = v232;
                        v368.m128i_i64[0] = v365.m128i_i64[0];
                      }
                      v367 = v231;
                      v364 = 0;
                      v363 = &v365;
                      v374 = (__m128i *)&v366;
                      v361 = 0x100000003LL;
                      v365.m128i_i8[0] = 0;
                      v377 = 260;
                      v234 = sub_BD2C40(88, unk_3F0FAE8);
                      if ( v234 )
                        sub_B30000((__int64)v234, v321, v228, 0, 7, v320, (__int64)&v374, 0, 0, v361, 0);
                      if ( v366 != &v368 )
                        j_j___libc_free_0((unsigned __int64)v366);
                      if ( v363 != &v365 )
                        j_j___libc_free_0((unsigned __int64)v363);
                      v235 = v317[6] & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (__int64 *)v235 == v317 + 6 )
                      {
                        v236 = 0;
                      }
                      else
                      {
                        if ( !v235 )
                          BUG();
                        v236 = v235 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v235 - 24) - 30 >= 0xB )
                          v236 = 0;
                      }
                      v237 = v236 + 24;
                      v238 = sub_BD2C40(80, unk_3F10A10);
                      if ( v238 )
                        sub_B4D460((__int64)v238, (__int64)(v161 - 3), (__int64)v234, v237, 0);
                      v239 = *(v161 - 2);
                      v374 = (__m128i *)sub_BD5D20((__int64)(v161 - 3));
                      v375 = v240;
                      v377 = 773;
                      v376 = ".guarded.output.load";
                      v241 = v347[6] & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (__int64 *)v241 == v347 + 6 )
                      {
                        v242 = 0;
                      }
                      else
                      {
                        if ( !v241 )
                          BUG();
                        v242 = v241 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v241 - 24) - 30 >= 0xB )
                          v242 = 0;
                      }
                      v335 = v242 + 24;
                      v243 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A14);
                      v245 = v243;
                      if ( v243 )
                        sub_B4D230((__int64)v243, v239, (__int64)v234, (__int64)&v374, v335, 0);
                      v246 = &v378[8 * (unsigned int)v379];
                      if ( v378 != v246 )
                      {
                        v336 = v161;
                        v247 = (__int64 *)v378;
                        do
                        {
                          v248 = *v247++;
                          sub_256E5A0(a2, v248, v245, v244, v159, v160);
                        }
                        while ( v246 != (const char *)v247 );
                        v161 = v336;
                        v246 = v378;
                      }
                      if ( v246 != v380 )
                        _libc_free((unsigned __int64)v246);
                      v162 = v314;
                    }
                    else if ( v378 != v380 )
                    {
                      _libc_free((unsigned __int64)v378);
                    }
                  }
                  v161 = (__int64 *)v161[1];
                }
                while ( v140 + 6 != v161 );
                v319 = v162;
              }
              v165 = *(_QWORD *)(v323 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              v166 = v165;
              if ( v313 == v165 )
                goto LABEL_329;
              if ( !v165 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v165 - 24) - 30 > 0xA )
LABEL_329:
                BUG();
              v167 = *(_QWORD *)(v165 + 24);
              v168 = *(_QWORD *)(a2 + 208);
              v363 = (__m128i *)v167;
              if ( !v167 )
                goto LABEL_170;
              sub_B96E90((__int64)&v363, v167, 1);
              v169 = *(_QWORD *)(v323 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              v166 = v169;
              if ( v313 != v169 )
                break;
LABEL_330:
              v170 = 0;
LABEL_171:
              sub_B43D60(v170);
              v368.m128i_i16[0] = 0;
              v366 = (__m128i *)v323;
              v368.m128i_i64[1] = (__int64)v363;
              v367 = v323 + 48;
              if ( v363 )
                sub_B96E90((__int64)&v368.m128i_i64[1], (__int64)v363, 1);
              v171 = v168 + 400;
              sub_2677420(v168 + 400, (__int64)&v366);
              v172 = sub_31376D0(v168 + 400, &v366, &v361);
              v334 = sub_313A9F0(v168 + 400, v172, (unsigned int)v361, 0, 0);
              sub_B43C20((__int64)&v378, v323);
              v173 = v379;
              v358 = v378;
              v174 = sub_BD2C40(72, 1u);
              v175 = v174;
              if ( v174 )
                sub_B4C8F0((__int64)v174, (__int64)v315, 1u, (__int64)v358, v173);
              v176 = v175 + 6;
              v374 = v363;
              if ( !v363 )
              {
                if ( v176 == (__int64 *)&v374 )
                  goto LABEL_179;
                v249 = v175[6];
                if ( !v249 )
                  goto LABEL_179;
LABEL_273:
                sub_B91220((__int64)(v175 + 6), v249);
                goto LABEL_274;
              }
              sub_B96E90((__int64)&v374, (__int64)v363, 1);
              if ( v176 == (__int64 *)&v374 )
              {
                if ( v374 )
                  sub_B91220((__int64)&v374, (__int64)v374);
                goto LABEL_179;
              }
              v249 = v175[6];
              if ( v249 )
                goto LABEL_273;
LABEL_274:
              v250 = (unsigned __int8 *)v374;
              v175[6] = v374;
              if ( v250 )
                sub_B976B0((__int64)&v374, v250, (__int64)(v175 + 6));
LABEL_179:
              v177 = v315[6] & 0xFFFFFFFFFFFFFFF8LL;
              if ( v315 + 6 == (__int64 *)v177 )
              {
                v179 = 0;
              }
              else
              {
                if ( !v177 )
                  BUG();
                v178 = *(unsigned __int8 *)(v177 - 24);
                v179 = 0;
                v180 = (_QWORD *)(v177 - 24);
                if ( (unsigned int)(v178 - 30) < 0xB )
                  v179 = v180;
              }
              sub_B43D60(v179);
              v370 = v315 + 6;
              v369 = (const char *)v315;
              LOWORD(v371) = 0;
              v372 = v363;
              if ( v363 )
                sub_B96E90((__int64)&v372, (__int64)v363, 1);
              sub_2677420(v168 + 400, (__int64)&v369);
              v181 = sub_312CF50(v168 + 400, v321, 6);
              v183 = v182;
              v341 = (unsigned int **)(v168 + 912);
              v382 = 257;
              v184 = sub_921880((unsigned int **)(v168 + 912), v181, v182, 0, 0, (__int64)&v378, 0);
              v185 = (unsigned __int8 *)v363;
              v186 = v184;
              v378 = (const char *)v363;
              if ( !v363 )
              {
                v187 = v184 + 48;
                if ( (const char **)(v184 + 48) == &v378 )
                  goto LABEL_189;
                v185 = *(unsigned __int8 **)(v184 + 48);
                if ( !v185 )
                  goto LABEL_189;
LABEL_268:
                v324 = v187;
                sub_B91220(v187, (__int64)v185);
                v187 = v324;
                goto LABEL_269;
              }
              sub_B96E90((__int64)&v378, (__int64)v363, 1);
              v187 = v186 + 48;
              if ( (const char **)(v186 + 48) == &v378 )
              {
                v185 = (unsigned __int8 *)v378;
                if ( v378 )
                  sub_B91220((__int64)&v378, (__int64)v378);
                goto LABEL_189;
              }
              v185 = *(unsigned __int8 **)(v186 + 48);
              if ( v185 )
                goto LABEL_268;
LABEL_269:
              v185 = (unsigned __int8 *)v378;
              *(_QWORD *)(v186 + 48) = v378;
              if ( v185 )
                sub_B976B0((__int64)&v378, v185, v187);
LABEL_189:
              if ( !*(_BYTE *)v183 )
                *(_WORD *)(v186 + 2) = *(_WORD *)(v186 + 2) & 0xF003 | (4 * ((*(_WORD *)(v183 + 2) >> 4) & 0x3FF));
              v377 = 257;
              v188 = sub_AD6530(*(_QWORD *)(v186 + 8), (__int64)v185);
              v189 = *(_QWORD *)(v168 + 992);
              v190 = (unsigned __int8 *)v188;
              v191 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v189 + 56LL);
              if ( v191 != sub_928890 )
              {
                v192 = v191(v189, 32u, (_BYTE *)v186, v190);
LABEL_195:
                if ( v192 )
                  goto LABEL_196;
                goto LABEL_307;
              }
              if ( *(_BYTE *)v186 <= 0x15u && *v190 <= 0x15u )
              {
                v192 = sub_AAB310(0x20u, (unsigned __int8 *)v186, v190);
                goto LABEL_195;
              }
LABEL_307:
              v382 = 257;
              v271 = sub_BD2C40(72, unk_3F10FD0);
              v272 = v271;
              if ( v271 )
              {
                v273 = *(_QWORD *)(v186 + 8);
                v325 = (__int64)v271;
                v274 = *(unsigned __int8 *)(v273 + 8);
                if ( (unsigned int)(v274 - 17) > 1 )
                {
                  v278 = sub_BCB2A0(*(_QWORD **)v273);
                }
                else
                {
                  v275 = *(_DWORD *)(v273 + 32);
                  v276 = *(_QWORD **)v273;
                  BYTE4(v362) = (_BYTE)v274 == 18;
                  LODWORD(v362) = v275;
                  v277 = (__int64 *)sub_BCB2A0(v276);
                  v278 = sub_BCE1B0(v277, v362);
                }
                sub_B523C0(v325, v278, 53, 32, v186, (__int64)v190, (__int64)&v378, 0, 0, 0);
                v272 = (_QWORD *)v325;
              }
              v326 = (__int64)v272;
              (*(void (__fastcall **)(_QWORD, _QWORD *, __m128i **, _QWORD, _QWORD))(**(_QWORD **)(v168 + 1000) + 16LL))(
                *(_QWORD *)(v168 + 1000),
                v272,
                &v374,
                *(_QWORD *)(v168 + 968),
                *(_QWORD *)(v168 + 976));
              v279 = *(_QWORD *)(v168 + 912);
              v192 = v326;
              if ( v279 != v279 + 16LL * *(unsigned int *)(v168 + 920) )
              {
                v327 = v140;
                v280 = v192;
                v281 = *(_QWORD *)(v168 + 912);
                v282 = v279 + 16LL * *(unsigned int *)(v168 + 920);
                do
                {
                  v283 = *(_QWORD *)(v281 + 8);
                  v284 = *(_DWORD *)v281;
                  v281 += 16;
                  sub_B99FD0(v280, v284, v283);
                }
                while ( v282 != v281 );
                v192 = v280;
                v171 = v168 + 400;
                v140 = v327;
              }
LABEL_196:
              v382 = 257;
              v318 = v192;
              v193 = sub_BD2C40(72, 3u);
              v194 = (__int64)v193;
              if ( v193 )
                sub_B4C9A0((__int64)v193, (__int64)v140, (__int64)v347, v318, 3u, 0, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(v168 + 1000) + 16LL))(
                *(_QWORD *)(v168 + 1000),
                v194,
                &v378,
                *(_QWORD *)(v168 + 968),
                *(_QWORD *)(v168 + 976));
              v195 = *(_QWORD *)(v168 + 912);
              v196 = v195 + 16LL * *(unsigned int *)(v168 + 920);
              while ( v196 != v195 )
              {
                v197 = *(_QWORD *)(v195 + 8);
                v198 = *(_DWORD *)v195;
                v195 += 16;
                sub_B99FD0(v194, v198, v197);
              }
              v199 = (void **)(v194 + 48);
              v378 = (const char *)v363;
              if ( !v363 )
              {
                if ( v199 == (void **)&v378 )
                  goto LABEL_204;
                v253 = *(_QWORD *)(v194 + 48);
                if ( !v253 )
                  goto LABEL_204;
LABEL_283:
                sub_B91220(v194 + 48, v253);
                goto LABEL_284;
              }
              sub_B96E90((__int64)&v378, (__int64)v363, 1);
              if ( v199 == (void **)&v378 )
              {
                if ( v378 )
                  sub_B91220((__int64)&v378, (__int64)v378);
                goto LABEL_204;
              }
              v253 = *(_QWORD *)(v194 + 48);
              if ( v253 )
                goto LABEL_283;
LABEL_284:
              v254 = (unsigned __int8 *)v378;
              *(_QWORD *)(v194 + 48) = v378;
              if ( v254 )
                sub_B976B0((__int64)&v378, v254, v194 + 48);
LABEL_204:
              v200 = sub_312CF50(v171, v321, 187);
              v202 = v201;
              v203 = sub_AA5190((__int64)v347);
              if ( v203 )
              {
                v205 = v204;
                v206 = HIBYTE(v204);
              }
              else
              {
                v206 = 0;
                v205 = 0;
              }
              v379 = v203;
              v380[0] = v205;
              v378 = (const char *)v347;
              v380[1] = v206;
              v381 = 0;
              sub_2677420(v171, (__int64)&v378);
              if ( v381 )
                sub_B91220((__int64)&v381, v381);
              v382 = 257;
              v374 = (__m128i *)v334;
              v375 = v186;
              v207 = sub_921880(v341, v200, v202, (int)&v374, 2, (__int64)&v378, 0);
              v208 = v207;
              v378 = (const char *)v363;
              if ( !v363 )
              {
                v209 = (void **)(v207 + 48);
                if ( (const char **)(v207 + 48) == &v378 )
                  goto LABEL_212;
                v251 = *(_QWORD *)(v207 + 48);
                if ( !v251 )
                  goto LABEL_212;
LABEL_278:
                sub_B91220((__int64)v209, v251);
                goto LABEL_279;
              }
              v209 = (void **)(v207 + 48);
              sub_B96E90((__int64)&v378, (__int64)v363, 1);
              if ( v209 == (void **)&v378 )
              {
                if ( v378 )
                  sub_B91220((__int64)&v378, (__int64)v378);
                goto LABEL_212;
              }
              v251 = *(_QWORD *)(v208 + 48);
              if ( v251 )
                goto LABEL_278;
LABEL_279:
              v252 = (unsigned __int8 *)v378;
              *(_QWORD *)(v208 + 48) = v378;
              if ( v252 )
                sub_B976B0((__int64)&v378, v252, (__int64)v209);
LABEL_212:
              if ( !*(_BYTE *)v202 )
                *(_WORD *)(v208 + 2) = *(_WORD *)(v208 + 2) & 0xF003 | (4 * ((*(_WORD *)(v202 + 2) >> 4) & 0x3FF));
              if ( !v319 )
                goto LABEL_215;
              v285 = sub_986580((__int64)v347);
              v382 = 257;
              v286 = v305;
              LOWORD(v286) = 0;
              v375 = v186;
              v374 = (__m128i *)v334;
              v288 = sub_2673A60(v200, v202, (__int64 *)&v374, 2, (__int64)&v378, v287, v285 + 24, v286);
              v289 = v288;
              v378 = (const char *)v363;
              if ( v363 )
              {
                v290 = v288 + 48;
                sub_266EF50((__int64 *)&v378);
                if ( (const char **)(v289 + 48) != &v378 )
                {
                  v292 = *(_QWORD *)(v289 + 48);
                  if ( v292 )
LABEL_349:
                    sub_B91220(v290, v292);
                  v293 = (unsigned __int8 *)v378;
                  *(_QWORD *)(v289 + 48) = v378;
                  if ( v293 )
                    sub_B976B0((__int64)&v378, v293, v290);
                  goto LABEL_322;
                }
                if ( v378 )
                  sub_B91220((__int64)&v378, (__int64)v378);
              }
              else
              {
                v290 = v288 + 48;
                if ( (const char **)(v288 + 48) != &v378 )
                {
                  v292 = *(_QWORD *)(v288 + 48);
                  if ( v292 )
                    goto LABEL_349;
                }
              }
LABEL_322:
              if ( !*(_BYTE *)v202 )
                *(_WORD *)(v289 + 2) = *(_WORD *)(v289 + 2) & 0xF003 | (4 * ((*(_WORD *)(v202 + 2) >> 4) & 0x3FF));
LABEL_215:
              if ( v372 )
                sub_B91220((__int64)&v372, (__int64)v372);
              if ( v368.m128i_i64[1] )
                sub_B91220((__int64)&v368.m128i_i64[1], v368.m128i_i64[1]);
              if ( v363 )
                sub_B91220((__int64)&v363, (__int64)v363);
              v316 += 2;
              if ( v306 == v316 )
              {
                v306 = v385;
                goto LABEL_223;
              }
            }
            if ( !v169 )
              BUG();
LABEL_170:
            v170 = (_QWORD *)(v166 - 24);
            if ( (unsigned int)*(unsigned __int8 *)(v166 - 24) - 30 <= 0xA )
              goto LABEL_171;
            goto LABEL_330;
          }
LABEL_223:
          if ( v306 != &v387 )
            _libc_free((unsigned __int64)v306);
LABEL_225:
          if ( !v400.m128i_i8[4] )
            _libc_free((unsigned __int64)v398);
          goto LABEL_227;
        }
      }
    }
    v44 = 1;
    while ( v55 != -4096 )
    {
      v51 = (unsigned int)(v44 + 1);
      v54 = (v52 - 1) & (v44 + v54);
      v44 = v53 + 8LL * v54;
      v55 = *(_QWORD *)v44;
      if ( v50 == *(_QWORD *)v44 )
        goto LABEL_54;
      v44 = (unsigned int)v51;
    }
    goto LABEL_44;
  }
  v4 = *(_QWORD *)(a1 + 280);
  if ( v4 != v4 + 8LL * *(unsigned int *)(a1 + 288) )
  {
    v5 = *(_QWORD *)(a2 + 208);
    v353 = (__int64 *)(v4 + 8LL * *(unsigned int *)(a1 + 288));
    for ( j = *(__int64 **)(a1 + 280); v353 != j; ++j )
    {
      v8 = (unsigned __int8 *)*j;
      if ( *j )
      {
        if ( (unsigned __int8)(*v8 - 34) > 0x33u )
          goto LABEL_14;
        v9 = 0x8000000000041LL;
        if ( !_bittest64(&v9, (unsigned int)*v8 - 34) )
          goto LABEL_14;
        v10 = *((_QWORD *)v8 - 4);
        if ( v10 )
        {
          if ( *(_BYTE *)v10 )
          {
            v10 = 0;
          }
          else if ( *(_QWORD *)(v10 + 24) != *((_QWORD *)v8 + 10) )
          {
            v10 = 0;
          }
        }
        v11 = *(_DWORD *)(v5 + 34968);
        v12 = *(_QWORD *)(v5 + 34952);
        if ( !v11 )
          goto LABEL_14;
        v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v14 = (__int64 *)(v12 + 8LL * v13);
        v15 = *v14;
        if ( v10 != *v14 )
        {
          v74 = 1;
          while ( v15 != -4096 )
          {
            v75 = v74 + 1;
            v76 = (v11 - 1) & (v13 + v74);
            v13 = v76;
            v14 = (__int64 *)(v12 + 8 * v76);
            v15 = *v14;
            if ( v10 == *v14 )
              goto LABEL_13;
            v74 = v75;
          }
LABEL_14:
          if ( *(_QWORD *)(a2 + 4392) )
          {
            v16 = sub_B43CB0(*j);
            v337 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 4392))(*(_QWORD *)(a2 + 4400), v16);
            v348 = *v337;
            v17 = sub_B2BE50(*v337);
            if ( sub_B6EA50(v17)
              || (v294 = sub_B2BE50(v348),
                  v295 = sub_B6F970(v294),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v295 + 48LL))(v295)) )
            {
              sub_B178C0((__int64)&v385, *(_QWORD *)(a2 + 4408), (__int64)"OMP121", 6, (__int64)v8);
              sub_B18290((__int64)&v385, "Value has potential side effects preventing SPMD-mode execution", 0x3Fu);
              if ( (unsigned __int8)(*v8 - 34) <= 0x33u )
              {
                v18 = 0x8000000000041LL;
                if ( _bittest64(&v18, (unsigned int)*v8 - 34) )
                  sub_B18290(
                    (__int64)&v385,
                    ". Add `[[omp::assume(\"ompx_spmd_amenable\")]]` to the called function to override",
                    0x50u);
              }
              sub_B18290((__int64)&v385, ".", 1u);
              v22 = _mm_loadu_si128(&v388);
              v23 = _mm_load_si128(&v390);
              v24 = _mm_load_si128(&v391);
              LODWORD(v398) = v386;
              v400 = v22;
              BYTE4(v398) = BYTE4(v386);
              v402 = v23;
              v399 = v387;
              v403 = v24;
              v397 = (const char *)&unk_49D9D40;
              v401 = v389;
              v404[0] = v405;
              v404[1] = 0x400000000LL;
              if ( v393 )
                sub_26781A0((__int64)v404, (__int64)v392, v19, v20, v21, (__int64)v404);
              v405[320] = v394;
              v406 = v395;
              v407 = v396;
              v397 = (const char *)&unk_49D9DE8;
              sub_B18290((__int64)&v397, " [", 2u);
              sub_B18290((__int64)&v397, "OMP121", 6u);
              sub_B18290((__int64)&v397, "]", 1u);
              sub_23FE290((__int64)&v378, (__int64)&v397, v25, v26, v27, v28);
              v378 = (const char *)&unk_49D9DE8;
              v384 = v407;
              v397 = (const char *)&unk_49D9D40;
              sub_23FD590((__int64)v404);
              v385 = (__int64 *)&unk_49D9D40;
              sub_23FD590((__int64)v392);
              sub_1049740(v337, (__int64)&v378);
              v378 = (const char *)&unk_49D9D40;
              sub_23FD590((__int64)v383);
            }
          }
          continue;
        }
LABEL_13:
        if ( v14 == (__int64 *)(v12 + 8LL * v11) )
          goto LABEL_14;
      }
    }
  }
  return v314;
}
