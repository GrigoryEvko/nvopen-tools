// Function: sub_309A610
// Address: 0x309a610
//
__int64 __fastcall sub_309A610(int a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 *a6, __int64 *a7)
{
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r11d
  _QWORD *v11; // rdx
  unsigned int v12; // r8d
  _QWORD *v13; // rax
  void *v14; // rdi
  __int64 *v15; // rbx
  unsigned int v16; // esi
  int v17; // r11d
  _QWORD *v18; // rdx
  unsigned int v19; // r8d
  _QWORD *v20; // rax
  void *v21; // rdi
  __int64 *v22; // rbx
  int v23; // ebx
  const char *v24; // r13
  char v25; // dl
  const char *v26; // rax
  size_t v27; // r8
  char **v28; // rdx
  bool v29; // al
  char *v30; // rdi
  __int64 v31; // rax
  __m128i si128; // xmm0
  char *v33; // rdx
  char *v34; // rcx
  __int64 v35; // r15
  char v36; // bl
  char v37; // al
  __int64 v38; // r13
  char *v39; // rdx
  const char *v40; // r10
  size_t v41; // r9
  char *v42; // rax
  __m128i *v43; // rdx
  const char *v44; // rax
  const void *v45; // r9
  size_t v46; // r8
  char **v47; // rdx
  __int64 v48; // rax
  const char *v49; // rax
  const void *v50; // r9
  size_t v51; // r8
  char **v52; // rdx
  __int64 v53; // rax
  char *v54; // rax
  char **v55; // rdi
  const char *v56; // rax
  const void *v57; // r9
  size_t v58; // r8
  char **v59; // rdx
  __int64 v60; // rax
  char *v61; // rax
  char **v62; // rdi
  const char *v63; // rax
  const void *v64; // r9
  size_t v65; // r8
  char **v66; // rdx
  __int64 v67; // rax
  char **v68; // r12
  unsigned int v69; // eax
  _QWORD **v70; // r14
  _QWORD **k; // r13
  __int64 v72; // rax
  _QWORD *v73; // rbx
  unsigned __int64 v74; // r15
  __int64 v75; // rdi
  const char *v76; // rax
  const void *v77; // r9
  size_t v78; // r8
  char **v79; // rdx
  __int64 v80; // rax
  unsigned int v81; // eax
  _QWORD *v82; // rbx
  _QWORD *v83; // r13
  __int64 v84; // rdi
  unsigned int v85; // eax
  _QWORD **v86; // r14
  _QWORD **m; // r13
  __int64 v88; // rax
  _QWORD *v89; // rbx
  unsigned __int64 v90; // r15
  __int64 v91; // rdi
  unsigned int v92; // eax
  _QWORD *v93; // rbx
  _QWORD *v94; // r13
  __int64 v95; // rdi
  __int64 v96; // rbx
  char *v97; // r13
  char *v99; // rax
  char **v100; // rdi
  const char *v101; // rax
  const void *v102; // r9
  size_t v103; // r8
  char **v104; // rdx
  __int64 v105; // rax
  char *v106; // rax
  const char *v107; // r13
  __m128i v108; // kr00_16
  _QWORD *v109; // r13
  __int64 v110; // rax
  char *v111; // rdi
  __int64 v112; // r13
  __int64 v113; // rax
  __int64 v114; // rdi
  __int64 v115; // r13
  __int64 v116; // rax
  __int64 v117; // rdi
  __int64 v118; // r13
  __int64 v119; // rax
  __int64 v120; // rdi
  __int64 v121; // rdi
  char *v122; // rax
  const char *v123; // r13
  __int64 v124; // rax
  void *v125; // r14
  __int64 v126; // r13
  char *v127; // rax
  __int64 (__fastcall **v128)(); // rax
  char *v129; // r13
  size_t v130; // rdx
  __m128i *v131; // rsi
  int v132; // eax
  _QWORD *v133; // rax
  __int64 v134; // rdi
  int v135; // eax
  _QWORD *v136; // rax
  __int64 v137; // rdi
  unsigned int v138; // ecx
  void *v139; // r8
  int v140; // edi
  _QWORD *v141; // rsi
  char *v142; // rax
  char **v143; // rdi
  char *v144; // rax
  char **v145; // rdi
  char *v146; // rsi
  __int64 v147; // r14
  unsigned int v148; // eax
  __int64 v149; // r13
  __int64 i; // rbx
  __int64 v151; // rax
  unsigned __int8 *v152; // r15
  const char *v153; // rdx
  const char *v154; // r9
  size_t v155; // r10
  const char *v156; // rax
  __m128i *v157; // rdx
  __m128i *v158; // rsi
  char *v159; // rax
  char **v160; // rdi
  __int64 *v161; // rax
  __int64 **v162; // r13
  __int64 v163; // rbx
  _QWORD *v164; // rax
  __int64 v165; // r12
  __int64 v166; // rax
  __int64 v167; // rax
  __int64 v168; // rax
  __int64 v169; // rax
  const char *v170; // r13
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // rax
  unsigned __int64 v174; // r8
  __int64 v175; // r13
  __int64 v176; // rbx
  _QWORD *v177; // rdi
  unsigned __int64 v178; // r8
  __int64 v179; // r13
  __int64 v180; // rbx
  _QWORD *v181; // rdi
  __int64 j; // rbx
  __int16 v183; // ax
  char *v184; // rax
  char **v185; // rdi
  char *v186; // rdi
  __int64 *v187; // rax
  __m128i *v188; // rsi
  int v189; // eax
  const char *v190; // rdx
  const char *v191; // r10
  size_t v192; // r9
  const char *v193; // rax
  __m128i *v194; // rdx
  __m128i *v195; // r9
  char v196; // al
  __int64 v197; // rcx
  __int64 v198; // r8
  int v199; // edi
  _QWORD *v200; // rsi
  int v201; // edi
  unsigned int v202; // ecx
  void *v203; // r8
  int v204; // edi
  __int64 v205; // rcx
  void *v206; // r8
  char *v207; // r12
  size_t v208; // rax
  __int64 v209; // rax
  __m128i *v210; // rdi
  __int64 v211; // rdx
  const char *v212; // rdx
  const char *v213; // r10
  size_t v214; // r9
  const char *v215; // rax
  __m128i *v216; // rdx
  __int64 *v217; // rax
  __int64 *v218; // rax
  __int64 *v219; // rax
  __int64 v220; // rax
  __m128i *v221; // rdi
  __int64 v222; // rax
  __m128i *v223; // rdi
  __int64 v224; // rax
  __m128i *v225; // rdi
  char v227; // [rsp+2Dh] [rbp-1233h]
  char v228; // [rsp+2Eh] [rbp-1232h]
  char v229; // [rsp+2Fh] [rbp-1231h]
  char v230; // [rsp+38h] [rbp-1228h]
  size_t v231; // [rsp+38h] [rbp-1228h]
  size_t v232; // [rsp+38h] [rbp-1228h]
  size_t v233; // [rsp+38h] [rbp-1228h]
  char v234; // [rsp+40h] [rbp-1220h]
  const char *v235; // [rsp+40h] [rbp-1220h]
  const char *v236; // [rsp+40h] [rbp-1220h]
  const char *v237; // [rsp+40h] [rbp-1220h]
  size_t na; // [rsp+48h] [rbp-1218h]
  size_t nb; // [rsp+48h] [rbp-1218h]
  size_t nc; // [rsp+48h] [rbp-1218h]
  size_t n; // [rsp+48h] [rbp-1218h]
  size_t nd; // [rsp+48h] [rbp-1218h]
  size_t ne; // [rsp+48h] [rbp-1218h]
  size_t nf; // [rsp+48h] [rbp-1218h]
  size_t ng; // [rsp+48h] [rbp-1218h]
  size_t nh; // [rsp+48h] [rbp-1218h]
  char v250; // [rsp+70h] [rbp-11F0h]
  __int128 v251; // [rsp+80h] [rbp-11E0h]
  __int64 v252; // [rsp+98h] [rbp-11C8h]
  __int64 v253; // [rsp+A0h] [rbp-11C0h]
  __int64 v254; // [rsp+A8h] [rbp-11B8h]
  __int64 src; // [rsp+B0h] [rbp-11B0h]
  const char *v256; // [rsp+B8h] [rbp-11A8h]
  _BYTE *v258; // [rsp+C8h] [rbp-1198h]
  __int64 v259; // [rsp+D8h] [rbp-1188h] BYREF
  unsigned int v260; // [rsp+E0h] [rbp-1180h] BYREF
  __int64 (__fastcall **v261)(); // [rsp+E8h] [rbp-1178h]
  void *v262; // [rsp+F0h] [rbp-1170h] BYREF
  size_t v263; // [rsp+F8h] [rbp-1168h]
  _BYTE v264[16]; // [rsp+100h] [rbp-1160h] BYREF
  char *s; // [rsp+110h] [rbp-1150h] BYREF
  __int64 v266; // [rsp+118h] [rbp-1148h]
  _QWORD v267[2]; // [rsp+120h] [rbp-1140h] BYREF
  const char *v268[2]; // [rsp+130h] [rbp-1130h] BYREF
  __int64 v269; // [rsp+140h] [rbp-1120h] BYREF
  char *v270; // [rsp+150h] [rbp-1110h] BYREF
  __int64 v271; // [rsp+158h] [rbp-1108h]
  char *v272; // [rsp+160h] [rbp-1100h]
  __int64 v273; // [rsp+168h] [rbp-10F8h]
  __int64 v274; // [rsp+170h] [rbp-10F0h]
  __int64 v275; // [rsp+180h] [rbp-10E0h] BYREF
  _QWORD *v276; // [rsp+188h] [rbp-10D8h]
  int v277; // [rsp+190h] [rbp-10D0h]
  int v278; // [rsp+194h] [rbp-10CCh]
  unsigned int v279; // [rsp+198h] [rbp-10C8h]
  __int64 v280; // [rsp+1A8h] [rbp-10B8h]
  unsigned int v281; // [rsp+1B8h] [rbp-10A8h]
  __int64 v282; // [rsp+1C8h] [rbp-1098h]
  unsigned int v283; // [rsp+1D8h] [rbp-1088h]
  __int64 v284; // [rsp+1E0h] [rbp-1080h] BYREF
  _QWORD *v285; // [rsp+1E8h] [rbp-1078h]
  int v286; // [rsp+1F0h] [rbp-1070h]
  int v287; // [rsp+1F4h] [rbp-106Ch]
  unsigned int v288; // [rsp+1F8h] [rbp-1068h]
  __int64 v289; // [rsp+208h] [rbp-1058h]
  unsigned int v290; // [rsp+218h] [rbp-1048h]
  __int64 v291; // [rsp+228h] [rbp-1038h]
  unsigned int v292; // [rsp+238h] [rbp-1028h]
  __m128i v293[14]; // [rsp+240h] [rbp-1020h] BYREF
  bool v294; // [rsp+328h] [rbp-F38h]
  char *v295[2]; // [rsp+340h] [rbp-F20h] BYREF
  char *v296; // [rsp+350h] [rbp-F10h] BYREF
  __int64 v297; // [rsp+358h] [rbp-F08h]
  __int64 v298; // [rsp+360h] [rbp-F00h]
  unsigned __int64 v299; // [rsp+368h] [rbp-EF8h] BYREF
  __int64 v300; // [rsp+370h] [rbp-EF0h]
  __int64 v301; // [rsp+378h] [rbp-EE8h]
  __int64 v302; // [rsp+468h] [rbp-DF8h]
  __int64 v303; // [rsp+470h] [rbp-DF0h]
  __int64 v304; // [rsp+478h] [rbp-DE8h]
  int v305; // [rsp+480h] [rbp-DE0h]
  __int64 *v306; // [rsp+488h] [rbp-DD8h]
  __int64 v307; // [rsp+490h] [rbp-DD0h]
  __int64 v308; // [rsp+498h] [rbp-DC8h]
  __int64 v309; // [rsp+4A0h] [rbp-DC0h]
  int v310; // [rsp+4A8h] [rbp-DB8h]
  __int64 v311; // [rsp+4B0h] [rbp-DB0h]
  __int64 v312; // [rsp+4B8h] [rbp-DA8h] BYREF
  __int64 *v313; // [rsp+4C0h] [rbp-DA0h]
  __int64 v314; // [rsp+4C8h] [rbp-D98h]
  __int64 v315; // [rsp+4D0h] [rbp-D90h]
  __int64 v316; // [rsp+4D8h] [rbp-D88h]
  __int64 v317; // [rsp+4E0h] [rbp-D80h]
  __int64 v318; // [rsp+4E8h] [rbp-D78h]
  __int64 v319; // [rsp+4F0h] [rbp-D70h] BYREF
  __int64 v320; // [rsp+4F8h] [rbp-D68h]
  __int64 v321; // [rsp+500h] [rbp-D60h]
  __int64 v322; // [rsp+508h] [rbp-D58h]
  int v323; // [rsp+510h] [rbp-D50h]
  __int64 v324; // [rsp+518h] [rbp-D48h]
  _BYTE *v325; // [rsp+520h] [rbp-D40h]
  __int64 v326; // [rsp+528h] [rbp-D38h]
  int v327; // [rsp+530h] [rbp-D30h]
  char v328; // [rsp+534h] [rbp-D2Ch]
  _BYTE v329[264]; // [rsp+538h] [rbp-D28h] BYREF
  __m128i v330[9]; // [rsp+640h] [rbp-C20h] BYREF
  char v331; // [rsp+6D8h] [rbp-B88h]
  __m128i v332[146]; // [rsp+940h] [rbp-920h] BYREF

  v262 = v264;
  v263 = 0;
  v264[0] = 0;
  v270 = 0;
  v271 = 0;
  v272 = 0;
  v273 = 0;
  v274 = 0;
  sub_BBB200((__int64)&v275);
  sub_BBB1A0((__int64)&v284);
  v331 = 0;
  sub_23A0D00((__int64)v295);
  sub_2356B40(
    (__int64)v332,
    0,
    (__int64)v330,
    0,
    v8,
    v9,
    *(_OWORD *)&_mm_loadu_si128((const __m128i *)v295),
    (__int64)v296,
    v297);
  if ( v331 )
  {
    v331 = 0;
    sub_23C66F0((unsigned __int64 *)v330);
  }
  sub_2362620((__int64)v332, (__int64)&v284);
  sub_23635D0(v332, (__int64)&v275);
  if ( !v288 )
  {
    ++v284;
    goto LABEL_434;
  }
  v10 = 1;
  v11 = 0;
  v12 = (v288 - 1) & (((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4));
  v13 = &v285[2 * v12];
  v14 = (void *)*v13;
  if ( (_UNKNOWN *)*v13 == &unk_4F82418 )
    goto LABEL_5;
  while ( 1 )
  {
    if ( v14 == (void *)-4096LL )
    {
      if ( !v11 )
        v11 = v13;
      ++v284;
      v135 = v286 + 1;
      if ( 4 * (v286 + 1) < 3 * v288 )
      {
        if ( v288 - v287 - v135 > v288 >> 3 )
        {
LABEL_263:
          v286 = v135;
          if ( *v11 != -4096 )
            --v287;
          v11[1] = 0;
          *v11 = &unk_4F82418;
          v15 = v11 + 1;
          goto LABEL_266;
        }
        sub_23622E0((__int64)&v284, v288);
        if ( v288 )
        {
          v204 = 1;
          v200 = 0;
          LODWORD(v205) = (v288 - 1) & (((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4));
          v135 = v286 + 1;
          v11 = &v285[2 * (unsigned int)v205];
          v206 = (void *)*v11;
          if ( (_UNKNOWN *)*v11 == &unk_4F82418 )
            goto LABEL_263;
          while ( v206 != (void *)-4096LL )
          {
            if ( !v200 && v206 == (void *)-8192LL )
              v200 = v11;
            v205 = (v288 - 1) & ((_DWORD)v205 + v204);
            v11 = &v285[2 * v205];
            v206 = (void *)*v11;
            if ( (_UNKNOWN *)*v11 == &unk_4F82418 )
              goto LABEL_263;
            ++v204;
          }
          goto LABEL_438;
        }
        goto LABEL_538;
      }
LABEL_434:
      sub_23622E0((__int64)&v284, 2 * v288);
      if ( v288 )
      {
        LODWORD(v197) = (v288 - 1) & (((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4));
        v135 = v286 + 1;
        v11 = &v285[2 * (unsigned int)v197];
        v198 = *v11;
        if ( (_UNKNOWN *)*v11 == &unk_4F82418 )
          goto LABEL_263;
        v199 = 1;
        v200 = 0;
        while ( v198 != -4096 )
        {
          if ( !v200 && v198 == -8192 )
            v200 = v11;
          v197 = (v288 - 1) & ((_DWORD)v197 + v199);
          v11 = &v285[2 * v197];
          v198 = *v11;
          if ( (_UNKNOWN *)*v11 == &unk_4F82418 )
            goto LABEL_263;
          ++v199;
        }
LABEL_438:
        if ( v200 )
          v11 = v200;
        goto LABEL_263;
      }
LABEL_538:
      ++v286;
      BUG();
    }
    if ( v14 != (void *)-8192LL || v11 )
      v13 = v11;
    v12 = (v288 - 1) & (v10 + v12);
    v14 = (void *)v285[2 * v12];
    if ( v14 == &unk_4F82418 )
      break;
    ++v10;
    v11 = v13;
    v13 = &v285[2 * v12];
  }
  v13 = &v285[2 * v12];
LABEL_5:
  v15 = v13 + 1;
  if ( v13[1] )
  {
LABEL_6:
    v16 = v279;
    if ( v279 )
      goto LABEL_7;
LABEL_270:
    ++v275;
LABEL_271:
    sub_2275E10((__int64)&v275, 2 * v16);
    if ( v279 )
    {
      v138 = (v279 - 1) & (((unsigned int)&unk_4F82410 >> 9) ^ ((unsigned int)&unk_4F82410 >> 4));
      v132 = v277 + 1;
      v18 = &v276[2 * v138];
      v139 = (void *)*v18;
      if ( (_UNKNOWN *)*v18 != &unk_4F82410 )
      {
        v140 = 1;
        v141 = 0;
        while ( v139 != (void *)-4096LL )
        {
          if ( !v141 && v139 == (void *)-8192LL )
            v141 = v18;
          v138 = (v279 - 1) & (v140 + v138);
          v18 = &v276[2 * v138];
          v139 = (void *)*v18;
          if ( (_UNKNOWN *)*v18 == &unk_4F82410 )
            goto LABEL_250;
          ++v140;
        }
LABEL_275:
        if ( v141 )
          v18 = v141;
      }
LABEL_250:
      v277 = v132;
      if ( *v18 != -4096 )
        --v278;
      v18[1] = 0;
      *v18 = &unk_4F82410;
      v22 = v18 + 1;
LABEL_253:
      v133 = (_QWORD *)sub_22077B0(0x10u);
      if ( v133 )
      {
        *v133 = &unk_4A157B8;
        v133[1] = &v284;
      }
      v134 = *v22;
      *v22 = (__int64)v133;
      if ( v134 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v134 + 8LL))(v134);
      goto LABEL_9;
    }
LABEL_537:
    ++v277;
    BUG();
  }
LABEL_266:
  v136 = (_QWORD *)sub_22077B0(0x10u);
  if ( v136 )
  {
    v136[1] = &v275;
    *v136 = &unk_4A156F8;
  }
  v137 = *v15;
  *v15 = (__int64)v136;
  if ( !v137 )
    goto LABEL_6;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v137 + 8LL))(v137);
  v16 = v279;
  if ( !v279 )
    goto LABEL_270;
LABEL_7:
  v17 = 1;
  v18 = 0;
  v19 = (v16 - 1) & (((unsigned int)&unk_4F82410 >> 9) ^ ((unsigned int)&unk_4F82410 >> 4));
  v20 = &v276[2 * v19];
  v21 = (void *)*v20;
  if ( (_UNKNOWN *)*v20 == &unk_4F82410 )
    goto LABEL_8;
  while ( 1 )
  {
    if ( v21 == (void *)-4096LL )
    {
      if ( !v18 )
        v18 = v20;
      ++v275;
      v132 = v277 + 1;
      if ( 4 * (v277 + 1) >= 3 * v16 )
        goto LABEL_271;
      if ( v16 - v278 - v132 > v16 >> 3 )
        goto LABEL_250;
      sub_2275E10((__int64)&v275, v16);
      if ( v279 )
      {
        v201 = 1;
        v141 = 0;
        v202 = (v279 - 1) & (((unsigned int)&unk_4F82410 >> 9) ^ ((unsigned int)&unk_4F82410 >> 4));
        v132 = v277 + 1;
        v18 = &v276[2 * v202];
        v203 = (void *)*v18;
        if ( (_UNKNOWN *)*v18 != &unk_4F82410 )
        {
          while ( v203 != (void *)-4096LL )
          {
            if ( v203 == (void *)-8192LL && !v141 )
              v141 = v18;
            v202 = (v279 - 1) & (v201 + v202);
            v18 = &v276[2 * v202];
            v203 = (void *)*v18;
            if ( (_UNKNOWN *)*v18 == &unk_4F82410 )
              goto LABEL_250;
            ++v201;
          }
          goto LABEL_275;
        }
        goto LABEL_250;
      }
      goto LABEL_537;
    }
    if ( v21 != (void *)-8192LL || v18 )
      v20 = v18;
    v19 = (v16 - 1) & (v17 + v19);
    v21 = (void *)v276[2 * v19];
    if ( v21 == &unk_4F82410 )
      break;
    ++v17;
    v18 = v20;
    v20 = &v276[2 * v19];
  }
  v20 = &v276[2 * v19];
LABEL_8:
  v22 = v20 + 1;
  if ( !v20[1] )
    goto LABEL_253;
LABEL_9:
  memset(v330, 0, 0x78u);
  sub_CCBB10(v293, v330);
  v293[0].m128i_i32[1] = 2;
  if ( a1 <= 0 )
  {
    v250 = 0;
    v228 = 0;
    v227 = 0;
    v253 = 0;
    v252 = 0;
    src = 0;
    v254 = 0;
    v251 = 0u;
    goto LABEL_289;
  }
  v251 = 0u;
  v23 = 0;
  v252 = 0;
  v254 = 0;
  v253 = 0;
  src = 0;
  v227 = 0;
  v228 = 0;
  v250 = 0;
  v229 = 0;
  v234 = 0;
  while ( 2 )
  {
    v24 = *(const char **)(a2 + 8LL * v23);
    if ( !memcmp(v24, "-arch=compute_", 0xEu) )
    {
      v295[0] = (char *)&v296;
      v26 = (const char *)strlen(v24 + 6);
      v268[0] = v26;
      v27 = (size_t)v26;
      if ( (unsigned __int64)v26 > 0xF )
      {
        na = (size_t)v26;
        v54 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v27 = na;
        v295[0] = v54;
        v55 = (char **)v54;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v26 == (const char *)1 )
        {
          LOBYTE(v296) = v24[6];
          v28 = &v296;
          goto LABEL_31;
        }
        if ( !v26 )
        {
          v28 = &v296;
          goto LABEL_31;
        }
        v55 = &v296;
      }
      memcpy(v55, v24 + 6, v27);
      v26 = v268[0];
      v28 = (char **)v295[0];
LABEL_31:
      v295[1] = (char *)v26;
      v26[(_QWORD)v28] = 0;
      v293[0].m128i_i32[0] = 10 * sub_1CFBEC0(v295[0]);
      v29 = sub_1CFBF00(v295[0]);
      v30 = v295[0];
      v294 = v29;
      if ( (char **)v295[0] == &v296 )
        goto LABEL_33;
      goto LABEL_32;
    }
    if ( !memcmp(*(const void **)(a2 + 8LL * v23), "-host-ref-ek=", 0xDu) )
    {
      v295[0] = (char *)&v296;
      v44 = (const char *)strlen(v24 + 13);
      v45 = v24 + 13;
      v268[0] = v44;
      v46 = (size_t)v44;
      if ( (unsigned __int64)v44 > 0xF )
      {
        nb = (size_t)v44;
        v61 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v45 = v24 + 13;
        v46 = nb;
        v295[0] = v61;
        v62 = (char **)v61;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v44 == (const char *)1 )
        {
          LOBYTE(v296) = v24[13];
          v47 = &v296;
          goto LABEL_58;
        }
        if ( !v44 )
        {
          v47 = &v296;
          goto LABEL_58;
        }
        v62 = &v296;
      }
      memcpy(v62, v45, v46);
      v44 = v268[0];
      v47 = (char **)v295[0];
LABEL_58:
      v295[1] = (char *)v44;
      v44[(_QWORD)v47] = 0;
      v48 = sub_1682150(v295[0]);
      v30 = v295[0];
      src = v48;
      if ( (char **)v295[0] == &v296 )
        goto LABEL_33;
      goto LABEL_32;
    }
    if ( !memcmp(*(const void **)(a2 + 8LL * v23), "-host-ref-ik=", 0xDu) )
    {
      v295[0] = (char *)&v296;
      v49 = (const char *)strlen(v24 + 13);
      v50 = v24 + 13;
      v268[0] = v49;
      v51 = (size_t)v49;
      if ( (unsigned __int64)v49 > 0xF )
      {
        nc = (size_t)v49;
        v99 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v50 = v24 + 13;
        v51 = nc;
        v295[0] = v99;
        v100 = (char **)v99;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v49 == (const char *)1 )
        {
          LOBYTE(v296) = v24[13];
          v52 = &v296;
          goto LABEL_63;
        }
        if ( !v49 )
        {
          v52 = &v296;
          goto LABEL_63;
        }
        v100 = &v296;
      }
      memcpy(v100, v50, v51);
      v49 = v268[0];
      v52 = (char **)v295[0];
LABEL_63:
      v295[1] = (char *)v49;
      v49[(_QWORD)v52] = 0;
      v53 = sub_1682150(v295[0]);
      v30 = v295[0];
      v253 = v53;
      if ( (char **)v295[0] == &v296 )
        goto LABEL_33;
      goto LABEL_32;
    }
    if ( !memcmp(*(const void **)(a2 + 8LL * v23), "-host-ref-ec=", 0xDu) )
    {
      v295[0] = (char *)&v296;
      v56 = (const char *)strlen(v24 + 13);
      v57 = v24 + 13;
      v268[0] = v56;
      v58 = (size_t)v56;
      if ( (unsigned __int64)v56 > 0xF )
      {
        nd = (size_t)v56;
        v142 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v57 = v24 + 13;
        v58 = nd;
        v295[0] = v142;
        v143 = (char **)v142;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v56 == (const char *)1 )
        {
          LOBYTE(v296) = v24[13];
          v59 = &v296;
          goto LABEL_72;
        }
        if ( !v56 )
        {
          v59 = &v296;
          goto LABEL_72;
        }
        v143 = &v296;
      }
      memcpy(v143, v57, v58);
      v56 = v268[0];
      v59 = (char **)v295[0];
LABEL_72:
      v295[1] = (char *)v56;
      v56[(_QWORD)v59] = 0;
      v60 = sub_1682150(v295[0]);
      v30 = v295[0];
      v254 = v60;
      if ( (char **)v295[0] == &v296 )
        goto LABEL_33;
      goto LABEL_32;
    }
    if ( !memcmp(*(const void **)(a2 + 8LL * v23), "-host-ref-ic=", 0xDu) )
    {
      v295[0] = (char *)&v296;
      v63 = (const char *)strlen(v24 + 13);
      v64 = v24 + 13;
      v268[0] = v63;
      v65 = (size_t)v63;
      if ( (unsigned __int64)v63 > 0xF )
      {
        ne = (size_t)v63;
        v144 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v64 = v24 + 13;
        v65 = ne;
        v295[0] = v144;
        v145 = (char **)v144;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v63 == (const char *)1 )
        {
          LOBYTE(v296) = v24[13];
          v66 = &v296;
          goto LABEL_81;
        }
        if ( !v63 )
        {
          v66 = &v296;
          goto LABEL_81;
        }
        v145 = &v296;
      }
      memcpy(v145, v64, v65);
      v63 = v268[0];
      v66 = (char **)v295[0];
LABEL_81:
      v295[1] = (char *)v63;
      v63[(_QWORD)v66] = 0;
      v67 = sub_1682150(v295[0]);
      v30 = v295[0];
      v252 = v67;
      if ( (char **)v295[0] == &v296 )
        goto LABEL_33;
      goto LABEL_32;
    }
    if ( !memcmp(*(const void **)(a2 + 8LL * v23), "-host-ref-eg=", 0xDu) )
    {
      v295[0] = (char *)&v296;
      v101 = (const char *)strlen(v24 + 13);
      v102 = v24 + 13;
      v268[0] = v101;
      v103 = (size_t)v101;
      if ( (unsigned __int64)v101 > 0xF )
      {
        ng = (size_t)v101;
        v184 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v102 = v24 + 13;
        v103 = ng;
        v295[0] = v184;
        v185 = (char **)v184;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v101 == (const char *)1 )
        {
          LOBYTE(v296) = v24[13];
          v104 = &v296;
          goto LABEL_160;
        }
        if ( !v101 )
        {
          v104 = &v296;
          goto LABEL_160;
        }
        v185 = &v296;
      }
      memcpy(v185, v102, v103);
      v101 = v268[0];
      v104 = (char **)v295[0];
LABEL_160:
      v295[1] = (char *)v101;
      v101[(_QWORD)v104] = 0;
      v105 = sub_1682150(v295[0]);
      v30 = v295[0];
      *((_QWORD *)&v251 + 1) = v105;
      if ( (char **)v295[0] == &v296 )
        goto LABEL_33;
LABEL_32:
      j_j___libc_free_0((unsigned __int64)v30);
      goto LABEL_33;
    }
    if ( !memcmp(*(const void **)(a2 + 8LL * v23), "-host-ref-ig=", 0xDu) )
    {
      v295[0] = (char *)&v296;
      v76 = (const char *)strlen(v24 + 13);
      v77 = v24 + 13;
      v268[0] = v76;
      v78 = (size_t)v76;
      if ( (unsigned __int64)v76 > 0xF )
      {
        nf = (size_t)v76;
        v159 = (char *)sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
        v77 = v24 + 13;
        v78 = nf;
        v295[0] = v159;
        v160 = (char **)v159;
        v296 = (char *)v268[0];
      }
      else
      {
        if ( v76 == (const char *)1 )
        {
          LOBYTE(v296) = v24[13];
          v79 = &v296;
          goto LABEL_115;
        }
        if ( !v76 )
        {
          v79 = &v296;
LABEL_115:
          v295[1] = (char *)v76;
          v76[(_QWORD)v79] = 0;
          v80 = sub_1682150(v295[0]);
          v30 = v295[0];
          *(_QWORD *)&v251 = v80;
          if ( (char **)v295[0] == &v296 )
            goto LABEL_33;
          goto LABEL_32;
        }
        v160 = &v296;
      }
      memcpy(v160, v77, v78);
      v76 = v268[0];
      v79 = (char **)v295[0];
      goto LABEL_115;
    }
    if ( !strcmp(*(const char **)(a2 + 8LL * v23), "-has-global-host-info") )
    {
      v228 = 1;
    }
    else if ( !strcmp(*(const char **)(a2 + 8LL * v23), "-optimize-unused-variables") )
    {
      v250 = 1;
    }
    else if ( !strcmp(*(const char **)(a2 + 8LL * v23), "-olto") )
    {
      v207 = *(char **)(a2 + 8LL * v23++ + 8);
      v208 = strlen(v207);
      sub_2241130((unsigned __int64 *)&v262, 0, v263, v207, v208);
    }
    else if ( !strcmp(*(const char **)(a2 + 8LL * v23), "--device-c") )
    {
      v234 = 1;
    }
    else if ( !strcmp(*(const char **)(a2 + 8LL * v23), "--force-device-c") )
    {
      v229 = 1;
    }
    else if ( !strcmp(*(const char **)(a2 + 8LL * v23), "-gen-lto") )
    {
      v230 = 1;
    }
    else if ( !strcmp(*(const char **)(a2 + 8LL * v23), "-link-lto") )
    {
      v230 = 0;
    }
    else
    {
      v25 = v227;
      if ( !strcmp(*(const char **)(a2 + 8LL * v23), "--trace") )
        v25 = 1;
      v227 = v25;
    }
LABEL_33:
    if ( a1 > ++v23 )
      continue;
    break;
  }
  if ( v250 )
  {
    if ( v254 | v252 | (unsigned __int64)v251 | *((_QWORD *)&v251 + 1) )
    {
      v268[0] = (const char *)113;
      v295[0] = (char *)&v296;
      v31 = sub_22409D0((__int64)v295, (unsigned __int64 *)v268, 0);
      v295[0] = (char *)v31;
      v296 = (char *)v268[0];
      *(__m128i *)v31 = _mm_load_si128((const __m128i *)&xmmword_4284F10);
      si128 = _mm_load_si128((const __m128i *)&xmmword_4284F20);
      *(_BYTE *)(v31 + 112) = 115;
      *(__m128i *)(v31 + 16) = si128;
      *(__m128i *)(v31 + 32) = _mm_load_si128((const __m128i *)&xmmword_4284F30);
      *(__m128i *)(v31 + 48) = _mm_load_si128((const __m128i *)&xmmword_4284F40);
      *(__m128i *)(v31 + 64) = _mm_load_si128((const __m128i *)&xmmword_4284F50);
      *(__m128i *)(v31 + 80) = _mm_load_si128((const __m128i *)&xmmword_4284F60);
      *(__m128i *)(v31 + 96) = _mm_load_si128((const __m128i *)&xmmword_4284F70);
      v33 = v295[0];
      v295[1] = (char *)v268[0];
      v295[0][(unsigned __int64)v268[0]] = 0;
      sub_CEB590(v295, 1, (__int64)v33, v34);
      if ( (char **)v295[0] != &v296 )
        j_j___libc_free_0((unsigned __int64)v295[0]);
      v250 = 0;
    }
    else
    {
      v252 = 0;
      v251 = 0u;
      v254 = 0;
    }
  }
  if ( !v230 )
  {
    if ( v234 )
    {
      *a4 = 1;
      if ( *(_QWORD *)(a3 + 32) == a3 + 24 )
        goto LABEL_88;
      v35 = *(_QWORD *)(a3 + 32);
      v36 = 0;
      while ( 2 )
      {
        v38 = v35 - 56;
        if ( !v35 )
          v38 = 0;
        if ( sub_B2FC80(v38) && (*(_BYTE *)(v38 + 33) & 0x20) == 0 && (unsigned __int8)sub_BD3660(v38, 1) )
        {
          v40 = sub_BD5D20(v38);
          v41 = (size_t)v39;
          if ( v40 )
          {
            v295[0] = v39;
            v42 = v39;
            v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
            if ( (unsigned __int64)v39 > 0xF )
            {
              nh = (size_t)v39;
              v256 = v40;
              v209 = sub_22409D0((__int64)v330, (unsigned __int64 *)v295, 0);
              v40 = v256;
              v41 = nh;
              v330[0].m128i_i64[0] = v209;
              v210 = (__m128i *)v209;
              v330[1].m128i_i64[0] = (__int64)v295[0];
            }
            else
            {
              if ( v39 == (char *)1 )
              {
                v330[1].m128i_i8[0] = *v40;
                v43 = &v330[1];
                goto LABEL_417;
              }
              if ( !v39 )
              {
                v43 = &v330[1];
                goto LABEL_417;
              }
              v210 = &v330[1];
            }
            memcpy(v210, v40, v41);
            v42 = v295[0];
            v43 = (__m128i *)v330[0].m128i_i64[0];
LABEL_417:
            v330[0].m128i_i64[1] = (__int64)v42;
            v42[(_QWORD)v43] = 0;
            v188 = (__m128i *)v330[0].m128i_i64[0];
          }
          else
          {
            v330[0].m128i_i64[1] = 0;
            v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
            v188 = &v330[1];
            v330[1].m128i_i8[0] = 0;
          }
          if ( !(unsigned __int8)sub_1681F50(0, v188) )
          {
            if ( (__m128i *)v330[0].m128i_i64[0] != &v330[1] )
              j_j___libc_free_0(v330[0].m128i_u64[0]);
            *a4 = 0;
            if ( v36 )
              goto LABEL_89;
LABEL_88:
            *a4 = 0;
LABEL_89:
            if ( src )
              sub_1688090(src, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
            if ( v253 )
              sub_1688090(v253, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
            if ( v254 )
              sub_1688090(v254, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
            if ( v252 )
              sub_1688090(v252, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
            if ( *((_QWORD *)&v251 + 1) )
              sub_1688090(*((__int64 *)&v251 + 1), (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
            if ( (_QWORD)v251 )
              sub_1688090(v251, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
            LODWORD(v68) = 1;
            sub_CEAF80(a6);
            goto LABEL_102;
          }
          if ( (__m128i *)v330[0].m128i_i64[0] != &v330[1] )
            j_j___libc_free_0(v330[0].m128i_u64[0]);
        }
        v37 = sub_CE9220(v38);
        v35 = *(_QWORD *)(v35 + 8);
        if ( v37 )
          v36 = v37;
        if ( v35 == a3 + 24 )
        {
          if ( !v36 )
            goto LABEL_88;
          if ( *a4 == 1 && !v229 )
            goto LABEL_290;
          goto LABEL_89;
        }
        continue;
      }
    }
    if ( v229 )
      goto LABEL_88;
LABEL_289:
    *a4 = 1;
LABEL_290:
    v146 = "llvm.used";
    v258 = sub_BA8CD0(a3, (__int64)"llvm.used", 9u, 0);
    if ( (src | v253 | *((_QWORD *)&v251 + 1) | (unsigned __int64)v251 | v254 | v252 || v250) && v258 )
    {
      v295[0] = 0;
      v295[1] = 0;
      v296 = 0;
      v147 = *((_QWORD *)v258 - 4);
      v148 = *(_DWORD *)(v147 + 4) & 0x7FFFFFF;
      if ( v148 )
      {
        v149 = v148 - 1;
        for ( i = 0; ; ++i )
        {
          v152 = sub_BD3990(*(unsigned __int8 **)(v147 + 32 * (i - v148)), (__int64)v146);
          if ( *v152 )
          {
            if ( *v152 != 3 )
              goto LABEL_296;
            v189 = *(_DWORD *)(*((_QWORD *)v152 + 1) + 8LL) >> 8;
            if ( v189 != 4 )
            {
              if ( v189 != 1 )
                goto LABEL_296;
              v191 = sub_BD5D20((__int64)v152);
              v192 = (size_t)v190;
              if ( v191 )
              {
                v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
                v193 = v190;
                v268[0] = v190;
                if ( (unsigned __int64)v190 > 0xF )
                {
                  v233 = (size_t)v190;
                  v237 = v191;
                  v224 = sub_22409D0((__int64)v330, (unsigned __int64 *)v268, 0);
                  v191 = v237;
                  v192 = v233;
                  v330[0].m128i_i64[0] = v224;
                  v225 = (__m128i *)v224;
                  v330[1].m128i_i64[0] = (__int64)v268[0];
                }
                else
                {
                  if ( v190 == (const char *)1 )
                  {
                    v330[1].m128i_i8[0] = *v191;
                    v194 = &v330[1];
                    goto LABEL_424;
                  }
                  if ( !v190 )
                  {
                    v194 = &v330[1];
                    goto LABEL_424;
                  }
                  v225 = &v330[1];
                }
                v146 = (char *)v191;
                memcpy(v225, v191, v192);
                v193 = v268[0];
                v194 = (__m128i *)v330[0].m128i_i64[0];
LABEL_424:
                v330[0].m128i_i64[1] = (__int64)v193;
                v193[(_QWORD)v194] = 0;
              }
              else
              {
                v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
                v330[0].m128i_i64[1] = 0;
                v330[1].m128i_i8[0] = 0;
              }
              if ( !v250 )
              {
                v195 = (__m128i *)v330[0].m128i_i64[0];
                if ( !v228 && v251 == 0 )
                  goto LABEL_430;
                v146 = (char *)v330[0].m128i_i64[0];
                if ( (v152[32] & 0xFu) - 7 > 1 )
                {
                  if ( (unsigned __int8)sub_16820A0(*((__int64 *)&v251 + 1), v330[0].m128i_i64[0]) )
                  {
                    v195 = (__m128i *)v330[0].m128i_i64[0];
LABEL_430:
                    if ( v195 != &v330[1] )
                      j_j___libc_free_0((unsigned __int64)v195);
LABEL_296:
                    v146 = v295[1];
                    v151 = *(_QWORD *)(v147 + 32 * (i - (*(_DWORD *)(v147 + 4) & 0x7FFFFFF)));
                    v330[0].m128i_i64[0] = v151;
                    if ( v295[1] == v296 )
                    {
                      sub_262AD50((__int64)v295, v295[1], v330);
                    }
                    else
                    {
                      if ( v295[1] )
                      {
                        *(_QWORD *)v295[1] = v151;
                        v146 = v295[1];
                      }
                      v146 += 8;
                      v295[1] = v146;
                    }
                    goto LABEL_300;
                  }
                }
                else
                {
                  v196 = sub_16820A0(v251, v330[0].m128i_i64[0]);
                  v195 = (__m128i *)v330[0].m128i_i64[0];
                  if ( v196 )
                    goto LABEL_430;
                }
LABEL_478:
                if ( v227 )
                {
                  v217 = sub_223E4D0(qword_4FD4D00, "no reference to variable ");
                  v218 = sub_223E0D0(v217, (const char *)v330[0].m128i_i64[0], v330[0].m128i_i64[1]);
                  v146 = "\n";
                  sub_223E4D0(v218, "\n");
                }
                goto LABEL_466;
              }
LABEL_480:
              if ( !v227 )
                goto LABEL_466;
              v211 = 25;
              v146 = "no reference to variable ";
LABEL_482:
              sub_223E0D0(qword_4FD4D00, v146, v211);
              v219 = sub_223E0D0(qword_4FD4D00, (const char *)v330[0].m128i_i64[0], v330[0].m128i_i64[1]);
              v146 = "\n";
              sub_223E0D0(v219, "\n", 1);
              goto LABEL_466;
            }
            v213 = sub_BD5D20((__int64)v152);
            v214 = (size_t)v212;
            if ( v213 )
            {
              v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
              v215 = v212;
              v268[0] = v212;
              if ( (unsigned __int64)v212 > 0xF )
              {
                v232 = (size_t)v212;
                v236 = v213;
                v222 = sub_22409D0((__int64)v330, (unsigned __int64 *)v268, 0);
                v213 = v236;
                v214 = v232;
                v330[0].m128i_i64[0] = v222;
                v223 = (__m128i *)v222;
                v330[1].m128i_i64[0] = (__int64)v268[0];
              }
              else
              {
                if ( v212 == (const char *)1 )
                {
                  v330[1].m128i_i8[0] = *v213;
                  v216 = &v330[1];
                  goto LABEL_474;
                }
                if ( !v212 )
                {
                  v216 = &v330[1];
                  goto LABEL_474;
                }
                v223 = &v330[1];
              }
              v146 = (char *)v213;
              memcpy(v223, v213, v214);
              v215 = v268[0];
              v216 = (__m128i *)v330[0].m128i_i64[0];
LABEL_474:
              v330[0].m128i_i64[1] = (__int64)v215;
              v215[(_QWORD)v216] = 0;
            }
            else
            {
              v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
              v330[0].m128i_i64[1] = 0;
              v330[1].m128i_i8[0] = 0;
            }
            if ( !v250 )
            {
              v146 = (char *)v330[0].m128i_i64[0];
              if ( (v152[32] & 0xFu) - 7 > 1 )
              {
                if ( (unsigned __int8)sub_16820A0(v254, v330[0].m128i_i64[0]) )
                {
LABEL_312:
                  if ( (__m128i *)v330[0].m128i_i64[0] != &v330[1] )
                    j_j___libc_free_0(v330[0].m128i_u64[0]);
                  goto LABEL_296;
                }
              }
              else if ( (unsigned __int8)sub_16820A0(v252, v330[0].m128i_i64[0]) )
              {
                goto LABEL_312;
              }
              goto LABEL_478;
            }
            goto LABEL_480;
          }
          if ( !(unsigned __int8)sub_CE9220((__int64)v152) || !(src | v253) )
            goto LABEL_296;
          v154 = sub_BD5D20((__int64)v152);
          v155 = (size_t)v153;
          if ( v154 )
            break;
          v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
          v158 = &v330[1];
          v330[0].m128i_i64[1] = 0;
          v330[1].m128i_i8[0] = 0;
LABEL_310:
          if ( (v152[32] & 0xF) == 7 )
          {
            if ( (unsigned __int8)sub_16820A0(v253, v158) )
              goto LABEL_312;
          }
          else if ( (unsigned __int8)sub_16820A0(src, v158) )
          {
            goto LABEL_312;
          }
          v211 = 23;
          v146 = "no reference to kernel ";
          if ( v227 )
            goto LABEL_482;
LABEL_466:
          if ( (__m128i *)v330[0].m128i_i64[0] != &v330[1] )
          {
            v146 = (char *)(v330[1].m128i_i64[0] + 1);
            j_j___libc_free_0(v330[0].m128i_u64[0]);
          }
LABEL_300:
          if ( v149 == i )
            goto LABEL_319;
          v148 = *(_DWORD *)(v147 + 4) & 0x7FFFFFF;
        }
        v330[0].m128i_i64[0] = (__int64)v330[1].m128i_i64;
        v156 = v153;
        v268[0] = v153;
        if ( (unsigned __int64)v153 > 0xF )
        {
          v231 = (size_t)v153;
          v235 = v154;
          v220 = sub_22409D0((__int64)v330, (unsigned __int64 *)v268, 0);
          v154 = v235;
          v155 = v231;
          v330[0].m128i_i64[0] = v220;
          v221 = (__m128i *)v220;
          v330[1].m128i_i64[0] = (__int64)v268[0];
        }
        else
        {
          if ( v153 == (const char *)1 )
          {
            v330[1].m128i_i8[0] = *v154;
            v157 = &v330[1];
LABEL_309:
            v330[0].m128i_i64[1] = (__int64)v156;
            v156[(_QWORD)v157] = 0;
            v158 = (__m128i *)v330[0].m128i_i64[0];
            goto LABEL_310;
          }
          if ( !v153 )
          {
            v157 = &v330[1];
            goto LABEL_309;
          }
          v221 = &v330[1];
        }
        memcpy(v221, v154, v155);
        v156 = v268[0];
        v157 = (__m128i *)v330[0].m128i_i64[0];
        goto LABEL_309;
      }
LABEL_319:
      sub_B30290((__int64)v258);
      v161 = (__int64 *)sub_BCE3C0(*(__int64 **)a3, 0);
      v162 = (__int64 **)sub_BCD420(v161, (v295[1] - v295[0]) >> 3);
      v163 = sub_AD1300(v162, (__int64 *)v295[0], (v295[1] - v295[0]) >> 3);
      v330[2].m128i_i16[0] = 259;
      v330[0].m128i_i64[0] = (__int64)"llvm.used";
      BYTE4(v268[0]) = 0;
      v164 = sub_BD2C40(88, unk_3F0FAE8);
      v165 = (__int64)v164;
      if ( v164 )
        sub_B30000((__int64)v164, a3, v162, 0, 6, v163, (__int64)v330, 0, 0, (__int64)v268[0], 0);
      v146 = "llvm.metadata";
      sub_B31A00(v165, (__int64)"llvm.metadata", 13);
      if ( v295[0] )
      {
        v146 = (char *)(v296 - v295[0]);
        j_j___libc_free_0((unsigned __int64)v295[0]);
      }
    }
    sub_F32B50((__int64)v295, v146);
    v330[1].m128i_i64[1] = 0;
    v330[0].m128i_i8[0] = (__int8)v295[0];
    if ( v297 )
    {
      ((void (__fastcall *)(unsigned __int64 *, char **, __int64))v297)(&v330[0].m128i_u64[1], &v295[1], 2);
      v330[2].m128i_i64[0] = v298;
      v330[1].m128i_i64[1] = v297;
    }
    v166 = v299;
    v299 = 0;
    v330[2].m128i_i64[1] = v166;
    v167 = v300;
    v300 = 0;
    v330[3].m128i_i64[0] = v167;
    v168 = v301;
    LODWORD(v301) = 0;
    v330[3].m128i_i64[1] = v168;
    v169 = sub_22077B0(0x48u);
    v170 = (const char *)v169;
    if ( v169 )
    {
      *(_QWORD *)(v169 + 32) = 0;
      *(_QWORD *)v169 = &unk_4A0E878;
      *(_BYTE *)(v169 + 8) = v330[0].m128i_i8[0];
      if ( v330[1].m128i_i64[1] )
      {
        ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v330[1].m128i_i64[1])(
          v169 + 16,
          &v330[0].m128i_u64[1],
          2);
        *((_QWORD *)v170 + 5) = v330[2].m128i_i64[0];
        *((_QWORD *)v170 + 4) = v330[1].m128i_i64[1];
      }
      v171 = v330[2].m128i_i64[1];
      v330[2].m128i_i64[1] = 0;
      *((_QWORD *)v170 + 6) = v171;
      v172 = v330[3].m128i_i64[0];
      v330[3].m128i_i64[0] = 0;
      *((_QWORD *)v170 + 7) = v172;
      v173 = v330[3].m128i_i64[1];
      v330[3].m128i_i32[2] = 0;
      *((_QWORD *)v170 + 8) = v173;
    }
    v268[0] = v170;
    if ( (char *)v271 == v272 )
    {
      sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v268);
      v170 = v268[0];
    }
    else
    {
      if ( v271 )
      {
        *(_QWORD *)v271 = v170;
        v271 += 8;
LABEL_332:
        v174 = v330[2].m128i_u64[1];
        if ( v330[3].m128i_i32[1] && v330[3].m128i_i32[0] )
        {
          v175 = 8LL * v330[3].m128i_u32[0];
          v176 = 0;
          do
          {
            v177 = *(_QWORD **)(v174 + v176);
            if ( v177 != (_QWORD *)-8LL && v177 )
            {
              sub_C7D6A0((__int64)v177, *v177 + 9LL, 8);
              v174 = v330[2].m128i_u64[1];
            }
            v176 += 8;
          }
          while ( v175 != v176 );
        }
        _libc_free(v174);
        if ( v330[1].m128i_i64[1] )
          ((void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v330[1].m128i_i64[1])(
            &v330[0].m128i_u64[1],
            &v330[0].m128i_u64[1],
            3);
        v178 = v299;
        if ( HIDWORD(v300) && (_DWORD)v300 )
        {
          v179 = 8LL * (unsigned int)v300;
          v180 = 0;
          do
          {
            v181 = *(_QWORD **)(v178 + v180);
            if ( v181 != (_QWORD *)-8LL && v181 )
            {
              sub_C7D6A0((__int64)v181, *v181 + 9LL, 8);
              v178 = v299;
            }
            v180 += 8;
          }
          while ( v179 != v180 );
        }
        _libc_free(v178);
        if ( v297 )
          ((void (__fastcall *)(char **, char **, __int64))v297)(&v295[1], &v295[1], 3);
        sub_BC0DB0((__int64)v330, (__int64)&v270, a3, (__int64)&v284);
        if ( !v330[4].m128i_i8[12] )
          _libc_free(v330[3].m128i_u64[1]);
        if ( !v330[1].m128i_i8[12] )
          _libc_free(v330[0].m128i_u64[1]);
        for ( j = *(_QWORD *)(a3 + 16); a3 + 8 != j; j = *(_QWORD *)(j + 8) )
        {
          if ( !j )
            BUG();
          if ( (*(_BYTE *)(j - 24) & 0xF) == 0
            && *(_DWORD *)(*(_QWORD *)(j - 48) + 8LL) >> 8 == 3
            && *(_BYTE *)(*(_QWORD *)(j - 32) + 8LL) == 16 )
          {
            v183 = (*(_WORD *)(j - 22) >> 1) & 0x3F;
            if ( !v183 || (unsigned __int64)(1LL << ((unsigned __int8)v183 - 1)) <= 0xF )
              sub_B2F770(j - 56, 4u);
          }
        }
        goto LABEL_89;
      }
      v271 = 8;
    }
    if ( v170 )
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v170 + 8LL))(v170);
    goto LABEL_332;
  }
  LOBYTE(v295[0]) = 0;
  v306 = &v312;
  v325 = v329;
  v295[1] = 0;
  v296 = (char *)&v299;
  v297 = 32;
  LODWORD(v298) = 0;
  BYTE4(v298) = 1;
  v302 = 0;
  v303 = 0;
  v304 = 0;
  v305 = 0;
  v307 = 1;
  v308 = 0;
  v309 = 0;
  v310 = 1065353216;
  v311 = 0;
  v312 = 0;
  v313 = &v319;
  v314 = 1;
  v315 = 0;
  v316 = 0;
  v317 = 1065353216;
  v318 = 0;
  v319 = 0;
  v320 = 0;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v326 = 32;
  v327 = 0;
  v328 = 1;
  sub_234B220((__int64)v330, (__int64)v295);
  v106 = (char *)sub_22077B0(0x300u);
  v107 = v106;
  if ( v106 )
  {
    *(_QWORD *)v106 = &unk_4A0E7F8;
    sub_234B220((__int64)(v106 + 8), (__int64)v330);
  }
  v268[0] = v107;
  if ( (char *)v271 == v272 )
  {
    v68 = &v270;
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v268);
    v107 = v268[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v107;
      v68 = &v270;
      v271 += 8;
      goto LABEL_167;
    }
    v271 = 8;
    v68 = &v270;
  }
  if ( v107 )
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v107 + 8LL))(v107);
LABEL_167:
  sub_233AAF0((__int64)v330);
  sub_233AAF0((__int64)v295);
  sub_29744A0((__int64)v330);
  n = v330[1].m128i_u64[0];
  v108 = v330[0];
  v109 = (_QWORD *)sub_22077B0(0x20u);
  if ( v109 )
  {
    v109[1] = v108.m128i_i64[0];
    *v109 = &unk_4A11BB8;
    v109[2] = v108.m128i_i64[1];
    v109[3] = n;
  }
  v110 = sub_22077B0(0x18u);
  v111 = (char *)v110;
  if ( v110 )
  {
    *(_BYTE *)(v110 + 16) = 0;
    *(_QWORD *)(v110 + 8) = v109;
    v109 = 0;
    *(_QWORD *)v110 = &unk_4A0C478;
  }
  v295[0] = (char *)v110;
  if ( (char *)v271 == v272 )
  {
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v295);
    v111 = v295[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v110;
      v271 += 8;
      goto LABEL_174;
    }
    v271 = 8;
  }
  if ( v111 )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v111 + 8LL))(v111);
LABEL_174:
  if ( v109 )
    (*(void (__fastcall **)(_QWORD *))(*v109 + 8LL))(v109);
  v112 = sub_22077B0(0x10u);
  if ( v112 )
    *(_QWORD *)v112 = &unk_4A0FFF8;
  v113 = sub_22077B0(0x18u);
  v114 = v113;
  if ( v113 )
  {
    *(_BYTE *)(v113 + 16) = 0;
    *(_QWORD *)(v113 + 8) = v112;
    v112 = 0;
    *(_QWORD *)v113 = &unk_4A0C478;
  }
  v330[0].m128i_i64[0] = v113;
  if ( (char *)v271 == v272 )
  {
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v330);
    v114 = v330[0].m128i_i64[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v113;
      v271 += 8;
      goto LABEL_183;
    }
    v271 = 8;
  }
  if ( v114 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v114 + 8LL))(v114);
LABEL_183:
  if ( v112 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v112 + 8LL))(v112);
  v115 = sub_22077B0(0x10u);
  if ( v115 )
    *(_QWORD *)v115 = &unk_4A0ED38;
  v116 = sub_22077B0(0x18u);
  v117 = v116;
  if ( v116 )
  {
    *(_BYTE *)(v116 + 16) = 0;
    *(_QWORD *)(v116 + 8) = v115;
    v115 = 0;
    *(_QWORD *)v116 = &unk_4A0C478;
  }
  v330[0].m128i_i64[0] = v116;
  if ( (char *)v271 == v272 )
  {
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v330);
    v117 = v330[0].m128i_i64[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v116;
      v271 += 8;
      goto LABEL_192;
    }
    v271 = 8;
  }
  if ( v117 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v117 + 8LL))(v117);
LABEL_192:
  if ( v115 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v115 + 8LL))(v115);
  v118 = sub_22077B0(0x10u);
  if ( v118 )
    *(_QWORD *)v118 = &unk_4A0F4B8;
  v119 = sub_22077B0(0x18u);
  v120 = v119;
  if ( v119 )
  {
    *(_BYTE *)(v119 + 16) = 0;
    *(_QWORD *)(v119 + 8) = v118;
    v118 = 0;
    *(_QWORD *)v119 = &unk_4A0C478;
  }
  v330[0].m128i_i64[0] = v119;
  if ( (char *)v271 == v272 )
  {
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v330);
    v120 = v330[0].m128i_i64[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v119;
      v271 += 8;
      goto LABEL_201;
    }
    v271 = 8;
  }
  if ( v120 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v120 + 8LL))(v120);
LABEL_201:
  if ( v118 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v118 + 8LL))(v118);
  v121 = sub_22077B0(0x10u);
  if ( v121 )
    *(_QWORD *)v121 = &unk_4A0E2B8;
  v330[0].m128i_i64[0] = v121;
  if ( (char *)v271 == v272 )
  {
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v330);
    v121 = v330[0].m128i_i64[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v121;
      v271 += 8;
      goto LABEL_208;
    }
    v271 = 8;
  }
  if ( v121 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v121 + 8LL))(v121);
LABEL_208:
  v296 = (char *)&v299;
  LOBYTE(v295[0]) = 0;
  v306 = &v312;
  v295[1] = 0;
  v325 = v329;
  v297 = 32;
  LODWORD(v298) = 0;
  BYTE4(v298) = 1;
  v302 = 0;
  v303 = 0;
  v304 = 0;
  v305 = 0;
  v307 = 1;
  v308 = 0;
  v309 = 0;
  v310 = 1065353216;
  v311 = 0;
  v312 = 0;
  v313 = &v319;
  v314 = 1;
  v315 = 0;
  v316 = 0;
  v317 = 1065353216;
  v318 = 0;
  v319 = 0;
  v320 = 0;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v326 = 32;
  v327 = 0;
  v328 = 1;
  sub_234B220((__int64)v330, (__int64)v295);
  v122 = (char *)sub_22077B0(0x300u);
  v123 = v122;
  if ( v122 )
  {
    *(_QWORD *)v122 = &unk_4A0E7F8;
    sub_234B220((__int64)(v122 + 8), (__int64)v330);
  }
  v268[0] = v123;
  if ( (char *)v271 == v272 )
  {
    sub_2275C60((unsigned __int64 *)&v270, (char *)v271, v268);
    v123 = v268[0];
  }
  else
  {
    if ( v271 )
    {
      *(_QWORD *)v271 = v123;
      v271 += 8;
      goto LABEL_213;
    }
    v271 = 8;
  }
  if ( v123 )
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v123 + 8LL))(v123);
LABEL_213:
  sub_233AAF0((__int64)v330);
  sub_233AAF0((__int64)v295);
  v124 = a7[1];
  v273 = *a7;
  v274 = v124;
  sub_BC0DB0((__int64)v330, (__int64)&v270, a3, (__int64)&v284);
  if ( !v330[4].m128i_i8[12] )
    _libc_free(v330[3].m128i_u64[1]);
  if ( !v330[1].m128i_i8[12] )
    _libc_free(v330[0].m128i_u64[1]);
  v125 = v262;
  v126 = v263;
  s = (char *)v267;
  LOBYTE(v68) = v262 == 0 && (char *)v262 + v263 != 0;
  if ( (_BYTE)v68 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v330[0].m128i_i64[0] = v263;
  if ( v263 > 0xF )
  {
    s = (char *)sub_22409D0((__int64)&s, (unsigned __int64 *)v330, 0);
    v186 = s;
    v267[0] = v330[0].m128i_i64[0];
    goto LABEL_373;
  }
  if ( v263 != 1 )
  {
    if ( !v263 )
    {
      v127 = (char *)v267;
      goto LABEL_221;
    }
    v186 = (char *)v267;
LABEL_373:
    memcpy(v186, v125, v126);
    v126 = v330[0].m128i_i64[0];
    v127 = s;
    goto LABEL_221;
  }
  LOBYTE(v267[0]) = *(_BYTE *)v262;
  v127 = (char *)v267;
LABEL_221:
  v266 = v126;
  v127[v126] = 0;
  sub_CCBAC0((__int64)v295, (__int64)sub_309A600, (__int64)sub_309A5F0, 0, 0);
  sub_CD07E0((__int64)&v259, a3, (__int64)v293, (__int64)v295, 1);
  if ( v266 )
  {
    v260 = 0;
    v128 = sub_2241E40();
    v129 = s;
    v130 = 0;
    v261 = v128;
    if ( s )
      v130 = strlen(s);
    sub_CB7060((__int64)v330, v129, v130, (__int64)&v260, 0);
    if ( v260 )
    {
      sub_223E0D0(qword_4FD4BE0, "IO error: ", 10);
      (*((void (__fastcall **)(const char **, __int64 (__fastcall **)(), _QWORD))*v261 + 4))(v268, v261, v260);
      v187 = sub_223E0D0(qword_4FD4BE0, v268[0], (__int64)v268[1]);
      v131 = (__m128i *)"\n";
      sub_223E0D0(v187, "\n", 1);
      if ( (__int64 *)v268[0] != &v269 )
      {
        v131 = (__m128i *)(v269 + 1);
        j_j___libc_free_0((unsigned __int64)v268[0]);
      }
    }
    else
    {
      LODWORD(v68) = 1;
      v131 = v330;
      sub_CDD2D0(v259, (__int64)v330, 1, 1);
      sub_CB7080((__int64)v330, (__int64)v330);
    }
    sub_CB5B00(v330[0].m128i_i32, (__int64)v131);
  }
  else
  {
    v330[0].m128i_i64[1] = 0;
    v330[2].m128i_i64[1] = 0x100000000LL;
    v330[0].m128i_i64[0] = (__int64)&unk_49DD210;
    memset(&v330[1], 0, 24);
    v330[3].m128i_i64[0] = a5;
    sub_CB5980((__int64)v330, 0, 0, 0);
    sub_CDD2D0(v259, (__int64)v330, 1, 1);
    if ( v330[2].m128i_i64[0] != v330[1].m128i_i64[0] )
      sub_CB5AE0(v330[0].m128i_i64);
    LODWORD(v68) = 1;
    v330[0].m128i_i64[0] = (__int64)&unk_49DD210;
    sub_CB5840((__int64)v330);
  }
  if ( v259 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v259 + 56LL))(v259);
  if ( s != (char *)v267 )
    j_j___libc_free_0((unsigned __int64)s);
  if ( (_BYTE)v68 )
    goto LABEL_89;
  if ( src )
    sub_1688090(src, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
  if ( v253 )
    sub_1688090(v253, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
  if ( v254 )
    sub_1688090(v254, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
  if ( v252 )
    sub_1688090(v252, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
  if ( *((_QWORD *)&v251 + 1) )
    sub_1688090(*((__int64 *)&v251 + 1), (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
  if ( (_QWORD)v251 )
    sub_1688090(v251, (void (__fastcall *)(_QWORD, __int64))sub_1683C50);
LABEL_102:
  sub_2272BE0((__int64)v332);
  sub_C7D6A0(v291, 24LL * v292, 8);
  v69 = v290;
  if ( v290 )
  {
    v70 = (_QWORD **)(v289 + 32LL * v290);
    for ( k = (_QWORD **)(v289 + 8); ; k += 4 )
    {
      v72 = (__int64)*(k - 1);
      if ( v72 != -8192 && v72 != -4096 )
      {
        v73 = *k;
        while ( v73 != k )
        {
          v74 = (unsigned __int64)v73;
          v73 = (_QWORD *)*v73;
          v75 = *(_QWORD *)(v74 + 24);
          if ( v75 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v75 + 8LL))(v75);
          j_j___libc_free_0(v74);
        }
      }
      if ( v70 == k + 3 )
        break;
    }
    v69 = v290;
  }
  sub_C7D6A0(v289, 32LL * v69, 8);
  v81 = v288;
  if ( v288 )
  {
    v82 = v285;
    v83 = &v285[2 * v288];
    do
    {
      if ( *v82 != -4096 && *v82 != -8192 )
      {
        v84 = v82[1];
        if ( v84 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v84 + 8LL))(v84);
      }
      v82 += 2;
    }
    while ( v83 != v82 );
    v81 = v288;
  }
  sub_C7D6A0((__int64)v285, 16LL * v81, 8);
  sub_C7D6A0(v282, 24LL * v283, 8);
  v85 = v281;
  if ( v281 )
  {
    v86 = (_QWORD **)(v280 + 32LL * v281);
    for ( m = (_QWORD **)(v280 + 8); ; m += 4 )
    {
      v88 = (__int64)*(m - 1);
      if ( v88 != -8192 && v88 != -4096 )
      {
        v89 = *m;
        while ( v89 != m )
        {
          v90 = (unsigned __int64)v89;
          v89 = (_QWORD *)*v89;
          v91 = *(_QWORD *)(v90 + 24);
          if ( v91 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v91 + 8LL))(v91);
          j_j___libc_free_0(v90);
        }
      }
      if ( v86 == m + 3 )
        break;
    }
    v85 = v281;
  }
  sub_C7D6A0(v280, 32LL * v85, 8);
  v92 = v279;
  if ( v279 )
  {
    v93 = v276;
    v94 = &v276[2 * v279];
    do
    {
      if ( *v93 != -4096 && *v93 != -8192 )
      {
        v95 = v93[1];
        if ( v95 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v95 + 8LL))(v95);
      }
      v93 += 2;
    }
    while ( v94 != v93 );
    v92 = v279;
  }
  sub_C7D6A0((__int64)v276, 16LL * v92, 8);
  v96 = v271;
  v97 = v270;
  if ( (char *)v271 != v270 )
  {
    do
    {
      if ( *(_QWORD *)v97 )
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v97 + 8LL))(*(_QWORD *)v97);
      v97 += 8;
    }
    while ( (char *)v96 != v97 );
    v97 = v270;
  }
  if ( v97 )
    j_j___libc_free_0((unsigned __int64)v97);
  if ( v262 != v264 )
    j_j___libc_free_0((unsigned __int64)v262);
  return (unsigned int)v68;
}
