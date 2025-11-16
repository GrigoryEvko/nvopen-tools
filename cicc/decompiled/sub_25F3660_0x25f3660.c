// Function: sub_25F3660
// Address: 0x25f3660
//
__int64 __fastcall sub_25F3660(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  int v7; // eax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // r12
  _QWORD *v25; // rbx
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // r15
  __int64 *v36; // rax
  __int64 *v37; // rdx
  unsigned __int64 v38; // r14
  unsigned int v39; // r12d
  __int64 v40; // rbx
  unsigned __int64 v41; // r13
  unsigned __int64 v42; // r15
  unsigned __int64 v43; // rdi
  __int64 v44; // rbx
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // r15
  unsigned __int64 v47; // rdi
  __int64 *v48; // rbx
  unsigned __int64 v49; // r13
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  __int64 *v55; // rax
  __int64 *v56; // rdx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // r9
  unsigned __int64 *v61; // r12
  unsigned __int64 v62; // r14
  __int64 v63; // rbx
  __int64 *v64; // r12
  __int64 v65; // rdx
  __int64 *v66; // rbx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rbx
  unsigned int v70; // r13d
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r13
  __int64 v74; // rsi
  __int64 *v75; // r15
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rsi
  __int64 v79; // r15
  __int64 v80; // rsi
  __int64 *v81; // r13
  __int64 v82; // r15
  __int64 v83; // rsi
  __int64 *v84; // r13
  unsigned __int64 v85; // rax
  __int64 *v86; // r14
  unsigned __int64 v87; // r12
  __int64 v88; // rbx
  __int64 v89; // rsi
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rax
  unsigned __int64 v93; // rdx
  unsigned __int64 v94; // rax
  __int64 *v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdx
  __int64 *v98; // rax
  __int64 v99; // rcx
  unsigned __int64 v100; // rdx
  unsigned int v101; // ecx
  unsigned __int64 v102; // rsi
  unsigned int v103; // edi
  unsigned __int64 v104; // rsi
  __int64 *v105; // rdi
  unsigned int v106; // ecx
  unsigned int v107; // esi
  unsigned int v108; // eax
  unsigned int v109; // r9d
  __int64 v110; // rax
  __int64 v111; // r10
  __int64 v112; // rdx
  __m128i *v113; // rax
  __int64 *v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 *v118; // r13
  __int64 v119; // rbx
  __int64 v120; // rax
  __int64 v121; // rbx
  __int64 *v122; // rbx
  unsigned __int64 v123; // rsi
  __int64 *v124; // rax
  __int64 *v125; // r12
  unsigned __int64 *v126; // r13
  unsigned __int8 v127; // di
  __int64 v128; // rsi
  __int64 *v129; // rax
  __int64 v130; // r8
  __int64 v131; // r9
  int v132; // eax
  __int64 v133; // rcx
  __int64 *v134; // rdx
  unsigned __int64 v135; // rbx
  __int64 v136; // r10
  __int64 v137; // rax
  unsigned __int64 v138; // rdx
  __int64 v139; // r13
  __int64 v140; // rax
  unsigned __int64 v141; // rdx
  __int64 *v142; // rbx
  __int64 v143; // r12
  __int64 *v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // r8
  __int64 v147; // r9
  char v148; // al
  __int64 *v149; // rax
  __int64 *v150; // rax
  __int64 v151; // rsi
  __int64 *v152; // rax
  __int64 v153; // rdi
  __int64 *v154; // rax
  __int64 *v155; // rax
  char v156; // al
  __int64 v157; // rcx
  __int64 v158; // rsi
  __int64 v159; // rsi
  __int64 v160; // rsi
  __int64 v161; // rax
  __int64 v162; // rcx
  __int64 v163; // r8
  __int64 v164; // r9
  __int64 v165; // rdx
  __int64 v166; // rdi
  __int64 v167; // rax
  __int64 v168; // rdi
  __int64 v169; // rdi
  __int64 *v170; // r13
  __int64 *v171; // r12
  __int64 *v172; // rbx
  __int64 *v173; // r13
  unsigned __int64 v174; // rdi
  unsigned __int64 v175; // rdi
  unsigned __int64 v176; // rdi
  unsigned __int64 v177; // rdi
  int v178; // r13d
  __int64 v179; // rsi
  __int64 v180; // rsi
  __int64 *v181; // rbx
  __int64 *v182; // r12
  __int64 v183; // rdx
  __int64 v184; // rsi
  __int64 v185; // rax
  __int64 v186; // r12
  __int64 v187; // rbx
  unsigned int v188; // eax
  unsigned __int64 v189; // rbx
  unsigned __int64 *v190; // r13
  __int64 v191; // rax
  __int64 v192; // rdi
  int v193; // eax
  __int64 v194; // rax
  __int64 v195; // rdi
  int v196; // eax
  __int64 *v197; // [rsp+10h] [rbp-E80h]
  unsigned int v198; // [rsp+24h] [rbp-E6Ch]
  __int64 *v199; // [rsp+30h] [rbp-E60h]
  unsigned __int64 v200; // [rsp+40h] [rbp-E50h]
  _BYTE *v202; // [rsp+50h] [rbp-E40h]
  _BYTE *v203; // [rsp+58h] [rbp-E38h]
  __int64 **v204; // [rsp+60h] [rbp-E30h]
  unsigned __int64 v206; // [rsp+70h] [rbp-E20h]
  __int64 v207; // [rsp+78h] [rbp-E18h]
  __int64 v208; // [rsp+80h] [rbp-E10h]
  __int64 v209; // [rsp+E8h] [rbp-DA8h]
  unsigned int v210; // [rsp+F0h] [rbp-DA0h]
  unsigned int v211; // [rsp+F4h] [rbp-D9Ch]
  unsigned int v212; // [rsp+F4h] [rbp-D9Ch]
  unsigned int v213; // [rsp+F4h] [rbp-D9Ch]
  __int64 *v214; // [rsp+F8h] [rbp-D98h]
  unsigned int v215; // [rsp+F8h] [rbp-D98h]
  __int64 *v216; // [rsp+F8h] [rbp-D98h]
  unsigned int v217; // [rsp+100h] [rbp-D90h]
  char v218; // [rsp+100h] [rbp-D90h]
  unsigned __int64 v219; // [rsp+108h] [rbp-D88h]
  __int64 *v220; // [rsp+108h] [rbp-D88h]
  __int64 *v221; // [rsp+108h] [rbp-D88h]
  __int64 *v222; // [rsp+108h] [rbp-D88h]
  __int64 v223; // [rsp+108h] [rbp-D88h]
  unsigned int v224; // [rsp+108h] [rbp-D88h]
  unsigned int v225; // [rsp+108h] [rbp-D88h]
  unsigned int v226; // [rsp+108h] [rbp-D88h]
  __int64 *v227; // [rsp+108h] [rbp-D88h]
  __int64 v228; // [rsp+110h] [rbp-D80h] BYREF
  __int64 *v229; // [rsp+118h] [rbp-D78h]
  __int64 v230; // [rsp+120h] [rbp-D70h]
  int v231; // [rsp+128h] [rbp-D68h]
  unsigned __int8 v232; // [rsp+12Ch] [rbp-D64h]
  char v233; // [rsp+130h] [rbp-D60h] BYREF
  __int64 v234; // [rsp+150h] [rbp-D40h] BYREF
  __int64 *v235; // [rsp+158h] [rbp-D38h]
  __int64 v236; // [rsp+160h] [rbp-D30h]
  int v237; // [rsp+168h] [rbp-D28h]
  char v238; // [rsp+16Ch] [rbp-D24h]
  char v239; // [rsp+170h] [rbp-D20h] BYREF
  __int64 v240; // [rsp+190h] [rbp-D00h] BYREF
  char *v241; // [rsp+198h] [rbp-CF8h]
  __int64 v242; // [rsp+1A0h] [rbp-CF0h]
  int v243; // [rsp+1A8h] [rbp-CE8h]
  char v244; // [rsp+1ACh] [rbp-CE4h]
  char v245; // [rsp+1B0h] [rbp-CE0h] BYREF
  _BYTE *v246; // [rsp+1D0h] [rbp-CC0h] BYREF
  __int64 v247; // [rsp+1D8h] [rbp-CB8h]
  _BYTE v248[64]; // [rsp+1E0h] [rbp-CB0h] BYREF
  unsigned __int64 v249[54]; // [rsp+220h] [rbp-C70h] BYREF
  unsigned __int64 v250; // [rsp+3D0h] [rbp-AC0h] BYREF
  __int64 *v251; // [rsp+3D8h] [rbp-AB8h]
  _DWORD v252[3]; // [rsp+3E0h] [rbp-AB0h] BYREF
  char v253; // [rsp+3ECh] [rbp-AA4h]
  __int64 v254; // [rsp+3F0h] [rbp-AA0h] BYREF
  unsigned __int64 *v255; // [rsp+430h] [rbp-A60h]
  __int64 v256; // [rsp+438h] [rbp-A58h]
  unsigned __int64 v257; // [rsp+440h] [rbp-A50h] BYREF
  int v258; // [rsp+448h] [rbp-A48h]
  unsigned __int64 v259; // [rsp+450h] [rbp-A40h]
  int v260; // [rsp+458h] [rbp-A38h]
  __int64 v261; // [rsp+460h] [rbp-A30h]
  unsigned __int64 *v262; // [rsp+580h] [rbp-910h] BYREF
  unsigned __int64 *v263; // [rsp+588h] [rbp-908h]
  char v264; // [rsp+59Ch] [rbp-8F4h]
  char *v265; // [rsp+5E0h] [rbp-8B0h]
  char v266; // [rsp+5F0h] [rbp-8A0h] BYREF
  __int64 *v267; // [rsp+730h] [rbp-760h] BYREF
  unsigned __int64 v268; // [rsp+738h] [rbp-758h]
  __int64 v269; // [rsp+740h] [rbp-750h] BYREF
  char v270; // [rsp+74Ch] [rbp-744h]
  char *v271; // [rsp+790h] [rbp-700h]
  char v272; // [rsp+7A0h] [rbp-6F0h] BYREF
  __int64 v273[3]; // [rsp+8E0h] [rbp-5B0h] BYREF
  char v274; // [rsp+8FCh] [rbp-594h]
  __int64 v275; // [rsp+920h] [rbp-570h]
  unsigned int v276; // [rsp+930h] [rbp-560h]
  unsigned __int64 *v277; // [rsp+938h] [rbp-558h]
  char *v278; // [rsp+940h] [rbp-550h]
  char *v279; // [rsp+948h] [rbp-548h] BYREF
  char v280; // [rsp+950h] [rbp-540h] BYREF
  char v281; // [rsp+958h] [rbp-538h] BYREF
  _QWORD *v282; // [rsp+988h] [rbp-508h]
  _QWORD v283[6]; // [rsp+998h] [rbp-4F8h] BYREF
  unsigned int v284; // [rsp+9C8h] [rbp-4C8h]
  char *v285; // [rsp+9D0h] [rbp-4C0h]
  char v286; // [rsp+9E0h] [rbp-4B0h] BYREF
  __m128i *v287; // [rsp+A90h] [rbp-400h] BYREF
  unsigned __int64 v288; // [rsp+A98h] [rbp-3F8h] BYREF
  __m128i v289; // [rsp+AA0h] [rbp-3F0h] BYREF
  __int64 v290; // [rsp+AD8h] [rbp-3B8h]
  unsigned int v291; // [rsp+AE8h] [rbp-3A8h]
  unsigned __int64 *v292; // [rsp+AF0h] [rbp-3A0h]
  unsigned __int64 v293[2]; // [rsp+B00h] [rbp-390h] BYREF
  char v294; // [rsp+B10h] [rbp-380h] BYREF
  __int64 v295; // [rsp+B28h] [rbp-368h]
  unsigned int v296; // [rsp+B38h] [rbp-358h]
  __int64 *v297; // [rsp+B40h] [rbp-350h]
  __int64 v298; // [rsp+B48h] [rbp-348h]
  __int64 v299; // [rsp+B50h] [rbp-340h] BYREF
  unsigned int v300; // [rsp+B58h] [rbp-338h]
  __int64 v301; // [rsp+B70h] [rbp-320h]
  unsigned int v302; // [rsp+B80h] [rbp-310h]
  char *v303; // [rsp+B88h] [rbp-308h]
  char v304; // [rsp+B98h] [rbp-2F8h] BYREF
  __int64 *v305; // [rsp+C40h] [rbp-250h] BYREF
  __int64 v306; // [rsp+C48h] [rbp-248h]
  _BYTE v307[576]; // [rsp+C50h] [rbp-240h] BYREF

  v229 = (__int64 *)&v233;
  v235 = (__int64 *)&v239;
  v241 = &v245;
  v305 = (__int64 *)v307;
  v306 = 0x200000000LL;
  v246 = v248;
  memset(v249, 0, sizeof(v249));
  v247 = 0x800000000LL;
  v249[1] = (unsigned __int64)&v249[4];
  v228 = 0;
  v230 = 4;
  v231 = 0;
  v232 = 1;
  v234 = 0;
  v236 = 4;
  v237 = 0;
  v238 = 1;
  v240 = 0;
  v242 = 4;
  v243 = 0;
  v244 = 1;
  LODWORD(v249[2]) = 8;
  v4 = *(_QWORD *)(a2 + 80);
  v256 = 0x800000000LL;
  v249[12] = (unsigned __int64)&v249[14];
  if ( v4 )
    v4 -= 24;
  HIDWORD(v249[13]) = 8;
  v251 = &v254;
  v252[0] = 8;
  v252[2] = 0;
  v253 = 1;
  v255 = &v257;
  v252[1] = 1;
  v254 = v4;
  v250 = 1;
  v5 = *(_QWORD *)(v4 + 48);
  BYTE4(v249[3]) = 1;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == v4 + 48 )
    goto LABEL_408;
  if ( !v6 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
  {
LABEL_408:
    v7 = 0;
    v9 = 0;
    v8 = 0;
  }
  else
  {
    v219 = v6 - 24;
    v7 = sub_B46E30(v6 - 24);
    v8 = v219;
    v9 = v219;
  }
  v258 = v7;
  v257 = v9;
  v259 = v8;
  v261 = v4;
  v260 = 0;
  LODWORD(v256) = 1;
  sub_CE27D0((__int64)&v250);
  sub_CE3710((__int64)v273, (__int64)v249, v10, v11, v12, v13);
  sub_CE35F0((__int64)&v287, (__int64)v273);
  sub_CE3710((__int64)&v262, (__int64)&v250, v14, v15, v16, v17);
  sub_CE35F0((__int64)&v267, (__int64)&v262);
  sub_CE37E0((__int64)&v267, (__int64)&v287, (__int64)&v246, v18, v19, v20);
  if ( v271 != &v272 )
    _libc_free((unsigned __int64)v271);
  if ( !v270 )
    _libc_free(v268);
  if ( v265 != &v266 )
    _libc_free((unsigned __int64)v265);
  if ( !v264 )
    _libc_free((unsigned __int64)v263);
  if ( v292 != v293 )
    _libc_free((unsigned __int64)v292);
  if ( !v289.m128i_i8[12] )
    _libc_free(v288);
  if ( v278 != &v280 )
    _libc_free((unsigned __int64)v278);
  if ( !v274 )
    _libc_free(v273[1]);
  if ( v255 != &v257 )
    _libc_free((unsigned __int64)v255);
  if ( !v253 )
    _libc_free((unsigned __int64)v251);
  if ( (unsigned __int64 *)v249[12] != &v249[14] )
    _libc_free(v249[12]);
  if ( !BYTE4(v249[3]) )
    _libc_free(v249[1]);
  v199 = 0;
  if ( a3 )
    v199 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 8))(*(_QWORD *)(a1 + 16), a2);
  v21 = *(_QWORD *)(a1 + 32);
  v204 = (__int64 **)(*(__int64 (__fastcall **)(__int64, __int64))(a1 + 24))(v21, a2);
  v23 = *(_QWORD *)(a1 + 40);
  if ( !*(_QWORD *)(v23 + 16) )
    sub_4263D6(v21, a2, v22);
  v197 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(v23 + 24))(v23, a2);
  v207 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 48))(*(_QWORD *)(a1 + 56), a2);
  v198 = 0x80000000 - sub_DF95A0(v204);
  v24 = sub_C52410();
  v25 = v24 + 1;
  v26 = sub_C959E0();
  v27 = (_QWORD *)v24[2];
  if ( v27 )
  {
    v28 = v24 + 1;
    do
    {
      while ( 1 )
      {
        v29 = v27[2];
        v30 = v27[3];
        if ( v26 <= v27[4] )
          break;
        v27 = (_QWORD *)v27[3];
        if ( !v30 )
          goto LABEL_39;
      }
      v28 = v27;
      v27 = (_QWORD *)v27[2];
    }
    while ( v29 );
LABEL_39:
    if ( v25 != v28 && v26 >= v28[4] )
      v25 = v28;
  }
  if ( v25 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v31 = v25[7];
    if ( v31 )
    {
      v32 = v25 + 6;
      do
      {
        while ( 1 )
        {
          v33 = *(_QWORD *)(v31 + 16);
          v34 = *(_QWORD *)(v31 + 24);
          if ( *(_DWORD *)(v31 + 32) >= dword_4FF1888 )
            break;
          v31 = *(_QWORD *)(v31 + 24);
          if ( !v34 )
            goto LABEL_48;
        }
        v32 = (_QWORD *)v31;
        v31 = *(_QWORD *)(v31 + 16);
      }
      while ( v33 );
LABEL_48:
      if ( v25 + 6 != v32 && dword_4FF1888 >= *((_DWORD *)v32 + 8) && *((_DWORD *)v32 + 9) )
      {
        sub_F02DB0(&v287, 1u, qword_4FF1908);
        v198 = (unsigned int)v287;
      }
    }
  }
  v202 = v246;
  v203 = &v246[8 * (unsigned int)v247];
  if ( v203 == v246 )
  {
    v39 = 0;
    if ( (_DWORD)v306 )
    {
      v206 = 0;
      v38 = 0;
      goto LABEL_389;
    }
    goto LABEL_86;
  }
  v206 = 0;
  v210 = 1;
  v35 = 0;
  do
  {
    v208 = *((_QWORD *)v203 - 1);
    if ( v232 )
    {
      v36 = v229;
      v37 = &v229[HIDWORD(v230)];
      if ( v229 != v37 )
      {
        do
        {
          if ( v208 == *v36 )
            goto LABEL_57;
          ++v36;
        }
        while ( v37 != v36 );
      }
    }
    else if ( sub_C8CA60((__int64)&v228, v208) )
    {
      goto LABEL_57;
    }
    if ( v238 )
    {
      v55 = v235;
      v56 = &v235[HIDWORD(v236)];
      if ( v235 != v56 )
      {
        do
        {
          if ( v208 == *v55 )
            goto LABEL_57;
          ++v55;
        }
        while ( v56 != v55 );
      }
    }
    else if ( sub_C8CA60((__int64)&v234, v208) )
    {
      goto LABEL_57;
    }
    if ( !(unsigned __int8)sub_25F0440((__int64 *)a1, v208, v198, (__int64)&v240, v199) )
      goto LABEL_57;
    if ( v35 || (v191 = sub_22077B0(0x80u), (v35 = v191) == 0) )
    {
      if ( !v206 )
        goto LABEL_412;
    }
    else
    {
      *(_BYTE *)(v191 + 112) = 0;
      v192 = v191;
      *(_QWORD *)v191 = v191 + 16;
      *(_QWORD *)(v191 + 8) = 0x100000000LL;
      *(_QWORD *)(v191 + 24) = v191 + 40;
      *(_QWORD *)(v191 + 32) = 0x600000000LL;
      *(_QWORD *)(v191 + 96) = 0;
      *(_QWORD *)(v191 + 104) = a2;
      v193 = *(_DWORD *)(a2 + 92);
      *(_DWORD *)(v35 + 116) = 0;
      *(_DWORD *)(v35 + 120) = v193;
      sub_B1F440(v192);
      if ( !v206 )
      {
LABEL_412:
        v194 = sub_22077B0(0x98u);
        v206 = v194;
        v195 = v194;
        if ( v194 )
        {
          *(_QWORD *)(v194 + 120) = 0;
          *(_QWORD *)v194 = v194 + 16;
          *(_QWORD *)(v194 + 8) = 0x400000000LL;
          *(_QWORD *)(v194 + 48) = v194 + 64;
          *(_QWORD *)(v194 + 56) = 0x600000000LL;
          *(_BYTE *)(v194 + 136) = 0;
          *(_QWORD *)(v194 + 128) = a2;
          v196 = *(_DWORD *)(a2 + 92);
          *(_DWORD *)(v195 + 140) = 0;
          *(_DWORD *)(v195 + 144) = v196;
          sub_B29120(v195);
        }
      }
    }
    sub_25F27C0((unsigned __int64 *)&v262, v208, v35, v206, v57, v58);
    v61 = v262;
    v200 = (unsigned __int64)v263;
    if ( v263 == v262 )
      goto LABEL_231;
    v62 = (unsigned __int64)v262;
    do
    {
      v63 = *(_QWORD *)(v62 + 16);
      if ( !v63 )
        goto LABEL_225;
      if ( *(_BYTE *)(v62 + 24) )
      {
        v38 = v35;
        v188 = sub_25EFA30(a2, 0);
        v189 = (unsigned __int64)v263;
        v190 = v262;
        v39 = v188;
        if ( v263 != v262 )
        {
          do
          {
            if ( (unsigned __int64 *)*v190 != v190 + 2 )
              _libc_free(*v190);
            v190 += 4;
          }
          while ( (unsigned __int64 *)v189 != v190 );
          v190 = v262;
        }
        if ( v190 )
          j_j___libc_free_0((unsigned __int64)v190);
        goto LABEL_59;
      }
      do
      {
        v251 = 0;
        v250 = (unsigned __int64)v252;
        sub_C8D5F0((__int64)&v250, v252, 1u, 8u, v59, v60);
        *(_QWORD *)(v250 + 8LL * (unsigned int)v251) = v63;
        LODWORD(v251) = (_DWORD)v251 + 1;
        v64 = *(__int64 **)v62;
        v65 = 16LL * *(unsigned int *)(v62 + 8);
        v66 = (__int64 *)(*(_QWORD *)v62 + v65);
        v67 = v65 >> 4;
        v68 = v65 >> 6;
        if ( v68 )
        {
          v220 = v66;
          v69 = v35;
          v217 = 0;
          v209 = 0;
          v214 = &v64[8 * v68];
          while ( 1 )
          {
            v82 = *v64;
            v83 = *(_QWORD *)(v62 + 16);
            if ( *v64 == v83 )
            {
              v35 = v69;
              v66 = v220;
              goto LABEL_143;
            }
            v70 = *((_DWORD *)v64 + 2);
            if ( (unsigned __int8)sub_B19720(v69, v83, *v64) )
              break;
            if ( v70 > v217 )
            {
              v217 = v70;
              v209 = v82;
            }
            v73 = v64[2];
            v74 = *(_QWORD *)(v62 + 16);
            v75 = v64 + 2;
            if ( v73 == v74 )
              goto LABEL_233;
            v211 = *((_DWORD *)v64 + 6);
            if ( (unsigned __int8)sub_B19720(v69, v74, v64[2]) )
              goto LABEL_236;
            if ( v211 > v217 )
            {
              v209 = v73;
              v217 = v211;
            }
            v73 = v64[4];
            v78 = *(_QWORD *)(v62 + 16);
            v75 = v64 + 4;
            if ( v73 == v78 )
            {
LABEL_233:
              v81 = v75;
LABEL_234:
              v35 = v69;
              v66 = v220;
LABEL_235:
              v64 = v81;
              goto LABEL_143;
            }
            v212 = *((_DWORD *)v64 + 10);
            if ( (unsigned __int8)sub_B19720(v69, v78, v64[4]) )
            {
LABEL_236:
              v136 = v73;
              v81 = v75;
              v35 = v69;
              v66 = v220;
              if ( v136 == *(_QWORD *)(v62 + 16) )
                goto LABEL_235;
              goto LABEL_237;
            }
            if ( v212 > v217 )
            {
              v209 = v73;
              v217 = v212;
            }
            v79 = v64[6];
            v80 = *(_QWORD *)(v62 + 16);
            v81 = v64 + 6;
            if ( v79 == v80 )
              goto LABEL_234;
            v213 = *((_DWORD *)v64 + 14);
            if ( (unsigned __int8)sub_B19720(v69, v80, v64[6]) )
            {
              v136 = v79;
              v35 = v69;
              v66 = v220;
              if ( v136 == *(_QWORD *)(v62 + 16) )
                goto LABEL_235;
LABEL_237:
              v137 = (unsigned int)v251;
              v138 = (unsigned int)v251 + 1LL;
              if ( v138 > HIDWORD(v251) )
              {
                v223 = v136;
                sub_C8D5F0((__int64)&v250, v252, v138, 8u, v76, v77);
                v137 = (unsigned int)v251;
                v136 = v223;
              }
              v64 = v81;
              *(_QWORD *)(v250 + 8 * v137) = v136;
              LODWORD(v251) = (_DWORD)v251 + 1;
LABEL_143:
              if ( v66 == v64 )
                goto LABEL_157;
              v84 = v64 + 2;
              if ( v66 == v64 + 2 )
                goto LABEL_157;
              v85 = v62;
              v221 = v66;
              v86 = v64;
              v87 = v85;
              while ( 1 )
              {
LABEL_150:
                v88 = *v84;
                v89 = *(_QWORD *)(v87 + 16);
                if ( *v84 == v89 )
                  goto LABEL_149;
                v215 = *((_DWORD *)v84 + 2);
                if ( !(unsigned __int8)sub_B19720(v35, v89, *v84) )
                  break;
                if ( v88 == *(_QWORD *)(v87 + 16) )
                  goto LABEL_149;
                v92 = (unsigned int)v251;
                v93 = (unsigned int)v251 + 1LL;
                if ( v93 > HIDWORD(v251) )
                {
                  sub_C8D5F0((__int64)&v250, v252, v93, 8u, v90, v91);
                  v92 = (unsigned int)v251;
                }
                v84 += 2;
                *(_QWORD *)(v250 + 8 * v92) = v88;
                LODWORD(v251) = (_DWORD)v251 + 1;
                if ( v221 == v84 )
                {
LABEL_156:
                  v94 = v87;
                  v66 = v221;
                  v64 = v86;
                  v62 = v94;
                  goto LABEL_157;
                }
              }
              if ( v217 < v215 )
              {
                v209 = v88;
                v217 = v215;
              }
              v86 += 2;
              *(v86 - 2) = *v84;
              *((_DWORD *)v86 - 2) = *((_DWORD *)v84 + 2);
LABEL_149:
              v84 += 2;
              if ( v221 == v84 )
                goto LABEL_156;
              goto LABEL_150;
            }
            if ( v217 < v213 )
            {
              v209 = v79;
              v217 = v213;
            }
            v64 += 8;
            if ( v214 == v64 )
            {
              v35 = v69;
              v66 = v220;
              v67 = ((char *)v220 - (char *)v64) >> 4;
              goto LABEL_317;
            }
          }
          v139 = v82;
          v35 = v69;
          v66 = v220;
          if ( v139 == *(_QWORD *)(v62 + 16) )
            goto LABEL_143;
LABEL_241:
          v140 = (unsigned int)v251;
          v141 = (unsigned int)v251 + 1LL;
          if ( v141 > HIDWORD(v251) )
          {
            sub_C8D5F0((__int64)&v250, v252, v141, 8u, v71, v72);
            v140 = (unsigned int)v251;
          }
          *(_QWORD *)(v250 + 8 * v140) = v139;
          LODWORD(v251) = (_DWORD)v251 + 1;
          goto LABEL_143;
        }
        v217 = 0;
        v209 = 0;
LABEL_317:
        if ( v67 != 2 )
        {
          if ( v67 != 3 )
          {
            if ( v67 != 1 )
            {
              v64 = v66;
              goto LABEL_157;
            }
            goto LABEL_331;
          }
          v139 = *v64;
          v158 = *(_QWORD *)(v62 + 16);
          if ( *v64 == v158 )
            goto LABEL_143;
          v224 = *((_DWORD *)v64 + 2);
          if ( (unsigned __int8)sub_B19720(v35, v158, *v64) )
          {
LABEL_336:
            if ( v139 == *(_QWORD *)(v62 + 16) )
              goto LABEL_143;
            goto LABEL_241;
          }
          if ( v217 < v224 )
          {
            v217 = v224;
            v209 = v139;
          }
          v64 += 2;
        }
        v139 = *v64;
        v159 = *(_QWORD *)(v62 + 16);
        if ( *v64 == v159 )
          goto LABEL_143;
        v225 = *((_DWORD *)v64 + 2);
        if ( (unsigned __int8)sub_B19720(v35, v159, *v64) )
          goto LABEL_336;
        if ( v225 > v217 )
        {
          v217 = v225;
          v209 = v139;
        }
        v64 += 2;
LABEL_331:
        v139 = *v64;
        v160 = *(_QWORD *)(v62 + 16);
        if ( *v64 == v160 )
          goto LABEL_143;
        v226 = *((_DWORD *)v64 + 2);
        if ( (unsigned __int8)sub_B19720(v35, v160, *v64) )
          goto LABEL_336;
        v64 = v66;
        if ( v217 >= v226 )
          v139 = v209;
        v209 = v139;
LABEL_157:
        v95 = *(__int64 **)v62;
        v96 = *(_QWORD *)v62 + 16LL * *(unsigned int *)(v62 + 8) - (_QWORD)v66;
        v97 = v96 >> 4;
        if ( v96 > 0 )
        {
          v98 = v64;
          do
          {
            v99 = *v66;
            v98 += 2;
            v66 += 2;
            *(v98 - 2) = v99;
            *((_DWORD *)v98 - 2) = *((_DWORD *)v66 - 2);
            --v97;
          }
          while ( v97 );
          v95 = *(__int64 **)v62;
          v64 = (__int64 *)((char *)v64 + v96);
        }
        *(_QWORD *)(v62 + 16) = v209;
        *(_DWORD *)(v62 + 8) = ((char *)v64 - (char *)v95) >> 4;
        if ( v210 <= 9 )
        {
          v267 = &v269;
          sub_2240A50((__int64 *)&v267, 1u, 0);
          v105 = v267;
          LOBYTE(v106) = v210;
        }
        else
        {
          if ( v210 <= 0x63 )
          {
            v267 = &v269;
            sub_2240A50((__int64 *)&v267, 2u, 0);
            v105 = v267;
            v106 = v210;
LABEL_311:
            v157 = 2 * v106;
            *((_BYTE *)v105 + 1) = a00010203040506[(unsigned int)(v157 + 1)];
            *(_BYTE *)v105 = a00010203040506[v157];
            goto LABEL_176;
          }
          if ( v210 <= 0x3E7 )
          {
            v104 = 3;
          }
          else
          {
            v100 = v210;
            if ( v210 <= 0x270F )
            {
              v104 = 4;
            }
            else
            {
              v101 = 1;
              do
              {
                v102 = v100;
                v103 = v101;
                v101 += 4;
                v100 /= 0x2710u;
                if ( v102 <= 0x1869F )
                {
                  v104 = v101;
                  goto LABEL_171;
                }
                if ( (unsigned int)v100 <= 0x63 )
                {
                  v104 = v103 + 5;
                  v267 = &v269;
                  goto LABEL_172;
                }
                if ( (unsigned int)v100 <= 0x3E7 )
                {
                  v104 = v103 + 6;
                  goto LABEL_171;
                }
              }
              while ( (unsigned int)v100 > 0x270F );
              v104 = v103 + 7;
            }
          }
LABEL_171:
          v267 = &v269;
LABEL_172:
          sub_2240A50((__int64 *)&v267, v104, 0);
          v105 = v267;
          v106 = v210;
          v107 = v268 - 1;
          do
          {
            v108 = v106 % 0x64;
            v109 = v106;
            v106 /= 0x64u;
            v110 = 2 * v108;
            v111 = (unsigned int)(v110 + 1);
            LOBYTE(v110) = a00010203040506[v110];
            *((_BYTE *)v105 + v107) = a00010203040506[v111];
            v112 = v107 - 1;
            v107 -= 2;
            *((_BYTE *)v105 + v112) = v110;
          }
          while ( v109 > 0x270F );
          if ( v109 > 0x3E7 )
            goto LABEL_311;
        }
        *(_BYTE *)v105 = v106 + 48;
LABEL_176:
        v113 = (__m128i *)sub_2241130((unsigned __int64 *)&v267, 0, 0, "cold.", 5u);
        v287 = &v289;
        if ( (__m128i *)v113->m128i_i64[0] == &v113[1] )
        {
          v289 = _mm_loadu_si128(v113 + 1);
        }
        else
        {
          v287 = (__m128i *)v113->m128i_i64[0];
          v289.m128i_i64[0] = v113[1].m128i_i64[0];
        }
        v288 = v113->m128i_u64[1];
        v113->m128i_i64[0] = (__int64)v113[1].m128i_i64;
        v113->m128i_i64[1] = 0;
        v113[1].m128i_i8[0] = 0;
        sub_29AFB10((unsigned int)v273, v250, (_DWORD)v251, v35, 0, 0, 0, v207, 0, 0, 0, (__int64)&v287, 0);
        if ( v287 != &v289 )
          j_j___libc_free_0((unsigned __int64)v287);
        if ( v267 != &v269 )
          j_j___libc_free_0((unsigned __int64)v267);
        if ( !(unsigned __int8)sub_29AB1F0(v273) )
          goto LABEL_252;
        v218 = sub_25F0930(a1, (__int64)v273, (__int64)&v250, v204);
        if ( !v218 )
          goto LABEL_252;
        v118 = (__int64 *)v250;
        v119 = 8LL * (unsigned int)v251;
        v216 = (__int64 *)(v250 + v119);
        v120 = v119 >> 3;
        v121 = v119 >> 5;
        if ( !v121 )
          goto LABEL_281;
        v122 = (__int64 *)(v250 + 32 * v121);
        do
        {
          v116 = *v118;
          if ( v232 )
          {
            v123 = (unsigned __int64)v229;
            v124 = &v229[HIDWORD(v230)];
            v115 = (__int64)v229;
            if ( v229 != v124 )
            {
              v114 = v229;
              while ( v116 != *v114 )
              {
                if ( v124 == ++v114 )
                  goto LABEL_246;
              }
              goto LABEL_191;
            }
LABEL_246:
            v116 = v118[1];
            v117 = (__int64)(v118 + 1);
            goto LABEL_247;
          }
          if ( sub_C8CA60((__int64)&v228, v116) )
            goto LABEL_191;
          v116 = v118[1];
          v117 = (__int64)(v118 + 1);
          if ( v232 )
          {
            v123 = (unsigned __int64)v229;
            v115 = (__int64)v229;
            v124 = &v229[HIDWORD(v230)];
LABEL_247:
            if ( v124 != (__int64 *)v123 )
            {
              v114 = (__int64 *)v123;
              while ( v116 != *v114 )
              {
                if ( v124 == ++v114 )
                  goto LABEL_263;
              }
LABEL_251:
              if ( v216 != (__int64 *)v117 )
                goto LABEL_252;
              goto LABEL_192;
            }
LABEL_263:
            v116 = v118[2];
            v117 = (__int64)(v118 + 2);
            goto LABEL_264;
          }
          v154 = sub_C8CA60((__int64)&v228, v116);
          v117 = (__int64)(v118 + 1);
          if ( v154 )
            goto LABEL_251;
          v116 = v118[2];
          v117 = (__int64)(v118 + 2);
          if ( v232 )
          {
            v123 = (unsigned __int64)v229;
            v115 = (__int64)v229;
            v124 = &v229[HIDWORD(v230)];
LABEL_264:
            if ( (__int64 *)v123 != v124 )
            {
              v114 = (__int64 *)v123;
              while ( v116 != *v114 )
              {
                if ( v124 == ++v114 )
                  goto LABEL_272;
              }
              goto LABEL_251;
            }
LABEL_272:
            v116 = v118[3];
            v114 = v118 + 3;
            goto LABEL_273;
          }
          v155 = sub_C8CA60((__int64)&v228, v116);
          v117 = (__int64)(v118 + 2);
          if ( v155 )
            goto LABEL_251;
          v116 = v118[3];
          v114 = v118 + 3;
          if ( !v232 )
          {
            v150 = sub_C8CA60((__int64)&v228, v116);
            v114 = v118 + 3;
            if ( v150 )
              goto LABEL_277;
            goto LABEL_279;
          }
          v123 = (unsigned __int64)v229;
          v115 = (__int64)v229;
          v124 = &v229[HIDWORD(v230)];
LABEL_273:
          if ( (__int64 *)v123 != v124 )
          {
            while ( v116 != *(_QWORD *)v115 )
            {
              v115 += 8;
              if ( v124 == (__int64 *)v115 )
                goto LABEL_279;
            }
LABEL_277:
            v118 = v114;
            goto LABEL_191;
          }
LABEL_279:
          v118 += 4;
        }
        while ( v122 != v118 );
        v120 = v216 - v118;
LABEL_281:
        switch ( v120 )
        {
          case 2LL:
            LOBYTE(v115) = v232;
            goto LABEL_364;
          case 3LL:
            v115 = v232;
            v180 = *v118;
            if ( v232 )
            {
              v152 = v229;
              v153 = HIDWORD(v230);
              v117 = (__int64)&v229[HIDWORD(v230)];
              v114 = v229;
              if ( v229 != (__int64 *)v117 )
              {
                while ( v180 != *v114 )
                {
                  if ( (__int64 *)v117 == ++v114 )
                    goto LABEL_386;
                }
                goto LABEL_191;
              }
              v179 = v118[1];
              ++v118;
              goto LABEL_366;
            }
            if ( sub_C8CA60((__int64)&v228, v180) )
              goto LABEL_191;
            LOBYTE(v115) = v232;
LABEL_386:
            ++v118;
LABEL_364:
            v179 = *v118;
            if ( !(_BYTE)v115 )
            {
              if ( sub_C8CA60((__int64)&v228, v179) )
                goto LABEL_191;
              v218 = v232;
LABEL_373:
              v151 = *++v118;
              if ( !v218 )
                goto LABEL_374;
LABEL_285:
              v152 = v229;
              v153 = HIDWORD(v230);
LABEL_286:
              v114 = &v152[v153];
              if ( v152 != v114 )
              {
                while ( v151 != *v152 )
                {
                  if ( v114 == ++v152 )
                    goto LABEL_192;
                }
                goto LABEL_191;
              }
              break;
            }
            v152 = v229;
            v153 = HIDWORD(v230);
LABEL_366:
            v115 = (__int64)&v152[v153];
            v114 = v152;
            if ( v152 != (__int64 *)v115 )
            {
              while ( *v114 != v179 )
              {
                if ( (__int64 *)v115 == ++v114 )
                  goto LABEL_373;
              }
              goto LABEL_191;
            }
            v151 = v118[1];
            ++v118;
            goto LABEL_286;
          case 1LL:
            v151 = *v118;
            if ( v232 )
              goto LABEL_285;
LABEL_374:
            if ( sub_C8CA60((__int64)&v228, v151) )
            {
LABEL_191:
              if ( v216 == v118 )
                break;
LABEL_252:
              v142 = (__int64 *)v250;
              if ( v250 + 8LL * (unsigned int)v251 == v250 )
                goto LABEL_214;
              v222 = (__int64 *)(v250 + 8LL * (unsigned int)v251);
              while ( 1 )
              {
                v143 = *v142;
                if ( (unsigned __int8)sub_B19720(v35, v208, *v142) && (sub_B19AA0(v206, v143, v208), v148) )
                {
                  if ( !v238 )
                    goto LABEL_300;
                }
                else
                {
                  sub_B19AA0(v206, v208, v143);
                  if ( !v156 || !(unsigned __int8)sub_B19720(v35, v143, v208) )
                    goto LABEL_261;
                  if ( !v238 )
                    goto LABEL_300;
                }
                v149 = v235;
                v145 = HIDWORD(v236);
                v144 = &v235[HIDWORD(v236)];
                if ( v235 != v144 )
                {
                  while ( v143 != *v149 )
                  {
                    if ( v144 == ++v149 )
                      goto LABEL_302;
                  }
LABEL_261:
                  if ( v222 == ++v142 )
                    goto LABEL_214;
                  continue;
                }
LABEL_302:
                if ( HIDWORD(v236) >= (unsigned int)v236 )
                {
LABEL_300:
                  ++v142;
                  sub_C8CC70((__int64)&v234, v143, (__int64)v144, v145, v146, v147);
                  if ( v222 == v142 )
                    goto LABEL_214;
                  continue;
                }
                ++v142;
                ++HIDWORD(v236);
                *v144 = v143;
                ++v234;
                if ( v222 == v142 )
                  goto LABEL_214;
              }
            }
            break;
        }
LABEL_192:
        v125 = (__int64 *)v250;
        v126 = (unsigned __int64 *)(v250 + 8LL * (unsigned int)v251);
        if ( (unsigned __int64 *)v250 == v126 )
          goto LABEL_201;
        v127 = v232;
        while ( 1 )
        {
LABEL_194:
          v128 = *v125;
          if ( !v127 )
          {
LABEL_305:
            ++v125;
            sub_C8CC70((__int64)&v228, v128, (__int64)v114, v115, v116, v117);
            v127 = v232;
            if ( v126 == (unsigned __int64 *)v125 )
              goto LABEL_200;
            continue;
          }
          v129 = v229;
          v115 = HIDWORD(v230);
          v114 = &v229[HIDWORD(v230)];
          if ( v229 != v114 )
            break;
LABEL_307:
          if ( HIDWORD(v230) >= (unsigned int)v230 )
            goto LABEL_305;
          v115 = (unsigned int)(HIDWORD(v230) + 1);
          ++v125;
          ++HIDWORD(v230);
          *v114 = v128;
          v127 = v232;
          ++v228;
          if ( v126 == (unsigned __int64 *)v125 )
            goto LABEL_200;
        }
        while ( v128 != *v129 )
        {
          if ( v114 == ++v129 )
            goto LABEL_307;
        }
        if ( v126 != (unsigned __int64 *)++v125 )
          goto LABEL_194;
LABEL_200:
        v126 = (unsigned __int64 *)v250;
LABEL_201:
        v287 = (__m128i *)*v126;
        sub_25F1790((__int64)&v288, v273, (__int64)v114, v115, v116, v117);
        v132 = v306;
        if ( HIDWORD(v306) <= (unsigned int)v306 )
        {
          v161 = sub_C8D7D0((__int64)&v305, (__int64)v307, 0, 0x108u, (unsigned __int64 *)&v267, v131);
          v165 = (unsigned int)v306;
          v227 = (__int64 *)v161;
          v166 = v161;
          v167 = 33LL * (unsigned int)v306;
          v168 = v167 * 8 + v166;
          if ( v168 )
          {
            v169 = v168 + 8;
            *(_QWORD *)(v169 - 8) = v287;
            sub_25F1790(v169, (__int64 *)&v288, v165, v162, v163, v164);
            v165 = (unsigned int)v306;
            v167 = 33LL * (unsigned int)v306;
          }
          v170 = v305;
          v171 = &v305[v167];
          if ( v305 != &v305[v167] )
          {
            v172 = v227;
            do
            {
              if ( v172 )
              {
                *v172 = *v170;
                sub_25F1790((__int64)(v172 + 1), v170 + 1, v165, v162, v163, v164);
              }
              v170 += 33;
              v172 += 33;
            }
            while ( v171 != v170 );
            v171 = v305;
            v173 = &v305[33 * (unsigned int)v306];
            if ( v305 != v173 )
            {
              do
              {
                v173 -= 33;
                v174 = v173[31];
                if ( (__int64 *)v174 != v173 + 33 )
                  _libc_free(v174);
                sub_C7D6A0(v173[28], 8LL * *((unsigned int *)v173 + 60), 8);
                v175 = v173[22];
                if ( (__int64 *)v175 != v173 + 24 )
                  j_j___libc_free_0(v175);
                v176 = v173[14];
                if ( (__int64 *)v176 != v173 + 16 )
                  _libc_free(v176);
                v177 = v173[12];
                if ( (__int64 *)v177 != v173 + 14 )
                  _libc_free(v177);
                sub_C7D6A0(v173[9], 8LL * *((unsigned int *)v173 + 22), 8);
              }
              while ( v171 != v173 );
              v171 = v305;
            }
          }
          v178 = (int)v267;
          if ( v171 != (__int64 *)v307 )
            _libc_free((unsigned __int64)v171);
          LODWORD(v306) = v306 + 1;
          HIDWORD(v306) = v178;
          v305 = v227;
        }
        else
        {
          v133 = (__int64)v305;
          v134 = &v305[33 * (unsigned int)v306];
          if ( v134 )
          {
            *v134 = (__int64)v287;
            sub_25F1790((__int64)(v134 + 1), (__int64 *)&v288, (__int64)v134, v133, v130, v131);
            v132 = v306;
          }
          LODWORD(v306) = v132 + 1;
        }
        if ( v303 != &v304 )
          _libc_free((unsigned __int64)v303);
        sub_C7D6A0(v301, 8LL * v302, 8);
        if ( v297 != &v299 )
          j_j___libc_free_0((unsigned __int64)v297);
        if ( (char *)v293[0] != &v294 )
          _libc_free(v293[0]);
        if ( v292 != v293 )
          _libc_free((unsigned __int64)v292);
        sub_C7D6A0(v290, 8LL * v291, 8);
        ++v210;
LABEL_214:
        if ( v285 != &v286 )
          _libc_free((unsigned __int64)v285);
        sub_C7D6A0(v283[4], 8LL * v284, 8);
        if ( v282 != v283 )
          j_j___libc_free_0((unsigned __int64)v282);
        if ( v279 != &v281 )
          _libc_free((unsigned __int64)v279);
        if ( v277 != (unsigned __int64 *)&v279 )
          _libc_free((unsigned __int64)v277);
        sub_C7D6A0(v275, 8LL * v276, 8);
        if ( (_DWORD *)v250 != v252 )
          _libc_free(v250);
        v63 = *(_QWORD *)(v62 + 16);
      }
      while ( v63 );
LABEL_225:
      v62 += 32LL;
    }
    while ( v200 != v62 );
    v135 = (unsigned __int64)v263;
    v61 = v262;
    if ( v263 != v262 )
    {
      do
      {
        if ( (unsigned __int64 *)*v61 != v61 + 2 )
          _libc_free(*v61);
        v61 += 4;
      }
      while ( (unsigned __int64 *)v135 != v61 );
      v61 = v262;
    }
LABEL_231:
    if ( v61 )
      j_j___libc_free_0((unsigned __int64)v61);
LABEL_57:
    v203 -= 8;
  }
  while ( v202 != v203 );
  v38 = v35;
  v39 = 0;
  if ( !(_DWORD)v306 )
    goto LABEL_59;
LABEL_389:
  sub_29B4290(&v287, a2);
  v181 = v305;
  v182 = &v305[33 * (unsigned int)v306];
  if ( v182 != v305 )
  {
    do
    {
      v183 = (__int64)(v181 + 1);
      v184 = *v181;
      v181 += 33;
      sub_25F25C0(a1, v184, v183, (__int64)&v287, (__int64)v199, (__int64)v204, v197);
    }
    while ( v182 != v181 );
  }
  sub_C7D6A0(v298, 8LL * v300, 8);
  v185 = v296;
  if ( v296 )
  {
    v186 = v295;
    v187 = v295 + 40LL * v296;
    do
    {
      if ( *(_QWORD *)v186 != -8192 && *(_QWORD *)v186 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v186 + 16), 8LL * *(unsigned int *)(v186 + 32), 8);
      v186 += 40;
    }
    while ( v187 != v186 );
    v185 = v296;
  }
  sub_C7D6A0(v295, 40 * v185, 8);
  if ( v287 != &v289 )
    _libc_free((unsigned __int64)v287);
  v39 = 1;
LABEL_59:
  if ( v206 )
  {
    v40 = *(_QWORD *)(v206 + 48);
    v41 = v40 + 8LL * *(unsigned int *)(v206 + 56);
    if ( v40 != v41 )
    {
      do
      {
        v42 = *(_QWORD *)(v41 - 8);
        v41 -= 8LL;
        if ( v42 )
        {
          v43 = *(_QWORD *)(v42 + 24);
          if ( v43 != v42 + 40 )
            _libc_free(v43);
          j_j___libc_free_0(v42);
        }
      }
      while ( v40 != v41 );
      v41 = *(_QWORD *)(v206 + 48);
    }
    if ( v41 != v206 + 64 )
      _libc_free(v41);
    if ( *(_QWORD *)v206 != v206 + 16 )
      _libc_free(*(_QWORD *)v206);
    j_j___libc_free_0(v206);
  }
  if ( v38 )
  {
    v44 = *(_QWORD *)(v38 + 24);
    v45 = v44 + 8LL * *(unsigned int *)(v38 + 32);
    if ( v44 != v45 )
    {
      do
      {
        v46 = *(_QWORD *)(v45 - 8);
        v45 -= 8LL;
        if ( v46 )
        {
          v47 = *(_QWORD *)(v46 + 24);
          if ( v47 != v46 + 40 )
            _libc_free(v47);
          j_j___libc_free_0(v46);
        }
      }
      while ( v44 != v45 );
      v45 = *(_QWORD *)(v38 + 24);
    }
    if ( v45 != v38 + 40 )
      _libc_free(v45);
    if ( *(_QWORD *)v38 != v38 + 16 )
      _libc_free(*(_QWORD *)v38);
    j_j___libc_free_0(v38);
  }
  v202 = v246;
LABEL_86:
  if ( v202 != v248 )
    _libc_free((unsigned __int64)v202);
  v48 = v305;
  v49 = (unsigned __int64)&v305[33 * (unsigned int)v306];
  if ( v305 != (__int64 *)v49 )
  {
    do
    {
      v49 -= 264LL;
      v50 = *(_QWORD *)(v49 + 248);
      if ( v50 != v49 + 264 )
        _libc_free(v50);
      sub_C7D6A0(*(_QWORD *)(v49 + 224), 8LL * *(unsigned int *)(v49 + 240), 8);
      v51 = *(_QWORD *)(v49 + 176);
      if ( v51 != v49 + 192 )
        j_j___libc_free_0(v51);
      v52 = *(_QWORD *)(v49 + 112);
      if ( v52 != v49 + 128 )
        _libc_free(v52);
      v53 = *(_QWORD *)(v49 + 96);
      if ( v53 != v49 + 112 )
        _libc_free(v53);
      sub_C7D6A0(*(_QWORD *)(v49 + 72), 8LL * *(unsigned int *)(v49 + 88), 8);
    }
    while ( v48 != (__int64 *)v49 );
    v49 = (unsigned __int64)v305;
  }
  if ( (_BYTE *)v49 != v307 )
    _libc_free(v49);
  if ( !v244 )
    _libc_free((unsigned __int64)v241);
  if ( !v238 )
    _libc_free((unsigned __int64)v235);
  if ( !v232 )
    _libc_free((unsigned __int64)v229);
  return v39;
}
