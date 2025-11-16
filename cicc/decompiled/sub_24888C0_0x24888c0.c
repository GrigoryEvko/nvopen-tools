// Function: sub_24888C0
// Address: 0x24888c0
//
_QWORD *__fastcall sub_24888C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdi
  const char *v7; // rax
  __int64 v8; // rdx
  const char *v10; // rax
  unsigned __int64 v11; // rdx
  const char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  const char *v15; // rbx
  __int64 v16; // rax
  char *v17; // rsi
  __int64 v18; // rdx
  char *v19; // rbx
  size_t v20; // rax
  __int64 *v21; // rax
  __m128i *v22; // rax
  __int64 v23; // rbx
  __int64 *v24; // rax
  __m128i *v25; // rdi
  __int64 v26; // rdx
  __int64 *v27; // rbx
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rbx
  __int64 *v33; // rbx
  unsigned __int64 v34; // r13
  __int64 v35; // r15
  unsigned __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 *v42; // r12
  unsigned __int64 v43; // r15
  unsigned __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // rdx
  __int64 v47; // rbx
  __int64 v48; // r15
  __int64 i; // rbx
  __int64 v50; // r13
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // r12
  _QWORD *v57; // rax
  _BYTE *v58; // r12
  __int64 v59; // r13
  __int64 v60; // rax
  char v61; // r15
  _QWORD *v62; // rax
  __int64 v63; // rbx
  __int64 v64; // r13
  const char *v65; // r12
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // r13
  __int64 v69; // r14
  unsigned int v70; // ebx
  __int64 v71; // rax
  __int64 v72; // rbx
  __int64 v73; // rdi
  unsigned int v74; // r13d
  unsigned __int64 v75; // r12
  _QWORD *v76; // rax
  _QWORD *v77; // r15
  __int64 v78; // rax
  __int64 v79; // r12
  __int64 v80; // r15
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rbx
  __int64 v84; // r13
  __int64 v85; // rdx
  __int64 v86; // r12
  _QWORD *v87; // rax
  __int64 v88; // r8
  __int64 v89; // rsi
  __int64 v90; // r9
  __int64 v91; // r15
  const char *v92; // rax
  int v93; // esi
  const char *v94; // rdx
  __int64 v95; // rax
  __int64 v96; // r11
  unsigned int v97; // r13d
  __int64 v98; // rbx
  const char *v99; // r13
  __int64 v100; // rdx
  unsigned int v101; // esi
  __int64 v102; // rax
  int v103; // eax
  __int64 v104; // rax
  __int64 v105; // r12
  __int64 v106; // r15
  __int64 v107; // rbx
  unsigned int v108; // r14d
  unsigned int v109; // r14d
  __int64 v110; // rbx
  __int64 v111; // r12
  __int64 v112; // r15
  __int64 v113; // rbx
  unsigned int v114; // r14d
  unsigned int v115; // r14d
  __int64 v116; // rbx
  __int64 v117; // r12
  __int64 v118; // rdx
  size_t v119; // rax
  __int64 v120; // r15
  __int64 v121; // rbx
  unsigned int v122; // r14d
  unsigned int v123; // r14d
  __int64 v124; // rbx
  __int64 v125; // rax
  __int64 *v126; // rdx
  _QWORD *v127; // rax
  __int64 v128; // r13
  const char *v129; // r12
  __int64 v130; // rdx
  unsigned int v131; // esi
  __int64 v132; // rax
  int v133; // edx
  int v134; // edx
  char v135; // dl
  int v136; // eax
  int v137; // r12d
  __int64 v138; // r12
  __int64 v139; // rax
  const char *v140; // r14
  __int64 v141; // rdx
  unsigned int v142; // esi
  int v143; // r12d
  __int64 v144; // r12
  __int64 v145; // rax
  const char *v146; // r14
  __int64 v147; // rdx
  unsigned int v148; // esi
  int v149; // r12d
  __int64 v150; // r12
  __int64 v151; // rax
  const char *v152; // r14
  __int64 v153; // rdx
  unsigned int v154; // esi
  __int64 v155; // [rsp+0h] [rbp-3C0h]
  __m128i *v156; // [rsp+18h] [rbp-3A8h]
  unsigned int v157; // [rsp+20h] [rbp-3A0h]
  int v158; // [rsp+24h] [rbp-39Ch]
  __int64 v159; // [rsp+28h] [rbp-398h]
  char *v160; // [rsp+38h] [rbp-388h]
  __int64 v161; // [rsp+40h] [rbp-380h]
  unsigned __int8 v162; // [rsp+48h] [rbp-378h]
  char v163; // [rsp+50h] [rbp-370h]
  __int64 v164; // [rsp+50h] [rbp-370h]
  unsigned __int64 v165; // [rsp+58h] [rbp-368h]
  __int64 v166; // [rsp+58h] [rbp-368h]
  __int64 v167; // [rsp+58h] [rbp-368h]
  __int64 v168; // [rsp+60h] [rbp-360h]
  __int64 v169; // [rsp+60h] [rbp-360h]
  char *v170; // [rsp+60h] [rbp-360h]
  __int64 v171; // [rsp+68h] [rbp-358h]
  __int64 v172; // [rsp+68h] [rbp-358h]
  __int64 v175; // [rsp+98h] [rbp-328h]
  __int64 *v176; // [rsp+98h] [rbp-328h]
  __int64 v177; // [rsp+98h] [rbp-328h]
  __int64 v178[4]; // [rsp+D0h] [rbp-2F0h] BYREF
  __int64 v179; // [rsp+F0h] [rbp-2D0h] BYREF
  unsigned __int8 v180; // [rsp+F8h] [rbp-2C8h]
  __int64 v181; // [rsp+100h] [rbp-2C0h]
  __int64 v182; // [rsp+108h] [rbp-2B8h]
  char v183; // [rsp+110h] [rbp-2B0h]
  char *v184; // [rsp+120h] [rbp-2A0h] BYREF
  size_t v185; // [rsp+128h] [rbp-298h]
  _QWORD v186[2]; // [rsp+130h] [rbp-290h] BYREF
  __int16 v187; // [rsp+140h] [rbp-280h]
  char *v188; // [rsp+150h] [rbp-270h] BYREF
  size_t v189; // [rsp+158h] [rbp-268h]
  _QWORD v190[2]; // [rsp+160h] [rbp-260h] BYREF
  __int16 v191; // [rsp+170h] [rbp-250h]
  _QWORD *v192; // [rsp+180h] [rbp-240h] BYREF
  __int64 v193; // [rsp+188h] [rbp-238h]
  _QWORD v194[2]; // [rsp+190h] [rbp-230h] BYREF
  __int16 v195; // [rsp+1A0h] [rbp-220h]
  _QWORD *v196; // [rsp+1B0h] [rbp-210h] BYREF
  unsigned __int64 v197; // [rsp+1B8h] [rbp-208h]
  _QWORD v198[2]; // [rsp+1C0h] [rbp-200h] BYREF
  __int16 v199; // [rsp+1D0h] [rbp-1F0h]
  __int64 *v200; // [rsp+1E0h] [rbp-1E0h] BYREF
  __int64 v201; // [rsp+1E8h] [rbp-1D8h]
  __int64 v202; // [rsp+1F0h] [rbp-1D0h]
  __int64 v203; // [rsp+1F8h] [rbp-1C8h]
  int v204; // [rsp+200h] [rbp-1C0h]
  int v205; // [rsp+204h] [rbp-1BCh]
  __int64 v206; // [rsp+208h] [rbp-1B8h]
  __int128 v207; // [rsp+210h] [rbp-1B0h]
  __int128 v208; // [rsp+220h] [rbp-1A0h]
  __int64 v209; // [rsp+230h] [rbp-190h] BYREF
  __int64 v210; // [rsp+238h] [rbp-188h]
  __int64 v211; // [rsp+240h] [rbp-180h] BYREF
  __int64 v212; // [rsp+248h] [rbp-178h]
  __int64 v213; // [rsp+250h] [rbp-170h]
  __int64 v214; // [rsp+258h] [rbp-168h]
  __int64 v215; // [rsp+260h] [rbp-160h]
  __m128i *v216; // [rsp+270h] [rbp-150h] BYREF
  unsigned __int64 v217; // [rsp+278h] [rbp-148h]
  __m128i v218; // [rsp+280h] [rbp-140h] BYREF
  __int64 v219; // [rsp+290h] [rbp-130h]
  const char *v220; // [rsp+300h] [rbp-C0h] BYREF
  __int64 v221; // [rsp+308h] [rbp-B8h]
  _BYTE v222[32]; // [rsp+310h] [rbp-B0h] BYREF
  __int64 v223; // [rsp+330h] [rbp-90h]
  __int64 v224; // [rsp+338h] [rbp-88h]
  __int64 v225; // [rsp+340h] [rbp-80h]
  _QWORD *v226; // [rsp+348h] [rbp-78h]
  void **v227; // [rsp+350h] [rbp-70h]
  _QWORD *v228; // [rsp+358h] [rbp-68h]
  __int64 v229; // [rsp+360h] [rbp-60h]
  int v230; // [rsp+368h] [rbp-58h]
  __int16 v231; // [rsp+36Ch] [rbp-54h]
  char v232; // [rsp+36Eh] [rbp-52h]
  __int64 v233; // [rsp+370h] [rbp-50h]
  __int64 v234; // [rsp+378h] [rbp-48h]
  void *v235; // [rsp+380h] [rbp-40h] BYREF
  _QWORD v236[7]; // [rsp+388h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a3 + 40);
  v204 = qword_4FE9AE8;
  if ( (_BYTE)qword_4FE93C8 )
  {
    v5 = -8;
    v4 = 8;
  }
  else
  {
    v4 = qword_4FE9A08;
    v5 = -(int)qword_4FE9A08;
  }
  v205 = v4;
  v6 = v3 + 312;
  v207 = 0;
  v208 = 0;
  v206 = v5;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  v200 = *(__int64 **)(v6 - 312);
  LODWORD(v201) = sub_AE2980(v6, 0)[1];
  v202 = sub_BCD140(v200, v201);
  v203 = sub_BCE3C0(v200, 0);
  if ( (*(_BYTE *)(a3 + 32) & 0xF) == 1 )
    goto LABEL_7;
  v7 = sub_BD5D20(a3);
  if ( qword_4FE9750 == v8 && (!qword_4FE9750 || !memcmp(qword_4FE9748, v7, qword_4FE9750)) )
    goto LABEL_7;
  v10 = sub_BD5D20(a3);
  if ( v11 > 9 && *(_QWORD *)v10 == 0x6F72706D656D5F5FLL && *((_WORD *)v10 + 4) == 24422 )
    goto LABEL_7;
  v12 = sub_BD5D20(a3);
  v221 = v13;
  v220 = v12;
  if ( sub_C931B0((__int64 *)&v220, " load]", 6u, 0) == -1 )
  {
    v163 = 0;
    goto LABEL_14;
  }
  v82 = sub_2A3F100(*(_QWORD *)(a3 + 40), "__memprof_init", 14, 0, 0, 0);
  v83 = *(_QWORD *)(a3 + 80);
  v84 = v82;
  v86 = v85;
  if ( !v83 )
    goto LABEL_210;
  v177 = *(_QWORD *)(v83 + 32);
  v87 = (_QWORD *)sub_AA48A0(v83 - 24);
  v88 = v177;
  v226 = v87;
  v227 = &v235;
  v228 = v236;
  v235 = &unk_49DA100;
  v220 = v222;
  v236[0] = &unk_49DA0B0;
  LOWORD(v225) = 1;
  v221 = 0x200000000LL;
  v229 = 0;
  v230 = 0;
  v231 = 512;
  v232 = 7;
  v233 = 0;
  v234 = 0;
  v223 = v83 - 24;
  v224 = v177;
  if ( v177 != v83 + 24 )
  {
    if ( v177 )
      v88 = v177 - 24;
    v89 = *(_QWORD *)sub_B46C60(v88);
    v216 = (__m128i *)v89;
    if ( v89 && (sub_B96E90((__int64)&v216, v89, 1), (v91 = (__int64)v216) != 0) )
    {
      v92 = v220;
      v93 = v221;
      v94 = &v220[16 * (unsigned int)v221];
      if ( v220 != v94 )
      {
        while ( 1 )
        {
          v90 = *(unsigned int *)v92;
          if ( !(_DWORD)v90 )
            break;
          v92 += 16;
          if ( v94 == v92 )
            goto LABEL_182;
        }
        *((_QWORD *)v92 + 1) = v216;
LABEL_106:
        sub_B91220((__int64)&v216, v91);
        goto LABEL_107;
      }
LABEL_182:
      if ( (unsigned int)v221 >= (unsigned __int64)HIDWORD(v221) )
      {
        if ( HIDWORD(v221) < (unsigned __int64)(unsigned int)v221 + 1 )
        {
          sub_C8D5F0((__int64)&v220, v222, (unsigned int)v221 + 1LL, 0x10u, (unsigned int)v221 + 1LL, v90);
          v94 = &v220[16 * (unsigned int)v221];
        }
        *(_QWORD *)v94 = 0;
        *((_QWORD *)v94 + 1) = v91;
        v91 = (__int64)v216;
        LODWORD(v221) = v221 + 1;
      }
      else
      {
        if ( v94 )
        {
          *(_DWORD *)v94 = 0;
          *((_QWORD *)v94 + 1) = v91;
          v93 = v221;
          v91 = (__int64)v216;
        }
        LODWORD(v221) = v93 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v220, 0);
      v91 = (__int64)v216;
    }
    if ( !v91 )
      goto LABEL_107;
    goto LABEL_106;
  }
LABEL_107:
  LOWORD(v219) = 257;
  sub_24869A0((__int64 *)&v220, v84, v86, 0, 0, (__int64)&v216, 0);
  nullsub_61();
  v235 = &unk_49DA100;
  nullsub_63();
  if ( v220 != v222 )
    _libc_free((unsigned __int64)v220);
  v163 = 1;
LABEL_14:
  v232 = 7;
  v14 = 0;
  v229 = 0;
  v15 = "load";
  v16 = *(_QWORD *)(a3 + 40);
  v230 = 0;
  v171 = v16;
  v233 = 0;
  v220 = v222;
  v221 = 0x200000000LL;
  v234 = 0;
  v226 = v200;
  v223 = 0;
  v227 = &v235;
  v224 = 0;
  v228 = v236;
  v231 = 512;
  LOWORD(v225) = 0;
  v235 = &unk_49DA100;
  v236[0] = &unk_49DA0B0;
  while ( 1 )
  {
    v184 = (char *)v186;
    v17 = (char *)v15;
    v18 = (__int64)&v15[strlen(v15)];
    v19 = (char *)byte_3F871B3;
    sub_2485610((__int64 *)&v184, v17, v18);
    if ( (_BYTE)qword_4FE93C8 )
      v19 = "hist_";
    v188 = (char *)v190;
    v20 = strlen(v19);
    sub_2485610((__int64 *)&v188, v19, (__int64)&v19[v20]);
    v192 = v194;
    v194[0] = v202;
    v193 = 0x200000001LL;
    v21 = (__int64 *)sub_BCB120(v226);
    v165 = sub_BCF480(v21, v194, 1, 0);
    v196 = v198;
    sub_2484EB0((__int64 *)&v196, (_BYTE *)qword_4FE9BC8, qword_4FE9BC8 + qword_4FE9BD0);
    sub_2241490((unsigned __int64 *)&v196, v188, v189);
    v22 = (__m128i *)sub_2241490((unsigned __int64 *)&v196, v184, v185);
    v216 = &v218;
    if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
    {
      v218 = _mm_loadu_si128(v22 + 1);
    }
    else
    {
      v216 = (__m128i *)v22->m128i_i64[0];
      v218.m128i_i64[0] = v22[1].m128i_i64[0];
    }
    v23 = 2 * (v14 + 3);
    v217 = v22->m128i_u64[1];
    v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
    v22->m128i_i64[1] = 0;
    v22[1].m128i_i8[0] = 0;
    v24 = (__int64 *)sub_BA8CA0(v171, (__int64)v216, v217, v165);
    v25 = v216;
    (&v200)[v23] = v24;
    *(__int64 *)((char *)&v201 + v23 * 8) = v26;
    if ( v25 != &v218 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( v196 != v198 )
      j_j___libc_free_0((unsigned __int64)v196);
    if ( v192 != v194 )
      _libc_free((unsigned __int64)v192);
    if ( v188 != (char *)v190 )
      j_j___libc_free_0((unsigned __int64)v188);
    if ( v184 != (char *)v186 )
      j_j___libc_free_0((unsigned __int64)v184);
    v15 = "store";
    if ( v14 == 1 )
      break;
    v14 = 1;
  }
  v196 = v198;
  v27 = (__int64 *)v203;
  v168 = v202;
  sub_2484EB0((__int64 *)&v196, (_BYTE *)qword_4FE9BC8, qword_4FE9BC8 + qword_4FE9BD0);
  if ( 0x3FFFFFFFFFFFFFFFLL - v197 <= 6 )
    goto LABEL_208;
  sub_2241490((unsigned __int64 *)&v196, "memmove", 7u);
  v218.m128i_i64[0] = (__int64)v27;
  v219 = v168;
  v28 = v197;
  v217 = 0x300000003LL;
  v166 = (__int64)v196;
  v218.m128i_i64[1] = (__int64)v27;
  v216 = &v218;
  v29 = sub_BCF480(v27, &v218, 3, 0);
  v30 = sub_BA8C10(v171, v166, v28, v29, 0);
  v32 = v31;
  if ( v216 != &v218 )
    _libc_free((unsigned __int64)v216);
  v209 = v30;
  v210 = v32;
  if ( v196 != v198 )
    j_j___libc_free_0((unsigned __int64)v196);
  v192 = v194;
  v33 = (__int64 *)v203;
  v169 = v202;
  sub_2484EB0((__int64 *)&v192, (_BYTE *)qword_4FE9BC8, qword_4FE9BC8 + qword_4FE9BD0);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 5 )
    goto LABEL_208;
  sub_2241490((unsigned __int64 *)&v192, "memcpy", 6u);
  v34 = v193;
  v218.m128i_i64[0] = (__int64)v33;
  v217 = 0x300000003LL;
  v35 = (__int64)v192;
  v219 = v169;
  v218.m128i_i64[1] = (__int64)v33;
  v216 = &v218;
  v36 = sub_BCF480(v33, &v218, 3, 0);
  v37 = sub_BA8C10(v171, v35, v34, v36, 0);
  v39 = v38;
  if ( v216 != &v218 )
    _libc_free((unsigned __int64)v216);
  v211 = v37;
  v212 = v39;
  if ( v192 != v194 )
    j_j___libc_free_0((unsigned __int64)v192);
  v40 = v202;
  v41 = sub_BCB2D0(v226);
  v42 = (__int64 *)v203;
  v188 = (char *)v190;
  sub_2484EB0((__int64 *)&v188, (_BYTE *)qword_4FE9BC8, qword_4FE9BC8 + qword_4FE9BD0);
  if ( 0x3FFFFFFFFFFFFFFFLL - v189 <= 5 )
LABEL_208:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v188, "memset", 6u);
  v218.m128i_i64[1] = v41;
  v218.m128i_i64[0] = (__int64)v42;
  v43 = v189;
  v170 = v188;
  v219 = v40;
  v217 = 0x300000003LL;
  v216 = &v218;
  v44 = sub_BCF480(v42, &v218, 3, 0);
  v45 = sub_BA8C10(v171, (__int64)v170, v43, v44, 0);
  v47 = v46;
  if ( v216 != &v218 )
    _libc_free((unsigned __int64)v216);
  v213 = v45;
  v214 = v47;
  if ( v188 != (char *)v190 )
    j_j___libc_free_0((unsigned __int64)v188);
  nullsub_61();
  v235 = &unk_49DA100;
  nullsub_63();
  if ( v220 != v222 )
    _libc_free((unsigned __int64)v220);
  v216 = &v218;
  v217 = 0x1000000000LL;
  v175 = *(_QWORD *)(a3 + 80);
  if ( v175 != a3 + 72 )
  {
    do
    {
      if ( !v175 )
        goto LABEL_210;
      v48 = *(_QWORD *)(v175 + 32);
      for ( i = v175 + 24; i != v48; v48 = *(_QWORD *)(v48 + 8) )
      {
        while ( 1 )
        {
          v50 = v48 - 24;
          if ( !v48 )
            v50 = 0;
          sub_2485090((__int64)&v220, v215, v50);
          if ( v222[16] )
            break;
          if ( *(_BYTE *)v50 == 85 )
          {
            v95 = *(_QWORD *)(v50 - 32);
            if ( v95 )
            {
              if ( !*(_BYTE *)v95
                && *(_QWORD *)(v95 + 24) == *(_QWORD *)(v50 + 80)
                && (*(_BYTE *)(v95 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v95 + 36) - 238) <= 7
                && ((1LL << (*(_BYTE *)(v95 + 36) + 18)) & 0xAD) != 0 )
              {
                break;
              }
            }
          }
          v48 = *(_QWORD *)(v48 + 8);
          if ( i == v48 )
            goto LABEL_59;
        }
        v53 = (unsigned int)v217;
        v54 = (unsigned int)v217 + 1LL;
        if ( v54 > HIDWORD(v217) )
        {
          sub_C8D5F0((__int64)&v216, &v218, v54, 8u, v51, v52);
          v53 = (unsigned int)v217;
        }
        v216->m128i_i64[v53] = v50;
        LODWORD(v217) = v217 + 1;
      }
LABEL_59:
      v175 = *(_QWORD *)(v175 + 8);
    }
    while ( a3 + 72 != v175 );
    if ( (_DWORD)v217 )
    {
      v55 = *(_QWORD *)(a3 + 80);
      if ( v55 )
      {
        v56 = *(_QWORD *)(v55 + 32);
        if ( v56 )
          v56 -= 24;
        v57 = (_QWORD *)sub_BD5C60(v56);
        v229 = 0;
        v226 = v57;
        v220 = v222;
        v227 = &v235;
        v221 = 0x200000000LL;
        v228 = v236;
        v231 = 512;
        LOWORD(v225) = 0;
        v230 = 0;
        v232 = 7;
        v235 = &unk_49DA100;
        v233 = 0;
        v234 = 0;
        v236[0] = &unk_49DA0B0;
        v223 = 0;
        v224 = 0;
        sub_D5F1F0((__int64)&v220, v56);
        v58 = sub_BA8D60(*(_QWORD *)(a3 + 40), (__int64)"__memprof_shadow_memory_dynamic_address", 0x27u, v202);
        if ( !(unsigned int)sub_BAA5B0(*(_QWORD *)(a3 + 40)) )
          v58[33] |= 0x40u;
        v59 = v202;
        v195 = 257;
        v60 = sub_AA4E30(v223);
        v61 = sub_AE5020(v60, v59);
        v199 = 257;
        v62 = sub_BD2C40(80, unk_3F10A14);
        v63 = (__int64)v62;
        if ( v62 )
          sub_B4D190((__int64)v62, v59, (__int64)v58, (__int64)&v196, 0, v61, 0, 0);
        (*(void (__fastcall **)(_QWORD *, __int64, _QWORD **, __int64, __int64))(*v228 + 16LL))(
          v228,
          v63,
          &v192,
          v224,
          v225);
        v64 = (__int64)v220;
        v65 = &v220[16 * (unsigned int)v221];
        if ( v220 != v65 )
        {
          do
          {
            v66 = *(_QWORD *)(v64 + 8);
            v67 = *(_DWORD *)v64;
            v64 += 16;
            sub_B99FD0(v63, v67, v66);
          }
          while ( v65 != (const char *)v64 );
        }
        v215 = v63;
        nullsub_61();
        v235 = &unk_49DA100;
        nullsub_63();
        if ( v220 != v222 )
          _libc_free((unsigned __int64)v220);
        v156 = (__m128i *)((char *)v216 + 8 * (unsigned int)v217);
        if ( v216 == v156 )
          goto LABEL_167;
        v176 = (__int64 *)v216;
        v158 = 0;
        while ( 1 )
        {
          if ( dword_4FE9668 < 0 || dword_4FE9588 < 0 || dword_4FE9588 >= v158 && dword_4FE9668 <= v158 )
          {
            v68 = *v176;
            sub_2485090((__int64)&v179, v215, *v176);
            if ( !v183 )
            {
              v226 = (_QWORD *)sub_BD5C60(v68);
              v220 = v222;
              v227 = &v235;
              v221 = 0x200000000LL;
              v228 = v236;
              LOWORD(v225) = 0;
              v229 = 0;
              v230 = 0;
              v235 = &unk_49DA100;
              v231 = 512;
              v232 = 7;
              v236[0] = &unk_49DA0B0;
              v233 = 0;
              v234 = 0;
              v223 = 0;
              v224 = 0;
              sub_D5F1F0((__int64)&v220, v68);
              v102 = *(_QWORD *)(v68 - 32);
              if ( !v102 || *(_BYTE *)v102 )
                BUG();
              if ( *(_QWORD *)(v102 + 24) != *(_QWORD *)(v68 + 80) )
LABEL_209:
                BUG();
              v103 = *(_DWORD *)(v102 + 36);
              if ( v103 == 238 || (unsigned int)(v103 - 240) <= 1 )
              {
                v195 = 257;
                v117 = v202;
                v118 = *(_DWORD *)(v68 + 4) & 0x7FFFFFF;
                v184 = *(char **)(v68 - 32 * v118);
                v119 = *(_QWORD *)(v68 + 32 * (1 - v118));
                v191 = 257;
                v185 = v119;
                v120 = *(_QWORD *)(v68 + 32 * (2 - v118));
                v121 = *(_QWORD *)(v120 + 8);
                v122 = sub_BCB060(v121);
                v123 = (v122 <= (unsigned int)sub_BCB060(v117)) + 38;
                if ( v117 == v121 )
                {
                  v124 = v120;
                }
                else
                {
                  v124 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v227 + 15))(
                           v227,
                           v123,
                           v120,
                           v117);
                  if ( !v124 )
                  {
                    v199 = 257;
                    v124 = sub_B51D30(v123, v120, v117, (__int64)&v196, 0, 0);
                    if ( (unsigned __int8)sub_920620(v124) )
                    {
                      v137 = v230;
                      if ( v229 )
                        sub_B99FD0(v124, 3u, v229);
                      sub_B45150(v124, v137);
                    }
                    (*(void (__fastcall **)(_QWORD *, __int64, char **, __int64, __int64))(*v228 + 16LL))(
                      v228,
                      v124,
                      &v188,
                      v224,
                      v225);
                    v138 = (__int64)v220;
                    v139 = 16LL * (unsigned int)v221;
                    v140 = &v220[v139];
                    if ( v220 != &v220[v139] )
                    {
                      do
                      {
                        v141 = *(_QWORD *)(v138 + 8);
                        v142 = *(_DWORD *)v138;
                        v138 += 16;
                        sub_B99FD0(v124, v142, v141);
                      }
                      while ( v140 != (const char *)v138 );
                    }
                  }
                }
                v186[0] = v124;
                v125 = *(_QWORD *)(v68 - 32);
                if ( !v125 || *(_BYTE *)v125 || *(_QWORD *)(v125 + 24) != *(_QWORD *)(v68 + 80) )
                  goto LABEL_209;
                v126 = &v209;
                if ( *(_DWORD *)(v125 + 36) != 241 )
                  v126 = &v211;
                sub_24869A0((__int64 *)&v220, *v126, v126[1], (__int64 *)&v184, 3, (__int64)&v192, 0);
              }
              else if ( ((v103 - 243) & 0xFFFFFFFD) == 0 )
              {
                v195 = 257;
                v104 = *(_QWORD *)(v68 - 32LL * (*(_DWORD *)(v68 + 4) & 0x7FFFFFF));
                v191 = 257;
                v178[0] = v104;
                v105 = sub_BCB2D0(v226);
                v106 = *(_QWORD *)(v68 + 32 * (1LL - (*(_DWORD *)(v68 + 4) & 0x7FFFFFF)));
                v107 = *(_QWORD *)(v106 + 8);
                v108 = sub_BCB060(v107);
                v109 = (v108 <= (unsigned int)sub_BCB060(v105)) + 38;
                if ( v105 == v107 )
                {
                  v110 = v106;
                }
                else
                {
                  v110 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v227 + 15))(
                           v227,
                           v109,
                           v106,
                           v105);
                  if ( !v110 )
                  {
                    v199 = 257;
                    v110 = sub_B51D30(v109, v106, v105, (__int64)&v196, 0, 0);
                    if ( (unsigned __int8)sub_920620(v110) )
                    {
                      v149 = v230;
                      if ( v229 )
                        sub_B99FD0(v110, 3u, v229);
                      sub_B45150(v110, v149);
                    }
                    (*(void (__fastcall **)(_QWORD *, __int64, char **, __int64, __int64))(*v228 + 16LL))(
                      v228,
                      v110,
                      &v188,
                      v224,
                      v225);
                    v150 = (__int64)v220;
                    v151 = 16LL * (unsigned int)v221;
                    v152 = &v220[v151];
                    if ( v220 != &v220[v151] )
                    {
                      do
                      {
                        v153 = *(_QWORD *)(v150 + 8);
                        v154 = *(_DWORD *)v150;
                        v150 += 16;
                        sub_B99FD0(v110, v154, v153);
                      }
                      while ( v152 != (const char *)v150 );
                    }
                  }
                }
                v178[1] = v110;
                v111 = v202;
                v187 = 257;
                v112 = *(_QWORD *)(v68 + 32 * (2LL - (*(_DWORD *)(v68 + 4) & 0x7FFFFFF)));
                v113 = *(_QWORD *)(v112 + 8);
                v114 = sub_BCB060(v113);
                v115 = (v114 <= (unsigned int)sub_BCB060(v111)) + 38;
                if ( v111 == v113 )
                {
                  v116 = v112;
                }
                else
                {
                  v116 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v227 + 15))(
                           v227,
                           v115,
                           v112,
                           v111);
                  if ( !v116 )
                  {
                    v199 = 257;
                    v116 = sub_B51D30(v115, v112, v111, (__int64)&v196, 0, 0);
                    if ( (unsigned __int8)sub_920620(v116) )
                    {
                      v143 = v230;
                      if ( v229 )
                        sub_B99FD0(v116, 3u, v229);
                      sub_B45150(v116, v143);
                    }
                    (*(void (__fastcall **)(_QWORD *, __int64, char **, __int64, __int64))(*v228 + 16LL))(
                      v228,
                      v116,
                      &v184,
                      v224,
                      v225);
                    v144 = (__int64)v220;
                    v145 = 16LL * (unsigned int)v221;
                    v146 = &v220[v145];
                    if ( v220 != &v220[v145] )
                    {
                      do
                      {
                        v147 = *(_QWORD *)(v144 + 8);
                        v148 = *(_DWORD *)v144;
                        v144 += 16;
                        sub_B99FD0(v116, v148, v147);
                      }
                      while ( v146 != (const char *)v144 );
                    }
                  }
                }
                v178[2] = v116;
                sub_24869A0((__int64 *)&v220, v213, v214, v178, 3, (__int64)&v192, 0);
              }
              sub_B43D60((_QWORD *)v68);
              nullsub_61();
              v235 = &unk_49DA100;
              nullsub_63();
              if ( v220 != v222 )
                _libc_free((unsigned __int64)v220);
              goto LABEL_76;
            }
            v172 = v179;
            sub_B2BEC0(a3);
            if ( byte_4FE9928 || *sub_98ACB0((unsigned __int8 *)v172, 6u) != 60 )
            {
              v69 = v182;
              v162 = v180;
              if ( !v182 )
              {
                sub_2487DC0(&v200, v68, v172, v180);
                goto LABEL_76;
              }
              v70 = *(_DWORD *)(v181 + 32);
              v164 = v181;
              v160 = (char *)sub_AD64C0(v202, 0, 0);
              if ( v70 )
                break;
            }
          }
LABEL_76:
          ++v176;
          ++v158;
          if ( v156 == (__m128i *)v176 )
          {
            v156 = v216;
LABEL_167:
            if ( v156 != &v218 )
              _libc_free((unsigned __int64)v156);
            goto LABEL_151;
          }
        }
        v71 = v70;
        v167 = v68;
        v72 = 0;
        v161 = v71;
        while ( *(_BYTE *)v69 == 11 )
        {
          v73 = *(_QWORD *)(v69 + 32 * (v72 - (*(_DWORD *)(v69 + 4) & 0x7FFFFFF)));
          if ( *(_BYTE *)v73 != 17 )
            goto LABEL_118;
          v74 = *(_DWORD *)(v73 + 32);
          if ( v74 <= 0x40 )
          {
            if ( *(_QWORD *)(v73 + 24) )
            {
LABEL_118:
              v75 = v167;
              goto LABEL_87;
            }
          }
          else
          {
            v75 = v167;
            if ( v74 != (unsigned int)sub_C444A0(v73 + 24) )
              goto LABEL_87;
          }
LABEL_90:
          if ( v161 == ++v72 )
            goto LABEL_76;
        }
        v226 = (_QWORD *)sub_BD5C60(v167);
        v220 = v222;
        v227 = &v235;
        v231 = 512;
        v228 = v236;
        v221 = 0x200000000LL;
        LOWORD(v225) = 0;
        v235 = &unk_49DA100;
        v229 = 0;
        v230 = 0;
        v236[0] = &unk_49DA0B0;
        v232 = 7;
        v233 = 0;
        v234 = 0;
        v223 = 0;
        v224 = 0;
        sub_D5F1F0((__int64)&v220, v167);
        v195 = 257;
        v78 = sub_BCB2E0(v226);
        v79 = sub_ACD640(v78, v72, 0);
        v80 = (*((__int64 (__fastcall **)(void **, __int64, __int64))*v227 + 12))(v227, v69, v79);
        if ( !v80 )
        {
          v199 = 257;
          v127 = sub_BD2C40(72, 2u);
          v80 = (__int64)v127;
          if ( v127 )
            sub_B4DE80((__int64)v127, v69, v79, (__int64)&v196, 0, 0);
          (*(void (__fastcall **)(_QWORD *, __int64, _QWORD **, __int64, __int64))(*v228 + 16LL))(
            v228,
            v80,
            &v192,
            v224,
            v225);
          v128 = (__int64)v220;
          v129 = &v220[16 * (unsigned int)v221];
          if ( v220 != v129 )
          {
            do
            {
              v130 = *(_QWORD *)(v128 + 8);
              v131 = *(_DWORD *)v128;
              v128 += 16;
              sub_B99FD0(v80, v131, v130);
            }
            while ( v129 != (const char *)v128 );
          }
        }
        v81 = v159;
        LOWORD(v81) = 0;
        v159 = v81;
        v75 = sub_F38250(v80, (__int64 *)(v167 + 24), v81, 0, 0, 0, 0, 0);
        nullsub_61();
        v235 = &unk_49DA100;
        nullsub_63();
        if ( v220 != v222 )
          _libc_free((unsigned __int64)v220);
LABEL_87:
        v76 = (_QWORD *)sub_BD5C60(v75);
        v232 = 7;
        v226 = v76;
        v220 = v222;
        v227 = &v235;
        v221 = 0x200000000LL;
        v228 = v236;
        v231 = 512;
        LOWORD(v225) = 0;
        v229 = 0;
        v230 = 0;
        v235 = &unk_49DA100;
        v233 = 0;
        v234 = 0;
        v236[0] = &unk_49DA0B0;
        v223 = 0;
        v224 = 0;
        sub_D5F1F0((__int64)&v220, v75);
        v195 = 257;
        v188 = v160;
        v189 = sub_AD64C0(v202, v72, 0);
        v77 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, char **, __int64, _QWORD))*v227 + 8))(
                          v227,
                          v164,
                          v172,
                          &v188,
                          2,
                          0);
        if ( v77 )
        {
LABEL_88:
          sub_2487DC0(&v200, v75, (__int64)v77, v162);
          nullsub_61();
          v235 = &unk_49DA100;
          nullsub_63();
          if ( v220 != v222 )
            _libc_free((unsigned __int64)v220);
          goto LABEL_90;
        }
        v199 = 257;
        v77 = sub_BD2C40(88, 3u);
        if ( !v77 )
        {
LABEL_122:
          sub_B4DDE0((__int64)v77, 0);
          (*(void (__fastcall **)(_QWORD *, _QWORD *, _QWORD **, __int64, __int64))(*v228 + 16LL))(
            v228,
            v77,
            &v192,
            v224,
            v225);
          if ( v220 != &v220[16 * (unsigned int)v221] )
          {
            v155 = v72;
            v98 = (__int64)v220;
            v99 = &v220[16 * (unsigned int)v221];
            do
            {
              v100 = *(_QWORD *)(v98 + 8);
              v101 = *(_DWORD *)v98;
              v98 += 16;
              sub_B99FD0((__int64)v77, v101, v100);
            }
            while ( v99 != (const char *)v98 );
            v72 = v155;
          }
          goto LABEL_88;
        }
        v96 = *(_QWORD *)(v172 + 8);
        v97 = v157 & 0xE0000000 | 3;
        v157 = v97;
        if ( (unsigned int)*(unsigned __int8 *)(v96 + 8) - 17 <= 1 )
        {
LABEL_121:
          sub_B44260((__int64)v77, v96, 34, v97, 0, 0);
          v77[9] = v164;
          v77[10] = sub_B4DC50(v164, (__int64)&v188, 2);
          sub_B4D9A0((__int64)v77, v172, (__int64 *)&v188, 2, (__int64)&v196);
          goto LABEL_122;
        }
        v132 = *((_QWORD *)v188 + 1);
        v133 = *(unsigned __int8 *)(v132 + 8);
        if ( v133 != 17 )
        {
          if ( v133 == 18 )
          {
LABEL_161:
            v135 = 1;
LABEL_163:
            v136 = *(_DWORD *)(v132 + 32);
            BYTE4(v184) = v135;
            LODWORD(v184) = v136;
            v96 = sub_BCE1B0((__int64 *)v96, (__int64)v184);
            goto LABEL_121;
          }
          v132 = *(_QWORD *)(v189 + 8);
          v134 = *(unsigned __int8 *)(v132 + 8);
          if ( v134 != 17 )
          {
            if ( v134 != 18 )
              goto LABEL_121;
            goto LABEL_161;
          }
        }
        v135 = 0;
        goto LABEL_163;
      }
LABEL_210:
      BUG();
    }
    if ( v216 != &v218 )
      _libc_free((unsigned __int64)v216);
  }
  if ( v163 )
  {
LABEL_151:
    memset(a1, 0, 0x60u);
    *((_DWORD *)a1 + 4) = 2;
    a1[1] = a1 + 4;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = a1 + 10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
LABEL_7:
  a1[6] = 0;
  a1[1] = a1 + 4;
  a1[7] = a1 + 10;
  a1[2] = 0x100000002LL;
  a1[8] = 2;
  *((_DWORD *)a1 + 18) = 0;
  *((_BYTE *)a1 + 76) = 1;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  a1[4] = &qword_4F82400;
  *a1 = 1;
  return a1;
}
