// Function: sub_8F9C90
// Address: 0x8f9c90
//
__int64 __fastcall sub_8F9C90(int a1, __int64 p_s2)
{
  __m128i *v2; // r12
  __int64 v3; // r13
  _BYTE *v4; // rcx
  int v5; // eax
  _BYTE *v6; // r14
  __int64 v7; // rax
  unsigned int v8; // r13d
  int v9; // r12d
  char *v10; // rdi
  __int64 v11; // rcx
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rdx
  __m128i *v14; // rdi
  int v15; // ebx
  int v16; // r13d
  _BYTE *v17; // r15
  __int64 v18; // r12
  const char **v19; // r14
  _BYTE *v20; // r8
  bool v21; // cf
  bool v22; // zf
  size_t v23; // rax
  size_t v24; // rax
  __int64 v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r12
  char *v31; // rax
  char **v32; // r8
  __int64 v33; // r12
  __m128i *v34; // rbx
  __m128i *v35; // r12
  char v37; // al
  bool v38; // cf
  bool v39; // zf
  unsigned __int64 v40; // rcx
  char *v41; // rdi
  unsigned __int64 v42; // r11
  __m128i *v43; // r9
  __m128i *v44; // r8
  const char *v45; // r8
  bool v46; // cf
  bool v47; // zf
  __int64 v48; // rax
  __m128i *v49; // rdx
  __m128i si128; // xmm0
  int v51; // eax
  __m128i *v52; // r14
  size_t v53; // r12
  __m128i *v54; // r8
  __m128i *v55; // r8
  __m128i *v56; // r13
  __int64 v57; // r14
  __m128i *v58; // rax
  __m128i v59; // xmm0
  __int64 v60; // r14
  _WORD *v61; // rdx
  _BYTE *v62; // rdi
  _BYTE *v63; // rax
  char v64; // al
  bool v65; // cf
  bool v66; // zf
  char v67; // al
  bool v68; // cf
  bool v69; // zf
  char v70; // al
  bool v71; // cf
  bool v72; // zf
  char v73; // al
  bool v74; // cf
  bool v75; // zf
  char v76; // al
  bool v77; // cf
  bool v78; // zf
  char v79; // al
  bool v80; // cf
  bool v81; // zf
  int v82; // eax
  const char *v83; // r8
  bool v84; // cf
  bool v85; // zf
  __int64 v86; // r14
  const char *v87; // r9
  char v88; // al
  bool v89; // cf
  bool v90; // zf
  const char *v91; // rax
  size_t v92; // rax
  const char *v93; // r8
  size_t v94; // r14
  _QWORD *v95; // rdx
  __int64 v96; // rax
  __m128i *v97; // rdx
  __m128i v98; // xmm0
  __int64 v99; // rax
  _QWORD *v100; // rdi
  __int64 v101; // rax
  __m128i v102; // xmm0
  __int64 v103; // rax
  __m128i v104; // xmm0
  int v105; // eax
  __m128i *v106; // r8
  __int8 v107; // al
  unsigned __int64 v108; // rdx
  char v109; // al
  __int32 v110; // eax
  unsigned __int8 *v111; // r8
  __m128i *v112; // r11
  size_t v113; // rax
  size_t v114; // r10
  __m128i *v115; // r9
  __m128i *v116; // r10
  __int8 v117; // dl
  char v118; // al
  __int64 v119; // rax
  size_t v120; // r9
  __m128i *v121; // r10
  __m128i *v122; // r11
  char *v123; // rax
  _BYTE *v124; // rax
  __int64 v125; // r12
  const char *v126; // r13
  unsigned __int64 v127; // rax
  __int64 v128; // rdi
  char **v129; // rbx
  size_t v130; // rax
  __int64 v131; // r13
  __int64 v132; // r12
  __int64 v133; // rbx
  __int64 v134; // r14
  char **v135; // r15
  char *v136; // rax
  __int32 v137; // ebx
  __int64 v138; // rax
  __int64 v139; // rdi
  _BYTE *v140; // rax
  __int64 v141; // rax
  __m128i *v142; // rdx
  __m128i v143; // xmm0
  __m128i *v144; // rax
  __int64 v145; // rax
  __m128i *v146; // rdx
  __m128i v147; // xmm0
  int v148; // eax
  char *v149; // rax
  const char *v150; // r13
  size_t v151; // r12
  __m128i *v152; // r14
  __m128i *v153; // r14
  bool v154; // bl
  __int64 v155; // rax
  __int64 v156; // rax
  __m128i *v157; // r8
  unsigned __int64 v158; // rdx
  __int8 v159; // al
  char v160; // al
  __int64 v161; // rax
  __m128i *v162; // r11
  char *v163; // rax
  __m128i *v164; // rax
  __int64 v165; // rax
  __m128i *v166; // rax
  __m128i v167; // xmm0
  __int64 v168; // rdi
  __m128i *v169; // rax
  unsigned __int64 v170; // rdx
  __m128i v171; // xmm0
  __int64 v172; // rcx
  __int64 v173; // rdi
  __m128i *v174; // rax
  __m128i v175; // xmm0
  __int64 v176; // rax
  size_t v177; // r10
  __m128i *v178; // r11
  __int64 v179; // r9
  __m128i *v180; // rax
  __m128i v181; // xmm0
  __m128i *v182; // rax
  __int64 v183; // rax
  __m128i v184; // xmm0
  __int64 v185; // rax
  __int64 v186; // rax
  __int64 v187; // rax
  int v188; // ebx
  int v189; // eax
  int v190; // eax
  const char *v191; // r12
  int v192; // eax
  int v193; // eax
  int v194; // eax
  size_t v195; // rax
  const char *v196; // r8
  size_t v197; // r14
  _QWORD *v198; // rdi
  __int64 v199; // rax
  __int64 v200; // rax
  __int64 v201; // rax
  __int64 v202; // rax
  __int64 v203; // rax
  int v204; // eax
  int v205; // eax
  __m128i *v206; // [rsp-10h] [rbp-9B0h]
  __int64 v207; // [rsp+10h] [rbp-990h]
  __m128i *v208; // [rsp+10h] [rbp-990h]
  __m128i *v209; // [rsp+10h] [rbp-990h]
  __int64 v210; // [rsp+10h] [rbp-990h]
  __m128i *v211; // [rsp+10h] [rbp-990h]
  const char *s1; // [rsp+18h] [rbp-988h]
  void *s1d; // [rsp+18h] [rbp-988h]
  __m128i *s1e; // [rsp+18h] [rbp-988h]
  __m128i *s1a; // [rsp+18h] [rbp-988h]
  __m128i *s1b; // [rsp+18h] [rbp-988h]
  __m128i *s1g; // [rsp+18h] [rbp-988h]
  __m128i *s1c; // [rsp+18h] [rbp-988h]
  __m128i *s1h; // [rsp+18h] [rbp-988h]
  __m128i *s1i; // [rsp+18h] [rbp-988h]
  __m128i *s1f; // [rsp+18h] [rbp-988h]
  size_t v222; // [rsp+20h] [rbp-980h]
  size_t v223; // [rsp+20h] [rbp-980h]
  size_t v224; // [rsp+20h] [rbp-980h]
  size_t v225; // [rsp+20h] [rbp-980h]
  size_t v226; // [rsp+20h] [rbp-980h]
  size_t v227; // [rsp+20h] [rbp-980h]
  size_t v228; // [rsp+20h] [rbp-980h]
  size_t v229; // [rsp+20h] [rbp-980h]
  size_t v230; // [rsp+20h] [rbp-980h]
  size_t v231; // [rsp+20h] [rbp-980h]
  size_t v232; // [rsp+20h] [rbp-980h]
  size_t v233; // [rsp+20h] [rbp-980h]
  size_t v234; // [rsp+20h] [rbp-980h]
  size_t v235; // [rsp+20h] [rbp-980h]
  size_t v236; // [rsp+20h] [rbp-980h]
  size_t v237; // [rsp+20h] [rbp-980h]
  size_t v238; // [rsp+20h] [rbp-980h]
  size_t v239; // [rsp+20h] [rbp-980h]
  size_t v240; // [rsp+20h] [rbp-980h]
  size_t v241; // [rsp+20h] [rbp-980h]
  unsigned __int32 v242; // [rsp+28h] [rbp-978h]
  char v243; // [rsp+37h] [rbp-969h]
  unsigned __int64 v244; // [rsp+38h] [rbp-968h]
  const char *v245; // [rsp+38h] [rbp-968h]
  size_t v246; // [rsp+38h] [rbp-968h]
  const char *v247; // [rsp+38h] [rbp-968h]
  size_t v248; // [rsp+38h] [rbp-968h]
  size_t v249; // [rsp+38h] [rbp-968h]
  unsigned __int8 v250; // [rsp+40h] [rbp-960h]
  const char *v251; // [rsp+40h] [rbp-960h]
  char *s; // [rsp+48h] [rbp-958h]
  int v253; // [rsp+50h] [rbp-950h]
  __int64 v254; // [rsp+50h] [rbp-950h]
  __int64 v255; // [rsp+50h] [rbp-950h]
  _BYTE *v256; // [rsp+60h] [rbp-940h]
  char *v257; // [rsp+78h] [rbp-928h]
  unsigned __int8 v258; // [rsp+80h] [rbp-920h]
  char v259; // [rsp+81h] [rbp-91Fh]
  unsigned __int8 v260; // [rsp+81h] [rbp-91Fh]
  unsigned __int8 v261; // [rsp+82h] [rbp-91Eh]
  unsigned __int8 v262; // [rsp+83h] [rbp-91Dh]
  int v263; // [rsp+84h] [rbp-91Ch]
  unsigned int v264; // [rsp+94h] [rbp-90Ch] BYREF
  __int64 v265; // [rsp+98h] [rbp-908h] BYREF
  __m128i *v266; // [rsp+A0h] [rbp-900h] BYREF
  __m128i *v267; // [rsp+A8h] [rbp-8F8h]
  __int64 v268; // [rsp+B0h] [rbp-8F0h]
  __m128i dest; // [rsp+C0h] [rbp-8E0h]
  _QWORD v270[2]; // [rsp+D0h] [rbp-8D0h] BYREF
  __m128i v271; // [rsp+E0h] [rbp-8C0h] BYREF
  _QWORD v272[2]; // [rsp+F0h] [rbp-8B0h] BYREF
  __m128i s2; // [rsp+100h] [rbp-8A0h] BYREF
  _QWORD v274[2]; // [rsp+110h] [rbp-890h] BYREF
  _QWORD *v275; // [rsp+120h] [rbp-880h] BYREF
  __int64 v276; // [rsp+128h] [rbp-878h]
  _QWORD v277[2]; // [rsp+130h] [rbp-870h] BYREF
  __int16 v278; // [rsp+140h] [rbp-860h]
  bool v279; // [rsp+142h] [rbp-85Eh]
  int v280; // [rsp+144h] [rbp-85Ch]
  char **v281; // [rsp+148h] [rbp-858h]
  _BYTE *v282; // [rsp+160h] [rbp-840h] BYREF
  __int64 v283; // [rsp+168h] [rbp-838h]
  _BYTE v284[2096]; // [rsp+170h] [rbp-830h] BYREF

  v2 = (__m128i *)p_s2;
  v3 = 8LL * a1;
  v282 = v284;
  v283 = 0x10000000000LL;
  if ( (unsigned __int64)v3 > 0x800 )
  {
    p_s2 = (__int64)v284;
    sub_16CD150(&v282, v284, a1, 8);
    v6 = v282;
    v5 = v283;
    v4 = &v282[8 * (unsigned int)v283];
  }
  else
  {
    v4 = v284;
    v5 = 0;
    v6 = v284;
  }
  if ( v3 > 0 )
  {
    v7 = 0;
    do
    {
      *(_QWORD *)&v4[8 * v7] = v2->m128i_i64[v7];
      ++v7;
    }
    while ( a1 - v7 > 0 );
    v6 = v282;
    v5 = v283;
  }
  v8 = 1;
  LODWORD(v283) = v5 + a1;
  v9 = v5 + a1;
  if ( v5 + a1 <= 1 )
    goto LABEL_71;
  dest.m128i_i64[1] = 0;
  dest.m128i_i64[0] = (__int64)v270;
  LOBYTE(v270[0]) = 0;
  v266 = 0;
  v267 = 0;
  v268 = 0;
  v10 = getenv("NVVMCCWIZ");
  if ( v10 && (unsigned int)strtol(v10, 0, 10) == 553282 )
    byte_4F6D280 = 1;
  p_s2 = *(_QWORD *)v6;
  sub_16C5290(&s2, *(_QWORD *)v6, 0);
  v11 = (__int64)v274;
  v12 = (_BYTE *)dest.m128i_i64[0];
  if ( (_QWORD *)s2.m128i_i64[0] == v274 )
  {
    v13 = s2.m128i_u64[1];
    if ( s2.m128i_i64[1] )
    {
      if ( s2.m128i_i64[1] == 1 )
      {
        *(_BYTE *)dest.m128i_i64[0] = v274[0];
      }
      else
      {
        p_s2 = (__int64)v274;
        memcpy((void *)dest.m128i_i64[0], v274, s2.m128i_u64[1]);
      }
      v13 = s2.m128i_u64[1];
      v12 = (_BYTE *)dest.m128i_i64[0];
    }
    dest.m128i_i64[1] = v13;
    v12[v13] = 0;
    v12 = (_BYTE *)s2.m128i_i64[0];
    goto LABEL_15;
  }
  v13 = v274[0];
  v11 = s2.m128i_i64[1];
  if ( (_QWORD *)dest.m128i_i64[0] == v270 )
  {
    dest = s2;
    v270[0] = v274[0];
    goto LABEL_182;
  }
  p_s2 = v270[0];
  dest = s2;
  v270[0] = v274[0];
  if ( !v12 )
  {
LABEL_182:
    s2.m128i_i64[0] = (__int64)v274;
    v12 = v274;
    goto LABEL_15;
  }
  s2.m128i_i64[0] = (__int64)v12;
  v274[0] = p_s2;
LABEL_15:
  s2.m128i_i64[1] = 0;
  *v12 = 0;
  v14 = (__m128i *)s2.m128i_i64[0];
  if ( (_QWORD *)s2.m128i_i64[0] != v274 )
  {
    p_s2 = v274[0] + 1LL;
    j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
  }
  v250 = 0;
  v15 = 1;
  v16 = v9;
  v242 = 0;
  v253 = 2;
  v256 = 0;
  v261 = 0;
  v262 = 0;
  v259 = 0;
  s = 0;
  v257 = 0;
  v263 = 0;
  v243 = 0;
  v258 = 0;
  v17 = v6;
  do
  {
    v18 = 8LL * v15;
    v19 = (const char **)&v17[v18];
    v20 = *(_BYTE **)&v17[v18];
    v21 = *v20 < 0x2Du;
    v22 = *v20 == 45;
    if ( *v20 == 45 )
    {
      v21 = v20[1] < 0x6Fu;
      v22 = v20[1] == 111;
      if ( v20[1] == 111 )
      {
        v21 = 0;
        v22 = v20[2] == 0;
        if ( !v20[2] )
        {
          if ( v16 - 1 <= v15 )
          {
            v48 = sub_16E8CB0(v14, p_s2, v13);
            v49 = *(__m128i **)(v48 + 24);
            if ( *(_QWORD *)(v48 + 16) - (_QWORD)v49 <= 0x13u )
            {
              p_s2 = (__int64)"Missing output file\n";
              sub_16E7EE0(v48, "Missing output file\n", 20);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_3C23AE0);
              v49[1].m128i_i32[0] = 174419049;
              *v49 = si128;
              *(_QWORD *)(v48 + 24) += 20LL;
            }
LABEL_60:
            v8 = 1;
            goto LABEL_61;
          }
          ++v15;
          v257 = *(char **)&v17[v18 + 8];
          goto LABEL_23;
        }
      }
    }
    v11 = 16;
    p_s2 = *(_QWORD *)&v17[8 * v15];
    v14 = (__m128i *)"-nvvmir-library";
    do
    {
      if ( !v11 )
        break;
      v21 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
      v22 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
      v14 = (__m128i *)((char *)v14 + 1);
      --v11;
    }
    while ( v22 );
    v37 = (!v21 && !v22) - v21;
    v38 = 0;
    v39 = v37 == 0;
    if ( !v37 )
    {
      if ( v16 - 1 <= v15 )
      {
        v96 = sub_16E8CB0(v14, p_s2, v13);
        v97 = *(__m128i **)(v96 + 24);
        if ( *(_QWORD *)(v96 + 16) - (_QWORD)v97 <= 0x1Cu )
        {
          p_s2 = (__int64)"Missing NVVM IR library file\n";
          sub_16E7EE0(v96, "Missing NVVM IR library file\n", 29);
        }
        else
        {
          v98 = _mm_load_si128((const __m128i *)&xmmword_3C23AF0);
          qmemcpy(&v97[1], "library file\n", 13);
          *v97 = v98;
          *(_QWORD *)(v96 + 24) += 29LL;
        }
        goto LABEL_60;
      }
      ++v15;
      v256 = *(_BYTE **)&v17[v18 + 8];
      goto LABEL_23;
    }
    v11 = 16;
    p_s2 = *(_QWORD *)&v17[8 * v15];
    v14 = (__m128i *)"-nvvmir-library=";
    do
    {
      if ( !v11 )
        break;
      v38 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
      v39 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
      v14 = (__m128i *)((char *)v14 + 1);
      --v11;
    }
    while ( v39 );
    if ( (!v38 && !v39) == v38 )
    {
      v256 = v20 + 16;
      goto LABEL_23;
    }
    v40 = 24;
    v41 = "--discard_value_names=1";
    if ( v263 == 1 )
      goto LABEL_90;
    v40 = 23;
    v41 = "-discard-value-names=1";
    if ( v263 == 2 )
      goto LABEL_90;
    v40 = 27;
    v41 = "-lnk-discard-value-names=1";
    if ( v263 == 3 )
      goto LABEL_90;
    v40 = 27;
    v41 = "-opt-discard-value-names=1";
    if ( v263 == 4 )
      goto LABEL_90;
    if ( !v263 )
    {
      v40 = 21;
      v41 = "-discard-value-names";
LABEL_90:
      if ( !memcmp(*(const void **)&v17[8 * v15], v41, v40) )
      {
        v251 = *(const char **)&v17[8 * v15];
        s2.m128i_i64[0] = (__int64)v274;
        v92 = strlen(v251);
        v93 = v251;
        v271.m128i_i64[0] = v92;
        v94 = v92;
        if ( v92 > 0xF )
        {
          v99 = sub_22409D0(&s2, &v271, 0);
          v93 = v251;
          s2.m128i_i64[0] = v99;
          v100 = (_QWORD *)v99;
          v274[0] = v271.m128i_i64[0];
        }
        else
        {
          if ( v92 == 1 )
          {
            LOBYTE(v274[0]) = *v251;
            v95 = v274;
LABEL_175:
            s2.m128i_i64[1] = v92;
            p_s2 = (__int64)&s2;
            *((_BYTE *)v95 + v92) = 0;
            sub_8F9C20(&v266, &s2);
            v14 = (__m128i *)s2.m128i_i64[0];
            if ( (_QWORD *)s2.m128i_i64[0] != v274 )
            {
              p_s2 = v274[0] + 1LL;
              j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
            }
            v250 = 1;
            goto LABEL_23;
          }
          if ( !v92 )
          {
            v95 = v274;
            goto LABEL_175;
          }
          v100 = v274;
        }
        memcpy(v100, v93, v94);
        v92 = v271.m128i_i64[0];
        v95 = (_QWORD *)s2.m128i_i64[0];
        goto LABEL_175;
      }
    }
    s1 = *(const char **)&v17[8 * v15];
    v14 = &s2;
    v244 = strlen(s1);
    p_s2 = (__int64)&byte_3C23AD6[-14];
    s2.m128i_i64[0] = (__int64)v274;
    sub_8F98A0(s2.m128i_i64, (char)&byte_3C23AD6[-14], &byte_3C23AD6[-14], byte_3C23AD6);
    v13 = s2.m128i_u64[1];
    v42 = v244;
    v43 = (__m128i *)s2.m128i_i64[0];
    v44 = (__m128i *)s1;
    if ( v244 < s2.m128i_i64[1] )
      goto LABEL_92;
    if ( s2.m128i_i64[1] )
    {
      p_s2 = s2.m128i_i64[0];
      v14 = (__m128i *)s1;
      v207 = s2.m128i_i64[1];
      s1e = (__m128i *)s2.m128i_i64[0];
      v223 = (size_t)v44;
      v51 = memcmp(v14, (const void *)s2.m128i_i64[0], s2.m128i_u64[1]);
      v44 = (__m128i *)v223;
      v43 = s1e;
      v13 = v207;
      v42 = v244;
      if ( v51 )
      {
LABEL_92:
        if ( v43 != (__m128i *)v274 )
        {
          v14 = v43;
          s1d = (void *)v42;
          v222 = (size_t)v44;
          p_s2 = v274[0] + 1LL;
          j_j___libc_free_0(v43, v274[0] + 1LL);
          v44 = (__m128i *)v222;
          v42 = (unsigned __int64)s1d;
        }
        if ( v42 <= 8 )
        {
          if ( v42 > 4 && v44->m128i_i32[0] == 1668440365 && v44->m128i_i8[4] == 104 )
            goto LABEL_229;
LABEL_96:
          if ( v244 == 4 )
          {
            v11 = v258;
            if ( v44->m128i_i32[0] == 1668705837 )
              v11 = 1;
            v258 = v11;
          }
LABEL_97:
          v45 = *v19;
          v46 = **v19 < 0x2Du;
          v47 = **v19 == 45;
          if ( **v19 == 45 )
          {
            v46 = v45[1] < 0x76u;
            v47 = v45[1] == 118;
            if ( v45[1] == 118 )
            {
              v46 = 0;
              v47 = v45[2] == 0;
              if ( !v45[2] )
              {
                v259 = byte_4F6D280;
                goto LABEL_23;
              }
            }
          }
          v11 = 8;
          p_s2 = (__int64)*v19;
          v14 = (__m128i *)"-dryrun";
          do
          {
            if ( !v11 )
              break;
            v46 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v47 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v47 );
          v64 = (!v46 && !v47) - v46;
          v65 = 0;
          v66 = v64 == 0;
          if ( !v64 )
          {
            v261 = 1;
            v259 = byte_4F6D280;
            goto LABEL_23;
          }
          v11 = 6;
          p_s2 = (__int64)*v19;
          v14 = (__m128i *)"-keep";
          do
          {
            if ( !v11 )
              break;
            v65 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v66 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v66 );
          v67 = (!v65 && !v66) - v65;
          v68 = 0;
          v69 = v67 == 0;
          if ( !v67 )
          {
            v262 = byte_4F6D280;
            goto LABEL_23;
          }
          v11 = 8;
          v14 = (__m128i *)"-lgenfe";
          p_s2 = (__int64)*v19;
          do
          {
            if ( !v11 )
              break;
            v68 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v69 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v69 );
          v70 = (!v68 && !v69) - v68;
          v71 = 0;
          v72 = v70 == 0;
          if ( !v70 )
          {
            v263 = 1;
            goto LABEL_23;
          }
          v11 = 9;
          v14 = (__m128i *)"-libnvvm";
          p_s2 = (__int64)*v19;
          do
          {
            if ( !v11 )
              break;
            v71 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v72 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v72 );
          v73 = (!v71 && !v72) - v71;
          v74 = 0;
          v75 = v73 == 0;
          if ( !v73 )
          {
            v263 = 2;
            goto LABEL_23;
          }
          v11 = 5;
          v14 = (__m128i *)"-lnk";
          p_s2 = (__int64)*v19;
          do
          {
            if ( !v11 )
              break;
            v74 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v75 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v75 );
          v76 = (!v74 && !v75) - v74;
          v77 = 0;
          v78 = v76 == 0;
          if ( !v76 )
          {
            v263 = 3;
            v262 = byte_4F6D280;
            goto LABEL_23;
          }
          v11 = 5;
          v14 = (__m128i *)"-opt";
          p_s2 = (__int64)*v19;
          do
          {
            if ( !v11 )
              break;
            v77 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v78 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v78 );
          v79 = (!v77 && !v78) - v77;
          v80 = 0;
          v81 = v79 == 0;
          if ( !v79 )
          {
            v263 = 4;
            v262 = byte_4F6D280;
            goto LABEL_23;
          }
          v11 = 5;
          v14 = (__m128i *)"-llc";
          p_s2 = (__int64)*v19;
          do
          {
            if ( !v11 )
              break;
            v80 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
            v81 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
            v14 = (__m128i *)((char *)v14 + 1);
            --v11;
          }
          while ( v81 );
          if ( (!v80 && !v81) == v80 )
          {
            v263 = 6;
            goto LABEL_23;
          }
          if ( !strcmp(*v19, "-irversion") )
          {
            v8 = 0;
            sub_12BC0E0(&v271, &s2, 0, 0);
            v137 = s2.m128i_i32[0] + 100 * v271.m128i_i32[0];
            v138 = ((__int64 (*)(void))sub_16E8C20)();
            p_s2 = v137;
            v139 = sub_16E7AB0(v138, v137);
            v140 = *(_BYTE **)(v139 + 24);
            if ( *(_BYTE **)(v139 + 16) == v140 )
            {
              p_s2 = (__int64)"\n";
              sub_16E7EE0(v139, "\n", 1);
            }
            else
            {
              *v140 = 10;
              ++*(_QWORD *)(v139 + 24);
            }
            goto LABEL_61;
          }
          v245 = *v19;
          v82 = strlen(*v19);
          v83 = v245;
          v84 = (unsigned int)v82 < 3;
          v85 = v82 == 3;
          if ( v82 > 3 )
          {
            v86 = v82;
            v11 = 4;
            v14 = (__m128i *)".bc";
            v87 = &v245[v82 - 3];
            p_s2 = (__int64)v87;
            do
            {
              if ( !v11 )
                break;
              v84 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
              v85 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
              v14 = (__m128i *)((char *)v14 + 1);
              --v11;
            }
            while ( v85 );
            v88 = (!v84 && !v85) - v84;
            v89 = 0;
            v90 = v88 == 0;
            if ( !v88 )
              goto LABEL_171;
            v11 = 4;
            v14 = (__m128i *)".ci";
            p_s2 = (__int64)v87;
            do
            {
              if ( !v11 )
                break;
              v89 = *(_BYTE *)p_s2 < v14->m128i_i8[0];
              v90 = *(_BYTE *)p_s2++ == v14->m128i_i8[0];
              v14 = (__m128i *)((char *)v14 + 1);
              --v11;
            }
            while ( v90 );
            if ( (!v89 && !v90) == v89 )
              goto LABEL_171;
            v91 = &v245[v86 - 2];
            if ( *v91 == 46 && v91[1] == 105 && !v245[v86] )
              goto LABEL_171;
            p_s2 = (__int64)".ii";
            v14 = (__m128i *)v87;
            v189 = strcmp(v87, ".ii");
            v83 = v245;
            if ( !v189 )
              goto LABEL_171;
            v190 = strcmp(&v245[v86 - 4], ".cup");
            v83 = v245;
            if ( !v190 )
            {
              v191 = *(const char **)&v17[v18 - 8];
              v192 = strcmp(v191, "--orig_src_path_name");
              v83 = v245;
              if ( v192 )
              {
                p_s2 = (__int64)"--orig_src_file_name";
                v14 = (__m128i *)v191;
                v193 = strcmp(v191, "--orig_src_file_name");
                v83 = v245;
                if ( v193 )
                  goto LABEL_171;
              }
            }
            v14 = (__m128i *)&v83[v86 - 8];
            p_s2 = (__int64)".optixir";
            v246 = (size_t)v83;
            v194 = strcmp(v14->m128i_i8, ".optixir");
            v83 = (const char *)v246;
            if ( !v194 )
            {
LABEL_171:
              s = (char *)v83;
              goto LABEL_23;
            }
          }
          v247 = v83;
          s2.m128i_i64[0] = (__int64)v274;
          v195 = strlen(v83);
          v196 = v247;
          v271.m128i_i64[0] = v195;
          v197 = v195;
          if ( v195 > 0xF )
          {
            v199 = sub_22409D0(&s2, &v271, 0);
            v196 = v247;
            s2.m128i_i64[0] = v199;
            v198 = (_QWORD *)v199;
            v274[0] = v271.m128i_i64[0];
LABEL_400:
            memcpy(v198, v196, v197);
          }
          else
          {
            if ( v195 == 1 )
            {
              LOBYTE(v274[0]) = *v247;
              goto LABEL_390;
            }
            if ( v195 )
            {
              v198 = v274;
              goto LABEL_400;
            }
          }
LABEL_390:
          p_s2 = (__int64)&s2;
          s2.m128i_i64[1] = v271.m128i_i64[0];
          *(_BYTE *)(s2.m128i_i64[0] + v271.m128i_i64[0]) = 0;
          sub_8F9C20(&v266, &s2);
          v14 = (__m128i *)s2.m128i_i64[0];
          if ( (_QWORD *)s2.m128i_i64[0] != v274 )
          {
            p_s2 = v274[0] + 1LL;
            j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
          }
          goto LABEL_23;
        }
        if ( v44->m128i_i64[0] == 0x6D733D7570636D2DLL && v44->m128i_i8[8] == 95 )
        {
          v106 = (__m128i *)((char *)&v44->m128i_u64[1] + 1);
          v244 = v42 - 9;
          goto LABEL_219;
        }
        if ( v42 > 0xC )
        {
          if ( v44->m128i_i64[0] == 0x6372612D74706F2DLL && v44->m128i_i32[2] == 1836268904 && v44->m128i_i8[12] == 95 )
          {
            v106 = (__m128i *)((char *)&v44->m128i_u64[1] + 5);
            v244 = v42 - 13;
            goto LABEL_219;
          }
          if ( v42 != 13
            && v44->m128i_i64[0] == 0x6F633D686372612DLL
            && v44->m128i_i32[2] == 1953853549
            && v44->m128i_i16[6] == 24421 )
          {
            v106 = (__m128i *)((char *)&v44->m128i_u64[1] + 6);
            v244 = v42 - 14;
LABEL_219:
            if ( v244 )
            {
              v107 = v106->m128i_i8[v244 - 1];
              v108 = v244 - 1;
              if ( v107 == 97 )
              {
                --v244;
              }
              else
              {
                if ( v107 != 102 )
                  v108 = v244;
                v244 = v108;
              }
            }
            p_s2 = v244;
            v14 = v106;
            v224 = (size_t)v106;
            v109 = sub_16D2B80(v106, v244, 10, &s2);
            v44 = (__m128i *)v224;
            if ( !v109 )
            {
              v110 = s2.m128i_i32[0];
              v13 = s2.m128i_u32[0];
              if ( s2.m128i_i64[0] == s2.m128i_u32[0] )
              {
LABEL_224:
                v242 = v110;
LABEL_225:
                if ( v244 != 15 )
                  goto LABEL_96;
LABEL_197:
                if ( v44->m128i_i64[0] == 0x6F2D74696D652D2DLL
                  && v44->m128i_i32[2] == 2020177008
                  && v44->m128i_i16[6] == 26925
                  && v44->m128i_i8[14] == 114 )
                {
                  s2.m128i_i64[0] = (__int64)v274;
                  strcpy((char *)v274, "--emit-optix-ir");
                  s2.m128i_i64[1] = 15;
                  sub_8F9C20(&v266, &s2);
                  if ( (_QWORD *)s2.m128i_i64[0] != v274 )
                    j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
                  v271.m128i_i64[0] = 18;
                  s2.m128i_i64[0] = (__int64)v274;
                  v101 = sub_22409D0(&s2, &v271, 0);
                  v102 = _mm_load_si128((const __m128i *)&xmmword_3C23B30);
                  s2.m128i_i64[0] = v101;
                  v274[0] = v271.m128i_i64[0];
                  *(_WORD *)(v101 + 16) = 14386;
                  *(__m128i *)v101 = v102;
                  s2.m128i_i64[1] = v271.m128i_i64[0];
                  *(_BYTE *)(s2.m128i_i64[0] + v271.m128i_i64[0]) = 0;
                  sub_8F9C20(&v266, &s2);
                  if ( (_QWORD *)s2.m128i_i64[0] != v274 )
                    j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
                  v271.m128i_i64[0] = 20;
                  s2.m128i_i64[0] = (__int64)v274;
                  v103 = sub_22409D0(&s2, &v271, 0);
                  p_s2 = (__int64)&s2;
                  v104 = _mm_load_si128((const __m128i *)&xmmword_3C23B40);
                  s2.m128i_i64[0] = v103;
                  v274[0] = v271.m128i_i64[0];
                  *(_DWORD *)(v103 + 16) = 942813556;
                  *(__m128i *)v103 = v104;
                  s2.m128i_i64[1] = v271.m128i_i64[0];
                  *(_BYTE *)(s2.m128i_i64[0] + v271.m128i_i64[0]) = 0;
                  sub_8F9C20(&v266, &s2);
                  v14 = (__m128i *)s2.m128i_i64[0];
                  if ( (_QWORD *)s2.m128i_i64[0] != v274 )
                  {
                    p_s2 = v274[0] + 1LL;
                    j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
                  }
                  v243 = 1;
                  v258 = 1;
                  goto LABEL_23;
                }
                goto LABEL_97;
              }
            }
            v141 = ((__int64 (*)(void))sub_16E8C20)();
            v44 = (__m128i *)v224;
            v142 = *(__m128i **)(v141 + 24);
            v122 = (__m128i *)v141;
            if ( *(_QWORD *)(v141 + 16) - (_QWORD)v142 <= 0x19u )
            {
              p_s2 = (__int64)"Unparseable architecture: ";
              v156 = sub_16E7EE0(v141, "Unparseable architecture: ", 26);
              v44 = (__m128i *)v224;
              v14 = *(__m128i **)(v156 + 24);
              v122 = (__m128i *)v156;
            }
            else
            {
              v143 = _mm_load_si128((const __m128i *)&xmmword_3C23B10);
              v11 = 8250;
              qmemcpy(&v142[1], "itecture: ", 10);
              *v142 = v143;
              v14 = (__m128i *)(*(_QWORD *)(v141 + 24) + 26LL);
              *(_QWORD *)(v141 + 24) = v14;
            }
            v144 = (__m128i *)v122[1].m128i_i64[0];
            v13 = (char *)v144 - (char *)v14;
            if ( (char *)v144 - (char *)v14 < v244 )
            {
              p_s2 = (__int64)v44;
              v228 = (size_t)v44;
              v155 = sub_16E7EE0(v122, v44, v244);
              v44 = (__m128i *)v228;
              v122 = (__m128i *)v155;
              v144 = *(__m128i **)(v155 + 16);
              v14 = (__m128i *)v122[1].m128i_i64[1];
            }
            else if ( v244 )
            {
              p_s2 = (__int64)v44;
              s1g = v122;
              v231 = (size_t)v44;
              memcpy(v14, v44, v244);
              v122 = s1g;
              v44 = (__m128i *)v231;
              v14 = (__m128i *)(s1g[1].m128i_i64[1] + v244);
              v144 = (__m128i *)s1g[1].m128i_i64[0];
              s1g[1].m128i_i64[1] = (__int64)v14;
            }
            if ( v144 != v14 )
            {
              v14->m128i_i8[0] = 10;
              ++v122[1].m128i_i64[1];
              goto LABEL_225;
            }
LABEL_308:
            p_s2 = (__int64)"\n";
            v14 = v122;
            v227 = (size_t)v44;
            sub_16E7EE0(v122, "\n", 1);
            v44 = (__m128i *)v227;
            goto LABEL_225;
          }
          if ( v44->m128i_i64[0] == 0x415F414455435F5FLL && v44->m128i_i32[2] == 1028146002 )
          {
            v157 = (__m128i *)((char *)&v44->m128i_u64[1] + 4);
            v158 = v42 - 12;
            v244 = v42 - 13;
            v159 = v157->m128i_i8[v42 - 13];
            if ( v159 != 97 )
            {
              if ( v159 == 102 )
                v158 = v42 - 13;
              v244 = v158;
            }
            goto LABEL_320;
          }
          goto LABEL_194;
        }
        v241 = v42;
        if ( v42 != 12 )
        {
          v14 = v44;
          p_s2 = (__int64)"-arch";
          v248 = (size_t)v44;
          v204 = memcmp(v44, "-arch", 5u);
          v44 = (__m128i *)v248;
          v42 = v241;
          if ( !v204 )
            goto LABEL_229;
          goto LABEL_195;
        }
        v14 = v44;
        p_s2 = (__int64)"__CUDA_ARCH=";
        v249 = (size_t)v44;
        v205 = memcmp(v44, "__CUDA_ARCH=", 0xCu);
        v44 = (__m128i *)v249;
        v42 = 12;
        if ( !v205 )
        {
          v244 = 0;
          v157 = (__m128i *)((char *)&v44->m128i_u64[1] + 4);
LABEL_320:
          p_s2 = v244;
          v14 = v157;
          v229 = (size_t)v157;
          v160 = sub_16D2B80(v157, v244, 10, &s2);
          v44 = (__m128i *)v229;
          if ( v160 || s2.m128i_i64[0] != s2.m128i_u32[0] )
          {
            v161 = ((__int64 (*)(void))sub_16E8C20)();
            v44 = (__m128i *)v229;
            v162 = (__m128i *)v161;
            v163 = *(char **)(v161 + 24);
            if ( v162[1].m128i_i64[0] - (__int64)v163 <= 0x19uLL )
            {
              p_s2 = (__int64)"Unparseable architecture: ";
              v201 = sub_16E7EE0(v162, "Unparseable architecture: ", 26);
              v44 = (__m128i *)v229;
              v162 = (__m128i *)v201;
            }
            else
            {
              qmemcpy(v163, "Unparseable architecture: ", 0x1Au);
              p_s2 = (__int64)"";
              v11 = 0;
              v162[1].m128i_i64[1] += 26;
            }
            v164 = (__m128i *)v162[1].m128i_i64[0];
            v14 = (__m128i *)v162[1].m128i_i64[1];
            if ( (char *)v164 - (char *)v14 < v244 )
            {
              p_s2 = (__int64)v44;
              v237 = (size_t)v44;
              v200 = sub_16E7EE0(v162, v44, v244);
              v44 = (__m128i *)v237;
              v162 = (__m128i *)v200;
              v164 = *(__m128i **)(v200 + 16);
              v14 = (__m128i *)v162[1].m128i_i64[1];
            }
            else if ( v244 )
            {
              p_s2 = (__int64)v44;
              s1h = v162;
              v236 = (size_t)v44;
              memcpy(v14, v44, v244);
              v162 = s1h;
              v44 = (__m128i *)v236;
              v14 = (__m128i *)(s1h[1].m128i_i64[1] + v244);
              v164 = (__m128i *)s1h[1].m128i_i64[0];
              s1h[1].m128i_i64[1] = (__int64)v14;
            }
            if ( v14 == v164 )
            {
              p_s2 = (__int64)"\n";
              v14 = v162;
              v235 = (size_t)v44;
              sub_16E7EE0(v162, "\n", 1);
              v44 = (__m128i *)v235;
            }
            else
            {
              v14->m128i_i8[0] = 10;
              ++v162[1].m128i_i64[1];
            }
          }
          else
          {
            v242 = s2.m128i_i32[0];
          }
          v13 = 3435973837LL;
          v242 /= 0xAu;
          goto LABEL_225;
        }
LABEL_194:
        if ( v44->m128i_i32[0] == 1668440365 && v44->m128i_i8[4] == 104 )
        {
LABEL_229:
          v111 = &v44->m128i_u8[5];
          v244 = v42 - 5;
        }
        else
        {
LABEL_195:
          if ( v44->m128i_i64[0] != 0x6372615F766E2D2DLL || v44->m128i_i8[8] != 104 )
          {
            if ( v42 != 15 )
              goto LABEL_97;
            goto LABEL_197;
          }
          v111 = &v44->m128i_u8[9];
          v244 = v42 - 9;
        }
        if ( v16 <= v15 + 1 )
        {
          v230 = (size_t)v111;
          v165 = ((__int64 (*)(void))sub_16E8C20)();
          v44 = (__m128i *)v230;
          v14 = (__m128i *)v165;
          v166 = *(__m128i **)(v165 + 24);
          v13 = v14[1].m128i_i64[0] - (_QWORD)v166;
          if ( v13 <= 0x2A )
          {
            p_s2 = (__int64)"Unparseable architecture: missing argument\n";
            sub_16E7EE0(v14, "Unparseable architecture: missing argument\n", 43);
            v44 = (__m128i *)v230;
          }
          else
          {
            v167 = _mm_load_si128((const __m128i *)&xmmword_3C23B10);
            v11 = 0x656D756772612067LL;
            qmemcpy(&v166[2], "g argument\n", 11);
            *v166 = v167;
            v166[1] = _mm_load_si128((const __m128i *)&xmmword_3C23B20);
            v14[1].m128i_i64[1] += 43;
          }
          goto LABEL_225;
        }
        v112 = *(__m128i **)&v17[v18 + 8];
        if ( v112 )
        {
          s1a = (__m128i *)v111;
          v225 = *(_QWORD *)&v17[v18 + 8];
          v113 = strlen((const char *)v225);
          v112 = (__m128i *)v225;
          v111 = (unsigned __int8 *)s1a;
          v114 = v113;
          if ( v113 <= 7 )
          {
            if ( v113 > 2 )
              goto LABEL_350;
          }
          else
          {
            if ( *(_QWORD *)v225 == 0x5F657475706D6F63LL )
            {
              v115 = (__m128i *)(v113 - 8);
              v116 = (__m128i *)(v225 + 8);
              goto LABEL_235;
            }
LABEL_350:
            if ( *(_WORD *)v225 == 28019 && *(_BYTE *)(v225 + 2) == 95 )
            {
              v115 = (__m128i *)(v113 - 3);
              v116 = (__m128i *)(v225 + 3);
LABEL_235:
              if ( v115 )
              {
                v117 = v115->m128i_i8[(_QWORD)v116 - 1];
                if ( v117 == 97 )
                {
                  v115 = (__m128i *)((char *)v115 - 1);
                }
                else if ( v117 == 102 )
                {
                  v115 = (__m128i *)((char *)v115 - 1);
                }
              }
              v14 = v116;
              p_s2 = (__int64)v115;
              v208 = s1a;
              s1b = v116;
              v226 = (size_t)v115;
              v118 = sub_16D2B80(v116, v115, 10, &s2);
              v44 = v208;
              if ( !v118 )
              {
                v110 = s2.m128i_i32[0];
                v13 = s2.m128i_u32[0];
                if ( s2.m128i_i64[0] == s2.m128i_u32[0] )
                  goto LABEL_224;
              }
              v119 = ((__int64 (*)(void))sub_16E8C20)();
              v120 = v226;
              v121 = s1b;
              v122 = (__m128i *)v119;
              v123 = *(char **)(v119 + 24);
              v44 = v208;
              v13 = v122[1].m128i_i64[0] - (_QWORD)v123;
              if ( v13 <= 0x19 )
              {
                p_s2 = (__int64)"Unparseable architecture: ";
                v203 = sub_16E7EE0(v122, "Unparseable architecture: ", 26);
                v120 = v226;
                v121 = s1b;
                v44 = v208;
                v122 = (__m128i *)v203;
              }
              else
              {
                qmemcpy(v123, "Unparseable architecture: ", 0x1Au);
                p_s2 = (__int64)"";
                v11 = 0;
                v122[1].m128i_i64[1] += 26;
              }
              v14 = (__m128i *)v122[1].m128i_i64[1];
              if ( v120 > v122[1].m128i_i64[0] - (__int64)v14 )
              {
                v14 = v122;
                p_s2 = (__int64)v121;
                v239 = (size_t)v44;
                v202 = sub_16E7EE0(v122, v121, v120);
                v44 = (__m128i *)v239;
                v122 = (__m128i *)v202;
              }
              else if ( v120 )
              {
                p_s2 = (__int64)v121;
                v211 = v122;
                s1f = v44;
                v240 = v120;
                memcpy(v14, v121, v120);
                v122 = v211;
                v44 = s1f;
                v211[1].m128i_i64[1] += v240;
              }
              v124 = (_BYTE *)v122[1].m128i_i64[1];
              if ( (_BYTE *)v122[1].m128i_i64[0] != v124 )
              {
                *v124 = 10;
                ++v122[1].m128i_i64[1];
                goto LABEL_225;
              }
              goto LABEL_308;
            }
          }
        }
        else
        {
          v114 = 0;
        }
        v209 = (__m128i *)v111;
        s1c = v112;
        v232 = v114;
        v176 = ((__int64 (*)(void))sub_16E8C20)();
        v177 = v232;
        v178 = s1c;
        v179 = v176;
        v180 = *(__m128i **)(v176 + 24);
        v44 = v209;
        if ( *(_QWORD *)(v179 + 16) - (_QWORD)v180 <= 0x19u )
        {
          p_s2 = (__int64)"Unparseable architecture: ";
          v187 = sub_16E7EE0(v179, "Unparseable architecture: ", 26);
          v177 = v232;
          v178 = s1c;
          v14 = *(__m128i **)(v187 + 24);
          v44 = v209;
          v179 = v187;
        }
        else
        {
          v181 = _mm_load_si128((const __m128i *)&xmmword_3C23B10);
          v11 = 0x6572757463657469LL;
          qmemcpy(&v180[1], "itecture: ", 10);
          *v180 = v181;
          v14 = (__m128i *)(*(_QWORD *)(v179 + 24) + 26LL);
          *(_QWORD *)(v179 + 24) = v14;
        }
        v182 = *(__m128i **)(v179 + 16);
        v13 = (char *)v182 - (char *)v14;
        if ( v177 > (char *)v182 - (char *)v14 )
        {
          p_s2 = (__int64)v178;
          v233 = (size_t)v44;
          v186 = sub_16E7EE0(v179, v178, v177);
          v44 = (__m128i *)v233;
          v179 = v186;
          v182 = *(__m128i **)(v186 + 16);
          v14 = *(__m128i **)(v179 + 24);
        }
        else if ( v177 )
        {
          p_s2 = (__int64)v178;
          v210 = v179;
          s1i = v44;
          v238 = v177;
          memcpy(v14, v178, v177);
          v179 = v210;
          v44 = s1i;
          v14 = (__m128i *)(*(_QWORD *)(v210 + 24) + v238);
          v182 = *(__m128i **)(v210 + 16);
          *(_QWORD *)(v210 + 24) = v14;
        }
        if ( v14 == v182 )
        {
          p_s2 = (__int64)"\n";
          v14 = (__m128i *)v179;
          v234 = (size_t)v44;
          sub_16E7EE0(v179, "\n", 1);
          v44 = (__m128i *)v234;
        }
        else
        {
          v14->m128i_i8[0] = 10;
          ++*(_QWORD *)(v179 + 24);
        }
        goto LABEL_225;
      }
    }
    v52 = (__m128i *)((char *)v44 + v13);
    v53 = v42 - v13;
    if ( v43 != (__m128i *)v274 )
      j_j___libc_free_0(v43, v274[0] + 1LL);
    v14 = &s2;
    p_s2 = (__int64)&byte_3C23AC3[-11];
    s2.m128i_i64[0] = (__int64)v274;
    sub_8F98A0(s2.m128i_i64, (char)&byte_3C23AC3[-11], &byte_3C23AC3[-11], byte_3C23AC3);
    v54 = (__m128i *)s2.m128i_i64[0];
    if ( v53 != s2.m128i_i64[1]
      || v53
      && (p_s2 = s2.m128i_i64[0],
          v14 = v52,
          v254 = s2.m128i_i64[0],
          v105 = memcmp(v52, (const void *)s2.m128i_i64[0], v53),
          v54 = (__m128i *)v254,
          v105) )
    {
      if ( v54 != (__m128i *)v274 )
        j_j___libc_free_0(v54, v274[0] + 1LL);
      v14 = &s2;
      p_s2 = (__int64)&byte_3C23AB4[-6];
      s2.m128i_i64[0] = (__int64)v274;
      sub_8F98A0(s2.m128i_i64, (char)&byte_3C23AB4[-6], &byte_3C23AB4[-6], byte_3C23AB4);
      v55 = (__m128i *)s2.m128i_i64[0];
      if ( v53 != s2.m128i_i64[1]
        || v53
        && (p_s2 = s2.m128i_i64[0],
            v14 = v52,
            v255 = s2.m128i_i64[0],
            v148 = memcmp(v52, (const void *)s2.m128i_i64[0], v53),
            v55 = (__m128i *)v255,
            v148) )
      {
        v56 = v52;
        if ( v55 != (__m128i *)v274 )
          j_j___libc_free_0(v55, v274[0] + 1LL);
        v57 = ((__int64 (*)(void))sub_16E8C20)();
        v58 = *(__m128i **)(v57 + 24);
        if ( *(_QWORD *)(v57 + 16) - (_QWORD)v58 <= 0x12u )
        {
          v57 = sub_16E7EE0(v57, "Invalid option for ", 19);
        }
        else
        {
          v59 = _mm_load_si128((const __m128i *)&xmmword_3C23B00);
          v58[1].m128i_i8[2] = 32;
          v58[1].m128i_i16[0] = 29295;
          *v58 = v59;
          *(_QWORD *)(v57 + 24) += 19LL;
        }
        s2.m128i_i64[0] = (__int64)v274;
        sub_8F98A0(s2.m128i_i64, (char)&byte_3C23AAD[-13], &byte_3C23AAD[-13], byte_3C23AAD);
        v60 = sub_16E7EE0(v57, s2.m128i_i64[0], s2.m128i_i64[1]);
        if ( (_QWORD *)s2.m128i_i64[0] != v274 )
          j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
        v61 = *(_WORD **)(v60 + 24);
        if ( *(_QWORD *)(v60 + 16) - (_QWORD)v61 <= 1u )
        {
          p_s2 = (__int64)": ";
          v185 = sub_16E7EE0(v60, ": ", 2);
          v62 = *(_BYTE **)(v185 + 24);
          v60 = v185;
        }
        else
        {
          p_s2 = 8250;
          *v61 = 8250;
          v62 = (_BYTE *)(*(_QWORD *)(v60 + 24) + 2LL);
          *(_QWORD *)(v60 + 24) = v62;
        }
        v63 = *(_BYTE **)(v60 + 16);
        if ( v53 > v63 - v62 )
        {
          p_s2 = (__int64)v56;
          v60 = sub_16E7EE0(v60, v56, v53);
          v63 = *(_BYTE **)(v60 + 16);
          v62 = *(_BYTE **)(v60 + 24);
        }
        else if ( v53 )
        {
          p_s2 = (__int64)v56;
          memcpy(v62, v56, v53);
          v63 = *(_BYTE **)(v60 + 16);
          v62 = (_BYTE *)(v53 + *(_QWORD *)(v60 + 24));
          *(_QWORD *)(v60 + 24) = v62;
        }
        if ( v62 == v63 )
        {
          p_s2 = (__int64)"\n";
          sub_16E7EE0(v60, "\n", 1);
        }
        else
        {
          *v62 = 10;
          ++*(_QWORD *)(v60 + 24);
        }
        goto LABEL_60;
      }
      if ( v55 != (__m128i *)v274 )
      {
        v14 = v55;
        p_s2 = v274[0] + 1LL;
        j_j___libc_free_0(v55, v274[0] + 1LL);
      }
      v253 = 0;
    }
    else
    {
      if ( v54 != (__m128i *)v274 )
      {
        v14 = v54;
        p_s2 = v274[0] + 1LL;
        j_j___libc_free_0(v54, v274[0] + 1LL);
      }
      v253 = 1;
    }
LABEL_23:
    ++v15;
  }
  while ( v16 > v15 );
  v260 = v262 & v259;
  if ( !s )
  {
    v168 = sub_16E8C20(v14, p_s2, v13, v11);
    v169 = *(__m128i **)(v168 + 24);
    v170 = *(_QWORD *)(v168 + 16) - (_QWORD)v169;
    if ( v170 <= 0x12 )
    {
      p_s2 = (__int64)"Missing input file\n";
      sub_16E7EE0(v168, "Missing input file\n", 19);
    }
    else
    {
      v171 = _mm_load_si128((const __m128i *)&xmmword_3C23B50);
      v172 = 25964;
      v169[1].m128i_i8[2] = 10;
      v169[1].m128i_i16[0] = 25964;
      *v169 = v171;
      *(_QWORD *)(v168 + 24) += 19LL;
    }
    v173 = sub_16E8C20(v168, p_s2, v170, v172);
    v174 = *(__m128i **)(v173 + 24);
    if ( *(_QWORD *)(v173 + 16) - (_QWORD)v174 <= 0x3Du )
    {
      p_s2 = (__int64)"Recognized input file extensions are: .bc .ci .i .cup .optixir";
      sub_16E7EE0(v173, "Recognized input file extensions are: .bc .ci .i .cup .optixir", 62);
    }
    else
    {
      v175 = _mm_load_si128((const __m128i *)&xmmword_3C23B60);
      qmemcpy(&v174[3], " .cup .optixir", 14);
      *v174 = v175;
      v174[1] = _mm_load_si128((const __m128i *)&xmmword_3C23B70);
      v174[2] = _mm_load_si128((const __m128i *)&xmmword_3C23B80);
      *(_QWORD *)(v173 + 24) += 62LL;
    }
    goto LABEL_60;
  }
  if ( v253 == 2 )
  {
    p_s2 = (__int64)&byte_3C23A9F[-15];
    s2.m128i_i64[0] = (__int64)v274;
    sub_8F98A0(s2.m128i_i64, (char)&byte_3C23A9F[-15], &byte_3C23A9F[-15], byte_3C23A9F);
    v149 = getenv((const char *)s2.m128i_i64[0]);
    v14 = (__m128i *)s2.m128i_i64[0];
    v150 = v149;
    if ( (_QWORD *)s2.m128i_i64[0] != v274 )
    {
      p_s2 = v274[0] + 1LL;
      j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
    }
    if ( !v150 )
      goto LABEL_303;
    v14 = &s2;
    p_s2 = (__int64)&byte_3C23A82[-6];
    v151 = strlen(v150);
    s2.m128i_i64[0] = (__int64)v274;
    sub_8F98A0(s2.m128i_i64, (char)&byte_3C23A82[-6], &byte_3C23A82[-6], byte_3C23A82);
    v152 = (__m128i *)s2.m128i_i64[0];
    if ( v151 != s2.m128i_i64[1] )
    {
      if ( (_QWORD *)s2.m128i_i64[0] != v274 )
        j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
LABEL_299:
      v14 = &s2;
      p_s2 = (__int64)&byte_3C23A7B[-11];
      s2.m128i_i64[0] = (__int64)v274;
      sub_8F98A0(s2.m128i_i64, (char)&byte_3C23A7B[-11], &byte_3C23A7B[-11], byte_3C23A7B);
      v153 = (__m128i *)s2.m128i_i64[0];
      if ( v151 == s2.m128i_i64[1] )
      {
        if ( v151 )
        {
          p_s2 = s2.m128i_i64[0];
          v14 = (__m128i *)v150;
          v154 = memcmp(v150, (const void *)s2.m128i_i64[0], v151) == 0;
          if ( v153 == (__m128i *)v274 )
          {
LABEL_302:
            if ( v154 )
              goto LABEL_346;
            goto LABEL_303;
          }
        }
        else
        {
          v154 = 1;
          if ( (_QWORD *)s2.m128i_i64[0] == v274 )
            goto LABEL_346;
        }
LABEL_301:
        v14 = v153;
        p_s2 = v274[0] + 1LL;
        j_j___libc_free_0(v153, v274[0] + 1LL);
        goto LABEL_302;
      }
      v154 = 0;
      if ( (_QWORD *)s2.m128i_i64[0] != v274 )
        goto LABEL_301;
LABEL_303:
      if ( v242 <= 0x63 || v258 )
        goto LABEL_305;
LABEL_346:
      if ( v243 || v263 <= 1 )
        goto LABEL_332;
LABEL_333:
      v253 = 1;
      goto LABEL_28;
    }
    if ( v151 )
    {
      p_s2 = s2.m128i_i64[0];
      v14 = (__m128i *)v150;
      v188 = memcmp(v150, (const void *)s2.m128i_i64[0], v151);
      if ( v152 != (__m128i *)v274 )
      {
        v14 = v152;
        p_s2 = v274[0] + 1LL;
        j_j___libc_free_0(v152, v274[0] + 1LL);
      }
      if ( v188 )
        goto LABEL_299;
    }
    else if ( (_QWORD *)s2.m128i_i64[0] != v274 )
    {
      v14 = (__m128i *)s2.m128i_i64[0];
      p_s2 = v274[0] + 1LL;
      j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
    }
LABEL_305:
    if ( v243 )
    {
      if ( !v263 )
        goto LABEL_249;
    }
    else if ( v263 <= 1 )
    {
      goto LABEL_360;
    }
    goto LABEL_307;
  }
  if ( v243 )
  {
    if ( !v263 )
    {
      if ( v253 == 1 )
        goto LABEL_267;
      goto LABEL_249;
    }
  }
  else
  {
    if ( v263 > 1 )
      goto LABEL_28;
    if ( v253 )
    {
LABEL_332:
      if ( v263 )
        goto LABEL_333;
LABEL_267:
      p_s2 = 0;
      v8 = sub_902D10(dest.m128i_i64[0], 0, &v266, s, v257, v256, v260, v262, v261);
      goto LABEL_61;
    }
LABEL_360:
    v271.m128i_i64[0] = 20;
    s2.m128i_i64[0] = (__int64)v274;
    v183 = sub_22409D0(&s2, &v271, 0);
    p_s2 = (__int64)&s2;
    v184 = _mm_load_si128((const __m128i *)&xmmword_3C23B40);
    s2.m128i_i64[0] = v183;
    v274[0] = v271.m128i_i64[0];
    *(_DWORD *)(v183 + 16) = 942813556;
    *(__m128i *)v183 = v184;
    s2.m128i_i64[1] = v271.m128i_i64[0];
    *(_BYTE *)(s2.m128i_i64[0] + v271.m128i_i64[0]) = 0;
    sub_8F9C20(&v266, &s2);
    v14 = (__m128i *)s2.m128i_i64[0];
    if ( (_QWORD *)s2.m128i_i64[0] != v274 )
    {
      p_s2 = v274[0] + 1LL;
      j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
    }
    if ( !v263 )
    {
LABEL_249:
      p_s2 = 0;
      v8 = sub_1262860(dest.m128i_i64[0], 0, &v266, s, v257, v256, v260, v262, v261);
      goto LABEL_61;
    }
LABEL_307:
    v253 = 0;
  }
LABEL_28:
  s2.m128i_i64[1] = 0;
  LOBYTE(v274[0]) = 0;
  s2.m128i_i64[0] = (__int64)v274;
  v275 = v277;
  v276 = 0;
  LOBYTE(v277[0]) = 0;
  v280 = 0;
  v281 = 0;
  v278 = 1;
  v279 = 0;
  if ( v257 )
  {
    v23 = strlen(s);
    sub_2241130(&s2, 0, 0, s, v23);
    v24 = strlen(v257);
    sub_2241130(&v275, 0, v276, v257, v24);
    v279 = (unsigned int)(v263 - 3) > 1;
    HIBYTE(v278) = v263 == 5;
    if ( v263 != 1 )
    {
      v271.m128i_i64[0] = (__int64)v272;
      if ( v253 == 1 )
      {
        v265 = 25;
        v25 = sub_22409D0(&v271, &v265, 0);
        v26 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
        v271.m128i_i64[0] = v25;
        v272[0] = v265;
        *(_QWORD *)(v25 + 16) = 0x736574616C2D6D76LL;
        *(_BYTE *)(v25 + 24) = 116;
      }
      else
      {
        v265 = 20;
        v25 = sub_22409D0(&v271, &v265, 0);
        v26 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
        v271.m128i_i64[0] = v25;
        v272[0] = v265;
        *(_DWORD *)(v25 + 16) = 808938870;
      }
      *(__m128i *)v25 = v26;
      v271.m128i_i64[1] = v265;
      *(_BYTE *)(v271.m128i_i64[0] + v265) = 0;
      sub_8F9C20(&v266, &v271);
      if ( (_QWORD *)v271.m128i_i64[0] != v272 )
        j_j___libc_free_0(v271.m128i_i64[0], v272[0] + 1LL);
      if ( v263 == 2 )
      {
        v280 = ((char *)v267 - (char *)v266) >> 5;
        v27 = 8LL * v280;
        if ( (unsigned __int64)v280 > 0xFFFFFFFFFFFFFFFLL )
          v27 = -1;
        v28 = sub_2207820(v27);
        v281 = (char **)v28;
        if ( v280 > 0 )
        {
          v29 = 0;
          while ( 1 )
          {
            v30 = v266[2 * v29].m128i_i64[1];
            *(_QWORD *)(v28 + 8 * v29) = sub_2207820(v30 + 1);
            sub_2241570(&v266[2 * v29], v281[v29], v30, 0);
            v31 = v281[v29++];
            v31[v30] = 0;
            if ( v280 <= (int)v29 )
              break;
            v28 = (__int64)v281;
          }
        }
        v265 = 0;
        v264 = 0;
        goto LABEL_42;
      }
    }
    v125 = ((char *)v267 - (char *)v266) >> 5;
    v126 = (const char *)*((_QWORD *)v282 + 1);
    v280 = v125 + 1;
    v127 = (int)v125 + 1;
    v128 = 8 * v127;
    if ( v127 > 0xFFFFFFFFFFFFFFFLL )
      v128 = -1;
    v129 = (char **)sub_2207820(v128);
    v281 = v129;
    v130 = strlen(v126);
    *v129 = (char *)sub_2207820(v130 + 1);
    strcpy(*v281, v126);
    if ( (int)v125 > 0 )
    {
      v131 = 8;
      v132 = 8LL * (unsigned int)(v125 - 1) + 16;
      do
      {
        v133 = 4 * v131 - 32;
        v134 = *(__int64 *)((char *)&v266->m128i_i64[1] + v133);
        v135 = &v281[(unsigned __int64)v131 / 8];
        *v135 = (char *)sub_2207820(v134 + 1);
        sub_2241570(&v266->m128i_i8[v133], v281[(unsigned __int64)v131 / 8], v134, 0);
        v136 = v281[(unsigned __int64)v131 / 8];
        v131 += 8;
        v136[v134] = 0;
      }
      while ( v132 != v131 );
    }
    v265 = 0;
    v264 = 0;
    if ( v263 == 1 )
    {
      if ( v253 == 1 )
      {
        sub_905E50(0, (unsigned int)&s2, (unsigned int)&v265, (unsigned int)&v264, v260, 0, v261, 1);
        p_s2 = (__int64)v206;
      }
      else
      {
        p_s2 = (__int64)&s2;
        sub_12658E0(0, (unsigned int)&s2, (unsigned int)&v265, (unsigned int)&v264, v260, 0, v261, 1);
      }
    }
    else
    {
LABEL_42:
      v271.m128i_i64[1] = 0;
      LOBYTE(v272[0]) = 0;
      v271.m128i_i64[0] = (__int64)v272;
      p_s2 = 0;
      if ( v253 == 1 )
        sub_905EE0(
          dest.m128i_i32[0],
          0,
          (unsigned int)&s2,
          (_DWORD)v256,
          v250,
          (unsigned int)&v271,
          (__int64)&v265,
          (__int64)&v264,
          v260,
          v262,
          0,
          v261);
      else
        sub_1265970(
          dest.m128i_i32[0],
          0,
          (unsigned int)&s2,
          (_DWORD)v256,
          v250,
          (unsigned int)&v271,
          (__int64)&v265,
          (__int64)&v264,
          v260,
          v262,
          0,
          v261);
      if ( (_QWORD *)v271.m128i_i64[0] != v272 )
      {
        p_s2 = v272[0] + 1LL;
        j_j___libc_free_0(v271.m128i_i64[0], v272[0] + 1LL);
      }
    }
    v8 = v264;
  }
  else
  {
    v145 = sub_16E8CB0(v14, p_s2, v13);
    v146 = *(__m128i **)(v145 + 24);
    if ( *(_QWORD *)(v145 + 16) - (_QWORD)v146 <= 0x35u )
    {
      p_s2 = (__int64)"Error: Output file was not specified (See -o option).\n";
      sub_16E7EE0(v145, "Error: Output file was not specified (See -o option).\n", 54);
    }
    else
    {
      v147 = _mm_load_si128((const __m128i *)&xmmword_3C23B90);
      v146[3].m128i_i32[0] = 695103337;
      v146[3].m128i_i16[2] = 2606;
      *v146 = v147;
      v146[1] = _mm_load_si128((const __m128i *)&xmmword_3C23BA0);
      v146[2] = _mm_load_si128((const __m128i *)&xmmword_3C23BB0);
      *(_QWORD *)(v145 + 24) += 54LL;
    }
    v8 = 1;
  }
  v32 = v281;
  if ( v281 )
  {
    if ( v280 <= 0 )
      goto LABEL_54;
    v33 = 0;
    do
    {
      if ( v32[v33] )
      {
        ((void (*)(void))j_j___libc_free_0_0)();
        v32 = v281;
      }
      ++v33;
    }
    while ( v280 > (int)v33 );
    if ( v32 )
LABEL_54:
      j_j___libc_free_0_0(v32);
  }
  if ( v275 != v277 )
  {
    p_s2 = v277[0] + 1LL;
    j_j___libc_free_0(v275, v277[0] + 1LL);
  }
  if ( (_QWORD *)s2.m128i_i64[0] != v274 )
  {
    p_s2 = v274[0] + 1LL;
    j_j___libc_free_0(s2.m128i_i64[0], v274[0] + 1LL);
  }
LABEL_61:
  v34 = v267;
  v35 = v266;
  if ( v267 != v266 )
  {
    do
    {
      if ( (__m128i *)v35->m128i_i64[0] != &v35[1] )
      {
        p_s2 = v35[1].m128i_i64[0] + 1;
        j_j___libc_free_0(v35->m128i_i64[0], p_s2);
      }
      v35 += 2;
    }
    while ( v34 != v35 );
    v35 = v266;
  }
  if ( v35 )
  {
    p_s2 = v268 - (_QWORD)v35;
    j_j___libc_free_0(v35, v268 - (_QWORD)v35);
  }
  if ( (_QWORD *)dest.m128i_i64[0] != v270 )
  {
    p_s2 = v270[0] + 1LL;
    j_j___libc_free_0(dest.m128i_i64[0], v270[0] + 1LL);
  }
  v6 = v282;
LABEL_71:
  if ( v6 != v284 )
    _libc_free(v6, p_s2);
  return v8;
}
