// Function: sub_2A05E00
// Address: 0x2a05e00
//
void __fastcall sub_2A05E00(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        int a4,
        __int64 *a5,
        __int64 *a6,
        __int64 a7,
        unsigned int a8)
{
  __int64 v8; // r15
  unsigned int v12; // eax
  _QWORD *v13; // r14
  _QWORD *v14; // r12
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned int v27; // ecx
  unsigned int v28; // eax
  __int64 **v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  unsigned int v32; // ebx
  unsigned int v33; // eax
  __int64 v34; // rdi
  __int64 (__fastcall **v35)(unsigned __int64 *, const __m128i **, int); // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 (__fastcall *v38)(unsigned __int64 *, const __m128i **, int); // rax
  __int64 v39; // r12
  __int64 v40; // r13
  char v41; // al
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rcx
  unsigned int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rbx
  __int64 v48; // r14
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rcx
  __int64 *v52; // rbx
  __int64 v53; // r8
  __int64 v54; // rax
  unsigned int v55; // eax
  int v56; // edi
  bool v57; // al
  bool v58; // r14
  unsigned int v59; // r14d
  __int64 v60; // rax
  _QWORD *v61; // rax
  _BYTE *v62; // rbx
  unsigned __int64 v63; // r12
  _BYTE *v64; // r15
  unsigned __int64 v65; // rbx
  unsigned __int64 v66; // rax
  unsigned int v67; // ebx
  unsigned int v68; // r13d
  unsigned int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned __int64 v72; // rax
  __int64 v73; // r13
  int v74; // eax
  unsigned int v75; // ebx
  int v76; // r12d
  __int64 v77; // rsi
  _QWORD *v78; // rax
  _QWORD *v79; // rdx
  __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  unsigned int v83; // r13d
  __int64 v84; // rbx
  __int64 v85; // r14
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  unsigned int *v89; // r8
  __int64 v90; // rax
  __int64 v91; // rsi
  unsigned int *v92; // rax
  unsigned int *v93; // rsi
  unsigned __int64 v94; // rdx
  __int64 v95; // rcx
  unsigned __int64 v96; // rdx
  __int64 v97; // rcx
  unsigned __int64 v98; // rdx
  __int64 v99; // rcx
  unsigned __int64 v100; // rdx
  signed __int64 v101; // rdx
  __int64 v102; // rax
  __int64 *v103; // rax
  __int64 v104; // r13
  __int64 v105; // rbx
  __int64 v106; // r15
  __int64 v107; // rcx
  __int64 v108; // r9
  __int64 *v109; // rax
  __int64 *v110; // rdx
  __int64 v111; // r14
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 (__fastcall **v114)(unsigned __int64 *, const __m128i **, int); // r12
  __int64 v115; // rax
  __int64 (__fastcall **v116)(unsigned __int64 *, const __m128i **, int); // rbx
  __int64 v117; // rax
  __int64 (__fastcall **v118)(unsigned __int64 *, const __m128i **, int); // r14
  unsigned __int64 v119; // rax
  int v120; // edx
  __int64 v121; // rsi
  __int64 v122; // rax
  _QWORD *v123; // rax
  __int64 v124; // rcx
  _QWORD *v125; // rdi
  _QWORD *v126; // rdx
  unsigned int v127; // ebx
  __int64 v128; // r14
  __int64 v129; // r8
  __int64 v130; // rsi
  _QWORD *v131; // rax
  unsigned int v132; // ebx
  char v133; // al
  __int64 *v134; // rdx
  __int64 v135; // r8
  __int64 v136; // r9
  __int64 v137; // rcx
  __int64 v138; // r14
  char v139; // di
  __int64 v140; // rsi
  _QWORD *v141; // rax
  __int64 *v142; // r14
  __int64 v143; // rax
  __int64 *v144; // r12
  __int64 v145; // rax
  __int64 *v146; // rbx
  unsigned __int64 v147; // rdi
  bool v148; // bl
  __int64 (__fastcall *v149)(unsigned __int64 *, const __m128i **, int); // rsi
  __int64 (__fastcall **v150)(unsigned __int64 *, const __m128i **, int); // r8
  unsigned __int64 v151; // rdx
  __int64 v152; // rsi
  _QWORD *v153; // rdi
  _QWORD *v154; // rdx
  __int64 *v155; // rax
  __int64 (__fastcall *v156)(unsigned __int64 *, const __m128i **, int); // rax
  unsigned __int64 v157; // rdx
  __int64 v158; // rsi
  _QWORD *v159; // rdi
  _QWORD *v160; // rdx
  __int64 (__fastcall *v161)(unsigned __int64 *, const __m128i **, int); // rsi
  __int64 (__fastcall *v162)(unsigned __int64 *, const __m128i **, int); // rsi
  unsigned __int64 v163; // rdx
  __int64 v164; // rsi
  _QWORD *v165; // rdx
  __int64 *v166; // rax
  signed __int64 v167; // rax
  __int64 *v168; // rax
  __int64 (__fastcall *v169)(unsigned __int64 *, const __m128i **, int); // rax
  __int64 (__fastcall *v170)(unsigned __int64 *, const __m128i **, int); // rax
  unsigned __int64 v171; // rdx
  unsigned __int64 v172; // rdx
  unsigned __int64 v173; // rdx
  __int64 (__fastcall *v174)(unsigned __int64 *, const __m128i **, int); // rsi
  __int64 (__fastcall *v175)(unsigned __int64 *, const __m128i **, int); // rsi
  unsigned __int64 v176; // rax
  __int64 v177; // rsi
  _QWORD *j; // rax
  unsigned __int64 v179; // rax
  __int64 v180; // rsi
  _QWORD *k; // rax
  unsigned __int64 v182; // rax
  __int64 v183; // rcx
  _QWORD *i; // rax
  signed __int64 v185; // rax
  __int64 (__fastcall *v186)(unsigned __int64 *, const __m128i **, int); // rax
  __int64 v187; // [rsp-10h] [rbp-1E0h]
  __int64 v188; // [rsp+8h] [rbp-1C8h]
  __int64 v189; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v190; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v191; // [rsp+28h] [rbp-1A8h]
  unsigned int v192; // [rsp+34h] [rbp-19Ch]
  unsigned int v193; // [rsp+38h] [rbp-198h]
  __int64 *v194; // [rsp+38h] [rbp-198h]
  __int64 v195; // [rsp+40h] [rbp-190h]
  __int64 v196; // [rsp+50h] [rbp-180h]
  __int64 v197; // [rsp+58h] [rbp-178h]
  __int64 v198; // [rsp+58h] [rbp-178h]
  unsigned int v200; // [rsp+68h] [rbp-168h]
  unsigned int v202; // [rsp+70h] [rbp-160h]
  __int64 *v203; // [rsp+70h] [rbp-160h]
  __int64 *v204; // [rsp+78h] [rbp-158h]
  __int64 v205; // [rsp+78h] [rbp-158h]
  __int64 v206; // [rsp+80h] [rbp-150h]
  __int64 v207; // [rsp+80h] [rbp-150h]
  __int64 *v209; // [rsp+90h] [rbp-140h]
  __int64 v210; // [rsp+90h] [rbp-140h]
  __int64 (__fastcall **v211)(unsigned __int64 *, const __m128i **, int); // [rsp+90h] [rbp-140h]
  __int64 (__fastcall **v212)(unsigned __int64 *, const __m128i **, int); // [rsp+90h] [rbp-140h]
  __int64 (__fastcall **v213)(unsigned __int64 *, const __m128i **, int); // [rsp+90h] [rbp-140h]
  __int64 v214; // [rsp+98h] [rbp-138h]
  __int64 *v215; // [rsp+98h] [rbp-138h]
  __int64 v216; // [rsp+98h] [rbp-138h]
  unsigned int v217; // [rsp+A8h] [rbp-128h] BYREF
  unsigned int v218; // [rsp+ACh] [rbp-124h] BYREF
  int v219; // [rsp+B0h] [rbp-120h] BYREF
  int v220; // [rsp+B4h] [rbp-11Ch] BYREF
  __int64 v221; // [rsp+B8h] [rbp-118h] BYREF
  unsigned int *v222; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v223; // [rsp+C8h] [rbp-108h]
  _BYTE v224[32]; // [rsp+D0h] [rbp-100h] BYREF
  __int64 (__fastcall **v225)(unsigned __int64 *, const __m128i **, int); // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v226; // [rsp+F8h] [rbp-D8h]
  __int64 (__fastcall *v227)(unsigned __int64 *, const __m128i **, int); // [rsp+100h] [rbp-D0h] BYREF
  __int16 (__fastcall *v228)(_QWORD *, unsigned __int8 **, unsigned int *); // [rsp+108h] [rbp-C8h]
  __int64 *v229; // [rsp+130h] [rbp-A0h] BYREF
  unsigned __int64 v230; // [rsp+138h] [rbp-98h]
  __int64 v231; // [rsp+140h] [rbp-90h] BYREF
  __int64 v232; // [rsp+148h] [rbp-88h]
  __int64 v233; // [rsp+150h] [rbp-80h] BYREF
  __int64 v234; // [rsp+158h] [rbp-78h] BYREF
  unsigned int v235; // [rsp+160h] [rbp-70h]
  char v236; // [rsp+198h] [rbp-38h] BYREF

  v8 = a1;
  v12 = *(_DWORD *)a3;
  *(_DWORD *)a3 = 0;
  v202 = v12;
  if ( !sub_2A04CA0(a1) || !*(_BYTE *)(a3 + 5) && *(_QWORD *)(a1 + 8) != *(_QWORD *)(a1 + 16) )
    return;
  v13 = sub_C52410();
  v14 = v13 + 1;
  v15 = sub_C959E0();
  v16 = (_QWORD *)v13[2];
  if ( v16 )
  {
    v17 = v13 + 1;
    do
    {
      while ( 1 )
      {
        v18 = v16[2];
        v19 = v16[3];
        if ( v15 <= v16[4] )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v19 )
          goto LABEL_9;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_9:
    if ( v17 != v14 && v15 >= v17[4] )
      v14 = v17;
  }
  if ( v14 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v23 = v14[7];
    v21 = (__int64)(v14 + 6);
    if ( v23 )
    {
      v24 = v14 + 6;
      do
      {
        while ( 1 )
        {
          v20 = *(_QWORD *)(v23 + 16);
          v25 = *(_QWORD *)(v23 + 24);
          if ( *(_DWORD *)(v23 + 32) >= dword_5009AA8 )
            break;
          v23 = *(_QWORD *)(v23 + 24);
          if ( !v25 )
            goto LABEL_18;
        }
        v24 = (_QWORD *)v23;
        v23 = *(_QWORD *)(v23 + 16);
      }
      while ( v20 );
LABEL_18:
      if ( v24 != (_QWORD *)v21 && dword_5009AA8 >= *((_DWORD *)v24 + 8) && *((int *)v24 + 9) > 0 )
      {
        *(_DWORD *)a3 = qword_5009B28;
        *(_BYTE *)(a3 + 6) = 1;
        return;
      }
    }
  }
  if ( !*(_BYTE *)(a3 + 4) || 2 * a2 > a8 )
    return;
  v26 = sub_D4A2B0(v8, "llvm.loop.peeled.count", 0x16u, v20, v21, v22);
  v27 = qword_5009C08;
  v229 = (__int64 *)v26;
  v28 = 0;
  if ( BYTE4(v26) )
    v28 = (unsigned int)v229;
  v192 = v28;
  if ( v28 >= (unsigned int)qword_5009C08 )
    return;
  if ( a8 / a2 - 1 <= (unsigned int)qword_5009C08 )
    v27 = a8 / a2 - 1;
  v200 = v27;
  if ( v202 < v27 )
  {
    BYTE4(v229) = 0;
    v80 = &v234;
    v230 = v8;
    LODWORD(v231) = v27;
    v232 = 0;
    v233 = 1;
    do
    {
      *v80 = -4096;
      v80 += 2;
    }
    while ( v80 != (__int64 *)&v236 );
    v81 = sub_AA5930(**(_QWORD **)(v230 + 32));
    if ( v81 == v82 )
      goto LABEL_235;
    v215 = a6;
    v83 = 0;
    v84 = v81;
    v85 = v82;
    do
    {
      v87 = sub_2A05820((int *)&v229, (unsigned __int8 *)v84);
      v225 = (__int64 (__fastcall **)(unsigned __int64 *, const __m128i **, int))v87;
      if ( BYTE4(v87) != BYTE4(v229) || BYTE4(v87) && (_DWORD)v87 != (_DWORD)v229 )
      {
        if ( v83 < (unsigned int)v87 )
          v83 = v87;
        if ( (_DWORD)v231 == v83 )
          break;
      }
      if ( !v84 )
LABEL_404:
        BUG();
      v86 = *(_QWORD *)(v84 + 32);
      if ( !v86 )
        goto LABEL_402;
      v84 = 0;
      if ( *(_BYTE *)(v86 - 24) == 84 )
        v84 = v86 - 24;
    }
    while ( v85 != v84 );
    v132 = v83;
    a6 = v215;
    if ( !v132 )
    {
LABEL_235:
      if ( (v233 & 1) == 0 )
        sub_C7D6A0(v234, 16LL * v235, 8);
    }
    else
    {
      if ( (v233 & 1) == 0 )
        sub_C7D6A0(v234, 16LL * v235, 8);
      if ( v202 >= v132 )
        v132 = v202;
      v202 = v132;
    }
  }
  v29 = (__int64 **)v8;
  v218 = 0;
  v217 = v200;
  v30 = sub_DCF3A0(a6, (char *)v8, 1);
  if ( !*(_WORD *)(v30 + 24) )
  {
    v31 = *(_QWORD *)(v30 + 32);
    v32 = *(_DWORD *)(v31 + 32);
    if ( v32 > 0x40 )
    {
      v127 = v32 - sub_C444A0(v31 + 24);
      v33 = -2;
      if ( v127 <= 0x40 )
        v33 = **(_QWORD **)(v31 + 24) - 1;
    }
    else
    {
      v33 = *(_QWORD *)(v31 + 24) - 1;
    }
    if ( v217 <= v33 )
      v33 = v217;
    v217 = v33;
  }
  v34 = 48;
  v223 = (__int64)a6;
  v222 = &v217;
  v227 = 0;
  v35 = (__int64 (__fastcall **)(unsigned __int64 *, const __m128i **, int))sub_22077B0(0x30u);
  if ( v35 )
  {
    v35[1] = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))a6;
    *v35 = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))&v225;
    v35[3] = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))&v218;
    v35[2] = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v8;
    v35[4] = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))&v222;
    v35[5] = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))&v217;
  }
  v36 = *(_QWORD *)(v8 + 40);
  v37 = *(_QWORD *)(v8 + 32);
  v225 = v35;
  v228 = sub_2A04840;
  v38 = sub_2A04470;
  v227 = sub_2A04470;
  v197 = v36;
  if ( v37 == v36 )
  {
    v67 = v218;
    goto LABEL_93;
  }
  v214 = v37;
  v209 = a6;
  do
  {
    v39 = *(_QWORD *)v214 + 48LL;
    v206 = *(_QWORD *)v214;
    v40 = *(_QWORD *)(*(_QWORD *)v214 + 56LL);
    if ( v40 != v39 )
    {
      while ( 1 )
      {
        if ( !v40 )
          goto LABEL_402;
        v41 = *(_BYTE *)(v40 - 24);
        if ( v41 == 86 )
        {
          v42 = *(_QWORD *)(v40 - 120);
          v219 = 0;
          v221 = v42;
          if ( !v227 )
            goto LABEL_275;
          v29 = (__int64 **)&v221;
          v34 = (__int64)&v225;
          v228(&v225, (unsigned __int8 **)&v221, (unsigned int *)&v219);
          v41 = *(_BYTE *)(v40 - 24);
        }
        if ( v41 != 85 )
          goto LABEL_43;
        v43 = *(_QWORD *)(v40 - 56);
        if ( !v43 )
          goto LABEL_43;
        if ( *(_BYTE *)v43 )
          goto LABEL_43;
        v44 = *(_QWORD *)(v40 + 56);
        if ( *(_QWORD *)(v43 + 24) != v44 || (*(_BYTE *)(v43 + 33) & 0x20) == 0 )
          goto LABEL_43;
        v45 = *(_DWORD *)(v43 + 36);
        if ( v45 > 0x14A )
        {
          if ( v45 - 365 <= 1 )
            goto LABEL_55;
          v40 = *(_QWORD *)(v40 + 8);
          if ( v39 == v40 )
            break;
        }
        else
        {
          if ( v45 <= 0x148 )
            goto LABEL_43;
LABEL_55:
          if ( *(_BYTE *)(*(_QWORD *)(v40 - 16) + 8LL) != 12 )
            goto LABEL_43;
          v46 = *(_DWORD *)(v40 - 20) & 0x7FFFFFF;
          v47 = *(_QWORD *)(v40 - 32 * v46 - 24);
          v48 = *(_QWORD *)(v40 + 32 * (1 - v46) - 24);
          if ( (unsigned __int8)sub_D48480(v8, v47, v46, v44) )
          {
            v34 = (__int64)v209;
            v29 = (__int64 **)v48;
            v204 = sub_DD8400((__int64)v209, v47);
            v52 = sub_DD8400((__int64)v209, v48);
          }
          else
          {
            v29 = (__int64 **)v48;
            v34 = v8;
            if ( !(unsigned __int8)sub_D48480(v8, v48, v49, v50) )
              goto LABEL_43;
            v29 = (__int64 **)v47;
            v34 = (__int64)v209;
            v204 = sub_DD8400((__int64)v209, v48);
            v52 = sub_DD8400((__int64)v209, v47);
          }
          if ( *((_WORD *)v52 + 12) != 8 || v52[5] != 2 || v8 != v52[6] )
            goto LABEL_43;
          v196 = sub_D33D80(v52, (__int64)v209, v37, v51, v53);
          v54 = *(_QWORD *)(v40 - 56);
          if ( !v54 || *(_BYTE *)v54 || *(_QWORD *)(v54 + 24) != *(_QWORD *)(v40 + 56) )
            BUG();
          v55 = *(_DWORD *)(v54 + 36);
          if ( v55 == 365 )
          {
            v56 = 34;
          }
          else if ( v55 > 0x16D )
          {
            if ( v55 != 366 )
              goto LABEL_404;
            v56 = 36;
          }
          else if ( v55 == 329 )
          {
            v56 = 38;
          }
          else
          {
            if ( v55 != 330 )
              goto LABEL_404;
            v56 = 40;
          }
          v57 = sub_B532B0(v56);
          v29 = (__int64 **)v196;
          v34 = (__int64)v209;
          v58 = v57;
          if ( (unsigned __int8)sub_DBEDC0((__int64)v209, v196) )
          {
            if ( !v58 )
            {
              v193 = 36;
              goto LABEL_71;
            }
            v193 = 40;
          }
          else
          {
            v29 = (__int64 **)v196;
            v34 = (__int64)v209;
            if ( !(unsigned __int8)sub_DBEC00((__int64)v209, v196) )
              goto LABEL_43;
            if ( !v58 )
            {
              v193 = 34;
LABEL_71:
              if ( (*((_BYTE *)v52 + 28) & 2) != 0 )
                goto LABEL_72;
              goto LABEL_43;
            }
            v193 = 38;
          }
          if ( (*((_BYTE *)v52 + 28) & 4) != 0 )
          {
LABEL_72:
            v59 = v218;
            v60 = sub_D95540(*(_QWORD *)v52[4]);
            v61 = sub_DA2C50((__int64)v209, v60, v59, 0);
            v62 = sub_DD0540((__int64)v52, (__int64)v61, v209);
            if ( v59 < *v222 )
            {
              v189 = v39;
              v63 = v190;
              v188 = v8;
              v64 = v62;
              do
              {
                v63 = v193 | v63 & 0xFFFFFF0000000000LL;
                if ( !(unsigned __int8)sub_DC3A60(v223, v63, v64, v204) )
                  break;
                v231 = (__int64)v64;
                v232 = v196;
                v229 = &v231;
                v230 = 0x200000002LL;
                v64 = sub_DC7EB0((__int64 *)v223, (__int64)&v229, 0, 0);
                if ( v229 != &v231 )
                  _libc_free((unsigned __int64)v229);
                ++v59;
              }
              while ( *v222 > v59 );
              v190 = v63;
              v62 = v64;
              v39 = v189;
              v8 = v188;
            }
            v34 = v223;
            v191 = (unsigned int)sub_B52870(v193) | v191 & 0xFFFFFF0000000000LL;
            v29 = (__int64 **)v191;
            if ( (unsigned __int8)sub_DC3A60(v34, v191, v62, v204) )
              v218 = v59;
          }
LABEL_43:
          v40 = *(_QWORD *)(v40 + 8);
          if ( v39 == v40 )
            break;
        }
      }
    }
    v65 = *(_QWORD *)(v206 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v39 == v65 || !v65 || (unsigned int)*(unsigned __int8 *)(v65 - 24) - 30 > 0xA )
      goto LABEL_402;
    if ( *(_BYTE *)(v65 - 24) == 31 && (*(_DWORD *)(v65 - 20) & 0x7FFFFFF) != 1 )
    {
      v34 = v8;
      if ( v206 != sub_D47930(v8) )
      {
        v66 = *(_QWORD *)(v65 - 120);
        v220 = 0;
        v229 = (__int64 *)v66;
        if ( !v227 )
LABEL_275:
          sub_4263D6(v34, v29, v37);
        v29 = &v229;
        v34 = (__int64)&v225;
        v228(&v225, (unsigned __int8 **)&v229, (unsigned int *)&v220);
      }
    }
    v214 += 8;
  }
  while ( v197 != v214 );
  v38 = v227;
  v67 = v218;
  if ( v227 )
LABEL_93:
    v38((unsigned __int64 *)&v225, (const __m128i **)&v225, 3);
  v68 = v202;
  if ( v202 < v67 )
    v68 = v67;
  if ( v68 )
    goto LABEL_97;
  v207 = sub_D46F00(v8);
  if ( !v207 )
  {
    v222 = (unsigned int *)v224;
    v223 = 0x400000000LL;
    sub_D4C2F0(v8, (__int64)&v222);
    v88 = 2LL * (unsigned int)v223;
    v89 = &v222[v88];
    v90 = (v88 * 4) >> 5;
    if ( v90 )
    {
      v91 = 8 * v90;
      v92 = v222;
      v93 = &v222[v91];
      while ( 1 )
      {
        v94 = *(_QWORD *)(*(_QWORD *)v92 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v94 == *(_QWORD *)v92 + 48LL || !v94 || (unsigned int)*(unsigned __int8 *)(v94 - 24) - 30 > 0xA )
          goto LABEL_402;
        if ( *(_BYTE *)(v94 - 24) != 36 )
          goto LABEL_253;
        v95 = *((_QWORD *)v92 + 1);
        v96 = *(_QWORD *)(v95 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v96 == v95 + 48 || !v96 || (unsigned int)*(unsigned __int8 *)(v96 - 24) - 30 > 0xA )
          goto LABEL_402;
        if ( *(_BYTE *)(v96 - 24) != 36 )
          break;
        v97 = *((_QWORD *)v92 + 2);
        v98 = *(_QWORD *)(v97 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v98 == v97 + 48 || !v98 || (unsigned int)*(unsigned __int8 *)(v98 - 24) - 30 > 0xA )
          goto LABEL_402;
        if ( *(_BYTE *)(v98 - 24) != 36 )
        {
          v92 += 4;
          goto LABEL_253;
        }
        v99 = *((_QWORD *)v92 + 3);
        v100 = *(_QWORD *)(v99 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v100 == v99 + 48 || !v100 || (unsigned int)*(unsigned __int8 *)(v100 - 24) - 30 > 0xA )
          goto LABEL_402;
        if ( *(_BYTE *)(v100 - 24) != 36 )
        {
          v92 += 6;
          goto LABEL_253;
        }
        v92 += 8;
        if ( v93 == v92 )
          goto LABEL_156;
      }
      v92 += 2;
LABEL_253:
      if ( v89 != v92 )
      {
        if ( v222 != (unsigned int *)v224 )
          _libc_free((unsigned __int64)v222);
        goto LABEL_100;
      }
      goto LABEL_159;
    }
    v92 = v222;
LABEL_156:
    v101 = (char *)v89 - (char *)v92;
    if ( (char *)v89 - (char *)v92 != 16 )
    {
      if ( v101 != 24 )
      {
        if ( v101 != 8 )
          goto LABEL_159;
        goto LABEL_331;
      }
      v171 = *(_QWORD *)(*(_QWORD *)v92 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v171 == *(_QWORD *)v92 + 48LL || !v171 || (unsigned int)*(unsigned __int8 *)(v171 - 24) - 30 > 0xA )
        goto LABEL_402;
      if ( *(_BYTE *)(v171 - 24) != 36 )
        goto LABEL_253;
      v92 += 2;
    }
    v172 = *(_QWORD *)(*(_QWORD *)v92 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v172 == *(_QWORD *)v92 + 48LL || !v172 || (unsigned int)*(unsigned __int8 *)(v172 - 24) - 30 > 0xA )
      goto LABEL_402;
    if ( *(_BYTE *)(v172 - 24) != 36 )
      goto LABEL_253;
    v92 += 2;
LABEL_331:
    v173 = *(_QWORD *)(*(_QWORD *)v92 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v173 == *(_QWORD *)v92 + 48LL || !v173 || (unsigned int)*(unsigned __int8 *)(v173 - 24) - 30 > 0xA )
      goto LABEL_402;
    if ( *(_BYTE *)(v173 - 24) != 36 )
      goto LABEL_253;
LABEL_159:
    v216 = **(_QWORD **)(v8 + 32);
    v102 = sub_D47930(v8);
    BYTE4(v232) = 1;
    v205 = v102;
    v230 = (unsigned __int64)&v233;
    v103 = *(__int64 **)(v8 + 32);
    v229 = 0;
    v231 = 8;
    LODWORD(v232) = 0;
    v195 = sub_AA4E30(*v103);
    v203 = *(__int64 **)(v8 + 32);
    v194 = *(__int64 **)(v8 + 40);
    if ( v203 != v194 )
    {
      v198 = v8;
      do
      {
        v104 = *v203;
        v105 = *(_QWORD *)(*v203 + 56);
        v210 = *v203 + 48;
        while ( v210 != v105 )
        {
          v106 = 0;
          if ( v105 )
            v106 = v105 - 24;
          if ( (unsigned __int8)sub_B46490(v106) )
          {
            v68 = 0;
            v8 = v198;
            goto LABEL_193;
          }
          if ( BYTE4(v232) )
          {
            v109 = (__int64 *)v230;
            v110 = (__int64 *)(v230 + 8LL * HIDWORD(v231));
            if ( (__int64 *)v230 != v110 )
            {
              while ( v106 != *v109 )
              {
                if ( v110 == ++v109 )
                  goto LABEL_173;
              }
            }
          }
          else
          {
            v109 = sub_C8CA60((__int64)&v229, v106);
            if ( v109 )
            {
              if ( BYTE4(v232) )
                v110 = (__int64 *)(v230 + 8LL * HIDWORD(v231));
              else
                v110 = (__int64 *)(v230 + 8LL * (unsigned int)v231);
            }
            else
            {
              if ( !BYTE4(v232) )
                goto LABEL_173;
              v109 = (__int64 *)(v230 + 8LL * HIDWORD(v231));
              v110 = v109;
            }
          }
          if ( v110 != v109 )
          {
            while ( (unsigned __int64)*v109 >= 0xFFFFFFFFFFFFFFFELL )
            {
              if ( v110 == ++v109 )
                goto LABEL_173;
            }
            if ( v110 != v109 )
            {
              v128 = *(_QWORD *)(v106 + 16);
              if ( v128 )
              {
                v129 = BYTE4(v232);
                do
                {
                  v130 = *(_QWORD *)(v128 + 24);
                  if ( !(_BYTE)v129 )
                    goto LABEL_218;
                  v131 = (_QWORD *)v230;
                  v107 = HIDWORD(v231);
                  v110 = (__int64 *)(v230 + 8LL * HIDWORD(v231));
                  if ( (__int64 *)v230 != v110 )
                  {
                    while ( v130 != *v131 )
                    {
                      if ( v110 == ++v131 )
                        goto LABEL_219;
                    }
                    goto LABEL_216;
                  }
LABEL_219:
                  if ( HIDWORD(v231) < (unsigned int)v231 )
                  {
                    v107 = (unsigned int)++HIDWORD(v231);
                    *v110 = v130;
                    v129 = BYTE4(v232);
                    v229 = (__int64 *)((char *)v229 + 1);
                  }
                  else
                  {
LABEL_218:
                    sub_C8CC70((__int64)&v229, v130, (__int64)v110, v107, v129, v108);
                    v129 = BYTE4(v232);
                  }
LABEL_216:
                  v128 = *(_QWORD *)(v128 + 8);
                }
                while ( v128 );
              }
            }
          }
LABEL_173:
          if ( v216 != v104 && *(_BYTE *)v106 == 61 )
          {
            v111 = *(_QWORD *)(v106 - 32);
            if ( (unsigned __int8)sub_B19720((__int64)a5, v104, v205) )
            {
              if ( (unsigned __int8)sub_D48480(v198, v111, v112, v113) )
              {
                v133 = sub_D30730(v111, *(_QWORD *)(v106 + 8), v195, v106, a7, a5, 0);
                v137 = v187;
                if ( !v133 )
                {
                  v138 = *(_QWORD *)(v106 + 16);
                  if ( v138 )
                  {
                    v139 = BYTE4(v232);
                    do
                    {
                      v140 = *(_QWORD *)(v138 + 24);
                      if ( !v139 )
                        goto LABEL_271;
                      v141 = (_QWORD *)v230;
                      v137 = HIDWORD(v231);
                      v134 = (__int64 *)(v230 + 8LL * HIDWORD(v231));
                      if ( (__int64 *)v230 != v134 )
                      {
                        while ( v140 != *v141 )
                        {
                          if ( v134 == ++v141 )
                            goto LABEL_272;
                        }
                        goto LABEL_250;
                      }
LABEL_272:
                      if ( HIDWORD(v231) < (unsigned int)v231 )
                      {
                        v137 = (unsigned int)++HIDWORD(v231);
                        *v134 = v140;
                        v139 = BYTE4(v232);
                        v229 = (__int64 *)((char *)v229 + 1);
                      }
                      else
                      {
LABEL_271:
                        sub_C8CC70((__int64)&v229, v140, (__int64)v134, v137, v135, v136);
                        v139 = BYTE4(v232);
                      }
LABEL_250:
                      v138 = *(_QWORD *)(v138 + 8);
                    }
                    while ( v138 );
                  }
                }
              }
            }
          }
          v105 = *(_QWORD *)(v105 + 8);
        }
        ++v203;
      }
      while ( v194 != v203 );
      v68 = 0;
      v8 = v198;
    }
    v225 = &v227;
    v226 = 0x600000000LL;
    sub_D46D90(v8, (__int64)&v225);
    v114 = v225;
    v115 = (unsigned int)v226;
    v116 = &v225[v115];
    v117 = (v115 * 8) >> 5;
    if ( v117 )
    {
      v118 = &v225[4 * v117];
      do
      {
        v119 = *((_QWORD *)*v114 + 6) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v119 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)*v114 + 48) )
        {
          v121 = 0;
        }
        else
        {
          if ( !v119 )
            goto LABEL_402;
          v120 = *(unsigned __int8 *)(v119 - 24);
          v121 = 0;
          v122 = v119 - 24;
          if ( (unsigned int)(v120 - 30) < 0xB )
            v121 = v122;
        }
        if ( BYTE4(v232) )
        {
          v123 = (_QWORD *)v230;
          v124 = HIDWORD(v231);
          v125 = (_QWORD *)(v230 + 8LL * HIDWORD(v231));
          v126 = (_QWORD *)v230;
          if ( (_QWORD *)v230 == v125 )
          {
            v174 = v114[1];
            v150 = v114 + 1;
            v151 = *((_QWORD *)v174 + 6) & 0xFFFFFFFFFFFFFFF8LL;
            if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v151 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v174 + 48) )
            {
              v152 = 0;
LABEL_281:
              v153 = &v123[v124];
              v154 = v123;
              if ( v123 != v153 )
                goto LABEL_284;
              v175 = v114[2];
              v150 = v114 + 2;
              v157 = *((_QWORD *)v175 + 6) & 0xFFFFFFFFFFFFFFF8LL;
              if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v157 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v175 + 48) )
              {
                v158 = 0;
                goto LABEL_292;
              }
LABEL_288:
              if ( !v157 )
                goto LABEL_402;
              v158 = v157 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v157 - 24) - 30 <= 0xA )
                goto LABEL_290;
              goto LABEL_337;
            }
          }
          else
          {
            do
            {
              if ( *v126 == v121 )
                goto LABEL_190;
              ++v126;
            }
            while ( v125 != v126 );
            v149 = v114[1];
            v150 = v114 + 1;
            v151 = *((_QWORD *)v149 + 6) & 0xFFFFFFFFFFFFFFF8LL;
            if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v151 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v149 + 48) )
            {
              v154 = (_QWORD *)v230;
              v153 = (_QWORD *)(v230 + 8LL * HIDWORD(v231));
              v152 = 0;
              do
              {
LABEL_284:
                if ( v152 == *v154 )
                  goto LABEL_285;
                ++v154;
              }
              while ( v153 != v154 );
              v161 = v114[2];
              v150 = v114 + 2;
              v157 = *((_QWORD *)v161 + 6) & 0xFFFFFFFFFFFFFFF8LL;
              if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v157 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v161 + 48) )
              {
                v160 = v123;
                v159 = &v123[v124];
                v158 = 0;
                do
                {
LABEL_295:
                  if ( v158 == *v160 )
                    goto LABEL_285;
                  ++v160;
                }
                while ( v159 != v160 );
                v162 = v114[3];
                v150 = v114 + 3;
                v163 = *((_QWORD *)v162 + 6) & 0xFFFFFFFFFFFFFFF8LL;
                if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v163 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v162 + 48) )
                {
                  v165 = &v123[v124];
                  v164 = 0;
LABEL_306:
                  while ( v164 != *v123 )
                  {
                    if ( v165 == ++v123 )
                      goto LABEL_309;
                  }
LABEL_285:
                  v114 = v150;
LABEL_190:
                  v68 = v116 != v114;
LABEL_191:
                  if ( v225 != &v227 )
                    _libc_free((unsigned __int64)v225);
LABEL_193:
                  if ( !BYTE4(v232) )
                    _libc_free(v230);
                  if ( v222 != (unsigned int *)v224 )
                    _libc_free((unsigned __int64)v222);
                  if ( v68 )
                  {
LABEL_97:
                    v69 = v68;
                    if ( v68 > v200 )
                      v69 = v200;
                    if ( v69 + v192 <= (unsigned int)qword_5009C08 )
                    {
                      *(_DWORD *)a3 = v69;
                      *(_BYTE *)(a3 + 6) = 0;
                      return;
                    }
                  }
                  goto LABEL_100;
                }
LABEL_300:
                if ( !v163 )
                  goto LABEL_402;
                v164 = v163 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v163 - 24) - 30 <= 0xA )
                {
LABEL_302:
                  if ( BYTE4(v232) )
                  {
                    v123 = (_QWORD *)v230;
                    v165 = (_QWORD *)(v230 + 8LL * HIDWORD(v231));
                    if ( (_QWORD *)v230 != v165 )
                      goto LABEL_306;
                  }
                  else
                  {
                    v212 = v150;
                    v166 = sub_C8CA60((__int64)&v229, v164);
                    v150 = v212;
                    if ( v166 )
                      goto LABEL_285;
                  }
                  goto LABEL_309;
                }
LABEL_316:
                v164 = 0;
                goto LABEL_302;
              }
              goto LABEL_288;
            }
          }
        }
        else
        {
          if ( sub_C8CA60((__int64)&v229, v121) )
            goto LABEL_190;
          v170 = v114[1];
          v150 = v114 + 1;
          v151 = *((_QWORD *)v170 + 6) & 0xFFFFFFFFFFFFFFF8LL;
          if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v151 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v170 + 48) )
          {
LABEL_320:
            v152 = 0;
            goto LABEL_279;
          }
        }
        if ( !v151 )
          goto LABEL_402;
        v152 = v151 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v151 - 24) - 30 > 0xA )
          goto LABEL_320;
LABEL_279:
        if ( BYTE4(v232) )
        {
          v123 = (_QWORD *)v230;
          v124 = HIDWORD(v231);
          goto LABEL_281;
        }
        v211 = v150;
        v155 = sub_C8CA60((__int64)&v229, v152);
        v150 = v211;
        if ( v155 )
          goto LABEL_285;
        v156 = v114[2];
        v150 = v114 + 2;
        v157 = *((_QWORD *)v156 + 6) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v157 != (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v156 + 48) )
          goto LABEL_288;
LABEL_337:
        v158 = 0;
LABEL_290:
        if ( !BYTE4(v232) )
        {
          v213 = v150;
          v168 = sub_C8CA60((__int64)&v229, v158);
          v150 = v213;
          if ( v168 )
            goto LABEL_285;
          v169 = v114[3];
          v150 = v114 + 3;
          v163 = *((_QWORD *)v169 + 6) & 0xFFFFFFFFFFFFFFF8LL;
          if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v163 != (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v169 + 48) )
            goto LABEL_300;
          goto LABEL_316;
        }
        v123 = (_QWORD *)v230;
        v124 = HIDWORD(v231);
LABEL_292:
        v159 = &v123[v124];
        v160 = v123;
        if ( v123 != v159 )
          goto LABEL_295;
        v186 = v114[3];
        v150 = v114 + 3;
        v163 = *((_QWORD *)v186 + 6) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v163 != (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)v186 + 48) )
          goto LABEL_300;
LABEL_309:
        v114 += 4;
      }
      while ( v114 != v118 );
    }
    v167 = (char *)v116 - (char *)v114;
    if ( (char *)v116 - (char *)v114 != 16 )
    {
      if ( v167 != 24 )
      {
        if ( v167 != 8 )
        {
          v114 = v116;
          goto LABEL_190;
        }
LABEL_367:
        v182 = *((_QWORD *)*v114 + 6) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v182 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)*v114 + 48) )
          goto LABEL_372;
        if ( v182 )
        {
          v183 = v182 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v182 - 24) - 30 > 0xA )
            v183 = 0;
          v207 = v183;
LABEL_372:
          if ( BYTE4(v232) )
          {
            for ( i = (_QWORD *)v230; (_QWORD *)(v230 + 8LL * HIDWORD(v231)) != i; ++i )
            {
              if ( *i == v207 )
                goto LABEL_190;
            }
          }
          else if ( sub_C8CA60((__int64)&v229, v207) )
          {
            goto LABEL_190;
          }
          goto LABEL_191;
        }
LABEL_402:
        BUG();
      }
      v176 = *((_QWORD *)*v114 + 6) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v176 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)*v114 + 48) )
      {
        v177 = 0;
      }
      else
      {
        if ( !v176 )
          goto LABEL_402;
        v177 = v176 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v176 - 24) - 30 >= 0xB )
          v177 = 0;
      }
      if ( BYTE4(v232) )
      {
        for ( j = (_QWORD *)v230; (_QWORD *)(v230 + 8LL * HIDWORD(v231)) != j; ++j )
        {
          if ( *j == v177 )
            goto LABEL_190;
        }
      }
      else if ( sub_C8CA60((__int64)&v229, v177) )
      {
        goto LABEL_190;
      }
      ++v114;
    }
    v179 = *((_QWORD *)*v114 + 6) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v179 == (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))((char *)*v114 + 48) )
    {
      v180 = 0;
    }
    else
    {
      if ( !v179 )
        goto LABEL_402;
      v180 = v179 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v179 - 24) - 30 >= 0xB )
        v180 = 0;
    }
    if ( BYTE4(v232) )
    {
      for ( k = (_QWORD *)v230; (_QWORD *)(v230 + 8LL * HIDWORD(v231)) != k; ++k )
      {
        if ( *k == v180 )
          goto LABEL_190;
      }
    }
    else if ( sub_C8CA60((__int64)&v229, v180) )
    {
      goto LABEL_190;
    }
    ++v114;
    goto LABEL_367;
  }
LABEL_100:
  if ( a4 )
    return;
  if ( !*(_BYTE *)(a3 + 6) )
    return;
  sub_B2EE70((__int64)&v229, *(_QWORD *)(**(_QWORD **)(v8 + 32) + 72LL), 0);
  if ( !(_BYTE)v231 )
    return;
  v70 = sub_D47930(v8);
  if ( !v70 )
    return;
  v71 = v70 + 48;
  v72 = *(_QWORD *)(v70 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v72 == v71 )
    goto LABEL_402;
  if ( !v72 )
    goto LABEL_402;
  v73 = v72 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v72 - 24) - 30 > 0xA )
    goto LABEL_402;
  if ( *(_BYTE *)(v72 - 24) != 31 )
    return;
  if ( (*(_DWORD *)(v72 - 20) & 0x7FFFFFF) != 3 )
    return;
  v74 = sub_B46E30(v73);
  if ( !v74 )
    return;
  v75 = 0;
  v76 = v74;
  while ( 1 )
  {
    v77 = sub_B46EC0(v73, v75);
    if ( !*(_BYTE *)(v8 + 84) )
      break;
    v78 = *(_QWORD **)(v8 + 64);
    v79 = &v78[*(unsigned int *)(v8 + 76)];
    if ( v78 == v79 )
      goto LABEL_257;
    while ( v77 != *v78 )
    {
      if ( v79 == ++v78 )
        goto LABEL_257;
    }
LABEL_116:
    if ( ++v75 == v76 )
      return;
  }
  if ( sub_C8CA60(v8 + 56, v77) )
    goto LABEL_116;
LABEL_257:
  v230 = 0x400000000LL;
  v229 = &v231;
  sub_D4C2F0(v8, (__int64)&v229);
  v142 = v229;
  v143 = (unsigned int)v230;
  v144 = &v229[v143];
  v145 = (v143 * 8) >> 5;
  if ( !v145 )
  {
LABEL_381:
    v185 = (char *)v144 - (char *)v142;
    if ( (char *)v144 - (char *)v142 != 16 )
    {
      if ( v185 != 24 )
      {
        if ( v185 != 8 )
          goto LABEL_384;
        goto LABEL_395;
      }
      if ( !sub_AA4F10(*v142) )
        goto LABEL_264;
      ++v142;
    }
    if ( !sub_AA4F10(*v142) )
      goto LABEL_264;
    ++v142;
LABEL_395:
    if ( !sub_AA4F10(*v142) )
      goto LABEL_264;
LABEL_384:
    v147 = (unsigned __int64)v229;
    v148 = 0;
    if ( v229 != &v231 )
      goto LABEL_265;
    goto LABEL_267;
  }
  v146 = &v229[4 * v145];
  while ( sub_AA4F10(*v142) )
  {
    if ( !sub_AA4F10(v142[1]) )
    {
      ++v142;
      break;
    }
    if ( !sub_AA4F10(v142[2]) )
    {
      v142 += 2;
      break;
    }
    if ( !sub_AA4F10(v142[3]) )
    {
      v142 += 3;
      break;
    }
    v142 += 4;
    if ( v146 == v142 )
      goto LABEL_381;
  }
LABEL_264:
  v147 = (unsigned __int64)v229;
  v148 = v144 != v142;
  if ( v229 != &v231 )
LABEL_265:
    _libc_free(v147);
  if ( !v148 )
  {
LABEL_267:
    v229 = (__int64 *)sub_F6EC60(v8, 0);
    if ( BYTE4(v229) )
    {
      if ( (unsigned int)v229 + v192 <= v200 )
        *(_DWORD *)a3 = (_DWORD)v229;
    }
  }
}
