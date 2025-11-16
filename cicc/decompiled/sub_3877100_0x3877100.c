// Function: sub_3877100
// Address: 0x3877100
//
__int64 __fastcall sub_3877100(
        __int64 *a1,
        _BYTE *a2,
        _BYTE *a3,
        __int64 a4,
        __int64 **a5,
        __int64 ***a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 *v14; // r14
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // r13
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rax
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned int v25; // r15d
  __int64 v26; // r14
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdx
  const void *v30; // r15
  signed __int64 v31; // r12
  __int64 v32; // r14
  _QWORD *v33; // rax
  _BYTE *v34; // r12
  __int64 *v35; // rsi
  __int64 v36; // r15
  __int64 **v37; // rcx
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  double v40; // xmm4_8
  double v41; // xmm5_8
  __int64 *v42; // rbx
  __int64 *v43; // r8
  int v44; // r9d
  __int64 v45; // rax
  char v46; // al
  __int64 v47; // r12
  __int64 v48; // rbx
  __int64 **v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r11
  __int64 v55; // r10
  const void *v56; // rax
  signed __int64 v57; // r10
  const void *v58; // r11
  __int64 v59; // r14
  const char *v60; // r9
  int v61; // edx
  char *v62; // rdi
  unsigned int v63; // r8d
  size_t v64; // r11
  __int64 v65; // rdx
  __int64 v66; // rbx
  int v67; // r9d
  int v68; // r8d
  __int64 v69; // r12
  __int64 *v70; // rbx
  bool v71; // al
  __int64 v72; // rdi
  __int64 v73; // rax
  _QWORD *v74; // rcx
  __int64 *v75; // rdi
  __int64 **v76; // rax
  double v77; // xmm4_8
  double v78; // xmm5_8
  __int64 ***v79; // r12
  __int64 *v80; // rax
  double v81; // xmm4_8
  double v82; // xmm5_8
  __int64 ***v83; // rax
  __int64 v84; // rcx
  int v85; // r8d
  int v86; // r9d
  __int64 *v87; // r13
  __int64 v88; // rax
  __int64 v89; // r12
  unsigned __int64 *v91; // r9
  __int64 v92; // rax
  unsigned __int64 v93; // rbx
  __int64 v94; // rax
  __int64 v95; // rax
  int v96; // r8d
  size_t v97; // r9
  __int64 *v98; // r11
  __int64 v99; // rax
  __int64 v100; // rax
  double v101; // xmm4_8
  double v102; // xmm5_8
  __int64 v103; // rdx
  int v104; // eax
  __int64 v105; // rcx
  __int64 v106; // rsi
  int v107; // edi
  unsigned int v108; // edx
  __int64 *v109; // rax
  __int64 v110; // r8
  __int64 v111; // r12
  __int64 *v112; // r15
  __int64 v113; // rbx
  __int64 v114; // rax
  __int64 v115; // rbx
  __int64 **v116; // rbx
  __int64 v117; // rdi
  unsigned __int64 v118; // rax
  __int64 v119; // r15
  __int64 *v120; // rbx
  __int64 v121; // rax
  _QWORD *v122; // rax
  int v123; // r8d
  _QWORD *v124; // r12
  __int64 **v125; // rax
  __int64 *v126; // rax
  __int64 *v127; // rax
  int v128; // r8d
  __int64 *v129; // r11
  __int64 *v130; // rcx
  __int64 *v131; // rax
  __int64 v132; // rdx
  __int64 *v133; // rax
  __int64 v134; // rax
  unsigned __int64 *v135; // rbx
  __int64 v136; // rax
  unsigned __int64 v137; // rcx
  __int64 *v138; // rax
  _QWORD *v139; // rsi
  _QWORD *v140; // rax
  _QWORD *v141; // rax
  __int64 ****v142; // rdx
  char v143; // dl
  __int64 v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // rsi
  int v147; // edi
  unsigned int v148; // edx
  __int64 *v149; // rax
  __int64 v150; // r8
  __int64 v151; // rbx
  __int64 v152; // rdi
  unsigned __int64 v153; // rax
  __int64 v154; // rdx
  int v155; // eax
  _QWORD *v156; // rdi
  __int64 v157; // rax
  __int64 v158; // rsi
  __int64 v159; // rbx
  __int64 v160; // rax
  int v161; // eax
  int v162; // r9d
  __int64 v163; // rdx
  int v164; // ebx
  __int64 v165; // r12
  __int64 v166; // rdx
  __int64 v167; // rdx
  int v168; // eax
  int v169; // r9d
  __int64 **v170; // rax
  __int64 *v171; // rax
  __int64 v172; // rax
  const char *src; // [rsp+20h] [rbp-210h]
  void *srca; // [rsp+20h] [rbp-210h]
  __int64 *n; // [rsp+28h] [rbp-208h]
  size_t na; // [rsp+28h] [rbp-208h]
  size_t nb; // [rsp+28h] [rbp-208h]
  size_t nc; // [rsp+28h] [rbp-208h]
  __int64 v180; // [rsp+38h] [rbp-1F8h]
  unsigned int v181; // [rsp+50h] [rbp-1E0h]
  unsigned int v182; // [rsp+50h] [rbp-1E0h]
  const void *v183; // [rsp+50h] [rbp-1E0h]
  int v187; // [rsp+70h] [rbp-1C0h]
  unsigned __int64 *v188; // [rsp+70h] [rbp-1C0h]
  size_t v189; // [rsp+70h] [rbp-1C0h]
  __int64 *v190; // [rsp+70h] [rbp-1C0h]
  int v191; // [rsp+78h] [rbp-1B8h]
  unsigned __int8 dest; // [rsp+80h] [rbp-1B0h]
  unsigned __int64 desta; // [rsp+80h] [rbp-1B0h]
  int destc; // [rsp+80h] [rbp-1B0h]
  __int64 *destb; // [rsp+80h] [rbp-1B0h]
  __int64 v196; // [rsp+88h] [rbp-1A8h]
  __int64 v197; // [rsp+88h] [rbp-1A8h]
  __int64 v198; // [rsp+90h] [rbp-1A0h]
  unsigned int v199; // [rsp+98h] [rbp-198h]
  char v200; // [rsp+98h] [rbp-198h]
  unsigned int v201; // [rsp+98h] [rbp-198h]
  __int64 *v202; // [rsp+98h] [rbp-198h]
  __int64 **v203; // [rsp+98h] [rbp-198h]
  unsigned int v204; // [rsp+98h] [rbp-198h]
  __int64 v205; // [rsp+98h] [rbp-198h]
  __int64 v206; // [rsp+A8h] [rbp-188h] BYREF
  const char *v207; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v208; // [rsp+B8h] [rbp-178h]
  _BYTE v209[32]; // [rsp+C0h] [rbp-170h] BYREF
  __int64 **v210; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v211; // [rsp+E8h] [rbp-148h]
  _BYTE v212[32]; // [rsp+F0h] [rbp-140h] BYREF
  void *v213; // [rsp+110h] [rbp-120h] BYREF
  __int64 v214; // [rsp+118h] [rbp-118h]
  _BYTE v215[64]; // [rsp+120h] [rbp-110h] BYREF
  const char *v216; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v217; // [rsp+168h] [rbp-C8h]
  _WORD v218[32]; // [rsp+170h] [rbp-C0h] BYREF
  void *v219; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v220; // [rsp+1B8h] [rbp-78h]
  _BYTE v221[112]; // [rsp+1C0h] [rbp-70h] BYREF

  v14 = a1;
  v180 = *(_QWORD *)(a4 + 24);
  v210 = (__int64 **)v212;
  v211 = 0x400000000LL;
  v213 = v215;
  v214 = 0x800000000LL;
  sub_145C5B0((__int64)&v213, a2, a3);
  v220 = 0x800000000LL;
  v17 = *a1;
  v219 = v221;
  v187 = v214;
  if ( !(_DWORD)v214 )
    goto LABEL_20;
  v199 = 0;
  do
  {
    v18 = *((_QWORD *)v213 + v199);
    if ( *(_WORD *)(v18 + 24) == 7 )
    {
      while ( 1 )
      {
        v19 = **(_QWORD **)(v18 + 32);
        if ( sub_14560B0(v19) )
          goto LABEL_16;
        v20 = sub_145CF80(v17, (__int64)a5, 0, 0);
        v23 = *(_QWORD *)(v18 + 48);
        v196 = v20;
        v24 = *(_QWORD *)(v18 + 40);
        v25 = *(_WORD *)(v18 + 26) & 1;
        if ( v24 != 2 )
          break;
        v26 = *(_QWORD *)(*(_QWORD *)(v18 + 32) + 8LL);
LABEL_7:
        v27 = sub_14799E0(v17, v196, v26, v23, v25);
        v28 = (unsigned int)v220;
        if ( (unsigned int)v220 >= HIDWORD(v220) )
        {
          sub_16CD150((__int64)&v219, v221, 0, 8, v15, v16);
          v28 = (unsigned int)v220;
        }
        *((_QWORD *)v219 + v28) = v27;
        LODWORD(v220) = v220 + 1;
        if ( *(_WORD *)(v19 + 24) == 4 )
        {
          *((_QWORD *)v213 + v199) = v196;
          v29 = (unsigned int)v214;
          v30 = *(const void **)(v19 + 32);
          v31 = 8LL * *(_QWORD *)(v19 + 40);
          v32 = v31 >> 3;
          if ( v31 >> 3 > HIDWORD(v214) - (unsigned __int64)(unsigned int)v214 )
          {
            sub_16CD150((__int64)&v213, v215, v32 + (unsigned int)v214, 8, v15, v16);
            v29 = (unsigned int)v214;
          }
          v33 = v213;
          if ( v31 )
          {
            memcpy((char *)v213 + 8 * v29, v30, v31);
            LODWORD(v29) = v214;
            v33 = v213;
          }
          v187 += *(_DWORD *)(v19 + 40);
          LODWORD(v214) = v32 + v29;
        }
        else
        {
          *((_QWORD *)v213 + v199) = v19;
          v33 = v213;
        }
        v18 = v33[v199];
        if ( *(_WORD *)(v18 + 24) != 7 )
          goto LABEL_16;
      }
      v54 = *(_QWORD *)(v18 + 32);
      v55 = 8 * v24;
      v56 = (const void *)(v54 + 8 * v24);
      v57 = v55 - 8;
      v207 = v209;
      v58 = (const void *)(v54 + 8);
      v208 = 0x300000000LL;
      v59 = v57 >> 3;
      if ( (unsigned __int64)v57 > 0x18 )
      {
        srca = (void *)v57;
        nb = (size_t)v58;
        v183 = v56;
        sub_16CD150((__int64)&v207, v209, v57 >> 3, 8, v21, v22);
        v60 = v207;
        v61 = v208;
        v56 = v183;
        v58 = (const void *)nb;
        v57 = (signed __int64)srca;
        v62 = (char *)&v207[8 * (unsigned int)v208];
      }
      else
      {
        v60 = v209;
        v61 = 0;
        v62 = v209;
      }
      if ( v56 != v58 )
      {
        memcpy(v62, v58, v57);
        v60 = v207;
        v61 = v208;
      }
      LODWORD(v208) = v59 + v61;
      v63 = v59 + v61;
      v216 = (const char *)v218;
      v64 = 8LL * (unsigned int)(v59 + v61);
      v217 = 0x400000000LL;
      if ( (unsigned int)(v59 + v61) > 4uLL )
      {
        src = v60;
        na = 8LL * v63;
        v181 = v59 + v61;
        sub_16CD150((__int64)&v216, v218, v63, 8, v63, (int)v60);
        v63 = v181;
        v64 = na;
        v60 = src;
        v75 = (__int64 *)&v216[8 * (unsigned int)v217];
      }
      else
      {
        if ( !v64 )
        {
LABEL_47:
          LODWORD(v217) = v64 + v63;
          v26 = sub_14785F0(v17, (__int64 **)&v216, v23, 0);
          if ( v216 != (const char *)v218 )
            _libc_free((unsigned __int64)v216);
          if ( v207 != v209 )
            _libc_free((unsigned __int64)v207);
          goto LABEL_7;
        }
        v75 = (__int64 *)v218;
      }
      v182 = v63;
      memcpy(v75, v60, v64);
      LODWORD(v64) = v217;
      v63 = v182;
      goto LABEL_47;
    }
LABEL_16:
    ++v199;
  }
  while ( v199 != v187 );
  v14 = a1;
  v34 = v219;
  if ( (_DWORD)v220 )
  {
    v159 = (unsigned int)v220;
    v160 = (unsigned int)v214;
    if ( (unsigned int)v220 > HIDWORD(v214) - (unsigned __int64)(unsigned int)v214 )
    {
      sub_16CD150((__int64)&v213, v215, (unsigned int)v220 + (unsigned __int64)(unsigned int)v214, 8, v15, v16);
      v160 = (unsigned int)v214;
    }
    memcpy((char *)v213 + 8 * v160, v34, 8 * v159);
    LODWORD(v214) = v159 + v214;
    sub_3870D00((__int64)&v213, (__int64)a5, v17, a7, a8);
    v34 = v219;
  }
  if ( v34 != v221 )
    _libc_free((unsigned __int64)v34);
LABEL_20:
  v35 = (__int64 *)a4;
  v200 = 0;
  v36 = v180;
  v198 = sub_15A9650(v14[1], a4);
  while ( 2 )
  {
    v216 = (const char *)v218;
    v217 = 0x800000000LL;
    v38 = *(unsigned __int8 *)(v36 + 8);
    if ( (unsigned __int8)v38 > 0xFu || (v65 = 35454, !_bittest64(&v65, v38)) )
    {
      v39 = (unsigned int)(v38 - 13);
      if ( (unsigned int)v39 > 1 && (_DWORD)v38 != 16 )
      {
LABEL_64:
        v42 = (__int64 *)sub_15A06D0(a5, (__int64)v35, v39, (__int64)v37);
        v45 = (unsigned int)v211;
        if ( (unsigned int)v211 < HIDWORD(v211) )
          goto LABEL_27;
        goto LABEL_65;
      }
      v35 = 0;
      if ( sub_16435F0(v36, 0) )
        goto LABEL_53;
LABEL_25:
      if ( !(_DWORD)v217 )
        goto LABEL_64;
      goto LABEL_26;
    }
LABEL_53:
    v35 = (__int64 *)v198;
    v66 = sub_145D050(*v14, v198, v36);
    if ( sub_14560B0(v66) )
      goto LABEL_25;
    v68 = (int)v213;
    v219 = v221;
    v220 = 0x800000000LL;
    n = (__int64 *)((char *)v213 + 8 * (unsigned int)v214);
    if ( v213 != n )
    {
      v69 = v66;
      v70 = (__int64 *)v213;
      do
      {
        while ( 1 )
        {
          v72 = *v14;
          v206 = *v70;
          v73 = sub_145CF80(v72, (__int64)a5, 0, 0);
          v74 = (_QWORD *)*v14;
          v207 = (const char *)v73;
          v35 = &v206;
          dest = sub_3871230(&v206, (__int64 *)&v207, v69, v74, a7, a8);
          if ( dest )
            break;
          ++v70;
          sub_1458920((__int64)&v219, &v206);
          if ( n == v70 )
            goto LABEL_61;
        }
        sub_1458920((__int64)&v216, &v206);
        v71 = sub_14560B0((__int64)v207);
        v39 = dest;
        v200 = v71;
        if ( !v71 )
        {
          v35 = (__int64 *)&v207;
          v200 = dest;
          sub_1458920((__int64)&v219, &v207);
        }
        ++v70;
      }
      while ( n != v70 );
    }
LABEL_61:
    if ( (_DWORD)v217 )
    {
      v163 = (unsigned int)v220;
      v164 = v220;
      if ( (unsigned int)v220 <= (unsigned __int64)(unsigned int)v214 )
      {
        if ( (_DWORD)v220 )
          memmove(v213, v219, 8LL * (unsigned int)v220);
      }
      else
      {
        if ( (unsigned int)v220 > (unsigned __int64)HIDWORD(v214) )
        {
          v165 = 0;
          LODWORD(v214) = 0;
          sub_16CD150((__int64)&v213, v215, (unsigned int)v220, 8, v68, v67);
          v163 = (unsigned int)v220;
        }
        else
        {
          v165 = 8LL * (unsigned int)v214;
          if ( (_DWORD)v214 )
          {
            memmove(v213, v219, 8LL * (unsigned int)v214);
            v163 = (unsigned int)v220;
          }
        }
        v166 = 8 * v163;
        if ( (char *)v219 + v165 != (char *)v219 + v166 )
          memcpy((char *)v213 + v165, (char *)v219 + v165, v166 - v165);
      }
      v167 = *v14;
      v35 = (__int64 *)a5;
      LODWORD(v214) = v164;
      sub_3870D00((__int64)&v213, (__int64)a5, v167, a7, a8);
    }
    if ( v219 == v221 )
      goto LABEL_25;
    _libc_free((unsigned __int64)v219);
    if ( !(_DWORD)v217 )
      goto LABEL_64;
LABEL_26:
    v35 = sub_147DD40(*v14, (__int64 *)&v216, 0, 0, a7, a8);
    v42 = (__int64 *)sub_38761C0(
                       v14,
                       (__int64)v35,
                       a5,
                       (__m128)a7,
                       *(double *)a8.m128i_i64,
                       a9,
                       a10,
                       v40,
                       v41,
                       a13,
                       a14);
    v45 = (unsigned int)v211;
    if ( (unsigned int)v211 >= HIDWORD(v211) )
    {
LABEL_65:
      v35 = (__int64 *)v212;
      sub_16CD150((__int64)&v210, v212, 0, 8, (int)v43, v44);
      v45 = (unsigned int)v211;
    }
LABEL_27:
    v210[v45] = v42;
    LODWORD(v211) = v211 + 1;
    v46 = *(_BYTE *)(v36 + 8);
    if ( v46 != 13 )
    {
LABEL_38:
      if ( v46 != 14 )
        goto LABEL_69;
      v36 = *(_QWORD *)(v36 + 24);
      if ( v216 != (const char *)v218 )
        _libc_free((unsigned __int64)v216);
      continue;
    }
    break;
  }
  v47 = v36;
  while ( *(_DWORD *)(v47 + 12) )
  {
    v44 = v214;
    if ( !(_DWORD)v214 )
      break;
    v48 = *(_QWORD *)v213;
    if ( *(_WORD *)(*(_QWORD *)v213 + 24LL)
      || (unsigned __int64)sub_1456C90(*v14, **(_QWORD **)(v48 + 32)) > 0x40
      || ((v91 = (unsigned __int64 *)sub_15A9930(v14[1], v47),
           v92 = *(_QWORD *)(v48 + 32),
           *(_DWORD *)(v92 + 32) <= 0x40u)
        ? (v93 = *(_QWORD *)(v92 + 24))
        : (v93 = **(_QWORD **)(v92 + 24)),
          *v91 <= v93) )
    {
      v35 = 0;
      v47 = sub_1643D80(v47, 0);
      v49 = (__int64 **)sub_1643350(*a5);
      v52 = (__int64 *)sub_15A06D0(v49, 0, v50, v51);
      v53 = (unsigned int)v211;
      if ( (unsigned int)v211 >= HIDWORD(v211) )
      {
        v35 = (__int64 *)v212;
        destb = v52;
        sub_16CD150((__int64)&v210, v212, 0, 8, (int)v43, v44);
        v53 = (unsigned int)v211;
        v52 = destb;
      }
      v37 = v210;
      v210[v53] = v52;
      LODWORD(v211) = v211 + 1;
    }
    else
    {
      v188 = v91;
      v201 = sub_15A8020((__int64)v91, v93);
      v94 = sub_1643350(*a5);
      desta = v201;
      v95 = sub_159C470(v94, v201, 0);
      v97 = (size_t)v188;
      v98 = (__int64 *)v95;
      v99 = (unsigned int)v211;
      if ( (unsigned int)v211 >= HIDWORD(v211) )
      {
        nc = (size_t)v188;
        v190 = v98;
        sub_16CD150((__int64)&v210, v212, 0, 8, v96, v97);
        v99 = (unsigned int)v211;
        v97 = nc;
        v98 = v190;
      }
      v189 = v97;
      v210[v99] = v98;
      LODWORD(v211) = v211 + 1;
      v35 = (__int64 *)a5;
      v47 = sub_1643D80(v47, v201);
      v202 = (__int64 *)v213;
      v100 = sub_145CF80(*v14, (__int64)a5, v93 - *(_QWORD *)(v189 + 8 * desta + 16), 0);
      v43 = v202;
      v200 = 1;
      *v43 = v100;
    }
    v46 = *(_BYTE *)(v47 + 8);
    if ( v46 != 13 )
    {
      v36 = v47;
      goto LABEL_38;
    }
  }
LABEL_69:
  if ( v216 != (const char *)v218 )
    _libc_free((unsigned __int64)v216);
  if ( v200 )
  {
    sub_38701C0((__int64 **)&v219, v14 + 33, (__int64)v14, (__int64)v37, (int)v43, v44);
    v103 = *(_QWORD *)(*v14 + 64);
    v104 = *(_DWORD *)(v103 + 24);
    if ( !v104 )
      goto LABEL_101;
    while ( 1 )
    {
      v105 = v14[34];
      v106 = *(_QWORD *)(v103 + 8);
      v107 = v104 - 1;
      v108 = (v104 - 1) & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
      v109 = (__int64 *)(v106 + 16LL * v108);
      v110 = *v109;
      if ( *v109 != v105 )
      {
        v161 = 1;
        while ( v110 != -8 )
        {
          v162 = v161 + 1;
          v108 = v107 & (v161 + v108);
          v109 = (__int64 *)(v106 + 16LL * v108);
          v110 = *v109;
          if ( *v109 == v105 )
            goto LABEL_89;
          v161 = v162;
        }
LABEL_101:
        if ( (__int64 **)a4 != *a6 )
          a6 = sub_38744E0(
                 v14,
                 (__int64)a6,
                 (__int64 **)a4,
                 (__m128)a7,
                 *(double *)a8.m128i_i64,
                 a9,
                 a10,
                 v101,
                 v102,
                 a13,
                 a14);
        v119 = (unsigned int)v211;
        v209[1] = 1;
        v207 = "scevgep";
        v209[0] = 3;
        v120 = (__int64 *)v210;
        if ( *((_BYTE *)a6 + 16) > 0x10u )
        {
LABEL_108:
          v218[0] = 257;
          if ( !v180 )
          {
            v170 = *a6;
            if ( *((_BYTE *)*a6 + 8) == 16 )
              v170 = (__int64 **)*v170[2];
            v180 = (__int64)v170[3];
          }
          v204 = v211 + 1;
          v122 = sub_1648A60(72, (int)v211 + 1);
          v123 = v204;
          v124 = v122;
          if ( v122 )
          {
            v197 = (__int64)v122;
            v205 = (__int64)&v122[-3 * v204];
            v125 = *a6;
            if ( *((_BYTE *)*a6 + 8) == 16 )
              v125 = (__int64 **)*v125[2];
            v191 = v123;
            destc = *((_DWORD *)v125 + 2) >> 8;
            v126 = (__int64 *)sub_15F9F50(v180, (__int64)v120, v119);
            v127 = (__int64 *)sub_1646BA0(v126, destc);
            v128 = v191;
            v129 = v127;
            if ( *((_BYTE *)*a6 + 8) == 16 )
            {
              v171 = sub_16463B0(v127, (unsigned int)(*a6)[4]);
              v128 = v191;
              v129 = v171;
            }
            else
            {
              v130 = &v120[v119];
              if ( v120 != v130 )
              {
                v131 = v120;
                while ( 1 )
                {
                  v132 = *(_QWORD *)*v131;
                  if ( *(_BYTE *)(v132 + 8) == 16 )
                    break;
                  if ( v130 == ++v131 )
                    goto LABEL_118;
                }
                v133 = sub_16463B0(v129, *(_QWORD *)(v132 + 32));
                v128 = v191;
                v129 = v133;
              }
            }
LABEL_118:
            sub_15F1EA0((__int64)v124, (__int64)v129, 32, v205, v128, 0);
            v124[7] = v180;
            v124[8] = sub_15F9F50(v180, (__int64)v120, v119);
            sub_15F9CE0((__int64)v124, (__int64)a6, v120, v119, (__int64)&v216);
          }
          else
          {
            v197 = 0;
          }
          v134 = v14[34];
          if ( v134 )
          {
            v135 = (unsigned __int64 *)v14[35];
            sub_157E9D0(v134 + 40, (__int64)v124);
            v136 = v124[3];
            v137 = *v135;
            v124[4] = v135;
            v137 &= 0xFFFFFFFFFFFFFFF8LL;
            v124[3] = v137 | v136 & 7;
            *(_QWORD *)(v137 + 8) = v124 + 3;
            *v135 = *v135 & 7 | (unsigned __int64)(v124 + 3);
          }
          sub_164B780(v197, (__int64 *)&v207);
          sub_12A86E0(v14 + 33, (__int64)v124);
        }
        else
        {
          if ( (_DWORD)v211 )
          {
            v121 = 0;
            while ( *((_BYTE *)v210[v121] + 16) <= 0x10u )
            {
              if ( (unsigned int)v211 == ++v121 )
                goto LABEL_185;
            }
            goto LABEL_108;
          }
LABEL_185:
          BYTE4(v216) = 0;
          v124 = (_QWORD *)sub_15A2E80(v180, (__int64)a6, v210, (unsigned int)v211, 0, (__int64)&v216, 0);
          v172 = sub_14DBA30((__int64)v124, v14[41], 0);
          if ( v172 )
            v124 = (_QWORD *)v172;
        }
        v216 = (const char *)sub_145DC80(*v14, (__int64)v124);
        sub_1458920((__int64)&v213, &v216);
        sub_38740E0((__int64)v14, (__int64)v124);
        sub_3870260((__int64)&v219);
        v138 = sub_147DD40(*v14, (__int64 *)&v213, 0, 0, a7, a8);
        v89 = sub_3875200(v14, (__int64)v138, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
        goto LABEL_75;
      }
LABEL_89:
      v111 = v109[1];
      if ( !v111 || !sub_13FC1A0(v109[1], (__int64)a6) )
        goto LABEL_101;
      v112 = (__int64 *)v210;
      v113 = 8LL * (unsigned int)v211;
      v203 = &v210[(unsigned __int64)v113 / 8];
      v114 = v113 >> 3;
      v115 = v113 >> 5;
      if ( !v115 )
        goto LABEL_124;
      v116 = &v210[4 * v115];
      do
      {
        if ( !sub_13FC1A0(v111, *v112) )
          goto LABEL_98;
        if ( !sub_13FC1A0(v111, v112[1]) )
        {
          ++v112;
          goto LABEL_98;
        }
        if ( !sub_13FC1A0(v111, v112[2]) )
        {
          v112 += 2;
          goto LABEL_98;
        }
        if ( !sub_13FC1A0(v111, v112[3]) )
        {
          v112 += 3;
          goto LABEL_98;
        }
        v112 += 4;
      }
      while ( v112 != (__int64 *)v116 );
      v114 = ((char *)v203 - (char *)v112) >> 3;
LABEL_124:
      if ( v114 != 2 )
      {
        if ( v114 != 3 )
        {
          if ( v114 == 1 && !sub_13FC1A0(v111, *v112) )
            goto LABEL_98;
          goto LABEL_99;
        }
        if ( !sub_13FC1A0(v111, *v112) )
          goto LABEL_98;
        ++v112;
      }
      if ( !sub_13FC1A0(v111, *v112) || (v158 = v112[1], ++v112, !sub_13FC1A0(v111, v158)) )
      {
LABEL_98:
        if ( v203 != (__int64 **)v112 )
          goto LABEL_101;
      }
LABEL_99:
      v117 = sub_13FC520(v111);
      if ( v117 )
      {
        v118 = sub_157EBA0(v117);
        sub_17050D0(v14 + 33, v118);
        v103 = *(_QWORD *)(*v14 + 64);
        v104 = *(_DWORD *)(v103 + 24);
        if ( v104 )
          continue;
      }
      goto LABEL_101;
    }
  }
  v76 = (__int64 **)sub_16471D0(*a5, *(_DWORD *)(a4 + 8) >> 8);
  v79 = sub_38744E0(v14, (__int64)a6, v76, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v77, v78, a13, a14);
  v80 = sub_147DD40(*v14, (__int64 *)&v213, 0, 0, a7, a8);
  v83 = sub_38761C0(v14, (__int64)v80, a5, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v81, v82, a13, a14);
  v87 = (__int64 *)v83;
  if ( *((_BYTE *)v79 + 16) <= 0x10u && *((_BYTE *)v83 + 16) <= 0x10u )
  {
    v88 = sub_1643330(*a5);
    BYTE4(v219) = 0;
    v216 = (const char *)v87;
    v89 = sub_15A2E80(v88, (__int64)v79, (__int64 **)&v216, 1u, 0, (__int64)&v219, 0);
    goto LABEL_75;
  }
  v139 = *(_QWORD **)(v14[34] + 48);
  v140 = (_QWORD *)v14[35];
  if ( v140 != v139 )
  {
    v84 = 6;
    v141 = (_QWORD *)(*v140 & 0xFFFFFFFFFFFFFFF8LL);
    while ( 1 )
    {
      if ( !v141 )
        BUG();
      v143 = *((_BYTE *)v141 - 8);
      if ( v143 != 78 )
        break;
      v144 = *(v141 - 6);
      if ( *(_BYTE *)(v144 + 16) || (*(_BYTE *)(v144 + 33) & 0x20) == 0 )
        goto LABEL_135;
      v84 = ((unsigned int)(*(_DWORD *)(v144 + 36) - 35) < 4) + (unsigned int)v84;
      if ( v139 == v141 )
        goto LABEL_147;
LABEL_136:
      v141 = (_QWORD *)(*v141 & 0xFFFFFFFFFFFFFFF8LL);
      v84 = (unsigned int)(v84 - 1);
      if ( !(_DWORD)v84 )
        goto LABEL_147;
    }
    if ( v143 == 56 )
    {
      if ( (*((_BYTE *)v141 - 1) & 0x40) != 0 )
      {
        v142 = (__int64 ****)*(v141 - 4);
        if ( v79 == *v142 )
          goto LABEL_142;
      }
      else
      {
        v142 = (__int64 ****)&v141[-3 * (*((_DWORD *)v141 - 1) & 0xFFFFFFF) - 3];
        if ( v79 == *v142 )
        {
LABEL_142:
          if ( v87 == (__int64 *)v142[3] )
          {
            v89 = (__int64)(v141 - 3);
            goto LABEL_75;
          }
        }
      }
    }
LABEL_135:
    if ( v139 == v141 )
      goto LABEL_147;
    goto LABEL_136;
  }
LABEL_147:
  sub_38701C0((__int64 **)&v219, v14 + 33, (__int64)v14, v84, v85, v86);
  while ( 1 )
  {
    v154 = *(_QWORD *)(*v14 + 64);
    v155 = *(_DWORD *)(v154 + 24);
    if ( !v155 )
      break;
    v145 = v14[34];
    v146 = *(_QWORD *)(v154 + 8);
    v147 = v155 - 1;
    v148 = (v155 - 1) & (((unsigned int)v145 >> 9) ^ ((unsigned int)v145 >> 4));
    v149 = (__int64 *)(v146 + 16LL * v148);
    v150 = *v149;
    if ( *v149 != v145 )
    {
      v168 = 1;
      while ( v150 != -8 )
      {
        v169 = v168 + 1;
        v148 = v147 & (v168 + v148);
        v149 = (__int64 *)(v146 + 16LL * v148);
        v150 = *v149;
        if ( *v149 == v145 )
          goto LABEL_149;
        v168 = v169;
      }
      break;
    }
LABEL_149:
    v151 = v149[1];
    if ( !v151 )
      break;
    if ( !sub_13FC1A0(v149[1], (__int64)v79) )
      break;
    if ( !sub_13FC1A0(v151, (__int64)v87) )
      break;
    v152 = sub_13FC520(v151);
    if ( !v152 )
      break;
    v153 = sub_157EBA0(v152);
    sub_17050D0(v14 + 33, v153);
  }
  v156 = (_QWORD *)v14[36];
  v216 = "uglygep";
  v218[0] = 259;
  v157 = sub_1643330(v156);
  v89 = (__int64)sub_3871660((__int64)(v14 + 33), v157, v79, (__int64)v87, (__int64 *)&v216);
  sub_38740E0((__int64)v14, v89);
  sub_3870260((__int64)&v219);
LABEL_75:
  if ( v213 != v215 )
    _libc_free((unsigned __int64)v213);
  if ( v210 != (__int64 **)v212 )
    _libc_free((unsigned __int64)v210);
  return v89;
}
