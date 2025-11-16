// Function: sub_17B09A0
// Address: 0x17b09a0
//
__int64 __fastcall sub_17B09A0(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rdx
  __m128 v15; // xmm0
  __m128i v16; // xmm1
  __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // r12
  _QWORD *v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  char v27; // al
  __int64 v28; // rbx
  int v29; // r8d
  int v30; // r9d
  _QWORD *v31; // r13
  __int64 v32; // rbx
  __int64 v33; // r12
  _QWORD *v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // r12
  __int64 v38; // rbx
  __int64 v39; // r12
  _QWORD *v40; // rax
  double v41; // xmm4_8
  double v42; // xmm5_8
  __int64 v43; // rsi
  __int64 v44; // rcx
  unsigned __int8 v45; // al
  __int64 v46; // rdx
  __int64 v47; // rbx
  unsigned int v48; // r12d
  __int64 v49; // r12
  __int64 **v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // r8
  __m128i *v54; // rdi
  __int64 *v55; // rdi
  __int64 i; // rbx
  __int64 *v57; // rax
  __int64 v58; // rax
  __int64 *v59; // rcx
  __int64 v60; // rax
  _QWORD *v61; // rdx
  __int64 v62; // rdi
  __int64 ***v63; // rax
  __int64 v64; // rdx
  __int64 ***v65; // r12
  _QWORD *v66; // r13
  __int64 v67; // rdi
  __int64 **v68; // r12
  __int64 v69; // rax
  __int64 v70; // rcx
  unsigned __int32 v71; // r15d
  __m128i *v72; // rdi
  _QWORD *v73; // r15
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rax
  int v77; // eax
  _QWORD *v78; // rax
  __m128i *v79; // rdi
  _BYTE *v80; // rcx
  _BYTE *v81; // rax
  __int64 v82; // rax
  unsigned int v83; // r13d
  _QWORD *v84; // rax
  __int64 *v85; // rax
  __int64 **v86; // rax
  _QWORD *v87; // r14
  __int64 v88; // rax
  __int64 v89; // r12
  _QWORD *v90; // rax
  char v91; // al
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 v94; // r10
  __int64 v95; // rax
  __int64 v96; // rcx
  _BYTE *v97; // rax
  __int64 v98; // r15
  __int64 v99; // rdx
  __int64 *v100; // r12
  __int64 v101; // rax
  _QWORD *v102; // rax
  _QWORD *v103; // r15
  __int64 v104; // rbx
  unsigned int v105; // r12d
  __int64 v106; // rdi
  __int64 v107; // rax
  __int64 v108; // r12
  __int64 *v109; // r13
  _QWORD *v110; // rax
  __int64 v111; // rax
  __int64 *v112; // rbx
  _QWORD *v113; // rax
  _QWORD *v114; // r12
  __int64 v115; // rbx
  __int64 v116; // r12
  _QWORD *v117; // rax
  double v118; // xmm4_8
  double v119; // xmm5_8
  __int64 v120; // rax
  __int64 v121; // rbx
  __int64 v122; // r13
  _QWORD *v123; // rax
  unsigned int v124; // ebx
  int v125; // r12d
  int v126; // eax
  __int64 v127; // rax
  size_t v128; // r12
  __m128i *v129; // rdi
  __int64 v130; // r12
  __int64 *v131; // rbx
  __int64 v132; // rbx
  __int64 *v133; // r13
  _QWORD *v134; // rax
  __int64 v135; // rax
  __int64 *v136; // rbx
  __int64 v137; // rax
  _QWORD *v138; // rax
  __int64 v139; // rax
  __int64 *v140; // r13
  int v141; // r8d
  int v142; // r9d
  __int64 v143; // r8
  _QWORD *v144; // rax
  __int64 v145; // rax
  __int64 *v146; // r13
  _QWORD *v147; // rax
  __int64 *v148; // rax
  __int64 v149; // rax
  unsigned __int64 *v150; // r15
  __int64 v151; // rax
  unsigned __int64 v152; // rcx
  __int64 v153; // rdx
  __int64 v154; // rsi
  __int64 v155; // rsi
  unsigned __int8 *v156; // rsi
  __int64 v157; // rdx
  __int64 v158; // [rsp+18h] [rbp-1F8h]
  __int64 v159; // [rsp+20h] [rbp-1F0h]
  __int32 v160; // [rsp+28h] [rbp-1E8h]
  size_t n; // [rsp+30h] [rbp-1E0h]
  __int64 na; // [rsp+30h] [rbp-1E0h]
  size_t nb; // [rsp+30h] [rbp-1E0h]
  int v164; // [rsp+38h] [rbp-1D8h]
  __int64 v165; // [rsp+38h] [rbp-1D8h]
  _QWORD *v166; // [rsp+38h] [rbp-1D8h]
  __int64 v167; // [rsp+38h] [rbp-1D8h]
  __int64 v168; // [rsp+40h] [rbp-1D0h]
  __int64 *v169; // [rsp+40h] [rbp-1D0h]
  __int64 v170; // [rsp+40h] [rbp-1D0h]
  unsigned __int64 v171; // [rsp+40h] [rbp-1D0h]
  __int64 *v172; // [rsp+40h] [rbp-1D0h]
  _BYTE *s; // [rsp+48h] [rbp-1C8h]
  int sd; // [rsp+48h] [rbp-1C8h]
  unsigned int sa; // [rsp+48h] [rbp-1C8h]
  void *sb; // [rsp+48h] [rbp-1C8h]
  void *sc; // [rsp+48h] [rbp-1C8h]
  __int64 v178; // [rsp+50h] [rbp-1C0h]
  __int64 v179; // [rsp+50h] [rbp-1C0h]
  __int64 v180; // [rsp+50h] [rbp-1C0h]
  _QWORD *v181; // [rsp+58h] [rbp-1B8h]
  __int64 v182; // [rsp+58h] [rbp-1B8h]
  __int64 v183; // [rsp+58h] [rbp-1B8h]
  __int64 v184; // [rsp+58h] [rbp-1B8h]
  _QWORD *v185; // [rsp+58h] [rbp-1B8h]
  __int64 v186; // [rsp+60h] [rbp-1B0h] BYREF
  unsigned int v187; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v188; // [rsp+70h] [rbp-1A0h] BYREF
  unsigned int v189; // [rsp+78h] [rbp-198h]
  _QWORD v190[2]; // [rsp+80h] [rbp-190h]
  _QWORD v191[2]; // [rsp+90h] [rbp-180h]
  _QWORD v192[2]; // [rsp+A0h] [rbp-170h] BYREF
  __int16 v193; // [rsp+B0h] [rbp-160h]
  void *v194; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v195; // [rsp+C8h] [rbp-148h]
  _WORD v196[64]; // [rsp+D0h] [rbp-140h] BYREF
  __m128 v197; // [rsp+150h] [rbp-C0h] BYREF
  __m128i v198; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v199; // [rsp+170h] [rbp-A0h]

  v12 = *(__int64 **)(a2 - 72);
  v13 = *(_QWORD *)(a2 - 48);
  v181 = (_QWORD *)a2;
  v14 = *(_QWORD *)(a2 - 24);
  v15 = (__m128)_mm_loadu_si128(a1 + 167);
  v199 = a2;
  v16 = _mm_loadu_si128(a1 + 168);
  s = (_BYTE *)v14;
  v197 = v15;
  v198 = v16;
  v17 = sub_13D1600(v12, v13, v14);
  if ( v17 )
  {
    v20 = *(_QWORD *)(a2 + 8);
    if ( v20 )
    {
      v21 = a1->m128i_i64[0];
      v22 = v17;
      do
      {
        v23 = sub_1648700(v20);
        sub_170B990(v21, (__int64)v23);
        v20 = *(_QWORD *)(v20 + 8);
      }
      while ( v20 );
      goto LABEL_5;
    }
    return 0;
  }
  v27 = *(_BYTE *)(v13 + 16);
  if ( v27 == 9 )
  {
    v37 = *(_QWORD *)(a2 + 8);
    if ( !v37 )
      goto LABEL_12;
  }
  else
  {
    if ( s[16] != 9 )
      goto LABEL_11;
    v37 = *(_QWORD *)(a2 + 8);
    if ( !v37 )
      goto LABEL_11;
  }
  v168 = v13;
  v38 = v37;
  v39 = a1->m128i_i64[0];
  do
  {
    v40 = sub_1648700(v38);
    sub_170B990(v39, (__int64)v40);
    v38 = *(_QWORD *)(v38 + 8);
  }
  while ( v38 );
  v13 = v168;
  v43 = (__int64)v12;
  if ( (__int64 *)a2 == v12 )
    v43 = sub_1599EF0(*(__int64 ***)a2);
  sub_164D160(a2, v43, v15, *(double *)v16.m128i_i64, a5, a6, v41, v42, a9, a10);
  v27 = *(_BYTE *)(v168 + 16);
LABEL_11:
  if ( v27 != 83 )
    goto LABEL_12;
  v58 = *(_QWORD *)(v13 - 24);
  if ( *(_BYTE *)(v58 + 16) != 13 || s[16] != 13 )
    goto LABEL_12;
  v59 = *(__int64 **)(v13 - 48);
  if ( *(_DWORD *)(v58 + 32) <= 0x40u )
    v60 = *(_QWORD *)(v58 + 24);
  else
    v60 = **(_QWORD **)(v58 + 24);
  v61 = (_QWORD *)*((_QWORD *)s + 3);
  if ( *((_DWORD *)s + 8) > 0x40u )
    v61 = (_QWORD *)*v61;
  if ( (unsigned int)v60 >= (unsigned int)*(_QWORD *)(*v59 + 32) )
  {
    v115 = *(_QWORD *)(a2 + 8);
    if ( v115 )
    {
      v116 = a1->m128i_i64[0];
      do
      {
        v117 = sub_1648700(v115);
        sub_170B990(v116, (__int64)v117);
        v115 = *(_QWORD *)(v115 + 8);
      }
      while ( v115 );
      if ( (__int64 *)a2 == v12 )
        v12 = (__int64 *)sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, (__int64)v12, v15, *(double *)v16.m128i_i64, a5, a6, v118, v119, a9, a10);
      return (__int64)v181;
    }
    return 0;
  }
  if ( *(_DWORD *)(*(_QWORD *)a2 + 32LL) <= (unsigned int)v61 )
  {
    v120 = sub_1599EF0(*(__int64 ***)a2);
    v121 = *(_QWORD *)(a2 + 8);
    v22 = v120;
    if ( v121 )
    {
      v122 = a1->m128i_i64[0];
      do
      {
        v123 = sub_1648700(v121);
        sub_170B990(v122, (__int64)v123);
        v121 = *(_QWORD *)(v121 + 8);
      }
      while ( v121 );
LABEL_5:
      if ( a2 == v22 )
        v22 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v22, v15, *(double *)v16.m128i_i64, a5, a6, v24, v25, a9, a10);
      return (__int64)v181;
    }
    return 0;
  }
  if ( (_DWORD)v60 != (_DWORD)v61 || v12 != v59 )
  {
    v62 = *(_QWORD *)(a2 + 8);
    if ( !v62 || *(_QWORD *)(v62 + 8) || *((_BYTE *)sub_1648700(v62) + 16) != 84 )
    {
      v197.m128_u64[0] = (unsigned __int64)&v198;
      v197.m128_u64[1] = 0x1000000000LL;
      v63 = (__int64 ***)sub_17AFF70(
                           (_BYTE *)a2,
                           (__int64)&v197,
                           0,
                           a1->m128i_i64,
                           v15,
                           *(double *)v16.m128i_i64,
                           a5,
                           a6,
                           v18,
                           v19,
                           a9,
                           a10);
      v65 = v63;
      if ( a2 != v64 && (__int64 ***)a2 != v63 )
      {
        if ( !v64 )
          v64 = sub_1599EF0(*v63);
        v178 = v64;
        v196[0] = 257;
        v66 = (_QWORD *)sub_15A01B0((__int64 *)v197.m128_u64[0], v197.m128_u32[2]);
        v181 = sub_1648A60(56, 3u);
        if ( v181 )
          sub_15FA660((__int64)v181, v65, v178, v66, (__int64)&v194, 0);
        if ( (__m128i *)v197.m128_u64[0] != &v198 )
          _libc_free(v197.m128_u64[0]);
        return (__int64)v181;
      }
      if ( (__m128i *)v197.m128_u64[0] != &v198 )
        _libc_free(v197.m128_u64[0]);
    }
LABEL_12:
    v28 = *(_QWORD *)(*v12 + 32);
    v187 = v28;
    if ( (unsigned int)v28 > 0x40 )
    {
      sub_16A4EF0((__int64)&v186, 0, 0);
      v189 = v28;
      sub_16A4EF0((__int64)&v188, -1, 1);
      v197.m128_i32[2] = v189;
      if ( v189 > 0x40 )
      {
        sub_16A4FD0((__int64)&v197, (const void **)&v188);
        goto LABEL_15;
      }
    }
    else
    {
      v186 = 0;
      v189 = v28;
      v197.m128_i32[2] = v28;
      v188 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v28;
    }
    v197.m128_u64[0] = v188;
LABEL_15:
    v31 = sub_17A4D70(a1->m128i_i64, (_BYTE *)a2, (__int64)&v197, &v186, 0);
    if ( v197.m128_i32[2] > 0x40u && v197.m128_u64[0] )
      j_j___libc_free_0_0(v197.m128_u64[0]);
    if ( v31 )
    {
      if ( (_QWORD *)a2 == v31 )
        goto LABEL_24;
      v32 = *(_QWORD *)(a2 + 8);
      if ( v32 )
      {
        v33 = a1->m128i_i64[0];
        do
        {
          v34 = sub_1648700(v32);
          sub_170B990(v33, (__int64)v34);
          v32 = *(_QWORD *)(v32 + 8);
        }
        while ( v32 );
        sub_164D160(a2, (__int64)v31, v15, *(double *)v16.m128i_i64, a5, a6, v35, v36, a9, a10);
        goto LABEL_24;
      }
      goto LABEL_128;
    }
    v44 = *(_QWORD *)(a2 - 72);
    v45 = *(_BYTE *)(v44 + 16);
    v182 = v44;
    if ( v45 <= 0x17u )
      goto LABEL_86;
    v46 = *(_QWORD *)(v44 + 8);
    if ( !v46 || *(_QWORD *)(v46 + 8) )
      goto LABEL_86;
    if ( v45 != 85 )
    {
      if ( v45 != 84 )
      {
LABEL_86:
        v67 = *(_QWORD *)(a2 + 8);
        if ( v67 && !*(_QWORD *)(v67 + 8) && *((_BYTE *)sub_1648700(v67) + 16) == 84
          || (v68 = *(__int64 ***)a2, v69 = *(_QWORD *)(*(_QWORD *)a2 + 32LL), sa = v69, (_DWORD)v69 == 1) )
        {
LABEL_128:
          v181 = 0;
          goto LABEL_24;
        }
        v70 = *(_QWORD *)(a2 - 48);
        v71 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
        v197.m128_u64[0] = (unsigned __int64)&v198;
        v72 = &v198;
        v183 = v70;
        v197.m128_u64[1] = 0x1000000000LL;
        if ( (unsigned __int64)(int)v69 > 0x10 )
        {
          sub_16CD150((__int64)&v197, &v198, (int)v69, 1, v29, v30);
          v72 = (__m128i *)v197.m128_u64[0];
        }
        v197.m128_i32[2] = v71;
        if ( v71 )
          memset(v72, 0, v71);
        v73 = (_QWORD *)a2;
        v74 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v74 + 16) != 13 )
        {
LABEL_108:
          v79 = (__m128i *)v197.m128_u64[0];
LABEL_109:
          if ( v79 != &v198 )
          {
LABEL_110:
            _libc_free((unsigned __int64)v79);
            v181 = v31;
            goto LABEL_24;
          }
          goto LABEL_128;
        }
        while ( 1 )
        {
          if ( v183 != *(v73 - 6) )
            goto LABEL_108;
          v75 = *(v73 - 9);
          if ( *(_BYTE *)(v75 + 16) != 84 )
            v75 = 0;
          if ( (_QWORD *)a2 != v73 )
          {
            v76 = v73[1];
            if ( !v76 || *(_QWORD *)(v76 + 8) )
              break;
          }
          v78 = *(_QWORD **)(v74 + 24);
          if ( *(_DWORD *)(v74 + 32) > 0x40u )
            goto LABEL_105;
LABEL_106:
          *((_BYTE *)v78 + v197.m128_u64[0]) = 1;
          if ( !v75 )
            goto LABEL_114;
          v73 = (_QWORD *)v75;
          v74 = *(_QWORD *)(v75 - 24);
          if ( *(_BYTE *)(v74 + 16) != 13 )
            goto LABEL_108;
        }
        if ( v75 )
          goto LABEL_108;
        if ( *(_DWORD *)(v74 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v74 + 24) )
            goto LABEL_108;
        }
        else
        {
          v164 = *(_DWORD *)(v74 + 32);
          v179 = v74;
          v77 = sub_16A57B0(v74 + 24);
          v74 = v179;
          v75 = 0;
          if ( v164 != v77 )
            goto LABEL_108;
        }
        v78 = *(_QWORD **)(v74 + 24);
        if ( *(_DWORD *)(v74 + 32) <= 0x40u )
        {
          *((_BYTE *)v78 + v197.m128_u64[0]) = 1;
LABEL_114:
          v79 = (__m128i *)v197.m128_u64[0];
          v80 = (_BYTE *)(v197.m128_u64[0] + v197.m128_u32[2]);
          v81 = (_BYTE *)v197.m128_u64[0];
          if ( (__int64)v197.m128_u32[2] >> 2 )
          {
            while ( *v81 )
            {
              if ( !v81[1] )
              {
                ++v81;
                goto LABEL_121;
              }
              if ( !v81[2] )
              {
                v81 += 2;
                goto LABEL_121;
              }
              if ( !v81[3] )
              {
                v81 += 3;
                goto LABEL_121;
              }
              v81 += 4;
              if ( (_BYTE *)(v197.m128_u64[0] + 4 * ((__int64)v197.m128_u32[2] >> 2)) == v81 )
                goto LABEL_241;
            }
            goto LABEL_121;
          }
LABEL_241:
          v157 = v80 - v81;
          if ( v80 - v81 == 2 )
            goto LABEL_250;
          if ( v157 != 3 )
          {
            if ( v157 == 1 )
              goto LABEL_244;
            goto LABEL_122;
          }
          if ( !*v81 )
            goto LABEL_121;
          ++v81;
LABEL_250:
          if ( !*v81 )
            goto LABEL_121;
          ++v81;
LABEL_244:
          if ( !*v81 )
          {
LABEL_121:
            if ( v80 != v81 )
              goto LABEL_109;
          }
LABEL_122:
          v82 = *(v73 - 3);
          v83 = *(_DWORD *)(v82 + 32);
          if ( v83 <= 0x40 )
          {
            if ( *(_QWORD *)(v82 + 24) )
              goto LABEL_226;
          }
          else
          {
            if ( v83 == (unsigned int)sub_16A57B0(v82 + 24) )
              goto LABEL_124;
LABEL_226:
            v196[0] = 257;
            v144 = (_QWORD *)sub_16498A0(a2);
            v145 = sub_1643350(v144);
            v180 = sub_159C470(v145, 0, 0);
            v146 = (__int64 *)sub_1599EF0(v68);
            v147 = sub_1648A60(56, 3u);
            v73 = v147;
            if ( v147 )
              sub_15FA480((__int64)v147, v146, v183, v180, (__int64)&v194, a2);
          }
LABEL_124:
          v84 = (_QWORD *)sub_16498A0(a2);
          v85 = (__int64 *)sub_1643350(v84);
          v86 = (__int64 **)sub_16463B0(v85, sa);
          v87 = (_QWORD *)sub_1598F00(v86);
          v88 = sub_1599EF0(v68);
          v196[0] = 257;
          v89 = v88;
          v90 = sub_1648A60(56, 3u);
          v31 = v90;
          if ( v90 )
            sub_15FA660((__int64)v90, v73, v89, v87, (__int64)&v194, 0);
          v79 = (__m128i *)v197.m128_u64[0];
          v181 = v31;
          if ( (__m128i *)v197.m128_u64[0] == &v198 )
            goto LABEL_24;
          goto LABEL_110;
        }
LABEL_105:
        v78 = (_QWORD *)*v78;
        goto LABEL_106;
      }
      v47 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v47 + 16) != 13 )
        goto LABEL_129;
      v48 = *(_DWORD *)(v47 + 32);
      if ( v48 > 0x40 )
      {
        if ( v48 - (unsigned int)sub_16A57B0(v47 + 24) > 0x40 )
          goto LABEL_129;
        v49 = **(_QWORD **)(v47 + 24);
      }
      else
      {
        v49 = *(_QWORD *)(v47 + 24);
      }
      v50 = *(__int64 ***)a2;
      v190[0] = v49;
      v169 = v50[4];
      if ( *(_BYTE *)(*(_QWORD *)(a2 - 48) + 16LL) <= 0x10u )
      {
        v191[0] = *(_QWORD *)(a2 - 48);
        v51 = *(_QWORD *)(v182 - 24);
        if ( *(_BYTE *)(v51 + 16) == 13 )
        {
          if ( *(_DWORD *)(v51 + 32) <= 0x40u )
          {
            v52 = *(_QWORD *)(v51 + 24);
            goto LABEL_53;
          }
          sd = *(_DWORD *)(v51 + 32);
          if ( sd - (unsigned int)sub_16A57B0(v51 + 24) <= 0x40 )
          {
            v52 = **(_QWORD **)(v51 + 24);
LABEL_53:
            v190[1] = v52;
            if ( *(_BYTE *)(*(_QWORD *)(v182 - 48) + 16LL) <= 0x10u )
            {
              v191[1] = *(_QWORD *)(v182 - 48);
              v53 = 8LL * (unsigned int)v169;
              v194 = v196;
              v195 = 0x1000000000LL;
              if ( (unsigned int)v169 > 0x10uLL )
              {
                na = 8LL * (unsigned int)v169;
                sub_16CD150((__int64)&v194, v196, (unsigned int)v169, 8, v53, 0);
                v143 = na;
                LODWORD(v195) = (_DWORD)v169;
                if ( v194 == (char *)v194 + na )
                {
                  v197.m128_u64[1] = 0x1000000000LL;
                  v197.m128_u64[0] = (unsigned __int64)&v198;
                }
                else
                {
                  memset(v194, 0, na);
                  v197.m128_u64[0] = (unsigned __int64)&v198;
                  v143 = 8LL * (unsigned int)v169;
                  v197.m128_u64[1] = 0x1000000000LL;
                }
                nb = v143;
                sub_16CD150((__int64)&v197, &v198, (unsigned int)v169, 8, v143, 0);
                v54 = (__m128i *)v197.m128_u64[0];
                v53 = nb;
              }
              else
              {
                LODWORD(v195) = (_DWORD)v169;
                if ( v196 == (_WORD *)((char *)v196 + v53) )
                {
                  v197.m128_i32[3] = 16;
                }
                else
                {
                  memset(v196, 0, 8LL * (unsigned int)v169);
                  v197.m128_i32[3] = 16;
                  v53 = 8LL * (unsigned int)v169;
                }
                v197.m128_u64[0] = (unsigned __int64)&v198;
                v54 = &v198;
              }
              v197.m128_i32[2] = (int)v169;
              if ( v53 )
                memset(v54, 0, v53);
              v55 = (__int64 *)v194;
              for ( i = 0; ; v49 = v190[i] )
              {
                v57 = &v55[v49];
                if ( !*v57 )
                {
                  *v57 = v191[i];
                  v138 = (_QWORD *)sub_16498A0(a2);
                  v139 = sub_1643350(v138);
                  v140 = (__int64 *)(v197.m128_u64[0] + 8 * v49);
                  *v140 = sub_159C470(v139, (unsigned int)v169 + v49, 0);
                  v55 = (__int64 *)v194;
                }
                if ( ++i == 2 )
                  break;
              }
              v31 = 0;
              if ( (_DWORD)v169 )
              {
                v108 = 0;
                do
                {
                  v109 = &v55[v108];
                  if ( !*v109 )
                  {
                    *v109 = sub_1599EF0(*(__int64 ***)(*(_QWORD *)a2 + 24LL));
                    v110 = (_QWORD *)sub_16498A0(a2);
                    v111 = sub_1643350(v110);
                    v112 = (__int64 *)(v197.m128_u64[0] + 8 * v108);
                    *v112 = sub_159C470(v111, v108, 0);
                    v55 = (__int64 *)v194;
                  }
                  ++v108;
                }
                while ( (unsigned int)v169 != v108 );
                v31 = 0;
              }
              v166 = *(_QWORD **)(v182 - 72);
              v170 = sub_15A01B0(v55, (unsigned int)v195);
              v185 = (_QWORD *)sub_15A01B0((__int64 *)v197.m128_u64[0], v197.m128_u32[2]);
              v193 = 257;
              v113 = sub_1648A60(56, 3u);
              v114 = v113;
              if ( v113 )
                goto LABEL_165;
              goto LABEL_166;
            }
          }
        }
      }
LABEL_129:
      v91 = 84;
      goto LABEL_130;
    }
    n = *(_QWORD *)(v44 - 48);
    if ( *(_BYTE *)(n + 16) <= 0x10u )
    {
      v158 = *(_QWORD *)(a2 - 48);
      if ( *(_BYTE *)(v158 + 16) <= 0x10u )
      {
        v104 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v104 + 16) == 13 )
        {
          v105 = *(_DWORD *)(v104 + 32);
          if ( v105 <= 0x40 )
          {
            v165 = *(_QWORD *)(v104 + 24);
            goto LABEL_153;
          }
          if ( v105 - (unsigned int)sub_16A57B0(v104 + 24) <= 0x40 )
          {
            v165 = **(_QWORD **)(v104 + 24);
LABEL_153:
            v106 = *(_QWORD *)(v182 - 24);
            v159 = v106;
            v107 = *(_QWORD *)(*(_QWORD *)v106 + 32LL);
            sc = *(void **)(**(_QWORD **)(v182 - 72) + 32LL);
            if ( (_DWORD)v107 == (_DWORD)sc )
            {
              if ( !(_DWORD)v107 )
                goto LABEL_190;
              v124 = 0;
              v125 = *(_QWORD *)(*(_QWORD *)v106 + 32LL);
              do
              {
                v126 = sub_15FA9D0(v106, v124);
                if ( v126 != -1 && v124 != v126 && v126 != v124 + (_DWORD)sc )
                {
                  v182 = *(_QWORD *)(a2 - 72);
                  goto LABEL_154;
                }
                ++v124;
                v106 = *(_QWORD *)(v182 - 24);
              }
              while ( v125 != v124 );
              v159 = *(_QWORD *)(v182 - 24);
LABEL_190:
              v127 = *(_QWORD *)(*(_QWORD *)v159 + 32LL);
              v194 = v196;
              v195 = 0x1000000000LL;
              v160 = v127;
              v171 = (unsigned int)v127;
              v128 = 8LL * (unsigned int)v127;
              if ( (unsigned int)v127 > 0x10uLL )
              {
                sub_16CD150((__int64)&v194, v196, (unsigned int)v127, 8, v29, v30);
                LODWORD(v195) = v160;
                if ( v194 == (char *)v194 + v128 )
                {
                  v197.m128_u64[1] = 0x1000000000LL;
                  v197.m128_u64[0] = (unsigned __int64)&v198;
                }
                else
                {
                  memset(v194, 0, v128);
                  v197.m128_u64[0] = (unsigned __int64)&v198;
                  v197.m128_u64[1] = 0x1000000000LL;
                }
                sub_16CD150((__int64)&v197, &v198, v171, 8, v141, v142);
                v129 = (__m128i *)v197.m128_u64[0];
              }
              else
              {
                LODWORD(v195) = v127;
                if ( v196 != &v196[v128 / 2] )
                  memset(v196, 0, v128);
                v197.m128_i32[3] = 16;
                v197.m128_u64[0] = (unsigned __int64)&v198;
                v129 = &v198;
              }
              v197.m128_i32[2] = v160;
              if ( v128 )
                memset(v129, 0, v128);
              if ( v160 )
              {
                v130 = 0;
                do
                {
                  v132 = 8 * v130;
                  v133 = (__int64 *)((char *)v194 + 8 * v130);
                  if ( v165 == v130 )
                  {
                    *v133 = v158;
                    v134 = (_QWORD *)sub_16498A0(v182);
                    v135 = sub_1643350(v134);
                    v136 = (__int64 *)(v197.m128_u64[0] + v132);
                    *v136 = sub_15A0680(v135, v165 + v171, 0);
                  }
                  else
                  {
                    *v133 = sub_15A0A60(n, v130);
                    v131 = (__int64 *)(v197.m128_u64[0] + v132);
                    *v131 = sub_15A0A60(v159, v130);
                  }
                  ++v130;
                }
                while ( v171 != v130 );
                v31 = 0;
              }
              v166 = *(_QWORD **)(v182 - 72);
              v170 = sub_15A01B0((__int64 *)v194, (unsigned int)v195);
              v137 = sub_15A01B0((__int64 *)v197.m128_u64[0], v197.m128_u32[2]);
              v193 = 257;
              v185 = (_QWORD *)v137;
              v113 = sub_1648A60(56, 3u);
              v114 = v113;
              if ( v113 )
LABEL_165:
                sub_15FA660((__int64)v113, v166, v170, v185, (__int64)v192, 0);
LABEL_166:
              if ( (__m128i *)v197.m128_u64[0] != &v198 )
                _libc_free(v197.m128_u64[0]);
              if ( v194 != v196 )
                _libc_free((unsigned __int64)v194);
              if ( v114 )
              {
                v181 = v114;
                goto LABEL_24;
              }
              v182 = *(_QWORD *)(a2 - 72);
              v91 = *(_BYTE *)(v182 + 16);
            }
            else
            {
LABEL_154:
              v91 = *(_BYTE *)(v182 + 16);
            }
LABEL_130:
            v92 = a1->m128i_i64[1];
            if ( v91 == 84 )
            {
              v93 = *(_QWORD *)(v182 + 8);
              if ( v93 )
              {
                if ( !*(_QWORD *)(v93 + 8) )
                {
                  v94 = *(_QWORD *)(v182 - 72);
                  v95 = v182;
                  if ( v94 )
                  {
                    v96 = *(_QWORD *)(v182 - 48);
                    v184 = v96;
                    if ( v96 )
                    {
                      if ( *(_BYTE *)(v96 + 16) > 0x10u )
                      {
                        v97 = *(_BYTE **)(v95 - 24);
                        sb = v97;
                        if ( v97[16] == 13 )
                        {
                          v98 = *(_QWORD *)(a2 - 48);
                          if ( *(_BYTE *)(v98 + 16) <= 0x10u )
                          {
                            v99 = *(_QWORD *)(a2 - 24);
                            if ( !v99 )
                              BUG();
                            if ( *(_BYTE *)(v99 + 16) == 13 && v97 != (_BYTE *)v99 )
                            {
                              v196[0] = 257;
                              if ( *(_BYTE *)(v94 + 16) > 0x10u
                                || *(_BYTE *)(v98 + 16) > 0x10u
                                || *(_BYTE *)(v99 + 16) > 0x10u )
                              {
                                v167 = v99;
                                v172 = (__int64 *)v94;
                                v198.m128i_i16[0] = 257;
                                v148 = sub_1648A60(56, 3u);
                                v100 = v148;
                                if ( v148 )
                                  sub_15FA480((__int64)v148, v172, v98, v167, (__int64)&v197, 0);
                                v149 = *(_QWORD *)(v92 + 8);
                                if ( v149 )
                                {
                                  v150 = *(unsigned __int64 **)(v92 + 16);
                                  sub_157E9D0(v149 + 40, (__int64)v100);
                                  v151 = v100[3];
                                  v152 = *v150;
                                  v100[4] = (__int64)v150;
                                  v152 &= 0xFFFFFFFFFFFFFFF8LL;
                                  v100[3] = v152 | v151 & 7;
                                  *(_QWORD *)(v152 + 8) = v100 + 3;
                                  *v150 = *v150 & 7 | (unsigned __int64)(v100 + 3);
                                }
                                sub_164B780((__int64)v100, (__int64 *)&v194);
                                v192[0] = v100;
                                if ( !*(_QWORD *)(v92 + 80) )
                                  sub_4263D6(v100, &v194, v153);
                                (*(void (__fastcall **)(__int64, _QWORD *))(v92 + 88))(v92 + 64, v192);
                                v154 = *(_QWORD *)v92;
                                if ( *(_QWORD *)v92 )
                                {
                                  v192[0] = *(_QWORD *)v92;
                                  sub_1623A60((__int64)v192, v154, 2);
                                  v155 = v100[6];
                                  if ( v155 )
                                    sub_161E7C0((__int64)(v100 + 6), v155);
                                  v156 = (unsigned __int8 *)v192[0];
                                  v100[6] = v192[0];
                                  if ( v156 )
                                    sub_1623210((__int64)v192, v156, (__int64)(v100 + 6));
                                }
                              }
                              else
                              {
                                v100 = (__int64 *)sub_15A3890((__int64 *)v94, v98, v99, 0);
                                v101 = sub_14DBA30((__int64)v100, *(_QWORD *)(v92 + 96), 0);
                                if ( v101 )
                                  v100 = (__int64 *)v101;
                              }
                              v198.m128i_i16[0] = 257;
                              v102 = sub_1648A60(56, 3u);
                              v103 = v102;
                              if ( v102 )
                              {
                                sub_15FA480((__int64)v102, v100, v184, (__int64)sb, (__int64)&v197, 0);
                                v181 = v103;
LABEL_24:
                                if ( v189 > 0x40 && v188 )
                                  j_j___libc_free_0_0(v188);
                                if ( v187 > 0x40 && v186 )
                                  j_j___libc_free_0_0(v186);
                                return (__int64)v181;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            goto LABEL_86;
          }
        }
      }
    }
    v91 = 85;
    goto LABEL_130;
  }
  return sub_170E100(a1->m128i_i64, a2, (__int64)v12, v15, *(double *)v16.m128i_i64, a5, a6, v18, v19, a9, a10);
}
