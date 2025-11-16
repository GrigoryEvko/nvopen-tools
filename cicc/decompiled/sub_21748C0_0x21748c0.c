// Function: sub_21748C0
// Address: 0x21748c0
//
__int64 *__fastcall sub_21748C0(__int64 a1, double a2, double a3, __m128i a4, __int64 a5, __int64 *a6)
{
  unsigned int v6; // r14d
  __int64 v8; // rax
  __int64 v9; // rsi
  int v10; // ecx
  __int64 v11; // rax
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  int v19; // r8d
  int v20; // r9d
  size_t v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // eax
  const void *v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rdx
  int v30; // r13d
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // rdx
  int v35; // r13d
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rax
  unsigned int v41; // edx
  unsigned __int8 v42; // al
  __int64 v43; // r13
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rdx
  __int64 v50; // r10
  unsigned __int64 v51; // r11
  unsigned __int8 *v52; // rax
  const void **v53; // r8
  __int64 v54; // rcx
  __int64 *v55; // rax
  int v56; // eax
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r8
  __int64 v60; // rdx
  __int64 v61; // r13
  int v62; // eax
  __int64 v63; // rsi
  __int64 v64; // rax
  const void **v65; // r8
  __int64 v66; // rdx
  __int64 v67; // rdx
  _QWORD *v68; // rsi
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 *v71; // rax
  __int64 **v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 *v75; // rax
  int v76; // edx
  unsigned __int8 v77; // cf
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r9
  __int64 v84; // rdx
  __int64 *v85; // rdx
  unsigned int *v86; // rax
  __int64 v87; // rcx
  unsigned __int64 v88; // rbx
  __int64 v89; // rax
  char v90; // di
  const void **v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rbx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rax
  __m128i *v98; // rax
  __m128i v99; // xmm2
  __int64 *v100; // r12
  const void **v102; // r8
  __int64 v103; // rcx
  unsigned int v104; // edx
  unsigned __int64 v105; // rcx
  unsigned __int8 *v106; // rdx
  const void **v107; // r8
  __int64 *v108; // rax
  unsigned int v109; // edx
  const void *v110; // rsi
  const void *v111; // rsi
  const void *v112; // rsi
  const void *v113; // rsi
  const void *v114; // rsi
  const void *v115; // rsi
  const void *v116; // rsi
  __int128 v117; // [rsp-10h] [rbp-3B0h]
  __int128 v118; // [rsp+0h] [rbp-3A0h]
  __int128 v119; // [rsp+10h] [rbp-390h]
  int v120; // [rsp+28h] [rbp-378h]
  __int64 v121; // [rsp+38h] [rbp-368h]
  __int64 v122; // [rsp+40h] [rbp-360h]
  __int64 v123; // [rsp+48h] [rbp-358h]
  int v124; // [rsp+50h] [rbp-350h]
  __int64 v125; // [rsp+50h] [rbp-350h]
  __int64 v126; // [rsp+58h] [rbp-348h]
  __int64 v127; // [rsp+58h] [rbp-348h]
  __int64 v128; // [rsp+60h] [rbp-340h]
  __int64 v129; // [rsp+68h] [rbp-338h]
  __m128i v130; // [rsp+70h] [rbp-330h] BYREF
  __int128 v131; // [rsp+80h] [rbp-320h]
  __m128i v132; // [rsp+90h] [rbp-310h]
  __int64 *v133; // [rsp+A0h] [rbp-300h]
  __int64 v134; // [rsp+A8h] [rbp-2F8h]
  __int64 v135; // [rsp+B0h] [rbp-2F0h]
  unsigned __int64 v136; // [rsp+B8h] [rbp-2E8h]
  __int64 v137; // [rsp+C0h] [rbp-2E0h]
  unsigned __int64 v138; // [rsp+C8h] [rbp-2D8h]
  __int64 *v139; // [rsp+D0h] [rbp-2D0h]
  __int64 v140; // [rsp+D8h] [rbp-2C8h]
  __int64 v141; // [rsp+E0h] [rbp-2C0h]
  __int64 v142; // [rsp+E8h] [rbp-2B8h]
  __int64 v143; // [rsp+F0h] [rbp-2B0h] BYREF
  int v144; // [rsp+F8h] [rbp-2A8h]
  __int64 v145; // [rsp+100h] [rbp-2A0h] BYREF
  const void **v146; // [rsp+108h] [rbp-298h]
  __int128 v147; // [rsp+110h] [rbp-290h]
  __int64 v148; // [rsp+120h] [rbp-280h]
  __int128 v149; // [rsp+130h] [rbp-270h] BYREF
  __int64 v150; // [rsp+140h] [rbp-260h]
  _QWORD *v151; // [rsp+150h] [rbp-250h] BYREF
  __int64 v152; // [rsp+158h] [rbp-248h]
  _QWORD v153[8]; // [rsp+160h] [rbp-240h] BYREF
  __int64 v154; // [rsp+1A0h] [rbp-200h] BYREF
  int v155; // [rsp+1A8h] [rbp-1F8h]
  int v156; // [rsp+1ACh] [rbp-1F4h]
  int v157; // [rsp+1B0h] [rbp-1F0h]
  void *v158; // [rsp+1B8h] [rbp-1E8h] BYREF
  size_t n; // [rsp+1C0h] [rbp-1E0h]
  char v160; // [rsp+1C8h] [rbp-1D8h] BYREF
  void *dest; // [rsp+1D0h] [rbp-1D0h] BYREF
  __int64 v162; // [rsp+1D8h] [rbp-1C8h]
  _BYTE v163[128]; // [rsp+1E0h] [rbp-1C0h] BYREF
  _QWORD v164[2]; // [rsp+260h] [rbp-140h] BYREF
  char v165; // [rsp+270h] [rbp-130h] BYREF
  void *v166; // [rsp+280h] [rbp-120h] BYREF
  __int64 v167; // [rsp+288h] [rbp-118h]
  _BYTE v168[160]; // [rsp+290h] [rbp-110h] BYREF
  __int64 v169; // [rsp+330h] [rbp-70h]
  void *v170; // [rsp+338h] [rbp-68h] BYREF
  __int64 v171; // [rsp+340h] [rbp-60h]
  _BYTE v172[88]; // [rsp+348h] [rbp-58h] BYREF

  v8 = a6[2];
  v9 = *(_QWORD *)(a1 + 72);
  v134 = a1;
  v126 = v8;
  v143 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v143, v9, 2);
  v10 = *(_DWORD *)(v134 + 60);
  v144 = *(_DWORD *)(v134 + 64);
  v11 = *(_QWORD *)(v134 + 32);
  v124 = v10;
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = *(__int64 **)(*(_QWORD *)(v11 + 80) + 88LL);
  v15 = *(_QWORD *)(v11 + 120);
  v130 = v12;
  v132 = v13;
  v16 = *(_QWORD *)(v15 + 88);
  v133 = v14;
  if ( *(_DWORD *)(v16 + 32) <= 0x40u )
    *(_QWORD *)&v131 = *(_QWORD *)(v16 + 24);
  else
    *(_QWORD *)&v131 = **(_QWORD **)(v16 + 24);
  v17 = *(_QWORD *)(*(_QWORD *)(v11 + 160) + 88LL);
  LODWORD(v135) = v131;
  if ( *(_DWORD *)(v17 + 32) <= 0x40u )
    v121 = *(_QWORD *)(v17 + 24);
  else
    v121 = **(_QWORD **)(v17 + 24);
  v18 = sub_1E0A0C0(a6[4]);
  v162 = 0x1000000000LL;
  v171 = 0x800000000LL;
  n = 0x800000000LL;
  v167 = 0x800000000LL;
  v158 = &v160;
  v137 = (__int64)&v160;
  v164[0] = &v165;
  v166 = v168;
  v170 = v172;
  dest = v163;
  v164[1] = 0;
  v165 = 0;
  v169 = 0;
  sub_15A9210((__int64)&v154);
  sub_2240AE0(v164, v18 + 192);
  LOBYTE(v154) = *(_BYTE *)v18;
  HIDWORD(v154) = *(_DWORD *)(v18 + 4);
  v155 = *(_DWORD *)(v18 + 8);
  v156 = *(_DWORD *)(v18 + 12);
  v157 = *(_DWORD *)(v18 + 16);
  if ( &v158 != (void **)(v18 + 24) )
  {
    v21 = *(unsigned int *)(v18 + 32);
    v22 = (unsigned int)n;
    v20 = *(_DWORD *)(v18 + 32);
    if ( v21 <= (unsigned int)n )
    {
      if ( *(_DWORD *)(v18 + 32) )
      {
        v111 = *(const void **)(v18 + 24);
        LODWORD(v137) = *(_DWORD *)(v18 + 32);
        memmove(v158, v111, v21);
        v20 = v137;
      }
    }
    else
    {
      if ( v21 > HIDWORD(n) )
      {
        v112 = (const void *)v137;
        LODWORD(v137) = *(_DWORD *)(v18 + 32);
        LODWORD(n) = 0;
        sub_16CD150((__int64)&v158, v112, v21, 1, v19, v20);
        v21 = *(unsigned int *)(v18 + 32);
        v20 = v137;
        v22 = 0;
        v23 = *(_DWORD *)(v18 + 32);
      }
      else
      {
        v23 = *(_DWORD *)(v18 + 32);
        if ( (_DWORD)n )
        {
          v113 = *(const void **)(v18 + 24);
          v120 = *(_DWORD *)(v18 + 32);
          v137 = (unsigned int)n;
          memmove(v158, v113, (unsigned int)n);
          v21 = *(unsigned int *)(v18 + 32);
          v20 = v120;
          v22 = v137;
          v23 = *(_DWORD *)(v18 + 32);
        }
      }
      v24 = (const void *)(v22 + *(_QWORD *)(v18 + 24));
      if ( v24 != (const void *)(*(_QWORD *)(v18 + 24) + v21) )
      {
        LODWORD(v137) = v20;
        memcpy((char *)v158 + v22, v24, v23 - v22);
        v20 = v137;
      }
    }
    LODWORD(n) = v20;
  }
  if ( &dest != (void **)(v18 + 48) )
  {
    v25 = *(unsigned int *)(v18 + 56);
    v20 = *(_DWORD *)(v18 + 56);
    if ( v25 <= (unsigned int)v162 )
    {
      if ( *(_DWORD *)(v18 + 56) )
      {
        v110 = *(const void **)(v18 + 48);
        LODWORD(v137) = *(_DWORD *)(v18 + 56);
        memmove(dest, v110, 8 * v25);
        v20 = v137;
      }
    }
    else
    {
      if ( v25 > HIDWORD(v162) )
      {
        v26 = 0;
        LODWORD(v137) = *(_DWORD *)(v18 + 56);
        LODWORD(v162) = 0;
        sub_16CD150((__int64)&dest, v163, v25, 8, v19, v20);
        v25 = *(unsigned int *)(v18 + 56);
        v20 = v137;
      }
      else
      {
        v26 = 8LL * (unsigned int)v162;
        if ( (_DWORD)v162 )
        {
          v114 = *(const void **)(v18 + 48);
          LODWORD(v137) = *(_DWORD *)(v18 + 56);
          memmove(dest, v114, 8LL * (unsigned int)v162);
          v25 = *(unsigned int *)(v18 + 56);
          v20 = v137;
        }
      }
      v27 = *(_QWORD *)(v18 + 48);
      v28 = 8 * v25;
      if ( v27 + v26 != v28 + v27 )
      {
        LODWORD(v137) = v20;
        memcpy((char *)dest + v26, (const void *)(v27 + v26), v28 - v26);
        v20 = v137;
      }
    }
    LODWORD(v162) = v20;
  }
  if ( &v166 != (void **)(v18 + 224) )
  {
    v29 = *(unsigned int *)(v18 + 232);
    v30 = *(_DWORD *)(v18 + 232);
    if ( v29 <= (unsigned int)v167 )
    {
      if ( *(_DWORD *)(v18 + 232) )
        memmove(v166, *(const void **)(v18 + 224), 20 * v29);
    }
    else
    {
      if ( v29 > HIDWORD(v167) )
      {
        LODWORD(v167) = 0;
        sub_16CD150((__int64)&v166, v168, v29, 20, v19, v20);
        v29 = *(unsigned int *)(v18 + 232);
        v31 = 0;
      }
      else
      {
        v31 = 20LL * (unsigned int)v167;
        if ( (_DWORD)v167 )
        {
          v115 = *(const void **)(v18 + 224);
          v137 = 20LL * (unsigned int)v167;
          memmove(v166, v115, v137);
          v29 = *(unsigned int *)(v18 + 232);
          v31 = v137;
        }
      }
      v32 = *(_QWORD *)(v18 + 224);
      v33 = 20 * v29;
      if ( v32 + v31 != v33 + v32 )
        memcpy((char *)v166 + v31, (const void *)(v32 + v31), v33 - v31);
    }
    LODWORD(v167) = v30;
  }
  if ( &v170 != (void **)(v18 + 408) )
  {
    v34 = *(unsigned int *)(v18 + 416);
    v35 = *(_DWORD *)(v18 + 416);
    if ( v34 <= (unsigned int)v171 )
    {
      if ( *(_DWORD *)(v18 + 416) )
        memmove(v170, *(const void **)(v18 + 408), 4 * v34);
    }
    else
    {
      if ( v34 > HIDWORD(v171) )
      {
        LODWORD(v171) = 0;
        sub_16CD150((__int64)&v170, v172, v34, 4, v19, v20);
        v34 = *(unsigned int *)(v18 + 416);
        v36 = 0;
      }
      else
      {
        v36 = 4LL * (unsigned int)v171;
        if ( (_DWORD)v171 )
        {
          v116 = *(const void **)(v18 + 408);
          v137 = 4LL * (unsigned int)v171;
          memmove(v170, v116, v137);
          v34 = *(unsigned int *)(v18 + 416);
          v36 = v137;
        }
      }
      v37 = *(_QWORD *)(v18 + 408);
      v38 = 4 * v34;
      if ( v37 + v36 != v38 + v37 )
        memcpy((char *)v170 + v36, (const void *)(v37 + v36), v38 - v36);
    }
    LODWORD(v171) = v35;
  }
  v39 = 0;
  v151 = 0;
  v152 = 0;
  v153[0] = 0;
  v149 = (unsigned __int64)v133;
  LOBYTE(v150) = 0;
  if ( v133 )
  {
    v40 = *v133;
    if ( *(_BYTE *)(*v133 + 8) == 16 )
      v40 = **(_QWORD **)(v40 + 16);
    v39 = *(_DWORD *)(v40 + 8) >> 8;
  }
  HIDWORD(v150) = v39;
  v41 = 8 * sub_15A9520((__int64)&v154, 0);
  if ( v41 == 32 )
  {
    v42 = 5;
  }
  else if ( v41 > 0x20 )
  {
    v42 = 6;
    if ( v41 != 64 )
    {
      v42 = 0;
      if ( v41 == 128 )
        v42 = 7;
    }
  }
  else
  {
    v42 = 3;
    if ( v41 != 8 )
      v42 = 4 * (v41 == 16);
  }
  v43 = sub_1D2B730(
          a6,
          v42,
          0,
          (__int64)&v143,
          v130.m128i_i64[0],
          v130.m128i_i64[1],
          v132.m128i_i64[0],
          v132.m128i_i64[1],
          v149,
          v150,
          0,
          0,
          (__int64)&v151,
          0);
  v137 = v43;
  v44 = *(_QWORD *)(v43 + 40);
  v138 = v45;
  v128 = (unsigned int)v45;
  v129 = 16LL * (unsigned int)v45;
  *(_QWORD *)&v119 = sub_1D38BB0(
                       (__int64)a6,
                       (unsigned int)(v135 - 1),
                       (__int64)&v143,
                       *(unsigned __int8 *)(v129 + v44),
                       *(const void ***)(v129 + v44 + 8),
                       0,
                       v12,
                       *(double *)v13.m128i_i64,
                       a4,
                       0);
  *((_QWORD *)&v119 + 1) = v46;
  *(_QWORD *)&v131 = sub_1D38BB0(
                       (__int64)a6,
                       -(__int64)(unsigned int)v131,
                       (__int64)&v143,
                       *(unsigned __int8 *)(v129 + *(_QWORD *)(v43 + 40)),
                       *(const void ***)(v129 + *(_QWORD *)(v43 + 40) + 8),
                       0,
                       v12,
                       *(double *)v13.m128i_i64,
                       a4,
                       0);
  v47 = *(_QWORD *)(v43 + 40);
  *((_QWORD *)&v131 + 1) = v48;
  v50 = sub_1D38BB0(
          (__int64)a6,
          (unsigned int)v121,
          (__int64)&v143,
          *(unsigned __int8 *)(v129 + v47),
          *(const void ***)(v129 + v47 + 8),
          0,
          v12,
          *(double *)v13.m128i_i64,
          a4,
          0);
  v51 = v49;
  if ( (unsigned int)v135 > *(_DWORD *)(v126 + 84) )
  {
    v102 = *(const void ***)(*(_QWORD *)(v43 + 40) + v129 + 8);
    v103 = *(unsigned __int8 *)(*(_QWORD *)(v43 + 40) + v129);
    v135 = v50;
    v136 = v49;
    v137 = (__int64)sub_1D332F0(
                      a6,
                      52,
                      (__int64)&v143,
                      v103,
                      v102,
                      0,
                      *(double *)v12.m128i_i64,
                      *(double *)v13.m128i_i64,
                      a4,
                      v137,
                      v138,
                      v119);
    v105 = v104 | v138 & 0xFFFFFFFF00000000LL;
    v106 = (unsigned __int8 *)(*(_QWORD *)(v137 + 40) + 16LL * v104);
    v107 = (const void **)*((_QWORD *)v106 + 1);
    v138 = v105;
    v108 = sub_1D332F0(
             a6,
             118,
             (__int64)&v143,
             *v106,
             v107,
             0,
             *(double *)v12.m128i_i64,
             *(double *)v13.m128i_i64,
             a4,
             v137,
             v105,
             v131);
    v50 = v135;
    v51 = v136;
    v127 = (__int64)v108;
    v129 = 16LL * v109;
    v128 = v109;
  }
  else
  {
    v127 = v137;
  }
  v52 = (unsigned __int8 *)(*(_QWORD *)(v127 + 40) + v129);
  v137 = v127;
  v53 = (const void **)*((_QWORD *)v52 + 1);
  v54 = *v52;
  *((_QWORD *)&v118 + 1) = v51;
  *(_QWORD *)&v118 = v50;
  v138 = v128 | v138 & 0xFFFFFFFF00000000LL;
  v55 = sub_1D332F0(
          a6,
          52,
          (__int64)&v143,
          v54,
          v53,
          0,
          *(double *)v12.m128i_i64,
          *(double *)v13.m128i_i64,
          a4,
          v127,
          v138,
          v118);
  LOBYTE(v150) = 0;
  v135 = (__int64)v55;
  v56 = 0;
  v136 = v57;
  v151 = 0;
  v152 = 0;
  v153[0] = 0;
  v149 = (unsigned __int64)v133;
  if ( v133 )
  {
    v58 = *v133;
    if ( *(_BYTE *)(*v133 + 8) == 16 )
      v58 = **(_QWORD **)(v58 + 16);
    v56 = *(_DWORD *)(v58 + 8) >> 8;
  }
  HIDWORD(v150) = v56;
  v141 = sub_1D2BF40(
           a6,
           v43,
           1,
           (__int64)&v143,
           v135,
           v136,
           v132.m128i_i64[0],
           v132.m128i_i64[1],
           v149,
           v150,
           0,
           0,
           (__int64)&v151);
  v130.m128i_i64[0] = v141;
  v142 = v60;
  v130.m128i_i64[1] = (unsigned int)v60 | v130.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v151 = v153;
  v152 = 0x400000000LL;
  if ( v124 == 1 )
  {
    v98 = (__m128i *)v153;
    goto LABEL_80;
  }
  v61 = 200;
  v133 = 0;
  v125 = 40LL * (unsigned int)(v124 - 2) + 240;
  v132.m128i_i64[0] = (__int64)&v149;
  do
  {
    v86 = (unsigned int *)(v61 + *(_QWORD *)(v134 + 32));
    v87 = *(_QWORD *)(*(_QWORD *)v86 + 88LL);
    v88 = *(_QWORD *)(v87 + 24);
    if ( *(_DWORD *)(v87 + 32) > 0x40u )
      v88 = *(_QWORD *)v88;
    v89 = *(_QWORD *)(*(_QWORD *)v86 + 40LL) + 16LL * v86[2];
    v90 = *(_BYTE *)v89;
    v91 = *(const void ***)(v89 + 8);
    LOBYTE(v145) = v90;
    v146 = v91;
    if ( v90 )
      v62 = sub_216FFF0(v90);
    else
      v62 = sub_1F58D40((__int64)&v145);
    LODWORD(v131) = v62 - 1;
    v63 = sub_1D38BB0(
            (__int64)a6,
            v88 & ~(-1LL << ((unsigned __int8)v62 - 1)),
            (__int64)&v143,
            (unsigned int)v145,
            v146,
            1,
            v12,
            *(double *)v13.m128i_i64,
            a4,
            0);
    v64 = *(_QWORD *)(v127 + 40) + v129;
    v137 = v127;
    v65 = *(const void ***)(v64 + 8);
    LOBYTE(v6) = *(_BYTE *)v64;
    *((_QWORD *)&v117 + 1) = v66;
    *(_QWORD *)&v117 = v63;
    v138 = v128 | v138 & 0xFFFFFFFF00000000LL;
    v139 = sub_1D332F0(
             a6,
             52,
             (__int64)&v143,
             v6,
             v65,
             0,
             *(double *)v12.m128i_i64,
             *(double *)v13.m128i_i64,
             a4,
             v127,
             v138,
             v117);
    v135 = (__int64)v139;
    v140 = v67;
    v68 = (_QWORD *)a6[6];
    v136 = (unsigned int)v67 | v136 & 0xFFFFFFFF00000000LL;
    v69 = (__int64)v133 + *(_QWORD *)(v134 + 40);
    LOBYTE(v67) = *(_BYTE *)v69;
    v70 = *(_QWORD *)(v69 + 8);
    LOBYTE(v149) = v67;
    *((_QWORD *)&v149 + 1) = v70;
    v71 = (__int64 *)sub_1F58E60(v132.m128i_i64[0], v68);
    v72 = (__int64 **)sub_1646BA0(v71, 5);
    v75 = (__int64 *)sub_15A06D0(v72, 5, v73, v74);
    v76 = 0;
    v149 = 0u;
    v77 = _bittest64((const __int64 *)&v88, (unsigned int)v131);
    v150 = 0;
    v147 = (unsigned __int64)v75;
    LOBYTE(v148) = 0;
    if ( v77 )
    {
      if ( v75 )
      {
        v78 = *v75;
        if ( *(_BYTE *)(v78 + 8) == 16 )
          v78 = **(_QWORD **)(v78 + 16);
        v76 = *(_DWORD *)(v78 + 8) >> 8;
      }
      HIDWORD(v148) = v76;
      v79 = (__int64)v133 + *(_QWORD *)(v134 + 40);
      v80 = v122;
      LOBYTE(v80) = *(_BYTE *)v79;
      v122 = v80;
      v81 = sub_1D2B810(
              a6,
              1u,
              (__int64)&v143,
              v80,
              *(_QWORD *)(v79 + 8),
              0,
              *(_OWORD *)&v130,
              v135,
              v136,
              v147,
              v148,
              3,
              0,
              0,
              v132.m128i_i64[0]);
      v83 = v82;
      v59 = v81;
      v84 = (unsigned int)v152;
      if ( (unsigned int)v152 >= HIDWORD(v152) )
        goto LABEL_72;
    }
    else
    {
      if ( v75 )
      {
        v92 = *v75;
        if ( *(_BYTE *)(v92 + 8) == 16 )
          v92 = **(_QWORD **)(v92 + 16);
        v76 = *(_DWORD *)(v92 + 8) >> 8;
      }
      HIDWORD(v148) = v76;
      v93 = (__int64)v133 + *(_QWORD *)(v134 + 40);
      v94 = v123;
      LOBYTE(v94) = *(_BYTE *)v93;
      v123 = v94;
      v95 = sub_1D2B730(
              a6,
              (unsigned int)v94,
              *(_QWORD *)(v93 + 8),
              (__int64)&v143,
              v130.m128i_i64[0],
              v130.m128i_i64[1],
              v135,
              v136,
              v147,
              v148,
              0,
              0,
              v132.m128i_i64[0],
              0);
      v83 = v96;
      v59 = v95;
      v84 = (unsigned int)v152;
      if ( (unsigned int)v152 >= HIDWORD(v152) )
      {
LABEL_72:
        *(_QWORD *)&v131 = v59;
        *((_QWORD *)&v131 + 1) = v83;
        sub_16CD150((__int64)&v151, v153, 0, 16, v59, v83);
        v84 = (unsigned int)v152;
        v83 = *((_QWORD *)&v131 + 1);
        v59 = v131;
      }
    }
    v85 = &v151[2 * v84];
    v61 += 40;
    *v85 = v59;
    v85[1] = v83;
    LODWORD(v152) = v152 + 1;
    v133 += 2;
  }
  while ( v61 != v125 );
  v97 = (unsigned int)v152;
  if ( (unsigned int)v152 >= HIDWORD(v152) )
  {
    sub_16CD150((__int64)&v151, v153, 0, 16, v59, v83);
    v97 = (unsigned int)v152;
  }
  v98 = (__m128i *)&v151[2 * v97];
LABEL_80:
  v99 = _mm_load_si128(&v130);
  *v98 = v99;
  LODWORD(v152) = v152 + 1;
  v100 = sub_1D37190(
           (__int64)a6,
           (__int64)v151,
           (unsigned int)v152,
           (__int64)&v143,
           v59,
           *(double *)v12.m128i_i64,
           *(double *)v13.m128i_i64,
           v99);
  if ( v151 != v153 )
    _libc_free((unsigned __int64)v151);
  sub_15A93E0(&v154);
  if ( v143 )
    sub_161E7C0((__int64)&v143, v143);
  return v100;
}
