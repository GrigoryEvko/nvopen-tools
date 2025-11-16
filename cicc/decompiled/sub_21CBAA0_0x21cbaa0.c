// Function: sub_21CBAA0
// Address: 0x21cbaa0
//
__int64 __fastcall sub_21CBAA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9,
        _QWORD *a10,
        __int64 a11,
        __int64 *a12)
{
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v15; // rdi
  __int64 v16; // rbx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r10
  unsigned __int64 v20; // r11
  __int64 v21; // rdx
  unsigned int v22; // eax
  const void *v23; // rsi
  unsigned __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  int v29; // r12d
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  int v34; // r12d
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // r9d
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // ecx
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r12
  __int64 v47; // rdx
  unsigned __int64 v48; // r13
  int v49; // edx
  __int64 v50; // rax
  __int64 *v51; // rax
  char *v52; // rcx
  int v53; // edx
  __int64 v54; // rax
  unsigned int v55; // r12d
  __int64 v56; // rax
  int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int8 v61; // dl
  __int64 v62; // rax
  __int64 v63; // rax
  unsigned int v64; // edx
  unsigned int v65; // eax
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // r12
  __int64 v73; // rsi
  unsigned __int64 v74; // r13
  unsigned __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rbx
  __int64 v78; // rax
  __int128 v79; // [rsp-20h] [rbp-580h]
  __int64 v80; // [rsp-10h] [rbp-570h]
  __int128 v81; // [rsp-10h] [rbp-570h]
  __int64 v82; // [rsp-8h] [rbp-568h]
  unsigned int v83; // [rsp+10h] [rbp-550h]
  __int64 v84; // [rsp+10h] [rbp-550h]
  __int64 v85; // [rsp+18h] [rbp-548h]
  __int64 v86; // [rsp+20h] [rbp-540h]
  __int64 v88; // [rsp+38h] [rbp-528h]
  __int64 v89; // [rsp+40h] [rbp-520h]
  __int64 v90; // [rsp+40h] [rbp-520h]
  __m128 v91; // [rsp+40h] [rbp-520h]
  __int64 v92; // [rsp+40h] [rbp-520h]
  __int64 v93; // [rsp+40h] [rbp-520h]
  __int64 v94; // [rsp+40h] [rbp-520h]
  __int64 v95; // [rsp+40h] [rbp-520h]
  __int64 v96; // [rsp+40h] [rbp-520h]
  int v97; // [rsp+40h] [rbp-520h]
  __int64 v98; // [rsp+40h] [rbp-520h]
  __int64 v99; // [rsp+50h] [rbp-510h]
  int v101; // [rsp+58h] [rbp-508h]
  int v102; // [rsp+58h] [rbp-508h]
  __int64 v103; // [rsp+58h] [rbp-508h]
  __int64 v104; // [rsp+58h] [rbp-508h]
  bool v105; // [rsp+58h] [rbp-508h]
  unsigned int v106; // [rsp+58h] [rbp-508h]
  __int64 v107; // [rsp+58h] [rbp-508h]
  __int64 v108; // [rsp+58h] [rbp-508h]
  int v109; // [rsp+58h] [rbp-508h]
  int v110; // [rsp+58h] [rbp-508h]
  int v111; // [rsp+58h] [rbp-508h]
  __int64 v112; // [rsp+58h] [rbp-508h]
  __int64 v113; // [rsp+58h] [rbp-508h]
  __int64 v114; // [rsp+58h] [rbp-508h]
  __int64 v115; // [rsp+58h] [rbp-508h]
  int v116; // [rsp+58h] [rbp-508h]
  __int64 v117; // [rsp+58h] [rbp-508h]
  __int64 *v118; // [rsp+60h] [rbp-500h]
  __int64 v119; // [rsp+68h] [rbp-4F8h]
  __m128i v120; // [rsp+70h] [rbp-4F0h] BYREF
  __int64 v121; // [rsp+80h] [rbp-4E0h]
  __int64 v122; // [rsp+88h] [rbp-4D8h]
  __int64 v123; // [rsp+90h] [rbp-4D0h]
  __int64 v124; // [rsp+98h] [rbp-4C8h]
  __int64 v125; // [rsp+A0h] [rbp-4C0h]
  __int64 v126; // [rsp+A8h] [rbp-4B8h]
  __int64 v127; // [rsp+B0h] [rbp-4B0h]
  __int64 v128; // [rsp+B8h] [rbp-4A8h]
  __int128 v129; // [rsp+C0h] [rbp-4A0h]
  __int64 v130; // [rsp+D0h] [rbp-490h]
  __m128i v131; // [rsp+E0h] [rbp-480h] BYREF
  __int64 v132; // [rsp+F0h] [rbp-470h]
  char *v133; // [rsp+100h] [rbp-460h] BYREF
  char v134; // [rsp+110h] [rbp-450h] BYREF
  __int64 *v135; // [rsp+150h] [rbp-410h] BYREF
  __int64 v136; // [rsp+158h] [rbp-408h]
  _BYTE v137[96]; // [rsp+160h] [rbp-400h] BYREF
  __int64 v138[2]; // [rsp+1C0h] [rbp-3A0h] BYREF
  _BYTE v139[128]; // [rsp+1D0h] [rbp-390h] BYREF
  _BYTE *v140; // [rsp+250h] [rbp-310h] BYREF
  __int64 v141; // [rsp+258h] [rbp-308h]
  _BYTE v142[256]; // [rsp+260h] [rbp-300h] BYREF
  __int64 v143; // [rsp+360h] [rbp-200h] BYREF
  int v144; // [rsp+368h] [rbp-1F8h]
  int v145; // [rsp+36Ch] [rbp-1F4h]
  int v146; // [rsp+370h] [rbp-1F0h]
  void *dest; // [rsp+378h] [rbp-1E8h] BYREF
  size_t n; // [rsp+380h] [rbp-1E0h]
  char v149[8]; // [rsp+388h] [rbp-1D8h] BYREF
  void *v150; // [rsp+390h] [rbp-1D0h] BYREF
  __int64 v151; // [rsp+398h] [rbp-1C8h]
  _BYTE v152[128]; // [rsp+3A0h] [rbp-1C0h] BYREF
  _QWORD v153[2]; // [rsp+420h] [rbp-140h] BYREF
  char v154; // [rsp+430h] [rbp-130h] BYREF
  void *v155; // [rsp+440h] [rbp-120h] BYREF
  __int64 v156; // [rsp+448h] [rbp-118h]
  _BYTE v157[160]; // [rsp+450h] [rbp-110h] BYREF
  __int64 v158; // [rsp+4F0h] [rbp-70h]
  void *v159; // [rsp+4F8h] [rbp-68h] BYREF
  __int64 v160; // [rsp+500h] [rbp-60h]
  _BYTE v161[88]; // [rsp+508h] [rbp-58h] BYREF

  v12 = *(_QWORD *)(a1 + 81552);
  v120.m128i_i64[0] = a2;
  v120.m128i_i64[1] = a3;
  if ( *(_DWORD *)(v12 + 252) <= 0x13u )
    return v120.m128i_i64[0];
  v15 = a12[4];
  v99 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v15 + 24LL) + 16LL);
  v16 = sub_1E0A0C0(v15);
  dest = v149;
  n = 0x800000000LL;
  v156 = 0x800000000LL;
  v160 = 0x800000000LL;
  v151 = 0x1000000000LL;
  v153[0] = &v154;
  v159 = v161;
  v150 = v152;
  v153[1] = 0;
  v154 = 0;
  v155 = v157;
  v158 = 0;
  sub_15A9210((__int64)&v143);
  sub_2240AE0(v153, v16 + 192);
  v19 = a1;
  LOBYTE(v143) = *(_BYTE *)v16;
  HIDWORD(v143) = *(_DWORD *)(v16 + 4);
  v144 = *(_DWORD *)(v16 + 8);
  v145 = *(_DWORD *)(v16 + 12);
  v146 = *(_DWORD *)(v16 + 16);
  if ( &dest != (void **)(v16 + 24) )
  {
    v20 = *(unsigned int *)(v16 + 32);
    v21 = (unsigned int)n;
    v18 = (unsigned int)v149;
    v17 = *(_DWORD *)(v16 + 32);
    if ( v20 <= (unsigned int)n )
    {
      if ( *(_DWORD *)(v16 + 32) )
      {
        v93 = a1;
        v106 = *(_DWORD *)(v16 + 32);
        memmove(dest, *(const void **)(v16 + 24), v106);
        v19 = v93;
        v17 = v106;
      }
    }
    else
    {
      if ( v20 > HIDWORD(n) )
      {
        v75 = *(unsigned int *)(v16 + 32);
        v96 = a1;
        v111 = *(_DWORD *)(v16 + 32);
        LODWORD(n) = 0;
        sub_16CD150((__int64)&dest, v149, v75, 1, v17, v18);
        v20 = *(unsigned int *)(v16 + 32);
        v17 = v111;
        v21 = 0;
        v19 = v96;
        v22 = *(_DWORD *)(v16 + 32);
      }
      else
      {
        v22 = *(_DWORD *)(v16 + 32);
        if ( (_DWORD)n )
        {
          v86 = a1;
          v97 = *(_DWORD *)(v16 + 32);
          v115 = (unsigned int)n;
          memmove(dest, *(const void **)(v16 + 24), (unsigned int)n);
          v20 = *(unsigned int *)(v16 + 32);
          v19 = v86;
          v17 = v97;
          v21 = v115;
          v22 = *(_DWORD *)(v16 + 32);
        }
      }
      v23 = (const void *)(v21 + *(_QWORD *)(v16 + 24));
      if ( v23 != (const void *)(*(_QWORD *)(v16 + 24) + v20) )
      {
        v89 = v19;
        v101 = v17;
        memcpy((char *)dest + v21, v23, v22 - v21);
        v19 = v89;
        v17 = v101;
      }
    }
    LODWORD(n) = v17;
  }
  if ( &v150 != (void **)(v16 + 48) )
  {
    v24 = *(unsigned int *)(v16 + 56);
    v17 = *(_DWORD *)(v16 + 56);
    if ( v24 <= (unsigned int)v151 )
    {
      if ( *(_DWORD *)(v16 + 56) )
      {
        v109 = *(_DWORD *)(v16 + 56);
        v94 = v19;
        memmove(v150, *(const void **)(v16 + 48), 8 * v24);
        v19 = v94;
        v17 = v109;
      }
    }
    else
    {
      if ( v24 > HIDWORD(v151) )
      {
        v95 = v19;
        v25 = 0;
        v110 = *(_DWORD *)(v16 + 56);
        LODWORD(v151) = 0;
        sub_16CD150((__int64)&v150, v152, v24, 8, v17, v18);
        v24 = *(unsigned int *)(v16 + 56);
        v17 = v110;
        v19 = v95;
      }
      else
      {
        v25 = 8LL * (unsigned int)v151;
        if ( (_DWORD)v151 )
        {
          v116 = *(_DWORD *)(v16 + 56);
          v98 = v19;
          memmove(v150, *(const void **)(v16 + 48), 8LL * (unsigned int)v151);
          v24 = *(unsigned int *)(v16 + 56);
          v19 = v98;
          v17 = v116;
        }
      }
      v26 = *(_QWORD *)(v16 + 48);
      v27 = 8 * v24;
      if ( v26 + v25 != v27 + v26 )
      {
        v90 = v19;
        v102 = v17;
        memcpy((char *)v150 + v25, (const void *)(v26 + v25), v27 - v25);
        v19 = v90;
        v17 = v102;
      }
    }
    LODWORD(v151) = v17;
  }
  if ( &v155 != (void **)(v16 + 224) )
  {
    v28 = *(unsigned int *)(v16 + 232);
    v29 = *(_DWORD *)(v16 + 232);
    if ( v28 <= (unsigned int)v156 )
    {
      if ( *(_DWORD *)(v16 + 232) )
      {
        v107 = v19;
        memmove(v155, *(const void **)(v16 + 224), 20 * v28);
        v19 = v107;
      }
    }
    else
    {
      if ( v28 > HIDWORD(v156) )
      {
        v30 = 0;
        v112 = v19;
        LODWORD(v156) = 0;
        sub_16CD150((__int64)&v155, v157, v28, 20, v17, v18);
        v28 = *(unsigned int *)(v16 + 232);
        v19 = v112;
      }
      else
      {
        v30 = 20LL * (unsigned int)v156;
        if ( (_DWORD)v156 )
        {
          v117 = v19;
          memmove(v155, *(const void **)(v16 + 224), 20LL * (unsigned int)v156);
          v28 = *(unsigned int *)(v16 + 232);
          v19 = v117;
        }
      }
      v31 = *(_QWORD *)(v16 + 224);
      v32 = 20 * v28;
      if ( v31 + v30 != v32 + v31 )
      {
        v103 = v19;
        memcpy((char *)v155 + v30, (const void *)(v31 + v30), v32 - v30);
        v19 = v103;
      }
    }
    LODWORD(v156) = v29;
  }
  if ( &v159 != (void **)(v16 + 408) )
  {
    v33 = *(unsigned int *)(v16 + 416);
    v34 = *(_DWORD *)(v16 + 416);
    if ( v33 <= (unsigned int)v160 )
    {
      if ( *(_DWORD *)(v16 + 416) )
      {
        v108 = v19;
        memmove(v159, *(const void **)(v16 + 408), 4 * v33);
        v19 = v108;
      }
    }
    else
    {
      if ( v33 > HIDWORD(v160) )
      {
        v35 = 0;
        v113 = v19;
        LODWORD(v160) = 0;
        sub_16CD150((__int64)&v159, v161, v33, 4, v17, v18);
        v33 = *(unsigned int *)(v16 + 416);
        v19 = v113;
      }
      else
      {
        v35 = 4LL * (unsigned int)v160;
        if ( (_DWORD)v160 )
        {
          v114 = v19;
          memmove(v159, *(const void **)(v16 + 408), 4LL * (unsigned int)v160);
          v33 = *(unsigned int *)(v16 + 416);
          v19 = v114;
        }
      }
      v36 = *(_QWORD *)(v16 + 408);
      v37 = 4 * v33;
      if ( v36 + v35 != v37 + v36 )
      {
        v104 = v19;
        memcpy((char *)v159 + v35, (const void *)(v36 + v35), v37 - v35);
        v19 = v104;
      }
    }
    LODWORD(v160) = v34;
  }
  v138[0] = (__int64)v139;
  v140 = v142;
  v141 = 0x1000000000LL;
  v138[1] = 0x1000000000LL;
  sub_21CAE40(v19, (__int64)&v143, v99, (__int64)&v140, (__int64)v138, 0);
  v39 = *(unsigned __int8 *)(v99 + 8);
  if ( (unsigned __int8)v39 <= 0xFu && (v40 = 35454, _bittest64(&v40, v39))
    || ((unsigned int)(v39 - 13) <= 1 || (_DWORD)v39 == 16) && sub_16435F0(v99, 0) )
  {
    v41 = sub_15A9FE0((__int64)&v143, v99);
  }
  else
  {
    v41 = 1;
  }
  sub_21CB650(&v133, (__int64 *)&v140, v138, v41, 0, v38);
  v105 = 0;
  if ( *(_BYTE *)(v99 + 8) == 11 )
  {
    v72 = 1;
    v73 = v99;
    v74 = (unsigned int)sub_15A9FE0((__int64)&v143, v99);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v73 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v78 = *(_QWORD *)(v73 + 32);
          v73 = *(_QWORD *)(v73 + 24);
          v72 *= v78;
          continue;
        case 1:
          v76 = 16;
          break;
        case 2:
          v76 = 32;
          break;
        case 3:
        case 9:
          v76 = 64;
          break;
        case 4:
          v76 = 80;
          break;
        case 5:
        case 6:
          v76 = 128;
          break;
        case 7:
          v76 = 8 * (unsigned int)sub_15A9520((__int64)&v143, 0);
          break;
        case 0xB:
          v76 = *(_DWORD *)(v73 + 8) >> 8;
          break;
        case 0xD:
          v76 = 8LL * *(_QWORD *)sub_15A9930((__int64)&v143, v73);
          break;
        case 0xE:
          v77 = *(_QWORD *)(v73 + 32);
          v76 = 8 * sub_12BE0A0((__int64)&v143, *(_QWORD *)(v73 + 24)) * v77;
          break;
        case 0xF:
          v76 = 8 * (unsigned int)sub_15A9520((__int64)&v143, *(_DWORD *)(v73 + 8) >> 8);
          break;
      }
      break;
    }
    v105 = 8 * v74 * ((v74 + ((unsigned __int64)(v76 * v72 + 7) >> 3) - 1) / v74) <= 0x1F;
  }
  v135 = (__int64 *)v137;
  v88 = 4LL * (unsigned int)v141;
  v42 = 0;
  v136 = 0x600000000LL;
  if ( (_DWORD)v141 )
  {
    do
    {
      v92 = 4 * v42;
      v46 = *(_QWORD *)(*a10 + 4 * v42);
      v48 = *(_QWORD *)(*a10 + 4 * v42 + 8);
      if ( v105 )
      {
        v43 = sub_1D309E0(
                a12,
                (unsigned int)((*(_BYTE *)(*a9 + 12 * v42) & 2) == 0) + 142,
                a11,
                5,
                0,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(double *)a6.m128i_i64,
                *(_OWORD *)(*a10 + 4 * v42));
        v44 = v80;
        v45 = v82;
        v127 = v43;
        v46 = v43;
        v128 = v47;
        v48 = (unsigned int)v47 | v48 & 0xFFFFFFFF00000000LL;
        v49 = *(_DWORD *)&v133[v42];
        if ( v49 != 3 )
          goto LABEL_43;
        v131 = (__m128i)5uLL;
        goto LABEL_57;
      }
      v60 = *(_QWORD *)(v46 + 40) + 16LL * (unsigned int)v48;
      v61 = *(_BYTE *)v60;
      v62 = *(_QWORD *)(v60 + 8);
      v131.m128i_i8[0] = v61;
      v131.m128i_i64[1] = v62;
      if ( v61 )
      {
        if ( (unsigned int)sub_1F3E310(&v131) <= 0xF )
          goto LABEL_81;
      }
      else if ( (unsigned int)sub_1F58D40((__int64)&v131) <= 0xF )
      {
LABEL_81:
        *((_QWORD *)&v81 + 1) = v48;
        *(_QWORD *)&v81 = v46;
        v125 = sub_1D309E0(
                 a12,
                 144,
                 a11,
                 4,
                 0,
                 0,
                 *(double *)a4.m128i_i64,
                 *(double *)a5.m128i_i64,
                 *(double *)a6.m128i_i64,
                 v81);
        v46 = v125;
        v126 = v70;
        v48 = (unsigned int)v70 | v48 & 0xFFFFFFFF00000000LL;
      }
      v49 = *(_DWORD *)&v133[v42];
      if ( v49 != 3 )
      {
LABEL_43:
        v50 = (unsigned int)v136;
        if ( (v49 & 1) != 0 )
          goto LABEL_60;
        goto LABEL_44;
      }
      a6 = _mm_loadu_si128((const __m128i *)&v140[v92]);
      v131 = a6;
LABEL_57:
      v63 = sub_1F58E60((__int64)&v131, *(_QWORD **)v99);
      v64 = sub_15A9FE0((__int64)&v143, v63);
      if ( (unsigned int)*(unsigned __int8 *)(v99 + 8) - 13 > 1
        || (v83 = v64,
            v65 = sub_15A9FE0((__int64)&v143, v99),
            v66 = *(_QWORD *)(v138[0] + 2 * v42),
            !((v66 + (unsigned __int64)v65) % v83)) )
      {
        v50 = (unsigned int)v136;
        if ( (*(_DWORD *)&v133[v42] & 1) != 0 )
        {
LABEL_60:
          if ( (unsigned int)v50 >= HIDWORD(v136) )
          {
            sub_16CD150((__int64)&v135, v137, 0, 16, v44, v45);
            v50 = (unsigned int)v136;
          }
          a4 = _mm_load_si128(&v120);
          *(__m128i *)&v135[2 * v50] = a4;
          LODWORD(v136) = v136 + 1;
          v44 = sub_1D38BB0(
                  (__int64)a12,
                  *(_QWORD *)(v138[0] + 2 * v42),
                  a11,
                  5,
                  0,
                  0,
                  a4,
                  *(double *)a5.m128i_i64,
                  a6,
                  0);
          v45 = v67;
          v68 = (unsigned int)v136;
          if ( (unsigned int)v136 >= HIDWORD(v136) )
          {
            v84 = v44;
            v85 = v67;
            sub_16CD150((__int64)&v135, v137, 0, 16, v44, v67);
            v68 = (unsigned int)v136;
            v44 = v84;
            v45 = v85;
          }
          v69 = &v135[2 * v68];
          *v69 = v44;
          v69[1] = v45;
          v50 = (unsigned int)(v136 + 1);
          LODWORD(v136) = v50;
          if ( HIDWORD(v136) <= (unsigned int)v50 )
          {
LABEL_65:
            sub_16CD150((__int64)&v135, v137, 0, 16, v44, v45);
            v50 = (unsigned int)v136;
          }
LABEL_45:
          v51 = &v135[2 * v50];
          *v51 = v46;
          v52 = v133;
          v51[1] = v48;
          v53 = v136;
          v54 = (unsigned int)(v136 + 1);
          LODWORD(v136) = v136 + 1;
          if ( (v52[v42] & 2) != 0 )
          {
            v55 = 676;
            if ( v53 != 3 )
              v55 = 2 * (v53 == 5) + 675;
            if ( v105 )
            {
              v91 = (__m128)5uLL;
            }
            else
            {
              a5 = _mm_loadu_si128((const __m128i *)&v140[v92]);
              v91 = (__m128)a5;
            }
            v131 = 0u;
            v118 = v135;
            v132 = 0;
            v129 = 0u;
            v130 = 0;
            v119 = v54;
            v56 = sub_1D29190((__int64)a12, 1u, 0, (__int64)v135, v44, v45);
            v58 = sub_1D251C0(
                    a12,
                    v55,
                    a11,
                    v56,
                    v57,
                    1,
                    v118,
                    v119,
                    v91.m128_i64[0],
                    v91.m128_i64[1],
                    v129,
                    v130,
                    2u,
                    0,
                    (__int64)&v131);
            LODWORD(v136) = 0;
            v121 = v58;
            v120.m128i_i64[0] = v58;
            v122 = v59;
            v120.m128i_i64[1] = (unsigned int)v59 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          }
          goto LABEL_51;
        }
LABEL_44:
        if ( HIDWORD(v136) <= (unsigned int)v50 )
          goto LABEL_65;
        goto LABEL_45;
      }
      *((_QWORD *)&v79 + 1) = v48;
      *(_QWORD *)&v79 = v46;
      v123 = sub_2176490(
               a12,
               v120.m128i_i64[0],
               v120.m128i_i64[1],
               v66,
               v131.m128i_u32[0],
               (const void **)v131.m128i_i64[1],
               a4,
               *(double *)a5.m128i_i64,
               a6,
               v79,
               a11);
      v120.m128i_i64[0] = v123;
      v124 = v71;
      v120.m128i_i64[1] = (unsigned int)v71 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL;
LABEL_51:
      v42 += 4;
    }
    while ( v88 != v42 );
  }
  v13 = sub_1D309E0(
          a12,
          262,
          a11,
          1,
          0,
          0,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          *(_OWORD *)&v120);
  if ( v135 != (__int64 *)v137 )
    _libc_free((unsigned __int64)v135);
  if ( v133 != &v134 )
    _libc_free((unsigned __int64)v133);
  if ( (_BYTE *)v138[0] != v139 )
    _libc_free(v138[0]);
  if ( v140 != v142 )
    _libc_free((unsigned __int64)v140);
  sub_15A93E0(&v143);
  return v13;
}
