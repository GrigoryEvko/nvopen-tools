// Function: sub_F39690
// Address: 0xf39690
//
__int64 __fastcall sub_F39690(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, char a6, __int64 a7)
{
  unsigned int v7; // r13d
  __int64 v9; // r15
  __int64 v11; // r12
  int v12; // eax
  unsigned __int8 v13; // bl
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  char *v20; // rax
  char *v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rcx
  char *v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // ecx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // r13
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  void *v34; // r8
  __int64 v35; // r9
  _BYTE *v36; // rax
  __int64 v37; // rbx
  _BYTE *v38; // rsi
  _QWORD **v39; // rbx
  _QWORD **v40; // r12
  _QWORD *v41; // rdi
  unsigned __int64 v42; // rax
  int v43; // esi
  unsigned __int64 v44; // rdi
  int v45; // r15d
  unsigned int i; // r14d
  unsigned int v47; // esi
  __int64 *v48; // rax
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // r13
  __int64 v53; // rdi
  unsigned __int64 v54; // rdi
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 v57; // r8
  __int64 v58; // rcx
  __int64 v59; // rax
  __m128i *v60; // rsi
  unsigned int v61; // r12d
  __int64 v62; // r15
  __m128i *v63; // r14
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  unsigned int v69; // eax
  __int64 v70; // r12
  unsigned int v71; // r15d
  __int64 v72; // r14
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  unsigned __int64 v76; // rax
  __int64 *v77; // r8
  __int64 v78; // r13
  _QWORD *v79; // rdi
  __int64 v80; // r13
  _QWORD *v81; // rdi
  __int64 v82; // rsi
  _QWORD *v83; // rbx
  _QWORD *v84; // r12
  __int64 v85; // rax
  _QWORD *v86; // rdi
  _QWORD *v87; // rdi
  __int64 v88; // rbx
  unsigned __int64 v89; // rax
  __int64 v90; // r8
  int v91; // esi
  __int64 v92; // rdi
  unsigned __int64 v93; // rdx
  int v94; // esi
  unsigned int v95; // ecx
  __int64 *v96; // rax
  __int64 v97; // r9
  __int64 *v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // r8
  __int64 v102; // r12
  __int64 v103; // rbx
  __int64 v104; // rcx
  unsigned __int64 *v105; // r13
  _BYTE *v106; // rsi
  unsigned __int64 v107; // r9
  int v108; // eax
  unsigned __int64 *v109; // rdi
  _BYTE *v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  char *v113; // r13
  _BYTE *v114; // rdi
  int v115; // eax
  __int64 v116; // [rsp+8h] [rbp-218h]
  char v117; // [rsp+8h] [rbp-218h]
  char v118; // [rsp+10h] [rbp-210h]
  _BOOL4 v119; // [rsp+14h] [rbp-20Ch]
  unsigned __int8 *v120; // [rsp+18h] [rbp-208h]
  __int64 v121; // [rsp+20h] [rbp-200h]
  __int64 v122; // [rsp+20h] [rbp-200h]
  __int64 v123; // [rsp+28h] [rbp-1F8h]
  size_t n; // [rsp+30h] [rbp-1F0h]
  char src; // [rsp+40h] [rbp-1E0h]
  void *srca; // [rsp+40h] [rbp-1E0h]
  __int64 v127; // [rsp+48h] [rbp-1D8h]
  __int64 v128; // [rsp+48h] [rbp-1D8h]
  __int64 v129; // [rsp+48h] [rbp-1D8h]
  __int64 v130; // [rsp+50h] [rbp-1D0h]
  __int64 v131; // [rsp+50h] [rbp-1D0h]
  int v132; // [rsp+50h] [rbp-1D0h]
  int v133; // [rsp+50h] [rbp-1D0h]
  __int64 v134; // [rsp+50h] [rbp-1D0h]
  unsigned __int8 *v135; // [rsp+58h] [rbp-1C8h]
  __int64 v140; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v141; // [rsp+88h] [rbp-198h]
  __int64 v142; // [rsp+90h] [rbp-190h]
  __m128i *v143; // [rsp+A0h] [rbp-180h] BYREF
  unsigned int v144; // [rsp+A8h] [rbp-178h]
  int v145; // [rsp+B8h] [rbp-168h]
  __m128i v146[2]; // [rsp+C0h] [rbp-160h] BYREF
  char v147; // [rsp+E0h] [rbp-140h]
  __int64 v148; // [rsp+F0h] [rbp-130h] BYREF
  _BYTE *v149; // [rsp+F8h] [rbp-128h]
  __int64 v150; // [rsp+100h] [rbp-120h]
  int v151; // [rsp+108h] [rbp-118h]
  char v152; // [rsp+10Ch] [rbp-114h]
  _BYTE v153[16]; // [rsp+110h] [rbp-110h] BYREF
  _BYTE *v154; // [rsp+120h] [rbp-100h] BYREF
  void *s; // [rsp+128h] [rbp-F8h]
  _BYTE v156[12]; // [rsp+130h] [rbp-F0h] BYREF
  char v157; // [rsp+13Ch] [rbp-E4h]
  char v158; // [rsp+140h] [rbp-E0h] BYREF
  _BYTE *v159; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v160; // [rsp+188h] [rbp-98h]
  _BYTE v161[144]; // [rsp+190h] [rbp-90h] BYREF

  if ( (*(_WORD *)(a1 + 2) & 0x7FFF) != 0 )
    return 0;
  v9 = a1;
  v11 = sub_AA5510(a1);
  LOBYTE(v7) = a1 == v11 || v11 == 0;
  if ( (_BYTE)v7 )
    return 0;
  v135 = (unsigned __int8 *)sub_986580(v11);
  v12 = *v135;
  v13 = *v135;
  if ( (unsigned int)(v12 - 29) > 6 )
  {
    if ( (unsigned int)(v12 - 37) <= 3 )
      return v7;
  }
  else if ( (unsigned int)(v12 - 29) > 4 )
  {
    return v7;
  }
  v7 = sub_B46970(v135);
  if ( (_BYTE)v7 )
    return 0;
  if ( !a6 )
  {
    if ( a1 == sub_AA5780(v11) )
    {
      v123 = 0;
      goto LABEL_13;
    }
    return 0;
  }
  if ( v13 != 31 )
    return 0;
  v14 = sub_986580(a1);
  if ( *(_BYTE *)v14 != 31 || (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) != 1 )
    return 0;
  v123 = *(_QWORD *)(v14 - 32);
  v120 = v135;
  v119 = *((_QWORD *)v135 - 4) != a1;
LABEL_13:
  v15 = sub_AA5930(a1);
  v17 = v16;
  v18 = v15;
  while ( v17 != v18 )
  {
    v19 = 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
    {
      v20 = *(char **)(v18 - 8);
      v21 = &v20[v19];
    }
    else
    {
      v21 = (char *)v18;
      v20 = (char *)(v18 - v19);
    }
    v22 = v19 >> 5;
    v23 = v19 >> 7;
    if ( v23 )
    {
      v24 = &v20[128 * v23];
      while ( 1 )
      {
        if ( v18 == *(_QWORD *)v20 )
          goto LABEL_24;
        if ( v18 == *((_QWORD *)v20 + 4) )
        {
          v20 += 32;
          goto LABEL_24;
        }
        if ( v18 == *((_QWORD *)v20 + 8) )
        {
          v20 += 64;
          goto LABEL_24;
        }
        if ( v18 == *((_QWORD *)v20 + 12) )
          break;
        v20 += 128;
        if ( v24 == v20 )
        {
          v22 = (v21 - v20) >> 5;
          goto LABEL_32;
        }
      }
      v20 += 96;
      goto LABEL_24;
    }
LABEL_32:
    if ( v22 == 2 )
      goto LABEL_44;
    if ( v22 != 3 )
    {
      if ( v22 != 1 )
        goto LABEL_25;
LABEL_35:
      if ( v18 != *(_QWORD *)v20 )
        goto LABEL_25;
      goto LABEL_24;
    }
    if ( v18 != *(_QWORD *)v20 )
    {
      v20 += 32;
LABEL_44:
      if ( v18 != *(_QWORD *)v20 )
      {
        v20 += 32;
        goto LABEL_35;
      }
    }
LABEL_24:
    if ( v21 != v20 )
      return v7;
LABEL_25:
    v25 = *(_QWORD *)(v18 + 32);
    if ( !v25 )
      goto LABEL_183;
    v18 = 0;
    if ( *(_BYTE *)(v25 - 24) == 84 )
      v18 = v25 - 24;
  }
  v159 = v161;
  v160 = 0x400000000LL;
  v26 = *(_QWORD *)(v9 + 56);
  if ( !v26 )
LABEL_183:
    BUG();
  if ( *(_BYTE *)(v26 - 24) == 84 )
  {
    v99 = sub_AA5930(v9);
    if ( v100 != v99 )
    {
      v134 = v11;
      v102 = v99;
      v103 = v100;
      do
      {
        v112 = **(_QWORD **)(v102 - 8);
        if ( *(_BYTE *)v112 != 84 || v9 != *(_QWORD *)(v112 + 40) )
        {
          v154 = 0;
          s = 0;
          *(_QWORD *)v156 = v112;
          if ( v112 != -4096 && v112 != -8192 )
            sub_BD73F0((__int64)&v154);
          v104 = (unsigned int)v160;
          v105 = (unsigned __int64 *)&v154;
          v106 = v159;
          v107 = (unsigned int)v160 + 1LL;
          v108 = v160;
          if ( v107 > HIDWORD(v160) )
          {
            if ( v159 > (_BYTE *)&v154 || &v154 >= (_BYTE **)&v159[24 * (unsigned int)v160] )
            {
              sub_F39560((__int64)&v159, (unsigned int)v160 + 1LL, v100, (unsigned int)v160, v101, v107);
              v104 = (unsigned int)v160;
              v106 = v159;
              v108 = v160;
            }
            else
            {
              v113 = (char *)((char *)&v154 - v159);
              sub_F39560((__int64)&v159, (unsigned int)v160 + 1LL, v100, (unsigned int)v160, v101, v107);
              v106 = v159;
              v104 = (unsigned int)v160;
              v105 = (unsigned __int64 *)&v113[(_QWORD)v159];
              v108 = v160;
            }
          }
          v109 = (unsigned __int64 *)&v106[24 * v104];
          if ( v109 )
          {
            *v109 = 0;
            v110 = (_BYTE *)v105[2];
            v109[1] = 0;
            v109[2] = (unsigned __int64)v110;
            if ( v110 + 4096 != 0 && v110 != 0 && v110 != (_BYTE *)-8192LL )
              sub_BD6050(v109, *v105 & 0xFFFFFFFFFFFFFFF8LL);
            v108 = v160;
          }
          LODWORD(v160) = v108 + 1;
          if ( *(_QWORD *)v156 != -4096 && *(_QWORD *)v156 != 0 && *(_QWORD *)v156 != -8192 )
            sub_BD60C0(&v154);
        }
        v111 = *(_QWORD *)(v102 + 32);
        if ( !v111 )
          goto LABEL_183;
        v102 = 0;
        if ( *(_BYTE *)(v111 - 24) == 84 )
          v102 = v111 - 24;
      }
      while ( v103 != v102 );
      v11 = v134;
    }
    sub_F34590(v9, a5);
  }
  if ( a7 )
  {
    v27 = *(_DWORD *)(a7 + 32);
    v28 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
    if ( (unsigned int)v28 < v27 )
    {
      v29 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
      v30 = *(_QWORD *)(a7 + 24);
      v31 = *(_QWORD *)(v30 + 8 * v28);
      v32 = 0;
      if ( v27 > (unsigned int)v29 )
        v32 = *(_QWORD *)(v30 + 8 * v29);
      if ( v31 )
      {
        v33 = *(unsigned int *)(v32 + 32);
        v34 = *(void **)(v32 + 24);
        v35 = 8 * v33;
        v154 = v156;
        s = (void *)0x600000000LL;
        if ( v33 > 6 )
        {
          srca = v34;
          sub_C8D5F0((__int64)&v154, v156, v33, 8u, (__int64)v34, v35);
          v34 = srca;
          v35 = 8 * v33;
          v114 = &v154[8 * (unsigned int)s];
        }
        else
        {
          v36 = v156;
          if ( !v35 )
            goto LABEL_55;
          v114 = v156;
        }
        memcpy(v114, v34, v35);
        v36 = v154;
        LODWORD(v35) = (_DWORD)s;
LABEL_55:
        LODWORD(s) = v35 + v33;
        v37 = (unsigned int)(v35 + v33);
        v38 = &v36[8 * v37];
        if ( v36 != v38 )
        {
          v127 = v11;
          v39 = (_QWORD **)&v36[8 * v37];
          v40 = (_QWORD **)v36;
          do
          {
            v41 = *v40;
            v38 = (_BYTE *)v31;
            ++v40;
            sub_B1AE50(v41, v31);
          }
          while ( v39 != v40 );
          v11 = v127;
        }
        if ( v154 != v156 )
          _libc_free(v154, v38);
      }
    }
  }
  v140 = 0;
  v141 = 0;
  v142 = 0;
  if ( a2 )
  {
    v154 = 0;
    s = &v158;
    *(_QWORD *)v156 = 8;
    *(_DWORD *)&v156[8] = 0;
    v157 = 1;
    v42 = *(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v42 == v11 + 48 )
      goto LABEL_172;
    if ( !v42 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
    {
LABEL_172:
      v148 = 0;
      v149 = v153;
      v150 = 2;
      v151 = 0;
      v152 = 1;
    }
    else
    {
      v152 = 1;
      v43 = sub_B46E30(v42 - 24);
      v148 = 0;
      v44 = sub_986580(v11);
      v149 = v153;
      v150 = 2;
      v151 = 0;
      if ( v43 )
      {
        v128 = v9;
        v45 = v43;
        src = a6;
        for ( i = 0; i != v45; ++i )
        {
          v47 = i;
          v48 = (__int64 *)sub_B46EC0(v44, v47);
          sub_D695C0((__int64)v146, (__int64)&v148, v48, v49, v50, v51);
        }
        v52 = v140;
        v9 = v128;
        a6 = src;
        v53 = v128;
        v129 = v141 - v140;
        v130 = ((v141 - v140) >> 4) + 1;
        v54 = sub_986580(v53);
        v55 = 0;
        if ( v54 )
LABEL_69:
          v55 = 2 * (unsigned int)sub_B46E30(v54);
        v56 = v55 + v130;
        if ( (unsigned __int64)(v55 + v130) > 0x7FFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::reserve");
        if ( v56 > (v142 - v52) >> 4 )
        {
          v57 = 0;
          v58 = 16 * v56;
          if ( v56 )
            goto LABEL_73;
          goto LABEL_74;
        }
        goto LABEL_75;
      }
    }
    v54 = sub_986580(v9);
    if ( v54 )
    {
      v130 = 1;
      v52 = 0;
      v129 = 0;
      goto LABEL_69;
    }
    v129 = 0;
    v58 = 16;
LABEL_73:
    v131 = v58;
    v59 = sub_22077B0(v58);
    v58 = v131;
    v57 = v59;
LABEL_74:
    v140 = v57;
    v142 = v57 + v58;
    v141 = v57 + v129;
LABEL_75:
    sub_F34070((__int64)&v143, v9);
    v60 = v143;
    v132 = v145;
    if ( v144 != v145 )
    {
      v121 = v11;
      v61 = v144;
      v116 = v9;
      v62 = (__int64)v143;
      v118 = a6;
      do
      {
        v60 = (__m128i *)sub_B46EC0(v62, v61);
        v63 = v60;
        if ( !(unsigned __int8)sub_B19060((__int64)&v148, (__int64)v60, v64, v65) )
        {
          v60 = (__m128i *)&v154;
          sub_D695C0((__int64)v146, (__int64)&v154, v63->m128i_i64, v66, v67, v68);
          if ( v147 )
          {
            v60 = v146;
            v146[0].m128i_i64[0] = v121;
            v146[0].m128i_i64[1] = (unsigned __int64)v63 & 0xFFFFFFFFFFFFFFFBLL;
            sub_F38D20((__int64)&v140, v146);
          }
        }
        ++v61;
      }
      while ( v132 != v61 );
      v11 = v121;
      v9 = v116;
      a6 = v118;
    }
    ++v154;
    if ( v157 )
    {
LABEL_87:
      *(_QWORD *)&v156[4] = 0;
    }
    else
    {
      v69 = 4 * (*(_DWORD *)&v156[4] - *(_DWORD *)&v156[8]);
      if ( v69 < 0x20 )
        v69 = 32;
      if ( v69 >= *(_DWORD *)v156 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v156);
        goto LABEL_87;
      }
      sub_C8C990((__int64)&v154, (__int64)v60);
    }
    sub_F34070((__int64)&v143, v9);
    v133 = v145;
    if ( v145 != v144 )
    {
      v122 = v11;
      v70 = (__int64)v143;
      n = v9;
      v71 = v144;
      v117 = a6;
      do
      {
        v72 = sub_B46EC0(v70, v71);
        sub_D695C0((__int64)v146, (__int64)&v154, (__int64 *)v72, v73, v74, v75);
        if ( v147 )
        {
          v146[0].m128i_i64[0] = n;
          v146[0].m128i_i64[1] = v72 | 4;
          sub_F38D20((__int64)&v140, v146);
        }
        ++v71;
      }
      while ( v133 != v71 );
      v11 = v122;
      v9 = n;
      a6 = v117;
    }
    v146[0].m128i_i64[0] = v11;
    v146[0].m128i_i64[1] = v9 | 4;
    sub_F38D20((__int64)&v140, v146);
    if ( !v152 )
      _libc_free(v149, v146);
    if ( !v157 )
      _libc_free(s, v146);
  }
  v76 = sub_986580(v9);
  v77 = *(__int64 **)(v9 + 56);
  v78 = (__int64)(v77 - 3);
  if ( !v77 )
    v78 = 0;
  if ( v78 == v76 )
    v78 = (__int64)v135;
  sub_AA80F0(v11, (unsigned __int64 *)v135 + 3, 0, v9, v77, 1, (__int64 *)(v76 + 24), 0);
  if ( a4 )
    sub_D6E030((__int64)a4, v9, v11, v78);
  sub_BD84D0(v9, v11);
  if ( a6 )
  {
    v79 = (_QWORD *)((*(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24);
    if ( (*(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v79 = 0;
    sub_B43D60(v79);
    sub_AC2B30((__int64)&v120[-32 * v119 - 32], v123);
  }
  else
  {
    v86 = (_QWORD *)((*(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24);
    if ( (*(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v86 = 0;
    sub_B43D60(v86);
    v87 = (_QWORD *)((*(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24);
    if ( (*(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v87 = 0;
    sub_B44560(v87, v11, (unsigned __int64 *)(v11 + 48), 0);
    if ( a4 )
    {
      v88 = *a4;
      v89 = sub_986580(v11);
      v91 = *(_DWORD *)(v88 + 56);
      v92 = *(_QWORD *)(v88 + 40);
      v93 = v89;
      if ( v91 )
      {
        v94 = v91 - 1;
        v95 = v94 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
        v96 = (__int64 *)(v92 + 16LL * v95);
        v97 = *v96;
        if ( v93 == *v96 )
        {
LABEL_138:
          v98 = (__int64 *)v96[1];
          if ( v98 )
            sub_D75590(a4, v98, v11, 1, v90, v97);
        }
        else
        {
          v115 = 1;
          while ( v97 != -4096 )
          {
            v90 = (unsigned int)(v115 + 1);
            v95 = v94 & (v115 + v95);
            v96 = (__int64 *)(v92 + 16LL * v95);
            v97 = *v96;
            if ( v93 == *v96 )
              goto LABEL_138;
            v115 = v90;
          }
        }
      }
    }
  }
  v80 = sub_AA48A0(v9);
  sub_B43C20((__int64)&v154, v9);
  v81 = sub_BD2C40(72, unk_3F148B8);
  if ( v81 )
    sub_B4C8A0((__int64)v81, v80, (__int64)v154, (unsigned __int16)s);
  if ( (*(_BYTE *)(v11 + 7) & 0x10) == 0 )
    sub_BD6B90((unsigned __int8 *)v11, (unsigned __int8 *)v9);
  if ( a3 )
    sub_D48300(a3, v9);
  if ( a5 )
    sub_102BA10(a5);
  if ( a2 )
    sub_FFB3D0(a2, v140, (v141 - v140) >> 4);
  if ( a7 )
    sub_B19380(a7, v9);
  v82 = a2;
  sub_F34560(v9, a2, 0);
  if ( v140 )
  {
    v82 = v142 - v140;
    j_j___libc_free_0(v140, v142 - v140);
  }
  v83 = v159;
  v84 = &v159[24 * (unsigned int)v160];
  if ( v159 != (_BYTE *)v84 )
  {
    do
    {
      v85 = *(v84 - 1);
      v84 -= 3;
      if ( v85 != 0 && v85 != -4096 && v85 != -8192 )
        sub_BD60C0(v84);
    }
    while ( v83 != v84 );
    v84 = v159;
  }
  if ( v84 != (_QWORD *)v161 )
    _libc_free(v84, v82);
  return 1;
}
