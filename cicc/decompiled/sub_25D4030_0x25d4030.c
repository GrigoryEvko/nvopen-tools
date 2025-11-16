// Function: sub_25D4030
// Address: 0x25d4030
//
bool *__fastcall sub_25D4030(bool *a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  const __m128i **v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  const __m128i *v12; // rbx
  __m128i *v13; // r12
  __int64 v14; // rsi
  char v15; // dl
  char v16; // al
  _QWORD **v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int64 v20; // rax
  _QWORD **v21; // r12
  unsigned int v22; // eax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  __int64 v25; // rsi
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 *v28; // rbx
  __int64 v29; // r14
  __int64 v30; // rsi
  __int64 v31; // r12
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // rcx
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r15
  unsigned __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  void *v48; // rcx
  _QWORD *v49; // rdx
  char v50; // cl
  __int64 v51; // r10
  int v52; // eax
  char v53; // cl
  char v54; // dl
  char v55; // al
  unsigned __int8 *v56; // r10
  unsigned int v57; // eax
  _QWORD *v58; // rbx
  __int64 v59; // rax
  unsigned __int64 v60; // r12
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r10
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 *v66; // rdi
  __int64 v67; // rax
  unsigned __int8 *v68; // rsi
  __int64 v69; // rdi
  char **v70; // rcx
  __int64 v71; // r8
  __int64 v72; // rdx
  __int64 *v73; // rdi
  __int64 v74; // rsi
  __int64 v75; // rdx
  __m128i *v76; // rdi
  int v77; // r13d
  __int64 v78; // rcx
  const __m128i **v79; // r12
  bool v80; // bl
  __int64 i; // r12
  __int64 v82; // r15
  __int64 (__fastcall **v83)(); // rbx
  unsigned __int64 v84; // rax
  bool v85; // zf
  char v86; // cl
  unsigned int v88; // eax
  _QWORD *v89; // rbx
  unsigned __int8 *v90; // r13
  _QWORD *v91; // r12
  __int64 v92; // rsi
  __int64 *v93; // r13
  const char *v94; // rax
  size_t v95; // rdx
  _WORD *v96; // rdi
  unsigned __int8 *v97; // rsi
  unsigned __int64 v98; // rax
  _BYTE *v99; // rax
  __int64 v100; // r15
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // r12
  __int64 v105; // rax
  __int64 v106; // rax
  bool v107; // al
  unsigned __int64 v108; // rbx
  __int64 v109; // [rsp-10h] [rbp-250h]
  __int64 v110; // [rsp-8h] [rbp-248h]
  __m128i *v111; // [rsp+8h] [rbp-238h]
  __m128i *v112; // [rsp+10h] [rbp-230h]
  __int64 *v113; // [rsp+10h] [rbp-230h]
  const __m128i *v114; // [rsp+18h] [rbp-228h]
  __m128i *v115; // [rsp+18h] [rbp-228h]
  const __m128i *v116; // [rsp+28h] [rbp-218h]
  int v117; // [rsp+3Ch] [rbp-204h]
  unsigned __int64 v121; // [rsp+60h] [rbp-1E0h]
  __int64 v122; // [rsp+68h] [rbp-1D8h]
  __int64 v123; // [rsp+68h] [rbp-1D8h]
  __int64 v124; // [rsp+68h] [rbp-1D8h]
  __int64 v125; // [rsp+68h] [rbp-1D8h]
  unsigned __int8 *v126; // [rsp+68h] [rbp-1D8h]
  unsigned __int8 *v127; // [rsp+68h] [rbp-1D8h]
  unsigned __int8 *v128; // [rsp+68h] [rbp-1D8h]
  __int64 v129; // [rsp+68h] [rbp-1D8h]
  const __m128i *v130; // [rsp+68h] [rbp-1D8h]
  unsigned __int8 *v131; // [rsp+68h] [rbp-1D8h]
  size_t v132; // [rsp+68h] [rbp-1D8h]
  unsigned __int64 v133; // [rsp+70h] [rbp-1D0h]
  unsigned __int64 v134; // [rsp+70h] [rbp-1D0h]
  __int64 *v135; // [rsp+70h] [rbp-1D0h]
  char v136; // [rsp+70h] [rbp-1D0h]
  __m128i *v137; // [rsp+70h] [rbp-1D0h]
  __int64 v138; // [rsp+88h] [rbp-1B8h] BYREF
  unsigned __int64 v139; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v140; // [rsp+98h] [rbp-1A8h] BYREF
  const __m128i *v141; // [rsp+A0h] [rbp-1A0h] BYREF
  int v142; // [rsp+A8h] [rbp-198h]
  _QWORD **v143; // [rsp+B0h] [rbp-190h] BYREF
  char v144; // [rsp+B8h] [rbp-188h]
  void *v145; // [rsp+C0h] [rbp-180h] BYREF
  _QWORD v146[2]; // [rsp+C8h] [rbp-178h] BYREF
  __int64 v147; // [rsp+D8h] [rbp-168h]
  __int64 v148; // [rsp+E0h] [rbp-160h]
  __int64 v149; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v150; // [rsp+F8h] [rbp-148h] BYREF
  __int64 v151; // [rsp+100h] [rbp-140h]
  __int64 v152; // [rsp+108h] [rbp-138h]
  void *j; // [rsp+110h] [rbp-130h]
  __int64 v154; // [rsp+120h] [rbp-120h] BYREF
  const __m128i **v155; // [rsp+128h] [rbp-118h]
  __int64 v156; // [rsp+130h] [rbp-110h]
  __int64 v157; // [rsp+138h] [rbp-108h]
  __m128i *v158; // [rsp+140h] [rbp-100h]
  __int64 v159; // [rsp+148h] [rbp-F8h]
  __m128i v160; // [rsp+150h] [rbp-F0h] BYREF
  void **v161; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v162; // [rsp+168h] [rbp-D8h]
  __int16 v163; // [rsp+170h] [rbp-D0h]
  _QWORD *v164; // [rsp+178h] [rbp-C8h]
  unsigned int v165; // [rsp+188h] [rbp-B8h]
  char v166; // [rsp+190h] [rbp-B0h]
  _BYTE v167[16]; // [rsp+1A0h] [rbp-A0h] BYREF
  __int64 v168; // [rsp+1B0h] [rbp-90h]
  unsigned int v169; // [rsp+1C0h] [rbp-80h]
  __int64 *v170; // [rsp+1D0h] [rbp-70h]
  unsigned int v171; // [rsp+1E0h] [rbp-60h]
  _QWORD *v172; // [rsp+1F0h] [rbp-50h]
  unsigned int v173; // [rsp+200h] [rbp-40h]

  sub_10643D0((__int64)v167, (_QWORD *)a3);
  v5 = &v141;
  v6 = (__int64)a4;
  sub_25D3620((__int64)&v141, a4, v7, v8, v9, v10);
  v12 = v141;
  v116 = &v141[v142];
  if ( v141 == v116 )
  {
    v80 = 0;
LABEL_124:
    if ( v116 != (const __m128i *)&v143 )
      _libc_free((unsigned __int64)v116);
    for ( i = *(_QWORD *)(a3 + 16); a3 + 8 != i; i = *(_QWORD *)(i + 8) )
    {
      v82 = i - 56;
      if ( !i )
        v82 = 0;
      if ( !sub_B2FC80(v82) && (unsigned __int8)sub_A73380((__int64 *)(v82 + 72), "thinlto-internalize", 0x13u) )
        *(_WORD *)(v82 + 32) = *(_WORD *)(v82 + 32) & 0xBCC0 | 0x4007;
    }
    v107 = a1[8];
    *a1 = v80;
    a1[8] = v107 & 0xFC | 2;
    goto LABEL_15;
  }
  v117 = 0;
LABEL_3:
  v85 = a2[3] == 0;
  v160 = _mm_loadu_si128(v12);
  if ( v85 )
    sub_4263D6(v5, v6, v11);
  v13 = &v160;
  v14 = (__int64)(a2 + 1);
  ((void (__fastcall *)(_QWORD ***, __int64 *, __m128i *))a2[4])(&v143, a2 + 1, &v160);
  v15 = v144 & 1;
  v16 = (2 * (v144 & 1)) | v144 & 0xFD;
  v144 = v16;
  if ( v15 )
  {
    v108 = (unsigned __int64)v143;
    v143 = 0;
    v144 = v16 & 0xFD;
    v160.m128i_i64[0] = 0;
    sub_9C8CB0(v160.m128i_i64);
    a1[8] |= 3u;
    *(_QWORD *)a1 = v108 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_9;
  }
  v17 = v143;
  v143 = 0;
  v14 = (__int64)v17;
  v121 = (unsigned __int64)v17;
  sub_BA9820(&v160, (__int64)v17);
  v20 = v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    a1[8] |= 3u;
    *(_QWORD *)a1 = v20;
    goto LABEL_7;
  }
  v158 = &v160;
  v159 = 0;
  v32 = *(_QWORD *)(v121 + 32);
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v133 = v121 + 24;
  if ( v32 == v121 + 24 )
  {
LABEL_43:
    v38 = *(_QWORD *)(v121 + 16);
    v134 = v121 + 8;
    if ( v121 + 8 != v38 )
    {
      while ( v38 )
      {
        if ( (*(_BYTE *)(v38 - 49) & 0x10) == 0 )
          goto LABEL_45;
        sub_B2F930(&v160, v38 - 56);
        v39 = sub_B2F650(v160.m128i_i64[0], v160.m128i_i64[1]);
        v40 = v39;
        if ( (void ***)v160.m128i_i64[0] != &v161 )
        {
          v123 = v39;
          j_j___libc_free_0(v160.m128i_u64[0]);
          v40 = v123;
        }
        v160.m128i_i64[0] = (__int64)sub_25CDC80(a4, v12->m128i_i64[0], v12->m128i_u64[1], v40);
        if ( v160.m128i_i8[4] && !v160.m128i_i32[0] )
        {
          sub_B2F620((__int64)&v149, v38 - 56);
          v35 = v149 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v149 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_53;
          v149 = v38 - 56;
          sub_25D3DB0((__int64)&v154, &v149);
          v38 = *(_QWORD *)(v38 + 8);
          if ( v134 == v38 )
            goto LABEL_58;
        }
        else
        {
LABEL_45:
          v38 = *(_QWORD *)(v38 + 8);
          if ( v134 == v38 )
            goto LABEL_58;
        }
      }
LABEL_183:
      BUG();
    }
LABEL_58:
    if ( v121 + 40 == *(_QWORD *)(v121 + 48) )
      goto LABEL_104;
    v135 = a4;
    v41 = *(_QWORD *)(v121 + 48);
    v42 = v121 + 40;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v41 )
          goto LABEL_183;
        if ( (*(_BYTE *)(v41 - 41) & 0x10) != 0 && *(_BYTE *)sub_B325F0(v41 - 48) != 2 )
        {
          sub_B2F930(v13, v41 - 48);
          v43 = sub_B2F650(v160.m128i_i64[0], v160.m128i_i64[1]);
          v44 = v43;
          if ( (void ***)v160.m128i_i64[0] != &v161 )
          {
            v124 = v43;
            j_j___libc_free_0(v160.m128i_u64[0]);
            v44 = v124;
          }
          v140 = (__int64)sub_25CDC80(v135, v12->m128i_i64[0], v12->m128i_u64[1], v44);
          if ( BYTE4(v140) )
          {
            if ( !(_DWORD)v140 )
              break;
          }
        }
        v41 = *(_QWORD *)(v41 + 8);
        if ( v42 == v41 )
          goto LABEL_103;
      }
      sub_B2F620((__int64)v13, v41 - 48);
      v35 = v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_53;
      v45 = sub_B325F0(v41 - 48);
      sub_B2F620((__int64)v13, v45);
      v35 = v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_53;
      v46 = sub_B325F0(v41 - 48);
      v160.m128i_i64[0] = 0;
      v125 = v46;
      LODWORD(v162) = 128;
      v47 = (_QWORD *)sub_C7D670(0x2000, 8);
      v48 = &unk_49DD7A0;
      v161 = 0;
      v160.m128i_i64[1] = (__int64)v47;
      v150 = 2;
      v49 = &v47[8 * (unsigned __int64)(unsigned int)v162];
      v149 = (__int64)&unk_49DD7B0;
      v151 = 0;
      v152 = -4096;
      for ( j = 0; v49 != v47; v47 += 8 )
      {
        if ( v47 )
        {
          v50 = v150;
          v47[2] = 0;
          v47[3] = -4096;
          *v47 = &unk_49DD7B0;
          v47[1] = v50 & 6;
          v48 = j;
          v47[4] = j;
        }
      }
      v166 = 0;
      v51 = sub_F4BFF0(v125, (__int64)v13, 0, (__int64)v48);
      v52 = *(_BYTE *)(v41 - 16) & 0xF;
      v53 = *(_BYTE *)(v41 - 16) & 0xF;
      if ( (unsigned int)(v52 - 7) > 1 )
      {
        *(_BYTE *)(v51 + 32) = v53 | *(_BYTE *)(v51 + 32) & 0xF0;
      }
      else
      {
        *(_WORD *)(v51 + 32) = *(_BYTE *)(v41 - 16) & 0xF | *(_WORD *)(v51 + 32) & 0xFCC0;
        if ( v52 == 7 )
          goto LABEL_76;
      }
      if ( v52 == 8 )
      {
LABEL_76:
        v54 = *(_BYTE *)(v51 + 33) | 0x40;
        v55 = *(_BYTE *)(v51 + 32) & 0xCF;
        *(_BYTE *)(v51 + 33) = v54;
        *(_BYTE *)(v51 + 32) = *(_BYTE *)(v41 - 16) & 0x30 | v55;
        goto LABEL_77;
      }
      v85 = v53 == 9;
      v86 = *(_BYTE *)(v51 + 32);
      if ( (v86 & 0x30) != 0 && !v85 )
      {
        v54 = *(_BYTE *)(v51 + 33) | 0x40;
        *(_BYTE *)(v51 + 33) = v54;
        *(_BYTE *)(v51 + 32) = *(_BYTE *)(v41 - 16) & 0x30 | v86 & 0xCF;
        if ( v52 == 7 )
          goto LABEL_77;
      }
      else
      {
        *(_BYTE *)(v51 + 32) = *(_BYTE *)(v41 - 16) & 0x30 | *(_BYTE *)(v51 + 32) & 0xCF;
      }
      if ( (*(_BYTE *)(v51 + 32) & 0x30) == 0 || v85 )
        goto LABEL_78;
      v54 = *(_BYTE *)(v51 + 33);
LABEL_77:
      *(_BYTE *)(v51 + 33) = v54 | 0x40;
LABEL_78:
      v126 = (unsigned __int8 *)v51;
      sub_BD84D0(v41 - 48, v51);
      sub_BD6B90(v126, (unsigned __int8 *)(v41 - 48));
      v56 = v126;
      if ( v166 )
      {
        v88 = v165;
        v166 = 0;
        if ( v165 )
        {
          v130 = v12;
          v115 = v13;
          v89 = v164;
          v90 = v56;
          v91 = &v164[2 * v165];
          do
          {
            if ( *v89 != -4096 && *v89 != -8192 )
            {
              v92 = v89[1];
              if ( v92 )
                sub_B91220((__int64)(v89 + 1), v92);
            }
            v89 += 2;
          }
          while ( v91 != v89 );
          v12 = v130;
          v13 = v115;
          v56 = v90;
          v88 = v165;
        }
        v131 = v56;
        sub_C7D6A0((__int64)v164, 16LL * v88, 8);
        v56 = v131;
      }
      v57 = v162;
      if ( (_DWORD)v162 )
      {
        v114 = v12;
        v112 = v13;
        v58 = (_QWORD *)v160.m128i_i64[1];
        v146[0] = 2;
        v145 = &unk_49DD7B0;
        v149 = (__int64)&unk_49DD7B0;
        v59 = -4096;
        v146[1] = 0;
        v60 = v160.m128i_i64[1] + ((unsigned __int64)(unsigned int)v162 << 6);
        v147 = -4096;
        v148 = 0;
        v150 = 2;
        v151 = 0;
        v152 = -8192;
        j = 0;
        v127 = v56;
        while ( 1 )
        {
          v61 = v58[3];
          if ( v61 != v59 )
          {
            v59 = v152;
            if ( v61 != v152 )
            {
              v62 = v58[7];
              if ( v62 != 0 && v62 != -4096 && v62 != -8192 )
              {
                sub_BD60C0(v58 + 5);
                v61 = v58[3];
              }
              v59 = v61;
            }
          }
          *v58 = &unk_49DB368;
          if ( v59 != -4096 && v59 != 0 && v59 != -8192 )
            sub_BD60C0(v58 + 1);
          v58 += 8;
          if ( (_QWORD *)v60 == v58 )
            break;
          v59 = v147;
        }
        v56 = v127;
        v12 = v114;
        v13 = v112;
        v149 = (__int64)&unk_49DB368;
        if ( v152 != 0 && v152 != -4096 && v152 != -8192 )
        {
          sub_BD60C0(&v150);
          v56 = v127;
        }
        v145 = &unk_49DB368;
        if ( v147 != -4096 && v147 != 0 && v147 != -8192 )
        {
          v128 = v56;
          sub_BD60C0(v146);
          v56 = v128;
        }
        v57 = v162;
      }
      v129 = (__int64)v56;
      sub_C7D6A0(v160.m128i_i64[1], (unsigned __int64)v57 << 6, 8);
      v63 = v129;
      if ( (_BYTE)qword_4FF0B08 || LOBYTE(qword_4FF33E0[17]) )
      {
        v160.m128i_i64[0] = sub_B9B140(*(__int64 **)a3, *(const void **)(v121 + 168), *(_QWORD *)(v121 + 176));
        v64 = sub_B9C770(*(__int64 **)a3, v13->m128i_i64, (__int64 *)1, 0, 1);
        sub_B99460(v129, "thinlto_src_module", 0x12u, v64);
        v65 = sub_B9B140(*(__int64 **)a3, *(const void **)(v121 + 200), *(_QWORD *)(v121 + 208));
        v66 = *(__int64 **)a3;
        v160.m128i_i64[0] = v65;
        v67 = sub_B9C770(v66, v13->m128i_i64, (__int64 *)1, 0, 1);
        sub_B99460(v129, "thinlto_src_file", 0x10u, v67);
        v63 = v129;
      }
      v160.m128i_i64[0] = v63;
      sub_25D3DB0((__int64)&v154, v13->m128i_i64);
      v41 = *(_QWORD *)(v41 + 8);
      if ( v42 == v41 )
      {
LABEL_103:
        a4 = v135;
LABEL_104:
        sub_A84D20((_QWORD *)v121);
        sub_BAAAA0((__int64 **)v121, *a2);
        v68 = (unsigned __int8 *)*a2;
        v69 = v121;
        sub_29DCE60(v121, *a2, *((unsigned __int8 *)a2 + 40), &v154);
        if ( (_BYTE)qword_4FF0DA8 )
        {
          v70 = (char **)v158;
          v71 = (unsigned int)v159;
          v137 = (__m128i *)((char *)v158 + 8 * (unsigned int)v159);
          if ( v137 != v158 )
          {
            v93 = (__int64 *)v158;
            v113 = a4;
            v111 = v13;
            do
            {
              v100 = *v93;
              v101 = sub_C5F790(v69, (__int64)v68);
              v102 = sub_CB6200(v101, *(unsigned __int8 **)(a3 + 200), *(_QWORD *)(a3 + 208));
              v103 = *(_QWORD *)(v102 + 32);
              v104 = v102;
              if ( (unsigned __int64)(*(_QWORD *)(v102 + 24) - v103) > 8 )
              {
                *(_BYTE *)(v103 + 8) = 32;
                *(_QWORD *)v103 = 0x74726F706D49203ALL;
                *(_QWORD *)(v102 + 32) += 9LL;
              }
              else
              {
                v104 = sub_CB6200(v102, ": Import ", 9u);
              }
              v94 = sub_BD5D20(v100);
              v96 = *(_WORD **)(v104 + 32);
              v97 = (unsigned __int8 *)v94;
              v98 = *(_QWORD *)(v104 + 24) - (_QWORD)v96;
              if ( v95 > v98 )
              {
                v105 = sub_CB6200(v104, v97, v95);
                v96 = *(_WORD **)(v105 + 32);
                v104 = v105;
                v98 = *(_QWORD *)(v105 + 24) - (_QWORD)v96;
              }
              else if ( v95 )
              {
                v132 = v95;
                memcpy(v96, v97, v95);
                v106 = *(_QWORD *)(v104 + 24);
                v96 = (_WORD *)(v132 + *(_QWORD *)(v104 + 32));
                *(_QWORD *)(v104 + 32) = v96;
                v98 = v106 - (_QWORD)v96;
              }
              if ( v98 <= 5 )
              {
                v104 = sub_CB6200(v104, (unsigned __int8 *)" from ", 6u);
              }
              else
              {
                *(_DWORD *)v96 = 1869768224;
                v96[2] = 8301;
                *(_QWORD *)(v104 + 32) += 6LL;
              }
              v68 = *(unsigned __int8 **)(v121 + 200);
              v69 = sub_CB6200(v104, v68, *(_QWORD *)(v121 + 208));
              v99 = *(_BYTE **)(v69 + 32);
              if ( *(_BYTE **)(v69 + 24) == v99 )
              {
                v68 = (unsigned __int8 *)"\n";
                sub_CB6200(v69, (unsigned __int8 *)"\n", 1u);
              }
              else
              {
                *v99 = 10;
                ++*(_QWORD *)(v69 + 32);
              }
              ++v93;
            }
            while ( v137 != (__m128i *)v93 );
            a4 = v113;
            v13 = v111;
            v70 = (char **)v158;
            v71 = (unsigned int)v159;
          }
        }
        else
        {
          v70 = (char **)v158;
          v71 = (unsigned int)v159;
        }
        v149 = v121;
        v162 = 0;
        sub_106AB30(&v138, (__int64)v167, (__int64 **)&v149, v70, v71, v13, 1, 0);
        v73 = (__int64 *)v149;
        v74 = v110;
        if ( v149 )
        {
          sub_BA9C10((_QWORD **)v149, v110, v72, v109);
          v74 = 880;
          j_j___libc_free_0((unsigned __int64)v73);
        }
        if ( (v162 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v75 = (v162 >> 1) & 1;
          if ( (v162 & 4) != 0 )
          {
            v136 = (v162 >> 1) & 1;
            v76 = v13;
            if ( !(_BYTE)v75 )
              v76 = (__m128i *)v160.m128i_i64[0];
            (*(void (__fastcall **)(__m128i *, __int64))((v162 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v76, v74);
            LOBYTE(v75) = v136;
          }
          if ( !(_BYTE)v75 )
            sub_C7D6A0(v160.m128i_i64[0], v160.m128i_i64[1], (__int64)v161);
        }
        if ( (v138 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v139 = v138 & 0xFFFFFFFFFFFFFFFELL | 1;
          v138 = 0;
          sub_C64870((__int64)&v145, (__int64 *)&v139);
          v161 = &v145;
          v163 = 1027;
          v160.m128i_i64[0] = (__int64)"Function Import: link error: ";
          v83 = sub_2241E50();
          sub_CA0F50(&v149, (void **)v13);
          sub_C63F00(&v140, (__int64)&v149, 0x16u, (__int64)v83);
          sub_2240A30((unsigned __int64 *)&v149);
          v84 = v140 & 0xFFFFFFFFFFFFFFFELL;
          a1[8] |= 3u;
          *(_QWORD *)a1 = v84;
          sub_2240A30((unsigned __int64 *)&v145);
          sub_9C66B0((__int64 *)&v139);
          sub_9C66B0(&v138);
          v121 = 0;
          goto LABEL_54;
        }
        v77 = v159 + v117;
        v117 += v159;
        if ( v158 != v13 )
          _libc_free((unsigned __int64)v158);
        v5 = v155;
        v6 = 8LL * (unsigned int)v157;
        sub_C7D6A0((__int64)v155, v6, 8);
        if ( (v144 & 2) != 0 )
          goto LABEL_178;
        v79 = (const __m128i **)v143;
        if ( (v144 & 1) != 0 )
        {
          if ( v143 )
          {
            v5 = (const __m128i **)v143;
            ((void (__fastcall *)(_QWORD **))(*v143)[1])(v143);
          }
        }
        else if ( v143 )
        {
          sub_BA9C10(v143, v6, v11, v78);
          v6 = 880;
          v5 = v79;
          j_j___libc_free_0((unsigned __int64)v79);
        }
        if ( v116 == ++v12 )
        {
          v80 = v77 != 0;
          v116 = v141;
          goto LABEL_124;
        }
        goto LABEL_3;
      }
    }
  }
  while ( 1 )
  {
    while ( 1 )
    {
      if ( !v32 )
        goto LABEL_183;
      if ( (*(_BYTE *)(v32 - 49) & 0x10) != 0 )
      {
        sub_B2F930(&v160, v32 - 56);
        v33 = sub_B2F650(v160.m128i_i64[0], v160.m128i_i64[1]);
        v34 = v33;
        if ( (void ***)v160.m128i_i64[0] != &v161 )
        {
          v122 = v33;
          j_j___libc_free_0(v160.m128i_u64[0]);
          v34 = v122;
        }
        v160.m128i_i64[0] = (__int64)sub_25CDC80(a4, v12->m128i_i64[0], v12->m128i_u64[1], v34);
        if ( v160.m128i_i8[4] )
        {
          if ( !v160.m128i_i32[0] )
            break;
        }
      }
      v32 = *(_QWORD *)(v32 + 8);
      if ( v133 == v32 )
        goto LABEL_43;
    }
    sub_B2F620((__int64)&v149, v32 - 56);
    v35 = v149 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v149 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      break;
    if ( (_BYTE)qword_4FF0B08 || LOBYTE(qword_4FF33E0[17]) )
    {
      v149 = sub_B9B140(*(__int64 **)a3, *(const void **)(v121 + 168), *(_QWORD *)(v121 + 176));
      v36 = sub_B9C770(*(__int64 **)a3, &v149, (__int64 *)1, 0, 1);
      sub_B99460(v32 - 56, "thinlto_src_module", 0x12u, v36);
      v149 = sub_B9B140(*(__int64 **)a3, *(const void **)(v121 + 200), *(_QWORD *)(v121 + 208));
      v37 = sub_B9C770(*(__int64 **)a3, &v149, (__int64 *)1, 0, 1);
      sub_B99460(v32 - 56, "thinlto_src_file", 0x10u, v37);
    }
    v149 = v32 - 56;
    sub_25D3DB0((__int64)&v154, &v149);
    v32 = *(_QWORD *)(v32 + 8);
    if ( v133 == v32 )
      goto LABEL_43;
  }
LABEL_53:
  a1[8] |= 3u;
  *(_QWORD *)a1 = v35;
LABEL_54:
  if ( v158 != v13 )
    _libc_free((unsigned __int64)v158);
  v14 = 8LL * (unsigned int)v157;
  sub_C7D6A0((__int64)v155, v14, 8);
LABEL_7:
  if ( v121 )
  {
    sub_BA9C10((_QWORD **)v121, v14, v18, v19);
    v14 = 880;
    j_j___libc_free_0(v121);
  }
LABEL_9:
  if ( (v144 & 2) != 0 )
LABEL_178:
    sub_904700(&v143);
  v21 = v143;
  if ( (v144 & 1) != 0 )
  {
    if ( v143 )
      ((void (__fastcall *)(_QWORD **))(*v143)[1])(v143);
  }
  else if ( v143 )
  {
    sub_BA9C10(v143, v14, v18, v19);
    j_j___libc_free_0((unsigned __int64)v21);
  }
  if ( v141 != (const __m128i *)&v143 )
    _libc_free((unsigned __int64)v141);
LABEL_15:
  v22 = v173;
  if ( v173 )
  {
    v23 = v172;
    v24 = &v172[2 * v173];
    do
    {
      if ( *v23 != -4096 && *v23 != -8192 )
      {
        v25 = v23[1];
        if ( v25 )
          sub_B91220((__int64)(v23 + 1), v25);
      }
      v23 += 2;
    }
    while ( v24 != v23 );
    v22 = v173;
  }
  sub_C7D6A0((__int64)v172, 16LL * v22, 8);
  if ( v171 )
  {
    v26 = sub_1061AC0();
    v27 = sub_1061AD0();
    v28 = v170;
    v29 = v27;
    v30 = v171;
    v31 = (__int64)&v170[v30];
    if ( v170 != &v170[v30] )
    {
      do
      {
        if ( !sub_1061B40(*v28, v26) )
          sub_1061B40(*v28, v29);
        ++v28;
      }
      while ( (__int64 *)v31 != v28 );
      v31 = (__int64)v170;
      v30 = v171;
    }
  }
  else
  {
    v31 = (__int64)v170;
    v30 = 0;
  }
  sub_C7D6A0(v31, v30 * 8, 8);
  sub_C7D6A0(v168, 8LL * v169, 8);
  return a1;
}
