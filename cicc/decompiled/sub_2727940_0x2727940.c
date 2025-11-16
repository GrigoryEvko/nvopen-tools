// Function: sub_2727940
// Address: 0x2727940
//
__int64 __fastcall sub_2727940(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 i; // r14
  __int64 v6; // r13
  __int64 v7; // r9
  int v8; // r11d
  unsigned int v9; // ecx
  __int64 *v10; // r10
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // rax
  unsigned __int8 v16; // dl
  __int64 v17; // r13
  __int64 v18; // rcx
  _BYTE **v19; // r14
  _BYTE *v20; // rdi
  unsigned __int8 v21; // al
  _BYTE **v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // r12d
  _QWORD *v26; // rbx
  size_t v27; // r15
  __int64 v28; // r8
  unsigned int v29; // r12d
  __int64 v30; // r9
  int v31; // eax
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rcx
  const __m128i *v35; // rdx
  __m128i *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // r12
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rsi
  unsigned __int64 *v44; // rbx
  __int64 v45; // r8
  unsigned __int64 *v46; // r14
  unsigned __int64 v47; // rdi
  __int64 *v48; // r12
  __int64 v49; // rax
  __int64 *v50; // rbx
  __int64 *v51; // r12
  unsigned __int64 v52; // rdi
  __int64 result; // rax
  __int64 v54; // r13
  unsigned int v55; // r12d
  int v56; // eax
  int v57; // r11d
  unsigned int j; // ecx
  __int64 v59; // r10
  bool v60; // al
  const void *v61; // rsi
  unsigned int v62; // ecx
  int v63; // eax
  unsigned int v64; // r12d
  int v65; // eax
  int v66; // r10d
  __int64 v67; // rcx
  unsigned int v68; // r13d
  const void *v69; // r12
  bool v70; // al
  unsigned int v71; // r13d
  unsigned __int64 v72; // rbx
  int v73; // eax
  _QWORD *v74; // rdi
  int v75; // r10d
  unsigned int v76; // ebx
  unsigned int v77; // r13d
  const void *v78; // r12
  bool v79; // al
  int v80; // eax
  int v81; // eax
  __int64 v82; // r12
  __int64 *v83; // rdx
  __int64 *v84; // rbx
  __int64 v85; // rax
  __int64 *v86; // r14
  _QWORD *v87; // rdx
  _QWORD *v88; // r12
  __int64 v89; // r15
  int v90; // ecx
  unsigned int v91; // edx
  __int64 v92; // r8
  int v93; // edi
  __int64 *v94; // rsi
  int v95; // edi
  unsigned int v96; // edx
  __int64 v97; // r8
  unsigned int v98; // r13d
  __int64 v99; // [rsp+8h] [rbp-378h]
  int v100; // [rsp+8h] [rbp-378h]
  int v101; // [rsp+8h] [rbp-378h]
  __int64 v102; // [rsp+10h] [rbp-370h]
  __int64 v103; // [rsp+10h] [rbp-370h]
  __int64 v104; // [rsp+20h] [rbp-360h]
  unsigned int v105; // [rsp+20h] [rbp-360h]
  int v106; // [rsp+28h] [rbp-358h]
  unsigned int v107; // [rsp+30h] [rbp-350h]
  __int64 v108; // [rsp+30h] [rbp-350h]
  __int64 v109; // [rsp+30h] [rbp-350h]
  __int64 v110; // [rsp+30h] [rbp-350h]
  __int64 v111; // [rsp+38h] [rbp-348h]
  __int64 v112; // [rsp+40h] [rbp-340h]
  __int64 v113; // [rsp+48h] [rbp-338h]
  _BYTE **v116; // [rsp+78h] [rbp-308h]
  __int64 *v117; // [rsp+78h] [rbp-308h]
  _QWORD *v118; // [rsp+78h] [rbp-308h]
  __int64 *v119; // [rsp+78h] [rbp-308h]
  __m128i v120; // [rsp+80h] [rbp-300h] BYREF
  __int64 v121[2]; // [rsp+90h] [rbp-2F0h] BYREF
  __int64 *v122; // [rsp+A0h] [rbp-2E0h]
  __int64 v123; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 *v124; // [rsp+B8h] [rbp-2C8h]
  __int64 v125; // [rsp+C0h] [rbp-2C0h]
  unsigned int v126; // [rsp+C8h] [rbp-2B8h]
  __int64 v127; // [rsp+D0h] [rbp-2B0h] BYREF
  __int64 v128; // [rsp+D8h] [rbp-2A8h]
  __int64 v129; // [rsp+E0h] [rbp-2A0h]
  unsigned int v130; // [rsp+E8h] [rbp-298h]
  __int64 *v131; // [rsp+F0h] [rbp-290h] BYREF
  __int64 v132; // [rsp+F8h] [rbp-288h]
  __int64 v133[2]; // [rsp+100h] [rbp-280h] BYREF
  __int64 v134; // [rsp+110h] [rbp-270h] BYREF
  __int64 *v135; // [rsp+120h] [rbp-260h]
  __int64 v136; // [rsp+130h] [rbp-250h] BYREF
  __int64 *v137; // [rsp+150h] [rbp-230h] BYREF
  size_t v138; // [rsp+158h] [rbp-228h]
  __int64 v139; // [rsp+160h] [rbp-220h] BYREF
  __int64 *v140; // [rsp+170h] [rbp-210h]
  __int64 v141; // [rsp+180h] [rbp-200h] BYREF
  void *v142; // [rsp+1A0h] [rbp-1E0h] BYREF
  __int64 *v143; // [rsp+1A8h] [rbp-1D8h]
  const char *v144; // [rsp+1B0h] [rbp-1D0h]
  __int64 v145; // [rsp+1B8h] [rbp-1C8h]
  __int64 v146; // [rsp+1C0h] [rbp-1C0h]
  __int64 v147; // [rsp+1C8h] [rbp-1B8h]
  unsigned __int64 *v148; // [rsp+1F0h] [rbp-190h]
  unsigned int v149; // [rsp+1F8h] [rbp-188h]
  char v150; // [rsp+200h] [rbp-180h] BYREF

  v2 = sub_B2BE50(a1);
  if ( !sub_B6EA50(v2) )
  {
    v82 = sub_B6F970(v2);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v82 + 32LL))(
            v82,
            "annotation-remarks",
            18)
      && !(*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v82 + 40LL))(
            v82,
            "annotation-remarks",
            18) )
    {
      result = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v82 + 24LL))(
                 v82,
                 "annotation-remarks",
                 18);
      if ( !(_BYTE)result )
        return result;
    }
  }
  v123 = 0;
  v3 = a1 + 72;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  sub_1049690(v121, a1);
  v4 = *(_QWORD *)(a1 + 80);
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = v133;
  v132 = 0;
  if ( a1 + 72 == v4 )
  {
    i = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    for ( i = *(_QWORD *)(v4 + 32); i == v4 + 24; i = *(_QWORD *)(v4 + 32) )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        goto LABEL_43;
      if ( !v4 )
        BUG();
    }
  }
  if ( v3 == v4 )
    goto LABEL_43;
  do
  {
    if ( !i )
      BUG();
    if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || !sub_B91C10(i - 24, 30) )
      goto LABEL_33;
    v6 = *(_QWORD *)(i + 24);
    if ( !v126 )
    {
      ++v123;
      goto LABEL_163;
    }
    v7 = v126 - 1;
    v8 = 1;
    v9 = v7 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v10 = &v124[7 * v9];
    v11 = 0;
    v12 = *v10;
    if ( v6 != *v10 )
    {
      while ( v12 != -4096 )
      {
        if ( v12 == -8192 && !v11 )
          v11 = v10;
        v9 = v7 & (v8 + v9);
        v10 = &v124[7 * v9];
        v12 = *v10;
        if ( v6 == *v10 )
          goto LABEL_11;
        ++v8;
      }
      if ( !v11 )
        v11 = v10;
      ++v123;
      v90 = v125 + 1;
      if ( 4 * ((int)v125 + 1) < 3 * v126 )
      {
        if ( v126 - HIDWORD(v125) - v90 > v126 >> 3 )
          goto LABEL_159;
        sub_27275E0((__int64)&v123, v126);
        if ( !v126 )
        {
LABEL_195:
          LODWORD(v125) = v125 + 1;
          BUG();
        }
        v95 = 1;
        v94 = 0;
        v96 = (v126 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v11 = &v124[7 * v96];
        v97 = *v11;
        v90 = v125 + 1;
        if ( v6 == *v11 )
          goto LABEL_159;
        while ( v97 != -4096 )
        {
          if ( !v94 && v97 == -8192 )
            v94 = v11;
          v96 = (v126 - 1) & (v95 + v96);
          v11 = &v124[7 * v96];
          v97 = *v11;
          if ( v6 == *v11 )
            goto LABEL_159;
          ++v95;
        }
        goto LABEL_167;
      }
LABEL_163:
      sub_27275E0((__int64)&v123, 2 * v126);
      if ( !v126 )
        goto LABEL_195;
      v91 = (v126 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v11 = &v124[7 * v91];
      v92 = *v11;
      v90 = v125 + 1;
      if ( v6 == *v11 )
        goto LABEL_159;
      v93 = 1;
      v94 = 0;
      while ( v92 != -4096 )
      {
        if ( v92 == -8192 && !v94 )
          v94 = v11;
        v91 = (v126 - 1) & (v93 + v91);
        v11 = &v124[7 * v91];
        v92 = *v11;
        if ( v6 == *v11 )
          goto LABEL_159;
        ++v93;
      }
LABEL_167:
      if ( v94 )
        v11 = v94;
LABEL_159:
      LODWORD(v125) = v90;
      if ( *v11 != -4096 )
        --HIDWORD(v125);
      *v11 = v6;
      v14 = v11 + 1;
      v11[1] = (__int64)(v11 + 3);
      v11[2] = 0x400000000LL;
      v13 = 0;
      goto LABEL_13;
    }
LABEL_11:
    v13 = *((unsigned int *)v10 + 4);
    v14 = v10 + 1;
    if ( v13 + 1 > (unsigned __int64)*((unsigned int *)v10 + 5) )
    {
      v119 = v10;
      sub_C8D5F0((__int64)(v10 + 1), v10 + 3, v13 + 1, 8u, (__int64)v124, v7);
      v13 = *((unsigned int *)v119 + 4);
    }
LABEL_13:
    *(_QWORD *)(*v14 + 8 * v13) = i - 24;
    ++*((_DWORD *)v14 + 2);
    if ( (*(_BYTE *)(i - 17) & 0x20) == 0 )
      BUG();
    v15 = sub_B91C10(i - 24, 30);
    v16 = *(_BYTE *)(v15 - 16);
    if ( (v16 & 2) != 0 )
    {
      v17 = *(_QWORD *)(v15 - 32);
      v18 = *(unsigned int *)(v15 - 24);
    }
    else
    {
      v18 = (*(_WORD *)(v15 - 16) >> 6) & 0xF;
      v17 = v15 - 8LL * ((v16 >> 2) & 0xF) - 16;
    }
    v116 = (_BYTE **)(v17 + 8 * v18);
    if ( v116 == (_BYTE **)v17 )
      goto LABEL_33;
    v113 = v3;
    v112 = v4;
    v111 = i;
    v19 = (_BYTE **)v17;
    do
    {
      v20 = *v19;
      if ( **v19 )
      {
        v21 = *(v20 - 16);
        if ( (v21 & 2) != 0 )
          v22 = (_BYTE **)*((_QWORD *)v20 - 4);
        else
          v22 = (_BYTE **)&v20[-8 * ((v21 >> 2) & 0xF) - 16];
        v20 = *v22;
      }
      v23 = (_QWORD *)sub_B91420((__int64)v20);
      v25 = v130;
      v26 = v23;
      v27 = v24;
      if ( !v130 )
      {
        ++v127;
        goto LABEL_24;
      }
      v137 = v23;
      v54 = v128;
      v138 = v24;
      v55 = v130 - 1;
      v56 = sub_C94890(v23, v24);
      v57 = 1;
      v30 = 0;
      for ( j = v55 & v56; ; j = v55 & v62 )
      {
        v59 = v54 + 24LL * j;
        v60 = (_QWORD *)((char *)v26 + 1) == 0;
        v61 = *(const void **)v59;
        if ( *(_QWORD *)v59 != -1 )
        {
          v60 = (_QWORD *)((char *)v26 + 2) == 0;
          if ( v61 != (const void *)-2LL )
          {
            if ( *(_QWORD *)(v59 + 8) != v27 )
              goto LABEL_83;
            v104 = v30;
            v106 = v57;
            v107 = j;
            if ( !v27 )
              goto LABEL_90;
            v99 = v54 + 24LL * j;
            v63 = memcmp(v26, v61, v27);
            v59 = v99;
            j = v107;
            v57 = v106;
            v30 = v104;
            v60 = v63 == 0;
          }
        }
        if ( v60 )
        {
LABEL_90:
          v37 = *(unsigned int *)(v59 + 16);
          goto LABEL_31;
        }
        if ( v61 == (const void *)-1LL )
          break;
LABEL_83:
        if ( v61 == (const void *)-2LL && !v30 )
          v30 = v59;
        v62 = v57 + j;
        ++v57;
      }
      v25 = v130;
      if ( !v30 )
        v30 = v59;
      ++v127;
      v31 = v129 + 1;
      if ( 4 * ((int)v129 + 1) < 3 * v130 )
      {
        if ( v130 - (v31 + HIDWORD(v129)) > v130 >> 3 )
          goto LABEL_26;
        sub_1253750((__int64)&v127, v130);
        v30 = 0;
        if ( !v130 )
          goto LABEL_25;
        v64 = v130 - 1;
        v137 = v26;
        v108 = v128;
        v138 = v27;
        v65 = sub_C94890(v26, v27);
        v66 = 1;
        v67 = 0;
        v28 = v64;
        v68 = v64 & v65;
        while ( 2 )
        {
          v30 = v108 + 24LL * v68;
          v69 = *(const void **)v30;
          if ( *(_QWORD *)v30 == -1 )
          {
            if ( v26 != (_QWORD *)-1LL )
              goto LABEL_125;
            goto LABEL_25;
          }
          v70 = (_QWORD *)((char *)v26 + 2) == 0;
          if ( v69 != (const void *)-2LL )
          {
            if ( v27 != *(_QWORD *)(v30 + 8) )
            {
LABEL_105:
              if ( v67 || v69 != (const void *)-2LL )
                v30 = v67;
              v71 = v66 + v68;
              v67 = v30;
              ++v66;
              v68 = v28 & v71;
              continue;
            }
            v101 = v66;
            v103 = v67;
            v105 = v28;
            if ( !v27 )
              goto LABEL_25;
            v81 = memcmp(v26, v69, v27);
            v30 = v108 + 24LL * v68;
            v28 = v105;
            v67 = v103;
            v66 = v101;
            v70 = v81 == 0;
          }
          break;
        }
        if ( v70 )
          goto LABEL_25;
        if ( v69 == (const void *)-1LL )
        {
LABEL_125:
          if ( v67 )
            v30 = v67;
          goto LABEL_25;
        }
        goto LABEL_105;
      }
LABEL_24:
      sub_1253750((__int64)&v127, 2 * v25);
      v29 = v130;
      v30 = 0;
      if ( !v130 )
        goto LABEL_25;
      v137 = v26;
      v138 = v27;
      v110 = v128;
      v73 = sub_C94890(v26, v27);
      v74 = v26;
      v75 = 1;
      v67 = 0;
      v76 = v29 - 1;
      v77 = (v29 - 1) & v73;
      while ( 2 )
      {
        v30 = v110 + 24LL * v77;
        v78 = *(const void **)v30;
        if ( *(_QWORD *)v30 != -1 )
        {
          v79 = (_QWORD *)((char *)v74 + 2) == 0;
          if ( v78 == (const void *)-2LL )
          {
LABEL_119:
            if ( v79 )
            {
LABEL_120:
              v26 = v74;
              goto LABEL_25;
            }
            if ( v78 == (const void *)-2LL && !v67 )
              v67 = v30;
          }
          else if ( *(_QWORD *)(v30 + 8) == v27 )
          {
            v100 = v75;
            v102 = v67;
            if ( !v27 )
              goto LABEL_120;
            v80 = memcmp(v74, v78, v27);
            v30 = v110 + 24LL * v77;
            v67 = v102;
            v75 = v100;
            v79 = v80 == 0;
            goto LABEL_119;
          }
          v98 = v75 + v77;
          ++v75;
          v77 = v76 & v98;
          continue;
        }
        break;
      }
      v26 = v74;
      if ( v74 != (_QWORD *)-1LL )
        goto LABEL_125;
LABEL_25:
      v31 = v129 + 1;
LABEL_26:
      LODWORD(v129) = v31;
      if ( *(_QWORD *)v30 != -1 )
        --HIDWORD(v129);
      *(_QWORD *)v30 = v26;
      *(_QWORD *)(v30 + 8) = v27;
      *(_DWORD *)(v30 + 16) = 0;
      v32 = (unsigned int)v132;
      v142 = v26;
      v33 = (unsigned int)v132 + 1LL;
      v143 = (__int64 *)v27;
      LODWORD(v144) = 0;
      if ( v33 > HIDWORD(v132) )
      {
        v72 = (unsigned __int64)v131;
        v109 = v30;
        if ( v131 > (__int64 *)&v142 || &v142 >= (void **)&v131[3 * (unsigned int)v132] )
        {
          sub_C8D5F0((__int64)&v131, v133, v33, 0x18u, v28, v30);
          v34 = (unsigned __int64)v131;
          v32 = (unsigned int)v132;
          v35 = (const __m128i *)&v142;
          v30 = v109;
        }
        else
        {
          sub_C8D5F0((__int64)&v131, v133, v33, 0x18u, v28, v30);
          v34 = (unsigned __int64)v131;
          v32 = (unsigned int)v132;
          v30 = v109;
          v35 = (const __m128i *)((char *)&v142 + (_QWORD)v131 - v72);
        }
      }
      else
      {
        v34 = (unsigned __int64)v131;
        v35 = (const __m128i *)&v142;
      }
      v36 = (__m128i *)(v34 + 24 * v32);
      *v36 = _mm_loadu_si128(v35);
      v36[1].m128i_i64[0] = v35[1].m128i_i64[0];
      v37 = (unsigned int)v132;
      LODWORD(v132) = v132 + 1;
      *(_DWORD *)(v30 + 16) = v37;
LABEL_31:
      ++v19;
      ++LODWORD(v131[3 * v37 + 2]);
    }
    while ( v116 != v19 );
    v3 = v113;
    v4 = v112;
    i = v111;
LABEL_33:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v4 + 32) )
    {
      v38 = v4 - 24;
      if ( !v4 )
        v38 = 0;
      if ( i != v38 + 48 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        goto LABEL_43;
      if ( !v4 )
        BUG();
    }
  }
  while ( v3 != v4 );
LABEL_43:
  v39 = (unsigned __int64)v131;
  v117 = &v131[3 * (unsigned int)v132];
  if ( v117 != v131 )
  {
    do
    {
      v40 = *(_QWORD *)(a1 + 80);
      if ( v40 )
        v40 -= 24;
      v41 = sub_B92180(a1);
      sub_B15890(&v120, v41);
      sub_B17850((__int64)&v142, (__int64)"annotation-remarks", (__int64)"AnnotationSummary", 17, &v120, v40);
      sub_B18290((__int64)&v142, "Annotated ", 0xAu);
      sub_B169E0(v133, "count", 5, *(_DWORD *)(v39 + 16));
      v42 = sub_B826F0((__int64)&v142, (__int64)v133);
      sub_B18290(v42, " instructions with ", 0x13u);
      sub_B16430((__int64)&v137, "type", 4u, *(_BYTE **)v39, *(_QWORD *)(v39 + 8));
      v43 = sub_B826F0(v42, (__int64)&v137);
      sub_1049740(v121, v43);
      if ( v140 != &v141 )
        j_j___libc_free_0((unsigned __int64)v140);
      if ( v137 != &v139 )
        j_j___libc_free_0((unsigned __int64)v137);
      if ( v135 != &v136 )
        j_j___libc_free_0((unsigned __int64)v135);
      if ( (__int64 *)v133[0] != &v134 )
        j_j___libc_free_0(v133[0]);
      v44 = v148;
      v142 = &unk_49D9D40;
      v45 = 10LL * v149;
      v46 = &v148[v45];
      if ( v148 != &v148[v45] )
      {
        do
        {
          v46 -= 10;
          v47 = v46[4];
          if ( (unsigned __int64 *)v47 != v46 + 6 )
            j_j___libc_free_0(v47);
          if ( (unsigned __int64 *)*v46 != v46 + 2 )
            j_j___libc_free_0(*v46);
        }
        while ( v44 != v46 );
        v46 = v148;
      }
      if ( v46 != (unsigned __int64 *)&v150 )
        _libc_free((unsigned __int64)v46);
      v39 += 24LL;
    }
    while ( v117 != (__int64 *)v39 );
  }
  if ( (_DWORD)v125 )
  {
    v83 = v124;
    v84 = &v124[7 * v126];
    if ( v124 != v84 )
    {
      while ( 1 )
      {
        v85 = *v83;
        v86 = v83;
        if ( *v83 != -4096 && v85 != -8192 )
          break;
        v83 += 7;
        if ( v84 == v83 )
          goto LABEL_65;
      }
      while ( v86 != v84 )
      {
        if ( v85 )
        {
          v87 = (_QWORD *)v86[1];
          v88 = v87;
          v118 = &v87[*((unsigned int *)v86 + 4)];
          if ( v87 != v118 )
          {
            do
            {
              v89 = *v88;
              if ( (unsigned __int8)sub_2A37EC0(*v88) )
              {
                v146 = sub_B2BEC0(*(_QWORD *)(*(_QWORD *)(v89 + 40) + 72LL));
                v143 = v121;
                v147 = a2;
                v144 = "annotation-remarks";
                v145 = 18;
                v142 = &unk_4A22D78;
                sub_2A398E0(&v142, v89);
                v142 = &unk_4A22D78;
                nullsub_1558(&v142);
              }
              ++v88;
            }
            while ( v118 != v88 );
          }
        }
        v86 += 7;
        if ( v86 == v84 )
          break;
        while ( 1 )
        {
          v85 = *v86;
          if ( *v86 != -8192 && v85 != -4096 )
            break;
          v86 += 7;
          if ( v84 == v86 )
            goto LABEL_65;
        }
      }
    }
  }
LABEL_65:
  if ( v131 != v133 )
    _libc_free((unsigned __int64)v131);
  sub_C7D6A0(v128, 24LL * v130, 8);
  v48 = v122;
  if ( v122 )
  {
    sub_FDC110(v122);
    j_j___libc_free_0((unsigned __int64)v48);
  }
  v49 = v126;
  if ( v126 )
  {
    v50 = v124;
    v51 = &v124[7 * v126];
    do
    {
      if ( *v50 != -8192 && *v50 != -4096 )
      {
        v52 = v50[1];
        if ( (__int64 *)v52 != v50 + 3 )
          _libc_free(v52);
      }
      v50 += 7;
    }
    while ( v51 != v50 );
    v49 = v126;
  }
  return sub_C7D6A0((__int64)v124, 56 * v49, 8);
}
