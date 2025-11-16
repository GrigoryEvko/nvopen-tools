// Function: sub_1B7A380
// Address: 0x1b7a380
//
__int64 __fastcall sub_1B7A380(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // r12
  int v5; // ebx
  __int64 v6; // rax
  char *v7; // rdi
  __int64 v8; // rdx
  unsigned int v9; // r8d
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  int v12; // r9d
  const void *v13; // rdi
  unsigned int v14; // r8d
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  size_t v17; // rdx
  unsigned __int64 v18; // rax
  unsigned int v19; // ecx
  bool v20; // r10
  __int64 v21; // r11
  __int64 v22; // rbx
  const void *v23; // rsi
  unsigned int v24; // ecx
  unsigned int v25; // ecx
  __int64 v26; // rax
  int v27; // r14d
  _QWORD *v28; // r10
  _QWORD *v29; // rdx
  int v30; // eax
  int v31; // eax
  char v32; // r8
  __int64 v33; // rax
  int v34; // edx
  int v35; // ecx
  int v36; // ecx
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 *v43; // rdi
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  __int64 v46; // rbx
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 *v49; // r14
  __int64 v50; // rax
  __int64 v51; // r13
  int v52; // ecx
  __int64 v53; // rax
  const char *v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  char v58; // r9
  __int64 v59; // rax
  int v60; // esi
  __int16 v61; // dx
  __int64 v62; // rsi
  unsigned __int8 *v63; // rsi
  size_t v64; // rdi
  __int64 v65; // rdi
  __int64 v66; // rbx
  __int64 v67; // r12
  int v69; // esi
  int v70; // eax
  __int64 v71; // r9
  int v72; // eax
  __int64 v73; // rax
  int v74; // esi
  int v75; // edx
  int v76; // edx
  __int64 v77; // rdi
  unsigned int v78; // ecx
  __int64 v79; // rsi
  int v80; // r11d
  _QWORD *v81; // r9
  __int64 v82; // rax
  int v83; // edx
  int v84; // esi
  int v85; // edx
  int v86; // edx
  int v87; // esi
  int v88; // edx
  int v89; // esi
  int v90; // r11d
  int v91; // edx
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rax
  __int64 *v95; // rdi
  __int64 v96; // rax
  __int64 v97; // r8
  __int64 *v98; // rdi
  __int64 v99; // rax
  bool v100; // al
  int v101; // eax
  unsigned int v102; // esi
  __int64 v103; // rdi
  char v104; // r8
  __int64 v105; // rax
  int v106; // edx
  int v107; // edx
  __int64 v108; // rdi
  int v109; // r11d
  unsigned int v110; // ecx
  __int64 v111; // rsi
  int v112; // edx
  int v113; // esi
  int v114; // edx
  __int64 v116; // [rsp+8h] [rbp-118h]
  int v117; // [rsp+10h] [rbp-110h]
  unsigned int v118; // [rsp+14h] [rbp-10Ch]
  size_t v119; // [rsp+18h] [rbp-108h]
  __int64 v120; // [rsp+28h] [rbp-F8h]
  unsigned int v121; // [rsp+30h] [rbp-F0h]
  int v122; // [rsp+34h] [rbp-ECh]
  unsigned int v123; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v124; // [rsp+38h] [rbp-E8h]
  bool v125; // [rsp+3Eh] [rbp-E2h]
  unsigned __int8 v126; // [rsp+3Fh] [rbp-E1h]
  unsigned int v127; // [rsp+40h] [rbp-E0h]
  bool v128; // [rsp+40h] [rbp-E0h]
  __int64 v129; // [rsp+40h] [rbp-E0h]
  __int64 v130; // [rsp+48h] [rbp-D8h]
  __int64 v131; // [rsp+50h] [rbp-D0h]
  __int64 v132; // [rsp+58h] [rbp-C8h]
  int v133; // [rsp+58h] [rbp-C8h]
  __int64 v134; // [rsp+68h] [rbp-B8h] BYREF
  __m128i v135; // [rsp+70h] [rbp-B0h] BYREF
  int v136; // [rsp+80h] [rbp-A0h]
  __int64 v137; // [rsp+90h] [rbp-90h] BYREF
  __int64 v138; // [rsp+98h] [rbp-88h]
  __int64 v139; // [rsp+A0h] [rbp-80h]
  unsigned int v140; // [rsp+A8h] [rbp-78h]
  __int64 v141; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v142; // [rsp+B8h] [rbp-68h]
  __int64 v143; // [rsp+C0h] [rbp-60h]
  unsigned int v144; // [rsp+C8h] [rbp-58h]
  size_t n[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v146; // [rsp+E0h] [rbp-40h]
  __int64 v147; // [rsp+E8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 80);
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v131 = v1;
  v120 = a1 + 72;
  if ( v1 == a1 + 72 )
  {
    v126 = 0;
    v65 = 0;
    goto LABEL_82;
  }
  v126 = 0;
  do
  {
    if ( !v131 )
      BUG();
    v2 = *(_QWORD *)(v131 + 24);
    v130 = v131 - 24;
    v132 = v131 + 16;
    if ( v2 == v131 + 16 )
      goto LABEL_55;
    v123 = ((unsigned int)(v131 - 24) >> 9) ^ ((unsigned int)(v131 - 24) >> 4);
    do
    {
      while ( 1 )
      {
        if ( !v2 )
          BUG();
        if ( *(_BYTE *)(v2 - 8) != 78 )
          break;
        v82 = *(_QWORD *)(v2 - 48);
        if ( *(_BYTE *)(v82 + 16) || (*(_BYTE *)(v82 + 33) & 0x20) == 0 )
          break;
        if ( (unsigned int)(*(_DWORD *)(v82 + 36) - 133) > 4 )
          goto LABEL_54;
        if ( ((1LL << (*(_BYTE *)(v82 + 36) + 123)) & 0x15) != 0 )
          break;
        v2 = *(_QWORD *)(v2 + 8);
        if ( v132 == v2 )
          goto LABEL_55;
      }
      v3 = sub_15C70A0(v2 + 24);
      v4 = v3;
      if ( !v3 )
        goto LABEL_54;
      v5 = *(_DWORD *)(v3 + 4);
      v6 = *(_QWORD *)(v3 - 8LL * *(unsigned int *)(v3 + 8));
      if ( *(_BYTE *)v6 == 15 || (v6 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8))) != 0 )
      {
        v7 = *(char **)(v6 - 8LL * *(unsigned int *)(v6 + 8));
        if ( v7 )
          v7 = (char *)sub_161E970((__int64)v7);
        else
          v8 = 0;
      }
      else
      {
        v8 = 0;
        v7 = (char *)byte_3F871B3;
      }
      v9 = v140;
      n[0] = (size_t)v7;
      n[1] = v8;
      LODWORD(v146) = v5;
      if ( !v140 )
      {
        ++v137;
        goto LABEL_93;
      }
      v127 = v140;
      v10 = v138;
      v11 = sub_16D3930(v7, v8);
      v12 = v146;
      v122 = 1;
      v13 = (const void *)n[0];
      v14 = v127 - 1;
      v15 = ((unsigned int)(37 * v146) | (v11 << 32)) - 1 - ((unsigned __int64)(unsigned int)(37 * v146) << 32);
      v16 = ((v15 >> 22) ^ v15) - 1 - (((v15 >> 22) ^ v15) << 13);
      v17 = n[1];
      v18 = ((9 * ((v16 >> 8) ^ v16)) >> 15) ^ (9 * ((v16 >> 8) ^ v16));
      v19 = (v127 - 1) & (((v18 - 1 - (v18 << 27)) >> 31) ^ (v18 - 1 - ((_DWORD)v18 << 27)));
      v128 = n[0] == -2;
      v20 = n[0] == -1;
      v21 = 0;
      while ( 1 )
      {
        v22 = v10 + 56LL * v19;
        v23 = *(const void **)v22;
        if ( *(_QWORD *)v22 != -1 )
          break;
        v100 = v20;
LABEL_152:
        if ( v100 && v12 == *(_DWORD *)(v22 + 16) )
          goto LABEL_164;
        if ( v23 == (const void *)-1LL )
        {
          if ( *(_DWORD *)(v22 + 16) == -1 )
          {
            v9 = v140;
            if ( v21 )
              v22 = v21;
            ++v137;
            v70 = v139 + 1;
            if ( 4 * ((int)v139 + 1) < 3 * v140 )
            {
              if ( v140 - (v70 + HIDWORD(v139)) <= v140 >> 3 )
              {
                v69 = v140;
LABEL_94:
                sub_1B79DE0((__int64)&v137, v69);
                sub_1B798C0((__int64)&v137, (__int64)n, &v135);
                v22 = v135.m128i_i64[0];
                v70 = v139 + 1;
              }
              LODWORD(v139) = v70;
              if ( *(_QWORD *)v22 != -1 || *(_DWORD *)(v22 + 16) != -1 )
                --HIDWORD(v139);
              v71 = v22 + 24;
              *(__m128i *)v22 = _mm_loadu_si128((const __m128i *)n);
              v72 = v146;
              *(_QWORD *)(v22 + 24) = 0;
              *(_QWORD *)(v22 + 32) = 0;
              *(_QWORD *)(v22 + 40) = 0;
              *(_DWORD *)(v22 + 48) = 0;
              *(_DWORD *)(v22 + 16) = v72;
              v73 = 1;
              goto LABEL_99;
            }
LABEL_93:
            v69 = 2 * v9;
            goto LABEL_94;
          }
        }
        else if ( *(_DWORD *)(v22 + 16) == -2 && !v21 )
        {
          v21 = v10 + 56LL * v19;
        }
LABEL_18:
        v24 = v122 + v19;
        ++v122;
        v19 = v14 & v24;
      }
      if ( v23 == (const void *)-2LL )
      {
        v100 = v128;
        goto LABEL_152;
      }
      if ( v17 != *(_QWORD *)(v22 + 8) )
        goto LABEL_18;
      if ( v17 )
      {
        v125 = v20;
        v117 = v12;
        v116 = v21;
        v118 = v19;
        v121 = v14;
        v119 = v17;
        v101 = memcmp(v13, v23, v17);
        v17 = v119;
        v14 = v121;
        v19 = v118;
        v21 = v116;
        v12 = v117;
        v20 = v125;
        if ( v101 )
          goto LABEL_18;
      }
      if ( v12 != *(_DWORD *)(v22 + 16) )
        goto LABEL_18;
LABEL_164:
      v102 = *(_DWORD *)(v22 + 48);
      v103 = *(_QWORD *)(v22 + 32);
      v71 = v22 + 24;
      if ( !v102 )
      {
        v73 = *(_QWORD *)(v22 + 24) + 1LL;
LABEL_99:
        *(_QWORD *)(v22 + 24) = v73;
        v74 = 0;
        goto LABEL_100;
      }
      v27 = 1;
      v28 = 0;
      v25 = (v102 - 1) & v123;
      v29 = (_QWORD *)(v103 + 8LL * v25);
      v26 = *v29;
      if ( v130 != *v29 )
      {
        while ( v26 != -8 )
        {
          if ( v28 || v26 != -16 )
            v29 = v28;
          v25 = (v102 - 1) & (v27 + v25);
          v26 = *(_QWORD *)(v103 + 8LL * v25);
          if ( v130 == v26 )
            goto LABEL_166;
          ++v27;
          v28 = v29;
          v29 = (_QWORD *)(v103 + 8LL * v25);
        }
        v30 = *(_DWORD *)(v22 + 40);
        if ( !v28 )
          v28 = v29;
        ++*(_QWORD *)(v22 + 24);
        v31 = v30 + 1;
        if ( 4 * v31 < 3 * v102 )
        {
          if ( v102 - *(_DWORD *)(v22 + 44) - v31 > v102 >> 3 )
            goto LABEL_28;
          sub_163D380(v22 + 24, v102);
          v106 = *(_DWORD *)(v22 + 48);
          if ( v106 )
          {
            v107 = v106 - 1;
            v108 = *(_QWORD *)(v22 + 32);
            v81 = 0;
            v109 = 1;
            v110 = v107 & v123;
            v28 = (_QWORD *)(v108 + 8LL * (v107 & v123));
            v111 = *v28;
            v31 = *(_DWORD *)(v22 + 40) + 1;
            if ( v130 != *v28 )
            {
              while ( v111 != -8 )
              {
                if ( !v81 && v111 == -16 )
                  v81 = v28;
                v110 = v107 & (v109 + v110);
                v28 = (_QWORD *)(v108 + 8LL * v110);
                v111 = *v28;
                if ( v130 == *v28 )
                  goto LABEL_28;
                ++v109;
              }
              goto LABEL_172;
            }
            goto LABEL_28;
          }
LABEL_216:
          ++*(_DWORD *)(v22 + 40);
          BUG();
        }
        v74 = 2 * v102;
LABEL_100:
        sub_163D380(v71, v74);
        v75 = *(_DWORD *)(v22 + 48);
        if ( !v75 )
          goto LABEL_216;
        v76 = v75 - 1;
        v77 = *(_QWORD *)(v22 + 32);
        v78 = v76 & v123;
        v28 = (_QWORD *)(v77 + 8LL * (v76 & v123));
        v79 = *v28;
        v31 = *(_DWORD *)(v22 + 40) + 1;
        if ( v130 != *v28 )
        {
          v80 = 1;
          v81 = 0;
          while ( v79 != -8 )
          {
            if ( v79 == -16 && !v81 )
              v81 = v28;
            v78 = v76 & (v80 + v78);
            v28 = (_QWORD *)(v77 + 8LL * v78);
            v79 = *v28;
            if ( v130 == *v28 )
              goto LABEL_28;
            ++v80;
          }
LABEL_172:
          if ( v81 )
            v28 = v81;
        }
LABEL_28:
        *(_DWORD *)(v22 + 40) = v31;
        if ( *v28 != -8 )
          --*(_DWORD *)(v22 + 44);
        *v28 = v130;
        if ( *(_DWORD *)(v22 + 40) == 1 )
          goto LABEL_54;
        v32 = sub_1B79A80((__int64)&v141, (__int64)n, &v135);
        v33 = v135.m128i_i64[0];
        if ( v32 )
        {
          v34 = *(_DWORD *)(v135.m128i_i64[0] + 24) + 1;
          *(_DWORD *)(v135.m128i_i64[0] + 24) = v34;
          goto LABEL_33;
        }
        ++v141;
        v83 = v143 + 1;
        v84 = v144;
        if ( 4 * ((int)v143 + 1) >= 3 * v144 )
        {
          v84 = 2 * v144;
        }
        else if ( v144 - HIDWORD(v143) - v83 > v144 >> 3 )
        {
          goto LABEL_119;
        }
        sub_1B7A020((__int64)&v141, v84);
        sub_1B79A80((__int64)&v141, (__int64)n, &v135);
        v33 = v135.m128i_i64[0];
        v83 = v143 + 1;
LABEL_119:
        LODWORD(v143) = v83;
        if ( *(_QWORD *)v33 != -1 || *(_DWORD *)(v33 + 16) != -1 )
          --HIDWORD(v143);
        v35 = 1;
        *(__m128i *)v33 = _mm_loadu_si128((const __m128i *)n);
        v85 = v146;
        *(_DWORD *)(v33 + 24) = 1;
        *(_DWORD *)(v33 + 16) = v85;
LABEL_36:
        v36 = 2 * v35;
        v37 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
        v38 = v37;
        if ( *(_BYTE *)v37 == 19 )
        {
          v39 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
          do
          {
            if ( !*(_DWORD *)(v39 + 24) )
              break;
            v39 = *(_QWORD *)(v39 + 8 * (1LL - *(unsigned int *)(v39 + 8)));
          }
          while ( *(_BYTE *)v39 == 19 );
LABEL_40:
          v38 = *(_QWORD *)(v37 - 8LL * *(unsigned int *)(v37 + 8));
          v37 = v39;
        }
        else if ( *(_BYTE *)v37 != 15 )
        {
          v39 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
          goto LABEL_40;
        }
        v40 = (__int64 *)(*(_QWORD *)(v4 + 16) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v4 + 16) & 4) != 0 )
          v40 = (__int64 *)*v40;
        v41 = sub_15C0C90(v40, v37, v38, v36, 0, 1);
        v42 = 0;
        if ( *(_DWORD *)(v4 + 8) == 2 )
          v42 = *(_QWORD *)(v4 - 8);
        v43 = (__int64 *)(*(_QWORD *)(v4 + 16) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v4 + 16) & 4) != 0 )
          v43 = (__int64 *)*v43;
        v4 = sub_15B9E00(v43, *(_DWORD *)(v4 + 4), *(unsigned __int16 *)(v4 + 2), v41, v42, 0, 1);
        goto LABEL_48;
      }
LABEL_166:
      if ( *(_DWORD *)(v22 + 40) == 1 )
        goto LABEL_54;
      v104 = sub_1B79A80((__int64)&v141, (__int64)n, &v135);
      v105 = v135.m128i_i64[0];
      if ( !v104 )
      {
        ++v141;
        v112 = v143 + 1;
        v113 = v144;
        if ( 4 * ((int)v143 + 1) >= 3 * v144 )
        {
          v113 = 2 * v144;
        }
        else if ( v144 - HIDWORD(v143) - v112 > v144 >> 3 )
        {
          goto LABEL_177;
        }
        sub_1B7A020((__int64)&v141, v113);
        sub_1B79A80((__int64)&v141, (__int64)n, &v135);
        v105 = v135.m128i_i64[0];
        v112 = v143 + 1;
LABEL_177:
        LODWORD(v143) = v112;
        if ( *(_QWORD *)v105 != -1 || *(_DWORD *)(v105 + 16) != -1 )
          --HIDWORD(v143);
        *(__m128i *)v105 = _mm_loadu_si128((const __m128i *)n);
        v114 = v146;
        *(_DWORD *)(v105 + 24) = 0;
        *(_DWORD *)(v105 + 16) = v114;
        goto LABEL_48;
      }
      v34 = *(_DWORD *)(v135.m128i_i64[0] + 24);
LABEL_33:
      if ( v34 )
      {
        v35 = v34 & 0xFFF;
        if ( (v34 & 0xFE0) != 0 )
          v35 = v34 & 0x1F | (2 * (_WORD)v35) & 0x1FC0 | 0x20;
        goto LABEL_36;
      }
LABEL_48:
      sub_15C7080(&v135, v4);
      if ( (__m128i *)(v2 + 24) == &v135 )
      {
        if ( v135.m128i_i64[0] )
          sub_161E7C0(v2 + 24, v135.m128i_i64[0]);
      }
      else
      {
        v44 = *(_QWORD *)(v2 + 24);
        if ( v44 )
          sub_161E7C0(v2 + 24, v44);
        v45 = (unsigned __int8 *)v135.m128i_i64[0];
        *(_QWORD *)(v2 + 24) = v135.m128i_i64[0];
        if ( v45 )
          sub_1623210((__int64)&v135, v45, v2 + 24);
      }
      v126 = 1;
LABEL_54:
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v132 != v2 );
LABEL_55:
    v131 = *(_QWORD *)(v131 + 8);
  }
  while ( v120 != v131 );
  v129 = *(_QWORD *)(a1 + 80);
  if ( v120 != v129 )
  {
    while ( 1 )
    {
      n[0] = 0;
      n[1] = 0;
      v146 = 0;
      v147 = 0;
      if ( !v129 )
        BUG();
      v46 = *(_QWORD *)(v129 + 24);
      v47 = v129 + 16;
      if ( v46 != v129 + 16 )
        break;
      v64 = 0;
LABEL_80:
      j___libc_free_0(v64);
      v129 = *(_QWORD *)(v129 + 8);
      if ( v131 == v129 )
        goto LABEL_81;
    }
    while ( 2 )
    {
      while ( 2 )
      {
        if ( !v46 )
          BUG();
        if ( *(_BYTE *)(v46 - 8) != 78 )
          goto LABEL_60;
        v48 = *(_QWORD *)(v46 - 48);
        if ( !*(_BYTE *)(v48 + 16) && (*(_BYTE *)(v48 + 33) & 0x20) != 0 )
          goto LABEL_60;
        v49 = (__int64 *)(v46 + 24);
        v50 = sub_15C70A0(v46 + 24);
        v51 = v50;
        if ( !v50 )
          goto LABEL_60;
        v52 = *(_DWORD *)(v50 + 4);
        v53 = *(_QWORD *)(v50 - 8LL * *(unsigned int *)(v50 + 8));
        if ( *(_BYTE *)v53 == 15 || (v53 = *(_QWORD *)(v53 - 8LL * *(unsigned int *)(v53 + 8))) != 0 )
        {
          v54 = *(const char **)(v53 - 8LL * *(unsigned int *)(v53 + 8));
          if ( v54 )
          {
            v133 = v52;
            v55 = sub_161E970((__int64)v54);
            v52 = v133;
            v54 = (const char *)v55;
          }
          else
          {
            v56 = 0;
          }
        }
        else
        {
          v56 = 0;
          v54 = byte_3F871B3;
        }
        v135.m128i_i64[0] = (__int64)v54;
        v135.m128i_i64[1] = v56;
        v136 = v52;
        v124 = sub_1B79C30((__int64)n, (__int64)&v135, &v134);
        v57 = v134;
        if ( !v124 )
        {
          ++n[0];
          v86 = v146 + 1;
          v87 = v147;
          if ( 4 * ((int)v146 + 1) >= (unsigned int)(3 * v147) )
          {
            v87 = 2 * v147;
          }
          else if ( (int)v147 - HIDWORD(v146) - v86 > (unsigned int)v147 >> 3 )
          {
            goto LABEL_125;
          }
          sub_1B7A1D0((__int64)n, v87);
          sub_1B79C30((__int64)n, (__int64)&v135, &v134);
          v57 = v134;
          v86 = v146 + 1;
LABEL_125:
          LODWORD(v146) = v86;
          if ( *(_QWORD *)v57 != -1 || *(_DWORD *)(v57 + 16) != -1 )
            --HIDWORD(v146);
          *(__m128i *)v57 = _mm_loadu_si128(&v135);
          *(_DWORD *)(v57 + 16) = v136;
LABEL_60:
          v46 = *(_QWORD *)(v46 + 8);
          if ( v47 == v46 )
            goto LABEL_79;
          continue;
        }
        break;
      }
      v58 = sub_1B79A80((__int64)&v141, (__int64)&v135, &v134);
      v59 = v134;
      if ( v58 )
      {
        v60 = *(_DWORD *)(v134 + 24);
        v61 = v60 + 1;
        *(_DWORD *)(v134 + 24) = v60 + 1;
        if ( v60 == -1 )
        {
          sub_15C7080(&v134, v51);
          if ( v49 != &v134 )
            goto LABEL_74;
LABEL_147:
          if ( v134 )
            sub_161E7C0((__int64)&v134, v134);
          goto LABEL_78;
        }
        v90 = (2 * v61) & 0x1FFE;
        if ( (((_WORD)v60 + 1) & 0xFE0) != 0 )
          v90 = (2 * ((2 * v61) & 0x1FC0 | v61 & 0x1F)) | 0x40;
LABEL_135:
        v92 = *(_QWORD *)(v51 - 8LL * *(unsigned int *)(v51 + 8));
        v93 = v92;
        if ( *(_BYTE *)v92 == 19 )
        {
          v94 = *(_QWORD *)(v51 - 8LL * *(unsigned int *)(v51 + 8));
          do
          {
            if ( !*(_DWORD *)(v94 + 24) )
              break;
            v94 = *(_QWORD *)(v94 + 8 * (1LL - *(unsigned int *)(v94 + 8)));
          }
          while ( *(_BYTE *)v94 == 19 );
LABEL_139:
          v93 = *(_QWORD *)(v92 - 8LL * *(unsigned int *)(v92 + 8));
          v92 = v94;
        }
        else if ( *(_BYTE *)v92 != 15 )
        {
          v94 = *(_QWORD *)(v51 - 8LL * *(unsigned int *)(v51 + 8));
          goto LABEL_139;
        }
        v95 = (__int64 *)(*(_QWORD *)(v51 + 16) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v51 + 16) & 4) != 0 )
          v95 = (__int64 *)*v95;
        v96 = sub_15C0C90(v95, v92, v93, v90, 0, 1);
        v97 = 0;
        if ( *(_DWORD *)(v51 + 8) == 2 )
          v97 = *(_QWORD *)(v51 - 8);
        v98 = (__int64 *)(*(_QWORD *)(v51 + 16) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v51 + 16) & 4) != 0 )
          v98 = (__int64 *)*v98;
        v99 = sub_15B9E00(v98, *(_DWORD *)(v51 + 4), *(unsigned __int16 *)(v51 + 2), v96, v97, 0, 1);
        sub_15C7080(&v134, v99);
        if ( v49 == &v134 )
          goto LABEL_147;
LABEL_74:
        v62 = *(_QWORD *)(v46 + 24);
        if ( v62 )
          sub_161E7C0(v46 + 24, v62);
        v63 = (unsigned __int8 *)v134;
        *(_QWORD *)(v46 + 24) = v134;
        if ( v63 )
          sub_1623210((__int64)&v134, v63, v46 + 24);
LABEL_78:
        v46 = *(_QWORD *)(v46 + 8);
        v126 = v124;
        if ( v47 == v46 )
        {
LABEL_79:
          v64 = n[1];
          goto LABEL_80;
        }
        continue;
      }
      break;
    }
    ++v141;
    v88 = v143 + 1;
    v89 = v144;
    if ( 4 * ((int)v143 + 1) >= 3 * v144 )
    {
      v89 = 2 * v144;
    }
    else if ( v144 - HIDWORD(v143) - v88 > v144 >> 3 )
    {
      goto LABEL_131;
    }
    sub_1B7A020((__int64)&v141, v89);
    sub_1B79A80((__int64)&v141, (__int64)&v135, &v134);
    v59 = v134;
    v88 = v143 + 1;
LABEL_131:
    LODWORD(v143) = v88;
    if ( *(_QWORD *)v59 != -1 || *(_DWORD *)(v59 + 16) != -1 )
      --HIDWORD(v143);
    v90 = 2;
    *(__m128i *)v59 = _mm_loadu_si128(&v135);
    v91 = v136;
    *(_DWORD *)(v59 + 24) = 1;
    *(_DWORD *)(v59 + 16) = v91;
    goto LABEL_135;
  }
LABEL_81:
  v65 = v142;
LABEL_82:
  j___libc_free_0(v65);
  if ( v140 )
  {
    v66 = v138;
    v67 = v138 + 56LL * v140;
    do
    {
      while ( *(_QWORD *)v66 == -1 )
      {
        if ( *(_DWORD *)(v66 + 16) != -1 )
          goto LABEL_85;
LABEL_86:
        v66 += 56;
        if ( v67 == v66 )
          goto LABEL_91;
      }
      if ( *(_QWORD *)v66 != -2 || *(_DWORD *)(v66 + 16) != -2 )
      {
LABEL_85:
        j___libc_free_0(*(_QWORD *)(v66 + 32));
        goto LABEL_86;
      }
      v66 += 56;
    }
    while ( v67 != v66 );
  }
LABEL_91:
  j___libc_free_0(v138);
  return v126;
}
