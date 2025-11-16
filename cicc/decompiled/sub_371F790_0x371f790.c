// Function: sub_371F790
// Address: 0x371f790
//
unsigned __int64 __fastcall sub_371F790(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  _QWORD *v7; // r15
  __int64 v8; // rbx
  __int64 v10; // rdx
  unsigned int v11; // esi
  __int64 v12; // rax
  __int64 v13; // r15
  int v14; // r11d
  __int64 *v15; // r12
  unsigned int v16; // edi
  __int64 *v17; // rdx
  _QWORD *v18; // r14
  _QWORD *v19; // rax
  char *v20; // r12
  char *v21; // r15
  unsigned __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 *v24; // rdi
  __int64 *v25; // r12
  __int64 *v26; // rbx
  __int64 v27; // rax
  _BYTE *v28; // rsi
  __int64 *v29; // rax
  __int64 v30; // r8
  __int64 *v31; // r14
  __int64 v32; // rbx
  char **v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r8
  unsigned __int64 *v36; // r12
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *v39; // r11
  _QWORD *v40; // r15
  _QWORD *v41; // r12
  __int64 **v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rbx
  int v45; // eax
  __int64 v46; // rbx
  __m128i *v47; // rax
  __int64 v48; // rsi
  __m128i *v49; // rbx
  __int64 m128i_i64; // rdx
  unsigned int v51; // esi
  __int64 v52; // r10
  __int64 v53; // rdi
  __int64 v54; // rcx
  __m128i **v55; // rax
  __m128i *v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rbx
  __int64 v59; // r10
  unsigned int v60; // esi
  __int64 v61; // rdx
  __int64 v62; // r8
  unsigned int v63; // edi
  _QWORD *v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rsi
  _QWORD *v69; // rdx
  __int64 v70; // rdi
  __int64 v71; // rdi
  _QWORD *v72; // rcx
  __int64 v73; // rsi
  _QWORD *v74; // rdx
  _QWORD *v75; // rdi
  __int64 v76; // rcx
  __int64 v77; // rcx
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  int v80; // r10d
  int v81; // r10d
  __int64 v82; // r8
  unsigned int v83; // eax
  int v84; // ecx
  __m128i **v85; // r11
  __m128i *v86; // rdx
  int v87; // r9d
  __m128i **v88; // rdi
  int v89; // r10d
  int v90; // r10d
  int v91; // ecx
  unsigned int v92; // eax
  _QWORD *v93; // r11
  __int64 v94; // rsi
  int v95; // r8d
  _QWORD *v96; // rdi
  int v97; // eax
  int v98; // eax
  int v99; // eax
  int v100; // r8d
  __int64 v101; // r10
  _QWORD *v102; // rsi
  unsigned int v103; // eax
  __int64 v104; // rdi
  int v105; // eax
  int v106; // esi
  __int64 v107; // r10
  int v108; // r8d
  unsigned int v109; // edi
  __m128i **v110; // rax
  __m128i *v111; // rdx
  int v112; // eax
  __m128i v113; // xmm1
  unsigned int v114; // edx
  __int64 v115; // r11
  int v116; // esi
  __int64 *v117; // rcx
  __int64 *v118; // rdx
  int v119; // esi
  unsigned int v120; // ecx
  __int64 v121; // r8
  _QWORD *v122; // [rsp+8h] [rbp-108h]
  __int64 v123; // [rsp+10h] [rbp-100h]
  int v124; // [rsp+10h] [rbp-100h]
  unsigned int v125; // [rsp+10h] [rbp-100h]
  _QWORD *v126; // [rsp+20h] [rbp-F0h]
  __int64 v127; // [rsp+20h] [rbp-F0h]
  int v128; // [rsp+28h] [rbp-E8h]
  int v129; // [rsp+28h] [rbp-E8h]
  __int64 v130; // [rsp+28h] [rbp-E8h]
  __int64 v131; // [rsp+30h] [rbp-E0h]
  __int64 v132; // [rsp+30h] [rbp-E0h]
  __int64 v133; // [rsp+38h] [rbp-D8h]
  __int64 **v134; // [rsp+38h] [rbp-D8h]
  unsigned int v135; // [rsp+38h] [rbp-D8h]
  __int64 v136; // [rsp+38h] [rbp-D8h]
  __int64 v137; // [rsp+38h] [rbp-D8h]
  __int64 v138; // [rsp+38h] [rbp-D8h]
  __int64 v139; // [rsp+48h] [rbp-C8h] BYREF
  __m128i v140; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v141; // [rsp+60h] [rbp-B0h]
  char **v142; // [rsp+70h] [rbp-A0h] BYREF
  _QWORD *v143; // [rsp+78h] [rbp-98h]
  __int64 v144; // [rsp+80h] [rbp-90h]
  __int64 v145; // [rsp+90h] [rbp-80h] BYREF
  _QWORD *v146; // [rsp+98h] [rbp-78h]
  __int64 v147; // [rsp+A0h] [rbp-70h]
  unsigned int v148; // [rsp+A8h] [rbp-68h]
  __m128i v149; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v150; // [rsp+C0h] [rbp-50h]
  char v151; // [rsp+D0h] [rbp-40h]
  char v152; // [rsp+D1h] [rbp-3Fh]

  v122 = a2 + 6;
  result = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 6 != (_QWORD *)result )
  {
    v7 = (_QWORD *)a2[7];
    if ( !v7 )
LABEL_214:
      BUG();
    if ( *((_BYTE *)v7 - 24) == 84 )
    {
      v8 = a2[2];
      v145 = 0;
      v146 = 0;
      v147 = 0;
      v148 = 0;
      if ( v8 )
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)(v8 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            goto LABEL_14;
        }
        a6 = (__int64)a2;
        v11 = 0;
        v12 = 0;
LABEL_10:
        v13 = *(_QWORD *)(v10 + 40);
        if ( !v11 )
        {
          ++v145;
          goto LABEL_169;
        }
        v14 = 1;
        v15 = 0;
        a4 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
        v16 = (v11 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v17 = (__int64 *)(v12 + 32LL * v16);
        a5 = *v17;
        if ( v13 == *v17 )
          goto LABEL_12;
        while ( 1 )
        {
          if ( a5 == -4096 )
          {
            if ( !v15 )
              v15 = v17;
            ++v145;
            v112 = v147 + 1;
            if ( 4 * ((int)v147 + 1) < 3 * v11 )
            {
              if ( v11 - (v112 + HIDWORD(v147)) > v11 >> 3 )
                goto LABEL_157;
              v132 = a6;
              sub_371F5A0((__int64)&v145, v11);
              if ( v148 )
              {
                v118 = 0;
                a6 = v132;
                v119 = 1;
                v120 = (v148 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
                v112 = v147 + 1;
                v15 = &v146[4 * v120];
                v121 = *v15;
                if ( v13 != *v15 )
                {
                  while ( v121 != -4096 )
                  {
                    if ( !v118 && v121 == -8192 )
                      v118 = v15;
                    v120 = (v148 - 1) & (v119 + v120);
                    v15 = &v146[4 * v120];
                    v121 = *v15;
                    if ( v13 == *v15 )
                      goto LABEL_157;
                    ++v119;
                  }
                  if ( v118 )
                    v15 = v118;
                }
                goto LABEL_157;
              }
              goto LABEL_215;
            }
LABEL_169:
            v138 = a6;
            sub_371F5A0((__int64)&v145, 2 * v11);
            if ( v148 )
            {
              a6 = v138;
              v114 = (v148 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
              v112 = v147 + 1;
              v15 = &v146[4 * v114];
              v115 = *v15;
              if ( v13 != *v15 )
              {
                v116 = 1;
                v117 = 0;
                while ( v115 != -4096 )
                {
                  if ( v115 == -8192 && !v117 )
                    v117 = v15;
                  v114 = (v148 - 1) & (v116 + v114);
                  v15 = &v146[4 * v114];
                  v115 = *v15;
                  if ( v13 == *v15 )
                    goto LABEL_157;
                  ++v116;
                }
                if ( v117 )
                  v15 = v117;
              }
LABEL_157:
              LODWORD(v147) = v112;
              if ( *v15 != -4096 )
                --HIDWORD(v147);
              *v15 = v13;
              v15[1] = 0;
              v15[2] = 0;
              *((_DWORD *)v15 + 6) = 0;
              v137 = a6;
              sub_371F160((__int64)&v140, a1, v13, v149.m128i_i64[0], 0);
              v113 = _mm_loadu_si128(&v140);
              a6 = v137;
              v15[3] = v141;
              *(__m128i *)(v15 + 1) = v113;
              break;
            }
LABEL_215:
            LODWORD(v147) = v147 + 1;
            BUG();
          }
          if ( a5 == -8192 && !v15 )
            v15 = v17;
          v16 = (v11 - 1) & (v14 + v16);
          v17 = (__int64 *)(v12 + 32LL * v16);
          a5 = *v17;
          if ( v13 == *v17 )
            break;
          ++v14;
        }
LABEL_12:
        while ( 1 )
        {
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            break;
          v10 = *(_QWORD *)(v8 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
          {
            v12 = (__int64)v146;
            v11 = v148;
            goto LABEL_10;
          }
        }
        v7 = *(_QWORD **)(a6 + 56);
      }
LABEL_14:
      v18 = v7;
      if ( v122 == v7 )
        return sub_C7D6A0((__int64)v146, 32LL * v148, 8);
      while ( 1 )
      {
        if ( !v18 )
          goto LABEL_214;
        if ( *((_BYTE *)v18 - 24) != 84 )
          return sub_C7D6A0((__int64)v146, 32LL * v148, 8);
        v19 = sub_371EDF0(a1, (__int64)(v18 - 3), 0, a4, a5, a6, *(_OWORD *)&v149, v150);
        a4 = (unsigned int)v147;
        v131 = (__int64)v19;
        if ( (_DWORD)v147 )
        {
          v37 = v146;
          v38 = 4LL * v148;
          v39 = &v146[v38];
          if ( v146 != &v146[v38] )
          {
            while ( 1 )
            {
              v40 = v37;
              if ( *v37 != -8192 && *v37 != -4096 )
                break;
              v37 += 4;
              if ( v39 == v37 )
                goto LABEL_18;
            }
            if ( v39 != v37 )
            {
              v41 = &v146[v38];
              while ( 1 )
              {
                v42 = (__int64 **)*(v18 - 2);
                v127 = v40[2];
                v128 = *((_DWORD *)v40 + 6);
                v43 = *(_QWORD *)(v40[1] + 40LL);
                v44 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v44 == v43 + 48 )
                {
                  v46 = 0;
                }
                else
                {
                  if ( !v44 )
                    goto LABEL_214;
                  v45 = *(unsigned __int8 *)(v44 - 24);
                  v46 = v44 - 24;
                  if ( (unsigned int)(v45 - 30) >= 0xB )
                    v46 = 0;
                }
                v152 = 1;
                v149.m128i_i64[0] = (__int64)"pcp";
                v151 = 3;
                v134 = v42;
                v142 = (char **)v40[1];
                v143 = (_QWORD *)sub_ACA8A0(v42);
                v139 = (__int64)v134;
                v47 = (__m128i *)sub_371CDC0(0x22D7u, (__int64)&v139, 1, (__int64 *)&v142, 2, (__int64)&v149, v46);
                v48 = v18[3];
                v49 = v47;
                v149.m128i_i64[0] = v48;
                if ( v48 )
                {
                  sub_B96E90((__int64)&v149, v48, 1);
                  m128i_i64 = (__int64)v49[3].m128i_i64;
                  if ( &v49[3] == &v149 )
                  {
                    if ( v149.m128i_i64[0] )
                      sub_B91220((__int64)&v149, v149.m128i_i64[0]);
LABEL_58:
                    v51 = *(_DWORD *)(a1 + 112);
                    v52 = a1 + 88;
                    if ( !v51 )
                      goto LABEL_96;
                    goto LABEL_59;
                  }
                }
                else
                {
                  m128i_i64 = (__int64)v47[3].m128i_i64;
                  if ( &v47[3] == &v149 )
                    goto LABEL_58;
                }
                v78 = v49[3].m128i_i64[0];
                if ( v78 )
                {
                  v123 = m128i_i64;
                  sub_B91220(m128i_i64, v78);
                  m128i_i64 = v123;
                }
                v79 = (unsigned __int8 *)v149.m128i_i64[0];
                v49[3].m128i_i64[0] = v149.m128i_i64[0];
                if ( !v79 )
                  goto LABEL_58;
                sub_B976B0((__int64)&v149, v79, m128i_i64);
                v51 = *(_DWORD *)(a1 + 112);
                v52 = a1 + 88;
                if ( !v51 )
                {
LABEL_96:
                  ++*(_QWORD *)(a1 + 88);
                  goto LABEL_97;
                }
LABEL_59:
                v53 = *(_QWORD *)(a1 + 96);
                v54 = (v51 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
                v55 = (__m128i **)(v53 + 16 * v54);
                v56 = *v55;
                if ( v49 == *v55 )
                  goto LABEL_60;
                v124 = 1;
                v85 = 0;
                while ( v56 != (__m128i *)-4096LL )
                {
                  if ( v85 || v56 != (__m128i *)-8192LL )
                    v55 = v85;
                  LODWORD(v54) = (v51 - 1) & (v124 + v54);
                  v56 = *(__m128i **)(v53 + 16LL * (unsigned int)v54);
                  if ( v49 == v56 )
                    goto LABEL_60;
                  ++v124;
                  v85 = v55;
                  v55 = (__m128i **)(v53 + 16LL * (unsigned int)v54);
                }
                if ( !v85 )
                  v85 = v55;
                v97 = *(_DWORD *)(a1 + 104);
                ++*(_QWORD *)(a1 + 88);
                v84 = v97 + 1;
                if ( 4 * (v97 + 1) < 3 * v51 )
                {
                  if ( v51 - *(_DWORD *)(a1 + 108) - v84 <= v51 >> 3 )
                  {
                    v125 = ((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4);
                    sub_A41E30(v52, v51);
                    v105 = *(_DWORD *)(a1 + 112);
                    if ( !v105 )
                    {
LABEL_217:
                      ++*(_DWORD *)(a1 + 104);
                      BUG();
                    }
                    v106 = v105 - 1;
                    v107 = *(_QWORD *)(a1 + 96);
                    v108 = 1;
                    v109 = v106 & v125;
                    v84 = *(_DWORD *)(a1 + 104) + 1;
                    v110 = 0;
                    v85 = (__m128i **)(v107 + 16LL * (v106 & v125));
                    v111 = *v85;
                    if ( *v85 != v49 )
                    {
                      while ( v111 != (__m128i *)-4096LL )
                      {
                        if ( !v110 && v111 == (__m128i *)-8192LL )
                          v110 = v85;
                        v109 = v106 & (v108 + v109);
                        v85 = (__m128i **)(v107 + 16LL * v109);
                        v111 = *v85;
                        if ( v49 == *v85 )
                          goto LABEL_120;
                        ++v108;
                      }
                      if ( v110 )
                        v85 = v110;
                    }
                  }
                  goto LABEL_120;
                }
LABEL_97:
                sub_A41E30(v52, 2 * v51);
                v80 = *(_DWORD *)(a1 + 112);
                if ( !v80 )
                  goto LABEL_217;
                v81 = v80 - 1;
                v82 = *(_QWORD *)(a1 + 96);
                v83 = v81 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
                v84 = *(_DWORD *)(a1 + 104) + 1;
                v85 = (__m128i **)(v82 + 16LL * v83);
                v86 = *v85;
                if ( v49 != *v85 )
                {
                  v87 = 1;
                  v88 = 0;
                  while ( v86 != (__m128i *)-4096LL )
                  {
                    if ( !v88 && v86 == (__m128i *)-8192LL )
                      v88 = v85;
                    v83 = v81 & (v87 + v83);
                    v85 = (__m128i **)(v82 + 16LL * v83);
                    v86 = *v85;
                    if ( v49 == *v85 )
                      goto LABEL_120;
                    ++v87;
                  }
                  if ( v88 )
                    v85 = v88;
                }
LABEL_120:
                *(_DWORD *)(a1 + 104) = v84;
                if ( *v85 != (__m128i *)-4096LL )
                  --*(_DWORD *)(a1 + 108);
                *v85 = v49;
                *((_DWORD *)v85 + 2) = v128;
LABEL_60:
                v149.m128i_i64[0] = (__int64)v49;
                v149.m128i_i64[1] = v127;
                LODWORD(v150) = v128;
                v57 = *(_QWORD *)(v131 + 8);
                if ( v57 == *(_QWORD *)(v131 + 16) )
                {
                  sub_371E0F0(v131, (_BYTE *)v57, &v149);
                  v58 = *(_QWORD *)(v131 + 8);
                  v60 = *(_DWORD *)(a1 + 80);
                  v59 = a1 + 56;
                  v61 = *(_QWORD *)(v58 - 24);
                  if ( !v60 )
                    goto LABEL_105;
                }
                else
                {
                  if ( v57 )
                  {
                    *(__m128i *)v57 = _mm_loadu_si128(&v149);
                    *(_QWORD *)(v57 + 16) = v150;
                    v57 = *(_QWORD *)(v131 + 8);
                  }
                  v58 = v57 + 24;
                  v59 = a1 + 56;
                  *(_QWORD *)(v131 + 8) = v57 + 24;
                  v60 = *(_DWORD *)(a1 + 80);
                  v61 = *(_QWORD *)(v58 - 24);
                  if ( !v60 )
                  {
LABEL_105:
                    ++*(_QWORD *)(a1 + 56);
                    goto LABEL_106;
                  }
                }
                a6 = v60 - 1;
                v62 = *(_QWORD *)(a1 + 64);
                v135 = ((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4);
                v63 = a6 & v135;
                v64 = (_QWORD *)(v62 + 16LL * ((unsigned int)a6 & v135));
                v65 = *v64;
                if ( *v64 != v61 )
                {
                  v129 = 1;
                  v93 = 0;
                  while ( v65 != -4096 )
                  {
                    if ( v93 || v65 != -8192 )
                      v64 = v93;
                    v63 = a6 & (v129 + v63);
                    v65 = *(_QWORD *)(v62 + 16LL * v63);
                    if ( v61 == v65 )
                      goto LABEL_65;
                    ++v129;
                    v93 = v64;
                    v64 = (_QWORD *)(v62 + 16LL * v63);
                  }
                  if ( !v93 )
                    v93 = v64;
                  v98 = *(_DWORD *)(a1 + 72);
                  ++*(_QWORD *)(a1 + 56);
                  v91 = v98 + 1;
                  if ( 4 * (v98 + 1) >= 3 * v60 )
                  {
LABEL_106:
                    v136 = v61;
                    sub_371EC10(v59, 2 * v60);
                    v89 = *(_DWORD *)(a1 + 80);
                    if ( !v89 )
                      goto LABEL_216;
                    v61 = v136;
                    v90 = v89 - 1;
                    a6 = *(_QWORD *)(a1 + 64);
                    v91 = *(_DWORD *)(a1 + 72) + 1;
                    v92 = v90 & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
                    v93 = (_QWORD *)(a6 + 16LL * v92);
                    v94 = *v93;
                    if ( v136 != *v93 )
                    {
                      v95 = 1;
                      v96 = 0;
                      while ( v94 != -4096 )
                      {
                        if ( !v96 && v94 == -8192 )
                          v96 = v93;
                        v92 = v90 & (v95 + v92);
                        v93 = (_QWORD *)(a6 + 16LL * v92);
                        v94 = *v93;
                        if ( v136 == *v93 )
                          goto LABEL_129;
                        ++v95;
                      }
                      if ( v96 )
                        v93 = v96;
                    }
                  }
                  else if ( v60 - *(_DWORD *)(a1 + 76) - v91 <= v60 >> 3 )
                  {
                    v130 = v61;
                    sub_371EC10(v59, v60);
                    v99 = *(_DWORD *)(a1 + 80);
                    if ( !v99 )
                    {
LABEL_216:
                      ++*(_DWORD *)(a1 + 72);
                      BUG();
                    }
                    v100 = v99 - 1;
                    v101 = *(_QWORD *)(a1 + 64);
                    v102 = 0;
                    v61 = v130;
                    a6 = 1;
                    v103 = v100 & v135;
                    v91 = *(_DWORD *)(a1 + 72) + 1;
                    v93 = (_QWORD *)(v101 + 16LL * (v100 & v135));
                    v104 = *v93;
                    if ( v130 != *v93 )
                    {
                      while ( v104 != -4096 )
                      {
                        if ( !v102 && v104 == -8192 )
                          v102 = v93;
                        v103 = v100 & (a6 + v103);
                        v93 = (_QWORD *)(v101 + 16LL * v103);
                        v104 = *v93;
                        if ( v130 == *v93 )
                          goto LABEL_129;
                        a6 = (unsigned int)(a6 + 1);
                      }
                      if ( v102 )
                        v93 = v102;
                    }
                  }
LABEL_129:
                  *(_DWORD *)(a1 + 72) = v91;
                  if ( *v93 != -4096 )
                    --*(_DWORD *)(a1 + 76);
                  *v93 = v61;
                  v93[1] = v131;
                }
LABEL_65:
                a4 = *((unsigned int *)v18 - 5);
                v66 = 0;
                a5 = 1;
                if ( (a4 & 0x7FFFFFF) != 0 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v67 = *(v18 - 4);
                      if ( *v40 == *(_QWORD *)(v67 + 32LL * *((unsigned int *)v18 + 12) + 8 * v66) )
                        break;
                      if ( ((unsigned int)a4 & 0x7FFFFFF) <= (unsigned int)++v66 )
                        goto LABEL_84;
                    }
                    a6 = *(_QWORD *)(v58 - 24);
                    v68 = *(_QWORD *)(v67 + 32 * v66);
                    v69 = (_QWORD *)(a6 + 32 * (1LL - (*(_DWORD *)(a6 + 4) & 0x7FFFFFF)));
                    if ( *v69 )
                    {
                      a6 = v69[2];
                      v70 = v69[1];
                      *(_QWORD *)a6 = v70;
                      if ( v70 )
                      {
                        a6 = v69[2];
                        *(_QWORD *)(v70 + 16) = a6;
                      }
                    }
                    *v69 = v68;
                    if ( v68 )
                    {
                      v71 = *(_QWORD *)(v68 + 16);
                      a6 = v68 + 16;
                      v69[1] = v71;
                      if ( v71 )
                        *(_QWORD *)(v71 + 16) = v69 + 1;
                      v69[2] = a6;
                      *(_QWORD *)(v68 + 16) = v69;
                    }
                    v72 = (_QWORD *)(*(v18 - 4) + 32 * v66);
                    v73 = *(_QWORD *)(v58 - 24);
                    v74 = v72;
                    if ( *v72 )
                    {
                      v75 = (_QWORD *)v72[2];
                      v76 = v72[1];
                      *v75 = v76;
                      if ( v76 )
                        *(_QWORD *)(v76 + 16) = v74[2];
                    }
                    *v74 = v73;
                    if ( v73 )
                    {
                      v77 = *(_QWORD *)(v73 + 16);
                      v74[1] = v77;
                      if ( v77 )
                      {
                        a6 = (__int64)(v74 + 1);
                        *(_QWORD *)(v77 + 16) = v74 + 1;
                      }
                      v74[2] = v73 + 16;
                      *(_QWORD *)(v73 + 16) = v74;
                    }
                    a4 = *((unsigned int *)v18 - 5);
                    ++v66;
                  }
                  while ( (*((_DWORD *)v18 - 5) & 0x7FFFFFFu) > (unsigned int)v66 );
                }
LABEL_84:
                v40 += 4;
                if ( v40 != v41 )
                {
                  while ( *v40 == -8192 || *v40 == -4096 )
                  {
                    v40 += 4;
                    if ( v41 == v40 )
                      goto LABEL_18;
                  }
                  if ( v41 != v40 )
                    continue;
                }
                break;
              }
            }
          }
        }
LABEL_18:
        v20 = *(char **)(v131 + 8);
        v21 = *(char **)v131;
        if ( *(char **)v131 != v20 )
        {
          _BitScanReverse64(&v22, 0xAAAAAAAAAAAAAAABLL * ((v20 - v21) >> 3));
          sub_371E870(*(_QWORD *)v131, *(__m128i **)(v131 + 8), 2LL * (int)(63 - (v22 ^ 0x3F)), a4, a5, a6);
          if ( v20 - v21 <= 384 )
          {
            sub_371D0F0(v21, v20);
          }
          else
          {
            v23 = (__int64 *)(v21 + 384);
            sub_371D0F0(v21, (_QWORD *)v21 + 48);
            if ( v20 != v21 + 384 )
            {
              do
              {
                v24 = v23;
                v23 += 3;
                sub_371D0A0(v24);
              }
              while ( v20 != (char *)v23 );
            }
          }
        }
        if ( *(_BYTE *)(a1 + 32) )
          break;
LABEL_23:
        v18 = (_QWORD *)v18[1];
        if ( v122 == v18 )
          return sub_C7D6A0((__int64)v146, 32LL * v148, 8);
      }
      v25 = (__int64 *)*(v18 - 4);
      v26 = &v25[4 * (*((_DWORD *)v18 - 5) & 0x7FFFFFF)];
      if ( v25 == v26 )
      {
LABEL_40:
        v32 = *(_QWORD *)(*(v18 - 1) + 24LL);
        v33 = (char **)sub_371EDF0(a1, v32, 0, a4, a5, a6, *(_OWORD *)&v149, v150);
        v139 = v32;
        v142 = v33;
        v36 = (unsigned __int64 *)v33;
        v144 = a1;
        v143 = &v139;
        if ( v33 == (char **)v131 )
        {
          sub_371DC50(&v142);
        }
        else if ( !(unsigned __int8)sub_371D8F0(a1, v33, (__int64 *)v131, v34, v35) )
        {
          sub_371DC50(&v142);
          sub_371E290(a1, v36, (__int64 *)v131);
        }
        goto LABEL_23;
      }
      v126 = v18;
      while ( 1 )
      {
        a4 = *v25;
        if ( *(_BYTE *)*v25 != 85 )
          goto LABEL_29;
        v27 = *(_QWORD *)(a4 - 32);
        if ( !v27 )
          goto LABEL_29;
        if ( *(_BYTE *)v27 )
          goto LABEL_29;
        if ( *(_QWORD *)(v27 + 24) != *(_QWORD *)(a4 + 80) )
          goto LABEL_29;
        if ( (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
          goto LABEL_29;
        v133 = *v25;
        v28 = *(_BYTE **)(a4 + 32 * (1LL - (*(_DWORD *)(a4 + 4) & 0x7FFFFFF)));
        if ( *v28 <= 0x1Cu )
          goto LABEL_29;
        v29 = sub_371EDF0(a1, (__int64)v28, 0, a4, a5, a6, *(_OWORD *)&v149, v150);
        v142 = (char **)v131;
        v31 = v29;
        v144 = a1;
        v139 = v133;
        v143 = &v139;
        if ( v29 == (__int64 *)v131 )
          break;
        if ( (unsigned __int8)sub_371D8F0(a1, (_QWORD *)v131, v29, v133, v30) )
        {
LABEL_29:
          v25 += 4;
          if ( v26 == v25 )
            goto LABEL_39;
        }
        else
        {
          v25 += 4;
          sub_371DC50(&v142);
          sub_371E290(a1, (unsigned __int64 *)v131, v31);
          if ( v26 == v25 )
          {
LABEL_39:
            v18 = v126;
            goto LABEL_40;
          }
        }
      }
      sub_371DC50(&v142);
      goto LABEL_29;
    }
  }
  return result;
}
