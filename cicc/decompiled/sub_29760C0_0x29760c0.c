// Function: sub_29760C0
// Address: 0x29760c0
//
__int64 __fastcall sub_29760C0(__int64 a1, __int64 **a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rcx
  __int64 v7; // r8
  unsigned __int64 v8; // r9
  __int64 *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  char v15; // bl
  const char **v16; // r14
  __int8 *v17; // rcx
  unsigned __int64 v18; // rdi
  const __m128i *v19; // rdx
  __int64 v20; // rbx
  unsigned __int64 v21; // rsi
  __m128i *v22; // r13
  __int64 v23; // rax
  __m128i *v24; // rax
  __m128i *v25; // rcx
  __int64 v26; // r13
  unsigned __int64 v27; // rbx
  int v28; // edi
  char *v29; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  int v34; // r13d
  __int64 v35; // rax
  int v36; // r12d
  unsigned __int64 v37; // rdx
  __int64 v38; // r13
  __int64 *v39; // r15
  __int64 *v40; // rax
  __int64 *v41; // r11
  _BYTE *v42; // rbx
  unsigned __int64 *v43; // r12
  __int64 v44; // r13
  __m128i v45; // rax
  __int64 v46; // rax
  __int64 v47; // r12
  _QWORD *v48; // rax
  __int64 v49; // rcx
  _QWORD *v50; // rbx
  __int64 *v51; // rax
  __int64 *v52; // rcx
  __int64 *v53; // rdx
  __int64 *v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // r8
  __int64 v57; // r8
  __int64 *v58; // r13
  __int64 v59; // r12
  unsigned __int64 v60; // rbx
  _QWORD *v61; // rcx
  int v62; // r15d
  unsigned __int64 v63; // rsi
  _QWORD *v64; // r15
  __int64 *v65; // r9
  const char **v66; // rdx
  __int64 *v67; // r14
  unsigned __int64 v68; // r13
  _QWORD *v69; // rbx
  __int64 v70; // rdi
  __int64 v71; // r15
  int v72; // eax
  int v73; // eax
  unsigned int v74; // r8d
  __int64 v75; // rax
  __int64 v76; // r8
  __int64 v77; // r8
  __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // r15
  __int64 v81; // rsi
  const char **v82; // rbx
  __m128i *v83; // rsi
  __int64 v84; // rsi
  unsigned __int8 *v85; // rsi
  unsigned __int64 *v86; // rax
  _BYTE *v87; // rbx
  unsigned __int64 v88; // r12
  unsigned __int64 v89; // rdi
  __int64 v90; // rsi
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  unsigned int v95; // r12d
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  _QWORD *v100; // rbx
  _QWORD *v101; // r15
  void (__fastcall *v102)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v103; // rax
  __int64 v105; // rsi
  unsigned __int8 *v106; // rsi
  unsigned __int64 v107; // r12
  unsigned __int64 v108; // rax
  __int64 v109; // rdx
  int v110; // edx
  char *v111; // rax
  __int64 v112; // rdx
  char *v113; // rsi
  __int64 v114; // rdx
  __int64 v115; // r12
  __int64 v116; // rdx
  unsigned int v117; // eax
  __int64 *v118; // rdx
  char v119; // al
  char v120; // r14
  __int64 v121; // rcx
  __int64 v122; // r8
  __int64 v123; // r9
  unsigned __int8 v125; // [rsp+27h] [rbp-4E9h]
  __int64 v128; // [rsp+38h] [rbp-4D8h]
  _BYTE *v130; // [rsp+48h] [rbp-4C8h]
  __int64 v131; // [rsp+58h] [rbp-4B8h]
  __int64 v132; // [rsp+60h] [rbp-4B0h]
  const char **v133; // [rsp+70h] [rbp-4A0h]
  __int64 *v134; // [rsp+78h] [rbp-498h]
  _QWORD *v135; // [rsp+78h] [rbp-498h]
  _BYTE *v136; // [rsp+80h] [rbp-490h]
  _QWORD *v137; // [rsp+80h] [rbp-490h]
  _BYTE *v138; // [rsp+88h] [rbp-488h]
  __int64 v139; // [rsp+90h] [rbp-480h]
  __int64 *v140; // [rsp+90h] [rbp-480h]
  const char **v141; // [rsp+98h] [rbp-478h]
  __int64 *v142; // [rsp+98h] [rbp-478h]
  const __m128i **v143; // [rsp+A0h] [rbp-470h]
  _BYTE *v144; // [rsp+A8h] [rbp-468h]
  __int64 v145; // [rsp+A8h] [rbp-468h]
  unsigned int v146; // [rsp+B0h] [rbp-460h]
  unsigned __int16 v147; // [rsp+B0h] [rbp-460h]
  __int64 v148; // [rsp+B8h] [rbp-458h]
  _QWORD *v149; // [rsp+C0h] [rbp-450h]
  int v150; // [rsp+C0h] [rbp-450h]
  __int64 v151; // [rsp+C8h] [rbp-448h]
  __int64 *v152; // [rsp+C8h] [rbp-448h]
  __int64 v153; // [rsp+C8h] [rbp-448h]
  __m128i *v154; // [rsp+D0h] [rbp-440h] BYREF
  __int8 *v155; // [rsp+D8h] [rbp-438h]
  __m128i *v156; // [rsp+E0h] [rbp-430h]
  __int64 *v157; // [rsp+F0h] [rbp-420h] BYREF
  __int64 v158; // [rsp+F8h] [rbp-418h]
  _BYTE v159[16]; // [rsp+100h] [rbp-410h] BYREF
  __m128i v160; // [rsp+110h] [rbp-400h] BYREF
  char *v161; // [rsp+120h] [rbp-3F0h]
  __int16 v162; // [rsp+130h] [rbp-3E0h]
  __int64 v163; // [rsp+140h] [rbp-3D0h] BYREF
  __int64 v164; // [rsp+148h] [rbp-3C8h]
  __int64 v165; // [rsp+150h] [rbp-3C0h] BYREF
  unsigned int v166; // [rsp+158h] [rbp-3B8h]
  _BYTE *v167; // [rsp+170h] [rbp-3A0h] BYREF
  __int64 v168; // [rsp+178h] [rbp-398h]
  _BYTE v169[160]; // [rsp+180h] [rbp-390h] BYREF
  unsigned __int64 v170[2]; // [rsp+220h] [rbp-2F0h] BYREF
  _BYTE v171[512]; // [rsp+230h] [rbp-2E0h] BYREF
  __int64 v172; // [rsp+430h] [rbp-E0h]
  __int64 v173; // [rsp+438h] [rbp-D8h]
  unsigned __int64 *v174; // [rsp+440h] [rbp-D0h]
  __int64 v175; // [rsp+448h] [rbp-C8h]
  char v176; // [rsp+450h] [rbp-C0h]
  __int64 v177; // [rsp+458h] [rbp-B8h]
  char *v178; // [rsp+460h] [rbp-B0h]
  __int64 v179; // [rsp+468h] [rbp-A8h]
  int v180; // [rsp+470h] [rbp-A0h]
  char v181; // [rsp+474h] [rbp-9Ch]
  char v182; // [rsp+478h] [rbp-98h] BYREF
  __int16 v183; // [rsp+4B8h] [rbp-58h]
  _QWORD *v184; // [rsp+4C0h] [rbp-50h]
  _QWORD *v185; // [rsp+4C8h] [rbp-48h]
  __int64 v186; // [rsp+4D0h] [rbp-40h]

  v174 = a3;
  if ( a3 )
    a3 = v170;
  v170[0] = (unsigned __int64)v171;
  v170[1] = 0x1000000000LL;
  v132 = (__int64)a3;
  v172 = 0;
  v173 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v178 = &v182;
  v179 = 8;
  v180 = 0;
  v181 = 1;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v163 = 0;
  v164 = 1;
  v125 = sub_F62E00(a1, (__int64)a3, 0, (__int64)v170, 0, a6);
  v9 = &v165;
  do
    *(_DWORD *)v9++ = -1;
  while ( v9 != (__int64 *)&v167 );
  v167 = v169;
  v168 = 0x400000000LL;
  v10 = *(_QWORD *)(a1 + 80);
  v151 = a1 + 72;
  if ( v10 == a1 + 72 )
  {
    v154 = 0;
    v155 = 0;
    v156 = 0;
    goto LABEL_211;
  }
  do
  {
    while ( 1 )
    {
      v11 = v10 - 24;
      if ( !v10 )
        v11 = 0;
      if ( v132 )
      {
        if ( *(_BYTE *)(v132 + 560) )
        {
          v12 = *(unsigned int *)(v132 + 588);
          if ( (_DWORD)v12 != *(_DWORD *)(v132 + 592) )
          {
            if ( *(_BYTE *)(v132 + 596) )
            {
              v13 = *(_QWORD **)(v132 + 576);
              v14 = &v13[v12];
              if ( v13 != v14 )
              {
                while ( v11 != *v13 )
                {
                  if ( v14 == ++v13 )
                    goto LABEL_156;
                }
                goto LABEL_16;
              }
            }
            else if ( sub_C8CA60(v132 + 568, v11) )
            {
              goto LABEL_16;
            }
          }
        }
      }
LABEL_156:
      v107 = *(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v107 == v11 + 48 )
        goto LABEL_223;
      if ( !v107 )
        BUG();
      v150 = *(unsigned __int8 *)(v107 - 24);
      if ( (unsigned int)(v150 - 30) > 0xA )
LABEL_223:
        BUG();
      if ( (unsigned int)sub_B46E30(v107 - 24) || v150 != 30 && v150 != 35 )
        goto LABEL_16;
      if ( sub_AA4E50(v11) )
        goto LABEL_16;
      v108 = sub_B46BC0(v107 - 24, 0);
      if ( v108 )
      {
        if ( *(_BYTE *)v108 == 85 )
        {
          v109 = *(_QWORD *)(v108 - 32);
          if ( v109 )
          {
            if ( !*(_BYTE *)v109 && *(_QWORD *)(v109 + 24) == *(_QWORD *)(v108 + 80) && *(_DWORD *)(v109 + 36) == 146 )
              goto LABEL_16;
          }
        }
      }
      if ( (*(_BYTE *)(v107 - 17) & 0x40) != 0 )
      {
        v111 = *(char **)(v107 - 32);
        v110 = *(_DWORD *)(v107 - 20);
      }
      else
      {
        v110 = *(_DWORD *)(v107 - 20);
        v111 = (char *)(v107 - 24 - 32LL * (v110 & 0x7FFFFFF));
      }
      v112 = 32LL * (v110 & 0x7FFFFFF);
      v113 = &v111[v112];
      v6 = v112 >> 5;
      v114 = v112 >> 7;
      if ( !v114 )
        break;
      v114 = (__int64)&v111[128 * v114];
      while ( 1 )
      {
        v6 = *(_QWORD *)(*(_QWORD *)v111 + 8LL);
        if ( *(_BYTE *)(v6 + 8) == 11 )
          break;
        v6 = *(_QWORD *)(*((_QWORD *)v111 + 4) + 8LL);
        if ( *(_BYTE *)(v6 + 8) == 11 )
        {
          v111 += 32;
          break;
        }
        v6 = *(_QWORD *)(*((_QWORD *)v111 + 8) + 8LL);
        if ( *(_BYTE *)(v6 + 8) == 11 )
        {
          v111 += 64;
          break;
        }
        v6 = *(_QWORD *)(*((_QWORD *)v111 + 12) + 8LL);
        if ( *(_BYTE *)(v6 + 8) == 11 )
        {
          v111 += 96;
          break;
        }
        v111 += 128;
        if ( (char *)v114 == v111 )
        {
          v6 = (v113 - v111) >> 5;
          goto LABEL_202;
        }
      }
LABEL_178:
      if ( v113 == v111 )
        goto LABEL_179;
LABEL_16:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v151 == v10 )
        goto LABEL_17;
    }
LABEL_202:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          goto LABEL_179;
        goto LABEL_205;
      }
      v114 = *(_QWORD *)(*(_QWORD *)v111 + 8LL);
      if ( *(_BYTE *)(v114 + 8) == 11 )
        goto LABEL_178;
      v111 += 32;
    }
    v114 = *(_QWORD *)(*(_QWORD *)v111 + 8LL);
    if ( *(_BYTE *)(v114 + 8) == 11 )
      goto LABEL_178;
    v111 += 32;
LABEL_205:
    v114 = *(_QWORD *)(*(_QWORD *)v111 + 8LL);
    if ( *(_BYTE *)(v114 + 8) == 11 )
      goto LABEL_178;
LABEL_179:
    v160.m128i_i32[0] = *(unsigned __int8 *)(v107 - 24) - 29;
    v115 = sub_2975CE0((__int64)&v163, v160.m128i_i32, v114, v6, v7, v8);
    v6 = *(unsigned int *)(v115 + 12);
    v117 = *(_DWORD *)(v115 + 8);
    v116 = v117;
    if ( v117 >= v6 )
    {
      v8 = v117 + 1LL;
      if ( v6 < v8 )
      {
        sub_C8D5F0(v115, (const void *)(v115 + 16), v117 + 1LL, 8u, v7, v8);
        v116 = *(unsigned int *)(v115 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v115 + 8 * v116) = v11;
      ++*(_DWORD *)(v115 + 8);
      goto LABEL_16;
    }
    v6 = *(_QWORD *)v115;
    v118 = (__int64 *)(*(_QWORD *)v115 + 8LL * v117);
    if ( v118 )
    {
      *v118 = v11;
      v117 = *(_DWORD *)(v115 + 8);
    }
    *(_DWORD *)(v115 + 8) = v117 + 1;
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v151 != v10 );
LABEL_17:
  v154 = 0;
  v155 = 0;
  v138 = v167;
  v156 = 0;
  v130 = &v167[40 * (unsigned int)v168];
  if ( v167 != v130 )
  {
    v15 = 0;
    v16 = (const char **)&v160;
    while ( 1 )
    {
      v146 = *((_DWORD *)v138 + 4);
      v152 = (__int64 *)*((_QWORD *)v138 + 1);
      v157 = (__int64 *)v159;
      v158 = 0x100000000LL;
      if ( v132 )
      {
        if ( v146 > 1uLL )
        {
          v17 = v155;
          v18 = (unsigned __int64)v154;
          v19 = v154;
          v20 = v155 - (__int8 *)v154;
          v21 = v146 + ((v155 - (__int8 *)v154) >> 4);
          if ( v21 > 0x7FFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"vector::reserve");
          v143 = (const __m128i **)&v154;
          if ( v21 > v156 - v154 )
          {
            v22 = 0;
            if ( v21 )
            {
              v23 = sub_22077B0(16 * v21);
              v18 = (unsigned __int64)v154;
              v17 = v155;
              v22 = (__m128i *)v23;
              v19 = v154;
            }
            if ( v17 != (__int8 *)v18 )
            {
              v24 = v22;
              v25 = (__m128i *)&v17[(_QWORD)v22 - v18];
              do
              {
                if ( v24 )
                  *v24 = _mm_loadu_si128(v19);
                ++v24;
                ++v19;
              }
              while ( v24 != v25 );
            }
            if ( v18 )
              j_j___libc_free_0(v18);
            v154 = v22;
            v155 = &v22->m128i_i8[v20];
            v156 = &v22[v21];
            v143 = (const __m128i **)&v154;
          }
          goto LABEL_33;
        }
      }
      else if ( v146 > 1uLL )
      {
        v143 = 0;
LABEL_33:
        v26 = *v152;
        v27 = *(_QWORD *)(*v152 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v27 == *v152 + 48 )
          goto LABEL_223;
        if ( !v27 )
          goto LABEL_223;
        v28 = *(unsigned __int8 *)(v27 - 24);
        v144 = (_BYTE *)(v27 - 24);
        if ( (unsigned int)(v28 - 30) > 0xA )
          goto LABEL_223;
        v29 = sub_B458E0(v28 - 29);
        if ( *v29 )
        {
          v161 = v29;
          v160.m128i_i64[0] = (__int64)"common.";
          v162 = 771;
        }
        else
        {
          v160.m128i_i64[0] = (__int64)"common.";
          v162 = 259;
        }
        v30 = sub_B2BE50(a1);
        v31 = sub_22077B0(0x50u);
        v148 = v31;
        if ( v31 )
          sub_AA4D50(v31, v30, (__int64)v16, a1, v26);
        v34 = *(_DWORD *)(v27 - 20);
        v35 = (unsigned int)v158;
        v36 = v34 & 0x7FFFFFF;
        v37 = v34 & 0x7FFFFFF;
        if ( v37 == (unsigned int)v158 )
        {
          v39 = v157;
          v41 = &v157[v37];
        }
        else
        {
          v38 = v37;
          if ( v37 < (unsigned int)v158 )
          {
            v39 = v157;
            LODWORD(v158) = *(_DWORD *)(v27 - 20) & 0x7FFFFFF;
            v41 = &v157[v38];
            v34 = *(_DWORD *)(v27 - 20);
          }
          else
          {
            if ( v37 > HIDWORD(v158) )
            {
              sub_C8D5F0((__int64)&v157, v159, v37, 8u, v32, v33);
              v35 = (unsigned int)v158;
            }
            v39 = v157;
            v40 = &v157[v35];
            v41 = &v157[v38];
            if ( v40 != &v157[v38] )
            {
              do
              {
                if ( v40 )
                  *v40 = 0;
                ++v40;
              }
              while ( v41 != v40 );
              v39 = v157;
              v41 = &v157[v38];
            }
            LODWORD(v158) = v36;
            v34 = *(_DWORD *)(v27 - 20);
          }
        }
        if ( (*(_BYTE *)(v27 - 17) & 0x40) != 0 )
          v42 = *(_BYTE **)(v27 - 32);
        else
          v42 = &v144[-32 * (v34 & 0x7FFFFFF)];
        v43 = (unsigned __int64 *)(v148 + 48);
        v136 = &v42[32 * (v34 & 0x7FFFFFF)];
        if ( v39 != v41 && &v42[32 * (v34 & 0x7FFFFFF)] != v42 )
        {
          v134 = v41;
          v44 = v128;
          do
          {
            v45.m128i_i64[0] = (__int64)sub_BD5D20(v148);
            v160 = v45;
            v162 = 773;
            v161 = ".op";
            v139 = *(_QWORD *)(*(_QWORD *)v42 + 8LL);
            v46 = sub_BD2DA0(80);
            v47 = v46;
            if ( v46 )
            {
              sub_B44260(v46, v139, 55, 0x8000000u, 0, 0);
              *(_DWORD *)(v47 + 72) = v146;
              sub_BD6B50((unsigned __int8 *)v47, v16);
              sub_BD2A10(v47, *(_DWORD *)(v47 + 72), 1);
            }
            *v39 = v47;
            LOWORD(v44) = 0;
            v42 += 32;
            ++v39;
            sub_B44240((_QWORD *)v47, v148, (unsigned __int64 *)(v148 + 48), v44);
          }
          while ( v39 != v134 && v42 != v136 );
          v128 = v44;
          v43 = (unsigned __int64 *)(v148 + 48);
        }
        v48 = (_QWORD *)sub_B47F80(v144);
        v49 = v131;
        v50 = v48;
        v137 = v48;
        LOWORD(v49) = 0;
        v131 = v49;
        sub_B44240(v48, v148, v43, v49);
        if ( (*((_BYTE *)v50 + 7) & 0x40) != 0 )
        {
          v51 = (__int64 *)*(v137 - 1);
          v52 = &v51[4 * (*((_DWORD *)v137 + 1) & 0x7FFFFFF)];
        }
        else
        {
          v52 = v137;
          v51 = &v137[-4 * (*((_DWORD *)v137 + 1) & 0x7FFFFFF)];
        }
        v53 = v157;
        v54 = &v157[(unsigned int)v158];
        if ( v51 != v52 && v157 != v54 )
        {
          do
          {
            v55 = *v53;
            if ( *v51 )
            {
              v56 = v51[1];
              *(_QWORD *)v51[2] = v56;
              if ( v56 )
                *(_QWORD *)(v56 + 16) = v51[2];
            }
            *v51 = v55;
            if ( v55 )
            {
              v57 = *(_QWORD *)(v55 + 16);
              v51[1] = v57;
              if ( v57 )
                *(_QWORD *)(v57 + 16) = v51 + 1;
              v51[2] = v55 + 16;
              *(_QWORD *)(v55 + 16) = v51;
            }
            v51 += 4;
            ++v53;
          }
          while ( v51 != v52 && v54 != v53 );
        }
        v140 = &v152[v146];
        if ( v152 != v140 )
        {
          v58 = v152;
          v153 = 0;
          while ( 1 )
          {
            v59 = *v58;
            v60 = *(_QWORD *)(*v58 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v60 == *v58 + 48 )
              goto LABEL_222;
            if ( !v60 )
              goto LABEL_223;
            v149 = (_QWORD *)(v60 - 24);
            if ( (unsigned int)*(unsigned __int8 *)(v60 - 24) - 30 > 0xA )
LABEL_222:
              BUG();
            if ( (*(_BYTE *)(v60 - 17) & 0x40) != 0 )
            {
              v61 = *(_QWORD **)(v60 - 32);
              v62 = *(_DWORD *)(v60 - 20);
            }
            else
            {
              v62 = *(_DWORD *)(v60 - 20);
              v61 = &v149[-4 * (v62 & 0x7FFFFFF)];
            }
            v63 = (unsigned __int64)v157;
            v64 = &v61[4 * (v62 & 0x7FFFFFF)];
            v65 = &v157[(unsigned int)v158];
            if ( v64 != v61 && v157 != v65 )
            {
              v66 = v16;
              v67 = v58;
              v68 = *(_QWORD *)(*v58 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              v69 = v64;
              do
              {
                v70 = *(_QWORD *)v63;
                v71 = *v61;
                v72 = *(_DWORD *)(*(_QWORD *)v63 + 4LL) & 0x7FFFFFF;
                if ( v72 == *(_DWORD *)(*(_QWORD *)v63 + 72LL) )
                {
                  v133 = v66;
                  v135 = v61;
                  v142 = v65;
                  sub_B48D90(v70);
                  v66 = v133;
                  v61 = v135;
                  v65 = v142;
                  v72 = *(_DWORD *)(v70 + 4) & 0x7FFFFFF;
                }
                v73 = (v72 + 1) & 0x7FFFFFF;
                v74 = v73 | *(_DWORD *)(v70 + 4) & 0xF8000000;
                v75 = *(_QWORD *)(v70 - 8) + 32LL * (unsigned int)(v73 - 1);
                *(_DWORD *)(v70 + 4) = v74;
                if ( *(_QWORD *)v75 )
                {
                  v76 = *(_QWORD *)(v75 + 8);
                  **(_QWORD **)(v75 + 16) = v76;
                  if ( v76 )
                    *(_QWORD *)(v76 + 16) = *(_QWORD *)(v75 + 16);
                }
                *(_QWORD *)v75 = v71;
                if ( v71 )
                {
                  v77 = *(_QWORD *)(v71 + 16);
                  *(_QWORD *)(v75 + 8) = v77;
                  if ( v77 )
                    *(_QWORD *)(v77 + 16) = v75 + 8;
                  *(_QWORD *)(v75 + 16) = v71 + 16;
                  *(_QWORD *)(v71 + 16) = v75;
                }
                v61 += 4;
                v63 += 8LL;
                *(_QWORD *)(*(_QWORD *)(v70 - 8)
                          + 32LL * *(unsigned int *)(v70 + 72)
                          + 8LL * ((*(_DWORD *)(v70 + 4) & 0x7FFFFFFu) - 1)) = v59;
              }
              while ( v61 != v69 && v65 != (__int64 *)v63 );
              v60 = v68;
              v58 = v67;
              v16 = v66;
            }
            if ( v153 )
            {
              v78 = sub_B10CD0(v60 + 24);
              v153 = (__int64)sub_B026B0(v153, v78);
            }
            else
            {
              v153 = sub_B10CD0(v60 + 24);
            }
            v141 = v16;
            sub_B43C20((__int64)v16, v59);
            v145 = v160.m128i_i64[0];
            v147 = v160.m128i_u16[4];
            v79 = sub_BD2C40(72, 1u);
            v80 = v79;
            if ( v79 )
              sub_B4C8F0((__int64)v79, v148, 1u, v145, v147);
            v81 = *(_QWORD *)(v60 + 24);
            v82 = (const char **)(v80 + 6);
            v160.m128i_i64[0] = v81;
            if ( v81 )
            {
              sub_B96E90((__int64)v16, v81, 1);
              if ( v82 != v16 )
              {
                v105 = v80[6];
                if ( v105 )
LABEL_149:
                  sub_B91220((__int64)(v80 + 6), v105);
                v106 = (unsigned __int8 *)v160.m128i_i64[0];
                v80[6] = v160.m128i_i64[0];
                if ( v106 )
                  sub_B976B0((__int64)v16, v106, (__int64)(v80 + 6));
                goto LABEL_103;
              }
              if ( v160.m128i_i64[0] )
                sub_B91220((__int64)v16, v160.m128i_i64[0]);
            }
            else if ( v82 != v16 )
            {
              v105 = v80[6];
              if ( v105 )
                goto LABEL_149;
            }
LABEL_103:
            sub_B43D60(v149);
            if ( v143 )
            {
              v83 = (__m128i *)v143[1];
              v160.m128i_i64[0] = v59;
              v160.m128i_i64[1] = v148 & 0xFFFFFFFFFFFFFFFBLL;
              if ( v83 == v143[2] )
              {
                sub_F38BA0(v143, v83, (const __m128i *)v16);
              }
              else
              {
                if ( v83 )
                {
                  *v83 = _mm_loadu_si128(&v160);
                  v83 = (__m128i *)v143[1];
                }
                v143[1] = v83 + 1;
              }
            }
            if ( v140 == ++v58 )
              goto LABEL_109;
          }
        }
        v153 = 0;
        v141 = v16;
LABEL_109:
        sub_B10CB0(v141, v153);
        if ( v137 + 6 == v141 )
        {
          if ( v160.m128i_i64[0] )
            sub_B91220((__int64)v141, v160.m128i_i64[0]);
        }
        else
        {
          v84 = v137[6];
          if ( v84 )
            sub_B91220((__int64)(v137 + 6), v84);
          v85 = (unsigned __int8 *)v160.m128i_i64[0];
          v137[6] = v160.m128i_i64[0];
          if ( v85 )
            sub_B976B0((__int64)v141, v85, (__int64)(v137 + 6));
        }
        v15 = 1;
        if ( v157 != (__int64 *)v159 )
          _libc_free((unsigned __int64)v157);
      }
      v138 += 40;
      if ( v138 == v130 )
      {
        v125 |= v15;
        v86 = (unsigned __int64 *)v154;
        if ( v132 )
          goto LABEL_118;
LABEL_119:
        if ( v86 )
          j_j___libc_free_0((unsigned __int64)v86);
        goto LABEL_121;
      }
    }
  }
LABEL_211:
  v86 = 0;
  if ( v132 )
  {
LABEL_118:
    sub_FFB3D0(v132, v86, (v155 - (__int8 *)v86) >> 4, v6, v7, v8);
    v86 = (unsigned __int64 *)v154;
    goto LABEL_119;
  }
LABEL_121:
  v87 = v167;
  v88 = (unsigned __int64)&v167[40 * (unsigned int)v168];
  if ( v167 != (_BYTE *)v88 )
  {
    do
    {
      v88 -= 40LL;
      v89 = *(_QWORD *)(v88 + 8);
      if ( v89 != v88 + 24 )
        _libc_free(v89);
    }
    while ( v87 != (_BYTE *)v88 );
    v88 = (unsigned __int64)v167;
  }
  if ( (_BYTE *)v88 != v169 )
    _libc_free(v88);
  if ( (v164 & 1) == 0 )
    sub_C7D6A0(v165, 8LL * v166, 4);
  v90 = (__int64)a2;
  v95 = v125;
  LOBYTE(v95) = sub_29751F0(a1, a2, v132, a4, a5) | v125;
  if ( (_BYTE)v95 )
  {
    v90 = v132;
    v119 = sub_F62E00(a1, v132, 0, v92, v93, v94);
    while ( v119 )
    {
      do
      {
        v90 = v132;
        v120 = sub_29751F0(a1, a2, v132, a4, a5);
        v119 = sub_F62E00(a1, v132, 0, v121, v122, v123);
      }
      while ( v120 );
    }
  }
  sub_FFCE90((__int64)v170, v90, v91, v92, v93, v94);
  sub_FFD870((__int64)v170, v90, v96, v97, v98, v99);
  sub_FFBC40((__int64)v170, v90);
  v100 = v185;
  v101 = v184;
  if ( v185 != v184 )
  {
    do
    {
      v102 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v101[7];
      *v101 = &unk_49E5048;
      if ( v102 )
        v102(v101 + 5, v101 + 5, 3);
      *v101 = &unk_49DB368;
      v103 = v101[3];
      if ( v103 != -4096 && v103 != 0 && v103 != -8192 )
        sub_BD60C0(v101 + 1);
      v101 += 9;
    }
    while ( v100 != v101 );
    v101 = v184;
  }
  if ( v101 )
    j_j___libc_free_0((unsigned __int64)v101);
  if ( !v181 )
    _libc_free((unsigned __int64)v178);
  if ( (_BYTE *)v170[0] != v171 )
    _libc_free(v170[0]);
  return v95;
}
