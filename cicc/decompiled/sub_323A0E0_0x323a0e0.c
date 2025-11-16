// Function: sub_323A0E0
// Address: 0x323a0e0
//
__int64 __fastcall sub_323A0E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rcx
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 result; // rax
  __m128i *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  unsigned __int8 v18; // dl
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int8 v21; // dl
  _BYTE **v22; // r13
  __int64 v23; // rax
  _BYTE *v24; // r12
  _BYTE *v25; // rbx
  __int64 v26; // rdx
  int v27; // r11d
  _QWORD *v28; // r10
  unsigned int v29; // edi
  _QWORD *v30; // rax
  _BYTE *v31; // rcx
  unsigned int v32; // edi
  unsigned int v33; // esi
  __int64 v34; // r9
  int v35; // r11d
  __int64 j; // rdx
  __int64 v37; // r8
  __int64 v38; // rcx
  __int64 v39; // rax
  _BYTE *v40; // rdi
  __int64 v41; // rbx
  char v42; // dl
  int v43; // edx
  __int64 v44; // rcx
  _QWORD *v45; // rax
  __int64 *v46; // rdi
  __int64 v47; // rsi
  _QWORD *v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // r12
  __int64 v52; // rsi
  int v53; // r14d
  _QWORD *v54; // rbx
  unsigned int v55; // eax
  _QWORD *v56; // r12
  unsigned __int64 v57; // rdi
  int v58; // r14d
  _QWORD *v59; // rbx
  unsigned int v60; // eax
  __int64 v61; // r13
  _QWORD *v62; // r12
  unsigned __int64 v63; // rdi
  unsigned int v64; // eax
  __int64 v65; // rdx
  int v66; // r13d
  __int64 v67; // rbx
  unsigned int v68; // eax
  __int64 v69; // r14
  __int64 v70; // r12
  unsigned __int64 v71; // rdi
  int v72; // ecx
  int v73; // ecx
  _QWORD *v74; // rax
  __int64 v75; // rax
  int v76; // edi
  int v77; // r9d
  _QWORD *v78; // r8
  unsigned int i; // ecx
  _BYTE *v80; // r10
  unsigned int v81; // ecx
  int v82; // r10d
  int v83; // r10d
  __int64 v84; // rdx
  __int64 v85; // rsi
  int v86; // r11d
  __int64 v87; // rdi
  int v88; // r10d
  int v89; // r10d
  int v90; // r11d
  __int64 v91; // rdx
  __int64 v92; // rsi
  __int64 v93; // rax
  __int64 v94; // rbx
  __int64 k; // r12
  unsigned int v96; // esi
  __int64 v97; // rbx
  __int64 v98; // r9
  __int64 *v99; // rax
  __int64 v100; // r8
  int v101; // edi
  __int64 *v102; // rcx
  int v103; // edx
  int v104; // r8d
  __int64 *v105; // rdi
  unsigned int v106; // r13d
  int v107; // ecx
  __int64 v108; // rsi
  unsigned __int64 v109; // rdi
  unsigned __int64 v110; // rdi
  __int64 v111; // rsi
  __int64 v112; // rdi
  int v113; // edx
  __int64 v114; // rbx
  unsigned int v115; // eax
  _QWORD *v116; // rdi
  unsigned __int64 v117; // rdx
  unsigned __int64 v118; // rax
  _QWORD *v119; // rax
  __int64 v120; // rcx
  _QWORD *n; // rdx
  _QWORD *v122; // rdi
  __int64 v123; // rdx
  int v124; // ebx
  unsigned int v125; // eax
  unsigned __int64 v126; // rax
  __int64 v127; // rax
  _QWORD *v128; // rax
  __int64 v129; // rdx
  _QWORD *ii; // rdx
  int v131; // edi
  unsigned int v132; // ebx
  _BYTE *v133; // r9
  unsigned int v134; // ebx
  __int64 v135; // rax
  _QWORD *v136; // r12
  _QWORD *v137; // r14
  __int64 v138; // r9
  __int64 v139; // rdx
  _QWORD *v140; // rax
  __int64 v141; // rbx
  unsigned int v142; // esi
  int v143; // r8d
  int v144; // r8d
  __int64 v145; // r10
  int v146; // edx
  __int64 v147; // rcx
  __int64 v148; // rbx
  _QWORD *v149; // r13
  unsigned __int64 v150; // rdi
  unsigned __int64 v151; // rbx
  unsigned __int64 v152; // rdi
  int v153; // edx
  __int64 v154; // rbx
  unsigned int v155; // eax
  _QWORD *v156; // rdi
  unsigned __int64 v157; // rdx
  unsigned __int64 v158; // rax
  _QWORD *v159; // rax
  __int64 v160; // rcx
  _QWORD *m; // rdx
  int v162; // r8d
  unsigned int v163; // ecx
  __int64 v164; // r11
  __int64 v165; // rax
  _QWORD *v166; // rax
  unsigned int v167; // eax
  __int64 v168; // rdx
  _QWORD *v169; // rax
  _QWORD *v170; // rax
  int v171; // edi
  _QWORD *v172; // rcx
  int v173; // ecx
  int v174; // r8d
  int v175; // r8d
  __int64 v176; // r10
  int v177; // edi
  _QWORD *v178; // rsi
  __int64 v179; // rcx
  __int64 v180; // rbx
  int v181; // edi
  __int64 *v182; // rsi
  int v183; // edi
  __int64 v185; // [rsp+8h] [rbp-B8h]
  __int64 v186; // [rsp+18h] [rbp-A8h]
  __int64 v187; // [rsp+20h] [rbp-A0h]
  __int64 *v188; // [rsp+28h] [rbp-98h]
  __int64 v189; // [rsp+30h] [rbp-90h]
  __int64 v190; // [rsp+38h] [rbp-88h]
  __int64 *v191; // [rsp+40h] [rbp-80h]
  __int64 v192; // [rsp+48h] [rbp-78h]
  __int64 v193; // [rsp+48h] [rbp-78h]
  unsigned int v194; // [rsp+48h] [rbp-78h]
  __int64 v195; // [rsp+48h] [rbp-78h]
  __int64 v196; // [rsp+50h] [rbp-70h]
  _BYTE **v197; // [rsp+58h] [rbp-68h]
  __int64 v198; // [rsp+58h] [rbp-68h]
  __m128i v199; // [rsp+60h] [rbp-60h] BYREF
  __int64 v200; // [rsp+70h] [rbp-50h] BYREF
  __int64 v201; // [rsp+78h] [rbp-48h]
  __int64 v202; // [rsp+80h] [rbp-40h]
  __int64 v203; // [rsp+88h] [rbp-38h]

  v3 = sub_B92180(*a2);
  v187 = v3;
  *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL) + 1912LL) = 0;
  v185 = *(_QWORD *)(a1 + 296);
  v4 = *(_BYTE *)(v3 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(v3 - 32);
  else
    v5 = v3 - 16 - 8LL * ((v4 >> 2) & 0xF);
  v196 = sub_3238860(a1, *(_QWORD *)(v5 + 40));
  result = *(_QWORD *)(v196 + 80);
  if ( *(_DWORD *)(result + 32) != 3 )
  {
    v7 = (__m128i *)v196;
    v200 = 0;
    v201 = 0;
    v202 = 0;
    v203 = 0;
    sub_3234490(a1, v196, v187, (__int64)&v200);
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *(_QWORD *)(v10 + 336);
    v12 = v11 + 24LL * *(unsigned int *)(v10 + 344);
    while ( v11 != v12 )
    {
      v7 = *(__m128i **)(v11 + 8);
      v13 = *(_QWORD *)(v11 + 16);
      v11 += 24;
      sub_3735B10(v196, v7, v13);
    }
    v14 = *(_QWORD *)(v196 + 80);
    v15 = *(unsigned int *)(a1 + 256);
    if ( *(_BYTE *)(v14 + 42) || *(_DWORD *)(v14 + 32) != 2 || (_DWORD)v15 )
    {
      v16 = *(__int64 **)(a1 + 248);
      v188 = &v16[v15];
      if ( v188 == v16 )
        goto LABEL_32;
      v191 = *(__int64 **)(a1 + 248);
      v186 = a1 + 72;
      while ( 1 )
      {
        v190 = *v191;
        v17 = *(_QWORD *)(*v191 + 8);
        v18 = *(_BYTE *)(v17 - 16);
        if ( (v18 & 2) != 0 )
          v19 = *(_QWORD *)(v17 - 32);
        else
          v19 = v17 - 16 - 8LL * ((v18 >> 2) & 0xF);
        v20 = *(_QWORD *)(v19 + 56);
        if ( !v20 )
          goto LABEL_31;
        v21 = *(_BYTE *)(v20 - 16);
        if ( (v21 & 2) != 0 )
        {
          v22 = *(_BYTE ***)(v20 - 32);
          v23 = *(unsigned int *)(v20 - 24);
        }
        else
        {
          v22 = (_BYTE **)(v20 - 16 - 8LL * ((v21 >> 2) & 0xF));
          v23 = (*(_WORD *)(v20 - 16) >> 6) & 0xF;
        }
        v197 = &v22[v23];
        if ( v22 == v197 )
          goto LABEL_31;
        v189 = a1 + 2968;
        do
        {
          v24 = *v22;
          v25 = sub_321C590(*v22);
          v26 = sub_3504CA0(v186, v25);
          if ( (unsigned __int8)(*v24 - 26) <= 1u )
          {
            if ( !(_DWORD)v203 )
            {
              ++v200;
              goto LABEL_125;
            }
            v27 = 1;
            v28 = 0;
            v29 = (969526130 * (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4))) & (v203 - 1);
            while ( 2 )
            {
              v30 = (_QWORD *)(v201 + 16LL * v29);
              v31 = (_BYTE *)*v30;
              if ( v24 == (_BYTE *)*v30 )
              {
                if ( !v30[1] )
                  goto LABEL_30;
                if ( v31 == (_BYTE *)-4096LL )
                  goto LABEL_113;
LABEL_21:
                if ( v31 == (_BYTE *)-8192LL && v30[1] == -8192 && !v28 )
                  v28 = (_QWORD *)(v201 + 16LL * v29);
              }
              else
              {
                if ( v31 != (_BYTE *)-4096LL )
                  goto LABEL_21;
LABEL_113:
                if ( v30[1] == -4096 )
                {
                  if ( v28 )
                    v30 = v28;
                  ++v200;
                  v76 = v202 + 1;
                  if ( 4 * ((int)v202 + 1) < (unsigned int)(3 * v203) )
                  {
                    if ( (int)v203 - HIDWORD(v202) - v76 > (unsigned int)v203 >> 3 )
                    {
LABEL_118:
                      LODWORD(v202) = v76;
                      if ( *v30 != -4096 || v30[1] != -4096 )
                        --HIDWORD(v202);
                      *v30 = v24;
                      v30[1] = 0;
                      v192 = v26;
                      if ( !sub_37362B0(v196, v24) )
                        sub_373BC10(v196, v24, v192);
                      goto LABEL_30;
                    }
                    v195 = v26;
                    sub_3201030((__int64)&v200, v203);
                    if ( (_DWORD)v203 )
                    {
                      v26 = v195;
                      v131 = 1;
                      v132 = (v203 - 1) & (969526130 * (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)));
                      v78 = 0;
                      while ( 1 )
                      {
                        v30 = (_QWORD *)(v201 + 16LL * v132);
                        v133 = (_BYTE *)*v30;
                        if ( v24 == (_BYTE *)*v30 && !v30[1] )
                          break;
                        if ( v133 == (_BYTE *)-4096LL )
                        {
                          if ( v30[1] == -4096 )
                            goto LABEL_290;
                        }
                        else if ( v133 == (_BYTE *)-8192LL && v30[1] == -8192 && !v78 )
                        {
                          v78 = (_QWORD *)(v201 + 16LL * v132);
                        }
                        v134 = v131 + v132;
                        ++v131;
                        v132 = (v203 - 1) & v134;
                      }
LABEL_194:
                      v76 = v202 + 1;
                      goto LABEL_118;
                    }
                    goto LABEL_375;
                  }
LABEL_125:
                  v193 = v26;
                  sub_3201030((__int64)&v200, 2 * v203);
                  if ( (_DWORD)v203 )
                  {
                    v26 = v193;
                    v77 = 1;
                    v78 = 0;
                    for ( i = (v203 - 1) & (969526130 * (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)));
                          ;
                          i = (v203 - 1) & v81 )
                    {
                      v30 = (_QWORD *)(v201 + 16LL * i);
                      v80 = (_BYTE *)*v30;
                      if ( v24 == (_BYTE *)*v30 && !v30[1] )
                        break;
                      if ( v80 == (_BYTE *)-4096LL )
                      {
                        if ( v30[1] == -4096 )
                        {
LABEL_290:
                          if ( v78 )
                            v30 = v78;
                          v76 = v202 + 1;
                          goto LABEL_118;
                        }
                      }
                      else if ( v80 == (_BYTE *)-8192LL && v30[1] == -8192 && !v78 )
                      {
                        v78 = (_QWORD *)(v201 + 16LL * i);
                      }
                      v81 = v77 + i;
                      ++v77;
                    }
                    goto LABEL_194;
                  }
LABEL_375:
                  LODWORD(v202) = v202 + 1;
                  BUG();
                }
              }
              v32 = v27 + v29;
              ++v27;
              v29 = (v203 - 1) & v32;
              continue;
            }
          }
          v33 = *(_DWORD *)(a1 + 2992);
          if ( !v33 )
          {
            ++*(_QWORD *)(a1 + 2968);
            goto LABEL_135;
          }
          v34 = *(_QWORD *)(a1 + 2976);
          v35 = 1;
          j = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
          v37 = (v33 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v38 = v34 + 88 * v37;
          v39 = 0;
          v40 = *(_BYTE **)v38;
          if ( v25 == *(_BYTE **)v38 )
            goto LABEL_28;
          while ( 1 )
          {
            if ( v40 == (_BYTE *)-4096LL )
            {
              if ( !v39 )
                v39 = v38;
              v72 = *(_DWORD *)(a1 + 2984);
              ++*(_QWORD *)(a1 + 2968);
              v73 = v72 + 1;
              if ( 4 * v73 < 3 * v33 )
              {
                v37 = v33 >> 3;
                if ( v33 - *(_DWORD *)(a1 + 2988) - v73 > (unsigned int)v37 )
                {
LABEL_100:
                  *(_DWORD *)(a1 + 2984) = v73;
                  if ( *(_QWORD *)v39 != -4096 )
                    --*(_DWORD *)(a1 + 2988);
                  *(_QWORD *)v39 = v25;
                  v41 = v39 + 8;
                  *(_OWORD *)(v39 + 8) = 0;
                  *(_QWORD *)(v39 + 16) = v39 + 40;
                  *(_OWORD *)(v39 + 24) = 0;
                  *(_QWORD *)(v39 + 56) = v39 + 72;
                  *(_QWORD *)(v39 + 24) = 2;
                  *(_DWORD *)(v39 + 32) = 0;
                  *(_BYTE *)(v39 + 36) = 1;
                  *(_QWORD *)(v39 + 64) = 0x200000000LL;
                  *(_OWORD *)(v39 + 40) = 0;
                  *(_OWORD *)(v39 + 72) = 0;
                  goto LABEL_103;
                }
                v194 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
                sub_3227C70(v189, v33);
                v88 = *(_DWORD *)(a1 + 2992);
                if ( v88 )
                {
                  v89 = v88 - 1;
                  v37 = *(_QWORD *)(a1 + 2976);
                  v87 = 0;
                  v90 = 1;
                  LODWORD(v91) = v89 & v194;
                  v39 = v37 + 88LL * (v89 & v194);
                  v92 = *(_QWORD *)v39;
                  v73 = *(_DWORD *)(a1 + 2984) + 1;
                  if ( v25 == *(_BYTE **)v39 )
                    goto LABEL_100;
                  while ( v92 != -4096 )
                  {
                    if ( v92 == -8192 && !v87 )
                      v87 = v39;
                    v34 = (unsigned int)(v90 + 1);
                    v91 = v89 & (unsigned int)(v91 + v90);
                    v39 = v37 + 88 * v91;
                    v92 = *(_QWORD *)v39;
                    if ( v25 == *(_BYTE **)v39 )
                      goto LABEL_100;
                    ++v90;
                  }
                  goto LABEL_139;
                }
                goto LABEL_376;
              }
LABEL_135:
              sub_3227C70(v189, 2 * v33);
              v82 = *(_DWORD *)(a1 + 2992);
              if ( v82 )
              {
                v83 = v82 - 1;
                v37 = *(_QWORD *)(a1 + 2976);
                LODWORD(v84) = v83 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
                v39 = v37 + 88LL * (unsigned int)v84;
                v85 = *(_QWORD *)v39;
                v73 = *(_DWORD *)(a1 + 2984) + 1;
                if ( v25 == *(_BYTE **)v39 )
                  goto LABEL_100;
                v86 = 1;
                v87 = 0;
                while ( v85 != -4096 )
                {
                  if ( v85 == -8192 && !v87 )
                    v87 = v39;
                  v34 = (unsigned int)(v86 + 1);
                  v84 = v83 & (unsigned int)(v84 + v86);
                  v39 = v37 + 88 * v84;
                  v85 = *(_QWORD *)v39;
                  if ( v25 == *(_BYTE **)v39 )
                    goto LABEL_100;
                  ++v86;
                }
LABEL_139:
                if ( v87 )
                  v39 = v87;
                goto LABEL_100;
              }
LABEL_376:
              ++*(_DWORD *)(a1 + 2984);
              BUG();
            }
            if ( v39 || v40 != (_BYTE *)-8192LL )
              v38 = v39;
            v37 = (v33 - 1) & (v35 + (_DWORD)v37);
            v40 = *(_BYTE **)(v34 + 88LL * (unsigned int)v37);
            if ( v25 == v40 )
              break;
            ++v35;
            v39 = v38;
            v38 = v34 + 88LL * (unsigned int)v37;
          }
          v38 = v34 + 88LL * (unsigned int)v37;
LABEL_28:
          v41 = v38 + 8;
          if ( !*(_BYTE *)(v38 + 36) )
          {
LABEL_29:
            sub_C8CC70(v41, (__int64)v24, j, v38, v37, v34);
            if ( !v42 )
              goto LABEL_30;
            goto LABEL_108;
          }
LABEL_103:
          v74 = *(_QWORD **)(v41 + 8);
          v38 = *(unsigned int *)(v41 + 20);
          for ( j = (__int64)&v74[v38]; (_QWORD *)j != v74; ++v74 )
          {
            if ( v24 == (_BYTE *)*v74 )
              goto LABEL_30;
          }
          if ( (unsigned int)v38 >= *(_DWORD *)(v41 + 16) )
            goto LABEL_29;
          *(_DWORD *)(v41 + 20) = v38 + 1;
          *(_QWORD *)j = v24;
          ++*(_QWORD *)v41;
LABEL_108:
          v75 = *(unsigned int *)(v41 + 56);
          if ( v75 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 60) )
          {
            sub_C8D5F0(v41 + 48, (const void *)(v41 + 64), v75 + 1, 8u, v37, v34);
            v75 = *(unsigned int *)(v41 + 56);
          }
          *(_QWORD *)(*(_QWORD *)(v41 + 48) + 8 * v75) = v24;
          ++*(_DWORD *)(v41 + 56);
LABEL_30:
          ++v22;
        }
        while ( v197 != v22 );
LABEL_31:
        sub_3238FB0(a1, v196, v190);
        if ( v188 == ++v191 )
          goto LABEL_32;
      }
    }
    if ( !*(_BYTE *)(a1 + 4801) )
    {
      v93 = *(_QWORD *)(a1 + 8);
      v94 = *(_QWORD *)(v93 + 336);
      for ( k = v94 + 24LL * *(unsigned int *)(v93 + 344); k != v94; v94 += 24 )
      {
        v7 = *(__m128i **)(a1 + 712);
        v199.m128i_i64[0] = *(_QWORD *)(v94 + 8);
        v199.m128i_i64[1] = v196;
        if ( v7 == *(__m128i **)(a1 + 720) )
        {
          sub_3223B70((unsigned __int64 *)(a1 + 704), v7, &v199);
        }
        else
        {
          if ( v7 )
          {
            *v7 = _mm_loadu_si128(&v199);
            v7 = *(__m128i **)(a1 + 712);
          }
          *(_QWORD *)(a1 + 712) = ++v7;
        }
      }
      ++*(_QWORD *)(a1 + 6264);
      if ( !*(_BYTE *)(a1 + 6292) )
      {
        v167 = 4 * (*(_DWORD *)(a1 + 6284) - *(_DWORD *)(a1 + 6288));
        v168 = *(unsigned int *)(a1 + 6280);
        if ( v167 < 0x20 )
          v167 = 32;
        if ( v167 < (unsigned int)v168 )
        {
          sub_C8C990(a1 + 6264, (__int64)v7);
          goto LABEL_191;
        }
        memset(*(void **)(a1 + 6272), -1, 8 * v168);
      }
      *(_QWORD *)(a1 + 6284) = 0;
      goto LABEL_191;
    }
LABEL_32:
    v43 = *(_DWORD *)(a1 + 2808);
    if ( !v43 )
    {
      v44 = *(unsigned int *)(a1 + 2832);
      v45 = *(_QWORD **)(a1 + 2824);
      v46 = &v45[v44];
      v47 = (8 * v44) >> 3;
      if ( !((8 * v44) >> 5) )
        goto LABEL_227;
      v48 = &v45[4 * ((8 * v44) >> 5)];
      do
      {
        if ( v187 == *v45 )
          goto LABEL_40;
        if ( v187 == v45[1] )
        {
          ++v45;
          goto LABEL_40;
        }
        if ( v187 == v45[2] )
        {
          v45 += 2;
          goto LABEL_40;
        }
        if ( v187 == v45[3] )
        {
          v45 += 3;
          goto LABEL_40;
        }
        v45 += 4;
      }
      while ( v45 != v48 );
      v47 = v46 - v45;
LABEL_227:
      if ( v47 != 2 )
      {
        if ( v47 != 3 )
        {
          if ( v47 != 1 )
            goto LABEL_230;
LABEL_256:
          if ( v187 == *v45 )
          {
LABEL_40:
            if ( v46 == v45 )
              goto LABEL_230;
LABEL_41:
            v49 = sub_373FC00(v196, v187, v185, *(_QWORD *)(a1 + 3672));
            v50 = *(_QWORD *)(v196 + 408);
            v51 = v49;
            if ( v50 && *(_DWORD *)(a1 + 256) && *(_BYTE *)(*(_QWORD *)(v196 + 80) + 41LL) )
              sub_373FC00(v50, v187, v185, *(_QWORD *)(a1 + 3672));
            *(_QWORD *)(a1 + 3672) = 0;
            v52 = v187;
            sub_3231960(a1, v187, v196, v51, (__int64)a2);
            v53 = *(_DWORD *)(a1 + 3432);
            ++*(_QWORD *)(a1 + 3416);
            if ( v53 || *(_DWORD *)(a1 + 3436) )
            {
              v54 = *(_QWORD **)(a1 + 3424);
              v55 = 4 * v53;
              if ( (unsigned int)(4 * v53) < 0x40 )
                v55 = 64;
              v56 = &v54[17 * *(unsigned int *)(a1 + 3440)];
              if ( v55 >= *(_DWORD *)(a1 + 3440) )
              {
                while ( v56 != v54 )
                {
                  if ( *v54 != -4096 )
                  {
                    if ( *v54 != -8192 )
                    {
                      v57 = v54[7];
                      if ( (_QWORD *)v57 != v54 + 9 )
                        _libc_free(v57);
                      sub_321A090(v54[3]);
                    }
                    *v54 = -4096;
                  }
                  v54 += 17;
                }
              }
              else
              {
                v198 = 136LL * *(unsigned int *)(a1 + 3440);
                v149 = *(_QWORD **)(a1 + 3424);
                do
                {
                  if ( *v149 != -8192 && *v149 != -4096 )
                  {
                    v150 = v149[7];
                    if ( (_QWORD *)v150 != v149 + 9 )
                      _libc_free(v150);
                    v151 = v149[3];
                    while ( v151 )
                    {
                      sub_321A090(*(_QWORD *)(v151 + 24));
                      v152 = v151;
                      v151 = *(_QWORD *)(v151 + 16);
                      v52 = 48;
                      j_j___libc_free_0(v152);
                    }
                  }
                  v149 += 17;
                }
                while ( v56 != v149 );
                v153 = *(_DWORD *)(a1 + 3440);
                if ( v53 )
                {
                  v154 = 64;
                  if ( v53 != 1 )
                  {
                    _BitScanReverse(&v155, v53 - 1);
                    v154 = (unsigned int)(1 << (33 - (v155 ^ 0x1F)));
                    if ( (int)v154 < 64 )
                      v154 = 64;
                  }
                  v156 = *(_QWORD **)(a1 + 3424);
                  if ( (_DWORD)v154 == v153 )
                  {
                    *(_QWORD *)(a1 + 3432) = 0;
                    v166 = &v156[17 * v154];
                    do
                    {
                      if ( v156 )
                        *v156 = -4096;
                      v156 += 17;
                    }
                    while ( v166 != v156 );
                  }
                  else
                  {
                    sub_C7D6A0((__int64)v156, v198, 8);
                    v52 = 8;
                    v157 = ((((((((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                              | (4 * (int)v154 / 3u + 1)
                              | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 4)
                            | (((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                            | (4 * (int)v154 / 3u + 1)
                            | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 8)
                          | (((((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                            | (4 * (int)v154 / 3u + 1)
                            | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 4)
                          | (((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                          | (4 * (int)v154 / 3u + 1)
                          | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 16;
                    v158 = (v157
                          | (((((((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                              | (4 * (int)v154 / 3u + 1)
                              | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 4)
                            | (((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                            | (4 * (int)v154 / 3u + 1)
                            | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 8)
                          | (((((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                            | (4 * (int)v154 / 3u + 1)
                            | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 4)
                          | (((4 * (int)v154 / 3u + 1) | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1)) >> 2)
                          | (4 * (int)v154 / 3u + 1)
                          | ((unsigned __int64)(4 * (int)v154 / 3u + 1) >> 1))
                         + 1;
                    *(_DWORD *)(a1 + 3440) = v158;
                    v159 = (_QWORD *)sub_C7D670(136 * v158, 8);
                    v160 = *(unsigned int *)(a1 + 3440);
                    *(_QWORD *)(a1 + 3432) = 0;
                    *(_QWORD *)(a1 + 3424) = v159;
                    for ( m = &v159[17 * v160]; m != v159; v159 += 17 )
                    {
                      if ( v159 )
                        *v159 = -4096;
                    }
                  }
                  goto LABEL_60;
                }
                if ( v153 )
                {
                  v52 = v198;
                  sub_C7D6A0(*(_QWORD *)(a1 + 3424), v198, 8);
                  *(_QWORD *)(a1 + 3424) = 0;
                  *(_QWORD *)(a1 + 3432) = 0;
                  *(_DWORD *)(a1 + 3440) = 0;
                  goto LABEL_60;
                }
              }
              *(_QWORD *)(a1 + 3432) = 0;
            }
LABEL_60:
            v58 = *(_DWORD *)(a1 + 3464);
            ++*(_QWORD *)(a1 + 3448);
            if ( v58 || *(_DWORD *)(a1 + 3468) )
            {
              v59 = *(_QWORD **)(a1 + 3456);
              v60 = 4 * v58;
              v61 = 56LL * *(unsigned int *)(a1 + 3472);
              if ( (unsigned int)(4 * v58) < 0x40 )
                v60 = 64;
              v62 = &v59[(unsigned __int64)v61 / 8];
              if ( v60 >= *(_DWORD *)(a1 + 3472) )
              {
                for ( ; v59 != v62; v59 += 7 )
                {
                  if ( *v59 != -4096 )
                  {
                    if ( *v59 != -8192 )
                    {
                      v63 = v59[1];
                      if ( (_QWORD *)v63 != v59 + 3 )
                        _libc_free(v63);
                    }
                    *v59 = -4096;
                  }
                }
                goto LABEL_72;
              }
              do
              {
                if ( *v59 != -8192 && *v59 != -4096 )
                {
                  v110 = v59[1];
                  if ( (_QWORD *)v110 != v59 + 3 )
                    _libc_free(v110);
                }
                v59 += 7;
              }
              while ( v59 != v62 );
              v113 = *(_DWORD *)(a1 + 3472);
              if ( v58 )
              {
                v114 = 64;
                if ( v58 != 1 )
                {
                  _BitScanReverse(&v115, v58 - 1);
                  v114 = (unsigned int)(1 << (33 - (v115 ^ 0x1F)));
                  if ( (int)v114 < 64 )
                    v114 = 64;
                }
                v116 = *(_QWORD **)(a1 + 3456);
                if ( (_DWORD)v114 == v113 )
                {
                  *(_QWORD *)(a1 + 3464) = 0;
                  v170 = &v116[7 * v114];
                  do
                  {
                    if ( v116 )
                      *v116 = -4096;
                    v116 += 7;
                  }
                  while ( v170 != v116 );
                }
                else
                {
                  sub_C7D6A0((__int64)v116, v61, 8);
                  v52 = 8;
                  v117 = ((((((((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                            | (4 * (int)v114 / 3u + 1)
                            | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 4)
                          | (((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                          | (4 * (int)v114 / 3u + 1)
                          | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 8)
                        | (((((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                          | (4 * (int)v114 / 3u + 1)
                          | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 4)
                        | (((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                        | (4 * (int)v114 / 3u + 1)
                        | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 16;
                  v118 = (v117
                        | (((((((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                            | (4 * (int)v114 / 3u + 1)
                            | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 4)
                          | (((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                          | (4 * (int)v114 / 3u + 1)
                          | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 8)
                        | (((((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                          | (4 * (int)v114 / 3u + 1)
                          | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 4)
                        | (((4 * (int)v114 / 3u + 1) | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1)) >> 2)
                        | (4 * (int)v114 / 3u + 1)
                        | ((unsigned __int64)(4 * (int)v114 / 3u + 1) >> 1))
                       + 1;
                  *(_DWORD *)(a1 + 3472) = v118;
                  v119 = (_QWORD *)sub_C7D670(56 * v118, 8);
                  v120 = *(unsigned int *)(a1 + 3472);
                  *(_QWORD *)(a1 + 3464) = 0;
                  *(_QWORD *)(a1 + 3456) = v119;
                  for ( n = &v119[7 * v120]; n != v119; v119 += 7 )
                  {
                    if ( v119 )
                      *v119 = -4096;
                  }
                }
              }
              else
              {
                if ( !v113 )
                {
LABEL_72:
                  *(_QWORD *)(a1 + 3464) = 0;
                  goto LABEL_73;
                }
                v52 = v61;
                sub_C7D6A0(*(_QWORD *)(a1 + 3456), v61, 8);
                *(_QWORD *)(a1 + 3456) = 0;
                *(_QWORD *)(a1 + 3464) = 0;
                *(_DWORD *)(a1 + 3472) = 0;
              }
            }
LABEL_73:
            ++*(_QWORD *)(a1 + 3696);
            if ( !*(_BYTE *)(a1 + 3724) )
            {
              v64 = 4 * (*(_DWORD *)(a1 + 3716) - *(_DWORD *)(a1 + 3720));
              v65 = *(unsigned int *)(a1 + 3712);
              if ( v64 < 0x20 )
                v64 = 32;
              if ( v64 < (unsigned int)v65 )
              {
                sub_C8C990(a1 + 3696, v52);
LABEL_79:
                v66 = *(_DWORD *)(a1 + 2984);
                ++*(_QWORD *)(a1 + 2968);
                if ( !v66 && !*(_DWORD *)(a1 + 2988) )
                  goto LABEL_191;
                v67 = *(_QWORD *)(a1 + 2976);
                v68 = 4 * v66;
                v69 = 88LL * *(unsigned int *)(a1 + 2992);
                if ( (unsigned int)(4 * v66) < 0x40 )
                  v68 = 64;
                v70 = v67 + v69;
                if ( v68 < *(_DWORD *)(a1 + 2992) )
                {
                  do
                  {
                    if ( *(_QWORD *)v67 != -4096 && *(_QWORD *)v67 != -8192 )
                    {
                      v109 = *(_QWORD *)(v67 + 56);
                      if ( v109 != v67 + 72 )
                        _libc_free(v109);
                      if ( !*(_BYTE *)(v67 + 36) )
                        _libc_free(*(_QWORD *)(v67 + 16));
                    }
                    v67 += 88;
                  }
                  while ( v70 != v67 );
                  v122 = *(_QWORD **)(a1 + 2976);
                  v123 = *(unsigned int *)(a1 + 2992);
                  if ( v66 )
                  {
                    v124 = 64;
                    if ( v66 != 1 )
                    {
                      _BitScanReverse(&v125, v66 - 1);
                      v124 = 1 << (33 - (v125 ^ 0x1F));
                      if ( v124 < 64 )
                        v124 = 64;
                    }
                    if ( (_DWORD)v123 == v124 )
                    {
                      *(_QWORD *)(a1 + 2984) = 0;
                      v169 = &v122[11 * v123];
                      do
                      {
                        if ( v122 )
                          *v122 = -4096;
                        v122 += 11;
                      }
                      while ( v169 != v122 );
                    }
                    else
                    {
                      sub_C7D6A0((__int64)v122, v69, 8);
                      v126 = (((((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                               | (4 * v124 / 3u + 1)
                               | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                             | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                             | (4 * v124 / 3u + 1)
                             | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 8)
                           | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                             | (4 * v124 / 3u + 1)
                             | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                           | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                           | (4 * v124 / 3u + 1)
                           | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1);
                      v127 = ((v126 >> 16) | v126) + 1;
                      *(_DWORD *)(a1 + 2992) = v127;
                      v128 = (_QWORD *)sub_C7D670(88 * v127, 8);
                      v129 = *(unsigned int *)(a1 + 2992);
                      *(_QWORD *)(a1 + 2984) = 0;
                      *(_QWORD *)(a1 + 2976) = v128;
                      for ( ii = &v128[11 * v129]; ii != v128; v128 += 11 )
                      {
                        if ( v128 )
                          *v128 = -4096;
                      }
                    }
                    goto LABEL_191;
                  }
                  if ( (_DWORD)v123 )
                  {
                    sub_C7D6A0((__int64)v122, v69, 8);
                    *(_QWORD *)(a1 + 2976) = 0;
                    *(_QWORD *)(a1 + 2984) = 0;
                    *(_DWORD *)(a1 + 2992) = 0;
                    goto LABEL_191;
                  }
                }
                else
                {
                  for ( ; v67 != v70; v67 += 88 )
                  {
                    if ( *(_QWORD *)v67 != -4096 )
                    {
                      if ( *(_QWORD *)v67 != -8192 )
                      {
                        v71 = *(_QWORD *)(v67 + 56);
                        if ( v71 != v67 + 72 )
                          _libc_free(v71);
                        if ( !*(_BYTE *)(v67 + 36) )
                          _libc_free(*(_QWORD *)(v67 + 16));
                      }
                      *(_QWORD *)v67 = -4096;
                    }
                  }
                }
                *(_QWORD *)(a1 + 2984) = 0;
LABEL_191:
                v111 = (unsigned int)v203;
                v112 = v201;
                *(_QWORD *)(a1 + 32) = 0;
                *(_QWORD *)(a1 + 3048) = 0;
                return sub_C7D6A0(v112, 16 * v111, 8);
              }
              memset(*(void **)(a1 + 3704), -1, 8 * v65);
            }
            *(_QWORD *)(a1 + 3716) = 0;
            goto LABEL_79;
          }
LABEL_230:
          if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 2836) )
          {
            sub_C8D5F0(a1 + 2824, (const void *)(a1 + 2840), v44 + 1, 8u, v8, v9);
            v46 = (__int64 *)(*(_QWORD *)(a1 + 2824) + 8LL * *(unsigned int *)(a1 + 2832));
          }
          *v46 = v187;
          v135 = (unsigned int)(*(_DWORD *)(a1 + 2832) + 1);
          *(_DWORD *)(a1 + 2832) = v135;
          if ( (unsigned int)v135 <= 0x10 )
            goto LABEL_41;
          v136 = *(_QWORD **)(a1 + 2824);
          v137 = &v136[v135];
          while ( 1 )
          {
            v142 = *(_DWORD *)(a1 + 2816);
            if ( !v142 )
              break;
            v138 = *(_QWORD *)(a1 + 2800);
            LODWORD(v139) = (v142 - 1) & (((unsigned int)*v136 >> 9) ^ ((unsigned int)*v136 >> 4));
            v140 = (_QWORD *)(v138 + 8LL * (unsigned int)v139);
            v141 = *v140;
            if ( *v136 != *v140 )
            {
              v171 = 1;
              v172 = 0;
              while ( v141 != -4096 )
              {
                if ( v141 == -8192 && !v172 )
                  v172 = v140;
                v139 = (v142 - 1) & ((_DWORD)v139 + v171);
                v140 = (_QWORD *)(v138 + 8 * v139);
                v141 = *v140;
                if ( *v136 == *v140 )
                  goto LABEL_235;
                ++v171;
              }
              if ( v172 )
                v140 = v172;
              v173 = *(_DWORD *)(a1 + 2808);
              ++*(_QWORD *)(a1 + 2792);
              v146 = v173 + 1;
              if ( 4 * (v173 + 1) < 3 * v142 )
              {
                if ( v142 - *(_DWORD *)(a1 + 2812) - v146 <= v142 >> 3 )
                {
                  sub_32026C0(a1 + 2792, v142);
                  v174 = *(_DWORD *)(a1 + 2816);
                  if ( !v174 )
                  {
LABEL_373:
                    ++*(_DWORD *)(a1 + 2808);
                    BUG();
                  }
                  v175 = v174 - 1;
                  v176 = *(_QWORD *)(a1 + 2800);
                  v177 = 1;
                  v146 = *(_DWORD *)(a1 + 2808) + 1;
                  v178 = 0;
                  LODWORD(v179) = v175 & (((unsigned int)*v136 >> 9) ^ ((unsigned int)*v136 >> 4));
                  v140 = (_QWORD *)(v176 + 8LL * (unsigned int)v179);
                  v180 = *v140;
                  if ( *v140 != *v136 )
                  {
                    while ( v180 != -4096 )
                    {
                      if ( !v178 && v180 == -8192 )
                        v178 = v140;
                      v179 = v175 & (unsigned int)(v179 + v177);
                      v140 = (_QWORD *)(v176 + 8 * v179);
                      v180 = *v140;
                      if ( *v136 == *v140 )
                        goto LABEL_240;
                      ++v177;
                    }
LABEL_327:
                    if ( v178 )
                      v140 = v178;
                  }
                }
LABEL_240:
                *(_DWORD *)(a1 + 2808) = v146;
                if ( *v140 != -4096 )
                  --*(_DWORD *)(a1 + 2812);
                *v140 = *v136;
                goto LABEL_235;
              }
LABEL_238:
              sub_32026C0(a1 + 2792, 2 * v142);
              v143 = *(_DWORD *)(a1 + 2816);
              if ( !v143 )
                goto LABEL_373;
              v144 = v143 - 1;
              v145 = *(_QWORD *)(a1 + 2800);
              v146 = *(_DWORD *)(a1 + 2808) + 1;
              LODWORD(v147) = v144 & (((unsigned int)*v136 >> 9) ^ ((unsigned int)*v136 >> 4));
              v140 = (_QWORD *)(v145 + 8LL * (unsigned int)v147);
              v148 = *v140;
              if ( *v136 != *v140 )
              {
                v183 = 1;
                v178 = 0;
                while ( v148 != -4096 )
                {
                  if ( v148 == -8192 && !v178 )
                    v178 = v140;
                  v147 = v144 & (unsigned int)(v147 + v183);
                  v140 = (_QWORD *)(v145 + 8 * v147);
                  v148 = *v140;
                  if ( *v136 == *v140 )
                    goto LABEL_240;
                  ++v183;
                }
                goto LABEL_327;
              }
              goto LABEL_240;
            }
LABEL_235:
            if ( v137 == ++v136 )
              goto LABEL_41;
          }
          ++*(_QWORD *)(a1 + 2792);
          goto LABEL_238;
        }
        if ( v187 == *v45 )
          goto LABEL_40;
        ++v45;
      }
      if ( v187 != *v45 )
      {
        ++v45;
        goto LABEL_256;
      }
      goto LABEL_40;
    }
    v96 = *(_DWORD *)(a1 + 2816);
    if ( v96 )
    {
      v97 = *(_QWORD *)(a1 + 2800);
      v98 = (v96 - 1) & (((unsigned int)v187 >> 4) ^ ((unsigned int)v187 >> 9));
      v99 = (__int64 *)(v97 + 8 * v98);
      v100 = *v99;
      if ( v187 == *v99 )
        goto LABEL_41;
      v101 = 1;
      v102 = 0;
      while ( v100 != -4096 )
      {
        if ( v100 == -8192 && !v102 )
          v102 = v99;
        v98 = (v96 - 1) & (v101 + (_DWORD)v98);
        v99 = (__int64 *)(v97 + 8LL * (unsigned int)v98);
        v100 = *v99;
        if ( v187 == *v99 )
          goto LABEL_41;
        ++v101;
      }
      if ( v102 )
        v99 = v102;
      v103 = v43 + 1;
      ++*(_QWORD *)(a1 + 2792);
      if ( 4 * v103 < 3 * v96 )
      {
        if ( v96 - *(_DWORD *)(a1 + 2812) - v103 > v96 >> 3 )
        {
LABEL_278:
          *(_DWORD *)(a1 + 2808) = v103;
          if ( *v99 != -4096 )
            --*(_DWORD *)(a1 + 2812);
          *v99 = v187;
          v165 = *(unsigned int *)(a1 + 2832);
          if ( v165 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 2836) )
          {
            sub_C8D5F0(a1 + 2824, (const void *)(a1 + 2840), v165 + 1, 8u, v100, v98);
            v165 = *(unsigned int *)(a1 + 2832);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 2824) + 8 * v165) = v187;
          ++*(_DWORD *)(a1 + 2832);
          goto LABEL_41;
        }
        sub_32026C0(a1 + 2792, v96);
        v104 = *(_DWORD *)(a1 + 2816);
        if ( v104 )
        {
          v100 = (unsigned int)(v104 - 1);
          v98 = *(_QWORD *)(a1 + 2800);
          v105 = 0;
          v106 = v100 & (((unsigned int)v187 >> 4) ^ ((unsigned int)v187 >> 9));
          v103 = *(_DWORD *)(a1 + 2808) + 1;
          v107 = 1;
          v99 = (__int64 *)(v98 + 8LL * v106);
          v108 = *v99;
          if ( v187 != *v99 )
          {
            while ( v108 != -4096 )
            {
              if ( !v105 && v108 == -8192 )
                v105 = v99;
              v106 = v100 & (v107 + v106);
              v99 = (__int64 *)(v98 + 8LL * v106);
              v108 = *v99;
              if ( v187 == *v99 )
                goto LABEL_278;
              ++v107;
            }
            if ( v105 )
              v99 = v105;
          }
          goto LABEL_278;
        }
LABEL_374:
        ++*(_DWORD *)(a1 + 2808);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 2792);
    }
    sub_32026C0(a1 + 2792, 2 * v96);
    v162 = *(_DWORD *)(a1 + 2816);
    if ( v162 )
    {
      v100 = (unsigned int)(v162 - 1);
      v98 = *(_QWORD *)(a1 + 2800);
      v103 = *(_DWORD *)(a1 + 2808) + 1;
      v163 = v100 & (((unsigned int)v187 >> 9) ^ ((unsigned int)v187 >> 4));
      v99 = (__int64 *)(v98 + 8LL * v163);
      v164 = *v99;
      if ( v187 != *v99 )
      {
        v181 = 1;
        v182 = 0;
        while ( v164 != -4096 )
        {
          if ( v164 == -8192 && !v182 )
            v182 = v99;
          v163 = v100 & (v181 + v163);
          v99 = (__int64 *)(v98 + 8LL * v163);
          v164 = *v99;
          if ( v187 == *v99 )
            goto LABEL_278;
          ++v181;
        }
        if ( v182 )
          v99 = v182;
      }
      goto LABEL_278;
    }
    goto LABEL_374;
  }
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 3048) = 0;
  return result;
}
