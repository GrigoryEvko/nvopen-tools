// Function: sub_187AD60
// Address: 0x187ad60
//
void __fastcall sub_187AD60(char *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  unsigned int v6; // esi
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // edi
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 *v13; // r10
  __int64 v14; // r11
  unsigned int v15; // r14d
  __int64 v16; // r13
  __int64 v17; // rbx
  unsigned int v18; // eax
  __int64 *v19; // r9
  __int64 v20; // r10
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rbx
  char *v25; // r11
  unsigned int v26; // esi
  unsigned int v27; // ecx
  __int64 v28; // rdx
  unsigned int v29; // r9d
  __int64 *v30; // rdi
  __int64 v31; // r8
  unsigned int v32; // r10d
  unsigned int v33; // r9d
  __int64 *v34; // rdi
  __int64 v35; // r8
  int v36; // esi
  int v37; // esi
  __int64 v38; // r8
  __int64 v39; // rcx
  int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // rdi
  int v43; // esi
  int v44; // esi
  __int64 v45; // r8
  unsigned int v46; // ecx
  int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // rdi
  char *v50; // rbx
  unsigned int v51; // edi
  __int64 v52; // rcx
  unsigned int v53; // r9d
  __int64 *v54; // rdx
  __int64 v55; // r8
  unsigned int v56; // r8d
  unsigned int v57; // r10d
  __int64 *v58; // rdx
  __int64 v59; // r9
  __int64 v60; // r12
  __int64 v61; // r13
  int v62; // esi
  int v63; // esi
  __int64 v64; // r8
  unsigned int v65; // ecx
  int v66; // edx
  __int64 *v67; // rax
  __int64 v68; // rdi
  int v69; // esi
  int v70; // esi
  __int64 v71; // r8
  unsigned int v72; // ecx
  int v73; // edx
  _QWORD *v74; // rax
  __int64 v75; // rdi
  __int64 v76; // rax
  int v77; // r14d
  int v78; // ebx
  int v79; // ecx
  int v80; // ecx
  __int64 v81; // rdi
  _QWORD *v82; // r8
  unsigned int v83; // r13d
  int v84; // r10d
  __int64 v85; // rsi
  int v86; // r10d
  int v87; // edx
  int v88; // ecx
  int v89; // ecx
  __int64 v90; // rdi
  __int64 *v91; // r8
  __int64 v92; // r14
  int v93; // r9d
  __int64 v94; // rsi
  int v95; // r13d
  int v96; // edx
  int v97; // ecx
  int v98; // ecx
  __int64 v99; // rdi
  __int64 *v100; // r8
  unsigned int v101; // ebx
  int v102; // r10d
  __int64 v103; // rsi
  int v104; // r14d
  int v105; // edx
  int v106; // ecx
  int v107; // ecx
  __int64 v108; // rdi
  __int64 *v109; // r9
  __int64 v110; // r13
  int v111; // r10d
  __int64 v112; // rsi
  int v113; // r11d
  __int64 *v114; // rcx
  int v115; // eax
  int v116; // eax
  __int64 v117; // rax
  int v118; // r10d
  __int64 *v119; // r9
  int v120; // r10d
  _QWORD *v121; // r9
  __int64 v122; // rbx
  __int64 i; // r12
  __int64 *v124; // rbx
  __int64 v125; // rcx
  __int64 v126; // r13
  __int64 v127; // rax
  __int64 v128; // rax
  int v129; // r13d
  __int64 *v130; // r9
  int v131; // ebx
  int v132; // edx
  int v133; // ebx
  __int64 *v134; // r9
  int v135; // r13d
  __int64 *v136; // r10
  int v137; // r9d
  unsigned int v138; // r10d
  _QWORD *v139; // rbx
  char *v140; // [rsp+0h] [rbp-90h]
  __int64 v141; // [rsp+10h] [rbp-80h]
  char *v142; // [rsp+18h] [rbp-78h]
  char *v143; // [rsp+20h] [rbp-70h]
  char *v144; // [rsp+20h] [rbp-70h]
  char *v145; // [rsp+20h] [rbp-70h]
  char *v146; // [rsp+20h] [rbp-70h]
  char *v147; // [rsp+20h] [rbp-70h]
  char *v148; // [rsp+20h] [rbp-70h]
  char *v149; // [rsp+28h] [rbp-68h]
  char *v150; // [rsp+30h] [rbp-60h]
  __int64 v152; // [rsp+40h] [rbp-50h] BYREF
  __int64 v153; // [rsp+48h] [rbp-48h] BYREF
  __int64 v154; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v155[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v141 = a3;
  v142 = a2;
  if ( a2 - a1 <= 128 )
    return;
  if ( !a3 )
  {
    v149 = a2;
    goto LABEL_136;
  }
  v140 = a1 + 8;
  while ( 2 )
  {
    v152 = a4;
    v6 = *(_DWORD *)(a4 + 24);
    --v141;
    v7 = (__int64 *)&a1[8 * ((__int64)(((v142 - a1) >> 3) + ((unsigned __int64)(v142 - a1) >> 63)) >> 1)];
    v8 = *((_QWORD *)a1 + 1);
    v9 = *v7;
    v153 = v8;
    v154 = v9;
    if ( !v6 )
    {
      ++*(_QWORD *)a4;
LABEL_171:
      v6 *= 2;
      goto LABEL_172;
    }
    v10 = v6 - 1;
    v11 = *(_QWORD *)(a4 + 8);
    LODWORD(v12) = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v13 = (__int64 *)(v11 + 40LL * (unsigned int)v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_6:
      v15 = *((_DWORD *)v13 + 2);
      v16 = a4;
      v17 = a4;
      goto LABEL_7;
    }
    v129 = 1;
    v130 = 0;
    while ( v14 != -4 )
    {
      if ( v14 == -8 && !v130 )
        v130 = v13;
      v12 = v10 & ((_DWORD)v12 + v129);
      v13 = (__int64 *)(v11 + 40 * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_6;
      ++v129;
    }
    v131 = *(_DWORD *)(a4 + 16);
    if ( !v130 )
      v130 = v13;
    ++*(_QWORD *)a4;
    v132 = v131 + 1;
    if ( 4 * (v131 + 1) >= 3 * v6 )
      goto LABEL_171;
    if ( v6 - *(_DWORD *)(a4 + 20) - v132 <= v6 >> 3 )
    {
LABEL_172:
      sub_1874B30(a4, v6);
      sub_18721D0(a4, &v153, v155);
      v130 = (__int64 *)v155[0];
      v8 = v153;
      v132 = *(_DWORD *)(a4 + 16) + 1;
    }
    *(_DWORD *)(a4 + 16) = v132;
    if ( *v130 != -4 )
      --*(_DWORD *)(a4 + 20);
    *v130 = v8;
    v17 = v152;
    *((_DWORD *)v130 + 2) = 0;
    v130[2] = 0;
    v16 = v17;
    v130[3] = 0;
    v130[4] = 0;
    v6 = *(_DWORD *)(v17 + 24);
    if ( !v6 )
    {
      ++*(_QWORD *)v17;
      goto LABEL_168;
    }
    v11 = *(_QWORD *)(v17 + 8);
    v9 = v154;
    v10 = v6 - 1;
    v15 = 0;
LABEL_7:
    v18 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v19 = (__int64 *)(v11 + 40LL * v18);
    v20 = *v19;
    if ( *v19 != v9 )
    {
      v113 = 1;
      v114 = 0;
      while ( v20 != -4 )
      {
        if ( !v114 && v20 == -8 )
          v114 = v19;
        v18 = v10 & (v113 + v18);
        v19 = (__int64 *)(v11 + 40LL * v18);
        v20 = *v19;
        if ( *v19 == v9 )
          goto LABEL_8;
        ++v113;
      }
      v115 = *(_DWORD *)(v17 + 16);
      if ( !v114 )
        v114 = v19;
      ++*(_QWORD *)v17;
      v116 = v115 + 1;
      if ( 4 * v116 < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v17 + 20) - v116 > v6 >> 3 )
        {
LABEL_120:
          *(_DWORD *)(v17 + 16) = v116;
          if ( *v114 != -4 )
            --*(_DWORD *)(v17 + 20);
          v117 = v154;
          *((_DWORD *)v114 + 2) = 0;
          v114[2] = 0;
          *v114 = v117;
          v114[3] = 0;
          v114[4] = 0;
          goto LABEL_123;
        }
LABEL_169:
        sub_1874B30(v16, v6);
        sub_18721D0(v16, &v154, v155);
        v114 = (__int64 *)v155[0];
        v116 = *(_DWORD *)(v17 + 16) + 1;
        goto LABEL_120;
      }
LABEL_168:
      v6 *= 2;
      goto LABEL_169;
    }
LABEL_8:
    if ( *((_DWORD *)v19 + 2) > v15 )
    {
      if ( !sub_18756C0(&v152, *v7, *((_QWORD *)v142 - 1)) )
      {
        if ( sub_18756C0(&v152, *((_QWORD *)a1 + 1), *((_QWORD *)v142 - 1)) )
        {
          v127 = *(_QWORD *)a1;
          *(_QWORD *)a1 = *((_QWORD *)v142 - 1);
          *((_QWORD *)v142 - 1) = v127;
          v23 = *(_QWORD *)a1;
          v24 = *((_QWORD *)a1 + 1);
        }
        else
        {
          v24 = *(_QWORD *)a1;
          v23 = *((_QWORD *)a1 + 1);
          *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
          *(_QWORD *)a1 = v23;
        }
        goto LABEL_12;
      }
      v21 = (__int64 *)a1;
      v22 = *(_QWORD *)a1;
      goto LABEL_11;
    }
LABEL_123:
    if ( sub_18756C0(&v152, *((_QWORD *)a1 + 1), *((_QWORD *)v142 - 1)) )
    {
      v24 = *(_QWORD *)a1;
      v23 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
      *(_QWORD *)a1 = v23;
      goto LABEL_12;
    }
    if ( sub_18756C0(&v152, *v7, *((_QWORD *)v142 - 1)) )
    {
      v128 = *(_QWORD *)a1;
      *(_QWORD *)a1 = *((_QWORD *)v142 - 1);
      *((_QWORD *)v142 - 1) = v128;
      v23 = *(_QWORD *)a1;
      v24 = *((_QWORD *)a1 + 1);
      goto LABEL_12;
    }
    v21 = (__int64 *)a1;
    v22 = *(_QWORD *)a1;
LABEL_11:
    *v21 = *v7;
    *v7 = v22;
    v23 = *v21;
    v24 = v21[1];
LABEL_12:
    v25 = v142;
    v26 = *(_DWORD *)(a4 + 24);
    v150 = v140;
    while ( 2 )
    {
      v149 = v150;
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a4 + 8);
        v29 = (v26 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v30 = (__int64 *)(v28 + 40LL * v29);
        v31 = *v30;
        if ( v24 == *v30 )
        {
LABEL_14:
          v32 = *((_DWORD *)v30 + 2);
          goto LABEL_15;
        }
        v104 = 1;
        v41 = 0;
        while ( v31 != -4 )
        {
          if ( !v41 && v31 == -8 )
            v41 = v30;
          v29 = v27 & (v104 + v29);
          v30 = (__int64 *)(v28 + 40LL * v29);
          v31 = *v30;
          if ( *v30 == v24 )
            goto LABEL_14;
          ++v104;
        }
        v105 = *(_DWORD *)(a4 + 16);
        if ( !v41 )
          v41 = v30;
        ++*(_QWORD *)a4;
        v40 = v105 + 1;
        if ( 4 * v40 < 3 * v26 )
        {
          if ( v26 - *(_DWORD *)(a4 + 20) - v40 <= v26 >> 3 )
          {
            v148 = v25;
            sub_1874B30(a4, v26);
            v106 = *(_DWORD *)(a4 + 24);
            if ( !v106 )
            {
LABEL_236:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
            v107 = v106 - 1;
            v108 = *(_QWORD *)(a4 + 8);
            v109 = 0;
            LODWORD(v110) = v107 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v25 = v148;
            v111 = 1;
            v40 = *(_DWORD *)(a4 + 16) + 1;
            v41 = (__int64 *)(v108 + 40LL * (unsigned int)v110);
            v112 = *v41;
            if ( v24 != *v41 )
            {
              while ( v112 != -4 )
              {
                if ( v112 == -8 && !v109 )
                  v109 = v41;
                v110 = v107 & (unsigned int)(v110 + v111);
                v41 = (__int64 *)(v108 + 40 * v110);
                v112 = *v41;
                if ( v24 == *v41 )
                  goto LABEL_22;
                ++v111;
              }
              if ( v109 )
                v41 = v109;
            }
          }
          goto LABEL_22;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v143 = v25;
      sub_1874B30(a4, 2 * v26);
      v36 = *(_DWORD *)(a4 + 24);
      if ( !v36 )
        goto LABEL_236;
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a4 + 8);
      v25 = v143;
      LODWORD(v39) = v37 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v40 = *(_DWORD *)(a4 + 16) + 1;
      v41 = (__int64 *)(v38 + 40LL * (unsigned int)v39);
      v42 = *v41;
      if ( v24 != *v41 )
      {
        v135 = 1;
        v136 = 0;
        while ( v42 != -4 )
        {
          if ( v42 == -8 && !v136 )
            v136 = v41;
          v39 = v37 & (unsigned int)(v39 + v135);
          v41 = (__int64 *)(v38 + 40 * v39);
          v42 = *v41;
          if ( v24 == *v41 )
            goto LABEL_22;
          ++v135;
        }
        if ( v136 )
          v41 = v136;
      }
LABEL_22:
      *(_DWORD *)(a4 + 16) = v40;
      if ( *v41 != -4 )
        --*(_DWORD *)(a4 + 20);
      *v41 = v24;
      *((_DWORD *)v41 + 2) = 0;
      v41[2] = 0;
      v41[3] = 0;
      v41[4] = 0;
      v26 = *(_DWORD *)(a4 + 24);
      if ( !v26 )
      {
        ++*(_QWORD *)a4;
        goto LABEL_26;
      }
      v28 = *(_QWORD *)(a4 + 8);
      v27 = v26 - 1;
      v32 = 0;
LABEL_15:
      v33 = v27 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v34 = (__int64 *)(v28 + 40LL * v33);
      v35 = *v34;
      if ( v23 == *v34 )
      {
LABEL_16:
        if ( *((_DWORD *)v34 + 2) > v32 )
          goto LABEL_17;
        goto LABEL_31;
      }
      v95 = 1;
      v48 = 0;
      while ( v35 != -4 )
      {
        if ( v35 == -8 && !v48 )
          v48 = v34;
        v33 = v27 & (v95 + v33);
        v34 = (__int64 *)(v28 + 40LL * v33);
        v35 = *v34;
        if ( v23 == *v34 )
          goto LABEL_16;
        ++v95;
      }
      v96 = *(_DWORD *)(a4 + 16);
      if ( !v48 )
        v48 = v34;
      ++*(_QWORD *)a4;
      v47 = v96 + 1;
      if ( 4 * v47 < 3 * v26 )
      {
        if ( v26 - (v47 + *(_DWORD *)(a4 + 20)) <= v26 >> 3 )
        {
          v147 = v25;
          sub_1874B30(a4, v26);
          v97 = *(_DWORD *)(a4 + 24);
          if ( !v97 )
          {
LABEL_239:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          v98 = v97 - 1;
          v99 = *(_QWORD *)(a4 + 8);
          v100 = 0;
          v101 = v98 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v25 = v147;
          v102 = 1;
          v47 = *(_DWORD *)(a4 + 16) + 1;
          v48 = (__int64 *)(v99 + 40LL * v101);
          v103 = *v48;
          if ( v23 != *v48 )
          {
            while ( v103 != -4 )
            {
              if ( v103 == -8 && !v100 )
                v100 = v48;
              v101 = v98 & (v102 + v101);
              v48 = (__int64 *)(v99 + 40LL * v101);
              v103 = *v48;
              if ( v23 == *v48 )
                goto LABEL_28;
              ++v102;
            }
            if ( v100 )
              v48 = v100;
          }
        }
        goto LABEL_28;
      }
LABEL_26:
      v144 = v25;
      sub_1874B30(a4, 2 * v26);
      v43 = *(_DWORD *)(a4 + 24);
      if ( !v43 )
        goto LABEL_239;
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a4 + 8);
      v25 = v144;
      v46 = v44 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v47 = *(_DWORD *)(a4 + 16) + 1;
      v48 = (__int64 *)(v45 + 40LL * v46);
      v49 = *v48;
      if ( v23 != *v48 )
      {
        v133 = 1;
        v134 = 0;
        while ( v49 != -4 )
        {
          if ( !v134 && v49 == -8 )
            v134 = v48;
          v46 = v44 & (v133 + v46);
          v48 = (__int64 *)(v45 + 40LL * v46);
          v49 = *v48;
          if ( v23 == *v48 )
            goto LABEL_28;
          ++v133;
        }
        if ( v134 )
          v48 = v134;
      }
LABEL_28:
      *(_DWORD *)(a4 + 16) = v47;
      if ( *v48 != -4 )
        --*(_DWORD *)(a4 + 20);
      *v48 = v23;
      *((_DWORD *)v48 + 2) = 0;
      v48[2] = 0;
      v48[3] = 0;
      v48[4] = 0;
      v26 = *(_DWORD *)(a4 + 24);
LABEL_31:
      v50 = v25 - 8;
      while ( 1 )
      {
        v60 = *(_QWORD *)v50;
        v25 = v50;
        v61 = *(_QWORD *)a1;
        if ( v26 )
        {
          v51 = v26 - 1;
          v52 = *(_QWORD *)(a4 + 8);
          v53 = (v26 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
          v54 = (__int64 *)(v52 + 40LL * v53);
          v55 = *v54;
          if ( v61 == *v54 )
          {
LABEL_33:
            v56 = *((_DWORD *)v54 + 2);
            goto LABEL_34;
          }
          v86 = 1;
          v67 = 0;
          while ( v55 != -4 )
          {
            if ( v55 == -8 && !v67 )
              v67 = v54;
            v53 = v51 & (v86 + v53);
            v54 = (__int64 *)(v52 + 40LL * v53);
            v55 = *v54;
            if ( v61 == *v54 )
              goto LABEL_33;
            ++v86;
          }
          if ( !v67 )
            v67 = v54;
          v87 = *(_DWORD *)(a4 + 16);
          ++*(_QWORD *)a4;
          v66 = v87 + 1;
          if ( 4 * v66 < 3 * v26 )
          {
            if ( v26 - *(_DWORD *)(a4 + 20) - v66 <= v26 >> 3 )
            {
              sub_1874B30(a4, v26);
              v88 = *(_DWORD *)(a4 + 24);
              if ( !v88 )
              {
LABEL_238:
                ++*(_DWORD *)(a4 + 16);
                BUG();
              }
              v89 = v88 - 1;
              v90 = *(_QWORD *)(a4 + 8);
              v91 = 0;
              LODWORD(v92) = v89 & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
              v25 = v50;
              v93 = 1;
              v66 = *(_DWORD *)(a4 + 16) + 1;
              v67 = (__int64 *)(v90 + 40LL * (unsigned int)v92);
              v94 = *v67;
              if ( v61 != *v67 )
              {
                while ( v94 != -4 )
                {
                  if ( v94 == -8 && !v91 )
                    v91 = v67;
                  v92 = v89 & (unsigned int)(v92 + v93);
                  v67 = (__int64 *)(v90 + 40 * v92);
                  v94 = *v67;
                  if ( v61 == *v67 )
                    goto LABEL_40;
                  ++v93;
                }
                if ( v91 )
                  v67 = v91;
              }
            }
            goto LABEL_40;
          }
        }
        else
        {
          ++*(_QWORD *)a4;
        }
        sub_1874B30(a4, 2 * v26);
        v62 = *(_DWORD *)(a4 + 24);
        if ( !v62 )
          goto LABEL_238;
        v63 = v62 - 1;
        v64 = *(_QWORD *)(a4 + 8);
        v25 = v50;
        v65 = v63 & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
        v66 = *(_DWORD *)(a4 + 16) + 1;
        v67 = (__int64 *)(v64 + 40LL * v65);
        v68 = *v67;
        if ( v61 != *v67 )
        {
          v118 = 1;
          v119 = 0;
          while ( v68 != -4 )
          {
            if ( v119 || v68 != -8 )
              v67 = v119;
            v65 = v63 & (v118 + v65);
            v68 = *(_QWORD *)(v64 + 40LL * v65);
            if ( v61 == v68 )
            {
              v67 = (__int64 *)(v64 + 40LL * v65);
              goto LABEL_40;
            }
            ++v118;
            v119 = v67;
            v67 = (__int64 *)(v64 + 40LL * v65);
          }
          if ( v119 )
            v67 = v119;
        }
LABEL_40:
        *(_DWORD *)(a4 + 16) = v66;
        if ( *v67 != -4 )
          --*(_DWORD *)(a4 + 20);
        *v67 = v61;
        *((_DWORD *)v67 + 2) = 0;
        v67[2] = 0;
        v67[3] = 0;
        v67[4] = 0;
        v26 = *(_DWORD *)(a4 + 24);
        if ( !v26 )
        {
          ++*(_QWORD *)a4;
          goto LABEL_44;
        }
        v52 = *(_QWORD *)(a4 + 8);
        v51 = v26 - 1;
        v56 = 0;
LABEL_34:
        v57 = v51 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
        v58 = (__int64 *)(v52 + 40LL * v57);
        v59 = *v58;
        if ( v60 != *v58 )
          break;
LABEL_35:
        v50 -= 8;
        if ( v56 >= *((_DWORD *)v58 + 2) )
          goto LABEL_49;
      }
      v77 = 1;
      v74 = 0;
      while ( v59 != -4 )
      {
        if ( !v74 && v59 == -8 )
          v74 = v58;
        v57 = v51 & (v77 + v57);
        v58 = (__int64 *)(v52 + 40LL * v57);
        v59 = *v58;
        if ( v60 == *v58 )
          goto LABEL_35;
        ++v77;
      }
      v78 = *(_DWORD *)(a4 + 16);
      if ( !v74 )
        v74 = v58;
      ++*(_QWORD *)a4;
      v73 = v78 + 1;
      if ( 4 * (v78 + 1) >= 3 * v26 )
      {
LABEL_44:
        v145 = v25;
        sub_1874B30(a4, 2 * v26);
        v69 = *(_DWORD *)(a4 + 24);
        if ( !v69 )
          goto LABEL_237;
        v70 = v69 - 1;
        v71 = *(_QWORD *)(a4 + 8);
        v25 = v145;
        v72 = v70 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
        v73 = *(_DWORD *)(a4 + 16) + 1;
        v74 = (_QWORD *)(v71 + 40LL * v72);
        v75 = *v74;
        if ( v60 != *v74 )
        {
          v120 = 1;
          v121 = 0;
          while ( v75 != -4 )
          {
            if ( v75 != -8 || v121 )
              v74 = v121;
            v137 = v120 + 1;
            v138 = v72 + v120;
            v72 = v70 & v138;
            v139 = (_QWORD *)(v71 + 40LL * (v70 & v138));
            v75 = *v139;
            if ( v60 == *v139 )
            {
              v74 = (_QWORD *)(v71 + 40LL * (v70 & v138));
              goto LABEL_46;
            }
            v120 = v137;
            v121 = v74;
            v74 = v139;
          }
          if ( v121 )
            v74 = v121;
        }
      }
      else if ( v26 - (v73 + *(_DWORD *)(a4 + 20)) <= v26 >> 3 )
      {
        v146 = v25;
        sub_1874B30(a4, v26);
        v79 = *(_DWORD *)(a4 + 24);
        if ( v79 )
        {
          v80 = v79 - 1;
          v81 = *(_QWORD *)(a4 + 8);
          v82 = 0;
          v83 = v80 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v25 = v146;
          v84 = 1;
          v73 = *(_DWORD *)(a4 + 16) + 1;
          v74 = (_QWORD *)(v81 + 40LL * v83);
          v85 = *v74;
          if ( v60 != *v74 )
          {
            while ( v85 != -4 )
            {
              if ( v85 == -8 && !v82 )
                v82 = v74;
              v83 = v80 & (v84 + v83);
              v74 = (_QWORD *)(v81 + 40LL * v83);
              v85 = *v74;
              if ( v60 == *v74 )
                goto LABEL_46;
              ++v84;
            }
            if ( v82 )
              v74 = v82;
          }
          goto LABEL_46;
        }
LABEL_237:
        ++*(_DWORD *)(a4 + 16);
        BUG();
      }
LABEL_46:
      *(_DWORD *)(a4 + 16) = v73;
      if ( *v74 != -4 )
        --*(_DWORD *)(a4 + 20);
      *v74 = v60;
      *((_DWORD *)v74 + 2) = 0;
      v74[2] = 0;
      v74[3] = 0;
      v74[4] = 0;
LABEL_49:
      if ( v150 < v25 )
      {
        v76 = *(_QWORD *)v150;
        *(_QWORD *)v150 = *(_QWORD *)v25;
        *(_QWORD *)v25 = v76;
        v26 = *(_DWORD *)(a4 + 24);
LABEL_17:
        v23 = *(_QWORD *)a1;
        v24 = *((_QWORD *)v150 + 1);
        v150 += 8;
        continue;
      }
      break;
    }
    sub_187AD60(v150, v142, v141, a4);
    v4 = v150 - a1;
    if ( v150 - a1 > 128 )
    {
      if ( v141 )
      {
        v142 = v150;
        continue;
      }
LABEL_136:
      v122 = v4 >> 3;
      for ( i = (v122 - 2) >> 1; ; --i )
      {
        sub_187A400((__int64)a1, i, v122, *(_QWORD *)&a1[8 * i], a4);
        if ( !i )
          break;
      }
      v124 = (__int64 *)(v149 - 8);
      do
      {
        v125 = *v124;
        v126 = (char *)v124-- - a1;
        v124[1] = *(_QWORD *)a1;
        sub_187A400((__int64)a1, 0, v126 >> 3, v125, a4);
      }
      while ( v126 > 8 );
    }
    break;
  }
}
