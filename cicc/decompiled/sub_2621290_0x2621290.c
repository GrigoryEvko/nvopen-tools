// Function: sub_2621290
// Address: 0x2621290
//
void __fastcall sub_2621290(char *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  unsigned int v6; // esi
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r8d
  __int64 v11; // rdi
  int v12; // r13d
  __int64 v13; // rcx
  __int64 *v14; // r10
  __int64 *v15; // r9
  __int64 v16; // r11
  unsigned int v17; // r10d
  __int64 v18; // r13
  __int64 v19; // rbx
  int v20; // r11d
  __int64 *v21; // r14
  unsigned int v22; // eax
  __int64 *v23; // rcx
  __int64 v24; // r9
  __int64 *v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rbx
  char *v29; // r11
  unsigned int v30; // esi
  unsigned int v31; // ecx
  __int64 v32; // rdx
  int v33; // r14d
  unsigned int v34; // r9d
  __int64 *v35; // rdi
  __int64 *v36; // rax
  __int64 v37; // r8
  unsigned int v38; // r10d
  int v39; // r13d
  unsigned int v40; // r9d
  __int64 *v41; // rdi
  __int64 *v42; // rax
  __int64 v43; // r8
  int v44; // esi
  int v45; // esi
  __int64 v46; // r8
  __int64 v47; // rcx
  int v48; // edx
  __int64 v49; // rdi
  int v50; // esi
  int v51; // esi
  __int64 v52; // r8
  unsigned int v53; // ecx
  int v54; // edx
  __int64 v55; // rdi
  char *v56; // rbx
  unsigned int v57; // edi
  __int64 v58; // rcx
  int v59; // r10d
  unsigned int v60; // r9d
  __int64 *v61; // rdx
  __int64 *v62; // rax
  __int64 v63; // r8
  unsigned int v64; // r8d
  int v65; // r14d
  unsigned int v66; // r13d
  unsigned int v67; // r10d
  __int64 *v68; // rdx
  __int64 *v69; // rax
  __int64 v70; // r9
  __int64 v71; // r12
  __int64 v72; // r13
  int v73; // esi
  int v74; // esi
  __int64 v75; // r8
  unsigned int v76; // ecx
  int v77; // edx
  __int64 v78; // rdi
  int v79; // esi
  int v80; // esi
  __int64 v81; // r8
  unsigned int v82; // ecx
  int v83; // edx
  __int64 v84; // rdi
  __int64 v85; // rax
  int v86; // ebx
  int v87; // ecx
  int v88; // ecx
  __int64 v89; // rdi
  __int64 *v90; // r8
  unsigned int v91; // r13d
  int v92; // r10d
  __int64 v93; // rsi
  int v94; // edx
  int v95; // ecx
  int v96; // ecx
  __int64 v97; // rdi
  __int64 *v98; // r8
  __int64 v99; // r14
  int v100; // r9d
  __int64 v101; // rsi
  int v102; // edx
  int v103; // ecx
  int v104; // ecx
  __int64 v105; // rdi
  __int64 *v106; // r8
  unsigned int v107; // ebx
  int v108; // r10d
  __int64 v109; // rsi
  int v110; // edx
  int v111; // ecx
  int v112; // ecx
  __int64 v113; // rdi
  __int64 *v114; // r9
  __int64 v115; // r13
  int v116; // r10d
  __int64 v117; // rsi
  int v118; // eax
  int v119; // eax
  __int64 v120; // rax
  __int64 v121; // rdx
  int v122; // r10d
  __int64 *v123; // r9
  int v124; // ebx
  __int64 *v125; // r9
  __int64 v126; // rbx
  __int64 i; // r12
  __int64 *v128; // rbx
  __int64 v129; // rcx
  __int64 v130; // r13
  __int64 v131; // rax
  __int64 v132; // rax
  int v133; // ebx
  int v134; // edx
  int v135; // ebx
  __int64 *v136; // r9
  int v137; // r13d
  __int64 *v138; // r10
  char *v139; // [rsp+0h] [rbp-90h]
  __int64 v140; // [rsp+10h] [rbp-80h]
  char *v141; // [rsp+18h] [rbp-78h]
  char *v142; // [rsp+20h] [rbp-70h]
  char *v143; // [rsp+20h] [rbp-70h]
  char *v144; // [rsp+20h] [rbp-70h]
  char *v145; // [rsp+20h] [rbp-70h]
  char *v146; // [rsp+20h] [rbp-70h]
  char *v147; // [rsp+20h] [rbp-70h]
  char *v148; // [rsp+28h] [rbp-68h]
  char *v149; // [rsp+30h] [rbp-60h]
  __int64 v151; // [rsp+40h] [rbp-50h] BYREF
  __int64 v152; // [rsp+48h] [rbp-48h] BYREF
  __int64 v153; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v154[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v140 = a3;
  v141 = a2;
  if ( a2 - a1 <= 128 )
    return;
  if ( !a3 )
  {
    v148 = a2;
    goto LABEL_146;
  }
  v139 = a1 + 8;
  while ( 2 )
  {
    v151 = a4;
    v6 = *(_DWORD *)(a4 + 24);
    --v140;
    v7 = (__int64 *)&a1[8 * ((__int64)(((v141 - a1) >> 3) + ((unsigned __int64)(v141 - a1) >> 63)) >> 1)];
    v8 = *((_QWORD *)a1 + 1);
    v9 = *v7;
    v152 = v8;
    v153 = v9;
    if ( !v6 )
    {
      ++*(_QWORD *)a4;
      v154[0] = 0;
LABEL_175:
      v6 *= 2;
      goto LABEL_176;
    }
    v10 = v6 - 1;
    v11 = *(_QWORD *)(a4 + 8);
    v12 = 1;
    LODWORD(v13) = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v14 = (__int64 *)(v11 + 40LL * (unsigned int)v13);
    v15 = 0;
    v16 = *v14;
    if ( v8 == *v14 )
    {
LABEL_6:
      v17 = *((_DWORD *)v14 + 2);
      v18 = a4;
      v19 = a4;
      goto LABEL_7;
    }
    while ( v16 != -4096 )
    {
      if ( v16 == -8192 && !v15 )
        v15 = v14;
      v13 = v10 & ((_DWORD)v13 + v12);
      v14 = (__int64 *)(v11 + 40 * v13);
      v16 = *v14;
      if ( v8 == *v14 )
        goto LABEL_6;
      ++v12;
    }
    v133 = *(_DWORD *)(a4 + 16);
    if ( !v15 )
      v15 = v14;
    ++*(_QWORD *)a4;
    v134 = v133 + 1;
    v154[0] = v15;
    if ( 4 * (v133 + 1) >= 3 * v6 )
      goto LABEL_175;
    if ( v6 - *(_DWORD *)(a4 + 20) - v134 <= v6 >> 3 )
    {
LABEL_176:
      sub_261D190(a4, v6);
      sub_2618CC0(a4, &v152, v154);
      v8 = v152;
      v15 = (__int64 *)v154[0];
      v134 = *(_DWORD *)(a4 + 16) + 1;
    }
    *(_DWORD *)(a4 + 16) = v134;
    if ( *v15 != -4096 )
      --*(_DWORD *)(a4 + 20);
    *v15 = v8;
    v19 = v151;
    *(_OWORD *)(v15 + 1) = 0;
    *(_OWORD *)(v15 + 3) = 0;
    v6 = *(_DWORD *)(v19 + 24);
    v18 = v19;
    if ( !v6 )
    {
      v154[0] = 0;
      ++*(_QWORD *)v19;
      goto LABEL_172;
    }
    v11 = *(_QWORD *)(v19 + 8);
    v9 = v153;
    v10 = v6 - 1;
    v17 = 0;
LABEL_7:
    v20 = 1;
    v21 = 0;
    v22 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v23 = (__int64 *)(v11 + 40LL * v22);
    v24 = *v23;
    if ( v9 != *v23 )
    {
      while ( v24 != -4096 )
      {
        if ( !v21 && v24 == -8192 )
          v21 = v23;
        v22 = v10 & (v20 + v22);
        v23 = (__int64 *)(v11 + 40LL * v22);
        v24 = *v23;
        if ( *v23 == v9 )
          goto LABEL_8;
        ++v20;
      }
      if ( !v21 )
        v21 = v23;
      v154[0] = v21;
      v118 = *(_DWORD *)(v19 + 16);
      ++*(_QWORD *)v19;
      v119 = v118 + 1;
      if ( 4 * v119 < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v19 + 20) - v119 > v6 >> 3 )
        {
LABEL_130:
          *(_DWORD *)(v19 + 16) = v119;
          v120 = v154[0];
          if ( *(_QWORD *)v154[0] != -4096 )
            --*(_DWORD *)(v19 + 20);
          v121 = v153;
          *(_OWORD *)(v120 + 8) = 0;
          *(_QWORD *)v120 = v121;
          *(_OWORD *)(v120 + 24) = 0;
          goto LABEL_133;
        }
LABEL_173:
        sub_261D190(v18, v6);
        sub_2618CC0(v18, &v153, v154);
        v119 = *(_DWORD *)(v19 + 16) + 1;
        goto LABEL_130;
      }
LABEL_172:
      v6 *= 2;
      goto LABEL_173;
    }
LABEL_8:
    if ( *((_DWORD *)v23 + 2) > v17 )
    {
      if ( !sub_261D3E0(&v151, *v7, *((_QWORD *)v141 - 1)) )
      {
        if ( sub_261D3E0(&v151, *((_QWORD *)a1 + 1), *((_QWORD *)v141 - 1)) )
        {
          v131 = *(_QWORD *)a1;
          *(_QWORD *)a1 = *((_QWORD *)v141 - 1);
          *((_QWORD *)v141 - 1) = v131;
          v27 = *(_QWORD *)a1;
          v28 = *((_QWORD *)a1 + 1);
        }
        else
        {
          v28 = *(_QWORD *)a1;
          v27 = *((_QWORD *)a1 + 1);
          *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
          *(_QWORD *)a1 = v27;
        }
        goto LABEL_12;
      }
      v25 = (__int64 *)a1;
      v26 = *(_QWORD *)a1;
      goto LABEL_11;
    }
LABEL_133:
    if ( sub_261D3E0(&v151, *((_QWORD *)a1 + 1), *((_QWORD *)v141 - 1)) )
    {
      v28 = *(_QWORD *)a1;
      v27 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
      *(_QWORD *)a1 = v27;
      goto LABEL_12;
    }
    if ( sub_261D3E0(&v151, *v7, *((_QWORD *)v141 - 1)) )
    {
      v132 = *(_QWORD *)a1;
      *(_QWORD *)a1 = *((_QWORD *)v141 - 1);
      *((_QWORD *)v141 - 1) = v132;
      v27 = *(_QWORD *)a1;
      v28 = *((_QWORD *)a1 + 1);
      goto LABEL_12;
    }
    v25 = (__int64 *)a1;
    v26 = *(_QWORD *)a1;
LABEL_11:
    *v25 = *v7;
    *v7 = v26;
    v27 = *v25;
    v28 = v25[1];
LABEL_12:
    v29 = v141;
    v30 = *(_DWORD *)(a4 + 24);
    v149 = v139;
    while ( 2 )
    {
      v148 = v149;
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a4 + 8);
        v33 = 1;
        v34 = (v30 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v35 = (__int64 *)(v32 + 40LL * v34);
        v36 = 0;
        v37 = *v35;
        if ( v28 == *v35 )
        {
LABEL_14:
          v38 = *((_DWORD *)v35 + 2);
          goto LABEL_15;
        }
        while ( v37 != -4096 )
        {
          if ( !v36 && v37 == -8192 )
            v36 = v35;
          v34 = v31 & (v33 + v34);
          v35 = (__int64 *)(v32 + 40LL * v34);
          v37 = *v35;
          if ( *v35 == v28 )
            goto LABEL_14;
          ++v33;
        }
        v110 = *(_DWORD *)(a4 + 16);
        if ( !v36 )
          v36 = v35;
        ++*(_QWORD *)a4;
        v48 = v110 + 1;
        if ( 4 * v48 < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(a4 + 20) - v48 <= v30 >> 3 )
          {
            v147 = v29;
            sub_261D190(a4, v30);
            v111 = *(_DWORD *)(a4 + 24);
            if ( !v111 )
            {
LABEL_232:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
            v112 = v111 - 1;
            v113 = *(_QWORD *)(a4 + 8);
            v114 = 0;
            LODWORD(v115) = v112 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v29 = v147;
            v116 = 1;
            v48 = *(_DWORD *)(a4 + 16) + 1;
            v36 = (__int64 *)(v113 + 40LL * (unsigned int)v115);
            v117 = *v36;
            if ( v28 != *v36 )
            {
              while ( v117 != -4096 )
              {
                if ( v117 == -8192 && !v114 )
                  v114 = v36;
                v115 = v112 & (unsigned int)(v115 + v116);
                v36 = (__int64 *)(v113 + 40 * v115);
                v117 = *v36;
                if ( *v36 == v28 )
                  goto LABEL_22;
                ++v116;
              }
              if ( v114 )
                v36 = v114;
            }
          }
          goto LABEL_22;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v142 = v29;
      sub_261D190(a4, 2 * v30);
      v44 = *(_DWORD *)(a4 + 24);
      if ( !v44 )
        goto LABEL_232;
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a4 + 8);
      v29 = v142;
      LODWORD(v47) = v45 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v48 = *(_DWORD *)(a4 + 16) + 1;
      v36 = (__int64 *)(v46 + 40LL * (unsigned int)v47);
      v49 = *v36;
      if ( v28 != *v36 )
      {
        v137 = 1;
        v138 = 0;
        while ( v49 != -4096 )
        {
          if ( v49 == -8192 && !v138 )
            v138 = v36;
          v47 = v45 & (unsigned int)(v47 + v137);
          v36 = (__int64 *)(v46 + 40 * v47);
          v49 = *v36;
          if ( *v36 == v28 )
            goto LABEL_22;
          ++v137;
        }
        if ( v138 )
          v36 = v138;
      }
LABEL_22:
      *(_DWORD *)(a4 + 16) = v48;
      if ( *v36 != -4096 )
        --*(_DWORD *)(a4 + 20);
      *v36 = v28;
      *(_OWORD *)(v36 + 1) = 0;
      *(_OWORD *)(v36 + 3) = 0;
      v30 = *(_DWORD *)(a4 + 24);
      if ( !v30 )
      {
        ++*(_QWORD *)a4;
        goto LABEL_26;
      }
      v32 = *(_QWORD *)(a4 + 8);
      v31 = v30 - 1;
      v38 = 0;
LABEL_15:
      v39 = 1;
      v40 = v31 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v41 = (__int64 *)(v32 + 40LL * v40);
      v42 = 0;
      v43 = *v41;
      if ( *v41 == v27 )
      {
LABEL_16:
        if ( *((_DWORD *)v41 + 2) > v38 )
          goto LABEL_17;
        goto LABEL_31;
      }
      while ( v43 != -4096 )
      {
        if ( v43 == -8192 && !v42 )
          v42 = v41;
        v40 = v31 & (v39 + v40);
        v41 = (__int64 *)(v32 + 40LL * v40);
        v43 = *v41;
        if ( *v41 == v27 )
          goto LABEL_16;
        ++v39;
      }
      v102 = *(_DWORD *)(a4 + 16);
      if ( !v42 )
        v42 = v41;
      ++*(_QWORD *)a4;
      v54 = v102 + 1;
      if ( 4 * v54 < 3 * v30 )
      {
        if ( v30 - (v54 + *(_DWORD *)(a4 + 20)) <= v30 >> 3 )
        {
          v146 = v29;
          sub_261D190(a4, v30);
          v103 = *(_DWORD *)(a4 + 24);
          if ( !v103 )
          {
LABEL_231:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          v104 = v103 - 1;
          v105 = *(_QWORD *)(a4 + 8);
          v106 = 0;
          v107 = v104 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v29 = v146;
          v108 = 1;
          v54 = *(_DWORD *)(a4 + 16) + 1;
          v42 = (__int64 *)(v105 + 40LL * v107);
          v109 = *v42;
          if ( *v42 != v27 )
          {
            while ( v109 != -4096 )
            {
              if ( v109 == -8192 && !v106 )
                v106 = v42;
              v107 = v104 & (v108 + v107);
              v42 = (__int64 *)(v105 + 40LL * v107);
              v109 = *v42;
              if ( *v42 == v27 )
                goto LABEL_28;
              ++v108;
            }
            if ( v106 )
              v42 = v106;
          }
        }
        goto LABEL_28;
      }
LABEL_26:
      v143 = v29;
      sub_261D190(a4, 2 * v30);
      v50 = *(_DWORD *)(a4 + 24);
      if ( !v50 )
        goto LABEL_231;
      v51 = v50 - 1;
      v52 = *(_QWORD *)(a4 + 8);
      v29 = v143;
      v53 = v51 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v54 = *(_DWORD *)(a4 + 16) + 1;
      v42 = (__int64 *)(v52 + 40LL * v53);
      v55 = *v42;
      if ( *v42 != v27 )
      {
        v135 = 1;
        v136 = 0;
        while ( v55 != -4096 )
        {
          if ( !v136 && v55 == -8192 )
            v136 = v42;
          v53 = v51 & (v135 + v53);
          v42 = (__int64 *)(v52 + 40LL * v53);
          v55 = *v42;
          if ( *v42 == v27 )
            goto LABEL_28;
          ++v135;
        }
        if ( v136 )
          v42 = v136;
      }
LABEL_28:
      *(_DWORD *)(a4 + 16) = v54;
      if ( *v42 != -4096 )
        --*(_DWORD *)(a4 + 20);
      *v42 = v27;
      *(_OWORD *)(v42 + 1) = 0;
      *(_OWORD *)(v42 + 3) = 0;
      v30 = *(_DWORD *)(a4 + 24);
LABEL_31:
      v56 = v29 - 8;
      while ( 1 )
      {
        v71 = *(_QWORD *)v56;
        v29 = v56;
        v72 = *(_QWORD *)a1;
        if ( v30 )
        {
          v57 = v30 - 1;
          v58 = *(_QWORD *)(a4 + 8);
          v59 = 1;
          v60 = (v30 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
          v61 = (__int64 *)(v58 + 40LL * v60);
          v62 = 0;
          v63 = *v61;
          if ( v72 == *v61 )
          {
LABEL_33:
            v64 = *((_DWORD *)v61 + 2);
            goto LABEL_34;
          }
          while ( v63 != -4096 )
          {
            if ( v63 == -8192 && !v62 )
              v62 = v61;
            v60 = v57 & (v59 + v60);
            v61 = (__int64 *)(v58 + 40LL * v60);
            v63 = *v61;
            if ( v72 == *v61 )
              goto LABEL_33;
            ++v59;
          }
          if ( !v62 )
            v62 = v61;
          v94 = *(_DWORD *)(a4 + 16);
          ++*(_QWORD *)a4;
          v77 = v94 + 1;
          if ( 4 * v77 < 3 * v30 )
          {
            if ( v30 - *(_DWORD *)(a4 + 20) - v77 <= v30 >> 3 )
            {
              sub_261D190(a4, v30);
              v95 = *(_DWORD *)(a4 + 24);
              if ( !v95 )
              {
LABEL_230:
                ++*(_DWORD *)(a4 + 16);
                BUG();
              }
              v96 = v95 - 1;
              v97 = *(_QWORD *)(a4 + 8);
              v98 = 0;
              LODWORD(v99) = v96 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
              v29 = v56;
              v100 = 1;
              v77 = *(_DWORD *)(a4 + 16) + 1;
              v62 = (__int64 *)(v97 + 40LL * (unsigned int)v99);
              v101 = *v62;
              if ( v72 != *v62 )
              {
                while ( v101 != -4096 )
                {
                  if ( v101 == -8192 && !v98 )
                    v98 = v62;
                  v99 = v96 & (unsigned int)(v99 + v100);
                  v62 = (__int64 *)(v97 + 40 * v99);
                  v101 = *v62;
                  if ( v72 == *v62 )
                    goto LABEL_40;
                  ++v100;
                }
                if ( v98 )
                  v62 = v98;
              }
            }
            goto LABEL_40;
          }
        }
        else
        {
          ++*(_QWORD *)a4;
        }
        sub_261D190(a4, 2 * v30);
        v73 = *(_DWORD *)(a4 + 24);
        if ( !v73 )
          goto LABEL_230;
        v74 = v73 - 1;
        v75 = *(_QWORD *)(a4 + 8);
        v29 = v56;
        v76 = v74 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v77 = *(_DWORD *)(a4 + 16) + 1;
        v62 = (__int64 *)(v75 + 40LL * v76);
        v78 = *v62;
        if ( v72 != *v62 )
        {
          v122 = 1;
          v123 = 0;
          while ( v78 != -4096 )
          {
            if ( v123 || v78 != -8192 )
              v62 = v123;
            v76 = v74 & (v122 + v76);
            v78 = *(_QWORD *)(v75 + 40LL * v76);
            if ( v72 == v78 )
            {
              v62 = (__int64 *)(v75 + 40LL * v76);
              goto LABEL_40;
            }
            ++v122;
            v123 = v62;
            v62 = (__int64 *)(v75 + 40LL * v76);
          }
          if ( v123 )
            v62 = v123;
        }
LABEL_40:
        *(_DWORD *)(a4 + 16) = v77;
        if ( *v62 != -4096 )
          --*(_DWORD *)(a4 + 20);
        *v62 = v72;
        *(_OWORD *)(v62 + 1) = 0;
        *(_OWORD *)(v62 + 3) = 0;
        v30 = *(_DWORD *)(a4 + 24);
        if ( !v30 )
        {
          ++*(_QWORD *)a4;
          goto LABEL_44;
        }
        v58 = *(_QWORD *)(a4 + 8);
        v57 = v30 - 1;
        v64 = 0;
LABEL_34:
        v65 = 1;
        v66 = ((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4);
        v67 = v66 & v57;
        v68 = (__int64 *)(v58 + 40LL * (v66 & v57));
        v69 = 0;
        v70 = *v68;
        if ( v71 != *v68 )
          break;
LABEL_35:
        v56 -= 8;
        if ( v64 >= *((_DWORD *)v68 + 2) )
          goto LABEL_49;
      }
      while ( v70 != -4096 )
      {
        if ( !v69 && v70 == -8192 )
          v69 = v68;
        v67 = v57 & (v65 + v67);
        v68 = (__int64 *)(v58 + 40LL * v67);
        v70 = *v68;
        if ( v71 == *v68 )
          goto LABEL_35;
        ++v65;
      }
      v86 = *(_DWORD *)(a4 + 16);
      if ( !v69 )
        v69 = v68;
      ++*(_QWORD *)a4;
      v83 = v86 + 1;
      if ( 4 * (v86 + 1) >= 3 * v30 )
      {
LABEL_44:
        v144 = v29;
        sub_261D190(a4, 2 * v30);
        v79 = *(_DWORD *)(a4 + 24);
        if ( !v79 )
          goto LABEL_229;
        v80 = v79 - 1;
        v81 = *(_QWORD *)(a4 + 8);
        v29 = v144;
        v82 = v80 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
        v83 = *(_DWORD *)(a4 + 16) + 1;
        v69 = (__int64 *)(v81 + 40LL * v82);
        v84 = *v69;
        if ( v71 != *v69 )
        {
          v124 = 1;
          v125 = 0;
          while ( v84 != -4096 )
          {
            if ( v84 == -8192 && !v125 )
              v125 = v69;
            v82 = v80 & (v124 + v82);
            v69 = (__int64 *)(v81 + 40LL * v82);
            v84 = *v69;
            if ( v71 == *v69 )
              goto LABEL_46;
            ++v124;
          }
          if ( v125 )
            v69 = v125;
        }
      }
      else if ( v30 - (v83 + *(_DWORD *)(a4 + 20)) <= v30 >> 3 )
      {
        v145 = v29;
        sub_261D190(a4, v30);
        v87 = *(_DWORD *)(a4 + 24);
        if ( v87 )
        {
          v88 = v87 - 1;
          v89 = *(_QWORD *)(a4 + 8);
          v90 = 0;
          v91 = v88 & v66;
          v29 = v145;
          v92 = 1;
          v83 = *(_DWORD *)(a4 + 16) + 1;
          v69 = (__int64 *)(v89 + 40LL * v91);
          v93 = *v69;
          if ( v71 != *v69 )
          {
            while ( v93 != -4096 )
            {
              if ( v93 == -8192 && !v90 )
                v90 = v69;
              v91 = v88 & (v92 + v91);
              v69 = (__int64 *)(v89 + 40LL * v91);
              v93 = *v69;
              if ( v71 == *v69 )
                goto LABEL_46;
              ++v92;
            }
            if ( v90 )
              v69 = v90;
          }
          goto LABEL_46;
        }
LABEL_229:
        ++*(_DWORD *)(a4 + 16);
        BUG();
      }
LABEL_46:
      *(_DWORD *)(a4 + 16) = v83;
      if ( *v69 != -4096 )
        --*(_DWORD *)(a4 + 20);
      *v69 = v71;
      *(_OWORD *)(v69 + 1) = 0;
      *(_OWORD *)(v69 + 3) = 0;
LABEL_49:
      if ( v149 < v29 )
      {
        v85 = *(_QWORD *)v149;
        *(_QWORD *)v149 = *(_QWORD *)v29;
        *(_QWORD *)v29 = v85;
        v30 = *(_DWORD *)(a4 + 24);
LABEL_17:
        v27 = *(_QWORD *)a1;
        v28 = *((_QWORD *)v149 + 1);
        v149 += 8;
        continue;
      }
      break;
    }
    sub_2621290(v149, v141, v140, a4);
    v4 = v149 - a1;
    if ( v149 - a1 > 128 )
    {
      if ( v140 )
      {
        v141 = v149;
        continue;
      }
LABEL_146:
      v126 = v4 >> 3;
      for ( i = (v126 - 2) >> 1; ; --i )
      {
        sub_2620730((__int64)a1, i, v126, *(_QWORD *)&a1[8 * i], a4);
        if ( !i )
          break;
      }
      v128 = (__int64 *)(v148 - 8);
      do
      {
        v129 = *v128;
        v130 = (char *)v128-- - a1;
        v128[1] = *(_QWORD *)a1;
        sub_2620730((__int64)a1, 0, v130 >> 3, v129, a4);
      }
      while ( v130 > 8 );
    }
    break;
  }
}
