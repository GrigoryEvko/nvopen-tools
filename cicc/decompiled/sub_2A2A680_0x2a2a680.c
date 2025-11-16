// Function: sub_2A2A680
// Address: 0x2a2a680
//
__int64 __fastcall sub_2A2A680(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // r15
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r12
  int v7; // r10d
  __int64 v8; // r8
  _QWORD *v9; // rdx
  unsigned int v10; // edi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // rax
  unsigned int *v14; // r12
  __int64 v15; // r9
  unsigned int *v16; // r11
  __int64 v17; // r8
  unsigned int v18; // edi
  _QWORD *v19; // rax
  __int64 v20; // rcx
  unsigned int v21; // esi
  __int64 v22; // r13
  int v23; // esi
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // ecx
  int v27; // eax
  _QWORD *v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rax
  _QWORD *v31; // r13
  unsigned int v32; // esi
  _QWORD *v33; // rdi
  _QWORD *v34; // r15
  int v35; // r10d
  unsigned int v36; // eax
  _QWORD *v37; // r12
  _QWORD *v38; // rdx
  __int64 v39; // r9
  __int64 v40; // r12
  unsigned int v41; // esi
  __int64 v42; // rcx
  int v43; // r14d
  __int64 *v44; // rdi
  __int64 v45; // r9
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // r8
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rdi
  int v53; // eax
  int v54; // eax
  int v55; // eax
  _QWORD *v56; // r10
  int v57; // r11d
  __int64 v58; // rcx
  __int64 v59; // rdi
  int v60; // eax
  int v61; // edx
  __int64 v62; // rax
  int v63; // eax
  int v64; // ecx
  int v65; // ecx
  __int64 v66; // rdi
  _QWORD *v67; // r8
  unsigned int v68; // r14d
  int v69; // r10d
  __int64 v70; // rsi
  unsigned int v71; // ecx
  _QWORD *v72; // rbx
  unsigned __int64 v73; // rdi
  int v75; // r10d
  int v76; // r10d
  __int64 v77; // r8
  __int64 v78; // rcx
  __int64 v79; // rsi
  int v80; // r13d
  _QWORD *v81; // rdi
  int v82; // eax
  __int64 v83; // rsi
  unsigned int v84; // eax
  __int64 v85; // rcx
  int v86; // r11d
  __int64 *v87; // r10
  int v88; // eax
  __int64 v89; // rsi
  int v90; // ecx
  int v91; // r11d
  unsigned int v92; // eax
  int v93; // r9d
  int v94; // r9d
  __int64 v95; // rdi
  _QWORD *v96; // rsi
  __int64 v97; // r13
  int v98; // r10d
  __int64 v99; // rcx
  __int64 *v100; // r13
  __int64 *v101; // rax
  __int64 *v102; // r12
  __int64 v103; // r15
  __int64 v104; // rax
  unsigned int v105; // esi
  __int64 v106; // rbx
  int v107; // r9d
  _QWORD *v108; // rdi
  __int64 v109; // r10
  unsigned int v110; // edx
  _QWORD *v111; // rax
  __int64 v112; // r8
  __int64 *v113; // rax
  int v114; // eax
  int v115; // eax
  __int64 v116; // rax
  int v117; // eax
  int v118; // r11d
  __int64 v119; // r10
  unsigned int v120; // edx
  __int64 v121; // rsi
  int v122; // r9d
  _QWORD *v123; // r8
  int v124; // eax
  int v125; // r11d
  __int64 v126; // r10
  int v127; // r9d
  unsigned int v128; // edx
  __int64 v129; // rsi
  int v130; // r14d
  _QWORD *v131; // r10
  _QWORD *v132; // r14
  __int64 v133; // rsi
  int v134; // r11d
  __int64 v135; // [rsp+8h] [rbp-A8h]
  __int64 *v136; // [rsp+10h] [rbp-A0h]
  __int64 v137; // [rsp+18h] [rbp-98h]
  __int64 v138; // [rsp+18h] [rbp-98h]
  unsigned int *v139; // [rsp+20h] [rbp-90h]
  int v140; // [rsp+20h] [rbp-90h]
  unsigned int *v141; // [rsp+20h] [rbp-90h]
  __int64 v142; // [rsp+28h] [rbp-88h]
  __int64 v143; // [rsp+38h] [rbp-78h]
  __int64 v144; // [rsp+40h] [rbp-70h]
  __int64 v145; // [rsp+48h] [rbp-68h]
  __int64 v146; // [rsp+48h] [rbp-68h]
  __int64 v147; // [rsp+48h] [rbp-68h]
  __int64 *v148; // [rsp+58h] [rbp-58h] BYREF
  __int64 v149; // [rsp+60h] [rbp-50h] BYREF
  _QWORD *v150; // [rsp+68h] [rbp-48h]
  __int64 v151; // [rsp+70h] [rbp-40h]
  unsigned int v152; // [rsp+78h] [rbp-38h]

  v145 = *(_QWORD *)(*(_QWORD *)(a1 + 280) + 8LL);
  v136 = (__int64 *)sub_AA48A0(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v148 = v136;
  v144 = sub_B8CD90(&v148, (__int64)"LVerDomain", 10, 0);
  v2 = *(_QWORD *)(v145 + 168);
  v135 = a1 + 216;
  v143 = v2 + 48LL * *(unsigned int *)(v145 + 176);
  if ( v2 == v143 )
    goto LABEL_19;
  v142 = v145;
  v3 = *(_QWORD *)(v145 + 168);
  do
  {
    v4 = sub_B8CD90(&v148, 0, 0, v144);
    v5 = *(_DWORD *)(a1 + 240);
    v6 = v4;
    if ( !v5 )
    {
      ++*(_QWORD *)(a1 + 216);
      goto LABEL_99;
    }
    v7 = 1;
    v8 = *(_QWORD *)(a1 + 224);
    v9 = 0;
    v10 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == v3 )
      goto LABEL_5;
    while ( 1 )
    {
      if ( v12 == -4096 )
      {
        if ( !v9 )
          v9 = v11;
        v54 = *(_DWORD *)(a1 + 232);
        ++*(_QWORD *)(a1 + 216);
        v55 = v54 + 1;
        if ( 4 * v55 < 3 * v5 )
        {
          if ( v5 - *(_DWORD *)(a1 + 236) - v55 > v5 >> 3 )
          {
LABEL_43:
            *(_DWORD *)(a1 + 232) = v55;
            if ( *v9 != -4096 )
              --*(_DWORD *)(a1 + 236);
            *v9 = v3;
            v13 = v9 + 1;
            v9[1] = 0;
            goto LABEL_6;
          }
          sub_2A29AF0(v135, v5);
          v93 = *(_DWORD *)(a1 + 240);
          if ( v93 )
          {
            v94 = v93 - 1;
            v95 = *(_QWORD *)(a1 + 224);
            v96 = 0;
            LODWORD(v97) = v94 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
            v98 = 1;
            v55 = *(_DWORD *)(a1 + 232) + 1;
            v9 = (_QWORD *)(v95 + 16LL * (unsigned int)v97);
            v99 = *v9;
            if ( *v9 != v3 )
            {
              while ( v99 != -4096 )
              {
                if ( v99 == -8192 && !v96 )
                  v96 = v9;
                v97 = v94 & (unsigned int)(v97 + v98);
                v9 = (_QWORD *)(v95 + 16 * v97);
                v99 = *v9;
                if ( *v9 == v3 )
                  goto LABEL_43;
                ++v98;
              }
              if ( v96 )
                v9 = v96;
            }
            goto LABEL_43;
          }
LABEL_237:
          ++*(_DWORD *)(a1 + 232);
          BUG();
        }
LABEL_99:
        sub_2A29AF0(v135, 2 * v5);
        v75 = *(_DWORD *)(a1 + 240);
        if ( v75 )
        {
          v76 = v75 - 1;
          v77 = *(_QWORD *)(a1 + 224);
          LODWORD(v78) = v76 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v55 = *(_DWORD *)(a1 + 232) + 1;
          v9 = (_QWORD *)(v77 + 16LL * (unsigned int)v78);
          v79 = *v9;
          if ( *v9 != v3 )
          {
            v80 = 1;
            v81 = 0;
            while ( v79 != -4096 )
            {
              if ( !v81 && v79 == -8192 )
                v81 = v9;
              v78 = v76 & (unsigned int)(v78 + v80);
              v9 = (_QWORD *)(v77 + 16 * v78);
              v79 = *v9;
              if ( *v9 == v3 )
                goto LABEL_43;
              ++v80;
            }
            if ( v81 )
              v9 = v81;
          }
          goto LABEL_43;
        }
        goto LABEL_237;
      }
      if ( v9 || v12 != -8192 )
        v11 = v9;
      v10 = (v5 - 1) & (v7 + v10);
      v12 = *(_QWORD *)(v8 + 16LL * v10);
      if ( v12 == v3 )
        break;
      ++v7;
      v9 = v11;
      v11 = (_QWORD *)(v8 + 16LL * v10);
    }
    v11 = (_QWORD *)(v8 + 16LL * v10);
LABEL_5:
    v13 = v11 + 1;
LABEL_6:
    *v13 = v6;
    v14 = *(unsigned int **)(v3 + 16);
    v146 = a1 + 184;
    if ( &v14[*(unsigned int *)(v3 + 24)] == v14 )
      goto LABEL_18;
    v15 = v3;
    v16 = &v14[*(unsigned int *)(v3 + 24)];
    while ( 2 )
    {
      while ( 2 )
      {
        v21 = *(_DWORD *)(a1 + 208);
        v22 = *(_QWORD *)(*(_QWORD *)(v142 + 8) + 72LL * *v14 + 16);
        if ( !v21 )
        {
          ++*(_QWORD *)(a1 + 184);
          goto LABEL_12;
        }
        v17 = *(_QWORD *)(a1 + 192);
        v18 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v19 = (_QWORD *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v22 == *v19 )
        {
LABEL_9:
          ++v14;
          v19[1] = v15;
          if ( v16 == v14 )
            goto LABEL_17;
          continue;
        }
        break;
      }
      v140 = 1;
      v28 = 0;
      while ( v20 != -4096 )
      {
        if ( !v28 && v20 == -8192 )
          v28 = v19;
        v18 = (v21 - 1) & (v140 + v18);
        v19 = (_QWORD *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v22 == *v19 )
          goto LABEL_9;
        ++v140;
      }
      if ( !v28 )
        v28 = v19;
      v63 = *(_DWORD *)(a1 + 200);
      ++*(_QWORD *)(a1 + 184);
      v27 = v63 + 1;
      if ( 4 * v27 >= 3 * v21 )
      {
LABEL_12:
        v137 = v15;
        v139 = v16;
        sub_2A2A140(v146, 2 * v21);
        v23 = *(_DWORD *)(a1 + 208);
        if ( !v23 )
          goto LABEL_234;
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 192);
        v16 = v139;
        v15 = v137;
        v26 = v24 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v27 = *(_DWORD *)(a1 + 200) + 1;
        v28 = (_QWORD *)(v25 + 16LL * v26);
        v29 = *v28;
        if ( v22 != *v28 )
        {
          v130 = 1;
          v131 = 0;
          while ( v29 != -4096 )
          {
            if ( v29 != -8192 || v131 )
              v28 = v131;
            v26 = v24 & (v130 + v26);
            v29 = *(_QWORD *)(v25 + 16LL * v26);
            if ( v22 == v29 )
            {
              v28 = (_QWORD *)(v25 + 16LL * v26);
              goto LABEL_14;
            }
            ++v130;
            v131 = v28;
            v28 = (_QWORD *)(v25 + 16LL * v26);
          }
          if ( v131 )
            v28 = v131;
        }
        goto LABEL_14;
      }
      if ( v21 - *(_DWORD *)(a1 + 204) - v27 <= v21 >> 3 )
      {
        v138 = v15;
        v141 = v16;
        sub_2A2A140(v146, v21);
        v64 = *(_DWORD *)(a1 + 208);
        if ( !v64 )
        {
LABEL_234:
          ++*(_DWORD *)(a1 + 200);
          BUG();
        }
        v65 = v64 - 1;
        v66 = *(_QWORD *)(a1 + 192);
        v67 = 0;
        v68 = v65 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v16 = v141;
        v15 = v138;
        v69 = 1;
        v27 = *(_DWORD *)(a1 + 200) + 1;
        v28 = (_QWORD *)(v66 + 16LL * v68);
        v70 = *v28;
        if ( v22 != *v28 )
        {
          while ( v70 != -4096 )
          {
            if ( !v67 && v70 == -8192 )
              v67 = v28;
            v68 = v65 & (v69 + v68);
            v28 = (_QWORD *)(v66 + 16LL * v68);
            v70 = *v28;
            if ( v22 == *v28 )
              goto LABEL_14;
            ++v69;
          }
          if ( v67 )
            v28 = v67;
        }
      }
LABEL_14:
      *(_DWORD *)(a1 + 200) = v27;
      if ( *v28 != -4096 )
        --*(_DWORD *)(a1 + 204);
      ++v14;
      *v28 = v22;
      v28[1] = 0;
      v28[1] = v15;
      if ( v16 != v14 )
        continue;
      break;
    }
LABEL_17:
    v3 = v15;
LABEL_18:
    v3 += 48;
  }
  while ( v143 != v3 );
LABEL_19:
  v30 = *(unsigned int *)(a1 + 104);
  v31 = *(_QWORD **)(a1 + 96);
  v152 = 0;
  v32 = 0;
  v149 = 0;
  v33 = 0;
  v30 *= 16;
  v150 = 0;
  v34 = (_QWORD *)((char *)v31 + v30);
  v151 = 0;
  if ( (_QWORD *)((char *)v31 + v30) == v31 )
  {
    v132 = 0;
    v133 = 0;
    return sub_C7D6A0((__int64)v132, v133 * 8, 8);
  }
  while ( 2 )
  {
    if ( !v32 )
    {
      ++v149;
      goto LABEL_32;
    }
    v35 = 1;
    v36 = (v32 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
    v37 = &v33[7 * v36];
    v38 = 0;
    v39 = *v37;
    if ( *v31 == *v37 )
      goto LABEL_22;
    while ( 2 )
    {
      if ( v39 == -4096 )
      {
        if ( !v38 )
          v38 = v37;
        ++v149;
        v53 = v151 + 1;
        if ( 4 * ((int)v151 + 1) >= 3 * v32 )
        {
LABEL_32:
          sub_2A2A320((__int64)&v149, 2 * v32);
          if ( v152 )
          {
            LODWORD(v51) = (v152 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
            v38 = &v150[7 * (unsigned int)v51];
            v52 = *v38;
            v53 = v151 + 1;
            if ( *v31 != *v38 )
            {
              v134 = 1;
              v56 = 0;
              while ( v52 != -4096 )
              {
                if ( v52 == -8192 && !v56 )
                  v56 = v38;
                v51 = (v152 - 1) & ((_DWORD)v51 + v134);
                v38 = &v150[7 * v51];
                v52 = *v38;
                if ( *v31 == *v38 )
                  goto LABEL_34;
                ++v134;
              }
              goto LABEL_55;
            }
LABEL_34:
            LODWORD(v151) = v53;
            if ( *v38 != -4096 )
              --HIDWORD(v151);
            v40 = (__int64)(v38 + 1);
            *v38 = *v31;
            v38[1] = v38 + 3;
            v38[2] = 0x400000000LL;
            goto LABEL_23;
          }
        }
        else
        {
          if ( v32 - (v53 + HIDWORD(v151)) > v32 >> 3 )
            goto LABEL_34;
          sub_2A2A320((__int64)&v149, v32);
          if ( v152 )
          {
            v56 = 0;
            v57 = 1;
            LODWORD(v58) = (v152 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
            v38 = &v150[7 * (unsigned int)v58];
            v59 = *v38;
            v53 = v151 + 1;
            if ( *v38 != *v31 )
            {
              while ( v59 != -4096 )
              {
                if ( v59 == -8192 && !v56 )
                  v56 = v38;
                v58 = (v152 - 1) & ((_DWORD)v58 + v57);
                v38 = &v150[7 * v58];
                v59 = *v38;
                if ( *v31 == *v38 )
                  goto LABEL_34;
                ++v57;
              }
LABEL_55:
              if ( v56 )
                v38 = v56;
            }
            goto LABEL_34;
          }
        }
        LODWORD(v151) = v151 + 1;
        BUG();
      }
      if ( v38 || v39 != -8192 )
        v37 = v38;
      v36 = (v32 - 1) & (v35 + v36);
      v39 = v33[7 * v36];
      if ( *v31 != v39 )
      {
        v38 = v37;
        ++v35;
        v37 = &v33[7 * v36];
        continue;
      }
      break;
    }
    v37 = &v33[7 * v36];
LABEL_22:
    v40 = (__int64)(v37 + 1);
LABEL_23:
    v41 = *(_DWORD *)(a1 + 240);
    if ( !v41 )
    {
      ++*(_QWORD *)(a1 + 216);
LABEL_107:
      sub_2A29AF0(a1 + 216, 2 * v41);
      v82 = *(_DWORD *)(a1 + 240);
      if ( v82 )
      {
        v83 = v31[1];
        v45 = (unsigned int)(v82 - 1);
        v48 = *(_QWORD *)(a1 + 224);
        v84 = v45 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
        v61 = *(_DWORD *)(a1 + 232) + 1;
        v44 = (__int64 *)(v48 + 16LL * v84);
        v85 = *v44;
        if ( v83 == *v44 )
          goto LABEL_68;
        v86 = 1;
        v87 = 0;
        while ( v85 != -4096 )
        {
          if ( !v87 && v85 == -8192 )
            v87 = v44;
          v84 = v45 & (v86 + v84);
          v44 = (__int64 *)(v48 + 16LL * v84);
          v85 = *v44;
          if ( v83 == *v44 )
            goto LABEL_68;
          ++v86;
        }
LABEL_111:
        if ( v87 )
          v44 = v87;
        goto LABEL_68;
      }
LABEL_233:
      ++*(_DWORD *)(a1 + 232);
      BUG();
    }
    v42 = v31[1];
    v43 = 1;
    v44 = 0;
    v45 = *(_QWORD *)(a1 + 224);
    v46 = (v41 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
    v47 = (__int64 *)(v45 + 16LL * v46);
    v48 = *v47;
    if ( *v47 == v42 )
    {
LABEL_25:
      v49 = v47[1];
      goto LABEL_26;
    }
    while ( v48 != -4096 )
    {
      if ( !v44 && v48 == -8192 )
        v44 = v47;
      v46 = (v41 - 1) & (v43 + v46);
      v47 = (__int64 *)(v45 + 16LL * v46);
      v48 = *v47;
      if ( v42 == *v47 )
        goto LABEL_25;
      ++v43;
    }
    if ( !v44 )
      v44 = v47;
    v60 = *(_DWORD *)(a1 + 232);
    ++*(_QWORD *)(a1 + 216);
    v61 = v60 + 1;
    if ( 4 * (v60 + 1) >= 3 * v41 )
      goto LABEL_107;
    if ( v41 - *(_DWORD *)(a1 + 236) - v61 <= v41 >> 3 )
    {
      sub_2A29AF0(a1 + 216, v41);
      v88 = *(_DWORD *)(a1 + 240);
      if ( v88 )
      {
        v89 = v31[1];
        v90 = v88 - 1;
        v91 = 1;
        v87 = 0;
        v45 = *(_QWORD *)(a1 + 224);
        v92 = (v88 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
        v61 = *(_DWORD *)(a1 + 232) + 1;
        v44 = (__int64 *)(v45 + 16LL * v92);
        v48 = *v44;
        if ( *v44 == v89 )
          goto LABEL_68;
        while ( v48 != -4096 )
        {
          if ( !v87 && v48 == -8192 )
            v87 = v44;
          v92 = v90 & (v91 + v92);
          v44 = (__int64 *)(v45 + 16LL * v92);
          v48 = *v44;
          if ( v89 == *v44 )
            goto LABEL_68;
          ++v91;
        }
        goto LABEL_111;
      }
      goto LABEL_233;
    }
LABEL_68:
    *(_DWORD *)(a1 + 232) = v61;
    if ( *v44 != -4096 )
      --*(_DWORD *)(a1 + 236);
    v62 = v31[1];
    v49 = 0;
    v44[1] = 0;
    *v44 = v62;
LABEL_26:
    v50 = *(unsigned int *)(v40 + 8);
    if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 12) )
    {
      sub_C8D5F0(v40, (const void *)(v40 + 16), v50 + 1, 8u, v48, v45);
      v50 = *(unsigned int *)(v40 + 8);
    }
    v31 += 2;
    *(_QWORD *)(*(_QWORD *)v40 + 8 * v50) = v49;
    ++*(_DWORD *)(v40 + 8);
    if ( v31 != v34 )
    {
      v33 = v150;
      v32 = v152;
      continue;
    }
    break;
  }
  v132 = v150;
  v71 = v152;
  v133 = 7LL * v152;
  if ( !(_DWORD)v151 )
    goto LABEL_84;
  v100 = &v150[v133];
  if ( &v150[v133] == v150 )
    goto LABEL_84;
  v101 = v150;
  while ( 1 )
  {
    v102 = v101;
    if ( *v101 != -8192 && *v101 != -4096 )
      break;
    v101 += 7;
    if ( v100 == v101 )
      goto LABEL_84;
  }
  if ( v100 != v101 )
  {
    v147 = a1 + 248;
    v103 = a1;
    while ( 1 )
    {
      v104 = sub_B9C770(v136, (__int64 *)v102[1], (__int64 *)*((unsigned int *)v102 + 4), 0, 1);
      v105 = *(_DWORD *)(v103 + 272);
      v106 = v104;
      if ( !v105 )
        break;
      v107 = 1;
      v108 = 0;
      v109 = *(_QWORD *)(v103 + 256);
      v110 = (v105 - 1) & (((unsigned int)*v102 >> 9) ^ ((unsigned int)*v102 >> 4));
      v111 = (_QWORD *)(v109 + 16LL * v110);
      v112 = *v111;
      if ( *v102 == *v111 )
        goto LABEL_137;
      while ( 1 )
      {
        if ( v112 == -4096 )
        {
          if ( !v108 )
            v108 = v111;
          v114 = *(_DWORD *)(v103 + 264);
          ++*(_QWORD *)(v103 + 248);
          v115 = v114 + 1;
          if ( 4 * v115 < 3 * v105 )
          {
            if ( v105 - *(_DWORD *)(v103 + 268) - v115 > v105 >> 3 )
            {
LABEL_149:
              *(_DWORD *)(v103 + 264) = v115;
              if ( *v108 != -4096 )
                --*(_DWORD *)(v103 + 268);
              v116 = *v102;
              v108[1] = 0;
              *v108 = v116;
              v113 = v108 + 1;
              goto LABEL_138;
            }
            sub_2A29AF0(v147, v105);
            v124 = *(_DWORD *)(v103 + 272);
            if ( v124 )
            {
              v125 = v124 - 1;
              v126 = *(_QWORD *)(v103 + 256);
              v123 = 0;
              v127 = 1;
              v128 = (v124 - 1) & (((unsigned int)*v102 >> 9) ^ ((unsigned int)*v102 >> 4));
              v115 = *(_DWORD *)(v103 + 264) + 1;
              v108 = (_QWORD *)(v126 + 16LL * v128);
              v129 = *v108;
              if ( *v108 == *v102 )
                goto LABEL_149;
              while ( v129 != -4096 )
              {
                if ( !v123 && v129 == -8192 )
                  v123 = v108;
                v128 = v125 & (v127 + v128);
                v108 = (_QWORD *)(v126 + 16LL * v128);
                v129 = *v108;
                if ( *v102 == *v108 )
                  goto LABEL_149;
                ++v127;
              }
              goto LABEL_159;
            }
            goto LABEL_236;
          }
LABEL_155:
          sub_2A29AF0(v147, 2 * v105);
          v117 = *(_DWORD *)(v103 + 272);
          if ( v117 )
          {
            v118 = v117 - 1;
            v119 = *(_QWORD *)(v103 + 256);
            v120 = (v117 - 1) & (((unsigned int)*v102 >> 9) ^ ((unsigned int)*v102 >> 4));
            v115 = *(_DWORD *)(v103 + 264) + 1;
            v108 = (_QWORD *)(v119 + 16LL * v120);
            v121 = *v108;
            if ( *v108 == *v102 )
              goto LABEL_149;
            v122 = 1;
            v123 = 0;
            while ( v121 != -4096 )
            {
              if ( !v123 && v121 == -8192 )
                v123 = v108;
              v120 = v118 & (v122 + v120);
              v108 = (_QWORD *)(v119 + 16LL * v120);
              v121 = *v108;
              if ( *v102 == *v108 )
                goto LABEL_149;
              ++v122;
            }
LABEL_159:
            if ( v123 )
              v108 = v123;
            goto LABEL_149;
          }
LABEL_236:
          ++*(_DWORD *)(v103 + 264);
          BUG();
        }
        if ( v108 || v112 != -8192 )
          v111 = v108;
        v110 = (v105 - 1) & (v107 + v110);
        v112 = *(_QWORD *)(v109 + 16LL * v110);
        if ( *v102 == v112 )
          break;
        ++v107;
        v108 = v111;
        v111 = (_QWORD *)(v109 + 16LL * v110);
      }
      v111 = (_QWORD *)(v109 + 16LL * v110);
LABEL_137:
      v113 = v111 + 1;
LABEL_138:
      v102 += 7;
      *v113 = v106;
      if ( v102 != v100 )
      {
        while ( *v102 == -4096 || *v102 == -8192 )
        {
          v102 += 7;
          if ( v100 == v102 )
            goto LABEL_142;
        }
        if ( v102 != v100 )
          continue;
      }
LABEL_142:
      v132 = v150;
      v71 = v152;
      v133 = 7LL * v152;
      goto LABEL_84;
    }
    ++*(_QWORD *)(v103 + 248);
    goto LABEL_155;
  }
LABEL_84:
  if ( v71 )
  {
    v72 = &v132[v133];
    do
    {
      if ( *v132 != -4096 && *v132 != -8192 )
      {
        v73 = v132[1];
        if ( (_QWORD *)v73 != v132 + 3 )
          _libc_free(v73);
      }
      v132 += 7;
    }
    while ( v72 != v132 );
    v132 = v150;
    v133 = 7LL * v152;
  }
  return sub_C7D6A0((__int64)v132, v133 * 8, 8);
}
