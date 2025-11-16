// Function: sub_26EDE20
// Address: 0x26ede20
//
void __fastcall sub_26EDE20(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  int v6; // r9d
  __int64 v7; // rdi
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  __int64 *v17; // rdx
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 *v20; // rcx
  int v21; // eax
  __int64 *v22; // rdx
  __int64 v23; // rcx
  unsigned int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r10
  int v28; // r13d
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  int v32; // ebx
  unsigned int v33; // r9d
  __int64 v34; // rsi
  unsigned int v35; // ecx
  __int64 v36; // rax
  __int64 v37; // r8
  int v38; // eax
  int v39; // ecx
  int v40; // r10d
  __int64 v41; // r9
  unsigned int v42; // ecx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdi
  __int64 v46; // rsi
  _QWORD *v47; // rax
  int v48; // eax
  int v49; // r11d
  __int64 v50; // r8
  unsigned int v51; // ecx
  __int64 v52; // rax
  __int64 v53; // rdi
  int v54; // edx
  int v55; // r8d
  int v56; // eax
  int v57; // r8d
  _QWORD *v58; // r12
  _QWORD *v59; // rbx
  unsigned int v60; // esi
  __int64 v61; // r9
  unsigned int v62; // eax
  _QWORD *v63; // rdi
  __int64 v64; // r8
  _QWORD *v65; // rax
  int v66; // r11d
  _QWORD *v67; // rcx
  int v68; // eax
  int v69; // eax
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // r12
  __int64 *v73; // rbx
  int v74; // r14d
  __int64 v75; // r9
  unsigned int v76; // esi
  __int64 v77; // r10
  __int64 v78; // r8
  _QWORD *v79; // rax
  __int64 v80; // rdi
  int v81; // ecx
  _QWORD *v82; // rdx
  int v83; // edi
  int v84; // edx
  int v85; // r9d
  int v86; // r9d
  __int64 v87; // r11
  unsigned int v88; // edx
  __int64 v89; // r8
  int v90; // edi
  _QWORD *v91; // rsi
  int v92; // edx
  __int64 v93; // rcx
  __int64 v94; // r9
  int v95; // edi
  __int64 v96; // rsi
  int v97; // r8d
  int v98; // r8d
  __int64 v99; // r10
  int v100; // esi
  _QWORD *v101; // r11
  unsigned int v102; // edx
  __int64 v103; // rdi
  __int64 v104; // r9
  __int64 v105; // r13
  int v106; // ecx
  __int64 v107; // rsi
  int v108; // ecx
  __int64 v109; // rsi
  unsigned int v110; // edx
  int v111; // edi
  __int64 v112; // r9
  unsigned int v113; // edx
  __int64 v114; // r9
  int v115; // edi
  int v116; // r8d
  int v117; // r8d
  __int64 v118; // r10
  __int64 v119; // rcx
  __int64 v120; // rdi
  _QWORD *v121; // rsi
  int v122; // r8d
  int v123; // r8d
  __int64 v124; // r10
  __int64 v125; // rsi
  _QWORD *v126; // rdi
  __int64 v127; // rcx
  int v128; // [rsp+0h] [rbp-A0h]
  __int64 v129; // [rsp+0h] [rbp-A0h]
  unsigned int v131; // [rsp+10h] [rbp-90h]
  __int64 v132; // [rsp+10h] [rbp-90h]
  __int64 *v134; // [rsp+28h] [rbp-78h] BYREF
  __int64 v135; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v136; // [rsp+38h] [rbp-68h]
  __int64 v137; // [rsp+40h] [rbp-60h]
  __int64 v138; // [rsp+48h] [rbp-58h]
  __int64 v139; // [rsp+50h] [rbp-50h] BYREF
  __int64 v140; // [rsp+58h] [rbp-48h]
  __int64 v141; // [rsp+60h] [rbp-40h]
  unsigned int v142; // [rsp+68h] [rbp-38h]

  v4 = *a1;
  v134 = &v135;
  v5 = *(_QWORD *)(v4 + 80);
  v135 = 0;
  v136 = 0;
  if ( v5 )
    v5 -= 24;
  v137 = 0;
  v138 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v139 = 1;
  sub_26E9AE0((__int64)&v139, 0);
  if ( !v142 )
  {
    LODWORD(v141) = v141 + 1;
    BUG();
  }
  v6 = 1;
  v7 = 0;
  v8 = (v142 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v9 = v140 + 16LL * v8;
  v10 = *(_QWORD *)v9;
  if ( v5 != *(_QWORD *)v9 )
  {
    while ( v10 != -4096 )
    {
      if ( !v7 && v10 == -8192 )
        v7 = v9;
      v8 = (v142 - 1) & (v6 + v8);
      v9 = v140 + 16LL * v8;
      v10 = *(_QWORD *)v9;
      if ( v5 == *(_QWORD *)v9 )
        goto LABEL_5;
      ++v6;
    }
    if ( v7 )
      v9 = v7;
  }
LABEL_5:
  LODWORD(v141) = v141 + 1;
  if ( *(_QWORD *)v9 != -4096 )
    --HIDWORD(v141);
  *(_QWORD *)v9 = v5;
  v11 = v4 + 72;
  *(_DWORD *)(v9 + 8) = 2;
  sub_26ED290((__int64 *)&v134, v5);
  v12 = *(_QWORD *)(v4 + 80);
  if ( v12 != v4 + 72 )
  {
    while ( 1 )
    {
      v13 = v12 - 24;
      if ( !v12 )
        v13 = 0;
      v14 = sub_AA4FF0(v13);
      if ( !v14 )
        BUG();
      v15 = (unsigned int)*(unsigned __int8 *)(v14 - 24) - 39;
      if ( (unsigned int)v15 > 0x38 )
        goto LABEL_13;
      v16 = 0x100060000000001LL;
      if ( !_bittest64(&v16, v15) )
        goto LABEL_13;
      sub_26ED290((__int64 *)&v134, v13);
      if ( !v142 )
        break;
      v49 = 1;
      v50 = 0;
      v51 = (v142 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v52 = v140 + 16LL * v51;
      v53 = *(_QWORD *)v52;
      if ( v13 != *(_QWORD *)v52 )
      {
        while ( v53 != -4096 )
        {
          if ( !v50 && v53 == -8192 )
            v50 = v52;
          v51 = (v142 - 1) & (v49 + v51);
          v52 = v140 + 16LL * v51;
          v53 = *(_QWORD *)v52;
          if ( v13 == *(_QWORD *)v52 )
            goto LABEL_53;
          ++v49;
        }
        if ( v50 )
          v52 = v50;
        ++v139;
        v108 = v141 + 1;
        if ( 4 * ((int)v141 + 1) < 3 * v142 )
        {
          if ( v142 - HIDWORD(v141) - v108 <= v142 >> 3 )
          {
            sub_26E9AE0((__int64)&v139, v142);
            if ( !v142 )
              goto LABEL_254;
            v109 = 0;
            v110 = (v142 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v108 = v141 + 1;
            v111 = 1;
            v52 = v140 + 16LL * v110;
            v112 = *(_QWORD *)v52;
            if ( v13 != *(_QWORD *)v52 )
            {
              while ( v112 != -4096 )
              {
                if ( !v109 && v112 == -8192 )
                  v109 = v52;
                v110 = (v142 - 1) & (v111 + v110);
                v52 = v140 + 16LL * v110;
                v112 = *(_QWORD *)v52;
                if ( v13 == *(_QWORD *)v52 )
                  goto LABEL_163;
                ++v111;
              }
              goto LABEL_179;
            }
          }
          goto LABEL_163;
        }
LABEL_175:
        sub_26E9AE0((__int64)&v139, 2 * v142);
        if ( !v142 )
        {
LABEL_254:
          LODWORD(v141) = v141 + 1;
          BUG();
        }
        v113 = (v142 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v108 = v141 + 1;
        v52 = v140 + 16LL * v113;
        v114 = *(_QWORD *)v52;
        if ( v13 != *(_QWORD *)v52 )
        {
          v115 = 1;
          v109 = 0;
          while ( v114 != -4096 )
          {
            if ( !v109 && v114 == -8192 )
              v109 = v52;
            v113 = (v142 - 1) & (v115 + v113);
            v52 = v140 + 16LL * v113;
            v114 = *(_QWORD *)v52;
            if ( v13 == *(_QWORD *)v52 )
              goto LABEL_163;
            ++v115;
          }
LABEL_179:
          if ( v109 )
            v52 = v109;
        }
LABEL_163:
        LODWORD(v141) = v108;
        if ( *(_QWORD *)v52 != -4096 )
          --HIDWORD(v141);
        *(_QWORD *)v52 = v13;
        *(_DWORD *)(v52 + 8) = 0;
      }
LABEL_53:
      *(_DWORD *)(v52 + 8) = 1;
LABEL_13:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v11 == v12 )
        goto LABEL_14;
    }
    ++v139;
    goto LABEL_175;
  }
LABEL_14:
  while ( (_DWORD)v137 )
  {
    v17 = &v136[(unsigned int)v138];
    v18 = *v136;
    if ( v136 != v17 )
    {
      v19 = v136;
      while ( 1 )
      {
        v18 = *v19;
        v20 = v19;
        if ( *v19 != -4096 && v18 != -8192 )
          break;
        if ( v17 == ++v19 )
        {
          v18 = v20[1];
          break;
        }
      }
    }
    if ( (_DWORD)v138 )
    {
      v21 = (v138 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v22 = &v136[v21];
      v23 = *v22;
      if ( v18 == *v22 )
      {
LABEL_23:
        *v22 = -8192;
        LODWORD(v137) = v137 - 1;
        ++HIDWORD(v137);
      }
      else
      {
        v54 = 1;
        while ( v23 != -4096 )
        {
          v55 = v54 + 1;
          v21 = (v138 - 1) & (v54 + v21);
          v22 = &v136[v21];
          v23 = *v22;
          if ( *v22 == v18 )
            goto LABEL_23;
          v54 = v55;
        }
      }
    }
    if ( v142 )
    {
      v24 = (v142 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v25 = v140 + 16LL * v24;
      v26 = *(_QWORD *)v25;
      if ( *(_QWORD *)v25 == v18 )
      {
LABEL_26:
        v27 = v140 + 16LL * v142;
        if ( v27 == v25 )
          v28 = 0;
        else
          v28 = *(_DWORD *)(v25 + 8);
        goto LABEL_28;
      }
      v56 = 1;
      while ( v26 != -4096 )
      {
        v57 = v56 + 1;
        v24 = (v142 - 1) & (v56 + v24);
        v25 = v140 + 16LL * v24;
        v26 = *(_QWORD *)v25;
        if ( *(_QWORD *)v25 == v18 )
          goto LABEL_26;
        v56 = v57;
      }
    }
    v28 = 0;
    v27 = v140 + 16LL * v142;
LABEL_28:
    v29 = *(_QWORD *)(v18 + 16);
    do
    {
      if ( !v29 )
        goto LABEL_14;
      v30 = *(_QWORD *)(v29 + 24);
      v31 = v29;
      v29 = *(_QWORD *)(v29 + 8);
    }
    while ( (unsigned __int8)(*(_BYTE *)v30 - 30) > 0xAu );
    v32 = v28;
    v33 = v142 - 1;
LABEL_33:
    v34 = *(_QWORD *)(v30 + 40);
    if ( v142 )
    {
      v35 = v33 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v36 = v140 + 16LL * v35;
      v37 = *(_QWORD *)v36;
      if ( *(_QWORD *)v36 == v34 )
      {
LABEL_35:
        if ( v36 != v27 )
        {
          v38 = *(_DWORD *)(v36 + 8);
          v39 = v38;
          goto LABEL_37;
        }
      }
      else
      {
        v48 = 1;
        while ( v37 != -4096 )
        {
          v35 = v33 & (v48 + v35);
          v128 = v48 + 1;
          v36 = v140 + 16LL * v35;
          v37 = *(_QWORD *)v36;
          if ( v34 == *(_QWORD *)v36 )
            goto LABEL_35;
          v48 = v128;
        }
      }
      v39 = 0;
      v38 = 0;
    }
    else
    {
      v38 = 0;
      v39 = 0;
    }
LABEL_37:
    if ( v32 < v39 )
      v32 = v38;
    while ( 1 )
    {
      v31 = *(_QWORD *)(v31 + 8);
      if ( !v31 )
        break;
      v30 = *(_QWORD *)(v31 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v30 - 30) <= 0xAu )
        goto LABEL_33;
    }
    if ( v32 == v28 )
      continue;
    sub_26ED290((__int64 *)&v134, v18);
    if ( !v142 )
    {
      ++v139;
      goto LABEL_134;
    }
    v40 = 1;
    v41 = 0;
    v42 = (v142 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v43 = v140 + 16LL * v42;
    v44 = *(_QWORD *)v43;
    if ( v18 != *(_QWORD *)v43 )
    {
      while ( v44 != -4096 )
      {
        if ( !v41 && v44 == -8192 )
          v41 = v43;
        v42 = (v142 - 1) & (v40 + v42);
        v43 = v140 + 16LL * v42;
        v44 = *(_QWORD *)v43;
        if ( v18 == *(_QWORD *)v43 )
          goto LABEL_43;
        ++v40;
      }
      if ( v41 )
        v43 = v41;
      ++v139;
      v92 = v141 + 1;
      if ( 4 * ((int)v141 + 1) >= 3 * v142 )
      {
LABEL_134:
        sub_26E9AE0((__int64)&v139, 2 * v142);
        if ( !v142 )
          goto LABEL_252;
        LODWORD(v93) = (v142 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v92 = v141 + 1;
        v43 = v140 + 16LL * (unsigned int)v93;
        v94 = *(_QWORD *)v43;
        if ( v18 != *(_QWORD *)v43 )
        {
          v95 = 1;
          v96 = 0;
          while ( v94 != -4096 )
          {
            if ( v94 == -8192 && !v96 )
              v96 = v43;
            v93 = (v142 - 1) & ((_DWORD)v93 + v95);
            v43 = v140 + 16 * v93;
            v94 = *(_QWORD *)v43;
            if ( v18 == *(_QWORD *)v43 )
              goto LABEL_130;
            ++v95;
          }
          if ( v96 )
            v43 = v96;
        }
      }
      else if ( v142 - HIDWORD(v141) - v92 <= v142 >> 3 )
      {
        sub_26E9AE0((__int64)&v139, v142);
        if ( !v142 )
        {
LABEL_252:
          LODWORD(v141) = v141 + 1;
          BUG();
        }
        v104 = 0;
        LODWORD(v105) = (v142 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v92 = v141 + 1;
        v106 = 1;
        v43 = v140 + 16LL * (unsigned int)v105;
        v107 = *(_QWORD *)v43;
        if ( v18 != *(_QWORD *)v43 )
        {
          while ( v107 != -4096 )
          {
            if ( v107 == -8192 && !v104 )
              v104 = v43;
            v105 = (v142 - 1) & ((_DWORD)v105 + v106);
            v43 = v140 + 16 * v105;
            v107 = *(_QWORD *)v43;
            if ( v18 == *(_QWORD *)v43 )
              goto LABEL_130;
            ++v106;
          }
          if ( v104 )
            v43 = v104;
        }
      }
LABEL_130:
      LODWORD(v141) = v92;
      if ( *(_QWORD *)v43 != -4096 )
        --HIDWORD(v141);
      *(_QWORD *)v43 = v18;
      *(_DWORD *)(v43 + 8) = 0;
    }
LABEL_43:
    *(_DWORD *)(v43 + 8) = v32;
  }
  v45 = v140;
  v46 = 16LL * v142;
  if ( (_DWORD)v141 )
  {
    v70 = v140 + v46;
    if ( v140 != v140 + v46 )
    {
      v71 = v140;
      while ( 1 )
      {
        v72 = *(_QWORD *)v71;
        v73 = (__int64 *)v71;
        if ( *(_QWORD *)v71 != -8192 && v72 != -4096 )
          break;
        v71 += 16;
        if ( v70 == v71 )
          goto LABEL_45;
      }
      if ( v70 != v71 )
      {
        v74 = *(_DWORD *)(v71 + 8);
        v75 = a2;
        if ( v74 == 1 )
          goto LABEL_101;
        while ( 1 )
        {
          do
          {
LABEL_94:
            v73 += 2;
            if ( v73 == (__int64 *)v70 )
              goto LABEL_98;
            while ( 1 )
            {
              v72 = *v73;
              if ( *v73 != -8192 && v72 != -4096 )
                break;
              v73 += 2;
              if ( (__int64 *)v70 == v73 )
                goto LABEL_98;
            }
            if ( v73 == (__int64 *)v70 )
            {
LABEL_98:
              v45 = v140;
              a2 = v75;
              v46 = 16LL * v142;
              goto LABEL_45;
            }
            v74 = *((_DWORD *)v73 + 2);
          }
          while ( v74 != 1 );
LABEL_101:
          v76 = *(_DWORD *)(a3 + 24);
          if ( !v76 )
            break;
          v77 = *(_QWORD *)(a3 + 8);
          v131 = ((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4);
          v78 = (v76 - 1) & v131;
          v79 = (_QWORD *)(v77 + 8 * v78);
          v80 = *v79;
          if ( v72 != *v79 )
          {
            v81 = 1;
            v82 = 0;
            while ( v80 != -4096 )
            {
              if ( v80 == -8192 && !v82 )
                v82 = v79;
              LODWORD(v78) = (v76 - 1) & (v81 + v78);
              v79 = (_QWORD *)(v77 + 8LL * (unsigned int)v78);
              v80 = *v79;
              if ( v72 == *v79 )
                goto LABEL_94;
              ++v81;
            }
            v83 = *(_DWORD *)(a3 + 16);
            if ( v82 )
              v79 = v82;
            ++*(_QWORD *)a3;
            v84 = v83 + 1;
            if ( 4 * (v83 + 1) >= 3 * v76 )
              goto LABEL_188;
            if ( v76 - *(_DWORD *)(a3 + 20) - v84 <= v76 >> 3 )
            {
              v129 = v75;
              sub_CF28B0(a3, v76);
              v122 = *(_DWORD *)(a3 + 24);
              if ( !v122 )
              {
LABEL_251:
                ++*(_DWORD *)(a3 + 16);
                BUG();
              }
              v123 = v122 - 1;
              v124 = *(_QWORD *)(a3 + 8);
              v75 = v129;
              LODWORD(v125) = v123 & v131;
              v84 = *(_DWORD *)(a3 + 16) + 1;
              v126 = 0;
              v79 = (_QWORD *)(v124 + 8LL * (v123 & v131));
              v127 = *v79;
              if ( v72 != *v79 )
              {
                while ( v127 != -4096 )
                {
                  if ( v127 == -8192 && !v126 )
                    v126 = v79;
                  v125 = v123 & (unsigned int)(v125 + v74);
                  v79 = (_QWORD *)(v124 + 8 * v125);
                  v127 = *v79;
                  if ( v72 == *v79 )
                    goto LABEL_109;
                  ++v74;
                }
                if ( v126 )
                  v79 = v126;
              }
            }
LABEL_109:
            *(_DWORD *)(a3 + 16) = v84;
            if ( *v79 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v79 = v72;
          }
        }
        ++*(_QWORD *)a3;
LABEL_188:
        v132 = v75;
        sub_CF28B0(a3, 2 * v76);
        v116 = *(_DWORD *)(a3 + 24);
        if ( !v116 )
          goto LABEL_251;
        v117 = v116 - 1;
        v118 = *(_QWORD *)(a3 + 8);
        v75 = v132;
        LODWORD(v119) = v117 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v84 = *(_DWORD *)(a3 + 16) + 1;
        v79 = (_QWORD *)(v118 + 8LL * (unsigned int)v119);
        v120 = *v79;
        if ( v72 != *v79 )
        {
          v121 = 0;
          while ( v120 != -4096 )
          {
            if ( v120 == -8192 && !v121 )
              v121 = v79;
            v119 = v117 & (unsigned int)(v119 + v74);
            v79 = (_QWORD *)(v118 + 8 * v119);
            v120 = *v79;
            if ( v72 == *v79 )
              goto LABEL_109;
            ++v74;
          }
          if ( v121 )
            v79 = v121;
        }
        goto LABEL_109;
      }
    }
  }
LABEL_45:
  sub_C7D6A0(v45, v46, 8);
  sub_C7D6A0((__int64)v136, 8LL * (unsigned int)v138, 8);
  sub_26EDB30((__int64)a1, a3);
  v47 = *(_QWORD **)(a3 + 8);
  if ( *(_DWORD *)(a3 + 16) )
  {
    v58 = &v47[*(unsigned int *)(a3 + 24)];
    if ( v47 != v58 )
    {
      while ( 1 )
      {
        v59 = v47;
        if ( *v47 != -4096 && *v47 != -8192 )
          break;
        if ( v58 == ++v47 )
          goto LABEL_46;
      }
      while ( v58 != v59 )
      {
        v60 = *(_DWORD *)(a2 + 24);
        if ( v60 )
        {
          v61 = *(_QWORD *)(a2 + 8);
          v62 = (v60 - 1) & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
          v63 = (_QWORD *)(v61 + 8LL * v62);
          v64 = *v63;
          if ( *v63 == *v59 )
            goto LABEL_73;
          v66 = 1;
          v67 = 0;
          while ( v64 != -4096 )
          {
            if ( v64 == -8192 && !v67 )
              v67 = v63;
            v62 = (v60 - 1) & (v66 + v62);
            v63 = (_QWORD *)(v61 + 8LL * v62);
            v64 = *v63;
            if ( *v59 == *v63 )
              goto LABEL_73;
            ++v66;
          }
          v68 = *(_DWORD *)(a2 + 16);
          if ( !v67 )
            v67 = v63;
          ++*(_QWORD *)a2;
          v69 = v68 + 1;
          if ( 4 * v69 < 3 * v60 )
          {
            if ( v60 - *(_DWORD *)(a2 + 20) - v69 <= v60 >> 3 )
            {
              sub_CF28B0(a2, v60);
              v97 = *(_DWORD *)(a2 + 24);
              if ( !v97 )
              {
LABEL_253:
                ++*(_DWORD *)(a2 + 16);
                BUG();
              }
              v98 = v97 - 1;
              v99 = *(_QWORD *)(a2 + 8);
              v100 = 1;
              v101 = 0;
              v102 = v98 & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
              v67 = (_QWORD *)(v99 + 8LL * v102);
              v103 = *v67;
              v69 = *(_DWORD *)(a2 + 16) + 1;
              if ( *v67 != *v59 )
              {
                while ( v103 != -4096 )
                {
                  if ( v103 == -8192 && !v101 )
                    v101 = v67;
                  v102 = v98 & (v100 + v102);
                  v67 = (_QWORD *)(v99 + 8LL * v102);
                  v103 = *v67;
                  if ( *v59 == *v67 )
                    goto LABEL_84;
                  ++v100;
                }
                if ( v101 )
                  v67 = v101;
              }
            }
            goto LABEL_84;
          }
        }
        else
        {
          ++*(_QWORD *)a2;
        }
        sub_CF28B0(a2, 2 * v60);
        v85 = *(_DWORD *)(a2 + 24);
        if ( !v85 )
          goto LABEL_253;
        v86 = v85 - 1;
        v87 = *(_QWORD *)(a2 + 8);
        v88 = v86 & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
        v67 = (_QWORD *)(v87 + 8LL * v88);
        v89 = *v67;
        v69 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v59 != *v67 )
        {
          v90 = 1;
          v91 = 0;
          while ( v89 != -4096 )
          {
            if ( !v91 && v89 == -8192 )
              v91 = v67;
            v88 = v86 & (v90 + v88);
            v67 = (_QWORD *)(v87 + 8LL * v88);
            v89 = *v67;
            if ( *v59 == *v67 )
              goto LABEL_84;
            ++v90;
          }
          if ( v91 )
            v67 = v91;
        }
LABEL_84:
        *(_DWORD *)(a2 + 16) = v69;
        if ( *v67 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v67 = *v59;
LABEL_73:
        v65 = v59 + 1;
        if ( v58 == v59 + 1 )
          break;
        while ( 1 )
        {
          v59 = v65;
          if ( *v65 != -4096 && *v65 != -8192 )
            break;
          if ( v58 == ++v65 )
            goto LABEL_46;
        }
      }
    }
  }
LABEL_46:
  sub_26ED580((__int64)a1, a2);
}
