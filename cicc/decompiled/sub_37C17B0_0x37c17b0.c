// Function: sub_37C17B0
// Address: 0x37c17b0
//
__int64 __fastcall sub_37C17B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r11
  __int64 *v7; // r12
  bool v8; // zf
  __int64 v9; // rdx
  bool v10; // al
  __int64 *v11; // r11
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 *v14; // r13
  unsigned int v15; // esi
  unsigned int v16; // edi
  __int64 v17; // rcx
  int v18; // r10d
  __int64 *v19; // rdx
  unsigned int v20; // r9d
  __int64 *v21; // rax
  __int64 v22; // r8
  unsigned int v23; // r10d
  __int64 v24; // r12
  int v25; // r15d
  __int64 *v26; // rdx
  unsigned int v27; // r9d
  __int64 *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r12
  int v31; // r9d
  int v32; // r9d
  __int64 v33; // rdi
  unsigned int v34; // ecx
  int v35; // eax
  __int64 v36; // rsi
  int v37; // esi
  int v38; // esi
  __int64 v39; // r8
  unsigned int v40; // ecx
  int v41; // eax
  __int64 v42; // rdi
  unsigned int v43; // edi
  __int64 v44; // rcx
  int v45; // r10d
  __int64 *v46; // rdx
  unsigned int v47; // r9d
  __int64 *v48; // rax
  __int64 v49; // r8
  unsigned int v50; // r8d
  __int64 v51; // r12
  int v52; // r15d
  __int64 *v53; // rdx
  unsigned int v54; // r10d
  __int64 *v55; // rax
  __int64 v56; // r9
  __int64 v57; // r15
  __int64 v58; // r12
  int v59; // eax
  int v60; // edi
  __int64 v61; // rsi
  unsigned int v62; // eax
  int v63; // ecx
  __int64 v64; // r8
  int v65; // esi
  int v66; // esi
  __int64 v67; // r8
  unsigned int v68; // ecx
  int v69; // eax
  __int64 v70; // rdi
  __int64 v71; // rax
  int v72; // eax
  int v73; // ecx
  int v74; // ecx
  __int64 *v75; // r9
  unsigned int v76; // r14d
  __int64 v77; // rdi
  int v78; // r10d
  __int64 v79; // rsi
  int v80; // eax
  int v81; // eax
  int v82; // eax
  __int64 *v83; // r8
  unsigned int v84; // r14d
  __int64 v85; // rdi
  int v86; // r9d
  __int64 v87; // rsi
  int v88; // eax
  int v89; // ecx
  int v90; // ecx
  __int64 *v91; // r9
  __int64 v92; // r14
  __int64 v93; // rdi
  int v94; // r10d
  __int64 v95; // rsi
  int v96; // eax
  int v97; // r8d
  int v98; // r8d
  __int64 *v99; // r9
  unsigned int v100; // r14d
  __int64 v101; // rsi
  int v102; // r10d
  __int64 v103; // rcx
  bool v104; // al
  __int64 v105; // r12
  __int64 v106; // r13
  __int64 *v107; // r12
  __int64 v108; // rcx
  int v109; // r10d
  int v110; // r14d
  __int64 *v111; // r9
  __int64 *v112; // r15
  bool v113; // al
  bool v114; // al
  int v115; // r14d
  __int64 *v116; // r10
  int v117; // r14d
  __int64 *v118; // r10
  __int64 *v119; // [rsp+0h] [rbp-80h]
  __int64 v120; // [rsp+10h] [rbp-70h]
  __int64 *v121; // [rsp+18h] [rbp-68h]
  __int64 *v122; // [rsp+20h] [rbp-60h]
  __int64 *v123; // [rsp+20h] [rbp-60h]
  __int64 *v124; // [rsp+20h] [rbp-60h]
  __int64 *v125; // [rsp+20h] [rbp-60h]
  __int64 *v126; // [rsp+20h] [rbp-60h]
  __int64 *v127; // [rsp+20h] [rbp-60h]
  __int64 *v128; // [rsp+20h] [rbp-60h]
  __int64 *v129; // [rsp+20h] [rbp-60h]
  __int64 *v130; // [rsp+20h] [rbp-60h]
  __int64 *v131; // [rsp+28h] [rbp-58h]
  __int64 v132; // [rsp+30h] [rbp-50h]
  __int64 *v133; // [rsp+38h] [rbp-48h]
  __int64 *i; // [rsp+38h] [rbp-48h]
  __int64 *v135; // [rsp+38h] [rbp-48h]
  __int64 v136[7]; // [rsp+48h] [rbp-38h] BYREF

  result = (__int64)a2 - a1;
  v121 = a2;
  v120 = a3;
  if ( (__int64)a2 - a1 <= 128 )
    return result;
  v5 = a1;
  if ( !a3 )
  {
    v131 = a2;
    goto LABEL_119;
  }
  v119 = (__int64 *)(a1 + 8);
  v132 = a4 + 664;
  while ( 2 )
  {
    v136[0] = a4;
    v7 = (__int64 *)(v5 + 8 * (result >> 4));
    --v120;
    v133 = (__int64 *)v5;
    v8 = !sub_37C0D30(v136, *(__int64 **)(*(_QWORD *)(v5 + 8) + 80LL), *v7);
    v9 = *(v121 - 1);
    if ( v8 )
    {
      v104 = sub_37C0D30(v136, *(__int64 **)(v133[1] + 80), v9);
      v11 = v133;
      if ( !v104 )
      {
        v112 = v121;
        v114 = sub_37C0D30(v136, *(__int64 **)(*v7 + 80), *(v121 - 1));
        v11 = v133;
        v8 = !v114;
        v12 = *v133;
        if ( v8 )
          goto LABEL_7;
LABEL_136:
        *v11 = *(v112 - 1);
        *(v112 - 1) = v12;
        v13 = *v11;
        v12 = v11[1];
        goto LABEL_8;
      }
      v12 = *v133;
LABEL_117:
      v13 = v11[1];
      v11[1] = v12;
      *v11 = v13;
      goto LABEL_8;
    }
    v10 = sub_37C0D30(v136, *(__int64 **)(*v7 + 80), v9);
    v11 = v133;
    if ( !v10 )
    {
      v112 = v121;
      v113 = sub_37C0D30(v136, *(__int64 **)(v133[1] + 80), *(v121 - 1));
      v11 = v133;
      v8 = !v113;
      v12 = *v133;
      if ( !v8 )
        goto LABEL_136;
      goto LABEL_117;
    }
    v12 = *v133;
LABEL_7:
    *v11 = *v7;
    *v7 = v12;
    v13 = *v11;
    v12 = v11[1];
LABEL_8:
    v14 = v121;
    v15 = *(_DWORD *)(a4 + 688);
    for ( i = v119; ; ++i )
    {
      v131 = i;
      v30 = **(_QWORD **)(v12 + 80);
      if ( v15 )
      {
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a4 + 672);
        v18 = 1;
        v19 = 0;
        v20 = (v15 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v21 = (__int64 *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( v30 == *v21 )
        {
LABEL_10:
          v23 = *((_DWORD *)v21 + 2);
          v24 = **(_QWORD **)(v13 + 80);
          goto LABEL_11;
        }
        while ( v22 != -4096 )
        {
          if ( !v19 && v22 == -8192 )
            v19 = v21;
          v20 = v16 & (v18 + v20);
          v21 = (__int64 *)(v17 + 16LL * v20);
          v22 = *v21;
          if ( v30 == *v21 )
            goto LABEL_10;
          ++v18;
        }
        if ( !v19 )
          v19 = v21;
        v96 = *(_DWORD *)(a4 + 680);
        ++*(_QWORD *)(a4 + 664);
        v35 = v96 + 1;
        if ( 4 * v35 < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(a4 + 684) - v35 <= v15 >> 3 )
          {
            v129 = v11;
            sub_2E515B0(v132, v15);
            v97 = *(_DWORD *)(a4 + 688);
            if ( !v97 )
            {
LABEL_190:
              ++*(_DWORD *)(a4 + 680);
              BUG();
            }
            v98 = v97 - 1;
            v99 = 0;
            v11 = v129;
            v100 = v98 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v101 = *(_QWORD *)(a4 + 672);
            v102 = 1;
            v35 = *(_DWORD *)(a4 + 680) + 1;
            v19 = (__int64 *)(v101 + 16LL * v100);
            v103 = *v19;
            if ( v30 != *v19 )
            {
              while ( v103 != -4096 )
              {
                if ( v103 == -8192 && !v99 )
                  v99 = v19;
                v100 = v98 & (v102 + v100);
                v19 = (__int64 *)(v101 + 16LL * v100);
                v103 = *v19;
                if ( v30 == *v19 )
                  goto LABEL_18;
                ++v102;
              }
              if ( v99 )
                v19 = v99;
            }
          }
          goto LABEL_18;
        }
      }
      else
      {
        ++*(_QWORD *)(a4 + 664);
      }
      v122 = v11;
      sub_2E515B0(v132, 2 * v15);
      v31 = *(_DWORD *)(a4 + 688);
      if ( !v31 )
        goto LABEL_190;
      v32 = v31 - 1;
      v11 = v122;
      v33 = *(_QWORD *)(a4 + 672);
      v34 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v35 = *(_DWORD *)(a4 + 680) + 1;
      v19 = (__int64 *)(v33 + 16LL * v34);
      v36 = *v19;
      if ( v30 != *v19 )
      {
        v117 = 1;
        v118 = 0;
        while ( v36 != -4096 )
        {
          if ( !v118 && v36 == -8192 )
            v118 = v19;
          v34 = v32 & (v117 + v34);
          v19 = (__int64 *)(v33 + 16LL * v34);
          v36 = *v19;
          if ( v30 == *v19 )
            goto LABEL_18;
          ++v117;
        }
        if ( v118 )
          v19 = v118;
      }
LABEL_18:
      *(_DWORD *)(a4 + 680) = v35;
      if ( *v19 != -4096 )
        --*(_DWORD *)(a4 + 684);
      *v19 = v30;
      *((_DWORD *)v19 + 2) = 0;
      v15 = *(_DWORD *)(a4 + 688);
      v24 = **(_QWORD **)(v13 + 80);
      if ( !v15 )
      {
        ++*(_QWORD *)(a4 + 664);
        goto LABEL_22;
      }
      v17 = *(_QWORD *)(a4 + 672);
      v16 = v15 - 1;
      v23 = 0;
LABEL_11:
      v25 = 1;
      v26 = 0;
      v27 = v16 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v28 = (__int64 *)(v17 + 16LL * v27);
      v29 = *v28;
      if ( *v28 != v24 )
        break;
LABEL_12:
      if ( v23 >= *((_DWORD *)v28 + 2) )
        goto LABEL_31;
LABEL_13:
      v13 = *v11;
      v12 = i[1];
    }
    while ( v29 != -4096 )
    {
      if ( !v26 && v29 == -8192 )
        v26 = v28;
      v27 = v16 & (v25 + v27);
      v28 = (__int64 *)(v17 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v24 )
        goto LABEL_12;
      ++v25;
    }
    if ( !v26 )
      v26 = v28;
    v88 = *(_DWORD *)(a4 + 680);
    ++*(_QWORD *)(a4 + 664);
    v41 = v88 + 1;
    if ( 4 * v41 >= 3 * v15 )
    {
LABEL_22:
      v123 = v11;
      sub_2E515B0(v132, 2 * v15);
      v37 = *(_DWORD *)(a4 + 688);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a4 + 672);
        v11 = v123;
        v40 = v38 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v41 = *(_DWORD *)(a4 + 680) + 1;
        v26 = (__int64 *)(v39 + 16LL * v40);
        v42 = *v26;
        if ( *v26 != v24 )
        {
          v115 = 1;
          v116 = 0;
          while ( v42 != -4096 )
          {
            if ( !v116 && v42 == -8192 )
              v116 = v26;
            v40 = v38 & (v40 + v115);
            v26 = (__int64 *)(v39 + 16LL * v40);
            v42 = *v26;
            if ( *v26 == v24 )
              goto LABEL_24;
            ++v115;
          }
          if ( v116 )
            v26 = v116;
        }
        goto LABEL_24;
      }
LABEL_188:
      ++*(_DWORD *)(a4 + 680);
      BUG();
    }
    if ( v15 - (v41 + *(_DWORD *)(a4 + 684)) > v15 >> 3 )
      goto LABEL_24;
    v128 = v11;
    sub_2E515B0(v132, v15);
    v89 = *(_DWORD *)(a4 + 688);
    if ( !v89 )
      goto LABEL_188;
    v90 = v89 - 1;
    v91 = 0;
    v11 = v128;
    LODWORD(v92) = v90 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v93 = *(_QWORD *)(a4 + 672);
    v94 = 1;
    v41 = *(_DWORD *)(a4 + 680) + 1;
    v26 = (__int64 *)(v93 + 16LL * (unsigned int)v92);
    v95 = *v26;
    if ( *v26 != v24 )
    {
      while ( v95 != -4096 )
      {
        if ( !v91 && v95 == -8192 )
          v91 = v26;
        v92 = v90 & (unsigned int)(v92 + v94);
        v26 = (__int64 *)(v93 + 16 * v92);
        v95 = *v26;
        if ( *v26 == v24 )
          goto LABEL_24;
        ++v94;
      }
      if ( v91 )
        v26 = v91;
    }
LABEL_24:
    *(_DWORD *)(a4 + 680) = v41;
    if ( *v26 != -4096 )
      --*(_DWORD *)(a4 + 684);
    *v26 = v24;
    *((_DWORD *)v26 + 2) = 0;
    v15 = *(_DWORD *)(a4 + 688);
LABEL_31:
    while ( 2 )
    {
      v57 = *--v14;
      v58 = **(_QWORD **)(*v11 + 80);
      if ( !v15 )
      {
        ++*(_QWORD *)(a4 + 664);
        goto LABEL_33;
      }
      v43 = v15 - 1;
      v44 = *(_QWORD *)(a4 + 672);
      v45 = 1;
      v46 = 0;
      v47 = (v15 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v48 = (__int64 *)(v44 + 16LL * v47);
      v49 = *v48;
      if ( *v48 == v58 )
      {
LABEL_28:
        v50 = *((_DWORD *)v48 + 2);
        v51 = **(_QWORD **)(v57 + 80);
      }
      else
      {
        while ( v49 != -4096 )
        {
          if ( !v46 && v49 == -8192 )
            v46 = v48;
          v47 = v43 & (v45 + v47);
          v48 = (__int64 *)(v44 + 16LL * v47);
          v49 = *v48;
          if ( v58 == *v48 )
            goto LABEL_28;
          ++v45;
        }
        if ( !v46 )
          v46 = v48;
        v80 = *(_DWORD *)(a4 + 680);
        ++*(_QWORD *)(a4 + 664);
        v63 = v80 + 1;
        if ( 4 * (v80 + 1) >= 3 * v15 )
        {
LABEL_33:
          v124 = v11;
          sub_2E515B0(v132, 2 * v15);
          v59 = *(_DWORD *)(a4 + 688);
          if ( !v59 )
            goto LABEL_189;
          v60 = v59 - 1;
          v61 = *(_QWORD *)(a4 + 672);
          v11 = v124;
          v62 = (v59 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
          v63 = *(_DWORD *)(a4 + 680) + 1;
          v46 = (__int64 *)(v61 + 16LL * v62);
          v64 = *v46;
          if ( *v46 != v58 )
          {
            v110 = 1;
            v111 = 0;
            while ( v64 != -4096 )
            {
              if ( v111 || v64 != -8192 )
                v46 = v111;
              v62 = v60 & (v110 + v62);
              v64 = *(_QWORD *)(v61 + 16LL * v62);
              if ( v58 == v64 )
              {
                v46 = (__int64 *)(v61 + 16LL * v62);
                goto LABEL_35;
              }
              ++v110;
              v111 = v46;
              v46 = (__int64 *)(v61 + 16LL * v62);
            }
            if ( v111 )
              v46 = v111;
          }
        }
        else if ( v15 - *(_DWORD *)(a4 + 684) - v63 <= v15 >> 3 )
        {
          v127 = v11;
          sub_2E515B0(v132, v15);
          v81 = *(_DWORD *)(a4 + 688);
          if ( !v81 )
          {
LABEL_189:
            ++*(_DWORD *)(a4 + 680);
            BUG();
          }
          v82 = v81 - 1;
          v83 = 0;
          v11 = v127;
          v84 = v82 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
          v85 = *(_QWORD *)(a4 + 672);
          v86 = 1;
          v63 = *(_DWORD *)(a4 + 680) + 1;
          v46 = (__int64 *)(v85 + 16LL * v84);
          v87 = *v46;
          if ( v58 != *v46 )
          {
            while ( v87 != -4096 )
            {
              if ( v87 == -8192 && !v83 )
                v83 = v46;
              v84 = v82 & (v86 + v84);
              v46 = (__int64 *)(v85 + 16LL * v84);
              v87 = *v46;
              if ( v58 == *v46 )
                goto LABEL_35;
              ++v86;
            }
            if ( v83 )
              v46 = v83;
          }
        }
LABEL_35:
        *(_DWORD *)(a4 + 680) = v63;
        if ( *v46 != -4096 )
          --*(_DWORD *)(a4 + 684);
        *v46 = v58;
        *((_DWORD *)v46 + 2) = 0;
        v15 = *(_DWORD *)(a4 + 688);
        v51 = **(_QWORD **)(v57 + 80);
        if ( !v15 )
        {
          ++*(_QWORD *)(a4 + 664);
          goto LABEL_39;
        }
        v44 = *(_QWORD *)(a4 + 672);
        v43 = v15 - 1;
        v50 = 0;
      }
      v52 = 1;
      v53 = 0;
      v54 = v43 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v55 = (__int64 *)(v44 + 16LL * v54);
      v56 = *v55;
      if ( *v55 == v51 )
      {
LABEL_30:
        if ( v50 >= *((_DWORD *)v55 + 2) )
          goto LABEL_44;
        continue;
      }
      break;
    }
    while ( v56 != -4096 )
    {
      if ( !v53 && v56 == -8192 )
        v53 = v55;
      v54 = v43 & (v52 + v54);
      v55 = (__int64 *)(v44 + 16LL * v54);
      v56 = *v55;
      if ( v51 == *v55 )
        goto LABEL_30;
      ++v52;
    }
    if ( !v53 )
      v53 = v55;
    v72 = *(_DWORD *)(a4 + 680);
    ++*(_QWORD *)(a4 + 664);
    v69 = v72 + 1;
    if ( 4 * v69 < 3 * v15 )
    {
      if ( v15 - (v69 + *(_DWORD *)(a4 + 684)) > v15 >> 3 )
        goto LABEL_41;
      v126 = v11;
      sub_2E515B0(v132, v15);
      v73 = *(_DWORD *)(a4 + 688);
      if ( v73 )
      {
        v74 = v73 - 1;
        v75 = 0;
        v11 = v126;
        v76 = v74 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
        v77 = *(_QWORD *)(a4 + 672);
        v78 = 1;
        v69 = *(_DWORD *)(a4 + 680) + 1;
        v53 = (__int64 *)(v77 + 16LL * v76);
        v79 = *v53;
        if ( v51 != *v53 )
        {
          while ( v79 != -4096 )
          {
            if ( v79 == -8192 && !v75 )
              v75 = v53;
            v76 = v74 & (v78 + v76);
            v53 = (__int64 *)(v77 + 16LL * v76);
            v79 = *v53;
            if ( v51 == *v53 )
              goto LABEL_41;
            ++v78;
          }
LABEL_59:
          if ( v75 )
            v53 = v75;
        }
        goto LABEL_41;
      }
LABEL_187:
      ++*(_DWORD *)(a4 + 680);
      BUG();
    }
LABEL_39:
    v125 = v11;
    sub_2E515B0(v132, 2 * v15);
    v65 = *(_DWORD *)(a4 + 688);
    if ( !v65 )
      goto LABEL_187;
    v66 = v65 - 1;
    v67 = *(_QWORD *)(a4 + 672);
    v11 = v125;
    v68 = v66 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v69 = *(_DWORD *)(a4 + 680) + 1;
    v53 = (__int64 *)(v67 + 16LL * v68);
    v70 = *v53;
    if ( *v53 != v51 )
    {
      v109 = 1;
      v75 = 0;
      while ( v70 != -4096 )
      {
        if ( v75 || v70 != -8192 )
          v53 = v75;
        v68 = v66 & (v109 + v68);
        v70 = *(_QWORD *)(v67 + 16LL * v68);
        if ( v51 == v70 )
        {
          v53 = (__int64 *)(v67 + 16LL * v68);
          goto LABEL_41;
        }
        ++v109;
        v75 = v53;
        v53 = (__int64 *)(v67 + 16LL * v68);
      }
      goto LABEL_59;
    }
LABEL_41:
    *(_DWORD *)(a4 + 680) = v69;
    if ( *v53 != -4096 )
      --*(_DWORD *)(a4 + 684);
    *v53 = v51;
    *((_DWORD *)v53 + 2) = 0;
LABEL_44:
    if ( i < v14 )
    {
      v71 = *i;
      *i = *v14;
      *v14 = v71;
      v15 = *(_DWORD *)(a4 + 688);
      goto LABEL_13;
    }
    v130 = v11;
    sub_37C17B0(i, v121, v120, a4);
    v5 = (__int64)v130;
    result = (char *)i - (char *)v130;
    if ( (char *)i - (char *)v130 > 128 )
    {
      if ( v120 )
      {
        v121 = i;
        continue;
      }
LABEL_119:
      v135 = (__int64 *)v5;
      v105 = result >> 3;
      v106 = ((result >> 3) - 2) >> 1;
      sub_37C1610(v5, v106, result >> 3, *(_QWORD *)(v5 + 8 * v106), a4);
      do
      {
        --v106;
        sub_37C1610((__int64)v135, v106, v105, v135[v106], a4);
      }
      while ( v106 );
      v107 = v131;
      do
      {
        v108 = *--v107;
        *v107 = *v135;
        result = sub_37C1610((__int64)v135, 0, v107 - v135, v108, a4);
      }
      while ( (char *)v107 - (char *)v135 > 8 );
    }
    return result;
  }
}
