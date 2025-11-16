// Function: sub_2620730
// Address: 0x2620730
//
__int64 __fastcall sub_2620730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  __int64 v6; // r11
  __int64 i; // r11
  unsigned int v9; // edi
  __int64 v10; // rcx
  __int64 *v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r9d
  unsigned int v14; // r13d
  __int64 *v15; // rdx
  __int64 v16; // r8
  __int64 *v17; // r12
  __int64 v18; // r15
  unsigned int v19; // esi
  __int64 v20; // r13
  __int64 v21; // r14
  int v22; // r8d
  int v23; // r8d
  __int64 v24; // rsi
  int v25; // edi
  unsigned int v26; // r9d
  __int64 *v27; // rdx
  __int64 v28; // rax
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  __int64 v32; // rcx
  int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rbx
  unsigned int v37; // ecx
  __int64 v38; // rdx
  unsigned int v39; // r9d
  __int64 *v40; // rdi
  __int64 v41; // r8
  unsigned int v42; // r8d
  int v43; // r14d
  __int64 *v44; // rdi
  unsigned int v45; // r12d
  __int64 *v46; // rax
  __int64 v47; // r9
  __int64 *v48; // rdx
  unsigned int v49; // esi
  __int64 *v50; // r13
  __int64 v51; // r12
  int v52; // esi
  int v53; // esi
  __int64 v54; // r8
  unsigned int v55; // ecx
  int v56; // edx
  __int64 *v57; // rax
  __int64 v58; // rdi
  int v59; // r9d
  __int64 *v60; // r14
  int v61; // eax
  int v62; // eax
  int v64; // ecx
  int v65; // eax
  int v66; // edx
  __int64 v67; // rsi
  unsigned int v68; // ebx
  __int64 v69; // rcx
  int v70; // r9d
  __int64 *v71; // r8
  int v72; // ecx
  int v73; // ecx
  int v74; // ecx
  __int64 v75; // rdi
  __int64 *v76; // r9
  __int64 v77; // r13
  int v78; // r12d
  __int64 v79; // rsi
  int v80; // ecx
  int v81; // r8d
  int v82; // r8d
  __int64 v83; // rsi
  unsigned int v84; // r9d
  __int64 v85; // rax
  __int64 *v86; // rcx
  int v87; // ecx
  int v88; // ecx
  __int64 v89; // rdi
  __int64 *v90; // r8
  unsigned int v91; // r14d
  int v92; // r9d
  __int64 v93; // rsi
  int v94; // eax
  int v95; // edx
  __int64 v96; // rsi
  int v97; // r9d
  unsigned int v98; // ebx
  __int64 v99; // rcx
  int v100; // r12d
  __int64 *v101; // r13
  int v102; // [rsp+8h] [rbp-78h]
  __int64 v103; // [rsp+8h] [rbp-78h]
  __int64 v104; // [rsp+10h] [rbp-70h]
  __int64 v105; // [rsp+10h] [rbp-70h]
  int v106; // [rsp+10h] [rbp-70h]
  __int64 v107; // [rsp+10h] [rbp-70h]
  __int64 v108; // [rsp+10h] [rbp-70h]
  unsigned int v110; // [rsp+20h] [rbp-60h]
  unsigned int v111; // [rsp+20h] [rbp-60h]
  __int64 v112; // [rsp+20h] [rbp-60h]
  __int64 v113; // [rsp+20h] [rbp-60h]
  __int64 v114; // [rsp+20h] [rbp-60h]
  unsigned int v115; // [rsp+20h] [rbp-60h]
  int v116; // [rsp+20h] [rbp-60h]
  int v117; // [rsp+20h] [rbp-60h]
  __int64 v118; // [rsp+28h] [rbp-58h]
  __int64 v119; // [rsp+28h] [rbp-58h]
  int v120; // [rsp+28h] [rbp-58h]
  __int64 v121; // [rsp+28h] [rbp-58h]
  __int64 v123; // [rsp+30h] [rbp-50h]
  __int64 v124; // [rsp+30h] [rbp-50h]
  __int64 *v126; // [rsp+40h] [rbp-40h]
  __int64 v127; // [rsp+40h] [rbp-40h]
  __int64 v128; // [rsp+40h] [rbp-40h]
  __int64 v129; // [rsp+40h] [rbp-40h]
  __int64 v130; // [rsp+40h] [rbp-40h]
  __int64 v131; // [rsp+48h] [rbp-38h]
  unsigned int v132; // [rsp+48h] [rbp-38h]

  v5 = a1;
  v6 = a5;
  v118 = (a3 - 1) / 2;
  if ( a2 < v118 )
  {
    for ( i = a2; ; i = v18 )
    {
      v19 = *(_DWORD *)(a5 + 24);
      v18 = 2 * (i + 1) - 1;
      v131 = 2 * (i + 1);
      v17 = (__int64 *)(v5 + 8 * v18);
      v126 = (__int64 *)(v5 + 16 * (i + 1));
      v20 = *v126;
      v21 = *v17;
      if ( v19 )
      {
        v9 = v19 - 1;
        v10 = *(_QWORD *)(a5 + 8);
        v110 = (v19 - 1) & (((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9));
        v11 = (__int64 *)(v10 + 40LL * v110);
        v12 = *v11;
        if ( v20 == *v11 )
        {
LABEL_4:
          v13 = *((_DWORD *)v11 + 2);
          goto LABEL_5;
        }
        v102 = 1;
        v27 = 0;
        while ( v12 != -4096 )
        {
          if ( !v27 && v12 == -8192 )
            v27 = v11;
          v110 = v9 & (v102 + v110);
          v11 = (__int64 *)(v10 + 40LL * v110);
          v12 = *v11;
          if ( v20 == *v11 )
            goto LABEL_4;
          ++v102;
        }
        v80 = *(_DWORD *)(a5 + 16);
        if ( !v27 )
          v27 = v11;
        ++*(_QWORD *)a5;
        v25 = v80 + 1;
        if ( 4 * (v80 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a5 + 20) - v25 > v19 >> 3 )
            goto LABEL_14;
          v103 = v5;
          v108 = i;
          v115 = ((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9);
          sub_261D190(a5, v19);
          v81 = *(_DWORD *)(a5 + 24);
          if ( !v81 )
          {
LABEL_180:
            ++*(_DWORD *)(a5 + 16);
            BUG();
          }
          v82 = v81 - 1;
          v83 = *(_QWORD *)(a5 + 8);
          i = v108;
          v5 = v103;
          v84 = v82 & v115;
          v25 = *(_DWORD *)(a5 + 16) + 1;
          v27 = (__int64 *)(v83 + 40LL * (v82 & v115));
          v85 = *v27;
          if ( v20 == *v27 )
            goto LABEL_14;
          v116 = 1;
          v86 = 0;
          while ( v85 != -4096 )
          {
            if ( !v86 && v85 == -8192 )
              v86 = v27;
            v84 = v82 & (v116 + v84);
            v27 = (__int64 *)(v83 + 40LL * v84);
            v85 = *v27;
            if ( v20 == *v27 )
              goto LABEL_14;
            ++v116;
          }
          goto LABEL_98;
        }
      }
      else
      {
        ++*(_QWORD *)a5;
      }
      v104 = v5;
      v112 = i;
      sub_261D190(a5, 2 * v19);
      v22 = *(_DWORD *)(a5 + 24);
      if ( !v22 )
        goto LABEL_180;
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a5 + 8);
      i = v112;
      v5 = v104;
      v25 = *(_DWORD *)(a5 + 16) + 1;
      v26 = v23 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v27 = (__int64 *)(v24 + 40LL * v26);
      v28 = *v27;
      if ( v20 == *v27 )
        goto LABEL_14;
      v117 = 1;
      v86 = 0;
      while ( v28 != -4096 )
      {
        if ( !v86 && v28 == -8192 )
          v86 = v27;
        v26 = v23 & (v117 + v26);
        v27 = (__int64 *)(v24 + 40LL * v26);
        v28 = *v27;
        if ( v20 == *v27 )
          goto LABEL_14;
        ++v117;
      }
LABEL_98:
      if ( v86 )
        v27 = v86;
LABEL_14:
      *(_DWORD *)(a5 + 16) = v25;
      if ( *v27 != -4096 )
        --*(_DWORD *)(a5 + 20);
      *v27 = v20;
      *(_OWORD *)(v27 + 1) = 0;
      *(_OWORD *)(v27 + 3) = 0;
      v19 = *(_DWORD *)(a5 + 24);
      if ( !v19 )
      {
        ++*(_QWORD *)a5;
        goto LABEL_18;
      }
      v10 = *(_QWORD *)(a5 + 8);
      v9 = v19 - 1;
      v13 = 0;
LABEL_5:
      v14 = ((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4);
      v111 = v14 & v9;
      v15 = (__int64 *)(v10 + 40LL * (v14 & v9));
      v16 = *v15;
      if ( v21 != *v15 )
      {
        v106 = 1;
        v34 = 0;
        while ( v16 != -4096 )
        {
          if ( !v34 && v16 == -8192 )
            v34 = v15;
          v111 = v9 & (v106 + v111);
          v15 = (__int64 *)(v10 + 40LL * v111);
          v16 = *v15;
          if ( v21 == *v15 )
            goto LABEL_6;
          ++v106;
        }
        v72 = *(_DWORD *)(a5 + 16);
        if ( !v34 )
          v34 = v15;
        ++*(_QWORD *)a5;
        v33 = v72 + 1;
        if ( 4 * (v72 + 1) >= 3 * v19 )
        {
LABEL_18:
          v105 = v5;
          v113 = i;
          sub_261D190(a5, 2 * v19);
          v29 = *(_DWORD *)(a5 + 24);
          if ( !v29 )
            goto LABEL_181;
          v30 = v29 - 1;
          v31 = *(_QWORD *)(a5 + 8);
          i = v113;
          v5 = v105;
          LODWORD(v32) = v30 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v33 = *(_DWORD *)(a5 + 16) + 1;
          v34 = (__int64 *)(v31 + 40LL * (unsigned int)v32);
          v35 = *v34;
          if ( v21 != *v34 )
          {
            v100 = 1;
            v101 = 0;
            while ( v35 != -4096 )
            {
              if ( !v101 && v35 == -8192 )
                v101 = v34;
              v32 = v30 & (unsigned int)(v32 + v100);
              v34 = (__int64 *)(v31 + 40 * v32);
              v35 = *v34;
              if ( v21 == *v34 )
                goto LABEL_20;
              ++v100;
            }
            if ( v101 )
              v34 = v101;
          }
        }
        else if ( v19 - (v33 + *(_DWORD *)(a5 + 20)) <= v19 >> 3 )
        {
          v107 = v5;
          v114 = i;
          sub_261D190(a5, v19);
          v73 = *(_DWORD *)(a5 + 24);
          if ( !v73 )
          {
LABEL_181:
            ++*(_DWORD *)(a5 + 16);
            BUG();
          }
          v74 = v73 - 1;
          v75 = *(_QWORD *)(a5 + 8);
          v76 = 0;
          LODWORD(v77) = v74 & v14;
          i = v114;
          v5 = v107;
          v78 = 1;
          v33 = *(_DWORD *)(a5 + 16) + 1;
          v34 = (__int64 *)(v75 + 40LL * (unsigned int)v77);
          v79 = *v34;
          if ( v21 != *v34 )
          {
            while ( v79 != -4096 )
            {
              if ( !v76 && v79 == -8192 )
                v76 = v34;
              v77 = v74 & (unsigned int)(v77 + v78);
              v34 = (__int64 *)(v75 + 40 * v77);
              v79 = *v34;
              if ( v21 == *v34 )
                goto LABEL_20;
              ++v78;
            }
            if ( v76 )
              v34 = v76;
          }
        }
LABEL_20:
        *(_DWORD *)(a5 + 16) = v33;
        if ( *v34 != -4096 )
          --*(_DWORD *)(a5 + 20);
        v18 = v131;
        *v34 = v21;
        *(_OWORD *)(v34 + 1) = 0;
        *(_OWORD *)(v34 + 3) = 0;
        *(_QWORD *)(v5 + 8 * i) = *v126;
        if ( v131 >= v118 )
          goto LABEL_23;
        continue;
      }
LABEL_6:
      if ( *((_DWORD *)v15 + 2) <= v13 )
      {
        v17 = v126;
        v18 = v131;
      }
      *(_QWORD *)(v5 + 8 * i) = *v17;
      if ( v18 >= v118 )
      {
LABEL_23:
        v6 = a5;
        if ( (a3 & 1) != 0 )
          goto LABEL_24;
        goto LABEL_42;
      }
    }
  }
  if ( (a3 & 1) != 0 )
  {
    v50 = (__int64 *)(a1 + 8 * a2);
    goto LABEL_57;
  }
  v18 = a2;
LABEL_42:
  if ( (a3 - 2) / 2 == v18 )
  {
    *(_QWORD *)(v5 + 8 * v18) = *(_QWORD *)(v5 + 8 * (2 * v18 + 1));
    v18 = 2 * v18 + 1;
  }
LABEL_24:
  v36 = (v18 - 1) / 2;
  if ( v18 <= a2 )
  {
    v50 = (__int64 *)(v5 + 8 * v18);
    goto LABEL_57;
  }
  v132 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  while ( 1 )
  {
    v49 = *(_DWORD *)(v6 + 24);
    v50 = (__int64 *)(v5 + 8 * v36);
    v51 = *v50;
    if ( v49 )
    {
      v37 = v49 - 1;
      v38 = *(_QWORD *)(v6 + 8);
      v39 = (v49 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v40 = (__int64 *)(v38 + 40LL * v39);
      v41 = *v40;
      if ( v51 == *v40 )
      {
LABEL_27:
        v42 = *((_DWORD *)v40 + 2);
        goto LABEL_28;
      }
      v120 = 1;
      v57 = 0;
      while ( v41 != -4096 )
      {
        if ( !v57 && v41 == -8192 )
          v57 = v40;
        v39 = v37 & (v120 + v39);
        v40 = (__int64 *)(v38 + 40LL * v39);
        v41 = *v40;
        if ( v51 == *v40 )
          goto LABEL_27;
        ++v120;
      }
      v64 = *(_DWORD *)(v6 + 16);
      if ( !v57 )
        v57 = v40;
      ++*(_QWORD *)v6;
      v56 = v64 + 1;
      if ( 4 * (v64 + 1) < 3 * v49 )
      {
        if ( v49 - *(_DWORD *)(v6 + 20) - v56 <= v49 >> 3 )
        {
          v129 = v6;
          v121 = v5;
          sub_261D190(v6, v49);
          v6 = v129;
          v87 = *(_DWORD *)(v129 + 24);
          if ( !v87 )
          {
LABEL_183:
            ++*(_DWORD *)(v6 + 16);
            BUG();
          }
          v88 = v87 - 1;
          v89 = *(_QWORD *)(v129 + 8);
          v90 = 0;
          v91 = v88 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
          v5 = v121;
          v92 = 1;
          v56 = *(_DWORD *)(v129 + 16) + 1;
          v57 = (__int64 *)(v89 + 40LL * v91);
          v93 = *v57;
          if ( *v57 != v51 )
          {
            while ( v93 != -4096 )
            {
              if ( v90 || v93 != -8192 )
                v57 = v90;
              v91 = v88 & (v92 + v91);
              v93 = *(_QWORD *)(v89 + 40LL * v91);
              if ( v51 == v93 )
              {
                v57 = (__int64 *)(v89 + 40LL * v91);
                goto LABEL_64;
              }
              ++v92;
              v90 = v57;
              v57 = (__int64 *)(v89 + 40LL * v91);
            }
            if ( v90 )
              v57 = v90;
          }
        }
        goto LABEL_64;
      }
    }
    else
    {
      ++*(_QWORD *)v6;
    }
    v127 = v6;
    v119 = v5;
    sub_261D190(v6, 2 * v49);
    v6 = v127;
    v52 = *(_DWORD *)(v127 + 24);
    if ( !v52 )
      goto LABEL_183;
    v53 = v52 - 1;
    v54 = *(_QWORD *)(v127 + 8);
    v5 = v119;
    v55 = v53 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v56 = *(_DWORD *)(v127 + 16) + 1;
    v57 = (__int64 *)(v54 + 40LL * v55);
    v58 = *v57;
    if ( v51 != *v57 )
    {
      v59 = 1;
      v60 = 0;
      while ( v58 != -4096 )
      {
        if ( v58 == -8192 && !v60 )
          v60 = v57;
        v55 = v53 & (v59 + v55);
        v57 = (__int64 *)(v54 + 40LL * v55);
        v58 = *v57;
        if ( v51 == *v57 )
          goto LABEL_64;
        ++v59;
      }
      if ( v60 )
        v57 = v60;
    }
LABEL_64:
    *(_DWORD *)(v6 + 16) = v56;
    if ( *v57 != -4096 )
      --*(_DWORD *)(v6 + 20);
    *v57 = v51;
    *(_OWORD *)(v57 + 1) = 0;
    *(_OWORD *)(v57 + 3) = 0;
    v49 = *(_DWORD *)(v6 + 24);
    if ( !v49 )
    {
      ++*(_QWORD *)v6;
      goto LABEL_68;
    }
    v38 = *(_QWORD *)(v6 + 8);
    v37 = v49 - 1;
    v42 = 0;
LABEL_28:
    v43 = 1;
    v44 = 0;
    v45 = v37 & v132;
    v46 = (__int64 *)(v38 + 40LL * (v37 & v132));
    v47 = *v46;
    if ( a4 != *v46 )
      break;
LABEL_29:
    v48 = (__int64 *)(v5 + 8 * v18);
    if ( v42 >= *((_DWORD *)v46 + 2) )
    {
      v50 = (__int64 *)(v5 + 8 * v18);
      goto LABEL_57;
    }
    v18 = v36;
    *v48 = *v50;
    if ( a2 >= v36 )
      goto LABEL_57;
    v36 = (v36 - 1) / 2;
  }
  while ( v47 != -4096 )
  {
    if ( v47 == -8192 && !v44 )
      v44 = v46;
    v45 = v37 & (v43 + v45);
    v46 = (__int64 *)(v38 + 40LL * v45);
    v47 = *v46;
    if ( a4 == *v46 )
      goto LABEL_29;
    ++v43;
  }
  if ( !v44 )
    v44 = v46;
  v61 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)v6;
  v62 = v61 + 1;
  if ( 4 * v62 < 3 * v49 )
  {
    if ( v49 - (v62 + *(_DWORD *)(v6 + 20)) > v49 >> 3 )
      goto LABEL_54;
    v130 = v6;
    v124 = v5;
    sub_261D190(v6, v49);
    v6 = v130;
    v94 = *(_DWORD *)(v130 + 24);
    if ( v94 )
    {
      v95 = v94 - 1;
      v96 = *(_QWORD *)(v130 + 8);
      v71 = 0;
      v5 = v124;
      v97 = 1;
      v98 = (v94 - 1) & v132;
      v44 = (__int64 *)(v96 + 40LL * v98);
      v99 = *v44;
      v62 = *(_DWORD *)(v130 + 16) + 1;
      if ( a4 != *v44 )
      {
        while ( v99 != -4096 )
        {
          if ( !v71 && v99 == -8192 )
            v71 = v44;
          v98 = v95 & (v97 + v98);
          v44 = (__int64 *)(v96 + 40LL * v98);
          v99 = *v44;
          if ( a4 == *v44 )
            goto LABEL_54;
          ++v97;
        }
LABEL_72:
        if ( v71 )
          v44 = v71;
      }
      goto LABEL_54;
    }
LABEL_182:
    ++*(_DWORD *)(v6 + 16);
    BUG();
  }
LABEL_68:
  v128 = v6;
  v123 = v5;
  sub_261D190(v6, 2 * v49);
  v6 = v128;
  v65 = *(_DWORD *)(v128 + 24);
  if ( !v65 )
    goto LABEL_182;
  v66 = v65 - 1;
  v67 = *(_QWORD *)(v128 + 8);
  v5 = v123;
  v68 = (v65 - 1) & v132;
  v44 = (__int64 *)(v67 + 40LL * v68);
  v69 = *v44;
  v62 = *(_DWORD *)(v128 + 16) + 1;
  if ( a4 != *v44 )
  {
    v70 = 1;
    v71 = 0;
    while ( v69 != -4096 )
    {
      if ( v69 == -8192 && !v71 )
        v71 = v44;
      v68 = v66 & (v70 + v68);
      v44 = (__int64 *)(v67 + 40LL * v68);
      v69 = *v44;
      if ( a4 == *v44 )
        goto LABEL_54;
      ++v70;
    }
    goto LABEL_72;
  }
LABEL_54:
  *(_DWORD *)(v6 + 16) = v62;
  if ( *v44 != -4096 )
    --*(_DWORD *)(v6 + 20);
  v50 = (__int64 *)(v5 + 8 * v18);
  *(_OWORD *)(v44 + 1) = 0;
  *v44 = a4;
  *(_OWORD *)(v44 + 3) = 0;
LABEL_57:
  *v50 = a4;
  return a4;
}
