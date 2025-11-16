// Function: sub_37C29D0
// Address: 0x37c29d0
//
__int64 __fastcall sub_37C29D0(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // r15
  __int64 v6; // r11
  __int64 *v8; // r11
  __int64 *v9; // r12
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // rdx
  unsigned int v13; // r13d
  __int64 *v14; // rax
  __int64 *v15; // r11
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 *v19; // r14
  unsigned int v20; // esi
  unsigned int v21; // edi
  __int64 v22; // rcx
  int v23; // r10d
  __int64 *v24; // rdx
  unsigned int v25; // r9d
  __int64 *v26; // rax
  __int64 v27; // r8
  unsigned int v28; // r10d
  int v29; // r15d
  __int64 *v30; // rdx
  unsigned int v31; // r12d
  unsigned int v32; // r9d
  __int64 *v33; // rax
  __int64 v34; // r8
  int v35; // r9d
  int v36; // r9d
  __int64 v37; // rdi
  unsigned int v38; // ecx
  int v39; // eax
  __int64 v40; // rsi
  int v41; // esi
  int v42; // esi
  __int64 v43; // r8
  __int64 v44; // rcx
  int v45; // eax
  __int64 v46; // rdi
  __int64 v47; // r12
  __int64 v48; // r13
  unsigned int v49; // ecx
  __int64 v50; // rdi
  int v51; // r10d
  __int64 *v52; // rdx
  unsigned int v53; // r9d
  __int64 *v54; // rax
  __int64 v55; // r8
  unsigned int v56; // r8d
  int v57; // r15d
  __int64 *v58; // rdx
  unsigned int v59; // r13d
  unsigned int v60; // r10d
  __int64 *v61; // rax
  __int64 v62; // r9
  int v63; // eax
  int v64; // edi
  __int64 v65; // rsi
  unsigned int v66; // eax
  int v67; // ecx
  __int64 v68; // r8
  int v69; // esi
  int v70; // esi
  __int64 v71; // r8
  unsigned int v72; // ecx
  int v73; // eax
  __int64 v74; // rdi
  __int64 v75; // rax
  int v76; // eax
  int v77; // ecx
  int v78; // ecx
  __int64 *v79; // r9
  unsigned int v80; // r13d
  __int64 v81; // rdi
  int v82; // r10d
  __int64 v83; // rsi
  int v84; // eax
  int v85; // eax
  int v86; // eax
  __int64 *v87; // r8
  unsigned int v88; // r15d
  __int64 v89; // rdi
  int v90; // r9d
  __int64 v91; // rsi
  int v92; // eax
  int v93; // ecx
  int v94; // ecx
  __int64 *v95; // r9
  __int64 v96; // r12
  __int64 v97; // rdi
  int v98; // r10d
  __int64 v99; // rsi
  int v100; // eax
  int v101; // r8d
  int v102; // r8d
  __int64 *v103; // r9
  unsigned int v104; // r15d
  __int64 v105; // rsi
  int v106; // r10d
  __int64 v107; // rcx
  __int64 v108; // rdx
  unsigned int v109; // r13d
  __int64 *v110; // rax
  __int64 v111; // rbx
  __int64 v112; // r12
  __int64 *v113; // rbx
  __int64 v114; // rcx
  int v115; // r10d
  int v116; // r15d
  __int64 *v117; // r9
  __int64 *v118; // r14
  __int64 v119; // rax
  unsigned int v120; // r12d
  __int64 *v121; // rax
  __int64 v122; // rax
  unsigned int v123; // r13d
  __int64 *v124; // rax
  bool v125; // cf
  int v126; // r12d
  __int64 *v127; // r10
  int v128; // r15d
  __int64 *v129; // r10
  __int64 *v130; // [rsp+8h] [rbp-88h]
  __int64 v131; // [rsp+20h] [rbp-70h]
  __int64 *v132; // [rsp+28h] [rbp-68h]
  __int64 *v133; // [rsp+30h] [rbp-60h]
  __int64 *v134; // [rsp+30h] [rbp-60h]
  __int64 *v135; // [rsp+30h] [rbp-60h]
  __int64 *v136; // [rsp+30h] [rbp-60h]
  __int64 *v137; // [rsp+30h] [rbp-60h]
  __int64 *v138; // [rsp+30h] [rbp-60h]
  __int64 *v139; // [rsp+30h] [rbp-60h]
  __int64 *v140; // [rsp+30h] [rbp-60h]
  __int64 *v141; // [rsp+30h] [rbp-60h]
  __int64 *v142; // [rsp+38h] [rbp-58h]
  __int64 v143; // [rsp+40h] [rbp-50h]
  __int64 *v144; // [rsp+48h] [rbp-48h]
  __int64 *i; // [rsp+48h] [rbp-48h]
  __int64 v146; // [rsp+48h] [rbp-48h]
  __int64 v147; // [rsp+50h] [rbp-40h] BYREF
  __int64 v148[7]; // [rsp+58h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v132 = (__int64 *)a2;
  v131 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  v5 = a1;
  v6 = a4;
  if ( !a3 )
  {
    v142 = (__int64 *)a2;
    goto LABEL_118;
  }
  v8 = a1;
  v130 = a1 + 1;
  v143 = a4 + 664;
  while ( 2 )
  {
    v144 = v8;
    v9 = &v8[result >> 4];
    --v131;
    v10 = *v9;
    v147 = v8[1];
    v148[0] = v10;
    v11 = *(_DWORD *)sub_2E51790(v143, &v147);
    if ( v11 >= *(_DWORD *)sub_2E51790(v143, v148) )
    {
      v108 = v144[1];
      v148[0] = *(v132 - 1);
      v147 = v108;
      v109 = *(_DWORD *)sub_2E51790(v143, &v147);
      v110 = sub_2E51790(v143, v148);
      v15 = v144;
      if ( v109 >= *(_DWORD *)v110 )
      {
        v118 = v132;
        v122 = *(v132 - 1);
        v147 = *v9;
        v148[0] = v122;
        v123 = *(_DWORD *)sub_2E51790(v143, &v147);
        v124 = sub_2E51790(v143, v148);
        v15 = v144;
        v125 = v123 < *(_DWORD *)v124;
        v16 = *v144;
        if ( !v125 )
          goto LABEL_7;
LABEL_136:
        *v15 = *(v118 - 1);
        *(v118 - 1) = v16;
        v17 = *v15;
        v18 = v15[1];
        goto LABEL_8;
      }
LABEL_116:
      v18 = *v15;
      v17 = v15[1];
      v15[1] = *v15;
      *v15 = v17;
      goto LABEL_8;
    }
    v12 = *v9;
    v148[0] = *(v132 - 1);
    v147 = v12;
    v13 = *(_DWORD *)sub_2E51790(v143, &v147);
    v14 = sub_2E51790(v143, v148);
    v15 = v144;
    if ( v13 >= *(_DWORD *)v14 )
    {
      v118 = v132;
      v119 = *(v132 - 1);
      v147 = v144[1];
      v148[0] = v119;
      v120 = *(_DWORD *)sub_2E51790(v143, &v147);
      v121 = sub_2E51790(v143, v148);
      v15 = v144;
      if ( v120 < *(_DWORD *)v121 )
      {
        v16 = *v144;
        goto LABEL_136;
      }
      goto LABEL_116;
    }
    v16 = *v144;
LABEL_7:
    *v15 = *v9;
    *v9 = v16;
    v17 = *v15;
    v18 = v15[1];
LABEL_8:
    v19 = v132;
    v20 = *(_DWORD *)(a4 + 688);
    for ( i = v130; ; ++i )
    {
      v142 = i;
      if ( v20 )
      {
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a4 + 672);
        v23 = 1;
        v24 = 0;
        v25 = (v20 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v26 = (__int64 *)(v22 + 16LL * v25);
        v27 = *v26;
        if ( v18 == *v26 )
        {
LABEL_10:
          v28 = *((_DWORD *)v26 + 2);
          goto LABEL_11;
        }
        while ( v27 != -4096 )
        {
          if ( !v24 && v27 == -8192 )
            v24 = v26;
          v25 = v21 & (v23 + v25);
          v26 = (__int64 *)(v22 + 16LL * v25);
          v27 = *v26;
          if ( v18 == *v26 )
            goto LABEL_10;
          ++v23;
        }
        if ( !v24 )
          v24 = v26;
        v100 = *(_DWORD *)(a4 + 680);
        ++*(_QWORD *)(a4 + 664);
        v39 = v100 + 1;
        if ( 4 * v39 < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a4 + 684) - v39 <= v20 >> 3 )
          {
            v140 = v15;
            sub_2E515B0(v143, v20);
            v101 = *(_DWORD *)(a4 + 688);
            if ( !v101 )
            {
LABEL_189:
              ++*(_DWORD *)(a4 + 680);
              BUG();
            }
            v102 = v101 - 1;
            v103 = 0;
            v15 = v140;
            v104 = v102 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v105 = *(_QWORD *)(a4 + 672);
            v106 = 1;
            v39 = *(_DWORD *)(a4 + 680) + 1;
            v24 = (__int64 *)(v105 + 16LL * v104);
            v107 = *v24;
            if ( v18 != *v24 )
            {
              while ( v107 != -4096 )
              {
                if ( v107 == -8192 && !v103 )
                  v103 = v24;
                v104 = v102 & (v106 + v104);
                v24 = (__int64 *)(v105 + 16LL * v104);
                v107 = *v24;
                if ( v18 == *v24 )
                  goto LABEL_18;
                ++v106;
              }
              if ( v103 )
                v24 = v103;
            }
          }
          goto LABEL_18;
        }
      }
      else
      {
        ++*(_QWORD *)(a4 + 664);
      }
      v133 = v15;
      sub_2E515B0(v143, 2 * v20);
      v35 = *(_DWORD *)(a4 + 688);
      if ( !v35 )
        goto LABEL_189;
      v36 = v35 - 1;
      v15 = v133;
      v37 = *(_QWORD *)(a4 + 672);
      v38 = v36 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v39 = *(_DWORD *)(a4 + 680) + 1;
      v24 = (__int64 *)(v37 + 16LL * v38);
      v40 = *v24;
      if ( v18 != *v24 )
      {
        v128 = 1;
        v129 = 0;
        while ( v40 != -4096 )
        {
          if ( !v129 && v40 == -8192 )
            v129 = v24;
          v38 = v36 & (v128 + v38);
          v24 = (__int64 *)(v37 + 16LL * v38);
          v40 = *v24;
          if ( v18 == *v24 )
            goto LABEL_18;
          ++v128;
        }
        if ( v129 )
          v24 = v129;
      }
LABEL_18:
      *(_DWORD *)(a4 + 680) = v39;
      if ( *v24 != -4096 )
        --*(_DWORD *)(a4 + 684);
      *v24 = v18;
      *((_DWORD *)v24 + 2) = 0;
      v20 = *(_DWORD *)(a4 + 688);
      if ( !v20 )
      {
        ++*(_QWORD *)(a4 + 664);
LABEL_22:
        v134 = v15;
        sub_2E515B0(v143, 2 * v20);
        v41 = *(_DWORD *)(a4 + 688);
        if ( !v41 )
          goto LABEL_191;
        v42 = v41 - 1;
        v43 = *(_QWORD *)(a4 + 672);
        v15 = v134;
        LODWORD(v44) = v42 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v45 = *(_DWORD *)(a4 + 680) + 1;
        v30 = (__int64 *)(v43 + 16LL * (unsigned int)v44);
        v46 = *v30;
        if ( *v30 != v17 )
        {
          v126 = 1;
          v127 = 0;
          while ( v46 != -4096 )
          {
            if ( !v127 && v46 == -8192 )
              v127 = v30;
            v44 = v42 & (unsigned int)(v44 + v126);
            v30 = (__int64 *)(v43 + 16 * v44);
            v46 = *v30;
            if ( *v30 == v17 )
              goto LABEL_24;
            ++v126;
          }
          if ( v127 )
            v30 = v127;
        }
        goto LABEL_24;
      }
      v22 = *(_QWORD *)(a4 + 672);
      v21 = v20 - 1;
      v28 = 0;
LABEL_11:
      v29 = 1;
      v30 = 0;
      v31 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
      v32 = v31 & v21;
      v33 = (__int64 *)(v22 + 16LL * (v31 & v21));
      v34 = *v33;
      if ( *v33 != v17 )
        break;
LABEL_12:
      if ( v28 >= *((_DWORD *)v33 + 2) )
        goto LABEL_31;
LABEL_13:
      v17 = *v15;
      v18 = i[1];
    }
    while ( v34 != -4096 )
    {
      if ( !v30 && v34 == -8192 )
        v30 = v33;
      v32 = v21 & (v29 + v32);
      v33 = (__int64 *)(v22 + 16LL * v32);
      v34 = *v33;
      if ( *v33 == v17 )
        goto LABEL_12;
      ++v29;
    }
    if ( !v30 )
      v30 = v33;
    v92 = *(_DWORD *)(a4 + 680);
    ++*(_QWORD *)(a4 + 664);
    v45 = v92 + 1;
    if ( 4 * v45 >= 3 * v20 )
      goto LABEL_22;
    if ( v20 - (v45 + *(_DWORD *)(a4 + 684)) <= v20 >> 3 )
    {
      v139 = v15;
      sub_2E515B0(v143, v20);
      v93 = *(_DWORD *)(a4 + 688);
      if ( !v93 )
      {
LABEL_191:
        ++*(_DWORD *)(a4 + 680);
        BUG();
      }
      v94 = v93 - 1;
      v95 = 0;
      v15 = v139;
      LODWORD(v96) = v94 & v31;
      v97 = *(_QWORD *)(a4 + 672);
      v98 = 1;
      v45 = *(_DWORD *)(a4 + 680) + 1;
      v30 = (__int64 *)(v97 + 16LL * (unsigned int)v96);
      v99 = *v30;
      if ( *v30 != v17 )
      {
        while ( v99 != -4096 )
        {
          if ( v99 == -8192 && !v95 )
            v95 = v30;
          v96 = v94 & (unsigned int)(v96 + v98);
          v30 = (__int64 *)(v97 + 16 * v96);
          v99 = *v30;
          if ( *v30 == v17 )
            goto LABEL_24;
          ++v98;
        }
        if ( v95 )
          v30 = v95;
      }
    }
LABEL_24:
    *(_DWORD *)(a4 + 680) = v45;
    if ( *v30 != -4096 )
      --*(_DWORD *)(a4 + 684);
    *v30 = v17;
    --v19;
    *((_DWORD *)v30 + 2) = 0;
    v20 = *(_DWORD *)(a4 + 688);
    v47 = *v19;
    v48 = *v15;
    if ( !v20 )
    {
      ++*(_QWORD *)(a4 + 664);
      goto LABEL_33;
    }
    while ( 1 )
    {
      v49 = v20 - 1;
      v50 = *(_QWORD *)(a4 + 672);
      v51 = 1;
      v52 = 0;
      v53 = (v20 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v54 = (__int64 *)(v50 + 16LL * v53);
      v55 = *v54;
      if ( v48 == *v54 )
      {
LABEL_28:
        v56 = *((_DWORD *)v54 + 2);
      }
      else
      {
        while ( v55 != -4096 )
        {
          if ( !v52 && v55 == -8192 )
            v52 = v54;
          v53 = v49 & (v51 + v53);
          v54 = (__int64 *)(v50 + 16LL * v53);
          v55 = *v54;
          if ( v48 == *v54 )
            goto LABEL_28;
          ++v51;
        }
        if ( !v52 )
          v52 = v54;
        v84 = *(_DWORD *)(a4 + 680);
        ++*(_QWORD *)(a4 + 664);
        v67 = v84 + 1;
        if ( 4 * (v84 + 1) >= 3 * v20 )
        {
LABEL_33:
          v135 = v15;
          sub_2E515B0(v143, 2 * v20);
          v63 = *(_DWORD *)(a4 + 688);
          if ( !v63 )
            goto LABEL_188;
          v64 = v63 - 1;
          v65 = *(_QWORD *)(a4 + 672);
          v15 = v135;
          v66 = (v63 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v67 = *(_DWORD *)(a4 + 680) + 1;
          v52 = (__int64 *)(v65 + 16LL * v66);
          v68 = *v52;
          if ( v48 != *v52 )
          {
            v116 = 1;
            v117 = 0;
            while ( v68 != -4096 )
            {
              if ( v117 || v68 != -8192 )
                v52 = v117;
              v66 = v64 & (v116 + v66);
              v68 = *(_QWORD *)(v65 + 16LL * v66);
              if ( v48 == v68 )
              {
                v52 = (__int64 *)(v65 + 16LL * v66);
                goto LABEL_35;
              }
              ++v116;
              v117 = v52;
              v52 = (__int64 *)(v65 + 16LL * v66);
            }
            if ( v117 )
              v52 = v117;
          }
        }
        else if ( v20 - *(_DWORD *)(a4 + 684) - v67 <= v20 >> 3 )
        {
          v138 = v15;
          sub_2E515B0(v143, v20);
          v85 = *(_DWORD *)(a4 + 688);
          if ( !v85 )
          {
LABEL_188:
            ++*(_DWORD *)(a4 + 680);
            BUG();
          }
          v86 = v85 - 1;
          v87 = 0;
          v15 = v138;
          v88 = v86 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v89 = *(_QWORD *)(a4 + 672);
          v90 = 1;
          v67 = *(_DWORD *)(a4 + 680) + 1;
          v52 = (__int64 *)(v89 + 16LL * v88);
          v91 = *v52;
          if ( v48 != *v52 )
          {
            while ( v91 != -4096 )
            {
              if ( !v87 && v91 == -8192 )
                v87 = v52;
              v88 = v86 & (v90 + v88);
              v52 = (__int64 *)(v89 + 16LL * v88);
              v91 = *v52;
              if ( v48 == *v52 )
                goto LABEL_35;
              ++v90;
            }
            if ( v87 )
              v52 = v87;
          }
        }
LABEL_35:
        *(_DWORD *)(a4 + 680) = v67;
        if ( *v52 != -4096 )
          --*(_DWORD *)(a4 + 684);
        *v52 = v48;
        *((_DWORD *)v52 + 2) = 0;
        v20 = *(_DWORD *)(a4 + 688);
        if ( !v20 )
        {
          ++*(_QWORD *)(a4 + 664);
          goto LABEL_39;
        }
        v50 = *(_QWORD *)(a4 + 672);
        v49 = v20 - 1;
        v56 = 0;
      }
      v57 = 1;
      v58 = 0;
      v59 = ((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4);
      v60 = v59 & v49;
      v61 = (__int64 *)(v50 + 16LL * (v59 & v49));
      v62 = *v61;
      if ( v47 != *v61 )
        break;
LABEL_30:
      if ( *((_DWORD *)v61 + 2) <= v56 )
        goto LABEL_44;
LABEL_31:
      --v19;
      v48 = *v15;
      v47 = *v19;
    }
    while ( v62 != -4096 )
    {
      if ( !v58 && v62 == -8192 )
        v58 = v61;
      v60 = v49 & (v57 + v60);
      v61 = (__int64 *)(v50 + 16LL * v60);
      v62 = *v61;
      if ( v47 == *v61 )
        goto LABEL_30;
      ++v57;
    }
    if ( !v58 )
      v58 = v61;
    v76 = *(_DWORD *)(a4 + 680);
    ++*(_QWORD *)(a4 + 664);
    v73 = v76 + 1;
    if ( 4 * v73 >= 3 * v20 )
    {
LABEL_39:
      v136 = v15;
      sub_2E515B0(v143, 2 * v20);
      v69 = *(_DWORD *)(a4 + 688);
      if ( !v69 )
        goto LABEL_190;
      v70 = v69 - 1;
      v71 = *(_QWORD *)(a4 + 672);
      v15 = v136;
      v72 = v70 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v73 = *(_DWORD *)(a4 + 680) + 1;
      v58 = (__int64 *)(v71 + 16LL * v72);
      v74 = *v58;
      if ( v47 != *v58 )
      {
        v115 = 1;
        v79 = 0;
        while ( v74 != -4096 )
        {
          if ( v79 || v74 != -8192 )
            v58 = v79;
          v72 = v70 & (v115 + v72);
          v74 = *(_QWORD *)(v71 + 16LL * v72);
          if ( v47 == v74 )
          {
            v58 = (__int64 *)(v71 + 16LL * v72);
            goto LABEL_41;
          }
          ++v115;
          v79 = v58;
          v58 = (__int64 *)(v71 + 16LL * v72);
        }
        goto LABEL_59;
      }
    }
    else if ( v20 - (v73 + *(_DWORD *)(a4 + 684)) <= v20 >> 3 )
    {
      v137 = v15;
      sub_2E515B0(v143, v20);
      v77 = *(_DWORD *)(a4 + 688);
      if ( !v77 )
      {
LABEL_190:
        ++*(_DWORD *)(a4 + 680);
        BUG();
      }
      v78 = v77 - 1;
      v79 = 0;
      v15 = v137;
      v80 = v78 & v59;
      v81 = *(_QWORD *)(a4 + 672);
      v82 = 1;
      v73 = *(_DWORD *)(a4 + 680) + 1;
      v58 = (__int64 *)(v81 + 16LL * v80);
      v83 = *v58;
      if ( v47 != *v58 )
      {
        while ( v83 != -4096 )
        {
          if ( v83 == -8192 && !v79 )
            v79 = v58;
          v80 = v78 & (v82 + v80);
          v58 = (__int64 *)(v81 + 16LL * v80);
          v83 = *v58;
          if ( v47 == *v58 )
            goto LABEL_41;
          ++v82;
        }
LABEL_59:
        if ( v79 )
          v58 = v79;
      }
    }
LABEL_41:
    *(_DWORD *)(a4 + 680) = v73;
    if ( *v58 != -4096 )
      --*(_DWORD *)(a4 + 684);
    *v58 = v47;
    *((_DWORD *)v58 + 2) = 0;
LABEL_44:
    if ( i < v19 )
    {
      v75 = *i;
      *i = *v19;
      *v19 = v75;
      v20 = *(_DWORD *)(a4 + 688);
      goto LABEL_13;
    }
    v141 = v15;
    sub_37C29D0(i, v132, v131, a4);
    v8 = v141;
    result = (char *)i - (char *)v141;
    if ( (char *)i - (char *)v141 <= 128 )
      return result;
    if ( v131 )
    {
      v132 = i;
      continue;
    }
    break;
  }
  v5 = v141;
  v6 = a4;
LABEL_118:
  v146 = v6;
  v111 = result >> 3;
  v112 = ((result >> 3) - 2) >> 1;
  sub_37C2470((__int64)v5, v112, result >> 3, v5[v112], v6);
  do
  {
    --v112;
    sub_37C2470((__int64)v5, v112, v111, v5[v112], v146);
  }
  while ( v112 );
  v113 = v142;
  do
  {
    v114 = *--v113;
    *v113 = *v5;
    result = sub_37C2470((__int64)v5, 0, v113 - v5, v114, v146);
  }
  while ( (char *)v113 - (char *)v5 > 8 );
  return result;
}
