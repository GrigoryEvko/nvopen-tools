// Function: sub_21ED230
// Address: 0x21ed230
//
__int64 *__fastcall sub_21ED230(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        int *a8,
        __int64 a9)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 *v12; // r15
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rdi
  int v18; // edx
  __int64 v19; // r9
  unsigned int v20; // r8d
  __int64 *v21; // rsi
  __int64 v22; // r13
  __int64 v23; // r13
  unsigned int v24; // esi
  unsigned int v25; // r10d
  __int64 v26; // r9
  unsigned int v27; // ecx
  int *v28; // rdi
  int v29; // r8d
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r12
  __int64 v36; // rdi
  int v37; // edx
  __int64 v38; // r9
  unsigned int v39; // r8d
  __int64 *v40; // rsi
  __int64 v41; // r13
  __int64 v42; // r13
  unsigned int v43; // esi
  unsigned int v44; // r10d
  __int64 v45; // r9
  unsigned int v46; // ecx
  int *v47; // rdi
  int v48; // r8d
  unsigned int v49; // eax
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r12
  __int64 v55; // rdi
  int v56; // edx
  __int64 v57; // r9
  unsigned int v58; // r8d
  __int64 *v59; // rsi
  __int64 v60; // r13
  __int64 v61; // r13
  unsigned int v62; // esi
  unsigned int v63; // r10d
  __int64 v64; // r9
  unsigned int v65; // ecx
  int *v66; // rdi
  int v67; // r8d
  unsigned int v68; // eax
  unsigned __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // r12
  __int64 v74; // rsi
  int v75; // eax
  __int64 v76; // rdi
  unsigned int v77; // r8d
  __int64 *v78; // rdx
  __int64 v79; // r10
  __int64 v80; // r13
  unsigned int v81; // esi
  unsigned int v82; // r9d
  __int64 v83; // r8
  unsigned int v84; // edx
  int *v85; // rcx
  int v86; // edi
  unsigned int v87; // eax
  unsigned __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rax
  int v92; // r10d
  unsigned int v93; // r14d
  int i; // r11d
  int v95; // esi
  int v96; // esi
  int v97; // esi
  int v98; // edx
  int v99; // r11d
  unsigned int v100; // r14d
  int v101; // r11d
  int *v102; // rax
  int v103; // edi
  int v104; // edi
  int v105; // r11d
  unsigned int v106; // r14d
  int v107; // r11d
  int *v108; // rax
  int v109; // edi
  int v110; // edi
  int v111; // r11d
  unsigned int v112; // r14d
  int v113; // r11d
  int *v114; // rax
  int v115; // edi
  int v116; // edi
  int v117; // r11d
  int v118; // r11d
  int v119; // r11d
  int v120; // r11d
  int v121; // r11d
  int *v122; // r10
  int v123; // edi
  int v124; // ecx
  int *v125; // r14
  unsigned int v126; // eax
  int *v127; // r14
  unsigned int v128; // eax
  int *v129; // r14
  unsigned int v130; // eax
  int *v131; // r14
  unsigned int v132; // eax
  int v133; // [rsp+4h] [rbp-4Ch]
  int v134; // [rsp+4h] [rbp-4Ch]
  int v135; // [rsp+4h] [rbp-4Ch]
  int v137; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v138[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = (a2 - (__int64)a1) >> 5;
  v10 = (a2 - (__int64)a1) >> 3;
  v11 = a1;
  if ( v9 <= 0 )
  {
LABEL_43:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          return (__int64 *)a2;
LABEL_101:
        if ( (unsigned __int8)sub_21ED010((__int64)&a7, *v11) )
          return v11;
        return (__int64 *)a2;
      }
      if ( (unsigned __int8)sub_21ED010((__int64)&a7, *v11) )
        return v11;
      ++v11;
    }
    if ( (unsigned __int8)sub_21ED010((__int64)&a7, *v11) )
      return v11;
    ++v11;
    goto LABEL_101;
  }
  v12 = &a1[4 * v9];
  while ( 1 )
  {
    v13 = a7;
    v14 = *v11;
    v15 = *a7;
    if ( *v11 == *a7 )
      goto LABEL_12;
    v16 = a9;
    v17 = *(unsigned int *)(a9 + 136);
    v18 = *a8;
    v19 = *(_QWORD *)(a9 + 120);
    if ( (_DWORD)v17 )
    {
      v20 = (v17 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( v14 == *v21 )
        goto LABEL_6;
      v95 = 1;
      while ( v22 != -8 )
      {
        v117 = v95 + 1;
        v20 = (v17 - 1) & (v95 + v20);
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( v14 == *v21 )
          goto LABEL_6;
        v95 = v117;
      }
    }
    v21 = (__int64 *)(v19 + 16 * v17);
LABEL_6:
    v23 = v21[1];
    v137 = *a8;
    v24 = *(_DWORD *)(a9 + 80);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a9 + 64);
      v27 = (v24 - 1) & (37 * v18);
      v28 = (int *)(v26 + 8LL * v27);
      v29 = *v28;
      if ( v18 == *v28 )
      {
        v30 = v28[1];
        v31 = v30 & 0x3F;
        v32 = 8LL * (v30 >> 6);
LABEL_9:
        v33 = *(_QWORD *)(*(_QWORD *)(v23 + 24) + v32);
        if ( _bittest64(&v33, v31) )
          return v11;
        v13 = a7;
        goto LABEL_11;
      }
      v133 = 1;
      v99 = *v28;
      v100 = (v24 - 1) & (37 * v18);
      do
      {
        if ( v99 == -1 )
          goto LABEL_11;
        v100 = v25 & (v133 + v100);
        ++v133;
        v99 = *(_DWORD *)(v26 + 8LL * v100);
      }
      while ( v18 != v99 );
      v101 = 1;
      v102 = 0;
      while ( v29 != -1 )
      {
        if ( v102 || v29 != -2 )
          v28 = v102;
        v27 = v25 & (v101 + v27);
        v125 = (int *)(v26 + 8LL * v27);
        v29 = *v125;
        if ( v18 == *v125 )
        {
          v126 = v125[1];
          v31 = v126 & 0x3F;
          v32 = 8LL * (v126 >> 6);
          goto LABEL_9;
        }
        ++v101;
        v102 = v28;
        v28 = (int *)(v26 + 8LL * v27);
      }
      if ( !v102 )
        v102 = v28;
      v103 = *(_DWORD *)(a9 + 72);
      ++*(_QWORD *)(a9 + 56);
      v104 = v103 + 1;
      if ( 4 * v104 >= 3 * v24 )
      {
        v24 *= 2;
      }
      else if ( v24 - *(_DWORD *)(v16 + 76) - v104 > v24 >> 3 )
      {
LABEL_70:
        *(_DWORD *)(v16 + 72) = v104;
        if ( *v102 != -1 )
          --*(_DWORD *)(v16 + 76);
        *v102 = v18;
        v31 = 0;
        v102[1] = 0;
        v32 = 0;
        goto LABEL_9;
      }
      sub_1BFDD60(v16 + 56, v24);
      sub_1BFD720(v16 + 56, &v137, v138);
      v102 = (int *)v138[0];
      v18 = v137;
      v104 = *(_DWORD *)(v16 + 72) + 1;
      goto LABEL_70;
    }
LABEL_11:
    v15 = *v13;
LABEL_12:
    v34 = v11[1];
    if ( v34 == v15 )
      goto LABEL_21;
    v35 = a9;
    v36 = *(unsigned int *)(a9 + 136);
    v37 = *a8;
    v38 = *(_QWORD *)(a9 + 120);
    if ( (_DWORD)v36 )
    {
      v39 = (v36 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v40 = (__int64 *)(v38 + 16LL * v39);
      v41 = *v40;
      if ( v34 == *v40 )
        goto LABEL_15;
      v96 = 1;
      while ( v41 != -8 )
      {
        v118 = v96 + 1;
        v39 = (v36 - 1) & (v96 + v39);
        v40 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( v34 == *v40 )
          goto LABEL_15;
        v96 = v118;
      }
    }
    v40 = (__int64 *)(v38 + 16 * v36);
LABEL_15:
    v42 = v40[1];
    v137 = *a8;
    v43 = *(_DWORD *)(a9 + 80);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a9 + 64);
      v46 = (v43 - 1) & (37 * v37);
      v47 = (int *)(v45 + 8LL * v46);
      v48 = *v47;
      if ( v37 == *v47 )
      {
        v49 = v47[1];
        v50 = v49 & 0x3F;
        v51 = 8LL * (v49 >> 6);
LABEL_18:
        v52 = *(_QWORD *)(*(_QWORD *)(v42 + 24) + v51);
        if ( _bittest64(&v52, v50) )
          return ++v11;
        v13 = a7;
        goto LABEL_20;
      }
      v134 = 1;
      v105 = *v47;
      v106 = (v43 - 1) & (37 * v37);
      do
      {
        if ( v105 == -1 )
          goto LABEL_20;
        v106 = v44 & (v134 + v106);
        ++v134;
        v105 = *(_DWORD *)(v45 + 8LL * v106);
      }
      while ( v37 != v105 );
      v107 = 1;
      v108 = 0;
      while ( v48 != -1 )
      {
        if ( v108 || v48 != -2 )
          v47 = v108;
        v46 = v44 & (v107 + v46);
        v129 = (int *)(v45 + 8LL * v46);
        v48 = *v129;
        if ( v37 == *v129 )
        {
          v130 = v129[1];
          v50 = v130 & 0x3F;
          v51 = 8LL * (v130 >> 6);
          goto LABEL_18;
        }
        ++v107;
        v108 = v47;
        v47 = (int *)(v45 + 8LL * v46);
      }
      if ( !v108 )
        v108 = v47;
      v109 = *(_DWORD *)(a9 + 72);
      ++*(_QWORD *)(a9 + 56);
      v110 = v109 + 1;
      if ( 4 * v110 >= 3 * v43 )
      {
        v43 *= 2;
      }
      else if ( v43 - *(_DWORD *)(v35 + 76) - v110 > v43 >> 3 )
      {
LABEL_82:
        *(_DWORD *)(v35 + 72) = v110;
        if ( *v108 != -1 )
          --*(_DWORD *)(v35 + 76);
        *v108 = v37;
        v50 = 0;
        v108[1] = 0;
        v51 = 0;
        goto LABEL_18;
      }
      sub_1BFDD60(v35 + 56, v43);
      sub_1BFD720(v35 + 56, &v137, v138);
      v108 = (int *)v138[0];
      v37 = v137;
      v110 = *(_DWORD *)(v35 + 72) + 1;
      goto LABEL_82;
    }
LABEL_20:
    v15 = *v13;
LABEL_21:
    v53 = v11[2];
    if ( v53 == v15 )
      goto LABEL_30;
    v54 = a9;
    v55 = *(unsigned int *)(a9 + 136);
    v56 = *a8;
    v57 = *(_QWORD *)(a9 + 120);
    if ( (_DWORD)v55 )
    {
      v58 = (v55 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v59 = (__int64 *)(v57 + 16LL * v58);
      v60 = *v59;
      if ( v53 == *v59 )
        goto LABEL_24;
      v97 = 1;
      while ( v60 != -8 )
      {
        v119 = v97 + 1;
        v58 = (v55 - 1) & (v97 + v58);
        v59 = (__int64 *)(v57 + 16LL * v58);
        v60 = *v59;
        if ( v53 == *v59 )
          goto LABEL_24;
        v97 = v119;
      }
    }
    v59 = (__int64 *)(v57 + 16 * v55);
LABEL_24:
    v61 = v59[1];
    v137 = *a8;
    v62 = *(_DWORD *)(a9 + 80);
    if ( v62 )
    {
      v63 = v62 - 1;
      v64 = *(_QWORD *)(a9 + 64);
      v65 = (v62 - 1) & (37 * v56);
      v66 = (int *)(v64 + 8LL * v65);
      v67 = *v66;
      if ( v56 == *v66 )
      {
        v68 = v66[1];
        v69 = v68 & 0x3F;
        v70 = 8LL * (v68 >> 6);
LABEL_27:
        v71 = *(_QWORD *)(*(_QWORD *)(v61 + 24) + v70);
        if ( _bittest64(&v71, v69) )
        {
          v11 += 2;
          return v11;
        }
        v13 = a7;
        goto LABEL_29;
      }
      v135 = 1;
      v111 = *v66;
      v112 = (v62 - 1) & (37 * v56);
      do
      {
        if ( v111 == -1 )
          goto LABEL_29;
        v112 = v63 & (v135 + v112);
        ++v135;
        v111 = *(_DWORD *)(v64 + 8LL * v112);
      }
      while ( v56 != v111 );
      v113 = 1;
      v114 = 0;
      while ( v67 != -1 )
      {
        if ( v67 != -2 || v114 )
          v66 = v114;
        v65 = v63 & (v113 + v65);
        v131 = (int *)(v64 + 8LL * v65);
        v67 = *v131;
        if ( v56 == *v131 )
        {
          v132 = v131[1];
          v69 = v132 & 0x3F;
          v70 = 8LL * (v132 >> 6);
          goto LABEL_27;
        }
        ++v113;
        v114 = v66;
        v66 = (int *)(v64 + 8LL * v65);
      }
      if ( !v114 )
        v114 = v66;
      v115 = *(_DWORD *)(a9 + 72);
      ++*(_QWORD *)(a9 + 56);
      v116 = v115 + 1;
      if ( 4 * v116 >= 3 * v62 )
      {
        v62 *= 2;
      }
      else if ( v62 - *(_DWORD *)(v54 + 76) - v116 > v62 >> 3 )
      {
LABEL_94:
        *(_DWORD *)(v54 + 72) = v116;
        if ( *v114 != -1 )
          --*(_DWORD *)(v54 + 76);
        *v114 = v56;
        v69 = 0;
        v114[1] = 0;
        v70 = 0;
        goto LABEL_27;
      }
      sub_1BFDD60(v54 + 56, v62);
      sub_1BFD720(v54 + 56, &v137, v138);
      v114 = (int *)v138[0];
      v56 = v137;
      v116 = *(_DWORD *)(v54 + 72) + 1;
      goto LABEL_94;
    }
LABEL_29:
    v15 = *v13;
LABEL_30:
    v72 = v11[3];
    if ( v72 != v15 )
      break;
LABEL_41:
    v11 += 4;
    if ( v12 == v11 )
    {
      v10 = (a2 - (__int64)v11) >> 3;
      goto LABEL_43;
    }
  }
  v73 = a9;
  v74 = *(unsigned int *)(a9 + 136);
  v75 = *a8;
  v76 = *(_QWORD *)(a9 + 120);
  if ( (_DWORD)v74 )
  {
    v77 = (v74 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
    v78 = (__int64 *)(v76 + 16LL * v77);
    v79 = *v78;
    if ( v72 == *v78 )
      goto LABEL_33;
    v98 = 1;
    while ( v79 != -8 )
    {
      v120 = v98 + 1;
      v77 = (v74 - 1) & (v98 + v77);
      v78 = (__int64 *)(v76 + 16LL * v77);
      v79 = *v78;
      if ( v72 == *v78 )
        goto LABEL_33;
      v98 = v120;
    }
  }
  v78 = (__int64 *)(v76 + 16 * v74);
LABEL_33:
  v80 = v78[1];
  v137 = *a8;
  v81 = *(_DWORD *)(a9 + 80);
  if ( !v81 )
    goto LABEL_41;
  v82 = v81 - 1;
  v83 = *(_QWORD *)(a9 + 64);
  v84 = (v81 - 1) & (37 * v75);
  v85 = (int *)(v83 + 8LL * v84);
  v86 = *v85;
  if ( v75 != *v85 )
  {
    v92 = *v85;
    v93 = (v81 - 1) & (37 * v75);
    for ( i = 1; ; ++i )
    {
      if ( v92 == -1 )
        goto LABEL_41;
      v93 = v82 & (i + v93);
      v92 = *(_DWORD *)(v83 + 8LL * v93);
      if ( v75 == v92 )
        break;
    }
    v121 = 1;
    v122 = 0;
    while ( v86 != -1 )
    {
      if ( v122 || v86 != -2 )
        v85 = v122;
      v84 = v82 & (v121 + v84);
      v127 = (int *)(v83 + 8LL * v84);
      v86 = *v127;
      if ( v75 == *v127 )
      {
        v128 = v127[1];
        v88 = v128 & 0x3F;
        v89 = 8LL * (v128 >> 6);
        goto LABEL_36;
      }
      ++v121;
      v122 = v85;
      v85 = (int *)(v83 + 8LL * v84);
    }
    v123 = *(_DWORD *)(a9 + 72);
    if ( !v122 )
      v122 = v85;
    ++*(_QWORD *)(a9 + 56);
    v124 = v123 + 1;
    if ( 4 * (v123 + 1) >= 3 * v81 )
    {
      v81 *= 2;
    }
    else if ( v81 - *(_DWORD *)(v73 + 76) - v124 > v81 >> 3 )
    {
LABEL_118:
      *(_DWORD *)(v73 + 72) = v124;
      if ( *v122 != -1 )
        --*(_DWORD *)(v73 + 76);
      *v122 = v75;
      v88 = 0;
      v89 = 0;
      v122[1] = 0;
      goto LABEL_36;
    }
    sub_1BFDD60(v73 + 56, v81);
    sub_1BFD720(v73 + 56, &v137, v138);
    v122 = (int *)v138[0];
    v75 = v137;
    v124 = *(_DWORD *)(v73 + 72) + 1;
    goto LABEL_118;
  }
  v87 = v85[1];
  v88 = v87 & 0x3F;
  v89 = 8LL * (v87 >> 6);
LABEL_36:
  v90 = *(_QWORD *)(*(_QWORD *)(v80 + 24) + v89);
  if ( !_bittest64(&v90, v88) )
    goto LABEL_41;
  v11 += 3;
  return v11;
}
