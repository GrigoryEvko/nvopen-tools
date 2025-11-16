// Function: sub_2610DF0
// Address: 0x2610df0
//
char *__fastcall sub_2610DF0(char *a1, char *a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  int v8; // ecx
  __int64 v9; // r11
  __int64 v10; // rsi
  int v11; // edx
  char *v12; // rbx
  __int64 v13; // r9
  int v14; // r10d
  int v15; // r8d
  unsigned int v16; // r12d
  __int64 *v17; // rax
  __int64 v18; // r14
  __int64 v19; // r13
  unsigned int v20; // r12d
  __int64 *v21; // rax
  __int64 v22; // r14
  __int64 v23; // r13
  unsigned int v24; // r12d
  __int64 *v25; // rax
  __int64 v26; // r14
  __int64 v27; // r13
  unsigned int v28; // r12d
  __int64 *v29; // rax
  __int64 v30; // r14
  __int64 v31; // r13
  unsigned int v32; // r12d
  __int64 *v33; // rax
  __int64 v34; // r14
  __int64 v35; // r13
  unsigned int v36; // r12d
  __int64 *v37; // rax
  __int64 v38; // r14
  __int64 v39; // r13
  unsigned int v40; // r12d
  __int64 *v41; // rax
  __int64 v42; // r14
  __int64 v43; // r13
  unsigned int v44; // r12d
  __int64 *v45; // rax
  __int64 v46; // r14
  __int64 v47; // r13
  char *result; // rax
  int v49; // eax
  int v50; // eax
  int v51; // eax
  int v52; // eax
  int v53; // eax
  int v54; // eax
  int v55; // eax
  int v56; // eax
  int v57; // edx
  __int64 v58; // r8
  __int64 v59; // rsi
  unsigned int v60; // ecx
  __int64 *v61; // rax
  __int64 v62; // r10
  __int64 v63; // r10
  int v64; // ecx
  __int64 v65; // r8
  unsigned int v66; // r9d
  __int64 *v67; // rax
  __int64 v68; // r12
  __int64 v69; // r10
  __int64 v70; // r9
  __int64 v71; // r11
  int v72; // edx
  unsigned int v73; // r10d
  __int64 *v74; // rax
  __int64 v75; // rbx
  __int64 v76; // rsi
  int v77; // ecx
  unsigned int v78; // edx
  __int64 *v79; // rax
  __int64 v80; // r10
  __int64 v81; // rax
  bool v82; // zf
  __int64 v83; // r11
  unsigned int v84; // r10d
  __int64 *v85; // rdi
  __int64 v86; // r13
  __int64 v87; // r11
  unsigned int v88; // r10d
  __int64 *v89; // rdi
  __int64 v90; // r13
  __int64 v91; // rdi
  int v92; // eax
  int v93; // r12d
  int v94; // eax
  int v95; // r11d
  int v96; // edi
  int v97; // r12d
  __int64 v98; // rdi
  int v99; // edi
  int v100; // r12d
  __int64 v101; // rdi
  int v102; // eax
  int v103; // ebx
  int v104; // eax
  int v105; // r11d
  int v106; // [rsp+0h] [rbp-3Ch]
  int v107; // [rsp+0h] [rbp-3Ch]
  int v108; // [rsp+0h] [rbp-3Ch]
  int v109; // [rsp+0h] [rbp-3Ch]
  int v110; // [rsp+0h] [rbp-3Ch]
  int v111; // [rsp+0h] [rbp-3Ch]
  int v112; // [rsp+0h] [rbp-3Ch]
  int v113; // [rsp+0h] [rbp-3Ch]

  v5 = (a2 - a1) >> 5;
  v6 = (a2 - a1) >> 3;
  if ( v5 > 0 )
  {
    v8 = *(_DWORD *)(a3 + 120);
    v9 = *(_QWORD *)(a3 + 104);
    v10 = *a4;
    v11 = *(_DWORD *)(a3 + 328);
    v12 = &a1[32 * v5];
    v13 = *(_QWORD *)(a3 + 312);
    v14 = v8 - 1;
    v15 = v11 - 1;
    while ( 1 )
    {
      v47 = *(_QWORD *)a1;
      if ( !v8 )
        goto LABEL_25;
      v16 = v14 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v17 = (__int64 *)(v9 + 16LL * v16);
      v18 = *v17;
      if ( v47 != *v17 )
        break;
LABEL_4:
      v19 = v17[1];
      if ( !v11 )
        goto LABEL_26;
LABEL_5:
      v20 = v15 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v21 = (__int64 *)(v13 + 16LL * v20);
      v22 = *v21;
      if ( v19 == *v21 )
      {
LABEL_6:
        if ( v10 == v21[1] )
          return a1;
        goto LABEL_7;
      }
      v54 = 1;
      while ( v22 != -4096 )
      {
        v20 = v15 & (v54 + v20);
        v107 = v54 + 1;
        v21 = (__int64 *)(v13 + 16LL * v20);
        v22 = *v21;
        if ( v19 == *v21 )
          goto LABEL_6;
        v54 = v107;
      }
LABEL_26:
      if ( !v10 )
        return a1;
LABEL_7:
      v23 = *((_QWORD *)a1 + 1);
      if ( !v8 )
        goto LABEL_31;
      v24 = v14 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v25 = (__int64 *)(v9 + 16LL * v24);
      v26 = *v25;
      if ( v23 != *v25 )
      {
        v49 = 1;
        while ( v26 != -4096 )
        {
          v24 = v14 & (v49 + v24);
          v110 = v49 + 1;
          v25 = (__int64 *)(v9 + 16LL * v24);
          v26 = *v25;
          if ( v23 == *v25 )
            goto LABEL_9;
          v49 = v110;
        }
LABEL_31:
        v27 = 0;
        if ( !v11 )
          goto LABEL_32;
        goto LABEL_10;
      }
LABEL_9:
      v27 = v25[1];
      if ( !v11 )
        goto LABEL_32;
LABEL_10:
      v28 = v15 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v29 = (__int64 *)(v13 + 16LL * v28);
      v30 = *v29;
      if ( v27 == *v29 )
      {
LABEL_11:
        if ( v10 == v29[1] )
          return a1 + 8;
        goto LABEL_12;
      }
      v55 = 1;
      while ( v30 != -4096 )
      {
        v28 = v15 & (v55 + v28);
        v108 = v55 + 1;
        v29 = (__int64 *)(v13 + 16LL * v28);
        v30 = *v29;
        if ( v27 == *v29 )
          goto LABEL_11;
        v55 = v108;
      }
LABEL_32:
      if ( !v10 )
        return a1 + 8;
LABEL_12:
      v31 = *((_QWORD *)a1 + 2);
      if ( v8 )
      {
        v32 = v14 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v33 = (__int64 *)(v9 + 16LL * v32);
        v34 = *v33;
        if ( v31 == *v33 )
        {
LABEL_14:
          v35 = v33[1];
          goto LABEL_15;
        }
        v51 = 1;
        while ( v34 != -4096 )
        {
          v32 = v14 & (v51 + v32);
          v112 = v51 + 1;
          v33 = (__int64 *)(v9 + 16LL * v32);
          v34 = *v33;
          if ( v31 == *v33 )
            goto LABEL_14;
          v51 = v112;
        }
      }
      v35 = 0;
LABEL_15:
      if ( v11 )
      {
        v36 = v15 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v37 = (__int64 *)(v13 + 16LL * v36);
        v38 = *v37;
        if ( v35 == *v37 )
        {
LABEL_17:
          if ( v10 == v37[1] )
            return a1 + 16;
          goto LABEL_18;
        }
        v50 = 1;
        while ( v38 != -4096 )
        {
          v36 = v15 & (v50 + v36);
          v111 = v50 + 1;
          v37 = (__int64 *)(v13 + 16LL * v36);
          v38 = *v37;
          if ( v35 == *v37 )
            goto LABEL_17;
          v50 = v111;
        }
      }
      if ( !v10 )
        return a1 + 16;
LABEL_18:
      v39 = *((_QWORD *)a1 + 3);
      if ( v8 )
      {
        v40 = v14 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v41 = (__int64 *)(v9 + 16LL * v40);
        v42 = *v41;
        if ( v39 == *v41 )
        {
LABEL_20:
          v43 = v41[1];
          if ( !v11 )
            goto LABEL_44;
          goto LABEL_21;
        }
        v52 = 1;
        while ( v42 != -4096 )
        {
          v40 = v14 & (v52 + v40);
          v113 = v52 + 1;
          v41 = (__int64 *)(v9 + 16LL * v40);
          v42 = *v41;
          if ( v39 == *v41 )
            goto LABEL_20;
          v52 = v113;
        }
      }
      v43 = 0;
      if ( !v11 )
        goto LABEL_44;
LABEL_21:
      v44 = v15 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v45 = (__int64 *)(v13 + 16LL * v44);
      v46 = *v45;
      if ( v43 != *v45 )
      {
        v56 = 1;
        while ( v46 != -4096 )
        {
          v44 = v15 & (v56 + v44);
          v109 = v56 + 1;
          v45 = (__int64 *)(v13 + 16LL * v44);
          v46 = *v45;
          if ( v43 == *v45 )
            goto LABEL_22;
          v56 = v109;
        }
LABEL_44:
        if ( !v10 )
          return a1 + 24;
        goto LABEL_23;
      }
LABEL_22:
      if ( v10 == v45[1] )
        return a1 + 24;
LABEL_23:
      a1 += 32;
      if ( v12 == a1 )
      {
        v6 = (a2 - a1) >> 3;
        goto LABEL_63;
      }
    }
    v53 = 1;
    while ( v18 != -4096 )
    {
      v16 = v14 & (v53 + v16);
      v106 = v53 + 1;
      v17 = (__int64 *)(v9 + 16LL * v16);
      v18 = *v17;
      if ( v47 == *v17 )
        goto LABEL_4;
      v53 = v106;
    }
LABEL_25:
    v19 = 0;
    if ( !v11 )
      goto LABEL_26;
    goto LABEL_5;
  }
LABEL_63:
  if ( v6 == 2 )
  {
    v59 = *(_QWORD *)(a3 + 104);
    v57 = *(_DWORD *)(a3 + 120);
    v65 = *(_QWORD *)(a3 + 312);
    v64 = *(_DWORD *)(a3 + 328);
    v70 = *a4;
    result = a1;
    goto LABEL_94;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return a2;
    v59 = *(_QWORD *)(a3 + 104);
    v57 = *(_DWORD *)(a3 + 120);
    v65 = *(_QWORD *)(a3 + 312);
    v70 = *a4;
    v64 = *(_DWORD *)(a3 + 328);
    goto LABEL_84;
  }
  v57 = *(_DWORD *)(a3 + 120);
  v58 = *(_QWORD *)a1;
  v59 = *(_QWORD *)(a3 + 104);
  if ( v57 )
  {
    v60 = (v57 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
    v61 = (__int64 *)(v59 + 16LL * v60);
    v62 = *v61;
    if ( v58 == *v61 )
    {
LABEL_77:
      v63 = v61[1];
      goto LABEL_78;
    }
    v104 = 1;
    while ( v62 != -4096 )
    {
      v105 = v104 + 1;
      v60 = (v57 - 1) & (v104 + v60);
      v61 = (__int64 *)(v59 + 16LL * v60);
      v62 = *v61;
      if ( v58 == *v61 )
        goto LABEL_77;
      v104 = v105;
    }
  }
  v63 = 0;
LABEL_78:
  v64 = *(_DWORD *)(a3 + 328);
  v65 = *(_QWORD *)(a3 + 312);
  if ( !v64 )
  {
LABEL_107:
    v69 = 0;
    goto LABEL_81;
  }
  v66 = (v64 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
  v67 = (__int64 *)(v65 + 16LL * v66);
  v68 = *v67;
  if ( *v67 != v63 )
  {
    v102 = 1;
    while ( v68 != -4096 )
    {
      v103 = v102 + 1;
      v66 = (v64 - 1) & (v102 + v66);
      v67 = (__int64 *)(v65 + 16LL * v66);
      v68 = *v67;
      if ( v63 == *v67 )
        goto LABEL_80;
      v102 = v103;
    }
    goto LABEL_107;
  }
LABEL_80:
  v69 = v67[1];
LABEL_81:
  v70 = *a4;
  result = a1;
  if ( *a4 == v69 )
    return result;
  result = a1 + 8;
LABEL_94:
  v83 = *(_QWORD *)result;
  if ( v57 )
  {
    v84 = (v57 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
    v85 = (__int64 *)(v59 + 16LL * v84);
    v86 = *v85;
    if ( v83 == *v85 )
    {
LABEL_96:
      v87 = v85[1];
      goto LABEL_97;
    }
    v99 = 1;
    while ( v86 != -4096 )
    {
      v100 = v99 + 1;
      v101 = (v57 - 1) & (v84 + v99);
      v84 = v101;
      v85 = (__int64 *)(v59 + 16 * v101);
      v86 = *v85;
      if ( v83 == *v85 )
        goto LABEL_96;
      v99 = v100;
    }
  }
  v87 = 0;
LABEL_97:
  if ( !v64 )
  {
LABEL_105:
    v91 = 0;
    goto LABEL_100;
  }
  v88 = (v64 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
  v89 = (__int64 *)(v65 + 16LL * v88);
  v90 = *v89;
  if ( v87 != *v89 )
  {
    v96 = 1;
    while ( v90 != -4096 )
    {
      v97 = v96 + 1;
      v98 = (v64 - 1) & (v88 + v96);
      v88 = v98;
      v89 = (__int64 *)(v65 + 16 * v98);
      v90 = *v89;
      if ( v87 == *v89 )
        goto LABEL_99;
      v96 = v97;
    }
    goto LABEL_105;
  }
LABEL_99:
  v91 = v89[1];
LABEL_100:
  if ( v91 == v70 )
    return result;
  a1 = result + 8;
LABEL_84:
  v71 = *(_QWORD *)a1;
  if ( v57 )
  {
    v72 = v57 - 1;
    v73 = v72 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
    v74 = (__int64 *)(v59 + 16LL * v73);
    v75 = *v74;
    if ( v71 == *v74 )
    {
LABEL_86:
      v76 = v74[1];
      goto LABEL_87;
    }
    v92 = 1;
    while ( v75 != -4096 )
    {
      v93 = v92 + 1;
      v73 = v72 & (v92 + v73);
      v74 = (__int64 *)(v59 + 16LL * v73);
      v75 = *v74;
      if ( v71 == *v74 )
        goto LABEL_86;
      v92 = v93;
    }
  }
  v76 = 0;
LABEL_87:
  if ( v64 )
  {
    v77 = v64 - 1;
    v78 = v77 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
    v79 = (__int64 *)(v65 + 16LL * v78);
    v80 = *v79;
    if ( *v79 == v76 )
    {
LABEL_89:
      v81 = v79[1];
      goto LABEL_90;
    }
    v94 = 1;
    while ( v80 != -4096 )
    {
      v95 = v94 + 1;
      v78 = v77 & (v94 + v78);
      v79 = (__int64 *)(v65 + 16LL * v78);
      v80 = *v79;
      if ( v76 == *v79 )
        goto LABEL_89;
      v94 = v95;
    }
  }
  v81 = 0;
LABEL_90:
  v82 = v70 == v81;
  result = a2;
  if ( v82 )
    return a1;
  return result;
}
