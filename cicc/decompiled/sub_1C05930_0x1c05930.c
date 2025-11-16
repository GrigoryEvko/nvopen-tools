// Function: sub_1C05930
// Address: 0x1c05930
//
__int64 __fastcall sub_1C05930(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  int v4; // r8d
  __int64 v5; // rcx
  __int64 v6; // r12
  unsigned int v8; // ebx
  unsigned int v9; // r10d
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 *v12; // rdi
  __int64 *v13; // rdx
  __int64 v14; // r15
  unsigned int v16; // r11d
  __int64 *v17; // rdi
  __int64 v18; // r15
  __int64 v19; // rbx
  unsigned int v20; // r9d
  __int64 *v21; // rax
  __int64 v22; // rdi
  char v23; // al
  __int64 v24; // r15
  int j; // r10d
  int v26; // r15d
  __int64 v27; // r10
  __int64 *v28; // r10
  int v29; // r15d
  __int64 *v30; // r10
  __int64 v31; // rdi
  int v32; // eax
  int v33; // eax
  __int64 v34; // r13
  __int64 v35; // rcx
  __int64 **v36; // rbx
  __int64 **v37; // r14
  __int64 **v38; // rbx
  __int64 **v39; // r14
  int v40; // eax
  __int64 v41; // rdx
  unsigned int v42; // r11d
  int v43; // edi
  int v44; // r15d
  bool v45; // dl
  __int64 v46; // r15
  __int64 v47; // r12
  __int64 *v48; // rax
  int v49; // edx
  int v50; // esi
  __int64 v51; // r8
  unsigned int v52; // edx
  __int64 *v53; // rdi
  __int64 *v54; // rax
  int v55; // edx
  int v56; // ecx
  __int64 v57; // rdi
  unsigned int v58; // edx
  __int64 *v59; // rsi
  int v60; // edi
  int v61; // r15d
  __int64 v62; // rdi
  int k; // r9d
  int v64; // r8d
  __int64 v65; // r15
  int i; // edi
  int v67; // r15d
  __int64 v68; // rdi
  __int64 *v69; // rdi
  int v70; // r15d
  __int64 *v71; // rdi
  int v72; // eax
  int v73; // edx
  int v74; // esi
  int v75; // esi
  __int64 v76; // r9
  __int64 v77; // rbx
  __int64 v78; // r8
  int v79; // ecx
  __int64 *v80; // rax
  int v81; // edi
  int v82; // edi
  __int64 v83; // r9
  __int64 v84; // r8
  __int64 v85; // rsi
  int v86; // ecx
  __int64 *v87; // rdx
  int v88; // esi
  int v89; // esi
  __int64 v90; // r9
  int v91; // ecx
  __int64 v92; // rbx
  __int64 v93; // r8
  int v94; // esi
  int v95; // esi
  __int64 v96; // r9
  int v97; // ecx
  __int64 v98; // r8
  __int64 v99; // rdi
  __int64 *v100; // [rsp+0h] [rbp-50h]
  int v101; // [rsp+0h] [rbp-50h]
  __int64 *v102; // [rsp+0h] [rbp-50h]
  int v103; // [rsp+8h] [rbp-48h]
  int v104; // [rsp+8h] [rbp-48h]
  int v105; // [rsp+Ch] [rbp-44h]
  int v106; // [rsp+Ch] [rbp-44h]
  bool v107; // [rsp+Ch] [rbp-44h]
  int v108; // [rsp+Ch] [rbp-44h]
  unsigned int v109; // [rsp+Ch] [rbp-44h]
  unsigned int v110; // [rsp+Ch] [rbp-44h]
  int v111[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = a2;
  LODWORD(a2) = *(_DWORD *)(a1 + 64);
  if ( !(_DWORD)a2 )
    goto LABEL_4;
  v4 = a2 - 1;
  v5 = *(_QWORD *)(a1 + 48);
  v6 = a1;
  v8 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v9 = (a2 - 1) & v8;
  v10 = (__int64 *)(v5 + 16LL * v9);
  v11 = *v10;
  v12 = v10;
  if ( v3 != *v10 )
  {
    v41 = *v10;
    v42 = (a2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v43 = 1;
    while ( v41 != -8 )
    {
      v44 = v43 + 1;
      v42 = v4 & (v43 + v42);
      v12 = (__int64 *)(v5 + 16LL * v42);
      v41 = *v12;
      if ( v3 == *v12 )
        goto LABEL_3;
      v43 = v44;
    }
    goto LABEL_4;
  }
LABEL_3:
  v13 = (__int64 *)(v5 + 16LL * (unsigned int)a2);
  if ( v13 == v12 )
    goto LABEL_4;
  v16 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v105 = v16 & v4;
  v17 = (__int64 *)(v5 + 16LL * (v16 & v4));
  v18 = *v17;
  if ( a3 != *v17 )
  {
    v60 = 1;
    while ( v18 != -8 )
    {
      v61 = v60 + 1;
      v62 = v4 & (unsigned int)(v105 + v60);
      v101 = v61;
      v105 = v62;
      v17 = (__int64 *)(v5 + 16 * v62);
      v18 = *v17;
      if ( a3 == *v17 )
        goto LABEL_7;
      v60 = v101;
    }
    goto LABEL_4;
  }
LABEL_7:
  if ( v13 == v17 )
    goto LABEL_4;
  if ( v3 == v11 )
  {
    if ( v13 != v10 )
    {
LABEL_10:
      v19 = v10[1];
      goto LABEL_11;
    }
    goto LABEL_32;
  }
  v108 = (a2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v65 = *v10;
  for ( i = 1; ; i = v104 )
  {
    if ( v65 == -8 )
      goto LABEL_32;
    v67 = i + 1;
    v68 = v4 & (unsigned int)(v108 + i);
    v104 = v67;
    v108 = v68;
    v69 = (__int64 *)(v5 + 16 * v68);
    v65 = *v69;
    v102 = v69;
    if ( v3 == *v69 )
      break;
  }
  v70 = 1;
  v71 = 0;
  if ( v13 == v102 )
  {
LABEL_32:
    v19 = 0;
    goto LABEL_11;
  }
  while ( v11 != -8 )
  {
    if ( !v71 && v11 == -16 )
      v71 = v10;
    v9 = v4 & (v70 + v9);
    v10 = (__int64 *)(v5 + 16LL * v9);
    v11 = *v10;
    if ( v3 == *v10 )
      goto LABEL_10;
    ++v70;
  }
  if ( !v71 )
    v71 = v10;
  v72 = *(_DWORD *)(v6 + 56);
  ++*(_QWORD *)(v6 + 40);
  v73 = v72 + 1;
  if ( 4 * (v72 + 1) >= (unsigned int)(3 * a2) )
  {
    sub_1C04E30(v6 + 40, 2 * a2);
    v74 = *(_DWORD *)(v6 + 64);
    if ( !v74 )
      goto LABEL_153;
    v75 = v74 - 1;
    v76 = *(_QWORD *)(v6 + 48);
    LODWORD(v77) = v75 & v8;
    v16 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
    v73 = *(_DWORD *)(v6 + 56) + 1;
    v71 = (__int64 *)(v76 + 16LL * (unsigned int)v77);
    v78 = *v71;
    if ( v3 == *v71 )
      goto LABEL_97;
    v79 = 1;
    v80 = 0;
    while ( v78 != -8 )
    {
      if ( v78 == -16 && !v80 )
        v80 = v71;
      v77 = v75 & (unsigned int)(v77 + v79);
      v71 = (__int64 *)(v76 + 16 * v77);
      v78 = *v71;
      if ( v3 == *v71 )
        goto LABEL_97;
      ++v79;
    }
  }
  else
  {
    if ( (int)a2 - *(_DWORD *)(v6 + 60) - v73 > (unsigned int)a2 >> 3 )
      goto LABEL_97;
    sub_1C04E30(v6 + 40, a2);
    v88 = *(_DWORD *)(v6 + 64);
    if ( !v88 )
      goto LABEL_153;
    v89 = v88 - 1;
    v90 = *(_QWORD *)(v6 + 48);
    v91 = 1;
    LODWORD(v92) = v89 & v8;
    v16 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
    v73 = *(_DWORD *)(v6 + 56) + 1;
    v80 = 0;
    v71 = (__int64 *)(v90 + 16LL * (unsigned int)v92);
    v93 = *v71;
    if ( v3 == *v71 )
      goto LABEL_97;
    while ( v93 != -8 )
    {
      if ( !v80 && v93 == -16 )
        v80 = v71;
      v92 = v89 & (unsigned int)(v92 + v91);
      v71 = (__int64 *)(v90 + 16 * v92);
      v93 = *v71;
      if ( v3 == *v71 )
        goto LABEL_97;
      ++v91;
    }
  }
  if ( v80 )
    v71 = v80;
LABEL_97:
  *(_DWORD *)(v6 + 56) = v73;
  if ( *v71 != -8 )
    --*(_DWORD *)(v6 + 60);
  *v71 = v3;
  v19 = 0;
  v71[1] = 0;
  v5 = *(_QWORD *)(v6 + 48);
  a2 = *(unsigned int *)(v6 + 64);
  v13 = (__int64 *)(v5 + 16 * a2);
  if ( !(_DWORD)a2 )
    goto LABEL_31;
  v4 = a2 - 1;
LABEL_11:
  v20 = v4 & v16;
  v21 = (__int64 *)(v5 + 16LL * (v4 & v16));
  v22 = *v21;
  if ( a3 != *v21 )
  {
    v106 = v4 & v16;
    v24 = *v21;
    for ( j = 1; ; j = v103 )
    {
      if ( v24 == -8 )
        goto LABEL_31;
      v26 = j + 1;
      v27 = v4 & (unsigned int)(v106 + j);
      v103 = v26;
      v106 = v27;
      v28 = (__int64 *)(v5 + 16 * v27);
      v24 = *v28;
      v100 = v28;
      if ( a3 == *v28 )
        break;
    }
    v29 = 1;
    v30 = 0;
    if ( v13 == v100 )
      goto LABEL_31;
    while ( v22 != -8 )
    {
      if ( v22 == -16 && !v30 )
        v30 = v21;
      v20 = v4 & (v29 + v20);
      v21 = (__int64 *)(v5 + 16LL * v20);
      v22 = *v21;
      if ( a3 == *v21 )
        goto LABEL_13;
      ++v29;
    }
    v31 = v6 + 40;
    if ( !v30 )
      v30 = v21;
    v32 = *(_DWORD *)(v6 + 56);
    ++*(_QWORD *)(v6 + 40);
    v33 = v32 + 1;
    if ( 4 * v33 >= (unsigned int)(3 * a2) )
    {
      v109 = v16;
      sub_1C04E30(v31, 2 * a2);
      v81 = *(_DWORD *)(v6 + 64);
      if ( v81 )
      {
        v82 = v81 - 1;
        v83 = *(_QWORD *)(v6 + 48);
        v33 = *(_DWORD *)(v6 + 56) + 1;
        v84 = v82 & v109;
        v30 = (__int64 *)(v83 + 16 * v84);
        v85 = *v30;
        if ( a3 == *v30 )
          goto LABEL_28;
        v86 = 1;
        v87 = 0;
        while ( v85 != -8 )
        {
          if ( v85 == -16 && !v87 )
            v87 = v30;
          v84 = v82 & (unsigned int)(v84 + v86);
          v30 = (__int64 *)(v83 + 16 * v84);
          v85 = *v30;
          if ( a3 == *v30 )
            goto LABEL_28;
          ++v86;
        }
LABEL_112:
        if ( v87 )
          v30 = v87;
        goto LABEL_28;
      }
    }
    else
    {
      if ( (int)a2 - (v33 + *(_DWORD *)(v6 + 60)) > (unsigned int)a2 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(v6 + 56) = v33;
        if ( *v30 != -8 )
          --*(_DWORD *)(v6 + 60);
        *v30 = a3;
        v30[1] = 0;
        goto LABEL_31;
      }
      v110 = v16;
      sub_1C04E30(v31, a2);
      v94 = *(_DWORD *)(v6 + 64);
      if ( v94 )
      {
        v95 = v94 - 1;
        v96 = *(_QWORD *)(v6 + 48);
        v87 = 0;
        v97 = 1;
        v33 = *(_DWORD *)(v6 + 56) + 1;
        v98 = v95 & v110;
        v30 = (__int64 *)(v96 + 16 * v98);
        v99 = *v30;
        if ( a3 == *v30 )
          goto LABEL_28;
        while ( v99 != -8 )
        {
          if ( v99 == -16 && !v87 )
            v87 = v30;
          v98 = v95 & (unsigned int)(v98 + v97);
          v30 = (__int64 *)(v96 + 16 * v98);
          v99 = *v30;
          if ( a3 == *v30 )
            goto LABEL_28;
          ++v97;
        }
        goto LABEL_112;
      }
    }
LABEL_153:
    ++*(_DWORD *)(v6 + 56);
    BUG();
  }
  if ( v13 != v21 )
  {
LABEL_13:
    v14 = v21[1];
    goto LABEL_14;
  }
LABEL_31:
  v14 = 0;
LABEL_14:
  v111[0] = 0;
  if ( (unsigned __int8)sub_1C08AB0(v6, v19, v14, v111) )
  {
    v23 = v111[0];
    if ( !v111[0] )
      goto LABEL_18;
    if ( (v111[0] & 1) == 0 )
      goto LABEL_17;
    if ( !(unsigned int)sub_1C04FF0(v6, v3) )
    {
      v23 = v111[0];
LABEL_17:
      if ( (v23 & 2) == 0 )
        goto LABEL_18;
      LOBYTE(v14) = (unsigned int)sub_1C04FF0(v6, a3) == 0;
      return (unsigned int)v14;
    }
LABEL_4:
    LODWORD(v14) = 0;
    return (unsigned int)v14;
  }
  v34 = *(_QWORD *)(v19 + 72);
  v35 = *(_QWORD *)(v14 + 72);
  if ( !v34 )
    goto LABEL_35;
  v36 = *(__int64 ***)(v34 + 8);
  v37 = &v36[*(unsigned int *)(v34 + 24)];
  if ( !*(_DWORD *)(v34 + 16) || v36 == v37 )
    goto LABEL_35;
  while ( 1 )
  {
    v45 = *v36 + 2 == 0;
    if ( !v45 && *v36 + 1 != 0 )
      break;
    if ( ++v36 == v37 )
      goto LABEL_35;
  }
  if ( v36 == v37 )
  {
LABEL_35:
    if ( !v35 )
      goto LABEL_18;
    goto LABEL_36;
  }
  v107 = v45 || *v36 + 1 == 0;
  v46 = v6;
  v47 = v35;
  while ( 1 )
  {
    v48 = *v36;
    if ( !v47 )
      break;
    v49 = *(_DWORD *)(v47 + 24);
    if ( !v49 )
      break;
    v50 = v49 - 1;
    v51 = *(_QWORD *)(v47 + 8);
    v52 = (v49 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v53 = *(__int64 **)(v51 + 8LL * v52);
    if ( v48 == v53 )
      goto LABEL_55;
    for ( k = 1; ; ++k )
    {
      if ( v53 == (__int64 *)-8LL )
        goto LABEL_61;
      v52 = v50 & (k + v52);
      v53 = *(__int64 **)(v51 + 8LL * v52);
      if ( v48 == v53 )
        break;
    }
    do
    {
LABEL_55:
      if ( ++v36 == v37 )
        goto LABEL_59;
    }
    while ( *v36 == (__int64 *)-16LL || *v36 == (__int64 *)-8LL );
    if ( v36 == v37 )
    {
LABEL_59:
      v35 = v47;
      v6 = v46;
      if ( !v35 )
        goto LABEL_18;
LABEL_36:
      v38 = *(__int64 ***)(v35 + 8);
      v39 = &v38[*(unsigned int *)(v35 + 24)];
      v40 = *(_DWORD *)(v35 + 16);
      if ( !v40 || v39 == v38 )
        goto LABEL_18;
      while ( 1 )
      {
        LOBYTE(v40) = *v38 + 1 == 0 || *v38 + 2 == 0;
        if ( !(_BYTE)v40 )
          break;
        if ( ++v38 == v39 )
          goto LABEL_18;
      }
      LODWORD(v14) = v40;
      if ( v38 == v39 )
      {
LABEL_18:
        LODWORD(v14) = 1;
        return (unsigned int)v14;
      }
      v54 = *v38;
      if ( !v34 )
        goto LABEL_72;
LABEL_65:
      v55 = *(_DWORD *)(v34 + 24);
      if ( !v55 )
        goto LABEL_72;
      v56 = v55 - 1;
      v57 = *(_QWORD *)(v34 + 8);
      v58 = (v55 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v59 = *(__int64 **)(v57 + 8LL * v58);
      if ( v54 != v59 )
      {
        v64 = 1;
        while ( v59 != (__int64 *)-8LL )
        {
          v58 = v56 & (v64 + v58);
          v59 = *(__int64 **)(v57 + 8LL * v58);
          if ( v54 == v59 )
            goto LABEL_67;
          ++v64;
        }
        goto LABEL_72;
      }
      while ( 1 )
      {
        do
        {
LABEL_67:
          if ( ++v38 == v39 )
            goto LABEL_18;
        }
        while ( *v38 == (__int64 *)-16LL || *v38 == (__int64 *)-8LL );
        if ( v38 == v39 )
          goto LABEL_18;
        v54 = *v38;
        if ( v34 )
          goto LABEL_65;
LABEL_72:
        if ( (unsigned int)sub_1C04FF0(v6, *v54) )
          return (unsigned int)v14;
      }
    }
  }
LABEL_61:
  if ( !(unsigned int)sub_1C04FF0(v46, *v48) )
    goto LABEL_55;
  LODWORD(v14) = v107;
  return (unsigned int)v14;
}
