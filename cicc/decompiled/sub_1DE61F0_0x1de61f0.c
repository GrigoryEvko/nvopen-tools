// Function: sub_1DE61F0
// Address: 0x1de61f0
//
__int64 __fastcall sub_1DE61F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  _QWORD *v5; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 *v10; // r11
  __int64 *v11; // r10
  __int64 *v12; // r13
  __int64 v13; // r8
  int v14; // edi
  __int64 *v15; // r15
  int v16; // esi
  int v17; // ebx
  __int64 v18; // r9
  __int64 *v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r14
  __int64 v23; // r9
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r14
  unsigned int v27; // ecx
  __int64 *v28; // rax
  __int64 v29; // r9
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // r9
  int v34; // eax
  int v35; // r14d
  int v36; // eax
  int v37; // eax
  int v38; // eax
  int v39; // r14d
  __int64 *v40; // rdx
  int v41; // r10d
  __int64 v42; // r8
  unsigned int v43; // ecx
  __int64 *v44; // rax
  __int64 v45; // r11
  int v46; // eax
  __int64 v47; // rsi
  __int64 v48; // rax
  size_t v49; // rdx
  char *v50; // r13
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 *v53; // r14
  __int64 v54; // r12
  __int64 v55; // r9
  unsigned int v56; // edi
  _QWORD *v57; // rax
  __int64 v58; // rcx
  unsigned int v59; // esi
  int v60; // ecx
  int v61; // ecx
  __int64 v62; // rdi
  unsigned int v63; // r10d
  int v64; // eax
  _QWORD *v65; // rdx
  __int64 v66; // rsi
  unsigned __int64 v67; // rax
  int v70; // r8d
  int v71; // eax
  int v72; // ecx
  int v73; // ecx
  __int64 v74; // rdi
  _QWORD *v75; // r11
  unsigned int v76; // r15d
  int v77; // r10d
  __int64 v78; // rsi
  int v79; // eax
  int v80; // r13d
  __int64 v81; // rcx
  int v82; // eax
  int v83; // edi
  unsigned int v84; // edx
  __int64 *v85; // rax
  __int64 v86; // r8
  unsigned int v87; // esi
  __int64 *v88; // rdx
  __int64 v89; // r10
  int v90; // r15d
  int v91; // eax
  int v92; // r9d
  int v93; // edx
  int v94; // r9d
  int v95; // [rsp+8h] [rbp-68h]
  int v96; // [rsp+8h] [rbp-68h]
  __int64 v97; // [rsp+8h] [rbp-68h]
  _QWORD *v98; // [rsp+10h] [rbp-60h]
  char v99; // [rsp+10h] [rbp-60h]
  __int64 v101; // [rsp+18h] [rbp-58h]
  __int64 *v103; // [rsp+20h] [rbp-50h]
  __int64 *src; // [rsp+28h] [rbp-48h]
  void *srca; // [rsp+28h] [rbp-48h]
  __int64 v106; // [rsp+30h] [rbp-40h] BYREF
  __int64 v107[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = a1;
  v5 = *(_QWORD **)a3;
  v6 = 8LL * *(unsigned int *)(a3 + 8);
  src = (__int64 *)(*(_QWORD *)a3 + v6);
  v7 = v6 >> 3;
  v8 = v6 >> 5;
  if ( v8 )
  {
    v98 = v5;
    v9 = v5 + 3;
    v10 = v5 + 2;
    v11 = v5 + 1;
    v12 = v5;
    v13 = *(_QWORD *)(a1 + 896);
    v14 = *(_DWORD *)(a1 + 912);
    v15 = &v5[4 * v8];
    v16 = v14 - 1;
    v17 = v14;
    while ( 1 )
    {
      if ( v17 )
      {
        v30 = *(v9 - 3);
        v31 = v16 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v32 = (__int64 *)(v13 + 16LL * v31);
        v33 = *v32;
        if ( *v32 == v30 )
        {
LABEL_3:
          if ( a2 == v32[1] )
          {
            v4 = a1;
            v19 = v12;
            goto LABEL_30;
          }
        }
        else
        {
          v34 = 1;
          while ( v33 != -8 )
          {
            v35 = v34 + 1;
            v31 = v16 & (v34 + v31);
            v32 = (__int64 *)(v13 + 16LL * v31);
            v33 = *v32;
            if ( v30 == *v32 )
              goto LABEL_3;
            v34 = v35;
          }
        }
        v18 = *(v9 - 2);
        v19 = v11;
        v20 = v16 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v21 = (__int64 *)(v13 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == v18 )
        {
LABEL_5:
          if ( a2 == v21[1] )
            goto LABEL_29;
        }
        else
        {
          v36 = 1;
          while ( v22 != -8 )
          {
            v20 = v16 & (v36 + v20);
            v95 = v36 + 1;
            v21 = (__int64 *)(v13 + 16LL * v20);
            v22 = *v21;
            if ( *v21 == v18 )
              goto LABEL_5;
            v36 = v95;
          }
        }
        v23 = *(v9 - 1);
        v19 = v10;
        v24 = v16 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v25 = (__int64 *)(v13 + 16LL * v24);
        v26 = *v25;
        if ( *v25 == v23 )
        {
LABEL_7:
          if ( a2 == v25[1] )
          {
LABEL_29:
            v4 = a1;
            goto LABEL_30;
          }
        }
        else
        {
          v37 = 1;
          while ( v26 != -8 )
          {
            v24 = v16 & (v37 + v24);
            v96 = v37 + 1;
            v25 = (__int64 *)(v13 + 16LL * v24);
            v26 = *v25;
            if ( *v25 == v23 )
              goto LABEL_7;
            v37 = v96;
          }
        }
        v27 = v16 & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
        v28 = (__int64 *)(v13 + 16LL * v27);
        v29 = *v28;
        if ( *v28 == *v9 )
        {
LABEL_9:
          if ( a2 == v28[1] )
          {
            v4 = a1;
            v19 = v9;
            goto LABEL_30;
          }
        }
        else
        {
          v38 = 1;
          while ( v29 != -8 )
          {
            v39 = v38 + 1;
            v27 = v16 & (v38 + v27);
            v28 = (__int64 *)(v13 + 16LL * v27);
            v29 = *v28;
            if ( *v28 == *v9 )
              goto LABEL_9;
            v38 = v39;
          }
        }
      }
      v12 += 4;
      v9 += 4;
      v10 += 4;
      v11 += 4;
      if ( v12 == v15 )
      {
        v5 = v98;
        v4 = a1;
        v7 = src - v12;
        goto LABEL_83;
      }
    }
  }
  v12 = v5;
LABEL_83:
  switch ( v7 )
  {
    case 2LL:
      v81 = *(_QWORD *)(v4 + 896);
      v82 = *(_DWORD *)(v4 + 912);
      goto LABEL_89;
    case 3LL:
      v82 = *(_DWORD *)(v4 + 912);
      v81 = *(_QWORD *)(v4 + 896);
      if ( v82 )
      {
        v87 = (v82 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v88 = (__int64 *)(v81 + 16LL * v87);
        v89 = *v88;
        if ( *v12 == *v88 )
        {
LABEL_99:
          if ( a2 == v88[1] )
            goto LABEL_96;
        }
        else
        {
          v93 = 1;
          while ( v89 != -8 )
          {
            v94 = v93 + 1;
            v87 = (v82 - 1) & (v93 + v87);
            v88 = (__int64 *)(v81 + 16LL * v87);
            v89 = *v88;
            if ( *v12 == *v88 )
              goto LABEL_99;
            v93 = v94;
          }
        }
      }
      ++v12;
LABEL_89:
      if ( !v82 )
        goto LABEL_92;
      v83 = v82 - 1;
      v84 = (v82 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
      v85 = (__int64 *)(v81 + 16LL * v84);
      v86 = *v85;
      if ( *v12 != *v85 )
      {
        v91 = 1;
        while ( v86 != -8 )
        {
          v92 = v91 + 1;
          v84 = v83 & (v91 + v84);
          v85 = (__int64 *)(v81 + 16LL * v84);
          v86 = *v85;
          if ( *v12 == *v85 )
            goto LABEL_91;
          v91 = v92;
        }
        goto LABEL_92;
      }
LABEL_91:
      if ( a2 != v85[1] )
      {
LABEL_92:
        ++v12;
        goto LABEL_93;
      }
LABEL_96:
      v19 = v12;
LABEL_30:
      if ( src == v19 )
      {
        v5 = *(_QWORD **)a3;
        v48 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
        v50 = (char *)v48;
        v49 = v48 - (_QWORD)src;
LABEL_39:
        if ( src != (__int64 *)v48 )
        {
          memmove(v19, src, v49);
          v5 = *(_QWORD **)a3;
        }
        goto LABEL_41;
      }
      v40 = v19 + 1;
      if ( src == v19 + 1 )
      {
LABEL_38:
        v5 = *(_QWORD **)a3;
        v48 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
        v49 = v48 - (_QWORD)src;
        v50 = (char *)v19 + v48 - (_QWORD)src;
        goto LABEL_39;
      }
      while ( 1 )
      {
        v46 = *(_DWORD *)(v4 + 912);
        v47 = *v40;
        if ( !v46 )
          goto LABEL_37;
        v41 = v46 - 1;
        v42 = *(_QWORD *)(v4 + 896);
        v43 = (v46 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v44 = (__int64 *)(v42 + 16LL * v43);
        v45 = *v44;
        if ( v47 != *v44 )
        {
          v79 = 1;
          while ( v45 != -8 )
          {
            v80 = v79 + 1;
            v43 = v41 & (v79 + v43);
            v44 = (__int64 *)(v42 + 16LL * v43);
            v45 = *v44;
            if ( v47 == *v44 )
              goto LABEL_34;
            v79 = v80;
          }
          goto LABEL_37;
        }
LABEL_34:
        if ( a2 == v44[1] )
        {
          if ( src == ++v40 )
            goto LABEL_38;
        }
        else
        {
LABEL_37:
          ++v40;
          *v19++ = v47;
          if ( src == v40 )
            goto LABEL_38;
        }
      }
    case 1LL:
LABEL_93:
      v106 = *v12;
      if ( !(unsigned __int8)sub_1DE30F0(v4 + 888, &v106, v107) || a2 != *(_QWORD *)(v107[0] + 8) )
      {
        v19 = src;
        v5 = *(_QWORD **)a3;
        v48 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
        v50 = (char *)v48;
        v49 = v48 - (_QWORD)src;
        goto LABEL_39;
      }
      goto LABEL_96;
  }
  v50 = (char *)src;
LABEL_41:
  v51 = (v50 - (char *)v5) >> 3;
  *(_DWORD *)(a3 + 8) = v51;
  if ( !(_DWORD)v51 )
    return 0;
  v52 = *v5;
  v101 = a2;
  srca = 0;
  v53 = v5 + 1;
  v54 = 0;
  v103 = &v5[(unsigned int)v51];
  v97 = v4 + 888;
  v99 = *(_BYTE *)(*v5 + 180LL);
  while ( 1 )
  {
    v59 = *(_DWORD *)(v4 + 912);
    if ( !v59 )
    {
      ++*(_QWORD *)(v4 + 888);
      goto LABEL_49;
    }
    v55 = *(_QWORD *)(v4 + 896);
    v56 = (v59 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    v57 = (_QWORD *)(v55 + 16LL * v56);
    v58 = *v57;
    if ( *v57 != v52 )
    {
      v70 = 1;
      v65 = 0;
      while ( v58 != -8 )
      {
        if ( v58 == -16 && !v65 )
          v65 = v57;
        v56 = (v59 - 1) & (v70 + v56);
        v57 = (_QWORD *)(v55 + 16LL * v56);
        v58 = *v57;
        if ( *v57 == v52 )
          goto LABEL_44;
        ++v70;
      }
      if ( !v65 )
        v65 = v57;
      v71 = *(_DWORD *)(v4 + 904);
      ++*(_QWORD *)(v4 + 888);
      v64 = v71 + 1;
      if ( 4 * v64 < 3 * v59 )
      {
        if ( v59 - *(_DWORD *)(v4 + 908) - v64 > v59 >> 3 )
          goto LABEL_51;
        sub_1DE4DF0(v97, v59);
        v72 = *(_DWORD *)(v4 + 912);
        if ( !v72 )
        {
LABEL_126:
          ++*(_DWORD *)(v4 + 904);
          BUG();
        }
        v73 = v72 - 1;
        v74 = *(_QWORD *)(v4 + 896);
        v75 = 0;
        v76 = v73 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        v77 = 1;
        v64 = *(_DWORD *)(v4 + 904) + 1;
        v65 = (_QWORD *)(v74 + 16LL * v76);
        v78 = *v65;
        if ( *v65 == v52 )
          goto LABEL_51;
        while ( v78 != -8 )
        {
          if ( !v75 && v78 == -16 )
            v75 = v65;
          v76 = v73 & (v77 + v76);
          v65 = (_QWORD *)(v74 + 16LL * v76);
          v78 = *v65;
          if ( *v65 == v52 )
            goto LABEL_51;
          ++v77;
        }
        goto LABEL_103;
      }
LABEL_49:
      sub_1DE4DF0(v97, 2 * v59);
      v60 = *(_DWORD *)(v4 + 912);
      if ( !v60 )
        goto LABEL_126;
      v61 = v60 - 1;
      v62 = *(_QWORD *)(v4 + 896);
      v63 = v61 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      v64 = *(_DWORD *)(v4 + 904) + 1;
      v65 = (_QWORD *)(v62 + 16LL * v63);
      v66 = *v65;
      if ( *v65 == v52 )
        goto LABEL_51;
      v90 = 1;
      v75 = 0;
      while ( v66 != -8 )
      {
        if ( !v75 && v66 == -16 )
          v75 = v65;
        v63 = v61 & (v90 + v63);
        v65 = (_QWORD *)(v62 + 16LL * v63);
        v66 = *v65;
        if ( *v65 == v52 )
          goto LABEL_51;
        ++v90;
      }
LABEL_103:
      if ( v75 )
        v65 = v75;
LABEL_51:
      *(_DWORD *)(v4 + 904) = v64;
      if ( *v65 != -8 )
        --*(_DWORD *)(v4 + 908);
      *v65 = v52;
      v65[1] = 0;
      goto LABEL_54;
    }
LABEL_44:
    if ( v101 == v57[1] )
      goto LABEL_45;
LABEL_54:
    v67 = sub_20D7490(*(_QWORD *)(v4 + 568), v52);
    if ( v54 )
      break;
    srca = (void *)v67;
    v54 = v52;
LABEL_45:
    if ( v103 == v53 )
      return v54;
LABEL_46:
    v52 = *v53++;
  }
  if ( v99 == v67 <= (unsigned __int64)srca )
    v54 = v52;
  else
    v67 = (unsigned __int64)srca;
  srca = (void *)v67;
  if ( v103 != v53 )
    goto LABEL_46;
  return v54;
}
