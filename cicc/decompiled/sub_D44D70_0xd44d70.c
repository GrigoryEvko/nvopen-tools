// Function: sub_D44D70
// Address: 0xd44d70
//
__int64 __fastcall sub_D44D70(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void **v7; // rax
  void **v8; // rdx
  void **v10; // rdx
  void **v11; // rcx
  void **v12; // rax
  __int64 v13; // r15
  char v14; // di
  __int64 v15; // rsi
  int v16; // edx
  unsigned int v17; // ecx
  __int64 v18; // rax
  void *v19; // r8
  __int64 v20; // rdx
  __int64 v21; // r15
  char v22; // di
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // edx
  unsigned int v26; // ecx
  __int64 v27; // rax
  void *v28; // r8
  __int64 v29; // rdx
  __int64 v30; // r15
  char v31; // si
  __int64 v32; // rcx
  int v33; // edi
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rax
  void *v38; // r8
  __int64 v39; // rdx
  __int64 v40; // r15
  char v41; // si
  __int64 v42; // rcx
  int v43; // edi
  unsigned int v44; // edx
  __int64 v45; // rax
  void *v46; // r8
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // r9d
  unsigned int i; // eax
  __int64 v53; // rsi
  unsigned int v54; // eax
  void **v55; // rcx
  __int64 v56; // rax
  int v57; // eax
  void **v58; // rcx
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 (__fastcall *v61)(__int64, __int64, __int64, __int64 *); // rax
  char v62; // al
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // r9d
  unsigned int j; // eax
  __int64 v68; // rsi
  unsigned int v69; // eax
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  int v76; // r8d
  unsigned int k; // eax
  __int64 v78; // rsi
  unsigned int v79; // eax
  __int64 v80; // rdi
  __int64 (__fastcall *v81)(__int64); // rax
  char v82; // al
  int v83; // eax
  int v84; // r9d
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rcx
  int v88; // edi
  unsigned int m; // eax
  __int64 v90; // rsi
  unsigned int v91; // eax
  __int64 v92; // rdi
  __int64 (__fastcall *v93)(__int64); // rax
  char v94; // al
  __int64 v95; // rax
  int v96; // eax
  int v97; // r9d
  __int64 v98; // rdi
  __int64 (__fastcall *v99)(__int64, __int64, __int64, __int64); // rax
  char v100; // al
  int v101; // r9d
  int v102; // r9d
  void *v103; // [rsp+0h] [rbp-70h] BYREF
  char v104[8]; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v105[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v106; // [rsp+20h] [rbp-50h]

  if ( *(_BYTE *)(a3 + 76) )
  {
    v7 = *(void ***)(a3 + 56);
    v8 = &v7[*(unsigned int *)(a3 + 68)];
    if ( v7 != v8 )
    {
      while ( *v7 != &unk_4F86D28 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4F86D28) )
  {
    return 1;
  }
LABEL_8:
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(void ***)(a3 + 8);
    v11 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v10 != v11 )
    {
      v12 = *(void ***)(a3 + 8);
      while ( *v12 != &unk_4F82400 )
      {
        if ( v11 == ++v12 )
          goto LABEL_69;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&unk_4F82400) )
    goto LABEL_13;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(void ***)(a3 + 8);
    v12 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v12 != v10 )
    {
LABEL_69:
      v58 = v10;
      while ( *v58 != &unk_4F86D28 )
      {
        if ( ++v58 == v12 )
          goto LABEL_57;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&unk_4F86D28) )
    goto LABEL_13;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(void ***)(a3 + 8);
    v12 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v10 != v12 )
    {
LABEL_57:
      v55 = v10;
      while ( *v10 != &unk_4F82400 )
      {
        if ( ++v10 == v12 )
          goto LABEL_95;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&unk_4F82400) )
    goto LABEL_13;
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( sub_C8CA60(a3, (__int64)&unk_4F82420) )
      goto LABEL_13;
    return 1;
  }
  v55 = *(void ***)(a3 + 8);
  v12 = &v55[*(unsigned int *)(a3 + 20)];
  if ( v55 == v12 )
    return 1;
LABEL_95:
  while ( *v55 != &unk_4F82420 )
  {
    if ( ++v55 == v12 )
      return 1;
  }
LABEL_13:
  v13 = *a4;
  v14 = *(_BYTE *)(*a4 + 8) & 1;
  if ( v14 )
  {
    v15 = v13 + 16;
    v16 = 7;
  }
  else
  {
    v34 = *(unsigned int *)(v13 + 24);
    v15 = *(_QWORD *)(v13 + 16);
    if ( !(_DWORD)v34 )
      goto LABEL_62;
    v16 = v34 - 1;
  }
  v17 = v16 & (((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4));
  v18 = v15 + 16LL * v17;
  v19 = *(void **)v18;
  if ( *(_UNKNOWN **)v18 == &unk_4F86540 )
    goto LABEL_16;
  v57 = 1;
  while ( v19 != (void *)-4096LL )
  {
    v84 = v57 + 1;
    v17 = v16 & (v57 + v17);
    v18 = v15 + 16LL * v17;
    v19 = *(void **)v18;
    if ( *(_UNKNOWN **)v18 == &unk_4F86540 )
      goto LABEL_16;
    v57 = v84;
  }
  if ( v14 )
  {
    v56 = 128;
    goto LABEL_63;
  }
  v34 = *(unsigned int *)(v13 + 24);
LABEL_62:
  v56 = 16 * v34;
LABEL_63:
  v18 = v15 + v56;
LABEL_16:
  v20 = 128;
  if ( !v14 )
    v20 = 16LL * *(unsigned int *)(v13 + 24);
  if ( v18 == v15 + v20 )
  {
    v48 = a4[1];
    v49 = *(unsigned int *)(v48 + 24);
    v50 = *(_QWORD *)(v48 + 8);
    if ( (_DWORD)v49 )
    {
      v51 = 1;
      for ( i = (v49 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v49 - 1) & v54 )
      {
        v53 = v50 + 24LL * i;
        if ( *(_UNKNOWN **)v53 == &unk_4F86540 && a2 == *(_QWORD *)(v53 + 8) )
          break;
        if ( *(_QWORD *)v53 == -4096 && *(_QWORD *)(v53 + 8) == -4096 )
          goto LABEL_77;
        v54 = v51 + i;
        ++v51;
      }
    }
    else
    {
LABEL_77:
      v53 = v50 + 24 * v49;
    }
    v60 = *(_QWORD *)(*(_QWORD *)(v53 + 16) + 24LL);
    v61 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v60 + 16LL);
    if ( v61 == sub_D32140 )
      v62 = sub_CF8780(v60 + 8, a2, a3, a4);
    else
      v62 = v61(v60, a2, a3, a4);
    v104[0] = v62;
    v103 = &unk_4F86540;
    sub_BBCF50((__int64)v105, v13, (__int64 *)&v103, v104);
    v18 = v106;
  }
  if ( *(_BYTE *)(v18 + 8) )
    return 1;
  v21 = *a4;
  v22 = *(_BYTE *)(*a4 + 8) & 1;
  if ( v22 )
  {
    v24 = v21 + 16;
    v25 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(v21 + 24);
    v24 = *(_QWORD *)(v21 + 16);
    if ( !(_DWORD)v23 )
      goto LABEL_98;
    v25 = v23 - 1;
  }
  v26 = v25 & (((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4));
  v27 = v24 + 16LL * v26;
  v28 = *(void **)v27;
  if ( *(_UNKNOWN **)v27 == &unk_4F881D0 )
    goto LABEL_24;
  v71 = 1;
  while ( v28 != (void *)-4096LL )
  {
    v97 = v71 + 1;
    v26 = v25 & (v71 + v26);
    v27 = v24 + 16LL * v26;
    v28 = *(void **)v27;
    if ( *(_UNKNOWN **)v27 == &unk_4F881D0 )
      goto LABEL_24;
    v71 = v97;
  }
  if ( v22 )
  {
    v70 = 128;
    goto LABEL_99;
  }
  v23 = *(unsigned int *)(v21 + 24);
LABEL_98:
  v70 = 16 * v23;
LABEL_99:
  v27 = v24 + v70;
LABEL_24:
  v29 = 128;
  if ( !v22 )
    v29 = 16LL * *(unsigned int *)(v21 + 24);
  if ( v27 == v24 + v29 )
  {
    v63 = a4[1];
    v64 = *(unsigned int *)(v63 + 24);
    v65 = *(_QWORD *)(v63 + 8);
    if ( (_DWORD)v64 )
    {
      v66 = 1;
      for ( j = (v64 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v64 - 1) & v69 )
      {
        v68 = v65 + 24LL * j;
        if ( *(_UNKNOWN **)v68 == &unk_4F881D0 && a2 == *(_QWORD *)(v68 + 8) )
          break;
        if ( *(_QWORD *)v68 == -4096 && *(_QWORD *)(v68 + 8) == -4096 )
          goto LABEL_113;
        v69 = v66 + j;
        ++v66;
      }
    }
    else
    {
LABEL_113:
      v68 = v65 + 24 * v64;
    }
    v80 = *(_QWORD *)(*(_QWORD *)(v68 + 16) + 24LL);
    v81 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v80 + 16LL);
    if ( v81 == sub_D32150 )
      v82 = sub_DF3010(v80 + 8);
    else
      v82 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v81)(v80, a2, a3, a4);
    v104[0] = v82;
    v103 = &unk_4F881D0;
    sub_BBCF50((__int64)v105, v21, (__int64 *)&v103, v104);
    v27 = v106;
  }
  if ( *(_BYTE *)(v27 + 8) )
    return 1;
  v30 = *a4;
  v31 = *(_BYTE *)(*a4 + 8) & 1;
  if ( v31 )
  {
    v32 = v30 + 16;
    v33 = 7;
  }
  else
  {
    v35 = *(unsigned int *)(v30 + 24);
    v32 = *(_QWORD *)(v30 + 16);
    if ( !(_DWORD)v35 )
      goto LABEL_74;
    v33 = v35 - 1;
  }
  v36 = v33 & (((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4));
  v37 = v32 + 16LL * v36;
  v38 = *(void **)v37;
  if ( *(_UNKNOWN **)v37 == &unk_4F875F0 )
    goto LABEL_36;
  v83 = 1;
  while ( v38 != (void *)-4096LL )
  {
    v101 = v83 + 1;
    v36 = v33 & (v83 + v36);
    v37 = v32 + 16LL * v36;
    v38 = *(void **)v37;
    if ( *(_UNKNOWN **)v37 == &unk_4F875F0 )
      goto LABEL_36;
    v83 = v101;
  }
  if ( v31 )
  {
    v59 = 128;
    goto LABEL_75;
  }
  v35 = *(unsigned int *)(v30 + 24);
LABEL_74:
  v59 = 16 * v35;
LABEL_75:
  v37 = v32 + v59;
LABEL_36:
  v39 = 128;
  if ( !v31 )
    v39 = 16LL * *(unsigned int *)(v30 + 24);
  if ( v37 == v32 + v39 )
  {
    v73 = a4[1];
    v74 = *(unsigned int *)(v73 + 24);
    v75 = *(_QWORD *)(v73 + 8);
    if ( (_DWORD)v74 )
    {
      v76 = 1;
      for ( k = (v74 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; k = (v74 - 1) & v79 )
      {
        v78 = v75 + 24LL * k;
        if ( *(_UNKNOWN **)v78 == &unk_4F875F0 && a2 == *(_QWORD *)(v78 + 8) )
          break;
        if ( *(_QWORD *)v78 == -4096 && *(_QWORD *)(v78 + 8) == -4096 )
          goto LABEL_132;
        v79 = v76 + k;
        ++v76;
      }
    }
    else
    {
LABEL_132:
      v78 = v75 + 24 * v74;
    }
    v92 = *(_QWORD *)(*(_QWORD *)(v78 + 16) + 24LL);
    v93 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v92 + 16LL);
    if ( v93 == sub_D32160 )
      v94 = sub_D49500(v92 + 8);
    else
      v94 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v93)(v92, a2, a3, a4);
    v104[0] = v94;
    v103 = &unk_4F875F0;
    sub_BBCF50((__int64)v105, v30, (__int64 *)&v103, v104);
    v37 = v106;
  }
  if ( *(_BYTE *)(v37 + 8) )
    return 1;
  v40 = *a4;
  v41 = *(_BYTE *)(*a4 + 8) & 1;
  if ( v41 )
  {
    v42 = v40 + 16;
    v43 = 7;
  }
  else
  {
    v72 = *(unsigned int *)(v40 + 24);
    v42 = *(_QWORD *)(v40 + 16);
    if ( !(_DWORD)v72 )
      goto LABEL_139;
    v43 = v72 - 1;
  }
  v44 = v43 & (((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4));
  v45 = v42 + 16LL * v44;
  v46 = *(void **)v45;
  if ( *(_UNKNOWN **)v45 != &unk_4F81450 )
  {
    v96 = 1;
    while ( v46 != (void *)-4096LL )
    {
      v102 = v96 + 1;
      v44 = v43 & (v96 + v44);
      v45 = v42 + 16LL * v44;
      v46 = *(void **)v45;
      if ( *(_UNKNOWN **)v45 == &unk_4F81450 )
        goto LABEL_43;
      v96 = v102;
    }
    if ( v41 )
    {
      v95 = 128;
      goto LABEL_140;
    }
    v72 = *(unsigned int *)(v40 + 24);
LABEL_139:
    v95 = 16 * v72;
LABEL_140:
    v45 = v42 + v95;
  }
LABEL_43:
  v47 = 128;
  if ( !v41 )
    v47 = 16LL * *(unsigned int *)(v40 + 24);
  if ( v45 == v42 + v47 )
  {
    v85 = a4[1];
    v86 = *(unsigned int *)(v85 + 24);
    v87 = *(_QWORD *)(v85 + 8);
    if ( (_DWORD)v86 )
    {
      v88 = 1;
      for ( m = (v86 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; m = (v86 - 1) & v91 )
      {
        v90 = v87 + 24LL * m;
        if ( *(_UNKNOWN **)v90 == &unk_4F81450 && a2 == *(_QWORD *)(v90 + 8) )
          break;
        if ( *(_QWORD *)v90 == -4096 && *(_QWORD *)(v90 + 8) == -4096 )
          goto LABEL_153;
        v91 = v88 + m;
        ++v88;
      }
    }
    else
    {
LABEL_153:
      v90 = v87 + 24 * v86;
    }
    v98 = *(_QWORD *)(*(_QWORD *)(v90 + 16) + 24LL);
    v99 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v98 + 16LL);
    if ( v99 == sub_D00100 )
      v100 = sub_B19B20(v98 + 8, a2, a3, (__int64)a4);
    else
      v100 = v99(v98, a2, a3, (__int64)a4);
    v104[0] = v100;
    v103 = &unk_4F81450;
    sub_BBCF50((__int64)v105, v40, (__int64 *)&v103, v104);
    v45 = v106;
  }
  return *(unsigned __int8 *)(v45 + 8);
}
