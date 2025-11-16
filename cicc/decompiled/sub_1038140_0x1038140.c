// Function: sub_1038140
// Address: 0x1038140
//
__int64 __fastcall sub_1038140(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
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
  char v21; // di
  __int64 v22; // r15
  char v23; // si
  __int64 v24; // r8
  int v25; // edx
  unsigned int v26; // ecx
  __int64 v27; // rax
  void *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // r15
  char v31; // si
  __int64 v32; // rcx
  int v33; // edi
  unsigned int v34; // edx
  __int64 v35; // rax
  void *v36; // r8
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  int v43; // r9d
  unsigned int i; // eax
  __int64 v45; // rsi
  unsigned int v46; // eax
  __int64 v47; // rax
  void **v48; // rcx
  __int64 v49; // rax
  int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  int v54; // r10d
  unsigned int j; // eax
  __int64 v56; // rsi
  unsigned int v57; // eax
  void **v58; // rcx
  __int64 (*v59)(); // rax
  __int64 v60; // rdi
  __int64 (__fastcall *v61)(__int64, __int64, __int64, __int64 *); // rax
  char v62; // al
  __int64 v63; // rax
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  int v68; // r9d
  unsigned int k; // eax
  __int64 v70; // rsi
  unsigned int v71; // eax
  __int64 v72; // rax
  int v73; // eax
  __int64 v74; // rdi
  __int64 (__fastcall *v75)(__int64, __int64, __int64, __int64); // rax
  char v76; // al
  int v77; // r9d
  int v78; // r10d
  int v79; // r9d
  void *v80; // [rsp+0h] [rbp-70h] BYREF
  char v81[8]; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v82[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v83; // [rsp+20h] [rbp-50h]

  if ( *(_BYTE *)(a3 + 76) )
  {
    v7 = *(void ***)(a3 + 56);
    v8 = &v7[*(unsigned int *)(a3 + 68)];
    if ( v7 != v8 )
    {
      while ( *v7 != &unk_4F8EE60 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4F8EE60) )
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
          goto LABEL_67;
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
    if ( v10 != v12 )
    {
LABEL_67:
      v58 = v10;
      while ( *v58 != &unk_4F8EE60 )
      {
        if ( ++v58 == v12 )
          goto LABEL_50;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&unk_4F8EE60) )
    goto LABEL_13;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(void ***)(a3 + 8);
    v12 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v10 != v12 )
    {
LABEL_50:
      v48 = v10;
      while ( *v10 != &unk_4F82400 )
      {
        if ( ++v10 == v12 )
          goto LABEL_96;
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
  v48 = *(void ***)(a3 + 8);
  v12 = &v48[*(unsigned int *)(a3 + 20)];
  if ( v12 == v48 )
    return 1;
LABEL_96:
  while ( *v48 != &unk_4F82420 )
  {
    if ( ++v48 == v12 )
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
    v38 = *(unsigned int *)(v13 + 24);
    v15 = *(_QWORD *)(v13 + 16);
    if ( !(_DWORD)v38 )
      goto LABEL_55;
    v16 = v38 - 1;
  }
  v17 = v16 & (((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4));
  v18 = v15 + 16LL * v17;
  v19 = *(void **)v18;
  if ( *(_UNKNOWN **)v18 == &unk_4F86540 )
    goto LABEL_16;
  v50 = 1;
  while ( v19 != (void *)-4096LL )
  {
    v77 = v50 + 1;
    v17 = v16 & (v50 + v17);
    v18 = v15 + 16LL * v17;
    v19 = *(void **)v18;
    if ( *(_UNKNOWN **)v18 == &unk_4F86540 )
      goto LABEL_16;
    v50 = v77;
  }
  if ( v14 )
  {
    v49 = 128;
    goto LABEL_56;
  }
  v38 = *(unsigned int *)(v13 + 24);
LABEL_55:
  v49 = 16 * v38;
LABEL_56:
  v18 = v15 + v49;
LABEL_16:
  v20 = 128;
  if ( !v14 )
    v20 = 16LL * *(unsigned int *)(v13 + 24);
  if ( v18 == v15 + v20 )
  {
    v40 = a4[1];
    v41 = *(unsigned int *)(v40 + 24);
    v42 = *(_QWORD *)(v40 + 8);
    if ( (_DWORD)v41 )
    {
      v43 = 1;
      for ( i = (v41 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v41 - 1) & v46 )
      {
        v45 = v42 + 24LL * i;
        if ( *(_UNKNOWN **)v45 == &unk_4F86540 && a2 == *(_QWORD *)(v45 + 8) )
          break;
        if ( *(_QWORD *)v45 == -4096 && *(_QWORD *)(v45 + 8) == -4096 )
          goto LABEL_76;
        v46 = v43 + i;
        ++v43;
      }
    }
    else
    {
LABEL_76:
      v45 = v42 + 24 * v41;
    }
    v60 = *(_QWORD *)(*(_QWORD *)(v45 + 16) + 24LL);
    v61 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v60 + 16LL);
    if ( v61 == sub_D32140 )
      v62 = sub_CF8780(v60 + 8, a2, a3, a4);
    else
      v62 = v61(v60, a2, a3, a4);
    v81[0] = v62;
    v80 = &unk_4F86540;
    sub_BBCF50((__int64)v82, v13, (__int64 *)&v80, v81);
    v18 = v83;
  }
  v21 = *(_BYTE *)(v18 + 8);
  if ( v21 )
    return 1;
  v22 = *a4;
  v23 = *(_BYTE *)(*a4 + 8) & 1;
  if ( v23 )
  {
    v24 = v22 + 16;
    v25 = 7;
  }
  else
  {
    v39 = *(unsigned int *)(v22 + 24);
    v24 = *(_QWORD *)(v22 + 16);
    if ( !(_DWORD)v39 )
      goto LABEL_81;
    v25 = v39 - 1;
  }
  v26 = v25 & (((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4));
  v27 = v24 + 16LL * v26;
  v28 = *(void **)v27;
  if ( *(_UNKNOWN **)v27 == &unk_4F86630 )
    goto LABEL_23;
  v64 = 1;
  while ( v28 != (void *)-4096LL )
  {
    v78 = v64 + 1;
    v26 = v25 & (v64 + v26);
    v27 = v24 + 16LL * v26;
    v28 = *(void **)v27;
    if ( *(_UNKNOWN **)v27 == &unk_4F86630 )
      goto LABEL_23;
    v64 = v78;
  }
  if ( v23 )
  {
    v63 = 128;
    goto LABEL_82;
  }
  v39 = *(unsigned int *)(v22 + 24);
LABEL_81:
  v63 = 16 * v39;
LABEL_82:
  v27 = v24 + v63;
LABEL_23:
  v29 = 128;
  if ( !v23 )
    v29 = 16LL * *(unsigned int *)(v22 + 24);
  if ( v27 == v24 + v29 )
  {
    v51 = a4[1];
    v52 = *(unsigned int *)(v51 + 24);
    v53 = *(_QWORD *)(v51 + 8);
    if ( (_DWORD)v52 )
    {
      v54 = 1;
      for ( j = (v52 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v52 - 1) & v57 )
      {
        v56 = v53 + 24LL * j;
        if ( *(_UNKNOWN **)v56 == &unk_4F86630 && a2 == *(_QWORD *)(v56 + 8) )
          break;
        if ( *(_QWORD *)v56 == -4096 && *(_QWORD *)(v56 + 8) == -4096 )
          goto LABEL_108;
        v57 = v54 + j;
        ++v54;
      }
    }
    else
    {
LABEL_108:
      v56 = v53 + 24 * v52;
    }
    v59 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(v56 + 16) + 24LL) + 16LL);
    if ( v59 != sub_D000F0 )
      v21 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64, __int64 *))v59)(
              *(_QWORD *)(*(_QWORD *)(v56 + 16) + 24LL),
              a2,
              a3,
              a4);
    v81[0] = v21;
    v80 = &unk_4F86630;
    sub_BBCF50((__int64)v82, v22, (__int64 *)&v80, v81);
    v27 = v83;
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
    v47 = *(unsigned int *)(v30 + 24);
    v32 = *(_QWORD *)(v30 + 16);
    if ( !(_DWORD)v47 )
      goto LABEL_105;
    v33 = v47 - 1;
  }
  v34 = v33 & (((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4));
  v35 = v32 + 16LL * v34;
  v36 = *(void **)v35;
  if ( *(_UNKNOWN **)v35 != &unk_4F81450 )
  {
    v73 = 1;
    while ( v36 != (void *)-4096LL )
    {
      v79 = v73 + 1;
      v34 = v33 & (v73 + v34);
      v35 = v32 + 16LL * v34;
      v36 = *(void **)v35;
      if ( *(_UNKNOWN **)v35 == &unk_4F81450 )
        goto LABEL_30;
      v73 = v79;
    }
    if ( v31 )
    {
      v72 = 128;
      goto LABEL_106;
    }
    v47 = *(unsigned int *)(v30 + 24);
LABEL_105:
    v72 = 16 * v47;
LABEL_106:
    v35 = v32 + v72;
  }
LABEL_30:
  v37 = 128;
  if ( !v31 )
    v37 = 16LL * *(unsigned int *)(v30 + 24);
  if ( v35 == v32 + v37 )
  {
    v65 = a4[1];
    v66 = *(unsigned int *)(v65 + 24);
    v67 = *(_QWORD *)(v65 + 8);
    if ( (_DWORD)v66 )
    {
      v68 = 1;
      for ( k = (v66 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; k = (v66 - 1) & v71 )
      {
        v70 = v67 + 24LL * k;
        if ( *(_UNKNOWN **)v70 == &unk_4F81450 && a2 == *(_QWORD *)(v70 + 8) )
          break;
        if ( *(_QWORD *)v70 == -4096 && *(_QWORD *)(v70 + 8) == -4096 )
          goto LABEL_114;
        v71 = v68 + k;
        ++v68;
      }
    }
    else
    {
LABEL_114:
      v70 = v67 + 24 * v66;
    }
    v74 = *(_QWORD *)(*(_QWORD *)(v70 + 16) + 24LL);
    v75 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v74 + 16LL);
    if ( v75 == sub_D00100 )
      v76 = sub_B19B20(v74 + 8, a2, a3, (__int64)a4);
    else
      v76 = v75(v74, a2, a3, (__int64)a4);
    v81[0] = v76;
    v80 = &unk_4F81450;
    sub_BBCF50((__int64)v82, v30, (__int64 *)&v80, v81);
    v35 = v83;
  }
  return *(unsigned __int8 *)(v35 + 8);
}
