// Function: sub_2299ED0
// Address: 0x2299ed0
//
__int64 __fastcall sub_2299ED0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void **v7; // rax
  void **v8; // rdx
  __int64 **v10; // rdx
  __int64 **v11; // rcx
  __int64 **v12; // rax
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
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  int v43; // r9d
  unsigned int i; // eax
  __int64 v45; // rsi
  unsigned int v46; // eax
  void **v47; // rcx
  __int64 v48; // rax
  int v49; // eax
  void **v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 (__fastcall *v53)(__int64, __int64, __int64, __int64 *); // rax
  char v54; // al
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  int v58; // r9d
  unsigned int j; // eax
  __int64 v60; // rsi
  unsigned int v61; // eax
  __int64 v62; // rax
  int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  int v67; // r8d
  unsigned int k; // eax
  __int64 v69; // rsi
  unsigned int v70; // eax
  __int64 v71; // rdi
  __int64 (__fastcall *v72)(__int64); // rax
  char v73; // al
  int v74; // eax
  int v75; // r9d
  __int64 v76; // rdi
  __int64 (__fastcall *v77)(__int64); // rax
  char v78; // al
  int v79; // r9d
  int v80; // r9d
  void *v81; // [rsp+0h] [rbp-70h] BYREF
  char v82[8]; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v83[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v84; // [rsp+20h] [rbp-50h]

  if ( *(_BYTE *)(a3 + 76) )
  {
    v7 = *(void ***)(a3 + 56);
    v8 = &v7[*(unsigned int *)(a3 + 68)];
    if ( v7 != v8 )
    {
      while ( *v7 != &unk_4FDB350 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4FDB350) )
  {
    return 1;
  }
LABEL_8:
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(__int64 ***)(a3 + 8);
    v11 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v10 != v11 )
    {
      v12 = *(__int64 ***)(a3 + 8);
      while ( *v12 != &qword_4F82400 )
      {
        if ( v11 == ++v12 )
          goto LABEL_62;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
    goto LABEL_13;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(__int64 ***)(a3 + 8);
    v12 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v10 != v12 )
    {
LABEL_62:
      v50 = (void **)v10;
      while ( *v50 != &unk_4FDB350 )
      {
        if ( v12 == (__int64 **)++v50 )
          goto LABEL_50;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&unk_4FDB350) )
    goto LABEL_13;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v10 = *(__int64 ***)(a3 + 8);
    v12 = &v10[*(unsigned int *)(a3 + 20)];
    if ( v10 != v12 )
    {
LABEL_50:
      v47 = (void **)v10;
      while ( *v10 != &qword_4F82400 )
      {
        if ( ++v10 == v12 )
          goto LABEL_88;
      }
      goto LABEL_13;
    }
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
    goto LABEL_13;
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( sub_C8CA60(a3, (__int64)&unk_4F82420) )
      goto LABEL_13;
    return 1;
  }
  v47 = *(void ***)(a3 + 8);
  v12 = (__int64 **)&v47[*(unsigned int *)(a3 + 20)];
  if ( v47 == (void **)v12 )
    return 1;
LABEL_88:
  while ( *v47 != &unk_4F82420 )
  {
    if ( ++v47 == (void **)v12 )
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
      goto LABEL_55;
    v16 = v34 - 1;
  }
  v17 = v16 & (((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4));
  v18 = v15 + 16LL * v17;
  v19 = *(void **)v18;
  if ( *(_UNKNOWN **)v18 == &unk_4F86540 )
    goto LABEL_16;
  v49 = 1;
  while ( v19 != (void *)-4096LL )
  {
    v75 = v49 + 1;
    v17 = v16 & (v49 + v17);
    v18 = v15 + 16LL * v17;
    v19 = *(void **)v18;
    if ( *(_UNKNOWN **)v18 == &unk_4F86540 )
      goto LABEL_16;
    v49 = v75;
  }
  if ( v14 )
  {
    v48 = 128;
    goto LABEL_56;
  }
  v34 = *(unsigned int *)(v13 + 24);
LABEL_55:
  v48 = 16 * v34;
LABEL_56:
  v18 = v15 + v48;
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
          goto LABEL_70;
        v46 = v43 + i;
        ++v43;
      }
    }
    else
    {
LABEL_70:
      v45 = v42 + 24 * v41;
    }
    v52 = *(_QWORD *)(*(_QWORD *)(v45 + 16) + 24LL);
    v53 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v52 + 16LL);
    if ( v53 == sub_D32140 )
      v54 = sub_CF8780(v52 + 8, a2, a3, a4);
    else
      v54 = v53(v52, a2, a3, a4);
    v82[0] = v54;
    v81 = &unk_4F86540;
    sub_BBCF50((__int64)v83, v13, (__int64 *)&v81, v82);
    v18 = v84;
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
      goto LABEL_91;
    v25 = v23 - 1;
  }
  v26 = v25 & (((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4));
  v27 = v24 + 16LL * v26;
  v28 = *(void **)v27;
  if ( *(_UNKNOWN **)v27 == &unk_4F881D0 )
    goto LABEL_24;
  v63 = 1;
  while ( v28 != (void *)-4096LL )
  {
    v79 = v63 + 1;
    v26 = v25 & (v63 + v26);
    v27 = v24 + 16LL * v26;
    v28 = *(void **)v27;
    if ( *(_UNKNOWN **)v27 == &unk_4F881D0 )
      goto LABEL_24;
    v63 = v79;
  }
  if ( v22 )
  {
    v62 = 128;
    goto LABEL_92;
  }
  v23 = *(unsigned int *)(v21 + 24);
LABEL_91:
  v62 = 16 * v23;
LABEL_92:
  v27 = v24 + v62;
LABEL_24:
  v29 = 128;
  if ( !v22 )
    v29 = 16LL * *(unsigned int *)(v21 + 24);
  if ( v27 == v24 + v29 )
  {
    v55 = a4[1];
    v56 = *(unsigned int *)(v55 + 24);
    v57 = *(_QWORD *)(v55 + 8);
    if ( (_DWORD)v56 )
    {
      v58 = 1;
      for ( j = (v56 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v56 - 1) & v61 )
      {
        v60 = v57 + 24LL * j;
        if ( *(_UNKNOWN **)v60 == &unk_4F881D0 && a2 == *(_QWORD *)(v60 + 8) )
          break;
        if ( *(_QWORD *)v60 == -4096 && *(_QWORD *)(v60 + 8) == -4096 )
          goto LABEL_104;
        v61 = v58 + j;
        ++v58;
      }
    }
    else
    {
LABEL_104:
      v60 = v57 + 24 * v56;
    }
    v71 = *(_QWORD *)(*(_QWORD *)(v60 + 16) + 24LL);
    v72 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v71 + 16LL);
    if ( v72 == sub_D32150 )
      v73 = sub_DF3010(v71 + 8, a2, a3, a4);
    else
      v73 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v72)(v71, a2, a3, a4);
    v82[0] = v73;
    v81 = &unk_4F881D0;
    sub_BBCF50((__int64)v83, v21, (__int64 *)&v81, v82);
    v27 = v84;
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
      goto LABEL_67;
    v33 = v35 - 1;
  }
  v36 = v33 & (((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4));
  v37 = v32 + 16LL * v36;
  v38 = *(void **)v37;
  if ( *(_UNKNOWN **)v37 != &unk_4F875F0 )
  {
    v74 = 1;
    while ( v38 != (void *)-4096LL )
    {
      v80 = v74 + 1;
      v36 = v33 & (v74 + v36);
      v37 = v32 + 16LL * v36;
      v38 = *(void **)v37;
      if ( *(_UNKNOWN **)v37 == &unk_4F875F0 )
        goto LABEL_36;
      v74 = v80;
    }
    if ( v31 )
    {
      v51 = 128;
      goto LABEL_68;
    }
    v35 = *(unsigned int *)(v30 + 24);
LABEL_67:
    v51 = 16 * v35;
LABEL_68:
    v37 = v32 + v51;
  }
LABEL_36:
  v39 = 128;
  if ( !v31 )
    v39 = 16LL * *(unsigned int *)(v30 + 24);
  if ( v37 == v32 + v39 )
  {
    v64 = a4[1];
    v65 = *(unsigned int *)(v64 + 24);
    v66 = *(_QWORD *)(v64 + 8);
    if ( (_DWORD)v65 )
    {
      v67 = 1;
      for ( k = (v65 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; k = (v65 - 1) & v70 )
      {
        v69 = v66 + 24LL * k;
        if ( *(_UNKNOWN **)v69 == &unk_4F875F0 && a2 == *(_QWORD *)(v69 + 8) )
          break;
        if ( *(_QWORD *)v69 == -4096 && *(_QWORD *)(v69 + 8) == -4096 )
          goto LABEL_118;
        v70 = v67 + k;
        ++v67;
      }
    }
    else
    {
LABEL_118:
      v69 = v66 + 24 * v65;
    }
    v76 = *(_QWORD *)(*(_QWORD *)(v69 + 16) + 24LL);
    v77 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v76 + 16LL);
    if ( v77 == sub_D32160 )
      v78 = sub_D49500(v76 + 8, a2, a3, (__int64)a4);
    else
      v78 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v77)(v76, a2, a3, a4);
    v82[0] = v78;
    v81 = &unk_4F875F0;
    sub_BBCF50((__int64)v83, v30, (__int64 *)&v81, v82);
    v37 = v84;
  }
  return *(unsigned __int8 *)(v37 + 8);
}
