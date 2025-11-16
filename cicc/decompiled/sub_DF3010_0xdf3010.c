// Function: sub_DF3010
// Address: 0xdf3010
//
__int64 __fastcall sub_DF3010(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v4; // r14
  void **v7; // rax
  void **v8; // rdx
  void **v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  char v13; // di
  __int64 v14; // rsi
  int v15; // edx
  unsigned int v16; // ecx
  __int64 v17; // rax
  void *v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r15
  char v21; // di
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // edx
  unsigned int v25; // ecx
  __int64 v26; // rax
  void *v27; // r8
  __int64 v28; // rdx
  __int64 v29; // r15
  char v30; // si
  __int64 v31; // rcx
  int v32; // edi
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 v36; // rax
  void *v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  int v42; // r9d
  unsigned int i; // eax
  __int64 v44; // rsi
  unsigned int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 (*v54)(); // r8
  char v55; // al
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  int v59; // r9d
  unsigned int j; // eax
  __int64 v61; // rsi
  unsigned int v62; // eax
  __int64 v63; // rax
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  int v68; // edi
  unsigned int k; // eax
  __int64 v70; // rsi
  unsigned int v71; // eax
  __int64 v72; // rdi
  __int64 (__fastcall *v73)(__int64, __int64, __int64, __int64); // rax
  char v74; // al
  int v75; // eax
  int v76; // r9d
  __int64 v77; // rdi
  __int64 (__fastcall *v78)(__int64); // rax
  char v79; // al
  int v80; // r9d
  int v81; // r9d
  void *v82; // [rsp+0h] [rbp-70h] BYREF
  char v83[8]; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v84[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v85; // [rsp+20h] [rbp-50h]

  v4 = a4;
  if ( *(_BYTE *)(a3 + 76) )
  {
    v7 = *(void ***)(a3 + 56);
    v8 = &v7[*(unsigned int *)(a3 + 68)];
    if ( v7 != v8 )
    {
      a4 = (__int64 *)&unk_4F881D0;
      while ( *v7 != &unk_4F881D0 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4F881D0) )
  {
    return 1;
  }
LABEL_8:
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( sub_C8CA60(a3, (__int64)&unk_4F82400) )
      goto LABEL_13;
LABEL_46:
    if ( (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F881D0, v11, (__int64)a4)
      || (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v46, v47)
      || (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82420, v48, v49) )
    {
      goto LABEL_13;
    }
    return 1;
  }
  v10 = *(void ***)(a3 + 8);
  v11 = (__int64)&v10[*(unsigned int *)(a3 + 20)];
  if ( v10 == (void **)v11 )
    goto LABEL_46;
  a4 = (__int64 *)&unk_4F82400;
  while ( *v10 != &unk_4F82400 )
  {
    if ( (void **)v11 == ++v10 )
      goto LABEL_46;
  }
LABEL_13:
  v12 = *v4;
  v13 = *(_BYTE *)(*v4 + 8) & 1;
  if ( v13 )
  {
    v14 = v12 + 16;
    v15 = 7;
  }
  else
  {
    v33 = *(unsigned int *)(v12 + 24);
    v14 = *(_QWORD *)(v12 + 16);
    if ( !(_DWORD)v33 )
      goto LABEL_51;
    v15 = v33 - 1;
  }
  v16 = v15 & (((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4));
  v17 = v14 + 16LL * v16;
  v18 = *(void **)v17;
  if ( *(_UNKNOWN **)v17 == &unk_4F86630 )
    goto LABEL_16;
  v51 = 1;
  while ( v18 != (void *)-4096LL )
  {
    v76 = v51 + 1;
    v16 = v15 & (v51 + v16);
    v17 = v14 + 16LL * v16;
    v18 = *(void **)v17;
    if ( *(_UNKNOWN **)v17 == &unk_4F86630 )
      goto LABEL_16;
    v51 = v76;
  }
  if ( v13 )
  {
    v50 = 128;
    goto LABEL_52;
  }
  v33 = *(unsigned int *)(v12 + 24);
LABEL_51:
  v50 = 16 * v33;
LABEL_52:
  v17 = v14 + v50;
LABEL_16:
  v19 = 128;
  if ( !v13 )
    v19 = 16LL * *(unsigned int *)(v12 + 24);
  if ( v17 == v14 + v19 )
  {
    v39 = v4[1];
    v40 = *(unsigned int *)(v39 + 24);
    v41 = *(_QWORD *)(v39 + 8);
    if ( (_DWORD)v40 )
    {
      v42 = 1;
      for ( i = (v40 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v40 - 1) & v45 )
      {
        v44 = v41 + 24LL * i;
        if ( *(_UNKNOWN **)v44 == &unk_4F86630 && a2 == *(_QWORD *)(v44 + 8) )
          break;
        if ( *(_QWORD *)v44 == -4096 && *(_QWORD *)(v44 + 8) == -4096 )
          goto LABEL_61;
        v45 = v42 + i;
        ++v42;
      }
    }
    else
    {
LABEL_61:
      v44 = v41 + 24 * v40;
    }
    v53 = *(_QWORD *)(*(_QWORD *)(v44 + 16) + 24LL);
    v54 = *(__int64 (**)())(*(_QWORD *)v53 + 16LL);
    v55 = 0;
    if ( v54 != sub_D000F0 )
      v55 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v54)(v53, a2, a3, v4);
    v83[0] = v55;
    v82 = &unk_4F86630;
    sub_BBCF50((__int64)v84, v12, (__int64 *)&v82, v83);
    v17 = v85;
  }
  if ( *(_BYTE *)(v17 + 8) )
    return 1;
  v20 = *v4;
  v21 = *(_BYTE *)(*v4 + 8) & 1;
  if ( v21 )
  {
    v23 = v20 + 16;
    v24 = 7;
  }
  else
  {
    v22 = *(unsigned int *)(v20 + 24);
    v23 = *(_QWORD *)(v20 + 16);
    if ( !(_DWORD)v22 )
      goto LABEL_73;
    v24 = v22 - 1;
  }
  v25 = v24 & (((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4));
  v26 = v23 + 16LL * v25;
  v27 = *(void **)v26;
  if ( *(_UNKNOWN **)v26 == &unk_4F81450 )
    goto LABEL_24;
  v64 = 1;
  while ( v27 != (void *)-4096LL )
  {
    v80 = v64 + 1;
    v25 = v24 & (v64 + v25);
    v26 = v23 + 16LL * v25;
    v27 = *(void **)v26;
    if ( *(_UNKNOWN **)v26 == &unk_4F81450 )
      goto LABEL_24;
    v64 = v80;
  }
  if ( v21 )
  {
    v63 = 128;
    goto LABEL_74;
  }
  v22 = *(unsigned int *)(v20 + 24);
LABEL_73:
  v63 = 16 * v22;
LABEL_74:
  v26 = v23 + v63;
LABEL_24:
  v28 = 128;
  if ( !v21 )
    v28 = 16LL * *(unsigned int *)(v20 + 24);
  if ( v26 == v23 + v28 )
  {
    v56 = v4[1];
    v57 = *(unsigned int *)(v56 + 24);
    v58 = *(_QWORD *)(v56 + 8);
    if ( (_DWORD)v57 )
    {
      v59 = 1;
      for ( j = (v57 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v57 - 1) & v62 )
      {
        v61 = v58 + 24LL * j;
        if ( *(_UNKNOWN **)v61 == &unk_4F81450 && a2 == *(_QWORD *)(v61 + 8) )
          break;
        if ( *(_QWORD *)v61 == -4096 && *(_QWORD *)(v61 + 8) == -4096 )
          goto LABEL_85;
        v62 = v59 + j;
        ++v59;
      }
    }
    else
    {
LABEL_85:
      v61 = v58 + 24 * v57;
    }
    v72 = *(_QWORD *)(*(_QWORD *)(v61 + 16) + 24LL);
    v73 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v72 + 16LL);
    if ( v73 == sub_D00100 )
      v74 = sub_B19B20(v72 + 8, a2, a3, (__int64)v4);
    else
      v74 = v73(v72, a2, a3, (__int64)v4);
    v83[0] = v74;
    v82 = &unk_4F81450;
    sub_BBCF50((__int64)v84, v20, (__int64 *)&v82, v83);
    v26 = v85;
  }
  if ( *(_BYTE *)(v26 + 8) )
    return 1;
  v29 = *v4;
  v30 = *(_BYTE *)(*v4 + 8) & 1;
  if ( v30 )
  {
    v31 = v29 + 16;
    v32 = 7;
  }
  else
  {
    v34 = *(unsigned int *)(v29 + 24);
    v31 = *(_QWORD *)(v29 + 16);
    if ( !(_DWORD)v34 )
      goto LABEL_58;
    v32 = v34 - 1;
  }
  v35 = v32 & (((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4));
  v36 = v31 + 16LL * v35;
  v37 = *(void **)v36;
  if ( *(_UNKNOWN **)v36 != &unk_4F875F0 )
  {
    v75 = 1;
    while ( v37 != (void *)-4096LL )
    {
      v81 = v75 + 1;
      v35 = v32 & (v75 + v35);
      v36 = v31 + 16LL * v35;
      v37 = *(void **)v36;
      if ( *(_UNKNOWN **)v36 == &unk_4F875F0 )
        goto LABEL_36;
      v75 = v81;
    }
    if ( v30 )
    {
      v52 = 128;
      goto LABEL_59;
    }
    v34 = *(unsigned int *)(v29 + 24);
LABEL_58:
    v52 = 16 * v34;
LABEL_59:
    v36 = v31 + v52;
  }
LABEL_36:
  v38 = 128;
  if ( !v30 )
    v38 = 16LL * *(unsigned int *)(v29 + 24);
  if ( v36 == v31 + v38 )
  {
    v65 = v4[1];
    v66 = *(unsigned int *)(v65 + 24);
    v67 = *(_QWORD *)(v65 + 8);
    if ( (_DWORD)v66 )
    {
      v68 = 1;
      for ( k = (v66 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; k = (v66 - 1) & v71 )
      {
        v70 = v67 + 24LL * k;
        if ( *(_UNKNOWN **)v70 == &unk_4F875F0 && a2 == *(_QWORD *)(v70 + 8) )
          break;
        if ( *(_QWORD *)v70 == -4096 && *(_QWORD *)(v70 + 8) == -4096 )
          goto LABEL_99;
        v71 = v68 + k;
        ++v68;
      }
    }
    else
    {
LABEL_99:
      v70 = v67 + 24 * v66;
    }
    v77 = *(_QWORD *)(*(_QWORD *)(v70 + 16) + 24LL);
    v78 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v77 + 16LL);
    if ( v78 == sub_D32160 )
      v79 = sub_D49500(v77 + 8, a2, a3, (__int64)v4);
    else
      v79 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v78)(v77, a2, a3, v4);
    v83[0] = v79;
    v82 = &unk_4F875F0;
    sub_BBCF50((__int64)v84, v29, (__int64 *)&v82, v83);
    v36 = v85;
  }
  return *(unsigned __int8 *)(v36 + 8);
}
