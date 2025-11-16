// Function: sub_1047D40
// Address: 0x1047d40
//
__int64 __fastcall sub_1047D40(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
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
  char v21; // si
  __int64 v22; // rax
  __int64 v23; // rcx
  int v24; // edi
  unsigned int v25; // edx
  __int64 v26; // rax
  void *v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  int v33; // r9d
  unsigned int i; // eax
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // rdi
  __int64 (__fastcall *v44)(__int64, __int64, __int64, __int64 *); // rax
  char v45; // al
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  int v49; // r9d
  unsigned int j; // eax
  __int64 v51; // rsi
  unsigned int v52; // eax
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // rdi
  __int64 (__fastcall *v56)(__int64, __int64, __int64, __int64); // rax
  char v57; // al
  int v58; // r9d
  int v59; // r9d
  void *v60; // [rsp+0h] [rbp-70h] BYREF
  char v61[8]; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v62[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v63; // [rsp+20h] [rbp-50h]

  v4 = a4;
  if ( *(_BYTE *)(a3 + 76) )
  {
    v7 = *(void ***)(a3 + 56);
    v8 = &v7[*(unsigned int *)(a3 + 68)];
    if ( v7 != v8 )
    {
      a4 = (__int64 *)&unk_4F8F810;
      while ( *v7 != &unk_4F8F810 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4F8F810) )
  {
    return 1;
  }
LABEL_8:
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( sub_C8CA60(a3, (__int64)&unk_4F82400) )
      goto LABEL_13;
LABEL_37:
    if ( (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F8F810, v11, (__int64)a4)
      || (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v37, v38)
      || (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82420, v39, v40) )
    {
      goto LABEL_13;
    }
    return 1;
  }
  v10 = *(void ***)(a3 + 8);
  v11 = (__int64)&v10[*(unsigned int *)(a3 + 20)];
  if ( v10 == (void **)v11 )
    goto LABEL_37;
  a4 = (__int64 *)&unk_4F82400;
  while ( *v10 != &unk_4F82400 )
  {
    if ( (void **)v11 == ++v10 )
      goto LABEL_37;
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
    v29 = *(unsigned int *)(v12 + 24);
    v14 = *(_QWORD *)(v12 + 16);
    if ( !(_DWORD)v29 )
      goto LABEL_42;
    v15 = v29 - 1;
  }
  v16 = v15 & (((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4));
  v17 = v14 + 16LL * v16;
  v18 = *(void **)v17;
  if ( *(_UNKNOWN **)v17 == &unk_4F86540 )
    goto LABEL_16;
  v42 = 1;
  while ( v18 != (void *)-4096LL )
  {
    v58 = v42 + 1;
    v16 = v15 & (v42 + v16);
    v17 = v14 + 16LL * v16;
    v18 = *(void **)v17;
    if ( *(_UNKNOWN **)v17 == &unk_4F86540 )
      goto LABEL_16;
    v42 = v58;
  }
  if ( v13 )
  {
    v41 = 128;
    goto LABEL_43;
  }
  v29 = *(unsigned int *)(v12 + 24);
LABEL_42:
  v41 = 16 * v29;
LABEL_43:
  v17 = v14 + v41;
LABEL_16:
  v19 = 128;
  if ( !v13 )
    v19 = 16LL * *(unsigned int *)(v12 + 24);
  if ( v17 == v14 + v19 )
  {
    v30 = v4[1];
    v31 = *(unsigned int *)(v30 + 24);
    v32 = *(_QWORD *)(v30 + 8);
    if ( (_DWORD)v31 )
    {
      v33 = 1;
      for ( i = (v31 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v31 - 1) & v36 )
      {
        v35 = v32 + 24LL * i;
        if ( *(_UNKNOWN **)v35 == &unk_4F86540 && a2 == *(_QWORD *)(v35 + 8) )
          break;
        if ( *(_QWORD *)v35 == -4096 && *(_QWORD *)(v35 + 8) == -4096 )
          goto LABEL_49;
        v36 = v33 + i;
        ++v33;
      }
    }
    else
    {
LABEL_49:
      v35 = v32 + 24 * v31;
    }
    v43 = *(_QWORD *)(*(_QWORD *)(v35 + 16) + 24LL);
    v44 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v43 + 16LL);
    if ( v44 == sub_D32140 )
      v45 = sub_CF8780(v43 + 8, a2, a3, v4);
    else
      v45 = v44(v43, a2, a3, v4);
    v61[0] = v45;
    v60 = &unk_4F86540;
    sub_BBCF50((__int64)v62, v12, (__int64 *)&v60, v61);
    v17 = v63;
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
      goto LABEL_61;
    v24 = v22 - 1;
  }
  v25 = v24 & (((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4));
  v26 = v23 + 16LL * v25;
  v27 = *(void **)v26;
  if ( *(_UNKNOWN **)v26 != &unk_4F81450 )
  {
    v54 = 1;
    while ( v27 != (void *)-4096LL )
    {
      v59 = v54 + 1;
      v25 = v24 & (v54 + v25);
      v26 = v23 + 16LL * v25;
      v27 = *(void **)v26;
      if ( *(_UNKNOWN **)v26 == &unk_4F81450 )
        goto LABEL_24;
      v54 = v59;
    }
    if ( v21 )
    {
      v53 = 128;
      goto LABEL_62;
    }
    v22 = *(unsigned int *)(v20 + 24);
LABEL_61:
    v53 = 16 * v22;
LABEL_62:
    v26 = v23 + v53;
  }
LABEL_24:
  v28 = 128;
  if ( !v21 )
    v28 = 16LL * *(unsigned int *)(v20 + 24);
  if ( v26 == v23 + v28 )
  {
    v46 = v4[1];
    v47 = *(unsigned int *)(v46 + 24);
    v48 = *(_QWORD *)(v46 + 8);
    if ( (_DWORD)v47 )
    {
      v49 = 1;
      for ( j = (v47 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v47 - 1) & v52 )
      {
        v51 = v48 + 24LL * j;
        if ( *(_UNKNOWN **)v51 == &unk_4F81450 && a2 == *(_QWORD *)(v51 + 8) )
          break;
        if ( *(_QWORD *)v51 == -4096 && *(_QWORD *)(v51 + 8) == -4096 )
          goto LABEL_69;
        v52 = v49 + j;
        ++v49;
      }
    }
    else
    {
LABEL_69:
      v51 = v48 + 24 * v47;
    }
    v55 = *(_QWORD *)(*(_QWORD *)(v51 + 16) + 24LL);
    v56 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v55 + 16LL);
    if ( v56 == sub_D00100 )
      v57 = sub_B19B20(v55 + 8, a2, a3, (__int64)v4);
    else
      v57 = v56(v55, a2, a3, (__int64)v4);
    v61[0] = v57;
    v60 = &unk_4F81450;
    sub_BBCF50((__int64)v62, v20, (__int64 *)&v60, v61);
    v26 = v63;
  }
  return *(unsigned __int8 *)(v26 + 8);
}
