// Function: sub_254EAA0
// Address: 0x254eaa0
//
__int64 __fastcall sub_254EAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r13
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned __int64 v25; // rax
  __int64 *v26; // rsi
  int v27; // r8d
  __int64 v28; // rcx
  __int64 v29; // rdi
  int v30; // r8d
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // r9
  char *v34; // rdx
  __int64 v35; // rbx
  unsigned int v36; // eax
  unsigned int v37; // eax
  __int64 v39; // rcx
  __int64 v40; // rsi
  int v41; // r10d
  unsigned int i; // eax
  __int64 v43; // r8
  unsigned int v44; // eax
  int v45; // eax
  int v46; // r10d
  __int64 v47; // [rsp+0h] [rbp-40h]

  v5 = (_QWORD *)(a2 + 72);
  if ( !sub_25096F0((_QWORD *)(a2 + 72)) )
    goto LABEL_27;
  v8 = *(_QWORD *)(a3 + 208);
  v9 = sub_25096F0((_QWORD *)(a2 + 72));
  v10 = *(_QWORD *)(v8 + 240);
  v11 = *(_QWORD *)v10;
  if ( *(_QWORD *)v10 )
  {
    if ( !*(_BYTE *)(v10 + 16) )
    {
      v12 = sub_BC1CD0(v11, &unk_4F881D0, v9);
LABEL_5:
      v47 = v12 + 8;
      goto LABEL_6;
    }
    v39 = *(unsigned int *)(v11 + 88);
    v40 = *(_QWORD *)(v11 + 72);
    if ( (_DWORD)v39 )
    {
      v41 = 1;
      for ( i = (v39 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)))); ; i = (v39 - 1) & v44 )
      {
        v43 = v40 + 24LL * i;
        if ( *(_UNKNOWN **)v43 == &unk_4F881D0 && v9 == *(_QWORD *)(v43 + 8) )
          break;
        if ( *(_QWORD *)v43 == -4096 && *(_QWORD *)(v43 + 8) == -4096 )
          goto LABEL_39;
        v44 = v41 + i;
        ++v41;
      }
      if ( v43 != v40 + 24 * v39 )
      {
        v12 = *(_QWORD *)(*(_QWORD *)(v43 + 16) + 24LL);
        if ( v12 )
          goto LABEL_5;
      }
    }
  }
LABEL_39:
  v47 = 0;
LABEL_6:
  if ( !sub_25096F0(v5) )
    goto LABEL_27;
  v13 = *(_QWORD *)(a3 + 208);
  v14 = sub_25096F0(v5);
  v15 = *(_QWORD *)(v13 + 240);
  v16 = *(_QWORD *)v15;
  if ( !*(_QWORD *)v15 )
    goto LABEL_35;
  if ( *(_BYTE *)(v15 + 16) )
  {
    v17 = sub_BBB550(v16, (__int64)&unk_4F881D0, v14);
    if ( v17 )
      goto LABEL_10;
LABEL_35:
    v18 = 0;
    goto LABEL_11;
  }
  v17 = sub_BC1CD0(v16, &unk_4F881D0, v14);
LABEL_10:
  v18 = v17 + 8;
LABEL_11:
  v19 = *(_QWORD *)(a3 + 208);
  v20 = sub_25096F0(v5);
  v21 = *(_QWORD *)(v19 + 240);
  v22 = *(_QWORD *)v21;
  if ( !*(_QWORD *)v21 )
    goto LABEL_27;
  if ( *(_BYTE *)(v21 + 16) )
  {
    v23 = sub_BBB550(v22, (__int64)&unk_4F875F0, v20);
    if ( !v23 )
      goto LABEL_27;
  }
  else
  {
    v23 = sub_BC1CD0(v22, &unk_4F875F0, v20);
  }
  v24 = v23 + 8;
  if ( !v18 )
  {
LABEL_27:
    sub_AADB10(a1, *(_DWORD *)(a2 + 96), 1);
    return a1;
  }
  v25 = sub_250D070(v5);
  v26 = sub_DD8400(v18, v25);
  if ( a4 )
  {
    v27 = *(_DWORD *)(v24 + 24);
    v28 = *(_QWORD *)(a4 + 40);
    v29 = *(_QWORD *)(v24 + 8);
    if ( v27 )
    {
      v30 = v27 - 1;
      v31 = v30 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v32 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v32;
      if ( *v32 == v28 )
      {
LABEL_18:
        v34 = (char *)v32[1];
LABEL_19:
        v26 = sub_DDF4E0(v18, (__int64 **)v26, v34);
        goto LABEL_20;
      }
      v45 = 1;
      while ( v33 != -4096 )
      {
        v46 = v45 + 1;
        v31 = v30 & (v45 + v31);
        v32 = (__int64 *)(v29 + 16LL * v31);
        v33 = *v32;
        if ( v28 == *v32 )
          goto LABEL_18;
        v45 = v46;
      }
    }
    v34 = 0;
    goto LABEL_19;
  }
LABEL_20:
  if ( !v47 || !v26 )
    goto LABEL_27;
  v35 = sub_DBB9F0(v47, (__int64)v26, 0, 0);
  v36 = *(_DWORD *)(v35 + 8);
  *(_DWORD *)(a1 + 8) = v36;
  if ( v36 > 0x40 )
    sub_C43780(a1, (const void **)v35);
  else
    *(_QWORD *)a1 = *(_QWORD *)v35;
  v37 = *(_DWORD *)(v35 + 24);
  *(_DWORD *)(a1 + 24) = v37;
  if ( v37 > 0x40 )
    sub_C43780(a1 + 16, (const void **)(v35 + 16));
  else
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(v35 + 16);
  return a1;
}
