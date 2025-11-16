// Function: sub_2BA0A80
// Address: 0x2ba0a80
//
__int64 __fastcall sub_2BA0A80(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rdx
  int v11; // r10d
  __int64 v12; // r15
  unsigned __int64 v13; // r12
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  int v19; // eax
  int v20; // edx
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  __int64 v24; // rcx
  char **v25; // rsi
  __int64 v26; // rdx
  char **v27; // rdi
  _BYTE *v28; // rdi
  int v29; // edx
  int v30; // ecx
  __int64 v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // rdi
  unsigned __int64 v34; // r14
  __int64 v35; // rdi
  int v36; // eax
  int v37; // ecx
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v42; // [rsp+48h] [rbp-78h]
  __int64 v43; // [rsp+50h] [rbp-70h]
  _BYTE v44[104]; // [rsp+58h] [rbp-68h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_27;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (0xBF58476D1CE4E5B9LL * v8) ^ ((0xBF58476D1CE4E5B9LL * v8) >> 31);
  v14 = v13 & (v9 - 1);
  v15 = v10 + 16LL * v14;
  v16 = *(_QWORD *)v15;
  if ( v8 == *(_QWORD *)v15 )
  {
LABEL_3:
    v17 = *(unsigned int *)(v15 + 8);
    return *(_QWORD *)(a1 + 32) + 72 * v17 + 8;
  }
  while ( v16 != -1 )
  {
    if ( !v12 && v16 == -2 )
      v12 = v15;
    a6 = (unsigned int)(v11 + 1);
    v14 = (v9 - 1) & (v11 + v14);
    v15 = v10 + 16LL * v14;
    v16 = *(_QWORD *)v15;
    if ( v8 == *(_QWORD *)v15 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v15;
  v19 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v9 )
  {
LABEL_27:
    sub_9E25D0(a1, 2 * v9);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 8);
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v32 = v30 & (((0xBF58476D1CE4E5B9LL * v8) >> 31) ^ (484763065 * v8));
      v12 = v31 + 16LL * v32;
      v33 = *(_QWORD *)v12;
      if ( v8 == *(_QWORD *)v12 )
        goto LABEL_15;
      a6 = 1;
      v16 = 0;
      while ( v33 != -1 )
      {
        if ( !v16 && v33 == -2 )
          v16 = v12;
        v32 = v30 & (a6 + v32);
        v12 = v31 + 16LL * v32;
        v33 = *(_QWORD *)v12;
        if ( v8 == *(_QWORD *)v12 )
          goto LABEL_15;
        a6 = (unsigned int)(a6 + 1);
      }
LABEL_31:
      if ( v16 )
        v12 = v16;
      goto LABEL_15;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v20 <= v9 >> 3 )
  {
    sub_9E25D0(a1, v9);
    v36 = *(_DWORD *)(a1 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 8);
      v39 = v37 & v13;
      a6 = 1;
      v16 = 0;
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v38 + 16LL * (v37 & (unsigned int)v13);
      v40 = *(_QWORD *)v12;
      if ( v8 == *(_QWORD *)v12 )
        goto LABEL_15;
      while ( v40 != -1 )
      {
        if ( !v16 && v40 == -2 )
          v16 = v12;
        v39 = v37 & (a6 + v39);
        v12 = v38 + 16LL * v39;
        v40 = *(_QWORD *)v12;
        if ( v8 == *(_QWORD *)v12 )
          goto LABEL_15;
        a6 = (unsigned int)(a6 + 1);
      }
      goto LABEL_31;
    }
    goto LABEL_51;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *(_QWORD *)v12 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v21 = *(unsigned int *)(a1 + 44);
  v41 = *a2;
  v22 = *(unsigned int *)(a1 + 40);
  v23 = v22 + 1;
  v43 = 0x600000000LL;
  v17 = v22;
  v42 = v44;
  if ( v22 + 1 > v21 )
  {
    v34 = *(_QWORD *)(a1 + 32);
    v35 = a1 + 32;
    if ( v34 > (unsigned __int64)&v41 || (unsigned __int64)&v41 >= v34 + 72 * v22 )
    {
      sub_2B552D0(v35, v23, v22, v21, v16, a6);
      v22 = *(unsigned int *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
      v25 = (char **)&v41;
      v17 = v22;
    }
    else
    {
      sub_2B552D0(v35, v23, v22, v21, v16, a6);
      v24 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v25 = (char **)((char *)&v41 + v24 - v34);
      v17 = v22;
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = (char **)&v41;
  }
  v26 = 9 * v22;
  v27 = (char **)(v24 + 8 * v26);
  if ( v27 )
  {
    *v27 = *v25;
    v27[1] = (char *)(v27 + 3);
    v27[2] = (char *)0x600000000LL;
    if ( *((_DWORD *)v25 + 4) )
      sub_2B0F6D0((__int64)(v27 + 1), v25 + 1, v26, v24, v16, a6);
    v17 = *(unsigned int *)(a1 + 40);
  }
  v28 = v42;
  *(_DWORD *)(a1 + 40) = v17 + 1;
  if ( v28 != v44 )
  {
    _libc_free((unsigned __int64)v28);
    v17 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v12 + 8) = v17;
  return *(_QWORD *)(a1 + 32) + 72 * v17 + 8;
}
