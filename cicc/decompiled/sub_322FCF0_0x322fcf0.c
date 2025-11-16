// Function: sub_322FCF0
// Address: 0x322fcf0
//
__int64 __fastcall sub_322FCF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r15
  unsigned int v9; // esi
  __int64 v10; // rdx
  int v11; // r10d
  __int64 v12; // r14
  unsigned __int64 v13; // r12
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  int v19; // eax
  int v20; // edx
  __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned __int64 v24; // r10
  __int64 v25; // r15
  char **v26; // rsi
  char **v27; // rdi
  _BYTE *v28; // rdi
  int v29; // edx
  int v30; // ecx
  __int64 v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r9
  char *v37; // r12
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  unsigned __int64 v42; // rdi
  int v43; // eax
  int v44; // eax
  int v45; // ecx
  __int64 v46; // rdi
  unsigned int v47; // eax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  unsigned __int64 v53; // rdi
  int v54; // [rsp+8h] [rbp-B8h]
  int v55; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v56[3]; // [rsp+18h] [rbp-A8h] BYREF
  char v57; // [rsp+30h] [rbp-90h] BYREF
  __int64 v58; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v59; // [rsp+58h] [rbp-68h]
  __int64 v60; // [rsp+60h] [rbp-60h]
  _BYTE v61[88]; // [rsp+68h] [rbp-58h] BYREF

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
    return *(_QWORD *)(a1 + 32) + 56 * v17 + 8;
  }
  while ( v16 != -1 )
  {
    if ( v16 == -2 && !v12 )
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
LABEL_57:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v20 <= v9 >> 3 )
  {
    sub_9E25D0(a1, v9);
    v44 = *(_DWORD *)(a1 + 24);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 8);
      v47 = v45 & v13;
      a6 = 1;
      v16 = 0;
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v46 + 16LL * (v45 & (unsigned int)v13);
      v48 = *(_QWORD *)v12;
      if ( v8 == *(_QWORD *)v12 )
        goto LABEL_15;
      while ( v48 != -1 )
      {
        if ( !v16 && v48 == -2 )
          v16 = v12;
        v47 = v45 & (a6 + v47);
        v12 = v46 + 16LL * v47;
        v48 = *(_QWORD *)v12;
        if ( v8 == *(_QWORD *)v12 )
          goto LABEL_15;
        a6 = (unsigned int)(a6 + 1);
      }
      goto LABEL_31;
    }
    goto LABEL_57;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *(_QWORD *)v12 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v21 = *a2;
  v22 = *(unsigned int *)(a1 + 44);
  v56[1] = (unsigned __int64)&v57;
  v58 = v21;
  v23 = *(unsigned int *)(a1 + 40);
  v56[2] = 0x200000000LL;
  v24 = v23 + 1;
  v60 = 0x200000000LL;
  v17 = v23;
  v59 = v61;
  if ( v23 + 1 > v22 )
  {
    v34 = *(_QWORD *)(a1 + 32);
    v35 = a1 + 32;
    v36 = a1 + 48;
    if ( v34 > (unsigned __int64)&v58 || (unsigned __int64)&v58 >= v34 + 56 * v23 )
    {
      v25 = sub_C8D7D0(v35, a1 + 48, v24, 0x38u, v56, v36);
      sub_3228460((__int64 **)(a1 + 32), v25, v49, v50, v51, v52);
      v53 = *(_QWORD *)(a1 + 32);
      a6 = a1 + 48;
      if ( a1 + 48 == v53 )
      {
        v23 = *(unsigned int *)(a1 + 40);
        *(_DWORD *)(a1 + 44) = v56[0];
        v26 = (char **)&v58;
        *(_QWORD *)(a1 + 32) = v25;
      }
      else
      {
        v55 = v56[0];
        _libc_free(v53);
        v23 = *(unsigned int *)(a1 + 40);
        *(_QWORD *)(a1 + 32) = v25;
        v26 = (char **)&v58;
        *(_DWORD *)(a1 + 44) = v55;
      }
      v17 = v23;
    }
    else
    {
      v37 = (char *)&v58 - v34;
      v25 = sub_C8D7D0(v35, a1 + 48, v24, 0x38u, v56, v36);
      sub_3228460((__int64 **)(a1 + 32), v25, v38, v39, v40, v41);
      v42 = *(_QWORD *)(a1 + 32);
      a6 = a1 + 48;
      v43 = v56[0];
      if ( v42 == a1 + 48 )
      {
        *(_QWORD *)(a1 + 32) = v25;
        *(_DWORD *)(a1 + 44) = v43;
      }
      else
      {
        v54 = v56[0];
        _libc_free(v42);
        *(_QWORD *)(a1 + 32) = v25;
        *(_DWORD *)(a1 + 44) = v54;
      }
      v23 = *(unsigned int *)(a1 + 40);
      v26 = (char **)&v37[v25];
      v17 = v23;
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = (char **)&v58;
  }
  v27 = (char **)(v25 + 56 * v23);
  if ( v27 )
  {
    *v27 = *v26;
    v27[1] = (char *)(v27 + 3);
    v27[2] = (char *)0x200000000LL;
    if ( *((_DWORD *)v26 + 4) )
      sub_32187E0((__int64)(v27 + 1), v26 + 1, v23, 7 * v23, v16, a6);
    v17 = *(unsigned int *)(a1 + 40);
  }
  v28 = v59;
  *(_DWORD *)(a1 + 40) = v17 + 1;
  if ( v28 != v61 )
  {
    _libc_free((unsigned __int64)v28);
    v17 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v12 + 8) = v17;
  return *(_QWORD *)(a1 + 32) + 56 * v17 + 8;
}
