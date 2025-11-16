// Function: sub_1BE87B0
// Address: 0x1be87b0
//
__int64 __fastcall sub_1BE87B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r13
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 *v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r14
  _BYTE *v20; // rsi
  unsigned int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // r8
  unsigned int v24; // r15d
  unsigned int v25; // ecx
  __int64 *v26; // r14
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // rdi
  __int64 v30; // r12
  __int64 v31; // rdi
  int v32; // eax
  int v33; // esi
  __int64 v34; // r8
  unsigned int v35; // edx
  int v36; // ecx
  __int64 *v37; // rax
  __int64 v38; // rdi
  int v39; // r10d
  int v40; // r10d
  int v41; // ecx
  int v42; // eax
  int v43; // edx
  __int64 v44; // rdi
  __int64 *v45; // r8
  unsigned int v46; // r15d
  int v47; // r9d
  __int64 v48; // rsi
  int v49; // ecx
  int v50; // r10d
  int v51; // r10d
  __int64 *v52; // r9
  __int64 v53; // [rsp+0h] [rbp-40h] BYREF
  __int64 v54[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a3 + 48);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a3 + 32);
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
      {
        v10 = v8[1];
        if ( v10 )
          return v10;
      }
    }
    else
    {
      v12 = 1;
      while ( v9 != -8 )
      {
        v39 = v12 + 1;
        v7 = (v5 - 1) & (v12 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v12 = v39;
      }
    }
  }
  v13 = *(unsigned int *)(a1 + 48);
  v14 = 0;
  if ( (_DWORD)v13 )
  {
    v15 = *(_QWORD *)(a1 + 32);
    v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 *)(v15 + 72LL * v16);
    v18 = *v17;
    if ( *v17 == a2 )
    {
LABEL_10:
      if ( v17 != (__int64 *)(v15 + 72 * v13) )
      {
        v14 = v17[4];
        goto LABEL_12;
      }
    }
    else
    {
      v49 = 1;
      while ( v18 != -8 )
      {
        v50 = v49 + 1;
        v16 = (v13 - 1) & (v49 + v16);
        v17 = (__int64 *)(v15 + 72LL * v16);
        v18 = *v17;
        if ( *v17 == a2 )
          goto LABEL_10;
        v49 = v50;
      }
    }
    v14 = 0;
  }
LABEL_12:
  v19 = sub_1BE87B0(a1, v14, a3);
  sub_1BE2190(&v53, a2, v19);
  v10 = v53;
  v20 = *(_BYTE **)(v19 + 32);
  v54[0] = v53;
  if ( v20 == *(_BYTE **)(v19 + 40) )
  {
    sub_1BE72B0(v19 + 24, v20, v54);
    v10 = v53;
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = v53;
      v20 = *(_BYTE **)(v19 + 32);
      v10 = v53;
    }
    *(_QWORD *)(v19 + 32) = v20 + 8;
  }
  v21 = *(_DWORD *)(a3 + 48);
  v53 = 0;
  v22 = a3 + 24;
  if ( !v21 )
  {
    ++*(_QWORD *)(a3 + 24);
    goto LABEL_28;
  }
  v23 = *(_QWORD *)(a3 + 32);
  v24 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v25 = (v21 - 1) & v24;
  v26 = (__int64 *)(v23 + 16LL * v25);
  v27 = *v26;
  if ( *v26 == a2 )
  {
LABEL_18:
    v28 = v26[1];
    v26[1] = v10;
    if ( !v28 )
      return v10;
    v29 = *(_QWORD *)(v28 + 24);
    if ( v29 )
      j_j___libc_free_0(v29, *(_QWORD *)(v28 + 40) - v29);
    j_j___libc_free_0(v28, 56);
    v10 = v26[1];
    goto LABEL_22;
  }
  v40 = 1;
  v37 = 0;
  while ( v27 != -8 )
  {
    if ( v27 == -16 && !v37 )
      v37 = v26;
    v25 = (v21 - 1) & (v40 + v25);
    v26 = (__int64 *)(v23 + 16LL * v25);
    v27 = *v26;
    if ( *v26 == a2 )
      goto LABEL_18;
    ++v40;
  }
  v41 = *(_DWORD *)(a3 + 40);
  if ( !v37 )
    v37 = v26;
  ++*(_QWORD *)(a3 + 24);
  v36 = v41 + 1;
  if ( 4 * v36 >= 3 * v21 )
  {
LABEL_28:
    sub_1BE8590(v22, 2 * v21);
    v32 = *(_DWORD *)(a3 + 48);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a3 + 32);
      v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = *(_DWORD *)(a3 + 40) + 1;
      v37 = (__int64 *)(v34 + 16LL * v35);
      v38 = *v37;
      if ( *v37 != a2 )
      {
        v51 = 1;
        v52 = 0;
        while ( v38 != -8 )
        {
          if ( !v52 && v38 == -16 )
            v52 = v37;
          v35 = v33 & (v51 + v35);
          v37 = (__int64 *)(v34 + 16LL * v35);
          v38 = *v37;
          if ( *v37 == a2 )
            goto LABEL_30;
          ++v51;
        }
        if ( v52 )
          v37 = v52;
      }
      goto LABEL_30;
    }
    goto LABEL_72;
  }
  if ( v21 - *(_DWORD *)(a3 + 44) - v36 <= v21 >> 3 )
  {
    sub_1BE8590(v22, v21);
    v42 = *(_DWORD *)(a3 + 48);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a3 + 32);
      v45 = 0;
      v46 = (v42 - 1) & v24;
      v47 = 1;
      v36 = *(_DWORD *)(a3 + 40) + 1;
      v37 = (__int64 *)(v44 + 16LL * v46);
      v48 = *v37;
      if ( *v37 != a2 )
      {
        while ( v48 != -8 )
        {
          if ( v48 == -16 && !v45 )
            v45 = v37;
          v46 = v43 & (v47 + v46);
          v37 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v37;
          if ( *v37 == a2 )
            goto LABEL_30;
          ++v47;
        }
        if ( v45 )
          v37 = v45;
      }
      goto LABEL_30;
    }
LABEL_72:
    ++*(_DWORD *)(a3 + 40);
    BUG();
  }
LABEL_30:
  *(_DWORD *)(a3 + 40) = v36;
  if ( *v37 != -8 )
    --*(_DWORD *)(a3 + 44);
  *v37 = a2;
  v37[1] = v10;
LABEL_22:
  v30 = v53;
  if ( v53 )
  {
    v31 = *(_QWORD *)(v53 + 24);
    if ( v31 )
      j_j___libc_free_0(v31, *(_QWORD *)(v53 + 40) - v31);
    j_j___libc_free_0(v30, 56);
  }
  return v10;
}
