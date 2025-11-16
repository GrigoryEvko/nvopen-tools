// Function: sub_AE0110
// Address: 0xae0110
//
__int64 __fastcall sub_AE0110(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v6; // r14
  unsigned int v7; // esi
  __int64 v8; // rcx
  int v9; // r11d
  __int64 *v10; // r8
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r10
  int v15; // eax
  int v16; // edx
  __int64 v17; // rcx
  int v18; // eax
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rax
  int v23; // eax
  int v24; // ecx
  __int64 v25; // rsi
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // r10d
  __int64 *v29; // r9
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rdi
  int v35; // r12d
  int v36; // eax
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  int v40; // r9d
  unsigned int v41; // r15d
  __int64 *v42; // rdi
  __int64 v43; // rcx
  _QWORD v44[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *a2;
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( v6 == *v12 )
    return *(_QWORD *)(a1 + 32) + 56LL * *((unsigned int *)v12 + 2);
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v10 )
      v10 = v12;
    v11 = (v7 - 1) & (v9 + v11);
    v12 = (__int64 *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
      return *(_QWORD *)(a1 + 32) + 56LL * *((unsigned int *)v12 + 2);
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v7 )
  {
LABEL_22:
    sub_ADFF30(a1, 2 * v7);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v26 = (v23 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (__int64 *)(v25 + 16LL * v26);
      v27 = *v10;
      if ( v6 != *v10 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -4096 )
        {
          if ( !v29 && v27 == -8192 )
            v29 = v10;
          v26 = v24 & (v28 + v26);
          v10 = (__int64 *)(v25 + 16LL * v26);
          v27 = *v10;
          if ( v6 == *v10 )
            goto LABEL_14;
          ++v28;
        }
        if ( v29 )
          v10 = v29;
      }
      goto LABEL_14;
    }
    goto LABEL_50;
  }
  if ( v7 - *(_DWORD *)(a1 + 20) - v16 <= v7 >> 3 )
  {
    sub_ADFF30(a1, v7);
    v37 = *(_DWORD *)(a1 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 8);
      v40 = 1;
      v41 = v38 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v42 = 0;
      v10 = (__int64 *)(v39 + 16LL * v41);
      v43 = *v10;
      if ( v6 != *v10 )
      {
        while ( v43 != -4096 )
        {
          if ( !v42 && v43 == -8192 )
            v42 = v10;
          v41 = v38 & (v40 + v41);
          v10 = (__int64 *)(v39 + 16LL * v41);
          v43 = *v10;
          if ( v6 == *v10 )
            goto LABEL_14;
          ++v40;
        }
        if ( v42 )
          v10 = v42;
      }
      goto LABEL_14;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *((_DWORD *)v10 + 2) = 0;
  *v10 = v6;
  *((_DWORD *)v10 + 2) = *(_DWORD *)(a1 + 40);
  v17 = *(unsigned int *)(a1 + 40);
  v18 = v17;
  if ( *(_DWORD *)(a1 + 44) <= (unsigned int)v17 )
  {
    v30 = sub_C8D7D0(a1 + 32, a1 + 48, 0, 56, v44);
    v31 = a1 + 32;
    v19 = v30;
    v32 = v30 + 56LL * *(unsigned int *)(a1 + 40);
    if ( v32 )
    {
      v33 = v32 + 8;
      *(_QWORD *)(v33 - 8) = *a2;
      sub_ADDA00(v33, a3);
      v31 = a1 + 32;
    }
    sub_ADFE70(v31, v19);
    v34 = *(_QWORD *)(a1 + 32);
    v35 = v44[0];
    if ( a1 + 48 != v34 )
      _libc_free(v34, v19);
    v36 = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v19;
    *(_DWORD *)(a1 + 44) = v35;
    v22 = (unsigned int)(v36 + 1);
    *(_DWORD *)(a1 + 40) = v22;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 32);
    v20 = v19 + 56 * v17;
    if ( v20 )
    {
      v21 = v20 + 8;
      *(_QWORD *)(v21 - 8) = *a2;
      sub_ADDA00(v21, a3);
      v18 = *(_DWORD *)(a1 + 40);
      v19 = *(_QWORD *)(a1 + 32);
    }
    v22 = (unsigned int)(v18 + 1);
    *(_DWORD *)(a1 + 40) = v22;
  }
  return v19 + 56 * v22 - 56;
}
