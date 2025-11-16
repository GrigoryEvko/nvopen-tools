// Function: sub_23FFD80
// Address: 0x23ffd80
//
__int64 __fastcall sub_23FFD80(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r15d
  unsigned int v13; // eax
  __int64 *v14; // rdi
  __int64 *v15; // rdx
  __int64 v16; // r10
  int v18; // eax
  int v19; // ecx
  _QWORD *v20; // rcx
  __int64 v21; // rax
  int v22; // edi
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    v27 = 0;
    *(_QWORD *)a2 = v9 + 1;
LABEL_22:
    LODWORD(v8) = 2 * v8;
    goto LABEL_23;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 1;
  v13 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v14 = (__int64 *)(v11 + 56LL * v13);
  v15 = 0;
  v16 = *v14;
  if ( v10 == *v14 )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v14;
    *(_QWORD *)(a1 + 24) = v11 + 56 * v8;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( v16 != -4096 )
  {
    if ( !v15 && v16 == -8192 )
      v15 = v14;
    v13 = (v8 - 1) & (v12 + v13);
    v14 = (__int64 *)(v11 + 56LL * v13);
    v16 = *v14;
    if ( v10 == *v14 )
      goto LABEL_3;
    ++v12;
  }
  v18 = *(_DWORD *)(a2 + 16);
  if ( !v15 )
    v15 = v14;
  v19 = v18 + 1;
  *(_QWORD *)a2 = v9 + 1;
  v27 = v15;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v8) )
    goto LABEL_22;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v8 >> 3 )
  {
LABEL_23:
    sub_23FFA90(a2, v8);
    sub_23FDEA0(a2, a3, &v27);
    v15 = v27;
    v19 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v20 = v15 + 2;
  *v15 = *a3;
  v21 = *(_QWORD *)(a4 + 16);
  if ( v21 )
  {
    v22 = *(_DWORD *)(a4 + 8);
    v15[3] = v21;
    *((_DWORD *)v15 + 4) = v22;
    v15[4] = *(_QWORD *)(a4 + 24);
    v15[5] = *(_QWORD *)(a4 + 32);
    *(_QWORD *)(v21 + 8) = v20;
    v23 = *(_QWORD *)(a4 + 40);
    *(_QWORD *)(a4 + 16) = 0;
    v15[6] = v23;
    *(_QWORD *)(a4 + 24) = a4 + 8;
    *(_QWORD *)(a4 + 32) = a4 + 8;
    *(_QWORD *)(a4 + 40) = 0;
  }
  else
  {
    *((_DWORD *)v15 + 4) = 0;
    v15[3] = 0;
    v15[4] = (__int64)v20;
    v15[5] = (__int64)v20;
    v15[6] = 0;
  }
  v24 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v15;
  *(_BYTE *)(a1 + 32) = 1;
  v25 = *(_QWORD *)(a2 + 8) + 56 * v24;
  v26 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = v25;
  *(_QWORD *)(a1 + 8) = v26;
  return a1;
}
