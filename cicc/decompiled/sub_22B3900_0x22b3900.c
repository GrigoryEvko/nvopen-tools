// Function: sub_22B3900
// Address: 0x22b3900
//
__int64 __fastcall sub_22B3900(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // r9
  int v10; // edx
  __int64 v11; // r8
  int v12; // r15d
  unsigned int v13; // ecx
  int *v14; // rdi
  int *v15; // rax
  int v16; // r10d
  int v18; // edi
  int v19; // ecx
  int v20; // edx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  int *v28; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    v28 = 0;
    *(_QWORD *)a2 = v9 + 1;
LABEL_19:
    LODWORD(v8) = 2 * v8;
    goto LABEL_20;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 1;
  v13 = (v8 - 1) & (37 * v10);
  v14 = (int *)(v11 + 40LL * v13);
  v15 = 0;
  v16 = *v14;
  if ( v10 == *v14 )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v14;
    *(_QWORD *)(a1 + 24) = v11 + 40 * v8;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( v16 != -1 )
  {
    if ( v16 == -2 && !v15 )
      v15 = v14;
    v13 = (v8 - 1) & (v12 + v13);
    v14 = (int *)(v11 + 40LL * v13);
    v16 = *v14;
    if ( v10 == *v14 )
      goto LABEL_3;
    ++v12;
  }
  if ( !v15 )
    v15 = v14;
  v18 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a2 = v9 + 1;
  v19 = v18 + 1;
  v28 = v15;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v8) )
    goto LABEL_19;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v8 >> 3 )
  {
LABEL_20:
    sub_22B36A0(a2, v8);
    sub_22B1BB0(a2, a3, &v28);
    v19 = *(_DWORD *)(a2 + 16) + 1;
    v15 = v28;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v15 != -1 )
    --*(_DWORD *)(a2 + 20);
  v20 = *a3;
  *((_QWORD *)v15 + 3) = 0;
  *((_QWORD *)v15 + 2) = 0;
  v15[8] = 0;
  *v15 = v20;
  *((_QWORD *)v15 + 1) = 1;
  v21 = *(_QWORD *)(a4 + 8);
  ++*(_QWORD *)a4;
  v22 = *((_QWORD *)v15 + 2);
  *((_QWORD *)v15 + 2) = v21;
  LODWORD(v21) = *(_DWORD *)(a4 + 16);
  *(_QWORD *)(a4 + 8) = v22;
  LODWORD(v22) = v15[6];
  v15[6] = v21;
  LODWORD(v21) = *(_DWORD *)(a4 + 20);
  *(_DWORD *)(a4 + 16) = v22;
  LODWORD(v22) = v15[7];
  v15[7] = v21;
  LODWORD(v21) = *(_DWORD *)(a4 + 24);
  *(_DWORD *)(a4 + 20) = v22;
  LODWORD(v22) = v15[8];
  v15[8] = v21;
  *(_DWORD *)(a4 + 24) = v22;
  v23 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)a1 = a2;
  v24 = 5 * v23;
  v25 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = v15;
  *(_BYTE *)(a1 + 32) = 1;
  v26 = v25 + 8 * v24;
  v27 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = v26;
  *(_QWORD *)(a1 + 8) = v27;
  return a1;
}
