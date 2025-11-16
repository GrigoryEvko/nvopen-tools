// Function: sub_1918850
// Address: 0x1918850
//
__int64 __fastcall sub_1918850(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  char v9; // di
  int v10; // edi
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v17; // esi
  unsigned int v18; // eax
  __int64 v19; // r13
  int v20; // ecx
  unsigned int v21; // r8d
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 *v24; // rax
  int v25; // r10d
  __int64 v26; // [rsp+8h] [rbp-38h] BYREF
  __int64 v27; // [rsp+10h] [rbp-30h] BYREF
  int v28; // [rsp+18h] [rbp-28h]

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8);
  v28 = 0;
  v27 = v8;
  v10 = v9 & 1;
  if ( v10 )
  {
    v11 = a1 + 16;
    v12 = 3;
  }
  else
  {
    v17 = *(_DWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    if ( !v17 )
    {
      v18 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v19 = 0;
      v20 = (v18 >> 1) + 1;
LABEL_9:
      v21 = 3 * v17;
      goto LABEL_10;
    }
    v12 = v17 - 1;
  }
  v13 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v11 + 16LL * v13;
  a6 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_4:
    v15 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 80) + 16 * v15 + 8;
  }
  v25 = 1;
  v19 = 0;
  while ( a6 != -8 )
  {
    if ( !v19 && a6 == -16 )
      v19 = v14;
    v13 = v12 & (v25 + v13);
    v14 = v11 + 16LL * v13;
    a6 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_4;
    ++v25;
  }
  if ( !v19 )
    v19 = v14;
  v18 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v20 = (v18 >> 1) + 1;
  if ( !(_BYTE)v10 )
  {
    v17 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v21 = 12;
  v17 = 4;
LABEL_10:
  if ( 4 * v20 >= v21 )
  {
    v17 *= 2;
    goto LABEL_24;
  }
  if ( v17 - *(_DWORD *)(a1 + 12) - v20 <= v17 >> 3 )
  {
LABEL_24:
    sub_1918480(a1, v17);
    sub_190F900(a1, &v27, &v26);
    v19 = v26;
    v8 = v27;
    v18 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *(_QWORD *)v19 != -8 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v19 = v8;
  *(_DWORD *)(v19 + 8) = v28;
  v22 = *a2;
  v23 = *(unsigned int *)(a1 + 88);
  if ( (unsigned int)v23 >= *(_DWORD *)(a1 + 92) )
  {
    sub_16CD150(a1 + 80, (const void *)(a1 + 96), 0, 16, v21, a6);
    v23 = *(unsigned int *)(a1 + 88);
  }
  v24 = (__int64 *)(*(_QWORD *)(a1 + 80) + 16 * v23);
  *v24 = v22;
  v24[1] = 0;
  v15 = *(unsigned int *)(a1 + 88);
  *(_DWORD *)(a1 + 88) = v15 + 1;
  *(_DWORD *)(v19 + 8) = v15;
  return *(_QWORD *)(a1 + 80) + 16 * v15 + 8;
}
