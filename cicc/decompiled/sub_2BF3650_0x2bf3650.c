// Function: sub_2BF3650
// Address: 0x2bf3650
//
__int64 __fastcall sub_2BF3650(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  char v6; // di
  int v7; // edi
  __int64 v8; // r8
  int v9; // esi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  unsigned int v14; // esi
  unsigned int v15; // edx
  int v16; // ecx
  unsigned int v17; // r8d
  __int64 *v18; // rdx
  int v19; // r11d
  __int64 *v20; // r10
  __int64 v21; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v22; // [rsp+8h] [rbp-18h] BYREF

  v3 = sub_2BF09E0(*(_QWORD *)(a2 + 80));
  v4 = 0;
  if ( *(_DWORD *)(v3 + 64) == 1 )
    v4 = **(_QWORD **)(v3 + 56);
  v5 = sub_2BF0520(v4);
  v6 = *(_BYTE *)(a1 + 32);
  v21 = v5;
  v7 = v6 & 1;
  if ( v7 )
  {
    v8 = a1 + 40;
    v9 = 3;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 48);
    v8 = *(_QWORD *)(a1 + 40);
    if ( !v14 )
    {
      v15 = *(_DWORD *)(a1 + 32);
      ++*(_QWORD *)(a1 + 24);
      v22 = 0;
      v16 = (v15 >> 1) + 1;
LABEL_10:
      v17 = 3 * v14;
      goto LABEL_11;
    }
    v9 = v14 - 1;
  }
  v10 = v9 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v11 = (__int64 *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v5 == *v11 )
    return v11[1];
  v19 = 1;
  v20 = 0;
  while ( v12 != -4096 )
  {
    if ( !v20 && v12 == -8192 )
      v20 = v11;
    v10 = v9 & (v19 + v10);
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v5 == *v11 )
      return v11[1];
    ++v19;
  }
  v17 = 12;
  v14 = 4;
  if ( !v20 )
    v20 = v11;
  v15 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 24);
  v22 = v20;
  v16 = (v15 >> 1) + 1;
  if ( !(_BYTE)v7 )
  {
    v14 = *(_DWORD *)(a1 + 48);
    goto LABEL_10;
  }
LABEL_11:
  if ( 4 * v16 >= v17 )
  {
    v14 *= 2;
    goto LABEL_17;
  }
  if ( v14 - *(_DWORD *)(a1 + 36) - v16 <= v14 >> 3 )
  {
LABEL_17:
    sub_2ACA3E0(a1 + 24, v14);
    sub_2ABFB80(a1 + 24, &v21, &v22);
    v5 = v21;
    v15 = *(_DWORD *)(a1 + 32);
  }
  *(_DWORD *)(a1 + 32) = (2 * (v15 >> 1) + 2) | v15 & 1;
  v18 = v22;
  if ( *v22 != -4096 )
    --*(_DWORD *)(a1 + 36);
  *v18 = v5;
  v18[1] = 0;
  return 0;
}
