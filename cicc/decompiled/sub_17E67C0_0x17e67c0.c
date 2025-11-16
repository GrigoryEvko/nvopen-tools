// Function: sub_17E67C0
// Address: 0x17e67c0
//
__int64 __fastcall sub_17E67C0(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 i; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // esi
  __int64 *v10; // rdx
  __int64 v11; // r11
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r8
  __int64 result; // rax
  int v19; // edx
  int v20; // eax
  int v21; // ebx
  int v22; // r10d

  v4 = *a2;
  for ( i = **a2; *(_BYTE *)(i + 27); ++v4 )
    i = v4[1];
  *(_QWORD *)(i + 32) = a3;
  *(_BYTE *)(i + 27) = 1;
  v6 = *(unsigned int *)(a1 + 296);
  v7 = *(_QWORD *)(a1 + 280);
  if ( (_DWORD)v6 )
  {
    v8 = *(_QWORD *)*v4;
    v9 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v8 == *v10 )
      goto LABEL_5;
    v19 = 1;
    while ( v11 != -8 )
    {
      v21 = v19 + 1;
      v9 = (v6 - 1) & (v19 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v8 == *v10 )
        goto LABEL_5;
      v19 = v21;
    }
  }
  v10 = (__int64 *)(v7 + 16 * v6);
LABEL_5:
  --*(_DWORD *)(v10[1] + 32);
  v12 = *(unsigned int *)(a1 + 296);
  v13 = *(_QWORD *)(a1 + 280);
  if ( (_DWORD)v12 )
  {
    v14 = *(_QWORD *)(*v4 + 8);
    v15 = (v12 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( *v16 == v14 )
    {
LABEL_7:
      result = v16[1];
      --*(_DWORD *)(result + 28);
      return result;
    }
    v20 = 1;
    while ( v17 != -8 )
    {
      v22 = v20 + 1;
      v15 = (v12 - 1) & (v20 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v14 == *v16 )
        goto LABEL_7;
      v20 = v22;
    }
  }
  result = *(_QWORD *)(v13 + 16 * v12 + 8);
  --*(_DWORD *)(result + 28);
  return result;
}
