// Function: sub_24A5F10
// Address: 0x24a5f10
//
__int64 __fastcall sub_24A5F10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 **v3; // rax
  __int64 v5; // rsi
  __int64 *v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r11
  unsigned int v13; // ecx
  __int64 v14; // rsi
  __int64 v15; // r8
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 result; // rax
  int v20; // edx
  int v21; // eax
  int v22; // r10d
  int v23; // ebx

  v3 = *(__int64 ***)a2;
  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( (__int64 **)v5 == v3 )
LABEL_20:
    BUG();
  while ( 1 )
  {
    v6 = *v3;
    if ( !*((_BYTE *)*v3 + 40) )
      break;
    if ( (__int64 **)v5 == ++v3 )
      goto LABEL_20;
  }
  v6[4] = a3;
  *((_BYTE *)v6 + 40) = 1;
  v7 = *(_DWORD *)(a1 + 296);
  v8 = *(_QWORD *)(a1 + 280);
  v9 = **v3;
  if ( v7 )
  {
    v10 = (v7 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v9 == *v11 )
      goto LABEL_7;
    v20 = 1;
    while ( v12 != -4096 )
    {
      v23 = v20 + 1;
      v10 = (v7 - 1) & (v20 + v10);
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v9 == *v11 )
        goto LABEL_7;
      v20 = v23;
    }
  }
  v11 = (__int64 *)(v8 + 16LL * v7);
LABEL_7:
  --*(_DWORD *)(v11[1] + 36);
  v13 = *(_DWORD *)(a1 + 296);
  v14 = *(_QWORD *)(a1 + 280);
  v15 = (*v3)[1];
  if ( v13 )
  {
    v16 = (v13 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v17 = (__int64 *)(v14 + 16LL * v16);
    v18 = *v17;
    if ( v15 == *v17 )
    {
LABEL_9:
      result = v17[1];
      --*(_DWORD *)(result + 32);
      return result;
    }
    v21 = 1;
    while ( v18 != -4096 )
    {
      v22 = v21 + 1;
      v16 = (v13 - 1) & (v21 + v16);
      v17 = (__int64 *)(v14 + 16LL * v16);
      v18 = *v17;
      if ( v15 == *v17 )
        goto LABEL_9;
      v21 = v22;
    }
  }
  result = *(_QWORD *)(v14 + 16LL * v13 + 8);
  --*(_DWORD *)(result + 32);
  return result;
}
