// Function: sub_1AFCC80
// Address: 0x1afcc80
//
__int64 __fastcall sub_1AFCC80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v7; // rax
  int v8; // r8d
  unsigned int v9; // edx
  __int64 *v10; // rcx
  __int64 v11; // r10
  __int64 *v12; // rdx
  __int64 result; // rax
  unsigned int v14; // r9d
  __int64 *v15; // rcx
  __int64 v16; // r10
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // ecx
  int v20; // ecx
  int v21; // r11d
  int v22; // r11d

  v5 = *(_QWORD *)(a1 + 32);
  v7 = *(unsigned int *)(a1 + 48);
  if ( !(_DWORD)v7 )
    return 0;
  v8 = v7 - 1;
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v5 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    v12 = (__int64 *)(v5 + 16 * v7);
    if ( v10 != v12 )
    {
      result = v10[1];
      goto LABEL_5;
    }
  }
  else
  {
    v20 = 1;
    while ( v11 != -8 )
    {
      v22 = v20 + 1;
      v9 = v8 & (v20 + v9);
      v10 = (__int64 *)(v5 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v20 = v22;
    }
    v12 = (__int64 *)(v5 + 16LL * (unsigned int)v7);
  }
  result = 0;
LABEL_5:
  v14 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v5 + 16LL * v14);
  v16 = *v15;
  if ( a3 != *v15 )
  {
    v19 = 1;
    while ( v16 != -8 )
    {
      v21 = v19 + 1;
      v14 = v8 & (v19 + v14);
      v15 = (__int64 *)(v5 + 16LL * v14);
      v16 = *v15;
      if ( a3 == *v15 )
        goto LABEL_6;
      v19 = v21;
    }
    return 0;
  }
LABEL_6:
  if ( v15 == v12 )
    return 0;
  v17 = v15[1];
  if ( !v17 || !result )
    return 0;
  while ( v17 != result )
  {
    if ( *(_DWORD *)(result + 16) < *(_DWORD *)(v17 + 16) )
    {
      v18 = result;
      result = v17;
      v17 = v18;
    }
    result = *(_QWORD *)(result + 8);
    if ( !result )
      return result;
  }
  return *(_QWORD *)result;
}
