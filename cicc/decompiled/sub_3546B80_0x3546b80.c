// Function: sub_3546B80
// Address: 0x3546b80
//
__int64 __fastcall sub_3546B80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 *v4; // rdx
  __int64 *v5; // r8
  int v6; // edi
  __int64 v7; // r9
  __int64 v8; // rsi
  int v9; // r10d
  int v10; // r11d
  unsigned int v11; // eax
  __int64 v12; // rcx

  v2 = *(unsigned int *)(a1 + 40);
  v3 = 0;
  if ( (unsigned int)v2 > *(_DWORD *)(a2 + 40) )
    return v3;
  v4 = *(__int64 **)(a1 + 32);
  v5 = &v4[v2];
  if ( v5 == v4 )
    return 1;
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *v4;
  v9 = v6 - 1;
  if ( !v6 )
    return 0;
  while ( 1 )
  {
    v10 = 1;
    v11 = v9 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = *(_QWORD *)(v7 + 8LL * v11);
    if ( v8 != v12 )
      break;
LABEL_5:
    if ( v5 == ++v4 )
      return 1;
    v8 = *v4;
  }
  while ( v12 != -4096 )
  {
    v11 = v9 & (v10 + v11);
    v12 = *(_QWORD *)(v7 + 8LL * v11);
    if ( v8 == v12 )
      goto LABEL_5;
    ++v10;
  }
  return 0;
}
