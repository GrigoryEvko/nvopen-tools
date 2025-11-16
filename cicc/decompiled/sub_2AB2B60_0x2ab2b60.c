// Function: sub_2AB2B60
// Address: 0x2ab2b60
//
__int64 __fastcall sub_2AB2B60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  unsigned int v4; // edi
  int v5; // ebx
  unsigned int i; // eax
  __int64 v7; // rdx
  unsigned int v8; // eax
  int v9; // eax
  __int64 v10; // rdi
  int v11; // edx
  unsigned int v12; // eax
  __int64 v13; // rcx
  int v15; // r8d

  v3 = *(_QWORD *)(a1 + 136);
  v4 = *(_DWORD *)(a1 + 152);
  if ( v4 )
  {
    v5 = 1;
    for ( i = (v4 - 1) & ((BYTE4(a3) == 0) + 37 * a3 - 1); ; i = (v4 - 1) & v8 )
    {
      v7 = v3 + 40LL * i;
      if ( (_DWORD)a3 == *(_DWORD *)v7 && BYTE4(a3) == *(_BYTE *)(v7 + 4) )
        break;
      if ( *(_DWORD *)v7 == -1 && *(_BYTE *)(v7 + 4) )
        goto LABEL_7;
      v8 = v5 + i;
      ++v5;
    }
  }
  else
  {
LABEL_7:
    v7 = v3 + 40LL * v4;
  }
  v9 = *(_DWORD *)(v7 + 32);
  v10 = *(_QWORD *)(v7 + 16);
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = *(_QWORD *)(v10 + 24LL * v12);
    if ( a2 == v13 )
      return 1;
    v15 = 1;
    while ( v13 != -4096 )
    {
      v12 = v11 & (v15 + v12);
      v13 = *(_QWORD *)(v10 + 24LL * v12);
      if ( a2 == v13 )
        return 1;
      ++v15;
    }
  }
  return 0;
}
