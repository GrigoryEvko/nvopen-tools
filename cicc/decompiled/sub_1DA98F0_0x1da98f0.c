// Function: sub_1DA98F0
// Address: 0x1da98f0
//
bool __fastcall sub_1DA98F0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // edx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  char v12; // dl
  unsigned __int64 v13; // rax
  __int64 v14; // rdx

  v5 = *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 8) + 16 * v5 - 16;
  v7 = *(_DWORD *)(v6 + 12);
  if ( !*(_DWORD *)(*(_QWORD *)a1 + 80LL) )
  {
    if ( v7 )
    {
      v9 = *(_QWORD *)v6;
      v10 = (unsigned int)(v7 - 1);
      if ( ((a3 ^ *(_DWORD *)(v9 + 4 * v10 + 64)) & 0x7FFFFFFF) == 0
        && ((*(_BYTE *)(v9 + 4 * (v10 + 16) + 3) ^ HIBYTE(a3)) & 0x80u) == 0 )
      {
        return *(_QWORD *)(v9 + 16 * v10 + 8) == a2;
      }
    }
    return 0;
  }
  if ( v7 )
  {
    v9 = *(_QWORD *)v6;
    v10 = (unsigned int)(v7 - 1);
    if ( ((a3 ^ *(_DWORD *)(v9 + 4 * v10 + 144)) & 0x7FFFFFFF) == 0
      && ((*(_BYTE *)(v9 + 4 * (v10 + 36) + 3) ^ HIBYTE(a3)) & 0x80u) == 0 )
    {
      return *(_QWORD *)(v9 + 16 * v10 + 8) == a2;
    }
    return 0;
  }
  v11 = sub_3945DA0(a1 + 8, (unsigned int)(v5 - 1));
  if ( !v11 )
    return 0;
  v12 = v11;
  v13 = v11 & 0xFFFFFFFFFFFFFFC0LL;
  v14 = v12 & 0x3F;
  if ( ((a3 ^ *(_DWORD *)(v13 + 4 * v14 + 144)) & 0x7FFFFFFF) != 0
    || ((*(_BYTE *)(v13 + 4 * (v14 + 36) + 3) ^ HIBYTE(a3)) & 0x80u) != 0 )
  {
    return 0;
  }
  return *(_QWORD *)(v13 + 16 * v14 + 8) == a2;
}
