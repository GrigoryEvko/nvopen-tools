// Function: sub_2B149F0
// Address: 0x2b149f0
//
bool __fastcall sub_2B149F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r10
  int v10; // ecx
  int v11; // r12d
  unsigned int v12; // edx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // ecx

  if ( a2 != a1 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(a1 + 24);
      if ( v14 != a7 && a8 != v14 )
      {
        v15 = *(_QWORD *)(a9 + 24);
        if ( (*(_BYTE *)(v15 + 88) & 1) != 0 )
        {
          v9 = v15 + 96;
          v10 = 3;
        }
        else
        {
          v16 = *(_DWORD *)(v15 + 104);
          v9 = *(_QWORD *)(v15 + 96);
          if ( !v16 )
            return a2 == a1;
          v10 = v16 - 1;
        }
        v11 = 1;
        v12 = v10 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v13 = *(_QWORD *)(v9 + 72LL * v12);
        if ( v14 != v13 )
          break;
      }
LABEL_5:
      a1 = *(_QWORD *)(a1 + 8);
      if ( a2 == a1 )
        return a2 == a1;
    }
    while ( v13 != -4096 )
    {
      v12 = v10 & (v11 + v12);
      v13 = *(_QWORD *)(v9 + 72LL * v12);
      if ( v14 == v13 )
        goto LABEL_5;
      ++v11;
    }
  }
  return a2 == a1;
}
