// Function: sub_2E8D840
// Address: 0x2e8d840
//
__int64 __fastcall sub_2E8D840(__int64 a1, int a2, char a3)
{
  _BYTE *v3; // rbx
  __int64 result; // rax
  _BYTE *v5; // r12
  _BYTE *v7; // r15
  int v8; // r14d
  _BYTE *v9; // rbx

  v3 = *(_BYTE **)(a1 + 32);
  result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  v5 = &v3[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)];
  if ( v3 != v5 )
  {
    while ( 1 )
    {
      v7 = v3;
      result = sub_2DADC00(v3);
      if ( (_BYTE)result )
        break;
      v3 += 40;
      if ( v5 == v3 )
        return result;
    }
    if ( v5 != v3 )
    {
      v8 = a3 & 1;
      do
      {
        if ( a2 == *((_DWORD *)v7 + 2) && (*(_DWORD *)v7 & 0xFFF00) != 0 )
        {
          result = v8 | v7[4] & 0xFEu;
          v7[4] = v8 | v7[4] & 0xFE;
        }
        v9 = v7 + 40;
        if ( v7 + 40 == v5 )
          break;
        while ( 1 )
        {
          v7 = v9;
          result = sub_2DADC00(v9);
          if ( (_BYTE)result )
            break;
          v9 += 40;
          if ( v5 == v9 )
            return result;
        }
      }
      while ( v5 != v9 );
    }
  }
  return result;
}
