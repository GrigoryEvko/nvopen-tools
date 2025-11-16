// Function: sub_2E8D7A0
// Address: 0x2e8d7a0
//
__int64 __fastcall sub_2E8D7A0(__int64 a1, int a2)
{
  _BYTE *v2; // rbx
  __int64 result; // rax
  _BYTE *v4; // r12
  _BYTE *v5; // r14
  _BYTE *v6; // rbx

  v2 = *(_BYTE **)(a1 + 32);
  result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  v4 = &v2[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)];
  if ( v2 != v4 )
  {
    while ( 1 )
    {
      v5 = v2;
      result = sub_2DADC00(v2);
      if ( (_BYTE)result )
        break;
      v2 += 40;
      if ( v4 == v2 )
        return result;
    }
    while ( v4 != v5 )
    {
      if ( a2 == *((_DWORD *)v5 + 2) )
        v5[3] &= ~0x40u;
      v6 = v5 + 40;
      if ( v5 + 40 == v4 )
        break;
      while ( 1 )
      {
        v5 = v6;
        result = sub_2DADC00(v6);
        if ( (_BYTE)result )
          break;
        v6 += 40;
        if ( v4 == v6 )
          return result;
      }
    }
  }
  return result;
}
