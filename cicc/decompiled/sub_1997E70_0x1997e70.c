// Function: sub_1997E70
// Address: 0x1997e70
//
__int64 __fastcall sub_1997E70(__int64 a1)
{
  unsigned int v1; // esi
  unsigned __int64 v2; // rdx
  unsigned int v3; // ebx
  __int64 v4; // rax
  int v5; // ecx
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 8);
  v2 = *(_QWORD *)a1;
  v3 = v1 + 1;
  v4 = 1LL << ((unsigned __int8)v1 - 1);
  if ( v1 > 0x40 )
  {
    if ( (*(_QWORD *)(v2 + 8LL * ((v1 - 1) >> 6)) & v4) != 0 )
    {
      v5 = sub_16A5810(a1);
      return v3 - v5;
    }
    return v3 - (unsigned int)sub_16A57B0(a1);
  }
  else
  {
    if ( (v4 & v2) != 0 )
    {
      v5 = 64;
      v6 = ~(v2 << (64 - (unsigned __int8)v1));
      if ( v6 )
      {
        _BitScanReverse64(&v7, v6);
        v5 = v7 ^ 0x3F;
      }
      return v3 - v5;
    }
    result = 1;
    if ( v2 )
    {
      _BitScanReverse64(&v2, v2);
      return 65 - ((unsigned int)v2 ^ 0x3F);
    }
  }
  return result;
}
