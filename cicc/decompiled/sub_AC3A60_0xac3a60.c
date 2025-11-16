// Function: sub_AC3A60
// Address: 0xac3a60
//
_BYTE *__fastcall sub_AC3A60(__int64 a1, char a2)
{
  unsigned int v2; // edx
  _BYTE *v3; // r8
  _BYTE **v4; // rax

  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v3 = *(_BYTE **)(a1 - 32LL * v2);
  if ( v2 <= 1 )
    return v3;
  v4 = (_BYTE **)(a1 - 32LL * v2 + 32);
  do
  {
    if ( *v4 != v3 )
    {
      if ( !a2 )
        return 0;
      if ( **v4 != 13 )
      {
        if ( *v3 != 13 )
          return 0;
        v3 = *v4;
      }
    }
    v4 += 4;
  }
  while ( v4 != (_BYTE **)(a1 + 32 * (v2 - 2 - (unsigned __int64)v2) + 64) );
  return v3;
}
