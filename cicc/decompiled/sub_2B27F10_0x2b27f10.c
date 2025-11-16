// Function: sub_2B27F10
// Address: 0x2b27f10
//
bool __fastcall sub_2B27F10(__int64 a1)
{
  bool result; // al
  _BYTE **v2; // rdx
  int v3; // eax
  unsigned int v4; // ecx
  int v5; // edx

  result = 0;
  if ( *(_BYTE *)a1 == 86 )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      v2 = *(_BYTE ***)(a1 - 8);
    else
      v2 = (_BYTE **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    result = 0;
    if ( (unsigned __int8)(**v2 - 82) <= 1u )
    {
      v3 = sub_2B27770(a1);
      v4 = v3 - 6;
      v5 = v3;
      result = 1;
      if ( v4 > 3 )
        return (unsigned int)(v5 - 12) <= 3;
    }
  }
  return result;
}
