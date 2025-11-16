// Function: sub_386EDD0
// Address: 0x386edd0
//
bool __fastcall sub_386EDD0(__int64 a1, __int64 a2)
{
  char v2; // dl
  bool result; // al
  unsigned int v4; // ebx

  v2 = *(_BYTE *)(a2 + 8);
  result = v2 == 11;
  if ( *(_BYTE *)(a1 + 8) == 11 )
  {
    result = 0;
    if ( v2 == 11 )
    {
      v4 = sub_1643030(a2);
      return v4 < (unsigned int)sub_1643030(a1);
    }
  }
  return result;
}
