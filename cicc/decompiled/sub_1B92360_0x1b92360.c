// Function: sub_1B92360
// Address: 0x1b92360
//
bool __fastcall sub_1B92360(_DWORD *a1, __int64 a2)
{
  char v2; // dl
  bool result; // al

  v2 = *(_BYTE *)(a2 + 16);
  result = 1;
  if ( v2 != 54 )
  {
    result = 0;
    if ( v2 == 55 )
      return *a1 > (unsigned int)dword_4FB83C0;
  }
  return result;
}
