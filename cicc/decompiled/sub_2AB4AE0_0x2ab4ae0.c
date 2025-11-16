// Function: sub_2AB4AE0
// Address: 0x2ab4ae0
//
bool __fastcall sub_2AB4AE0(_DWORD *a1, _BYTE *a2)
{
  bool result; // al

  result = 1;
  if ( *a2 != 61 )
  {
    result = 0;
    if ( *a2 == 62 )
      return *a1 > (unsigned int)dword_500D9E8;
  }
  return result;
}
