// Function: sub_1642F90
// Address: 0x1642f90
//
bool __fastcall sub_1642F90(__int64 a1, int a2)
{
  bool result; // al

  result = 0;
  if ( *(_BYTE *)(a1 + 8) == 11 )
    return *(_DWORD *)(a1 + 8) >> 8 == a2;
  return result;
}
