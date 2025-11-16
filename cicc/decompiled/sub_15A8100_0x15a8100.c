// Function: sub_15A8100
// Address: 0x15a8100
//
bool __fastcall sub_15A8100(_DWORD *a1, _DWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a1 == *a2 && a1[3] == a2[3] && a1[1] == a2[1] && a1[2] == a2[2] )
    return a1[4] == a2[4];
  return result;
}
