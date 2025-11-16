// Function: sub_C33720
// Address: 0xc33720
//
bool __fastcall sub_C33720(_DWORD *a1, _DWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a1 < *a2 && a1[1] > a2[1] )
    return a2[2] >= a1[2];
  return result;
}
