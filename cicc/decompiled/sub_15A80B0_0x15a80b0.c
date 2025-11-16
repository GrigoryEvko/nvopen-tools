// Function: sub_15A80B0
// Address: 0x15a80b0
//
bool __fastcall sub_15A80B0(_DWORD *a1, _DWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *(_BYTE *)a1 == *(_BYTE *)a2 && a1[1] == a2[1] )
    return ((*a2 ^ *a1) & 0xFFFFFF00) == 0;
  return result;
}
