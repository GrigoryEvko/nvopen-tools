// Function: sub_31B8DE0
// Address: 0x31b8de0
//
bool __fastcall sub_31B8DE0(_QWORD *a1, _QWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a2 == *a1 && a1[1] == a2[1] )
    return a1[8] == a2[8];
  return result;
}
