// Function: sub_AE74A0
// Address: 0xae74a0
//
bool __fastcall sub_AE74A0(_QWORD *a1, _QWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a1 == *a2 )
    return a1[1] == a2[1];
  return result;
}
