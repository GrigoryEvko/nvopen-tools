// Function: sub_C80200
// Address: 0xc80200
//
bool __fastcall sub_C80200(_QWORD *a1, _QWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a1 == *a2 )
    return a1[4] == a2[4];
  return result;
}
