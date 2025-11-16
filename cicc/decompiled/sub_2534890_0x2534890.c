// Function: sub_2534890
// Address: 0x2534890
//
bool __fastcall sub_2534890(_QWORD *a1, _QWORD *a2)
{
  bool result; // al

  result = 1;
  if ( *a1 >= *a2 )
  {
    result = 0;
    if ( *a1 == *a2 )
      return a1[1] < a2[1];
  }
  return result;
}
