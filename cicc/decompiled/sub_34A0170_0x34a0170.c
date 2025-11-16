// Function: sub_34A0170
// Address: 0x34a0170
//
bool __fastcall sub_34A0170(_QWORD *a1, _QWORD *a2)
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
