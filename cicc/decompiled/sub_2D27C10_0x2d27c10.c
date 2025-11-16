// Function: sub_2D27C10
// Address: 0x2d27c10
//
bool __fastcall sub_2D27C10(_QWORD *a1, _QWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a1 == *a2 )
    return a1[1] == a2[1];
  return result;
}
