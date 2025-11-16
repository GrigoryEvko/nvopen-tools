// Function: sub_16C36A0
// Address: 0x16c36a0
//
bool __fastcall sub_16C36A0(_QWORD *a1, _QWORD *a2)
{
  bool result; // al

  result = 0;
  if ( *a1 == *a2 )
    return a1[4] == a2[4];
  return result;
}
