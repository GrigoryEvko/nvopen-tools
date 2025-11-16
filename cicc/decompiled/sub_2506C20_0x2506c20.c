// Function: sub_2506C20
// Address: 0x2506c20
//
bool __fastcall sub_2506C20(_BYTE *a1)
{
  bool result; // al

  result = 0;
  if ( a1[8] != 0xFF && a1[9] != 0xFF && a1[10] != 0xFF )
    return a1[11] != 0xFF;
  return result;
}
