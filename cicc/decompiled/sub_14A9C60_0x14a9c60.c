// Function: sub_14A9C60
// Address: 0x14a9c60
//
bool __fastcall sub_14A9C60(__int64 a1)
{
  bool result; // al

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    return (unsigned int)sub_16A5940(a1) == 1;
  result = 0;
  if ( *(_QWORD *)a1 )
    return (*(_QWORD *)a1 & (*(_QWORD *)a1 - 1LL)) == 0;
  return result;
}
