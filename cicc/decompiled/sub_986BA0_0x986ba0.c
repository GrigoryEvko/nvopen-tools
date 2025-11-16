// Function: sub_986BA0
// Address: 0x986ba0
//
bool __fastcall sub_986BA0(__int64 a1)
{
  bool result; // al

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    return (unsigned int)sub_C44630(a1) == 1;
  result = 0;
  if ( *(_QWORD *)a1 )
    return (*(_QWORD *)a1 & (*(_QWORD *)a1 - 1LL)) == 0;
  return result;
}
