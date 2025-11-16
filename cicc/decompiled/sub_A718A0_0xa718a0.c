// Function: sub_A718A0
// Address: 0xa718a0
//
bool __fastcall sub_A718A0(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 5;
  return result;
}
