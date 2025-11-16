// Function: sub_A71860
// Address: 0xa71860
//
bool __fastcall sub_A71860(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 3;
  return result;
}
