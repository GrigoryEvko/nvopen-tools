// Function: sub_A71820
// Address: 0xa71820
//
bool __fastcall sub_A71820(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 1;
  return result;
}
