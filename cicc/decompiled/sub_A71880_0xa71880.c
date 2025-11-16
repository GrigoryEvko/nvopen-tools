// Function: sub_A71880
// Address: 0xa71880
//
bool __fastcall sub_A71880(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 4;
  return result;
}
