// Function: sub_A71840
// Address: 0xa71840
//
bool __fastcall sub_A71840(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 2;
  return result;
}
