// Function: sub_A71800
// Address: 0xa71800
//
bool __fastcall sub_A71800(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 0;
  return result;
}
