// Function: sub_1CCB220
// Address: 0x1ccb220
//
bool __fastcall sub_1CCB220(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( !(*(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8) )
    return *(_DWORD *)(**(_QWORD **)(a1 - 24) + 8LL) >> 8 == 5;
  return result;
}
