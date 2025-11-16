// Function: sub_CEFE10
// Address: 0xcefe10
//
bool __fastcall sub_CEFE10(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( !(*(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) >> 8) )
    return *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL) + 8LL) >> 8 == 5;
  return result;
}
