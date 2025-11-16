// Function: sub_CEFE40
// Address: 0xcefe40
//
bool __fastcall sub_CEFE40(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( !(*(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) >> 8) )
    return *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL) + 8LL) >> 8 == 3;
  return result;
}
