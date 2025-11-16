// Function: sub_1F6BB70
// Address: 0x1f6bb70
//
bool __fastcall sub_1F6BB70(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  bool result; // al

  result = 0;
  if ( a3 )
    return *(_QWORD *)(a1 + 8LL * a3 + 120) != 0;
  return result;
}
