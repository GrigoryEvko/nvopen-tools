// Function: sub_302E170
// Address: 0x302e170
//
bool __fastcall sub_302E170(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  bool result; // al

  result = 0;
  if ( a3 )
    return *(_QWORD *)(a1 + 8LL * a3 + 112) != 0;
  return result;
}
