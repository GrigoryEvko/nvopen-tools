// Function: sub_1F6C830
// Address: 0x1f6c830
//
bool __fastcall sub_1F6C830(__int64 a1, unsigned int a2, unsigned __int8 a3)
{
  bool result; // al

  if ( a3 == 1 || (result = 0, a3) && *(_QWORD *)(a1 + 8LL * a3 + 120) )
  {
    result = 0;
    if ( a2 <= 0x102 )
      return *(_BYTE *)(a2 + 259LL * a3 + a1 + 2422) == 0;
  }
  return result;
}
