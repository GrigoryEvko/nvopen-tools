// Function: sub_2241C50
// Address: 0x2241c50
//
bool __fastcall sub_2241C50(__int64 a1, __int64 a2, int a3)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)(a2 + 8) == a1 )
    return *(_DWORD *)a2 == a3;
  return result;
}
