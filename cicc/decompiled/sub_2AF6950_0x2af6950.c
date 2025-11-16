// Function: sub_2AF6950
// Address: 0x2af6950
//
bool __fastcall sub_2AF6950(__int64 a1, unsigned int a2, __int64 a3, char a4, _DWORD *a5)
{
  bool result; // al
  char v7; // r8

  *a5 = 0;
  result = 0;
  if ( (1LL << a4) % (unsigned __int64)a2 )
  {
    v7 = sub_DFAE90(*(_QWORD *)(a1 + 40));
    result = 1;
    if ( v7 )
      return *a5 == 0;
  }
  return result;
}
