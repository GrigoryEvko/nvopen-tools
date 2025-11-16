// Function: sub_2C0D510
// Address: 0x2c0d510
//
bool __fastcall sub_2C0D510(__int64 a1, __int64 a2)
{
  bool result; // al

  result = sub_B5A760(*(_DWORD *)(a1 + 160));
  if ( result )
    return *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1)) == a2;
  return result;
}
