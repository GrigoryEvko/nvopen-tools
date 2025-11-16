// Function: sub_2C0DD20
// Address: 0x2c0dd20
//
bool __fastcall sub_2C0DD20(__int64 a1, __int64 a2)
{
  bool result; // al

  result = sub_B5A760(*(_DWORD *)(a1 + 120));
  if ( result )
    return *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 16) - 1)) == a2;
  return result;
}
