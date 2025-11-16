// Function: sub_10BBBF0
// Address: 0x10bbbf0
//
bool __fastcall sub_10BBBF0(__int64 a1, int a2)
{
  unsigned int v2; // r14d
  int v3; // r13d
  bool result; // al

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2) == *(_QWORD *)a1;
  v3 = sub_C445E0(a1);
  result = 0;
  if ( v3 == a2 )
    return (unsigned int)sub_C444A0(a1) + v3 == v2;
  return result;
}
