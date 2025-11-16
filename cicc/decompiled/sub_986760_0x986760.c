// Function: sub_986760
// Address: 0x986760
//
bool __fastcall sub_986760(__int64 a1)
{
  bool result; // al
  unsigned int v2; // ebx

  result = 1;
  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 )
  {
    if ( v2 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v2) == *(_QWORD *)a1;
    else
      return v2 == (unsigned int)sub_C445E0(a1);
  }
  return result;
}
