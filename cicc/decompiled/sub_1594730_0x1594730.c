// Function: sub_1594730
// Address: 0x1594730
//
bool __fastcall sub_1594730(__int64 a1, unsigned __int64 a2)
{
  unsigned int v2; // ebx
  bool result; // al

  v2 = *(_DWORD *)(a1 + 8) >> 8;
  if ( (unsigned __int8)sub_1642F90(a1, 1) )
    return a2 <= 1;
  result = 1;
  if ( v2 <= 0x3F )
    return a2 <= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v2);
  return result;
}
