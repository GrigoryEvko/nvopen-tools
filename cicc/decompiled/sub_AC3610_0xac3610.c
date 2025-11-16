// Function: sub_AC3610
// Address: 0xac3610
//
bool __fastcall sub_AC3610(__int64 a1, unsigned __int64 a2)
{
  unsigned int v2; // ebx
  bool result; // al
  unsigned __int64 v4; // rax

  v2 = *(_DWORD *)(a1 + 8) >> 8;
  if ( (unsigned __int8)sub_BCAC40(a1, 1) )
    return a2 <= 1;
  result = 1;
  if ( v2 <= 0x3F )
  {
    v4 = 0;
    if ( v2 )
      v4 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v2);
    return a2 <= v4;
  }
  return result;
}
