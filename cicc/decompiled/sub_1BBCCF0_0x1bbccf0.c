// Function: sub_1BBCCF0
// Address: 0x1bbccf0
//
bool __fastcall sub_1BBCCF0(__int64 a1)
{
  unsigned int v1; // ecx

  v1 = *(unsigned __int8 *)(a1 + 16) - 24;
  return v1 <= 0x1C && ((1LL << v1) & 0x1C019800) != 0;
}
