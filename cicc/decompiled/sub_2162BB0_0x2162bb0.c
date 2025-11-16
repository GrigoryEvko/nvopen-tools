// Function: sub_2162BB0
// Address: 0x2162bb0
//
bool __fastcall sub_2162BB0(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // cx

  v2 = **(_WORD **)(a2 + 16) - 3103;
  return v2 <= 0x13u && ((1LL << v2) & 0xF3F3F) != 0;
}
