// Function: sub_3243300
// Address: 0x3243300
//
__int64 __fastcall sub_3243300(__int64 a1, unsigned __int64 a2)
{
  *(_BYTE *)(a1 + 100) = *(_BYTE *)(a1 + 100) & 0xF8 | 3;
  return sub_3242070(a1, a2);
}
