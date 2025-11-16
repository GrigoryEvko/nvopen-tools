// Function: sub_222AAF0
// Address: 0x222aaf0
//
int __fastcall sub_222AAF0(__int64 a1)
{
  int result; // eax

  result = getc(*(FILE **)(a1 + 64));
  *(_DWORD *)(a1 + 72) = result;
  return result;
}
