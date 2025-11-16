// Function: sub_222AB10
// Address: 0x222ab10
//
int __fastcall sub_222AB10(__int64 a1)
{
  int v1; // eax

  v1 = getc(*(FILE **)(a1 + 64));
  return ungetc(v1, *(FILE **)(a1 + 64));
}
