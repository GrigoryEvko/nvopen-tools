// Function: sub_222ABE0
// Address: 0x222abe0
//
__off64_t __fastcall sub_222ABE0(__int64 a1, __off64_t a2, int a3)
{
  int v3; // r8d
  int v4; // r8d
  __off64_t result; // rax

  v3 = 0;
  if ( a3 )
    v3 = (a3 != 1) + 1;
  v4 = fseeko64(*(FILE **)(a1 + 64), a2, v3);
  result = -1;
  if ( !v4 )
    return ftello64(*(FILE **)(a1 + 64));
  return result;
}
