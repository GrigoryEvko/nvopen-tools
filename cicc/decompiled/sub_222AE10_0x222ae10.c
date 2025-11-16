// Function: sub_222AE10
// Address: 0x222ae10
//
wint_t __fastcall sub_222AE10(__int64 a1, wchar_t a2)
{
  if ( a2 == -1 )
    return -(fflush(*(FILE **)(a1 + 64)) != 0);
  else
    return putwc(a2, *(__FILE **)(a1 + 64));
}
