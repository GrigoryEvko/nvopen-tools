// Function: sub_222ACC0
// Address: 0x222acc0
//
wint_t __fastcall sub_222ACC0(__int64 a1)
{
  wint_t v1; // eax

  v1 = getwc(*(__FILE **)(a1 + 64));
  return ungetwc(v1, *(__FILE **)(a1 + 64));
}
