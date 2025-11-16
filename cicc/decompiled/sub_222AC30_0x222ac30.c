// Function: sub_222AC30
// Address: 0x222ac30
//
wint_t __fastcall sub_222AC30(__int64 a1)
{
  wint_t result; // eax

  result = getwc(*(__FILE **)(a1 + 64));
  *(_DWORD *)(a1 + 72) = result;
  return result;
}
