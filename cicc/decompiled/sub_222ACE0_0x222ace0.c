// Function: sub_222ACE0
// Address: 0x222ace0
//
wint_t __fastcall sub_222ACE0(__int64 a1, wint_t a2)
{
  wint_t result; // eax

  if ( a2 == -1 )
  {
    result = *(_DWORD *)(a1 + 72);
    if ( result != -1 )
      result = ungetwc(result, *(__FILE **)(a1 + 64));
    *(_DWORD *)(a1 + 72) = -1;
  }
  else
  {
    result = ungetwc(a2, *(__FILE **)(a1 + 64));
    *(_DWORD *)(a1 + 72) = -1;
  }
  return result;
}
