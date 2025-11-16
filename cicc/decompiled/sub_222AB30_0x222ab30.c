// Function: sub_222AB30
// Address: 0x222ab30
//
int __fastcall sub_222AB30(__int64 a1, int a2)
{
  int result; // eax

  if ( a2 == -1 )
  {
    result = *(_DWORD *)(a1 + 72);
    if ( result != -1 )
      result = ungetc(result, *(FILE **)(a1 + 64));
    *(_DWORD *)(a1 + 72) = -1;
  }
  else
  {
    result = ungetc(a2, *(FILE **)(a1 + 64));
    *(_DWORD *)(a1 + 72) = -1;
  }
  return result;
}
