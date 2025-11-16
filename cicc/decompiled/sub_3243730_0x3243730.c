// Function: sub_3243730
// Address: 0x3243730
//
__int64 __fastcall sub_3243730(__int64 a1)
{
  unsigned int v1; // esi

  v1 = *(unsigned __int16 *)(a1 + 98);
  if ( (_WORD)v1 )
    sub_3242330(a1, v1);
  return sub_3242360(a1, (1 << *(_WORD *)(a1 + 96)) - 1);
}
