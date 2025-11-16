// Function: sub_B2E700
// Address: 0xb2e700
//
__int16 __fastcall sub_B2E700(__int64 a1, char a2, char a3)
{
  int v3; // eax

  LOWORD(v3) = *(_WORD *)(a1 + 2) & ~(1 << a2);
  if ( a3 )
    v3 = (1 << a2) | *(unsigned __int16 *)(a1 + 2);
  *(_WORD *)(a1 + 2) = v3;
  return v3;
}
