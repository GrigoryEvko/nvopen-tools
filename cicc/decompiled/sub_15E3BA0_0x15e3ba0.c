// Function: sub_15E3BA0
// Address: 0x15e3ba0
//
__int16 __fastcall sub_15E3BA0(__int64 a1, char a2, char a3)
{
  int v3; // eax

  LOWORD(v3) = *(_WORD *)(a1 + 18) & ~(1 << a2);
  if ( a3 )
    v3 = (1 << a2) | *(unsigned __int16 *)(a1 + 18);
  *(_WORD *)(a1 + 18) = v3;
  return v3;
}
