// Function: sub_B123F0
// Address: 0xb123f0
//
__int64 __fastcall sub_B123F0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  char v4; // al

  v4 = *(_BYTE *)(a1 + 32);
  if ( !v4 )
    return sub_A690C0(a1, a2, a3, a4);
  if ( v4 != 1 )
    BUG();
  return sub_A69280(a1, a2, a3, a4);
}
