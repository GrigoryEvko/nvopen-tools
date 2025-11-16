// Function: sub_B13CF0
// Address: 0xb13cf0
//
__int64 __fastcall sub_B13CF0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al

  v3 = *(_BYTE *)(a1 + 32);
  if ( !v3 )
    return sub_B13880(a1, a2, a3);
  if ( v3 != 1 )
    BUG();
  return sub_B13150(a1, a2, a3);
}
