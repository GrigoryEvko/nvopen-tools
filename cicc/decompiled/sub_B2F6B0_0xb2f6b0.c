// Function: sub_B2F6B0
// Address: 0xb2f6b0
//
__int64 __fastcall sub_B2F6B0(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 + 32) & 0xF;
  if ( v1 > 0xAu )
    BUG();
  return ((__int64 (*)(void))((char *)funcs_B2F6D0 + (int)funcs_B2F6D0[v1]))();
}
