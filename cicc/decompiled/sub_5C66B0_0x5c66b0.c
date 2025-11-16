// Function: sub_5C66B0
// Address: 0x5c66b0
//
__int64 __fastcall sub_5C66B0(__int64 a1, __int64 a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a2 + 169);
  if ( (v2 & 4) != 0 )
    sub_6851C0(2963, a1 + 56);
  else
    *(_BYTE *)(a2 + 169) = v2 | 2;
  return a2;
}
