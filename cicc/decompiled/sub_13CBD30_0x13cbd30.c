// Function: sub_13CBD30
// Address: 0x13cbd30
//
__int64 __fastcall sub_13CBD30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // al

  v5 = *(_BYTE *)(a1 + 16);
  if ( v5 > 0x17u )
  {
    LOBYTE(a5) = v5 == 69;
    return a5;
  }
  a5 = 0;
  if ( v5 != 5 )
    return a5;
  LOBYTE(a5) = *(_WORD *)(a1 + 18) == 45;
  return a5;
}
