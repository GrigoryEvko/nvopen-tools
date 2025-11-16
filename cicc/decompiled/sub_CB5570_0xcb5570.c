// Function: sub_CB5570
// Address: 0xcb5570
//
__int64 __fastcall sub_CB5570(__int64 a1)
{
  char v2; // al

  if ( !*(_BYTE *)(a1 + 56) )
  {
    v2 = sub_C86300(*(unsigned int *)(a1 + 48));
    *(_BYTE *)(a1 + 56) = 1;
    *(_BYTE *)(a1 + 55) = v2;
  }
  return *(unsigned __int8 *)(a1 + 55);
}
