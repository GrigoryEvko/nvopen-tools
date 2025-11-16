// Function: sub_1AD4D60
// Address: 0x1ad4d60
//
unsigned __int64 __fastcall sub_1AD4D60(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
    return 0;
  if ( v1 == 78 )
    return a1 | 4;
  if ( v1 != 29 )
    return 0;
  return a1 & 0xFFFFFFFFFFFFFFFBLL;
}
