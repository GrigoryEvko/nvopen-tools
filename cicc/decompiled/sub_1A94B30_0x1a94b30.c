// Function: sub_1A94B30
// Address: 0x1a94b30
//
_BOOL8 __fastcall sub_1A94B30(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u || (v1 & 0xFD) != 0x4D && (unsigned __int8)(v1 - 83) > 2u )
    return 1;
  if ( *(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0 )
    return sub_1625940(a1, "is_base_value", 0xDu) != 0;
  return 0;
}
