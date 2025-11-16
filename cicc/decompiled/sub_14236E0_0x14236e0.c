// Function: sub_14236E0
// Address: 0x14236e0
//
_BYTE *__fastcall sub_14236E0(__int64 a1, __int64 a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 22 )
    return (_BYTE *)sub_1422EF0(a1, a2);
  if ( v2 == 23 )
    return sub_14231B0(a1, a2);
  return sub_1423540(a1, a2);
}
