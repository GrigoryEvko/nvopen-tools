// Function: sub_73E690
// Address: 0x73e690
//
_BYTE *__fastcall sub_73E690(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  unsigned __int8 v3; // al
  _BYTE *v4; // rdi

  *(_QWORD *)(a1 + 16) = a2;
  v2 = *a2;
  v3 = sub_6E9930(56, *a2);
  v4 = sub_73DBF0(v3, v2, a1);
  if ( unk_4D04810 )
    v4[60] |= 2u;
  return sub_732B10((__int64)v4);
}
