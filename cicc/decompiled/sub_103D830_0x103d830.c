// Function: sub_103D830
// Address: 0x103d830
//
_BYTE *__fastcall sub_103D830(__int64 a1, __int64 a2)
{
  char v2; // al

  v2 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 27 )
    return (_BYTE *)sub_103D1E0(a1, a2);
  if ( v2 == 28 )
    return sub_103D3D0(a1, a2);
  if ( v2 != 26 )
    BUG();
  return sub_103D700((_BYTE *)a1, a2);
}
