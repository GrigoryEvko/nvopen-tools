// Function: sub_15E5B20
// Address: 0x15e5b20
//
__int64 __fastcall sub_15E5B20(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 2 )
    return sub_15E5AB0((_QWORD *)a1);
  if ( v1 > 2u )
    return sub_15E55B0(a1);
  if ( v1 )
    return sub_15E58C0((_QWORD *)a1);
  return sub_15E3D00(a1);
}
