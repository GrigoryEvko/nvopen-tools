// Function: sub_15FE030
// Address: 0x15fe030
//
__int64 __fastcall sub_15FE030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al

  v4 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v4 == 15 )
  {
    if ( *(_BYTE *)(a2 + 8) != 11 )
      return sub_15FDBD0(47, a1, a2, a3, a4);
    return sub_15FDBD0(45, a1, a2, a3, a4);
  }
  else
  {
    if ( v4 != 11 || *(_BYTE *)(a2 + 8) != 15 )
      return sub_15FDBD0(47, a1, a2, a3, a4);
    return sub_15FDBD0(46, a1, a2, a3, a4);
  }
}
