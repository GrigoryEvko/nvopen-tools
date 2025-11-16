// Function: sub_B52260
// Address: 0xb52260
//
__int64 __fastcall sub_B52260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  char v5; // al

  v5 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL);
  if ( v5 == 14 )
  {
    if ( *(_BYTE *)(a2 + 8) != 12 )
      return sub_B51D30(49, a1, a2, a3, a4, a5);
    return sub_B51D30(47, a1, a2, a3, a4, a5);
  }
  else
  {
    if ( v5 != 12 || *(_BYTE *)(a2 + 8) != 14 )
      return sub_B51D30(49, a1, a2, a3, a4, a5);
    return sub_B51D30(48, a1, a2, a3, a4, a5);
  }
}
