// Function: sub_15FDFF0
// Address: 0x15fdff0
//
__int64 __fastcall sub_15FDFF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al

  v4 = *(_BYTE *)(a2 + 8);
  if ( v4 == 16 )
    v4 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( v4 == 11 )
    return sub_15FDBD0(45, a1, a2, a3, a4);
  else
    return sub_15FDF90(a1, a2, a3, a4);
}
