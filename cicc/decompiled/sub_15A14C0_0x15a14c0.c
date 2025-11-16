// Function: sub_15A14C0
// Address: 0x15a14c0
//
__int64 __fastcall sub_15A14C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al

  v4 = *(_BYTE *)(a1 + 8);
  if ( v4 == 16 )
    v4 = *(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL);
  if ( (unsigned __int8)(v4 - 1) > 5u )
    return sub_15A06D0((__int64 **)a1, a2, a3, a4);
  else
    return sub_15A1390(a1, a2, a3, a4);
}
