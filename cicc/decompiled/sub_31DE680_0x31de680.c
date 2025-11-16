// Function: sub_31DE680
// Address: 0x31de680
//
__int64 __fastcall sub_31DE680(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v6; // r14

  v4 = *(_QWORD *)(a1 + 200);
  if ( *(_DWORD *)(v4 + 564) != 3 )
    return sub_23CF390(v4, a2);
  if ( !sub_B326E0((_BYTE *)a2, a2, a3)
    || (v6 = *(_QWORD *)(a2 + 40), !(unsigned int)sub_23CF1A0(*(_QWORD *)(a1 + 200)))
    || (unsigned int)sub_BAA5E0(v6)
    || (*(_BYTE *)(a2 + 33) & 0x40) == 0 )
  {
    v4 = *(_QWORD *)(a1 + 200);
    return sub_23CF390(v4, a2);
  }
  return sub_31DE640(a1, a2, "$local", 6u);
}
