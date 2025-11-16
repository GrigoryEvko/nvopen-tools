// Function: sub_ADE610
// Address: 0xade610
//
__int64 __fastcall sub_ADE610(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v9; // rax

  v6 = sub_B12000(a2 + 72);
  sub_ADDDC0(a1, v6);
  v7 = sub_B11F60(a2 + 80);
  sub_ADDDC0(a1, v7);
  if ( *(_BYTE *)(a2 + 64) == 2 )
  {
    v9 = sub_B11F60(a2 + 88);
    sub_ADDDC0(a1, v9);
  }
  return sub_AA8770(*(_QWORD *)(a3 + 16), a2, a3, a4);
}
