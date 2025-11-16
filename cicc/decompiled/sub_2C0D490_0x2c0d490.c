// Function: sub_2C0D490
// Address: 0x2c0d490
//
__int64 __fastcall sub_2C0D490(char a1, char a2, int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v9; // rax

  v6 = sub_AA4E30(*(_QWORD *)(a4 + 48));
  if ( !a1 )
    return sub_BCB2D0(*(_QWORD **)(a4 + 72));
  v7 = v6;
  if ( !a3 && !a2 )
    return sub_BCB2D0(*(_QWORD **)(a4 + 72));
  v9 = sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  return sub_AE4570(v7, v9);
}
