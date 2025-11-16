// Function: sub_B43C40
// Address: 0xb43c40
//
__int64 __fastcall sub_B43C40(__int64 a1)
{
  __int64 v2; // rax

  if ( (*(_BYTE *)(a1 + 7) & 8) != 0 )
  {
    v2 = sub_ACADE0(*(__int64 ***)(a1 + 8));
    sub_BA6240(a1, v2);
  }
  sub_B99FD0(a1, 38, 0);
  if ( *(_QWORD *)(a1 + 48) )
    sub_B91220(a1 + 48);
  return sub_BD7260(a1);
}
