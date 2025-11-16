// Function: sub_2DE1890
// Address: 0x2de1890
//
__int64 __fastcall sub_2DE1890(__int64 a1, __int64 a2)
{
  __int64 v3; // rax

  sub_2DDCDD0(a1, a2);
  if ( *(_DWORD *)a1 == 2 )
  {
    v3 = sub_3111D40();
    return sub_2DDE8C0(a1, a2, *(_QWORD *)(v3 + 8));
  }
  else
  {
    sub_2DDC700(a1, a2);
    if ( *(_DWORD *)a1 == 1 )
      sub_2DDCF00(a1, a2);
    sub_311BE60(*(_QWORD *)(a1 + 8), 0);
    return sub_2DDE8C0(a1, a2, *(_QWORD *)(a1 + 8));
  }
}
