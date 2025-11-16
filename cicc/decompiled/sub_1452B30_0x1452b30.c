// Function: sub_1452B30
// Address: 0x1452b30
//
char __fastcall sub_1452B30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  sub_16BDCA0(*(_QWORD *)(a1 + 64) + 816LL, a1 + 32);
  v2 = *(_QWORD *)(a1 + 24);
  if ( a2 != v2 )
  {
    if ( v2 != -8 && v2 != 0 && v2 != -16 )
      sub_1649B30(a1 + 8);
    *(_QWORD *)(a1 + 24) = a2;
    LOBYTE(v2) = a2 != 0;
    if ( a2 != 0 && a2 != -8 && a2 != -16 )
      LOBYTE(v2) = sub_164C220(a1 + 8);
  }
  return v2;
}
