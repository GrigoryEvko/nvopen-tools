// Function: sub_1480810
// Address: 0x1480810
//
__int64 __fastcall sub_1480810(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  if ( *(_WORD *)(a2 + 24) )
  {
    v4 = sub_1456040(a2);
    v5 = sub_1456E10(a1, v4);
    v6 = sub_15A04A0(v5);
    v7 = sub_145CE20(a1, v6);
    return sub_14806B0(a1, v7, a2, 0, 0);
  }
  else
  {
    v2 = sub_15A2B00(*(_QWORD *)(a2 + 32));
    return sub_145CE20(a1, v2);
  }
}
