// Function: sub_1480620
// Address: 0x1480620
//
__int64 __fastcall sub_1480620(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  if ( *(_WORD *)(a2 + 24) )
  {
    v6 = sub_1456040(a2);
    v7 = sub_1456E10(a1, v6);
    v8 = sub_15A04A0(v7);
    v9 = sub_145CE20(a1, v8);
    return sub_13A5B60(a1, a2, v9, a3, 0);
  }
  else
  {
    v3 = sub_15A2B90(*(_QWORD *)(a2 + 32), 0, 0);
    return sub_145CE20(a1, v3);
  }
}
