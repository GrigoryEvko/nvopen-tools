// Function: sub_19E4B80
// Address: 0x19e4b80
//
__int64 __fastcall sub_19E4B80(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rdx
  __int64 v5; // rax

  if ( a3 )
    sub_1263B40(a2, "ExpressionTypePhi, ");
  sub_1930810(a1, a2, 0);
  v3 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v3) <= 4 )
  {
    v5 = sub_16E7EE0(a2, "bb = ", 5u);
    return sub_16E7B40(v5, *(_QWORD *)(a1 + 48));
  }
  else
  {
    *(_DWORD *)v3 = 1025532514;
    *(_BYTE *)(v3 + 4) = 32;
    *(_QWORD *)(a2 + 24) += 5LL;
    return sub_16E7B40(a2, *(_QWORD *)(a1 + 48));
  }
}
