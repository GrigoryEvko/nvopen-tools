// Function: sub_28CA310
// Address: 0x28ca310
//
__int64 __fastcall sub_28CA310(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rdx
  __int64 v5; // rax

  if ( a3 )
    sub_904010(a2, "ExpressionTypePhi, ");
  sub_27AFB90(a1, a2, 0);
  v3 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v3) <= 4 )
  {
    v5 = sub_CB6200(a2, "bb = ", 5u);
    return sub_CB5A80(v5, *(_QWORD *)(a1 + 48));
  }
  else
  {
    *(_DWORD *)v3 = 1025532514;
    *(_BYTE *)(v3 + 4) = 32;
    *(_QWORD *)(a2 + 32) += 5LL;
    return sub_CB5A80(a2, *(_QWORD *)(a1 + 48));
  }
}
