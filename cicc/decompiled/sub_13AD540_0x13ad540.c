// Function: sub_13AD540
// Address: 0x13ad540
//
__int64 __fastcall sub_13AD540(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 v7; // rax

  v4 = (_QWORD *)a2;
  if ( *(_WORD *)(a2 + 24) == 7 )
  {
    while ( a3 != v4[6] )
    {
      v5 = v4[4];
      v4 = *(_QWORD **)v5;
      if ( *(_WORD *)(*(_QWORD *)v5 + 24LL) != 7 )
        goto LABEL_4;
    }
    return sub_13A5BC0(v4, *(_QWORD *)(a1 + 8));
  }
  else
  {
LABEL_4:
    v6 = *(_QWORD *)(a1 + 8);
    v7 = sub_1456040(v4);
    return sub_145CF80(v6, v7, 0, 0);
  }
}
