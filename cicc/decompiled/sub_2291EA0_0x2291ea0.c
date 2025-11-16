// Function: sub_2291EA0
// Address: 0x2291ea0
//
_QWORD *__fastcall sub_2291EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rax

  v6 = (_QWORD *)a2;
  if ( *(_WORD *)(a2 + 24) == 8 )
  {
    while ( a3 != v6[6] )
    {
      a4 = v6[4];
      v6 = *(_QWORD **)a4;
      if ( *(_WORD *)(*(_QWORD *)a4 + 24LL) != 8 )
        goto LABEL_4;
    }
    return (_QWORD *)sub_D33D80(v6, *(_QWORD *)(a1 + 8), a3, a4, a5);
  }
  else
  {
LABEL_4:
    v7 = *(_QWORD *)(a1 + 8);
    v8 = sub_D95540((__int64)v6);
    return sub_DA2C50(v7, v8, 0, 0);
  }
}
