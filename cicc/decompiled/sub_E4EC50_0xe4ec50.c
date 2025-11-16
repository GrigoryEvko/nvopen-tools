// Function: sub_E4EC50
// Address: 0xe4ec50
//
_BYTE *__fastcall sub_E4EC50(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // rax
  __int64 v8; // rdi
  _BYTE *v9; // rax

  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(_QWORD *)(v5 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v6) <= 4 )
  {
    v5 = sub_CB6200(v5, (unsigned __int8 *)".desc", 5u);
    v7 = *(_BYTE **)(v5 + 32);
    if ( *(_QWORD *)(v5 + 24) > (unsigned __int64)v7 )
      goto LABEL_3;
  }
  else
  {
    *(_DWORD *)v6 = 1936024622;
    *(_BYTE *)(v6 + 4) = 99;
    v7 = (_BYTE *)(*(_QWORD *)(v5 + 32) + 5LL);
    *(_QWORD *)(v5 + 32) = v7;
    if ( *(_QWORD *)(v5 + 24) > (unsigned __int64)v7 )
    {
LABEL_3:
      *(_QWORD *)(v5 + 32) = v7 + 1;
      *v7 = 32;
      goto LABEL_4;
    }
  }
  sub_CB5D20(v5, 32);
LABEL_4:
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v8 = *(_QWORD *)(a1 + 304);
  v9 = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
  {
    v8 = sub_CB5D20(v8, 44);
  }
  else
  {
    *(_QWORD *)(v8 + 32) = v9 + 1;
    *v9 = 44;
  }
  sub_CB59D0(v8, a3);
  return sub_E4D880(a1);
}
