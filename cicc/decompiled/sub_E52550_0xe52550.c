// Function: sub_E52550
// Address: 0xe52550
//
_BYTE *__fastcall sub_E52550(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, unsigned __int8 a5)
{
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rdi
  _WORD *v12; // rdx
  __int64 v13; // rax

  *a3 = a2 + 56;
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_QWORD *)(v9 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v9 + 24) - v10) <= 5 )
  {
    sub_CB6200(v9, ".tbss ", 6u);
  }
  else
  {
    *(_DWORD *)v10 = 1935832110;
    *(_WORD *)(v10 + 4) = 8307;
    *(_QWORD *)(v9 + 32) += 6LL;
  }
  sub_EA12C0(a3, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v11 = *(_QWORD *)(a1 + 304);
  v12 = *(_WORD **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 1u )
  {
    v11 = sub_CB6200(v11, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v12 = 8236;
    *(_QWORD *)(v11 + 32) += 2LL;
  }
  sub_CB59D0(v11, a4);
  if ( (unsigned __int64)(1LL << a5) > 1 )
  {
    v13 = sub_904010(*(_QWORD *)(a1 + 304), ", ");
    sub_CB59D0(v13, a5);
  }
  return sub_E4D880(a1);
}
