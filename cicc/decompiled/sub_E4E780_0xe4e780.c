// Function: sub_E4E780
// Address: 0xe4e780
//
_BYTE *__fastcall sub_E4E780(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  __int64 v7; // rdi
  __int64 v9; // rdx
  __int64 v10; // rdi
  _WORD *v11; // rdx
  __int64 v12; // rax
  _WORD *v13; // rdx
  __int64 v14; // rdi

  v7 = *(_QWORD *)(a1 + 304);
  v9 = *(_QWORD *)(v7 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v7 + 24) - v9) <= 8 )
  {
    sub_CB6200(v7, "\t.except\t", 9u);
  }
  else
  {
    *(_BYTE *)(v9 + 8) = 9;
    *(_QWORD *)v9 = 0x7470656378652E09LL;
    *(_QWORD *)(v7 + 32) += 9LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v10 = *(_QWORD *)(a1 + 304);
  v11 = *(_WORD **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 1u )
  {
    v10 = sub_CB6200(v10, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v11 = 8236;
    *(_QWORD *)(v10 + 32) += 2LL;
  }
  v12 = sub_CB59D0(v10, a4);
  v13 = *(_WORD **)(v12 + 32);
  v14 = v12;
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 1u )
  {
    v14 = sub_CB6200(v12, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v13 = 8236;
    *(_QWORD *)(v12 + 32) += 2LL;
  }
  sub_CB59D0(v14, a5);
  return sub_E4D880(a1);
}
