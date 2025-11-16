// Function: sub_E4D9F0
// Address: 0xe4d9f0
//
_BYTE *__fastcall sub_E4D9F0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v6; // rdi
  void *v8; // rdx
  __int64 v9; // rdi
  _WORD *v10; // rdx
  __int64 v11; // rdi
  _WORD *v12; // rdx

  v6 = *(_QWORD *)(a1 + 304);
  v8 = *(void **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v8 <= 0xCu )
  {
    sub_CB6200(v6, "\t.cg_profile ", 0xDu);
  }
  else
  {
    qmemcpy(v8, "\t.cg_profile ", 13);
    *(_QWORD *)(v6 + 32) += 13LL;
  }
  sub_EA12C0(*(_QWORD *)(a2 + 16), *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_WORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
  {
    sub_CB6200(v9, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(v9 + 32) += 2LL;
  }
  sub_EA12C0(*(_QWORD *)(a3 + 16), *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
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
  return sub_E4D880(a1);
}
