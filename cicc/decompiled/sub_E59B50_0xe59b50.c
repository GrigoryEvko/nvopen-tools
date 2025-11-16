// Function: sub_E59B50
// Address: 0xe59b50
//
_BYTE *__fastcall sub_E59B50(__int64 a1, signed __int64 a2, signed __int64 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  __int64 v7; // rdi
  _WORD *v8; // rdx

  sub_E9D130();
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xDu )
  {
    sub_CB6200(v5, "\t.cfi_def_cfa ", 0xEu);
  }
  else
  {
    qmemcpy(v6, "\t.cfi_def_cfa ", 14);
    *(_QWORD *)(v5 + 32) += 14LL;
  }
  sub_E4C9A0(a1, a2);
  v7 = *(_QWORD *)(a1 + 304);
  v8 = *(_WORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    v7 = sub_CB6200(v7, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  sub_CB59F0(v7, a3);
  return sub_E4D880(a1);
}
