// Function: sub_E594F0
// Address: 0xe594f0
//
_BYTE *__fastcall sub_E594F0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  __int64 v7; // rax
  _WORD *v8; // rdx

  sub_E99410();
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xAu )
  {
    v5 = sub_CB6200(v5, "\t.cfi_lsda ", 0xBu);
  }
  else
  {
    qmemcpy(v6, "\t.cfi_lsda ", 11);
    *(_QWORD *)(v5 + 32) += 11LL;
  }
  v7 = sub_CB59D0(v5, a3);
  v8 = *(_WORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(v7, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  return sub_E4D880(a1);
}
