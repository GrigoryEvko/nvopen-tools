// Function: sub_E59EF0
// Address: 0xe59ef0
//
__int64 __fastcall sub_E59EF0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  void *v8; // rdx
  __int64 v9; // rax
  _WORD *v10; // rdx
  __int64 v11; // rdi
  _WORD *v12; // rdx

  v7 = *(_QWORD *)(a1 + 304);
  v8 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 0xEu )
  {
    v7 = sub_CB6200(v7, "\t.cv_linetable\t", 0xFu);
  }
  else
  {
    qmemcpy(v8, "\t.cv_linetable\t", 15);
    *(_QWORD *)(v7 + 32) += 15LL;
  }
  v9 = sub_CB59D0(v7, a2);
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
  sub_EA12C0(a3, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v11 = *(_QWORD *)(a1 + 304);
  v12 = *(_WORD **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 1u )
  {
    sub_CB6200(v11, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v12 = 8236;
    *(_QWORD *)(v11 + 32) += 2LL;
  }
  sub_EA12C0(a4, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  sub_E4D880(a1);
  return nullsub_346(a1, a2, a3, a4);
}
