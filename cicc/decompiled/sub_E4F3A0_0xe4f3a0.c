// Function: sub_E4F3A0
// Address: 0xe4f3a0
//
_BYTE *__fastcall sub_E4F3A0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  void *v6; // rdx
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  _WORD *v11; // rdx
  __int64 v12; // rdi

  sub_E4CF20(a1, a2, a3);
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xAu )
  {
    sub_CB6200(v5, ", reg_rel, ", 0xBu);
  }
  else
  {
    qmemcpy(v6, ", reg_rel, ", 11);
    *(_QWORD *)(v5 + 32) += 11LL;
  }
  v7 = sub_CB59F0(*(_QWORD *)(a1 + 304), (unsigned __int16)a4);
  v8 = *(_WORD **)(v7 + 32);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    v9 = sub_CB6200(v7, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  v10 = sub_CB59F0(v9, WORD1(a4));
  v11 = *(_WORD **)(v10 + 32);
  v12 = v10;
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 1u )
  {
    v12 = sub_CB6200(v10, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v11 = 8236;
    *(_QWORD *)(v10 + 32) += 2LL;
  }
  sub_CB59F0(v12, SHIDWORD(a4));
  return sub_E4D880(a1);
}
