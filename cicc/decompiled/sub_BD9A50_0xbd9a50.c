// Function: sub_BD9A50
// Address: 0xbd9a50
//
__int64 __fastcall sub_BD9A50(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // rax
  _WORD *v4; // rdx
  __int64 v5; // rdi

  v2 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 0xDu )
  {
    a1 = sub_CB6200(a1, "; ModuleID = '", 14);
  }
  else
  {
    qmemcpy(v2, "; ModuleID = '", 14);
    *(_QWORD *)(a1 + 32) += 14LL;
  }
  v3 = sub_CB6200(a1, *(_QWORD *)(a2 + 168), *(_QWORD *)(a2 + 176));
  v4 = *(_WORD **)(v3 + 32);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 1u )
    return sub_CB6200(v3, "'\n", 2);
  *v4 = 2599;
  *(_QWORD *)(v5 + 32) += 2LL;
  return 2599;
}
