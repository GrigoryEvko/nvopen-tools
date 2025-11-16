// Function: sub_164EDD0
// Address: 0x164edd0
//
__int64 __fastcall sub_164EDD0(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // rax
  _WORD *v4; // rdx
  __int64 v5; // rdi

  v2 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v2 <= 0xDu )
  {
    a1 = sub_16E7EE0(a1, "; ModuleID = '", 14);
  }
  else
  {
    qmemcpy(v2, "; ModuleID = '", 14);
    *(_QWORD *)(a1 + 24) += 14LL;
  }
  v3 = sub_16E7EE0(a1, *(const char **)(a2 + 176), *(_QWORD *)(a2 + 184));
  v4 = *(_WORD **)(v3 + 24);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 1u )
    return sub_16E7EE0(v3, "'\n", 2);
  *v4 = 2599;
  *(_QWORD *)(v5 + 24) += 2LL;
  return 2599;
}
