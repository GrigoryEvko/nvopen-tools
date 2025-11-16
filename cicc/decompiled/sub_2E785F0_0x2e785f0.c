// Function: sub_2E785F0
// Address: 0x2e785f0
//
__int64 __fastcall sub_2E785F0(unsigned int *a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v4; // rax

  v2 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0xBu )
  {
    v4 = sub_CB6200(a2, "%jump-table.", 0xCu);
    return sub_CB59D0(v4, *a1);
  }
  else
  {
    qmemcpy(v2, "%jump-table.", 12);
    *(_QWORD *)(a2 + 32) += 12LL;
    return sub_CB59D0(a2, *a1);
  }
}
