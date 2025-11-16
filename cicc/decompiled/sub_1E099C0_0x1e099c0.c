// Function: sub_1E099C0
// Address: 0x1e099c0
//
__int64 __fastcall sub_1E099C0(unsigned int *a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v4; // rax

  v2 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v2 <= 0xBu )
  {
    v4 = sub_16E7EE0(a2, "%jump-table.", 0xCu);
    return sub_16E7A90(v4, *a1);
  }
  else
  {
    qmemcpy(v2, "%jump-table.", 12);
    *(_QWORD *)(a2 + 24) += 12LL;
    return sub_16E7A90(a2, *a1);
  }
}
