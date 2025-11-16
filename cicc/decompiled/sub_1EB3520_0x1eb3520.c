// Function: sub_1EB3520
// Address: 0x1eb3520
//
__int64 __fastcall sub_1EB3520(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v4; // rax

  v2 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v2 <= 9u )
  {
    v4 = sub_16E7EE0(a2, "FixedStack", 0xAu);
    return sub_16E7AB0(v4, *(int *)(a1 + 16));
  }
  else
  {
    qmemcpy(v2, "FixedStack", 10);
    *(_QWORD *)(a2 + 24) += 10LL;
    return sub_16E7AB0(a2, *(int *)(a1 + 16));
  }
}
