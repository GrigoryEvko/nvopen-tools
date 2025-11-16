// Function: sub_2F3F220
// Address: 0x2f3f220
//
__int64 __fastcall sub_2F3F220(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v4; // rax

  v2 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 9u )
  {
    v4 = sub_CB6200(a2, "FixedStack", 0xAu);
    return sub_CB59F0(v4, *(int *)(a1 + 16));
  }
  else
  {
    qmemcpy(v2, "FixedStack", 10);
    *(_QWORD *)(a2 + 32) += 10LL;
    return sub_CB59F0(a2, *(int *)(a1 + 16));
  }
}
