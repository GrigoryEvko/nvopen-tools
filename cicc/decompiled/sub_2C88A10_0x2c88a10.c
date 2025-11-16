// Function: sub_2C88A10
// Address: 0x2c88a10
//
__int64 __fastcall sub_2C88A10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  void *v3; // rdx
  __int64 v4; // rdi

  v2 = sub_C5F790(a1, a2);
  v3 = *(void **)(v2 + 32);
  v4 = v2;
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v3 <= 0xCu )
    return sub_CB6200(v2, "PHI-operand:\n", 0xDu);
  qmemcpy(v3, "PHI-operand:\n", 13);
  *(_QWORD *)(v4 + 32) += 13LL;
  return 0x7265706F2D494850LL;
}
