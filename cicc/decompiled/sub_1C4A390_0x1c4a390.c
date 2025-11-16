// Function: sub_1C4A390
// Address: 0x1c4a390
//
__int64 __fastcall sub_1C4A390(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  void *v4; // rdx
  __int64 v5; // rdi

  v3 = sub_16BA580(a1, a2, a3);
  v4 = *(void **)(v3 + 24);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0xCu )
    return sub_16E7EE0(v3, "PHI-operand:\n", 0xDu);
  qmemcpy(v4, "PHI-operand:\n", 13);
  *(_QWORD *)(v5 + 24) += 13LL;
  return 0x7265706F2D494850LL;
}
