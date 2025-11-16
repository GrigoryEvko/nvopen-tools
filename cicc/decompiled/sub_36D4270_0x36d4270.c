// Function: sub_36D4270
// Address: 0x36d4270
//
__int64 __fastcall sub_36D4270(__int64 a1, _QWORD *a2)
{
  int v2; // ebx

  v2 = sub_36D4010(a2, (__int64)"llvm.global_ctors", 0x11u, 1);
  return v2 | (unsigned int)sub_36D4010(a2, (__int64)"llvm.global_dtors", 0x11u, 0);
}
