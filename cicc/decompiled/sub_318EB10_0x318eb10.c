// Function: sub_318EB10
// Address: 0x318eb10
//
void *__fastcall sub_318EB10(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  *(_DWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)a1 = &unk_4A34780;
  return &unk_4A34780;
}
