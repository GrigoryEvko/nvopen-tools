// Function: sub_D9AB00
// Address: 0xd9ab00
//
void *__fastcall sub_D9AB00(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)a1 = &unk_49DE890;
  *(_DWORD *)(a1 + 32) = a4;
  return &unk_49DE890;
}
