// Function: sub_D9AB30
// Address: 0xd9ab30
//
void *__fastcall sub_D9AB30(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  sub_D9AB00(a1, a2, a3, 1);
  *(_DWORD *)(a1 + 36) = a4;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 48) = a6;
  *(_QWORD *)a1 = &unk_49DE9E0;
  return &unk_49DE9E0;
}
