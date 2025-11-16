// Function: sub_D9AB80
// Address: 0xd9ab80
//
void *__fastcall sub_D9AB80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  sub_D9AB00(a1, a2, a3, 2);
  *(_QWORD *)(a1 + 40) = a4;
  *(_DWORD *)(a1 + 48) = a5;
  *(_QWORD *)a1 = &unk_49DEA10;
  return &unk_49DEA10;
}
