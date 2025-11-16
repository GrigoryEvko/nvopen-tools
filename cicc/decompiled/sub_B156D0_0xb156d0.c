// Function: sub_B156D0
// Address: 0xb156d0
//
void *__fastcall sub_B156D0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  *(_DWORD *)(a1 + 8) = 2;
  *(_BYTE *)(a1 + 12) = a4;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)a1 = &unk_49D9B88;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = 0;
  return &unk_49D9B88;
}
