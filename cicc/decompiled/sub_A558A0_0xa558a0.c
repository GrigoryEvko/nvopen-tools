// Function: sub_A558A0
// Address: 0xa558a0
//
void *__fastcall sub_A558A0(__int64 a1, __int64 a2, char a3)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 17) = a3;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)a1 = &unk_49D9A00;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 16) = a2 != 0;
  return &unk_49D9A00;
}
