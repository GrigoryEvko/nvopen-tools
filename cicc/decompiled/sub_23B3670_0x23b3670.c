// Function: sub_23B3670
// Address: 0x23b3670
//
void *__fastcall sub_23B3670(__int64 a1, char a2)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 32) = 1;
  *(_BYTE *)(a1 + 33) = a2;
  *(_QWORD *)a1 = &unk_4A160B8;
  *(_DWORD *)(a1 + 36) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  return &unk_4A160B8;
}
