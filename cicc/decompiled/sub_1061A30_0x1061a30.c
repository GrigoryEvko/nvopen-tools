// Function: sub_1061A30
// Address: 0x1061a30
//
void *__fastcall sub_1061A30(__int64 a1, char a2, __int64 a3)
{
  *(_DWORD *)(a1 + 8) = 6;
  *(_BYTE *)(a1 + 12) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)a1 = &unk_49E5EA8;
  return &unk_49E5EA8;
}
