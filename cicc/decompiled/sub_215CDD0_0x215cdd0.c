// Function: sub_215CDD0
// Address: 0x215cdd0
//
void *__fastcall sub_215CDD0(__int64 a1, char a2)
{
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 1;
  *(_BYTE *)(a1 + 24) = 1;
  *(_DWORD *)(a1 + 12) = a2 == 0 ? 16 : 128;
  *(_QWORD *)a1 = &unk_4A01970;
  return &unk_4A01970;
}
