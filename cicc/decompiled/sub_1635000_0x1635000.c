// Function: sub_1635000
// Address: 0x1635000
//
void *__fastcall sub_1635000(__int64 a1)
{
  bool v2; // zf

  *(_DWORD *)(a1 + 12) = 0;
  v2 = dword_4F9EE20 == 0x7FFFFFFF;
  *(_QWORD *)a1 = &unk_49EDE30;
  *(_BYTE *)(a1 + 8) = !v2;
  return &unk_49EDE30;
}
