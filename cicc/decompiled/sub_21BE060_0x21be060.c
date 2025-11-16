// Function: sub_21BE060
// Address: 0x21be060
//
void *__fastcall sub_21BE060(__int64 a1, __int64 a2, int a3)
{
  sub_1D483C0(a1, a2, a3);
  *(_QWORD *)(a1 + 464) = a2;
  *(_BYTE *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)a1 = &unk_4A03528;
  *(_BYTE *)(a1 + 472) = a3 > 0;
  return &unk_4A03528;
}
