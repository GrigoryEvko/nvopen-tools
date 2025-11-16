// Function: sub_7ABFD0
// Address: 0x7abfd0
//
__int64 sub_7ABFD0()
{
  __int64 v0; // rbx
  __int64 v1; // r15
  _BYTE *v2; // rax
  _BYTE *v3; // r12
  __int64 result; // rax

  v0 = unk_4F06490 - unk_4F06498;
  v1 = 2LL * (unk_4F06490 - unk_4F06498);
  v2 = (_BYTE *)sub_822C60(unk_4F06498 - 1LL, unk_4F06490 - unk_4F06498 + 2LL, v1 + 2);
  *v2 = 32;
  v3 = v2 + 1;
  qword_4F06488 = (_QWORD *)sub_822C60(qword_4F06488, 8 * v0, 16 * v0);
  result = sub_81A600(unk_4F06498, unk_4F06490, v3, 1);
  unk_4F06498 = v3;
  unk_4F06490 = &v3[v1];
  return result;
}
