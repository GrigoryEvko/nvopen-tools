// Function: sub_7AEC30
// Address: 0x7aec30
//
_QWORD *sub_7AEC30()
{
  __int64 v0; // rbx
  const __m128i *v1; // rdi
  __int64 v2; // rdx
  int v3; // eax
  _QWORD *v4; // rax

  v0 = (__int64)qword_4F061C0;
  v1 = (const __m128i *)(qword_4F061C0 + 3);
  qword_4F061C0 = (_QWORD *)*qword_4F061C0;
  v2 = *(_QWORD *)(v0 + 16);
  *(_QWORD *)v0 = qword_4F08528;
  *(_QWORD *)dword_4F07508 = v2;
  v3 = *(_BYTE *)(v0 + 56) & 1;
  unk_4F04D80 = (*(_BYTE *)(v0 + 56) & 2) != 0;
  unk_4F04D84 = v3;
  sub_7AEA70(v1);
  qword_4F08528 = v0;
  v4 = (_QWORD *)qword_4F061C8;
  qword_4F061C8 = *(_QWORD *)qword_4F061C8;
  *v4 = qword_4F08530;
  qword_4F08530 = (__int64)v4;
  qword_4F061D0 = *(_QWORD *)&dword_4F077C8;
  return &qword_4F061D0;
}
