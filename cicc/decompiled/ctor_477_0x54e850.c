// Function: ctor_477
// Address: 0x54e850
//
int ctor_477()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  dword_5004668 = sub_246D160("newgvn-vn", 9, "Controls which instructions are value numbered", 46);
  sub_246D160("newgvn-phi", 10, "Controls which instructions we create phi of ops for", 52);
  qword_50045A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50045F0 = 0x100000000LL;
  dword_50045AC &= 0x8000u;
  word_50045B0 = 0;
  qword_50045B8 = 0;
  qword_50045C0 = 0;
  dword_50045A8 = v0;
  qword_50045C8 = 0;
  qword_50045D0 = 0;
  qword_50045D8 = 0;
  qword_50045E0 = 0;
  qword_50045E8 = (__int64)&unk_50045F8;
  qword_5004600 = 0;
  qword_5004608 = (__int64)&unk_5004620;
  qword_5004610 = 1;
  dword_5004618 = 0;
  byte_500461C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50045F0;
  v3 = (unsigned int)qword_50045F0 + 1LL;
  if ( v3 > HIDWORD(qword_50045F0) )
  {
    sub_C8D5F0((char *)&unk_50045F8 - 16, &unk_50045F8, v3, 8);
    v2 = (unsigned int)qword_50045F0;
  }
  *(_QWORD *)(qword_50045E8 + 8 * v2) = v1;
  qword_5004630 = (__int64)&unk_49D9748;
  LODWORD(qword_50045F0) = qword_50045F0 + 1;
  qword_5004628 = 0;
  qword_50045A0 = (__int64)&unk_49DC090;
  qword_5004640 = (__int64)&unk_49DC1D0;
  qword_5004638 = 0;
  qword_5004660 = (__int64)nullsub_23;
  qword_5004658 = (__int64)sub_984030;
  sub_C53080(&qword_50045A0, "enable-store-refinement", 23);
  LOWORD(qword_5004638) = 256;
  LOBYTE(qword_5004628) = 0;
  LOBYTE(dword_50045AC) = dword_50045AC & 0x9F | 0x20;
  sub_C53130(&qword_50045A0);
  __cxa_atexit(sub_984900, &qword_50045A0, &qword_4A427C0);
  qword_50044C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5004510 = 0x100000000LL;
  word_50044D0 = 0;
  dword_50044CC &= 0x8000u;
  qword_50044D8 = 0;
  qword_50044E0 = 0;
  dword_50044C8 = v4;
  qword_50044E8 = 0;
  qword_50044F0 = 0;
  qword_50044F8 = 0;
  qword_5004500 = 0;
  qword_5004508 = (__int64)&unk_5004518;
  qword_5004520 = 0;
  qword_5004528 = (__int64)&unk_5004540;
  qword_5004530 = 1;
  dword_5004538 = 0;
  byte_500453C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5004510;
  v7 = (unsigned int)qword_5004510 + 1LL;
  if ( v7 > HIDWORD(qword_5004510) )
  {
    sub_C8D5F0((char *)&unk_5004518 - 16, &unk_5004518, v7, 8);
    v6 = (unsigned int)qword_5004510;
  }
  *(_QWORD *)(qword_5004508 + 8 * v6) = v5;
  qword_5004550 = (__int64)&unk_49D9748;
  LODWORD(qword_5004510) = qword_5004510 + 1;
  qword_5004548 = 0;
  qword_50044C0 = (__int64)&unk_49DC090;
  qword_5004560 = (__int64)&unk_49DC1D0;
  qword_5004558 = 0;
  qword_5004580 = (__int64)nullsub_23;
  qword_5004578 = (__int64)sub_984030;
  sub_C53080(&qword_50044C0, "enable-phi-of-ops", 17);
  LOBYTE(qword_5004548) = 1;
  LOWORD(qword_5004558) = 257;
  LOBYTE(dword_50044CC) = dword_50044CC & 0x9F | 0x20;
  sub_C53130(&qword_50044C0);
  return __cxa_atexit(sub_984900, &qword_50044C0, &qword_4A427C0);
}
