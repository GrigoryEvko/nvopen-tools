// Function: ctor_474
// Address: 0x54db70
//
int ctor_474()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5003A80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5003AFC = 1;
  qword_5003AD0 = 0x100000000LL;
  dword_5003A8C &= 0x8000u;
  qword_5003A98 = 0;
  qword_5003AA0 = 0;
  qword_5003AA8 = 0;
  dword_5003A88 = v0;
  word_5003A90 = 0;
  qword_5003AB0 = 0;
  qword_5003AB8 = 0;
  qword_5003AC0 = 0;
  qword_5003AC8 = (__int64)&unk_5003AD8;
  qword_5003AE0 = 0;
  qword_5003AE8 = (__int64)&unk_5003B00;
  qword_5003AF0 = 1;
  dword_5003AF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5003AD0;
  v3 = (unsigned int)qword_5003AD0 + 1LL;
  if ( v3 > HIDWORD(qword_5003AD0) )
  {
    sub_C8D5F0((char *)&unk_5003AD8 - 16, &unk_5003AD8, v3, 8);
    v2 = (unsigned int)qword_5003AD0;
  }
  *(_QWORD *)(qword_5003AC8 + 8 * v2) = v1;
  qword_5003B10 = (__int64)&unk_49D9728;
  LODWORD(qword_5003AD0) = qword_5003AD0 + 1;
  qword_5003B08 = 0;
  qword_5003A80 = (__int64)&unk_49DBF10;
  qword_5003B20 = (__int64)&unk_49DC290;
  qword_5003B18 = 0;
  qword_5003B40 = (__int64)nullsub_24;
  qword_5003B38 = (__int64)sub_984050;
  sub_C53080(&qword_5003A80, "likely-branch-weight", 20);
  LODWORD(qword_5003B08) = 2000;
  BYTE4(qword_5003B18) = 1;
  LODWORD(qword_5003B18) = 2000;
  qword_5003AB0 = 56;
  LOBYTE(dword_5003A8C) = dword_5003A8C & 0x9F | 0x20;
  qword_5003AA8 = (__int64)"Weight of the branch likely to be taken (default = 2000)";
  sub_C53130(&qword_5003A80);
  __cxa_atexit(sub_984970, &qword_5003A80, &qword_4A427C0);
  qword_50039A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50039AC &= 0x8000u;
  word_50039B0 = 0;
  qword_50039F0 = 0x100000000LL;
  qword_50039B8 = 0;
  qword_50039C0 = 0;
  qword_50039C8 = 0;
  dword_50039A8 = v4;
  qword_50039D0 = 0;
  qword_50039D8 = 0;
  qword_50039E0 = 0;
  qword_50039E8 = (__int64)&unk_50039F8;
  qword_5003A00 = 0;
  qword_5003A08 = (__int64)&unk_5003A20;
  qword_5003A10 = 1;
  dword_5003A18 = 0;
  byte_5003A1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50039F0;
  v7 = (unsigned int)qword_50039F0 + 1LL;
  if ( v7 > HIDWORD(qword_50039F0) )
  {
    sub_C8D5F0((char *)&unk_50039F8 - 16, &unk_50039F8, v7, 8);
    v6 = (unsigned int)qword_50039F0;
  }
  *(_QWORD *)(qword_50039E8 + 8 * v6) = v5;
  qword_5003A30 = (__int64)&unk_49D9728;
  LODWORD(qword_50039F0) = qword_50039F0 + 1;
  qword_5003A28 = 0;
  qword_50039A0 = (__int64)&unk_49DBF10;
  qword_5003A40 = (__int64)&unk_49DC290;
  qword_5003A38 = 0;
  qword_5003A60 = (__int64)nullsub_24;
  qword_5003A58 = (__int64)sub_984050;
  sub_C53080(&qword_50039A0, "unlikely-branch-weight", 22);
  LODWORD(qword_5003A28) = 1;
  BYTE4(qword_5003A38) = 1;
  LODWORD(qword_5003A38) = 1;
  qword_50039D0 = 55;
  LOBYTE(dword_50039AC) = dword_50039AC & 0x9F | 0x20;
  qword_50039C8 = (__int64)"Weight of the branch unlikely to be taken (default = 1)";
  sub_C53130(&qword_50039A0);
  return __cxa_atexit(sub_984970, &qword_50039A0, &qword_4A427C0);
}
