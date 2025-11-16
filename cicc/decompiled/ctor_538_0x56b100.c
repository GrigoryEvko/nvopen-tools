// Function: ctor_538
// Address: 0x56b100
//
int ctor_538()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5016020 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5016038 = 0;
  qword_5016040 = 0;
  qword_5016048 = 0;
  qword_5016050 = 0;
  dword_501602C = dword_501602C & 0x8000 | 1;
  qword_5016070 = 0x100000000LL;
  dword_5016028 = v0;
  word_5016030 = 0;
  qword_5016058 = 0;
  qword_5016060 = 0;
  qword_5016068 = (__int64)&unk_5016078;
  qword_5016080 = 0;
  qword_5016088 = (__int64)&unk_50160A0;
  qword_5016090 = 1;
  dword_5016098 = 0;
  byte_501609C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5016070;
  v3 = (unsigned int)qword_5016070 + 1LL;
  if ( v3 > HIDWORD(qword_5016070) )
  {
    sub_C8D5F0((char *)&unk_5016078 - 16, &unk_5016078, v3, 8);
    v2 = (unsigned int)qword_5016070;
  }
  *(_QWORD *)(qword_5016068 + 8 * v2) = v1;
  LODWORD(qword_5016070) = qword_5016070 + 1;
  qword_50160A8 = 0;
  qword_5016020 = (__int64)&unk_49DAD08;
  qword_50160B0 = 0;
  qword_50160B8 = 0;
  qword_50160F8 = (__int64)&unk_49DC350;
  qword_50160C0 = 0;
  qword_5016118 = (__int64)nullsub_81;
  qword_50160C8 = 0;
  qword_5016110 = (__int64)sub_BB8600;
  qword_50160D0 = 0;
  byte_50160D8 = 0;
  qword_50160E0 = 0;
  qword_50160E8 = 0;
  qword_50160F0 = 0;
  sub_C53080(&qword_5016020, "select-kernel-list", 18);
  BYTE1(dword_501602C) |= 2u;
  qword_5016058 = (__int64)"list";
  qword_5016048 = (__int64)"A list of kernel to optimize";
  qword_5016060 = 4;
  qword_5016050 = 28;
  sub_C53130(&qword_5016020);
  __cxa_atexit(sub_BB89D0, &qword_5016020, &qword_4A427C0);
  qword_5015F20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5015F38 = 0;
  qword_5015F40 = 0;
  qword_5015F48 = 0;
  qword_5015F50 = 0;
  dword_5015F2C = dword_5015F2C & 0x8000 | 1;
  word_5015F30 = 0;
  qword_5015F70 = 0x100000000LL;
  dword_5015F28 = v4;
  qword_5015F58 = 0;
  qword_5015F60 = 0;
  qword_5015F68 = (__int64)&unk_5015F78;
  qword_5015F80 = 0;
  qword_5015F88 = (__int64)&unk_5015FA0;
  qword_5015F90 = 1;
  dword_5015F98 = 0;
  byte_5015F9C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5015F70;
  v7 = (unsigned int)qword_5015F70 + 1LL;
  if ( v7 > HIDWORD(qword_5015F70) )
  {
    sub_C8D5F0((char *)&unk_5015F78 - 16, &unk_5015F78, v7, 8);
    v6 = (unsigned int)qword_5015F70;
  }
  *(_QWORD *)(qword_5015F68 + 8 * v6) = v5;
  LODWORD(qword_5015F70) = qword_5015F70 + 1;
  qword_5015FA8 = 0;
  qword_5015F20 = (__int64)&unk_4A25DD0;
  qword_5015FB0 = 0;
  qword_5015FB8 = 0;
  qword_5015FF8 = (__int64)&unk_49DC290;
  qword_5015FC0 = 0;
  qword_5016018 = (__int64)nullsub_1574;
  qword_5015FC8 = 0;
  qword_5016010 = (__int64)sub_2D177C0;
  qword_5015FD0 = 0;
  byte_5015FD8 = 0;
  qword_5015FE0 = 0;
  qword_5015FE8 = 0;
  qword_5015FF0 = 0;
  sub_C53080(&qword_5015F20, "select-kernel-range", 19);
  BYTE1(dword_5015F2C) |= 2u;
  qword_5015F58 = (__int64)"list";
  qword_5015F60 = 4;
  qword_5015F48 = (__int64)"A set of kernels to optimize";
  qword_5015F50 = 28;
  sub_C53130(&qword_5015F20);
  return __cxa_atexit(sub_2D178F0, &qword_5015F20, &qword_4A427C0);
}
