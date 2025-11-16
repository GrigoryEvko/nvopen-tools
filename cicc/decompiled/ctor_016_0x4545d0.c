// Function: ctor_016
// Address: 0x4545d0
//
int ctor_016()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4F80B00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F80B50 = 0x100000000LL;
  word_4F80B10 = 0;
  dword_4F80B0C &= 0x8000u;
  qword_4F80B18 = 0;
  qword_4F80B20 = 0;
  dword_4F80B08 = v0;
  qword_4F80B28 = 0;
  qword_4F80B30 = 0;
  qword_4F80B38 = 0;
  qword_4F80B40 = 0;
  qword_4F80B48 = (__int64)&unk_4F80B58;
  qword_4F80B60 = 0;
  qword_4F80B68 = (__int64)&unk_4F80B80;
  qword_4F80B70 = 1;
  dword_4F80B78 = 0;
  byte_4F80B7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F80B50;
  v3 = (unsigned int)qword_4F80B50 + 1LL;
  if ( v3 > HIDWORD(qword_4F80B50) )
  {
    sub_C8D5F0((char *)&unk_4F80B58 - 16, &unk_4F80B58, v3, 8);
    v2 = (unsigned int)qword_4F80B50;
  }
  *(_QWORD *)(qword_4F80B48 + 8 * v2) = v1;
  qword_4F80B90 = (__int64)&unk_49D9748;
  qword_4F80B00 = (__int64)&unk_49DC090;
  qword_4F80BA0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F80B50) = qword_4F80B50 + 1;
  qword_4F80BC0 = (__int64)nullsub_23;
  qword_4F80B88 = 0;
  qword_4F80BB8 = (__int64)sub_984030;
  qword_4F80B98 = 0;
  sub_C53080(&qword_4F80B00, "print-inst-addrs", 16);
  qword_4F80B30 = 44;
  LOBYTE(dword_4F80B0C) = dword_4F80B0C & 0x9F | 0x20;
  qword_4F80B28 = (__int64)"Print addresses of instructions when dumping";
  sub_C53130(&qword_4F80B00);
  __cxa_atexit(sub_984900, &qword_4F80B00, &qword_4A427C0);
  qword_4F80A20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F80A9C = 1;
  qword_4F80A70 = 0x100000000LL;
  dword_4F80A2C &= 0x8000u;
  qword_4F80A68 = (__int64)&unk_4F80A78;
  qword_4F80A38 = 0;
  qword_4F80A40 = 0;
  dword_4F80A28 = v4;
  word_4F80A30 = 0;
  qword_4F80A48 = 0;
  qword_4F80A50 = 0;
  qword_4F80A58 = 0;
  qword_4F80A60 = 0;
  qword_4F80A80 = 0;
  qword_4F80A88 = (__int64)&unk_4F80AA0;
  qword_4F80A90 = 1;
  dword_4F80A98 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F80A70;
  if ( (unsigned __int64)(unsigned int)qword_4F80A70 + 1 > HIDWORD(qword_4F80A70) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4F80A78 - 16, &unk_4F80A78, (unsigned int)qword_4F80A70 + 1LL, 8);
    v6 = (unsigned int)qword_4F80A70;
    v5 = v12;
  }
  *(_QWORD *)(qword_4F80A68 + 8 * v6) = v5;
  qword_4F80AB0 = (__int64)&unk_49D9748;
  qword_4F80A20 = (__int64)&unk_49DC090;
  qword_4F80AC0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F80A70) = qword_4F80A70 + 1;
  qword_4F80AE0 = (__int64)nullsub_23;
  qword_4F80AA8 = 0;
  qword_4F80AD8 = (__int64)sub_984030;
  qword_4F80AB8 = 0;
  sub_C53080(&qword_4F80A20, "print-inst-debug-locs", 21);
  qword_4F80A50 = 57;
  LOBYTE(dword_4F80A2C) = dword_4F80A2C & 0x9F | 0x20;
  qword_4F80A48 = (__int64)"Pretty print debug locations of instructions when dumping";
  sub_C53130(&qword_4F80A20);
  __cxa_atexit(sub_984900, &qword_4F80A20, &qword_4A427C0);
  qword_4F80940 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8094C &= 0x8000u;
  word_4F80950 = 0;
  qword_4F80990 = 0x100000000LL;
  qword_4F80988 = (__int64)&unk_4F80998;
  qword_4F80958 = 0;
  qword_4F80960 = 0;
  dword_4F80948 = v7;
  qword_4F80968 = 0;
  qword_4F80970 = 0;
  qword_4F80978 = 0;
  qword_4F80980 = 0;
  qword_4F809A0 = 0;
  qword_4F809A8 = (__int64)&unk_4F809C0;
  qword_4F809B0 = 1;
  dword_4F809B8 = 0;
  byte_4F809BC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F80990;
  v10 = (unsigned int)qword_4F80990 + 1LL;
  if ( v10 > HIDWORD(qword_4F80990) )
  {
    sub_C8D5F0((char *)&unk_4F80998 - 16, &unk_4F80998, v10, 8);
    v9 = (unsigned int)qword_4F80990;
  }
  *(_QWORD *)(qword_4F80988 + 8 * v9) = v8;
  qword_4F809D0 = (__int64)&unk_49D9748;
  qword_4F80940 = (__int64)&unk_49DC090;
  qword_4F809E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F80990) = qword_4F80990 + 1;
  qword_4F80A00 = (__int64)nullsub_23;
  qword_4F809C8 = 0;
  qword_4F809F8 = (__int64)sub_984030;
  qword_4F809D8 = 0;
  sub_C53080(&qword_4F80940, "print-prof-data", 15);
  qword_4F80970 = 57;
  LOBYTE(dword_4F8094C) = dword_4F8094C & 0x9F | 0x20;
  qword_4F80968 = (__int64)"Pretty print perf data (branch weights, etc) when dumping";
  sub_C53130(&qword_4F80940);
  return __cxa_atexit(sub_984900, &qword_4F80940, &qword_4A427C0);
}
