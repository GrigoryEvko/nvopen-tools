// Function: ctor_535
// Address: 0x569d60
//
int ctor_535()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5014D80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5014DD0 = 0x100000000LL;
  word_5014D90 = 0;
  dword_5014D8C &= 0x8000u;
  qword_5014D98 = 0;
  qword_5014DA0 = 0;
  dword_5014D88 = v0;
  qword_5014DA8 = 0;
  qword_5014DB0 = 0;
  qword_5014DB8 = 0;
  qword_5014DC0 = 0;
  qword_5014DC8 = (__int64)&unk_5014DD8;
  qword_5014DE0 = 0;
  qword_5014DE8 = (__int64)&unk_5014E00;
  qword_5014DF0 = 1;
  dword_5014DF8 = 0;
  byte_5014DFC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5014DD0;
  v3 = (unsigned int)qword_5014DD0 + 1LL;
  if ( v3 > HIDWORD(qword_5014DD0) )
  {
    sub_C8D5F0((char *)&unk_5014DD8 - 16, &unk_5014DD8, v3, 8);
    v2 = (unsigned int)qword_5014DD0;
  }
  *(_QWORD *)(qword_5014DC8 + 8 * v2) = v1;
  LODWORD(qword_5014DD0) = qword_5014DD0 + 1;
  qword_5014E08 = 0;
  qword_5014E10 = (__int64)&unk_49D9748;
  qword_5014E18 = 0;
  qword_5014D80 = (__int64)&unk_49DC090;
  qword_5014E20 = (__int64)&unk_49DC1D0;
  qword_5014E40 = (__int64)nullsub_23;
  qword_5014E38 = (__int64)sub_984030;
  sub_C53080(&qword_5014D80, "nvvm-reflect-enable", 19);
  LOBYTE(qword_5014E08) = 1;
  LOWORD(qword_5014E18) = 257;
  qword_5014DB0 = 35;
  LOBYTE(dword_5014D8C) = dword_5014D8C & 0x9F | 0x20;
  qword_5014DA8 = (__int64)"NVVM reflection, enabled by default";
  sub_C53130(&qword_5014D80);
  __cxa_atexit(sub_984900, &qword_5014D80, &qword_4A427C0);
  qword_5014C60 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5014C78 = 0;
  qword_5014C80 = 0;
  qword_5014C88 = 0;
  qword_5014C90 = 0;
  dword_5014C6C = dword_5014C6C & 0x8000 | 1;
  qword_5014CB0 = 0x100000000LL;
  dword_5014C68 = v4;
  word_5014C70 = 0;
  qword_5014C98 = 0;
  qword_5014CA0 = 0;
  qword_5014CA8 = (__int64)&unk_5014CB8;
  qword_5014CC0 = 0;
  qword_5014CC8 = (__int64)&unk_5014CE0;
  qword_5014CD0 = 1;
  dword_5014CD8 = 0;
  byte_5014CDC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5014CB0;
  v7 = (unsigned int)qword_5014CB0 + 1LL;
  if ( v7 > HIDWORD(qword_5014CB0) )
  {
    sub_C8D5F0((char *)&unk_5014CB8 - 16, &unk_5014CB8, v7, 8);
    v6 = (unsigned int)qword_5014CB0;
  }
  *(_QWORD *)(qword_5014CA8 + 8 * v6) = v5;
  LODWORD(qword_5014CB0) = qword_5014CB0 + 1;
  qword_5014CE8 = 0;
  qword_5014C60 = (__int64)&unk_49DAD08;
  qword_5014CF0 = 0;
  qword_5014CF8 = 0;
  qword_5014D38 = (__int64)&unk_49DC350;
  qword_5014D00 = 0;
  qword_5014D58 = (__int64)nullsub_81;
  qword_5014D08 = 0;
  qword_5014D50 = (__int64)sub_BB8600;
  qword_5014D10 = 0;
  byte_5014D18 = 0;
  qword_5014D20 = 0;
  qword_5014D28 = 0;
  qword_5014D30 = 0;
  sub_C53080(&qword_5014C60, "R", 1);
  qword_5014CA0 = 10;
  qword_5014C98 = (__int64)"name=<int>";
  qword_5014C88 = (__int64)"A list of string=num assignments";
  qword_5014C90 = 32;
  LOBYTE(dword_5014C6C) = dword_5014C6C & 0x87 | 0x30;
  sub_C53130(&qword_5014C60);
  return __cxa_atexit(sub_BB89D0, &qword_5014C60, &qword_4A427C0);
}
