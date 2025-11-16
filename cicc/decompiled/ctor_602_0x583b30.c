// Function: ctor_602
// Address: 0x583b30
//
int ctor_602()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  char v7; // [rsp+3h] [rbp-4Dh] BYREF
  int v8; // [rsp+4h] [rbp-4Ch] BYREF
  char *v9; // [rsp+8h] [rbp-48h] BYREF
  const char *v10; // [rsp+10h] [rbp-40h] BYREF
  __int64 v11; // [rsp+18h] [rbp-38h]

  v10 = "Use TargetSchedModel for latency lookup";
  v11 = 39;
  v7 = 1;
  v9 = &v7;
  v8 = 1;
  sub_2FF8600(&unk_502A3C0, "schedmodel", &v8, &v9, &v10);
  __cxa_atexit(sub_984900, &unk_502A3C0, &qword_4A427C0);
  v7 = 1;
  v10 = "Use InstrItineraryData for latency lookup";
  v11 = 41;
  v9 = &v7;
  v8 = 1;
  sub_2FF8600(&unk_502A2E0, "scheditins", &v8, &v9, &v10);
  __cxa_atexit(sub_984900, &unk_502A2E0, &qword_4A427C0);
  qword_502A200 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_502A2E0, v0, v1), 1u);
  byte_502A27C = 1;
  qword_502A250 = 0x100000000LL;
  dword_502A20C &= 0x8000u;
  qword_502A218 = 0;
  qword_502A220 = 0;
  qword_502A228 = 0;
  dword_502A208 = v2;
  word_502A210 = 0;
  qword_502A230 = 0;
  qword_502A238 = 0;
  qword_502A240 = 0;
  qword_502A248 = (__int64)&unk_502A258;
  qword_502A260 = 0;
  qword_502A268 = (__int64)&unk_502A280;
  qword_502A270 = 1;
  dword_502A278 = 0;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_502A250;
  v5 = (unsigned int)qword_502A250 + 1LL;
  if ( v5 > HIDWORD(qword_502A250) )
  {
    sub_C8D5F0((char *)&unk_502A258 - 16, &unk_502A258, v5, 8);
    v4 = (unsigned int)qword_502A250;
  }
  *(_QWORD *)(qword_502A248 + 8 * v4) = v3;
  LODWORD(qword_502A250) = qword_502A250 + 1;
  qword_502A288 = 0;
  qword_502A290 = (__int64)&unk_49D9748;
  qword_502A298 = 0;
  qword_502A200 = (__int64)&unk_49DC090;
  qword_502A2A0 = (__int64)&unk_49DC1D0;
  qword_502A2C0 = (__int64)nullsub_23;
  qword_502A2B8 = (__int64)sub_984030;
  sub_C53080(&qword_502A200, "sched-model-force-enable-intervals", 34);
  LOBYTE(qword_502A288) = 0;
  qword_502A230 = 57;
  LOBYTE(dword_502A20C) = dword_502A20C & 0x9F | 0x20;
  LOWORD(qword_502A298) = 256;
  qword_502A228 = (__int64)"Force the use of resource intervals in the schedule model";
  sub_C53130(&qword_502A200);
  return __cxa_atexit(sub_984900, &qword_502A200, &qword_4A427C0);
}
