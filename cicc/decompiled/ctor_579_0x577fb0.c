// Function: ctor_579
// Address: 0x577fb0
//
int ctor_579()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v5; // [rsp+0h] [rbp-50h] BYREF
  int v6; // [rsp+4h] [rbp-4Ch] BYREF
  int *v7; // [rsp+8h] [rbp-48h] BYREF
  const char *v8; // [rsp+10h] [rbp-40h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h]

  qword_5023280 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50232FC = 1;
  qword_50232D0 = 0x100000000LL;
  dword_502328C &= 0x8000u;
  qword_5023298 = 0;
  qword_50232A0 = 0;
  qword_50232A8 = 0;
  dword_5023288 = v0;
  word_5023290 = 0;
  qword_50232B0 = 0;
  qword_50232B8 = 0;
  qword_50232C0 = 0;
  qword_50232C8 = (__int64)&unk_50232D8;
  qword_50232E0 = 0;
  qword_50232E8 = (__int64)&unk_5023300;
  qword_50232F0 = 1;
  dword_50232F8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50232D0;
  v3 = (unsigned int)qword_50232D0 + 1LL;
  if ( v3 > HIDWORD(qword_50232D0) )
  {
    sub_C8D5F0((char *)&unk_50232D8 - 16, &unk_50232D8, v3, 8);
    v2 = (unsigned int)qword_50232D0;
  }
  *(_QWORD *)(qword_50232C8 + 8 * v2) = v1;
  LODWORD(qword_50232D0) = qword_50232D0 + 1;
  qword_5023308 = 0;
  qword_5023310 = (__int64)&unk_49D9748;
  qword_5023318 = 0;
  qword_5023280 = (__int64)&unk_49DC090;
  qword_5023320 = (__int64)&unk_49DC1D0;
  qword_5023340 = (__int64)nullsub_23;
  qword_5023338 = (__int64)sub_984030;
  sub_C53080(&qword_5023280, "post-RA-scheduler", 17);
  qword_50232B0 = 43;
  qword_50232A8 = (__int64)"Enable scheduling after register allocation";
  LOWORD(qword_5023318) = 256;
  LOBYTE(qword_5023308) = 0;
  LOBYTE(dword_502328C) = dword_502328C & 0x9F | 0x20;
  sub_C53130(&qword_5023280);
  __cxa_atexit(sub_984900, &qword_5023280, &qword_4A427C0);
  v6 = 1;
  v7 = (int *)"none";
  v8 = "Break post-RA scheduling anti-dependencies: \"critical\", \"all\", or \"none\"";
  v9 = 72;
  sub_2F39D10(&unk_5023180, "break-anti-dependencies", &v8, &v7, &v6);
  __cxa_atexit(sub_BC5A40, &unk_5023180, &qword_4A427C0);
  v7 = &v6;
  v8 = "Debug control MBBs that are scheduled";
  v5 = 1;
  v6 = 0;
  v9 = 37;
  ((void (__fastcall *)(void *, const char *, const char **, int **, int *))sub_2F3A030)(
    &unk_50230A0,
    "postra-sched-debugdiv",
    &v8,
    &v7,
    &v5);
  __cxa_atexit(sub_B2B680, &unk_50230A0, &qword_4A427C0);
  v7 = &v6;
  v5 = 1;
  v6 = 0;
  v8 = "Debug control MBBs that are scheduled";
  v9 = 37;
  ((void (__fastcall *)(void *, const char *, const char **, int **, int *, const char *))sub_2F3A030)(
    &unk_5022FC0,
    "postra-sched-debugmod",
    &v8,
    &v7,
    &v5,
    "Debug control MBBs that are scheduled");
  return __cxa_atexit(sub_B2B680, &unk_5022FC0, &qword_4A427C0);
}
