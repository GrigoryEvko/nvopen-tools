// Function: ctor_434
// Address: 0x53b5c0
//
int ctor_434()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-58h]
  int v13; // [rsp+10h] [rbp-50h] BYREF
  int v14; // [rsp+14h] [rbp-4Ch] BYREF
  int *v15; // [rsp+18h] [rbp-48h] BYREF
  const char *v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  qword_4FF8860 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF88B0 = 0x100000000LL;
  dword_4FF886C &= 0x8000u;
  word_4FF8870 = 0;
  qword_4FF8878 = 0;
  qword_4FF8880 = 0;
  dword_4FF8868 = v0;
  qword_4FF8888 = 0;
  qword_4FF8890 = 0;
  qword_4FF8898 = 0;
  qword_4FF88A0 = 0;
  qword_4FF88A8 = (__int64)&unk_4FF88B8;
  qword_4FF88C0 = 0;
  qword_4FF88C8 = (__int64)&unk_4FF88E0;
  qword_4FF88D0 = 1;
  dword_4FF88D8 = 0;
  byte_4FF88DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF88B0;
  v3 = (unsigned int)qword_4FF88B0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF88B0) )
  {
    sub_C8D5F0((char *)&unk_4FF88B8 - 16, &unk_4FF88B8, v3, 8);
    v2 = (unsigned int)qword_4FF88B0;
  }
  *(_QWORD *)(qword_4FF88A8 + 8 * v2) = v1;
  LODWORD(qword_4FF88B0) = qword_4FF88B0 + 1;
  qword_4FF88E8 = 0;
  qword_4FF88F0 = (__int64)&unk_49D9728;
  qword_4FF88F8 = 0;
  qword_4FF8860 = (__int64)&unk_49DBF10;
  qword_4FF8900 = (__int64)&unk_49DC290;
  qword_4FF8920 = (__int64)nullsub_24;
  qword_4FF8918 = (__int64)sub_984050;
  sub_C53080(&qword_4FF8860, "func-profile-similarity-threshold", 33);
  LODWORD(qword_4FF88E8) = 80;
  BYTE4(qword_4FF88F8) = 1;
  LODWORD(qword_4FF88F8) = 80;
  qword_4FF8890 = 116;
  LOBYTE(dword_4FF886C) = dword_4FF886C & 0x9F | 0x20;
  qword_4FF8888 = (__int64)"Consider a profile matches a function if the similarity of their callee sequences is above th"
                           "e specified percentile.";
  sub_C53130(&qword_4FF8860);
  __cxa_atexit(sub_984970, &qword_4FF8860, &qword_4A427C0);
  v16 = "The minimum number of basic blocks required for a function to run stale profile call graph matching.";
  v17 = 100;
  v13 = 5;
  v15 = &v13;
  v14 = 1;
  sub_26E12F0(&unk_4FF8780, "min-func-count-for-cg-matching", &v14, &v15);
  __cxa_atexit(sub_984970, &unk_4FF8780, &qword_4A427C0);
  v17 = 100;
  v16 = "The minimum number of call anchors required for a function to run stale profile call graph matching.";
  v13 = 3;
  v15 = &v13;
  v14 = 1;
  sub_26E12F0(&unk_4FF86A0, "min-call-count-for-cg-matching", &v14, &v15);
  __cxa_atexit(sub_984970, &unk_4FF86A0, &qword_4A427C0);
  qword_4FF85C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF8610 = 0x100000000LL;
  word_4FF85D0 = 0;
  dword_4FF85CC &= 0x8000u;
  qword_4FF85D8 = 0;
  qword_4FF85E0 = 0;
  dword_4FF85C8 = v4;
  qword_4FF85E8 = 0;
  qword_4FF85F0 = 0;
  qword_4FF85F8 = 0;
  qword_4FF8600 = 0;
  qword_4FF8608 = (__int64)&unk_4FF8618;
  qword_4FF8620 = 0;
  qword_4FF8628 = (__int64)&unk_4FF8640;
  qword_4FF8630 = 1;
  dword_4FF8638 = 0;
  byte_4FF863C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF8610;
  if ( (unsigned __int64)(unsigned int)qword_4FF8610 + 1 > HIDWORD(qword_4FF8610) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FF8618 - 16, &unk_4FF8618, (unsigned int)qword_4FF8610 + 1LL, 8);
    v6 = (unsigned int)qword_4FF8610;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FF8608 + 8 * v6) = v5;
  LODWORD(qword_4FF8610) = qword_4FF8610 + 1;
  qword_4FF8648 = 0;
  qword_4FF8650 = (__int64)&unk_49D9748;
  qword_4FF8658 = 0;
  qword_4FF85C0 = (__int64)&unk_49DC090;
  qword_4FF8660 = (__int64)&unk_49DC1D0;
  qword_4FF8680 = (__int64)nullsub_23;
  qword_4FF8678 = (__int64)sub_984030;
  sub_C53080(&qword_4FF85C0, "load-func-profile-for-cg-matching", 33);
  LOBYTE(qword_4FF8648) = 1;
  qword_4FF85F0 = 137;
  LOBYTE(dword_4FF85CC) = dword_4FF85CC & 0x9F | 0x20;
  LOWORD(qword_4FF8658) = 257;
  qword_4FF85E8 = (__int64)"Load top-level profiles that the sample reader initially skipped for the call-graph matching "
                           "(only meaningful for extended binary format)";
  sub_C53130(&qword_4FF85C0);
  __cxa_atexit(sub_984900, &qword_4FF85C0, &qword_4A427C0);
  qword_4FF84E0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF855C = 1;
  qword_4FF8530 = 0x100000000LL;
  dword_4FF84EC &= 0x8000u;
  qword_4FF84F8 = 0;
  qword_4FF8500 = 0;
  qword_4FF8508 = 0;
  dword_4FF84E8 = v7;
  word_4FF84F0 = 0;
  qword_4FF8510 = 0;
  qword_4FF8518 = 0;
  qword_4FF8520 = 0;
  qword_4FF8528 = (__int64)&unk_4FF8538;
  qword_4FF8540 = 0;
  qword_4FF8548 = (__int64)&unk_4FF8560;
  qword_4FF8550 = 1;
  dword_4FF8558 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FF8530;
  v10 = (unsigned int)qword_4FF8530 + 1LL;
  if ( v10 > HIDWORD(qword_4FF8530) )
  {
    sub_C8D5F0((char *)&unk_4FF8538 - 16, &unk_4FF8538, v10, 8);
    v9 = (unsigned int)qword_4FF8530;
  }
  *(_QWORD *)(qword_4FF8528 + 8 * v9) = v8;
  LODWORD(qword_4FF8530) = qword_4FF8530 + 1;
  qword_4FF8568 = 0;
  qword_4FF8570 = (__int64)&unk_49D9728;
  qword_4FF8578 = 0;
  qword_4FF84E0 = (__int64)&unk_49DBF10;
  qword_4FF8580 = (__int64)&unk_49DC290;
  qword_4FF85A0 = (__int64)nullsub_24;
  qword_4FF8598 = (__int64)sub_984050;
  sub_C53080(&qword_4FF84E0, "salvage-stale-profile-max-callsites", 35);
  LODWORD(qword_4FF8568) = -1;
  BYTE4(qword_4FF8578) = 1;
  LODWORD(qword_4FF8578) = -1;
  qword_4FF8510 = 98;
  LOBYTE(dword_4FF84EC) = dword_4FF84EC & 0x9F | 0x20;
  qword_4FF8508 = (__int64)"The maximum number of callsites in a function, above which stale profile matching will be skipped.";
  sub_C53130(&qword_4FF84E0);
  return __cxa_atexit(sub_984970, &qword_4FF84E0, &qword_4A427C0);
}
