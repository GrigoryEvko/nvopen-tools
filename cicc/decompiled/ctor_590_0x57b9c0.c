// Function: ctor_590
// Address: 0x57b9c0
//
int __fastcall ctor_590(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

  qword_5025520 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5025570 = 0x100000000LL;
  dword_502552C &= 0x8000u;
  word_5025530 = 0;
  qword_5025538 = 0;
  qword_5025540 = 0;
  dword_5025528 = v4;
  qword_5025548 = 0;
  qword_5025550 = 0;
  qword_5025558 = 0;
  qword_5025560 = 0;
  qword_5025568 = (__int64)&unk_5025578;
  qword_5025580 = 0;
  qword_5025588 = (__int64)&unk_50255A0;
  qword_5025590 = 1;
  dword_5025598 = 0;
  byte_502559C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5025570;
  v7 = (unsigned int)qword_5025570 + 1LL;
  if ( v7 > HIDWORD(qword_5025570) )
  {
    sub_C8D5F0((char *)&unk_5025578 - 16, &unk_5025578, v7, 8);
    v6 = (unsigned int)qword_5025570;
  }
  *(_QWORD *)(qword_5025568 + 8 * v6) = v5;
  qword_50255B0 = (__int64)&unk_49D9748;
  qword_5025520 = (__int64)&unk_49DC090;
  LODWORD(qword_5025570) = qword_5025570 + 1;
  qword_50255A8 = 0;
  qword_50255C0 = (__int64)&unk_49DC1D0;
  qword_50255B8 = 0;
  qword_50255E0 = (__int64)nullsub_23;
  qword_50255D8 = (__int64)sub_984030;
  sub_C53080(&qword_5025520, "enable-aa-sched-mi", 18);
  qword_5025550 = 43;
  LOBYTE(dword_502552C) = dword_502552C & 0x9F | 0x20;
  qword_5025548 = (__int64)"Enable use of AA during MI DAG construction";
  sub_C53130(&qword_5025520);
  __cxa_atexit(sub_984900, &qword_5025520, &qword_4A427C0);
  qword_5025440 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5025520, v8, v9), 1u);
  qword_5025490 = 0x100000000LL;
  dword_502544C &= 0x8000u;
  word_5025450 = 0;
  qword_5025458 = 0;
  qword_5025460 = 0;
  dword_5025448 = v10;
  qword_5025468 = 0;
  qword_5025470 = 0;
  qword_5025478 = 0;
  qword_5025480 = 0;
  qword_5025488 = (__int64)&unk_5025498;
  qword_50254A0 = 0;
  qword_50254A8 = (__int64)&unk_50254C0;
  qword_50254B0 = 1;
  dword_50254B8 = 0;
  byte_50254BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5025490;
  if ( (unsigned __int64)(unsigned int)qword_5025490 + 1 > HIDWORD(qword_5025490) )
  {
    v26 = v11;
    sub_C8D5F0((char *)&unk_5025498 - 16, &unk_5025498, (unsigned int)qword_5025490 + 1LL, 8);
    v12 = (unsigned int)qword_5025490;
    v11 = v26;
  }
  *(_QWORD *)(qword_5025488 + 8 * v12) = v11;
  qword_50254D0 = (__int64)&unk_49D9748;
  qword_5025440 = (__int64)&unk_49DC090;
  LODWORD(qword_5025490) = qword_5025490 + 1;
  qword_50254C8 = 0;
  qword_50254E0 = (__int64)&unk_49DC1D0;
  qword_50254D8 = 0;
  qword_5025500 = (__int64)nullsub_23;
  qword_50254F8 = (__int64)sub_984030;
  sub_C53080(&qword_5025440, "use-tbaa-in-sched-mi", 20);
  LOWORD(qword_50254D8) = 257;
  LOBYTE(qword_50254C8) = 1;
  qword_5025470 = 45;
  LOBYTE(dword_502544C) = dword_502544C & 0x9F | 0x20;
  qword_5025468 = (__int64)"Enable use of TBAA during MI DAG construction";
  sub_C53130(&qword_5025440);
  __cxa_atexit(sub_984900, &qword_5025440, &qword_4A427C0);
  qword_5025360 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5025440, v13, v14), 1u);
  qword_50253B0 = 0x100000000LL;
  word_5025370 = 0;
  dword_502536C &= 0x8000u;
  qword_5025378 = 0;
  qword_5025380 = 0;
  dword_5025368 = v15;
  qword_5025388 = 0;
  qword_5025390 = 0;
  qword_5025398 = 0;
  qword_50253A0 = 0;
  qword_50253A8 = (__int64)&unk_50253B8;
  qword_50253C0 = 0;
  qword_50253C8 = (__int64)&unk_50253E0;
  qword_50253D0 = 1;
  dword_50253D8 = 0;
  byte_50253DC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_50253B0;
  v18 = (unsigned int)qword_50253B0 + 1LL;
  if ( v18 > HIDWORD(qword_50253B0) )
  {
    sub_C8D5F0((char *)&unk_50253B8 - 16, &unk_50253B8, v18, 8);
    v17 = (unsigned int)qword_50253B0;
  }
  *(_QWORD *)(qword_50253A8 + 8 * v17) = v16;
  qword_50253F0 = (__int64)&unk_49D9728;
  qword_5025360 = (__int64)&unk_49DBF10;
  LODWORD(qword_50253B0) = qword_50253B0 + 1;
  qword_50253E8 = 0;
  qword_5025400 = (__int64)&unk_49DC290;
  qword_50253F8 = 0;
  qword_5025420 = (__int64)nullsub_24;
  qword_5025418 = (__int64)sub_984050;
  sub_C53080(&qword_5025360, "dag-maps-huge-region", 20);
  LODWORD(qword_50253E8) = 1000;
  BYTE4(qword_50253F8) = 1;
  LODWORD(qword_50253F8) = 1000;
  qword_5025390 = 132;
  LOBYTE(dword_502536C) = dword_502536C & 0x9F | 0x20;
  qword_5025388 = (__int64)"The limit to use while constructing the DAG prior to scheduling, at which point a trade-off i"
                           "s made to avoid excessive compile time.";
  sub_C53130(&qword_5025360);
  __cxa_atexit(sub_984970, &qword_5025360, &qword_4A427C0);
  qword_5025280 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5025360, v19, v20), 1u);
  dword_502528C &= 0x8000u;
  word_5025290 = 0;
  qword_50252D0 = 0x100000000LL;
  qword_5025298 = 0;
  qword_50252A0 = 0;
  qword_50252A8 = 0;
  dword_5025288 = v21;
  qword_50252B0 = 0;
  qword_50252B8 = 0;
  qword_50252C0 = 0;
  qword_50252C8 = (__int64)&unk_50252D8;
  qword_50252E0 = 0;
  qword_50252E8 = (__int64)&unk_5025300;
  qword_50252F0 = 1;
  dword_50252F8 = 0;
  byte_50252FC = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_50252D0;
  v24 = (unsigned int)qword_50252D0 + 1LL;
  if ( v24 > HIDWORD(qword_50252D0) )
  {
    sub_C8D5F0((char *)&unk_50252D8 - 16, &unk_50252D8, v24, 8);
    v23 = (unsigned int)qword_50252D0;
  }
  *(_QWORD *)(qword_50252C8 + 8 * v23) = v22;
  qword_5025310 = (__int64)&unk_49D9728;
  qword_5025280 = (__int64)&unk_49DBF10;
  LODWORD(qword_50252D0) = qword_50252D0 + 1;
  qword_5025308 = 0;
  qword_5025320 = (__int64)&unk_49DC290;
  qword_5025318 = 0;
  qword_5025340 = (__int64)nullsub_24;
  qword_5025338 = (__int64)sub_984050;
  sub_C53080(&qword_5025280, "dag-maps-reduction-size", 23);
  qword_50252B0 = 105;
  LOBYTE(dword_502528C) = dword_502528C & 0x9F | 0x20;
  qword_50252A8 = (__int64)"A huge scheduling region will have maps reduced by this many nodes at a time. Defaults to HugeRegion / 2.";
  sub_C53130(&qword_5025280);
  return __cxa_atexit(sub_984970, &qword_5025280, &qword_4A427C0);
}
