// Function: ctor_614
// Address: 0x589c70
//
int ctor_614()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // edx
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rdx

  if ( getenv("bar") == (char *)-1LL )
  {
    nullsub_1866();
    nullsub_1857();
  }
  if ( getenv("bar") == (char *)-1LL )
  {
    sub_2F42900();
    sub_35B6440();
    sub_2F504C0();
    sub_35B9BC0();
    sub_3355700(0, 2);
    sub_33553F0(0, 2);
    sub_3354FA0(0, 2);
    sub_334CB40(0, 2);
    sub_341EBC0(0, 2);
    sub_3363950(0, 2);
  }
  qword_502D708 = 1;
  qword_502D700 = (__int64)&qword_502D730;
  qword_502D710 = 0;
  qword_502D718 = 0;
  dword_502D720 = 1065353216;
  qword_502D728 = 0;
  qword_502D730 = 0;
  __cxa_atexit(sub_8565C0, &qword_502D730 - 6, &qword_4A427C0);
  if ( getenv("bar") == (char *)-1LL )
    sub_5895F0((__int64)"bar", (int)(&qword_502D730 - 6));
  sub_2208040(&unk_502D6E9);
  __cxa_atexit(sub_2208810, &unk_502D6E9, &qword_4A427C0);
  sub_2D97F20(&unk_502D6E8);
  qword_502D620 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(&unk_502D6E8, &unk_502D6E9, v0, v1), 1u);
  byte_502D69C = 1;
  qword_502D670 = 0x100000000LL;
  dword_502D62C &= 0x8000u;
  qword_502D638 = 0;
  qword_502D640 = 0;
  qword_502D648 = 0;
  dword_502D628 = v2;
  word_502D630 = 0;
  qword_502D650 = 0;
  qword_502D658 = 0;
  qword_502D660 = 0;
  qword_502D668 = (__int64)&unk_502D678;
  qword_502D680 = 0;
  qword_502D688 = (__int64)&unk_502D6A0;
  qword_502D690 = 1;
  dword_502D698 = 0;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_502D670;
  v5 = (unsigned int)qword_502D670 + 1LL;
  if ( v5 > HIDWORD(qword_502D670) )
  {
    sub_C8D5F0((char *)&unk_502D678 - 16, &unk_502D678, v5, 8);
    v4 = (unsigned int)qword_502D670;
  }
  *(_QWORD *)(qword_502D668 + 8 * v4) = v3;
  LODWORD(qword_502D670) = qword_502D670 + 1;
  qword_502D6A8 = 0;
  qword_502D6B0 = (__int64)&unk_49DA090;
  qword_502D6B8 = 0;
  qword_502D620 = (__int64)&unk_49DBF90;
  qword_502D6C0 = (__int64)&unk_49DC230;
  qword_502D6E0 = (__int64)nullsub_58;
  qword_502D6D8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502D620, "heavy-const-expr-size", 21);
  qword_502D648 = (__int64)"This option is a no-op and will be removed";
  qword_502D650 = 42;
  LOBYTE(dword_502D62C) = dword_502D62C & 0x9F | 0x20;
  sub_C53130(&qword_502D620);
  __cxa_atexit(sub_B2B680, &qword_502D620, &qword_4A427C0);
  qword_502D540 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502D620, v6, v7), 1u);
  dword_502D54C &= 0x8000u;
  word_502D550 = 0;
  qword_502D590 = 0x100000000LL;
  qword_502D558 = 0;
  qword_502D560 = 0;
  qword_502D568 = 0;
  dword_502D548 = v8;
  qword_502D570 = 0;
  qword_502D578 = 0;
  qword_502D580 = 0;
  qword_502D588 = (__int64)&unk_502D598;
  qword_502D5A0 = 0;
  qword_502D5A8 = (__int64)&unk_502D5C0;
  qword_502D5B0 = 1;
  dword_502D5B8 = 0;
  byte_502D5BC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_502D590;
  v11 = (unsigned int)qword_502D590 + 1LL;
  if ( v11 > HIDWORD(qword_502D590) )
  {
    sub_C8D5F0((char *)&unk_502D598 - 16, &unk_502D598, v11, 8);
    v10 = (unsigned int)qword_502D590;
  }
  *(_QWORD *)(qword_502D588 + 8 * v10) = v9;
  LODWORD(qword_502D590) = qword_502D590 + 1;
  qword_502D5C8 = 0;
  qword_502D5D0 = (__int64)&unk_49D9728;
  qword_502D5D8 = 0;
  qword_502D540 = (__int64)&unk_49DBF10;
  qword_502D5E0 = (__int64)&unk_49DC290;
  qword_502D600 = (__int64)nullsub_24;
  qword_502D5F8 = (__int64)sub_984050;
  sub_C53080(&qword_502D540, "remat-loop-trip", 15);
  qword_502D568 = (__int64)"This option is a no-op and will be removed";
  qword_502D570 = 42;
  LOBYTE(dword_502D54C) = dword_502D54C & 0x9F | 0x20;
  sub_C53130(&qword_502D540);
  return __cxa_atexit(sub_984970, &qword_502D540, &qword_4A427C0);
}
