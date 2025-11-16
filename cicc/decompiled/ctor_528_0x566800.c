// Function: ctor_528
// Address: 0x566800
//
int ctor_528()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_50134C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501353C = 1;
  qword_5013510 = 0x100000000LL;
  dword_50134CC &= 0x8000u;
  qword_50134D8 = 0;
  qword_50134E0 = 0;
  qword_50134E8 = 0;
  dword_50134C8 = v0;
  word_50134D0 = 0;
  qword_50134F0 = 0;
  qword_50134F8 = 0;
  qword_5013500 = 0;
  qword_5013508 = (__int64)&unk_5013518;
  qword_5013520 = 0;
  qword_5013528 = (__int64)&unk_5013540;
  qword_5013530 = 1;
  dword_5013538 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5013510;
  v3 = (unsigned int)qword_5013510 + 1LL;
  if ( v3 > HIDWORD(qword_5013510) )
  {
    sub_C8D5F0((char *)&unk_5013518 - 16, &unk_5013518, v3, 8);
    v2 = (unsigned int)qword_5013510;
  }
  *(_QWORD *)(qword_5013508 + 8 * v2) = v1;
  qword_5013550 = (__int64)&unk_49DA090;
  LODWORD(qword_5013510) = qword_5013510 + 1;
  qword_5013548 = 0;
  qword_50134C0 = (__int64)&unk_49DBF90;
  qword_5013560 = (__int64)&unk_49DC230;
  qword_5013558 = 0;
  qword_5013580 = (__int64)nullsub_58;
  qword_5013578 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_50134C0, "dump-ip-msp", 11);
  LODWORD(qword_5013548) = 0;
  BYTE4(qword_5013558) = 1;
  LODWORD(qword_5013558) = 0;
  qword_50134F0 = 63;
  LOBYTE(dword_50134CC) = dword_50134CC & 0x9F | 0x20;
  qword_50134E8 = (__int64)"Dump information from Inter-Procedural Memory Space Propagation";
  sub_C53130(&qword_50134C0);
  __cxa_atexit(sub_B2B680, &qword_50134C0, &qword_4A427C0);
  qword_50133E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50133EC &= 0x8000u;
  word_50133F0 = 0;
  qword_5013430 = 0x100000000LL;
  qword_50133F8 = 0;
  qword_5013400 = 0;
  qword_5013408 = 0;
  dword_50133E8 = v4;
  qword_5013410 = 0;
  qword_5013418 = 0;
  qword_5013420 = 0;
  qword_5013428 = (__int64)&unk_5013438;
  qword_5013440 = 0;
  qword_5013448 = (__int64)&unk_5013460;
  qword_5013450 = 1;
  dword_5013458 = 0;
  byte_501345C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5013430;
  v7 = (unsigned int)qword_5013430 + 1LL;
  if ( v7 > HIDWORD(qword_5013430) )
  {
    sub_C8D5F0((char *)&unk_5013438 - 16, &unk_5013438, v7, 8);
    v6 = (unsigned int)qword_5013430;
  }
  *(_QWORD *)(qword_5013428 + 8 * v6) = v5;
  qword_5013470 = (__int64)&unk_49DA090;
  LODWORD(qword_5013430) = qword_5013430 + 1;
  qword_5013468 = 0;
  qword_50133E0 = (__int64)&unk_49DBF90;
  qword_5013480 = (__int64)&unk_49DC230;
  qword_5013478 = 0;
  qword_50134A0 = (__int64)nullsub_58;
  qword_5013498 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_50133E0, "do-clone-for-ip-msp", 19);
  LODWORD(qword_5013468) = -1;
  BYTE4(qword_5013478) = 1;
  LODWORD(qword_5013478) = -1;
  qword_5013410 = 70;
  LOBYTE(dword_50133EC) = dword_50133EC & 0x9F | 0x20;
  qword_5013408 = (__int64)"Control number of clones for inter-procedural Memory Space Propagation";
  sub_C53130(&qword_50133E0);
  return __cxa_atexit(sub_B2B680, &qword_50133E0, &qword_4A427C0);
}
