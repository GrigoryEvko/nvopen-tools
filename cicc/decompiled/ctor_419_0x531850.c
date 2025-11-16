// Function: ctor_419
// Address: 0x531850
//
int ctor_419()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_4FF05E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF0630 = 0x100000000LL;
  dword_4FF05EC &= 0x8000u;
  word_4FF05F0 = 0;
  qword_4FF05F8 = 0;
  qword_4FF0600 = 0;
  dword_4FF05E8 = v0;
  qword_4FF0608 = 0;
  qword_4FF0610 = 0;
  qword_4FF0618 = 0;
  qword_4FF0620 = 0;
  qword_4FF0628 = (__int64)&unk_4FF0638;
  qword_4FF0640 = 0;
  qword_4FF0648 = (__int64)&unk_4FF0660;
  qword_4FF0650 = 1;
  dword_4FF0658 = 0;
  byte_4FF065C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF0630;
  v3 = (unsigned int)qword_4FF0630 + 1LL;
  if ( v3 > HIDWORD(qword_4FF0630) )
  {
    sub_C8D5F0((char *)&unk_4FF0638 - 16, &unk_4FF0638, v3, 8);
    v2 = (unsigned int)qword_4FF0630;
  }
  *(_QWORD *)(qword_4FF0628 + 8 * v2) = v1;
  qword_4FF0670 = (__int64)&unk_49D9748;
  qword_4FF05E0 = (__int64)&unk_49DC090;
  qword_4FF0680 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FF0630) = qword_4FF0630 + 1;
  qword_4FF06A0 = (__int64)nullsub_23;
  qword_4FF0668 = 0;
  qword_4FF0698 = (__int64)sub_984030;
  qword_4FF0678 = 0;
  sub_C53080(&qword_4FF05E0, "enable-nonnull-arg-prop", 23);
  LOWORD(qword_4FF0678) = 257;
  LOBYTE(qword_4FF0668) = 1;
  qword_4FF0610 = 80;
  LOBYTE(dword_4FF05EC) = dword_4FF05EC & 0x9F | 0x20;
  qword_4FF0608 = (__int64)"Try to propagate nonnull argument attributes from callsites to caller functions.";
  sub_C53130(&qword_4FF05E0);
  __cxa_atexit(sub_984900, &qword_4FF05E0, &qword_4A427C0);
  qword_4FF0500 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF0550 = 0x100000000LL;
  dword_4FF050C &= 0x8000u;
  qword_4FF0548 = (__int64)&unk_4FF0558;
  word_4FF0510 = 0;
  qword_4FF0518 = 0;
  dword_4FF0508 = v4;
  qword_4FF0520 = 0;
  qword_4FF0528 = 0;
  qword_4FF0530 = 0;
  qword_4FF0538 = 0;
  qword_4FF0540 = 0;
  qword_4FF0560 = 0;
  qword_4FF0568 = (__int64)&unk_4FF0580;
  qword_4FF0570 = 1;
  dword_4FF0578 = 0;
  byte_4FF057C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF0550;
  if ( (unsigned __int64)(unsigned int)qword_4FF0550 + 1 > HIDWORD(qword_4FF0550) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_4FF0558 - 16, &unk_4FF0558, (unsigned int)qword_4FF0550 + 1LL, 8);
    v6 = (unsigned int)qword_4FF0550;
    v5 = v15;
  }
  *(_QWORD *)(qword_4FF0548 + 8 * v6) = v5;
  qword_4FF0590 = (__int64)&unk_49D9748;
  qword_4FF0500 = (__int64)&unk_49DC090;
  qword_4FF05A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FF0550) = qword_4FF0550 + 1;
  qword_4FF05C0 = (__int64)nullsub_23;
  qword_4FF0588 = 0;
  qword_4FF05B8 = (__int64)sub_984030;
  qword_4FF0598 = 0;
  sub_C53080(&qword_4FF0500, "disable-nounwind-inference", 26);
  qword_4FF0530 = 60;
  LOBYTE(dword_4FF050C) = dword_4FF050C & 0x9F | 0x20;
  qword_4FF0528 = (__int64)"Stop inferring nounwind attribute during function-attrs pass";
  sub_C53130(&qword_4FF0500);
  __cxa_atexit(sub_984900, &qword_4FF0500, &qword_4A427C0);
  qword_4FF0420 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF0470 = 0x100000000LL;
  dword_4FF042C &= 0x8000u;
  word_4FF0430 = 0;
  qword_4FF0468 = (__int64)&unk_4FF0478;
  qword_4FF0438 = 0;
  dword_4FF0428 = v7;
  qword_4FF0440 = 0;
  qword_4FF0448 = 0;
  qword_4FF0450 = 0;
  qword_4FF0458 = 0;
  qword_4FF0460 = 0;
  qword_4FF0480 = 0;
  qword_4FF0488 = (__int64)&unk_4FF04A0;
  qword_4FF0490 = 1;
  dword_4FF0498 = 0;
  byte_4FF049C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FF0470;
  if ( (unsigned __int64)(unsigned int)qword_4FF0470 + 1 > HIDWORD(qword_4FF0470) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_4FF0478 - 16, &unk_4FF0478, (unsigned int)qword_4FF0470 + 1LL, 8);
    v9 = (unsigned int)qword_4FF0470;
    v8 = v16;
  }
  *(_QWORD *)(qword_4FF0468 + 8 * v9) = v8;
  qword_4FF04B0 = (__int64)&unk_49D9748;
  qword_4FF0420 = (__int64)&unk_49DC090;
  qword_4FF04C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FF0470) = qword_4FF0470 + 1;
  qword_4FF04E0 = (__int64)nullsub_23;
  qword_4FF04A8 = 0;
  qword_4FF04D8 = (__int64)sub_984030;
  qword_4FF04B8 = 0;
  sub_C53080(&qword_4FF0420, "disable-nofree-inference", 24);
  qword_4FF0450 = 58;
  LOBYTE(dword_4FF042C) = dword_4FF042C & 0x9F | 0x20;
  qword_4FF0448 = (__int64)"Stop inferring nofree attribute during function-attrs pass";
  sub_C53130(&qword_4FF0420);
  __cxa_atexit(sub_984900, &qword_4FF0420, &qword_4A427C0);
  qword_4FF0340 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF03BC = 1;
  qword_4FF0390 = 0x100000000LL;
  dword_4FF034C &= 0x8000u;
  qword_4FF0388 = (__int64)&unk_4FF0398;
  qword_4FF0358 = 0;
  qword_4FF0360 = 0;
  dword_4FF0348 = v10;
  word_4FF0350 = 0;
  qword_4FF0368 = 0;
  qword_4FF0370 = 0;
  qword_4FF0378 = 0;
  qword_4FF0380 = 0;
  qword_4FF03A0 = 0;
  qword_4FF03A8 = (__int64)&unk_4FF03C0;
  qword_4FF03B0 = 1;
  dword_4FF03B8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FF0390;
  v13 = (unsigned int)qword_4FF0390 + 1LL;
  if ( v13 > HIDWORD(qword_4FF0390) )
  {
    sub_C8D5F0((char *)&unk_4FF0398 - 16, &unk_4FF0398, v13, 8);
    v12 = (unsigned int)qword_4FF0390;
  }
  *(_QWORD *)(qword_4FF0388 + 8 * v12) = v11;
  qword_4FF03D0 = (__int64)&unk_49D9748;
  qword_4FF0340 = (__int64)&unk_49DC090;
  qword_4FF03E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FF0390) = qword_4FF0390 + 1;
  qword_4FF0400 = (__int64)nullsub_23;
  qword_4FF03C8 = 0;
  qword_4FF03F8 = (__int64)sub_984030;
  qword_4FF03D8 = 0;
  sub_C53080(&qword_4FF0340, "disable-thinlto-funcattrs", 25);
  LOBYTE(qword_4FF03C8) = 1;
  LOWORD(qword_4FF03D8) = 257;
  qword_4FF0370 = 41;
  LOBYTE(dword_4FF034C) = dword_4FF034C & 0x9F | 0x20;
  qword_4FF0368 = (__int64)"Don't propagate function-attrs in thinLTO";
  sub_C53130(&qword_4FF0340);
  return __cxa_atexit(sub_984900, &qword_4FF0340, &qword_4A427C0);
}
