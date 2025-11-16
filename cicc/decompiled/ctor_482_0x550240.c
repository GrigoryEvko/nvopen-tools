// Function: ctor_482
// Address: 0x550240
//
int ctor_482()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_5005640 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5005690 = 0x100000000LL;
  dword_500564C &= 0x8000u;
  word_5005650 = 0;
  qword_5005658 = 0;
  qword_5005660 = 0;
  dword_5005648 = v0;
  qword_5005668 = 0;
  qword_5005670 = 0;
  qword_5005678 = 0;
  qword_5005680 = 0;
  qword_5005688 = (__int64)&unk_5005698;
  qword_50056A0 = 0;
  qword_50056A8 = (__int64)&unk_50056C0;
  qword_50056B0 = 1;
  dword_50056B8 = 0;
  byte_50056BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5005690;
  v3 = (unsigned int)qword_5005690 + 1LL;
  if ( v3 > HIDWORD(qword_5005690) )
  {
    sub_C8D5F0((char *)&unk_5005698 - 16, &unk_5005698, v3, 8);
    v2 = (unsigned int)qword_5005690;
  }
  *(_QWORD *)(qword_5005688 + 8 * v2) = v1;
  LODWORD(qword_5005690) = qword_5005690 + 1;
  byte_50056E0 = 0;
  qword_50056D0 = (__int64)&unk_49DB998;
  qword_50056C8 = 0;
  qword_50056D8 = 0;
  qword_5005640 = (__int64)&unk_49DB9B8;
  qword_50056E8 = (__int64)&unk_49DC2C0;
  qword_5005708 = (__int64)nullsub_121;
  qword_5005700 = (__int64)sub_C1A370;
  sub_C53080(&qword_5005640, "sroa-size-limit", 15);
  qword_5005670 = 44;
  qword_50056C8 = 0x2000;
  byte_50056E0 = 1;
  qword_50056D8 = 0x2000;
  LOBYTE(dword_500564C) = dword_500564C & 0x9F | 0x20;
  qword_5005668 = (__int64)"Limit the size of aggregate that is replaced";
  sub_C53130(&qword_5005640);
  __cxa_atexit(sub_C1A610, &qword_5005640, &qword_4A427C0);
  qword_5005560 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50055B0 = 0x100000000LL;
  dword_500556C &= 0x8000u;
  word_5005570 = 0;
  qword_5005578 = 0;
  qword_5005580 = 0;
  dword_5005568 = v4;
  qword_5005588 = 0;
  qword_5005590 = 0;
  qword_5005598 = 0;
  qword_50055A0 = 0;
  qword_50055A8 = (__int64)&unk_50055B8;
  qword_50055C0 = 0;
  qword_50055C8 = (__int64)&unk_50055E0;
  qword_50055D0 = 1;
  dword_50055D8 = 0;
  byte_50055DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50055B0;
  v7 = (unsigned int)qword_50055B0 + 1LL;
  if ( v7 > HIDWORD(qword_50055B0) )
  {
    sub_C8D5F0((char *)&unk_50055B8 - 16, &unk_50055B8, v7, 8);
    v6 = (unsigned int)qword_50055B0;
  }
  *(_QWORD *)(qword_50055A8 + 8 * v6) = v5;
  qword_50055F0 = (__int64)&unk_49D9748;
  qword_5005560 = (__int64)&unk_49DC090;
  qword_5005600 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50055B0) = qword_50055B0 + 1;
  qword_5005620 = (__int64)nullsub_23;
  qword_50055E8 = 0;
  qword_5005618 = (__int64)sub_984030;
  qword_50055F8 = 0;
  sub_C53080(&qword_5005560, "sroa-builder-prepass", 20);
  LOWORD(qword_50055F8) = 256;
  LOBYTE(qword_50055E8) = 0;
  qword_5005590 = 62;
  LOBYTE(dword_500556C) = dword_500556C & 0x9F | 0x20;
  qword_5005588 = (__int64)"Prepass check to disable FCA splitting if SliceBuilder aborts.";
  sub_C53130(&qword_5005560);
  __cxa_atexit(sub_984900, &qword_5005560, &qword_4A427C0);
  qword_5005480 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50054D0 = 0x100000000LL;
  dword_500548C &= 0x8000u;
  qword_50054C8 = (__int64)&unk_50054D8;
  word_5005490 = 0;
  qword_5005498 = 0;
  dword_5005488 = v8;
  qword_50054A0 = 0;
  qword_50054A8 = 0;
  qword_50054B0 = 0;
  qword_50054B8 = 0;
  qword_50054C0 = 0;
  qword_50054E0 = 0;
  qword_50054E8 = (__int64)&unk_5005500;
  qword_50054F0 = 1;
  dword_50054F8 = 0;
  byte_50054FC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_50054D0;
  if ( (unsigned __int64)(unsigned int)qword_50054D0 + 1 > HIDWORD(qword_50054D0) )
  {
    v16 = v9;
    sub_C8D5F0((char *)&unk_50054D8 - 16, &unk_50054D8, (unsigned int)qword_50054D0 + 1LL, 8);
    v10 = (unsigned int)qword_50054D0;
    v9 = v16;
  }
  *(_QWORD *)(qword_50054C8 + 8 * v10) = v9;
  qword_5005510 = (__int64)&unk_49D9748;
  qword_5005480 = (__int64)&unk_49DC090;
  qword_5005520 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50054D0) = qword_50054D0 + 1;
  qword_5005540 = (__int64)nullsub_23;
  qword_5005508 = 0;
  qword_5005538 = (__int64)sub_984030;
  qword_5005518 = 0;
  sub_C53080(&qword_5005480, "sroa-replace-bitwise-integer", 28);
  LOWORD(qword_5005518) = 257;
  LOBYTE(qword_5005508) = 1;
  qword_50054B0 = 80;
  LOBYTE(dword_500548C) = dword_500548C & 0x9F | 0x20;
  qword_50054A8 = (__int64)"Replace bitwise integer operations with vector insert/extract for 1/2 size types";
  sub_C53130(&qword_5005480);
  __cxa_atexit(sub_984900, &qword_5005480, &qword_4A427C0);
  qword_50053A0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50053F0 = 0x100000000LL;
  dword_50053AC &= 0x8000u;
  word_50053B0 = 0;
  qword_50053E8 = (__int64)&unk_50053F8;
  qword_50053B8 = 0;
  dword_50053A8 = v11;
  qword_50053C0 = 0;
  qword_50053C8 = 0;
  qword_50053D0 = 0;
  qword_50053D8 = 0;
  qword_50053E0 = 0;
  qword_5005400 = 0;
  qword_5005408 = (__int64)&unk_5005420;
  qword_5005410 = 1;
  dword_5005418 = 0;
  byte_500541C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_50053F0;
  v14 = (unsigned int)qword_50053F0 + 1LL;
  if ( v14 > HIDWORD(qword_50053F0) )
  {
    sub_C8D5F0((char *)&unk_50053F8 - 16, &unk_50053F8, v14, 8);
    v13 = (unsigned int)qword_50053F0;
  }
  *(_QWORD *)(qword_50053E8 + 8 * v13) = v12;
  qword_5005430 = (__int64)&unk_49D9748;
  qword_50053A0 = (__int64)&unk_49DC090;
  qword_5005440 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50053F0) = qword_50053F0 + 1;
  qword_5005460 = (__int64)nullsub_23;
  qword_5005428 = 0;
  qword_5005458 = (__int64)sub_984030;
  qword_5005438 = 0;
  sub_C53080(&qword_50053A0, "sroa-skip-mem2reg", 17);
  LOBYTE(qword_5005428) = 0;
  LOWORD(qword_5005438) = 256;
  LOBYTE(dword_50053AC) = dword_50053AC & 0x9F | 0x20;
  sub_C53130(&qword_50053A0);
  return __cxa_atexit(sub_984900, &qword_50053A0, &qword_4A427C0);
}
