// Function: ctor_502
// Address: 0x55a430
//
int ctor_502()
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

  qword_500A7C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500A810 = 0x100000000LL;
  dword_500A7CC &= 0x8000u;
  word_500A7D0 = 0;
  qword_500A7D8 = 0;
  qword_500A7E0 = 0;
  dword_500A7C8 = v0;
  qword_500A7E8 = 0;
  qword_500A7F0 = 0;
  qword_500A7F8 = 0;
  qword_500A800 = 0;
  qword_500A808 = (__int64)&unk_500A818;
  qword_500A820 = 0;
  qword_500A828 = (__int64)&unk_500A840;
  qword_500A830 = 1;
  dword_500A838 = 0;
  byte_500A83C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500A810;
  v3 = (unsigned int)qword_500A810 + 1LL;
  if ( v3 > HIDWORD(qword_500A810) )
  {
    sub_C8D5F0((char *)&unk_500A818 - 16, &unk_500A818, v3, 8);
    v2 = (unsigned int)qword_500A810;
  }
  *(_QWORD *)(qword_500A808 + 8 * v2) = v1;
  qword_500A850 = (__int64)&unk_49D9748;
  qword_500A7C0 = (__int64)&unk_49DC090;
  qword_500A860 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A810) = qword_500A810 + 1;
  qword_500A880 = (__int64)nullsub_23;
  qword_500A848 = 0;
  qword_500A878 = (__int64)sub_984030;
  qword_500A858 = 0;
  sub_C53080(&qword_500A7C0, "unroll-runtime-multi-exit", 25);
  LOWORD(qword_500A858) = 256;
  LOBYTE(qword_500A848) = 0;
  qword_500A7F0 = 79;
  LOBYTE(dword_500A7CC) = dword_500A7CC & 0x9F | 0x20;
  qword_500A7E8 = (__int64)"Allow runtime unrolling for loops with multiple exits, when epilog is generated";
  sub_C53130(&qword_500A7C0);
  __cxa_atexit(sub_984900, &qword_500A7C0, &qword_4A427C0);
  qword_500A6E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500A730 = 0x100000000LL;
  dword_500A6EC &= 0x8000u;
  qword_500A728 = (__int64)&unk_500A738;
  word_500A6F0 = 0;
  qword_500A6F8 = 0;
  dword_500A6E8 = v4;
  qword_500A700 = 0;
  qword_500A708 = 0;
  qword_500A710 = 0;
  qword_500A718 = 0;
  qword_500A720 = 0;
  qword_500A740 = 0;
  qword_500A748 = (__int64)&unk_500A760;
  qword_500A750 = 1;
  dword_500A758 = 0;
  byte_500A75C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500A730;
  if ( (unsigned __int64)(unsigned int)qword_500A730 + 1 > HIDWORD(qword_500A730) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_500A738 - 16, &unk_500A738, (unsigned int)qword_500A730 + 1LL, 8);
    v6 = (unsigned int)qword_500A730;
    v5 = v15;
  }
  *(_QWORD *)(qword_500A728 + 8 * v6) = v5;
  qword_500A770 = (__int64)&unk_49D9748;
  qword_500A6E0 = (__int64)&unk_49DC090;
  qword_500A780 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A730) = qword_500A730 + 1;
  qword_500A7A0 = (__int64)nullsub_23;
  qword_500A768 = 0;
  qword_500A798 = (__int64)sub_984030;
  qword_500A778 = 0;
  sub_C53080(&qword_500A6E0, "unroll-runtime-other-exit-predictable", 37);
  LOWORD(qword_500A778) = 256;
  LOBYTE(qword_500A768) = 0;
  qword_500A710 = 49;
  LOBYTE(dword_500A6EC) = dword_500A6EC & 0x9F | 0x20;
  qword_500A708 = (__int64)"Assume the non latch exit block to be predictable";
  sub_C53130(&qword_500A6E0);
  __cxa_atexit(sub_984900, &qword_500A6E0, &qword_4A427C0);
  qword_500A600 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500A650 = 0x100000000LL;
  dword_500A60C &= 0x8000u;
  qword_500A648 = (__int64)&unk_500A658;
  word_500A610 = 0;
  qword_500A618 = 0;
  dword_500A608 = v7;
  qword_500A620 = 0;
  qword_500A628 = 0;
  qword_500A630 = 0;
  qword_500A638 = 0;
  qword_500A640 = 0;
  qword_500A660 = 0;
  qword_500A668 = (__int64)&unk_500A680;
  qword_500A670 = 1;
  dword_500A678 = 0;
  byte_500A67C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_500A650;
  if ( (unsigned __int64)(unsigned int)qword_500A650 + 1 > HIDWORD(qword_500A650) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_500A658 - 16, &unk_500A658, (unsigned int)qword_500A650 + 1LL, 8);
    v9 = (unsigned int)qword_500A650;
    v8 = v16;
  }
  *(_QWORD *)(qword_500A648 + 8 * v9) = v8;
  qword_500A690 = (__int64)&unk_49D9748;
  qword_500A600 = (__int64)&unk_49DC090;
  qword_500A6A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A650) = qword_500A650 + 1;
  qword_500A6C0 = (__int64)nullsub_23;
  qword_500A688 = 0;
  qword_500A6B8 = (__int64)sub_984030;
  qword_500A698 = 0;
  sub_C53080(&qword_500A600, "waterfall-unrolling-remove-epilogue-backedge", 44);
  LOWORD(qword_500A698) = 257;
  LOBYTE(qword_500A688) = 1;
  qword_500A630 = 143;
  LOBYTE(dword_500A60C) = dword_500A60C & 0x9F | 0x20;
  qword_500A628 = (__int64)"Epilogues in waterfall unrolled loops can only execute 0-1 times. This would remove the backe"
                           "dge.Seems to increase register pressure sometimes.";
  sub_C53130(&qword_500A600);
  __cxa_atexit(sub_984900, &qword_500A600, &qword_4A427C0);
  qword_500A520 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500A570 = 0x100000000LL;
  dword_500A52C &= 0x8000u;
  word_500A530 = 0;
  qword_500A568 = (__int64)&unk_500A578;
  qword_500A538 = 0;
  dword_500A528 = v10;
  qword_500A540 = 0;
  qword_500A548 = 0;
  qword_500A550 = 0;
  qword_500A558 = 0;
  qword_500A560 = 0;
  qword_500A580 = 0;
  qword_500A588 = (__int64)&unk_500A5A0;
  qword_500A590 = 1;
  dword_500A598 = 0;
  byte_500A59C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_500A570;
  v13 = (unsigned int)qword_500A570 + 1LL;
  if ( v13 > HIDWORD(qword_500A570) )
  {
    sub_C8D5F0((char *)&unk_500A578 - 16, &unk_500A578, v13, 8);
    v12 = (unsigned int)qword_500A570;
  }
  *(_QWORD *)(qword_500A568 + 8 * v12) = v11;
  qword_500A5B0 = (__int64)&unk_49D9748;
  qword_500A520 = (__int64)&unk_49DC090;
  qword_500A5C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A570) = qword_500A570 + 1;
  qword_500A5E0 = (__int64)nullsub_23;
  qword_500A5A8 = 0;
  qword_500A5D8 = (__int64)sub_984030;
  qword_500A5B8 = 0;
  sub_C53080(&qword_500A520, "unroll-runtime-nv-expensive", 27);
  LOBYTE(qword_500A5A8) = 1;
  LOWORD(qword_500A5B8) = 257;
  qword_500A550 = 59;
  LOBYTE(dword_500A52C) = dword_500A52C & 0x9F | 0x20;
  qword_500A548 = (__int64)"Use NV heuristics for allowing unrolling of expensive loops";
  sub_C53130(&qword_500A520);
  return __cxa_atexit(sub_984900, &qword_500A520, &qword_4A427C0);
}
