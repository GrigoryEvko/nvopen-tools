// Function: ctor_015
// Address: 0x453fb0
//
int ctor_015()
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
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4F80840 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F80890 = 0x100000000LL;
  dword_4F8084C &= 0x8000u;
  word_4F80850 = 0;
  qword_4F80858 = 0;
  qword_4F80860 = 0;
  dword_4F80848 = v0;
  qword_4F80868 = 0;
  qword_4F80870 = 0;
  qword_4F80878 = 0;
  qword_4F80880 = 0;
  qword_4F80888 = (__int64)&unk_4F80898;
  qword_4F808A0 = 0;
  qword_4F808A8 = (__int64)&unk_4F808C0;
  qword_4F808B0 = 1;
  dword_4F808B8 = 0;
  byte_4F808BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F80890;
  v3 = (unsigned int)qword_4F80890 + 1LL;
  if ( v3 > HIDWORD(qword_4F80890) )
  {
    sub_C8D5F0((char *)&unk_4F80898 - 16, &unk_4F80898, v3, 8);
    v2 = (unsigned int)qword_4F80890;
  }
  *(_QWORD *)(qword_4F80888 + 8 * v2) = v1;
  qword_4F808D0 = (__int64)&unk_49D9728;
  qword_4F80840 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F80890) = qword_4F80890 + 1;
  qword_4F808C8 = 0;
  qword_4F808E0 = (__int64)&unk_49DC290;
  qword_4F808D8 = 0;
  qword_4F80900 = (__int64)nullsub_24;
  qword_4F808F8 = (__int64)sub_984050;
  sub_C53080(&qword_4F80840, "bitcode-mdindex-threshold", 25);
  LODWORD(qword_4F808C8) = 25;
  BYTE4(qword_4F808D8) = 1;
  LODWORD(qword_4F808D8) = 25;
  qword_4F80870 = 71;
  LOBYTE(dword_4F8084C) = dword_4F8084C & 0x9F | 0x20;
  qword_4F80868 = (__int64)"Number of metadatas above which we emit an index to enable lazy-loading";
  sub_C53130(&qword_4F80840);
  __cxa_atexit(sub_984970, &qword_4F80840, &qword_4A427C0);
  qword_4F80760 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F807B0 = 0x100000000LL;
  word_4F80770 = 0;
  dword_4F8076C &= 0x8000u;
  qword_4F80778 = 0;
  qword_4F80780 = 0;
  dword_4F80768 = v4;
  qword_4F80788 = 0;
  qword_4F80790 = 0;
  qword_4F80798 = 0;
  qword_4F807A0 = 0;
  qword_4F807A8 = (__int64)&unk_4F807B8;
  qword_4F807C0 = 0;
  qword_4F807C8 = (__int64)&unk_4F807E0;
  qword_4F807D0 = 1;
  dword_4F807D8 = 0;
  byte_4F807DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F807B0;
  if ( (unsigned __int64)(unsigned int)qword_4F807B0 + 1 > HIDWORD(qword_4F807B0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4F807B8 - 16, &unk_4F807B8, (unsigned int)qword_4F807B0 + 1LL, 8);
    v6 = (unsigned int)qword_4F807B0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4F807A8 + 8 * v6) = v5;
  qword_4F807F0 = (__int64)&unk_49D9728;
  qword_4F80760 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F807B0) = qword_4F807B0 + 1;
  qword_4F807E8 = 0;
  qword_4F80800 = (__int64)&unk_49DC290;
  qword_4F807F8 = 0;
  qword_4F80820 = (__int64)nullsub_24;
  qword_4F80818 = (__int64)sub_984050;
  sub_C53080(&qword_4F80760, "bitcode-flush-threshold", 23);
  LODWORD(qword_4F807E8) = 512;
  BYTE4(qword_4F807F8) = 1;
  LODWORD(qword_4F807F8) = 512;
  qword_4F80790 = 49;
  LOBYTE(dword_4F8076C) = dword_4F8076C & 0x9F | 0x20;
  qword_4F80788 = (__int64)"The threshold (unit M) for flushing LLVM bitcode.";
  sub_C53130(&qword_4F80760);
  __cxa_atexit(sub_984970, &qword_4F80760, &qword_4A427C0);
  qword_4F80680 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F806FC = 1;
  qword_4F806D0 = 0x100000000LL;
  dword_4F8068C &= 0x8000u;
  qword_4F80698 = 0;
  qword_4F806A0 = 0;
  qword_4F806A8 = 0;
  dword_4F80688 = v7;
  word_4F80690 = 0;
  qword_4F806B0 = 0;
  qword_4F806B8 = 0;
  qword_4F806C0 = 0;
  qword_4F806C8 = (__int64)&unk_4F806D8;
  qword_4F806E0 = 0;
  qword_4F806E8 = (__int64)&unk_4F80700;
  qword_4F806F0 = 1;
  dword_4F806F8 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F806D0;
  v10 = (unsigned int)qword_4F806D0 + 1LL;
  if ( v10 > HIDWORD(qword_4F806D0) )
  {
    sub_C8D5F0((char *)&unk_4F806D8 - 16, &unk_4F806D8, v10, 8);
    v9 = (unsigned int)qword_4F806D0;
  }
  *(_QWORD *)(qword_4F806C8 + 8 * v9) = v8;
  LODWORD(qword_4F806D0) = qword_4F806D0 + 1;
  qword_4F80708 = 0;
  qword_4F80710 = (__int64)&unk_49D9748;
  qword_4F80718 = 0;
  qword_4F80680 = (__int64)&unk_49DC090;
  qword_4F80720 = (__int64)&unk_49DC1D0;
  qword_4F80740 = (__int64)nullsub_23;
  qword_4F80738 = (__int64)sub_984030;
  sub_C53080(&qword_4F80680, "write-relbf-to-summary", 22);
  LOBYTE(qword_4F80708) = 0;
  qword_4F806B0 = 51;
  LOBYTE(dword_4F8068C) = dword_4F8068C & 0x9F | 0x20;
  LOWORD(qword_4F80718) = 256;
  qword_4F806A8 = (__int64)"Write relative block frequency to function summary ";
  sub_C53130(&qword_4F80680);
  return __cxa_atexit(sub_984900, &qword_4F80680, &qword_4A427C0);
}
