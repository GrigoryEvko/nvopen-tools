// Function: ctor_071
// Address: 0x498f60
//
int ctor_071()
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
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx

  qword_4F8C060 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8C0B0 = 0x100000000LL;
  dword_4F8C06C &= 0x8000u;
  word_4F8C070 = 0;
  qword_4F8C078 = 0;
  qword_4F8C080 = 0;
  dword_4F8C068 = v0;
  qword_4F8C088 = 0;
  qword_4F8C090 = 0;
  qword_4F8C098 = 0;
  qword_4F8C0A0 = 0;
  qword_4F8C0A8 = (__int64)&unk_4F8C0B8;
  qword_4F8C0C0 = 0;
  qword_4F8C0C8 = (__int64)&unk_4F8C0E0;
  qword_4F8C0D0 = 1;
  dword_4F8C0D8 = 0;
  byte_4F8C0DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8C0B0;
  v3 = (unsigned int)qword_4F8C0B0 + 1LL;
  if ( v3 > HIDWORD(qword_4F8C0B0) )
  {
    sub_C8D5F0((char *)&unk_4F8C0B8 - 16, &unk_4F8C0B8, v3, 8);
    v2 = (unsigned int)qword_4F8C0B0;
  }
  *(_QWORD *)(qword_4F8C0A8 + 8 * v2) = v1;
  LODWORD(qword_4F8C0B0) = qword_4F8C0B0 + 1;
  qword_4F8C0E8 = 0;
  qword_4F8C0F0 = (__int64)&unk_49D9748;
  qword_4F8C0F8 = 0;
  qword_4F8C060 = (__int64)&unk_49DC090;
  qword_4F8C100 = (__int64)&unk_49DC1D0;
  qword_4F8C120 = (__int64)nullsub_23;
  qword_4F8C118 = (__int64)sub_984030;
  sub_C53080(&qword_4F8C060, "phicse-debug-hash", 17);
  LOWORD(qword_4F8C0F8) = 256;
  LOBYTE(qword_4F8C0E8) = 0;
  qword_4F8C090 = 117;
  LOBYTE(dword_4F8C06C) = dword_4F8C06C & 0x9F | 0x20;
  qword_4F8C088 = (__int64)"Perform extra assertion checking to verify that PHINodes's hash function is well-behaved w.r."
                           "t. its isEqual predicate";
  sub_C53130(&qword_4F8C060);
  __cxa_atexit(sub_984900, &qword_4F8C060, &qword_4A427C0);
  qword_4F8BF80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8BFD0 = 0x100000000LL;
  word_4F8BF90 = 0;
  dword_4F8BF8C &= 0x8000u;
  qword_4F8BF98 = 0;
  qword_4F8BFA0 = 0;
  dword_4F8BF88 = v4;
  qword_4F8BFA8 = 0;
  qword_4F8BFB0 = 0;
  qword_4F8BFB8 = 0;
  qword_4F8BFC0 = 0;
  qword_4F8BFC8 = (__int64)&unk_4F8BFD8;
  qword_4F8BFE0 = 0;
  qword_4F8BFE8 = (__int64)&unk_4F8C000;
  qword_4F8BFF0 = 1;
  dword_4F8BFF8 = 0;
  byte_4F8BFFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8BFD0;
  v7 = (unsigned int)qword_4F8BFD0 + 1LL;
  if ( v7 > HIDWORD(qword_4F8BFD0) )
  {
    sub_C8D5F0((char *)&unk_4F8BFD8 - 16, &unk_4F8BFD8, v7, 8);
    v6 = (unsigned int)qword_4F8BFD0;
  }
  *(_QWORD *)(qword_4F8BFC8 + 8 * v6) = v5;
  qword_4F8C010 = (__int64)&unk_49D9728;
  qword_4F8BF80 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F8BFD0) = qword_4F8BFD0 + 1;
  qword_4F8C008 = 0;
  qword_4F8C020 = (__int64)&unk_49DC290;
  qword_4F8C018 = 0;
  qword_4F8C040 = (__int64)nullsub_24;
  qword_4F8C038 = (__int64)sub_984050;
  sub_C53080(&qword_4F8BF80, "phicse-num-phi-smallsize", 24);
  LODWORD(qword_4F8C008) = 32;
  BYTE4(qword_4F8C018) = 1;
  LODWORD(qword_4F8C018) = 32;
  qword_4F8BFB0 = 134;
  LOBYTE(dword_4F8BF8C) = dword_4F8BF8C & 0x9F | 0x20;
  qword_4F8BFA8 = (__int64)"When the basic block contains not more than this number of PHI nodes, perform a (faster!) exh"
                           "austive search instead of set-driven one.";
  sub_C53130(&qword_4F8BF80);
  __cxa_atexit(sub_984970, &qword_4F8BF80, &qword_4A427C0);
  qword_4F8BEA0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8BEAC &= 0x8000u;
  word_4F8BEB0 = 0;
  qword_4F8BEF0 = 0x100000000LL;
  qword_4F8BEB8 = 0;
  qword_4F8BEC0 = 0;
  qword_4F8BEC8 = 0;
  dword_4F8BEA8 = v8;
  qword_4F8BED0 = 0;
  qword_4F8BED8 = 0;
  qword_4F8BEE0 = 0;
  qword_4F8BEE8 = (__int64)&unk_4F8BEF8;
  qword_4F8BF00 = 0;
  qword_4F8BF08 = (__int64)&unk_4F8BF20;
  qword_4F8BF10 = 1;
  dword_4F8BF18 = 0;
  byte_4F8BF1C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4F8BEF0;
  v11 = (unsigned int)qword_4F8BEF0 + 1LL;
  if ( v11 > HIDWORD(qword_4F8BEF0) )
  {
    sub_C8D5F0((char *)&unk_4F8BEF8 - 16, &unk_4F8BEF8, v11, 8);
    v10 = (unsigned int)qword_4F8BEF0;
  }
  *(_QWORD *)(qword_4F8BEE8 + 8 * v10) = v9;
  qword_4F8BF30 = (__int64)&unk_49D9728;
  qword_4F8BEA0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F8BEF0) = qword_4F8BEF0 + 1;
  qword_4F8BF28 = 0;
  qword_4F8BF40 = (__int64)&unk_49DC290;
  qword_4F8BF38 = 0;
  qword_4F8BF60 = (__int64)nullsub_24;
  qword_4F8BF58 = (__int64)sub_984050;
  sub_C53080(&qword_4F8BEA0, "max-phi-entries-increase-after-removing-empty-block", 51);
  LODWORD(qword_4F8BF28) = 1000;
  BYTE4(qword_4F8BF38) = 1;
  LODWORD(qword_4F8BF38) = 1000;
  qword_4F8BED0 = 112;
  LOBYTE(dword_4F8BEAC) = dword_4F8BEAC & 0x9F | 0x20;
  qword_4F8BEC8 = (__int64)"Stop removing an empty block if removing it will introduce more than this number of phi entri"
                           "es in its successor";
  sub_C53130(&qword_4F8BEA0);
  return __cxa_atexit(sub_984970, &qword_4F8BEA0, &qword_4A427C0);
}
