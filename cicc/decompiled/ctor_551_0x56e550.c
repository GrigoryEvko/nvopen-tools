// Function: ctor_551
// Address: 0x56e550
//
int ctor_551()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_501D940 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501D990 = 0x100000000LL;
  dword_501D94C &= 0x8000u;
  word_501D950 = 0;
  qword_501D958 = 0;
  qword_501D960 = 0;
  dword_501D948 = v0;
  qword_501D968 = 0;
  qword_501D970 = 0;
  qword_501D978 = 0;
  qword_501D980 = 0;
  qword_501D988 = (__int64)&unk_501D998;
  qword_501D9A0 = 0;
  qword_501D9A8 = (__int64)&unk_501D9C0;
  qword_501D9B0 = 1;
  dword_501D9B8 = 0;
  byte_501D9BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501D990;
  v3 = (unsigned int)qword_501D990 + 1LL;
  if ( v3 > HIDWORD(qword_501D990) )
  {
    sub_C8D5F0((char *)&unk_501D998 - 16, &unk_501D998, v3, 8);
    v2 = (unsigned int)qword_501D990;
  }
  *(_QWORD *)(qword_501D988 + 8 * v2) = v1;
  qword_501D9D0 = (__int64)&unk_49D9748;
  qword_501D940 = (__int64)&unk_49DC090;
  qword_501D9E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501D990) = qword_501D990 + 1;
  qword_501DA00 = (__int64)nullsub_23;
  qword_501D9C8 = 0;
  qword_501D9F8 = (__int64)sub_984030;
  qword_501D9D8 = 0;
  sub_C53080(&qword_501D940, "fixup-scs-extend-slot-size", 26);
  LOBYTE(qword_501D9C8) = 0;
  LOWORD(qword_501D9D8) = 256;
  qword_501D970 = 60;
  LOBYTE(dword_501D94C) = dword_501D94C & 0x9F | 0x20;
  qword_501D968 = (__int64)"Allow spill in spill slot of greater size than register size";
  sub_C53130(&qword_501D940);
  __cxa_atexit(sub_984900, &qword_501D940, &qword_4A427C0);
  qword_501D860 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501D8B0 = 0x100000000LL;
  dword_501D86C &= 0x8000u;
  word_501D870 = 0;
  qword_501D8A8 = (__int64)&unk_501D8B8;
  qword_501D878 = 0;
  dword_501D868 = v4;
  qword_501D880 = 0;
  qword_501D888 = 0;
  qword_501D890 = 0;
  qword_501D898 = 0;
  qword_501D8A0 = 0;
  qword_501D8C0 = 0;
  qword_501D8C8 = (__int64)&unk_501D8E0;
  qword_501D8D0 = 1;
  dword_501D8D8 = 0;
  byte_501D8DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_501D8B0;
  if ( (unsigned __int64)(unsigned int)qword_501D8B0 + 1 > HIDWORD(qword_501D8B0) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_501D8B8 - 16, &unk_501D8B8, (unsigned int)qword_501D8B0 + 1LL, 8);
    v6 = (unsigned int)qword_501D8B0;
    v5 = v15;
  }
  *(_QWORD *)(qword_501D8A8 + 8 * v6) = v5;
  qword_501D8F0 = (__int64)&unk_49D9748;
  qword_501D860 = (__int64)&unk_49DC090;
  qword_501D900 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501D8B0) = qword_501D8B0 + 1;
  qword_501D920 = (__int64)nullsub_23;
  qword_501D8E8 = 0;
  qword_501D918 = (__int64)sub_984030;
  qword_501D8F8 = 0;
  sub_C53080(&qword_501D860, "fixup-allow-gcptr-in-csr", 24);
  LOWORD(qword_501D8F8) = 256;
  LOBYTE(qword_501D8E8) = 0;
  qword_501D890 = 60;
  LOBYTE(dword_501D86C) = dword_501D86C & 0x9F | 0x20;
  qword_501D888 = (__int64)"Allow passing GC Pointer arguments in callee saved registers";
  sub_C53130(&qword_501D860);
  __cxa_atexit(sub_984900, &qword_501D860, &qword_4A427C0);
  qword_501D780 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501D7FC = 1;
  word_501D790 = 0;
  qword_501D7D0 = 0x100000000LL;
  dword_501D78C &= 0x8000u;
  qword_501D7C8 = (__int64)&unk_501D7D8;
  qword_501D798 = 0;
  dword_501D788 = v7;
  qword_501D7A0 = 0;
  qword_501D7A8 = 0;
  qword_501D7B0 = 0;
  qword_501D7B8 = 0;
  qword_501D7C0 = 0;
  qword_501D7E0 = 0;
  qword_501D7E8 = (__int64)&unk_501D800;
  qword_501D7F0 = 1;
  dword_501D7F8 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_501D7D0;
  if ( (unsigned __int64)(unsigned int)qword_501D7D0 + 1 > HIDWORD(qword_501D7D0) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_501D7D8 - 16, &unk_501D7D8, (unsigned int)qword_501D7D0 + 1LL, 8);
    v9 = (unsigned int)qword_501D7D0;
    v8 = v16;
  }
  *(_QWORD *)(qword_501D7C8 + 8 * v9) = v8;
  qword_501D810 = (__int64)&unk_49D9748;
  qword_501D780 = (__int64)&unk_49DC090;
  qword_501D820 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501D7D0) = qword_501D7D0 + 1;
  qword_501D840 = (__int64)nullsub_23;
  qword_501D808 = 0;
  qword_501D838 = (__int64)sub_984030;
  qword_501D818 = 0;
  sub_C53080(&qword_501D780, "fixup-scs-enable-copy-propagation", 33);
  LOBYTE(qword_501D808) = 1;
  qword_501D7B0 = 56;
  LOBYTE(dword_501D78C) = dword_501D78C & 0x9F | 0x20;
  LOWORD(qword_501D818) = 257;
  qword_501D7A8 = (__int64)"Enable simple copy propagation during register reloading";
  sub_C53130(&qword_501D780);
  __cxa_atexit(sub_984900, &qword_501D780, &qword_4A427C0);
  qword_501D6A0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501D71C = 1;
  qword_501D6F0 = 0x100000000LL;
  dword_501D6AC &= 0x8000u;
  qword_501D6B8 = 0;
  qword_501D6C0 = 0;
  qword_501D6C8 = 0;
  dword_501D6A8 = v10;
  word_501D6B0 = 0;
  qword_501D6D0 = 0;
  qword_501D6D8 = 0;
  qword_501D6E0 = 0;
  qword_501D6E8 = (__int64)&unk_501D6F8;
  qword_501D700 = 0;
  qword_501D708 = (__int64)&unk_501D720;
  qword_501D710 = 1;
  dword_501D718 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_501D6F0;
  v13 = (unsigned int)qword_501D6F0 + 1LL;
  if ( v13 > HIDWORD(qword_501D6F0) )
  {
    sub_C8D5F0((char *)&unk_501D6F8 - 16, &unk_501D6F8, v13, 8);
    v12 = (unsigned int)qword_501D6F0;
  }
  *(_QWORD *)(qword_501D6E8 + 8 * v12) = v11;
  LODWORD(qword_501D6F0) = qword_501D6F0 + 1;
  qword_501D728 = 0;
  qword_501D730 = (__int64)&unk_49D9728;
  qword_501D738 = 0;
  qword_501D6A0 = (__int64)&unk_49DBF10;
  qword_501D740 = (__int64)&unk_49DC290;
  qword_501D760 = (__int64)nullsub_24;
  qword_501D758 = (__int64)sub_984050;
  sub_C53080(&qword_501D6A0, "fixup-max-csr-statepoints", 25);
  qword_501D6D0 = 62;
  LOBYTE(dword_501D6AC) = dword_501D6AC & 0x9F | 0x20;
  qword_501D6C8 = (__int64)"Max number of statepoints allowed to pass GC Ptrs in registers";
  sub_C53130(&qword_501D6A0);
  return __cxa_atexit(sub_984970, &qword_501D6A0, &qword_4A427C0);
}
