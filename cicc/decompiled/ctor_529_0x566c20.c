// Function: ctor_529
// Address: 0x566c20
//
int ctor_529()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_5013920 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013970 = 0x100000000LL;
  dword_501392C &= 0x8000u;
  word_5013930 = 0;
  qword_5013938 = 0;
  qword_5013940 = 0;
  dword_5013928 = v0;
  qword_5013948 = 0;
  qword_5013950 = 0;
  qword_5013958 = 0;
  qword_5013960 = 0;
  qword_5013968 = (__int64)&unk_5013978;
  qword_5013980 = 0;
  qword_5013988 = (__int64)&unk_50139A0;
  qword_5013990 = 1;
  dword_5013998 = 0;
  byte_501399C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5013970;
  v3 = (unsigned int)qword_5013970 + 1LL;
  if ( v3 > HIDWORD(qword_5013970) )
  {
    sub_C8D5F0((char *)&unk_5013978 - 16, &unk_5013978, v3, 8);
    v2 = (unsigned int)qword_5013970;
  }
  *(_QWORD *)(qword_5013968 + 8 * v2) = v1;
  qword_50139B0 = (__int64)&unk_49D9748;
  qword_5013920 = (__int64)&unk_49DC090;
  LODWORD(qword_5013970) = qword_5013970 + 1;
  qword_50139A8 = 0;
  qword_50139C0 = (__int64)&unk_49DC1D0;
  qword_50139B8 = 0;
  qword_50139E0 = (__int64)nullsub_23;
  qword_50139D8 = (__int64)sub_984030;
  sub_C53080(&qword_5013920, "devicefn-param-always-local", 27);
  LOBYTE(qword_50139A8) = 1;
  LOWORD(qword_50139B8) = 257;
  qword_5013950 = 56;
  LOBYTE(dword_501392C) = dword_501392C & 0x9F | 0x20;
  qword_5013948 = (__int64)"Treat Paramater space as local space in Device functions";
  sub_C53130(&qword_5013920);
  __cxa_atexit(sub_984900, &qword_5013920, &qword_4A427C0);
  qword_5013840 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013890 = 0x100000000LL;
  word_5013850 = 0;
  dword_501384C &= 0x8000u;
  qword_5013858 = 0;
  qword_5013860 = 0;
  dword_5013848 = v4;
  qword_5013868 = 0;
  qword_5013870 = 0;
  qword_5013878 = 0;
  qword_5013880 = 0;
  qword_5013888 = (__int64)&unk_5013898;
  qword_50138A0 = 0;
  qword_50138A8 = (__int64)&unk_50138C0;
  qword_50138B0 = 1;
  dword_50138B8 = 0;
  byte_50138BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5013890;
  if ( (unsigned __int64)(unsigned int)qword_5013890 + 1 > HIDWORD(qword_5013890) )
  {
    v19 = v5;
    sub_C8D5F0((char *)&unk_5013898 - 16, &unk_5013898, (unsigned int)qword_5013890 + 1LL, 8);
    v6 = (unsigned int)qword_5013890;
    v5 = v19;
  }
  *(_QWORD *)(qword_5013888 + 8 * v6) = v5;
  qword_50138D0 = (__int64)&unk_49D9748;
  qword_5013840 = (__int64)&unk_49DC090;
  LODWORD(qword_5013890) = qword_5013890 + 1;
  qword_50138C8 = 0;
  qword_50138E0 = (__int64)&unk_49DC1D0;
  qword_50138D8 = 0;
  qword_5013900 = (__int64)nullsub_23;
  qword_50138F8 = (__int64)sub_984030;
  sub_C53080(&qword_5013840, "skiploweraggcopysafechk", 23);
  LOWORD(qword_50138D8) = 256;
  LOBYTE(qword_50138C8) = 0;
  qword_5013870 = 37;
  LOBYTE(dword_501384C) = dword_501384C & 0x9F | 0x20;
  qword_5013868 = (__int64)"Skip the safety check in loweraggcopy";
  sub_C53130(&qword_5013840);
  __cxa_atexit(sub_984900, &qword_5013840, &qword_4A427C0);
  qword_5013760 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50137B0 = 0x100000000LL;
  dword_501376C &= 0x8000u;
  word_5013770 = 0;
  qword_5013778 = 0;
  qword_5013780 = 0;
  dword_5013768 = v7;
  qword_5013788 = 0;
  qword_5013790 = 0;
  qword_5013798 = 0;
  qword_50137A0 = 0;
  qword_50137A8 = (__int64)&unk_50137B8;
  qword_50137C0 = 0;
  qword_50137C8 = (__int64)&unk_50137E0;
  qword_50137D0 = 1;
  dword_50137D8 = 0;
  byte_50137DC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_50137B0;
  v10 = (unsigned int)qword_50137B0 + 1LL;
  if ( v10 > HIDWORD(qword_50137B0) )
  {
    sub_C8D5F0((char *)&unk_50137B8 - 16, &unk_50137B8, v10, 8);
    v9 = (unsigned int)qword_50137B0;
  }
  *(_QWORD *)(qword_50137A8 + 8 * v9) = v8;
  qword_50137F0 = (__int64)&unk_49D9728;
  qword_5013760 = (__int64)&unk_49DBF10;
  qword_5013800 = (__int64)&unk_49DC290;
  LODWORD(qword_50137B0) = qword_50137B0 + 1;
  qword_5013820 = (__int64)nullsub_24;
  qword_50137E8 = 0;
  qword_5013818 = (__int64)sub_984050;
  qword_50137F8 = 0;
  sub_C53080(&qword_5013760, "large-aggr-store-limit", 22);
  LODWORD(qword_50137E8) = 10000;
  BYTE4(qword_50137F8) = 1;
  LODWORD(qword_50137F8) = 10000;
  qword_5013790 = 61;
  LOBYTE(dword_501376C) = dword_501376C & 0x9F | 0x20;
  qword_5013788 = (__int64)"Try to create loops for store of aggregate greater than limit";
  sub_C53130(&qword_5013760);
  __cxa_atexit(sub_984970, &qword_5013760, &qword_4A427C0);
  qword_5013680 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50136FC = 1;
  qword_50136D0 = 0x100000000LL;
  dword_501368C &= 0x8000u;
  qword_50136C8 = (__int64)&unk_50136D8;
  qword_5013698 = 0;
  qword_50136A0 = 0;
  dword_5013688 = v11;
  word_5013690 = 0;
  qword_50136A8 = 0;
  qword_50136B0 = 0;
  qword_50136B8 = 0;
  qword_50136C0 = 0;
  qword_50136E0 = 0;
  qword_50136E8 = (__int64)&unk_5013700;
  qword_50136F0 = 1;
  dword_50136F8 = 0;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_50136D0;
  if ( (unsigned __int64)(unsigned int)qword_50136D0 + 1 > HIDWORD(qword_50136D0) )
  {
    v20 = v12;
    sub_C8D5F0((char *)&unk_50136D8 - 16, &unk_50136D8, (unsigned int)qword_50136D0 + 1LL, 8);
    v13 = (unsigned int)qword_50136D0;
    v12 = v20;
  }
  *(_QWORD *)(qword_50136C8 + 8 * v13) = v12;
  qword_5013710 = (__int64)&unk_49D9728;
  qword_5013680 = (__int64)&unk_49DBF10;
  qword_5013720 = (__int64)&unk_49DC290;
  LODWORD(qword_50136D0) = qword_50136D0 + 1;
  qword_5013740 = (__int64)nullsub_24;
  qword_5013708 = 0;
  qword_5013738 = (__int64)sub_984050;
  qword_5013718 = 0;
  sub_C53080(&qword_5013680, "max-aggr-copy-size", 18);
  LODWORD(qword_5013708) = 128;
  BYTE4(qword_5013718) = 1;
  LODWORD(qword_5013718) = 128;
  qword_50136B0 = 52;
  LOBYTE(dword_501368C) = dword_501368C & 0x9F | 0x20;
  qword_50136A8 = (__int64)"Create loops for copying aggregate greater than size";
  sub_C53130(&qword_5013680);
  __cxa_atexit(sub_984970, &qword_5013680, &qword_4A427C0);
  qword_50135A0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50135AC &= 0x8000u;
  word_50135B0 = 0;
  qword_50135F0 = 0x100000000LL;
  qword_50135E8 = (__int64)&unk_50135F8;
  qword_50135B8 = 0;
  qword_50135C0 = 0;
  dword_50135A8 = v14;
  qword_50135C8 = 0;
  qword_50135D0 = 0;
  qword_50135D8 = 0;
  qword_50135E0 = 0;
  qword_5013600 = 0;
  qword_5013608 = (__int64)&unk_5013620;
  qword_5013610 = 1;
  dword_5013618 = 0;
  byte_501361C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_50135F0;
  v17 = (unsigned int)qword_50135F0 + 1LL;
  if ( v17 > HIDWORD(qword_50135F0) )
  {
    sub_C8D5F0((char *)&unk_50135F8 - 16, &unk_50135F8, v17, 8);
    v16 = (unsigned int)qword_50135F0;
  }
  *(_QWORD *)(qword_50135E8 + 8 * v16) = v15;
  qword_5013630 = (__int64)&unk_49D9728;
  qword_50135A0 = (__int64)&unk_49DBF10;
  qword_5013640 = (__int64)&unk_49DC290;
  LODWORD(qword_50135F0) = qword_50135F0 + 1;
  qword_5013660 = (__int64)nullsub_24;
  qword_5013628 = 0;
  qword_5013658 = (__int64)sub_984050;
  qword_5013638 = 0;
  sub_C53080(&qword_50135A0, "lower-aggr-unrolled-stores-limit", 32);
  qword_50135D0 = 46;
  LODWORD(qword_5013628) = 16;
  BYTE4(qword_5013638) = 1;
  LODWORD(qword_5013638) = 16;
  LOBYTE(dword_50135AC) = dword_50135AC & 0x9F | 0x20;
  qword_50135C8 = (__int64)"Limit no. of stores generated in unrolled mode";
  sub_C53130(&qword_50135A0);
  return __cxa_atexit(sub_984970, &qword_50135A0, &qword_4A427C0);
}
