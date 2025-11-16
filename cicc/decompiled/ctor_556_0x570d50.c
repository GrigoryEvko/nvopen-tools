// Function: ctor_556
// Address: 0x570d50
//
int ctor_556()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501E840 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501E8BC = 1;
  qword_501E890 = 0x100000000LL;
  dword_501E84C &= 0x8000u;
  qword_501E858 = 0;
  qword_501E860 = 0;
  qword_501E868 = 0;
  dword_501E848 = v0;
  word_501E850 = 0;
  qword_501E870 = 0;
  qword_501E878 = 0;
  qword_501E880 = 0;
  qword_501E888 = (__int64)&unk_501E898;
  qword_501E8A0 = 0;
  qword_501E8A8 = (__int64)&unk_501E8C0;
  qword_501E8B0 = 1;
  dword_501E8B8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501E890;
  v3 = (unsigned int)qword_501E890 + 1LL;
  if ( v3 > HIDWORD(qword_501E890) )
  {
    sub_C8D5F0((char *)&unk_501E898 - 16, &unk_501E898, v3, 8);
    v2 = (unsigned int)qword_501E890;
  }
  *(_QWORD *)(qword_501E888 + 8 * v2) = v1;
  LODWORD(qword_501E890) = qword_501E890 + 1;
  qword_501E8C8 = 0;
  qword_501E8D0 = (__int64)&unk_49D9748;
  qword_501E8D8 = 0;
  qword_501E840 = (__int64)&unk_49DC090;
  qword_501E8E0 = (__int64)&unk_49DC1D0;
  qword_501E900 = (__int64)nullsub_23;
  qword_501E8F8 = (__int64)sub_984030;
  sub_C53080(&qword_501E840, "disable-interleaved-load-combine", 32);
  LOBYTE(qword_501E8C8) = 0;
  LOWORD(qword_501E8D8) = 256;
  qword_501E870 = 38;
  LOBYTE(dword_501E84C) = dword_501E84C & 0x9F | 0x20;
  qword_501E868 = (__int64)"Disable combining of interleaved loads";
  sub_C53130(&qword_501E840);
  return __cxa_atexit(sub_984900, &qword_501E840, &qword_4A427C0);
}
