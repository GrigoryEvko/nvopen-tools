// Function: ctor_555
// Address: 0x570b40
//
int ctor_555()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501E760 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501E7DC = 1;
  qword_501E7B0 = 0x100000000LL;
  dword_501E76C &= 0x8000u;
  qword_501E778 = 0;
  qword_501E780 = 0;
  qword_501E788 = 0;
  dword_501E768 = v0;
  word_501E770 = 0;
  qword_501E790 = 0;
  qword_501E798 = 0;
  qword_501E7A0 = 0;
  qword_501E7A8 = (__int64)&unk_501E7B8;
  qword_501E7C0 = 0;
  qword_501E7C8 = (__int64)&unk_501E7E0;
  qword_501E7D0 = 1;
  dword_501E7D8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501E7B0;
  v3 = (unsigned int)qword_501E7B0 + 1LL;
  if ( v3 > HIDWORD(qword_501E7B0) )
  {
    sub_C8D5F0((char *)&unk_501E7B8 - 16, &unk_501E7B8, v3, 8);
    v2 = (unsigned int)qword_501E7B0;
  }
  *(_QWORD *)(qword_501E7A8 + 8 * v2) = v1;
  LODWORD(qword_501E7B0) = qword_501E7B0 + 1;
  qword_501E7E8 = 0;
  qword_501E7F0 = (__int64)&unk_49D9748;
  qword_501E7F8 = 0;
  qword_501E760 = (__int64)&unk_49DC090;
  qword_501E800 = (__int64)&unk_49DC1D0;
  qword_501E820 = (__int64)nullsub_23;
  qword_501E818 = (__int64)sub_984030;
  sub_C53080(&qword_501E760, "lower-interleaved-accesses", 26);
  qword_501E790 = 50;
  qword_501E788 = (__int64)"Enable lowering interleaved accesses to intrinsics";
  LOWORD(qword_501E7F8) = 257;
  LOBYTE(qword_501E7E8) = 1;
  LOBYTE(dword_501E76C) = dword_501E76C & 0x9F | 0x20;
  sub_C53130(&qword_501E760);
  return __cxa_atexit(sub_984900, &qword_501E760, &qword_4A427C0);
}
