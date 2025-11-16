// Function: ctor_533
// Address: 0x569360
//
int ctor_533()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5014800 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501487C = 1;
  qword_5014850 = 0x100000000LL;
  dword_501480C &= 0x8000u;
  qword_5014818 = 0;
  qword_5014820 = 0;
  qword_5014828 = 0;
  dword_5014808 = v0;
  word_5014810 = 0;
  qword_5014830 = 0;
  qword_5014838 = 0;
  qword_5014840 = 0;
  qword_5014848 = (__int64)&unk_5014858;
  qword_5014860 = 0;
  qword_5014868 = (__int64)&unk_5014880;
  qword_5014870 = 1;
  dword_5014878 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5014850;
  v3 = (unsigned int)qword_5014850 + 1LL;
  if ( v3 > HIDWORD(qword_5014850) )
  {
    sub_C8D5F0((char *)&unk_5014858 - 16, &unk_5014858, v3, 8);
    v2 = (unsigned int)qword_5014850;
  }
  *(_QWORD *)(qword_5014848 + 8 * v2) = v1;
  LODWORD(qword_5014850) = qword_5014850 + 1;
  qword_5014888 = 0;
  qword_5014890 = (__int64)&unk_49D9748;
  qword_5014898 = 0;
  qword_5014800 = (__int64)&unk_49DC090;
  qword_50148A0 = (__int64)&unk_49DC1D0;
  qword_50148C0 = (__int64)nullsub_23;
  qword_50148B8 = (__int64)sub_984030;
  sub_C53080(&qword_5014800, "nvvm-lower-printf", 17);
  LOBYTE(qword_5014888) = 1;
  LOWORD(qword_5014898) = 257;
  qword_5014830 = 43;
  LOBYTE(dword_501480C) = dword_501480C & 0x9F | 0x20;
  qword_5014828 = (__int64)"Enable printf lowering (enabled by default)";
  sub_C53130(&qword_5014800);
  return __cxa_atexit(sub_984900, &qword_5014800, &qword_4A427C0);
}
