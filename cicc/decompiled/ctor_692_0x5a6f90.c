// Function: ctor_692
// Address: 0x5a6f90
//
int __fastcall ctor_692(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5040840 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_50408BC = 1;
  qword_5040890 = 0x100000000LL;
  dword_504084C &= 0x8000u;
  qword_5040858 = 0;
  qword_5040860 = 0;
  qword_5040868 = 0;
  dword_5040848 = v4;
  word_5040850 = 0;
  qword_5040870 = 0;
  qword_5040878 = 0;
  qword_5040880 = 0;
  qword_5040888 = (__int64)&unk_5040898;
  qword_50408A0 = 0;
  qword_50408A8 = (__int64)&unk_50408C0;
  qword_50408B0 = 1;
  dword_50408B8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040890;
  v7 = (unsigned int)qword_5040890 + 1LL;
  if ( v7 > HIDWORD(qword_5040890) )
  {
    sub_C8D5F0((char *)&unk_5040898 - 16, &unk_5040898, v7, 8);
    v6 = (unsigned int)qword_5040890;
  }
  *(_QWORD *)(qword_5040888 + 8 * v6) = v5;
  LODWORD(qword_5040890) = qword_5040890 + 1;
  qword_50408C8 = 0;
  qword_50408D0 = (__int64)&unk_49D9748;
  qword_50408D8 = 0;
  qword_5040840 = (__int64)&unk_49DC090;
  qword_50408E0 = (__int64)&unk_49DC1D0;
  qword_5040900 = (__int64)nullsub_23;
  qword_50408F8 = (__int64)sub_984030;
  sub_C53080(&qword_5040840, "nvptx-use-max-local-array-alignment", 35);
  LOBYTE(qword_50408C8) = 0;
  LOWORD(qword_50408D8) = 256;
  qword_5040870 = 38;
  LOBYTE(dword_504084C) = dword_504084C & 0x9F | 0x20;
  qword_5040868 = (__int64)"Use maximum alignment for local memory";
  sub_C53130(&qword_5040840);
  return __cxa_atexit(sub_984900, &qword_5040840, &qword_4A427C0);
}
