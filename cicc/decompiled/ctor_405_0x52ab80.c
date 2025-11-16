// Function: ctor_405
// Address: 0x52ab80
//
int ctor_405()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FEA7E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEA7F8 = 0;
  qword_4FEA800 = 0;
  qword_4FEA808 = 0;
  qword_4FEA810 = 0;
  dword_4FEA7EC = dword_4FEA7EC & 0x8000 | 1;
  word_4FEA7F0 = 0;
  qword_4FEA830 = 0x100000000LL;
  dword_4FEA7E8 = v0;
  qword_4FEA818 = 0;
  qword_4FEA820 = 0;
  qword_4FEA828 = (__int64)&unk_4FEA838;
  qword_4FEA840 = 0;
  qword_4FEA848 = (__int64)&unk_4FEA860;
  qword_4FEA850 = 1;
  dword_4FEA858 = 0;
  byte_4FEA85C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEA830;
  v3 = (unsigned int)qword_4FEA830 + 1LL;
  if ( v3 > HIDWORD(qword_4FEA830) )
  {
    sub_C8D5F0((char *)&unk_4FEA838 - 16, &unk_4FEA838, v3, 8);
    v2 = (unsigned int)qword_4FEA830;
  }
  *(_QWORD *)(qword_4FEA828 + 8 * v2) = v1;
  LODWORD(qword_4FEA830) = qword_4FEA830 + 1;
  qword_4FEA868 = 0;
  qword_4FEA7E0 = (__int64)&unk_49DAD08;
  qword_4FEA870 = 0;
  qword_4FEA878 = 0;
  qword_4FEA8B8 = (__int64)&unk_49DC350;
  qword_4FEA880 = 0;
  qword_4FEA8D8 = (__int64)nullsub_81;
  qword_4FEA888 = 0;
  qword_4FEA8D0 = (__int64)sub_BB8600;
  qword_4FEA890 = 0;
  byte_4FEA898 = 0;
  qword_4FEA8A0 = 0;
  qword_4FEA8A8 = 0;
  qword_4FEA8B0 = 0;
  sub_C53080(&qword_4FEA7E0, "profile-context-root", 20);
  qword_4FEA810 = 161;
  LOBYTE(dword_4FEA7EC) = dword_4FEA7EC & 0x9F | 0x20;
  qword_4FEA808 = (__int64)"A function name, assumed to be global, which will be treated as the root of an interesting gr"
                           "aph, which will be profiled independently from other similar graphs.";
  sub_C53130(&qword_4FEA7E0);
  return __cxa_atexit(sub_BB89D0, &qword_4FEA7E0, &qword_4A427C0);
}
