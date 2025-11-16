// Function: ctor_036
// Address: 0x48cc90
//
char *ctor_036()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  char *result; // rax

  qword_4F83800 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8380C &= 0x8000u;
  word_4F83810 = 0;
  qword_4F83850 = 0x100000000LL;
  qword_4F83818 = 0;
  qword_4F83820 = 0;
  qword_4F83828 = 0;
  dword_4F83808 = v0;
  qword_4F83830 = 0;
  qword_4F83838 = 0;
  qword_4F83840 = 0;
  qword_4F83848 = (__int64)&unk_4F83858;
  qword_4F83860 = 0;
  qword_4F83868 = (__int64)&unk_4F83880;
  qword_4F83870 = 1;
  dword_4F83878 = 0;
  byte_4F8387C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F83850;
  v3 = (unsigned int)qword_4F83850 + 1LL;
  if ( v3 > HIDWORD(qword_4F83850) )
  {
    sub_C8D5F0((char *)&unk_4F83858 - 16, &unk_4F83858, v3, 8);
    v2 = (unsigned int)qword_4F83850;
  }
  *(_QWORD *)(qword_4F83848 + 8 * v2) = v1;
  LODWORD(qword_4F83850) = qword_4F83850 + 1;
  qword_4F83888 = 0;
  qword_4F83890 = (__int64)&unk_49D9748;
  qword_4F83898 = 0;
  qword_4F83800 = (__int64)&unk_49DC090;
  qword_4F838A0 = (__int64)&unk_49DC1D0;
  qword_4F838C0 = (__int64)nullsub_23;
  qword_4F838B8 = (__int64)sub_984030;
  sub_C53080(&qword_4F83800, "disable-bitcode-version-upgrade", 31);
  qword_4F83830 = 54;
  LOBYTE(dword_4F8380C) = dword_4F8380C & 0x9F | 0x20;
  qword_4F83828 = (__int64)"Disable automatic bitcode upgrade for version mismatch";
  sub_C53130(&qword_4F83800);
  __cxa_atexit(sub_984900, &qword_4F83800, &qword_4A427C0);
  result = getenv("LLVM_OVERRIDE_PRODUCER");
  if ( !result )
    result = a2000;
  qword_4F837E0 = result;
  return result;
}
