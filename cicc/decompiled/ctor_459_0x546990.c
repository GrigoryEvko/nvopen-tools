// Function: ctor_459
// Address: 0x546990
//
int ctor_459()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FFEA80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFEAFC = 1;
  qword_4FFEAD0 = 0x100000000LL;
  dword_4FFEA8C &= 0x8000u;
  qword_4FFEA98 = 0;
  qword_4FFEAA0 = 0;
  qword_4FFEAA8 = 0;
  dword_4FFEA88 = v0;
  word_4FFEA90 = 0;
  qword_4FFEAB0 = 0;
  qword_4FFEAB8 = 0;
  qword_4FFEAC0 = 0;
  qword_4FFEAC8 = (__int64)&unk_4FFEAD8;
  qword_4FFEAE0 = 0;
  qword_4FFEAE8 = (__int64)&unk_4FFEB00;
  qword_4FFEAF0 = 1;
  dword_4FFEAF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFEAD0;
  v3 = (unsigned int)qword_4FFEAD0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFEAD0) )
  {
    sub_C8D5F0((char *)&unk_4FFEAD8 - 16, &unk_4FFEAD8, v3, 8);
    v2 = (unsigned int)qword_4FFEAD0;
  }
  *(_QWORD *)(qword_4FFEAC8 + 8 * v2) = v1;
  LODWORD(qword_4FFEAD0) = qword_4FFEAD0 + 1;
  qword_4FFEB08 = 0;
  qword_4FFEB10 = (__int64)&unk_49D9748;
  qword_4FFEB18 = 0;
  qword_4FFEA80 = (__int64)&unk_49DC090;
  qword_4FFEB20 = (__int64)&unk_49DC1D0;
  qword_4FFEB40 = (__int64)nullsub_23;
  qword_4FFEB38 = (__int64)sub_984030;
  sub_C53080(&qword_4FFEA80, "loop-deletion-enable-symbolic-execution", 39);
  LOBYTE(qword_4FFEB08) = 1;
  qword_4FFEAB0 = 111;
  LOBYTE(dword_4FFEA8C) = dword_4FFEA8C & 0x9F | 0x20;
  LOWORD(qword_4FFEB18) = 257;
  qword_4FFEAA8 = (__int64)"Break backedge through symbolic execution of 1st iteration attempting to prove that the backe"
                           "dge is never taken";
  sub_C53130(&qword_4FFEA80);
  return __cxa_atexit(sub_984900, &qword_4FFEA80, &qword_4A427C0);
}
