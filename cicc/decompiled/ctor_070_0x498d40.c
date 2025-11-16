// Function: ctor_070
// Address: 0x498d40
//
int ctor_070()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F8BDC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8BDCC &= 0x8000u;
  word_4F8BDD0 = 0;
  qword_4F8BE10 = 0x100000000LL;
  qword_4F8BDD8 = 0;
  qword_4F8BDE0 = 0;
  qword_4F8BDE8 = 0;
  dword_4F8BDC8 = v0;
  qword_4F8BDF0 = 0;
  qword_4F8BDF8 = 0;
  qword_4F8BE00 = 0;
  qword_4F8BE08 = (__int64)&unk_4F8BE18;
  qword_4F8BE20 = 0;
  qword_4F8BE28 = (__int64)&unk_4F8BE40;
  qword_4F8BE30 = 1;
  dword_4F8BE38 = 0;
  byte_4F8BE3C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8BE10;
  v3 = (unsigned int)qword_4F8BE10 + 1LL;
  if ( v3 > HIDWORD(qword_4F8BE10) )
  {
    sub_C8D5F0((char *)&unk_4F8BE18 - 16, &unk_4F8BE18, v3, 8);
    v2 = (unsigned int)qword_4F8BE10;
  }
  *(_QWORD *)(qword_4F8BE08 + 8 * v2) = v1;
  LODWORD(qword_4F8BE10) = qword_4F8BE10 + 1;
  qword_4F8BE48 = 0;
  qword_4F8BE50 = (__int64)&unk_49D9728;
  qword_4F8BE58 = 0;
  qword_4F8BDC0 = (__int64)&unk_49DBF10;
  qword_4F8BE60 = (__int64)&unk_49DC290;
  qword_4F8BE80 = (__int64)nullsub_24;
  qword_4F8BE78 = (__int64)sub_984050;
  sub_C53080(&qword_4F8BDC0, "max-deopt-or-unreachable-succ-check-depth", 41);
  LODWORD(qword_4F8BE48) = 8;
  BYTE4(qword_4F8BE58) = 1;
  LODWORD(qword_4F8BE58) = 8;
  qword_4F8BDF0 = 171;
  LOBYTE(dword_4F8BDCC) = dword_4F8BDCC & 0x9F | 0x20;
  qword_4F8BDE8 = (__int64)"Set the maximum path length when checking whether a basic block is followed by a block that e"
                           "ither has a terminating deoptimizing call or is terminated with an unreachable";
  sub_C53130(&qword_4F8BDC0);
  return __cxa_atexit(sub_984970, &qword_4F8BDC0, &qword_4A427C0);
}
