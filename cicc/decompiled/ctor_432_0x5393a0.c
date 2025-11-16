// Function: ctor_432
// Address: 0x5393a0
//
int ctor_432()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FF5FA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF5FAC &= 0x8000u;
  word_4FF5FB0 = 0;
  qword_4FF5FF0 = 0x100000000LL;
  qword_4FF5FB8 = 0;
  qword_4FF5FC0 = 0;
  qword_4FF5FC8 = 0;
  dword_4FF5FA8 = v0;
  qword_4FF5FD0 = 0;
  qword_4FF5FD8 = 0;
  qword_4FF5FE0 = 0;
  qword_4FF5FE8 = (__int64)&unk_4FF5FF8;
  qword_4FF6000 = 0;
  qword_4FF6008 = (__int64)&unk_4FF6020;
  qword_4FF6010 = 1;
  dword_4FF6018 = 0;
  byte_4FF601C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF5FF0;
  v3 = (unsigned int)qword_4FF5FF0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF5FF0) )
  {
    sub_C8D5F0((char *)&unk_4FF5FF8 - 16, &unk_4FF5FF8, v3, 8);
    v2 = (unsigned int)qword_4FF5FF0;
  }
  *(_QWORD *)(qword_4FF5FE8 + 8 * v2) = v1;
  LODWORD(qword_4FF5FF0) = qword_4FF5FF0 + 1;
  qword_4FF6028 = 0;
  qword_4FF6030 = (__int64)&unk_49D9728;
  qword_4FF6038 = 0;
  qword_4FF5FA0 = (__int64)&unk_49DBF10;
  qword_4FF6040 = (__int64)&unk_49DC290;
  qword_4FF6060 = (__int64)nullsub_24;
  qword_4FF6058 = (__int64)sub_984050;
  sub_C53080(&qword_4FF5FA0, "funcspec-max-iters", 18);
  LODWORD(qword_4FF6028) = 10;
  BYTE4(qword_4FF6038) = 1;
  LODWORD(qword_4FF6038) = 10;
  qword_4FF5FD0 = 63;
  LOBYTE(dword_4FF5FAC) = dword_4FF5FAC & 0x9F | 0x20;
  qword_4FF5FC8 = (__int64)"The maximum number of iterations function specialization is run";
  sub_C53130(&qword_4FF5FA0);
  return __cxa_atexit(sub_984970, &qword_4FF5FA0, &qword_4A427C0);
}
