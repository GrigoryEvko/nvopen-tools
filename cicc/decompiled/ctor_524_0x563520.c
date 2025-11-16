// Function: ctor_524
// Address: 0x563520
//
int ctor_524()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_50111A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501121C = 1;
  qword_50111F0 = 0x100000000LL;
  dword_50111AC &= 0x8000u;
  qword_50111B8 = 0;
  qword_50111C0 = 0;
  qword_50111C8 = 0;
  dword_50111A8 = v0;
  word_50111B0 = 0;
  qword_50111D0 = 0;
  qword_50111D8 = 0;
  qword_50111E0 = 0;
  qword_50111E8 = (__int64)&unk_50111F8;
  qword_5011200 = 0;
  qword_5011208 = (__int64)&unk_5011220;
  qword_5011210 = 1;
  dword_5011218 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50111F0;
  v3 = (unsigned int)qword_50111F0 + 1LL;
  if ( v3 > HIDWORD(qword_50111F0) )
  {
    sub_C8D5F0((char *)&unk_50111F8 - 16, &unk_50111F8, v3, 8);
    v2 = (unsigned int)qword_50111F0;
  }
  *(_QWORD *)(qword_50111E8 + 8 * v2) = v1;
  LODWORD(qword_50111F0) = qword_50111F0 + 1;
  qword_5011228 = 0;
  qword_5011230 = (__int64)&unk_49D9748;
  qword_5011238 = 0;
  qword_50111A0 = (__int64)&unk_49DC090;
  qword_5011240 = (__int64)&unk_49DC1D0;
  qword_5011260 = (__int64)nullsub_23;
  qword_5011258 = (__int64)sub_984030;
  sub_C53080(&qword_50111A0, "basic-dbe", 9);
  LOBYTE(qword_5011228) = 0;
  LOWORD(qword_5011238) = 256;
  qword_50111D0 = 100;
  LOBYTE(dword_50111AC) = dword_50111AC & 0x9F | 0x20;
  qword_50111C8 = (__int64)"Run the basic dead barrier elimination which uses one bit for the liveness of memory at the barrier.";
  sub_C53130(&qword_50111A0);
  return __cxa_atexit(sub_984900, &qword_50111A0, &qword_4A427C0);
}
