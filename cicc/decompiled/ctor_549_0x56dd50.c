// Function: ctor_549
// Address: 0x56dd50
//
int ctor_549()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501D300 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501D30C &= 0x8000u;
  word_501D310 = 0;
  qword_501D350 = 0x100000000LL;
  qword_501D318 = 0;
  qword_501D320 = 0;
  qword_501D328 = 0;
  dword_501D308 = v0;
  qword_501D330 = 0;
  qword_501D338 = 0;
  qword_501D340 = 0;
  qword_501D348 = (__int64)&unk_501D358;
  qword_501D360 = 0;
  qword_501D368 = (__int64)&unk_501D380;
  qword_501D370 = 1;
  dword_501D378 = 0;
  byte_501D37C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501D350;
  v3 = (unsigned int)qword_501D350 + 1LL;
  if ( v3 > HIDWORD(qword_501D350) )
  {
    sub_C8D5F0((char *)&unk_501D358 - 16, &unk_501D358, v3, 8);
    v2 = (unsigned int)qword_501D350;
  }
  *(_QWORD *)(qword_501D348 + 8 * v2) = v1;
  LODWORD(qword_501D350) = qword_501D350 + 1;
  qword_501D388 = 0;
  qword_501D390 = (__int64)&unk_49D9728;
  qword_501D398 = 0;
  qword_501D300 = (__int64)&unk_49DBF10;
  qword_501D3A0 = (__int64)&unk_49DC290;
  qword_501D3C0 = (__int64)nullsub_24;
  qword_501D3B8 = (__int64)sub_984050;
  sub_C53080(&qword_501D300, "expand-fp-convert-bits", 22);
  LODWORD(qword_501D388) = 0x800000;
  BYTE4(qword_501D398) = 1;
  LODWORD(qword_501D398) = 0x800000;
  qword_501D330 = 73;
  LOBYTE(dword_501D30C) = dword_501D30C & 0x9F | 0x20;
  qword_501D328 = (__int64)"fp convert instructions on integers with more than <N> bits are expanded.";
  sub_C53130(&qword_501D300);
  return __cxa_atexit(sub_984970, &qword_501D300, &qword_4A427C0);
}
