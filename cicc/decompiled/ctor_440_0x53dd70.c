// Function: ctor_440
// Address: 0x53dd70
//
int ctor_440()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FF9CE0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF9CEC &= 0x8000u;
  word_4FF9CF0 = 0;
  qword_4FF9D30 = 0x100000000LL;
  qword_4FF9CF8 = 0;
  qword_4FF9D00 = 0;
  qword_4FF9D08 = 0;
  dword_4FF9CE8 = v0;
  qword_4FF9D10 = 0;
  qword_4FF9D18 = 0;
  qword_4FF9D20 = 0;
  qword_4FF9D28 = (__int64)&unk_4FF9D38;
  qword_4FF9D40 = 0;
  qword_4FF9D48 = (__int64)&unk_4FF9D60;
  qword_4FF9D50 = 1;
  dword_4FF9D58 = 0;
  byte_4FF9D5C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF9D30;
  v3 = (unsigned int)qword_4FF9D30 + 1LL;
  if ( v3 > HIDWORD(qword_4FF9D30) )
  {
    sub_C8D5F0((char *)&unk_4FF9D38 - 16, &unk_4FF9D38, v3, 8);
    v2 = (unsigned int)qword_4FF9D30;
  }
  *(_QWORD *)(qword_4FF9D28 + 8 * v2) = v1;
  LODWORD(qword_4FF9D30) = qword_4FF9D30 + 1;
  qword_4FF9D68 = 0;
  qword_4FF9D70 = (__int64)&unk_49D9728;
  qword_4FF9D78 = 0;
  qword_4FF9CE0 = (__int64)&unk_49DBF10;
  qword_4FF9D80 = (__int64)&unk_49DC290;
  qword_4FF9DA0 = (__int64)nullsub_24;
  qword_4FF9D98 = (__int64)sub_984050;
  sub_C53080(&qword_4FF9CE0, "callsite-splitting-duplication-threshold", 40);
  qword_4FF9D10 = 82;
  LODWORD(qword_4FF9D68) = 5;
  BYTE4(qword_4FF9D78) = 1;
  LODWORD(qword_4FF9D78) = 5;
  LOBYTE(dword_4FF9CEC) = dword_4FF9CEC & 0x9F | 0x20;
  qword_4FF9D08 = (__int64)"Only allow instructions before a call, if their cost is below DuplicationThreshold";
  sub_C53130(&qword_4FF9CE0);
  return __cxa_atexit(sub_984970, &qword_4FF9CE0, &qword_4A427C0);
}
