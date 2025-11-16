// Function: ctor_568
// Address: 0x573890
//
int ctor_568()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5020100 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_502010C &= 0x8000u;
  word_5020110 = 0;
  qword_5020150 = 0x100000000LL;
  qword_5020118 = 0;
  qword_5020120 = 0;
  qword_5020128 = 0;
  dword_5020108 = v0;
  qword_5020130 = 0;
  qword_5020138 = 0;
  qword_5020140 = 0;
  qword_5020148 = (__int64)&unk_5020158;
  qword_5020160 = 0;
  qword_5020168 = (__int64)&unk_5020180;
  qword_5020170 = 1;
  dword_5020178 = 0;
  byte_502017C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5020150;
  v3 = (unsigned int)qword_5020150 + 1LL;
  if ( v3 > HIDWORD(qword_5020150) )
  {
    sub_C8D5F0((char *)&unk_5020158 - 16, &unk_5020158, v3, 8);
    v2 = (unsigned int)qword_5020150;
  }
  *(_QWORD *)(qword_5020148 + 8 * v2) = v1;
  LODWORD(qword_5020150) = qword_5020150 + 1;
  qword_5020188 = 0;
  qword_5020190 = (__int64)&unk_49D9748;
  qword_5020198 = 0;
  qword_5020100 = (__int64)&unk_49DC090;
  qword_50201A0 = (__int64)&unk_49DC1D0;
  qword_50201C0 = (__int64)nullsub_23;
  qword_50201B8 = (__int64)sub_984030;
  sub_C53080(&qword_5020100, "print-mi-addrs", 14);
  qword_5020130 = 45;
  LOBYTE(dword_502010C) = dword_502010C & 0x9F | 0x20;
  qword_5020128 = (__int64)"Print addresses of MachineInstrs when dumping";
  sub_C53130(&qword_5020100);
  return __cxa_atexit(sub_984900, &qword_5020100, &qword_4A427C0);
}
