// Function: ctor_012
// Address: 0x4536f0
//
int ctor_012()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F80140 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8014C &= 0x8000u;
  word_4F80150 = 0;
  qword_4F80190 = 0x100000000LL;
  qword_4F80158 = 0;
  qword_4F80160 = 0;
  qword_4F80168 = 0;
  dword_4F80148 = v0;
  qword_4F80170 = 0;
  qword_4F80178 = 0;
  qword_4F80180 = 0;
  qword_4F80188 = (__int64)&unk_4F80198;
  qword_4F801A0 = 0;
  qword_4F801A8 = (__int64)&unk_4F801C0;
  qword_4F801B0 = 1;
  dword_4F801B8 = 0;
  byte_4F801BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F80190;
  v3 = (unsigned int)qword_4F80190 + 1LL;
  if ( v3 > HIDWORD(qword_4F80190) )
  {
    sub_C8D5F0((char *)&unk_4F80198 - 16, &unk_4F80198, v3, 8);
    v2 = (unsigned int)qword_4F80190;
  }
  *(_QWORD *)(qword_4F80188 + 8 * v2) = v1;
  LODWORD(qword_4F80190) = qword_4F80190 + 1;
  qword_4F801C8 = 0;
  qword_4F801D0 = (__int64)&unk_49D9728;
  qword_4F801D8 = 0;
  qword_4F80140 = (__int64)&unk_49DBF10;
  qword_4F801E0 = (__int64)&unk_49DC290;
  qword_4F80200 = (__int64)nullsub_24;
  qword_4F801F8 = (__int64)sub_984050;
  sub_C53080(&qword_4F80140, "max-interleave-group-factor", 27);
  qword_4F80170 = 60;
  LODWORD(qword_4F801C8) = 8;
  BYTE4(qword_4F801D8) = 1;
  LODWORD(qword_4F801D8) = 8;
  LOBYTE(dword_4F8014C) = dword_4F8014C & 0x9F | 0x20;
  qword_4F80168 = (__int64)"Maximum factor for an interleaved access group (default = 8)";
  sub_C53130(&qword_4F80140);
  return __cxa_atexit(sub_984970, &qword_4F80140, &qword_4A427C0);
}
