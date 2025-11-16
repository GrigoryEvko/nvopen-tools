// Function: ctor_023
// Address: 0x455c20
//
int ctor_023()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F81540 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8154C &= 0x8000u;
  word_4F81550 = 0;
  qword_4F81590 = 0x100000000LL;
  qword_4F81558 = 0;
  qword_4F81560 = 0;
  qword_4F81568 = 0;
  dword_4F81548 = v0;
  qword_4F81570 = 0;
  qword_4F81578 = 0;
  qword_4F81580 = 0;
  qword_4F81588 = (__int64)&unk_4F81598;
  qword_4F815A0 = 0;
  qword_4F815A8 = (__int64)&unk_4F815C0;
  qword_4F815B0 = 1;
  dword_4F815B8 = 0;
  byte_4F815BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F81590;
  v3 = (unsigned int)qword_4F81590 + 1LL;
  if ( v3 > HIDWORD(qword_4F81590) )
  {
    sub_C8D5F0((char *)&unk_4F81598 - 16, &unk_4F81598, v3, 8);
    v2 = (unsigned int)qword_4F81590;
  }
  *(_QWORD *)(qword_4F81588 + 8 * v2) = v1;
  LODWORD(qword_4F81590) = qword_4F81590 + 1;
  qword_4F815C8 = 0;
  qword_4F815D0 = (__int64)&unk_49DA090;
  qword_4F815D8 = 0;
  qword_4F81540 = (__int64)&unk_49DBF90;
  qword_4F815E0 = (__int64)&unk_49DC230;
  qword_4F81600 = (__int64)nullsub_58;
  qword_4F815F8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4F81540, "non-global-value-max-name-size", 30);
  LODWORD(qword_4F815C8) = 1024;
  BYTE4(qword_4F815D8) = 1;
  LODWORD(qword_4F815D8) = 1024;
  qword_4F81570 = 47;
  LOBYTE(dword_4F8154C) = dword_4F8154C & 0x9F | 0x20;
  qword_4F81568 = (__int64)"Maximum size for the name of non-global values.";
  sub_C53130(&qword_4F81540);
  return __cxa_atexit(sub_B2B680, &qword_4F81540, &qword_4A427C0);
}
