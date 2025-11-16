// Function: ctor_571
// Address: 0x5743a0
//
int ctor_571()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5020A00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5020A7C = 1;
  qword_5020A50 = 0x100000000LL;
  dword_5020A0C &= 0x8000u;
  qword_5020A18 = 0;
  qword_5020A20 = 0;
  qword_5020A28 = 0;
  dword_5020A08 = v0;
  word_5020A10 = 0;
  qword_5020A30 = 0;
  qword_5020A38 = 0;
  qword_5020A40 = 0;
  qword_5020A48 = (__int64)&unk_5020A58;
  qword_5020A60 = 0;
  qword_5020A68 = (__int64)&unk_5020A80;
  qword_5020A70 = 1;
  dword_5020A78 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5020A50;
  v3 = (unsigned int)qword_5020A50 + 1LL;
  if ( v3 > HIDWORD(qword_5020A50) )
  {
    sub_C8D5F0((char *)&unk_5020A58 - 16, &unk_5020A58, v3, 8);
    v2 = (unsigned int)qword_5020A50;
  }
  *(_QWORD *)(qword_5020A48 + 8 * v2) = v1;
  LODWORD(qword_5020A50) = qword_5020A50 + 1;
  qword_5020A88 = 0;
  qword_5020A90 = (__int64)&unk_49D9748;
  qword_5020A98 = 0;
  qword_5020A00 = (__int64)&unk_49DC090;
  qword_5020AA0 = (__int64)&unk_49DC1D0;
  qword_5020AC0 = (__int64)nullsub_23;
  qword_5020AB8 = (__int64)sub_984030;
  sub_C53080(&qword_5020A00, "enable-subreg-liveness", 22);
  LOBYTE(qword_5020A88) = 1;
  qword_5020A30 = 37;
  LOBYTE(dword_5020A0C) = dword_5020A0C & 0x9F | 0x20;
  LOWORD(qword_5020A98) = 257;
  qword_5020A28 = (__int64)"Enable subregister liveness tracking.";
  sub_C53130(&qword_5020A00);
  return __cxa_atexit(sub_984900, &qword_5020A00, &qword_4A427C0);
}
