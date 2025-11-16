// Function: ctor_429
// Address: 0x536d40
//
int ctor_429()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FF4480 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF44FC = 1;
  qword_4FF44D0 = 0x100000000LL;
  dword_4FF448C &= 0x8000u;
  qword_4FF4498 = 0;
  qword_4FF44A0 = 0;
  qword_4FF44A8 = 0;
  dword_4FF4488 = v0;
  word_4FF4490 = 0;
  qword_4FF44B0 = 0;
  qword_4FF44B8 = 0;
  qword_4FF44C0 = 0;
  qword_4FF44C8 = (__int64)&unk_4FF44D8;
  qword_4FF44E0 = 0;
  qword_4FF44E8 = (__int64)&unk_4FF4500;
  qword_4FF44F0 = 1;
  dword_4FF44F8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF44D0;
  v3 = (unsigned int)qword_4FF44D0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF44D0) )
  {
    sub_C8D5F0((char *)&unk_4FF44D8 - 16, &unk_4FF44D8, v3, 8);
    v2 = (unsigned int)qword_4FF44D0;
  }
  *(_QWORD *)(qword_4FF44C8 + 8 * v2) = v1;
  LODWORD(qword_4FF44D0) = qword_4FF44D0 + 1;
  qword_4FF4508 = 0;
  qword_4FF4510 = (__int64)&unk_49D9748;
  qword_4FF4518 = 0;
  qword_4FF4480 = (__int64)&unk_49DC090;
  qword_4FF4520 = (__int64)&unk_49DC1D0;
  qword_4FF4540 = (__int64)nullsub_23;
  qword_4FF4538 = (__int64)sub_984030;
  sub_C53080(&qword_4FF4480, "ctx-prof-promote-alwaysinline", 29);
  LOBYTE(qword_4FF4508) = 0;
  LOWORD(qword_4FF4518) = 256;
  qword_4FF44B0 = 240;
  LOBYTE(dword_4FF448C) = dword_4FF448C & 0x9F | 0x20;
  qword_4FF44A8 = (__int64)"If using a contextual profile in this module, and an indirect call target is marked as always"
                           "inline, perform indirect call promotion for that target. If multiple targets for an indirect "
                           "call site fit this description, they are all promoted.";
  sub_C53130(&qword_4FF4480);
  return __cxa_atexit(sub_984900, &qword_4FF4480, &qword_4A427C0);
}
