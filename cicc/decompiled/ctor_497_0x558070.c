// Function: ctor_497
// Address: 0x558070
//
int ctor_497()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5009560 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50095DC = 1;
  qword_50095B0 = 0x100000000LL;
  dword_500956C &= 0x8000u;
  qword_5009578 = 0;
  qword_5009580 = 0;
  qword_5009588 = 0;
  dword_5009568 = v0;
  word_5009570 = 0;
  qword_5009590 = 0;
  qword_5009598 = 0;
  qword_50095A0 = 0;
  qword_50095A8 = (__int64)&unk_50095B8;
  qword_50095C0 = 0;
  qword_50095C8 = (__int64)&unk_50095E0;
  qword_50095D0 = 1;
  dword_50095D8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50095B0;
  v3 = (unsigned int)qword_50095B0 + 1LL;
  if ( v3 > HIDWORD(qword_50095B0) )
  {
    sub_C8D5F0((char *)&unk_50095B8 - 16, &unk_50095B8, v3, 8);
    v2 = (unsigned int)qword_50095B0;
  }
  *(_QWORD *)(qword_50095A8 + 8 * v2) = v1;
  LODWORD(qword_50095B0) = qword_50095B0 + 1;
  qword_50095E8 = 0;
  qword_50095F0 = (__int64)&unk_49D9748;
  qword_50095F8 = 0;
  qword_5009560 = (__int64)&unk_49DC090;
  qword_5009600 = (__int64)&unk_49DC1D0;
  qword_5009620 = (__int64)nullsub_23;
  qword_5009618 = (__int64)sub_984030;
  sub_C53080(&qword_5009560, "ignore-redundant-instrumentation", 32);
  qword_5009590 = 32;
  qword_5009588 = (__int64)"Ignore redundant instrumentation";
  LOBYTE(qword_50095E8) = 0;
  LOBYTE(dword_500956C) = dword_500956C & 0x9F | 0x20;
  LOWORD(qword_50095F8) = 256;
  sub_C53130(&qword_5009560);
  return __cxa_atexit(sub_984900, &qword_5009560, &qword_4A427C0);
}
