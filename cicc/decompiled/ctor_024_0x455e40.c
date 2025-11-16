// Function: ctor_024
// Address: 0x455e40
//
int ctor_024()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F81620 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8169C = 1;
  qword_4F81670 = 0x100000000LL;
  dword_4F8162C &= 0x8000u;
  qword_4F81638 = 0;
  qword_4F81640 = 0;
  qword_4F81648 = 0;
  dword_4F81628 = v0;
  word_4F81630 = 0;
  qword_4F81650 = 0;
  qword_4F81658 = 0;
  qword_4F81660 = 0;
  qword_4F81668 = (__int64)&unk_4F81678;
  qword_4F81680 = 0;
  qword_4F81688 = (__int64)&unk_4F816A0;
  qword_4F81690 = 1;
  dword_4F81698 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F81670;
  v3 = (unsigned int)qword_4F81670 + 1LL;
  if ( v3 > HIDWORD(qword_4F81670) )
  {
    sub_C8D5F0((char *)&unk_4F81678 - 16, &unk_4F81678, v3, 8);
    v2 = (unsigned int)qword_4F81670;
  }
  *(_QWORD *)(qword_4F81668 + 8 * v2) = v1;
  LODWORD(qword_4F81670) = qword_4F81670 + 1;
  qword_4F816A8 = 0;
  qword_4F816B0 = (__int64)&unk_49D9748;
  qword_4F816B8 = 0;
  qword_4F81620 = (__int64)&unk_49DC090;
  qword_4F816C0 = (__int64)&unk_49DC1D0;
  qword_4F816E0 = (__int64)nullsub_23;
  qword_4F816D8 = (__int64)sub_984030;
  sub_C53080(&qword_4F81620, "disable-ipo-derefinement", 24);
  qword_4F81650 = 96;
  LOBYTE(qword_4F816A8) = 0;
  LOBYTE(dword_4F8162C) = dword_4F8162C & 0x9F | 0x20;
  qword_4F81648 = (__int64)"Stop inter-procedural optimizations on linkonce_odr/weak_odr functions that may get derefinement";
  LOWORD(qword_4F816B8) = 256;
  sub_C53130(&qword_4F81620);
  return __cxa_atexit(sub_984900, &qword_4F81620, &qword_4A427C0);
}
