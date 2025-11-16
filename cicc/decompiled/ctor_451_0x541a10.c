// Function: ctor_451
// Address: 0x541a10
//
int ctor_451()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FFC600 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFC67C = 1;
  qword_4FFC650 = 0x100000000LL;
  dword_4FFC60C &= 0x8000u;
  qword_4FFC618 = 0;
  qword_4FFC620 = 0;
  qword_4FFC628 = 0;
  dword_4FFC608 = v0;
  word_4FFC610 = 0;
  qword_4FFC630 = 0;
  qword_4FFC638 = 0;
  qword_4FFC640 = 0;
  qword_4FFC648 = (__int64)&unk_4FFC658;
  qword_4FFC660 = 0;
  qword_4FFC668 = (__int64)&unk_4FFC680;
  qword_4FFC670 = 1;
  dword_4FFC678 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFC650;
  v3 = (unsigned int)qword_4FFC650 + 1LL;
  if ( v3 > HIDWORD(qword_4FFC650) )
  {
    sub_C8D5F0((char *)&unk_4FFC658 - 16, &unk_4FFC658, v3, 8);
    v2 = (unsigned int)qword_4FFC650;
  }
  *(_QWORD *)(qword_4FFC648 + 8 * v2) = v1;
  LODWORD(qword_4FFC650) = qword_4FFC650 + 1;
  qword_4FFC688 = 0;
  qword_4FFC690 = (__int64)&unk_49D9748;
  qword_4FFC698 = 0;
  qword_4FFC600 = (__int64)&unk_49DC090;
  qword_4FFC6A0 = (__int64)&unk_49DC1D0;
  qword_4FFC6C0 = (__int64)nullsub_23;
  qword_4FFC6B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFC600, "guard-widening-widen-branch-guards", 34);
  qword_4FFC630 = 84;
  LOBYTE(qword_4FFC688) = 1;
  LOBYTE(dword_4FFC60C) = dword_4FFC60C & 0x9F | 0x20;
  qword_4FFC628 = (__int64)"Whether or not we should widen guards  expressed as branches by widenable conditions";
  LOWORD(qword_4FFC698) = 257;
  sub_C53130(&qword_4FFC600);
  return __cxa_atexit(sub_984900, &qword_4FFC600, &qword_4A427C0);
}
