// Function: ctor_710
// Address: 0x5bea80
//
int __fastcall ctor_710(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5051360 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_50513DC = 1;
  qword_50513B0 = 0x100000000LL;
  dword_505136C &= 0x8000u;
  qword_5051378 = 0;
  qword_5051380 = 0;
  qword_5051388 = 0;
  dword_5051368 = v4;
  word_5051370 = 0;
  qword_5051390 = 0;
  qword_5051398 = 0;
  qword_50513A0 = 0;
  qword_50513A8 = (__int64)&unk_50513B8;
  qword_50513C0 = 0;
  qword_50513C8 = (__int64)&unk_50513E0;
  qword_50513D0 = 1;
  dword_50513D8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50513B0;
  v7 = (unsigned int)qword_50513B0 + 1LL;
  if ( v7 > HIDWORD(qword_50513B0) )
  {
    sub_C8D5F0((char *)&unk_50513B8 - 16, &unk_50513B8, v7, 8);
    v6 = (unsigned int)qword_50513B0;
  }
  *(_QWORD *)(qword_50513A8 + 8 * v6) = v5;
  LODWORD(qword_50513B0) = qword_50513B0 + 1;
  qword_50513E8 = 0;
  qword_50513F0 = (__int64)&unk_49D9748;
  qword_50513F8 = 0;
  qword_5051360 = (__int64)&unk_49DC090;
  qword_5051400 = (__int64)&unk_49DC1D0;
  qword_5051420 = (__int64)nullsub_23;
  qword_5051418 = (__int64)sub_984030;
  sub_C53080(&qword_5051360, "verify-cfiinstrs", 16);
  qword_5051390 = 42;
  qword_5051388 = (__int64)"Verify Call Frame Information instructions";
  LOWORD(qword_50513F8) = 256;
  LOBYTE(qword_50513E8) = 0;
  LOBYTE(dword_505136C) = dword_505136C & 0x9F | 0x20;
  sub_C53130(&qword_5051360);
  return __cxa_atexit(sub_984900, &qword_5051360, &qword_4A427C0);
}
