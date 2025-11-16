// Function: ctor_613
// Address: 0x5893e0
//
int __fastcall ctor_613(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_502D460 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_502D4DC = 1;
  qword_502D4B0 = 0x100000000LL;
  dword_502D46C &= 0x8000u;
  qword_502D478 = 0;
  qword_502D480 = 0;
  qword_502D488 = 0;
  dword_502D468 = v4;
  word_502D470 = 0;
  qword_502D490 = 0;
  qword_502D498 = 0;
  qword_502D4A0 = 0;
  qword_502D4A8 = (__int64)&unk_502D4B8;
  qword_502D4C0 = 0;
  qword_502D4C8 = (__int64)&unk_502D4E0;
  qword_502D4D0 = 1;
  dword_502D4D8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502D4B0;
  v7 = (unsigned int)qword_502D4B0 + 1LL;
  if ( v7 > HIDWORD(qword_502D4B0) )
  {
    sub_C8D5F0((char *)&unk_502D4B8 - 16, &unk_502D4B8, v7, 8);
    v6 = (unsigned int)qword_502D4B0;
  }
  *(_QWORD *)(qword_502D4A8 + 8 * v6) = v5;
  LODWORD(qword_502D4B0) = qword_502D4B0 + 1;
  qword_502D4E8 = 0;
  qword_502D4F0 = (__int64)&unk_49D9748;
  qword_502D4F8 = 0;
  qword_502D460 = (__int64)&unk_49DC090;
  qword_502D500 = (__int64)&unk_49DC1D0;
  qword_502D520 = (__int64)nullsub_23;
  qword_502D518 = (__int64)sub_984030;
  sub_C53080(&qword_502D460, "sink-ld-param", 13);
  LOBYTE(qword_502D4E8) = 0;
  LOWORD(qword_502D4F8) = 256;
  qword_502D490 = 38;
  LOBYTE(dword_502D46C) = dword_502D46C & 0x9F | 0x20;
  qword_502D488 = (__int64)"Sink one-use ld.param to the use point";
  sub_C53130(&qword_502D460);
  return __cxa_atexit(sub_984900, &qword_502D460, &qword_4A427C0);
}
