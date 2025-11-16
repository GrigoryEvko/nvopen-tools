// Function: ctor_690
// Address: 0x5a6b40
//
int __fastcall ctor_690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5040200 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_504027C = 1;
  qword_5040250 = 0x100000000LL;
  dword_504020C &= 0x8000u;
  qword_5040218 = 0;
  qword_5040220 = 0;
  qword_5040228 = 0;
  dword_5040208 = v4;
  word_5040210 = 0;
  qword_5040230 = 0;
  qword_5040238 = 0;
  qword_5040240 = 0;
  qword_5040248 = (__int64)&unk_5040258;
  qword_5040260 = 0;
  qword_5040268 = (__int64)&unk_5040280;
  qword_5040270 = 1;
  dword_5040278 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040250;
  v7 = (unsigned int)qword_5040250 + 1LL;
  if ( v7 > HIDWORD(qword_5040250) )
  {
    sub_C8D5F0((char *)&unk_5040258 - 16, &unk_5040258, v7, 8);
    v6 = (unsigned int)qword_5040250;
  }
  *(_QWORD *)(qword_5040248 + 8 * v6) = v5;
  LODWORD(qword_5040250) = qword_5040250 + 1;
  qword_5040288 = 0;
  qword_5040290 = (__int64)&unk_49D9748;
  qword_5040298 = 0;
  qword_5040200 = (__int64)&unk_49DC090;
  qword_50402A0 = (__int64)&unk_49DC1D0;
  qword_50402C0 = (__int64)nullsub_23;
  qword_50402B8 = (__int64)sub_984030;
  sub_C53080(&qword_5040200, "disable-type-promotion", 22);
  LOBYTE(qword_5040288) = 0;
  qword_5040230 = 27;
  LOBYTE(dword_504020C) = dword_504020C & 0x9F | 0x20;
  LOWORD(qword_5040298) = 256;
  qword_5040228 = (__int64)"Disable type promotion pass";
  sub_C53130(&qword_5040200);
  return __cxa_atexit(sub_984900, &qword_5040200, &qword_4A427C0);
}
