// Function: ctor_606
// Address: 0x584950
//
int __fastcall ctor_606(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_502AE00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_502AE7C = 1;
  qword_502AE50 = 0x100000000LL;
  dword_502AE0C &= 0x8000u;
  qword_502AE18 = 0;
  qword_502AE20 = 0;
  qword_502AE28 = 0;
  dword_502AE08 = v4;
  word_502AE10 = 0;
  qword_502AE30 = 0;
  qword_502AE38 = 0;
  qword_502AE40 = 0;
  qword_502AE48 = (__int64)&unk_502AE58;
  qword_502AE60 = 0;
  qword_502AE68 = (__int64)&unk_502AE80;
  qword_502AE70 = 1;
  dword_502AE78 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502AE50;
  v7 = (unsigned int)qword_502AE50 + 1LL;
  if ( v7 > HIDWORD(qword_502AE50) )
  {
    sub_C8D5F0((char *)&unk_502AE58 - 16, &unk_502AE58, v7, 8);
    v6 = (unsigned int)qword_502AE50;
  }
  *(_QWORD *)(qword_502AE48 + 8 * v6) = v5;
  LODWORD(qword_502AE50) = qword_502AE50 + 1;
  qword_502AE88 = 0;
  qword_502AE90 = (__int64)&unk_49D9748;
  qword_502AE98 = 0;
  qword_502AE00 = (__int64)&unk_49DC090;
  qword_502AEA0 = (__int64)&unk_49DC1D0;
  qword_502AEC0 = (__int64)nullsub_23;
  qword_502AEB8 = (__int64)sub_984030;
  sub_C53080(&qword_502AE00, "nvptx-lower-global-ctor-dtor", 28);
  qword_502AE30 = 48;
  qword_502AE28 = (__int64)"Lower GPU ctor / dtors to globals on the device.";
  LOWORD(qword_502AE98) = 256;
  LOBYTE(qword_502AE88) = 0;
  LOBYTE(dword_502AE0C) = dword_502AE0C & 0x9F | 0x20;
  sub_C53130(&qword_502AE00);
  return __cxa_atexit(sub_984900, &qword_502AE00, &qword_4A427C0);
}
