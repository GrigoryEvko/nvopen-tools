// Function: ctor_595
// Address: 0x57d520
//
int __fastcall ctor_595(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5026180 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_50261D0 = 0x100000000LL;
  dword_502618C &= 0x8000u;
  word_5026190 = 0;
  qword_5026198 = 0;
  qword_50261A0 = 0;
  dword_5026188 = v4;
  qword_50261A8 = 0;
  qword_50261B0 = 0;
  qword_50261B8 = 0;
  qword_50261C0 = 0;
  qword_50261C8 = (__int64)&unk_50261D8;
  qword_50261E0 = 0;
  qword_50261E8 = (__int64)&unk_5026200;
  qword_50261F0 = 1;
  dword_50261F8 = 0;
  byte_50261FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50261D0;
  v7 = (unsigned int)qword_50261D0 + 1LL;
  if ( v7 > HIDWORD(qword_50261D0) )
  {
    sub_C8D5F0((char *)&unk_50261D8 - 16, &unk_50261D8, v7, 8);
    v6 = (unsigned int)qword_50261D0;
  }
  *(_QWORD *)(qword_50261C8 + 8 * v6) = v5;
  qword_5026210 = (__int64)&unk_49D9748;
  LODWORD(qword_50261D0) = qword_50261D0 + 1;
  qword_5026208 = 0;
  qword_5026180 = (__int64)&unk_49DC090;
  qword_5026220 = (__int64)&unk_49DC1D0;
  qword_5026218 = 0;
  qword_5026240 = (__int64)nullsub_23;
  qword_5026238 = (__int64)sub_984030;
  sub_C53080(&qword_5026180, "enable-selectiondag-sp", 22);
  LOWORD(qword_5026218) = 257;
  LOBYTE(qword_5026208) = 1;
  LOBYTE(dword_502618C) = dword_502618C & 0x9F | 0x20;
  sub_C53130(&qword_5026180);
  __cxa_atexit(sub_984900, &qword_5026180, &qword_4A427C0);
  qword_50260A0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5026180, v8, v9), 1u);
  qword_50260F0 = 0x100000000LL;
  word_50260B0 = 0;
  dword_50260AC &= 0x8000u;
  qword_50260B8 = 0;
  qword_50260C0 = 0;
  dword_50260A8 = v10;
  qword_50260C8 = 0;
  qword_50260D0 = 0;
  qword_50260D8 = 0;
  qword_50260E0 = 0;
  qword_50260E8 = (__int64)&unk_50260F8;
  qword_5026100 = 0;
  qword_5026108 = (__int64)&unk_5026120;
  qword_5026110 = 1;
  dword_5026118 = 0;
  byte_502611C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50260F0;
  v13 = (unsigned int)qword_50260F0 + 1LL;
  if ( v13 > HIDWORD(qword_50260F0) )
  {
    sub_C8D5F0((char *)&unk_50260F8 - 16, &unk_50260F8, v13, 8);
    v12 = (unsigned int)qword_50260F0;
  }
  *(_QWORD *)(qword_50260E8 + 8 * v12) = v11;
  qword_5026130 = (__int64)&unk_49D9748;
  LODWORD(qword_50260F0) = qword_50260F0 + 1;
  qword_5026128 = 0;
  qword_50260A0 = (__int64)&unk_49DC090;
  qword_5026140 = (__int64)&unk_49DC1D0;
  qword_5026138 = 0;
  qword_5026160 = (__int64)nullsub_23;
  qword_5026158 = (__int64)sub_984030;
  sub_C53080(&qword_50260A0, "disable-check-noreturn-call", 27);
  LOBYTE(qword_5026128) = 0;
  LOWORD(qword_5026138) = 256;
  LOBYTE(dword_50260AC) = dword_50260AC & 0x9F | 0x20;
  sub_C53130(&qword_50260A0);
  return __cxa_atexit(sub_984900, &qword_50260A0, &qword_4A427C0);
}
