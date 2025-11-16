// Function: ctor_689
// Address: 0x5a6930
//
int __fastcall ctor_689(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5040120 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_504019C = 1;
  qword_5040170 = 0x100000000LL;
  dword_504012C &= 0x8000u;
  qword_5040138 = 0;
  qword_5040140 = 0;
  qword_5040148 = 0;
  dword_5040128 = v4;
  word_5040130 = 0;
  qword_5040150 = 0;
  qword_5040158 = 0;
  qword_5040160 = 0;
  qword_5040168 = (__int64)&unk_5040178;
  qword_5040180 = 0;
  qword_5040188 = (__int64)&unk_50401A0;
  qword_5040190 = 1;
  dword_5040198 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040170;
  v7 = (unsigned int)qword_5040170 + 1LL;
  if ( v7 > HIDWORD(qword_5040170) )
  {
    sub_C8D5F0((char *)&unk_5040178 - 16, &unk_5040178, v7, 8);
    v6 = (unsigned int)qword_5040170;
  }
  *(_QWORD *)(qword_5040168 + 8 * v6) = v5;
  LODWORD(qword_5040170) = qword_5040170 + 1;
  qword_50401A8 = 0;
  qword_50401B0 = (__int64)&unk_49D9748;
  qword_50401B8 = 0;
  qword_5040120 = (__int64)&unk_49DC090;
  qword_50401C0 = (__int64)&unk_49DC1D0;
  qword_50401E0 = (__int64)nullsub_23;
  qword_50401D8 = (__int64)sub_984030;
  sub_C53080(&qword_5040120, "enable-patchpoint-liveness", 26);
  LOBYTE(qword_50401A8) = 1;
  qword_5040150 = 40;
  LOBYTE(dword_504012C) = dword_504012C & 0x9F | 0x20;
  LOWORD(qword_50401B8) = 257;
  qword_5040148 = (__int64)"Enable PatchPoint Liveness Analysis Pass";
  sub_C53130(&qword_5040120);
  return __cxa_atexit(sub_984900, &qword_5040120, &qword_4A427C0);
}
