// Function: ctor_596
// Address: 0x57d900
//
int __fastcall ctor_596(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5026340 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5026390 = 0x100000000LL;
  word_5026350 = 0;
  dword_502634C &= 0x8000u;
  qword_5026358 = 0;
  qword_5026360 = 0;
  dword_5026348 = v4;
  qword_5026368 = 0;
  qword_5026370 = 0;
  qword_5026378 = 0;
  qword_5026380 = 0;
  qword_5026388 = (__int64)&unk_5026398;
  qword_50263A0 = 0;
  qword_50263A8 = (__int64)&unk_50263C0;
  qword_50263B0 = 1;
  dword_50263B8 = 0;
  byte_50263BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5026390;
  v7 = (unsigned int)qword_5026390 + 1LL;
  if ( v7 > HIDWORD(qword_5026390) )
  {
    sub_C8D5F0((char *)&unk_5026398 - 16, &unk_5026398, v7, 8);
    v6 = (unsigned int)qword_5026390;
  }
  *(_QWORD *)(qword_5026388 + 8 * v6) = v5;
  LODWORD(qword_5026390) = qword_5026390 + 1;
  qword_50263C8 = 0;
  qword_50263D0 = (__int64)&unk_49D9748;
  qword_50263D8 = 0;
  qword_5026340 = (__int64)&unk_49DC090;
  qword_50263E0 = (__int64)&unk_49DC1D0;
  qword_5026400 = (__int64)nullsub_23;
  qword_50263F8 = (__int64)sub_984030;
  sub_C53080(&qword_5026340, "no-stack-slot-sharing", 21);
  LOBYTE(qword_50263C8) = 0;
  LOWORD(qword_50263D8) = 256;
  qword_5026370 = 43;
  LOBYTE(dword_502634C) = dword_502634C & 0x9F | 0x20;
  qword_5026368 = (__int64)"Suppress slot sharing during stack coloring";
  sub_C53130(&qword_5026340);
  __cxa_atexit(sub_984900, &qword_5026340, &qword_4A427C0);
  qword_5026260 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5026340, v8, v9), 1u);
  byte_50262DC = 1;
  qword_50262B0 = 0x100000000LL;
  dword_502626C &= 0x8000u;
  qword_5026278 = 0;
  qword_5026280 = 0;
  qword_5026288 = 0;
  dword_5026268 = v10;
  word_5026270 = 0;
  qword_5026290 = 0;
  qword_5026298 = 0;
  qword_50262A0 = 0;
  qword_50262A8 = (__int64)&unk_50262B8;
  qword_50262C0 = 0;
  qword_50262C8 = (__int64)&unk_50262E0;
  qword_50262D0 = 1;
  dword_50262D8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50262B0;
  v13 = (unsigned int)qword_50262B0 + 1LL;
  if ( v13 > HIDWORD(qword_50262B0) )
  {
    sub_C8D5F0((char *)&unk_50262B8 - 16, &unk_50262B8, v13, 8);
    v12 = (unsigned int)qword_50262B0;
  }
  *(_QWORD *)(qword_50262A8 + 8 * v12) = v11;
  LODWORD(qword_50262B0) = qword_50262B0 + 1;
  qword_50262E8 = 0;
  qword_50262F0 = (__int64)&unk_49DA090;
  qword_50262F8 = 0;
  qword_5026260 = (__int64)&unk_49DBF90;
  qword_5026300 = (__int64)&unk_49DC230;
  qword_5026320 = (__int64)nullsub_58;
  qword_5026318 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5026260, "ssc-dce-limit", 13);
  LODWORD(qword_50262E8) = -1;
  BYTE4(qword_50262F8) = 1;
  LODWORD(qword_50262F8) = -1;
  LOBYTE(dword_502626C) = dword_502626C & 0x9F | 0x20;
  sub_C53130(&qword_5026260);
  return __cxa_atexit(sub_B2B680, &qword_5026260, &qword_4A427C0);
}
