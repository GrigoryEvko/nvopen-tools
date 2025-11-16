// Function: ctor_628
// Address: 0x58f580
//
int __fastcall ctor_628(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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

  qword_5031400 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503147C = 1;
  qword_5031450 = 0x100000000LL;
  dword_503140C &= 0x8000u;
  qword_5031418 = 0;
  qword_5031420 = 0;
  qword_5031428 = 0;
  dword_5031408 = v4;
  word_5031410 = 0;
  qword_5031430 = 0;
  qword_5031438 = 0;
  qword_5031440 = 0;
  qword_5031448 = (__int64)&unk_5031458;
  qword_5031460 = 0;
  qword_5031468 = (__int64)&unk_5031480;
  qword_5031470 = 1;
  dword_5031478 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5031450;
  v7 = (unsigned int)qword_5031450 + 1LL;
  if ( v7 > HIDWORD(qword_5031450) )
  {
    sub_C8D5F0((char *)&unk_5031458 - 16, &unk_5031458, v7, 8);
    v6 = (unsigned int)qword_5031450;
  }
  *(_QWORD *)(qword_5031448 + 8 * v6) = v5;
  qword_5031490 = (__int64)&unk_49D9728;
  LODWORD(qword_5031450) = qword_5031450 + 1;
  qword_5031488 = 0;
  qword_5031400 = (__int64)&unk_49DBF10;
  qword_50314A0 = (__int64)&unk_49DC290;
  qword_5031498 = 0;
  qword_50314C0 = (__int64)nullsub_24;
  qword_50314B8 = (__int64)sub_984050;
  sub_C53080(&qword_5031400, "default-trip-count", 18);
  LODWORD(qword_5031488) = 100;
  BYTE4(qword_5031498) = 1;
  LODWORD(qword_5031498) = 100;
  qword_5031430 = 52;
  LOBYTE(dword_503140C) = dword_503140C & 0x9F | 0x20;
  qword_5031428 = (__int64)"Use this to specify the default trip count of a loop";
  sub_C53130(&qword_5031400);
  __cxa_atexit(sub_984970, &qword_5031400, &qword_4A427C0);
  qword_5031320 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5031400, v8, v9), 1u);
  dword_503132C &= 0x8000u;
  word_5031330 = 0;
  qword_5031370 = 0x100000000LL;
  qword_5031338 = 0;
  qword_5031340 = 0;
  qword_5031348 = 0;
  dword_5031328 = v10;
  qword_5031350 = 0;
  qword_5031358 = 0;
  qword_5031360 = 0;
  qword_5031368 = (__int64)&unk_5031378;
  qword_5031380 = 0;
  qword_5031388 = (__int64)&unk_50313A0;
  qword_5031390 = 1;
  dword_5031398 = 0;
  byte_503139C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5031370;
  v13 = (unsigned int)qword_5031370 + 1LL;
  if ( v13 > HIDWORD(qword_5031370) )
  {
    sub_C8D5F0((char *)&unk_5031378 - 16, &unk_5031378, v13, 8);
    v12 = (unsigned int)qword_5031370;
  }
  *(_QWORD *)(qword_5031368 + 8 * v12) = v11;
  qword_50313B0 = (__int64)&unk_49D9728;
  LODWORD(qword_5031370) = qword_5031370 + 1;
  qword_50313A8 = 0;
  qword_5031320 = (__int64)&unk_49DBF10;
  qword_50313C0 = (__int64)&unk_49DC290;
  qword_50313B8 = 0;
  qword_50313E0 = (__int64)nullsub_24;
  qword_50313D8 = (__int64)sub_984050;
  sub_C53080(&qword_5031320, "temporal-reuse-threshold", 24);
  LODWORD(qword_50313A8) = 2;
  BYTE4(qword_50313B8) = 1;
  LODWORD(qword_50313B8) = 2;
  qword_5031350 = 138;
  LOBYTE(dword_503132C) = dword_503132C & 0x9F | 0x20;
  qword_5031348 = (__int64)"Use this to specify the max. distance between array elements accessed in a loop so that the e"
                           "lements are classified to have temporal reuse";
  sub_C53130(&qword_5031320);
  return __cxa_atexit(sub_984970, &qword_5031320, &qword_4A427C0);
}
