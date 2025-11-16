// Function: ctor_013
// Address: 0x453910
//
int ctor_013()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  qword_4F803E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F80430 = 0x100000000LL;
  dword_4F803EC &= 0x8000u;
  word_4F803F0 = 0;
  qword_4F803F8 = 0;
  qword_4F80400 = 0;
  dword_4F803E8 = v0;
  qword_4F80408 = 0;
  qword_4F80410 = 0;
  qword_4F80418 = 0;
  qword_4F80420 = 0;
  qword_4F80428 = (__int64)&unk_4F80438;
  qword_4F80440 = 0;
  qword_4F80448 = (__int64)&unk_4F80460;
  qword_4F80450 = 1;
  dword_4F80458 = 0;
  byte_4F8045C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F80430;
  v3 = (unsigned int)qword_4F80430 + 1LL;
  if ( v3 > HIDWORD(qword_4F80430) )
  {
    sub_C8D5F0((char *)&unk_4F80438 - 16, &unk_4F80438, v3, 8);
    v2 = (unsigned int)qword_4F80430;
  }
  *(_QWORD *)(qword_4F80428 + 8 * v2) = v1;
  qword_4F80470 = (__int64)&unk_49D9748;
  qword_4F803E0 = (__int64)&unk_49DC090;
  LODWORD(qword_4F80430) = qword_4F80430 + 1;
  qword_4F80468 = 0;
  qword_4F80480 = (__int64)&unk_49DC1D0;
  qword_4F80478 = 0;
  qword_4F804A0 = (__int64)nullsub_23;
  qword_4F80498 = (__int64)sub_984030;
  sub_C53080(&qword_4F803E0, "print-summary-global-ids", 24);
  LOWORD(qword_4F80478) = 256;
  LOBYTE(qword_4F80468) = 0;
  qword_4F80410 = 66;
  LOBYTE(dword_4F803EC) = dword_4F803EC & 0x9F | 0x20;
  qword_4F80408 = (__int64)"Print the global id for each value when reading the module summary";
  sub_C53130(&qword_4F803E0);
  __cxa_atexit(sub_984900, &qword_4F803E0, &qword_4A427C0);
  qword_4F80300 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F80350 = 0x100000000LL;
  word_4F80310 = 0;
  dword_4F8030C &= 0x8000u;
  qword_4F80318 = 0;
  qword_4F80320 = 0;
  dword_4F80308 = v4;
  qword_4F80328 = 0;
  qword_4F80330 = 0;
  qword_4F80338 = 0;
  qword_4F80340 = 0;
  qword_4F80348 = (__int64)&unk_4F80358;
  qword_4F80360 = 0;
  qword_4F80368 = (__int64)&unk_4F80380;
  qword_4F80370 = 1;
  dword_4F80378 = 0;
  byte_4F8037C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F80350;
  if ( (unsigned __int64)(unsigned int)qword_4F80350 + 1 > HIDWORD(qword_4F80350) )
  {
    v11 = v5;
    sub_C8D5F0((char *)&unk_4F80358 - 16, &unk_4F80358, (unsigned int)qword_4F80350 + 1LL, 8);
    v6 = (unsigned int)qword_4F80350;
    v5 = v11;
  }
  *(_QWORD *)(qword_4F80348 + 8 * v6) = v5;
  qword_4F80390 = (__int64)&unk_49D9748;
  qword_4F80300 = (__int64)&unk_49DC090;
  LODWORD(qword_4F80350) = qword_4F80350 + 1;
  qword_4F80388 = 0;
  qword_4F803A0 = (__int64)&unk_49DC1D0;
  qword_4F80398 = 0;
  qword_4F803C0 = (__int64)nullsub_23;
  qword_4F803B8 = (__int64)sub_984030;
  sub_C53080(&qword_4F80300, "expand-constant-exprs", 21);
  qword_4F80330 = 64;
  LOBYTE(dword_4F8030C) = dword_4F8030C & 0x9F | 0x20;
  qword_4F80328 = (__int64)"Expand constant expressions to instructions for testing purposes";
  sub_C53130(&qword_4F80300);
  __cxa_atexit(sub_984900, &qword_4F80300, &qword_4A427C0);
  qword_4F80220 = &unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8022C = word_4F8022C & 0x8000;
  unk_4F80230 = 0;
  qword_4F80268[1] = 0x100000000LL;
  unk_4F80228 = v7;
  unk_4F80238 = 0;
  unk_4F80240 = 0;
  unk_4F80248 = 0;
  unk_4F80250 = 0;
  unk_4F80258 = 0;
  unk_4F80260 = 0;
  qword_4F80268[0] = &qword_4F80268[2];
  qword_4F80268[3] = 0;
  qword_4F80268[4] = &qword_4F80268[7];
  qword_4F80268[5] = 1;
  LODWORD(qword_4F80268[6]) = 0;
  BYTE4(qword_4F80268[6]) = 1;
  v8 = sub_C57470();
  v9 = LODWORD(qword_4F80268[1]);
  if ( (unsigned __int64)LODWORD(qword_4F80268[1]) + 1 > HIDWORD(qword_4F80268[1]) )
  {
    sub_C8D5F0(qword_4F80268, &qword_4F80268[2], LODWORD(qword_4F80268[1]) + 1LL, 8);
    v9 = LODWORD(qword_4F80268[1]);
  }
  *(_QWORD *)(qword_4F80268[0] + 8 * v9) = v8;
  ++LODWORD(qword_4F80268[1]);
  qword_4F80268[8] = 0;
  qword_4F80268[9] = &unk_49DC110;
  qword_4F80268[10] = 0;
  qword_4F80220 = &unk_49D97F0;
  qword_4F80268[11] = &unk_49DC200;
  qword_4F80268[15] = nullsub_26;
  qword_4F80268[14] = sub_9C26D0;
  sub_C53080(&qword_4F80220, "load-bitcode-into-experimental-debuginfo-iterators", 50);
  unk_4F80250 = 81;
  LOBYTE(word_4F8022C) = word_4F8022C & 0x9F | 0x20;
  unk_4F80248 = "Load bitcode directly into the new debug info format (regardless of input format)";
  sub_C53130(&qword_4F80220);
  return __cxa_atexit(sub_9C44F0, &qword_4F80220, &qword_4A427C0);
}
