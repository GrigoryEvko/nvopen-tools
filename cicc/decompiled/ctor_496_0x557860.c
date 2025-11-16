// Function: ctor_496
// Address: 0x557860
//
int ctor_496()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_5009480 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50094D0 = 0x100000000LL;
  dword_500948C &= 0x8000u;
  word_5009490 = 0;
  qword_5009498 = 0;
  qword_50094A0 = 0;
  dword_5009488 = v0;
  qword_50094A8 = 0;
  qword_50094B0 = 0;
  qword_50094B8 = 0;
  qword_50094C0 = 0;
  qword_50094C8 = (__int64)&unk_50094D8;
  qword_50094E0 = 0;
  qword_50094E8 = (__int64)&unk_5009500;
  qword_50094F0 = 1;
  dword_50094F8 = 0;
  byte_50094FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50094D0;
  v3 = (unsigned int)qword_50094D0 + 1LL;
  if ( v3 > HIDWORD(qword_50094D0) )
  {
    sub_C8D5F0((char *)&unk_50094D8 - 16, &unk_50094D8, v3, 8);
    v2 = (unsigned int)qword_50094D0;
  }
  *(_QWORD *)(qword_50094C8 + 8 * v2) = v1;
  qword_5009510 = (__int64)&unk_49D9748;
  qword_5009480 = (__int64)&unk_49DC090;
  qword_5009520 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50094D0) = qword_50094D0 + 1;
  qword_5009540 = (__int64)nullsub_23;
  qword_5009508 = 0;
  qword_5009538 = (__int64)sub_984030;
  qword_5009518 = 0;
  sub_C53080(&qword_5009480, "enable-noalias-to-md-conversion", 31);
  LOBYTE(qword_5009508) = 1;
  LOWORD(qword_5009518) = 257;
  qword_50094B0 = 55;
  LOBYTE(dword_500948C) = dword_500948C & 0x9F | 0x20;
  qword_50094A8 = (__int64)"Convert noalias attributes to metadata during inlining.";
  sub_C53130(&qword_5009480);
  __cxa_atexit(sub_984900, &qword_5009480, &qword_4A427C0);
  qword_50093A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50093F0 = 0x100000000LL;
  dword_50093AC &= 0x8000u;
  word_50093B0 = 0;
  qword_50093E8 = (__int64)&unk_50093F8;
  qword_50093B8 = 0;
  dword_50093A8 = v4;
  qword_50093C0 = 0;
  qword_50093C8 = 0;
  qword_50093D0 = 0;
  qword_50093D8 = 0;
  qword_50093E0 = 0;
  qword_5009400 = 0;
  qword_5009408 = (__int64)&unk_5009420;
  qword_5009410 = 1;
  dword_5009418 = 0;
  byte_500941C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50093F0;
  if ( (unsigned __int64)(unsigned int)qword_50093F0 + 1 > HIDWORD(qword_50093F0) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_50093F8 - 16, &unk_50093F8, (unsigned int)qword_50093F0 + 1LL, 8);
    v6 = (unsigned int)qword_50093F0;
    v5 = v15;
  }
  *(_QWORD *)(qword_50093E8 + 8 * v6) = v5;
  qword_5009430 = (__int64)&unk_49D9748;
  qword_50093A0 = (__int64)&unk_49DC090;
  qword_5009440 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50093F0) = qword_50093F0 + 1;
  qword_5009460 = (__int64)nullsub_23;
  qword_5009428 = 0;
  qword_5009458 = (__int64)sub_984030;
  qword_5009438 = 0;
  sub_C53080(&qword_50093A0, "use-noalias-intrinsic-during-inlining", 37);
  LOWORD(qword_5009438) = 257;
  LOBYTE(qword_5009428) = 1;
  qword_50093D0 = 71;
  LOBYTE(dword_50093AC) = dword_50093AC & 0x9F | 0x20;
  qword_50093C8 = (__int64)"Use the llvm.experimental.noalias.scope.decl intrinsic during inlining.";
  sub_C53130(&qword_50093A0);
  __cxa_atexit(sub_984900, &qword_50093A0, &qword_4A427C0);
  qword_50092C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500933C = 1;
  word_50092D0 = 0;
  qword_5009310 = 0x100000000LL;
  dword_50092CC &= 0x8000u;
  qword_5009308 = (__int64)&unk_5009318;
  qword_50092D8 = 0;
  dword_50092C8 = v7;
  qword_50092E0 = 0;
  qword_50092E8 = 0;
  qword_50092F0 = 0;
  qword_50092F8 = 0;
  qword_5009300 = 0;
  qword_5009320 = 0;
  qword_5009328 = (__int64)&unk_5009340;
  qword_5009330 = 1;
  dword_5009338 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5009310;
  if ( (unsigned __int64)(unsigned int)qword_5009310 + 1 > HIDWORD(qword_5009310) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_5009318 - 16, &unk_5009318, (unsigned int)qword_5009310 + 1LL, 8);
    v9 = (unsigned int)qword_5009310;
    v8 = v16;
  }
  *(_QWORD *)(qword_5009308 + 8 * v9) = v8;
  qword_5009350 = (__int64)&unk_49D9748;
  qword_50092C0 = (__int64)&unk_49DC090;
  qword_5009360 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5009310) = qword_5009310 + 1;
  qword_5009380 = (__int64)nullsub_23;
  qword_5009348 = 0;
  qword_5009378 = (__int64)sub_984030;
  qword_5009358 = 0;
  sub_C53080(&qword_50092C0, "preserve-alignment-assumptions-during-inlining", 46);
  LOBYTE(qword_5009348) = 0;
  LOWORD(qword_5009358) = 256;
  qword_50092F0 = 56;
  LOBYTE(dword_50092CC) = dword_50092CC & 0x9F | 0x20;
  qword_50092E8 = (__int64)"Convert align attributes to assumptions during inlining.";
  sub_C53130(&qword_50092C0);
  __cxa_atexit(sub_984900, &qword_50092C0, &qword_4A427C0);
  qword_50091E0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500925C = 1;
  qword_5009230 = 0x100000000LL;
  dword_50091EC &= 0x8000u;
  qword_50091F8 = 0;
  qword_5009200 = 0;
  qword_5009208 = 0;
  dword_50091E8 = v10;
  word_50091F0 = 0;
  qword_5009210 = 0;
  qword_5009218 = 0;
  qword_5009220 = 0;
  qword_5009228 = (__int64)&unk_5009238;
  qword_5009240 = 0;
  qword_5009248 = (__int64)&unk_5009260;
  qword_5009250 = 1;
  dword_5009258 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5009230;
  v13 = (unsigned int)qword_5009230 + 1LL;
  if ( v13 > HIDWORD(qword_5009230) )
  {
    sub_C8D5F0((char *)&unk_5009238 - 16, &unk_5009238, v13, 8);
    v12 = (unsigned int)qword_5009230;
  }
  *(_QWORD *)(qword_5009228 + 8 * v12) = v11;
  LODWORD(qword_5009230) = qword_5009230 + 1;
  qword_5009268 = 0;
  qword_5009270 = (__int64)&unk_49D9728;
  qword_5009278 = 0;
  qword_50091E0 = (__int64)&unk_49DBF10;
  qword_5009280 = (__int64)&unk_49DC290;
  qword_50092A0 = (__int64)nullsub_24;
  qword_5009298 = (__int64)sub_984050;
  sub_C53080(&qword_50091E0, "max-inst-checked-for-throw-during-inlining", 42);
  qword_5009210 = 100;
  LODWORD(qword_5009268) = 4;
  BYTE4(qword_5009278) = 1;
  LODWORD(qword_5009278) = 4;
  LOBYTE(dword_50091EC) = dword_50091EC & 0x9F | 0x20;
  qword_5009208 = (__int64)"the maximum number of instructions analyzed for may throw during attribute inference in inlined body";
  sub_C53130(&qword_50091E0);
  return __cxa_atexit(sub_984970, &qword_50091E0, &qword_4A427C0);
}
