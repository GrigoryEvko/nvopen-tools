// Function: ctor_487
// Address: 0x552e70
//
int ctor_487()
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
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_5007260 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50072B0 = 0x100000000LL;
  dword_500726C &= 0x8000u;
  word_5007270 = 0;
  qword_5007278 = 0;
  qword_5007280 = 0;
  dword_5007268 = v0;
  qword_5007288 = 0;
  qword_5007290 = 0;
  qword_5007298 = 0;
  qword_50072A0 = 0;
  qword_50072A8 = (__int64)&unk_50072B8;
  qword_50072C0 = 0;
  qword_50072C8 = (__int64)&unk_50072E0;
  qword_50072D0 = 1;
  dword_50072D8 = 0;
  byte_50072DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50072B0;
  v3 = (unsigned int)qword_50072B0 + 1LL;
  if ( v3 > HIDWORD(qword_50072B0) )
  {
    sub_C8D5F0((char *)&unk_50072B8 - 16, &unk_50072B8, v3, 8);
    v2 = (unsigned int)qword_50072B0;
  }
  *(_QWORD *)(qword_50072A8 + 8 * v2) = v1;
  qword_50072F0 = (__int64)&unk_49D9728;
  qword_5007260 = (__int64)&unk_49DBF10;
  LODWORD(qword_50072B0) = qword_50072B0 + 1;
  qword_50072E8 = 0;
  qword_5007300 = (__int64)&unk_49DC290;
  qword_50072F8 = 0;
  qword_5007320 = (__int64)nullsub_24;
  qword_5007318 = (__int64)sub_984050;
  sub_C53080(&qword_5007260, "spec-exec-max-speculation-cost", 30);
  LODWORD(qword_50072E8) = 7;
  BYTE4(qword_50072F8) = 1;
  LODWORD(qword_50072F8) = 7;
  qword_5007290 = 132;
  LOBYTE(dword_500726C) = dword_500726C & 0x9F | 0x20;
  qword_5007288 = (__int64)"Speculative execution is not applied to basic blocks where the cost of the instructions to sp"
                           "eculatively execute exceeds this limit.";
  sub_C53130(&qword_5007260);
  __cxa_atexit(sub_984970, &qword_5007260, &qword_4A427C0);
  qword_5007180 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50071D0 = 0x100000000LL;
  word_5007190 = 0;
  dword_500718C &= 0x8000u;
  qword_5007198 = 0;
  qword_50071A0 = 0;
  dword_5007188 = v4;
  qword_50071A8 = 0;
  qword_50071B0 = 0;
  qword_50071B8 = 0;
  qword_50071C0 = 0;
  qword_50071C8 = (__int64)&unk_50071D8;
  qword_50071E0 = 0;
  qword_50071E8 = (__int64)&unk_5007200;
  qword_50071F0 = 1;
  dword_50071F8 = 0;
  byte_50071FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50071D0;
  if ( (unsigned __int64)(unsigned int)qword_50071D0 + 1 > HIDWORD(qword_50071D0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_50071D8 - 16, &unk_50071D8, (unsigned int)qword_50071D0 + 1LL, 8);
    v6 = (unsigned int)qword_50071D0;
    v5 = v12;
  }
  *(_QWORD *)(qword_50071C8 + 8 * v6) = v5;
  qword_5007210 = (__int64)&unk_49D9728;
  qword_5007180 = (__int64)&unk_49DBF10;
  LODWORD(qword_50071D0) = qword_50071D0 + 1;
  qword_5007208 = 0;
  qword_5007220 = (__int64)&unk_49DC290;
  qword_5007218 = 0;
  qword_5007240 = (__int64)nullsub_24;
  qword_5007238 = (__int64)sub_984050;
  sub_C53080(&qword_5007180, "spec-exec-max-not-hoisted", 25);
  LODWORD(qword_5007208) = 5;
  BYTE4(qword_5007218) = 1;
  LODWORD(qword_5007218) = 5;
  qword_50071B0 = 146;
  LOBYTE(dword_500718C) = dword_500718C & 0x9F | 0x20;
  qword_50071A8 = (__int64)"Speculative execution is not applied to basic blocks where the number of instructions that wo"
                           "uld not be speculatively executed exceeds this limit.";
  sub_C53130(&qword_5007180);
  __cxa_atexit(sub_984970, &qword_5007180, &qword_4A427C0);
  qword_50070A0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500711C = 1;
  qword_50070F0 = 0x100000000LL;
  dword_50070AC &= 0x8000u;
  qword_50070B8 = 0;
  qword_50070C0 = 0;
  qword_50070C8 = 0;
  dword_50070A8 = v7;
  word_50070B0 = 0;
  qword_50070D0 = 0;
  qword_50070D8 = 0;
  qword_50070E0 = 0;
  qword_50070E8 = (__int64)&unk_50070F8;
  qword_5007100 = 0;
  qword_5007108 = (__int64)&unk_5007120;
  qword_5007110 = 1;
  dword_5007118 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_50070F0;
  v10 = (unsigned int)qword_50070F0 + 1LL;
  if ( v10 > HIDWORD(qword_50070F0) )
  {
    sub_C8D5F0((char *)&unk_50070F8 - 16, &unk_50070F8, v10, 8);
    v9 = (unsigned int)qword_50070F0;
  }
  *(_QWORD *)(qword_50070E8 + 8 * v9) = v8;
  LODWORD(qword_50070F0) = qword_50070F0 + 1;
  qword_5007128 = 0;
  qword_5007130 = (__int64)&unk_49D9748;
  qword_5007138 = 0;
  qword_50070A0 = (__int64)&unk_49DC090;
  qword_5007140 = (__int64)&unk_49DC1D0;
  qword_5007160 = (__int64)nullsub_23;
  qword_5007158 = (__int64)sub_984030;
  sub_C53080(&qword_50070A0, "spec-exec-only-if-divergent-target", 34);
  LOBYTE(qword_5007128) = 0;
  LOWORD(qword_5007138) = 256;
  qword_50070D0 = 135;
  LOBYTE(dword_50070AC) = dword_50070AC & 0x9F | 0x20;
  qword_50070C8 = (__int64)"Speculative execution is applied only to targets with divergent branches, even if the pass wa"
                           "s configured to apply only to all targets.";
  sub_C53130(&qword_50070A0);
  return __cxa_atexit(sub_984900, &qword_50070A0, &qword_4A427C0);
}
