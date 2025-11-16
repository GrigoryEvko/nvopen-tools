// Function: ctor_445
// Address: 0x53f5c0
//
int ctor_445()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+8h] [rbp-58h]
  char v17; // [rsp+13h] [rbp-4Dh] BYREF
  int v18; // [rsp+14h] [rbp-4Ch] BYREF
  char *v19; // [rsp+18h] [rbp-48h] BYREF
  const char *v20; // [rsp+20h] [rbp-40h] BYREF
  __int64 v21; // [rsp+28h] [rbp-38h]

  v19 = &v17;
  v20 = "View the CFG before DFA Jump Threading";
  v17 = 0;
  v18 = 1;
  v21 = 38;
  sub_243B8A0(&unk_4FFB020, "dfa-jump-view-cfg-before", &v20, &v18, &v19);
  __cxa_atexit(sub_984900, &unk_4FFB020, &qword_4A427C0);
  v19 = &v17;
  v20 = "Exit early if an unpredictable value come from the same loop";
  v17 = 1;
  v18 = 1;
  v21 = 60;
  sub_243B8A0(&unk_4FFAF40, "dfa-early-exit-heuristic", &v20, &v18, &v19);
  __cxa_atexit(sub_984900, &unk_4FFAF40, &qword_4A427C0);
  qword_4FFAE60 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFAEB0 = 0x100000000LL;
  dword_4FFAE6C &= 0x8000u;
  word_4FFAE70 = 0;
  qword_4FFAE78 = 0;
  qword_4FFAE80 = 0;
  dword_4FFAE68 = v0;
  qword_4FFAE88 = 0;
  qword_4FFAE90 = 0;
  qword_4FFAE98 = 0;
  qword_4FFAEA0 = 0;
  qword_4FFAEA8 = (__int64)&unk_4FFAEB8;
  qword_4FFAEC0 = 0;
  qword_4FFAEC8 = (__int64)&unk_4FFAEE0;
  qword_4FFAED0 = 1;
  dword_4FFAED8 = 0;
  byte_4FFAEDC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFAEB0;
  v3 = (unsigned int)qword_4FFAEB0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFAEB0) )
  {
    sub_C8D5F0((char *)&unk_4FFAEB8 - 16, &unk_4FFAEB8, v3, 8);
    v2 = (unsigned int)qword_4FFAEB0;
  }
  *(_QWORD *)(qword_4FFAEA8 + 8 * v2) = v1;
  qword_4FFAEF0 = (__int64)&unk_49D9728;
  qword_4FFAE60 = (__int64)&unk_49DBF10;
  qword_4FFAF00 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFAEB0) = qword_4FFAEB0 + 1;
  qword_4FFAF20 = (__int64)nullsub_24;
  qword_4FFAEE8 = 0;
  qword_4FFAF18 = (__int64)sub_984050;
  qword_4FFAEF8 = 0;
  sub_C53080(&qword_4FFAE60, "dfa-max-path-length", 19);
  qword_4FFAE90 = 54;
  qword_4FFAE88 = (__int64)"Max number of blocks searched to find a threading path";
  LODWORD(qword_4FFAEE8) = 20;
  BYTE4(qword_4FFAEF8) = 1;
  LODWORD(qword_4FFAEF8) = 20;
  LOBYTE(dword_4FFAE6C) = dword_4FFAE6C & 0x9F | 0x20;
  sub_C53130(&qword_4FFAE60);
  __cxa_atexit(sub_984970, &qword_4FFAE60, &qword_4A427C0);
  qword_4FFAD80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFADD0 = 0x100000000LL;
  dword_4FFAD8C &= 0x8000u;
  word_4FFAD90 = 0;
  qword_4FFADC8 = (__int64)&unk_4FFADD8;
  qword_4FFAD98 = 0;
  dword_4FFAD88 = v4;
  qword_4FFADA0 = 0;
  qword_4FFADA8 = 0;
  qword_4FFADB0 = 0;
  qword_4FFADB8 = 0;
  qword_4FFADC0 = 0;
  qword_4FFADE0 = 0;
  qword_4FFADE8 = (__int64)&unk_4FFAE00;
  qword_4FFADF0 = 1;
  dword_4FFADF8 = 0;
  byte_4FFADFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFADD0;
  if ( (unsigned __int64)(unsigned int)qword_4FFADD0 + 1 > HIDWORD(qword_4FFADD0) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_4FFADD8 - 16, &unk_4FFADD8, (unsigned int)qword_4FFADD0 + 1LL, 8);
    v6 = (unsigned int)qword_4FFADD0;
    v5 = v15;
  }
  *(_QWORD *)(qword_4FFADC8 + 8 * v6) = v5;
  qword_4FFAE10 = (__int64)&unk_49D9728;
  qword_4FFAD80 = (__int64)&unk_49DBF10;
  qword_4FFAE20 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFADD0) = qword_4FFADD0 + 1;
  qword_4FFAE40 = (__int64)nullsub_24;
  qword_4FFAE08 = 0;
  qword_4FFAE38 = (__int64)sub_984050;
  qword_4FFAE18 = 0;
  sub_C53080(&qword_4FFAD80, "dfa-max-num-visited-paths", 25);
  qword_4FFADB0 = 68;
  qword_4FFADA8 = (__int64)"Max number of blocks visited while enumerating paths around a switch";
  LODWORD(qword_4FFAE08) = 2500;
  BYTE4(qword_4FFAE18) = 1;
  LODWORD(qword_4FFAE18) = 2500;
  LOBYTE(dword_4FFAD8C) = dword_4FFAD8C & 0x9F | 0x20;
  sub_C53130(&qword_4FFAD80);
  __cxa_atexit(sub_984970, &qword_4FFAD80, &qword_4A427C0);
  qword_4FFACA0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFAD1C = 1;
  qword_4FFACF0 = 0x100000000LL;
  dword_4FFACAC &= 0x8000u;
  qword_4FFACE8 = (__int64)&unk_4FFACF8;
  qword_4FFACB8 = 0;
  qword_4FFACC0 = 0;
  dword_4FFACA8 = v7;
  word_4FFACB0 = 0;
  qword_4FFACC8 = 0;
  qword_4FFACD0 = 0;
  qword_4FFACD8 = 0;
  qword_4FFACE0 = 0;
  qword_4FFAD00 = 0;
  qword_4FFAD08 = (__int64)&unk_4FFAD20;
  qword_4FFAD10 = 1;
  dword_4FFAD18 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FFACF0;
  if ( (unsigned __int64)(unsigned int)qword_4FFACF0 + 1 > HIDWORD(qword_4FFACF0) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_4FFACF8 - 16, &unk_4FFACF8, (unsigned int)qword_4FFACF0 + 1LL, 8);
    v9 = (unsigned int)qword_4FFACF0;
    v8 = v16;
  }
  *(_QWORD *)(qword_4FFACE8 + 8 * v9) = v8;
  qword_4FFAD30 = (__int64)&unk_49D9728;
  qword_4FFACA0 = (__int64)&unk_49DBF10;
  qword_4FFAD40 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFACF0) = qword_4FFACF0 + 1;
  qword_4FFAD60 = (__int64)nullsub_24;
  qword_4FFAD28 = 0;
  qword_4FFAD58 = (__int64)sub_984050;
  qword_4FFAD38 = 0;
  sub_C53080(&qword_4FFACA0, "dfa-max-num-paths", 17);
  qword_4FFACD0 = 46;
  qword_4FFACC8 = (__int64)"Max number of paths enumerated around a switch";
  LODWORD(qword_4FFAD28) = 200;
  BYTE4(qword_4FFAD38) = 1;
  LODWORD(qword_4FFAD38) = 200;
  LOBYTE(dword_4FFACAC) = dword_4FFACAC & 0x9F | 0x20;
  sub_C53130(&qword_4FFACA0);
  __cxa_atexit(sub_984970, &qword_4FFACA0, &qword_4A427C0);
  qword_4FFABC0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFABCC &= 0x8000u;
  word_4FFABD0 = 0;
  qword_4FFAC10 = 0x100000000LL;
  qword_4FFAC08 = (__int64)&unk_4FFAC18;
  qword_4FFABD8 = 0;
  qword_4FFABE0 = 0;
  dword_4FFABC8 = v10;
  qword_4FFABE8 = 0;
  qword_4FFABF0 = 0;
  qword_4FFABF8 = 0;
  qword_4FFAC00 = 0;
  qword_4FFAC20 = 0;
  qword_4FFAC28 = (__int64)&unk_4FFAC40;
  qword_4FFAC30 = 1;
  dword_4FFAC38 = 0;
  byte_4FFAC3C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FFAC10;
  v13 = (unsigned int)qword_4FFAC10 + 1LL;
  if ( v13 > HIDWORD(qword_4FFAC10) )
  {
    sub_C8D5F0((char *)&unk_4FFAC18 - 16, &unk_4FFAC18, v13, 8);
    v12 = (unsigned int)qword_4FFAC10;
  }
  *(_QWORD *)(qword_4FFAC08 + 8 * v12) = v11;
  qword_4FFAC50 = (__int64)&unk_49D9728;
  qword_4FFABC0 = (__int64)&unk_49DBF10;
  qword_4FFAC60 = (__int64)&unk_49DC290;
  LODWORD(qword_4FFAC10) = qword_4FFAC10 + 1;
  qword_4FFAC80 = (__int64)nullsub_24;
  qword_4FFAC48 = 0;
  qword_4FFAC78 = (__int64)sub_984050;
  qword_4FFAC58 = 0;
  sub_C53080(&qword_4FFABC0, "dfa-cost-threshold", 18);
  qword_4FFABF0 = 44;
  qword_4FFABE8 = (__int64)"Maximum cost accepted for the transformation";
  LODWORD(qword_4FFAC48) = 50;
  BYTE4(qword_4FFAC58) = 1;
  LODWORD(qword_4FFAC58) = 50;
  LOBYTE(dword_4FFABCC) = dword_4FFABCC & 0x9F | 0x20;
  sub_C53130(&qword_4FFABC0);
  return __cxa_atexit(sub_984970, &qword_4FFABC0, &qword_4A427C0);
}
