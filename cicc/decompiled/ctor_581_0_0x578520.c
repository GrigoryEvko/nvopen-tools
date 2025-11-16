// Function: ctor_581_0
// Address: 0x578520
//
int ctor_581_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  int v8; // [rsp+0h] [rbp-F0h] BYREF
  int v9; // [rsp+4h] [rbp-ECh] BYREF
  int *v10; // [rsp+8h] [rbp-E8h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v12[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v13[2]; // [rsp+30h] [rbp-C0h] BYREF
  int v14; // [rsp+40h] [rbp-B0h]
  char *v15; // [rsp+48h] [rbp-A8h]
  __int64 v16; // [rsp+50h] [rbp-A0h]
  char *v17; // [rsp+58h] [rbp-98h]
  __int64 v18; // [rsp+60h] [rbp-90h]
  int v19; // [rsp+68h] [rbp-88h]
  const char *v20; // [rsp+70h] [rbp-80h]
  __int64 v21; // [rsp+78h] [rbp-78h]
  const char *v22; // [rsp+80h] [rbp-70h]
  __int64 v23; // [rsp+88h] [rbp-68h]
  int v24; // [rsp+90h] [rbp-60h]
  const char *v25; // [rsp+98h] [rbp-58h]
  __int64 v26; // [rsp+A0h] [rbp-50h]

  v13[0] = "default";
  v15 = "Default";
  v17 = "release";
  v20 = "precompiled";
  v22 = "development";
  v25 = "for training";
  v12[1] = 0x400000003LL;
  v11[0] = "Enable regalloc advisor mode";
  v12[0] = v13;
  v13[1] = 7;
  v14 = 0;
  v16 = 7;
  v18 = 7;
  v19 = 1;
  v21 = 11;
  v23 = 11;
  v24 = 2;
  v26 = 12;
  v11[1] = 28;
  v9 = 0;
  v10 = &v9;
  v8 = 1;
  ((void (__fastcall *)(void *, const char *, int *, int **, _QWORD *, _QWORD *))sub_2F40DA0)(
    &unk_5023600,
    "regalloc-enable-advisor",
    &v8,
    &v10,
    v11,
    v12);
  if ( (_QWORD *)v12[0] != v13 )
    _libc_free(v12[0], "regalloc-enable-advisor");
  __cxa_atexit(sub_2F402B0, &unk_5023600, &qword_4A427C0);
  qword_5023520 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5023570 = 0x100000000LL;
  word_5023530 = 0;
  dword_502352C &= 0x8000u;
  qword_5023538 = 0;
  qword_5023540 = 0;
  dword_5023528 = v0;
  qword_5023548 = 0;
  qword_5023550 = 0;
  qword_5023558 = 0;
  qword_5023560 = 0;
  qword_5023568 = (__int64)&unk_5023578;
  qword_5023580 = 0;
  qword_5023588 = (__int64)&unk_50235A0;
  qword_5023590 = 1;
  dword_5023598 = 0;
  byte_502359C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5023570;
  v3 = (unsigned int)qword_5023570 + 1LL;
  if ( v3 > HIDWORD(qword_5023570) )
  {
    sub_C8D5F0((char *)&unk_5023578 - 16, &unk_5023578, v3, 8);
    v2 = (unsigned int)qword_5023570;
  }
  *(_QWORD *)(qword_5023568 + 8 * v2) = v1;
  LODWORD(qword_5023570) = qword_5023570 + 1;
  qword_50235A8 = 0;
  qword_50235B0 = (__int64)&unk_49D9748;
  qword_50235B8 = 0;
  qword_5023520 = (__int64)&unk_49DC090;
  qword_50235C0 = (__int64)&unk_49DC1D0;
  qword_50235E0 = (__int64)nullsub_23;
  qword_50235D8 = (__int64)sub_984030;
  sub_C53080(&qword_5023520, "enable-local-reassign", 21);
  qword_5023550 = 91;
  LOBYTE(qword_50235A8) = 0;
  LOBYTE(dword_502352C) = dword_502352C & 0x9F | 0x20;
  qword_5023548 = (__int64)"Local reassignment can yield better allocation decisions, but may be compile time intensive";
  LOWORD(qword_50235B8) = 256;
  sub_C53130(&qword_5023520);
  __cxa_atexit(sub_984900, &qword_5023520, &qword_4A427C0);
  qword_5023440 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_502344C = word_502344C & 0x8000;
  unk_5023448 = v4;
  qword_5023488[1] = 0x100000000LL;
  unk_5023450 = 0;
  unk_5023458 = 0;
  unk_5023460 = 0;
  unk_5023468 = 0;
  unk_5023470 = 0;
  unk_5023478 = 0;
  unk_5023480 = 0;
  qword_5023488[0] = &qword_5023488[2];
  qword_5023488[3] = 0;
  qword_5023488[4] = &qword_5023488[7];
  qword_5023488[5] = 1;
  LODWORD(qword_5023488[6]) = 0;
  BYTE4(qword_5023488[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_5023488[1]);
  if ( (unsigned __int64)LODWORD(qword_5023488[1]) + 1 > HIDWORD(qword_5023488[1]) )
  {
    sub_C8D5F0(qword_5023488, &qword_5023488[2], LODWORD(qword_5023488[1]) + 1LL, 8);
    v6 = LODWORD(qword_5023488[1]);
  }
  *(_QWORD *)(qword_5023488[0] + 8 * v6) = v5;
  ++LODWORD(qword_5023488[1]);
  qword_5023488[8] = 0;
  qword_5023488[9] = &unk_49D9728;
  qword_5023488[10] = 0;
  qword_5023440 = &unk_49DBF10;
  qword_5023488[11] = &unk_49DC290;
  qword_5023488[15] = nullsub_24;
  qword_5023488[14] = sub_984050;
  sub_C53080(&qword_5023440, "regalloc-eviction-max-interference-cutoff", 41);
  unk_5023470 = 175;
  LODWORD(qword_5023488[8]) = 10;
  BYTE4(qword_5023488[10]) = 1;
  LODWORD(qword_5023488[10]) = 10;
  LOBYTE(word_502344C) = word_502344C & 0x9F | 0x20;
  unk_5023468 = "Number of interferences after which we declare an interference unevictable and bail out. This is a compi"
                "lation cost-saving consideration. To disable, pass a very large number.";
  sub_C53130(&qword_5023440);
  return __cxa_atexit(sub_984970, &qword_5023440, &qword_4A427C0);
}
