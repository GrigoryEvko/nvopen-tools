// Function: ctor_626_0
// Address: 0x58f170
//
int ctor_626_0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v7; // [rsp+0h] [rbp-E0h] BYREF
  __int64 *v8; // [rsp+8h] [rbp-D8h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v10[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v11[2]; // [rsp+30h] [rbp-B0h] BYREF
  int v12; // [rsp+40h] [rbp-A0h]
  const char *v13; // [rsp+48h] [rbp-98h]
  __int64 v14; // [rsp+50h] [rbp-90h]
  char *v15; // [rsp+58h] [rbp-88h]
  __int64 v16; // [rsp+60h] [rbp-80h]
  int v17; // [rsp+68h] [rbp-78h]
  const char *v18; // [rsp+70h] [rbp-70h]
  __int64 v19; // [rsp+78h] [rbp-68h]
  const char *v20; // [rsp+80h] [rbp-60h]
  __int64 v21; // [rsp+88h] [rbp-58h]
  int v22; // [rsp+90h] [rbp-50h]
  const char *v23; // [rsp+98h] [rbp-48h]
  __int64 v24; // [rsp+A0h] [rbp-40h]
  char *v25; // [rsp+A8h] [rbp-38h]
  __int64 v26; // [rsp+B0h] [rbp-30h]
  int v27; // [rsp+B8h] [rbp-28h]
  const char *v28; // [rsp+C0h] [rbp-20h]
  __int64 v29; // [rsp+C8h] [rbp-18h]

  v11[0] = "size";
  v13 = "Use callee size priority.";
  v15 = "cost";
  v18 = "Use inline cost priority.";
  v20 = "cost-benefit";
  v23 = "Use cost-benefit ratio.";
  v25 = "ml";
  v28 = "Use ML.";
  v10[1] = 0x400000004LL;
  v9[0] = "Choose the priority mode to use in module inline";
  v10[0] = v11;
  v11[1] = 4;
  v12 = 0;
  v14 = 25;
  v16 = 4;
  v17 = 1;
  v19 = 25;
  v21 = 12;
  v22 = 2;
  v24 = 23;
  v26 = 2;
  v27 = 3;
  v29 = 7;
  v9[1] = 48;
  v8 = &v7;
  ((void (__fastcall *)(void *, const char *, __int64 **, char *, _QWORD *, _QWORD *, __int64))sub_30E2E40)(
    &unk_5030FC0,
    "inline-priority-mode",
    &v8,
    (char *)&v7 + 4,
    v9,
    v10,
    0x100000000LL);
  if ( (_QWORD *)v10[0] != v11 )
    _libc_free(v10[0], "inline-priority-mode");
  __cxa_atexit(sub_30E2210, &unk_5030FC0, &qword_4A427C0);
  qword_5030EE0 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_30E2210, &unk_5030FC0, v0, v1), 1u);
  dword_5030EEC &= 0x8000u;
  word_5030EF0 = 0;
  qword_5030F30 = 0x100000000LL;
  qword_5030EF8 = 0;
  qword_5030F00 = 0;
  qword_5030F08 = 0;
  dword_5030EE8 = v2;
  qword_5030F10 = 0;
  qword_5030F18 = 0;
  qword_5030F20 = 0;
  qword_5030F28 = (__int64)&unk_5030F38;
  qword_5030F40 = 0;
  qword_5030F48 = (__int64)&unk_5030F60;
  qword_5030F50 = 1;
  dword_5030F58 = 0;
  byte_5030F5C = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_5030F30;
  v5 = (unsigned int)qword_5030F30 + 1LL;
  if ( v5 > HIDWORD(qword_5030F30) )
  {
    sub_C8D5F0((char *)&unk_5030F38 - 16, &unk_5030F38, v5, 8);
    v4 = (unsigned int)qword_5030F30;
  }
  *(_QWORD *)(qword_5030F28 + 8 * v4) = v3;
  LODWORD(qword_5030F30) = qword_5030F30 + 1;
  qword_5030F68 = 0;
  qword_5030F70 = (__int64)&unk_49DA090;
  qword_5030F78 = 0;
  qword_5030EE0 = (__int64)&unk_49DBF90;
  qword_5030F80 = (__int64)&unk_49DC230;
  qword_5030FA0 = (__int64)nullsub_58;
  qword_5030F98 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5030EE0, "module-inliner-top-priority-threshold", 37);
  LODWORD(qword_5030F68) = 0;
  BYTE4(qword_5030F78) = 1;
  LODWORD(qword_5030F78) = 0;
  qword_5030F10 = 84;
  LOBYTE(dword_5030EEC) = dword_5030EEC & 0x9F | 0x20;
  qword_5030F08 = (__int64)"The cost threshold for call sites that get inlined without the cost-benefit analysis";
  sub_C53130(&qword_5030EE0);
  return __cxa_atexit(sub_B2B680, &qword_5030EE0, &qword_4A427C0);
}
