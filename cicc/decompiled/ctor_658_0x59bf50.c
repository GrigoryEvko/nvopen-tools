// Function: ctor_658
// Address: 0x59bf50
//
int __fastcall ctor_658(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v26; // [rsp+8h] [rbp-68h]
  int v27; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 (__fastcall **v28)(_QWORD, _QWORD); // [rsp+20h] [rbp-50h] BYREF
  __int64 (__fastcall *v29)(_QWORD, _QWORD); // [rsp+28h] [rbp-48h] BYREF
  _QWORD v30[8]; // [rsp+30h] [rbp-40h] BYREF

  qword_5039D80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5039DD0 = 0x100000000LL;
  dword_5039D8C &= 0x8000u;
  word_5039D90 = 0;
  qword_5039D98 = 0;
  qword_5039DA0 = 0;
  dword_5039D88 = v4;
  qword_5039DA8 = 0;
  qword_5039DB0 = 0;
  qword_5039DB8 = 0;
  qword_5039DC0 = 0;
  qword_5039DC8 = (__int64)&unk_5039DD8;
  qword_5039DE0 = 0;
  qword_5039DE8 = (__int64)&unk_5039E00;
  qword_5039DF0 = 1;
  dword_5039DF8 = 0;
  byte_5039DFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5039DD0;
  v7 = (unsigned int)qword_5039DD0 + 1LL;
  if ( v7 > HIDWORD(qword_5039DD0) )
  {
    sub_C8D5F0((char *)&unk_5039DD8 - 16, &unk_5039DD8, v7, 8);
    v6 = (unsigned int)qword_5039DD0;
  }
  *(_QWORD *)(qword_5039DC8 + 8 * v6) = v5;
  LODWORD(qword_5039DD0) = qword_5039DD0 + 1;
  qword_5039E08 = 0;
  qword_5039E10 = (__int64)&unk_49DA090;
  qword_5039E18 = 0;
  qword_5039D80 = (__int64)&unk_49DBF90;
  qword_5039E20 = (__int64)&unk_49DC230;
  qword_5039E40 = (__int64)nullsub_58;
  qword_5039E38 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5039D80, "fast-isel-abort", 15);
  qword_5039DB0 = 238;
  LOBYTE(dword_5039D8C) = dword_5039D8C & 0x9F | 0x20;
  qword_5039DA8 = (__int64)"Enable abort calls when \"fast\" instruction selection fails to lower an instruction: 0 disab"
                           "le the abort, 1 will abort but for args, calls and terminators, 2 will also abort for argumen"
                           "t lowering, and 3 will never fallback to SelectionDAG.";
  sub_C53130(&qword_5039D80);
  __cxa_atexit(sub_B2B680, &qword_5039D80, &qword_4A427C0);
  qword_5039CA0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_5039D80, v8, v9), 1u);
  qword_5039CF0 = 0x100000000LL;
  dword_5039CAC &= 0x8000u;
  word_5039CB0 = 0;
  qword_5039CB8 = 0;
  qword_5039CC0 = 0;
  dword_5039CA8 = v10;
  qword_5039CC8 = 0;
  qword_5039CD0 = 0;
  qword_5039CD8 = 0;
  qword_5039CE0 = 0;
  qword_5039CE8 = (__int64)&unk_5039CF8;
  qword_5039D00 = 0;
  qword_5039D08 = (__int64)&unk_5039D20;
  qword_5039D10 = 1;
  dword_5039D18 = 0;
  byte_5039D1C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5039CF0;
  v13 = (unsigned int)qword_5039CF0 + 1LL;
  if ( v13 > HIDWORD(qword_5039CF0) )
  {
    sub_C8D5F0((char *)&unk_5039CF8 - 16, &unk_5039CF8, v13, 8);
    v12 = (unsigned int)qword_5039CF0;
  }
  *(_QWORD *)(qword_5039CE8 + 8 * v12) = v11;
  qword_5039D30 = (__int64)&unk_49D9748;
  qword_5039CA0 = (__int64)&unk_49DC090;
  qword_5039D40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5039CF0) = qword_5039CF0 + 1;
  qword_5039D60 = (__int64)nullsub_23;
  qword_5039D28 = 0;
  qword_5039D58 = (__int64)sub_984030;
  qword_5039D38 = 0;
  sub_C53080(&qword_5039CA0, "fast-isel-report-on-fallback", 28);
  qword_5039CD0 = 79;
  LOBYTE(dword_5039CAC) = dword_5039CAC & 0x9F | 0x20;
  qword_5039CC8 = (__int64)"Emit a diagnostic when \"fast\" instruction selection falls back to SelectionDAG.";
  sub_C53130(&qword_5039CA0);
  __cxa_atexit(sub_984900, &qword_5039CA0, &qword_4A427C0);
  qword_5039BC0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5039CA0, v14, v15), 1u);
  byte_5039C3C = 1;
  word_5039BD0 = 0;
  qword_5039C10 = 0x100000000LL;
  dword_5039BCC &= 0x8000u;
  qword_5039C08 = (__int64)&unk_5039C18;
  qword_5039BD8 = 0;
  dword_5039BC8 = v16;
  qword_5039BE0 = 0;
  qword_5039BE8 = 0;
  qword_5039BF0 = 0;
  qword_5039BF8 = 0;
  qword_5039C00 = 0;
  qword_5039C20 = 0;
  qword_5039C28 = (__int64)&unk_5039C40;
  qword_5039C30 = 1;
  dword_5039C38 = 0;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_5039C10;
  if ( (unsigned __int64)(unsigned int)qword_5039C10 + 1 > HIDWORD(qword_5039C10) )
  {
    v26 = v17;
    sub_C8D5F0((char *)&unk_5039C18 - 16, &unk_5039C18, (unsigned int)qword_5039C10 + 1LL, 8);
    v18 = (unsigned int)qword_5039C10;
    v17 = v26;
  }
  *(_QWORD *)(qword_5039C08 + 8 * v18) = v17;
  qword_5039C50 = (__int64)&unk_49D9748;
  qword_5039BC0 = (__int64)&unk_49DC090;
  qword_5039C60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5039C10) = qword_5039C10 + 1;
  qword_5039C80 = (__int64)nullsub_23;
  qword_5039C48 = 0;
  qword_5039C78 = (__int64)sub_984030;
  qword_5039C58 = 0;
  sub_C53080(&qword_5039BC0, "use-mbpi", 8);
  qword_5039BE8 = (__int64)"use Machine Branch Probability Info";
  LOWORD(qword_5039C58) = 257;
  LOBYTE(qword_5039C48) = 1;
  qword_5039BF0 = 35;
  LOBYTE(dword_5039BCC) = dword_5039BCC & 0x9F | 0x20;
  sub_C53130(&qword_5039BC0);
  __cxa_atexit(sub_984900, &qword_5039BC0, &qword_4A427C0);
  qword_5039AE0 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5039BC0, v19, v20), 1u);
  qword_5039B30 = 0x100000000LL;
  dword_5039AEC &= 0x8000u;
  word_5039AF0 = 0;
  qword_5039B28 = (__int64)&unk_5039B38;
  qword_5039AF8 = 0;
  dword_5039AE8 = v21;
  qword_5039B00 = 0;
  qword_5039B08 = 0;
  qword_5039B10 = 0;
  qword_5039B18 = 0;
  qword_5039B20 = 0;
  qword_5039B40 = 0;
  qword_5039B48 = (__int64)&unk_5039B60;
  qword_5039B50 = 1;
  dword_5039B58 = 0;
  byte_5039B5C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_5039B30;
  v24 = (unsigned int)qword_5039B30 + 1LL;
  if ( v24 > HIDWORD(qword_5039B30) )
  {
    sub_C8D5F0((char *)&unk_5039B38 - 16, &unk_5039B38, v24, 8);
    v23 = (unsigned int)qword_5039B30;
  }
  *(_QWORD *)(qword_5039B28 + 8 * v23) = v22;
  qword_5039B70 = (__int64)&unk_49D9748;
  qword_5039AE0 = (__int64)&unk_49DC090;
  qword_5039B80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5039B30) = qword_5039B30 + 1;
  qword_5039BA0 = (__int64)nullsub_23;
  qword_5039B68 = 0;
  qword_5039B98 = (__int64)sub_984030;
  qword_5039B78 = 0;
  sub_C53080(&qword_5039AE0, "dag-disable-combine", 19);
  qword_5039B10 = 35;
  LOBYTE(qword_5039B68) = 0;
  LOBYTE(dword_5039AEC) = dword_5039AEC & 0x9F | 0x20;
  qword_5039B08 = (__int64)"Disable DAG Combining optimizations";
  LOWORD(qword_5039B78) = 256;
  sub_C53130(&qword_5039AE0);
  __cxa_atexit(sub_984900, &qword_5039AE0, &qword_4A427C0);
  v30[0] = "Instruction schedulers available (before register allocation):";
  v29 = sub_341EBC0;
  v28 = &v29;
  v30[1] = 62;
  v27 = 1;
  sub_3432910(&unk_5039800, "pre-RA-sched", &v28, &v27, v30);
  __cxa_atexit(sub_341F350, &unk_5039800, &qword_4A427C0);
  sub_334D620(&unk_50397C0, "default", "Best scheduler for the target", sub_341EBC0);
  return __cxa_atexit(sub_334CAC0, &unk_50397C0, &qword_4A427C0);
}
