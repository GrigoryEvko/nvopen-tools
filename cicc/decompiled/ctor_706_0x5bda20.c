// Function: ctor_706
// Address: 0x5bda20
//
int __fastcall ctor_706(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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

  qword_5050D20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5050D70 = 0x100000000LL;
  word_5050D30 = 0;
  dword_5050D2C &= 0x8000u;
  qword_5050D38 = 0;
  qword_5050D40 = 0;
  dword_5050D28 = v4;
  qword_5050D48 = 0;
  qword_5050D50 = 0;
  qword_5050D58 = 0;
  qword_5050D60 = 0;
  qword_5050D68 = (__int64)&unk_5050D78;
  qword_5050D80 = 0;
  qword_5050D88 = (__int64)&unk_5050DA0;
  qword_5050D90 = 1;
  dword_5050D98 = 0;
  byte_5050D9C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5050D70;
  v7 = (unsigned int)qword_5050D70 + 1LL;
  if ( v7 > HIDWORD(qword_5050D70) )
  {
    sub_C8D5F0((char *)&unk_5050D78 - 16, &unk_5050D78, v7, 8);
    v6 = (unsigned int)qword_5050D70;
  }
  *(_QWORD *)(qword_5050D68 + 8 * v6) = v5;
  LODWORD(qword_5050D70) = qword_5050D70 + 1;
  qword_5050DA8 = 0;
  qword_5050DB0 = (__int64)&unk_49DC110;
  qword_5050DB8 = 0;
  qword_5050D20 = (__int64)&unk_49D97F0;
  qword_5050DC0 = (__int64)&unk_49DC200;
  qword_5050DE0 = (__int64)nullsub_26;
  qword_5050DD8 = (__int64)sub_9C26D0;
  sub_C53080(&qword_5050D20, "add-linkage-names-to-declaration-call-origins", 45);
  qword_5050D50 = 141;
  LOBYTE(dword_5050D2C) = dword_5050D2C & 0x9F | 0x20;
  qword_5050D48 = (__int64)"Add DW_AT_linkage_name to function declaration DIEs referenced by DW_AT_call_origin attribute"
                           "s. Enabled by default for -gsce debugger tuning.";
  sub_C53130(&qword_5050D20);
  __cxa_atexit(sub_9C44F0, &qword_5050D20, &qword_4A427C0);
  qword_5050C40 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_9C44F0, &qword_5050D20, v8, v9), 1u);
  byte_5050CBC = 1;
  qword_5050C90 = 0x100000000LL;
  dword_5050C4C &= 0x8000u;
  qword_5050C58 = 0;
  qword_5050C60 = 0;
  qword_5050C68 = 0;
  dword_5050C48 = v10;
  word_5050C50 = 0;
  qword_5050C70 = 0;
  qword_5050C78 = 0;
  qword_5050C80 = 0;
  qword_5050C88 = (__int64)&unk_5050C98;
  qword_5050CA0 = 0;
  qword_5050CA8 = (__int64)&unk_5050CC0;
  qword_5050CB0 = 1;
  dword_5050CB8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5050C90;
  v13 = (unsigned int)qword_5050C90 + 1LL;
  if ( v13 > HIDWORD(qword_5050C90) )
  {
    sub_C8D5F0((char *)&unk_5050C98 - 16, &unk_5050C98, v13, 8);
    v12 = (unsigned int)qword_5050C90;
  }
  *(_QWORD *)(qword_5050C88 + 8 * v12) = v11;
  LODWORD(qword_5050C90) = qword_5050C90 + 1;
  qword_5050CC8 = 0;
  qword_5050CD0 = (__int64)&unk_49D9748;
  qword_5050CD8 = 0;
  qword_5050C40 = (__int64)&unk_49DC090;
  qword_5050CE0 = (__int64)&unk_49DC1D0;
  qword_5050D00 = (__int64)nullsub_23;
  qword_5050CF8 = (__int64)sub_984030;
  sub_C53080(&qword_5050C40, "emit-func-debug-line-table-offsets", 34);
  qword_5050C70 = 105;
  LOBYTE(qword_5050CC8) = 0;
  LOBYTE(dword_5050C4C) = dword_5050C4C & 0x9F | 0x20;
  qword_5050C68 = (__int64)"Include line table offset in function's debug info and emit end sequence after each function's line data.";
  LOWORD(qword_5050CD8) = 256;
  sub_C53130(&qword_5050C40);
  return __cxa_atexit(sub_984900, &qword_5050C40, &qword_4A427C0);
}
