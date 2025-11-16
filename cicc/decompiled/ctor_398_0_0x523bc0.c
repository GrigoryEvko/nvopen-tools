// Function: ctor_398_0
// Address: 0x523bc0
//
int ctor_398_0()
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
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  int v25; // [rsp+10h] [rbp-50h] BYREF
  int v26; // [rsp+14h] [rbp-4Ch] BYREF
  int *v27; // [rsp+18h] [rbp-48h] BYREF
  const char *v28; // [rsp+20h] [rbp-40h] BYREF
  __int64 v29; // [rsp+28h] [rbp-38h]

  qword_4FE6200 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE6250 = 0x100000000LL;
  dword_4FE620C &= 0x8000u;
  word_4FE6210 = 0;
  qword_4FE6218 = 0;
  qword_4FE6220 = 0;
  dword_4FE6208 = v0;
  qword_4FE6228 = 0;
  qword_4FE6230 = 0;
  qword_4FE6238 = 0;
  qword_4FE6240 = 0;
  qword_4FE6248 = (__int64)&unk_4FE6258;
  qword_4FE6260 = 0;
  qword_4FE6268 = (__int64)&unk_4FE6280;
  qword_4FE6270 = 1;
  dword_4FE6278 = 0;
  byte_4FE627C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FE6250;
  v3 = (unsigned int)qword_4FE6250 + 1LL;
  if ( v3 > HIDWORD(qword_4FE6250) )
  {
    sub_C8D5F0((char *)&unk_4FE6258 - 16, &unk_4FE6258, v3, 8);
    v2 = (unsigned int)qword_4FE6250;
  }
  *(_QWORD *)(qword_4FE6248 + 8 * v2) = v1;
  LODWORD(qword_4FE6250) = qword_4FE6250 + 1;
  qword_4FE6288 = 0;
  qword_4FE6290 = (__int64)&unk_49D9748;
  qword_4FE6298 = 0;
  qword_4FE6200 = (__int64)&unk_49DC090;
  qword_4FE62A0 = (__int64)&unk_49DC1D0;
  qword_4FE62C0 = (__int64)nullsub_23;
  qword_4FE62B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE6200, "disable-icp", 11);
  LOWORD(qword_4FE6298) = 256;
  LOBYTE(qword_4FE6288) = 0;
  qword_4FE6230 = 31;
  LOBYTE(dword_4FE620C) = dword_4FE620C & 0x9F | 0x20;
  qword_4FE6228 = (__int64)"Disable indirect call promotion";
  sub_C53130(&qword_4FE6200);
  __cxa_atexit(sub_984900, &qword_4FE6200, &qword_4A427C0);
  v29 = 45;
  v28 = "Max number of promotions for this compilation";
  v26 = 1;
  v25 = 0;
  v27 = &v25;
  sub_2445210(&unk_4FE6120, "icp-cutoff", &v27, &v26, &v28);
  __cxa_atexit(sub_984970, &unk_4FE6120, &qword_4A427C0);
  v29 = 52;
  v28 = "Skip Callsite up to this number for this compilation";
  v26 = 1;
  v25 = 0;
  v27 = &v25;
  sub_2445210(&unk_4FE6040, "icp-csskip", &v27, &v26, &v28);
  __cxa_atexit(sub_984970, &unk_4FE6040, &qword_4A427C0);
  qword_4FE5F60 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE5FB0 = 0x100000000LL;
  dword_4FE5F6C &= 0x8000u;
  qword_4FE5FA8 = (__int64)&unk_4FE5FB8;
  word_4FE5F70 = 0;
  qword_4FE5F78 = 0;
  dword_4FE5F68 = v4;
  qword_4FE5F80 = 0;
  qword_4FE5F88 = 0;
  qword_4FE5F90 = 0;
  qword_4FE5F98 = 0;
  qword_4FE5FA0 = 0;
  qword_4FE5FC0 = 0;
  qword_4FE5FC8 = (__int64)&unk_4FE5FE0;
  qword_4FE5FD0 = 1;
  dword_4FE5FD8 = 0;
  byte_4FE5FDC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FE5FB0;
  if ( (unsigned __int64)(unsigned int)qword_4FE5FB0 + 1 > HIDWORD(qword_4FE5FB0) )
  {
    v23 = v5;
    sub_C8D5F0((char *)&unk_4FE5FB8 - 16, &unk_4FE5FB8, (unsigned int)qword_4FE5FB0 + 1LL, 8);
    v6 = (unsigned int)qword_4FE5FB0;
    v5 = v23;
  }
  *(_QWORD *)(qword_4FE5FA8 + 8 * v6) = v5;
  LODWORD(qword_4FE5FB0) = qword_4FE5FB0 + 1;
  qword_4FE5FE8 = 0;
  qword_4FE5FF0 = (__int64)&unk_49D9748;
  qword_4FE5FF8 = 0;
  qword_4FE5F60 = (__int64)&unk_49DC090;
  qword_4FE6000 = (__int64)&unk_49DC1D0;
  qword_4FE6020 = (__int64)nullsub_23;
  qword_4FE6018 = (__int64)sub_984030;
  sub_C53080(&qword_4FE5F60, "icp-lto", 7);
  LOBYTE(qword_4FE5FE8) = 0;
  LOWORD(qword_4FE5FF8) = 256;
  qword_4FE5F90 = 39;
  LOBYTE(dword_4FE5F6C) = dword_4FE5F6C & 0x9F | 0x20;
  qword_4FE5F88 = (__int64)"Run indirect-call promotion in LTO mode";
  sub_C53130(&qword_4FE5F60);
  __cxa_atexit(sub_984900, &qword_4FE5F60, &qword_4A427C0);
  v29 = 45;
  v28 = "Run indirect-call promotion in SamplePGO mode";
  v26 = 1;
  LOBYTE(v25) = 0;
  v27 = &v25;
  sub_23A1BD0(&unk_4FE5E80, "icp-samplepgo", &v27, &v26, &v28);
  __cxa_atexit(sub_984900, &unk_4FE5E80, &qword_4A427C0);
  v29 = 54;
  v28 = "Run indirect-call promotion for call instructions only";
  v26 = 1;
  LOBYTE(v25) = 0;
  v27 = &v25;
  sub_23A1BD0(&unk_4FE5DA0, "icp-call-only", &v27, &v26, &v28);
  __cxa_atexit(sub_984900, &unk_4FE5DA0, &qword_4A427C0);
  qword_4FE5CC0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE5D10 = 0x100000000LL;
  dword_4FE5CCC &= 0x8000u;
  word_4FE5CD0 = 0;
  qword_4FE5D08 = (__int64)&unk_4FE5D18;
  qword_4FE5CD8 = 0;
  dword_4FE5CC8 = v7;
  qword_4FE5CE0 = 0;
  qword_4FE5CE8 = 0;
  qword_4FE5CF0 = 0;
  qword_4FE5CF8 = 0;
  qword_4FE5D00 = 0;
  qword_4FE5D20 = 0;
  qword_4FE5D28 = (__int64)&unk_4FE5D40;
  qword_4FE5D30 = 1;
  dword_4FE5D38 = 0;
  byte_4FE5D3C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FE5D10;
  if ( (unsigned __int64)(unsigned int)qword_4FE5D10 + 1 > HIDWORD(qword_4FE5D10) )
  {
    v24 = v8;
    sub_C8D5F0((char *)&unk_4FE5D18 - 16, &unk_4FE5D18, (unsigned int)qword_4FE5D10 + 1LL, 8);
    v9 = (unsigned int)qword_4FE5D10;
    v8 = v24;
  }
  *(_QWORD *)(qword_4FE5D08 + 8 * v9) = v8;
  LODWORD(qword_4FE5D10) = qword_4FE5D10 + 1;
  qword_4FE5D48 = 0;
  qword_4FE5D50 = (__int64)&unk_49D9748;
  qword_4FE5D58 = 0;
  qword_4FE5CC0 = (__int64)&unk_49DC090;
  qword_4FE5D60 = (__int64)&unk_49DC1D0;
  qword_4FE5D80 = (__int64)nullsub_23;
  qword_4FE5D78 = (__int64)sub_984030;
  sub_C53080(&qword_4FE5CC0, "icp-invoke-only", 15);
  LOWORD(qword_4FE5D58) = 256;
  LOBYTE(qword_4FE5D48) = 0;
  qword_4FE5CF0 = 55;
  LOBYTE(dword_4FE5CCC) = dword_4FE5CCC & 0x9F | 0x20;
  qword_4FE5CE8 = (__int64)"Run indirect-call promotion for invoke instruction only";
  sub_C53130(&qword_4FE5CC0);
  __cxa_atexit(sub_984900, &qword_4FE5CC0, &qword_4A427C0);
  v29 = 36;
  v28 = "Dump IR after transformation happens";
  v26 = 1;
  LOBYTE(v25) = 0;
  v27 = &v25;
  sub_23A1BD0(&unk_4FE5BE0, "icp-dumpafter", &v27, &v26, &v28);
  __cxa_atexit(sub_984900, &unk_4FE5BE0, &qword_4A427C0);
  qword_4FE5B00 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE5B50 = 0x100000000LL;
  dword_4FE5B0C &= 0x8000u;
  word_4FE5B10 = 0;
  qword_4FE5B18 = 0;
  qword_4FE5B20 = 0;
  dword_4FE5B08 = v10;
  qword_4FE5B28 = 0;
  qword_4FE5B30 = 0;
  qword_4FE5B38 = 0;
  qword_4FE5B40 = 0;
  qword_4FE5B48 = (__int64)&unk_4FE5B58;
  qword_4FE5B60 = 0;
  qword_4FE5B68 = (__int64)&unk_4FE5B80;
  qword_4FE5B70 = 1;
  dword_4FE5B78 = 0;
  byte_4FE5B7C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FE5B50;
  v13 = (unsigned int)qword_4FE5B50 + 1LL;
  if ( v13 > HIDWORD(qword_4FE5B50) )
  {
    sub_C8D5F0((char *)&unk_4FE5B58 - 16, &unk_4FE5B58, v13, 8);
    v12 = (unsigned int)qword_4FE5B50;
  }
  *(_QWORD *)(qword_4FE5B48 + 8 * v12) = v11;
  LODWORD(qword_4FE5B50) = qword_4FE5B50 + 1;
  qword_4FE5B88 = 0;
  qword_4FE5B90 = (__int64)&unk_49E5940;
  qword_4FE5B98 = 0;
  qword_4FE5B00 = (__int64)&unk_49E5960;
  qword_4FE5BA0 = (__int64)&unk_49DC320;
  qword_4FE5BC0 = (__int64)nullsub_385;
  qword_4FE5BB8 = (__int64)sub_1038930;
  sub_C53080(&qword_4FE5B00, "icp-vtable-percentage-threshold", 31);
  BYTE4(qword_4FE5B98) = 1;
  LODWORD(qword_4FE5B88) = 1065269330;
  LODWORD(qword_4FE5B98) = 1065269330;
  LOBYTE(dword_4FE5B0C) = dword_4FE5B0C & 0x9F | 0x20;
  qword_4FE5B28 = (__int64)"The percentage threshold of vtable-count / function-count for cost-benefit analysis.";
  qword_4FE5B30 = 84;
  sub_C53130(&qword_4FE5B00);
  __cxa_atexit(sub_1038DB0, &qword_4FE5B00, &qword_4A427C0);
  qword_4FE5A20 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FE5A9C = 1;
  qword_4FE5A70 = 0x100000000LL;
  dword_4FE5A2C &= 0x8000u;
  qword_4FE5A38 = 0;
  qword_4FE5A40 = 0;
  qword_4FE5A48 = 0;
  dword_4FE5A28 = v14;
  word_4FE5A30 = 0;
  qword_4FE5A50 = 0;
  qword_4FE5A58 = 0;
  qword_4FE5A60 = 0;
  qword_4FE5A68 = (__int64)&unk_4FE5A78;
  qword_4FE5A80 = 0;
  qword_4FE5A88 = (__int64)&unk_4FE5AA0;
  qword_4FE5A90 = 1;
  dword_4FE5A98 = 0;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FE5A70;
  v17 = (unsigned int)qword_4FE5A70 + 1LL;
  if ( v17 > HIDWORD(qword_4FE5A70) )
  {
    sub_C8D5F0((char *)&unk_4FE5A78 - 16, &unk_4FE5A78, v17, 8);
    v16 = (unsigned int)qword_4FE5A70;
  }
  *(_QWORD *)(qword_4FE5A68 + 8 * v16) = v15;
  LODWORD(qword_4FE5A70) = qword_4FE5A70 + 1;
  qword_4FE5AA8 = 0;
  qword_4FE5AB0 = (__int64)&unk_49DA090;
  qword_4FE5AB8 = 0;
  qword_4FE5A20 = (__int64)&unk_49DBF90;
  qword_4FE5AC0 = (__int64)&unk_49DC230;
  qword_4FE5AE0 = (__int64)nullsub_58;
  qword_4FE5AD8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE5A20, "icp-max-num-vtable-last-candidate", 33);
  LODWORD(qword_4FE5AA8) = 1;
  BYTE4(qword_4FE5AB8) = 1;
  LODWORD(qword_4FE5AB8) = 1;
  qword_4FE5A50 = 52;
  LOBYTE(dword_4FE5A2C) = dword_4FE5A2C & 0x9F | 0x20;
  qword_4FE5A48 = (__int64)"The maximum number of vtable for the last candidate.";
  sub_C53130(&qword_4FE5A20);
  __cxa_atexit(sub_B2B680, &qword_4FE5A20, &qword_4A427C0);
  qword_4FE5920 = (__int64)&unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE5938 = 0;
  qword_4FE5940 = 0;
  qword_4FE5948 = 0;
  qword_4FE5950 = 0;
  dword_4FE592C = dword_4FE592C & 0x8000 | 1;
  word_4FE5930 = 0;
  qword_4FE5970 = 0x100000000LL;
  dword_4FE5928 = v18;
  qword_4FE5958 = 0;
  qword_4FE5960 = 0;
  qword_4FE5968 = (__int64)&unk_4FE5978;
  qword_4FE5980 = 0;
  qword_4FE5988 = (__int64)&unk_4FE59A0;
  qword_4FE5990 = 1;
  dword_4FE5998 = 0;
  byte_4FE599C = 1;
  v19 = sub_C57470();
  v20 = (unsigned int)qword_4FE5970;
  v21 = (unsigned int)qword_4FE5970 + 1LL;
  if ( v21 > HIDWORD(qword_4FE5970) )
  {
    sub_C8D5F0((char *)&unk_4FE5978 - 16, &unk_4FE5978, v21, 8);
    v20 = (unsigned int)qword_4FE5970;
  }
  *(_QWORD *)(qword_4FE5968 + 8 * v20) = v19;
  LODWORD(qword_4FE5970) = qword_4FE5970 + 1;
  qword_4FE59A8 = 0;
  qword_4FE5920 = (__int64)&unk_49DAD08;
  qword_4FE59B0 = 0;
  qword_4FE59B8 = 0;
  qword_4FE59F8 = (__int64)&unk_49DC350;
  qword_4FE59C0 = 0;
  qword_4FE5A18 = (__int64)nullsub_81;
  qword_4FE59C8 = 0;
  qword_4FE5A10 = (__int64)sub_BB8600;
  qword_4FE59D0 = 0;
  byte_4FE59D8 = 0;
  qword_4FE59E0 = 0;
  qword_4FE59E8 = 0;
  qword_4FE59F0 = 0;
  sub_C53080(&qword_4FE5920, "icp-ignored-base-types", 22);
  qword_4FE5950 = 322;
  LOBYTE(dword_4FE592C) = dword_4FE592C & 0x9F | 0x20;
  qword_4FE5948 = (__int64)"A list of mangled vtable type info names. Classes specified by the type info names and their "
                           "derived ones will not be vtable-ICP'ed. Useful when the profiled types and actual types in th"
                           "e optimized binary could be different due to profiling limitations. Type info names are those"
                           " string literals used in LLVM type metadata";
  sub_C53130(&qword_4FE5920);
  return __cxa_atexit(sub_BB89D0, &qword_4FE5920, &qword_4A427C0);
}
