// Function: ctor_531_0
// Address: 0x567c20
//
int ctor_531_0()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // edx
  __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+8h] [rbp-78h]
  char v38; // [rsp+13h] [rbp-6Dh] BYREF
  int v39; // [rsp+14h] [rbp-6Ch] BYREF
  char *v40; // [rsp+18h] [rbp-68h] BYREF
  const char *v41; // [rsp+20h] [rbp-60h] BYREF
  __int64 v42; // [rsp+28h] [rbp-58h]
  char v43; // [rsp+40h] [rbp-40h]
  char v44; // [rsp+41h] [rbp-3Fh]

  sub_2208040(&unk_5014628);
  __cxa_atexit(sub_2208810, &unk_5014628, &qword_4A427C0);
  v38 = 1;
  v41 = "Enable handling alloca unconditionally";
  v40 = &v38;
  v42 = 38;
  v39 = 1;
  sub_2CE0720(&unk_5014560, &v39, "process-alloca-always", &v41, &v40);
  __cxa_atexit(sub_984900, &unk_5014560, &qword_4A427C0);
  v38 = 1;
  v41 = "Enable Memory Space Optimization for Wmma";
  v40 = &v38;
  v42 = 41;
  v39 = 1;
  sub_2CE0720(&unk_5014480, &v39, "wmma-memory-space-opt", &v41, &v40);
  __cxa_atexit(sub_984900, &unk_5014480, &qword_4A427C0);
  qword_50143A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501441C = 1;
  qword_50143F0 = 0x100000000LL;
  dword_50143AC &= 0x8000u;
  qword_50143B8 = 0;
  qword_50143C0 = 0;
  qword_50143C8 = 0;
  dword_50143A8 = v0;
  word_50143B0 = 0;
  qword_50143D0 = 0;
  qword_50143D8 = 0;
  qword_50143E0 = 0;
  qword_50143E8 = (__int64)&unk_50143F8;
  qword_5014400 = 0;
  qword_5014408 = (__int64)&unk_5014420;
  qword_5014410 = 1;
  dword_5014418 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50143F0;
  v3 = (unsigned int)qword_50143F0 + 1LL;
  if ( v3 > HIDWORD(qword_50143F0) )
  {
    sub_C8D5F0((char *)&unk_50143F8 - 16, &unk_50143F8, v3, 8);
    v2 = (unsigned int)qword_50143F0;
  }
  *(_QWORD *)(qword_50143E8 + 8 * v2) = v1;
  qword_5014430 = (__int64)&unk_49D9748;
  LODWORD(qword_50143F0) = qword_50143F0 + 1;
  qword_5014428 = 0;
  qword_50143A0 = (__int64)&unk_49DC090;
  qword_5014440 = (__int64)&unk_49DC1D0;
  qword_5014438 = 0;
  qword_5014460 = (__int64)nullsub_23;
  qword_5014458 = (__int64)sub_984030;
  LOBYTE(dword_50143AC) = dword_50143AC & 0x9F | 0x20;
  sub_C53080(&qword_50143A0, "process-builtin-assume", 22);
  qword_50143D0 = 45;
  qword_50143C8 = (__int64)"Process __builtin_assume(__is*(p)) assertions";
  LOWORD(qword_5014438) = 257;
  LOBYTE(qword_5014428) = 1;
  sub_C53130(&qword_50143A0);
  __cxa_atexit(sub_984900, &qword_50143A0, &qword_4A427C0);
  qword_50142C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50142CC &= 0x8000u;
  word_50142D0 = 0;
  qword_5014310 = 0x100000000LL;
  qword_5014308 = (__int64)&unk_5014318;
  qword_50142D8 = 0;
  qword_50142E0 = 0;
  dword_50142C8 = v4;
  qword_50142E8 = 0;
  qword_50142F0 = 0;
  qword_50142F8 = 0;
  qword_5014300 = 0;
  qword_5014320 = 0;
  qword_5014328 = (__int64)&unk_5014340;
  qword_5014330 = 1;
  dword_5014338 = 0;
  byte_501433C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5014310;
  if ( (unsigned __int64)(unsigned int)qword_5014310 + 1 > HIDWORD(qword_5014310) )
  {
    v36 = v5;
    sub_C8D5F0((char *)&unk_5014318 - 16, &unk_5014318, (unsigned int)qword_5014310 + 1LL, 8);
    v6 = (unsigned int)qword_5014310;
    v5 = v36;
  }
  *(_QWORD *)(qword_5014308 + 8 * v6) = v5;
  LODWORD(qword_5014310) = qword_5014310 + 1;
  qword_5014348 = 0;
  qword_5014350 = (__int64)&unk_49DA090;
  qword_5014358 = 0;
  qword_50142C0 = (__int64)&unk_49DBF90;
  qword_5014360 = (__int64)&unk_49DC230;
  qword_5014380 = (__int64)nullsub_58;
  qword_5014378 = (__int64)sub_B2B5F0;
  LOBYTE(dword_50142CC) = dword_50142CC & 0x9F | 0x20;
  sub_C53080(&qword_50142C0, "dump-process-builtin-assume", 27);
  qword_50142F0 = 49;
  qword_50142E8 = (__int64)"Dump traces from __builtin_assume(...) processing";
  LODWORD(qword_5014348) = 0;
  BYTE4(qword_5014358) = 1;
  LODWORD(qword_5014358) = 0;
  sub_C53130(&qword_50142C0);
  __cxa_atexit(sub_B2B680, &qword_50142C0, &qword_4A427C0);
  qword_50141E0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50141EC &= 0x8000u;
  word_50141F0 = 0;
  qword_5014230 = 0x100000000LL;
  qword_5014228 = (__int64)&unk_5014238;
  qword_50141F8 = 0;
  qword_5014200 = 0;
  dword_50141E8 = v7;
  qword_5014208 = 0;
  qword_5014210 = 0;
  qword_5014218 = 0;
  qword_5014220 = 0;
  qword_5014240 = 0;
  qword_5014248 = (__int64)&unk_5014260;
  qword_5014250 = 1;
  dword_5014258 = 0;
  byte_501425C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5014230;
  if ( (unsigned __int64)(unsigned int)qword_5014230 + 1 > HIDWORD(qword_5014230) )
  {
    v37 = v8;
    sub_C8D5F0((char *)&unk_5014238 - 16, &unk_5014238, (unsigned int)qword_5014230 + 1LL, 8);
    v9 = (unsigned int)qword_5014230;
    v8 = v37;
  }
  *(_QWORD *)(qword_5014228 + 8 * v9) = v8;
  qword_5014270 = (__int64)&unk_49D9748;
  LODWORD(qword_5014230) = qword_5014230 + 1;
  qword_5014268 = 0;
  qword_50141E0 = (__int64)&unk_49DC090;
  qword_5014280 = (__int64)&unk_49DC1D0;
  qword_5014278 = 0;
  qword_50142A0 = (__int64)nullsub_23;
  qword_5014298 = (__int64)sub_984030;
  LOBYTE(dword_50141EC) = dword_50141EC & 0x9F | 0x20;
  sub_C53080(&qword_50141E0, "strong-global-assumptions", 25);
  qword_5014210 = 76;
  qword_5014208 = (__int64)"Make stronger assumptions that const buffer pointers always point to globals";
  LOWORD(qword_5014278) = 257;
  LOBYTE(qword_5014268) = 1;
  sub_C53130(&qword_50141E0);
  __cxa_atexit(sub_984900, &qword_50141E0, &qword_4A427C0);
  qword_5014100 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501410C &= 0x8000u;
  word_5014110 = 0;
  qword_5014150 = 0x100000000LL;
  qword_5014148 = (__int64)&unk_5014158;
  qword_5014118 = 0;
  qword_5014120 = 0;
  dword_5014108 = v10;
  qword_5014128 = 0;
  qword_5014130 = 0;
  qword_5014138 = 0;
  qword_5014140 = 0;
  qword_5014160 = 0;
  qword_5014168 = (__int64)&unk_5014180;
  qword_5014170 = 1;
  dword_5014178 = 0;
  byte_501417C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5014150;
  if ( (unsigned __int64)(unsigned int)qword_5014150 + 1 > HIDWORD(qword_5014150) )
  {
    v35 = v11;
    sub_C8D5F0((char *)&unk_5014158 - 16, &unk_5014158, (unsigned int)qword_5014150 + 1LL, 8);
    v12 = (unsigned int)qword_5014150;
    v11 = v35;
  }
  *(_QWORD *)(qword_5014148 + 8 * v12) = v11;
  qword_5014190 = (__int64)&unk_49D9748;
  LODWORD(qword_5014150) = qword_5014150 + 1;
  byte_5014199 = 0;
  qword_5014100 = (__int64)&unk_49D9AD8;
  qword_50141A0 = (__int64)&unk_49DC1D0;
  qword_5014188 = 0;
  qword_50141C0 = (__int64)nullsub_39;
  qword_50141B8 = (__int64)sub_AA4180;
  LOBYTE(dword_501410C) = dword_501410C & 0x9F | 0x20;
  sub_C53080(&qword_5014100, "param-always-point-to-global", 28);
  qword_5014130 = 42;
  qword_5014128 = (__int64)"Parameter Pointers Always Point To Globals";
  if ( qword_5014188 )
  {
    v13 = sub_CEADF0();
    v44 = 1;
    v41 = "cl::location(x) specified more than once!";
    v43 = 3;
    sub_C53280(&qword_5014100, &v41, 0, 0, v13);
  }
  else
  {
    qword_5014188 = (__int64)&unk_50142AD;
  }
  *(_BYTE *)qword_5014188 = 1;
  unk_5014198 = 257;
  sub_C53130(&qword_5014100);
  __cxa_atexit(sub_AA4490, &qword_5014100, &qword_4A427C0);
  qword_5014020 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5014070 = 0x100000000LL;
  dword_501402C &= 0x8000u;
  word_5014030 = 0;
  qword_5014068 = (__int64)&unk_5014078;
  qword_5014038 = 0;
  dword_5014028 = v14;
  qword_5014040 = 0;
  qword_5014048 = 0;
  qword_5014050 = 0;
  qword_5014058 = 0;
  qword_5014060 = 0;
  qword_5014080 = 0;
  qword_5014088 = (__int64)&unk_50140A0;
  qword_5014090 = 1;
  dword_5014098 = 0;
  byte_501409C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_5014070;
  v17 = (unsigned int)qword_5014070 + 1LL;
  if ( v17 > HIDWORD(qword_5014070) )
  {
    sub_C8D5F0((char *)&unk_5014078 - 16, &unk_5014078, v17, 8);
    v16 = (unsigned int)qword_5014070;
  }
  *(_QWORD *)(qword_5014068 + 8 * v16) = v15;
  qword_50140B0 = (__int64)&unk_49D9748;
  LODWORD(qword_5014070) = qword_5014070 + 1;
  qword_50140A8 = 0;
  qword_5014020 = (__int64)&unk_49DC090;
  qword_50140C0 = (__int64)&unk_49DC1D0;
  qword_50140B8 = 0;
  qword_50140E0 = (__int64)nullsub_23;
  qword_50140D8 = (__int64)sub_984030;
  LOBYTE(dword_501402C) = dword_501402C & 0x9F | 0x20;
  sub_C53080(&qword_5014020, "dump-ir-before-memory-space-opt", 31);
  LOWORD(qword_50140B8) = 256;
  qword_5014048 = (__int64)"Dump LLVM IR before Memory Space Opt";
  qword_5014050 = 36;
  LOBYTE(qword_50140A8) = 0;
  sub_C53130(&qword_5014020);
  __cxa_atexit(sub_984900, &qword_5014020, &qword_4A427C0);
  qword_5013F40 = (__int64)&unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013F90 = 0x100000000LL;
  dword_5013F4C &= 0x8000u;
  qword_5013F88 = (__int64)&unk_5013F98;
  word_5013F50 = 0;
  qword_5013F58 = 0;
  dword_5013F48 = v18;
  qword_5013F60 = 0;
  qword_5013F68 = 0;
  qword_5013F70 = 0;
  qword_5013F78 = 0;
  qword_5013F80 = 0;
  qword_5013FA0 = 0;
  qword_5013FA8 = (__int64)&unk_5013FC0;
  qword_5013FB0 = 1;
  dword_5013FB8 = 0;
  byte_5013FBC = 1;
  v19 = sub_C57470();
  v20 = (unsigned int)qword_5013F90;
  v21 = (unsigned int)qword_5013F90 + 1LL;
  if ( v21 > HIDWORD(qword_5013F90) )
  {
    sub_C8D5F0((char *)&unk_5013F98 - 16, &unk_5013F98, v21, 8);
    v20 = (unsigned int)qword_5013F90;
  }
  *(_QWORD *)(qword_5013F88 + 8 * v20) = v19;
  qword_5013FD0 = (__int64)&unk_49D9748;
  LODWORD(qword_5013F90) = qword_5013F90 + 1;
  qword_5013FC8 = 0;
  qword_5013F40 = (__int64)&unk_49DC090;
  qword_5013FE0 = (__int64)&unk_49DC1D0;
  qword_5013FD8 = 0;
  qword_5014000 = (__int64)nullsub_23;
  qword_5013FF8 = (__int64)sub_984030;
  LOBYTE(dword_5013F4C) = dword_5013F4C & 0x9F | 0x20;
  sub_C53080(&qword_5013F40, "dump-ir-after-memory-space-opt", 30);
  qword_5013F70 = 35;
  LOWORD(qword_5013FD8) = 256;
  qword_5013F68 = (__int64)"Dump LLVM IR after Memory Space Opt";
  LOBYTE(qword_5013FC8) = 0;
  sub_C53130(&qword_5013F40);
  __cxa_atexit(sub_984900, &qword_5013F40, &qword_4A427C0);
  qword_5013E60 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013EB0 = 0x100000000LL;
  dword_5013E6C &= 0x8000u;
  qword_5013EA8 = (__int64)&unk_5013EB8;
  word_5013E70 = 0;
  qword_5013E78 = 0;
  dword_5013E68 = v22;
  qword_5013E80 = 0;
  qword_5013E88 = 0;
  qword_5013E90 = 0;
  qword_5013E98 = 0;
  qword_5013EA0 = 0;
  qword_5013EC0 = 0;
  qword_5013EC8 = (__int64)&unk_5013EE0;
  qword_5013ED0 = 1;
  dword_5013ED8 = 0;
  byte_5013EDC = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_5013EB0;
  v25 = (unsigned int)qword_5013EB0 + 1LL;
  if ( v25 > HIDWORD(qword_5013EB0) )
  {
    sub_C8D5F0((char *)&unk_5013EB8 - 16, &unk_5013EB8, v25, 8);
    v24 = (unsigned int)qword_5013EB0;
  }
  *(_QWORD *)(qword_5013EA8 + 8 * v24) = v23;
  qword_5013EF0 = (__int64)&unk_49D9748;
  LODWORD(qword_5013EB0) = qword_5013EB0 + 1;
  qword_5013EE8 = 0;
  qword_5013E60 = (__int64)&unk_49DC090;
  qword_5013F00 = (__int64)&unk_49DC1D0;
  qword_5013EF8 = 0;
  qword_5013F20 = (__int64)nullsub_23;
  qword_5013F18 = (__int64)sub_984030;
  LOBYTE(dword_5013E6C) = dword_5013E6C & 0x9F | 0x20;
  sub_C53080(&qword_5013E60, "track-indir-load", 16);
  LOWORD(qword_5013EF8) = 257;
  qword_5013E88 = (__int64)"Enable tracking indirect loads during Memory Space Optimization";
  qword_5013E90 = 63;
  LOBYTE(qword_5013EE8) = 1;
  sub_C53130(&qword_5013E60);
  __cxa_atexit(sub_984900, &qword_5013E60, &qword_4A427C0);
  qword_5013D80 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5013DFC = 1;
  word_5013D90 = 0;
  qword_5013DD0 = 0x100000000LL;
  dword_5013D8C &= 0x8000u;
  qword_5013DC8 = (__int64)&unk_5013DD8;
  qword_5013D98 = 0;
  dword_5013D88 = v26;
  qword_5013DA0 = 0;
  qword_5013DA8 = 0;
  qword_5013DB0 = 0;
  qword_5013DB8 = 0;
  qword_5013DC0 = 0;
  qword_5013DE0 = 0;
  qword_5013DE8 = (__int64)&unk_5013E00;
  qword_5013DF0 = 1;
  dword_5013DF8 = 0;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_5013DD0;
  v29 = (unsigned int)qword_5013DD0 + 1LL;
  if ( v29 > HIDWORD(qword_5013DD0) )
  {
    sub_C8D5F0((char *)&unk_5013DD8 - 16, &unk_5013DD8, v29, 8);
    v28 = (unsigned int)qword_5013DD0;
  }
  *(_QWORD *)(qword_5013DC8 + 8 * v28) = v27;
  LODWORD(qword_5013DD0) = qword_5013DD0 + 1;
  qword_5013E08 = 0;
  qword_5013E10 = (__int64)&unk_49D9728;
  qword_5013E18 = 0;
  qword_5013D80 = (__int64)&unk_49DBF10;
  qword_5013E20 = (__int64)&unk_49DC290;
  qword_5013E40 = (__int64)nullsub_24;
  qword_5013E38 = (__int64)sub_984050;
  LOBYTE(dword_5013D8C) = dword_5013D8C & 0x9F | 0x20;
  sub_C53080(&qword_5013D80, "mem-space-alg", 13);
  qword_5013DB0 = 66;
  qword_5013DA8 = (__int64)"Switch between different algorithms for Address Space Optimization";
  LODWORD(qword_5013E08) = 2;
  BYTE4(qword_5013E18) = 1;
  LODWORD(qword_5013E18) = 2;
  sub_C53130(&qword_5013D80);
  __cxa_atexit(sub_984970, &qword_5013D80, &qword_4A427C0);
  qword_5013CA0 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5013D1C = 1;
  qword_5013CF0 = 0x100000000LL;
  dword_5013CAC &= 0x8000u;
  qword_5013CB8 = 0;
  qword_5013CC0 = 0;
  qword_5013CC8 = 0;
  dword_5013CA8 = v30;
  word_5013CB0 = 0;
  qword_5013CD0 = 0;
  qword_5013CD8 = 0;
  qword_5013CE0 = 0;
  qword_5013CE8 = (__int64)&unk_5013CF8;
  qword_5013D00 = 0;
  qword_5013D08 = (__int64)&unk_5013D20;
  qword_5013D10 = 1;
  dword_5013D18 = 0;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_5013CF0;
  v33 = (unsigned int)qword_5013CF0 + 1LL;
  if ( v33 > HIDWORD(qword_5013CF0) )
  {
    sub_C8D5F0((char *)&unk_5013CF8 - 16, &unk_5013CF8, v33, 8);
    v32 = (unsigned int)qword_5013CF0;
  }
  *(_QWORD *)(qword_5013CE8 + 8 * v32) = v31;
  qword_5013D30 = (__int64)&unk_49D9748;
  LODWORD(qword_5013CF0) = qword_5013CF0 + 1;
  qword_5013D28 = 0;
  qword_5013CA0 = (__int64)&unk_49DC090;
  qword_5013D40 = (__int64)&unk_49DC1D0;
  qword_5013D38 = 0;
  qword_5013D60 = (__int64)nullsub_23;
  qword_5013D58 = (__int64)sub_984030;
  LOBYTE(dword_5013CAC) = dword_5013CAC & 0x9F | 0x20;
  sub_C53080(&qword_5013CA0, "track-int2ptr", 13);
  qword_5013CD0 = 53;
  qword_5013CC8 = (__int64)"Enable tracking IntToPtr in Memory Space Optimization";
  LOBYTE(qword_5013D28) = 1;
  LOWORD(qword_5013D38) = 257;
  sub_C53130(&qword_5013CA0);
  return __cxa_atexit(sub_984900, &qword_5013CA0, &qword_4A427C0);
}
