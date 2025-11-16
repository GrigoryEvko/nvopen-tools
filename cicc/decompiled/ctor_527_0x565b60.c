// Function: ctor_527
// Address: 0x565b60
//
int ctor_527()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // edx
  __int64 v20; // rbx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v24; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  _QWORD v26[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v27[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v28[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v29[8]; // [rsp+50h] [rbp-40h] BYREF

  qword_5013300 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013350 = 0x100000000LL;
  word_5013310 = 0;
  dword_501330C &= 0x8000u;
  qword_5013318 = 0;
  qword_5013320 = 0;
  dword_5013308 = v0;
  qword_5013328 = 0;
  qword_5013330 = 0;
  qword_5013338 = 0;
  qword_5013340 = 0;
  qword_5013348 = (__int64)&unk_5013358;
  qword_5013360 = 0;
  qword_5013368 = (__int64)&unk_5013380;
  qword_5013370 = 1;
  dword_5013378 = 0;
  byte_501337C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5013350;
  v3 = (unsigned int)qword_5013350 + 1LL;
  if ( v3 > HIDWORD(qword_5013350) )
  {
    sub_C8D5F0((char *)&unk_5013358 - 16, &unk_5013358, v3, 8);
    v2 = (unsigned int)qword_5013350;
  }
  *(_QWORD *)(qword_5013348 + 8 * v2) = v1;
  qword_5013390 = (__int64)&unk_49D9748;
  qword_5013300 = (__int64)&unk_49DC090;
  LODWORD(qword_5013350) = qword_5013350 + 1;
  qword_5013388 = 0;
  qword_50133A0 = (__int64)&unk_49DC1D0;
  qword_5013398 = 0;
  qword_50133C0 = (__int64)nullsub_23;
  qword_50133B8 = (__int64)sub_984030;
  sub_C53080(&qword_5013300, "balance-dot-chain", 17);
  LOWORD(qword_5013398) = 256;
  LOBYTE(qword_5013388) = 0;
  qword_5013330 = 35;
  LOBYTE(dword_501330C) = dword_501330C & 0x9F | 0x20;
  qword_5013328 = (__int64)"Balance the chain of dot operations";
  sub_C53130(&qword_5013300);
  __cxa_atexit(sub_984900, &qword_5013300, &qword_4A427C0);
  qword_5013220 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013270 = 0x100000000LL;
  dword_501322C &= 0x8000u;
  qword_5013268 = (__int64)&unk_5013278;
  word_5013230 = 0;
  qword_5013238 = 0;
  dword_5013228 = v4;
  qword_5013240 = 0;
  qword_5013248 = 0;
  qword_5013250 = 0;
  qword_5013258 = 0;
  qword_5013260 = 0;
  qword_5013280 = 0;
  qword_5013288 = (__int64)&unk_50132A0;
  qword_5013290 = 1;
  dword_5013298 = 0;
  byte_501329C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5013270;
  v7 = (unsigned int)qword_5013270 + 1LL;
  if ( v7 > HIDWORD(qword_5013270) )
  {
    sub_C8D5F0((char *)&unk_5013278 - 16, &unk_5013278, v7, 8);
    v6 = (unsigned int)qword_5013270;
  }
  *(_QWORD *)(qword_5013268 + 8 * v6) = v5;
  LODWORD(qword_5013270) = qword_5013270 + 1;
  qword_50132A8 = 0;
  qword_50132B0 = (__int64)&unk_49D9728;
  qword_50132B8 = 0;
  qword_5013220 = (__int64)&unk_49DBF10;
  qword_50132C0 = (__int64)&unk_49DC290;
  qword_50132E0 = (__int64)nullsub_24;
  qword_50132D8 = (__int64)sub_984050;
  sub_C53080(&qword_5013220, "max-chain-width", 15);
  LODWORD(qword_50132A8) = 2;
  BYTE4(qword_50132B8) = 1;
  LODWORD(qword_50132B8) = 2;
  qword_5013250 = 54;
  LOBYTE(dword_501322C) = dword_501322C & 0x9F | 0x20;
  qword_5013248 = (__int64)"The width of the tree to use while balancing dot chain";
  sub_C53130(&qword_5013220);
  __cxa_atexit(sub_984970, &qword_5013220, &qword_4A427C0);
  qword_5013140 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013190 = 0x100000000LL;
  dword_501314C &= 0x8000u;
  qword_5013188 = (__int64)&unk_5013198;
  word_5013150 = 0;
  qword_5013158 = 0;
  dword_5013148 = v8;
  qword_5013160 = 0;
  qword_5013168 = 0;
  qword_5013170 = 0;
  qword_5013178 = 0;
  qword_5013180 = 0;
  qword_50131A0 = 0;
  qword_50131A8 = (__int64)&unk_50131C0;
  qword_50131B0 = 1;
  dword_50131B8 = 0;
  byte_50131BC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5013190;
  if ( (unsigned __int64)(unsigned int)qword_5013190 + 1 > HIDWORD(qword_5013190) )
  {
    v25 = v9;
    sub_C8D5F0((char *)&unk_5013198 - 16, &unk_5013198, (unsigned int)qword_5013190 + 1LL, 8);
    v10 = (unsigned int)qword_5013190;
    v9 = v25;
  }
  *(_QWORD *)(qword_5013188 + 8 * v10) = v9;
  LODWORD(qword_5013190) = qword_5013190 + 1;
  qword_50131C8 = 0;
  qword_50131D0 = (__int64)&unk_49D9728;
  qword_50131D8 = 0;
  qword_5013140 = (__int64)&unk_49DBF10;
  qword_50131E0 = (__int64)&unk_49DC290;
  qword_5013200 = (__int64)nullsub_24;
  qword_50131F8 = (__int64)sub_984050;
  sub_C53080(&qword_5013140, "max-chain-length", 16);
  LODWORD(qword_50131C8) = 64;
  BYTE4(qword_50131D8) = 1;
  LODWORD(qword_50131D8) = 64;
  qword_5013170 = 66;
  LOBYTE(dword_501314C) = dword_501314C & 0x9F | 0x20;
  qword_5013168 = (__int64)"Max Length of the chain of operations selected for idpa generation";
  sub_C53130(&qword_5013140);
  __cxa_atexit(sub_984970, &qword_5013140, &qword_4A427C0);
  qword_5013060 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50130B0 = 0x100000000LL;
  dword_501306C &= 0x8000u;
  qword_50130A8 = (__int64)&unk_50130B8;
  word_5013070 = 0;
  qword_5013078 = 0;
  dword_5013068 = v11;
  qword_5013080 = 0;
  qword_5013088 = 0;
  qword_5013090 = 0;
  qword_5013098 = 0;
  qword_50130A0 = 0;
  qword_50130C0 = 0;
  qword_50130C8 = (__int64)&unk_50130E0;
  qword_50130D0 = 1;
  dword_50130D8 = 0;
  byte_50130DC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_50130B0;
  v14 = (unsigned int)qword_50130B0 + 1LL;
  if ( v14 > HIDWORD(qword_50130B0) )
  {
    sub_C8D5F0((char *)&unk_50130B8 - 16, &unk_50130B8, v14, 8);
    v13 = (unsigned int)qword_50130B0;
  }
  *(_QWORD *)(qword_50130A8 + 8 * v13) = v12;
  qword_50130F0 = (__int64)&unk_49D9748;
  qword_5013060 = (__int64)&unk_49DC090;
  LODWORD(qword_50130B0) = qword_50130B0 + 1;
  qword_50130E8 = 0;
  qword_5013100 = (__int64)&unk_49DC1D0;
  qword_50130F8 = 0;
  qword_5013120 = (__int64)nullsub_23;
  qword_5013118 = (__int64)sub_984030;
  sub_C53080(&qword_5013060, "aggressive-no-sink", 18);
  LOWORD(qword_50130F8) = 257;
  LOBYTE(qword_50130E8) = 1;
  qword_5013090 = 31;
  LOBYTE(dword_501306C) = dword_501306C & 0x9F | 0x20;
  qword_5013088 = (__int64)"Sink all generated instructions";
  sub_C53130(&qword_5013060);
  __cxa_atexit(sub_984900, &qword_5013060, &qword_4A427C0);
  qword_5012F80 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5012FD0 = 0x100000000LL;
  dword_5012F8C &= 0x8000u;
  qword_5012FC8 = (__int64)&unk_5012FD8;
  word_5012F90 = 0;
  qword_5012F98 = 0;
  dword_5012F88 = v15;
  qword_5012FA0 = 0;
  qword_5012FA8 = 0;
  qword_5012FB0 = 0;
  qword_5012FB8 = 0;
  qword_5012FC0 = 0;
  qword_5012FE0 = 0;
  qword_5012FE8 = (__int64)&unk_5013000;
  qword_5012FF0 = 1;
  dword_5012FF8 = 0;
  byte_5012FFC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5012FD0;
  v18 = (unsigned int)qword_5012FD0 + 1LL;
  if ( v18 > HIDWORD(qword_5012FD0) )
  {
    sub_C8D5F0((char *)&unk_5012FD8 - 16, &unk_5012FD8, v18, 8);
    v17 = (unsigned int)qword_5012FD0;
  }
  *(_QWORD *)(qword_5012FC8 + 8 * v17) = v16;
  qword_5013010 = (__int64)&unk_49D9748;
  qword_5012F80 = (__int64)&unk_49DC090;
  LODWORD(qword_5012FD0) = qword_5012FD0 + 1;
  qword_5013008 = 0;
  qword_5013020 = (__int64)&unk_49DC1D0;
  qword_5013018 = 0;
  qword_5013040 = (__int64)nullsub_23;
  qword_5013038 = (__int64)sub_984030;
  sub_C53080(&qword_5012F80, "enable-dot", 10);
  LOWORD(qword_5013018) = 257;
  qword_5012FA8 = (__int64)"Enable Dot Transformation";
  qword_5012FB0 = 25;
  LOBYTE(qword_5013008) = 1;
  sub_C53130(&qword_5012F80);
  __cxa_atexit(sub_984900, &qword_5012F80, &qword_4A427C0);
  v28[0] = v29;
  v24 = sub_C60B10();
  sub_2CAF610(v28, "Controls dot transformations.");
  v26[0] = v27;
  sub_2CAF610(v26, "dot-counter");
  sub_CF9810(v24, v26, v28);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0(v26[0], v27[0] + 1LL);
  if ( (_QWORD *)v28[0] != v29 )
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  qword_5012EA0 = (__int64)&unk_49DC150;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5012F1C = 1;
  qword_5012EF0 = 0x100000000LL;
  dword_5012EAC &= 0x8000u;
  qword_5012EB8 = 0;
  qword_5012EC0 = 0;
  qword_5012EC8 = 0;
  dword_5012EA8 = v19;
  word_5012EB0 = 0;
  qword_5012ED0 = 0;
  qword_5012ED8 = 0;
  qword_5012EE0 = 0;
  qword_5012EE8 = (__int64)&unk_5012EF8;
  qword_5012F00 = 0;
  qword_5012F08 = (__int64)&unk_5012F20;
  qword_5012F10 = 1;
  dword_5012F18 = 0;
  v20 = sub_C57470();
  v21 = (unsigned int)qword_5012EF0;
  v22 = (unsigned int)qword_5012EF0 + 1LL;
  if ( v22 > HIDWORD(qword_5012EF0) )
  {
    sub_C8D5F0((char *)&unk_5012EF8 - 16, &unk_5012EF8, v22, 8);
    v21 = (unsigned int)qword_5012EF0;
  }
  *(_QWORD *)(qword_5012EE8 + 8 * v21) = v20;
  qword_5012F30 = (__int64)&unk_49D9748;
  qword_5012EA0 = (__int64)&unk_49DC090;
  LODWORD(qword_5012EF0) = qword_5012EF0 + 1;
  qword_5012F28 = 0;
  qword_5012F40 = (__int64)&unk_49DC1D0;
  qword_5012F38 = 0;
  qword_5012F60 = (__int64)nullsub_23;
  qword_5012F58 = (__int64)sub_984030;
  sub_C53080(&qword_5012EA0, "enable-fma-to-ffma2", 19);
  LOBYTE(qword_5012F28) = 0;
  LOWORD(qword_5012F38) = 256;
  qword_5012ED0 = 36;
  LOBYTE(dword_5012EAC) = dword_5012EAC & 0x9F | 0x20;
  qword_5012EC8 = (__int64)"Enable 2xFMA to FFMA2 transformation";
  sub_C53130(&qword_5012EA0);
  return __cxa_atexit(sub_984900, &qword_5012EA0, &qword_4A427C0);
}
