// Function: ctor_203_0
// Address: 0x4e1cd0
//
int ctor_203_0()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v3; // eax
  int v4; // eax
  int v6; // [rsp+20h] [rbp-100h] BYREF
  int v7; // [rsp+24h] [rbp-FCh] BYREF
  int *v8; // [rsp+28h] [rbp-F8h] BYREF
  _QWORD v9[2]; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD v10[2]; // [rsp+40h] [rbp-E0h] BYREF
  _QWORD v11[2]; // [rsp+50h] [rbp-D0h] BYREF
  int v12; // [rsp+60h] [rbp-C0h]
  const char *v13; // [rsp+68h] [rbp-B8h]
  __int64 v14; // [rsp+70h] [rbp-B0h]
  const char *v15; // [rsp+78h] [rbp-A8h]
  __int64 v16; // [rsp+80h] [rbp-A0h]
  int v17; // [rsp+88h] [rbp-98h]
  const char *v18; // [rsp+90h] [rbp-90h]
  __int64 v19; // [rsp+98h] [rbp-88h]
  char *v20; // [rsp+A0h] [rbp-80h]
  __int64 v21; // [rsp+A8h] [rbp-78h]
  int v22; // [rsp+B0h] [rbp-70h]
  const char *v23; // [rsp+B8h] [rbp-68h]
  __int64 v24; // [rsp+C0h] [rbp-60h]

  qword_4FAFA20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAFAD0 = 256;
  word_4FAFA2C &= 0xF000u;
  qword_4FAFA30 = 0;
  qword_4FAFA38 = 0;
  qword_4FAFA40 = 0;
  dword_4FAFA28 = v0;
  qword_4FAFA78 = (__int64)&unk_4FAFA98;
  qword_4FAFA80 = (__int64)&unk_4FAFA98;
  qword_4FAFAC8 = (__int64)&unk_49E74E8;
  qword_4FAFA48 = 0;
  qword_4FAFA50 = 0;
  qword_4FAFA20 = (__int64)&unk_49EEC70;
  qword_4FAFA58 = 0;
  qword_4FAFA60 = 0;
  qword_4FAFAD8 = (__int64)&unk_49EEDB0;
  qword_4FAFA68 = (__int64)qword_4FA01C0;
  qword_4FAFA70 = 0;
  qword_4FAFA88 = 4;
  dword_4FAFA90 = 0;
  byte_4FAFAB8 = 0;
  byte_4FAFAC0 = 0;
  sub_16B8280(&qword_4FAFA20, "verify-indvars", 14);
  qword_4FAFA50 = 55;
  LOBYTE(word_4FAFA2C) = word_4FAFA2C & 0x9F | 0x20;
  qword_4FAFA48 = (__int64)"Verify the ScalarEvolution result after running indvars";
  sub_16B88A0(&qword_4FAFA20);
  __cxa_atexit(sub_12EDEC0, &qword_4FAFA20, &qword_4A427C0);
  v10[0] = v11;
  v11[0] = "never";
  v13 = "never replace exit value";
  v15 = "cheap";
  v18 = "only replace exit value when the cost is cheap";
  v20 = "always";
  v23 = "always replace exit value whenever possible";
  v10[1] = 0x400000003LL;
  v8 = &v7;
  v11[1] = 5;
  v12 = 0;
  v14 = 24;
  v16 = 5;
  v17 = 1;
  v19 = 46;
  v21 = 6;
  v22 = 2;
  v24 = 43;
  v9[0] = "Choose the strategy to replace exit value in IndVarSimplify";
  v9[1] = 59;
  v7 = 1;
  v6 = 1;
  sub_1940280(&unk_4FAF7C0, "replexitval", &v6, &v8, v9, v10);
  if ( (_QWORD *)v10[0] != v11 )
    _libc_free(v10[0], "replexitval");
  __cxa_atexit(sub_193DEF0, &unk_4FAF7C0, &qword_4A427C0);
  qword_4FAF6E0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAF6EC &= 0xF000u;
  qword_4FAF6F0 = 0;
  qword_4FAF6F8 = 0;
  qword_4FAF700 = 0;
  qword_4FAF708 = 0;
  dword_4FAF6E8 = v1;
  qword_4FAF738 = (__int64)&unk_4FAF758;
  qword_4FAF740 = (__int64)&unk_4FAF758;
  word_4FAF790 = 256;
  qword_4FAF788 = (__int64)&unk_49E74E8;
  qword_4FAF6E0 = (__int64)&unk_49EEC70;
  qword_4FAF798 = (__int64)&unk_49EEDB0;
  qword_4FAF728 = (__int64)qword_4FA01C0;
  qword_4FAF710 = 0;
  qword_4FAF718 = 0;
  qword_4FAF720 = 0;
  qword_4FAF730 = 0;
  qword_4FAF748 = 4;
  dword_4FAF750 = 0;
  byte_4FAF778 = 0;
  byte_4FAF780 = 0;
  sub_16B8280(&qword_4FAF6E0, "indvars-post-increment-ranges", 29);
  word_4FAF790 = 257;
  byte_4FAF780 = 1;
  qword_4FAF710 = 61;
  LOBYTE(word_4FAF6EC) = word_4FAF6EC & 0x9F | 0x20;
  qword_4FAF708 = (__int64)"Use post increment control-dependent ranges in IndVarSimplify";
  sub_16B88A0(&qword_4FAF6E0);
  __cxa_atexit(sub_12EDEC0, &qword_4FAF6E0, &qword_4A427C0);
  qword_4FAF600 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAF6B0 = 256;
  qword_4FAF610 = 0;
  word_4FAF60C &= 0xF000u;
  qword_4FAF6A8 = (__int64)&unk_49E74E8;
  qword_4FAF600 = (__int64)&unk_49EEC70;
  dword_4FAF608 = v2;
  qword_4FAF6B8 = (__int64)&unk_49EEDB0;
  qword_4FAF648 = (__int64)qword_4FA01C0;
  qword_4FAF658 = (__int64)&unk_4FAF678;
  qword_4FAF660 = (__int64)&unk_4FAF678;
  qword_4FAF618 = 0;
  qword_4FAF620 = 0;
  qword_4FAF628 = 0;
  qword_4FAF630 = 0;
  qword_4FAF638 = 0;
  qword_4FAF640 = 0;
  qword_4FAF650 = 0;
  qword_4FAF668 = 4;
  dword_4FAF670 = 0;
  byte_4FAF698 = 0;
  byte_4FAF6A0 = 0;
  sub_16B8280((char *)&unk_4FAF678 - 120, "disable-lftr", 12);
  word_4FAF6B0 = 256;
  byte_4FAF6A0 = 0;
  qword_4FAF630 = 49;
  LOBYTE(word_4FAF60C) = word_4FAF60C & 0x9F | 0x20;
  qword_4FAF628 = (__int64)"Disable Linear Function Test Replace optimization";
  sub_16B88A0(&qword_4FAF600);
  __cxa_atexit(sub_12EDEC0, &qword_4FAF600, &qword_4A427C0);
  qword_4FAF520[0] = &unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FAF520[1]) &= 0xF000u;
  LODWORD(qword_4FAF520[1]) = v3;
  qword_4FAF520[0] = &unk_49EEC70;
  qword_4FAF520[23] = &unk_49EEDB0;
  qword_4FAF520[11] = &qword_4FAF520[15];
  qword_4FAF520[12] = &qword_4FAF520[15];
  qword_4FAF520[21] = &unk_49E74E8;
  LOWORD(qword_4FAF520[22]) = 256;
  qword_4FAF520[2] = 0;
  qword_4FAF520[3] = 0;
  qword_4FAF520[4] = 0;
  qword_4FAF520[5] = 0;
  qword_4FAF520[6] = 0;
  qword_4FAF520[7] = 0;
  qword_4FAF520[8] = 0;
  qword_4FAF520[9] = qword_4FA01C0;
  qword_4FAF520[10] = 0;
  qword_4FAF520[13] = 4;
  LODWORD(qword_4FAF520[14]) = 0;
  LOBYTE(qword_4FAF520[19]) = 0;
  LOBYTE(qword_4FAF520[20]) = 0;
  sub_16B8280(qword_4FAF520, "Disable-unknown-trip-iv", 23);
  qword_4FAF520[5] = "Disable IV-subst for unknown trip loop ";
  LOWORD(qword_4FAF520[22]) = 257;
  LOBYTE(qword_4FAF520[20]) = 1;
  qword_4FAF520[6] = 39;
  BYTE4(qword_4FAF520[1]) = BYTE4(qword_4FAF520[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4FAF520);
  __cxa_atexit(sub_12EDEC0, qword_4FAF520, &qword_4A427C0);
  qword_4FAF440[0] = &unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FAF440[1]) &= 0xF000u;
  LODWORD(qword_4FAF440[1]) = v4;
  qword_4FAF440[11] = &qword_4FAF440[15];
  qword_4FAF440[12] = &qword_4FAF440[15];
  qword_4FAF440[9] = qword_4FA01C0;
  qword_4FAF440[2] = 0;
  qword_4FAF440[21] = &unk_49E74A8;
  qword_4FAF440[3] = 0;
  qword_4FAF440[4] = 0;
  qword_4FAF440[0] = &unk_49EEAF0;
  qword_4FAF440[5] = 0;
  qword_4FAF440[6] = 0;
  qword_4FAF440[23] = &unk_49EEE10;
  qword_4FAF440[7] = 0;
  qword_4FAF440[8] = 0;
  qword_4FAF440[10] = 0;
  qword_4FAF440[13] = 4;
  LODWORD(qword_4FAF440[14]) = 0;
  LOBYTE(qword_4FAF440[19]) = 0;
  LODWORD(qword_4FAF440[20]) = 0;
  BYTE4(qword_4FAF440[22]) = 1;
  LODWORD(qword_4FAF440[22]) = 0;
  sub_16B8280(qword_4FAF440, "iv-loop-level", 13);
  BYTE4(qword_4FAF440[22]) = 1;
  LODWORD(qword_4FAF440[20]) = 1;
  qword_4FAF440[6] = 41;
  LODWORD(qword_4FAF440[22]) = 1;
  BYTE4(qword_4FAF440[1]) = BYTE4(qword_4FAF440[1]) & 0x9F | 0x20;
  qword_4FAF440[5] = "Control loop-levels to apply the IV-subst";
  sub_16B88A0(qword_4FAF440);
  return __cxa_atexit(sub_12EDE60, qword_4FAF440, &qword_4A427C0);
}
