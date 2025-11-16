// Function: ctor_698_0
// Address: 0x5a8320
//
int __fastcall ctor_698_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
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
  const char *v15; // [rsp+48h] [rbp-A8h]
  __int64 v16; // [rsp+50h] [rbp-A0h]
  char *v17; // [rsp+58h] [rbp-98h]
  __int64 v18; // [rsp+60h] [rbp-90h]
  int v19; // [rsp+68h] [rbp-88h]
  const char *v20; // [rsp+70h] [rbp-80h]
  __int64 v21; // [rsp+78h] [rbp-78h]

  v11[0] = "Enable inliner stats for imported functions";
  v13[0] = "basic";
  v15 = "basic statistics";
  v17 = "verbose";
  v20 = "printing of statistics for each inlined function";
  v12[1] = 0x400000002LL;
  v10 = &v8;
  v11[1] = 43;
  v9 = 1;
  qword_5041320 = &unk_49DC150;
  v12[0] = v13;
  v13[1] = 5;
  v14 = 1;
  v16 = 16;
  v18 = 7;
  v19 = 2;
  v21 = 48;
  v8 = 0;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_504132C = word_504132C & 0x8000;
  unk_5041330 = 0;
  qword_5041368[1] = 0x100000000LL;
  unk_5041328 = v4;
  unk_5041338 = 0;
  unk_5041340 = 0;
  unk_5041348 = 0;
  unk_5041350 = 0;
  unk_5041358 = 0;
  unk_5041360 = 0;
  qword_5041368[0] = &qword_5041368[2];
  qword_5041368[3] = 0;
  qword_5041368[4] = &qword_5041368[7];
  qword_5041368[5] = 1;
  LODWORD(qword_5041368[6]) = 0;
  BYTE4(qword_5041368[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_5041368[1]);
  if ( (unsigned __int64)LODWORD(qword_5041368[1]) + 1 > HIDWORD(qword_5041368[1]) )
  {
    sub_C8D5F0(qword_5041368, &qword_5041368[2], LODWORD(qword_5041368[1]) + 1LL, 8);
    v6 = LODWORD(qword_5041368[1]);
  }
  *(_QWORD *)(qword_5041368[0] + 8 * v6) = v5;
  ++LODWORD(qword_5041368[1]);
  qword_5041368[12] = &qword_5041320;
  qword_5041368[9] = &unk_4A3C490;
  qword_5041368[8] = 0;
  qword_5041368[10] = 0;
  qword_5041320 = &unk_4A3C500;
  qword_5041368[11] = &unk_4A3C4B0;
  qword_5041368[13] = &qword_5041368[15];
  qword_5041368[14] = 0x800000000LL;
  qword_5041368[66] = nullsub_1912;
  qword_5041368[65] = sub_36FB5A0;
  sub_36FE630(&qword_5041320, "inliner-function-import-stats", &v10, v12, &v9, v11);
  sub_C53130(&qword_5041320);
  if ( (_QWORD *)v12[0] != v13 )
    _libc_free(v12[0], "inliner-function-import-stats");
  return __cxa_atexit(sub_36FB9A0, &qword_5041320, &qword_4A427C0);
}
