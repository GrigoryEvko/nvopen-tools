// Function: ctor_118_0
// Address: 0x4ac770
//
int ctor_118_0()
{
  int v0; // edx
  int v2; // [rsp+4h] [rbp-DCh] BYREF
  int *v3; // [rsp+8h] [rbp-D8h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v6[2]; // [rsp+30h] [rbp-B0h] BYREF
  int v7; // [rsp+40h] [rbp-A0h]
  const char *v8; // [rsp+48h] [rbp-98h]
  __int64 v9; // [rsp+50h] [rbp-90h]
  char *v10; // [rsp+58h] [rbp-88h]
  __int64 v11; // [rsp+60h] [rbp-80h]
  int v12; // [rsp+68h] [rbp-78h]
  const char *v13; // [rsp+70h] [rbp-70h]
  __int64 v14; // [rsp+78h] [rbp-68h]
  const char *v15; // [rsp+80h] [rbp-60h]
  __int64 v16; // [rsp+88h] [rbp-58h]
  int v17; // [rsp+90h] [rbp-50h]
  const char *v18; // [rsp+98h] [rbp-48h]
  __int64 v19; // [rsp+A0h] [rbp-40h]

  v6[0] = "throughput";
  v8 = "Reciprocal throughput";
  v10 = "latency";
  v13 = "Instruction latency";
  v15 = "code-size";
  v18 = "Code size";
  v5[1] = 0x400000003LL;
  v3 = &v2;
  v4[0] = "Target cost kind";
  v5[0] = v6;
  v6[1] = 10;
  qword_4F98AC0 = (__int64)&unk_49EED30;
  v7 = 0;
  v9 = 21;
  v11 = 7;
  v12 = 1;
  v14 = 19;
  v16 = 9;
  v17 = 2;
  v19 = 9;
  v2 = 0;
  v4[1] = 16;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F98ACC &= 0xF000u;
  qword_4F98AD0 = 0;
  qword_4F98B08 = (__int64)&unk_4FA01C0;
  qword_4F98AD8 = 0;
  qword_4F98AE0 = 0;
  qword_4F98AE8 = 0;
  dword_4F98AC8 = v0;
  qword_4F98B18 = (__int64)&unk_4F98B38;
  qword_4F98B20 = (__int64)&unk_4F98B38;
  qword_4F98AF0 = 0;
  qword_4F98AF8 = 0;
  qword_4F98B68 = (__int64)&unk_49E93C8;
  qword_4F98B00 = 0;
  qword_4F98B10 = 0;
  qword_4F98AC0 = (__int64)&unk_49E9438;
  qword_4F98B28 = 4;
  dword_4F98B30 = 0;
  qword_4F98B78 = (__int64)&unk_49E93E8;
  qword_4F98B88 = (__int64)&unk_4F98B98;
  qword_4F98B90 = 0x800000000LL;
  byte_4F98B58 = 0;
  dword_4F98B60 = 0;
  byte_4F98B74 = 1;
  dword_4F98B70 = 0;
  qword_4F98B80 = (__int64)&qword_4F98AC0;
  sub_139DE90(&qword_4F98AC0, "cost-kind", v4, &v3, v5);
  sub_16B88A0(&qword_4F98AC0);
  if ( (_QWORD *)v5[0] != v6 )
    _libc_free(v5[0], "cost-kind");
  return __cxa_atexit(sub_139D410, &qword_4F98AC0, &qword_4A427C0);
}
