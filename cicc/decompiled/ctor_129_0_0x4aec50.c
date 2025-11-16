// Function: ctor_129_0
// Address: 0x4aec50
//
int ctor_129_0()
{
  int v0; // eax
  __int64 v1; // rax
  int v2; // eax
  int v4; // [rsp+4h] [rbp-FCh] BYREF
  void *v5; // [rsp+8h] [rbp-F8h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-F0h] BYREF
  _QWORD v7[2]; // [rsp+20h] [rbp-E0h] BYREF
  _QWORD v8[2]; // [rsp+30h] [rbp-D0h] BYREF
  int v9; // [rsp+40h] [rbp-C0h]
  const char *v10; // [rsp+48h] [rbp-B8h]
  __int64 v11; // [rsp+50h] [rbp-B0h]
  char *v12; // [rsp+58h] [rbp-A8h]
  __int64 v13; // [rsp+60h] [rbp-A0h]
  int v14; // [rsp+68h] [rbp-98h]
  const char *v15; // [rsp+70h] [rbp-90h]
  __int64 v16; // [rsp+78h] [rbp-88h]
  char *v17; // [rsp+80h] [rbp-80h]
  __int64 v18; // [rsp+88h] [rbp-78h]
  int v19; // [rsp+90h] [rbp-70h]
  const char *v20; // [rsp+98h] [rbp-68h]
  __int64 v21; // [rsp+A0h] [rbp-60h]

  qword_4F9A2C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9A2CC &= 0xF000u;
  qword_4F9A2D0 = 0;
  qword_4F9A2D8 = 0;
  qword_4F9A2E0 = 0;
  qword_4F9A2E8 = 0;
  dword_4F9A2C8 = v0;
  qword_4F9A318 = (__int64)&unk_4F9A338;
  qword_4F9A320 = (__int64)&unk_4F9A338;
  qword_4F9A2F0 = 0;
  qword_4F9A308 = (__int64)&unk_4FA01C0;
  qword_4F9A368 = (__int64)&unk_49E74E8;
  qword_4F9A2F8 = 0;
  qword_4F9A300 = 0;
  qword_4F9A2C0 = (__int64)&unk_49EAB58;
  qword_4F9A310 = 0;
  byte_4F9A358 = 0;
  qword_4F9A328 = 4;
  dword_4F9A330 = 0;
  qword_4F9A360 = 0;
  byte_4F9A371 = 0;
  qword_4F9A378 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4F9A2C0, "verify-region-info", 18);
  if ( qword_4F9A360 )
  {
    v1 = sub_16E8CB0();
    v7[0] = "cl::location(x) specified more than once!";
    LOWORD(v8[0]) = 259;
    sub_16B1F90(&qword_4F9A2C0, v7, 0, 0, v1);
  }
  else
  {
    byte_4F9A371 = 1;
    qword_4F9A360 = (__int64)&unk_4F9A38C;
    byte_4F9A370 = unk_4F9A38C;
  }
  qword_4F9A2F0 = 35;
  qword_4F9A2E8 = (__int64)"Verify region info (time consuming)";
  sub_16B88A0(&qword_4F9A2C0);
  __cxa_atexit(sub_13F9A70, &qword_4F9A2C0, &qword_4A427C0);
  v7[0] = v8;
  v8[0] = "none";
  v10 = "print no details";
  v12 = "bb";
  v15 = "print regions in detail with block_iterator";
  v17 = "rn";
  v20 = "print regions in detail with element_iterator";
  v7[1] = 0x400000003LL;
  v6[0] = "style of printing regions";
  v8[1] = 4;
  v5 = &unk_4F9A388;
  qword_4F9A060 = (__int64)&unk_49EED30;
  v9 = 0;
  v11 = 16;
  v13 = 2;
  v14 = 1;
  v16 = 43;
  v18 = 2;
  v19 = 2;
  v21 = 45;
  v6[1] = 25;
  v4 = 1;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4F9A0F8 = 0;
  word_4F9A06C &= 0xF000u;
  qword_4F9A070 = 0;
  qword_4F9A078 = 0;
  qword_4F9A080 = 0;
  qword_4F9A088 = 0;
  dword_4F9A068 = v2;
  qword_4F9A0B8 = (__int64)&unk_4F9A0D8;
  qword_4F9A0C0 = (__int64)&unk_4F9A0D8;
  qword_4F9A090 = 0;
  qword_4F9A0A8 = (__int64)&unk_4FA01C0;
  qword_4F9A108 = (__int64)&unk_49EBB00;
  qword_4F9A098 = 0;
  qword_4F9A0A0 = 0;
  qword_4F9A060 = (__int64)&unk_49EBB70;
  qword_4F9A0B0 = 0;
  qword_4F9A0C8 = 4;
  qword_4F9A118 = (__int64)&unk_49EBB20;
  qword_4F9A128 = (__int64)&unk_4F9A138;
  qword_4F9A130 = 0x800000000LL;
  dword_4F9A0D0 = 0;
  qword_4F9A100 = 0;
  byte_4F9A114 = 0;
  qword_4F9A120 = (__int64)&qword_4F9A060;
  sub_1444680(&qword_4F9A060, "print-region-style", &v5, &v4, v6, v7);
  sub_16B88A0(&qword_4F9A060);
  if ( (_QWORD *)v7[0] != v8 )
    _libc_free(v7[0], "print-region-style");
  return __cxa_atexit(sub_1442640, &qword_4F9A060, &qword_4A427C0);
}
