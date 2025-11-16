// Function: ctor_384_0
// Address: 0x51ad60
//
int ctor_384_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  int v10; // [rsp+4h] [rbp-ECh] BYREF
  void *v11; // [rsp+8h] [rbp-E8h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v14[2]; // [rsp+30h] [rbp-C0h] BYREF
  int v15; // [rsp+40h] [rbp-B0h]
  const char *v16; // [rsp+48h] [rbp-A8h]
  __int64 v17; // [rsp+50h] [rbp-A0h]
  char *v18; // [rsp+58h] [rbp-98h]
  __int64 v19; // [rsp+60h] [rbp-90h]
  int v20; // [rsp+68h] [rbp-88h]
  const char *v21; // [rsp+70h] [rbp-80h]
  __int64 v22; // [rsp+78h] [rbp-78h]
  char *v23; // [rsp+80h] [rbp-70h]
  __int64 v24; // [rsp+88h] [rbp-68h]
  int v25; // [rsp+90h] [rbp-60h]
  const char *v26; // [rsp+98h] [rbp-58h]
  __int64 v27; // [rsp+A0h] [rbp-50h]

  qword_4FDBF80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FDBFFC = 1;
  qword_4FDBFD0 = 0x100000000LL;
  dword_4FDBF8C &= 0x8000u;
  qword_4FDBF98 = 0;
  qword_4FDBFA0 = 0;
  qword_4FDBFA8 = 0;
  dword_4FDBF88 = v0;
  word_4FDBF90 = 0;
  qword_4FDBFB0 = 0;
  qword_4FDBFB8 = 0;
  qword_4FDBFC0 = 0;
  qword_4FDBFC8 = (__int64)&unk_4FDBFD8;
  qword_4FDBFE0 = 0;
  qword_4FDBFE8 = (__int64)&unk_4FDC000;
  qword_4FDBFF0 = 1;
  dword_4FDBFF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FDBFD0;
  v3 = (unsigned int)qword_4FDBFD0 + 1LL;
  if ( v3 > HIDWORD(qword_4FDBFD0) )
  {
    sub_C8D5F0((char *)&unk_4FDBFD8 - 16, &unk_4FDBFD8, v3, 8);
    v2 = (unsigned int)qword_4FDBFD0;
  }
  *(_QWORD *)(qword_4FDBFC8 + 8 * v2) = v1;
  LODWORD(qword_4FDBFD0) = qword_4FDBFD0 + 1;
  byte_4FDC019 = 0;
  qword_4FDC010 = (__int64)&unk_49D9748;
  qword_4FDC008 = 0;
  qword_4FDBF80 = (__int64)&unk_49D9AD8;
  qword_4FDC020 = (__int64)&unk_49DC1D0;
  qword_4FDC040 = (__int64)nullsub_39;
  qword_4FDC038 = (__int64)sub_AA4180;
  sub_C53080(&qword_4FDBF80, "verify-region-info", 18);
  if ( qword_4FDC008 )
  {
    v4 = sub_CEADF0();
    v13[0] = "cl::location(x) specified more than once!";
    LOWORD(v15) = 259;
    sub_C53280(&qword_4FDBF80, v13, 0, 0, v4);
  }
  else
  {
    byte_4FDC019 = 1;
    qword_4FDC008 = (__int64)&unk_4FDC04C;
    byte_4FDC018 = unk_4FDC04C;
  }
  qword_4FDBFB0 = 35;
  qword_4FDBFA8 = (__int64)"Verify region info (time consuming)";
  sub_C53130(&qword_4FDBF80);
  __cxa_atexit(sub_AA4490, &qword_4FDBF80, &qword_4A427C0);
  v13[0] = v14;
  v14[0] = "none";
  v16 = "print no details";
  v18 = "bb";
  v21 = "print regions in detail with block_iterator";
  v23 = "rn";
  v26 = "print regions in detail with element_iterator";
  v13[1] = 0x400000003LL;
  v12[0] = "style of printing regions";
  v14[1] = 4;
  v11 = &unk_4FDC048;
  v15 = 0;
  v17 = 16;
  v19 = 2;
  v20 = 1;
  v22 = 43;
  v24 = 2;
  v25 = 2;
  v27 = 45;
  v12[1] = 25;
  v10 = 1;
  qword_4FDBD20 = (__int64)&unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FDBD2C &= 0x8000u;
  word_4FDBD30 = 0;
  qword_4FDBD70 = 0x100000000LL;
  qword_4FDBD38 = 0;
  qword_4FDBD40 = 0;
  qword_4FDBD48 = 0;
  dword_4FDBD28 = v5;
  qword_4FDBD50 = 0;
  qword_4FDBD58 = 0;
  qword_4FDBD60 = 0;
  qword_4FDBD68 = (__int64)&unk_4FDBD78;
  qword_4FDBD80 = 0;
  qword_4FDBD88 = (__int64)&unk_4FDBDA0;
  qword_4FDBD90 = 1;
  dword_4FDBD98 = 0;
  byte_4FDBD9C = 1;
  v6 = sub_C57470();
  v7 = (unsigned int)qword_4FDBD70;
  v8 = (unsigned int)qword_4FDBD70 + 1LL;
  if ( v8 > HIDWORD(qword_4FDBD70) )
  {
    sub_C8D5F0((char *)&unk_4FDBD78 - 16, &unk_4FDBD78, v8, 8);
    v7 = (unsigned int)qword_4FDBD70;
  }
  *(_QWORD *)(qword_4FDBD68 + 8 * v7) = v6;
  LODWORD(qword_4FDBD70) = qword_4FDBD70 + 1;
  byte_4FDBDBC = 0;
  qword_4FDBDB0 = (__int64)&unk_4A09FD0;
  qword_4FDBDA8 = 0;
  qword_4FDBDC8 = (__int64)&qword_4FDBD20;
  qword_4FDBD20 = (__int64)&unk_4A0A040;
  qword_4FDBDC0 = (__int64)&unk_4A09FF0;
  qword_4FDBDD0 = (__int64)&unk_4FDBDE0;
  qword_4FDBDD8 = 0x800000000LL;
  qword_4FDBF78 = (__int64)nullsub_828;
  qword_4FDBF70 = (__int64)sub_22DA230;
  sub_22E27E0(&qword_4FDBD20, "print-region-style", &v11, &v10, v12, v13);
  sub_C53130(&qword_4FDBD20);
  if ( (_QWORD *)v13[0] != v14 )
    _libc_free(v13[0], "print-region-style");
  return __cxa_atexit(sub_22DA7E0, &qword_4FDBD20, &qword_4A427C0);
}
