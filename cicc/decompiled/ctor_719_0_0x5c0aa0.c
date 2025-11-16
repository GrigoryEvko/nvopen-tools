// Function: ctor_719_0
// Address: 0x5c0aa0
//
int ctor_719_0()
{
  int v0; // edx
  int v2; // [rsp+0h] [rbp-E0h] BYREF
  int v3; // [rsp+4h] [rbp-DCh] BYREF
  int *v4; // [rsp+8h] [rbp-D8h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v6[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v7[2]; // [rsp+30h] [rbp-B0h] BYREF
  int v8; // [rsp+40h] [rbp-A0h]
  const char *v9; // [rsp+48h] [rbp-98h]
  __int64 v10; // [rsp+50h] [rbp-90h]
  const char *v11; // [rsp+58h] [rbp-88h]
  __int64 v12; // [rsp+60h] [rbp-80h]
  int v13; // [rsp+68h] [rbp-78h]
  char *v14; // [rsp+70h] [rbp-70h]
  __int64 v15; // [rsp+78h] [rbp-68h]
  const char *v16; // [rsp+80h] [rbp-60h]
  __int64 v17; // [rsp+88h] [rbp-58h]
  int v18; // [rsp+90h] [rbp-50h]
  char *v19; // [rsp+98h] [rbp-48h]
  __int64 v20; // [rsp+A0h] [rbp-40h]

  v4 = &v3;
  v7[0] = "Default";
  v9 = "Default for platform";
  v11 = "Enable";
  v14 = "Enabled";
  v16 = "Disable";
  v19 = "Disabled";
  v6[1] = 0x400000003LL;
  v5[0] = "Disable emission of the extended flags in .loc directives.";
  v3 = 0;
  v6[0] = v7;
  qword_50526A0 = (__int64)&unk_49EED30;
  v7[1] = 7;
  v8 = 0;
  v10 = 20;
  v12 = 6;
  v13 = 1;
  v15 = 7;
  v17 = 7;
  v18 = 2;
  v20 = 8;
  v5[1] = 58;
  v2 = 1;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_50526AC &= 0xF000u;
  qword_50526B0 = 0;
  qword_50526E8 = (__int64)qword_4FA01C0;
  qword_50526B8 = 0;
  qword_50526C0 = 0;
  qword_50526C8 = 0;
  dword_50526A8 = v0;
  qword_50526F8 = (__int64)&unk_5052718;
  qword_5052700 = (__int64)&unk_5052718;
  qword_50526D0 = 0;
  qword_50526D8 = 0;
  qword_5052748 = (__int64)&unk_4A3DEB8;
  qword_50526E0 = 0;
  qword_50526F0 = 0;
  qword_50526A0 = (__int64)&unk_4A3DF28;
  qword_5052708 = 4;
  dword_5052710 = 0;
  qword_5052758 = (__int64)&unk_4A3DED8;
  qword_5052768 = (__int64)&unk_5052778;
  qword_5052770 = 0x800000000LL;
  byte_5052738 = 0;
  dword_5052740 = 0;
  byte_5052754 = 1;
  dword_5052750 = 0;
  qword_5052760 = (__int64)&qword_50526A0;
  ((void (__fastcall *)(__int64 *, const char *, int *, _QWORD *, _QWORD *, int **))sub_38BB4C0)(
    &qword_50526A0,
    "dwarf-extended-loc",
    &v2,
    v5,
    v6,
    &v4);
  sub_16B88A0(&qword_50526A0);
  if ( (_QWORD *)v6[0] != v7 )
    _libc_free(v6[0], "dwarf-extended-loc");
  return __cxa_atexit(sub_38BADC0, &qword_50526A0, &qword_4A427C0);
}
