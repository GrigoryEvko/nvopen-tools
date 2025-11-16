// Function: ctor_192_0
// Address: 0x4df2e0
//
int ctor_192_0()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v4; // [rsp+1Ch] [rbp-F4h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v6[2]; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD v7[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v8; // [rsp+50h] [rbp-C0h]
  const char *v9; // [rsp+58h] [rbp-B8h]
  __int64 v10; // [rsp+60h] [rbp-B0h]
  char *v11; // [rsp+68h] [rbp-A8h]
  __int64 v12; // [rsp+70h] [rbp-A0h]
  int v13; // [rsp+78h] [rbp-98h]
  const char *v14; // [rsp+80h] [rbp-90h]
  __int64 v15; // [rsp+88h] [rbp-88h]
  char *v16; // [rsp+90h] [rbp-80h]
  __int64 v17; // [rsp+98h] [rbp-78h]
  int v18; // [rsp+A0h] [rbp-70h]
  const char *v19; // [rsp+A8h] [rbp-68h]
  __int64 v20; // [rsp+B0h] [rbp-60h]

  v7[0] = "none";
  v9 = "Do nothing";
  v11 = "import";
  v14 = "Import typeid resolutions from summary and globals";
  v16 = "export";
  v19 = "Export typeid resolutions to summary and globals";
  v6[1] = 0x400000003LL;
  v4 = 1;
  v6[0] = v7;
  v7[1] = 4;
  v8 = 0;
  v10 = 10;
  v12 = 6;
  v13 = 1;
  v15 = 50;
  v17 = 6;
  v18 = 2;
  v20 = 48;
  v5[0] = "What to do with the summary when running this pass";
  v5[1] = 50;
  sub_18B90D0(&unk_4FADA20, "wholeprogramdevirt-summary-action", v5, v6, &v4);
  if ( (_QWORD *)v6[0] != v7 )
    _libc_free(v6[0], "wholeprogramdevirt-summary-action");
  __cxa_atexit(sub_1872400, &unk_4FADA20, &qword_4A427C0);
  qword_4FAD900 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAD90C &= 0xF000u;
  qword_4FAD948 = (__int64)qword_4FA01C0;
  qword_4FAD910 = 0;
  qword_4FAD918 = 0;
  qword_4FAD920 = 0;
  dword_4FAD908 = v0;
  qword_4FAD958 = (__int64)&unk_4FAD978;
  qword_4FAD960 = (__int64)&unk_4FAD978;
  qword_4FAD9A0 = (__int64)&byte_4FAD9B0;
  qword_4FAD9C8 = (__int64)&byte_4FAD9D8;
  qword_4FAD928 = 0;
  qword_4FAD930 = 0;
  qword_4FAD9C0 = (__int64)&unk_49EED10;
  qword_4FAD900 = (__int64)&unk_49EEBF0;
  qword_4FAD938 = 0;
  qword_4FAD940 = 0;
  qword_4FAD950 = 0;
  qword_4FAD968 = 4;
  dword_4FAD970 = 0;
  byte_4FAD998 = 0;
  qword_4FAD9A8 = 0;
  byte_4FAD9B0 = 0;
  qword_4FAD9D0 = 0;
  byte_4FAD9D8 = 0;
  byte_4FAD9E8 = 0;
  qword_4FAD9F0 = (__int64)&unk_49EEE90;
  qword_4FAD9F8 = (__int64)&byte_4FADA08;
  qword_4FADA00 = 0;
  byte_4FADA08 = 0;
  sub_16B8280(&byte_4FADA08 - 264, "wholeprogramdevirt-read-summary", 31);
  qword_4FAD930 = 53;
  qword_4FAD928 = (__int64)"Read summary from given YAML file before running pass";
  LOBYTE(word_4FAD90C) = word_4FAD90C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAD900);
  __cxa_atexit(sub_12F0C20, &qword_4FAD900, &qword_4A427C0);
  qword_4FAD7E0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAD7EC &= 0xF000u;
  qword_4FAD7F0 = 0;
  qword_4FAD7F8 = 0;
  qword_4FAD800 = 0;
  qword_4FAD8A0 = (__int64)&unk_49EED10;
  qword_4FAD7E0 = (__int64)&unk_49EEBF0;
  dword_4FAD7E8 = v1;
  qword_4FAD838 = (__int64)&unk_4FAD858;
  qword_4FAD840 = (__int64)&unk_4FAD858;
  qword_4FAD880 = (__int64)&byte_4FAD890;
  qword_4FAD8A8 = (__int64)&byte_4FAD8B8;
  qword_4FAD828 = (__int64)qword_4FA01C0;
  qword_4FAD8D0 = (__int64)&unk_49EEE90;
  qword_4FAD8D8 = (__int64)&byte_4FAD8E8;
  qword_4FAD808 = 0;
  qword_4FAD810 = 0;
  qword_4FAD818 = 0;
  qword_4FAD820 = 0;
  qword_4FAD830 = 0;
  qword_4FAD848 = 4;
  dword_4FAD850 = 0;
  byte_4FAD878 = 0;
  qword_4FAD888 = 0;
  byte_4FAD890 = 0;
  qword_4FAD8B0 = 0;
  byte_4FAD8B8 = 0;
  byte_4FAD8C8 = 0;
  qword_4FAD8E0 = 0;
  byte_4FAD8E8 = 0;
  sub_16B8280(&byte_4FAD8E8 - 264, "wholeprogramdevirt-write-summary", 32);
  qword_4FAD810 = 51;
  qword_4FAD808 = (__int64)"Write summary to given YAML file after running pass";
  LOBYTE(word_4FAD7EC) = word_4FAD7EC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAD7E0);
  __cxa_atexit(sub_12F0C20, &qword_4FAD7E0, &qword_4A427C0);
  qword_4FAD700 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAD70C &= 0xF000u;
  qword_4FAD710 = 0;
  qword_4FAD718 = 0;
  qword_4FAD720 = 0;
  qword_4FAD728 = 0;
  dword_4FAD708 = v2;
  qword_4FAD758 = (__int64)&unk_4FAD778;
  qword_4FAD760 = (__int64)&unk_4FAD778;
  qword_4FAD748 = (__int64)qword_4FA01C0;
  qword_4FAD730 = 0;
  qword_4FAD7A8 = (__int64)&unk_49E74A8;
  qword_4FAD738 = 0;
  qword_4FAD740 = 0;
  qword_4FAD700 = (__int64)&unk_49EEAF0;
  qword_4FAD750 = 0;
  byte_4FAD798 = 0;
  qword_4FAD7B8 = (__int64)&unk_49EEE10;
  qword_4FAD768 = 4;
  dword_4FAD770 = 0;
  dword_4FAD7A0 = 0;
  byte_4FAD7B4 = 1;
  dword_4FAD7B0 = 0;
  sub_16B8280(&qword_4FAD700, "wholeprogramdevirt-branch-funnel-threshold", 42);
  dword_4FAD7A0 = 10;
  byte_4FAD7B4 = 1;
  dword_4FAD7B0 = 10;
  qword_4FAD730 = 69;
  LOBYTE(word_4FAD70C) = word_4FAD70C & 0x98 | 0x21;
  qword_4FAD728 = (__int64)"Maximum number of call targets per call site to enable branch funnels";
  sub_16B88A0(&qword_4FAD700);
  return __cxa_atexit(sub_12EDE60, &qword_4FAD700, &qword_4A427C0);
}
