// Function: ctor_343
// Address: 0x50af10
//
int ctor_343()
{
  int v0; // eax
  __int64 v1; // rax
  int v2; // eax
  const char *v4; // [rsp+0h] [rbp-50h] BYREF
  char v5; // [rsp+10h] [rbp-40h]
  char v6; // [rsp+11h] [rbp-3Fh]

  qword_4FCEEC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCEECC &= 0xF000u;
  qword_4FCEED0 = 0;
  qword_4FCEED8 = 0;
  qword_4FCEEE0 = 0;
  qword_4FCEEE8 = 0;
  dword_4FCEEC8 = v0;
  qword_4FCEF18 = (__int64)&unk_4FCEF38;
  qword_4FCEF20 = (__int64)&unk_4FCEF38;
  qword_4FCEF68 = (__int64)&unk_49E74A8;
  qword_4FCEEF0 = 0;
  qword_4FCEF08 = (__int64)qword_4FA01C0;
  qword_4FCEEC0 = (__int64)&unk_49FFFF8;
  qword_4FCEF78 = (__int64)&unk_49EEE10;
  qword_4FCEEF8 = 0;
  qword_4FCEF00 = 0;
  qword_4FCEF10 = 0;
  qword_4FCEF28 = 4;
  dword_4FCEF30 = 0;
  byte_4FCEF58 = 0;
  qword_4FCEF60 = 0;
  byte_4FCEF74 = 0;
  sub_16B8280(&qword_4FCEEC0, "limit-float-precision", 21);
  qword_4FCEEF0 = 63;
  qword_4FCEEE8 = (__int64)"Generate low-precision inline sequences for some float libcalls";
  if ( qword_4FCEF60 )
  {
    v1 = sub_16E8CB0();
    v6 = 1;
    v4 = "cl::location(x) specified more than once!";
    v5 = 3;
    sub_16B1F90(&qword_4FCEEC0, &v4, 0, 0, v1);
  }
  else
  {
    qword_4FCEF60 = (__int64)&dword_4FCEF88;
  }
  LOBYTE(word_4FCEECC) = word_4FCEECC & 0x9F | 0x20;
  *(_DWORD *)qword_4FCEF60 = 0;
  byte_4FCEF74 = 1;
  dword_4FCEF70 = 0;
  sub_16B88A0(&qword_4FCEEC0);
  __cxa_atexit(sub_2044000, &qword_4FCEEC0, &qword_4A427C0);
  qword_4FCEDE0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCEDEC &= 0xF000u;
  qword_4FCEDF0 = 0;
  qword_4FCEDF8 = 0;
  qword_4FCEE00 = 0;
  qword_4FCEE08 = 0;
  dword_4FCEDE8 = v2;
  qword_4FCEE38 = (__int64)&unk_4FCEE58;
  qword_4FCEE40 = (__int64)&unk_4FCEE58;
  qword_4FCEE88 = (__int64)&unk_49E74A8;
  qword_4FCEE28 = (__int64)qword_4FA01C0;
  qword_4FCEE10 = 0;
  qword_4FCEDE0 = (__int64)&unk_49EEAF0;
  qword_4FCEE98 = (__int64)&unk_49EEE10;
  qword_4FCEE18 = 0;
  qword_4FCEE20 = 0;
  qword_4FCEE30 = 0;
  qword_4FCEE48 = 4;
  dword_4FCEE50 = 0;
  byte_4FCEE78 = 0;
  dword_4FCEE80 = 0;
  byte_4FCEE94 = 1;
  dword_4FCEE90 = 0;
  sub_16B8280(&qword_4FCEDE0, "switch-peel-threshold", 21);
  dword_4FCEE80 = 66;
  byte_4FCEE94 = 1;
  dword_4FCEE90 = 66;
  qword_4FCEE10 = 133;
  LOBYTE(word_4FCEDEC) = word_4FCEDEC & 0x9F | 0x20;
  qword_4FCEE08 = (__int64)"Set the case probability threshold for peeling the case from a switch statement. A value grea"
                           "ter than 100 will void this optimization";
  sub_16B88A0(&qword_4FCEDE0);
  return __cxa_atexit(sub_12EDE60, &qword_4FCEDE0, &qword_4A427C0);
}
