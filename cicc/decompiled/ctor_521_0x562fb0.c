// Function: ctor_521
// Address: 0x562fb0
//
int ctor_521()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  char v5; // [rsp+13h] [rbp-4Dh] BYREF
  int v6; // [rsp+14h] [rbp-4Ch] BYREF
  char *v7; // [rsp+18h] [rbp-48h] BYREF
  char *v8; // [rsp+20h] [rbp-40h] BYREF
  __int64 v9; // [rsp+28h] [rbp-38h]

  qword_5010F80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5010FFC = 1;
  qword_5010FD0 = 0x100000000LL;
  dword_5010F8C &= 0x8000u;
  qword_5010F98 = 0;
  qword_5010FA0 = 0;
  qword_5010FA8 = 0;
  dword_5010F88 = v0;
  word_5010F90 = 0;
  qword_5010FB0 = 0;
  qword_5010FB8 = 0;
  qword_5010FC0 = 0;
  qword_5010FC8 = (__int64)&unk_5010FD8;
  qword_5010FE0 = 0;
  qword_5010FE8 = (__int64)&unk_5011000;
  qword_5010FF0 = 1;
  dword_5010FF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5010FD0;
  v3 = (unsigned int)qword_5010FD0 + 1LL;
  if ( v3 > HIDWORD(qword_5010FD0) )
  {
    sub_C8D5F0((char *)&unk_5010FD8 - 16, &unk_5010FD8, v3, 8);
    v2 = (unsigned int)qword_5010FD0;
  }
  *(_QWORD *)(qword_5010FC8 + 8 * v2) = v1;
  LODWORD(qword_5010FD0) = qword_5010FD0 + 1;
  qword_5011008 = 0;
  qword_5011010 = (__int64)&unk_49D9748;
  qword_5011018 = 0;
  qword_5010F80 = (__int64)&unk_49DC090;
  qword_5011020 = (__int64)&unk_49DC1D0;
  qword_5011040 = (__int64)nullsub_23;
  qword_5011038 = (__int64)sub_984030;
  sub_C53080(&qword_5010F80, "nv-ocl", 6);
  qword_5010FA8 = (__int64)"deprecated";
  qword_5010FB0 = 10;
  LOBYTE(qword_5011008) = 0;
  LOBYTE(dword_5010F8C) = dword_5010F8C & 0x9F | 0x20;
  LOWORD(qword_5011018) = 256;
  sub_C53130(&qword_5010F80);
  __cxa_atexit(sub_984900, &qword_5010F80, &qword_4A427C0);
  v7 = &v5;
  v8 = "deprecated";
  v5 = 0;
  v6 = 1;
  v9 = 10;
  sub_2C744F0(&unk_5010EA0, "nv-cuda", &v8, &v6, &v7);
  __cxa_atexit(sub_984900, &unk_5010EA0, &qword_4A427C0);
  v8 = "deprecated";
  v7 = &v5;
  v5 = 0;
  v6 = 1;
  v9 = 10;
  sub_2C744F0(&unk_5010DC0, "drvcuda", &v8, &v6, &v7);
  __cxa_atexit(sub_984900, &unk_5010DC0, &qword_4A427C0);
  v8 = "deprecated";
  v5 = 0;
  v7 = &v5;
  v6 = 1;
  v9 = 10;
  sub_2C744F0(&unk_5010CE0, "drvnvcl", &v8, &v6, &v7);
  return __cxa_atexit(sub_984900, &unk_5010CE0, &qword_4A427C0);
}
