// Function: ctor_435
// Address: 0x53bcb0
//
int ctor_435()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v4; // [rsp+8h] [rbp-58h]
  char v5; // [rsp+13h] [rbp-4Dh] BYREF
  int v6; // [rsp+14h] [rbp-4Ch] BYREF
  char *v7; // [rsp+18h] [rbp-48h] BYREF
  const char *v8; // [rsp+20h] [rbp-40h] BYREF
  __int64 v9; // [rsp+28h] [rbp-38h]

  v8 = "Do pseudo probe verification";
  v9 = 28;
  v6 = 1;
  v5 = 0;
  v7 = &v5;
  sub_26EB6D0(&unk_4FF8B60, "verify-pseudo-probe", &v7, &v6, &v8);
  __cxa_atexit(sub_984900, &unk_4FF8B60, &qword_4A427C0);
  qword_4FF8A60 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF8A78 = 0;
  qword_4FF8A80 = 0;
  qword_4FF8A88 = 0;
  qword_4FF8A90 = 0;
  dword_4FF8A6C = dword_4FF8A6C & 0x8000 | 1;
  word_4FF8A70 = 0;
  qword_4FF8AB0 = 0x100000000LL;
  dword_4FF8A68 = v0;
  qword_4FF8A98 = 0;
  qword_4FF8AA0 = 0;
  qword_4FF8AA8 = (__int64)&unk_4FF8AB8;
  qword_4FF8AC0 = 0;
  qword_4FF8AC8 = (__int64)&unk_4FF8AE0;
  qword_4FF8AD0 = 1;
  dword_4FF8AD8 = 0;
  byte_4FF8ADC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF8AB0;
  if ( (unsigned __int64)(unsigned int)qword_4FF8AB0 + 1 > HIDWORD(qword_4FF8AB0) )
  {
    v4 = v1;
    sub_C8D5F0((char *)&unk_4FF8AB8 - 16, &unk_4FF8AB8, (unsigned int)qword_4FF8AB0 + 1LL, 8);
    v2 = (unsigned int)qword_4FF8AB0;
    v1 = v4;
  }
  *(_QWORD *)(qword_4FF8AA8 + 8 * v2) = v1;
  LODWORD(qword_4FF8AB0) = qword_4FF8AB0 + 1;
  qword_4FF8AE8 = 0;
  qword_4FF8A60 = (__int64)&unk_49DAD08;
  qword_4FF8AF0 = 0;
  qword_4FF8AF8 = 0;
  qword_4FF8B38 = (__int64)&unk_49DC350;
  qword_4FF8B00 = 0;
  qword_4FF8B58 = (__int64)nullsub_81;
  qword_4FF8B08 = 0;
  qword_4FF8B50 = (__int64)sub_BB8600;
  qword_4FF8B10 = 0;
  byte_4FF8B18 = 0;
  qword_4FF8B20 = 0;
  qword_4FF8B28 = 0;
  qword_4FF8B30 = 0;
  sub_C53080(&qword_4FF8A60, "verify-pseudo-probe-funcs", 25);
  qword_4FF8A90 = 58;
  LOBYTE(dword_4FF8A6C) = dword_4FF8A6C & 0x9F | 0x20;
  qword_4FF8A88 = (__int64)"The option to specify the name of the functions to verify.";
  sub_C53130(&qword_4FF8A60);
  __cxa_atexit(sub_BB89D0, &qword_4FF8A60, &qword_4A427C0);
  v7 = &v5;
  v8 = "Update pseudo probe distribution factor";
  v9 = 39;
  v6 = 1;
  v5 = 1;
  sub_26EB6D0(&unk_4FF8980, "update-pseudo-probe", &v7, &v6, &v8);
  return __cxa_atexit(sub_984900, &unk_4FF8980, &qword_4A427C0);
}
