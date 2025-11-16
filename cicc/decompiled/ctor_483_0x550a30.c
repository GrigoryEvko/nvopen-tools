// Function: ctor_483
// Address: 0x550a30
//
int ctor_483()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  char v5; // [rsp+3h] [rbp-4Dh] BYREF
  int v6; // [rsp+4h] [rbp-4Ch] BYREF
  char *v7; // [rsp+8h] [rbp-48h] BYREF
  const char *v8; // [rsp+10h] [rbp-40h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h]

  qword_5005900 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500597C = 1;
  qword_5005950 = 0x100000000LL;
  dword_500590C &= 0x8000u;
  qword_5005918 = 0;
  qword_5005920 = 0;
  qword_5005928 = 0;
  dword_5005908 = v0;
  word_5005910 = 0;
  qword_5005930 = 0;
  qword_5005938 = 0;
  qword_5005940 = 0;
  qword_5005948 = (__int64)&unk_5005958;
  qword_5005960 = 0;
  qword_5005968 = (__int64)&unk_5005980;
  qword_5005970 = 1;
  dword_5005978 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5005950;
  v3 = (unsigned int)qword_5005950 + 1LL;
  if ( v3 > HIDWORD(qword_5005950) )
  {
    sub_C8D5F0((char *)&unk_5005958 - 16, &unk_5005958, v3, 8);
    v2 = (unsigned int)qword_5005950;
  }
  *(_QWORD *)(qword_5005948 + 8 * v2) = v1;
  LODWORD(qword_5005950) = qword_5005950 + 1;
  qword_5005988 = 0;
  qword_5005990 = (__int64)&unk_49D9748;
  qword_5005998 = 0;
  qword_5005900 = (__int64)&unk_49DC090;
  qword_50059A0 = (__int64)&unk_49DC1D0;
  qword_50059C0 = (__int64)nullsub_23;
  qword_50059B8 = (__int64)sub_984030;
  sub_C53080(&qword_5005900, "disable-separate-const-offset-from-gep", 38);
  LOBYTE(qword_5005988) = 0;
  LOWORD(qword_5005998) = 256;
  qword_5005928 = (__int64)"Do not separate the constant offset from a GEP instruction";
  qword_5005930 = 58;
  LOBYTE(dword_500590C) = dword_500590C & 0x9F | 0x20;
  sub_C53130(&qword_5005900);
  __cxa_atexit(sub_984900, &qword_5005900, &qword_4A427C0);
  v7 = &v5;
  v8 = "Verify this pass produces no dead code";
  v6 = 1;
  v9 = 38;
  v5 = 0;
  sub_29522C0(&unk_5005820, "reassociate-geps-verify-no-dead-code", &v7, &v8, &v6);
  __cxa_atexit(sub_984900, &unk_5005820, &qword_4A427C0);
  v7 = &v5;
  v6 = 1;
  v8 = "Clone indexes at their definitions instead of immediately before the GEP";
  v9 = 72;
  v5 = 1;
  sub_29522C0(&unk_5005740, "separate-geps-clone-definition-point", &v7, &v8, &v6);
  return __cxa_atexit(sub_984900, &unk_5005740, &qword_4A427C0);
}
