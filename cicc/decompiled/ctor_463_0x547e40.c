// Function: ctor_463
// Address: 0x547e40
//
int ctor_463()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  char v9; // [rsp+1Bh] [rbp-55h] BYREF
  int v10; // [rsp+1Ch] [rbp-54h]
  char *v11; // [rsp+20h] [rbp-50h] BYREF
  void *v12; // [rsp+28h] [rbp-48h] BYREF
  const char *v13; // [rsp+30h] [rbp-40h] BYREF
  __int64 v14; // [rsp+38h] [rbp-38h]

  LODWORD(v12) = 2;
  LOBYTE(v11) = 0;
  v13 = (const char *)&v11;
  qword_4FFF920 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFF970 = 0x100000000LL;
  word_4FFF930 = 0;
  dword_4FFF92C &= 0x8000u;
  qword_4FFF938 = 0;
  qword_4FFF940 = 0;
  dword_4FFF928 = v0;
  qword_4FFF948 = 0;
  qword_4FFF950 = 0;
  qword_4FFF958 = 0;
  qword_4FFF960 = 0;
  qword_4FFF968 = (__int64)&unk_4FFF978;
  qword_4FFF980 = 0;
  qword_4FFF988 = (__int64)&unk_4FFF9A0;
  qword_4FFF990 = 1;
  dword_4FFF998 = 0;
  byte_4FFF99C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFF970;
  v3 = (unsigned int)qword_4FFF970 + 1LL;
  if ( v3 > HIDWORD(qword_4FFF970) )
  {
    sub_C8D5F0((char *)&unk_4FFF978 - 16, &unk_4FFF978, v3, 8);
    v2 = (unsigned int)qword_4FFF970;
  }
  *(_QWORD *)(qword_4FFF968 + 8 * v2) = v1;
  LODWORD(qword_4FFF970) = qword_4FFF970 + 1;
  byte_4FFF9B9 = 0;
  qword_4FFF9B0 = (__int64)&unk_49D9748;
  qword_4FFF9A8 = 0;
  qword_4FFF920 = (__int64)&unk_49D9AD8;
  qword_4FFF9C0 = (__int64)&unk_49DC1D0;
  qword_4FFF9E0 = (__int64)nullsub_39;
  qword_4FFF9D8 = (__int64)sub_AA4180;
  sub_C53080(&qword_4FFF920, "disable-loop-idiom-all", 22);
  qword_4FFF950 = 45;
  qword_4FFF948 = (__int64)"Options to disable Loop Idiom Recognize Pass.";
  sub_281DEA0(&qword_4FFF920, &unk_4FFF9E8, &v13, &v12);
  sub_C53130(&qword_4FFF920);
  __cxa_atexit(sub_AA4490, &qword_4FFF920, &qword_4A427C0);
  v10 = 2;
  v12 = &unk_4FFF908;
  v13 = "Proceed with loop idiom recognize pass, but do not convert loop(s) to memset.";
  v9 = 0;
  v11 = &v9;
  v14 = 77;
  sub_2820CC0(&unk_4FFF840, "disable-loop-idiom-memset", &v13, &v12, &v11);
  __cxa_atexit(sub_AA4490, &unk_4FFF840, &qword_4A427C0);
  v10 = 2;
  v12 = &unk_4FFF828;
  v13 = "Proceed with loop idiom recognize pass, but do not convert loop(s) to memcpy.";
  v9 = 0;
  v11 = &v9;
  v14 = 77;
  sub_2820CC0(&unk_4FFF760, "disable-loop-idiom-memcpy", &v13, &v12, &v11);
  __cxa_atexit(sub_AA4490, &unk_4FFF760, &qword_4A427C0);
  qword_4FFF680 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFF6FC = 1;
  qword_4FFF6D0 = 0x100000000LL;
  dword_4FFF68C &= 0x8000u;
  qword_4FFF698 = 0;
  qword_4FFF6A0 = 0;
  qword_4FFF6A8 = 0;
  dword_4FFF688 = v4;
  word_4FFF690 = 0;
  qword_4FFF6B0 = 0;
  qword_4FFF6B8 = 0;
  qword_4FFF6C0 = 0;
  qword_4FFF6C8 = (__int64)&unk_4FFF6D8;
  qword_4FFF6E0 = 0;
  qword_4FFF6E8 = (__int64)&unk_4FFF700;
  qword_4FFF6F0 = 1;
  dword_4FFF6F8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFF6D0;
  v7 = (unsigned int)qword_4FFF6D0 + 1LL;
  if ( v7 > HIDWORD(qword_4FFF6D0) )
  {
    sub_C8D5F0((char *)&unk_4FFF6D8 - 16, &unk_4FFF6D8, v7, 8);
    v6 = (unsigned int)qword_4FFF6D0;
  }
  *(_QWORD *)(qword_4FFF6C8 + 8 * v6) = v5;
  LODWORD(qword_4FFF6D0) = qword_4FFF6D0 + 1;
  qword_4FFF708 = 0;
  qword_4FFF710 = (__int64)&unk_49D9748;
  qword_4FFF718 = 0;
  qword_4FFF680 = (__int64)&unk_49DC090;
  qword_4FFF720 = (__int64)&unk_49DC1D0;
  qword_4FFF740 = (__int64)nullsub_23;
  qword_4FFF738 = (__int64)sub_984030;
  sub_C53080(&qword_4FFF680, "use-lir-code-size-heurs", 23);
  qword_4FFF6B0 = 75;
  qword_4FFF6A8 = (__int64)"Use loop idiom recognition code size heuristics when compiling with -Os/-Oz";
  LOWORD(qword_4FFF718) = 257;
  LOBYTE(qword_4FFF708) = 1;
  LOBYTE(dword_4FFF68C) = dword_4FFF68C & 0x9F | 0x20;
  sub_C53130(&qword_4FFF680);
  return __cxa_atexit(sub_984900, &qword_4FFF680, &qword_4A427C0);
}
