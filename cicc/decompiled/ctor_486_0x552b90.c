// Function: ctor_486
// Address: 0x552b90
//
int ctor_486()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  char v5; // [rsp+3h] [rbp-4Dh] BYREF
  int v6; // [rsp+4h] [rbp-4Ch] BYREF
  char *v7; // [rsp+8h] [rbp-48h] BYREF
  const char *v8; // [rsp+10h] [rbp-40h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h]

  v8 = "Checking sinking scheduling effect";
  v9 = 34;
  v6 = 1;
  v5 = 1;
  v7 = &v5;
  sub_2977F40(&unk_5006FC0, "sink-check-sched", &v7, &v6, &v8);
  __cxa_atexit(sub_984900, &unk_5006FC0, &qword_4A427C0);
  v9 = 36;
  v8 = "Sinking single-use only instructions";
  v6 = 1;
  v5 = 1;
  v7 = &v5;
  sub_2977F40(&unk_5006EE0, "sink-single-only", &v7, &v6, &v8);
  __cxa_atexit(sub_984900, &unk_5006EE0, &qword_4A427C0);
  qword_5006E00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5006E7C = 1;
  qword_5006E50 = 0x100000000LL;
  dword_5006E0C &= 0x8000u;
  qword_5006E18 = 0;
  qword_5006E20 = 0;
  qword_5006E28 = 0;
  dword_5006E08 = v0;
  word_5006E10 = 0;
  qword_5006E30 = 0;
  qword_5006E38 = 0;
  qword_5006E40 = 0;
  qword_5006E48 = (__int64)&unk_5006E58;
  qword_5006E60 = 0;
  qword_5006E68 = (__int64)&unk_5006E80;
  qword_5006E70 = 1;
  dword_5006E78 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5006E50;
  v3 = (unsigned int)qword_5006E50 + 1LL;
  if ( v3 > HIDWORD(qword_5006E50) )
  {
    sub_C8D5F0((char *)&unk_5006E58 - 16, &unk_5006E58, v3, 8);
    v2 = (unsigned int)qword_5006E50;
  }
  *(_QWORD *)(qword_5006E48 + 8 * v2) = v1;
  LODWORD(qword_5006E50) = qword_5006E50 + 1;
  qword_5006E88 = 0;
  qword_5006E90 = (__int64)&unk_49D9748;
  qword_5006E98 = 0;
  qword_5006E00 = (__int64)&unk_49DC090;
  qword_5006EA0 = (__int64)&unk_49DC1D0;
  qword_5006EC0 = (__int64)nullsub_23;
  qword_5006EB8 = (__int64)sub_984030;
  sub_C53080(&qword_5006E00, "rp-aware-sink", 13);
  LOBYTE(qword_5006E88) = 0;
  LOWORD(qword_5006E98) = 256;
  qword_5006E30 = 46;
  LOBYTE(dword_5006E0C) = dword_5006E0C & 0x9F | 0x20;
  qword_5006E28 = (__int64)"Consider register pressure impact when sinking";
  sub_C53130(&qword_5006E00);
  return __cxa_atexit(sub_984900, &qword_5006E00, &qword_4A427C0);
}
