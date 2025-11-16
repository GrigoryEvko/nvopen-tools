// Function: ctor_505
// Address: 0x55b510
//
int ctor_505()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_500AF60 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500AFB0 = 0x100000000LL;
  word_500AF70 = 0;
  dword_500AF6C &= 0x8000u;
  qword_500AF78 = 0;
  qword_500AF80 = 0;
  dword_500AF68 = v0;
  qword_500AF88 = 0;
  qword_500AF90 = 0;
  qword_500AF98 = 0;
  qword_500AFA0 = 0;
  qword_500AFA8 = (__int64)&unk_500AFB8;
  qword_500AFC0 = 0;
  qword_500AFC8 = (__int64)&unk_500AFE0;
  qword_500AFD0 = 1;
  dword_500AFD8 = 0;
  byte_500AFDC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500AFB0;
  v3 = (unsigned int)qword_500AFB0 + 1LL;
  if ( v3 > HIDWORD(qword_500AFB0) )
  {
    sub_C8D5F0((char *)&unk_500AFB8 - 16, &unk_500AFB8, v3, 8);
    v2 = (unsigned int)qword_500AFB0;
  }
  *(_QWORD *)(qword_500AFA8 + 8 * v2) = v1;
  LODWORD(qword_500AFB0) = qword_500AFB0 + 1;
  qword_500AFE8 = 0;
  qword_500AFF0 = (__int64)&unk_49D9748;
  qword_500AFF8 = 0;
  qword_500AF60 = (__int64)&unk_49DC090;
  qword_500B000 = (__int64)&unk_49DC1D0;
  qword_500B020 = (__int64)nullsub_23;
  qword_500B018 = (__int64)sub_984030;
  sub_C53080(&qword_500AF60, "pgo-warn-misexpect", 18);
  LOBYTE(qword_500AFE8) = 0;
  LOWORD(qword_500AFF8) = 256;
  qword_500AF90 = 88;
  LOBYTE(dword_500AF6C) = dword_500AF6C & 0x9F | 0x20;
  qword_500AF88 = (__int64)"Use this option to turn on/off warnings about incorrect usage of llvm.expect intrinsics.";
  sub_C53130(&qword_500AF60);
  __cxa_atexit(sub_984900, &qword_500AF60, &qword_4A427C0);
  qword_500AE80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500AEFC = 1;
  qword_500AED0 = 0x100000000LL;
  dword_500AE8C &= 0x8000u;
  qword_500AE98 = 0;
  qword_500AEA0 = 0;
  qword_500AEA8 = 0;
  dword_500AE88 = v4;
  word_500AE90 = 0;
  qword_500AEB0 = 0;
  qword_500AEB8 = 0;
  qword_500AEC0 = 0;
  qword_500AEC8 = (__int64)&unk_500AED8;
  qword_500AEE0 = 0;
  qword_500AEE8 = (__int64)&unk_500AF00;
  qword_500AEF0 = 1;
  dword_500AEF8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500AED0;
  v7 = (unsigned int)qword_500AED0 + 1LL;
  if ( v7 > HIDWORD(qword_500AED0) )
  {
    sub_C8D5F0((char *)&unk_500AED8 - 16, &unk_500AED8, v7, 8);
    v6 = (unsigned int)qword_500AED0;
  }
  *(_QWORD *)(qword_500AEC8 + 8 * v6) = v5;
  LODWORD(qword_500AED0) = qword_500AED0 + 1;
  qword_500AF08 = 0;
  qword_500AF10 = (__int64)&unk_49D9728;
  qword_500AF18 = 0;
  qword_500AE80 = (__int64)&unk_49DBF10;
  qword_500AF20 = (__int64)&unk_49DC290;
  qword_500AF40 = (__int64)nullsub_24;
  qword_500AF38 = (__int64)sub_984050;
  sub_C53080(&qword_500AE80, "misexpect-tolerance", 19);
  LODWORD(qword_500AF08) = 0;
  BYTE4(qword_500AF18) = 1;
  LODWORD(qword_500AF18) = 0;
  qword_500AEA8 = (__int64)"Prevents emitting diagnostics when profile counts are within N% of the threshold..";
  qword_500AEB0 = 82;
  sub_C53130(&qword_500AE80);
  return __cxa_atexit(sub_984970, &qword_500AE80, &qword_4A427C0);
}
