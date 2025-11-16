// Function: ctor_460
// Address: 0x546ba0
//
int ctor_460()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // rdx
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v11; // [rsp+0h] [rbp-60h]
  __int64 v12; // [rsp+0h] [rbp-60h]
  char v13; // [rsp+13h] [rbp-4Dh] BYREF
  int v14; // [rsp+14h] [rbp-4Ch] BYREF
  char *v15; // [rsp+18h] [rbp-48h] BYREF
  const char *v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]

  v15 = &v13;
  v16 = "Turn on DominatorTree and LoopInfo verification after Loop Distribution";
  v13 = 0;
  v17 = 71;
  v14 = 1;
  sub_2808FF0(&unk_4FFEEE0, "loop-distribute-verify", &v14, &v16, &v15);
  __cxa_atexit(sub_984900, &unk_4FFEEE0, &qword_4A427C0);
  qword_4FFEE00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFEE7C = 1;
  word_4FFEE10 = 0;
  qword_4FFEE50 = 0x100000000LL;
  dword_4FFEE0C &= 0x8000u;
  qword_4FFEE18 = 0;
  qword_4FFEE20 = 0;
  dword_4FFEE08 = v0;
  qword_4FFEE28 = 0;
  qword_4FFEE30 = 0;
  qword_4FFEE38 = 0;
  qword_4FFEE40 = 0;
  qword_4FFEE48 = (__int64)&unk_4FFEE58;
  qword_4FFEE60 = 0;
  qword_4FFEE68 = (__int64)&unk_4FFEE80;
  qword_4FFEE70 = 1;
  dword_4FFEE78 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFEE50;
  if ( (unsigned __int64)(unsigned int)qword_4FFEE50 + 1 > HIDWORD(qword_4FFEE50) )
  {
    v11 = v1;
    sub_C8D5F0((char *)&unk_4FFEE58 - 16, &unk_4FFEE58, (unsigned int)qword_4FFEE50 + 1LL, 8);
    v2 = (unsigned int)qword_4FFEE50;
    v1 = v11;
  }
  *(_QWORD *)(qword_4FFEE48 + 8 * v2) = v1;
  LODWORD(qword_4FFEE50) = qword_4FFEE50 + 1;
  qword_4FFEE88 = 0;
  qword_4FFEE90 = (__int64)&unk_49D9748;
  qword_4FFEE98 = 0;
  qword_4FFEE00 = (__int64)&unk_49DC090;
  qword_4FFEEA0 = (__int64)&unk_49DC1D0;
  qword_4FFEEC0 = (__int64)nullsub_23;
  qword_4FFEEB8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFEE00, "loop-distribute-non-if-convertible", 34);
  LOWORD(qword_4FFEE98) = 256;
  LOBYTE(qword_4FFEE88) = 0;
  qword_4FFEE30 = 87;
  LOBYTE(dword_4FFEE0C) = dword_4FFEE0C & 0x9F | 0x20;
  qword_4FFEE28 = (__int64)"Whether to distribute into a loop that may not be if-convertible by the loop vectorizer";
  sub_C53130(&qword_4FFEE00);
  __cxa_atexit(sub_984900, &qword_4FFEE00, &qword_4A427C0);
  qword_4FFED20 = (__int64)&unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFED70 = 0x100000000LL;
  dword_4FFED2C &= 0x8000u;
  word_4FFED30 = 0;
  qword_4FFED68 = (__int64)&unk_4FFED78;
  qword_4FFED38 = 0;
  dword_4FFED28 = v3;
  qword_4FFED40 = 0;
  qword_4FFED48 = 0;
  qword_4FFED50 = 0;
  qword_4FFED58 = 0;
  qword_4FFED60 = 0;
  qword_4FFED80 = 0;
  qword_4FFED88 = (__int64)&unk_4FFEDA0;
  qword_4FFED90 = 1;
  dword_4FFED98 = 0;
  byte_4FFED9C = 1;
  v4 = sub_C57470();
  v5 = (unsigned int)qword_4FFED70;
  if ( (unsigned __int64)(unsigned int)qword_4FFED70 + 1 > HIDWORD(qword_4FFED70) )
  {
    v12 = v4;
    sub_C8D5F0((char *)&unk_4FFED78 - 16, &unk_4FFED78, (unsigned int)qword_4FFED70 + 1LL, 8);
    v5 = (unsigned int)qword_4FFED70;
    v4 = v12;
  }
  *(_QWORD *)(qword_4FFED68 + 8 * v5) = v4;
  LODWORD(qword_4FFED70) = qword_4FFED70 + 1;
  qword_4FFEDA8 = 0;
  qword_4FFEDB0 = (__int64)&unk_49D9728;
  qword_4FFEDB8 = 0;
  qword_4FFED20 = (__int64)&unk_49DBF10;
  qword_4FFEDC0 = (__int64)&unk_49DC290;
  qword_4FFEDE0 = (__int64)nullsub_24;
  qword_4FFEDD8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFED20, "loop-distribute-scev-check-threshold", 36);
  LODWORD(qword_4FFEDA8) = 8;
  BYTE4(qword_4FFEDB8) = 1;
  LODWORD(qword_4FFEDB8) = 8;
  qword_4FFED50 = 63;
  LOBYTE(dword_4FFED2C) = dword_4FFED2C & 0x9F | 0x20;
  qword_4FFED48 = (__int64)"The maximum number of SCEV checks allowed for Loop Distribution";
  sub_C53130(&qword_4FFED20);
  __cxa_atexit(sub_984970, &qword_4FFED20, &qword_4A427C0);
  qword_4FFEC40 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFEC4C &= 0x8000u;
  word_4FFEC50 = 0;
  qword_4FFEC90 = 0x100000000LL;
  qword_4FFEC88 = (__int64)&unk_4FFEC98;
  qword_4FFEC58 = 0;
  qword_4FFEC60 = 0;
  dword_4FFEC48 = v6;
  qword_4FFEC68 = 0;
  qword_4FFEC70 = 0;
  qword_4FFEC78 = 0;
  qword_4FFEC80 = 0;
  qword_4FFECA0 = 0;
  qword_4FFECA8 = (__int64)&unk_4FFECC0;
  qword_4FFECB0 = 1;
  dword_4FFECB8 = 0;
  byte_4FFECBC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FFEC90;
  v9 = (unsigned int)qword_4FFEC90 + 1LL;
  if ( v9 > HIDWORD(qword_4FFEC90) )
  {
    sub_C8D5F0((char *)&unk_4FFEC98 - 16, &unk_4FFEC98, v9, 8);
    v8 = (unsigned int)qword_4FFEC90;
  }
  *(_QWORD *)(qword_4FFEC88 + 8 * v8) = v7;
  LODWORD(qword_4FFEC90) = qword_4FFEC90 + 1;
  qword_4FFECC8 = 0;
  qword_4FFECD0 = (__int64)&unk_49D9728;
  qword_4FFECD8 = 0;
  qword_4FFEC40 = (__int64)&unk_49DBF10;
  qword_4FFECE0 = (__int64)&unk_49DC290;
  qword_4FFED00 = (__int64)nullsub_24;
  qword_4FFECF8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFEC40, "loop-distribute-scev-check-threshold-with-pragma", 48);
  LODWORD(qword_4FFECC8) = 128;
  BYTE4(qword_4FFECD8) = 1;
  LODWORD(qword_4FFECD8) = 128;
  qword_4FFEC70 = 122;
  LOBYTE(dword_4FFEC4C) = dword_4FFEC4C & 0x9F | 0x20;
  qword_4FFEC68 = (__int64)"The maximum number of SCEV checks allowed for Loop Distribution for loop marked with #pragma "
                           "clang loop distribute(enable)";
  sub_C53130(&qword_4FFEC40);
  __cxa_atexit(sub_984970, &qword_4FFEC40, &qword_4A427C0);
  v13 = 0;
  v15 = &v13;
  v16 = "Enable the new, experimental LoopDistribution Pass";
  v17 = 50;
  v14 = 1;
  sub_2808FF0(&unk_4FFEB60, "enable-loop-distribute", &v14, &v16, &v15);
  return __cxa_atexit(sub_984900, &unk_4FFEB60, &qword_4A427C0);
}
