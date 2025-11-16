// Function: ctor_464
// Address: 0x548340
//
int ctor_464()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r9
  int v10; // [rsp+0h] [rbp-50h] BYREF
  int v11; // [rsp+4h] [rbp-4Ch] BYREF
  int *v12; // [rsp+8h] [rbp-48h] BYREF
  const char *v13; // [rsp+10h] [rbp-40h] BYREF
  __int64 v14; // [rsp+18h] [rbp-38h]

  qword_4FFFCA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFFD1C = 1;
  qword_4FFFCF0 = 0x100000000LL;
  dword_4FFFCAC &= 0x8000u;
  qword_4FFFCB8 = 0;
  qword_4FFFCC0 = 0;
  qword_4FFFCC8 = 0;
  dword_4FFFCA8 = v0;
  word_4FFFCB0 = 0;
  qword_4FFFCD0 = 0;
  qword_4FFFCD8 = 0;
  qword_4FFFCE0 = 0;
  qword_4FFFCE8 = (__int64)&unk_4FFFCF8;
  qword_4FFFD00 = 0;
  qword_4FFFD08 = (__int64)&unk_4FFFD20;
  qword_4FFFD10 = 1;
  dword_4FFFD18 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFFCF0;
  v3 = (unsigned int)qword_4FFFCF0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFFCF0) )
  {
    sub_C8D5F0((char *)&unk_4FFFCF8 - 16, &unk_4FFFCF8, v3, 8);
    v2 = (unsigned int)qword_4FFFCF0;
  }
  *(_QWORD *)(qword_4FFFCE8 + 8 * v2) = v1;
  LODWORD(qword_4FFFCF0) = qword_4FFFCF0 + 1;
  qword_4FFFD28 = 0;
  qword_4FFFD30 = (__int64)&unk_49DA090;
  qword_4FFFD38 = 0;
  qword_4FFFCA0 = (__int64)&unk_49DBF90;
  qword_4FFFD40 = (__int64)&unk_49DC230;
  qword_4FFFD60 = (__int64)nullsub_58;
  qword_4FFFD58 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FFFCA0, "loop-interchange-threshold", 26);
  LODWORD(qword_4FFFD28) = 0;
  BYTE4(qword_4FFFD38) = 1;
  LODWORD(qword_4FFFD38) = 0;
  qword_4FFFCD0 = 45;
  LOBYTE(dword_4FFFCAC) = dword_4FFFCAC & 0x9F | 0x20;
  qword_4FFFCC8 = (__int64)"Interchange if you gain more than this number";
  sub_C53130(&qword_4FFFCA0);
  __cxa_atexit(sub_B2B680, &qword_4FFFCA0, &qword_4A427C0);
  qword_4FFFBC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFFBCC &= 0x8000u;
  word_4FFFBD0 = 0;
  qword_4FFFC10 = 0x100000000LL;
  qword_4FFFBD8 = 0;
  qword_4FFFBE0 = 0;
  qword_4FFFBE8 = 0;
  dword_4FFFBC8 = v4;
  qword_4FFFBF0 = 0;
  qword_4FFFBF8 = 0;
  qword_4FFFC00 = 0;
  qword_4FFFC08 = (__int64)&unk_4FFFC18;
  qword_4FFFC20 = 0;
  qword_4FFFC28 = (__int64)&unk_4FFFC40;
  qword_4FFFC30 = 1;
  dword_4FFFC38 = 0;
  byte_4FFFC3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFFC10;
  v7 = (unsigned int)qword_4FFFC10 + 1LL;
  if ( v7 > HIDWORD(qword_4FFFC10) )
  {
    sub_C8D5F0((char *)&unk_4FFFC18 - 16, &unk_4FFFC18, v7, 8);
    v6 = (unsigned int)qword_4FFFC10;
  }
  *(_QWORD *)(qword_4FFFC08 + 8 * v6) = v5;
  LODWORD(qword_4FFFC10) = qword_4FFFC10 + 1;
  qword_4FFFC48 = 0;
  qword_4FFFC50 = (__int64)&unk_49D9728;
  qword_4FFFC58 = 0;
  qword_4FFFBC0 = (__int64)&unk_49DBF10;
  qword_4FFFC60 = (__int64)&unk_49DC290;
  qword_4FFFC80 = (__int64)nullsub_24;
  qword_4FFFC78 = (__int64)sub_984050;
  sub_C53080(&qword_4FFFBC0, "loop-interchange-max-meminstr-count", 35);
  LODWORD(qword_4FFFC48) = 64;
  BYTE4(qword_4FFFC58) = 1;
  LODWORD(qword_4FFFC58) = 64;
  qword_4FFFBF0 = 161;
  LOBYTE(dword_4FFFBCC) = dword_4FFFBCC & 0x9F | 0x20;
  qword_4FFFBE8 = (__int64)"Maximum number of load-store instructions that should be handled in the dependency matrix. Hi"
                           "gher value may lead to more interchanges at the cost of compile-time";
  sub_C53130(&qword_4FFFBC0);
  __cxa_atexit(sub_984970, &qword_4FFFBC0, &qword_4A427C0);
  v12 = &v11;
  v13 = "Minimum depth of loop nest considered for the transform";
  v14 = 55;
  v10 = 1;
  v11 = 2;
  ((void (__fastcall *)(void *, const char *, int **, int *, const char **))sub_2830A40)(
    &unk_4FFFAE0,
    "loop-interchange-min-loop-nest-depth",
    &v12,
    &v10,
    &v13);
  __cxa_atexit(sub_984970, &unk_4FFFAE0, &qword_4A427C0);
  v12 = &v11;
  v13 = "Maximum depth of loop nest considered for the transform";
  v14 = 55;
  v10 = 1;
  v11 = 10;
  ((void (__fastcall *)(void *, const char *, int **, int *, const char **, __int64))sub_2830A40)(
    &unk_4FFFA00,
    "loop-interchange-max-loop-nest-depth",
    &v12,
    &v10,
    &v13,
    v8);
  return __cxa_atexit(sub_984970, &unk_4FFFA00, &qword_4A427C0);
}
