// Function: ctor_462_0
// Address: 0x547ab0
//
int ctor_462_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v5; // [rsp+0h] [rbp-E0h] BYREF
  int v6; // [rsp+4h] [rbp-DCh] BYREF
  int *v7; // [rsp+8h] [rbp-D8h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-B0h] BYREF
  int v11; // [rsp+40h] [rbp-A0h]
  const char *v12; // [rsp+48h] [rbp-98h]
  __int64 v13; // [rsp+50h] [rbp-90h]
  char *v14; // [rsp+58h] [rbp-88h]
  __int64 v15; // [rsp+60h] [rbp-80h]
  int v16; // [rsp+68h] [rbp-78h]
  const char *v17; // [rsp+70h] [rbp-70h]
  __int64 v18; // [rsp+78h] [rbp-68h]
  char *v19; // [rsp+80h] [rbp-60h]
  __int64 v20; // [rsp+88h] [rbp-58h]
  int v21; // [rsp+90h] [rbp-50h]
  const char *v22; // [rsp+98h] [rbp-48h]
  __int64 v23; // [rsp+A0h] [rbp-40h]

  v7 = &v6;
  v10[0] = "scev";
  v12 = "Use the scalar evolution interface";
  v14 = "da";
  v17 = "Use the dependence analysis interface";
  v19 = "all";
  v22 = "Use all available analyses";
  v9[1] = 0x400000003LL;
  v6 = 2;
  v5 = 1;
  v9[0] = v10;
  v10[1] = 4;
  v11 = 0;
  v13 = 34;
  v15 = 2;
  v16 = 1;
  v18 = 37;
  v20 = 3;
  v21 = 2;
  v23 = 26;
  v8[0] = "Which dependence analysis should loop fusion use?";
  v8[1] = 49;
  ((void (__fastcall *)(void *, const char *, _QWORD *, _QWORD *, int *, int **))sub_281D8C0)(
    &unk_4FFF420,
    "loop-fusion-dependence-analysis",
    v8,
    v9,
    &v5,
    &v7);
  if ( (_QWORD *)v9[0] != v10 )
    _libc_free(v9[0], "loop-fusion-dependence-analysis");
  __cxa_atexit(sub_28141D0, &unk_4FFF420, &qword_4A427C0);
  qword_4FFF340 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFF34C &= 0x8000u;
  word_4FFF350 = 0;
  qword_4FFF390 = 0x100000000LL;
  qword_4FFF358 = 0;
  qword_4FFF360 = 0;
  qword_4FFF368 = 0;
  dword_4FFF348 = v0;
  qword_4FFF370 = 0;
  qword_4FFF378 = 0;
  qword_4FFF380 = 0;
  qword_4FFF388 = (__int64)&unk_4FFF398;
  qword_4FFF3A0 = 0;
  qword_4FFF3A8 = (__int64)&unk_4FFF3C0;
  qword_4FFF3B0 = 1;
  dword_4FFF3B8 = 0;
  byte_4FFF3BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFF390;
  v3 = (unsigned int)qword_4FFF390 + 1LL;
  if ( v3 > HIDWORD(qword_4FFF390) )
  {
    sub_C8D5F0((char *)&unk_4FFF398 - 16, &unk_4FFF398, v3, 8);
    v2 = (unsigned int)qword_4FFF390;
  }
  *(_QWORD *)(qword_4FFF388 + 8 * v2) = v1;
  LODWORD(qword_4FFF390) = qword_4FFF390 + 1;
  qword_4FFF3C8 = 0;
  qword_4FFF3D0 = (__int64)&unk_49D9728;
  qword_4FFF3D8 = 0;
  qword_4FFF340 = (__int64)&unk_49DBF10;
  qword_4FFF3E0 = (__int64)&unk_49DC290;
  qword_4FFF400 = (__int64)nullsub_24;
  qword_4FFF3F8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFF340, "loop-fusion-peel-max-count", 26);
  LODWORD(qword_4FFF3C8) = 0;
  BYTE4(qword_4FFF3D8) = 1;
  LODWORD(qword_4FFF3D8) = 0;
  qword_4FFF370 = 82;
  LOBYTE(dword_4FFF34C) = dword_4FFF34C & 0x9F | 0x20;
  qword_4FFF368 = (__int64)"Max number of iterations to be peeled from a loop, such that fusion can take place";
  sub_C53130(&qword_4FFF340);
  return __cxa_atexit(sub_984970, &qword_4FFF340, &qword_4A427C0);
}
