// Function: ctor_569_0
// Address: 0x573a90
//
int ctor_569_0()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // rdx
  int v3; // edx
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v8; // [rsp+8h] [rbp-108h]
  int v9; // [rsp+10h] [rbp-100h] BYREF
  int v10; // [rsp+14h] [rbp-FCh] BYREF
  int *v11; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v12[2]; // [rsp+20h] [rbp-F0h] BYREF
  const char *v13; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v14; // [rsp+38h] [rbp-D8h]
  _QWORD v15[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v16; // [rsp+50h] [rbp-C0h]
  const char *v17; // [rsp+58h] [rbp-B8h]
  __int64 v18; // [rsp+60h] [rbp-B0h]
  char *v19; // [rsp+68h] [rbp-A8h]
  __int64 v20; // [rsp+70h] [rbp-A0h]
  int v21; // [rsp+78h] [rbp-98h]
  const char *v22; // [rsp+80h] [rbp-90h]
  __int64 v23; // [rsp+88h] [rbp-88h]
  char *v24; // [rsp+90h] [rbp-80h]
  __int64 v25; // [rsp+98h] [rbp-78h]
  int v26; // [rsp+A0h] [rbp-70h]
  const char *v27; // [rsp+A8h] [rbp-68h]
  __int64 v28; // [rsp+B0h] [rbp-60h]

  v13 = "MachineLICM should avoid speculation";
  LODWORD(v11) = 1;
  LOBYTE(v10) = 1;
  v12[0] = &v10;
  v14 = 36;
  sub_2E99010(&unk_50207E0, "avoid-speculation", &v13, v12, &v11);
  __cxa_atexit(sub_984900, &unk_50207E0, &qword_4A427C0);
  LODWORD(v11) = 1;
  v13 = "MachineLICM should hoist even cheap instructions";
  LOBYTE(v10) = 0;
  v12[0] = &v10;
  v14 = 48;
  sub_2E99010(&unk_5020700, "hoist-cheap-insts", &v13, v12, &v11);
  __cxa_atexit(sub_984900, &unk_5020700, &qword_4A427C0);
  qword_5020620 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5020670 = 0x100000000LL;
  word_5020630 = 0;
  dword_502062C &= 0x8000u;
  qword_5020638 = 0;
  qword_5020640 = 0;
  dword_5020628 = v0;
  qword_5020648 = 0;
  qword_5020650 = 0;
  qword_5020658 = 0;
  qword_5020660 = 0;
  qword_5020668 = (__int64)&unk_5020678;
  qword_5020680 = 0;
  qword_5020688 = (__int64)&unk_50206A0;
  qword_5020690 = 1;
  dword_5020698 = 0;
  byte_502069C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5020670;
  if ( (unsigned __int64)(unsigned int)qword_5020670 + 1 > HIDWORD(qword_5020670) )
  {
    v8 = v1;
    sub_C8D5F0((char *)&unk_5020678 - 16, &unk_5020678, (unsigned int)qword_5020670 + 1LL, 8);
    v2 = (unsigned int)qword_5020670;
    v1 = v8;
  }
  *(_QWORD *)(qword_5020668 + 8 * v2) = v1;
  LODWORD(qword_5020670) = qword_5020670 + 1;
  qword_50206A8 = 0;
  qword_50206B0 = (__int64)&unk_49D9748;
  qword_50206B8 = 0;
  qword_5020620 = (__int64)&unk_49DC090;
  qword_50206C0 = (__int64)&unk_49DC1D0;
  qword_50206E0 = (__int64)nullsub_23;
  qword_50206D8 = (__int64)sub_984030;
  sub_C53080(&qword_5020620, "hoist-const-stores", 18);
  qword_5020650 = 22;
  qword_5020648 = (__int64)"Hoist invariant stores";
  LOWORD(qword_50206B8) = 257;
  LOBYTE(qword_50206A8) = 1;
  LOBYTE(dword_502062C) = dword_502062C & 0x9F | 0x20;
  sub_C53130(&qword_5020620);
  __cxa_atexit(sub_984900, &qword_5020620, &qword_4A427C0);
  LODWORD(v11) = 1;
  v13 = "Hoist invariant loads";
  LOBYTE(v10) = 1;
  v12[0] = &v10;
  v14 = 21;
  sub_2E99010(&unk_5020540, "hoist-const-loads", &v13, v12, &v11);
  __cxa_atexit(sub_984900, &unk_5020540, &qword_4A427C0);
  qword_5020460 = (__int64)&unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50204DC = 1;
  qword_50204B0 = 0x100000000LL;
  dword_502046C &= 0x8000u;
  qword_50204A8 = (__int64)&unk_50204B8;
  qword_5020478 = 0;
  qword_5020480 = 0;
  dword_5020468 = v3;
  word_5020470 = 0;
  qword_5020488 = 0;
  qword_5020490 = 0;
  qword_5020498 = 0;
  qword_50204A0 = 0;
  qword_50204C0 = 0;
  qword_50204C8 = (__int64)&unk_50204E0;
  qword_50204D0 = 1;
  dword_50204D8 = 0;
  v4 = sub_C57470();
  v5 = (unsigned int)qword_50204B0;
  v6 = (unsigned int)qword_50204B0 + 1LL;
  if ( v6 > HIDWORD(qword_50204B0) )
  {
    sub_C8D5F0((char *)&unk_50204B8 - 16, &unk_50204B8, v6, 8);
    v5 = (unsigned int)qword_50204B0;
  }
  *(_QWORD *)(qword_50204A8 + 8 * v5) = v4;
  LODWORD(qword_50204B0) = qword_50204B0 + 1;
  qword_50204E8 = 0;
  qword_50204F0 = (__int64)&unk_49D9728;
  qword_50204F8 = 0;
  qword_5020460 = (__int64)&unk_49DBF10;
  qword_5020500 = (__int64)&unk_49DC290;
  qword_5020520 = (__int64)nullsub_24;
  qword_5020518 = (__int64)sub_984050;
  sub_C53080(&qword_5020460, "block-freq-ratio-threshold", 26);
  qword_5020490 = 75;
  qword_5020488 = (__int64)"Do not hoist instructions if targetblock is N times hotter than the source.";
  LODWORD(qword_50204E8) = 100;
  BYTE4(qword_50204F8) = 1;
  LODWORD(qword_50204F8) = 100;
  LOBYTE(dword_502046C) = dword_502046C & 0x9F | 0x20;
  sub_C53130(&qword_5020460);
  __cxa_atexit(sub_984970, &qword_5020460, &qword_4A427C0);
  v13 = (const char *)v15;
  v15[0] = "none";
  v17 = "disable the feature";
  v19 = "pgo";
  v22 = "enable the feature when using profile data";
  v24 = "all";
  v27 = "enable the feature with/wo profile data";
  v14 = 0x400000003LL;
  v11 = &v9;
  v15[1] = 4;
  v16 = 0;
  v18 = 19;
  v20 = 3;
  v21 = 1;
  v23 = 42;
  v25 = 3;
  v26 = 2;
  v28 = 39;
  v10 = 1;
  v9 = 1;
  v12[0] = "Disable hoisting instructions to hotter blocks";
  v12[1] = 46;
  sub_2EA3C20(&unk_5020200, "disable-hoisting-to-hotter-blocks", v12, &v11, &v10, &v13);
  if ( v13 != (const char *)v15 )
    _libc_free(v13, "disable-hoisting-to-hotter-blocks");
  return __cxa_atexit(sub_2E97FC0, &unk_5020200, &qword_4A427C0);
}
