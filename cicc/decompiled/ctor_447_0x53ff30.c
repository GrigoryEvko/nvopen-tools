// Function: ctor_447
// Address: 0x53ff30
//
int ctor_447()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // r12
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD v11[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v14[8]; // [rsp+30h] [rbp-40h] BYREF

  v0 = sub_C60B10();
  v13[0] = v14;
  v1 = v0;
  sub_27786E0(v13, "Controls which instructions are removed");
  v11[0] = v12;
  sub_27786E0(v11, "early-cse");
  sub_CF9810(v1, v11, v13);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], v14[0] + 1LL);
  qword_4FFB1E0 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFB230 = 0x100000000LL;
  word_4FFB1F0 = 0;
  dword_4FFB1EC &= 0x8000u;
  qword_4FFB1F8 = 0;
  qword_4FFB200 = 0;
  dword_4FFB1E8 = v2;
  qword_4FFB208 = 0;
  qword_4FFB210 = 0;
  qword_4FFB218 = 0;
  qword_4FFB220 = 0;
  qword_4FFB228 = (__int64)&unk_4FFB238;
  qword_4FFB240 = 0;
  qword_4FFB248 = (__int64)&unk_4FFB260;
  qword_4FFB250 = 1;
  dword_4FFB258 = 0;
  byte_4FFB25C = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_4FFB230;
  v5 = (unsigned int)qword_4FFB230 + 1LL;
  if ( v5 > HIDWORD(qword_4FFB230) )
  {
    sub_C8D5F0((char *)&unk_4FFB238 - 16, &unk_4FFB238, v5, 8);
    v4 = (unsigned int)qword_4FFB230;
  }
  *(_QWORD *)(qword_4FFB228 + 8 * v4) = v3;
  LODWORD(qword_4FFB230) = qword_4FFB230 + 1;
  qword_4FFB268 = 0;
  qword_4FFB270 = (__int64)&unk_49D9728;
  qword_4FFB278 = 0;
  qword_4FFB1E0 = (__int64)&unk_49DBF10;
  qword_4FFB280 = (__int64)&unk_49DC290;
  qword_4FFB2A0 = (__int64)nullsub_24;
  qword_4FFB298 = (__int64)sub_984050;
  sub_C53080(&qword_4FFB1E0, "earlycse-mssa-optimization-cap", 30);
  LODWORD(qword_4FFB268) = 500;
  BYTE4(qword_4FFB278) = 1;
  LODWORD(qword_4FFB278) = 500;
  qword_4FFB210 = 122;
  LOBYTE(dword_4FFB1EC) = dword_4FFB1EC & 0x9F | 0x20;
  qword_4FFB208 = (__int64)"Enable imprecision in EarlyCSE in pathological cases, in exchange for faster compile. Caps th"
                           "e MemorySSA clobbering calls.";
  sub_C53130(&qword_4FFB1E0);
  __cxa_atexit(sub_984970, &qword_4FFB1E0, &qword_4A427C0);
  qword_4FFB100 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFB17C = 1;
  qword_4FFB150 = 0x100000000LL;
  dword_4FFB10C &= 0x8000u;
  qword_4FFB118 = 0;
  qword_4FFB120 = 0;
  qword_4FFB128 = 0;
  dword_4FFB108 = v6;
  word_4FFB110 = 0;
  qword_4FFB130 = 0;
  qword_4FFB138 = 0;
  qword_4FFB140 = 0;
  qword_4FFB148 = (__int64)&unk_4FFB158;
  qword_4FFB160 = 0;
  qword_4FFB168 = (__int64)&unk_4FFB180;
  qword_4FFB170 = 1;
  dword_4FFB178 = 0;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FFB150;
  v9 = (unsigned int)qword_4FFB150 + 1LL;
  if ( v9 > HIDWORD(qword_4FFB150) )
  {
    sub_C8D5F0((char *)&unk_4FFB158 - 16, &unk_4FFB158, v9, 8);
    v8 = (unsigned int)qword_4FFB150;
  }
  *(_QWORD *)(qword_4FFB148 + 8 * v8) = v7;
  LODWORD(qword_4FFB150) = qword_4FFB150 + 1;
  qword_4FFB188 = 0;
  qword_4FFB190 = (__int64)&unk_49D9748;
  qword_4FFB198 = 0;
  qword_4FFB100 = (__int64)&unk_49DC090;
  qword_4FFB1A0 = (__int64)&unk_49DC1D0;
  qword_4FFB1C0 = (__int64)nullsub_23;
  qword_4FFB1B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFB100, "earlycse-debug-hash", 19);
  LOBYTE(qword_4FFB188) = 0;
  LOWORD(qword_4FFB198) = 256;
  qword_4FFB130 = 120;
  LOBYTE(dword_4FFB10C) = dword_4FFB10C & 0x9F | 0x20;
  qword_4FFB128 = (__int64)"Perform extra assertion checking to verify that SimpleValue's hash function is well-behaved w"
                           ".r.t. its isEqual predicate";
  sub_C53130(&qword_4FFB100);
  return __cxa_atexit(sub_984900, &qword_4FFB100, &qword_4A427C0);
}
