// Function: ctor_666
// Address: 0x59eaa0
//
int __fastcall ctor_666(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *v15; // [rsp+0h] [rbp-50h] BYREF
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[8]; // [rsp+10h] [rbp-40h] BYREF

  qword_503B020 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503B09C = 1;
  qword_503B070 = 0x100000000LL;
  dword_503B02C &= 0x8000u;
  qword_503B038 = 0;
  qword_503B040 = 0;
  qword_503B048 = 0;
  dword_503B028 = v4;
  word_503B030 = 0;
  qword_503B050 = 0;
  qword_503B058 = 0;
  qword_503B060 = 0;
  qword_503B068 = (__int64)&unk_503B078;
  qword_503B080 = 0;
  qword_503B088 = (__int64)&unk_503B0A0;
  qword_503B090 = 1;
  dword_503B098 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503B070;
  v7 = (unsigned int)qword_503B070 + 1LL;
  if ( v7 > HIDWORD(qword_503B070) )
  {
    sub_C8D5F0((char *)&unk_503B078 - 16, &unk_503B078, v7, 8);
    v6 = (unsigned int)qword_503B070;
  }
  *(_QWORD *)(qword_503B068 + 8 * v6) = v5;
  qword_503B0A8 = (__int64)&byte_503B0B8;
  qword_503B0D0 = (__int64)&byte_503B0E0;
  LODWORD(qword_503B070) = qword_503B070 + 1;
  qword_503B0B0 = 0;
  qword_503B0C8 = (__int64)&unk_49DC130;
  byte_503B0B8 = 0;
  byte_503B0E0 = 0;
  qword_503B020 = (__int64)&unk_49DC010;
  qword_503B0D8 = 0;
  byte_503B0F0 = 0;
  qword_503B0F8 = (__int64)&unk_49DC350;
  qword_503B118 = (__int64)nullsub_92;
  qword_503B110 = (__int64)sub_BC4D70;
  sub_C53080(&qword_503B020, "expandvp-override-evl-transform", 31);
  v15 = v17;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  sub_2240AE0(&qword_503B0A8, &v15);
  byte_503B0F0 = 1;
  sub_2240AE0(&qword_503B0D0, &v15);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  qword_503B050 = 157;
  LOBYTE(dword_503B02C) = dword_503B02C & 0x9F | 0x20;
  qword_503B048 = (__int64)"Options: <empty>|Legal|Discard|Convert. If non-empty, ignore TargetTransformInfo and always u"
                           "se this transformation for the %evl parameter (Used in testing).";
  sub_C53130(&qword_503B020);
  __cxa_atexit(sub_BC5A40, &qword_503B020, &qword_4A427C0);
  qword_503AF20 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_503B020, v8, v9), 1u);
  dword_503AF2C &= 0x8000u;
  word_503AF30 = 0;
  qword_503AF70 = 0x100000000LL;
  qword_503AF38 = 0;
  qword_503AF40 = 0;
  qword_503AF48 = 0;
  dword_503AF28 = v10;
  qword_503AF50 = 0;
  qword_503AF58 = 0;
  qword_503AF60 = 0;
  qword_503AF68 = (__int64)&unk_503AF78;
  qword_503AF80 = 0;
  qword_503AF88 = (__int64)&unk_503AFA0;
  qword_503AF90 = 1;
  dword_503AF98 = 0;
  byte_503AF9C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503AF70;
  v13 = (unsigned int)qword_503AF70 + 1LL;
  if ( v13 > HIDWORD(qword_503AF70) )
  {
    sub_C8D5F0((char *)&unk_503AF78 - 16, &unk_503AF78, v13, 8);
    v12 = (unsigned int)qword_503AF70;
  }
  *(_QWORD *)(qword_503AF68 + 8 * v12) = v11;
  qword_503AFA8 = (__int64)&byte_503AFB8;
  qword_503AFD0 = (__int64)&byte_503AFE0;
  LODWORD(qword_503AF70) = qword_503AF70 + 1;
  qword_503AFB0 = 0;
  qword_503AFC8 = (__int64)&unk_49DC130;
  byte_503AFB8 = 0;
  byte_503AFE0 = 0;
  qword_503AF20 = (__int64)&unk_49DC010;
  qword_503AFD8 = 0;
  byte_503AFF0 = 0;
  qword_503AFF8 = (__int64)&unk_49DC350;
  qword_503B018 = (__int64)nullsub_92;
  qword_503B010 = (__int64)sub_BC4D70;
  sub_C53080(&qword_503AF20, "expandvp-override-mask-transform", 32);
  v15 = v17;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  sub_2240AE0(&qword_503AFA8, &v15);
  byte_503AFF0 = 1;
  sub_2240AE0(&qword_503AFD0, &v15);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  qword_503AF50 = 158;
  LOBYTE(dword_503AF2C) = dword_503AF2C & 0x9F | 0x20;
  qword_503AF48 = (__int64)"Options: <empty>|Legal|Discard|Convert. If non-empty, Ignore TargetTransformInfo and always u"
                           "se this transformation for the %mask parameter (Used in testing).";
  sub_C53130(&qword_503AF20);
  return __cxa_atexit(sub_BC5A40, &qword_503AF20, &qword_4A427C0);
}
