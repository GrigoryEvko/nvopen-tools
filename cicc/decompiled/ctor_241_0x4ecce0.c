// Function: ctor_241
// Address: 0x4ecce0
//
__int64 ctor_241()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 result; // rax
  _QWORD v4[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v6[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v7[8]; // [rsp+30h] [rbp-40h] BYREF

  qword_4FB6C40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB6C4C &= 0xF000u;
  qword_4FB6C88 = (__int64)qword_4FA01C0;
  qword_4FB6C98 = (__int64)&unk_4FB6CB8;
  qword_4FB6CA0 = (__int64)&unk_4FB6CB8;
  qword_4FB6C50 = 0;
  dword_4FB6C48 = v0;
  qword_4FB6CE8 = (__int64)&unk_49E74E8;
  word_4FB6CF0 = 256;
  qword_4FB6C58 = 0;
  qword_4FB6C60 = 0;
  qword_4FB6C40 = (__int64)&unk_49EEC70;
  qword_4FB6C68 = 0;
  qword_4FB6C70 = 0;
  qword_4FB6CF8 = (__int64)&unk_49EEDB0;
  qword_4FB6C78 = 0;
  qword_4FB6C80 = 0;
  qword_4FB6C90 = 0;
  qword_4FB6CA8 = 4;
  dword_4FB6CB0 = 0;
  byte_4FB6CD8 = 0;
  byte_4FB6CE0 = 0;
  sub_16B8280(&qword_4FB6C40, "verify-predicateinfo", 20);
  word_4FB6CF0 = 256;
  byte_4FB6CE0 = 0;
  qword_4FB6C70 = 44;
  LOBYTE(word_4FB6C4C) = word_4FB6C4C & 0x9F | 0x20;
  qword_4FB6C68 = (__int64)"Verify PredicateInfo in legacy printer pass.";
  sub_16B88A0(&qword_4FB6C40);
  __cxa_atexit(sub_12EDEC0, &qword_4FB6C40, &qword_4A427C0);
  v1 = sub_16BAF20();
  v6[0] = v7;
  v2 = v1;
  sub_1B29F00(v6, "Controls which variables are renamed with predicateinfo");
  v4[0] = v5;
  sub_1B29F00(v4, "predicateinfo-rename");
  result = sub_14C9E50(v2, v4, v6);
  if ( (_QWORD *)v4[0] != v5 )
    result = j_j___libc_free_0(v4[0], v5[0] + 1LL);
  if ( (_QWORD *)v6[0] != v7 )
    return j_j___libc_free_0(v6[0], v7[0] + 1LL);
  return result;
}
