// Function: ctor_507
// Address: 0x55bb40
//
__int64 ctor_507()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 result; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v10[8]; // [rsp+30h] [rbp-40h] BYREF

  qword_500B120 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500B19C = 1;
  qword_500B170 = 0x100000000LL;
  dword_500B12C &= 0x8000u;
  qword_500B138 = 0;
  qword_500B140 = 0;
  qword_500B148 = 0;
  dword_500B128 = v0;
  word_500B130 = 0;
  qword_500B150 = 0;
  qword_500B158 = 0;
  qword_500B160 = 0;
  qword_500B168 = (__int64)&unk_500B178;
  qword_500B180 = 0;
  qword_500B188 = (__int64)&unk_500B1A0;
  qword_500B190 = 1;
  dword_500B198 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500B170;
  v3 = (unsigned int)qword_500B170 + 1LL;
  if ( v3 > HIDWORD(qword_500B170) )
  {
    sub_C8D5F0((char *)&unk_500B178 - 16, &unk_500B178, v3, 8);
    v2 = (unsigned int)qword_500B170;
  }
  *(_QWORD *)(qword_500B168 + 8 * v2) = v1;
  LODWORD(qword_500B170) = qword_500B170 + 1;
  qword_500B1A8 = 0;
  qword_500B1B0 = (__int64)&unk_49D9748;
  qword_500B1B8 = 0;
  qword_500B120 = (__int64)&unk_49DC090;
  qword_500B1C0 = (__int64)&unk_49DC1D0;
  qword_500B1E0 = (__int64)nullsub_23;
  qword_500B1D8 = (__int64)sub_984030;
  sub_C53080(&qword_500B120, "verify-predicateinfo", 20);
  LOBYTE(qword_500B1A8) = 0;
  LOWORD(qword_500B1B8) = 256;
  qword_500B150 = 44;
  LOBYTE(dword_500B12C) = dword_500B12C & 0x9F | 0x20;
  qword_500B148 = (__int64)"Verify PredicateInfo in legacy printer pass.";
  sub_C53130(&qword_500B120);
  __cxa_atexit(sub_984900, &qword_500B120, &qword_4A427C0);
  v4 = sub_C60B10();
  v9[0] = v10;
  v5 = v4;
  sub_2A44420(v9, "Controls which variables are renamed with predicateinfo");
  v7[0] = v8;
  sub_2A44420(v7, "predicateinfo-rename");
  result = sub_CF9810(v5, v7, v9);
  if ( (_QWORD *)v7[0] != v8 )
    result = j_j___libc_free_0(v7[0], v8[0] + 1LL);
  if ( (_QWORD *)v9[0] != v10 )
    return j_j___libc_free_0(v9[0], v10[0] + 1LL);
  return result;
}
