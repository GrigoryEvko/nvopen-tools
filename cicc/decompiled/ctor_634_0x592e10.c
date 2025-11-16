// Function: ctor_634
// Address: 0x592e10
//
int __fastcall ctor_634(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5032A00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5032A50 = 0x100000000LL;
  word_5032A10 = 0;
  dword_5032A0C &= 0x8000u;
  qword_5032A18 = 0;
  qword_5032A20 = 0;
  dword_5032A08 = v4;
  qword_5032A28 = 0;
  qword_5032A30 = 0;
  qword_5032A38 = 0;
  qword_5032A40 = 0;
  qword_5032A48 = (__int64)&unk_5032A58;
  qword_5032A60 = 0;
  qword_5032A68 = (__int64)&unk_5032A80;
  qword_5032A70 = 1;
  dword_5032A78 = 0;
  byte_5032A7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5032A50;
  v7 = (unsigned int)qword_5032A50 + 1LL;
  if ( v7 > HIDWORD(qword_5032A50) )
  {
    sub_C8D5F0((char *)&unk_5032A58 - 16, &unk_5032A58, v7, 8);
    v6 = (unsigned int)qword_5032A50;
  }
  *(_QWORD *)(qword_5032A48 + 8 * v6) = v5;
  LODWORD(qword_5032A50) = qword_5032A50 + 1;
  qword_5032A88 = 0;
  qword_5032A90 = (__int64)&unk_49D9748;
  qword_5032A98 = 0;
  qword_5032A00 = (__int64)&unk_49DC090;
  qword_5032AA0 = (__int64)&unk_49DC1D0;
  qword_5032AC0 = (__int64)nullsub_23;
  qword_5032AB8 = (__int64)sub_984030;
  sub_C53080(&qword_5032A00, "openmp-ir-builder-optimistic-attributes", 39);
  qword_5032A30 = 73;
  LOBYTE(qword_5032A88) = 0;
  LOBYTE(dword_5032A0C) = dword_5032A0C & 0x9F | 0x20;
  qword_5032A28 = (__int64)"Use optimistic attributes describing 'as-if' properties of runtime calls.";
  LOWORD(qword_5032A98) = 256;
  sub_C53130(&qword_5032A00);
  __cxa_atexit(sub_984900, &qword_5032A00, &qword_4A427C0);
  qword_5032920 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5032A00, v8, v9), 1u);
  byte_503299C = 1;
  qword_5032970 = 0x100000000LL;
  dword_503292C &= 0x8000u;
  qword_5032938 = 0;
  qword_5032940 = 0;
  qword_5032948 = 0;
  dword_5032928 = v10;
  word_5032930 = 0;
  qword_5032950 = 0;
  qword_5032958 = 0;
  qword_5032960 = 0;
  qword_5032968 = (__int64)&unk_5032978;
  qword_5032980 = 0;
  qword_5032988 = (__int64)&unk_50329A0;
  qword_5032990 = 1;
  dword_5032998 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5032970;
  v13 = (unsigned int)qword_5032970 + 1LL;
  if ( v13 > HIDWORD(qword_5032970) )
  {
    sub_C8D5F0((char *)&unk_5032978 - 16, &unk_5032978, v13, 8);
    v12 = (unsigned int)qword_5032970;
  }
  *(_QWORD *)(qword_5032968 + 8 * v12) = v11;
  LODWORD(qword_5032970) = qword_5032970 + 1;
  byte_50329C0 = 0;
  qword_50329B0 = (__int64)&unk_49DE5F0;
  qword_50329A8 = 0;
  qword_50329B8 = 0;
  qword_5032920 = (__int64)&unk_49DE610;
  qword_50329C8 = (__int64)&unk_49DC2F0;
  qword_50329E8 = (__int64)nullsub_190;
  qword_50329E0 = (__int64)sub_D83E80;
  sub_C53080(&qword_5032920, "openmp-ir-builder-unroll-threshold-factor", 41);
  qword_5032950 = 86;
  byte_50329C0 = 1;
  qword_50329A8 = 0x3FF8000000000000LL;
  LOBYTE(dword_503292C) = dword_503292C & 0x9F | 0x20;
  qword_5032948 = (__int64)"Factor for the unroll threshold to account for code simplifications still taking place";
  qword_50329B8 = 0x3FF8000000000000LL;
  sub_C53130(&qword_5032920);
  return __cxa_atexit(sub_D84280, &qword_5032920, &qword_4A427C0);
}
