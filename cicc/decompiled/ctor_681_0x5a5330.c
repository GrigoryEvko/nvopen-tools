// Function: ctor_681
// Address: 0x5a5330
//
int __fastcall ctor_681(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_503F6C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503F73C = 1;
  qword_503F710 = 0x100000000LL;
  dword_503F6CC &= 0x8000u;
  qword_503F6D8 = 0;
  qword_503F6E0 = 0;
  qword_503F6E8 = 0;
  dword_503F6C8 = v4;
  word_503F6D0 = 0;
  qword_503F6F0 = 0;
  qword_503F6F8 = 0;
  qword_503F700 = 0;
  qword_503F708 = (__int64)&unk_503F718;
  qword_503F720 = 0;
  qword_503F728 = (__int64)&unk_503F740;
  qword_503F730 = 1;
  dword_503F738 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503F710;
  v7 = (unsigned int)qword_503F710 + 1LL;
  if ( v7 > HIDWORD(qword_503F710) )
  {
    sub_C8D5F0((char *)&unk_503F718 - 16, &unk_503F718, v7, 8);
    v6 = (unsigned int)qword_503F710;
  }
  *(_QWORD *)(qword_503F708 + 8 * v6) = v5;
  LODWORD(qword_503F710) = qword_503F710 + 1;
  qword_503F748 = 0;
  qword_503F750 = (__int64)&unk_49D9748;
  qword_503F758 = 0;
  qword_503F6C0 = (__int64)&unk_49DC090;
  qword_503F760 = (__int64)&unk_49DC1D0;
  qword_503F780 = (__int64)nullsub_23;
  qword_503F778 = (__int64)sub_984030;
  sub_C53080(&qword_503F6C0, "mir-vreg-namer-use-stable-hash", 30);
  LOBYTE(qword_503F748) = 0;
  LOWORD(qword_503F758) = 256;
  qword_503F6F0 = 40;
  LOBYTE(dword_503F6CC) = dword_503F6CC & 0x9F | 0x20;
  qword_503F6E8 = (__int64)"Use Stable Hashing for MIR VReg Renaming";
  sub_C53130(&qword_503F6C0);
  return __cxa_atexit(sub_984900, &qword_503F6C0, &qword_4A427C0);
}
