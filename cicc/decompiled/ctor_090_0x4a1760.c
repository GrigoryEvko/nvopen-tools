// Function: ctor_090
// Address: 0x4a1760
//
int ctor_090()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // rbx
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
  sub_1168C90(v13, "Controls Negator transformations in InstCombine pass");
  v11[0] = v12;
  sub_1168C90(v11, "instcombine-negator");
  sub_CF9810(v1, v11, v13);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], v14[0] + 1LL);
  qword_4F90900 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F90950 = 0x100000000LL;
  word_4F90910 = 0;
  dword_4F9090C &= 0x8000u;
  qword_4F90918 = 0;
  qword_4F90920 = 0;
  dword_4F90908 = v2;
  qword_4F90928 = 0;
  qword_4F90930 = 0;
  qword_4F90938 = 0;
  qword_4F90940 = 0;
  qword_4F90948 = (__int64)&unk_4F90958;
  qword_4F90960 = 0;
  qword_4F90968 = (__int64)&unk_4F90980;
  qword_4F90970 = 1;
  dword_4F90978 = 0;
  byte_4F9097C = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_4F90950;
  v5 = (unsigned int)qword_4F90950 + 1LL;
  if ( v5 > HIDWORD(qword_4F90950) )
  {
    sub_C8D5F0((char *)&unk_4F90958 - 16, &unk_4F90958, v5, 8);
    v4 = (unsigned int)qword_4F90950;
  }
  *(_QWORD *)(qword_4F90948 + 8 * v4) = v3;
  LODWORD(qword_4F90950) = qword_4F90950 + 1;
  qword_4F90988 = 0;
  qword_4F90990 = (__int64)&unk_49D9748;
  qword_4F90998 = 0;
  qword_4F90900 = (__int64)&unk_49DC090;
  qword_4F909A0 = (__int64)&unk_49DC1D0;
  qword_4F909C0 = (__int64)nullsub_23;
  qword_4F909B8 = (__int64)sub_984030;
  sub_C53080(&qword_4F90900, "instcombine-negator-enabled", 27);
  LOBYTE(qword_4F90988) = 1;
  LOWORD(qword_4F90998) = 257;
  qword_4F90928 = (__int64)"Should we attempt to sink negations?";
  qword_4F90930 = 36;
  sub_C53130(&qword_4F90900);
  __cxa_atexit(sub_984900, &qword_4F90900, &qword_4A427C0);
  qword_4F90820 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F9089C = 1;
  qword_4F90870 = 0x100000000LL;
  dword_4F9082C &= 0x8000u;
  qword_4F90838 = 0;
  qword_4F90840 = 0;
  qword_4F90848 = 0;
  dword_4F90828 = v6;
  word_4F90830 = 0;
  qword_4F90850 = 0;
  qword_4F90858 = 0;
  qword_4F90860 = 0;
  qword_4F90868 = (__int64)&unk_4F90878;
  qword_4F90880 = 0;
  qword_4F90888 = (__int64)&unk_4F908A0;
  qword_4F90890 = 1;
  dword_4F90898 = 0;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4F90870;
  v9 = (unsigned int)qword_4F90870 + 1LL;
  if ( v9 > HIDWORD(qword_4F90870) )
  {
    sub_C8D5F0((char *)&unk_4F90878 - 16, &unk_4F90878, v9, 8);
    v8 = (unsigned int)qword_4F90870;
  }
  *(_QWORD *)(qword_4F90868 + 8 * v8) = v7;
  LODWORD(qword_4F90870) = qword_4F90870 + 1;
  qword_4F908A8 = 0;
  qword_4F908B0 = (__int64)&unk_49D9728;
  qword_4F908B8 = 0;
  qword_4F90820 = (__int64)&unk_49DBF10;
  qword_4F908C0 = (__int64)&unk_49DC290;
  qword_4F908E0 = (__int64)nullsub_24;
  qword_4F908D8 = (__int64)sub_984050;
  sub_C53080(&qword_4F90820, "instcombine-negator-max-depth", 29);
  LODWORD(qword_4F908A8) = -1;
  BYTE4(qword_4F908B8) = 1;
  LODWORD(qword_4F908B8) = -1;
  qword_4F90848 = (__int64)"What is the maximal lookup depth when trying to check for viability of negation sinking.";
  qword_4F90850 = 88;
  sub_C53130(&qword_4F90820);
  return __cxa_atexit(sub_984970, &qword_4F90820, &qword_4A427C0);
}
