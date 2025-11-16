// Function: ctor_641
// Address: 0x595b90
//
int __fastcall ctor_641(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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

  qword_50358E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5035930 = 0x100000000LL;
  word_50358F0 = 0;
  dword_50358EC &= 0x8000u;
  qword_50358F8 = 0;
  qword_5035900 = 0;
  dword_50358E8 = v4;
  qword_5035908 = 0;
  qword_5035910 = 0;
  qword_5035918 = 0;
  qword_5035920 = 0;
  qword_5035928 = (__int64)&unk_5035938;
  qword_5035940 = 0;
  qword_5035948 = (__int64)&unk_5035960;
  qword_5035950 = 1;
  dword_5035958 = 0;
  byte_503595C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5035930;
  v7 = (unsigned int)qword_5035930 + 1LL;
  if ( v7 > HIDWORD(qword_5035930) )
  {
    sub_C8D5F0((char *)&unk_5035938 - 16, &unk_5035938, v7, 8);
    v6 = (unsigned int)qword_5035930;
  }
  *(_QWORD *)(qword_5035928 + 8 * v6) = v5;
  LODWORD(qword_5035930) = qword_5035930 + 1;
  qword_5035968 = 0;
  qword_5035970 = (__int64)&unk_49D9728;
  qword_5035978 = 0;
  qword_50358E0 = (__int64)&unk_49DBF10;
  qword_5035980 = (__int64)&unk_49DC290;
  qword_50359A0 = (__int64)nullsub_24;
  qword_5035998 = (__int64)sub_984050;
  sub_C53080(&qword_50358E0, "sbvec-vec-reg-bits", 18);
  LODWORD(qword_5035968) = 0;
  BYTE4(qword_5035978) = 1;
  LODWORD(qword_5035978) = 0;
  qword_5035910 = 84;
  LOBYTE(dword_50358EC) = dword_50358EC & 0x9F | 0x20;
  qword_5035908 = (__int64)"Override the vector register size in bits, which is otherwise found by querying TTI.";
  sub_C53130(&qword_50358E0);
  __cxa_atexit(sub_984970, &qword_50358E0, &qword_4A427C0);
  qword_5035800 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_50358E0, v8, v9), 1u);
  byte_503587C = 1;
  qword_5035850 = 0x100000000LL;
  dword_503580C &= 0x8000u;
  qword_5035818 = 0;
  qword_5035820 = 0;
  qword_5035828 = 0;
  dword_5035808 = v10;
  word_5035810 = 0;
  qword_5035830 = 0;
  qword_5035838 = 0;
  qword_5035840 = 0;
  qword_5035848 = (__int64)&unk_5035858;
  qword_5035860 = 0;
  qword_5035868 = (__int64)&unk_5035880;
  qword_5035870 = 1;
  dword_5035878 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5035850;
  v13 = (unsigned int)qword_5035850 + 1LL;
  if ( v13 > HIDWORD(qword_5035850) )
  {
    sub_C8D5F0((char *)&unk_5035858 - 16, &unk_5035858, v13, 8);
    v12 = (unsigned int)qword_5035850;
  }
  *(_QWORD *)(qword_5035848 + 8 * v12) = v11;
  LODWORD(qword_5035850) = qword_5035850 + 1;
  qword_5035888 = 0;
  qword_5035890 = (__int64)&unk_49D9748;
  qword_5035898 = 0;
  qword_5035800 = (__int64)&unk_49DC090;
  qword_50358A0 = (__int64)&unk_49DC1D0;
  qword_50358C0 = (__int64)nullsub_23;
  qword_50358B8 = (__int64)sub_984030;
  sub_C53080(&qword_5035800, "sbvec-allow-non-pow2", 20);
  LOBYTE(qword_5035888) = 0;
  LOWORD(qword_5035898) = 256;
  qword_5035830 = 35;
  LOBYTE(dword_503580C) = dword_503580C & 0x9F | 0x20;
  qword_5035828 = (__int64)"Allow non-power-of-2 vectorization.";
  sub_C53130(&qword_5035800);
  return __cxa_atexit(sub_984900, &qword_5035800, &qword_4A427C0);
}
