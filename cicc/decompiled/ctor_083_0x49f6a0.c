// Function: ctor_083
// Address: 0x49f6a0
//
int ctor_083()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-58h]
  _QWORD v14[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v15[64]; // [rsp+20h] [rbp-40h] BYREF

  qword_4F8F9E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8FA30 = 0x100000000LL;
  word_4F8F9F0 = 0;
  dword_4F8F9EC &= 0x8000u;
  qword_4F8F9F8 = 0;
  qword_4F8FA00 = 0;
  dword_4F8F9E8 = v0;
  qword_4F8FA08 = 0;
  qword_4F8FA10 = 0;
  qword_4F8FA18 = 0;
  qword_4F8FA20 = 0;
  qword_4F8FA28 = (__int64)&unk_4F8FA38;
  qword_4F8FA40 = 0;
  qword_4F8FA48 = (__int64)&unk_4F8FA60;
  qword_4F8FA50 = 1;
  dword_4F8FA58 = 0;
  byte_4F8FA5C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8FA30;
  v3 = (unsigned int)qword_4F8FA30 + 1LL;
  if ( v3 > HIDWORD(qword_4F8FA30) )
  {
    sub_C8D5F0((char *)&unk_4F8FA38 - 16, &unk_4F8FA38, v3, 8);
    v2 = (unsigned int)qword_4F8FA30;
  }
  *(_QWORD *)(qword_4F8FA28 + 8 * v2) = v1;
  qword_4F8FA68 = &byte_4F8FA78;
  qword_4F8FA90 = (__int64)&byte_4F8FAA0;
  LODWORD(qword_4F8FA30) = qword_4F8FA30 + 1;
  qword_4F8FA70 = 0;
  qword_4F8FA88 = (__int64)&unk_49DC130;
  byte_4F8FA78 = 0;
  byte_4F8FAA0 = 0;
  qword_4F8F9E0 = (__int64)&unk_49DC010;
  qword_4F8FA98 = 0;
  byte_4F8FAB0 = 0;
  qword_4F8FAB8 = (__int64)&unk_49DC350;
  qword_4F8FAD8 = (__int64)nullsub_92;
  qword_4F8FAD0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4F8F9E0, "dot-cfg-mssa", 12);
  qword_4F8FA18 = (__int64)"file name for generated dot file";
  qword_4F8FA08 = (__int64)"file name for generated dot file";
  v14[0] = v15;
  v14[1] = 0;
  v15[0] = 0;
  qword_4F8FA20 = 32;
  qword_4F8FA10 = 32;
  sub_2240AE0(&qword_4F8FA68, v14);
  byte_4F8FAB0 = 1;
  sub_2240AE0(&qword_4F8FA90, v14);
  sub_2240A30(v14);
  sub_C53130(&qword_4F8F9E0);
  __cxa_atexit(sub_BC5A40, &qword_4F8F9E0, &qword_4A427C0);
  qword_4F8F900 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8F97C = 1;
  qword_4F8F950 = 0x100000000LL;
  dword_4F8F90C &= 0x8000u;
  qword_4F8F918 = 0;
  qword_4F8F920 = 0;
  qword_4F8F928 = 0;
  dword_4F8F908 = v4;
  word_4F8F910 = 0;
  qword_4F8F930 = 0;
  qword_4F8F938 = 0;
  qword_4F8F940 = 0;
  qword_4F8F948 = (__int64)&unk_4F8F958;
  qword_4F8F960 = 0;
  qword_4F8F968 = (__int64)&unk_4F8F980;
  qword_4F8F970 = 1;
  dword_4F8F978 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8F950;
  if ( (unsigned __int64)(unsigned int)qword_4F8F950 + 1 > HIDWORD(qword_4F8F950) )
  {
    v13 = v5;
    sub_C8D5F0((char *)&unk_4F8F958 - 16, &unk_4F8F958, (unsigned int)qword_4F8F950 + 1LL, 8);
    v6 = (unsigned int)qword_4F8F950;
    v5 = v13;
  }
  *(_QWORD *)(qword_4F8F948 + 8 * v6) = v5;
  LODWORD(qword_4F8F950) = qword_4F8F950 + 1;
  qword_4F8F988 = 0;
  qword_4F8F990 = (__int64)&unk_49D9728;
  qword_4F8F998 = 0;
  qword_4F8F900 = (__int64)&unk_49DBF10;
  qword_4F8F9A0 = (__int64)&unk_49DC290;
  qword_4F8F9C0 = (__int64)nullsub_24;
  qword_4F8F9B8 = (__int64)sub_984050;
  sub_C53080(&qword_4F8F900, "memssa-check-limit", 18);
  LODWORD(qword_4F8F988) = 100;
  BYTE4(qword_4F8F998) = 1;
  LODWORD(qword_4F8F998) = 100;
  qword_4F8F930 = 92;
  LOBYTE(dword_4F8F90C) = dword_4F8F90C & 0x9F | 0x20;
  qword_4F8F928 = (__int64)"The maximum number of stores/phis MemorySSAwill consider trying to walk past (default = 100)";
  sub_C53130(&qword_4F8F900);
  __cxa_atexit(sub_984970, &qword_4F8F900, &qword_4A427C0);
  qword_4F8F820 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8F82C &= 0x8000u;
  word_4F8F830 = 0;
  qword_4F8F870 = 0x100000000LL;
  qword_4F8F838 = 0;
  qword_4F8F840 = 0;
  qword_4F8F848 = 0;
  dword_4F8F828 = v7;
  qword_4F8F850 = 0;
  qword_4F8F858 = 0;
  qword_4F8F860 = 0;
  qword_4F8F868 = (__int64)&unk_4F8F878;
  qword_4F8F880 = 0;
  qword_4F8F888 = (__int64)&unk_4F8F8A0;
  qword_4F8F890 = 1;
  dword_4F8F898 = 0;
  byte_4F8F89C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F8F870;
  v10 = (unsigned int)qword_4F8F870 + 1LL;
  if ( v10 > HIDWORD(qword_4F8F870) )
  {
    sub_C8D5F0((char *)&unk_4F8F878 - 16, &unk_4F8F878, v10, 8);
    v9 = (unsigned int)qword_4F8F870;
  }
  *(_QWORD *)(qword_4F8F868 + 8 * v9) = v8;
  LODWORD(qword_4F8F870) = qword_4F8F870 + 1;
  byte_4F8F8B9 = 0;
  qword_4F8F8B0 = (__int64)&unk_49D9748;
  qword_4F8F8A8 = 0;
  qword_4F8F820 = (__int64)&unk_49D9AD8;
  qword_4F8F8C0 = (__int64)&unk_49DC1D0;
  qword_4F8F8E0 = (__int64)nullsub_39;
  qword_4F8F8D8 = (__int64)sub_AA4180;
  sub_C53080(&qword_4F8F820, "verify-memoryssa", 16);
  if ( qword_4F8F8A8 )
  {
    v11 = sub_CEADF0();
    v15[17] = 1;
    v14[0] = "cl::location(x) specified more than once!";
    v15[16] = 3;
    sub_C53280(&qword_4F8F820, v14, 0, 0, v11);
  }
  else
  {
    byte_4F8F8B9 = 1;
    qword_4F8F8A8 = (__int64)byte_4F8F8E8;
    byte_4F8F8B8 = byte_4F8F8E8[0];
  }
  qword_4F8F850 = 33;
  LOBYTE(dword_4F8F82C) = dword_4F8F82C & 0x9F | 0x20;
  qword_4F8F848 = (__int64)"Enable verification of MemorySSA.";
  sub_C53130(&qword_4F8F820);
  return __cxa_atexit(sub_AA4490, &qword_4F8F820, &qword_4A427C0);
}
