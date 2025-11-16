// Function: ctor_610
// Address: 0x5888a0
//
int __fastcall ctor_610(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v26; // [rsp+8h] [rbp-58h]
  char v27; // [rsp+13h] [rbp-4Dh] BYREF
  int v28; // [rsp+14h] [rbp-4Ch] BYREF
  char *v29; // [rsp+18h] [rbp-48h] BYREF
  const char *v30; // [rsp+20h] [rbp-40h] BYREF
  __int64 v31; // [rsp+28h] [rbp-38h]

  qword_502D120 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_502D170 = 0x100000000LL;
  dword_502D12C &= 0x8000u;
  word_502D130 = 0;
  qword_502D138 = 0;
  qword_502D140 = 0;
  dword_502D128 = v4;
  qword_502D148 = 0;
  qword_502D150 = 0;
  qword_502D158 = 0;
  qword_502D160 = 0;
  qword_502D168 = (__int64)&unk_502D178;
  qword_502D180 = 0;
  qword_502D188 = (__int64)&unk_502D1A0;
  qword_502D190 = 1;
  dword_502D198 = 0;
  byte_502D19C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502D170;
  v7 = (unsigned int)qword_502D170 + 1LL;
  if ( v7 > HIDWORD(qword_502D170) )
  {
    sub_C8D5F0((char *)&unk_502D178 - 16, &unk_502D178, v7, 8);
    v6 = (unsigned int)qword_502D170;
  }
  *(_QWORD *)(qword_502D168 + 8 * v6) = v5;
  LODWORD(qword_502D170) = qword_502D170 + 1;
  qword_502D1A8 = 0;
  qword_502D1B0 = (__int64)&unk_49D9728;
  qword_502D1B8 = 0;
  qword_502D120 = (__int64)&unk_49DBF10;
  qword_502D1C0 = (__int64)&unk_49DC290;
  qword_502D1E0 = (__int64)nullsub_24;
  qword_502D1D8 = (__int64)sub_984050;
  sub_C53080(&qword_502D120, "unroll-assumed-size", 19);
  LODWORD(qword_502D1A8) = 4;
  BYTE4(qword_502D1B8) = 1;
  LODWORD(qword_502D1B8) = 4;
  qword_502D150 = 45;
  LOBYTE(dword_502D12C) = dword_502D12C & 0x9F | 0x20;
  qword_502D148 = (__int64)"Assumed size for unknown types of local array";
  sub_C53130(&qword_502D120);
  __cxa_atexit(sub_984970, &qword_502D120, &qword_4A427C0);
  qword_502D040 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_502D120, v8, v9), 1u);
  qword_502D090 = 0x100000000LL;
  dword_502D04C &= 0x8000u;
  word_502D050 = 0;
  qword_502D058 = 0;
  qword_502D060 = 0;
  dword_502D048 = v10;
  qword_502D068 = 0;
  qword_502D070 = 0;
  qword_502D078 = 0;
  qword_502D080 = 0;
  qword_502D088 = (__int64)&unk_502D098;
  qword_502D0A0 = 0;
  qword_502D0A8 = (__int64)&unk_502D0C0;
  qword_502D0B0 = 1;
  dword_502D0B8 = 0;
  byte_502D0BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502D090;
  v13 = (unsigned int)qword_502D090 + 1LL;
  if ( v13 > HIDWORD(qword_502D090) )
  {
    sub_C8D5F0((char *)&unk_502D098 - 16, &unk_502D098, v13, 8);
    v12 = (unsigned int)qword_502D090;
  }
  *(_QWORD *)(qword_502D088 + 8 * v12) = v11;
  qword_502D0D0 = (__int64)&unk_49D9748;
  qword_502D040 = (__int64)&unk_49DC090;
  qword_502D0E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502D090) = qword_502D090 + 1;
  qword_502D100 = (__int64)nullsub_23;
  qword_502D0C8 = 0;
  qword_502D0F8 = (__int64)sub_984030;
  qword_502D0D8 = 0;
  sub_C53080(&qword_502D040, "enable-loop-peeling", 19);
  LOWORD(qword_502D0D8) = 256;
  LOBYTE(qword_502D0C8) = 0;
  qword_502D070 = 19;
  LOBYTE(dword_502D04C) = dword_502D04C & 0x9F | 0x20;
  qword_502D068 = (__int64)"Enable loop peeling";
  sub_C53130(&qword_502D040);
  __cxa_atexit(sub_984900, &qword_502D040, &qword_4A427C0);
  qword_502CF60 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502D040, v14, v15), 1u);
  qword_502CFB0 = 0x100000000LL;
  dword_502CF6C &= 0x8000u;
  qword_502CFA8 = (__int64)&unk_502CFB8;
  word_502CF70 = 0;
  qword_502CF78 = 0;
  dword_502CF68 = v16;
  qword_502CF80 = 0;
  qword_502CF88 = 0;
  qword_502CF90 = 0;
  qword_502CF98 = 0;
  qword_502CFA0 = 0;
  qword_502CFC0 = 0;
  qword_502CFC8 = (__int64)&unk_502CFE0;
  qword_502CFD0 = 1;
  dword_502CFD8 = 0;
  byte_502CFDC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_502CFB0;
  if ( (unsigned __int64)(unsigned int)qword_502CFB0 + 1 > HIDWORD(qword_502CFB0) )
  {
    v26 = v17;
    sub_C8D5F0((char *)&unk_502CFB8 - 16, &unk_502CFB8, (unsigned int)qword_502CFB0 + 1LL, 8);
    v18 = (unsigned int)qword_502CFB0;
    v17 = v26;
  }
  *(_QWORD *)(qword_502CFA8 + 8 * v18) = v17;
  qword_502CFF0 = (__int64)&unk_49D9748;
  qword_502CF60 = (__int64)&unk_49DC090;
  qword_502D000 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502CFB0) = qword_502CFB0 + 1;
  qword_502D020 = (__int64)nullsub_23;
  qword_502CFE8 = 0;
  qword_502D018 = (__int64)sub_984030;
  qword_502CFF8 = 0;
  sub_C53080(&qword_502CF60, "ias-strong-global-assumptions", 29);
  LOWORD(qword_502CFF8) = 257;
  LOBYTE(qword_502CFE8) = 1;
  qword_502CF90 = 76;
  LOBYTE(dword_502CF6C) = dword_502CF6C & 0x9F | 0x20;
  qword_502CF88 = (__int64)"Make stronger assumptions that const buffer pointers always point to globals";
  sub_C53130(&qword_502CF60);
  __cxa_atexit(sub_984900, &qword_502CF60, &qword_4A427C0);
  qword_502CE80 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502CF60, v19, v20), 1u);
  qword_502CED0 = 0x100000000LL;
  dword_502CE8C &= 0x8000u;
  word_502CE90 = 0;
  qword_502CEC8 = (__int64)&unk_502CED8;
  qword_502CE98 = 0;
  dword_502CE88 = v21;
  qword_502CEA0 = 0;
  qword_502CEA8 = 0;
  qword_502CEB0 = 0;
  qword_502CEB8 = 0;
  qword_502CEC0 = 0;
  qword_502CEE0 = 0;
  qword_502CEE8 = (__int64)&unk_502CF00;
  qword_502CEF0 = 1;
  dword_502CEF8 = 0;
  byte_502CEFC = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_502CED0;
  v24 = (unsigned int)qword_502CED0 + 1LL;
  if ( v24 > HIDWORD(qword_502CED0) )
  {
    sub_C8D5F0((char *)&unk_502CED8 - 16, &unk_502CED8, v24, 8);
    v23 = (unsigned int)qword_502CED0;
  }
  *(_QWORD *)(qword_502CEC8 + 8 * v23) = v22;
  qword_502CF10 = (__int64)&unk_49D9748;
  qword_502CE80 = (__int64)&unk_49DC090;
  qword_502CF20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502CED0) = qword_502CED0 + 1;
  qword_502CF40 = (__int64)nullsub_23;
  qword_502CF08 = 0;
  qword_502CF38 = (__int64)sub_984030;
  qword_502CF18 = 0;
  sub_C53080(&qword_502CE80, "ias-param-always-point-to-global", 32);
  LOBYTE(qword_502CF08) = 1;
  LOWORD(qword_502CF18) = 257;
  qword_502CEB0 = 42;
  LOBYTE(dword_502CE8C) = dword_502CE8C & 0x9F | 0x20;
  qword_502CEA8 = (__int64)"Parameter pointers always point to globals";
  sub_C53130(&qword_502CE80);
  __cxa_atexit(sub_984900, &qword_502CE80, &qword_4A427C0);
  v29 = &v27;
  v30 = "Enable Memory Space Optimization for Wmma";
  v31 = 41;
  v28 = 1;
  v27 = 1;
  sub_24AB9D0(&unk_502CDA0, "ias-wmma-memory-space-opt", &v29, &v28, &v30);
  __cxa_atexit(sub_984900, &unk_502CDA0, &qword_4A427C0);
  v29 = &v27;
  v30 = "Enable 256-bit vector loads and stores for certain architectures";
  v31 = 64;
  v28 = 1;
  v27 = 1;
  sub_24AB9D0(&unk_502CCC0, "enable-256-bit-load-store", &v29, &v28, &v30);
  return __cxa_atexit(sub_984900, &unk_502CCC0, &qword_4A427C0);
}
