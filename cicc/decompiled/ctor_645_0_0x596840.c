// Function: ctor_645_0
// Address: 0x596840
//
int ctor_645_0()
{
  __m128i *v0; // rax
  __m128i v1; // xmm1
  __m128i v2; // xmm2
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v21; // [rsp+8h] [rbp-1D8h]
  int v22; // [rsp+1Ch] [rbp-1C4h] BYREF
  _QWORD v23[2]; // [rsp+20h] [rbp-1C0h] BYREF
  _BYTE *v24; // [rsp+30h] [rbp-1B0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-1A8h]
  _BYTE v26[160]; // [rsp+40h] [rbp-1A0h] BYREF
  __m128i v27; // [rsp+E0h] [rbp-100h] BYREF
  __m128i v28; // [rsp+F0h] [rbp-F0h] BYREF
  __m128i v29; // [rsp+100h] [rbp-E0h] BYREF
  __m128i v30; // [rsp+110h] [rbp-D0h] BYREF
  __m128i v31; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v32; // [rsp+130h] [rbp-B0h] BYREF
  __m128i v33; // [rsp+140h] [rbp-A0h] BYREF
  __m128i v34; // [rsp+150h] [rbp-90h] BYREF
  __m128i v35; // [rsp+160h] [rbp-80h] BYREF
  __m128i v36; // [rsp+170h] [rbp-70h] BYREF
  __m128i v37; // [rsp+180h] [rbp-60h] BYREF
  __m128i v38; // [rsp+190h] [rbp-50h] BYREF
  __int64 v39; // [rsp+1A0h] [rbp-40h]

  v23[0] = "Enable extended information within the SHT_LLVM_BB_ADDR_MAP that is extracted from PGO related analysis.";
  v27.m128i_i64[0] = (__int64)"none";
  v28.m128i_i64[1] = (__int64)"Disable all options";
  v29.m128i_i64[1] = (__int64)"func-entry-count";
  v31.m128i_i64[0] = (__int64)"Function Entry Count";
  v32.m128i_i64[0] = (__int64)"bb-freq";
  v33.m128i_i64[1] = (__int64)"Basic Block Frequency";
  v34.m128i_i64[1] = (__int64)"br-prob";
  v36.m128i_i64[0] = (__int64)"Branch Probability";
  v37.m128i_i64[0] = (__int64)"all";
  v23[1] = 104;
  v27.m128i_i64[1] = 4;
  v28.m128i_i32[0] = 0;
  v29.m128i_i64[0] = 19;
  v30.m128i_i64[0] = 16;
  v30.m128i_i32[2] = 1;
  v31.m128i_i64[1] = 20;
  v32.m128i_i64[1] = 7;
  v33.m128i_i32[0] = 2;
  v34.m128i_i64[0] = 21;
  v35.m128i_i64[0] = 7;
  v35.m128i_i32[2] = 3;
  v36.m128i_i64[1] = 18;
  v37.m128i_i64[1] = 3;
  v38.m128i_i32[0] = 4;
  v38.m128i_i64[1] = (__int64)"Enable all options";
  v25 = 0x400000000LL;
  v39 = 18;
  v24 = v26;
  sub_C8D5F0(&v24, v26, 5, 40);
  v0 = (__m128i *)&v24[40 * (unsigned int)v25];
  *v0 = _mm_loadu_si128(&v27);
  v1 = _mm_loadu_si128(&v28);
  LODWORD(v25) = v25 + 5;
  v0[1] = v1;
  v2 = _mm_loadu_si128(&v29);
  v22 = 1;
  v0[2] = v2;
  v0[3] = _mm_loadu_si128(&v30);
  v0[4] = _mm_loadu_si128(&v31);
  v0[5] = _mm_loadu_si128(&v32);
  v0[6] = _mm_loadu_si128(&v33);
  v0[7] = _mm_loadu_si128(&v34);
  v0[8] = _mm_loadu_si128(&v35);
  v0[9] = _mm_loadu_si128(&v36);
  v0[10] = _mm_loadu_si128(&v37);
  v0[11] = _mm_loadu_si128(&v38);
  v0[12].m128i_i64[0] = v39;
  v27.m128i_i32[0] = 1;
  sub_31F0350(&unk_5036020, "pgo-analysis-map", &v22, &v27, &v24, v23);
  if ( v24 != v26 )
    _libc_free(v24, "pgo-analysis-map");
  __cxa_atexit(sub_31D5F70, &unk_5036020, &qword_4A427C0);
  qword_5035F40 = (__int64)&unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_31D5F70, &unk_5036020, v3, v4), 1u);
  qword_5035F90 = 0x100000000LL;
  dword_5035F4C &= 0x8000u;
  word_5035F50 = 0;
  qword_5035F58 = 0;
  qword_5035F60 = 0;
  dword_5035F48 = v5;
  qword_5035F68 = 0;
  qword_5035F70 = 0;
  qword_5035F78 = 0;
  qword_5035F80 = 0;
  qword_5035F88 = (__int64)&unk_5035F98;
  qword_5035FA0 = 0;
  qword_5035FA8 = (__int64)&unk_5035FC0;
  qword_5035FB0 = 1;
  dword_5035FB8 = 0;
  byte_5035FBC = 1;
  v6 = sub_C57470();
  v7 = (unsigned int)qword_5035F90;
  v8 = (unsigned int)qword_5035F90 + 1LL;
  if ( v8 > HIDWORD(qword_5035F90) )
  {
    sub_C8D5F0((char *)&unk_5035F98 - 16, &unk_5035F98, v8, 8);
    v7 = (unsigned int)qword_5035F90;
  }
  *(_QWORD *)(qword_5035F88 + 8 * v7) = v6;
  qword_5035FD0 = (__int64)&unk_49D9748;
  qword_5035F40 = (__int64)&unk_49DC090;
  qword_5035FE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5035F90) = qword_5035F90 + 1;
  qword_5036000 = (__int64)nullsub_23;
  qword_5035FC8 = 0;
  qword_5035FF8 = (__int64)sub_984030;
  qword_5035FD8 = 0;
  sub_C53080(&qword_5035F40, "basic-block-address-map-skip-bb-entries", 39);
  qword_5035F68 = (__int64)"Skip emitting basic block entries in the SHT_LLVM_BB_ADDR_MAP section. It's used to save bina"
                           "ry size when BB entries are unnecessary for some PGOAnalysisMap features.";
  LOWORD(qword_5035FD8) = 256;
  LOBYTE(qword_5035FC8) = 0;
  qword_5035F70 = 166;
  LOBYTE(dword_5035F4C) = dword_5035F4C & 0x9F | 0x20;
  sub_C53130(&qword_5035F40);
  __cxa_atexit(sub_984900, &qword_5035F40, &qword_4A427C0);
  qword_5035E60 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5035F40, v9, v10), 1u);
  qword_5035EB0 = 0x100000000LL;
  dword_5035E6C &= 0x8000u;
  qword_5035EA8 = (__int64)&unk_5035EB8;
  word_5035E70 = 0;
  qword_5035E78 = 0;
  dword_5035E68 = v11;
  qword_5035E80 = 0;
  qword_5035E88 = 0;
  qword_5035E90 = 0;
  qword_5035E98 = 0;
  qword_5035EA0 = 0;
  qword_5035EC0 = 0;
  qword_5035EC8 = (__int64)&unk_5035EE0;
  qword_5035ED0 = 1;
  dword_5035ED8 = 0;
  byte_5035EDC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_5035EB0;
  if ( (unsigned __int64)(unsigned int)qword_5035EB0 + 1 > HIDWORD(qword_5035EB0) )
  {
    v21 = v12;
    sub_C8D5F0((char *)&unk_5035EB8 - 16, &unk_5035EB8, (unsigned int)qword_5035EB0 + 1LL, 8);
    v13 = (unsigned int)qword_5035EB0;
    v12 = v21;
  }
  *(_QWORD *)(qword_5035EA8 + 8 * v13) = v12;
  qword_5035EF0 = (__int64)&unk_49D9748;
  qword_5035E60 = (__int64)&unk_49DC090;
  qword_5035F00 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5035EB0) = qword_5035EB0 + 1;
  qword_5035F20 = (__int64)nullsub_23;
  qword_5035EE8 = 0;
  qword_5035F18 = (__int64)sub_984030;
  qword_5035EF8 = 0;
  sub_C53080(&qword_5035E60, "emit-jump-table-sizes-section", 29);
  qword_5035E88 = (__int64)"Emit a section containing jump table addresses and sizes";
  LOWORD(qword_5035EF8) = 256;
  LOBYTE(qword_5035EE8) = 0;
  qword_5035E90 = 56;
  LOBYTE(dword_5035E6C) = dword_5035E6C & 0x9F | 0x20;
  sub_C53130(&qword_5035E60);
  __cxa_atexit(sub_984900, &qword_5035E60, &qword_4A427C0);
  qword_5035D80 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5035E60, v14, v15), 1u);
  qword_5035DD0 = 0x100000000LL;
  dword_5035D8C &= 0x8000u;
  word_5035D90 = 0;
  qword_5035DC8 = (__int64)&unk_5035DD8;
  qword_5035D98 = 0;
  dword_5035D88 = v16;
  qword_5035DA0 = 0;
  qword_5035DA8 = 0;
  qword_5035DB0 = 0;
  qword_5035DB8 = 0;
  qword_5035DC0 = 0;
  qword_5035DE0 = 0;
  qword_5035DE8 = (__int64)&unk_5035E00;
  qword_5035DF0 = 1;
  dword_5035DF8 = 0;
  byte_5035DFC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_5035DD0;
  v19 = (unsigned int)qword_5035DD0 + 1LL;
  if ( v19 > HIDWORD(qword_5035DD0) )
  {
    sub_C8D5F0((char *)&unk_5035DD8 - 16, &unk_5035DD8, v19, 8);
    v18 = (unsigned int)qword_5035DD0;
  }
  *(_QWORD *)(qword_5035DC8 + 8 * v18) = v17;
  qword_5035E10 = (__int64)&unk_49D9748;
  qword_5035D80 = (__int64)&unk_49DC090;
  qword_5035E20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5035DD0) = qword_5035DD0 + 1;
  qword_5035E40 = (__int64)nullsub_23;
  qword_5035E08 = 0;
  qword_5035E38 = (__int64)sub_984030;
  qword_5035E18 = 0;
  sub_C53080(&qword_5035D80, "asm-print-latency", 17);
  qword_5035DB0 = 51;
  qword_5035DA8 = (__int64)"Print instruction latencies as verbose asm comments";
  LOBYTE(qword_5035E08) = 0;
  LOBYTE(dword_5035D8C) = dword_5035D8C & 0x9F | 0x20;
  LOWORD(qword_5035E18) = 256;
  sub_C53130(&qword_5035D80);
  return __cxa_atexit(sub_984900, &qword_5035D80, &qword_4A427C0);
}
