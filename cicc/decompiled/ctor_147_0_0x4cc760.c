// Function: ctor_147_0
// Address: 0x4cc760
//
int ctor_147_0()
{
  __m128i *v0; // rax
  __m128i v1; // xmm5
  const char *v2; // rsi
  _BYTE *v3; // r12
  _BYTE *v4; // r15
  __int64 v5; // rsi
  const char *v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // rbx
  int v13; // eax
  int v14; // eax
  int v15; // eax
  const char *v16; // rsi
  __int64 v17; // rbx
  unsigned int v18; // edx
  const char *v19; // r15
  __int64 v20; // r14
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // r8
  __m128i *v24; // r13
  __m128i *v25; // r13
  int v26; // eax
  int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rsi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // r10
  __int64 v35; // rax
  int v36; // r10d
  __int64 v37; // rdx
  _QWORD *v38; // r11
  __m128i *v39; // rax
  __int64 v40; // rsi
  const __m128i *v41; // rdx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // r10
  __int64 v45; // rax
  int v46; // r10d
  void *v47; // r11
  __m128i *v48; // rax
  __int64 v49; // rdi
  const __m128i *v50; // rsi
  __int8 v51; // dl
  __int64 v52; // [rsp+0h] [rbp-1F0h]
  __int64 v53; // [rsp+8h] [rbp-1E8h]
  __int64 v54; // [rsp+8h] [rbp-1E8h]
  __int64 v55; // [rsp+8h] [rbp-1E8h]
  unsigned int v56; // [rsp+10h] [rbp-1E0h]
  __int64 v57; // [rsp+10h] [rbp-1E0h]
  __int64 v58; // [rsp+10h] [rbp-1E0h]
  int v59; // [rsp+10h] [rbp-1E0h]
  int v60; // [rsp+10h] [rbp-1E0h]
  unsigned int v61; // [rsp+18h] [rbp-1D8h]
  unsigned int v62; // [rsp+18h] [rbp-1D8h]
  unsigned int v63; // [rsp+18h] [rbp-1D8h]
  __int64 v64; // [rsp+18h] [rbp-1D8h]
  int v65; // [rsp+20h] [rbp-1D0h]
  int v66; // [rsp+20h] [rbp-1D0h]
  __int64 v67; // [rsp+20h] [rbp-1D0h]
  __int64 v68; // [rsp+20h] [rbp-1D0h]
  __int64 v69; // [rsp+20h] [rbp-1D0h]
  unsigned int v70; // [rsp+28h] [rbp-1C8h]
  __m128i *v71; // [rsp+30h] [rbp-1C0h]
  int v72; // [rsp+30h] [rbp-1C0h]
  __int64 v73; // [rsp+30h] [rbp-1C0h]
  __int64 v74; // [rsp+30h] [rbp-1C0h]
  __int64 v75; // [rsp+30h] [rbp-1C0h]
  __int64 v76; // [rsp+38h] [rbp-1B8h]
  _BYTE *v77; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v78; // [rsp+48h] [rbp-1A8h]
  _BYTE v79[160]; // [rsp+50h] [rbp-1A0h] BYREF
  __m128i v80; // [rsp+F0h] [rbp-100h] BYREF
  __m128i v81; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v82; // [rsp+110h] [rbp-E0h] BYREF
  __m128i v83; // [rsp+120h] [rbp-D0h] BYREF
  __m128i v84; // [rsp+130h] [rbp-C0h] BYREF
  __m128i v85; // [rsp+140h] [rbp-B0h] BYREF
  __m128i v86; // [rsp+150h] [rbp-A0h] BYREF
  __m128i v87; // [rsp+160h] [rbp-90h] BYREF
  __m128i v88; // [rsp+170h] [rbp-80h] BYREF
  __m128i v89; // [rsp+180h] [rbp-70h] BYREF
  __m128i v90; // [rsp+190h] [rbp-60h] BYREF
  __m128i v91; // [rsp+1A0h] [rbp-50h] BYREF
  __int64 v92; // [rsp+1B0h] [rbp-40h]

  v80.m128i_i64[0] = (__int64)"Disabled";
  v81.m128i_i64[1] = (__int64)"disable debug output";
  v82.m128i_i64[1] = (__int64)"Arguments";
  v84.m128i_i64[0] = (__int64)"print pass arguments to pass to 'opt'";
  v85.m128i_i64[0] = (__int64)"Structure";
  v86.m128i_i64[1] = (__int64)"print pass structure before run()";
  v87.m128i_i64[1] = (__int64)"Executions";
  v89.m128i_i64[0] = (__int64)"print pass name before it is executed";
  v90.m128i_i64[0] = (__int64)"Details";
  v91.m128i_i64[1] = (__int64)"print pass details when it is executed";
  v80.m128i_i64[1] = 8;
  v81.m128i_i32[0] = 0;
  v82.m128i_i64[0] = 20;
  v83.m128i_i64[0] = 9;
  v83.m128i_i32[2] = 1;
  v84.m128i_i64[1] = 37;
  v85.m128i_i64[1] = 9;
  v86.m128i_i32[0] = 2;
  v87.m128i_i64[0] = 33;
  v88.m128i_i64[0] = 10;
  v88.m128i_i32[2] = 3;
  v89.m128i_i64[1] = 37;
  v90.m128i_i64[1] = 7;
  v91.m128i_i32[0] = 4;
  v92 = 38;
  v78 = 0x400000000LL;
  v77 = v79;
  sub_16CD150(&v77, v79, 5, 40);
  v0 = (__m128i *)&v77[40 * (unsigned int)v78];
  *v0 = _mm_loadu_si128(&v80);
  v1 = _mm_loadu_si128(&v81);
  LODWORD(v78) = v78 + 5;
  v0[1] = v1;
  v0[2] = _mm_loadu_si128(&v82);
  v0[3] = _mm_loadu_si128(&v83);
  v0[4] = _mm_loadu_si128(&v84);
  v0[5] = _mm_loadu_si128(&v85);
  v0[6] = _mm_loadu_si128(&v86);
  v0[7] = _mm_loadu_si128(&v87);
  v0[8] = _mm_loadu_si128(&v88);
  v0[9] = _mm_loadu_si128(&v89);
  v0[10] = _mm_loadu_si128(&v90);
  v0[11] = _mm_loadu_si128(&v91);
  v0[12].m128i_i64[0] = v92;
  qword_4F9EAA0 = (__int64)&unk_49EED30;
  LODWORD(v0) = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  v2 = "debug-pass";
  word_4F9EAAC &= 0xF000u;
  qword_4F9EAB0 = 0;
  qword_4F9EAB8 = 0;
  qword_4F9EAC0 = 0;
  qword_4F9EAC8 = 0;
  dword_4F9EAA8 = (int)v0;
  qword_4F9EAD0 = 0;
  qword_4F9EAE8 = (__int64)&unk_4FA01C0;
  qword_4F9EAF8 = (__int64)&unk_4F9EB18;
  qword_4F9EB00 = (__int64)&unk_4F9EB18;
  qword_4F9EAA0 = (__int64)off_49ED5D0;
  qword_4F9EB58 = (__int64)&off_49ED580;
  qword_4F9EB68 = (__int64)&unk_4F9EB78;
  qword_4F9EB70 = 0x800000000LL;
  qword_4F9EAD8 = 0;
  qword_4F9EAE0 = 0;
  qword_4F9EAF0 = 0;
  qword_4F9EB08 = 4;
  dword_4F9EB10 = 0;
  byte_4F9EB38 = 0;
  dword_4F9EB40 = 0;
  qword_4F9EB48 = (__int64)&off_49ED560;
  byte_4F9EB54 = 1;
  dword_4F9EB50 = 0;
  qword_4F9EB60 = (__int64)&qword_4F9EAA0;
  sub_16B8280(&qword_4F9EAA0, "debug-pass", 10);
  v3 = v77;
  qword_4F9EAD0 = 39;
  LOBYTE(word_4F9EAAC) = word_4F9EAAC & 0x9F | 0x20;
  qword_4F9EAC8 = (__int64)"Print PassManager debugging information";
  v4 = &v77[40 * (unsigned int)v78];
  if ( v77 != v4 )
  {
    do
    {
      LODWORD(v5) = qword_4F9EB70;
      v6 = *(const char **)v3;
      v7 = *((_QWORD *)v3 + 1);
      v8 = *((unsigned int *)v3 + 4);
      v9 = *((_QWORD *)v3 + 3);
      v10 = *((_QWORD *)v3 + 4);
      if ( (unsigned int)qword_4F9EB70 >= HIDWORD(qword_4F9EB70) )
      {
        v55 = *((_QWORD *)v3 + 4);
        v58 = *((_QWORD *)v3 + 3);
        v63 = *((_DWORD *)v3 + 4);
        v42 = (((unsigned __int64)HIDWORD(qword_4F9EB70) + 2) >> 1) | (HIDWORD(qword_4F9EB70) + 2LL);
        v67 = *((_QWORD *)v3 + 1);
        v43 = (((v42 >> 2) | v42) >> 4) | (v42 >> 2) | v42;
        v44 = ((v43 >> 8) | v43 | (((v43 >> 8) | v43) >> 16) | (((v43 >> 8) | v43) >> 32)) + 1;
        if ( v44 > 0xFFFFFFFF )
          v44 = 0xFFFFFFFFLL;
        v72 = v44;
        v45 = malloc(48 * v44, (unsigned int)qword_4F9EB70, v7, v8, v10, v9);
        v46 = v72;
        v5 = (unsigned int)v5;
        v8 = v63;
        v7 = v67;
        v11 = v45;
        v9 = v58;
        v10 = v55;
        if ( !v45 )
        {
          v60 = v72;
          v69 = v9;
          v75 = v7;
          sub_16BD1C0("Allocation failed");
          v5 = (unsigned int)qword_4F9EB70;
          v46 = v60;
          v10 = v55;
          v9 = v69;
          v8 = v63;
          v7 = v75;
        }
        v47 = (void *)qword_4F9EB68;
        v48 = (__m128i *)v11;
        v49 = qword_4F9EB68 + 48 * v5;
        v50 = (const __m128i *)qword_4F9EB68;
        if ( qword_4F9EB68 != v49 )
        {
          v73 = v7;
          do
          {
            if ( v48 )
            {
              *v48 = _mm_loadu_si128(v50);
              v48[1] = _mm_loadu_si128(v50 + 1);
              v48[2].m128i_i32[2] = v50[2].m128i_i32[2];
              v51 = v50[2].m128i_i8[12];
              v48[2].m128i_i64[0] = (__int64)&off_49ED560;
              v48[2].m128i_i8[12] = v51;
            }
            v50 += 3;
            v48 += 3;
          }
          while ( (const __m128i *)v49 != v50 );
          v7 = v73;
        }
        if ( v47 != &unk_4F9EB78 )
        {
          v59 = v46;
          v64 = v10;
          v68 = v9;
          v70 = v8;
          v74 = v7;
          _libc_free(v47, v50);
          v46 = v59;
          v10 = v64;
          v9 = v68;
          v8 = v70;
          v7 = v74;
        }
        qword_4F9EB68 = v11;
        LODWORD(v5) = qword_4F9EB70;
        HIDWORD(qword_4F9EB70) = v46;
      }
      else
      {
        v11 = qword_4F9EB68;
      }
      v12 = 48LL * (unsigned int)v5 + v11;
      if ( v12 )
      {
        *(_QWORD *)v12 = v6;
        *(_QWORD *)(v12 + 8) = v7;
        *(_QWORD *)(v12 + 16) = v9;
        *(_QWORD *)(v12 + 24) = v10;
        *(_DWORD *)(v12 + 40) = v8;
        *(_BYTE *)(v12 + 44) = 1;
        *(_QWORD *)(v12 + 32) = &off_49ED560;
        LODWORD(v5) = qword_4F9EB70;
      }
      v3 += 40;
      LODWORD(qword_4F9EB70) = v5 + 1;
      v2 = v6;
      sub_16B7FD0(qword_4F9EB60, v6, v7, v8);
    }
    while ( v4 != v3 );
  }
  sub_16B88A0(&qword_4F9EAA0);
  if ( v77 != v79 )
    _libc_free(v77, v2);
  __cxa_atexit(sub_160D3D0, &qword_4F9EAA0, &qword_4A427C0);
  qword_4F9E9C0 = (__int64)&unk_49EED30;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E9CC &= 0xF000u;
  qword_4F9E9D0 = 0;
  qword_4F9E9D8 = 0;
  qword_4F9E9E0 = 0;
  qword_4F9E9E8 = 0;
  dword_4F9E9C8 = v13;
  qword_4F9E9F0 = 0;
  qword_4F9EA08 = (__int64)&unk_4FA01C0;
  qword_4F9EA18 = (__int64)&unk_4F9EA38;
  qword_4F9EA20 = (__int64)&unk_4F9EA38;
  qword_4F9E9F8 = 0;
  qword_4F9EA00 = 0;
  qword_4F9EA68 = (__int64)&unk_49E74C8;
  qword_4F9EA10 = 0;
  byte_4F9EA58 = 0;
  qword_4F9E9C0 = (__int64)&unk_49EEB70;
  qword_4F9EA28 = 4;
  byte_4F9EA74 = 1;
  qword_4F9EA78 = (__int64)&unk_49EEDF0;
  dword_4F9EA30 = 0;
  dword_4F9EA60 = 0;
  dword_4F9EA70 = 0;
  sub_16B8280(&qword_4F9E9C0, "pass-control", 12);
  qword_4F9E9F0 = 55;
  dword_4F9EA60 = -1;
  byte_4F9EA74 = 1;
  dword_4F9EA70 = -1;
  LOBYTE(word_4F9E9CC) = word_4F9E9CC & 0x9F | 0x20;
  qword_4F9E9E8 = (__int64)"Disable all optional passes after specified pass number";
  sub_16B88A0(&qword_4F9E9C0);
  __cxa_atexit(sub_12EDEA0, &qword_4F9E9C0, &qword_4A427C0);
  qword_4F9E8E0 = (__int64)&unk_49EED30;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4F9E8F0 = 0;
  qword_4F9E8F8 = 0;
  qword_4F9E900 = 0;
  qword_4F9E908 = 0;
  qword_4F9E910 = 0;
  dword_4F9E8E8 = v14;
  qword_4F9E9B0 = (__int64)&unk_49EEDF0;
  qword_4F9E918 = 0;
  qword_4F9E920 = 0;
  word_4F9E8EC = word_4F9E8EC & 0xF000 | 1;
  qword_4F9E930 = 0;
  qword_4F9E928 = (__int64)&unk_4FA01C0;
  qword_4F9E938 = (__int64)&unk_4F9E958;
  qword_4F9E940 = (__int64)&unk_4F9E958;
  qword_4F9E948 = 4;
  dword_4F9E950 = 0;
  qword_4F9E8E0 = (__int64)&unk_49ED650;
  byte_4F9E978 = 0;
  qword_4F9E980 = 0;
  qword_4F9E988 = 0;
  qword_4F9E990 = 0;
  qword_4F9E998 = 0;
  qword_4F9E9A0 = 0;
  qword_4F9E9A8 = 0;
  sub_16B8280(&qword_4F9E8E0, "disable-passno", 14);
  HIBYTE(word_4F9E8EC) |= 2u;
  qword_4F9E920 = 4;
  qword_4F9E910 = 88;
  LOBYTE(word_4F9E8EC) = word_4F9E8EC & 0x9F | 0x20;
  qword_4F9E918 = (__int64)"list";
  qword_4F9E908 = (__int64)"Disable any optional pass(es) by specifying thepass number(s) in a comma separated list.";
  sub_16B88A0(&qword_4F9E8E0);
  __cxa_atexit(sub_160D5A0, &qword_4F9E8E0, &qword_4A427C0);
  qword_4F9E800 = (__int64)&unk_49EED30;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E8B0 = 256;
  word_4F9E80C &= 0xF000u;
  qword_4F9E810 = 0;
  qword_4F9E818 = 0;
  qword_4F9E820 = 0;
  dword_4F9E808 = v15;
  qword_4F9E828 = 0;
  qword_4F9E848 = (__int64)&unk_4FA01C0;
  qword_4F9E858 = (__int64)&unk_4F9E878;
  qword_4F9E860 = (__int64)&unk_4F9E878;
  qword_4F9E830 = 0;
  qword_4F9E838 = 0;
  qword_4F9E8A8 = (__int64)&unk_49E74E8;
  qword_4F9E840 = 0;
  qword_4F9E850 = 0;
  qword_4F9E800 = (__int64)&unk_49EEC70;
  qword_4F9E868 = 4;
  byte_4F9E898 = 0;
  qword_4F9E8B8 = (__int64)&unk_49EEDB0;
  dword_4F9E870 = 0;
  byte_4F9E8A0 = 0;
  sub_16B8280(&qword_4F9E800, "verify-after-all", 16);
  word_4F9E8B0 = 256;
  qword_4F9E828 = (__int64)"Run the IR verification pass after each non-Analysis pass.";
  qword_4F9E830 = 58;
  byte_4F9E8A0 = 0;
  sub_16B88A0(&qword_4F9E800);
  __cxa_atexit(sub_12EDEC0, &qword_4F9E800, &qword_4A427C0);
  v81.m128i_i64[1] = 4;
  v80.m128i_i64[0] = (__int64)&v81;
  v81.m128i_i64[0] = (__int64)"regs";
  v82.m128i_i64[1] = (__int64)"print register pressure";
  v83.m128i_i64[1] = (__int64)"fnsize";
  v85.m128i_i64[0] = (__int64)"print function IR size";
  v86.m128i_i64[0] = (__int64)"modsize";
  v87.m128i_i64[1] = (__int64)"print module IR size";
  v88.m128i_i64[1] = (__int64)byte_3F871B3;
  v90.m128i_i64[0] = (__int64)"(default) print everything";
  v80.m128i_i64[1] = 0x400000004LL;
  v82.m128i_i32[0] = 1;
  v83.m128i_i64[0] = 23;
  v84.m128i_i64[0] = 6;
  v84.m128i_i32[2] = 2;
  v85.m128i_i64[1] = 22;
  v86.m128i_i64[1] = 7;
  v87.m128i_i32[0] = 4;
  v88.m128i_i64[0] = 20;
  v89.m128i_i64[0] = 0;
  v89.m128i_i32[2] = 255;
  v90.m128i_i64[1] = 26;
  qword_4F9E580[0] = &unk_49EED30;
  v16 = "extra-print-after-all";
  LODWORD(qword_4F9E580[1]) = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4F9E580[2] = 0;
  qword_4F9E580[3] = 0;
  qword_4F9E580[4] = 0;
  WORD2(qword_4F9E580[1]) = WORD2(qword_4F9E580[1]) & 0xF000 | 1;
  qword_4F9E580[5] = 0;
  qword_4F9E580[9] = &unk_4FA01C0;
  qword_4F9E580[11] = &qword_4F9E580[15];
  qword_4F9E580[12] = &qword_4F9E580[15];
  qword_4F9E580[6] = 0;
  qword_4F9E580[7] = 0;
  qword_4F9E580[0] = &unk_49ED740;
  qword_4F9E580[8] = 0;
  qword_4F9E580[10] = 0;
  qword_4F9E580[26] = &unk_49ED6F0;
  qword_4F9E580[28] = &qword_4F9E580[30];
  qword_4F9E580[29] = 0x800000000LL;
  qword_4F9E580[13] = 4;
  LODWORD(qword_4F9E580[14]) = 0;
  LOBYTE(qword_4F9E580[19]) = 0;
  qword_4F9E580[20] = 0;
  qword_4F9E580[21] = 0;
  qword_4F9E580[22] = 0;
  qword_4F9E580[23] = 0;
  qword_4F9E580[24] = 0;
  qword_4F9E580[25] = 0;
  qword_4F9E580[27] = qword_4F9E580;
  sub_16B8280(qword_4F9E580, "extra-print-after-all", 21);
  BYTE5(qword_4F9E580[1]) |= 2u;
  qword_4F9E580[5] = "Print extra information after each pass";
  qword_4F9E580[6] = 39;
  v17 = v80.m128i_i64[0];
  v71 = (__m128i *)v80.m128i_i64[0];
  BYTE4(qword_4F9E580[1]) = BYTE4(qword_4F9E580[1]) & 0x87 | 0x28;
  v76 = v80.m128i_i64[0] + 40LL * v80.m128i_u32[2];
  if ( v80.m128i_i64[0] != v76 )
  {
    do
    {
      v18 = qword_4F9E580[29];
      v19 = *(const char **)v17;
      v20 = *(_QWORD *)(v17 + 8);
      v21 = *(unsigned int *)(v17 + 16);
      v22 = *(_QWORD *)(v17 + 24);
      v23 = *(_QWORD *)(v17 + 32);
      if ( LODWORD(qword_4F9E580[29]) >= HIDWORD(qword_4F9E580[29]) )
      {
        v52 = *(_QWORD *)(v17 + 32);
        v53 = *(_QWORD *)(v17 + 24);
        v56 = *(_DWORD *)(v17 + 16);
        v31 = (((unsigned __int64)HIDWORD(qword_4F9E580[29]) + 2) >> 1) | (HIDWORD(qword_4F9E580[29]) + 2LL);
        v61 = qword_4F9E580[29];
        v32 = v31 >> 2;
        v33 = (((v31 >> 2) | v31) >> 4) | (v31 >> 2) | v31;
        v34 = ((v33 >> 8) | v33 | (((v33 >> 8) | v33) >> 16) | (((v33 >> 8) | v33) >> 32)) + 1;
        if ( v34 > 0xFFFFFFFF )
          v34 = 0xFFFFFFFFLL;
        v65 = v34;
        v35 = malloc(48 * v34, v32, LODWORD(qword_4F9E580[29]), v21, v23, v22);
        v36 = v65;
        v37 = v61;
        v21 = v56;
        v24 = (__m128i *)v35;
        v22 = v53;
        v23 = v52;
        if ( !v35 )
        {
          sub_16BD1C0("Allocation failed");
          v37 = LODWORD(qword_4F9E580[29]);
          v23 = v52;
          v22 = v53;
          v21 = v56;
          v36 = v65;
        }
        v38 = (_QWORD *)qword_4F9E580[28];
        v39 = v24;
        v40 = qword_4F9E580[28] + 48 * v37;
        v41 = (const __m128i *)qword_4F9E580[28];
        if ( qword_4F9E580[28] != v40 )
        {
          do
          {
            if ( v39 )
            {
              *v39 = _mm_loadu_si128(v41);
              v39[1] = _mm_loadu_si128(v41 + 1);
              v39[2].m128i_i8[8] = v41[2].m128i_i8[8];
              v39[2].m128i_i8[9] = v41[2].m128i_i8[9];
              v39[2].m128i_i64[0] = (__int64)&unk_49ED6D0;
            }
            v41 += 3;
            v39 += 3;
          }
          while ( (const __m128i *)v40 != v41 );
        }
        if ( v38 != &qword_4F9E580[30] )
        {
          v54 = v23;
          v57 = v22;
          v62 = v21;
          v66 = v36;
          _libc_free(v38, v40);
          v23 = v54;
          v22 = v57;
          v21 = v62;
          v36 = v66;
        }
        qword_4F9E580[28] = v24;
        v18 = qword_4F9E580[29];
        HIDWORD(qword_4F9E580[29]) = v36;
      }
      else
      {
        v24 = (__m128i *)qword_4F9E580[28];
      }
      v25 = &v24[3 * v18];
      if ( v25 )
      {
        v25->m128i_i64[0] = (__int64)v19;
        v25->m128i_i64[1] = v20;
        v25[1].m128i_i64[0] = v22;
        v25[1].m128i_i64[1] = v23;
        v25[2].m128i_i8[8] = v21;
        v25[2].m128i_i8[9] = 1;
        v25[2].m128i_i64[0] = (__int64)&unk_49ED6D0;
        v18 = qword_4F9E580[29];
      }
      v16 = v19;
      v17 += 40;
      LODWORD(qword_4F9E580[29]) = v18 + 1;
      sub_16B7FD0(qword_4F9E580[27], v19, v20, v21);
    }
    while ( v76 != v17 );
  }
  sub_16B88A0(qword_4F9E580);
  if ( v71 != &v81 )
    _libc_free(v71, v16);
  __cxa_atexit(sub_160CDC0, qword_4F9E580, &qword_4A427C0);
  qword_4F9E4A0 = (__int64)&unk_49EED30;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E4AC &= 0xF000u;
  qword_4F9E4B0 = 0;
  qword_4F9E4B8 = 0;
  qword_4F9E4C0 = 0;
  qword_4F9E4C8 = 0;
  dword_4F9E4A8 = v26;
  qword_4F9E4D0 = 0;
  qword_4F9E4E8 = (__int64)&unk_4FA01C0;
  qword_4F9E4F8 = (__int64)&unk_4F9E518;
  qword_4F9E500 = (__int64)&unk_4F9E518;
  qword_4F9E4D8 = 0;
  qword_4F9E4E0 = 0;
  word_4F9E550 = 256;
  qword_4F9E548 = (__int64)&unk_49E74E8;
  qword_4F9E4F0 = 0;
  qword_4F9E4A0 = (__int64)&unk_49EEC70;
  byte_4F9E538 = 0;
  qword_4F9E558 = (__int64)&unk_49EEDB0;
  qword_4F9E508 = 4;
  dword_4F9E510 = 0;
  byte_4F9E540 = 0;
  sub_16B8280(&qword_4F9E4A0, "print-module-scope", 18);
  qword_4F9E4C8 = (__int64)"When printing IR for print-[before|after]{-all} always print a module IR";
  word_4F9E550 = 256;
  byte_4F9E540 = 0;
  qword_4F9E4D0 = 72;
  LOBYTE(word_4F9E4AC) = word_4F9E4AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9E4A0);
  __cxa_atexit(sub_12EDEC0, &qword_4F9E4A0, &qword_4A427C0);
  qword_4F9E3C0 = (__int64)&unk_49EED30;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E470 = 256;
  word_4F9E3CC &= 0xF000u;
  qword_4F9E3D0 = 0;
  qword_4F9E3D8 = 0;
  qword_4F9E3E0 = 0;
  dword_4F9E3C8 = v27;
  qword_4F9E468 = (__int64)&unk_49E74E8;
  qword_4F9E408 = (__int64)&unk_4FA01C0;
  qword_4F9E418 = (__int64)&unk_4F9E438;
  qword_4F9E420 = (__int64)&unk_4F9E438;
  qword_4F9E3C0 = (__int64)&unk_49EEC70;
  qword_4F9E478 = (__int64)&unk_49EEDB0;
  qword_4F9E3E8 = 0;
  qword_4F9E3F0 = 0;
  qword_4F9E3F8 = 0;
  qword_4F9E400 = 0;
  qword_4F9E410 = 0;
  qword_4F9E428 = 4;
  dword_4F9E430 = 0;
  byte_4F9E458 = 0;
  byte_4F9E460 = 0;
  sub_16B8280((char *)&unk_4F9E438 - 120, "print-loop-func-scope", 21);
  qword_4F9E3E8 = (__int64)"When printing IR for print-[before|after]{-all} for a loop pass, always print function IR";
  word_4F9E470 = 256;
  byte_4F9E460 = 0;
  qword_4F9E3F0 = 89;
  LOBYTE(word_4F9E3CC) = word_4F9E3CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9E3C0);
  __cxa_atexit(sub_12EDEC0, &qword_4F9E3C0, &qword_4A427C0);
  qword_4F9E2C0 = (__int64)&unk_49EED30;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E2CC &= 0xF000u;
  qword_4F9E2D0 = 0;
  qword_4F9E2D8 = 0;
  qword_4F9E2E0 = 0;
  qword_4F9E2E8 = 0;
  dword_4F9E2C8 = v28;
  qword_4F9E2F0 = 0;
  qword_4F9E308 = (__int64)&unk_4FA01C0;
  qword_4F9E318 = (__int64)&unk_4F9E338;
  qword_4F9E320 = (__int64)&unk_4F9E338;
  qword_4F9E2F8 = 0;
  qword_4F9E300 = 0;
  qword_4F9E310 = 0;
  qword_4F9E328 = 4;
  dword_4F9E330 = 0;
  byte_4F9E358 = 0;
  qword_4F9E360 = 0;
  byte_4F9E371 = 0;
  qword_4F9E368 = (__int64)&unk_49E74E8;
  qword_4F9E2C0 = (__int64)&unk_49EAB58;
  qword_4F9E378 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4F9E2C0, "time-passes", 11);
  if ( qword_4F9E360 )
  {
    v29 = sub_16E8CB0();
    v80.m128i_i64[0] = (__int64)"cl::location(x) specified more than once!";
    v81.m128i_i16[0] = 259;
    sub_16B1F90(&qword_4F9E2C0, &v80, 0, 0, v29);
  }
  else
  {
    byte_4F9E371 = 1;
    qword_4F9E360 = (__int64)&unk_4F9E388;
    byte_4F9E370 = unk_4F9E388;
  }
  qword_4F9E2F0 = 54;
  LOBYTE(word_4F9E2CC) = word_4F9E2CC & 0x9F | 0x20;
  qword_4F9E2E8 = (__int64)"Time each pass, printing elapsed time for each on exit";
  sub_16B88A0(&qword_4F9E2C0);
  return __cxa_atexit(sub_13F9A70, &qword_4F9E2C0, &qword_4A427C0);
}
