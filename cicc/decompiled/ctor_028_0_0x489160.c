// Function: ctor_028_0
// Address: 0x489160
//
int ctor_028_0()
{
  __m128i v0; // xmm5
  __m128i v1; // xmm6
  __m128i v2; // xmm7
  __m128i *v3; // rax
  __int64 v4; // rdx
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __m128i v8; // xmm7
  __m128i v9; // xmm4
  __m128i v10; // xmm5
  __m128i v11; // xmm6
  __m128i v12; // xmm7
  int v13; // edx
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  const char *v17; // rsi
  _BYTE *v18; // r14
  _BYTE *v19; // r13
  __int32 v20; // eax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // r10
  __int64 v25; // rax
  __m128i *v26; // rcx
  int v27; // edx
  __int64 v28; // rax
  __m128i v29; // xmm1
  __int8 v30; // dl
  __int64 v31; // rdx
  int v32; // edx
  __int64 v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  int v36; // edx
  __int64 v37; // r12
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __m128i *v41; // rax
  void *v42; // rdi
  __int64 v43; // rsi
  const __m128i *v44; // rdx
  __int64 v45; // rcx
  __int8 v46; // di
  int v47; // eax
  int v48; // [rsp+8h] [rbp-1E8h]
  unsigned __int64 v49; // [rsp+10h] [rbp-1E0h]
  char v50; // [rsp+1Fh] [rbp-1D1h]
  __int64 v51; // [rsp+20h] [rbp-1D0h]
  __int64 v52; // [rsp+38h] [rbp-1B8h] BYREF
  _BYTE *v53; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v54; // [rsp+48h] [rbp-1A8h]
  _BYTE v55[160]; // [rsp+50h] [rbp-1A0h] BYREF
  __m128i v56; // [rsp+F0h] [rbp-100h] BYREF
  __m128i v57; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v58; // [rsp+110h] [rbp-E0h] BYREF
  __m128i v59; // [rsp+120h] [rbp-D0h] BYREF
  __m128i v60; // [rsp+130h] [rbp-C0h] BYREF
  __m128i v61; // [rsp+140h] [rbp-B0h] BYREF
  __m128i v62; // [rsp+150h] [rbp-A0h] BYREF
  __m128i v63; // [rsp+160h] [rbp-90h] BYREF
  __m128i v64; // [rsp+170h] [rbp-80h] BYREF
  __m128i v65; // [rsp+180h] [rbp-70h] BYREF
  __m128i v66; // [rsp+190h] [rbp-60h] BYREF
  __m128i v67; // [rsp+1A0h] [rbp-50h] BYREF
  __int64 v68; // [rsp+1B0h] [rbp-40h]

  v56.m128i_i64[0] = (__int64)"Disabled";
  v57.m128i_i64[1] = (__int64)"disable debug output";
  v58.m128i_i64[1] = (__int64)"Arguments";
  v60.m128i_i64[0] = (__int64)"print pass arguments to pass to 'opt'";
  v61.m128i_i64[0] = (__int64)"Structure";
  v62.m128i_i64[1] = (__int64)"print pass structure before run()";
  v63.m128i_i64[1] = (__int64)"Executions";
  v65.m128i_i64[0] = (__int64)"print pass name before it is executed";
  v66.m128i_i64[0] = (__int64)"Details";
  v67.m128i_i64[1] = (__int64)"print pass details when it is executed";
  v56.m128i_i64[1] = 8;
  v57.m128i_i32[0] = 0;
  v58.m128i_i64[0] = 20;
  v59.m128i_i64[0] = 9;
  v59.m128i_i32[2] = 1;
  v60.m128i_i64[1] = 37;
  v61.m128i_i64[1] = 9;
  v62.m128i_i32[0] = 2;
  v63.m128i_i64[0] = 33;
  v64.m128i_i64[0] = 10;
  v64.m128i_i32[2] = 3;
  v65.m128i_i64[1] = 37;
  v66.m128i_i64[1] = 7;
  v67.m128i_i32[0] = 4;
  v68 = 38;
  v53 = v55;
  v54 = 0x400000000LL;
  sub_C8D5F0(&v53, v55, 5, 40);
  v0 = _mm_loadu_si128(&v57);
  v1 = _mm_loadu_si128(&v58);
  v2 = _mm_loadu_si128(&v59);
  v3 = (__m128i *)&v53[40 * (unsigned int)v54];
  v4 = v68;
  *v3 = _mm_loadu_si128(&v56);
  v5 = _mm_loadu_si128(&v60);
  v3[1] = v0;
  v6 = _mm_loadu_si128(&v61);
  v3[2] = v1;
  v7 = _mm_loadu_si128(&v62);
  v3[3] = v2;
  v8 = _mm_loadu_si128(&v63);
  v3[4] = v5;
  v9 = _mm_loadu_si128(&v64);
  v3[5] = v6;
  v10 = _mm_loadu_si128(&v65);
  v3[6] = v7;
  v11 = _mm_loadu_si128(&v66);
  v3[7] = v8;
  v12 = _mm_loadu_si128(&v67);
  v3[12].m128i_i64[0] = v4;
  v3[8] = v9;
  v3[9] = v10;
  v3[10] = v11;
  v3[11] = v12;
  LODWORD(v54) = v54 + 5;
  qword_4F81B00 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F81B50 = 0x100000000LL;
  word_4F81B10 = 0;
  dword_4F81B0C &= 0x8000u;
  qword_4F81B18 = 0;
  qword_4F81B20 = 0;
  dword_4F81B08 = v13;
  qword_4F81B28 = 0;
  qword_4F81B30 = 0;
  qword_4F81B38 = 0;
  qword_4F81B40 = 0;
  qword_4F81B48 = (__int64)&unk_4F81B58;
  qword_4F81B60 = 0;
  qword_4F81B68 = (__int64)&unk_4F81B80;
  qword_4F81B70 = 1;
  dword_4F81B78 = 0;
  byte_4F81B7C = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_4F81B50;
  v16 = (unsigned int)qword_4F81B50 + 1LL;
  if ( v16 > HIDWORD(qword_4F81B50) )
  {
    sub_C8D5F0((char *)&unk_4F81B58 - 16, &unk_4F81B58, v16, 8);
    v15 = (unsigned int)qword_4F81B50;
  }
  v17 = "debug-pass";
  *(_QWORD *)(qword_4F81B48 + 8 * v15) = v14;
  qword_4F81B00 = (__int64)&off_49DA648;
  qword_4F81BA0 = (__int64)&off_49DA5F8;
  qword_4F81BB0 = (__int64)&unk_4F81BC0;
  qword_4F81BB8 = 0x800000000LL;
  qword_4F81D58 = (__int64)nullsub_72;
  qword_4F81D50 = (__int64)sub_B7E710;
  LODWORD(qword_4F81B50) = qword_4F81B50 + 1;
  qword_4F81B88 = 0;
  qword_4F81B98 = 0;
  qword_4F81B90 = (__int64)&off_49DA5D8;
  qword_4F81BA8 = (__int64)&qword_4F81B00;
  sub_C53080(&qword_4F81B00, "debug-pass", 10);
  qword_4F81B30 = 46;
  v18 = v53;
  LOBYTE(dword_4F81B0C) = dword_4F81B0C & 0x9F | 0x20;
  qword_4F81B28 = (__int64)"Print legacy PassManager debugging information";
  v19 = &v53[40 * (unsigned int)v54];
  if ( v53 != v19 )
  {
    do
    {
      v20 = *((_DWORD *)v18 + 4);
      v21 = *((_QWORD *)v18 + 3);
      v58.m128i_i64[0] = (__int64)&off_49DA5D8;
      v22 = *((_QWORD *)v18 + 4);
      v23 = *(_QWORD *)v18;
      v58.m128i_i8[12] = 1;
      v24 = *((_QWORD *)v18 + 1);
      v58.m128i_i32[2] = v20;
      v25 = (unsigned int)qword_4F81BB8;
      v57.m128i_i64[0] = v21;
      v26 = &v56;
      v57.m128i_i64[1] = v22;
      v56.m128i_i64[0] = v23;
      v27 = qword_4F81BB8;
      v56.m128i_i64[1] = v24;
      if ( (unsigned __int64)(unsigned int)qword_4F81BB8 + 1 > HIDWORD(qword_4F81BB8) )
      {
        if ( qword_4F81BB0 > (unsigned __int64)&v56
          || (unsigned __int64)&v56 >= qword_4F81BB0 + 48 * (unsigned __int64)(unsigned int)qword_4F81BB8 )
        {
          v49 = -1;
          v50 = 0;
        }
        else
        {
          v50 = 1;
          v49 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v56.m128i_i64 - qword_4F81BB0) >> 4);
        }
        v51 = v24;
        v41 = (__m128i *)sub_C8D7D0(
                           (char *)&unk_4F81BC0 - 16,
                           &unk_4F81BC0,
                           (unsigned int)qword_4F81BB8 + 1LL,
                           48,
                           &v52);
        v42 = (void *)qword_4F81BB0;
        v24 = v51;
        v43 = (__int64)v41;
        v44 = (const __m128i *)qword_4F81BB0;
        v45 = qword_4F81BB0 + 48LL * (unsigned int)qword_4F81BB8;
        if ( qword_4F81BB0 != v45 )
        {
          do
          {
            if ( v41 )
            {
              *v41 = _mm_loadu_si128(v44);
              v41[1] = _mm_loadu_si128(v44 + 1);
              v41[2].m128i_i32[2] = v44[2].m128i_i32[2];
              v46 = v44[2].m128i_i8[12];
              v41[2].m128i_i64[0] = (__int64)&off_49DA5D8;
              v41[2].m128i_i8[12] = v46;
            }
            v44 += 3;
            v41 += 3;
          }
          while ( (const __m128i *)v45 != v44 );
          v42 = (void *)qword_4F81BB0;
        }
        v47 = v52;
        if ( v42 != &unk_4F81BC0 )
        {
          v48 = v52;
          _libc_free(v42, v43, v44);
          v24 = v51;
          v47 = v48;
        }
        HIDWORD(qword_4F81BB8) = v47;
        v25 = (unsigned int)qword_4F81BB8;
        v26 = &v56;
        qword_4F81BB0 = v43;
        v27 = qword_4F81BB8;
        if ( v50 )
          v26 = (__m128i *)(v43 + 48 * v49);
      }
      v28 = qword_4F81BB0 + 48 * v25;
      if ( v28 )
      {
        v29 = _mm_loadu_si128(v26 + 1);
        *(__m128i *)v28 = _mm_loadu_si128(v26);
        *(__m128i *)(v28 + 16) = v29;
        *(_DWORD *)(v28 + 40) = v26[2].m128i_i32[2];
        v30 = v26[2].m128i_i8[12];
        *(_QWORD *)(v28 + 32) = &off_49DA5D8;
        *(_BYTE *)(v28 + 44) = v30;
        v27 = qword_4F81BB8;
      }
      v17 = (const char *)v23;
      v18 += 40;
      LODWORD(qword_4F81BB8) = v27 + 1;
      sub_C52F90(qword_4F81BA8, v23, v24, v26);
    }
    while ( v19 != v18 );
  }
  sub_C53130(&qword_4F81B00);
  if ( v53 != v55 )
    _libc_free(v53, v17, v31);
  __cxa_atexit(sub_B7F280, &qword_4F81B00, &qword_4A427C0);
  qword_4F81A20 = (__int64)&unk_49DC150;
  v32 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F81A9C = 1;
  qword_4F81A70 = 0x100000000LL;
  dword_4F81A2C &= 0x8000u;
  qword_4F81A38 = 0;
  qword_4F81A40 = 0;
  qword_4F81A48 = 0;
  dword_4F81A28 = v32;
  word_4F81A30 = 0;
  qword_4F81A50 = 0;
  qword_4F81A58 = 0;
  qword_4F81A60 = 0;
  qword_4F81A68 = (__int64)&unk_4F81A78;
  qword_4F81A80 = 0;
  qword_4F81A88 = (__int64)&unk_4F81AA0;
  qword_4F81A90 = 1;
  dword_4F81A98 = 0;
  v33 = sub_C57470();
  v34 = (unsigned int)qword_4F81A70;
  v35 = (unsigned int)qword_4F81A70 + 1LL;
  if ( v35 > HIDWORD(qword_4F81A70) )
  {
    sub_C8D5F0((char *)&unk_4F81A78 - 16, &unk_4F81A78, v35, 8);
    v34 = (unsigned int)qword_4F81A70;
  }
  *(_QWORD *)(qword_4F81A68 + 8 * v34) = v33;
  LODWORD(qword_4F81A70) = qword_4F81A70 + 1;
  qword_4F81AA8 = 0;
  qword_4F81AB0 = (__int64)&unk_49DA090;
  qword_4F81AB8 = 0;
  qword_4F81A20 = (__int64)&unk_49DBF90;
  qword_4F81AC0 = (__int64)&unk_49DC230;
  qword_4F81AE0 = (__int64)nullsub_58;
  qword_4F81AD8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4F81A20, "pass-control", 12);
  qword_4F81A50 = 55;
  LODWORD(qword_4F81AA8) = -1;
  BYTE4(qword_4F81AB8) = 1;
  LODWORD(qword_4F81AB8) = -1;
  LOBYTE(dword_4F81A2C) = dword_4F81A2C & 0x9F | 0x20;
  qword_4F81A48 = (__int64)"Disable all optional passes after specified pass number";
  sub_C53130(&qword_4F81A20);
  __cxa_atexit(sub_B2B680, &qword_4F81A20, &qword_4A427C0);
  qword_4F81920 = (__int64)&unk_49DC150;
  v36 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F81938 = 0;
  qword_4F81940 = 0;
  qword_4F81948 = 0;
  qword_4F81950 = 0;
  dword_4F8192C = dword_4F8192C & 0x8000 | 1;
  word_4F81930 = 0;
  qword_4F81970 = 0x100000000LL;
  dword_4F81928 = v36;
  qword_4F81958 = 0;
  qword_4F81960 = 0;
  qword_4F81968 = (__int64)&unk_4F81978;
  qword_4F81980 = 0;
  qword_4F81988 = (__int64)&unk_4F819A0;
  qword_4F81990 = 1;
  dword_4F81998 = 0;
  byte_4F8199C = 1;
  v37 = sub_C57470();
  v38 = (unsigned int)qword_4F81970;
  v39 = (unsigned int)qword_4F81970 + 1LL;
  if ( v39 > HIDWORD(qword_4F81970) )
  {
    sub_C8D5F0((char *)&unk_4F81978 - 16, &unk_4F81978, v39, 8);
    v38 = (unsigned int)qword_4F81970;
  }
  *(_QWORD *)(qword_4F81968 + 8 * v38) = v37;
  LODWORD(qword_4F81970) = qword_4F81970 + 1;
  qword_4F819A8 = 0;
  qword_4F81920 = (__int64)&unk_49DA6C8;
  qword_4F819F8 = (__int64)&unk_49DC230;
  qword_4F819B0 = 0;
  qword_4F81A18 = (__int64)nullsub_73;
  qword_4F819B8 = 0;
  qword_4F81A10 = (__int64)sub_B7E730;
  qword_4F819C0 = 0;
  qword_4F819C8 = 0;
  qword_4F819D0 = 0;
  byte_4F819D8 = 0;
  qword_4F819E0 = 0;
  qword_4F819E8 = 0;
  qword_4F819F0 = 0;
  sub_C53080(&qword_4F81920, "disable-passno", 14);
  BYTE1(dword_4F8192C) |= 2u;
  qword_4F81960 = 4;
  qword_4F81950 = 88;
  LOBYTE(dword_4F8192C) = dword_4F8192C & 0x9F | 0x20;
  qword_4F81958 = (__int64)"list";
  qword_4F81948 = (__int64)"Disable any optional pass(es) by specifying thepass number(s) in a comma separated list.";
  sub_C53130(&qword_4F81920);
  return __cxa_atexit(sub_B7F3A0, &qword_4F81920, &qword_4A427C0);
}
