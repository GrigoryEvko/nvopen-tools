// Function: ctor_493_0
// Address: 0x556960
//
__int64 ctor_493_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  const char *v12; // rsi
  _QWORD *v13; // r14
  _QWORD *v14; // r13
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rdx
  const char *v18; // r15
  __int64 v19; // r10
  __int64 v20; // rax
  const __m128i *v21; // rcx
  int v22; // edx
  __int64 v23; // rax
  __m128i v24; // xmm1
  __int8 v25; // dl
  pthread_rwlock_t *v26; // rdi
  pthread_rwlock_t *v27; // rdi
  pthread_rwlock_t *v28; // rdi
  pthread_rwlock_t *v29; // rax
  __m128i *v31; // rax
  void *v32; // rdi
  __int64 v33; // rsi
  const __m128i *v34; // rdx
  __int64 v35; // rcx
  __int8 v36; // di
  int v37; // eax
  int v38; // [rsp+8h] [rbp-148h]
  unsigned __int64 v39; // [rsp+10h] [rbp-140h]
  char v40; // [rsp+1Fh] [rbp-131h]
  __int64 v41; // [rsp+20h] [rbp-130h]
  __int64 v42; // [rsp+38h] [rbp-118h] BYREF
  _QWORD v43[5]; // [rsp+40h] [rbp-110h] BYREF
  int v44; // [rsp+68h] [rbp-E8h]
  char v45; // [rsp+6Ch] [rbp-E4h]
  _QWORD *v46; // [rsp+70h] [rbp-E0h]
  __int64 v47; // [rsp+78h] [rbp-D8h]
  _QWORD v48[2]; // [rsp+80h] [rbp-D0h] BYREF
  int v49; // [rsp+90h] [rbp-C0h]
  const char *v50; // [rsp+98h] [rbp-B8h]
  __int64 v51; // [rsp+A0h] [rbp-B0h]
  const char *v52; // [rsp+A8h] [rbp-A8h]
  __int64 v53; // [rsp+B0h] [rbp-A0h]
  int v54; // [rsp+B8h] [rbp-98h]
  const char *v55; // [rsp+C0h] [rbp-90h]
  __int64 v56; // [rsp+C8h] [rbp-88h]

  qword_5008F40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5008F90 = 0x100000000LL;
  dword_5008F4C &= 0x8000u;
  word_5008F50 = 0;
  qword_5008F58 = 0;
  qword_5008F60 = 0;
  dword_5008F48 = v0;
  qword_5008F68 = 0;
  qword_5008F70 = 0;
  qword_5008F78 = 0;
  qword_5008F80 = 0;
  qword_5008F88 = (__int64)&unk_5008F98;
  qword_5008FA0 = 0;
  qword_5008FA8 = (__int64)&unk_5008FC0;
  qword_5008FB0 = 1;
  dword_5008FB8 = 0;
  byte_5008FBC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5008F90;
  v3 = (unsigned int)qword_5008F90 + 1LL;
  if ( v3 > HIDWORD(qword_5008F90) )
  {
    sub_C8D5F0((char *)&unk_5008F98 - 16, &unk_5008F98, v3, 8);
    v2 = (unsigned int)qword_5008F90;
  }
  *(_QWORD *)(qword_5008F88 + 8 * v2) = v1;
  LODWORD(qword_5008F90) = qword_5008F90 + 1;
  qword_5008FC8 = 0;
  qword_5008FD0 = (__int64)&unk_49D9748;
  qword_5008FD8 = 0;
  qword_5008F40 = (__int64)&unk_49DC090;
  qword_5008FE0 = (__int64)&unk_49DC1D0;
  qword_5009000 = (__int64)nullsub_23;
  qword_5008FF8 = (__int64)sub_984030;
  sub_C53080(&qword_5008F40, "debugify-quiet", 14);
  qword_5008F70 = 32;
  qword_5008F68 = (__int64)"Suppress verbose debugify output";
  sub_C53130(&qword_5008F40);
  __cxa_atexit(sub_984900, &qword_5008F40, &qword_4A427C0);
  qword_5008E60 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5008EB0 = 0x100000000LL;
  dword_5008E6C &= 0x8000u;
  word_5008E70 = 0;
  qword_5008E78 = 0;
  qword_5008E80 = 0;
  dword_5008E68 = v4;
  qword_5008E88 = 0;
  qword_5008E90 = 0;
  qword_5008E98 = 0;
  qword_5008EA0 = 0;
  qword_5008EA8 = (__int64)&unk_5008EB8;
  qword_5008EC0 = 0;
  qword_5008EC8 = (__int64)&unk_5008EE0;
  qword_5008ED0 = 1;
  dword_5008ED8 = 0;
  byte_5008EDC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5008EB0;
  v7 = (unsigned int)qword_5008EB0 + 1LL;
  if ( v7 > HIDWORD(qword_5008EB0) )
  {
    sub_C8D5F0((char *)&unk_5008EB8 - 16, &unk_5008EB8, v7, 8);
    v6 = (unsigned int)qword_5008EB0;
  }
  *(_QWORD *)(qword_5008EA8 + 8 * v6) = v5;
  LODWORD(qword_5008EB0) = qword_5008EB0 + 1;
  byte_5008F00 = 0;
  qword_5008EF0 = (__int64)&unk_49DB998;
  qword_5008EE8 = 0;
  qword_5008EF8 = 0;
  qword_5008E60 = (__int64)&unk_49DB9B8;
  qword_5008F08 = (__int64)&unk_49DC2C0;
  qword_5008F28 = (__int64)nullsub_121;
  qword_5008F20 = (__int64)sub_C1A370;
  sub_C53080(&qword_5008E60, "debugify-func-limit", 19);
  qword_5008E90 = 47;
  qword_5008E88 = (__int64)"Set max number of processed functions per pass.";
  qword_5008EE8 = 0xFFFFFFFFLL;
  qword_5008EF8 = 0xFFFFFFFFLL;
  byte_5008F00 = 1;
  sub_C53130(&qword_5008E60);
  __cxa_atexit(sub_C1A610, &qword_5008E60, &qword_4A427C0);
  v46 = v48;
  v48[0] = "locations";
  v50 = "Locations only";
  v52 = "location+variables";
  v55 = "Locations and Variables";
  v47 = 0x400000002LL;
  v48[1] = 9;
  v49 = 0;
  v51 = 14;
  v53 = 18;
  v54 = 1;
  v56 = 23;
  qword_5008C00 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5008C50 = 0x100000000LL;
  dword_5008C0C &= 0x8000u;
  word_5008C10 = 0;
  qword_5008C18 = 0;
  qword_5008C20 = 0;
  dword_5008C08 = v8;
  qword_5008C28 = 0;
  qword_5008C30 = 0;
  qword_5008C38 = 0;
  qword_5008C40 = 0;
  qword_5008C48 = (__int64)&unk_5008C58;
  qword_5008C60 = 0;
  qword_5008C68 = (__int64)&unk_5008C80;
  qword_5008C70 = 1;
  dword_5008C78 = 0;
  byte_5008C7C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5008C50;
  v11 = (unsigned int)qword_5008C50 + 1LL;
  if ( v11 > HIDWORD(qword_5008C50) )
  {
    sub_C8D5F0((char *)&unk_5008C58 - 16, &unk_5008C58, v11, 8);
    v10 = (unsigned int)qword_5008C50;
  }
  v12 = "debugify-level";
  *(_QWORD *)(qword_5008C48 + 8 * v10) = v9;
  qword_5008C00 = (__int64)&off_4A22668;
  qword_5008CA0 = (__int64)&off_4A22618;
  qword_5008CB0 = (__int64)&unk_5008CC0;
  qword_5008CB8 = 0x800000000LL;
  qword_5008E58 = (__int64)nullsub_1557;
  qword_5008E50 = (__int64)sub_29C0A20;
  LODWORD(qword_5008C50) = qword_5008C50 + 1;
  qword_5008C88 = 0;
  qword_5008C98 = 0;
  qword_5008C90 = (__int64)&off_4A225F8;
  qword_5008CA8 = (__int64)&qword_5008C00;
  sub_C53080(&qword_5008C00, "debugify-level", 14);
  qword_5008C30 = 25;
  qword_5008C28 = (__int64)"Kind of debug info to add";
  v13 = v46;
  v14 = &v46[5 * (unsigned int)v47];
  if ( v46 != v14 )
  {
    do
    {
      v15 = *((_DWORD *)v13 + 4);
      v16 = v13[3];
      v43[4] = &off_4A225F8;
      v17 = v13[4];
      v18 = (const char *)*v13;
      v45 = 1;
      v19 = v13[1];
      v44 = v15;
      v20 = (unsigned int)qword_5008CB8;
      v43[2] = v16;
      v21 = (const __m128i *)v43;
      v43[3] = v17;
      v43[0] = v18;
      v22 = qword_5008CB8;
      v43[1] = v19;
      if ( (unsigned __int64)(unsigned int)qword_5008CB8 + 1 > HIDWORD(qword_5008CB8) )
      {
        if ( qword_5008CB0 > (unsigned __int64)v43
          || (unsigned __int64)v43 >= qword_5008CB0 + 48 * (unsigned __int64)(unsigned int)qword_5008CB8 )
        {
          v39 = -1;
          v40 = 0;
        }
        else
        {
          v40 = 1;
          v39 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v43 - qword_5008CB0) >> 4);
        }
        v41 = v19;
        v31 = (__m128i *)sub_C8D7D0(
                           (char *)&unk_5008CC0 - 16,
                           &unk_5008CC0,
                           (unsigned int)qword_5008CB8 + 1LL,
                           48,
                           &v42);
        v32 = (void *)qword_5008CB0;
        v19 = v41;
        v33 = (__int64)v31;
        v34 = (const __m128i *)qword_5008CB0;
        v35 = qword_5008CB0 + 48LL * (unsigned int)qword_5008CB8;
        if ( qword_5008CB0 != v35 )
        {
          do
          {
            if ( v31 )
            {
              *v31 = _mm_loadu_si128(v34);
              v31[1] = _mm_loadu_si128(v34 + 1);
              v31[2].m128i_i32[2] = v34[2].m128i_i32[2];
              v36 = v34[2].m128i_i8[12];
              v31[2].m128i_i64[0] = (__int64)&off_4A225F8;
              v31[2].m128i_i8[12] = v36;
            }
            v34 += 3;
            v31 += 3;
          }
          while ( (const __m128i *)v35 != v34 );
          v32 = (void *)qword_5008CB0;
        }
        v37 = v42;
        if ( v32 != &unk_5008CC0 )
        {
          v38 = v42;
          _libc_free(v32, v33);
          v19 = v41;
          v37 = v38;
        }
        HIDWORD(qword_5008CB8) = v37;
        v20 = (unsigned int)qword_5008CB8;
        v21 = (const __m128i *)v43;
        qword_5008CB0 = v33;
        v22 = qword_5008CB8;
        if ( v40 )
          v21 = (const __m128i *)(v33 + 48 * v39);
      }
      v23 = qword_5008CB0 + 48 * v20;
      if ( v23 )
      {
        v24 = _mm_loadu_si128(v21 + 1);
        *(__m128i *)v23 = _mm_loadu_si128(v21);
        *(__m128i *)(v23 + 16) = v24;
        *(_DWORD *)(v23 + 40) = v21[2].m128i_i32[2];
        v25 = v21[2].m128i_i8[12];
        *(_QWORD *)(v23 + 32) = &off_4A225F8;
        *(_BYTE *)(v23 + 44) = v25;
        v22 = qword_5008CB8;
      }
      v12 = v18;
      v13 += 5;
      LODWORD(qword_5008CB8) = v22 + 1;
      sub_C52F90(qword_5008CA8, v18, v19, v21);
    }
    while ( v14 != v13 );
  }
  LODWORD(qword_5008C88) = 1;
  BYTE4(qword_5008C98) = 1;
  LODWORD(qword_5008C98) = 1;
  sub_C53130(&qword_5008C00);
  if ( v46 != v48 )
    _libc_free(v46, v12);
  __cxa_atexit(sub_29C1620, &qword_5008C00, &qword_4A427C0);
  qword_5008BC8 = 31;
  qword_5008BC0 = (__int64)"Attach debug info to everything";
  qword_5008BD0 = (__int64)"debugify";
  qword_5008BE0 = (__int64)&unk_5008BF8;
  word_5008BE8 = 0;
  qword_5008BD8 = 8;
  qword_5008BF0 = (__int64)sub_29C1850;
  v26 = (pthread_rwlock_t *)((__int64 (*)(void))sub_BC2B00)();
  sub_BC3090(v26);
  qword_5008B88 = 31;
  qword_5008B80 = (__int64)"Check debug info from -debugify";
  qword_5008B90 = (__int64)"check-debugify";
  qword_5008BA0 = (__int64)&unk_5008BB8;
  word_5008BA8 = 0;
  qword_5008B98 = 14;
  qword_5008BB0 = (__int64)sub_29C1AA0;
  v27 = (pthread_rwlock_t *)sub_BC2B00(v26, &qword_5008BC0);
  sub_BC3090(v27);
  qword_5008B48 = 31;
  qword_5008B40 = (__int64)"Attach debug info to a function";
  qword_5008B50 = (__int64)"debugify-function";
  qword_5008B60 = (__int64)&unk_5008B78;
  word_5008B68 = 0;
  qword_5008B58 = 17;
  qword_5008B70 = (__int64)sub_29C1740;
  v28 = (pthread_rwlock_t *)sub_BC2B00(v27, &qword_5008B80);
  sub_BC3090(v28);
  qword_5008B08 = 40;
  qword_5008B00 = (__int64)"Check debug info from -debugify-function";
  qword_5008B10 = (__int64)"check-debugify-function";
  qword_5008B20 = (__int64)&unk_5008B38;
  word_5008B28 = 0;
  qword_5008B18 = 23;
  qword_5008B30 = (__int64)sub_29C1960;
  v29 = (pthread_rwlock_t *)sub_BC2B00(v28, 0);
  return sub_BC3090(v29);
}
