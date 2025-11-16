// Function: ctor_126_0
// Address: 0x4ade70
//
int ctor_126_0()
{
  int v0; // edx
  const char *v1; // rsi
  __int64 v2; // rax
  const char **v3; // r13
  __int64 v4; // r14
  const char *v5; // r8
  const char *v6; // rdx
  __int64 v7; // rcx
  const char *v8; // r10
  const char *v9; // r9
  __m128i *v10; // rbx
  __m128i *v11; // rbx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r11
  __int64 v17; // rax
  int v18; // r11d
  _QWORD *v19; // rdi
  __m128i *v20; // rax
  __int64 v21; // r14
  const __m128i *v22; // rsi
  __int8 v23; // dl
  const char *v24; // [rsp+0h] [rbp-150h]
  const char *v25; // [rsp+0h] [rbp-150h]
  const char *v26; // [rsp+8h] [rbp-148h]
  const char *v27; // [rsp+8h] [rbp-148h]
  unsigned int v28; // [rsp+14h] [rbp-13Ch]
  unsigned int v29; // [rsp+14h] [rbp-13Ch]
  const char *v30; // [rsp+18h] [rbp-138h]
  const char *v31; // [rsp+18h] [rbp-138h]
  const char *v32; // [rsp+20h] [rbp-130h]
  const char *v33; // [rsp+20h] [rbp-130h]
  int v34; // [rsp+28h] [rbp-128h]
  const char *v35; // [rsp+28h] [rbp-128h]
  int v36; // [rsp+28h] [rbp-128h]
  const char **v37; // [rsp+40h] [rbp-110h]
  const char **v38; // [rsp+48h] [rbp-108h]
  _QWORD v39[2]; // [rsp+50h] [rbp-100h] BYREF
  char v40; // [rsp+60h] [rbp-F0h]
  char v41; // [rsp+61h] [rbp-EFh]
  const char **v42; // [rsp+70h] [rbp-E0h]
  __int64 v43; // [rsp+78h] [rbp-D8h]
  _QWORD v44[2]; // [rsp+80h] [rbp-D0h] BYREF
  int v45; // [rsp+90h] [rbp-C0h]
  const char *v46; // [rsp+98h] [rbp-B8h]
  __int64 v47; // [rsp+A0h] [rbp-B0h]
  const char *v48; // [rsp+A8h] [rbp-A8h]
  __int64 v49; // [rsp+B0h] [rbp-A0h]
  int v50; // [rsp+B8h] [rbp-98h]
  const char *v51; // [rsp+C0h] [rbp-90h]
  __int64 v52; // [rsp+C8h] [rbp-88h]
  char *v53; // [rsp+D0h] [rbp-80h]
  __int64 v54; // [rsp+D8h] [rbp-78h]
  int v55; // [rsp+E0h] [rbp-70h]
  const char *v56; // [rsp+E8h] [rbp-68h]
  __int64 v57; // [rsp+F0h] [rbp-60h]

  v42 = (const char **)v44;
  v44[0] = "none";
  v46 = "None.";
  v48 = "all-non-critical";
  v51 = "All non-critical edges.";
  v53 = "all";
  v56 = "All edges.";
  v43 = 0x400000003LL;
  v44[1] = 4;
  v45 = 0;
  qword_4F99960[0] = &unk_49EED30;
  v47 = 5;
  v49 = 16;
  v50 = 1;
  v52 = 23;
  v54 = 3;
  v55 = 2;
  v57 = 10;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  v1 = "force-summary-edges-cold";
  WORD2(qword_4F99960[1]) &= 0xF000u;
  qword_4F99960[9] = &unk_4FA01C0;
  qword_4F99960[11] = &qword_4F99960[15];
  qword_4F99960[12] = &qword_4F99960[15];
  qword_4F99960[21] = &unk_49EB4F8;
  LODWORD(qword_4F99960[1]) = v0;
  qword_4F99960[2] = 0;
  qword_4F99960[0] = &unk_49EB568;
  qword_4F99960[3] = 0;
  qword_4F99960[4] = 0;
  qword_4F99960[23] = &unk_49EB518;
  qword_4F99960[25] = &qword_4F99960[27];
  qword_4F99960[26] = 0x800000000LL;
  qword_4F99960[5] = 0;
  qword_4F99960[6] = 0;
  qword_4F99960[7] = 0;
  qword_4F99960[8] = 0;
  qword_4F99960[10] = 0;
  qword_4F99960[13] = 4;
  LODWORD(qword_4F99960[14]) = 0;
  LOBYTE(qword_4F99960[19]) = 0;
  qword_4F99960[20] = 0;
  BYTE4(qword_4F99960[22]) = 0;
  qword_4F99960[24] = qword_4F99960;
  sub_16B8280(qword_4F99960, "force-summary-edges-cold", 24);
  BYTE4(qword_4F99960[1]) = BYTE4(qword_4F99960[1]) & 0x9F | 0x20;
  if ( qword_4F99960[20] )
  {
    v2 = sub_16E8CB0();
    v1 = (const char *)v39;
    v41 = 1;
    v39[0] = "cl::location(x) specified more than once!";
    v40 = 3;
    sub_16B1F90(qword_4F99960, v39, 0, 0, v2);
  }
  else
  {
    BYTE4(qword_4F99960[22]) = 1;
    qword_4F99960[20] = &dword_4F99BC0;
    LODWORD(qword_4F99960[22]) = dword_4F99BC0;
  }
  qword_4F99960[6] = 47;
  qword_4F99960[5] = "Force all edges in the function summary to cold";
  v37 = v42;
  v3 = v42;
  v38 = &v42[5 * (unsigned int)v43];
  if ( v42 != v38 )
  {
    do
    {
      v4 = LODWORD(qword_4F99960[26]);
      v5 = *v3;
      v6 = v3[1];
      v7 = *((unsigned int *)v3 + 4);
      v8 = v3[3];
      v9 = v3[4];
      if ( LODWORD(qword_4F99960[26]) >= HIDWORD(qword_4F99960[26]) )
      {
        v24 = v3[4];
        v26 = v3[3];
        v28 = *((_DWORD *)v3 + 4);
        v13 = (((unsigned __int64)HIDWORD(qword_4F99960[26]) + 2) >> 1) | (HIDWORD(qword_4F99960[26]) + 2LL);
        v30 = v3[1];
        v32 = *v3;
        v14 = v13 >> 2;
        v15 = (((v13 >> 2) | v13) >> 4) | (v13 >> 2) | v13;
        v16 = ((v15 >> 8) | v15 | (((v15 >> 8) | v15) >> 16) | (((v15 >> 8) | v15) >> 32)) + 1;
        if ( v16 > 0xFFFFFFFF )
          v16 = 0xFFFFFFFFLL;
        v34 = v16;
        v17 = malloc(48 * v16, v14, v6, v7, v5, v9);
        v18 = v34;
        v5 = v32;
        v7 = v28;
        v6 = v30;
        v10 = (__m128i *)v17;
        v8 = v26;
        v9 = v24;
        if ( !v17 )
        {
          sub_16BD1C0("Allocation failed");
          v4 = LODWORD(qword_4F99960[26]);
          v9 = v24;
          v8 = v26;
          v7 = v28;
          v6 = v30;
          v5 = v32;
          v18 = v34;
        }
        v19 = (_QWORD *)qword_4F99960[25];
        v20 = v10;
        v21 = qword_4F99960[25] + 48 * v4;
        v22 = (const __m128i *)qword_4F99960[25];
        if ( qword_4F99960[25] != v21 )
        {
          v35 = v6;
          do
          {
            if ( v20 )
            {
              *v20 = _mm_loadu_si128(v22);
              v20[1] = _mm_loadu_si128(v22 + 1);
              v20[2].m128i_i32[2] = v22[2].m128i_i32[2];
              v23 = v22[2].m128i_i8[12];
              v20[2].m128i_i64[0] = (__int64)&unk_49EB4F8;
              v20[2].m128i_i8[12] = v23;
            }
            v22 += 3;
            v20 += 3;
          }
          while ( (const __m128i *)v21 != v22 );
          v6 = v35;
        }
        if ( v19 != &qword_4F99960[27] )
        {
          v25 = v9;
          v27 = v8;
          v29 = v7;
          v31 = v6;
          v33 = v5;
          v36 = v18;
          _libc_free(v19, v22);
          v9 = v25;
          v8 = v27;
          v7 = v29;
          v6 = v31;
          v5 = v33;
          v18 = v36;
        }
        qword_4F99960[25] = v10;
        LODWORD(v4) = qword_4F99960[26];
        HIDWORD(qword_4F99960[26]) = v18;
      }
      else
      {
        v10 = (__m128i *)qword_4F99960[25];
      }
      v11 = &v10[3 * (unsigned int)v4];
      if ( v11 )
      {
        v11->m128i_i64[0] = (__int64)v5;
        v11->m128i_i64[1] = (__int64)v6;
        v11[1].m128i_i64[0] = (__int64)v8;
        v11[1].m128i_i64[1] = (__int64)v9;
        v11[2].m128i_i32[2] = v7;
        v11[2].m128i_i8[12] = 1;
        v11[2].m128i_i64[0] = (__int64)&unk_49EB4F8;
        LODWORD(v4) = qword_4F99960[26];
      }
      v1 = v5;
      v3 += 5;
      LODWORD(qword_4F99960[26]) = v4 + 1;
      sub_16B7FD0(qword_4F99960[24], v5, v6, v7);
    }
    while ( v38 != v3 );
  }
  sub_16B88A0(qword_4F99960);
  if ( v37 != v44 )
    _libc_free(v37, v1);
  return __cxa_atexit(sub_142B470, qword_4F99960, &qword_4A427C0);
}
