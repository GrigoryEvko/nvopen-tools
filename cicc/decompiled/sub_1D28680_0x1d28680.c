// Function: sub_1D28680
// Address: 0x1d28680
//
__int64 __fastcall sub_1D28680(
        _QWORD *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int32 a6,
        int a7,
        __int128 a8,
        __int128 a9,
        __int64 a10,
        unsigned int a11,
        char a12)
{
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i v20; // xmm0
  __m128i *v21; // rsi
  __m128i *v22; // rax
  __m128i v23; // xmm1
  __m128i *v24; // rsi
  __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rdi
  __int64 v28; // rax
  unsigned int v29; // edx
  unsigned __int8 v30; // al
  __int64 v31; // r14
  __int64 v32; // rax
  int v33; // edx
  int v34; // r15d
  __int64 v35; // rax
  __int64 v36; // rdx
  const __m128i *v37; // rdi
  unsigned int v38; // r14d
  const __m128i *v39; // rsi
  const __m128i *v40; // rax
  void (***v41)(); // rdi
  void (*v42)(); // r8
  __int64 v43; // rsi
  __int64 v44; // r14
  __m128i v46; // xmm2
  char v47; // [rsp+1Ch] [rbp-1004h]
  int v49; // [rsp+28h] [rbp-FF8h]
  __int64 v50; // [rsp+28h] [rbp-FF8h]
  const __m128i *v51; // [rsp+90h] [rbp-F90h] BYREF
  __m128i *v52; // [rsp+98h] [rbp-F88h]
  const __m128i *v53; // [rsp+A0h] [rbp-F80h]
  char v54[16]; // [rsp+B0h] [rbp-F70h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-F60h]
  __m128i v56; // [rsp+D0h] [rbp-F50h] BYREF
  __m128i v57; // [rsp+E0h] [rbp-F40h] BYREF
  __int64 v58; // [rsp+F0h] [rbp-F30h]
  __int64 v59; // [rsp+100h] [rbp-F20h] BYREF
  __int64 v60; // [rsp+108h] [rbp-F18h]
  __int64 v61; // [rsp+110h] [rbp-F10h]
  unsigned __int64 v62; // [rsp+118h] [rbp-F08h]
  __int64 v63; // [rsp+120h] [rbp-F00h]
  __int64 v64; // [rsp+128h] [rbp-EF8h]
  __int64 v65; // [rsp+130h] [rbp-EF0h]
  const __m128i *v66; // [rsp+138h] [rbp-EE8h] BYREF
  __m128i *v67; // [rsp+140h] [rbp-EE0h]
  const __m128i *v68; // [rsp+148h] [rbp-ED8h]
  _QWORD *v69; // [rsp+150h] [rbp-ED0h]
  __int64 v70; // [rsp+158h] [rbp-EC8h] BYREF
  int v71; // [rsp+160h] [rbp-EC0h]
  __int64 v72; // [rsp+168h] [rbp-EB8h]
  _BYTE *v73; // [rsp+170h] [rbp-EB0h]
  __int64 v74; // [rsp+178h] [rbp-EA8h]
  _BYTE v75[1536]; // [rsp+180h] [rbp-EA0h] BYREF
  _BYTE *v76; // [rsp+780h] [rbp-8A0h]
  __int64 v77; // [rsp+788h] [rbp-898h]
  _BYTE v78[512]; // [rsp+790h] [rbp-890h] BYREF
  _BYTE *v79; // [rsp+990h] [rbp-690h]
  __int64 v80; // [rsp+998h] [rbp-688h]
  _BYTE v81[1536]; // [rsp+9A0h] [rbp-680h] BYREF
  _BYTE *v82; // [rsp+FA0h] [rbp-80h]
  __int64 v83; // [rsp+FA8h] [rbp-78h]
  _BYTE v84[112]; // [rsp+FB0h] [rbp-70h] BYREF

  v17 = a1[4];
  v51 = 0;
  v47 = a12;
  v52 = 0;
  v53 = 0;
  v56 = 0u;
  v57 = 0u;
  LODWORD(v58) = 0;
  v18 = sub_1E0A0C0(v17);
  v57.m128i_i64[1] = sub_15A9620(v18, a1[6], 0);
  v56.m128i_i64[1] = a5;
  v57.m128i_i32[0] = a6;
  sub_1D27190(&v51, 0, &v56);
  v19 = sub_1643330((_QWORD *)a1[6]);
  v20 = _mm_loadu_si128((const __m128i *)&a8);
  v21 = v52;
  v57.m128i_i64[1] = v19;
  v56.m128i_i64[1] = a8;
  v57.m128i_i32[0] = v20.m128i_i32[2];
  v22 = (__m128i *)v53;
  if ( v52 == v53 )
  {
    sub_1D27190(&v51, v52, &v56);
    v46 = _mm_loadu_si128((const __m128i *)&a9);
    v24 = v52;
    v57.m128i_i64[1] = a10;
    v56.m128i_i64[1] = a9;
    v57.m128i_i32[0] = v46.m128i_i32[2];
    if ( v53 != v52 )
    {
      if ( !v52 )
        goto LABEL_6;
      goto LABEL_5;
    }
  }
  else
  {
    if ( v52 )
    {
      *v52 = _mm_loadu_si128(&v56);
      v21[1] = _mm_loadu_si128(&v57);
      v21[2].m128i_i64[0] = v58;
      v21 = v52;
      v22 = (__m128i *)v53;
    }
    v23 = _mm_loadu_si128((const __m128i *)&a9);
    v24 = (__m128i *)((char *)v21 + 40);
    v52 = v24;
    v56.m128i_i64[1] = a9;
    v57.m128i_i64[1] = a10;
    v57.m128i_i32[0] = v23.m128i_i32[2];
    if ( v24 != v22 )
    {
LABEL_5:
      *v24 = _mm_loadu_si128(&v56);
      v24[1] = _mm_loadu_si128(&v57);
      v24[2].m128i_i64[0] = v58;
      v24 = v52;
LABEL_6:
      v52 = (__m128i *)((char *)v24 + 40);
      goto LABEL_7;
    }
  }
  sub_1D27190(&v51, v24, &v56);
LABEL_7:
  v49 = sub_1F405B0(a11);
  if ( v49 == 462 )
    sub_16BD130("Unsupported element size", 1u);
  v25 = *(_QWORD *)a4;
  v62 = 0xFFFFFFFF00000020LL;
  v73 = v75;
  v74 = 0x2000000000LL;
  v77 = 0x2000000000LL;
  v80 = 0x2000000000LL;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = a1;
  v71 = 0;
  v72 = 0;
  v76 = v78;
  v79 = v81;
  v82 = v84;
  v83 = 0x400000000LL;
  v70 = v25;
  if ( v25 )
    sub_1623A60((__int64)&v70, v25, 2);
  v26 = *(_DWORD *)(a4 + 8);
  v27 = a1[4];
  v59 = a2;
  v71 = v26;
  LODWORD(v60) = a3;
  v28 = sub_1E0A0C0(v27);
  v29 = 8 * sub_15A9520(v28, 0);
  if ( v29 == 32 )
  {
    v30 = 5;
  }
  else if ( v29 > 0x20 )
  {
    v30 = 6;
    if ( v29 != 64 )
    {
      v30 = 0;
      if ( v29 == 128 )
        v30 = 7;
    }
  }
  else
  {
    v30 = 3;
    if ( v29 != 8 )
      v30 = 4 * (v29 == 16);
  }
  v31 = v49;
  v32 = sub_1D27640((__int64)a1, *(char **)(a1[2] + 8LL * v49 + 74096), v30, 0);
  v34 = v33;
  v50 = v32;
  v35 = sub_1643270((_QWORD *)a1[6]);
  v36 = a1[2];
  v61 = v35;
  v37 = v66;
  v38 = *(_DWORD *)(v36 + 4 * v31 + 79648);
  v64 = v50;
  LODWORD(v65) = v34;
  LODWORD(v63) = v38;
  v66 = v51;
  LODWORD(v35) = -858993459 * (((char *)v52 - (char *)v51) >> 3);
  v67 = v52;
  v39 = v68;
  v51 = 0;
  v52 = 0;
  HIDWORD(v62) = v35;
  v40 = v53;
  v53 = 0;
  v68 = v40;
  if ( v37 )
    j_j___libc_free_0(v37, (char *)v39 - (char *)v37);
  v41 = (void (***)())v69[2];
  v42 = **v41;
  if ( v42 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v42)(v41, v69[4], v38, &v66);
  v43 = a1[2];
  LOBYTE(v62) = v62 & 0xDF;
  BYTE1(v62) = v47;
  sub_2056920(v54, v43, &v59);
  v44 = v55;
  if ( v82 != v84 )
    _libc_free((unsigned __int64)v82);
  if ( v79 != v81 )
    _libc_free((unsigned __int64)v79);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 )
    sub_161E7C0((__int64)&v70, v70);
  if ( v66 )
    j_j___libc_free_0(v66, (char *)v68 - (char *)v66);
  if ( v51 )
    j_j___libc_free_0(v51, (char *)v53 - (char *)v51);
  return v44;
}
