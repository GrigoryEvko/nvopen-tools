// Function: sub_1D278A0
// Address: 0x1d278a0
//
__int64 __fastcall sub_1D278A0(
        _QWORD *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int32 a6,
        int a7,
        __int128 a8,
        int a9,
        __int128 a10,
        __int64 a11,
        unsigned int a12,
        char a13)
{
  __int64 v18; // rdi
  __int64 v19; // rax
  __m128i v20; // xmm1
  __m128i *v21; // rsi
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rax
  unsigned int v26; // edx
  unsigned __int8 v27; // al
  __int64 v28; // r14
  __int64 v29; // rax
  int v30; // edx
  int v31; // r15d
  __int64 v32; // rax
  __int64 v33; // rdx
  const __m128i *v34; // rdi
  unsigned int v35; // r14d
  const __m128i *v36; // rsi
  const __m128i *v37; // rax
  void (***v38)(); // rdi
  void (*v39)(); // r8
  __int64 v40; // rsi
  __int64 v41; // r14
  __m128i v43; // xmm2
  __m128i *v44; // rsi
  __m128i *v45; // rax
  __m128i v46; // xmm3
  char v47; // [rsp+1Ch] [rbp-1014h]
  int v49; // [rsp+28h] [rbp-1008h]
  __int64 v50; // [rsp+28h] [rbp-1008h]
  const __m128i *v51; // [rsp+A0h] [rbp-F90h] BYREF
  __m128i *v52; // [rsp+A8h] [rbp-F88h]
  const __m128i *v53; // [rsp+B0h] [rbp-F80h]
  char v54[16]; // [rsp+C0h] [rbp-F70h] BYREF
  __int64 v55; // [rsp+D0h] [rbp-F60h]
  __m128i v56; // [rsp+E0h] [rbp-F50h] BYREF
  __m128i v57; // [rsp+F0h] [rbp-F40h] BYREF
  __int64 v58; // [rsp+100h] [rbp-F30h]
  __int64 v59; // [rsp+110h] [rbp-F20h] BYREF
  __int64 v60; // [rsp+118h] [rbp-F18h]
  __int64 v61; // [rsp+120h] [rbp-F10h]
  unsigned __int64 v62; // [rsp+128h] [rbp-F08h]
  __int64 v63; // [rsp+130h] [rbp-F00h]
  __int64 v64; // [rsp+138h] [rbp-EF8h]
  __int64 v65; // [rsp+140h] [rbp-EF0h]
  const __m128i *v66; // [rsp+148h] [rbp-EE8h] BYREF
  __m128i *v67; // [rsp+150h] [rbp-EE0h]
  const __m128i *v68; // [rsp+158h] [rbp-ED8h]
  _QWORD *v69; // [rsp+160h] [rbp-ED0h]
  __int64 v70; // [rsp+168h] [rbp-EC8h] BYREF
  int v71; // [rsp+170h] [rbp-EC0h]
  __int64 v72; // [rsp+178h] [rbp-EB8h]
  _BYTE *v73; // [rsp+180h] [rbp-EB0h]
  __int64 v74; // [rsp+188h] [rbp-EA8h]
  _BYTE v75[1536]; // [rsp+190h] [rbp-EA0h] BYREF
  _BYTE *v76; // [rsp+790h] [rbp-8A0h]
  __int64 v77; // [rsp+798h] [rbp-898h]
  _BYTE v78[512]; // [rsp+7A0h] [rbp-890h] BYREF
  _BYTE *v79; // [rsp+9A0h] [rbp-690h]
  __int64 v80; // [rsp+9A8h] [rbp-688h]
  _BYTE v81[1536]; // [rsp+9B0h] [rbp-680h] BYREF
  _BYTE *v82; // [rsp+FB0h] [rbp-80h]
  __int64 v83; // [rsp+FB8h] [rbp-78h]
  _BYTE v84[112]; // [rsp+FC0h] [rbp-70h] BYREF

  v18 = a1[4];
  v51 = 0;
  v47 = a13;
  v52 = 0;
  v53 = 0;
  v56 = 0u;
  v57 = 0u;
  LODWORD(v58) = 0;
  v19 = sub_1E0A0C0(v18);
  v57.m128i_i64[1] = sub_15A9620(v19, a1[6], 0);
  v57.m128i_i32[0] = a6;
  v56.m128i_i64[1] = a5;
  sub_1D27190(&v51, 0, &v56);
  v43 = _mm_loadu_si128((const __m128i *)&a8);
  v44 = v52;
  v45 = (__m128i *)v53;
  v56.m128i_i64[1] = a8;
  v57.m128i_i32[0] = v43.m128i_i32[2];
  if ( v52 == v53 )
  {
    sub_1D27190(&v51, v52, &v56);
    v46 = _mm_loadu_si128((const __m128i *)&a10);
    v21 = v52;
    v57.m128i_i64[1] = a11;
    v56.m128i_i64[1] = a10;
    v57.m128i_i32[0] = v46.m128i_i32[2];
    if ( v52 != v53 )
    {
      if ( !v52 )
        goto LABEL_4;
      goto LABEL_3;
    }
  }
  else
  {
    if ( v52 )
    {
      *v52 = _mm_loadu_si128(&v56);
      v44[1] = _mm_loadu_si128(&v57);
      v44[2].m128i_i64[0] = v58;
      v44 = v52;
      v45 = (__m128i *)v53;
    }
    v20 = _mm_loadu_si128((const __m128i *)&a10);
    v21 = (__m128i *)((char *)v44 + 40);
    v52 = v21;
    v56.m128i_i64[1] = a10;
    v57.m128i_i64[1] = a11;
    v57.m128i_i32[0] = v20.m128i_i32[2];
    if ( v21 != v45 )
    {
LABEL_3:
      *v21 = _mm_loadu_si128(&v56);
      v21[1] = _mm_loadu_si128(&v57);
      v21[2].m128i_i64[0] = v58;
      v21 = v52;
LABEL_4:
      v52 = (__m128i *)((char *)v21 + 40);
      goto LABEL_5;
    }
  }
  sub_1D27190(&v51, v21, &v56);
LABEL_5:
  v49 = sub_1F40570(a12);
  if ( v49 == 462 )
    sub_16BD130("Unsupported element size", 1u);
  v22 = *(_QWORD *)a4;
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
  v70 = v22;
  if ( v22 )
    sub_1623A60((__int64)&v70, v22, 2);
  v23 = *(_DWORD *)(a4 + 8);
  v24 = a1[4];
  v59 = a2;
  v71 = v23;
  LODWORD(v60) = a3;
  v25 = sub_1E0A0C0(v24);
  v26 = 8 * sub_15A9520(v25, 0);
  if ( v26 == 32 )
  {
    v27 = 5;
  }
  else if ( v26 > 0x20 )
  {
    v27 = 6;
    if ( v26 != 64 )
    {
      v27 = 0;
      if ( v26 == 128 )
        v27 = 7;
    }
  }
  else
  {
    v27 = 3;
    if ( v26 != 8 )
      v27 = 4 * (v26 == 16);
  }
  v28 = v49;
  v29 = sub_1D27640((__int64)a1, *(char **)(a1[2] + 8LL * v49 + 74096), v27, 0);
  v31 = v30;
  v50 = v29;
  v32 = sub_1643270((_QWORD *)a1[6]);
  v33 = a1[2];
  v61 = v32;
  v34 = v66;
  v35 = *(_DWORD *)(v33 + 4 * v28 + 79648);
  v64 = v50;
  LODWORD(v65) = v31;
  LODWORD(v63) = v35;
  v66 = v51;
  LODWORD(v32) = -858993459 * (((char *)v52 - (char *)v51) >> 3);
  v67 = v52;
  v36 = v68;
  v51 = 0;
  v52 = 0;
  HIDWORD(v62) = v32;
  v37 = v53;
  v53 = 0;
  v68 = v37;
  if ( v34 )
    j_j___libc_free_0(v34, (char *)v36 - (char *)v34);
  v38 = (void (***)())v69[2];
  v39 = **v38;
  if ( v39 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v39)(v38, v69[4], v35, &v66);
  v40 = a1[2];
  LOBYTE(v62) = v62 & 0xDF;
  BYTE1(v62) = v47;
  sub_2056920(v54, v40, &v59);
  v41 = v55;
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
  return v41;
}
