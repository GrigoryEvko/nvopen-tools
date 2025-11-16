// Function: sub_1FF2520
// Address: 0x1ff2520
//
__int64 __fastcall sub_1FF2520(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, char a6, __int64 a7)
{
  __m128i *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned int v17; // edx
  unsigned __int8 v18; // al
  __int64 v19; // r12
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  signed __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // r14d
  signed __int64 v28; // rdx
  const __m128i *v29; // rdx
  void (***v30)(); // rdi
  void (*v31)(); // rax
  __int64 v32; // rsi
  __int64 v33; // r14
  __int64 v35; // [rsp+0h] [rbp-FE0h]
  int v36; // [rsp+8h] [rbp-FD8h]
  __int64 v39; // [rsp+28h] [rbp-FB8h]
  _QWORD v40[2]; // [rsp+40h] [rbp-FA0h] BYREF
  const __m128i *v41; // [rsp+50h] [rbp-F90h] BYREF
  __m128i *v42; // [rsp+58h] [rbp-F88h]
  const __m128i *v43; // [rsp+60h] [rbp-F80h]
  __int64 v44; // [rsp+70h] [rbp-F70h] BYREF
  __m128i v45; // [rsp+90h] [rbp-F50h] BYREF
  __m128i v46; // [rsp+A0h] [rbp-F40h] BYREF
  __int64 v47; // [rsp+B0h] [rbp-F30h]
  __int64 v48; // [rsp+C0h] [rbp-F20h] BYREF
  __int64 v49; // [rsp+C8h] [rbp-F18h]
  __int64 v50; // [rsp+D0h] [rbp-F10h]
  unsigned __int64 v51; // [rsp+D8h] [rbp-F08h]
  __int64 v52; // [rsp+E0h] [rbp-F00h]
  __int64 v53; // [rsp+E8h] [rbp-EF8h]
  __int64 v54; // [rsp+F0h] [rbp-EF0h]
  const __m128i *v55; // [rsp+F8h] [rbp-EE8h] BYREF
  __m128i *v56; // [rsp+100h] [rbp-EE0h]
  const __m128i *v57; // [rsp+108h] [rbp-ED8h]
  __int64 v58; // [rsp+110h] [rbp-ED0h]
  __int64 v59; // [rsp+118h] [rbp-EC8h] BYREF
  int v60; // [rsp+120h] [rbp-EC0h]
  __int64 v61; // [rsp+128h] [rbp-EB8h]
  _BYTE *v62; // [rsp+130h] [rbp-EB0h]
  __int64 v63; // [rsp+138h] [rbp-EA8h]
  _BYTE v64[1536]; // [rsp+140h] [rbp-EA0h] BYREF
  _BYTE *v65; // [rsp+740h] [rbp-8A0h]
  __int64 v66; // [rsp+748h] [rbp-898h]
  _BYTE v67[512]; // [rsp+750h] [rbp-890h] BYREF
  _BYTE *v68; // [rsp+950h] [rbp-690h]
  __int64 v69; // [rsp+958h] [rbp-688h]
  _BYTE v70[1536]; // [rsp+960h] [rbp-680h] BYREF
  _BYTE *v71; // [rsp+F60h] [rbp-80h]
  __int64 v72; // [rsp+F68h] [rbp-78h]
  _BYTE v73[112]; // [rsp+F70h] [rbp-70h] BYREF

  v40[1] = a4;
  v40[0] = a3;
  v41 = (const __m128i *)sub_22077B0(160);
  v42 = (__m128i *)v41;
  v43 = v41 + 10;
  v45.m128i_i64[0] = 0;
  v39 = a5 + 64;
  v46.m128i_i64[1] = 0;
  LODWORD(v47) = 0;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)a5;
      v46.m128i_i32[0] = *(_DWORD *)(a5 + 8);
      v12 = *(_QWORD *)(v11 + 40) + 16LL * v46.m128i_u32[0];
      v45.m128i_i64[1] = v11;
      LOBYTE(v11) = *(_BYTE *)v12;
      v13 = *(_QWORD *)(v12 + 8);
      LOBYTE(v48) = v11;
      v49 = v13;
      v14 = sub_1F58E60((__int64)&v48, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 48LL));
      v10 = v42;
      v46.m128i_i64[1] = v14;
      LOBYTE(v47) = (2 * ((a6 ^ 1) & 1)) | a6 & 1 | v47 & 0xFC;
      if ( v42 != v43 )
        break;
      a5 += 16;
      sub_1D27190(&v41, v42, &v45);
      if ( v39 == a5 )
        goto LABEL_7;
    }
    if ( v42 )
    {
      *v42 = _mm_loadu_si128(&v45);
      v10[1] = _mm_loadu_si128(&v46);
      v10[2].m128i_i64[0] = v47;
      v10 = v42;
    }
    a5 += 16;
    v42 = (__m128i *)((char *)v10 + 40);
  }
  while ( v39 != a5 );
LABEL_7:
  v15 = *(_QWORD *)(a1 + 16);
  v16 = sub_1E0A0C0(*(_QWORD *)(v15 + 32));
  v17 = 8 * sub_15A9520(v16, 0);
  if ( v17 == 32 )
  {
    v18 = 5;
  }
  else if ( v17 > 0x20 )
  {
    v18 = 6;
    if ( v17 != 64 )
    {
      v18 = 0;
      if ( v17 == 128 )
        v18 = 7;
    }
  }
  else
  {
    v18 = 3;
    if ( v17 != 8 )
      v18 = 4 * (v17 == 16);
  }
  v19 = sub_1D27640(v15, *(char **)(*(_QWORD *)(a1 + 8) + 8LL * a2 + 74096), v18, 0);
  v36 = v20;
  v21 = sub_1F58E60((__int64)v40, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 48LL));
  v22 = *(_QWORD *)(a1 + 16);
  v71 = v73;
  v23 = 0;
  v24 = v21;
  v48 = 0;
  v51 = 0xFFFFFFFF00000020LL;
  v62 = v64;
  v63 = 0x2000000000LL;
  v66 = 0x2000000000LL;
  v69 = 0x2000000000LL;
  v72 = 0x400000000LL;
  v25 = *(_QWORD *)a7;
  v65 = v67;
  v49 = 0;
  v50 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = v22;
  v60 = 0;
  v61 = 0;
  v68 = v70;
  v59 = v25;
  if ( v25 )
  {
    v35 = v24;
    sub_1623A60((__int64)&v59, v25, 2);
    v25 = (__int64)v55;
    v22 = *(_QWORD *)(a1 + 16);
    v24 = v35;
    v23 = (char *)v57 - (char *)v55;
  }
  LODWORD(v49) = 0;
  v48 = v22 + 88;
  v26 = *(_QWORD *)(a1 + 8);
  v60 = *(_DWORD *)(a7 + 8);
  v27 = *(_DWORD *)(v26 + 4LL * a2 + 79648);
  v50 = v24;
  v53 = v19;
  LODWORD(v52) = v27;
  LODWORD(v54) = v36;
  v55 = v41;
  v28 = ((char *)v42 - (char *)v41) >> 3;
  v56 = v42;
  v41 = 0;
  v42 = 0;
  HIDWORD(v51) = -858993459 * v28;
  v29 = v43;
  v43 = 0;
  v57 = v29;
  if ( v25 )
    j_j___libc_free_0(v25, v23);
  v30 = *(void (****)())(v58 + 16);
  v31 = **v30;
  if ( v31 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v31)(v30, *(_QWORD *)(v58 + 32), v27, &v55);
  v32 = *(_QWORD *)(a1 + 8);
  BYTE2(v51) = 1;
  LOBYTE(v51) = (2 * ((a6 ^ 1) & 1)) | a6 & 1 | v51 & 0xFC;
  sub_2056920(&v44, v32, &v48);
  v33 = v44;
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( v59 )
    sub_161E7C0((__int64)&v59, v59);
  if ( v55 )
    j_j___libc_free_0(v55, (char *)v57 - (char *)v55);
  if ( v41 )
    j_j___libc_free_0(v41, (char *)v43 - (char *)v41);
  return v33;
}
