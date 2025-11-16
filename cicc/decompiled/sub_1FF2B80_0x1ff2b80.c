// Function: sub_1FF2B80
// Address: 0x1ff2b80
//
__int64 __fastcall sub_1FF2B80(__int64 a1, int a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // r15
  unsigned __int8 v5; // bl
  __int64 v6; // rax
  unsigned int *v7; // r8
  unsigned int *v8; // r15
  __int64 (__fastcall *v10)(__int64, __int64, __int64, unsigned int); // r10
  unsigned __int8 v11; // al
  __m128i *v12; // rsi
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int32 v18; // edx
  __int64 (__fastcall *v19)(__int64, __int64, __int64, unsigned int); // r10
  unsigned __int8 v20; // al
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int8 v24; // al
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 *v31; // rax
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 (__fastcall *v36)(__int64, __int64, __int64, unsigned int); // rax
  __int64 v37; // rsi
  __m128i *v38; // rdx
  const __m128i *v39; // rdi
  unsigned int v40; // r8d
  __int64 v41; // rax
  const __m128i *v42; // rsi
  const __m128i *v43; // rax
  void (***v44)(); // rdi
  void (*v45)(); // rax
  __int64 v46; // r14
  int v48; // [rsp+8h] [rbp-1018h]
  __int64 v49; // [rsp+10h] [rbp-1010h]
  __int64 v50; // [rsp+28h] [rbp-FF8h]
  char v51; // [rsp+3Fh] [rbp-FE1h]
  __int64 v52; // [rsp+40h] [rbp-FE0h]
  int v54; // [rsp+48h] [rbp-FD8h]
  unsigned int v55; // [rsp+4Ch] [rbp-FD4h]
  unsigned int *v57; // [rsp+58h] [rbp-FC8h]
  __int64 v58; // [rsp+58h] [rbp-FC8h]
  unsigned int v59; // [rsp+58h] [rbp-FC8h]
  unsigned int v60; // [rsp+70h] [rbp-FB0h] BYREF
  __int64 v61; // [rsp+78h] [rbp-FA8h]
  __int64 v62; // [rsp+80h] [rbp-FA0h] BYREF
  int v63; // [rsp+88h] [rbp-F98h]
  const __m128i *v64; // [rsp+90h] [rbp-F90h] BYREF
  __m128i *v65; // [rsp+98h] [rbp-F88h]
  const __m128i *v66; // [rsp+A0h] [rbp-F80h]
  __int64 v67; // [rsp+B0h] [rbp-F70h] BYREF
  __int64 v68; // [rsp+B8h] [rbp-F68h]
  __int64 v69; // [rsp+C0h] [rbp-F60h]
  __m128i v70; // [rsp+D0h] [rbp-F50h] BYREF
  __m128i v71; // [rsp+E0h] [rbp-F40h] BYREF
  __int64 v72; // [rsp+F0h] [rbp-F30h]
  __int64 v73; // [rsp+100h] [rbp-F20h] BYREF
  __int64 v74; // [rsp+108h] [rbp-F18h]
  __int64 v75; // [rsp+110h] [rbp-F10h]
  unsigned __int64 v76; // [rsp+118h] [rbp-F08h]
  __int64 v77; // [rsp+120h] [rbp-F00h]
  __int64 v78; // [rsp+128h] [rbp-EF8h]
  __int64 v79; // [rsp+130h] [rbp-EF0h]
  const __m128i *v80; // [rsp+138h] [rbp-EE8h] BYREF
  __m128i *v81; // [rsp+140h] [rbp-EE0h]
  const __m128i *v82; // [rsp+148h] [rbp-ED8h]
  __int64 v83; // [rsp+150h] [rbp-ED0h]
  __int64 v84; // [rsp+158h] [rbp-EC8h] BYREF
  int v85; // [rsp+160h] [rbp-EC0h]
  __int64 v86; // [rsp+168h] [rbp-EB8h]
  _BYTE *v87; // [rsp+170h] [rbp-EB0h]
  __int64 v88; // [rsp+178h] [rbp-EA8h]
  _BYTE v89[1536]; // [rsp+180h] [rbp-EA0h] BYREF
  _BYTE *v90; // [rsp+780h] [rbp-8A0h]
  __int64 v91; // [rsp+788h] [rbp-898h]
  _BYTE v92[512]; // [rsp+790h] [rbp-890h] BYREF
  _BYTE *v93; // [rsp+990h] [rbp-690h]
  __int64 v94; // [rsp+998h] [rbp-688h]
  _BYTE v95[1536]; // [rsp+9A0h] [rbp-680h] BYREF
  _BYTE *v96; // [rsp+FA0h] [rbp-80h]
  __int64 v97; // [rsp+FA8h] [rbp-78h]
  _BYTE v98[112]; // [rsp+FB0h] [rbp-70h] BYREF

  v4 = a1;
  v5 = a4;
  v6 = *(unsigned int *)(a3 + 56);
  v7 = *(unsigned int **)(a3 + 32);
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v70 = 0u;
  v71 = 0u;
  LODWORD(v72) = 0;
  v57 = &v7[10 * v6];
  v55 = a4;
  if ( v57 != v7 )
  {
    v8 = v7;
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)v8 + 40LL) + 16LL * v8[2];
        v14 = *(_BYTE *)v13;
        v74 = *(_QWORD *)(v13 + 8);
        v15 = *(_QWORD *)(a1 + 16);
        LOBYTE(v73) = v14;
        v16 = sub_1F58E60((__int64)&v73, *(_QWORD **)(v15 + 48));
        v17 = *(_QWORD *)(a1 + 8);
        v70.m128i_i64[1] = *(_QWORD *)v8;
        v18 = v8[2];
        v71.m128i_i64[1] = v16;
        v71.m128i_i32[0] = v18;
        v19 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v17 + 656LL);
        v20 = v5;
        if ( v19 != sub_1F3CB00 )
        {
          v20 = v19(v17, (unsigned int)v73, v74, v55);
          v17 = *(_QWORD *)(a1 + 8);
        }
        LOBYTE(v72) = v20 & 1 | v72 & 0xFE;
        v10 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v17 + 656LL);
        v11 = v5;
        if ( v10 != sub_1F3CB00 )
          v11 = v10(v17, (unsigned int)v73, v74, v55);
        v12 = v65;
        LOBYTE(v72) = v72 & 0xFD | (2 * ((v11 ^ 1) & 1));
        if ( v65 != v66 )
          break;
        v8 += 10;
        sub_1D27190(&v64, v65, &v70);
        if ( v57 == v8 )
          goto LABEL_12;
      }
      if ( v65 )
      {
        *v65 = _mm_loadu_si128(&v70);
        v12[1] = _mm_loadu_si128(&v71);
        v12[2].m128i_i64[0] = v72;
        v12 = v65;
      }
      v8 += 10;
      v65 = (__m128i *)((char *)v12 + 40);
    }
    while ( v57 != v8 );
LABEL_12:
    v4 = a1;
  }
  v21 = *(_QWORD *)(v4 + 16);
  v22 = sub_1E0A0C0(*(_QWORD *)(v21 + 32));
  v23 = 8 * sub_15A9520(v22, 0);
  if ( v23 == 32 )
  {
    v24 = 5;
  }
  else if ( v23 > 0x20 )
  {
    v24 = 6;
    if ( v23 != 64 )
    {
      v24 = 0;
      if ( v23 == 128 )
        v24 = 7;
    }
  }
  else
  {
    v24 = 3;
    if ( v23 != 8 )
      v24 = 4 * (v23 == 16);
  }
  v50 = a2;
  v49 = sub_1D27640(v21, *(char **)(*(_QWORD *)(v4 + 8) + 8LL * a2 + 74096), v24, 0);
  v48 = v25;
  v26 = *(_QWORD *)(a3 + 40);
  LOBYTE(v25) = *(_BYTE *)v26;
  v61 = *(_QWORD *)(v26 + 8);
  v27 = *(_QWORD *)(v4 + 16);
  LOBYTE(v60) = v25;
  v28 = sub_1F58E60((__int64)&v60, *(_QWORD **)(v27 + 48));
  v29 = *(_QWORD *)(v4 + 16);
  v30 = *(_QWORD *)(v4 + 8);
  v63 = 0;
  v58 = v28;
  v31 = *(__int64 **)(v29 + 32);
  v62 = v29 + 88;
  v52 = v29 + 88;
  v32 = *v31;
  v54 = 0;
  v51 = sub_20A1B30(v30, v29, a3, &v62);
  if ( v51 )
  {
    v33 = **(_QWORD **)(*(_QWORD *)(v32 + 24) + 16LL);
    if ( v58 != v33 && *(_BYTE *)(v33 + 8) )
    {
      v51 = 0;
    }
    else
    {
      v52 = v62;
      v54 = v63;
    }
  }
  v34 = *(_QWORD *)(v4 + 16);
  v35 = *(_QWORD *)(v4 + 8);
  v76 = 0xFFFFFFFF00000020LL;
  v83 = v34;
  v87 = v89;
  v88 = 0x2000000000LL;
  v90 = v92;
  v91 = 0x2000000000LL;
  v94 = 0x2000000000LL;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v93 = v95;
  v96 = v98;
  v97 = 0x400000000LL;
  v36 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v35 + 656LL);
  if ( v36 != sub_1F3CB00 )
    v5 = v36(v35, v60, v61, v55);
  v37 = *(_QWORD *)(a3 + 72);
  v67 = v37;
  if ( v37 )
    sub_1623A60((__int64)&v67, v37, 2);
  LODWORD(v68) = *(_DWORD *)(a3 + 64);
  if ( v84 )
    sub_161E7C0((__int64)&v84, v84);
  v84 = v67;
  if ( v67 )
    sub_1623A60((__int64)&v84, v67, 2);
  v38 = v65;
  v39 = v80;
  v65 = 0;
  v85 = v68;
  v73 = v52;
  LODWORD(v74) = v54;
  v40 = *(_DWORD *)(*(_QWORD *)(v4 + 8) + 4 * v50 + 79648);
  v81 = v38;
  v75 = v58;
  v78 = v49;
  LODWORD(v77) = v40;
  LODWORD(v79) = v48;
  v41 = (char *)v38 - (char *)v64;
  v80 = v64;
  v64 = 0;
  v42 = v82;
  HIDWORD(v76) = -858993459 * (v41 >> 3);
  v43 = v66;
  v66 = 0;
  v82 = v43;
  if ( v39 )
  {
    v59 = v40;
    j_j___libc_free_0(v39, (char *)v42 - (char *)v39);
    v40 = v59;
  }
  v44 = *(void (****)())(v83 + 16);
  v45 = **v44;
  if ( v45 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v45)(v44, *(_QWORD *)(v83 + 32), v40, &v80);
  BYTE2(v76) = 1;
  BYTE1(v76) = v51;
  LOBYTE(v76) = v5 & 1 | v76 & 0xFC | (2 * ((v5 ^ 1) & 1));
  if ( v67 )
    sub_161E7C0((__int64)&v67, v67);
  sub_2056920(&v67, *(_QWORD *)(v4 + 8), &v73);
  if ( v69 )
    v46 = v67;
  else
    v46 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 176LL);
  if ( v96 != v98 )
    _libc_free((unsigned __int64)v96);
  if ( v93 != v95 )
    _libc_free((unsigned __int64)v93);
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
  if ( v87 != v89 )
    _libc_free((unsigned __int64)v87);
  if ( v84 )
    sub_161E7C0((__int64)&v84, v84);
  if ( v80 )
    j_j___libc_free_0(v80, (char *)v82 - (char *)v80);
  if ( v64 )
    j_j___libc_free_0(v64, (char *)v66 - (char *)v64);
  return v46;
}
