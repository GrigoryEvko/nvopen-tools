// Function: sub_20BE530
// Address: 0x20be530
//
__int64 __fastcall sub_20BE530(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __int64 a10,
        unsigned __int64 a11,
        unsigned __int8 a12,
        __int64 a13,
        char a14,
        char a15)
{
  unsigned int v15; // r13d
  unsigned int v16; // r14d
  __int64 v17; // r8
  _DWORD *v19; // r15
  unsigned int v20; // eax
  unsigned int v21; // r13d
  unsigned int v22; // r14d
  unsigned __int8 v23; // dl
  __m128i *v24; // rsi
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  __int8 v29; // dl
  unsigned __int8 v30; // si
  __int64 v31; // rcx
  __int64 (__fastcall *v32)(__int64, __int64, __int64, unsigned int); // rax
  __int64 v33; // rax
  unsigned int v34; // edx
  unsigned __int8 v35; // al
  __int64 v36; // rbx
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 (__fastcall *v40)(__int64, __int64, __int64, unsigned int); // rax
  __int64 v41; // rsi
  __m128i *v42; // rdx
  const __m128i *v43; // rcx
  int v44; // eax
  unsigned __int32 v45; // r13d
  const __m128i *v46; // rdi
  const __m128i *v47; // rsi
  const __m128i *v48; // rax
  void (***v49)(); // rdi
  void (*v50)(); // rax
  const __m128i *v52; // rcx
  int v54; // [rsp+10h] [rbp-FD0h]
  __int64 v56; // [rsp+20h] [rbp-FC0h]
  _DWORD *v57; // [rsp+30h] [rbp-FB0h]
  __int64 v59; // [rsp+40h] [rbp-FA0h]
  __int64 v60; // [rsp+40h] [rbp-FA0h]
  unsigned __int8 v61; // [rsp+4Fh] [rbp-F91h]
  __int64 v62; // [rsp+60h] [rbp-F80h] BYREF
  __int64 v63; // [rsp+68h] [rbp-F78h]
  const __m128i *v64; // [rsp+70h] [rbp-F70h] BYREF
  __m128i *v65; // [rsp+78h] [rbp-F68h]
  const __m128i *v66; // [rsp+80h] [rbp-F60h]
  __m128i v67; // [rsp+90h] [rbp-F50h] BYREF
  __m128i v68; // [rsp+A0h] [rbp-F40h] BYREF
  __int64 v69; // [rsp+B0h] [rbp-F30h]
  __m128i v70; // [rsp+C0h] [rbp-F20h] BYREF
  __int64 v71; // [rsp+D0h] [rbp-F10h]
  unsigned __int64 v72; // [rsp+D8h] [rbp-F08h]
  __int64 v73; // [rsp+E0h] [rbp-F00h]
  __int64 v74; // [rsp+E8h] [rbp-EF8h]
  __int64 v75; // [rsp+F0h] [rbp-EF0h]
  const __m128i *v76; // [rsp+F8h] [rbp-EE8h] BYREF
  __m128i *v77; // [rsp+100h] [rbp-EE0h]
  const __m128i *v78; // [rsp+108h] [rbp-ED8h]
  __int64 v79; // [rsp+110h] [rbp-ED0h]
  __int64 v80; // [rsp+118h] [rbp-EC8h] BYREF
  int v81; // [rsp+120h] [rbp-EC0h]
  __int64 v82; // [rsp+128h] [rbp-EB8h]
  _BYTE *v83; // [rsp+130h] [rbp-EB0h]
  __int64 v84; // [rsp+138h] [rbp-EA8h]
  _BYTE v85[1536]; // [rsp+140h] [rbp-EA0h] BYREF
  _BYTE *v86; // [rsp+740h] [rbp-8A0h]
  __int64 v87; // [rsp+748h] [rbp-898h]
  _BYTE v88[512]; // [rsp+750h] [rbp-890h] BYREF
  _BYTE *v89; // [rsp+950h] [rbp-690h]
  __int64 v90; // [rsp+958h] [rbp-688h]
  _BYTE v91[1536]; // [rsp+960h] [rbp-680h] BYREF
  _BYTE *v92; // [rsp+F60h] [rbp-80h]
  __int64 v93; // [rsp+F68h] [rbp-78h]
  _BYTE v94[112]; // [rsp+F70h] [rbp-70h] BYREF

  v62 = a5;
  v17 = a10;
  v61 = a12;
  v63 = a6;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  if ( a11 > 0x333333333333333LL )
    sub_4262D8((__int64)"vector::reserve");
  if ( a11 )
  {
    v52 = (const __m128i *)sub_22077B0(40 * a11);
    v17 = a10;
    v64 = v52;
    v65 = (__m128i *)v52;
    v66 = (const __m128i *)((char *)v52 + 40 * a11);
  }
  v67 = 0u;
  v68 = 0u;
  LODWORD(v69) = 0;
  v57 = (_DWORD *)(v17 + 16 * a11);
  if ( (_DWORD *)v17 != v57 )
  {
    v19 = (_DWORD *)v17;
    v20 = v15;
    v21 = v16;
    v22 = v20;
    do
    {
      while ( 1 )
      {
        v25 = *(_QWORD *)v19;
        v26 = *(_QWORD *)(*(_QWORD *)v19 + 40LL);
        v68.m128i_i32[0] = v19[2];
        v27 = 16LL * v68.m128i_u32[0];
        v67.m128i_i64[1] = v25;
        v28 = v27 + v26;
        v59 = v25;
        v29 = *(_BYTE *)v28;
        v70.m128i_i64[1] = *(_QWORD *)(v28 + 8);
        v70.m128i_i8[0] = v29;
        v30 = a12;
        v68.m128i_i64[1] = sub_1F58E60((__int64)&v70, *(_QWORD **)(a3 + 48));
        v31 = *(_QWORD *)(v59 + 40) + v27;
        v32 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(a2->m128i_i64[0] + 656);
        if ( v32 != sub_1F3CB00 )
        {
          LOBYTE(v22) = *(_BYTE *)v31;
          v30 = v32((__int64)a2, v22, *(_QWORD *)(v31 + 8), a12);
          v32 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(a2->m128i_i64[0] + 656);
          v31 = *(_QWORD *)(v59 + 40) + v27;
        }
        LOBYTE(v69) = v30 & 1 | v69 & 0xFE;
        v23 = a12;
        if ( v32 != sub_1F3CB00 )
        {
          LOBYTE(v21) = *(_BYTE *)v31;
          v23 = v32((__int64)a2, v21, *(_QWORD *)(v31 + 8), a12);
        }
        v24 = v65;
        LOBYTE(v69) = v69 & 0xFD | (2 * ((v23 ^ 1) & 1));
        if ( v65 != v66 )
          break;
        v19 += 4;
        sub_1D27190(&v64, v65, &v67);
        if ( v57 == v19 )
          goto LABEL_15;
      }
      if ( v65 )
      {
        a7 = _mm_loadu_si128(&v67);
        *v65 = a7;
        a8 = _mm_loadu_si128(&v68);
        v24[1] = a8;
        v24[2].m128i_i64[0] = v69;
        v24 = v65;
      }
      v19 += 4;
      v65 = (__m128i *)((char *)v24 + 40);
    }
    while ( v57 != v19 );
  }
LABEL_15:
  if ( a4 == 462 )
    sub_16BD130("Unsupported library call operation!", 1u);
  v33 = sub_1E0A0C0(*(_QWORD *)(a3 + 32));
  v34 = 8 * sub_15A9520(v33, 0);
  if ( v34 == 32 )
  {
    v35 = 5;
  }
  else if ( v34 > 0x20 )
  {
    v35 = 6;
    if ( v34 != 64 )
    {
      v35 = 0;
      if ( v34 == 128 )
        v35 = 7;
    }
  }
  else
  {
    v35 = 3;
    if ( v34 != 8 )
      v35 = 4 * (v34 == 16);
  }
  v36 = a4;
  v56 = sub_1D27640(a3, (char *)a2[4631].m128i_i64[a4], v35, 0);
  v54 = v37;
  v38 = sub_1F58E60((__int64)&v62, *(_QWORD **)(a3 + 48));
  v79 = a3;
  v60 = v38;
  v72 = 0xFFFFFFFF00000020LL;
  v83 = v85;
  v84 = 0x2000000000LL;
  v87 = 0x2000000000LL;
  v90 = 0x2000000000LL;
  v93 = 0x400000000LL;
  v39 = a2->m128i_i64[0];
  v70 = 0u;
  v71 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v86 = v88;
  v89 = v91;
  v92 = v94;
  v40 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(v39 + 656);
  if ( v40 != sub_1F3CB00 )
  {
    v61 = v40((__int64)a2, (unsigned int)v62, v63, a12);
    if ( v80 )
      sub_161E7C0((__int64)&v80, v80);
  }
  v41 = *(_QWORD *)a13;
  v80 = v41;
  if ( v41 )
    sub_1623A60((__int64)&v80, v41, 2);
  v70.m128i_i32[2] = 0;
  v42 = v65;
  v43 = v64;
  v65 = 0;
  v44 = *(_DWORD *)(a13 + 8);
  v64 = 0;
  v70.m128i_i64[0] = a3 + 88;
  v45 = a2[4978].m128i_u32[v36];
  v81 = v44;
  v71 = v60;
  v74 = v56;
  v46 = v76;
  LODWORD(v75) = v54;
  LODWORD(v73) = v45;
  v76 = v43;
  v77 = v42;
  v47 = v78;
  HIDWORD(v72) = -858993459 * (((char *)v42 - (char *)v43) >> 3);
  v48 = v66;
  v66 = 0;
  v78 = v48;
  if ( v46 )
    j_j___libc_free_0(v46, (char *)v47 - (char *)v46);
  v49 = *(void (****)())(v79 + 16);
  v50 = **v49;
  if ( v50 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v50)(v49, *(_QWORD *)(v79 + 32), v45, &v76);
  LOBYTE(v72) = v72 & 0xCC | ((32 * a15) | (16 * a14) | v61 | (2 * (v61 ^ 1))) & 0x33;
  sub_2056920(a1, a2, &v70, a7, a8, a9);
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  if ( v80 )
    sub_161E7C0((__int64)&v80, v80);
  if ( v76 )
    j_j___libc_free_0(v76, (char *)v78 - (char *)v76);
  if ( v64 )
    j_j___libc_free_0(v64, (char *)v66 - (char *)v64);
  return a1;
}
