// Function: sub_20BDF40
// Address: 0x20bdf40
//
__int64 __fastcall sub_20BDF40(__m128i *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rax
  unsigned int v9; // edx
  unsigned __int8 v10; // al
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rax
  __m128i *v19; // rsi
  __int32 v20; // edx
  __int64 v21; // rax
  signed __int64 v22; // rsi
  __int64 v23; // r12
  int v24; // edx
  int v25; // r13d
  __int64 v26; // rax
  int v27; // edx
  void (***v28)(); // rdi
  void (*v29)(); // rax
  __int64 v30; // r12
  __int64 v33; // [rsp+38h] [rbp-FE8h]
  __int64 v34; // [rsp+60h] [rbp-FC0h] BYREF
  int v35; // [rsp+68h] [rbp-FB8h]
  const __m128i *v36; // [rsp+70h] [rbp-FB0h] BYREF
  __m128i *v37; // [rsp+78h] [rbp-FA8h]
  const __m128i *v38; // [rsp+80h] [rbp-FA0h]
  __int64 v39[2]; // [rsp+90h] [rbp-F90h] BYREF
  __int64 v40; // [rsp+A0h] [rbp-F80h] BYREF
  _QWORD v41[4]; // [rsp+B0h] [rbp-F70h] BYREF
  __m128i v42; // [rsp+D0h] [rbp-F50h] BYREF
  __m128i v43; // [rsp+E0h] [rbp-F40h] BYREF
  __int64 v44; // [rsp+F0h] [rbp-F30h]
  __m128i v45; // [rsp+100h] [rbp-F20h] BYREF
  __int64 v46; // [rsp+110h] [rbp-F10h]
  unsigned __int64 v47; // [rsp+118h] [rbp-F08h]
  __int64 v48; // [rsp+120h] [rbp-F00h]
  __int64 v49; // [rsp+128h] [rbp-EF8h]
  __int64 v50; // [rsp+130h] [rbp-EF0h]
  const __m128i *v51; // [rsp+138h] [rbp-EE8h] BYREF
  __m128i *v52; // [rsp+140h] [rbp-EE0h]
  const __m128i *v53; // [rsp+148h] [rbp-ED8h]
  __int64 v54; // [rsp+150h] [rbp-ED0h]
  __int64 v55; // [rsp+158h] [rbp-EC8h] BYREF
  int v56; // [rsp+160h] [rbp-EC0h]
  __int64 v57; // [rsp+168h] [rbp-EB8h]
  _BYTE *v58; // [rsp+170h] [rbp-EB0h]
  __int64 v59; // [rsp+178h] [rbp-EA8h]
  _BYTE v60[1536]; // [rsp+180h] [rbp-EA0h] BYREF
  _BYTE *v61; // [rsp+780h] [rbp-8A0h]
  __int64 v62; // [rsp+788h] [rbp-898h]
  _BYTE v63[512]; // [rsp+790h] [rbp-890h] BYREF
  _BYTE *v64; // [rsp+990h] [rbp-690h]
  __int64 v65; // [rsp+998h] [rbp-688h]
  _BYTE v66[1536]; // [rsp+9A0h] [rbp-680h] BYREF
  _BYTE *v67; // [rsp+FA0h] [rbp-80h]
  __int64 v68; // [rsp+FA8h] [rbp-78h]
  _BYTE v69[112]; // [rsp+FB0h] [rbp-70h] BYREF

  v8 = sub_1E0A0C0(*(_QWORD *)(a3 + 32));
  v9 = 8 * sub_15A9520(v8, 0);
  if ( v9 == 32 )
  {
    v10 = 5;
  }
  else if ( v9 > 0x20 )
  {
    v10 = 6;
    if ( v9 != 64 )
    {
      v10 = 0;
      if ( v9 == 128 )
        v10 = 7;
    }
  }
  else
  {
    v10 = 3;
    if ( v9 != 8 )
      v10 = 4 * (v9 == 16);
  }
  v11 = v10;
  v12 = sub_16471D0(*(_QWORD **)(a3 + 48), 0);
  v13 = *(_QWORD *)(a2 + 72);
  v33 = v12;
  v34 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v34, v13, 2);
  v14 = *(_DWORD *)(a2 + 64);
  v15 = *(_QWORD *)(a2 + 88);
  v36 = 0;
  v37 = 0;
  v35 = v14;
  v38 = 0;
  v42 = 0u;
  v43 = 0u;
  LODWORD(v44) = 0;
  v41[0] = sub_1649960(v15);
  v45.m128i_i64[0] = (__int64)"__emutls_v.";
  v41[1] = v16;
  LOWORD(v46) = 1283;
  v45.m128i_i64[1] = (__int64)v41;
  sub_16E2FC0(v39, (__int64)&v45);
  v17 = sub_16321C0(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 40LL), v39[0], v39[1], 1);
  v18 = sub_1D29600((_QWORD *)a3, v17, (__int64)&v34, v11, 0, 0, 0, 0);
  v19 = v37;
  v42.m128i_i64[1] = (__int64)v18;
  v43.m128i_i32[0] = v20;
  v43.m128i_i64[1] = v33;
  if ( v37 == v38 )
  {
    sub_1D27190(&v36, v37, &v42);
  }
  else
  {
    if ( v37 )
    {
      a4 = _mm_loadu_si128(&v42);
      *v37 = a4;
      a5 = _mm_loadu_si128(&v43);
      v19[1] = a5;
      v19[2].m128i_i64[0] = v44;
      v19 = v37;
    }
    v37 = (__m128i *)((char *)v19 + 40);
  }
  v21 = sub_1D27640(a3, "__emutls_get_address", v11, 0);
  v54 = a3;
  v22 = 0;
  v23 = v21;
  v25 = v24;
  v47 = 0xFFFFFFFF00000020LL;
  v58 = v60;
  v59 = 0x2000000000LL;
  v62 = 0x2000000000LL;
  v65 = 0x2000000000LL;
  v67 = v69;
  v68 = 0x400000000LL;
  v26 = v34;
  v61 = v63;
  v45 = 0u;
  v46 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v56 = 0;
  v57 = 0;
  v64 = v66;
  v55 = v34;
  if ( v34 )
  {
    sub_1623A60((__int64)&v55, v34, 2);
    v26 = (__int64)v51;
    v22 = (char *)v53 - (char *)v51;
  }
  v56 = v35;
  v45.m128i_i64[0] = a3 + 88;
  v46 = v33;
  LODWORD(v50) = v25;
  v45.m128i_i32[2] = 0;
  v49 = v23;
  v51 = v36;
  v27 = -858993459 * (((char *)v37 - (char *)v36) >> 3);
  LODWORD(v48) = 0;
  v52 = v37;
  v36 = 0;
  HIDWORD(v47) = v27;
  v37 = 0;
  v53 = v38;
  v38 = 0;
  if ( v26 )
    j_j___libc_free_0(v26, v22);
  v28 = *(void (****)())(v54 + 16);
  v29 = **v28;
  if ( v29 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v29)(v28, *(_QWORD *)(v54 + 32), 0, &v51);
  sub_2056920((__int64)v41, a1, &v45, a4, a5, a6);
  *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 32) + 56LL) + 64LL) = 257;
  v30 = v41[0];
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( v55 )
    sub_161E7C0((__int64)&v55, v55);
  if ( v51 )
    j_j___libc_free_0(v51, (char *)v53 - (char *)v51);
  if ( (__int64 *)v39[0] != &v40 )
    j_j___libc_free_0(v39[0], v40 + 1);
  if ( v36 )
    j_j___libc_free_0(v36, (char *)v38 - (char *)v36);
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  return v30;
}
