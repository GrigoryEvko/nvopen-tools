// Function: sub_25F1D20
// Address: 0x25f1d20
//
void __fastcall sub_25F1D20(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int8 **a8,
        unsigned __int8 **a9)
{
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  unsigned __int64 *v19; // r14
  __int64 v20; // r8
  unsigned __int64 *v21; // r15
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  _BYTE *v29[2]; // [rsp+10h] [rbp-430h] BYREF
  __int64 v30; // [rsp+20h] [rbp-420h] BYREF
  __int64 *v31; // [rsp+30h] [rbp-410h]
  __int64 v32; // [rsp+38h] [rbp-408h]
  __int64 v33; // [rsp+40h] [rbp-400h] BYREF
  __m128i v34; // [rsp+50h] [rbp-3F0h] BYREF
  _BYTE *v35[2]; // [rsp+60h] [rbp-3E0h] BYREF
  __int64 v36; // [rsp+70h] [rbp-3D0h] BYREF
  __int64 *v37; // [rsp+80h] [rbp-3C0h]
  __int64 v38; // [rsp+88h] [rbp-3B8h]
  __int64 v39; // [rsp+90h] [rbp-3B0h] BYREF
  __m128i v40; // [rsp+A0h] [rbp-3A0h] BYREF
  __int64 *v41; // [rsp+B0h] [rbp-390h] BYREF
  int v42; // [rsp+B8h] [rbp-388h]
  char v43; // [rsp+BCh] [rbp-384h]
  __int64 v44; // [rsp+C0h] [rbp-380h] BYREF
  __m128i v45; // [rsp+C8h] [rbp-378h] BYREF
  __int64 v46; // [rsp+D8h] [rbp-368h]
  __m128i v47; // [rsp+E0h] [rbp-360h] BYREF
  __m128i v48; // [rsp+F0h] [rbp-350h]
  unsigned __int64 *v49; // [rsp+100h] [rbp-340h] BYREF
  __int64 v50; // [rsp+108h] [rbp-338h]
  _BYTE v51[324]; // [rsp+110h] [rbp-330h] BYREF
  int v52; // [rsp+254h] [rbp-1ECh]
  __int64 v53; // [rsp+258h] [rbp-1E8h]
  void *v54; // [rsp+260h] [rbp-1E0h] BYREF
  int v55; // [rsp+268h] [rbp-1D8h]
  char v56; // [rsp+26Ch] [rbp-1D4h]
  __int64 v57; // [rsp+270h] [rbp-1D0h]
  __m128i v58; // [rsp+278h] [rbp-1C8h] BYREF
  __int64 v59; // [rsp+288h] [rbp-1B8h]
  __m128i v60; // [rsp+290h] [rbp-1B0h] BYREF
  __m128i v61; // [rsp+2A0h] [rbp-1A0h] BYREF
  unsigned __int64 *v62; // [rsp+2B0h] [rbp-190h] BYREF
  unsigned int v63; // [rsp+2B8h] [rbp-188h]
  _BYTE v64[324]; // [rsp+2C0h] [rbp-180h] BYREF
  int v65; // [rsp+404h] [rbp-3Ch]
  __int64 v66; // [rsp+408h] [rbp-38h]

  v9 = *a1;
  v10 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v10)
    || (v26 = sub_B2BE50(v9),
        v27 = sub_B6F970(v26),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v27 + 48LL))(v27)) )
  {
    v11 = *(_QWORD *)(a7 + 56);
    if ( v11 )
      v11 -= 24;
    sub_B174A0((__int64)&v54, (__int64)"hotcoldsplit", (__int64)"HotColdSplit", 12, v11);
    sub_B16080((__int64)v35, "Original", 8, *a8);
    v41 = &v44;
    sub_25F0260((__int64 *)&v41, v35[0], (__int64)&v35[0][(unsigned __int64)v35[1]]);
    v45.m128i_i64[1] = (__int64)&v47;
    sub_25F0260(&v45.m128i_i64[1], v37, (__int64)v37 + v38);
    v48 = _mm_loadu_si128(&v40);
    sub_B180C0((__int64)&v54, (unsigned __int64)&v41);
    if ( (__m128i *)v45.m128i_i64[1] != &v47 )
      j_j___libc_free_0(v45.m128i_u64[1]);
    if ( v41 != &v44 )
      j_j___libc_free_0((unsigned __int64)v41);
    sub_B18290((__int64)&v54, " split cold code into ", 0x16u);
    sub_B16080((__int64)v29, "Split", 5, *a9);
    v41 = &v44;
    sub_25F0260((__int64 *)&v41, v29[0], (__int64)&v29[0][(unsigned __int64)v29[1]]);
    v45.m128i_i64[1] = (__int64)&v47;
    sub_25F0260(&v45.m128i_i64[1], v31, (__int64)v31 + v32);
    v48 = _mm_loadu_si128(&v34);
    sub_B180C0((__int64)&v54, (unsigned __int64)&v41);
    if ( (__m128i *)v45.m128i_i64[1] != &v47 )
      j_j___libc_free_0(v45.m128i_u64[1]);
    if ( v41 != &v44 )
      j_j___libc_free_0((unsigned __int64)v41);
    v16 = _mm_loadu_si128(&v58);
    v17 = _mm_loadu_si128(&v60);
    v49 = (unsigned __int64 *)v51;
    v42 = v55;
    v18 = _mm_loadu_si128(&v61);
    v45 = v16;
    v43 = v56;
    v47 = v17;
    v44 = v57;
    v41 = (__int64 *)&unk_49D9D40;
    v48 = v18;
    v46 = v59;
    v50 = 0x400000000LL;
    if ( v63 )
      sub_25F1AA0((__int64)&v49, (__int64)&v62, v12, v13, v14, v15);
    v51[320] = v64[320];
    v52 = v65;
    v53 = v66;
    v41 = (__int64 *)&unk_49D9D78;
    if ( v31 != &v33 )
      j_j___libc_free_0((unsigned __int64)v31);
    if ( (__int64 *)v29[0] != &v30 )
      j_j___libc_free_0((unsigned __int64)v29[0]);
    if ( v37 != &v39 )
      j_j___libc_free_0((unsigned __int64)v37);
    if ( (__int64 *)v35[0] != &v36 )
      j_j___libc_free_0((unsigned __int64)v35[0]);
    v19 = v62;
    v54 = &unk_49D9D40;
    v20 = 10LL * v63;
    v21 = &v62[v20];
    if ( v62 != &v62[v20] )
    {
      do
      {
        v21 -= 10;
        v22 = v21[4];
        if ( (unsigned __int64 *)v22 != v21 + 6 )
          j_j___libc_free_0(v22);
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21);
      }
      while ( v19 != v21 );
      v21 = v62;
    }
    if ( v21 != (unsigned __int64 *)v64 )
      _libc_free((unsigned __int64)v21);
    sub_1049740(a1, (__int64)&v41);
    v23 = v49;
    v41 = (__int64 *)&unk_49D9D40;
    v24 = &v49[10 * (unsigned int)v50];
    if ( v49 != v24 )
    {
      do
      {
        v24 -= 10;
        v25 = v24[4];
        if ( (unsigned __int64 *)v25 != v24 + 6 )
          j_j___libc_free_0(v25);
        if ( (unsigned __int64 *)*v24 != v24 + 2 )
          j_j___libc_free_0(*v24);
      }
      while ( v23 != v24 );
      v24 = v49;
    }
    if ( v24 != (unsigned __int64 *)v51 )
      _libc_free((unsigned __int64)v24);
  }
}
