// Function: sub_30CE2E0
// Address: 0x30ce2e0
//
void __fastcall sub_30CE2E0(__int64 a1, const char **a2)
{
  char *v2; // r13
  __int64 v3; // rdx
  unsigned __int64 *v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rsi
  unsigned __int64 *v7; // rax
  _OWORD *v8; // rsi
  unsigned __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  size_t v12; // r8
  char *v13; // r12
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i v17; // xmm0
  __m128i v18; // xmm2
  __int64 v19; // rdx
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // [rsp+8h] [rbp-4B8h]
  __int64 v29; // [rsp+10h] [rbp-4B0h]
  __int64 v31; // [rsp+20h] [rbp-4A0h]
  __int64 v32; // [rsp+20h] [rbp-4A0h]
  __int64 v33; // [rsp+20h] [rbp-4A0h]
  __m128i v34; // [rsp+30h] [rbp-490h] BYREF
  unsigned __int64 v35[2]; // [rsp+40h] [rbp-480h] BYREF
  __int64 v36; // [rsp+50h] [rbp-470h] BYREF
  __int64 *v37; // [rsp+60h] [rbp-460h]
  __int64 v38; // [rsp+70h] [rbp-450h] BYREF
  _QWORD *v39; // [rsp+90h] [rbp-430h] BYREF
  __int64 v40; // [rsp+98h] [rbp-428h]
  _QWORD v41[2]; // [rsp+A0h] [rbp-420h] BYREF
  __int64 *v42; // [rsp+B0h] [rbp-410h]
  __int64 v43; // [rsp+C0h] [rbp-400h] BYREF
  __m128i *v44; // [rsp+E0h] [rbp-3E0h] BYREF
  size_t v45; // [rsp+E8h] [rbp-3D8h]
  __m128i v46; // [rsp+F0h] [rbp-3D0h] BYREF
  __int64 *v47; // [rsp+100h] [rbp-3C0h]
  __int64 v48; // [rsp+110h] [rbp-3B0h] BYREF
  char *v49; // [rsp+130h] [rbp-390h] BYREF
  size_t v50; // [rsp+138h] [rbp-388h]
  unsigned __int64 v51; // [rsp+140h] [rbp-380h] BYREF
  __m128i v52; // [rsp+148h] [rbp-378h]
  __int64 v53; // [rsp+158h] [rbp-368h]
  __m128i v54; // [rsp+160h] [rbp-360h]
  __m128i v55; // [rsp+170h] [rbp-350h]
  unsigned __int64 *v56; // [rsp+180h] [rbp-340h] BYREF
  __int64 v57; // [rsp+188h] [rbp-338h]
  _BYTE v58[324]; // [rsp+190h] [rbp-330h] BYREF
  int v59; // [rsp+2D4h] [rbp-1ECh]
  __int64 v60; // [rsp+2D8h] [rbp-1E8h]
  _OWORD *v61; // [rsp+2E0h] [rbp-1E0h] BYREF
  size_t v62; // [rsp+2E8h] [rbp-1D8h]
  _OWORD v63[4]; // [rsp+2F0h] [rbp-1D0h] BYREF
  unsigned __int64 *v64; // [rsp+330h] [rbp-190h]
  unsigned int v65; // [rsp+338h] [rbp-188h]
  char v66; // [rsp+340h] [rbp-180h] BYREF

  sub_30CAD10((__int64 *)&v49, a1 + 72);
  v2 = (char *)*a2;
  v3 = -1;
  v39 = v41;
  if ( v2 )
    v3 = (__int64)&v2[strlen(v2)];
  sub_30CA380((__int64 *)&v39, v2, v3);
  if ( v40 == 0x3FFFFFFFFFFFFFFFLL || v40 == 4611686018427387902LL )
    sub_4262D8((__int64)"basic_string::append");
  v4 = sub_2241490((unsigned __int64 *)&v39, "; ", 2u);
  v44 = &v46;
  if ( (unsigned __int64 *)*v4 == v4 + 2 )
  {
    v46 = _mm_loadu_si128((const __m128i *)v4 + 1);
  }
  else
  {
    v44 = (__m128i *)*v4;
    v46.m128i_i64[0] = v4[2];
  }
  v45 = v4[1];
  *v4 = (unsigned __int64)(v4 + 2);
  v4[1] = 0;
  *((_BYTE *)v4 + 16) = 0;
  v5 = 15;
  v6 = 15;
  if ( v44 != &v46 )
    v6 = v46.m128i_i64[0];
  if ( v45 + v50 <= v6 )
    goto LABEL_12;
  if ( v49 != (char *)&v51 )
    v5 = v51;
  if ( v45 + v50 <= v5 )
  {
    v7 = sub_2241130((unsigned __int64 *)&v49, 0, 0, v44, v45);
    v61 = v63;
    v8 = (_OWORD *)*v7;
    v9 = v7 + 2;
    if ( (unsigned __int64 *)*v7 != v7 + 2 )
      goto LABEL_13;
  }
  else
  {
LABEL_12:
    v7 = sub_2241490((unsigned __int64 *)&v44, v49, v50);
    v61 = v63;
    v8 = (_OWORD *)*v7;
    v9 = v7 + 2;
    if ( (unsigned __int64 *)*v7 != v7 + 2 )
    {
LABEL_13:
      v61 = v8;
      *(_QWORD *)&v63[0] = v7[2];
      goto LABEL_14;
    }
  }
  v63[0] = _mm_loadu_si128((const __m128i *)v7 + 1);
LABEL_14:
  v62 = v7[1];
  *v7 = (unsigned __int64)v9;
  v7[1] = 0;
  *((_BYTE *)v7 + 16) = 0;
  sub_30CB170(*(_QWORD *)(a1 + 64), v61, v62);
  if ( v61 != v63 )
    j_j___libc_free_0((unsigned __int64)v61);
  if ( v44 != &v46 )
    j_j___libc_free_0((unsigned __int64)v44);
  if ( v39 != v41 )
    j_j___libc_free_0((unsigned __int64)v39);
  if ( v49 != (char *)&v51 )
    j_j___libc_free_0((unsigned __int64)v49);
  v28 = *(__int64 **)(a1 + 48);
  v31 = *v28;
  v10 = sub_B2BE50(*v28);
  if ( sub_B6EA50(v10)
    || (v26 = sub_B2BE50(v31),
        v27 = sub_B6F970(v26),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v27 + 48LL))(v27)) )
  {
    v29 = *(_QWORD *)(a1 + 40);
    sub_B157E0((__int64)&v34, (_QWORD *)(a1 + 32));
    sub_B17640((__int64)&v61, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL), (__int64)"NotInlined", 10, &v34, v29);
    sub_B18290((__int64)&v61, "'", 1u);
    sub_B16080((__int64)&v44, "Callee", 6, *(unsigned __int8 **)(a1 + 24));
    v32 = sub_2445430((__int64)&v61, (__int64)&v44);
    sub_B18290(v32, "' is not inlined into '", 0x17u);
    sub_B16080((__int64)&v39, "Caller", 6, *(unsigned __int8 **)(a1 + 16));
    v11 = sub_2445430(v32, (__int64)&v39);
    sub_B18290(v11, "': ", 3u);
    v12 = 0;
    v13 = (char *)*a2;
    if ( *a2 )
      v12 = strlen(*a2);
    sub_B16430((__int64)v35, "Reason", 6u, v13, v12);
    v14 = sub_2445430(v11, (__int64)v35);
    LODWORD(v50) = *(_DWORD *)(v14 + 8);
    BYTE4(v50) = *(_BYTE *)(v14 + 12);
    v51 = *(_QWORD *)(v14 + 16);
    v17 = _mm_loadu_si128((const __m128i *)(v14 + 24));
    v49 = (char *)&unk_49D9D40;
    v52 = v17;
    v53 = *(_QWORD *)(v14 + 40);
    v54 = _mm_loadu_si128((const __m128i *)(v14 + 48));
    v18 = _mm_loadu_si128((const __m128i *)(v14 + 64));
    v56 = (unsigned __int64 *)v58;
    v57 = 0x400000000LL;
    v55 = v18;
    v19 = *(unsigned int *)(v14 + 88);
    if ( (_DWORD)v19 )
    {
      v33 = v14;
      sub_30CDBD0((__int64)&v56, v14 + 80, v19, 0x400000000LL, v15, v16);
      v14 = v33;
    }
    v58[320] = *(_BYTE *)(v14 + 416);
    v59 = *(_DWORD *)(v14 + 420);
    v60 = *(_QWORD *)(v14 + 424);
    v49 = (char *)&unk_49D9DB0;
    if ( v37 != &v38 )
      j_j___libc_free_0((unsigned __int64)v37);
    if ( (__int64 *)v35[0] != &v36 )
      j_j___libc_free_0(v35[0]);
    if ( v42 != &v43 )
      j_j___libc_free_0((unsigned __int64)v42);
    if ( v39 != v41 )
      j_j___libc_free_0((unsigned __int64)v39);
    if ( v47 != &v48 )
      j_j___libc_free_0((unsigned __int64)v47);
    if ( v44 != &v46 )
      j_j___libc_free_0((unsigned __int64)v44);
    v20 = v64;
    v61 = &unk_49D9D40;
    v21 = &v64[10 * v65];
    if ( v64 != v21 )
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
      while ( v20 != v21 );
      v21 = v64;
    }
    if ( v21 != (unsigned __int64 *)&v66 )
      _libc_free((unsigned __int64)v21);
    sub_1049740(v28, (__int64)&v49);
    v23 = v56;
    v49 = (char *)&unk_49D9D40;
    v24 = &v56[10 * (unsigned int)v57];
    if ( v56 != v24 )
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
      v24 = v56;
    }
    if ( v24 != (unsigned __int64 *)v58 )
      _libc_free((unsigned __int64)v24);
  }
}
