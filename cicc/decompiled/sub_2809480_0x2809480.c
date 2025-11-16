// Function: sub_2809480
// Address: 0x2809480
//
__int64 __fastcall sub_2809480(__int64 *a1, __int64 a2, __int64 a3, __int8 *a4, size_t a5)
{
  __int64 *v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  unsigned __int64 *v20; // r14
  unsigned __int64 *v21; // r15
  unsigned __int64 *v22; // r14
  unsigned __int64 v23; // rdi
  __int64 *v24; // r15
  __int64 v25; // r14
  char *v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r9
  unsigned __int64 *v30; // r15
  __int64 v31; // r8
  unsigned __int64 *v32; // r14
  unsigned __int64 v33; // rdi
  __int64 v35; // rsi
  __int64 v36; // rax
  __m128i v37; // xmm3
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // r12
  unsigned __int64 v40; // rdi
  __int64 v41; // r8
  unsigned __int64 *v42; // rbx
  unsigned __int64 *v43; // r14
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // [rsp+10h] [rbp-420h]
  __int64 *v48; // [rsp+18h] [rbp-418h]
  char v54; // [rsp+5Fh] [rbp-3D1h]
  __int64 v55; // [rsp+68h] [rbp-3C8h] BYREF
  __m128i v56; // [rsp+70h] [rbp-3C0h] BYREF
  __m128i v57; // [rsp+80h] [rbp-3B0h] BYREF
  __int64 v58; // [rsp+90h] [rbp-3A0h] BYREF
  __m128i v59; // [rsp+A0h] [rbp-390h] BYREF
  __int64 v60; // [rsp+B0h] [rbp-380h]
  __m128i v61; // [rsp+B8h] [rbp-378h]
  __int64 v62; // [rsp+C8h] [rbp-368h]
  __m128i v63; // [rsp+D0h] [rbp-360h]
  __m128i v64; // [rsp+E0h] [rbp-350h]
  unsigned __int64 *v65; // [rsp+F0h] [rbp-340h] BYREF
  __int64 v66; // [rsp+F8h] [rbp-338h]
  _BYTE v67[320]; // [rsp+100h] [rbp-330h] BYREF
  char v68; // [rsp+240h] [rbp-1F0h]
  int v69; // [rsp+244h] [rbp-1ECh]
  __int64 v70; // [rsp+248h] [rbp-1E8h]
  void *v71; // [rsp+250h] [rbp-1E0h] BYREF
  __int64 v72; // [rsp+258h] [rbp-1D8h]
  __int64 v73; // [rsp+260h] [rbp-1D0h]
  __m128i v74; // [rsp+268h] [rbp-1C8h] BYREF
  __int64 v75; // [rsp+278h] [rbp-1B8h]
  __m128i v76; // [rsp+280h] [rbp-1B0h] BYREF
  __m128i v77; // [rsp+290h] [rbp-1A0h] BYREF
  unsigned __int64 *v78; // [rsp+2A0h] [rbp-190h] BYREF
  __int64 v79; // [rsp+2A8h] [rbp-188h]
  _BYTE v80[320]; // [rsp+2B0h] [rbp-180h] BYREF
  char v81; // [rsp+3F0h] [rbp-40h]
  int v82; // [rsp+3F4h] [rbp-3Ch]
  __int64 v83; // [rsp+3F8h] [rbp-38h]

  v5 = a1;
  v47 = sub_B2BE50(a1[1]);
  v54 = *((_BYTE *)a1 + 65);
  if ( v54 )
    v54 = *((_BYTE *)a1 + 64);
  v6 = *(_QWORD *)a1[7];
  v48 = (__int64 *)a1[7];
  v7 = sub_B2BE50(v6);
  if ( sub_B6EA50(v7)
    || (v45 = sub_B2BE50(v6),
        v46 = sub_B6F970(v45),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v46 + 48LL))(v46)) )
  {
    v12 = **(_QWORD **)(*a1 + 32);
    sub_D4BD20(&v56, *a1, v8, v9, v10 * 8, v11);
    sub_B157E0((__int64)&v57, &v56);
    sub_B17640((__int64)&v71, (__int64)"loop-distribute", (__int64)"NotDistributed", 14, &v57, v12);
    sub_B18290((__int64)&v71, "loop not distributed: use -Rpass-analysis=loop-distribute for more info", 0x47u);
    v17 = _mm_loadu_si128(&v74);
    v18 = _mm_loadu_si128(&v76);
    v19 = _mm_loadu_si128(&v77);
    v59.m128i_i32[2] = v72;
    v61 = v17;
    v59.m128i_i8[12] = BYTE4(v72);
    v63 = v18;
    v60 = v73;
    v64 = v19;
    v59.m128i_i64[0] = (__int64)&unk_49D9D40;
    v62 = v75;
    v65 = (unsigned __int64 *)v67;
    v66 = 0x400000000LL;
    if ( (_DWORD)v79 )
    {
      sub_2809200((__int64)&v65, (__int64)&v78, v13, v14, v15, v16);
      v71 = &unk_49D9D40;
      v68 = v81;
      v69 = v82;
      v70 = v83;
      v59.m128i_i64[0] = (__int64)&unk_49D9DB0;
      v41 = 10LL * (unsigned int)v79;
      v20 = &v78[v41];
      if ( v78 != &v78[v41] )
      {
        v42 = &v78[v41];
        v43 = v78;
        do
        {
          v42 -= 10;
          v44 = v42[4];
          if ( (unsigned __int64 *)v44 != v42 + 6 )
            j_j___libc_free_0(v44);
          if ( (unsigned __int64 *)*v42 != v42 + 2 )
            j_j___libc_free_0(*v42);
        }
        while ( v43 != v42 );
        v5 = a1;
        v20 = v78;
      }
    }
    else
    {
      v20 = v78;
      v68 = v81;
      v69 = v82;
      v70 = v83;
      v59.m128i_i64[0] = (__int64)&unk_49D9DB0;
    }
    if ( v20 != (unsigned __int64 *)v80 )
      _libc_free((unsigned __int64)v20);
    if ( v56.m128i_i64[0] )
      sub_B91220((__int64)&v56, v56.m128i_i64[0]);
    sub_1049740(v48, (__int64)&v59);
    v21 = v65;
    v59.m128i_i64[0] = (__int64)&unk_49D9D40;
    v10 = 10LL * (unsigned int)v66;
    v22 = &v65[v10];
    if ( v65 != &v65[v10] )
    {
      do
      {
        v22 -= 10;
        v23 = v22[4];
        if ( (unsigned __int64 *)v23 != v22 + 6 )
          j_j___libc_free_0(v23);
        if ( (unsigned __int64 *)*v22 != v22 + 2 )
          j_j___libc_free_0(*v22);
      }
      while ( v21 != v22 );
      v22 = v65;
    }
    if ( v22 != (unsigned __int64 *)v67 )
      _libc_free((unsigned __int64)v22);
  }
  v24 = (__int64 *)v5[7];
  v25 = **(_QWORD **)(*v5 + 32);
  sub_D4BD20(&v57, *v5, v8, v9, v10 * 8, v11);
  sub_B157E0((__int64)&v59, &v57);
  v26 = "loop-distribute";
  if ( v54 )
    v26 = (char *)off_4B91160;
  sub_B17850((__int64)&v71, (__int64)v26, a2, a3, &v59, v25);
  sub_B18290((__int64)&v71, "loop not distributed: ", 0x16u);
  sub_B18290((__int64)&v71, a4, a5);
  sub_1049740(v24, (__int64)&v71);
  v30 = v78;
  v71 = &unk_49D9D40;
  v31 = 10LL * (unsigned int)v79;
  v32 = &v78[v31];
  if ( v78 != &v78[v31] )
  {
    do
    {
      v32 -= 10;
      v33 = v32[4];
      if ( (unsigned __int64 *)v33 != v32 + 6 )
        j_j___libc_free_0(v33);
      if ( (unsigned __int64 *)*v32 != v32 + 2 )
        j_j___libc_free_0(*v32);
    }
    while ( v30 != v32 );
    v32 = v78;
  }
  if ( v32 != (unsigned __int64 *)v80 )
    _libc_free((unsigned __int64)v32);
  if ( v57.m128i_i64[0] )
    sub_B91220((__int64)&v57, v57.m128i_i64[0]);
  if ( v54 )
  {
    v35 = *v5;
    v59.m128i_i64[0] = (__int64)"loop not distributed: failed explicitly specified loop distribution";
    v61.m128i_i16[4] = 259;
    sub_D4BD20(&v55, v35, v27, v28, v31 * 8, v29);
    sub_B157E0((__int64)&v56, &v55);
    v36 = v5[1];
    v81 = 0;
    v37 = _mm_loadu_si128(&v56);
    v75 = 0;
    v73 = v36;
    v76.m128i_i64[0] = (__int64)byte_3F871B3;
    v72 = 0x100000012LL;
    v78 = (unsigned __int64 *)v80;
    v79 = 0x400000000LL;
    v74 = v37;
    v77.m128i_i8[8] = 0;
    v71 = &unk_49D9D08;
    v76.m128i_i64[1] = 0;
    v82 = -1;
    v83 = 0;
    sub_CA0F50(v57.m128i_i64, (void **)&v59);
    sub_B18290((__int64)&v71, (__int8 *)v57.m128i_i64[0], v57.m128i_u64[1]);
    if ( (__int64 *)v57.m128i_i64[0] != &v58 )
      j_j___libc_free_0(v57.m128i_u64[0]);
    v71 = &unk_49D9E50;
    sub_B6EB20(v47, (__int64)&v71);
    v38 = v78;
    v71 = &unk_49D9D40;
    v39 = &v78[10 * (unsigned int)v79];
    if ( v78 != v39 )
    {
      do
      {
        v39 -= 10;
        v40 = v39[4];
        if ( (unsigned __int64 *)v40 != v39 + 6 )
          j_j___libc_free_0(v40);
        if ( (unsigned __int64 *)*v39 != v39 + 2 )
          j_j___libc_free_0(*v39);
      }
      while ( v38 != v39 );
      v39 = v78;
    }
    if ( v39 != (unsigned __int64 *)v80 )
      _libc_free((unsigned __int64)v39);
    if ( v55 )
      sub_B91220((__int64)&v55, v55);
  }
  return 0;
}
