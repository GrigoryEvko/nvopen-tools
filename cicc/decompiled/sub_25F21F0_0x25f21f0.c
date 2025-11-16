// Function: sub_25F21F0
// Address: 0x25f21f0
//
void __fastcall sub_25F21F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  unsigned __int64 *v13; // r13
  __int64 v14; // r8
  unsigned __int64 *v15; // r15
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  _BYTE *v22[2]; // [rsp+0h] [rbp-3E0h] BYREF
  __int64 v23; // [rsp+10h] [rbp-3D0h] BYREF
  __int64 *v24; // [rsp+20h] [rbp-3C0h]
  __int64 v25; // [rsp+28h] [rbp-3B8h]
  __int64 v26; // [rsp+30h] [rbp-3B0h] BYREF
  __m128i v27; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 *v28; // [rsp+50h] [rbp-390h] BYREF
  int v29; // [rsp+58h] [rbp-388h]
  char v30; // [rsp+5Ch] [rbp-384h]
  __int64 v31; // [rsp+60h] [rbp-380h] BYREF
  __m128i v32; // [rsp+68h] [rbp-378h] BYREF
  __int64 v33; // [rsp+78h] [rbp-368h]
  __m128i v34; // [rsp+80h] [rbp-360h] BYREF
  __m128i v35; // [rsp+90h] [rbp-350h]
  unsigned __int64 *v36; // [rsp+A0h] [rbp-340h] BYREF
  __int64 v37; // [rsp+A8h] [rbp-338h]
  _BYTE v38[324]; // [rsp+B0h] [rbp-330h] BYREF
  int v39; // [rsp+1F4h] [rbp-1ECh]
  __int64 v40; // [rsp+1F8h] [rbp-1E8h]
  void *v41; // [rsp+200h] [rbp-1E0h] BYREF
  int v42; // [rsp+208h] [rbp-1D8h]
  char v43; // [rsp+20Ch] [rbp-1D4h]
  __int64 v44; // [rsp+210h] [rbp-1D0h]
  __m128i v45; // [rsp+218h] [rbp-1C8h] BYREF
  __int64 v46; // [rsp+228h] [rbp-1B8h]
  __m128i v47; // [rsp+230h] [rbp-1B0h] BYREF
  __m128i v48; // [rsp+240h] [rbp-1A0h] BYREF
  unsigned __int64 *v49; // [rsp+250h] [rbp-190h] BYREF
  unsigned int v50; // [rsp+258h] [rbp-188h]
  _BYTE v51[324]; // [rsp+260h] [rbp-180h] BYREF
  int v52; // [rsp+3A4h] [rbp-3Ch]
  __int64 v53; // [rsp+3A8h] [rbp-38h]

  v3 = *a1;
  v4 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v4)
    || (v20 = sub_B2BE50(v3),
        v21 = sub_B6F970(v20),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 48LL))(v21)) )
  {
    v5 = *(_QWORD *)(a2 + 56);
    if ( v5 )
      v5 -= 24;
    sub_B176B0((__int64)&v41, (__int64)"hotcoldsplit", (__int64)"ExtractFailed", 13, v5);
    sub_B18290((__int64)&v41, "Failed to extract region at block ", 0x22u);
    sub_B16080((__int64)v22, "Block", 5, (unsigned __int8 *)a2);
    v28 = &v31;
    sub_25F0260((__int64 *)&v28, v22[0], (__int64)&v22[0][(unsigned __int64)v22[1]]);
    v32.m128i_i64[1] = (__int64)&v34;
    sub_25F0260(&v32.m128i_i64[1], v24, (__int64)v24 + v25);
    v35 = _mm_loadu_si128(&v27);
    sub_B180C0((__int64)&v41, (unsigned __int64)&v28);
    if ( (__m128i *)v32.m128i_i64[1] != &v34 )
      j_j___libc_free_0(v32.m128i_u64[1]);
    if ( v28 != &v31 )
      j_j___libc_free_0((unsigned __int64)v28);
    v10 = _mm_loadu_si128(&v45);
    v11 = _mm_loadu_si128(&v47);
    v12 = _mm_loadu_si128(&v48);
    v36 = (unsigned __int64 *)v38;
    v29 = v42;
    v32 = v10;
    v30 = v43;
    v34 = v11;
    v31 = v44;
    v35 = v12;
    v28 = (__int64 *)&unk_49D9D40;
    v33 = v46;
    v37 = 0x400000000LL;
    if ( v50 )
      sub_25F1AA0((__int64)&v36, (__int64)&v49, v6, v7, v8, v9);
    v38[320] = v51[320];
    v39 = v52;
    v40 = v53;
    v28 = (__int64 *)&unk_49D9DB0;
    if ( v24 != &v26 )
      j_j___libc_free_0((unsigned __int64)v24);
    if ( (__int64 *)v22[0] != &v23 )
      j_j___libc_free_0((unsigned __int64)v22[0]);
    v13 = v49;
    v41 = &unk_49D9D40;
    v14 = 10LL * v50;
    v15 = &v49[v14];
    if ( v49 != &v49[v14] )
    {
      do
      {
        v15 -= 10;
        v16 = v15[4];
        if ( (unsigned __int64 *)v16 != v15 + 6 )
          j_j___libc_free_0(v16);
        if ( (unsigned __int64 *)*v15 != v15 + 2 )
          j_j___libc_free_0(*v15);
      }
      while ( v13 != v15 );
      v15 = v49;
    }
    if ( v15 != (unsigned __int64 *)v51 )
      _libc_free((unsigned __int64)v15);
    sub_1049740(a1, (__int64)&v28);
    v17 = v36;
    v28 = (__int64 *)&unk_49D9D40;
    v18 = &v36[10 * (unsigned int)v37];
    if ( v36 != v18 )
    {
      do
      {
        v18 -= 10;
        v19 = v18[4];
        if ( (unsigned __int64 *)v19 != v18 + 6 )
          j_j___libc_free_0(v19);
        if ( (unsigned __int64 *)*v18 != v18 + 2 )
          j_j___libc_free_0(*v18);
      }
      while ( v17 != v18 );
      v18 = v36;
    }
    if ( v18 != (unsigned __int64 *)v38 )
      _libc_free((unsigned __int64)v18);
  }
}
